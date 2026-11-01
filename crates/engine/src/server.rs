//! OpenAI-compatible HTTP server: /v1/chat/completions (streaming SSE, tools, reasoning_content),
//! /v1/completions, /v1/models, /health. One engine, requests are served one at a time; the
//! shared session gives prefix (prompt) caching across requests of the same conversation.

use crate::chat::{parse_output, ChatTemplate, Piece, StreamParser};
use crate::session::{GenStats, Sampler, SamplerCfg, Session};
use crate::tokenizer::Vocab;
use anyhow::{Context, Result};
use serde_json::{json, Value as Json};
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Server {
    pub session: Session,
    pub vocab: Vocab,
    pub tmpl: ChatTemplate,
    pub model_name: String,
    pub default_max_tokens: usize,
    pub sampler: SamplerCfg,
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn id(prefix: &str) -> String {
    let n = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    format!("{prefix}-{n:x}")
}

pub fn serve(addr: &str, srv: Server) -> Result<()> {
    let http = tiny_http::Server::http(addr).map_err(|e| anyhow::anyhow!("binding {addr}: {e}"))?;
    eprintln!("listening on http://{addr}  (POST /v1/chat/completions, /v1/completions; GET /v1/models)");
    let srv = Arc::new(Mutex::new(srv));
    for req in http.incoming_requests() {
        let srv = srv.clone();
        std::thread::spawn(move || {
            if let Err(e) = handle(req, &srv) {
                eprintln!("request error: {e:#}");
            }
        });
    }
    Ok(())
}

fn json_response(req: tiny_http::Request, code: u16, body: &Json) -> Result<()> {
    let r = tiny_http::Response::from_string(body.to_string())
        .with_status_code(code)
        .with_header("Content-Type: application/json".parse::<tiny_http::Header>().unwrap())
        .with_header("Access-Control-Allow-Origin: *".parse::<tiny_http::Header>().unwrap());
    req.respond(r)?;
    Ok(())
}

fn error(req: tiny_http::Request, code: u16, msg: &str) -> Result<()> {
    json_response(req, code, &json!({"error": {"message": msg, "type": "invalid_request_error"}}))
}

fn handle(mut req: tiny_http::Request, srv: &Arc<Mutex<Server>>) -> Result<()> {
    let url = req.url().split('?').next().unwrap_or("").to_string();
    let method = req.method().clone();
    match (&method, url.as_str()) {
        (tiny_http::Method::Options, _) => {
            let r = tiny_http::Response::empty(204)
                .with_header("Access-Control-Allow-Origin: *".parse::<tiny_http::Header>().unwrap())
                .with_header("Access-Control-Allow-Headers: *".parse::<tiny_http::Header>().unwrap())
                .with_header("Access-Control-Allow-Methods: GET, POST, OPTIONS".parse::<tiny_http::Header>().unwrap());
            req.respond(r)?;
            Ok(())
        }
        (tiny_http::Method::Get, "/health") => json_response(req, 200, &json!({"status": "ok"})),
        (tiny_http::Method::Get, "/v1/models") => {
            let name = srv.lock().unwrap().model_name.clone();
            json_response(req, 200, &json!({"object": "list", "data": [{"id": name, "object": "model", "created": now(), "owned_by": "glm-ssd"}]}))
        }
        (tiny_http::Method::Post, "/v1/chat/completions") | (tiny_http::Method::Post, "/v1/completions") => {
            let mut body = String::new();
            req.as_reader().read_to_string(&mut body)?;
            let j: Json = match serde_json::from_str(&body) {
                Ok(j) => j,
                Err(e) => return error(req, 400, &format!("invalid JSON: {e}")),
            };
            let chat = url == "/v1/chat/completions";
            let mut s = srv.lock().unwrap();
            completion(req, &mut s, &j, chat)
        }
        _ => error(req, 404, "not found"),
    }
}

struct Sse {
    w: Box<dyn Write + Send>,
    ok: bool,
}
impl Sse {
    fn send(&mut self, v: &Json) -> bool {
        if self.ok {
            let r = write!(self.w, "data: {v}\n\n").and_then(|_| self.w.flush());
            self.ok = r.is_ok();
        }
        self.ok
    }
}

fn completion(req: tiny_http::Request, s: &mut Server, j: &Json, chat: bool) -> Result<()> {
    let stream = j["stream"].as_bool().unwrap_or(false);
    let include_usage = j["stream_options"]["include_usage"].as_bool().unwrap_or(false);
    let max_tokens = j["max_completion_tokens"].as_u64().or(j["max_tokens"].as_u64()).map(|v| v as usize).unwrap_or(s.default_max_tokens);
    let mut sc = s.sampler.clone();
    if let Some(t) = j["temperature"].as_f64() {
        sc.temperature = t as f32;
    }
    if let Some(t) = j["top_p"].as_f64() {
        sc.top_p = t as f32;
    }
    if let Some(t) = j["top_k"].as_u64() {
        sc.top_k = t as usize;
    }
    if let Some(t) = j["min_p"].as_f64() {
        sc.min_p = t as f32;
    }
    if let Some(t) = j["seed"].as_u64() {
        sc.seed = t;
    }
    let stops: Vec<String> = match &j["stop"] {
        Json::String(x) => vec![x.clone()],
        Json::Array(a) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => vec![],
    };
    let prompt = if chat {
        let Some(msgs) = j.get("messages").filter(|m| m.is_array()) else { return error(req, 400, "messages must be an array") };
        let effort = j["reasoning_effort"].as_str();
        match s.tmpl.render(msgs, j.get("tools"), effort, true) {
            Ok(p) => p,
            Err(e) => return error(req, 400, &format!("chat template: {e:#}")),
        }
    } else {
        match &j["prompt"] {
            Json::String(p) => p.clone(),
            Json::Array(a) if a.len() == 1 && a[0].is_string() => a[0].as_str().unwrap().to_string(),
            _ => return error(req, 400, "prompt must be a string"),
        }
    };
    let toks = s.vocab.encode(&prompt, true);
    let in_think = chat && prompt.trim_end().ends_with("<think>");
    let rid = id(if chat { "chatcmpl" } else { "cmpl" });
    let model = s.model_name.clone();
    let created = now();

    let (mut sse, mut req_opt) = if stream {
        let mut w = req.into_writer();
        write!(
            w,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
        )?;
        w.flush()?;
        let mut sse = Sse { w, ok: true };
        if chat {
            sse.send(&json!({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}));
        }
        (Some(sse), None)
    } else {
        (None, Some(req))
    };

    let mut stats = GenStats::default();
    let t0 = std::time::Instant::now();
    let logits = match s.session.prefill(&toks, &mut stats, |_, _| {}) {
        Ok(l) => l,
        Err(e) => {
            if let Some(r) = req_opt.take() {
                return error(r, 500, &format!("{e:#}"));
            }
            return Err(e);
        }
    };
    let mut sampler = Sampler::new(sc);
    let vocab = &s.vocab;
    let mut parser = StreamParser::new(vocab, in_think);
    let mut n_gen = 0usize;
    let mut hit_eos = false;
    let mut stopped = false;
    let eos = s.session.eng.cfg.eos.clone();
    let mut content_sent = String::new();
    let res = s.session.generate(logits, max_tokens, &mut sampler, &mut stats, |tok| {
        n_gen += 1;
        if eos.contains(&tok) {
            hit_eos = true;
            return false;
        }
        let piece = parser.push(tok);
        if !stops.is_empty() {
            let tail_start = parser.text.len().saturating_sub(64 + stops.iter().map(String::len).max().unwrap_or(0));
            if stops.iter().any(|st| parser.text[tail_start..].contains(st.as_str())) {
                stopped = true;
                return false;
            }
        }
        if let Some(sse) = sse.as_mut() {
            let delta = match piece {
                Piece::Reasoning(t) if !t.is_empty() => Some(json!({"reasoning_content": t})),
                Piece::Content(t) if !t.is_empty() => {
                    content_sent.push_str(&t);
                    Some(json!({"content": t}))
                }
                _ => None,
            };
            if let Some(d) = delta {
                let chunk = if chat {
                    json!({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                        "choices": [{"index": 0, "delta": d, "finish_reason": null}]})
                } else {
                    json!({"id": rid, "object": "text_completion", "created": created, "model": model,
                        "choices": [{"index": 0, "text": d["content"].as_str().or(d["reasoning_content"].as_str()).unwrap_or(""), "finish_reason": null}]})
                };
                if !sse.send(&chunk) {
                    return false; // client went away
                }
            }
        }
        true
    });
    if let Err(e) = res {
        if let Some(r) = req_opt.take() {
            return error(r, 500, &format!("{e:#}"));
        }
        return Err(e);
    }
    let mut text = parser.text.clone();
    for st in &stops {
        if let Some(i) = text.find(st.as_str()) {
            text.truncate(i);
        }
    }
    let (reasoning, content, calls) = if chat { parse_output(&text, in_think) } else { (String::new(), text.clone(), vec![]) };
    let finish = if !calls.is_empty() {
        "tool_calls"
    } else if hit_eos || stopped {
        "stop"
    } else {
        "length"
    };
    let usage = json!({"prompt_tokens": toks.len(), "completion_tokens": n_gen, "total_tokens": toks.len() + n_gen,
        "prompt_tokens_details": {"cached_tokens": stats.reused_tokens}});
    let tool_json: Vec<Json> = calls
        .iter()
        .enumerate()
        .map(|(i, c)| json!({"index": i, "id": format!("call_{}_{i}", &rid[rid.len().saturating_sub(8)..]), "type": "function",
            "function": {"name": c.name, "arguments": c.arguments.to_string()}}))
        .collect();
    eprintln!("[{}] {} | {}", rid, stats.line(), s.session.eng.experts.summary());
    let _ = t0;

    if let Some(mut sse) = sse {
        if chat {
            if !tool_json.is_empty() {
                sse.send(&json!({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{"index": 0, "delta": {"tool_calls": tool_json}, "finish_reason": null}]}));
            }
            sse.send(&json!({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}));
        } else {
            sse.send(&json!({"id": rid, "object": "text_completion", "created": created, "model": model,
                "choices": [{"index": 0, "text": "", "finish_reason": finish}]}));
        }
        if include_usage {
            sse.send(&json!({"id": rid, "object": "chat.completion.chunk", "created": created, "model": model, "choices": [], "usage": usage}));
        }
        if sse.ok {
            let _ = write!(sse.w, "data: [DONE]\n\n").and_then(|_| sse.w.flush());
        }
        return Ok(());
    }
    let req = req_opt.context("request already answered")?;
    let body = if chat {
        let mut msg = json!({"role": "assistant", "content": if content.is_empty() && !tool_json.is_empty() { Json::Null } else { json!(content) }});
        if !reasoning.is_empty() {
            msg["reasoning_content"] = json!(reasoning);
        }
        if !tool_json.is_empty() {
            msg["tool_calls"] = json!(tool_json.iter().map(|t| { let mut t = t.clone(); t.as_object_mut().unwrap().remove("index"); t }).collect::<Vec<_>>());
        }
        json!({"id": rid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})
    } else {
        json!({"id": rid, "object": "text_completion", "created": created, "model": model,
            "choices": [{"index": 0, "text": content, "finish_reason": finish}], "usage": usage})
    };
    json_response(req, 200, &body)
}
