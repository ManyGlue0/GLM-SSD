//! OpenAI-compatible HTTP server: /v1/chat/completions (streaming SSE, tools, reasoning_content),
//! /v1/completions, /v1/models, /health. One engine, requests are served one at a time; the
//! shared session gives prefix (prompt) caching across requests of the same conversation.

use crate::chat::{parse_output, ChatTemplate, Piece, StreamParser};
use crate::holder::ModelHolder;
use crate::session::{GenStats, Sampler, SamplerCfg};
use crate::tokenizer::Vocab;
use anyhow::{Context, Result};
use serde_json::{json, Value as Json};
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Server {
    /// the engine, loaded on demand and unloaded when idle
    pub model: Arc<ModelHolder>,
    pub vocab: Vocab,
    pub tmpl: ChatTemplate,
    pub model_name: String,
    pub default_max_tokens: usize,
    pub sampler: SamplerCfg,
    /// context window in tokens (--ctx): prompt + answer
    pub ctx: usize,
    /// reasoning of recent answers, put back when a client returns them without it (None = off)
    pub replay: Option<Mutex<Replay>>,
}

/// Reasoning of our recent answers, keyed by what the client sees of them (content + tool calls).
///
/// OpenAI-style clients drop `reasoning_content` when they send the conversation back. The chat
/// template would then render that answer as `<think></think>content`, which no longer matches
/// the tokens the engine processed: the prompt cache has to rewind to before the answer and
/// re-read it (an SSD sweep, 10+ s). Putting the reasoning back keeps the conversation
/// token-identical, and is also what the model expects (the template keeps past reasoning).
#[derive(Default)]
pub struct Replay {
    map: HashMap<String, String>,
    order: VecDeque<String>,
}

const REPLAY_MAX: usize = 1024;

impl Replay {
    fn put(&mut self, key: String, reasoning: &str) {
        if key.is_empty() || reasoning.trim().is_empty() {
            return;
        }
        if self.map.insert(key.clone(), reasoning.to_string()).is_none() {
            self.order.push_back(key);
            if self.order.len() > REPLAY_MAX {
                if let Some(k) = self.order.pop_front() {
                    self.map.remove(&k);
                }
            }
        }
    }

    /// Copy of `msgs` with the reasoning of known answers filled in; also the number filled.
    fn fill(&self, msgs: &Json) -> (Json, usize) {
        let mut out = msgs.clone();
        let mut n = 0;
        for m in out.as_array_mut().into_iter().flatten() {
            if m["role"] != "assistant" || m["reasoning_content"].as_str().is_some_and(|r| !r.is_empty()) {
                continue;
            }
            let content = message_text(&m["content"]);
            if content.contains("</think>") {
                continue;
            }
            let calls: Vec<(String, Json)> = m["tool_calls"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|tc| {
                    let f = if tc.get("function").is_some() { &tc["function"] } else { tc };
                    let args = match &f["arguments"] {
                        Json::String(s) => serde_json::from_str(s).unwrap_or_else(|_| Json::String(s.clone())),
                        v => v.clone(),
                    };
                    (f["name"].as_str().unwrap_or("").to_string(), args)
                })
                .collect();
            if let Some(r) = self.map.get(&answer_key(&content, &calls)) {
                m["reasoning_content"] = json!(r);
                n += 1;
            }
        }
        (out, n)
    }
}

/// Text of a message `content`: a string, or the text parts of an array.
fn message_text(c: &Json) -> String {
    match c {
        Json::String(s) => s.clone(),
        Json::Array(parts) => parts.iter().filter_map(|p| p["text"].as_str()).collect(),
        _ => String::new(),
    }
}

fn answer_key(content: &str, calls: &[(String, Json)]) -> String {
    let content = content.trim();
    if content.is_empty() && calls.is_empty() {
        return String::new();
    }
    let calls: Vec<String> = calls.iter().map(|(n, a)| format!("{n}\u{2}{a}")).collect();
    format!("{content}\u{1}{}", calls.join("\u{3}"))
}

impl Server {
    /// The /v1/models entry. The context window is given under the names the common clients read
    /// (OpenRouter/LiteLLM `context_length`, vLLM `max_model_len`, LM Studio `max_context_length`).
    fn model_card(&self) -> Json {
        json!({
            "id": self.model_name, "object": "model", "created": now(), "owned_by": "glm-ssd",
            "context_length": self.ctx, "context_window": self.ctx,
            "max_model_len": self.ctx, "max_context_length": self.ctx,
            "max_output_tokens": if self.default_max_tokens == usize::MAX { self.ctx } else { self.default_max_tokens.min(self.ctx) },
            "reasoning_efforts": crate::chat::EFFORTS, "default_reasoning_effort": "max",
        })
    }
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
    let srv = Arc::new(srv);
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

fn handle(mut req: tiny_http::Request, srv: &Arc<Server>) -> Result<()> {
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
        (tiny_http::Method::Get, "/health") => {
            json_response(req, 200, &json!({"status": "ok", "model_loaded": srv.model.is_loaded(), "context_length": srv.ctx}))
        }
        (tiny_http::Method::Get, "/v1/models") => json_response(req, 200, &json!({"object": "list", "data": [srv.model_card()]})),
        (tiny_http::Method::Get, path) if path.strip_prefix("/v1/models/") == Some(srv.model_name.as_str()) => {
            json_response(req, 200, &srv.model_card())
        }
        // llama.cpp-style properties (read by some clients to learn the context size)
        (tiny_http::Method::Get, "/props") => json_response(
            req,
            200,
            &json!({"default_generation_settings": {"n_ctx": srv.ctx}, "n_ctx": srv.ctx, "model_alias": srv.model_name}),
        ),
        (tiny_http::Method::Post, "/v1/chat/completions") | (tiny_http::Method::Post, "/v1/completions") => {
            let mut body = String::new();
            req.as_reader().read_to_string(&mut body)?;
            let j: Json = match serde_json::from_str(&body) {
                Ok(j) => j,
                Err(e) => return error(req, 400, &format!("invalid JSON: {e}")),
            };
            let chat = url == "/v1/chat/completions";
            completion(req, srv, &j, chat)
        }
        _ => error(req, 404, "not found"),
    }
}

struct Sse {
    w: Box<dyn Write + Send>,
    ok: bool,
}
impl Sse {
    /// One SSE event = one HTTP chunk (chunked transfer encoding marks the end of the response
    /// explicitly, independent of what the HTTP library does with the socket afterwards).
    fn raw(&mut self, payload: &str) -> bool {
        if self.ok {
            let r = write!(self.w, "{:x}\r\n{payload}\r\n", payload.len()).and_then(|_| self.w.flush());
            self.ok = r.is_ok();
        }
        self.ok
    }
    fn send(&mut self, v: &Json) -> bool {
        self.raw(&format!("data: {v}\n\n"))
    }
    fn end(&mut self) {
        if self.ok {
            self.raw("data: [DONE]\n\n");
            let _ = write!(self.w, "0\r\n\r\n").and_then(|_| self.w.flush());
        }
    }
}

fn completion(req: tiny_http::Request, s: &Server, j: &Json, chat: bool) -> Result<()> {
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
        // `reasoning_effort` (Chat Completions) or `reasoning: {effort}` (Responses / OpenRouter style)
        let effort = match j["reasoning_effort"].as_str().or(j["reasoning"]["effort"].as_str()) {
            None => None,
            Some(e) => match crate::chat::normalize_effort(e) {
                Some(e) => Some(e),
                None => {
                    return error(
                        req,
                        400,
                        &format!("unsupported reasoning_effort {e:?}: use low, high or max (GLM-5.3 always reasons; none/minimal map to low, medium to high)"),
                    )
                }
            },
        };
        let (msgs, replayed) = match &s.replay {
            Some(r) => r.lock().unwrap_or_else(|e| e.into_inner()).fill(msgs),
            None => (msgs.clone(), 0),
        };
        if replayed > 0 {
            eprintln!("(reasoning of {replayed} earlier answer(s) put back into the conversation)");
        }
        match s.tmpl.render(&msgs, j.get("tools"), effort, true) {
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
    if toks.len() >= s.ctx {
        let msg = format!(
            "this model's maximum context length is {} tokens, but the prompt has {} tokens (no room left for the answer); restart the server with a larger --ctx",
            s.ctx,
            toks.len()
        );
        let body = json!({"error": {"message": msg, "type": "invalid_request_error", "code": "context_length_exceeded"}});
        return json_response(req, 400, &body);
    }
    // one request at a time; loads the model if it was unloaded for being idle
    let mut st = match s.model.lock() {
        Ok(st) => st,
        Err(e) => return error(req, 500, &format!("loading the model: {e:#}")),
    };
    let sess = st.get();
    let in_think = chat && prompt.trim_end().ends_with("<think>");
    let rid = id(if chat { "chatcmpl" } else { "cmpl" });
    let model = s.model_name.clone();
    let created = now();

    let (mut sse, mut req_opt) = if stream {
        let mut w = req.into_writer();
        write!(
            w,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n"
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

    log_divergence(&s.vocab, sess.history(), &toks);
    let mut stats = GenStats::default();
    let t0 = std::time::Instant::now();
    let logits = match sess.prefill(&toks, &mut stats, |_, _| {}) {
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
    let eos = sess.eng.cfg.eos.clone();
    let mut content_sent = String::new();
    let res = sess.generate(logits, max_tokens, &mut sampler, &mut stats, |tok| {
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
    if let (true, Some(r)) = (chat, &s.replay) {
        let kc: Vec<(String, Json)> = calls.iter().map(|c| (c.name.clone(), c.arguments.clone())).collect();
        let mut r = r.lock().unwrap_or_else(|e| e.into_inner());
        r.put(answer_key(&content, &kc), &reasoning);
        if sse.is_some() && content_sent.trim() != content.trim() {
            r.put(answer_key(&content_sent, &kc), &reasoning); // what a streaming client assembled
        }
    }
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
    eprintln!("[{}] {} | {}", rid, stats.line(), sess.eng.experts.summary());
    if let Err(e) = sess.eng.experts.save_profile() {
        eprintln!("warning: saving the hot-expert profile: {e:#}");
    }
    let _ = t0;
    st.touch();
    drop(st);

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
        sse.end();
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

/// When a request does not continue the cached conversation, say where and how it differs: the
/// tokens before that point are reused, the rest is processed again.
fn log_divergence(vocab: &Vocab, hist: &[u32], toks: &[u32]) {
    let lcp = hist.iter().zip(toks).take_while(|(a, b)| a == b).count();
    if hist.is_empty() || lcp >= hist.len() {
        return; // nothing cached, or an exact continuation
    }
    let show = |t: &[u32]| {
        if t.is_empty() {
            return "(end)".to_string();
        }
        let s = vocab.decode(t);
        let s: String = s.chars().take(160).collect();
        format!("{s:?}")
    };
    let before = &toks[lcp.saturating_sub(12)..lcp];
    eprintln!(
        "(prompt cache: request differs from the cached conversation at token {lcp} of {}; after {}
   cached:  {}
   request: {})",
        hist.len(),
        show(before),
        show(&hist[lcp..(lcp + 24).min(hist.len())]),
        show(&toks[lcp..(lcp + 24).min(toks.len())]),
    );
}
