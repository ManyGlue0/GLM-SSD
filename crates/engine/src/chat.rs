//! Chat formatting: the GGUF's own Jinja chat template (rendered with minijinja) and parsing of
//! the model's output into reasoning / content / tool calls (GLM XML tool-call format).

use crate::tokenizer::Vocab;
use anyhow::{Context, Result};
use minijinja::value::Kwargs;
use minijinja::{Environment, Error, ErrorKind, Value};
use serde_json::{json, Value as Json};

pub struct ChatTemplate {
    env: Environment<'static>,
}

fn tojson(v: Value, kwargs: Kwargs) -> Result<Value, Error> {
    let _: Option<bool> = kwargs.get("ensure_ascii")?;
    let _: Option<Value> = kwargs.get("indent")?;
    kwargs.assert_all_used()?;
    let s = serde_json::to_string(&v).map_err(|e| Error::new(ErrorKind::InvalidOperation, e.to_string()))?;
    Ok(Value::from_safe_string(s))
}

impl ChatTemplate {
    pub fn new(src: &str) -> Result<Self> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", tojson);
        env.set_trim_blocks(false);
        env.set_lstrip_blocks(false);
        env.add_template_owned("chat", src.to_string()).context("parsing the chat template")?;
        Ok(Self { env })
    }

    pub fn from_gguf(g: &gguf::Gguf) -> Result<Self> {
        let src = g.get("tokenizer.chat_template").and_then(|v| v.as_str()).context("GGUF has no tokenizer.chat_template")?;
        Self::new(src)
    }

    /// messages: OpenAI-style [{role, content, reasoning_content?, tool_calls?}], tools: OpenAI tools.
    pub fn render(&self, messages: &Json, tools: Option<&Json>, reasoning_effort: Option<&str>, add_generation_prompt: bool) -> Result<String> {
        let t = self.env.get_template("chat")?;
        let mut ctx = json!({
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
        });
        if let Some(tl) = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
            ctx["tools"] = tl.clone();
        }
        if let Some(r) = reasoning_effort {
            ctx["reasoning_effort"] = json!(r);
        }
        // the template expects tool call arguments as a mapping
        if let Some(msgs) = ctx["messages"].as_array_mut() {
            for m in msgs {
                if let Some(tcs) = m.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
                    for tc in tcs {
                        let f = if tc.get("function").is_some() { &mut tc["function"] } else { tc };
                        if let Some(Json::String(s)) = f.get("arguments") {
                            if let Ok(v) = serde_json::from_str::<Json>(s) {
                                f["arguments"] = v;
                            }
                        }
                    }
                }
            }
        }
        Ok(t.render(Value::from_serialize(&ctx))?)
    }
}

/// One parsed tool call.
#[derive(Clone, Debug)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Json,
}

/// Split a finished assistant output (text after the prompt's `<think>`) into
/// (reasoning, content, tool calls). `started_in_think`: the prompt ended inside `<think>`.
pub fn parse_output(text: &str, started_in_think: bool) -> (String, String, Vec<ToolCall>) {
    let (reasoning, rest) = match text.find("</think>") {
        Some(i) => (text[..i].to_string(), &text[i + "</think>".len()..]),
        None if started_in_think => (text.to_string(), ""),
        None => (String::new(), text),
    };
    let reasoning = reasoning.trim_start_matches("<think>").trim().to_string();
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut s = rest;
    while let Some(i) = s.find("<tool_call>") {
        content.push_str(&s[..i]);
        let body = &s[i + "<tool_call>".len()..];
        let (inner, after) = match body.find("</tool_call>") {
            Some(j) => (&body[..j], &body[j + "</tool_call>".len()..]),
            None => (body, ""),
        };
        calls.push(parse_tool_call(inner));
        s = after;
    }
    content.push_str(s);
    // strip stop tokens that may have been decoded
    for stop in ["<|user|>", "<|observation|>", "<|endoftext|>"] {
        if let Some(i) = content.find(stop) {
            content.truncate(i);
        }
    }
    (reasoning, content.trim().to_string(), calls)
}

fn parse_tool_call(inner: &str) -> ToolCall {
    let name_end = inner.find("<arg_key>").unwrap_or(inner.len());
    let name = inner[..name_end].trim().to_string();
    let mut args = serde_json::Map::new();
    let mut s = &inner[name_end..];
    while let Some(i) = s.find("<arg_key>") {
        let after = &s[i + 9..];
        let Some(j) = after.find("</arg_key>") else { break };
        let key = after[..j].trim().to_string();
        let after = &after[j + 10..];
        let Some(k) = after.find("<arg_value>") else { break };
        let v0 = &after[k + 11..];
        let (val, rest) = match v0.find("</arg_value>") {
            Some(e) => (&v0[..e], &v0[e + 12..]),
            None => (v0, ""),
        };
        let parsed = serde_json::from_str::<Json>(val).ok().filter(|v| !v.is_string()).unwrap_or(Json::String(val.to_string()));
        args.insert(key, parsed);
        s = rest;
    }
    ToolCall { name, arguments: Json::Object(args) }
}

/// Incremental detokenizer + think/content/tool-call splitter for streaming.
pub struct StreamParser<'a> {
    vocab: &'a Vocab,
    bytes: Vec<u8>,
    pub in_think: bool,
    in_tool: bool,
    pub text: String, // everything decoded so far
}

#[derive(Debug, PartialEq, Eq)]
pub enum Piece {
    Reasoning(String),
    Content(String),
    /// text inside a tool call (not streamed to clients as content)
    Tool(String),
    None,
}

impl<'a> StreamParser<'a> {
    pub fn new(vocab: &'a Vocab, in_think: bool) -> Self {
        Self { vocab, bytes: Vec::new(), in_think, in_tool: false, text: String::new() }
    }
    /// Feed one generated token; returns the newly decodable text and where it belongs.
    pub fn push(&mut self, tok: u32) -> Piece {
        let piece = self.vocab.token_bytes(tok);
        let s = String::from_utf8_lossy(&piece);
        match s.as_ref() {
            "<think>" => {
                self.in_think = true;
                self.text.push_str("<think>");
                return Piece::None;
            }
            "</think>" => {
                self.in_think = false;
                self.text.push_str("</think>");
                return Piece::None;
            }
            "<tool_call>" => {
                self.in_tool = true;
                self.text.push_str("<tool_call>");
                return Piece::Tool(String::new());
            }
            "</tool_call>" => {
                self.in_tool = false;
                self.text.push_str("</tool_call>");
                return Piece::Tool(String::new());
            }
            _ => {}
        }
        self.bytes.extend_from_slice(&piece);
        // emit the longest valid UTF-8 prefix
        let valid = match std::str::from_utf8(&self.bytes) {
            Ok(_) => self.bytes.len(),
            Err(e) if e.error_len().is_none() => e.valid_up_to(),
            Err(_) => self.bytes.len(), // invalid bytes: emit lossily
        };
        if valid == 0 {
            return Piece::None;
        }
        let out = String::from_utf8_lossy(&self.bytes[..valid]).into_owned();
        self.bytes.drain(..valid);
        self.text.push_str(&out);
        if self.in_tool {
            Piece::Tool(out)
        } else if self.in_think {
            Piece::Reasoning(out)
        } else {
            Piece::Content(out)
        }
    }
}
