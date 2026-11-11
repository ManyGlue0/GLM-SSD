//! GLM byte-level BPE tokenizer from the GGUF vocabulary, matching llama.cpp's `glm4`/`glm5`
//! pre-tokenizer: CHATGLM4 split regex, `ignore_merges`, no automatic BOS, special tokens parsed.

use anyhow::{Context, Result};
use gguf::{Gguf, Value};
use std::collections::HashMap;

const SPLIT_RE: &str = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct Vocab {
    tokens: Vec<String>,
    byte_of: HashMap<char, u8>,
    char_of: [char; 256],
    id_of: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    /// control / user-defined tokens, longest first
    specials: Vec<(String, u32)>,
    re: fancy_regex::Regex,
}

/// GPT-2 `bytes_to_unicode`: printable bytes map to themselves, the rest to U+0100.. in order.
fn bytes_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter().zip(cs).map(|(b, c)| (b as u8, char::from_u32(c).unwrap())).collect()
}

impl Vocab {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let toks = g.get("tokenizer.ggml.tokens").and_then(Value::as_arr).context("tokenizer.ggml.tokens missing")?;
        let tokens: Vec<String> = toks.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        let b2u = bytes_to_unicode();
        let byte_of = b2u.iter().map(|(b, c)| (*c, *b)).collect();
        let mut char_of = ['\0'; 256];
        for (b, c) in &b2u {
            char_of[*b as usize] = *c;
        }
        let id_of: HashMap<String, u32> = tokens.iter().enumerate().map(|(i, t)| (t.clone(), i as u32)).collect();
        let mut ranks = HashMap::new();
        if let Some(m) = g.get("tokenizer.ggml.merges").and_then(Value::as_arr) {
            for (i, v) in m.iter().enumerate() {
                if let Some((a, b)) = v.as_str().and_then(|s| s.split_once(' ')) {
                    ranks.insert((a.to_string(), b.to_string()), i as u32);
                }
            }
        }
        // token types: 1 normal, 2 unknown, 3 control, 4 user defined, 5 unused, 6 byte
        let mut specials: Vec<(String, u32)> = Vec::new();
        if let Some(tt) = g.get("tokenizer.ggml.token_type").and_then(Value::as_arr) {
            for (i, t) in tt.iter().enumerate() {
                if matches!(t.as_u64(), Some(3) | Some(4)) && !tokens[i].is_empty() {
                    specials.push((tokens[i].clone(), i as u32));
                }
            }
        }
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let re = fancy_regex::Regex::new(SPLIT_RE).context("tokenizer regex")?;
        Ok(Self { tokens, byte_of, char_of, id_of, ranks, specials, re })
    }

    /// Encode text. With `parse_special`, special-token strings (e.g. `<|user|>`) become their ids.
    pub fn encode(&self, text: &str, parse_special: bool) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // earliest special token occurrence (longest wins at equal position)
            let mut hit: Option<(usize, &str, u32)> = None;
            if parse_special {
                for (s, id) in &self.specials {
                    if let Some(p) = rest.find(s.as_str()) {
                        if hit.is_none_or(|(hp, hs, _)| p < hp || (p == hp && s.len() > hs.len())) {
                            hit = Some((p, s, *id));
                        }
                    }
                }
            }
            let (plain, special) = match hit {
                Some((p, s, id)) => (&rest[..p], Some((s.len(), id, p))),
                None => (rest, None),
            };
            self.encode_plain(plain, &mut out);
            match special {
                Some((len, id, p)) => {
                    out.push(id);
                    rest = &rest[p + len..];
                }
                None => break,
            }
        }
        out
    }

    fn encode_plain(&self, text: &str, out: &mut Vec<u32>) {
        for m in self.re.find_iter(text) {
            let Ok(m) = m else { continue };
            let word: String = m.as_str().bytes().map(|b| self.char_of[b as usize]).collect();
            if let Some(&id) = self.id_of.get(&word) {
                out.push(id); // ignore_merges: whole pre-token is in the vocab
                continue;
            }
            self.bpe(&word, out);
        }
    }

    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                if let Some(&r) = self.ranks.get(&(syms[i].clone(), syms[i + 1].clone())) {
                    if best.is_none_or(|(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", syms[i], syms[i + 1]);
            syms[i] = merged;
            syms.remove(i + 1);
        }
        for s in syms {
            match self.id_of.get(&s) {
                Some(&id) => out.push(id),
                None => {
                    // byte fallback: one token per byte of the symbol
                    for c in s.chars() {
                        if let Some(&id) = self.id_of.get(&c.to_string()) {
                            out.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Raw bytes of one token (special tokens like `<|user|>` come back verbatim).
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        let Some(t) = self.tokens.get(id as usize) else { return Vec::new() };
        if t.starts_with("<|") || t.starts_with('[') && t.ends_with(']') || t.starts_with("<") && t.ends_with('>') {
            if !t.chars().all(|c| self.byte_of.contains_key(&c)) {
                return t.as_bytes().to_vec();
            }
        }
        let mut out = Vec::with_capacity(t.len());
        for c in t.chars() {
            match self.byte_of.get(&c) {
                Some(b) => out.push(*b),
                None => out.extend_from_slice(c.to_string().as_bytes()),
            }
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids.iter().flat_map(|i| self.token_bytes(*i)).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

}
