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

/// Finds special-token strings in a text, earliest first (longest at equal position), in time
/// roughly linear in the text (T2): per string it caches the position of its next acceptable
/// occurrence and searches again only when the scan has moved past it. Never-occurring strings
/// are searched once.
pub(crate) struct SpecialScan<'a> {
    text: &'a str,
    specials: Vec<&'a str>,
    /// per special: None = not searched yet, Some(None) = no further occurrence, Some(Some(p))
    cache: Vec<Option<Option<usize>>>,
}

impl<'a> SpecialScan<'a> {
    pub(crate) fn new(text: &'a str, specials: impl Iterator<Item = &'a str>) -> Self {
        let specials: Vec<&str> = specials.collect();
        let cache = vec![None; specials.len()];
        Self { text, specials, cache }
    }

    /// Earliest occurrence starting at or after `at` for which `ok(start, end)` holds: (position,
    /// index into the specials). `ok` must depend only on the range (not on `at`).
    pub(crate) fn next(&mut self, at: usize, ok: impl Fn(usize, usize) -> bool) -> Option<(usize, usize)> {
        let mut hit: Option<(usize, usize)> = None;
        for (i, s) in self.specials.iter().enumerate() {
            if s.is_empty() {
                continue;
            }
            let p = match self.cache[i] {
                Some(None) => continue,
                Some(Some(p)) if p >= at => p,
                _ => {
                    // first acceptable occurrence at or after `at`
                    let mut from = at;
                    let found = loop {
                        let Some(q) = self.text[from..].find(s) else { break None };
                        let q = from + q;
                        if ok(q, q + s.len()) {
                            break Some(q);
                        }
                        from = q + self.text[q..].chars().next().map_or(1, char::len_utf8);
                    };
                    self.cache[i] = Some(found);
                    match found {
                        Some(p) => p,
                        None => continue,
                    }
                }
            };
            if hit.is_none_or(|(hp, hi)| p < hp || (p == hp && s.len() > self.specials[hi].len())) {
                hit = Some((p, i));
            }
        }
        hit
    }
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
                // token_type may be longer than tokens in a malformed file (T1): skip the extra entries
                if matches!(t.as_u64(), Some(3) | Some(4)) && tokens.get(i).is_some_and(|s| !s.is_empty()) {
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
        if parse_special {
            self.encode_with_literals(text, &[])
        } else {
            let mut out = Vec::new();
            self.encode_plain(text, &mut out);
            out
        }
    }

    /// The special-token strings (control / user-defined tokens), longest first.
    pub fn special_strings(&self) -> impl Iterator<Item = &str> {
        self.specials.iter().map(|(s, _)| s.as_str())
    }

    /// Encode with special-token parsing, except that a special-token match overlapping one of the
    /// `literals` byte ranges is not taken: that text stays ordinary text (S3: special strings
    /// typed inside user / tool content). Everything else is tokenized exactly as `encode(.., true)`.
    pub fn encode_with_literals(&self, text: &str, literals: &[std::ops::Range<usize>]) -> Vec<u32> {
        // ranges from restore_specials are ascending and disjoint; anything else is sorted and merged
        let merged;
        let literals = if literals.windows(2).all(|w| w[0].end <= w[1].start) {
            literals
        } else {
            let mut v = literals.to_vec();
            v.sort_by_key(|r| r.start);
            let mut m: Vec<std::ops::Range<usize>> = Vec::with_capacity(v.len());
            for r in v {
                match m.last_mut() {
                    Some(l) if r.start <= l.end => l.end = l.end.max(r.end),
                    _ => m.push(r),
                }
            }
            merged = m;
            &merged[..]
        };
        // binary search: the first range ending after `a` is the only candidate for overlap
        let overlaps = |a: usize, b: usize| {
            let i = literals.partition_point(|r| r.end <= a);
            i < literals.len() && literals[i].start < b
        };
        let mut scan = SpecialScan::new(text, self.specials.iter().map(|(s, _)| s.as_str()));
        let mut out = Vec::new();
        let mut at = 0usize;
        while at < text.len() {
            let hit = scan.next(at, |a, b| !overlaps(a, b));
            let end = hit.map_or(text.len(), |(p, _)| p);
            self.encode_plain(&text[at..end], &mut out);
            let Some((p, i)) = hit else { break };
            out.push(self.specials[i].1);
            at = p + self.specials[i].0.len();
        }
        out
    }

    /// The original scan (every special searched again after every hit), kept to check against.
    #[cfg(test)]
    fn encode_with_literals_reference(&self, text: &str, literals: &[std::ops::Range<usize>]) -> Vec<u32> {
        let overlaps = |a: usize, b: usize| literals.iter().any(|r| a < r.end && r.start < b);
        let mut out = Vec::new();
        let mut base = 0usize; // offset of `rest` in `text`
        let mut rest = text;
        while !rest.is_empty() {
            // earliest special token occurrence (longest wins at equal position)
            let mut hit: Option<(usize, &str, u32)> = None;
            for (s, id) in &self.specials {
                let mut from = 0;
                while let Some(p) = rest[from..].find(s.as_str()) {
                    let p = from + p;
                    if !overlaps(base + p, base + p + s.len()) {
                        if hit.is_none_or(|(hp, hs, _)| p < hp || (p == hp && s.len() > hs.len())) {
                            hit = Some((p, s, *id));
                        }
                        break;
                    }
                    from = p + rest[p..].chars().next().map_or(1, char::len_utf8);
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
                    base += p + len;
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

    /// Byte-pair merges of one pre-token: repeatedly merge the adjacent pair with the lowest rank
    /// (leftmost on ties). O(n log n) with a heap of candidate pairs (T2); stale candidates are
    /// recognised by their symbol boundaries. Same output as the quadratic `bpe_reference`.
    fn bpe(&self, word: &str, out: &mut Vec<u32>) {
        use std::cmp::Reverse;
        let chars: Vec<char> = word.chars().collect();
        let n = chars.len();
        if n == 0 {
            return;
        }
        let text = |a: usize, b: usize| -> String { chars[a..b].iter().collect() };
        let rank = |a: usize, m: usize, b: usize| self.ranks.get(&(text(a, m), text(m, b))).copied();
        // symbol = chars[start..end[start]], identified by its start; linked through end / prev
        let mut end: Vec<usize> = (1..=n).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive = vec![true; n];
        // candidates (rank, left start, right start, right end): min rank, then leftmost
        let mut heap = std::collections::BinaryHeap::new();
        for i in 0..n - 1 {
            if let Some(r) = rank(i, i + 1, i + 2) {
                heap.push(Reverse((r, i, i + 1, i + 2)));
            }
        }
        while let Some(Reverse((_, l, m, e))) = heap.pop() {
            // still the same two adjacent symbols? (starts never move, so boundaries identify them)
            if !alive[l] || !alive[m] || end[l] != m || end[m] != e {
                continue;
            }
            alive[m] = false;
            end[l] = e;
            if e < n {
                prev[e] = l;
            }
            let p = prev[l];
            if p != usize::MAX {
                if let Some(r) = rank(p, l, e) {
                    heap.push(Reverse((r, p, l, e)));
                }
            }
            if e < n {
                if let Some(r) = rank(l, e, end[e]) {
                    heap.push(Reverse((r, l, e, end[e])));
                }
            }
        }
        let mut i = 0;
        while i < n {
            self.push_symbol(&text(i, end[i]), out);
            i = end[i];
        }
    }

    /// Id of a final BPE symbol, or byte fallback: one token per byte of the symbol.
    fn push_symbol(&self, s: &str, out: &mut Vec<u32>) {
        match self.id_of.get(s) {
            Some(&id) => out.push(id),
            None => {
                for c in s.chars() {
                    if let Some(&id) = self.id_of.get(&c.to_string()) {
                        out.push(id);
                    }
                }
            }
        }
    }

    /// The original quadratic BPE, kept to check `bpe` against.
    #[cfg(test)]
    fn bpe_reference(&self, word: &str, out: &mut Vec<u32>) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn t1_token_type_longer_than_tokens_does_not_panic() {
        let mut kv = HashMap::new();
        let strs = |v: &[&str]| Value::Arr(v.iter().map(|s| Value::Str(s.to_string())).collect());
        kv.insert("tokenizer.ggml.tokens".to_string(), strs(&["a", "<|user|>"]));
        kv.insert("tokenizer.ggml.token_type".to_string(), Value::Arr(vec![Value::I32(1), Value::I32(3), Value::I32(3), Value::I32(4)]));
        let g = Gguf { shards: vec![], kv, tensors: vec![], by_name: HashMap::new() };
        let v = Vocab::from_gguf(&g).unwrap();
        assert_eq!(v.specials, vec![("<|user|>".to_string(), 1)]);
    }

    /// Byte-level vocab (256 byte symbols) + a few merges + two control tokens.
    fn small_vocab() -> Vocab {
        let mut tokens: Vec<String> = bytes_to_unicode().into_iter().map(|(_, c)| c.to_string()).collect();
        let merges = ["h e", "l l", "he ll", "Ġ w", "o r"];
        for m in merges {
            tokens.push(m.replace(' ', ""));
        }
        tokens.push("<|user|>".into());
        tokens.push("<|assistant|>".into());
        let n = tokens.len();
        let mut types = vec![Value::I32(1); n];
        types[n - 2] = Value::I32(3);
        types[n - 1] = Value::I32(3);
        let mut kv = HashMap::new();
        kv.insert("tokenizer.ggml.tokens".to_string(), Value::Arr(tokens.iter().map(|s| Value::Str(s.clone())).collect()));
        kv.insert("tokenizer.ggml.token_type".to_string(), Value::Arr(types));
        kv.insert("tokenizer.ggml.merges".to_string(), Value::Arr(merges.iter().map(|s| Value::Str(s.to_string())).collect()));
        Vocab::from_gguf(&Gguf { shards: vec![], kv, tensors: vec![], by_name: HashMap::new() }).unwrap()
    }

    #[test]
    fn s3_special_inside_literal_range_is_not_a_control_token() {
        let v = small_vocab();
        let user = v.id_of["<|user|>"];
        let asst = v.id_of["<|assistant|>"];
        let text = "<|user|>hello <|assistant|> world<|assistant|>";
        let all = v.encode(text, true);
        assert_eq!(all.iter().filter(|&&t| t == asst).count(), 2);
        // the first <|assistant|> lies in a literal range (typed by the user)
        let p = text.find("<|assistant|>").unwrap();
        let lit = v.encode_with_literals(text, &[p..p + "<|assistant|>".len()]);
        assert_eq!(lit[0], user);
        assert_eq!(lit.iter().filter(|&&t| t == asst).count(), 1, "only the template's one stays special");
        assert_eq!(*lit.last().unwrap(), asst);
        assert_eq!(v.decode(&lit), text, "the literal decodes back to the same text");
        // a range that only partly overlaps also disables the match
        let lit2 = v.encode_with_literals(text, &[p + 3..p + 4]);
        assert_eq!(lit2, lit);
    }

    #[test]
    fn t2_special_scan_is_fast_and_matches_reference() {
        let v = small_vocab();
        let p = "<|assistant|>".len();
        // 20,000 special strings; one of them in a literal range
        let big = "hi <|user|>x<|assistant|>".repeat(10_000);
        let lit = [3..3 + 8, 112..112 + p]; // the 1st <|user|> and the 5th <|assistant|>
        let t0 = std::time::Instant::now();
        let fast = v.encode_with_literals(&big, &lit);
        assert!(t0.elapsed().as_secs_f64() < 1.0, "took {:?}", t0.elapsed());
        // the old scan is quadratic: compare on a prefix, plus counts on the whole text
        let n = 2000 * "hi <|user|>x<|assistant|>".len();
        assert_eq!(v.encode_with_literals(&big[..n], &lit), v.encode_with_literals_reference(&big[..n], &lit));
        let user = v.id_of["<|user|>"];
        assert_eq!(fast.iter().filter(|&&t| t == user).count(), 9_999);
        // every special inside a literal range: 20,000 ranges, nothing may become a control token
        let unit = "hi <|user|>x<|assistant|>";
        let all: Vec<std::ops::Range<usize>> = (0..10_000)
            .flat_map(|k| {
                let o = k * unit.len();
                [o + 3..o + 11, o + 12..o + 12 + p]
            })
            .collect();
        let t0 = std::time::Instant::now();
        let lit_all = v.encode_with_literals(&big, &all);
        assert!(t0.elapsed().as_secs_f64() < 1.0, "took {:?}", t0.elapsed());
        assert!(!lit_all.contains(&user) && !lit_all.contains(&v.id_of["<|assistant|>"]));
        let m = 1000 * unit.len();
        let few: Vec<_> = all.iter().filter(|r| r.end <= m).cloned().collect();
        assert_eq!(v.encode_with_literals(&big[..m], &few), v.encode_with_literals_reference(&big[..m], &few));
        // unsorted / overlapping input is still handled exactly
        let messy = vec![12..25, 3..11, 5..9, 40..45];
        assert_eq!(v.encode_with_literals(&big[..m], &messy), v.encode_with_literals_reference(&big[..m], &messy));
        // specials that never occur: plain text of the same size
        let plain = "hello world ".repeat(20_000);
        let t0 = std::time::Instant::now();
        let a = v.encode_with_literals(&plain, &[]);
        assert!(t0.elapsed().as_secs_f64() < 1.0, "took {:?}", t0.elapsed());
        assert_eq!(a, v.encode_with_literals_reference(&plain, &[]));
        for t in ["", "<|user|><|user|>", "<|user|<|assistant|>", "é<|user|>ü", "<|assistant|><|user"] {
            for l in [vec![], vec![0..1], vec![1..3, 9..10]] {
                assert_eq!(v.encode_with_literals(t, &l), v.encode_with_literals_reference(t, &l), "{t:?} {l:?}");
            }
        }
    }

    #[test]
    fn s3_benign_text_tokenizes_identically() {
        let v = small_vocab();
        for text in ["hello world", "<|user|>hello<|assistant|>", "héllo wörld 123 \n\t ok", "", "<|user", "x<|assistant|>"] {
            let a = v.encode(text, true);
            assert_eq!(a, v.encode_with_literals(text, &[]), "{text:?}");
            // a literal range over text that holds no special string changes nothing
            if !text.contains("<|") {
                assert_eq!(a, v.encode_with_literals(text, &[0..text.len()]), "{text:?}");
            }
        }
        // pre-tokenization around a literal is unchanged: the literal's bytes are ordinary text
        let t = "hello <|user|> world";
        let p = t.find('<').unwrap();
        let lit = v.encode_with_literals(t, &[p..p + 8]);
        let mut plain = Vec::new();
        v.encode_plain(t, &mut plain);
        assert_eq!(lit, plain);
    }
}
#[cfg(test)]
mod t2_tests {
    use super::*;

    /// Tiny LCG for reproducible pseudo-random data.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 33
        }
    }

    fn vocab_with(tokens: Vec<String>, merges: Vec<String>) -> Vocab {
        let mut kv = HashMap::new();
        kv.insert("tokenizer.ggml.tokens".to_string(), Value::Arr(tokens.into_iter().map(Value::Str).collect()));
        kv.insert("tokenizer.ggml.merges".to_string(), Value::Arr(merges.into_iter().map(Value::Str).collect()));
        Vocab::from_gguf(&Gguf { shards: vec![], kv, tensors: vec![], by_name: HashMap::new() }).unwrap()
    }

    fn same(v: &Vocab, w: &str) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        v.bpe(w, &mut a);
        v.bpe_reference(w, &mut b);
        assert_eq!(a, b, "{w:?}");
    }

    #[test]
    fn t2_fast_bpe_matches_reference_on_random_strings() {
        let mut rng = Lcg(42);
        let mut syms: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let mut tokens = syms.clone();
        let mut merges = Vec::new();
        while merges.len() < 80 {
            let x = syms[rng.next() as usize % syms.len()].clone();
            let y = syms[rng.next() as usize % syms.len()].clone();
            let m = format!("{x} {y}");
            if merges.contains(&m) || x.len() + y.len() > 8 {
                continue;
            }
            merges.push(m);
            syms.push(format!("{x}{y}"));
            if merges.len() % 7 != 0 {
                tokens.push(format!("{x}{y}")); // some merged symbols are not tokens: byte fallback
            }
        }
        let v = vocab_with(tokens, merges);
        for _ in 0..4000 {
            let len = 1 + rng.next() as usize % 48;
            let w: String = (0..len).map(|_| (b'a' + (rng.next() % 4) as u8) as char).collect();
            same(&v, &w);
        }
        same(&v, &"ab".repeat(500));
        same(&v, &"a".repeat(777));
    }

    #[test]
    fn t2_fast_bpe_matches_reference_on_words() {
        let mut tokens: Vec<String> = bytes_to_unicode().into_iter().map(|(_, c)| c.to_string()).collect();
        let merges: Vec<String> =
            ["t h", "th e", "i n", "e r", "a n", "Ġ t", "Ġt he", "o n", "r e", "in g", "e n", "Ġ a", "an d", "Ġa nd", "o r", "e s", "l l", "e ll"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        tokens.extend(merges.iter().map(|m| m.replace(' ', "")));
        let v = vocab_with(tokens, merges);
        for w in ["the", "Ġthe", "theremin", "inning", "Ġand", "andand", "hello", "ŀ", "rerere", "ellen", "", "x", "tttthhhheeee"] {
            same(&v, w);
        }
    }
}