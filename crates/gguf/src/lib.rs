//! Minimal GGUF v3 reader.
//!
//! Only the header (metadata + tensor infos) is parsed; tensor data is never touched here.
//! Split models (`*-00001-of-0000N.gguf`) are opened as one logical model: every tensor gets
//! an absolute location `(shard, offset, n_bytes)` that the engine uses for direct I/O.
//!
//! The file is untrusted: every count, length and size is bounded and computed with checked
//! arithmetic, so a crafted header yields an error instead of a panic, an abort on allocation or
//! a wrapped size that would pass the bounds checks.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

mod types;
pub use types::GgmlType;

/// Most tensors / metadata keys a header may declare.
pub const MAX_COUNT: u64 = 1 << 20;
/// Longest string in the metadata (bytes).
pub const MAX_LEN: u64 = 1 << 26;
/// Longest array in the metadata (elements; each becomes a 32-byte `Value`).
pub const MAX_ARR: u64 = 1 << 24;
/// Most elements pre-allocated from a declared count before they are actually read.
const MAX_PREALLOC: u64 = 65536;
/// Deepest array nesting accepted (an array of arrays).
const MAX_DEPTH: u32 = 2;
/// Largest `general.alignment`.
const MAX_ALIGN: u64 = 1 << 20;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64,
            Value::U16(v) => v as u64,
            Value::U32(v) => v as u64,
            Value::U64(v) => v,
            Value::I8(v) if v >= 0 => v as u64,
            Value::I16(v) if v >= 0 => v as u64,
            Value::I32(v) if v >= 0 => v as u64,
            Value::I64(v) if v >= 0 => v as u64,
            Value::Bool(v) => v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        Some(match *self {
            Value::F32(v) => v as f64,
            Value::F64(v) => v,
            _ => return self.as_u64().map(|v| v as f64),
        })
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(a) => Some(a),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// GGML order: dims[0] is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    pub shard: usize,
    /// Absolute byte offset inside the shard file.
    pub offset: u64,
    pub n_bytes: u64,
}

impl TensorInfo {
    /// Product of the dims (saturating; the parser guarantees it fits for parsed tensors).
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().try_fold(1u64, |a, &d| a.checked_mul(d)).unwrap_or(u64::MAX)
    }
    /// Bytes of one row (dims[0] elements); 0 for a malformed tensor (never produced by the parser).
    pub fn row_bytes(&self) -> u64 {
        self.dims.first().and_then(|&d| self.ty.try_row_bytes(d)).unwrap_or(0)
    }
    /// For 3-D expert tensors `[in, out, n_expert]`: byte range of expert `e`'s slice.
    pub fn slice_range(&self, e: u64) -> (u64, u64) {
        let last = self.dims.last().copied().unwrap_or(1).max(1);
        let per = self.n_bytes / last;
        (self.offset.saturating_add(e.saturating_mul(per)), per)
    }
}

pub struct Gguf {
    pub shards: Vec<PathBuf>,
    pub kv: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub by_name: HashMap<String, usize>,
}

impl Gguf {
    /// Open a model. `path` may be any shard of a split model or a single-file model.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_impl(path, true)
    }

    /// Like [`open`](Self::open) but doesn't require tensor data to be present
    /// (for inspecting truncated header-only copies of shards).
    pub fn open_headers_only(path: &Path) -> Result<Self> {
        Self::open_impl(path, false)
    }

    /// Open exactly one file (one shard of a split model), without looking for siblings.
    pub fn open_shard(path: &Path) -> Result<Self> {
        let (kv, tensors) = read_header(path, 0, true)?;
        let by_name = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(Self { shards: vec![path.to_path_buf()], kv, tensors, by_name })
    }

    fn open_impl(path: &Path, check_len: bool) -> Result<Self> {
        let shards = discover_shards(path)?;
        let mut kv = HashMap::new();
        let mut tensors = Vec::new();
        for (si, sp) in shards.iter().enumerate() {
            let (skv, mut st) = read_header(sp, si, check_len).with_context(|| format!("reading {}", sp.display()))?;
            if si == 0 {
                kv = skv;
            }
            tensors.append(&mut st);
            if tensors.len() as u64 > MAX_COUNT {
                bail!("too many tensors across shards");
            }
        }
        if let Some(n) = kv.get("split.tensors.count").and_then(Value::as_u64) {
            if n != tensors.len() as u64 {
                bail!("split.tensors.count = {n} but found {} tensors across shards", tensors.len());
            }
        }
        let by_name = tensors.iter().enumerate().map(|(i, t)| (t.name.clone(), i)).collect();
        Ok(Self { shards, kv, tensors, by_name })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.kv.get(key)
    }
    pub fn u64(&self, key: &str) -> Result<u64> {
        self.kv.get(key).and_then(Value::as_u64).with_context(|| format!("missing/invalid key {key}"))
    }
    pub fn f64(&self, key: &str) -> Result<f64> {
        self.kv.get(key).and_then(Value::as_f64).with_context(|| format!("missing/invalid key {key}"))
    }
    pub fn str(&self, key: &str) -> Result<&str> {
        self.kv.get(key).and_then(Value::as_str).with_context(|| format!("missing/invalid key {key}"))
    }
    pub fn arch(&self) -> Result<&str> {
        self.str("general.architecture")
    }
    /// `<arch>.<suffix>`
    pub fn arch_key(&self, suffix: &str) -> String {
        format!("{}.{}", self.arch().unwrap_or("?"), suffix)
    }
}

/// `foo-00002-of-00004` (stem without `.gguf`) -> `Some(("foo", 4))` when the shard pattern is
/// present; errors when it is present but invalid (total 0 or > 9999, index out of range).
fn split_pattern(stem: &str) -> Result<Option<(&str, usize)>> {
    // the pattern is ASCII: compare bytes, and only cut the str where the tail starts
    let b = stem.as_bytes();
    if b.len() <= 15 {
        return Ok(None);
    }
    let tail = &b[b.len() - 15..];
    let digits = |s: &[u8]| s.iter().all(|c| c.is_ascii_digit());
    if !(tail[0] == b'-' && &tail[6..10] == b"-of-" && digits(&tail[1..6]) && digits(&tail[10..])) {
        return Ok(None);
    }
    // tail is all ASCII, so b.len()-15 is a char boundary
    let base = &stem[..b.len() - 15];
    let num = |s: &[u8]| s.iter().fold(0usize, |a, &c| a * 10 + (c - b'0') as usize);
    let (idx, total) = (num(&tail[1..6]), num(&tail[10..]));
    if !(1..=9999).contains(&total) {
        bail!("invalid shard count {total} in {stem}");
    }
    if !(1..=total).contains(&idx) {
        bail!("shard index {idx} out of range 1..={total} in {stem}");
    }
    Ok(Some((base, total)))
}

/// `foo-00002-of-00004.gguf` -> all 4 shard paths (that must exist). Otherwise just `path`.
fn discover_shards(path: &Path) -> Result<Vec<PathBuf>> {
    let name = path.file_name().and_then(|n| n.to_str()).context("bad path")?;
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    if let Some((base, total)) = split_pattern(stem)? {
        let dir = path.parent().unwrap_or(Path::new("."));
        let v: Vec<PathBuf> = (1..=total).map(|i| dir.join(format!("{base}-{i:05}-of-{total:05}.gguf"))).collect();
        for p in &v {
            if !p.exists() {
                bail!("missing shard {}", p.display());
            }
        }
        return Ok(v);
    }
    Ok(vec![path.to_path_buf()])
}

struct Rd<R: Read> {
    r: R,
    pos: u64,
    /// total length of the input (strings / arrays can never be longer than what is left)
    len: u64,
}
impl<R: Read> Rd<R> {
    fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        self.pos += N as u64;
        Ok(b)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > MAX_LEN || n > self.remaining() {
            bail!("string too long ({n} bytes)");
        }
        let mut v = Vec::new();
        (&mut self.r).take(n).read_to_end(&mut v)?;
        if v.len() as u64 != n {
            bail!("truncated string");
        }
        self.pos += n;
        Ok(String::from_utf8_lossy(&v).into_owned())
    }
    fn value(&mut self, t: u32, depth: u32) -> Result<Value> {
        Ok(match t {
            0 => Value::U8(self.bytes::<1>()?[0]),
            1 => Value::I8(self.bytes::<1>()?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.bytes()?)),
            3 => Value::I16(i16::from_le_bytes(self.bytes()?)),
            4 => Value::U32(self.u32()?),
            5 => Value::I32(i32::from_le_bytes(self.bytes()?)),
            6 => Value::F32(f32::from_le_bytes(self.bytes()?)),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth >= MAX_DEPTH {
                    bail!("metadata arrays nested too deeply");
                }
                let et = self.u32()?;
                let n = self.u64()?;
                // every element takes at least one byte of the file
                if n > MAX_ARR || n > self.remaining() {
                    bail!("metadata array too long ({n} elements)");
                }
                let mut v = Vec::with_capacity(n.min(MAX_PREALLOC) as usize);
                for _ in 0..n {
                    v.push(self.value(et, depth + 1)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(self.bytes()?)),
            12 => Value::F64(f64::from_le_bytes(self.bytes()?)),
            _ => bail!("unknown GGUF value type {t}"),
        })
    }
}

fn read_header(path: &Path, shard: usize, check_len: bool) -> Result<(HashMap<String, Value>, Vec<TensorInfo>)> {
    let f = File::open(path)?;
    let file_len = f.metadata()?.len();
    parse_header(BufReader::with_capacity(1 << 20, f), file_len, shard, check_len)
        .with_context(|| format!("parsing {}", path.display()))
}

/// Parse a GGUF header from `r` (`file_len` bytes long). With `check_len`, every tensor must lie
/// inside the file; without it (header-only copies) only the arithmetic is checked.
fn parse_header(r: impl Read, file_len: u64, shard: usize, check_len: bool) -> Result<(HashMap<String, Value>, Vec<TensorInfo>)> {
    let mut r = Rd { r, pos: 0, len: file_len };
    if &r.bytes::<4>()? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let ver = r.u32()?;
    if ver != 3 {
        bail!("unsupported GGUF version {ver}");
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > MAX_COUNT || n_kv > MAX_COUNT {
        bail!("implausible header: {n_tensors} tensors, {n_kv} metadata keys");
    }
    let mut kv = HashMap::with_capacity(n_kv.min(MAX_PREALLOC) as usize);
    for _ in 0..n_kv {
        let k = r.string()?;
        let t = r.u32()?;
        let v = r.value(t, 0)?;
        kv.insert(k, v);
    }
    let align = match kv.get("general.alignment") {
        None => 32,
        Some(v) => match v.as_u64() {
            Some(a) if a.is_power_of_two() && a <= MAX_ALIGN => a,
            _ => bail!("invalid general.alignment {v:?} (must be a power of two <= {MAX_ALIGN})"),
        },
    };
    let mut raw = Vec::with_capacity(n_tensors.min(MAX_PREALLOC) as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()?;
        if !(1..=4).contains(&nd) {
            bail!("tensor {name}: {nd} dimensions (must be 1..=4)");
        }
        let dims = (0..nd).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
        if dims.contains(&0) {
            bail!("tensor {name}: zero dimension in {dims:?}");
        }
        let ty = GgmlType::from_id(r.u32()?).with_context(|| format!("tensor {name}"))?;
        let off = r.u64()?;
        raw.push((name, dims, ty, off));
    }
    // r.pos <= file_len here (everything up to it was read), so this cannot overflow
    let data_start = r.pos.div_ceil(align) * align;
    let tensors = raw
        .into_iter()
        .map(|(name, dims, ty, off)| {
            let n_bytes = tensor_bytes(ty, &dims).with_context(|| format!("tensor {name}: bad shape {dims:?} for {ty:?}"))?;
            let offset = data_start.checked_add(off).with_context(|| format!("tensor {name}: offset overflows"))?;
            let end = offset.checked_add(n_bytes).with_context(|| format!("tensor {name}: end overflows"))?;
            if check_len && end > file_len {
                bail!("tensor {name} extends past the end of the file");
            }
            Ok(TensorInfo { name, dims, ty, shard, offset, n_bytes })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((kv, tensors))
}

/// Bytes of a tensor: whole blocks per row times the other dims, all checked.
fn tensor_bytes(ty: GgmlType, dims: &[u64]) -> Option<u64> {
    let (&d0, rest) = dims.split_first()?;
    rest.iter().try_fold(ty.try_row_bytes(d0)?, |a, &d| a.checked_mul(d))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny GGUF writer for tests.
    #[derive(Default)]
    struct W {
        kv: Vec<u8>,
        n_kv: u64,
        ti: Vec<u8>,
        n_t: u64,
    }
    fn s(b: &mut Vec<u8>, v: &str) {
        b.extend((v.len() as u64).to_le_bytes());
        b.extend(v.as_bytes());
    }
    impl W {
        fn kv_u32(&mut self, k: &str, v: u32) -> &mut Self {
            s(&mut self.kv, k);
            self.kv.extend(4u32.to_le_bytes());
            self.kv.extend(v.to_le_bytes());
            self.n_kv += 1;
            self
        }
        fn kv_raw(&mut self, k: &str, ty: u32, body: &[u8]) -> &mut Self {
            s(&mut self.kv, k);
            self.kv.extend(ty.to_le_bytes());
            self.kv.extend(body);
            self.n_kv += 1;
            self
        }
        fn tensor(&mut self, name: &str, dims: &[u64], ty: u32, off: u64) -> &mut Self {
            s(&mut self.ti, name);
            self.ti.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                self.ti.extend(d.to_le_bytes());
            }
            self.ti.extend(ty.to_le_bytes());
            self.ti.extend(off.to_le_bytes());
            self.n_t += 1;
            self
        }
        /// header + `data` bytes of tensor data after the aligned data start
        fn build(&self, data: usize) -> Vec<u8> {
            self.build_counts(self.n_t, self.n_kv, data)
        }
        fn build_counts(&self, n_t: u64, n_kv: u64, data: usize) -> Vec<u8> {
            let mut b = b"GGUF".to_vec();
            b.extend(3u32.to_le_bytes());
            b.extend(n_t.to_le_bytes());
            b.extend(n_kv.to_le_bytes());
            b.extend(&self.kv);
            b.extend(&self.ti);
            b.resize(b.len().div_ceil(32) * 32 + data, 0);
            b
        }
    }
    fn parse(b: &[u8]) -> Result<(HashMap<String, Value>, Vec<TensorInfo>)> {
        parse_header(b, b.len() as u64, 0, true)
    }
    fn err(b: &[u8]) -> String {
        format!("{:#}", parse(b).expect_err("must be rejected"))
    }

    #[test]
    fn g_valid_minimal_file_parses() {
        let b = W::default().kv_u32("general.alignment", 32).tensor("a", &[64, 2], 0, 0).tensor("q", &[256, 3], 8, 512).build(512 + 3 * 272);
        let (kv, t) = parse(&b).unwrap();
        assert_eq!(kv["general.alignment"].as_u64(), Some(32));
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].n_bytes, 64 * 2 * 4);
        assert_eq!(t[1].n_bytes, 3 * 8 * 34);
        assert_eq!(t[1].row_bytes(), 8 * 34);
        assert_eq!(t[1].offset, t[0].offset + 512);
        // one byte short: rejected with check_len, accepted header-only
        let short = &b[..b.len() - 1];
        assert!(parse(short).is_err());
        assert!(parse_header(short, short.len() as u64, 0, false).is_ok());
    }

    #[test]
    fn g1_huge_tensor_count_is_rejected() {
        let b = W::default().build_counts(u64::MAX, 0, 0);
        assert!(err(&b).contains("implausible"));
        let b = W::default().build_counts(MAX_COUNT + 1, 0, 0);
        assert!(parse(&b).is_err());
        let b = W::default().build_counts(0, u64::MAX / 2, 0);
        assert!(parse(&b).is_err());
        // a plausible count with nothing behind it: plain EOF error, no huge pre-allocation
        let b = W::default().build_counts(MAX_COUNT, 0, 0);
        assert!(parse(&b).is_err());
    }

    #[test]
    fn g2_zero_dims_is_rejected() {
        let b = W::default().tensor("t", &[], 0, 0).build(64);
        assert!(err(&b).contains("dimensions"));
    }

    #[test]
    fn g2_five_dims_is_rejected() {
        let b = W::default().tensor("t", &[1, 1, 1, 1, 1], 0, 0).build(64);
        assert!(err(&b).contains("dimensions"));
    }

    #[test]
    fn g2_zero_dimension_is_rejected() {
        let b = W::default().tensor("t", &[32, 0], 0, 0).build(64);
        assert!(err(&b).contains("zero dimension"));
    }

    #[test]
    fn g3_dims_product_overflow_is_rejected() {
        let b = W::default().tensor("t", &[1 << 32, 1 << 32, 4], 0, 0).build(64);
        assert!(err(&b).contains("bad shape"));
        // row_bytes overflow (huge row of a block type)
        let b = W::default().tensor("t", &[u64::MAX / 32 * 32], 8, 0).build(64);
        assert!(err(&b).contains("bad shape"));
        // row not a multiple of the block: an error, not an assert
        let b = W::default().tensor("t", &[33], 8, 0).build(64);
        assert!(err(&b).contains("bad shape"));
    }

    #[test]
    fn g3_offset_plus_size_overflow_is_rejected() {
        let b = W::default().tensor("t", &[64], 0, u64::MAX - 8).build(64);
        let e = err(&b);
        assert!(e.contains("overflows"), "{e}");
        // also caught header-only (no file-length comparison there)
        assert!(parse_header(&b[..], b.len() as u64, 0, false).is_err());
        // a wrapped end must not pass the bounds check
        let b = W::default().tensor("t", &[1 << 40, 1 << 20], 0, 0).build(64);
        assert!(parse(&b).is_err());
        let b = W::default().tensor("t", &[64], 0, 1 << 20).build(64);
        assert!(err(&b).contains("past the end"));
    }

    #[test]
    fn g4_alignment_zero_and_three_are_rejected() {
        for a in [0u32, 3, 1 << 21] {
            let b = W::default().kv_u32("general.alignment", a).tensor("t", &[8], 0, 0).build(64);
            assert!(err(&b).contains("alignment"), "alignment {a}");
        }
        let b = W::default().kv_u32("general.alignment", 64).tensor("t", &[8], 0, 0).build(128);
        assert!(parse(&b).is_ok());
    }

    #[test]
    fn g5_array_nesting_is_limited() {
        // array (type 9) of arrays of arrays of u8
        let mut body = Vec::new();
        body.extend(9u32.to_le_bytes());
        body.extend(1u64.to_le_bytes());
        body.extend(9u32.to_le_bytes());
        body.extend(1u64.to_le_bytes());
        body.extend(0u32.to_le_bytes());
        body.extend(1u64.to_le_bytes());
        body.push(7);
        let b = W::default().kv_raw("deep", 9, &body).build(0);
        assert!(err(&b).contains("nested"));
        // depth 2 (array of arrays) is fine
        let mut ok = Vec::new();
        ok.extend(9u32.to_le_bytes());
        ok.extend(1u64.to_le_bytes());
        ok.extend(0u32.to_le_bytes());
        ok.extend(2u64.to_le_bytes());
        ok.extend([1, 2]);
        let b = W::default().kv_raw("aa", 9, &ok).build(0);
        assert!(parse(&b).is_ok());
        // an array claiming more elements than bytes left
        let mut long = Vec::new();
        long.extend(0u32.to_le_bytes());
        long.extend((1u64 << 40).to_le_bytes());
        let b = W::default().kv_raw("long", 9, &long).build(0);
        assert!(err(&b).contains("too long"));
        // over the element cap even when the file would hold that many bytes
        let mut long = Vec::new();
        long.extend(0u32.to_le_bytes());
        long.extend((MAX_ARR + 1).to_le_bytes());
        long.resize(long.len() + MAX_ARR as usize + 1, 0);
        let b = W::default().kv_raw("long", 9, &long).build(0);
        assert!(err(&b).contains("too long"));
    }

    #[test]
    fn g5_oversized_string_is_rejected() {
        let mut body = Vec::new();
        body.extend((1u64 << 29).to_le_bytes());
        body.extend(b"abc");
        let b = W::default().kv_raw("s", 8, &body).build(0);
        assert!(err(&b).contains("too long"));
        let mut body = Vec::new();
        body.extend(u64::MAX.to_le_bytes());
        let b = W::default().kv_raw("s", 8, &body).build(0);
        assert!(parse(&b).is_err());
    }

    #[test]
    fn g6_tensor_info_helpers_do_not_panic() {
        let t = TensorInfo { name: "x".into(), dims: vec![], ty: GgmlType::Q8_0, shard: 0, offset: 0, n_bytes: 10 };
        assert_eq!(t.row_bytes(), 0);
        assert_eq!(t.slice_range(3), (30, 10));
        let t = TensorInfo { dims: vec![33, 0], ..t };
        assert_eq!(t.row_bytes(), 0);
        let _ = t.slice_range(u64::MAX);
        assert_eq!(t.n_elements(), 0);
        let t = TensorInfo { dims: vec![u64::MAX, 2], ..t };
        assert_eq!(t.n_elements(), u64::MAX);
    }

    #[test]
    fn g7_non_ascii_shard_name_does_not_panic() {
        // multi-byte characters right where the 15-byte tail would be cut
        for stem in ["modèle-0001-of-00002", "ééééééééééééééééé", "x€€€€€-of-€€€€€", "a-00001-of-0000é"] {
            let r = split_pattern(stem);
            assert!(matches!(r, Ok(None)), "{stem}: {r:?}");
            let p = std::env::temp_dir().join(format!("{stem}.gguf"));
            assert_eq!(discover_shards(&p).unwrap(), vec![p.clone()]);
        }
        // non-ASCII base name with a valid pattern
        assert_eq!(split_pattern("modèle-00001-of-00002").unwrap(), Some(("modèle", 2)));
    }

    #[test]
    fn g7_shard_total_zero_is_rejected() {
        assert!(split_pattern("m-00001-of-00000").is_err());
        assert!(split_pattern("m-00000-of-00002").is_err());
        assert!(split_pattern("m-00003-of-00002").is_err());
        assert!(split_pattern("m-00001-of-10000").is_err());
        assert_eq!(split_pattern("m-00002-of-00002").unwrap(), Some(("m", 2)));
        let p = std::env::temp_dir().join("m-00001-of-00000.gguf");
        assert!(discover_shards(&p).is_err());
    }
}
