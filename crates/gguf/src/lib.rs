//! Minimal GGUF v3 reader.
//!
//! Only the header (metadata + tensor infos) is parsed; tensor data is never touched here.
//! Split models (`*-00001-of-0000N.gguf`) are opened as one logical model: every tensor gets
//! an absolute location `(shard, offset, n_bytes)` that the engine uses for direct I/O.

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

mod types;
pub use types::GgmlType;

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
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    /// Bytes of one row (dims[0] elements).
    pub fn row_bytes(&self) -> u64 {
        self.ty.row_bytes(self.dims[0])
    }
    /// For 3-D expert tensors `[in, out, n_expert]`: byte range of expert `e`'s slice.
    pub fn slice_range(&self, e: u64) -> (u64, u64) {
        let per = self.n_bytes / self.dims[self.dims.len() - 1];
        (self.offset + e * per, per)
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
        }
        if let Some(n) = kv.get("split.tensors.count").and_then(Value::as_u64) {
            if n as usize != tensors.len() {
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

/// `foo-00002-of-00004.gguf` -> all 4 shard paths (that must exist). Otherwise just `path`.
fn discover_shards(path: &Path) -> Result<Vec<PathBuf>> {
    let name = path.file_name().and_then(|n| n.to_str()).context("bad path")?;
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    // look for "-NNNNN-of-MMMMM"
    if stem.len() > 15 {
        let tail = &stem[stem.len() - 15..];
        let b = tail.as_bytes();
        if b[0] == b'-' && &tail[6..10] == "-of-" && tail[1..6].bytes().all(|c| c.is_ascii_digit()) && tail[10..].bytes().all(|c| c.is_ascii_digit()) {
            let base = &stem[..stem.len() - 15];
            let total: usize = tail[10..].parse()?;
            let dir = path.parent().unwrap_or(Path::new("."));
            let v: Vec<PathBuf> = (1..=total).map(|i| dir.join(format!("{base}-{i:05}-of-{total:05}.gguf"))).collect();
            for p in &v {
                if !p.exists() {
                    bail!("missing shard {}", p.display());
                }
            }
            return Ok(v);
        }
    }
    Ok(vec![path.to_path_buf()])
}

struct Rd<R: Read> {
    r: R,
    pos: u64,
}
impl<R: Read> Rd<R> {
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
        if n > (1 << 30) {
            bail!("string too long ({n})");
        }
        let mut v = vec![0u8; n as usize];
        self.r.read_exact(&mut v)?;
        self.pos += n;
        Ok(String::from_utf8_lossy(&v).into_owned())
    }
    fn value(&mut self, t: u32) -> Result<Value> {
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
                let et = self.u32()?;
                let n = self.u64()?;
                let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
                for _ in 0..n {
                    v.push(self.value(et)?);
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
    let mut r = Rd { r: BufReader::with_capacity(1 << 20, f), pos: 0 };
    if &r.bytes::<4>()? != b"GGUF" {
        bail!("not a GGUF file");
    }
    let ver = r.u32()?;
    if ver != 3 {
        bail!("unsupported GGUF version {ver}");
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    let mut kv = HashMap::new();
    for _ in 0..n_kv {
        let k = r.string()?;
        let t = r.u32()?;
        kv.insert(k, r.value(t)?);
    }
    let align = kv.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
    let mut raw = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()?;
        let dims = (0..nd).map(|_| r.u64()).collect::<Result<Vec<_>>>()?;
        let ty = GgmlType::from_id(r.u32()?).with_context(|| format!("tensor {name}"))?;
        let off = r.u64()?;
        raw.push((name, dims, ty, off));
    }
    let data_start = r.pos.div_ceil(align) * align;
    let tensors = raw
        .into_iter()
        .map(|(name, dims, ty, off)| {
            let n_bytes = ty.row_bytes(dims[0]) * dims[1..].iter().product::<u64>();
            let offset = data_start + off;
            if check_len && offset + n_bytes > file_len {
                bail!("tensor {name} extends past end of {}", path.display());
            }
            Ok(TensorInfo { name, dims, ty, shard, offset, n_bytes })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((kv, tensors))
}
