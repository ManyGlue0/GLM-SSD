//! Resident weights: every non-expert tensor uploaded once into a single VRAM allocation.

use crate::plan::layer_of;
use anyhow::{bail, Context, Result};
use cuda::{DeviceBuffer, Stream};
use gguf::{GgmlType, Gguf, TensorInfo};
use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::time::Instant;

/// A tensor resident in device memory.
#[derive(Clone, Copy, Debug)]
pub struct DTensor {
    pub ptr: *mut c_void,
    pub ty: GgmlType,
    /// ggml order: d0 = row length (K), d1 = rows, d2 = batch (e.g. heads)
    pub d0: usize,
    pub d1: usize,
    pub d2: usize,
    pub row_bytes: usize,
}
unsafe impl Send for DTensor {}
unsafe impl Sync for DTensor {}

impl DTensor {
    pub fn f32(&self) -> *const f32 {
        debug_assert_eq!(self.ty, GgmlType::F32);
        self.ptr as *const f32
    }
    /// bytes of one [d0 x d1] matrix (for batched/per-head tensors)
    pub fn matrix_bytes(&self) -> usize {
        self.row_bytes * self.d1
    }
}

pub fn is_expert_tensor(name: &str) -> bool {
    name.ends_with("_exps.weight")
}

pub struct Weights {
    _buf: DeviceBuffer,
    map: HashMap<String, DTensor>,
    pub bytes: usize,
}

pub fn read_at(f: &File, mut off: u64, mut buf: &mut [u8]) -> std::io::Result<()> {
    while !buf.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(f, buf, off)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(f, buf, off)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "short read"));
        }
        buf = &mut buf[n..];
        off += n as u64;
    }
    Ok(())
}

impl Weights {
    /// Upload all resident tensors of layers < n_layers (plus globals except token_embd).
    pub fn load(g: &Gguf, files: &[File], n_layers: usize, s: &Stream) -> Result<Self> {
        let keep = |t: &TensorInfo| {
            !is_expert_tensor(&t.name) && t.name != "token_embd.weight" && layer_of(&t.name).is_none_or(|l| l < n_layers)
        };
        const ALIGN: usize = 256;
        let mut total = 0usize;
        for t in g.tensors.iter().filter(|t| keep(t)) {
            total = total.next_multiple_of(ALIGN) + t.n_bytes as usize;
        }
        let buf = DeviceBuffer::new(total).context("allocating resident weights in VRAM")?;
        let t0 = Instant::now();
        let mut map = HashMap::new();
        let mut off = 0usize;
        let mut host = Vec::new();
        for t in g.tensors.iter().filter(|t| keep(t)) {
            off = off.next_multiple_of(ALIGN);
            host.resize(t.n_bytes as usize, 0u8);
            read_at(&files[t.shard], t.offset, &mut host).with_context(|| format!("reading {}", t.name))?;
            buf.upload(off, &host, s)?;
            s.sync()?; // `host` is reused for the next tensor
            map.insert(
                t.name.clone(),
                DTensor {
                    ptr: unsafe { (buf.ptr() as *mut u8).add(off) } as *mut c_void,
                    ty: t.ty,
                    d0: t.dims[0] as usize,
                    d1: t.dims.get(1).copied().unwrap_or(1) as usize,
                    d2: t.dims.get(2).copied().unwrap_or(1) as usize,
                    row_bytes: t.ty.row_bytes(t.dims[0]) as usize,
                },
            );
            off += t.n_bytes as usize;
        }
        eprintln!(
            "loaded {} resident tensors, {:.2} GiB in {:.1}s",
            map.len(),
            total as f64 / (1u64 << 30) as f64,
            t0.elapsed().as_secs_f64()
        );
        Ok(Self { _buf: buf, map, bytes: total })
    }

    pub fn get(&self, name: &str) -> Result<DTensor> {
        match self.map.get(name) {
            Some(t) => Ok(*t),
            None => bail!("missing tensor {name}"),
        }
    }
    pub fn blk(&self, il: usize, suffix: &str) -> Result<DTensor> {
        self.get(&format!("blk.{il}.{suffix}"))
    }
}

/// Read a small F32 tensor to the host.
pub fn read_f32(g: &Gguf, files: &[File], name: &str) -> Result<Vec<f32>> {
    let t = g.tensor(name).with_context(|| format!("missing tensor {name}"))?;
    if t.ty != GgmlType::F32 {
        bail!("{name} is {:?}, expected F32", t.ty);
    }
    let mut raw = vec![0u8; t.n_bytes as usize];
    read_at(&files[t.shard], t.offset, &mut raw)?;
    Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

/// Per-layer expert tensor locations (for direct reads of single experts).
#[derive(Clone, Debug)]
pub struct ExpertTensors {
    pub gate: TensorInfo,
    pub up: TensorInfo,
    pub down: TensorInfo,
}

impl ExpertTensors {
    pub fn for_layer(g: &Gguf, il: usize) -> Result<Self> {
        let get = |n: &str| g.tensor(&format!("blk.{il}.{n}")).cloned().with_context(|| format!("blk.{il}.{n} missing"));
        Ok(Self { gate: get("ffn_gate_exps.weight")?, up: get("ffn_up_exps.weight")?, down: get("ffn_down_exps.weight")? })
    }
    pub fn unit_bytes(&self) -> usize {
        [&self.gate, &self.up, &self.down].iter().map(|t| (t.n_bytes / t.dims[2]) as usize).sum()
    }
}
