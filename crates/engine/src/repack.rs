//! Expert repack: one file next to the model where every routed expert is a single 4 KiB-aligned
//! record laid out exactly like an expert-cache slot (gate | up | down at the slot's regions).
//! The cache then loads an expert with ONE direct read straight into its slot instead of three
//! reads of the GGUF's per-matrix slices (fewer, larger requests -> more SSD throughput).
//!
//! File: [header, padded to 4 KiB] [layer records...]. The header carries a fingerprint of the
//! GGUF's expert tensors (names, types, shapes, offsets, shard sizes, and a sample of each
//! tensor's bytes); a file that doesn't match the model is ignored with a warning. It is written
//! as `<name>.tmp` and renamed when complete, so an interrupted repack never looks valid.

use crate::io::{AlignedBuf, ALIGN};
use anyhow::{bail, Context, Result};
use gguf::{Gguf, TensorInfo};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

const MAGIC: &[u8; 8] = b"GLMSSDX1";
const VERSION: u32 = 1;
const PARTS: [&str; 3] = ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"];
/// Experts assembled per write.
const BATCH: usize = 16;

/// Byte offsets of the gate / up / down regions inside a slot, and the slot size. Each region
/// has one spare page so an unaligned GGUF slice fits after widening the read to 4 KiB.
pub fn slot_layout(per: [u64; 3]) -> ([usize; 3], usize) {
    let mut regions = [0usize; 3];
    let mut at = 0usize;
    for i in 0..3 {
        regions[i] = at;
        at += (per[i] as usize).next_multiple_of(ALIGN as usize) + ALIGN as usize;
    }
    (regions, at)
}

/// Bytes of a packed record that hold data (the read length): up to the end of the down matrix.
pub fn record_len(per: [u64; 3]) -> usize {
    let (r, _) = slot_layout(per);
    r[2] + (per[2] as usize).next_multiple_of(ALIGN as usize)
}

/// Where the repacked experts of the model whose first shard is `shard1` live.
pub fn packed_path(shard1: &Path) -> PathBuf {
    let stem = shard1.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    // "Name-00001-of-00004" -> "Name"
    let base = match stem.rsplit_once("-of-") {
        Some((head, tail)) if tail.chars().all(|c| c.is_ascii_digit()) => match head.rsplit_once('-') {
            Some((name, n)) if n.chars().all(|c| c.is_ascii_digit()) => name.to_string(),
            _ => stem.clone(),
        },
        _ => stem.clone(),
    };
    shard1.with_file_name(format!("{base}.glm-ssd-experts"))
}

/// One layer's experts inside the packed file.
#[derive(Clone, Copy, Debug)]
pub struct PackedLayer {
    pub offset: u64, // file offset of expert 0's record
    pub stride: u64, // bytes per record (= slot size)
}

pub struct Packed {
    pub path: PathBuf,
    pub layers: HashMap<usize, PackedLayer>,
}

fn expert_layers(g: &Gguf) -> Result<Vec<(usize, [TensorInfo; 3])>> {
    let mut out = Vec::new();
    for il in 0.. {
        if g.tensor(&format!("blk.{il}.attn_norm.weight")).is_none() && g.tensor(&format!("blk.{il}.{}", PARTS[0])).is_none() {
            break;
        }
        let get = |n: &str| g.tensor(&format!("blk.{il}.{n}")).cloned();
        if let (Some(a), Some(b), Some(c)) = (get(PARTS[0]), get(PARTS[1]), get(PARTS[2])) {
            out.push((il, [a, b, c]));
        }
    }
    if out.is_empty() {
        bail!("the model has no routed-expert tensors");
    }
    Ok(out)
}

fn pers(t: &[TensorInfo; 3]) -> [u64; 3] {
    [0, 1, 2].map(|i| t[i].n_bytes / t[i].dims[2])
}

struct Fnv(u64);
impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn put(&mut self, b: &[u8]) {
        for &x in b {
            self.0 = (self.0 ^ x as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn u64(&mut self, v: u64) {
        self.put(&v.to_le_bytes());
    }
}

fn read_at(f: &File, off: u64, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(f, &mut buf[got..], off + got as u64)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(f, &mut buf[got..], off + got as u64)?;
        if n == 0 {
            break;
        }
        got += n;
    }
    Ok(got)
}

/// Identity of the GGUF's expert data: layout of every expert tensor, shard sizes, and the first
/// 4 KiB of each expert tensor (catches a different model with an identical layout).
fn fingerprint(g: &Gguf, layers: &[(usize, [TensorInfo; 3])]) -> Result<u64> {
    let mut h = Fnv::new();
    let files: Vec<File> = g.shards.iter().map(|p| File::open(p).with_context(|| format!("open {}", p.display()))).collect::<Result<_>>()?;
    for f in &files {
        h.u64(f.metadata()?.len());
    }
    let mut sample = vec![0u8; 4096];
    for (il, ts) in layers {
        h.u64(*il as u64);
        for t in ts {
            h.put(t.name.as_bytes());
            h.u64(t.ty.id() as u64);
            for d in &t.dims {
                h.u64(*d);
            }
            h.u64(t.shard as u64);
            h.u64(t.offset);
            h.u64(t.n_bytes);
            let n = read_at(&files[t.shard], t.offset, &mut sample[..(t.n_bytes as usize).min(4096)])?;
            h.put(&sample[..n]);
        }
    }
    Ok(h.0)
}

fn header_bytes(n_layers: usize) -> usize {
    (32 + n_layers * 48).next_multiple_of(ALIGN as usize)
}

/// Plan the packed file: per layer (layer, record offset, stride, pers).
fn plan(layers: &[(usize, [TensorInfo; 3])], n_expert: u64) -> (Vec<(usize, PackedLayer, [u64; 3])>, u64) {
    let mut at = header_bytes(layers.len()) as u64;
    let mut out = Vec::new();
    for (il, ts) in layers {
        let p = pers(ts);
        let (_, slot) = slot_layout(p);
        out.push((*il, PackedLayer { offset: at, stride: slot as u64 }, p));
        at += slot as u64 * n_expert;
    }
    (out, at)
}

fn encode_header(fp: u64, plan: &[(usize, PackedLayer, [u64; 3])], n_expert: u64) -> Vec<u8> {
    let mut h = Vec::with_capacity(header_bytes(plan.len()));
    h.extend_from_slice(MAGIC);
    h.extend_from_slice(&VERSION.to_le_bytes());
    h.extend_from_slice(&(plan.len() as u32).to_le_bytes());
    h.extend_from_slice(&fp.to_le_bytes());
    h.extend_from_slice(&n_expert.to_le_bytes());
    for (il, pl, p) in plan {
        h.extend_from_slice(&(*il as u32).to_le_bytes());
        h.extend_from_slice(&(n_expert as u32).to_le_bytes());
        h.extend_from_slice(&pl.offset.to_le_bytes());
        h.extend_from_slice(&pl.stride.to_le_bytes());
        for x in p {
            h.extend_from_slice(&x.to_le_bytes());
        }
    }
    h.resize(header_bytes(plan.len()), 0);
    h
}

fn n_expert_of(layers: &[(usize, [TensorInfo; 3])]) -> u64 {
    layers[0].1[0].dims[2]
}

impl Packed {
    /// The repacked experts for this model, if a valid file exists. A file made from a different
    /// model (or an older format) is reported and ignored.
    pub fn open(g: &Gguf) -> Result<Option<Packed>> {
        let path = packed_path(&g.shards[0]);
        if !path.exists() {
            return Ok(None);
        }
        let layers = expert_layers(g)?;
        let n_expert = n_expert_of(&layers);
        let (plan, total) = plan(&layers, n_expert);
        let want = encode_header(fingerprint(g, &layers)?, &plan, n_expert);
        let f = File::open(&path).with_context(|| format!("open {}", path.display()))?;
        let mut have = vec![0u8; want.len()];
        let n = read_at(&f, 0, &mut have)?;
        let len = f.metadata()?.len();
        if n != want.len() || have != want || len < total {
            eprintln!(
                "warning: {} doesn't match this model (made from another model or version); \
                 reading experts from the GGUF. Run `glm-ssd repack` to rebuild it.",
                path.display()
            );
            return Ok(None);
        }
        Ok(Some(Packed { path, layers: plan.into_iter().map(|(il, pl, _)| (il, pl)).collect() }))
    }
}

fn open_out(path: &Path) -> Result<File> {
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        const FILE_FLAG_WRITE_THROUGH: u32 = 0x8000_0000;
        o.custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH);
    }
    o.open(path).with_context(|| format!("create {}", path.display()))
}

fn write_at(f: &File, off: u64, buf: &[u8]) -> std::io::Result<()> {
    let mut done = 0;
    while done < buf.len() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_write(f, &buf[done..], off + done as u64)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::write_at(f, &buf[done..], off + done as u64)?;
        done += n;
    }
    Ok(())
}

/// Write the packed expert file for `g` (skips if an up-to-date one exists, unless `force`).
pub fn repack(g: &Gguf, force: bool) -> Result<()> {
    let path = packed_path(&g.shards[0]);
    if !force && Packed::open(g)?.is_some() {
        println!("{} is already up to date (use --force to rebuild it)", path.display());
        return Ok(());
    }
    let layers = expert_layers(g)?;
    let n_expert = n_expert_of(&layers);
    let (plan, total) = plan(&layers, n_expert);
    println!(
        "repacking {} layers x {n_expert} experts ({:.1} GB) -> {}",
        layers.len(),
        total as f64 / 1e9,
        path.display()
    );
    let fp = fingerprint(g, &layers)?;
    let tmp = path.with_extension("glm-ssd-experts.tmp");
    let out = open_out(&tmp)?;
    let files: Vec<File> = g.shards.iter().map(|p| crate::io::open_direct(p)).collect::<Result<_>>()?;

    // writer thread: overlaps writing one batch with reading the next
    let (tx, rx) = mpsc::sync_channel::<(u64, AlignedBuf, usize)>(2);
    let (back_tx, back_rx) = mpsc::channel::<AlignedBuf>();
    let writer = std::thread::spawn(move || -> Result<File> {
        for (off, buf, len) in rx {
            let mut buf = buf;
            write_at(&out, off, &buf.as_mut_slice()[..len])?;
            let _ = back_tx.send(buf);
        }
        Ok(out)
    });

    let max_slot = plan.iter().map(|p| p.1.stride as usize).max().unwrap();
    let max_per = plan.iter().flat_map(|p| p.2).max().unwrap() as usize;
    let mut spare: Vec<AlignedBuf> = (0..3).map(|_| AlignedBuf::new(BATCH * max_slot)).collect();
    let mut src = AlignedBuf::new(BATCH * max_per + 2 * ALIGN as usize);
    let t0 = Instant::now();
    let mut written = 0u64;
    for (li, ((il, pl, per), (_, ts))) in plan.iter().zip(&layers).enumerate() {
        let (regions, slot) = slot_layout(*per);
        let mut e0 = 0u64;
        while e0 < n_expert {
            let n = (n_expert - e0).min(BATCH as u64) as usize;
            let mut rec = match spare.pop() {
                Some(b) => b,
                None => back_rx.recv().context("writer thread stopped")?,
            };
            let dst = rec.as_mut_slice();
            dst[..n * slot].fill(0);
            for p in 0..3 {
                let t = &ts[p];
                let (a, alen, skip) = crate::io::window(t.offset + e0 * per[p], n as u64 * per[p]);
                let got = crate::experts::read_direct(&files[t.shard], a, &mut src.as_mut_slice()[..alen])
                    .with_context(|| format!("read {} of layer {il}", t.name))?;
                if (got as usize) < skip + n * per[p] as usize {
                    bail!("short read in {} (layer {il}): is the model complete?", t.name);
                }
                let s = src.as_mut_slice();
                for i in 0..n {
                    let from = skip + i * per[p] as usize;
                    let to = i * slot + regions[p];
                    dst[to..to + per[p] as usize].copy_from_slice(&s[from..from + per[p] as usize]);
                }
            }
            tx.send((pl.offset + e0 * pl.stride, rec, n * slot)).context("writer thread stopped")?;
            written += (n * slot) as u64;
            e0 += n as u64;
        }
        let secs = t0.elapsed().as_secs_f64();
        let rate = written as f64 / secs.max(1e-9);
        eprint!(
            "\r  layer {}/{}  {:.1}/{:.1} GB  {:.2} GB/s  {:.0}s left   ",
            li + 1,
            plan.len(),
            written as f64 / 1e9,
            total as f64 / 1e9,
            rate / 1e9,
            (total - written) as f64 / rate.max(1.0)
        );
    }
    drop(tx);
    let out = writer.join().map_err(|_| anyhow::anyhow!("writer thread panicked"))??;
    // header last: a file without it is never mistaken for a complete one
    let mut hdr = AlignedBuf::new(header_bytes(plan.len()));
    hdr.as_mut_slice().copy_from_slice(&encode_header(fp, &plan, n_expert));
    write_at(&out, 0, hdr.as_mut_slice())?;
    drop(out);
    if path.exists() {
        std::fs::remove_file(&path).with_context(|| format!("remove old {}", path.display()))?;
    }
    std::fs::rename(&tmp, &path).with_context(|| format!("rename {} -> {}", tmp.display(), path.display()))?;
    eprintln!();
    println!(
        "done in {:.0}s: {} ({:.1} GB). The engine uses it automatically; delete it to go back to \
         the GGUF, `glm-ssd repack` recreates it.",
        t0.elapsed().as_secs_f64(),
        path.display(),
        total as f64 / 1e9
    );
    Ok(())
}
