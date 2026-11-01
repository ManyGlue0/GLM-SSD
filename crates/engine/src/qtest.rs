//! `glm-ssd qtest`: cross-checks the CUDA quant kernels against the CPU reference (crates/quants).
//!
//! - random blocks (with sane f16 scales) for every supported type
//! - optionally real tensor rows from a downloaded model
//! - dequant must match the CPU exactly (up to float rounding); GEMV within a relative tolerance

use anyhow::{bail, Result};
use cuda::{bytes_of, bytes_of_mut, ops, DeviceBuffer, Stream};
use gguf::GgmlType;
use std::path::Path;

pub const TYPES: [GgmlType; 12] = [
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::Q8_0,
    GgmlType::Q2_K,
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::IQ2_XXS,
    GgmlType::IQ2_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ4_XS,
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn random_blocks(ty: GgmlType, n_elems: usize, rng: &mut Rng) -> Vec<u8> {
    let (be, bb) = ty.block();
    let n_blocks = n_elems / be as usize;
    let mut v: Vec<u8> = (0..n_blocks * bb as usize).map(|_| rng.next() as u8).collect();
    let halves = quants::half_offsets(ty);
    for blk in v.chunks_exact_mut(bb as usize) {
        match ty {
            GgmlType::F32 => blk.copy_from_slice(&(rng.f32() * 2.0 - 1.0).to_le_bytes()),
            GgmlType::F16 => blk.copy_from_slice(&quants::f32_to_f16(rng.f32() * 2.0 - 1.0).to_le_bytes()),
            _ => {
                for &o in halves {
                    let mag = 1e-3 + rng.f32() * 5e-2;
                    let val = if rng.next() & 1 == 0 { mag } else { -mag };
                    blk[o..o + 2].copy_from_slice(&quants::f32_to_f16(val).to_le_bytes());
                }
            }
        }
    }
    v
}

fn max_rel_err(got: &[f32], want: &[f32]) -> (f32, usize) {
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-20);
    let mut worst = (0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let e = (g - w).abs() / scale;
        if !(e <= worst.0) {
            worst = (e, i);
        }
    }
    worst
}

/// Check one quantized matrix (rows x k, raw bytes) on the GPU against the CPU reference.
fn check_matrix(ty: GgmlType, raw: &[u8], rows: usize, k: usize, label: &str, rng: &mut Rng, s: &Stream) -> Result<bool> {
    let n = rows * k;
    let mut cpu = vec![0f32; n];
    quants::dequantize(ty, raw, &mut cpu)?;

    let dw = DeviceBuffer::new(raw.len())?;
    dw.upload(0, raw, s)?;
    let dy = DeviceBuffer::new(n * 4)?;
    ops::dequant_f32(ty.id(), dw.ptr(), n, dy.ptr() as *mut f32, s)?;
    let mut gpu = vec![0f32; n];
    dy.download(0, bytes_of_mut(&mut gpu), s)?;
    s.sync()?;
    let (de, di) = max_rel_err(&gpu, &cpu);

    // GEMV with 1..=8 vectors (test 1, 3 and 8)
    let row_stride = ty.row_bytes(k as u64) as usize;
    let mut worst_g = 0f32;
    for nv in [1usize, 3, 8] {
        let x: Vec<f32> = (0..nv * k).map(|_| rng.f32() * 2.0 - 1.0).collect();
        let dx = DeviceBuffer::new(x.len() * 4)?;
        dx.upload(0, bytes_of(&x), s)?;
        let dyy = DeviceBuffer::new(nv * rows * 4)?;
        ops::gemv(ty.id(), dw.ptr(), row_stride, k, rows, dx.ptr() as *const f32, k, dyy.ptr() as *mut f32, rows, nv, false, s)?;
        let mut out = vec![0f32; nv * rows];
        dyy.download(0, bytes_of_mut(&mut out), s)?;
        s.sync()?;
        let mut want = vec![0f32; nv * rows];
        for b in 0..nv {
            for r in 0..rows {
                let w = &cpu[r * k..(r + 1) * k];
                let xv = &x[b * k..(b + 1) * k];
                want[b * rows + r] = w.iter().zip(xv).map(|(a, b)| (*a as f64 * *b as f64)).sum::<f64>() as f32;
            }
        }
        worst_g = worst_g.max(max_rel_err(&out, &want).0);
    }
    let absmax = cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
    let distinct = {
        let mut v: Vec<u32> = cpu.iter().map(|f| f.to_bits()).collect();
        v.sort_unstable();
        v.dedup();
        v.len()
    };
    // a degenerate decode (all zeros / constant) would trivially "match"
    let ok = de < 1e-6 && worst_g < 1e-4 && absmax > 0.0 && distinct > 16;
    println!(
        "  {:<8} {:<34} dequant err {:.1e}  gemv err {:.1e}  |w|max {:.3e}  {:>7} distinct  {}",
        format!("{ty:?}"), label, de, worst_g, absmax, distinct, if ok { "OK" } else { "FAIL" }
    );
    if !ok && de >= 1e-6 {
        println!("      first mismatch: gpu {} vs cpu {}", gpu[di], cpu[di]);
    }
    Ok(ok)
}

pub fn run(model: Option<&Path>) -> Result<()> {
    let s = Stream::new()?;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut all_ok = true;

    println!("random blocks (rows=37, k=4096):");
    for ty in TYPES {
        let (rows, k) = (37, 4096);
        let raw = random_blocks(ty, rows * k, &mut rng);
        all_ok &= check_matrix(ty, &raw, rows, k, "random", &mut rng, &s)?;
    }

    if let Some(m) = model {
        println!("real tensors from {}:", m.display());
        // a single complete shard is enough (lets us test while other shards still download)
        let g = gguf::Gguf::open_shard(m)?;
        let mut seen = std::collections::HashSet::new();
        for t in &g.tensors {
            if !quants::is_supported(t.ty) || t.dims.len() < 2 || seen.contains(&t.ty) {
                continue;
            }
            seen.insert(t.ty);
            let k = t.dims[0] as usize;
            let rows = (t.dims[1] as usize).min(64);
            let bytes = t.ty.row_bytes(k as u64) as usize * rows;
            let mut raw = vec![0u8; bytes];
            read_at(&g.shards[t.shard], t.offset, &mut raw)?;
            all_ok &= check_matrix(t.ty, &raw, rows, k, &t.name, &mut rng, &s)?;
        }
    }
    if !all_ok {
        bail!("quant kernel mismatches found");
    }
    println!("all quant kernels match the CPU reference");
    Ok(())
}

fn read_at(path: &Path, off: u64, buf: &mut [u8]) -> Result<()> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(path)?;
    f.seek(SeekFrom::Start(off))?;
    f.read_exact(buf)?;
    Ok(())
}

/// GEMV throughput per type: rows x 4096 matrix, effective weight bandwidth in GB/s.
pub fn bench() -> Result<()> {
    let s = Stream::new()?;
    let mut rng = Rng(42);
    let (rows, k) = (16384usize, 4096usize);
    println!("GEMV {rows}x{k} (weights streamed from VRAM; RTX 3090 peak ~936 GB/s)");
    println!("  {:<8} {:>9} {:>12} {:>12} {:>12}", "type", "MiB", "B=1 GB/s", "B=4 GB/s", "B=8 GB/s");
    for ty in TYPES {
        let raw = random_blocks(ty, rows * k, &mut rng);
        let dw = DeviceBuffer::new(raw.len())?;
        dw.upload(0, &raw, &s)?;
        let x: Vec<f32> = (0..8 * k).map(|_| rng.f32()).collect();
        let dx = DeviceBuffer::new(x.len() * 4)?;
        dx.upload(0, bytes_of(&x), &s)?;
        let dy = DeviceBuffer::new(8 * rows * 4)?;
        let rs = ty.row_bytes(k as u64) as usize;
        let mut cols = Vec::new();
        for nv in [1usize, 4, 8] {
            let run = |s: &Stream| ops::gemv(ty.id(), dw.ptr(), rs, k, rows, dx.ptr() as *const f32, k, dy.ptr() as *mut f32, rows, nv, false, s);
            for _ in 0..3 {
                run(&s)?;
            }
            let (a, b) = (cuda::Event::timed()?, cuda::Event::timed()?);
            let iters = 50;
            a.record(&s)?;
            for _ in 0..iters {
                run(&s)?;
            }
            b.record(&s)?;
            b.sync()?;
            let ms = a.elapsed_ms(&b)? / iters as f32;
            cols.push(raw.len() as f64 / (ms as f64 * 1e-3) / 1e9);
        }
        println!("  {:<8} {:>9.1} {:>12.0} {:>12.0} {:>12.0}", format!("{ty:?}"), raw.len() as f64 / 1048576.0, cols[0], cols[1], cols[2]);
    }
    Ok(())
}
