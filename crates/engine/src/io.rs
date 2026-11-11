//! Direct (unbuffered) expert I/O.
//!
//! Experts are read with FILE_FLAG_NO_BUFFERING so they bypass the Windows file cache (the
//! engine manages its own RAM cache; double caching would waste memory). Unbuffered reads need
//! offset, length and buffer address aligned to the sector size, so every read is widened to
//! 4 KiB boundaries and the caller gets the offset of the wanted bytes inside the buffer.

use anyhow::{Context, Result};
use std::fs::{File, OpenOptions};
use std::path::Path;

pub const ALIGN: u64 = 4096;

/// Physical RAM available right now (Windows: GlobalMemoryStatusEx).
pub fn available_ram() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        len: u32,
        load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page: u64,
        avail_page: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_ext_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(m: *mut MemoryStatusEx) -> i32;
    }
    let mut m: MemoryStatusEx = unsafe { std::mem::zeroed() };
    m.len = std::mem::size_of::<MemoryStatusEx>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some(m.avail_phys)
}

pub fn open_direct(path: &Path) -> Result<File> {
    let mut o = OpenOptions::new();
    o.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
        const FILE_FLAG_RANDOM_ACCESS: u32 = 0x1000_0000;
        o.custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS);
    }
    o.open(path).with_context(|| format!("open (direct) {}", path.display()))
}

/// Aligned read window covering [off, off+len): (aligned_off, aligned_len, skip)
pub fn window(off: u64, len: u64) -> (u64, usize, usize) {
    let a = off / ALIGN * ALIGN;
    let end = (off + len).div_ceil(ALIGN) * ALIGN;
    (a, (end - a) as usize, (off - a) as usize)
}

/// Page-aligned heap buffer (for direct reads without CUDA pinned memory).
pub struct AlignedBuf {
    raw: Vec<u8>,
    off: usize,
    len: usize,
}
impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let raw = vec![0u8; len + ALIGN as usize];
        let off = raw.as_ptr().align_offset(ALIGN as usize);
        Self { raw, off, len }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.raw[self.off..self.off + self.len]
    }
}

/// Benchmark random expert-sized direct reads at several thread counts.
pub fn bench(path: &Path, unit: usize, seconds: f64) -> Result<()> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{Duration, Instant};
    let size = std::fs::metadata(path)?.len();
    println!("direct reads from {} ({:.2} GB), {:.2} MiB per read", path.display(), size as f64 / 1e9, unit as f64 / 1048576.0);
    for threads in [1usize, 2, 4, 8, 16, 32] {
        let bytes = AtomicU64::new(0);
        let reads = AtomicU64::new(0);
        let t0 = Instant::now();
        std::thread::scope(|s| -> Result<()> {
            let mut hs = Vec::new();
            for ti in 0..threads {
                let (bytes, reads) = (&bytes, &reads);
                hs.push(s.spawn(move || -> Result<()> {
                    let f = open_direct(path)?;
                    let mut buf = AlignedBuf::new(unit + 2 * ALIGN as usize);
                    let mut rng = 0x9E37_79B9_7F4A_7C15u64 ^ (ti as u64 * 0x1234_5678_9ABC);
                    while t0.elapsed() < Duration::from_secs_f64(seconds) {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        let off = rng % (size - unit as u64);
                        let (a, l, _) = window(off, unit as u64);
                        let l = l.min((size - a) as usize).div_ceil(ALIGN as usize) * ALIGN as usize;
                        crate::model::read_at(&f, a, &mut buf.as_mut_slice()[..l]).ok();
                        bytes.fetch_add(l as u64, Ordering::Relaxed);
                        reads.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(())
                }));
            }
            for h in hs {
                h.join().map_err(|_| anyhow::anyhow!("thread panicked"))??;
            }
            Ok(())
        })?;
        let dt = t0.elapsed().as_secs_f64();
        let gbs = bytes.load(Ordering::Relaxed) as f64 / dt / 1e9;
        let n = reads.load(Ordering::Relaxed);
        println!(
            "  {threads:>2} threads: {gbs:>5.2} GB/s  {:>7.0} reads/s  avg latency {:>6.2} ms",
            n as f64 / dt,
            dt * 1e3 * threads as f64 / n.max(1) as f64
        );
    }
    Ok(())
}
