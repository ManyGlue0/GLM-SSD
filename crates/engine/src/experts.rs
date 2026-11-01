//! Routed-expert streaming: a two-tier cache (VRAM slots + pinned RAM slots) fed by direct
//! (unbuffered) SSD reads on a small pool of reader threads.
//!
//! Life of an expert: SSD --(reader thread, NO_BUFFERING)--> pinned RAM slot --(copy stream)-->
//! VRAM slot --> GEMV/GEMM on the compute stream. Both tiers evict by a decayed use frequency
//! (seeded from the learned hot-expert profile). RAM prefers to evict experts that also sit in
//! VRAM, so the two tiers stay mostly exclusive without device-to-host copies.
//!
//! Synchronisation: every RAM slot remembers the event of its last H2D copy (it may only be
//! overwritten once that copy finished), every VRAM slot remembers the event of its last use on
//! the compute stream (the copy stream waits for it before overwriting the slot).

use crate::io::ALIGN;
use anyhow::{anyhow, bail, Context, Result};
use cuda::{DeviceBuffer, Event, PinnedBuffer, Stream};
use gguf::{Gguf, TensorInfo};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

pub type Key = (u16, u16); // (layer, expert)

/// Per-token decay of the use frequency (half-life ~350 tokens).
const DECAY: f32 = 0.998;
/// Minimum slots per size class in each tier (enough to pipeline one layer).
const MIN_SLOTS: usize = 24;
/// Slots allocated per pinned / device chunk.
const CHUNK_SLOTS: usize = 32;
/// Demand reads in flight per layer fetch (the SSD saturates at low queue depth).
const MAX_INFLIGHT: usize = 12;
const MAX_PREFETCH_INFLIGHT: usize = 12;

#[derive(Clone, Debug)]
struct Part {
    info: TensorInfo,
    per: u64,      // bytes of one expert slice
    region: usize, // offset of this part's region inside a slot
}

#[derive(Clone, Debug)]
struct LayerLayout {
    parts: [Part; 3],
    class: usize,
}

/// Device pointers of one expert's gate / up / down matrices.
#[derive(Clone, Copy, Debug)]
pub struct ExpertPtrs {
    pub gate: *const u8,
    pub up: *const u8,
    pub down: *const u8,
}

// ---- reader threads ------------------------------------------------------------------------------

struct Req {
    id: u64,
    reads: Vec<(usize, u64, usize, usize)>, // shard, aligned file offset, aligned length, dst address
}
#[derive(Default)]
struct IoQueue {
    hi: VecDeque<Req>,
    lo: VecDeque<Req>,
    quit: bool,
}
struct IoPool {
    q: Arc<(Mutex<IoQueue>, Condvar)>,
    done: mpsc::Receiver<(u64, Result<u64, String>)>,
    threads: Vec<std::thread::JoinHandle<()>>,
    next_id: u64,
}

fn read_direct(f: &File, off: u64, buf: &mut [u8]) -> std::io::Result<u64> {
    let mut got = 0usize;
    while got < buf.len() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_read(f, &mut buf[got..], off + got as u64)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::read_at(f, &mut buf[got..], off + got as u64)?;
        if n == 0 {
            break; // EOF inside the aligned tail window: the wanted bytes are already in
        }
        got += n;
    }
    Ok(got as u64)
}

impl IoPool {
    fn new(shards: &[PathBuf], n_threads: usize, direct: bool) -> Result<Self> {
        let q: Arc<(Mutex<IoQueue>, Condvar)> = Arc::default();
        let (tx, rx) = mpsc::channel();
        let mut threads = Vec::new();
        for ti in 0..n_threads {
            let files = shards
                .iter()
                .map(|p| if direct { crate::io::open_direct(p) } else { File::open(p).map_err(Into::into) })
                .collect::<Result<Vec<_>>>()?;
            let q = q.clone();
            let tx = tx.clone();
            threads.push(
                std::thread::Builder::new()
                    .name(format!("expert-io-{ti}"))
                    .spawn(move || loop {
                        let req = {
                            let (m, cv) = &*q;
                            let mut g = m.lock().unwrap();
                            loop {
                                if g.quit {
                                    return;
                                }
                                if let Some(r) = g.hi.pop_front().or_else(|| g.lo.pop_front()) {
                                    break r;
                                }
                                g = cv.wait(g).unwrap();
                            }
                        };
                        let mut res = Ok(0u64);
                        for &(shard, off, len, dst) in &req.reads {
                            let buf = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, len) };
                            match read_direct(&files[shard], off, buf) {
                                Ok(n) => res = res.map(|t| t + n),
                                Err(e) => {
                                    res = Err(format!("read shard {shard} @ {off}: {e}"));
                                    break;
                                }
                            }
                        }
                        if tx.send((req.id, res)).is_err() {
                            return;
                        }
                    })?,
            );
        }
        Ok(Self { q, done: rx, threads, next_id: 1 })
    }
    fn submit(&mut self, reads: Vec<(usize, u64, usize, usize)>, hi: bool) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let (m, cv) = &*self.q;
        let mut g = m.lock().unwrap();
        if hi { g.hi.push_back(Req { id, reads }) } else { g.lo.push_back(Req { id, reads }) }
        cv.notify_one();
        id
    }
    /// Move a queued low-priority request to the high queue (a prefetch became a demand).
    fn promote(&self, id: u64) {
        let (m, _) = &*self.q;
        let mut g = m.lock().unwrap();
        if let Some(i) = g.lo.iter().position(|r| r.id == id) {
            let r = g.lo.remove(i).unwrap();
            g.hi.push_front(r);
        }
    }
}
impl Drop for IoPool {
    fn drop(&mut self) {
        {
            let (m, cv) = &*self.q;
            m.lock().unwrap().quit = true;
            cv.notify_all();
        }
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

// ---- slots ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RState {
    Free,
    Loading(u64),
    Ready,
}

struct RSlot {
    key: Option<Key>,
    state: RState,
    ptr: *mut u8,
    h2d: Event,
    prefetched: bool,
}
struct VSlot {
    key: Option<Key>,
    ptr: *mut u8,
    copy_done: Event,
    last_use: Event,
}

struct Class {
    slot_bytes: usize,
    ram: Vec<RSlot>,
    vram: Vec<VSlot>,
    _pinned: Vec<PinnedBuffer>,
    _device: Vec<DeviceBuffer>,
}

#[derive(Default, Clone, Debug)]
pub struct Stats {
    pub uses: u64,
    pub hit_vram: u64,
    pub hit_ram: u64,
    pub miss: u64,
    pub read_bytes: u64,
    pub read_time: Duration, // main-thread time spent waiting for reads
    pub prefetch_issued: u64,
    pub prefetch_useful: u64,
}

/// What one MoE layer still has to get onto the GPU.
pub struct Fetch {
    layer: usize,
    vram: VecDeque<u16>,
    ram: VecDeque<u16>,
    to_read: VecDeque<u16>,
    waiting: Vec<(u16, u64)>, // expert, request id
}

pub struct ExpertCache {
    layouts: HashMap<usize, LayerLayout>,
    classes: Vec<Class>,
    ram_map: HashMap<Key, usize>,
    vram_map: HashMap<Key, usize>,
    freq: HashMap<Key, (f32, u64)>, // decayed frequency, last use (token clock)
    prior: HashMap<Key, f32>,
    protect: HashSet<Key>,
    loading: HashMap<u64, (usize, usize)>, // request id -> (class, ram slot)
    prefetch_ids: HashSet<u64>,
    io: IoPool,
    now: u64,
    pub stats: Stats,
    /// lifetime use counts (the learned hot-expert profile)
    pub counts: HashMap<Key, u64>,
    profile_path: Option<PathBuf>,
    profile_tokens: u64,
}
unsafe impl Send for ExpertCache {}

pub struct CacheConfig {
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    pub io_threads: usize,
    pub direct_io: bool,
    pub profile: Option<PathBuf>,
}

impl ExpertCache {
    pub fn new(g: &Gguf, moe_layers: &[usize], n_expert: usize, cfg: &CacheConfig) -> Result<Self> {
        // size classes = distinct (gate, up, down) slice sizes
        let mut layouts = HashMap::new();
        let mut class_sizes: Vec<[u64; 3]> = Vec::new();
        let mut class_layers: Vec<usize> = Vec::new();
        for &il in moe_layers {
            let get = |n: &str| g.tensor(&format!("blk.{il}.{n}")).cloned().with_context(|| format!("blk.{il}.{n} missing"));
            let infos = [get("ffn_gate_exps.weight")?, get("ffn_up_exps.weight")?, get("ffn_down_exps.weight")?];
            let pers = [0, 1, 2].map(|i| infos[i].n_bytes / infos[i].dims[2]);
            let class = match class_sizes.iter().position(|c| *c == pers) {
                Some(c) => c,
                None => {
                    class_sizes.push(pers);
                    class_layers.push(0);
                    class_sizes.len() - 1
                }
            };
            class_layers[class] += 1;
            let mut region = 0usize;
            let parts = [0, 1, 2].map(|i| {
                let p = Part { info: infos[i].clone(), per: pers[i], region };
                region += (pers[i] as usize).next_multiple_of(ALIGN as usize) + ALIGN as usize;
                p
            });
            layouts.insert(il, LayerLayout { parts, class });
        }
        let slot_bytes: Vec<usize> = class_sizes
            .iter()
            .map(|c| c.iter().map(|p| (*p as usize).next_multiple_of(ALIGN as usize) + ALIGN as usize).sum())
            .collect();
        // split the budgets across classes in proportion to their total expert bytes
        let total: f64 = (0..class_sizes.len()).map(|c| (class_layers[c] * n_expert * slot_bytes[c]) as f64).sum();
        let mut classes = Vec::new();
        for c in 0..class_sizes.len() {
            let share = (class_layers[c] * n_expert * slot_bytes[c]) as f64 / total;
            let cap = class_layers[c] * n_expert;
            let n_v = (((cfg.vram_bytes as f64 * share) as usize) / slot_bytes[c]).clamp(MIN_SLOTS, cap);
            let n_r = (((cfg.ram_bytes as f64 * share) as usize) / slot_bytes[c]).clamp(MIN_SLOTS, cap);
            classes.push(Self::alloc_class(slot_bytes[c], n_v, n_r)?);
        }
        let io = IoPool::new(&g.shards, cfg.io_threads.max(1), cfg.direct_io)?;
        let mut me = Self {
            layouts,
            classes,
            ram_map: HashMap::new(),
            vram_map: HashMap::new(),
            freq: HashMap::new(),
            prior: HashMap::new(),
            protect: HashSet::new(),
            loading: HashMap::new(),
            prefetch_ids: HashSet::new(),
            io,
            now: 0,
            stats: Stats::default(),
            counts: HashMap::new(),
            profile_path: cfg.profile.clone(),
            profile_tokens: 0,
        };
        if let Some(p) = &cfg.profile {
            if p.exists() {
                me.load_profile(p)?;
            }
        }
        Ok(me)
    }

    fn alloc_class(slot_bytes: usize, n_v: usize, n_r: usize) -> Result<Class> {
        let mut c = Class { slot_bytes, ram: Vec::new(), vram: Vec::new(), _pinned: Vec::new(), _device: Vec::new() };
        let mut left = n_r;
        while left > 0 {
            let n = left.min(CHUNK_SLOTS);
            let buf = match PinnedBuffer::new(n * slot_bytes) {
                Ok(b) => b,
                Err(e) if c.ram.len() >= MIN_SLOTS => {
                    eprintln!("warning: pinned RAM allocation stopped at {} slots: {e:#}", c.ram.len());
                    break;
                }
                Err(e) => return Err(e.context("allocating pinned expert slots")),
            };
            for i in 0..n {
                c.ram.push(RSlot {
                    key: None,
                    state: RState::Free,
                    ptr: unsafe { buf.ptr().add(i * slot_bytes) },
                    h2d: Event::new()?,
                    prefetched: false,
                });
            }
            c._pinned.push(buf);
            left -= n;
        }
        let mut left = n_v;
        while left > 0 {
            let n = left.min(CHUNK_SLOTS * 4);
            let buf = match DeviceBuffer::new(n * slot_bytes) {
                Ok(b) => b,
                Err(e) if c.vram.len() >= MIN_SLOTS => {
                    eprintln!("warning: VRAM expert allocation stopped at {} slots: {e:#}", c.vram.len());
                    break;
                }
                Err(e) => return Err(e.context("allocating VRAM expert slots")),
            };
            for i in 0..n {
                c.vram.push(VSlot {
                    key: None,
                    ptr: unsafe { (buf.ptr() as *mut u8).add(i * slot_bytes) },
                    copy_done: Event::new()?,
                    last_use: Event::new()?,
                });
            }
            c._device.push(buf);
            left -= n;
        }
        Ok(c)
    }

    pub fn capacity(&self) -> (usize, usize, u64, u64) {
        let v: usize = self.classes.iter().map(|c| c.vram.len()).sum();
        let r: usize = self.classes.iter().map(|c| c.ram.len()).sum();
        let vb: u64 = self.classes.iter().map(|c| (c.vram.len() * c.slot_bytes) as u64).sum();
        let rb: u64 = self.classes.iter().map(|c| (c.ram.len() * c.slot_bytes) as u64).sum();
        (v, r, vb, rb)
    }

    /// Advance the token clock (call once per generated / processed token).
    pub fn tick(&mut self, n: u64) {
        self.now += n;
        self.profile_tokens += n;
    }

    fn score(&self, k: Key) -> f32 {
        let (f, last) = self.freq.get(&k).copied().unwrap_or((0.0, 0));
        let p = self.prior.get(&k).copied().unwrap_or(0.0);
        f * DECAY.powi((self.now - last).min(1 << 20) as i32) + p
    }
    fn touch(&mut self, k: Key) {
        let now = self.now;
        let e = self.freq.entry(k).or_insert((0.0, now));
        e.0 = e.0 * DECAY.powi((now - e.1).min(1 << 20) as i32) + 1.0;
        e.1 = now;
        *self.counts.entry(k).or_insert(0) += 1;
    }

    fn reads_for(&self, layer: usize, e: u16, dst: *mut u8) -> Vec<(usize, u64, usize, usize)> {
        let lay = &self.layouts[&layer];
        lay.parts
            .iter()
            .map(|p| {
                let off = p.info.offset + e as u64 * p.per;
                let a = off / ALIGN * ALIGN;
                let len = (off + p.per).div_ceil(ALIGN) * ALIGN - a;
                (p.info.shard, a, len as usize, dst as usize + p.region)
            })
            .collect()
    }
    fn ptrs(&self, layer: usize, e: u16, base: *const u8) -> ExpertPtrs {
        let lay = &self.layouts[&layer];
        let p = |i: usize| {
            let part = &lay.parts[i];
            let off = part.info.offset + e as u64 * part.per;
            unsafe { base.add(part.region + (off % ALIGN) as usize) }
        };
        ExpertPtrs { gate: p(0), up: p(1), down: p(2) }
    }

    /// Pick a RAM slot to (re)use in class `c`; None if every slot is busy.
    fn evict_ram(&mut self, c: usize) -> Option<usize> {
        let mut best: Option<(usize, bool, f32)> = None;
        let mut blocked: Option<usize> = None; // evictable once its H2D copy finishes
        for (i, s) in self.classes[c].ram.iter().enumerate() {
            match s.state {
                RState::Loading(_) => continue,
                RState::Free => return Some(i),
                RState::Ready => {}
            }
            let k = s.key.unwrap();
            if self.protect.contains(&k) && !self.vram_map.contains_key(&k) {
                continue;
            }
            if !s.h2d.done() {
                blocked.get_or_insert(i);
                continue;
            }
            let dup = self.vram_map.contains_key(&k);
            let sc = self.score(k);
            let better = match best {
                None => true,
                Some((_, bdup, bsc)) => (dup && !bdup) || (dup == bdup && sc < bsc),
            };
            if better {
                best = Some((i, dup, sc));
            }
        }
        let i = match best {
            Some((i, _, _)) => i,
            None => {
                // every candidate is still being copied to the GPU: wait for one copy
                let i = blocked?;
                self.classes[c].ram[i].h2d.sync().ok()?;
                i
            }
        };
        let k = self.classes[c].ram[i].key.take().unwrap();
        self.ram_map.remove(&k);
        self.classes[c].ram[i].state = RState::Free;
        Some(i)
    }
    fn evict_vram(&mut self, c: usize) -> Option<usize> {
        let mut best: Option<(usize, f32)> = None;
        for (i, s) in self.classes[c].vram.iter().enumerate() {
            let Some(k) = s.key else { return Some(i) };
            if self.protect.contains(&k) {
                continue;
            }
            let sc = self.score(k);
            if best.is_none_or(|(_, b)| sc < b) {
                best = Some((i, sc));
            }
        }
        let (i, _) = best?;
        let k = self.classes[c].vram[i].key.take().unwrap();
        self.vram_map.remove(&k);
        Some(i)
    }

    fn submit_read(&mut self, layer: usize, e: u16, hi: bool) -> Option<u64> {
        let c = self.layouts[&layer].class;
        let slot = self.evict_ram(c)?;
        let k = (layer as u16, e);
        let dst = self.classes[c].ram[slot].ptr;
        let reads = self.reads_for(layer, e, dst);
        let id = self.io.submit(reads, hi);
        let s = &mut self.classes[c].ram[slot];
        s.key = Some(k);
        s.state = RState::Loading(id);
        s.prefetched = !hi;
        self.ram_map.insert(k, slot);
        self.loading.insert(id, (c, slot));
        if !hi {
            self.prefetch_ids.insert(id);
            self.stats.prefetch_issued += 1;
        }
        Some(id)
    }

    /// Handle one finished read (blocking if `block`). Returns the request id.
    fn complete_one(&mut self, block: bool) -> Result<Option<u64>> {
        let msg = if block {
            let t0 = Instant::now();
            let m = self.io.done.recv().map_err(|_| anyhow!("expert reader threads died"))?;
            self.stats.read_time += t0.elapsed();
            m
        } else {
            match self.io.done.try_recv() {
                Ok(m) => m,
                Err(_) => return Ok(None),
            }
        };
        let (id, res) = msg;
        let bytes = res.map_err(|e| anyhow!(e))?;
        self.stats.read_bytes += bytes;
        self.prefetch_ids.remove(&id);
        if let Some((c, slot)) = self.loading.remove(&id) {
            let s = &mut self.classes[c].ram[slot];
            if s.state == RState::Loading(id) {
                s.state = RState::Ready;
            }
        }
        Ok(Some(id))
    }

    /// Start getting `experts` of `layer` onto the GPU.
    pub fn fetch_layer(&mut self, layer: usize, experts: &[u16]) -> Result<Fetch> {
        while self.complete_one(false)?.is_some() {}
        let mut f = Fetch { layer, vram: VecDeque::new(), ram: VecDeque::new(), to_read: VecDeque::new(), waiting: Vec::new() };
        self.protect.clear();
        let c = self.layouts.get(&layer).with_context(|| format!("layer {layer} has no routed experts"))?.class;
        for &e in experts {
            let k = (layer as u16, e);
            self.touch(k);
            self.protect.insert(k);
            self.stats.uses += 1;
            if self.vram_map.contains_key(&k) {
                self.stats.hit_vram += 1;
                f.vram.push_back(e);
            } else if let Some(&slot) = self.ram_map.get(&k) {
                let s = &mut self.classes[c].ram[slot];
                if s.prefetched {
                    s.prefetched = false;
                    self.stats.prefetch_useful += 1;
                }
                match s.state {
                    RState::Ready => {
                        self.stats.hit_ram += 1;
                        f.ram.push_back(e);
                    }
                    RState::Loading(id) => {
                        self.stats.hit_ram += 1;
                        self.io.promote(id);
                        f.waiting.push((e, id));
                    }
                    RState::Free => unreachable!(),
                }
            } else {
                self.stats.miss += 1;
                f.to_read.push_back(e);
            }
        }
        self.pump(&mut f);
        Ok(f)
    }

    fn pump(&mut self, f: &mut Fetch) {
        while f.waiting.len() < MAX_INFLIGHT {
            let Some(&e) = f.to_read.front() else { break };
            match self.submit_read(f.layer, e, true) {
                Some(id) => {
                    f.to_read.pop_front();
                    f.waiting.push((e, id));
                }
                None => break, // every RAM slot busy: retry after something completes
            }
        }
    }

    /// Copy a RAM-resident expert into a VRAM slot (copy stream) and make `compute` wait for it.
    fn upload(&mut self, layer: usize, e: u16, copy: &Stream, compute: &Stream) -> Result<ExpertPtrs> {
        let k = (layer as u16, e);
        let c = self.layouts[&layer].class;
        let rslot = self.ram_map[&k];
        let vslot = self.evict_vram(c).context("no evictable VRAM expert slot")?;
        let cls = &self.classes[c];
        let (src, dst, n) = (cls.ram[rslot].ptr, cls.vram[vslot].ptr, cls.slot_bytes);
        copy.wait(&cls.vram[vslot].last_use)?;
        unsafe { copy.h2d(dst as _, src, n)? };
        cls.vram[vslot].copy_done.record(copy)?;
        cls.ram[rslot].h2d.record(copy)?;
        compute.wait(&cls.vram[vslot].copy_done)?;
        self.classes[c].vram[vslot].key = Some(k);
        self.vram_map.insert(k, vslot);
        Ok(self.ptrs(layer, e, dst))
    }

    /// Next expert of the fetch whose weights are (or will be, in stream order) on the GPU.
    pub fn next(&mut self, f: &mut Fetch, copy: &Stream, compute: &Stream) -> Result<Option<(u16, ExpertPtrs)>> {
        let layer = f.layer;
        let c = self.layouts[&layer].class;
        if let Some(e) = f.vram.pop_front() {
            let k = (layer as u16, e);
            let v = self.vram_map[&k];
            let s = &self.classes[c].vram[v];
            compute.wait(&s.copy_done)?;
            self.protect.remove(&k);
            return Ok(Some((e, self.ptrs(layer, e, s.ptr))));
        }
        loop {
            // collect finished reads without blocking
            while let Some(id) = self.complete_one(false)? {
                if let Some(i) = f.waiting.iter().position(|w| w.1 == id) {
                    let (e, _) = f.waiting.swap_remove(i);
                    f.ram.push_back(e);
                }
            }
            if let Some(e) = f.ram.pop_front() {
                let p = self.upload(layer, e, copy, compute)?;
                self.protect.remove(&(layer as u16, e));
                self.pump(f);
                return Ok(Some((e, p)));
            }
            if f.waiting.is_empty() {
                if f.to_read.is_empty() {
                    return Ok(None);
                }
                self.pump(f);
                if f.waiting.is_empty() {
                    bail!("expert cache deadlock: no RAM slot can be freed (RAM budget too small?)");
                }
            }
            // wait for any read; ours or a prefetch
            let id = self.complete_one(true)?.unwrap();
            if let Some(i) = f.waiting.iter().position(|w| w.1 == id) {
                let (e, _) = f.waiting.swap_remove(i);
                f.ram.push_back(e);
            }
            // a request we promoted may have been submitted as a prefetch (its id is in `waiting`)
        }
    }

    /// The compute stream finished enqueuing work on `e`; its VRAM slot may be reused after that.
    pub fn done_with(&mut self, layer: usize, e: u16, compute: &Stream) -> Result<()> {
        let k = (layer as u16, e);
        if let Some(&v) = self.vram_map.get(&k) {
            let c = self.layouts[&layer].class;
            self.classes[c].vram[v].last_use.record(compute)?;
        }
        Ok(())
    }

    /// Speculatively start reading experts that are likely needed soon (low priority).
    pub fn prefetch(&mut self, layer: usize, experts: &[u16]) {
        if !self.layouts.contains_key(&layer) {
            return;
        }
        for &e in experts {
            if self.prefetch_ids.len() >= MAX_PREFETCH_INFLIGHT {
                break;
            }
            let k = (layer as u16, e);
            if self.vram_map.contains_key(&k) || self.ram_map.contains_key(&k) {
                continue;
            }
            if self.submit_read(layer, e, false).is_none() {
                break;
            }
        }
    }

    /// Fill the caches with the hottest experts of the profile (blocking, with progress).
    pub fn preload(&mut self, copy: &Stream, compute: &Stream) -> Result<()> {
        if self.counts.is_empty() {
            return Ok(());
        }
        let mut keys: Vec<(Key, u64)> = self.counts.iter().map(|(k, v)| (*k, *v)).collect();
        keys.sort_by(|a, b| b.1.cmp(&a.1));
        let t0 = Instant::now();
        let (nv, nr, _, _) = self.capacity();
        let mut per_class_v = vec![0usize; self.classes.len()];
        let mut per_class_r = vec![0usize; self.classes.len()];
        let mut want_v = Vec::new();
        let mut want_r = Vec::new();
        for (k, _) in &keys {
            let Some(lay) = self.layouts.get(&(k.0 as usize)) else { continue };
            let c = lay.class;
            if per_class_v[c] < self.classes[c].vram.len() - MIN_SLOTS / 2 {
                per_class_v[c] += 1;
                want_v.push(*k);
            } else if per_class_r[c] < self.classes[c].ram.len() - MIN_SLOTS / 2 {
                per_class_r[c] += 1;
                want_r.push(*k);
            }
        }
        let bytes0 = self.stats.read_bytes;
        eprint!("preloading {} experts into VRAM and {} into RAM (of {nv}/{nr} slots)...", want_v.len(), want_r.len());
        // VRAM set: read through RAM slots and upload as they arrive
        for chunk in want_v.chunks(MIN_SLOTS / 2) {
            for k in chunk {
                let c = self.layouts[&(k.0 as usize)].class;
                let slot = self.evict_ram(c).context("preload: no RAM slot")?;
                let dst = self.classes[c].ram[slot].ptr;
                let reads = self.reads_for(k.0 as usize, k.1, dst);
                let id = self.io.submit(reads, true);
                let s = &mut self.classes[c].ram[slot];
                s.key = Some(*k);
                s.state = RState::Loading(id);
                s.prefetched = false;
                self.ram_map.insert(*k, slot);
                self.loading.insert(id, (c, slot));
                self.protect.insert(*k);
            }
            for _ in chunk {
                self.complete_one(true)?;
            }
            for k in chunk {
                self.upload(k.0 as usize, k.1, copy, compute)?;
            }
            copy.sync()?;
            self.protect.clear();
        }
        for chunk in want_r.chunks(MAX_INFLIGHT) {
            for k in chunk {
                self.submit_read(k.0 as usize, k.1, true);
            }
            for _ in chunk {
                self.complete_one(true)?;
            }
        }
        let gb = (self.stats.read_bytes - bytes0) as f64 / 1e9;
        let dt = t0.elapsed().as_secs_f64();
        eprintln!(" {gb:.1} GB in {dt:.1}s ({:.2} GB/s)", gb / dt);
        // preloading is not a use: give the profile priors instead of counts to the scores
        self.stats = Stats::default();
        Ok(())
    }

    // ---- hot-expert profile -----------------------------------------------------------------------
    fn load_profile(&mut self, p: &Path) -> Result<()> {
        let text = std::fs::read_to_string(p).with_context(|| format!("reading profile {}", p.display()))?;
        let mut tokens = 0u64;
        for line in text.lines() {
            let mut it = line.split_whitespace();
            match (it.next(), it.next(), it.next()) {
                (Some("tokens"), Some(n), None) => tokens = n.parse().unwrap_or(0),
                (Some(l), Some(e), Some(n)) => {
                    if let (Ok(l), Ok(e), Ok(n)) = (l.parse::<u16>(), e.parse::<u16>(), n.parse::<u64>()) {
                        self.counts.insert((l, e), n);
                    }
                }
                _ => {}
            }
        }
        self.profile_tokens = tokens;
        // prior score: expected uses over ~a half-life worth of tokens, relative to this expert's share
        let per_layer: HashMap<u16, u64> = self.counts.iter().fold(HashMap::new(), |mut m, (k, v)| {
            *m.entry(k.0).or_insert(0) += v;
            m
        });
        for (k, v) in &self.counts {
            let tot = per_layer[&k.0].max(1) as f32;
            self.prior.insert(*k, 8.0 * *v as f32 / tot * 50.0);
        }
        Ok(())
    }

    pub fn save_profile(&self) -> Result<()> {
        let Some(p) = &self.profile_path else { return Ok(()) };
        let mut keys: Vec<_> = self.counts.iter().collect();
        keys.sort();
        let mut s = format!("tokens {}\n", self.profile_tokens);
        for ((l, e), n) in keys {
            s += &format!("{l} {e} {n}\n");
        }
        let tmp = p.with_extension("tmp");
        std::fs::write(&tmp, s)?;
        std::fs::rename(&tmp, p)?;
        Ok(())
    }

    pub fn summary(&self) -> String {
        let s = &self.stats;
        let u = s.uses.max(1) as f64;
        format!(
            "experts: {} uses | VRAM hit {:.0}% | RAM hit {:.0}% | SSD {:.0}% | read {:.2} GB, waited {:.1}s | prefetch {}/{} useful",
            s.uses,
            100.0 * s.hit_vram as f64 / u,
            100.0 * s.hit_ram as f64 / u,
            100.0 * s.miss as f64 / u,
            s.read_bytes as f64 / 1e9,
            s.read_time.as_secs_f64(),
            s.prefetch_useful,
            s.prefetch_issued
        )
    }
}
