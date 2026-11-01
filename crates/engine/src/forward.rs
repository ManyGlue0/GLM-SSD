//! Batched forward pass of glm5-next (GLM-5.3-Flash) on the GPU.
//!
//! `forward` runs T tokens of one sequence through the trunk: T = 1 for decode, chunks of up to
//! `chunk` tokens for prefill, T = 2 for MTP verification. Non-expert weights are resident in
//! VRAM; routed experts come from the two-tier `ExpertCache`. Per layer:
//!
//!   mHC pre -> RMSNorm -> KDA (recurrent scan) | MLA+DSA (flash attention over selected keys)
//!   -> mHC post -> mHC pre -> RMSNorm -> dense FFN | MoE (shared expert + top-8 routed) -> mHC post
//!
//! Small batches (T <= 8) use the quantized GEMV kernels, larger ones the tensor-core GEMM.

use crate::config::{AttnKind, Config};
use crate::experts::{CacheConfig, ExpertCache, ExpertPtrs};
use crate::model::{read_at, read_f32, DTensor, Weights};
use anyhow::{bail, Context, Result};
use cuda::{bytes_of, bytes_of_mut, check, ffi, ops, DeviceBuffer, Stream};
use gguf::{Gguf, TensorInfo};
use std::fs::File;
use std::time::{Duration, Instant};

macro_rules! ck {
    ($e:expr) => {
        check(unsafe { $e })?
    };
}

/// Queries per sub-batch for the MLA attention / indexer scoring (bounds scratch memory).
const TQ: usize = 256;

struct KdaW {
    q: DTensor,
    k: DTensor,
    v: DTensor,
    conv_q: DTensor,
    conv_k: DTensor,
    conv_v: DTensor,
    f_a: DTensor,
    f_b: DTensor,
    dt_b: DTensor,
    a: DTensor,
    beta: DTensor,
    g_a: DTensor,
    g_b: DTensor,
    o_norm: DTensor,
    o: DTensor,
}

struct MlaW {
    q_a: DTensor,
    q_a_norm: DTensor,
    q_b: DTensor,
    kv_a: DTensor,
    kv_a_norm: DTensor,
    k_b: DTensor,
    v_b: DTensor,
    o: DTensor,
    iq_b: DTensor,
    ik: DTensor,
    ik_norm_w: DTensor,
    ik_norm_b: DTensor,
    iproj: DTensor,
    igate: DTensor,
    iape: DTensor,
}

enum Attn {
    Kda(Box<KdaW>),
    Mla(Box<MlaW>),
}

enum Ffn {
    Dense { gate: DTensor, up: DTensor, down: DTensor },
    Moe { router: DTensor, bias: Vec<f32>, sh_gate: DTensor, sh_up: DTensor, sh_down: DTensor },
}

struct Hc {
    fn_: DTensor,
    scale: DTensor,
    base: DTensor,
}

struct Layer {
    attn_norm: DTensor,
    ffn_norm: DTensor,
    hc: Option<(Hc, Hc)>, // (attn, ffn); the MTP layer has a plain residual
    attn: Attn,
    ffn: Ffn,
    clamp_exp: f32,
    clamp_shexp: f32,
}

struct MtpHead {
    eh: DTensor,
    enorm: DTensor,
    hnorm: DTensor,
    head_norm: DTensor,
}

enum LayerState {
    Kda { s: DeviceBuffer, conv: DeviceBuffer, s_save: Option<DeviceBuffer>, conv_save: Option<DeviceBuffer> },
    /// latent: f16 [ctx][R], or int8 [ctx][R] with `scales` f16 [ctx][R/32] (Q8 cache)
    Mla { latent: DeviceBuffer, scales: Option<DeviceBuffer>, pooled: DeviceBuffer, ring_k: DeviceBuffer, ring_g: DeviceBuffer },
}

/// Which rows of a forward call produce logits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rows {
    None,
    Last,
    All,
}

#[derive(Default, Debug, Clone)]
pub struct Timings {
    pub tokens: usize,
    pub total: Duration,
    pub moe: Duration,
}

pub struct EngineOpts {
    pub ctx: usize,
    pub chunk: usize,
    pub mtp: bool,
    /// store MLA latents as Q8 (int8 + f16 scale per 32) instead of f16
    pub kv_q8: bool,
    /// decode: swap an expert that would need an SSD read for a cached one whose routing
    /// score is within this margin (0 = exact routing)
    pub cache_aware: f32,
    pub cache: CacheConfig,
}

/// Host copy of the sequence-dependent state that cannot be recomputed from the caches
/// (prefix caching): KDA states + conv states + indexer rings, at position `pos`.
pub struct Snapshot {
    pub pos: usize,
    mtp_pos: usize,
    pend_n: usize,
    data: Vec<Vec<u8>>,
}
impl Snapshot {
    pub fn bytes(&self) -> usize {
        self.data.iter().map(Vec::len).sum()
    }
}

pub struct Engine {
    pub cfg: Config,
    files: Vec<File>,
    tok_embd: TensorInfo,
    _w: Weights,
    layers: Vec<Layer>, // trunk layers, then the MTP layer if enabled
    mtp: Option<MtpHead>,
    states: Vec<LayerState>,
    output_norm: DTensor,
    output: DTensor,
    pub experts: ExpertCache,
    exp_dims: Vec<Option<[DTensor; 3]>>,
    s: Stream,
    copy: Stream,
    pub ctx: usize,
    pub chunk: usize,
    pub pos: usize,
    pub mtp_pos: usize,
    pub trace: bool,
    pub timings: Timings,
    pub cache_aware: f32,
    pub swaps: u64,
    // per-call position of the first row (for MLA / indexer), and the layer being run
    p0: usize,

    // device scratch, sized for `chunk` tokens
    streams: DeviceBuffer,
    flat: DeviceBuffer,
    x: DeviceBuffer,
    xn: DeviceBuffer,
    out: DeviceBuffer,
    a: DeviceBuffer,
    b: DeviceBuffer,
    c: DeviceBuffer,
    d: DeviceBuffer,
    s1: DeviceBuffer,
    s2: DeviceBuffer,
    s3: DeviceBuffer,
    mixes: DeviceBuffer,
    hcw: DeviceBuffer,
    qabs: DeviceBuffer,
    olat: DeviceBuffer,
    part: DeviceBuffer,
    sel: DeviceBuffer,
    cnt: DeviceBuffer,
    scores: DeviceBuffer,
    router: DeviceBuffer,
    moe_acc: DeviceBuffer,
    xg: DeviceBuffer,
    eg: DeviceBuffer,
    eu: DeviceBuffer,
    ey: DeviceBuffer,
    tok_idx: DeviceBuffer,
    tok_w: DeviceBuffer,
    logits: DeviceBuffer,
    /// normed final hidden (MTP input) of the logit rows: [8][n_embd]
    pub hid: DeviceBuffer,
    /// MTP rows waiting for their next token: hidden [2][n_embd]
    pend: DeviceBuffer,
    pend_n: usize,
    hall: DeviceBuffer, // all rows' hidden during MTP-fed prefill [chunk][n_embd]
    sel_stride: usize,
}

fn fbuf(n: usize, s: &Stream) -> Result<DeviceBuffer> {
    let b = DeviceBuffer::new(n * 4)?;
    b.zero(s)?;
    Ok(b)
}

fn moe_layers(cfg: &Config, with_mtp: bool) -> Vec<usize> {
    let n = if with_mtp { cfg.n_layer_all() } else { cfg.n_layer };
    (0..n).filter(|&l| cfg.is_moe(l)).collect()
}

impl Engine {
    pub fn new(g: &Gguf, cfg: Config, o: EngineOpts) -> Result<Self> {
        let s = Stream::new()?;
        let copy = Stream::new()?;
        let free0 = cuda::mem_info()?.0;
        let ctx = o.ctx.next_multiple_of(cfg.idx_kpool);
        let chunk = o.chunk.clamp(1, 4096);
        let with_mtp = o.mtp && cfg.n_layer_nextn > 0;
        let n_all = if with_mtp { cfg.n_layer + 1 } else { cfg.n_layer };
        let files = g.shards.iter().map(File::open).collect::<std::io::Result<Vec<_>>>()?;
        let w = Weights::load(g, &files, n_all, &s)?;
        let d_inner = cfg.n_head * cfg.kda_head_dim;
        let mut layers = Vec::with_capacity(n_all);
        let mut states = Vec::with_capacity(n_all);

        for il in 0..n_all {
            let b = |n: &str| w.blk(il, n);
            let hc = |p: &str| -> Result<Hc> {
                Ok(Hc {
                    fn_: b(&format!("hc_{p}_fn.weight"))?,
                    scale: b(&format!("hc_{p}_scale.weight"))?,
                    base: b(&format!("hc_{p}_base.weight"))?,
                })
            };
            let attn = match cfg.attn[il] {
                AttnKind::Kda => {
                    let sn = cfg.n_head * cfg.kda_head_dim * cfg.kda_head_dim;
                    let cn = 3 * d_inner * (cfg.kda_conv - 1);
                    states.push(LayerState::Kda {
                        s: fbuf(sn, &s)?,
                        conv: fbuf(cn, &s)?,
                        s_save: if with_mtp { Some(fbuf(sn, &s)?) } else { None },
                        conv_save: if with_mtp { Some(fbuf(cn, &s)?) } else { None },
                    });
                    Attn::Kda(Box::new(KdaW {
                        q: b("attn_q.weight")?,
                        k: b("attn_k.weight")?,
                        v: b("attn_v.weight")?,
                        conv_q: b("ssm_conv1d_q.weight")?,
                        conv_k: b("ssm_conv1d_k.weight")?,
                        conv_v: b("ssm_conv1d_v.weight")?,
                        f_a: b("ssm_f_a.weight")?,
                        f_b: b("ssm_f_b.weight")?,
                        dt_b: b("ssm_dt.bias")?,
                        a: b("ssm_a")?,
                        beta: b("ssm_beta.weight")?,
                        g_a: b("ssm_g_a.weight")?,
                        g_b: b("ssm_g_b.weight")?,
                        o_norm: b("ssm_norm.weight")?,
                        o: b("attn_output.weight")?,
                    }))
                }
                AttnKind::Mla => {
                    let r = cfg.kv_lora_rank;
                    let latent = DeviceBuffer::new(ctx * r * if o.kv_q8 { 1 } else { 2 }).context("MLA latent cache")?;
                    latent.zero(&s)?;
                    let scales = if o.kv_q8 {
                        let b = DeviceBuffer::new(ctx * (r / 32) * 2).context("MLA latent scales")?;
                        b.zero(&s)?;
                        Some(b)
                    } else {
                        None
                    };
                    states.push(LayerState::Mla {
                        latent,
                        scales,
                        pooled: fbuf((ctx / cfg.idx_kpool + 1) * cfg.idx_head_dim, &s)?,
                        ring_k: fbuf(cfg.idx_kpool * cfg.idx_head_dim, &s)?,
                        ring_g: fbuf(cfg.idx_kpool * cfg.idx_head_dim, &s)?,
                    });
                    Attn::Mla(Box::new(MlaW {
                        q_a: b("attn_q_a.weight")?,
                        q_a_norm: b("attn_q_a_norm.weight")?,
                        q_b: b("attn_q_b.weight")?,
                        kv_a: b("attn_kv_a_mqa.weight")?,
                        kv_a_norm: b("attn_kv_a_norm.weight")?,
                        k_b: b("attn_k_b.weight")?,
                        v_b: b("attn_v_b.weight")?,
                        o: b("attn_output.weight")?,
                        iq_b: b("indexer.attn_q_b.weight")?,
                        ik: b("indexer.attn_k.weight")?,
                        ik_norm_w: b("indexer.k_norm.weight")?,
                        ik_norm_b: b("indexer.k_norm.bias")?,
                        iproj: b("indexer.proj.weight")?,
                        igate: b("indexer_compressor_gate.weight")?,
                        iape: b("indexer_compressor_ape.weight")?,
                    }))
                }
            };
            let ffn = if cfg.is_moe(il) {
                Ffn::Moe {
                    router: b("ffn_gate_inp.weight")?,
                    bias: read_f32(g, &files, &format!("blk.{il}.exp_probs_b.bias"))?,
                    sh_gate: b("ffn_gate_shexp.weight")?,
                    sh_up: b("ffn_up_shexp.weight")?,
                    sh_down: b("ffn_down_shexp.weight")?,
                }
            } else {
                Ffn::Dense { gate: b("ffn_gate.weight")?, up: b("ffn_up.weight")?, down: b("ffn_down.weight")? }
            };
            layers.push(Layer {
                attn_norm: b("attn_norm.weight")?,
                ffn_norm: b("ffn_norm.weight")?,
                hc: if il < cfg.n_layer { Some((hc("attn")?, hc("ffn")?)) } else { None },
                attn,
                ffn,
                clamp_exp: cfg.swiglu_clamp_exp[il],
                clamp_shexp: cfg.swiglu_clamp_shexp[il],
            });
        }
        let mtp = if with_mtp {
            let il = cfg.n_layer;
            Some(MtpHead {
                eh: w.blk(il, "nextn.eh_proj.weight")?,
                enorm: w.blk(il, "nextn.enorm.weight")?,
                hnorm: w.blk(il, "nextn.hnorm.weight")?,
                head_norm: w.blk(il, "nextn.shared_head_norm.weight").or_else(|_| w.get("output_norm.weight"))?,
            })
        } else {
            None
        };

        let n = cfg.n_embd;
        let t = chunk;
        let tq = t.min(TQ);
        let wide = 16384usize.max(cfg.n_ff).max(cfg.n_head * cfg.mla_head_k).max(2 * d_inner);
        let sel_stride = cfg.idx_top_k + cfg.idx_kpool;
        let tok_embd = g.tensor("token_embd.weight").context("token_embd")?.clone();
        let mut exp_dims = vec![None; n_all];
        for il in moe_layers(&cfg, with_mtp) {
            let d = |nm: &str| -> Result<DTensor> {
                let ti = g.tensor(&format!("blk.{il}.{nm}")).with_context(|| format!("blk.{il}.{nm}"))?;
                Ok(DTensor {
                    ptr: std::ptr::null_mut(),
                    ty: ti.ty,
                    d0: ti.dims[0] as usize,
                    d1: ti.dims[1] as usize,
                    d2: 1,
                    row_bytes: ti.ty.row_bytes(ti.dims[0]) as usize,
                })
            };
            exp_dims[il] = Some([d("ffn_gate_exps.weight")?, d("ffn_up_exps.weight")?, d("ffn_down_exps.weight")?]);
        }
        let hr = cfg.n_head * cfg.kv_lora_rank;
        let eng = Self {
            output_norm: w.get("output_norm.weight")?,
            output: w.get("output.weight")?,
            streams: fbuf(t * cfg.hc * n, &s)?,
            flat: fbuf(t * cfg.hc * n, &s)?,
            x: fbuf(t * n, &s)?,
            xn: fbuf(t * n, &s)?,
            out: fbuf(t * n, &s)?,
            a: fbuf(t * wide, &s)?,
            b: fbuf(t * wide, &s)?,
            c: fbuf(t * wide, &s)?,
            d: fbuf(t * wide, &s)?,
            s1: fbuf(t * 2048, &s)?,
            s2: fbuf(t * 2048, &s)?,
            s3: fbuf(t * 2048, &s)?,
            mixes: fbuf(t * 24, &s)?,
            hcw: fbuf(t * 24, &s)?,
            qabs: fbuf(tq * hr, &s)?,
            olat: fbuf(tq * hr, &s)?,
            part: fbuf(128 * (16 * cfg.kv_lora_rank + 32), &s)?,
            sel: DeviceBuffer::new(t * sel_stride * 4)?,
            cnt: DeviceBuffer::new(t * 4)?,
            scores: fbuf(tq * (ctx / cfg.idx_kpool + 1), &s)?,
            router: fbuf(t * cfg.n_expert, &s)?,
            moe_acc: fbuf(t * n, &s)?,
            xg: fbuf(t * n, &s)?,
            eg: fbuf(t * cfg.n_ff_exp, &s)?,
            eu: fbuf(t * cfg.n_ff_exp, &s)?,
            ey: fbuf(t * n, &s)?,
            tok_idx: DeviceBuffer::new(t * cfg.n_expert_used * 4)?,
            tok_w: fbuf(t * cfg.n_expert_used, &s)?,
            logits: fbuf(8 * cfg.n_vocab, &s)?,
            hid: fbuf(8 * n, &s)?,
            pend: fbuf(2 * n, &s)?,
            pend_n: 0,
            hall: fbuf(if with_mtp { (t + 1) * n } else { 1 }, &s)?,
            sel_stride,
            tok_embd,
            files,
            _w: w,
            layers,
            mtp,
            states,
            exp_dims,
            s,
            copy,
            ctx,
            chunk,
            pos: 0,
            mtp_pos: 0,
            trace: false,
            timings: Timings::default(),
            cache_aware: o.cache_aware,
            swaps: 0,
            p0: 0,
            experts: {
                // whatever the budget leaves after weights, caches and scratch goes to experts
                check(unsafe { ffi::gs_device_sync() })?;
                let free1 = cuda::mem_info()?.0;
                let used = free0.saturating_sub(free1);
                let margin = 768u64 << 20;
                let vram = o.cache.vram_bytes.saturating_sub(used + margin).min(free1.saturating_sub(margin));
                let cc = CacheConfig { vram_bytes: vram, ..o.cache };
                ExpertCache::new(g, &moe_layers(&cfg, with_mtp), cfg.n_expert, &cc)?
            },
            cfg,
        };
        eng.s.sync()?;
        Ok(eng)
    }

    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Warm the expert caches from the hot-expert profile.
    pub fn preload(&mut self) -> Result<()> {
        self.experts.preload(&self.copy, &self.s)?;
        self.s.sync()
    }

    /// Forget the sequence.
    pub fn reset(&mut self) -> Result<()> {
        for st in &self.states {
            if let LayerState::Kda { s, conv, .. } = st {
                s.zero(&self.s)?;
                conv.zero(&self.s)?;
            }
        }
        self.pos = 0;
        self.mtp_pos = 0;
        self.pend_n = 0;
        self.s.sync()
    }

    // ---- prefix-cache snapshots ---------------------------------------------------------------------
    pub fn snapshot(&self) -> Result<Snapshot> {
        self.s.sync()?;
        let mut data = Vec::new();
        let mut grab = |b: &DeviceBuffer| -> Result<()> {
            let mut v = vec![0u8; b.len()];
            b.download(0, &mut v, &self.s)?;
            data.push(v);
            Ok(())
        };
        for st in &self.states {
            match st {
                LayerState::Kda { s, conv, .. } => {
                    grab(s)?;
                    grab(conv)?;
                }
                LayerState::Mla { ring_k, ring_g, .. } => {
                    grab(ring_k)?;
                    grab(ring_g)?;
                }
            }
        }
        let mut pend = vec![0u8; self.pend.len()];
        self.pend.download(0, &mut pend, &self.s)?;
        data.push(pend);
        self.s.sync()?;
        Ok(Snapshot { pos: self.pos, mtp_pos: self.mtp_pos, pend_n: self.pend_n, data })
    }
    pub fn restore(&mut self, snap: &Snapshot) -> Result<()> {
        let mut it = snap.data.iter();
        for st in &self.states {
            let bufs: [&DeviceBuffer; 2] = match st {
                LayerState::Kda { s, conv, .. } => [s, conv],
                LayerState::Mla { ring_k, ring_g, .. } => [ring_k, ring_g],
            };
            for b in bufs {
                b.upload(0, it.next().context("snapshot layout mismatch")?, &self.s)?;
            }
        }
        self.pend.upload(0, it.next().context("snapshot layout mismatch")?, &self.s)?;
        self.s.sync()?;
        self.pos = snap.pos;
        self.mtp_pos = snap.mtp_pos;
        self.pend_n = snap.pend_n;
        Ok(())
    }

    // ---- helpers ------------------------------------------------------------------------------------
    /// y[t] (+)= W x[t] for t < n_tok, rows of x `xs` floats apart, rows of y `ys` floats apart.
    #[allow(clippy::too_many_arguments)]
    fn mm_s(&self, w: &DTensor, x: *const f32, xs: usize, y: *mut f32, ys: usize, t: usize, acc: bool) -> Result<()> {
        if t <= 8 {
            ops::gemv(w.ty.id(), w.ptr, w.row_bytes, w.d0, w.d1, x, xs, y, ys, t, acc, &self.s)
        } else {
            ck!(ffi::gs_gemm(
                w.ty.id() as i32, w.ptr, w.row_bytes as i64, w.d0 as i32, w.d1 as i32, x, xs as i64, y, ys as i64, t as i32,
                acc as i32, 1, 0, 0, 0, self.s.raw()
            ));
            Ok(())
        }
    }
    fn mm(&self, w: &DTensor, x: *const f32, y: *mut f32, t: usize) -> Result<()> {
        self.mm_s(w, x, w.d0, y, w.d1, t, false)
    }
    /// Per-head matrices (3-D weight [K, rows, heads]): y[t][h] = W[h] x[t][h].
    #[allow(clippy::too_many_arguments)]
    fn mm_heads(&self, w: &DTensor, x: *const f32, xs: usize, xbs: usize, y: *mut f32, ys: usize, ybs: usize, t: usize) -> Result<()> {
        let heads = w.d2 as i32;
        let wbs = w.matrix_bytes() as i64;
        if t <= 8 {
            for tt in 0..t {
                ck!(ffi::gs_gemv_batched(
                    w.ty.id() as i32, w.ptr, w.row_bytes as i64, w.d0 as i32, w.d1 as i32, x.wrapping_add(tt * xs),
                    y.wrapping_add(tt * ys), heads, wbs, xbs as i64, ybs as i64, self.s.raw()
                ));
            }
        } else {
            ck!(ffi::gs_gemm(
                w.ty.id() as i32, w.ptr, w.row_bytes as i64, w.d0 as i32, w.d1 as i32, x, xs as i64, y, ys as i64, t as i32, 0,
                heads, wbs, xbs as i64, ybs as i64, self.s.raw()
            ));
        }
        Ok(())
    }
    fn rms(&self, x: *const f32, w: Option<&DTensor>, y: *mut f32, rows: usize, n: usize) -> Result<()> {
        ck!(ffi::gs_rmsnorm(x, w.map_or(std::ptr::null(), |w| w.f32()), y, rows as i32, n as i32, self.cfg.rms_eps, self.s.raw()));
        Ok(())
    }

    /// Trace: stats of the last row of a [t][n] activation (names follow llama-eval-callback).
    fn dbg(&self, name: &str, il: isize, p: *const f32, n: usize, t: usize) -> Result<()> {
        if !self.trace {
            return Ok(());
        }
        let mut v = vec![0f32; n];
        check(unsafe { ffi::gs_memcpy_d2h(v.as_mut_ptr() as _, p.wrapping_add((t - 1) * n) as _, n * 4, self.s.raw()) })?;
        self.s.sync()?;
        let sum: f64 = v.iter().map(|x| *x as f64).sum();
        let l = if il >= 0 { format!("{name}-{il}") } else { name.to_string() };
        eprintln!(
            "  [{:>4}] {:<22} n={:<6} sum={:<14.6} [{:.6}, {:.6}, {:.6} ... {:.6}]",
            self.p0 + t - 1, l, n, sum, v[0], v[1.min(n - 1)], v[2.min(n - 1)], v[n - 1]
        );
        Ok(())
    }

    /// Embedding rows on the CPU, uploaded to `dst` ([toks.len()][n_embd]).
    fn embed(&self, toks: &[u32], dst: *mut f32) -> Result<()> {
        let t = &self.tok_embd;
        let n = t.dims[0] as usize;
        let rb = t.ty.row_bytes(t.dims[0]) as usize;
        let mut raw = vec![0u8; rb];
        let mut rows = vec![0f32; toks.len() * n];
        for (i, &tok) in toks.iter().enumerate() {
            if tok as u64 >= t.dims[1] {
                bail!("token id {tok} out of range");
            }
            read_at(&self.files[t.shard], t.offset + tok as u64 * rb as u64, &mut raw)?;
            quants::dequantize(t.ty, &raw, &mut rows[i * n..(i + 1) * n])?;
        }
        ck!(ffi::gs_memcpy_h2d(dst as _, rows.as_ptr() as _, rows.len() * 4, self.s.raw()));
        self.s.sync() // `rows` is pageable host memory
    }

    // ---- mHC ----------------------------------------------------------------------------------------
    fn hc_pre(&self, hc: &Hc, t: usize) -> Result<()> {
        let n = self.cfg.n_embd;
        let hn = self.cfg.hc * n;
        self.rms(self.streams.f32(), None, self.flat.f32(), t, hn)?;
        self.mm(&hc.fn_, self.flat.f32(), self.mixes.f32(), t)?;
        ck!(ffi::gs_hc_pre(
            self.streams.f32(), self.mixes.f32(), hc.scale.f32(), hc.base.f32(), self.hcw.f32(), self.x.f32(), t as i32,
            n as i32, self.cfg.hc_eps, self.cfg.hc_sinkhorn_iters as i32, self.s.raw()
        ));
        Ok(())
    }
    fn hc_post(&self, t: usize) -> Result<()> {
        ck!(ffi::gs_hc_post(self.streams.f32(), self.out.f32(), self.hcw.f32(), t as i32, self.cfg.n_embd as i32, self.s.raw()));
        Ok(())
    }

    // ---- attention ----------------------------------------------------------------------------------
    fn kda(&self, il: usize, w: &KdaW, t: usize, save_first: bool) -> Result<()> {
        let c = &self.cfg;
        let (h, d) = (c.n_head, c.kda_head_dim);
        let di = h * d;
        let LayerState::Kda { s: state, conv, s_save, conv_save } = &self.states[il] else { unreachable!() };
        let save_at = if save_first { 0 } else { -1 };
        let conv_n = di * (c.kda_conv - 1);
        let xn = self.xn.f32();
        let (q, k, v, g) = (self.a.f32(), self.b.f32(), self.c.f32(), self.d.f32());
        self.mm(&w.q, xn, q, t)?;
        self.mm(&w.k, xn, k, t)?;
        self.mm(&w.v, xn, v, t)?;
        let cs = conv.f32();
        let csv = conv_save.as_ref().map_or(std::ptr::null_mut(), |b| b.f32());
        let kk = c.kda_conv as i32;
        for (i, (buf, cw)) in [(q, &w.conv_q), (k, &w.conv_k), (v, &w.conv_v)].into_iter().enumerate() {
            ck!(ffi::gs_conv_scan(
                buf, cs.wrapping_add(i * conv_n), cw.f32(), buf, t as i32, di as i32, kk, save_at,
                csv.wrapping_add(i * conv_n), self.s.raw()
            ));
        }
        ck!(ffi::gs_l2norm(q, q, (t * h) as i32, d as i32, 1e-6, 1.0 / (d as f32).sqrt(), self.s.raw()));
        ck!(ffi::gs_l2norm(k, k, (t * h) as i32, d as i32, 1e-6, 1.0, self.s.raw()));
        // forget gate
        let fa = self.s1.f32();
        self.mm(&w.f_a, xn, fa, t)?;
        self.mm(&w.f_b, fa, g, t)?;
        ck!(ffi::gs_kda_gate(g, w.dt_b.f32(), w.a.f32(), g, (t * di) as i64, d as i32, h as i32, c.kda_gate_lower_bound, self.s.raw()));
        let beta = self.s2.f32();
        self.mm(&w.beta, xn, beta, t)?;
        let ssv = s_save.as_ref().map_or(std::ptr::null_mut(), |b| b.f32());
        ck!(ffi::gs_kda_scan(state.f32(), q, k, v, g, beta, v, t as i32, h as i32, d as i32, save_at, ssv, self.s.raw()));
        self.dbg("kda_scan_out", il as isize, v, di, t)?;
        // output gate + gated RMSNorm (in place in v), then o_proj
        let ga = self.s3.f32();
        self.mm(&w.g_a, xn, ga, t)?;
        self.mm(&w.g_b, ga, g, t)?;
        ck!(ffi::gs_rmsnorm_gated(v, w.o_norm.f32(), g, v, (t * h) as i32, d as i32, c.rms_eps, self.s.raw()));
        self.mm(&w.o, v, self.out.f32(), t)
    }

    fn mla(&self, il: usize, w: &MlaW, t: usize) -> Result<()> {
        let c = &self.cfg;
        let (h, dq, r) = (c.n_head, c.mla_head_k, c.kv_lora_rank);
        let (dk, kp) = (c.idx_head_dim, c.idx_kpool);
        let LayerState::Mla { latent, scales, pooled, ring_k, ring_g } = &self.states[il] else { unreachable!() };
        let p0 = self.p0;
        if p0 + t > self.ctx {
            bail!("context full ({} tokens)", self.ctx);
        }
        let xn = self.xn.f32();
        let (qa, qr, kv) = (self.s1.f32(), self.s2.f32(), self.s3.f32());
        self.mm(&w.q_a, xn, qa, t)?;
        self.rms(qa, Some(&w.q_a_norm), qr, t, c.q_lora_rank)?;
        let q = self.a.f32();
        self.mm(&w.q_b, qr, q, t)?;
        // latents of these positions -> f16 cache
        self.mm(&w.kv_a, xn, kv, t)?;
        self.rms(kv, Some(&w.kv_a_norm), kv, t, r)?;
        match scales {
            Some(sc) => {
                let lat = (latent.ptr() as *mut u8).wrapping_add(p0 * r);
                let d = (sc.ptr() as *mut u8).wrapping_add(p0 * (r / 32) * 2);
                ck!(ffi::gs_quant_q8_rows(kv, lat as _, d as _, (t * r) as i64, self.s.raw()));
            }
            None => {
                let lat = (latent.ptr() as *mut u8).wrapping_add(p0 * r * 2);
                ck!(ffi::gs_f32_to_f16(kv, lat as _, (t * r) as i64, self.s.raw()));
            }
        }
        let lat_scales = scales.as_ref().map_or(std::ptr::null_mut(), |b| b.ptr());
        // indexer keys / gates of these positions, pooled when a pool of `kp` completes
        let (kc, gc) = (self.b.f32(), self.b.f32().wrapping_add(t * dk));
        self.mm(&w.ik, xn, kc, t)?;
        ck!(ffi::gs_layernorm(kc, w.ik_norm_w.f32(), w.ik_norm_b.f32(), kc, t as i32, dk as i32, 1e-6, self.s.raw()));
        self.mm(&w.igate, xn, gc, t)?;
        ck!(ffi::gs_idx_pool(
            kc, gc, ring_k.f32(), ring_g.f32(), w.iape.f32(), pooled.f32(), p0 as i32, t as i32, dk as i32, kp as i32,
            self.s.raw()
        ));
        // per-query attended positions (dense below top_k pools, else top-k pools + tail)
        let sparse = (p0 + t) / kp > c.idx_top_k / kp;
        let (iq, iw) = (self.c.f32(), self.s1.f32());
        if sparse {
            self.mm(&w.iq_b, qr, iq, t)?;
            self.mm(&w.iproj, xn, iw, t)?;
        }
        let ss = (self.ctx / kp + 1) as i64;
        let sel = self.sel.ptr() as *mut i32;
        let cnt = self.cnt.ptr() as *mut i32;
        for t0 in (0..t).step_by(TQ) {
            let tn = TQ.min(t - t0);
            if sparse {
                ck!(ffi::gs_idx_score(
                    iq.wrapping_add(t0 * c.idx_n_head * dk), iw.wrapping_add(t0 * c.idx_n_head), pooled.f32(), self.scores.f32(),
                    ss, (p0 + t0) as i32, tn as i32, kp as i32, c.idx_n_head as i32, dk as i32, 1.0 / (dk as f32).sqrt(),
                    self.s.raw()
                ));
            }
            ck!(ffi::gs_idx_select(
                self.scores.f32(), ss, (p0 + t0) as i32, tn as i32, kp as i32, (c.idx_top_k / kp) as i32,
                sel.wrapping_add(t0 * self.sel_stride), self.sel_stride as i32, cnt.wrapping_add(t0), self.s.raw()
            ));
        }
        // attention in sub-batches: absorb k_b into q, flash attention over latents, then v_b
        let ov = self.d.f32();
        for t0 in (0..t).step_by(TQ) {
            let tn = TQ.min(t - t0);
            self.mm_heads(&w.k_b, q.wrapping_add(t0 * h * dq), h * dq, dq, self.qabs.f32(), h * r, r, tn)?;
            let splits = if tn == 1 { 16 } else if tn <= 8 { 4 } else { 1 };
            ck!(ffi::gs_mla_attn(
                self.qabs.f32(), latent.ptr(), lat_scales, sel.wrapping_add(t0 * self.sel_stride), self.sel_stride as i32,
                cnt.wrapping_add(t0), 1.0 / (dq as f32).sqrt(), self.olat.f32(), self.part.f32(), tn as i32, h as i32,
                r as i32, splits, self.s.raw()
            ));
            self.mm_heads(&w.v_b, self.olat.f32(), h * r, r, ov.wrapping_add(t0 * h * c.mla_head_v), h * c.mla_head_v, c.mla_head_v, tn)?;
        }
        self.dbg("kqv_out", il as isize, ov, h * c.mla_head_v, t)?;
        self.mm(&w.o, ov, self.out.f32(), t)
    }

    // ---- FFN ----------------------------------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn swiglu_ffn(&self, gate: &DTensor, up: &DTensor, down: &DTensor, x: *const f32, out: *mut f32, lim: f32, t: usize) -> Result<()> {
        let (g, u) = (self.a.f32(), self.b.f32());
        self.mm(gate, x, g, t)?;
        self.mm(up, x, u, t)?;
        ck!(ffi::gs_swiglu(g, u, g, (t * gate.d1) as i64, lim, self.s.raw()));
        self.mm(down, g, out, t)
    }

    /// Routing with cache-aware substitution (decode only): chosen experts of rank 3-8 that are
    /// not resident are replaced by the best resident unchosen expert whose selection score is
    /// within `cache_aware` of theirs. Weights are recomputed over the final set.
    fn route_cached(&mut self, il: usize, bias: &[f32], logits: &[f32], t: usize) -> Vec<Vec<(u16, f32)>> {
        let m = self.cache_aware;
        if m <= 0.0 || t > 2 {
            return self.route(bias, logits, t);
        }
        let c = &self.cfg;
        let ne = c.n_expert;
        let k = c.n_expert_used;
        let mut swaps = 0;
        let out = (0..t)
            .map(|ti| {
                let l = &logits[ti * ne..(ti + 1) * ne];
                let scores: Vec<f32> = l.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
                let sel = |e: usize| scores[e] + bias[e];
                let mut order: Vec<usize> = (0..ne).collect();
                order.sort_by(|a, b| sel(*b).total_cmp(&sel(*a)));
                let mut chosen: Vec<usize> = order[..k].to_vec();
                let mut spare: Vec<usize> =
                    order[k..].iter().copied().filter(|&e| self.experts.resident(il, e as u16)).collect();
                for j in 2..k {
                    let e = chosen[j];
                    if self.experts.resident(il, e as u16) {
                        continue;
                    }
                    if let Some(pos) = spare.iter().position(|&cand| sel(cand) >= sel(e) - m) {
                        chosen[j] = spare.remove(pos);
                        swaps += 1;
                    }
                }
                let wsum: f32 = if c.expert_weights_norm { chosen.iter().map(|e| scores[*e]).sum() } else { 1.0 };
                chosen.iter().map(|&e| (e as u16, scores[e] / (wsum + 1e-20) * c.expert_weights_scale)).collect()
            })
            .collect();
        self.swaps += swaps;
        out
    }

    /// Host-side routing: per token the chosen experts and their weights.
    fn route(&self, bias: &[f32], logits: &[f32], t: usize) -> Vec<Vec<(u16, f32)>> {
        let c = &self.cfg;
        let ne = c.n_expert;
        (0..t)
            .map(|ti| {
                let l = &logits[ti * ne..(ti + 1) * ne];
                let scores: Vec<f32> = l.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
                let mut order: Vec<usize> = (0..ne).collect();
                let k = c.n_expert_used;
                order.select_nth_unstable_by(k - 1, |a, b| (scores[*b] + bias[*b]).total_cmp(&(scores[*a] + bias[*a])));
                order.truncate(k);
                order.sort_by(|a, b| (scores[*b] + bias[*b]).total_cmp(&(scores[*a] + bias[*a])));
                let wsum: f32 = if c.expert_weights_norm { order.iter().map(|e| scores[*e]).sum() } else { 1.0 };
                order.iter().map(|&e| (e as u16, scores[e] / (wsum + 1e-20) * c.expert_weights_scale)).collect()
            })
            .collect()
    }

    fn moe(&mut self, il: usize, t: usize) -> Result<()> {
        let t_start = Instant::now();
        let n = self.cfg.n_embd;
        let ne = self.cfg.n_expert;
        let xn = self.xn.f32();
        let Ffn::Moe { router, bias, sh_gate, sh_up, sh_down } = &self.layers[il].ffn else { unreachable!() };
        let (router, bias, sh_gate, sh_up, sh_down) = (*router, bias.clone(), *sh_gate, *sh_up, *sh_down);
        let next_router = match self.layers.get(il + 1).map(|l| &l.ffn) {
            Some(Ffn::Moe { router, bias, .. }) if il + 1 < self.cfg.n_layer => Some((*router, bias.clone())),
            _ => None,
        };
        self.mm(&router, xn, self.router.f32(), t)?;
        let mut logits = vec![0f32; t * ne];
        self.router.download(0, bytes_of_mut(&mut logits), &self.s)?;
        self.s.sync()?;
        let routes = self.route_cached(il, &bias, &logits, t);
        if self.trace {
            let r = &routes[t - 1];
            eprintln!(
                "  [{:>4}] experts-{il}: {:?} w={:?}",
                self.p0 + t - 1,
                r.iter().map(|x| x.0).collect::<Vec<_>>(),
                r.iter().map(|x| (x.1 * 1000.0).round() / 1000.0).collect::<Vec<_>>()
            );
        }
        // group tokens by expert
        let mut groups: Vec<Vec<(i32, f32)>> = vec![Vec::new(); ne];
        for (ti, r) in routes.iter().enumerate() {
            for &(e, w) in r {
                groups[e as usize].push((ti as i32, w));
            }
        }
        let used: Vec<u16> = (0..ne as u16).filter(|e| !groups[*e as usize].is_empty()).collect();
        let mut offs = vec![0usize; ne];
        let mut flat_idx = Vec::with_capacity(t * self.cfg.n_expert_used);
        let mut flat_w = Vec::with_capacity(t * self.cfg.n_expert_used);
        for &e in &used {
            offs[e as usize] = flat_idx.len();
            for &(ti, w) in &groups[e as usize] {
                flat_idx.push(ti);
                flat_w.push(w);
            }
        }
        if t > 1 {
            self.tok_idx.upload(0, bytes_of(&flat_idx), &self.s)?;
            self.tok_w.upload(0, bytes_of(&flat_w), &self.s)?;
        }

        // speculative prefetch for the next MoE layer during decode
        let pk = self.experts.prefetch_k;
        if let Some((r2, b2)) = next_router.filter(|_| t <= 2 && pk > 0) {
            self.mm(&r2, xn, self.router.f32(), t)?;
            let mut l2 = vec![0f32; t * ne];
            self.router.download(0, bytes_of_mut(&mut l2), &self.s)?;
            self.s.sync()?;
            let mut guess: Vec<u16> = self.route(&b2, &l2, t).into_iter().flat_map(|r| r.into_iter().take(pk)).map(|x| x.0).collect();
            guess.sort_unstable();
            guess.dedup();
            self.experts.prefetch(il + 1, &guess);
        }

        let mut fetch = self.experts.fetch_layer(il, &used)?;
        // shared expert first: overlaps with the first expert uploads
        self.swiglu_ffn(&sh_gate, &sh_up, &sh_down, xn, self.out.f32(), self.layers[il].clamp_shexp, t)?;
        self.dbg("ffn_shexp", il as isize, self.out.f32(), n, t)?;
        self.moe_acc.zero(&self.s)?;
        let lim = self.layers[il].clamp_exp;
        let parts = self.exp_dims[il].context("no expert tensors")?;
        while let Some((e, p)) = self.experts.next(&mut fetch, &self.copy, &self.s)? {
            let g = &groups[e as usize];
            self.run_expert(il, &parts, p, g, offs[e as usize], lim, t)?;
            self.experts.done_with(il, e, &self.s)?;
        }
        self.dbg("ffn_moe_out", il as isize, self.moe_acc.f32(), n, t)?;
        ck!(ffi::gs_axpy(self.out.f32(), self.moe_acc.f32(), 1.0, (t * n) as i64, self.s.raw()));
        self.timings.moe += t_start.elapsed();
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_expert(&self, _il: usize, dims: &[DTensor; 3], p: ExpertPtrs, g: &[(i32, f32)], off: usize, lim: f32, t: usize) -> Result<()> {
        let n = self.cfg.n_embd;
        let mk = |i: usize, ptr: *const u8| DTensor { ptr: ptr as _, ..dims[i] };
        let (wg, wu, wd) = (mk(0, p.gate), mk(1, p.up), mk(2, p.down));
        let (eg, eu) = (self.eg.f32(), self.eu.f32());
        if t == 1 {
            self.mm(&wg, self.xn.f32(), eg, 1)?;
            self.mm(&wu, self.xn.f32(), eu, 1)?;
            ck!(ffi::gs_swiglu(eg, eu, eg, wg.d1 as i64, lim, self.s.raw()));
            self.mm(&wd, eg, self.ey.f32(), 1)?;
            ck!(ffi::gs_axpy(self.moe_acc.f32(), self.ey.f32(), g[0].1, n as i64, self.s.raw()));
            return Ok(());
        }
        let m = g.len();
        let idx = (self.tok_idx.ptr() as *const i32).wrapping_add(off);
        let w = self.tok_w.f32().wrapping_add(off);
        ck!(ffi::gs_gather_rows(self.xn.f32(), idx, self.xg.f32(), m as i32, n as i32, self.s.raw()));
        self.mm(&wg, self.xg.f32(), eg, m)?;
        self.mm(&wu, self.xg.f32(), eu, m)?;
        ck!(ffi::gs_swiglu(eg, eu, eg, (m * wg.d1) as i64, lim, self.s.raw()));
        self.mm(&wd, eg, self.ey.f32(), m)?;
        ck!(ffi::gs_scatter_add_rows(self.moe_acc.f32(), idx, w, self.ey.f32(), m as i32, n as i32, self.s.raw()));
        Ok(())
    }

    // ---- full passes --------------------------------------------------------------------------------
    /// Run `toks` (positions pos..pos+len) through the trunk. Returns logits of the requested rows.
    /// With `save_first`, the KDA/conv state after the first token is kept for `rollback_to_first`.
    pub fn forward(&mut self, toks: &[u32], rows: Rows, save_first: bool) -> Result<Vec<Vec<f32>>> {
        if toks.is_empty() {
            return Ok(Vec::new());
        }
        let n_chunks = toks.len().div_ceil(self.chunk);
        let mut out = Vec::new();
        for (ci, ch) in toks.chunks(self.chunk).enumerate() {
            let last = ci + 1 == n_chunks;
            // the token after this chunk (feeds the MTP row of the chunk's last position)
            let next = if last { None } else { Some(toks[(ci + 1) * self.chunk]) };
            out = self.forward_chunk(ch, if last { rows } else { Rows::None }, save_first && ci == 0, next)?;
        }
        Ok(out)
    }

    fn forward_chunk(&mut self, toks: &[u32], rows: Rows, save_first: bool, next: Option<u32>) -> Result<Vec<Vec<f32>>> {
        let t0 = Instant::now();
        let t = toks.len();
        let n = self.cfg.n_embd;
        if self.pos + t > self.ctx {
            bail!("context full: {} + {t} tokens > --ctx {}", self.pos, self.ctx);
        }
        if rows == Rows::All && t > 8 {
            bail!("logits for all rows need <= 8 tokens");
        }
        self.p0 = self.pos;
        self.embed(toks, self.x.f32())?;
        ck!(ffi::gs_hc_expand(self.x.f32(), self.streams.f32(), t as i32, n as i32, self.s.raw()));
        self.dbg("inp_embd", -1, self.x.f32(), n, t)?;

        for il in 0..self.cfg.n_layer {
            let layer: &Layer = unsafe { &*(&self.layers[il] as *const Layer) }; // layers are never mutated
            let (hca, hcf) = layer.hc.as_ref().unwrap();
            self.hc_pre(hca, t)?;
            self.rms(self.x.f32(), Some(&layer.attn_norm), self.xn.f32(), t, n)?;
            self.dbg("attn_norm", il as isize, self.xn.f32(), n, t)?;
            match &layer.attn {
                Attn::Kda(w) => self.kda(il, w, t, save_first)?,
                Attn::Mla(w) => self.mla(il, w, t)?,
            }
            self.dbg("attn_out", il as isize, self.out.f32(), n, t)?;
            self.hc_post(t)?;
            self.hc_pre(hcf, t)?;
            self.rms(self.x.f32(), Some(&layer.ffn_norm), self.xn.f32(), t, n)?;
            self.dbg("ffn_norm", il as isize, self.xn.f32(), n, t)?;
            let dense = match &layer.ffn {
                Ffn::Dense { gate, up, down } => Some((*gate, *up, *down, layer.clamp_shexp)),
                Ffn::Moe { .. } => None,
            };
            match dense {
                Some((g, u, d, lim)) => self.swiglu_ffn(&g, &u, &d, self.xn.f32(), self.out.f32(), lim, t)?,
                None => self.moe(il, t)?,
            }
            self.dbg("ffn_out", il as isize, self.out.f32(), n, t)?;
            self.hc_post(t)?;
            self.dbg("l_out", il as isize, self.streams.f32(), self.cfg.hc * n, t)?;
        }
        self.pos += t;
        self.experts.tick(t as u64);

        // normed final hidden of every row the caller or the MTP head needs
        let mtp_feed = self.mtp.is_some() && rows != Rows::All;
        let (r0, nr) = match rows {
            Rows::None => (t - 1, if mtp_feed { 1 } else { 0 }),
            Rows::Last => (t - 1, 1),
            Rows::All => (0, t),
        };
        if mtp_feed {
            // all rows: hall[1..=t]; MTP consumes (pending, rows 0..t-2) with the chunk's tokens
            let hall1 = self.hall.f32().wrapping_add(n);
            ck!(ffi::gs_hc_mean(self.streams.f32(), self.x.f32(), t as i32, n as i32, self.s.raw()));
            self.rms(self.x.f32(), Some(&self.output_norm), hall1, t, n)?;
        }
        let mut result = Vec::new();
        if nr > 0 {
            let hs = self.streams.f32().wrapping_add(r0 * self.cfg.hc * n);
            ck!(ffi::gs_hc_mean(hs, self.x.f32(), nr as i32, n as i32, self.s.raw()));
            self.rms(self.x.f32(), Some(&self.output_norm), self.hid.f32(), nr, n)?;
            self.dbg("result_norm", -1, self.hid.f32(), n, nr)?;
            if rows != Rows::None {
                self.mm(&self.output, self.hid.f32(), self.logits.f32(), nr)?;
                let v = self.cfg.n_vocab;
                let mut l = vec![0f32; nr * v];
                self.logits.download(0, bytes_of_mut(&mut l), &self.s)?;
                self.s.sync()?;
                if self.trace {
                    let ll = &l[(nr - 1) * v..];
                    let sum: f64 = ll.iter().map(|x| *x as f64).sum();
                    eprintln!("  [{:>4}] result_output          sum={sum:.6} [{:.6}, {:.6}, {:.6}]", self.p0 + t - 1, ll[0], ll[1], ll[2]);
                }
                result = l.chunks(v).map(|c| c.to_vec()).collect();
            }
        }
        if mtp_feed {
            self.mtp_feed_chunk(toks, next)?;
        }
        self.s.sync()?;
        self.timings.tokens += t;
        self.timings.total += t0.elapsed();
        Ok(result)
    }

    /// After a prefill chunk: run the MTP layer over (pending row, rows 0..t-2) so its attention
    /// cache covers the prompt, and keep the last row pending until its next token is known.
    fn mtp_feed_chunk(&mut self, toks: &[u32], next: Option<u32>) -> Result<()> {
        let n = self.cfg.n_embd;
        let t = toks.len();
        let hall = self.hall.f32();
        let mut htoks: Vec<u32> = Vec::new();
        let first_row;
        if self.pend_n == 1 {
            // pending hidden (position p0-1) goes to hall[0] and pairs with toks[0]
            unsafe { self.s.d2d(hall as _, self.pend.ptr(), n * 4)? };
            first_row = 0;
            htoks.extend_from_slice(toks);
        } else {
            first_row = 1;
            htoks.extend_from_slice(&toks[1..]);
        }
        if let Some(nx) = next {
            htoks.push(nx);
        }
        // rows first_row .. first_row+htoks.len() of hall, tokens htoks
        let rows = htoks.len();
        // the row that has no next token yet becomes pending
        let last_row = 1 + t - 1; // hall index of the chunk's last position
        if next.is_none() {
            unsafe { self.s.d2d(self.pend.ptr(), hall.wrapping_add(last_row * n) as _, n * 4)? };
            self.pend_n = 1;
        } else {
            self.pend_n = 0;
        }
        if rows > 0 {
            let p = self.mtp_pos;
            self.mtp_rows(hall.wrapping_add(first_row * n), &htoks, p, false)?;
        }
        Ok(())
    }

    /// The MTP block over `rows.len()` rows: inputs h (device, [rows][n]) and next tokens.
    /// Writes its attention cache at positions p..p+rows. Returns logits of the last row if asked.
    fn mtp_rows(&mut self, h: *const f32, toks: &[u32], p: usize, want: bool) -> Result<Option<Vec<f32>>> {
        let n = self.cfg.n_embd;
        let t = toks.len();
        let il = self.cfg.n_layer;
        let Some(head) = self.mtp.as_ref() else { bail!("MTP head not loaded") };
        let (eh, enorm, hnorm, head_norm) = (head.eh, head.enorm, head.hnorm, head.head_norm);
        // e = rmsnorm(emb) -> c ; hn = rmsnorm(h) -> d ; x = eh_lo e + eh_hi hn
        let (ce, dh) = (self.c.f32(), self.d.f32());
        self.embed(toks, ce)?;
        self.rms(ce, Some(&enorm), ce, t, n)?;
        self.rms(h, Some(&hnorm), dh, t, n)?;
        let half = DTensor { d0: n, ..eh };
        let hi = DTensor { ptr: (eh.ptr as *const u8).wrapping_add(eh.ty.row_bytes(n as u64) as usize) as _, d0: n, ..eh };
        self.mm_s(&half, ce, n, self.x.f32(), n, t, false)?;
        self.mm_s(&hi, dh, n, self.x.f32(), n, t, true)?;
        let saved_p0 = self.p0;
        self.p0 = p;
        let layer = &self.layers[il];
        self.rms(self.x.f32(), Some(&layer.attn_norm), self.xn.f32(), t, n)?;
        let Attn::Mla(w) = &layer.attn else { bail!("MTP layer is not MLA") };
        let w: &MlaW = unsafe { &*(w.as_ref() as *const MlaW) };
        self.mla(il, w, t)?;
        // residual: x += attn_out ; FFN on rmsnorm(x) ; x += ffn
        ck!(ffi::gs_axpy(self.x.f32(), self.out.f32(), 1.0, (t * n) as i64, self.s.raw()));
        let ffn_norm = self.layers[il].ffn_norm;
        self.rms(self.x.f32(), Some(&ffn_norm), self.xn.f32(), t, n)?;
        self.moe(il, t)?;
        ck!(ffi::gs_axpy(self.x.f32(), self.out.f32(), 1.0, (t * n) as i64, self.s.raw()));
        self.p0 = saved_p0;
        self.mtp_pos = p + t;
        if !want {
            return Ok(None);
        }
        let last = self.x.f32().wrapping_add((t - 1) * n);
        self.rms(last, Some(&head_norm), self.xn.f32(), 1, n)?;
        self.mm(&self.output, self.xn.f32(), self.logits.f32(), 1)?;
        let mut l = vec![0f32; self.cfg.n_vocab];
        self.logits.download(0, bytes_of_mut(&mut l), &self.s)?;
        self.s.sync()?;
        Ok(Some(l))
    }

    // ---- speculative decoding (MTP) -----------------------------------------------------------------
    /// Run the MTP block over `rows` = (hidden row, next token) in position order and return the
    /// draft logits of the last one. A hidden row is an index into `hid` (rows of the last
    /// forward) or `usize::MAX` for the pending row left by a prefill.
    pub fn mtp_draft(&mut self, rows: &[(usize, u32)]) -> Result<Vec<f32>> {
        // rows: (index into `hid`, next token) in position order, the last one is the newest
        let n = self.cfg.n_embd;
        let toks: Vec<u32> = rows.iter().map(|r| r.1).collect();
        // gather the hidden rows into `hall[0..k]`
        for (i, (hi, _)) in rows.iter().enumerate() {
            let src = if *hi == usize::MAX { self.pend.f32() } else { self.hid.f32().wrapping_add(hi * n) };
            unsafe { self.s.d2d(self.hall.f32().wrapping_add(i * n) as _, src as _, n * 4)? };
        }
        let p = self.mtp_pos;
        let l = self.mtp_rows(self.hall.f32(), &toks, p, true)?.unwrap();
        self.pend_n = 0;
        Ok(l)
    }

    /// Undo the second token of a 2-token verification (restore state saved after the first).
    pub fn rollback_to_first(&mut self) -> Result<()> {
        for st in &self.states {
            if let LayerState::Kda { s, conv, s_save: Some(ss), conv_save: Some(cs) } = st {
                unsafe {
                    self.s.d2d(s.ptr(), ss.ptr(), s.len())?;
                    self.s.d2d(conv.ptr(), cs.ptr(), conv.len())?;
                }
            }
        }
        self.pos -= 1;
        self.s.sync()
    }

    /// Copy hidden row `i` of the last forward into the pending slot (for the next draft).
    pub fn keep_pending(&mut self, i: usize) -> Result<()> {
        let n = self.cfg.n_embd;
        unsafe { self.s.d2d(self.pend.ptr(), self.hid.f32().wrapping_add(i * n) as _, n * 4)? };
        self.pend_n = 1;
        Ok(())
    }
}
