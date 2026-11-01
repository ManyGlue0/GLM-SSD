//! Single-token forward pass of glm5-next (GLM-5.3-Flash) on the GPU.
//!
//! M2 "correct first" version: non-expert weights are resident in VRAM, routed experts are read
//! from the GGUF on demand (8 parallel reads per MoE layer) with no caching yet (that's M3).
//! The prompt is processed one token at a time through the same decode path.

use crate::config::{AttnKind, Config};
use crate::model::{read_at, read_f32, DTensor, ExpertTensors, Weights};
use anyhow::{bail, Context, Result};
use cuda::{bytes_of, bytes_of_mut, check, ffi, ops, DeviceBuffer, PinnedBuffer, Stream};
use gguf::{Gguf, TensorInfo};
use std::fs::File;
use std::time::{Duration, Instant};

macro_rules! ck {
    ($e:expr) => {
        check(unsafe { $e })?
    };
}

const MAX_SEL_IDX: usize = 1 << 16;

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
    // indexer
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
    Moe { router: DTensor, bias: Vec<f32>, experts: ExpertTensors, sh_gate: DTensor, sh_up: DTensor, sh_down: DTensor },
}

struct Hc {
    fn_: DTensor,
    scale: DTensor,
    base: DTensor,
}

struct Layer {
    attn_norm: DTensor,
    ffn_norm: DTensor,
    hc_attn: Hc,
    hc_ffn: Hc,
    attn: Attn,
    ffn: Ffn,
    clamp_exp: f32,
    clamp_shexp: f32,
}

/// Per-layer persistent state.
enum LayerState {
    Kda { s: DeviceBuffer, conv: DeviceBuffer },
    Mla { latent: DeviceBuffer, idx: DeviceBuffer, pooled: DeviceBuffer },
}

#[derive(Default, Debug, Clone)]
pub struct Timings {
    pub tokens: usize,
    pub total: Duration,
    pub expert_read: Duration,
    pub expert_bytes: u64,
}

pub struct Engine {
    pub cfg: Config,
    files: Vec<File>,
    tok_embd: TensorInfo,
    _w: Weights,
    layers: Vec<Layer>,
    states: Vec<LayerState>,
    output_norm: DTensor,
    output: DTensor,
    s: Stream,
    pub ctx: usize,
    pub pos: usize,
    pub trace: bool,
    pub timings: Timings,

    // scratch (device)
    streams: DeviceBuffer,
    x: DeviceBuffer,
    xn: DeviceBuffer,
    flat: DeviceBuffer,
    mixes: DeviceBuffer,
    hcw: DeviceBuffer,
    out: DeviceBuffer,
    t: [DeviceBuffer; 8], // generic large temporaries (>= 16384 floats each)
    small: [DeviceBuffer; 4],
    qabs: DeviceBuffer,
    olat: DeviceBuffer,
    att_scratch: DeviceBuffer,
    sel_idx: DeviceBuffer,
    pool_scores: DeviceBuffer,
    router: DeviceBuffer,
    moe_acc: DeviceBuffer,
    logits: DeviceBuffer,
    // expert staging
    exp_slot: usize,
    exp_host: PinnedBuffer,
    exp_dev: DeviceBuffer,
}

fn fbuf(n: usize, s: &Stream) -> Result<DeviceBuffer> {
    let b = DeviceBuffer::new(n * 4)?;
    b.zero(s)?;
    Ok(b)
}

impl Engine {
    pub fn new(g: &Gguf, cfg: Config, ctx: usize) -> Result<Self> {
        let s = Stream::new()?;
        let files = g.shards.iter().map(File::open).collect::<std::io::Result<Vec<_>>>()?;
        let w = Weights::load(g, &files, cfg.n_layer, &s)?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let mut states = Vec::with_capacity(cfg.n_layer);
        let d_inner = cfg.n_head * cfg.kda_head_dim;
        let mut exp_slot = 0usize;

        for il in 0..cfg.n_layer {
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
                    states.push(LayerState::Kda {
                        s: fbuf(cfg.n_head * cfg.kda_head_dim * cfg.kda_head_dim, &s)?,
                        conv: fbuf(3 * d_inner * (cfg.kda_conv - 1), &s)?,
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
                    states.push(LayerState::Mla {
                        latent: fbuf(ctx * cfg.kv_lora_rank, &s)?,
                        idx: fbuf(ctx * 2 * cfg.idx_head_dim, &s)?,
                        pooled: fbuf((ctx / cfg.idx_kpool + 1) * cfg.idx_head_dim, &s)?,
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
                let experts = ExpertTensors::for_layer(g, il)?;
                exp_slot = exp_slot.max(experts.unit_bytes());
                Ffn::Moe {
                    router: b("ffn_gate_inp.weight")?,
                    bias: read_f32(g, &files, &format!("blk.{il}.exp_probs_b.bias"))?,
                    experts,
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
                hc_attn: hc("attn")?,
                hc_ffn: hc("ffn")?,
                attn,
                ffn,
                clamp_exp: cfg.swiglu_clamp_exp[il],
                clamp_shexp: cfg.swiglu_clamp_shexp[il],
            });
        }

        let n = cfg.n_embd;
        let big = 16384usize.max(cfg.n_ff).max(cfg.n_head * cfg.mla_head_k);
        let exp_slot = exp_slot.next_multiple_of(4096) + 3 * 4096;
        let n_exp = cfg.n_expert_used;
        let tok_embd = g.tensor("token_embd.weight").context("token_embd")?.clone();
        let eng = Self {
            output_norm: w.get("output_norm.weight")?,
            output: w.get("output.weight")?,
            streams: fbuf(cfg.hc * n, &s)?,
            x: fbuf(n, &s)?,
            xn: fbuf(n, &s)?,
            flat: fbuf(cfg.hc * n, &s)?,
            mixes: fbuf(64, &s)?,
            hcw: fbuf(64, &s)?,
            out: fbuf(n, &s)?,
            t: [
                fbuf(big, &s)?, fbuf(big, &s)?, fbuf(big, &s)?, fbuf(big, &s)?,
                fbuf(big, &s)?, fbuf(big, &s)?, fbuf(big, &s)?, fbuf(big, &s)?,
            ],
            small: [fbuf(4096, &s)?, fbuf(4096, &s)?, fbuf(4096, &s)?, fbuf(4096, &s)?],
            qabs: fbuf(cfg.n_head * cfg.kv_lora_rank, &s)?,
            olat: fbuf(cfg.n_head * cfg.kv_lora_rank, &s)?,
            att_scratch: fbuf(cfg.n_head * (cfg.idx_top_k + cfg.idx_kpool).max(ctx), &s)?,
            sel_idx: DeviceBuffer::new(MAX_SEL_IDX * 4)?,
            pool_scores: fbuf(ctx / cfg.idx_kpool + 1, &s)?,
            router: fbuf(cfg.n_expert, &s)?,
            moe_acc: fbuf(n, &s)?,
            logits: fbuf(cfg.n_vocab, &s)?,
            exp_slot,
            exp_host: PinnedBuffer::new(exp_slot * n_exp)?,
            exp_dev: DeviceBuffer::new(exp_slot * n_exp)?,
            tok_embd,
            files,
            _w: w,
            layers,
            states,
            s,
            ctx,
            pos: 0,
            trace: false,
            timings: Timings::default(),
            cfg,
        };
        eng.s.sync()?;
        Ok(eng)
    }

    /// Forget the sequence (positions, recurrent states, caches).
    pub fn reset(&mut self) -> Result<()> {
        for st in &self.states {
            match st {
                LayerState::Kda { s, conv } => {
                    s.zero(&self.s)?;
                    conv.zero(&self.s)?;
                }
                LayerState::Mla { .. } => {} // overwritten as positions are written
            }
        }
        self.pos = 0;
        self.s.sync()
    }

    // ---- helpers ------------------------------------------------------------------------------
    fn mv(&self, w: &DTensor, x: *const f32, y: *mut f32) -> Result<()> {
        ops::gemv(w.ty.id(), w.ptr, w.row_bytes, w.d0, w.d1, x, w.d0, y, w.d1, 1, false, &self.s)
    }
    fn rms(&self, x: *const f32, w: Option<&DTensor>, y: *mut f32, rows: usize, n: usize) -> Result<()> {
        ck!(ffi::gs_rmsnorm(x, w.map_or(std::ptr::null(), |w| w.f32()), y, rows as i32, n as i32, self.cfg.rms_eps, self.s.raw()));
        Ok(())
    }
    fn tp(&self, i: usize) -> *mut f32 {
        self.t[i].f32()
    }

    fn dbg(&self, name: &str, il: isize, p: *const f32, n: usize) -> Result<()> {
        if !self.trace {
            return Ok(());
        }
        let mut v = vec![0f32; n];
        check(unsafe { ffi::gs_memcpy_d2h(v.as_mut_ptr() as _, p as _, n * 4, self.s.raw()) })?;
        self.s.sync()?;
        let sum: f64 = v.iter().map(|x| *x as f64).sum();
        let l = if il >= 0 { format!("{name}-{il}") } else { name.to_string() };
        eprintln!(
            "  [{:>4}] {:<22} n={:<6} sum={:<14.6} [{:.6}, {:.6}, {:.6} ... {:.6}]",
            self.pos, l, n, sum, v[0], v[1.min(n - 1)], v[2.min(n - 1)], v[n - 1]
        );
        Ok(())
    }

    /// Embedding row lookup on the CPU (only one row is needed per token).
    fn embed(&self, tok: u32) -> Result<Vec<f32>> {
        let t = &self.tok_embd;
        if tok as u64 >= t.dims[1] {
            bail!("token id {tok} out of range");
        }
        let rb = t.ty.row_bytes(t.dims[0]) as usize;
        let mut raw = vec![0u8; rb];
        read_at(&self.files[t.shard], t.offset + tok as u64 * rb as u64, &mut raw)?;
        let mut row = vec![0f32; t.dims[0] as usize];
        quants::dequantize(t.ty, &raw, &mut row)?;
        Ok(row)
    }

    // ---- mHC --------------------------------------------------------------------------------------
    /// Computes the hc weights and collapses the streams into `out_collapsed` (n_embd).
    fn hc_pre(&self, hc: &Hc, collapsed: *mut f32) -> Result<()> {
        let n = self.cfg.n_embd;
        self.rms(self.streams.f32(), None, self.flat.f32(), 1, self.cfg.hc * n)?;
        self.mv(&hc.fn_, self.flat.f32(), self.mixes.f32())?;
        ck!(ffi::gs_hc_pre(
            self.streams.f32(), self.mixes.f32(), hc.scale.f32(), hc.base.f32(), self.hcw.f32(), collapsed, 1, n as i32,
            self.cfg.hc_eps, self.cfg.hc_sinkhorn_iters as i32, self.s.raw()
        ));
        Ok(())
    }
    fn hc_post(&self, out: *const f32) -> Result<()> {
        ck!(ffi::gs_hc_post(self.streams.f32(), out, self.hcw.f32(), 1, self.cfg.n_embd as i32, self.s.raw()));
        Ok(())
    }

    // ---- attention ----------------------------------------------------------------------------------
    fn kda(&self, il: usize, w: &KdaW, x: *const f32, out: *mut f32) -> Result<()> {
        let c = &self.cfg;
        let (h, d) = (c.n_head, c.kda_head_dim);
        let di = h * d;
        let LayerState::Kda { s: state, conv } = &self.states[il] else { unreachable!() };
        let conv_n = di * (c.kda_conv - 1);
        let (q, k, v, qc, kc, vc) = (self.tp(0), self.tp(1), self.tp(2), self.tp(3), self.tp(4), self.tp(5));
        self.mv(&w.q, x, q)?;
        self.mv(&w.k, x, k)?;
        self.mv(&w.v, x, v)?;
        let cs = conv.f32();
        let kk = c.kda_conv as i32;
        ck!(ffi::gs_conv_step(q, cs, w.conv_q.f32(), qc, di as i32, kk, self.s.raw()));
        ck!(ffi::gs_conv_step(k, cs.wrapping_add(conv_n), w.conv_k.f32(), kc, di as i32, kk, self.s.raw()));
        ck!(ffi::gs_conv_step(v, cs.wrapping_add(2 * conv_n), w.conv_v.f32(), vc, di as i32, kk, self.s.raw()));
        ck!(ffi::gs_l2norm(qc, qc, h as i32, d as i32, 1e-6, 1.0 / (d as f32).sqrt(), self.s.raw()));
        ck!(ffi::gs_l2norm(kc, kc, h as i32, d as i32, 1e-6, 1.0, self.s.raw()));
        // forget gate
        let (fa, f, g) = (self.small[0].f32(), self.tp(0), self.tp(1));
        self.mv(&w.f_a, x, fa)?;
        self.mv(&w.f_b, fa, f)?;
        ck!(ffi::gs_kda_gate(f, w.dt_b.f32(), w.a.f32(), g, di as i32, d as i32, c.kda_gate_lower_bound, self.s.raw()));
        let beta = self.small[1].f32();
        self.mv(&w.beta, x, beta)?;
        let o = self.tp(2);
        ck!(ffi::gs_kda_step(state.f32(), qc, kc, vc, g, beta, o, h as i32, d as i32, self.s.raw()));
        self.dbg("kda_scan_out", il as isize, o, di)?;
        // output gate + gated norm
        let (ga, gate, gated) = (self.small[2].f32(), self.tp(3), self.tp(4));
        self.mv(&w.g_a, x, ga)?;
        self.mv(&w.g_b, ga, gate)?;
        ck!(ffi::gs_rmsnorm_gated(o, w.o_norm.f32(), gate, gated, h as i32, d as i32, c.rms_eps, self.s.raw()));
        self.mv(&w.o, gated, out)
    }

    /// Indexer bookkeeping for the current position; returns how many positions attention uses and
    /// whether `sel_idx` holds an explicit list (sparse) or attention is dense over 0..=pos.
    fn indexer(&self, il: usize, w: &MlaW, x: *const f32, qr: *const f32) -> Result<(usize, bool)> {
        let c = &self.cfg;
        let (dk, kp) = (c.idx_head_dim, c.idx_kpool);
        let LayerState::Mla { idx, pooled, .. } = &self.states[il] else { unreachable!() };
        let pos = self.pos;
        let row = idx.f32().wrapping_add(pos * 2 * dk);
        let ktmp = self.small[0].f32();
        self.mv(&w.ik, x, ktmp)?;
        ck!(ffi::gs_layernorm(ktmp, w.ik_norm_w.f32(), w.ik_norm_b.f32(), row, 1, dk as i32, 1e-6, self.s.raw()));
        self.mv(&w.igate, x, row.wrapping_add(dk))?;
        if (pos + 1) % kp == 0 {
            let p = pos / kp;
            let first = idx.f32().wrapping_add(p * kp * 2 * dk);
            ck!(ffi::gs_idx_pool(first, w.iape.f32(), pooled.f32().wrapping_add(p * dk), dk as i32, kp as i32, self.s.raw()));
        }
        let n_pool = (pos + 1) / kp;
        let max_pools = c.idx_top_k / kp;
        if n_pool <= max_pools {
            return Ok((pos + 1, false)); // every complete pool + the tail = all tokens: dense
        }
        // sparse: score pools, take the top max_pools on the host, append the tail
        let iq = self.tp(6);
        self.mv(&w.iq_b, qr, iq)?;
        let iw = self.small[1].f32();
        self.mv(&w.iproj, x, iw)?;
        ck!(ffi::gs_idx_score(
            iq, iw, pooled.f32(), n_pool as i32, self.pool_scores.f32(), c.idx_n_head as i32, dk as i32,
            1.0 / (dk as f32).sqrt(), self.s.raw()
        ));
        let mut scores = vec![0f32; n_pool];
        self.pool_scores.download(0, bytes_of_mut(&mut scores), &self.s)?;
        self.s.sync()?;
        let mut order: Vec<u32> = (0..n_pool as u32).collect();
        order.select_nth_unstable_by(max_pools - 1, |a, b| scores[*b as usize].total_cmp(&scores[*a as usize]));
        let mut sel: Vec<i32> = Vec::with_capacity(max_pools * kp + kp);
        for &p in &order[..max_pools] {
            for m in 0..kp {
                sel.push((p as usize * kp + m) as i32);
            }
        }
        let tail = (pos + 1) % kp; // incomplete pool: always attended
        for t in (pos + 1 - tail)..(pos + 1) {
            sel.push(t as i32);
        }
        self.sel_idx.upload(0, bytes_of(&sel), &self.s)?;
        Ok((sel.len(), true))
    }

    fn mla(&self, il: usize, w: &MlaW, x: *const f32, out: *mut f32) -> Result<()> {
        let c = &self.cfg;
        let (h, dq, r) = (c.n_head, c.mla_head_k, c.kv_lora_rank);
        let LayerState::Mla { latent, .. } = &self.states[il] else { unreachable!() };
        if self.pos >= self.ctx {
            bail!("context full ({} tokens)", self.ctx);
        }
        let (qa, qr, q) = (self.tp(0), self.tp(1), self.tp(2));
        self.mv(&w.q_a, x, qa)?;
        self.rms(qa, Some(&w.q_a_norm), qr, 1, c.q_lora_rank)?;
        self.mv(&w.q_b, qr, q)?;
        // latent for this position, written straight into the cache
        let kv = self.tp(3);
        self.mv(&w.kv_a, x, kv)?;
        let cache_row = latent.f32().wrapping_add(self.pos * r);
        self.rms(kv, Some(&w.kv_a_norm), cache_row, 1, r)?;
        // absorb k_b into q: per head q_abs[512] = Wkb[h] (512 x 256) . q[h]
        ck!(ffi::gs_gemv_batched(
            w.k_b.ty.id() as i32, w.k_b.ptr, w.k_b.row_bytes as i64, w.k_b.d0 as i32, w.k_b.d1 as i32, q, self.qabs.f32(),
            h as i32, w.k_b.matrix_bytes() as i64, dq as i64, r as i64, self.s.raw()
        ));
        let (n_sel, sparse) = self.indexer(il, w, x, qr)?;
        let idx = if sparse { self.sel_idx.ptr() as *const i32 } else { std::ptr::null() };
        ck!(ffi::gs_mla_attn(
            self.qabs.f32(), latent.f32(), idx, n_sel as i32, 1.0 / (dq as f32).sqrt(), self.att_scratch.f32(),
            self.olat.f32(), h as i32, r as i32, self.s.raw()
        ));
        // per head v[256] = Wvb[h] (256 x 512) . o_lat[h]
        let ov = self.tp(4);
        ck!(ffi::gs_gemv_batched(
            w.v_b.ty.id() as i32, w.v_b.ptr, w.v_b.row_bytes as i64, w.v_b.d0 as i32, w.v_b.d1 as i32, self.olat.f32(), ov,
            h as i32, w.v_b.matrix_bytes() as i64, r as i64, c.mla_head_v as i64, self.s.raw()
        ));
        self.dbg("kqv_out", il as isize, ov, h * c.mla_head_v)?;
        self.mv(&w.o, ov, out)
    }

    // ---- FFN ----------------------------------------------------------------------------------------
    fn swiglu_ffn(&self, gate: &DTensor, up: &DTensor, down: &DTensor, x: *const f32, out: *mut f32, lim: f32) -> Result<()> {
        let (g, u) = (self.tp(5), self.tp(6));
        self.mv(gate, x, g)?;
        self.mv(up, x, u)?;
        ck!(ffi::gs_swiglu(g, u, g, gate.d1 as i64, lim, self.s.raw()));
        self.mv(down, g, out)
    }

    fn moe(&mut self, il: usize, x: *const f32, out: *mut f32) -> Result<()> {
        let c = &self.cfg;
        let Ffn::Moe { router, bias, experts, sh_gate, sh_up, sh_down } = &self.layers[il].ffn else { unreachable!() };
        let lim_e = self.layers[il].clamp_exp;
        let lim_s = self.layers[il].clamp_shexp;
        // routing (noaux_tc, sigmoid scores; bias only affects the choice)
        self.mv(router, x, self.router.f32())?;
        let mut logits = vec![0f32; c.n_expert];
        self.router.download(0, bytes_of_mut(&mut logits), &self.s)?;
        self.s.sync()?;
        let scores: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
        let mut order: Vec<usize> = (0..c.n_expert).collect();
        order.sort_by(|a, b| (scores[*b] + bias[*b]).total_cmp(&(scores[*a] + bias[*a])));
        let chosen = &order[..c.n_expert_used];
        let mut wsum: f32 = chosen.iter().map(|e| scores[*e]).sum();
        if !c.expert_weights_norm {
            wsum = 1.0;
        }
        let weights: Vec<f32> = chosen.iter().map(|e| scores[*e] / (wsum + 1e-20) * c.expert_weights_scale).collect();
        if self.trace {
            eprintln!("  [{:>4}] experts-{il}: {:?} w={:?}", self.pos, chosen, weights.iter().map(|w| (w * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        }

        // read the chosen experts from disk in parallel into pinned memory
        let t0 = Instant::now();
        let slot = self.exp_slot;
        let files = &self.files;
        let parts = [&experts.gate, &experts.up, &experts.down];
        let mut offs = [0usize; 3];
        let mut acc = 0usize;
        for (i, t) in parts.iter().enumerate() {
            offs[i] = acc;
            acc = (acc + (t.n_bytes / t.dims[2]) as usize).next_multiple_of(4096);
        }
        let host = self.exp_host.as_mut_slice();
        std::thread::scope(|sc| -> Result<()> {
            let mut hs = Vec::new();
            for (buf, &e) in host.chunks_mut(slot).zip(chosen.iter()) {
                hs.push(sc.spawn(move || -> std::io::Result<()> {
                    for (i, t) in parts.iter().enumerate() {
                        let (off, len) = t.slice_range(e as u64);
                        read_at(&files[t.shard], off, &mut buf[offs[i]..offs[i] + len as usize])?;
                    }
                    Ok(())
                }));
            }
            for h in hs {
                h.join().map_err(|_| anyhow::anyhow!("expert reader panicked"))??;
            }
            Ok(())
        })?;
        self.timings.expert_read += t0.elapsed();
        self.timings.expert_bytes += (acc * chosen.len()) as u64;
        let n_used = chosen.len();
        self.exp_dev.upload(0, &self.exp_host.as_slice()[..slot * n_used], &self.s)?;

        // run them
        let n = c.n_embd;
        self.moe_acc.zero(&self.s)?;
        let (g, u, eo) = (self.tp(5), self.tp(6), self.tp(7));
        for (k, &wgt) in weights.iter().enumerate() {
            let base = unsafe { (self.exp_dev.ptr() as *mut u8).add(k * slot) };
            let mat = |i: usize, t: &TensorInfo| DTensor {
                ptr: unsafe { base.add(offs[i]) } as _,
                ty: t.ty,
                d0: t.dims[0] as usize,
                d1: t.dims[1] as usize,
                d2: 1,
                row_bytes: t.ty.row_bytes(t.dims[0]) as usize,
            };
            let (wg, wu, wd) = (mat(0, &experts.gate), mat(1, &experts.up), mat(2, &experts.down));
            self.mv(&wg, x, g)?;
            self.mv(&wu, x, u)?;
            ck!(ffi::gs_swiglu(g, u, g, wg.d1 as i64, lim_e, self.s.raw()));
            self.mv(&wd, g, eo)?;
            ck!(ffi::gs_axpy(self.moe_acc.f32(), eo, wgt, n as i64, self.s.raw()));
        }
        self.dbg("ffn_moe_out", il as isize, self.moe_acc.f32(), n)?;
        self.swiglu_ffn(sh_gate, sh_up, sh_down, x, out, lim_s)?;
        self.dbg("ffn_shexp", il as isize, out, n)?;
        ck!(ffi::gs_axpy(out, self.moe_acc.f32(), 1.0, n as i64, self.s.raw()));
        Ok(())
    }

    /// Run one token. Returns logits if `want_logits`.
    pub fn step(&mut self, tok: u32, want_logits: bool) -> Result<Option<Vec<f32>>> {
        let t0 = Instant::now();
        let n = self.cfg.n_embd;
        let emb = self.embed(tok)?;
        self.x.upload(0, bytes_of(&emb), &self.s)?;
        ck!(ffi::gs_hc_expand(self.x.f32(), self.streams.f32(), 1, n as i32, self.s.raw()));
        self.dbg("inp_embd", -1, self.x.f32(), n)?;

        for il in 0..self.cfg.n_layer {
            let layer = &self.layers[il];
            // attention block
            self.hc_pre(&layer.hc_attn, self.x.f32())?;
            self.rms(self.x.f32(), Some(&layer.attn_norm), self.xn.f32(), 1, n)?;
            self.dbg("attn_norm", il as isize, self.xn.f32(), n)?;
            match &layer.attn {
                Attn::Kda(w) => self.kda(il, w, self.xn.f32(), self.out.f32())?,
                Attn::Mla(w) => self.mla(il, w, self.xn.f32(), self.out.f32())?,
            }
            self.dbg("attn_out", il as isize, self.out.f32(), n)?;
            self.hc_post(self.out.f32())?;
            // FFN block
            let layer = &self.layers[il];
            self.hc_pre(&layer.hc_ffn, self.x.f32())?;
            self.rms(self.x.f32(), Some(&layer.ffn_norm), self.xn.f32(), 1, n)?;
            self.dbg("ffn_norm", il as isize, self.xn.f32(), n)?;
            match &layer.ffn {
                Ffn::Dense { gate, up, down } => {
                    self.swiglu_ffn(gate, up, down, self.xn.f32(), self.out.f32(), layer.clamp_shexp)?
                }
                Ffn::Moe { .. } => self.moe(il, self.xn.f32(), self.out.f32())?,
            }
            self.dbg("ffn_out", il as isize, self.out.f32(), n)?;
            self.hc_post(self.out.f32())?;
            self.dbg("l_out", il as isize, self.streams.f32(), self.cfg.hc * n)?;
        }

        let result = if want_logits {
            ck!(ffi::gs_hc_mean(self.streams.f32(), self.x.f32(), 1, n as i32, self.s.raw()));
            self.rms(self.x.f32(), Some(&self.output_norm), self.xn.f32(), 1, n)?;
            self.dbg("result_norm", -1, self.xn.f32(), n)?;
            self.mv(&self.output, self.xn.f32(), self.logits.f32())?;
            let mut l = vec![0f32; self.cfg.n_vocab];
            self.logits.download(0, bytes_of_mut(&mut l), &self.s)?;
            self.s.sync()?;
            if self.trace {
                let sum: f64 = l.iter().map(|x| *x as f64).sum();
                eprintln!("  [{:>4}] result_output          sum={sum:.6} [{:.6}, {:.6}, {:.6}]", self.pos, l[0], l[1], l[2]);
            }
            Some(l)
        } else {
            self.s.sync()?;
            None
        };
        self.pos += 1;
        self.timings.tokens += 1;
        self.timings.total += t0.elapsed();
        Ok(result)
    }
}
