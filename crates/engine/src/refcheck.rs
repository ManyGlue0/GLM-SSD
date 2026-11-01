//! Independent f64 CPU reference for the first layer on the first token (KDA path + mHC).
//!
//! Used to arbitrate when glm-ssd and llama.cpp disagree: llama.cpp quantizes activations to
//! Q8_1 inside its matmuls, glm-ssd keeps them in f32, and this module keeps everything in f64
//! with weights dequantized by the CPU reference (bit-exact with gguf-py).

use crate::config::Config;
use crate::model::read_at;
use anyhow::{Context, Result};
use gguf::Gguf;
use std::fs::File;

struct Ctx<'a> {
    g: &'a Gguf,
    files: Vec<File>,
}

impl Ctx<'_> {
    /// Dequantize a whole tensor to f64.
    fn tensor(&self, name: &str) -> Result<Vec<f64>> {
        let t = self.g.tensor(name).with_context(|| format!("missing tensor {name}"))?;
        let mut raw = vec![0u8; t.n_bytes as usize];
        read_at(&self.files[t.shard], t.offset, &mut raw)?;
        let n: u64 = t.dims.iter().product();
        let mut y = vec![0f32; n as usize];
        quants::dequantize(t.ty, &raw, &mut y)?;
        Ok(y.into_iter().map(f64::from).collect())
    }
    /// y = W x for a GGUF matrix with dims [K, rows].
    fn mv(&self, name: &str, x: &[f64]) -> Result<Vec<f64>> {
        let w = self.tensor(name)?;
        let k = x.len();
        anyhow::ensure!(w.len() % k == 0, "{name}: {} not divisible by K={k}", w.len());
        Ok(w.chunks_exact(k).map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum()).collect())
    }
}

fn rms(x: &[f64], eps: f64) -> Vec<f64> {
    let ms = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().map(|v| v * r).collect()
}
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn show(name: &str, v: &[f64]) {
    let sum: f64 = v.iter().sum();
    println!(
        "  {name:<16} n={:<6} sum={sum:<14.6} [{:.6}, {:.6}, {:.6} ... {:.6}]",
        v.len(), v[0], v[1], v[2], v[v.len() - 1]
    );
}

pub fn layer0(g: &Gguf, cfg: &Config, tok: u32) -> Result<()> {
    let files = g.shards.iter().map(File::open).collect::<std::io::Result<Vec<_>>>()?;
    let c = Ctx { g, files };
    let n = cfg.n_embd;
    let (h, d) = (cfg.n_head, cfg.kda_head_dim);
    let eps = cfg.rms_eps as f64;
    let hc_eps = cfg.hc_eps as f64;

    // embedding row
    let te = g.tensor("token_embd.weight").context("token_embd")?;
    let rb = te.ty.row_bytes(te.dims[0]) as usize;
    let mut raw = vec![0u8; rb];
    read_at(&c.files[te.shard], te.offset + tok as u64 * rb as u64, &mut raw)?;
    let mut e32 = vec![0f32; n];
    quants::dequantize(te.ty, &raw, &mut e32)?;
    let emb: Vec<f64> = e32.into_iter().map(f64::from).collect();
    show("inp_embd", &emb);

    // mHC pre (all 4 streams equal the embedding)
    let flat: Vec<f64> = (0..4).flat_map(|_| emb.iter().copied()).collect();
    let mixes = c.mv("blk.0.hc_attn_fn.weight", &rms(&flat, eps))?;
    let scale = c.tensor("blk.0.hc_attn_scale.weight")?;
    let base = c.tensor("blk.0.hc_attn_base.weight")?;
    show("hc_mixes-0", &mixes);
    let pre: Vec<f64> = (0..4).map(|i| sigmoid(mixes[i] * scale[0] + base[i]) + hc_eps).collect();
    let post: Vec<f64> = (0..4).map(|i| 2.0 * sigmoid(mixes[4 + i] * scale[1] + base[4 + i])).collect();
    let mut comb = [[0f64; 4]; 4]; // [src][dst]
    for s in 0..4 {
        let v: Vec<f64> = (0..4).map(|t| mixes[8 + 4 * s + t] * scale[2] + base[8 + 4 * s + t]).collect();
        let mx = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ex: Vec<f64> = v.iter().map(|x| (x - mx).exp()).collect();
        let sum: f64 = ex.iter().sum();
        for t in 0..4 {
            comb[s][t] = ex[t] / sum + hc_eps;
        }
    }
    let norm_cols = |c: &mut [[f64; 4]; 4]| {
        for t in 0..4 {
            let s: f64 = (0..4).map(|s| c[s][t]).sum::<f64>() + hc_eps;
            for r in c.iter_mut() {
                r[t] /= s;
            }
        }
    };
    let norm_rows = |c: &mut [[f64; 4]; 4]| {
        for r in c.iter_mut() {
            let s: f64 = r.iter().sum::<f64>() + hc_eps;
            r.iter_mut().for_each(|x| *x /= s);
        }
    };
    norm_cols(&mut comb);
    for _ in 1..cfg.hc_sinkhorn_iters {
        norm_rows(&mut comb);
        norm_cols(&mut comb);
    }
    show("hc_pre-0", &pre);
    show("hc_post-0", &post);
    let psum: f64 = pre.iter().sum();
    let x: Vec<f64> = emb.iter().map(|v| v * psum).collect();
    let attn_norm = c.tensor("blk.0.attn_norm.weight")?;
    let xn: Vec<f64> = rms(&x, eps).iter().zip(&attn_norm).map(|(a, b)| a * b).collect();
    show("attn_norm-0", &xn);

    // KDA, first token: conv state is zero, so conv output = silu(w[K-1] * x)
    let kk = cfg.kda_conv;
    let conv = |name: &str, x: &[f64]| -> Result<Vec<f64>> {
        let w = c.tensor(name)?;
        Ok(x.iter().enumerate().map(|(ch, v)| {
            let a = w[ch * kk + kk - 1] * v;
            a * sigmoid(a)
        }).collect())
    };
    let q = conv("blk.0.ssm_conv1d_q.weight", &c.mv("blk.0.attn_q.weight", &xn)?)?;
    let k = conv("blk.0.ssm_conv1d_k.weight", &c.mv("blk.0.attn_k.weight", &xn)?)?;
    let v = conv("blk.0.ssm_conv1d_v.weight", &c.mv("blk.0.attn_v.weight", &xn)?)?;
    let l2 = |x: &[f64], s: f64| -> Vec<f64> {
        x.chunks_exact(d).flat_map(|hd| {
            let r = 1.0 / (hd.iter().map(|v| v * v).sum::<f64>() + 1e-6).sqrt();
            hd.iter().map(move |v| v * r * s).collect::<Vec<_>>()
        }).collect()
    };
    let qn = l2(&q, 1.0 / (d as f64).sqrt());
    let kn = l2(&k, 1.0);
    let beta: Vec<f64> = c.mv("blk.0.ssm_beta.weight", &xn)?.into_iter().map(sigmoid).collect();
    let mut o = vec![0f64; h * d];
    let mut cos = Vec::new();
    for hh in 0..h {
        let r = hh * d..(hh + 1) * d;
        let kq: f64 = kn[r.clone()].iter().zip(&qn[r.clone()]).map(|(a, b)| a * b).sum();
        cos.push(kq * (d as f64).sqrt());
        for j in r {
            o[j] = v[j] * beta[hh] * kq;
        }
    }
    let mut ac: Vec<f64> = cos.iter().map(|x| x.abs()).collect();
    ac.sort_by(|a, b| a.total_cmp(b));
    println!("  |cos(q,k)| per head: min {:.4}  median {:.4}  max {:.4}", ac[0], ac[h / 2], ac[h - 1]);
    show("kda_scan_out-0", &o);

    let gate = c.mv("blk.0.ssm_g_b.weight", &c.mv("blk.0.ssm_g_a.weight", &xn)?)?;
    let onw = c.tensor("blk.0.ssm_norm.weight")?;
    let mut gated = vec![0f64; h * d];
    for hh in 0..h {
        let r = hh * d..(hh + 1) * d;
        let nrm = rms(&o[r.clone()], eps);
        for (i, j) in r.enumerate() {
            gated[j] = nrm[i] * onw[i] * sigmoid(gate[j]);
        }
    }
    let out = c.mv("blk.0.attn_output.weight", &gated)?;
    show("kda_out-0", &out);
    let post_streams: Vec<f64> = (0..4).flat_map(|dst| {
        let cs: f64 = (0..4).map(|s| comb[s][dst]).sum();
        let p = post[dst];
        out.iter().zip(&emb).map(move |(o, e)| p * o + cs * e).collect::<Vec<_>>()
    }).collect();
    show("hc_attn_post-0", &post_streams);
    Ok(())
}
