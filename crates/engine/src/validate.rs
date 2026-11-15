//! Load-time validation of the model hyper-parameters and tensor shapes (audit C1 / C2).
//!
//! The CUDA kernels trust the sizes the host hands them: K / rows / heads come from the tensor
//! headers while the scratch buffers are sized from the [`Config`]. A model file whose metadata or
//! tensor shapes disagree with what `forward.rs` assumes would make the kernels read or write past
//! device buffers. Everything here runs once, before anything is allocated, so a malformed model
//! is an error message instead of GPU memory corruption.
//!
//! Each rule mirrors one use in `forward.rs` (the `mm()` input width K, the output scratch the
//! rows go to, norm / bias vector lengths, the fixed sizes some kernels are written for).

use crate::config::{AttnKind, Config};
use anyhow::{bail, ensure, Context, Result};
use gguf::{GgmlType, Gguf};
use std::collections::HashMap;

/// Floats per token row of the small scratch buffers s1 / s2 / s3.
pub const SMALL: usize = 2048;
/// Floats per token of the mHC mixing weights (hc = 4: 4 pre + 4 post + 16 residual mix).
pub const HCW: usize = 24;
/// Upper bound for the model widths, so every buffer-size product fits easily in 64 bits.
const MAX_DIM: usize = 1 << 24;

/// Floats per token row of the wide scratch buffers a / b / c / d.
pub fn wide(c: &Config) -> usize {
    16384usize.max(c.n_ff).max(c.n_head * c.mla_head_k).max(2 * c.n_head * c.kda_head_dim)
}

/// Ranges of the hyper-parameters, including the sizes some kernels are written for.
pub fn validate_config(c: &Config) -> Result<()> {
    let n_total = c.n_layer_all();
    ensure!(c.n_layer >= 1 && n_total <= 65535, "layer count {n_total} out of range 1..=65535");
    ensure!(c.attn.len() == n_total, "{} attention kinds for {n_total} layers", c.attn.len());
    ensure!(c.n_dense_lead <= n_total, "leading_dense_block_count {} > {n_total} layers", c.n_dense_lead);
    ensure!(
        c.swiglu_clamp_exp.len() >= n_total && c.swiglu_clamp_shexp.len() >= n_total,
        "swiglu clamp arrays have {} / {} entries for {n_total} layers",
        c.swiglu_clamp_exp.len(),
        c.swiglu_clamp_shexp.len()
    );
    for (name, v) in [
        ("embedding_length", c.n_embd),
        ("vocab_size", c.n_vocab),
        ("feed_forward_length", c.n_ff),
        ("expert_feed_forward_length", c.n_ff_exp),
    ] {
        ensure!((1..=MAX_DIM).contains(&v), "{name} = {v} out of range 1..={MAX_DIM}");
    }
    // GEMM / GEMV inputs must be multiples of 32 floats
    ensure!(c.n_embd % 32 == 0, "embedding_length {} is not a multiple of 32", c.n_embd);
    ensure!(c.n_ff_exp % 32 == 0, "expert_feed_forward_length {} is not a multiple of 32", c.n_ff_exp);
    // MLA attention kernel: heads in groups of 16, partial buffer for <= 64 heads, R == 512
    ensure!(c.n_head >= 16 && c.n_head % 16 == 0 && c.n_head <= 64, "head_count {} unsupported (16, 32, 48 or 64)", c.n_head);
    ensure!(c.kv_lora_rank == 512, "kv_lora_rank {} unsupported (512)", c.kv_lora_rank);
    ensure!(
        (1..=1024).contains(&c.mla_head_k) && c.mla_head_k % 32 == 0 && (1..=1024).contains(&c.mla_head_v),
        "MLA head sizes {} / {} unsupported",
        c.mla_head_k,
        c.mla_head_v
    );
    ensure!(
        (32..=SMALL).contains(&c.q_lora_rank) && c.q_lora_rank % 32 == 0,
        "q_lora_rank {} unsupported (multiple of 32, <= {SMALL})",
        c.q_lora_rank
    );
    // KDA scan is written for D = 128, the conv scan for kernels of 2..=8
    ensure!(c.kda_head_dim == 128, "kda.head_dim {} unsupported (128)", c.kda_head_dim);
    ensure!((2..=8).contains(&c.kda_conv), "ssm.conv_kernel {} unsupported (2..=8)", c.kda_conv);
    // DSA indexer
    ensure!((1..=128).contains(&c.idx_head_dim), "indexer key_length {} unsupported (<= 128)", c.idx_head_dim);
    ensure!(
        c.idx_n_head >= 1 && c.idx_n_head <= SMALL && c.idx_n_head * c.idx_head_dim <= 12288,
        "indexer head_count {} x key_length {} unsupported",
        c.idx_n_head,
        c.idx_head_dim
    );
    ensure!((1..=4096).contains(&c.idx_kpool), "indexer kpool {} out of range 1..=4096", c.idx_kpool);
    ensure!(
        c.idx_top_k >= c.idx_kpool && c.idx_top_k <= 1 << 20,
        "indexer top_k {} out of range {}..={}",
        c.idx_top_k,
        c.idx_kpool,
        1 << 20
    );
    // mHC: 4 streams (24 mixing weights per token), bounded Sinkhorn loop
    ensure!(c.hc == 4, "hyper_connection.count {} unsupported (4)", c.hc);
    ensure!(c.hc_sinkhorn_iters <= 64, "hyper_connection.sinkhorn_iterations {} > 64", c.hc_sinkhorn_iters);
    // MoE: expert ids are u16
    ensure!(
        c.n_expert_used >= 1 && c.n_expert_used <= c.n_expert && c.n_expert <= 65535,
        "expert counts {} used of {} out of range (1 <= used <= count <= 65535)",
        c.n_expert_used,
        c.n_expert
    );
    ensure!(c.rms_eps.is_finite() && c.rms_eps >= 0.0, "invalid rms epsilon {}", c.rms_eps);
    ensure!(c.n_head * c.mla_head_v <= wide(c), "MLA output width {} too large", c.n_head * c.mla_head_v);
    Ok(())
}

/// (name, type, dims) of one tensor.
pub type Shape<'a> = (&'a str, GgmlType, &'a [u64]);

/// Config + shapes of every tensor the forward pass uses (layers `0..n_layer`, plus the first MTP
/// layer and head when `with_mtp`).
pub fn validate_model(g: &Gguf, cfg: &Config, with_mtp: bool) -> Result<()> {
    validate_config(cfg).context("invalid model hyper-parameters")?;
    let table: Vec<Shape> = g.tensors.iter().map(|t| (t.name.as_str(), t.ty, t.dims.as_slice())).collect();
    validate_shapes(&table, cfg, with_mtp).context("model tensors do not match its hyper-parameters")
}

/// Expected size of one dimension.
#[derive(Clone, Copy, Debug)]
enum Dim {
    Is(usize),
    AtMost(usize),
}
use Dim::{AtMost, Is};

struct V<'a> {
    map: HashMap<&'a str, (GgmlType, &'a [u64])>,
}

impl V<'_> {
    fn get(&self, name: &str) -> Result<(GgmlType, &[u64])> {
        self.map.get(name).copied().with_context(|| format!("missing tensor {name}"))
    }
    fn dim(name: &str, what: &str, have: u64, want: Dim) -> Result<usize> {
        let ok = match want {
            Is(n) => have == n as u64,
            AtMost(n) => have >= 1 && have <= n as u64,
        };
        ensure!(ok, "{name}: {what} is {have}, expected {want:?}");
        Ok(have as usize)
    }
    /// 2-D matrix used through mm(): [K, rows]. Returns rows.
    fn mat(&self, name: &str, k: Dim, rows: Dim) -> Result<usize> {
        let (_, d) = self.get(name)?;
        ensure!(d.len() == 2, "{name}: expected a 2-D matrix, dims {d:?}");
        let kk = Self::dim(name, "K (input width)", d[0], k)?;
        ensure!(kk % 32 == 0, "{name}: K = {kk} is not a multiple of 32");
        Self::dim(name, "rows (output width)", d[1], rows)
    }
    /// Per-head matrices [K, rows, heads] used through mm_heads().
    fn heads(&self, name: &str, k: usize, rows: usize, heads: usize) -> Result<()> {
        let (_, d) = self.get(name)?;
        ensure!(d == [k as u64, rows as u64, heads as u64], "{name}: dims {d:?}, expected [{k}, {rows}, {heads}]");
        ensure!(k % 32 == 0, "{name}: K = {k} is not a multiple of 32");
        Ok(())
    }
    /// F32 tensor with exactly these dims (norm weights, biases, small parameters).
    fn f32(&self, name: &str, dims: &[usize]) -> Result<()> {
        let (ty, d) = self.get(name)?;
        ensure!(ty == GgmlType::F32, "{name}: type {ty:?}, expected F32");
        ensure!(d.len() == dims.len() && d.iter().zip(dims).all(|(a, &b)| *a == b as u64), "{name}: dims {d:?}, expected {dims:?}");
        Ok(())
    }
    /// Quantized expert tensor [K, rows, n_expert].
    fn experts(&self, name: &str, k: usize, rows: usize, ne: usize) -> Result<()> {
        let (_, d) = self.get(name)?;
        ensure!(d == [k as u64, rows as u64, ne as u64], "{name}: dims {d:?}, expected [{k}, {rows}, {ne}]");
        Ok(())
    }
}

pub fn validate_shapes(tensors: &[Shape], c: &Config, with_mtp: bool) -> Result<()> {
    let v = V { map: tensors.iter().map(|&(n, t, d)| (n, (t, d))).collect() };
    let n = c.n_embd;
    let wide = wide(c);
    let (h, d, di) = (c.n_head, c.kda_head_dim, c.n_head * c.kda_head_dim);
    let (r, qlr, dk) = (c.kv_lora_rank, c.q_lora_rank, c.idx_head_dim);
    let ne = c.n_expert;

    // embedding rows / logits: token ids < n_vocab index token_embd rows
    v.mat("token_embd.weight", Is(n), Is(c.n_vocab))?;
    v.mat("output.weight", Is(n), Is(c.n_vocab))?;
    v.f32("output_norm.weight", &[n])?;

    let with_mtp = with_mtp && c.n_layer_nextn > 0;
    let n_all = if with_mtp { c.n_layer + 1 } else { c.n_layer };
    for il in 0..n_all {
        let p = |s: &str| format!("blk.{il}.{s}");
        let res: Result<()> = (|| {
            v.f32(&p("attn_norm.weight"), &[n])?;
            v.f32(&p("ffn_norm.weight"), &[n])?;
            if il < c.n_layer {
                for hp in ["attn", "ffn"] {
                    // rms over the hc streams -> mixes [t][24]
                    v.mat(&p(&format!("hc_{hp}_fn.weight")), Is(c.hc * n), Is(HCW))?;
                    v.f32(&p(&format!("hc_{hp}_scale.weight")), &[3])?;
                    v.f32(&p(&format!("hc_{hp}_base.weight")), &[HCW])?;
                }
            }
            match c.attn[il] {
                AttnKind::Kda => {
                    // q/k/v -> a/b/c [t][di]; conv + scan over di; f/g low rank via s1/s3; beta -> s2
                    for x in ["attn_q", "attn_k", "attn_v"] {
                        v.mat(&p(&format!("{x}.weight")), Is(n), Is(di))?;
                    }
                    for x in ["q", "k", "v"] {
                        v.f32(&p(&format!("ssm_conv1d_{x}.weight")), &[c.kda_conv, 1, di])?;
                    }
                    let fr = v.mat(&p("ssm_f_a.weight"), Is(n), AtMost(SMALL))?;
                    v.mat(&p("ssm_f_b.weight"), Is(fr), Is(di))?;
                    v.f32(&p("ssm_dt.bias"), &[di])?;
                    v.f32(&p("ssm_a"), &[h])?;
                    v.mat(&p("ssm_beta.weight"), Is(n), Is(h))?;
                    let gr = v.mat(&p("ssm_g_a.weight"), Is(n), AtMost(SMALL))?;
                    v.mat(&p("ssm_g_b.weight"), Is(gr), Is(di))?;
                    v.f32(&p("ssm_norm.weight"), &[d])?;
                    v.mat(&p("attn_output.weight"), Is(di), Is(n))?;
                }
                AttnKind::Mla => {
                    // q_a -> s1, q_b -> a (wide), kv_a -> s3, k_b / v_b per head, o from d (wide)
                    v.mat(&p("attn_q_a.weight"), Is(n), Is(qlr))?;
                    v.f32(&p("attn_q_a_norm.weight"), &[qlr])?;
                    v.mat(&p("attn_q_b.weight"), Is(qlr), Is(h * c.mla_head_k))?;
                    v.mat(&p("attn_kv_a_mqa.weight"), Is(n), Is(r))?;
                    v.f32(&p("attn_kv_a_norm.weight"), &[r])?;
                    v.heads(&p("attn_k_b.weight"), c.mla_head_k, r, h)?;
                    v.heads(&p("attn_v_b.weight"), r, c.mla_head_v, h)?;
                    v.mat(&p("attn_output.weight"), Is(h * c.mla_head_v), Is(n))?;
                    // indexer: keys / gates [t][dk] in b, queries [t][heads*dk] in c, weights in s1
                    v.mat(&p("indexer.attn_k.weight"), Is(n), Is(dk))?;
                    v.f32(&p("indexer.k_norm.weight"), &[dk])?;
                    v.f32(&p("indexer.k_norm.bias"), &[dk])?;
                    v.mat(&p("indexer_compressor_gate.weight"), Is(n), Is(dk))?;
                    v.mat(&p("indexer.attn_q_b.weight"), Is(qlr), Is(c.idx_n_head * dk))?;
                    v.mat(&p("indexer.proj.weight"), Is(n), Is(c.idx_n_head))?;
                    v.f32(&p("indexer_compressor_ape.weight"), &[dk, c.idx_kpool])?;
                }
            }
            if c.is_moe(il) {
                // router -> [t][n_expert]; shared expert through a/b (wide); experts through eg/eu
                v.mat(&p("ffn_gate_inp.weight"), Is(n), Is(ne))?;
                v.f32(&p("exp_probs_b.bias"), &[ne])?;
                let fs = v.mat(&p("ffn_gate_shexp.weight"), Is(n), AtMost(wide))?;
                v.mat(&p("ffn_up_shexp.weight"), Is(n), Is(fs))?;
                v.mat(&p("ffn_down_shexp.weight"), Is(fs), Is(n))?;
                v.experts(&p("ffn_gate_exps.weight"), n, c.n_ff_exp, ne)?;
                v.experts(&p("ffn_up_exps.weight"), n, c.n_ff_exp, ne)?;
                v.experts(&p("ffn_down_exps.weight"), c.n_ff_exp, n, ne)?;
                // the fused gate/up kernel decodes both with the gate tensor's type and row size
                let (tg, tu) = (v.get(&p("ffn_gate_exps.weight"))?.0, v.get(&p("ffn_up_exps.weight"))?.0);
                ensure!(tg == tu, "{}: gate experts are {tg:?} but up experts are {tu:?} (must match)", p("ffn_up_exps.weight"));
            } else {
                v.mat(&p("ffn_gate.weight"), Is(n), Is(c.n_ff))?;
                v.mat(&p("ffn_up.weight"), Is(n), Is(c.n_ff))?;
                v.mat(&p("ffn_down.weight"), Is(c.n_ff), Is(n))?;
            }
            if with_mtp && il == c.n_layer {
                ensure!(c.attn[il] == AttnKind::Mla, "the MTP layer must use MLA attention");
                // x = eh[:, :n] e + eh[:, n:] h: the halves start at row_bytes(n)
                let eh = p("nextn.eh_proj.weight");
                v.mat(&eh, Is(2 * n), Is(n))?;
                let (ty, _) = v.get(&eh)?;
                if ty.try_row_bytes(n as u64).is_none() {
                    bail!("{eh}: half row of {n} is not a whole number of {ty:?} blocks");
                }
                v.f32(&p("nextn.enorm.weight"), &[n])?;
                v.f32(&p("nextn.hnorm.weight"), &[n])?;
                let hn = p("nextn.shared_head_norm.weight");
                if v.map.contains_key(hn.as_str()) {
                    v.f32(&hn, &[n])?;
                }
            }
            Ok(())
        })();
        res.with_context(|| format!("layer {il}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use GgmlType::*;

    /// Real GLM 5.3 Flash hyper-parameters, shrunk to 3 trunk layers (KDA dense, KDA MoE,
    /// MLA MoE) + 1 MTP layer.
    fn cfg() -> Config {
        Config {
            n_layer: 3,
            n_layer_nextn: 1,
            n_embd: 4096,
            n_vocab: 154880,
            n_head: 64,
            attn: vec![AttnKind::Kda, AttnKind::Kda, AttnKind::Mla, AttnKind::Mla],
            rms_eps: 1e-5,
            n_ff: 12288,
            n_ff_exp: 2048,
            n_expert: 288,
            n_expert_used: 8,
            n_expert_shared: 1,
            n_dense_lead: 1,
            expert_weights_scale: 2.5,
            expert_weights_norm: true,
            swiglu_clamp_exp: vec![0.0; 4],
            swiglu_clamp_shexp: vec![0.0; 4],
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            mla_head_k: 256,
            mla_head_v: 256,
            kda_head_dim: 128,
            kda_conv: 4,
            kda_gate_lower_bound: -5.0,
            idx_n_head: 32,
            idx_head_dim: 128,
            idx_top_k: 2048,
            idx_kpool: 4,
            hc: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            eos: vec![],
        }
    }

    /// The real model's shapes (from `glm-ssd inspect --tensors`), per layer kind.
    fn table() -> Vec<(String, GgmlType, Vec<u64>)> {
        let mut t: Vec<(String, GgmlType, Vec<u64>)> = vec![
            ("token_embd.weight".into(), Q4_K, vec![4096, 154880]),
            ("output.weight".into(), Q4_K, vec![4096, 154880]),
            ("output_norm.weight".into(), F32, vec![4096]),
        ];
        let mut add = |il: usize, list: &[(&str, GgmlType, &[u64])]| {
            for (n, ty, d) in list {
                t.push((format!("blk.{il}.{n}"), *ty, d.to_vec()));
            }
        };
        let common: &[(&str, GgmlType, &[u64])] =
            &[("attn_norm.weight", F32, &[4096]), ("ffn_norm.weight", F32, &[4096])];
        let hc: &[(&str, GgmlType, &[u64])] = &[
            ("hc_attn_fn.weight", Q8_0, &[16384, 24]),
            ("hc_attn_scale.weight", F32, &[3]),
            ("hc_attn_base.weight", F32, &[24]),
            ("hc_ffn_fn.weight", Q8_0, &[16384, 24]),
            ("hc_ffn_scale.weight", F32, &[3]),
            ("hc_ffn_base.weight", F32, &[24]),
        ];
        let kda: &[(&str, GgmlType, &[u64])] = &[
            ("attn_q.weight", Q5_K, &[4096, 8192]),
            ("attn_k.weight", Q5_K, &[4096, 8192]),
            ("attn_v.weight", Q5_K, &[4096, 8192]),
            ("attn_output.weight", Q5_K, &[8192, 4096]),
            ("ssm_a", F32, &[64]),
            ("ssm_beta.weight", Q8_0, &[4096, 64]),
            ("ssm_conv1d_q.weight", F32, &[4, 1, 8192]),
            ("ssm_conv1d_k.weight", F32, &[4, 1, 8192]),
            ("ssm_conv1d_v.weight", F32, &[4, 1, 8192]),
            ("ssm_dt.bias", F32, &[8192]),
            ("ssm_f_a.weight", Q8_0, &[4096, 128]),
            ("ssm_f_b.weight", Q8_0, &[128, 8192]),
            ("ssm_g_a.weight", Q8_0, &[4096, 128]),
            ("ssm_g_b.weight", Q8_0, &[128, 8192]),
            ("ssm_norm.weight", F32, &[128]),
        ];
        let mla: &[(&str, GgmlType, &[u64])] = &[
            ("attn_q_a.weight", Q5_K, &[4096, 1536]),
            ("attn_q_a_norm.weight", F32, &[1536]),
            ("attn_q_b.weight", Q8_0, &[1536, 16384]),
            ("attn_kv_a_mqa.weight", Q8_0, &[4096, 512]),
            ("attn_kv_a_norm.weight", F32, &[512]),
            ("attn_k_b.weight", Q8_0, &[256, 512, 64]),
            ("attn_v_b.weight", Q8_0, &[512, 256, 64]),
            ("attn_output.weight", Q6_K, &[16384, 4096]),
            ("indexer.attn_k.weight", Q8_0, &[4096, 128]),
            ("indexer.attn_q_b.weight", Q8_0, &[1536, 4096]),
            ("indexer.k_norm.weight", F32, &[128]),
            ("indexer.k_norm.bias", F32, &[128]),
            ("indexer.proj.weight", F32, &[4096, 32]),
            ("indexer_compressor_gate.weight", Q8_0, &[4096, 128]),
            ("indexer_compressor_ape.weight", F32, &[128, 4]),
        ];
        let dense: &[(&str, GgmlType, &[u64])] = &[
            ("ffn_gate.weight", Q5_K, &[4096, 12288]),
            ("ffn_up.weight", Q5_K, &[4096, 12288]),
            ("ffn_down.weight", Q6_K, &[12288, 4096]),
        ];
        let moe: &[(&str, GgmlType, &[u64])] = &[
            ("ffn_gate_inp.weight", F32, &[4096, 288]),
            ("exp_probs_b.bias", F32, &[288]),
            ("ffn_gate_shexp.weight", Q5_K, &[4096, 2048]),
            ("ffn_up_shexp.weight", Q5_K, &[4096, 2048]),
            ("ffn_down_shexp.weight", Q6_K, &[2048, 4096]),
            ("ffn_gate_exps.weight", IQ2_XXS, &[4096, 2048, 288]),
            ("ffn_up_exps.weight", IQ2_XXS, &[4096, 2048, 288]),
            ("ffn_down_exps.weight", IQ3_XXS, &[2048, 4096, 288]),
        ];
        let nextn: &[(&str, GgmlType, &[u64])] = &[
            ("nextn.eh_proj.weight", Q8_0, &[8192, 4096]),
            ("nextn.enorm.weight", F32, &[4096]),
            ("nextn.hnorm.weight", F32, &[4096]),
            ("nextn.shared_head_norm.weight", F32, &[4096]),
        ];
        for (il, parts) in [
            (0, vec![common, hc, kda, dense]),
            (1, vec![common, hc, kda, moe]),
            (2, vec![common, hc, mla, moe]),
            (3, vec![common, mla, moe, nextn]),
        ] {
            for p in parts {
                add(il, p);
            }
        }
        t
    }

    fn check(t: &[(String, GgmlType, Vec<u64>)], c: &Config, mtp: bool) -> Result<()> {
        let s: Vec<Shape> = t.iter().map(|(n, ty, d)| (n.as_str(), *ty, d.as_slice())).collect();
        validate_shapes(&s, c, mtp)
    }
    fn set(t: &mut [(String, GgmlType, Vec<u64>)], name: &str, dims: &[u64]) {
        let e = t.iter_mut().find(|e| e.0 == name).unwrap_or_else(|| panic!("{name}"));
        e.2 = dims.to_vec();
    }
    fn rejects(name: &str, dims: &[u64]) -> String {
        let mut t = table();
        set(&mut t, name, dims);
        let e = check(&t, &cfg(), true).expect_err(&format!("{name} {dims:?} must be rejected"));
        format!("{e:#}")
    }

    #[test]
    fn c1_c2_real_model_shapes_pass() {
        validate_config(&cfg()).unwrap();
        check(&table(), &cfg(), true).unwrap();
        check(&table(), &cfg(), false).unwrap();
        // without MTP the MTP layer's tensors are not needed
        let t: Vec<_> = table().into_iter().filter(|e| !e.0.starts_with("blk.3.")).collect();
        check(&t, &cfg(), false).unwrap();
        assert!(check(&t, &cfg(), true).is_err());
    }

    #[test]
    fn c2_wrong_k_is_rejected() {
        assert!(rejects("blk.1.attn_q.weight", &[2048, 8192]).contains("K (input width)"));
        assert!(rejects("blk.0.ssm_f_b.weight", &[256, 8192]).contains("ssm_f_b"));
        assert!(rejects("blk.2.attn_output.weight", &[8192, 4096]).contains("attn_output"));
        assert!(rejects("blk.3.nextn.eh_proj.weight", &[4096, 4096]).contains("eh_proj"));
        assert!(rejects("blk.2.attn_k_b.weight", &[128, 512, 64]).contains("attn_k_b"));
        assert!(rejects("token_embd.weight", &[2048, 154880]).contains("token_embd"));
    }

    #[test]
    fn c2_too_many_rows_is_rejected() {
        assert!(rejects("blk.0.ssm_f_a.weight", &[4096, 4096]).contains("rows"));
        assert!(rejects("blk.0.hc_attn_fn.weight", &[16384, 25]).contains("hc_attn_fn"));
        assert!(rejects("blk.1.ffn_gate_shexp.weight", &[4096, 32768]).contains("rows"));
        assert!(rejects("blk.2.attn_kv_a_mqa.weight", &[4096, 1024]).contains("attn_kv_a_mqa"));
        assert!(rejects("blk.2.indexer.proj.weight", &[4096, 64]).contains("indexer.proj"));
        assert!(rejects("output.weight", &[4096, 154881]).contains("output.weight"));
        assert!(rejects("blk.1.ssm_beta.weight", &[4096, 128]).contains("ssm_beta"));
    }

    #[test]
    fn c2_wrong_expert_count_is_rejected() {
        assert!(rejects("blk.1.exp_probs_b.bias", &[287]).contains("exp_probs_b"));
        assert!(rejects("blk.1.ffn_gate_inp.weight", &[4096, 256]).contains("ffn_gate_inp"));
        assert!(rejects("blk.2.ffn_gate_exps.weight", &[4096, 2048, 287]).contains("ffn_gate_exps"));
        assert!(rejects("blk.2.ffn_down_exps.weight", &[4096, 2048, 288]).contains("ffn_down_exps"));
        assert!(rejects("blk.3.ffn_up_exps.weight", &[4096, 4096, 288]).contains("ffn_up_exps"));
    }

    #[test]
    fn c2_gate_and_up_experts_must_share_a_type() {
        let mut t = table();
        t.iter_mut().find(|e| e.0 == "blk.2.ffn_up_exps.weight").unwrap().1 = IQ2_S;
        let e = format!("{:#}", check(&t, &cfg(), true).unwrap_err());
        assert!(e.contains("must match"), "{e}");
        // down may differ (it has its own kernel call)
        let mut t = table();
        t.iter_mut().find(|e| e.0 == "blk.2.ffn_down_exps.weight").unwrap().1 = IQ4_XS;
        check(&t, &cfg(), true).unwrap();
    }

    #[test]
    fn c2_norm_and_vector_shapes_are_checked() {
        assert!(rejects("blk.0.attn_norm.weight", &[4095]).contains("attn_norm"));
        assert!(rejects("blk.2.attn_q_a_norm.weight", &[2048]).contains("attn_q_a_norm"));
        assert!(rejects("blk.0.ssm_conv1d_q.weight", &[9, 1, 8192]).contains("ssm_conv1d_q"));
        assert!(rejects("blk.0.ssm_a", &[63]).contains("ssm_a"));
        // wrong type for a vector read as f32
        let mut t = table();
        t.iter_mut().find(|e| e.0 == "output_norm.weight").unwrap().1 = F16;
        assert!(format!("{:#}", check(&t, &cfg(), true).unwrap_err()).contains("F32"));
        // missing tensor
        let t: Vec<_> = table().into_iter().filter(|e| e.0 != "blk.2.indexer.k_norm.bias").collect();
        assert!(format!("{:#}", check(&t, &cfg(), true).unwrap_err()).contains("missing tensor"));
    }

    #[test]
    fn c1_bad_config_values_are_rejected() {
        let bad: Vec<(&str, Box<dyn Fn(&mut Config)>)> = vec![
            ("n_head 72", Box::new(|c| c.n_head = 72)),
            ("n_head 40", Box::new(|c| c.n_head = 40)),
            ("kda_head_dim", Box::new(|c| c.kda_head_dim = 64)),
            ("kv_lora_rank", Box::new(|c| c.kv_lora_rank = 256)),
            ("idx_head_dim", Box::new(|c| c.idx_head_dim = 256)),
            ("idx heads", Box::new(|c| c.idx_n_head = 128)),
            ("kda_conv 1", Box::new(|c| c.kda_conv = 1)),
            ("kda_conv 9", Box::new(|c| c.kda_conv = 9)),
            ("hc", Box::new(|c| c.hc = 3)),
            ("kpool 0", Box::new(|c| c.idx_kpool = 0)),
            ("top_k < kpool", Box::new(|c| c.idx_top_k = 2)),
            ("used 0", Box::new(|c| c.n_expert_used = 0)),
            ("used > count", Box::new(|c| c.n_expert_used = 289)),
            ("experts > u16", Box::new(|c| c.n_expert = 70000)),
            ("sinkhorn", Box::new(|c| c.hc_sinkhorn_iters = 1000)),
            ("clamp short", Box::new(|c| c.swiglu_clamp_exp.truncate(2))),
            ("n_embd huge", Box::new(|c| c.n_embd = 1 << 40)),
            ("n_embd 0", Box::new(|c| c.n_embd = 0)),
            ("n_embd % 32", Box::new(|c| c.n_embd = 4100)),
            ("vocab huge", Box::new(|c| c.n_vocab = usize::MAX)),
            ("layers", Box::new(|c| c.n_layer = 70000)),
            ("no trunk", Box::new(|c| c.n_layer = 0)),
            ("attn kinds", Box::new(|c| c.attn.pop().map(|_| ()).unwrap_or(()))),
            ("q_lora_rank", Box::new(|c| c.q_lora_rank = 4096)),
            ("eps", Box::new(|c| c.rms_eps = f32::NAN)),
        ];
        for (what, f) in bad {
            let mut c = cfg();
            f(&mut c);
            assert!(validate_config(&c).is_err(), "{what} must be rejected");
        }
    }
}
