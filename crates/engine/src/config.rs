//! Hyper-parameters of a `glm5-next` model, read from GGUF metadata.

use anyhow::{bail, Context, Result};
use gguf::{Gguf, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Kimi Delta Attention (linear, recurrent state)
    Kda,
    /// nope-MLA with DSA k-pool sparse indexer
    Mla,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub n_layer: usize,       // trunk layers (45)
    pub n_layer_nextn: usize, // MTP layers appended after the trunk (1)
    pub n_embd: usize,
    pub n_vocab: usize,
    pub n_head: usize,
    pub attn: Vec<AttnKind>, // per layer, incl. MTP layers
    pub rms_eps: f32,
    pub norm_eps: f32,

    // FFN / MoE
    pub n_ff: usize,
    pub n_ff_exp: usize,
    pub n_expert: usize,
    pub n_expert_used: usize,
    pub n_expert_shared: usize,
    pub n_dense_lead: usize,
    pub expert_weights_scale: f32,
    pub expert_weights_norm: bool,
    pub swiglu_clamp_exp: Vec<f32>,
    pub swiglu_clamp_shexp: Vec<f32>,

    // MLA
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub mla_head_k: usize,
    pub mla_head_v: usize,

    // KDA
    pub kda_head_dim: usize,
    pub kda_conv: usize,
    pub kda_gate_lower_bound: f32,

    // DSA indexer
    pub idx_n_head: usize,
    pub idx_head_dim: usize,
    pub idx_top_k: usize,
    pub idx_kpool: usize,

    // mHC
    pub hc: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,

    // tokens
    pub bos: Option<u32>,
    pub eos: Vec<u32>,
}

fn f32_arr(g: &Gguf, key: &str, n: usize, default: f32) -> Vec<f32> {
    match g.get(key) {
        Some(Value::Arr(a)) => a.iter().map(|v| v.as_f64().unwrap_or(default as f64) as f32).collect(),
        Some(v) => vec![v.as_f64().unwrap_or(default as f64) as f32; n],
        None => vec![default; n],
    }
}

impl Config {
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        let arch = g.arch()?;
        if arch != "glm5-next" {
            bail!("unsupported architecture '{arch}' (expected glm5-next)");
        }
        let k = |s: &str| g.arch_key(s);
        let u = |s: &str| g.u64(&k(s)).map(|v| v as usize);
        let f = |s: &str| g.f64(&k(s)).map(|v| v as f32);

        let n_total = u("block_count")?;
        let n_layer_nextn = u("nextn_predict_layers").unwrap_or(0);
        let n_layer = n_total - n_layer_nextn;

        let kv_heads = g.get(&k("attention.head_count_kv")).and_then(Value::as_arr).context("head_count_kv array")?;
        if kv_heads.len() != n_total {
            bail!("head_count_kv has {} entries, expected {n_total}", kv_heads.len());
        }
        let attn = kv_heads
            .iter()
            .map(|v| if v.as_u64() == Some(0) { AttnKind::Kda } else { AttnKind::Mla })
            .collect();

        let tok = |s: &str| g.u64(&format!("tokenizer.ggml.{s}")).ok().map(|v| v as u32);
        let mut eos: Vec<u32> = ["eos_token_id", "eot_token_id", "eom_token_id"].iter().filter_map(|s| tok(s)).collect();
        eos.dedup();

        Ok(Self {
            n_layer,
            n_layer_nextn,
            n_embd: u("embedding_length")?,
            n_vocab: u("vocab_size")?,
            n_head: u("attention.head_count")?,
            attn,
            rms_eps: f("attention.layer_norm_rms_epsilon")?,
            norm_eps: f("attention.layer_norm_epsilon").unwrap_or(1e-6),
            n_ff: u("feed_forward_length")?,
            n_ff_exp: u("expert_feed_forward_length")?,
            n_expert: u("expert_count")?,
            n_expert_used: u("expert_used_count")?,
            n_expert_shared: u("expert_shared_count").unwrap_or(1),
            n_dense_lead: u("leading_dense_block_count").unwrap_or(0),
            expert_weights_scale: f("expert_weights_scale").unwrap_or(1.0),
            expert_weights_norm: g.get(&k("expert_weights_norm")).and_then(Value::as_u64).unwrap_or(0) != 0,
            swiglu_clamp_exp: f32_arr(g, &k("swiglu_clamp_exp"), n_total, 0.0),
            swiglu_clamp_shexp: f32_arr(g, &k("swiglu_clamp_shexp"), n_total, 0.0),
            q_lora_rank: u("attention.q_lora_rank")?,
            kv_lora_rank: u("attention.kv_lora_rank")?,
            mla_head_k: u("attention.key_length_mla")?,
            mla_head_v: u("attention.value_length_mla")?,
            kda_head_dim: u("kda.head_dim")?,
            kda_conv: u("ssm.conv_kernel")?,
            kda_gate_lower_bound: f("kda.gate_lower_bound").unwrap_or(f32::NEG_INFINITY),
            idx_n_head: u("attention.indexer.head_count")?,
            idx_head_dim: u("attention.indexer.key_length")?,
            idx_top_k: u("attention.indexer.top_k")?,
            idx_kpool: u("attention.indexer.kpool")?,
            hc: u("hyper_connection.count")?,
            hc_sinkhorn_iters: u("hyper_connection.sinkhorn_iterations")?,
            hc_eps: f("hyper_connection.epsilon")?,
            bos: tok("bos_token_id"),
            eos,
        })
    }

    pub fn n_layer_all(&self) -> usize {
        self.n_layer + self.n_layer_nextn
    }
    pub fn is_moe(&self, il: usize) -> bool {
        il >= self.n_dense_lead
    }
    pub fn n_mla(&self, with_mtp: bool) -> usize {
        let n = if with_mtp { self.n_layer_all() } else { self.n_layer };
        self.attn[..n].iter().filter(|a| **a == AttnKind::Mla).count()
    }
    pub fn n_kda(&self) -> usize {
        self.attn[..self.n_layer].iter().filter(|a| **a == AttnKind::Kda).count()
    }
}
