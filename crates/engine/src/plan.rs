//! Placement planner: given memory budgets, decide what lives where.
//!
//! - token embedding         -> RAM (row gather on CPU, only 1 row per token is needed)
//! - all other dense weights -> VRAM, resident
//! - routed experts          -> streamed; VRAM and RAM hold exclusive expert caches
//!
//! The VRAM budget also has to cover the KV / recurrent state for the requested context and a
//! compute workspace.

use crate::config::{AttnKind, Config};
use anyhow::{bail, Result};
use gguf::Gguf;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct LayerExperts {
    pub layer: usize,
    pub gate: (usize, gguf::GgmlType), // tensor index, type
    pub up: (usize, gguf::GgmlType),
    pub down: (usize, gguf::GgmlType),
    /// bytes of one expert (gate+up+down slices)
    pub unit_bytes: u64,
}

pub struct Plan {
    pub with_mtp: bool,
    pub ctx: usize,
    pub vram_resident: u64,
    pub ram_resident: u64,
    pub kv_per_token: u64,
    pub kv_bytes: u64,
    pub state_bytes: u64,
    pub workspace: u64,
    pub experts: Vec<LayerExperts>,
    pub expert_total_bytes: u64,
    pub n_expert_units: usize,
    pub vram_expert_bytes: u64,
    pub ram_expert_bytes: u64,
    pub ram_reserve: u64,
    /// resident tensors grouped by a short category, for reporting
    pub categories: BTreeMap<String, u64>,
}

pub struct Budgets {
    pub vram: u64,
    pub ram: u64,
    pub ctx: usize,
    pub workspace: u64,
    pub with_mtp: bool,
}

pub fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

fn category(name: &str) -> String {
    let rest = match layer_of(name) {
        Some(_) => name.splitn(3, '.').nth(2).unwrap_or(name),
        None => name,
    };
    let c = if rest.starts_with("ssm_") || rest == "ssm_a" {
        "kda (ssm_*)"
    } else if rest.starts_with("indexer") {
        "dsa indexer"
    } else if rest.starts_with("hc_") {
        "mHC"
    } else if rest.starts_with("nextn") {
        "mtp head"
    } else if rest.contains("shexp") {
        "shared experts"
    } else if rest.starts_with("ffn_gate_inp") || rest.starts_with("exp_probs") {
        "routers"
    } else if rest.starts_with("ffn_") {
        "dense ffn (layers 0-2)"
    } else if rest.starts_with("attn_") && rest.contains("norm") || rest.ends_with("norm.weight") {
        "norms"
    } else if rest.starts_with("attn_") {
        "attention proj"
    } else if rest.starts_with("output") {
        "lm head"
    } else {
        rest
    };
    c.to_string()
}

impl Plan {
    pub fn build(g: &Gguf, cfg: &Config, b: &Budgets) -> Result<Plan> {
        let n_used_layers = if b.with_mtp { cfg.n_layer_all() } else { cfg.n_layer };
        let mut vram_resident = 0u64;
        let mut ram_resident = 0u64;
        let mut categories = BTreeMap::new();
        let mut exp: BTreeMap<usize, [Option<(usize, gguf::GgmlType, u64)>; 3]> = BTreeMap::new();

        for (ti, t) in g.tensors.iter().enumerate() {
            let layer = layer_of(&t.name);
            if layer.is_some_and(|l| l >= n_used_layers) {
                continue; // MTP layer disabled
            }
            let slot = if t.name.ends_with("ffn_gate_exps.weight") {
                Some(0)
            } else if t.name.ends_with("ffn_up_exps.weight") {
                Some(1)
            } else if t.name.ends_with("ffn_down_exps.weight") {
                Some(2)
            } else {
                None
            };
            if let (Some(s), Some(l)) = (slot, layer) {
                let per = t.n_bytes / t.dims[2];
                exp.entry(l).or_default()[s] = Some((ti, t.ty, per));
                continue;
            }
            if t.name == "token_embd.weight" {
                ram_resident += t.n_bytes;
                *categories.entry("token embedding (RAM)".into()).or_insert(0) += t.n_bytes;
            } else {
                vram_resident += t.n_bytes;
                *categories.entry(category(&t.name)).or_insert(0) += t.n_bytes;
            }
        }

        let mut experts = Vec::new();
        for (l, parts) in exp {
            let [Some(gt), Some(up), Some(dn)] = parts else { bail!("layer {l}: incomplete expert tensors") };
            experts.push(LayerExperts {
                layer: l,
                gate: (gt.0, gt.1),
                up: (up.0, up.1),
                down: (dn.0, dn.1),
                unit_bytes: gt.2 + up.2 + dn.2,
            });
        }
        let expert_total_bytes: u64 = experts.iter().map(|e| e.unit_bytes * cfg.n_expert as u64).sum();
        let n_expert_units = experts.len() * cfg.n_expert;

        // KV cache per token: MLA latent (f16) + indexer row key|gate|pooled (f16) per MLA layer.
        let n_mla = cfg.n_mla(b.with_mtp) as u64;
        let kv_per_token = n_mla * (cfg.kv_lora_rank as u64 * 2 + 3 * cfg.idx_head_dim as u64 * 2);
        let kv_bytes = kv_per_token * b.ctx as u64;
        // KDA: fp32 state [head_dim x head_dim] per head + conv states for q,k,v.
        let d_inner = (cfg.n_head * cfg.kda_head_dim) as u64;
        let state_bytes = cfg.n_kda() as u64
            * (cfg.n_head as u64 * (cfg.kda_head_dim * cfg.kda_head_dim) as u64 * 4 + 3 * d_inner * (cfg.kda_conv as u64 - 1) * 4);

        let vram_fixed = vram_resident + kv_bytes + state_bytes + b.workspace;
        if vram_fixed > b.vram {
            bail!(
                "VRAM budget {:.2} GiB is too small: resident weights {:.2} + KV {:.2} + state {:.2} + workspace {:.2} GiB",
                gib(b.vram), gib(vram_resident), gib(kv_bytes), gib(state_bytes), gib(b.workspace)
            );
        }
        // RAM: embedding + I/O staging + small host buffers.
        let ram_reserve = 1 << 30;
        let ram_fixed = ram_resident + ram_reserve;
        if ram_fixed > b.ram {
            bail!("RAM budget {:.2} GiB is too small (need at least {:.2} GiB)", gib(b.ram), gib(ram_fixed));
        }

        Ok(Plan {
            with_mtp: b.with_mtp,
            ctx: b.ctx,
            vram_resident,
            ram_resident,
            kv_per_token,
            kv_bytes,
            state_bytes,
            workspace: b.workspace,
            experts,
            expert_total_bytes,
            n_expert_units,
            vram_expert_bytes: b.vram - vram_fixed,
            ram_expert_bytes: b.ram - ram_fixed,
            ram_reserve,
            categories,
        })
    }

    pub fn avg_unit(&self) -> f64 {
        self.expert_total_bytes as f64 / self.n_expert_units as f64
    }

    pub fn print(&self, cfg: &Config) {
        println!("\n== Model ==");
        let n_kda = cfg.attn[..cfg.n_layer].iter().filter(|a| **a == AttnKind::Kda).count();
        println!(
            "  {} trunk layers ({} KDA linear-attn, {} nope-MLA+DSA) + {} MTP layer(s){}",
            cfg.n_layer, n_kda, cfg.n_layer - n_kda, cfg.n_layer_nextn,
            if self.with_mtp { "" } else { " [MTP disabled]" }
        );
        println!(
            "  hidden {} | vocab {} | MoE: {} experts, top-{}, +{} shared, ff {} | mHC x{}",
            cfg.n_embd, cfg.n_vocab, cfg.n_expert, cfg.n_expert_used, cfg.n_expert_shared, cfg.n_ff_exp, cfg.hc
        );

        println!("\n== Resident weights ==");
        for (c, b) in &self.categories {
            println!("  {:<26} {:>9.1} MiB", c, *b as f64 / 1048576.0);
        }
        println!("  {:<26} {:>9.2} GiB (VRAM)  + {:.2} GiB (RAM)", "total", gib(self.vram_resident), gib(self.ram_resident));

        println!("\n== Routed experts ==");
        let mut by_size: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
        for e in &self.experts {
            by_size.entry(e.unit_bytes).or_default().push(e.layer);
        }
        for (sz, layers) in &by_size {
            let e = self.experts.iter().find(|e| e.unit_bytes == *sz).unwrap();
            println!(
                "  {:>6.2} MiB/expert  gate {:?} up {:?} down {:?}  layers {}",
                *sz as f64 / 1048576.0, e.gate.1, e.up.1, e.down.1, compact(layers)
            );
        }
        println!(
            "  {} expert units, {:.2} GiB total, avg {:.2} MiB",
            self.n_expert_units, gib(self.expert_total_bytes), self.avg_unit() / 1048576.0
        );
        let per_tok = self.avg_unit() * (self.experts.len().min(cfg.n_layer - cfg.n_dense_lead) * cfg.n_expert_used) as f64;
        println!("  touched per token (trunk): {:.2} GiB", per_tok / (1u64 << 30) as f64);

        println!("\n== Memory plan (ctx {}) ==", self.ctx);
        println!(
            "  VRAM: resident {:.2} + KV {:.2} ({:.1} KiB/token) + KDA state {:.2} + workspace {:.2} -> expert cache {:.2} GiB",
            gib(self.vram_resident), gib(self.kv_bytes), self.kv_per_token as f64 / 1024.0,
            gib(self.state_bytes), gib(self.workspace), gib(self.vram_expert_bytes)
        );
        println!(
            "  RAM:  embedding {:.2} + staging/reserve {:.2} -> expert cache {:.2} GiB",
            gib(self.ram_resident), gib(self.ram_reserve), gib(self.ram_expert_bytes)
        );
        let cached = (self.vram_expert_bytes + self.ram_expert_bytes) as f64;
        let coverage = (cached / self.expert_total_bytes as f64).min(1.0);
        let n_v = (self.vram_expert_bytes as f64 / self.avg_unit()) as u64;
        let n_r = (self.ram_expert_bytes as f64 / self.avg_unit()) as u64;
        println!(
            "  cache slots: ~{n_v} in VRAM + ~{n_r} in RAM = {:.1}% of all experts",
            coverage * 100.0
        );
        let miss_uniform = per_tok * (1.0 - coverage);
        println!(
            "  worst case (uniform routing, no locality): {:.2} GiB from SSD per token -> <= {:.1} tok/s at 6.5 GB/s",
            miss_uniform / (1u64 << 30) as f64,
            if miss_uniform > 0.0 { 6.5e9 / miss_uniform } else { f64::INFINITY }
        );
        println!("  (real routing is skewed + temporally local, so the hit rate should beat the coverage figure)");
    }
}

pub fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

fn compact(v: &[usize]) -> String {
    let mut out = Vec::new();
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
            j += 1;
        }
        out.push(if i == j { format!("{}", v[i]) } else { format!("{}-{}", v[i], v[j]) });
        i = j + 1;
    }
    out.join(",")
}

/// "22G", "22GiB", "512M", "1.5G" -> bytes (binary units).
pub fn parse_size(s: &str) -> Result<u64> {
    let t = s.trim().to_ascii_uppercase();
    let t = t.trim_end_matches("IB").trim_end_matches('B');
    let (num, mul) = match t.chars().last() {
        Some('K') => (&t[..t.len() - 1], 1u64 << 10),
        Some('M') => (&t[..t.len() - 1], 1 << 20),
        Some('G') => (&t[..t.len() - 1], 1 << 30),
        Some('T') => (&t[..t.len() - 1], 1 << 40),
        _ => (t, 1),
    };
    let v: f64 = num.trim().parse().map_err(|_| anyhow::anyhow!("bad size '{s}' (use e.g. 22G, 512M)"))?;
    Ok((v * mul as f64) as u64)
}
