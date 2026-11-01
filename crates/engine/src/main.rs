mod config;
mod plan;
mod qtest;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

const DEFAULT_MODEL: &str = "models/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf";

#[derive(Parser)]
#[command(name = "glm-ssd", version, about = "SSD-streaming inference engine for GLM-5.3-Flash")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct ModelArgs {
    /// Any shard of the GGUF model
    #[arg(short, long, default_value = DEFAULT_MODEL)]
    model: PathBuf,
    /// Parse headers only (works on truncated shards, for inspection before download completes)
    #[arg(long)]
    headers_only: bool,
}

#[derive(clap::Args, Clone)]
struct BudgetArgs {
    /// VRAM the engine may use (e.g. 22G). G = GiB.
    #[arg(long)]
    vram_budget: String,
    /// RAM the engine may use (e.g. 24G)
    #[arg(long)]
    ram_budget: String,
    /// Max context length (tokens)
    #[arg(long, default_value_t = 65536)]
    ctx: usize,
    /// GPU compute workspace reserve
    #[arg(long, default_value = "1G")]
    workspace: String,
    /// Don't load the MTP (speculative decoding) layer
    #[arg(long)]
    no_mtp: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show GPU info and run a CUDA smoke test
    Gpu,
    /// Cross-check CUDA quant kernels against the CPU reference
    Qtest {
        /// Also test real tensors from this (fully downloaded) model
        #[arg(short, long)]
        model: Option<PathBuf>,
    },
    /// Benchmark the quantized GEMV kernels
    Qbench,
    /// Print model structure and the memory placement plan for the given budgets
    Inspect {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        budget: BudgetArgs,
        /// Also list every tensor
        #[arg(long)]
        tensors: bool,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Gpu => {
            let d = cuda::device_info(0)?;
            println!(
                "{}  sm_{}{}  {} SMs  {:.2}/{:.2} GiB free  copy engines: {}",
                d.name, d.compute_capability.0, d.compute_capability.1, d.sm_count,
                plan::gib(d.free_mem), plan::gib(d.total_mem), d.async_engines
            );
            cuda::smoke_test()?;
            println!("CUDA smoke test OK");
        }
        Cmd::Qtest { model } => qtest::run(model.as_deref())?,
        Cmd::Qbench => qtest::bench()?,
        Cmd::Inspect { model, budget, tensors } => {
            let g = if model.headers_only {
                gguf::Gguf::open_headers_only(&model.model)?
            } else {
                gguf::Gguf::open(&model.model)?
            };
            let cfg = config::Config::from_gguf(&g)?;
            println!(
                "{}: {} shard(s), {} tensors, {:.2} GB",
                g.str("general.name").unwrap_or("?"),
                g.shards.len(),
                g.tensors.len(),
                g.tensors.iter().map(|t| t.n_bytes).sum::<u64>() as f64 / 1e9
            );
            if tensors {
                for t in &g.tensors {
                    println!("  {:<44} {:<8?} {:?} shard {} @ {}", t.name, t.ty, t.dims, t.shard, t.offset);
                }
            }
            let b = plan::Budgets {
                vram: plan::parse_size(&budget.vram_budget)?,
                ram: plan::parse_size(&budget.ram_budget)?,
                ctx: budget.ctx,
                workspace: plan::parse_size(&budget.workspace)?,
                with_mtp: !budget.no_mtp,
            };
            let p = plan::Plan::build(&g, &cfg, &b)?;
            p.print(&cfg);
        }
    }
    Ok(())
}
