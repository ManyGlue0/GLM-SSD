mod config;
mod forward;
mod model;
mod plan;
mod qtest;
mod tokenizer;

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
        /// Compare the CPU reference with gguf-py vectors in this dir (tools/qref.py)
        #[arg(long)]
        ref_dir: Option<PathBuf>,
    },
    /// Benchmark the quantized GEMV kernels
    Qbench,
    /// Run token ids through the model (M2 validation path): greedy generation + optional trace
    Run {
        #[command(flatten)]
        model: ModelArgs,
        /// Comma-separated prompt token ids
        #[arg(long)]
        tokens: String,
        /// Tokens to generate (greedy)
        #[arg(short = 'n', long, default_value_t = 16)]
        n_predict: usize,
        /// Max context
        #[arg(long, default_value_t = 4096)]
        ctx: usize,
        /// Print per-layer tensor stats (compare with llama-eval-callback)
        #[arg(long)]
        trace: bool,
        /// Print the top-k logits for every predicted position
        #[arg(long, default_value_t = 5)]
        top: usize,
    },
    /// Tokenize text with the model's tokenizer (only needs the first shard)
    Tokenize {
        #[arg(short, long, default_value = DEFAULT_MODEL)]
        model: PathBuf,
        /// Text to encode; special tokens like <|user|> are parsed
        text: String,
    },
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
        Cmd::Qtest { model, ref_dir } => match ref_dir {
            Some(d) => qtest::check_ref_dir(&d)?,
            None => qtest::run(model.as_deref())?,
        },
        Cmd::Qbench => qtest::bench()?,
        Cmd::Tokenize { model, text } => {
            let g = gguf::Gguf::open_shard(&model)?;
            let v = tokenizer::Vocab::from_gguf(&g)?;
            let ids = v.encode(&text, true);
            println!("{} tokens: {:?}", ids.len(), ids);
            let pieces: Vec<String> = ids.iter().map(|&i| format!("{:?}", v.decode(&[i]))).collect();
            println!("{}", pieces.join(" | "));
            println!("roundtrip ok: {}", v.decode(&ids) == text);
        }
        Cmd::Run { model, tokens, n_predict, ctx, trace, top } => run(&model, &tokens, n_predict, ctx, trace, top)?,
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

fn run(m: &ModelArgs, tokens: &str, n_predict: usize, ctx: usize, trace: bool, top: usize) -> Result<()> {
    let g = gguf::Gguf::open(&m.model)?;
    let cfg = config::Config::from_gguf(&g)?;
    let vocab = tokenizer::Vocab::from_gguf(&g)?;
    let prompt: Vec<u32> = tokens.split(',').map(|t| t.trim().parse()).collect::<Result<_, _>>()?;
    let mut eng = forward::Engine::new(&g, cfg, ctx)?;
    eng.trace = trace;
    println!("prompt ({} tokens): {:?}", prompt.len(), vocab.decode(&prompt));

    let t0 = std::time::Instant::now();
    let mut logits = None;
    for (i, &t) in prompt.iter().enumerate() {
        logits = eng.step(t, i + 1 == prompt.len())?;
    }
    let prompt_time = t0.elapsed();
    let mut out = Vec::new();
    let t1 = std::time::Instant::now();
    for _ in 0..n_predict {
        let l = logits.take().unwrap();
        let mut idx: Vec<usize> = (0..l.len()).collect();
        idx.select_nth_unstable_by(top.max(1) - 1, |a, b| l[*b].total_cmp(&l[*a]));
        let mut best: Vec<usize> = idx[..top.max(1)].to_vec();
        best.sort_by(|a, b| l[*b].total_cmp(&l[*a]));
        let next = best[0] as u32;
        if top > 1 {
            let alts: Vec<String> = best.iter().map(|&i| format!("{:?}={:.2}", vocab.decode(&[i as u32]), l[i])).collect();
            println!("  pos {:>4}: {}", eng.pos, alts.join("  "));
        }
        out.push(next);
        if eng.cfg.eos.contains(&next) {
            break;
        }
        logits = eng.step(next, true)?;
    }
    let gen_time = t1.elapsed();
    println!("
output: {:?}", vocab.decode(&out));
    println!("tokens: {out:?}");
    let tm = &eng.timings;
    println!(
        "prompt {:.1}s ({:.2} tok/s) | generation {:.2} tok/s | expert reads {:.1}s, {:.1} GB ({:.2} GB/s)",
        prompt_time.as_secs_f64(),
        prompt.len() as f64 / prompt_time.as_secs_f64(),
        out.len() as f64 / gen_time.as_secs_f64(),
        tm.expert_read.as_secs_f64(),
        tm.expert_bytes as f64 / 1e9,
        tm.expert_bytes as f64 / 1e9 / tm.expert_read.as_secs_f64().max(1e-9)
    );
    Ok(())
}
