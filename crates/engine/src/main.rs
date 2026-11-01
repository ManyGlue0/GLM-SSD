mod chat;
mod config;
mod experts;
mod forward;
mod io;
mod model;
mod plan;
mod qtest;
mod refcheck;
mod server;
mod session;
mod tokenizer;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::{json, Value as Json};
use std::io::{BufRead, Write};
use std::path::PathBuf;

/// Absolute paths into this repository, so the installed binary works from any folder
/// (override with -m / GLMSSD_MODEL and --profile).
const DEFAULT_MODEL: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf");
const DEFAULT_PROFILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/hot-experts.txt");

#[derive(Parser)]
#[command(name = "glm-ssd", version, about = "SSD-streaming inference engine for GLM-5.3-Flash")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct ModelArgs {
    /// Any shard of the GGUF model
    #[arg(short, long, env = "GLMSSD_MODEL", default_value = DEFAULT_MODEL)]
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

/// Everything needed to start the engine.
#[derive(clap::Args, Clone)]
struct EngineArgs {
    /// Any shard of the GGUF model
    #[arg(short, long, env = "GLMSSD_MODEL", default_value = DEFAULT_MODEL)]
    model: PathBuf,
    /// Total VRAM the engine may use: weights + caches + scratch; the rest becomes the expert cache (e.g. 22G)
    #[arg(long)]
    vram_budget: String,
    /// RAM for the pinned expert cache + prompt-cache snapshots (e.g. 16G)
    #[arg(long)]
    ram_budget: String,
    /// Max context length (tokens)
    #[arg(long, default_value_t = 32768)]
    ctx: usize,
    /// Prefill chunk (tokens per batched forward; bigger = faster prefill, more VRAM)
    #[arg(long, default_value_t = 512)]
    chunk: usize,
    /// Enable MTP speculative decoding (draft 1 token with the NextN layer, verify in 1 pass)
    #[arg(long)]
    mtp: bool,
    /// SSD reader threads
    #[arg(long, default_value_t = 8)]
    io_threads: usize,
    /// Learned hot-expert profile (read at start for cache priorities, updated on exit)
    #[arg(long, default_value = DEFAULT_PROFILE)]
    profile: PathBuf,
    /// Don't read or write the hot-expert profile
    #[arg(long)]
    no_profile: bool,
    /// Fill the expert caches from the profile at start-up
    #[arg(long)]
    preload: bool,
    /// Read experts through the OS file cache instead of unbuffered direct I/O
    #[arg(long)]
    buffered_io: bool,
    /// Max prompt-cache snapshots kept in RAM (~150 MB each)
    #[arg(long, default_value_t = 6)]
    snapshots: usize,
    /// Context (MLA latent) cache format: f16, or q8 (47% smaller, tiny precision loss)
    #[arg(long, default_value = "f16", value_parser = ["f16", "q8"])]
    kv_cache: String,
    /// Decode: replace an uncached routed expert (rank 3-8) by a cached one whose routing score is
    /// within this margin (0 = exact routing; trades a little quality for fewer SSD reads)
    #[arg(long, default_value_t = 0.0)]
    cache_aware: f32,
    /// Decode: prefetch this many of the next layer's predicted experts (0 = off)
    #[arg(long, default_value_t = 2)]
    prefetch: usize,
}

#[derive(clap::Args, Clone)]
struct SampleArgs {
    /// Sampling temperature (0 = greedy)
    #[arg(long, default_value_t = 0.7)]
    temp: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,
    /// RNG seed (0 = random)
    #[arg(long, default_value_t = 0)]
    seed: u64,
}
impl SampleArgs {
    fn cfg(&self) -> session::SamplerCfg {
        session::SamplerCfg { temperature: self.temp, top_p: self.top_p, top_k: self.top_k, min_p: self.min_p, seed: self.seed }
    }
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
    /// Benchmark the quantized GEMV / GEMM kernels
    Qbench,
    /// Run a prompt (token ids or text) through the model with greedy decoding (validation)
    Run {
        #[command(flatten)]
        eng: EngineArgs,
        /// Comma-separated prompt token ids
        #[arg(long, conflicts_with = "prompt")]
        tokens: Option<String>,
        /// Prompt text (special tokens like <|user|> are parsed)
        #[arg(long)]
        prompt: Option<String>,
        /// Read the prompt text from a file
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        /// Tokens to generate (greedy)
        #[arg(short = 'n', long, default_value_t = 16)]
        n_predict: usize,
        /// Feed the prompt one token at a time (compare with the batched path)
        #[arg(long)]
        sequential: bool,
        /// Print per-layer tensor stats (compare with llama-eval-callback)
        #[arg(long)]
        trace: bool,
        /// Print the top-k logits for every predicted position
        #[arg(long, default_value_t = 5)]
        top: usize,
    },
    /// Interactive chat in the terminal
    Chat {
        #[command(flatten)]
        eng: EngineArgs,
        #[command(flatten)]
        sample: SampleArgs,
        /// System prompt
        #[arg(long)]
        system: Option<String>,
        /// Reasoning effort: low | high | max
        #[arg(long)]
        effort: Option<String>,
        /// Hide the model's reasoning
        #[arg(long)]
        hide_think: bool,
        /// Max tokens per answer
        #[arg(long, default_value_t = 8192)]
        max_tokens: usize,
    },
    /// OpenAI-compatible HTTP server
    Serve {
        #[command(flatten)]
        eng: EngineArgs,
        #[command(flatten)]
        sample: SampleArgs,
        #[arg(long, default_value = "127.0.0.1:8080")]
        addr: String,
        /// Model id reported by /v1/models
        #[arg(long, default_value = "glm-5.3-flash")]
        name: String,
        /// Default max tokens when the request has none
        #[arg(long, default_value_t = 16384)]
        max_tokens: usize,
    },
    /// Decode benchmark: chat prompt -> N generated tokens (greedy), steady-state speed
    Bench {
        #[command(flatten)]
        eng: EngineArgs,
        /// User message
        #[arg(long, default_value = "Write a detailed essay about the history of the printing press.")]
        message: String,
        #[arg(short = 'n', long, default_value_t = 200)]
        n_gen: usize,
    },
    /// Quality + speed under decode conditions: prefill the first tokens of a text, then feed the
    /// rest one token at a time and report the perplexity of each next token
    Ppl {
        #[command(flatten)]
        eng: EngineArgs,
        /// Text file
        #[arg(long)]
        file: PathBuf,
        /// Tokens prefilled before measuring
        #[arg(long, default_value_t = 64)]
        warm: usize,
        /// Tokens scored in decode mode
        #[arg(short = 'n', long, default_value_t = 300)]
        n: usize,
    },
    /// f64 CPU reference of layer 0 (KDA + mHC) for one token, to arbitrate GPU vs llama.cpp
    RefL0 {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long, default_value_t = 785)]
        token: u32,
    },
    /// Tokenize text with the model's tokenizer (only needs the first shard)
    Tokenize {
        #[arg(short, long, env = "GLMSSD_MODEL", default_value = DEFAULT_MODEL)]
        model: PathBuf,
        /// Text to encode; special tokens like <|user|> are parsed
        text: String,
    },
    /// Render a chat with the model's template (messages as JSON) and tokenize it
    Template {
        #[arg(short, long, env = "GLMSSD_MODEL", default_value = DEFAULT_MODEL)]
        model: PathBuf,
        /// JSON array of messages
        messages: String,
    },
    /// Benchmark direct (unbuffered) random reads of expert-sized blocks from a file
    Iobench {
        file: PathBuf,
        /// Read size in MiB (one expert is ~7.2 MiB)
        #[arg(long, default_value_t = 7.2)]
        mib: f64,
        /// Seconds per thread count
        #[arg(long, default_value_t = 4.0)]
        secs: f64,
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
        Cmd::Iobench { file, mib, secs } => io::bench(&file, (mib * 1048576.0) as usize, secs)?,
        Cmd::Tokenize { model, text } => {
            let g = gguf::Gguf::open_shard(&model)?;
            let v = tokenizer::Vocab::from_gguf(&g)?;
            let ids = v.encode(&text, true);
            println!("{} tokens: {:?}", ids.len(), ids);
            let pieces: Vec<String> = ids.iter().map(|&i| format!("{:?}", v.decode(&[i]))).collect();
            println!("{}", pieces.join(" | "));
            println!("roundtrip ok: {}", v.decode(&ids) == text);
        }
        Cmd::Template { model, messages } => {
            let g = gguf::Gguf::open_shard(&model)?;
            let t = chat::ChatTemplate::from_gguf(&g)?;
            let m: Json = serde_json::from_str(&messages)?;
            let s = t.render(&m, None, None, true)?;
            println!("{s}");
            let v = tokenizer::Vocab::from_gguf(&g)?;
            println!("--- {} tokens", v.encode(&s, true).len());
        }
        Cmd::RefL0 { model, token } => {
            let g = gguf::Gguf::open(&model.model)?;
            let cfg = config::Config::from_gguf(&g)?;
            refcheck::layer0(&g, &cfg, token)?;
        }
        Cmd::Run { eng, tokens, prompt, prompt_file, n_predict, sequential, trace, top } => {
            let prompt = match prompt_file {
                Some(f) => Some(std::fs::read_to_string(&f).with_context(|| format!("reading {}", f.display()))?),
                None => prompt,
            };
            run(&eng, tokens.as_deref(), prompt.as_deref(), n_predict, sequential, trace, top)?
        }
        Cmd::Bench { eng, message, n_gen } => bench(&eng, &message, n_gen)?,
        Cmd::Ppl { eng, file, warm, n } => ppl(&eng, &file, warm, n)?,
        Cmd::Chat { eng, sample, system, effort, hide_think, max_tokens } => {
            chat_repl(&eng, &sample, system, effort, hide_think, max_tokens)?
        }
        Cmd::Serve { eng, sample, addr, name, max_tokens } => {
            let (g, session) = start(&eng)?;
            let srv = server::Server {
                vocab: tokenizer::Vocab::from_gguf(&g)?,
                tmpl: chat::ChatTemplate::from_gguf(&g)?,
                session,
                model_name: name,
                default_max_tokens: max_tokens,
                sampler: sample.cfg(),
            };
            server::serve(&addr, srv)?;
        }
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

/// Open the model and start the engine + session.
fn start(a: &EngineArgs) -> Result<(gguf::Gguf, session::Session)> {
    let g = gguf::Gguf::open(&a.model)?;
    let cfg = config::Config::from_gguf(&g)?;
    let vram = plan::parse_size(&a.vram_budget)?;
    let ram = plan::parse_size(&a.ram_budget)?;
    let snap_bytes = (a.snapshots as u64) * (150 << 20);
    let ram_cache = ram.saturating_sub((1u64 << 30) + snap_bytes);
    let opts = forward::EngineOpts {
        ctx: a.ctx,
        chunk: a.chunk,
        mtp: a.mtp,
        kv_q8: a.kv_cache == "q8",
        cache_aware: a.cache_aware,
        cache: experts::CacheConfig {
            prefetch: a.prefetch,
            vram_bytes: vram,
            ram_bytes: ram_cache,
            io_threads: a.io_threads,
            direct_io: !a.buffered_io,
            profile: if a.no_profile { None } else { Some(a.profile.clone()) },
        },
    };
    let t0 = std::time::Instant::now();
    let mut eng = forward::Engine::new(&g, cfg, opts)?;
    let (nv, nr, vb, rb) = eng.experts.capacity();
    eprintln!(
        "engine ready in {:.1}s: ctx {} | chunk {} | MTP {} | expert cache: {nv} VRAM slots ({:.1} GiB) + {nr} RAM slots ({:.1} GiB)",
        t0.elapsed().as_secs_f64(),
        eng.ctx,
        eng.chunk,
        if eng.has_mtp() { "on" } else { "off" },
        plan::gib(vb),
        plan::gib(rb)
    );
    if a.preload {
        eng.preload()?;
    }
    Ok((g, session::Session::new(eng, a.snapshots)))
}

fn run(a: &EngineArgs, tokens: Option<&str>, text: Option<&str>, n_predict: usize, sequential: bool, trace: bool, top: usize) -> Result<()> {
    let (g, mut sess) = start(a)?;
    let vocab = tokenizer::Vocab::from_gguf(&g)?;
    let prompt: Vec<u32> = match (tokens, text) {
        (Some(t), _) => t.split(',').map(|t| t.trim().parse()).collect::<Result<_, _>>()?,
        (None, Some(p)) => vocab.encode(&p.replace("\\n", "\n"), true),
        _ => anyhow::bail!("give --tokens or --prompt"),
    };
    let eng = &mut sess.eng;
    eng.trace = trace;
    let shown: String = vocab.decode(&prompt).chars().take(300).collect();
    println!("prompt ({} tokens): {:?}{}", prompt.len(), shown, if shown.len() < vocab.decode(&prompt).len() { " ..." } else { "" });
    let t0 = std::time::Instant::now();
    let mut logits = if sequential {
        let mut l = Vec::new();
        for (i, &t) in prompt.iter().enumerate() {
            let rows = if i + 1 == prompt.len() { forward::Rows::Last } else { forward::Rows::None };
            if let Some(x) = eng.forward(&[t], rows, false)?.into_iter().next() {
                l = x;
            }
        }
        l
    } else {
        eng.forward(&prompt, forward::Rows::Last, false)?.into_iter().next().unwrap()
    };
    let prompt_time = t0.elapsed();
    eprintln!("prefill: {}", eng.experts.summary());
    let mut out = Vec::new();
    let t1 = std::time::Instant::now();
    for _ in 0..n_predict {
        let l = &logits;
        let k = top.max(1);
        let mut idx: Vec<usize> = (0..l.len()).collect();
        idx.select_nth_unstable_by(k - 1, |a, b| l[*b].total_cmp(&l[*a]));
        let mut best: Vec<usize> = idx[..k].to_vec();
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
        logits = eng.forward(&[next], forward::Rows::Last, false)?.into_iter().next().unwrap();
    }
    let gen_time = t1.elapsed();
    println!("\noutput: {:?}", vocab.decode(&out));
    println!("tokens: {out:?}");
    println!(
        "prompt {:.1}s ({:.2} tok/s) | generation {:.2} tok/s | MoE time {:.1}s",
        prompt_time.as_secs_f64(),
        prompt.len() as f64 / prompt_time.as_secs_f64(),
        out.len() as f64 / gen_time.as_secs_f64(),
        eng.timings.moe.as_secs_f64()
    );
    println!("{}", eng.experts.summary());
    eng.experts.save_profile()?;
    Ok(())
}

fn chat_repl(a: &EngineArgs, sample: &SampleArgs, system: Option<String>, effort: Option<String>, hide_think: bool, max_tokens: usize) -> Result<()> {
    let (g, mut sess) = start(a)?;
    let vocab = tokenizer::Vocab::from_gguf(&g)?;
    let tmpl = chat::ChatTemplate::from_gguf(&g)?;
    let mut sampler = session::Sampler::new(sample.cfg());
    let mut messages: Vec<Json> = Vec::new();
    if let Some(s) = &system {
        messages.push(json!({"role": "system", "content": s}));
    }
    let mut effort = effort;
    let mut show_think = !hide_think;
    eprintln!("commands: /exit  /reset  /stats  /effort low|high|max  /think (toggle reasoning display); end a line with \\ to continue it");
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("\n\x1b[1m> \x1b[0m");
        std::io::stdout().flush()?;
        let mut input = String::new();
        loop {
            let Some(line) = lines.next() else { return finish(&mut sess) };
            let line = line?;
            if let Some(l) = line.strip_suffix('\\') {
                input.push_str(l);
                input.push('\n');
            } else {
                input.push_str(&line);
                break;
            }
        }
        let input = input.trim().to_string();
        match input.as_str() {
            "" => continue,
            "/exit" | "/quit" => return finish(&mut sess),
            "/reset" => {
                messages.retain(|m| m["role"] == "system");
                println!("(conversation cleared)");
                continue;
            }
            "/stats" => {
                println!("{}", sess.eng.experts.summary());
                continue;
            }
            "/think" => {
                show_think = !show_think;
                println!("(reasoning display {})", if show_think { "on" } else { "off" });
                continue;
            }
            s if s.starts_with("/effort") => {
                effort = s.split_whitespace().nth(1).map(String::from);
                println!("(reasoning effort: {})", effort.as_deref().unwrap_or("default"));
                continue;
            }
            _ => {}
        }
        messages.push(json!({"role": "user", "content": input}));
        let prompt = tmpl.render(&Json::Array(messages.clone()), None, effort.as_deref(), true)?;
        let toks = vocab.encode(&prompt, true);
        let mut stats = session::GenStats::default();
        let logits = sess.prefill(&toks, &mut stats, |i, n| {
            if n > 256 {
                eprint!("\r\x1b[2mprefill {i}/{n}\x1b[0m");
            }
        })?;
        if toks.len() > 256 {
            eprint!("\r\x1b[2K");
        }
        let in_think = prompt.trim_end().ends_with("<think>");
        let mut parser = chat::StreamParser::new(&vocab, in_think);
        let eos = sess.eng.cfg.eos.clone();
        let mut out = std::io::stdout();
        let mut was_think = false;
        if in_think && show_think {
            print!("\x1b[2m");
            was_think = true;
        }
        sess.generate(logits, max_tokens, &mut sampler, &mut stats, |tok| {
            if eos.contains(&tok) {
                return false;
            }
            match parser.push(tok) {
                chat::Piece::Reasoning(t) => {
                    if show_think {
                        let _ = write!(out, "{t}");
                    }
                }
                chat::Piece::Content(t) => {
                    if was_think {
                        let _ = write!(out, "\x1b[0m\n");
                        was_think = false;
                    }
                    let _ = write!(out, "{t}");
                }
                chat::Piece::Tool(t) => {
                    let _ = write!(out, "{t}");
                }
                chat::Piece::None => {}
            }
            let _ = out.flush();
            true
        })?;
        println!("\x1b[0m");
        let (reasoning, content, _) = chat::parse_output(&parser.text, in_think);
        let mut msg = json!({"role": "assistant", "content": content});
        if !reasoning.is_empty() {
            msg["reasoning_content"] = json!(reasoning);
        }
        messages.push(msg);
        eprintln!("\x1b[2m[{} | {}]\x1b[0m", stats.line(), sess.eng.experts.summary());
        sess.eng.experts.save_profile().context("saving the hot-expert profile")?;
    }
}

fn finish(sess: &mut session::Session) -> Result<()> {
    sess.eng.experts.save_profile()?;
    Ok(())
}

fn bench(a: &EngineArgs, message: &str, n_gen: usize) -> Result<()> {
    let (g, mut sess) = start(a)?;
    let vocab = tokenizer::Vocab::from_gguf(&g)?;
    let tmpl = chat::ChatTemplate::from_gguf(&g)?;
    let prompt = tmpl.render(&json!([{"role": "user", "content": message}]), None, Some("low"), true)?;
    let toks = vocab.encode(&prompt, true);
    let mut stats = session::GenStats::default();
    let logits = sess.prefill(&toks, &mut stats, |_, _| {})?;
    let mut sampler = session::Sampler::new(session::SamplerCfg { temperature: 0.0, ..Default::default() });
    let mut times = Vec::new();
    let t0 = std::time::Instant::now();
    let mut text = Vec::new();
    let eos = sess.eng.cfg.eos.clone();
    let mut snap = None;
    sess.generate(logits, n_gen, &mut sampler, &mut stats, |tok| {
        times.push(t0.elapsed().as_secs_f64());
        text.push(tok);
        if times.len() == n_gen / 2 {
            snap = Some(std::time::Instant::now());
        }
        !eos.contains(&tok)
    })?;
    let n = times.len();
    let half = n / 2;
    let steady = if n > 4 { (n - half) as f64 / (times[n - 1] - times[half - 1]).max(1e-9) } else { 0.0 };
    println!("{}", vocab.decode(&text).chars().take(400).collect::<String>());
    println!("
{}", stats.line());
    println!("steady-state decode (2nd half): {steady:.2} tok/s");
    println!("{}", sess.eng.experts.summary());
    let _ = snap;
    sess.eng.experts.save_profile()?;
    Ok(())
}

fn ppl(a: &EngineArgs, file: &std::path::Path, warm: usize, n: usize) -> Result<()> {
    let (g, mut sess) = start(a)?;
    let vocab = tokenizer::Vocab::from_gguf(&g)?;
    let text = std::fs::read_to_string(file)?;
    let mut toks = vocab.encode("[gMASK]<sop>", true);
    toks.extend(vocab.encode(&text, false));
    anyhow::ensure!(toks.len() > warm + 1, "text too short ({} tokens)", toks.len());
    let end = (warm + n).min(toks.len() - 1);
    let eng = &mut sess.eng;
    let mut logits = eng.forward(&toks[..warm], forward::Rows::Last, false)?.pop().unwrap();
    let s0 = eng.experts.stats.clone();
    let t0 = std::time::Instant::now();
    let (mut nll, mut top1) = (0f64, 0usize);
    for i in warm..=end {
        let target = toks[i] as usize;
        let mx = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let lse = mx as f64 + logits.iter().map(|x| ((x - mx) as f64).exp()).sum::<f64>().ln();
        nll += lse - logits[target] as f64;
        top1 += (session::Sampler::greedy(&logits) as usize == target) as usize;
        if i == end {
            break;
        }
        logits = eng.forward(&toks[i..i + 1], forward::Rows::Last, false)?.pop().unwrap();
    }
    let scored = end - warm + 1;
    let dt = t0.elapsed().as_secs_f64();
    let s = &eng.experts.stats;
    let uses = (s.uses - s0.uses).max(1) as f64;
    println!(
        "ppl {:.4} | top-1 {:.1}% | {} tokens in {:.1}s = {:.2} tok/s | VRAM {:.1}% RAM {:.1}% SSD {:.1}% | {:.1} GB read | swaps {}",
        (nll / scored as f64).exp(),
        100.0 * top1 as f64 / scored as f64,
        scored,
        dt,
        (scored - 1) as f64 / dt,
        100.0 * (s.hit_vram - s0.hit_vram) as f64 / uses,
        100.0 * (s.hit_ram - s0.hit_ram) as f64 / uses,
        100.0 * (s.miss - s0.miss) as f64 / uses,
        (s.read_bytes - s0.read_bytes) as f64 / 1e9,
        eng.swaps
    );
    Ok(())
}
