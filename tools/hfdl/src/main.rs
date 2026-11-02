//! hfdl: download files from a Hugging Face repo as fast as the link allows.
//!
//! - Splits every file into fixed-size chunks fetched with HTTP Range requests over
//!   N independent connections (HTTP/1.1, one TCP stream per request).
//! - Resolves the signed CDN URL once per file and refreshes it when it expires (401/403/410)
//!   or gets old, so we don't hammer huggingface.co with one redirect per chunk.
//! - Resumable: progress is tracked per chunk in `<file>.part.json` next to `<file>.part`.
//! - Verifies the LFS SHA-256 of each file as soon as it completes (while others download).

use anyhow::{anyhow, bail, Context, Result};
use clap::Parser;
use futures_util::StreamExt;
use globset::{Glob, GlobSetBuilder};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const HF: &str = "https://huggingface.co";
const DEFAULT_REPO: &str = "unsloth/GLM-5.3-Flash-GGUF";
const URL_MAX_AGE: Duration = Duration::from_secs(20 * 60);
const FLUSH_BYTES: usize = 8 << 20;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Repository id (owner/name) or a Hugging Face URL
    #[arg(long, default_value = DEFAULT_REPO)]
    repo: String,
    /// Hugging Face URL of a repo, folder or file, e.g.
    /// https://huggingface.co/owner/name/tree/main/UD-IQ2_XXS (sets --repo, --revision, --include)
    #[arg(long, conflicts_with = "repo")]
    url: Option<String>,
    /// Branch / tag / commit
    #[arg(long, default_value = "main")]
    revision: String,
    /// Glob(s) matched against repo paths (repeatable). Default: UD-IQ2_XXS/* for the official
    /// repo; for other repos the GGUF model is picked automatically when there is only one
    /// (or only one IQ2_XXS), otherwise the choices are listed
    #[arg(long)]
    include: Vec<String>,
    /// Output directory (repo folder structure is preserved inside it). Default: models/ for the
    /// official repo, models/<owner>--<name>/ for others
    #[arg(long)]
    out: Option<PathBuf>,
    /// Parallel connections
    #[arg(short = 'c', long, default_value_t = 16)]
    connections: usize,
    /// Chunk size in MiB
    #[arg(long, default_value_t = 64)]
    chunk_mb: u64,
    /// Skip SHA-256 verification
    #[arg(long)]
    no_verify: bool,
    /// Only list what would be downloaded
    #[arg(long)]
    dry_run: bool,
}

#[derive(Deserialize, Debug)]
struct TreeEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    lfs: Option<Lfs>,
}
#[derive(Deserialize, Debug)]
struct Lfs {
    oid: String,
}

#[derive(Serialize, Deserialize)]
struct PartState {
    size: u64,
    sha256: Option<String>,
    chunk_size: u64,
    done: Vec<bool>,
}

struct FileJob {
    repo_path: String,
    size: u64,
    sha256: Option<String>,
    final_path: PathBuf,
    part_path: PathBuf,
    state_path: PathBuf,
    file: File,
    state: Mutex<PartState>,
    last_save: Mutex<Instant>,
    remaining: AtomicU64, // chunks still to do
    url: tokio::sync::Mutex<Option<(String, Instant)>>,
}

struct Chunk {
    job: Arc<FileJob>,
    idx: usize,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());

    let api = reqwest::Client::builder()
        .user_agent("hfdl/0.1")
        .connect_timeout(Duration::from_secs(20))
        .build()?;
    // CDN client: no redirects (we resolve ourselves), many idle connections kept alive.
    let cdn = reqwest::Client::builder()
        .user_agent("hfdl/0.1")
        .redirect(reqwest::redirect::Policy::none())
        .connect_timeout(Duration::from_secs(20))
        .read_timeout(Duration::from_secs(45))
        .pool_max_idle_per_host(args.connections * 2)
        .tcp_nodelay(true)
        .build()?;

    // ---- list files --------------------------------------------------------------------
    let mut args = args;
    let link = args.url.take().or_else(|| {
        let r = &args.repo;
        (r.contains("://") || r.starts_with("huggingface.co/") || r.starts_with("hf.co/")).then(|| r.clone())
    });
    if let Some(u) = link {
        let (repo, rev, path) = parse_url(&u)?;
        args.repo = repo;
        if let Some(r) = rev {
            args.revision = r;
        }
        if let (Some(p), true) = (path, args.include.is_empty()) {
            args.include.push(p);
        }
    }
    if args.repo.split('/').filter(|s| !s.is_empty()).count() != 2 {
        bail!("--repo must look like owner/name or be a Hugging Face URL (got {:?})", args.repo);
    }
    let official = args.repo.eq_ignore_ascii_case(DEFAULT_REPO);
    // Other repos get their own folder: their files often have the same names as the official
    // ones (UD-IQ2_XXS/...) and would otherwise overwrite them.
    let out = args.out.clone().unwrap_or_else(|| {
        if official { PathBuf::from("models") } else { Path::new("models").join(args.repo.replace('/', "--")) }
    });
    let entries = list_tree(&api, &args.repo, &args.revision, token.as_deref()).await?;
    let entries: Vec<TreeEntry> = entries.into_iter().filter(|e| e.kind == "file").collect();
    if args.include.is_empty() {
        args.include = if official { vec!["UD-IQ2_XXS/*".to_string()] } else { pick_model(&args.repo, &entries)? };
    }
    let mut gs = GlobSetBuilder::new();
    for g in &args.include {
        gs.add(Glob::new(g).with_context(|| format!("bad glob {g}"))?);
    }
    let gs = gs.build()?;
    let mut files: Vec<TreeEntry> = entries.into_iter().filter(|e| gs.is_match(&e.path)).collect();
    if files.is_empty() {
        bail!("no files in {} match {:?}", args.repo, args.include);
    }
    files.sort_by_key(|e| e.size); // small files first -> metadata available early

    let total: u64 = files.iter().map(|f| f.size).sum();
    println!(
        "{} @ {}: {} file(s), {:.2} GB total -> {}",
        args.repo,
        args.revision,
        files.len(),
        total as f64 / 1e9,
        out.display()
    );
    let first = shard_one(&files.iter().map(|f| f.path.clone()).collect::<Vec<_>>());
    for f in &files {
        println!("  {:>10.3} GB  {}", f.size as f64 / 1e9, f.path);
    }
    if args.dry_run {
        return Ok(());
    }

    // ---- prepare jobs / resume state ---------------------------------------------------
    let chunk_size = args.chunk_mb.max(1) << 20;
    let mut queue = VecDeque::new();
    let mut jobs = Vec::new();
    let mut already = 0u64;
    for e in files {
        let final_path = out.join(&e.path);
        if let Some(p) = final_path.parent() {
            std::fs::create_dir_all(p)?;
        }
        if std::fs::metadata(&final_path).map(|m| m.len() == e.size).unwrap_or(false) {
            println!("  skip (complete): {}", e.path);
            already += e.size;
            continue;
        }
        let part_path = with_suffix(&final_path, ".part");
        let state_path = with_suffix(&final_path, ".part.json");
        let n_chunks = e.size.div_ceil(chunk_size).max(1) as usize;
        let sha = e.lfs.as_ref().map(|l| l.oid.clone());

        let mut state = std::fs::read(&state_path)
            .ok()
            .and_then(|b| serde_json::from_slice::<PartState>(&b).ok())
            .filter(|s| s.size == e.size && s.sha256 == sha && s.done.len() as u64 == s.size.div_ceil(s.chunk_size).max(1))
            .filter(|_| part_path.exists());
        if state.is_none() {
            let _ = std::fs::remove_file(&part_path);
        }
        let state = state.take().unwrap_or(PartState { size: e.size, sha256: sha.clone(), chunk_size, done: vec![false; n_chunks] });

        let file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&part_path)
            .with_context(|| format!("open {}", part_path.display()))?;
        if file.metadata()?.len() != e.size {
            file.set_len(e.size)?;
        }
        let todo: Vec<usize> = (0..state.done.len()).filter(|&i| !state.done[i]).collect();
        already += state.done.iter().enumerate().filter(|(_, d)| **d).map(|(i, _)| chunk_len(e.size, state.chunk_size, i)).sum::<u64>();

        let job = Arc::new(FileJob {
            repo_path: e.path.clone(),
            size: e.size,
            sha256: sha,
            final_path,
            part_path,
            state_path,
            file,
            remaining: AtomicU64::new(todo.len() as u64),
            state: Mutex::new(state),
            last_save: Mutex::new(Instant::now()),
            url: tokio::sync::Mutex::new(None),
        });
        save_state(&job)?;
        for idx in todo {
            queue.push_back(Chunk { job: job.clone(), idx });
        }
        jobs.push(job);
    }

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner} [{elapsed_precise}] [{wide_bar}] {bytes}/{total_bytes} {binary_bytes_per_sec} ETA {eta}",
        )?
        .progress_chars("=> "),
    );
    pb.set_position(already);
    pb.enable_steady_tick(Duration::from_millis(200));
    if pb.is_hidden() {
        // Not a terminal (e.g. redirected to a log): print plain progress lines instead.
        let pb = pb.clone();
        tokio::spawn(async move {
            let (mut last, mut t) = (pb.position(), Instant::now());
            loop {
                tokio::time::sleep(Duration::from_secs(15)).await;
                let (pos, len) = (pb.position(), pb.length().unwrap_or(1));
                let speed = (pos.saturating_sub(last)) as f64 / t.elapsed().as_secs_f64();
                let eta = if speed > 0.0 { (len - pos) as f64 / speed } else { f64::NAN };
                println!(
                    "progress {:.2}/{:.2} GB ({:.1}%)  {:.1} MB/s  ETA {:.0} min",
                    pos as f64 / 1e9, len as f64 / 1e9, 100.0 * pos as f64 / len as f64, speed / 1e6, eta / 60.0
                );
                (last, t) = (pos, Instant::now());
            }
        });
    }

    // Files that were already fully downloaded in a previous run but never verified/renamed.
    let mut verifiers = Vec::new();
    for job in &jobs {
        if job.remaining.load(Ordering::SeqCst) == 0 {
            verifiers.push(spawn_finish(job.clone(), !args.no_verify, pb.clone()));
        }
    }

    // ---- workers ------------------------------------------------------------------------
    let queue = Arc::new(Mutex::new(queue));
    let verifiers = Arc::new(Mutex::new(verifiers));
    let mut workers = Vec::new();
    for _ in 0..args.connections.max(1) {
        let queue = queue.clone();
        let cdn = cdn.clone();
        let pb = pb.clone();
        let token = token.clone();
        let repo = args.repo.clone();
        let rev = args.revision.clone();
        let verifiers = verifiers.clone();
        let verify = !args.no_verify;
        workers.push(tokio::spawn(async move {
            loop {
                let Some(c) = queue.lock().unwrap().pop_front() else { break };
                let mut attempt = 0u32;
                loop {
                    match fetch_chunk(&cdn, &repo, &rev, token.as_deref(), &c, &pb).await {
                        Ok(()) => break,
                        Err(e) => {
                            attempt += 1;
                            if attempt >= 40 {
                                return Err(e.context(format!("giving up on {} chunk {}", c.job.repo_path, c.idx)));
                            }
                            if attempt % 5 == 1 {
                                say(&pb, format!("  retry {attempt} {} #{}: {e:#}", c.job.repo_path, c.idx));
                            }
                            let backoff = Duration::from_millis((500u64 << attempt.min(6)).min(30_000));
                            tokio::time::sleep(backoff).await;
                        }
                    }
                }
                {
                    let mut st = c.job.state.lock().unwrap();
                    st.done[c.idx] = true;
                }
                let left = c.job.remaining.fetch_sub(1, Ordering::SeqCst) - 1;
                let due = { c.job.last_save.lock().unwrap().elapsed() > Duration::from_secs(3) };
                if due || left == 0 {
                    if let Err(e) = save_state(&c.job) {
                        say(&pb, format!("  warning: saving resume state for {}: {e:#}", c.job.repo_path));
                    }
                }
                if left == 0 {
                    verifiers.lock().unwrap().push(spawn_finish(c.job.clone(), verify, pb.clone()));
                }
            }
            anyhow::Ok(())
        }));
    }
    let mut first_err = None;
    for w in workers {
        if let Err(e) = w.await? {
            first_err.get_or_insert(e);
        }
    }
    for job in &jobs {
        let _ = save_state(job);
    }
    if let Some(e) = first_err {
        pb.abandon();
        return Err(e.context("download incomplete - rerun the same command to resume"));
    }
    say(&pb, "all chunks downloaded, waiting for verification...");
    let handles: Vec<_> = std::mem::take(&mut *verifiers.lock().unwrap());
    let mut failed = false;
    for h in handles {
        if let Err(e) = h.await? {
            say(&pb, format!("  ERROR: {e:#}"));
            failed = true;
        }
    }
    pb.finish();
    if failed {
        bail!("some files failed verification (they were renamed to *.corrupt); rerun to re-download them");
    }
    println!("done -> {}", out.display());
    if let (Some(_), false) = (first, official) {
        // glm-ssd -m takes the folder (any folder name inside models/ works on its own too)
        let p = std::fs::canonicalize(&out).unwrap_or_else(|_| out.clone());
        let p = p.to_string_lossy();
        let p = p.strip_prefix(r"\\?\").unwrap_or(&p);
        println!("run it with:  glm-ssd chat -m \"{p}\" --vram-budget 22G --ram-budget 12G");
    }
    Ok(())
}

/// `https://huggingface.co/owner/name[/tree|blob|resolve/REV[/path]]` -> (repo, rev, include glob).
/// A folder becomes `folder/*`; a shard `X-00002-of-00004.gguf` becomes all shards of X.
fn parse_url(u: &str) -> Result<(String, Option<String>, Option<String>)> {
    let s = u.trim();
    let s = s.split(['?', '#']).next().unwrap_or(s).trim_end_matches('/');
    let s = s.strip_prefix("https://").or_else(|| s.strip_prefix("http://")).unwrap_or(s);
    let s = s.strip_prefix("www.").unwrap_or(s);
    let s = s
        .strip_prefix("huggingface.co/")
        .or_else(|| s.strip_prefix("hf.co/"))
        .ok_or_else(|| anyhow!("not a Hugging Face URL: {u}"))?;
    let parts: Vec<&str> = s.split('/').filter(|p| !p.is_empty()).collect();
    if parts.len() < 2 || matches!(parts[0], "datasets" | "spaces") {
        bail!("expected a model repo URL like https://huggingface.co/owner/name (got {u})");
    }
    let repo = format!("{}/{}", parts[0], parts[1]);
    let (mut rev, mut path) = (None, None);
    if parts.len() >= 4 && matches!(parts[2], "tree" | "blob" | "resolve") {
        rev = Some(parts[3].to_string());
        if parts.len() > 4 {
            let p = parts[4..].join("/");
            path = Some(if parts[2] == "tree" { format!("{p}/*") } else { shard_glob(&p) });
        }
    }
    Ok((repo, rev, path))
}

/// `X-00002-of-00004.gguf` -> ("X", "00004")
fn shard_parts(p: &str) -> Option<(&str, &str)> {
    let stem = p.strip_suffix(".gguf")?;
    let (head, n) = stem.rsplit_once("-of-")?;
    let (base, i) = head.rsplit_once('-')?;
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    (digits(i) && digits(n)).then_some((base, n))
}

/// `dir/X-00002-of-00004.gguf` -> `dir/X-*-of-00004.gguf`; other paths unchanged.
fn shard_glob(p: &str) -> String {
    match shard_parts(p) {
        Some((base, n)) => format!("{base}-*-of-{n}.gguf"),
        None => p.to_string(),
    }
}

/// The file to pass to `-m`: shard 1 of a split model, or the single .gguf.
fn shard_one(paths: &[String]) -> Option<String> {
    let ggufs: Vec<&String> = paths.iter().filter(|p| p.ends_with(".gguf")).collect();
    let one = ggufs.iter().find(|p| shard_parts(p).is_some() && p.contains("-00001-of-"));
    one.or(ggufs.first()).map(|p| p.to_string())
}

/// No --include for a repo other than the official one: group its GGUF files into models (all
/// shards of a split file are one model; mmproj / imatrix files are skipped) and pick one.
fn pick_model(repo: &str, entries: &[TreeEntry]) -> Result<Vec<String>> {
    let mut groups: Vec<(String, u64, usize)> = Vec::new(); // (glob, bytes, files)
    for e in entries {
        let name = e.path.rsplit('/').next().unwrap_or(&e.path).to_ascii_lowercase();
        if !name.ends_with(".gguf") || name.contains("mmproj") || name.contains("imatrix") {
            continue;
        }
        let g = shard_glob(&e.path);
        match groups.iter_mut().find(|x| x.0 == g) {
            Some(x) => {
                x.1 += e.size;
                x.2 += 1;
            }
            None => groups.push((g, e.size, 1)),
        }
    }
    if groups.is_empty() {
        bail!("no .gguf files in {repo}: this engine needs a GGUF model (use --include to fetch other files)");
    }
    let iq2: Vec<&(String, u64, usize)> = groups.iter().filter(|g| g.0.to_ascii_uppercase().contains("IQ2_XXS")).collect();
    let pick = if groups.len() == 1 { Some(&groups[0]) } else if iq2.len() == 1 { Some(iq2[0]) } else { None };
    if let Some(g) = pick {
        println!("picked {} ({:.2} GB in {} file(s))", g.0, g.1 as f64 / 1e9, g.2);
        return Ok(vec![g.0.clone()]);
    }
    groups.sort_by(|a, b| a.0.cmp(&b.0));
    let mut msg = format!("{repo} has {} GGUF models; choose one with --include:\n", groups.len());
    for g in &groups {
        msg += &format!("  --include \"{}\"   ({:.2} GB, {} file(s))\n", g.0, g.1 as f64 / 1e9, g.2);
    }
    bail!("{}", msg.trim_end())
}

fn chunk_len(size: u64, cs: u64, idx: usize) -> u64 {
    let start = idx as u64 * cs;
    (size - start).min(cs)
}

fn with_suffix(p: &Path, s: &str) -> PathBuf {
    let mut o = p.as_os_str().to_owned();
    o.push(s);
    PathBuf::from(o)
}

fn save_state(job: &FileJob) -> Result<()> {
    // Hold the state lock for the whole write+rename: workers save concurrently and share one
    // tmp path, so an unserialized second rename would fail with "file not found".
    let st = job.state.lock().unwrap();
    if !job.part_path.exists() {
        // Already verified and renamed (or moved to .corrupt): no state to keep.
        let _ = std::fs::remove_file(&job.state_path);
        return Ok(());
    }
    let json = serde_json::to_vec(&*st)?;
    // Persist data before claiming chunks are done.
    job.file.sync_data().ok();
    let tmp = with_suffix(&job.state_path, ".tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &job.state_path)?;
    drop(st);
    *job.last_save.lock().unwrap() = Instant::now();
    Ok(())
}

/// `ProgressBar::println` prints nothing when the bar is hidden (output redirected to a log).
fn say(pb: &ProgressBar, msg: impl AsRef<str>) {
    if pb.is_hidden() {
        println!("{}", msg.as_ref());
    } else {
        pb.println(msg);
    }
}

async fn list_tree(api: &reqwest::Client, repo: &str, rev: &str, token: Option<&str>) -> Result<Vec<TreeEntry>> {
    let mut url = Some(format!("{HF}/api/models/{repo}/tree/{rev}?recursive=true&expand=false"));
    let mut out = Vec::new();
    while let Some(u) = url.take() {
        let mut rq = api.get(&u);
        if let Some(t) = token {
            rq = rq.bearer_auth(t);
        }
        let resp = rq.send().await?.error_for_status().context("listing repo tree")?;
        // pagination: Link: <...>; rel="next"
        url = resp
            .headers()
            .get("link")
            .and_then(|v| v.to_str().ok())
            .and_then(|l| l.split(',').find(|p| p.contains("rel=\"next\"")))
            .and_then(|p| p.split(['<', '>']).nth(1))
            .map(str::to_string);
        out.extend(resp.json::<Vec<TreeEntry>>().await?);
    }
    Ok(out)
}

/// Returns the (possibly signed CDN) URL for a file, resolving it if missing/stale.
/// `api` must NOT follow redirects: we read the Location header ourselves.
async fn file_url(api: &reqwest::Client, repo: &str, rev: &str, token: Option<&str>, job: &FileJob) -> Result<String> {
    let mut g = job.url.lock().await;
    if let Some((u, t)) = g.as_ref() {
        if t.elapsed() < URL_MAX_AGE {
            return Ok(u.clone());
        }
    }
    let resolve = format!("{HF}/{repo}/resolve/{rev}/{}", job.repo_path);
    let mut rq = api.get(&resolve).header("Range", "bytes=0-0");
    if let Some(t) = token {
        rq = rq.bearer_auth(t);
    }
    let resp = rq.send().await?;
    let st = resp.status();
    let u = if st.is_redirection() {
        let loc = resp.headers().get("location").and_then(|v| v.to_str().ok()).ok_or_else(|| anyhow!("redirect without location"))?;
        if loc.starts_with('/') { format!("{HF}{loc}") } else { loc.to_string() }
    } else if st.is_success() {
        resolve // served directly (small / non-LFS file)
    } else {
        bail!("resolve {} -> HTTP {st}", job.repo_path);
    };
    *g = Some((u.clone(), Instant::now()));
    Ok(u)
}

async fn fetch_chunk(
    cdn: &reqwest::Client,
    repo: &str,
    rev: &str,
    token: Option<&str>,
    c: &Chunk,
    pb: &ProgressBar,
) -> Result<()> {
    let job = &c.job;
    let cs = job.state.lock().unwrap().chunk_size;
    let start = c.idx as u64 * cs;
    let len = chunk_len(job.size, cs, c.idx);
    if len == 0 {
        return Ok(());
    }
    let end = start + len - 1;

    let url = file_url(cdn, repo, rev, token, job).await?;
    let mut rq = cdn.get(&url).header("Range", format!("bytes={start}-{end}"));
    if url.starts_with(HF) {
        if let Some(t) = token {
            rq = rq.bearer_auth(t);
        }
    }
    let resp = rq.send().await?;
    let st = resp.status();
    if matches!(st.as_u16(), 401 | 403 | 404 | 410) || st.is_redirection() {
        *job.url.lock().await = None; // signed URL expired -> re-resolve next try
        bail!("HTTP {st}");
    }
    if st.as_u16() != 206 && !(st.as_u16() == 200 && start == 0 && len == job.size) {
        bail!("unexpected HTTP {st} for range {start}-{end}");
    }

    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::with_capacity(FLUSH_BYTES);
    let mut off = start;
    let mut got = 0u64;
    let res: Result<()> = async {
        while let Some(piece) = stream.next().await {
            let piece = piece?;
            got += piece.len() as u64;
            if got > len {
                bail!("server sent more than requested");
            }
            buf.extend_from_slice(&piece);
            pb.inc(piece.len() as u64);
            if buf.len() >= FLUSH_BYTES {
                tokio::task::block_in_place(|| write_at(&job.file, &buf, off))?;
                off += buf.len() as u64;
                buf.clear();
            }
        }
        if !buf.is_empty() {
            tokio::task::block_in_place(|| write_at(&job.file, &buf, off))?;
        }
        if got != len {
            bail!("short read: {got}/{len} bytes");
        }
        Ok(())
    }
    .await;
    if res.is_err() {
        pb.dec(got); // will be re-downloaded
    }
    res
}

fn write_at(f: &File, mut buf: &[u8], mut off: u64) -> std::io::Result<()> {
    while !buf.is_empty() {
        #[cfg(windows)]
        let n = std::os::windows::fs::FileExt::seek_write(f, buf, off)?;
        #[cfg(unix)]
        let n = std::os::unix::fs::FileExt::write_at(f, buf, off)?;
        if n == 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::WriteZero, "write returned 0"));
        }
        buf = &buf[n..];
        off += n as u64;
    }
    Ok(())
}

/// Verify SHA-256 (in a blocking thread) and rename `.part` -> final name.
fn spawn_finish(job: Arc<FileJob>, verify: bool, pb: ProgressBar) -> tokio::task::JoinHandle<Result<()>> {
    tokio::task::spawn_blocking(move || {
        job.file.sync_all()?;
        if verify {
            if let Some(expected) = &job.sha256 {
                let t = Instant::now();
                let mut f = File::open(&job.part_path)?;
                let mut h = Sha256::new();
                let mut buf = vec![0u8; 16 << 20];
                loop {
                    let n = f.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    h.update(&buf[..n]);
                }
                let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
                if &got != expected {
                    drop(f);
                    let bad = with_suffix(&job.final_path, ".corrupt");
                    let _ = std::fs::rename(&job.part_path, &bad);
                    let _ = std::fs::remove_file(&job.state_path);
                    bail!("SHA-256 mismatch for {} (got {got}, want {expected})", job.repo_path);
                }
                say(&pb, format!("  verified {} ({:.0}s)", job.repo_path, t.elapsed().as_secs_f64()));
            }
        }
        rename_when_free(&job)?;
        say(&pb, format!("  done {}", job.repo_path));
        Ok(())
    })
}

fn rename_when_free(job: &Arc<FileJob>) -> Result<()> {
    // Rust's File on Windows is opened with FILE_SHARE_DELETE, so rename works even while
    // the handle is open. Retry a few times in case AV/indexer briefly holds the file.
    let mut last = None;
    for _ in 0..20 {
        match std::fs::rename(&job.part_path, &job.final_path) {
            Ok(()) => {
                let _ = std::fs::remove_file(&job.state_path);
                return Ok(());
            }
            Err(e) => {
                last = Some(e);
                std::thread::sleep(Duration::from_millis(500));
            }
        }
    }
    Err(anyhow!("rename {} failed: {:?}", job.part_path.display(), last))
}
