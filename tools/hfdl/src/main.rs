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
const URL_MAX_AGE: Duration = Duration::from_secs(20 * 60);
const FLUSH_BYTES: usize = 8 << 20;

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Repository id
    #[arg(long, default_value = "unsloth/GLM-5.3-Flash-GGUF")]
    repo: String,
    /// Branch / tag / commit
    #[arg(long, default_value = "main")]
    revision: String,
    /// Glob(s) matched against repo paths (repeatable)
    #[arg(long, default_values_t = vec!["UD-IQ2_XXS/*".to_string()])]
    include: Vec<String>,
    /// Output directory (repo folder structure is preserved inside it)
    #[arg(long, default_value = "models")]
    out: PathBuf,
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
    let mut gs = GlobSetBuilder::new();
    for g in &args.include {
        gs.add(Glob::new(g).with_context(|| format!("bad glob {g}"))?);
    }
    let gs = gs.build()?;
    let entries = list_tree(&api, &args.repo, &args.revision, token.as_deref()).await?;
    let mut files: Vec<TreeEntry> =
        entries.into_iter().filter(|e| e.kind == "file" && gs.is_match(&e.path)).collect();
    if files.is_empty() {
        bail!("no files in {} match {:?}", args.repo, args.include);
    }
    files.sort_by_key(|e| e.size); // small files first -> metadata available early

    let total: u64 = files.iter().map(|f| f.size).sum();
    println!("{} file(s), {:.2} GB total:", files.len(), total as f64 / 1e9);
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
        let final_path = args.out.join(&e.path);
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
                                pb.println(format!("  retry {attempt} {} #{}: {e:#}", c.job.repo_path, c.idx));
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
                    save_state(&c.job)?;
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
    pb.println("all chunks downloaded, waiting for verification...");
    let handles: Vec<_> = std::mem::take(&mut *verifiers.lock().unwrap());
    let mut failed = false;
    for h in handles {
        if let Err(e) = h.await? {
            pb.println(format!("  ERROR: {e:#}"));
            failed = true;
        }
    }
    pb.finish();
    if failed {
        bail!("some files failed verification (they were renamed to *.corrupt); rerun to re-download them");
    }
    println!("done -> {}", args.out.display());
    Ok(())
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
    let json = { serde_json::to_vec(&*job.state.lock().unwrap())? };
    // Persist data before claiming chunks are done.
    job.file.sync_data().ok();
    let tmp = with_suffix(&job.state_path, ".tmp");
    std::fs::write(&tmp, json)?;
    std::fs::rename(&tmp, &job.state_path)?;
    *job.last_save.lock().unwrap() = Instant::now();
    Ok(())
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
                pb.println(format!("  verified {} ({:.0}s)", job.repo_path, t.elapsed().as_secs_f64()));
            }
        }
        rename_when_free(&job)?;
        pb.println(format!("  done {}", job.repo_path));
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
