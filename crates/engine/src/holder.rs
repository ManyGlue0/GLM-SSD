//! Lazily loaded model with idle auto-unload: the engine (VRAM weights, expert caches, pinned
//! RAM, reader threads) is dropped after `idle` without use, together with the CUDA context, and
//! reloaded on the next request. The conversation state (prompt cache) is parked in a temp file
//! meanwhile, so the next message continues it instead of re-reading the whole conversation.

use crate::session::Session;
use anyhow::Result;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

pub struct State {
    pub session: Option<Session>,
    last: Instant,
    /// conversation state saved at the last unload
    saved: Option<PathBuf>,
}

impl State {
    /// The loaded session (call through `ModelHolder::lock`, which guarantees it is loaded).
    pub fn get(&mut self) -> &mut Session {
        self.session.as_mut().expect("model not loaded")
    }
    /// Mark activity now (call when a request finishes, so the idle clock starts from there).
    pub fn touch(&mut self) {
        self.last = Instant::now();
    }
}

pub struct ModelHolder {
    load: Box<dyn Fn() -> Result<Session> + Send + Sync>,
    state: Mutex<State>,
}

const PARK_PREFIX: &str = "glm-ssd-session-";

/// A 64-bit value from the OS random number generator.
pub(crate) fn os_random_u64() -> Result<u64> {
    let mut b = [0u8; 8];
    getrandom::getrandom(&mut b).map_err(|e| anyhow::anyhow!("OS random number generator: {e}"))?;
    Ok(u64::from_le_bytes(b))
}

/// Create a new park file in `dir` with an unpredictable name (pid + 64 random bits), refusing
/// to open anything that already exists (no following planted files or links, H1).
fn create_park_file(dir: &std::path::Path) -> Result<(PathBuf, std::fs::File)> {
    let mut last = None;
    for _ in 0..8 {
        let p = dir.join(format!("{PARK_PREFIX}{}-{:016x}.bin", std::process::id(), os_random_u64()?));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&p) {
            Ok(f) => return Ok((p, f)),
            Err(e) => last = Some(e),
        }
    }
    Err(anyhow::anyhow!("creating a park file in {}: {:?}", dir.display(), last))
}

/// Remove parked sessions of processes that ended without reloading (older than a day).
fn clean_stale() {
    let Ok(dir) = std::fs::read_dir(std::env::temp_dir()) else { return };
    for e in dir.flatten() {
        let name = e.file_name();
        let old = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|a| a > Duration::from_secs(86400));
        if name.to_string_lossy().starts_with(PARK_PREFIX) && old {
            let _ = std::fs::remove_file(e.path());
        }
    }
}

impl ModelHolder {
    /// `idle`: unload after this long without use (None = never).
    pub fn new(load: impl Fn() -> Result<Session> + Send + Sync + 'static, idle: Option<Duration>) -> Arc<Self> {
        clean_stale();
        let h = Arc::new(Self { load: Box::new(load), state: Mutex::new(State { session: None, last: Instant::now(), saved: None }) });
        if let Some(d) = idle {
            let w = Arc::downgrade(&h);
            std::thread::Builder::new()
                .name("idle-unload".into())
                .spawn(move || loop {
                    std::thread::sleep(Duration::from_secs(1).min(d));
                    let Some(h) = w.upgrade() else { return };
                    // try_lock: a request in progress holds the lock, so it is never unloaded mid-use
                    if let Ok(mut st) = h.state.try_lock() {
                        if st.session.is_some() && st.last.elapsed() >= d {
                            Self::unload_locked(&mut st, d);
                        }
                    };
                })
                .expect("spawning the idle-unload thread");
        }
        h
    }

    /// Lock the model, loading it first if it was unloaded.
    pub fn lock(&self) -> Result<MutexGuard<'_, State>> {
        let mut st = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if st.session.is_none() {
            let t0 = Instant::now();
            eprintln!("loading the model...");
            let mut sess = (self.load)()?;
            if let Some(p) = st.saved.take() {
                let t1 = Instant::now();
                match sess.load_from(&p) {
                    Ok(()) => eprintln!(
                        "(conversation cache restored: {} tokens in {:.1}s)",
                        sess.history().len(),
                        t1.elapsed().as_secs_f64()
                    ),
                    Err(e) => eprintln!("warning: restoring the conversation cache: {e:#}"),
                }
                let _ = std::fs::remove_file(&p);
            }
            st.session = Some(sess);
            eprintln!("model loaded in {:.1}s", t0.elapsed().as_secs_f64());
        }
        st.touch();
        Ok(st)
    }

    pub fn is_loaded(&self) -> bool {
        self.state.lock().map(|s| s.session.is_some()).unwrap_or(false)
    }

    /// Delete the parked conversation state, if any (call on exit).
    pub fn discard_parked(&self) {
        if let Ok(mut st) = self.state.lock() {
            if let Some(p) = st.saved.take() {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    /// Save the profile and the conversation state, drop the engine, release the CUDA context.
    fn unload_locked(st: &mut State, after: Duration) {
        if let Some(s) = st.session.as_mut() {
            if let Err(e) = s.eng.experts.save_profile() {
                eprintln!("warning: saving the hot-expert profile: {e:#}");
            }
            if !s.history().is_empty() {
                let t0 = Instant::now();
                let (p, res) = match create_park_file(&std::env::temp_dir()) {
                    Ok((p, f)) => {
                        let r = s.save_to_file(f);
                        (p, r)
                    }
                    Err(e) => (PathBuf::new(), Err(e)),
                };
                match res {
                    Ok(bytes) => {
                        eprintln!(
                            "(conversation cache saved: {} tokens, {:.0} MB in {:.1}s)",
                            s.history().len(),
                            bytes as f64 / 1e6,
                            t0.elapsed().as_secs_f64()
                        );
                        st.saved = Some(p);
                    }
                    Err(e) => {
                        eprintln!("warning: saving the conversation cache: {e:#}");
                        if !p.as_os_str().is_empty() {
                            let _ = std::fs::remove_file(&p);
                        }
                    }
                }
            }
        }
        st.session = None; // frees VRAM buffers, pinned RAM, reader threads
        if let Err(e) = cuda::device_reset() {
            eprintln!("warning: releasing the CUDA context: {e:#}");
        }
        eprintln!("\n(model unloaded after {}s idle: VRAM and RAM released; it reloads on the next request)", after.as_secs());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h1_park_file_names_are_unpredictable_and_created_exclusively() {
        let dir = std::env::temp_dir().join(format!("glmssd-h1-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (p1, f1) = create_park_file(&dir).unwrap();
        let (p2, f2) = create_park_file(&dir).unwrap();
        assert_ne!(p1, p2);
        let name = p1.file_name().unwrap().to_string_lossy().into_owned();
        let pid = format!("{PARK_PREFIX}{}-", std::process::id());
        assert!(name.starts_with(&pid) && name.ends_with(".bin"));
        let hex = &name[pid.len()..name.len() - 4];
        assert!(hex.len() == 16 && hex.bytes().all(|b| b.is_ascii_hexdigit()), "{name}");
        // the old predictable name is not used
        assert_ne!(name, format!("{PARK_PREFIX}{}.bin", std::process::id()));
        drop((f1, f2));
        // create_new: an existing file at the chosen path is never opened
        assert!(std::fs::OpenOptions::new().write(true).create_new(true).open(&p1).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}