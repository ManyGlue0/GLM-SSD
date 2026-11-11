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

fn park_path() -> PathBuf {
    std::env::temp_dir().join(format!("{PARK_PREFIX}{}.bin", std::process::id()))
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
                let p = park_path();
                let t0 = Instant::now();
                match s.save_to(&p) {
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
                        let _ = std::fs::remove_file(&p);
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
