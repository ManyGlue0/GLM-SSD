//! Thin, safe-ish wrappers over the CUDA runtime functions exported by `kernels/*.cu`.

use anyhow::{bail, Result};
use std::ffi::{c_char, c_void, CStr};
use std::marker::PhantomData;

pub mod ffi {
    use std::ffi::{c_char, c_void};

    #[repr(C)]
    pub struct GsDeviceInfo {
        pub name: [c_char; 256],
        pub cc_major: i32,
        pub cc_minor: i32,
        pub sm_count: i32,
        pub total_mem: u64,
        pub free_mem: u64,
        pub pci_bus: i32,
        pub pci_device: i32,
        pub async_engines: i32,
        pub can_map_host: i32,
    }

    extern "C" {
        pub fn gs_device_info(dev: i32, out: *mut GsDeviceInfo) -> i32;
        pub fn gs_error_string(e: i32) -> *const c_char;
        pub fn gs_malloc(p: *mut *mut c_void, n: usize) -> i32;
        pub fn gs_free(p: *mut c_void) -> i32;
        pub fn gs_host_alloc(p: *mut *mut c_void, n: usize) -> i32;
        pub fn gs_host_free(p: *mut c_void) -> i32;
        pub fn gs_host_register(p: *mut c_void, n: usize) -> i32;
        pub fn gs_host_unregister(p: *mut c_void) -> i32;
        pub fn gs_stream_create(s: *mut *mut c_void) -> i32;
        pub fn gs_stream_destroy(s: *mut c_void) -> i32;
        pub fn gs_stream_sync(s: *mut c_void) -> i32;
        pub fn gs_device_sync() -> i32;
        pub fn gs_memcpy_h2d(dst: *mut c_void, src: *const c_void, n: usize, s: *mut c_void) -> i32;
        pub fn gs_memcpy_d2h(dst: *mut c_void, src: *const c_void, n: usize, s: *mut c_void) -> i32;
        pub fn gs_memcpy_d2d(dst: *mut c_void, src: *const c_void, n: usize, s: *mut c_void) -> i32;
        pub fn gs_event_create(e: *mut *mut c_void) -> i32;
        pub fn gs_event_create_timed(e: *mut *mut c_void) -> i32;
        pub fn gs_event_destroy(e: *mut c_void) -> i32;
        pub fn gs_event_record(e: *mut c_void, s: *mut c_void) -> i32;
        pub fn gs_event_sync(e: *mut c_void) -> i32;
        pub fn gs_event_query(e: *mut c_void) -> i32;
        pub fn gs_stream_wait_event(s: *mut c_void, e: *mut c_void) -> i32;
        pub fn gs_event_elapsed_ms(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> i32;
        pub fn gs_memset(p: *mut c_void, v: i32, n: usize, s: *mut c_void) -> i32;
        pub fn gs_saxpy(n: i32, a: f32, x: *const f32, y: *mut f32, s: *mut c_void) -> i32;

        pub fn gs_dequant_f32(ty: i32, src: *const c_void, n: i64, dst: *mut f32, s: *mut c_void) -> i32;
        pub fn gs_dequant_f16(ty: i32, src: *const c_void, n: i64, dst: *mut c_void, s: *mut c_void) -> i32;
        pub fn gs_gemv(
            ty: i32, w: *const c_void, row_stride: i64, k: i32, rows: i32, x: *const f32, x_stride: i64,
            y: *mut f32, y_stride: i64, b: i32, accumulate: i32, s: *mut c_void,
        ) -> i32;
        pub fn gs_gemv_batched(
            ty: i32, w: *const c_void, row_stride: i64, k: i32, rows: i32, x: *const f32, y: *mut f32, n_batch: i32,
            w_bs: i64, x_bs: i64, y_bs: i64, s: *mut c_void,
        ) -> i32;

        // ops.cu
        pub fn gs_rmsnorm(x: *const f32, w: *const f32, y: *mut f32, rows: i32, n: i32, eps: f32, s: *mut c_void) -> i32;
        pub fn gs_layernorm(x: *const f32, w: *const f32, b: *const f32, y: *mut f32, rows: i32, n: i32, eps: f32, s: *mut c_void) -> i32;
        pub fn gs_l2norm(x: *const f32, y: *mut f32, rows: i32, n: i32, eps: f32, scale: f32, s: *mut c_void) -> i32;
        pub fn gs_rmsnorm_gated(x: *const f32, w: *const f32, gate: *const f32, y: *mut f32, rows: i32, n: i32, eps: f32, s: *mut c_void) -> i32;
        pub fn gs_swiglu(g: *const f32, u: *const f32, y: *mut f32, n: i64, lim: f32, s: *mut c_void) -> i32;
        pub fn gs_axpy(y: *mut f32, x: *const f32, a: f32, n: i64, s: *mut c_void) -> i32;
        pub fn gs_hc_pre(
            streams: *const f32, mixes: *const f32, scale: *const f32, base: *const f32, hcw: *mut f32, collapsed: *mut f32,
            t: i32, d: i32, eps: f32, iters: i32, s: *mut c_void,
        ) -> i32;
        pub fn gs_hc_post(streams: *mut f32, out: *const f32, hcw: *const f32, t: i32, d: i32, s: *mut c_void) -> i32;
        pub fn gs_hc_mean(streams: *const f32, y: *mut f32, t: i32, d: i32, s: *mut c_void) -> i32;
        pub fn gs_hc_expand(x: *const f32, streams: *mut f32, t: i32, d: i32, s: *mut c_void) -> i32;
        pub fn gs_conv_step(x: *const f32, state: *mut f32, w: *const f32, y: *mut f32, n: i32, k: i32, s: *mut c_void) -> i32;
        pub fn gs_kda_gate(f: *const f32, dt_b: *const f32, ssm_a: *const f32, g: *mut f32, n: i32, head_dim: i32, lb: f32, s: *mut c_void) -> i32;
        pub fn gs_kda_step(
            state: *mut f32, q: *const f32, k: *const f32, v: *const f32, g: *const f32, beta_logit: *const f32, out: *mut f32,
            h: i32, d: i32, s: *mut c_void,
        ) -> i32;
        pub fn gs_mla_attn(
            q: *const f32, c: *const f32, idx: *const i32, n_sel: i32, scale: f32, scratch: *mut f32, o: *mut f32, h: i32, r: i32,
            s: *mut c_void,
        ) -> i32;
        pub fn gs_idx_pool(rows: *const f32, ape: *const f32, p: *mut f32, dk: i32, kpool: i32, s: *mut c_void) -> i32;
        pub fn gs_idx_score(q: *const f32, w: *const f32, p: *const f32, n_pool: i32, score: *mut f32, hq: i32, dk: i32, scale: f32, s: *mut c_void) -> i32;
    }
}

/// Quantized ops. All pointers are device pointers; `ty` is the ggml type id.
pub mod ops {
    use super::*;

    /// Dequantize `n` elements of type `ty` at `src` into f32 at `dst`.
    pub fn dequant_f32(ty: u32, src: *const c_void, n: usize, dst: *mut f32, s: &Stream) -> Result<()> {
        check(unsafe { ffi::gs_dequant_f32(ty as i32, src, n as i64, dst, s.raw()) })
    }

    /// Dequantize into f16.
    pub fn dequant_f16(ty: u32, src: *const c_void, n: usize, dst: *mut c_void, s: &Stream) -> Result<()> {
        check(unsafe { ffi::gs_dequant_f16(ty as i32, src, n as i64, dst, s.raw()) })
    }

    /// `y[b][r] (+)= W[r] · x[b]` for `b < n_vec` (1..=8).
    /// W is `rows` rows of `k` elements, `row_stride` bytes apart.
    #[allow(clippy::too_many_arguments)]
    pub fn gemv(
        ty: u32, w: *const c_void, row_stride: usize, k: usize, rows: usize, x: *const f32, x_stride: usize,
        y: *mut f32, y_stride: usize, n_vec: usize, accumulate: bool, s: &Stream,
    ) -> Result<()> {
        check(unsafe {
            ffi::gs_gemv(
                ty as i32, w, row_stride as i64, k as i32, rows as i32, x, x_stride as i64, y, y_stride as i64,
                n_vec as i32, accumulate as i32, s.raw(),
            )
        })
    }
}

pub fn check(code: i32) -> Result<()> {
    if code == 0 {
        return Ok(());
    }
    let msg = unsafe { CStr::from_ptr(ffi::gs_error_string(code)) }.to_string_lossy();
    bail!("CUDA error {code}: {msg}")
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub sm_count: i32,
    pub total_mem: u64,
    pub free_mem: u64,
    pub async_engines: i32,
}

pub fn device_info(dev: i32) -> Result<DeviceInfo> {
    let mut i: ffi::GsDeviceInfo = unsafe { std::mem::zeroed() };
    check(unsafe { ffi::gs_device_info(dev, &mut i) })?;
    let name = unsafe { CStr::from_ptr(i.name.as_ptr() as *const c_char) }.to_string_lossy().into_owned();
    Ok(DeviceInfo {
        name,
        compute_capability: (i.cc_major, i.cc_minor),
        sm_count: i.sm_count,
        total_mem: i.total_mem,
        free_mem: i.free_mem,
        async_engines: i.async_engines,
    })
}

/// A CUDA stream (non-blocking w.r.t. the legacy default stream).
pub struct Stream(pub(crate) *mut c_void);
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}
impl Stream {
    pub fn new() -> Result<Self> {
        let mut s = std::ptr::null_mut();
        check(unsafe { ffi::gs_stream_create(&mut s) })?;
        Ok(Self(s))
    }
    pub fn sync(&self) -> Result<()> {
        check(unsafe { ffi::gs_stream_sync(self.0) })
    }
    pub fn raw(&self) -> *mut c_void {
        self.0
    }
}
impl Drop for Stream {
    fn drop(&mut self) {
        unsafe { ffi::gs_stream_destroy(self.0) };
    }
}

/// CUDA event with timing enabled.
pub struct Event(*mut c_void);
unsafe impl Send for Event {}
unsafe impl Sync for Event {}
impl Event {
    pub fn timed() -> Result<Self> {
        let mut e = std::ptr::null_mut();
        check(unsafe { ffi::gs_event_create_timed(&mut e) })?;
        Ok(Self(e))
    }
    pub fn record(&self, s: &Stream) -> Result<()> {
        check(unsafe { ffi::gs_event_record(self.0, s.0) })
    }
    pub fn sync(&self) -> Result<()> {
        check(unsafe { ffi::gs_event_sync(self.0) })
    }
    /// Milliseconds from `self` to `later` (both must be recorded and complete).
    pub fn elapsed_ms(&self, later: &Event) -> Result<f32> {
        let mut ms = 0f32;
        check(unsafe { ffi::gs_event_elapsed_ms(&mut ms, self.0, later.0) })?;
        Ok(ms)
    }
}
impl Drop for Event {
    fn drop(&mut self) {
        unsafe { ffi::gs_event_destroy(self.0) };
    }
}

/// Untyped device allocation.
pub struct DeviceBuffer {
    ptr: *mut c_void,
    len: usize,
}
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}
impl DeviceBuffer {
    pub fn new(len: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        check(unsafe { ffi::gs_malloc(&mut ptr, len.max(1)) })?;
        Ok(Self { ptr, len })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
    pub fn upload(&self, off: usize, src: &[u8], s: &Stream) -> Result<()> {
        assert!(off + src.len() <= self.len);
        check(unsafe { ffi::gs_memcpy_h2d((self.ptr as *mut u8).add(off) as _, src.as_ptr() as _, src.len(), s.0) })
    }
    pub fn zero(&self, s: &Stream) -> Result<()> {
        check(unsafe { ffi::gs_memset(self.ptr, 0, self.len, s.0) })
    }
    pub fn f32(&self) -> *mut f32 {
        self.ptr as *mut f32
    }
    pub fn download(&self, off: usize, dst: &mut [u8], s: &Stream) -> Result<()> {
        assert!(off + dst.len() <= self.len);
        check(unsafe { ffi::gs_memcpy_d2h(dst.as_mut_ptr() as _, (self.ptr as *mut u8).add(off) as _, dst.len(), s.0) })
    }
}
impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        unsafe { ffi::gs_free(self.ptr) };
    }
}

/// Page-locked host allocation (needed for truly async H2D copies and for direct disk reads).
pub struct PinnedBuffer {
    ptr: *mut u8,
    len: usize,
    _p: PhantomData<u8>,
}
unsafe impl Send for PinnedBuffer {}
unsafe impl Sync for PinnedBuffer {}
impl PinnedBuffer {
    pub fn new(len: usize) -> Result<Self> {
        let mut p = std::ptr::null_mut();
        check(unsafe { ffi::gs_host_alloc(&mut p, len.max(1)) })?;
        Ok(Self { ptr: p as *mut u8, len, _p: PhantomData })
    }
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
    pub fn ptr(&self) -> *mut u8 {
        self.ptr
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}
impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        unsafe { ffi::gs_host_free(self.ptr as _) };
    }
}

/// End-to-end toolchain check: H2D, kernel, D2H.
pub fn smoke_test() -> Result<()> {
    let n = 1 << 20;
    let s = Stream::new()?;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = vec![1.0; n];
    let dx = DeviceBuffer::new(n * 4)?;
    let dy = DeviceBuffer::new(n * 4)?;
    dx.upload(0, bytes_of(&x), &s)?;
    dy.upload(0, bytes_of(&y), &s)?;
    check(unsafe { ffi::gs_saxpy(n as i32, 2.0, dx.ptr() as _, dy.ptr() as _, s.raw()) })?;
    let mut out = vec![0f32; n];
    dy.download(0, bytes_of_mut(&mut out), &s)?;
    s.sync()?;
    for (i, v) in out.iter().enumerate() {
        if *v != 2.0 * i as f32 + 1.0 {
            bail!("saxpy mismatch at {i}: {v}");
        }
    }
    Ok(())
}

pub fn bytes_of<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
pub fn bytes_of_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}
