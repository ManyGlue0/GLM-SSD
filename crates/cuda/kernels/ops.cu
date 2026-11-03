// Element-wise / normalization / attention kernels for the GLM-5.3-Flash (glm5-next) forward pass.
// Semantics follow transformers' modeling_glm5_next.py; comments name the reference module.
// All activations are f32. Kernels take a token count T where batching is cheap; the decode
// path calls them with T = 1.
#include "common.cuh"
#include <math.h>

static __device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
static __device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}
// blockDim.x must be a multiple of 32 and <= 1024. Result is valid in all threads.
static __device__ float block_sum(float v) {
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = lane < nw ? red[lane] : 0.f;
    return warp_sum(r);
}
static __device__ float block_max(float v) {
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = lane < nw ? red[lane] : -INFINITY;
    return warp_max(r);
}
static __device__ __forceinline__ float sigmoidf(float x) { return 1.f / (1.f + expf(-x)); }

// ---- norms ---------------------------------------------------------------------------------
// RMSNorm (Glm5NextTextRMSNorm / UnweightedRMSNorm): y = x * rsqrt(mean(x^2) + eps) [* w]
__global__ void k_rmsnorm(const float *x, const float *w, float *y, int n, float eps) {
    const float *xr = x + (int64_t) blockIdx.x * n;
    float *yr = y + (int64_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    const float r = rsqrtf(block_sum(ss) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * (w ? w[i] : 1.f);
}
GS_EXPORT int gs_rmsnorm(const float *x, const float *w, float *y, int rows, int n, float eps, void *s) {
    k_rmsnorm<<<rows, n >= 1024 ? 512 : 128, 0, (cudaStream_t) s>>>(x, w, y, n, eps);
    return (int) cudaGetLastError();
}

// LayerNorm with affine (indexer k_norm): y = (x - mean) / sqrt(var + eps) * w + b
__global__ void k_layernorm(const float *x, const float *w, const float *b, float *y, int n, float eps) {
    const float *xr = x + (int64_t) blockIdx.x * n;
    float *yr = y + (int64_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s) / n;
    float v = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += (xr[i] - mean) * (xr[i] - mean);
    const float r = rsqrtf(block_sum(v) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mean) * r * w[i] + b[i];
}
GS_EXPORT int gs_layernorm(const float *x, const float *w, const float *b, float *y, int rows, int n, float eps, void *s) {
    k_layernorm<<<rows, 128, 0, (cudaStream_t) s>>>(x, w, b, y, n, eps);
    return (int) cudaGetLastError();
}

// FLA-style l2norm (x / sqrt(sum x^2 + eps)) followed by a scale, per row.
__global__ void k_l2norm(const float *x, float *y, int n, float eps, float scale) {
    const float *xr = x + (int64_t) blockIdx.x * n;
    float *yr = y + (int64_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    const float r = scale / sqrtf(block_sum(ss) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r;
}
GS_EXPORT int gs_l2norm(const float *x, float *y, int rows, int n, float eps, float scale, void *s) {
    k_l2norm<<<rows, 128, 0, (cudaStream_t) s>>>(x, y, n, eps, scale);
    return (int) cudaGetLastError();
}

// Gated RMSNorm (Glm5NextTextRMSNormGated): y = rmsnorm(x) * w * sigmoid(gate), per row of n
__global__ void k_rmsnorm_gated(const float *x, const float *w, const float *gate, float *y, int n, float eps) {
    const int64_t o = (int64_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[o + i] * x[o + i];
    const float r = rsqrtf(block_sum(ss) / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[o + i] = x[o + i] * r * w[i] * sigmoidf(gate[o + i]);
}
GS_EXPORT int gs_rmsnorm_gated(const float *x, const float *w, const float *gate, float *y, int rows, int n, float eps, void *s) {
    k_rmsnorm_gated<<<rows, 128, 0, (cudaStream_t) s>>>(x, w, gate, y, n, eps);
    return (int) cudaGetLastError();
}

// ---- element-wise ---------------------------------------------------------------------------
// SwiGLU with clamp (ggml_swiglu_clamp): silu(min(g, L)) * clamp(u, -L, L); L <= 0 -> no clamp
__global__ void k_swiglu(const float *g, const float *u, float *y, int64_t n, float lim) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = g[i], b = u[i];
    if (lim > 0.f) {
        a = fminf(a, lim);
        b = fminf(fmaxf(b, -lim), lim);
    }
    y[i] = a / (1.f + expf(-a)) * b;
}
GS_EXPORT int gs_swiglu(const float *g, const float *u, float *y, int64_t n, float lim, void *s) {
    k_swiglu<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) s>>>(g, u, y, n, lim);
    return (int) cudaGetLastError();
}

// y += a * x
__global__ void k_axpy(float *y, const float *x, float a, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}
GS_EXPORT int gs_axpy(float *y, const float *x, float a, int64_t n, void *s) {
    k_axpy<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) s>>>(y, x, a, n);
    return (int) cudaGetLastError();
}

// ---- mHC (Glm5NextTextHyperConnection) ----------------------------------------------------------
// streams: [T][HC][D]. mixes: [T][24] = fn @ rmsnorm(flatten(streams)) (computed by gemv).
// Writes hcw[T][24] = pre[4] | post[4] | comb[4][4] (comb[src][dst]) and collapsed[T][D] = sum_i pre_i * stream_i.
constexpr int HC = 4;
__global__ void k_hc_pre(const float *streams, const float *mixes, const float *scale, const float *base,
                         float *hcw, float *collapsed, int D, float eps, int iters) {
    const int t = blockIdx.x;
    __shared__ float w[24];
    if (threadIdx.x == 0) {
        const float *m = mixes + t * 24;
        for (int i = 0; i < HC; i++) {
            w[i] = sigmoidf(m[i] * scale[0] + base[i]) + eps;
            w[HC + i] = 2.f * sigmoidf(m[HC + i] * scale[1] + base[HC + i]);
        }
        float c[HC][HC];
        for (int i = 0; i < HC; i++) {
            float mx = -INFINITY;
            for (int j = 0; j < HC; j++) {
                c[i][j] = m[2 * HC + HC * i + j] * scale[2] + base[2 * HC + HC * i + j];
                mx = fmaxf(mx, c[i][j]);
            }
            float sum = 0.f;
            for (int j = 0; j < HC; j++) {
                c[i][j] = expf(c[i][j] - mx);
                sum += c[i][j];
            }
            for (int j = 0; j < HC; j++) c[i][j] = c[i][j] / sum + eps;
        }
        // column normalize (sum over dim -2 == over i), then (iters-1) x (rows, cols)
        auto norm_cols = [&]() {
            for (int j = 0; j < HC; j++) {
                float s = 0.f;
                for (int i = 0; i < HC; i++) s += c[i][j];
                for (int i = 0; i < HC; i++) c[i][j] /= (s + eps);
            }
        };
        auto norm_rows = [&]() {
            for (int i = 0; i < HC; i++) {
                float s = 0.f;
                for (int j = 0; j < HC; j++) s += c[i][j];
                for (int j = 0; j < HC; j++) c[i][j] /= (s + eps);
            }
        };
        norm_cols();
        for (int it = 1; it < iters; it++) {
            norm_rows();
            norm_cols();
        }
        for (int i = 0; i < HC; i++)
            for (int j = 0; j < HC; j++) w[2 * HC + HC * i + j] = c[i][j];
        for (int i = 0; i < 24; i++) hcw[t * 24 + i] = w[i];
    }
    __syncthreads();
    const float *st = streams + (int64_t) t * HC * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < HC; i++) acc += w[i] * st[i * D + d];
        collapsed[(int64_t) t * D + d] = acc;
    }
}
GS_EXPORT int gs_hc_pre(const float *streams, const float *mixes, const float *scale, const float *base, float *hcw,
                        float *collapsed, int T, int D, float eps, int iters, void *s) {
    k_hc_pre<<<T, 256, 0, (cudaStream_t) s>>>(streams, mixes, scale, base, hcw, collapsed, D, eps, iters);
    return (int) cudaGetLastError();
}

// streams[t][d][:] = post[d] * out[t][:] + sum_s comb[s][d] * streams[t][s][:]   (in place)
__global__ void k_hc_post(float *streams, const float *out, const float *hcw, int D) {
    const int t = blockIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= D) return;
    const float *w = hcw + t * 24;
    float *st = streams + (int64_t) t * HC * D;
    float r[HC];
#pragma unroll
    for (int i = 0; i < HC; i++) r[i] = st[i * D + x];
    const float o = out[(int64_t) t * D + x];
#pragma unroll
    for (int d = 0; d < HC; d++) {
        float acc = w[HC + d] * o;
#pragma unroll
        for (int src = 0; src < HC; src++) acc += w[2 * HC + HC * src + d] * r[src];
        st[d * D + x] = acc;
    }
}
GS_EXPORT int gs_hc_post(float *streams, const float *out, const float *hcw, int T, int D, void *s) {
    dim3 grid((D + 255) / 256, T);
    k_hc_post<<<grid, 256, 0, (cudaStream_t) s>>>(streams, out, hcw, D);
    return (int) cudaGetLastError();
}

// y[t][:] = mean over the 4 streams (Glm5NextTextHyperHead)
__global__ void k_hc_mean(const float *streams, float *y, int D) {
    const int t = blockIdx.y;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= D) return;
    const float *st = streams + (int64_t) t * HC * D;
    y[(int64_t) t * D + x] = 0.25f * (st[x] + st[D + x] + st[2 * D + x] + st[3 * D + x]);
}
GS_EXPORT int gs_hc_mean(const float *streams, float *y, int T, int D, void *s) {
    dim3 grid((D + 255) / 256, T);
    k_hc_mean<<<grid, 256, 0, (cudaStream_t) s>>>(streams, y, D);
    return (int) cudaGetLastError();
}

// streams[t][i][:] = x[t][:] for all 4 streams (model input expansion)
__global__ void k_hc_expand(const float *x, float *streams, int D) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D) return;
    const float v = x[(int64_t) t * D + i];
#pragma unroll
    for (int s = 0; s < HC; s++) streams[((int64_t) t * HC + s) * D + i] = v;
}
GS_EXPORT int gs_hc_expand(const float *x, float *streams, int T, int D, void *s) {
    dim3 grid((D + 255) / 256, T);
    k_hc_expand<<<grid, 256, 0, (cudaStream_t) s>>>(x, streams, D);
    return (int) cudaGetLastError();
}

// ---- KDA (Glm5NextTextLinearAttention) ------------------------------------------------------------
// Depthwise causal conv (kernel K <= 8) + SiLU over T consecutive tokens.
// x, y: [T][n]; state: [n][K-1] previous inputs (oldest first); w: [n][K] (w[0] multiplies the oldest).
// If 0 <= save_at < T, the state after token `save_at` is also written to `save` (speculative rollback).
__global__ void k_conv_scan(const float *x, float *state, const float *w, float *y, int T, int n, int K, int save_at,
                            float *save) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float st[7], wc[8];
    for (int j = 0; j < K - 1; j++) st[j] = state[(int64_t) c * (K - 1) + j];
    for (int j = 0; j < K; j++) wc[j] = w[(int64_t) c * K + j];
    for (int t = 0; t < T; t++) {
        const float xn = x[(int64_t) t * n + c];
        float acc = wc[K - 1] * xn;
        for (int j = 0; j < K - 1; j++) acc += wc[j] * st[j];
        for (int j = 0; j < K - 2; j++) st[j] = st[j + 1];
        st[K - 2] = xn;
        y[(int64_t) t * n + c] = acc / (1.f + expf(-acc));
        if (t == save_at)
            for (int j = 0; j < K - 1; j++) save[(int64_t) c * (K - 1) + j] = st[j];
    }
    for (int j = 0; j < K - 1; j++) state[(int64_t) c * (K - 1) + j] = st[j];
}
GS_EXPORT int gs_conv_scan(const float *x, float *state, const float *w, float *y, int T, int n, int K, int save_at,
                           float *save, void *s) {
    if (K < 2 || K > 8) return (int) cudaErrorInvalidValue;
    k_conv_scan<<<(n + 255) / 256, 256, 0, (cudaStream_t) s>>>(x, state, w, y, T, n, K, save_at, save);
    return (int) cudaGetLastError();
}

// Forget gate (Glm5NextTextForgetGate). ssm_a holds -exp(A_log) (llama.cpp convention).
// f, g: [T][H][head_dim]. lower bound lb > -inf: g = lb * sigmoid(exp(A_log) * (f + dt_b));
// else g = -exp(A_log) * softplus(f + dt_b)
__global__ void k_kda_gate(const float *f, const float *dt_b, const float *ssm_a, float *g, int64_t n, int head_dim, int H,
                           float lb) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = (int) (i % ((int64_t) head_dim * H));
    const float a = ssm_a[c / head_dim];
    const float x = f[i] + dt_b[c];
    if (isinf(lb)) {
        const float sp = x > 20.f ? x : log1pf(expf(x));
        g[i] = a * sp;
    } else {
        g[i] = lb * sigmoidf(-a * x);
    }
}
GS_EXPORT int gs_kda_gate(const float *f, const float *dt_b, const float *ssm_a, float *g, int64_t n, int head_dim, int H,
                          float lb, void *s) {
    k_kda_gate<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) s>>>(f, dt_b, ssm_a, g, n, head_dim, H, lb);
    return (int) cudaGetLastError();
}

// Recurrent delta-rule scan (recurrent_kimi_delta_attention) over T tokens. One block per head,
// thread j owns column j of the head's [D][D] state in registers for the whole scan.
// S: [H][D][D]; q (l2normed, pre-scaled by 1/sqrt(D)), k (l2normed), v, g (log-decay): [T][H][D];
// beta_logit: [T][H]; out: [T][H][D]. save_at/S_save: state snapshot after token save_at.
template <int D>
__global__ void __launch_bounds__(D) k_kda_scan(float *S, const float *q, const float *k, const float *v, const float *g,
                                                const float *beta_logit, float *out, int T, int H, int save_at,
                                                float *S_save) {
    const int h = blockIdx.x, j = threadIdx.x;
    __shared__ float sq[D], sk[D], sg[D];
    float s[D];
    float *Sh = S + (int64_t) h * D * D;
#pragma unroll
    for (int i = 0; i < D; i++) s[i] = Sh[i * D + j];
    for (int t = 0; t < T; t++) {
        const int64_t base = ((int64_t) t * H + h) * D;
        __syncthreads();
        sq[j] = q[base + j];
        sk[j] = k[base + j];
        sg[j] = expf(g[base + j]);
        __syncthreads();
        const float beta = sigmoidf(beta_logit[(int64_t) t * H + h]);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < D; i++) {
            s[i] *= sg[i];
            kv += s[i] * sk[i];
        }
        const float delta = (v[base + j] - kv) * beta;
        float o = 0.f;
#pragma unroll
        for (int i = 0; i < D; i++) {
            s[i] += sk[i] * delta;
            o += s[i] * sq[i];
        }
        out[base + j] = o;
        if (t == save_at) {
            float *Ss = S_save + (int64_t) h * D * D;
#pragma unroll
            for (int i = 0; i < D; i++) Ss[i * D + j] = s[i];
        }
    }
#pragma unroll
    for (int i = 0; i < D; i++) Sh[i * D + j] = s[i];
}
// Same recurrence for a few tokens (decode / MTP verify): the columns of a head's state are
// independent, so a head is split over D/32 blocks of 32 columns, and each block's 8 warps own
// D/8 rows each (coalesced 128-byte row segments); the two dot products over rows (k.S and q.S)
// are reduced through shared memory. 256 blocks instead of 64, 16 state values per thread.
template <int D>
__global__ void __launch_bounds__(256) k_kda_step(float *S, const float *q, const float *k, const float *v, const float *g,
                                                 const float *beta_logit, float *out, int T, int H, int save_at,
                                                 float *S_save) {
    constexpr int RG = 8, RPT = D / RG;
    const int h = blockIdx.x, lane = threadIdx.x & 31, r = threadIdx.x >> 5;
    const int j = blockIdx.y * 32 + lane, i0 = r * RPT;
    __shared__ float sq[D], sk[D], sg[D], red[RG][32];
    float s[RPT];
    float *Sh = S + (int64_t) h * D * D;
#pragma unroll
    for (int ii = 0; ii < RPT; ii++) s[ii] = Sh[(i0 + ii) * D + j];
    for (int t = 0; t < T; t++) {
        const int64_t base = ((int64_t) t * H + h) * D;
        __syncthreads();
        if (threadIdx.x < D) {
            sq[threadIdx.x] = q[base + threadIdx.x];
            sk[threadIdx.x] = k[base + threadIdx.x];
            sg[threadIdx.x] = expf(g[base + threadIdx.x]);
        }
        const float vj = v[base + j]; // read before `out` (may alias v) is written below
        __syncthreads();
        float kv = 0.f;
#pragma unroll
        for (int ii = 0; ii < RPT; ii++) {
            s[ii] *= sg[i0 + ii];
            kv += s[ii] * sk[i0 + ii];
        }
        red[r][lane] = kv;
        __syncthreads();
        kv = 0.f;
#pragma unroll
        for (int x = 0; x < RG; x++) kv += red[x][lane];
        const float delta = (vj - kv) * sigmoidf(beta_logit[(int64_t) t * H + h]);
        float o = 0.f;
#pragma unroll
        for (int ii = 0; ii < RPT; ii++) {
            s[ii] += sk[i0 + ii] * delta;
            o += s[ii] * sq[i0 + ii];
        }
        __syncthreads(); // everyone has read red (kv) before it is reused
        red[r][lane] = o;
        __syncthreads();
        if (r == 0) {
            o = 0.f;
#pragma unroll
            for (int x = 0; x < RG; x++) o += red[x][lane];
            out[base + j] = o;
        }
        if (t == save_at) {
            float *Ss = S_save + (int64_t) h * D * D;
#pragma unroll
            for (int ii = 0; ii < RPT; ii++) Ss[(i0 + ii) * D + j] = s[ii];
        }
    }
#pragma unroll
    for (int ii = 0; ii < RPT; ii++) Sh[(i0 + ii) * D + j] = s[ii];
}

GS_EXPORT int gs_kda_scan(float *S, const float *q, const float *k, const float *v, const float *g, const float *beta_logit,
                          float *out, int T, int H, int D, int save_at, float *S_save, void *s) {
    if (D != 128) return (int) cudaErrorInvalidValue;
    if (T <= 4) {
        k_kda_step<128><<<dim3(H, 128 / 32), 256, 0, (cudaStream_t) s>>>(S, q, k, v, g, beta_logit, out, T, H, save_at, S_save);
    } else {
        k_kda_scan<128><<<H, 128, 0, (cudaStream_t) s>>>(S, q, k, v, g, beta_logit, out, T, H, save_at, S_save);
    }
    return (int) cudaGetLastError();
}

// ---- MLA attention (absorbed, flash-style, f16 latent cache) -----------------------------------------
// For query t and head h: score_s = scale * q[t][h] . C[sel[t][s]], softmax over s < cnt[t],
// o[t][h] = sum_s p_s * C[sel[t][s]]. All 16 heads of a block share each key tile, keys are streamed
// in tiles of 32 with an online softmax. With splits > 1 the key range is split across blocks and
// partial results go to `part` (combined by k_mla_combine), which keeps decode (T=1) parallel.
constexpr int MLA_HG = 16, MLA_KT = 32;

// Latent cache formats: f16 [ctx][R], or Q8 = int8 [ctx][R] + f16 scale per 32 values [ctx][R/32].
template <int R, bool Q8>
__global__ void __launch_bounds__(256) k_mla_flash(const float *q, const void *C, const half *Cs, const int *sel,
                                                   int sel_stride, const int *cnt, float scale, float *o, float *part,
                                                   int H, int splits) {
    constexpr int RC = R / 256;
    const int split = blockIdx.x, hg = blockIdx.y, t = blockIdx.z;
    const int h0 = hg * MLA_HG;
    extern __shared__ __align__(16) float smf[];
    float *sq = smf;                                // [HG][R]
    half *sc = (half *) (sq + MLA_HG * R);          // [KT][R]
    float *sp = (float *) (sc + MLA_KT * R);        // [HG][KT]
    float *sm_m = sp + MLA_HG * MLA_KT, *sm_l = sm_m + MLA_HG, *sm_f = sm_l + MLA_HG;
    int *sidx = (int *) (sm_f + MLA_HG);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < MLA_HG * R; i += 256) sq[i] = q[((int64_t) t * H + h0) * R + i];
    if (tid < MLA_HG) {
        sm_m[tid] = -INFINITY;
        sm_l[tid] = 0.f;
    }
    const int n = cnt[t];
    const int per = (n + splits - 1) / splits;
    const int s0 = split * per, s1 = min(n, s0 + per);
    float acc[MLA_HG][RC];
#pragma unroll
    for (int h = 0; h < MLA_HG; h++)
#pragma unroll
        for (int c = 0; c < RC; c++) acc[h][c] = 0.f;
    const int *srow = sel + (int64_t) t * sel_stride;
    __syncthreads();
    for (int kb = s0; kb < s1; kb += MLA_KT) {
        const int nk = min(MLA_KT, s1 - kb);
        if (tid < MLA_KT) sidx[tid] = tid < nk ? srow[kb + tid] : 0;
        __syncthreads();
        if constexpr (Q8) {
            // 16 int8 per thread-iteration -> 16 halves, scaled by the block's f16 scale
            for (int i = tid; i < MLA_KT * R / 16; i += 256) {
                const int kk = i / (R / 16), c = i % (R / 16);
                half2 *dst = (half2 *) (sc + kk * R + c * 16);
                if (kk < nk) {
                    const int64_t row = sidx[kk];
                    const uint4 v = ((const uint4 *) ((const int8_t *) C + row * R))[c];
                    const float d = __half2float(Cs[row * (R / 32) + c / 2]);
                    const int8_t *b = (const int8_t *) &v;
#pragma unroll
                    for (int j = 0; j < 8; j++) dst[j] = __floats2half2_rn(d * b[2 * j], d * b[2 * j + 1]);
                } else {
#pragma unroll
                    for (int j = 0; j < 8; j++) dst[j] = __floats2half2_rn(0.f, 0.f);
                }
            }
        } else {
            for (int i = tid; i < MLA_KT * R / 8; i += 256) {
                const int kk = i / (R / 8), c = i % (R / 8);
                uint4 val = make_uint4(0, 0, 0, 0);
                if (kk < nk) val = ((const uint4 *) ((const half *) C + (int64_t) sidx[kk] * R))[c];
                ((uint4 *) (sc + kk * R))[c] = val;
            }
        }
        __syncthreads();
        for (int kk = warp * 4; kk < warp * 4 + 4; kk++) {
            float d[MLA_HG];
#pragma unroll
            for (int h = 0; h < MLA_HG; h++) d[h] = 0.f;
            for (int r = lane; r < R; r += 32) {
                const float cv = __half2float(sc[kk * R + r]);
#pragma unroll
                for (int h = 0; h < MLA_HG; h++) d[h] += sq[h * R + r] * cv;
            }
#pragma unroll
            for (int h = 0; h < MLA_HG; h++) d[h] = warp_sum(d[h]);
            if (lane == 0)
#pragma unroll
                for (int h = 0; h < MLA_HG; h++) sp[h * MLA_KT + kk] = kk < nk ? d[h] * scale : -INFINITY;
        }
        __syncthreads();
        if (tid < MLA_HG) {
            const float m_old = sm_m[tid];
            float mx = m_old;
            for (int kk = 0; kk < nk; kk++) mx = fmaxf(mx, sp[tid * MLA_KT + kk]);
            const float f = m_old == -INFINITY ? 0.f : expf(m_old - mx);
            float l = sm_l[tid] * f;
            for (int kk = 0; kk < MLA_KT; kk++) {
                const float p = kk < nk ? expf(sp[tid * MLA_KT + kk] - mx) : 0.f;
                sp[tid * MLA_KT + kk] = p;
                l += p;
            }
            sm_m[tid] = mx;
            sm_l[tid] = l;
            sm_f[tid] = f;
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < MLA_HG; h++) {
            const float f = sm_f[h];
#pragma unroll
            for (int c = 0; c < RC; c++) {
                float a = acc[h][c] * f;
                for (int kk = 0; kk < nk; kk++) a += sp[h * MLA_KT + kk] * __half2float(sc[kk * R + tid + 256 * c]);
                acc[h][c] = a;
            }
        }
        __syncthreads();
    }
    if (splits == 1) {
#pragma unroll
        for (int h = 0; h < MLA_HG; h++) {
            const float l = sm_l[h];
            const float inv = l > 0.f ? 1.f / l : 0.f;
#pragma unroll
            for (int c = 0; c < RC; c++) o[((int64_t) t * H + h0 + h) * R + tid + 256 * c] = acc[h][c] * inv;
        }
    } else {
        float *pp = part + (((int64_t) t * (H / MLA_HG) + hg) * splits + split) * (MLA_HG * R + 2 * MLA_HG);
#pragma unroll
        for (int h = 0; h < MLA_HG; h++)
#pragma unroll
            for (int c = 0; c < RC; c++) pp[h * R + tid + 256 * c] = acc[h][c];
        if (tid < MLA_HG) {
            pp[MLA_HG * R + tid] = sm_m[tid];
            pp[MLA_HG * R + MLA_HG + tid] = sm_l[tid];
        }
    }
}

template <int R>
__global__ void k_mla_combine(const float *part, float *o, int H, int splits) {
    const int h = blockIdx.x, t = blockIdx.y;
    const int hg = h / MLA_HG, hh = h % MLA_HG;
    const int64_t stride = MLA_HG * R + 2 * MLA_HG;
    const float *base = part + ((int64_t) t * (H / MLA_HG) + hg) * splits * stride;
    float M = -INFINITY;
    for (int s = 0; s < splits; s++) M = fmaxf(M, base[s * stride + MLA_HG * R + hh]);
    float L = 0.f;
    for (int s = 0; s < splits; s++) {
        const float m = base[s * stride + MLA_HG * R + hh];
        if (m != -INFINITY) L += base[s * stride + MLA_HG * R + MLA_HG + hh] * expf(m - M);
    }
    const float inv = L > 0.f ? 1.f / L : 0.f;
    for (int r = threadIdx.x; r < R; r += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < splits; s++) {
            const float m = base[s * stride + MLA_HG * R + hh];
            if (m != -INFINITY) a += base[s * stride + hh * R + r] * expf(m - M);
        }
        o[((int64_t) t * H + h) * R + r] = a * inv;
    }
}

static size_t mla_smem(int R) { return (size_t) MLA_HG * R * 4 + MLA_KT * R * 2 + MLA_HG * MLA_KT * 4 + 3 * MLA_HG * 4 + MLA_KT * 4; }

// part must hold T * (H/16) * splits * (16*R + 32) floats when splits > 1.
GS_EXPORT int gs_mla_attn(const float *q, const void *C, const void *C_scales, const int *sel, int sel_stride,
                          const int *cnt, float scale, float *o, float *part, int T, int H, int R, int splits, void *s) {
    if (R != 512 || H % MLA_HG != 0 || splits < 1) return (int) cudaErrorInvalidValue;
    const size_t sm = mla_smem(R);
    // set on every call (cheap): a cached flag would go stale after gs_device_reset
    if (C_scales)
        cudaFuncSetAttribute(k_mla_flash<512, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    else
        cudaFuncSetAttribute(k_mla_flash<512, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) sm);
    dim3 grid(splits, H / MLA_HG, T);
    if (C_scales)
        k_mla_flash<512, true><<<grid, 256, sm, (cudaStream_t) s>>>(q, C, (const half *) C_scales, sel, sel_stride, cnt, scale,
                                                                    o, part, H, splits);
    else
        k_mla_flash<512, false><<<grid, 256, sm, (cudaStream_t) s>>>(q, C, nullptr, sel, sel_stride, cnt, scale, o, part, H,
                                                                     splits);
    if (splits > 1) k_mla_combine<512><<<dim3(H, T), 256, 0, (cudaStream_t) s>>>(part, o, H, splits);
    return (int) cudaGetLastError();
}

// Q8 latent rows: per block of 32 values, d = amax / 127, q = round(x / d)
__global__ void k_quant_q8_rows(const float *x, int8_t *q, half *d, int64_t n_blocks) {
    const int64_t b = (int64_t) blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (b >= n_blocks) return;
    const float v = x[b * 32 + lane];
    const float amax = warp_max(fabsf(v));
    const float dd = amax / 127.f;
    const float inv = dd > 0.f ? 1.f / dd : 0.f;
    q[b * 32 + lane] = (int8_t) __float2int_rn(v * inv);
    if (lane == 0) d[b] = __float2half(dd);
}
GS_EXPORT int gs_quant_q8_rows(const float *x, void *q, void *d, int64_t n, void *s) {
    if (n % 32 != 0) return (int) cudaErrorInvalidValue;
    const int64_t nb = n / 32;
    if (nb == 0) return 0;
    k_quant_q8_rows<<<(unsigned) ((nb + 7) / 8), 256, 0, (cudaStream_t) s>>>(x, (int8_t *) q, (half *) d, nb);
    return (int) cudaGetLastError();
}

// ---- DSA k-pool indexer (Glm5NextTextIndexer) --------------------------------------------------------
// Raw key/gate rows are only needed until their pool of `kpool` positions is complete, so they live
// in the current chunk ([T][Dk], positions p0..p0+T-1) or in a ring of kpool rows (older positions).
static __device__ __forceinline__ const float *idx_row(const float *chunk, const float *ring, int64_t pos, int p0, int Dk,
                                                       int kpool) {
    return pos >= p0 ? chunk + (pos - p0) * Dk : ring + (pos % kpool) * Dk;
}

// Pooled key of every pool j in [j0, j0+n_pools): P[j][c] = sum_m softmax_m(gate + ape)[m][c] * key[m][c]
__global__ void k_idx_pool(const float *kc, const float *gc, const float *rk, const float *rg, const float *ape, float *P,
                           int p0, int j0, int Dk, int kpool) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t j = j0 + blockIdx.y;
    if (c >= Dk) return;
    float mx = -INFINITY;
    for (int m = 0; m < kpool; m++) {
        const int64_t pos = j * kpool + m;
        mx = fmaxf(mx, idx_row(gc, rg, pos, p0, Dk, kpool)[c] + ape[m * Dk + c]);
    }
    float sum = 0.f, acc = 0.f;
    for (int m = 0; m < kpool; m++) {
        const int64_t pos = j * kpool + m;
        const float e = expf(idx_row(gc, rg, pos, p0, Dk, kpool)[c] + ape[m * Dk + c] - mx);
        sum += e;
        acc += e * idx_row(kc, rk, pos, p0, Dk, kpool)[c];
    }
    P[j * Dk + c] = acc / sum;
}
// After a chunk: copy rows of the trailing incomplete pool into the ring.
__global__ void k_idx_ring_store(const float *kc, const float *gc, float *rk, float *rg, int p0, int first, int last,
                                 int Dk, int kpool) {
    const int64_t pos = first + blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= last || c >= Dk) return;
    rk[(pos % kpool) * Dk + c] = kc[(pos - p0) * Dk + c];
    rg[(pos % kpool) * Dk + c] = gc[(pos - p0) * Dk + c];
}
GS_EXPORT int gs_idx_pool(const float *kc, const float *gc, float *rk, float *rg, const float *ape, float *P, int p0, int T,
                          int Dk, int kpool, void *s) {
    const int end = p0 + T;
    // pools completing inside [p0, end): jf (the pool holding p0) .. jl-1
    const int jf = p0 / kpool, jl = end / kpool;
    if (jl > jf) {
        dim3 grid((Dk + 127) / 128, jl - jf);
        k_idx_pool<<<grid, 128, 0, (cudaStream_t) s>>>(kc, gc, rk, rg, ape, P, p0, jf, Dk, kpool);
    }
    const int first = max(p0, jl * kpool);
    if (end > first) {
        dim3 grid((Dk + 127) / 128, end - first);
        k_idx_ring_store<<<grid, 128, 0, (cudaStream_t) s>>>(kc, gc, rk, rg, p0, first, end, Dk, kpool);
    }
    return (int) cudaGetLastError();
}

// score[t][p] = sum_h w[t][h] * relu(scale * q[t][h] . P[p])   for p < n_pool(t) = (pos0 + t + 1) / kpool
// One block per (group of 8 pools, query); a warp per pool.
__global__ void k_idx_score(const float *q, const float *w, const float *P, float *score, int64_t score_stride, int pos0,
                            int kpool, int Hq, int Dk, float scale) {
    const int t = blockIdx.y;
    const int n_pool = (pos0 + t + 1) / kpool;
    const int p = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if ((int) (blockIdx.x * (blockDim.x >> 5)) >= n_pool) return;
    extern __shared__ float sq[];
    const float *qt = q + (int64_t) t * Hq * Dk;
    for (int i = threadIdx.x; i < Hq * Dk; i += blockDim.x) sq[i] = qt[i];
    __syncthreads();
    if (p >= n_pool) return;
    const int lane = threadIdx.x & 31;
    const float *pk = P + (int64_t) p * Dk;
    float kreg[4];
#pragma unroll
    for (int r = 0; r < 4; r++) kreg[r] = (lane + 32 * r) < Dk ? pk[lane + 32 * r] : 0.f;
    const float *wt = w + (int64_t) t * Hq;
    float total = 0.f;
    for (int h = 0; h < Hq; h++) {
        float d = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) d += (lane + 32 * r) < Dk ? sq[h * Dk + lane + 32 * r] * kreg[r] : 0.f;
        d = warp_sum(d);
        total += wt[h] * fmaxf(d * scale, 0.f);
    }
    if (lane == 0) score[(int64_t) t * score_stride + p] = total;
}
GS_EXPORT int gs_idx_score(const float *q, const float *w, const float *P, float *score, int64_t score_stride, int pos0, int T,
                           int kpool, int Hq, int Dk, float scale, void *s) {
    const int max_pool = (pos0 + T) / kpool;
    if (max_pool <= 0 || T <= 0) return 0;
    if (Dk > 128) return (int) cudaErrorInvalidValue;
    const int warps = 8;
    dim3 grid((max_pool + warps - 1) / warps, T);
    k_idx_score<<<grid, warps * 32, Hq * Dk * sizeof(float), (cudaStream_t) s>>>(q, w, P, score, score_stride, pos0, kpool,
                                                                              Hq, Dk, scale);
    return (int) cudaGetLastError();
}

// Build the attended position list of each query t (position pos0 + t):
//   n_pool <= top_k: every position 0..pos (dense);
//   else: the 4 positions of each of the top_k highest-scoring pools + the incomplete tail.
// Top-k is an exact radix select on the order-preserving uint32 image of the scores.
static __device__ __forceinline__ uint32_t f2ord(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__global__ void k_idx_select(const float *score, int64_t score_stride, int pos0, int kpool, int top_k, int *sel,
                             int sel_stride, int *cnt) {
    const int t = blockIdx.x;
    const int pos = pos0 + t;
    const int n_pool = (pos + 1) / kpool;
    int *out = sel + (int64_t) t * sel_stride;
    if (n_pool <= top_k) {
        for (int i = threadIdx.x; i <= pos; i += blockDim.x) out[i] = i;
        if (threadIdx.x == 0) cnt[t] = pos + 1;
        return;
    }
    const float *sc = score + (int64_t) t * score_stride;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_mask, s_need, s_count;
    if (threadIdx.x == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_need = top_k;
    }
    __syncthreads();
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix, mask = s_mask;
        for (int p = threadIdx.x; p < n_pool; p += blockDim.x) {
            const uint32_t u = f2ord(sc[p]);
            if ((u & mask) == prefix) atomicAdd(&hist[(u >> shift) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t need = s_need, acc = 0;
            int d = 255;
            for (; d > 0; d--) {
                if (acc + hist[d] >= need) break;
                acc += hist[d];
            }
            s_need = need - acc;  // how many to take among those equal to the prefix so far
            s_prefix = prefix | ((uint32_t) d << shift);
            s_mask = mask | (255u << shift);
        }
        __syncthreads();
    }
    // threshold value U = s_prefix; take all > U, and s_need of those == U
    const uint32_t U = s_prefix;
    if (threadIdx.x == 0) s_count = 0;
    __shared__ uint32_t s_eq;
    if (threadIdx.x == 0) s_eq = 0;
    __syncthreads();
    const uint32_t need_eq = s_need;
    for (int p = threadIdx.x; p < n_pool; p += blockDim.x) {
        const uint32_t u = f2ord(sc[p]);
        bool take = u > U;
        if (u == U) take = atomicAdd(&s_eq, 1u) < need_eq;
        if (take) {
            const uint32_t slot = atomicAdd(&s_count, 1u);
            for (int m = 0; m < kpool; m++) out[slot * kpool + m] = p * kpool + m;
        }
    }
    __syncthreads();
    const int base = top_k * kpool;
    const int tail0 = n_pool * kpool;
    for (int i = threadIdx.x; tail0 + i <= pos; i += blockDim.x) out[base + i] = tail0 + i;
    if (threadIdx.x == 0) cnt[t] = base + (pos + 1 - tail0);
}
GS_EXPORT int gs_idx_select(const float *score, int64_t score_stride, int pos0, int T, int kpool, int top_k, int *sel,
                            int sel_stride, int *cnt, void *s) {
    if (T <= 0) return 0;
    k_idx_select<<<T, 256, 0, (cudaStream_t) s>>>(score, score_stride, pos0, kpool, top_k, sel, sel_stride, cnt);
    return (int) cudaGetLastError();
}
