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
// One decode step of the depthwise causal conv (kernel K=4) + SiLU.
// state: [n][K-1] previous inputs (oldest first), w: [n][K] (w[0] multiplies the oldest input)
__global__ void k_conv_step(const float *x, float *state, const float *w, float *y, int n, int K) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    float *st = state + (int64_t) c * (K - 1);
    const float *wc = w + (int64_t) c * K;
    const float xn = x[c];
    float acc = wc[K - 1] * xn;
    for (int j = 0; j < K - 1; j++) acc += wc[j] * st[j];
    for (int j = 0; j < K - 2; j++) st[j] = st[j + 1];
    st[K - 2] = xn;
    y[c] = acc / (1.f + expf(-acc));
}
GS_EXPORT int gs_conv_step(const float *x, float *state, const float *w, float *y, int n, int K, void *s) {
    k_conv_step<<<(n + 255) / 256, 256, 0, (cudaStream_t) s>>>(x, state, w, y, n, K);
    return (int) cudaGetLastError();
}

// Forget gate (Glm5NextTextForgetGate). ssm_a holds -exp(A_log) (llama.cpp convention).
// lower bound lb > -inf: g = lb * sigmoid(exp(A_log) * (f + dt_b));  else g = -exp(A_log) * softplus(f + dt_b)
__global__ void k_kda_gate(const float *f, const float *dt_b, const float *ssm_a, float *g, int n, int head_dim, float lb) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float a = ssm_a[i / head_dim];
    const float x = f[i] + dt_b[i];
    if (isinf(lb)) {
        const float sp = x > 20.f ? x : log1pf(expf(x));
        g[i] = a * sp;
    } else {
        g[i] = lb * sigmoidf(-a * x);
    }
}
GS_EXPORT int gs_kda_gate(const float *f, const float *dt_b, const float *ssm_a, float *g, int n, int head_dim, float lb, void *s) {
    k_kda_gate<<<(n + 255) / 256, 256, 0, (cudaStream_t) s>>>(f, dt_b, ssm_a, g, n, head_dim, lb);
    return (int) cudaGetLastError();
}

// One recurrent step (recurrent_kimi_delta_attention) for every head.
// S: [H][Dk][Dv] fp32 state, q (l2normed and pre-scaled by 1/sqrt(Dk)), k (l2normed), v: [H][D];
// g: [H][Dk] log-decay; beta_logit: [H] (sigmoid applied here). One block per head, thread = v column.
__global__ void k_kda_step(float *S, const float *q, const float *k, const float *v, const float *g,
                           const float *beta_logit, float *out, int D) {
    const int h = blockIdx.x, j = threadIdx.x;
    extern __shared__ float sm[];
    float *sq = sm, *sk = sm + D, *seg = sm + 2 * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sq[i] = q[h * D + i];
        sk[i] = k[h * D + i];
        seg[i] = expf(g[h * D + i]);
    }
    __syncthreads();
    if (j >= D) return;
    float *Sh = S + (int64_t) h * D * D;
    const float beta = sigmoidf(beta_logit[h]);
    float kv = 0.f;
    for (int i = 0; i < D; i++) {
        const float s = Sh[i * D + j] * seg[i];
        Sh[i * D + j] = s;
        kv += s * sk[i];
    }
    const float delta = (v[h * D + j] - kv) * beta;
    float o = 0.f;
    for (int i = 0; i < D; i++) {
        const float s = Sh[i * D + j] + sk[i] * delta;
        Sh[i * D + j] = s;
        o += s * sq[i];
    }
    out[h * D + j] = o;
}
GS_EXPORT int gs_kda_step(float *S, const float *q, const float *k, const float *v, const float *g, const float *beta_logit,
                          float *out, int H, int D, void *s) {
    k_kda_step<<<H, D, 3 * D * sizeof(float), (cudaStream_t) s>>>(S, q, k, v, g, beta_logit, out, D);
    return (int) cudaGetLastError();
}

// ---- MLA attention (decode, absorbed) ---------------------------------------------------------------
// For each head h: score_s = scale * q[h] . C[pos_s], softmax over s, o[h] = sum_s p_s * C[pos_s].
// C: latent cache [n_ctx][R]. idx: selected positions (n_sel) or null for 0..n_sel-1.
// scratch: [H][n_sel] floats. One block per head, a warp per key for the dot products.
__global__ void k_mla_attn(const float *q, const float *C, const int *idx, int n_sel, float scale, float *scratch,
                           float *o, int R) {
    const int h = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    extern __shared__ float sq[];
    for (int i = threadIdx.x; i < R; i += blockDim.x) sq[i] = q[(int64_t) h * R + i];
    __syncthreads();
    float *sc = scratch + (int64_t) h * n_sel;
    for (int s = warp; s < n_sel; s += nw) {
        const float *c = C + (int64_t) (idx ? idx[s] : s) * R;
        float d = 0.f;
        for (int i = lane; i < R; i += 32) d += sq[i] * c[i];
        d = warp_sum(d);
        if (lane == 0) sc[s] = d * scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int s = threadIdx.x; s < n_sel; s += blockDim.x) mx = fmaxf(mx, sc[s]);
    mx = block_max(mx);
    float sum = 0.f;
    for (int s = threadIdx.x; s < n_sel; s += blockDim.x) {
        const float e = expf(sc[s] - mx);
        sc[s] = e;
        sum += e;
    }
    sum = block_sum(sum);
    const float inv = 1.f / sum;
    __syncthreads();
    for (int i = threadIdx.x; i < R; i += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < n_sel; s++) acc += sc[s] * C[(int64_t) (idx ? idx[s] : s) * R + i];
        o[(int64_t) h * R + i] = acc * inv;
    }
}
GS_EXPORT int gs_mla_attn(const float *q, const float *C, const int *idx, int n_sel, float scale, float *scratch, float *o,
                          int H, int R, void *s) {
    k_mla_attn<<<H, 256, R * sizeof(float), (cudaStream_t) s>>>(q, C, idx, n_sel, scale, scratch, o, R);
    return (int) cudaGetLastError();
}

// ---- DSA k-pool indexer (Glm5NextTextIndexer) --------------------------------------------------------
// Pooled key of one complete pool: P[c] = sum_m softmax_m(gate[m][c] + ape[m][c]) * key[m][c]
// rows: kpool consecutive cache rows of [key(Dk) | gate(Dk)]
__global__ void k_idx_pool(const float *rows, const float *ape, float *P, int Dk, int kpool) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= Dk) return;
    float mx = -INFINITY;
    for (int m = 0; m < kpool; m++) mx = fmaxf(mx, rows[m * 2 * Dk + Dk + c] + ape[m * Dk + c]);
    float sum = 0.f, acc = 0.f;
    for (int m = 0; m < kpool; m++) {
        const float e = expf(rows[m * 2 * Dk + Dk + c] + ape[m * Dk + c] - mx);
        sum += e;
        acc += e * rows[m * 2 * Dk + c];
    }
    P[c] = acc / sum;
}
GS_EXPORT int gs_idx_pool(const float *rows, const float *ape, float *P, int Dk, int kpool, void *s) {
    k_idx_pool<<<(Dk + 127) / 128, 128, 0, (cudaStream_t) s>>>(rows, ape, P, Dk, kpool);
    return (int) cudaGetLastError();
}

// score[p] = sum_h w[h] * relu(scale * q[h] . P[p])   for p < n_pool   (warp per pool)
__global__ void k_idx_score(const float *q, const float *w, const float *P, int n_pool, float *score, int Hq, int Dk, float scale) {
    extern __shared__ float sq[];
    for (int i = threadIdx.x; i < Hq * Dk; i += blockDim.x) sq[i] = q[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int p = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (p >= n_pool) return;
    const float *pk = P + (int64_t) p * Dk;
    float kreg[4];
#pragma unroll
    for (int r = 0; r < 4; r++) kreg[r] = (lane + 32 * r) < Dk ? pk[lane + 32 * r] : 0.f;
    float total = 0.f;
    for (int h = 0; h < Hq; h++) {
        float d = 0.f;
#pragma unroll
        for (int r = 0; r < 4; r++) d += (lane + 32 * r) < Dk ? sq[h * Dk + lane + 32 * r] * kreg[r] : 0.f;
        d = warp_sum(d);
        total += w[h] * fmaxf(d * scale, 0.f);
    }
    if (lane == 0) score[p] = total;
}
GS_EXPORT int gs_idx_score(const float *q, const float *w, const float *P, int n_pool, float *score, int Hq, int Dk, float scale, void *s) {
    if (n_pool <= 0) return 0;
    if (Dk > 128) return (int) cudaErrorInvalidValue;
    const int warps = 8;
    k_idx_score<<<(n_pool + warps - 1) / warps, warps * 32, Hq * Dk * sizeof(float), (cudaStream_t) s>>>(q, w, P, n_pool, score, Hq, Dk, scale);
    return (int) cudaGetLastError();
}
