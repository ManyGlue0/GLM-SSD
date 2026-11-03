// Dequantization and quantized GEMV kernels.
#include "quants.cuh"
#include <cstdlib>

// ---- dequantize: one thread per 8-element group ------------------------------------------
template <class Q, typename T>
__global__ void k_dequant(const uint8_t *__restrict__ src, int64_t n_groups, T *__restrict__ dst) {
    extern __shared__ __align__(16) uint8_t sm[];
    Q::init_smem(sm, threadIdx.x, blockDim.x);
    __syncthreads();
    const int64_t g = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= n_groups) return;
    constexpr int GPB = Q::QK / 8;
    float v[8];
    Q::deq8(src + (g / GPB) * Q::BB, (int) (g % GPB), v, sm);
#pragma unroll
    for (int i = 0; i < 8; i++) dst[g * 8 + i] = (T) v[i];
}

// n = number of elements (multiple of the block size)
GS_EXPORT int gs_dequant_f32(int type, const void *src, int64_t n, float *dst, void *stream) {
    const int64_t ng = n / 8;
    const int threads = 256;
    const unsigned blocks = (unsigned) ((ng + threads - 1) / threads);
    bool ok = gs_dispatch_type(type, [&]<class Q>() {
        k_dequant<Q, float><<<blocks, threads, Q::SMEM, (cudaStream_t) stream>>>((const uint8_t *) src, ng, dst);
    });
    if (!ok) return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}

GS_EXPORT int gs_dequant_f16(int type, const void *src, int64_t n, __half *dst, void *stream) {
    const int64_t ng = n / 8;
    const int threads = 256;
    const unsigned blocks = (unsigned) ((ng + threads - 1) / threads);
    bool ok = gs_dispatch_type(type, [&]<class Q>() {
        k_dequant<Q, __half><<<blocks, threads, Q::SMEM, (cudaStream_t) stream>>>((const uint8_t *) src, ng, dst);
    });
    if (!ok) return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}

// ---- GEMV: y[b][r] (+)= sum_k W[r][k] * x[b][k]   for b < B (B = 1..8 vectors) ------------
// Each warp computes R rows; lanes stride over 8-element groups of a row so that a warp reads a
// contiguous stretch of each block, and each group of x (read through the read-only cache) is
// loaded once for the warp's R rows instead of once per row.
constexpr int GEMV_WARPS = 4;

template <class Q, int B, int R>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
k_gemv(const uint8_t *__restrict__ W, int64_t row_stride, int K, int rows,
       const float *__restrict__ x, int64_t x_stride, float *__restrict__ y, int64_t y_stride, int accumulate,
       int64_t w_bs, int64_t x_bs, int64_t y_bs) {
    // blockIdx.y selects one of several independent matrices (batched GEMV, e.g. per attention head)
    W += blockIdx.y * w_bs;
    x += blockIdx.y * x_bs;
    y += blockIdx.y * y_bs;
    extern __shared__ __align__(16) uint8_t sm[];
    Q::init_smem(sm, threadIdx.x, blockDim.x);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int GPB = Q::QK / 8;
    const int n_groups = K / 8;

    // persistent blocks: each warp walks many row groups so the table setup above is amortized
    for (int row0 = (blockIdx.x * GEMV_WARPS + warp) * R; row0 < rows; row0 += gridDim.x * GEMV_WARPS * R) {
        float acc[R][B];
#pragma unroll
        for (int r = 0; r < R; r++)
#pragma unroll
            for (int b = 0; b < B; b++) acc[r][b] = 0.f;
        const int nr = rows - row0 < R ? rows - row0 : R;

        for (int g = lane; g < n_groups; g += 32) {
            float4 xa[B], xc[B];
#pragma unroll
            for (int b = 0; b < B; b++) {
                const float4 *xp = (const float4 *) (x + b * x_stride + 8 * g);
                xa[b] = __ldg(xp);
                xc[b] = __ldg(xp + 1);
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (r < nr) {
                    float v[8];
                    Q::deq8(W + (int64_t) (row0 + r) * row_stride + (g / GPB) * Q::BB, g % GPB, v, sm);
#pragma unroll
                    for (int b = 0; b < B; b++) {
                        const float4 a = xa[b], c = xc[b];
                        acc[r][b] += v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * c.x + v[5] * c.y + v[6] * c.z + v[7] * c.w;
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
#pragma unroll
            for (int b = 0; b < B; b++) {
                float s = acc[r][b];
#pragma unroll
                for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
                if (lane == 0 && r < nr) {
                    float *dst = y + b * y_stride + row0 + r;
                    *dst = accumulate ? *dst + s : s;
                }
            }
        }
    }
}

// Few rows, long K (e.g. the 24-row hyper-connection projections): one block of GEMV_WIDE warps
// per row, the warps split the row and reduce through shared memory.
constexpr int GEMV_WIDE = 8;
template <class Q, int B>
__global__ void __launch_bounds__(GEMV_WIDE * 32)
k_gemv_wide(const uint8_t *__restrict__ W, int64_t row_stride, int K, int rows,
            const float *__restrict__ x, int64_t x_stride, float *__restrict__ y, int64_t y_stride, int accumulate,
            int64_t w_bs, int64_t x_bs, int64_t y_bs) {
    W += blockIdx.y * w_bs;
    x += blockIdx.y * x_bs;
    y += blockIdx.y * y_bs;
    extern __shared__ __align__(16) uint8_t sm[];
    Q::init_smem(sm, threadIdx.x, blockDim.x);
    __shared__ float red[GEMV_WIDE][B];
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int GPB = Q::QK / 8;
    const int n_groups = K / 8;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        const uint8_t *wr = W + (int64_t) row * row_stride;
        float acc[B];
#pragma unroll
        for (int b = 0; b < B; b++) acc[b] = 0.f;
        for (int g = threadIdx.x; g < n_groups; g += GEMV_WIDE * 32) {
            float v[8];
            Q::deq8(wr + (g / GPB) * Q::BB, g % GPB, v, sm);
#pragma unroll
            for (int b = 0; b < B; b++) {
                const float4 *xp = (const float4 *) (x + b * x_stride + 8 * g);
                const float4 a = __ldg(xp), c = __ldg(xp + 1);
                acc[b] += v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * c.x + v[5] * c.y + v[6] * c.z + v[7] * c.w;
            }
        }
#pragma unroll
        for (int b = 0; b < B; b++) {
            float s = acc[b];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
            if (lane == 0) red[warp][b] = s;
        }
        __syncthreads();
        if (threadIdx.x < B) {
            float s = 0.f;
#pragma unroll
            for (int w = 0; w < GEMV_WIDE; w++) s += red[w][threadIdx.x];
            float *dst = y + threadIdx.x * y_stride + row;
            *dst = accumulate ? *dst + s : s;
        }
        __syncthreads();
    }
}

static int g_sm_count = 0;
static int g_gemv_r = -1; // GLMSSD_GEMV_R overrides the rows-per-warp choice (tuning)


template <class Q, int B>
static void launch_gemv(const void *W, int64_t row_stride, int K, int rows, const float *x, int64_t x_stride,
                        float *y, int64_t y_stride, int acc, cudaStream_t s,
                        int n_batch = 1, int64_t w_bs = 0, int64_t x_bs = 0, int64_t y_bs = 0) {
    if (g_sm_count == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev);
    }
    if (g_gemv_r < 0) {
        const char *e = getenv("GLMSSD_GEMV_R");
        g_gemv_r = e ? atoi(e) : 0;
    }
    // few rows with long K: split each row over a block
    if (rows * n_batch < g_sm_count * 4 && K >= 2048) {
        dim3 grid((unsigned) rows, n_batch);
        k_gemv_wide<Q, B><<<grid, GEMV_WIDE * 32, Q::SMEM, s>>>((const uint8_t *) W, row_stride, K, rows, x, x_stride, y,
                                                                y_stride, acc, w_bs, x_bs, y_bs);
        return;
    }
    // rows per warp: more rows share each x load, but keep enough warps to fill the GPU
    int R = 1;
    if (B <= 2) {
        const int warps = rows * n_batch / GEMV_WARPS; // blocks at R = 1
        R = warps >= g_sm_count * 16 ? 4 : warps >= g_sm_count * 8 ? 2 : 1;
        if (Q::QK == 256) R = 1; // K-quants / IQ types: instruction-bound, extra rows only add registers
        if (g_gemv_r > 0) R = g_gemv_r;
    }
    // types without tables: one row group per warp; with tables: cap at ~16 blocks per SM
    const int need = (rows + GEMV_WARPS * R - 1) / (GEMV_WARPS * R);
    const int cap = Q::SMEM > 0 ? g_sm_count * 16 : need;
    const unsigned blocks = (unsigned) (need < cap ? need : cap);
    dim3 grid(blocks, n_batch);
#define GS_GEMV_R(RR)                                                                                                    \
    k_gemv<Q, B, RR><<<grid, GEMV_WARPS * 32, Q::SMEM, s>>>((const uint8_t *) W, row_stride, K, rows, x, x_stride, y,    \
                                                           y_stride, acc, w_bs, x_bs, y_bs)
    if constexpr (B <= 2) {
        switch (R) {
            case 4: GS_GEMV_R(4); break;
            case 2: GS_GEMV_R(2); break;
            default: GS_GEMV_R(1); break;
        }
    } else {
        GS_GEMV_R(1);
    }
#undef GS_GEMV_R
}

// W: rows x K quantized matrix (row-major, row_stride bytes per row). x: B vectors of K floats
// (16-byte aligned, x_stride elements apart). y: B vectors of `rows` floats.
GS_EXPORT int gs_gemv(int type, const void *W, int64_t row_stride, int K, int rows, const float *x, int64_t x_stride,
                      float *y, int64_t y_stride, int B, int accumulate, void *stream) {
    if (K % 8 != 0 || B < 1 || B > 8) return (int) cudaErrorInvalidValue;
    cudaStream_t s = (cudaStream_t) stream;
    bool bad_k = false;
    bool ok = gs_dispatch_type(type, [&]<class Q>() {
        if (K % Q::QK != 0) { bad_k = true; return; }
        switch (B) {
            case 1: launch_gemv<Q, 1>(W, row_stride, K, rows, x, x_stride, y, y_stride, accumulate, s); break;
            case 2: launch_gemv<Q, 2>(W, row_stride, K, rows, x, x_stride, y, y_stride, accumulate, s); break;
            case 3: launch_gemv<Q, 3>(W, row_stride, K, rows, x, x_stride, y, y_stride, accumulate, s); break;
            case 4: launch_gemv<Q, 4>(W, row_stride, K, rows, x, x_stride, y, y_stride, accumulate, s); break;
            case 5: launch_gemv<Q, 5>(W, row_stride, K, rows, x, x_stride, y, y_stride, accumulate, s); break;
            case 6: launch_gemv<Q, 6>(W, row_stride, K, rows, x, x_stride, y, y_stride, accumulate, s); break;
            case 7: launch_gemv<Q, 7>(W, row_stride, K, rows, x, x_stride, y, y_stride, accumulate, s); break;
            case 8: launch_gemv<Q, 8>(W, row_stride, K, rows, x, x_stride, y, y_stride, accumulate, s); break;
        }
    });
    if (!ok || bad_k) return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}

// n_batch independent single-vector GEMVs: matrix b at W + b*w_bs bytes, x at x + b*x_bs, y at y + b*y_bs (floats).
GS_EXPORT int gs_gemv_batched(int type, const void *W, int64_t row_stride, int K, int rows, const float *x, float *y,
                              int n_batch, int64_t w_bs, int64_t x_bs, int64_t y_bs, void *stream) {
    if (K % 8 != 0 || n_batch < 1) return (int) cudaErrorInvalidValue;
    bool bad_k = false;
    bool ok = gs_dispatch_type(type, [&]<class Q>() {
        if (K % Q::QK != 0) { bad_k = true; return; }
        launch_gemv<Q, 1>(W, row_stride, K, rows, x, 0, y, 0, 0, (cudaStream_t) stream, n_batch, w_bs, x_bs, y_bs);
    });
    if (!ok || bad_k) return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}

// ---- fused routed-expert FFN for decode (1 token, or 2 for an MTP verify step) -----------------
// A batch of up to MOE_MAX experts of one layer whose weights are on the GPU. Two launches per
// batch instead of five per expert and token group:
//   gate/up: h[e][t][r] = swiglu(gate_e[r] . x[t], up_e[r] . x[t])   (blockIdx.y = expert)
//   down:    y[t][r] += sum_e w[t][e] * (down_e[r] . h[e][t])        (one warp per output row)
// Every expert runs on all NT tokens (the weights are read once either way); w[t][e] = 0 marks
// a token not routed to e, whose contribution is skipped.
constexpr int MOE_MAX = 16;
constexpr int MOE_NT = 2;
struct MoeBatch {
    const uint8_t *gate[MOE_MAX];
    const uint8_t *up[MOE_MAX];
    const uint8_t *down[MOE_MAX];
    float w[MOE_NT][MOE_MAX];
    int n;
};

static __device__ __forceinline__ float dot8(const float *v, const float4 p, const float4 q) {
    return v[0] * p.x + v[1] * p.y + v[2] * p.z + v[3] * p.w + v[4] * q.x + v[5] * q.y + v[6] * q.z + v[7] * q.w;
}

template <class Q, int NT>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
k_moe_gateup(MoeBatch b, int64_t row_stride, int K, int rows, const float *__restrict__ x, float *__restrict__ h, float lim) {
    extern __shared__ __align__(16) uint8_t sm[];
    Q::init_smem(sm, threadIdx.x, blockDim.x);
    __syncthreads();
    const int e = blockIdx.y, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int GPB = Q::QK / 8;
    const int n_groups = K / 8;
    for (int row = blockIdx.x * GEMV_WARPS + warp; row < rows; row += gridDim.x * GEMV_WARPS) {
        const uint8_t *wg = b.gate[e] + (int64_t) row * row_stride, *wu = b.up[e] + (int64_t) row * row_stride;
        float ag[NT], au[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) ag[t] = au[t] = 0.f;
        for (int g = lane; g < n_groups; g += 32) {
            float vg[8], vu[8];
            Q::deq8(wg + (g / GPB) * Q::BB, g % GPB, vg, sm);
            Q::deq8(wu + (g / GPB) * Q::BB, g % GPB, vu, sm);
#pragma unroll
            for (int t = 0; t < NT; t++) {
                const float4 *xp = (const float4 *) (x + (int64_t) t * K + 8 * g);
                const float4 p = __ldg(xp), q = __ldg(xp + 1);
                ag[t] += dot8(vg, p, q);
                au[t] += dot8(vu, p, q);
            }
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            float a = ag[t], c = au[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                a += __shfl_xor_sync(0xffffffff, a, o);
                c += __shfl_xor_sync(0xffffffff, c, o);
            }
            if (lane == 0) {
                if (lim > 0.f) {
                    a = fminf(a, lim);
                    c = fminf(fmaxf(c, -lim), lim);
                }
                h[((int64_t) e * NT + t) * rows + row] = a / (1.f + expf(-a)) * c;
            }
        }
    }
}

template <class Q, int NT>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
k_moe_down(MoeBatch b, int64_t row_stride, int K, int rows, const float *__restrict__ h, float *__restrict__ y) {
    extern __shared__ __align__(16) uint8_t sm[];
    Q::init_smem(sm, threadIdx.x, blockDim.x);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int GPB = Q::QK / 8;
    const int n_groups = K / 8;
    for (int row = blockIdx.x * GEMV_WARPS + warp; row < rows; row += gridDim.x * GEMV_WARPS) {
        float acc[NT];
#pragma unroll
        for (int t = 0; t < NT; t++) acc[t] = 0.f;
        for (int e = 0; e < b.n; e++) {
            const uint8_t *wr = b.down[e] + (int64_t) row * row_stride;
            float a[NT];
#pragma unroll
            for (int t = 0; t < NT; t++) a[t] = 0.f;
            for (int g = lane; g < n_groups; g += 32) {
                float v[8];
                Q::deq8(wr + (g / GPB) * Q::BB, g % GPB, v, sm);
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    const float4 *xp = (const float4 *) (h + ((int64_t) e * NT + t) * K + 8 * g);
                    a[t] += dot8(v, xp[0], xp[1]);
                }
            }
#pragma unroll
            for (int t = 0; t < NT; t++)
                if (b.w[t][e] != 0.f) acc[t] += b.w[t][e] * a[t];
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            float s = acc[t];
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
            if (lane == 0) y[(int64_t) t * rows + row] += s;
        }
    }
}

static unsigned moe_blocks(int rows, int smem) {
    if (g_sm_count == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev);
    }
    const int need = (rows + GEMV_WARPS - 1) / GEMV_WARPS;
    const int cap = smem > 0 ? g_sm_count * 16 : need;
    return (unsigned) (need < cap ? need : cap);
}

// gate/up: K = n_embd (row length), rows = n_ff; x: [nt][K]; h: [n][nt][rows].
// down: K = n_ff, rows = n_embd; y: [nt][rows]. nt = 1 or 2 tokens.
GS_EXPORT int gs_moe_gateup(int type, const MoeBatch *b, int nt, int64_t row_stride, int K, int rows, const float *x,
                            float *h, float lim, void *s) {
    if (b->n <= 0 || b->n > MOE_MAX || nt < 1 || nt > MOE_NT) return (int) cudaErrorInvalidValue;
    bool ok = gs_dispatch_type(type, [&]<class Q>() {
        dim3 grid(moe_blocks(rows, Q::SMEM), b->n);
        if (nt == 1)
            k_moe_gateup<Q, 1><<<grid, GEMV_WARPS * 32, Q::SMEM, (cudaStream_t) s>>>(*b, row_stride, K, rows, x, h, lim);
        else
            k_moe_gateup<Q, 2><<<grid, GEMV_WARPS * 32, Q::SMEM, (cudaStream_t) s>>>(*b, row_stride, K, rows, x, h, lim);
    });
    if (!ok) return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}
GS_EXPORT int gs_moe_down(int type, const MoeBatch *b, int nt, int64_t row_stride, int K, int rows, const float *h, float *y,
                          void *s) {
    if (b->n <= 0 || b->n > MOE_MAX || nt < 1 || nt > MOE_NT) return (int) cudaErrorInvalidValue;
    bool ok = gs_dispatch_type(type, [&]<class Q>() {
        const unsigned blocks = moe_blocks(rows, Q::SMEM);
        if (nt == 1)
            k_moe_down<Q, 1><<<blocks, GEMV_WARPS * 32, Q::SMEM, (cudaStream_t) s>>>(*b, row_stride, K, rows, h, y);
        else
            k_moe_down<Q, 2><<<blocks, GEMV_WARPS * 32, Q::SMEM, (cudaStream_t) s>>>(*b, row_stride, K, rows, h, y);
    });
    if (!ok) return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}
