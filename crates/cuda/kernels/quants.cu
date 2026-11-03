// Dequantization and quantized GEMV kernels.
#include "quants.cuh"

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
// One warp per output row; lanes stride over 8-element groups of the row so that a warp reads
// a contiguous stretch of each block. x is read through the read-only cache.
constexpr int GEMV_WARPS = 4;

template <class Q, int B>
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

    // persistent blocks: each warp walks many rows so the table setup above is amortized
    for (int row = blockIdx.x * GEMV_WARPS + warp; row < rows; row += gridDim.x * GEMV_WARPS) {
        const uint8_t *wr = W + (int64_t) row * row_stride;
        float acc[B];
#pragma unroll
        for (int b = 0; b < B; b++) acc[b] = 0.f;

        for (int g = lane; g < n_groups; g += 32) {
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
            if (lane == 0) {
                float *dst = y + b * y_stride + row;
                *dst = accumulate ? *dst + s : s;
            }
        }
    }
}

static int g_sm_count = 0;

template <class Q, int B>
static void launch_gemv(const void *W, int64_t row_stride, int K, int rows, const float *x, int64_t x_stride,
                        float *y, int64_t y_stride, int acc, cudaStream_t s,
                        int n_batch = 1, int64_t w_bs = 0, int64_t x_bs = 0, int64_t y_bs = 0) {
    if (g_sm_count == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev);
    }
    // types without tables: one row per warp; with tables: cap at ~16 blocks per SM
    const int need = (rows + GEMV_WARPS - 1) / GEMV_WARPS;
    const int cap = Q::SMEM > 0 ? g_sm_count * 16 : need;
    const unsigned blocks = (unsigned) (need < cap ? need : cap);
    dim3 grid(blocks, n_batch);
    k_gemv<Q, B><<<grid, GEMV_WARPS * 32, Q::SMEM, s>>>((const uint8_t *) W, row_stride, K, rows, x, x_stride, y, y_stride, acc,
                                                       w_bs, x_bs, y_bs);
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

// ---- fused routed-expert FFN for one token (decode) ---------------------------------------------
// A batch of up to MOE_MAX experts of one layer whose weights are on the GPU. Two launches per
// batch instead of five per expert:
//   gate/up: h[e][r] = swiglu(gate_e[r] . x, up_e[r] . x)      (blockIdx.y = expert)
//   down:    y[r] += sum_e w_e * (down_e[r] . h[e])             (one warp per output row)
constexpr int MOE_MAX = 16;
struct MoeBatch {
    const uint8_t *gate[MOE_MAX];
    const uint8_t *up[MOE_MAX];
    const uint8_t *down[MOE_MAX];
    float w[MOE_MAX];
    int n;
};

template <class Q>
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
        float ag = 0.f, au = 0.f;
        for (int g = lane; g < n_groups; g += 32) {
            const float4 *xp = (const float4 *) (x + 8 * g);
            const float4 p = __ldg(xp), q = __ldg(xp + 1);
            float v[8];
            Q::deq8(wg + (g / GPB) * Q::BB, g % GPB, v, sm);
            ag += v[0] * p.x + v[1] * p.y + v[2] * p.z + v[3] * p.w + v[4] * q.x + v[5] * q.y + v[6] * q.z + v[7] * q.w;
            Q::deq8(wu + (g / GPB) * Q::BB, g % GPB, v, sm);
            au += v[0] * p.x + v[1] * p.y + v[2] * p.z + v[3] * p.w + v[4] * q.x + v[5] * q.y + v[6] * q.z + v[7] * q.w;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            ag += __shfl_xor_sync(0xffffffff, ag, o);
            au += __shfl_xor_sync(0xffffffff, au, o);
        }
        if (lane == 0) {
            if (lim > 0.f) {
                ag = fminf(ag, lim);
                au = fminf(fmaxf(au, -lim), lim);
            }
            h[(int64_t) e * rows + row] = ag / (1.f + expf(-ag)) * au;
        }
    }
}

template <class Q>
__global__ void __launch_bounds__(GEMV_WARPS * 32)
k_moe_down(MoeBatch b, int64_t row_stride, int K, int rows, const float *__restrict__ h, float *__restrict__ y) {
    extern __shared__ __align__(16) uint8_t sm[];
    Q::init_smem(sm, threadIdx.x, blockDim.x);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    constexpr int GPB = Q::QK / 8;
    const int n_groups = K / 8;
    for (int row = blockIdx.x * GEMV_WARPS + warp; row < rows; row += gridDim.x * GEMV_WARPS) {
        float acc = 0.f;
        for (int e = 0; e < b.n; e++) {
            const uint8_t *wr = b.down[e] + (int64_t) row * row_stride;
            const float *he = h + (int64_t) e * K;
            float a = 0.f;
            for (int g = lane; g < n_groups; g += 32) {
                float v[8];
                Q::deq8(wr + (g / GPB) * Q::BB, g % GPB, v, sm);
                const float4 *xp = (const float4 *) (he + 8 * g);
                const float4 p = xp[0], q = xp[1];
                a += v[0] * p.x + v[1] * p.y + v[2] * p.z + v[3] * p.w + v[4] * q.x + v[5] * q.y + v[6] * q.z + v[7] * q.w;
            }
            acc += b.w[e] * a;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
        if (lane == 0) y[row] += acc;
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

// gate/up: K = n_embd (row length), rows = n_ff; h: [n][rows]. down: K = n_ff, rows = n_embd.
GS_EXPORT int gs_moe_gateup(int type, const MoeBatch *b, int64_t row_stride, int K, int rows, const float *x, float *h,
                            float lim, void *s) {
    if (b->n <= 0 || b->n > MOE_MAX) return (int) cudaErrorInvalidValue;
    bool ok = gs_dispatch_type(type, [&]<class Q>() {
        dim3 grid(moe_blocks(rows, Q::SMEM), b->n);
        k_moe_gateup<Q><<<grid, GEMV_WARPS * 32, Q::SMEM, (cudaStream_t) s>>>(*b, row_stride, K, rows, x, h, lim);
    });
    if (!ok) return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}
GS_EXPORT int gs_moe_down(int type, const MoeBatch *b, int64_t row_stride, int K, int rows, const float *h, float *y, void *s) {
    if (b->n <= 0 || b->n > MOE_MAX) return (int) cudaErrorInvalidValue;
    bool ok = gs_dispatch_type(type, [&]<class Q>() {
        k_moe_down<Q><<<moe_blocks(rows, Q::SMEM), GEMV_WARPS * 32, Q::SMEM, (cudaStream_t) s>>>(*b, row_stride, K, rows, h, y);
    });
    if (!ok) return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}
