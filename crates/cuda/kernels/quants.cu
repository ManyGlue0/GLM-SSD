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
       const float *__restrict__ x, int64_t x_stride, float *__restrict__ y, int64_t y_stride, int accumulate) {
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
                        float *y, int64_t y_stride, int acc, cudaStream_t s) {
    if (g_sm_count == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&g_sm_count, cudaDevAttrMultiProcessorCount, dev);
    }
    // types without tables: one row per warp; with tables: cap at ~16 blocks per SM
    const int need = (rows + GEMV_WARPS - 1) / GEMV_WARPS;
    const int cap = Q::SMEM > 0 ? g_sm_count * 16 : need;
    const unsigned blocks = (unsigned) (need < cap ? need : cap);
    k_gemv<Q, B><<<blocks, GEMV_WARPS * 32, Q::SMEM, s>>>((const uint8_t *) W, row_stride, K, rows, x, x_stride, y, y_stride, acc);
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
