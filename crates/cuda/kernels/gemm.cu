// Quantized GEMM for batches of tokens (prefill, MoE expert groups, MTP verification).
//
//   Y[n][m] (+)= sum_k W[m][k] * X[n][k]      m < rows (weight rows), n < N (tokens)
//
// Weight tiles are dequantized on the fly (the same per-8-element `deq8` decoders as the GEMV)
// into shared memory as f16, activations are converted to f16, and the product runs on tensor
// cores (WMMA 16x16x16, f32 accumulation). Tile: 64 weight rows x 64 tokens x 32 K.
// blockIdx.z selects one of several independent problems (per-head matrices).
#include "quants.cuh"
#include <mma.h>
using namespace nvcuda;

constexpr int GBM = 64, GBN = 64, GBK = 32, GLD = GBK + 8, GLDC = GBM + 4;

static __host__ __device__ constexpr int align16(int x) { return (x + 15) & ~15; }

template <class Q>
__global__ void __launch_bounds__(256)
k_gemm(const uint8_t *__restrict__ W, int64_t row_stride, int K, int rows, const float *__restrict__ X, int64_t x_stride,
       float *__restrict__ Y, int64_t y_stride, int N, int accumulate, int64_t w_bs, int64_t x_bs, int64_t y_bs) {
    W += blockIdx.z * w_bs;
    X += blockIdx.z * x_bs;
    Y += blockIdx.z * y_bs;
    extern __shared__ __align__(32) uint8_t smraw[];
    uint8_t *tab = smraw;
    half *As = (half *) (smraw + align16(Q::SMEM) + 16);  // +16 keeps 32-byte alignment for wmma
    As = (half *) (((uintptr_t) As + 31) & ~(uintptr_t) 31);
    half *Bs = As + GBM * GLD;
    float *Cs = (float *) (Bs + GBN * GLD);
    Q::init_smem(tab, threadIdx.x, blockDim.x);
    __syncthreads();

    const int m0 = blockIdx.x * GBM, n0 = blockIdx.y * GBN;
    const int warp = threadIdx.x >> 5;
    const int wm = (warp & 3) * 16, wn = (warp >> 2) * 32;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2];
    wmma::fill_fragment(acc[0], 0.f);
    wmma::fill_fragment(acc[1], 0.f);
    constexpr int GPB = Q::QK / 8;

    const int ar = threadIdx.x >> 2, ag = threadIdx.x & 3;  // A: row, 8-group within the K step
    const int bn = threadIdx.x >> 2, bk = (threadIdx.x & 3) * 8;
    const int arow = m0 + ar, btok = n0 + bn;
    const uint8_t *wrow = W + (int64_t) (arow < rows ? arow : 0) * row_stride;
    const float *xrow = X + (int64_t) (btok < N ? btok : 0) * x_stride;

    for (int k0 = 0; k0 < K; k0 += GBK) {
        {
            float v[8];
            if (arow < rows) {
                const int g = (k0 >> 3) + ag;
                Q::deq8(wrow + (g / GPB) * Q::BB, g % GPB, v, tab);
            } else {
#pragma unroll
                for (int i = 0; i < 8; i++) v[i] = 0.f;
            }
            half2 *d = (half2 *) (As + ar * GLD + ag * 8);
#pragma unroll
            for (int i = 0; i < 4; i++) d[i] = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
        }
        {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f), b = a;
            if (btok < N) {
                const float4 *p = (const float4 *) (xrow + k0 + bk);
                a = __ldg(p);
                b = __ldg(p + 1);
            }
            half2 *d = (half2 *) (Bs + bn * GLD + bk);
            d[0] = __floats2half2_rn(a.x, a.y);
            d[1] = __floats2half2_rn(a.z, a.w);
            d[2] = __floats2half2_rn(b.x, b.y);
            d[3] = __floats2half2_rn(b.z, b.w);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < GBK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
            wmma::load_matrix_sync(fa, As + wm * GLD + kk, GLD);
#pragma unroll
            for (int j = 0; j < 2; j++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;
                wmma::load_matrix_sync(fb, Bs + (wn + 16 * j) * GLD + kk, GLD);
                wmma::mma_sync(acc[j], fa, fb, acc[j]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 2; j++) wmma::store_matrix_sync(Cs + (wn + 16 * j) * GLDC + wm, acc[j], GLDC, wmma::mem_col_major);
    __syncthreads();
    for (int i = threadIdx.x; i < GBM * GBN; i += 256) {
        const int n = i / GBM, m = i % GBM;
        if (n0 + n < N && m0 + m < rows) {
            float *dst = Y + (int64_t) (n0 + n) * y_stride + m0 + m;
            const float v = Cs[n * GLDC + m];
            *dst = accumulate ? *dst + v : v;
        }
    }
}

template <class Q>
static int gemm_smem() {
    return align16(Q::SMEM) + 16 + 32 + (GBM + GBN) * GLD * 2 + GBN * GLDC * 4;
}

// K must be a multiple of 32 (and of the type's block size), x rows 16-byte aligned.
GS_EXPORT int gs_gemm(int type, const void *W, int64_t row_stride, int K, int rows, const float *X, int64_t x_stride,
                      float *Y, int64_t y_stride, int N, int accumulate, int n_batch, int64_t w_bs, int64_t x_bs,
                      int64_t y_bs, void *stream) {
    if (K % GBK != 0 || N < 1 || n_batch < 1 || x_stride % 4 != 0) return (int) cudaErrorInvalidValue;
    bool bad = false;
    bool ok = gs_dispatch_type(type, [&]<class Q>() {
        if (K % Q::QK != 0) { bad = true; return; }
        const int sm = gemm_smem<Q>();
        static bool attr_set = false;
        if (!attr_set && sm > 48 * 1024) {
            cudaFuncSetAttribute(k_gemm<Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
            attr_set = true;
        }
        dim3 grid((rows + GBM - 1) / GBM, (N + GBN - 1) / GBN, n_batch);
        k_gemm<Q><<<grid, 256, sm, (cudaStream_t) stream>>>((const uint8_t *) W, row_stride, K, rows, X, x_stride, Y,
                                                             y_stride, N, accumulate, w_bs, x_bs, y_bs);
    });
    if (!ok || bad) return (int) cudaErrorInvalidValue;
    return (int) cudaGetLastError();
}

// ---- row gather / scatter for MoE token groups ------------------------------------------------------
// y[i][:] = x[idx[i]][:]
__global__ void k_gather_rows(const float *x, const int *idx, float *y, int n) {
    const int i = blockIdx.y;
    const float *src = x + (int64_t) idx[i] * n;
    float *dst = y + (int64_t) i * n;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += gridDim.x * blockDim.x) dst[c] = src[c];
}
GS_EXPORT int gs_gather_rows(const float *x, const int *idx, float *y, int rows, int n, void *s) {
    if (rows <= 0) return 0;
    dim3 grid((n + 255) / 256, rows);
    k_gather_rows<<<grid, 256, 0, (cudaStream_t) s>>>(x, idx, y, n);
    return (int) cudaGetLastError();
}
// acc[idx[i]][:] += w[i] * y[i][:]   (idx unique within one call)
__global__ void k_scatter_add_rows(float *acc, const int *idx, const float *w, const float *y, int n) {
    const int i = blockIdx.y;
    float *dst = acc + (int64_t) idx[i] * n;
    const float *src = y + (int64_t) i * n;
    const float a = w[i];
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += gridDim.x * blockDim.x) dst[c] += a * src[c];
}
GS_EXPORT int gs_scatter_add_rows(float *acc, const int *idx, const float *w, const float *y, int rows, int n, void *s) {
    if (rows <= 0) return 0;
    dim3 grid((n + 255) / 256, rows);
    k_scatter_add_rows<<<grid, 256, 0, (cudaStream_t) s>>>(acc, idx, w, y, n);
    return (int) cudaGetLastError();
}

// f32 -> f16 conversion (latent cache writes)
__global__ void k_f32_to_f16(const float *x, half *y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}
GS_EXPORT int gs_f32_to_f16(const float *x, void *y, int64_t n, void *s) {
    if (n <= 0) return 0;
    k_f32_to_f16<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) s>>>(x, (half *) y, n);
    return (int) cudaGetLastError();
}
