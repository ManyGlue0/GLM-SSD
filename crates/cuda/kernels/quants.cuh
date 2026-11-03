// Per-type decoders for GGML quantized blocks.
//
// Every type exposes:
//   QK  - elements per block,  BB - bytes per block
//   deq8(block, g, v) - decode the 8 consecutive elements [8g, 8g+8) of one block into v.
// Decoding a group on its own (instead of whole blocks) lets 32 lanes of a warp cooperate on
// one 256-element super-block with coalesced loads, and keeps register pressure low.
//
// Index math is derived from ggml's dequantize_row_* (ggml-quants.c, MIT); it is written
// independently from the CPU port in crates/quants so the two can cross-check each other.
#pragma once
#include "common.cuh"
#include "iq_tables.cuh"

// ggml type ids
enum GsType : int {
    GS_F32 = 0, GS_F16 = 1, GS_Q8_0 = 8, GS_Q2_K = 10, GS_Q3_K = 11, GS_Q4_K = 12, GS_Q5_K = 13,
    GS_Q6_K = 14, GS_IQ2_XXS = 16, GS_IQ3_XXS = 18, GS_IQ2_S = 22, GS_IQ4_XS = 23, GS_BF16 = 30,
};

static __device__ __forceinline__ float ldh(const uint8_t *p) { return __half2float(*(const __half *) p); }
// u32 from a 2-byte aligned address
static __device__ __forceinline__ uint32_t ld32(const uint8_t *p) {
    const uint16_t *q = (const uint16_t *) p;
    return (uint32_t) q[0] | ((uint32_t) q[1] << 16);
}

// Types with lookup tables copy them into shared memory (SMEM bytes) once per thread block;
// deq8 then receives that shared-memory pointer as `sm`.
// Exact small-integer -> float without I2F (which is slow and shares the L1TEX/MIO queue):
// 0x4B000000 is 2^23, so its mantissa holds the integer exactly.
static __device__ __forceinline__ float u8f(uint32_t x) { return __uint_as_float(0x4B000000u | x) - 8388608.0f; }
// signed byte -> float, same trick
static __device__ __forceinline__ float s8f(int8_t x) { return __uint_as_float(0x4B000000u | (uint32_t) (uint8_t) (x ^ 0x80)) - 8388736.0f; }

struct QBase {
    static constexpr int SMEM = 0;
    static __device__ __forceinline__ void init_smem(uint8_t *, int, int) {}
};

template <typename T, int N>
static __device__ __forceinline__ void copy_table(uint8_t *sm, const T (&tab)[N], int tid, int nt) {
    T *d = (T *) sm;
    for (int i = tid; i < N; i += nt) d[i] = tab[i];
}

// ksigns_iq2xs[i] == i | (parity(i) << 7)
static __device__ __forceinline__ uint8_t ksigns(uint32_t i) { return (uint8_t) (i | ((__popc(i) & 1) << 7)); }

struct QF32 : QBase {
    static constexpr int QK = 8, BB = 32;
    static __device__ __forceinline__ void deq8(const uint8_t *b, int, float *v, const uint8_t *) {
        const float4 *p = (const float4 *) b;
        float4 a = p[0], c = p[1];
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = c.x; v[5] = c.y; v[6] = c.z; v[7] = c.w;
    }
};

struct QF16 : QBase {
    static constexpr int QK = 8, BB = 16;
    static __device__ __forceinline__ void deq8(const uint8_t *b, int, float *v, const uint8_t *) {
        const __half *p = (const __half *) b;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = __half2float(p[i]);
    }
};

struct QBF16 : QBase {
    static constexpr int QK = 8, BB = 16;
    static __device__ __forceinline__ void deq8(const uint8_t *b, int, float *v, const uint8_t *) {
        const uint16_t *p = (const uint16_t *) b;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = __uint_as_float((uint32_t) p[i] << 16);
    }
};

// block_q8_0 { half d; int8 qs[32]; }
struct QQ8_0 : QBase {
    static constexpr int QK = 32, BB = 34;
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const float d = ldh(b);
        const int8_t *q = (const int8_t *) (b + 2 + 8 * g);
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = d * s8f(q[i]);
    }
};

// block_q2_K { u8 scales[16]; u8 qs[64]; half d; half dmin; }
struct QQ2_K : QBase {
    static constexpr int QK = 256, BB = 84;
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const int e0 = 8 * g, hh = e0 >> 7, r = e0 & 127, j = r >> 5, part = (r >> 4) & 1, l0 = r & 15;
        const uint8_t sc = b[8 * hh + 2 * j + part];
        const float dl = ldh(b + 80) * (sc & 15), ml = ldh(b + 82) * (sc >> 4);
        const uint8_t *q = b + 16 + 32 * hh + 16 * part + l0;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = dl * u8f((q[i] >> (2 * j)) & 3) - ml;
    }
};

// block_q3_K { u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d; }
struct QQ3_K : QBase {
    static constexpr int QK = 256, BB = 110;
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const int e0 = 8 * g, hh = e0 >> 7, r = e0 & 127, j = r >> 5, part = (r >> 4) & 1, l0 = r & 15;
        const int is = 8 * hh + 2 * j + part;
        const uint8_t *raw = b + 96;
        const int low = is < 8 ? (raw[is] & 15) : (raw[is - 8] >> 4);
        const int high = (raw[8 + (is & 3)] >> (2 * (is >> 2))) & 3;
        const float dl = ldh(b + 108) * ((low | (high << 4)) - 32);
        const uint8_t m = 1 << (4 * hh + j);
        const uint8_t *hm = b + 16 * part + l0;
        const uint8_t *q = b + 32 + 32 * hh + 16 * part + l0;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = dl * (u8f(((q[i] >> (2 * j)) & 3) + ((hm[i] & m) ? 4 : 0)) - 4.0f);
    }
};

static __device__ __forceinline__ void scale_min_k4(int j, const uint8_t *q, int &d, int &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// block_q4_K { half d; half dmin; u8 scales[12]; u8 qs[128]; }
struct QQ4_K : QBase {
    static constexpr int QK = 256, BB = 144;
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const int e0 = 8 * g, jb = e0 >> 6, hi = (e0 >> 5) & 1, l0 = e0 & 31;
        int sc, mn;
        scale_min_k4(2 * jb + hi, b + 4, sc, mn);
        const float d = ldh(b) * sc, m = ldh(b + 2) * mn;
        // blocks are 16-byte aligned (144 B) and l0 is a multiple of 8: one 8-byte load
        const uint2 qq = *(const uint2 *) (b + 16 + 32 * jb + l0);
        const int sh = hi ? 4 : 0;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = d * u8f((((i < 4 ? qq.x : qq.y) >> (8 * (i & 3) + sh)) & 15)) - m;
    }
};

// block_q5_K { half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; }
struct QQ5_K : QBase {
    static constexpr int QK = 256, BB = 176;
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const int e0 = 8 * g, jb = e0 >> 6, hi = (e0 >> 5) & 1, l0 = e0 & 31;
        int sc, mn;
        scale_min_k4(2 * jb + hi, b + 4, sc, mn);
        const float d = ldh(b) * sc, m = ldh(b + 2) * mn;
        // blocks are 16-byte aligned (176 B) and l0 is a multiple of 8: two 8-byte loads
        const uint2 hh = *(const uint2 *) (b + 16 + l0);
        const uint2 qq = *(const uint2 *) (b + 48 + 32 * jb + l0);
        const int sh = hi ? 4 : 0, hb = 2 * jb + hi;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int o = 8 * (i & 3);
            const uint32_t lo = ((i < 4 ? qq.x : qq.y) >> (o + sh)) & 15, h = ((i < 4 ? hh.x : hh.y) >> (o + hb)) & 1;
            v[i] = d * u8f(lo | (h << 4)) - m;
        }
    }
};

// block_q6_K { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d; }
struct QQ6_K : QBase {
    static constexpr int QK = 256, BB = 210;
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const int e0 = 8 * g, hh = e0 >> 7, r = e0 & 127, quarter = r >> 5, l0 = r & 31;
        const uint8_t *ql = b + 64 * hh + l0 + 32 * (quarter & 1);
        const uint8_t *qh = b + 128 + 32 * hh + l0;
        const float d = ldh(b + 208) * (float) (int8_t) b[192 + 8 * hh + (l0 >> 4) + 2 * quarter];
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int nib = quarter >= 2 ? (ql[i] >> 4) : (ql[i] & 15);
            const int hb = (qh[i] >> (2 * quarter)) & 3;
            v[i] = d * (u8f(nib | (hb << 4)) - 32.0f);
        }
    }
};


static __device__ __forceinline__ void grid8(uint64_t grid, uint8_t signs, float dl, float *v) {
    const uint32_t lo = (uint32_t) grid, hi = (uint32_t) (grid >> 32);
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float x = dl * u8f(((j < 4 ? lo : hi) >> (8 * (j & 3))) & 0xff);
        v[j] = (signs >> j) & 1 ? -x : x;
    }
}

// block_iq2_xxs { half d; u16 qs[32]; }
struct QIQ2_XXS : QBase {
    static constexpr int QK = 256, BB = 66, SMEM = sizeof(iq2xxs_grid);
    static __device__ __forceinline__ void init_smem(uint8_t *sm, int t, int n) { copy_table(sm, iq2xxs_grid, t, n); }
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const int ib32 = g >> 2, l = g & 3;
        const uint32_t a0 = ld32(b + 2 + 8 * ib32), a1 = ld32(b + 6 + 8 * ib32);
        const float db = ldh(b) * (0.5f + (float) (a1 >> 28)) * 0.25f;
        grid8(((const uint64_t *) sm)[(a0 >> (8 * l)) & 0xff], ksigns((a1 >> (7 * l)) & 127), db, v);
    }
};

// block_iq2_s { half d; u8 qs[32]; u8 signs[32]; u8 qh[8]; u8 scales[8]; }
struct QIQ2_S : QBase {
    static constexpr int QK = 256, BB = 82, SMEM = sizeof(iq2s_grid);
    static __device__ __forceinline__ void init_smem(uint8_t *sm, int t, int n) { copy_table(sm, iq2s_grid, t, n); }
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const int ib32 = g >> 2, l = g & 3;
        const float dl = ldh(b) * (0.5f + (float) ((b[74 + ib32] >> (4 * (l >> 1))) & 15)) * 0.25f;
        const int idx = b[2 + 4 * ib32 + l] | ((b[66 + ib32] << (8 - 2 * l)) & 0x300);
        grid8(((const uint64_t *) sm)[idx], b[34 + 4 * ib32 + l], dl, v);
    }
};

// block_iq3_xxs { half d; u8 qs[64]; u8 scales_and_signs[32]; }
struct QIQ3_XXS : QBase {
    static constexpr int QK = 256, BB = 98, SMEM = sizeof(iq3xxs_grid);
    static __device__ __forceinline__ void init_smem(uint8_t *sm, int t, int n) { copy_table(sm, iq3xxs_grid, t, n); }
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const int ib32 = g >> 2, l = g & 3;
        const uint32_t aux = ld32(b + 66 + 4 * ib32);
        const float db = ldh(b) * (0.5f + (float) (aux >> 28)) * 0.5f;
        const uint8_t signs = ksigns((aux >> (7 * l)) & 127);
        const uint32_t g1 = ((const uint32_t *) sm)[b[2 + 8 * ib32 + 2 * l]];
        const uint32_t g2 = ((const uint32_t *) sm)[b[2 + 8 * ib32 + 2 * l + 1]];
        grid8((uint64_t) g1 | ((uint64_t) g2 << 32), signs, db, v);
    }
};

// block_iq4_xs { half d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }
struct QIQ4_XS : QBase {
    static constexpr int QK = 256, BB = 136, SMEM = sizeof(kvalues_iq4nl);
    static __device__ __forceinline__ void init_smem(uint8_t *sm, int t, int n) { copy_table(sm, kvalues_iq4nl, t, n); }
    static __device__ __forceinline__ void deq8(const uint8_t *b, int g, float *v, const uint8_t *sm) {
        const int e0 = 8 * g, ib = e0 >> 5, hi = (e0 >> 4) & 1, j0 = e0 & 15;
        const uint16_t scales_h = *(const uint16_t *) (b + 2);
        const int ls = ((b[4 + (ib >> 1)] >> (4 * (ib & 1))) & 15) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = ldh(b) * (ls - 32);
        const uint8_t *q = b + 8 + 16 * ib + j0;
#pragma unroll
        for (int i = 0; i < 8; i++) v[i] = dl * s8f(((const int8_t *) sm)[hi ? (q[i] >> 4) : (q[i] & 15)]);
    }
};

// Calls F.template operator()<Q>() for the quant struct matching `type`; returns false if unknown.
template <class F> static bool gs_dispatch_type(int type, F &&f) {
    switch (type) {
        case GS_F32: f.template operator()<QF32>(); return true;
        case GS_F16: f.template operator()<QF16>(); return true;
        case GS_BF16: f.template operator()<QBF16>(); return true;
        case GS_Q8_0: f.template operator()<QQ8_0>(); return true;
        case GS_Q2_K: f.template operator()<QQ2_K>(); return true;
        case GS_Q3_K: f.template operator()<QQ3_K>(); return true;
        case GS_Q4_K: f.template operator()<QQ4_K>(); return true;
        case GS_Q5_K: f.template operator()<QQ5_K>(); return true;
        case GS_Q6_K: f.template operator()<QQ6_K>(); return true;
        case GS_IQ2_XXS: f.template operator()<QIQ2_XXS>(); return true;
        case GS_IQ2_S: f.template operator()<QIQ2_S>(); return true;
        case GS_IQ3_XXS: f.template operator()<QIQ3_XXS>(); return true;
        case GS_IQ4_XS: f.template operator()<QIQ4_XS>(); return true;
        default: return false;
    }
}
