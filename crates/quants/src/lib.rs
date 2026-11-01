//! CPU reference dequantization, ported 1:1 from ggml's `dequantize_row_*` (ggml-quants.c).
//!
//! This is the ground truth the CUDA kernels are tested against, so it favours being an
//! obviously-faithful transcription over speed.

mod tables;

use anyhow::{bail, Result};
use gguf::GgmlType;
pub use tables::*;

pub const QK_K: usize = 256;

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: renormalize
            let mut e = 113u32; // 127 - 15 + 1
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[inline]
pub fn f32_to_f16(f: f32) -> u16 {
    // round-to-nearest-even; adequate for tests and for writing f16 caches on the CPU side
    let x = f.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let man = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1 << shift) - 1);
        let mut r = m >> shift;
        if rem > half || (rem == half && r & 1 == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let rem = man & 0x1fff;
    let mut r = ((e as u32) << 10) | (man >> 13);
    if rem > 0x1000 || (rem == 0x1000 && r & 1 == 1) {
        r += 1;
    }
    sign | r as u16
}

#[inline]
fn h(b: &[u8], off: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[off], b[off + 1]]))
}
#[inline]
fn u32le(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

/// ggml `get_scale_min_k4`
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Dequantize `y.len()` elements from raw `src` bytes of type `ty`.
pub fn dequantize(ty: GgmlType, src: &[u8], y: &mut [f32]) -> Result<()> {
    let n = y.len() as u64;
    let need = ty.row_bytes(n) as usize;
    if src.len() < need {
        bail!("{ty:?}: need {need} bytes for {n} elements, got {}", src.len());
    }
    let (be, bb) = ty.block();
    let (be, bb) = (be as usize, bb as usize);
    for (blk, out) in src[..need].chunks_exact(bb).zip(y.chunks_exact_mut(be)) {
        match ty {
            GgmlType::F32 => out[0] = f32::from_le_bytes(blk.try_into().unwrap()),
            GgmlType::F16 => out[0] = h(blk, 0),
            GgmlType::BF16 => out[0] = f32::from_bits((u16::from_le_bytes([blk[0], blk[1]]) as u32) << 16),
            GgmlType::Q8_0 => q8_0(blk, out),
            GgmlType::Q2_K => q2_k(blk, out),
            GgmlType::Q3_K => q3_k(blk, out),
            GgmlType::Q4_K => q4_k(blk, out),
            GgmlType::Q5_K => q5_k(blk, out),
            GgmlType::Q6_K => q6_k(blk, out),
            GgmlType::IQ2_XXS => iq2_xxs(blk, out),
            GgmlType::IQ2_S => iq2_s(blk, out),
            GgmlType::IQ3_XXS => iq3_xxs(blk, out),
            GgmlType::IQ4_XS => iq4_xs(blk, out),
            _ => bail!("dequantize: {ty:?} not implemented"),
        }
    }
    Ok(())
}

pub fn is_supported(ty: GgmlType) -> bool {
    use GgmlType::*;
    matches!(ty, F32 | F16 | BF16 | Q8_0 | Q2_K | Q3_K | Q4_K | Q5_K | Q6_K | IQ2_XXS | IQ2_S | IQ3_XXS | IQ4_XS)
}

// block_q8_0 { half d; int8 qs[32]; }
fn q8_0(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for j in 0..32 {
        y[j] = (b[2 + j] as i8) as f32 * d;
    }
}

// block_q2_K { u8 scales[16]; u8 qs[64]; half d; half dmin; }
fn q2_k(b: &[u8], y: &mut [f32]) {
    let (scales, qs) = (&b[0..16], &b[16..80]);
    let (d, min) = (h(b, 80), h(b, 82));
    let mut yi = 0;
    let mut is = 0;
    for n in (0..QK_K).step_by(128) {
        let q = &qs[n / 4..];
        let mut shift = 0;
        for _j in 0..4 {
            for half in 0..2 {
                let sc = scales[is];
                is += 1;
                let (dl, ml) = (d * (sc & 0xF) as f32, min * (sc >> 4) as f32);
                for l in 0..16 {
                    y[yi] = dl * ((q[l + 16 * half] >> shift) & 3) as f32 - ml;
                    yi += 1;
                }
            }
            shift += 2;
        }
    }
}

// block_q3_K { u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d; }
fn q3_k(b: &[u8], y: &mut [f32]) {
    let (hm, qs, raw) = (&b[0..32], &b[32..96], &b[96..108]);
    let d_all = h(b, 108);
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [u32le(raw, 0), u32le(raw, 4), u32le(raw, 8), 0u32];
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for (i, a) in aux.iter().enumerate() {
        for k in 0..4 {
            scales[4 * i + k] = (a >> (8 * k)) as u8 as i8;
        }
    }
    let mut yi = 0;
    let mut is = 0;
    let mut m = 1u8;
    for n in (0..QK_K).step_by(128) {
        let q = &qs[n / 4..];
        let mut shift = 0;
        for _j in 0..4 {
            for half in 0..2 {
                let dl = d_all * (scales[is] as i32 - 32) as f32;
                is += 1;
                for l in 0..16 {
                    let qv = ((q[l + 16 * half] >> shift) & 3) as i32;
                    let hv = if hm[l + 16 * half] & m != 0 { 0 } else { 4 };
                    y[yi] = dl * (qv - hv) as f32;
                    yi += 1;
                }
            }
            shift += 2;
            m <<= 1;
        }
    }
}

// block_q4_K { half d; half dmin; u8 scales[12]; u8 qs[128]; }
fn q4_k(b: &[u8], y: &mut [f32]) {
    let (d, min) = (h(b, 0), h(b, 2));
    let (scales, qs) = (&b[4..16], &b[16..144]);
    let mut yi = 0;
    for (jb, q) in qs.chunks_exact(32).enumerate() {
        let (sc1, m1) = scale_min_k4(2 * jb, scales);
        let (sc2, m2) = scale_min_k4(2 * jb + 1, scales);
        let (d1, m1, d2, m2) = (d * sc1 as f32, min * m1 as f32, d * sc2 as f32, min * m2 as f32);
        for l in 0..32 {
            y[yi + l] = d1 * (q[l] & 0xF) as f32 - m1;
            y[yi + 32 + l] = d2 * (q[l] >> 4) as f32 - m2;
        }
        yi += 64;
    }
}

// block_q5_K { half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; }
fn q5_k(b: &[u8], y: &mut [f32]) {
    let (d, min) = (h(b, 0), h(b, 2));
    let (scales, qh, qs) = (&b[4..16], &b[16..48], &b[48..176]);
    let mut yi = 0;
    let (mut u1, mut u2) = (1u8, 2u8);
    for (jb, ql) in qs.chunks_exact(32).enumerate() {
        let (sc1, m1) = scale_min_k4(2 * jb, scales);
        let (sc2, m2) = scale_min_k4(2 * jb + 1, scales);
        let (d1, m1, d2, m2) = (d * sc1 as f32, min * m1 as f32, d * sc2 as f32, min * m2 as f32);
        for l in 0..32 {
            let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
            let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
            y[yi + l] = d1 * ((ql[l] & 0xF) + hi1) as f32 - m1;
            y[yi + 32 + l] = d2 * ((ql[l] >> 4) + hi2) as f32 - m2;
        }
        yi += 64;
        u1 <<= 2;
        u2 <<= 2;
    }
}

// block_q6_K { u8 ql[128]; u8 qh[64]; i8 scales[16]; half d; }
fn q6_k(b: &[u8], y: &mut [f32]) {
    let d = h(b, 208);
    for n in 0..2 {
        let ql = &b[64 * n..];
        let qh = &b[128 + 32 * n..];
        let sc = &b[192 + 8 * n..];
        let yo = &mut y[128 * n..];
        for l in 0..32 {
            let is = l / 16;
            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
            yo[l] = d * (sc[is] as i8) as f32 * q1 as f32;
            yo[l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
            yo[l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
            yo[l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
        }
    }
}

#[inline]
fn sgn(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

// block_iq2_xxs { half d; u16 qs[32]; }
fn iq2_xxs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    for ib32 in 0..8 {
        let a0 = u32le(b, 2 + 8 * ib32);
        let a1 = u32le(b, 2 + 8 * ib32 + 4);
        let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
        for l in 0..4 {
            let grid = IQ2XXS_GRID[((a0 >> (8 * l)) & 0xff) as usize].to_le_bytes();
            let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
            for j in 0..8 {
                y[32 * ib32 + 8 * l + j] = db * grid[j] as f32 * sgn(signs, j);
            }
        }
    }
}

// block_iq2_s { half d; u8 qs[64]; u8 qh[8]; u8 scales[8]; }  (qs[32..64] are the signs)
fn iq2_s(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let (qs, signs, qh, scales) = (&b[2..34], &b[34..66], &b[66..74], &b[74..82]);
    for ib32 in 0..8 {
        let db = [
            d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25,
            d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25,
        ];
        for l in 0..4 {
            let dl = db[l / 2];
            let idx = qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
            let grid = IQ2S_GRID[idx].to_le_bytes();
            let s = signs[4 * ib32 + l];
            for j in 0..8 {
                y[32 * ib32 + 8 * l + j] = dl * grid[j] as f32 * sgn(s, j);
            }
        }
    }
}

// block_iq3_xxs { half d; u8 qs[96]; }  (qs[64..96] = scales_and_signs)
fn iq3_xxs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let qs = &b[2..66];
    for ib32 in 0..8 {
        let aux = u32le(b, 66 + 4 * ib32);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for l in 0..4 {
            let signs = KSIGNS_IQ2XS[((aux >> (7 * l)) & 127) as usize];
            let g1 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l] as usize].to_le_bytes();
            let g2 = IQ3XXS_GRID[qs[8 * ib32 + 2 * l + 1] as usize].to_le_bytes();
            for j in 0..4 {
                y[32 * ib32 + 8 * l + j] = db * g1[j] as f32 * sgn(signs, j);
                y[32 * ib32 + 8 * l + j + 4] = db * g2[j] as f32 * sgn(signs, j + 4);
            }
        }
    }
}

// block_iq4_xs { half d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }
fn iq4_xs(b: &[u8], y: &mut [f32]) {
    let d = h(b, 0);
    let scales_h = u16::from_le_bytes([b[2], b[3]]);
    let (scales_l, qs) = (&b[4..8], &b[8..136]);
    for ib in 0..8 {
        let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32 | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
        let dl = d * (ls - 32) as f32;
        for j in 0..16 {
            y[32 * ib + j] = dl * KVALUES_IQ4NL[(qs[16 * ib + j] & 0xf) as usize] as f32;
            y[32 * ib + j + 16] = dl * KVALUES_IQ4NL[(qs[16 * ib + j] >> 4) as usize] as f32;
        }
    }
}

/// Byte offsets of the f16 fields inside a block (used to generate sane random test blocks).
pub fn half_offsets(ty: GgmlType) -> &'static [usize] {
    use GgmlType::*;
    match ty {
        Q8_0 | IQ2_XXS | IQ2_S | IQ3_XXS | IQ4_XS => &[0],
        Q2_K => &[80, 82],
        Q3_K => &[108],
        Q4_K | Q5_K => &[0, 2],
        Q6_K => &[208],
        _ => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f16_roundtrip() {
        for h in 0u16..=0xffff {
            let f = f16_to_f32(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "h={h:#x} f={f}");
        }
    }
}
