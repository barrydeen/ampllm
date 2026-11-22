use ampllm_gguf::quants::{
    f16_bytes_to_f32, get_scale_min_k4, ue4m3_to_f32, QK4_0, QK4_NL, QK8_0, QK_K, QK_NVFP4,
};
use ampllm_gguf::{
    BlockIq4Nl, BlockIq4Xs, BlockNvfp4, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8_0,
    GgmlType, KVALUES_IQ4NL, KVALUES_NVFP4,
};
use half::{bf16, f16};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// 32-byte-aligned CPU activation block so AVX2/AVX-512 loads of `qs` are always aligned.
#[repr(C, align(32))]
#[derive(Clone, Copy, Default)]
pub struct CpuBlockQ8_1 {
    pub qs: [i8; 32],
    pub d: f32,
    pub s: f32,
    pub sum_q: i32,
}

/// Pre-biased unsigned codebook `(KVALUES_IQ4NL[i] + 128) as u8` for `vpdpbusd` / `vpmaddubsw`.
const KVALUES_IQ4NL_U8: [u8; 16] = [
    1, 24, 45, 63, 79, 93, 106, 118, 129, 141, 153, 166, 181, 197, 217, 241,
];

/// Pre-biased unsigned codebook `(KVALUES_NVFP4[i] + 128) as u8` for `vpdpbusd` / `vpmaddubsw`.
const KVALUES_NVFP4_U8: [u8; 16] = [
    128, 129, 130, 131, 132, 134, 136, 140, 128, 127, 126, 125, 124, 122, 120, 116,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdTier {
    Avx512Vnni,
    Avx2,
    Scalar,
}

pub fn active_simd_tier() -> SimdTier {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vnni")
        {
            return SimdTier::Avx512Vnni;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return SimdTier::Avx2;
        }
    }
    SimdTier::Scalar
}

/// Quantize `x` (`x.len() % 32 == 0`) into `CpuBlockQ8_1` matching GPU `quantize_f32_to_q8_1_kernel`.
pub fn quantize_f32_to_q8_1_cpu(x: &[f32], out: &mut [CpuBlockQ8_1]) {
    debug_assert_eq!(x.len() % 32, 0);
    debug_assert_eq!(out.len(), x.len() / 32);

    for (chunk, dst) in x.chunks_exact(32).zip(out.iter_mut()) {
        let mut amax = 0.0f32;
        for &v in chunk {
            let a = v.abs();
            if a > amax {
                amax = a;
            }
        }
        let d = amax / 127.0f32;
        let mut sum_q = 0i32;
        for i in 0..32 {
            let q = if d > 0.0f32 {
                (chunk[i] / d).round_ties_even().clamp(-127.0, 127.0) as i8
            } else {
                0
            };
            dst.qs[i] = q;
            sum_q += q as i32;
        }
        dst.d = f16::from_f32(d).to_f32();
        dst.s = f16::from_f32(d * (sum_q as f32)).to_f32();
        dst.sum_q = sum_q;
    }
}

#[inline]
pub fn vec_dot_q_q8_1(ggml_type: GgmlType, w_row: &[u8], x_q8: &[CpuBlockQ8_1], x_f32: Option<&[f32]>) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vnni")
        {
            return unsafe { vec_dot_avx512_vnni(ggml_type, w_row, x_q8, x_f32) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_avx2(ggml_type, w_row, x_q8, x_f32) };
        }
    }
    vec_dot_scalar(ggml_type, w_row, x_q8, x_f32)
}

pub fn vec_dot_scalar(
    ggml_type: GgmlType,
    w_row: &[u8],
    x_q8: &[CpuBlockQ8_1],
    x_f32: Option<&[f32]>,
) -> f32 {
    let k = x_q8.len() * 32;
    match ggml_type {
        GgmlType::F32 => {
            let xf = x_f32.expect("x_f32 required for dense F32 dot");
            let w: &[f32] = bytemuck::cast_slice(&w_row[..k * 4]);
            w.iter().zip(xf.iter()).map(|(&a, &b)| a * b).sum()
        }
        GgmlType::F16 => {
            let xf = x_f32.expect("x_f32 required for dense F16 dot");
            let w: &[[u8; 2]] = bytemuck::cast_slice(&w_row[..k * 2]);
            w.iter().zip(xf.iter()).map(|(&a, &b)| f16::from_le_bytes(a).to_f32() * b).sum()
        }
        GgmlType::Bf16 => {
            let xf = x_f32.expect("x_f32 required for dense BF16 dot");
            let w: &[[u8; 2]] = bytemuck::cast_slice(&w_row[..k * 2]);
            w.iter().zip(xf.iter()).map(|(&a, &b)| bf16::from_le_bytes(a).to_f32() * b).sum()
        }
        GgmlType::Q8_0 => {
            let blocks: &[BlockQ8_0] = bytemuck::cast_slice(w_row);
            let mut sumf = 0.0f32;
            for (b, x) in blocks.iter().zip(x_q8.iter()) {
                let mut sumi = 0i32;
                for j in 0..QK8_0 {
                    sumi += (b.qs[j] as i32) * (x.qs[j] as i32);
                }
                sumf += f16_bytes_to_f32(b.d) * x.d * (sumi as f32);
            }
            sumf
        }
        GgmlType::Q4_0 => {
            let blocks: &[BlockQ4_0] = bytemuck::cast_slice(w_row);
            let mut sumf = 0.0f32;
            for (b, x) in blocks.iter().zip(x_q8.iter()) {
                let mut sumi = 0i32;
                for j in 0..(QK4_0 / 2) {
                    let q0 = (b.qs[j] & 0xF) as i32;
                    let q1 = (b.qs[j] >> 4) as i32;
                    sumi += q0 * (x.qs[j] as i32) + q1 * (x.qs[j + QK4_0 / 2] as i32);
                }
                sumi -= 8 * x.sum_q;
                sumf += f16_bytes_to_f32(b.d) * x.d * (sumi as f32);
            }
            sumf
        }
        GgmlType::Iq4Nl => {
            let blocks: &[BlockIq4Nl] = bytemuck::cast_slice(w_row);
            let mut sumf = 0.0f32;
            for (b, x) in blocks.iter().zip(x_q8.iter()) {
                let mut sumi = 0i32;
                for j in 0..(QK4_NL / 2) {
                    let q0 = KVALUES_IQ4NL[(b.qs[j] & 0xF) as usize] as i32;
                    let q1 = KVALUES_IQ4NL[(b.qs[j] >> 4) as usize] as i32;
                    sumi += q0 * (x.qs[j] as i32) + q1 * (x.qs[j + QK4_NL / 2] as i32);
                }
                sumf += f16_bytes_to_f32(b.d) * x.d * (sumi as f32);
            }
            sumf
        }
        GgmlType::Iq4Xs => {
            let blocks: &[BlockIq4Xs] = bytemuck::cast_slice(w_row);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                let dw = f16_bytes_to_f32(b.d);
                let scales_h = u16::from_le_bytes(b.scales_h);
                let xb = &x_q8[bi * (QK_K / 32)..(bi + 1) * (QK_K / 32)];
                for ib in 0..(QK_K / 32) {
                    let ls = (((b.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32)
                        | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                    let dl = dw * ((ls - 32) as f32);
                    let qs = &b.qs[ib * 16..(ib + 1) * 16];
                    let x = &xb[ib];
                    let mut sumi = 0i32;
                    for j in 0..16 {
                        let q0 = KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as i32;
                        let q1 = KVALUES_IQ4NL[(qs[j] >> 4) as usize] as i32;
                        sumi += q0 * (x.qs[j] as i32) + q1 * (x.qs[j + 16] as i32);
                    }
                    sumf += dl * x.d * (sumi as f32);
                }
            }
            sumf
        }
        GgmlType::Q4K => {
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(w_row);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                let dw = f16_bytes_to_f32(b.d);
                let dmin = f16_bytes_to_f32(b.dmin);
                let xb = &x_q8[bi * 8..(bi + 1) * 8];
                for ib in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(ib * 2, &b.scales);
                    let (sc2, m2) = get_scale_min_k4(ib * 2 + 1, &b.scales);
                    let qs = &b.qs[ib * 32..(ib + 1) * 32];
                    let x0 = &xb[ib * 2];
                    let x1 = &xb[ib * 2 + 1];
                    let (mut sumi0, mut sumi1) = (0i32, 0i32);
                    for j in 0..32 {
                        sumi0 += ((qs[j] & 0xF) as i32) * (x0.qs[j] as i32);
                        sumi1 += ((qs[j] >> 4) as i32) * (x1.qs[j] as i32);
                    }
                    sumf += dw * ((sc1 as f32) * x0.d * (sumi0 as f32) + (sc2 as f32) * x1.d * (sumi1 as f32))
                        - dmin * ((m1 as f32) * x0.s + (m2 as f32) * x1.s);
                }
            }
            sumf
        }
        GgmlType::Q5K => {
            let blocks: &[BlockQ5K] = bytemuck::cast_slice(w_row);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                let dw = f16_bytes_to_f32(b.d);
                let dmin = f16_bytes_to_f32(b.dmin);
                let xb = &x_q8[bi * 8..(bi + 1) * 8];
                let mut u1: u8 = 1;
                let mut u2: u8 = 2;
                for ib in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(ib * 2, &b.scales);
                    let (sc2, m2) = get_scale_min_k4(ib * 2 + 1, &b.scales);
                    let qs = &b.qs[ib * 32..(ib + 1) * 32];
                    let x0 = &xb[ib * 2];
                    let x1 = &xb[ib * 2 + 1];
                    let (mut sumi0, mut sumi1) = (0i32, 0i32);
                    for j in 0..32 {
                        let q0 = (qs[j] & 0xF) as i32 + if b.qh[j] & u1 != 0 { 16 } else { 0 };
                        let q1 = (qs[j] >> 4) as i32 + if b.qh[j] & u2 != 0 { 16 } else { 0 };
                        sumi0 += q0 * (x0.qs[j] as i32);
                        sumi1 += q1 * (x1.qs[j] as i32);
                    }
                    sumf += dw * ((sc1 as f32) * x0.d * (sumi0 as f32) + (sc2 as f32) * x1.d * (sumi1 as f32))
                        - dmin * ((m1 as f32) * x0.s + (m2 as f32) * x1.s);
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
            sumf
        }
        GgmlType::Q6K => {
            let blocks: &[BlockQ6K] = bytemuck::cast_slice(w_row);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                let dw = f16_bytes_to_f32(b.d);
                let xb = &x_q8[bi * 8..(bi + 1) * 8];
                for half in 0..2 {
                    let ql = &b.ql[half * 64..(half + 1) * 64];
                    let qh = &b.qh[half * 32..(half + 1) * 32];
                    let sc = &b.scales[half * 8..(half + 1) * 8];
                    let xh = &xb[half * 4..(half + 1) * 4];
                    for is in 0..2 {
                        let (mut s1, mut s2, mut s3, mut s4) = (0i32, 0i32, 0i32, 0i32);
                        for l16 in 0..16 {
                            let l = is * 16 + l16;
                            let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                            let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                            s1 += q1 * (xh[0].qs[l] as i32);
                            s2 += q2 * (xh[1].qs[l] as i32);
                            s3 += q3 * (xh[2].qs[l] as i32);
                            s4 += q4 * (xh[3].qs[l] as i32);
                        }
                        sumf += dw
                            * ((sc[is] as f32) * xh[0].d * (s1 as f32)
                                + (sc[is + 2] as f32) * xh[1].d * (s2 as f32)
                                + (sc[is + 4] as f32) * xh[2].d * (s3 as f32)
                                + (sc[is + 6] as f32) * xh[3].d * (s4 as f32));
                    }
                }
            }
            sumf
        }
        GgmlType::Nvfp4 => {
            let _ = QK_NVFP4;
            let blocks: &[BlockNvfp4] = bytemuck::cast_slice(w_row);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                for sb in 0..4 {
                    let d = ue4m3_to_f32(b.d[sb]);
                    let qs = &b.qs[sb * 8..(sb + 1) * 8];
                    let x = &x_q8[bi * 2 + (sb / 2)];
                    let x_off = (sb % 2) * 16;
                    let mut sumi = 0i32;
                    for j in 0..8 {
                        let q0 = KVALUES_NVFP4[(qs[j] & 0xF) as usize] as i32;
                        let q1 = KVALUES_NVFP4[(qs[j] >> 4) as usize] as i32;
                        sumi += q0 * (x.qs[x_off + j] as i32) + q1 * (x.qs[x_off + 8 + j] as i32);
                    }
                    sumf += d * x.d * (sumi as f32);
                }
            }
            sumf
        }
        other => panic!("vec_dot_scalar unsupported type {other:?}"),
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_epi32_128(v: __m128i) -> i32 {
    let hi = _mm_shuffle_epi32::<0x4E>(v);
    let s1 = _mm_add_epi32(v, hi);
    let s2 = _mm_add_epi32(s1, _mm_shuffle_epi32::<0xB1>(s1));
    _mm_cvtsi128_si32(s2)
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn hsum_epi32_256(v: __m256i) -> i32 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    hsum_epi32_128(_mm_add_epi32(lo, hi))
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn unpack_nibbles_32_128(raw16: __m128i, mask0f: __m128i) -> __m256i {
    let lo = _mm_and_si128(raw16, mask0f);
    let hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f);
    _mm256_set_m128i(hi, lo)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub unsafe fn vec_dot_avx512_vnni(
    ggml_type: GgmlType,
    w_row: &[u8],
    x_q8: &[CpuBlockQ8_1],
    x_f32: Option<&[f32]>,
) -> f32 {
    match ggml_type {
        GgmlType::Iq4Nl => {
            let blocks: &[BlockIq4Nl] = bytemuck::cast_slice(w_row);
            let nb = blocks.len();
            let lut128 = _mm_loadu_si128(KVALUES_IQ4NL_U8.as_ptr() as *const __m128i);
            let lut512 = _mm512_broadcast_i32x4(lut128);
            let mask0f = _mm_set1_epi8(0x0F);

            let mut sumf = 0.0f32;
            let mut b = 0usize;
            while b + 1 < nb {
                let raw0 = _mm_loadu_si128(blocks[b].qs.as_ptr() as *const __m128i);
                let raw1 = _mm_loadu_si128(blocks[b + 1].qs.as_ptr() as *const __m128i);
                let idx0 = unpack_nibbles_32_128(raw0, mask0f);
                let idx1 = unpack_nibbles_32_128(raw1, mask0f);
                let idx512 = _mm512_inserti64x4(_mm512_castsi256_si512(idx0), idx1, 1);

                let w_u8 = _mm512_shuffle_epi8(lut512, idx512);
                let x0 = _mm256_load_si256(x_q8[b].qs.as_ptr() as *const __m256i);
                let x1 = _mm256_load_si256(x_q8[b + 1].qs.as_ptr() as *const __m256i);
                let x512 = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);

                let dots = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_u8, x512);
                let s0 = hsum_epi32_256(_mm512_castsi512_si256(dots)) - 128 * x_q8[b].sum_q;
                let s1 = hsum_epi32_256(_mm512_extracti64x4_epi64::<1>(dots)) - 128 * x_q8[b + 1].sum_q;

                sumf += f16_bytes_to_f32(blocks[b].d) * x_q8[b].d * (s0 as f32);
                sumf += f16_bytes_to_f32(blocks[b + 1].d) * x_q8[b + 1].d * (s1 as f32);
                b += 2;
            }
            if b < nb {
                let raw0 = _mm_loadu_si128(blocks[b].qs.as_ptr() as *const __m128i);
                let idx0 = unpack_nibbles_32_128(raw0, mask0f);
                let lut256 = _mm256_broadcastsi128_si256(lut128);
                let w_u8 = _mm256_shuffle_epi8(lut256, idx0);
                let x0 = _mm256_load_si256(x_q8[b].qs.as_ptr() as *const __m256i);
                let dots = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_u8, x0);
                let s0 = hsum_epi32_256(dots) - 128 * x_q8[b].sum_q;
                sumf += f16_bytes_to_f32(blocks[b].d) * x_q8[b].d * (s0 as f32);
            }
            sumf
        }
        GgmlType::Q8_0 => {
            let blocks: &[BlockQ8_0] = bytemuck::cast_slice(w_row);
            let nb = blocks.len();
            let sign_flip = _mm512_set1_epi8(-128i8);
            let sign_flip256 = _mm256_set1_epi8(-128i8);

            let mut sumf = 0.0f32;
            let mut b = 0usize;
            while b + 1 < nb {
                let w0 = _mm256_loadu_si256(blocks[b].qs.as_ptr() as *const __m256i);
                let w1 = _mm256_loadu_si256(blocks[b + 1].qs.as_ptr() as *const __m256i);
                let w512 = _mm512_xor_si512(
                    _mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1),
                    sign_flip,
                );
                let x0 = _mm256_load_si256(x_q8[b].qs.as_ptr() as *const __m256i);
                let x1 = _mm256_load_si256(x_q8[b + 1].qs.as_ptr() as *const __m256i);
                let x512 = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);

                let dots = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w512, x512);
                let s0 = hsum_epi32_256(_mm512_castsi512_si256(dots)) - 128 * x_q8[b].sum_q;
                let s1 = hsum_epi32_256(_mm512_extracti64x4_epi64::<1>(dots)) - 128 * x_q8[b + 1].sum_q;

                sumf += f16_bytes_to_f32(blocks[b].d) * x_q8[b].d * (s0 as f32);
                sumf += f16_bytes_to_f32(blocks[b + 1].d) * x_q8[b + 1].d * (s1 as f32);
                b += 2;
            }
            if b < nb {
                let w0 = _mm256_xor_si256(
                    _mm256_loadu_si256(blocks[b].qs.as_ptr() as *const __m256i),
                    sign_flip256,
                );
                let x0 = _mm256_load_si256(x_q8[b].qs.as_ptr() as *const __m256i);
                let dots = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w0, x0);
                let s0 = hsum_epi32_256(dots) - 128 * x_q8[b].sum_q;
                sumf += f16_bytes_to_f32(blocks[b].d) * x_q8[b].d * (s0 as f32);
            }
            sumf
        }
        GgmlType::Q4_0 => {
            let blocks: &[BlockQ4_0] = bytemuck::cast_slice(w_row);
            let nb = blocks.len();
            let mask0f = _mm_set1_epi8(0x0F);

            let mut sumf = 0.0f32;
            let mut b = 0usize;
            while b + 1 < nb {
                let idx0 = unpack_nibbles_32_128(_mm_loadu_si128(blocks[b].qs.as_ptr() as *const __m128i), mask0f);
                let idx1 = unpack_nibbles_32_128(_mm_loadu_si128(blocks[b + 1].qs.as_ptr() as *const __m128i), mask0f);
                let w512 = _mm512_inserti64x4(_mm512_castsi256_si512(idx0), idx1, 1);
                let x0 = _mm256_load_si256(x_q8[b].qs.as_ptr() as *const __m256i);
                let x1 = _mm256_load_si256(x_q8[b + 1].qs.as_ptr() as *const __m256i);
                let x512 = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);

                let dots = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w512, x512);
                let s0 = hsum_epi32_256(_mm512_castsi512_si256(dots)) - 8 * x_q8[b].sum_q;
                let s1 = hsum_epi32_256(_mm512_extracti64x4_epi64::<1>(dots)) - 8 * x_q8[b + 1].sum_q;
                sumf += f16_bytes_to_f32(blocks[b].d) * x_q8[b].d * (s0 as f32);
                sumf += f16_bytes_to_f32(blocks[b + 1].d) * x_q8[b + 1].d * (s1 as f32);
                b += 2;
            }
            if b < nb {
                let idx0 = unpack_nibbles_32_128(_mm_loadu_si128(blocks[b].qs.as_ptr() as *const __m128i), mask0f);
                let x0 = _mm256_load_si256(x_q8[b].qs.as_ptr() as *const __m256i);
                let dots = _mm256_dpbusd_epi32(_mm256_setzero_si256(), idx0, x0);
                let s0 = hsum_epi32_256(dots) - 8 * x_q8[b].sum_q;
                sumf += f16_bytes_to_f32(blocks[b].d) * x_q8[b].d * (s0 as f32);
            }
            sumf
        }
        GgmlType::Iq4Xs => {
            let blocks: &[BlockIq4Xs] = bytemuck::cast_slice(w_row);
            let lut128 = _mm_loadu_si128(KVALUES_IQ4NL_U8.as_ptr() as *const __m128i);
            let lut512 = _mm512_broadcast_i32x4(lut128);
            let mask0f = _mm_set1_epi8(0x0F);

            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                let dw = f16_bytes_to_f32(b.d);
                let scales_h = u16::from_le_bytes(b.scales_h);
                let xb = &x_q8[bi * 8..(bi + 1) * 8];
                for pair in 0..4 {
                    let ib0 = pair * 2;
                    let ib1 = ib0 + 1;
                    let idx0 = unpack_nibbles_32_128(_mm_loadu_si128(b.qs[ib0 * 16..].as_ptr() as *const __m128i), mask0f);
                    let idx1 = unpack_nibbles_32_128(_mm_loadu_si128(b.qs[ib1 * 16..].as_ptr() as *const __m128i), mask0f);
                    let w512 = _mm512_shuffle_epi8(lut512, _mm512_inserti64x4(_mm512_castsi256_si512(idx0), idx1, 1));
                    let x0 = _mm256_load_si256(xb[ib0].qs.as_ptr() as *const __m256i);
                    let x1 = _mm256_load_si256(xb[ib1].qs.as_ptr() as *const __m256i);
                    let x512 = _mm512_inserti64x4(_mm512_castsi256_si512(x0), x1, 1);
                    let dots = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w512, x512);

                    let ls0 = (((b.scales_l[ib0 / 2] >> (4 * (ib0 % 2))) & 0xF) as i32)
                        | ((((scales_h >> (2 * ib0)) & 3) as i32) << 4);
                    let ls1 = (((b.scales_l[ib1 / 2] >> (4 * (ib1 % 2))) & 0xF) as i32)
                        | ((((scales_h >> (2 * ib1)) & 3) as i32) << 4);

                    let s0 = hsum_epi32_256(_mm512_castsi512_si256(dots)) - 128 * xb[ib0].sum_q;
                    let s1 = hsum_epi32_256(_mm512_extracti64x4_epi64::<1>(dots)) - 128 * xb[ib1].sum_q;
                    sumf += dw * ((ls0 - 32) as f32) * xb[ib0].d * (s0 as f32)
                        + dw * ((ls1 - 32) as f32) * xb[ib1].d * (s1 as f32);
                }
            }
            sumf
        }
        GgmlType::Q4K => {
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(w_row);
            let mask0f = _mm256_set1_epi8(0x0F);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                let dw = f16_bytes_to_f32(b.d);
                let dmin = f16_bytes_to_f32(b.dmin);
                let xb = &x_q8[bi * 8..(bi + 1) * 8];
                for ib in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(ib * 2, &b.scales);
                    let (sc2, m2) = get_scale_min_k4(ib * 2 + 1, &b.scales);
                    let raw = _mm256_loadu_si256(b.qs[ib * 32..].as_ptr() as *const __m256i);
                    let lo = _mm256_and_si256(raw, mask0f);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0f);
                    let x0 = _mm256_load_si256(xb[ib * 2].qs.as_ptr() as *const __m256i);
                    let x1 = _mm256_load_si256(xb[ib * 2 + 1].qs.as_ptr() as *const __m256i);
                    let sumi0 = hsum_epi32_256(_mm256_dpbusd_epi32(_mm256_setzero_si256(), lo, x0));
                    let sumi1 = hsum_epi32_256(_mm256_dpbusd_epi32(_mm256_setzero_si256(), hi, x1));
                    sumf += dw
                        * ((sc1 as f32) * xb[ib * 2].d * (sumi0 as f32)
                            + (sc2 as f32) * xb[ib * 2 + 1].d * (sumi1 as f32))
                        - dmin * ((m1 as f32) * xb[ib * 2].s + (m2 as f32) * xb[ib * 2 + 1].s);
                }
            }
            sumf
        }
        GgmlType::Q5K => {
            let blocks: &[BlockQ5K] = bytemuck::cast_slice(w_row);
            let mask0f = _mm256_set1_epi8(0x0F);
            let v16 = _mm256_set1_epi8(16);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                let dw = f16_bytes_to_f32(b.d);
                let dmin = f16_bytes_to_f32(b.dmin);
                let xb = &x_q8[bi * 8..(bi + 1) * 8];
                let qh_vec = _mm256_loadu_si256(b.qh.as_ptr() as *const __m256i);
                let mut u1: i8 = 1;
                let mut u2: i8 = 2;
                for ib in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(ib * 2, &b.scales);
                    let (sc2, m2) = get_scale_min_k4(ib * 2 + 1, &b.scales);
                    let raw = _mm256_loadu_si256(b.qs[ib * 32..].as_ptr() as *const __m256i);
                    let m_u1 = _mm256_set1_epi8(u1);
                    let m_u2 = _mm256_set1_epi8(u2);
                    let h0 = _mm256_and_si256(
                        _mm256_cmpeq_epi8(_mm256_and_si256(qh_vec, m_u1), m_u1),
                        v16,
                    );
                    let h1 = _mm256_and_si256(
                        _mm256_cmpeq_epi8(_mm256_and_si256(qh_vec, m_u2), m_u2),
                        v16,
                    );
                    let lo = _mm256_or_si256(_mm256_and_si256(raw, mask0f), h0);
                    let hi = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0f), h1);
                    let x0 = _mm256_load_si256(xb[ib * 2].qs.as_ptr() as *const __m256i);
                    let x1 = _mm256_load_si256(xb[ib * 2 + 1].qs.as_ptr() as *const __m256i);
                    let sumi0 = hsum_epi32_256(_mm256_dpbusd_epi32(_mm256_setzero_si256(), lo, x0));
                    let sumi1 = hsum_epi32_256(_mm256_dpbusd_epi32(_mm256_setzero_si256(), hi, x1));
                    sumf += dw
                        * ((sc1 as f32) * xb[ib * 2].d * (sumi0 as f32)
                            + (sc2 as f32) * xb[ib * 2 + 1].d * (sumi1 as f32))
                        - dmin * ((m1 as f32) * xb[ib * 2].s + (m2 as f32) * xb[ib * 2 + 1].s);
                    u1 = u1.wrapping_shl(2);
                    u2 = u2.wrapping_shl(2);
                }
            }
            sumf
        }
        GgmlType::Q6K => {
            let blocks: &[BlockQ6K] = bytemuck::cast_slice(w_row);
            let mask0f = _mm256_set1_epi8(0x0F);
            let mask03 = _mm256_set1_epi8(0x03);
            let ones_u8 = _mm256_set1_epi8(1);
            let c32 = _mm256_set1_epi32(32);

            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                let dw = f16_bytes_to_f32(b.d);
                let xb = &x_q8[bi * 8..(bi + 1) * 8];
                for half in 0..2 {
                    let ql = &b.ql[half * 64..];
                    let qh = &b.qh[half * 32..];
                    let sc = &b.scales[half * 8..];
                    let xh = &xb[half * 4..];

                    let ql0 = _mm256_loadu_si256(ql.as_ptr() as *const __m256i);
                    let ql1 = _mm256_loadu_si256(ql[32..].as_ptr() as *const __m256i);
                    let qhv = _mm256_loadu_si256(qh.as_ptr() as *const __m256i);

                    let q1 = _mm256_or_si256(
                        _mm256_and_si256(ql0, mask0f),
                        _mm256_slli_epi16(_mm256_and_si256(qhv, mask03), 4),
                    );
                    let q2 = _mm256_or_si256(
                        _mm256_and_si256(ql1, mask0f),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), mask03), 4),
                    );
                    let q3 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql0, 4), mask0f),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), mask03), 4),
                    );
                    let q4 = _mm256_or_si256(
                        _mm256_and_si256(_mm256_srli_epi16(ql1, 4), mask0f),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), mask03), 4),
                    );

                    let x0 = _mm256_load_si256(xh[0].qs.as_ptr() as *const __m256i);
                    let x1 = _mm256_load_si256(xh[1].qs.as_ptr() as *const __m256i);
                    let x2 = _mm256_load_si256(xh[2].qs.as_ptr() as *const __m256i);
                    let x3 = _mm256_load_si256(xh[3].qs.as_ptr() as *const __m256i);

                    // Lower 128 bits = is=0 (16 elems), upper 128 bits = is=1 (16 elems)
                    let p0 = _mm256_sub_epi32(
                        _mm256_dpbusd_epi32(_mm256_setzero_si256(), q1, x0),
                        _mm256_mullo_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), ones_u8, x0), c32),
                    );
                    let p1 = _mm256_sub_epi32(
                        _mm256_dpbusd_epi32(_mm256_setzero_si256(), q2, x1),
                        _mm256_mullo_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), ones_u8, x1), c32),
                    );
                    let p2 = _mm256_sub_epi32(
                        _mm256_dpbusd_epi32(_mm256_setzero_si256(), q3, x2),
                        _mm256_mullo_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), ones_u8, x2), c32),
                    );
                    let p3 = _mm256_sub_epi32(
                        _mm256_dpbusd_epi32(_mm256_setzero_si256(), q4, x3),
                        _mm256_mullo_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), ones_u8, x3), c32),
                    );

                    let sum_pairs = |v: __m256i| -> (i32, i32) {
                        (
                            hsum_epi32_128(_mm256_castsi256_si128(v)),
                            hsum_epi32_128(_mm256_extracti128_si256::<1>(v)),
                        )
                    };
                    let (a0, b0) = sum_pairs(p0);
                    let (a1, b1) = sum_pairs(p1);
                    let (a2, b2) = sum_pairs(p2);
                    let (a3, b3) = sum_pairs(p3);

                    sumf += dw
                        * (((sc[0] as f32) * (a0 as f32) + (sc[1] as f32) * (b0 as f32)) * xh[0].d
                            + ((sc[2] as f32) * (a1 as f32) + (sc[3] as f32) * (b1 as f32)) * xh[1].d
                            + ((sc[4] as f32) * (a2 as f32) + (sc[5] as f32) * (b2 as f32)) * xh[2].d
                            + ((sc[6] as f32) * (a3 as f32) + (sc[7] as f32) * (b3 as f32)) * xh[3].d);
                }
            }
            sumf
        }
        GgmlType::F32 => {
            let xf = x_f32.expect("x_f32 required for dense F32 dot");
            let k = xf.len();
            let wf = w_row.as_ptr() as *const f32;
            let mut acc = _mm512_setzero_ps();
            let mut i = 0usize;
            while i + 16 <= k {
                let va = _mm512_loadu_ps(wf.add(i));
                let vb = _mm512_loadu_ps(xf.as_ptr().add(i));
                acc = _mm512_fmadd_ps(va, vb, acc);
                i += 16;
            }
            let mut sum = _mm512_reduce_add_ps(acc);
            while i < k {
                sum += *wf.add(i) * xf[i];
                i += 1;
            }
            sum
        }
        GgmlType::Nvfp4 => {
            let blocks: &[BlockNvfp4] = bytemuck::cast_slice(w_row);
            let lut128 = _mm_loadu_si128(KVALUES_NVFP4_U8.as_ptr() as *const __m128i);
            let lut256 = _mm256_broadcastsi128_si256(lut128);
            let mask0f = _mm_set1_epi8(0x0F);
            let ones_u8 = _mm256_set1_epi8(1);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                for half in 0..2 {
                    let raw16 = _mm_loadu_si128(b.qs[half * 16..].as_ptr() as *const __m128i);
                    let lo = _mm_and_si128(raw16, mask0f);
                    let hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f);
                    let idx_sb0 = _mm_unpacklo_epi64(lo, hi);
                    let idx_sb1 = _mm_unpackhi_epi64(lo, hi);
                    let idx256 = _mm256_set_m128i(idx_sb1, idx_sb0);
                    let w_u8 = _mm256_shuffle_epi8(lut256, idx256);
                    let x = &x_q8[bi * 2 + half];
                    let x_i8 = _mm256_load_si256(x.qs.as_ptr() as *const __m256i);
                    let dots = _mm256_sub_epi32(
                        _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_u8, x_i8),
                        _mm256_slli_epi32(_mm256_dpbusd_epi32(_mm256_setzero_si256(), ones_u8, x_i8), 7),
                    );
                    let s0 = hsum_epi32_128(_mm256_castsi256_si128(dots));
                    let s1 = hsum_epi32_128(_mm256_extracti128_si256::<1>(dots));
                    let d0 = ue4m3_to_f32(b.d[half * 2]);
                    let d1 = ue4m3_to_f32(b.d[half * 2 + 1]);
                    sumf += (d0 * (s0 as f32) + d1 * (s1 as f32)) * x.d;
                }
            }
            sumf
        }
        _ => vec_dot_scalar(ggml_type, w_row, x_q8, x_f32),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_dot_avx2(
    ggml_type: GgmlType,
    w_row: &[u8],
    x_q8: &[CpuBlockQ8_1],
    x_f32: Option<&[f32]>,
) -> f32 {
    let ones16 = _mm256_set1_epi16(1);
    match ggml_type {
        GgmlType::Iq4Nl => {
            let blocks: &[BlockIq4Nl] = bytemuck::cast_slice(w_row);
            let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(KVALUES_IQ4NL_U8.as_ptr() as *const __m128i));
            let mask0f = _mm_set1_epi8(0x0F);
            let mut sumf = 0.0f32;
            for (b, x) in blocks.iter().zip(x_q8.iter()) {
                let idx = unpack_nibbles_32_128(_mm_loadu_si128(b.qs.as_ptr() as *const __m128i), mask0f);
                let w_u8 = _mm256_shuffle_epi8(lut, idx);
                let x_i8 = _mm256_load_si256(x.qs.as_ptr() as *const __m256i);
                let p16 = _mm256_maddubs_epi16(w_u8, x_i8);
                let s32 = hsum_epi32_256(_mm256_madd_epi16(p16, ones16)) - 128 * x.sum_q;
                sumf += f16_bytes_to_f32(b.d) * x.d * (s32 as f32);
            }
            sumf
        }
        GgmlType::Q8_0 => {
            let blocks: &[BlockQ8_0] = bytemuck::cast_slice(w_row);
            let sign_flip = _mm256_set1_epi8(-128i8);
            let mut sumf = 0.0f32;
            for (b, x) in blocks.iter().zip(x_q8.iter()) {
                let w_u8 = _mm256_xor_si256(_mm256_loadu_si256(b.qs.as_ptr() as *const __m256i), sign_flip);
                let x_i8 = _mm256_load_si256(x.qs.as_ptr() as *const __m256i);
                let p16 = _mm256_maddubs_epi16(w_u8, x_i8);
                let s32 = hsum_epi32_256(_mm256_madd_epi16(p16, ones16)) - 128 * x.sum_q;
                sumf += f16_bytes_to_f32(b.d) * x.d * (s32 as f32);
            }
            sumf
        }
        GgmlType::Q4_0 => {
            let blocks: &[BlockQ4_0] = bytemuck::cast_slice(w_row);
            let mask0f = _mm_set1_epi8(0x0F);
            let mut sumf = 0.0f32;
            for (b, x) in blocks.iter().zip(x_q8.iter()) {
                let w_u8 = unpack_nibbles_32_128(_mm_loadu_si128(b.qs.as_ptr() as *const __m128i), mask0f);
                let x_i8 = _mm256_load_si256(x.qs.as_ptr() as *const __m256i);
                let p16 = _mm256_maddubs_epi16(w_u8, x_i8);
                let s32 = hsum_epi32_256(_mm256_madd_epi16(p16, ones16)) - 8 * x.sum_q;
                sumf += f16_bytes_to_f32(b.d) * x.d * (s32 as f32);
            }
            sumf
        }
        GgmlType::Q4K => {
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(w_row);
            let mask0f = _mm256_set1_epi8(0x0F);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                let dw = f16_bytes_to_f32(b.d);
                let dmin = f16_bytes_to_f32(b.dmin);
                let xb = &x_q8[bi * 8..(bi + 1) * 8];
                for ib in 0..4 {
                    let (sc1, m1) = get_scale_min_k4(ib * 2, &b.scales);
                    let (sc2, m2) = get_scale_min_k4(ib * 2 + 1, &b.scales);
                    let raw = _mm256_loadu_si256(b.qs[ib * 32..].as_ptr() as *const __m256i);
                    let lo = _mm256_and_si256(raw, mask0f);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask0f);
                    let x0 = _mm256_load_si256(xb[ib * 2].qs.as_ptr() as *const __m256i);
                    let x1 = _mm256_load_si256(xb[ib * 2 + 1].qs.as_ptr() as *const __m256i);
                    let s0 = hsum_epi32_256(_mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), ones16));
                    let s1 = hsum_epi32_256(_mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), ones16));
                    sumf += dw
                        * ((sc1 as f32) * xb[ib * 2].d * (s0 as f32)
                            + (sc2 as f32) * xb[ib * 2 + 1].d * (s1 as f32))
                        - dmin * ((m1 as f32) * xb[ib * 2].s + (m2 as f32) * xb[ib * 2 + 1].s);
                }
            }
            sumf
        }
        GgmlType::Nvfp4 => {
            let blocks: &[BlockNvfp4] = bytemuck::cast_slice(w_row);
            let lut128 = _mm_loadu_si128(KVALUES_NVFP4_U8.as_ptr() as *const __m128i);
            let lut256 = _mm256_broadcastsi128_si256(lut128);
            let mask0f = _mm_set1_epi8(0x0F);
            let ones_u8 = _mm256_set1_epi8(1);
            let mut sumf = 0.0f32;
            for (bi, b) in blocks.iter().enumerate() {
                for half in 0..2 {
                    let raw16 = _mm_loadu_si128(b.qs[half * 16..].as_ptr() as *const __m128i);
                    let lo = _mm_and_si128(raw16, mask0f);
                    let hi = _mm_and_si128(_mm_srli_epi16(raw16, 4), mask0f);
                    let idx_sb0 = _mm_unpacklo_epi64(lo, hi);
                    let idx_sb1 = _mm_unpackhi_epi64(lo, hi);
                    let idx256 = _mm256_set_m128i(idx_sb1, idx_sb0);
                    let w_u8 = _mm256_shuffle_epi8(lut256, idx256);
                    let x = &x_q8[bi * 2 + half];
                    let x_i8 = _mm256_load_si256(x.qs.as_ptr() as *const __m256i);
                    let dots = _mm256_sub_epi32(
                        _mm256_madd_epi16(_mm256_maddubs_epi16(w_u8, x_i8), ones16),
                        _mm256_slli_epi32(_mm256_madd_epi16(_mm256_maddubs_epi16(ones_u8, x_i8), ones16), 7),
                    );
                    let s0 = hsum_epi32_128(_mm256_castsi256_si128(dots));
                    let s1 = hsum_epi32_128(_mm256_extracti128_si256::<1>(dots));
                    let d0 = ue4m3_to_f32(b.d[half * 2]);
                    let d1 = ue4m3_to_f32(b.d[half * 2 + 1]);
                    sumf += (d0 * (s0 as f32) + d1 * (s1 as f32)) * x.d;
                }
            }
            sumf
        }
        _ => vec_dot_scalar(ggml_type, w_row, x_q8, x_f32),
    }
}
