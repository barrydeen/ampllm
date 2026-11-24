use crate::format::GgmlType;
use crate::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS,
};
use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};

pub const QK4_0: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK4_NL: usize = 32;
pub const QK2_0: usize = 64;
pub const QK_NVFP4: usize = 64;
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

/// Non-linear 4-bit codebook values from ggml-common.h.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Twice the E2M1 codebook values (`2 * e2m1`, paired with `ue4m3` scale which incorporates `0.5`).
pub const KVALUES_NVFP4: [i8; 16] = [
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12,
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4_0 {
    pub d: [u8; 2],
    pub qs: [u8; QK4_0 / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_0 {
    pub d: [u8; 2],
    pub qh: [u8; 4],
    pub qs: [u8; QK5_0 / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5_1 {
    pub d: [u8; 2],
    pub m: [u8; 2],
    pub qh: [u8; 4],
    pub qs: [u8; QK5_1 / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ2_0 {
    pub d: [u8; 2],
    pub qs: [u8; QK2_0 / 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_0 {
    pub d: [u8; 2],
    pub qs: [i8; QK8_0],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ8_1 {
    pub d: [u8; 2],
    pub s: [u8; 2],
    pub qs: [i8; QK8_0],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ4K {
    pub d: [u8; 2],
    pub dmin: [u8; 2],
    pub scales: [u8; K_SCALE_SIZE],
    pub qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ5K {
    pub d: [u8; 2],
    pub dmin: [u8; 2],
    pub scales: [u8; K_SCALE_SIZE],
    pub qh: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ6K {
    pub ql: [u8; QK_K / 2],
    pub qh: [u8; QK_K / 4],
    pub scales: [i8; QK_K / 16],
    pub d: [u8; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq4Nl {
    pub d: [u8; 2],
    pub qs: [u8; QK4_NL / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq4Xs {
    pub d: [u8; 2],
    pub scales_h: [u8; 2],
    pub scales_l: [u8; QK_K / 64],
    pub qs: [u8; QK_K / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockNvfp4 {
    pub d: [u8; 4],
    pub qs: [u8; QK_NVFP4 / 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockQ3K {
    pub hmask: [u8; QK_K / 8],
    pub qs: [u8; QK_K / 4],
    pub scales: [u8; 12],
    pub d: [u8; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq2Xxs {
    pub d: [u8; 2],
    pub qs: [u8; QK_K / 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq2Xs {
    pub d: [u8; 2],
    pub qs: [u8; QK_K / 4],
    pub scales: [u8; QK_K / 32],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq1S {
    pub d: [u8; 2],
    pub qs: [u8; QK_K / 8],
    pub qh: [u8; QK_K / 16],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq2S {
    pub d: [u8; 2],
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub scales: [u8; QK_K / 32],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq3Xxs {
    pub d: [u8; 2],
    pub qs: [u8; 3 * QK_K / 8],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BlockIq3S {
    pub d: [u8; 2],
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub signs: [u8; QK_K / 8],
    pub scales: [u8; QK_K / 64],
}

const _: () = {
    assert!(std::mem::size_of::<BlockQ4_0>() == 18);
    assert!(std::mem::size_of::<BlockQ5_0>() == 22);
    assert!(std::mem::size_of::<BlockQ5_1>() == 24);
    assert!(std::mem::size_of::<BlockQ8_0>() == 34);
    assert!(std::mem::size_of::<BlockQ8_1>() == 36);
    assert!(std::mem::size_of::<BlockQ2_0>() == 18);
    assert!(std::mem::size_of::<BlockQ3K>() == 110);
    assert!(std::mem::size_of::<BlockQ4K>() == 144);
    assert!(std::mem::size_of::<BlockQ5K>() == 176);
    assert!(std::mem::size_of::<BlockQ6K>() == 210);
    assert!(std::mem::size_of::<BlockIq2Xxs>() == 66);
    assert!(std::mem::size_of::<BlockIq2Xs>() == 74);
    assert!(std::mem::size_of::<BlockIq1S>() == 50);
    assert!(std::mem::size_of::<BlockIq2S>() == 82);
    assert!(std::mem::size_of::<BlockIq3Xxs>() == 98);
    assert!(std::mem::size_of::<BlockIq3S>() == 110);
    assert!(std::mem::size_of::<BlockIq4Nl>() == 18);
    assert!(std::mem::size_of::<BlockIq4Xs>() == 136);
    assert!(std::mem::size_of::<BlockNvfp4>() == 36);
};

#[inline]
pub fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0x7F {
        return 0.0;
    }
    let e = ((x >> 3) & 0x0F) as u32;
    let m = (x & 0x07) as u32;
    if e == 0 {
        (m as f32) * (1.0 / 1024.0)
    } else {
        f32::from_bits(((e + 119) << 23) | (m << 20))
    }
}

#[inline]
pub fn f16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    f16::from_le_bytes(bytes).to_f32()
}

#[inline]
pub fn get_scale_min_k4(j: usize, q: &[u8; K_SCALE_SIZE]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Reference GGML dequantization of one row of `k` elements to `f32`.
pub fn dequantize_row_to_f32(ggml_type: GgmlType, src: &[u8], dst: &mut [f32]) {
    let k = dst.len();
    match ggml_type {
        GgmlType::F32 => {
            let s: &[f32] = bytemuck::cast_slice(src);
            dst.copy_from_slice(&s[..k]);
        }
        GgmlType::F16 => {
            let s: &[[u8; 2]] = bytemuck::cast_slice(src);
            for (d, &b) in dst.iter_mut().zip(&s[..k]) {
                *d = f16_bytes_to_f32(b);
            }
        }
        GgmlType::Bf16 => {
            let s: &[[u8; 2]] = bytemuck::cast_slice(src);
            for (d, &b) in dst.iter_mut().zip(&s[..k]) {
                *d = bf16::from_le_bytes(b).to_f32();
            }
        }
        GgmlType::Q8_0 => {
            assert_eq!(k % QK8_0, 0);
            let blocks: &[BlockQ8_0] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK8_0)) {
                let d = f16_bytes_to_f32(b.d);
                for (o, &q) in out.iter_mut().zip(&b.qs) {
                    *o = d * (q as f32);
                }
            }
        }
        GgmlType::Q4_0 => {
            assert_eq!(k % QK4_0, 0);
            let blocks: &[BlockQ4_0] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK4_0)) {
                let d = f16_bytes_to_f32(b.d);
                for j in 0..(QK4_0 / 2) {
                    let x0 = ((b.qs[j] & 0x0F) as i32) - 8;
                    let x1 = ((b.qs[j] >> 4) as i32) - 8;
                    out[j] = d * (x0 as f32);
                    out[j + QK4_0 / 2] = d * (x1 as f32);
                }
            }
        }
        GgmlType::Q5_0 => {
            assert_eq!(k % QK5_0, 0);
            let blocks: &[BlockQ5_0] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK5_0)) {
                let d = f16_bytes_to_f32(b.d);
                let qh = u32::from_le_bytes(b.qh);
                for j in 0..(QK5_0 / 2) {
                    let x0 = (((b.qs[j] & 0x0F) | ((((qh >> j) & 1) as u8) << 4)) as i32) - 16;
                    let x1 = (((b.qs[j] >> 4) | ((((qh >> (j + 16)) & 1) as u8) << 4)) as i32) - 16;
                    out[j] = d * (x0 as f32);
                    out[j + QK5_0 / 2] = d * (x1 as f32);
                }
            }
        }
        GgmlType::Q5_1 => {
            assert_eq!(k % QK5_1, 0);
            let blocks: &[BlockQ5_1] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK5_1)) {
                let d = f16_bytes_to_f32(b.d);
                let m = f16_bytes_to_f32(b.m);
                let qh = u32::from_le_bytes(b.qh);
                for j in 0..(QK5_1 / 2) {
                    let x0 = ((b.qs[j] & 0x0F) as u32) | (((qh >> j) & 1) << 4);
                    let x1 = ((b.qs[j] >> 4) as u32) | (((qh >> (j + 16)) & 1) << 4);
                    out[j] = d * (x0 as f32) + m;
                    out[j + QK5_1 / 2] = d * (x1 as f32) + m;
                }
            }
        }
        GgmlType::Q2_0 => {
            assert_eq!(k % QK2_0, 0);
            let blocks: &[BlockQ2_0] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK2_0)) {
                let d = f16_bytes_to_f32(b.d);
                for j in 0..(QK2_0 / 4) {
                    let byte = b.qs[j];
                    for s in 0..4 {
                        let q = ((byte >> (2 * s)) & 3) as f32 - 1.0;
                        out[4 * j + s] = q * d;
                    }
                }
            }
        }
        GgmlType::Iq4Nl => {
            assert_eq!(k % QK4_NL, 0);
            let blocks: &[BlockIq4Nl] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK4_NL)) {
                let d = f16_bytes_to_f32(b.d);
                for j in 0..(QK4_NL / 2) {
                    out[j] = d * (KVALUES_IQ4NL[(b.qs[j] & 0x0F) as usize] as f32);
                    out[j + QK4_NL / 2] = d * (KVALUES_IQ4NL[(b.qs[j] >> 4) as usize] as f32);
                }
            }
        }
        GgmlType::Iq4Xs => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockIq4Xs] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                let scales_h = u16::from_le_bytes(b.scales_h);
                for ib in 0..(QK_K / 32) {
                    let ls = (((b.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) as i32)
                        | ((((scales_h >> (2 * ib)) & 3) as i32) << 4);
                    let dl = d * ((ls - 32) as f32);
                    let qs = &b.qs[ib * 16..(ib + 1) * 16];
                    let sub = &mut out[ib * 32..(ib + 1) * 32];
                    for j in 0..16 {
                        sub[j] = dl * (KVALUES_IQ4NL[(qs[j] & 0xF) as usize] as f32);
                        sub[j + 16] = dl * (KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32);
                    }
                }
            }
        }
        GgmlType::Q4K => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                let min = f16_bytes_to_f32(b.dmin);
                let mut is = 0;
                let mut q_off = 0;
                let mut y_off = 0;
                for _ in (0..QK_K).step_by(64) {
                    let (sc1, m1_u) = get_scale_min_k4(is, &b.scales);
                    let (sc2, m2_u) = get_scale_min_k4(is + 1, &b.scales);
                    let d1 = d * (sc1 as f32);
                    let m1 = min * (m1_u as f32);
                    let d2 = d * (sc2 as f32);
                    let m2 = min * (m2_u as f32);
                    for l in 0..32 {
                        out[y_off + l] = d1 * ((b.qs[q_off + l] & 0xF) as f32) - m1;
                    }
                    for l in 0..32 {
                        out[y_off + 32 + l] = d2 * ((b.qs[q_off + l] >> 4) as f32) - m2;
                    }
                    y_off += 64;
                    q_off += 32;
                    is += 2;
                }
            }
        }
        GgmlType::Q5K => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockQ5K] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                let min = f16_bytes_to_f32(b.dmin);
                let mut is = 0;
                let mut q_off = 0;
                let mut y_off = 0;
                let mut u1: u8 = 1;
                let mut u2: u8 = 2;
                for _ in (0..QK_K).step_by(64) {
                    let (sc1, m1_u) = get_scale_min_k4(is, &b.scales);
                    let (sc2, m2_u) = get_scale_min_k4(is + 1, &b.scales);
                    let d1 = d * (sc1 as f32);
                    let m1 = min * (m1_u as f32);
                    let d2 = d * (sc2 as f32);
                    let m2 = min * (m2_u as f32);
                    for l in 0..32 {
                        let hi = if b.qh[l] & u1 != 0 { 16 } else { 0 };
                        out[y_off + l] = d1 * (((b.qs[q_off + l] & 0xF) + hi) as f32) - m1;
                    }
                    for l in 0..32 {
                        let hi = if b.qh[l] & u2 != 0 { 16 } else { 0 };
                        out[y_off + 32 + l] = d2 * (((b.qs[q_off + l] >> 4) + hi) as f32) - m2;
                    }
                    y_off += 64;
                    q_off += 32;
                    is += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            }
        }
        GgmlType::Q6K => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockQ6K] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                for half in 0..2 {
                    let ql = &b.ql[half * 64..(half + 1) * 64];
                    let qh = &b.qh[half * 32..(half + 1) * 32];
                    let sc = &b.scales[half * 8..(half + 1) * 8];
                    let y = &mut out[half * 128..(half + 1) * 128];
                    for l in 0..32 {
                        let is = l / 16;
                        let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                        let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                        y[l] = d * (sc[is] as f32) * (q1 as f32);
                        y[l + 32] = d * (sc[is + 2] as f32) * (q2 as f32);
                        y[l + 64] = d * (sc[is + 4] as f32) * (q3 as f32);
                        y[l + 96] = d * (sc[is + 6] as f32) * (q4 as f32);
                    }
                }
            }
        }
        GgmlType::Q3K => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockQ3K] = bytemuck::cast_slice(src);
            const KMASK1: u32 = 0x0303_0303;
            const KMASK2: u32 = 0x0F0F_0F0F;
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d_all = f16_bytes_to_f32(b.d);
                let mut aux = [0u32; 4];
                aux[0] = u32::from_le_bytes(b.scales[0..4].try_into().unwrap());
                aux[1] = u32::from_le_bytes(b.scales[4..8].try_into().unwrap());
                aux[2] = u32::from_le_bytes(b.scales[8..12].try_into().unwrap());
                let tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
                aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
                aux[0] = (aux[0] & KMASK2) | (((tmp >> 0) & KMASK1) << 4);
                aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
                let mut scales = [0i8; 16];
                for i in 0..4 {
                    let bytes = aux[i].to_le_bytes();
                    scales[i * 4..i * 4 + 4].copy_from_slice(&bytes.map(|v| v as i8));
                }
                let mut m: u8 = 1;
                let mut is = 0usize;
                let mut q_off = 0usize;
                let mut y_off = 0usize;
                for _n in (0..QK_K).step_by(128) {
                    let mut shift = 0;
                    for _j in 0..4 {
                        let dl0 = d_all * ((scales[is] as i32 - 32) as f32);
                        is += 1;
                        for l in 0..16 {
                            let qv = ((b.qs[q_off + l] >> shift) & 3) as i32
                                - if b.hmask[l] & m != 0 { 0 } else { 4 };
                            out[y_off + l] = dl0 * (qv as f32);
                        }
                        y_off += 16;

                        let dl1 = d_all * ((scales[is] as i32 - 32) as f32);
                        is += 1;
                        for l in 0..16 {
                            let qv = ((b.qs[q_off + l + 16] >> shift) & 3) as i32
                                - if b.hmask[l + 16] & m != 0 { 0 } else { 4 };
                            out[y_off + l] = dl1 * (qv as f32);
                        }
                        y_off += 16;
                        shift += 2;
                        m <<= 1;
                    }
                    q_off += 32;
                }
            }
        }
        GgmlType::Iq2Xxs => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockIq2Xxs] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                let mut y_off = 0usize;
                for ib32 in 0..(QK_K / 32) {
                    let aux0 = u32::from_le_bytes(b.qs[8 * ib32..8 * ib32 + 4].try_into().unwrap());
                    let aux1 = u32::from_le_bytes(b.qs[8 * ib32 + 4..8 * ib32 + 8].try_into().unwrap());
                    let db = d * (0.5 + ((aux1 >> 28) as f32)) * 0.25;
                    for l in 0..4 {
                        let grid_idx = ((aux0 >> (8 * l)) & 0xFF) as usize;
                        let grid = IQ2XXS_GRID[grid_idx].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                        for j in 0..8 {
                            let sign = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                            out[y_off + j] = db * (grid[j] as f32) * sign;
                        }
                        y_off += 8;
                    }
                }
            }
        }
        GgmlType::Iq2Xs => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockIq2Xs] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                let mut y_off = 0usize;
                for ib32 in 0..(QK_K / 32) {
                    let db0 = d * (0.5 + ((b.scales[ib32] & 0xF) as f32)) * 0.25;
                    let db1 = d * (0.5 + ((b.scales[ib32] >> 4) as f32)) * 0.25;
                    for l in 0..4 {
                        let q16 = u16::from_le_bytes([
                            b.qs[8 * ib32 + 2 * l],
                            b.qs[8 * ib32 + 2 * l + 1],
                        ]);
                        let dl = if l < 2 { db0 } else { db1 };
                        let grid = IQ2XS_GRID[(q16 & 511) as usize].to_le_bytes();
                        let signs = KSIGNS_IQ2XS[(q16 >> 9) as usize];
                        for j in 0..8 {
                            let sign = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                            out[y_off + j] = dl * (grid[j] as f32) * sign;
                        }
                        y_off += 8;
                    }
                }
            }
        }
        GgmlType::Iq1S => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockIq1S] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                let mut y_off = 0usize;
                for ib in 0..(QK_K / 32) {
                    let qh = u16::from_le_bytes([b.qh[2 * ib], b.qh[2 * ib + 1]]);
                    let dl = d * ((2 * ((qh >> 12) & 7) + 1) as f32);
                    let delta = if (qh & 0x8000) != 0 { -0.125f32 } else { 0.125f32 };
                    for l in 0..4 {
                        let idx = (b.qs[4 * ib + l] as usize)
                            | ((((qh >> (3 * l)) & 7) as usize) << 8);
                        let grid = IQ1S_GRID[idx].to_le_bytes();
                        for j in 0..8 {
                            out[y_off + j] = dl * ((grid[j] as i8 as f32) + delta);
                        }
                        y_off += 8;
                    }
                }
            }
        }
        GgmlType::Iq2S => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockIq2S] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                let mut qs_off = 0usize;
                let mut signs_off = QK_K / 8;
                let mut y_off = 0usize;
                for ib32 in 0..(QK_K / 32) {
                    let db0 = d * (0.5 + ((b.scales[ib32] & 0xF) as f32)) * 0.25;
                    let db1 = d * (0.5 + ((b.scales[ib32] >> 4) as f32)) * 0.25;
                    for l in 0..4 {
                        let dl = if l < 2 { db0 } else { db1 };
                        let idx = (b.qs[qs_off + l] as usize)
                            | (((b.qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                        let grid = IQ2S_GRID[idx].to_le_bytes();
                        let sbyte = b.qs[signs_off + l];
                        for j in 0..8 {
                            let sign = if sbyte & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                            out[y_off + j] = dl * (grid[j] as f32) * sign;
                        }
                        y_off += 8;
                    }
                    qs_off += 4;
                    signs_off += 4;
                }
            }
        }
        GgmlType::Iq3Xxs => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockIq3Xxs] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                let mut qs_off = 0usize;
                let ss_off = QK_K / 4;
                let mut y_off = 0usize;
                for ib32 in 0..(QK_K / 32) {
                    let aux32 = u32::from_le_bytes(
                        b.qs[ss_off + 4 * ib32..ss_off + 4 * ib32 + 4].try_into().unwrap(),
                    );
                    let db = d * (0.5 + ((aux32 >> 28) as f32)) * 0.5;
                    for l in 0..4 {
                        let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                        let grid1 = IQ3XXS_GRID[b.qs[qs_off + 2 * l] as usize].to_le_bytes();
                        let grid2 = IQ3XXS_GRID[b.qs[qs_off + 2 * l + 1] as usize].to_le_bytes();
                        for j in 0..4 {
                            let s0 = if signs & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                            let s1 = if signs & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                            out[y_off + j] = db * (grid1[j] as f32) * s0;
                            out[y_off + j + 4] = db * (grid2[j] as f32) * s1;
                        }
                        y_off += 8;
                    }
                    qs_off += 8;
                }
            }
        }
        GgmlType::Iq3S => {
            assert_eq!(k % QK_K, 0);
            let blocks: &[BlockIq3S] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_K)) {
                let d = f16_bytes_to_f32(b.d);
                let mut qs_off = 0usize;
                let mut qh_off = 0usize;
                let mut signs_off = 0usize;
                let mut y_off = 0usize;
                for ib32 in (0..(QK_K / 32)).step_by(2) {
                    let sc_byte = b.scales[ib32 / 2];
                    let db1 = d * ((1 + 2 * ((sc_byte & 0xF) as i32)) as f32);
                    let db2 = d * ((1 + 2 * ((sc_byte >> 4) as i32)) as f32);
                    for l in 0..4 {
                        let i1 = (b.qs[qs_off + 2 * l] as usize)
                            | (((b.qh[qh_off] as usize) << (8 - 2 * l)) & 256);
                        let i2 = (b.qs[qs_off + 2 * l + 1] as usize)
                            | (((b.qh[qh_off] as usize) << (7 - 2 * l)) & 256);
                        let grid1 = IQ3S_GRID[i1].to_le_bytes();
                        let grid2 = IQ3S_GRID[i2].to_le_bytes();
                        let sbyte = b.signs[signs_off + l];
                        for j in 0..4 {
                            let s0 = if sbyte & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                            let s1 = if sbyte & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                            out[y_off + j] = db1 * (grid1[j] as f32) * s0;
                            out[y_off + j + 4] = db1 * (grid2[j] as f32) * s1;
                        }
                        y_off += 8;
                    }
                    qs_off += 8;
                    signs_off += 4;
                    for l in 0..4 {
                        let i1 = (b.qs[qs_off + 2 * l] as usize)
                            | (((b.qh[qh_off + 1] as usize) << (8 - 2 * l)) & 256);
                        let i2 = (b.qs[qs_off + 2 * l + 1] as usize)
                            | (((b.qh[qh_off + 1] as usize) << (7 - 2 * l)) & 256);
                        let grid1 = IQ3S_GRID[i1].to_le_bytes();
                        let grid2 = IQ3S_GRID[i2].to_le_bytes();
                        let sbyte = b.signs[signs_off + l];
                        for j in 0..4 {
                            let s0 = if sbyte & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
                            let s1 = if sbyte & KMASK_IQ2XS[j + 4] != 0 { -1.0f32 } else { 1.0f32 };
                            out[y_off + j] = db2 * (grid1[j] as f32) * s0;
                            out[y_off + j + 4] = db2 * (grid2[j] as f32) * s1;
                        }
                        y_off += 8;
                    }
                    qh_off += 2;
                    qs_off += 8;
                    signs_off += 4;
                }
            }
        }
        GgmlType::Nvfp4 => {
            assert_eq!(k % QK_NVFP4, 0);
            let blocks: &[BlockNvfp4] = bytemuck::cast_slice(src);
            for (b, out) in blocks.iter().zip(dst.chunks_exact_mut(QK_NVFP4)) {
                for sb in 0..4 {
                    let d = ue4m3_to_f32(b.d[sb]);
                    let qs = &b.qs[sb * 8..(sb + 1) * 8];
                    let sub = &mut out[sb * 16..(sb + 1) * 16];
                    for j in 0..8 {
                        sub[j] = d * (KVALUES_NVFP4[(qs[j] & 0x0F) as usize] as f32);
                        sub[j + 8] = d * (KVALUES_NVFP4[(qs[j] >> 4) as usize] as f32);
                    }
                }
            }
        }
        other => panic!("dequantize_row_to_f32 unsupported type: {other:?}"),
    }
}

/// Reference dot product between a quantized weight row and an f32 activation vector.
pub fn vec_dot_q_f32_ref(ggml_type: GgmlType, weight_row: &[u8], act: &[f32]) -> f32 {
    let mut tmp = vec![0.0f32; act.len()];
    dequantize_row_to_f32(ggml_type, weight_row, &mut tmp);
    let mut acc = 0.0f64;
    for (&w, &x) in tmp.iter().zip(act.iter()) {
        acc += (w as f64) * (x as f64);
    }
    acc as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_glm5_iq_and_q3k_dequant_nonzero() {
        // Verify non-NaN, deterministic dequant on 256-element blocks for Q3K, IQ2S, IQ3XXS, IQ3S
        let one_f16 = f16::from_f32(1.0).to_le_bytes();

        let mut bq3 = BlockQ3K {
            hmask: [0xAA; 32],
            qs: [0x5A; 64],
            scales: [0x33; 12],
            d: one_f16,
        };
        bq3.d = one_f16;
        let mut out = vec![0.0f32; QK_K];
        dequantize_row_to_f32(GgmlType::Q3K, bytemuck::bytes_of(&bq3), &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|&v| v != 0.0));

        let biq2s = BlockIq2S {
            d: one_f16,
            qs: [0x17; 64],
            qh: [0x55; 8],
            scales: [0x42; 8],
        };
        dequantize_row_to_f32(GgmlType::Iq2S, bytemuck::bytes_of(&biq2s), &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|&v| v != 0.0));

        let biq3xxs = BlockIq3Xxs {
            d: one_f16,
            qs: [0x35; 96],
        };
        dequantize_row_to_f32(GgmlType::Iq3Xxs, bytemuck::bytes_of(&biq3xxs), &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|&v| v != 0.0));

        let biq3s = BlockIq3S {
            d: one_f16,
            qs: [0x29; 64],
            qh: [0xF0; 8],
            signs: [0x3C; 32],
            scales: [0x21; 4],
        };
        dequantize_row_to_f32(GgmlType::Iq3S, bytemuck::bytes_of(&biq3s), &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        assert!(out.iter().any(|&v| v != 0.0));
    }

    #[test]
    fn test_qwen4_q5_1_and_q2_0_dequant_and_dot() {
        let mut qh_bits = 0u32;
        qh_bits |= 1 << 0; // elem 0 gets +16
        qh_bits |= 1 << 17; // elem 17 gets +16
        let mut qs5 = [0u8; 16];
        qs5[0] = 0x23; // elem 0 low=3 -> 19, elem 16 high=2 -> 2
        qs5[1] = 0x57; // elem 1 low=7 -> 7, elem 17 high=5 -> 21
        let bq5 = BlockQ5_1 {
            d: f16::from_f32(0.5).to_le_bytes(),
            m: f16::from_f32(-4.0).to_le_bytes(),
            qh: qh_bits.to_le_bytes(),
            qs: qs5,
        };
        let mut out5 = vec![0.0f32; QK5_1];
        dequantize_row_to_f32(GgmlType::Q5_1, bytemuck::bytes_of(&bq5), &mut out5);
        assert_eq!(out5[0], 0.5 * 19.0 - 4.0);
        assert_eq!(out5[1], 0.5 * 7.0 - 4.0);
        assert_eq!(out5[16], 0.5 * 2.0 - 4.0);
        assert_eq!(out5[17], 0.5 * 21.0 - 4.0);
        let ones5 = vec![1.0f32; QK5_1];
        let expected_dot5: f32 = out5.iter().sum();
        assert!((vec_dot_q_f32_ref(GgmlType::Q5_1, bytemuck::bytes_of(&bq5), &ones5) - expected_dot5).abs() < 1e-4);

        // Q5_0: same bit packing as Q5_1, signed (x - 16) * d
        let bq5_0 = BlockQ5_0 {
            d: f16::from_f32(0.5).to_le_bytes(),
            qh: qh_bits.to_le_bytes(),
            qs: qs5,
        };
        let mut out5_0 = vec![0.0f32; QK5_0];
        dequantize_row_to_f32(GgmlType::Q5_0, bytemuck::bytes_of(&bq5_0), &mut out5_0);
        assert_eq!(out5_0[0], 0.5 * (19.0 - 16.0));
        assert_eq!(out5_0[1], 0.5 * (7.0 - 16.0));
        assert_eq!(out5_0[16], 0.5 * (2.0 - 16.0));
        assert_eq!(out5_0[17], 0.5 * (21.0 - 16.0));
        let expected_dot5_0: f32 = out5_0.iter().sum();
        assert!((vec_dot_q_f32_ref(GgmlType::Q5_0, bytemuck::bytes_of(&bq5_0), &ones5) - expected_dot5_0).abs() < 1e-4);

        // Q2_0: codes {0,1,2,3} -> {-1, 0, +1, +2} * d
        // byte 0xE4 = 0b11_10_01_00 -> codes [0, 1, 2, 3] -> values [-1, 0, +1, +2] * d
        let mut qs2 = [0x55u8; 16]; // code=1 -> 0
        qs2[0] = 0xE4;
        let bq2 = BlockQ2_0 {
            d: f16::from_f32(0.25).to_le_bytes(),
            qs: qs2,
        };
        let mut out2 = vec![0.0f32; QK2_0];
        dequantize_row_to_f32(GgmlType::Q2_0, bytemuck::bytes_of(&bq2), &mut out2);
        assert_eq!(&out2[0..5], &[-0.25, 0.0, 0.25, 0.5, 0.0]);
        let act2: Vec<f32> = (0..QK2_0).map(|i| (i as f32) * 0.1 + 1.0).collect();
        let dot2 = vec_dot_q_f32_ref(GgmlType::Q2_0, bytemuck::bytes_of(&bq2), &act2);
        let exp2 = -0.25 * 1.0 + 0.0 * 1.1 + 0.25 * 1.2 + 0.5 * 1.3;
        assert!((dot2 - exp2).abs() < 1e-5, "dot2={dot2} exp2={exp2}");

        // IQ2_XXS / IQ2_XS / IQ1_S roundtrip & dot sanity checks
        let one_f16 = f16::from_f32(1.0).to_le_bytes();
        let mut biq2xxs = BlockIq2Xxs {
            d: one_f16,
            qs: [0u8; QK_K / 4],
        };
        biq2xxs.qs[0] = 3;
        biq2xxs.qs[4] = 1; // sign idx 1 -> KSIGNS_IQ2XS[1] = 129 (bits 0, 7 negative)
        biq2xxs.qs[7] = 0x20; // nibble 2 -> (0.5 + 2) * 0.25 = 0.625
        let mut out_xxs = vec![0.0f32; QK_K];
        dequantize_row_to_f32(GgmlType::Iq2Xxs, bytemuck::bytes_of(&biq2xxs), &mut out_xxs);
        let g3 = IQ2XXS_GRID[3].to_le_bytes();
        assert!((out_xxs[0] - (-0.625 * g3[0] as f32)).abs() < 1e-5);
        assert!((out_xxs[1] - (0.625 * g3[1] as f32)).abs() < 1e-5);
        assert!((out_xxs[7] - (-0.625 * g3[7] as f32)).abs() < 1e-5);

        let mut biq2xs = BlockIq2Xs {
            d: one_f16,
            qs: [0u8; QK_K / 4],
            scales: [0x31u8; QK_K / 32],
        };
        let q0: u16 = 300 | (2u16 << 9);
        biq2xs.qs[0..2].copy_from_slice(&q0.to_le_bytes());
        let mut out_xs = vec![0.0f32; QK_K];
        dequantize_row_to_f32(GgmlType::Iq2Xs, bytemuck::bytes_of(&biq2xs), &mut out_xs);
        let gx = IQ2XS_GRID[300].to_le_bytes();
        let s2 = KSIGNS_IQ2XS[2]; // 130 = bit 1 | bit 7
        for j in 0..8 {
            let sg = if s2 & KMASK_IQ2XS[j] != 0 { -1.0f32 } else { 1.0f32 };
            assert!((out_xs[j] - (0.375 * (gx[j] as f32) * sg)).abs() < 1e-5);
        }

        let mut biq1s = BlockIq1S {
            d: one_f16,
            qs: [0u8; QK_K / 8],
            qh: [0u8; QK_K / 16],
        };
        biq1s.qs[0] = 42;
        let qh0: u16 = 0x8000 | (2u16 << 12) | 3; // delta = -0.125, dl = 5.0, high 3 bits of idx0 = 3
        biq1s.qh[0..2].copy_from_slice(&qh0.to_le_bytes());
        let mut out_1s = vec![0.0f32; QK_K];
        dequantize_row_to_f32(GgmlType::Iq1S, bytemuck::bytes_of(&biq1s), &mut out_1s);
        let g1 = IQ1S_GRID[(3usize << 8) | 42].to_le_bytes();
        for j in 0..8 {
            let exp = 5.0 * ((g1[j] as i8 as f32) - 0.125);
            assert!((out_1s[j] - exp).abs() < 1e-5);
        }
    }
}
