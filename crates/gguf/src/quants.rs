use crate::format::GgmlType;
use crate::iq_tables::{IQ2S_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS};
use bytemuck::{Pod, Zeroable};
use half::{bf16, f16};

pub const QK4_0: usize = 32;
pub const QK8_0: usize = 32;
pub const QK4_NL: usize = 32;
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
    assert!(std::mem::size_of::<BlockQ8_0>() == 34);
    assert!(std::mem::size_of::<BlockQ8_1>() == 36);
    assert!(std::mem::size_of::<BlockQ3K>() == 110);
    assert!(std::mem::size_of::<BlockQ4K>() == 144);
    assert!(std::mem::size_of::<BlockQ5K>() == 176);
    assert!(std::mem::size_of::<BlockQ6K>() == 210);
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
}
