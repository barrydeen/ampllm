use ampllm_cpu_kernels::mm3_cpu::{
    ampllm_cpu_mm3_quantize_q8_k, run_mm3_cpu_moe_grouped, run_mm3_cpu_moe_m1, Mm3CpuExpertJob,
    Mm3CpuGroupedExpert, Mm3CpuScratch,
};
use ampllm_cpu_kernels::pool::CpuThreadPool;
use ampllm_cuda_kernels::ffi::*;
use ampllm_cuda_kernels::{mm3, CudaStream, DeviceBuffer, PinnedMappedBuffer};
use ampllm_gguf::{dequantize_row_to_f32, GgmlType};
use half::{bf16, f16};
use std::sync::Mutex;

static GPU_LOCK: Mutex<()> = Mutex::new(());

extern "C" {
    fn erff(x: f32) -> f32;
}

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn next_u8(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }
    fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        let u = (self.next_u32() & 0xFFFF) as f32 / 65535.0;
        lo + u * (hi - lo)
    }
    fn next_f32(&mut self) -> f32 {
        self.next_f32_range(-1.5, 1.5)
    }
}

fn rel_err(got: &[f32], exp: &[f32]) -> f32 {
    assert_eq!(got.len(), exp.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &e) in got.iter().zip(exp.iter()) {
        assert!(g.is_finite(), "non-finite got value: {g}");
        assert!(e.is_finite(), "non-finite exp value: {e}");
        num += ((g - e) as f64).abs();
        den += (e as f64).abs();
    }
    (num / den.max(1e-12)) as f32
}

fn max_abs_err(got: &[f32], exp: &[f32]) -> f32 {
    assert_eq!(got.len(), exp.len());
    got.iter()
        .zip(exp.iter())
        .map(|(&g, &e)| {
            assert!(g.is_finite() && e.is_finite());
            (g - e).abs()
        })
        .fold(0.0f32, f32::max)
}

fn to_dev_f32(data: &[f32]) -> DeviceBuffer {
    let buf = DeviceBuffer::alloc(0, data.len() * 4).unwrap();
    buf.copy_from_host(bytemuck::cast_slice(data)).unwrap();
    buf
}

fn to_dev_u8(data: &[u8]) -> DeviceBuffer {
    let buf = DeviceBuffer::alloc(0, data.len()).unwrap();
    buf.copy_from_host(data).unwrap();
    buf
}

fn to_dev_u16(data: &[u16]) -> DeviceBuffer {
    let buf = DeviceBuffer::alloc(0, data.len() * 2).unwrap();
    buf.copy_from_host(bytemuck::cast_slice(data)).unwrap();
    buf
}

fn to_dev_i32(data: &[i32]) -> DeviceBuffer {
    let buf = DeviceBuffer::alloc(0, data.len() * 4).unwrap();
    buf.copy_from_host(bytemuck::cast_slice(data)).unwrap();
    buf
}

fn to_dev_u64(data: &[u64]) -> DeviceBuffer {
    let buf = DeviceBuffer::alloc(0, data.len() * 8).unwrap();
    buf.copy_from_host(bytemuck::cast_slice(data)).unwrap();
    buf
}

fn read_dev_f32(buf: &DeviceBuffer, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    buf.copy_to_host(bytemuck::cast_slice_mut(&mut out)).unwrap();
    out
}

fn read_dev_i32(buf: &DeviceBuffer, len: usize) -> Vec<i32> {
    let mut out = vec![0i32; len];
    buf.copy_to_host(bytemuck::cast_slice_mut(&mut out)).unwrap();
    out
}

fn read_dev_u16(buf: &DeviceBuffer, len: usize) -> Vec<u16> {
    let mut out = vec![0u16; len];
    buf.copy_to_host(bytemuck::cast_slice_mut(&mut out))
        .unwrap();
    out
}

fn read_dev_u8(buf: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    buf.copy_to_host(&mut out).unwrap();
    out
}

fn fill_mm3_quant_row(t: GgmlType, k: usize, rng: &mut Rng) -> Vec<u8> {
    assert_eq!(k % t.block_size(), 0);
    let nb = k / t.block_size();
    let bsz = t.type_size();
    let mut bytes = vec![0u8; nb * bsz];
    for b in 0..nb {
        let blk = &mut bytes[b * bsz..(b + 1) * bsz];
        let d_hi = match t {
            GgmlType::Iq1S => 0.012,
            GgmlType::Iq2Xxs | GgmlType::Iq2Xs | GgmlType::Iq2S => 0.025,
            GgmlType::Iq3Xxs => 0.02,
            _ => unreachable!("unexpected quant type {t:?}"),
        };
        let d = f16::from_f32(rng.next_f32_range(0.005, d_hi)).to_le_bytes();
        blk[0..2].copy_from_slice(&d);
        for x in &mut blk[2..bsz] {
            *x = rng.next_u8();
        }
    }
    bytes
}

fn fill_mm3_quant_matrix(t: GgmlType, n: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, usize) {
    let rb = t.row_bytes(k);
    let mut out = Vec::with_capacity(n * rb);
    for _ in 0..n {
        out.extend_from_slice(&fill_mm3_quant_row(t, k, rng));
    }
    (out, rb)
}

fn dequant_matrix_to_f32(t: GgmlType, raw: &[u8], n: usize, k: usize) -> Vec<f32> {
    let rb = t.row_bytes(k);
    let mut out = vec![0.0f32; n * k];
    for r in 0..n {
        dequantize_row_to_f32(t, &raw[r * rb..(r + 1) * rb], &mut out[r * k..(r + 1) * k]);
    }
    out
}

fn dequant_q8_1_blocks(bytes: &[u8], k: usize) -> Vec<f32> {
    assert_eq!(k % 32, 0);
    let nb = k / 32;
    assert_eq!(bytes.len(), nb * 36);
    let mut out = vec![0.0f32; k];
    for b in 0..nb {
        let blk = &bytes[b * 36..(b + 1) * 36];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let s = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let mut qsum = 0i32;
        for i in 0..32 {
            let q = blk[4 + i] as i8;
            qsum += q as i32;
            out[b * 32 + i] = d * (q as f32);
        }
        let s_exp = d * (qsum as f32);
        assert!(
            (s - s_exp).abs() <= 5e-2 * s_exp.abs().max(1.0),
            "q8_1 s mismatch at block {b}: got {s} exp {s_exp}"
        );
    }
    out
}

fn dequant_q8_k_blocks(bytes: &[u8], k: usize) -> Vec<f32> {
    assert_eq!(k % 256, 0);
    let nb = k / 256;
    assert_eq!(bytes.len(), nb * mm3::Q8K_BLOCK_BYTES);
    let mut out = vec![0.0f32; k];
    for b in 0..nb {
        let blk = &bytes[b * mm3::Q8K_BLOCK_BYTES..(b + 1) * mm3::Q8K_BLOCK_BYTES];
        let d = f32::from_le_bytes([blk[0], blk[1], blk[2], blk[3]]);
        for g in 0..16 {
            let mut sum16 = 0i32;
            for i in 0..16 {
                let idx = g * 16 + i;
                let q = blk[4 + idx] as i8;
                sum16 += q as i32;
                out[b * 256 + idx] = d * (q as f32);
            }
            let bsum = i16::from_le_bytes([blk[260 + 2 * g], blk[260 + 2 * g + 1]]);
            assert_eq!(
                bsum as i32, sum16,
                "q8_k bsum mismatch block={b} g={g}: got {bsum}, exp {sum16}"
            );
        }
    }
    out
}

fn swoai_cpu(g: f32, u: f32, alpha: f32, limit: f32) -> f32 {
    let gc = if limit > 0.0 { g.min(limit) } else { g };
    let uc = if limit > 0.0 {
        u.clamp(-limit, limit)
    } else {
        u
    };
    (gc / (1.0f32 + (-alpha * gc).exp())) * (uc + 1.0f32)
}

fn had128_cpu(x: &[f32]) -> Vec<f32> {
    assert_eq!(x.len(), 128);
    let mut a = x.to_vec();
    let mut len = 1usize;
    while len < 128 {
        for i in (0..128).step_by(len * 2) {
            for j in 0..len {
                let u = a[i + j];
                let v = a[i + j + len];
                a[i + j] = u + v;
                a[i + j + len] = u - v;
            }
        }
        len *= 2;
    }
    for v in &mut a {
        *v *= 0.088388347648f32;
    }
    a
}

fn norm_rope_cpu_head(
    x: &[f32],
    w_opt: Option<&[f32]>,
    pos: i32,
    n_rot: usize,
    inv_freq: &[f32],
    eps: f32,
) -> Vec<f32> {
    let hd = x.len();
    let ss: f32 = x.iter().map(|&v| v * v).sum();
    let inv = 1.0f32 / (ss / (hd as f32) + eps).sqrt();
    let mut xn = vec![0.0f32; hd];
    for d in 0..hd {
        let wt = w_opt.map_or(1.0f32, |w| w[d]);
        xn[d] = x[d] * inv * wt;
    }
    let half = n_rot / 2;
    let mut out = xn.clone();
    for d in 0..half {
        let ang = (pos as f32) * inv_freq[d];
        let (sn, cs) = ang.sin_cos();
        let x1 = xn[d];
        let x2 = xn[d + half];
        out[d] = x1 * cs - x2 * sn;
        out[d + half] = x2 * cs + x1 * sn;
    }
    out
}

fn decode_fp8_e4m3_safe(b: u8) -> f32 {
    let sign = if (b & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let man = (b & 0x07) as i32;
    if exp == 15 && man == 7 {
        0.0f32
    } else if exp == 0 {
        sign * ((man as f32) / 8.0f32) * 2.0f32.powi(-6)
    } else {
        sign * (1.0f32 + (man as f32) / 8.0f32) * 2.0f32.powi(exp - 7)
    }
}

fn decode_ue8m0(e: u8) -> f32 {
    if e == 0 {
        2.0f32.powi(-127)
    } else {
        f32::from_bits((e as u32) << 23)
    }
}

// ============================================================================
// Test 1: IQ2_XXS / IQ2_XS / IQ1_S GEMV, MMQ MMA, Q8_K, SwiGLU-OAI & GPU MoE
// ============================================================================

#[test]
fn test_mm3_quants_iq2xxs_iq2xs_gemv_mmq_moe() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x3331_0001_55AA_7788);

    // 1) GEMV (M=1, M=4) & MMQ MMA Tiled/Grouped across all 5 MM3 quant types
    let k = 512usize;
    let n = 128usize;
    let m_max = 40usize;
    let x: Vec<f32> = (0..m_max * k).map(|_| rng.next_f32()).collect();
    let d_x = to_dev_f32(&x);
    let nb_q8_1 = k / 32;
    let d_xq8 = DeviceBuffer::alloc(0, m_max * nb_q8_1 * 36).unwrap();
    unsafe {
        launch_quantize_q8_1(d_x.ptr as *const f32, d_xq8.ptr, m_max as i32, k as i32, stream.raw);
    }
    stream.sync().unwrap();
    let x_q8_host = read_dev_u8(&d_xq8, m_max * nb_q8_1 * 36);
    let mut x_q8_deq = Vec::with_capacity(m_max * k);
    for mi in 0..m_max {
        x_q8_deq.extend_from_slice(&dequant_q8_1_blocks(
            &x_q8_host[mi * nb_q8_1 * 36..(mi + 1) * nb_q8_1 * 36],
            k,
        ));
    }

    for &ty in &[
        GgmlType::Iq2Xxs,
        GgmlType::Iq2Xs,
        GgmlType::Iq1S,
        GgmlType::Iq2S,
        GgmlType::Iq3Xxs,
    ] {
        let (w_bytes, rb) = fill_mm3_quant_matrix(ty, n, k, &mut rng);
        let d_w = to_dev_u8(&w_bytes);
        let w_f32 = dequant_matrix_to_f32(ty, &w_bytes, n, k);

        let mut ref_fp32 = vec![0.0f32; m_max * n];
        let mut ref_q8 = vec![0.0f32; m_max * n];
        for mi in 0..m_max {
            let xr = &x[mi * k..(mi + 1) * k];
            let xq = &x_q8_deq[mi * k..(mi + 1) * k];
            for ni in 0..n {
                let wr = &w_f32[ni * k..(ni + 1) * k];
                ref_fp32[mi * n + ni] = wr.iter().zip(xr.iter()).map(|(&a, &b)| a * b).sum();
                ref_q8[mi * n + ni] = wr.iter().zip(xq.iter()).map(|(&a, &b)| a * b).sum();
            }
        }

        // launch_linear_q8_1 at M=1 and M=4
        for &m in &[1usize, 4usize] {
            let d_y = DeviceBuffer::alloc(0, m * n * 4).unwrap();
            unsafe {
                launch_linear_q8_1(
                    ty as i32,
                    d_w.ptr,
                    d_xq8.ptr,
                    d_x.ptr as *const f32,
                    d_y.ptr as *mut f32,
                    n as i32,
                    k as i32,
                    m as i32,
                    stream.raw,
                );
            }
            stream.sync().unwrap();
            let got = read_dev_f32(&d_y, m * n);
            let e_fp32 = rel_err(&got, &ref_fp32[..m * n]);
            let e_q8 = rel_err(&got, &ref_q8[..m * n]);
            println!("linear_q8_1 {ty:?} M={m}: rel_fp32={e_fp32:.6} rel_q8={e_q8:.6}");
            assert!(e_fp32 < 2.5e-2, "{ty:?} M={m} rel_fp32={e_fp32}");
            assert!(e_q8 < 1e-3, "{ty:?} M={m} rel_q8={e_q8}");
        }

        // launch_gemm_mmq_mma_tiled across tile_mode = 0, 2, 3
        let d_y = DeviceBuffer::alloc(0, m_max * n * 4).unwrap();
        for &tile_mode in &[0i32, 2, 3] {
            unsafe {
                launch_gemm_mmq_mma_tiled(
                    ty as i32,
                    d_xq8.ptr,
                    d_w.ptr,
                    rb,
                    d_y.ptr as *mut f32,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    m_max as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    0.0,
                    tile_mode,
                    stream.raw,
                );
            }
            stream.sync().unwrap();
            let got = read_dev_f32(&d_y, m_max * n);
            let e_fp32 = rel_err(&got, &ref_fp32);
            let e_q8 = rel_err(&got, &ref_q8);
            println!("mmq_mma_tiled {ty:?} tile={tile_mode}: rel_fp32={e_fp32:.6} rel_q8={e_q8:.6}");
            assert!(e_fp32 < 2.5e-2, "mmq tiled {ty:?} {tile_mode} rel_fp32={e_fp32}");
            assert!(e_q8 < 1e-3, "mmq tiled {ty:?} {tile_mode} rel_q8={e_q8}");
        }

        // launch_gemm_mmq_mma_grouped over 2 experts: exp 0 (12 rows), exp 1 (28 rows)
        let (w1_bytes, _) = fill_mm3_quant_matrix(ty, n, k, &mut rng);
        let d_w1 = to_dev_u8(&w1_bytes);
        let w1_f32 = dequant_matrix_to_f32(ty, &w1_bytes, n, k);
        let d_ptrs = to_dev_u64(&[d_w.as_u64(), d_w1.as_u64()]);
        let d_off = to_dev_i32(&[0i32, 12i32]);
        let d_nt = to_dev_i32(&[12i32, 28i32]);
        let d_te = to_dev_i32(&[0i32, 1i32]);
        let d_tm = to_dev_i32(&[0i32, 0i32]);
        d_y.zero().unwrap();
        unsafe {
            launch_gemm_mmq_mma_grouped(
                ty as i32,
                d_xq8.ptr,
                d_ptrs.ptr as *const u64,
                0,
                rb,
                d_y.ptr as *mut f32,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                d_off.ptr as *const i32,
                d_nt.ptr as *const i32,
                d_te.ptr as *const i32,
                d_tm.ptr as *const i32,
                2,
                n as i32,
                k as i32,
                1.0,
                0,
                3,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let got_grp = read_dev_f32(&d_y, m_max * n);
        let mut exp_grp_q8 = vec![0.0f32; m_max * n];
        for mi in 0..m_max {
            let wf = if mi < 12 { &w_f32 } else { &w1_f32 };
            let xq = &x_q8_deq[mi * k..(mi + 1) * k];
            for ni in 0..n {
                exp_grp_q8[mi * n + ni] = wf[ni * k..(ni + 1) * k]
                    .iter()
                    .zip(xq.iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
            }
        }
        let e_grp = rel_err(&got_grp, &exp_grp_q8);
        println!("mmq_mma_grouped {ty:?}: rel_q8={e_grp:.6}");
        assert!(e_grp < 1e-3, "mmq grouped {ty:?} rel_q8={e_grp}");
    }

    // 2) GPU Q8_K quantization vs CPU AVX-512 Q8_K
    {
        let m = 4usize;
        let kq = 768usize;
        let nb_k = kq / 256;
        let xq_in: Vec<f32> = (0..m * kq).map(|_| rng.next_f32() * 2.5).collect();
        let d_in = to_dev_f32(&xq_in);
        let d_out = DeviceBuffer::alloc(0, m * nb_k * mm3::Q8K_BLOCK_BYTES).unwrap();
        unsafe {
            mm3::quantize_q8_k(d_in.ptr as *const f32, d_out.ptr, m as i32, kq as i32, &stream);
        }
        stream.sync().unwrap();
        let gpu_bytes = read_dev_u8(&d_out, m * nb_k * mm3::Q8K_BLOCK_BYTES);
        let mut cpu_bytes = vec![0u8; m * nb_k * mm3::Q8K_BLOCK_BYTES];
        for r in 0..m {
            unsafe {
                ampllm_cpu_mm3_quantize_q8_k(
                    xq_in.as_ptr().add(r * kq),
                    cpu_bytes.as_mut_ptr().add(r * nb_k * mm3::Q8K_BLOCK_BYTES).cast(),
                    kq as i32,
                );
            }
        }
        let mut max_qs_diff = 0i32;
        for blk in 0..m * nb_k {
            let gb = &gpu_bytes[blk * mm3::Q8K_BLOCK_BYTES..(blk + 1) * mm3::Q8K_BLOCK_BYTES];
            let cb = &cpu_bytes[blk * mm3::Q8K_BLOCK_BYTES..(blk + 1) * mm3::Q8K_BLOCK_BYTES];
            let gd = f32::from_le_bytes([gb[0], gb[1], gb[2], gb[3]]);
            let cd = f32::from_le_bytes([cb[0], cb[1], cb[2], cb[3]]);
            assert!((gd - cd).abs() < 1e-6, "q8k scale diff at blk {blk}: {gd} vs {cd}");
            for i in 0..256 {
                let d = ((gb[4 + i] as i8 as i32) - (cb[4 + i] as i8 as i32)).abs();
                max_qs_diff = max_qs_diff.max(d);
            }
        }
        assert!(max_qs_diff <= 1, "q8k max_qs_diff={max_qs_diff}");
        let mut deq_gpu = Vec::with_capacity(m * kq);
        let mut deq_cpu = Vec::with_capacity(m * kq);
        for r in 0..m {
            let off = r * nb_k * mm3::Q8K_BLOCK_BYTES;
            deq_gpu.extend_from_slice(&dequant_q8_k_blocks(
                &gpu_bytes[off..off + nb_k * mm3::Q8K_BLOCK_BYTES],
                kq,
            ));
            deq_cpu.extend_from_slice(&dequant_q8_k_blocks(
                &cpu_bytes[off..off + nb_k * mm3::Q8K_BLOCK_BYTES],
                kq,
            ));
        }
        let re_q8k = rel_err(&deq_gpu, &deq_cpu);
        println!("quantize_q8_k: max_qs_diff={max_qs_diff}, deq_rel_err={re_q8k:.8}");
        assert!(re_q8k < 5e-4, "q8k deq rel_err={re_q8k}");
    }

    // 3) swigluoai with simultaneous F32, Q8_1, Q8_K, BF16 outputs
    {
        let rows = 5usize;
        let ff = 768usize;
        let alpha = 1.702f32;
        let limit = 7.0f32;
        let gate: Vec<f32> = (0..rows * ff).map(|_| rng.next_f32_range(-10.0, 10.0)).collect();
        let up: Vec<f32> = (0..rows * ff).map(|_| rng.next_f32_range(-10.0, 10.0)).collect();
        let d_g = to_dev_f32(&gate);
        let d_u = to_dev_f32(&up);
        let d_f32 = DeviceBuffer::alloc(0, rows * ff * 4).unwrap();
        let d_q81 = DeviceBuffer::alloc(0, rows * (ff / 32) * 36).unwrap();
        let d_q8k = DeviceBuffer::alloc(0, rows * (ff / 256) * mm3::Q8K_BLOCK_BYTES).unwrap();
        let d_bf16 = DeviceBuffer::alloc(0, rows * ff * 2).unwrap();
        unsafe {
            mm3::swigluoai(
                d_g.ptr as *const f32,
                d_u.ptr as *const f32,
                rows as i32,
                ff as i32,
                alpha,
                limit,
                d_f32.ptr as *mut f32,
                d_q81.ptr,
                d_q8k.ptr,
                d_bf16.ptr as *mut u16,
                &stream,
            );
        }
        stream.sync().unwrap();
        let exp: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&g, &u)| swoai_cpu(g, u, alpha, limit))
            .collect();
        let got_f32 = read_dev_f32(&d_f32, rows * ff);
        let got_bf16: Vec<f32> = read_dev_u16(&d_bf16, rows * ff)
            .into_iter()
            .map(|b| bf16::from_bits(b).to_f32())
            .collect();
        let q81_raw = read_dev_u8(&d_q81, rows * (ff / 32) * 36);
        let q8k_raw = read_dev_u8(&d_q8k, rows * (ff / 256) * mm3::Q8K_BLOCK_BYTES);
        let mut got_q81 = Vec::with_capacity(rows * ff);
        let mut got_q8k = Vec::with_capacity(rows * ff);
        for r in 0..rows {
            got_q81.extend_from_slice(&dequant_q8_1_blocks(
                &q81_raw[r * (ff / 32) * 36..(r + 1) * (ff / 32) * 36],
                ff,
            ));
            got_q8k.extend_from_slice(&dequant_q8_k_blocks(
                &q8k_raw[r * (ff / 256) * mm3::Q8K_BLOCK_BYTES
                    ..(r + 1) * (ff / 256) * mm3::Q8K_BLOCK_BYTES],
                ff,
            ));
        }
        assert!(max_abs_err(&got_f32, &exp) < 1e-5);
        assert!(rel_err(&got_bf16, &exp) < 5e-3);
        assert!(rel_err(&got_q81, &exp) < 1e-2);
        assert!(rel_err(&got_q8k, &exp) < 1e-2);
    }

    // 4) Full GPU MoE Q8_K pipeline vs CPU AVX-512 MM3 MoE + pure FP32 ref
    let pool = CpuThreadPool::new(Some(4));
    let n_embd = 512usize;
    let n_ff = 512usize;
    let num_exp = 8usize;
    let topk = 4usize;
    let mut cpu_scratch = Mm3CpuScratch::new(n_embd, n_ff, 7 * topk);

    let configs = [
        (GgmlType::Iq2Xxs, GgmlType::Iq2Xxs, GgmlType::Iq2Xxs),
        (GgmlType::Iq1S, GgmlType::Iq1S, GgmlType::Iq1S),
        (GgmlType::Iq2Xs, GgmlType::Iq2Xxs, GgmlType::Iq3Xxs),
    ];

    for &(g_ty, u_ty, d_ty) in &configs {
        let g_rb = g_ty.row_bytes(n_embd);
        let u_rb = u_ty.row_bytes(n_embd);
        let d_rb = d_ty.row_bytes(n_ff);

        let mut host_gates = Vec::with_capacity(num_exp);
        let mut host_ups = Vec::with_capacity(num_exp);
        let mut host_downs = Vec::with_capacity(num_exp);
        let mut dev_gates = Vec::with_capacity(num_exp);
        let mut dev_ups = Vec::with_capacity(num_exp);
        let mut dev_downs = Vec::with_capacity(num_exp);
        let mut g_ptrs = Vec::with_capacity(num_exp);
        let mut u_ptrs = Vec::with_capacity(num_exp);
        let mut d_ptrs = Vec::with_capacity(num_exp);
        for _ in 0..num_exp {
            let (gb, _) = fill_mm3_quant_matrix(g_ty, n_ff, n_embd, &mut rng);
            let (ub, _) = fill_mm3_quant_matrix(u_ty, n_ff, n_embd, &mut rng);
            let (db, _) = fill_mm3_quant_matrix(d_ty, n_embd, n_ff, &mut rng);
            let dg = to_dev_u8(&gb);
            let du = to_dev_u8(&ub);
            let dd = to_dev_u8(&db);
            g_ptrs.push(dg.as_u64());
            u_ptrs.push(du.as_u64());
            d_ptrs.push(dd.as_u64());
            host_gates.push(gb);
            host_ups.push(ub);
            host_downs.push(db);
            dev_gates.push(dg);
            dev_ups.push(du);
            dev_downs.push(dd);
        }
        let d_g_ptrs = to_dev_u64(&g_ptrs);
        let d_u_ptrs = to_dev_u64(&u_ptrs);
        let d_d_ptrs = to_dev_u64(&d_ptrs);

        for &m in &[1usize, 7usize] {
            let x_f32: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32()).collect();
            let res: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32() * 0.5).collect();
            let shared: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32() * 0.4).collect();
            let mut expert_ids = vec![0i32; m * topk];
            let mut weights = vec![0.0f32; m * topk];
            for mi in 0..m {
                let mut sum_w = 0.0f32;
                for s in 0..topk {
                    expert_ids[mi * topk + s] = ((mi * 3 + s * 2 + 1) % num_exp) as i32;
                    let w = 0.15 + (s as f32) * 0.1;
                    weights[mi * topk + s] = w;
                    sum_w += w;
                }
                for s in 0..topk {
                    weights[mi * topk + s] /= sum_w;
                }
            }

            let d_x = to_dev_f32(&x_f32);
            let d_res = to_dev_f32(&res);
            let d_shared = to_dev_f32(&shared);
            let d_eids = to_dev_i32(&expert_ids);
            let d_wts = to_dev_f32(&weights);
            let d_x_q8k = DeviceBuffer::alloc(0, m * (n_embd / 256) * mm3::Q8K_BLOCK_BYTES).unwrap();
            let d_inter_q8k =
                DeviceBuffer::alloc(0, m * topk * (n_ff / 256) * mm3::Q8K_BLOCK_BYTES).unwrap();
            let d_inter_f32 = DeviceBuffer::alloc(0, m * topk * n_ff * 4).unwrap();
            let d_exp_out = DeviceBuffer::alloc(0, m * topk * n_embd * 4).unwrap();
            let d_out = DeviceBuffer::alloc(0, m * n_embd * 4).unwrap();

            unsafe {
                mm3::quantize_q8_k(
                    d_x.ptr as *const f32,
                    d_x_q8k.ptr,
                    m as i32,
                    n_embd as i32,
                    &stream,
                );
                mm3::moe_gate_up_q8k(
                    g_ty as i32,
                    u_ty as i32,
                    d_g_ptrs.ptr as *const u64,
                    d_u_ptrs.ptr as *const u64,
                    g_rb,
                    u_rb,
                    d_x_q8k.ptr,
                    d_eids.ptr as *const i32,
                    d_inter_q8k.ptr,
                    d_inter_f32.ptr as *mut f32,
                    m as i32,
                    n_embd as i32,
                    n_ff as i32,
                    topk as i32,
                    1.702,
                    7.0,
                    std::ptr::null(),
                    &stream,
                );
                mm3::moe_down_q8k(
                    d_ty as i32,
                    d_d_ptrs.ptr as *const u64,
                    d_rb,
                    d_inter_q8k.ptr,
                    d_eids.ptr as *const i32,
                    d_exp_out.ptr as *mut f32,
                    m as i32,
                    n_ff as i32,
                    n_embd as i32,
                    topk as i32,
                    std::ptr::null(),
                    &stream,
                );
                mm3::moe_combine(
                    d_res.ptr as *const f32,
                    d_exp_out.ptr as *const f32,
                    d_wts.ptr as *const f32,
                    d_eids.ptr as *const i32,
                    std::ptr::null(),
                    d_shared.ptr as *const f32,
                    m as i32,
                    topk as i32,
                    n_embd as i32,
                    d_out.ptr as *mut f32,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let gpu_out = read_dev_f32(&d_out, m * n_embd);

            // CPU AVX-512 MM3 MoE reference
            let mut cpu_moe = vec![0.0f32; m * n_embd];
            if m == 1 {
                let jobs: Vec<Mm3CpuExpertJob> = (0..topk)
                    .map(|s| {
                        let e = expert_ids[s] as usize;
                        Mm3CpuExpertJob {
                            gate_ptr: host_gates[e].as_ptr(),
                            up_ptr: host_ups[e].as_ptr(),
                            down_ptr: host_downs[e].as_ptr(),
                            weight: weights[s],
                        }
                    })
                    .collect();
                run_mm3_cpu_moe_m1(
                    &pool,
                    &mut cpu_scratch,
                    g_ty,
                    u_ty,
                    d_ty,
                    n_embd,
                    n_ff,
                    1.702,
                    7.0,
                    &x_f32,
                    None,
                    &jobs,
                    &mut cpu_moe,
                );
            } else {
                let mut per_exp_toks: Vec<Vec<usize>> = vec![Vec::new(); num_exp];
                let mut per_exp_wts: Vec<Vec<f32>> = vec![Vec::new(); num_exp];
                for mi in 0..m {
                    for s in 0..topk {
                        let e = expert_ids[mi * topk + s] as usize;
                        per_exp_toks[e].push(mi);
                        per_exp_wts[e].push(weights[mi * topk + s]);
                    }
                }
                let groups: Vec<Mm3CpuGroupedExpert<'_>> = (0..num_exp)
                    .filter(|&e| !per_exp_toks[e].is_empty())
                    .map(|e| Mm3CpuGroupedExpert {
                        gate_ptr: host_gates[e].as_ptr(),
                        up_ptr: host_ups[e].as_ptr(),
                        down_ptr: host_downs[e].as_ptr(),
                        token_indices: &per_exp_toks[e],
                        token_weights: &per_exp_wts[e],
                    })
                    .collect();
                run_mm3_cpu_moe_grouped(
                    &pool,
                    &mut cpu_scratch,
                    g_ty,
                    u_ty,
                    d_ty,
                    n_embd,
                    n_ff,
                    1.702,
                    7.0,
                    m,
                    &x_f32,
                    &groups,
                    &mut cpu_moe,
                    true,
                );
            }
            let cpu_combined: Vec<f32> = cpu_moe
                .iter()
                .zip(res.iter().zip(shared.iter()))
                .map(|(&c, (&r, &s))| c + r + s)
                .collect();

            let e_vs_cpu = rel_err(&gpu_out, &cpu_combined);

            // Pure FP32 dequantized reference
            let mut fp32_ref = vec![0.0f32; m * n_embd];
            for mi in 0..m {
                let xr = &x_f32[mi * n_embd..(mi + 1) * n_embd];
                for d in 0..n_embd {
                    fp32_ref[mi * n_embd + d] = res[mi * n_embd + d] + shared[mi * n_embd + d];
                }
                for s in 0..topk {
                    let e = expert_ids[mi * topk + s] as usize;
                    let w = weights[mi * topk + s];
                    let gf = dequant_matrix_to_f32(g_ty, &host_gates[e], n_ff, n_embd);
                    let uf = dequant_matrix_to_f32(u_ty, &host_ups[e], n_ff, n_embd);
                    let df = dequant_matrix_to_f32(d_ty, &host_downs[e], n_embd, n_ff);
                    let mut inter = vec![0.0f32; n_ff];
                    for r in 0..n_ff {
                        let gdot: f32 = gf[r * n_embd..(r + 1) * n_embd]
                            .iter()
                            .zip(xr.iter())
                            .map(|(&a, &b)| a * b)
                            .sum();
                        let udot: f32 = uf[r * n_embd..(r + 1) * n_embd]
                            .iter()
                            .zip(xr.iter())
                            .map(|(&a, &b)| a * b)
                            .sum();
                        inter[r] = swoai_cpu(gdot, udot, 1.702, 7.0);
                    }
                    for d in 0..n_embd {
                        let ddot: f32 = df[d * n_ff..(d + 1) * n_ff]
                            .iter()
                            .zip(inter.iter())
                            .map(|(&a, &b)| a * b)
                            .sum();
                        fp32_ref[mi * n_embd + d] += w * ddot;
                    }
                }
            }
            let e_vs_fp32 = rel_err(&gpu_out, &fp32_ref);
            println!(
                "gpu_moe ({g_ty:?},{u_ty:?},{d_ty:?}) M={m}: rel_vs_cpu={e_vs_cpu:.7} rel_vs_fp32={e_vs_fp32:.6}"
            );
            assert!(
                e_vs_cpu < 1e-4,
                "gpu_moe ({g_ty:?},{u_ty:?},{d_ty:?}) M={m} rel_vs_cpu={e_vs_cpu}"
            );
            assert!(
                e_vs_fp32 < 3e-2,
                "gpu_moe ({g_ty:?},{u_ty:?},{d_ty:?}) M={m} rel_vs_fp32={e_vs_fp32}"
            );

            // Also verify vram_ptrs_opt filtering in moe_combine at M=1
            if m == 1 {
                let mut vram_ptrs = vec![1u64; num_exp];
                let cold_eid = expert_ids[0] as usize;
                vram_ptrs[cold_eid] = 0u64;
                let d_vp = to_dev_u64(&vram_ptrs);
                let exp_out_host = read_dev_f32(&d_exp_out, topk * n_embd);
                unsafe {
                    mm3::moe_combine(
                        d_res.ptr as *const f32,
                        d_exp_out.ptr as *const f32,
                        d_wts.ptr as *const f32,
                        d_eids.ptr as *const i32,
                        d_vp.ptr as *const u64,
                        std::ptr::null(),
                        1,
                        topk as i32,
                        n_embd as i32,
                        d_out.ptr as *mut f32,
                        &stream,
                    );
                }
                stream.sync().unwrap();
                let got_filt = read_dev_f32(&d_out, n_embd);
                let mut exp_filt = res[..n_embd].to_vec();
                for s in 0..topk {
                    if vram_ptrs[expert_ids[s] as usize] != 0 {
                        for d in 0..n_embd {
                            exp_filt[d] += weights[s] * exp_out_host[s * n_embd + d];
                        }
                    }
                }
                let ma_filt = max_abs_err(&got_filt, &exp_filt);
                let re_filt = rel_err(&got_filt, &exp_filt);
                println!("moe_combine vram_ptrs filter M=1 ({g_ty:?}): max_abs={ma_filt:.7} rel={re_filt:.7}");
                assert!(re_filt < 1e-5 && ma_filt < 5e-4, "ma_filt={ma_filt} re_filt={re_filt}");
            }
        }
    }
}

// ============================================================================
// Test 2: RMSNorm, QK-Norm-RoPE-KV-Store (BF16/INT8/Had128), MSA Sparse Attn, Router
// ============================================================================

#[test]
fn test_mm3_msa_kv_int8_bf16_and_router() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x3331_0002_CAFE_BEEF);

    // 1) rms_norm_f32: dim=512 (F32, Q8_1, Q8_K, BF16) with add_one = false & true, plus dim=320
    for &(dim, add_one) in &[(512usize, false), (512usize, true), (320usize, false)] {
        let rows = 4usize;
        let eps = 1e-6f32;
        let x: Vec<f32> = (0..rows * dim).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..dim).map(|_| rng.next_f32_range(0.5, 1.5)).collect();
        let d_x = to_dev_f32(&x);
        let d_w = to_dev_f32(&w);
        let d_f32 = DeviceBuffer::alloc(0, rows * dim * 4).unwrap();
        let d_q81 = DeviceBuffer::alloc(0, rows * (dim / 32) * 36).unwrap();
        let d_q8k = DeviceBuffer::alloc(0, rows * (dim.div_ceil(256)) * mm3::Q8K_BLOCK_BYTES).unwrap();
        let d_bf16 = DeviceBuffer::alloc(0, rows * dim * 2).unwrap();
        let q8k_ptr = if dim % 256 == 0 { d_q8k.ptr } else { std::ptr::null_mut() };

        unsafe {
            mm3::rms_norm_f32(
                d_x.ptr as *const f32,
                d_w.ptr as *const f32,
                rows as i32,
                dim as i32,
                eps,
                add_one,
                d_f32.ptr as *mut f32,
                d_q81.ptr,
                q8k_ptr,
                d_bf16.ptr as *mut u16,
                &stream,
            );
        }
        stream.sync().unwrap();

        let mut exp = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let xr = &x[r * dim..(r + 1) * dim];
            let ss: f32 = xr.iter().map(|&v| v * v).sum();
            let inv = 1.0f32 / (ss / (dim as f32) + eps).sqrt();
            for i in 0..dim {
                let wt = if add_one { 1.0 + w[i] } else { w[i] };
                exp[r * dim + i] = xr[i] * inv * wt;
            }
        }
        let got_f32 = read_dev_f32(&d_f32, rows * dim);
        let max_d = max_abs_err(&got_f32, &exp);
        println!("rms_norm_f32 dim={dim} add_one={add_one}: max_abs={max_d:.7}");
        assert!(max_d < 1e-5, "rms_norm_f32 max_d={max_d}");

        let got_bf16: Vec<f32> = read_dev_u16(&d_bf16, rows * dim)
            .into_iter()
            .map(|b| bf16::from_bits(b).to_f32())
            .collect();
        assert!(rel_err(&got_bf16, &exp) < 5e-3);

        let q81_raw = read_dev_u8(&d_q81, rows * (dim / 32) * 36);
        let mut got_q81 = Vec::with_capacity(rows * dim);
        for r in 0..rows {
            got_q81.extend_from_slice(&dequant_q8_1_blocks(
                &q81_raw[r * (dim / 32) * 36..(r + 1) * (dim / 32) * 36],
                dim,
            ));
        }
        assert!(rel_err(&got_q81, &exp) < 1e-2);

        if dim % 256 == 0 {
            let nb_k = dim / 256;
            let q8k_raw = read_dev_u8(&d_q8k, rows * nb_k * mm3::Q8K_BLOCK_BYTES);
            let mut got_q8k = Vec::with_capacity(rows * dim);
            for r in 0..rows {
                got_q8k.extend_from_slice(&dequant_q8_k_blocks(
                    &q8k_raw[r * nb_k * mm3::Q8K_BLOCK_BYTES..(r + 1) * nb_k * mm3::Q8K_BLOCK_BYTES],
                    dim,
                ));
            }
            assert!(rel_err(&got_q8k, &exp) < 1e-2);
        }
    }

    // 2) qk_norm_rope_kv_store + msa_sparse_attn across BF16 (kv_fmt=0) & INT8 Had128 (kv_fmt=1, rot_mask=2 & 3)
    for &(kv_fmt, rot_mask, ring) in &[(0i32, 0i32, 0i32), (1, 2, 0), (1, 3, 64)] {
        let seq_len = if ring > 0 { 64usize } else { 96usize };
        let t_q = 4usize;
        let hq = 16usize;
        let hkv = 2usize; // G = 8
        let hd = 128usize;
        let rot_dim = 64usize;
        let eps = 1e-6f32;
        let blk = 16usize;
        let topk = 4usize;
        let pos_offset = if ring > 0 { 32i32 } else { 0i32 }; // fill positions [32..96) into ring=64 -> tests wraparound!

        let inv_freq: Vec<f32> = (0..rot_dim / 2)
            .map(|i| 10000.0f32.powf(-((2 * i) as f32) / (rot_dim as f32)))
            .collect();
        let qn_w: Vec<f32> = (0..hd).map(|_| rng.next_f32_range(0.6, 1.4)).collect();
        let kn_w: Vec<f32> = (0..hd).map(|_| rng.next_f32_range(0.6, 1.4)).collect();
        let all_k: Vec<f32> = (0..seq_len * hkv * hd).map(|_| rng.next_f32()).collect();
        let all_v: Vec<f32> = (0..seq_len * hkv * hd).map(|_| rng.next_f32()).collect();
        let all_pos: Vec<i32> = (0..seq_len).map(|i| pos_offset + i as i32).collect();
        let q_pos: Vec<i32> = (0..t_q).map(|i| pos_offset + (seq_len - t_q + i) as i32).collect();
        let q_init: Vec<f32> = (0..t_q * hq * hd).map(|_| rng.next_f32()).collect();

        let d_inv = to_dev_f32(&inv_freq);
        let d_qnw = to_dev_f32(&qn_w);
        let d_knw = to_dev_f32(&kn_w);
        let d_k_all = to_dev_f32(&all_k);
        let d_v_all = to_dev_f32(&all_v);
        let d_pos_all = to_dev_i32(&all_pos);
        let d_q_pos = to_dev_i32(&q_pos);
        let d_q = to_dev_f32(&q_init);

        let cap = if ring > 0 { ring as usize } else { seq_len };
        let d_kc_bf16 = DeviceBuffer::alloc_zeroed(0, cap * hkv * hd * 2).unwrap();
        let d_vc_bf16 = DeviceBuffer::alloc_zeroed(0, cap * hkv * hd * 2).unwrap();
        let d_kc_i8 = DeviceBuffer::alloc_zeroed(0, cap * hkv * hd).unwrap();
        let d_ks_i8 = DeviceBuffer::alloc_zeroed(0, cap * hkv * 4).unwrap();
        let d_vc_i8 = DeviceBuffer::alloc_zeroed(0, cap * hkv * hd).unwrap();
        let d_vs_i8 = DeviceBuffer::alloc_zeroed(0, cap * hkv * 4).unwrap();

        let host_cap = (pos_offset as usize) + seq_len;
        let mut h_tag = PinnedMappedBuffer::<i32>::alloc(cap).unwrap();
        h_tag.as_mut_slice().fill(-1);
        let h_kc_bf16 = PinnedMappedBuffer::<u16>::alloc(host_cap * hkv * hd).unwrap();
        let h_vc_bf16 = PinnedMappedBuffer::<u16>::alloc(host_cap * hkv * hd).unwrap();
        let h_kc_i8 = PinnedMappedBuffer::<i8>::alloc(host_cap * hkv * hd).unwrap();
        let h_ks_i8 = PinnedMappedBuffer::<f32>::alloc(host_cap * hkv).unwrap();
        let h_vc_i8 = PinnedMappedBuffer::<i8>::alloc(host_cap * hkv * hd).unwrap();
        let h_vs_i8 = PinnedMappedBuffer::<f32>::alloc(host_cap * hkv).unwrap();

        unsafe {
            // Store KV sequence [0..seq_len) while leaving Q null for first call, then update Q on q_pos
            if kv_fmt == 0 {
                mm3::qk_norm_rope_kv_store(
                    std::ptr::null_mut(),
                    d_k_all.ptr as *const f32,
                    d_v_all.ptr as *const f32,
                    d_qnw.ptr as *const f32,
                    d_knw.ptr as *const f32,
                    d_pos_all.ptr as *const i32,
                    d_inv.ptr as *const f32,
                    seq_len as i32,
                    hq as i32,
                    hkv as i32,
                    hd as i32,
                    rot_dim as i32,
                    eps,
                    kv_fmt,
                    rot_mask,
                    ring,
                    d_kc_bf16.ptr as *mut u16,
                    d_vc_bf16.ptr as *mut u16,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    h_kc_bf16.dev_ptr(),
                    std::ptr::null_mut(),
                    h_vc_bf16.dev_ptr(),
                    std::ptr::null_mut(),
                    h_tag.dev_ptr() as *mut i32,
                    &stream,
                );
            } else {
                mm3::qk_norm_rope_kv_store(
                    std::ptr::null_mut(),
                    d_k_all.ptr as *const f32,
                    d_v_all.ptr as *const f32,
                    d_qnw.ptr as *const f32,
                    d_knw.ptr as *const f32,
                    d_pos_all.ptr as *const i32,
                    d_inv.ptr as *const f32,
                    seq_len as i32,
                    hq as i32,
                    hkv as i32,
                    hd as i32,
                    rot_dim as i32,
                    eps,
                    kv_fmt,
                    rot_mask,
                    ring,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    d_kc_i8.ptr as *mut i8,
                    d_ks_i8.ptr as *mut f32,
                    d_vc_i8.ptr as *mut i8,
                    d_vs_i8.ptr as *mut f32,
                    h_kc_i8.dev_ptr(),
                    h_ks_i8.dev_ptr() as *mut f32,
                    h_vc_i8.dev_ptr(),
                    h_vs_i8.dev_ptr() as *mut f32,
                    h_tag.dev_ptr() as *mut i32,
                    &stream,
                );
            }
            // Now normalize+RoPE Q and store last t_q positions again identically
            let off_kv = (seq_len - t_q) * hkv * hd * 4;
            mm3::qk_norm_rope_kv_store(
                d_q.ptr as *mut f32,
                d_k_all.ptr.add(off_kv) as *const f32,
                d_v_all.ptr.add(off_kv) as *const f32,
                d_qnw.ptr as *const f32,
                d_knw.ptr as *const f32,
                d_q_pos.ptr as *const i32,
                d_inv.ptr as *const f32,
                t_q as i32,
                hq as i32,
                hkv as i32,
                hd as i32,
                rot_dim as i32,
                eps,
                kv_fmt,
                rot_mask,
                ring,
                d_kc_bf16.ptr as *mut u16,
                d_vc_bf16.ptr as *mut u16,
                d_kc_i8.ptr as *mut i8,
                d_ks_i8.ptr as *mut f32,
                d_vc_i8.ptr as *mut i8,
                d_vs_i8.ptr as *mut f32,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &stream,
            );
        }
        stream.sync().unwrap();

        let got_q = read_dev_f32(&d_q, t_q * hq * hd);
        let mut exp_q = vec![0.0f32; t_q * hq * hd];
        for t in 0..t_q {
            for h in 0..hq {
                let sl = &q_init[(t * hq + h) * hd..(t * hq + h + 1) * hd];
                let res = norm_rope_cpu_head(sl, Some(&qn_w), q_pos[t], rot_dim, &inv_freq, eps);
                exp_q[(t * hq + h) * hd..(t * hq + h + 1) * hd].copy_from_slice(&res);
            }
        }
        assert!(
            max_abs_err(&got_q, &exp_q) < 1e-5,
            "q norm+rope max_abs={}",
            max_abs_err(&got_q, &exp_q)
        );

        // Verify host tags & CPU-dequantized KV cache vs pure F32 K_norm_rope and V
        let mut ref_k_rope = vec![0.0f32; seq_len * hkv * hd];
        for i in 0..seq_len {
            let p = all_pos[i];
            let slot = if ring > 0 { (p as usize) % (ring as usize) } else { p as usize };
            assert_eq!(h_tag.as_slice()[slot], p);
            for kh in 0..hkv {
                let kr = norm_rope_cpu_head(
                    &all_k[(i * hkv + kh) * hd..(i * hkv + kh + 1) * hd],
                    Some(&kn_w),
                    p,
                    rot_dim,
                    &inv_freq,
                    eps,
                );
                ref_k_rope[(i * hkv + kh) * hd..(i * hkv + kh + 1) * hd].copy_from_slice(&kr);
            }
        }

        // Verify host mirrored KV matches expectation
        for i in 0..seq_len {
            let p = all_pos[i] as usize;
            for kh in 0..hkv {
                let ah = p * hkv + kh;
                let mut k_got = vec![0.0f32; hd];
                let mut v_got = vec![0.0f32; hd];
                if kv_fmt == 0 {
                    for d in 0..hd {
                        k_got[d] = bf16::from_bits(h_kc_bf16.as_slice()[ah * hd + d]).to_f32();
                        v_got[d] = bf16::from_bits(h_vc_bf16.as_slice()[ah * hd + d]).to_f32();
                    }
                } else {
                    let ks = h_ks_i8.as_slice()[ah];
                    let vs = h_vs_i8.as_slice()[ah];
                    for d in 0..hd {
                        k_got[d] = (h_kc_i8.as_slice()[ah * hd + d] as f32) * ks;
                        v_got[d] = (h_vc_i8.as_slice()[ah * hd + d] as f32) * vs;
                    }
                    if (rot_mask & 1) != 0 {
                        k_got = had128_cpu(&k_got);
                    }
                    if (rot_mask & 2) != 0 {
                        v_got = had128_cpu(&v_got);
                    }
                }
                let k_exp = &ref_k_rope[(i * hkv + kh) * hd..(i * hkv + kh + 1) * hd];
                let v_exp = &all_v[(i * hkv + kh) * hd..(i * hkv + kh + 1) * hd];
                let tol = if kv_fmt == 0 { 2e-2 } else { 4e-2 };
                assert!(max_abs_err(&k_got, k_exp) < tol);
                assert!(max_abs_err(&v_got, v_exp) < tol);
            }
        }

        // Kept blocks within valid positions stored in cache:
        // for ring == 0, pos_offset == 0, valid blocks 0..6; select [0, 2, 4, 5]
        // for ring == 64, pos_offset == 32 (pos 32..95), valid blocks 2..6
        let kept_blocks: Vec<i32> = if ring > 0 {
            vec![5i32, 2, 4, 3]
        } else {
            vec![5i32, 0, 2, 4]
        };
        let mut kept_all = Vec::with_capacity(hkv * t_q * topk);
        for _g in 0..hkv {
            for _t in 0..t_q {
                kept_all.extend_from_slice(&kept_blocks);
            }
        }
        let d_kept = to_dev_i32(&kept_all);
        let d_out1 = DeviceBuffer::alloc(0, t_q * hq * hd * 4).unwrap();
        let d_out4 = DeviceBuffer::alloc(0, t_q * hq * hd * 4).unwrap();
        let d_part = DeviceBuffer::alloc(
            0,
            t_q * hkv * 4 * (hq / hkv) * mm3::ATTN_PART_STRIDE * 4,
        )
        .unwrap();
        let scale = 1.0f32 / (hd as f32).sqrt();

        unsafe {
            mm3::msa_sparse_attn(
                d_q.ptr as *const f32,
                if kv_fmt == 0 { d_kc_bf16.ptr as *const u16 } else { std::ptr::null() },
                if kv_fmt == 0 { d_vc_bf16.ptr as *const u16 } else { std::ptr::null() },
                if kv_fmt == 1 { d_kc_i8.ptr as *const i8 } else { std::ptr::null() },
                if kv_fmt == 1 { d_ks_i8.ptr as *const f32 } else { std::ptr::null() },
                if kv_fmt == 1 { d_vc_i8.ptr as *const i8 } else { std::ptr::null() },
                if kv_fmt == 1 { d_vs_i8.ptr as *const f32 } else { std::ptr::null() },
                d_kept.ptr as *const i32,
                d_q_pos.ptr as *const i32,
                t_q as i32,
                hq as i32,
                hkv as i32,
                topk as i32,
                blk as i32,
                kv_fmt,
                rot_mask,
                ring,
                scale,
                std::ptr::null_mut(),
                1,
                d_out1.ptr as *mut f32,
                &stream,
            );
            mm3::msa_sparse_attn(
                d_q.ptr as *const f32,
                if kv_fmt == 0 { d_kc_bf16.ptr as *const u16 } else { std::ptr::null() },
                if kv_fmt == 0 { d_vc_bf16.ptr as *const u16 } else { std::ptr::null() },
                if kv_fmt == 1 { d_kc_i8.ptr as *const i8 } else { std::ptr::null() },
                if kv_fmt == 1 { d_ks_i8.ptr as *const f32 } else { std::ptr::null() },
                if kv_fmt == 1 { d_vc_i8.ptr as *const i8 } else { std::ptr::null() },
                if kv_fmt == 1 { d_vs_i8.ptr as *const f32 } else { std::ptr::null() },
                d_kept.ptr as *const i32,
                d_q_pos.ptr as *const i32,
                t_q as i32,
                hq as i32,
                hkv as i32,
                topk as i32,
                blk as i32,
                kv_fmt,
                rot_mask,
                ring,
                scale,
                d_part.ptr as *mut f32,
                4,
                d_out4.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();

        let out1 = read_dev_f32(&d_out1, t_q * hq * hd);
        let out4 = read_dev_f32(&d_out4, t_q * hq * hd);
        let s_diff = max_abs_err(&out1, &out4);
        assert!(s_diff < 1e-5, "msa split=1 vs split=4 diff={s_diff}");

        // CPU sparse attention over pure F32 K/V reference
        let mut cpu_out = vec![0.0f32; t_q * hq * hd];
        let gqa = hq / hkv;
        for t in 0..t_q {
            let qp = q_pos[t];
            for g in 0..hkv {
                let mut valid_keys = Vec::new();
                for &b in &kept_blocks {
                    if b < 0 {
                        continue;
                    }
                    let ks = (b as usize) * blk;
                    let ke = ((b as usize + 1) * blk).min((qp as usize) + 1);
                    for kp in ks..ke {
                        valid_keys.push(kp);
                    }
                }
                for gh in 0..gqa {
                    let h = g * gqa + gh;
                    let qh = &got_q[(t * hq + h) * hd..(t * hq + h + 1) * hd];
                    let scores: Vec<f32> = valid_keys
                        .iter()
                        .map(|&kp| {
                            let idx = kp - (pos_offset as usize);
                            let kr = &ref_k_rope[(idx * hkv + g) * hd..(idx * hkv + g + 1) * hd];
                            qh.iter().zip(kr.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale
                        })
                        .collect();
                    let m_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps = vec![0.0f32; scores.len()];
                    let mut sum_e = 0.0f32;
                    for (j, &sc) in scores.iter().enumerate() {
                        let e = (sc - m_max).exp();
                        exps[j] = e;
                        sum_e += e;
                    }
                    for d in 0..hd {
                        let mut acc = 0.0f32;
                        for (j, &kp) in valid_keys.iter().enumerate() {
                            let idx = kp - (pos_offset as usize);
                            acc += (exps[j] / sum_e) * all_v[(idx * hkv + g) * hd + d];
                        }
                        cpu_out[(t * hq + h) * hd + d] = acc;
                    }
                }
            }
        }
        let re_msa = rel_err(&out1, &cpu_out);
        println!(
            "msa_sparse_attn kv_fmt={kv_fmt} rot_mask={rot_mask} ring={ring}: split_diff={s_diff:.7} rel_vs_fp32={re_msa:.6}"
        );
        let tol_re = if kv_fmt == 0 { 3e-3 } else { 2e-2 };
        assert!(re_msa < tol_re, "msa rel_err={re_msa} >= {tol_re}");
    }

    // 3) index_norm_rope_store + msa_block_scores + msa_select_topk
    {
        let t_cnt = 5usize;
        let hi = 4usize;
        let hd = 128usize;
        let rot_dim = 64usize;
        let eps = 1e-6f32;
        let blk = 16usize;
        let nblk = 8usize;
        let topk = 4usize;
        let local_blocks = 1usize;

        let pos_q = vec![10i32, 25, 50, 79, 120];
        let max_ctx = 128usize;
        let inv_freq: Vec<f32> = (0..rot_dim / 2)
            .map(|i| 10000.0f32.powf(-((2 * i) as f32) / (rot_dim as f32)))
            .collect();
        let iq_nw: Vec<f32> = (0..hd).map(|_| rng.next_f32_range(0.7, 1.3)).collect();
        let ik_nw: Vec<f32> = (0..hd).map(|_| rng.next_f32_range(0.7, 1.3)).collect();

        // Populate indexer cache at all 0..128 positions first
        let all_ik: Vec<f32> = (0..max_ctx * hd).map(|_| rng.next_f32()).collect();
        let all_pos: Vec<i32> = (0..max_ctx as i32).collect();
        let mut iq_buf: Vec<f32> = (0..t_cnt * hi * hd).map(|_| rng.next_f32()).collect();

        let d_inv = to_dev_f32(&inv_freq);
        let d_iqnw = to_dev_f32(&iq_nw);
        let d_iknw = to_dev_f32(&ik_nw);
        let d_ik_all = to_dev_f32(&all_ik);
        let d_pos_all = to_dev_i32(&all_pos);
        let d_pos_q = to_dev_i32(&pos_q);
        let d_iq = to_dev_f32(&iq_buf);
        let d_iq_dummy = DeviceBuffer::alloc(0, max_ctx * hi * hd * 4).unwrap();
        let d_ic = DeviceBuffer::alloc_zeroed(0, max_ctx * hd * 2).unwrap();

        unsafe {
            mm3::index_norm_rope_store(
                d_iq_dummy.ptr as *mut f32,
                d_ik_all.ptr as *const f32,
                d_iqnw.ptr as *const f32,
                d_iknw.ptr as *const f32,
                d_pos_all.ptr as *const i32,
                d_inv.ptr as *const f32,
                max_ctx as i32,
                hi as i32,
                hd as i32,
                rot_dim as i32,
                eps,
                d_ic.ptr as *mut u16,
                &stream,
            );
            mm3::index_norm_rope_store(
                d_iq.ptr as *mut f32,
                d_ik_all.ptr as *const f32,
                d_iqnw.ptr as *const f32,
                d_iknw.ptr as *const f32,
                d_pos_q.ptr as *const i32,
                d_inv.ptr as *const f32,
                t_cnt as i32,
                hi as i32,
                hd as i32,
                rot_dim as i32,
                eps,
                d_ic.ptr as *mut u16,
                &stream,
            );
        }
        stream.sync().unwrap();

        // Check normalized+RoPE+BF16-rounded iq & ic
        let got_iq = read_dev_f32(&d_iq, t_cnt * hi * hd);
        for t in 0..t_cnt {
            for h in 0..hi {
                let src = &iq_buf[(t * hi + h) * hd..(t * hi + h + 1) * hd];
                let exp: Vec<f32> = norm_rope_cpu_head(src, Some(&iq_nw), pos_q[t], rot_dim, &inv_freq, eps)
                    .into_iter()
                    .map(|v| bf16::from_f32(v).to_f32())
                    .collect();
                assert!(max_abs_err(&got_iq[(t * hi + h) * hd..(t * hi + h + 1) * hd], &exp) < 1e-2);
            }
        }
        iq_buf.copy_from_slice(&got_iq);
        let got_ic: Vec<f32> = read_dev_u16(&d_ic, max_ctx * hd)
            .into_iter()
            .map(|b| bf16::from_bits(b).to_f32())
            .collect();

        // msa_block_scores + msa_select_topk
        let d_bs = DeviceBuffer::alloc(0, hi * t_cnt * nblk * 4).unwrap();
        let d_kept = DeviceBuffer::alloc(0, hi * t_cnt * topk * 4).unwrap();
        unsafe {
            mm3::msa_block_scores(
                d_iq.ptr as *const f32,
                d_ic.ptr as *const u16,
                d_pos_q.ptr as *const i32,
                t_cnt as i32,
                hi as i32,
                nblk as i32,
                blk as i32,
                d_bs.ptr as *mut f32,
                &stream,
            );
            mm3::msa_select_topk(
                d_bs.ptr as *const f32,
                d_pos_q.ptr as *const i32,
                t_cnt as i32,
                hi as i32,
                nblk as i32,
                blk as i32,
                topk as i32,
                local_blocks as i32,
                d_kept.ptr as *mut i32,
                &stream,
            );
        }
        stream.sync().unwrap();

        let got_bs = read_dev_f32(&d_bs, hi * t_cnt * nblk);
        let got_kept = read_dev_i32(&d_kept, hi * t_cnt * topk);
        let mut exp_bs = vec![f32::NEG_INFINITY; hi * t_cnt * nblk];
        for h in 0..hi {
            for t in 0..t_cnt {
                let p = pos_q[t] as usize;
                let qr = &iq_buf[(t * hi + h) * hd..(t * hi + h + 1) * hd];
                for b in 0..nblk {
                    let ks = b * blk;
                    if ks > p {
                        continue;
                    }
                    let ke = (ks + blk).min(p + 1);
                    let mut best = f32::NEG_INFINITY;
                    for k_i in ks..ke {
                        let kr = &got_ic[k_i * hd..(k_i + 1) * hd];
                        let dot: f32 = qr.iter().zip(kr.iter()).map(|(&a, &b)| a * b).sum();
                        best = best.max(dot);
                    }
                    exp_bs[(h * t_cnt + t) * nblk + b] = best;
                    let got = got_bs[(h * t_cnt + t) * nblk + b];
                    assert!(
                        (got - best).abs() < 1e-4,
                        "bs mismatch h={h} t={t} b={b}: got={got} exp={best}"
                    );
                }

                // Exact select check
                let qb = p / blk;
                let gslice = &got_kept[(h * t_cnt + t) * topk..(h * t_cnt + t + 1) * topk];
                if qb + 1 <= topk {
                    for i in 0..topk {
                        let expected = if i <= qb { i as i32 } else { -1 };
                        assert_eq!(gslice[i], expected, "early fill mismatch h={h} t={t}");
                    }
                } else {
                    assert_eq!(gslice[0], qb as i32, "own block not forced h={h} t={t}");
                    let mut cands: Vec<(f32, usize)> = (0..qb)
                        .map(|b| (exp_bs[(h * t_cnt + t) * nblk + b], b))
                        .collect();
                    cands.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.1.cmp(&b.1)));
                    for r in 1..topk {
                        assert_eq!(gslice[r], cands[r - 1].1 as i32);
                    }
                }
            }
        }

        // Crafted tie-break test for msa_select_topk: identical scores at block 1 and block 2
        let tie_bs = vec![-1.0f32, 3.5, 3.5, 0.2, 1.0, 9.0, 0.0, 0.0];
        let tie_pos = vec![(5 * blk + 3) as i32]; // qb = 5 -> block 5 forced first, then topk-1 from 0..5 -> 1, 2, 4
        let d_tbs = to_dev_f32(&tie_bs);
        let d_tpos = to_dev_i32(&tie_pos);
        let d_tkept = DeviceBuffer::alloc(0, 4 * 4).unwrap();
        unsafe {
            mm3::msa_select_topk(
                d_tbs.ptr as *const f32,
                d_tpos.ptr as *const i32,
                1,
                1,
                8,
                blk as i32,
                4,
                1,
                d_tkept.ptr as *mut i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        assert_eq!(read_dev_i32(&d_tkept, 4), vec![5, 1, 2, 4]);
    }

    // 4) router_f32 (x * gate_w^T mode + precomputed logits mode + tie-break)
    {
        let t_cnt = 6usize;
        let d_dim = 256usize;
        let e_cnt = 64usize;
        let topk = 6usize;
        let scale = 1.75f32;
        let x: Vec<f32> = (0..t_cnt * d_dim).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..e_cnt * d_dim).map(|_| rng.next_f32() * 0.2).collect();
        let bias: Vec<f32> = (0..e_cnt).map(|_| rng.next_f32_range(-0.2, 0.2)).collect();

        let d_x = to_dev_f32(&x);
        let d_w = to_dev_f32(&w);
        let d_b = to_dev_f32(&bias);
        let d_logits = DeviceBuffer::alloc(0, t_cnt * e_cnt * 4).unwrap();
        let d_ids = DeviceBuffer::alloc(0, t_cnt * topk * 4).unwrap();
        let d_wts = DeviceBuffer::alloc(0, t_cnt * topk * 4).unwrap();

        unsafe {
            mm3::router_f32(
                d_x.ptr as *const f32,
                d_w.ptr as *const f32,
                d_logits.ptr as *mut f32,
                d_b.ptr as *const f32,
                t_cnt as i32,
                d_dim as i32,
                e_cnt as i32,
                topk as i32,
                scale,
                d_ids.ptr as *mut i32,
                d_wts.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();

        let got_ids = read_dev_i32(&d_ids, t_cnt * topk);
        let got_wts = read_dev_f32(&d_wts, t_cnt * topk);
        for t in 0..t_cnt {
            let mut scored = Vec::with_capacity(e_cnt);
            for e in 0..e_cnt {
                let dot: f32 = x[t * d_dim..(t + 1) * d_dim]
                    .iter()
                    .zip(w[e * d_dim..(e + 1) * d_dim].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
                let sg = 1.0f32 / (1.0f32 + (-dot).exp());
                scored.push((sg + bias[e], sg, e));
            }
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.2.cmp(&b.2)));
            let wsum: f32 = scored[..topk].iter().map(|s| s.1).sum();
            for r in 0..topk {
                assert_eq!(got_ids[t * topk + r], scored[r].2 as i32);
                let exp_w = scored[r].1 * (scale / wsum);
                assert!((got_wts[t * topk + r] - exp_w).abs() < 1e-5);
            }
        }

        // Precomputed logits mode with explicit tie at expert 3 vs 17
        let mut logits_pre = vec![-2.0f32; e_cnt];
        logits_pre[3] = 0.5;
        logits_pre[17] = 0.5;
        logits_pre[9] = 1.5;
        logits_pre[1] = 1.2;
        logits_pre[20] = 0.9;
        logits_pre[30] = 0.7;
        let bias_zero = vec![0.0f32; e_cnt];
        let d_lp = to_dev_f32(&logits_pre);
        let d_bz = to_dev_f32(&bias_zero);
        let d_pids = DeviceBuffer::alloc(0, topk * 4).unwrap();
        let d_pwts = DeviceBuffer::alloc(0, topk * 4).unwrap();
        unsafe {
            mm3::router_f32(
                std::ptr::null(),
                std::ptr::null(),
                d_lp.ptr as *mut f32,
                d_bz.ptr as *const f32,
                1,
                d_dim as i32,
                e_cnt as i32,
                topk as i32,
                1.0,
                d_pids.ptr as *mut i32,
                d_pwts.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        assert_eq!(read_dev_i32(&d_pids, topk), vec![9, 1, 20, 30, 3, 17]);
    }
}

// ============================================================================
// Test 3: DSpark Custom BF16 MMA, MXFP8, Window Attention, and Markov Greedy
// ============================================================================

#[test]
fn test_mm3_dspark_bf16_mma_mxfp8_window_attn_and_markov() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x3331_0003_9876_5432);

    // 1) Custom BF16 MMA across all 4 dispatch regimes + strides + alpha/beta
    let gemm_cases = [
        (17usize, 53usize, 77usize, 80usize, 83usize, 61usize, 1.25f32, 0.0f32),
        (19, 55, 79, 79, 81, 57, 0.8, 0.6),
        (64, 96, 128, 128, 128, 96, 1.0, 0.0),
        (70, 140, 96, 104, 100, 144, 1.0, 0.5),
        (135, 150, 160, 168, 160, 152, 1.0, 0.25),
    ];
    for &(m, n, k, ld_a, ld_b, ld_c, alpha, beta) in &gemm_cases {
        let a_bf16: Vec<u16> = (0..m * ld_a)
            .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
            .collect();
        let b_bf16: Vec<u16> = (0..n * ld_b)
            .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
            .collect();
        let c_init: Vec<f32> = (0..m * ld_c).map(|_| rng.next_f32()).collect();

        let d_a = to_dev_u16(&a_bf16);
        let d_b = to_dev_u16(&b_bf16);
        let d_c = to_dev_f32(&c_init);
        unsafe {
            mm3::bf16_gemm_mma(
                d_a.ptr as *const u16,
                ld_a as i32,
                d_b.ptr as *const u16,
                ld_b as i32,
                d_c.ptr as *mut f32,
                ld_c as i32,
                m as i32,
                n as i32,
                k as i32,
                alpha,
                beta,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got_full = read_dev_f32(&d_c, m * ld_c);

        let mut got = Vec::with_capacity(m * n);
        let mut exp = Vec::with_capacity(m * n);
        for mi in 0..m {
            for ni in 0..n {
                let mut dot = 0.0f32;
                for ki in 0..k {
                    let av = bf16::from_bits(a_bf16[mi * ld_a + ki]).to_f32();
                    let bv = bf16::from_bits(b_bf16[ni * ld_b + ki]).to_f32();
                    dot += av * bv;
                }
                let e = alpha * dot + if beta == 0.0 { 0.0 } else { beta * c_init[mi * ld_c + ni] };
                got.push(got_full[mi * ld_c + ni]);
                exp.push(e);
            }
            for pad in n..ld_c {
                assert_eq!(got_full[mi * ld_c + pad], c_init[mi * ld_c + pad]);
            }
        }
        let re = rel_err(&got, &exp);
        let ma = max_abs_err(&got, &exp);
        println!("bf16_gemm_mma {m}x{n}x{k} ld=({ld_a},{ld_b},{ld_c}) b={beta}: rel={re:.7} max={ma:.6}");
        assert!(re < 1e-4 && ma < 2e-3);
    }

    // 2) MXFP8 dequant + linear (including E4M3 subnormals & UE8M0 e=0, split-K K=2048, unaligned ld, BF16 scratch)
    {
        let n = 96usize;
        let k = 2048usize;
        let ks = k / 32;
        let mut w_fp8 = vec![0u8; n * k];
        for (idx, b) in w_fp8.iter_mut().enumerate() {
            if idx % 29 == 0 {
                *b = [0x01u8, 0x04, 0x07, 0x81, 0x85][idx % 5]; // E4M3 subnormals
            } else {
                let mut v = rng.next_u8();
                if (v & 0x7F) == 0x7F {
                    v &= 0xFE; // keep finite
                }
                *b = v;
            }
        }
        let mut sc_ue8m0 = vec![0u8; n * ks];
        for (idx, s) in sc_ue8m0.iter_mut().enumerate() {
            *s = if idx % 17 == 0 {
                0u8 // e = 0 -> 2^-127
            } else {
                122u8 + (rng.next_u8() % 10)
            };
        }

        let mut w_ref = vec![0.0f32; n * k];
        for ni in 0..n {
            for ki in 0..k {
                let sc = decode_ue8m0(sc_ue8m0[ni * ks + ki / 32]);
                w_ref[ni * k + ki] = decode_fp8_e4m3_safe(w_fp8[ni * k + ki]) * sc;
            }
        }

        let d_w = to_dev_u8(&w_fp8);
        let d_sc = to_dev_u8(&sc_ue8m0);
        let d_w_bf16 = DeviceBuffer::alloc(0, n * k * 2).unwrap();
        unsafe {
            mm3::mxfp8_to_bf16(d_w.ptr as *const u8, d_sc.ptr as *const u8, n as i32, k as i32, d_w_bf16.ptr as *mut u16, &stream);
        }
        stream.sync().unwrap();
        let got_bf16_u16 = read_dev_u16(&d_w_bf16, n * k);
        for (i, &gb) in got_bf16_u16.iter().enumerate() {
            let eb = bf16::from_f32(w_ref[i]).to_bits();
            assert_eq!(gb, eb, "mxfp8_to_bf16 mismatch at {i}: got {gb:#x} exp {eb:#x}");
        }

        for &(rows, ld_x, ld_y, beta, use_scratch) in &[
            (1usize, k, n, 0.0f32, false),
            (7usize, k + 3, n + 3, 0.5f32, false),
            (16usize, k, n, 0.0f32, false),
            (24usize, k, n, 0.25f32, false),
            (24usize, k + 4, n + 2, 0.5f32, true),
        ] {
            let x: Vec<f32> = (0..rows * ld_x).map(|_| rng.next_f32()).collect();
            let y_init: Vec<f32> = (0..rows * ld_y).map(|_| rng.next_f32()).collect();
            let d_x = to_dev_f32(&x);
            let d_y = to_dev_f32(&y_init);
            let d_part = DeviceBuffer::alloc(0, 4 * rows * n * 4).unwrap();
            let d_x_bf16 = DeviceBuffer::alloc(0, rows * k * 2).unwrap();

            unsafe {
                mm3::mxfp8_linear(
                    d_w.ptr as *const u8,
                    d_sc.ptr as *const u8,
                    n as i32,
                    k as i32,
                    d_x.ptr as *const f32,
                    ld_x as i32,
                    rows as i32,
                    d_y.ptr as *mut f32,
                    ld_y as i32,
                    beta,
                    d_part.ptr as *mut f32,
                    if use_scratch { d_w_bf16.ptr as *mut u16 } else { std::ptr::null_mut() },
                    if use_scratch { d_x_bf16.ptr as *mut u16 } else { std::ptr::null_mut() },
                    &stream,
                );
            }
            stream.sync().unwrap();
            let got_full = read_dev_f32(&d_y, rows * ld_y);
            let mut got = Vec::with_capacity(rows * n);
            let mut exp = Vec::with_capacity(rows * n);
            for r in 0..rows {
                for ni in 0..n {
                    let dot: f32 = (0..k).map(|ki| x[r * ld_x + ki] * w_ref[ni * k + ki]).sum();
                    let e = dot + if beta == 0.0 { 0.0 } else { beta * y_init[r * ld_y + ni] };
                    got.push(got_full[r * ld_y + ni]);
                    exp.push(e);
                }
            }
            let re = rel_err(&got, &exp);
            println!("mxfp8_linear rows={rows} ld=({ld_x},{ld_y}) beta={beta} scratch={use_scratch}: rel={re:.6}");
            assert!(re < if use_scratch { 5e-3 } else { 1e-4 }, "mxfp8 rel={re}");
        }
    }

    // 3) DSpark sliding-window BF16 attention with ring wraparound
    {
        let t_cnt = 8usize;
        let hq = 16usize;
        let hkv = 2usize;
        let hd = 128usize;
        let ring = 32usize;
        let hi = 45i32;
        let lo = vec![30i32, 31, 32, 33, 38, 39, 44, 45];
        let scale = 1.0f32 / (hd as f32).sqrt();

        let q: Vec<f32> = (0..t_cnt * hq * hd).map(|_| rng.next_f32()).collect();
        let kc_bf16: Vec<u16> = (0..ring * hkv * hd)
            .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
            .collect();
        let vc_bf16: Vec<u16> = (0..ring * hkv * hd)
            .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
            .collect();

        let d_q = to_dev_f32(&q);
        let d_kc = to_dev_u16(&kc_bf16);
        let d_vc = to_dev_u16(&vc_bf16);
        let d_lo = to_dev_i32(&lo);
        let d_out1 = DeviceBuffer::alloc(0, t_cnt * hq * hd * 4).unwrap();
        let d_out4 = DeviceBuffer::alloc(0, t_cnt * hq * hd * 4).unwrap();
        let d_part = DeviceBuffer::alloc(
            0,
            t_cnt * hkv * 4 * (hq / hkv) * mm3::ATTN_PART_STRIDE * 4,
        )
        .unwrap();

        unsafe {
            mm3::dspark_window_attn(
                d_q.ptr as *const f32,
                d_kc.ptr as *const u16,
                d_vc.ptr as *const u16,
                ring as i32,
                d_lo.ptr as *const i32,
                hi,
                t_cnt as i32,
                hq as i32,
                hkv as i32,
                scale,
                std::ptr::null_mut(),
                1,
                d_out1.ptr as *mut f32,
                &stream,
            );
            mm3::dspark_window_attn(
                d_q.ptr as *const f32,
                d_kc.ptr as *const u16,
                d_vc.ptr as *const u16,
                ring as i32,
                d_lo.ptr as *const i32,
                hi,
                t_cnt as i32,
                hq as i32,
                hkv as i32,
                scale,
                d_part.ptr as *mut f32,
                4,
                d_out4.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let out1 = read_dev_f32(&d_out1, t_cnt * hq * hd);
        let out4 = read_dev_f32(&d_out4, t_cnt * hq * hd);
        assert!(max_abs_err(&out1, &out4) < 1e-5);

        let mut exp = vec![0.0f32; t_cnt * hq * hd];
        let gqa = hq / hkv;
        for t in 0..t_cnt {
            for g in 0..hkv {
                for gh in 0..gqa {
                    let h = g * gqa + gh;
                    let qr = &q[(t * hq + h) * hd..(t * hq + h + 1) * hd];
                    let scores: Vec<f32> = (lo[t]..=hi)
                        .map(|kp| {
                            let row = (kp as usize) % ring;
                            let mut dot = 0.0f32;
                            for d in 0..hd {
                                dot += qr[d] * bf16::from_bits(kc_bf16[(row * hkv + g) * hd + d]).to_f32();
                            }
                            dot * scale
                        })
                        .collect();
                    let m_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut exps = Vec::with_capacity(scores.len());
                    let mut sum_e = 0.0f32;
                    for &s in &scores {
                        let e = (s - m_max).exp();
                        exps.push(e);
                        sum_e += e;
                    }
                    for d in 0..hd {
                        let mut acc = 0.0f32;
                        for (j, kp) in (lo[t]..=hi).enumerate() {
                            let row = (kp as usize) % ring;
                            acc += (exps[j] / sum_e) * bf16::from_bits(vc_bf16[(row * hkv + g) * hd + d]).to_f32();
                        }
                        exp[(t * hq + h) * hd + d] = acc;
                    }
                }
            }
        }
        let ma = max_abs_err(&out1, &exp);
        println!("dspark_window_attn max_abs={ma:.7}");
        assert!(ma < 1e-4, "dspark_window_attn max_abs={ma}");
    }

    // 4) DSpark rank-256 Markov greedy + bias + lower-ID tie-break + confidence
    {
        let v_cnt = 4096usize;
        let r_dim = 256usize;
        let d_dim = 128usize;
        let init_prev = 137i32;

        let mut w1 = vec![0u16; v_cnt * r_dim];
        let mut w2 = vec![0u16; v_cnt * r_dim];
        for x in &mut w1 {
            *x = bf16::from_f32(rng.next_f32_range(-0.3, 0.3)).to_bits();
        }
        for x in &mut w2 {
            *x = bf16::from_f32(rng.next_f32_range(-0.3, 0.3)).to_bits();
        }
        // Engineer exact tie at token 512 & 2048 so 512 must beat 2048 (and override raw logits!)
        for k in 0..r_dim {
            w2[512 * r_dim + k] = bf16::from_f32(1.0).to_bits();
            w2[2048 * r_dim + k] = bf16::from_f32(1.0).to_bits();
            w1[init_prev as usize * r_dim + k] = bf16::from_f32(0.05).to_bits();
        }
        let mut logits: Vec<f32> = (0..v_cnt).map(|_| rng.next_f32_range(-2.0, 2.0)).collect();
        logits[10] = 3.0; // raw argmax would be 10, but bias on 512/2048 is +12.8
        logits[512] = 0.0;
        logits[2048] = 0.0;

        let cw: Vec<f32> = (0..d_dim + r_dim).map(|_| rng.next_f32_range(-0.2, 0.2)).collect();
        let cb = vec![0.125f32];
        let hidden: Vec<f32> = (0..d_dim).map(|_| rng.next_f32()).collect();

        let d_logits = to_dev_f32(&logits);
        let d_w2 = to_dev_u16(&w2);
        let d_w1 = to_dev_u16(&w1);
        let d_prev = to_dev_i32(&[init_prev]);
        let d_out = DeviceBuffer::alloc_zeroed(0, 4).unwrap();
        let d_cw = to_dev_f32(&cw);
        let d_cb = to_dev_f32(&cb);
        let d_hid = to_dev_f32(&hidden);
        let d_conf = DeviceBuffer::alloc_zeroed(0, 4).unwrap();
        let d_part = DeviceBuffer::alloc_zeroed(0, mm3::MARKOV_PART_FLOATS * 4).unwrap();
        let d_bias = DeviceBuffer::alloc_zeroed(0, v_cnt * 4).unwrap();

        unsafe {
            mm3::dspark_markov_greedy(
                d_logits.ptr as *const f32,
                d_w2.ptr as *const u16,
                d_w1.ptr as *const u16,
                v_cnt as i32,
                r_dim as i32,
                d_prev.ptr as *mut i32,
                d_out.ptr as *mut i32,
                d_cw.ptr as *const f32,
                d_cb.ptr as *const f32,
                d_hid.ptr as *const f32,
                d_dim as i32,
                d_conf.ptr as *mut f32,
                d_part.ptr as *mut f32,
                d_bias.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();

        assert_eq!(read_dev_i32(&d_out, 1)[0], 512);
        assert_eq!(read_dev_i32(&d_prev, 1)[0], 512);

        let got_bias = read_dev_f32(&d_bias, v_cnt);
        let u_old: Vec<f32> = w1[init_prev as usize * r_dim..(init_prev as usize + 1) * r_dim]
            .iter()
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect();
        let mut exp_bias = vec![0.0f32; v_cnt];
        for v in 0..v_cnt {
            exp_bias[v] = (0..r_dim)
                .map(|k| bf16::from_bits(w2[v * r_dim + k]).to_f32() * u_old[k])
                .sum();
        }
        assert!(max_abs_err(&got_bias, &exp_bias) < 5e-3);

        let got_conf = read_dev_f32(&d_conf, 1)[0];
        let mut s_conf = cb[0];
        for i in 0..d_dim + r_dim {
            let feat = if i < d_dim { hidden[i] } else { u_old[i - d_dim] };
            s_conf += cw[i] * feat;
        }
        let exp_conf = 1.0f32 / (1.0f32 + (-s_conf).exp());
        println!("dspark_markov_greedy: conf got={got_conf:.7} exp={exp_conf:.7}");
        assert!((got_conf - exp_conf).abs() < 1e-5);
    }
}

// ============================================================================
// Test 4: Vision Tower Kernels & BF16 MMA Vision Shapes
// ============================================================================

#[test]
fn test_mm3_vision_kernels() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x3331_0004_1122_3344);

    // 1) vision_patch_embed_f32
    {
        let (n, d, p) = (37usize, 96usize, 588usize);
        let px: Vec<f32> = (0..n * p).map(|_| rng.next_f32()).collect();
        let wp: Vec<f32> = (0..d * p).map(|_| rng.next_f32() * 0.25).collect();
        let d_px = to_dev_f32(&px);
        let d_wp = to_dev_f32(&wp);
        let d_out = DeviceBuffer::alloc(0, n * d * 4).unwrap();
        unsafe {
            mm3::vision_patch_embed_f32(
                d_px.ptr as *const f32,
                d_wp.ptr as *const f32,
                n as i32,
                d as i32,
                p as i32,
                d_out.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got = read_dev_f32(&d_out, n * d);
        let mut exp = vec![0.0f32; n * d];
        for ni in 0..n {
            for di in 0..d {
                exp[ni * d + di] = px[ni * p..(ni + 1) * p]
                    .iter()
                    .zip(wp[di * p..(di + 1) * p].iter())
                    .map(|(&a, &b)| a * b)
                    .sum();
            }
        }
        assert!(max_abs_err(&got, &exp) < 1e-4);
    }

    // 2) vision_layernorm (F32 & BF16)
    {
        let (rows, dim, eps) = (16usize, 1280usize, 1e-6f32);
        let x: Vec<f32> = (0..rows * dim).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..dim).map(|_| rng.next_f32_range(0.5, 1.5)).collect();
        let b: Vec<f32> = (0..dim).map(|_| rng.next_f32_range(-0.3, 0.3)).collect();
        let d_x = to_dev_f32(&x);
        let d_w = to_dev_f32(&w);
        let d_b = to_dev_f32(&b);
        let d_yf = DeviceBuffer::alloc(0, rows * dim * 4).unwrap();
        let d_yb = DeviceBuffer::alloc(0, rows * dim * 2).unwrap();
        unsafe {
            mm3::vision_layernorm(
                d_x.ptr as *const f32,
                d_w.ptr as *const f32,
                d_b.ptr as *const f32,
                rows as i32,
                dim as i32,
                eps,
                d_yf.ptr as *mut f32,
                d_yb.ptr as *mut u16,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut exp = vec![0.0f32; rows * dim];
        for r in 0..rows {
            let xr = &x[r * dim..(r + 1) * dim];
            let mean = xr.iter().sum::<f32>() / (dim as f32);
            let var = xr.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / (dim as f32);
            let inv = 1.0f32 / (var + eps).sqrt();
            for i in 0..dim {
                exp[r * dim + i] = (xr[i] - mean) * inv * w[i] + b[i];
            }
        }
        let got_f = read_dev_f32(&d_yf, rows * dim);
        let got_b: Vec<f32> = read_dev_u16(&d_yb, rows * dim)
            .into_iter()
            .map(|u| bf16::from_bits(u).to_f32())
            .collect();
        assert!(max_abs_err(&got_f, &exp) < 1e-5);
        assert!(max_abs_err(&got_b, &exp) < 2e-2);
    }

    // 3) vision_qkv_rope_3d (dims 0..78 rotated, dims 78..80 untouched, + bias)
    {
        let (n, dim, heads, hd, nf) = (16usize, 1280usize, 16usize, 80usize, 13usize);
        let qkv: Vec<f32> = (0..n * 3 * dim).map(|_| rng.next_f32()).collect();
        let bias: Vec<f32> = (0..3 * dim).map(|_| rng.next_f32() * 0.25).collect();
        let pos3d: Vec<i32> = (0..n)
            .flat_map(|t| [(t / 4) as i32, ((t / 2) % 2) as i32, (t % 2) as i32])
            .collect();
        let inv_freq: Vec<f32> = (0..nf)
            .map(|i| 10000.0f32.powf(-((2 * i) as f32) / (2 * nf) as f32))
            .collect();

        let d_qkv = to_dev_f32(&qkv);
        let d_bias = to_dev_f32(&bias);
        let d_pos3d = to_dev_i32(&pos3d);
        let d_inv = to_dev_f32(&inv_freq);
        unsafe {
            mm3::vision_qkv_rope_3d(
                d_qkv.ptr as *mut f32,
                d_bias.ptr as *const f32,
                d_pos3d.ptr as *const i32,
                d_inv.ptr as *const f32,
                n as i32,
                dim as i32,
                heads as i32,
                hd as i32,
                nf as i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got = read_dev_f32(&d_qkv, n * 3 * dim);
        let mut exp = qkv.clone();
        let half = 3 * nf; // 39 -> pairs 0..78
        for t in 0..n {
            for j in 0..3 * dim {
                exp[t * 3 * dim + j] += bias[j];
            }
            for which in 0..2 {
                for hh in 0..heads {
                    let base = t * 3 * dim + which * dim + hh * hd;
                    let orig = exp[base..base + hd].to_vec();
                    for i in 0..half {
                        let ang = (pos3d[t * 3 + i / nf] as f32) * inv_freq[i % nf];
                        let (sn, cs) = ang.sin_cos();
                        exp[base + i] = orig[i] * cs - orig[i + half] * sn;
                        exp[base + i + half] = orig[i + half] * cs + orig[i] * sn;
                    }
                }
            }
        }
        assert!(max_abs_err(&got, &exp) < 1e-5);
    }

    // 4) vision_full_attn (N=400, heads=16, hd=80)
    {
        let (n, dim, heads, hd) = (400usize, 1280usize, 16usize, 80usize);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let qkv: Vec<f32> = (0..n * 3 * dim).map(|_| rng.next_f32() * 0.7).collect();
        let d_qkv = to_dev_f32(&qkv);
        let d_out = DeviceBuffer::alloc(0, n * dim * 4).unwrap();
        unsafe {
            mm3::vision_full_attn(
                d_qkv.ptr as *const f32,
                n as i32,
                dim as i32,
                heads as i32,
                hd as i32,
                scale,
                d_out.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got = read_dev_f32(&d_out, n * dim);
        let mut exp = vec![0.0f32; n * dim];
        for h in 0..heads {
            for t in 0..n {
                let qr = &qkv[t * 3 * dim + h * hd..t * 3 * dim + (h + 1) * hd];
                let mut scores = vec![0.0f32; n];
                let mut m_max = f32::NEG_INFINITY;
                for k in 0..n {
                    let kr = &qkv[k * 3 * dim + dim + h * hd..k * 3 * dim + dim + (h + 1) * hd];
                    let s: f32 = qr.iter().zip(kr.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                    scores[k] = s;
                    m_max = m_max.max(s);
                }
                let mut sum_e = 0.0f32;
                for s in &mut scores {
                    *s = (*s - m_max).exp();
                    sum_e += *s;
                }
                let inv = 1.0f32 / sum_e;
                for k in 0..n {
                    let p = scores[k] * inv;
                    let vr = &qkv[k * 3 * dim + 2 * dim + h * hd..k * 3 * dim + 2 * dim + (h + 1) * hd];
                    for d in 0..hd {
                        exp[t * dim + h * hd + d] += p * vr[d];
                    }
                }
            }
        }
        assert!(max_abs_err(&got, &exp) < 5e-5);
    }

    // 5) vision_bias_gelu + vision_residual_bias
    {
        let (rows, m) = (17usize, 320usize);
        let n = rows * m;
        let y: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0).collect();
        let bias: Vec<f32> = (0..m).map(|_| rng.next_f32() * 0.4).collect();
        let mut h: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let t: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();

        let d_y = to_dev_f32(&y);
        let d_b = to_dev_f32(&bias);
        let d_yf = DeviceBuffer::alloc(0, n * 4).unwrap();
        let d_yb = DeviceBuffer::alloc(0, n * 2).unwrap();
        let d_h = to_dev_f32(&h);
        let d_t = to_dev_f32(&t);

        for &apply_gelu in &[false, true] {
            unsafe {
                mm3::vision_bias_gelu(
                    d_y.ptr as *const f32,
                    d_b.ptr as *const f32,
                    n,
                    m as i32,
                    apply_gelu,
                    d_yf.ptr as *mut f32,
                    d_yb.ptr as *mut u16,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let got_f = read_dev_f32(&d_yf, n);
            let got_b: Vec<f32> = read_dev_u16(&d_yb, n)
                .into_iter()
                .map(|u| bf16::from_bits(u).to_f32())
                .collect();
            let exp: Vec<f32> = (0..n)
                .map(|i| {
                    let v = y[i] + bias[i % m];
                    if apply_gelu {
                        0.5f32 * v * (1.0f32 + unsafe { erff(v * 0.7071067811865475f32) })
                    } else {
                        v
                    }
                })
                .collect();
            assert!(max_abs_err(&got_f, &exp) < 1e-5);
            assert!(max_abs_err(&got_b, &exp) < 3e-2);
        }

        unsafe {
            mm3::vision_residual_bias(
                d_h.ptr as *mut f32,
                d_t.ptr as *const f32,
                d_b.ptr as *const f32,
                n,
                m as i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got_h = read_dev_f32(&d_h, n);
        for i in 0..n {
            h[i] += t[i] + bias[i % m];
        }
        assert!(max_abs_err(&got_h, &h) < 1e-5);
    }

    // 6) Vision MLP / QKV BF16 MMA shapes (196x3840x1280 & 196x1280x5120)
    for &(m, n, k) in &[(196usize, 3840usize, 1280usize), (196usize, 1280usize, 5120usize)] {
        let a_bf16: Vec<u16> = (0..m * k)
            .map(|_| bf16::from_f32(rng.next_f32_range(-0.5, 0.5)).to_bits())
            .collect();
        let b_bf16: Vec<u16> = (0..n * k)
            .map(|_| bf16::from_f32(rng.next_f32_range(-0.5, 0.5)).to_bits())
            .collect();
        let d_a = to_dev_u16(&a_bf16);
        let d_b = to_dev_u16(&b_bf16);
        let d_c = DeviceBuffer::alloc(0, m * n * 4).unwrap();
        unsafe {
            mm3::bf16_gemm_mma(
                d_a.ptr as *const u16,
                k as i32,
                d_b.ptr as *const u16,
                k as i32,
                d_c.ptr as *mut f32,
                n as i32,
                m as i32,
                n as i32,
                k as i32,
                1.0,
                0.0,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got = read_dev_f32(&d_c, m * n);
        let a_f32: Vec<f32> = a_bf16.iter().map(|&u| bf16::from_bits(u).to_f32()).collect();
        let b_f32: Vec<f32> = b_bf16.iter().map(|&u| bf16::from_bits(u).to_f32()).collect();
        let mut exp = vec![0.0f32; m * n];
        for mi in 0..m {
            let ar = &a_f32[mi * k..(mi + 1) * k];
            for ni in 0..n {
                let br = &b_f32[ni * k..(ni + 1) * k];
                exp[mi * n + ni] = ar.iter().zip(br.iter()).map(|(&a, &b)| a * b).sum();
            }
        }
        let re = rel_err(&got, &exp);
        println!("vision bf16_gemm_mma {m}x{n}x{k}: rel_err={re:.7}");
        assert!(re < 1e-4, "vision bf16_gemm_mma {m}x{n}x{k} rel_err={re}");
    }
}

#[test]
fn test_mm3_grouped_decode_moe_and_tiled_msa_parity() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x3331_0004_1234_5678);
    let pool = CpuThreadPool::new(Some(4));
    let n_embd = 512usize;
    let n_ff = 512usize;
    let num_exp = 6usize;
    let topk = 4usize;
    let mut cpu_scratch = Mm3CpuScratch::new(n_embd, n_ff, 6 * topk);

    // 1) Grouped decode/verify MoE kernels (plan0 + plan1 + cpu_out) vs CPU & per-slot GPU baseline
    let configs = [
        (GgmlType::Iq2Xxs, GgmlType::Iq2Xxs, GgmlType::Iq2Xxs),
        (GgmlType::Iq2Xs, GgmlType::Iq2Xxs, GgmlType::Iq3Xxs),
        (GgmlType::Iq1S, GgmlType::Iq1S, GgmlType::Iq1S),
    ];

    for &(g_ty, u_ty, d_ty) in &configs {
        let g_rb = g_ty.row_bytes(n_embd);
        let u_rb = u_ty.row_bytes(n_embd);
        let d_rb = d_ty.row_bytes(n_ff);
        let u_off = (n_ff * g_rb + 255) & !255;
        let d_off = (u_off + n_ff * u_rb + 255) & !255;
        let blob_bytes = (d_off + n_embd * d_rb + 255) & !255;

        let mut host_blobs = Vec::with_capacity(num_exp);
        let mut dev_blobs = Vec::with_capacity(num_exp);
        for _ in 0..num_exp {
            let (gb, _) = fill_mm3_quant_matrix(g_ty, n_ff, n_embd, &mut rng);
            let (ub, _) = fill_mm3_quant_matrix(u_ty, n_ff, n_embd, &mut rng);
            let (db, _) = fill_mm3_quant_matrix(d_ty, n_embd, n_ff, &mut rng);
            let mut blob = vec![0u8; blob_bytes];
            blob[..gb.len()].copy_from_slice(&gb);
            blob[u_off..u_off + ub.len()].copy_from_slice(&ub);
            blob[d_off..d_off + db.len()].copy_from_slice(&db);
            dev_blobs.push(to_dev_u8(&blob));
            host_blobs.push(blob);
        }

        for &m in &[1usize, 6usize] {
            let x_f32: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32()).collect();
            let res: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32() * 0.5).collect();
            let shared: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32() * 0.4).collect();
            // Force expert 0,1,2 in plan0 (VRAM), expert 3,4 in plan1 (staged), expert 5 in cpu_out!
            let mut expert_ids = vec![0i32; m * topk];
            let mut weights = vec![0.0f32; m * topk];
            for mi in 0..m {
                let chosen = [
                    (mi % 3) as i32,
                    ((mi + 1) % 3) as i32,
                    (3 + (mi % 2)) as i32,
                    5i32,
                ];
                let mut sum_w = 0.0f32;
                for s in 0..topk {
                    expert_ids[mi * topk + s] = chosen[s];
                    let w = 0.2 + 0.05 * (s as f32);
                    weights[mi * topk + s] = w;
                    sum_w += w;
                }
                for s in 0..topk {
                    weights[mi * topk + s] /= sum_w;
                }
            }

            let build_plan = |target_eids: &[usize]| -> (mm3::Mm3DecodePlanC, usize, usize, usize) {
                let mut plan = mm3::Mm3DecodePlanC::default();
                let mut g = 0usize;
                let mut p = 0usize;
                let mut max_nt = 0usize;
                for &eu in target_eids {
                    let p0 = p;
                    for mi in 0..m {
                        for s in 0..topk {
                            if expert_ids[mi * topk + s] as usize == eu {
                                plan.entry_tok[p] = mi as i32;
                                plan.entry_wt[p] = weights[mi * topk + s];
                                p += 1;
                            }
                        }
                    }
                    if p > p0 {
                        plan.base_ptrs[g] = dev_blobs[eu].as_u64();
                        plan.grp_off[g] = p0 as i32;
                        max_nt = max_nt.max(p - p0);
                        g += 1;
                    }
                }
                plan.grp_off[g] = p as i32;
                (plan, g, p, max_nt)
            };

            let (plan0, ng0, ne0, max_nt0) = build_plan(&[0, 1, 2]);
            let (plan1, ng1, ne1, max_nt1) = build_plan(&[3, 4]);

            // Compute CPU reference for ALL experts combined
            let mut per_exp_toks: Vec<Vec<usize>> = vec![Vec::new(); num_exp];
            let mut per_exp_wts: Vec<Vec<f32>> = vec![Vec::new(); num_exp];
            for mi in 0..m {
                for s in 0..topk {
                    let e = expert_ids[mi * topk + s] as usize;
                    per_exp_toks[e].push(mi);
                    per_exp_wts[e].push(weights[mi * topk + s]);
                }
            }
            let cpu_only_groups = [Mm3CpuGroupedExpert {
                gate_ptr: host_blobs[5].as_ptr(),
                up_ptr: unsafe { host_blobs[5].as_ptr().add(u_off) },
                down_ptr: unsafe { host_blobs[5].as_ptr().add(d_off) },
                token_indices: &per_exp_toks[5],
                token_weights: &per_exp_wts[5],
            }];
            let mut cpu_only_out = vec![0.0f32; m * n_embd];
            run_mm3_cpu_moe_grouped(
                &pool,
                &mut cpu_scratch,
                g_ty,
                u_ty,
                d_ty,
                n_embd,
                n_ff,
                1.702,
                7.0,
                m,
                &x_f32,
                &cpu_only_groups,
                &mut cpu_only_out,
                true,
            );

            let all_groups: Vec<Mm3CpuGroupedExpert<'_>> = (0..num_exp)
                .filter(|&e| !per_exp_toks[e].is_empty())
                .map(|e| Mm3CpuGroupedExpert {
                    gate_ptr: host_blobs[e].as_ptr(),
                    up_ptr: unsafe { host_blobs[e].as_ptr().add(u_off) },
                    down_ptr: unsafe { host_blobs[e].as_ptr().add(d_off) },
                    token_indices: &per_exp_toks[e],
                    token_weights: &per_exp_wts[e],
                })
                .collect();
            let mut cpu_all = vec![0.0f32; m * n_embd];
            run_mm3_cpu_moe_grouped(
                &pool,
                &mut cpu_scratch,
                g_ty,
                u_ty,
                d_ty,
                n_embd,
                n_ff,
                1.702,
                7.0,
                m,
                &x_f32,
                &all_groups,
                &mut cpu_all,
                true,
            );
            let cpu_ref: Vec<f32> = cpu_all
                .iter()
                .zip(res.iter().zip(shared.iter()))
                .map(|(&c, (&r, &s))| c + r + s)
                .collect();

            let d_x = to_dev_f32(&x_f32);
            let d_res = to_dev_f32(&res);
            let d_shared = to_dev_f32(&shared);
            let d_cpu_part = to_dev_f32(&cpu_only_out);
            let d_plan0 = to_dev_u8(unsafe {
                std::slice::from_raw_parts(
                    (&plan0 as *const mm3::Mm3DecodePlanC) as *const u8,
                    std::mem::size_of::<mm3::Mm3DecodePlanC>(),
                )
            });
            let d_plan1 = to_dev_u8(unsafe {
                std::slice::from_raw_parts(
                    (&plan1 as *const mm3::Mm3DecodePlanC) as *const u8,
                    std::mem::size_of::<mm3::Mm3DecodePlanC>(),
                )
            });

            let tot_e = ne0 + ne1;
            let d_x_q8k = DeviceBuffer::alloc(0, m * (n_embd / 256) * mm3::Q8K_BLOCK_BYTES).unwrap();
            let d_inter_f32 = DeviceBuffer::alloc(0, tot_e * n_ff * 4).unwrap();
            let d_inter_q8k =
                DeviceBuffer::alloc(0, tot_e * (n_ff / 256) * mm3::Q8K_BLOCK_BYTES).unwrap();
            let d_exp_out = DeviceBuffer::alloc(0, tot_e * n_embd * 4).unwrap();
            let d_out = DeviceBuffer::alloc(0, m * n_embd * 4).unwrap();

            unsafe {
                mm3::quantize_q8_k(
                    d_x.ptr as *const f32,
                    d_x_q8k.ptr,
                    m as i32,
                    n_embd as i32,
                    &stream,
                );
                mm3::moe_grp_gate_up_f32(
                    g_ty as i32,
                    u_ty as i32,
                    d_plan0.ptr,
                    u_off,
                    g_rb,
                    u_rb,
                    d_x_q8k.ptr,
                    d_inter_f32.ptr as *mut f32,
                    n_embd as i32,
                    n_ff as i32,
                    ng0 as i32,
                    max_nt0 as i32,
                    1.702,
                    7.0,
                    &stream,
                );
                mm3::quantize_q8_k(
                    d_inter_f32.ptr as *const f32,
                    d_inter_q8k.ptr,
                    ne0 as i32,
                    n_ff as i32,
                    &stream,
                );
                mm3::moe_grp_down_q8k(
                    d_ty as i32,
                    d_plan0.ptr,
                    d_off,
                    d_rb,
                    d_inter_q8k.ptr,
                    d_exp_out.ptr as *mut f32,
                    n_ff as i32,
                    n_embd as i32,
                    ng0 as i32,
                    max_nt0 as i32,
                    &stream,
                );

                mm3::moe_grp_gate_up_f32(
                    g_ty as i32,
                    u_ty as i32,
                    d_plan1.ptr,
                    u_off,
                    g_rb,
                    u_rb,
                    d_x_q8k.ptr,
                    (d_inter_f32.ptr as *mut f32).add(ne0 * n_ff),
                    n_embd as i32,
                    n_ff as i32,
                    ng1 as i32,
                    max_nt1 as i32,
                    1.702,
                    7.0,
                    &stream,
                );
                mm3::quantize_q8_k(
                    (d_inter_f32.ptr as *const f32).add(ne0 * n_ff),
                    d_inter_q8k.ptr.add(ne0 * (n_ff / 256) * mm3::Q8K_BLOCK_BYTES),
                    ne1 as i32,
                    n_ff as i32,
                    &stream,
                );
                mm3::moe_grp_down_q8k(
                    d_ty as i32,
                    d_plan1.ptr,
                    d_off,
                    d_rb,
                    d_inter_q8k.ptr.add(ne0 * (n_ff / 256) * mm3::Q8K_BLOCK_BYTES),
                    (d_exp_out.ptr as *mut f32).add(ne0 * n_embd),
                    n_ff as i32,
                    n_embd as i32,
                    ng1 as i32,
                    max_nt1 as i32,
                    &stream,
                );

                mm3::moe_grp_combine(
                    d_res.ptr as *const f32,
                    d_shared.ptr as *const f32,
                    d_exp_out.ptr as *const f32,
                    d_plan0.ptr,
                    ne0 as i32,
                    d_plan1.ptr,
                    ne1 as i32,
                    d_cpu_part.ptr as *const f32,
                    m as i32,
                    n_embd as i32,
                    d_out.ptr as *mut f32,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let got = read_dev_f32(&d_out, m * n_embd);
            let re = rel_err(&got, &cpu_ref);
            let ma = max_abs_err(&got, &cpu_ref);
            println!("moe_grp ({g_ty:?},{u_ty:?},{d_ty:?}) M={m}: max_abs={ma:.6} rel_vs_cpu={re:.7}");
            assert!(re < 1e-4, "moe_grp rel_vs_cpu={re}");
        }
    }

    // 2) Tiled4 msa_block_scores (T >= 16, Hi == 4) vs CPU reference
    {
        let t_cnt = 67usize; // not divisible by 4 to test tail tile
        let hi = 4usize;
        let hd = 128usize;
        let blk = 16usize;
        let nblk = 32usize;
        let ctx = nblk * blk;
        let iq: Vec<f32> = (0..t_cnt * hi * hd).map(|_| rng.next_f32()).collect();
        let ic_f32: Vec<f32> = (0..ctx * hd)
            .map(|_| bf16::from_f32(rng.next_f32()).to_f32())
            .collect();
        let ic_bf16: Vec<u16> = ic_f32.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let pos: Vec<i32> = (0..t_cnt).map(|t| (ctx - t_cnt + t) as i32).collect();

        let d_iq = to_dev_f32(&iq);
        let d_ic = to_dev_u16(&ic_bf16);
        let d_pos = to_dev_i32(&pos);
        let d_bs = DeviceBuffer::alloc(0, hi * t_cnt * nblk * 4).unwrap();
        unsafe {
            mm3::msa_block_scores(
                d_iq.ptr as *const f32,
                d_ic.ptr as *const u16,
                d_pos.ptr as *const i32,
                t_cnt as i32,
                hi as i32,
                nblk as i32,
                blk as i32,
                d_bs.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got = read_dev_f32(&d_bs, hi * t_cnt * nblk);
        for h in 0..hi {
            for t in 0..t_cnt {
                let p = pos[t] as usize;
                let qr = &iq[(t * hi + h) * hd..(t * hi + h + 1) * hd];
                for b in 0..nblk {
                    let ks = b * blk;
                    let idx = (h * t_cnt + t) * nblk + b;
                    if ks > p {
                        assert_eq!(got[idx], f32::NEG_INFINITY);
                        continue;
                    }
                    let ke = (ks + blk).min(p + 1);
                    let mut best = f32::NEG_INFINITY;
                    for kp in ks..ke {
                        let kr = &ic_f32[kp * hd..(kp + 1) * hd];
                        let dot: f32 = qr.iter().zip(kr.iter()).map(|(&a, &b)| a * b).sum();
                        best = best.max(dot);
                    }
                    assert!(
                        (got[idx] - best).abs() < 1e-4,
                        "tiled4 bs h={h} t={t} b={b}: got={} exp={best}",
                        got[idx]
                    );
                }
            }
        }
        println!("msa_block_scores_tiled4 T={t_cnt} Hi={hi} nblk={nblk}: PASSED");
    }
}

#[test]
fn test_mm3_pass2_kernels_parity() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x3331_0005_ABCD_EF01);

    let fill_q6k_or_q8_0 = |ty: GgmlType, n: usize, k: usize, r: &mut Rng| -> Vec<u8> {
        let rb = ty.row_bytes(k);
        let bsz = ty.type_size();
        let nb = k / ty.block_size();
        let mut out = vec![0u8; n * rb];
        for ri in 0..n {
            for b in 0..nb {
                let blk = &mut out[ri * rb + b * bsz..ri * rb + (b + 1) * bsz];
                let d_bytes = f16::from_f32(r.next_f32_range(0.01, 0.05)).to_le_bytes();
                match ty {
                    GgmlType::Q8_0 => {
                        blk[0..2].copy_from_slice(&d_bytes);
                        for x in &mut blk[2..34] {
                            *x = r.next_u8();
                        }
                    }
                    GgmlType::Q6K => {
                        for x in &mut blk[0..208] {
                            *x = r.next_u8();
                        }
                        blk[208..210].copy_from_slice(&d_bytes);
                    }
                    _ => unreachable!(),
                }
            }
        }
        out
    };

    // 1) Coalesced multi-job Q6_K & Q8_0 decode/verify GEMV (M = 1, 2, 5, 7, 8) vs launch_linear_q8_1
    for &(ty, k, n0, n1, n2) in &[
        (GgmlType::Q6K, 1024usize, 512usize, 384usize, 256usize),
        (GgmlType::Q8_0, 1024usize, 1024usize, 256usize, 256usize),
    ] {
        let w0 = fill_q6k_or_q8_0(ty, n0, k, &mut rng);
        let w1 = fill_q6k_or_q8_0(ty, n1, k, &mut rng);
        let w2 = fill_q6k_or_q8_0(ty, n2, k, &mut rng);
        let d_w0 = to_dev_u8(&w0);
        let d_w1 = to_dev_u8(&w1);
        let d_w2 = to_dev_u8(&w2);

        let m_max = 8usize;
        let x: Vec<f32> = (0..m_max * k).map(|_| rng.next_f32()).collect();
        let d_x = to_dev_f32(&x);
        let d_xq8 = DeviceBuffer::alloc(0, m_max * (k / 32) * 36).unwrap();
        unsafe {
            launch_quantize_q8_1(d_x.ptr as *const f32, d_xq8.ptr, m_max as i32, k as i32, stream.raw);
        }
        stream.sync().unwrap();

        for &m in &[1usize, 2, 5, 7, 8] {
            let d_y0_got = DeviceBuffer::alloc(0, m * n0 * 4).unwrap();
            let d_y1_got = DeviceBuffer::alloc(0, m * n1 * 4).unwrap();
            let d_y2_got = DeviceBuffer::alloc(0, m * n2 * 4).unwrap();
            let d_y0_ref = DeviceBuffer::alloc(0, m * n0 * 4).unwrap();
            let d_y1_ref = DeviceBuffer::alloc(0, m * n1 * 4).unwrap();
            let d_y2_ref = DeviceBuffer::alloc(0, m * n2 * 4).unwrap();

            unsafe {
                if ty == GgmlType::Q6K {
                    mm3::gv_q6k_multi(
                        3,
                        d_w0.ptr, d_y0_got.ptr as *mut f32, n0 as i32,
                        d_w1.ptr, d_y1_got.ptr as *mut f32, n1 as i32,
                        d_w2.ptr, d_y2_got.ptr as *mut f32, n2 as i32,
                        d_xq8.ptr, k as i32, m as i32, &stream,
                    );
                } else {
                    mm3::gv_q8_0_multi(
                        3,
                        d_w0.ptr, d_y0_got.ptr as *mut f32, n0 as i32,
                        d_w1.ptr, d_y1_got.ptr as *mut f32, n1 as i32,
                        d_w2.ptr, d_y2_got.ptr as *mut f32, n2 as i32,
                        d_xq8.ptr, k as i32, m as i32, &stream,
                    );
                }
                launch_linear_q8_1(ty as i32, d_w0.ptr, d_xq8.ptr, std::ptr::null(), d_y0_ref.ptr as *mut f32, n0 as i32, k as i32, m as i32, stream.raw);
                launch_linear_q8_1(ty as i32, d_w1.ptr, d_xq8.ptr, std::ptr::null(), d_y1_ref.ptr as *mut f32, n1 as i32, k as i32, m as i32, stream.raw);
                launch_linear_q8_1(ty as i32, d_w2.ptr, d_xq8.ptr, std::ptr::null(), d_y2_ref.ptr as *mut f32, n2 as i32, k as i32, m as i32, stream.raw);
            }
            stream.sync().unwrap();

            for (ji, (dg, dr, nj)) in [(&d_y0_got, &d_y0_ref, n0), (&d_y1_got, &d_y1_ref, n1), (&d_y2_got, &d_y2_ref, n2)].iter().enumerate() {
                let got = read_dev_f32(dg, m * nj);
                let exp = read_dev_f32(dr, m * nj);
                let re = rel_err(&got, &exp);
                assert!(re < 1e-4, "gv {ty:?} M={m} job={ji}: rel={re}");
            }
        }
        println!("mm3_gv {ty:?} multi3 M=1,2,5,7,8 vs linear_q8_1: PASSED");
    }

    // 2) MXFP8 split-K mxc_kernel (rows = 1, 4, 8, K = 6144, N = 512, beta = 0.0 & 1.0)
    {
        let (n, k) = (512usize, 6144usize);
        let ks = k / 32;
        let w_fp8: Vec<u8> = (0..n * k)
            .map(|i| {
                let v = if i % 31 == 0 { [0x02u8, 0x05, 0x83][i % 3] } else { rng.next_u8() };
                if (v & 0x7F) == 0x7F { v & 0xFE } else { v }
            })
            .collect();
        let sc: Vec<u8> = (0..n * ks)
            .map(|i| if i % 23 == 0 { 0u8 } else { 122u8 + (rng.next_u8() % 8) })
            .collect();
        let mut w_ref = vec![0.0f32; n * k];
        for ni in 0..n {
            for ki in 0..k {
                w_ref[ni * k + ki] = decode_fp8_e4m3_safe(w_fp8[ni * k + ki]) * decode_ue8m0(sc[ni * ks + ki / 32]);
            }
        }
        let d_w = to_dev_u8(&w_fp8);
        let d_s = to_dev_u8(&sc);
        let d_part = DeviceBuffer::alloc(0, 16 * 16 * 8192 * 4).unwrap();
        for &(rows, beta) in &[(1usize, 0.0f32), (4, 1.0), (8, 0.0)] {
            let x: Vec<f32> = (0..rows * k).map(|_| rng.next_f32()).collect();
            let y_init: Vec<f32> = (0..rows * n).map(|_| rng.next_f32()).collect();
            let d_x = to_dev_f32(&x);
            let d_y = to_dev_f32(&y_init);
            unsafe {
                mm3::mxfp8_linear(
                    d_w.ptr as *const u8,
                    d_s.ptr as *const u8,
                    n as i32,
                    k as i32,
                    d_x.ptr as *const f32,
                    k as i32,
                    rows as i32,
                    d_y.ptr as *mut f32,
                    n as i32,
                    beta,
                    d_part.ptr as *mut f32,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &stream,
                );
            }
            stream.sync().unwrap();
            let got = read_dev_f32(&d_y, rows * n);
            let mut exp = vec![0.0f32; rows * n];
            for r in 0..rows {
                for ni in 0..n {
                    let dot: f32 = x[r * k..(r + 1) * k].iter().zip(w_ref[ni * k..(ni + 1) * k].iter()).map(|(&a, &b)| a * b).sum();
                    exp[r * n + ni] = dot + if beta == 0.0 { 0.0 } else { beta * y_init[r * n + ni] };
                }
            }
            let re = rel_err(&got, &exp);
            assert!(re < 1e-4, "mxc rows={rows} beta={beta}: rel={re}");
        }
        println!("mxc_kernel K=6144 N=512 rows=1,4,8: PASSED");
    }

    // 3) Wide 512-thread sparse attention (Hq == 16 * Hkv, T=1,5, splits=16, BF16 & INT8+Had128-V) + DSpark window attn
    {
        let (hq, hkv, hd, blk, topk, max_ctx) = (32usize, 2usize, 128usize, 16usize, 16usize, 320usize);
        let scale = 1.0f32 / (hd as f32).sqrt();
        let kept_blocks: Vec<i32> = (0..topk as i32).collect();
        let kc_f32: Vec<f32> = (0..max_ctx * hkv * hd).map(|_| bf16::from_f32(rng.next_f32() * 0.5).to_f32()).collect();
        let vc_f32: Vec<f32> = (0..max_ctx * hkv * hd).map(|_| bf16::from_f32(rng.next_f32() * 0.5).to_f32()).collect();
        let kc_bf16: Vec<u16> = kc_f32.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let vc_bf16: Vec<u16> = vc_f32.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let mut kc_i8 = vec![0i8; max_ctx * hkv * hd];
        let mut ks_i8 = vec![0.0f32; max_ctx * hkv];
        let mut vc_i8 = vec![0i8; max_ctx * hkv * hd];
        let mut vs_i8 = vec![0.0f32; max_ctx * hkv];
        for row in 0..max_ctx * hkv {
            let kr = &kc_f32[row * hd..(row + 1) * hd];
            let vr = had128_cpu(&vc_f32[row * hd..(row + 1) * hd]);
            let km = kr.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let vm = vr.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            ks_i8[row] = km / 127.0;
            vs_i8[row] = vm / 127.0;
            for d in 0..hd {
                kc_i8[row * hd + d] = if km > 0.0 { (kr[d] / ks_i8[row]).round().clamp(-127.0, 127.0) as i8 } else { 0 };
                vc_i8[row * hd + d] = if vm > 0.0 { (vr[d] / vs_i8[row]).round().clamp(-127.0, 127.0) as i8 } else { 0 };
            }
        }
        let d_kc16 = to_dev_u16(&kc_bf16);
        let d_vc16 = to_dev_u16(&vc_bf16);
        let d_kci8 = to_dev_u8(bytemuck::cast_slice(&kc_i8));
        let d_vci8 = to_dev_u8(bytemuck::cast_slice(&vc_i8));
        let d_ksi8 = to_dev_f32(&ks_i8);
        let d_vsi8 = to_dev_f32(&vs_i8);

        for &t_cnt in &[1usize, 5usize] {
            let q: Vec<f32> = (0..t_cnt * hq * hd).map(|_| rng.next_f32()).collect();
            let pos: Vec<i32> = (0..t_cnt).map(|i| (max_ctx - 20 + i) as i32).collect();
            let mut all_kept = Vec::with_capacity(hkv * t_cnt * topk);
            for _ in 0..hkv * t_cnt {
                all_kept.extend_from_slice(&kept_blocks);
            }
            let d_q = to_dev_f32(&q);
            let d_pos = to_dev_i32(&pos);
            let d_kept = to_dev_i32(&all_kept);
            let d_part = DeviceBuffer::alloc(0, t_cnt * hkv * 16 * (hq / hkv) * mm3::ATTN_PART_STRIDE * 4).unwrap();
            let d_out = DeviceBuffer::alloc(0, t_cnt * hq * hd * 4).unwrap();

            for &kv_fmt in &[0i32, 1] {
                unsafe {
                    mm3::msa_sparse_attn(
                        d_q.ptr as *const f32,
                        if kv_fmt == 0 { d_kc16.ptr as *const u16 } else { std::ptr::null() },
                        if kv_fmt == 0 { d_vc16.ptr as *const u16 } else { std::ptr::null() },
                        if kv_fmt == 1 { d_kci8.ptr as *const i8 } else { std::ptr::null() },
                        if kv_fmt == 1 { d_ksi8.ptr as *const f32 } else { std::ptr::null() },
                        if kv_fmt == 1 { d_vci8.ptr as *const i8 } else { std::ptr::null() },
                        if kv_fmt == 1 { d_vsi8.ptr as *const f32 } else { std::ptr::null() },
                        d_kept.ptr as *const i32,
                        d_pos.ptr as *const i32,
                        t_cnt as i32,
                        hq as i32,
                        hkv as i32,
                        topk as i32,
                        blk as i32,
                        kv_fmt,
                        if kv_fmt == 1 { 2 } else { 0 },
                        0,
                        scale,
                        d_part.ptr as *mut f32,
                        16,
                        d_out.ptr as *mut f32,
                        &stream,
                    );
                }
                stream.sync().unwrap();
                let got = read_dev_f32(&d_out, t_cnt * hq * hd);
                let gqa = hq / hkv;
                let mut exp = vec![0.0f32; t_cnt * hq * hd];
                let valid_end = topk * blk;
                for t in 0..t_cnt {
                    for g in 0..hkv {
                        for gh in 0..gqa {
                            let h = g * gqa + gh;
                            let qr = &q[(t * hq + h) * hd..(t * hq + h + 1) * hd];
                            let scores: Vec<f32> = (0..valid_end)
                                .map(|kp| {
                                    let kr = &kc_f32[(kp * hkv + g) * hd..(kp * hkv + g + 1) * hd];
                                    qr.iter().zip(kr.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale
                                })
                                .collect();
                            let m_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let exps: Vec<f32> = scores.iter().map(|&s| (s - m_max).exp()).collect();
                            let sum_e: f32 = exps.iter().sum();
                            for kp in 0..valid_end {
                                let p = exps[kp] / sum_e;
                                let vr = &vc_f32[(kp * hkv + g) * hd..(kp * hkv + g + 1) * hd];
                                for d in 0..hd {
                                    exp[(t * hq + h) * hd + d] += p * vr[d];
                                }
                            }
                        }
                    }
                }
                let re = rel_err(&got, &exp);
                let tol = if kv_fmt == 0 { 1e-3 } else { 2.5e-2 };
                assert!(re < tol, "wide msa T={t_cnt} kv_fmt={kv_fmt} rel={re}");
            }
        }
        // DSpark window attention wide (T=8, Hq=32, Hkv=2, splits=16)
        {
            let t_cnt = 8usize;
            let ring = max_ctx;
            let hi = 250i32;
            let lo: Vec<i32> = (0..t_cnt).map(|i| 10 + i as i32 * 5).collect();
            let q: Vec<f32> = (0..t_cnt * hq * hd).map(|_| rng.next_f32()).collect();
            let d_q = to_dev_f32(&q);
            let d_lo = to_dev_i32(&lo);
            let d_part = DeviceBuffer::alloc(0, t_cnt * hkv * 16 * 16 * mm3::ATTN_PART_STRIDE * 4).unwrap();
            let d_out = DeviceBuffer::alloc(0, t_cnt * hq * hd * 4).unwrap();
            unsafe {
                mm3::dspark_window_attn(
                    d_q.ptr as *const f32,
                    d_kc16.ptr as *const u16,
                    d_vc16.ptr as *const u16,
                    ring as i32,
                    d_lo.ptr as *const i32,
                    hi,
                    t_cnt as i32,
                    hq as i32,
                    hkv as i32,
                    scale,
                    d_part.ptr as *mut f32,
                    16,
                    d_out.ptr as *mut f32,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let got = read_dev_f32(&d_out, t_cnt * hq * hd);
            let mut exp = vec![0.0f32; t_cnt * hq * hd];
            for t in 0..t_cnt {
                for g in 0..hkv {
                    for gh in 0..16 {
                        let h = g * 16 + gh;
                        let qr = &q[(t * hq + h) * hd..(t * hq + h + 1) * hd];
                        let scores: Vec<f32> = (lo[t]..=hi)
                            .map(|kp| {
                                let kr = &kc_f32[(kp as usize * hkv + g) * hd..(kp as usize * hkv + g + 1) * hd];
                                qr.iter().zip(kr.iter()).map(|(&a, &b)| a * b).sum::<f32>() * scale
                            })
                            .collect();
                        let m_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let exps: Vec<f32> = scores.iter().map(|&s| (s - m_max).exp()).collect();
                        let sum_e: f32 = exps.iter().sum();
                        for (j, kp) in (lo[t]..=hi).enumerate() {
                            let vr = &vc_f32[(kp as usize * hkv + g) * hd..(kp as usize * hkv + g + 1) * hd];
                            for d in 0..hd {
                                exp[(t * hq + h) * hd + d] += (exps[j] / sum_e) * vr[d];
                            }
                        }
                    }
                }
            }
            let re = rel_err(&got, &exp);
            assert!(re < 1e-3, "wide dspark window attn rel={re}");
        }
        println!("wide 512-thread attention (msa & dspark splits=16): PASSED");
    }

    // 4) dspark_draft_block + iota + wait_u32 mapped synchronization
    {
        let (d, b, pos0, window, anchor_id) = (256usize, 8usize, 100i32, 4096i32, 777i32);
        let anc: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let mask: Vec<f32> = (0..d).map(|_| rng.next_f32()).collect();
        let mut anc_mapped = PinnedMappedBuffer::<f32>::alloc(d).unwrap();
        anc_mapped.as_mut_slice().copy_from_slice(&anc);
        let d_mask = to_dev_f32(&mask);
        let d_h = DeviceBuffer::alloc(0, b * d * 4).unwrap();
        let d_pos = DeviceBuffer::alloc(0, b * 4).unwrap();
        let d_lo = DeviceBuffer::alloc(0, b * 4).unwrap();
        let d_prev = DeviceBuffer::alloc(0, 4).unwrap();
        let mut wait_mapped = PinnedMappedBuffer::<u32>::alloc(1).unwrap();
        wait_mapped.as_mut_slice()[0] = 0;

        unsafe {
            mm3::dspark_draft_block(
                anc_mapped.dev_ptr() as *const f32,
                d_mask.ptr as *const f32,
                d as i32,
                b as i32,
                pos0,
                window,
                anchor_id,
                d_h.ptr as *mut f32,
                d_pos.ptr as *mut i32,
                d_lo.ptr as *mut i32,
                d_prev.ptr as *mut i32,
                &stream,
            );
            mm3::wait_u32(wait_mapped.dev_ptr() as *const u32, 42, &stream);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            std::ptr::write_volatile(wait_mapped.host_ptr(), 42);
        }
        stream.sync().unwrap();

        let got_h = read_dev_f32(&d_h, b * d);
        assert!(max_abs_err(&got_h[..d], &anc) == 0.0);
        for t in 1..b {
            assert!(max_abs_err(&got_h[t * d..(t + 1) * d], &mask) == 0.0);
        }
        assert_eq!(read_dev_i32(&d_pos, b), (pos0..pos0 + b as i32).collect::<Vec<_>>());
        assert_eq!(read_dev_i32(&d_lo, b), vec![0; b]);
        assert_eq!(read_dev_i32(&d_prev, 1)[0], anchor_id);

        let d_iota = DeviceBuffer::alloc(0, 12 * 4).unwrap();
        unsafe {
            mm3::iota(d_iota.ptr as *mut i32, 12, 55, &stream);
        }
        stream.sync().unwrap();
        assert_eq!(read_dev_i32(&d_iota, 12), (55..67).collect::<Vec<_>>());
        println!("dspark_draft_block + iota + wait_u32: PASSED");
    }

    // 5) Small-M BF16 GEMV dispatch (M = 1, 5, 7, ldB = 5120, beta = 0.0 & 1.0)
    {
        let (n, k, ld_b) = (512usize, 1024usize, 5120usize);
        let b_f32: Vec<f32> = (0..n * ld_b)
            .map(|_| bf16::from_f32(rng.next_f32()).to_f32())
            .collect();
        let b_u16: Vec<u16> = b_f32.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let d_b = to_dev_u16(&b_u16);
        for &(m, beta) in &[(1usize, 0.0f32), (5, 1.0), (7, 0.0)] {
            let a_f32: Vec<f32> = (0..m * k)
                .map(|_| bf16::from_f32(rng.next_f32()).to_f32())
                .collect();
            let a_u16: Vec<u16> = a_f32.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
            let y_init: Vec<f32> = (0..m * n).map(|_| rng.next_f32()).collect();
            let d_a = to_dev_u16(&a_u16);
            let d_c = to_dev_f32(&y_init);
            unsafe {
                mm3::bf16_gemm_mma(
                    d_a.ptr as *const u16,
                    k as i32,
                    d_b.ptr as *const u16,
                    ld_b as i32,
                    d_c.ptr as *mut f32,
                    n as i32,
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    beta,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let got = read_dev_f32(&d_c, m * n);
            let mut exp = vec![0.0f32; m * n];
            for r in 0..m {
                for ni in 0..n {
                    let dot: f32 = a_f32[r * k..(r + 1) * k]
                        .iter()
                        .zip(b_f32[ni * ld_b..ni * ld_b + k].iter())
                        .map(|(&av, &bv)| av * bv)
                        .sum();
                    exp[r * n + ni] = dot + if beta == 0.0 { 0.0 } else { beta * y_init[r * n + ni] };
                }
            }
            let re = rel_err(&got, &exp);
            assert!(re < 1e-4, "bf16_gemv_small_m M={m} beta={beta}: rel={re}");
        }
        println!("bf16_gemv_small_m M=1,5,7 ldB=5120: PASSED");
    }
}
