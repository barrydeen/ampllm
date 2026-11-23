use ampllm_cuda_kernels::ffi::*;
use ampllm_cuda_kernels::{glm5, qwen4, CudaEvent, CudaStream, DeviceBuffer, PinnedMappedBuffer};
use ampllm_gguf::{dequantize_row_to_f32, vec_dot_q_f32_ref, GgmlType};
use half::{bf16, f16};
use std::sync::Mutex;

static GPU_LOCK: Mutex<()> = Mutex::new(());

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
        num += ((g - e) as f64).abs();
        den += (e as f64).abs();
    }
    (num / den.max(1e-12)) as f32
}

fn max_abs_err(got: &[f32], exp: &[f32]) -> f32 {
    assert_eq!(got.len(), exp.len());
    got.iter()
        .zip(exp.iter())
        .map(|(&g, &e)| (g - e).abs())
        .fold(0.0f32, f32::max)
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0f32 / (1.0f32 + (-x).exp())
}

fn silu_f32(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

fn glm_swiglu_clamp_val(g: f32, u: f32, lim: f32) -> f32 {
    if lim > 0.0 {
        let gc = g.min(lim);
        let uc = u.clamp(-lim, lim);
        (gc / (1.0f32 + (-gc).exp())) * uc
    } else {
        (g / (1.0f32 + (-g).exp())) * u
    }
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
        if (s - s_exp).abs() > 5e-2 * s_exp.abs().max(1.0) {
            panic!("q8_1 s mismatch at block {b}: got {s} exp {s_exp}");
        }
    }
    out
}

fn fill_quant_row(t: GgmlType, k: usize, rng: &mut Rng) -> Vec<u8> {
    let nb = k / t.block_size();
    let bsz = t.type_size();
    let mut bytes = vec![0u8; nb * bsz];
    for b in 0..nb {
        let blk = &mut bytes[b * bsz..(b + 1) * bsz];
        match t {
            GgmlType::Q5_1 => {
                let d = f16::from_f32(rng.next_f32_range(0.008, 0.025)).to_le_bytes();
                let m = f16::from_f32(rng.next_f32_range(-0.12, 0.12)).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                blk[2..4].copy_from_slice(&m);
                for x in &mut blk[4..24] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Q2_0 => {
                let d = f16::from_f32(rng.next_f32_range(0.01, 0.03)).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for x in &mut blk[2..18] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Q4_0 => {
                let d = f16::from_f32(rng.next_f32_range(0.01, 0.03)).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for x in &mut blk[2..18] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Q4K => {
                let d = f16::from_f32(rng.next_f32_range(0.007, 0.015)).to_le_bytes();
                let dm = f16::from_f32(rng.next_f32_range(0.004, 0.010)).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                blk[2..4].copy_from_slice(&dm);
                for x in &mut blk[4..144] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Q5K => {
                let d = f16::from_f32(rng.next_f32_range(0.007, 0.015)).to_le_bytes();
                let dm = f16::from_f32(rng.next_f32_range(0.004, 0.010)).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                blk[2..4].copy_from_slice(&dm);
                for x in &mut blk[4..176] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Q8_0 => {
                let d = f16::from_f32(rng.next_f32_range(0.01, 0.02)).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for x in &mut blk[2..34] {
                    *x = rng.next_u8();
                }
            }
            _ => unreachable!(),
        }
    }
    bytes
}

fn fill_quant_matrix(t: GgmlType, n: usize, k: usize, rng: &mut Rng) -> (Vec<u8>, usize) {
    let rb = t.row_bytes(k);
    let mut buf = Vec::with_capacity(n * rb);
    for _ in 0..n {
        buf.extend_from_slice(&fill_quant_row(t, k, rng));
    }
    (buf, rb)
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

fn read_dev_u8(buf: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len];
    buf.copy_to_host(&mut out).unwrap();
    out
}

// ============================================================================
// Test 1: Q5_1 & Q2_0 GEMV, MMQ MMA, Grouped MMA, MoE Plan/Resident & Dequant
// ============================================================================

#[test]
fn test_qwen4_quants_q5_1_q2_0_gemv_mmq_moe_dequant() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x5151_2020_4444_0001);

    let cases = [
        (GgmlType::Q5_1, 640usize, 2560usize),
        (GgmlType::Q2_0, 2560usize, 640usize),
        (GgmlType::Q2_0, 640usize, 2560usize),
        (GgmlType::Q4K, 2560usize, 640usize),
    ];

    for &(ty, k, n) in &cases {
        let (w_bytes, rb) = fill_quant_matrix(ty, n, k, &mut rng);
        let d_w = to_dev_u8(&w_bytes);

        let m_max = 128usize;
        let x: Vec<f32> = (0..m_max * k).map(|_| rng.next_f32()).collect();
        let d_x = to_dev_f32(&x);
        let nb_x = k / 32;
        let d_xq8 = DeviceBuffer::alloc(0, m_max * nb_x * 36).unwrap();
        unsafe {
            launch_quantize_q8_1(d_x.ptr as *const f32, d_xq8.ptr, m_max as i32, k as i32, stream.raw);
        }
        stream.sync().unwrap();
        let x_q8_host = read_dev_u8(&d_xq8, m_max * nb_x * 36);

        // Precompute CPU FP32 reference and Q8_1 reference for all 128 rows
        let mut ref_fp32 = vec![0.0f32; m_max * n];
        let mut ref_q8 = vec![0.0f32; m_max * n];
        for mi in 0..m_max {
            let x_f32_row = &x[mi * k..(mi + 1) * k];
            let x_q8_deq = dequant_q8_1_blocks(&x_q8_host[mi * nb_x * 36..(mi + 1) * nb_x * 36], k);
            for ni in 0..n {
                let wr = &w_bytes[ni * rb..(ni + 1) * rb];
                ref_fp32[mi * n + ni] = vec_dot_q_f32_ref(ty, wr, x_f32_row);
                ref_q8[mi * n + ni] = vec_dot_q_f32_ref(ty, wr, &x_q8_deq);
            }
        }

        // 1) GEMV at M=1 and M=4
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
            println!("linear_q8_1 {ty:?} M={m} ({k}->{n}): rel_fp32={e_fp32:.6} rel_q8={e_q8:.6}");
            assert!(e_fp32 < 2e-2, "{ty:?} M={m} rel_fp32={e_fp32}");
            assert!(e_q8 < 5e-4, "{ty:?} M={m} rel_q8={e_q8}");
        }

        // 2) MMQ MMA at M=16, 64, 128
        for &m in &[16usize, 64usize, 128usize] {
            let d_y = DeviceBuffer::alloc(0, m * n * 4).unwrap();
            unsafe {
                glm5::gemm_mmq_mma(
                    ty as i32,
                    d_xq8.ptr,
                    d_w.ptr,
                    rb,
                    d_y.ptr as *mut f32,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    0.0,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let got = read_dev_f32(&d_y, m * n);
            let e_fp32 = rel_err(&got, &ref_fp32[..m * n]);
            let e_q8 = rel_err(&got, &ref_q8[..m * n]);
            println!("mmq_mma {ty:?} M={m} ({k}->{n}): rel_fp32={e_fp32:.6} rel_q8={e_q8:.6}");
            assert!(e_fp32 < 2e-2, "mmq {ty:?} M={m} rel_fp32={e_fp32}");
            assert!(e_q8 < 5e-4, "mmq {ty:?} M={m} rel_q8={e_q8}");
        }

        // 3) Dequant to F16 for Q5_1 & Q2_0
        if ty == GgmlType::Q5_1 || ty == GgmlType::Q2_0 {
            let total_elems = n * k;
            let d_f16 = DeviceBuffer::alloc(0, total_elems * 2).unwrap();
            unsafe {
                glm5::dequant_to_f16(ty as i32, d_w.ptr, d_f16.ptr, total_elems as i64, &stream);
            }
            stream.sync().unwrap();
            let got_raw = read_dev_u8(&d_f16, total_elems * 2);
            let got_f16: Vec<f32> = bytemuck::cast_slice::<u8, u16>(&got_raw)
                .iter()
                .map(|&b| f16::from_bits(b).to_f32())
                .collect();
            let mut exp_f32 = vec![0.0f32; total_elems];
            for r in 0..n {
                dequantize_row_to_f32(ty, &w_bytes[r * rb..(r + 1) * rb], &mut exp_f32[r * k..(r + 1) * k]);
            }
            let max_d = max_abs_err(&got_f16, &exp_f32);
            println!("dequant_to_f16 {ty:?} ({n}x{k}): max_abs={max_d:.6}");
            assert!(max_d < 5e-3, "dequant {ty:?} max_d={max_d}");
        }
    }

    // 3b) Coalesced Q4_0 M=1 GEMV (dot_row_q4_0_sub, used by MTP Q4_0 draft head)
    {
        let (ty, k, n) = (GgmlType::Q4_0, 2560usize, 640usize);
        let (w_bytes, rb) = fill_quant_matrix(ty, n, k, &mut rng);
        let d_w = to_dev_u8(&w_bytes);
        let x: Vec<f32> = (0..k).map(|_| rng.next_f32()).collect();
        let d_x = to_dev_f32(&x);
        let nb_x = k / 32;
        let d_xq8 = DeviceBuffer::alloc(0, nb_x * 36).unwrap();
        let d_y = DeviceBuffer::alloc(0, n * 4).unwrap();
        unsafe {
            launch_quantize_q8_1(d_x.ptr as *const f32, d_xq8.ptr, 1, k as i32, stream.raw);
            launch_linear_q8_1(
                ty as i32,
                d_w.ptr,
                d_xq8.ptr,
                d_x.ptr as *const f32,
                d_y.ptr as *mut f32,
                n as i32,
                k as i32,
                1,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let x_q8_host = read_dev_u8(&d_xq8, nb_x * 36);
        let x_q8_deq = dequant_q8_1_blocks(&x_q8_host, k);
        let mut ref_fp32 = vec![0.0f32; n];
        let mut ref_q8 = vec![0.0f32; n];
        for ni in 0..n {
            let wr = &w_bytes[ni * rb..(ni + 1) * rb];
            ref_fp32[ni] = vec_dot_q_f32_ref(ty, wr, &x);
            ref_q8[ni] = vec_dot_q_f32_ref(ty, wr, &x_q8_deq);
        }
        let got = read_dev_f32(&d_y, n);
        let e_fp32 = rel_err(&got, &ref_fp32);
        let e_q8 = rel_err(&got, &ref_q8);
        println!("linear_q8_1 {ty:?} M=1 ({k}->{n}): rel_fp32={e_fp32:.6} rel_q8={e_q8:.6}");
        assert!(e_fp32 < 6e-3, "{ty:?} M=1 rel_fp32={e_fp32}");
        assert!(e_q8 < 1e-4, "{ty:?} M=1 rel_q8={e_q8}");
    }

    // 4) Grouped MMQ MMA for Q5_1 and Q2_0 (4 experts, 86 rows total, tile_mode=0 BM=32)
    for &(ty, k, n) in &[(GgmlType::Q5_1, 640usize, 2560usize), (GgmlType::Q2_0, 640usize, 2560usize)] {
        let counts = [18i32, 25, 11, 32];
        let total_rows: usize = counts.iter().sum::<i32>() as usize; // 86
        let exp_row_off = vec![0i32, 18, 43, 54];
        let tile_exp_id = vec![0i32, 1, 2, 3];
        let tile_m_off = vec![0i32, 0, 0, 0];

        let mut exp_bufs = Vec::new();
        let mut exp_ptrs = Vec::new();
        let mut exp_w_host = Vec::new();
        let rb = ty.row_bytes(k);
        for _ in 0..4 {
            let (wb, _) = fill_quant_matrix(ty, n, k, &mut rng);
            let db = to_dev_u8(&wb);
            exp_ptrs.push(db.ptr as u64);
            exp_w_host.push(wb);
            exp_bufs.push(db);
        }

        let x: Vec<f32> = (0..total_rows * k).map(|_| rng.next_f32()).collect();
        let d_x = to_dev_f32(&x);
        let nb_x = k / 32;
        let d_xq8 = DeviceBuffer::alloc(0, total_rows * nb_x * 36).unwrap();
        unsafe {
            launch_quantize_q8_1(d_x.ptr as *const f32, d_xq8.ptr, total_rows as i32, k as i32, stream.raw);
        }
        let d_ptrs = DeviceBuffer::alloc(0, 4 * 8).unwrap();
        d_ptrs.copy_from_host(bytemuck::cast_slice(&exp_ptrs)).unwrap();
        let d_off = to_dev_i32(&exp_row_off);
        let d_nt = to_dev_i32(&counts);
        let d_teid = to_dev_i32(&tile_exp_id);
        let d_tmoff = to_dev_i32(&tile_m_off);
        let d_c = DeviceBuffer::alloc_zeroed(0, total_rows * n * 4).unwrap();

        unsafe {
            glm5::gemm_mmq_mma_grouped(
                ty as i32,
                d_xq8.ptr,
                d_ptrs.ptr as *const u64,
                0,
                rb,
                d_c.ptr as *mut f32,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                d_off.ptr as *const i32,
                d_nt.ptr as *const i32,
                d_teid.ptr as *const i32,
                d_tmoff.ptr as *const i32,
                4,
                n as i32,
                k as i32,
                1.0,
                false,
                0,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got = read_dev_f32(&d_c, total_rows * n);
        let xq8_host = read_dev_u8(&d_xq8, total_rows * nb_x * 36);
        let mut exp_q8 = vec![0.0f32; total_rows * n];
        let mut exp_fp32 = vec![0.0f32; total_rows * n];
        for e in 0..4 {
            let r0 = exp_row_off[e] as usize;
            let cnt = counts[e] as usize;
            for loc in 0..cnt {
                let row = r0 + loc;
                let xf = &x[row * k..(row + 1) * k];
                let xqd = dequant_q8_1_blocks(&xq8_host[row * nb_x * 36..(row + 1) * nb_x * 36], k);
                for ni in 0..n {
                    let wr = &exp_w_host[e][ni * rb..(ni + 1) * rb];
                    exp_fp32[row * n + ni] = vec_dot_q_f32_ref(ty, wr, xf);
                    exp_q8[row * n + ni] = vec_dot_q_f32_ref(ty, wr, &xqd);
                }
            }
        }
        let e_fp32 = rel_err(&got, &exp_fp32);
        let e_q8 = rel_err(&got, &exp_q8);
        println!("mmq_grouped {ty:?}: rel_fp32={e_fp32:.6} rel_q8={e_q8:.6}");
        assert!(e_fp32 < 2e-2);
        assert!(e_q8 < 5e-4);
    }

    // 5) MoE Plan & MoE Resident with topk=10 (Qwen3.8 Q4_K/Q5_1 and MTP Q2_0/Q2_0)
    let n_embd = 2560usize;
    let n_ff = 640usize;
    let topk = 10usize;
    let n_exp = 16usize;
    for &(gu_ty, d_ty) in &[(GgmlType::Q4K, GgmlType::Q5_1), (GgmlType::Q2_0, GgmlType::Q2_0)] {
        let gu_rb = gu_ty.row_bytes(n_embd);
        let d_rb = d_ty.row_bytes(n_ff);
        let g_bytes = n_ff * gu_rb;
        let u_bytes = n_ff * gu_rb;
        let d_bytes = n_embd * d_rb;
        let blob_bytes = g_bytes + u_bytes + d_bytes;

        let mut blobs_host = Vec::with_capacity(n_exp);
        let mut blobs_dev = Vec::with_capacity(n_exp);
        let mut gate_ptrs = Vec::with_capacity(n_exp);
        let mut up_ptrs = Vec::with_capacity(n_exp);
        let mut down_ptrs = Vec::with_capacity(n_exp);
        for _ in 0..n_exp {
            let (gw, _) = fill_quant_matrix(gu_ty, n_ff, n_embd, &mut rng);
            let (uw, _) = fill_quant_matrix(gu_ty, n_ff, n_embd, &mut rng);
            let (dw, _) = fill_quant_matrix(d_ty, n_embd, n_ff, &mut rng);
            let mut blob = Vec::with_capacity(blob_bytes);
            blob.extend_from_slice(&gw);
            blob.extend_from_slice(&uw);
            blob.extend_from_slice(&dw);
            let db = to_dev_u8(&blob);
            let base = db.ptr as u64;
            gate_ptrs.push(base);
            up_ptrs.push(base + g_bytes as u64);
            down_ptrs.push(base + (g_bytes + u_bytes) as u64);
            blobs_host.push((gw, uw, dw));
            blobs_dev.push(db);
        }

        for &m in &[1usize, 2usize] {
            let x: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32()).collect();
            let d_x = to_dev_f32(&x);
            let d_xq8 = DeviceBuffer::alloc(0, m * (n_embd / 32) * 36).unwrap();
            unsafe {
                launch_quantize_q8_1(d_x.ptr as *const f32, d_xq8.ptr, m as i32, n_embd as i32, stream.raw);
            }

            let mut eids = vec![0i32; m * topk];
            let mut rws = vec![0.0f32; m * topk];
            let mut plan_ptrs = vec![0u64; m * topk];
            for mi in 0..m {
                for s in 0..topk {
                    let e = ((mi * 5 + s) % n_exp) as i32;
                    eids[mi * topk + s] = e;
                    rws[mi * topk + s] = 0.05 + (s as f32) * 0.015;
                    plan_ptrs[mi * topk + s] = gate_ptrs[e as usize];
                }
            }
            let d_eids = to_dev_i32(&eids);
            let d_rws = to_dev_f32(&rws);
            let d_plan_ptrs = DeviceBuffer::alloc(0, plan_ptrs.len() * 8).unwrap();
            d_plan_ptrs.copy_from_host(bytemuck::cast_slice(&plan_ptrs)).unwrap();
            let d_gp = DeviceBuffer::alloc(0, gate_ptrs.len() * 8).unwrap();
            let d_up = DeviceBuffer::alloc(0, up_ptrs.len() * 8).unwrap();
            let d_dp = DeviceBuffer::alloc(0, down_ptrs.len() * 8).unwrap();
            d_gp.copy_from_host(bytemuck::cast_slice(&gate_ptrs)).unwrap();
            d_up.copy_from_host(bytemuck::cast_slice(&up_ptrs)).unwrap();
            d_dp.copy_from_host(bytemuck::cast_slice(&down_ptrs)).unwrap();

            let d_inter_q8 = DeviceBuffer::alloc(0, m * topk * (n_ff / 32) * 36).unwrap();
            let d_out_plan = DeviceBuffer::alloc(0, m * n_embd * 4).unwrap();
            let d_out_res = DeviceBuffer::alloc(0, m * n_embd * 4).unwrap();
            let clamp_lim = 12.0f32;

            unsafe {
                glm5::moe_plan_filtered(
                    gu_ty as i32,
                    d_ty as i32,
                    d_plan_ptrs.ptr as *const u64,
                    g_bytes as u32,
                    (g_bytes + u_bytes) as u32,
                    gu_rb,
                    d_rb,
                    d_xq8.ptr,
                    d_rws.ptr as *const f32,
                    d_inter_q8.ptr,
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    d_out_plan.ptr as *mut f32,
                    n_embd as i32,
                    n_ff as i32,
                    topk as i32,
                    m as i32,
                    clamp_lim,
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    &stream,
                );
                glm5::moe_resident(
                    gu_ty as i32,
                    d_ty as i32,
                    d_gp.ptr as *const u64,
                    d_up.ptr as *const u64,
                    d_dp.ptr as *const u64,
                    gu_rb,
                    d_rb,
                    d_xq8.ptr,
                    d_eids.ptr as *const i32,
                    d_rws.ptr as *const f32,
                    d_inter_q8.ptr,
                    std::ptr::null_mut(),
                    d_out_res.ptr as *mut f32,
                    n_embd as i32,
                    n_ff as i32,
                    topk as i32,
                    m as i32,
                    clamp_lim,
                    &stream,
                );
            }
            stream.sync().unwrap();

            let got_plan = read_dev_f32(&d_out_plan, m * n_embd);
            let got_res = read_dev_f32(&d_out_res, m * n_embd);
            let xq8_host = read_dev_u8(&d_xq8, m * (n_embd / 32) * 36);

            let mut ref_fp32 = vec![0.0f32; m * n_embd];
            let mut ref_q8 = vec![0.0f32; m * n_embd];
            for mi in 0..m {
                let x_f32 = &x[mi * n_embd..(mi + 1) * n_embd];
                let xqd = dequant_q8_1_blocks(
                    &xq8_host[mi * (n_embd / 32) * 36..(mi + 1) * (n_embd / 32) * 36],
                    n_embd,
                );
                for s in 0..topk {
                    let e = eids[mi * topk + s] as usize;
                    let rw = rws[mi * topk + s];
                    let (ref gw, ref uw, ref dw) = blobs_host[e];
                    let mut inter_f32 = vec![0.0f32; n_ff];
                    let mut inter_q8 = vec![0.0f32; n_ff];
                    for r in 0..n_ff {
                        let vg_f = vec_dot_q_f32_ref(gu_ty, &gw[r * gu_rb..(r + 1) * gu_rb], x_f32);
                        let vu_f = vec_dot_q_f32_ref(gu_ty, &uw[r * gu_rb..(r + 1) * gu_rb], x_f32);
                        inter_f32[r] = glm_swiglu_clamp_val(vg_f, vu_f, clamp_lim);

                        let vg_q = vec_dot_q_f32_ref(gu_ty, &gw[r * gu_rb..(r + 1) * gu_rb], &xqd);
                        let vu_q = vec_dot_q_f32_ref(gu_ty, &uw[r * gu_rb..(r + 1) * gu_rb], &xqd);
                        inter_q8[r] = glm_swiglu_clamp_val(vg_q, vu_q, clamp_lim);
                    }
                    // Simulate Q8_1 quantization of intermediate vector matching inter_q8
                    let mut inter_q8_quant = vec![0.0f32; n_ff];
                    for b in 0..(n_ff / 32) {
                        let sl = &inter_q8[b * 32..(b + 1) * 32];
                        let amax = sl.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                        let d = f16::from_f32(amax / 127.0).to_f32();
                        let inv_d = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                        for i in 0..32 {
                            let q = (sl[i] * inv_d).round().clamp(-127.0, 127.0);
                            inter_q8_quant[b * 32 + i] = q * d;
                        }
                    }
                    for r in 0..n_embd {
                        let d_f = vec_dot_q_f32_ref(d_ty, &dw[r * d_rb..(r + 1) * d_rb], &inter_f32);
                        ref_fp32[mi * n_embd + r] += rw * d_f;
                        let d_q = vec_dot_q_f32_ref(d_ty, &dw[r * d_rb..(r + 1) * d_rb], &inter_q8_quant);
                        ref_q8[mi * n_embd + r] += rw * d_q;
                    }
                }
            }

            let e_plan = rel_err(&got_plan, &ref_fp32);
            let e_res = rel_err(&got_res, &ref_fp32);
            let e_plan_q8 = rel_err(&got_plan, &ref_q8);
            let e_res_q8 = rel_err(&got_res, &ref_q8);
            let e_diff = rel_err(&got_plan, &got_res);
            println!(
                "MoE ({gu_ty:?}/{d_ty:?}) M={m}: plan_fp32={e_plan:.6} res_fp32={e_res:.6} plan_q8={e_plan_q8:.6} res_q8={e_res_q8:.6} plan_vs_res={e_diff:.6}"
            );
            assert!(e_plan < 5e-2);
            assert!(e_res < 5e-2);
            assert!(e_plan_q8 < 5e-4);
            assert!(e_res_q8 < 5e-4);
            assert!(e_diff < 2e-5);
        }
    }

    // 4) Prefill small MoE kernels (glm5::moe_prefill_small_gu / down) with n_ff=640 (not % 256!) + Q5_1/Q8_0 down
    for &(gu_ty, d_ty) in &[(GgmlType::Q4K, GgmlType::Q5_1), (GgmlType::Q5K, GgmlType::Q8_0)] {
        let n_embd = 2560usize;
        let n_ff = 640usize;
        let n_jobs = 2usize;
        let nt_per_job = 5usize;
        let m_tot = 6usize;
        let gu_rb = gu_ty.row_bytes(n_embd);
        let d_rb = d_ty.row_bytes(n_ff);
        let u_off = (n_ff * gu_rb + 255) & !255;
        let d_off = (u_off + n_ff * gu_rb + 255) & !255;
        let blob_b = (d_off + n_embd * d_rb + 255) & !255;

        let x: Vec<f32> = (0..m_tot * n_embd).map(|_| rng.next_f32_range(-0.08, 0.08)).collect();
        let mut blobs_bytes = vec![0u8; n_jobs * blob_b];
        let mut blobs_host = Vec::with_capacity(n_jobs);
        for e in 0..n_jobs {
            let (gw, _) = fill_quant_matrix(gu_ty, n_ff, n_embd, &mut rng);
            let (uw, _) = fill_quant_matrix(gu_ty, n_ff, n_embd, &mut rng);
            let (dw, _) = fill_quant_matrix(d_ty, n_embd, n_ff, &mut rng);
            blobs_bytes[e * blob_b..e * blob_b + n_ff * gu_rb].copy_from_slice(&gw);
            blobs_bytes[e * blob_b + u_off..e * blob_b + u_off + n_ff * gu_rb].copy_from_slice(&uw);
            blobs_bytes[e * blob_b + d_off..e * blob_b + d_off + n_embd * d_rb].copy_from_slice(&dw);
            blobs_host.push((gw, uw, dw));
        }

        let mut rows_all = Vec::new();
        let mut wts_all = Vec::new();
        let mut slot_off = Vec::new();
        let mut slot_nt = Vec::new();
        for j in 0..n_jobs {
            slot_off.push(rows_all.len() as i32);
            slot_nt.push(nt_per_job as i32);
            for k in 0..nt_per_job {
                rows_all.push(((j * 2 + k) % m_tot) as i32);
                wts_all.push(rng.next_f32_range(0.05, 0.4));
            }
        }
        let d_blobs = to_dev_u8(&blobs_bytes);
        let exp_ptrs: Vec<u64> = (0..n_jobs).map(|j| d_blobs.as_u64() + (j * blob_b) as u64).collect();
        let d_exp_ptrs = to_dev_u64(&exp_ptrs);
        let d_slot_off = to_dev_i32(&slot_off);
        let d_slot_nt = to_dev_i32(&slot_nt);
        let d_rows = to_dev_i32(&rows_all);
        let d_wts = to_dev_f32(&wts_all);

        let d_x = to_dev_f32(&x);
        let d_xq8 = DeviceBuffer::alloc(0, m_tot * (n_embd / 32) * 36).unwrap();
        let d_inter_q8 = DeviceBuffer::alloc(0, n_jobs * nt_per_job * (n_ff / 32) * 36).unwrap();
        let d_out = DeviceBuffer::alloc_zeroed(0, m_tot * n_embd * 4).unwrap();

        unsafe {
            ampllm_cuda_kernels::ffi::launch_quantize_q8_1(
                d_x.ptr as *const f32,
                d_xq8.ptr,
                m_tot as i32,
                n_embd as i32,
                stream.raw,
            );
            glm5::moe_prefill_small_gu(
                gu_ty as i32,
                d_exp_ptrs.ptr as *const u64,
                d_slot_off.ptr as *const i32,
                d_slot_nt.ptr as *const i32,
                d_rows.ptr as *const i32,
                u_off as u32,
                gu_rb,
                d_xq8.ptr,
                d_inter_q8.ptr,
                n_embd as i32,
                n_ff as i32,
                n_jobs as i32,
                0.0,
                &stream,
            );
            glm5::moe_prefill_small_down(
                d_ty as i32,
                d_exp_ptrs.ptr as *const u64,
                d_slot_off.ptr as *const i32,
                d_slot_nt.ptr as *const i32,
                d_rows.ptr as *const i32,
                d_wts.ptr as *const f32,
                d_off as u32,
                d_rb,
                d_inter_q8.ptr,
                d_out.ptr as *mut f32,
                n_embd as i32,
                n_ff as i32,
                n_jobs as i32,
                &stream,
            );
        }
        stream.sync().unwrap();

        let got = read_dev_f32(&d_out, m_tot * n_embd);
        let xq8_host = read_dev_u8(&d_xq8, m_tot * (n_embd / 32) * 36);
        let iq8_host = read_dev_u8(&d_inter_q8, n_jobs * nt_per_job * (n_ff / 32) * 36);
        let mut ref_q8 = vec![0.0f32; m_tot * n_embd];
        for j in 0..n_jobs {
            let (ref gw, ref uw, ref dw) = blobs_host[j];
            let off = slot_off[j] as usize;
            let nt = slot_nt[j] as usize;
            for ti in 0..nt {
                let row = rows_all[off + ti] as usize;
                let rw = wts_all[off + ti];
                let xqd = dequant_q8_1_blocks(
                    &xq8_host[row * (n_embd / 32) * 36..(row + 1) * (n_embd / 32) * 36],
                    n_embd,
                );
                let mut inter_f = vec![0.0f32; n_ff];
                for r in 0..n_ff {
                    let vg = vec_dot_q_f32_ref(gu_ty, &gw[r * gu_rb..(r + 1) * gu_rb], &xqd);
                    let vu = vec_dot_q_f32_ref(gu_ty, &uw[r * gu_rb..(r + 1) * gu_rb], &xqd);
                    inter_f[r] = glm_swiglu_clamp_val(vg, vu, 0.0);
                }
                let mut inter_q8_sim = vec![0.0f32; n_ff];
                for b in 0..(n_ff / 32) {
                    let sl = &inter_f[b * 32..(b + 1) * 32];
                    let amax = sl.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                    let d = f16::from_f32(amax / 127.0).to_f32();
                    let inv_d = if amax > 0.0 { 127.0 / amax } else { 0.0 };
                    for i in 0..32 {
                        let q = (sl[i] * inv_d).round().clamp(-127.0, 127.0);
                        inter_q8_sim[b * 32 + i] = q * d;
                    }
                }
                let iq_slot = off + ti;
                let iqd = dequant_q8_1_blocks(
                    &iq8_host[iq_slot * (n_ff / 32) * 36..(iq_slot + 1) * (n_ff / 32) * 36],
                    n_ff,
                );
                let er_q8 = rel_err(&iqd, &inter_q8_sim);
                let er_f = rel_err(&iqd, &inter_f);
                assert!(
                    er_q8 < 5e-4 && er_f < 3e-2,
                    "small_gu ({gu_ty:?}) inter mismatch job={j} ti={ti} rel_q8={er_q8:.6} rel_f={er_f:.6}"
                );
                for r in 0..n_embd {
                    let vd = vec_dot_q_f32_ref(d_ty, &dw[r * d_rb..(r + 1) * d_rb], &iqd);
                    ref_q8[row * n_embd + r] += rw * vd;
                }
            }
        }
        let err_small = rel_err(&got, &ref_q8);
        println!("moe_prefill_small ({gu_ty:?}/{d_ty:?}) n_ff=640: rel_q8={err_small:.6}");
        assert!(err_small < 5e-4);
    }
}

// ============================================================================
// Test 2: HC-4 Gated Residual & PLE Post-Ops
// ============================================================================

#[test]
fn test_qwen4_hc4_and_ple_parity() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x4C34_71E0_0000_0002);

    let hc = 4usize;
    let e = 2560usize;
    let lr = 320usize;
    let eps = 1e-6f32;

    for &t in &[1usize, 19usize] {
        let emb: Vec<f32> = (0..t * e).map(|_| rng.next_f32()).collect();
        let w_norm: Vec<f32> = (0..hc * e).map(|_| rng.next_f32_range(0.5, 1.5)).collect();
        let up_logits: Vec<f32> = (0..t * hc * e).map(|_| rng.next_f32()).collect();
        let lo_in: Vec<f32> = (0..t * lr).map(|_| rng.next_f32()).collect();
        let w_inject_f32: Vec<f32> = (0..hc * hc * e).map(|_| rng.next_f32_range(-0.1, 0.1)).collect();
        let w_inject_bf16_bits: Vec<u16> = w_inject_f32.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let w_inject_bf16_f32: Vec<f32> = w_inject_bf16_bits.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        let block_out: Vec<f32> = (0..t * e).map(|_| rng.next_f32()).collect();
        let sh_out: Vec<f32> = (0..t * e).map(|_| rng.next_f32()).collect();
        let sh_gate: Vec<f32> = (0..t).map(|_| rng.next_f32()).collect();
        let add_h: Vec<f32> = (0..t * hc * e).map(|_| rng.next_f32()).collect();

        let d_emb = to_dev_f32(&emb);
        let d_r = DeviceBuffer::alloc(0, t * hc * e * 4).unwrap();
        let d_wn = to_dev_f32(&w_norm);
        let d_xn = DeviceBuffer::alloc(0, t * hc * e * 4).unwrap();
        let d_xn_q8 = DeviceBuffer::alloc(0, t * hc * (e / 32) * 36).unwrap();
        let d_xn_q8_ref = DeviceBuffer::alloc(0, t * hc * (e / 32) * 36).unwrap();
        let d_lo_in = to_dev_f32(&lo_in);
        let d_lo_out = DeviceBuffer::alloc(0, t * lr * 4).unwrap();
        let d_lo_q8 = DeviceBuffer::alloc(0, t * (lr / 32) * 36).unwrap();
        let d_lo_q8_ref = DeviceBuffer::alloc(0, t * (lr / 32) * 36).unwrap();
        let d_up = to_dev_f32(&up_logits);
        let d_mix = DeviceBuffer::alloc(0, t * e * 4).unwrap();
        let d_mix_q8 = DeviceBuffer::alloc(0, t * (e / 32) * 36).unwrap();
        let d_mix_q8_ref = DeviceBuffer::alloc(0, t * (e / 32) * 36).unwrap();
        let d_winj_f32 = to_dev_f32(&w_inject_f32);
        let d_winj_bf16 = to_dev_u16(&w_inject_bf16_bits);
        let d_inj_f32 = DeviceBuffer::alloc(0, t * hc * 4).unwrap();
        let d_inj_bf16 = DeviceBuffer::alloc(0, t * hc * 4).unwrap();
        let d_bout = to_dev_f32(&block_out);
        let d_shout = to_dev_f32(&sh_out);
        let d_shgate = to_dev_f32(&sh_gate);
        let d_r_nosg = DeviceBuffer::alloc(0, t * hc * e * 4).unwrap();
        let d_r_add = DeviceBuffer::alloc(0, t * hc * e * 4).unwrap();
        let d_add_h = to_dev_f32(&add_h);

        unsafe {
            qwen4::hc_expand(d_emb.ptr as *const f32, t as i32, hc as i32, e as i32, d_r.ptr as *mut f32, &stream);
            qwen4::hc_rmsnorm_q8(
                d_r.ptr as *const f32,
                d_wn.ptr as *const f32,
                t as i32,
                hc as i32,
                e as i32,
                eps,
                d_xn.ptr as *mut f32,
                d_xn_q8.ptr,
                &stream,
            );
            launch_quantize_q8_1(d_xn.ptr as *const f32, d_xn_q8_ref.ptr, (t * hc) as i32, e as i32, stream.raw);

            qwen4::hc_down_silu_q8(
                d_lo_in.ptr as *const f32,
                t as i32,
                hc as i32,
                lr as i32,
                d_lo_out.ptr as *mut f32,
                d_lo_q8.ptr,
                &stream,
            );
            launch_quantize_q8_1(d_lo_out.ptr as *const f32, d_lo_q8_ref.ptr, t as i32, lr as i32, stream.raw);

            qwen4::hc_mix_q8(
                d_xn.ptr as *const f32,
                d_up.ptr as *const f32,
                t as i32,
                hc as i32,
                e as i32,
                d_mix.ptr as *mut f32,
                d_mix_q8.ptr,
                &stream,
            );
            launch_quantize_q8_1(d_mix.ptr as *const f32, d_mix_q8_ref.ptr, t as i32, e as i32, stream.raw);

            qwen4::hc_inject_dot(
                d_xn.ptr as *const f32,
                d_winj_f32.ptr,
                GgmlType::F32 as i32,
                t as i32,
                hc as i32,
                e as i32,
                d_inj_f32.ptr as *mut f32,
                &stream,
            );
            qwen4::hc_inject_dot(
                d_xn.ptr as *const f32,
                d_winj_bf16.ptr,
                GgmlType::Bf16 as i32,
                t as i32,
                hc as i32,
                e as i32,
                d_inj_bf16.ptr as *mut f32,
                &stream,
            );

            // Copy initial R to d_r_nosg for non-shared check, then run both hc_write modes
            ampllm_cuda_kernels::copy_d2d_raw_async(d_r_nosg.ptr, d_r.ptr, t * hc * e * 4, &stream).unwrap();
            qwen4::hc_write(
                d_r_nosg.ptr as *mut f32,
                d_bout.ptr as *const f32,
                std::ptr::null(),
                std::ptr::null(),
                d_inj_f32.ptr as *const f32,
                t as i32,
                hc as i32,
                e as i32,
                &stream,
            );
            qwen4::hc_write(
                d_r.ptr as *mut f32,
                d_bout.ptr as *const f32,
                d_shout.ptr as *const f32,
                d_shgate.ptr as *const f32,
                d_inj_f32.ptr as *const f32,
                t as i32,
                hc as i32,
                e as i32,
                &stream,
            );
            qwen4::add_streams_broadcast(
                d_add_h.ptr as *const f32,
                d_emb.ptr as *const f32,
                t as i32,
                hc as i32,
                e as i32,
                d_r_add.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();

        let got_xn = read_dev_f32(&d_xn, t * hc * e);
        let got_lo = read_dev_f32(&d_lo_out, t * lr);
        let got_mix = read_dev_f32(&d_mix, t * e);
        let got_inj_f32 = read_dev_f32(&d_inj_f32, t * hc);
        let got_inj_bf16 = read_dev_f32(&d_inj_bf16, t * hc);
        let got_r_nosg = read_dev_f32(&d_r_nosg, t * hc * e);
        let got_r_sg = read_dev_f32(&d_r, t * hc * e);
        let got_r_add = read_dev_f32(&d_r_add, t * hc * e);

        // CPU references
        let mut exp_r = vec![0.0f32; t * hc * e];
        let mut exp_xn = vec![0.0f32; t * hc * e];
        let mut exp_lo = vec![0.0f32; t * lr];
        let mut exp_mix = vec![0.0f32; t * e];
        let mut exp_inj_f32 = vec![0.0f32; t * hc];
        let mut exp_inj_bf16 = vec![0.0f32; t * hc];
        let mut exp_r_nosg = vec![0.0f32; t * hc * e];
        let mut exp_r_sg = vec![0.0f32; t * hc * e];
        let mut exp_r_add = vec![0.0f32; t * hc * e];

        for ti in 0..t {
            for c in 0..hc {
                let mut ss = 0.0f64;
                for d in 0..e {
                    let v = emb[ti * e + d];
                    exp_r[(ti * hc + c) * e + d] = v;
                    exp_r_add[(ti * hc + c) * e + d] = add_h[(ti * hc + c) * e + d] + v;
                    ss += (v * v) as f64;
                }
                let sc = 1.0f32 / ((ss as f32) / (e as f32) + eps).sqrt();
                for d in 0..e {
                    exp_xn[(ti * hc + c) * e + d] = emb[ti * e + d] * sc * w_norm[c * e + d];
                }
            }
            for d in 0..lr {
                exp_lo[ti * lr + d] = silu_f32(lo_in[ti * lr + d] / hc as f32);
            }
            for d in 0..e {
                let mut sum = 0.0f32;
                for c in 0..hc {
                    let off = (ti * hc + c) * e + d;
                    sum += exp_xn[off] * sigmoid_f32(up_logits[off]);
                }
                exp_mix[ti * e + d] = sum / hc as f32;
            }
            for c in 0..hc {
                let mut s_f32 = 0.0f64;
                let mut s_bf16 = 0.0f64;
                for i in 0..hc * e {
                    let xv = exp_xn[ti * hc * e + i] as f64;
                    s_f32 += xv * w_inject_f32[c * hc * e + i] as f64;
                    s_bf16 += xv * w_inject_bf16_f32[c * hc * e + i] as f64;
                }
                exp_inj_f32[ti * hc + c] = s_f32 as f32;
                exp_inj_bf16[ti * hc + c] = s_bf16 as f32;

                let wc = 2.0f32 * sigmoid_f32(exp_inj_f32[ti * hc + c] / hc as f32);
                let sg = sigmoid_f32(sh_gate[ti]);
                for d in 0..e {
                    let off = (ti * hc + c) * e + d;
                    let b_idx = ti * e + d;
                    exp_r_nosg[off] = exp_r[off] + block_out[b_idx] * wc;
                    exp_r_sg[off] = exp_r[off] + (block_out[b_idx] + sh_out[b_idx] * sg) * wc;
                }
            }
        }

        assert!(max_abs_err(&got_xn, &exp_xn) < 1e-4);
        assert!(max_abs_err(&got_lo, &exp_lo) < 1e-4);
        assert!(max_abs_err(&got_mix, &exp_mix) < 1e-4);
        assert!(max_abs_err(&got_inj_f32, &exp_inj_f32) < 1e-4);
        assert!(max_abs_err(&got_inj_bf16, &exp_inj_bf16) < 1e-4);
        assert!(max_abs_err(&got_r_nosg, &exp_r_nosg) < 1e-4);
        assert!(max_abs_err(&got_r_sg, &exp_r_sg) < 1e-4);
        assert!(max_abs_err(&got_r_add, &exp_r_add) < 1e-5);

        // Verify fused q8_1 stores match quantize_q8_1
        let check_q8 = |a: &DeviceBuffer, b: &DeviceBuffer, n_elems: usize, tag: &str| {
            let ha = dequant_q8_1_blocks(&read_dev_u8(a, (n_elems / 32) * 36), n_elems);
            let hb = dequant_q8_1_blocks(&read_dev_u8(b, (n_elems / 32) * 36), n_elems);
            let er = rel_err(&ha, &hb);
            assert!(er < 1e-3, "{tag} q8 mismatch rel={er}");
        };
        check_q8(&d_xn_q8, &d_xn_q8_ref, t * hc * e, "hc_rmsnorm");
        check_q8(&d_lo_q8, &d_lo_q8_ref, t * lr, "hc_down_silu");
        check_q8(&d_mix_q8, &d_mix_q8_ref, t * e, "hc_mix");
    }

    // General rmsnorm_f32 at cols=2560 and cols=10240 with fused q8_1
    for &(rows, cols) in &[(3usize, 2560usize), (2usize, 10240usize)] {
        let x: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..cols).map(|_| rng.next_f32_range(0.5, 1.5)).collect();
        let d_x = to_dev_f32(&x);
        let d_w = to_dev_f32(&w);
        let d_y = DeviceBuffer::alloc(0, rows * cols * 4).unwrap();
        let d_yq8 = DeviceBuffer::alloc(0, rows * (cols / 32) * 36).unwrap();
        let d_yq8_ref = DeviceBuffer::alloc(0, rows * (cols / 32) * 36).unwrap();
        unsafe {
            qwen4::rmsnorm_f32(
                d_x.ptr as *const f32,
                d_w.ptr as *const f32,
                rows as i32,
                cols as i32,
                eps,
                d_y.ptr as *mut f32,
                d_yq8.ptr,
                &stream,
            );
            launch_quantize_q8_1(d_y.ptr as *const f32, d_yq8_ref.ptr, rows as i32, cols as i32, stream.raw);
        }
        stream.sync().unwrap();
        let got_y = read_dev_f32(&d_y, rows * cols);
        let mut exp_y = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let ss: f64 = x[r * cols..(r + 1) * cols].iter().map(|&v| (v * v) as f64).sum();
            let sc = 1.0f32 / ((ss as f32) / (cols as f32) + eps).sqrt();
            for c in 0..cols {
                exp_y[r * cols + c] = x[r * cols + c] * sc * w[c];
            }
        }
        assert!(max_abs_err(&got_y, &exp_y) < 1e-4);
        let q_got = dequant_q8_1_blocks(&read_dev_u8(&d_yq8, rows * (cols / 32) * 36), rows * cols);
        let q_ref = dequant_q8_1_blocks(&read_dev_u8(&d_yq8_ref, rows * (cols / 32) * 36), rows * cols);
        assert!(rel_err(&q_got, &q_ref) < 1e-3);
    }

    // PLE Post-Ops: T=1 stepped 12x vs T=12 batched + snap_opt at t=5 + CPU FP32 reference
    let n = 2560usize;
    let h = 4usize;
    let d_total = n * h; // 10240
    let t_ple = 12usize;
    let key_init: Vec<f32> = (0..t_ple * d_total).map(|_| rng.next_f32()).collect();
    let hid_init: Vec<f32> = (0..t_ple * d_total).map(|_| rng.next_f32()).collect();
    let val_init: Vec<f32> = (0..t_ple * n).map(|_| rng.next_f32()).collect();
    let hist_init: Vec<f32> = (0..d_total * 9).map(|_| rng.next_f32()).collect();
    let w_nk: Vec<f32> = (0..h * n).map(|_| rng.next_f32_range(0.5, 1.5)).collect();
    let w_nq: Vec<f32> = (0..h * n).map(|_| rng.next_f32_range(0.5, 1.5)).collect();
    let w_nc: Vec<f32> = (0..h * n).map(|_| rng.next_f32_range(0.5, 1.5)).collect();
    let w_conv: Vec<f32> = (0..d_total * 4).map(|_| rng.next_f32_range(-0.5, 0.5)).collect();

    let d_wnk = to_dev_f32(&w_nk);
    let d_wnq = to_dev_f32(&w_nq);
    let d_wnc = to_dev_f32(&w_nc);
    let d_wconv = to_dev_f32(&w_conv);

    // Path A: T=1 stepped 12x
    let d_key_step = to_dev_f32(&key_init);
    let d_hid_step = to_dev_f32(&hid_init);
    let d_val_step = to_dev_f32(&val_init);
    let d_hist_step = to_dev_f32(&hist_init);
    let d_sqn = DeviceBuffer::alloc(0, t_ple * d_total * 4).unwrap();
    let d_sgated = DeviceBuffer::alloc(0, t_ple * d_total * 4).unwrap();
    let d_sgate = DeviceBuffer::alloc(0, t_ple * h * 4).unwrap();
    let mut hist_at_t5_step = vec![0.0f32; d_total * 9];

    for ti in 0..t_ple {
        unsafe {
            qwen4::ple_postops(
                (d_key_step.ptr as *mut f32).add(ti * d_total),
                (d_hid_step.ptr as *mut f32).add(ti * d_total),
                (d_val_step.ptr as *const f32).add(ti * n),
                d_hist_step.ptr as *mut f32,
                d_wnk.ptr as *const f32,
                d_wnq.ptr as *const f32,
                d_wnc.ptr as *const f32,
                d_wconv.ptr as *const f32,
                d_sqn.ptr as *mut f32,
                d_sgated.ptr as *mut f32,
                d_sgate.ptr as *mut f32,
                1,
                n as i32,
                h as i32,
                eps,
                std::ptr::null_mut(),
                &stream,
            );
        }
        if ti == 5 {
            stream.sync().unwrap();
            hist_at_t5_step = read_dev_f32(&d_hist_step, d_total * 9);
        }
    }
    stream.sync().unwrap();
    let got_hid_step = read_dev_f32(&d_hid_step, t_ple * d_total);
    let got_hist_step = read_dev_f32(&d_hist_step, d_total * 9);

    // Path B: T=12 batched with snap
    let d_key_bat = to_dev_f32(&key_init);
    let d_hid_bat = to_dev_f32(&hid_init);
    let d_val_bat = to_dev_f32(&val_init);
    let d_hist_bat = to_dev_f32(&hist_init);
    let d_snap = DeviceBuffer::alloc(0, t_ple * d_total * 9 * 4).unwrap();
    unsafe {
        qwen4::ple_postops(
            d_key_bat.ptr as *mut f32,
            d_hid_bat.ptr as *mut f32,
            d_val_bat.ptr as *const f32,
            d_hist_bat.ptr as *mut f32,
            d_wnk.ptr as *const f32,
            d_wnq.ptr as *const f32,
            d_wnc.ptr as *const f32,
            d_wconv.ptr as *const f32,
            d_sqn.ptr as *mut f32,
            d_sgated.ptr as *mut f32,
            d_sgate.ptr as *mut f32,
            t_ple as i32,
            n as i32,
            h as i32,
            eps,
            d_snap.ptr as *mut f32,
            &stream,
        );
    }
    stream.sync().unwrap();
    let got_hid_bat = read_dev_f32(&d_hid_bat, t_ple * d_total);
    let got_hist_bat = read_dev_f32(&d_hist_bat, d_total * 9);
    let got_snap = read_dev_f32(&d_snap, t_ple * d_total * 9);

    // CPU reference
    let mut cpu_hid = hid_init.clone();
    let mut cpu_hist = hist_init.clone();
    for ti in 0..t_ple {
        let mut qn = vec![0.0f32; d_total];
        let mut gated = vec![0.0f32; d_total];
        for c in 0..h {
            let k_slice = &key_init[ti * d_total + c * n..ti * d_total + (c + 1) * n];
            let r_slice = &cpu_hid[ti * d_total + c * n..ti * d_total + (c + 1) * n];
            let ssq_k: f64 = k_slice.iter().map(|&v| (v * v) as f64).sum();
            let ssq_r: f64 = r_slice.iter().map(|&v| (v * v) as f64).sum();
            let sk = 1.0f32 / ((ssq_k as f32) / n as f32 + eps).sqrt();
            let sr = 1.0f32 / ((ssq_r as f32) / n as f32 + eps).sqrt();
            let mut dot = 0.0f64;
            for d in 0..n {
                let kd = k_slice[d] * sk * w_nk[c * n + d];
                let qd = r_slice[d] * sr * w_nq[c * n + d];
                dot += (kd * qd) as f64;
            }
            let s = (dot as f32) / (n as f32).sqrt();
            let sign = if s > 0.0 { 1.0 } else if s < 0.0 { -1.0 } else { 0.0 };
            let g = sigmoid_f32(sign * s.abs().max(1e-6).sqrt());
            for d in 0..n {
                gated[c * n + d] = val_init[ti * n + d] * g;
            }
            let ssq_g: f64 = gated[c * n..(c + 1) * n].iter().map(|&v| (v * v) as f64).sum();
            let sg = 1.0f32 / ((ssq_g as f32) / n as f32 + eps).sqrt();
            for d in 0..n {
                qn[c * n + d] = gated[c * n + d] * sg * w_nc[c * n + d];
            }
        }
        for ch in 0..d_total {
            let sum = cpu_hist[ch * 9 + 0] * w_conv[ch * 4 + 0]
                + cpu_hist[ch * 9 + 3] * w_conv[ch * 4 + 1]
                + cpu_hist[ch * 9 + 6] * w_conv[ch * 4 + 2]
                + qn[ch] * w_conv[ch * 4 + 3];
            cpu_hid[ti * d_total + ch] += gated[ch] + silu_f32(sum);
            for r in 0..8 {
                cpu_hist[ch * 9 + r] = cpu_hist[ch * 9 + r + 1];
            }
            cpu_hist[ch * 9 + 8] = qn[ch];
        }
    }

    assert!(max_abs_err(&got_hid_step, &cpu_hid) < 1e-4);
    assert!(max_abs_err(&got_hid_bat, &cpu_hid) < 1e-4);
    assert!(max_abs_err(&got_hid_step, &got_hid_bat) < 1e-5);
    assert!(max_abs_err(&got_hist_bat, &cpu_hist) < 1e-5);
    assert!(max_abs_err(&got_hist_step, &got_hist_bat) < 1e-6);
    assert!(
        max_abs_err(&got_snap[5 * d_total * 9..6 * d_total * 9], &hist_at_t5_step) < 1e-6,
        "PLE snap_opt rollback mismatch"
    );
    println!("HC-4 and PLE parity PASSED");
}

// ============================================================================
// Test 3: GDN ab/gate_beta, conv_l2, decode, verify, TC prefill + T=2048 timing
// ============================================================================

#[test]
fn test_qwen4_gdn_decode_verify_and_tc_prefill_parity() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x6D40_7C00_0000_0003);

    let h_k = 16usize;
    let h_v = 48usize;
    let s_dim = 128usize;
    let n_embd = 2560usize;
    let channels = (2 * h_k + h_v) * s_dim; // 10240
    let eps = 1e-6f32;
    let t_len = 100usize; // non-multiple of 32!

    assert_eq!(qwen4::gdn_tc_scratch_bytes(h_k as i32, h_v as i32), 60_305_408);

    // 1) gdn_ab_f32 & gdn_gate_beta_postop
    for &t in &[1usize, 17usize] {
        let x: Vec<f32> = (0..t * n_embd).map(|_| rng.next_f32_range(-0.2, 0.2)).collect();
        let wa: Vec<f32> = (0..h_v * n_embd).map(|_| rng.next_f32_range(-0.1, 0.1)).collect();
        let wb: Vec<f32> = (0..h_v * n_embd).map(|_| rng.next_f32_range(-0.1, 0.1)).collect();
        let dt_b: Vec<f32> = (0..h_v).map(|_| rng.next_f32_range(0.0, 2.0)).collect();
        let ssm_a: Vec<f32> = (0..h_v).map(|_| rng.next_f32_range(-0.4, -0.05)).collect();

        let d_x = to_dev_f32(&x);
        let d_wa = to_dev_f32(&wa);
        let d_wb = to_dev_f32(&wb);
        let d_dtb = to_dev_f32(&dt_b);
        let d_a = to_dev_f32(&ssm_a);
        let d_g = DeviceBuffer::alloc(0, t * h_v * 4).unwrap();
        let d_b = DeviceBuffer::alloc(0, t * h_v * 4).unwrap();
        unsafe {
            qwen4::gdn_ab_f32(
                d_x.ptr as *const f32,
                d_wa.ptr as *const f32,
                d_wb.ptr as *const f32,
                d_dtb.ptr as *const f32,
                d_a.ptr as *const f32,
                d_g.ptr as *mut f32,
                d_b.ptr as *mut f32,
                t as i32,
                n_embd as i32,
                h_v as i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got_g = read_dev_f32(&d_g, t * h_v);
        let got_b = read_dev_f32(&d_b, t * h_v);

        let mut raw_a = vec![0.0f32; t * h_v];
        let mut raw_b = vec![0.0f32; t * h_v];
        let mut exp_g = vec![0.0f32; t * h_v];
        let mut exp_b = vec![0.0f32; t * h_v];
        for ti in 0..t {
            for hi in 0..h_v {
                let mut sa = 0.0f64;
                let mut sb = 0.0f64;
                for d in 0..n_embd {
                    let xv = x[ti * n_embd + d] as f64;
                    sa += xv * wa[hi * n_embd + d] as f64;
                    sb += xv * wb[hi * n_embd + d] as f64;
                }
                raw_a[ti * h_v + hi] = sa as f32;
                raw_b[ti * h_v + hi] = sb as f32;
                let v = raw_a[ti * h_v + hi] + dt_b[hi];
                let sp = if v > 20.0 { v } else { v.exp().ln_1p() };
                exp_g[ti * h_v + hi] = sp * ssm_a[hi];
                exp_b[ti * h_v + hi] = sigmoid_f32(raw_b[ti * h_v + hi]);
            }
        }
        assert!(max_abs_err(&got_g, &exp_g) < 1e-4);
        assert!(max_abs_err(&got_b, &exp_b) < 1e-4);

        let d_ra = to_dev_f32(&raw_a);
        let d_rb = to_dev_f32(&raw_b);
        unsafe {
            qwen4::gdn_gate_beta_postop(
                d_ra.ptr as *const f32,
                d_rb.ptr as *const f32,
                d_dtb.ptr as *const f32,
                d_a.ptr as *const f32,
                d_g.ptr as *mut f32,
                d_b.ptr as *mut f32,
                t as i32,
                h_v as i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        assert!(max_abs_err(&read_dev_f32(&d_g, t * h_v), &exp_g) < 1e-5);
        assert!(max_abs_err(&read_dev_f32(&d_b, t * h_v), &exp_b) < 1e-5);
    }

    // 2) gdn_conv_l2 (T=100 stepped vs batched + hist_snap_opt at t=37)
    let hist0: Vec<f32> = (0..channels * 3).map(|_| rng.next_f32()).collect();
    let qkv: Vec<f32> = (0..t_len * channels).map(|_| rng.next_f32()).collect();
    let conv_w: Vec<f32> = (0..channels * 4).map(|_| rng.next_f32_range(-0.5, 0.5)).collect();

    let d_qkv = to_dev_f32(&qkv);
    let d_cw = to_dev_f32(&conv_w);
    let d_hist_step = to_dev_f32(&hist0);
    let d_h_step = DeviceBuffer::alloc(0, t_len * channels * 4).unwrap();
    let mut snap37_step = vec![0.0f32; channels * 3];

    for ti in 0..t_len {
        unsafe {
            qwen4::gdn_conv_l2(
                d_hist_step.ptr as *mut f32,
                (d_qkv.ptr as *const f32).add(ti * channels),
                d_cw.ptr as *const f32,
                (d_h_step.ptr as *mut f32).add(ti * channels),
                1,
                channels as i32,
                (2 * h_k) as i32,
                eps,
                std::ptr::null_mut(),
                &stream,
            );
        }
        if ti == 37 {
            stream.sync().unwrap();
            snap37_step = read_dev_f32(&d_hist_step, channels * 3);
        }
    }
    let d_hist_bat = to_dev_f32(&hist0);
    let d_h_bat = DeviceBuffer::alloc(0, t_len * channels * 4).unwrap();
    let d_hsnap = DeviceBuffer::alloc(0, t_len * channels * 3 * 4).unwrap();
    unsafe {
        qwen4::gdn_conv_l2(
            d_hist_bat.ptr as *mut f32,
            d_qkv.ptr as *const f32,
            d_cw.ptr as *const f32,
            d_h_bat.ptr as *mut f32,
            t_len as i32,
            channels as i32,
            (2 * h_k) as i32,
            eps,
            d_hsnap.ptr as *mut f32,
            &stream,
        );
    }
    stream.sync().unwrap();

    let got_h_step = read_dev_f32(&d_h_step, t_len * channels);
    let got_h_bat = read_dev_f32(&d_h_bat, t_len * channels);
    let got_hist_bat = read_dev_f32(&d_hist_bat, channels * 3);
    let got_hsnap = read_dev_f32(&d_hsnap, t_len * channels * 3);

    // CPU ref for conv_l2
    let mut cpu_hist = hist0.clone();
    let mut cpu_h = vec![0.0f32; t_len * channels];
    for ti in 0..t_len {
        for hd in 0..(2 * h_k + h_v) {
            let mut ys = [0.0f32; 128];
            let mut ss = 0.0f64;
            for d in 0..s_dim {
                let c = hd * s_dim + d;
                let s = cpu_hist[c * 3 + 0] * conv_w[c * 4 + 0]
                    + cpu_hist[c * 3 + 1] * conv_w[c * 4 + 1]
                    + cpu_hist[c * 3 + 2] * conv_w[c * 4 + 2]
                    + qkv[ti * channels + c] * conv_w[c * 4 + 3];
                let y = silu_f32(s);
                ys[d] = y;
                ss += (y * y) as f64;
                cpu_hist[c * 3 + 0] = cpu_hist[c * 3 + 1];
                cpu_hist[c * 3 + 1] = cpu_hist[c * 3 + 2];
                cpu_hist[c * 3 + 2] = qkv[ti * channels + c];
            }
            let sc = if hd < 2 * h_k { 1.0f32 / ((ss as f32) + eps).sqrt() } else { 1.0f32 };
            for d in 0..s_dim {
                cpu_h[ti * channels + hd * s_dim + d] = ys[d] * sc;
            }
        }
    }
    assert!(max_abs_err(&got_h_step, &cpu_h) < 1e-4);
    assert!(max_abs_err(&got_h_bat, &cpu_h) < 1e-4);
    assert!(max_abs_err(&got_hist_bat, &cpu_hist) < 1e-6);
    assert!(max_abs_err(&got_hsnap[37 * channels * 3..38 * channels * 3], &snap37_step) < 1e-6);

    // 3) GDN Recurrence: decode (1x100) vs verify (5x20) vs gdn_tc_prefill (y_f32 + y_f16) vs CPU FP32
    let state0: Vec<f32> = (0..h_v * s_dim * s_dim).map(|_| rng.next_f32_range(-0.2, 0.2)).collect();
    let gate: Vec<f32> = (0..t_len * h_v).map(|_| rng.next_f32_range(-0.35, -0.02)).collect();
    let beta: Vec<f32> = (0..t_len * h_v).map(|_| rng.next_f32_range(0.05, 0.90)).collect();
    let z_gate: Vec<f32> = (0..t_len * h_v * s_dim).map(|_| rng.next_f32_range(-2.0, 2.0)).collect();
    let ssm_norm_w: Vec<f32> = (0..s_dim).map(|_| rng.next_f32_range(0.5, 1.5)).collect();

    // CPU reference recurrence on got_h_bat (L2-normalized Q/K, raw SiLU V)
    let mut cpu_state = state0.clone();
    let mut cpu_y = vec![0.0f32; t_len * h_v * s_dim];
    let mut cpu_state_t2_chunk1 = vec![0.0f32; h_v * s_dim * s_dim];
    let out_scale = 1.0f32 / (s_dim as f32).sqrt();
    for ti in 0..t_len {
        let ht = &got_h_bat[ti * channels..(ti + 1) * channels];
        for vh in 0..h_v {
            let qh = vh % h_k;
            let q = &ht[qh * s_dim..(qh + 1) * s_dim];
            let k = &ht[(h_k + qh) * s_dim..(h_k + qh + 1) * s_dim];
            let v = &ht[(2 * h_k + vh) * s_dim..(2 * h_k + vh + 1) * s_dim];
            let g = gate[ti * h_v + vh].exp();
            let b = beta[ti * h_v + vh];
            let sb = vh * s_dim * s_dim;
            let mut o = [0.0f32; 128];
            for col in 0..s_dim {
                let mut kv = 0.0f64;
                for row in 0..s_dim {
                    kv += (cpu_state[sb + row * s_dim + col] as f64) * (k[row] as f64);
                }
                let delta = (v[col] - g * (kv as f32)) * b;
                let mut os = 0.0f64;
                for row in 0..s_dim {
                    let s_new = g * cpu_state[sb + row * s_dim + col] + k[row] * delta;
                    cpu_state[sb + row * s_dim + col] = s_new;
                    os += (s_new as f64) * (q[row] as f64);
                }
                o[col] = (os as f32) * out_scale;
            }
            let ss: f64 = o.iter().map(|&x| (x * x) as f64).sum();
            let inv_rms = 1.0f32 / ((ss as f32) / s_dim as f32 + eps).sqrt();
            for col in 0..s_dim {
                let yoff = (ti * h_v + vh) * s_dim + col;
                cpu_y[yoff] = o[col] * inv_rms * ssm_norm_w[col] * sigmoid_f32(z_gate[yoff]);
            }
        }
        if ti == 7 {
            cpu_state_t2_chunk1.copy_from_slice(&cpu_state);
        }
    }

    let d_gate = to_dev_f32(&gate);
    let d_beta = to_dev_f32(&beta);
    let d_z = to_dev_f32(&z_gate);
    let d_nw = to_dev_f32(&ssm_norm_w);

    // (b) Decode 100 steps
    let d_st_dec = to_dev_f32(&state0);
    let d_y_dec = DeviceBuffer::alloc(0, t_len * h_v * s_dim * 4).unwrap();
    let d_yq8_dec = DeviceBuffer::alloc(0, t_len * h_v * (s_dim / 32) * 36).unwrap();
    let d_yq8_ref = DeviceBuffer::alloc(0, t_len * h_v * (s_dim / 32) * 36).unwrap();
    for ti in 0..t_len {
        unsafe {
            qwen4::gdn_decode_step_norm_q8(
                d_st_dec.ptr as *mut f32,
                (d_h_bat.ptr as *const f32).add(ti * channels),
                (d_gate.ptr as *const f32).add(ti * h_v),
                (d_beta.ptr as *const f32).add(ti * h_v),
                (d_z.ptr as *const f32).add(ti * h_v * s_dim),
                d_nw.ptr as *const f32,
                eps,
                (d_y_dec.ptr as *mut f32).add(ti * h_v * s_dim),
                d_yq8_dec.ptr.add(ti * h_v * (s_dim / 32) * 36),
                h_k as i32,
                h_v as i32,
                &stream,
            );
        }
    }
    unsafe {
        launch_quantize_q8_1(
            d_y_dec.ptr as *const f32,
            d_yq8_ref.ptr,
            (t_len * h_v) as i32,
            s_dim as i32,
            stream.raw,
        );
    }
    stream.sync().unwrap();
    let got_y_dec = read_dev_f32(&d_y_dec, t_len * h_v * s_dim);
    let got_st_dec = read_dev_f32(&d_st_dec, h_v * s_dim * s_dim);
    assert!(rel_err(&got_y_dec, &cpu_y) < 3e-4);
    assert!(rel_err(&got_st_dec, &cpu_state) < 3e-4);
    let q_dec = dequant_q8_1_blocks(&read_dev_u8(&d_yq8_dec, t_len * h_v * 4 * 36), t_len * h_v * s_dim);
    let q_ref = dequant_q8_1_blocks(&read_dev_u8(&d_yq8_ref, t_len * h_v * 4 * 36), t_len * h_v * s_dim);
    assert!(rel_err(&q_dec, &q_ref) < 1e-3);

    // (c) Verify 20 chunks of 5 steps + state_snap_opt rollback at chunk 1, step 2 (ti=7)
    let d_st_ver = to_dev_f32(&state0);
    let d_y_ver = DeviceBuffer::alloc(0, t_len * h_v * s_dim * 4).unwrap();
    let d_snap_ver = DeviceBuffer::alloc(0, 5 * h_v * s_dim * s_dim * 4).unwrap();
    let mut got_snap_t7 = vec![0.0f32; h_v * s_dim * s_dim];
    for c in 0..20 {
        let t0 = c * 5;
        unsafe {
            qwen4::gdn_verify_steps_norm_q8(
                d_st_ver.ptr as *mut f32,
                (d_h_bat.ptr as *const f32).add(t0 * channels),
                (d_gate.ptr as *const f32).add(t0 * h_v),
                (d_beta.ptr as *const f32).add(t0 * h_v),
                (d_z.ptr as *const f32).add(t0 * h_v * s_dim),
                d_nw.ptr as *const f32,
                eps,
                (d_y_ver.ptr as *mut f32).add(t0 * h_v * s_dim),
                std::ptr::null_mut(),
                if c == 1 { d_snap_ver.ptr as *mut f32 } else { std::ptr::null_mut() },
                5,
                h_k as i32,
                h_v as i32,
                &stream,
            );
        }
        if c == 1 {
            stream.sync().unwrap();
            let snaps = read_dev_f32(&d_snap_ver, 5 * h_v * s_dim * s_dim);
            let one = h_v * s_dim * s_dim;
            got_snap_t7.copy_from_slice(&snaps[2 * one..3 * one]);
        }
    }
    stream.sync().unwrap();
    let got_y_ver = read_dev_f32(&d_y_ver, t_len * h_v * s_dim);
    let got_st_ver = read_dev_f32(&d_st_ver, h_v * s_dim * s_dim);
    assert!(rel_err(&got_y_ver, &cpu_y) < 3e-4);
    assert!(rel_err(&got_st_ver, &cpu_state) < 3e-4);
    assert!(rel_err(&got_snap_t7, &cpu_state_t2_chunk1) < 3e-4);

    // (c2) gdn_steps_split_norm_q8 (T=1 and T=5 with pre-step snapshot) + gdn_replay_steps rollback
    let d_st_sp1 = to_dev_f32(&state0);
    let d_yws_sp = DeviceBuffer::alloc(0, 5 * h_v * s_dim * 4).unwrap();
    let d_y_sp1 = DeviceBuffer::alloc(0, t_len * h_v * s_dim * 4).unwrap();
    let d_yq8_sp1 = DeviceBuffer::alloc(0, t_len * h_v * (s_dim / 32) * 36).unwrap();
    for ti in 0..t_len {
        unsafe {
            qwen4::gdn_steps_split_norm_q8(
                d_st_sp1.ptr as *mut f32,
                (d_h_bat.ptr as *const f32).add(ti * channels),
                (d_gate.ptr as *const f32).add(ti * h_v),
                (d_beta.ptr as *const f32).add(ti * h_v),
                (d_z.ptr as *const f32).add(ti * h_v * s_dim),
                d_nw.ptr as *const f32,
                eps,
                d_yws_sp.ptr as *mut f32,
                (d_y_sp1.ptr as *mut f32).add(ti * h_v * s_dim),
                d_yq8_sp1.ptr.add(ti * h_v * (s_dim / 32) * 36),
                std::ptr::null_mut(),
                1,
                h_k as i32,
                h_v as i32,
                &stream,
            );
        }
    }
    let d_st_sp5 = to_dev_f32(&state0);
    let d_y_sp5 = DeviceBuffer::alloc(0, t_len * h_v * s_dim * 4).unwrap();
    let d_yq8_sp5 = DeviceBuffer::alloc(0, t_len * h_v * (s_dim / 32) * 36).unwrap();
    let d_sb_chunk1 = DeviceBuffer::alloc(0, h_v * s_dim * s_dim * 4).unwrap();
    let d_st_replay = DeviceBuffer::alloc(0, h_v * s_dim * s_dim * 4).unwrap();
    for c in 0..20 {
        let t0 = c * 5;
        unsafe {
            qwen4::gdn_steps_split_norm_q8(
                d_st_sp5.ptr as *mut f32,
                (d_h_bat.ptr as *const f32).add(t0 * channels),
                (d_gate.ptr as *const f32).add(t0 * h_v),
                (d_beta.ptr as *const f32).add(t0 * h_v),
                (d_z.ptr as *const f32).add(t0 * h_v * s_dim),
                d_nw.ptr as *const f32,
                eps,
                d_yws_sp.ptr as *mut f32,
                (d_y_sp5.ptr as *mut f32).add(t0 * h_v * s_dim),
                d_yq8_sp5.ptr.add(t0 * h_v * (s_dim / 32) * 36),
                if c == 1 { d_sb_chunk1.ptr as *mut f32 } else { std::ptr::null_mut() },
                5,
                h_k as i32,
                h_v as i32,
                &stream,
            );
            if c == 1 {
                qwen4::gdn_replay_steps(
                    d_sb_chunk1.ptr as *const f32,
                    d_st_replay.ptr as *mut f32,
                    (d_h_bat.ptr as *const f32).add(t0 * channels),
                    (d_gate.ptr as *const f32).add(t0 * h_v),
                    (d_beta.ptr as *const f32).add(t0 * h_v),
                    3, // ti = 5, 6, 7 -> matches cpu_state_t2_chunk1
                    h_k as i32,
                    h_v as i32,
                    &stream,
                );
            }
        }
    }
    stream.sync().unwrap();
    assert!(rel_err(&read_dev_f32(&d_y_sp1, t_len * h_v * s_dim), &cpu_y) < 3e-4);
    assert!(rel_err(&read_dev_f32(&d_st_sp1, h_v * s_dim * s_dim), &cpu_state) < 3e-4);
    assert!(rel_err(&read_dev_f32(&d_y_sp5, t_len * h_v * s_dim), &cpu_y) < 3e-4);
    assert!(rel_err(&read_dev_f32(&d_st_sp5, h_v * s_dim * s_dim), &cpu_state) < 3e-4);
    assert!(rel_err(&read_dev_f32(&d_st_replay, h_v * s_dim * s_dim), &cpu_state_t2_chunk1) < 3e-4);
    let q_sp1 = dequant_q8_1_blocks(&read_dev_u8(&d_yq8_sp1, t_len * h_v * 4 * 36), t_len * h_v * s_dim);
    let q_sp5 = dequant_q8_1_blocks(&read_dev_u8(&d_yq8_sp5, t_len * h_v * 4 * 36), t_len * h_v * s_dim);
    assert!(rel_err(&q_sp1, &q_ref) < 1e-3);
    assert!(rel_err(&q_sp5, &q_ref) < 1e-3);

    // (d) gdn_tc_prefill (both y_f32 and y_f16)
    let sc_bytes = qwen4::gdn_tc_scratch_bytes(h_k as i32, h_v as i32);
    let d_tc_sc = DeviceBuffer::alloc(0, sc_bytes).unwrap();
    let d_st_tc = to_dev_f32(&state0);
    let d_y32_tc = DeviceBuffer::alloc(0, t_len * h_v * s_dim * 4).unwrap();
    let d_y16_tc = DeviceBuffer::alloc(0, t_len * h_v * s_dim * 2).unwrap();
    unsafe {
        qwen4::gdn_tc_prefill(
            d_st_tc.ptr as *mut f32,
            d_h_bat.ptr as *const f32,
            d_gate.ptr as *const f32,
            d_beta.ptr as *const f32,
            d_z.ptr as *const f32,
            d_nw.ptr as *const f32,
            eps,
            d_y32_tc.ptr as *mut f32,
            d_y16_tc.ptr,
            d_tc_sc.ptr,
            t_len as i32,
            h_k as i32,
            h_v as i32,
            &stream,
        );
    }
    stream.sync().unwrap();
    let got_y32_tc = read_dev_f32(&d_y32_tc, t_len * h_v * s_dim);
    let got_y16_raw = read_dev_u8(&d_y16_tc, t_len * h_v * s_dim * 2);
    let got_y16_tc: Vec<f32> = bytemuck::cast_slice::<u8, u16>(&got_y16_raw)
        .iter()
        .map(|&b| f16::from_bits(b).to_f32())
        .collect();
    let got_st_tc = read_dev_f32(&d_st_tc, h_v * s_dim * s_dim);

    let e_tc_y32 = rel_err(&got_y32_tc, &cpu_y);
    let e_tc_y16 = rel_err(&got_y16_tc, &cpu_y);
    let e_tc_st = rel_err(&got_st_tc, &cpu_state);
    println!("GDN T=100 parity: tc_y32={e_tc_y32:.6} tc_y16={e_tc_y16:.6} tc_state={e_tc_st:.6}");
    assert!(e_tc_y32 < 3e-4);
    assert!(e_tc_y16 < 2e-3);
    assert!(e_tc_st < 3e-4);

    // Timing T=2048 for gdn_tc_prefill on RTX 3090
    let t_bench = 2048usize;
    let d_h_2k = DeviceBuffer::alloc_zeroed(0, t_bench * channels * 4).unwrap();
    let d_g_2k = DeviceBuffer::alloc_zeroed(0, t_bench * h_v * 4).unwrap();
    let d_b_2k = DeviceBuffer::alloc_zeroed(0, t_bench * h_v * 4).unwrap();
    let d_z_2k = DeviceBuffer::alloc_zeroed(0, t_bench * h_v * s_dim * 4).unwrap();
    let d_y16_2k = DeviceBuffer::alloc(0, t_bench * h_v * s_dim * 2).unwrap();

    // Fill one valid unit-norm token pattern across 2K buffers from got_h_bat/gate/beta
    let h_tile: Vec<f32> = (0..t_bench)
        .flat_map(|i| got_h_bat[(i % t_len) * channels..(i % t_len + 1) * channels].iter().copied())
        .collect();
    let g_tile: Vec<f32> = (0..t_bench)
        .flat_map(|i| gate[(i % t_len) * h_v..(i % t_len + 1) * h_v].iter().copied())
        .collect();
    let b_tile: Vec<f32> = (0..t_bench)
        .flat_map(|i| beta[(i % t_len) * h_v..(i % t_len + 1) * h_v].iter().copied())
        .collect();
    d_h_2k.copy_from_host(bytemuck::cast_slice(&h_tile)).unwrap();
    d_g_2k.copy_from_host(bytemuck::cast_slice(&g_tile)).unwrap();
    d_b_2k.copy_from_host(bytemuck::cast_slice(&b_tile)).unwrap();

    // Warmup
    unsafe {
        qwen4::gdn_tc_prefill(
            d_st_tc.ptr as *mut f32,
            d_h_2k.ptr as *const f32,
            d_g_2k.ptr as *const f32,
            d_b_2k.ptr as *const f32,
            d_z_2k.ptr as *const f32,
            d_nw.ptr as *const f32,
            eps,
            std::ptr::null_mut(),
            d_y16_2k.ptr,
            d_tc_sc.ptr,
            t_bench as i32,
            h_k as i32,
            h_v as i32,
            &stream,
        );
    }
    stream.sync().unwrap();

    let ev0 = CudaEvent::new_timed(0).unwrap();
    let ev1 = CudaEvent::new_timed(0).unwrap();
    ev0.record(&stream).unwrap();
    for _ in 0..5 {
        unsafe {
            qwen4::gdn_tc_prefill(
                d_st_tc.ptr as *mut f32,
                d_h_2k.ptr as *const f32,
                d_g_2k.ptr as *const f32,
                d_b_2k.ptr as *const f32,
                d_z_2k.ptr as *const f32,
                d_nw.ptr as *const f32,
                eps,
                std::ptr::null_mut(),
                d_y16_2k.ptr,
                d_tc_sc.ptr,
                t_bench as i32,
                h_k as i32,
                h_v as i32,
                &stream,
            );
        }
    }
    ev1.record(&stream).unwrap();
    ev1.sync().unwrap();
    let ms = ev0.elapsed_ms(&ev1).unwrap() / 5.0;
    println!(
        "GDN TC prefill T=2048 (16/48/128, F16 fused norm out): {:.3} ms ({:.2} us/tok, {:.0} tok/s)",
        ms,
        ms * 1000.0 / t_bench as f32,
        t_bench as f32 / (ms * 1e-3)
    );
}

// ============================================================================
// Test 4: QSA IMRoPE, Indexer Append & Score/Select, INT8 KV, Decode & Prefill
// ============================================================================

#[test]
fn test_qwen4_qsa_imrope_indexer_kv_decode_and_prefill_parity() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x95A0_4444_0000_0004);

    let eps = 1e-6f32;
    let freq_base = 10_000_000.0f32;
    let n_rot = 64usize;

    // 1) qsa_norm_imrope for Q (24, stride=512, hd=256), K (2, 256, 256), Q_idx (4, 128, 128)
    let t_rope = 5usize;
    let pos_rope = vec![0i32, 3, 17, 42, 65];
    let max_mpos = 80usize;
    let mut mrope_pos3 = vec![0i32; max_mpos * 3];
    for p in 0..max_mpos {
        mrope_pos3[p * 3 + 0] = p as i32;
        mrope_pos3[p * 3 + 1] = p as i32 + 5;
        mrope_pos3[p * 3 + 2] = p as i32 + 11;
    }
    let d_pos_rope = to_dev_i32(&pos_rope);
    let d_mrope = to_dev_i32(&mrope_pos3);

    for &(n_heads, in_stride, hd) in &[(24usize, 512usize, 256usize), (2, 256, 256), (4, 128, 128)] {
        let src: Vec<f32> = (0..t_rope * n_heads * in_stride).map(|_| rng.next_f32()).collect();
        let gamma: Vec<f32> = (0..hd).map(|_| rng.next_f32_range(0.5, 1.5)).collect();
        let d_src = to_dev_f32(&src);
        let d_gamma = to_dev_f32(&gamma);
        let d_out_txt = DeviceBuffer::alloc(0, t_rope * n_heads * hd * 4).unwrap();
        let d_out_mm = DeviceBuffer::alloc(0, t_rope * n_heads * hd * 4).unwrap();

        unsafe {
            qwen4::qsa_norm_imrope(
                d_src.ptr as *const f32,
                in_stride as i32,
                d_gamma.ptr as *const f32,
                d_out_txt.ptr as *mut f32,
                t_rope as i32,
                n_heads as i32,
                hd as i32,
                n_rot as i32,
                eps,
                freq_base,
                d_pos_rope.ptr as *const i32,
                std::ptr::null(),
                &stream,
            );
            qwen4::qsa_norm_imrope(
                d_src.ptr as *const f32,
                in_stride as i32,
                d_gamma.ptr as *const f32,
                d_out_mm.ptr as *mut f32,
                t_rope as i32,
                n_heads as i32,
                hd as i32,
                n_rot as i32,
                eps,
                freq_base,
                d_pos_rope.ptr as *const i32,
                d_mrope.ptr as *const i32,
                &stream,
            );
        }
        stream.sync().unwrap();

        for (mm, d_o) in [(false, &d_out_txt), (true, &d_out_mm)] {
            let got = read_dev_f32(d_o, t_rope * n_heads * hd);
            let mut exp = vec![0.0f32; t_rope * n_heads * hd];
            for ti in 0..t_rope {
                let raw_p = pos_rope[ti] as usize;
                for h in 0..n_heads {
                    let s = &src[(ti * n_heads + h) * in_stride..(ti * n_heads + h) * in_stride + hd];
                    let ss: f64 = s.iter().map(|&v| (v * v) as f64).sum();
                    let sc = 1.0f32 / ((ss as f32) / hd as f32 + eps).sqrt();
                    let normed: Vec<f32> = (0..hd).map(|c| s[c] * sc * gamma[c]).collect();
                    let o = &mut exp[(ti * n_heads + h) * hd..(ti * n_heads + h + 1) * hd];
                    for pair in 0..n_rot / 2 {
                        let p = if mm { mrope_pos3[raw_p * 3 + (pair % 3)] } else { raw_p as i32 };
                        let theta = (p as f32) * freq_base.powf(-2.0 * (pair as f32) / (n_rot as f32));
                        let (sv, cv) = theta.sin_cos();
                        let a = normed[pair];
                        let b = normed[pair + n_rot / 2];
                        o[pair] = a * cv - b * sv;
                        o[pair + n_rot / 2] = a * sv + b * cv;
                    }
                    o[n_rot..].copy_from_slice(&normed[n_rot..]);
                }
            }
            assert!(max_abs_err(&got, &exp) < 1e-5, "imrope mm={mm} ({n_heads}, {in_stride}->{hd})");
        }
    }

    // 2) idx_append_steps vs idx_append_batch across T=2100 (>2051!)
    let t_idx = 2100usize;
    let max_cells = 2176usize;
    let max_blocks = max_cells / 4 + 1;
    let raw_k: Vec<f32> = (0..t_idx * 128).map(|_| rng.next_f32()).collect();
    let idx_gamma: Vec<f32> = (0..128).map(|_| rng.next_f32_range(0.5, 1.5)).collect();
    let d_rawk = to_dev_f32(&raw_k);
    let d_igamma = to_dev_f32(&idx_gamma);

    let pos_all: Vec<i32> = (0..t_idx as i32).collect();
    let d_pos_all = to_dev_i32(&pos_all);

    let d_tail_s = DeviceBuffer::alloc_zeroed(0, 3 * 128 * 4).unwrap();
    let d_dead_s = DeviceBuffer::alloc_zeroed(0, 128 * 4).unwrap();
    let d_pool_s = DeviceBuffer::alloc_zeroed(0, max_blocks * 128 * 4).unwrap();
    // Test append_steps with chunks of 5 and singles
    let mut off = 0usize;
    while off < t_idx {
        let step = if off < 2080 { 5 } else { 1 };
        let cnt = step.min(t_idx - off);
        unsafe {
            qwen4::idx_append_steps(
                (d_rawk.ptr as *const f32).add(off * 128),
                (d_pos_all.ptr as *const i32).add(off),
                cnt as i32,
                d_igamma.ptr as *const f32,
                eps,
                freq_base,
                std::ptr::null(),
                d_tail_s.ptr as *mut f32,
                d_dead_s.ptr as *mut f32,
                d_pool_s.ptr as *mut f32,
                max_cells as i32,
                &stream,
            );
        }
        off += cnt;
    }

    let d_tail_b = DeviceBuffer::alloc_zeroed(0, 3 * 128 * 4).unwrap();
    let d_dead_b = DeviceBuffer::alloc_zeroed(0, 128 * 4).unwrap();
    let d_pool_b = DeviceBuffer::alloc_zeroed(0, max_blocks * 128 * 4).unwrap();
    unsafe {
        qwen4::idx_append_batch(
            d_rawk.ptr as *const f32,
            2095,
            0,
            d_igamma.ptr as *const f32,
            eps,
            freq_base,
            std::ptr::null(),
            d_tail_b.ptr as *mut f32,
            d_dead_b.ptr as *mut f32,
            d_pool_b.ptr as *mut f32,
            max_cells as i32,
            &stream,
        );
        qwen4::idx_append_batch(
            (d_rawk.ptr as *const f32).add(2095 * 128),
            5,
            2095,
            d_igamma.ptr as *const f32,
            eps,
            freq_base,
            std::ptr::null(),
            d_tail_b.ptr as *mut f32,
            d_dead_b.ptr as *mut f32,
            d_pool_b.ptr as *mut f32,
            max_cells as i32,
            &stream,
        );
    }
    stream.sync().unwrap();

    let n_bid_tot = t_idx / 4; // 525 (pooled uses 0..=525)
    let got_pool_s = read_dev_f32(&d_pool_s, (n_bid_tot + 1) * 128);
    let got_pool_b = read_dev_f32(&d_pool_b, (n_bid_tot + 1) * 128);
    let got_dead_s = read_dev_f32(&d_dead_s, 128);
    let got_dead_b = read_dev_f32(&d_dead_b, 128);
    let got_tail_s = read_dev_f32(&d_tail_s, 3 * 128);
    let got_tail_b = read_dev_f32(&d_tail_b, 3 * 128);

    // CPU reference with F16 rounding of raw_k
    let mut cpu_pool = vec![0.0f32; (n_bid_tot + 1) * 128];
    let mut cpu_dead = vec![0.0f32; 128];
    let mut cpu_tail = vec![0.0f32; 3 * 128];
    for pos in 0..t_idx {
        let mut inc = [0.0f32; 128];
        for d in 0..128 {
            inc[d] = f16::from_f32(raw_k[pos * 128 + d]).to_f32();
            if pos % 4 < 3 {
                cpu_tail[(pos % 4) * 128 + d] = inc[d];
            }
        }
        if pos == 0 || pos % 4 == 3 {
            let mut mean = [0.0f32; 128];
            let mut ss = 0.0f64;
            for d in 0..128 {
                let s = if pos == 0 {
                    inc[d] + inc[d] + inc[d] + inc[d]
                } else {
                    ((cpu_tail[0 * 128 + d] + cpu_tail[1 * 128 + d]) + cpu_tail[2 * 128 + d]) + inc[d]
                };
                mean[d] = 0.25 * s;
                ss += (mean[d] * mean[d]) as f64;
            }
            let sc = 1.0f32 / ((ss as f32) / 128.0 + eps).sqrt();
            let normed: Vec<f32> = (0..128).map(|d| mean[d] * sc * idx_gamma[d]).collect();
            let b = pos / 4;
            let rp = if pos == 0 { 0 } else { 4 * b };
            let mut rot = normed.clone();
            for pair in 0..32 {
                let theta = (rp as f32) * freq_base.powf(-2.0 * (pair as f32) / 64.0);
                let (sv, cv) = theta.sin_cos();
                rot[pair] = normed[pair] * cv - normed[pair + 32] * sv;
                rot[pair + 32] = normed[pair] * sv + normed[pair + 32] * cv;
            }
            cpu_pool[b * 128..(b + 1) * 128].copy_from_slice(&rot);
            if pos == 0 {
                cpu_dead.copy_from_slice(&rot);
            } else {
                cpu_pool[(b + 1) * 128..(b + 2) * 128].copy_from_slice(&cpu_dead);
            }
        }
    }

    let mut max_i = 0usize;
    let mut max_d = 0.0f32;
    for (i, (&g, &c)) in got_pool_s.iter().zip(cpu_pool.iter()).enumerate() {
        let e = (g - c).abs();
        if e > max_d {
            max_d = e;
            max_i = i;
        }
    }
    let max_non_rope = got_pool_s
        .chunks_exact(128)
        .zip(cpu_pool.chunks_exact(128))
        .flat_map(|(g, c)| g[64..].iter().zip(c[64..].iter()))
        .map(|(&g, &c)| (g - c).abs())
        .fold(0.0f32, f32::max);
    println!(
        "idx_append T=2100: steps_vs_batch={:.6} steps_vs_cpu={:.6} (at b={} d={}) non_rope={:.6} batch_vs_cpu={:.6} dead=({:.6},{:.6}) tail=({:.6},{:.6})",
        max_abs_err(&got_pool_s, &got_pool_b),
        max_abs_err(&got_pool_s, &cpu_pool),
        max_i / 128,
        max_i % 128,
        max_non_rope,
        max_abs_err(&got_pool_b, &cpu_pool),
        max_abs_err(&got_dead_s, &cpu_dead),
        max_abs_err(&got_dead_b, &cpu_dead),
        max_abs_err(&got_tail_s, &cpu_tail),
        max_abs_err(&got_tail_b, &cpu_tail),
    );
    assert!(max_abs_err(&got_pool_s, &got_pool_b) < 1e-6);
    assert!(max_non_rope < 1e-6);
    // CUDA powf/sincosf at rope_pos up to 2096 (~1300 rad) differs from libm by <= 3e-4.
    assert!(max_abs_err(&got_pool_s, &cpu_pool) < 5e-4);
    assert!(max_abs_err(&got_pool_b, &cpu_pool) < 5e-4);
    assert!(max_abs_err(&got_dead_s, &cpu_dead) < 1e-5);
    assert!(max_abs_err(&got_dead_b, &cpu_dead) < 1e-5);
    assert!(max_abs_err(&got_tail_s, &cpu_tail) < 1e-5);
    assert!(max_abs_err(&got_tail_b, &cpu_tail) < 1e-5);

    // 3) idx_score_select at T=1, T=4 (warp path), T=32 (TF32 TC path) covering 2050, 2051, 2053, 2100
    let top_k_cap = 2051usize;
    let test_positions = [2049i32, 2050, 2052, 2099]; // n_kv = 2050, 2051, 2053, 2100
    for &tq in &[1usize, 4usize, 32usize] {
        let mut q_idx = vec![0.0f32; tq * 4 * 128];
        for v in &mut q_idx {
            *v = rng.next_f32();
        }
        let pos_q: Vec<i32> = (0..tq).map(|i| {
            if tq == 1 {
                2052 // n_kv = 2053 (%4 == 1 partial tail test!)
            } else {
                test_positions[i % test_positions.len()]
            }
        }).collect();
        let d_qidx = to_dev_f32(&q_idx);
        let d_posq = to_dev_i32(&pos_q);
        let d_scores = DeviceBuffer::alloc(0, tq * max_blocks * 4).unwrap();
        let d_ids = DeviceBuffer::alloc(0, tq * top_k_cap * 4).unwrap();
        let d_widths = DeviceBuffer::alloc(0, tq * 4).unwrap();
        unsafe {
            qwen4::idx_score_select(
                d_pool_b.ptr as *const f32,
                d_dead_b.ptr as *const f32,
                d_qidx.ptr as *const f32,
                d_posq.ptr as *const i32,
                tq as i32,
                max_blocks as i32,
                top_k_cap as i32,
                d_scores.ptr as *mut f32,
                d_ids.ptr as *mut i32,
                d_widths.ptr as *mut i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got_ids = read_dev_i32(&d_ids, tq * top_k_cap);
        let got_widths = read_dev_i32(&d_widths, tq);

        for qi in 0..tq {
            let n_kv = (pos_q[qi] + 1) as usize;
            let width = n_kv.min(2051);
            assert_eq!(got_widths[qi] as usize, width);
            let slice = &got_ids[qi * top_k_cap..qi * top_k_cap + width];
            for w in slice.windows(2) {
                assert!(w[0] < w[1], "ids not strictly ascending at qi={qi}: {} >= {}", w[0], w[1]);
            }
            if n_kv <= 2051 {
                let exp: Vec<i32> = (0..width as i32).collect();
                assert_eq!(slice, &exp[..]);
            } else {
                let n_bid = n_kv / 4;
                if n_kv % 4 != 0 {
                    assert_eq!(*slice.last().unwrap(), (n_kv - 1) as i32, "partial tail cell missing!");
                }
                let mut b_scores = Vec::with_capacity(n_bid + 1);
                for b in 0..=n_bid {
                    let kvec = if b == n_bid { &cpu_dead } else { &cpu_pool[b * 128..(b + 1) * 128] };
                    let mut sc = 0.0f32;
                    for h in 0..4 {
                        let dot: f32 = (0..128)
                            .map(|d| kvec[d] * q_idx[(qi * 4 + h) * 128 + d])
                            .sum();
                        sc += dot.max(0.0);
                    }
                    if b == n_bid && n_kv % 4 != 0 {
                        sc += 1e9f32;
                    }
                    b_scores.push((b, sc));
                }
                let mut sorted = b_scores.clone();
                sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                let mut rem = width;
                let mut exp_cells = Vec::with_capacity(width);
                let mut cutoff = 0.0f32;
                for (b, sc) in &sorted {
                    if rem == 0 {
                        break;
                    }
                    cutoff = *sc;
                    let w = if *b < n_bid { 4 } else { n_kv - n_bid * 4 };
                    let take = w.min(rem);
                    for c in 0..take {
                        exp_cells.push((b * 4 + c) as i32);
                    }
                    rem -= take;
                }
                exp_cells.sort();
                if slice != &exp_cells[..] {
                    // Only allow swaps on near-ties within TF32 tolerance (<1e-4) at cutoff
                    let got_set: std::collections::BTreeSet<i32> = slice.iter().copied().collect();
                    let exp_set: std::collections::BTreeSet<i32> = exp_cells.iter().copied().collect();
                    for &c in got_set.symmetric_difference(&exp_set) {
                        let blk = (c as usize) / 4;
                        let bsc = b_scores[blk].1;
                        assert!(
                            (bsc - cutoff).abs() < 1e-4,
                            "idx_score_select mismatch beyond tie tol: cell={c} bsc={bsc} cutoff={cutoff}"
                        );
                    }
                }
            }
        }
    }

    // 4) kv_append_q8, qsa_decode_attn_q8 (T=1, 4, dense & sparse + gate + q8), qsa_prefill_attn_q8 (T=32)
    let n_kv_heads = 2usize;
    let n_q_heads = 24usize;
    let hd = 256usize;
    let scale = 0.0625f32;
    let total_kv = 2100usize;

    let k_host: Vec<f32> = (0..total_kv * n_kv_heads * hd).map(|_| rng.next_f32_range(-2.0, 2.0)).collect();
    let v_host: Vec<f32> = (0..total_kv * n_kv_heads * hd).map(|_| rng.next_f32_range(-2.0, 2.0)).collect();
    let pos_kv: Vec<i32> = (0..total_kv as i32).collect();
    let d_k_in = to_dev_f32(&k_host);
    let d_v_in = to_dev_f32(&v_host);
    let d_pos_kv = to_dev_i32(&pos_kv);
    let d_k_q = DeviceBuffer::alloc(0, total_kv * n_kv_heads * hd).unwrap();
    let d_v_q = DeviceBuffer::alloc(0, total_kv * n_kv_heads * hd).unwrap();
    let d_k_sc = DeviceBuffer::alloc(0, total_kv * n_kv_heads * 4 * 2).unwrap();
    let d_v_sc = DeviceBuffer::alloc(0, total_kv * n_kv_heads * 4 * 2).unwrap();

    unsafe {
        qwen4::kv_append_q8(
            d_k_in.ptr as *const f32,
            d_v_in.ptr as *const f32,
            d_pos_kv.ptr as *const i32,
            total_kv as i32,
            n_kv_heads as i32,
            hd as i32,
            d_k_q.ptr as *mut i8,
            d_v_q.ptr as *mut i8,
            d_k_sc.ptr as *mut u16,
            d_v_sc.ptr as *mut u16,
            &stream,
        );
    }
    stream.sync().unwrap();

    // Read back quantized KV cache and dequantize for CPU INT8-KV attention reference
    let k_q_raw = read_dev_u8(&d_k_q, total_kv * n_kv_heads * hd);
    let v_q_raw = read_dev_u8(&d_v_q, total_kv * n_kv_heads * hd);
    let k_sc_raw = read_dev_u8(&d_k_sc, total_kv * n_kv_heads * 4 * 2);
    let v_sc_raw = read_dev_u8(&d_v_sc, total_kv * n_kv_heads * 4 * 2);
    let k_sc_u16: &[u16] = bytemuck::cast_slice(&k_sc_raw);
    let v_sc_u16: &[u16] = bytemuck::cast_slice(&v_sc_raw);

    let mut k_deq = vec![0.0f32; total_kv * n_kv_heads * hd];
    let mut v_deq = vec![0.0f32; total_kv * n_kv_heads * hd];
    for row in 0..(total_kv * n_kv_heads) {
        for d in 0..hd {
            let g = d / 64;
            k_deq[row * hd + d] = (k_q_raw[row * hd + d] as i8 as f32) * f16::from_bits(k_sc_u16[row * 4 + g]).to_f32();
            v_deq[row * hd + d] = (v_q_raw[row * hd + d] as i8 as f32) * f16::from_bits(v_sc_u16[row * 4 + g]).to_f32();
        }
    }
    assert!(rel_err(&k_deq, &k_host) < 1e-2);
    assert!(rel_err(&v_deq, &v_host) < 1e-2);

    let cpu_qsa_ref = |q: &[f32], qfull_gate: Option<&[f32]>, n_ids_per_q: &[i32], ids_opt: Option<&[i32]>, id_cap: usize| -> Vec<f32> {
        let tq = n_ids_per_q.len();
        let mut out = vec![0.0f32; tq * n_q_heads * hd];
        for qi in 0..tq {
            let n_ids = n_ids_per_q[qi] as usize;
            for h in 0..n_q_heads {
                let kvh = h / 12;
                let qv = &q[(qi * n_q_heads + h) * hd..(qi * n_q_heads + h + 1) * hd];
                let mut scores = Vec::with_capacity(n_ids);
                let mut cells = Vec::with_capacity(n_ids);
                let mut mx = f32::NEG_INFINITY;
                for c in 0..n_ids {
                    let cell = if let Some(ids) = ids_opt { ids[qi * id_cap + c] as usize } else { c };
                    let krow = (cell * n_kv_heads + kvh) * hd;
                    let dot: f64 = (0..hd).map(|d| (qv[d] as f64) * (k_deq[krow + d] as f64)).sum();
                    let s = (dot as f32) * scale;
                    mx = mx.max(s);
                    scores.push(s);
                    cells.push(cell);
                }
                let mut denom = 0.0f64;
                let mut probs = vec![0.0f32; n_ids];
                for (i, &s) in scores.iter().enumerate() {
                    let e = (s - mx).exp();
                    probs[i] = e;
                    denom += e as f64;
                }
                let inv_l = 1.0f32 / denom as f32;
                let o = &mut out[(qi * n_q_heads + h) * hd..(qi * n_q_heads + h + 1) * hd];
                for (i, &cell) in cells.iter().enumerate() {
                    let w = probs[i] * inv_l;
                    let vrow = (cell * n_kv_heads + kvh) * hd;
                    for d in 0..hd {
                        o[d] += w * v_deq[vrow + d];
                    }
                }
                if let Some(qf) = qfull_gate {
                    let goff = (qi * n_q_heads + h) * 512 + 256;
                    for d in 0..hd {
                        o[d] *= sigmoid_f32(qf[goff + d]);
                    }
                }
            }
        }
        out
    };

    // Decode attention (T=1 dense=130, T=4 sparse=2051 with fused gate + q8, T=1 multi-chunk n_ids=2100 with n_chunks=48 via cap=4096)
    for &(tq, sparse, gated, cap_ids) in &[
        (1usize, false, false, 130i32),
        (4usize, true, true, 2051i32),
        (1usize, false, true, 4096i32),
    ] {
        let q: Vec<f32> = (0..tq * n_q_heads * hd).map(|_| rng.next_f32_range(-1.0, 1.0)).collect();
        let q_full: Vec<f32> = (0..tq * n_q_heads * 512).map(|_| rng.next_f32_range(-2.0, 2.0)).collect();
        let n_ids = if sparse { 2051i32 } else if cap_ids >= 4096 { 2099i32 } else { 130i32 };
        let n_ids_v = vec![n_ids; tq];
        let id_cap = n_ids as usize;
        let ids_host: Vec<i32> = if sparse {
            (0..tq)
                .flat_map(|qi| (0..id_cap).map(move |c| ((c + qi) % total_kv) as i32))
                .collect()
        } else {
            vec![]
        };
        let d_q = to_dev_f32(&q);
        let d_qfull = to_dev_f32(&q_full);
        let d_nids = to_dev_i32(&n_ids_v);
        let d_ids = if sparse { Some(to_dev_i32(&ids_host)) } else { None };
        let ws_len = qwen4::qsa_decode_ws_floats(tq as i32, n_kv_heads as i32, cap_ids);
        let d_ws = DeviceBuffer::alloc(0, ws_len * 4).unwrap();
        let d_out = DeviceBuffer::alloc(0, tq * n_q_heads * hd * 4).unwrap();
        let d_out_q8 = DeviceBuffer::alloc(0, tq * n_q_heads * 8 * 36).unwrap();
        let d_out_q8_ref = DeviceBuffer::alloc(0, tq * n_q_heads * 8 * 36).unwrap();

        unsafe {
            qwen4::qsa_decode_attn_q8(
                d_q.ptr as *const f32,
                if gated { d_qfull.ptr as *const f32 } else { std::ptr::null() },
                d_k_q.ptr as *const i8,
                d_v_q.ptr as *const i8,
                d_k_sc.ptr as *const u16,
                d_v_sc.ptr as *const u16,
                d_ids.as_ref().map_or(std::ptr::null(), |b| b.ptr as *const i32),
                d_nids.ptr as *const i32,
                cap_ids,
                id_cap as i32,
                tq as i32,
                n_kv_heads as i32,
                scale,
                d_ws.ptr as *mut f32,
                d_out.ptr as *mut f32,
                d_out_q8.ptr,
                &stream,
            );
            launch_quantize_q8_1(
                d_out.ptr as *const f32,
                d_out_q8_ref.ptr,
                (tq * n_q_heads) as i32,
                hd as i32,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let got = read_dev_f32(&d_out, tq * n_q_heads * hd);
        let exp = cpu_qsa_ref(
            &q,
            if gated { Some(&q_full) } else { None },
            &n_ids_v,
            if sparse { Some(&ids_host) } else { None },
            id_cap,
        );
        let er = rel_err(&got, &exp);
        println!("QSA decode (T={tq}, sparse={sparse}, gated={gated}): rel={er:.6}");
        assert!(er < 3e-3);
        let qg = dequant_q8_1_blocks(&read_dev_u8(&d_out_q8, tq * n_q_heads * 8 * 36), tq * n_q_heads * hd);
        let qr = dequant_q8_1_blocks(&read_dev_u8(&d_out_q8_ref, tq * n_q_heads * 8 * 36), tq * n_q_heads * hd);
        assert!(rel_err(&qg, &qr) < 1e-3);
    }

    // Explicit 4096-token multi-chunk online softmax parity test (exercises > 1 chunk/CTA with QWEN4_QSA_MAX_CHUNKS=48)
    {
        let kv4k = 4096usize;
        let k4_h: Vec<f32> = (0..kv4k * n_kv_heads * hd).map(|_| rng.next_f32_range(-1.5, 1.5)).collect();
        let v4_h: Vec<f32> = (0..kv4k * n_kv_heads * hd).map(|_| rng.next_f32_range(-1.5, 1.5)).collect();
        let p4_h: Vec<i32> = (0..kv4k as i32).collect();
        let d_k4_in = to_dev_f32(&k4_h);
        let d_v4_in = to_dev_f32(&v4_h);
        let d_p4 = to_dev_i32(&p4_h);
        let d_k4_q = DeviceBuffer::alloc(0, kv4k * n_kv_heads * hd).unwrap();
        let d_v4_q = DeviceBuffer::alloc(0, kv4k * n_kv_heads * hd).unwrap();
        let d_k4_sc = DeviceBuffer::alloc(0, kv4k * n_kv_heads * 4 * 2).unwrap();
        let d_v4_sc = DeviceBuffer::alloc(0, kv4k * n_kv_heads * 4 * 2).unwrap();
        unsafe {
            qwen4::kv_append_q8(
                d_k4_in.ptr as *const f32,
                d_v4_in.ptr as *const f32,
                d_p4.ptr as *const i32,
                kv4k as i32,
                n_kv_heads as i32,
                hd as i32,
                d_k4_q.ptr as *mut i8,
                d_v4_q.ptr as *mut i8,
                d_k4_sc.ptr as *mut u16,
                d_v4_sc.ptr as *mut u16,
                &stream,
            );
        }
        stream.sync().unwrap();
        let kq_raw = read_dev_u8(&d_k4_q, kv4k * n_kv_heads * hd);
        let vq_raw = read_dev_u8(&d_v4_q, kv4k * n_kv_heads * hd);
        let ks_raw = read_dev_u8(&d_k4_sc, kv4k * n_kv_heads * 4 * 2);
        let vs_raw = read_dev_u8(&d_v4_sc, kv4k * n_kv_heads * 4 * 2);
        let ks_u16: &[u16] = bytemuck::cast_slice(&ks_raw);
        let vs_u16: &[u16] = bytemuck::cast_slice(&vs_raw);
        let mut kd = vec![0.0f32; kv4k * n_kv_heads * hd];
        let mut vd = vec![0.0f32; kv4k * n_kv_heads * hd];
        for row in 0..(kv4k * n_kv_heads) {
            for d in 0..hd {
                let g = d / 64;
                kd[row * hd + d] = (kq_raw[row * hd + d] as i8 as f32) * f16::from_bits(ks_u16[row * 4 + g]).to_f32();
                vd[row * hd + d] = (vq_raw[row * hd + d] as i8 as f32) * f16::from_bits(vs_u16[row * 4 + g]).to_f32();
            }
        }
        let q4: Vec<f32> = (0..n_q_heads * hd).map(|_| rng.next_f32_range(-1.0, 1.0)).collect();
        let qf4: Vec<f32> = (0..n_q_heads * 512).map(|_| rng.next_f32_range(-2.0, 2.0)).collect();
        let d_q4 = to_dev_f32(&q4);
        let d_qf4 = to_dev_f32(&qf4);
        let d_n4 = to_dev_i32(&[4096i32]);
        let ws4 = qwen4::qsa_decode_ws_floats(1, n_kv_heads as i32, 4096);
        let d_ws4 = DeviceBuffer::alloc(0, ws4 * 4).unwrap();
        let d_o4 = DeviceBuffer::alloc(0, n_q_heads * hd * 4).unwrap();
        unsafe {
            qwen4::qsa_decode_attn_q8(
                d_q4.ptr as *const f32,
                d_qf4.ptr as *const f32,
                d_k4_q.ptr as *const i8,
                d_v4_q.ptr as *const i8,
                d_k4_sc.ptr as *const u16,
                d_v4_sc.ptr as *const u16,
                std::ptr::null(),
                d_n4.ptr as *const i32,
                4096,
                4096,
                1,
                n_kv_heads as i32,
                scale,
                d_ws4.ptr as *mut f32,
                d_o4.ptr as *mut f32,
                std::ptr::null_mut(),
                &stream,
            );
        }
        stream.sync().unwrap();
        let got4 = read_dev_f32(&d_o4, n_q_heads * hd);
        let mut exp4 = vec![0.0f32; n_q_heads * hd];
        for h in 0..n_q_heads {
            let kvh = h / 12;
            let qv = &q4[h * hd..(h + 1) * hd];
            let mut mx = f32::NEG_INFINITY;
            let mut scores = Vec::with_capacity(kv4k);
            for c in 0..kv4k {
                let kr = (c * n_kv_heads + kvh) * hd;
                let s = ((0..hd).map(|d| (qv[d] as f64) * (kd[kr + d] as f64)).sum::<f64>() as f32) * scale;
                mx = mx.max(s);
                scores.push(s);
            }
            let exps: Vec<f32> = scores.iter().map(|&s| (s - mx).exp()).collect();
            let inv_l = 1.0f32 / (exps.iter().map(|&e| e as f64).sum::<f64>() as f32);
            let o = &mut exp4[h * hd..(h + 1) * hd];
            for (c, &e) in exps.iter().enumerate() {
                let w = e * inv_l;
                let vr = (c * n_kv_heads + kvh) * hd;
                for d in 0..hd {
                    o[d] += w * vd[vr + d];
                }
            }
            for d in 0..hd {
                o[d] *= sigmoid_f32(qf4[h * 512 + 256 + d]);
            }
        }
        let e4 = rel_err(&got4, &exp4);
        println!("QSA decode 4096 multi-chunk (n_chunks=48): rel={e4:.6}");
        assert!(e4 < 3e-4);
    }

    // Prefill attention (T=32, dense & sparse at n_kv=2100 with fused gate)
    for &sparse in &[false, true] {
        let tq = 32usize;
        let q: Vec<f32> = (0..tq * n_q_heads * hd).map(|_| rng.next_f32_range(-1.0, 1.0)).collect();
        let q_full: Vec<f32> = (0..tq * n_q_heads * 512).map(|_| rng.next_f32_range(-2.0, 2.0)).collect();
        let id_cap = 2051usize;
        let n_ids_v: Vec<i32> = (0..tq)
            .map(|qi| if sparse { 2051 } else { 150 + qi as i32 * 4 })
            .collect();
        let ids_host: Vec<i32> = if sparse {
            (0..tq)
                .flat_map(|qi| (0..id_cap).map(move |c| ((c + qi) % total_kv) as i32))
                .collect()
        } else {
            vec![]
        };
        let d_q = to_dev_f32(&q);
        let d_qfull = to_dev_f32(&q_full);
        let d_nids = to_dev_i32(&n_ids_v);
        let d_ids = if sparse { Some(to_dev_i32(&ids_host)) } else { None };
        let d_out = DeviceBuffer::alloc(0, tq * n_q_heads * hd * 4).unwrap();

        unsafe {
            qwen4::qsa_prefill_attn_q8(
                d_q.ptr as *const f32,
                d_qfull.ptr as *const f32,
                d_k_q.ptr as *const i8,
                d_v_q.ptr as *const i8,
                d_k_sc.ptr as *const u16,
                d_v_sc.ptr as *const u16,
                d_ids.as_ref().map_or(std::ptr::null(), |b| b.ptr as *const i32),
                d_nids.ptr as *const i32,
                id_cap as i32,
                tq as i32,
                n_kv_heads as i32,
                scale,
                d_out.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got = read_dev_f32(&d_out, tq * n_q_heads * hd);
        let exp = cpu_qsa_ref(
            &q,
            Some(&q_full),
            &n_ids_v,
            if sparse { Some(&ids_host) } else { None },
            id_cap,
        );
        let er = rel_err(&got, &exp);
        println!("QSA prefill TC (T={tq}, sparse={sparse}): rel={er:.6}");
        assert!(er < 3e-3);
    }
}

// ============================================================================
// Test 5: Router top-10, Shared Expert Scalar Gate, Vision Helpers
// ============================================================================

#[test]
fn test_qwen4_router_shared_gate_and_vision_helpers_parity() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x5217_4150_0000_0005);

    // 1) router_top10: random (T=1, T=17), exact tie-break, tiny/degenerate sum clamp
    let n_exp = 512usize;
    let topk = 10usize;
    for &t in &[1usize, 17usize] {
        let mut logits: Vec<f32> = (0..t * n_exp).map(|_| rng.next_f32_range(-6.0, 6.0)).collect();
        // Force row 0 (for T=17) or check separate tie test:
        if t == 17 {
            for e in 0..n_exp {
                logits[0 * n_exp + e] = 1.25; // exact tie across all 512 experts -> must pick 0..9
                logits[1 * n_exp + e] = if e < 10 { -200.0 } else { 0.0 }; // top-10 are still > 2^-14 collectively via exp
            }
        }
        let d_logits = to_dev_f32(&logits);
        let d_ids = DeviceBuffer::alloc(0, t * topk * 4).unwrap();
        let d_wts = DeviceBuffer::alloc(0, t * topk * 4).unwrap();
        unsafe {
            qwen4::router_top10(
                d_logits.ptr as *const f32,
                t as i32,
                n_exp as i32,
                topk as i32,
                d_ids.ptr as *mut i32,
                d_wts.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got_ids = read_dev_i32(&d_ids, t * topk);
        let got_wts = read_dev_f32(&d_wts, t * topk);

        for ti in 0..t {
            let row = &logits[ti * n_exp..(ti + 1) * n_exp];
            let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut exps: Vec<(usize, f32)> = row.iter().enumerate().map(|(i, &v)| (i, (v - mx).exp())).collect();
            let tot: f32 = exps.iter().map(|p| p.1).sum();
            for p in &mut exps {
                p.1 /= tot;
            }
            exps.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let sel_sum: f32 = exps[..topk].iter().map(|p| p.1).sum::<f32>().max(6.103515625e-5);
            for k in 0..topk {
                assert_eq!(got_ids[ti * topk + k] as usize, exps[k].0, "router id mismatch t={ti} k={k}");
                assert!(
                    (got_wts[ti * topk + k] - exps[k].1 / sel_sum).abs() < 1e-5,
                    "router weight mismatch t={ti} k={k}"
                );
            }
            if t == 17 && ti == 0 {
                assert_eq!(&got_ids[0..10], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
            }
        }
    }

    // Extreme dynamic-range finite logits test (1 dominant expert +0 vs 511 -100 logits)
    {
        let mut ext_logits = vec![-100.0f32; n_exp];
        ext_logits[317] = 20.0;
        let d_logits = to_dev_f32(&ext_logits);
        let d_ids = DeviceBuffer::alloc(0, topk * 4).unwrap();
        let d_wts = DeviceBuffer::alloc(0, topk * 4).unwrap();
        unsafe {
            qwen4::router_top10(
                d_logits.ptr as *const f32,
                1,
                n_exp as i32,
                topk as i32,
                d_ids.ptr as *mut i32,
                d_wts.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let ids = read_dev_i32(&d_ids, topk);
        let wts = read_dev_f32(&d_wts, topk);
        assert_eq!(ids[0], 317);
        assert!((wts[0] - 1.0).abs() < 1e-6);
        assert!(wts.iter().all(|v| v.is_finite()));
    }

    // 2) shared_gate_dot (F32 & BF16, T=1 & T=8)
    let n_embd = 2560usize;
    let w_f32: Vec<f32> = (0..n_embd).map(|_| rng.next_f32()).collect();
    let w_bf16_u16: Vec<u16> = w_f32.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
    let w_bf16_f32: Vec<f32> = w_bf16_u16.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
    let d_wf32 = to_dev_f32(&w_f32);
    let d_wbf16 = to_dev_u16(&w_bf16_u16);

    for &t in &[1usize, 8usize] {
        let x: Vec<f32> = (0..t * n_embd).map(|_| rng.next_f32()).collect();
        let d_x = to_dev_f32(&x);
        let d_o1 = DeviceBuffer::alloc(0, t * 4).unwrap();
        let d_o2 = DeviceBuffer::alloc(0, t * 4).unwrap();
        unsafe {
            qwen4::shared_gate_dot(
                d_x.ptr as *const f32,
                d_wf32.ptr,
                GgmlType::F32 as i32,
                t as i32,
                n_embd as i32,
                d_o1.ptr as *mut f32,
                &stream,
            );
            qwen4::shared_gate_dot(
                d_x.ptr as *const f32,
                d_wbf16.ptr,
                GgmlType::Bf16 as i32,
                t as i32,
                n_embd as i32,
                d_o2.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got1 = read_dev_f32(&d_o1, t);
        let got2 = read_dev_f32(&d_o2, t);
        let exp1: Vec<f32> = (0..t)
            .map(|ti| (0..n_embd).map(|d| (x[ti * n_embd + d] as f64) * (w_f32[d] as f64)).sum::<f64>() as f32)
            .collect();
        let exp2: Vec<f32> = (0..t)
            .map(|ti| (0..n_embd).map(|d| (x[ti * n_embd + d] as f64) * (w_bf16_f32[d] as f64)).sum::<f64>() as f32)
            .collect();
        assert!(max_abs_err(&got1, &exp1) < 1e-4);
        assert!(max_abs_err(&got2, &exp2) < 1e-4);
    }

    // 3) vis_rope2d (16x16 grid, heads=16, hd=72) & vis_merge_2x2_f16 (64 merged patches)
    let n_patches = 256usize;
    let vheads = 16usize;
    let vhd = 72usize;
    let vd = vheads * vhd; // 1152
    let vbase = 10_000.0f32;
    let pos_yx: Vec<i32> = (0..16)
        .flat_map(|y| (0..16).flat_map(move |x| [y, x]))
        .collect();
    let qkv_init: Vec<f32> = (0..n_patches * 3 * vd).map(|_| rng.next_f32()).collect();
    let d_qkv = to_dev_f32(&qkv_init);
    let d_pyx = to_dev_i32(&pos_yx);
    unsafe {
        qwen4::vis_rope2d(
            d_qkv.ptr as *mut f32,
            d_pyx.ptr as *const i32,
            n_patches as i32,
            vheads as i32,
            vhd as i32,
            vbase,
            &stream,
        );
    }
    let d_xf16 = DeviceBuffer::alloc(0, n_patches * vd * 2).unwrap();
    let x_ln: Vec<f32> = (0..n_patches * vd).map(|_| rng.next_f32()).collect();
    let d_xln = to_dev_f32(&x_ln);
    unsafe {
        qwen4::vis_merge_2x2_f16(
            d_xln.ptr as *const f32,
            (n_patches / 4) as i32,
            vd as i32,
            d_xf16.ptr as *mut u16,
            &stream,
        );
    }
    stream.sync().unwrap();

    let got_qkv = read_dev_f32(&d_qkv, n_patches * 3 * vd);
    let mut exp_qkv = qkv_init.clone();
    let half = vhd / 2;
    let quarter = vhd / 4;
    for n in 0..n_patches {
        let py = pos_yx[n * 2 + 0] as f32;
        let px = pos_yx[n * 2 + 1] as f32;
        for h in 0..vheads {
            for qk in 0..2 {
                let off = (n * 3 + qk) * vd + h * vhd;
                for p in 0..half {
                    let (pos, freq_idx) = if p < quarter { (py, p) } else { (px, p - quarter) };
                    let theta = pos * vbase.powf(-2.0 * (freq_idx as f32) / (half as f32));
                    let (sv, cv) = theta.sin_cos();
                    let a = qkv_init[off + p];
                    let b = qkv_init[off + p + half];
                    exp_qkv[off + p] = a * cv - b * sv;
                    exp_qkv[off + p + half] = a * sv + b * cv;
                }
            }
        }
    }
    assert!(max_abs_err(&got_qkv, &exp_qkv) < 1e-5);

    let got_m_raw = read_dev_u8(&d_xf16, n_patches * vd * 2);
    let got_m: Vec<f32> = bytemuck::cast_slice::<u8, u16>(&got_m_raw)
        .iter()
        .map(|&b| f16::from_bits(b).to_f32())
        .collect();
    let exp_m: Vec<f32> = x_ln.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    assert_eq!(got_m, exp_m);
    println!("Router, Shared Gate, and Vision Helpers parity PASSED");
}

#[test]
fn test_qwen4_phase4_fusion_and_doorbell_parity() {
    let _g = GPU_LOCK.lock().unwrap();
    let stream = CudaStream::new(0).expect("GPU 0");
    let mut rng = Rng(0x5048_3446_5553_4501);

    // 1) qwen4_fill_dense_widths
    let pos_h = [0i32, 1, 5, 19, 255, 1024];
    let d_pos = to_dev_i32(&pos_h);
    let d_wid = DeviceBuffer::alloc(0, pos_h.len() * 4).unwrap();
    unsafe {
        qwen4::fill_dense_widths(
            d_pos.ptr as *const i32,
            d_wid.ptr as *mut i32,
            pos_h.len() as i32,
            &stream,
        );
    }
    stream.sync().unwrap();
    let got_wid = read_dev_i32(&d_wid, pos_h.len());
    let exp_wid: Vec<i32> = pos_h.iter().map(|&p| p + 1).collect();
    assert_eq!(got_wid, exp_wid);

    // 2) qwen4_hc_up_mix_q8 vs linear_q8_1(Q8_0) + hc_mix_q8 across T = 1, 4, 33
    let hc = 4usize;
    let e = 2560usize;
    let lr = 320usize;
    let (w_up_bytes, _) = fill_quant_matrix(GgmlType::Q8_0, hc * e, lr, &mut rng);
    let d_wup = to_dev_u8(&w_up_bytes);

    for &t in &[1usize, 4usize, 33usize] {
        let lo_f32: Vec<f32> = (0..t * lr).map(|_| rng.next_f32()).collect();
        let xn_f32: Vec<f32> = (0..t * hc * e).map(|_| rng.next_f32()).collect();
        let d_lo = to_dev_f32(&lo_f32);
        let d_lo_q8 = DeviceBuffer::alloc(0, t * (lr / 32) * 36).unwrap();
        let d_xn = to_dev_f32(&xn_f32);
        let d_up_ref = DeviceBuffer::alloc(0, t * hc * e * 4).unwrap();
        let d_mix_ref = DeviceBuffer::alloc(0, t * e * 4).unwrap();
        let d_mix_q8_ref = DeviceBuffer::alloc(0, t * (e / 32) * 36).unwrap();
        let d_mix_fused = DeviceBuffer::alloc(0, t * e * 4).unwrap();
        let d_mix_q8_fused = DeviceBuffer::alloc(0, t * (e / 32) * 36).unwrap();

        unsafe {
            launch_quantize_q8_1(d_lo.ptr as *const f32, d_lo_q8.ptr, t as i32, lr as i32, stream.raw);
            launch_linear_q8_1(
                GgmlType::Q8_0 as i32,
                d_wup.ptr,
                d_lo_q8.ptr,
                d_lo.ptr as *const f32,
                d_up_ref.ptr as *mut f32,
                (hc * e) as i32,
                lr as i32,
                t as i32,
                stream.raw,
            );
            qwen4::hc_mix_q8(
                d_xn.ptr as *const f32,
                d_up_ref.ptr as *const f32,
                t as i32,
                hc as i32,
                e as i32,
                d_mix_ref.ptr as *mut f32,
                d_mix_q8_ref.ptr,
                &stream,
            );
            qwen4::hc_up_mix_q8(
                d_wup.ptr,
                d_lo_q8.ptr,
                d_xn.ptr as *const f32,
                t as i32,
                hc as i32,
                e as i32,
                lr as i32,
                d_mix_fused.ptr as *mut f32,
                d_mix_q8_fused.ptr,
                &stream,
            );
        }
        stream.sync().unwrap();

        let got_mix = read_dev_f32(&d_mix_fused, t * e);
        let exp_mix = read_dev_f32(&d_mix_ref, t * e);
        let diff_f32 = max_abs_err(&got_mix, &exp_mix);
        assert!(diff_f32 < 1e-4, "hc_up_mix_q8 T={t} f32 max_abs={diff_f32}");

        let got_q8 = dequant_q8_1_blocks(&read_dev_u8(&d_mix_q8_fused, t * (e / 32) * 36), t * e);
        let exp_q8 = dequant_q8_1_blocks(&read_dev_u8(&d_mix_q8_ref, t * (e / 32) * 36), t * e);
        let diff_q8 = max_abs_err(&got_q8, &exp_q8);
        assert!(diff_q8 < 2e-2, "hc_up_mix_q8 T={t} q8 max_abs={diff_q8}");
    }

    // 3) qwen4_router_doorbell_plan + qwen4_doorbell_wait_hc_write
    let t_b = 4usize;
    let n_exp = 512usize;
    let topk = 10usize;
    let logits: Vec<f32> = (0..t_b * n_exp).map(|_| rng.next_f32_range(-5.0, 5.0)).collect();
    let mixed: Vec<f32> = (0..t_b * e).map(|_| rng.next_f32()).collect();
    let mut vram_ptrs = vec![0u64; n_exp];
    let mut ram_ptrs = vec![0u64; n_exp];
    for eid in 0..n_exp {
        if eid % 2 == 0 {
            vram_ptrs[eid] = 0x1000_0000 + (eid as u64) * 4096;
        } else {
            ram_ptrs[eid] = 0x2000_0000 + (eid as u64) * 4096;
        }
    }

    let d_logits = to_dev_f32(&logits);
    let d_mixed = to_dev_f32(&mixed);
    let d_vp = to_dev_u64(&vram_ptrs);
    let d_rp = to_dev_u64(&ram_ptrs);
    let d_ids_ref = DeviceBuffer::alloc(0, t_b * topk * 4).unwrap();
    let d_wts_ref = DeviceBuffer::alloc(0, t_b * topk * 4).unwrap();
    let d_ids_db = DeviceBuffer::alloc(0, t_b * topk * 4).unwrap();
    let d_wts_db = DeviceBuffer::alloc(0, t_b * topk * 4).unwrap();
    let d_plan = DeviceBuffer::alloc(0, t_b * topk * 8).unwrap();
    let d_seq_state = DeviceBuffer::alloc_zeroed(0, 8).unwrap();
    let d_exp_seq = DeviceBuffer::alloc_zeroed(0, 8).unwrap();

    let mut db_pin = PinnedMappedBuffer::<qwen4::Qwen4Doorbell>::alloc(1).unwrap();
    db_pin.as_mut_slice()[0] = qwen4::Qwen4Doorbell::default();
    let db_dev_ptr = db_pin.dev_ptr();

    unsafe {
        qwen4::router_top10(
            d_logits.ptr as *const f32,
            t_b as i32,
            n_exp as i32,
            topk as i32,
            d_ids_ref.ptr as *mut i32,
            d_wts_ref.ptr as *mut f32,
            &stream,
        );
        qwen4::router_doorbell_plan(
            d_logits.ptr as *const f32,
            t_b as i32,
            n_exp as i32,
            topk as i32,
            d_vp.ptr as *const u64,
            d_rp.ptr as *const u64,
            0,
            0,
            d_mixed.ptr as *const f32,
            e as i32,
            640,
            7,
            GgmlType::Q4K as i32,
            GgmlType::Q5_1 as i32,
            d_ids_db.ptr as *mut i32,
            d_wts_db.ptr as *mut f32,
            d_plan.ptr as *mut u64,
            db_dev_ptr,
            d_seq_state.ptr as *mut u64,
            d_exp_seq.ptr as *mut u64,
            &stream,
        );
    }
    stream.sync().unwrap();

    assert_eq!(read_dev_i32(&d_ids_db, t_b * topk), read_dev_i32(&d_ids_ref, t_b * topk));
    assert!(max_abs_err(&read_dev_f32(&d_wts_db, t_b * topk), &read_dev_f32(&d_wts_ref, t_b * topk)) < 1e-6);

    let got_ids = read_dev_i32(&d_ids_db, t_b * topk);
    let got_wts = read_dev_f32(&d_wts_db, t_b * topk);
    let mut got_plan = vec![0u64; t_b * topk];
    d_plan.copy_to_host(bytemuck::cast_slice_mut(&mut got_plan)).unwrap();
    let db = &mut db_pin.as_mut_slice()[0];
    assert_eq!(db.layer_idx, 7);
    assert_eq!(db.m_batch, t_b as i32);
    assert_eq!(db.routed_n_pairs[7], (t_b * topk) as i32);

    let mut exp_mc = 0usize;
    for idx in 0..(t_b * topk) {
        let eid = got_ids[idx] as usize;
        assert_eq!(db.routed_ids[7][idx], eid as i32);
        if vram_ptrs[eid] != 0 {
            assert_eq!(got_plan[idx], vram_ptrs[eid]);
        } else {
            assert_eq!(got_plan[idx], 0);
            assert_eq!(db.miss_tok[exp_mc], (idx / topk) as i32);
            assert_eq!(db.miss_ids[exp_mc], eid as i32);
            assert!((db.miss_weights[exp_mc] - got_wts[idx]).abs() < 1e-6);
            exp_mc += 1;
        }
    }
    assert_eq!(db.miss_count, exp_mc as i32);
    if exp_mc > 0 {
        assert_eq!(db.req_seq, 1);
        assert!(max_abs_err(&db.x_in[..t_b * e], &mixed) == 0.0);
        // Simulate CPU doorbell writing db.out and bumping done_seq
        let cpu_out: Vec<f32> = (0..t_b * e).map(|_| rng.next_f32()).collect();
        db.out[..t_b * e].copy_from_slice(&cpu_out);
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        unsafe { std::ptr::write_volatile(&mut db.done_seq, db.req_seq); }

        let r_init: Vec<f32> = (0..t_b * hc * e).map(|_| rng.next_f32()).collect();
        let bout: Vec<f32> = (0..t_b * e).map(|_| rng.next_f32()).collect();
        let shout: Vec<f32> = (0..t_b * e).map(|_| rng.next_f32()).collect();
        let shgate: Vec<f32> = (0..t_b).map(|_| rng.next_f32()).collect();
        let inject: Vec<f32> = (0..t_b * hc).map(|_| rng.next_f32()).collect();
        let sum_b_c: Vec<f32> = bout.iter().zip(cpu_out.iter()).map(|(a, b)| a + b).collect();

        let d_r_ref = to_dev_f32(&r_init);
        let d_r_db = to_dev_f32(&r_init);
        let d_bout = to_dev_f32(&bout);
        let d_sum_bc = to_dev_f32(&sum_b_c);
        let d_shout = to_dev_f32(&shout);
        let d_shgate = to_dev_f32(&shgate);
        let d_inj = to_dev_f32(&inject);

        unsafe {
            qwen4::hc_write(
                d_r_ref.ptr as *mut f32,
                d_sum_bc.ptr as *const f32,
                d_shout.ptr as *const f32,
                d_shgate.ptr as *const f32,
                d_inj.ptr as *const f32,
                t_b as i32,
                hc as i32,
                e as i32,
                &stream,
            );
            qwen4::doorbell_wait_hc_write(
                db_dev_ptr,
                d_exp_seq.ptr as *const u64,
                d_r_db.ptr as *mut f32,
                d_bout.ptr as *const f32,
                d_shout.ptr as *const f32,
                d_shgate.ptr as *const f32,
                d_inj.ptr as *const f32,
                t_b as i32,
                hc as i32,
                e as i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let d = max_abs_err(&read_dev_f32(&d_r_db, t_b * hc * e), &read_dev_f32(&d_r_ref, t_b * hc * e));
        assert!(d < 1e-5, "doorbell_wait_hc_write max_abs={d}");
    }
    println!("Phase-4 Qwen4 fusion + doorbell kernels parity PASSED");

    // 4) Phase-5E utility kernels: reduce_pairs_f32, patch_u64, copy_u32_sm
    {
        let rt = 4usize;
        let rtopk = 10usize;
        let rdim = 2560usize;
        let pdata: Vec<f32> = (0..rt * rtopk * rdim).map(|_| rng.next_f32()).collect();
        let d_pairs = to_dev_f32(&pdata);
        let d_bout = DeviceBuffer::alloc(0, rt * rdim * 4).unwrap();
        unsafe {
            qwen4::reduce_pairs_f32(
                d_pairs.ptr as *const f32,
                d_bout.ptr as *mut f32,
                rt as i32,
                rtopk as i32,
                rdim as i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let got_r = read_dev_f32(&d_bout, rt * rdim);
        let mut exp_r = vec![0.0f32; rt * rdim];
        for ti in 0..rt {
            for d in 0..rdim {
                let mut s = 0.0f32;
                for k in 0..rtopk {
                    s += pdata[(ti * rtopk + k) * rdim + d];
                }
                exp_r[ti * rdim + d] = s;
            }
        }
        assert_eq!(got_r, exp_r, "reduce_pairs_f32 bit-exact match");

        let init_u64 = vec![0x1111_2222_3333_4444u64; 64];
        let d_tab = to_dev_u64(&init_u64);
        let mut patch_buf = PinnedMappedBuffer::<u64>::alloc(8).unwrap();
        let patch_dev = patch_buf.dev_ptr_for_device(0).unwrap() as u64;
        patch_buf.as_mut_slice().copy_from_slice(&[
            d_tab.as_u64() + 3 * 8,
            0xDEAD_BEEF_CAFE_0003u64,
            d_tab.as_u64() + 17 * 8,
            0xABCD_EF01_2345_0017u64,
            d_tab.as_u64() + 63 * 8,
            0u64,
            0,
            0,
        ]);
        unsafe {
            qwen4::patch_u64(patch_dev as *const u64, 3, &stream);
        }
        let mut src_pin = PinnedMappedBuffer::<u32>::alloc(2565).unwrap();
        for i in 0..2565 {
            src_pin.as_mut_slice()[i] = (i as u32).wrapping_mul(2654435761);
        }
        let src_dev = src_pin.dev_ptr_for_device(0).unwrap();
        let d_dst = DeviceBuffer::alloc(0, 2565 * 4).unwrap();
        let back_pin = PinnedMappedBuffer::<u32>::alloc(2565).unwrap();
        let back_dev = back_pin.dev_ptr_for_device(0).unwrap();
        unsafe {
            qwen4::copy_u32_sm(d_dst.ptr, src_dev, 2565, false, &stream);
            qwen4::copy_u32_sm(back_dev, d_dst.ptr, 2565, true, &stream);
        }
        stream.sync().unwrap();
        let mut got_tab = vec![0u64; 64];
        d_tab.copy_to_host(bytemuck::cast_slice_mut(&mut got_tab)).unwrap();
        assert_eq!(got_tab[3], 0xDEAD_BEEF_CAFE_0003u64);
        assert_eq!(got_tab[17], 0xABCD_EF01_2345_0017u64);
        assert_eq!(got_tab[63], 0u64);
        assert_eq!(got_tab[0], 0x1111_2222_3333_4444u64);
        assert_eq!(back_pin.as_slice(), src_pin.as_slice());
        println!("Phase-5E utility kernels (reduce_pairs_f32, patch_u64, copy_u32_sm) PASSED");
    }
}
