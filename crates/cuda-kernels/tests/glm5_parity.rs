use ampllm_cuda_kernels::ffi::*;
use ampllm_cuda_kernels::{glm5, CudaStream, DeviceBuffer, PinnedMappedBuffer};
use ampllm_gguf::dequantize_row_to_f32;
use ampllm_gguf::GgmlType;
use half::{bf16, f16};

static GPU_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u32() % 2000) as f32 - 1000.0) / 500.0
    }
    fn next_u8(&mut self) -> u8 {
        (self.next_u32() & 0xFF) as u8
    }
}

fn fill_glm5_quant_row(t: GgmlType, k: usize, rng: &mut Rng) -> Vec<u8> {
    let nb = k / t.block_size();
    let bsz = t.type_size();
    let mut bytes = vec![0u8; nb * bsz];
    for b in 0..nb {
        let blk = &mut bytes[b * bsz..(b + 1) * bsz];
        match t {
            GgmlType::Q8_0 => {
                let d = f16::from_f32(0.015 + (b as f32) * 0.0002).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for x in &mut blk[2..34] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Q3K => {
                let d = f16::from_f32(0.012 + (b as f32) * 0.0001).to_le_bytes();
                for x in &mut blk[0..108] {
                    *x = rng.next_u8();
                }
                blk[108..110].copy_from_slice(&d);
            }
            GgmlType::Q4K => {
                let d = f16::from_f32(0.011).to_le_bytes();
                let dm = f16::from_f32(0.007).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                blk[2..4].copy_from_slice(&dm);
                for x in &mut blk[4..144] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Q6K => {
                let d = f16::from_f32(0.009).to_le_bytes();
                for x in &mut blk[0..208] {
                    *x = rng.next_u8();
                }
                blk[208..210].copy_from_slice(&d);
            }
            GgmlType::Iq3Xxs => {
                let d = f16::from_f32(0.008).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for x in &mut blk[2..98] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Iq3S => {
                let d = f16::from_f32(0.006).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for x in &mut blk[2..110] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Iq2S => {
                let d = f16::from_f32(0.007).to_le_bytes();
                blk[0..2].copy_from_slice(&d);
                for x in &mut blk[2..82] {
                    *x = rng.next_u8();
                }
            }
            _ => unreachable!(),
        }
    }
    bytes
}

fn rel_err(got: &[f32], exp: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &e) in got.iter().zip(exp.iter()) {
        num += ((g - e) as f64).abs();
        den += (e as f64).abs();
    }
    (num / den.max(1e-12)) as f32
}

fn max_abs_err(got: &[f32], exp: &[f32]) -> f32 {
    got.iter()
        .zip(exp.iter())
        .map(|(&g, &e)| (g - e).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn test_glm5_quant_gemv_and_resident_moe_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).expect("CUDA GPU 0 available");
    let mut rng = Rng(0x55aa1234deadbeef);

    let k = 2048usize;
    let n = 64usize;
    let x_m1: Vec<f32> = (0..k).map(|_| rng.next_f32() * 0.5).collect();
    let x_m4: Vec<f32> = (0..(4 * k)).map(|_| rng.next_f32() * 0.5).collect();

    let d_x1 = DeviceBuffer::alloc(0, k * 4).unwrap();
    let d_x4 = DeviceBuffer::alloc(0, 4 * k * 4).unwrap();
    let d_q8_1 = DeviceBuffer::alloc(0, (k / 32) * 36).unwrap();
    let d_q8_4 = DeviceBuffer::alloc(0, 4 * (k / 32) * 36).unwrap();
    let d_y = DeviceBuffer::alloc(0, 4 * n * 4).unwrap();
    d_x1.copy_from_host(bytemuck::cast_slice(&x_m1)).unwrap();
    d_x4.copy_from_host(bytemuck::cast_slice(&x_m4)).unwrap();

    unsafe {
        launch_quantize_q8_1(d_x1.ptr.cast(), d_q8_1.ptr, 1, k as i32, stream.raw);
        launch_quantize_q8_1(d_x4.ptr.cast(), d_q8_4.ptr, 4, k as i32, stream.raw);
    }

    for &wt in &[
        GgmlType::Q3K,
        GgmlType::Q4K,
        GgmlType::Q6K,
        GgmlType::Iq2S,
        GgmlType::Iq3Xxs,
        GgmlType::Iq3S,
    ] {
        let bpr = (k / wt.block_size()) * wt.type_size();
        let mut w_bytes = Vec::with_capacity(n * bpr);
        let mut w_f32 = vec![0.0f32; n * k];
        for r in 0..n {
            let row = fill_glm5_quant_row(wt, k, &mut rng);
            dequantize_row_to_f32(wt, &row, &mut w_f32[r * k..(r + 1) * k]);
            w_bytes.extend_from_slice(&row);
        }
        let d_w = DeviceBuffer::alloc(0, w_bytes.len()).unwrap();
        d_w.copy_from_host(&w_bytes).unwrap();

        // M=1 sub256 GEMV
        let mut ref1 = vec![0.0f32; n];
        for r in 0..n {
            let mut s = 0.0f64;
            for c in 0..k {
                s += w_f32[r * k + c] as f64 * x_m1[c] as f64;
            }
            ref1[r] = s as f32;
        }
        unsafe {
            launch_linear_q8_1(
                wt as i32,
                d_w.ptr,
                d_q8_1.ptr,
                d_x1.ptr.cast(),
                d_y.ptr.cast(),
                n as i32,
                k as i32,
                1,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let mut got1 = vec![0.0f32; n];
        d_y.copy_to_host(bytemuck::cast_slice_mut(&mut got1)).unwrap();
        let re1 = rel_err(&got1, &ref1);
        assert!(
            re1 < 2e-2,
            "M=1 GEMV {:?} rel_err={re1} got0={:?} ref0={:?}",
            wt,
            &got1[..4],
            &ref1[..4]
        );

        // M=4 GEMV
        let mut ref4 = vec![0.0f32; 4 * n];
        for m in 0..4 {
            for r in 0..n {
                let mut s = 0.0f64;
                for c in 0..k {
                    s += w_f32[r * k + c] as f64 * x_m4[m * k + c] as f64;
                }
                ref4[m * n + r] = s as f32;
            }
        }
        unsafe {
            launch_linear_q8_1(
                wt as i32,
                d_w.ptr,
                d_q8_4.ptr,
                d_x4.ptr.cast(),
                d_y.ptr.cast(),
                n as i32,
                k as i32,
                4,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let mut got4 = vec![0.0f32; 4 * n];
        d_y.copy_to_host(bytemuck::cast_slice_mut(&mut got4)).unwrap();
        let re4 = rel_err(&got4, &ref4);
        assert!(re4 < 2e-2, "M=4 GEMV {:?} rel_err={re4}", wt);
    }

    // Split gate/up/down resident MoE with clamped SwiGLU + null CPU slots
    let n_embd = 512usize;
    let n_ff = 256usize;
    let n_exp = 6usize;
    let topk = 4usize;
    let clamp_lim = 10.0f32;
    let x_moe: Vec<f32> = (0..n_embd).map(|_| rng.next_f32() * 1.5).collect();
    let d_xmoe = DeviceBuffer::alloc(0, n_embd * 4).unwrap();
    let d_xmoe_q8 = DeviceBuffer::alloc(0, (n_embd / 32) * 36).unwrap();
    d_xmoe.copy_from_host(bytemuck::cast_slice(&x_moe)).unwrap();
    unsafe {
        launch_quantize_q8_1(d_xmoe.ptr.cast(), d_xmoe_q8.ptr, 1, n_embd as i32, stream.raw);
    }

    for &(gu_t, d_t) in &[
        (GgmlType::Iq2S, GgmlType::Iq3Xxs),
        (GgmlType::Iq2S, GgmlType::Iq3S),
        (GgmlType::Q3K, GgmlType::Q4K),
    ] {
        let gu_bpr = (n_embd / gu_t.block_size()) * gu_t.type_size();
        let d_bpr = (n_ff / d_t.block_size()) * d_t.type_size();
        let mut g_bufs = Vec::new();
        let mut u_bufs = Vec::new();
        let mut d_bufs = Vec::new();
        let mut g_ptrs = vec![0u64; n_exp];
        let mut u_ptrs = vec![0u64; n_exp];
        let mut d_ptrs = vec![0u64; n_exp];
        let mut g_f32 = vec![vec![0.0f32; n_ff * n_embd]; n_exp];
        let mut u_f32 = vec![vec![0.0f32; n_ff * n_embd]; n_exp];
        let mut d_f32 = vec![vec![0.0f32; n_embd * n_ff]; n_exp];

        for e in 0..n_exp {
            let mut gb = Vec::with_capacity(n_ff * gu_bpr);
            let mut ub = Vec::with_capacity(n_ff * gu_bpr);
            for r in 0..n_ff {
                let gr = fill_glm5_quant_row(gu_t, n_embd, &mut rng);
                let ur = fill_glm5_quant_row(gu_t, n_embd, &mut rng);
                dequantize_row_to_f32(gu_t, &gr, &mut g_f32[e][r * n_embd..(r + 1) * n_embd]);
                dequantize_row_to_f32(gu_t, &ur, &mut u_f32[e][r * n_embd..(r + 1) * n_embd]);
                gb.extend_from_slice(&gr);
                ub.extend_from_slice(&ur);
            }
            let mut db = Vec::with_capacity(n_embd * d_bpr);
            for r in 0..n_embd {
                let dr = fill_glm5_quant_row(d_t, n_ff, &mut rng);
                dequantize_row_to_f32(d_t, &dr, &mut d_f32[e][r * n_ff..(r + 1) * n_ff]);
                db.extend_from_slice(&dr);
            }
            if e != 2 {
                let dg = DeviceBuffer::alloc(0, gb.len()).unwrap();
                let du = DeviceBuffer::alloc(0, ub.len()).unwrap();
                let dd = DeviceBuffer::alloc(0, db.len()).unwrap();
                dg.copy_from_host(&gb).unwrap();
                du.copy_from_host(&ub).unwrap();
                dd.copy_from_host(&db).unwrap();
                g_ptrs[e] = dg.as_u64();
                u_ptrs[e] = du.as_u64();
                d_ptrs[e] = dd.as_u64();
                g_bufs.push(dg);
                u_bufs.push(du);
                d_bufs.push(dd);
            }
        }
        let sel_ids = vec![1i32, 2, 4, 0];
        let sel_rw = vec![0.4f32, 0.3, 0.2, 0.1];
        let mut ref_moe = vec![0.0f32; n_embd];
        for (slot, &ei) in sel_ids.iter().enumerate() {
            if ei == 2 {
                continue; // CPU miss skipped on GPU
            }
            let e = ei as usize;
            let mut inter = vec![0.0f32; n_ff];
            for r in 0..n_ff {
                let mut sg = 0.0f64;
                let mut su = 0.0f64;
                for c in 0..n_embd {
                    sg += g_f32[e][r * n_embd + c] as f64 * x_moe[c] as f64;
                    su += u_f32[e][r * n_embd + c] as f64 * x_moe[c] as f64;
                }
                let gc = (sg as f32).min(clamp_lim);
                let uc = (su as f32).clamp(-clamp_lim, clamp_lim);
                inter[r] = (gc / (1.0 + (-gc).exp())) * uc;
            }
            for r in 0..n_embd {
                let mut sd = 0.0f64;
                for c in 0..n_ff {
                    sd += d_f32[e][r * n_ff + c] as f64 * inter[c] as f64;
                }
                ref_moe[r] += sel_rw[slot] * sd as f32;
            }
        }

        let dev_gp = DeviceBuffer::alloc(0, g_ptrs.len() * 8).unwrap();
        let dev_up = DeviceBuffer::alloc(0, u_ptrs.len() * 8).unwrap();
        let dev_dp = DeviceBuffer::alloc(0, d_ptrs.len() * 8).unwrap();
        let dev_ids = DeviceBuffer::alloc(0, topk * 4).unwrap();
        let dev_rw = DeviceBuffer::alloc(0, topk * 4).unwrap();
        let dev_iq8 = DeviceBuffer::alloc(0, topk * (n_ff / 32) * 36).unwrap();
        let dev_out = DeviceBuffer::alloc(0, n_embd * 4).unwrap();
        dev_gp.copy_from_host(bytemuck::cast_slice(&g_ptrs)).unwrap();
        dev_up.copy_from_host(bytemuck::cast_slice(&u_ptrs)).unwrap();
        dev_dp.copy_from_host(bytemuck::cast_slice(&d_ptrs)).unwrap();
        dev_ids.copy_from_host(bytemuck::cast_slice(&sel_ids)).unwrap();
        dev_rw.copy_from_host(bytemuck::cast_slice(&sel_rw)).unwrap();

        unsafe {
            glm5::moe_resident(
                gu_t as i32,
                d_t as i32,
                dev_gp.ptr.cast(),
                dev_up.ptr.cast(),
                dev_dp.ptr.cast(),
                gu_bpr,
                d_bpr,
                d_xmoe_q8.ptr,
                dev_ids.ptr.cast(),
                dev_rw.ptr.cast(),
                dev_iq8.ptr,
                std::ptr::null_mut(),
                dev_out.ptr.cast(),
                n_embd as i32,
                n_ff as i32,
                topk as i32,
                1,
                clamp_lim,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got_moe = vec![0.0f32; n_embd];
        dev_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_moe)).unwrap();
        let re = rel_err(&got_moe, &ref_moe);
        assert!(re < 2.5e-2, "MoE ({:?}/{:?}) rel_err={re}", gu_t, d_t);
    }
}

#[test]
fn test_glm5_mma_gemm_and_dequant_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).expect("CUDA GPU 0 available");
    let mut rng = Rng(0xabcdef0123456789);
    let m = 77usize;
    let n = 192usize;
    let k = 512usize;

    let a_f32: Vec<f32> = (0..(m * k)).map(|_| rng.next_f32() * 0.5).collect();
    let d_af32 = DeviceBuffer::alloc(0, a_f32.len() * 4).unwrap();
    let d_af16 = DeviceBuffer::alloc(0, a_f32.len() * 2).unwrap();
    let d_bf16 = DeviceBuffer::alloc(0, n * k * 2).unwrap();
    let d_c = DeviceBuffer::alloc(0, m * n * 4).unwrap();
    d_af32.copy_from_host(bytemuck::cast_slice(&a_f32)).unwrap();

    unsafe {
        glm5::f32_to_f16(d_af32.ptr.cast(), d_af16.ptr, (m * k) as i32, &stream);
    }

    for &wt in &[GgmlType::Q6K, GgmlType::Q3K, GgmlType::Iq2S, GgmlType::Iq3Xxs] {
        let bpr = (k / wt.block_size()) * wt.type_size();
        let mut w_raw = Vec::with_capacity(n * bpr);
        let mut b_f32 = vec![0.0f32; n * k];
        for r in 0..n {
            let row = fill_glm5_quant_row(wt, k, &mut rng);
            dequantize_row_to_f32(wt, &row, &mut b_f32[r * k..(r + 1) * k]);
            w_raw.extend_from_slice(&row);
        }
        let d_wraw = DeviceBuffer::alloc(0, w_raw.len()).unwrap();
        d_wraw.copy_from_host(&w_raw).unwrap();

        let mut c_ref = vec![0.0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut s = 0.0f64;
                for i in 0..k {
                    s += a_f32[r * k + i] as f64 * b_f32[col * k + i] as f64;
                }
                c_ref[r * n + col] = s as f32;
            }
        }

        unsafe {
            glm5::dequant_to_f16(wt as i32, d_wraw.ptr, d_bf16.ptr, (n * k) as i64, &stream);
            glm5::gemm_f16_mma_nt(
                d_af16.ptr,
                d_bf16.ptr,
                d_c.ptr.cast(),
                m as i32,
                n as i32,
                k as i32,
                1.0,
                0.0,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut c_got = vec![0.0f32; m * n];
        d_c.copy_to_host(bytemuck::cast_slice_mut(&mut c_got)).unwrap();
        let re = rel_err(&c_got, &c_ref);
        assert!(re < 5e-3, "MMA GEMM {:?} rel_err={re}", wt);
    }
}

#[test]
fn test_glm5_hc_kda_dsa_router_doorbell_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).expect("CUDA GPU 0 available");
    let mut rng = Rng(0x9988776655443322);

    // 1. HC Pre/Post F64 Sinkhorn parity
    let n_embd = 256usize;
    let hc = 4usize;
    let m = 2usize;
    let hc_dim = hc * n_embd;
    let hc_mix = (2 + hc) * hc;
    let iters = 20usize;
    let norm_eps = 1e-5f32;
    let hc_eps = 1e-6f32;

    let r_in: Vec<f32> = (0..(m * hc_dim)).map(|_| rng.next_f32() * 0.3).collect();
    let w_fn: Vec<f32> = (0..(hc_mix * hc_dim)).map(|_| rng.next_f32() * 0.05).collect();
    let w_scale = vec![0.5f32, 0.7f32, 0.4f32];
    let w_base: Vec<f32> = (0..hc_mix).map(|_| rng.next_f32() * 0.1).collect();
    let norm_w: Vec<f32> = (0..n_embd).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();

    let mut ref_post = vec![0.0f32; m * hc];
    let mut ref_comb = vec![0.0f32; m * hc * hc];
    let mut ref_x = vec![0.0f32; m * n_embd];
    for t in 0..m {
        let rt = &r_in[t * hc_dim..(t + 1) * hc_dim];
        let ss: f64 = rt.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = 1.0 / (ss / hc_dim as f64 + norm_eps as f64).sqrt();
        let mut mix = vec![0.0f64; hc_mix];
        for mm in 0..hc_mix {
            let mut acc = 0.0f64;
            for k in 0..hc_dim {
                acc += (rt[k] as f64 * inv) * w_fn[mm * hc_dim + k] as f64;
            }
            mix[mm] = acc;
        }
        let mut pre = vec![0.0f64; hc];
        for s in 0..hc {
            let v = mix[s] * w_scale[0] as f64 + w_base[s] as f64;
            pre[s] = 1.0 / (1.0 + (-v).exp()) + hc_eps as f64;
            let pv = mix[hc + s] * w_scale[1] as f64 + w_base[hc + s] as f64;
            ref_post[t * hc + s] = (2.0 / (1.0 + (-pv).exp())) as f32;
        }
        let mut cb = vec![0.0f64; hc * hc];
        for d in 0..hc {
            for s in 0..hc {
                cb[d * hc + s] = mix[2 * hc + d + hc * s] * w_scale[2] as f64 + w_base[2 * hc + d + hc * s] as f64;
            }
        }
        for s in 0..hc {
            let mut mx = f64::NEG_INFINITY;
            for d in 0..hc { mx = mx.max(cb[d * hc + s]); }
            let mut den = 0.0;
            for d in 0..hc { den += (cb[d * hc + s] - mx).exp(); }
            for d in 0..hc { cb[d * hc + s] = (cb[d * hc + s] - mx).exp() / den + hc_eps as f64; }
        }
        let norm_dst = |cb: &mut [f64]| {
            for d in 0..hc {
                let s: f64 = hc_eps as f64 + (0..hc).map(|c| cb[d * hc + c]).sum::<f64>();
                for c in 0..hc { cb[d * hc + c] /= s; }
            }
        };
        let norm_src = |cb: &mut [f64]| {
            for s in 0..hc {
                let c: f64 = hc_eps as f64 + (0..hc).map(|d| cb[d * hc + s]).sum::<f64>();
                for d in 0..hc { cb[d * hc + s] /= c; }
            }
        };
        norm_dst(&mut cb);
        for _ in 1..iters { norm_src(&mut cb); norm_dst(&mut cb); }
        for i in 0..(hc * hc) { ref_comb[t * hc * hc + i] = cb[i] as f32; }

        let mut mixed = vec![0.0f64; n_embd];
        let mut mss = 0.0f64;
        for e in 0..n_embd {
            let val: f64 = (0..hc).map(|s| pre[s] * rt[s * n_embd + e] as f64).sum();
            mixed[e] = val;
            mss += val * val;
        }
        let inv_n = 1.0 / (mss / n_embd as f64 + norm_eps as f64).sqrt();
        for e in 0..n_embd {
            ref_x[t * n_embd + e] = (mixed[e] * inv_n * norm_w[e] as f64) as f32;
        }
    }

    let d_r = DeviceBuffer::alloc(0, r_in.len() * 4).unwrap();
    let d_wfn = DeviceBuffer::alloc(0, w_fn.len() * 4).unwrap();
    let d_ws = DeviceBuffer::alloc(0, 12).unwrap();
    let d_wb = DeviceBuffer::alloc(0, w_base.len() * 4).unwrap();
    let d_nw = DeviceBuffer::alloc(0, norm_w.len() * 4).unwrap();
    let d_po = DeviceBuffer::alloc(0, m * hc * 4).unwrap();
    let d_co = DeviceBuffer::alloc(0, m * hc * hc * 4).unwrap();
    let d_xo = DeviceBuffer::alloc(0, m * n_embd * 4).unwrap();
    let d_xq = DeviceBuffer::alloc(0, m * (n_embd / 32) * 36).unwrap();
    d_r.copy_from_host(bytemuck::cast_slice(&r_in)).unwrap();
    d_wfn.copy_from_host(bytemuck::cast_slice(&w_fn)).unwrap();
    d_ws.copy_from_host(bytemuck::cast_slice(&w_scale)).unwrap();
    d_wb.copy_from_host(bytemuck::cast_slice(&w_base)).unwrap();
    d_nw.copy_from_host(bytemuck::cast_slice(&norm_w)).unwrap();

    unsafe {
        glm5::hc_pre_norm_q8(
            d_r.ptr.cast(), d_wfn.ptr, GgmlType::F32 as i32, d_ws.ptr.cast(), d_wb.ptr.cast(),
            d_nw.ptr.cast(), n_embd as i32, hc as i32, iters as i32, norm_eps, hc_eps,
            d_po.ptr.cast(), d_co.ptr.cast(), d_xo.ptr.cast(), d_xq.ptr, m as i32, &stream,
        );
    }
    stream.sync().unwrap();
    let mut got_post = vec![0.0f32; m * hc];
    let mut got_comb = vec![0.0f32; m * hc * hc];
    let mut got_x = vec![0.0f32; m * n_embd];
    d_po.copy_to_host(bytemuck::cast_slice_mut(&mut got_post)).unwrap();
    d_co.copy_to_host(bytemuck::cast_slice_mut(&mut got_comb)).unwrap();
    d_xo.copy_to_host(bytemuck::cast_slice_mut(&mut got_x)).unwrap();
    assert!(max_abs_err(&got_post, &ref_post) < 1e-4);
    assert!(max_abs_err(&got_comb, &ref_comb) < 1e-4);
    assert!(max_abs_err(&got_x, &ref_x) < 1e-4);

    // BF16 w_fn check
    let w_fn_bf16: Vec<u16> = w_fn.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
    let d_wfn_b16 = DeviceBuffer::alloc(0, w_fn_bf16.len() * 2).unwrap();
    d_wfn_b16.copy_from_host(bytemuck::cast_slice(&w_fn_bf16)).unwrap();
    unsafe {
        glm5::hc_pre_norm_q8(
            d_r.ptr.cast(), d_wfn_b16.ptr, GgmlType::Bf16 as i32, d_ws.ptr.cast(), d_wb.ptr.cast(),
            d_nw.ptr.cast(), n_embd as i32, hc as i32, iters as i32, norm_eps, hc_eps,
            d_po.ptr.cast(), d_co.ptr.cast(), d_xo.ptr.cast(), d_xq.ptr, m as i32, &stream,
        );
    }
    stream.sync().unwrap();
    d_xo.copy_to_host(bytemuck::cast_slice_mut(&mut got_x)).unwrap();
    assert!(max_abs_err(&got_x, &ref_x) < 1e-3);

    // 2. KDA step-by-step T=1 vs batched T=8 parity
    let dk = 128usize;
    let nh = 4usize;
    let di = nh * dk;
    let t_steps = 8usize;
    let q_in: Vec<f32> = (0..(t_steps * di)).map(|_| rng.next_f32() * 0.25).collect();
    let k_in: Vec<f32> = (0..(t_steps * di)).map(|_| rng.next_f32() * 0.25).collect();
    let v_in: Vec<f32> = (0..(t_steps * di)).map(|_| rng.next_f32() * 0.25).collect();
    let g_in: Vec<f32> = (0..(t_steps * di)).map(|_| rng.next_f32()).collect();
    let beta_raw: Vec<f32> = (0..(t_steps * nh)).map(|_| rng.next_f32()).collect();
    let dt_bias: Vec<f32> = (0..di).map(|_| rng.next_f32() * 0.1).collect();
    let ssm_a: Vec<f32> = (0..nh).map(|_| 0.8 + 0.1 * rng.next_f32()).collect();
    let cw: Vec<f32> = (0..(di * 4)).map(|_| rng.next_f32() * 0.2).collect();

    let d_cs1 = DeviceBuffer::alloc_zeroed(0, di * 3 * 4).unwrap();
    let d_cs2 = DeviceBuffer::alloc_zeroed(0, di * 3 * 4).unwrap();
    let d_st1 = DeviceBuffer::alloc_zeroed(0, nh * dk * dk * 4).unwrap();
    let d_st2 = DeviceBuffer::alloc_zeroed(0, nh * dk * dk * 4).unwrap();
    let d_q = DeviceBuffer::alloc(0, q_in.len() * 4).unwrap();
    let d_k = DeviceBuffer::alloc(0, k_in.len() * 4).unwrap();
    let d_v = DeviceBuffer::alloc(0, v_in.len() * 4).unwrap();
    let d_gin = DeviceBuffer::alloc(0, g_in.len() * 4).unwrap();
    let d_b1 = DeviceBuffer::alloc(0, beta_raw.len() * 4).unwrap();
    let d_b2 = DeviceBuffer::alloc(0, beta_raw.len() * 4).unwrap();
    let d_cw = DeviceBuffer::alloc(0, cw.len() * 4).unwrap();
    let d_dt = DeviceBuffer::alloc(0, dt_bias.len() * 4).unwrap();
    let d_sa = DeviceBuffer::alloc(0, ssm_a.len() * 4).unwrap();
    let d_qc = DeviceBuffer::alloc(0, q_in.len() * 4).unwrap();
    let d_kc = DeviceBuffer::alloc(0, k_in.len() * 4).unwrap();
    let d_vc = DeviceBuffer::alloc(0, v_in.len() * 4).unwrap();
    let d_g1 = DeviceBuffer::alloc(0, g_in.len() * 4).unwrap();
    let d_o1 = DeviceBuffer::alloc(0, q_in.len() * 4).unwrap();
    let d_o2 = DeviceBuffer::alloc(0, q_in.len() * 4).unwrap();

    d_q.copy_from_host(bytemuck::cast_slice(&q_in)).unwrap();
    d_k.copy_from_host(bytemuck::cast_slice(&k_in)).unwrap();
    d_v.copy_from_host(bytemuck::cast_slice(&v_in)).unwrap();
    d_gin.copy_from_host(bytemuck::cast_slice(&g_in)).unwrap();
    d_b1.copy_from_host(bytemuck::cast_slice(&beta_raw)).unwrap();
    d_b2.copy_from_host(bytemuck::cast_slice(&beta_raw)).unwrap();
    d_cw.copy_from_host(bytemuck::cast_slice(&cw)).unwrap();
    d_dt.copy_from_host(bytemuck::cast_slice(&dt_bias)).unwrap();
    d_sa.copy_from_host(bytemuck::cast_slice(&ssm_a)).unwrap();

    unsafe {
        // Batched T=8
        glm5::kda_conv_silu_l2(d_q.ptr.cast(), d_cw.ptr.cast(), d_cs2.ptr.cast(), di as i32, 4, dk as i32, nh as i32, true, t_steps as i32, d_qc.ptr.cast(), &stream);
        glm5::kda_conv_silu_l2(d_k.ptr.cast(), d_cw.ptr.cast(), d_cs1.ptr.cast(), di as i32, 4, dk as i32, nh as i32, true, t_steps as i32, d_kc.ptr.cast(), &stream);
        glm5::kda_conv_silu_l2(d_v.ptr.cast(), d_cw.ptr.cast(), d_cs2.ptr.cast(), di as i32, 4, dk as i32, nh as i32, false, t_steps as i32, d_vc.ptr.cast(), &stream);
        glm5::kda_gate_beta(d_gin.ptr.cast(), d_dt.ptr.cast(), d_sa.ptr.cast(), -5.0, dk as i32, nh as i32, t_steps as i32, d_g1.ptr.cast(), d_b2.ptr.cast(), &stream);
        glm5::kda_recurrence(d_qc.ptr.cast(), d_kc.ptr.cast(), d_vc.ptr.cast(), d_g1.ptr.cast(), d_b2.ptr.cast(), d_st2.ptr.cast(), nh as i32, dk as i32, t_steps as i32, d_o2.ptr.cast(), &stream);

        // Step-by-step T=1
        for t in 0..t_steps {
            let off_f = (t * di * 4) as usize;
            let off_b = (t * nh * 4) as usize;
            glm5::kda_conv_silu_l2(d_q.ptr.cast::<u8>().add(off_f).cast(), d_cw.ptr.cast(), d_cs1.ptr.cast(), di as i32, 4, dk as i32, nh as i32, true, 1, d_qc.ptr.cast(), &stream);
            glm5::kda_conv_silu_l2(d_k.ptr.cast::<u8>().add(off_f).cast(), d_cw.ptr.cast(), d_cs2.ptr.cast(), di as i32, 4, dk as i32, nh as i32, true, 1, d_kc.ptr.cast(), &stream);
            // Reset cs states properly? Only testing recurrence parity here using exact same conv inputs
            glm5::kda_gate_beta(d_gin.ptr.cast::<u8>().add(off_f).cast(), d_dt.ptr.cast(), d_sa.ptr.cast(), -5.0, dk as i32, nh as i32, 1, d_g1.ptr.cast(), d_b1.ptr.cast::<u8>().add(off_b).cast(), &stream);
            glm5::kda_recurrence(d_qc.ptr.cast(), d_kc.ptr.cast(), d_vc.ptr.cast::<u8>().add(off_f).cast(), d_g1.ptr.cast(), d_b1.ptr.cast::<u8>().add(off_b).cast(), d_st1.ptr.cast(), nh as i32, dk as i32, 1, d_o1.ptr.cast::<u8>().add(off_f).cast(), &stream);
        }
    }
    stream.sync().unwrap();
    // Run exact clean conv parity for Q: T=8 vs T=1 loop with dedicated states
    let d_ca = DeviceBuffer::alloc_zeroed(0, di * 3 * 4).unwrap();
    let d_cb = DeviceBuffer::alloc_zeroed(0, di * 3 * 4).unwrap();
    let d_oa = DeviceBuffer::alloc(0, t_steps * di * 4).unwrap();
    let d_ob = DeviceBuffer::alloc(0, t_steps * di * 4).unwrap();
    unsafe {
        glm5::kda_conv_silu_l2(d_q.ptr.cast(), d_cw.ptr.cast(), d_ca.ptr.cast(), di as i32, 4, dk as i32, nh as i32, true, t_steps as i32, d_oa.ptr.cast(), &stream);
        for t in 0..t_steps {
            let off_f = t * di * 4;
            glm5::kda_conv_silu_l2(d_q.ptr.cast::<u8>().add(off_f).cast(), d_cw.ptr.cast(), d_cb.ptr.cast(), di as i32, 4, dk as i32, nh as i32, true, 1, d_ob.ptr.cast::<u8>().add(off_f).cast(), &stream);
        }
    }
    stream.sync().unwrap();
    let mut oa = vec![0.0f32; t_steps * di];
    let mut ob = vec![0.0f32; t_steps * di];
    d_oa.copy_to_host(bytemuck::cast_slice_mut(&mut oa)).unwrap();
    d_ob.copy_to_host(bytemuck::cast_slice_mut(&mut ob)).unwrap();
    assert!(max_abs_err(&oa, &ob) < 1e-5);

    // Re-test recurrence cleanly T=8 vs T=1 with identical normalized q, k, v, g1, beta
    d_st1.zero().unwrap();
    d_st2.zero().unwrap();
    unsafe {
        glm5::kda_conv_silu_l2(d_q.ptr.cast(), d_cw.ptr.cast(), d_ca.ptr.cast(), di as i32, 4, dk as i32, nh as i32, true, t_steps as i32, d_qc.ptr.cast(), &stream);
        glm5::kda_gate_beta(d_gin.ptr.cast(), d_dt.ptr.cast(), d_sa.ptr.cast(), -5.0, dk as i32, nh as i32, t_steps as i32, d_g1.ptr.cast(), d_b2.ptr.cast(), &stream);
        glm5::kda_recurrence(d_qc.ptr.cast(), d_oa.ptr.cast(), d_vc.ptr.cast(), d_g1.ptr.cast(), d_b2.ptr.cast(), d_st2.ptr.cast(), nh as i32, dk as i32, t_steps as i32, d_o2.ptr.cast(), &stream);
        for t in 0..t_steps {
            let off_f = t * di * 4;
            let off_b = t * nh * 4;
            glm5::kda_recurrence(
                d_qc.ptr.cast::<u8>().add(off_f).cast(),
                d_oa.ptr.cast::<u8>().add(off_f).cast(),
                d_vc.ptr.cast::<u8>().add(off_f).cast(),
                d_g1.ptr.cast::<u8>().add(off_f).cast(),
                d_b2.ptr.cast::<u8>().add(off_b).cast(),
                d_st1.ptr.cast(),
                nh as i32,
                dk as i32,
                1,
                d_o1.ptr.cast::<u8>().add(off_f).cast(),
                &stream,
            );
        }
    }
    stream.sync().unwrap();
    let mut o1 = vec![0.0f32; t_steps * di];
    let mut o2 = vec![0.0f32; t_steps * di];
    d_o1.copy_to_host(bytemuck::cast_slice_mut(&mut o1)).unwrap();
    d_o2.copy_to_host(bytemuck::cast_slice_mut(&mut o2)).unwrap();
    assert!(max_abs_err(&o1, &o2) < 1e-5, "KDA T=1 vs T=8 max_err={}", max_abs_err(&o1, &o2));

    // 3. DSA Parallel Top-512 Select with ties + >512 pools vs exact CPU sort
    let n_vis_pools = 600usize;
    let kpool = 4usize;
    let top_cap = 512usize;
    let pos = (n_vis_pools * kpool + 1) as i32; // +2 tail cells
    let n_sel = top_cap * kpool + 3;
    let mut scores = vec![0.0f32; n_vis_pools];
    for p in 0..n_vis_pools {
        scores[p] = if p % 17 == 0 { 3.5 } else { rng.next_f32() * 5.0 };
    }
    let mut order: Vec<usize> = (0..n_vis_pools).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap()
            .then_with(|| a.cmp(&b))
    });
    let mut expected_cells = vec![-1i32; n_sel];
    for r in 0..top_cap {
        let p = order[r] as i32;
        for m_i in 0..4 {
            expected_cells[r * 4 + m_i] = p * 4 + m_i as i32;
        }
    }
    expected_cells[top_cap * 4 + 0] = (n_vis_pools * 4) as i32;
    expected_cells[top_cap * 4 + 1] = (n_vis_pools * 4 + 1) as i32;

    let d_sc = DeviceBuffer::alloc(0, n_vis_pools * 4).unwrap();
    let d_pos = DeviceBuffer::alloc(0, 4).unwrap();
    let d_cells = DeviceBuffer::alloc(0, n_sel * 4).unwrap();
    let dummy_iq = DeviceBuffer::alloc(0, 32 * 128 * 4).unwrap();
    let dummy_pl = DeviceBuffer::alloc(0, n_vis_pools * 128 * 2).unwrap();
    let dummy_w = DeviceBuffer::alloc(0, 32 * 4).unwrap();
    d_sc.copy_from_host(bytemuck::cast_slice(&scores)).unwrap();
    d_pos.copy_from_host(bytemuck::cast_slice(&[pos])).unwrap();
    unsafe {
        // Call score_select with cur_pool_done=0 to isolate select kernel on d_sc!
        glm5::dsa_score_select(
            dummy_iq.ptr.cast(),
            dummy_pl.ptr,
            dummy_w.ptr.cast(),
            128,
            32,
            kpool as i32,
            top_cap as i32,
            true,
            n_sel as i32,
            d_pos.ptr.cast(),
            d_sc.ptr.cast(),
            d_cells.ptr.cast(),
            1,
            0,
            &stream,
        );
    }
    stream.sync().unwrap();
    let mut got_cells = vec![0i32; n_sel];
    d_cells.copy_to_host(bytemuck::cast_slice_mut(&mut got_cells)).unwrap();
    assert_eq!(got_cells, expected_cells, "Parallel top-512 select mismatch!");

    // 4. Router Sigmoid + PinnedMappedBuffer<Glm5Doorbell> end-to-end handshake
    let n_exp = 288usize;
    let mut r_logits = vec![0.0f32; n_exp];
    let mut r_bias = vec![0.0f32; n_exp];
    for e in 0..n_exp {
        r_logits[e] = rng.next_f32();
        r_bias[e] = rng.next_f32() * 0.2;
    }
    let mut res_mask = vec![1u8; n_exp];
    for e in 0..n_exp {
        if e % 3 == 0 {
            res_mask[e] = 0; // miss to CPU
        }
    }
    let mut sel_pairs: Vec<(f32, f32, usize)> = (0..n_exp)
        .map(|e| {
            let p = 1.0f32 / (1.0 + (-r_logits[e]).exp());
            (p + r_bias[e], p, e)
        })
        .collect();
    sel_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap().then_with(|| a.2.cmp(&b.2)));
    let top8 = &sel_pairs[..8];
    let sum_p: f64 = top8.iter().map(|x| x.1 as f64).sum::<f64>().max(1.0 / 16384.0);

    let d_rl = DeviceBuffer::alloc(0, n_exp * 4).unwrap();
    let d_rb = DeviceBuffer::alloc(0, n_exp * 4).unwrap();
    let d_rm = DeviceBuffer::alloc(0, n_exp).unwrap();
    let d_rx = DeviceBuffer::alloc(0, 64 * 4).unwrap();
    let d_rids = DeviceBuffer::alloc(0, 8 * 4).unwrap();
    let d_rrw = DeviceBuffer::alloc(0, 8 * 4).unwrap();
    let d_seq = DeviceBuffer::alloc_zeroed(0, 8).unwrap();
    let d_exp = DeviceBuffer::alloc_zeroed(0, 8).unwrap();
    let d_sh = DeviceBuffer::alloc_zeroed(0, 64 * 4).unwrap();
    let d_gmoe = DeviceBuffer::alloc_zeroed(0, 64 * 4).unwrap();
    let d_rout = DeviceBuffer::alloc_zeroed(0, 64 * 4).unwrap();
    d_rl.copy_from_host(bytemuck::cast_slice(&r_logits)).unwrap();
    d_rb.copy_from_host(bytemuck::cast_slice(&r_bias)).unwrap();
    d_rm.copy_from_host(&res_mask).unwrap();

    let db_buf = PinnedMappedBuffer::<glm5::Glm5Doorbell>::alloc(1).unwrap();
    let db_dev_ptr = db_buf.dev_ptr_for_device(0).unwrap();
    let db_host_ptr = db_buf.host_ptr() as usize;
    let worker = std::thread::spawn(move || unsafe {
        let db = &mut *(db_host_ptr as *mut glm5::Glm5Doorbell);
        loop {
            let rseq = std::ptr::read_volatile(&db.req_seq);
            if rseq >= 1 {
                for i in 0..64 {
                    db.out[i] = 1.25f32;
                }
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                std::ptr::write_volatile(&mut db.done_seq, rseq);
                break;
            }
            std::thread::yield_now();
        }
    });

    unsafe {
        glm5::router_sigmoid_probe(
            d_rl.ptr.cast(),
            d_rb.ptr.cast(),
            n_exp as i32,
            8,
            2.5,
            true,
            d_rm.ptr.cast(),
            d_rx.ptr.cast(),
            64,
            32,
            7,
            GgmlType::Iq2S as i32,
            GgmlType::Iq3Xxs as i32,
            10.0,
            d_rids.ptr.cast(),
            d_rrw.ptr.cast(),
            db_dev_ptr,
            d_seq.ptr.cast(),
            d_exp.ptr.cast(),
            1,
            &stream,
        );
        glm5::doorbell_wait_hc_post(
            db_dev_ptr,
            d_exp.ptr.cast(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            d_sh.ptr.cast(),
            d_gmoe.ptr.cast(),
            d_rout.ptr.cast(),
            64,
            1,
            1,
            &stream,
        );
    }
    stream.sync().unwrap();
    worker.join().unwrap();

    let mut got_ids = [0i32; 8];
    let mut got_rw = [0.0f32; 8];
    let mut got_rout = vec![0.0f32; 64];
    d_rids.copy_to_host(bytemuck::cast_slice_mut(&mut got_ids)).unwrap();
    d_rrw.copy_to_host(bytemuck::cast_slice_mut(&mut got_rw)).unwrap();
    d_rout.copy_to_host(bytemuck::cast_slice_mut(&mut got_rout)).unwrap();
    for k in 0..8 {
        assert_eq!(got_ids[k], top8[k].2 as i32);
        let ew = ((top8[k].1 as f64 / sum_p) * 2.5) as f32;
        assert!((got_rw[k] - ew).abs() < 1e-5);
    }
    for &v in &got_rout {
        assert!((v - 1.25).abs() < 1e-5);
    }
}

#[test]
fn test_glm5_dsa_lat8_mla_mtp_argmax_and_mv_batch_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).expect("CUDA GPU 0 available");
    let mut rng = Rng(0x1122334455667788);

    // 1. DSA prep_kv_ik (VRAM + Pinned RAM split), pool, score, select, q_absorb, mla_attn
    let kv_lora = 512usize;
    let idx_key = 128usize;
    let idx_heads = 32usize;
    let kpool = 4usize;
    let qk_nope = 256usize;
    let v_head = 64usize;
    let n_head = 4usize;
    let total_t = 12usize;
    let res_cap = 6usize; // cells 0..5 in VRAM, 6..11 in pinned RAM!
    let eps = 1e-5f32;

    let kv_raw: Vec<f32> = (0..(total_t * kv_lora)).map(|_| rng.next_f32() * 0.6).collect();
    let kv_nw: Vec<f32> = (0..kv_lora).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();
    let ik_raw: Vec<f32> = (0..(total_t * idx_key)).map(|_| rng.next_f32() * 0.4).collect();
    let ik_nw: Vec<f32> = (0..idx_key).map(|_| 1.0 + 0.1 * rng.next_f32()).collect();
    let ik_nb: Vec<f32> = (0..idx_key).map(|_| rng.next_f32() * 0.1).collect();
    let ig_raw: Vec<f32> = (0..(total_t * idx_key)).map(|_| rng.next_f32() * 0.5).collect();
    let ape: Vec<f32> = (0..(kpool * idx_key)).map(|_| rng.next_f32() * 0.1).collect();
    let wk_b: Vec<f32> = (0..(n_head * kv_lora * qk_nope)).map(|_| rng.next_f32() * 0.05).collect();
    let wv_b: Vec<f32> = (0..(n_head * v_head * kv_lora)).map(|_| rng.next_f32() * 0.05).collect();
    let wk_b_f16: Vec<u16> = wk_b.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let wv_b_f16: Vec<u16> = wv_b.iter().map(|&v| f16::from_f32(v).to_bits()).collect();

    let lat_stride = kv_lora + kv_lora / 16;
    let d_kvraw = DeviceBuffer::alloc(0, kv_raw.len() * 4).unwrap();
    let d_kvnw = DeviceBuffer::alloc(0, kv_nw.len() * 4).unwrap();
    let d_vlat = DeviceBuffer::alloc_zeroed(0, res_cap * lat_stride).unwrap();
    let ram_lat = PinnedMappedBuffer::<u8>::alloc(total_t * lat_stride).unwrap();
    let ram_lat_dev = ram_lat.dev_ptr_for_device(0).unwrap();
    let d_ikraw = DeviceBuffer::alloc(0, ik_raw.len() * 4).unwrap();
    let d_iknw = DeviceBuffer::alloc(0, ik_nw.len() * 4).unwrap();
    let d_iknb = DeviceBuffer::alloc(0, ik_nb.len() * 4).unwrap();
    let d_ikcache = DeviceBuffer::alloc(0, total_t * idx_key * 4).unwrap();
    let d_igraw = DeviceBuffer::alloc(0, ig_raw.len() * 4).unwrap();
    let d_igcache = DeviceBuffer::alloc(0, total_t * idx_key * 4).unwrap();
    let d_ape = DeviceBuffer::alloc(0, ape.len() * 4).unwrap();
    let n_pools = total_t / kpool;
    let d_pooled = DeviceBuffer::alloc(0, n_pools * idx_key * 2).unwrap();
    let d_wkb = DeviceBuffer::alloc(0, wk_b_f16.len() * 2).unwrap();
    let d_wvb = DeviceBuffer::alloc(0, wv_b_f16.len() * 2).unwrap();

    d_kvraw.copy_from_host(bytemuck::cast_slice(&kv_raw)).unwrap();
    d_kvnw.copy_from_host(bytemuck::cast_slice(&kv_nw)).unwrap();
    d_ikraw.copy_from_host(bytemuck::cast_slice(&ik_raw)).unwrap();
    d_iknw.copy_from_host(bytemuck::cast_slice(&ik_nw)).unwrap();
    d_iknb.copy_from_host(bytemuck::cast_slice(&ik_nb)).unwrap();
    d_igraw.copy_from_host(bytemuck::cast_slice(&ig_raw)).unwrap();
    d_ape.copy_from_host(bytemuck::cast_slice(&ape)).unwrap();
    d_wkb.copy_from_host(bytemuck::cast_slice(&wk_b_f16)).unwrap();
    d_wvb.copy_from_host(bytemuck::cast_slice(&wv_b_f16)).unwrap();

    unsafe {
        glm5::dsa_prep_kv_ik(
            d_kvraw.ptr.cast(), d_kvnw.ptr.cast(), d_vlat.ptr, ram_lat_dev, res_cap as i32,
            d_ikraw.ptr.cast(), d_iknw.ptr.cast(), d_iknb.ptr.cast(), d_ikcache.ptr.cast(),
            d_igraw.ptr.cast(), d_igcache.ptr.cast(), kv_lora as i32, idx_key as i32, 0, 0, eps,
            total_t as i32, &stream,
        );
        glm5::dsa_pool(
            d_ikcache.ptr.cast(), d_igcache.ptr.cast(), d_ape.ptr.cast(),
            idx_key as i32, kpool as i32, 0, n_pools as i32, 0, d_pooled.ptr, &stream,
        );
    }

    let q_t: Vec<f32> = (0..(n_head * qk_nope)).map(|_| rng.next_f32() * 0.25).collect();
    let iq_t: Vec<f32> = (0..(idx_heads * idx_key)).map(|_| rng.next_f32() * 0.2).collect();
    let iw_t: Vec<f32> = (0..idx_heads).map(|_| rng.next_f32() * 0.2).collect();
    let d_qt = DeviceBuffer::alloc(0, q_t.len() * 4).unwrap();
    let d_qabs = DeviceBuffer::alloc(0, n_head * kv_lora * 4).unwrap();
    let d_iqt = DeviceBuffer::alloc(0, iq_t.len() * 4).unwrap();
    let d_iwt = DeviceBuffer::alloc(0, iw_t.len() * 4).unwrap();
    let pos_cur = (total_t - 1) as i32; // all 12 cells visible!
    let n_sel = n_pools * kpool + 3;
    let d_pos = DeviceBuffer::alloc(0, 4).unwrap();
    let d_sc = DeviceBuffer::alloc(0, n_pools * 4).unwrap();
    let d_cells = DeviceBuffer::alloc(0, n_sel * 4).unwrap();
    let d_attn_out = DeviceBuffer::alloc(0, n_head * v_head * 4).unwrap();
    d_qt.copy_from_host(bytemuck::cast_slice(&q_t)).unwrap();
    d_iqt.copy_from_host(bytemuck::cast_slice(&iq_t)).unwrap();
    d_iwt.copy_from_host(bytemuck::cast_slice(&iw_t)).unwrap();
    d_pos.copy_from_host(bytemuck::cast_slice(&[pos_cur])).unwrap();

    unsafe {
        glm5::dsa_q_absorb(
            d_wkb.ptr, GgmlType::F16 as i32, d_qt.ptr.cast(), d_qabs.ptr.cast(),
            qk_nope as i32, kv_lora as i32, n_head as i32, 1, &stream,
        );
        glm5::dsa_score_select(
            d_iqt.ptr.cast(), d_pooled.ptr, d_iwt.ptr.cast(), idx_key as i32, idx_heads as i32,
            kpool as i32, n_pools as i32, true, n_sel as i32, d_pos.ptr.cast(), d_sc.ptr.cast(),
            d_cells.ptr.cast(), 1, n_pools as i32, &stream,
        );
        glm5::dsa_mla_attn(
            d_qabs.ptr.cast(), d_vlat.ptr, ram_lat_dev, res_cap as i32, d_cells.ptr.cast(),
            d_wvb.ptr, GgmlType::F16 as i32, kv_lora as i32, n_head as i32, v_head as i32,
            qk_nope as i32, n_sel as i32, d_attn_out.ptr.cast(), 1, &stream,
        );
    }
    stream.sync().unwrap();

    let mut got_attn = vec![0.0f32; n_head * v_head];
    d_attn_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_attn)).unwrap();

    // CPU F64 reference against unquantized rmsnorm(kv_raw)
    let mut lat_ref = vec![0.0f64; total_t * kv_lora];
    for t in 0..total_t {
        let row = &kv_raw[t * kv_lora..(t + 1) * kv_lora];
        let ss: f64 = row.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let inv = 1.0 / (ss / kv_lora as f64 + eps as f64).sqrt();
        for c in 0..kv_lora {
            lat_ref[t * kv_lora + c] = row[c] as f64 * inv * kv_nw[c] as f64;
        }
    }
    let mut ref_attn = vec![0.0f32; n_head * v_head];
    let scale = 1.0 / (qk_nope as f64).sqrt();
    for h in 0..n_head {
        let mut qabs = vec![0.0f64; kv_lora];
        for c in 0..kv_lora {
            let mut s = 0.0f64;
            for e in 0..qk_nope {
                s += wk_b[(h * kv_lora + c) * qk_nope + e] as f64 * q_t[h * qk_nope + e] as f64;
            }
            qabs[c] = s;
        }
        let mut probs = vec![0.0f64; total_t];
        let mut mx = f64::NEG_INFINITY;
        for cell in 0..total_t {
            let dot: f64 = (0..kv_lora).map(|c| qabs[c] * lat_ref[cell * kv_lora + c]).sum();
            probs[cell] = dot * scale;
            mx = mx.max(probs[cell]);
        }
        let den: f64 = probs.iter().map(|&p| (p - mx).exp()).sum();
        let mut ctx = vec![0.0f64; kv_lora];
        for cell in 0..total_t {
            let w = (probs[cell] - mx).exp() / den;
            for c in 0..kv_lora { ctx[c] += w * lat_ref[cell * kv_lora + c]; }
        }
        for v in 0..v_head {
            let mut s = 0.0f64;
            for c in 0..kv_lora {
                s += wv_b[(h * v_head + v) * kv_lora + c] as f64 * ctx[c];
            }
            ref_attn[h * v_head + v] = s as f32;
        }
    }
    let dsa_re = rel_err(&got_attn, &ref_attn);
    assert!(dsa_re < 2e-2, "DSA MLA (with lat8 + VRAM/RAM split) rel_err={dsa_re}");

    // 2. Argmax with pruned vocab extra_ids remap + mv_batch
    let mut logits = vec![-1.0f32; 5000];
    logits[4123] = 9.5;
    let extra_ids = vec![154828i32; 500]; // local 4123 = 4000 + 123 -> global 154828
    let d_log = DeviceBuffer::alloc(0, logits.len() * 4).unwrap();
    let d_extra = DeviceBuffer::alloc(0, extra_ids.len() * 4).unwrap();
    let d_pv = DeviceBuffer::alloc(0, 128 * 4).unwrap();
    let d_pi = DeviceBuffer::alloc(0, 128 * 4).unwrap();
    let d_tok = DeviceBuffer::alloc(0, 4).unwrap();
    d_log.copy_from_host(bytemuck::cast_slice(&logits)).unwrap();
    d_extra.copy_from_host(bytemuck::cast_slice(&extra_ids)).unwrap();
    unsafe {
        glm5::argmax(
            d_log.ptr.cast(), 5000, 4000, d_extra.ptr.cast(),
            d_pv.ptr.cast(), d_pi.ptr.cast(), d_tok.ptr.cast(), &stream,
        );
    }
    stream.sync().unwrap();
    let mut got_tok = [0i32; 1];
    d_tok.copy_to_host(bytemuck::cast_slice_mut(&mut got_tok)).unwrap();
    assert_eq!(got_tok[0], 154828);
}

#[test]
fn test_glm5_dsa_dyn_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ampllm_cuda_kernels::set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x55aa112233445566);

    let kv_lora = 512usize;
    let idx_key = 128usize;
    let idx_heads = 32usize;
    let kpool = 4usize;
    let total_t = 17usize; // tests pos = 0..16 across multiple pool boundaries
    let pool_cap = 8usize;
    let res_cap = 10usize;
    let eps = 1e-5f32;
    let lat_stride = kv_lora + kv_lora / 16;

    let kv_raw: Vec<f32> = (0..(total_t * kv_lora)).map(|_| rng.next_f32() * 0.5).collect();
    let kv_nw: Vec<f32> = (0..kv_lora).map(|_| 0.8 + rng.next_f32() * 0.4).collect();
    let ik_raw: Vec<f32> = (0..(total_t * idx_key)).map(|_| rng.next_f32() * 0.5).collect();
    let ik_nw: Vec<f32> = (0..idx_key).map(|_| 0.9 + rng.next_f32() * 0.2).collect();
    let ik_nb: Vec<f32> = (0..idx_key).map(|_| rng.next_f32() * 0.1).collect();
    let ig_raw: Vec<f32> = (0..(total_t * idx_key)).map(|_| rng.next_f32() * 0.4).collect();
    let ape: Vec<f32> = (0..(kpool * idx_key)).map(|_| rng.next_f32() * 0.2).collect();
    let iq_t: Vec<f32> = (0..(idx_heads * idx_key)).map(|_| rng.next_f32() * 0.2).collect();
    let iw_t: Vec<f32> = (0..idx_heads).map(|_| rng.next_f32() * 0.2).collect();

    let d_kvnw = DeviceBuffer::alloc(0, kv_nw.len() * 4).unwrap();
    let d_iknw = DeviceBuffer::alloc(0, ik_nw.len() * 4).unwrap();
    let d_iknb = DeviceBuffer::alloc(0, ik_nb.len() * 4).unwrap();
    let d_ape = DeviceBuffer::alloc(0, ape.len() * 4).unwrap();
    let d_iqt = DeviceBuffer::alloc(0, iq_t.len() * 4).unwrap();
    let d_iwt = DeviceBuffer::alloc(0, iw_t.len() * 4).unwrap();
    let d_kv1 = DeviceBuffer::alloc(0, kv_lora * 4).unwrap();
    let d_ik1 = DeviceBuffer::alloc(0, idx_key * 4).unwrap();
    let d_ig1 = DeviceBuffer::alloc(0, idx_key * 4).unwrap();

    d_kvnw.copy_from_host(bytemuck::cast_slice(&kv_nw)).unwrap();
    d_iknw.copy_from_host(bytemuck::cast_slice(&ik_nw)).unwrap();
    d_iknb.copy_from_host(bytemuck::cast_slice(&ik_nb)).unwrap();
    d_ape.copy_from_host(bytemuck::cast_slice(&ape)).unwrap();
    d_iqt.copy_from_host(bytemuck::cast_slice(&iq_t)).unwrap();
    d_iwt.copy_from_host(bytemuck::cast_slice(&iw_t)).unwrap();

    let d_vlat_s = DeviceBuffer::alloc_zeroed(0, res_cap * lat_stride).unwrap();
    let d_vlat_d = DeviceBuffer::alloc_zeroed(0, res_cap * lat_stride).unwrap();
    let d_ikc_s = DeviceBuffer::alloc_zeroed(0, 516 * idx_key * 4).unwrap();
    let d_ikc_d = DeviceBuffer::alloc_zeroed(0, 516 * idx_key * 4).unwrap();
    let d_igc_s = DeviceBuffer::alloc_zeroed(0, 516 * idx_key * 4).unwrap();
    let d_igc_d = DeviceBuffer::alloc_zeroed(0, 516 * idx_key * 4).unwrap();
    let d_pl_s = DeviceBuffer::alloc_zeroed(0, pool_cap * idx_key * 2).unwrap();
    let d_pl_d = DeviceBuffer::alloc_zeroed(0, pool_cap * idx_key * 2).unwrap();
    let d_sc_s = DeviceBuffer::alloc_zeroed(0, pool_cap * 4).unwrap();
    let d_sc_d = DeviceBuffer::alloc_zeroed(0, pool_cap * 4).unwrap();
    let top_cap = 2usize;
    let n_sel = top_cap * kpool + kpool - 1;
    let d_cells_s = DeviceBuffer::alloc_zeroed(0, n_sel * 4).unwrap();
    let d_cells_d = DeviceBuffer::alloc_zeroed(0, n_sel * 4).unwrap();
    let d_pos = DeviceBuffer::alloc(0, 4).unwrap();

    for pos in 0..total_t {
        d_kv1.copy_from_host(bytemuck::cast_slice(&kv_raw[pos * kv_lora..(pos + 1) * kv_lora])).unwrap();
        d_ik1.copy_from_host(bytemuck::cast_slice(&ik_raw[pos * idx_key..(pos + 1) * idx_key])).unwrap();
        d_ig1.copy_from_host(bytemuck::cast_slice(&ig_raw[pos * idx_key..(pos + 1) * idx_key])).unwrap();
        d_pos.copy_from_host(bytemuck::cast_slice(&[pos as i32])).unwrap();

        unsafe {
            // Static path
            glm5::dsa_prep_kv_ik(
                d_kv1.ptr.cast(), d_kvnw.ptr.cast(), d_vlat_s.ptr, std::ptr::null_mut(), res_cap as i32,
                d_ik1.ptr.cast(), d_iknw.ptr.cast(), d_iknb.ptr.cast(), d_ikc_s.ptr.cast(),
                d_ig1.ptr.cast(), d_igc_s.ptr.cast(), kv_lora as i32, idx_key as i32, pos as i32, 516, eps, 1, &stream,
            );
            if (pos + 1) % kpool == 0 {
                glm5::dsa_pool(
                    d_ikc_s.ptr.cast(), d_igc_s.ptr.cast(), d_ape.ptr.cast(),
                    idx_key as i32, kpool as i32, (pos / kpool) as i32, 1, 516, d_pl_s.ptr, &stream,
                );
            }
            glm5::dsa_score_select(
                d_iqt.ptr.cast(), d_pl_s.ptr, d_iwt.ptr.cast(), idx_key as i32, idx_heads as i32,
                kpool as i32, top_cap as i32, true, n_sel as i32, d_pos.ptr.cast(),
                d_sc_s.ptr.cast(), d_cells_s.ptr.cast(), 1, ((pos + 1) / kpool) as i32, &stream,
            );

            // Dynamic path
            glm5::dsa_prep_kv_ik_dyn(
                d_kv1.ptr.cast(), d_kvnw.ptr.cast(), d_vlat_d.ptr, std::ptr::null_mut(), res_cap as i32,
                d_ik1.ptr.cast(), d_iknw.ptr.cast(), d_iknb.ptr.cast(), d_ikc_d.ptr.cast(),
                d_ig1.ptr.cast(), d_igc_d.ptr.cast(), kv_lora as i32, idx_key as i32, d_pos.ptr.cast(), 516, eps, 1, &stream,
            );
            glm5::dsa_pool_m1_dyn(
                d_ikc_d.ptr.cast(), d_igc_d.ptr.cast(), d_ape.ptr.cast(),
                idx_key as i32, kpool as i32, d_pos.ptr.cast(), 516, d_pl_d.ptr, &stream,
            );
            glm5::dsa_score_select_dyn(
                d_iqt.ptr.cast(), d_pl_d.ptr, d_iwt.ptr.cast(), idx_key as i32, idx_heads as i32,
                kpool as i32, top_cap as i32, true, n_sel as i32, pool_cap as i32,
                d_pos.ptr.cast(), d_sc_d.ptr.cast(), d_cells_d.ptr.cast(), &stream,
            );
        }
        stream.sync().unwrap();

        let mut cs = vec![0i32; n_sel];
        let mut cd = vec![0i32; n_sel];
        d_cells_s.copy_to_host(bytemuck::cast_slice_mut(&mut cs)).unwrap();
        d_cells_d.copy_to_host(bytemuck::cast_slice_mut(&mut cd)).unwrap();
        assert_eq!(cs, cd, "dsa dyn cells mismatch at pos={pos}");
    }

    let mut vs = vec![0u8; res_cap * lat_stride];
    let mut vd = vec![0u8; res_cap * lat_stride];
    let mut ps = vec![0u16; pool_cap * idx_key];
    let mut pd = vec![0u16; pool_cap * idx_key];
    d_vlat_s.copy_to_host(&mut vs).unwrap();
    d_vlat_d.copy_to_host(&mut vd).unwrap();
    d_pl_s.copy_to_host(bytemuck::cast_slice_mut(&mut ps)).unwrap();
    d_pl_d.copy_to_host(bytemuck::cast_slice_mut(&mut pd)).unwrap();
    assert_eq!(vs, vd, "dsa dyn vram_lat8 mismatch");
    assert_eq!(ps, pd, "dsa dyn pooled_f16 mismatch");
}

#[test]
fn test_glm5_mmvq_batch_and_doorbell_m3_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ampllm_cuda_kernels::set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x7711993355224466);

    // 1. M=2, 3, 4 batched MMVQ vs M=1 row-by-row for Q3_K, Q4_K, Q6_K, Q8_0 at K=4096
    let k = 4096usize;
    let n = 64usize;
    let max_m = 4usize;
    let x_host: Vec<f32> = (0..(max_m * k)).map(|_| rng.next_f32() * 0.4).collect();
    let d_x = DeviceBuffer::alloc(0, x_host.len() * 4).unwrap();
    let d_q8 = DeviceBuffer::alloc(0, max_m * (k / 32) * 36).unwrap();
    let d_y1 = DeviceBuffer::alloc(0, n * 4).unwrap();
    let d_ym = DeviceBuffer::alloc(0, max_m * n * 4).unwrap();
    d_x.copy_from_host_async(bytemuck::cast_slice(&x_host), &stream).unwrap();
    unsafe {
        launch_quantize_q8_1(d_x.ptr.cast(), d_q8.ptr, max_m as i32, k as i32, stream.raw);
    }

    for &wt in &[GgmlType::Q3K, GgmlType::Q4K, GgmlType::Q6K, GgmlType::Q8_0] {
        let bpr = (k / wt.block_size()) * wt.type_size();
        let mut w_bytes = Vec::with_capacity(n * bpr);
        for _ in 0..n {
            w_bytes.extend_from_slice(&fill_glm5_quant_row(wt, k, &mut rng));
        }
        let d_w = DeviceBuffer::alloc(0, w_bytes.len()).unwrap();
        d_w.copy_from_host_async(&w_bytes, &stream).unwrap();

        let mut expected = vec![0.0f32; max_m * n];
        let q8_row_bytes = (k / 32) * 36;
        for m_i in 0..max_m {
            unsafe {
                launch_linear_q8_1(
                    wt as i32,
                    d_w.ptr,
                    d_q8.ptr.cast::<u8>().add(m_i * q8_row_bytes).cast(),
                    d_x.ptr.cast::<f32>().add(m_i * k),
                    d_y1.ptr.cast(),
                    n as i32,
                    k as i32,
                    1,
                    stream.raw,
                );
            }
            d_y1.copy_to_host_async(bytemuck::cast_slice_mut(&mut expected[m_i * n..(m_i + 1) * n]), &stream).unwrap();
            stream.sync().unwrap();
        }

        for m in [2usize, 3, 4] {
            unsafe {
                launch_linear_q8_1(
                    wt as i32,
                    d_w.ptr,
                    d_q8.ptr,
                    d_x.ptr.cast(),
                    d_ym.ptr.cast(),
                    n as i32,
                    k as i32,
                    m as i32,
                    stream.raw,
                );
            }
            let mut got = vec![0.0f32; m * n];
            d_ym.copy_to_host_async(bytemuck::cast_slice_mut(&mut got), &stream).unwrap();
            stream.sync().unwrap();
            let mae = max_abs_err(&got, &expected[..m * n]);
            assert!(mae < 1e-4, "MMVQ batch parity fail {:?} M={m}: max_abs={mae}", wt);
        }
    }

    // 2. Batched Doorbell (M=3) router + doorbell wait hc_post parity vs M=1 row-by-row
    let m = 3usize;
    let n_exp = 288usize;
    let n_embd = 64usize;
    let hc = 4usize;
    let mut logits = vec![0.0f32; m * n_exp];
    let mut bias = vec![0.0f32; n_exp];
    for v in &mut logits { *v = rng.next_f32(); }
    for v in &mut bias { *v = rng.next_f32() * 0.25; }
    let mut res_mask = vec![1u8; n_exp];
    for e in 0..n_exp { if e % 5 == 0 { res_mask[e] = 0; } }
    let x_f32: Vec<f32> = (0..(m * n_embd)).map(|_| rng.next_f32()).collect();
    let r_in: Vec<f32> = (0..(m * hc * n_embd)).map(|_| rng.next_f32()).collect();
    let post: Vec<f32> = (0..(m * hc)).map(|_| 0.5 + 0.1 * rng.next_f32()).collect();
    let comb: Vec<f32> = (0..(m * hc * hc)).map(|_| 0.25 * rng.next_f32()).collect();
    let sh: Vec<f32> = (0..(m * n_embd)).map(|_| rng.next_f32()).collect();
    let gmoe: Vec<f32> = (0..(m * n_embd)).map(|_| rng.next_f32()).collect();

    let d_l = DeviceBuffer::alloc(0, logits.len() * 4).unwrap();
    let d_b = DeviceBuffer::alloc(0, bias.len() * 4).unwrap();
    let d_rm = DeviceBuffer::alloc(0, n_exp).unwrap();
    let d_x = DeviceBuffer::alloc(0, x_f32.len() * 4).unwrap();
    let d_ri = DeviceBuffer::alloc(0, r_in.len() * 4).unwrap();
    let d_pt = DeviceBuffer::alloc(0, post.len() * 4).unwrap();
    let d_cb = DeviceBuffer::alloc(0, comb.len() * 4).unwrap();
    let d_sh = DeviceBuffer::alloc(0, sh.len() * 4).unwrap();
    let d_gm = DeviceBuffer::alloc(0, gmoe.len() * 4).unwrap();
    let d_ids_b = DeviceBuffer::alloc(0, m * 8 * 4).unwrap();
    let d_rw_b = DeviceBuffer::alloc(0, m * 8 * 4).unwrap();
    let d_ids_1 = DeviceBuffer::alloc(0, m * 8 * 4).unwrap();
    let d_rw_1 = DeviceBuffer::alloc(0, m * 8 * 4).unwrap();
    let d_ro_b = DeviceBuffer::alloc_zeroed(0, r_in.len() * 4).unwrap();
    let d_ro_1 = DeviceBuffer::alloc_zeroed(0, r_in.len() * 4).unwrap();
    let d_seq = DeviceBuffer::alloc_zeroed(0, 8).unwrap();
    let d_exp = DeviceBuffer::alloc_zeroed(0, 8).unwrap();

    d_l.copy_from_host(bytemuck::cast_slice(&logits)).unwrap();
    d_b.copy_from_host(bytemuck::cast_slice(&bias)).unwrap();
    d_rm.copy_from_host(&res_mask).unwrap();
    d_x.copy_from_host(bytemuck::cast_slice(&x_f32)).unwrap();
    d_ri.copy_from_host(bytemuck::cast_slice(&r_in)).unwrap();
    d_pt.copy_from_host(bytemuck::cast_slice(&post)).unwrap();
    d_cb.copy_from_host(bytemuck::cast_slice(&comb)).unwrap();
    d_sh.copy_from_host(bytemuck::cast_slice(&sh)).unwrap();
    d_gm.copy_from_host(bytemuck::cast_slice(&gmoe)).unwrap();

    let db_buf = PinnedMappedBuffer::<glm5::Glm5Doorbell>::alloc(1).unwrap();
    let db_dev = db_buf.dev_ptr_for_device(0).unwrap();
    let db_host = db_buf.host_ptr() as usize;
    let worker = std::thread::spawn(move || unsafe {
        let db = &mut *(db_host as *mut glm5::Glm5Doorbell);
        let mut seen = 0u64;
        loop {
            let rseq = std::ptr::read_volatile(&db.req_seq);
            if rseq > seen {
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                let mb = db.m_batch.max(1) as usize;
                for t in 0..mb {
                    for e in 0..n_embd {
                        db.out[t * n_embd + e] = 0.75 * db.x_in[t * n_embd + e] + (db.miss_count as f32) * 0.01;
                    }
                }
                std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
                std::ptr::write_volatile(&mut db.done_seq, rseq);
                seen = rseq;
                if db.layer_idx == 9 { break; }
            }
            std::thread::yield_now();
        }
    });

    // Ensure at least 1 miss for M=3 batch by making all experts missed via mask copy? Actually 1/5 missed -> >=1 miss across 24 draws! Let's also verify CPU formula directly against ref!
    unsafe {
        glm5::router_sigmoid_probe(
            d_l.ptr.cast(), d_b.ptr.cast(), n_exp as i32, 8, 2.5, true,
            d_rm.ptr.cast(), d_x.ptr.cast(), n_embd as i32, 32, 9,
            GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32, 10.0,
            d_ids_b.ptr.cast(), d_rw_b.ptr.cast(), db_dev, d_seq.ptr.cast(), d_exp.ptr.cast(),
            m as i32, &stream,
        );
        glm5::doorbell_wait_hc_post(
            db_dev, d_exp.ptr.cast(), d_ri.ptr.cast(), d_pt.ptr.cast(), d_cb.ptr.cast(),
            d_sh.ptr.cast(), d_gm.ptr.cast(), d_ro_b.ptr.cast(), n_embd as i32, hc as i32,
            m as i32, &stream,
        );
        // Compare ids/rw against null-doorbell parallel probe across M
        glm5::router_sigmoid_probe(
            d_l.ptr.cast(), d_b.ptr.cast(), n_exp as i32, 8, 2.5, true,
            d_rm.ptr.cast(), std::ptr::null(), n_embd as i32, 32, 9,
            GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32, 10.0,
            d_ids_1.ptr.cast(), d_rw_1.ptr.cast(), std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
            m as i32, &stream,
        );
    }
    stream.sync().unwrap();
    // If mc==0 (unlikely 0.8^24 ~ 0.0047), worker would still wait unless triggered; check req_seq!
    let db_snap = db_buf.as_slice()[0];
    if db_snap.req_seq == 0 {
        unsafe {
            let db = &mut *(db_buf.host_ptr() as *mut glm5::Glm5Doorbell);
            db.layer_idx = 9;
            std::ptr::write_volatile(&mut db.req_seq, 1);
        }
    }
    worker.join().unwrap();

    let mut ids_b = vec![0i32; m * 8];
    let mut ids_1 = vec![0i32; m * 8];
    let mut rw_b = vec![0.0f32; m * 8];
    let mut rw_1 = vec![0.0f32; m * 8];
    d_ids_b.copy_to_host(bytemuck::cast_slice_mut(&mut ids_b)).unwrap();
    d_ids_1.copy_to_host(bytemuck::cast_slice_mut(&mut ids_1)).unwrap();
    d_rw_b.copy_to_host(bytemuck::cast_slice_mut(&mut rw_b)).unwrap();
    d_rw_1.copy_to_host(bytemuck::cast_slice_mut(&mut rw_1)).unwrap();
    assert_eq!(ids_b, ids_1, "batched router ids mismatch");
    assert!(max_abs_err(&rw_b, &rw_1) < 1e-6, "batched router weights mismatch");

    let mut got_ro = vec![0.0f32; r_in.len()];
    d_ro_b.copy_to_host(bytemuck::cast_slice_mut(&mut got_ro)).unwrap();
    let mc_exp = ids_b.iter().filter(|&&eid| res_mask[eid as usize] == 0).count();
    let mut exp_ro = vec![0.0f32; r_in.len()];
    for t in 0..m {
        let cpu_add = if mc_exp > 0 { 1.0 } else { 0.0 };
        for e in 0..n_embd {
            let ffn = sh[t * n_embd + e] + gmoe[t * n_embd + e]
                + cpu_add * (0.75 * x_f32[t * n_embd + e] + (mc_exp as f32) * 0.01);
            for d in 0..hc {
                let mut v = ffn * post[t * hc + d];
                for s in 0..hc {
                    v += comb[t * hc * hc + d * hc + s] * r_in[(t * hc + s) * n_embd + e];
                }
                exp_ro[(t * hc + d) * n_embd + e] = v;
            }
        }
    }
    assert!(max_abs_err(&got_ro, &exp_ro) < 1e-5, "doorbell wait hc post M=3 mismatch");
    let _ = d_ro_1;
}

fn ref_quantize_q8_k(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % 256, 0);
    let nb = x.len() / 256;
    let mut out = vec![0u8; nb * 292];
    for b in 0..nb {
        let xp = &x[b * 256..(b + 1) * 256];
        let mut amax = 0.0f32;
        for &v in xp {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        let d = amax / 127.0f32;
        let id = if amax > 0.0f32 { 127.0f32 / amax } else { 0.0f32 };
        let dst = &mut out[b * 292..(b + 1) * 292];
        dst[0..4].copy_from_slice(&d.to_le_bytes());
        for g in 0..16 {
            let mut bsum = 0i32;
            for k in 0..16 {
                let scaled = (xp[g * 16 + k] * id).clamp(-127.0f32, 127.0f32);
                let q = scaled.round_ties_even() as i32;
                dst[4 + g * 16 + k] = (q as i8) as u8;
                bsum += q;
            }
            dst[260 + g * 2..260 + g * 2 + 2].copy_from_slice(&(bsum as i16).to_le_bytes());
        }
    }
    out
}

#[test]
fn test_glm5_doorbell_q8k_and_skip_gate_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ampllm_cuda_kernels::set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x2026112233445566);

    assert_eq!(std::mem::offset_of!(glm5::Glm5Doorbell, cancel_flag), 48);
    assert_eq!(std::mem::offset_of!(glm5::Glm5Doorbell, x_q8k), 64);
    assert_eq!(std::mem::offset_of!(glm5::Glm5Doorbell, miss_tok), 64 + 16 * 292);

    let n_embd = 4096usize;
    let n_exp = 288usize;
    let x_host: Vec<f32> = (0..n_embd).map(|_| rng.next_f32() * 1.5).collect();
    let logits: Vec<f32> = (0..n_exp).map(|_| rng.next_f32() * 2.0).collect();
    let bias: Vec<f32> = (0..n_exp).map(|_| rng.next_f32() * 0.1).collect();
    let res_mask = vec![0u8; n_exp]; // guarantee misses!

    let d_x = DeviceBuffer::alloc(0, n_embd * 4).unwrap();
    let d_l = DeviceBuffer::alloc(0, n_exp * 4).unwrap();
    let d_b = DeviceBuffer::alloc(0, n_exp * 4).unwrap();
    let d_rm = DeviceBuffer::alloc(0, n_exp).unwrap();
    let d_ids = DeviceBuffer::alloc(0, 32).unwrap();
    let d_rw = DeviceBuffer::alloc(0, 32).unwrap();
    let d_seq = DeviceBuffer::alloc(0, 8).unwrap();
    let d_exp = DeviceBuffer::alloc(0, 8).unwrap();
    let d_skip = DeviceBuffer::alloc(0, 4).unwrap();
    d_seq.zero_async(&stream).unwrap();
    d_exp.zero_async(&stream).unwrap();
    d_skip.zero_async(&stream).unwrap();

    d_x.copy_from_host_async(bytemuck::cast_slice(&x_host), &stream).unwrap();
    d_l.copy_from_host_async(bytemuck::cast_slice(&logits), &stream).unwrap();
    d_b.copy_from_host_async(bytemuck::cast_slice(&bias), &stream).unwrap();
    d_rm.copy_from_host_async(&res_mask, &stream).unwrap();

    let db_buf = PinnedMappedBuffer::<glm5::Glm5Doorbell>::alloc(1).unwrap();
    let db_dev = db_buf.dev_ptr_for_device(0).unwrap();

    // 1. Non-canceled M=1 router probe writes x_q8k matching CPU ref_quantize_q8_k
    unsafe {
        glm5::skip_gate(db_dev, d_skip.ptr.cast(), &stream);
        glm5::router_sigmoid_probe_skip(
            d_l.ptr.cast(), d_b.ptr.cast(), n_exp as i32, 8, 2.5, true,
            d_rm.ptr.cast(), d_x.ptr.cast(), n_embd as i32, 2048, 5,
            GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32, 10.0,
            d_ids.ptr.cast(), d_rw.ptr.cast(), db_dev, d_seq.ptr.cast(), d_exp.ptr.cast(),
            d_skip.ptr.cast(), 1, &stream,
        );
    }
    stream.sync().unwrap();

    let db_snap = db_buf.as_slice()[0];
    assert_eq!(db_snap.req_seq, 1);
    assert_eq!(db_snap.x_q8k_valid, 1);

    let ref_q8k = ref_quantize_q8_k(&x_host);
    for b in 0..(n_embd / 256) {
        let got_b = &db_snap.x_q8k[b * 292..(b + 1) * 292];
        let exp_b = &ref_q8k[b * 292..(b + 1) * 292];
        let gd = f32::from_le_bytes(got_b[0..4].try_into().unwrap());
        let ed = f32::from_le_bytes(exp_b[0..4].try_into().unwrap());
        assert!((gd - ed).abs() <= 1e-6, "q8_K scale mismatch blk {b}: got={gd} exp={ed}");
        for j in 0..256 {
            let gq = got_b[4 + j] as i8 as i32;
            let eq = exp_b[4 + j] as i8 as i32;
            assert!((gq - eq).abs() <= 1, "q8_K q mismatch blk {b}[{j}]: got={gq} exp={eq}");
        }
        for g in 0..16 {
            let mut sum = 0i32;
            for k in 0..16 {
                sum += got_b[4 + g * 16 + k] as i8 as i32;
            }
            let bs = i16::from_le_bytes(got_b[260 + g * 2..260 + g * 2 + 2].try_into().unwrap()) as i32;
            assert_eq!(bs, sum, "q8_K bsum inconsistent blk {b}[{g}]");
        }
    }

    // 2. skip_gate with cancel_flag=1 sets skip_dev=1, skips router ring, skips mv_batch/linear_q8_1_skip
    unsafe {
        let db_mut = &mut *(db_buf.host_ptr() as *mut glm5::Glm5Doorbell);
        db_mut.cancel_flag = 1;
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        glm5::skip_gate(db_dev, d_skip.ptr.cast(), &stream);
        glm5::router_sigmoid_probe_skip(
            d_l.ptr.cast(), d_b.ptr.cast(), n_exp as i32, 8, 2.5, true,
            d_rm.ptr.cast(), d_x.ptr.cast(), n_embd as i32, 2048, 6,
            GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32, 10.0,
            d_ids.ptr.cast(), d_rw.ptr.cast(), db_dev, d_seq.ptr.cast(), d_exp.ptr.cast(),
            d_skip.ptr.cast(), 1, &stream,
        );
    }
    let y_test = DeviceBuffer::alloc(0, 256 * 4).unwrap();
    let d_xq8 = DeviceBuffer::alloc(0, (n_embd / 32) * 36).unwrap();
    let d_w = DeviceBuffer::alloc(0, 256 * (n_embd / 32) * 34).unwrap();
    y_test.zero_async(&stream).unwrap();
    d_xq8.zero_async(&stream).unwrap();
    d_w.zero_async(&stream).unwrap();
    unsafe {
        glm5::linear_q8_1_skip(
            GgmlType::Q8_0 as i32, d_w.ptr, d_xq8.ptr, d_x.ptr.cast(),
            y_test.ptr.cast(), 256, n_embd as i32, 1, d_skip.ptr.cast(), &stream,
        );
        glm5::mv_batch_skip(
            &[d_w.ptr], &[y_test.ptr.cast()], &[GgmlType::Q8_0 as i32], &[256],
            None, n_embd as i32, d_xq8.ptr, d_x.ptr.cast(), d_skip.ptr.cast(), &stream,
        );
    }
    stream.sync().unwrap();

    let mut skip_host = [0i32; 1];
    let mut exp_seq_host = [99u64; 1];
    let mut seq_state_host = [0u64; 1];
    d_skip.copy_to_host_async(bytemuck::cast_slice_mut(&mut skip_host), &stream).unwrap();
    d_exp.copy_to_host_async(bytemuck::cast_slice_mut(&mut exp_seq_host), &stream).unwrap();
    d_seq.copy_to_host_async(bytemuck::cast_slice_mut(&mut seq_state_host), &stream).unwrap();
    stream.sync().unwrap();
    assert_eq!(skip_host[0], 1);
    assert_eq!(exp_seq_host[0], 0);
    assert_eq!(seq_state_host[0], 1);
    assert_eq!(db_buf.as_slice()[0].req_seq, 1);
}

#[test]
fn test_glm5_dual_lane_fetch_and_moe_plan_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ampllm_cuda_kernels::set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x5A17D0A11A0E0001);

    assert_eq!(std::mem::offset_of!(glm5::Glm5Doorbell, u_off), 56);
    assert_eq!(std::mem::offset_of!(glm5::Glm5Doorbell, d_off), 60);
    assert!(
        std::mem::offset_of!(glm5::Glm5Doorbell, promo_vram_ptr)
            > std::mem::offset_of!(glm5::Glm5Doorbell, promo_eid)
    );

    let n_embd = 512usize;
    let n_ff = 256usize;
    let n_ff_sh = 256usize;
    let n_exp = 6usize;
    let topk = 4usize;
    let gu_t = GgmlType::Iq2S;
    let d_t = GgmlType::Iq3Xxs;
    let gu_bpr = gu_t.row_bytes(n_embd);
    let d_bpr = d_t.row_bytes(n_ff);
    let g_eb = n_ff * gu_bpr;
    let d_eb = n_embd * d_bpr;
    let u_off = (g_eb + 15) & !15;
    let d_off = (u_off + g_eb + 15) & !15;
    let blob_stride = (d_off + d_eb + 15) & !15;
    let clamp_lim = 8.0f32;

    let mut g_f32 = vec![vec![0.0f32; n_ff * n_embd]; n_exp];
    let mut u_f32 = vec![vec![0.0f32; n_ff * n_embd]; n_exp];
    let mut d_f32 = vec![vec![0.0f32; n_embd * n_ff]; n_exp];
    let mut blobs = vec![vec![0u8; blob_stride]; n_exp];

    for e in 0..n_exp {
        for r in 0..n_ff {
            let gr = fill_glm5_quant_row(gu_t, n_embd, &mut rng);
            let ur = fill_glm5_quant_row(gu_t, n_embd, &mut rng);
            dequantize_row_to_f32(gu_t, &gr, &mut g_f32[e][r * n_embd..(r + 1) * n_embd]);
            dequantize_row_to_f32(gu_t, &ur, &mut u_f32[e][r * n_embd..(r + 1) * n_embd]);
            blobs[e][r * gu_bpr..(r + 1) * gu_bpr].copy_from_slice(&gr);
            blobs[e][u_off + r * gu_bpr..u_off + (r + 1) * gu_bpr].copy_from_slice(&ur);
        }
        for r in 0..n_embd {
            let dr = fill_glm5_quant_row(d_t, n_ff, &mut rng);
            dequantize_row_to_f32(d_t, &dr, &mut d_f32[e][r * n_ff..(r + 1) * n_ff]);
            blobs[e][d_off + r * d_bpr..d_off + (r + 1) * d_bpr].copy_from_slice(&dr);
        }
    }

    // e=0 in VRAM slot, e=1..=5 in pinned-mapped host buffer; select top-4 = [0, 3, 2, 1]
    // dcnt for RAM experts: e=3 -> 10 (hottest -> PCIe fetch + spare promo), e=2 -> 4, e=1 -> 2 (both coldest -> CPU)
    let vram_pool = DeviceBuffer::alloc(0, 4 * blob_stride).unwrap(); // slot 0 = e0, slots 1..=2 = spares, slot 3 = scratch
    vram_pool.copy_from_host(&blobs[0]).unwrap();
    let vram_base = vram_pool.as_u64();

    let mut host_cold = ampllm_cuda_kernels::RegisteredMappedHostBuffer::alloc(n_exp * blob_stride).unwrap();
    let host_cold_dev = host_cold.dev_ptr_for_device(0).unwrap();
    unsafe {
        for e in 1..n_exp {
            std::ptr::copy_nonoverlapping(
                blobs[e].as_ptr(),
                host_cold.as_mut_ptr().add(e * blob_stride),
                blob_stride,
            );
        }
    }

    let mut vram_ptrs_h = vec![0u64; n_exp];
    vram_ptrs_h[0] = vram_base;
    let mut ram_ptrs_h = vec![0u64; n_exp];
    for e in 1..n_exp {
        ram_ptrs_h[e] = host_cold_dev + (e * blob_stride) as u64;
    }
    let spares_h = [vram_base + blob_stride as u64, vram_base + (2 * blob_stride) as u64];
    let scratch_h = [vram_base + (3 * blob_stride) as u64; 8];
    let dcnt_h = [20u32, 1, 3, 9, 0, 0]; // after router +1: e1=2, e2=4, e3=10

    let logits_h = [10.0f32, 3.0, 4.0, 5.0, -5.0, -5.0]; // top4: [0, 3, 2, 1]
    let x_h: Vec<f32> = (0..n_embd).map(|_| rng.next_f32() * 0.5).collect();
    let sh_g_h: Vec<f32> = (0..n_ff_sh).map(|_| rng.next_f32() * 1.5).collect();
    let sh_u_h: Vec<f32> = (0..n_ff_sh).map(|_| rng.next_f32() * 1.5).collect();
    let sh_out_h: Vec<f32> = (0..n_embd).map(|_| rng.next_f32() * 0.25).collect();

    let d_logits = DeviceBuffer::alloc(0, n_exp * 4).unwrap();
    let d_vram_ptrs = DeviceBuffer::alloc(0, n_exp * 8).unwrap();
    let d_ram_ptrs = DeviceBuffer::alloc(0, n_exp * 8).unwrap();
    let d_spares = DeviceBuffer::alloc(0, spares_h.len() * 8).unwrap();
    let d_dcnt = DeviceBuffer::alloc(0, n_exp * 4).unwrap();
    let d_rcnt = DeviceBuffer::alloc(0, 4).unwrap();
    let d_scratch = DeviceBuffer::alloc(0, 8 * 8).unwrap();
    let d_plan = DeviceBuffer::alloc(0, 8 * 8).unwrap();
    let d_fsrc = DeviceBuffer::alloc(0, 8 * 8).unwrap();
    let d_x = DeviceBuffer::alloc(0, n_embd * 4).unwrap();
    let d_xq8 = DeviceBuffer::alloc(0, (n_embd / 32) * 36).unwrap();
    let d_sh_g = DeviceBuffer::alloc(0, n_ff_sh * 4).unwrap();
    let d_sh_u = DeviceBuffer::alloc(0, n_ff_sh * 4).unwrap();
    let d_sh_hq = DeviceBuffer::alloc(0, (n_ff_sh / 32) * 36).unwrap();
    let d_sh_out = DeviceBuffer::alloc(0, n_embd * 4).unwrap();
    let d_ids = DeviceBuffer::alloc(0, topk * 4).unwrap();
    let d_rw = DeviceBuffer::alloc(0, topk * 4).unwrap();
    let d_iq8 = DeviceBuffer::alloc(0, topk * (n_ff / 32) * 36).unwrap();
    let d_out = DeviceBuffer::alloc(0, n_embd * 4).unwrap();
    let d_seq = DeviceBuffer::alloc(0, 16).unwrap();
    let d_exp = DeviceBuffer::alloc(0, 8).unwrap();
    let d_skip = DeviceBuffer::alloc(0, 4).unwrap();

    d_logits.copy_from_host(bytemuck::cast_slice(&logits_h)).unwrap();
    d_vram_ptrs.copy_from_host(bytemuck::cast_slice(&vram_ptrs_h)).unwrap();
    d_ram_ptrs.copy_from_host(bytemuck::cast_slice(&ram_ptrs_h)).unwrap();
    d_spares.copy_from_host(bytemuck::cast_slice(&spares_h)).unwrap();
    d_dcnt.copy_from_host(bytemuck::cast_slice(&dcnt_h)).unwrap();
    d_rcnt.zero_async(&stream).unwrap();
    d_scratch.copy_from_host(bytemuck::cast_slice(&scratch_h)).unwrap();
    d_x.copy_from_host(bytemuck::cast_slice(&x_h)).unwrap();
    d_sh_g.copy_from_host(bytemuck::cast_slice(&sh_g_h)).unwrap();
    d_sh_u.copy_from_host(bytemuck::cast_slice(&sh_u_h)).unwrap();
    d_sh_out.copy_from_host(bytemuck::cast_slice(&sh_out_h)).unwrap();
    d_seq.zero_async(&stream).unwrap();
    d_exp.zero_async(&stream).unwrap();
    d_skip.zero_async(&stream).unwrap();

    let db_buf = PinnedMappedBuffer::<glm5::Glm5Doorbell>::alloc(1).unwrap();
    let db_dev = db_buf.dev_ptr_for_device(0).unwrap();

    let d_pos_arr = DeviceBuffer::alloc_zeroed(0, 4).unwrap();
    let d_sh_down = DeviceBuffer::alloc(0, n_embd * GgmlType::Q4K.row_bytes(n_ff_sh)).unwrap();
    let mut sh_down_f32 = vec![0.0f32; n_embd * n_ff_sh];
    let mut sh_down_bytes = Vec::with_capacity(n_embd * GgmlType::Q4K.row_bytes(n_ff_sh));
    for r in 0..n_embd {
        let row = fill_glm5_quant_row(GgmlType::Q4K, n_ff_sh, &mut rng);
        dequantize_row_to_f32(GgmlType::Q4K, &row, &mut sh_down_f32[r * n_ff_sh..(r + 1) * n_ff_sh]);
        sh_down_bytes.extend_from_slice(&row);
    }
    d_sh_down.copy_from_host(&sh_down_bytes).unwrap();

    unsafe {
        launch_quantize_q8_1(d_x.ptr.cast(), d_xq8.ptr, 1, n_embd as i32, stream.raw);
        glm5::skip_gate(db_dev, d_skip.ptr.cast(), &stream);
        glm5::router_dual_lane(
            d_logits.ptr.cast(),
            std::ptr::null(),
            n_exp as i32,
            topk as i32,
            2.5,
            true,
            d_vram_ptrs.ptr.cast(),
            d_ram_ptrs.ptr.cast(),
            d_spares.ptr.cast(),
            2,
            d_dcnt.ptr.cast(),
            d_rcnt.ptr.cast(),
            d_scratch.ptr.cast(),
            0x655432210u64, // f_ram=3 -> 2 CPU (e=1,2), 1 PCIe (e=3)
            1,
            topk as i32,
            d_plan.ptr.cast(),
            d_fsrc.ptr.cast(),
            d_x.ptr.cast(),
            n_embd as i32,
            n_ff as i32,
            7,
            gu_t as i32,
            d_t as i32,
            clamp_lim,
            u_off as u32,
            d_off as u32,
            d_sh_g.ptr.cast(),
            d_sh_u.ptr.cast(),
            clamp_lim,
            n_ff_sh as i32,
            d_sh_hq.ptr,
            d_ids.ptr.cast(),
            d_rw.ptr.cast(),
            db_dev,
            d_seq.ptr.cast(),
            d_exp.ptr.cast(),
            d_pos_arr.ptr.cast(),
            d_skip.ptr.cast(),
            &stream,
        );
        glm5::moe_fetch(d_fsrc.ptr.cast(), d_plan.ptr.cast(), topk as i32, blob_stride, &stream);
        glm5::moe_plan(
            gu_t as i32,
            d_t as i32,
            d_plan.ptr.cast(),
            u_off as u32,
            d_off as u32,
            gu_bpr,
            d_bpr,
            d_xq8.ptr,
            d_rw.ptr.cast(),
            d_iq8.ptr,
            0,
            std::ptr::null(),
            0,
            std::ptr::null(),
            0,
            d_sh_out.ptr.cast(),
            d_out.ptr.cast(),
            n_embd as i32,
            n_ff as i32,
            topk as i32,
            1,
            clamp_lim,
            d_skip.ptr.cast(),
            &stream,
        );
    }
    stream.sync().unwrap();

    // Verify doorbell routing + spare promotion + sh_hq + rslot=0 + pci_fetch_total==1
    let db = db_buf.as_slice()[0];
    assert_eq!(db.miss_count, 2, "expected 2 coldest RAM experts on CPU");
    assert_eq!(db.pci_fetch_total, 1, "expected 1 PCIe fetch recorded");
    assert_eq!(db.routed_ids[7][0], 0);
    assert_eq!(db.routed_ids[7][1], 3);
    assert_eq!(db.routed_ids[7][8], -1);
    let mut cpu_ids = db.miss_ids[..2].to_vec();
    cpu_ids.sort();
    assert_eq!(cpu_ids, vec![1, 2]);
    assert_eq!(db.u_off, u_off as u32);
    assert_eq!(db.d_off, d_off as u32);
    assert_eq!(db.promo_eid[7][0], 3, "e=3 (hottest cold) should claim spare 0");
    assert_eq!(db.promo_vram_ptr[7][0], spares_h[0]);
    assert_eq!(db.promo_eid[7][1], -1);

    let mut vptrs_after = vec![0u64; n_exp];
    d_vram_ptrs.copy_to_host(bytemuck::cast_slice_mut(&mut vptrs_after)).unwrap();
    assert_eq!(vptrs_after[3], spares_h[0]);
    let mut spares_after = [0u64; 2];
    d_spares.copy_to_host(bytemuck::cast_slice_mut(&mut spares_after)).unwrap();
    assert_eq!(spares_after[0], 0);
    assert_eq!(spares_after[1], spares_h[1]);

    // Check shared SwiGLU + q8_1 matches standalone swiglu_clamp_q8
    let d_sh_hq_ref = DeviceBuffer::alloc(0, (n_ff_sh / 32) * 36).unwrap();
    unsafe {
        glm5::swiglu_clamp_q8(
            d_sh_g.ptr.cast(),
            d_sh_u.ptr.cast(),
            clamp_lim,
            n_ff_sh as i32,
            std::ptr::null_mut(),
            d_sh_hq_ref.ptr,
            1,
            &stream,
        );
    }
    stream.sync().unwrap();
    let mut got_shq = vec![0u8; (n_ff_sh / 32) * 36];
    let mut ref_shq = vec![0u8; (n_ff_sh / 32) * 36];
    d_sh_hq.copy_to_host(&mut got_shq).unwrap();
    d_sh_hq_ref.copy_to_host(&mut ref_shq).unwrap();
    assert_eq!(got_shq, ref_shq, "fused shared swiglu_q8 mismatch");

    // Verify moe_fetch + moe_plan GPU output for [e=0, e=3] + sh_out against CPU dequant ref
    let mut rw_got = vec![0.0f32; topk];
    d_rw.copy_to_host(bytemuck::cast_slice_mut(&mut rw_got)).unwrap();
    let mut ref_out = sh_out_h.clone();
    for &(slot, e) in [(0usize, 0usize), (1usize, 3usize)].iter() {
        let mut inter = vec![0.0f32; n_ff];
        for r in 0..n_ff {
            let mut sg = 0.0f64;
            let mut su = 0.0f64;
            for c in 0..n_embd {
                sg += g_f32[e][r * n_embd + c] as f64 * x_h[c] as f64;
                su += u_f32[e][r * n_embd + c] as f64 * x_h[c] as f64;
            }
            let gc = (sg as f32).min(clamp_lim);
            let uc = (su as f32).clamp(-clamp_lim, clamp_lim);
            inter[r] = (gc / (1.0 + (-gc).exp())) * uc;
        }
        for r in 0..n_embd {
            let mut sd = 0.0f64;
            for c in 0..n_ff {
                sd += d_f32[e][r * n_ff + c] as f64 * inter[c] as f64;
            }
            ref_out[r] += rw_got[slot] * sd as f32;
        }
    }
    let mut got_out = vec![0.0f32; n_embd];
    d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_out)).unwrap();
    let re = rel_err(&got_out, &ref_out);
    assert!(re < 2.5e-2, "dual-lane fetch + moe_plan rel_err={re}");

    // Verify fused Q4_K shared down path inside glm5::moe_plan + pos=1 double-buffered routed_ids
    let d_sh_out_fused = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
    let d_out_fused = DeviceBuffer::alloc(0, n_embd * 4).unwrap();
    d_pos_arr.copy_from_host(bytemuck::cast_slice(&[1i32])).unwrap();
    unsafe {
        glm5::router_dual_lane(
            d_logits.ptr.cast(),
            std::ptr::null(),
            n_exp as i32,
            topk as i32,
            2.5,
            true,
            d_vram_ptrs.ptr.cast(),
            d_ram_ptrs.ptr.cast(),
            d_spares.ptr.cast(),
            2,
            d_dcnt.ptr.cast(),
            d_rcnt.ptr.cast(),
            d_scratch.ptr.cast(),
            0x655432210u64,
            1,
            topk as i32,
            d_plan.ptr.cast(),
            d_fsrc.ptr.cast(),
            d_x.ptr.cast(),
            n_embd as i32,
            n_ff as i32,
            7,
            gu_t as i32,
            d_t as i32,
            clamp_lim,
            u_off as u32,
            d_off as u32,
            d_sh_g.ptr.cast(),
            d_sh_u.ptr.cast(),
            clamp_lim,
            n_ff_sh as i32,
            d_sh_hq.ptr,
            d_ids.ptr.cast(),
            d_rw.ptr.cast(),
            db_dev,
            d_seq.ptr.cast(),
            d_exp.ptr.cast(),
            d_pos_arr.ptr.cast(),
            d_skip.ptr.cast(),
            &stream,
        );
        glm5::moe_plan(
            gu_t as i32,
            d_t as i32,
            d_plan.ptr.cast(),
            u_off as u32,
            d_off as u32,
            gu_bpr,
            d_bpr,
            d_xq8.ptr,
            d_rw.ptr.cast(),
            d_iq8.ptr,
            GgmlType::Q4K as i32,
            d_sh_down.ptr,
            GgmlType::Q4K.row_bytes(n_ff_sh),
            d_sh_hq.ptr,
            n_ff_sh as i32,
            d_sh_out_fused.ptr.cast(),
            d_out_fused.ptr.cast(),
            n_embd as i32,
            n_ff as i32,
            topk as i32,
            1,
            clamp_lim,
            d_skip.ptr.cast(),
            &stream,
        );
    }
    stream.sync().unwrap();
    let db1 = db_buf.as_slice()[0];
    assert_eq!(db1.routed_ids[7][8], 0, "pos=1 must populate slot 8..");
    assert_eq!(db1.routed_ids[7][9], 3);
    let mut ref_fused = vec![0.0f32; n_embd];
    for r in 0..n_embd {
        let mut s_sh = 0.0f64;
        for c in 0..n_ff_sh {
            let gc = sh_g_h[c].min(clamp_lim);
            let uc = sh_u_h[c].clamp(-clamp_lim, clamp_lim);
            let h = (gc / (1.0 + (-gc).exp())) * uc;
            s_sh += sh_down_f32[r * n_ff_sh + c] as f64 * h as f64;
        }
        let mut acc = s_sh;
        for &(slot, e) in [(0usize, 0usize), (1usize, 3usize)].iter() {
            let mut inter = vec![0.0f32; n_ff];
            for k_r in 0..n_ff {
                let mut sg = 0.0f64;
                let mut su = 0.0f64;
                for c in 0..n_embd {
                    sg += g_f32[e][k_r * n_embd + c] as f64 * x_h[c] as f64;
                    su += u_f32[e][k_r * n_embd + c] as f64 * x_h[c] as f64;
                }
                let gc = (sg as f32).min(clamp_lim);
                let uc = (su as f32).clamp(-clamp_lim, clamp_lim);
                inter[k_r] = (gc / (1.0 + (-gc).exp())) * uc;
            }
            let mut sd = 0.0f64;
            for c in 0..n_ff {
                sd += d_f32[e][r * n_ff + c] as f64 * inter[c] as f64;
            }
            acc += rw_got[slot] as f64 * sd;
        }
        ref_fused[r] = acc as f32;
    }
    let mut got_out_fused = vec![0.0f32; n_embd];
    d_out_fused.copy_to_host(bytemuck::cast_slice_mut(&mut got_out_fused)).unwrap();
    let re_f = rel_err(&got_out_fused, &ref_fused);
    assert!(re_f < 2.5e-2, "fused sh_down moe_plan rel_err={re_f}");

    // Verify cancel_flag=1 aborts cleanly without touching spares[1]
    unsafe {
        let db_mut = &mut *(db_buf.host_ptr() as *mut glm5::Glm5Doorbell);
        db_mut.cancel_flag = 1;
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        glm5::skip_gate(db_dev, d_skip.ptr.cast(), &stream);
        glm5::router_dual_lane(
            d_logits.ptr.cast(),
            std::ptr::null(),
            n_exp as i32,
            topk as i32,
            2.5,
            true,
            d_vram_ptrs.ptr.cast(),
            d_ram_ptrs.ptr.cast(),
            d_spares.ptr.cast(),
            2,
            d_dcnt.ptr.cast(),
            d_rcnt.ptr.cast(),
            d_scratch.ptr.cast(),
            0x655432210u64,
            1,
            topk as i32,
            d_plan.ptr.cast(),
            d_fsrc.ptr.cast(),
            d_x.ptr.cast(),
            n_embd as i32,
            n_ff as i32,
            7,
            gu_t as i32,
            d_t as i32,
            clamp_lim,
            u_off as u32,
            d_off as u32,
            d_sh_g.ptr.cast(),
            d_sh_u.ptr.cast(),
            clamp_lim,
            n_ff_sh as i32,
            d_sh_hq.ptr,
            d_ids.ptr.cast(),
            d_rw.ptr.cast(),
            db_dev,
            d_seq.ptr.cast(),
            d_exp.ptr.cast(),
            d_pos_arr.ptr.cast(),
            d_skip.ptr.cast(),
            &stream,
        );
    }
    stream.sync().unwrap();
    d_spares.copy_to_host(bytemuck::cast_slice_mut(&mut spares_after)).unwrap();
    assert_eq!(spares_after[1], spares_h[1], "canceled router must not consume spare");
    let mut fsrc_after = [99u64; 8];
    d_fsrc.copy_to_host(bytemuck::cast_slice_mut(&mut fsrc_after)).unwrap();
    assert_eq!(fsrc_after, [0u64; 8], "canceled router must zero fetch_src");
    assert_eq!(db_buf.as_slice()[0].promo_eid[7][0], 3, "canceled router must preserve unread promo_eid[7][0]");
    assert_eq!(db_buf.as_slice()[0].promo_eid[7][1], -1);

    // Verify skip_from=2: cold misses at rank >= 2 are dropped (plan=0, fetch_src=0, no CPU),
    // while VRAM hits at rank >= 2 and cold misses at rank < 2 still execute.
    let logits_sf = [-5.0f32, 3.0, 10.0, 2.0, -5.0, -5.0]; // top4: [2 (cold), 1 (cold), 3 (VRAM), 0 (VRAM)]
    d_logits.copy_from_host(bytemuck::cast_slice(&logits_sf)).unwrap();
    unsafe {
        let db_mut = &mut *(db_buf.host_ptr() as *mut glm5::Glm5Doorbell);
        db_mut.cancel_flag = 0;
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        glm5::skip_gate(db_dev, d_skip.ptr.cast(), &stream);
        glm5::router_dual_lane(
            d_logits.ptr.cast(),
            std::ptr::null(),
            n_exp as i32,
            topk as i32,
            2.5,
            true,
            d_vram_ptrs.ptr.cast(),
            d_ram_ptrs.ptr.cast(),
            d_spares.ptr.cast(),
            2,
            d_dcnt.ptr.cast(),
            d_rcnt.ptr.cast(),
            d_scratch.ptr.cast(),
            0u64, // force PCIe for rank < 2 cold misses to check fetch_src
            1,
            2,
            d_plan.ptr.cast(),
            d_fsrc.ptr.cast(),
            d_x.ptr.cast(),
            n_embd as i32,
            n_ff as i32,
            7,
            gu_t as i32,
            d_t as i32,
            clamp_lim,
            u_off as u32,
            d_off as u32,
            d_sh_g.ptr.cast(),
            d_sh_u.ptr.cast(),
            clamp_lim,
            n_ff_sh as i32,
            d_sh_hq.ptr,
            d_ids.ptr.cast(),
            d_rw.ptr.cast(),
            db_dev,
            d_seq.ptr.cast(),
            d_exp.ptr.cast(),
            d_pos_arr.ptr.cast(),
            d_skip.ptr.cast(),
            &stream,
        );
    }
    stream.sync().unwrap();
    let mut plan_sf = [0u64; 8];
    let mut fsrc_sf = [0u64; 8];
    let mut exp_sf = [99u64; 1];
    d_plan.copy_to_host(bytemuck::cast_slice_mut(&mut plan_sf)).unwrap();
    d_fsrc.copy_to_host(bytemuck::cast_slice_mut(&mut fsrc_sf)).unwrap();
    d_exp.copy_to_host(bytemuck::cast_slice_mut(&mut exp_sf)).unwrap();
    // Wait: logits_sf sorted descending is: 2(10.0), 1(3.0), 3(2.0), 0(-5.0)? Actually 0(-5.0), 4(-5.0), 5(-5.0) tie broken by lower id -> 0.
    // k=0 is e=2 (cold < skip_from=2 -> promoted/fetched), k=1 is e=1 (cold < skip_from=2 -> promoted/fetched? spare 1 only had 1 spare, second cold hits scratch),
    // Now test logits_sf2 where k=0=e3 (VRAM), k=1=e0 (VRAM), k=2=e2 (cold >= 2), k=3=e1 (cold >= 2)
    assert_ne!(fsrc_sf[0], 0, "rank 0 cold miss should be fetched");
    assert_eq!(plan_sf[2], spares_h[0], "VRAM hit at rank 2 >= skip_from must still run");
    assert_eq!(plan_sf[3], vram_base, "VRAM hit at rank 3 >= skip_from must still run");
    assert_eq!(exp_sf[0], 0, "cpu_plan=0 expected no CPU sequence");

    let logits_sf3 = [5.0f32, -5.0, -5.0, 10.0, 2.0, 1.0]; // top4: [3 (VRAM), 0 (VRAM), 4 (cold >= 2), 5 (cold >= 2)]
    d_logits.copy_from_host(bytemuck::cast_slice(&logits_sf3)).unwrap();
    let pci_before = db_buf.as_slice()[0].pci_fetch_total;
    unsafe {
        glm5::router_dual_lane(
            d_logits.ptr.cast(),
            std::ptr::null(),
            n_exp as i32,
            topk as i32,
            2.5,
            true,
            d_vram_ptrs.ptr.cast(),
            d_ram_ptrs.ptr.cast(),
            d_spares.ptr.cast(),
            2,
            d_dcnt.ptr.cast(),
            d_rcnt.ptr.cast(),
            d_scratch.ptr.cast(),
            0x655432210u64,
            1,
            2,
            d_plan.ptr.cast(),
            d_fsrc.ptr.cast(),
            d_x.ptr.cast(),
            n_embd as i32,
            n_ff as i32,
            7,
            gu_t as i32,
            d_t as i32,
            clamp_lim,
            u_off as u32,
            d_off as u32,
            d_sh_g.ptr.cast(),
            d_sh_u.ptr.cast(),
            clamp_lim,
            n_ff_sh as i32,
            d_sh_hq.ptr,
            d_ids.ptr.cast(),
            d_rw.ptr.cast(),
            db_dev,
            d_seq.ptr.cast(),
            d_exp.ptr.cast(),
            d_pos_arr.ptr.cast(),
            d_skip.ptr.cast(),
            &stream,
        );
    }
    stream.sync().unwrap();
    d_plan.copy_to_host(bytemuck::cast_slice_mut(&mut plan_sf)).unwrap();
    d_fsrc.copy_to_host(bytemuck::cast_slice_mut(&mut fsrc_sf)).unwrap();
    d_exp.copy_to_host(bytemuck::cast_slice_mut(&mut exp_sf)).unwrap();
    assert_eq!(plan_sf[0], spares_h[0]);
    assert_eq!(plan_sf[1], vram_base);
    assert_eq!(plan_sf[2], 0, "cold miss e=4 at rank 2 >= skip_from=2 must be skipped");
    assert_eq!(plan_sf[3], 0, "cold miss e=5 at rank 3 >= skip_from=2 must be skipped");
    assert_eq!(fsrc_sf, [0u64; 8], "skipped cold misses must not set fetch_src");
    assert_eq!(exp_sf[0], 0, "skipped cold misses must not wake CPU");
    assert_eq!(db_buf.as_slice()[0].pci_fetch_total, pci_before);
}

#[test]
fn test_glm5_fused_m1_kernels_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use ampllm_cuda_kernels::{
        ffi::launch_quantize_q8_1, glm5, set_device, CudaStream, DeviceBuffer,
    };
    set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();

    fn next_rng(s: &mut u32) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        let u = (*s >> 8) as f32 / ((1u32 << 24) as f32);
        u * 1.6 - 0.8
    }
    fn f32_to_bf16(v: f32) -> u16 {
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
    }
    fn f32_to_f16(v: f32) -> u16 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = b & 0x007F_FFFF;
        if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7C00
        } else {
            let round = (mant + 0x1000) >> 13;
            (sign as u32 | ((exp as u32) << 10) | (round & 0x3FF)) as u16
        }
    }
    fn f16_to_f32(h: u16) -> f32 {
        let sign = ((h as u32) & 0x8000) << 16;
        let exp = (h >> 10) & 0x1F;
        let mant = (h as u32) & 0x3FF;
        if exp == 0 {
            if mant == 0 {
                f32::from_bits(sign)
            } else {
                let mut e = 127 - 14;
                let mut m = mant;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x3FF;
                f32::from_bits(sign | ((e as u32) << 23) | (m << 13))
            }
        } else if exp == 31 {
            f32::from_bits(sign | 0x7F80_0000 | (mant << 13))
        } else {
            f32::from_bits(sign | (((exp as u32) + 127 - 15) << 23) | (mant << 13))
        }
    }
    fn dequant_q8_1(raw: &[u8], k: usize) -> Vec<f32> {
        let nb = k / 32;
        let mut out = vec![0.0f32; k];
        for b in 0..nb {
            let off = b * 36;
            let d = f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]]));
            for i in 0..32 {
                out[b * 32 + i] = (raw[off + 4 + i] as i8) as f32 * d;
            }
        }
        out
    }
    fn encode_q8_0_mat(rows: usize, k: usize, rng: &mut u32) -> Vec<u8> {
        let nb = k / 32;
        let mut out = Vec::with_capacity(rows * nb * 34);
        let mut tmp = [0.0f32; 32];
        for _r in 0..rows {
            for _b in 0..nb {
                let mut amax = 0.0f32;
                for i in 0..32 {
                    tmp[i] = next_rng(rng) * 0.25;
                    amax = amax.max(tmp[i].abs());
                }
                let d = amax / 127.0;
                let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
                out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
                for i in 0..32 {
                    let q = (tmp[i] * inv).round().clamp(-127.0, 127.0) as i8;
                    out.push(q as u8);
                }
            }
        }
        out
    }
    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let mut m = 0.0f32;
        for i in 0..a.len() {
            let d = (a[i] - b[i]).abs();
            assert!(!d.is_nan(), "NaN at index {i}: a={} b={}", a[i], b[i]);
            m = m.max(d);
        }
        m
    }
    fn copy_f32(dev: i32, data: &[f32]) -> DeviceBuffer {
        let buf = DeviceBuffer::alloc(dev, data.len() * 4).unwrap();
        buf.copy_from_host(bytemuck::cast_slice(data)).unwrap();
        buf
    }
    fn read_f32(buf: &DeviceBuffer, len: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; len];
        buf.copy_to_host(bytemuck::cast_slice_mut(&mut v)).unwrap();
        v
    }

    let mut rng = 0x5EED_A4B1u32;

    // 1. hc_fused_m1: null block_out AND non-null aliased post/comb
    {
        let n_embd = 4096usize;
        let hc = 4usize;
        let iters = 20i32;
        let norm_eps = 1e-5f32;
        let hc_eps = 1e-6f32;
        let n_hc_out = hc * hc + 2 * hc; // 24

        let r0: Vec<f32> = (0..hc * n_embd).map(|_| next_rng(&mut rng)).collect();
        let blk_out: Vec<f32> = (0..n_embd).map(|_| next_rng(&mut rng) * 0.5).collect();
        let post_init: Vec<f32> = (0..hc).map(|i| 0.25 + 0.05 * i as f32).collect();
        let comb_init: Vec<f32> = (0..hc * hc).map(|i| if i % 5 == 0 { 0.85 } else { 0.05 }).collect();
        let w_fn_f32: Vec<f32> = (0..n_hc_out * hc * n_embd).map(|_| next_rng(&mut rng) * 0.02).collect();
        let w_fn_bf16: Vec<u16> = w_fn_f32.iter().map(|&v| f32_to_bf16(v)).collect();
        let w_scale: Vec<f32> = (0..n_hc_out).map(|_| 0.1).collect();
        let w_base: Vec<f32> = (0..n_hc_out).map(|_| next_rng(&mut rng) * 0.1).collect();
        let norm_w: Vec<f32> = (0..n_embd).map(|_| 0.8 + 0.4 * next_rng(&mut rng)).collect();

        let d_r0 = copy_f32(0, &r0);
        let d_blk = copy_f32(0, &blk_out);
        let d_wfn = DeviceBuffer::alloc(0, w_fn_bf16.len() * 2).unwrap();
        d_wfn.copy_from_host(bytemuck::cast_slice(&w_fn_bf16)).unwrap();
        let d_ws = copy_f32(0, &w_scale);
        let d_wb = copy_f32(0, &w_base);
        let d_nw = copy_f32(0, &norm_w);

        let d_hc_part = DeviceBuffer::alloc_zeroed(0, 26 * 32 * 4).unwrap();
        let d_hc_ctr = DeviceBuffer::alloc_zeroed(0, 8).unwrap();

        // Case A: null block_out
        let d_post_ref = DeviceBuffer::alloc_zeroed(0, hc * 4).unwrap();
        let d_comb_ref = DeviceBuffer::alloc_zeroed(0, hc * hc * 4).unwrap();
        let d_xf32_ref = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let d_xq8_ref = DeviceBuffer::alloc_zeroed(0, (n_embd / 32) * 36).unwrap();

        let d_post_fus = DeviceBuffer::alloc_zeroed(0, hc * 4).unwrap();
        let d_comb_fus = DeviceBuffer::alloc_zeroed(0, hc * hc * 4).unwrap();
        let d_xf32_fus = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let d_xq8_fus = DeviceBuffer::alloc_zeroed(0, (n_embd / 32) * 36).unwrap();

        unsafe {
            glm5::hc_pre_norm_q8(
                d_r0.ptr.cast(), d_wfn.ptr, GgmlType::Bf16 as i32, d_ws.ptr.cast(), d_wb.ptr.cast(),
                d_nw.ptr.cast(), n_embd as i32, hc as i32, iters, norm_eps, hc_eps,
                d_post_ref.ptr.cast(), d_comb_ref.ptr.cast(), d_xf32_ref.ptr.cast(), d_xq8_ref.ptr, 1, &stream,
            );
            glm5::hc_fused_m1(
                std::ptr::null(), d_r0.ptr.cast(), std::ptr::null_mut(), std::ptr::null(), std::ptr::null(),
                d_post_fus.ptr.cast(), d_comb_fus.ptr.cast(), d_wfn.ptr, GgmlType::Bf16 as i32,
                d_ws.ptr.cast(), d_wb.ptr.cast(), d_nw.ptr.cast(), n_embd as i32, hc as i32, iters,
                norm_eps, hc_eps, d_xf32_fus.ptr.cast(), d_xq8_fus.ptr,
                d_hc_part.ptr.cast(), d_hc_ctr.ptr.cast(), std::ptr::null(), &stream,
            );
        }
        stream.sync().unwrap();
        let pr = read_f32(&d_post_ref, hc);
        let pf = read_f32(&d_post_fus, hc);
        let cr = read_f32(&d_comb_ref, hc * hc);
        let cf = read_f32(&d_comb_fus, hc * hc);
        let xr = read_f32(&d_xf32_ref, n_embd);
        let xf = read_f32(&d_xf32_fus, n_embd);
        let mut qr_b = vec![0u8; (n_embd / 32) * 36];
        let mut qf_b = vec![0u8; (n_embd / 32) * 36];
        d_xq8_ref.copy_to_host(&mut qr_b).unwrap();
        d_xq8_fus.copy_to_host(&mut qf_b).unwrap();
        assert!(max_abs_diff(&pr, &pf) < 1e-4, "hc null post diff");
        assert!(max_abs_diff(&cr, &cf) < 1e-4, "hc null comb diff");
        assert!(max_abs_diff(&xr, &xf) < 1e-3, "hc null xf32 diff");
        assert!(max_abs_diff(&dequant_q8_1(&qr_b, n_embd), &dequant_q8_1(&qf_b, n_embd)) < 5e-2, "hc null xq8 diff");

        // Case B: non-null block_out with in-place aliased post_in==post_out & comb_in==comb_out
        let d_post_in = copy_f32(0, &post_init);
        let d_comb_in = copy_f32(0, &comb_init);
        let d_r1_ref = DeviceBuffer::alloc_zeroed(0, hc * n_embd * 4).unwrap();
        let d_r1_fus = DeviceBuffer::alloc_zeroed(0, hc * n_embd * 4).unwrap();
        d_post_fus.copy_from_host(bytemuck::cast_slice(&post_init)).unwrap();
        d_comb_fus.copy_from_host(bytemuck::cast_slice(&comb_init)).unwrap();

        unsafe {
            glm5::hc_post_add(
                d_r0.ptr.cast(), d_post_in.ptr.cast(), d_comb_in.ptr.cast(), d_blk.ptr.cast(),
                std::ptr::null(), std::ptr::null(), d_r1_ref.ptr.cast(), n_embd as i32, hc as i32, 1, &stream,
            );
            glm5::hc_pre_norm_q8(
                d_r1_ref.ptr.cast(), d_wfn.ptr, GgmlType::Bf16 as i32, d_ws.ptr.cast(), d_wb.ptr.cast(),
                d_nw.ptr.cast(), n_embd as i32, hc as i32, iters, norm_eps, hc_eps,
                d_post_ref.ptr.cast(), d_comb_ref.ptr.cast(), d_xf32_ref.ptr.cast(), d_xq8_ref.ptr, 1, &stream,
            );
            glm5::hc_fused_m1(
                d_blk.ptr.cast(), d_r0.ptr.cast(), d_r1_fus.ptr.cast(),
                d_post_fus.ptr.cast(), d_comb_fus.ptr.cast(),
                d_post_fus.ptr.cast(), d_comb_fus.ptr.cast(),
                d_wfn.ptr, GgmlType::Bf16 as i32, d_ws.ptr.cast(), d_wb.ptr.cast(), d_nw.ptr.cast(),
                n_embd as i32, hc as i32, iters, norm_eps, hc_eps, d_xf32_fus.ptr.cast(), d_xq8_fus.ptr,
                d_hc_part.ptr.cast(), d_hc_ctr.ptr.cast(), std::ptr::null(), &stream,
            );
        }
        stream.sync().unwrap();
        assert!(max_abs_diff(&read_f32(&d_r1_ref, hc * n_embd), &read_f32(&d_r1_fus, hc * n_embd)) < 1e-5, "hc post r1 diff");
        assert!(max_abs_diff(&read_f32(&d_post_ref, hc), &read_f32(&d_post_fus, hc)) < 1e-4, "hc aliased post diff");
        assert!(max_abs_diff(&read_f32(&d_comb_ref, hc * hc), &read_f32(&d_comb_fus, hc * hc)) < 1e-4, "hc aliased comb diff");
        assert!(max_abs_diff(&read_f32(&d_xf32_ref, n_embd), &read_f32(&d_xf32_fus, n_embd)) < 1e-3, "hc aliased xf32 diff");
        d_xq8_ref.copy_to_host(&mut qr_b).unwrap();
        d_xq8_fus.copy_to_host(&mut qf_b).unwrap();
        assert!(max_abs_diff(&dequant_q8_1(&qr_b, n_embd), &dequant_q8_1(&qf_b, n_embd)) < 5e-2, "hc aliased xq8 diff");
    }

    // 2. kda_prep_m1 + kda_rec_gate_q8_m1 (n_head=64, head_dim=128, d_conv=4)
    {
        let n_head = 64usize;
        let dk = 128usize;
        let d_conv = 4usize;
        let d_inner = n_head * dk; // 8192
        let eps = 1e-5f32;
        let lb = 1e-4f32;

        let qp: Vec<f32> = (0..d_inner).map(|_| next_rng(&mut rng)).collect();
        let kp: Vec<f32> = (0..d_inner).map(|_| next_rng(&mut rng)).collect();
        let vp: Vec<f32> = (0..d_inner).map(|_| next_rng(&mut rng)).collect();
        let cw_q: Vec<f32> = (0..d_conv * d_inner).map(|_| next_rng(&mut rng) * 0.5).collect();
        let cw_k: Vec<f32> = (0..d_conv * d_inner).map(|_| next_rng(&mut rng) * 0.5).collect();
        let cw_v: Vec<f32> = (0..d_conv * d_inner).map(|_| next_rng(&mut rng) * 0.5).collect();
        let cst_q: Vec<f32> = (0..d_conv * d_inner).map(|_| next_rng(&mut rng) * 0.3).collect();
        let cst_k: Vec<f32> = (0..d_conv * d_inner).map(|_| next_rng(&mut rng) * 0.3).collect();
        let cst_v: Vec<f32> = (0..d_conv * d_inner).map(|_| next_rng(&mut rng) * 0.3).collect();
        let fa: Vec<f32> = (0..dk).map(|_| next_rng(&mut rng)).collect();
        let ga: Vec<f32> = (0..dk).map(|_| next_rng(&mut rng)).collect();
        let fb_q8 = encode_q8_0_mat(d_inner, dk, &mut rng);
        let gb_q8 = encode_q8_0_mat(d_inner, dk, &mut rng);
        let dt_bias: Vec<f32> = (0..n_head).map(|_| next_rng(&mut rng) * 0.5).collect();
        let ssm_a: Vec<f32> = (0..n_head).map(|_| -(0.2 + next_rng(&mut rng).abs())).collect();
        let beta_raw: Vec<f32> = (0..n_head).map(|_| next_rng(&mut rng)).collect();
        let s_init: Vec<f32> = (0..n_head * dk * dk).map(|_| next_rng(&mut rng) * 0.1).collect();
        let norm_w: Vec<f32> = (0..dk).map(|_| 0.9 + 0.2 * next_rng(&mut rng)).collect();

        let d_qp = copy_f32(0, &qp);
        let d_kp = copy_f32(0, &kp);
        let d_vp = copy_f32(0, &vp);
        let d_cwq = copy_f32(0, &cw_q);
        let d_cwk = copy_f32(0, &cw_k);
        let d_cwv = copy_f32(0, &cw_v);
        let d_fa = copy_f32(0, &fa);
        let d_ga = copy_f32(0, &ga);
        let d_fb = DeviceBuffer::alloc(0, fb_q8.len()).unwrap();
        let d_gb = DeviceBuffer::alloc(0, gb_q8.len()).unwrap();
        d_fb.copy_from_host(&fb_q8).unwrap();
        d_gb.copy_from_host(&gb_q8).unwrap();
        let d_dt = copy_f32(0, &dt_bias);
        let d_sa = copy_f32(0, &ssm_a);
        let d_nw = copy_f32(0, &norm_w);

        let d_cstq_ref = copy_f32(0, &cst_q);
        let d_cstk_ref = copy_f32(0, &cst_k);
        let d_cstv_ref = copy_f32(0, &cst_v);
        let d_beta_ref = copy_f32(0, &beta_raw);
        let d_s_ref = copy_f32(0, &s_init);
        let d_qc_ref = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_kc_ref = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_vc_ref = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_faq8_ref = DeviceBuffer::alloc_zeroed(0, (dk / 32) * 36).unwrap();
        let d_gaq8_ref = DeviceBuffer::alloc_zeroed(0, (dk / 32) * 36).unwrap();
        let d_graw_ref = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_g1_ref = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_g2_ref = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_scan_ref = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_gf_ref = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_gq8_ref = DeviceBuffer::alloc_zeroed(0, (d_inner / 32) * 36).unwrap();

        let d_cstq_fus = copy_f32(0, &cst_q);
        let d_cstk_fus = copy_f32(0, &cst_k);
        let d_cstv_fus = copy_f32(0, &cst_v);
        let d_beta_fus = copy_f32(0, &beta_raw);
        let d_s_fus = copy_f32(0, &s_init);
        let d_qc_fus = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_kc_fus = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_vc_fus = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_g1_fus = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_g2_fus = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_gf_fus = DeviceBuffer::alloc_zeroed(0, d_inner * 4).unwrap();
        let d_gq8_fus = DeviceBuffer::alloc_zeroed(0, (d_inner / 32) * 36).unwrap();

        unsafe {
            glm5::kda_conv_silu_l2(d_qp.ptr.cast(), d_cwq.ptr.cast(), d_cstq_ref.ptr.cast(), d_inner as i32, d_conv as i32, dk as i32, n_head as i32, true, 1, d_qc_ref.ptr.cast(), &stream);
            glm5::kda_conv_silu_l2(d_kp.ptr.cast(), d_cwk.ptr.cast(), d_cstk_ref.ptr.cast(), d_inner as i32, d_conv as i32, dk as i32, n_head as i32, true, 1, d_kc_ref.ptr.cast(), &stream);
            glm5::kda_conv_silu_l2(d_vp.ptr.cast(), d_cwv.ptr.cast(), d_cstv_ref.ptr.cast(), d_inner as i32, d_conv as i32, dk as i32, n_head as i32, false, 1, d_vc_ref.ptr.cast(), &stream);
            launch_quantize_q8_1(d_fa.ptr.cast(), d_faq8_ref.ptr, 1, dk as i32, stream.raw);
            launch_quantize_q8_1(d_ga.ptr.cast(), d_gaq8_ref.ptr, 1, dk as i32, stream.raw);
            launch_linear_q8_1(GgmlType::Q8_0 as i32, d_fb.ptr, d_faq8_ref.ptr, d_fa.ptr.cast(), d_graw_ref.ptr.cast(), d_inner as i32, dk as i32, 1, stream.raw);
            launch_linear_q8_1(GgmlType::Q8_0 as i32, d_gb.ptr, d_gaq8_ref.ptr, d_ga.ptr.cast(), d_g2_ref.ptr.cast(), d_inner as i32, dk as i32, 1, stream.raw);
            glm5::kda_gate_beta(d_graw_ref.ptr.cast(), d_dt.ptr.cast(), d_sa.ptr.cast(), lb, dk as i32, n_head as i32, 1, d_g1_ref.ptr.cast(), d_beta_ref.ptr.cast(), &stream);
            glm5::kda_recurrence(d_qc_ref.ptr.cast(), d_kc_ref.ptr.cast(), d_vc_ref.ptr.cast(), d_g1_ref.ptr.cast(), d_beta_ref.ptr.cast(), d_s_ref.ptr.cast(), n_head as i32, dk as i32, 1, d_scan_ref.ptr.cast(), &stream);
            glm5::kda_out_gate_q8(d_scan_ref.ptr.cast(), d_nw.ptr.cast(), d_g2_ref.ptr.cast(), dk as i32, n_head as i32, 1, eps, d_gf_ref.ptr.cast(), d_gq8_ref.ptr, &stream);

            glm5::kda_prep_m1(
                d_qp.ptr.cast(), d_kp.ptr.cast(), d_vp.ptr.cast(), d_cwq.ptr.cast(), d_cwk.ptr.cast(), d_cwv.ptr.cast(),
                d_cstq_fus.ptr.cast(), d_cstk_fus.ptr.cast(), d_cstv_fus.ptr.cast(),
                d_qc_fus.ptr.cast(), d_kc_fus.ptr.cast(), d_vc_fus.ptr.cast(),
                d_fa.ptr.cast(), d_ga.ptr.cast(), d_fb.ptr, d_gb.ptr, d_dt.ptr.cast(), d_sa.ptr.cast(), lb,
                n_head as i32, d_g1_fus.ptr.cast(), d_g2_fus.ptr.cast(), std::ptr::null(), &stream,
            );
            glm5::kda_rec_gate_q8_m1(
                d_qc_fus.ptr.cast(), d_kc_fus.ptr.cast(), d_vc_fus.ptr.cast(), d_g1_fus.ptr.cast(), d_beta_fus.ptr.cast(),
                d_s_fus.ptr.cast(), d_g2_fus.ptr.cast(), d_nw.ptr.cast(), eps, n_head as i32,
                d_gf_fus.ptr.cast(), d_gq8_fus.ptr, std::ptr::null(), &stream,
            );
        }
        stream.sync().unwrap();
        assert!(max_abs_diff(&read_f32(&d_qc_ref, d_inner), &read_f32(&d_qc_fus, d_inner)) < 1e-4, "kda qc diff");
        assert!(max_abs_diff(&read_f32(&d_kc_ref, d_inner), &read_f32(&d_kc_fus, d_inner)) < 1e-4, "kda kc diff");
        assert!(max_abs_diff(&read_f32(&d_vc_ref, d_inner), &read_f32(&d_vc_fus, d_inner)) < 1e-4, "kda vc diff");
        assert!(max_abs_diff(&read_f32(&d_g1_ref, d_inner), &read_f32(&d_g1_fus, d_inner)) < 1e-2, "kda g1 diff");
        assert!(max_abs_diff(&read_f32(&d_g2_ref, d_inner), &read_f32(&d_g2_fus, d_inner)) < 1e-2, "kda g2 diff");
        assert!(max_abs_diff(&read_f32(&d_s_ref, n_head * dk * dk), &read_f32(&d_s_fus, n_head * dk * dk)) < 1e-3, "kda state diff");
        assert!(max_abs_diff(&read_f32(&d_gf_ref, d_inner), &read_f32(&d_gf_fus, d_inner)) < 5e-3, "kda gated f32 diff");
        let mut qr_b = vec![0u8; (d_inner / 32) * 36];
        let mut qf_b = vec![0u8; (d_inner / 32) * 36];
        d_gq8_ref.copy_to_host(&mut qr_b).unwrap();
        d_gq8_fus.copy_to_host(&mut qf_b).unwrap();
        assert!(max_abs_diff(&dequant_q8_1(&qr_b, d_inner), &dequant_q8_1(&qf_b, d_inner)) < 2e-2, "kda gated q8 diff");
    }

    // 3. dsa_prep_m1_dyn (pos=0..7) + 4. dsa_mla_vproj_q8_m1 (n_head=64, v_head=256)
    {
        let q_lora = 1536usize;
        let kv_lora = 512usize;
        let idx_key = 128usize;
        let kpool = 4usize;
        let ring_cap = 516usize;
        let res_cap = 64usize;
        let pool_cap = 16usize;
        let eps = 1e-5f32;
        let lat_row_bytes = kv_lora + 4; // quantized row + fp32 scale

        let qnw: Vec<f32> = (0..q_lora).map(|_| 0.9 + 0.2 * next_rng(&mut rng)).collect();
        let kvnw: Vec<f32> = (0..kv_lora).map(|_| 0.9 + 0.2 * next_rng(&mut rng)).collect();
        let iknw: Vec<f32> = (0..idx_key).map(|_| 0.9 + 0.2 * next_rng(&mut rng)).collect();
        let iknb: Vec<f32> = (0..idx_key).map(|_| next_rng(&mut rng) * 0.1).collect();
        let ape: Vec<f32> = (0..idx_key).map(|_| next_rng(&mut rng).abs() * 0.5).collect();

        let d_qnw = copy_f32(0, &qnw);
        let d_kvnw = copy_f32(0, &kvnw);
        let d_iknw = copy_f32(0, &iknw);
        let d_iknb = copy_f32(0, &iknb);
        let d_ape = copy_f32(0, &ape);
        let d_pos = DeviceBuffer::alloc_zeroed(0, 4).unwrap();

        let d_qr_ref = DeviceBuffer::alloc_zeroed(0, q_lora * 4).unwrap();
        let d_qrq8_ref = DeviceBuffer::alloc_zeroed(0, (q_lora / 32) * 36).unwrap();
        let d_vram_ref = DeviceBuffer::alloc_zeroed(0, res_cap * lat_row_bytes).unwrap();
        let d_ikr_ref = DeviceBuffer::alloc_zeroed(0, ring_cap * idx_key * 4).unwrap();
        let d_igr_ref = DeviceBuffer::alloc_zeroed(0, ring_cap * idx_key * 4).unwrap();
        let d_pool_ref = DeviceBuffer::alloc_zeroed(0, pool_cap * idx_key * 2).unwrap();

        let d_qr_fus = DeviceBuffer::alloc_zeroed(0, q_lora * 4).unwrap();
        let d_qrq8_fus = DeviceBuffer::alloc_zeroed(0, (q_lora / 32) * 36).unwrap();
        let d_vram_fus = DeviceBuffer::alloc_zeroed(0, res_cap * lat_row_bytes).unwrap();
        let d_ikr_fus = DeviceBuffer::alloc_zeroed(0, ring_cap * idx_key * 4).unwrap();
        let d_igr_fus = DeviceBuffer::alloc_zeroed(0, ring_cap * idx_key * 4).unwrap();
        let d_pool_fus = DeviceBuffer::alloc_zeroed(0, pool_cap * idx_key * 2).unwrap();

        for pos in 0i32..8 {
            let qr_in: Vec<f32> = (0..q_lora).map(|_| next_rng(&mut rng)).collect();
            let kv_in: Vec<f32> = (0..kv_lora).map(|_| next_rng(&mut rng)).collect();
            let ik_in: Vec<f32> = (0..idx_key).map(|_| next_rng(&mut rng)).collect();
            let ig_in: Vec<f32> = (0..idx_key).map(|_| next_rng(&mut rng)).collect();
            d_qr_ref.copy_from_host(bytemuck::cast_slice(&qr_in)).unwrap();
            d_qr_fus.copy_from_host(bytemuck::cast_slice(&qr_in)).unwrap();
            let d_kv = copy_f32(0, &kv_in);
            let d_ik = copy_f32(0, &ik_in);
            let d_ig = copy_f32(0, &ig_in);
            d_pos.copy_from_host(bytemuck::cast_slice(&[pos])).unwrap();

            unsafe {
                glm5::rms_norm_q8(d_qr_ref.ptr.cast(), std::ptr::null(), d_qnw.ptr.cast(), eps, q_lora as i32, false, d_qr_ref.ptr.cast(), d_qrq8_ref.ptr, 1, &stream);
                glm5::dsa_prep_kv_ik_dyn(
                    d_kv.ptr.cast(), d_kvnw.ptr.cast(), d_vram_ref.ptr, std::ptr::null_mut(), res_cap as i32,
                    d_ik.ptr.cast(), d_iknw.ptr.cast(), d_iknb.ptr.cast(), d_ikr_ref.ptr.cast(), d_ig.ptr.cast(), d_igr_ref.ptr.cast(),
                    kv_lora as i32, idx_key as i32, d_pos.ptr.cast(), ring_cap as i32, eps, 1, &stream,
                );
                glm5::dsa_pool_m1_dyn(d_ikr_ref.ptr.cast(), d_igr_ref.ptr.cast(), d_ape.ptr.cast(), idx_key as i32, kpool as i32, d_pos.ptr.cast(), ring_cap as i32, d_pool_ref.ptr, &stream);

                glm5::dsa_prep_m1_dyn(
                    d_qr_fus.ptr.cast(), d_qnw.ptr.cast(), d_qrq8_fus.ptr, q_lora as i32,
                    d_kv.ptr.cast(), d_kvnw.ptr.cast(), d_vram_fus.ptr, std::ptr::null_mut(), res_cap as i32, kv_lora as i32,
                    d_ik.ptr.cast(), d_iknw.ptr.cast(), d_iknb.ptr.cast(), d_ikr_fus.ptr.cast(),
                    d_ig.ptr.cast(), d_igr_fus.ptr.cast(), d_ape.ptr.cast(),
                    idx_key as i32, kpool as i32, ring_cap as i32, d_pos.ptr.cast(), d_pool_fus.ptr, eps, std::ptr::null(), &stream,
                );
            }
            stream.sync().unwrap();
            assert!(max_abs_diff(&read_f32(&d_qr_ref, q_lora), &read_f32(&d_qr_fus, q_lora)) < 1e-4, "dsa prep qr diff at pos={pos}");
            let mut qr_b = vec![0u8; (q_lora / 32) * 36];
            let mut qf_b = vec![0u8; (q_lora / 32) * 36];
            d_qrq8_ref.copy_to_host(&mut qr_b).unwrap();
            d_qrq8_fus.copy_to_host(&mut qf_b).unwrap();
            assert!(max_abs_diff(&dequant_q8_1(&qr_b, q_lora), &dequant_q8_1(&qf_b, q_lora)) < 2e-2, "dsa prep qrq8 diff at pos={pos}");
            assert!(max_abs_diff(&read_f32(&d_ikr_ref, ring_cap * idx_key), &read_f32(&d_ikr_fus, ring_cap * idx_key)) < 1e-4, "dsa ik_ring diff at pos={pos}");
            assert!(max_abs_diff(&read_f32(&d_igr_ref, ring_cap * idx_key), &read_f32(&d_igr_fus, ring_cap * idx_key)) < 1e-5, "dsa ig_ring diff at pos={pos}");
            let mut vr = vec![0u8; res_cap * lat_row_bytes];
            let mut vf = vec![0u8; res_cap * lat_row_bytes];
            d_vram_ref.copy_to_host(&mut vr).unwrap();
            d_vram_fus.copy_to_host(&mut vf).unwrap();
            assert_eq!(vr, vf, "dsa vram_lat8 byte mismatch at pos={pos}");
            let mut pr_u = vec![0u16; pool_cap * idx_key];
            let mut pf_u = vec![0u16; pool_cap * idx_key];
            d_pool_ref.copy_to_host(bytemuck::cast_slice_mut(&mut pr_u)).unwrap();
            d_pool_fus.copy_to_host(bytemuck::cast_slice_mut(&mut pf_u)).unwrap();
            let pr_f: Vec<f32> = pr_u.into_iter().map(f16_to_f32).collect();
            let pf_f: Vec<f32> = pf_u.into_iter().map(f16_to_f32).collect();
            assert!(max_abs_diff(&pr_f, &pf_f) < 5e-3, "dsa pooled_f16 diff at pos={pos}");
        }

        // 4. dsa_mla_vproj_q8_m1 (n_head=64, v_head=256, n_sel=8 valid cells)
        let n_head = 64usize;
        let v_head = 256usize;
        let qk_nope = 192usize;
        let n_sel = 8usize;
        let v_tot = n_head * v_head; // 16384

        let q_abs: Vec<f32> = (0..n_head * kv_lora).map(|_| next_rng(&mut rng) * 0.5).collect();
        let cells: Vec<i32> = (0..n_sel as i32).collect();
        let wv_f16: Vec<u16> = (0..n_head * v_head * kv_lora).map(|_| f32_to_f16(next_rng(&mut rng) * 0.2)).collect();

        let d_qabs = copy_f32(0, &q_abs);
        let d_ctx = copy_f32(0, &q_abs); // test in-place overwrite of q_abs as ctx_scratch
        let d_cells = DeviceBuffer::alloc(0, n_sel * 4).unwrap();
        d_cells.copy_from_host(bytemuck::cast_slice(&cells)).unwrap();
        let d_wvb = DeviceBuffer::alloc(0, wv_f16.len() * 2).unwrap();
        d_wvb.copy_from_host(bytemuck::cast_slice(&wv_f16)).unwrap();

        let d_out_ref = DeviceBuffer::alloc_zeroed(0, v_tot * 4).unwrap();
        let d_oq8_ref = DeviceBuffer::alloc_zeroed(0, (v_tot / 32) * 36).unwrap();
        let d_out_fus = DeviceBuffer::alloc_zeroed(0, v_tot * 4).unwrap();
        let d_oq8_fus = DeviceBuffer::alloc_zeroed(0, (v_tot / 32) * 36).unwrap();

        unsafe {
            glm5::dsa_mla_attn(
                d_qabs.ptr.cast(), d_vram_ref.ptr, std::ptr::null(), res_cap as i32, d_cells.ptr.cast(),
                d_wvb.ptr, GgmlType::F16 as i32, kv_lora as i32, n_head as i32, v_head as i32,
                qk_nope as i32, n_sel as i32, d_out_ref.ptr.cast(), 1, &stream,
            );
            launch_quantize_q8_1(d_out_ref.ptr.cast(), d_oq8_ref.ptr, 1, v_tot as i32, stream.raw);

            glm5::dsa_mla_vproj_q8_m1(
                d_ctx.ptr.cast(), d_vram_fus.ptr, std::ptr::null(), res_cap as i32, d_cells.ptr.cast(),
                d_wvb.ptr, GgmlType::F16 as i32, kv_lora as i32, n_head as i32, v_head as i32,
                qk_nope as i32, n_sel as i32, d_ctx.ptr.cast(), std::ptr::null_mut(), d_out_fus.ptr.cast(),
                d_oq8_fus.ptr, std::ptr::null(), &stream,
            );
        }
        stream.sync().unwrap();
        assert!(max_abs_diff(&read_f32(&d_out_ref, v_tot), &read_f32(&d_out_fus, v_tot)) < 2e-3, "dsa mla vproj f32 diff");
        let mut or_b = vec![0u8; (v_tot / 32) * 36];
        let mut of_b = vec![0u8; (v_tot / 32) * 36];
        d_oq8_ref.copy_to_host(&mut or_b).unwrap();
        d_oq8_fus.copy_to_host(&mut of_b).unwrap();
        assert!(max_abs_diff(&dequant_q8_1(&or_b, v_tot), &dequant_q8_1(&of_b, v_tot)) < 2e-2, "dsa mla vproj q8 diff");
    }
}

#[test]
fn test_glm5_a4c_pair_mv_wait_add_and_prefetch_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use ampllm_cuda_kernels::{
        ffi::{launch_linear_q8_1_skip, launch_quantize_q8_1},
        glm5, set_device, CudaStream, DeviceBuffer, PinnedMappedBuffer, RegisteredMappedHostBuffer,
    };
    use ampllm_gguf::vec_dot_q_f32_ref;
    use std::ffi::c_void;
    set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();

    fn next_rng(s: &mut u32) -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        let u = (*s >> 8) as f32 / ((1u32 << 24) as f32);
        u * 1.6 - 0.8
    }
    fn next_u32(s: &mut u32) -> u32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        *s
    }
    fn f32_to_bf16(v: f32) -> u16 {
        let bits = v.to_bits();
        let lsb = (bits >> 16) & 1;
        ((bits.wrapping_add(0x7FFF + lsb)) >> 16) as u16
    }
    fn f32_to_f16(v: f32) -> u16 {
        let b = v.to_bits();
        let sign = ((b >> 16) & 0x8000) as u16;
        let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = b & 0x007F_FFFF;
        if exp <= 0 {
            sign
        } else if exp >= 31 {
            sign | 0x7C00
        } else {
            let round = (mant + 0x1000) >> 13;
            (sign as u32 | ((exp as u32) << 10) | (round & 0x3FF)) as u16
        }
    }
    fn make_quant_mat(ty: GgmlType, n: usize, k: usize, rng: &mut u32) -> Vec<u8> {
        let rb = ty.row_bytes(k);
        let mut out = vec![0u8; n * rb];
        for r in 0..n {
            let dst = &mut out[r * rb..(r + 1) * rb];
            match ty {
                GgmlType::F32 => {
                    let s: &mut [f32] = bytemuck::cast_slice_mut(dst);
                    for v in s.iter_mut() {
                        *v = next_rng(rng) * 0.25;
                    }
                }
                GgmlType::Bf16 => {
                    let s: &mut [u16] = bytemuck::cast_slice_mut(dst);
                    for v in s.iter_mut() {
                        *v = f32_to_bf16(next_rng(rng) * 0.25);
                    }
                }
                GgmlType::Q8_0 => {
                    let nb = k / 32;
                    for b in 0..nb {
                        let off = b * 34;
                        let d = f32_to_f16(0.02 + next_rng(rng).abs() * 0.05).to_le_bytes();
                        dst[off..off + 2].copy_from_slice(&d);
                        for j in 2..34 {
                            dst[off + j] = (next_u32(rng) & 0xFF) as u8;
                        }
                    }
                }
                GgmlType::Q3K | GgmlType::Q4K | GgmlType::Q6K => {
                    for b in dst.iter_mut() {
                        *b = (next_u32(rng) & 0xFF) as u8;
                    }
                    let bsz = ty.row_bytes(256);
                    for bi in 0..(k / 256) {
                        let bo = bi * bsz;
                        let d = f32_to_f16(0.02 + next_rng(rng).abs() * 0.04).to_le_bytes();
                        match ty {
                            GgmlType::Q6K => dst[bo + 208..bo + 210].copy_from_slice(&d),
                            GgmlType::Q3K => dst[bo + 108..bo + 110].copy_from_slice(&d),
                            GgmlType::Q4K => {
                                let dm = f32_to_f16(0.01 + next_rng(rng).abs() * 0.02).to_le_bytes();
                                dst[bo..bo + 2].copy_from_slice(&d);
                                dst[bo + 2..bo + 4].copy_from_slice(&dm);
                            }
                            _ => {}
                        }
                    }
                }
                _ => unreachable!(),
            }
        }
        out
    }
    fn check_rel(label: &str, got: &[f32], ref_v: &[f32], tol: f64) {
        assert_eq!(got.len(), ref_v.len());
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (&g, &r) in got.iter().zip(ref_v.iter()) {
            assert!(!g.is_nan(), "{label} NaN");
            let d = (g - r) as f64;
            num += d * d;
            den += (r as f64) * (r as f64);
        }
        let rel = (num / den.max(1e-12)).sqrt();
        assert!(rel < tol, "{label}: rel_err={rel:.6} >= tol={tol} got0={:?} ref0={:?}", &got[..got.len().min(4)], &ref_v[..ref_v.len().min(4)]);
    }

    let mut rng = 0xA4C0_0001u32;

    // 1. 2-row/warp glm5_mv_batch parity (Q6_K, Q4_K, Q3_K, Q8_0, BF16, F32, odd + prod shapes)
    {
        let k = 4096usize;
        let x: Vec<f32> = (0..k).map(|_| next_rng(&mut rng)).collect();
        let d_x = DeviceBuffer::alloc(0, k * 4).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&x)).unwrap();
        let d_xq8 = DeviceBuffer::alloc(0, (k / 32) * 36).unwrap();
        unsafe { launch_quantize_q8_1(d_x.ptr.cast(), d_xq8.ptr, 1, k as i32, stream.raw); }

        let jobs_spec = [
            (GgmlType::Q6K, 8192usize, 1.0f32),
            (GgmlType::Q4K, 13usize, 0.5f32),
            (GgmlType::Q3K, 17usize, 1.25f32),
            (GgmlType::Q8_0, 19usize, 0.75f32),
            (GgmlType::Bf16, 15usize, 1.0f32),
            (GgmlType::F32, 11usize, 0.9f32),
        ];
        let mut w_mats = Vec::new();
        let mut d_ws = Vec::new();
        let mut d_ys = Vec::new();
        let mut refs = Vec::new();
        for &(ty, n, sc) in &jobs_spec {
            let mat = make_quant_mat(ty, n, k, &mut rng);
            let rb = ty.row_bytes(k);
            let mut rvec = Vec::with_capacity(n);
            for r in 0..n {
                rvec.push(vec_dot_q_f32_ref(ty, &mat[r * rb..(r + 1) * rb], &x) * sc);
            }
            let dw = DeviceBuffer::alloc(0, mat.len()).unwrap();
            dw.copy_from_host(&mat).unwrap();
            let dy = DeviceBuffer::alloc_zeroed(0, n * 4).unwrap();
            w_mats.push(mat);
            d_ws.push(dw);
            d_ys.push(dy);
            refs.push(rvec);
        }
        let w_ptrs: Vec<*const c_void> = d_ws.iter().map(|b| b.ptr as *const c_void).collect();
        let y_ptrs: Vec<*mut f32> = d_ys.iter().map(|b| b.ptr.cast()).collect();
        let w_types: Vec<i32> = jobs_spec.iter().map(|&(ty, _, _)| ty as i32).collect();
        let n_outs: Vec<i32> = jobs_spec.iter().map(|&(_, n, _)| n as i32).collect();
        let scales: Vec<f32> = jobs_spec.iter().map(|&(_, _, s)| s).collect();
        unsafe {
            glm5::mv_batch(&w_ptrs, &y_ptrs, &w_types, &n_outs, Some(&scales), k as i32, d_xq8.ptr, d_x.ptr.cast(), &stream);
        }
        stream.sync().unwrap();
        for (i, &(ty, n, _)) in jobs_spec.iter().enumerate() {
            let mut got = vec![0.0f32; n];
            d_ys[i].copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
            check_rel(&format!("mv_batch_pair_{ty:?}_{n}"), &got, &refs[i], 2e-2);
        }

        // mmvq_sub256_pair_kernel parity: Q6_K (N=4096, K=8192) and odd N=17 across Q6_K/Q4_K/Q3_K
        let k2 = 8192usize;
        let x2: Vec<f32> = (0..k2).map(|_| next_rng(&mut rng)).collect();
        let d_x2 = DeviceBuffer::alloc(0, k2 * 4).unwrap();
        d_x2.copy_from_host(bytemuck::cast_slice(&x2)).unwrap();
        let d_x2q8 = DeviceBuffer::alloc(0, (k2 / 32) * 36).unwrap();
        unsafe { launch_quantize_q8_1(d_x2.ptr.cast(), d_x2q8.ptr, 1, k2 as i32, stream.raw); }
        for &(ty, n) in &[
            (GgmlType::Q6K, 4096usize),
            (GgmlType::Q6K, 17usize),
            (GgmlType::Q4K, 17usize),
            (GgmlType::Q3K, 17usize),
        ] {
            let mat = make_quant_mat(ty, n, k2, &mut rng);
            let rb = ty.row_bytes(k2);
            let ref_y: Vec<f32> = (0..n)
                .map(|r| vec_dot_q_f32_ref(ty, &mat[r * rb..(r + 1) * rb], &x2))
                .collect();
            let dw = DeviceBuffer::alloc(0, mat.len()).unwrap();
            dw.copy_from_host(&mat).unwrap();
            let dy = DeviceBuffer::alloc_zeroed(0, n * 4).unwrap();
            unsafe {
                launch_linear_q8_1_skip(ty as i32, dw.ptr, d_x2q8.ptr, std::ptr::null(), dy.ptr.cast(), n as i32, k2 as i32, 1, std::ptr::null(), stream.raw);
            }
            stream.sync().unwrap();
            let mut got = vec![0.0f32; n];
            dy.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
            check_rel(&format!("mmvq_pair_{ty:?}_{n}"), &got, &ref_y, 2e-2);
        }
    }

    // 2. doorbell_wait_add parity (exp==0 no-op and exp>0 add)
    {
        let n_embd = 4096usize;
        let mut db_buf = PinnedMappedBuffer::<glm5::Glm5Doorbell>::alloc(1).unwrap();
        let base_ffn: Vec<f32> = (0..n_embd).map(|_| next_rng(&mut rng)).collect();
        let cpu_add: Vec<f32> = (0..n_embd).map(|_| next_rng(&mut rng)).collect();
        {
            let db = &mut db_buf.as_mut_slice()[0];
            *db = glm5::Glm5Doorbell::default();
            db.done_seq = 5;
            db.out[..n_embd].copy_from_slice(&cpu_add);
        }
        let db_dev = db_buf.dev_ptr_for_device(0).unwrap();
        let d_ffn = DeviceBuffer::alloc(0, n_embd * 4).unwrap();
        d_ffn.copy_from_host(bytemuck::cast_slice(&base_ffn)).unwrap();
        let d_exp = DeviceBuffer::alloc(0, 8).unwrap();

        // exp==0 -> no-op
        d_exp.copy_from_host(bytemuck::cast_slice(&[0u64])).unwrap();
        unsafe { glm5::doorbell_wait_add(db_dev.cast_const(), d_exp.ptr.cast(), d_ffn.ptr.cast(), n_embd as i32, &stream); }
        stream.sync().unwrap();
        let mut got0 = vec![0.0f32; n_embd];
        d_ffn.copy_to_host(bytemuck::cast_slice_mut(&mut got0)).unwrap();
        check_rel("doorbell_wait_add_noop", &got0, &base_ffn, 1e-6);

        // exp==5 -> add
        d_exp.copy_from_host(bytemuck::cast_slice(&[5u64])).unwrap();
        unsafe { glm5::doorbell_wait_add(db_dev.cast_const(), d_exp.ptr.cast(), d_ffn.ptr.cast(), n_embd as i32, &stream); }
        stream.sync().unwrap();
        let mut got1 = vec![0.0f32; n_embd];
        d_ffn.copy_to_host(bytemuck::cast_slice_mut(&mut got1)).unwrap();
        let ref1: Vec<f32> = base_ffn.iter().zip(cpu_add.iter()).map(|(a, b)| a + b).collect();
        check_rel("doorbell_wait_add_sum", &got1, &ref1, 1e-6);
    }

    // 3. router_dual_lane_pf + moe_prefetch parity
    {
        let n_exp = 6usize;
        let topk = 4usize;
        let n_embd = 256usize;
        let n_ff = 256usize;
        let blob_bytes = 4096usize;

        let mut db_buf = PinnedMappedBuffer::<glm5::Glm5Doorbell>::alloc(1).unwrap();
        db_buf.as_mut_slice()[0] = glm5::Glm5Doorbell::default();
        let db_dev = db_buf.dev_ptr_for_device(0).unwrap();

        let mut host_exp = RegisteredMappedHostBuffer::alloc(n_exp * blob_bytes).unwrap();
        for e in 0..n_exp {
            let s = e * blob_bytes;
            unsafe {
                std::ptr::write_bytes(host_exp.as_mut_ptr().add(s), 0x40 + e as u8, blob_bytes);
            }
        }
        let ram_base = host_exp.dev_ptr_for_device(0).unwrap();
        let ram_ptrs: Vec<u64> = (0..n_exp).map(|e| ram_base + (e * blob_bytes) as u64).collect();
        let vram_ptrs = vec![0u64; n_exp];

        let d_scratch_pool = DeviceBuffer::alloc_zeroed(0, 8 * blob_bytes).unwrap();
        let scratch_base = d_scratch_pool.as_u64();
        let scratch_ptrs_h: Vec<u64> = (0..8).map(|i| scratch_base + (i * blob_bytes) as u64).collect();
        let d_scratch_ptrs = DeviceBuffer::alloc(0, 8 * 8).unwrap();
        d_scratch_ptrs.copy_from_host(bytemuck::cast_slice(&scratch_ptrs_h)).unwrap();

        let d_vp_l = DeviceBuffer::alloc(0, n_exp * 8).unwrap();
        let d_rp_l = DeviceBuffer::alloc(0, n_exp * 8).unwrap();
        let d_vp_n = DeviceBuffer::alloc(0, n_exp * 8).unwrap();
        let d_rp_n = DeviceBuffer::alloc(0, n_exp * 8).unwrap();
        d_vp_l.copy_from_host(bytemuck::cast_slice(&vram_ptrs)).unwrap();
        d_rp_l.copy_from_host(bytemuck::cast_slice(&ram_ptrs)).unwrap();
        d_vp_n.copy_from_host(bytemuck::cast_slice(&vram_ptrs)).unwrap();
        d_rp_n.copy_from_host(bytemuck::cast_slice(&ram_ptrs)).unwrap();

        // Layer 6 routes [0,1,2,3], predicts layer 7 top-1 cold expert e_n = 4
        let logits_6 = [5.0f32, 4.0, 3.0, 2.0, -5.0, -6.0];
        let logits_7 = [-5.0f32, -5.0, -5.0, -5.0, 9.0, -5.0];
        let d_log6 = DeviceBuffer::alloc(0, n_exp * 4).unwrap();
        let d_log7 = DeviceBuffer::alloc(0, n_exp * 4).unwrap();
        d_log6.copy_from_host(bytemuck::cast_slice(&logits_6)).unwrap();
        d_log7.copy_from_host(bytemuck::cast_slice(&logits_7)).unwrap();

        let d_plan = DeviceBuffer::alloc_zeroed(0, 8 * 8).unwrap();
        let d_fsrc = DeviceBuffer::alloc_zeroed(0, 8 * 8).unwrap();
        let d_x = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let d_ids = DeviceBuffer::alloc_zeroed(0, topk * 4).unwrap();
        let d_rw = DeviceBuffer::alloc_zeroed(0, topk * 4).unwrap();
        let d_seq = DeviceBuffer::alloc_zeroed(0, 16).unwrap();
        let d_exp = DeviceBuffer::alloc_zeroed(0, 8).unwrap();
        let d_skip = DeviceBuffer::alloc_zeroed(0, 4).unwrap();
        let d_pf_src = DeviceBuffer::alloc_zeroed(0, 16).unwrap();
        let d_pf_dst = DeviceBuffer::alloc_zeroed(0, 16).unwrap();
        let d_pf_eids = DeviceBuffer::alloc(0, 8).unwrap();
        let d_pf_ptrs = DeviceBuffer::alloc_zeroed(0, 16).unwrap();
        let d_pf_n = DeviceBuffer::alloc_zeroed(0, 4).unwrap();

        unsafe {
            glm5::router_dual_lane_pf(
                d_log6.ptr.cast(), std::ptr::null(), n_exp as i32, topk as i32, 2.5, true,
                d_vp_l.ptr.cast(), d_rp_l.ptr.cast(), std::ptr::null_mut(), 0,
                std::ptr::null_mut(), std::ptr::null_mut(), d_scratch_ptrs.ptr.cast(),
                0u64, 1, 8, d_plan.ptr.cast(), d_fsrc.ptr.cast(), d_x.ptr.cast(),
                n_embd as i32, n_ff as i32, 6, GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32,
                10.0, 0, 0, std::ptr::null(), std::ptr::null(), 10.0, 0, std::ptr::null_mut(),
                d_ids.ptr.cast(), d_rw.ptr.cast(), db_dev, d_seq.ptr.cast(), d_exp.ptr.cast(),
                std::ptr::null(), d_skip.ptr.cast(), std::ptr::null(), std::ptr::null(),
                d_log7.ptr.cast(), std::ptr::null(), d_vp_n.ptr.cast(), d_rp_n.ptr.cast(), 1,
                d_pf_src.ptr.cast(), d_pf_dst.ptr.cast(), d_pf_eids.ptr.cast(),
                d_pf_ptrs.ptr.cast(), d_pf_n.ptr.cast(), &stream,
            );
            glm5::moe_prefetch(d_pf_src.ptr.cast(), d_pf_dst.ptr.cast(), d_pf_n.ptr.cast(), blob_bytes, &stream);
        }
        stream.sync().unwrap();

        let mut pfn = [0i32; 1];
        let mut pf_e = [0i32; 2];
        let mut pf_p = [0u64; 2];
        d_pf_n.copy_to_host(bytemuck::cast_slice_mut(&mut pfn)).unwrap();
        d_pf_eids.copy_to_host(bytemuck::cast_slice_mut(&mut pf_e)).unwrap();
        d_pf_ptrs.copy_to_host(bytemuck::cast_slice_mut(&mut pf_p)).unwrap();
        assert_eq!(pfn[0], 1, "lookahead should pick 1 cold expert");
        assert_eq!(pf_e, [4, -1]);
        // (layer_idx=6 -> layer+1=7 is odd -> slot_base = 4 + (7&1)*2 = 6)
        assert_eq!(pf_p[0], scratch_ptrs_h[6]);

        let mut pool_h = vec![0u8; 8 * blob_bytes];
        d_scratch_pool.copy_to_host(&mut pool_h).unwrap();
        assert_eq!(&pool_h[6 * blob_bytes..7 * blob_bytes], &vec![0x44u8; blob_bytes]);

        // Layer 7 receives [4 (prefetched!), 0, 1, 2] -> expert 4 hits prefetch (fetch_src[0] == 0, plan == pf_p[0])
        let logits_7_run = [3.0f32, 2.0, 1.0, -5.0, 9.0, -5.0];
        d_log7.copy_from_host(bytemuck::cast_slice(&logits_7_run)).unwrap();
        unsafe {
            glm5::router_dual_lane_pf(
                d_log7.ptr.cast(), std::ptr::null(), n_exp as i32, topk as i32, 2.5, true,
                d_vp_n.ptr.cast(), d_rp_n.ptr.cast(), std::ptr::null_mut(), 0,
                std::ptr::null_mut(), std::ptr::null_mut(), d_scratch_ptrs.ptr.cast(),
                0x654432210u64, 1, 8, d_plan.ptr.cast(), d_fsrc.ptr.cast(), d_x.ptr.cast(),
                n_embd as i32, n_ff as i32, 7, GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32,
                10.0, 0, 0, std::ptr::null(), std::ptr::null(), 10.0, 0, std::ptr::null_mut(),
                d_ids.ptr.cast(), d_rw.ptr.cast(), db_dev, d_seq.ptr.cast(), d_exp.ptr.cast(),
                std::ptr::null(), d_skip.ptr.cast(), d_pf_eids.ptr.cast(), d_pf_ptrs.ptr.cast(),
                std::ptr::null(), std::ptr::null(), std::ptr::null(), std::ptr::null(), 0,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
                std::ptr::null_mut(), std::ptr::null_mut(), &stream,
            );
        }
        stream.sync().unwrap();
        let mut plan7 = [0u64; 8];
        let mut fsrc7 = [0u64; 8];
        d_plan.copy_to_host(bytemuck::cast_slice_mut(&mut plan7)).unwrap();
        d_fsrc.copy_to_host(bytemuck::cast_slice_mut(&mut fsrc7)).unwrap();
        assert_eq!(plan7[0], pf_p[0], "prefetched expert must reuse prefetch staging slot");
        assert_eq!(fsrc7[0], 0, "prefetched expert must not re-fetch over PCIe");

        // Verify skip_dev=1 zeroes pf_out_n
        d_skip.copy_from_host(bytemuck::cast_slice(&[1i32])).unwrap();
        unsafe {
            glm5::router_dual_lane_pf(
                d_log6.ptr.cast(), std::ptr::null(), n_exp as i32, topk as i32, 2.5, true,
                d_vp_l.ptr.cast(), d_rp_l.ptr.cast(), std::ptr::null_mut(), 0,
                std::ptr::null_mut(), std::ptr::null_mut(), d_scratch_ptrs.ptr.cast(),
                0u64, 1, 8, d_plan.ptr.cast(), d_fsrc.ptr.cast(), d_x.ptr.cast(),
                n_embd as i32, n_ff as i32, 6, GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32,
                10.0, 0, 0, std::ptr::null(), std::ptr::null(), 10.0, 0, std::ptr::null_mut(),
                d_ids.ptr.cast(), d_rw.ptr.cast(), db_dev, d_seq.ptr.cast(), d_exp.ptr.cast(),
                std::ptr::null(), d_skip.ptr.cast(), std::ptr::null(), std::ptr::null(),
                d_log7.ptr.cast(), std::ptr::null(), d_vp_n.ptr.cast(), d_rp_n.ptr.cast(), 1,
                d_pf_src.ptr.cast(), d_pf_dst.ptr.cast(), d_pf_eids.ptr.cast(),
                d_pf_ptrs.ptr.cast(), d_pf_n.ptr.cast(), &stream,
            );
        }
        stream.sync().unwrap();
        d_pf_n.copy_to_host(bytemuck::cast_slice_mut(&mut pfn)).unwrap();
        assert_eq!(pfn[0], 0, "skip_dev=1 must zero pf_out_n");
    }
}

#[test]
fn bench_glm5_decode_microkernels() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use ampllm_cuda_kernels::{set_device, CudaEvent, RegisteredMappedHostBuffer};
    set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let ev_s = CudaEvent::new_timed(0).unwrap();
    let ev_e = CudaEvent::new_timed(0).unwrap();

    let n_embd = 4096usize;
    let n_ff = 2048usize;
    let n_ff_sh = 2048usize;
    let topk = 8usize;

    let gu_rb = GgmlType::Iq2S.row_bytes(n_embd);
    let d_rb = GgmlType::Iq3Xxs.row_bytes(n_ff);
    let g_eb = n_ff * gu_rb;
    let d_eb = n_embd * d_rb;
    let u_off = (g_eb + 15) & !15;
    let d_off = (u_off + g_eb + 15) & !15;
    let blob_stride = (d_off + d_eb + 15) & !15;

    // 1. moe_fetch PCIe BW from RegisteredMappedHostBuffer -> DeviceBuffer (1 and 2 blobs)
    let mut host_blobs = RegisteredMappedHostBuffer::alloc(2 * blob_stride).unwrap();
    unsafe { std::ptr::write_bytes(host_blobs.as_mut_ptr(), 0x3C, 2 * blob_stride); }
    let host_dev = host_blobs.dev_ptr_for_device(0).unwrap();
    let dev_blobs = DeviceBuffer::alloc_zeroed(0, 6 * blob_stride).unwrap();
    let mut plan_ptrs_h = [0u64; 8];
    for i in 0..6 {
        plan_ptrs_h[i] = dev_blobs.as_u64() + (i * blob_stride) as u64;
    }
    let d_plan = DeviceBuffer::alloc(0, 8 * 8).unwrap();
    let d_fsrc = DeviceBuffer::alloc(0, 8 * 8).unwrap();
    d_plan.copy_from_host(bytemuck::cast_slice(&plan_ptrs_h)).unwrap();

    for n_fetch in [1usize, 2usize] {
        let mut fsrc_h = [0u64; 8];
        for i in 0..n_fetch {
            fsrc_h[i] = host_dev + (i * blob_stride) as u64;
        }
        d_fsrc.copy_from_host(bytemuck::cast_slice(&fsrc_h)).unwrap();
        for _ in 0..5 {
            unsafe { glm5::moe_fetch(d_fsrc.ptr.cast(), d_plan.ptr.cast(), topk as i32, blob_stride, &stream); }
        }
        stream.sync().unwrap();
        ev_s.record(&stream).unwrap();
        for _ in 0..30 {
            unsafe { glm5::moe_fetch(d_fsrc.ptr.cast(), d_plan.ptr.cast(), topk as i32, blob_stride, &stream); }
        }
        ev_e.record(&stream).unwrap();
        ev_e.sync().unwrap();
        let avg_ms = ev_s.elapsed_ms(&ev_e).unwrap() / 30.0;
        let mb = (n_fetch * blob_stride) as f64 / (1024.0 * 1024.0);
        let gbps = ((n_fetch * blob_stride) as f64 / 1e9) / ((avg_ms as f64) / 1e3);
        let us_exp = (avg_ms * 1000.0) / n_fetch as f32;
        println!(
            "MICROBENCH moe_fetch n={n_fetch} ({mb:.2} MiB): {:.1} us ({us_exp:.1} us/exp), {gbps:.2} GB/s",
            avg_ms * 1000.0
        );
    }

    // 2. moe_plan (6 active of 8 experts: IQ2_S gate/up + fused Q4_K ds + IQ3_XXS down)
    let d_xq8 = DeviceBuffer::alloc_zeroed(0, (n_embd / 32) * 36).unwrap();
    let d_rw = DeviceBuffer::alloc(0, topk * 4).unwrap();
    d_rw.copy_from_host(bytemuck::cast_slice(&[0.25f32; 8])).unwrap();
    let d_iq8 = DeviceBuffer::alloc_zeroed(0, topk * (n_ff / 32) * 36).unwrap();
    let d_sh_down = DeviceBuffer::alloc_zeroed(0, n_embd * GgmlType::Q4K.row_bytes(n_ff_sh)).unwrap();
    let d_sh_hq = DeviceBuffer::alloc_zeroed(0, (n_ff_sh / 32) * 36).unwrap();
    let d_sh_out = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
    let d_out = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
    for _ in 0..5 {
        unsafe {
            glm5::moe_plan(
                GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32, d_plan.ptr.cast(),
                u_off as u32, d_off as u32, gu_rb, d_rb, d_xq8.ptr, d_rw.ptr.cast(),
                d_iq8.ptr, GgmlType::Q4K as i32, d_sh_down.ptr, GgmlType::Q4K.row_bytes(n_ff_sh),
                d_sh_hq.ptr, n_ff_sh as i32, d_sh_out.ptr.cast(), d_out.ptr.cast(),
                n_embd as i32, n_ff as i32, topk as i32, 1, 10.0, std::ptr::null(), &stream,
            );
        }
    }
    stream.sync().unwrap();
    ev_s.record(&stream).unwrap();
    for _ in 0..30 {
        unsafe {
            glm5::moe_plan(
                GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32, d_plan.ptr.cast(),
                u_off as u32, d_off as u32, gu_rb, d_rb, d_xq8.ptr, d_rw.ptr.cast(),
                d_iq8.ptr, GgmlType::Q4K as i32, d_sh_down.ptr, GgmlType::Q4K.row_bytes(n_ff_sh),
                d_sh_hq.ptr, n_ff_sh as i32, d_sh_out.ptr.cast(), d_out.ptr.cast(),
                n_embd as i32, n_ff as i32, topk as i32, 1, 10.0, std::ptr::null(), &stream,
            );
        }
    }
    ev_e.record(&stream).unwrap();
    ev_e.sync().unwrap();
    let moe_us = (ev_s.elapsed_ms(&ev_e).unwrap() / 30.0) * 1000.0;
    println!("MICROBENCH moe_plan (6/8 experts + fused Q4_K ds): {moe_us:.1} us/layer");

    // 3. mmvq_sub256 Q6_K LM head (N=151552, K=4096)
    let lm_n = 151552usize;
    let lm_bytes = lm_n * GgmlType::Q6K.row_bytes(n_embd);
    let d_lm = DeviceBuffer::alloc_zeroed(0, lm_bytes).unwrap();
    let d_logits = DeviceBuffer::alloc_zeroed(0, lm_n * 4).unwrap();
    for _ in 0..5 {
        unsafe { launch_linear_q8_1(GgmlType::Q6K as i32, d_lm.ptr, d_xq8.ptr, std::ptr::null(), d_logits.ptr.cast(), lm_n as i32, n_embd as i32, 1, stream.raw); }
    }
    stream.sync().unwrap();
    ev_s.record(&stream).unwrap();
    for _ in 0..30 {
        unsafe { launch_linear_q8_1(GgmlType::Q6K as i32, d_lm.ptr, d_xq8.ptr, std::ptr::null(), d_logits.ptr.cast(), lm_n as i32, n_embd as i32, 1, stream.raw); }
    }
    ev_e.record(&stream).unwrap();
    ev_e.sync().unwrap();
    let lm_ms = ev_s.elapsed_ms(&ev_e).unwrap() / 30.0;
    let lm_gbps = (lm_bytes as f64 / 1e9) / (lm_ms as f64 / 1e3);
    println!("MICROBENCH mmvq_sub256 Q6_K (151552x4096): {:.1} us, {lm_gbps:.2} GB/s", lm_ms * 1000.0);

    // 4. kda_recurrence_m1 (n_head=32, head_dim=128)
    let nh = 32usize;
    let dk = 128usize;
    let d_kq = DeviceBuffer::alloc_zeroed(0, nh * dk * 4).unwrap();
    let d_kb = DeviceBuffer::alloc_zeroed(0, nh * 4).unwrap();
    let d_ks = DeviceBuffer::alloc_zeroed(0, nh * dk * dk * 4).unwrap();
    let d_ko = DeviceBuffer::alloc_zeroed(0, nh * dk * 4).unwrap();
    for _ in 0..5 {
        unsafe { glm5::kda_recurrence(d_kq.ptr.cast(), d_kq.ptr.cast(), d_kq.ptr.cast(), d_kq.ptr.cast(), d_kb.ptr.cast(), d_ks.ptr.cast(), nh as i32, dk as i32, 1, d_ko.ptr.cast(), &stream); }
    }
    stream.sync().unwrap();
    ev_s.record(&stream).unwrap();
    for _ in 0..30 {
        unsafe { glm5::kda_recurrence(d_kq.ptr.cast(), d_kq.ptr.cast(), d_kq.ptr.cast(), d_kq.ptr.cast(), d_kb.ptr.cast(), d_ks.ptr.cast(), nh as i32, dk as i32, 1, d_ko.ptr.cast(), &stream); }
    }
    ev_e.record(&stream).unwrap();
    ev_e.sync().unwrap();
    let kda_us = (ev_s.elapsed_ms(&ev_e).unwrap() / 30.0) * 1000.0;
    println!("MICROBENCH kda_recurrence_m1 (32x128x128): {kda_us:.1} us/layer");

    // 5. Stage A.4b fused non-MoE M=1 kernels (GLM-5.3-Flash shapes)
    {
        let hc = 4usize;
        let d_hc_r = DeviceBuffer::alloc_zeroed(0, hc * n_embd * 4).unwrap();
        let d_hc_rn = DeviceBuffer::alloc_zeroed(0, hc * n_embd * 4).unwrap();
        let d_hc_blk = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let d_hc_p = DeviceBuffer::alloc_zeroed(0, hc * 4).unwrap();
        let d_hc_c = DeviceBuffer::alloc_zeroed(0, hc * hc * 4).unwrap();
        let d_hc_wfn = DeviceBuffer::alloc_zeroed(0, 24 * hc * n_embd * 2).unwrap();
        let d_hc_ws = DeviceBuffer::alloc_zeroed(0, 24 * 4).unwrap();
        let d_hc_wb = DeviceBuffer::alloc_zeroed(0, 24 * 4).unwrap();
        let d_hc_nw = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let d_hc_xf = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let d_hc_part = DeviceBuffer::alloc_zeroed(0, 26 * 32 * 4).unwrap();
        let d_hc_ctr = DeviceBuffer::alloc_zeroed(0, 8).unwrap();
        for _ in 0..5 {
            unsafe {
                glm5::hc_fused_m1(
                    d_hc_blk.ptr.cast(), d_hc_r.ptr.cast(), d_hc_rn.ptr.cast(),
                    d_hc_p.ptr.cast(), d_hc_c.ptr.cast(), d_hc_p.ptr.cast(), d_hc_c.ptr.cast(),
                    d_hc_wfn.ptr, GgmlType::Bf16 as i32, d_hc_ws.ptr.cast(), d_hc_wb.ptr.cast(), d_hc_nw.ptr.cast(),
                    n_embd as i32, hc as i32, 20, 1e-5, 1e-6, d_hc_xf.ptr.cast(), d_xq8.ptr,
                    d_hc_part.ptr.cast(), d_hc_ctr.ptr.cast(), std::ptr::null(), &stream,
                );
            }
        }
        stream.sync().unwrap();
        ev_s.record(&stream).unwrap();
        for _ in 0..30 {
            unsafe {
                glm5::hc_fused_m1(
                    d_hc_blk.ptr.cast(), d_hc_r.ptr.cast(), d_hc_rn.ptr.cast(),
                    d_hc_p.ptr.cast(), d_hc_c.ptr.cast(), d_hc_p.ptr.cast(), d_hc_c.ptr.cast(),
                    d_hc_wfn.ptr, GgmlType::Bf16 as i32, d_hc_ws.ptr.cast(), d_hc_wb.ptr.cast(), d_hc_nw.ptr.cast(),
                    n_embd as i32, hc as i32, 20, 1e-5, 1e-6, d_hc_xf.ptr.cast(), d_xq8.ptr,
                    d_hc_part.ptr.cast(), d_hc_ctr.ptr.cast(), std::ptr::null(), &stream,
                );
            }
        }
        ev_e.record(&stream).unwrap();
        ev_e.sync().unwrap();
        println!("MICROBENCH hc_fused_m1 (4x4096 BF16): {:.1} us/call", (ev_s.elapsed_ms(&ev_e).unwrap() / 30.0) * 1000.0);

        let nh64 = 64usize;
        let di64 = nh64 * dk;
        let d_kp = DeviceBuffer::alloc_zeroed(0, di64 * 4).unwrap();
        let d_kw = DeviceBuffer::alloc_zeroed(0, 4 * di64 * 4).unwrap();
        let d_kst = DeviceBuffer::alloc_zeroed(0, 4 * di64 * 4).unwrap();
        let d_kfa = DeviceBuffer::alloc_zeroed(0, dk * 4).unwrap();
        let d_kfb = DeviceBuffer::alloc_zeroed(0, di64 * GgmlType::Q8_0.row_bytes(dk)).unwrap();
        let d_kh = DeviceBuffer::alloc_zeroed(0, nh64 * 4).unwrap();
        let d_ks64 = DeviceBuffer::alloc_zeroed(0, nh64 * dk * dk * 4).unwrap();
        let d_knw = DeviceBuffer::alloc_zeroed(0, dk * 4).unwrap();
        let d_kgf = DeviceBuffer::alloc_zeroed(0, di64 * 4).unwrap();
        let d_kgq8 = DeviceBuffer::alloc_zeroed(0, (di64 / 32) * 36).unwrap();
        for _ in 0..5 {
            unsafe {
                glm5::kda_prep_m1(
                    d_kp.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(), d_kw.ptr.cast(), d_kw.ptr.cast(), d_kw.ptr.cast(),
                    d_kst.ptr.cast(), d_kst.ptr.cast(), d_kst.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(),
                    d_kfa.ptr.cast(), d_kfa.ptr.cast(), d_kfb.ptr, d_kfb.ptr, d_kh.ptr.cast(), d_kh.ptr.cast(), 1e-4,
                    nh64 as i32, d_kp.ptr.cast(), d_kp.ptr.cast(), std::ptr::null(), &stream,
                );
                glm5::kda_rec_gate_q8_m1(
                    d_kp.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(), d_kh.ptr.cast(),
                    d_ks64.ptr.cast(), d_kp.ptr.cast(), d_knw.ptr.cast(), 1e-5, nh64 as i32,
                    d_kgf.ptr.cast(), d_kgq8.ptr, std::ptr::null(), &stream,
                );
            }
        }
        stream.sync().unwrap();
        ev_s.record(&stream).unwrap();
        for _ in 0..30 {
            unsafe {
                glm5::kda_prep_m1(
                    d_kp.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(), d_kw.ptr.cast(), d_kw.ptr.cast(), d_kw.ptr.cast(),
                    d_kst.ptr.cast(), d_kst.ptr.cast(), d_kst.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(),
                    d_kfa.ptr.cast(), d_kfa.ptr.cast(), d_kfb.ptr, d_kfb.ptr, d_kh.ptr.cast(), d_kh.ptr.cast(), 1e-4,
                    nh64 as i32, d_kp.ptr.cast(), d_kp.ptr.cast(), std::ptr::null(), &stream,
                );
            }
        }
        ev_e.record(&stream).unwrap();
        ev_e.sync().unwrap();
        println!("MICROBENCH kda_prep_m1 (64x128): {:.1} us/layer", (ev_s.elapsed_ms(&ev_e).unwrap() / 30.0) * 1000.0);

        ev_s.record(&stream).unwrap();
        for _ in 0..30 {
            unsafe {
                glm5::kda_rec_gate_q8_m1(
                    d_kp.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(), d_kp.ptr.cast(), d_kh.ptr.cast(),
                    d_ks64.ptr.cast(), d_kp.ptr.cast(), d_knw.ptr.cast(), 1e-5, nh64 as i32,
                    std::ptr::null_mut(), d_kgq8.ptr, std::ptr::null(), &stream,
                );
            }
        }
        ev_e.record(&stream).unwrap();
        ev_e.sync().unwrap();
        println!("MICROBENCH kda_rec_gate_q8_m1 (64x128): {:.1} us/layer", (ev_s.elapsed_ms(&ev_e).unwrap() / 30.0) * 1000.0);

        let kv_lora = 512usize;
        let q_lora = 1536usize;
        let v_head = 256usize;
        let n_sel = 2051usize;
        let d_dqr = DeviceBuffer::alloc_zeroed(0, q_lora * 4).unwrap();
        let d_dqw = DeviceBuffer::alloc_zeroed(0, q_lora * 4).unwrap();
        let d_dqr8 = DeviceBuffer::alloc_zeroed(0, (q_lora / 32) * 36).unwrap();
        let d_dkv = DeviceBuffer::alloc_zeroed(0, kv_lora * 4).unwrap();
        let d_dik = DeviceBuffer::alloc_zeroed(0, 128 * 4).unwrap();
        let d_dlat = DeviceBuffer::alloc_zeroed(0, 2056 * (kv_lora + 4)).unwrap();
        let d_dikr = DeviceBuffer::alloc_zeroed(0, 516 * 128 * 4).unwrap();
        let d_dpool = DeviceBuffer::alloc_zeroed(0, 516 * 128 * 2).unwrap();
        let d_dpos = DeviceBuffer::alloc_zeroed(0, 4).unwrap();
        let d_dqabs = DeviceBuffer::alloc_zeroed(0, nh64 * kv_lora * 4).unwrap();
        let d_dcells = DeviceBuffer::alloc_zeroed(0, n_sel * 4).unwrap();
        let d_dwvb = DeviceBuffer::alloc_zeroed(0, nh64 * v_head * kv_lora * 2).unwrap();
        let d_doq8 = DeviceBuffer::alloc_zeroed(0, ((nh64 * v_head) / 32) * 36).unwrap();
        let d_split_sc = DeviceBuffer::alloc_zeroed(0, nh64 * 32 * 514 * 4).unwrap();
        for _ in 0..5 {
            unsafe {
                glm5::dsa_prep_m1_dyn(
                    d_dqr.ptr.cast(), d_dqw.ptr.cast(), d_dqr8.ptr, q_lora as i32,
                    d_dkv.ptr.cast(), d_dkv.ptr.cast(), d_dlat.ptr, std::ptr::null_mut(), 2056, kv_lora as i32,
                    d_dik.ptr.cast(), d_dik.ptr.cast(), d_dik.ptr.cast(), d_dikr.ptr.cast(),
                    d_dik.ptr.cast(), d_dikr.ptr.cast(), d_dik.ptr.cast(), 128, 4, 516,
                    d_dpos.ptr.cast(), d_dpool.ptr, 1e-5, std::ptr::null(), &stream,
                );
                glm5::dsa_mla_vproj_q8_m1(
                    d_dqabs.ptr.cast(), d_dlat.ptr, std::ptr::null(), 2056, d_dcells.ptr.cast(),
                    d_dwvb.ptr, GgmlType::F16 as i32, kv_lora as i32, nh64 as i32, v_head as i32,
                    256, n_sel as i32, d_dqabs.ptr.cast(), d_split_sc.ptr.cast(), std::ptr::null_mut(), d_doq8.ptr,
                    std::ptr::null(), &stream,
                );
            }
        }
        stream.sync().unwrap();
        ev_s.record(&stream).unwrap();
        for _ in 0..30 {
            unsafe {
                glm5::dsa_prep_m1_dyn(
                    d_dqr.ptr.cast(), d_dqw.ptr.cast(), d_dqr8.ptr, q_lora as i32,
                    d_dkv.ptr.cast(), d_dkv.ptr.cast(), d_dlat.ptr, std::ptr::null_mut(), 2056, kv_lora as i32,
                    d_dik.ptr.cast(), d_dik.ptr.cast(), d_dik.ptr.cast(), d_dikr.ptr.cast(),
                    d_dik.ptr.cast(), d_dikr.ptr.cast(), d_dik.ptr.cast(), 128, 4, 516,
                    d_dpos.ptr.cast(), d_dpool.ptr, 1e-5, std::ptr::null(), &stream,
                );
            }
        }
        ev_e.record(&stream).unwrap();
        ev_e.sync().unwrap();
        println!("MICROBENCH dsa_prep_m1_dyn: {:.1} us/layer", (ev_s.elapsed_ms(&ev_e).unwrap() / 30.0) * 1000.0);

        ev_s.record(&stream).unwrap();
        for _ in 0..30 {
            unsafe {
                glm5::dsa_mla_vproj_q8_m1(
                    d_dqabs.ptr.cast(), d_dlat.ptr, std::ptr::null(), 2056, d_dcells.ptr.cast(),
                    d_dwvb.ptr, GgmlType::F16 as i32, kv_lora as i32, nh64 as i32, v_head as i32,
                    256, n_sel as i32, d_dqabs.ptr.cast(), d_split_sc.ptr.cast(), std::ptr::null_mut(), d_doq8.ptr,
                    std::ptr::null(), &stream,
                );
            }
        }
        ev_e.record(&stream).unwrap();
        ev_e.sync().unwrap();
        println!("MICROBENCH dsa_mla_vproj_q8_m1 (split, n_head=64, n_sel=2051): {:.1} us/layer", (ev_s.elapsed_ms(&ev_e).unwrap() / 30.0) * 1000.0);

        // 6. Stage A.4c 2-row Q6_K GEMV + 32-block moe_prefetch
        let n8192 = 8192usize;
        let q6_rb4 = GgmlType::Q6K.row_bytes(n_embd);
        let q6_rb8 = GgmlType::Q6K.row_bytes(n8192);
        let d_wq6 = DeviceBuffer::alloc_zeroed(0, n8192 * q6_rb4).unwrap();
        let d_yq6 = DeviceBuffer::alloc_zeroed(0, n8192 * 4).unwrap();
        let d_yq6_2 = DeviceBuffer::alloc_zeroed(0, n8192 * 4).unwrap();
        let d_yq6_3 = DeviceBuffer::alloc_zeroed(0, n8192 * 4).unwrap();
        let wps = [d_wq6.ptr as *const std::ffi::c_void; 3];
        let yps = [d_yq6.ptr.cast::<f32>(), d_yq6_2.ptr.cast::<f32>(), d_yq6_3.ptr.cast::<f32>()];
        let wts = [GgmlType::Q6K as i32; 3];
        let nos = [n8192 as i32; 3];
        for _ in 0..5 {
            unsafe { glm5::mv_batch(&wps, &yps, &wts, &nos, None, n_embd as i32, d_xq8.ptr, std::ptr::null(), &stream); }
        }
        stream.sync().unwrap();
        ev_s.record(&stream).unwrap();
        for _ in 0..30 {
            unsafe { glm5::mv_batch(&wps, &yps, &wts, &nos, None, n_embd as i32, d_xq8.ptr, std::ptr::null(), &stream); }
        }
        ev_e.record(&stream).unwrap();
        ev_e.sync().unwrap();
        let mvb_ms = ev_s.elapsed_ms(&ev_e).unwrap() / 30.0;
        let mvb_bytes = 3 * n8192 * q6_rb4;
        println!(
            "MICROBENCH mv_batch Q6_K (3x8192 x 4096): {:.1} us, {:.2} GB/s",
            mvb_ms * 1000.0,
            (mvb_bytes as f64 / 1e9) / (mvb_ms as f64 / 1e3)
        );

        let d_xq8_8k = DeviceBuffer::alloc_zeroed(0, (n8192 / 32) * 36).unwrap();
        let d_wo8 = DeviceBuffer::alloc_zeroed(0, n_embd * q6_rb8).unwrap();
        for _ in 0..5 {
            unsafe { launch_linear_q8_1(GgmlType::Q6K as i32, d_wo8.ptr, d_xq8_8k.ptr, std::ptr::null(), d_out.ptr.cast(), n_embd as i32, n8192 as i32, 1, stream.raw); }
        }
        stream.sync().unwrap();
        ev_s.record(&stream).unwrap();
        for _ in 0..30 {
            unsafe { launch_linear_q8_1(GgmlType::Q6K as i32, d_wo8.ptr, d_xq8_8k.ptr, std::ptr::null(), d_out.ptr.cast(), n_embd as i32, n8192 as i32, 1, stream.raw); }
        }
        ev_e.record(&stream).unwrap();
        ev_e.sync().unwrap();
        let mmv_ms = ev_s.elapsed_ms(&ev_e).unwrap() / 30.0;
        let mmv_bytes = n_embd * q6_rb8;
        println!(
            "MICROBENCH mmvq_sub256 Q6_K (4096 x 8192): {:.1} us, {:.2} GB/s",
            mmv_ms * 1000.0,
            (mmv_bytes as f64 / 1e9) / (mmv_ms as f64 / 1e3)
        );

        let d_pf_src = DeviceBuffer::alloc(0, 16).unwrap();
        let d_pf_dst = DeviceBuffer::alloc(0, 16).unwrap();
        let d_pf_n = DeviceBuffer::alloc(0, 4).unwrap();
        d_pf_src.copy_from_host(bytemuck::cast_slice(&[host_dev, 0u64])).unwrap();
        d_pf_dst.copy_from_host(bytemuck::cast_slice(&[plan_ptrs_h[4], 0u64])).unwrap();
        d_pf_n.copy_from_host(bytemuck::cast_slice(&[1i32])).unwrap();
        for _ in 0..5 {
            unsafe { glm5::moe_prefetch(d_pf_src.ptr.cast(), d_pf_dst.ptr.cast(), d_pf_n.ptr.cast(), blob_stride, &stream); }
        }
        stream.sync().unwrap();
        ev_s.record(&stream).unwrap();
        for _ in 0..30 {
            unsafe { glm5::moe_prefetch(d_pf_src.ptr.cast(), d_pf_dst.ptr.cast(), d_pf_n.ptr.cast(), blob_stride, &stream); }
        }
        ev_e.record(&stream).unwrap();
        ev_e.sync().unwrap();
        let pf_ms = ev_s.elapsed_ms(&ev_e).unwrap() / 30.0;
        println!(
            "MICROBENCH moe_prefetch 32-block n=1 ({:.2} MiB): {:.1} us, {:.2} GB/s",
            blob_stride as f64 / (1024.0 * 1024.0),
            pf_ms * 1000.0,
            (blob_stride as f64 / 1e9) / (pf_ms as f64 / 1e3)
        );
    }
}

#[test]
fn test_glm5_stage_b_prefill_small_moe_and_inplace_hc_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).expect("CUDA GPU 0 available");
    let mut rng = Rng(0x57A6EB01CAFEBABEu64);

    // 1. In-place hc_post_add (R_in == R_out) bit-exact parity for hc=4 and hc=6 (hc != 4 path)
    let m_hc = 17usize;
    let n_embd_hc = 512usize;
    for &hc in &[4usize, 6usize] {
        let r_in: Vec<f32> = (0..(m_hc * hc * n_embd_hc)).map(|_| rng.next_f32() * 0.4).collect();
        let post: Vec<f32> = (0..(m_hc * hc)).map(|_| 0.5 + 0.3 * rng.next_f32()).collect();
        let comb: Vec<f32> = (0..(m_hc * hc * hc)).map(|_| 0.25 + 0.1 * rng.next_f32()).collect();
        let a: Vec<f32> = (0..(m_hc * n_embd_hc)).map(|_| rng.next_f32() * 0.3).collect();
        let b: Vec<f32> = (0..(m_hc * n_embd_hc)).map(|_| rng.next_f32() * 0.2).collect();

        let d_rin = DeviceBuffer::alloc(0, r_in.len() * 4).unwrap();
        let d_rin_inplace = DeviceBuffer::alloc(0, r_in.len() * 4).unwrap();
        let d_rout = DeviceBuffer::alloc_zeroed(0, r_in.len() * 4).unwrap();
        let d_post = DeviceBuffer::alloc(0, post.len() * 4).unwrap();
        let d_comb = DeviceBuffer::alloc(0, comb.len() * 4).unwrap();
        let d_a = DeviceBuffer::alloc(0, a.len() * 4).unwrap();
        let d_b = DeviceBuffer::alloc(0, b.len() * 4).unwrap();

        d_rin.copy_from_host(bytemuck::cast_slice(&r_in)).unwrap();
        d_rin_inplace.copy_from_host(bytemuck::cast_slice(&r_in)).unwrap();
        d_post.copy_from_host(bytemuck::cast_slice(&post)).unwrap();
        d_comb.copy_from_host(bytemuck::cast_slice(&comb)).unwrap();
        d_a.copy_from_host(bytemuck::cast_slice(&a)).unwrap();
        d_b.copy_from_host(bytemuck::cast_slice(&b)).unwrap();

        unsafe {
            glm5::hc_post_add(
                d_rin.ptr.cast(),
                d_post.ptr.cast(),
                d_comb.ptr.cast(),
                d_a.ptr.cast(),
                d_b.ptr.cast(),
                std::ptr::null(),
                d_rout.ptr.cast(),
                n_embd_hc as i32,
                hc as i32,
                m_hc as i32,
                &stream,
            );
            glm5::hc_post_add(
                d_rin_inplace.ptr.cast(),
                d_post.ptr.cast(),
                d_comb.ptr.cast(),
                d_a.ptr.cast(),
                d_b.ptr.cast(),
                std::ptr::null(),
                d_rin_inplace.ptr.cast(),
                n_embd_hc as i32,
                hc as i32,
                m_hc as i32,
                &stream,
            );
        }
        stream.sync().unwrap();

        let mut got_oop = vec![0.0f32; r_in.len()];
        let mut got_inp = vec![0.0f32; r_in.len()];
        d_rout.copy_to_host(bytemuck::cast_slice_mut(&mut got_oop)).unwrap();
        d_rin_inplace.copy_to_host(bytemuck::cast_slice_mut(&mut got_inp)).unwrap();
        assert_eq!(got_oop, got_inp, "hc_post_add in-place hc={hc} must be bit-exact");
    }

    // 2. moe_prefill_small_gu + moe_prefill_small_down parity (nt=[1, 2, 5, 8], M=16, n_embd=4096, n_ff=1536)
    let m = 16usize;
    let n_embd = 4096usize;
    let n_ff = 1536usize;
    let clamp_lim = 7.0f32;
    let nt_list = [1i32, 2, 5, 8];
    let n_jobs = nt_list.len();

    let x_f32: Vec<f32> = (0..(m * n_embd)).map(|_| rng.next_f32() * 0.35).collect();
    let d_x = DeviceBuffer::alloc(0, x_f32.len() * 4).unwrap();
    d_x.copy_from_host(bytemuck::cast_slice(&x_f32)).unwrap();
    let x_q8_bytes = (n_embd / 32) * 36;
    let d_xq8 = DeviceBuffer::alloc_zeroed(0, m * x_q8_bytes).unwrap();
    unsafe {
        launch_quantize_q8_1(d_x.ptr.cast(), d_xq8.ptr, m as i32, n_embd as i32, stream.raw);
    }
    stream.sync().unwrap();

    // Build deterministic overlapping routing across the 4 experts (total_nt = 16)
    let mut slot_off_h = Vec::with_capacity(n_jobs);
    let mut rows_all_h = Vec::new();
    let mut wts_all_h = Vec::new();
    let mut cur_off = 0i32;
    for (j, &nt) in nt_list.iter().enumerate() {
        slot_off_h.push(cur_off);
        for ti in 0..nt {
            let tok = ((j * 5 + (ti as usize) * 3) % m) as i32;
            rows_all_h.push(tok);
            wts_all_h.push(0.2 + 0.15 * ((j + ti as usize) % 5) as f32);
        }
        cur_off += nt;
    }
    let total_nt = cur_off as usize;

    let d_slot_off = DeviceBuffer::alloc(0, slot_off_h.len() * 4).unwrap();
    let d_slot_nt = DeviceBuffer::alloc(0, nt_list.len() * 4).unwrap();
    let d_rows_all = DeviceBuffer::alloc(0, rows_all_h.len() * 4).unwrap();
    let d_wts_all = DeviceBuffer::alloc(0, wts_all_h.len() * 4).unwrap();
    d_slot_off.copy_from_host(bytemuck::cast_slice(&slot_off_h)).unwrap();
    d_slot_nt.copy_from_host(bytemuck::cast_slice(&nt_list)).unwrap();
    d_rows_all.copy_from_host(bytemuck::cast_slice(&rows_all_h)).unwrap();
    d_wts_all.copy_from_host(bytemuck::cast_slice(&wts_all_h)).unwrap();

    let inter_q8_bytes = (n_ff / 32) * 36;
    let d_inter_q8 = DeviceBuffer::alloc_zeroed(0, total_nt * inter_q8_bytes).unwrap();

    for &(gu_t, d_t) in &[
        (GgmlType::Iq2S, GgmlType::Iq3Xxs),
        (GgmlType::Q3K, GgmlType::Q4K),
        (GgmlType::Iq2S, GgmlType::Iq3S),
        (GgmlType::Q6K, GgmlType::Iq3S),
    ] {
        let gu_rb = gu_t.row_bytes(n_embd);
        let d_rb = d_t.row_bytes(n_ff);
        let gu_bytes = n_ff * gu_rb;
        let dn_bytes = n_embd * d_rb;
        let u_off = gu_bytes;
        let d_off = 2 * gu_bytes;
        let blob_bytes = 2 * gu_bytes + dn_bytes;

        let mut g_f32_exp = Vec::with_capacity(n_jobs);
        let mut u_f32_exp = Vec::with_capacity(n_jobs);
        let mut d_f32_exp = Vec::with_capacity(n_jobs);
        let mut exp_bufs = Vec::with_capacity(n_jobs);
        let mut exp_ptrs_h = Vec::with_capacity(n_jobs);

        for _ in 0..n_jobs {
            let mut blob = vec![0u8; blob_bytes];
            let mut gf = vec![0.0f32; n_ff * n_embd];
            let mut uf = vec![0.0f32; n_ff * n_embd];
            let mut df = vec![0.0f32; n_embd * n_ff];
            for r in 0..n_ff {
                let gr = fill_glm5_quant_row(gu_t, n_embd, &mut rng);
                let ur = fill_glm5_quant_row(gu_t, n_embd, &mut rng);
                dequantize_row_to_f32(gu_t, &gr, &mut gf[r * n_embd..(r + 1) * n_embd]);
                dequantize_row_to_f32(gu_t, &ur, &mut uf[r * n_embd..(r + 1) * n_embd]);
                blob[r * gu_rb..(r + 1) * gu_rb].copy_from_slice(&gr);
                blob[u_off + r * gu_rb..u_off + (r + 1) * gu_rb].copy_from_slice(&ur);
            }
            for r in 0..n_embd {
                let dr = fill_glm5_quant_row(d_t, n_ff, &mut rng);
                dequantize_row_to_f32(d_t, &dr, &mut df[r * n_ff..(r + 1) * n_ff]);
                blob[d_off + r * d_rb..d_off + (r + 1) * d_rb].copy_from_slice(&dr);
            }
            let db = DeviceBuffer::alloc(0, blob_bytes).unwrap();
            db.copy_from_host(&blob).unwrap();
            exp_ptrs_h.push(db.as_u64());
            exp_bufs.push(db);
            g_f32_exp.push(gf);
            u_f32_exp.push(uf);
            d_f32_exp.push(df);
        }

        let d_exp_ptrs = DeviceBuffer::alloc(0, n_jobs * 8).unwrap();
        d_exp_ptrs.copy_from_host(bytemuck::cast_slice(&exp_ptrs_h)).unwrap();
        let d_out = DeviceBuffer::alloc_zeroed(0, m * n_embd * 4).unwrap();

        unsafe {
            glm5::moe_prefill_small_gu(
                gu_t as i32,
                d_exp_ptrs.ptr.cast(),
                d_slot_off.ptr.cast(),
                d_slot_nt.ptr.cast(),
                d_rows_all.ptr.cast(),
                u_off as u32,
                gu_rb,
                d_xq8.ptr,
                d_inter_q8.ptr,
                n_embd as i32,
                n_ff as i32,
                n_jobs as i32,
                clamp_lim,
                &stream,
            );
            glm5::moe_prefill_small_down(
                d_t as i32,
                d_exp_ptrs.ptr.cast(),
                d_slot_off.ptr.cast(),
                d_slot_nt.ptr.cast(),
                d_rows_all.ptr.cast(),
                d_wts_all.ptr.cast(),
                d_off as u32,
                d_rb,
                d_inter_q8.ptr,
                d_out.ptr.cast(),
                n_embd as i32,
                n_ff as i32,
                n_jobs as i32,
                &stream,
            );
        }
        stream.sync().unwrap();

        let mut got = vec![0.0f32; m * n_embd];
        d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();

        let mut ref_out = vec![0.0f64; m * n_embd];
        for j in 0..n_jobs {
            let off = slot_off_h[j] as usize;
            let nt = nt_list[j] as usize;
            let gf = &g_f32_exp[j];
            let uf = &u_f32_exp[j];
            let df = &d_f32_exp[j];
            for ti in 0..nt {
                let tok = rows_all_h[off + ti] as usize;
                let rw = wts_all_h[off + ti] as f64;
                let xt = &x_f32[tok * n_embd..(tok + 1) * n_embd];
                let mut h = vec![0.0f32; n_ff];
                for r in 0..n_ff {
                    let mut sg = 0.0f64;
                    let mut su = 0.0f64;
                    let gr = &gf[r * n_embd..(r + 1) * n_embd];
                    let ur = &uf[r * n_embd..(r + 1) * n_embd];
                    for c in 0..n_embd {
                        sg += gr[c] as f64 * xt[c] as f64;
                        su += ur[c] as f64 * xt[c] as f64;
                    }
                    let gc = (sg as f32).min(clamp_lim);
                    let uc = (su as f32).clamp(-clamp_lim, clamp_lim);
                    h[r] = (gc / (1.0 + (-gc).exp())) * uc;
                }
                for r in 0..n_embd {
                    let dr = &df[r * n_ff..(r + 1) * n_ff];
                    let mut sd = 0.0f64;
                    for c in 0..n_ff {
                        sd += dr[c] as f64 * h[c] as f64;
                    }
                    ref_out[tok * n_embd + r] += rw * sd;
                }
            }
        }
        let ref_f32: Vec<f32> = ref_out.iter().map(|&v| v as f32).collect();
        let re = rel_err(&got, &ref_f32);
        println!(
            "Stage B prefill small MoE parity ({:?} + {:?}): rel_err={re:.6}",
            gu_t, d_t
        );
        if gu_t != GgmlType::Q6K {
            assert!(
                re < 2e-2,
                "moe_prefill_small ({:?}+{:?}) rel_err={re}",
                gu_t,
                d_t
            );
        }

        // Cross-check exact equivalence against moe_resident kernel (same Q8_1 arithmetic)
        let mut res_out = vec![0.0f64; m * n_embd];
        let d_gp = DeviceBuffer::alloc(0, 8).unwrap();
        let d_up = DeviceBuffer::alloc(0, 8).unwrap();
        let d_dp = DeviceBuffer::alloc(0, 8).unwrap();
        let d_res_ids = DeviceBuffer::alloc_zeroed(0, 4).unwrap();
        let d_res_rw = DeviceBuffer::alloc(0, 4).unwrap();
        let d_res_iq8 = DeviceBuffer::alloc_zeroed(0, inter_q8_bytes).unwrap();
        let d_res_out = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let mut tmp_row = vec![0.0f32; n_embd];
        for j in 0..n_jobs {
            let base = exp_ptrs_h[j];
            d_gp.copy_from_host(bytemuck::cast_slice(&[base])).unwrap();
            d_up.copy_from_host(bytemuck::cast_slice(&[base + u_off as u64])).unwrap();
            d_dp.copy_from_host(bytemuck::cast_slice(&[base + d_off as u64])).unwrap();
            let off = slot_off_h[j] as usize;
            let nt = nt_list[j] as usize;
            for ti in 0..nt {
                let tok = rows_all_h[off + ti] as usize;
                let rw = wts_all_h[off + ti];
                d_res_rw.copy_from_host(bytemuck::cast_slice(&[rw])).unwrap();
                let xq_ptr = unsafe { (d_xq8.ptr as *const u8).add(tok * x_q8_bytes).cast() };
                unsafe {
                    glm5::moe_resident(
                        gu_t as i32,
                        d_t as i32,
                        d_gp.ptr.cast(),
                        d_up.ptr.cast(),
                        d_dp.ptr.cast(),
                        gu_rb,
                        d_rb,
                        xq_ptr,
                        d_res_ids.ptr.cast(),
                        d_res_rw.ptr.cast(),
                        d_res_iq8.ptr,
                        std::ptr::null_mut(),
                        d_res_out.ptr.cast(),
                        n_embd as i32,
                        n_ff as i32,
                        1,
                        1,
                        clamp_lim,
                        &stream,
                    );
                }
                stream.sync().unwrap();
                d_res_out.copy_to_host(bytemuck::cast_slice_mut(&mut tmp_row)).unwrap();
                for r in 0..n_embd {
                    res_out[tok * n_embd + r] += tmp_row[r] as f64;
                }
            }
        }
        let res_f32: Vec<f32> = res_out.iter().map(|&v| v as f32).collect();
        let re_res = rel_err(&got, &res_f32);
        println!(
            "Stage B prefill small vs moe_resident ({:?} + {:?}): rel_err={re_res:.6}",
            gu_t, d_t
        );
        assert!(
            re_res < 1e-4,
            "moe_prefill_small vs moe_resident ({:?}+{:?}) rel_err={re_res}",
            gu_t,
            d_t
        );
    }
}

#[test]
fn test_glm5_stage_b2_flashmla_split_and_interleaved_swiglu_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use ampllm_cuda_kernels::{glm5, set_device, CudaStream, DeviceBuffer};
    use half::f16;

    set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = 0x51A9_B201u32;
    let next_f = |s: &mut u32| -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        ((*s >> 8) as f32 / (1u32 << 24) as f32) * 1.4 - 0.7
    };

    // 1. FlashMLA Split-KV parity (n_head = 64, kv_lora = 512, v_head = 256, qk_nope = 256)
    // Test both long context (n_sel = 2051, mixed VRAM+RAM cells + trailing -1) and short (n_sel = 7).
    let n_head = 64usize;
    let kv_lora = 512usize;
    let v_head = 256usize;
    let qk_nope = 256usize;
    let lat_stride = kv_lora + kv_lora / 16; // 544
    let res_cap = 1600usize;
    let total_cap = 2200usize;
    let mut h_lat = vec![0u8; total_cap * lat_stride];
    for c in 0..total_cap {
        let rec = &mut h_lat[c * lat_stride..(c + 1) * lat_stride];
        for e in 0..kv_lora {
            rec[e] = (((c * 37 + e * 13) & 127) as i32 - 64) as i8 as u8;
        }
        for sb in 0..(kv_lora / 32) {
            let d = f16::from_f32(0.02 + (((c + sb) % 11) as f32) * 0.005).to_le_bytes();
            rec[kv_lora + sb * 2..kv_lora + sb * 2 + 2].copy_from_slice(&d);
        }
    }
    let d_vram = DeviceBuffer::alloc(0, res_cap * lat_stride).unwrap();
    let d_ram = DeviceBuffer::alloc(0, total_cap * lat_stride).unwrap();
    d_vram.copy_from_host(&h_lat[..res_cap * lat_stride]).unwrap();
    d_ram.copy_from_host(&h_lat).unwrap();

    let mut q_abs = vec![0.0f32; n_head * kv_lora];
    for v in &mut q_abs {
        *v = next_f(&mut rng) * 0.35;
    }
    let mut wv_f16 = vec![0u16; n_head * v_head * kv_lora];
    for v in &mut wv_f16 {
        *v = f16::from_f32(next_f(&mut rng) * 0.15).to_bits();
    }
    let d_qabs = DeviceBuffer::alloc(0, q_abs.len() * 4).unwrap();
    d_qabs.copy_from_host(bytemuck::cast_slice(&q_abs)).unwrap();
    let d_wvb = DeviceBuffer::alloc(0, wv_f16.len() * 2).unwrap();
    d_wvb.copy_from_host(bytemuck::cast_slice(&wv_f16)).unwrap();
    let d_split_sc = DeviceBuffer::alloc_zeroed(0, n_head * 32 * 514 * 4).unwrap();
    let d_oq8 = DeviceBuffer::alloc_zeroed(0, ((n_head * v_head) / 32) * 36).unwrap();

    for &(n_sel_alloc, valid_n) in &[(2051usize, 2049usize), (32usize, 7usize), (8usize, 0usize)] {
        let mut cells = vec![-1i32; n_sel_alloc];
        for i in 0..valid_n {
            cells[i] = ((i * 7) % 2100) as i32;
        }
        let d_cells = DeviceBuffer::alloc(0, n_sel_alloc * 4).unwrap();
        d_cells.copy_from_host(bytemuck::cast_slice(&cells)).unwrap();

        let d_ctx_split = DeviceBuffer::alloc(0, n_head * kv_lora * 4).unwrap();
        let d_out_ref = DeviceBuffer::alloc_zeroed(0, n_head * v_head * 4).unwrap();
        let d_out_split = DeviceBuffer::alloc_zeroed(0, n_head * v_head * 4).unwrap();

        unsafe {
            glm5::dsa_mla_attn(
                d_qabs.ptr.cast(),
                d_vram.ptr,
                d_ram.ptr,
                res_cap as i32,
                d_cells.ptr.cast(),
                d_wvb.ptr,
                GgmlType::F16 as i32,
                kv_lora as i32,
                n_head as i32,
                v_head as i32,
                qk_nope as i32,
                n_sel_alloc as i32,
                d_out_ref.ptr.cast(),
                1,
                &stream,
            );
            glm5::dsa_mla_vproj_q8_m1(
                d_qabs.ptr.cast(),
                d_vram.ptr,
                d_ram.ptr,
                res_cap as i32,
                d_cells.ptr.cast(),
                d_wvb.ptr,
                GgmlType::F16 as i32,
                kv_lora as i32,
                n_head as i32,
                v_head as i32,
                qk_nope as i32,
                n_sel_alloc as i32,
                d_ctx_split.ptr.cast(),
                d_split_sc.ptr.cast(),
                d_out_split.ptr.cast(),
                d_oq8.ptr,
                std::ptr::null(),
                &stream,
            );
        }
        stream.sync().unwrap();

        let mut ref_v = vec![0.0f32; n_head * v_head];
        let mut split_v = vec![0.0f32; n_head * v_head];
        d_out_ref.copy_to_host(bytemuck::cast_slice_mut(&mut ref_v)).unwrap();
        d_out_split.copy_to_host(bytemuck::cast_slice_mut(&mut split_v)).unwrap();
        assert!(split_v.iter().all(|v| v.is_finite()), "NaN/Inf in split MLA for valid_n={valid_n}");
        let num: f64 = ref_v.iter().zip(split_v.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = ref_v.iter().map(|&a| (a as f64).powi(2)).sum();
        let re = (num / den.max(1e-12)).sqrt();
        println!("FlashMLA Split-KV parity (n_head=64, valid={valid_n}/{n_sel_alloc}): rel_err={re:.8}");
        assert!(re < 1e-4, "FlashMLA split rel_err={re} at valid_n={valid_n}");
    }

    // 2. Interleaved SwiGLU to F16 parity
    let nt = 13usize;
    let n_ff = 2048usize;
    let lim = 7.0f32;
    let mut h_gu = vec![0.0f32; nt * 2 * n_ff];
    let mut h_g = vec![0.0f32; nt * n_ff];
    let mut h_u = vec![0.0f32; nt * n_ff];
    for t in 0..nt {
        for c in 0..n_ff {
            let gv = next_f(&mut rng) * 12.0;
            let uv = next_f(&mut rng) * 12.0;
            h_g[t * n_ff + c] = gv;
            h_u[t * n_ff + c] = uv;
            h_gu[t * 2 * n_ff + c] = gv;
            h_gu[t * 2 * n_ff + n_ff + c] = uv;
        }
    }
    let d_gu = DeviceBuffer::alloc(0, h_gu.len() * 4).unwrap();
    let d_g = DeviceBuffer::alloc(0, h_g.len() * 4).unwrap();
    let d_u = DeviceBuffer::alloc(0, h_u.len() * 4).unwrap();
    let d_ref16 = DeviceBuffer::alloc(0, nt * n_ff * 2).unwrap();
    let d_got16 = DeviceBuffer::alloc(0, nt * n_ff * 2).unwrap();
    d_gu.copy_from_host(bytemuck::cast_slice(&h_gu)).unwrap();
    d_g.copy_from_host(bytemuck::cast_slice(&h_g)).unwrap();
    d_u.copy_from_host(bytemuck::cast_slice(&h_u)).unwrap();
    unsafe {
        glm5::swiglu_clamp_to_f16(d_g.ptr.cast(), d_u.ptr.cast(), lim, (nt * n_ff) as i32, d_ref16.ptr, &stream);
        glm5::swiglu_clamp_interleaved_to_f16(d_gu.ptr.cast(), lim, nt as i32, n_ff as i32, d_got16.ptr, &stream);
    }
    stream.sync().unwrap();
    let mut r16 = vec![0u16; nt * n_ff];
    let mut g16 = vec![0u16; nt * n_ff];
    d_ref16.copy_to_host(bytemuck::cast_slice_mut(&mut r16)).unwrap();
    d_got16.copy_to_host(bytemuck::cast_slice_mut(&mut g16)).unwrap();
    assert_eq!(r16, g16, "swiglu_clamp_interleaved_to_f16 mismatch");
}

#[test]
fn test_glm5_stage_c1_prefill_kernels_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use ampllm_cuda_kernels::{glm5, set_device, CudaStream, DeviceBuffer};
    use half::f16;

    set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = 0x51A9_C101u32;
    let next_f = |s: &mut u32| -> f32 {
        *s ^= *s << 13;
        *s ^= *s >> 17;
        *s ^= *s << 5;
        ((*s >> 8) as f32 / (1u32 << 24) as f32) * 1.4 - 0.7
    };

    // 1. Split KDA recurrence parity (T=128, n_head=64, head_dim=128)
    {
        let t_len = 128usize;
        let n_head = 64usize;
        let dk = 128usize;
        let tot = t_len * n_head * dk;
        let mut h_q = vec![0.0f32; tot];
        let mut h_k = vec![0.0f32; tot];
        let mut h_v = vec![0.0f32; tot];
        let mut h_g1 = vec![0.0f32; tot];
        let mut h_beta = vec![0.0f32; t_len * n_head];
        let mut h_s = vec![0.0f32; n_head * dk * dk];
        for v in &mut h_q { *v = next_f(&mut rng) * 0.25; }
        for v in &mut h_k { *v = next_f(&mut rng) * 0.25; }
        for v in &mut h_v { *v = next_f(&mut rng) * 0.5; }
        for v in &mut h_g1 { *v = -((next_f(&mut rng).abs()) * 0.4 + 0.05); }
        for v in &mut h_beta { *v = next_f(&mut rng).abs() * 0.8 + 0.05; }
        for v in &mut h_s { *v = next_f(&mut rng) * 0.1; }

        let mut ref_s = h_s.clone();
        let mut ref_out = vec![0.0f32; tot];
        let scale = 1.0f32 / (dk as f32).sqrt();
        for t in 0..t_len {
            for h in 0..n_head {
                let base = (t * n_head + h) * dk;
                let sg = &mut ref_s[h * dk * dk..(h + 1) * dk * dk];
                let b = h_beta[t * n_head + h];
                let dec: Vec<f32> = (0..dk).map(|i| h_g1[base + i].exp()).collect();
                for j in 0..dk {
                    let mut u = 0.0f32;
                    for i in 0..dk {
                        let sv = sg[i * dk + j] * dec[i];
                        sg[i * dk + j] = sv;
                        u += sv * h_k[base + i];
                    }
                    let dl = b * (h_v[base + j] - u);
                    let mut o = 0.0f32;
                    for i in 0..dk {
                        let sv = sg[i * dk + j] + h_k[base + i] * dl;
                        sg[i * dk + j] = sv;
                        o += sv * h_q[base + i];
                    }
                    ref_out[base + j] = o * scale;
                }
            }
        }

        let d_q = DeviceBuffer::alloc(0, tot * 4).unwrap();
        let d_k = DeviceBuffer::alloc(0, tot * 4).unwrap();
        let d_v = DeviceBuffer::alloc(0, tot * 4).unwrap();
        let d_g1 = DeviceBuffer::alloc(0, tot * 4).unwrap();
        let d_b = DeviceBuffer::alloc(0, t_len * n_head * 4).unwrap();
        let d_s = DeviceBuffer::alloc(0, h_s.len() * 4).unwrap();
        let d_o = DeviceBuffer::alloc_zeroed(0, tot * 4).unwrap();
        d_q.copy_from_host(bytemuck::cast_slice(&h_q)).unwrap();
        d_k.copy_from_host(bytemuck::cast_slice(&h_k)).unwrap();
        d_v.copy_from_host(bytemuck::cast_slice(&h_v)).unwrap();
        d_g1.copy_from_host(bytemuck::cast_slice(&h_g1)).unwrap();
        d_b.copy_from_host(bytemuck::cast_slice(&h_beta)).unwrap();
        d_s.copy_from_host(bytemuck::cast_slice(&h_s)).unwrap();
        unsafe {
            glm5::kda_recurrence(
                d_q.ptr.cast(), d_k.ptr.cast(), d_v.ptr.cast(),
                d_g1.ptr.cast(), d_b.ptr.cast(), d_s.ptr.cast(),
                n_head as i32, dk as i32, t_len as i32, d_o.ptr.cast(), &stream,
            );
        }
        stream.sync().unwrap();
        let mut got_o = vec![0.0f32; tot];
        let mut got_s = vec![0.0f32; h_s.len()];
        d_o.copy_to_host(bytemuck::cast_slice_mut(&mut got_o)).unwrap();
        d_s.copy_to_host(bytemuck::cast_slice_mut(&mut got_s)).unwrap();
        let num_o: f64 = ref_out.iter().zip(got_o.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let den_o: f64 = ref_out.iter().map(|&a| (a as f64).powi(2)).sum();
        let re_o = (num_o / den_o.max(1e-12)).sqrt();
        let num_s: f64 = ref_s.iter().zip(got_s.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let den_s: f64 = ref_s.iter().map(|&a| (a as f64).powi(2)).sum();
        let re_s = (num_s / den_s.max(1e-12)).sqrt();
        println!("Split KDA recurrence (T=128, n_head=64): rel_err_out={re_o:.9} rel_err_state={re_s:.9}");
        assert!(re_o < 1e-5 && re_s < 1e-5);
    }

    // 2. Grouped MMA Prefill MLA parity & speed (T=32, n_head=64, kv_lora=512, v_head=256, n_sel=2048)
    {
        let t_len = 32usize;
        let n_head = 64usize;
        let kv_lora = 512usize;
        let v_head = 256usize;
        let qk_nope = 256usize;
        let n_sel = 256usize;
        let lat_stride = kv_lora + kv_lora / 16;
        let res_cap = 200usize;
        let total_cap = 400usize;

        let mut h_lat = vec![0u8; total_cap * lat_stride];
        for c in 0..total_cap {
            let rec = &mut h_lat[c * lat_stride..(c + 1) * lat_stride];
            for e in 0..kv_lora {
                rec[e] = (((c * 37 + e * 13) & 127) as i32 - 64) as i8 as u8;
            }
            for sb in 0..(kv_lora / 32) {
                let d = f16::from_f32(0.02 + (((c + sb) % 11) as f32) * 0.004).to_le_bytes();
                rec[kv_lora + sb * 2..kv_lora + sb * 2 + 2].copy_from_slice(&d);
            }
        }
        let d_vram = DeviceBuffer::alloc(0, res_cap * lat_stride).unwrap();
        let d_ram = DeviceBuffer::alloc(0, total_cap * lat_stride).unwrap();
        d_vram.copy_from_host(&h_lat[..res_cap * lat_stride]).unwrap();
        d_ram.copy_from_host(&h_lat).unwrap();

        let mut q_abs = vec![0.0f32; t_len * n_head * kv_lora];
        for v in &mut q_abs { *v = next_f(&mut rng) * 0.35; }
        let mut wv_f16 = vec![0u16; n_head * v_head * kv_lora];
        for v in &mut wv_f16 { *v = f16::from_f32(next_f(&mut rng) * 0.12).to_bits(); }
        let mut cells = vec![-1i32; t_len * n_sel];
        for t in 0..t_len {
            let valid = if t == t_len - 1 { 0 } else { (17 + t * 7).min(n_sel) };
            for i in 0..valid {
                cells[t * n_sel + i] = ((i * 5 + t * 3) % (total_cap - 10)) as i32;
            }
        }
        let d_q = DeviceBuffer::alloc(0, q_abs.len() * 4).unwrap();
        d_q.copy_from_host(bytemuck::cast_slice(&q_abs)).unwrap();
        let d_wvb = DeviceBuffer::alloc(0, wv_f16.len() * 2).unwrap();
        d_wvb.copy_from_host(bytemuck::cast_slice(&wv_f16)).unwrap();
        let d_cells = DeviceBuffer::alloc(0, cells.len() * 4).unwrap();
        d_cells.copy_from_host(bytemuck::cast_slice(&cells)).unwrap();
        let d_ref = DeviceBuffer::alloc_zeroed(0, t_len * n_head * v_head * 4).unwrap();
        let d_mma = DeviceBuffer::alloc_zeroed(0, t_len * n_head * v_head * 4).unwrap();

        unsafe {
            glm5::dsa_mla_attn(
                d_q.ptr.cast(), d_vram.ptr, d_ram.ptr, res_cap as i32,
                d_cells.ptr.cast(), d_wvb.ptr, GgmlType::F16 as i32,
                kv_lora as i32, n_head as i32, v_head as i32, qk_nope as i32, n_sel as i32,
                d_ref.ptr.cast(), t_len as i32, &stream,
            );
            // In-place ctx_scratch == q_abs test
            glm5::dsa_mla_prefill_mma(
                d_q.ptr.cast(), d_vram.ptr, d_ram.ptr, res_cap as i32,
                d_cells.ptr.cast(), d_wvb.ptr, GgmlType::F16 as i32,
                kv_lora as i32, n_head as i32, v_head as i32, qk_nope as i32, n_sel as i32,
                d_q.ptr.cast(), d_mma.ptr.cast(), t_len as i32, &stream,
            );
        }
        stream.sync().unwrap();
        let mut h_ref = vec![0.0f32; t_len * n_head * v_head];
        let mut h_mma = vec![0.0f32; t_len * n_head * v_head];
        d_ref.copy_to_host(bytemuck::cast_slice_mut(&mut h_ref)).unwrap();
        d_mma.copy_to_host(bytemuck::cast_slice_mut(&mut h_mma)).unwrap();
        assert!(h_mma.iter().all(|v| v.is_finite()), "NaN/Inf in dsa_mla_prefill_mma");
        let num: f64 = h_ref.iter().zip(h_mma.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = h_ref.iter().map(|&a| (a as f64).powi(2)).sum();
        let re = (num / den.max(1e-12)).sqrt();
        println!("DSA Prefill MMA parity (T={t_len}, n_head={n_head}, n_sel={n_sel}, inplace): rel_err={re:.6}");
        assert!(re < 5e-3, "dsa_mla_prefill_mma rel_err={re}");

        // Quick microbench T=128, n_sel=2048
        let tb = 128usize;
        let sb = 2048usize;
        let d_qb = DeviceBuffer::alloc_zeroed(0, tb * n_head * kv_lora * 4).unwrap();
        let d_ctxb = DeviceBuffer::alloc_zeroed(0, tb * n_head * kv_lora * 4).unwrap();
        let d_cb = DeviceBuffer::alloc_zeroed(0, tb * sb * 4).unwrap();
        let d_ob = DeviceBuffer::alloc_zeroed(0, tb * n_head * v_head * 4).unwrap();
        for _ in 0..2 {
            unsafe {
                glm5::dsa_mla_prefill_mma(
                    d_qb.ptr.cast(), d_vram.ptr, d_ram.ptr, res_cap as i32,
                    d_cb.ptr.cast(), d_wvb.ptr, GgmlType::F16 as i32,
                    kv_lora as i32, n_head as i32, v_head as i32, qk_nope as i32, sb as i32,
                    d_ctxb.ptr.cast(), d_ob.ptr.cast(), tb as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            unsafe {
                glm5::dsa_mla_prefill_mma(
                    d_qb.ptr.cast(), d_vram.ptr, d_ram.ptr, res_cap as i32,
                    d_cb.ptr.cast(), d_wvb.ptr, GgmlType::F16 as i32,
                    kv_lora as i32, n_head as i32, v_head as i32, qk_nope as i32, sb as i32,
                    d_ctxb.ptr.cast(), d_ob.ptr.cast(), tb as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let mma_ms = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
        println!("MICROBENCH dsa_mla_prefill_mma (T=128, n_sel=2048): {mma_ms:.3} ms");
    }

    // 3. 3-stage XOR ldmatrix GEMM parity & TFLOPs microbench (32x64, 64x64, edge, 64x128, 128x128)
    for &(m, n, k) in &[
        (24usize, 2048usize, 4096usize),
        (32usize, 4096usize, 4096usize),
        (48usize, 256usize, 4096usize),
        (77usize, 333usize, 128usize),
        (160usize, 2048usize, 4096usize),
        (256usize, 4096usize, 2048usize),
        (256usize, 2048usize, 4096usize),
        (1024usize, 8192usize, 4096usize),
    ] {
        let mut ha = vec![0u16; m * k];
        let mut hb = vec![0u16; n * k];
        for v in &mut ha { *v = f16::from_f32(next_f(&mut rng) * 0.5).to_bits(); }
        for v in &mut hb { *v = f16::from_f32(next_f(&mut rng) * 0.5).to_bits(); }
        let da = DeviceBuffer::alloc(0, m * k * 2).unwrap();
        let db = DeviceBuffer::alloc(0, n * k * 2).unwrap();
        let dc = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
        da.copy_from_host(bytemuck::cast_slice(&ha)).unwrap();
        db.copy_from_host(bytemuck::cast_slice(&hb)).unwrap();
        unsafe {
            glm5::gemm_f16_mma_nt(da.ptr, db.ptr, dc.ptr.cast(), m as i32, n as i32, k as i32, 1.0, 0.0, &stream);
        }
        stream.sync().unwrap();
        let mut hc = vec![0.0f32; m * n];
        dc.copy_to_host(bytemuck::cast_slice_mut(&mut hc)).unwrap();
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        let r_step = if m <= 96 { 1 } else { 23 };
        let c_step = if n <= 512 { 1 } else { 97 };
        for row in (0..m).step_by(r_step) {
            for col in (0..n).step_by(c_step) {
                let mut r_acc = 0.0f32;
                for kk in 0..k {
                    r_acc += f16::from_bits(ha[row * k + kk]).to_f32() * f16::from_bits(hb[col * k + kk]).to_f32();
                }
                let got = hc[row * n + col];
                num += ((r_acc - got) as f64).powi(2);
                den += (r_acc as f64).powi(2);
            }
        }
        let re = (num / den.max(1e-12)).sqrt();
        assert!(re < 2e-3, "gemm_f16_mma_nt ({m}x{n}x{k}) rel_err={re}");
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            unsafe {
                glm5::gemm_f16_mma_nt(da.ptr, db.ptr, dc.ptr.cast(), m as i32, n as i32, k as i32, 1.0, 0.0, &stream);
            }
        }
        stream.sync().unwrap();
        let s = t0.elapsed().as_secs_f64() / 20.0;
        let tflops = (2.0 * m as f64 * n as f64 * k as f64) / (s * 1e12);
        println!("MICROBENCH gemm_f16_mma_nt ({m}x{n}x{k}): rel_err={re:.7}, {:.3} ms ({tflops:.2} TFLOPs)", s * 1000.0);
    }
}

#[test]
fn test_glm5_stage_c2_kernels_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use ampllm_cuda_kernels::ffi::launch_quantize_q8_1;
    let stream = CudaStream::new(0).unwrap();
    let mut rng = 0x5EED_C201u64;
    let next_f = |r: &mut u64| -> f32 {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*r >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };

    // 1. dsa_q_absorb MMA parity (T=32) & microbench (T=128)
    {
        let t_len = 32usize;
        let n_head = 64usize;
        let qk_nope = 256usize;
        let kv_lora = 512usize;
        let mut h_q = vec![0.0f32; t_len * n_head * qk_nope];
        let mut h_wk = vec![0u16; n_head * kv_lora * qk_nope];
        for v in &mut h_q { *v = next_f(&mut rng) * 1.5; }
        for v in &mut h_wk { *v = f16::from_f32(next_f(&mut rng) * 0.25).to_bits(); }

        let d_q = DeviceBuffer::alloc(0, h_q.len() * 4).unwrap();
        let d_wk = DeviceBuffer::alloc(0, h_wk.len() * 2).unwrap();
        let d_qa = DeviceBuffer::alloc_zeroed(0, t_len * n_head * kv_lora * 4).unwrap();
        d_q.copy_from_host(bytemuck::cast_slice(&h_q)).unwrap();
        d_wk.copy_from_host(bytemuck::cast_slice(&h_wk)).unwrap();

        unsafe {
            glm5::dsa_q_absorb(
                d_wk.ptr, GgmlType::F16 as i32,
                d_q.ptr.cast(), d_qa.ptr.cast(),
                qk_nope as i32, kv_lora as i32, n_head as i32, t_len as i32, &stream,
            );
        }
        stream.sync().unwrap();
        let mut got = vec![0.0f32; t_len * n_head * kv_lora];
        d_qa.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();

        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for t in (0..t_len).step_by(5) {
            for h in (0..n_head).step_by(11) {
                for c in (0..kv_lora).step_by(37) {
                    let mut r = 0.0f32;
                    let q_sl = &h_q[((t * n_head + h) * qk_nope)..((t * n_head + h + 1) * qk_nope)];
                    let w_sl = &h_wk[((h * kv_lora + c) * qk_nope)..((h * kv_lora + c + 1) * qk_nope)];
                    for k in 0..qk_nope {
                        r += q_sl[k] * f16::from_bits(w_sl[k]).to_f32();
                    }
                    let g = got[(t * n_head + h) * kv_lora + c];
                    num += ((r - g) as f64).powi(2);
                    den += (r as f64).powi(2);
                }
            }
        }
        let re = (num / den.max(1e-12)).sqrt();
        println!("DSA q_absorb MMA parity (T={t_len}, n_head={n_head}): rel_err={re:.6}");
        assert!(re < 2e-3, "dsa_q_absorb MMA rel_err={re}");

        let tb = 128usize;
        let d_qb = DeviceBuffer::alloc_zeroed(0, tb * n_head * qk_nope * 4).unwrap();
        let d_qab = DeviceBuffer::alloc_zeroed(0, tb * n_head * kv_lora * 4).unwrap();
        for _ in 0..2 {
            unsafe {
                glm5::dsa_q_absorb(
                    d_wk.ptr, GgmlType::F16 as i32, d_qb.ptr.cast(), d_qab.ptr.cast(),
                    qk_nope as i32, kv_lora as i32, n_head as i32, tb as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            unsafe {
                glm5::dsa_q_absorb(
                    d_wk.ptr, GgmlType::F16 as i32, d_qb.ptr.cast(), d_qab.ptr.cast(),
                    qk_nope as i32, kv_lora as i32, n_head as i32, tb as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        println!("MICROBENCH dsa_q_absorb MMA (T={tb}, n_head={n_head}): {:.3} ms", t0.elapsed().as_secs_f64() * 100.0);
    }

    // 2. Filtered two-pass moe_plan_filtered vs unified moe_plan parity (n_pci=2 and n_pci=0)
    {
        let n_embd = 4096usize;
        let n_ff = 2048usize;
        let n_ff_sh = 2048usize;
        let topk = 8usize;
        let gu_rb = GgmlType::Iq2S.row_bytes(n_embd);
        let d_rb = GgmlType::Iq3Xxs.row_bytes(n_ff);
        let sh_d_rb = GgmlType::Q4K.row_bytes(n_ff_sh);
        let g_eb = n_ff * gu_rb;
        let d_eb = n_embd * d_rb;
        let u_off = (g_eb + 15) & !15;
        let d_off = (u_off + g_eb + 15) & !15;
        let blob_stride = (d_off + d_eb + 15) & !15;

        let n_blobs = 6usize;
        let mut h_blobs = vec![0u8; n_blobs * blob_stride];
        let d_f16 = f16::from_f32(0.05).to_bits().to_le_bytes();
        for b in 0..n_blobs {
            let bp = &mut h_blobs[b * blob_stride..(b + 1) * blob_stride];
            for off in (0..g_eb).step_by(82) {
                bp[off..off + 2].copy_from_slice(&d_f16);
                for k in 2..82 { bp[off + k] = (next_f(&mut rng).abs() * 255.0) as u8; }
            }
            for off in (u_off..u_off + g_eb).step_by(82) {
                bp[off..off + 2].copy_from_slice(&d_f16);
                for k in 2..82 { bp[off + k] = (next_f(&mut rng).abs() * 255.0) as u8; }
            }
            for off in (d_off..d_off + d_eb).step_by(98) {
                bp[off..off + 2].copy_from_slice(&d_f16);
                for k in 2..98 { bp[off + k] = (next_f(&mut rng).abs() * 255.0) as u8; }
            }
        }
        let d_blobs = DeviceBuffer::alloc(0, h_blobs.len()).unwrap();
        d_blobs.copy_from_host(&h_blobs).unwrap();

        let mut h_sh_w = vec![0u8; n_embd * sh_d_rb];
        for off in (0..h_sh_w.len()).step_by(144) {
            h_sh_w[off..off + 2].copy_from_slice(&d_f16);
            h_sh_w[off + 2..off + 4].copy_from_slice(&d_f16);
            for k in 4..144 { h_sh_w[off + k] = (next_f(&mut rng).abs() * 255.0) as u8; }
        }
        let d_sh_w = DeviceBuffer::alloc(0, h_sh_w.len()).unwrap();
        d_sh_w.copy_from_host(&h_sh_w).unwrap();

        let mut h_x = vec![0.0f32; n_embd];
        let mut h_sh = vec![0.0f32; n_ff_sh];
        for v in &mut h_x { *v = next_f(&mut rng); }
        for v in &mut h_sh { *v = next_f(&mut rng); }
        let d_x = DeviceBuffer::alloc(0, n_embd * 4).unwrap();
        let d_sh = DeviceBuffer::alloc(0, n_ff_sh * 4).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&h_x)).unwrap();
        d_sh.copy_from_host(bytemuck::cast_slice(&h_sh)).unwrap();

        let d_xq = DeviceBuffer::alloc(0, (n_embd / 32) * 36).unwrap();
        let d_shq = DeviceBuffer::alloc(0, (n_ff_sh / 32) * 36).unwrap();
        unsafe {
            launch_quantize_q8_1(d_x.ptr.cast(), d_xq.ptr, 1, n_embd as i32, stream.raw);
            launch_quantize_q8_1(d_sh.ptr.cast(), d_shq.ptr, 1, n_ff_sh as i32, stream.raw);
        }

        let h_rw: Vec<f32> = (0..topk).map(|i| 0.08 + 0.02 * i as f32).collect();
        let d_rw = DeviceBuffer::alloc(0, topk * 4).unwrap();
        d_rw.copy_from_host(bytemuck::cast_slice(&h_rw)).unwrap();

        let base = d_blobs.as_u64();
        let h_plan: Vec<u64> = vec![
            base,
            base + blob_stride as u64,
            0, // CPU miss
            base + (2 * blob_stride) as u64,
            base + (3 * blob_stride) as u64,
            0, // CPU miss
            base + (4 * blob_stride) as u64,
            base + (5 * blob_stride) as u64,
        ];
        let d_plan = DeviceBuffer::alloc(0, topk * 8).unwrap();
        d_plan.copy_from_host(bytemuck::cast_slice(&h_plan)).unwrap();

        let d_inter_ref = DeviceBuffer::alloc_zeroed(0, topk * (n_ff / 32) * 36).unwrap();
        let d_inter_fil = DeviceBuffer::alloc_zeroed(0, topk * (n_ff / 32) * 36).unwrap();
        let d_shout_ref = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let d_shout_fil = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let d_out_ref = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();
        let d_out_fil = DeviceBuffer::alloc_zeroed(0, n_embd * 4).unwrap();

        for (label, h_fs) in [
            ("n_pci=2", vec![0u64, 1, 0, 0, 2, 0, 0, 0]),
            ("n_pci=0", vec![0u64; 8]),
        ] {
            let d_fs = DeviceBuffer::alloc(0, topk * 8).unwrap();
            d_fs.copy_from_host(bytemuck::cast_slice(&h_fs)).unwrap();
            unsafe {
                glm5::moe_plan(
                    GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32, d_plan.ptr.cast(),
                    u_off as u32, d_off as u32, gu_rb, d_rb, d_xq.ptr, d_rw.ptr.cast(),
                    d_inter_ref.ptr, GgmlType::Q4K as i32, d_sh_w.ptr, sh_d_rb, d_shq.ptr,
                    n_ff_sh as i32, d_shout_ref.ptr.cast(), d_out_ref.ptr.cast(),
                    n_embd as i32, n_ff as i32, topk as i32, 1, 7.0, std::ptr::null(), &stream,
                );
                glm5::moe_plan_filtered(
                    GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32, d_plan.ptr.cast(),
                    u_off as u32, d_off as u32, gu_rb, d_rb, d_xq.ptr, d_rw.ptr.cast(),
                    d_inter_fil.ptr, GgmlType::Q4K as i32, d_sh_w.ptr, sh_d_rb, d_shq.ptr,
                    n_ff_sh as i32, d_shout_fil.ptr.cast(), d_out_fil.ptr.cast(),
                    n_embd as i32, n_ff as i32, topk as i32, 1, 7.0, std::ptr::null(),
                    d_fs.ptr.cast(), 1, &stream,
                );
                glm5::moe_plan_filtered(
                    GgmlType::Iq2S as i32, GgmlType::Iq3Xxs as i32, d_plan.ptr.cast(),
                    u_off as u32, d_off as u32, gu_rb, d_rb, d_xq.ptr, d_rw.ptr.cast(),
                    d_inter_fil.ptr, 0, std::ptr::null(), 0, std::ptr::null(), 0,
                    std::ptr::null_mut(), d_out_fil.ptr.cast(),
                    n_embd as i32, n_ff as i32, topk as i32, 1, 7.0, std::ptr::null(),
                    d_fs.ptr.cast(), 2, &stream,
                );
            }
            stream.sync().unwrap();
            let mut ref_v = vec![0.0f32; n_embd];
            let mut fil_v = vec![0.0f32; n_embd];
            d_out_ref.copy_to_host(bytemuck::cast_slice_mut(&mut ref_v)).unwrap();
            d_out_fil.copy_to_host(bytemuck::cast_slice_mut(&mut fil_v)).unwrap();
            let num: f64 = ref_v.iter().zip(fil_v.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
            let den: f64 = ref_v.iter().map(|&a| (a as f64).powi(2)).sum();
            let re = (num / den.max(1e-12)).sqrt();
            println!("moe_plan_filtered parity ({label}): rel_err={re:.9}");
            assert!(re < 1e-5, "moe_plan_filtered ({label}) rel_err={re}");
        }
    }

    // 3. Mapped host zero-DMA-queue copy_to_mapped_f32 & mapped argmax parity
    {
        use ampllm_cuda_kernels::{set_device, PinnedMappedBuffer};

        set_device(0).unwrap();
        let n = 16384usize;
        let h_src: Vec<f32> = (0..n).map(|i| ((i % 997) as f32) * 0.013 - 4.2).collect();
        let d_src = DeviceBuffer::alloc(0, n * 4).unwrap();
        d_src.copy_from_host(bytemuck::cast_slice(&h_src)).unwrap();
        let dst_pin = PinnedMappedBuffer::<f32>::alloc(n).unwrap();
        let dst_dev = dst_pin.dev_ptr_for_device(0).unwrap();
        unsafe {
            glm5::copy_to_mapped_f32(d_src.ptr.cast(), dst_dev.cast(), n as i32, &stream);
        }
        stream.sync().unwrap();
        assert_eq!(dst_pin.as_slice(), h_src.as_slice(), "copy_to_mapped_f32 mismatch");

        let n_rows = 4096usize;
        let mut h_logits = vec![-5.0f32; n_rows];
        h_logits[2719] = 9.75;
        let d_logits = DeviceBuffer::alloc(0, n_rows * 4).unwrap();
        d_logits.copy_from_host(bytemuck::cast_slice(&h_logits)).unwrap();
        let d_pv = DeviceBuffer::alloc(0, 128 * 4).unwrap();
        let d_pi = DeviceBuffer::alloc(0, 128 * 4).unwrap();
        let tok_pin = PinnedMappedBuffer::<i32>::alloc(1).unwrap();
        let tok_dev = tok_pin.dev_ptr_for_device(0).unwrap();
        unsafe {
            glm5::argmax(
                d_logits.ptr.cast(),
                n_rows as i32,
                n_rows as i32,
                std::ptr::null(),
                d_pv.ptr.cast(),
                d_pi.ptr.cast(),
                tok_dev.cast(),
                &stream,
            );
        }
        stream.sync().unwrap();
        assert_eq!(tok_pin.as_slice()[0], 2719, "mapped argmax mismatch");
    }
}

#[test]
fn test_glm5_stage_c5_mmq_mma_and_moe_plan_parity() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use ampllm_cuda_kernels::ffi::launch_quantize_q8_1;
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0xC5_5EED_A1B2_C3D4);

    // 1. Test gemm_mmq_mma across all 6 quant types and all 3 tile regimes vs launch_linear_q8_1
    let types = [
        (GgmlType::Iq2S, 24usize, 4096usize, 4096usize),
        (GgmlType::Iq3Xxs, 96usize, 4096usize, 2048usize),
        (GgmlType::Iq3S, 256usize, 4096usize, 2048usize),
        (GgmlType::Q6K, 256usize, 4096usize, 4096usize),
        (GgmlType::Q8_0, 96usize, 4096usize, 128usize),
        (GgmlType::Q4K, 24usize, 1024usize, 1536usize),
    ];
    for &(w_ty, m, n, k) in &types {
        let rb = w_ty.row_bytes(k);
        let mut h_w = Vec::with_capacity(n * rb);
        for _ in 0..n {
            h_w.extend_from_slice(&fill_glm5_quant_row(w_ty, k, &mut rng));
        }
        let h_x: Vec<f32> = (0..m * k).map(|_| rng.next_f32() * 0.5).collect();

        let d_w = DeviceBuffer::alloc(0, h_w.len()).unwrap();
        let d_x = DeviceBuffer::alloc(0, h_x.len() * 4).unwrap();
        let d_xq = DeviceBuffer::alloc(0, m * (k / 32) * 36).unwrap();
        let d_ref = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
        let d_got = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
        d_w.copy_from_host(&h_w).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&h_x)).unwrap();

        unsafe {
            launch_quantize_q8_1(d_x.ptr.cast(), d_xq.ptr, m as i32, k as i32, stream.raw);
            launch_linear_q8_1(
                w_ty as i32,
                d_w.ptr,
                d_xq.ptr,
                d_x.ptr.cast(),
                d_ref.ptr.cast(),
                n as i32,
                k as i32,
                m as i32,
                stream.raw,
            );
            glm5::gemm_mmq_mma(
                w_ty as i32,
                d_xq.ptr,
                d_w.ptr,
                rb,
                d_got.ptr.cast(),
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

        let mut h_ref = vec![0.0f32; m * n];
        let mut h_got = vec![0.0f32; m * n];
        d_ref.copy_to_host(bytemuck::cast_slice_mut(&mut h_ref)).unwrap();
        d_got.copy_to_host(bytemuck::cast_slice_mut(&mut h_got)).unwrap();
        let re = rel_err(&h_got, &h_ref);
        println!("MMQ MMA parity ({w_ty:?}, {m}x{n}x{k}): rel_err={re:.7}");
        assert!(re < 2e-4, "MMQ MMA parity failed ({w_ty:?}, {m}x{n}x{k}): rel_err={re}");
    }

    // 2. Test fused MoE GateUp (rows_in_opt + swiglu_clamp_interleaved_q8) + Down (rows_out_opt, wts_out_opt scatter)
    {
        let m_tot = 128usize;
        let nt = 73usize;
        let n_embd = 4096usize;
        let n_ff = 2048usize;
        let gu_rb = GgmlType::Iq2S.row_bytes(n_embd);
        let d_rb = GgmlType::Iq3Xxs.row_bytes(n_ff);
        let clamp_lim = 7.0f32;

        let mut h_gu = Vec::with_capacity(2 * n_ff * gu_rb);
        for _ in 0..2 * n_ff {
            h_gu.extend_from_slice(&fill_glm5_quant_row(GgmlType::Iq2S, n_embd, &mut rng));
        }
        let mut h_dn = Vec::with_capacity(n_embd * d_rb);
        for _ in 0..n_embd {
            h_dn.extend_from_slice(&fill_glm5_quant_row(GgmlType::Iq3Xxs, n_ff, &mut rng));
        }
        let h_x: Vec<f32> = (0..m_tot * n_embd).map(|_| rng.next_f32() * 0.4).collect();
        let h_rows: Vec<i32> = (0..nt).map(|i| ((i * 5 + 3) % m_tot) as i32).collect();
        // Ensure distinct tokens in rows_out_opt (gcd(5, 128) = 1 -> all 73 are distinct!)
        let h_wts: Vec<f32> = (0..nt).map(|i| 0.15 + (i % 7) as f32 * 0.05).collect();
        let h_base: Vec<f32> = (0..m_tot * n_embd).map(|_| rng.next_f32() * 0.25).collect();

        let d_gu = DeviceBuffer::alloc(0, h_gu.len()).unwrap();
        let d_dn = DeviceBuffer::alloc(0, h_dn.len()).unwrap();
        let d_x = DeviceBuffer::alloc(0, h_x.len() * 4).unwrap();
        let d_xq = DeviceBuffer::alloc(0, m_tot * (n_embd / 32) * 36).unwrap();
        let d_rows = DeviceBuffer::alloc(0, nt * 4).unwrap();
        let d_wts = DeviceBuffer::alloc(0, nt * 4).unwrap();
        let d_gu_out = DeviceBuffer::alloc_zeroed(0, nt * 2 * n_ff * 4).unwrap();
        let d_iq8 = DeviceBuffer::alloc_zeroed(0, nt * (n_ff / 32) * 36).unwrap();
        let d_out_got = DeviceBuffer::alloc(0, m_tot * n_embd * 4).unwrap();
        let d_out_ref = DeviceBuffer::alloc(0, m_tot * n_embd * 4).unwrap();
        let d_x_gathered = DeviceBuffer::alloc(0, nt * n_embd * 4).unwrap();
        let d_xq_gathered = DeviceBuffer::alloc(0, nt * (n_embd / 32) * 36).unwrap();
        let d_gu_ref = DeviceBuffer::alloc_zeroed(0, nt * 2 * n_ff * 4).unwrap();
        let d_iq8_ref = DeviceBuffer::alloc_zeroed(0, nt * (n_ff / 32) * 36).unwrap();
        let d_dn_ref = DeviceBuffer::alloc_zeroed(0, nt * n_embd * 4).unwrap();

        d_gu.copy_from_host(&h_gu).unwrap();
        d_dn.copy_from_host(&h_dn).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&h_x)).unwrap();
        d_rows.copy_from_host(bytemuck::cast_slice(&h_rows)).unwrap();
        d_wts.copy_from_host(bytemuck::cast_slice(&h_wts)).unwrap();
        d_out_got.copy_from_host(bytemuck::cast_slice(&h_base)).unwrap();
        d_out_ref.copy_from_host(bytemuck::cast_slice(&h_base)).unwrap();

        let mut h_x_g = vec![0.0f32; nt * n_embd];
        for (i, &r) in h_rows.iter().enumerate() {
            let ru = r as usize;
            h_x_g[i * n_embd..(i + 1) * n_embd].copy_from_slice(&h_x[ru * n_embd..(ru + 1) * n_embd]);
        }
        d_x_gathered.copy_from_host(bytemuck::cast_slice(&h_x_g)).unwrap();

        unsafe {
            // Reference path
            launch_quantize_q8_1(d_x_gathered.ptr.cast(), d_xq_gathered.ptr, nt as i32, n_embd as i32, stream.raw);
            launch_linear_q8_1(
                GgmlType::Iq2S as i32,
                d_gu.ptr,
                d_xq_gathered.ptr,
                d_x_gathered.ptr.cast(),
                d_gu_ref.ptr.cast(),
                (2 * n_ff) as i32,
                n_embd as i32,
                nt as i32,
                stream.raw,
            );
            glm5::swiglu_clamp_interleaved_q8(d_gu_ref.ptr.cast(), clamp_lim, nt as i32, n_ff as i32, d_iq8_ref.ptr, &stream);
            launch_linear_q8_1(
                GgmlType::Iq3Xxs as i32,
                d_dn.ptr,
                d_iq8_ref.ptr,
                std::ptr::null(),
                d_dn_ref.ptr.cast(),
                n_embd as i32,
                n_ff as i32,
                nt as i32,
                stream.raw,
            );
            glm5::scatter_weighted_add_f32(
                d_dn_ref.ptr.cast(),
                d_rows.ptr.cast(),
                d_wts.ptr.cast(),
                d_out_ref.ptr.cast(),
                nt as i32,
                n_embd as i32,
                &stream,
            );

            // Fused MMQ MoE path
            launch_quantize_q8_1(d_x.ptr.cast(), d_xq.ptr, m_tot as i32, n_embd as i32, stream.raw);
            glm5::gemm_mmq_mma(
                GgmlType::Iq2S as i32,
                d_xq.ptr,
                d_gu.ptr,
                gu_rb,
                d_gu_out.ptr.cast(),
                d_rows.ptr.cast(),
                std::ptr::null(),
                std::ptr::null(),
                nt as i32,
                (2 * n_ff) as i32,
                n_embd as i32,
                1.0,
                0.0,
                &stream,
            );
            glm5::swiglu_clamp_interleaved_q8(d_gu_out.ptr.cast(), clamp_lim, nt as i32, n_ff as i32, d_iq8.ptr, &stream);
            glm5::gemm_mmq_mma(
                GgmlType::Iq3Xxs as i32,
                d_iq8.ptr,
                d_dn.ptr,
                d_rb,
                d_out_got.ptr.cast(),
                std::ptr::null(),
                d_rows.ptr.cast(),
                d_wts.ptr.cast(),
                nt as i32,
                n_embd as i32,
                n_ff as i32,
                1.0,
                0.0,
                &stream,
            );
        }
        stream.sync().unwrap();

        let mut h_out_ref = vec![0.0f32; m_tot * n_embd];
        let mut h_out_got = vec![0.0f32; m_tot * n_embd];
        d_out_ref.copy_to_host(bytemuck::cast_slice_mut(&mut h_out_ref)).unwrap();
        d_out_got.copy_to_host(bytemuck::cast_slice_mut(&mut h_out_got)).unwrap();
        let re = rel_err(&h_out_got, &h_out_ref);
        println!("MMQ MMA fused MoE gather+swiglu_q8+scatter parity (nt={nt}): rel_err={re:.7}");
        assert!(re < 2e-4, "MMQ MMA fused MoE rel_err={re}");
    }

    // 3. Microbenchmark GLM-5.3 real MMQ shapes
    for &(w_ty, m, n, k) in &[
        (GgmlType::Iq2S, 128usize, 4096usize, 4096usize),
        (GgmlType::Iq3Xxs, 128usize, 4096usize, 2048usize),
        (GgmlType::Q6K, 1024usize, 8192usize, 4096usize),
        (GgmlType::Q6K, 512usize, 12288usize, 1536usize),
    ] {
        let rb = w_ty.row_bytes(k);
        let d_w = DeviceBuffer::alloc_zeroed(0, n * rb).unwrap();
        let d_xq = DeviceBuffer::alloc_zeroed(0, m * (k / 32) * 36).unwrap();
        let d_c = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
        for _ in 0..3 {
            unsafe {
                glm5::gemm_mmq_mma(
                    w_ty as i32, d_xq.ptr, d_w.ptr, rb, d_c.ptr.cast(),
                    std::ptr::null(), std::ptr::null(), std::ptr::null(),
                    m as i32, n as i32, k as i32, 1.0, 0.0, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..20 {
            unsafe {
                glm5::gemm_mmq_mma(
                    w_ty as i32, d_xq.ptr, d_w.ptr, rb, d_c.ptr.cast(),
                    std::ptr::null(), std::ptr::null(), std::ptr::null(),
                    m as i32, n as i32, k as i32, 1.0, 0.0, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
        let tops = (2.0 * m as f64 * n as f64 * k as f64) / ((ms * 1e-3) * 1e12);
        println!("MICROBENCH gemm_mmq_mma ({w_ty:?} {m}x{n}x{k}): {ms:.3} ms ({tops:.2} TOPS)");
    }
}

#[test]
fn test_glm5_stage_c6_kernels_parity_and_bench() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ampllm_cuda_kernels::set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x57A6EC60000001);

    // 1. glm5_dsa_score_tc_kernel parity vs scalar FP32 reference (T=64, n_pools=512)
    {
        let t_len = 64usize;
        let idx_heads = 32usize;
        let key_dim = 128usize;
        let kpool = 16usize;
        let n_pools = 512usize;
        let top_pools_cap = 32usize;
        let n_sel = top_pools_cap * kpool + kpool - 1;

        let h_iq: Vec<f32> = (0..t_len * idx_heads * key_dim)
            .map(|_| rng.next_f32() * 1.4)
            .collect();
        let h_w: Vec<f32> = (0..t_len * idx_heads)
            .map(|_| rng.next_f32() * 0.5 + 0.1)
            .collect();
        let h_pooled_f32: Vec<f32> = (0..n_pools * key_dim)
            .map(|_| rng.next_f32() * 1.2)
            .collect();
        let mut h_pooled_f16 = vec![0u16; n_pools * key_dim];
        let d_pf32 = DeviceBuffer::alloc(0, h_pooled_f32.len() * 4).unwrap();
        let d_pf16 = DeviceBuffer::alloc(0, h_pooled_f16.len() * 2).unwrap();
        d_pf32.copy_from_host(bytemuck::cast_slice(&h_pooled_f32)).unwrap();
        unsafe {
            glm5::f32_to_f16(d_pf32.ptr.cast(), d_pf16.ptr, h_pooled_f32.len() as i32, &stream);
        }
        stream.sync().unwrap();
        d_pf16.copy_to_host(bytemuck::cast_slice_mut(&mut h_pooled_f16)).unwrap();

        let h_pos: Vec<i32> = (0..t_len)
            .map(|t| (kpool * (8 * (t + 1) - 1) - 1) as i32)
            .collect();

        let d_iq = DeviceBuffer::alloc(0, h_iq.len() * 4).unwrap();
        let d_w = DeviceBuffer::alloc(0, h_w.len() * 4).unwrap();
        let d_pos = DeviceBuffer::alloc(0, h_pos.len() * 4).unwrap();
        let d_scores = DeviceBuffer::alloc_zeroed(0, t_len * n_pools * 4).unwrap();
        let d_cells = DeviceBuffer::alloc_zeroed(0, t_len * n_sel * 4).unwrap();
        d_iq.copy_from_host(bytemuck::cast_slice(&h_iq)).unwrap();
        d_w.copy_from_host(bytemuck::cast_slice(&h_w)).unwrap();
        d_pos.copy_from_host(bytemuck::cast_slice(&h_pos)).unwrap();

        unsafe {
            glm5::dsa_score_select(
                d_iq.ptr.cast(),
                d_pf16.ptr,
                d_w.ptr.cast(),
                key_dim as i32,
                idx_heads as i32,
                kpool as i32,
                top_pools_cap as i32,
                true,
                n_sel as i32,
                d_pos.ptr.cast(),
                d_scores.ptr.cast(),
                d_cells.ptr.cast(),
                t_len as i32,
                n_pools as i32,
                &stream,
            );
        }
        stream.sync().unwrap();

        let mut got_scores = vec![0.0f32; t_len * n_pools];
        d_scores.copy_to_host(bytemuck::cast_slice_mut(&mut got_scores)).unwrap();

        let f16_to_f32 = |h: u16| -> f32 {
            let s = ((h >> 15) as u32) << 31;
            let e = ((h >> 10) & 0x1F) as u32;
            let m = (h & 0x3FF) as u32;
            if e == 0 {
                if m == 0 {
                    f32::from_bits(s)
                } else {
                    let mut v = m as f32;
                    v *= 2.0f32.powi(-24);
                    if s != 0 { -v } else { v }
                }
            } else if e == 31 {
                f32::from_bits(s | 0x7F80_0000 | (m << 13))
            } else {
                f32::from_bits(s | ((e + 112) << 23) | (m << 13))
            }
        };

        let mut max_abs_f16_ref = 0.0f32;
        let mut max_abs_f32_ref = 0.0f32;
        let mut sum_sq_ref = 0.0f64;
        let mut sum_sq_diff_f32 = 0.0f64;
        for t in 0..t_len {
            let n_vis = ((h_pos[t] + 1) as usize) / kpool;
            for p in 0..n_vis.min(n_pools) {
                let mut ref_f32 = 0.0f32;
                let mut ref_f16 = 0.0f32;
                for h in 0..idx_heads {
                    let mut d32 = 0.0f32;
                    let mut d16 = 0.0f32;
                    for k in 0..key_dim {
                        let pf = f16_to_f32(h_pooled_f16[p * key_dim + k]);
                        let q32 = h_iq[(t * idx_heads + h) * key_dim + k];
                        let q16 = f16::from_f32(q32).to_f32();
                        d32 += q32 * pf;
                        d16 += q16 * pf;
                    }
                    let w = h_w[t * idx_heads + h];
                    if d32 > 0.0 {
                        ref_f32 += d32 * w;
                    }
                    if d16 > 0.0 {
                        ref_f16 += d16 * w;
                    }
                }
                let got = got_scores[t * n_pools + p];
                max_abs_f16_ref = max_abs_f16_ref.max((got - ref_f16).abs());
                max_abs_f32_ref = max_abs_f32_ref.max((got - ref_f32).abs());
                let df = (got - ref_f32) as f64;
                sum_sq_diff_f32 += df * df;
                sum_sq_ref += (ref_f32 as f64) * (ref_f32 as f64);
            }
        }
        let rel_f32 = (sum_sq_diff_f32 / sum_sq_ref.max(1e-12)).sqrt();
        println!(
            "DSA score TC parity (T={t_len}, n_pools={n_pools}): max_abs_vs_f16={max_abs_f16_ref:.6} max_abs_vs_f32={max_abs_f32_ref:.6} rel_vs_f32={rel_f32:.6}"
        );
        assert!(max_abs_f16_ref < 5e-4, "DSA score TC parity vs F16 failed: {max_abs_f16_ref}");
        assert!(rel_f32 < 1e-3, "DSA score TC rel parity vs F32 failed: {rel_f32}");

        // Bench T=512, n_pools=5000
        let bt = 512usize;
        let bp = 5000usize;
        let d_biq = DeviceBuffer::alloc_zeroed(0, bt * idx_heads * key_dim * 4).unwrap();
        let d_bp = DeviceBuffer::alloc_zeroed(0, bp * key_dim * 2).unwrap();
        let d_bw = DeviceBuffer::alloc_zeroed(0, bt * idx_heads * 4).unwrap();
        let bpos: Vec<i32> = (0..bt).map(|i| ((i + 1) * 16 - 1).min(bp * kpool - 1) as i32).collect();
        let d_bpos = DeviceBuffer::alloc(0, bt * 4).unwrap();
        d_bpos.copy_from_host(bytemuck::cast_slice(&bpos)).unwrap();
        let d_bsc = DeviceBuffer::alloc_zeroed(0, bt * bp * 4).unwrap();
        let d_bcell = DeviceBuffer::alloc_zeroed(0, bt * n_sel * 4).unwrap();
        for _ in 0..2 {
            unsafe {
                glm5::dsa_score_select(
                    d_biq.ptr.cast(), d_bp.ptr, d_bw.ptr.cast(), key_dim as i32, idx_heads as i32,
                    kpool as i32, top_pools_cap as i32, true, n_sel as i32, d_bpos.ptr.cast(),
                    d_bsc.ptr.cast(), d_bcell.ptr.cast(), bt as i32, bp as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            unsafe {
                glm5::dsa_score_select(
                    d_biq.ptr.cast(), d_bp.ptr, d_bw.ptr.cast(), key_dim as i32, idx_heads as i32,
                    kpool as i32, top_pools_cap as i32, true, n_sel as i32, d_bpos.ptr.cast(),
                    d_bsc.ptr.cast(), d_bcell.ptr.cast(), bt as i32, bp as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let dsa_ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
        println!("MICROBENCH dsa_score_select TC (T={bt}, n_pools={bp}): {dsa_ms:.3} ms");
    }

    // 2. glm5_kda_recurrence_split_kernel (__expf, KJ=16 & KJ=32) parity
    for &n_head in &[32usize, 64usize] {
        let t_len = 128usize;
        let hd = 128usize;
        let di = n_head * hd;
        let h_q: Vec<f32> = (0..t_len * di).map(|_| rng.next_f32() * 0.15).collect();
        let h_k: Vec<f32> = (0..t_len * di).map(|_| rng.next_f32() * 0.15).collect();
        let h_v: Vec<f32> = (0..t_len * di).map(|_| rng.next_f32() * 0.15).collect();
        let h_g1: Vec<f32> = (0..t_len * di).map(|_| -rng.next_f32().abs() * 0.5).collect();
        let h_beta: Vec<f32> = (0..t_len * n_head).map(|_| rng.next_f32().abs() * 0.3).collect();
        let mut h_state = vec![0.0f32; n_head * hd * hd];

        let d_q = DeviceBuffer::alloc(0, h_q.len() * 4).unwrap();
        let d_k = DeviceBuffer::alloc(0, h_k.len() * 4).unwrap();
        let d_v = DeviceBuffer::alloc(0, h_v.len() * 4).unwrap();
        let d_g1 = DeviceBuffer::alloc(0, h_g1.len() * 4).unwrap();
        let d_beta = DeviceBuffer::alloc(0, h_beta.len() * 4).unwrap();
        let d_state = DeviceBuffer::alloc_zeroed(0, h_state.len() * 4).unwrap();
        let d_out = DeviceBuffer::alloc_zeroed(0, t_len * di * 4).unwrap();
        d_q.copy_from_host(bytemuck::cast_slice(&h_q)).unwrap();
        d_k.copy_from_host(bytemuck::cast_slice(&h_k)).unwrap();
        d_v.copy_from_host(bytemuck::cast_slice(&h_v)).unwrap();
        d_g1.copy_from_host(bytemuck::cast_slice(&h_g1)).unwrap();
        d_beta.copy_from_host(bytemuck::cast_slice(&h_beta)).unwrap();

        unsafe {
            glm5::kda_recurrence_skip(
                d_q.ptr.cast(), d_k.ptr.cast(), d_v.ptr.cast(), d_g1.ptr.cast(),
                d_beta.ptr.cast(), d_state.ptr.cast(), n_head as i32, hd as i32,
                t_len as i32, d_out.ptr.cast(), std::ptr::null(), &stream,
            );
        }
        stream.sync().unwrap();

        let mut got_out = vec![0.0f32; t_len * di];
        d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_out)).unwrap();

        let scale = (hd as f32).sqrt().recip();
        let mut ref_out = vec![0.0f32; t_len * di];
        for h in 0..n_head {
            let s_h = &mut h_state[h * hd * hd..(h + 1) * hd * hd];
            for t in 0..t_len {
                let off = (t * n_head + h) * hd;
                let b = h_beta[t * n_head + h];
                for i in 0..hd {
                    let dec = h_g1[off + i].exp();
                    for j in 0..hd {
                        s_h[i * hd + j] *= dec;
                    }
                }
                for j in 0..hd {
                    let mut a = 0.0f32;
                    for i in 0..hd {
                        a += s_h[i * hd + j] * h_k[off + i];
                    }
                    let dl = b * (h_v[off + j] - a);
                    let mut o = 0.0f32;
                    for i in 0..hd {
                        s_h[i * hd + j] += h_k[off + i] * dl;
                        o += s_h[i * hd + j] * h_q[off + i];
                    }
                    ref_out[off + j] = o * scale;
                }
            }
        }
        let max_err = got_out.iter().zip(ref_out.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("KDA recurrence parity (n_head={n_head}, T={t_len}): max_abs_err={max_err:.7}");
        assert!(max_err < 1e-4, "KDA recurrence parity failed (n_head={n_head}): {max_err}");
    }

    // 3. glm5_moe_fetch_kernel parity + n_blocks bandwidth sweep (8.585 MB blob)
    {
        let blob_bytes = 9_002_000usize;
        let mut h_pinned = ampllm_cuda_kernels::RegisteredMappedHostBuffer::alloc(blob_bytes).unwrap();
        let mut expected = vec![0u8; blob_bytes];
        for (i, b) in expected.iter_mut().enumerate() {
            *b = ((i * 131 + 17) & 0xFF) as u8;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(expected.as_ptr(), h_pinned.as_mut_ptr(), blob_bytes);
        }
        let src_dev = h_pinned.dev_ptr_for_device(0).unwrap();
        let d_dst = DeviceBuffer::alloc_zeroed(0, blob_bytes).unwrap();
        let d_fsrc = DeviceBuffer::alloc(0, 8 * 8).unwrap();
        let d_plan = DeviceBuffer::alloc(0, 8 * 8).unwrap();
        let mut fsrc_h = [0u64; 8];
        let mut plan_h = [0u64; 8];
        fsrc_h[0] = src_dev;
        plan_h[0] = d_dst.as_u64();
        d_fsrc.copy_from_host(bytemuck::cast_slice(&fsrc_h)).unwrap();
        d_plan.copy_from_host(bytemuck::cast_slice(&plan_h)).unwrap();

        unsafe {
            glm5::moe_fetch(d_fsrc.ptr.cast(), d_plan.ptr.cast(), 8, blob_bytes, &stream);
        }
        stream.sync().unwrap();
        let mut got = vec![0u8; blob_bytes];
        d_dst.copy_to_host(&mut got).unwrap();
        assert_eq!(got, expected, "moe_fetch byte parity failed");

        for &n_blk in &[16i32, 32, 48, 64, 82] {
            for _ in 0..3 {
                unsafe {
                    glm5::moe_fetch_nblocks(d_fsrc.ptr.cast(), d_plan.ptr.cast(), 8, blob_bytes, n_blk, &stream);
                }
            }
            stream.sync().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..20 {
                unsafe {
                    glm5::moe_fetch_nblocks(d_fsrc.ptr.cast(), d_plan.ptr.cast(), 8, blob_bytes, n_blk, &stream);
                }
            }
            stream.sync().unwrap();
            let us = t0.elapsed().as_secs_f64() * 1e6 / 20.0;
            let gbps = (blob_bytes as f64) / (us * 1e-6) / 1e9;
            println!("MICROBENCH moe_fetch (8.585MB, n_blocks={n_blk}): {us:.1} us ({gbps:.2} GB/s)");
        }
    }

    // 4. MoE expert MMQ vs Hoisted FP16-MMA microbench (IQ2_S Gu + IQ3_XXS Down)
    {
        let n_embd = 4096usize;
        let n_ff = 2048usize;
        let gu_rb = GgmlType::Iq2S.row_bytes(n_embd);
        let d_rb = GgmlType::Iq3Xxs.row_bytes(n_ff);
        let max_nt = 512usize;
        let d_gu_q = DeviceBuffer::alloc_zeroed(0, 2 * n_ff * gu_rb).unwrap();
        let d_dn_q = DeviceBuffer::alloc_zeroed(0, n_embd * d_rb).unwrap();
        let d_gu_f16 = DeviceBuffer::alloc_zeroed(0, 2 * n_ff * n_embd * 2).unwrap();
        let d_dn_f16 = DeviceBuffer::alloc_zeroed(0, n_embd * n_ff * 2).unwrap();
        let d_xf32 = DeviceBuffer::alloc_zeroed(0, max_nt * n_embd * 4).unwrap();
        let d_xq8 = DeviceBuffer::alloc_zeroed(0, max_nt * (n_embd / 32) * 36).unwrap();
        let d_af16 = DeviceBuffer::alloc_zeroed(0, max_nt * n_embd * 2).unwrap();
        let d_gu_out = DeviceBuffer::alloc_zeroed(0, max_nt * 2 * n_ff * 4).unwrap();
        let d_iq8 = DeviceBuffer::alloc_zeroed(0, max_nt * (n_ff / 32) * 36).unwrap();
        let d_if16 = DeviceBuffer::alloc_zeroed(0, max_nt * n_ff * 2).unwrap();
        let d_dn_tmp = DeviceBuffer::alloc_zeroed(0, max_nt * n_embd * 4).unwrap();
        let d_out = DeviceBuffer::alloc_zeroed(0, max_nt * n_embd * 4).unwrap();
        let d_rows = DeviceBuffer::alloc_zeroed(0, max_nt * 4).unwrap();
        let d_wts = DeviceBuffer::alloc_zeroed(0, max_nt * 4).unwrap();

        for &nt in &[64usize, 128, 160, 256, 512] {
            for _ in 0..2 {
                unsafe {
                    glm5::gemm_mmq_mma(
                        GgmlType::Iq2S as i32, d_xq8.ptr, d_gu_q.ptr, gu_rb, d_gu_out.ptr.cast(),
                        d_rows.ptr.cast(), std::ptr::null(), std::ptr::null(),
                        nt as i32, (2 * n_ff) as i32, n_embd as i32, 1.0, 0.0, &stream,
                    );
                    glm5::swiglu_clamp_interleaved_q8(d_gu_out.ptr.cast(), 7.0, nt as i32, n_ff as i32, d_iq8.ptr, &stream);
                    glm5::gemm_mmq_mma(
                        GgmlType::Iq3Xxs as i32, d_iq8.ptr, d_dn_q.ptr, d_rb, d_out.ptr.cast(),
                        std::ptr::null(), d_rows.ptr.cast(), d_wts.ptr.cast(),
                        nt as i32, n_embd as i32, n_ff as i32, 1.0, 0.0, &stream,
                    );
                }
            }
            stream.sync().unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..10 {
                unsafe {
                    glm5::gemm_mmq_mma(
                        GgmlType::Iq2S as i32, d_xq8.ptr, d_gu_q.ptr, gu_rb, d_gu_out.ptr.cast(),
                        d_rows.ptr.cast(), std::ptr::null(), std::ptr::null(),
                        nt as i32, (2 * n_ff) as i32, n_embd as i32, 1.0, 0.0, &stream,
                    );
                    glm5::swiglu_clamp_interleaved_q8(d_gu_out.ptr.cast(), 7.0, nt as i32, n_ff as i32, d_iq8.ptr, &stream);
                    glm5::gemm_mmq_mma(
                        GgmlType::Iq3Xxs as i32, d_iq8.ptr, d_dn_q.ptr, d_rb, d_out.ptr.cast(),
                        std::ptr::null(), d_rows.ptr.cast(), d_wts.ptr.cast(),
                        nt as i32, n_embd as i32, n_ff as i32, 1.0, 0.0, &stream,
                    );
                }
            }
            stream.sync().unwrap();
            let mmq_us = t0.elapsed().as_secs_f64() * 1e6 / 10.0;

            for _ in 0..2 {
                unsafe {
                    glm5::dequant_to_f16(GgmlType::Iq2S as i32, d_gu_q.ptr, d_gu_f16.ptr, (2 * n_ff * n_embd) as i64, &stream);
                    glm5::dequant_to_f16(GgmlType::Iq3Xxs as i32, d_dn_q.ptr, d_dn_f16.ptr, (n_embd * n_ff) as i64, &stream);
                    glm5::gather_rows_f16(d_xf32.ptr.cast(), d_rows.ptr.cast(), d_af16.ptr, nt as i32, n_embd as i32, &stream);
                    glm5::gemm_f16_mma_nt(d_af16.ptr, d_gu_f16.ptr, d_gu_out.ptr.cast(), nt as i32, (2 * n_ff) as i32, n_embd as i32, 1.0, 0.0, &stream);
                    glm5::swiglu_clamp_interleaved_to_f16(d_gu_out.ptr.cast(), 7.0, nt as i32, n_ff as i32, d_if16.ptr, &stream);
                    glm5::gemm_f16_mma_nt(d_if16.ptr, d_dn_f16.ptr, d_dn_tmp.ptr.cast(), nt as i32, n_embd as i32, n_ff as i32, 1.0, 0.0, &stream);
                    glm5::scatter_weighted_add_f32(d_dn_tmp.ptr.cast(), d_rows.ptr.cast(), d_wts.ptr.cast(), d_out.ptr.cast(), nt as i32, n_embd as i32, &stream);
                }
            }
            stream.sync().unwrap();
            let t1 = std::time::Instant::now();
            for _ in 0..10 {
                unsafe {
                    glm5::dequant_to_f16(GgmlType::Iq2S as i32, d_gu_q.ptr, d_gu_f16.ptr, (2 * n_ff * n_embd) as i64, &stream);
                    glm5::dequant_to_f16(GgmlType::Iq3Xxs as i32, d_dn_q.ptr, d_dn_f16.ptr, (n_embd * n_ff) as i64, &stream);
                    glm5::gather_rows_f16(d_xf32.ptr.cast(), d_rows.ptr.cast(), d_af16.ptr, nt as i32, n_embd as i32, &stream);
                    glm5::gemm_f16_mma_nt(d_af16.ptr, d_gu_f16.ptr, d_gu_out.ptr.cast(), nt as i32, (2 * n_ff) as i32, n_embd as i32, 1.0, 0.0, &stream);
                    glm5::swiglu_clamp_interleaved_to_f16(d_gu_out.ptr.cast(), 7.0, nt as i32, n_ff as i32, d_if16.ptr, &stream);
                    glm5::gemm_f16_mma_nt(d_if16.ptr, d_dn_f16.ptr, d_dn_tmp.ptr.cast(), nt as i32, n_embd as i32, n_ff as i32, 1.0, 0.0, &stream);
                    glm5::scatter_weighted_add_f32(d_dn_tmp.ptr.cast(), d_rows.ptr.cast(), d_wts.ptr.cast(), d_out.ptr.cast(), nt as i32, n_embd as i32, &stream);
                }
            }
            stream.sync().unwrap();
            let f16_us = t1.elapsed().as_secs_f64() * 1e6 / 10.0;
            println!("MICROBENCH MoE expert nt={nt}: mmq={mmq_us:.1} us | hoisted_f16={f16_us:.1} us");
        }
    }
}

#[test]
fn test_glm5_stage_c7_kernels_parity_and_bench() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use ampllm_cuda_kernels::{
        ffi::{launch_linear_q8_1, launch_quantize_q8_1},
        glm5, set_device, CudaStream, DeviceBuffer,
    };
    set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x5EED_C701u64);

    // 1. Upgraded 32x32 dsa_mla_prefill_mma parity (T=70, dual-tier + -1 invalid cells) + microbench (T=512, n_sel=2048)
    {
        let t_len = 70usize;
        let n_head = 64usize;
        let kv_lora = 512usize;
        let v_head = 256usize;
        let qk_nope = 256usize;
        let n_sel = 512usize;
        let res_cap = 350usize;
        let total_cap = 600usize;
        let lat_stride = kv_lora + kv_lora / 16;

        let mut h_lat = vec![0u8; total_cap * lat_stride];
        for c in 0..total_cap {
            let rec = &mut h_lat[c * lat_stride..(c + 1) * lat_stride];
            for sb in 0..(kv_lora / 32) {
                let mut vals = [0.0f32; 32];
                let mut max_a = 0.0f32;
                for v in &mut vals {
                    *v = rng.next_f32() * 0.5;
                    max_a = max_a.max(v.abs());
                }
                let scale = (max_a / 127.0).max(1e-6);
                rec[kv_lora + sb * 2..kv_lora + sb * 2 + 2]
                    .copy_from_slice(&f16::from_f32(scale).to_le_bytes());
                let inv = 1.0 / scale;
                for (j, &v) in vals.iter().enumerate() {
                    rec[sb * 32 + j] = (v * inv).round().clamp(-127.0, 127.0) as i8 as u8;
                }
            }
        }
        let d_vram = DeviceBuffer::alloc(0, res_cap * lat_stride).unwrap();
        let d_ram = DeviceBuffer::alloc(0, total_cap * lat_stride).unwrap();
        d_vram.copy_from_host(&h_lat[..res_cap * lat_stride]).unwrap();
        d_ram.copy_from_host(&h_lat).unwrap();

        let mut q_abs = vec![0.0f32; t_len * n_head * kv_lora];
        for v in &mut q_abs {
            *v = rng.next_f32() * 0.7;
        }
        let mut wv_f16 = vec![0u16; n_head * v_head * kv_lora];
        for v in &mut wv_f16 {
            *v = f16::from_f32(rng.next_f32() * 0.25).to_bits();
        }
        let mut cells = vec![-1i32; t_len * n_sel];
        for t in 0..t_len {
            let valid = if t == t_len - 1 {
                0
            } else if t == 31 || t == 32 || t == 64 {
                n_sel
            } else {
                (19 + t * 9).min(n_sel)
            };
            for s in 0..valid {
                cells[t * n_sel + s] = ((t * 13 + s * 7) % total_cap) as i32;
            }
        }

        let d_q = DeviceBuffer::alloc(0, q_abs.len() * 4).unwrap();
        let d_ctx = DeviceBuffer::alloc_zeroed(0, q_abs.len() * 4).unwrap();
        let d_wvb = DeviceBuffer::alloc(0, wv_f16.len() * 2).unwrap();
        let d_cells = DeviceBuffer::alloc(0, cells.len() * 4).unwrap();
        let d_ref = DeviceBuffer::alloc_zeroed(0, t_len * n_head * v_head * 4).unwrap();
        let d_mma = DeviceBuffer::alloc_zeroed(0, t_len * n_head * v_head * 4).unwrap();
        d_q.copy_from_host(bytemuck::cast_slice(&q_abs)).unwrap();
        d_wvb.copy_from_host(bytemuck::cast_slice(&wv_f16)).unwrap();
        d_cells.copy_from_host(bytemuck::cast_slice(&cells)).unwrap();

        unsafe {
            glm5::dsa_mla_attn(
                d_q.ptr.cast(), d_vram.ptr, d_ram.ptr, res_cap as i32,
                d_cells.ptr.cast(), d_wvb.ptr, GgmlType::F16 as i32,
                kv_lora as i32, n_head as i32, v_head as i32, qk_nope as i32, n_sel as i32,
                d_ref.ptr.cast(), t_len as i32, &stream,
            );
            glm5::dsa_mla_prefill_mma(
                d_q.ptr.cast(), d_vram.ptr, d_ram.ptr, res_cap as i32,
                d_cells.ptr.cast(), d_wvb.ptr, GgmlType::F16 as i32,
                kv_lora as i32, n_head as i32, v_head as i32, qk_nope as i32, n_sel as i32,
                d_ctx.ptr.cast(), d_mma.ptr.cast(), t_len as i32, &stream,
            );
        }
        stream.sync().unwrap();

        let mut h_ref = vec![0.0f32; t_len * n_head * v_head];
        let mut h_mma = vec![0.0f32; t_len * n_head * v_head];
        d_ref.copy_to_host(bytemuck::cast_slice_mut(&mut h_ref)).unwrap();
        d_mma.copy_to_host(bytemuck::cast_slice_mut(&mut h_mma)).unwrap();
        assert!(h_mma.iter().all(|v| v.is_finite()), "NaN/Inf in Stage C.7 dsa_mla_prefill_mma");
        let re = rel_err(&h_mma, &h_ref);
        println!("Stage C.7 dsa_mla_prefill_mma (T={t_len}, n_sel={n_sel}): rel_err={re:.6}");
        assert!(re < 5e-3, "Stage C.7 dsa_mla_prefill_mma rel_err={re}");

        // Microbench T=512, n_sel=2048
        let tb = 512usize;
        let sb = 2048usize;
        let mut h_qb = vec![0.0f32; tb * n_head * kv_lora];
        for v in &mut h_qb {
            *v = rng.next_f32() * 0.5;
        }
        let mut h_cb = vec![0i32; tb * sb];
        for t in 0..tb {
            for s in 0..sb {
                h_cb[t * sb + s] = ((t * 11 + s) % res_cap) as i32;
            }
        }
        let d_qb = DeviceBuffer::alloc(0, h_qb.len() * 4).unwrap();
        let d_cb = DeviceBuffer::alloc(0, h_cb.len() * 4).unwrap();
        let d_ctxb = DeviceBuffer::alloc_zeroed(0, h_qb.len() * 4).unwrap();
        let d_ob = DeviceBuffer::alloc_zeroed(0, tb * n_head * v_head * 4).unwrap();
        d_qb.copy_from_host(bytemuck::cast_slice(&h_qb)).unwrap();
        d_cb.copy_from_host(bytemuck::cast_slice(&h_cb)).unwrap();
        for _ in 0..2 {
            unsafe {
                glm5::dsa_mla_prefill_mma(
                    d_qb.ptr.cast(), d_vram.ptr, d_ram.ptr, res_cap as i32,
                    d_cb.ptr.cast(), d_wvb.ptr, GgmlType::F16 as i32,
                    kv_lora as i32, n_head as i32, v_head as i32, qk_nope as i32, sb as i32,
                    d_ctxb.ptr.cast(), d_ob.ptr.cast(), tb as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..5 {
            unsafe {
                glm5::dsa_mla_prefill_mma(
                    d_qb.ptr.cast(), d_vram.ptr, d_ram.ptr, res_cap as i32,
                    d_cb.ptr.cast(), d_wvb.ptr, GgmlType::F16 as i32,
                    kv_lora as i32, n_head as i32, v_head as i32, qk_nope as i32, sb as i32,
                    d_ctxb.ptr.cast(), d_ob.ptr.cast(), tb as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let mma_ms = t0.elapsed().as_secs_f64() * 1000.0 / 5.0;
        println!("MICROBENCH Stage C.7 dsa_mla_prefill_mma (T={tb}, n_sel={sb}): {mma_ms:.3} ms");
    }

    // 2. Upgraded BM=64 dsa_q_absorb parity (T=70, n_head=64) & microbench (T=512)
    {
        let t_len = 70usize;
        let n_head = 64usize;
        let qk_nope = 256usize;
        let kv_lora = 512usize;
        let mut h_q = vec![0.0f32; t_len * n_head * qk_nope];
        let mut h_wk = vec![0u16; n_head * kv_lora * qk_nope];
        for v in &mut h_q {
            *v = rng.next_f32() * 0.75;
        }
        for v in &mut h_wk {
            *v = f16::from_f32(rng.next_f32() * 0.25).to_bits();
        }
        let d_q = DeviceBuffer::alloc(0, h_q.len() * 4).unwrap();
        let d_wk = DeviceBuffer::alloc(0, h_wk.len() * 2).unwrap();
        let d_qa = DeviceBuffer::alloc_zeroed(0, t_len * n_head * kv_lora * 4).unwrap();
        d_q.copy_from_host(bytemuck::cast_slice(&h_q)).unwrap();
        d_wk.copy_from_host(bytemuck::cast_slice(&h_wk)).unwrap();

        unsafe {
            glm5::dsa_q_absorb(
                d_wk.ptr, GgmlType::F16 as i32, d_q.ptr.cast(), d_qa.ptr.cast(),
                qk_nope as i32, kv_lora as i32, n_head as i32, t_len as i32, &stream,
            );
        }
        stream.sync().unwrap();
        let mut h_got = vec![0.0f32; t_len * n_head * kv_lora];
        d_qa.copy_to_host(bytemuck::cast_slice_mut(&mut h_got)).unwrap();

        let mut h_ref = vec![0.0f32; t_len * n_head * kv_lora];
        for t in 0..t_len {
            for h in 0..n_head {
                let q_sl = &h_q[(t * n_head + h) * qk_nope..(t * n_head + h + 1) * qk_nope];
                for c in 0..kv_lora {
                    let w_sl = &h_wk[((h * kv_lora + c) * qk_nope)..((h * kv_lora + c + 1) * qk_nope)];
                    let mut acc = 0.0f32;
                    for d in 0..qk_nope {
                        acc += q_sl[d] * f16::from_bits(w_sl[d]).to_f32();
                    }
                    h_ref[(t * n_head + h) * kv_lora + c] = acc;
                }
            }
        }
        let re = rel_err(&h_got, &h_ref);
        println!("Stage C.7 dsa_q_absorb MMA parity (T={t_len}, n_head={n_head}): rel_err={re:.6}");
        assert!(re < 2e-3, "Stage C.7 dsa_q_absorb MMA rel_err={re}");

        let tb = 512usize;
        let d_qb = DeviceBuffer::alloc_zeroed(0, tb * n_head * qk_nope * 4).unwrap();
        let d_qab = DeviceBuffer::alloc_zeroed(0, tb * n_head * kv_lora * 4).unwrap();
        for _ in 0..2 {
            unsafe {
                glm5::dsa_q_absorb(
                    d_wk.ptr, GgmlType::F16 as i32, d_qb.ptr.cast(), d_qab.ptr.cast(),
                    qk_nope as i32, kv_lora as i32, n_head as i32, tb as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            unsafe {
                glm5::dsa_q_absorb(
                    d_wk.ptr, GgmlType::F16 as i32, d_qb.ptr.cast(), d_qab.ptr.cast(),
                    qk_nope as i32, kv_lora as i32, n_head as i32, tb as i32, &stream,
                );
            }
        }
        stream.sync().unwrap();
        let qa_ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
        println!("MICROBENCH Stage C.7 dsa_q_absorb (T={tb}, n_head={n_head}): {qa_ms:.3} ms");
    }

    // 3. gemm_mmq_mma_tiled parity & tile microbench: force_tile=1 (128x64) vs force_tile=2 (64x64)
    {
        let shapes = [
            (GgmlType::Q6K, 1024usize, 8192usize, 4096usize),
            (GgmlType::Q6K, 512usize, 8192usize, 4096usize),
            (GgmlType::Iq2S, 256usize, 4096usize, 4096usize),
            (GgmlType::Iq2S, 160usize, 4096usize, 4096usize),
            (GgmlType::Iq3S, 256usize, 4096usize, 2048usize),
        ];
        for &(w_ty, m, n, k) in &shapes {
            let rb = w_ty.row_bytes(k);
            let mut h_x = vec![0.0f32; m * k];
            for v in &mut h_x {
                *v = rng.next_f32() * 0.5;
            }
            let mut h_w = Vec::with_capacity(n * rb);
            for _ in 0..n {
                h_w.extend_from_slice(&fill_glm5_quant_row(w_ty, k, &mut rng));
            }
            let d_x = DeviceBuffer::alloc(0, m * k * 4).unwrap();
            let d_xq = DeviceBuffer::alloc_zeroed(0, m * (k / 32) * 36).unwrap();
            let d_w = DeviceBuffer::alloc(0, n * rb).unwrap();
            let d_ref = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
            let d_out1 = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
            let d_out2 = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
            d_x.copy_from_host(bytemuck::cast_slice(&h_x)).unwrap();
            d_w.copy_from_host(&h_w).unwrap();

            unsafe {
                launch_quantize_q8_1(d_x.ptr.cast(), d_xq.ptr, m as i32, k as i32, stream.raw);
                launch_linear_q8_1(
                    w_ty as i32, d_w.ptr, d_xq.ptr, d_x.ptr.cast(), d_ref.ptr.cast(),
                    n as i32, k as i32, m as i32, stream.raw,
                );
                glm5::gemm_mmq_mma_tiled(
                    w_ty as i32, d_xq.ptr, d_w.ptr, rb, d_out1.ptr.cast(),
                    std::ptr::null(), std::ptr::null(), std::ptr::null(),
                    m as i32, n as i32, k as i32, 1.0, 0.0, 1, &stream,
                );
                glm5::gemm_mmq_mma_tiled(
                    w_ty as i32, d_xq.ptr, d_w.ptr, rb, d_out2.ptr.cast(),
                    std::ptr::null(), std::ptr::null(), std::ptr::null(),
                    m as i32, n as i32, k as i32, 1.0, 0.0, 2, &stream,
                );
            }
            stream.sync().unwrap();

            let mut h_ref = vec![0.0f32; m * n];
            let mut h_out1 = vec![0.0f32; m * n];
            let mut h_out2 = vec![0.0f32; m * n];
            d_ref.copy_to_host(bytemuck::cast_slice_mut(&mut h_ref)).unwrap();
            d_out1.copy_to_host(bytemuck::cast_slice_mut(&mut h_out1)).unwrap();
            d_out2.copy_to_host(bytemuck::cast_slice_mut(&mut h_out2)).unwrap();
            let re1 = rel_err(&h_out1, &h_ref);
            let re2 = rel_err(&h_out2, &h_ref);
            assert!(re1 < 5e-4, "MMQ tile=1 parity ({w_ty:?}, {m}x{n}x{k}) rel_err={re1}");
            assert!(re2 < 5e-4, "MMQ tile=2 parity ({w_ty:?}, {m}x{n}x{k}) rel_err={re2}");

            let mut tops = [0.0f64; 2];
            let mut ms_arr = [0.0f64; 2];
            for (ti, &ft) in [1i32, 2i32].iter().enumerate() {
                for _ in 0..3 {
                    unsafe {
                        glm5::gemm_mmq_mma_tiled(
                            w_ty as i32, d_xq.ptr, d_w.ptr, rb, d_out1.ptr.cast(),
                            std::ptr::null(), std::ptr::null(), std::ptr::null(),
                            m as i32, n as i32, k as i32, 1.0, 0.0, ft, &stream,
                        );
                    }
                }
                stream.sync().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..10 {
                    unsafe {
                        glm5::gemm_mmq_mma_tiled(
                            w_ty as i32, d_xq.ptr, d_w.ptr, rb, d_out1.ptr.cast(),
                            std::ptr::null(), std::ptr::null(), std::ptr::null(),
                            m as i32, n as i32, k as i32, 1.0, 0.0, ft, &stream,
                        );
                    }
                }
                stream.sync().unwrap();
                let s = t0.elapsed().as_secs_f64() / 10.0;
                ms_arr[ti] = s * 1000.0;
                tops[ti] = (2.0 * m as f64 * n as f64 * k as f64) / (s * 1e12);
            }
            println!(
                "MICROBENCH MMQ ({w_ty:?} {m}x{n}x{k}): re1={re1:.7} re2={re2:.7} | tile128x64={:.3}ms ({:.2} TOPS) | tile64x64={:.3}ms ({:.2} TOPS)",
                ms_arr[0], tops[0], ms_arr[1], tops[1]
            );
        }
    }

    // 4. kda_conv_silu_l2 (d_conv=4, T=64) & kda_gate_beta (T=64) parity vs T=1 sequential
    {
        let t_steps = 64usize;
        let head_dim = 128usize;
        let n_head = 16usize;
        let d_inner = head_dim * n_head;
        let d_conv = 4usize;

        let mut h_x = vec![0.0f32; t_steps * d_inner];
        let mut h_cw = vec![0.0f32; d_inner * d_conv];
        let mut h_cs_init = vec![0.0f32; d_inner * (d_conv - 1)];
        for v in &mut h_x {
            *v = rng.next_f32() * 0.5;
        }
        for v in &mut h_cw {
            *v = rng.next_f32() * 0.3;
        }
        for v in &mut h_cs_init {
            *v = rng.next_f32() * 0.4;
        }

        let d_x = DeviceBuffer::alloc(0, h_x.len() * 4).unwrap();
        let d_cw = DeviceBuffer::alloc(0, h_cw.len() * 4).unwrap();
        let d_cs_bulk = DeviceBuffer::alloc(0, h_cs_init.len() * 4).unwrap();
        let d_cs_seq = DeviceBuffer::alloc(0, h_cs_init.len() * 4).unwrap();
        let d_o_bulk = DeviceBuffer::alloc_zeroed(0, h_x.len() * 4).unwrap();
        let d_o_seq = DeviceBuffer::alloc_zeroed(0, h_x.len() * 4).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&h_x)).unwrap();
        d_cw.copy_from_host(bytemuck::cast_slice(&h_cw)).unwrap();

        for &apply_l2 in &[true, false] {
            d_cs_bulk.copy_from_host(bytemuck::cast_slice(&h_cs_init)).unwrap();
            d_cs_seq.copy_from_host(bytemuck::cast_slice(&h_cs_init)).unwrap();
            unsafe {
                glm5::kda_conv_silu_l2(
                    d_x.ptr.cast(), d_cw.ptr.cast(), d_cs_bulk.ptr.cast(),
                    d_inner as i32, d_conv as i32, head_dim as i32, n_head as i32,
                    apply_l2, t_steps as i32, d_o_bulk.ptr.cast(), &stream,
                );
                for t in 0..t_steps {
                    let off_b = t * d_inner * 4;
                    glm5::kda_conv_silu_l2(
                        d_x.ptr.cast::<u8>().add(off_b).cast(),
                        d_cw.ptr.cast(),
                        d_cs_seq.ptr.cast(),
                        d_inner as i32, d_conv as i32, head_dim as i32, n_head as i32,
                        apply_l2, 1,
                        d_o_seq.ptr.cast::<u8>().add(off_b).cast(),
                        &stream,
                    );
                }
            }
            stream.sync().unwrap();
            let mut h_bulk = vec![0.0f32; h_x.len()];
            let mut h_seq = vec![0.0f32; h_x.len()];
            let mut h_cs1 = vec![0.0f32; h_cs_init.len()];
            let mut h_cs2 = vec![0.0f32; h_cs_init.len()];
            d_o_bulk.copy_to_host(bytemuck::cast_slice_mut(&mut h_bulk)).unwrap();
            d_o_seq.copy_to_host(bytemuck::cast_slice_mut(&mut h_seq)).unwrap();
            d_cs_bulk.copy_to_host(bytemuck::cast_slice_mut(&mut h_cs1)).unwrap();
            d_cs_seq.copy_to_host(bytemuck::cast_slice_mut(&mut h_cs2)).unwrap();
            let me_o = max_abs_err(&h_bulk, &h_seq);
            let me_s = max_abs_err(&h_cs1, &h_cs2);
            println!("Stage C.7 kda_conv_silu_l2 (d_conv=4, T={t_steps}, l2={apply_l2}): max_err_out={me_o:.8}, max_err_state={me_s:.8}");
            assert!(me_o < 1e-4 && me_s < 1e-5, "kda_conv_silu_l2 parity failed: out={me_o} state={me_s}");
        }

        let mut h_gin = vec![0.0f32; t_steps * d_inner];
        let mut h_dt = vec![0.0f32; d_inner];
        let mut h_sa = vec![0.0f32; n_head];
        let mut h_beta_init = vec![0.0f32; t_steps * n_head];
        for v in &mut h_gin {
            *v = rng.next_f32() * 0.8;
        }
        for v in &mut h_dt {
            *v = rng.next_f32() * 0.4;
        }
        for v in &mut h_sa {
            *v = -(0.5 + rng.next_f32().abs() * 0.5);
        }
        for v in &mut h_beta_init {
            *v = rng.next_f32() * 1.2;
        }

        let d_gin = DeviceBuffer::alloc(0, h_gin.len() * 4).unwrap();
        let d_dt = DeviceBuffer::alloc(0, h_dt.len() * 4).unwrap();
        let d_sa = DeviceBuffer::alloc(0, h_sa.len() * 4).unwrap();
        let d_g1_bulk = DeviceBuffer::alloc_zeroed(0, h_gin.len() * 4).unwrap();
        let d_g1_seq = DeviceBuffer::alloc_zeroed(0, h_gin.len() * 4).unwrap();
        let d_b_bulk = DeviceBuffer::alloc(0, h_beta_init.len() * 4).unwrap();
        let d_b_seq = DeviceBuffer::alloc(0, h_beta_init.len() * 4).unwrap();
        d_gin.copy_from_host(bytemuck::cast_slice(&h_gin)).unwrap();
        d_dt.copy_from_host(bytemuck::cast_slice(&h_dt)).unwrap();
        d_sa.copy_from_host(bytemuck::cast_slice(&h_sa)).unwrap();
        d_b_bulk.copy_from_host(bytemuck::cast_slice(&h_beta_init)).unwrap();
        d_b_seq.copy_from_host(bytemuck::cast_slice(&h_beta_init)).unwrap();

        unsafe {
            glm5::kda_gate_beta(
                d_gin.ptr.cast(), d_dt.ptr.cast(), d_sa.ptr.cast(), -5.0,
                head_dim as i32, n_head as i32, t_steps as i32,
                d_g1_bulk.ptr.cast(), d_b_bulk.ptr.cast(), &stream,
            );
            for t in 0..t_steps {
                let off_g = t * d_inner * 4;
                let off_b = t * n_head * 4;
                glm5::kda_gate_beta(
                    d_gin.ptr.cast::<u8>().add(off_g).cast(),
                    d_dt.ptr.cast(), d_sa.ptr.cast(), -5.0,
                    head_dim as i32, n_head as i32, 1,
                    d_g1_seq.ptr.cast::<u8>().add(off_g).cast(),
                    d_b_seq.ptr.cast::<u8>().add(off_b).cast(),
                    &stream,
                );
            }
        }
        stream.sync().unwrap();
        let mut h_g1_b = vec![0.0f32; h_gin.len()];
        let mut h_g1_s = vec![0.0f32; h_gin.len()];
        let mut h_b_b = vec![0.0f32; h_beta_init.len()];
        let mut h_b_s = vec![0.0f32; h_beta_init.len()];
        d_g1_bulk.copy_to_host(bytemuck::cast_slice_mut(&mut h_g1_b)).unwrap();
        d_g1_seq.copy_to_host(bytemuck::cast_slice_mut(&mut h_g1_s)).unwrap();
        d_b_bulk.copy_to_host(bytemuck::cast_slice_mut(&mut h_b_b)).unwrap();
        d_b_seq.copy_to_host(bytemuck::cast_slice_mut(&mut h_b_s)).unwrap();
        let me_g = max_abs_err(&h_g1_b, &h_g1_s);
        let me_b = max_abs_err(&h_b_b, &h_b_s);
        assert!(h_g1_b.iter().all(|v| v.is_finite()) && h_b_b.iter().all(|v| v.is_finite()));
        println!("Stage C.7 kda_gate_beta (T={t_steps}): max_err_g1={me_g:.8}, max_err_beta={me_b:.8}");
        assert!(me_g < 1e-4 && me_b < 1e-4, "kda_gate_beta parity failed: g1={me_g} beta={me_b}");
    }
}

#[test]
fn test_glm5_stage_c8_kernels_parity_and_bench() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    use std::ffi::c_void;
    ampllm_cuda_kernels::set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x57A6_EC80_0001_u64);

    // 1. gemm_mmq_mma_grouped parity (12 experts, overlapping tokens, IQ2_S gate+up,
    //    IQ3_XXS & IQ3_S atomic-scatter down, tile_mode=1 bm=128 & tile_mode=2 bm=64)
    //    + 16-expert (total_rows=2400) microbenchmark vs sequential launches.
    {
        let n_exp_total = 16usize;
        let n_exp_par = 12usize;
        let m_tot_par = 256usize;
        let n_embd = 4096usize;
        let n_ff = 2048usize;
        let clamp_lim = 7.0f32;
        let gu_rb = GgmlType::Iq2S.row_bytes(n_embd);
        let d_rb_xxs = GgmlType::Iq3Xxs.row_bytes(n_ff);
        let d_rb_s = GgmlType::Iq3S.row_bytes(n_ff);
        let gu_bytes = 2 * n_ff * gu_rb;
        let dn_xxs_bytes = n_embd * d_rb_xxs;
        let dn_s_bytes = n_embd * d_rb_s;
        let off_xxs = gu_bytes;
        let off_s = gu_bytes + dn_xxs_bytes;
        let exp_stride = off_s + dn_s_bytes;

        let mut d_exp_bufs = Vec::with_capacity(n_exp_total);
        for _ in 0..n_exp_total {
            let mut h_exp = Vec::with_capacity(exp_stride);
            for _ in 0..(2 * n_ff) {
                h_exp.extend_from_slice(&fill_glm5_quant_row(GgmlType::Iq2S, n_embd, &mut rng));
            }
            for _ in 0..n_embd {
                h_exp.extend_from_slice(&fill_glm5_quant_row(GgmlType::Iq3Xxs, n_ff, &mut rng));
            }
            for _ in 0..n_embd {
                h_exp.extend_from_slice(&fill_glm5_quant_row(GgmlType::Iq3S, n_ff, &mut rng));
            }
            let db = DeviceBuffer::alloc(0, h_exp.len()).unwrap();
            db.copy_from_host(&h_exp).unwrap();
            d_exp_bufs.push(db);
        }

        let par_nts = [17usize, 65, 130, 33, 128, 64, 256, 19, 97, 145, 48, 200];
        let total_rows_par: usize = par_nts.iter().sum();
        let mut h_row_off = Vec::with_capacity(n_exp_par);
        let mut h_exp_nt = Vec::with_capacity(n_exp_par);
        let mut h_rows_all = Vec::with_capacity(total_rows_par);
        let mut h_wts_all = Vec::with_capacity(total_rows_par);
        let mut cur_off = 0usize;
        for (e, &nt) in par_nts.iter().enumerate() {
            h_row_off.push(cur_off as i32);
            h_exp_nt.push(nt as i32);
            for r in 0..nt {
                // Distinct within expert e (gcd(5, 256) == 1), overlapping across experts
                h_rows_all.push(((e * 19 + r * 5 + 3) % m_tot_par) as i32);
                h_wts_all.push(0.12 + ((e + r) % 9) as f32 * 0.04);
            }
            cur_off += nt;
        }
        let h_exp_ptrs_par: Vec<u64> = (0..n_exp_par).map(|e| d_exp_bufs[e].ptr as u64).collect();
        let h_x: Vec<f32> = (0..m_tot_par * n_embd).map(|_| rng.next_f32() * 0.35).collect();
        let h_base: Vec<f32> = (0..m_tot_par * n_embd).map(|_| rng.next_f32() * 0.2).collect();

        let d_x = DeviceBuffer::alloc(0, h_x.len() * 4).unwrap();
        let d_xq = DeviceBuffer::alloc(0, m_tot_par * (n_embd / 32) * 36).unwrap();
        let d_rows_all = DeviceBuffer::alloc(0, h_rows_all.len() * 4).unwrap();
        let d_wts_all = DeviceBuffer::alloc(0, h_wts_all.len() * 4).unwrap();
        let d_exp_ptrs = DeviceBuffer::alloc(0, n_exp_total * 8).unwrap();
        let d_exp_row_off = DeviceBuffer::alloc(0, n_exp_total * 4).unwrap();
        let d_exp_nt = DeviceBuffer::alloc(0, n_exp_total * 4).unwrap();
        let max_tiles = 64usize;
        let d_tile_exp = DeviceBuffer::alloc(0, max_tiles * 4).unwrap();
        let d_tile_m = DeviceBuffer::alloc(0, max_tiles * 4).unwrap();

        let bench_total_rows = 2400usize;
        let max_rows_buf = total_rows_par.max(bench_total_rows);
        let m_tot_bench = 300usize;
        let max_m_tot = m_tot_par.max(m_tot_bench);
        let d_gu_ref = DeviceBuffer::alloc_zeroed(0, max_rows_buf * 2 * n_ff * 4).unwrap();
        let d_iq8_ref = DeviceBuffer::alloc_zeroed(0, max_rows_buf * (n_ff / 32) * 36).unwrap();
        let d_out_ref = DeviceBuffer::alloc(0, max_m_tot * n_embd * 4).unwrap();
        let d_gu_grp = DeviceBuffer::alloc_zeroed(0, max_rows_buf * 2 * n_ff * 4).unwrap();
        let d_iq8_grp = DeviceBuffer::alloc_zeroed(0, max_rows_buf * (n_ff / 32) * 36).unwrap();
        let d_out_grp = DeviceBuffer::alloc(0, max_m_tot * n_embd * 4).unwrap();

        d_x.copy_from_host(bytemuck::cast_slice(&h_x)).unwrap();
        d_rows_all.copy_from_host(bytemuck::cast_slice(&h_rows_all)).unwrap();
        d_wts_all.copy_from_host(bytemuck::cast_slice(&h_wts_all)).unwrap();
        d_exp_ptrs.copy_from_host(bytemuck::cast_slice(&h_exp_ptrs_par)).unwrap();
        d_exp_row_off.copy_from_host(bytemuck::cast_slice(&h_row_off)).unwrap();
        d_exp_nt.copy_from_host(bytemuck::cast_slice(&h_exp_nt)).unwrap();
        unsafe {
            launch_quantize_q8_1(d_x.ptr.cast(), d_xq.ptr, m_tot_par as i32, n_embd as i32, stream.raw);
        }

        let iq8_row_bytes = (n_ff / 32) * 36;
        for &(dn_ty, dn_off, dn_rb) in &[
            (GgmlType::Iq3Xxs, off_xxs, d_rb_xxs),
            (GgmlType::Iq3S, off_s, d_rb_s),
        ] {
            d_out_ref
                .copy_from_host(bytemuck::cast_slice(&h_base))
                .unwrap();
            unsafe {
                for (e, &nt) in par_nts.iter().enumerate() {
                    let off = h_row_off[e] as usize;
                    let w_gu = d_exp_bufs[e].ptr;
                    let w_dn = d_exp_bufs[e].ptr.cast::<u8>().add(dn_off).cast();
                    let gu_dst = d_gu_ref.ptr.cast::<f32>().add(off * 2 * n_ff);
                    let iq8_dst = d_iq8_ref.ptr.cast::<u8>().add(off * iq8_row_bytes).cast();
                    let rows_p = d_rows_all.ptr.cast::<i32>().add(off);
                    let wts_p = d_wts_all.ptr.cast::<f32>().add(off);
                    glm5::gemm_mmq_mma(
                        GgmlType::Iq2S as i32,
                        d_xq.ptr,
                        w_gu,
                        gu_rb,
                        gu_dst,
                        rows_p,
                        std::ptr::null(),
                        std::ptr::null(),
                        nt as i32,
                        (2 * n_ff) as i32,
                        n_embd as i32,
                        1.0,
                        0.0,
                        &stream,
                    );
                    glm5::swiglu_clamp_interleaved_q8(
                        gu_dst,
                        clamp_lim,
                        nt as i32,
                        n_ff as i32,
                        iq8_dst,
                        &stream,
                    );
                    glm5::gemm_mmq_mma(
                        dn_ty as i32,
                        iq8_dst,
                        w_dn,
                        dn_rb,
                        d_out_ref.ptr.cast(),
                        std::ptr::null(),
                        rows_p,
                        wts_p,
                        nt as i32,
                        n_embd as i32,
                        n_ff as i32,
                        1.0,
                        0.0,
                        &stream,
                    );
                }
            }
            stream.sync().unwrap();
            let mut h_out_ref = vec![0.0f32; m_tot_par * n_embd];
            d_out_ref
                .copy_to_host(bytemuck::cast_slice_mut(&mut h_out_ref))
                .unwrap();

            for &tile_mode in &[1i32, 2i32] {
                let bm = if tile_mode == 2 { 64usize } else { 128usize };
                let mut h_tile_exp = Vec::new();
                let mut h_tile_m = Vec::new();
                for (e, &nt) in par_nts.iter().enumerate() {
                    let mut m0 = 0usize;
                    while m0 < nt {
                        h_tile_exp.push(e as i32);
                        h_tile_m.push(m0 as i32);
                        m0 += bm;
                    }
                }
                let n_m_tiles = h_tile_exp.len() as i32;
                d_tile_exp
                    .copy_from_host(bytemuck::cast_slice(&h_tile_exp))
                    .unwrap();
                d_tile_m
                    .copy_from_host(bytemuck::cast_slice(&h_tile_m))
                    .unwrap();
                d_out_grp
                    .copy_from_host(bytemuck::cast_slice(&h_base))
                    .unwrap();

                unsafe {
                    glm5::gemm_mmq_mma_grouped(
                        GgmlType::Iq2S as i32,
                        d_xq.ptr,
                        d_exp_ptrs.ptr.cast(),
                        0,
                        gu_rb,
                        d_gu_grp.ptr.cast(),
                        d_rows_all.ptr.cast(),
                        std::ptr::null(),
                        std::ptr::null(),
                        d_exp_row_off.ptr.cast(),
                        d_exp_nt.ptr.cast(),
                        d_tile_exp.ptr.cast(),
                        d_tile_m.ptr.cast(),
                        n_m_tiles,
                        (2 * n_ff) as i32,
                        n_embd as i32,
                        1.0,
                        false,
                        tile_mode,
                        &stream,
                    );
                    glm5::swiglu_clamp_interleaved_q8(
                        d_gu_grp.ptr.cast(),
                        clamp_lim,
                        total_rows_par as i32,
                        n_ff as i32,
                        d_iq8_grp.ptr,
                        &stream,
                    );
                    glm5::gemm_mmq_mma_grouped(
                        dn_ty as i32,
                        d_iq8_grp.ptr,
                        d_exp_ptrs.ptr.cast(),
                        dn_off,
                        dn_rb,
                        d_out_grp.ptr.cast(),
                        std::ptr::null(),
                        d_rows_all.ptr.cast(),
                        d_wts_all.ptr.cast(),
                        d_exp_row_off.ptr.cast(),
                        d_exp_nt.ptr.cast(),
                        d_tile_exp.ptr.cast(),
                        d_tile_m.ptr.cast(),
                        n_m_tiles,
                        n_embd as i32,
                        n_ff as i32,
                        1.0,
                        true,
                        tile_mode,
                        &stream,
                    );
                }
                stream.sync().unwrap();

                let mut h_out_grp = vec![0.0f32; m_tot_par * n_embd];
                d_out_grp
                    .copy_to_host(bytemuck::cast_slice_mut(&mut h_out_grp))
                    .unwrap();
                assert!(
                    h_out_grp.iter().all(|v| v.is_finite()),
                    "NaN/Inf in grouped MoE ({dn_ty:?}, tile_mode={tile_mode})"
                );
                let re = rel_err(&h_out_grp, &h_out_ref);
                println!(
                    "Stage C.8 gemm_mmq_mma_grouped ({dn_ty:?}, tile_mode={tile_mode}, exp={n_exp_par}, rows={total_rows_par}): rel_err={re:.7}"
                );
                assert!(
                    re < 3e-4,
                    "grouped MoE parity failed ({dn_ty:?}, tile_mode={tile_mode}): rel_err={re}"
                );
            }
        }

        // 16-expert (2400 rows) microbench: grouped vs 16 sequential launches
        let bench_nts = [
            220usize, 180, 140, 160, 120, 200, 90, 110, 190, 170, 130, 150, 80, 160, 140, 160,
        ];
        assert_eq!(bench_nts.iter().sum::<usize>(), bench_total_rows);
        let mut b_row_off = Vec::with_capacity(n_exp_total);
        let mut b_exp_nt = Vec::with_capacity(n_exp_total);
        let mut b_rows_all = Vec::with_capacity(bench_total_rows);
        let mut b_wts_all = Vec::with_capacity(bench_total_rows);
        let mut cur = 0usize;
        for (e, &nt) in bench_nts.iter().enumerate() {
            b_row_off.push(cur as i32);
            b_exp_nt.push(nt as i32);
            for r in 0..nt {
                b_rows_all.push(((e * 17 + r * 7 + 1) % m_tot_bench) as i32);
                b_wts_all.push(0.15f32);
            }
            cur += nt;
        }
        let h_exp_ptrs_b: Vec<u64> = d_exp_bufs.iter().map(|b| b.ptr as u64).collect();
        let d_b_rows = DeviceBuffer::alloc(0, b_rows_all.len() * 4).unwrap();
        let d_b_wts = DeviceBuffer::alloc(0, b_wts_all.len() * 4).unwrap();
        let d_xq_b = DeviceBuffer::alloc_zeroed(0, m_tot_bench * (n_embd / 32) * 36).unwrap();
        d_b_rows.copy_from_host(bytemuck::cast_slice(&b_rows_all)).unwrap();
        d_b_wts.copy_from_host(bytemuck::cast_slice(&b_wts_all)).unwrap();
        d_exp_ptrs.copy_from_host(bytemuck::cast_slice(&h_exp_ptrs_b)).unwrap();
        d_exp_row_off.copy_from_host(bytemuck::cast_slice(&b_row_off)).unwrap();
        d_exp_nt.copy_from_host(bytemuck::cast_slice(&b_exp_nt)).unwrap();

        let run_seq_16 = || unsafe {
            for (e, &nt) in bench_nts.iter().enumerate() {
                let off = b_row_off[e] as usize;
                let w_gu = d_exp_bufs[e].ptr;
                let w_dn = d_exp_bufs[e].ptr.cast::<u8>().add(off_xxs).cast();
                let gu_dst = d_gu_ref.ptr.cast::<f32>().add(off * 2 * n_ff);
                let iq8_dst = d_iq8_ref.ptr.cast::<u8>().add(off * iq8_row_bytes).cast();
                let rows_p = d_b_rows.ptr.cast::<i32>().add(off);
                let wts_p = d_b_wts.ptr.cast::<f32>().add(off);
                glm5::gemm_mmq_mma(
                    GgmlType::Iq2S as i32,
                    d_xq_b.ptr,
                    w_gu,
                    gu_rb,
                    gu_dst,
                    rows_p,
                    std::ptr::null(),
                    std::ptr::null(),
                    nt as i32,
                    (2 * n_ff) as i32,
                    n_embd as i32,
                    1.0,
                    0.0,
                    &stream,
                );
                glm5::swiglu_clamp_interleaved_q8(
                    gu_dst,
                    clamp_lim,
                    nt as i32,
                    n_ff as i32,
                    iq8_dst,
                    &stream,
                );
                glm5::gemm_mmq_mma(
                    GgmlType::Iq3Xxs as i32,
                    iq8_dst,
                    w_dn,
                    d_rb_xxs,
                    d_out_ref.ptr.cast(),
                    std::ptr::null(),
                    rows_p,
                    wts_p,
                    nt as i32,
                    n_embd as i32,
                    n_ff as i32,
                    1.0,
                    0.0,
                    &stream,
                );
            }
        };
        for _ in 0..3 {
            run_seq_16();
        }
        stream.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..15 {
            run_seq_16();
        }
        stream.sync().unwrap();
        let ms_seq = t0.elapsed().as_secs_f64() * 1000.0 / 15.0;

        let mut ms_grp = [0.0f64; 2];
        for (idx, &tile_mode) in [1i32, 2i32].iter().enumerate() {
            let bm = if tile_mode == 2 { 64usize } else { 128usize };
            let mut b_tile_exp = Vec::new();
            let mut b_tile_m = Vec::new();
            for (e, &nt) in bench_nts.iter().enumerate() {
                let mut m0 = 0usize;
                while m0 < nt {
                    b_tile_exp.push(e as i32);
                    b_tile_m.push(m0 as i32);
                    m0 += bm;
                }
            }
            let n_m_tiles = b_tile_exp.len() as i32;
            d_tile_exp
                .copy_from_host(bytemuck::cast_slice(&b_tile_exp))
                .unwrap();
            d_tile_m
                .copy_from_host(bytemuck::cast_slice(&b_tile_m))
                .unwrap();
            let run_grp_16 = || unsafe {
                glm5::gemm_mmq_mma_grouped(
                    GgmlType::Iq2S as i32,
                    d_xq_b.ptr,
                    d_exp_ptrs.ptr.cast(),
                    0,
                    gu_rb,
                    d_gu_grp.ptr.cast(),
                    d_b_rows.ptr.cast(),
                    std::ptr::null(),
                    std::ptr::null(),
                    d_exp_row_off.ptr.cast(),
                    d_exp_nt.ptr.cast(),
                    d_tile_exp.ptr.cast(),
                    d_tile_m.ptr.cast(),
                    n_m_tiles,
                    (2 * n_ff) as i32,
                    n_embd as i32,
                    1.0,
                    false,
                    tile_mode,
                    &stream,
                );
                glm5::swiglu_clamp_interleaved_q8(
                    d_gu_grp.ptr.cast(),
                    clamp_lim,
                    bench_total_rows as i32,
                    n_ff as i32,
                    d_iq8_grp.ptr,
                    &stream,
                );
                glm5::gemm_mmq_mma_grouped(
                    GgmlType::Iq3Xxs as i32,
                    d_iq8_grp.ptr,
                    d_exp_ptrs.ptr.cast(),
                    off_xxs,
                    d_rb_xxs,
                    d_out_grp.ptr.cast(),
                    std::ptr::null(),
                    d_b_rows.ptr.cast(),
                    d_b_wts.ptr.cast(),
                    d_exp_row_off.ptr.cast(),
                    d_exp_nt.ptr.cast(),
                    d_tile_exp.ptr.cast(),
                    d_tile_m.ptr.cast(),
                    n_m_tiles,
                    n_embd as i32,
                    n_ff as i32,
                    1.0,
                    true,
                    tile_mode,
                    &stream,
                );
            };
            for _ in 0..3 {
                run_grp_16();
            }
            stream.sync().unwrap();
            let t1 = std::time::Instant::now();
            for _ in 0..15 {
                run_grp_16();
            }
            stream.sync().unwrap();
            ms_grp[idx] = t1.elapsed().as_secs_f64() * 1000.0 / 15.0;
        }
        println!(
            "MICROBENCH Stage C.8 MoE 16exp/2400rows: seq={ms_seq:.3} ms | grouped_bm128={:.3} ms | grouped_bm64={:.3} ms",
            ms_grp[0], ms_grp[1]
        );
        assert!(
            ms_grp[0].min(ms_grp[1]) < ms_seq,
            "grouped MoE did not beat sequential: seq={ms_seq:.3}ms grp=({:.3}, {:.3})ms",
            ms_grp[0],
            ms_grp[1]
        );
    }

    // 2. glm5::hc_pre_norm_q8 TC fast-path (BM=32, BN=24, BK=64) parity & microbench vs chunked old kernel
    {
        let n_embd = 4096usize;
        let hc = 4usize;
        let hc_dim = hc * n_embd;
        let mix_dim = (2 + hc) * hc;
        let iters = 20usize;
        let norm_eps = 1e-5f32;
        let hc_eps = 0.05f32;
        let q8_row_bytes = (n_embd / 32) * 36;

        let unpack_q8_1 = |bytes: &[u8], m_rows: usize| -> Vec<f32> {
            let n_blk = n_embd / 32;
            let mut out = vec![0.0f32; m_rows * n_embd];
            for r in 0..m_rows {
                for b in 0..n_blk {
                    let blk = &bytes[(r * n_blk + b) * 36..][..36];
                    let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    let dst = &mut out[r * n_embd + b * 32..][..32];
                    for j in 0..32 {
                        dst[j] = d * (blk[4 + j] as i8 as f32);
                    }
                }
            }
            out
        };

        let m_par = 503usize;
        let h_r: Vec<f32> = (0..m_par * hc_dim).map(|_| rng.next_f32() * 0.5).collect();
        let h_wfn: Vec<f32> = (0..mix_dim * hc_dim).map(|_| rng.next_f32() * 0.04).collect();
        let h_ws = [0.4f32, 0.35, 0.3];
        let h_wb: Vec<f32> = (0..mix_dim).map(|_| rng.next_f32() * 0.1).collect();
        let h_nw: Vec<f32> = (0..n_embd).map(|_| 1.0 + rng.next_f32() * 0.1).collect();

        let d_r = DeviceBuffer::alloc(0, h_r.len() * 4).unwrap();
        let d_wfn_f32 = DeviceBuffer::alloc(0, h_wfn.len() * 4).unwrap();
        let h_wfn_bf16: Vec<u16> = h_wfn.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let d_wfn_bf16 = DeviceBuffer::alloc(0, h_wfn_bf16.len() * 2).unwrap();
        let d_ws = DeviceBuffer::alloc(0, 12).unwrap();
        let d_wb = DeviceBuffer::alloc(0, h_wb.len() * 4).unwrap();
        let d_nw = DeviceBuffer::alloc(0, h_nw.len() * 4).unwrap();

        let d_po_ref = DeviceBuffer::alloc_zeroed(0, m_par * hc * 4).unwrap();
        let d_co_ref = DeviceBuffer::alloc_zeroed(0, m_par * hc * hc * 4).unwrap();
        let d_xo_ref = DeviceBuffer::alloc_zeroed(0, m_par * n_embd * 4).unwrap();
        let d_xq_ref = DeviceBuffer::alloc_zeroed(0, m_par * q8_row_bytes).unwrap();
        let d_po_tc = DeviceBuffer::alloc_zeroed(0, m_par * hc * 4).unwrap();
        let d_co_tc = DeviceBuffer::alloc_zeroed(0, m_par * hc * hc * 4).unwrap();
        let d_xo_tc = DeviceBuffer::alloc_zeroed(0, m_par * n_embd * 4).unwrap();
        let d_xq_tc = DeviceBuffer::alloc_zeroed(0, m_par * q8_row_bytes).unwrap();

        d_r.copy_from_host(bytemuck::cast_slice(&h_r)).unwrap();
        d_wfn_f32.copy_from_host(bytemuck::cast_slice(&h_wfn)).unwrap();
        d_wfn_bf16.copy_from_host(bytemuck::cast_slice(&h_wfn_bf16)).unwrap();
        d_ws.copy_from_host(bytemuck::cast_slice(&h_ws)).unwrap();
        d_wb.copy_from_host(bytemuck::cast_slice(&h_wb)).unwrap();
        d_nw.copy_from_host(bytemuck::cast_slice(&h_nw)).unwrap();

        for &(w_ty, w_ptr) in &[
            (GgmlType::F32, d_wfn_f32.ptr),
            (GgmlType::Bf16, d_wfn_bf16.ptr),
        ] {
            unsafe {
                for c0 in (0..m_par).step_by(64) {
                    let cm = (m_par - c0).min(64);
                    glm5::hc_pre_norm_q8(
                        d_r.ptr.cast::<f32>().add(c0 * hc_dim),
                        w_ptr,
                        w_ty as i32,
                        d_ws.ptr.cast(),
                        d_wb.ptr.cast(),
                        d_nw.ptr.cast(),
                        n_embd as i32,
                        hc as i32,
                        iters as i32,
                        norm_eps,
                        hc_eps,
                        d_po_ref.ptr.cast::<f32>().add(c0 * hc),
                        d_co_ref.ptr.cast::<f32>().add(c0 * hc * hc),
                        d_xo_ref.ptr.cast::<f32>().add(c0 * n_embd),
                        std::ptr::null_mut(),
                        cm as i32,
                        &stream,
                    );
                }
                launch_quantize_q8_1(
                    d_xo_ref.ptr.cast(),
                    d_xq_ref.ptr,
                    m_par as i32,
                    n_embd as i32,
                    stream.raw,
                );
                glm5::hc_pre_norm_q8(
                    d_r.ptr.cast(),
                    w_ptr,
                    w_ty as i32,
                    d_ws.ptr.cast(),
                    d_wb.ptr.cast(),
                    d_nw.ptr.cast(),
                    n_embd as i32,
                    hc as i32,
                    iters as i32,
                    norm_eps,
                    hc_eps,
                    d_po_tc.ptr.cast(),
                    d_co_tc.ptr.cast(),
                    d_xo_tc.ptr.cast(),
                    d_xq_tc.ptr,
                    m_par as i32,
                    &stream,
                );
            }
            stream.sync().unwrap();

            let mut po_ref = vec![0.0f32; m_par * hc];
            let mut po_tc = vec![0.0f32; m_par * hc];
            let mut co_ref = vec![0.0f32; m_par * hc * hc];
            let mut co_tc = vec![0.0f32; m_par * hc * hc];
            let mut xo_ref = vec![0.0f32; m_par * n_embd];
            let mut xo_tc = vec![0.0f32; m_par * n_embd];
            let mut xq_ref_bytes = vec![0u8; m_par * q8_row_bytes];
            let mut xq_tc_bytes = vec![0u8; m_par * q8_row_bytes];

            d_po_ref.copy_to_host(bytemuck::cast_slice_mut(&mut po_ref)).unwrap();
            d_po_tc.copy_to_host(bytemuck::cast_slice_mut(&mut po_tc)).unwrap();
            d_co_ref.copy_to_host(bytemuck::cast_slice_mut(&mut co_ref)).unwrap();
            d_co_tc.copy_to_host(bytemuck::cast_slice_mut(&mut co_tc)).unwrap();
            d_xo_ref.copy_to_host(bytemuck::cast_slice_mut(&mut xo_ref)).unwrap();
            d_xo_tc.copy_to_host(bytemuck::cast_slice_mut(&mut xo_tc)).unwrap();
            d_xq_ref.copy_to_host(&mut xq_ref_bytes).unwrap();
            d_xq_tc.copy_to_host(&mut xq_tc_bytes).unwrap();

            assert!(
                po_tc.iter().all(|v| v.is_finite())
                    && co_tc.iter().all(|v| v.is_finite())
                    && xo_tc.iter().all(|v| v.is_finite()),
                "NaN/Inf in hc_pre_norm_q8 TC ({w_ty:?})"
            );
            let ma_p = max_abs_err(&po_tc, &po_ref);
            let re_p = rel_err(&po_tc, &po_ref);
            let ma_c = max_abs_err(&co_tc, &co_ref);
            let re_c = rel_err(&co_tc, &co_ref);
            let ma_x = max_abs_err(&xo_tc, &xo_ref);
            let re_x = rel_err(&xo_tc, &xo_ref);
            let xq_ref = unpack_q8_1(&xq_ref_bytes, m_par);
            let xq_tc = unpack_q8_1(&xq_tc_bytes, m_par);
            let re_q = rel_err(&xq_tc, &xq_ref);

            println!(
                "Stage C.8 hc_pre_norm_q8 TC ({w_ty:?}, M={m_par}): post=({ma_p:.6},{re_p:.6}) comb=({ma_c:.6},{re_c:.6}) x_f32=({ma_x:.6},{re_x:.6}) x_q8_re={re_q:.6}"
            );
            assert!(
                ma_p < 2e-3 && re_p < 1e-3 && ma_c < 2e-3 && re_c < 1e-3
                    && ma_x < 2e-3 && re_x < 1e-3 && re_q < 2e-3,
                "hc_pre_norm_q8 TC parity failed ({w_ty:?})"
            );
        }

        // Microbench: M=5003 full TC vs chunk-64 old kernel
        let m_b = 5003usize;
        let d_rb = DeviceBuffer::alloc_zeroed(0, m_b * hc_dim * 4).unwrap();
        let d_pob = DeviceBuffer::alloc_zeroed(0, m_b * hc * 4).unwrap();
        let d_cob = DeviceBuffer::alloc_zeroed(0, m_b * hc * hc * 4).unwrap();
        let d_xob = DeviceBuffer::alloc_zeroed(0, m_b * n_embd * 4).unwrap();
        let d_xqb = DeviceBuffer::alloc_zeroed(0, m_b * q8_row_bytes).unwrap();
        d_rb.copy_from_host(bytemuck::cast_slice(&h_r[..m_par * hc_dim])).unwrap();

        let run_tc = || unsafe {
            glm5::hc_pre_norm_q8(
                d_rb.ptr.cast(),
                d_wfn_f32.ptr,
                GgmlType::F32 as i32,
                d_ws.ptr.cast(),
                d_wb.ptr.cast(),
                d_nw.ptr.cast(),
                n_embd as i32,
                hc as i32,
                iters as i32,
                norm_eps,
                hc_eps,
                d_pob.ptr.cast(),
                d_cob.ptr.cast(),
                d_xob.ptr.cast(),
                d_xqb.ptr,
                m_b as i32,
                &stream,
            );
        };
        let run_old = || unsafe {
            for c0 in (0..m_b).step_by(64) {
                let cm = (m_b - c0).min(64);
                glm5::hc_pre_norm_q8(
                    d_rb.ptr.cast::<f32>().add(c0 * hc_dim),
                    d_wfn_f32.ptr,
                    GgmlType::F32 as i32,
                    d_ws.ptr.cast(),
                    d_wb.ptr.cast(),
                    d_nw.ptr.cast(),
                    n_embd as i32,
                    hc as i32,
                    iters as i32,
                    norm_eps,
                    hc_eps,
                    d_pob.ptr.cast::<f32>().add(c0 * hc),
                    d_cob.ptr.cast::<f32>().add(c0 * hc * hc),
                    d_xob.ptr.cast::<f32>().add(c0 * n_embd),
                    (d_xqb.ptr as *mut u8).add(c0 * q8_row_bytes).cast(),
                    cm as i32,
                    &stream,
                );
            }
        };

        for _ in 0..2 {
            run_tc();
            run_old();
        }
        stream.sync().unwrap();

        let t0 = std::time::Instant::now();
        for _ in 0..10 {
            run_tc();
        }
        stream.sync().unwrap();
        let ms_tc = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;

        let t1 = std::time::Instant::now();
        for _ in 0..10 {
            run_old();
        }
        stream.sync().unwrap();
        let ms_old = t1.elapsed().as_secs_f64() * 1000.0 / 10.0;

        println!(
            "MICROBENCH Stage C.8 hc_pre_norm_q8 (M={m_b}, n_embd={n_embd}, F32): tc={ms_tc:.3} ms | old_c64={ms_old:.3} ms ({:.2}x)",
            ms_old / ms_tc
        );
        assert!(
            ms_tc * 3.0 < ms_old,
            "hc_pre_norm_q8 TC speedup < 3x: tc={ms_tc:.3}ms vs old={ms_old:.3}ms"
        );
    }

    // 3. glm5::mv_batch_skip parity (K=4096 & K=1536, mixed Q6_K, Q8_0, F32 testing staged s_xq8)
    {
        for &k in &[4096usize, 1536usize] {
            let h_x: Vec<f32> = (0..k).map(|_| rng.next_f32() * 0.4).collect();
            let d_x = DeviceBuffer::alloc(0, k * 4).unwrap();
            let d_xq8 = DeviceBuffer::alloc(0, (k / 32) * 36).unwrap();
            d_x.copy_from_host(bytemuck::cast_slice(&h_x)).unwrap();
            unsafe {
                launch_quantize_q8_1(d_x.ptr.cast(), d_xq8.ptr, 1, k as i32, stream.raw);
            }

            let jobs = [
                (GgmlType::Q6K, 256usize, 1.0f32),
                (GgmlType::Q8_0, 128usize, 0.75f32),
                (GgmlType::F32, 64usize, 1.25f32),
            ];
            let mut d_ws = Vec::new();
            let mut d_ys = Vec::new();
            let mut d_refs = Vec::new();
            let mut f32_refs: Vec<Option<Vec<f32>>> = Vec::new();
            for &(ty, n, sc) in &jobs {
                let dw = if ty == GgmlType::F32 {
                    let hf: Vec<f32> = (0..n * k).map(|_| rng.next_f32() * 0.2).collect();
                    let mut rvec = Vec::with_capacity(n);
                    for r in 0..n {
                        let mut acc = 0.0f64;
                        for c in 0..k {
                            acc += hf[r * k + c] as f64 * h_x[c] as f64;
                        }
                        rvec.push((acc * sc as f64) as f32);
                    }
                    f32_refs.push(Some(rvec));
                    let b = DeviceBuffer::alloc(0, hf.len() * 4).unwrap();
                    b.copy_from_host(bytemuck::cast_slice(&hf)).unwrap();
                    b
                } else {
                    let rb = ty.row_bytes(k);
                    let mut h_w = Vec::with_capacity(n * rb);
                    for _ in 0..n {
                        h_w.extend_from_slice(&fill_glm5_quant_row(ty, k, &mut rng));
                    }
                    f32_refs.push(None);
                    let b = DeviceBuffer::alloc(0, h_w.len()).unwrap();
                    b.copy_from_host(&h_w).unwrap();
                    b
                };
                let d_y = DeviceBuffer::alloc_zeroed(0, n * 4).unwrap();
                let d_ref = DeviceBuffer::alloc_zeroed(0, n * 4).unwrap();
                if ty != GgmlType::F32 {
                    unsafe {
                        launch_linear_q8_1(
                            ty as i32,
                            dw.ptr,
                            d_xq8.ptr,
                            d_x.ptr.cast(),
                            d_ref.ptr.cast(),
                            n as i32,
                            k as i32,
                            1,
                            stream.raw,
                        );
                    }
                }
                d_ws.push(dw);
                d_ys.push(d_y);
                d_refs.push(d_ref);
            }
            let w_ptrs: Vec<*const c_void> = d_ws.iter().map(|b| b.ptr as *const c_void).collect();
            let y_ptrs: Vec<*mut f32> = d_ys.iter().map(|b| b.ptr.cast()).collect();
            let w_types: Vec<i32> = jobs.iter().map(|&(ty, _, _)| ty as i32).collect();
            let n_outs: Vec<i32> = jobs.iter().map(|&(_, n, _)| n as i32).collect();
            let scales: Vec<f32> = jobs.iter().map(|&(_, _, sc)| sc).collect();
            let d_skip = DeviceBuffer::alloc_zeroed(0, 4).unwrap();
            d_skip.copy_from_host(bytemuck::cast_slice(&[0i32])).unwrap();

            unsafe {
                glm5::mv_batch_skip(
                    &w_ptrs,
                    &y_ptrs,
                    &w_types,
                    &n_outs,
                    Some(&scales),
                    k as i32,
                    d_xq8.ptr,
                    d_x.ptr.cast(),
                    d_skip.ptr.cast(),
                    &stream,
                );
            }
            stream.sync().unwrap();

            for (i, &(ty, n, sc)) in jobs.iter().enumerate() {
                let mut got = vec![0.0f32; n];
                d_ys[i].copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
                let exp = if let Some(ref rv) = f32_refs[i] {
                    rv.clone()
                } else {
                    let mut rv = vec![0.0f32; n];
                    d_refs[i].copy_to_host(bytemuck::cast_slice_mut(&mut rv)).unwrap();
                    for v in &mut rv {
                        *v *= sc;
                    }
                    rv
                };
                let re = rel_err(&got, &exp);
                println!("Stage C.8 mv_batch_skip ({ty:?}, N={n}, K={k}): rel_err={re:.7}");
                assert!(re < 2e-4, "mv_batch_skip parity failed ({ty:?}, K={k}): rel_err={re}");
            }

            // skip=1 must leave zeroed buffers untouched
            let d_zero = DeviceBuffer::alloc_zeroed(0, jobs[0].1 * 4).unwrap();
            let y0 = [d_zero.ptr.cast::<f32>()];
            d_skip.copy_from_host(bytemuck::cast_slice(&[1i32])).unwrap();
            unsafe {
                glm5::mv_batch_skip(
                    &w_ptrs[..1],
                    &y0,
                    &w_types[..1],
                    &n_outs[..1],
                    Some(&scales[..1]),
                    k as i32,
                    d_xq8.ptr,
                    d_x.ptr.cast(),
                    d_skip.ptr.cast(),
                    &stream,
                );
            }
            stream.sync().unwrap();
            let mut got_skip = vec![1.0f32; jobs[0].1];
            d_zero.copy_to_host(bytemuck::cast_slice_mut(&mut got_skip)).unwrap();
            assert!(got_skip.iter().all(|&v| v == 0.0), "mv_batch_skip did not respect skip=1");
        }
    }
}

#[test]
fn test_glm5_stage_c9_kernels_parity_and_bench() {
    let _gpu_guard = GPU_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    ampllm_cuda_kernels::set_device(0).unwrap();
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x57A6_EC90_0001_u64);

    // 1. tile128x128 (force_tile=4) parity & sustained (10-iter + 60-iter) speed vs tile128x64 (force_tile=1)
    {
        let shapes = [
            (GgmlType::Q6K, 1024usize, 8192usize, 4096usize),
            (GgmlType::Q6K, 1024usize, 12288usize, 4096usize),
            (GgmlType::Q6K, 1024usize, 4096usize, 12288usize),
            (GgmlType::Iq2S, 256usize, 4096usize, 4096usize),
            (GgmlType::Iq3S, 256usize, 4096usize, 2048usize),
            (GgmlType::Q8_0, 512usize, 8192usize, 128usize),
        ];
        for &(w_ty, m, n, k) in &shapes {
            let rb = w_ty.row_bytes(k);
            let h_x: Vec<f32> = (0..m * k).map(|_| rng.next_f32() * 0.45).collect();
            let mut h_w = Vec::with_capacity(n * rb);
            for _ in 0..n {
                h_w.extend_from_slice(&fill_glm5_quant_row(w_ty, k, &mut rng));
            }
            let d_x = DeviceBuffer::alloc(0, h_x.len() * 4).unwrap();
            let d_xq = DeviceBuffer::alloc_zeroed(0, m * (k / 32) * 36).unwrap();
            let d_w = DeviceBuffer::alloc(0, h_w.len()).unwrap();
            let d_ref = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
            let d_out1 = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
            let d_out4 = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
            d_x.copy_from_host(bytemuck::cast_slice(&h_x)).unwrap();
            d_w.copy_from_host(&h_w).unwrap();

            unsafe {
                launch_quantize_q8_1(d_x.ptr.cast(), d_xq.ptr, m as i32, k as i32, stream.raw);
                launch_linear_q8_1(
                    w_ty as i32,
                    d_w.ptr,
                    d_xq.ptr,
                    d_x.ptr.cast(),
                    d_ref.ptr.cast(),
                    n as i32,
                    k as i32,
                    m as i32,
                    stream.raw,
                );
                glm5::gemm_mmq_mma_tiled(
                    w_ty as i32,
                    d_xq.ptr,
                    d_w.ptr,
                    rb,
                    d_out1.ptr.cast(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    0.0,
                    1,
                    &stream,
                );
                glm5::gemm_mmq_mma_tiled(
                    w_ty as i32,
                    d_xq.ptr,
                    d_w.ptr,
                    rb,
                    d_out4.ptr.cast(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    m as i32,
                    n as i32,
                    k as i32,
                    1.0,
                    0.0,
                    4,
                    &stream,
                );
            }
            stream.sync().unwrap();

            let mut h_ref = vec![0.0f32; m * n];
            let mut h_out1 = vec![0.0f32; m * n];
            let mut h_out4 = vec![0.0f32; m * n];
            d_ref.copy_to_host(bytemuck::cast_slice_mut(&mut h_ref)).unwrap();
            d_out1.copy_to_host(bytemuck::cast_slice_mut(&mut h_out1)).unwrap();
            d_out4.copy_to_host(bytemuck::cast_slice_mut(&mut h_out4)).unwrap();
            let re1 = rel_err(&h_out1, &h_ref);
            let re4 = rel_err(&h_out4, &h_ref);
            assert!(re1 < 5e-4, "MMQ tile=1 parity ({w_ty:?}, {m}x{n}x{k}) rel_err={re1}");
            assert!(re4 < 5e-4, "MMQ tile=4 parity ({w_ty:?}, {m}x{n}x{k}) rel_err={re4}");

            let mut ms_10 = [0.0f64; 2];
            let mut tops_10 = [0.0f64; 2];
            let mut ms_60 = [0.0f64; 2];
            let mut tops_60 = [0.0f64; 2];
            for (ti, &ft) in [1i32, 4i32].iter().enumerate() {
                for _ in 0..3 {
                    unsafe {
                        glm5::gemm_mmq_mma_tiled(
                            w_ty as i32,
                            d_xq.ptr,
                            d_w.ptr,
                            rb,
                            d_out4.ptr.cast(),
                            std::ptr::null(),
                            std::ptr::null(),
                            std::ptr::null(),
                            m as i32,
                            n as i32,
                            k as i32,
                            1.0,
                            0.0,
                            ft,
                            &stream,
                        );
                    }
                }
                stream.sync().unwrap();
                for &is_60 in &[false, true] {
                    let iters = if is_60 { 60usize } else { 10usize };
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        unsafe {
                            glm5::gemm_mmq_mma_tiled(
                                w_ty as i32,
                                d_xq.ptr,
                                d_w.ptr,
                                rb,
                                d_out4.ptr.cast(),
                                std::ptr::null(),
                                std::ptr::null(),
                                std::ptr::null(),
                                m as i32,
                                n as i32,
                                k as i32,
                                1.0,
                                0.0,
                                ft,
                                &stream,
                            );
                        }
                    }
                    stream.sync().unwrap();
                    let s = t0.elapsed().as_secs_f64() / iters as f64;
                    if is_60 {
                        ms_60[ti] = s * 1000.0;
                        tops_60[ti] = (2.0 * m as f64 * n as f64 * k as f64) / (s * 1e12);
                    } else {
                        ms_10[ti] = s * 1000.0;
                        tops_10[ti] = (2.0 * m as f64 * n as f64 * k as f64) / (s * 1e12);
                    }
                }
            }
            println!(
                "MICROBENCH Stage C.9 MMQ ({w_ty:?} {m}x{n}x{k}): re4={re4:.7} | t128x64: 10i={:.3}ms({:.2}T) 60i={:.3}ms({:.2}T) | t128x128: 10i={:.3}ms({:.2}T) 60i={:.3}ms({:.2}T)",
                ms_10[0], tops_10[0], ms_60[0], tops_60[0], ms_10[1], tops_10[1], ms_60[1], tops_60[1]
            );

        }
    }

    // 2. Diagnostic: 60-iter sustained dequant_to_f16 + gemm_f16_mma_nt on Q6_K (1024x8192x4096) at 200W
    {
        let (m, n, k) = (1024usize, 8192usize, 4096usize);
        let rb = GgmlType::Q6K.row_bytes(k);
        let d_xf32 = DeviceBuffer::alloc_zeroed(0, m * k * 4).unwrap();
        let d_af16 = DeviceBuffer::alloc_zeroed(0, m * k * 2).unwrap();
        let d_w = DeviceBuffer::alloc_zeroed(0, n * rb).unwrap();
        let d_bf16 = DeviceBuffer::alloc_zeroed(0, n * k * 2).unwrap();
        let d_out = DeviceBuffer::alloc_zeroed(0, m * n * 4).unwrap();
        for _ in 0..3 {
            unsafe {
                glm5::f32_to_f16(d_xf32.ptr.cast(), d_af16.ptr, (m * k) as i32, &stream);
                glm5::dequant_to_f16(GgmlType::Q6K as i32, d_w.ptr, d_bf16.ptr, (n * k) as i64, &stream);
                glm5::gemm_f16_mma_nt(d_af16.ptr, d_bf16.ptr, d_out.ptr.cast(), m as i32, n as i32, k as i32, 1.0, 0.0, &stream);
            }
        }
        stream.sync().unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..60 {
            unsafe {
                glm5::f32_to_f16(d_xf32.ptr.cast(), d_af16.ptr, (m * k) as i32, &stream);
                glm5::dequant_to_f16(GgmlType::Q6K as i32, d_w.ptr, d_bf16.ptr, (n * k) as i64, &stream);
                glm5::gemm_f16_mma_nt(d_af16.ptr, d_bf16.ptr, d_out.ptr.cast(), m as i32, n as i32, k as i32, 1.0, 0.0, &stream);
            }
        }
        stream.sync().unwrap();
        let s_deq_gemm = t0.elapsed().as_secs_f64() / 60.0;
        let t1 = std::time::Instant::now();
        for _ in 0..60 {
            unsafe {
                glm5::gemm_f16_mma_nt(d_af16.ptr, d_bf16.ptr, d_out.ptr.cast(), m as i32, n as i32, k as i32, 1.0, 0.0, &stream);
            }
        }
        stream.sync().unwrap();
        let s_gemm = t1.elapsed().as_secs_f64() / 60.0;
        println!(
            "MICROBENCH Stage C.9 F16 MMA 60i (Q6K 1024x8192x4096): f16_only={:.3}ms ({:.2}T) | with_f32_deq={:.3}ms ({:.2}T)",
            s_gemm * 1000.0,
            (2.0 * m as f64 * n as f64 * k as f64) / (s_gemm * 1e12),
            s_deq_gemm * 1000.0,
            (2.0 * m as f64 * n as f64 * k as f64) / (s_deq_gemm * 1e12)
        );
    }

    // 3. gemm_mmq_mma_grouped tile_mode=3 (BM=32) parity on IQ2_S Gate+Up + IQ3_S/IQ3_XXS Down
    {
        let n_exp_par = 10usize;
        let m_tot_par = 128usize;
        let n_embd = 4096usize;
        let n_ff = 2048usize;
        let clamp_lim = 7.0f32;
        let gu_rb = GgmlType::Iq2S.row_bytes(n_embd);
        let d_rb_xxs = GgmlType::Iq3Xxs.row_bytes(n_ff);
        let d_rb_s = GgmlType::Iq3S.row_bytes(n_ff);
        let gu_bytes = 2 * n_ff * gu_rb;
        let off_xxs = gu_bytes;
        let off_s = gu_bytes + n_embd * d_rb_xxs;
        let exp_stride = off_s + n_embd * d_rb_s;

        let mut d_exp_bufs = Vec::with_capacity(n_exp_par);
        for _ in 0..n_exp_par {
            let mut h_exp = Vec::with_capacity(exp_stride);
            for _ in 0..(2 * n_ff) {
                h_exp.extend_from_slice(&fill_glm5_quant_row(GgmlType::Iq2S, n_embd, &mut rng));
            }
            for _ in 0..n_embd {
                h_exp.extend_from_slice(&fill_glm5_quant_row(GgmlType::Iq3Xxs, n_ff, &mut rng));
            }
            for _ in 0..n_embd {
                h_exp.extend_from_slice(&fill_glm5_quant_row(GgmlType::Iq3S, n_ff, &mut rng));
            }
            let db = DeviceBuffer::alloc(0, h_exp.len()).unwrap();
            db.copy_from_host(&h_exp).unwrap();
            d_exp_bufs.push(db);
        }

        let par_nts = [17usize, 32, 19, 25, 31, 18, 29, 22, 17, 27];
        let total_rows_par: usize = par_nts.iter().sum();
        let mut h_row_off = Vec::with_capacity(n_exp_par);
        let mut h_exp_nt = Vec::with_capacity(n_exp_par);
        let mut h_rows_all = Vec::with_capacity(total_rows_par);
        let mut h_wts_all = Vec::with_capacity(total_rows_par);
        let mut h_tile_exp = Vec::with_capacity(n_exp_par);
        let mut h_tile_m = Vec::with_capacity(n_exp_par);
        let mut cur_off = 0usize;
        for (e, &nt) in par_nts.iter().enumerate() {
            h_row_off.push(cur_off as i32);
            h_exp_nt.push(nt as i32);
            h_tile_exp.push(e as i32);
            h_tile_m.push(0i32);
            for r in 0..nt {
                h_rows_all.push(((e * 13 + r * 5 + 1) % m_tot_par) as i32);
                h_wts_all.push(0.14 + ((e + r) % 7) as f32 * 0.03);
            }
            cur_off += nt;
        }
        let h_exp_ptrs: Vec<u64> = d_exp_bufs.iter().map(|b| b.ptr as u64).collect();
        let h_x: Vec<f32> = (0..m_tot_par * n_embd).map(|_| rng.next_f32() * 0.35).collect();
        let h_base: Vec<f32> = (0..m_tot_par * n_embd).map(|_| rng.next_f32() * 0.2).collect();

        let d_x = DeviceBuffer::alloc(0, h_x.len() * 4).unwrap();
        let d_xq = DeviceBuffer::alloc(0, m_tot_par * (n_embd / 32) * 36).unwrap();
        let d_rows_all = DeviceBuffer::alloc(0, h_rows_all.len() * 4).unwrap();
        let d_wts_all = DeviceBuffer::alloc(0, h_wts_all.len() * 4).unwrap();
        let d_exp_ptrs = DeviceBuffer::alloc(0, n_exp_par * 8).unwrap();
        let d_exp_row_off = DeviceBuffer::alloc(0, n_exp_par * 4).unwrap();
        let d_exp_nt = DeviceBuffer::alloc(0, n_exp_par * 4).unwrap();
        let d_tile_exp = DeviceBuffer::alloc(0, n_exp_par * 4).unwrap();
        let d_tile_m = DeviceBuffer::alloc(0, n_exp_par * 4).unwrap();
        let d_gu_ref = DeviceBuffer::alloc_zeroed(0, total_rows_par * 2 * n_ff * 4).unwrap();
        let d_iq8_ref = DeviceBuffer::alloc_zeroed(0, total_rows_par * (n_ff / 32) * 36).unwrap();
        let d_out_ref = DeviceBuffer::alloc(0, m_tot_par * n_embd * 4).unwrap();
        let d_gu_grp = DeviceBuffer::alloc_zeroed(0, total_rows_par * 2 * n_ff * 4).unwrap();
        let d_iq8_grp = DeviceBuffer::alloc_zeroed(0, total_rows_par * (n_ff / 32) * 36).unwrap();
        let d_out_grp = DeviceBuffer::alloc(0, m_tot_par * n_embd * 4).unwrap();

        d_x.copy_from_host(bytemuck::cast_slice(&h_x)).unwrap();
        d_rows_all.copy_from_host(bytemuck::cast_slice(&h_rows_all)).unwrap();
        d_wts_all.copy_from_host(bytemuck::cast_slice(&h_wts_all)).unwrap();
        d_exp_ptrs.copy_from_host(bytemuck::cast_slice(&h_exp_ptrs)).unwrap();
        d_exp_row_off.copy_from_host(bytemuck::cast_slice(&h_row_off)).unwrap();
        d_exp_nt.copy_from_host(bytemuck::cast_slice(&h_exp_nt)).unwrap();
        d_tile_exp.copy_from_host(bytemuck::cast_slice(&h_tile_exp)).unwrap();
        d_tile_m.copy_from_host(bytemuck::cast_slice(&h_tile_m)).unwrap();
        unsafe {
            launch_quantize_q8_1(d_x.ptr.cast(), d_xq.ptr, m_tot_par as i32, n_embd as i32, stream.raw);
        }

        let iq8_row_bytes = (n_ff / 32) * 36;
        for &(dn_ty, dn_off, dn_rb) in &[
            (GgmlType::Iq3Xxs, off_xxs, d_rb_xxs),
            (GgmlType::Iq3S, off_s, d_rb_s),
        ] {
            d_out_ref.copy_from_host(bytemuck::cast_slice(&h_base)).unwrap();
            d_out_grp.copy_from_host(bytemuck::cast_slice(&h_base)).unwrap();
            unsafe {
                for (e, &nt) in par_nts.iter().enumerate() {
                    let off = h_row_off[e] as usize;
                    let w_gu = d_exp_bufs[e].ptr;
                    let w_dn = d_exp_bufs[e].ptr.cast::<u8>().add(dn_off).cast();
                    let gu_dst = d_gu_ref.ptr.cast::<f32>().add(off * 2 * n_ff);
                    let iq8_dst = d_iq8_ref.ptr.cast::<u8>().add(off * iq8_row_bytes).cast();
                    let rows_p = d_rows_all.ptr.cast::<i32>().add(off);
                    let wts_p = d_wts_all.ptr.cast::<f32>().add(off);
                    glm5::gemm_mmq_mma(
                        GgmlType::Iq2S as i32, d_xq.ptr, w_gu, gu_rb, gu_dst,
                        rows_p, std::ptr::null(), std::ptr::null(),
                        nt as i32, (2 * n_ff) as i32, n_embd as i32, 1.0, 0.0, &stream,
                    );
                    glm5::swiglu_clamp_interleaved_q8(
                        gu_dst, clamp_lim, nt as i32, n_ff as i32, iq8_dst, &stream,
                    );
                    glm5::gemm_mmq_mma(
                        dn_ty as i32, iq8_dst, w_dn, dn_rb, d_out_ref.ptr.cast(),
                        std::ptr::null(), rows_p, wts_p,
                        nt as i32, n_embd as i32, n_ff as i32, 1.0, 0.0, &stream,
                    );
                }
                glm5::gemm_mmq_mma_grouped(
                    GgmlType::Iq2S as i32, d_xq.ptr, d_exp_ptrs.ptr.cast(), 0, gu_rb,
                    d_gu_grp.ptr.cast(), d_rows_all.ptr.cast(), std::ptr::null(), std::ptr::null(),
                    d_exp_row_off.ptr.cast(), d_exp_nt.ptr.cast(), d_tile_exp.ptr.cast(), d_tile_m.ptr.cast(),
                    n_exp_par as i32, (2 * n_ff) as i32, n_embd as i32, 1.0, false, 3, &stream,
                );
                glm5::swiglu_clamp_interleaved_q8(
                    d_gu_grp.ptr.cast(), clamp_lim, total_rows_par as i32, n_ff as i32, d_iq8_grp.ptr, &stream,
                );
                glm5::gemm_mmq_mma_grouped(
                    dn_ty as i32, d_iq8_grp.ptr, d_exp_ptrs.ptr.cast(), dn_off, dn_rb,
                    d_out_grp.ptr.cast(), std::ptr::null(), d_rows_all.ptr.cast(), d_wts_all.ptr.cast(),
                    d_exp_row_off.ptr.cast(), d_exp_nt.ptr.cast(), d_tile_exp.ptr.cast(), d_tile_m.ptr.cast(),
                    n_exp_par as i32, n_embd as i32, n_ff as i32, 1.0, true, 3, &stream,
                );
            }
            stream.sync().unwrap();
            let mut h_out_ref = vec![0.0f32; m_tot_par * n_embd];
            let mut h_out_grp = vec![0.0f32; m_tot_par * n_embd];
            d_out_ref.copy_to_host(bytemuck::cast_slice_mut(&mut h_out_ref)).unwrap();
            d_out_grp.copy_to_host(bytemuck::cast_slice_mut(&mut h_out_grp)).unwrap();
            assert!(h_out_grp.iter().all(|v| v.is_finite()), "NaN/Inf in grouped BM=32 ({dn_ty:?})");
            let re = rel_err(&h_out_grp, &h_out_ref);
            println!("Stage C.9 gemm_mmq_mma_grouped ({dn_ty:?}, tile_mode=3 bm=32): rel_err={re:.7}");
            assert!(re < 3e-4, "grouped MoE tile_mode=3 parity failed ({dn_ty:?}): rel_err={re}");
        }
    }
}





