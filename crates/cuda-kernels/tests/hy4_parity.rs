use ampllm_cuda_kernels::ffi::*;
use ampllm_cuda_kernels::{hy4, CudaStream, DeviceBuffer};
use ampllm_gguf::{dequantize_row_to_f32, vec_dot_q_f32_ref, GgmlType};
use half::{bf16, f16};

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

fn rel_err(got: &[f32], exp: &[f32]) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (g, e) in got.iter().zip(exp.iter()) {
        let d = (*g - *e) as f64;
        num += d * d;
        den += (*e as f64) * (*e as f64);
    }
    (num.sqrt() / den.sqrt().max(1e-12)) as f32
}

fn max_abs_err(got: &[f32], exp: &[f32]) -> f32 {
    got.iter()
        .zip(exp.iter())
        .map(|(g, e)| (*g - *e).abs())
        .fold(0.0f32, f32::max)
}

fn fill_quant_row(ty: GgmlType, k: usize, rng: &mut Rng) -> Vec<u8> {
    let nb = k / ty.block_size();
    let bsz = ty.type_size();
    let mut out = vec![0u8; nb * bsz];
    for b in 0..nb {
        let blk = &mut out[b * bsz..(b + 1) * bsz];
        match ty {
            GgmlType::Q8_0 => {
                blk[0..2].copy_from_slice(&f16::from_f32(0.015 + b as f32 * 0.0002).to_le_bytes());
                for x in &mut blk[2..34] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Q5K => {
                blk[0..2].copy_from_slice(&f16::from_f32(0.010).to_le_bytes());
                blk[2..4].copy_from_slice(&f16::from_f32(0.007).to_le_bytes());
                for x in &mut blk[4..176] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Iq4Xs => {
                blk[0..2].copy_from_slice(&f16::from_f32(0.006).to_le_bytes());
                for x in &mut blk[2..136] {
                    *x = rng.next_u8();
                }
            }
            _ => unreachable!(),
        }
    }
    out
}

fn swiglu_clamp_ref(g: f32, u: f32, lim: f32) -> f32 {
    let gc = g.min(lim);
    let uc = u.clamp(-lim, lim);
    (gc / (1.0 + (-gc).exp())) * uc
}

fn f8e4m3_to_f32_ref(x: u8) -> f32 {
    let ax = x & 0x7f;
    if ax == 0x7f {
        return f32::NAN;
    }
    let e = ((ax >> 3) & 0xf) as i32;
    let m = (ax & 7) as f32;
    let v = if e == 0 {
        m * (-9.0f32).exp2()
    } else {
        (1.0 + m / 8.0) * ((e - 7) as f32).exp2()
    };
    if (x & 0x80) != 0 { -v } else { v }
}

fn rope_vec_ref(v: &mut [f32], pos: i32, n_rot: usize, base: f32) {
    let theta_scale = base.powf(-2.0 / n_rot as f32);
    for p in 0..(n_rot / 2) {
        let theta = (pos as f32) * theta_scale.powi(p as i32);
        let (s, c) = theta.sin_cos();
        let a = v[2 * p];
        let b = v[2 * p + 1];
        v[2 * p] = a * c - b * s;
        v[2 * p + 1] = a * s + b * c;
    }
}

#[test]
fn test_hy4_quants_linear_mmq_moe_dequant() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    // 1. Dequant to F16 for Bf16, Q5K, Iq4Xs
    {
        let k = 512usize;
        let n = 8usize;
        let bf16_vals: Vec<u16> = (0..n * k).map(|_| bf16::from_f32(rng.next_f32()).to_bits()).collect();
        let d_src = DeviceBuffer::alloc(0, bf16_vals.len() * 2).unwrap();
        d_src.copy_from_host(bytemuck::cast_slice(&bf16_vals)).unwrap();
        let d_dst = DeviceBuffer::alloc(0, n * k * 2).unwrap();
        unsafe {
            launch_dequant_to_f16(
                GgmlType::Bf16 as i32,
                d_src.ptr,
                d_dst.ptr,
                (n * k) as i64,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let mut got_u16 = vec![0u16; n * k];
        d_dst.copy_to_host(bytemuck::cast_slice_mut(&mut got_u16)).unwrap();
        let got: Vec<f32> = got_u16.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
        let exp: Vec<f32> = bf16_vals.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        assert!(max_abs_err(&got, &exp) < 2e-3, "Bf16 dequant mismatch");

        for ty in [GgmlType::Q5K, GgmlType::Iq4Xs] {
            let rb = ty.row_bytes(k);
            let mut raw = Vec::with_capacity(n * rb);
            for _ in 0..n {
                raw.extend_from_slice(&fill_quant_row(ty, k, &mut rng));
            }
            let d_q = DeviceBuffer::alloc(0, raw.len()).unwrap();
            d_q.copy_from_host(&raw).unwrap();
            unsafe {
                launch_dequant_to_f16(ty as i32, d_q.ptr, d_dst.ptr, (n * k) as i64, stream.raw);
            }
            stream.sync().unwrap();
            d_dst.copy_to_host(bytemuck::cast_slice_mut(&mut got_u16)).unwrap();
            let got: Vec<f32> = got_u16.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
            let mut exp = vec![0.0f32; n * k];
            for r in 0..n {
                dequantize_row_to_f32(ty, &raw[r * rb..(r + 1) * rb], &mut exp[r * k..(r + 1) * k]);
            }
            let re = rel_err(&got, &exp);
            println!("dequant_to_f16 {ty:?}: rel_err={re:.6}");
            assert!(re < 2e-3, "dequant_to_f16 {ty:?} failed: {re}");
        }
    }

    // 2. MMQ MMA Tiled & Grouped on Q5K and Iq4Xs
    {
        let m = 40usize;
        let k = 512usize;
        let n = 128usize;
        let x: Vec<f32> = (0..m * k).map(|_| rng.next_f32()).collect();
        let d_x = DeviceBuffer::alloc(0, m * k * 4).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&x)).unwrap();
        let d_xq8 = DeviceBuffer::alloc(0, m * (k / 32) * 36).unwrap();
        unsafe {
            launch_quantize_q8_1(d_x.ptr as *const f32, d_xq8.ptr, m as i32, k as i32, stream.raw);
        }
        let d_y = DeviceBuffer::alloc(0, m * n * 4).unwrap();

        for ty in [GgmlType::Q5K, GgmlType::Iq4Xs] {
            let rb = ty.row_bytes(k);
            let mut w = Vec::with_capacity(n * rb);
            for _ in 0..n {
                w.extend_from_slice(&fill_quant_row(ty, k, &mut rng));
            }
            let d_w = DeviceBuffer::alloc(0, w.len()).unwrap();
            d_w.copy_from_host(&w).unwrap();
            let mut exp = vec![0.0f32; m * n];
            for mi in 0..m {
                for ni in 0..n {
                    exp[mi * n + ni] = vec_dot_q_f32_ref(ty, &w[ni * rb..(ni + 1) * rb], &x[mi * k..(mi + 1) * k]);
                }
            }
            for tile_mode in [0i32, 2, 3] {
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
                        m as i32,
                        n as i32,
                        k as i32,
                        1.0,
                        0.0,
                        tile_mode,
                        stream.raw,
                    );
                }
                stream.sync().unwrap();
                let mut got = vec![0.0f32; m * n];
                d_y.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
                let re = rel_err(&got, &exp);
                println!("mmq_mma_tiled {ty:?} tile={tile_mode}: rel_err={re:.6}");
                assert!(re < 0.008, "mmq_mma_tiled {ty:?} tile={tile_mode} failed: {re}");
            }

            // Grouped MMQ over 2 experts: exp 0 gets first 12 rows, exp 1 gets next 28 rows
            let mut w1 = Vec::with_capacity(n * rb);
            for _ in 0..n {
                w1.extend_from_slice(&fill_quant_row(ty, k, &mut rng));
            }
            let d_w1 = DeviceBuffer::alloc(0, w1.len()).unwrap();
            d_w1.copy_from_host(&w1).unwrap();
            let ptrs = [d_w.as_u64(), d_w1.as_u64()];
            let exp_row_off = [0i32, 12i32];
            let exp_nt = [12i32, 28i32];
            let tile_exp_id = [0i32, 1i32];
            let tile_m_off = [0i32, 0i32];
            let d_ptrs = DeviceBuffer::alloc(0, ptrs.len() * 8).unwrap();
            let d_off = DeviceBuffer::alloc(0, 8).unwrap();
            let d_nt = DeviceBuffer::alloc(0, 8).unwrap();
            let d_te = DeviceBuffer::alloc(0, 8).unwrap();
            let d_tm = DeviceBuffer::alloc(0, 8).unwrap();
            d_ptrs.copy_from_host(bytemuck::cast_slice(&ptrs)).unwrap();
            d_off.copy_from_host(bytemuck::cast_slice(&exp_row_off)).unwrap();
            d_nt.copy_from_host(bytemuck::cast_slice(&exp_nt)).unwrap();
            d_te.copy_from_host(bytemuck::cast_slice(&tile_exp_id)).unwrap();
            d_tm.copy_from_host(bytemuck::cast_slice(&tile_m_off)).unwrap();

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
            let mut got_grp = vec![0.0f32; m * n];
            d_y.copy_to_host(bytemuck::cast_slice_mut(&mut got_grp)).unwrap();
            let mut exp_grp = vec![0.0f32; m * n];
            for mi in 0..m {
                let wr = if mi < 12 { &w } else { &w1 };
                for ni in 0..n {
                    exp_grp[mi * n + ni] = vec_dot_q_f32_ref(ty, &wr[ni * rb..(ni + 1) * rb], &x[mi * k..(mi + 1) * k]);
                }
            }
            let re_grp = rel_err(&got_grp, &exp_grp);
            println!("mmq_mma_grouped {ty:?}: rel_err={re_grp:.6}");
            assert!(re_grp < 0.008, "mmq_mma_grouped {ty:?} failed: {re_grp}");
        }
    }

    // 3. MoE Plan Filtered with topk=16 (Q5K & Iq4Xs & Q8_0) + Shared Q8_0 Down Fusion + Prefill Small
    for ty in [GgmlType::Q5K, GgmlType::Iq4Xs, GgmlType::Q8_0] {
        let m = 2usize;
        let topk = 16usize;
        let n_embd = 512usize;
        let n_ff = 256usize;
        let n_ff_sh = 256usize;
        let clamp_lim = 10.0f32;

        let rb_gu = ty.row_bytes(n_embd);
        let rb_d = ty.row_bytes(n_ff);
        let u_off = n_ff * rb_gu;
        let d_off = 2 * n_ff * rb_gu;
        let exp_bytes = d_off + n_embd * rb_d;

        // 4 distinct experts rotated across 16 slots per token
        let mut blobs = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut b = Vec::with_capacity(exp_bytes);
            for _ in 0..(2 * n_ff) {
                b.extend_from_slice(&fill_quant_row(ty, n_embd, &mut rng));
            }
            for _ in 0..n_embd {
                b.extend_from_slice(&fill_quant_row(ty, n_ff, &mut rng));
            }
            assert_eq!(b.len(), exp_bytes);
            blobs.push(b);
        }
        let d_blobs: Vec<DeviceBuffer> = blobs
            .iter()
            .map(|b| {
                let db = DeviceBuffer::alloc(0, b.len()).unwrap();
                db.copy_from_host(b).unwrap();
                db
            })
            .collect();

        let mut plan_ptrs = vec![0u64; m * topk];
        let mut router_w = vec![0.0f32; m * topk];
        for mi in 0..m {
            for s in 0..topk {
                plan_ptrs[mi * topk + s] = d_blobs[(mi + s) % 4].as_u64();
                router_w[mi * topk + s] = 0.05 + (s as f32) * 0.01;
            }
        }
        let d_plan = DeviceBuffer::alloc(0, plan_ptrs.len() * 8).unwrap();
        let d_rw = DeviceBuffer::alloc(0, router_w.len() * 4).unwrap();
        d_plan.copy_from_host(bytemuck::cast_slice(&plan_ptrs)).unwrap();
        d_rw.copy_from_host(bytemuck::cast_slice(&router_w)).unwrap();

        let x: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32() * 0.5).collect();
        let d_x = DeviceBuffer::alloc(0, x.len() * 4).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&x)).unwrap();
        let d_xq8 = DeviceBuffer::alloc(0, m * (n_embd / 32) * 36).unwrap();
        let d_iq8 = DeviceBuffer::alloc(0, m * topk * (n_ff / 32) * 36).unwrap();

        // Shared Q8_0 down expert
        let sh_rb = GgmlType::Q8_0.row_bytes(n_ff_sh);
        let mut sh_w = Vec::with_capacity(n_embd * sh_rb);
        for _ in 0..n_embd {
            sh_w.extend_from_slice(&fill_quant_row(GgmlType::Q8_0, n_ff_sh, &mut rng));
        }
        let d_sh_w = DeviceBuffer::alloc(0, sh_w.len()).unwrap();
        d_sh_w.copy_from_host(&sh_w).unwrap();
        let sh_h: Vec<f32> = (0..m * n_ff_sh).map(|_| rng.next_f32() * 0.5).collect();
        let d_sh_h = DeviceBuffer::alloc(0, sh_h.len() * 4).unwrap();
        d_sh_h.copy_from_host(bytemuck::cast_slice(&sh_h)).unwrap();
        let d_sh_hq = DeviceBuffer::alloc(0, m * (n_ff_sh / 32) * 36).unwrap();

        let d_sh_out = DeviceBuffer::alloc(0, m * n_embd * 4).unwrap();
        let d_out = DeviceBuffer::alloc(0, m * n_embd * 4).unwrap();

        unsafe {
            launch_quantize_q8_1(d_x.ptr as *const f32, d_xq8.ptr, m as i32, n_embd as i32, stream.raw);
            launch_quantize_q8_1(d_sh_h.ptr as *const f32, d_sh_hq.ptr, m as i32, n_ff_sh as i32, stream.raw);
            launch_glm5_moe_plan_filtered(
                ty as i32,
                ty as i32,
                d_plan.ptr as *const u64,
                u_off as u32,
                d_off as u32,
                rb_gu,
                rb_d,
                d_xq8.ptr,
                d_rw.ptr as *const f32,
                d_iq8.ptr,
                GgmlType::Q8_0 as i32,
                d_sh_w.ptr,
                sh_rb,
                d_sh_hq.ptr,
                n_ff_sh as i32,
                d_sh_out.ptr as *mut f32,
                d_out.ptr as *mut f32,
                n_embd as i32,
                n_ff as i32,
                topk as i32,
                m as i32,
                clamp_lim,
                std::ptr::null(),
                std::ptr::null(),
                0,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let mut got_out = vec![0.0f32; m * n_embd];
        d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_out)).unwrap();

        let mut exp_out = vec![0.0f32; m * n_embd];
        for mi in 0..m {
            let xr = &x[mi * n_embd..(mi + 1) * n_embd];
            for r in 0..n_embd {
                exp_out[mi * n_embd + r] += vec_dot_q_f32_ref(
                    GgmlType::Q8_0,
                    &sh_w[r * sh_rb..(r + 1) * sh_rb],
                    &sh_h[mi * n_ff_sh..(mi + 1) * n_ff_sh],
                );
            }
            for s in 0..topk {
                let blob = &blobs[(mi + s) % 4];
                let mut h = vec![0.0f32; n_ff];
                for r in 0..n_ff {
                    let g = vec_dot_q_f32_ref(ty, &blob[r * rb_gu..(r + 1) * rb_gu], xr);
                    let u = vec_dot_q_f32_ref(ty, &blob[u_off + r * rb_gu..u_off + (r + 1) * rb_gu], xr);
                    h[r] = swiglu_clamp_ref(g, u, clamp_lim);
                }
                let rw = router_w[mi * topk + s];
                for r in 0..n_embd {
                    let dv = vec_dot_q_f32_ref(ty, &blob[d_off + r * rb_d..d_off + (r + 1) * rb_d], &h);
                    exp_out[mi * n_embd + r] += rw * dv;
                }
            }
        }
        let re = rel_err(&got_out, &exp_out);
        println!("moe_plan_filtered topk=16 {ty:?}: rel_err={re:.6}");

        // Compare routed-only plan_filtered vs moe_prefill_small (both use identical Q8_1 pipeline)
        d_out.zero().unwrap();
        unsafe {
            launch_glm5_moe_plan_filtered(
                ty as i32,
                ty as i32,
                d_plan.ptr as *const u64,
                u_off as u32,
                d_off as u32,
                rb_gu,
                rb_d,
                d_xq8.ptr,
                d_rw.ptr as *const f32,
                d_iq8.ptr,
                0,
                std::ptr::null(),
                0,
                std::ptr::null(),
                0,
                std::ptr::null_mut(),
                d_out.ptr as *mut f32,
                n_embd as i32,
                n_ff as i32,
                topk as i32,
                m as i32,
                clamp_lim,
                std::ptr::null(),
                std::ptr::null(),
                0,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let mut got_routed_plan = vec![0.0f32; m * n_embd];
        d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_routed_plan)).unwrap();

        let mut exp_routed = vec![0.0f32; m * n_embd];
        for mi in 0..m {
            let xr = &x[mi * n_embd..(mi + 1) * n_embd];
            for s in 0..topk {
                let blob = &blobs[(mi + s) % 4];
                let mut h = vec![0.0f32; n_ff];
                for r in 0..n_ff {
                    let g = vec_dot_q_f32_ref(ty, &blob[r * rb_gu..(r + 1) * rb_gu], xr);
                    let u = vec_dot_q_f32_ref(ty, &blob[u_off + r * rb_gu..u_off + (r + 1) * rb_gu], xr);
                    h[r] = swiglu_clamp_ref(g, u, clamp_lim);
                }
                let rw = router_w[mi * topk + s];
                for r in 0..n_embd {
                    let dv = vec_dot_q_f32_ref(ty, &blob[d_off + r * rb_d..d_off + (r + 1) * rb_d], &h);
                    exp_routed[mi * n_embd + r] += rw * dv;
                }
            }
        }
        let re_routed = rel_err(&got_routed_plan, &exp_routed);
        println!("moe_plan_filtered routed-only topk=16 {ty:?}: rel_err={re_routed:.6}");

        // Prefill small_gu & small_down over all m * topk pairs using the same Q8_1 pipeline
        let mut job_ptrs = Vec::with_capacity(m * topk);
        let mut slot_off = Vec::with_capacity(m * topk);
        let mut slot_nt = Vec::with_capacity(m * topk);
        let mut rows_all = Vec::with_capacity(m * topk);
        let mut wts_all = Vec::with_capacity(m * topk);
        for mi in 0..m {
            for s in 0..topk {
                job_ptrs.push(d_blobs[(mi + s) % 4].as_u64());
                slot_off.push((mi * topk + s) as i32);
                slot_nt.push(1i32);
                rows_all.push(mi as i32);
                wts_all.push(router_w[mi * topk + s]);
            }
        }
        let d_jptrs = DeviceBuffer::alloc(0, job_ptrs.len() * 8).unwrap();
        let d_soff = DeviceBuffer::alloc(0, slot_off.len() * 4).unwrap();
        let d_snt = DeviceBuffer::alloc(0, slot_nt.len() * 4).unwrap();
        let d_rows = DeviceBuffer::alloc(0, rows_all.len() * 4).unwrap();
        let d_wts = DeviceBuffer::alloc(0, wts_all.len() * 4).unwrap();
        d_jptrs.copy_from_host(bytemuck::cast_slice(&job_ptrs)).unwrap();
        d_soff.copy_from_host(bytemuck::cast_slice(&slot_off)).unwrap();
        d_snt.copy_from_host(bytemuck::cast_slice(&slot_nt)).unwrap();
        d_rows.copy_from_host(bytemuck::cast_slice(&rows_all)).unwrap();
        d_wts.copy_from_host(bytemuck::cast_slice(&wts_all)).unwrap();
        d_out.zero().unwrap();
        unsafe {
            launch_glm5_moe_prefill_small_gu(
                ty as i32,
                d_jptrs.ptr as *const u64,
                d_soff.ptr as *const i32,
                d_snt.ptr as *const i32,
                d_rows.ptr as *const i32,
                u_off as u32,
                rb_gu,
                d_xq8.ptr,
                d_iq8.ptr,
                n_embd as i32,
                n_ff as i32,
                (m * topk) as i32,
                clamp_lim,
                stream.raw,
            );
            launch_glm5_moe_prefill_small_down(
                ty as i32,
                d_jptrs.ptr as *const u64,
                d_soff.ptr as *const i32,
                d_snt.ptr as *const i32,
                d_rows.ptr as *const i32,
                d_wts.ptr as *const f32,
                d_off as u32,
                rb_d,
                d_iq8.ptr,
                d_out.ptr as *mut f32,
                n_embd as i32,
                n_ff as i32,
                (m * topk) as i32,
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let mut got_sm = vec![0.0f32; m * n_embd];
        d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_sm)).unwrap();
        let re_plan_vs_sm = rel_err(&got_routed_plan, &got_sm);
        let re_sm = rel_err(&got_sm, &exp_routed);
        println!("moe_prefill_small {ty:?}: rel_vs_fp32={re_sm:.6}, plan_vs_prefill_small={re_plan_vs_sm:.6}");
        assert!(re_plan_vs_sm < 1e-4, "plan_filtered vs prefill_small mismatch for {ty:?}: {re_plan_vs_sm}");
        assert!(re < 0.02 && re_sm < 0.02, "moe two-stage Q8_1 error too high for {ty:?}: {re} / {re_sm}");
    }
}

#[test]
fn test_hy4_ihc4_rope_and_engram() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x4152_4d39_1234_5678);

    // 1. expand_streams + hc_pre (by_el=true/false, head=true/false) + hc_post_norm
    let t = 3usize;
    let hc = 4usize;
    let e = 256usize;
    let eps = 1e-6f32;
    let hc_eps = 1e-6f32;
    let mag = 2.0f32;
    let neps = 1e-5f32;

    let emb: Vec<f32> = (0..t * e).map(|_| rng.next_f32()).collect();
    let d_emb = DeviceBuffer::alloc(0, emb.len() * 4).unwrap();
    let d_x = DeviceBuffer::alloc(0, t * hc * e * 4).unwrap();
    d_emb.copy_from_host(bytemuck::cast_slice(&emb)).unwrap();
    unsafe {
        hy4::expand_streams(d_emb.ptr as *const f32, t as i32, hc as i32, e as i32, d_x.ptr as *mut f32, &stream);
    }
    stream.sync().unwrap();
    let mut x_host = vec![0.0f32; t * hc * e];
    d_x.copy_to_host(bytemuck::cast_slice_mut(&mut x_host)).unwrap();
    for ti in 0..t {
        for s in 0..hc {
            assert!(max_abs_err(&x_host[(ti * hc + s) * e..(ti * hc + s + 1) * e], &emb[ti * e..(ti + 1) * e]) < 1e-6);
        }
    }
    // Give streams distinct values for HC test
    for v in &mut x_host {
        *v = rng.next_f32();
    }
    d_x.copy_from_host(bytemuck::cast_slice(&x_host)).unwrap();

    let part_floats = hy4::hc_part_floats(t);
    let d_part = DeviceBuffer::alloc(0, part_floats * 4).unwrap();
    let d_cur = DeviceBuffer::alloc(0, t * e * 4).unwrap();
    let d_post = DeviceBuffer::alloc(0, t * hc * 4).unwrap();

    for fn_by_el in [true, false] {
        for is_head in [false, true] {
            let nm = if is_head { hc } else { 2 * hc };
            let fn_w: Vec<f32> = (0..hc * e * nm).map(|_| rng.next_f32() * 0.25).collect();
            let scale = [1.1f32, 0.9f32];
            let base: Vec<f32> = (0..nm).map(|i| 0.05 * i as f32 - 0.1).collect();
            let nw: Vec<f32> = (0..e).map(|_| 0.8 + 0.3 * rng.next_f32().abs()).collect();

            let d_fn = DeviceBuffer::alloc(0, fn_w.len() * 4).unwrap();
            let d_sc = DeviceBuffer::alloc(0, 8).unwrap();
            let d_bs = DeviceBuffer::alloc(0, base.len() * 4).unwrap();
            let d_nw = DeviceBuffer::alloc(0, nw.len() * 4).unwrap();
            d_fn.copy_from_host(bytemuck::cast_slice(&fn_w)).unwrap();
            d_sc.copy_from_host(bytemuck::cast_slice(&scale)).unwrap();
            d_bs.copy_from_host(bytemuck::cast_slice(&base)).unwrap();
            d_nw.copy_from_host(bytemuck::cast_slice(&nw)).unwrap();

            unsafe {
                hy4::hc_pre(
                    d_x.ptr as *const f32,
                    t as i32,
                    hc as i32,
                    e as i32,
                    d_fn.ptr as *const f32,
                    fn_by_el,
                    d_sc.ptr as *const f32,
                    d_bs.ptr as *const f32,
                    eps,
                    hc_eps,
                    mag,
                    if is_head { std::ptr::null() } else { d_nw.ptr as *const f32 },
                    neps,
                    is_head,
                    d_part.ptr as *mut f32,
                    d_cur.ptr as *mut f32,
                    if is_head { std::ptr::null_mut() } else { d_post.ptr as *mut f32 },
                    &stream,
                );
            }
            stream.sync().unwrap();
            let mut got_cur = vec![0.0f32; t * e];
            let mut got_post = vec![0.0f32; t * hc];
            d_cur.copy_to_host(bytemuck::cast_slice_mut(&mut got_cur)).unwrap();
            if !is_head {
                d_post.copy_to_host(bytemuck::cast_slice_mut(&mut got_post)).unwrap();
            }

            let mut exp_cur = vec![0.0f32; t * e];
            let mut exp_post = vec![0.0f32; t * hc];
            let he = hc * e;
            for ti in 0..t {
                let xt = &x_host[ti * he..(ti + 1) * he];
                let ss: f32 = xt.iter().map(|v| v * v).sum();
                let r = 1.0 / (ss / he as f32 + eps).sqrt();
                let mut mix = vec![0.0f32; nm];
                for j in 0..nm {
                    let mut dot = 0.0f32;
                    for i in 0..he {
                        let wv = if fn_by_el { fn_w[i * nm + j] } else { fn_w[j * he + i] };
                        dot += xt[i] * wv;
                    }
                    mix[j] = dot * r;
                }
                let mut pre = vec![0.0f32; hc];
                for j in 0..hc {
                    pre[j] = 1.0 / (1.0 + (-(mix[j] * scale[0] + base[j])).exp()) + hc_eps;
                    if !is_head {
                        exp_post[ti * hc + j] =
                            1.0 / (1.0 + (-(mix[hc + j] * scale[1] + base[hc + j])).exp()) * mag + hc_eps;
                    }
                }
                let mut tmp = vec![0.0f32; e];
                for d in 0..e {
                    let mut v = 0.0f32;
                    for j in 0..hc {
                        v += pre[j] * xt[j * e + d];
                    }
                    tmp[d] = v;
                }
                if !is_head {
                    let tss: f32 = tmp.iter().map(|v| v * v).sum();
                    let rn = 1.0 / (tss / e as f32 + neps).sqrt();
                    for d in 0..e {
                        exp_cur[ti * e + d] = tmp[d] * rn * nw[d];
                    }
                } else {
                    exp_cur[ti * e..(ti + 1) * e].copy_from_slice(&tmp);
                }
            }
            assert!(max_abs_err(&got_cur, &exp_cur) < 1e-4, "hc_pre cur mismatch");
            if !is_head {
                assert!(max_abs_err(&got_post, &exp_post) < 1e-4, "hc_pre post mismatch");
            }
        }
    }

    // hc_post_norm (with norm weight)
    {
        let y: Vec<f32> = (0..t * e).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..e).map(|_| 0.7 + 0.5 * rng.next_f32().abs()).collect();
        let post: Vec<f32> = (0..t * hc).map(|_| 0.3 + 0.5 * rng.next_f32().abs()).collect();
        let mut x_ref = x_host.clone();
        let d_y = DeviceBuffer::alloc(0, y.len() * 4).unwrap();
        let d_w = DeviceBuffer::alloc(0, w.len() * 4).unwrap();
        let d_p = DeviceBuffer::alloc(0, post.len() * 4).unwrap();
        let d_yo = DeviceBuffer::alloc(0, y.len() * 4).unwrap();
        d_y.copy_from_host(bytemuck::cast_slice(&y)).unwrap();
        d_w.copy_from_host(bytemuck::cast_slice(&w)).unwrap();
        d_p.copy_from_host(bytemuck::cast_slice(&post)).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&x_host)).unwrap();
        unsafe {
            hy4::hc_post_norm(
                d_x.ptr as *mut f32,
                t as i32,
                hc as i32,
                e as i32,
                d_y.ptr as *const f32,
                d_p.ptr as *const f32,
                d_w.ptr as *const f32,
                neps,
                d_yo.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got_x = vec![0.0f32; x_ref.len()];
        let mut got_yo = vec![0.0f32; y.len()];
        d_x.copy_to_host(bytemuck::cast_slice_mut(&mut got_x)).unwrap();
        d_yo.copy_to_host(bytemuck::cast_slice_mut(&mut got_yo)).unwrap();
        let mut exp_yo = vec![0.0f32; y.len()];
        for ti in 0..t {
            let ss: f32 = y[ti * e..(ti + 1) * e].iter().map(|v| v * v).sum();
            let r = 1.0 / (ss / e as f32 + neps).sqrt();
            for d in 0..e {
                let yn = y[ti * e + d] * r * w[d];
                exp_yo[ti * e + d] = yn;
                for s in 0..hc {
                    x_ref[(ti * hc + s) * e + d] += yn * post[ti * hc + s];
                }
            }
        }
        assert!(max_abs_err(&got_yo, &exp_yo) < 1e-5, "hc_post_norm y_norm mismatch");
        assert!(max_abs_err(&got_x, &x_ref) < 1e-5, "hc_post_norm x mismatch");
    }

    // 2. RoPE test
    {
        let nh = 4usize;
        let ld_h = 128usize;
        let ld_t = nh * ld_h;
        let off = 64usize;
        let n_rot = 64usize;
        let pos = [5i32, 17i32, 0i32];
        let mut x = vec![0.0f32; t * ld_t];
        for v in &mut x {
            *v = rng.next_f32();
        }
        let mut exp = x.clone();
        for ti in 0..t {
            for hi in 0..nh {
                let s = ti * ld_t + hi * ld_h + off;
                rope_vec_ref(&mut exp[s..s + n_rot], pos[ti], n_rot, 10000.0);
            }
        }
        let d_rx = DeviceBuffer::alloc(0, x.len() * 4).unwrap();
        let d_pos = DeviceBuffer::alloc(0, pos.len() * 4).unwrap();
        d_rx.copy_from_host(bytemuck::cast_slice(&x)).unwrap();
        d_pos.copy_from_host(bytemuck::cast_slice(&pos)).unwrap();
        unsafe {
            hy4::rope(
                d_rx.ptr as *mut f32,
                t as i32,
                nh as i32,
                ld_t,
                ld_h,
                off as i32,
                n_rot as i32,
                d_pos.ptr as *const i32,
                10000.0,
                &stream,
            );
        }
        stream.sync().unwrap();
        d_rx.copy_to_host(bytemuck::cast_slice_mut(&mut x)).unwrap();
        assert!(max_abs_err(&x, &exp) < 1e-5, "hy4::rope mismatch");
    }

    // 3. Engram rows + Engram mix
    {
        let cols = 16usize;
        let dim = 96usize;
        let valid = [1.0f32, 0.0f32, 1.0f32];
        let mut rows = vec![0u8; t * cols * dim];
        for (i, b) in rows.iter_mut().enumerate() {
            // Avoid 0x7f / 0xff (NaN) while covering subnormals and negatives
            let mut v = ((i * 37 + 5) & 0xff) as u8;
            if (v & 0x7f) == 0x7f {
                v = 0x7e;
            }
            *b = v;
        }
        let scale: Vec<f32> = (0..t * cols).map(|_| 0.8 + 0.3 * rng.next_f32()).collect();
        let ln: Vec<f32> = (0..cols * dim).map(|_| 0.5 + 0.5 * rng.next_f32()).collect();
        let d_rows = DeviceBuffer::alloc(0, rows.len()).unwrap();
        let d_valid = DeviceBuffer::alloc(0, valid.len() * 4).unwrap();
        let d_scale = DeviceBuffer::alloc(0, scale.len() * 4).unwrap();
        let d_ln = DeviceBuffer::alloc(0, ln.len() * 4).unwrap();
        let d_out = DeviceBuffer::alloc(0, t * cols * dim * 4).unwrap();
        d_rows.copy_from_host(&rows).unwrap();
        d_valid.copy_from_host(bytemuck::cast_slice(&valid)).unwrap();
        d_scale.copy_from_host(bytemuck::cast_slice(&scale)).unwrap();
        d_ln.copy_from_host(bytemuck::cast_slice(&ln)).unwrap();
        unsafe {
            hy4::engram_rows(
                d_rows.ptr as *const u8,
                d_valid.ptr as *const f32,
                t as i32,
                cols as i32,
                dim as i32,
                d_ln.ptr as *const f32,
                1e-5,
                d_scale.ptr as *const f32,
                d_out.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got_r = vec![0.0f32; t * cols * dim];
        d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_r)).unwrap();
        let mut exp_r = vec![0.0f32; t * cols * dim];
        for ti in 0..t {
            for col in 0..cols {
                let base_i = (ti * cols + col) * dim;
                let sc = scale[ti * cols + col];
                let ss: f32 = (0..dim)
                    .map(|i| {
                        let v = f8e4m3_to_f32_ref(rows[base_i + i]) * sc;
                        v * v
                    })
                    .sum();
                let rr = 1.0 / (ss / dim as f32 + 1e-5).sqrt();
                for i in 0..dim {
                    let v = f8e4m3_to_f32_ref(rows[base_i + i]) * sc;
                    exp_r[base_i + i] = v * rr * ln[col * dim + i] * valid[ti];
                }
            }
        }
        assert!(max_abs_err(&got_r, &exp_r) < 1e-5, "engram_rows mismatch");

        // Engram mix
        let kv: Vec<f32> = (0..t * (hc + 1) * e).map(|_| rng.next_f32()).collect();
        let kv_b: Vec<f32> = (0..(hc + 1) * e).map(|_| rng.next_f32() * 0.2).collect();
        let egate: Vec<f32> = (0..hc * e).map(|_| rng.next_f32()).collect();
        let gamma: Vec<f32> = (0..e).map(|_| 0.7 + rng.next_f32() * 0.2).collect();
        let hcs: Vec<f32> = (0..hc * e).map(|_| rng.next_f32() * 0.3).collect();
        let mut x_mix = x_host.clone();

        let d_kv = DeviceBuffer::alloc(0, kv.len() * 4).unwrap();
        let d_kvb = DeviceBuffer::alloc(0, kv_b.len() * 4).unwrap();
        let d_eg = DeviceBuffer::alloc(0, egate.len() * 4).unwrap();
        let d_gm = DeviceBuffer::alloc(0, gamma.len() * 4).unwrap();
        let d_hcs = DeviceBuffer::alloc(0, hcs.len() * 4).unwrap();
        d_kv.copy_from_host(bytemuck::cast_slice(&kv)).unwrap();
        d_kvb.copy_from_host(bytemuck::cast_slice(&kv_b)).unwrap();
        d_eg.copy_from_host(bytemuck::cast_slice(&egate)).unwrap();
        d_gm.copy_from_host(bytemuck::cast_slice(&gamma)).unwrap();
        d_hcs.copy_from_host(bytemuck::cast_slice(&hcs)).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&x_mix)).unwrap();

        unsafe {
            hy4::engram_mix(
                d_x.ptr as *mut f32,
                d_kv.ptr as *const f32,
                d_kvb.ptr as *const f32,
                t as i32,
                hc as i32,
                e as i32,
                d_eg.ptr as *const f32,
                d_gm.ptr as *const f32,
                d_hcs.ptr as *const f32,
                d_valid.ptr as *const f32,
                1e-5,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got_mix = vec![0.0f32; x_mix.len()];
        d_x.copy_to_host(bytemuck::cast_slice_mut(&mut got_mix)).unwrap();
        for ti in 0..t {
            if valid[ti] == 0.0 {
                continue;
            }
            let kt = &kv[ti * (hc + 1) * e..];
            let mut gates = vec![0.0f32; hc];
            for s in 0..hc {
                let (mut sx, mut sk, mut sd) = (0.0f32, 0.0f32, 0.0f32);
                for d in 0..e {
                    let xv = x_mix[(ti * hc + s) * e + d];
                    let kvv = kt[s * e + d] + kv_b[s * e + d];
                    sx += xv * xv;
                    sk += kvv * kvv;
                    sd += xv * kvv * egate[s * e + d];
                }
                let rx = 1.0 / (sx / e as f32 + 1e-5).sqrt();
                let rk = 1.0 / (sk / e as f32 + 1e-5).sqrt();
                let dot = sd * rx * rk / (e as f32).sqrt();
                let mag = dot.abs().max(1e-6).sqrt();
                gates[s] = 1.0 / (1.0 + (-(if dot < 0.0 { -mag } else { mag })).exp());
            }
            let mut sv = 0.0f32;
            for d in 0..e {
                let v = kt[hc * e + d] + kv_b[hc * e + d];
                sv += v * v;
            }
            let rv = 1.0 / (sv / e as f32 + 1e-5).sqrt();
            for d in 0..e {
                let sig = (kt[hc * e + d] + kv_b[hc * e + d]) * rv * gamma[d];
                for s in 0..hc {
                    x_mix[(ti * hc + s) * e + d] += sig * (gates[s] + hcs[s * e + d]);
                }
            }
        }
        assert!(max_abs_err(&got_mix, &x_mix) < 1e-5, "engram_mix mismatch");
    }
}

#[test]
fn test_hy4_mla_kv_indexer_and_heads() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x7a2b_3c4d_5e6f_7081);

    // 1. batched_heads_q8 (simulating q_nope[T, H, 256] -> K=192 @ W[H, 512, 192])
    {
        let t = 3usize;
        let h = 8usize;
        let n = 64usize;
        let k = 192usize;
        let ld_h = 256usize;
        let ld_t = h * ld_h;
        let mut x = vec![0.0f32; t * ld_t];
        for v in &mut x {
            *v = rng.next_f32();
        }
        let rb = GgmlType::Q8_0.row_bytes(k);
        let mut w = Vec::with_capacity(h * n * rb);
        for _ in 0..(h * n) {
            w.extend_from_slice(&fill_quant_row(GgmlType::Q8_0, k, &mut rng));
        }
        let d_x = DeviceBuffer::alloc(0, x.len() * 4).unwrap();
        let d_w = DeviceBuffer::alloc(0, w.len()).unwrap();
        let d_xq8 = DeviceBuffer::alloc(0, t * h * (k / 32) * 36).unwrap();
        let d_y = DeviceBuffer::alloc(0, t * h * n * 4).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&x)).unwrap();
        d_w.copy_from_host(&w).unwrap();
        unsafe {
            hy4::quantize_heads_q8_1(d_x.ptr as *const f32, d_xq8.ptr, t as i32, h as i32, k as i32, ld_t, ld_h, &stream);
            hy4::batched_heads_q8(d_w.ptr, d_xq8.ptr, d_y.ptr as *mut f32, t as i32, h as i32, n as i32, k as i32, h * (k / 32), h * n, n, &stream);
        }
        stream.sync().unwrap();
        let mut got = vec![0.0f32; t * h * n];
        d_y.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
        let mut exp = vec![0.0f32; t * h * n];
        for ti in 0..t {
            for hi in 0..h {
                let xh = &x[ti * ld_t + hi * ld_h..ti * ld_t + hi * ld_h + k];
                for ni in 0..n {
                    let row = &w[(hi * n + ni) * rb..(hi * n + ni + 1) * rb];
                    exp[(ti * h + hi) * n + ni] = vec_dot_q_f32_ref(GgmlType::Q8_0, row, xh);
                }
            }
        }
        let re = rel_err(&got, &exp);
        println!("batched_heads_q8: rel_err={re:.6}");
        assert!(re < 0.005, "batched_heads_q8 failed: {re}");
    }

    // 2. KV store + MLA Decode + MLA Prefill (F16 and Q8_0 KV cache, SWA and DSA, with sinks)
    {
        let kv_lora = 512usize;
        let n_rot = 64usize;
        let row_elems = kv_lora + n_rot;
        let seq_len = 45usize;
        let ring_cap = 32usize;
        let h = 16usize;
        let eps = 1e-5f32;
        let base = 10000.0f32;
        let scale = 1.0f32 / (row_elems as f32).sqrt();

        let kv_in: Vec<f32> = (0..seq_len * row_elems).map(|_| rng.next_f32()).collect();
        let kv_w: Vec<f32> = (0..kv_lora).map(|_| 0.8 + 0.4 * rng.next_f32().abs()).collect();
        let pos_all: Vec<i32> = (0..seq_len as i32).collect();
        let sinks: Vec<f32> = (0..h).map(|i| -0.5 + 0.1 * i as f32).collect();

        // Build clean F32 reference KV rows in slot = p % ring_cap (only last 32 entries live in ring!)
        let mut kv_ref_f32 = vec![0.0f32; ring_cap * row_elems];
        for p in 0..seq_len {
            let slot = p % ring_cap;
            let src = &kv_in[p * row_elems..(p + 1) * row_elems];
            let ss: f32 = src[..kv_lora].iter().map(|v| v * v).sum();
            let r = 1.0 / (ss / kv_lora as f32 + eps).sqrt();
            for i in 0..kv_lora {
                kv_ref_f32[slot * row_elems + i] = src[i] * r * kv_w[i];
            }
            let mut rot = src[kv_lora..row_elems].to_vec();
            rope_vec_ref(&mut rot, p as i32, n_rot, base);
            kv_ref_f32[slot * row_elems + kv_lora..slot * row_elems + row_elems].copy_from_slice(&rot);
        }

        let d_kvin = DeviceBuffer::alloc(0, kv_in.len() * 4).unwrap();
        let d_kvw = DeviceBuffer::alloc(0, kv_w.len() * 4).unwrap();
        let d_posall = DeviceBuffer::alloc(0, pos_all.len() * 4).unwrap();
        let d_sinks = DeviceBuffer::alloc(0, sinks.len() * 4).unwrap();
        d_kvin.copy_from_host(bytemuck::cast_slice(&kv_in)).unwrap();
        d_kvw.copy_from_host(bytemuck::cast_slice(&kv_w)).unwrap();
        d_posall.copy_from_host(bytemuck::cast_slice(&pos_all)).unwrap();
        d_sinks.copy_from_host(bytemuck::cast_slice(&sinks)).unwrap();

        // Query T=3: SWA windows & DSA indirect keys inside live ring [seq_len - ring_cap .. seq_len)
        let t_q = 3usize;
        let q: Vec<f32> = (0..t_q * h * row_elems).map(|_| rng.next_f32() * 0.5).collect();
        let d_q = DeviceBuffer::alloc(0, q.len() * 4).unwrap();
        d_q.copy_from_host(bytemuck::cast_slice(&q)).unwrap();

        // Test SWA (`lo_opt` [15, 20, 28], `nk` [30, 25, 17]) -> max pos 44 >= 15 in ring_cap 32!
        let lo = [15i32, 20i32, 28i32];
        let nk = [30i32, 25i32, 17i32];
        let d_lo = DeviceBuffer::alloc(0, t_q * 4).unwrap();
        let d_nk = DeviceBuffer::alloc(0, t_q * 4).unwrap();
        d_lo.copy_from_host(bytemuck::cast_slice(&lo)).unwrap();
        d_nk.copy_from_host(bytemuck::cast_slice(&nk)).unwrap();

        // Also test DSA keys (`keys` [3, 24])
        let ldk = 24usize;
        let dsa_nk = [24i32, 12i32, 19i32];
        let mut dsa_keys = vec![0i32; t_q * ldk];
        for ti in 0..t_q {
            for j in 0..dsa_nk[ti] as usize {
                dsa_keys[ti * ldk + j] = (seq_len - ring_cap + ((j * 7 + ti * 3) % ring_cap)) as i32;
            }
        }
        let d_dsa_keys = DeviceBuffer::alloc(0, dsa_keys.len() * 4).unwrap();
        let d_dsa_nk = DeviceBuffer::alloc(0, dsa_nk.len() * 4).unwrap();
        d_dsa_keys.copy_from_host(bytemuck::cast_slice(&dsa_keys)).unwrap();
        d_dsa_nk.copy_from_host(bytemuck::cast_slice(&dsa_nk)).unwrap();

        let ws_floats = hy4::mla_decode_ws_floats(t_q, h, 32);
        let d_ws = DeviceBuffer::alloc(0, ws_floats * 4).unwrap();
        let d_out_dec = DeviceBuffer::alloc(0, t_q * h * kv_lora * 4).unwrap();
        let d_out_pre = DeviceBuffer::alloc(0, t_q * h * kv_lora * 4).unwrap();

        for is_q8 in [false, true] {
            let rb = if is_q8 { hy4::kv_q8_row_bytes(kv_lora, n_rot) } else { row_elems * 2 };
            let d_cache = DeviceBuffer::alloc(0, ring_cap * rb).unwrap();
            // Store in chunks <= ring_cap (just like chunked prefill) so slots do not race within one launch
            for start in (0..seq_len).step_by(ring_cap) {
                let cnt = (seq_len - start).min(ring_cap);
                unsafe {
                    hy4::kv_store(
                        (d_kvin.ptr as *const f32).add(start * row_elems),
                        cnt as i32,
                        kv_lora as i32,
                        n_rot as i32,
                        d_kvw.ptr as *const f32,
                        eps,
                        (d_posall.ptr as *const i32).add(start),
                        base,
                        d_cache.ptr,
                        ring_cap as i32,
                        is_q8,
                        rb as i32,
                        &stream,
                    );
                }
            }

            // SWA decode vs prefill vs CPU ref
            unsafe {
                hy4::mla_decode(
                    d_q.ptr as *const f32,
                    t_q as i32,
                    h as i32,
                    d_cache.ptr,
                    ring_cap as i32,
                    std::ptr::null(),
                    0,
                    d_lo.ptr as *const i32,
                    d_nk.ptr as *const i32,
                    32,
                    scale,
                    d_sinks.ptr as *const f32,
                    d_ws.ptr as *mut f32,
                    d_out_dec.ptr as *mut f32,
                    is_q8,
                    rb as i32,
                    &stream,
                );
                hy4::mla_prefill(
                    d_q.ptr as *const f32,
                    t_q as i32,
                    h as i32,
                    d_cache.ptr,
                    ring_cap as i32,
                    std::ptr::null(),
                    0,
                    d_lo.ptr as *const i32,
                    d_nk.ptr as *const i32,
                    scale,
                    d_sinks.ptr as *const f32,
                    d_out_pre.ptr as *mut f32,
                    is_q8,
                    rb as i32,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let mut got_dec = vec![0.0f32; t_q * h * kv_lora];
            let mut got_pre = vec![0.0f32; t_q * h * kv_lora];
            d_out_dec.copy_to_host(bytemuck::cast_slice_mut(&mut got_dec)).unwrap();
            d_out_pre.copy_to_host(bytemuck::cast_slice_mut(&mut got_pre)).unwrap();

            let mut exp_swa = vec![0.0f32; t_q * h * kv_lora];
            for ti in 0..t_q {
                for hi in 0..h {
                    let qr = &q[(ti * h + hi) * row_elems..(ti * h + hi + 1) * row_elems];
                    let mut m = sinks[hi];
                    let mut scs = Vec::with_capacity(nk[ti] as usize);
                    for j in 0..nk[ti] as usize {
                        let slot = (lo[ti] as usize + j) % ring_cap;
                        let kr = &kv_ref_f32[slot * row_elems..(slot + 1) * row_elems];
                        let s: f32 = qr.iter().zip(kr.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                        m = m.max(s);
                        scs.push((slot, s));
                    }
                    let mut den = (sinks[hi] - m).exp();
                    let mut acc = vec![0.0f32; kv_lora];
                    for (slot, s) in scs {
                        let p = (s - m).exp();
                        den += p;
                        let vr = &kv_ref_f32[slot * row_elems..slot * row_elems + kv_lora];
                        for d in 0..kv_lora {
                            acc[d] += p * vr[d];
                        }
                    }
                    for d in 0..kv_lora {
                        exp_swa[(ti * h + hi) * kv_lora + d] = acc[d] / den;
                    }
                }
            }
            let tol = if is_q8 { 0.015 } else { 2e-3 };
            let r_dec = rel_err(&got_dec, &exp_swa);
            let r_pre = rel_err(&got_pre, &exp_swa);
            let m_dp = max_abs_err(&got_dec, &got_pre);
            println!("MLA SWA q8={is_q8}: dec_rel={r_dec:.6} pre_rel={r_pre:.6} dec_vs_pre={m_dp:.6}");
            assert!(r_dec < tol, "MLA SWA decode q8={is_q8} rel_err too high: {r_dec}");
            assert!(r_pre < tol, "MLA SWA prefill q8={is_q8} rel_err too high: {r_pre}");
            // prefill uses f16 Tensor Core MMA (eps ~ 5e-4 on Q and P) vs FP32 FMA decode.
            assert!(m_dp < tol, "MLA SWA decode vs prefill q8={is_q8}: {m_dp}");

            // DSA indirect keys decode vs prefill vs CPU ref
            unsafe {
                hy4::mla_decode(
                    d_q.ptr as *const f32,
                    t_q as i32,
                    h as i32,
                    d_cache.ptr,
                    ring_cap as i32,
                    d_dsa_keys.ptr as *const i32,
                    ldk as i32,
                    std::ptr::null(),
                    d_dsa_nk.ptr as *const i32,
                    ldk as i32,
                    scale,
                    d_sinks.ptr as *const f32,
                    d_ws.ptr as *mut f32,
                    d_out_dec.ptr as *mut f32,
                    is_q8,
                    rb as i32,
                    &stream,
                );
                hy4::mla_prefill(
                    d_q.ptr as *const f32,
                    t_q as i32,
                    h as i32,
                    d_cache.ptr,
                    ring_cap as i32,
                    d_dsa_keys.ptr as *const i32,
                    ldk as i32,
                    std::ptr::null(),
                    d_dsa_nk.ptr as *const i32,
                    scale,
                    d_sinks.ptr as *const f32,
                    d_out_pre.ptr as *mut f32,
                    is_q8,
                    rb as i32,
                    &stream,
                );
            }
            stream.sync().unwrap();
            d_out_dec.copy_to_host(bytemuck::cast_slice_mut(&mut got_dec)).unwrap();
            d_out_pre.copy_to_host(bytemuck::cast_slice_mut(&mut got_pre)).unwrap();
            let mut exp_dsa = vec![0.0f32; t_q * h * kv_lora];
            for ti in 0..t_q {
                for hi in 0..h {
                    let qr = &q[(ti * h + hi) * row_elems..(ti * h + hi + 1) * row_elems];
                    let mut m = sinks[hi];
                    let mut scs = Vec::with_capacity(dsa_nk[ti] as usize);
                    for j in 0..dsa_nk[ti] as usize {
                        let slot = (dsa_keys[ti * ldk + j] as usize) % ring_cap;
                        let kr = &kv_ref_f32[slot * row_elems..(slot + 1) * row_elems];
                        let s: f32 = qr.iter().zip(kr.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                        m = m.max(s);
                        scs.push((slot, s));
                    }
                    let mut den = (sinks[hi] - m).exp();
                    let mut acc = vec![0.0f32; kv_lora];
                    for (slot, s) in scs {
                        let p = (s - m).exp();
                        den += p;
                        let vr = &kv_ref_f32[slot * row_elems..slot * row_elems + kv_lora];
                        for d in 0..kv_lora {
                            acc[d] += p * vr[d];
                        }
                    }
                    for d in 0..kv_lora {
                        exp_dsa[(ti * h + hi) * kv_lora + d] = acc[d] / den;
                    }
                }
            }
            let r_d_dec = rel_err(&got_dec, &exp_dsa);
            let r_d_pre = rel_err(&got_pre, &exp_dsa);
            println!("MLA DSA q8={is_q8}: dec_rel={r_d_dec:.6} pre_rel={r_d_pre:.6}");
            assert!(r_d_dec < tol, "MLA DSA decode q8={is_q8} rel_err too high: {r_d_dec}");
            assert!(r_d_pre < tol, "MLA DSA prefill q8={is_q8} rel_err too high: {r_d_pre}");
        }
    }

    // 3. Pooled Indexer: ix_pool, ix_score, ix_select (with tie breaking and tail)
    {
        let kpool = 4usize;
        let dim = 128usize;
        let n_rot = 64usize;
        let heads = 16usize;
        let n_pool = 60usize;
        let total_tok = n_pool * kpool + 3; // pos = 243 -> nv = 60, ntail = 4
        let ring_cap = total_tok;
        let ape: Vec<f32> = (0..kpool * dim).map(|_| rng.next_f32() * 0.3).collect();
        let w: Vec<f32> = (0..dim).map(|_| 0.8 + 0.3 * rng.next_f32().abs()).collect();
        let ivg_f32: Vec<f32> = (0..ring_cap * 2 * dim).map(|_| rng.next_f32()).collect();
        let iq: Vec<f32> = (0..2 * heads * dim).map(|_| rng.next_f32() * 0.4).collect();
        let iwt: Vec<f32> = (0..2 * heads).map(|_| rng.next_f32().abs() + 0.1).collect();
        let ix_pos = [243i32, 37i32];

        let d_ape = DeviceBuffer::alloc(0, ape.len() * 4).unwrap();
        let d_w = DeviceBuffer::alloc(0, w.len() * 4).unwrap();
        let d_iq = DeviceBuffer::alloc(0, iq.len() * 4).unwrap();
        let d_iwt = DeviceBuffer::alloc(0, iwt.len() * 4).unwrap();
        let d_ixpos = DeviceBuffer::alloc(0, ix_pos.len() * 4).unwrap();
        let d_score = DeviceBuffer::alloc(0, 2 * n_pool * 4).unwrap();
        d_ape.copy_from_host(bytemuck::cast_slice(&ape)).unwrap();
        d_w.copy_from_host(bytemuck::cast_slice(&w)).unwrap();
        d_iq.copy_from_host(bytemuck::cast_slice(&iq)).unwrap();
        d_iwt.copy_from_host(bytemuck::cast_slice(&iwt)).unwrap();
        d_ixpos.copy_from_host(bytemuck::cast_slice(&ix_pos)).unwrap();

        let mut exp_pooled = vec![0.0f32; n_pool * dim];
        for g in 0..n_pool {
            for c in 0..dim {
                let mut lgs = [0.0f32; 4];
                let mut mx = f32::NEG_INFINITY;
                for k in 0..kpool {
                    let p = (g * kpool + k) % ring_cap;
                    lgs[k] = ivg_f32[p * 2 * dim + dim + c] + ape[k * dim + c];
                    mx = mx.max(lgs[k]);
                }
                let mut den = 0.0f32;
                let mut num = 0.0f32;
                for k in 0..kpool {
                    let p = (g * kpool + k) % ring_cap;
                    let e = (lgs[k] - mx).exp();
                    den += e;
                    num += e * ivg_f32[p * 2 * dim + c];
                }
                exp_pooled[g * dim + c] = num / den;
            }
            let ss: f32 = exp_pooled[g * dim..(g + 1) * dim].iter().map(|v| v * v).sum();
            let r = 1.0 / (ss / dim as f32 + 1e-5).sqrt();
            for c in 0..dim {
                exp_pooled[g * dim + c] *= r * w[c];
            }
            rope_vec_ref(&mut exp_pooled[g * dim + (dim - n_rot)..(g + 1) * dim], (g * kpool) as i32, n_rot, 10000.0);
        }

        for is_f16 in [false, true] {
            let d_ivg = DeviceBuffer::alloc(0, ivg_f32.len() * if is_f16 { 2 } else { 4 }).unwrap();
            let d_pooled = DeviceBuffer::alloc(0, n_pool * dim * if is_f16 { 2 } else { 4 }).unwrap();
            if is_f16 {
                let h_ivg: Vec<u16> = ivg_f32.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
                d_ivg.copy_from_host(bytemuck::cast_slice(&h_ivg)).unwrap();
            } else {
                d_ivg.copy_from_host(bytemuck::cast_slice(&ivg_f32)).unwrap();
            }
            unsafe {
                hy4::ix_pool(
                    d_ivg.ptr,
                    ring_cap as i32,
                    0,
                    n_pool as i32,
                    kpool as i32,
                    dim as i32,
                    d_ape.ptr as *const f32,
                    d_w.ptr as *const f32,
                    1e-5,
                    n_rot as i32,
                    10000.0,
                    d_pooled.ptr,
                    is_f16,
                    &stream,
                );
                hy4::ix_score(
                    d_iq.ptr as *const f32,
                    d_iwt.ptr as *const f32,
                    2,
                    heads as i32,
                    dim as i32,
                    d_pooled.ptr,
                    d_ixpos.ptr as *const i32,
                    kpool as i32,
                    n_pool as i32,
                    d_score.ptr as *mut f32,
                    n_pool,
                    is_f16,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let mut got_sc = vec![0.0f32; 2 * n_pool];
            d_score.copy_to_host(bytemuck::cast_slice_mut(&mut got_sc)).unwrap();
            let mut exp_sc = vec![0.0f32; 2 * n_pool];
            for ti in 0..2 {
                let nv = (ix_pos[ti] as usize) / kpool;
                for g in 0..n_pool {
                    if g >= nv {
                        exp_sc[ti * n_pool + g] = f32::NEG_INFINITY;
                    } else {
                        let mut s = 0.0f32;
                        for hi in 0..heads {
                            let qh = &iq[(ti * heads + hi) * dim..(ti * heads + hi + 1) * dim];
                            let pk = &exp_pooled[g * dim..(g + 1) * dim];
                            let d: f32 = qh.iter().zip(pk.iter()).map(|(a, b)| a * b).sum();
                            s += d.max(0.0) * iwt[ti * heads + hi];
                        }
                        exp_sc[ti * n_pool + g] = s;
                    }
                }
            }
            let tol = if is_f16 { 5e-3 } else { 1e-4 };
            for ti in 0..2 {
                let nv = (ix_pos[ti] as usize) / kpool;
                assert!(
                    rel_err(&got_sc[ti * n_pool..ti * n_pool + nv], &exp_sc[ti * n_pool..ti * n_pool + nv]) < tol,
                    "ix_score f16={is_f16} mismatch"
                );
                for g in nv..n_pool {
                    assert!(got_sc[ti * n_pool + g] == f32::NEG_INFINITY);
                }
            }
        }

        // ix_select with exact tie at threshold (top = 16, nv = 60, ntail = 4)
        let mut sc_host = vec![0.0f32; 2 * n_pool];
        for g in 0..n_pool {
            sc_host[g] = if g < 10 { 100.0 - g as f32 } else if g % 3 == 0 { 12.5 } else { 1.0 };
        }
        for g in 0..9 {
            sc_host[n_pool + g] = 5.0 + g as f32;
        }
        let top = 16usize;
        let ldk = top * kpool + kpool;
        let d_keys = DeviceBuffer::alloc(0, 2 * ldk * 4).unwrap();
        let d_ixnk = DeviceBuffer::alloc(0, 8).unwrap();
        d_score.copy_from_host(bytemuck::cast_slice(&sc_host)).unwrap();
        unsafe {
            hy4::ix_select(
                d_score.ptr as *const f32,
                n_pool,
                2,
                d_ixpos.ptr as *const i32,
                kpool as i32,
                top as i32,
                d_keys.ptr as *mut i32,
                ldk as i32,
                d_ixnk.ptr as *mut i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got_keys = vec![0i32; 2 * ldk];
        let mut got_nk = [0i32; 2];
        d_keys.copy_to_host(bytemuck::cast_slice_mut(&mut got_keys)).unwrap();
        d_ixnk.copy_to_host(bytemuck::cast_slice_mut(&mut got_nk)).unwrap();

        // Expected token 0: pools 0..9 + first 6 multiples of 3 >= 10 -> 12,15,18,21,24,27; then tail 240,241,242,243
        assert_eq!(got_nk[0], (top * kpool + 4) as i32);
        let mut exp_pools0: Vec<usize> = (0..10).chain([12, 15, 18, 21, 24, 27]).collect();
        exp_pools0.sort_unstable();
        for (slot, &pg) in exp_pools0.iter().enumerate() {
            for k in 0..kpool {
                assert_eq!(got_keys[slot * kpool + k], (pg * kpool + k) as i32);
            }
        }
        for k in 0..4 {
            assert_eq!(got_keys[top * kpool + k], (60 * kpool + k) as i32);
        }
        // Token 1: pos=37 -> nv=9 <= top=16 -> all 9*4 + 2 = 38 keys
        assert_eq!(got_nk[1], 38);
        for k in 0..38 {
            assert_eq!(got_keys[ldk + k], k as i32);
        }
    }
}

#[test]
fn test_hy4_router_and_vision() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x55aa_1357_2468_9abc);

    // 1. Router (n_exp = 512, n_used = 16, bias ranking + ties + norm + scale)
    {
        let t = 3usize;
        let n_exp = 512usize;
        let n_used = 16usize;
        let scale = 3.998f32;
        let mut logits = vec![-2.0f32; t * n_exp];
        let mut bias = vec![0.0f32; n_exp];
        for i in 0..n_exp {
            logits[i] = -3.0 + ((i * 73) % 100) as f32 * 0.04;
            logits[n_exp + i] = 0.0; // tie test on token 1!
            logits[2 * n_exp + i] = rng.next_f32() * 2.0;
            bias[i] = rng.next_f32() * 0.1;
        }
        // Constant tie checks covered on token 1 above (zero bias variation could break only on id)

        let d_lg = DeviceBuffer::alloc(0, logits.len() * 4).unwrap();
        let d_bs = DeviceBuffer::alloc(0, bias.len() * 4).unwrap();
        let d_ids = DeviceBuffer::alloc(0, t * n_used * 4).unwrap();
        let d_wout = DeviceBuffer::alloc(0, t * n_used * 4).unwrap();
        d_lg.copy_from_host(bytemuck::cast_slice(&logits)).unwrap();
        d_bs.copy_from_host(bytemuck::cast_slice(&bias)).unwrap();
        unsafe {
            hy4::router_sigmoid_topk(
                d_lg.ptr as *const f32,
                t as i32,
                n_exp as i32,
                d_bs.ptr as *const f32,
                n_used as i32,
                true,
                scale,
                d_ids.ptr as *mut i32,
                d_wout.ptr as *mut f32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got_ids = vec![0i32; t * n_used];
        let mut got_w = vec![0.0f32; t * n_used];
        d_ids.copy_to_host(bytemuck::cast_slice_mut(&mut got_ids)).unwrap();
        d_wout.copy_to_host(bytemuck::cast_slice_mut(&mut got_w)).unwrap();

        for ti in 0..t {
            let mut order: Vec<(f32, usize, f32)> = (0..n_exp)
                .map(|e| {
                    let p = 1.0 / (1.0 + (-logits[ti * n_exp + e]).exp());
                    (p + bias[e], e, p)
                })
                .collect();
            order.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let sum: f32 = order[..n_used].iter().map(|x| x.2).sum::<f32>().max(6.103515625e-5);
            for s in 0..n_used {
                assert_eq!(got_ids[ti * n_used + s], order[s].1 as i32, "router id mismatch t={ti} s={s}");
                let ew = order[s].2 / sum * scale;
                assert!((got_w[ti * n_used + s] - ew).abs() < 1e-5, "router weight mismatch");
            }
        }
    }

    // 2. Vision Ops (LayerNorm->F16, RMSNorm, Bias+GELU tanh/erf, Residual+Bias, MergeGather, PlaceRows, FlashAttn hd=72)
    {
        let ow = 4usize;
        let oh = 4usize;
        let gw = ow * 2;
        let gh = oh * 2;
        let n = gw * gh; // 64
        let heads = 16usize;
        let hd = 72usize;
        let d = heads * hd; // 1152
        let out_dim = 256usize;
        let eps = 1e-5f32;

        // FlashAttention with hd=72, N=64, heads=16
        let qkv: Vec<f32> = (0..n * 3 * d).map(|_| rng.next_f32() * 0.4).collect();
        let scale = 1.0f32 / (hd as f32).sqrt();
        let d_qkv = DeviceBuffer::alloc(0, qkv.len() * 4).unwrap();
        let d_att = DeviceBuffer::alloc(0, n * d * 4).unwrap();
        d_qkv.copy_from_host(bytemuck::cast_slice(&qkv)).unwrap();
        unsafe {
            hy4::vis_flash_attn(d_qkv.ptr as *const f32, n as i32, heads as i32, hd as i32, scale, d_att.ptr as *mut f32, &stream);
        }
        stream.sync().unwrap();
        let mut got_att = vec![0.0f32; n * d];
        d_att.copy_to_host(bytemuck::cast_slice_mut(&mut got_att)).unwrap();
        let mut exp_att = vec![0.0f32; n * d];
        for qi in 0..n {
            for hi in 0..heads {
                let qr = &qkv[qi * 3 * d + hi * hd..qi * 3 * d + (hi + 1) * hd];
                let mut scores = Vec::with_capacity(n);
                let mut mx = f32::NEG_INFINITY;
                for ki in 0..n {
                    let kr = &qkv[ki * 3 * d + d + hi * hd..ki * 3 * d + d + (hi + 1) * hd];
                    let s = qr.iter().zip(kr.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                    mx = mx.max(s);
                    scores.push(s);
                }
                let mut den = 0.0f32;
                for ki in 0..n {
                    let p = (scores[ki] - mx).exp();
                    den += p;
                    let vr = &qkv[ki * 3 * d + 2 * d + hi * hd..ki * 3 * d + 2 * d + (hi + 1) * hd];
                    for idx in 0..hd {
                        exp_att[qi * d + hi * hd + idx] += p * vr[idx];
                    }
                }
                for idx in 0..hd {
                    exp_att[qi * d + hi * hd + idx] /= den;
                }
            }
        }
        assert!(max_abs_err(&got_att, &exp_att) < 1e-5, "vis_flash_attn hd=72 mismatch");

        // LayerNorm -> F16 + F16 MMA check + Bias GELU (erf & tanh) + RMSNorm + MergeGather + PlaceRows
        let x: Vec<f32> = (0..n * d).map(|_| rng.next_f32()).collect();
        let lnw: Vec<f32> = (0..d).map(|_| 0.8 + 0.3 * rng.next_f32().abs()).collect();
        let lnb: Vec<f32> = (0..d).map(|_| rng.next_f32() * 0.2).collect();
        let d_vx = DeviceBuffer::alloc(0, x.len() * 4).unwrap();
        let d_lnw = DeviceBuffer::alloc(0, d * 4).unwrap();
        let d_lnb = DeviceBuffer::alloc(0, d * 4).unwrap();
        let d_y16 = DeviceBuffer::alloc(0, n * 4 * d * 2).unwrap();
        d_vx.copy_from_host(bytemuck::cast_slice(&x)).unwrap();
        d_lnw.copy_from_host(bytemuck::cast_slice(&lnw)).unwrap();
        d_lnb.copy_from_host(bytemuck::cast_slice(&lnb)).unwrap();

        unsafe {
            hy4::vis_layernorm_f16(d_vx.ptr as *const f32, d_lnw.ptr as *const f32, d_lnb.ptr as *const f32, n as i32, d as i32, eps, d_y16.ptr as *mut u16, &stream);
        }
        stream.sync().unwrap();
        let mut got_ln16 = vec![0u16; n * d];
        d_y16.copy_to_host(bytemuck::cast_slice_mut(&mut got_ln16[..n * d])).unwrap();
        let mut exp_ln = vec![0.0f32; n * d];
        for r in 0..n {
            let row = &x[r * d..(r + 1) * d];
            let mean = row.iter().sum::<f32>() / d as f32;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let inv = 1.0 / (var + eps).sqrt();
            for c in 0..d {
                exp_ln[r * d + c] = (row[c] - mean) * inv * lnw[c] + lnb[c];
            }
        }
        let got_ln: Vec<f32> = got_ln16[..n * d].iter().map(|&b| f16::from_bits(b).to_f32()).collect();
        assert!(rel_err(&got_ln, &exp_ln) < 2e-3, "vis_layernorm_f16 mismatch");

        // rmsnorm + merge_gather + place_rows
        let mut x_rms = x.clone();
        unsafe {
            hy4::vis_rmsnorm_inplace(d_vx.ptr as *mut f32, d_lnw.ptr as *const f32, n as i32, d as i32, eps, &stream);
            hy4::vis_merge_gather_f16(d_vx.ptr as *const f32, gw as i32, d as i32, ow as i32, oh as i32, d_y16.ptr as *mut u16, &stream);
        }
        stream.sync().unwrap();
        let nb = ow * oh;
        let mut got_m16 = vec![0u16; nb * 4 * d];
        d_y16.copy_to_host(bytemuck::cast_slice_mut(&mut got_m16)).unwrap();
        for r in 0..n {
            let ss = x_rms[r * d..(r + 1) * d].iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0 / (ss / d as f32 + eps).sqrt();
            for c in 0..d {
                x_rms[r * d + c] *= inv * lnw[c];
            }
        }
        for b in 0..nb {
            let by = b / ow;
            let bx = b % ow;
            for ic in 0..d {
                for ky in 0..2 {
                    for kx in 0..2 {
                        let exp_v = f16::from_f32(x_rms[((2 * by + ky) * gw + 2 * bx + kx) * d + ic]).to_f32();
                        let got_v = f16::from_bits(got_m16[b * 4 * d + ic * 4 + ky * 2 + kx]).to_f32();
                        assert!((got_v - exp_v).abs() < 5e-3, "merge_gather mismatch");
                    }
                }
            }
        }

        // place_rows
        let py: Vec<f32> = (0..nb * out_dim).map(|_| rng.next_f32()).collect();
        let pb: Vec<f32> = (0..out_dim).map(|_| rng.next_f32()).collect();
        let nl: Vec<f32> = (0..out_dim).map(|_| rng.next_f32()).collect();
        let d_py = DeviceBuffer::alloc(0, py.len() * 4).unwrap();
        let d_pb = DeviceBuffer::alloc(0, pb.len() * 4).unwrap();
        let d_nl = DeviceBuffer::alloc(0, nl.len() * 4).unwrap();
        let d_pout = DeviceBuffer::alloc(0, (ow + 1) * oh * out_dim * 4).unwrap();
        d_py.copy_from_host(bytemuck::cast_slice(&py)).unwrap();
        d_pb.copy_from_host(bytemuck::cast_slice(&pb)).unwrap();
        d_nl.copy_from_host(bytemuck::cast_slice(&nl)).unwrap();
        unsafe {
            hy4::vis_place_rows(d_py.ptr as *const f32, d_pb.ptr as *const f32, d_nl.ptr as *const f32, ow as i32, oh as i32, out_dim as i32, d_pout.ptr as *mut f32, &stream);
        }
        stream.sync().unwrap();
        let mut got_pout = vec![0.0f32; (ow + 1) * oh * out_dim];
        d_pout.copy_to_host(bytemuck::cast_slice_mut(&mut got_pout)).unwrap();
        for by in 0..oh {
            for bx in 0..=ow {
                let r = by * (ow + 1) + bx;
                for c in 0..out_dim {
                    let ev = if bx == ow { nl[c] } else { py[(by * ow + bx) * out_dim + c] + pb[c] };
                    assert!((got_pout[r * out_dim + c] - ev).abs() < 1e-6);
                }
            }
        }

        // vis_bias_act_f16 (tanh=1 & erf=2), vis_add_bias, vis_residual_bias, launch_gemm_f16_mma_nt
        let m_vis = 32usize;
        let k_vis = 128usize;
        let n_vis = 64usize;
        let act_in: Vec<f32> = (0..m_vis * n_vis).map(|_| rng.next_f32()).collect();
        let act_b: Vec<f32> = (0..n_vis).map(|_| rng.next_f32() * 0.25).collect();
        let d_ain = DeviceBuffer::alloc(0, act_in.len() * 4).unwrap();
        let d_ab = DeviceBuffer::alloc(0, act_b.len() * 4).unwrap();
        let d_ay16 = DeviceBuffer::alloc(0, act_in.len() * 2).unwrap();
        d_ain.copy_from_host(bytemuck::cast_slice(&act_in)).unwrap();
        d_ab.copy_from_host(bytemuck::cast_slice(&act_b)).unwrap();
        for act in [1i32, 2] {
            unsafe {
                hy4::vis_bias_act_f16(d_ain.ptr as *const f32, d_ab.ptr as *const f32, act_in.len(), n_vis as i32, act, d_ay16.ptr as *mut u16, &stream);
            }
            stream.sync().unwrap();
            let mut got16 = vec![0u16; act_in.len()];
            d_ay16.copy_to_host(bytemuck::cast_slice_mut(&mut got16)).unwrap();
            let got: Vec<f32> = got16.iter().map(|&b| f16::from_bits(b).to_f32()).collect();
            let exp: Vec<f32> = (0..act_in.len())
                .map(|i| {
                    let v = act_in[i] + act_b[i % n_vis];
                    if act == 1 {
                        let k = 0.7978845608f32;
                        0.5 * v * (1.0 + (k * (v + 0.044715 * v * v * v)).tanh())
                    } else {
                        // erf approximation check via Abramowitz-Stegun or exact comparison
                        let t = 1.0 / (1.0 + 0.3275911 * (v * 0.7071067812f32).abs());
                        let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) * (-(v * v * 0.5)).exp();
                        let erf_v = if v < 0.0 { -y } else { y };
                        0.5 * v * (1.0 + erf_v)
                    }
                })
                .collect();
            assert!(rel_err(&got, &exp) < 2e-3, "vis_bias_act_f16 act={act} mismatch");
        }

        // gemv_f32 & gemv_bf16 + swiglu_f32 + sigmoid_mul + store_rows + F16 MMA
        let wf32: Vec<f32> = (0..n_vis * k_vis).map(|_| rng.next_f32()).collect();
        let xf32: Vec<f32> = (0..m_vis * k_vis).map(|_| rng.next_f32()).collect();
        let wbf16: Vec<u16> = wf32.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let d_wf = DeviceBuffer::alloc(0, wf32.len() * 4).unwrap();
        let d_wbf = DeviceBuffer::alloc(0, wbf16.len() * 2).unwrap();
        let d_xf = DeviceBuffer::alloc(0, xf32.len() * 4).unwrap();
        let d_yf = DeviceBuffer::alloc(0, m_vis * n_vis * 4).unwrap();
        let d_a16 = DeviceBuffer::alloc(0, m_vis * k_vis * 2).unwrap();
        let d_b16 = DeviceBuffer::alloc(0, n_vis * k_vis * 2).unwrap();
        d_wf.copy_from_host(bytemuck::cast_slice(&wf32)).unwrap();
        d_wbf.copy_from_host(bytemuck::cast_slice(&wbf16)).unwrap();
        d_xf.copy_from_host(bytemuck::cast_slice(&xf32)).unwrap();

        unsafe {
            hy4::gemv_f32(d_wf.ptr as *const f32, n_vis as i32, k_vis as i32, d_xf.ptr as *const f32, m_vis as i32, d_yf.ptr as *mut f32, 0.75, &stream);
        }
        stream.sync().unwrap();
        let mut got_gf = vec![0.0f32; m_vis * n_vis];
        d_yf.copy_to_host(bytemuck::cast_slice_mut(&mut got_gf)).unwrap();
        let mut exp_gf = vec![0.0f32; m_vis * n_vis];
        for mi in 0..m_vis {
            for ni in 0..n_vis {
                let s: f32 = (0..k_vis).map(|ki| wf32[ni * k_vis + ki] * xf32[mi * k_vis + ki]).sum();
                exp_gf[mi * n_vis + ni] = s * 0.75;
            }
        }
        assert!(max_abs_err(&got_gf, &exp_gf) < 1e-4, "gemv_f32 mismatch");

        unsafe {
            hy4::gemv_bf16(d_wbf.ptr as *const u16, n_vis as i32, k_vis as i32, d_xf.ptr as *const f32, m_vis as i32, d_yf.ptr as *mut f32, 1.0, &stream);
        }
        stream.sync().unwrap();
        d_yf.copy_to_host(bytemuck::cast_slice_mut(&mut got_gf)).unwrap();
        for mi in 0..m_vis {
            for ni in 0..n_vis {
                let s: f32 = (0..k_vis)
                    .map(|ki| bf16::from_bits(wbf16[ni * k_vis + ki]).to_f32() * xf32[mi * k_vis + ki])
                    .sum();
                exp_gf[mi * n_vis + ni] = s;
            }
        }
        assert!(max_abs_err(&got_gf, &exp_gf) < 1e-4, "gemv_bf16 mismatch");

        // F16 MMA with Bf16 weights dequanted via launch_dequant_to_f16
        unsafe {
            launch_dequant_to_f16(GgmlType::F32 as i32, d_xf.ptr, d_a16.ptr, (m_vis * k_vis) as i64, stream.raw);
            launch_dequant_to_f16(GgmlType::Bf16 as i32, d_wbf.ptr, d_b16.ptr, (n_vis * k_vis) as i64, stream.raw);
            launch_gemm_f16_mma_nt(d_a16.ptr, d_b16.ptr, d_yf.ptr as *mut f32, m_vis as i32, n_vis as i32, k_vis as i32, 1.0, 0.0, stream.raw);
        }
        stream.sync().unwrap();
        d_yf.copy_to_host(bytemuck::cast_slice_mut(&mut got_gf)).unwrap();
        assert!(rel_err(&got_gf, &exp_gf) < 2e-3, "gemm_f16_mma_nt mismatch");
    }
}

#[test]
fn test_hy4_engine_helpers() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x7337_beef_1234_9abc);

    // 1. copy_cols
    {
        let rows = 5usize;
        let ld_src = 12usize;
        let ld_dst = 16usize;
        let width = 7usize;
        let src: Vec<f32> = (0..rows * ld_src).map(|_| rng.next_f32()).collect();
        let d_src = DeviceBuffer::alloc(0, src.len() * 4).unwrap();
        let d_dst = DeviceBuffer::alloc_zeroed(0, rows * ld_dst * 4).unwrap();
        d_src.copy_from_host(bytemuck::cast_slice(&src)).unwrap();
        unsafe {
            hy4::copy_cols(
                d_src.ptr as *const f32,
                ld_src,
                d_dst.ptr as *mut f32,
                ld_dst,
                rows as i32,
                width as i32,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got = vec![0.0f32; rows * ld_dst];
        d_dst.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
        for r in 0..rows {
            for c in 0..ld_dst {
                let exp = if c < width { src[r * ld_src + c] } else { 0.0 };
                assert!((got[r * ld_dst + c] - exp).abs() < 1e-6);
            }
        }
    }

    // 2. add_f32 (both aligned vec4 multiple-of-4 and unaligned scalar fallback)
    for n in [1024usize, 1027usize] {
        let a: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let b: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
        let d_a = DeviceBuffer::alloc(0, n * 4).unwrap();
        let d_b = DeviceBuffer::alloc(0, n * 4).unwrap();
        d_a.copy_from_host(bytemuck::cast_slice(&a)).unwrap();
        d_b.copy_from_host(bytemuck::cast_slice(&b)).unwrap();
        unsafe {
            hy4::add_f32(d_a.ptr as *mut f32, d_b.ptr as *const f32, n, &stream);
        }
        stream.sync().unwrap();
        let mut got = vec![0.0f32; n];
        d_a.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
        let exp: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        assert!(max_abs_err(&got, &exp) < 1e-6);
    }

    // 3. rmsnorm_strided (strided output + in-place)
    {
        let rows = 4usize;
        let dim = 512usize;
        let ld_x = dim + 64;
        let ld_y = 2 * dim;
        let eps = 1e-5f32;
        let x: Vec<f32> = (0..rows * ld_x).map(|_| rng.next_f32()).collect();
        let w: Vec<f32> = (0..dim).map(|_| 0.7 + 0.5 * rng.next_f32().abs()).collect();
        let d_x = DeviceBuffer::alloc(0, x.len() * 4).unwrap();
        let d_w = DeviceBuffer::alloc(0, dim * 4).unwrap();
        let d_y = DeviceBuffer::alloc_zeroed(0, rows * ld_y * 4).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&x)).unwrap();
        d_w.copy_from_host(bytemuck::cast_slice(&w)).unwrap();
        unsafe {
            hy4::rmsnorm_strided(
                d_x.ptr as *const f32,
                ld_x,
                d_w.ptr as *const f32,
                d_y.ptr as *mut f32,
                ld_y,
                dim as i32,
                rows as i32,
                eps,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got = vec![0.0f32; rows * ld_y];
        d_y.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
        for r in 0..rows {
            let xr = &x[r * ld_x..r * ld_x + dim];
            let ss = xr.iter().map(|v| v * v).sum::<f32>();
            let inv = 1.0 / (ss / dim as f32 + eps).sqrt();
            for c in 0..dim {
                let ev = xr[c] * inv * w[c];
                assert!((got[r * ld_y + c] - ev).abs() < 1e-5);
            }
            for c in dim..ld_y {
                assert_eq!(got[r * ld_y + c], 0.0);
            }
        }
        // In-place check
        unsafe {
            hy4::rmsnorm_strided(
                d_x.ptr as *const f32,
                ld_x,
                d_w.ptr as *const f32,
                d_x.ptr as *mut f32,
                ld_x,
                dim as i32,
                rows as i32,
                eps,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got_in = vec![0.0f32; rows * ld_x];
        d_x.copy_to_host(bytemuck::cast_slice_mut(&mut got_in)).unwrap();
        for r in 0..rows {
            for c in 0..dim {
                assert!((got_in[r * ld_x + c] - got[r * ld_y + c]).abs() < 1e-5);
            }
        }
    }

    // 4. top1_prob (120832 vocab + deterministic smaller-index tie-break)
    {
        let n_vocab = 120_832usize;
        let mut logits: Vec<f32> = (0..n_vocab).map(|_| rng.next_f32() * 3.0).collect();
        // Set exact tie at two indices across different blocks: smaller index must win
        logits[35_001] = 12.5;
        logits[99_777] = 12.5;
        let d_logits = DeviceBuffer::alloc(0, n_vocab * 4).unwrap();
        let d_out = DeviceBuffer::alloc(0, hy4::HY4_TOP1_SCRATCH_FLOATS * 4).unwrap();
        d_logits.copy_from_host(bytemuck::cast_slice(&logits)).unwrap();
        unsafe {
            hy4::top1_prob(d_logits.ptr as *const f32, n_vocab as i32, d_out.ptr as *mut f32, &stream);
        }
        stream.sync().unwrap();
        let mut all_out = vec![0.0f32; hy4::HY4_TOP1_SCRATCH_FLOATS];
        d_out.copy_to_host(bytemuck::cast_slice_mut(&mut all_out)).unwrap();
        let got_id = all_out[0].to_bits() as i32;
        let got_prob = all_out[1];
        let sum_exp: f64 = logits.iter().map(|&v| ((v as f64) - 12.5).exp()).sum();
        let exp_prob = (1.0 / sum_exp) as f32;
        assert_eq!(got_id, 35_001, "tie-break must pick smallest index");
        assert!((got_prob - exp_prob).abs() < 1e-5, "top1_prob {got_prob} != {exp_prob}");
    }

    // 5. build_moe_plan
    {
        let ids = [3i32, 0, 5, 2, 1, 7];
        let slot_of = [10i32, -1, 0, 4, -1, -1, 2, 1];
        let vram_base = 0x1_0000_0000u64;
        let blob_bytes = 4_890_624usize;
        let d_ids = DeviceBuffer::alloc(0, ids.len() * 4).unwrap();
        let d_slot = DeviceBuffer::alloc(0, slot_of.len() * 4).unwrap();
        let d_plan = DeviceBuffer::alloc(0, ids.len() * 8).unwrap();
        d_ids.copy_from_host(bytemuck::cast_slice(&ids)).unwrap();
        d_slot.copy_from_host(bytemuck::cast_slice(&slot_of)).unwrap();
        unsafe {
            hy4::build_moe_plan(
                d_ids.ptr as *const i32,
                d_slot.ptr as *const i32,
                vram_base,
                blob_bytes,
                ids.len() as i32,
                d_plan.ptr as *mut u64,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got_plan = vec![0u64; ids.len()];
        d_plan.copy_to_host(bytemuck::cast_slice_mut(&mut got_plan)).unwrap();
        for (i, &id) in ids.iter().enumerate() {
            let s = slot_of[id as usize];
            let exp = if s >= 0 { vram_base + (s as u64) * (blob_bytes as u64) } else { 0u64 };
            assert_eq!(got_plan[i], exp);
        }
    }
}

#[test]
fn test_hy4_prefill_mma_kernels_parity() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x2026_1122_dead_beef);

    // 1. batched_heads_q8_mma for k_b (H=32, N=512, K=192) and v_b (H=64, N=256, K=512)
    for &(t, h, n, k, ld_h) in &[(73usize, 32usize, 512usize, 192usize, 256usize), (67, 64, 256, 512, 512)] {
        let ld_t = h * ld_h;
        let x: Vec<f32> = (0..t * ld_t).map(|_| rng.next_f32() * 0.5).collect();
        let rb = k / 32 * 34;
        let mut w = Vec::with_capacity(h * n * rb);
        for _ in 0..(h * n) {
            w.extend_from_slice(&fill_quant_row(GgmlType::Q8_0, k, &mut rng));
        }
        let d_x = DeviceBuffer::alloc(0, x.len() * 4).unwrap();
        let d_w = DeviceBuffer::alloc(0, w.len()).unwrap();
        let d_y = DeviceBuffer::alloc(0, t * h * n * 4).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&x)).unwrap();
        d_w.copy_from_host(&w).unwrap();
        unsafe {
            hy4::batched_heads_q8_mma(
                d_w.ptr,
                d_x.ptr as *const f32,
                d_y.ptr as *mut f32,
                t as i32,
                h as i32,
                n as i32,
                k as i32,
                ld_t,
                ld_h,
                h * n,
                n,
                &stream,
            );
        }
        stream.sync().unwrap();
        let mut got = vec![0.0f32; t * h * n];
        d_y.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
        // Verify sampled rows thoroughly (full exactness across arbitrary T=73/67 non-multiple of 64)
        let mut exp = vec![0.0f32; got.len()];
        for ti in [0usize, 1, 31, 63, 64, t - 1] {
            for hi in [0usize, 15, h - 1] {
                let xh = &x[ti * ld_t + hi * ld_h..ti * ld_t + hi * ld_h + k];
                for ni in 0..n {
                    let row = &w[(hi * n + ni) * rb..(hi * n + ni + 1) * rb];
                    exp[(ti * h + hi) * n + ni] = vec_dot_q_f32_ref(GgmlType::Q8_0, row, xh);
                }
            }
        }
        let mut got_sub = Vec::new();
        let mut exp_sub = Vec::new();
        for ti in [0usize, 1, 31, 63, 64, t - 1] {
            for hi in [0usize, 15, h - 1] {
                let off = (ti * h + hi) * n;
                got_sub.extend_from_slice(&got[off..off + n]);
                exp_sub.extend_from_slice(&exp[off..off + n]);
            }
        }
        let re = rel_err(&got_sub, &exp_sub);
        println!("batched_heads_q8_mma T={t} H={h} N={n} K={k}: rel_err={re:.6}");
        assert!(re < 2e-3, "batched_heads_q8_mma rel_err={re}");
    }

    // 2. hy4_mla_prefill MMA path (H=32 and H=64, both F16 and Q8_0 KV, SWA + DSA partial tile nt < 32)
    let kv_lora = 512usize;
    let n_rot = 64usize;
    let row_elems = kv_lora + n_rot;
    let seq_len = 79usize; // exercises full tiles (32, 32) AND partial tile (15 < 32)
    let ring_cap = 96usize;
    let kvr: Vec<f32> = (0..seq_len * row_elems).map(|_| rng.next_f32() * 0.4).collect();
    let kv_norm_w: Vec<f32> = (0..kv_lora).map(|_| 0.8 + 0.3 * rng.next_f32().abs()).collect();
    let pos_all: Vec<i32> = (0..seq_len as i32).collect();
    let base_rope = 10000.0f32;
    let eps = 1e-5f32;
    let scale = 1.0f32 / (256.0f32).sqrt();

    let d_kvr = DeviceBuffer::alloc(0, kvr.len() * 4).unwrap();
    let d_kvn = DeviceBuffer::alloc(0, kv_norm_w.len() * 4).unwrap();
    let d_pos_all = DeviceBuffer::alloc(0, pos_all.len() * 4).unwrap();
    d_kvr.copy_from_host(bytemuck::cast_slice(&kvr)).unwrap();
    d_kvn.copy_from_host(bytemuck::cast_slice(&kv_norm_w)).unwrap();
    d_pos_all.copy_from_host(bytemuck::cast_slice(&pos_all)).unwrap();

    for &h in &[32usize, 64usize] {
        let t_q = 4usize;
        let q: Vec<f32> = (0..t_q * h * row_elems).map(|_| rng.next_f32() * 0.4).collect();
        let sinks: Vec<f32> = (0..h).map(|i| -2.0 + 0.05 * i as f32).collect();
        let d_q = DeviceBuffer::alloc(0, q.len() * 4).unwrap();
        let d_sinks = DeviceBuffer::alloc(0, sinks.len() * 4).unwrap();
        let d_out = DeviceBuffer::alloc(0, t_q * h * kv_lora * 4).unwrap();
        d_q.copy_from_host(bytemuck::cast_slice(&q)).unwrap();
        d_sinks.copy_from_host(bytemuck::cast_slice(&sinks)).unwrap();

        let dsa_nk = [45i32, 79i32, 17i32, 32i32];
        let ldk = 80usize;
        let mut dsa_keys = vec![0i32; t_q * ldk];
        for ti in 0..t_q {
            for j in 0..dsa_nk[ti] as usize {
                dsa_keys[ti * ldk + j] = ((j * 7 + ti) % seq_len) as i32;
            }
        }
        let d_keys = DeviceBuffer::alloc(0, dsa_keys.len() * 4).unwrap();
        let d_nk = DeviceBuffer::alloc(0, dsa_nk.len() * 4).unwrap();
        d_keys.copy_from_host(bytemuck::cast_slice(&dsa_keys)).unwrap();
        d_nk.copy_from_host(bytemuck::cast_slice(&dsa_nk)).unwrap();

        for &is_q8 in &[false, true] {
            let rb = if is_q8 { hy4::kv_q8_row_bytes(kv_lora, n_rot) } else { row_elems * 2 };
            let d_cache = DeviceBuffer::alloc_zeroed(0, ring_cap * rb).unwrap();
            unsafe {
                hy4::kv_store(
                    d_kvr.ptr as *const f32,
                    seq_len as i32,
                    kv_lora as i32,
                    n_rot as i32,
                    d_kvn.ptr as *const f32,
                    eps,
                    d_pos_all.ptr as *const i32,
                    base_rope,
                    d_cache.ptr,
                    ring_cap as i32,
                    is_q8,
                    rb as i32,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let mut cache_bytes = vec![0u8; ring_cap * rb];
            d_cache.copy_to_host(&mut cache_bytes).unwrap();
            let mut kv_f32 = vec![0.0f32; ring_cap * row_elems];
            for s in 0..ring_cap {
                let row = &cache_bytes[s * rb..(s + 1) * rb];
                if is_q8 {
                    for b in 0..16 {
                        let d = f16::from_bits(u16::from_le_bytes([row[512 + 2 * b], row[512 + 2 * b + 1]])).to_f32();
                        for i in 0..32 {
                            kv_f32[s * row_elems + b * 32 + i] = d * (row[b * 32 + i] as i8 as f32);
                        }
                    }
                    for j in 0..n_rot {
                        let o = 512 + 32 + j * 2;
                        kv_f32[s * row_elems + kv_lora + j] =
                            f16::from_bits(u16::from_le_bytes([row[o], row[o + 1]])).to_f32();
                    }
                } else {
                    for j in 0..row_elems {
                        kv_f32[s * row_elems + j] =
                            f16::from_bits(u16::from_le_bytes([row[2 * j], row[2 * j + 1]])).to_f32();
                    }
                }
            }
            unsafe {
                hy4::mla_prefill(
                    d_q.ptr as *const f32,
                    t_q as i32,
                    h as i32,
                    d_cache.ptr,
                    ring_cap as i32,
                    d_keys.ptr as *const i32,
                    ldk as i32,
                    std::ptr::null(),
                    d_nk.ptr as *const i32,
                    scale,
                    d_sinks.ptr as *const f32,
                    d_out.ptr as *mut f32,
                    is_q8,
                    rb as i32,
                    &stream,
                );
            }
            stream.sync().unwrap();
            let mut got = vec![0.0f32; t_q * h * kv_lora];
            d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got)).unwrap();
            let mut exp = vec![0.0f32; got.len()];
            for ti in 0..t_q {
                for hi in 0..h {
                    let qr = &q[(ti * h + hi) * row_elems..(ti * h + hi + 1) * row_elems];
                    let mut m = sinks[hi];
                    let mut scs = Vec::with_capacity(dsa_nk[ti] as usize);
                    for j in 0..dsa_nk[ti] as usize {
                        let slot = (dsa_keys[ti * ldk + j] as usize) % ring_cap;
                        let kr = &kv_f32[slot * row_elems..(slot + 1) * row_elems];
                        let s: f32 = qr.iter().zip(kr.iter()).map(|(a, b)| a * b).sum::<f32>() * scale;
                        m = m.max(s);
                        scs.push((slot, s));
                    }
                    let mut den = (sinks[hi] - m).exp();
                    let mut acc = vec![0.0f32; kv_lora];
                    for (slot, s) in scs {
                        let p = (s - m).exp();
                        den += p;
                        let vr = &kv_f32[slot * row_elems..slot * row_elems + kv_lora];
                        for d in 0..kv_lora {
                            acc[d] += p * vr[d];
                        }
                    }
                    for d in 0..kv_lora {
                        exp[(ti * h + hi) * kv_lora + d] = acc[d] / den;
                    }
                }
            }
            let re = rel_err(&got, &exp);
            println!("mla_prefill MMA H={h} q8={is_q8}: rel_err={re:.6}");
            assert!(re < 5e-3, "mla_prefill MMA H={h} q8={is_q8} rel_err={re}");
        }
    }
}
