use ampllm_cuda_kernels::ffi::*;
use ampllm_cuda_kernels::{CudaStream, DeviceBuffer};
use ampllm_gguf::{dequantize_row_to_f32, vec_dot_q_f32_ref, GgmlType};
use half::f16;

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

fn swiglu_clamp_ref(g: f32, u: f32, lim: f32) -> f32 {
    let gc = g.min(lim);
    let uc = u.clamp(-lim, lim);
    (gc / (1.0 + (-gc).exp())) * uc
}

fn fill_quant_row(ty: GgmlType, k: usize, rng: &mut Rng) -> Vec<u8> {
    let nb = k / ty.block_size();
    let bsz = ty.type_size();
    let mut out = vec![0u8; nb * bsz];
    for b in 0..nb {
        let blk = &mut out[b * bsz..(b + 1) * bsz];
        match ty {
            GgmlType::Q5_0 => {
                blk[0..2].copy_from_slice(&f16::from_f32(0.015 + (b as f32) * 0.0002).to_le_bytes());
                for x in &mut blk[2..22] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Iq4Nl => {
                blk[0..2].copy_from_slice(&f16::from_f32(0.012 + (b as f32) * 0.0002).to_le_bytes());
                for x in &mut blk[2..18] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Iq3S => {
                blk[0..2].copy_from_slice(&f16::from_f32(0.012 + (b as f32) * 0.0001).to_le_bytes());
                for x in &mut blk[2..110] {
                    *x = rng.next_u8();
                }
            }
            GgmlType::Q8_0 => {
                blk[0..2].copy_from_slice(&f16::from_f32(0.015 + (b as f32) * 0.0002).to_le_bytes());
                for x in &mut blk[2..34] {
                    *x = rng.next_u8();
                }
            }
            _ => unreachable!("unsupported test quant {ty:?}"),
        }
    }
    out
}

#[test]
fn test_iq4_nl_q5_0_iq3_s_linear_mmq_dequant_parity() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x5149_4f35_304e_4c01);

    // 1. dequant_to_f16 parity (Iq4Nl, Q5_0)
    {
        let k = 512usize;
        let n = 64usize;
        let d_dst = DeviceBuffer::alloc(0, n * k * 2).unwrap();
        let mut got_u16 = vec![0u16; n * k];
        for ty in [GgmlType::Iq4Nl, GgmlType::Q5_0] {
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

    // 2. linear_q8_1 (MMVQ M=1 and M=4) + mmq_mma_tiled + mmq_mma_grouped (Iq4Nl, Q5_0)
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

        for ty in [GgmlType::Iq4Nl, GgmlType::Q5_0] {
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

            // MMVQ M=1 and M=4
            for mmvq_m in [1usize, 4usize] {
                unsafe {
                    launch_linear_q8_1(
                        ty as i32,
                        d_w.ptr,
                        d_xq8.ptr,
                        d_x.ptr as *const f32,
                        d_y.ptr as *mut f32,
                        n as i32,
                        k as i32,
                        mmvq_m as i32,
                        stream.raw,
                    );
                }
                stream.sync().unwrap();
                let mut got_mmvq = vec![0.0f32; mmvq_m * n];
                d_y.copy_to_host(bytemuck::cast_slice_mut(&mut got_mmvq)).unwrap();
                let re_mmvq = rel_err(&got_mmvq, &exp[..mmvq_m * n]);
                println!("linear_q8_1 {ty:?} m={mmvq_m}: rel_err={re_mmvq:.6}");
                assert!(re_mmvq < 0.015, "linear_q8_1 {ty:?} m={mmvq_m} failed: {re_mmvq}");
            }

            // Tiled MMQ MMA
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

            // Grouped MMQ MMA
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
            let d_ptrs = DeviceBuffer::alloc(0, 16).unwrap();
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
}

#[test]
fn test_moe_iq4_nl_q5_0_iq3_s_dispatch_parity() {
    let _g = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let stream = CudaStream::new(0).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);

    let configs = [
        (GgmlType::Iq3S, GgmlType::Iq4Nl, 512usize, 256usize),
        (GgmlType::Iq4Nl, GgmlType::Iq4Nl, 2560usize, 640usize),
        (GgmlType::Q5_0, GgmlType::Q5_0, 2560usize, 640usize),
        (GgmlType::Iq3S, GgmlType::Iq4Nl, 2560usize, 640usize),
    ];

    for (gu_ty, d_ty, n_embd, n_ff) in configs {
        let m = 2usize;
        let topk = 4usize;
        let n_ff_sh = 256usize;
        let clamp_lim = 10.0f32;

        let rb_gu = gu_ty.row_bytes(n_embd);
        let rb_d = d_ty.row_bytes(n_ff);
        let u_off = n_ff * rb_gu;
        let d_off = 2 * n_ff * rb_gu;
        let exp_bytes = d_off + n_embd * rb_d;

        let mut blobs = Vec::with_capacity(4);
        for _ in 0..4 {
            let mut b = Vec::with_capacity(exp_bytes);
            for _ in 0..(2 * n_ff) {
                b.extend_from_slice(&fill_quant_row(gu_ty, n_embd, &mut rng));
            }
            for _ in 0..n_embd {
                b.extend_from_slice(&fill_quant_row(d_ty, n_ff, &mut rng));
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
        let mut exp_ids = vec![0i32; m * topk];
        let mut router_w = vec![0.0f32; m * topk];
        for mi in 0..m {
            for s in 0..topk {
                let eid = (mi + s) % 4;
                plan_ptrs[mi * topk + s] = d_blobs[eid].as_u64();
                exp_ids[mi * topk + s] = eid as i32;
                router_w[mi * topk + s] = 0.1 + (s as f32) * 0.05;
            }
        }
        let g_ptrs_res: Vec<u64> = d_blobs.iter().map(|b| b.as_u64()).collect();
        let u_ptrs_res: Vec<u64> = d_blobs.iter().map(|b| b.as_u64() + u_off as u64).collect();
        let d_ptrs_res: Vec<u64> = d_blobs.iter().map(|b| b.as_u64() + d_off as u64).collect();

        let d_plan = DeviceBuffer::alloc(0, plan_ptrs.len() * 8).unwrap();
        let d_eids = DeviceBuffer::alloc(0, exp_ids.len() * 4).unwrap();
        let d_rw = DeviceBuffer::alloc(0, router_w.len() * 4).unwrap();
        let d_gp = DeviceBuffer::alloc(0, g_ptrs_res.len() * 8).unwrap();
        let d_up = DeviceBuffer::alloc(0, u_ptrs_res.len() * 8).unwrap();
        let d_dp = DeviceBuffer::alloc(0, d_ptrs_res.len() * 8).unwrap();
        d_plan.copy_from_host(bytemuck::cast_slice(&plan_ptrs)).unwrap();
        d_eids.copy_from_host(bytemuck::cast_slice(&exp_ids)).unwrap();
        d_rw.copy_from_host(bytemuck::cast_slice(&router_w)).unwrap();
        d_gp.copy_from_host(bytemuck::cast_slice(&g_ptrs_res)).unwrap();
        d_up.copy_from_host(bytemuck::cast_slice(&u_ptrs_res)).unwrap();
        d_dp.copy_from_host(bytemuck::cast_slice(&d_ptrs_res)).unwrap();

        let x: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32() * 0.4).collect();
        let d_x = DeviceBuffer::alloc(0, x.len() * 4).unwrap();
        d_x.copy_from_host(bytemuck::cast_slice(&x)).unwrap();
        let d_xq8 = DeviceBuffer::alloc(0, m * (n_embd / 32) * 36).unwrap();
        let d_iq8 = DeviceBuffer::alloc(0, m * topk * (n_ff / 32) * 36).unwrap();

        let sh_rb = GgmlType::Q8_0.row_bytes(n_ff_sh);
        let mut sh_w = Vec::with_capacity(n_embd * sh_rb);
        for _ in 0..n_embd {
            sh_w.extend_from_slice(&fill_quant_row(GgmlType::Q8_0, n_ff_sh, &mut rng));
        }
        let d_sh_w = DeviceBuffer::alloc(0, sh_w.len()).unwrap();
        d_sh_w.copy_from_host(&sh_w).unwrap();
        let sh_h: Vec<f32> = (0..m * n_ff_sh).map(|_| rng.next_f32() * 0.4).collect();
        let d_sh_h = DeviceBuffer::alloc(0, sh_h.len() * 4).unwrap();
        d_sh_h.copy_from_host(bytemuck::cast_slice(&sh_h)).unwrap();
        let d_sh_hq = DeviceBuffer::alloc(0, m * (n_ff_sh / 32) * 36).unwrap();

        let d_sh_out = DeviceBuffer::alloc(0, m * n_embd * 4).unwrap();
        let d_out = DeviceBuffer::alloc(0, m * n_embd * 4).unwrap();

        unsafe {
            launch_quantize_q8_1(d_x.ptr as *const f32, d_xq8.ptr, m as i32, n_embd as i32, stream.raw);
            launch_quantize_q8_1(d_sh_h.ptr as *const f32, d_sh_hq.ptr, m as i32, n_ff_sh as i32, stream.raw);
            launch_glm5_moe_plan_filtered(
                gu_ty as i32,
                d_ty as i32,
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
        let mut got_plan = vec![0.0f32; m * n_embd];
        d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_plan)).unwrap();

        let mut exp_with_sh = vec![0.0f32; m * n_embd];
        let mut exp_routed = vec![0.0f32; m * n_embd];
        for mi in 0..m {
            let xr = &x[mi * n_embd..(mi + 1) * n_embd];
            for r in 0..n_embd {
                exp_with_sh[mi * n_embd + r] += vec_dot_q_f32_ref(
                    GgmlType::Q8_0,
                    &sh_w[r * sh_rb..(r + 1) * sh_rb],
                    &sh_h[mi * n_ff_sh..(mi + 1) * n_ff_sh],
                );
            }
            for s in 0..topk {
                let blob = &blobs[(mi + s) % 4];
                let mut h = vec![0.0f32; n_ff];
                for r in 0..n_ff {
                    let g = vec_dot_q_f32_ref(gu_ty, &blob[r * rb_gu..(r + 1) * rb_gu], xr);
                    let u = vec_dot_q_f32_ref(gu_ty, &blob[u_off + r * rb_gu..u_off + (r + 1) * rb_gu], xr);
                    h[r] = swiglu_clamp_ref(g, u, clamp_lim);
                }
                let rw = router_w[mi * topk + s];
                for r in 0..n_embd {
                    let dv = vec_dot_q_f32_ref(d_ty, &blob[d_off + r * rb_d..d_off + (r + 1) * rb_d], &h);
                    exp_routed[mi * n_embd + r] += rw * dv;
                    exp_with_sh[mi * n_embd + r] += rw * dv;
                }
            }
        }
        let re_plan = rel_err(&got_plan, &exp_with_sh);
        println!("moe_plan_filtered ({gu_ty:?}/{d_ty:?} {n_embd}->{n_ff}): rel_err={re_plan:.6}");
        assert!(re_plan < 0.02, "moe_plan_filtered ({gu_ty:?}/{d_ty:?}) failed: {re_plan}");

        // Resident path
        d_out.zero().unwrap();
        unsafe {
            launch_glm5_moe_resident(
                gu_ty as i32,
                d_ty as i32,
                d_gp.ptr as *const u64,
                d_up.ptr as *const u64,
                d_dp.ptr as *const u64,
                rb_gu,
                rb_d,
                d_xq8.ptr,
                d_eids.ptr as *const i32,
                d_rw.ptr as *const f32,
                d_iq8.ptr,
                std::ptr::null_mut(),
                d_out.ptr as *mut f32,
                n_embd as i32,
                n_ff as i32,
                topk as i32,
                m as i32,
                clamp_lim,
                std::ptr::null(),
                stream.raw,
            );
        }
        stream.sync().unwrap();
        let mut got_res = vec![0.0f32; m * n_embd];
        d_out.copy_to_host(bytemuck::cast_slice_mut(&mut got_res)).unwrap();
        let re_res = rel_err(&got_res, &exp_routed);
        println!("moe_resident ({gu_ty:?}/{d_ty:?} {n_embd}->{n_ff}): rel_err={re_res:.6}");
        assert!(re_res < 0.02, "moe_resident ({gu_ty:?}/{d_ty:?}) failed: {re_res}");

        // Prefill small_gu + small_down
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
                gu_ty as i32,
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
                d_ty as i32,
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
        let re_sm = rel_err(&got_sm, &exp_routed);
        let re_res_vs_sm = rel_err(&got_res, &got_sm);
        println!("moe_prefill_small ({gu_ty:?}/{d_ty:?} {n_embd}->{n_ff}): rel_err={re_sm:.6}, res_vs_sm={re_res_vs_sm:.6}");
        assert!(re_sm < 0.02, "moe_prefill_small ({gu_ty:?}/{d_ty:?}) failed: {re_sm}");
        assert!(re_res_vs_sm < 2e-4, "resident vs prefill_small mismatch ({gu_ty:?}/{d_ty:?}): {re_res_vs_sm}");
    }
}
