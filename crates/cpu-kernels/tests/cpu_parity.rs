use ampllm_cpu_kernels::{
    active_simd_tier, quantize_f32_to_q8_1_cpu, quantize_f32_to_q8_k_cpu, quantize_glm5_cpu_x_q8k,
    run_cpu_moe_m1, run_glm5_cpu_moe_grouped, run_glm5_cpu_moe_grouped_prequant,
    run_glm5_cpu_moe_m1, vec_dot_q_q8_1, vec_dot_scalar, CpuBlockQ8K, CpuBlockQ8_1, CpuExpertJob,
    CpuMoeScratch, CpuThreadPool, Glm5CpuExpertJob, Glm5CpuGroupedExpert, Glm5CpuScratch,
};
use ampllm_gguf::{dequantize_row_to_f32, vec_dot_q_f32_ref, GgmlType, KVALUES_IQ4NL};
use half::f16;
use std::time::Instant;

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        let u = (self.next_u32() as f64) / (u32::MAX as f64);
        (u * 2.0 - 1.0) as f32
    }
}

fn fill_random_quant_rows(ty: GgmlType, n: usize, k: usize, seed: u64) -> Vec<u8> {
    let mut rng = Rng(seed);
    let rb = ty.row_bytes(k);
    let mut out = vec![0u8; n * rb];
    for row in 0..n {
        let dst = &mut out[row * rb..(row + 1) * rb];
        match ty {
            GgmlType::F32 => {
                let s: &mut [f32] = bytemuck::cast_slice_mut(dst);
                for v in s.iter_mut() {
                    *v = rng.next_f32() * 0.5;
                }
            }
            GgmlType::Q8_0 => {
                let nb = k / 32;
                for b in 0..nb {
                    let off = b * 34;
                    let d = f16::from_f32(0.02 + rng.next_f32().abs() * 0.08).to_le_bytes();
                    dst[off..off + 2].copy_from_slice(&d);
                    for j in 2..34 {
                        dst[off + j] = (rng.next_u32() & 0xFF) as u8;
                    }
                }
            }
            GgmlType::Q4_0 | GgmlType::Iq4Nl => {
                let nb = k / 32;
                for b in 0..nb {
                    let off = b * 18;
                    let d = f16::from_f32(0.02 + rng.next_f32().abs() * 0.08).to_le_bytes();
                    dst[off..off + 2].copy_from_slice(&d);
                    for j in 2..18 {
                        dst[off + j] = (rng.next_u32() & 0xFF) as u8;
                    }
                }
            }
            GgmlType::Nvfp4 => {
                let nb = k / 64;
                for b in 0..nb {
                    let off = b * 36;
                    for sb in 0..4 {
                        dst[off + sb] = 0x38 + ((rng.next_u32() & 0x07) as u8);
                    }
                    for j in 4..36 {
                        dst[off + j] = (rng.next_u32() & 0xFF) as u8;
                    }
                }
            }
            GgmlType::Q3K
            | GgmlType::Q4K
            | GgmlType::Q5K
            | GgmlType::Q6K
            | GgmlType::Iq2S
            | GgmlType::Iq3Xxs
            | GgmlType::Iq3S
            | GgmlType::Iq4Xs => {
                for byte in dst.iter_mut() {
                    *byte = (rng.next_u32() & 0xFF) as u8;
                }
                let block_sz = ty.row_bytes(256);
                let nb = k / 256;
                for b in 0..nb {
                    let boff = b * block_sz;
                    let d_val = f16::from_f32(0.03 + rng.next_f32().abs() * 0.05).to_le_bytes();
                    match ty {
                        GgmlType::Q6K => dst[boff + 208..boff + 210].copy_from_slice(&d_val),
                        GgmlType::Q3K => dst[boff + 108..boff + 110].copy_from_slice(&d_val),
                        _ => {
                            dst[boff..boff + 2].copy_from_slice(&d_val);
                            if matches!(ty, GgmlType::Q4K | GgmlType::Q5K) {
                                let dm = f16::from_f32(0.01 + rng.next_f32().abs() * 0.03).to_le_bytes();
                                dst[boff + 2..boff + 4].copy_from_slice(&dm);
                            }
                        }
                    }
                }
            }
            _ => unreachable!(),
        }
    }
    out
}

#[test]
fn test_cpu_simd_quant_parity_and_iq4nl_bench() {
    let tier = active_simd_tier();
    println!("CPU SIMD tier: {tier:?}");
    assert_eq!((KVALUES_IQ4NL[0] as i32 + 128) as u8, 1);

    let k = 2304usize;
    let n = 64usize;
    let mut rng = Rng(12345);
    let act: Vec<f32> = (0..k).map(|_| rng.next_f32() * 1.5).collect();
    let mut act_q8 = vec![CpuBlockQ8_1::default(); k / 32];
    quantize_f32_to_q8_1_cpu(&act, &mut act_q8);

    let types = [
        GgmlType::Q8_0,
        GgmlType::Iq4Nl,
        GgmlType::Q4_0,
        GgmlType::Iq4Xs,
        GgmlType::Q4K,
        GgmlType::Q5K,
        GgmlType::Q6K,
        GgmlType::Nvfp4,
        GgmlType::F32,
    ];

    for ty in types {
        let w = fill_random_quant_rows(ty, n, k, 900 + ty as u64);
        let rb = ty.row_bytes(k);
        let mut ref_norm = 0.0f64;
        let mut diff_norm = 0.0f64;
        for row in 0..n {
            let row_bytes = &w[row * rb..(row + 1) * rb];
            let r = vec_dot_q_f32_ref(ty, row_bytes, &act);
            let simd = vec_dot_q_q8_1(ty, row_bytes, &act_q8, Some(&act));
            let scal = vec_dot_scalar(ty, row_bytes, &act_q8, Some(&act));
            assert!(
                (simd - scal).abs() <= 1e-3 * scal.abs().max(1.0),
                "SIMD vs scalar mismatch for {ty:?} row {row}: simd={simd} scalar={scal}"
            );
            ref_norm += (r as f64) * (r as f64);
            let d = (simd - r) as f64;
            diff_norm += d * d;
        }
        let rel = (diff_norm / ref_norm.max(1e-12)).sqrt();
        println!("CPU parity {ty:?}: max_rel_norm = {rel:.6}");
        assert!(rel < 1e-2, "{ty:?} rel error {rel} too high");
    }

    // Benchmark 1 cold IQ4_NL Inkling-2-Mini expert (n_embd=2304, n_ff=768) on physical-core spin pool
    let pool = CpuThreadPool::new(None);
    println!(
        "CpuThreadPool initialized: threads={} physical_cores={:?}",
        pool.num_threads(),
        pool.physical_cores()
    );
    let n_embd = 2304usize;
    let n_ff = 768usize;
    let gu = fill_random_quant_rows(GgmlType::Iq4Nl, 2 * n_ff, n_embd, 42);
    let dn = fill_random_quant_rows(GgmlType::Iq4Nl, n_embd, n_ff, 43);
    let mut scratch = CpuMoeScratch::new(n_embd, n_ff, 4);
    let mut out = vec![0.0f32; n_embd];
    let jobs = [CpuExpertJob {
        gate_up: &gu,
        down: &dn,
        gate_up_scale: 1.0,
        weight: 0.75,
    }];

    for _ in 0..20 {
        run_cpu_moe_m1(
            &pool,
            &mut scratch,
            GgmlType::Iq4Nl,
            GgmlType::Iq4Nl,
            n_embd,
            n_ff,
            4.0,
            25.0,
            &act,
            &jobs,
            &mut out,
        );
    }
    let iters = 200;
    let t0 = Instant::now();
    for _ in 0..iters {
        run_cpu_moe_m1(
            &pool,
            &mut scratch,
            GgmlType::Iq4Nl,
            GgmlType::Iq4Nl,
            n_embd,
            n_ff,
            4.0,
            25.0,
            &act,
            &jobs,
            &mut out,
        );
    }
    let us_per_exp = t0.elapsed().as_secs_f64() * 1e6 / (iters as f64);
    println!("CPU single IQ4_NL expert (2304x768) latency: {us_per_exp:.1} us");
}

#[test]
fn test_glm5_avx512_iq_and_q_k_parity_and_16core_bw() {
    if !ampllm_cpu_kernels::glm5_cpu::has_glm5_avx512() {
        eprintln!("skipping test_glm5_avx512_iq_and_q_k_parity_and_16core_bw: CPU lacks AVX-512 VNNI/VPOPCNTDQ");
        return;
    }
    let n_embd = 4096usize;
    let n_ff = 2048usize;
    let clamp = 10.0f32;
    let mut rng = Rng(77777);

    let fp32_moe = |gu_ty: GgmlType,
                    dn_ty: GgmlType,
                    gate_w: &[u8],
                    up_w: &[u8],
                    dn_w: &[u8],
                    x: &[f32],
                    w: f32|
     -> Vec<f32> {
        let gu_rb = gu_ty.row_bytes(n_embd);
        let dn_rb = dn_ty.row_bytes(n_ff);
        let mut ff = vec![0.0f32; n_ff];
        let mut row_f = vec![0.0f32; n_embd];
        for r in 0..n_ff {
            dequantize_row_to_f32(gu_ty, &gate_w[r * gu_rb..(r + 1) * gu_rb], &mut row_f);
            let mut g: f32 = row_f.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            dequantize_row_to_f32(gu_ty, &up_w[r * gu_rb..(r + 1) * gu_rb], &mut row_f);
            let mut u: f32 = row_f.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            g = g.min(clamp);
            u = u.clamp(-clamp, clamp);
            ff[r] = (g / (1.0 + (-g).exp())) * u;
        }
        let mut out = vec![0.0f32; n_embd];
        let mut d_row = vec![0.0f32; n_ff];
        for r in 0..n_embd {
            dequantize_row_to_f32(dn_ty, &dn_w[r * dn_rb..(r + 1) * dn_rb], &mut d_row);
            let s: f32 = d_row.iter().zip(ff.iter()).map(|(&a, &b)| a * b).sum();
            out[r] = w * s;
        }
        out
    };

    let pool16 = CpuThreadPool::new_all_physical(None);
    println!(
        "GLM5 CpuThreadPool: threads={} cores={:?}",
        pool16.num_threads(),
        pool16.physical_cores()
    );

    // 1) Parity test for (IQ2_S gate/up, IQ3_XXS down), (IQ2_S gate/up, IQ3_S down), (Q3_K gate/up, Q4_K down)
    for (gu_ty, dn_ty, seed) in [
        (GgmlType::Iq2S, GgmlType::Iq3Xxs, 101u64),
        (GgmlType::Iq2S, GgmlType::Iq3S, 202u64),
        (GgmlType::Q3K, GgmlType::Q4K, 303u64),
    ] {
        let gate_w = fill_random_quant_rows(gu_ty, n_ff, n_embd, seed);
        let up_w = fill_random_quant_rows(gu_ty, n_ff, n_embd, seed + 1);
        let dn_w = fill_random_quant_rows(dn_ty, n_embd, n_ff, seed + 2);
        let x: Vec<f32> = (0..n_embd).map(|_| rng.next_f32() * 0.25).collect();
        let w = 0.65f32;

        let ref_out = fp32_moe(gu_ty, dn_ty, &gate_w, &up_w, &dn_w, &x, w);
        let mut scratch = Glm5CpuScratch::new(n_embd, n_ff, 8);
        let mut got_m1 = vec![0.0f32; n_embd];
        let jobs = [Glm5CpuExpertJob {
            gate_ptr: gate_w.as_ptr(),
            up_ptr: up_w.as_ptr(),
            down_ptr: dn_w.as_ptr(),
            weight: w,
        }];
        run_glm5_cpu_moe_m1(
            &pool16,
            &mut scratch,
            gu_ty,
            dn_ty,
            n_embd,
            n_ff,
            clamp,
            &x,
            None,
            &jobs,
            &mut got_m1,
        );

        let ref_norm: f64 = ref_out.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let diff_norm: f64 = ref_out
            .iter()
            .zip(got_m1.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        let rel = (diff_norm / ref_norm.max(1e-12)).sqrt();
        println!("GLM5 M=1 CPU MoE ({gu_ty:?}/{dn_ty:?}) rel_err = {rel:.6}");
        assert!(rel < 1.5e-2, "rel error {rel} too high for {gu_ty:?}/{dn_ty:?}");

        // Test grouped wide path (NT=8) consistency with M=1
        if gu_ty == GgmlType::Iq2S && dn_ty == GgmlType::Iq3Xxs {
            let x8: Vec<f32> = (0..8 * n_embd)
                .map(|i| x[i % n_embd] * (1.0 + 0.05 * ((i / n_embd) as f32)))
                .collect();
            let toks: Vec<usize> = (0..8).collect();
            let ws = [w; 8];
            let grp = [Glm5CpuGroupedExpert {
                gate_ptr: gate_w.as_ptr(),
                up_ptr: up_w.as_ptr(),
                down_ptr: dn_w.as_ptr(),
                token_indices: &toks,
                token_weights: &ws,
            }];
            let mut got8 = vec![0.0f32; 8 * n_embd];
            run_glm5_cpu_moe_grouped(
                &pool16,
                &mut scratch,
                gu_ty,
                dn_ty,
                n_embd,
                n_ff,
                clamp,
                8,
                &x8,
                &grp,
                &mut got8,
                true,
            );
            let d0: f64 = got_m1
                .iter()
                .zip(got8[..n_embd].iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            let n0: f64 = got_m1.iter().map(|&v| (v as f64).powi(2)).sum();
            let g_rel = (d0 / n0.max(1e-12)).sqrt();
            println!("GLM5 grouped NT=8 vs M=1 tok0 rel_err = {g_rel:.6}");
            assert!(g_rel < 5e-4, "grouped tok0 diverged: {g_rel}");
        }
    }

    // 2) Benchmark DRAM-sized pool of 64 distinct GLM-5.3 IQ2_S/IQ3_XXS experts (~550 MB > L3 cache!)
    let n_pool = 64usize;
    let exp_gu_bytes = GgmlType::Iq2S.row_bytes(n_embd) * n_ff;
    let exp_dn_bytes = GgmlType::Iq3Xxs.row_bytes(n_ff) * n_embd;
    let one_exp_bytes = 2 * exp_gu_bytes + exp_dn_bytes;
    let mut big_arena = vec![0x35u8; n_pool * one_exp_bytes];
    // Touch pages and set valid scales
    for e in 0..n_pool {
        let base = e * one_exp_bytes;
        let f16_one = f16::from_f32(0.05).to_le_bytes();
        for b in 0..(2 * n_ff * (n_embd / 256)) {
            let off = base + b * 82;
            big_arena[off..off + 2].copy_from_slice(&f16_one);
        }
        for b in 0..(n_embd * (n_ff / 256)) {
            let off = base + 2 * exp_gu_bytes + b * 98;
            big_arena[off..off + 2].copy_from_slice(&f16_one);
        }
    }
    let x: Vec<f32> = (0..n_embd).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let mut out = vec![0.0f32; n_embd];
    let mut scratch = Glm5CpuScratch::new(n_embd, n_ff, 8);
    let mut _qtmp = vec![CpuBlockQ8K::default(); n_embd / 256];
    quantize_f32_to_q8_k_cpu(&x, &mut _qtmp);

    let all_phys = ampllm_cpu_kernels::pool::discover_physical_cores_with_options(None, false);
    let (ccd0, ccd1) = ampllm_cpu_kernels::pool::discover_cross_ccd_interleaved_cores();
    let half = (all_phys.len() / 2).max(1);
    let mut configs: Vec<(String, Option<Vec<usize>>)> = vec![
        ("default (excl cpu0)".to_owned(), None),
        (format!("{}t (all phys)", all_phys.len()), Some(all_phys.clone())),
        (format!("{half}t half-phys"), Some(all_phys[..half].to_vec())),
    ];
    if !ccd0.is_empty() {
        configs.push((format!("{}t cross-CCD-0", ccd0.len()), Some(ccd0)));
    }
    if !ccd1.is_empty() {
        configs.push((format!("{}t cross-CCD-1", ccd1.len()), Some(ccd1)));
    }
    for (label, cores_opt) in configs {
        let pool = match cores_opt {
            None => CpuThreadPool::new(None),
            Some(cores) => CpuThreadPool::from_cores(cores),
        };
        let rounds = 16usize;
        let k_per_round = 4usize; // 4 misses/layer * 16 rounds = 64 distinct experts (~550MB > L3!)
        let mut all_jobs = Vec::with_capacity(rounds * k_per_round);
        for e in 0..(rounds * k_per_round) {
            let base = unsafe { big_arena.as_ptr().add((e % n_pool) * one_exp_bytes) };
            all_jobs.push(Glm5CpuExpertJob {
                gate_ptr: base,
                up_ptr: unsafe { base.add(exp_gu_bytes) },
                down_ptr: unsafe { base.add(2 * exp_gu_bytes) },
                weight: 0.25,
            });
        }
        // warmup
        for r in 0..4 {
            run_glm5_cpu_moe_m1(
                &pool,
                &mut scratch,
                GgmlType::Iq2S,
                GgmlType::Iq3Xxs,
                n_embd,
                n_ff,
                clamp,
                &x,
                None,
                &all_jobs[r * k_per_round..(r + 1) * k_per_round],
                &mut out,
            );
        }
        let t0 = Instant::now();
        let reps = 4usize;
        for _ in 0..reps {
            for r in 0..rounds {
                run_glm5_cpu_moe_m1(
                    &pool,
                    &mut scratch,
                    GgmlType::Iq2S,
                    GgmlType::Iq3Xxs,
                    n_embd,
                    n_ff,
                    clamp,
                    &x,
                    None,
                    &all_jobs[r * k_per_round..(r + 1) * k_per_round],
                    &mut out,
                );
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        let total_exps = (reps * rounds * k_per_round) as f64;
        let ms_per_exp = dt * 1e3 / total_exps;
        let gbps = ((one_exp_bytes as f64) * total_exps / dt) / 1e9;
        println!("GLM5 DRAM-resident IQ2_S+IQ3_XXS ({label}): {ms_per_exp:.4} ms/exp | {gbps:.2} GB/s");
    }
}

#[test]
fn test_glm5_wide_gu_and_prequant_parity() {
    if !ampllm_cpu_kernels::glm5_cpu::has_glm5_avx512() {
        eprintln!("skipping test_glm5_wide_gu_and_prequant_parity: CPU lacks AVX-512 VNNI/VPOPCNTDQ");
        return;
    }
    let n_embd = 4096usize;
    let n_ff = 2048usize;
    let clamp = 10.0f32;
    let m = 25usize;
    let pool = CpuThreadPool::new_all_physical(None);
    let mut rng = Rng(20261122);
    let x: Vec<f32> = (0..m * n_embd).map(|_| rng.next_f32() * 0.25).collect();

    for (gu_ty, dn_ty, seed) in [
        (GgmlType::Iq2S, GgmlType::Iq3Xxs, 1111u64),
        (GgmlType::Iq2S, GgmlType::Iq3S, 2222u64),
        (GgmlType::Iq3S, GgmlType::Iq3S, 3333u64),
    ] {
        let nt_list = [1usize, 4, 7, 16, 25];
        let mut gates = Vec::with_capacity(nt_list.len());
        let mut ups = Vec::with_capacity(nt_list.len());
        let mut dns = Vec::with_capacity(nt_list.len());
        let mut tok_idxs = Vec::with_capacity(nt_list.len());
        let mut tok_wts = Vec::with_capacity(nt_list.len());

        for (ei, &nt) in nt_list.iter().enumerate() {
            gates.push(fill_random_quant_rows(gu_ty, n_ff, n_embd, seed + ei as u64 * 10));
            ups.push(fill_random_quant_rows(gu_ty, n_ff, n_embd, seed + ei as u64 * 10 + 1));
            dns.push(fill_random_quant_rows(dn_ty, n_embd, n_ff, seed + ei as u64 * 10 + 2));
            let idxs: Vec<usize> = (0..nt).map(|k| (k * 3 + ei) % m).collect();
            let wts: Vec<f32> = (0..nt).map(|k| 0.15 + 0.05 * ((k + ei) % 7) as f32).collect();
            tok_idxs.push(idxs);
            tok_wts.push(wts);
        }

        let groups: Vec<Glm5CpuGroupedExpert<'_>> = (0..nt_list.len())
            .map(|ei| Glm5CpuGroupedExpert {
                gate_ptr: gates[ei].as_ptr(),
                up_ptr: ups[ei].as_ptr(),
                down_ptr: dns[ei].as_ptr(),
                token_indices: &tok_idxs[ei],
                token_weights: &tok_wts[ei],
            })
            .collect();

        // Reference via run_glm5_cpu_moe_m1 per (expert, token)
        let mut scratch_ref = Glm5CpuScratch::new(n_embd, n_ff, 4);
        let mut ref_out = vec![0.0f32; m * n_embd];
        let mut tmp_m1 = vec![0.0f32; n_embd];
        for ei in 0..groups.len() {
            for (&tok, &wt) in tok_idxs[ei].iter().zip(tok_wts[ei].iter()) {
                let job = [Glm5CpuExpertJob {
                    gate_ptr: gates[ei].as_ptr(),
                    up_ptr: ups[ei].as_ptr(),
                    down_ptr: dns[ei].as_ptr(),
                    weight: wt,
                }];
                run_glm5_cpu_moe_m1(
                    &pool,
                    &mut scratch_ref,
                    gu_ty,
                    dn_ty,
                    n_embd,
                    n_ff,
                    clamp,
                    &x[tok * n_embd..(tok + 1) * n_embd],
                    None,
                    &job,
                    &mut tmp_m1,
                );
                for r in 0..n_embd {
                    ref_out[tok * n_embd + r] += tmp_m1[r];
                }
            }
        }

        // Pre-quantize once, then run grouped prequant in two batches (zero_out=true, then false)
        let mut scratch = Glm5CpuScratch::new(n_embd, n_ff, 64);
        let mut got_out = vec![0.0f32; m * n_embd];
        quantize_glm5_cpu_x_q8k(&pool, &mut scratch, n_embd, m, &x);
        run_glm5_cpu_moe_grouped_prequant(
            &pool,
            &mut scratch,
            gu_ty,
            dn_ty,
            n_embd,
            n_ff,
            clamp,
            m,
            &groups[..2],
            &mut got_out,
            true,
        );
        run_glm5_cpu_moe_grouped_prequant(
            &pool,
            &mut scratch,
            gu_ty,
            dn_ty,
            n_embd,
            n_ff,
            clamp,
            m,
            &groups[2..],
            &mut got_out,
            false,
        );

        let ref_norm: f64 = ref_out.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let diff_norm: f64 = ref_out
            .iter()
            .zip(got_out.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        let rel = (diff_norm / ref_norm.max(1e-12)).sqrt();
        println!("GLM5 wide_gu + prequant ({gu_ty:?}/{dn_ty:?}) rel_err = {rel:.8}");
        assert!(rel < 1e-4, "wide_gu+prequant {gu_ty:?}/{dn_ty:?} rel_err={rel} >= 1e-4");
    }
}
