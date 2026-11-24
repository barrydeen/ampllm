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
            GgmlType::Q5_0 => {
                let nb = k / 32;
                for b in 0..nb {
                    let off = b * 22;
                    let d = f16::from_f32(0.02 + rng.next_f32().abs() * 0.06).to_le_bytes();
                    dst[off..off + 2].copy_from_slice(&d);
                    for j in 2..22 {
                        dst[off + j] = (rng.next_u32() & 0xFF) as u8;
                    }
                }
            }
            GgmlType::Q5_1 => {
                let nb = k / 32;
                for b in 0..nb {
                    let off = b * 24;
                    let d = f16::from_f32(0.02 + rng.next_f32().abs() * 0.06).to_le_bytes();
                    let m = f16::from_f32(-0.35 + rng.next_f32().abs() * 0.5).to_le_bytes();
                    dst[off..off + 2].copy_from_slice(&d);
                    dst[off + 2..off + 4].copy_from_slice(&m);
                    for j in 4..24 {
                        dst[off + j] = (rng.next_u32() & 0xFF) as u8;
                    }
                }
            }
            GgmlType::Q2_0 => {
                let nb = k / 64;
                for b in 0..nb {
                    let off = b * 18;
                    let d = f16::from_f32(0.03 + rng.next_f32().abs() * 0.06).to_le_bytes();
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
    println!("Testing GLM5 CPU ISA tier: {}", ampllm_cpu_kernels::active_cpu_isa());
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

#[test]
fn test_hy4_avx512_iq4xs_q5k_q6k_q80_parity() {
    let n_embd = 1536usize;
    let n_ff = 1536usize;
    let clamp = 7.0f32;
    let pool = CpuThreadPool::new_all_physical(None);
    let mut rng = Rng(20261123);

    let fp32_moe = |ty: GgmlType, gate_w: &[u8], up_w: &[u8], dn_w: &[u8], x: &[f32], w: f32| -> Vec<f32> {
        let gu_rb = ty.row_bytes(n_embd);
        let dn_rb = ty.row_bytes(n_ff);
        let mut ff = vec![0.0f32; n_ff];
        let mut row_f = vec![0.0f32; n_embd];
        for r in 0..n_ff {
            dequantize_row_to_f32(ty, &gate_w[r * gu_rb..(r + 1) * gu_rb], &mut row_f);
            let mut g: f32 = row_f.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            dequantize_row_to_f32(ty, &up_w[r * gu_rb..(r + 1) * gu_rb], &mut row_f);
            let mut u: f32 = row_f.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum();
            g = g.min(clamp);
            u = u.clamp(-clamp, clamp);
            ff[r] = (g / (1.0 + (-g).exp())) * u;
        }
        let mut out = vec![0.0f32; n_embd];
        let mut d_row = vec![0.0f32; n_ff];
        for r in 0..n_embd {
            dequantize_row_to_f32(ty, &dn_w[r * dn_rb..(r + 1) * dn_rb], &mut d_row);
            let s: f32 = d_row.iter().zip(ff.iter()).map(|(&a, &b)| a * b).sum();
            out[r] = w * s;
        }
        out
    };

    for (ty, seed) in [
        (GgmlType::Iq4Xs, 4100u64),
        (GgmlType::Q5K, 4200u64),
        (GgmlType::Q6K, 4300u64),
        (GgmlType::Q8_0, 4400u64),
    ] {
        let gate_w = fill_random_quant_rows(ty, n_ff, n_embd, seed);
        let up_w = fill_random_quant_rows(ty, n_ff, n_embd, seed + 1);
        let dn_w = fill_random_quant_rows(ty, n_embd, n_ff, seed + 2);
        let m_toks = 8usize;
        let x: Vec<f32> = (0..m_toks * n_embd).map(|_| rng.next_f32() * 0.3).collect();
        let w = 0.72f32;

        let mut scratch = Glm5CpuScratch::new(n_embd, n_ff, 16);
        let mut got_m1 = vec![0.0f32; n_embd];
        let job_m1 = [Glm5CpuExpertJob {
            gate_ptr: gate_w.as_ptr(),
            up_ptr: up_w.as_ptr(),
            down_ptr: dn_w.as_ptr(),
            weight: w,
        }];
        run_glm5_cpu_moe_m1(
            &pool,
            &mut scratch,
            ty,
            ty,
            n_embd,
            n_ff,
            clamp,
            &x[..n_embd],
            None,
            &job_m1,
            &mut got_m1,
        );

        let ref_m1 = fp32_moe(ty, &gate_w, &up_w, &dn_w, &x[..n_embd], w);
        let ref_norm: f64 = ref_m1.iter().map(|&v| (v as f64).powi(2)).sum();
        let diff_norm: f64 = ref_m1
            .iter()
            .zip(got_m1.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        let rel = (diff_norm / ref_norm.max(1e-12)).sqrt();

        // Exact check vs scalar dequantized weights using the exact same Q8_1 / Q8_K activations
        let q8k_moe = |ty: GgmlType, gate_w: &[u8], up_w: &[u8], dn_w: &[u8], x: &[f32], w: f32| -> Vec<f32> {
            use ampllm_cpu_kernels::glm5_cpu::{ampllm_cpu_quantize_q8_1_g8, CpuBlockQ8_1G8};
            let quant_dequant = |src: &[f32], n: usize, use_q8_1: bool| -> Vec<f32> {
                if use_q8_1 {
                    let nb = n.div_ceil(256);
                    let mut blks = vec![CpuBlockQ8_1G8::default(); nb];
                    for bi in 0..nb {
                        let b0 = bi * 256;
                        let blk_n = (n - b0).min(256) as i32;
                        unsafe {
                            ampllm_cpu_quantize_q8_1_g8(
                                src[b0..].as_ptr(),
                                blks.as_mut_ptr().add(bi) as *mut std::ffi::c_void,
                                blk_n,
                            );
                        }
                    }
                    let mut dq = vec![0.0f32; n];
                    for (bi, ch) in dq.chunks_mut(256).enumerate() {
                        for (j, o) in ch.iter_mut().enumerate() {
                            *o = blks[bi].d[j / 32] * (blks[bi].qs[j] as f32);
                        }
                    }
                    dq
                } else {
                    let mut xq = vec![CpuBlockQ8K::default(); n.div_ceil(256)];
                    quantize_f32_to_q8_k_cpu(src, &mut xq);
                    let xq_bytes = unsafe {
                        std::slice::from_raw_parts(xq.as_ptr() as *const u8, (n / 256) * 292)
                    };
                    let mut dq = vec![0.0f32; n];
                    for bi in 0..(n / 256) {
                        let blk = &xq_bytes[bi * 292..(bi + 1) * 292];
                        let d = f32::from_le_bytes(blk[0..4].try_into().unwrap());
                        for j in 0..256 {
                            dq[bi * 256 + j] = d * (blk[4 + j] as i8 as f32);
                        }
                    }
                    dq
                }
            };
            let x_dq = quant_dequant(x, n_embd, matches!(ty, GgmlType::Q4K | GgmlType::Q5K));
            let gu_rb = ty.row_bytes(n_embd);
            let dn_rb = ty.row_bytes(n_ff);
            let mut ff = vec![0.0f32; n_ff];
            let mut row_f = vec![0.0f32; n_embd];
            for r in 0..n_ff {
                dequantize_row_to_f32(ty, &gate_w[r * gu_rb..(r + 1) * gu_rb], &mut row_f);
                let mut g: f32 = row_f.iter().zip(x_dq.iter()).map(|(&a, &b)| a * b).sum();
                dequantize_row_to_f32(ty, &up_w[r * gu_rb..(r + 1) * gu_rb], &mut row_f);
                let mut u: f32 = row_f.iter().zip(x_dq.iter()).map(|(&a, &b)| a * b).sum();
                g = g.min(clamp);
                u = u.clamp(-clamp, clamp);
                ff[r] = (g / (1.0 + (-g).exp())) * u;
            }
            let ff_dq = quant_dequant(
                &ff,
                n_ff,
                matches!(
                    ty,
                    GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 | GgmlType::Iq4Nl | GgmlType::Q2_0
                ),
            );
            let mut out = vec![0.0f32; n_embd];
            let mut d_row = vec![0.0f32; n_ff];
            for r in 0..n_embd {
                dequantize_row_to_f32(ty, &dn_w[r * dn_rb..(r + 1) * dn_rb], &mut d_row);
                let s: f32 = d_row.iter().zip(ff_dq.iter()).map(|(&a, &b)| a * b).sum();
                out[r] = w * s;
            }
            out
        };
        let q8k_m1 = q8k_moe(ty, &gate_w, &up_w, &dn_w, &x[..n_embd], w);
        let qn: f64 = q8k_m1.iter().map(|&v| (v as f64).powi(2)).sum();
        let qd: f64 = q8k_m1.iter().zip(got_m1.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let rel_q8k = (qd / qn.max(1e-12)).sqrt();
        println!("HY4 CPU M=1 MoE ({ty:?}) rel_vs_fp32 = {rel:.6} | rel_vs_q8k = {rel_q8k:.8}");
        assert!(rel_q8k < 1e-4, "HY4 M=1 {ty:?} kernel mismatch vs exact Q8K ref: {rel_q8k}");
        assert!(rel < 2.5e-2, "HY4 M=1 {ty:?} rel_vs_fp32={rel} >= 2.5e-2");

        // Grouped NT=8 parity against M=1 and Q8K ref
        let toks: Vec<usize> = (0..m_toks).collect();
        let wts = vec![w; m_toks];
        let grp = [Glm5CpuGroupedExpert {
            gate_ptr: gate_w.as_ptr(),
            up_ptr: up_w.as_ptr(),
            down_ptr: dn_w.as_ptr(),
            token_indices: &toks,
            token_weights: &wts,
        }];
        let mut got_grp = vec![0.0f32; m_toks * n_embd];
        run_glm5_cpu_moe_grouped(
            &pool,
            &mut scratch,
            ty,
            ty,
            n_embd,
            n_ff,
            clamp,
            m_toks,
            &x,
            &grp,
            &mut got_grp,
            true,
        );
        for t in 0..m_toks {
            let ref_t = q8k_moe(ty, &gate_w, &up_w, &dn_w, &x[t * n_embd..(t + 1) * n_embd], w);
            let rn: f64 = ref_t.iter().map(|&v| (v as f64).powi(2)).sum();
            let dn: f64 = ref_t
                .iter()
                .zip(got_grp[t * n_embd..(t + 1) * n_embd].iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            let rel_t = (dn / rn.max(1e-12)).sqrt();
            assert!(rel_t < 1e-4, "HY4 grouped {ty:?} tok {t} rel_vs_q8k={rel_t}");
        }
        let d0: f64 = got_m1
            .iter()
            .zip(got_grp[..n_embd].iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        let n0: f64 = got_m1.iter().map(|&v| (v as f64).powi(2)).sum();
        let g_rel = (d0 / n0.max(1e-12)).sqrt();
        println!("HY4 CPU grouped NT=8 vs M=1 ({ty:?}) tok0 rel_err = {g_rel:.8}");
        assert!(g_rel < 5e-4, "HY4 grouped {ty:?} tok0 diverged: {g_rel}");

        // Fast CPU tok_row_to_f32 parity vs dequantize_row_to_f32
        let rb = ty.row_bytes(n_embd);
        let mut fast_deq = vec![0.0f32; n_embd];
        let mut ref_deq = vec![0.0f32; n_embd];
        ampllm_cpu_kernels::tok_row_to_f32_cpu(ty, &gate_w[..rb], &mut fast_deq);
        dequantize_row_to_f32(ty, &gate_w[..rb], &mut ref_deq);
        let max_deq_diff = fast_deq
            .iter()
            .zip(ref_deq.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_deq_diff < 1e-5, "tok_row_to_f32 {ty:?} max_diff={max_deq_diff}");
    }

    // BF16 -> Q8_0 and F32 -> Q8_0 parity
    let rows = 4usize;
    let k = 512usize;
    let src_f32: Vec<f32> = (0..rows * k).map(|_| rng.next_f32() * 2.5).collect();
    let src_bf16: Vec<u16> = src_f32.iter().map(|&v| half::bf16::from_f32(v).to_bits()).collect();
    let mut q_from_f32 = vec![0u8; rows * (k / 32) * 34];
    let mut q_from_bf16 = vec![0u8; rows * (k / 32) * 34];
    ampllm_cpu_kernels::f32_to_q8_0_cpu(&src_f32, rows, k, &mut q_from_f32);
    ampllm_cpu_kernels::bf16_to_q8_0_cpu(&src_bf16, rows, k, &mut q_from_bf16);
    let mut deq_f32 = vec![0.0f32; k];
    let mut deq_bf16 = vec![0.0f32; k];
    for r in 0..rows {
        let rb = (k / 32) * 34;
        dequantize_row_to_f32(GgmlType::Q8_0, &q_from_f32[r * rb..(r + 1) * rb], &mut deq_f32);
        dequantize_row_to_f32(GgmlType::Q8_0, &q_from_bf16[r * rb..(r + 1) * rb], &mut deq_bf16);
        let s_row = &src_f32[r * k..(r + 1) * k];
        let sn: f64 = s_row.iter().map(|&v| (v as f64).powi(2)).sum();
        let df: f64 = s_row.iter().zip(deq_f32.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let db: f64 = s_row.iter().zip(deq_bf16.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        assert!((df / sn).sqrt() < 1.5e-2, "f32_to_q8_0 row {r} rel too high");
        assert!((db / sn).sqrt() < 2e-2, "bf16_to_q8_0 row {r} rel too high");
    }
}

#[test]
fn test_qwen4_avx512_q4k_q51_q80_640_and_iq4nl_parity() {
    use ampllm_cpu_kernels::glm5_cpu::{
        ampllm_cpu_quantize_q8_1_g8, run_glm5_cpu_moe_grouped_small, CpuBlockQ8_1G8,
    };
    let pool = CpuThreadPool::new(None);
    let mut rng = Rng(20261122);

    let dequant_q8k_prefix = |blocks: &[CpuBlockQ8K], n: usize| -> Vec<f32> {
        let nb = n.div_ceil(256);
        let bytes = unsafe { std::slice::from_raw_parts(blocks.as_ptr() as *const u8, nb * 292) };
        let mut out = vec![0.0f32; n];
        for (idx, chunk) in out.chunks_mut(256).enumerate() {
            let raw = &bytes[idx * 292..(idx + 1) * 292];
            let d = f32::from_le_bytes(raw[0..4].try_into().unwrap());
            for (j, o) in chunk.iter_mut().enumerate() {
                *o = d * (raw[4 + j] as i8 as f32);
            }
        }
        out
    };

    let dequant_q8_1_g8_prefix = |blocks: &[CpuBlockQ8_1G8], n: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; n];
        for (idx, chunk) in out.chunks_mut(256).enumerate() {
            let b = &blocks[idx];
            for (j, o) in chunk.iter_mut().enumerate() {
                *o = b.d[j / 32] * (b.qs[j] as f32);
            }
        }
        out
    };

    let ref_moe = |gu_ty: GgmlType,
                   dn_ty: GgmlType,
                   n_embd: usize,
                   n_ff: usize,
                   gate_w: &[u8],
                   up_w: &[u8],
                   dn_w: &[u8],
                   x_f32: &[f32],
                   w: f32,
                   quant_act: bool|
     -> Vec<f32> {
        let x_vec = if quant_act {
            if matches!(gu_ty, GgmlType::Q4K | GgmlType::Q5K) {
                let mut x_q81 = vec![CpuBlockQ8_1G8::default(); n_embd.div_ceil(256)];
                for b in 0..x_q81.len() {
                    let b0 = b * 256;
                    let blk_n = (n_embd - b0).min(256) as i32;
                    unsafe {
                        ampllm_cpu_quantize_q8_1_g8(
                            x_f32[b0..].as_ptr(),
                            x_q81.as_mut_ptr().add(b) as *mut std::ffi::c_void,
                            blk_n,
                        );
                    }
                }
                dequant_q8_1_g8_prefix(&x_q81, n_embd)
            } else {
                let mut x_q8k = vec![CpuBlockQ8K::default(); n_embd.div_ceil(256)];
                quantize_f32_to_q8_k_cpu(x_f32, &mut x_q8k);
                dequant_q8k_prefix(&x_q8k, n_embd)
            }
        } else {
            x_f32.to_vec()
        };
        let gu_rb = gu_ty.row_bytes(n_embd);
        let dn_rb = dn_ty.row_bytes(n_ff);
        let mut w_row = vec![0.0f32; n_embd.max(n_ff)];
        let mut ff = vec![0.0f32; n_ff];
        for r in 0..n_ff {
            dequantize_row_to_f32(gu_ty, &gate_w[r * gu_rb..(r + 1) * gu_rb], &mut w_row[..n_embd]);
            let g: f32 = w_row[..n_embd].iter().zip(x_vec.iter()).map(|(&a, &b)| a * b).sum();
            dequantize_row_to_f32(gu_ty, &up_w[r * gu_rb..(r + 1) * gu_rb], &mut w_row[..n_embd]);
            let u: f32 = w_row[..n_embd].iter().zip(x_vec.iter()).map(|(&a, &b)| a * b).sum();
            ff[r] = (g / (1.0 + (-g).exp())) * u;
        }
        let ff_vec = if quant_act {
            if matches!(
                dn_ty,
                GgmlType::Q5_0 | GgmlType::Q5_1 | GgmlType::Q8_0 | GgmlType::Iq4Nl | GgmlType::Q2_0
            ) {
                let mut ff_q81 = vec![CpuBlockQ8_1G8::default(); n_ff.div_ceil(256)];
                for b in 0..ff_q81.len() {
                    let b0 = b * 256;
                    let blk_n = (n_ff - b0).min(256) as i32;
                    unsafe {
                        ampllm_cpu_quantize_q8_1_g8(
                            ff[b0..].as_ptr(),
                            ff_q81.as_mut_ptr().add(b) as *mut std::ffi::c_void,
                            blk_n,
                        );
                    }
                }
                dequant_q8_1_g8_prefix(&ff_q81, n_ff)
            } else {
                let mut ff_q8k = vec![CpuBlockQ8K::default(); n_ff.div_ceil(256)];
                quantize_f32_to_q8_k_cpu(&ff, &mut ff_q8k);
                dequant_q8k_prefix(&ff_q8k, n_ff)
            }
        } else {
            ff
        };
        let mut out = vec![0.0f32; n_embd];
        for r in 0..n_embd {
            dequantize_row_to_f32(dn_ty, &dn_w[r * dn_rb..(r + 1) * dn_rb], &mut w_row[..n_ff]);
            let d: f32 = w_row[..n_ff].iter().zip(ff_vec.iter()).map(|(&a, &b)| a * b).sum();
            out[r] = w * d;
        }
        out
    };

    let configs = [
        (GgmlType::Q4K, GgmlType::Q5_1, 2560usize, 640usize, 13usize, 1u64),
        (GgmlType::Q5K, GgmlType::Q8_0, 2560usize, 640usize, 7usize, 2u64),
        (GgmlType::Q4K, GgmlType::Q5_1, 512usize, 288usize, 4usize, 3u64),
        (GgmlType::Q4K, GgmlType::Q8_0, 512usize, 352usize, 2usize, 4u64),
        (GgmlType::Iq4Nl, GgmlType::Iq4Nl, 2560usize, 640usize, 8usize, 5u64),
        (GgmlType::Q5_0, GgmlType::Q5_0, 2560usize, 640usize, 7usize, 6u64),
        (GgmlType::Iq3S, GgmlType::Iq4Nl, 2560usize, 640usize, 8usize, 7u64),
        (GgmlType::Q4K, GgmlType::Q5_0, 1536usize, 1536usize, 6usize, 8u64),
        (GgmlType::Q2_0, GgmlType::Q2_0, 2560usize, 640usize, 8usize, 9u64),
    ];

    for (gu_ty, dn_ty, n_embd, n_ff, m_toks, seed) in configs {
        let gate_w = fill_random_quant_rows(gu_ty, n_ff, n_embd, seed * 10 + 1);
        let up_w = fill_random_quant_rows(gu_ty, n_ff, n_embd, seed * 10 + 2);
        let dn_w = fill_random_quant_rows(dn_ty, n_embd, n_ff, seed * 10 + 3);
        let x: Vec<f32> = (0..m_toks * n_embd).map(|_| rng.next_f32() * 0.6).collect();
        let w = 0.72f32;

        // M=1 parity (single job + multi-job)
        let mut scratch = Glm5CpuScratch::new(n_embd, n_ff, 4);
        let mut got_m1 = vec![0.0f32; n_embd];
        let job1 = [Glm5CpuExpertJob {
            gate_ptr: gate_w.as_ptr(),
            up_ptr: up_w.as_ptr(),
            down_ptr: dn_w.as_ptr(),
            weight: w,
        }];
        run_glm5_cpu_moe_m1(
            &pool,
            &mut scratch,
            gu_ty,
            dn_ty,
            n_embd,
            n_ff,
            0.0,
            &x[..n_embd],
            None,
            &job1,
            &mut got_m1,
        );
        let r_q = ref_moe(gu_ty, dn_ty, n_embd, n_ff, &gate_w, &up_w, &dn_w, &x[..n_embd], w, true);
        let r_f32 = ref_moe(gu_ty, dn_ty, n_embd, n_ff, &gate_w, &up_w, &dn_w, &x[..n_embd], w, false);
        let qn: f64 = r_q.iter().map(|&v| (v as f64).powi(2)).sum();
        let qd: f64 = r_q.iter().zip(got_m1.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let fn_: f64 = r_f32.iter().map(|&v| (v as f64).powi(2)).sum();
        let fd: f64 = r_f32.iter().zip(got_m1.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        let rel_q = (qd / qn.max(1e-12)).sqrt();
        let rel_f32 = (fd / fn_.max(1e-12)).sqrt();
        println!(
            "Qwen4 CPU M=1 ({gu_ty:?}/{dn_ty:?}, {n_embd}->{n_ff}) rel_vs_q={rel_q:.8} rel_vs_fp32={rel_f32:.6}"
        );
        assert!(rel_q < 1e-4, "Qwen4 M=1 ({gu_ty:?}/{dn_ty:?}, {n_ff}) rel_vs_q={rel_q}");
        if n_embd == 2560 && n_ff == 640 {
            assert!(rel_f32 < 1.5e-2, "Qwen4 M=1 ({gu_ty:?}/{dn_ty:?}, {n_ff}) rel_vs_fp32={rel_f32}");
        } else {
            assert!(rel_f32 < 6e-2, "Qwen4 M=1 ({gu_ty:?}/{dn_ty:?}, {n_ff}) rel_vs_fp32={rel_f32}");
        }

        // Multi-job (n_jobs = 3 -> exercises `if n_jobs >= 2` path in M=1)
        let mut got_m1_3 = vec![0.0f32; n_embd];
        let job3 = [
            Glm5CpuExpertJob {
                gate_ptr: gate_w.as_ptr(),
                up_ptr: up_w.as_ptr(),
                down_ptr: dn_w.as_ptr(),
                weight: 0.3 * w,
            },
            Glm5CpuExpertJob {
                gate_ptr: gate_w.as_ptr(),
                up_ptr: up_w.as_ptr(),
                down_ptr: dn_w.as_ptr(),
                weight: 0.5 * w,
            },
            Glm5CpuExpertJob {
                gate_ptr: gate_w.as_ptr(),
                up_ptr: up_w.as_ptr(),
                down_ptr: dn_w.as_ptr(),
                weight: 0.2 * w,
            },
        ];
        run_glm5_cpu_moe_m1(
            &pool,
            &mut scratch,
            gu_ty,
            dn_ty,
            n_embd,
            n_ff,
            0.0,
            &x[..n_embd],
            None,
            &job3,
            &mut got_m1_3,
        );
        let d3: f64 = got_m1.iter().zip(got_m1_3.iter()).map(|(&a, &b)| ((a - b) as f64).powi(2)).sum();
        assert!((d3 / qn.max(1e-12)).sqrt() < 1e-4, "Qwen4 M=1 multi-job sum diverged");

        // Grouped prefill parity (`NT = m_toks`)
        let toks: Vec<usize> = (0..m_toks).collect();
        let wts = vec![w; m_toks];
        let grp = [Glm5CpuGroupedExpert {
            gate_ptr: gate_w.as_ptr(),
            up_ptr: up_w.as_ptr(),
            down_ptr: dn_w.as_ptr(),
            token_indices: &toks,
            token_weights: &wts,
        }];
        let mut got_grp = vec![0.0f32; m_toks * n_embd];
        run_glm5_cpu_moe_grouped(
            &pool,
            &mut scratch,
            gu_ty,
            dn_ty,
            n_embd,
            n_ff,
            0.0,
            m_toks,
            &x,
            &grp,
            &mut got_grp,
            true,
        );
        for t in 0..m_toks {
            let rt = ref_moe(
                gu_ty,
                dn_ty,
                n_embd,
                n_ff,
                &gate_w,
                &up_w,
                &dn_w,
                &x[t * n_embd..(t + 1) * n_embd],
                w,
                true,
            );
            let rn: f64 = rt.iter().map(|&v| (v as f64).powi(2)).sum();
            let rd: f64 = rt
                .iter()
                .zip(got_grp[t * n_embd..(t + 1) * n_embd].iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            let rel_t = (rd / rn.max(1e-12)).sqrt();
            assert!(
                rel_t < 1e-4,
                "Qwen4 grouped ({gu_ty:?}/{dn_ty:?}, {n_ff}, m={m_toks}) tok {t} rel_vs_q={rel_t}"
            );
        }

        // Grouped small batch parity (`run_glm5_cpu_moe_grouped_small`, `m_small <= 8`)
        let m_small = m_toks.min(8);
        let small_toks: Vec<usize> = (0..m_small).collect();
        let small_wts = vec![w; m_small];
        let small_grp = [Glm5CpuGroupedExpert {
            gate_ptr: gate_w.as_ptr(),
            up_ptr: up_w.as_ptr(),
            down_ptr: dn_w.as_ptr(),
            token_indices: &small_toks,
            token_weights: &small_wts,
        }];
        let mut got_small = vec![0.0f32; m_small * n_embd];
        run_glm5_cpu_moe_grouped_small(
            &pool,
            &mut scratch,
            gu_ty,
            dn_ty,
            n_embd,
            n_ff,
            0.0,
            m_small,
            &x[..m_small * n_embd],
            &small_grp,
            &mut got_small,
            true,
        );
        for t in 0..m_small {
            let rt = ref_moe(
                gu_ty,
                dn_ty,
                n_embd,
                n_ff,
                &gate_w,
                &up_w,
                &dn_w,
                &x[t * n_embd..(t + 1) * n_embd],
                w,
                true,
            );
            let rn: f64 = rt.iter().map(|&v| (v as f64).powi(2)).sum();
            let rd: f64 = rt
                .iter()
                .zip(got_small[t * n_embd..(t + 1) * n_embd].iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            let rel_t = (rd / rn.max(1e-12)).sqrt();
            assert!(
                rel_t < 1e-4,
                "Qwen4 grouped_small ({gu_ty:?}/{dn_ty:?}, {n_ff}, m={m_small}) tok {t} rel_vs_q={rel_t}"
            );
        }
    }

    // `tok_row_to_f32_cpu` parity for Qwen3.8 PLE / embeddings (`k=160`, `2560`, `640`)
    for (ty, k) in [
        (GgmlType::Iq4Nl, 160usize),
        (GgmlType::Iq4Nl, 2560usize),
        (GgmlType::Q5_1, 640usize),
        (GgmlType::Q5_0, 640usize),
        (GgmlType::Q2_0, 640usize),
        (GgmlType::Q4_0, 640usize),
        (GgmlType::Q3K, 512usize),
    ] {
        let w = fill_random_quant_rows(ty, 1, k, 991 + k as u64);
        let mut fast = vec![0.0f32; k];
        let mut ref_deq = vec![0.0f32; k];
        ampllm_cpu_kernels::tok_row_to_f32_cpu(ty, &w, &mut fast);
        dequantize_row_to_f32(ty, &w, &mut ref_deq);
        let max_diff = fast
            .iter()
            .zip(ref_deq.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("tok_row_to_f32_cpu ({ty:?}, k={k}) max_diff = {max_diff:.8}");
        assert!(max_diff < 1e-5, "tok_row_to_f32_cpu ({ty:?}, k={k}) max_diff={max_diff}");
    }
}

#[test]
fn test_mm3_avx512_iq2xxs_iq2xs_iq1s_and_mixed_swigluoai_parity() {

    let fill_mm3_rows = |ty: GgmlType, n: usize, k: usize, seed: u64| -> Vec<u8> {
        let mut rng = Rng(seed);
        let rb = ty.row_bytes(k);
        let blk_sz = ty.type_size();
        let nb = k / 256;
        let mut out = vec![0u8; n * rb];
        for r in 0..n {
            let row = &mut out[r * rb..(r + 1) * rb];
            for b in 0..nb {
                let blk = &mut row[b * blk_sz..(b + 1) * blk_sz];
                for byte in blk.iter_mut() {
                    *byte = (rng.next_u32() & 0xFF) as u8;
                }
                let d_bytes = f16::from_f32(0.02 + rng.next_f32().abs() * 0.05).to_le_bytes();
                if ty == GgmlType::Q6K {
                    blk[208..210].copy_from_slice(&d_bytes);
                } else {
                    blk[0..2].copy_from_slice(&d_bytes);
                }
            }
        }
        out
    };

    let quant_q8k_ref = |x: &[f32]| -> (Vec<u8>, Vec<f32>) {
        let nb = x.len() / 256;
        let q8k_bytes = ampllm_cpu_kernels::glm5_cpu::Q8K_BLOCK_BYTES;
        let mut raw = vec![0u8; nb * q8k_bytes];
        unsafe {
            ampllm_cpu_kernels::mm3_cpu::ampllm_cpu_mm3_quantize_q8_k(
                x.as_ptr(),
                raw.as_mut_ptr() as *mut std::ffi::c_void,
                x.len() as i32,
            );
        }
        let mut deq = Vec::with_capacity(x.len());
        for b in raw.chunks_exact(q8k_bytes) {
            let d = f32::from_le_bytes(b[0..4].try_into().unwrap());
            for j in 0..256 {
                deq.push(d * ((b[4 + j] as i8) as f32));
            }
        }
        (raw, deq)
    };

    let ref_mm3_moe = |g_ty: GgmlType,
                       u_ty: GgmlType,
                       d_ty: GgmlType,
                       n_embd: usize,
                       n_ff: usize,
                       gate_w: &[u8],
                       up_w: &[u8],
                       dn_w: &[u8],
                       x: &[f32],
                       weight: f32,
                       quantize_act: bool| -> Vec<f32> {
        let g_rb = g_ty.row_bytes(n_embd);
        let u_rb = u_ty.row_bytes(n_embd);
        let d_rb = d_ty.row_bytes(n_ff);
        let alpha = 1.702f32;
        let limit = 7.0f32;
        let x_use: Vec<f32> = if quantize_act {
            quant_q8k_ref(x).1
        } else {
            x.to_vec()
        };
        let mut hidden = vec![0.0f32; n_ff];
        let mut dg_row = vec![0.0f32; n_embd];
        let mut du_row = vec![0.0f32; n_embd];
        for r in 0..n_ff {
            dequantize_row_to_f32(g_ty, &gate_w[r * g_rb..(r + 1) * g_rb], &mut dg_row);
            dequantize_row_to_f32(u_ty, &up_w[r * u_rb..(r + 1) * u_rb], &mut du_row);
            let g_sum: f64 = dg_row
                .iter()
                .zip(x_use.iter())
                .map(|(&w, &v)| (w as f64) * (v as f64))
                .sum();
            let u_sum: f64 = du_row
                .iter()
                .zip(x_use.iter())
                .map(|(&w, &v)| (w as f64) * (v as f64))
                .sum();
            let g_c = (g_sum as f32).min(limit);
            let u_c = (u_sum as f32).clamp(-limit, limit);
            hidden[r] = (g_c / (1.0f32 + (-alpha * g_c).exp())) * (u_c + 1.0f32);
        }
        let h_use: Vec<f32> = if quantize_act {
            quant_q8k_ref(&hidden).1
        } else {
            hidden
        };
        let mut out = vec![0.0f32; n_embd];
        let mut dd_row = vec![0.0f32; n_ff];
        for r in 0..n_embd {
            dequantize_row_to_f32(d_ty, &dn_w[r * d_rb..(r + 1) * d_rb], &mut dd_row);
            let d_sum: f64 = dd_row
                .iter()
                .zip(h_use.iter())
                .map(|(&w, &v)| (w as f64) * (v as f64))
                .sum();
            out[r] = weight * (d_sum as f32);
        }
        out
    };

    let pool = CpuThreadPool::new(Some(8));
    let mut scratch = ampllm_cpu_kernels::Mm3CpuScratch::new(1024, 512, 16);

    let combos = [
        (GgmlType::Iq2Xxs, GgmlType::Iq2Xxs, GgmlType::Iq2Xxs),
        (GgmlType::Iq2Xs, GgmlType::Iq2Xxs, GgmlType::Iq3Xxs),
        (GgmlType::Iq2Xs, GgmlType::Iq2Xxs, GgmlType::Iq2Xs),
        (GgmlType::Iq2S, GgmlType::Iq2S, GgmlType::Iq3Xxs),
        (GgmlType::Iq1S, GgmlType::Iq1S, GgmlType::Iq1S),
    ];

    let n_embd = 1024usize;
    let n_ff = 512usize;
    let max_m = 12usize;
    let mut rng = Rng(20260301);
    let x: Vec<f32> = (0..max_m * n_embd).map(|_| rng.next_f32() * 0.6).collect();
    let alpha = 1.702f32;
    let limit = 7.0f32;

    for (idx, (g_ty, u_ty, d_ty)) in combos.into_iter().enumerate() {
        let gate_w = fill_mm3_rows(g_ty, n_ff, n_embd, 100 + idx as u64);
        let up_w = fill_mm3_rows(u_ty, n_ff, n_embd, 200 + idx as u64);
        let dn_w = fill_mm3_rows(d_ty, n_embd, n_ff, 300 + idx as u64);
        let w = 0.85f32;

        // 1. M=1 single-job & 3-job parity
        let mut got_m1 = vec![0.0f32; n_embd];
        let job1 = [ampllm_cpu_kernels::Mm3CpuExpertJob {
            gate_ptr: gate_w.as_ptr(),
            up_ptr: up_w.as_ptr(),
            down_ptr: dn_w.as_ptr(),
            weight: w,
        }];
        ampllm_cpu_kernels::run_mm3_cpu_moe_m1(
            &pool,
            &mut scratch,
            g_ty,
            u_ty,
            d_ty,
            n_embd,
            n_ff,
            alpha,
            limit,
            &x[..n_embd],
            None,
            &job1,
            &mut got_m1,
        );
        let r_q = ref_mm3_moe(g_ty, u_ty, d_ty, n_embd, n_ff, &gate_w, &up_w, &dn_w, &x[..n_embd], w, true);
        let r_f = ref_mm3_moe(g_ty, u_ty, d_ty, n_embd, n_ff, &gate_w, &up_w, &dn_w, &x[..n_embd], w, false);
        let rel_err = |a: &[f32], b: &[f32]| -> f64 {
            let n2: f64 = a.iter().map(|&v| (v as f64).powi(2)).sum();
            let d2: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| ((x - y) as f64).powi(2)).sum();
            (d2 / n2.max(1e-12)).sqrt()
        };
        let rel_q = rel_err(&r_q, &got_m1);
        let rel_f = rel_err(&r_f, &got_m1);
        println!("MM3 CPU M=1 ({g_ty:?}/{u_ty:?}/{d_ty:?}) rel_vs_q={rel_q:.8} rel_vs_fp32={rel_f:.6}");
        assert!(rel_q < 1e-4, "MM3 M=1 rel_vs_q={rel_q}");
        assert!(rel_f < 3e-2, "MM3 M=1 rel_vs_fp32={rel_f}");

        let mut got_m1_3 = vec![0.0f32; n_embd];
        let job3 = [
            ampllm_cpu_kernels::Mm3CpuExpertJob {
                gate_ptr: gate_w.as_ptr(),
                up_ptr: up_w.as_ptr(),
                down_ptr: dn_w.as_ptr(),
                weight: 0.25 * w,
            },
            ampllm_cpu_kernels::Mm3CpuExpertJob {
                gate_ptr: gate_w.as_ptr(),
                up_ptr: up_w.as_ptr(),
                down_ptr: dn_w.as_ptr(),
                weight: 0.45 * w,
            },
            ampllm_cpu_kernels::Mm3CpuExpertJob {
                gate_ptr: gate_w.as_ptr(),
                up_ptr: up_w.as_ptr(),
                down_ptr: dn_w.as_ptr(),
                weight: 0.30 * w,
            },
        ];
        ampllm_cpu_kernels::run_mm3_cpu_moe_m1(
            &pool,
            &mut scratch,
            g_ty,
            u_ty,
            d_ty,
            n_embd,
            n_ff,
            alpha,
            limit,
            &x[..n_embd],
            None,
            &job3,
            &mut got_m1_3,
        );
        let rel_3 = rel_err(&got_m1, &got_m1_3);
        assert!(rel_3 < 1e-4, "MM3 M=1 3-job sum diverged: {rel_3}");

        // 2. Grouped M=7 (NT<=6 + mixed) and M=12 (wide-tiled NT>=7) parity
        for &m_toks in &[7usize, 12usize] {
            let toks: Vec<usize> = (0..m_toks).collect();
            let wts: Vec<f32> = (0..m_toks).map(|t| 0.4 + 0.1 * ((t % 5) as f32)).collect();
            let grp = [ampllm_cpu_kernels::Mm3CpuGroupedExpert {
                gate_ptr: gate_w.as_ptr(),
                up_ptr: up_w.as_ptr(),
                down_ptr: dn_w.as_ptr(),
                token_indices: &toks,
                token_weights: &wts,
            }];
            let mut got_grp = vec![0.0f32; m_toks * n_embd];
            ampllm_cpu_kernels::run_mm3_cpu_moe_grouped(
                &pool,
                &mut scratch,
                g_ty,
                u_ty,
                d_ty,
                n_embd,
                n_ff,
                alpha,
                limit,
                m_toks,
                &x[..m_toks * n_embd],
                &grp,
                &mut got_grp,
                true,
            );
            for t in 0..m_toks {
                let rt = ref_mm3_moe(
                    g_ty,
                    u_ty,
                    d_ty,
                    n_embd,
                    n_ff,
                    &gate_w,
                    &up_w,
                    &dn_w,
                    &x[t * n_embd..(t + 1) * n_embd],
                    wts[t],
                    true,
                );
                let rel_t = rel_err(&rt, &got_grp[t * n_embd..(t + 1) * n_embd]);
                assert!(
                    rel_t < 1e-4,
                    "MM3 grouped ({g_ty:?}/{u_ty:?}/{d_ty:?}, m={m_toks}) tok {t} rel_vs_q={rel_t}"
                );
            }
        }
    }

    // 3. q6_k_row_to_f32_cpu parity
    let q6k_w = fill_mm3_rows(GgmlType::Q6K, 1, 6144, 777);
    let mut fast_q6 = vec![0.0f32; 6144];
    let mut ref_q6 = vec![0.0f32; 6144];
    ampllm_cpu_kernels::q6_k_row_to_f32_cpu(&q6k_w, &mut fast_q6);
    dequantize_row_to_f32(GgmlType::Q6K, &q6k_w, &mut ref_q6);
    let max_diff = fast_q6
        .iter()
        .zip(ref_q6.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("q6_k_row_to_f32_cpu max_diff = {max_diff:.8}");
    assert!(max_diff < 1e-5, "q6_k_row_to_f32_cpu max_diff={max_diff}");
}
