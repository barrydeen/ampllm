use ampllm_cache::CacheBudget;
use ampllm_engine::{
    EngineConfig, KvQuantMode, Mm3Engine, MultiGpuMode, SamplingParams,
};
use ampllm_engine::mm3::vision::Mm3Vision;
use ampllm_gguf::{discover_mm3_aux_files, GgufFile, Mm3VisionConfig};
use std::path::{Path, PathBuf};
use std::time::Instant;

fn mm3_gguf_path() -> PathBuf {
    PathBuf::from(
        std::env::var("AMPLLM_MM3_MODEL")
            .expect("set AMPLLM_MM3_MODEL to MiniMax M3.1 Flash GGUF shard 1"),
    )
}

fn mm3_profile_path() -> Option<PathBuf> {
    std::env::var("AMPLLM_MM3_PROFILE").ok().map(PathBuf::from)
}

const REF_DIR: &str = "/tmp/mm3_ref";

const MAYA_TOPIC0_32_TOKS: [u32; 32] = [
    2765, 25959, 10, 1408, 65894, 10, 4707, 136696, 1424, 15818, 10, 4707, 30309, 1424, 9256, 44,
    53346, 44, 5279, 44, 10959, 4368, 2601, 16407, 56632, 8832, 22333, 115, 2624, 326, 3067, 10417,
];

fn read_ids(path: &Path) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed reading {}: {e}", path.display()))
        .trim()
        .split([',', '\n', ' '])
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>().expect("token id u32"))
        .collect()
}

fn read_mm3_logits(path: &Path) -> (usize, usize, Vec<u32>, Vec<f32>) {
    let bytes = std::fs::read(path)
        .unwrap_or_else(|e| panic!("failed reading {}: {e}", path.display()));
    assert!(bytes.len() >= 16, "truncated MM3L file");
    assert_eq!(&bytes[0..4], b"MM3L");
    let rows = i32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let vocab = i32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let n_ids = i32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let mut off = 16usize;
    let mut ids = Vec::with_capacity(n_ids);
    for _ in 0..n_ids {
        ids.push(i32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as u32);
        off += 4;
    }
    off += rows * 4; // skip positions
    let mut logits = Vec::with_capacity(rows * vocab);
    for _ in 0..(rows * vocab) {
        logits.push(f32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()));
        off += 4;
    }
    assert_eq!(off, bytes.len());
    (rows, vocab, ids, logits)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xd = x as f64;
        let yd = y as f64;
        dot += xd * yd;
        na += xd * xd;
        nb += yd * yd;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f64) {
    let mut max_d = 0.0f32;
    let mut sum_d = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        if d > max_d {
            max_d = d;
        }
        sum_d += d as f64;
    }
    (max_d, sum_d / (a.len().max(1) as f64))
}

fn argmax_id(v: &[f32]) -> u32 {
    let mut bi = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &val) in v.iter().enumerate() {
        if val > bv {
            bv = val;
            bi = i;
        }
    }
    bi as u32
}

fn top_k_logits(v: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut pairs: Vec<(u32, f32)> = v.iter().enumerate().map(|(i, &val)| (i as u32, val)).collect();
    pairs.sort_by(|a, b| b.1.total_cmp(&a.1));
    pairs.truncate(k);
    pairs
}

#[test]
#[ignore]
fn test_mm3_vision_e2e_parity() {
    let gguf_path = mm3_gguf_path();
    let ref_dir = Path::new(REF_DIR);
    assert!(gguf_path.exists(), "missing {}", gguf_path.display());
    assert!(ref_dir.exists(), "missing {}", ref_dir.display());

    let aux = discover_mm3_aux_files(&gguf_path, None, None).unwrap();
    let v_st = aux.vision_safetensors.expect("vision safetensors");
    let v_cfg_p = aux.vision_config_json.expect("vision config.json");
    let v_cfg = Mm3VisionConfig::from_companion_jsons(&v_cfg_p, aux.preprocessor_json.as_deref())
        .expect("parse Mm3VisionConfig");
    let vision = Mm3Vision::load(&v_st, v_cfg).expect("load Mm3Vision");

    let rgb = std::fs::read(ref_dir.join("test_rgb.bin")).expect("test_rgb.bin");
    let ref_bytes = std::fs::read(ref_dir.join("vision_ref.bin")).expect("vision_ref.bin");
    let ref_f32: &[f32] = bytemuck::cast_slice(&ref_bytes);

    let _warm = vision.encode_image(&rgb, 280, 280, 0).expect("warmup encode");
    let t0 = Instant::now();
    let out = vision.encode_image(&rgb, 280, 280, 0).expect("encode image");
    let ms = t0.elapsed().as_secs_f64() * 1000.0;

    assert_eq!(out.len(), ref_f32.len());
    assert!(
        out.iter().all(|v| v.is_finite()),
        "MM3 vision output has non-finite values"
    );
    let cos = cosine(&out, ref_f32);
    let (max_d, mean_d) = diff_stats(&out, ref_f32);
    println!(
        "[MM3 VISION E2E] shape={} ({}x{}) time={ms:.1} ms cos={cos:.6} max_diff={max_d:.5} mean_diff={mean_d:.6}",
        out.len(),
        out.len() / 6144,
        6144
    );
    assert!(cos > 0.999, "vision cosine too low: {cos}");
}

#[test]
#[ignore]
fn test_mm3_e2e_logits_parity_and_bench() {
    let gguf_path = mm3_gguf_path();
    let ref_dir = Path::new(REF_DIR);
    let profile_path = mm3_profile_path();
    assert!(gguf_path.exists(), "missing {}", gguf_path.display());
    assert!(ref_dir.exists(), "missing {}", ref_dir.display());
    if let Some(ref p) = profile_path {
        assert!(p.exists(), "missing {}", p.display());
    }

    let warmup_ids = read_ids(&ref_dir.join("warmup_ids.txt"));
    let topic0_ids = read_ids(&ref_dir.join("topic0_ids.txt"));
    let topic1_ids = read_ids(&ref_dir.join("topic1_ids.txt"));
    let topic2_ids = read_ids(&ref_dir.join("topic2_ids.txt"));
    let p2048_ids = read_ids(&ref_dir.join("p2048_ids.txt"));
    let p8192_ids = read_ids(&ref_dir.join("p8192_ids.txt"));

    // 1. BF16 KV parity check vs Maya reference logits_bf16.bin + 32-tok greedy sequence
    {
        let gguf = GgufFile::open(&gguf_path).expect("open mm3 gguf");
        let cfg_bf16 = EngineConfig {
            device: 0,
            max_batch: 8,
            max_ctx: 8192,
            expert_cache: CacheBudget::Auto,
            kv_mode: KvQuantMode::F16,
            multi_gpu: MultiGpuMode::Pipeline,
            cpu_threads: None,
            use_cuda_graph: false,
            use_spec: false,
            load_profile: profile_path.clone(),
            save_profile: None,
            dma_prefill_threshold: 8,
        };
        let mut engine = Mm3Engine::load(gguf, cfg_bf16).expect("load MM3 BF16 KV");
        let (rows, vocab, ref_ids, ref_logits) = read_mm3_logits(&ref_dir.join("logits_bf16.bin"));
        assert_eq!(rows, 1);
        assert_eq!(ref_ids, topic0_ids);

        let (logits, tr_a, _) = engine
            .eval_prompt_traced(&topic0_ids, false)
            .expect("eval BF16 logits A");
        let (_logits_b, tr_b, _) = engine
            .eval_prompt_traced(&topic0_ids, false)
            .expect("eval BF16 logits B");
        let l03_cos = cosine(&tr_a.l03_all_ffn, &tr_b.l03_all_ffn);
        let (l03_max, l03_mean) = diff_stats(&tr_a.l03_all_ffn, &tr_b.l03_all_ffn);
        println!(
            "[MM3 L03 ALL-TOK DETERMINISM] elems={} cos={l03_cos:.8} max_diff={l03_max:.6} mean_diff={l03_mean:.8}",
            tr_a.l03_all_ffn.len()
        );
        assert!(
            l03_max < 1e-3,
            "L03 all-token prefill MoE non-deterministic beyond atomicAdd rounding: max_diff={l03_max}"
        );

        let logits_step = engine
            .eval_prompt_logits_stepwise(&topic0_ids)
            .expect("eval BF16 stepwise logits");
        let last_ref = &ref_logits[(rows - 1) * vocab..rows * vocab];
        let cos = cosine(&logits, last_ref);
        let (max_d, mean_d) = diff_stats(&logits, last_ref);
        let top1 = argmax_id(&logits);
        let ref_top1 = argmax_id(last_ref);
        let cos_step = cosine(&logits_step, last_ref);
        let (max_d_step, mean_d_step) = diff_stats(&logits_step, last_ref);
        let top1_step = argmax_id(&logits_step);
        let cos_self = cosine(&logits, &logits_step);
        let (max_d_self, mean_d_self) = diff_stats(&logits, &logits_step);
        println!(
            "[MM3 BF16 KV PARITY batch vs ref] top1={top1} (ref={ref_top1}) cos={cos:.6} max_diff={max_d:.4} mean_diff={mean_d:.5} cache={}",
            engine.cache_summary()
        );
        println!("[MM3 BF16 KV top5 batch] {:?}", top_k_logits(&logits, 5));
        println!("[MM3 BF16 KV top5 ref]   {:?}", top_k_logits(last_ref, 5));
        println!(
            "[MM3 BF16 KV PARITY step1 vs ref] top1={top1_step} cos={cos_step:.6} max_diff={max_d_step:.4} mean_diff={mean_d_step:.5}"
        );
        println!("[MM3 BF16 KV top5 step1] {:?}", top_k_logits(&logits_step, 5));
        println!(
            "[MM3 BF16 KV SELF (batch vs step1)] cos_self={cos_self:.6} max_diff={max_d_self:.4} mean_diff={mean_d_self:.5}"
        );
        assert_eq!(top1, 2765);
        assert_eq!(top1, ref_top1);
        assert_eq!(top1_step, 2765);
        assert!(
            cos_self > 0.995,
            "BF16 batched prefill vs stepwise decode diverged: cos_self={cos_self}"
        );
        assert!(cos > 0.997, "BF16 batch logits cosine too low: {cos}");
        assert!(cos_step > 0.997, "BF16 stepwise logits cosine too low: {cos_step}");

        let gen32 = engine.generate_greedy(&topic0_ids, 32, &[]).expect("BF16 greedy 32");
        let matched = gen32
            .iter()
            .zip(MAYA_TOPIC0_32_TOKS.iter())
            .take_while(|(a, b)| a == b)
            .count();
        if matched < MAYA_TOPIC0_32_TOKS.len() {
            println!(
                "[MM3 BF16 32-TOK GREEDY] matched={matched}/32 first_mismatch_idx={matched} got={} ref={} gen32={:?}",
                gen32[matched],
                MAYA_TOPIC0_32_TOKS[matched],
                gen32
            );
        } else {
            println!("[MM3 BF16 32-TOK GREEDY] matched=32/32 gen32={:?}", gen32);
        }
    }

    // 2. INT8+Had128-V KV + DSpark lossless ON vs OFF + Maya benchmark suite
    let gguf = GgufFile::open(&gguf_path).expect("reopen mm3 gguf");
    let cfg_int8 = EngineConfig {
        device: 0,
        max_batch: 8,
        max_ctx: 16384,
        expert_cache: CacheBudget::Auto,
        kv_mode: KvQuantMode::Q8_0,
        multi_gpu: MultiGpuMode::Pipeline,
        cpu_threads: None,
        use_cuda_graph: false,
        use_spec: true,
        load_profile: profile_path.clone(),
        save_profile: None,
        dma_prefill_threshold: 8,
    };
    let mut engine = Mm3Engine::load(gguf, cfg_int8).expect("load MM3 INT8+DSpark");
    let (rows, vocab, _, ref_int8_logits) = read_mm3_logits(&ref_dir.join("logits_int8.bin"));
    let last_ref_i8 = &ref_int8_logits[(rows - 1) * vocab..rows * vocab];
    let int8_logits = engine.eval_prompt_logits(&topic0_ids).expect("eval INT8 logits");
    let cos_i8 = cosine(&int8_logits, last_ref_i8);
    let (max_d_i8, mean_d_i8) = diff_stats(&int8_logits, last_ref_i8);
    let top1_i8 = argmax_id(&int8_logits);
    println!(
        "[MM3 INT8 KV PARITY] top1={top1_i8} (ref={}) cos={cos_i8:.6} max_diff={max_d_i8:.4} mean_diff={mean_d_i8:.5} {}",
        argmax_id(last_ref_i8),
        engine.cache_summary()
    );
    println!("[MM3 INT8 KV top5]     {:?}", top_k_logits(&int8_logits, 5));
    println!("[MM3 INT8 KV top5 ref] {:?}", top_k_logits(last_ref_i8, 5));
    assert_eq!(top1_i8, 2765);
    assert!(cos_i8 > 0.995, "INT8 logits cosine too low: {cos_i8}");

    // DSpark lossless greedy parity: ON vs OFF must be 100% token-identical
    let loss_p = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        seed: 42,
        max_tokens: 32,
        stop_token_ids: Vec::new(),
    };
    engine.set_use_dspark(true);
    let r_on = engine
        .generate_stream(&topic0_ids, &loss_p, |_| true)
        .expect("DSpark ON 32");
    engine.set_use_dspark(false);
    let r_off = engine
        .generate_stream(&topic0_ids, &loss_p, |_| true)
        .expect("DSpark OFF 32");
    let i8_matched = r_on
        .generated_tokens
        .iter()
        .zip(MAYA_TOPIC0_32_TOKS.iter())
        .take_while(|(a, b)| a == b)
        .count();
    assert_eq!(
        r_on.generated_tokens, r_off.generated_tokens,
        "DSpark ON diverged from DSpark OFF on greedy decode!"
    );
    let loss_acc_pct = if r_on.draft_n > 0 {
        100.0 * r_on.draft_accepted as f64 / r_on.draft_n as f64
    } else {
        0.0
    };
    println!(
        "[MM3 DSPARK LOSSLESS CHECK] 32/32 identical ON==OFF acc={}/{} ({loss_acc_pct:.1}%) (matched_maya_bf16_ref={i8_matched}/32, top1..8={:?})",
        r_on.draft_accepted,
        r_on.draft_n,
        &r_on.generated_tokens[..8]
    );

    let run_suite = |eng: &mut Mm3Engine, dspark_on: bool, tag: &str| {
        eng.set_use_dspark(dspark_on);
        let warm_p = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            seed: 42,
            max_tokens: 128,
            stop_token_ids: Vec::new(),
        };
        let _ = eng.generate_stream(&warmup_ids, &warm_p, |_| true).unwrap();

        let dec_p = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            seed: 42,
            max_tokens: 256,
            stop_token_ids: Vec::new(),
        };
        let mut dec_tps_sum = 0.0f64;
        for (ti, t_ids) in [&topic0_ids, &topic1_ids, &topic2_ids].iter().enumerate() {
            let res = eng.generate_stream(t_ids, &dec_p, |_| true).unwrap();
            dec_tps_sum += res.decode_tps;
            let acc_pct = if res.draft_n > 0 {
                100.0 * res.draft_accepted as f64 / res.draft_n as f64
            } else {
                0.0
            };
            println!(
                "[BENCH {tag}] topic{ti} p={} n={}: pp={:.1} tok/s tg={:.2} tok/s acc={}/{} ({acc_pct:.1}%) | {}",
                t_ids.len(),
                res.generated_tokens.len(),
                res.prefill_tps,
                res.decode_tps,
                res.draft_accepted,
                res.draft_n,
                eng.cache_summary()
            );
        }
        println!("[BENCH {tag}] MEAN decode={:.2} tok/s", dec_tps_sum / 3.0);

        let pp_p = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            seed: 42,
            max_tokens: 1,
            stop_token_ids: Vec::new(),
        };
        let r2048 = eng.generate_stream(&p2048_ids, &pp_p, |_| true).unwrap();
        println!(
            "[BENCH {tag}] p2048: pp={:.1} tok/s | {}",
            r2048.prefill_tps,
            eng.cache_summary()
        );
        let r8192 = eng.generate_stream(&p8192_ids, &pp_p, |_| true).unwrap();
        println!(
            "[BENCH {tag}] p8192: pp={:.1} tok/s | {}",
            r8192.prefill_tps,
            eng.cache_summary()
        );
    };

    run_suite(&mut engine, true, "dspark=ON");
    drop(engine);
    let gguf_off = GgufFile::open(&gguf_path).expect("reopen mm3 gguf for dspark=OFF");
    let cfg_off = EngineConfig {
        device: 0,
        max_batch: 8,
        max_ctx: 16384,
        expert_cache: CacheBudget::Auto,
        kv_mode: KvQuantMode::Q8_0,
        multi_gpu: MultiGpuMode::Pipeline,
        cpu_threads: None,
        use_cuda_graph: false,
        use_spec: false,
        load_profile: profile_path.clone(),
        save_profile: None,
        dma_prefill_threshold: 8,
    };
    let mut engine_off = Mm3Engine::load(gguf_off, cfg_off).expect("load MM3 INT8 dspark=OFF");
    run_suite(&mut engine_off, false, "dspark=OFF");
    println!(
        "[MAYA REF] dspark=ON: mean_tg=22.66 tok/s, pp2048=124.3/121.9, pp8192=122.8 | dspark=OFF: mean_tg=15.83 tok/s, pp8192=126.6"
    );
}

#[test]
#[ignore]
fn test_mm3_bisect_batch_vs_step1() {
    let gguf_path = mm3_gguf_path();
    let ref_dir = Path::new(REF_DIR);
    let profile_path = mm3_profile_path();
    assert!(gguf_path.exists(), "missing {}", gguf_path.display());
    assert!(ref_dir.exists(), "missing {}", ref_dir.display());
    if let Some(ref p) = profile_path {
        assert!(p.exists(), "missing {}", p.display());
    }

    let topic0_ids = read_ids(&ref_dir.join("topic0_ids.txt"));
    let (rows, vocab, ref_ids, ref_logits) = read_mm3_logits(&ref_dir.join("logits_bf16.bin"));
    assert_eq!(rows, 1);
    assert_eq!(ref_ids, topic0_ids);
    let last_ref = &ref_logits[(rows - 1) * vocab..rows * vocab];

    let gguf = GgufFile::open(&gguf_path).expect("open mm3 gguf");
    let cfg_bf16 = EngineConfig {
        device: 0,
        max_batch: 8,
        max_ctx: 8192,
        expert_cache: CacheBudget::Auto,
        kv_mode: KvQuantMode::F16,
        multi_gpu: MultiGpuMode::Pipeline,
        cpu_threads: None,
        use_cuda_graph: false,
        use_spec: false,
        load_profile: profile_path.clone(),
        save_profile: None,
        dma_prefill_threshold: 8,
    };
    let mut engine = Mm3Engine::load(gguf, cfg_bf16).expect("load MM3 BF16 KV");

    let (logits_ba, tr_ba, sum_ba) = engine
        .eval_prompt_traced(&topic0_ids, false)
        .expect("eval batch_a");
    let (logits_bb, tr_bb, sum_bb) = engine
        .eval_prompt_traced(&topic0_ids, false)
        .expect("eval batch_b");
    let (logits_st, tr_st, sum_st) = engine
        .eval_prompt_traced(&topic0_ids, true)
        .expect("eval step1");

    let (cos_ba_ref, (max_ba_ref, mean_ba_ref)) =
        (cosine(&logits_ba, last_ref), diff_stats(&logits_ba, last_ref));
    let (cos_bb_ref, (max_bb_ref, mean_bb_ref)) =
        (cosine(&logits_bb, last_ref), diff_stats(&logits_bb, last_ref));
    let (cos_ba_bb, (max_ba_bb, mean_ba_bb)) =
        (cosine(&logits_ba, &logits_bb), diff_stats(&logits_ba, &logits_bb));
    let (cos_st_ref, (max_st_ref, mean_st_ref)) =
        (cosine(&logits_st, last_ref), diff_stats(&logits_st, last_ref));
    let (cos_ba_st, (max_ba_st, mean_ba_st)) =
        (cosine(&logits_ba, &logits_st), diff_stats(&logits_ba, &logits_st));

    println!(
        "[BISECT] batch_a vs ref: top1={} cos={cos_ba_ref:.6} max={max_ba_ref:.4} mean={mean_ba_ref:.5} {sum_ba}",
        argmax_id(&logits_ba)
    );
    println!(
        "[BISECT] batch_b vs ref: top1={} cos={cos_bb_ref:.6} max={max_bb_ref:.4} mean={mean_bb_ref:.5} {sum_bb}",
        argmax_id(&logits_bb)
    );
    println!(
        "[BISECT] batch_a vs batch_b (determinism): cos={cos_ba_bb:.6} max={max_ba_bb:.6} mean={mean_ba_bb:.6}"
    );
    println!(
        "[BISECT] step1   vs ref: top1={} cos={cos_st_ref:.6} max={max_st_ref:.4} mean={mean_st_ref:.5} {sum_st}",
        argmax_id(&logits_st)
    );
    println!(
        "[BISECT] batch_a vs step1: cos={cos_ba_st:.6} max={max_ba_st:.4} mean={mean_ba_st:.5}"
    );
    println!("[BISECT] top5 ref:     {:?}", top_k_logits(last_ref, 5));
    println!("[BISECT] top5 batch_a: {:?}", top_k_logits(&logits_ba, 5));
    println!("[BISECT] top5 batch_b: {:?}", top_k_logits(&logits_bb, 5));
    println!("[BISECT] top5 step1:   {:?}", top_k_logits(&logits_st, 5));

    for il in 0..engine.cfg.n_layers {
        let cos_ab_a = cosine(&tr_ba.after_attn[il], &tr_bb.after_attn[il]);
        let (max_ab_a, _) = diff_stats(&tr_ba.after_attn[il], &tr_bb.after_attn[il]);
        let cos_ab_f = cosine(&tr_ba.after_ffn[il], &tr_bb.after_ffn[il]);
        let (max_ab_f, _) = diff_stats(&tr_ba.after_ffn[il], &tr_bb.after_ffn[il]);
        let cos_bs_a = cosine(&tr_ba.after_attn[il], &tr_st.after_attn[il]);
        let (max_bs_a, _) = diff_stats(&tr_ba.after_attn[il], &tr_st.after_attn[il]);
        let cos_bs_f = cosine(&tr_ba.after_ffn[il], &tr_st.after_ffn[il]);
        let (max_bs_f, _) = diff_stats(&tr_ba.after_ffn[il], &tr_st.after_ffn[il]);

        let ids_ba: Vec<i32> = tr_ba.router_last[il].iter().map(|x| x.0).collect();
        let ids_bb: Vec<i32> = tr_bb.router_last[il].iter().map(|x| x.0).collect();
        let ids_st: Vec<i32> = tr_st.router_last[il].iter().map(|x| x.0).collect();
        let rtr_str = if ids_ba.is_empty() {
            "-".to_string()
        } else if ids_ba == ids_bb && ids_ba == ids_st {
            format!("match {:?}", &tr_ba.router_last[il])
        } else {
            format!(
                "DIFF ba={:?} bb={:?} st={:?}",
                &tr_ba.router_last[il], &tr_bb.router_last[il], &tr_st.router_last[il]
            )
        };

        println!(
            "L{il:02} {:26} | ab_attn=({cos_ab_a:.6},{max_ab_a:.5}) ab_ffn=({cos_ab_f:.6},{max_ab_f:.5}) | bs_attn=({cos_bs_a:.6},{max_bs_a:.5}) bs_ffn=({cos_bs_f:.6},{max_bs_f:.5}) | {rtr_str}",
            engine.layer_quant_tag(il)
        );
    }
}
