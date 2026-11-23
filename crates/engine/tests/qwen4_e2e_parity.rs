use ampllm_cache::CacheBudget;
use ampllm_engine::{EngineConfig, KvQuantMode, MultiGpuMode, Qwen4Engine, SamplingParams};
use ampllm_gguf::GgufFile;
use std::path::Path;

fn model_path() -> String {
    std::env::var("AMPLLM_QWEN4_MODEL")
        .expect("set AMPLLM_QWEN4_MODEL to Qwen3.8-Flash-Next GGUF shard 1")
}

fn read_id_file(path: &str) -> Vec<u32> {
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {path}: {e}"))
        .split_whitespace()
        .map(|s| s.parse::<u32>().expect("valid u32 token id"))
        .collect()
}

fn read_f32_file(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "{path} byte length not multiple of 4");
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn read_sve1_vision_ref(path: &str) -> (usize, usize, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"));
    assert!(bytes.len() >= 20, "SVE1 header too short");
    let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    assert_eq!(magic, 0x3145_5653, "bad SVE1 magic");
    let n_toks = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let nx = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let ny = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
    assert_eq!(n_toks, nx * ny);
    let feats: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes[20..]).to_vec();
    assert_eq!(feats.len(), n_toks * dim);
    (nx, ny, feats)
}

fn cosine_and_mae(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut mae = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xf = x as f64;
        let yf = y as f64;
        dot += xf * yf;
        na += xf * xf;
        nb += yf * yf;
        mae += (xf - yf).abs();
    }
    let cos = dot / (na.sqrt() * nb.sqrt()).max(1e-30);
    (cos, mae / (a.len().max(1) as f64))
}

fn top5(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut v: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1));
    v.truncate(5);
    v
}

#[test]
#[ignore = "loads full 70GB Qwen3.8-Flash-Next model on 2x RTX 3090"]
fn test_qwen4_e2e_parity_and_bench() {
    let model_p = model_path();
    assert!(
        Path::new(&model_p).exists(),
        "missing model file {model_p}"
    );

    let gguf = GgufFile::open(&model_p).expect("open Qwen3.8 GGUF");
    let cfg = EngineConfig {
        device: 0,
        max_ctx: 20480,
        max_batch: 512,
        kv_mode: KvQuantMode::F16,
        expert_cache: CacheBudget::Auto,
        multi_gpu: MultiGpuMode::Pipeline,
        cpu_threads: None,
        use_cuda_graph: true,
        load_profile: std::env::var("AMPLLM_QWEN4_EXPERT_PROFILE")
            .ok()
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from),
        save_profile: None,
        dma_prefill_threshold: 8,
        use_spec: true,
    };
    let mut engine = Qwen4Engine::load(gguf, cfg).expect("load Qwen4Engine");
    let initial_mtp_k = engine.mtp_k();
    assert!(initial_mtp_k > 0, "expected MTP drafter loaded");

    let mut failures: Vec<String> = Vec::new();

    // 1. Vision Parity
    let rgb = std::fs::read("/tmp/qwen4_test_rgb.bin").expect("read /tmp/qwen4_test_rgb.bin");
    let (nx, ny, vis_feats) = engine
        .encode_image(&rgb, 256, 256)
        .expect("encode 256x256 RGB image");
    let (ref_nx, ref_ny, ref_vis) = read_sve1_vision_ref("/tmp/qwen4_vision_ref.bin");
    assert_eq!((nx, ny), (ref_nx, ref_ny));
    let (vis_cos, vis_mae) = cosine_and_mae(&vis_feats, &ref_vis);
    println!(
        "[CHECK 1 Vision] shape=({nx},{ny}) len={} cos_sim={vis_cos:.6} mae={vis_mae:.6}",
        vis_feats.len()
    );
    if vis_cos <= 0.999 {
        failures.push(format!("Vision cosine similarity {vis_cos:.6} <= 0.999"));
    }

    // 2. Logit Parity + 32-token greedy parity (MTP OFF)
    let short_ids = read_id_file("/tmp/qwen4_short_ids.txt");
    let ref_logits0 = read_f32_file("/tmp/qwen4_strata_logits.bin.0");
    let ref_short_gen = read_id_file("/tmp/qwen4_strata_short_gen.txt");

    engine.set_mtp_k(0);
    let logits0 = engine
        .forward_logits_debug(&short_ids)
        .expect("forward_logits_debug step 0");
    let (cos0, mae0) = cosine_and_mae(&logits0, &ref_logits0);
    let t5_0 = top5(&logits0);
    let r5_0 = top5(&ref_logits0);
    println!(
        "[CHECK 2a Logits0 t=26 vs ref] cos_sim={cos0:.6} mae={mae0:.6} ampllm_top5={t5_0:?} ref_top5={r5_0:?}"
    );

    let logits0_t1 = engine
        .forward_logits_debug_stepwise(&short_ids)
        .expect("forward_logits_debug_stepwise step 0");
    let (cos0_t1, mae0_t1) = cosine_and_mae(&logits0_t1, &ref_logits0);
    let t1_5_0 = top5(&logits0_t1);
    println!(
        "[CHECK 2a' Logits0 t=1 vs ref] cos_sim={cos0_t1:.6} mae={mae0_t1:.6} t1_top5={t1_5_0:?}"
    );
    let (cos_self, mae_self) = cosine_and_mae(&logits0, &logits0_t1);
    println!(
        "[CHECK 2a'' Logits0 t=26 vs t=1] cos_sim={cos_self:.6} mae={mae_self:.6}"
    );

    if cos0 <= 0.99 {
        failures.push(format!("Logits0 t=26 cosine similarity {cos0:.6} <= 0.99"));
    }
    if cos_self <= 0.99 {
        failures.push(format!("Logits0 t=26 vs t=1 cosine similarity {cos_self:.6} <= 0.99"));
    }
    if t5_0[0].0 != r5_0[0].0 {
        failures.push(format!("Logits0 top-1 mismatch: {} != {}", t5_0[0].0, r5_0[0].0));
    }
    if t5_0[0].0 != ref_short_gen[0] as usize {
        failures.push(format!(
            "Logits0 {} != ref_short_gen[0] {}",
            t5_0[0].0, ref_short_gen[0]
        ));
    }

    let mut ids_step1 = short_ids.clone();
    ids_step1.push(ref_short_gen[0]);
    let logits1 = engine
        .forward_logits_debug(&ids_step1)
        .expect("forward_logits_debug step 1");
    let t5_1 = top5(&logits1);
    println!(
        "[CHECK 2b Logits1 (next={})] ampllm_top5={t5_1:?}",
        ref_short_gen[1]
    );
    if t5_1[0].0 != ref_short_gen[1] as usize {
        failures.push(format!(
            "Logits1 top-1 mismatch {} != ref_short_gen[1] {}",
            t5_1[0].0, ref_short_gen[1]
        ));
    }

    let (gen_off, pp26_off, tg32_off, _) = engine
        .generate_greedy(&short_ids, 32)
        .expect("greedy generate 32 tokens (MTP OFF)");
    let exact_matches = gen_off
        .iter()
        .zip(ref_short_gen.iter())
        .filter(|(a, b)| a == b)
        .count();
    let prefix_matches = gen_off
        .iter()
        .zip(ref_short_gen.iter())
        .take_while(|(a, b)| a == b)
        .count();
    println!(
        "[CHECK 2c MTP OFF 32-tok] prefix_match={prefix_matches}/32 total_match={exact_matches}/32 pp={pp26_off:.1} tg={tg32_off:.1}\n  gen_off={gen_off:?}\n  ref_gen={ref_short_gen:?}"
    );
    if gen_off != ref_short_gen {
        failures.push(format!(
            "MTP OFF 32-token greedy mismatch vs Strata (prefix_match={prefix_matches}/32, total_match={exact_matches}/32)"
        ));
    }

    // 3. MTP ON (K=3) vs OFF equivalence
    engine.set_mtp_k(initial_mtp_k);
    let params32 = SamplingParams {
        temperature: 0.0,
        max_tokens: 32,
        ..Default::default()
    };
    let res_on = engine
        .generate_stream(&short_ids, &params32, |_| true)
        .expect("MTP ON generate 32 tokens");
    let mtp_matches = res_on
        .generated_tokens
        .iter()
        .zip(gen_off.iter())
        .filter(|(a, b)| a == b)
        .count();
    let acc_rate = if res_on.draft_n > 0 {
        100.0 * (res_on.draft_accepted as f64) / (res_on.draft_n as f64)
    } else {
        0.0
    };
    println!(
        "[CHECK 3 MTP ON K={initial_mtp_k}] match={mtp_matches}/32 draft_acc={}/{} ({acc_rate:.1}%) pp={:.1} tg={:.1}\n  gen_on={:?}",
        res_on.draft_accepted,
        res_on.draft_n,
        res_on.prefill_tps,
        res_on.decode_tps,
        res_on.generated_tokens,
    );
    if res_on.generated_tokens != gen_off {
        failures.push(format!(
            "MTP ON greedy tokens diverged from MTP OFF ({mtp_matches}/32)"
        ));
    }

    // 4. Benchmark Snapshot vs Strata 0.1.64 baselines (p=26/n=71 -> 80.0 tg, p=4096 -> 945.5/66.5, p=16384 -> 2273.0/100.6)
    engine.set_adaptive_mtp(true);
    engine.reset_cache_stats();
    let params71 = SamplingParams {
        temperature: 0.0,
        max_tokens: 71,
        ..Default::default()
    };
    let res_p26 = engine
        .generate_stream(&short_ids, &params71, |_| true)
        .expect("bench p=26 n=71");
    println!(
        "[BENCH p=26 n=71] ampllm: pp={:.1} tok/s, tg={:.1} tok/s (acc {}/{}) | strata_0.1.64: pp=128.7, tg=80.0 | {}",
        res_p26.prefill_tps, res_p26.decode_tps, res_p26.draft_accepted, res_p26.draft_n, engine.cache_summary()
    );

    let p4096_ids = read_id_file("/tmp/qwen4_p4096_ids.txt");
    engine.reset_cache_stats();
    let res_p4096 = engine
        .generate_stream(&p4096_ids, &params32, |_| true)
        .expect("bench p=4096 n=32");
    println!(
        "[BENCH p={} n=32] ampllm: pp={:.1} tok/s, tg={:.1} tok/s (acc {}/{}) | strata_0.1.64: pp=945.5, tg=66.5 | {}",
        p4096_ids.len(),
        res_p4096.prefill_tps,
        res_p4096.decode_tps,
        res_p4096.draft_accepted,
        res_p4096.draft_n,
        engine.cache_summary()
    );

    let p16384_ids = read_id_file("/tmp/qwen4_p16384_ids.txt");
    engine.reset_cache_stats();
    let res_p16384 = engine
        .generate_stream(&p16384_ids, &params32, |_| true)
        .expect("bench p=16384 n=32");
    println!(
        "[BENCH p={} n=32] ampllm: pp={:.1} tok/s, tg={:.1} tok/s (acc {}/{}) | strata_0.1.64: pp=2273.0, tg=100.6 | {}",
        p16384_ids.len(),
        res_p16384.prefill_tps,
        res_p16384.decode_tps,
        res_p16384.draft_accepted,
        res_p16384.draft_n,
        engine.cache_summary()
    );

    assert!(
        failures.is_empty(),
        "Qwen4 E2E parity checks had failures:\n - {}",
        failures.join("\n - ")
    );
}
