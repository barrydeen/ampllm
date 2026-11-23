use ampllm_engine::hy4::Hy4Engine;
use ampllm_engine::{EngineConfig, MultiGpuMode, SamplingParams};
use ampllm_gguf::GgufFile;
use anyhow::Result;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("AMPLLM_HY4_MODEL")
        .unwrap_or_else(|_| "Hy4-Flash-Maya-L-IQ4_XS-00001-of-00002.gguf".to_owned())
}

fn read_token_ids(path: &str) -> Result<Vec<u32>> {
    let txt = std::fs::read_to_string(path)?;
    let ids = txt
        .split_whitespace()
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(ids)
}

fn greedy_params(max_tokens: usize) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        seed: 42,
        max_tokens,
        stop_token_ids: Vec::new(),
    }
}

fn run_vision(engine: &mut Hy4Engine) -> Result<()> {
    let rgb = std::fs::read("/tmp/hy4_test_rgb.bin")?;
    anyhow::ensure!(rgb.len() == 256 * 256 * 3, "unexpected rgb len {}", rgb.len());

    let t0 = Instant::now();
    let _cold = engine.encode_image(&rgb, 256, 256)?;
    let cold_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t1 = Instant::now();
    let warm = engine.encode_image(&rgb, 256, 256)?;
    let warm_ms = t1.elapsed().as_secs_f64() * 1e3;

    let ref_bytes = std::fs::read("/tmp/hy4_vision_ref.bin")?;
    anyhow::ensure!(ref_bytes.len() % 4 == 0, "unexpected ref len {}", ref_bytes.len());
    let ref_f32: Vec<f32> = ref_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    anyhow::ensure!(
        warm.len() == ref_f32.len(),
        "vision length mismatch: ours={} vs ref={}",
        warm.len(),
        ref_f32.len()
    );

    let mut dot = 0.0f64;
    let mut n2_a = 0.0f64;
    let mut n2_b = 0.0f64;
    let mut diff2 = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&a, &b) in warm.iter().zip(ref_f32.iter()) {
        let af = a as f64;
        let bf = b as f64;
        dot += af * bf;
        n2_a += af * af;
        n2_b += bf * bf;
        let d = (af - bf).abs();
        diff2 += d * d;
        max_abs = max_abs.max(d as f32);
    }
    let cos_sim = dot / (n2_a.sqrt() * n2_b.sqrt()).max(1e-12);
    let rel_l2 = diff2.sqrt() / n2_b.sqrt().max(1e-12);

    println!(
        "VISION: len={} (72x3072={}) | cold={:.1} ms warm={:.1} ms | cos_sim={:.6} max_abs={:.5} rel_l2={:.6}",
        warm.len(),
        72 * 3072,
        cold_ms,
        warm_ms,
        cos_sim,
        max_abs,
        rel_l2
    );
    println!("VISION ours[..6]: {:?}", &warm[..6]);
    println!("VISION  ref[..6]: {:?}", &ref_f32[..6]);
    Ok(())
}

fn run_prompt(engine: &mut Hy4Engine, ids: &[u32], max_tokens: usize, label: &str) -> Result<()> {
    engine.reset()?;
    let res = engine.generate_stream(ids, &greedy_params(max_tokens), |_| true)?;
    let acc_pct = if res.draft_n > 0 {
        100.0 * (res.draft_accepted as f64) / (res.draft_n as f64)
    } else {
        0.0
    };
    println!(
        "{label}: prompt_eval={} prefill={:.2} tok/s | gen={} decode={:.2} tok/s | draft={}/{} ({:.1}%)",
        res.prompt_tokens_evaluated,
        res.prefill_tps,
        res.generated_tokens.len(),
        res.decode_tps,
        res.draft_accepted,
        res.draft_n,
        acc_pct
    );
    engine.print_profile(label);
    let head_n = res.generated_tokens.len().min(20);
    println!("{label} first {head_n}: {:?}", &res.generated_tokens[..head_n]);
    if max_tokens <= 128 {
        println!("{label} all: {:?}", res.generated_tokens);
    }
    Ok(())
}

fn main() -> Result<()> {
    let mode = std::env::var("VERIFY_MODE").unwrap_or_else(|_| "all".to_string());
    let max_ctx: usize = std::env::var("VERIFY_CTX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(131072);
    let max_batch: usize = std::env::var("VERIFY_BATCH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let cpu_threads: usize = std::env::var("VERIFY_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);

    let t0 = Instant::now();
    let gguf = GgufFile::open(model_path())?;
    let cfg = EngineConfig {
        max_ctx,
        max_batch,
        multi_gpu: MultiGpuMode::Pipeline,
        use_spec: true,
        cpu_threads: Some(cpu_threads),
        ..Default::default()
    };
    let mut engine = Hy4Engine::load(gguf, cfg)?;
    let load_s = t0.elapsed().as_secs_f64();
    let (pinned, tot_moe) = engine.cold_pinned_layers();
    println!(
        "LOADED in {:.2}s (max_ctx={}, max_draft={}, cold_pinned={pinned}/{tot_moe}): {}",
        load_s,
        engine.max_ctx,
        engine.max_draft,
        engine.cache_summary()
    );

    let short_ids = read_token_ids("/tmp/hy4_short_ids.txt")?;
    match mode.as_str() {
        "vision" => run_vision(&mut engine)?,
        "dump" => run_prompt(&mut engine, &short_ids, 2, "DUMP_RUN")?,
        "dump+vision" => {
            run_vision(&mut engine)?;
            run_prompt(&mut engine, &short_ids, 2, "DUMP_RUN")?;
        }
        "gen" => run_prompt(&mut engine, &short_ids, 128, "GEN_128")?,
        "bench_short" => {
            run_prompt(&mut engine, &short_ids, 128, "SHORT_COLD")?;
            run_prompt(&mut engine, &short_ids, 128, "SHORT_WARM")?;
        }
        "bench_long" => {
            let p2048 = read_token_ids("/tmp/hy4_p2048_ids.txt")?;
            let p4096 = read_token_ids("/tmp/hy4_p4096_ids.txt")?;
            // Warmup kernel paths with a short run first
            run_prompt(&mut engine, &short_ids, 16, "WARMUP_16")?;
            run_prompt(&mut engine, &p2048, 32, "P2048")?;
            run_prompt(&mut engine, &p4096, 32, "P4096")?;
        }
        "all" => {
            run_vision(&mut engine)?;
            run_prompt(&mut engine, &short_ids, 128, "GEN_128")?;
        }
        other => anyhow::bail!("unknown VERIFY_MODE={other}"),
    }
    Ok(())
}
