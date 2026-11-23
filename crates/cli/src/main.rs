use ampllm_cache::CacheBudget;
use ampllm_engine::{
    AnyEngine, EngineConfig, Glm5Engine, Hy4Engine, Inkling2Engine, KvQuantMode, Mm3Engine,
    MultiGpuMode, Qwen4Engine, SamplingParams,
};
use ampllm_gguf::{
    GgmlType, GgufFile, Glm5Config, Hy4Config, Inkling2Config, Mm3Config, ModelArch, Qwen4Config,
};
use ampllm_tokenizer::BpeTokenizer;
use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliKvMode {
    F16,
    #[value(name = "q8_0")]
    Q8_0,
}

impl From<CliKvMode> for KvQuantMode {
    fn from(v: CliKvMode) -> Self {
        match v {
            CliKvMode::F16 => KvQuantMode::F16,
            CliKvMode::Q8_0 => KvQuantMode::Q8_0,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CliMultiGpu {
    Single,
    Helper,
    Pipeline,
    Tp2,
}

impl From<CliMultiGpu> for MultiGpuMode {
    fn from(v: CliMultiGpu) -> Self {
        match v {
            CliMultiGpu::Single => MultiGpuMode::Single,
            CliMultiGpu::Helper => MultiGpuMode::Helper,
            CliMultiGpu::Pipeline => MultiGpuMode::Pipeline,
            CliMultiGpu::Tp2 => MultiGpuMode::Tp2,
        }
    }
}

fn parse_cache_budget(s: &str) -> Result<CacheBudget, String> {
    if s.eq_ignore_ascii_case("auto") {
        Ok(CacheBudget::Auto)
    } else {
        let mb: usize = s
            .parse()
            .map_err(|_| format!("expected 'auto' or MiB integer, got '{s}'"))?;
        Ok(CacheBudget::Mb(mb))
    }
}

#[derive(Parser)]
#[command(
    name = "ampllm",
    version,
    about = "Pure Rust + CUDA MoE LLM inference engine for Inkling-2-Mini, GLM-5.3-Flash, and Hy4-Flash"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Inspect {
        model: PathBuf,
    },
    Tokenize {
        model: PathBuf,
        prompt: String,
    },
    Verify {
        model: PathBuf,
        #[arg(long, default_value = "tests/golden_ref.json")]
        golden: PathBuf,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value = "auto", value_parser = parse_cache_budget)]
        expert_cache_mb: CacheBudget,
        #[arg(long, value_enum)]
        kv: Option<CliKvMode>,
        #[arg(long, value_enum)]
        multi_gpu: Option<CliMultiGpu>,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        no_cuda_graph: bool,
        #[arg(long)]
        load_profile: Option<PathBuf>,
        #[arg(long)]
        save_profile: Option<PathBuf>,
    },
    Run {
        model: PathBuf,
        #[arg(short, long)]
        prompt: String,
        #[arg(long)]
        image: Vec<PathBuf>,
        #[arg(short = 'n', long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value_t = 4096)]
        max_ctx: usize,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value = "auto", value_parser = parse_cache_budget)]
        expert_cache_mb: CacheBudget,
        #[arg(long, value_enum)]
        kv: Option<CliKvMode>,
        #[arg(long, value_enum)]
        multi_gpu: Option<CliMultiGpu>,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        no_cuda_graph: bool,
        #[arg(long)]
        no_spec: bool,
        #[arg(long)]
        load_profile: Option<PathBuf>,
        #[arg(long)]
        save_profile: Option<PathBuf>,
    },
    Serve {
        model: PathBuf,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value_t = 4096)]
        max_ctx: usize,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value = "auto", value_parser = parse_cache_budget)]
        expert_cache_mb: CacheBudget,
        #[arg(long, value_enum)]
        kv: Option<CliKvMode>,
        #[arg(long, value_enum)]
        multi_gpu: Option<CliMultiGpu>,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        no_cuda_graph: bool,
        #[arg(long)]
        no_spec: bool,
        #[arg(long)]
        load_profile: Option<PathBuf>,
        #[arg(long)]
        save_profile: Option<PathBuf>,
        #[arg(long, default_value_t = 4)]
        parallel: usize,
        #[arg(long, default_value_t = 32)]
        slot_quantum: usize,
    },
}

#[derive(Deserialize)]
struct GoldenFile {
    prompts: Vec<GoldenPrompt>,
}

#[derive(Deserialize)]
struct GoldenPrompt {
    prompt: String,
    input_tokens: Vec<u32>,
    generated_tokens: Vec<u32>,
    step0_top5: Vec<GoldenTop5>,
}

#[derive(Deserialize)]
struct GoldenTop5 {
    id: u32,
    logit: f32,
}

#[allow(clippy::too_many_arguments)]
fn build_engine_cfg(
    arch: ModelArch,
    max_ctx: usize,
    device: i32,
    expert_cache: CacheBudget,
    kv: Option<CliKvMode>,
    multi_gpu: Option<CliMultiGpu>,
    cpu_threads: Option<usize>,
    no_cuda_graph: bool,
    no_spec: bool,
    load_profile: Option<PathBuf>,
    save_profile: Option<PathBuf>,
) -> EngineConfig {
    let resolved_multi_gpu = multi_gpu.map(Into::into).unwrap_or(match arch {
        ModelArch::Inkling2 => MultiGpuMode::Single,
        ModelArch::Glm5 | ModelArch::Hy4 | ModelArch::Qwen4Exp | ModelArch::MinimaxM3 => {
            MultiGpuMode::Pipeline
        }
    });
    let resolved_kv = kv.map(Into::into).unwrap_or(match arch {
        ModelArch::Hy4 if max_ctx > 16384 => KvQuantMode::Q8_0,
        ModelArch::Qwen4Exp if max_ctx > 16384 => KvQuantMode::Q8_0,
        ModelArch::MinimaxM3 => KvQuantMode::Q8_0,
        _ => KvQuantMode::F16,
    });
    EngineConfig {
        device,
        max_ctx,
        max_batch: 512,
        kv_mode: resolved_kv,
        expert_cache,
        multi_gpu: resolved_multi_gpu,
        cpu_threads,
        use_cuda_graph: !no_cuda_graph,
        use_spec: !no_spec,
        load_profile,
        save_profile,
        ..Default::default()
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Inspect { model } => {
            let gguf = GgufFile::open(&model)?;
            let mut by_type: BTreeMap<String, (usize, u64)> = BTreeMap::new();
            for t in &gguf.tensors {
                let e = by_type.entry(format!("{:?}", t.ggml_type)).or_default();
                e.0 += 1;
                e.1 += t.byte_len as u64;
            }
            let expert_bytes: u64 = gguf
                .tensors
                .iter()
                .filter(|t| t.name.contains("_exps."))
                .map(|t| t.byte_len as u64)
                .sum();
            match gguf.architecture()? {
                ModelArch::Inkling2 => {
                    let cfg = Inkling2Config::from_gguf(&gguf)?;
                    println!(
                        "GGUF v{} | shards={} | tensors={} | kv={} | arch={}",
                        gguf.version,
                        gguf.shard_count(),
                        gguf.tensors.len(),
                        gguf.metadata.len(),
                        cfg.arch
                    );
                    println!(
                        "Config: phys_layers={} logical_layers={} embd={} vocab={}/{} heads={}:{} head_dim={} swa={} experts={}/{} ff_exp={}",
                        cfg.n_layer_phys,
                        cfg.n_layer_logical,
                        cfg.n_embd,
                        cfg.n_vocab_unpadded,
                        cfg.n_vocab_padded,
                        cfg.n_head,
                        cfg.n_head_kv,
                        cfg.head_dim,
                        cfg.sliding_window,
                        cfg.n_expert,
                        cfg.n_expert_used,
                        cfg.n_ff_exp
                    );
                    println!(
                        "Execution order ({} slots): {:?}",
                        cfg.execution_order().len(),
                        cfg.execution_order()
                    );
                }
                ModelArch::Glm5 => {
                    let cfg = Glm5Config::from_gguf(&gguf)?;
                    println!(
                        "GGUF v{} | shards={} | tensors={} | kv={} | arch={}",
                        gguf.version,
                        gguf.shard_count(),
                        gguf.tensors.len(),
                        gguf.metadata.len(),
                        cfg.arch
                    );
                    let n_kda = (0..cfg.n_layers).filter(|&l| cfg.is_kda(l)).count();
                    let n_dsa = (0..cfg.n_layers).filter(|&l| cfg.is_dsa(l)).count();
                    println!(
                        "Config: layers={} (kda={} dsa={} dense_ffn={} moe={} mtp={:?}) embd={} vocab={} heads={} hc={} experts={}/{} ff_exp={} ff_dense={} idx_topk={}",
                        cfg.n_layers,
                        n_kda,
                        n_dsa,
                        cfg.leading_dense_block_count,
                        cfg.n_layers - cfg.leading_dense_block_count,
                        cfg.mtp_layer,
                        cfg.n_embd,
                        cfg.n_vocab,
                        cfg.n_head,
                        cfg.hc,
                        cfg.n_expert,
                        cfg.n_expert_used,
                        cfg.n_ff_exp,
                        cfg.n_ff_dense,
                        cfg.idx_top_k
                    );
                }
                ModelArch::Hy4 => {
                    let cfg = Hy4Config::from_gguf(&gguf)?;
                    println!(
                        "GGUF v{} | shards={} | tensors={} | kv={} | arch={}",
                        gguf.version,
                        gguf.shard_count(),
                        gguf.tensors.len(),
                        gguf.metadata.len(),
                        cfg.arch
                    );
                    let n_swa = (0..cfg.n_layer).filter(|&l| cfg.swa[l]).count();
                    let n_dsa = (0..cfg.n_layer).filter(|&l| cfg.indexer[l]).count();
                    println!(
                        "Config: layers={} (swa={} dsa={} dense_lead={} moe={} mtp={}) embd={} vocab={} hc={} experts={}/{} shared={} latent={} ff_exp={} ff_dense={} ix_topk={}",
                        cfg.n_layer,
                        n_swa,
                        n_dsa,
                        cfg.n_dense_lead,
                        cfg.n_layer.saturating_sub(cfg.n_dense_lead),
                        cfg.n_mtp,
                        cfg.n_embd,
                        cfg.n_vocab,
                        cfg.hc,
                        cfg.n_expert,
                        cfg.n_expert_used,
                        cfg.n_expert_shared,
                        cfg.n_expert_latent,
                        cfg.n_ff_exp,
                        cfg.n_ff,
                        cfg.ix_top_k
                    );
                    if let Some(ref eng) = cfg.engram {
                        println!(
                            "Engram: layers={:?} table_rows={} external_scales={}",
                            eng.layers,
                            eng.table_rows,
                            eng.scales_path.is_some()
                        );
                    }
                }
                ModelArch::Qwen4Exp => {
                    let cfg = Qwen4Config::from_gguf(&gguf)?;
                    println!(
                        "GGUF v{} | shards={} | tensors={} | kv={} | arch={}",
                        gguf.version,
                        gguf.shard_count(),
                        gguf.tensors.len(),
                        gguf.metadata.len(),
                        cfg.arch
                    );
                    let n_qsa = (0..cfg.n_layer).filter(|&l| cfg.is_qsa(l)).count();
                    let n_gdn = cfg.n_layer.saturating_sub(n_qsa);
                    println!(
                        "Config: layers={} (gdn={} qsa={}) embd={} vocab={} heads={}:{} head_dim={} hc={}/{} experts={}/{} ff_exp={} ff_sh={} idx={}:{}x{} rope={}/{:?}",
                        cfg.n_layer,
                        n_gdn,
                        n_qsa,
                        cfg.n_embd,
                        cfg.n_vocab,
                        cfg.n_head,
                        cfg.n_head_kv,
                        cfg.head_dim,
                        cfg.hc,
                        cfg.hc_low_rank,
                        cfg.n_expert,
                        cfg.n_expert_used,
                        cfg.n_ff_exp,
                        cfg.n_ff_shared,
                        cfg.idx_heads,
                        cfg.idx_key_dim,
                        cfg.idx_top_k,
                        cfg.rope_dim,
                        cfg.rope_sections,
                    );
                    if let Some(ref ple) = cfg.ple {
                        println!(
                            "PLE: layers={:?} ngram={} heads={} conv_k={} dim={} table_rows={} type={:?}",
                            ple.layers,
                            ple.ngram_size,
                            ple.n_heads(),
                            ple.conv_kernel,
                            ple.head_dim,
                            ple.table_rows,
                            ple.table_type
                        );
                    }
                    if let Ok((mtp_opt, mm_opt)) =
                        ampllm_gguf::discover_qwen4_aux_files(&model, None, None)
                    {
                        println!(
                            "Aux files: mtp={} | mmproj={}",
                            mtp_opt
                                .as_ref()
                                .map(|p| p.display().to_string())
                                .unwrap_or_else(|| "none".to_owned()),
                            mm_opt
                                .as_ref()
                                .map(|p| p.display().to_string())
                                .unwrap_or_else(|| "none".to_owned()),
                        );
                    }
                }
                ModelArch::MinimaxM3 => {
                    let cfg = Mm3Config::from_gguf(&gguf)?;
                    println!(
                        "GGUF v{} | shards={} | tensors={} | kv={} | arch=minimax-m3",
                        gguf.version,
                        gguf.shard_count(),
                        gguf.tensors.len(),
                        gguf.metadata.len(),
                    );
                    println!(
                        "Config: layers={} (dense={} moe={}) embd={} vocab={} heads={}:{} head_dim={} experts={}/{} ff_exp={} ff_dense={} idx={}:{}x{} topk={}",
                        cfg.n_layers,
                        cfg.n_dense_layers,
                        cfg.n_layers.saturating_sub(cfg.n_dense_layers),
                        cfg.n_embd,
                        cfg.vocab_size,
                        cfg.n_heads,
                        cfg.n_kv_heads,
                        cfg.head_dim,
                        cfg.n_experts,
                        cfg.n_experts_used,
                        cfg.expert_ff,
                        cfg.n_dense_ff,
                        cfg.ix_heads,
                        cfg.ix_head_dim,
                        cfg.ix_block_size,
                        cfg.ix_top_k
                    );
                    if let Ok(aux) = ampllm_gguf::discover_mm3_aux_files(&model, None, None) {
                        println!(
                            "Aux files: indexers={} | dspark={} | vision={}",
                            aux.indexers_bin
                                .as_ref()
                                .map(|p| p.display().to_string())
                                .unwrap_or_else(|| "embedded".to_owned()),
                            aux.dspark_safetensors
                                .as_ref()
                                .map(|p| p.display().to_string())
                                .unwrap_or_else(|| "none".to_owned()),
                            aux.vision_safetensors
                                .as_ref()
                                .map(|p| p.display().to_string())
                                .unwrap_or_else(|| "none".to_owned()),
                        );
                    }
                }
            }
            println!(
                "Expert pool bytes: {:.2} GiB",
                expert_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            );
            for (ty, (cnt, bytes)) in by_type {
                println!("  {ty:8}: {cnt:4} tensors, {:.2} MiB", bytes as f64 / (1024.0 * 1024.0));
            }
            for t in &gguf.tensors {
                match t.ggml_type {
                    GgmlType::F32
                    | GgmlType::F16
                    | GgmlType::Bf16
                    | GgmlType::Q8_0
                    | GgmlType::Q4_0
                    | GgmlType::Q5_1
                    | GgmlType::Q2_0
                    | GgmlType::Q3K
                    | GgmlType::Q4K
                    | GgmlType::Q5K
                    | GgmlType::Q6K
                    | GgmlType::Iq1S
                    | GgmlType::Iq2Xxs
                    | GgmlType::Iq2Xs
                    | GgmlType::Iq2S
                    | GgmlType::Iq3Xxs
                    | GgmlType::Iq3S
                    | GgmlType::Iq4Nl
                    | GgmlType::Iq4Xs
                    | GgmlType::F8E4m3
                    | GgmlType::Nvfp4 => {}
                    other => anyhow::bail!("unsupported tensor type {other:?} on {}", t.name),
                }
            }
            println!("ALL TENSORS VALIDATED.");
        }
        Cmd::Tokenize { model, prompt } => {
            let gguf = GgufFile::open(&model)?;
            let tok = BpeTokenizer::from_gguf(&gguf)?;
            let ids = tok.encode(&prompt, true)?;
            let roundtrip = tok.decode(&ids, false);
            println!("ids={ids:?}");
            println!("roundtrip={roundtrip:?}");
        }
        Cmd::Verify {
            model,
            golden,
            device,
            expert_cache_mb,
            kv,
            multi_gpu,
            threads,
            no_cuda_graph,
            load_profile,
            save_profile,
        } => {
            let golden_str = std::fs::read_to_string(&golden)?;
            let golden_data: GoldenFile = serde_json::from_str(&golden_str)?;
            let gguf = GgufFile::open(&model)?;
            if gguf.architecture()? != ModelArch::Inkling2 {
                anyhow::bail!(
                    "verify currently supports Inkling-2-Mini golden parity files; use `ampllm inspect` or `ampllm run` for GLM-5.3-Flash and Hy4-Flash"
                );
            }
            let tok = BpeTokenizer::from_gguf(&gguf)?;
            let engine_cfg = build_engine_cfg(
                ModelArch::Inkling2,
                2048,
                device,
                expert_cache_mb,
                kv,
                multi_gpu,
                threads,
                no_cuda_graph,
                false,
                load_profile,
                save_profile,
            );
            println!(
                "Loading Inkling-2-Mini (mode={:?}, cache={expert_cache_mb:?}, kv={:?})...",
                engine_cfg.multi_gpu,
                engine_cfg.kv_mode
            );
            let mut engine = Inkling2Engine::load(&gguf, engine_cfg)?;
            println!("Engine ready: {}", engine.cache_summary());
            println!("Running verification against {}...", golden.display());
            let mut all_ok = true;
            for (idx, gp) in golden_data.prompts.iter().enumerate() {
                let ids = tok.encode(&gp.prompt, true)?;
                assert_eq!(ids, gp.input_tokens, "tokenizer mismatch on prompt {idx}");
                let (gen, pf_tps, dc_tps, top5) =
                    engine.generate_greedy(&ids, gp.generated_tokens.len())?;
                let matched = gen
                    .iter()
                    .zip(gp.generated_tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count();
                println!(
                    "Prompt {idx}: matched {matched}/{} tokens | prefill={pf_tps:.1} tok/s | decode={dc_tps:.1} tok/s",
                    gp.generated_tokens.len()
                );
                println!("  Step-0 top5 got:      {top5:?}");
                let ref_top5: Vec<(u32, f32)> = gp.step0_top5.iter().map(|x| (x.id, x.logit)).collect();
                println!("  Step-0 top5 expected: {ref_top5:?}");
                println!("  Text got: {:?}", tok.decode(&gen, false));
                if matched != gp.generated_tokens.len() {
                    all_ok = false;
                    println!("  GOT: {gen:?}");
                    println!("  EXP: {:?}", gp.generated_tokens);
                }
            }
            let stats = engine.cache_stats();
            println!(
                "Cache stats: hits={} misses={} helper_hits={} promotions={} hit_rate={:.2}%",
                stats.hits,
                stats.misses,
                stats.helper_hits,
                stats.promotions,
                stats.vram_hit_rate() * 100.0
            );
            engine.save_profile()?;
            if !all_ok {
                anyhow::bail!("Verification mismatch against golden_ref.json");
            }
            println!("VERIFICATION PASSED 100% EXACT TOKEN PARITY!");
        }
        Cmd::Run {
            model,
            prompt,
            image,
            max_tokens,
            max_ctx,
            device,
            expert_cache_mb,
            kv,
            multi_gpu,
            threads,
            no_cuda_graph,
            no_spec,
            load_profile,
            save_profile,
        } => {
            let gguf = GgufFile::open(&model)?;
            let tok = BpeTokenizer::from_gguf(&gguf)?;
            let arch = gguf.architecture()?;
            let engine_cfg = build_engine_cfg(
                arch,
                max_ctx,
                device,
                expert_cache_mb,
                kv,
                multi_gpu,
                threads,
                no_cuda_graph,
                no_spec,
                load_profile,
                save_profile,
            );
            let mut engine = match arch {
                ModelArch::Inkling2 => {
                    AnyEngine::Inkling2(Box::new(Inkling2Engine::load(&gguf, engine_cfg)?))
                }
                ModelArch::Glm5 => AnyEngine::Glm5(Box::new(Glm5Engine::load(gguf, &tok, engine_cfg)?)),
                ModelArch::Hy4 => AnyEngine::Hy4(Box::new(Hy4Engine::load(gguf, engine_cfg)?)),
                ModelArch::Qwen4Exp => Qwen4Engine::load(gguf, engine_cfg)
                    .map(|e| AnyEngine::Qwen4(Box::new(e)))?,
                ModelArch::MinimaxM3 => Mm3Engine::load(gguf, engine_cfg)
                    .map(|e| AnyEngine::Mm3(Box::new(e)))?,
            };

            let (gen, pf_tps, dc_tps) = if !image.is_empty() {
                if !engine.has_vision() {
                    anyhow::bail!(
                        "--image requires a multimodal model loaded with mmproj vision weights"
                    );
                }
                let mut parts: Vec<serde_json::Value> = image
                    .iter()
                    .map(|p| {
                        serde_json::json!({
                            "type": "image_url",
                            "image_url": { "url": p.to_string_lossy().as_ref() }
                        })
                    })
                    .collect();
                parts.push(serde_json::json!({ "type": "text", "text": prompt }));
                let msg = ampllm_server::routes::ChatMessage {
                    role: "user".to_owned(),
                    content: Some(serde_json::Value::Array(parts)),
                    ..Default::default()
                };
                if arch == ModelArch::Qwen4Exp {
                    let (rendered, _, _, img_sources) = ampllm_server::routes::format_qwen4_chat(
                        &[msg],
                        None,
                        None,
                        true,
                        "xhigh",
                        true,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let raw_ids = tok.encode(&rendered, true)?;
                    let (eval_ids, spans) = ampllm_server::vision::prepare_and_encode_qwen4_images(
                        &mut engine,
                        &raw_ids,
                        &img_sources,
                    )?;
                    let params = SamplingParams {
                        temperature: 0.0,
                        top_p: 1.0,
                        seed: 42,
                        max_tokens,
                        stop_token_ids: ampllm_server::qwen4_stop_ids(&tok),
                    };
                    let res = engine.generate_stream_with_qwen4_images(
                        &eval_ids,
                        &spans,
                        &params,
                        |_| true,
                    )?;
                    (res.generated_tokens, res.prefill_tps, res.decode_tps)
                } else if arch == ModelArch::MinimaxM3 {
                    let (rendered, _, _, img_sources) = ampllm_server::routes::format_mm3_chat(
                        &[msg],
                        None,
                        None,
                        "disabled",
                        None,
                        true,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let raw_ids = tok.encode(&rendered, true)?;
                    let (eval_ids, spans) = ampllm_server::vision::prepare_and_encode_mm3_images(
                        &mut engine,
                        &raw_ids,
                        &img_sources,
                    )?;
                    let params = SamplingParams {
                        temperature: 0.0,
                        top_p: 1.0,
                        seed: 42,
                        max_tokens,
                        stop_token_ids: ampllm_server::mm3_stop_ids(&tok),
                    };
                    let res = engine.generate_stream_with_mm3_images(
                        &eval_ids,
                        &spans,
                        &params,
                        |_| true,
                    )?;
                    (res.generated_tokens, res.prefill_tps, res.decode_tps)
                } else {
                    let hytk = tok
                        .hy4_tag_suffix()
                        .map(|s| format!(":{s}"))
                        .unwrap_or_else(|| ":opensource".to_owned());
                    let (rendered, _, _, img_sources) = ampllm_server::routes::format_hy4_chat(
                        &[msg],
                        None,
                        None,
                        true,
                        "high",
                        &hytk,
                        true,
                    )
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let raw_ids = tok.encode(&rendered, true)?;
                    let (eval_ids, spans) = ampllm_server::vision::prepare_and_encode_hy4_images(
                        &mut engine,
                        &raw_ids,
                        &img_sources,
                    )?;
                    let params = SamplingParams {
                        temperature: 0.0,
                        top_p: 1.0,
                        seed: 42,
                        max_tokens,
                        stop_token_ids: ampllm_server::hy4_stop_ids(&tok, &hytk),
                    };
                    let res =
                        engine.generate_stream_with_images(&eval_ids, &spans, &params, |_| true)?;
                    (res.generated_tokens, res.prefill_tps, res.decode_tps)
                }
            } else {
                let mut ids = tok.encode(&prompt, true)?;
                if arch == ModelArch::Hy4 && !prompt.starts_with("<｜") {
                    if let Some(bos) = tok.bos_token_id {
                        if ids.first() != Some(&bos) {
                            ids.insert(0, bos);
                        }
                    }
                }
                let (gen, pf_tps, dc_tps, _) = engine.generate_greedy(&ids, max_tokens)?;
                (gen, pf_tps, dc_tps)
            };

            let decoded = if arch == ModelArch::Hy4 {
                tok.decode_hy4_aliased(&gen)
            } else if arch == ModelArch::Qwen4Exp {
                tok.decode_skip_control(&gen)
            } else {
                tok.decode(&gen, false)
            };
            println!("{decoded}");
            let stats = engine.cache_stats();
            eprintln!(
                "\n[prefill={pf_tps:.1} tok/s, decode={dc_tps:.1} tok/s | hit_rate={:.1}% promotions={} | {}]",
                stats.vram_hit_rate() * 100.0,
                stats.promotions,
                engine.cache_summary()
            );
            engine.save_profile()?;
        }
        Cmd::Serve {
            model,
            host,
            port,
            max_ctx,
            device,
            expert_cache_mb,
            kv,
            multi_gpu,
            threads,
            no_cuda_graph,
            no_spec,
            load_profile,
            save_profile,
            parallel,
            slot_quantum,
        } => {
            let gguf = GgufFile::open(&model)?;
            let tok = BpeTokenizer::from_gguf(&gguf)?;
            let arch = gguf.architecture()?;
            let default_name = match arch {
                ModelArch::Inkling2 => "inkling2-mini",
                ModelArch::Glm5 => "glm-5.3-flash",
                ModelArch::Hy4 => "hy4-flash",
                ModelArch::Qwen4Exp => "qwen3.8-flash-next",
                ModelArch::MinimaxM3 => "minimax-m3.1-flash",
            };
            let model_name = model
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(default_name)
                .to_owned();
            let engine_cfg = build_engine_cfg(
                arch,
                max_ctx,
                device,
                expert_cache_mb,
                kv,
                multi_gpu,
                threads,
                no_cuda_graph,
                no_spec,
                load_profile,
                save_profile,
            );
            let engine = match arch {
                ModelArch::Inkling2 => {
                    AnyEngine::Inkling2(Box::new(Inkling2Engine::load(&gguf, engine_cfg)?))
                }
                ModelArch::Glm5 => AnyEngine::Glm5(Box::new(Glm5Engine::load(gguf, &tok, engine_cfg)?)),
                ModelArch::Hy4 => AnyEngine::Hy4(Box::new(Hy4Engine::load(gguf, engine_cfg)?)),
                ModelArch::Qwen4Exp => Qwen4Engine::load(gguf, engine_cfg)
                    .map(|e| AnyEngine::Qwen4(Box::new(e)))?,
                ModelArch::MinimaxM3 => Mm3Engine::load(gguf, engine_cfg)
                    .map(|e| AnyEngine::Mm3(Box::new(e)))?,
            };
            let server_cfg = ampllm_server::ServerConfig::from_cli(parallel, slot_quantum);
            println!(
                "Serving {model_name} (parallel={} quantum={}): {}",
                server_cfg.parallel_slots,
                server_cfg.slot_quantum,
                engine.cache_summary()
            );
            let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(ampllm_server::run_server_with_config(
                engine, tok, model_name, addr, server_cfg,
            ))?;
        }
    }
    Ok(())
}
