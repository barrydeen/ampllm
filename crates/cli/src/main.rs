use ampllm_cache::CacheBudget;
use ampllm_engine::{AnyEngine, EngineConfig, Glm5Engine, Inkling2Engine, KvQuantMode, MultiGpuMode};
use ampllm_gguf::{GgmlType, GgufFile, Glm5Config, Inkling2Config, ModelArch};
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
    about = "Pure Rust + CUDA MoE LLM inference engine for Inkling-2-Mini and GLM-5.3-Flash"
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
        #[arg(long, value_enum, default_value_t = CliKvMode::F16)]
        kv: CliKvMode,
        #[arg(long, value_enum, default_value_t = CliMultiGpu::Single)]
        multi_gpu: CliMultiGpu,
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
        #[arg(short = 'n', long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value_t = 4096)]
        max_ctx: usize,
        #[arg(long, default_value_t = 0)]
        device: i32,
        #[arg(long, default_value = "auto", value_parser = parse_cache_budget)]
        expert_cache_mb: CacheBudget,
        #[arg(long, value_enum, default_value_t = CliKvMode::F16)]
        kv: CliKvMode,
        #[arg(long, value_enum, default_value_t = CliMultiGpu::Single)]
        multi_gpu: CliMultiGpu,
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
        #[arg(long, value_enum, default_value_t = CliKvMode::F16)]
        kv: CliKvMode,
        #[arg(long, value_enum, default_value_t = CliMultiGpu::Single)]
        multi_gpu: CliMultiGpu,
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
                    | GgmlType::Q3K
                    | GgmlType::Q4K
                    | GgmlType::Q5K
                    | GgmlType::Q6K
                    | GgmlType::Iq2S
                    | GgmlType::Iq3Xxs
                    | GgmlType::Iq3S
                    | GgmlType::Iq4Nl
                    | GgmlType::Iq4Xs
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
                    "verify currently supports Inkling-2-Mini golden parity files; use `ampllm inspect` or `ampllm run` for GLM-5.3-Flash"
                );
            }
            let tok = BpeTokenizer::from_gguf(&gguf)?;
            let engine_cfg = EngineConfig {
                device,
                max_ctx: 2048,
                max_batch: 512,
                kv_mode: kv.into(),
                expert_cache: expert_cache_mb,
                multi_gpu: multi_gpu.into(),
                cpu_threads: threads,
                use_cuda_graph: !no_cuda_graph,
                load_profile,
                save_profile,
                ..Default::default()
            };
            println!(
                "Loading Inkling-2-Mini (mode={multi_gpu:?}, cache={expert_cache_mb:?}, kv={kv:?})..."
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
            let engine_cfg = EngineConfig {
                device,
                max_ctx,
                max_batch: 512,
                kv_mode: kv.into(),
                expert_cache: expert_cache_mb,
                multi_gpu: multi_gpu.into(),
                cpu_threads: threads,
                use_cuda_graph: !no_cuda_graph,
                use_spec: !no_spec,
                load_profile,
                save_profile,
                ..Default::default()
            };
            let mut engine = match arch {
                ModelArch::Inkling2 => {
                    AnyEngine::Inkling2(Box::new(Inkling2Engine::load(&gguf, engine_cfg)?))
                }
                ModelArch::Glm5 => AnyEngine::Glm5(Box::new(Glm5Engine::load(gguf, &tok, engine_cfg)?)),
            };
            let ids = tok.encode(&prompt, true)?;
            let (gen, pf_tps, dc_tps, _) = engine.generate_greedy(&ids, max_tokens)?;
            println!("{}", tok.decode(&gen, false));
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
        } => {
            let gguf = GgufFile::open(&model)?;
            let tok = BpeTokenizer::from_gguf(&gguf)?;
            let arch = gguf.architecture()?;
            let default_name = match arch {
                ModelArch::Inkling2 => "inkling2-mini",
                ModelArch::Glm5 => "glm-5.3-flash",
            };
            let model_name = model
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or(default_name)
                .to_owned();
            let engine_cfg = EngineConfig {
                device,
                max_ctx,
                max_batch: 512,
                kv_mode: kv.into(),
                expert_cache: expert_cache_mb,
                multi_gpu: multi_gpu.into(),
                cpu_threads: threads,
                use_cuda_graph: !no_cuda_graph,
                use_spec: !no_spec,
                load_profile,
                save_profile,
                ..Default::default()
            };
            let engine = match arch {
                ModelArch::Inkling2 => {
                    AnyEngine::Inkling2(Box::new(Inkling2Engine::load(&gguf, engine_cfg)?))
                }
                ModelArch::Glm5 => AnyEngine::Glm5(Box::new(Glm5Engine::load(gguf, &tok, engine_cfg)?)),
            };
            println!("Serving {model_name}: {}", engine.cache_summary());
            let addr: std::net::SocketAddr = format!("{host}:{port}").parse()?;
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(ampllm_server::run_server(engine, tok, model_name, addr))?;
        }
    }
    Ok(())
}
