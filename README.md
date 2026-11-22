# ampllm

Pure Rust + custom CUDA (`sm_86`) + AVX-512/AVX2 Mixture-of-Experts (MoE) inference engine tuned for **NVIDIA RTX 3090** GPUs paired with host DDR4/DDR5 CPUs.

**Zero cuBLAS and zero Python runtime dependencies:** quantized GEMV, Tensor-Core MMQ (`Q3_K`–`Q8_0`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `NVFP4`), attention (FlashMLA split-KV & relative-position SWA/global), KDA linear attention, Hyper-Connections, and speculative MTP drafting execute in standalone `nvcc` kernels, orchestrated from Rust with pinned physical-core CPU MoE offloading and mapped host RAM communication.

---

## Supported Models & Hardware Requirements

| Model | GGUF Architecture | GPUs Required | CPU Requirements | Host RAM | Key Engine Features |
| --- | --- | --- | --- | --- | --- |
| **Inkling-2-Mini** | `inkling2_loop`, `inkling2` | **1× or 2× 24 GB NVIDIA GPU** (`sm_86`+) | x86-64 with AVX2 + FMA (AVX-512 VNNI supported) | ~32+ GiB (for hybrid CPU+GPU cache) | Looped 48-slot schedule, Block `AttnRes`, relative-bias SWA/global attention, Loop-Core-3x prioritized VRAM expert cache, `F16`/`Q8_0` KV, CUDA Graphs, `single`/`helper`/`pipeline`/`tp2` |
| **GLM-5.3-Flash** | `glm5*` (e.g. `glm5-next`) | **2× 24 GB NVIDIA GPUs** (`sm_86`+) | **x86-64 CPU with AVX-512** (`avx512f`, `avx512bw`, `avx512vl`, `avx512dq`, `avx512vnni`, `avx512vpopcntdq`) | **~95+ GiB free** (`IQ2_S` 3-shard GGUF) | 2-GPU auto-balanced pipeline split, HC-4 Sinkhorn, hybrid KDA linear attention + DSA NoPE MLA + Lightning Indexer, online EWMA dynamic VRAM expert cache, NextN MTP speculative decoding with KDA rollback, 256K context (`--max-ctx 262144`) |

> **Notes:**
> - Systems with PCIe PHB topologies where `cudaDeviceCanAccessPeer == 0` are supported automatically via mapped pinned host RAM bounce buffers and system-scope doorbells.
> - Default CUDA compute capability is `sm_86` (RTX 3090 / A10 / A40 / A6000); override with `AMPLLM_CUDA_ARCH` if needed (for instance `sm_89` for RTX 4090).
> - Running GLM-5.3-Flash on fewer than 2 GPUs or without the required AVX-512 ISA extensions exits cleanly with a descriptive error.

---

## Prerequisites & Building

### Dependencies
- **Rust toolchain** (`cargo`, `rustc` 2021 edition)
- **CUDA Toolkit** (`nvcc`, `libcudart`, `libcuda`)
- **C++17 compiler** (`g++` recommended)
- Linux `x86_64`

### Build & Test

```bash
git clone <your-repo-url> ampllm
cd ampllm

# Standard release build
cargo build --release

# Run full workspace tests (CUDA parity tests automatically serialize GPU access)
cargo test --release
```

Binary output: `./target/release/ampllm`

### Build-Time Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `NVCC` | auto-detected | Path to `nvcc` (checks `NVCC`, `CUDA_HOME/bin/nvcc`, `CUDA_PATH/bin/nvcc`, `/usr/local/cuda/bin/nvcc`, `/opt/cuda/bin/nvcc`, then `PATH`) |
| `CUDA_HOME` / `CUDA_PATH` | unset | CUDA Toolkit root directory used for `nvcc` and `lib64`/`lib` search paths |
| `AMPLLM_CUDA_ARCH` | `sm_86` | Target CUDA architecture passed to `nvcc -arch=...` |
| `CXX` | `g++` | C++17 compiler for CPU AVX-512 kernels |
| `AMPLLM_CPU_TUNE` | `generic` | Value passed to `-mtune=...` when compiling `iq_avx512_glm5.cpp` (e.g. `znver4` or `native`) |
| `AMPLLM_NVCC_VERBOSE` | unset | Set to `1` to print all `ptxas` register/SMEM info lines during build (non-zero register spills and warnings are always surfaced) |

---

## Quick Start: Inkling-2-Mini

### 1. Inspect Model & Tokenizer

```bash
./target/release/ampllm inspect /path/to/Inkling-2-Mini-IQ4_NL.gguf
./target/release/ampllm tokenize /path/to/Inkling-2-Mini-IQ4_NL.gguf "Hello world"
```

### 2. Verify Greedy Token Parity

```bash
# Single GPU, auto VRAM expert cache + CUDA Graph fast path
./target/release/ampllm verify /path/to/Inkling-2-Mini-IQ4_NL.gguf --golden tests/golden_ref.json

# Hybrid GPU+CPU expert cache (e.g. 8 GiB VRAM expert budget)
./target/release/ampllm verify /path/to/Inkling-2-Mini-IQ4_NL.gguf --expert-cache-mb 8192

# Q8_0 KV cache (halves KV VRAM usage)
./target/release/ampllm verify /path/to/Inkling-2-Mini-IQ4_NL.gguf --kv q8_0

# Dual RTX 3090 modes
./target/release/ampllm verify /path/to/Inkling-2-Mini-IQ4_NL.gguf --multi-gpu helper --expert-cache-mb auto
./target/release/ampllm verify /path/to/Inkling-2-Mini-IQ4_NL.gguf --multi-gpu pipeline
./target/release/ampllm verify /path/to/Inkling-2-Mini-IQ4_NL.gguf --multi-gpu tp2
```

### 3. CLI Generation & HTTP Serving

```bash
# Single-prompt greedy generation across 2 GPUs (tensor-parallel)
./target/release/ampllm run /path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  -p "Write a Rust function to compute fibonacci:" -n 128 --multi-gpu tp2

# OpenAI-compatible HTTP/SSE server
./target/release/ampllm serve /path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 --port 8080 --max-ctx 32768 --multi-gpu tp2 --kv f16
```

Warm expert cache profiles can be persisted and pre-loaded on Inkling-2-Mini hybrid runs:

```bash
./target/release/ampllm serve /path/to/model.gguf --expert-cache-mb 8192 --save-profile inkling_cache.json
./target/release/ampllm serve /path/to/model.gguf --expert-cache-mb 8192 --load-profile inkling_cache.json
```

Query the Inkling-2-Mini server:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inkling2-mini",
    "messages": [{"role": "user", "content": "Explain lock-free ring buffers concisely."}],
    "max_tokens": 256,
    "temperature": 0.0
  }'
```

---

## Quick Start: GLM-5.3-Flash

GLM-5.3-Flash GGUF releases are commonly split across multiple shards (for example `...-00001-of-00003.gguf`). Pass the **first shard path**; `ampllm` discovers and memory-maps all shards automatically.

### 1. Inspect & Run

```bash
# Validate GGUF metadata, layer counts, KDA/DSA/MoE/MTP config, and all tensor layouts
./target/release/ampllm inspect /path/to/GLM-5.3-Flash-IQ2_S-00001-of-00003.gguf

# Single greedy CLI generation
./target/release/ampllm run /path/to/GLM-5.3-Flash-IQ2_S-00001-of-00003.gguf \
  -p "Compare Raft and Multi-Paxos consensus." -n 128 --max-ctx 8192 --multi-gpu pipeline
```

### 2. Serve with 256K Context & Optional Expert Profile

```bash
# Recommended dual-RTX 3090 launch (auto-splits layers across GPU 0 and GPU 1, NextN MTP enabled)
./target/release/ampllm serve /path/to/GLM-5.3-Flash-IQ2_S-00001-of-00003.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --max-ctx 262144 \
  --multi-gpu pipeline

# Optional: preload an existing warm expert profile (JSON or STRP v1 binary)
./target/release/ampllm serve /path/to/GLM-5.3-Flash-IQ2_S-00001-of-00003.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --max-ctx 262144 \
  --multi-gpu pipeline \
  --load-profile /path/to/expert-profile-glm.bin
```

Flags useful for GLM-5.3-Flash:
- `--max-ctx 262144`: enables 256K context (DSA latent KV windows automatically use mapped pinned host RAM beyond `24576` tokens while keeping recent/selected tokens resident on GPU).
- `--load-profile <path>` / `--save-profile <path>`: load or save warm expert hotness rankings (`.json` generated by `--save-profile`, or pre-built `STRP` v1 `.bin` profiles).
- `--no-spec`: disable NextN MTP speculative decoding (`AMPLLM_VERIFY_M` controls speculative verify batch width, default `1`).
- Note: GLM-5.3-Flash always uses the dual-GPU auto-balanced pipeline split and automatic free-VRAM expert cache sizing.

### 3. OpenAI Chat API & Controlling Reasoning (`<think>`)

`ampllm serve` renders the native GLM-5.3-Flash chat template (`[gMASK]<sop>...`) and streams `<think>...</think>` blocks cleanly as `reasoning_content` (and `content` after `</think>`):

- **Reasoning effort**: pass `"reasoning_effort": "low" | "high" | "max"` (default `"max"`).
- **Disable thinking**: pass `"reasoning_effort": "off"` (or `"none"` / `"minimal"` / `false`) or `"chat_template_kwargs": {"enable_thinking": false}`.

**Streaming chat with thinking disabled:**

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.3-Flash",
    "messages": [{"role": "user", "content": "Write a C++20 lock-free SPSC ring buffer."}],
    "max_tokens": 512,
    "temperature": 0.0,
    "stream": true,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

**Streaming chat with high reasoning effort:**

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "GLM-5.3-Flash",
    "messages": [{"role": "user", "content": "Prove correctness of Peterson mutual exclusion."}],
    "max_tokens": 1024,
    "temperature": 1.0,
    "top_p": 0.95,
    "stream": true,
    "reasoning_effort": "high"
  }'
```

Any OpenAI SDK client works by pointing `base_url` at `http://127.0.0.1:8080/v1` (no API key required).

---

## HTTP Endpoints & Prometheus Metrics

Both `/v1/*` and root OpenAI aliases (`/models`, `/chat/completions`, `/completions`) are served so clients that append either `/v1/...` or `/models` directly work out of the box:

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | JSON readiness check (`status`, `model`, `max_context`) |
| `/v1/models`, `/models` | `GET` | OpenAI model list (includes `context_length`) |
| `/v1/chat/completions`, `/chat/completions` | `POST` | OpenAI chat completions (`stream: false` and SSE `stream: true`) with exact prefix KV reuse, `reasoning_content`, OpenAI `tools`/`tool_choice`/`tool_calls`, and per-request `usage` + `timings` |
| `/v1/completions`, `/completions` | `POST` | Raw prompt completion (`stream: false` and SSE `stream: true`) |
| `/metrics` | `GET` | Prometheus text exposition format |

Tool calling (`tools`, `tool_choice`: `"auto"` | `"none"` | `"required"` | `{"type":"function","function":{"name":"..."}}`, and multi-turn `tool` role responses) is natively formatted and parsed for both **GLM-5.3-Flash** (`<tool_call>...<arg_key>...<arg_value>...`) and **Inkling-2-Mini** (`<|content_invoke_tool_json|>`).

Exported `/metrics` series:
- `ampllm_requests_total`, `ampllm_tokens_prefilled_total`, `ampllm_tokens_generated_total`, `ampllm_prefix_reuse_tokens_total`
- `ampllm_cache_hits_total`, `ampllm_cache_helper_hits_total`, `ampllm_cache_misses_total`, `ampllm_cache_hit_ratio`, `ampllm_cache_promotions_total`
- `ampllm_vram_free_bytes{gpu="..."}`, `ampllm_vram_total_bytes{gpu="..."}`

---

## Measured Benchmarks

Test platform: **2× NVIDIA RTX 3090** (24 GB GDDR6X, PCIe PHB, P2P disabled), **AMD Ryzen 9 7950X** (16 physical cores, AVX-512 VNNI), 124 GiB DDR5.

### 1. Inkling-2-Mini Verification (`Inkling-2-Mini-IQ4_NL.gguf`, 300 W)

`tests/golden_ref.json` — P0: 5 prompt + 32 greedy decode tokens; P1: 9 prompt + 32 greedy decode tokens.

| Mode | Expert Cache | KV Mode | Parity (P0 / P1) | Hit Rate | GPU 1 Free VRAM | Prefill (P0 / P1 tok/s) | Decode (P0 / P1 tok/s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `single` (CUDA Graph) | `auto` (3072/3072) | `f16` (227.7 MiB) | 32/32 & 32/32 | 100.00% | — | 175.9 / 184.5 | 107.1 / 105.7 |
| `tp2` (Tensor Parallel 2) | Sharded (100%) | `f16` | 32/32 & 32/32 | 100.00% | 12,848.3 MiB | 178.1 / 243.8 | 107.6 / 111.0 |
| `pipeline` (L0–11 \| L12–23) | `auto` (1536+1536) | `f16` | 32/32 & 32/32 | 100.00% | 11,856.3 MiB | 168.4 / 190.1 | 97.3 / 96.7 |
| `helper` (Disjoint 2-GPU) | `auto` (1536+1536) | `f16` | 32/32 & 32/32 | 100.00% | 15,246.3 MiB | 182.1 / 218.5 | 97.6 / 96.6 |

### 2. Inkling-2-Mini Throughput (`Inkling-2-Mini-UD-Q4_K_XL.gguf`, 128-token Greedy Decode, 300 W)

- **P0:** `"The capital of France is"` (`5 -> 128` tokens)
- **P1:** `"The Mixture of Experts architecture"` (`9 -> 128` tokens)
- **P2:** 718-token technical MoE prompt (`718 -> 128` tokens)

| Mode | Expert Cache | KV Mode | Prompt | Prefill | Decode |
| --- | --- | --- | --- | --- | --- |
| `single` (1× 3090 Resident) | `auto` (3072/3072) | `f16` | P0 (`5 -> 128`) | 169.7 tok/s | 108.0 tok/s |
| `single` (1× 3090 Resident) | `auto` (3072/3072) | `f16` | P1 (`9 -> 128`) | 183.9 tok/s | 108.0 tok/s |
| `single` (1× 3090 Resident) | `auto` (3072/3072) | `f16` | P2 (`718 -> 128`) | 166.3 tok/s | 100.9 tok/s |
| `single` (1× 3090 Resident) | `auto` (3072/3072) | `q8_0` | P0 (`5 -> 128`) | 168.8 tok/s | 101.9 tok/s |
| `single` (1× 3090 Hybrid) | `7680 MiB` (1677/3072) | `f16` | P0 (`5 -> 128`) | 65.7 tok/s | 56.6 tok/s |
| `single` (1× 3090 Hybrid) | `8192 MiB` (1786/3072) | `f16` | P0 (`5 -> 128`) | 71.7 tok/s | 60.1 tok/s |
| `tp2` (2× 3090 Tensor Parallel) | Sharded (100%) | `f16` | P0 (`5 -> 128`) | 179.7 tok/s | 113.0 tok/s |
| `tp2` (2× 3090 Tensor Parallel) | Sharded (100%) | `f16` | P2 (`718 -> 128`) | 225.3 tok/s | 107.6 tok/s |
| `pipeline` (2× 3090 Pipeline) | `auto` (1536+1536) | `f16` | P0 (`5 -> 128`) | 159.8 tok/s | 100.2 tok/s |
| `helper` (2× 3090 Disjoint) | `auto` (1536+1536) | `f16` | P0 (`5 -> 128`) | 173.6 tok/s | 95.3 tok/s |

### 3. GLM-5.3-Flash Throughput (`IQ2_S`, 256K Context, **200 W Power Limit**)

Served with `--max-ctx 262144 --multi-gpu pipeline` and an expert cache profile (`split_layer=23`, `VRAM experts 4364/12384 (35.2%)`, `GPU0=23922 MiB`, `GPU1=23322 MiB`, Host RAM `90 GiB`).

| Workload | Prompt Tokens | Output Tokens | Engine Prefill | Engine Decode | Notes |
| --- | --- | --- | --- | --- | --- |
| Short prompts (3 greedy × 128t + 2 sampled × 256t) | 43–101 | 128 / 256 | 85.36–127.90 tok/s | **38.71 tok/s median** (`36.22–41.17`) | NextN MTP speculative decoding active |
| `4K` needle-in-haystack | 5,003 | 64 | **557.03 tok/s** | **34.35 tok/s** | `wall=10.824s`, exact needle retrieved |
| `16K` needle-in-haystack | 19,914 | 59 | **823.8 tok/s** | **34.95 tok/s** | `wall=25.841s`, exact needle retrieved |
| Short follow-up after 16K | 59 | 128 | 96.44 tok/s | 35.21 tok/s | Clean post-long-context state |

---

## Runtime Tuning & Debug Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `AMPLLM_VERIFY_M` | `1` | GLM-5.3 speculative verification batch width (`1..4`; default pipelines NextN MTP with M=1 verification) |
| `AMPLLM_NGRAM` | `0` | Enable n-gram draft fallback (`1`) when speculative batching (`verify_m >= 2`) |
| `AMPLLM_DRAFT_CHAIN` | auto | Enable chained MTP drafting when `verify_m >= 3` (`0` to disable) |
| `AMPLLM_SPLIT_LAYER` | auto | Override GLM-5.3 GPU 0/1 pipeline split layer (`10..n_layers-1`) |
| `AMPLLM_MTP_VRAM` | `40` | Max GLM-5.3 NextN MTP experts pinned in GPU 1 VRAM |
| `AMPLLM_FREEZE_CACHE` | unset | Disable online dynamic expert promotions/evictions when set |
| `AMPLLM_PREFILL_CHUNK` | `512` | Core GLM-5.3 prefill chunk size (`128..1024`) |
| `AMPLLM_PREFILL_MAX_CHUNK` | `8192` | Max lent prefill chunk size (`AMPLLM_PREFILL_CHUNK..8192`) |
| `AMPLLM_CPU_TOPOLOGY` | auto | Set `cross` for interleaved cross-CCD core groups; default splits L3 caches |
| `AMPLLM_CPU_SMT` | `0` | Set `1` to include SMT sibling threads per CCD group |
| `AMPLLM_CPU_MAX_NT` | auto | Override max token-expert rows offloaded to CPU during GLM-5.3 prefill |
| `AMPLLM_PROFILE` | unset | Log per-request decode, MTP acceptance, hit-rate, and CPU timings |
| `AMPLLM_PREFILL_PROFILE` | unset | Log per-stage prefill timing breakdown (`1`/`true`) |
| `AMPLLM_PHASE_PROFILE` | unset | Collect per-layer CUDA event breakdowns (`AMPLLM_PHASE_BREAKDOWN`) |

---

## Workspace Architecture

- [`crates/gguf`](crates/gguf): Zero-copy mmap GGUF v2/v3 parser, GGML block types & reference kernels, and `Inkling2Config` / `Glm5Config` parsers.
- [`crates/tokenizer`](crates/tokenizer): GGUF GPT-2 byte-level BPE tokenizer with Inkling-2 and GLM-5 pre-tokenization.
- [`crates/cuda-kernels`](crates/cuda-kernels): Standalone `nvcc` CUDA kernels and safe Rust FFI/device memory wrappers (`sm_86` default).
- [`crates/cpu-kernels`](crates/cpu-kernels): Physical-core spin-barrier thread pool, AVX2/AVX-512 Inkling-2 kernels, and AVX-512 VNNI/VPOPCNTDQ GLM-5.3 MoE kernels.
- [`crates/cache`](crates/cache): Mapped pinned host arenas, VRAM expert LRU/LFU cache planning, and `CacheProfile` / `STRP` v1 persistence.
- [`crates/engine`](crates/engine): Unified `Inkling2Engine` (`single`, `helper`, `pipeline`, `tp2`) and `Glm5Engine` inference backends, prefix reuse, and Leviathan speculative sampling.
- [`crates/server`](crates/server): `axum` OpenAI-compatible HTTP/SSE server and Prometheus metrics exporter.
- [`crates/cli`](crates/cli): `ampllm` CLI (`inspect`, `tokenize`, `verify`, `run`, `serve`).
- [`docs/INKLING2_SPEC.md`](docs/INKLING2_SPEC.md): Exact Inkling-2-Mini GGUF tensor and forward-pass specification.

---

## License

MIT License — see [LICENSE](LICENSE).
