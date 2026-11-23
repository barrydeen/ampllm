# AmpLLM

Pure Rust + custom CUDA (`sm_86`) + AVX-512/AVX2 Mixture-of-Experts (MoE) inference engine **built specifically for the NVIDIA RTX 30-series (Ampere `sm_86`) GPU generation** paired with DDR4/DDR5 host RAM and x86-64 CPUs.

Designed for consumer and workstation Ampere GPUs — **RTX 3090 / 3090 Ti (24 GB)**, **RTX 3080 Ti / 3080 (10–12 GB)**, **RTX 3070 / 3070 Ti (8 GB)**, and **RTX 3060 (12 GB)** (plus RTX A6000 / A40 / A10) — **AmpLLM** combines custom Tensor-Core MMQ/GEMV kernels with physical-core CPU MoE offloading so frontier 100B–300B+ MoE models run comfortably on 1× or 2× RTX 30-series cards even without NVLink (`cudaDeviceCanAccessPeer == 0`).

**Highlights:**
- **Zero cuBLAS & zero Python runtime dependencies:** standalone `nvcc` kernels compiled for `sm_86`, orchestrated from Rust over mapped pinned host RAM and lock-free system-scope doorbells.
- **5 supported frontier MoE architectures** with native multimodal **Vision** (`--image` CLI & OpenAI `image_url`), **Tool Calling** (`tools` / `tool_choice` / `tool_calls`), **Reasoning** (`reasoning_content`, `reasoning_effort`), and **Speculative Decoding** (NextN MTP / DSpark).
- **Multi-slot concurrent agent serving (`--parallel 4`, `--slot-quantum 32`):** host-RAM sequence KV/state snapshotting keeps **100%** of VRAM dedicated to resident MoE expert caches while serving up to 4 (configurable `1..=16`) concurrent coding subagents with zero swap overhead when running single-stream.

---

## Supported Models & Hugging Face GGUF Links

> **Scope:** **AmpLLM** is an **architecture-specialized engine** (not a generic llama.cpp wrapper). Only the 5 MoE architectures below are supported — unrelated GGUF architectures (Llama, dense Qwen2/3, Mistral, Gemma, DeepSeek) will be rejected cleanly at load time.

| Model | GGUF `general.architecture` | Recommended Hugging Face GGUF Repos | GPUs & CPU ISA | Supported GGUF Weight Quants | Key Engine Features |
| --- | --- | --- | --- | --- | --- |
| **Inkling-2-Mini** | `inkling2_loop`, `inkling2` | **[`unsloth/Inkling-2-Mini-GGUF`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)** | **1× or 2× RTX 30-series** (AVX2 + FMA; AVX-512 optional) | `NVFP4`, `Q4_0`, `IQ4_NL`, `IQ4_XS`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `F16`/`BF16`/`F32` | Looped 48-slot schedule, Block `AttnRes`, relative-bias SWA/global attention, `F16`/`Q8_0` KV, CUDA Graphs, `single` / `helper` / `pipeline` / `tp2`, tool calling |
| **Qwen3.8-Flash-Next** | `qwen4exp`, `qwen4_exp` | **[`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** *(base: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next))* | **1× or 2× 24 GB RTX 3090** *(2× recommended)* + **AVX-512 VNNI** CPU | `Q2_0`, `Q3_K`, `Q4_K`, `Q5_1`, `Q5_K`, `Q6_K`, `Q8_0`, `IQ4_NL`, `IQ4_XS`, `F16`/`BF16`/`F32` *(optional MTP: `Q8_0`/`BF16`/`F32` + `mmproj`)* | 2-GPU pipeline, HC-4, hybrid GDN linear attention + QSA sparse indexer, cold-pinned MoE + zero-copy mmap PLE table, async GPU/CPU expert adaptation, NextN MTP (`K=4`), ViT Vision, XML `<tool_call>` |
| **GLM-5.3-Flash** | `glm5*` (e.g. `glm5-next`) | **[`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)** & **[`peasantsmith/GLM-5.3-Flash-Maya-GGUF`](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF)** | **2× RTX 30-series GPUs required** *(2× RTX 3090 recommended)* + **AVX-512 VNNI** CPU | `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`, `Q3_K`, `Q4_K`, `Q5_K`, `Q5_1`, `Q6_K`, `Q8_0`, `F16`/`BF16`/`F32` | 2-GPU auto-balanced pipeline split, HC-4 Sinkhorn, KDA linear attention + DSA NoPE MLA + Lightning Indexer, online EWMA VRAM cache, NextN MTP with KDA rollback, 256K context (`--max-ctx 262144`) |
| **Hy4-Flash** | `hy_v4_flash`, `hy4*` | **[`peasantsmith/Hy4-Flash-Maya-GGUF`](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF)**, **[`Xylolabs/Hy4-flash-GGUF`](https://huggingface.co/Xylolabs/Hy4-flash-GGUF)**, **[`AngelSlim/Hy4-flash-GGUF`](https://huggingface.co/AngelSlim/Hy4-flash-GGUF)** *(base: [`tencent/Hy4-flash`](https://huggingface.co/tencent/Hy4-flash))* | **1× or 2× 24 GB RTX 3090** *(2× recommended)* + **AVX-512 VNNI** CPU | `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`, `Q3_K`, `Q4_K`, `Q5_K`, `Q5_1`, `Q6_K`, `Q8_0`, `F16`/`BF16`/`F32` *(+ Engram mmap + `mmproj`)* | 2-GPU pipeline, iHC-4 Hyper-Connections, hybrid SWA/DSA MLA + Pooled Indexer (`Q8_0`/`F16` KV), LatentMoE (`512/16` routed + shared), zero-copy disk-mmap Engram table (`59 GiB`), NextN MTP (`draft=3`), zero-dedicated-VRAM SigLIP Vision, 131K context |
| **MiniMax M3.1 Flash** | `minimax-m3`, `minimax_m3`, `mm3` | **[`peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF`](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF)** *(plug-and-play)* · [`unsloth/MiniMax-M3.1-Flash-GGUF`](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF) *(see quant note below; base: [`MiniMaxAI/MiniMax-M3.1-Flash`](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash))* | **1× or 2× 24 GB RTX 3090** *(2× recommended)* + **AVX-512 VNNI** CPU | `Q6_K` `token_embd.weight`; **`IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`** routed MoE; `Q8_0`/`Q6_K` dense; `*-maya-indexers.bin` + SafeTensors `dspark` & `vision` | Adaptive `<mm:think>`, MSA + DSpark Markov speculative drafting (`dspark=ON/OFF`), Int8 Hadamard-V / BF16 KV, hybrid AVX-512/GPU MoE, ViT Vision (`1.6 GiB` BF16), `<minimax:tool_call>` |

> **GGUF Quant Compatibility Notes:**
> - **Quant formats without MoE matmul kernels (`Q2_K`, `IQ1_M`, `Q4_1`, `Q5_0`):** For Unsloth GGUF repos, choose supported Dynamic quant folders — e.g. pick `UD-Q2_0`, `UD-Q3_K_XL`, `UD-IQ4_XS`, or `UD-Q4_K_XL` on Qwen3.8-Flash-Next rather than `UD-Q2_K_XL` or `UD-IQ1_M`; pick `UD-IQ2_S` / `UD-IQ3_XXS` / `UD-Q3_K_XL` / `UD-IQ4_XS` / `UD-Q4_K_XL` on GLM-5.3-Flash rather than `UD-IQ1_S` / `UD-IQ2_XXS`.
> - **MiniMax M3.1 Flash companion & quant requirements:** `mm3` MoE kernels are specialized for `IQ1_S`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, and `IQ3_XXS` with `Q6_K` token embeddings, plus dense-layer indexer weights (`*-maya-indexers.bin`) and optional SafeTensors `dspark` (`2.1 GiB`) and `vision` (`1.6 GiB`) companions. Standard `UD-Q4_K_XL`/`Q8` GGUFs without those layouts will fail on load; text-only inference works without the `vision` or `dspark` SafeTensors companions.
> - **Dual-GPU PCIe & CPU notes:** Systems with `cudaDeviceCanAccessPeer == 0` (PCIe PHB without NVLink) work automatically via mapped pinned host RAM bounce buffers and doorbells. GLM-5.3-Flash, Qwen3.8-Flash-Next, Hy4-Flash, and MiniMax M3.1 Flash require an **AVX-512** CPU (`avx512f`, `avx512bw`, `avx512vl`, `avx512dq`, `avx512vnni`, `avx512vpopcntdq`) for offloaded MoE experts; Inkling-2-Mini runs on **AVX2+**.
> - **Target CUDA arch:** Defaults to **`sm_86`** (RTX 30-series / Ampere). Override at build time via `AMPLLM_CUDA_ARCH` if compiling for another NVIDIA generation (e.g., `sm_89` for RTX 4090).

---

## Recommended Quant Size by VRAM + Host RAM (RTX 30-Series Guide)

In **AmpLLM**, **non-expert layers, KV cache, and as many hot MoE experts as fit live in RTX 30-series VRAM**, while remaining cold experts stay pinned/staged in **host RAM** (and massive static embedding tables like Hy4's `59.0 GiB` Engram and Qwen3.8's `26.8 GiB` PLE shard stay **zero-copy disk-mmap'd**, consuming only hot OS page cache). When running multiple subagents (`--parallel 4`), paused slot snapshots use host RAM (capped by `AMPLLM_SLOT_CACHE_MB`, default `8192` MiB), leaving **0 MiB stolen from VRAM**.

| Hardware Tier (RTX 30-Series + System RAM) | Inkling-2-Mini | Qwen3.8-Flash-Next *(Unsloth GGUF)* | Hy4-Flash | GLM-5.3-Flash | MiniMax M3.1 Flash |
| --- | --- | --- | --- | --- | --- |
| **8–12 GB VRAM** *(1× RTX 3060 12GB / 3070 / 3080)* + **32 GB RAM** *(AVX2+)* | **`UD-IQ4_XS`** (`14.65 GiB`) or **`UD-Q4_K_XL`** (`16.42 GiB`, `--expert-cache-mb 6144`–`8192`) | — *(too large)* | — *(too large)* | — *(requires 2 GPUs)* | — *(too large)* |
| **24 GB VRAM** *(1× RTX 3090 / 3090 Ti)* + **32–64 GB RAM** | **`UD-Q4_K_XL`** (`16.42 GiB`, 100% in VRAM) or **`UD-Q8_K_XL`** (`21.30 GiB`) | **`UD-Q2_0`** (`65.70 GiB` total = `38.9 GiB` active + `26.8 GiB` mmap PLE; needs 64 GB RAM + AVX-512) | — *(needs 96+ GB RAM)* | — *(requires 2 GPUs)* | — *(needs 96+ GB RAM)* |
| **24 GB VRAM** *(1× RTX 3090)* + **96–128 GB RAM** *(AVX-512)* | **`UD-Q4_K_XL`** / **`UD-Q8_K_XL`** (`16.4–21.3 GiB`) | **`UD-Q3_K_XL`** (`83.32 GiB`), **`UD-IQ4_XS`** (`87.24 GiB`), or **`UD-Q4_K_XL`** (`103.69 GiB`) | [`Maya-S IQ2_S`](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF) (`49.3 GiB` + `59 GiB` mmap) or [`Maya-M IQ3_XXS`](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF) (`63.7 GiB` + mmap) | — *(requires 2 GPUs)* | [`Maya-XS IQ1_S`](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF) (`86.61 GiB` + `dspark`) |
| **48 GB VRAM** *(2× RTX 3090 / 3090 Ti)* + **96 GB RAM** *(AVX-512)* | **`UD-Q4_K_XL`** / **`UD-Q8_K_XL`** (`--multi-gpu tp2`, 100% VRAM) | **`UD-Q3_K_XL`** (`83.32 GiB`) or **`UD-IQ4_XS`** (`87.24 GiB`) + `mtp-Q8_0` (`3.85 GiB`) | [`Maya-M IQ3_XXS`](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF) (`63.7 GiB` + `59 GiB` mmap) or [`Maya-L IQ4_XS`](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF) (`83.4 GiB` + mmap) | Unsloth **`UD-IQ2_S`** / [`Maya-S24 IQ2_XXS_S`](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) (`~88 GiB`) | [`Maya-XS IQ1_S`](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF) (`86.61 GiB`) |
| **48 GB VRAM** *(2× RTX 3090 / 3090 Ti)* + **128 GB RAM** *(AVX-512 — Reference Rig)* | **`UD-Q8_K_XL`** (`21.30 GiB`, `--multi-gpu tp2`) | **`UD-Q4_K_XL`** (`103.69 GiB`: `76.9 GiB` active + `26.8 GiB` mmap PLE + `mmproj` + `mtp-Q8_0`) ***(tested & verified)*** | [`Maya-L IQ4_XS`](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF) (`83.4 GiB` + `59 GiB` mmap) ***(verified)*** or Xylolabs `Q4_K_M` (`91.1 + 59 GiB`) | [`Maya-M IQ2_S`](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF) (`108.04 GiB`, **verified**) or Unsloth **`UD-IQ3_XXS`** (`112.10 GiB`) | [`Maya-S IQ2_XXS`](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF) (`114.05 GiB` + `dspark` + `vision`, **verified**) |
| **48 GB VRAM** *(2× RTX 3090)* + **192–256+ GB RAM** *(AVX-512)* | **`UD-Q8_K_XL`** (`21.30 GiB`) | **`UD-Q5_K_XL`** (`147.42 GiB`), **`UD-Q6_K_XL`** (`157.55 GiB`), or **`Q8_0`** (`175.30 GiB`) | Xylolabs **`Q5_K_M`** (`105.6 + 59 GiB`), **`Q6_K`** (`121.0 + 59 GiB`), or **`Q8_0`** (`154.1 + 59 GiB`) | Unsloth **`UD-Q3_K_XL`** (`137.40 GiB`), **`UD-IQ4_XS`** (`146.05 GiB`), **`UD-Q4_K_XL`** (`185.99 GiB`), or **`UD-Q5_K_XL`** (`223.80 GiB`) | `Maya-S` (`114.05 GiB`) or `IQ3_XXS` Maya-format GGUF (`~149 GiB` w/ `Q6_K` embd + indexers) |

> **Tip for 128 GB RAM + 2× RTX 3090:** With `AMPLLM_SLOT_CACHE_MB=8192` and `--kv q8_0`, 4 concurrent subagents fit easily alongside `104–114 GiB` models (~40–50% of MoE experts resident in 48 GB VRAM, cold experts pinned/stashed in ~70 GB host RAM).

---

## Prerequisites & Building

### Dependencies
- **NVIDIA RTX 30-series GPU(s)** (or compatible CUDA GPU) + **CUDA Toolkit** (`nvcc`, `libcudart`, `libcuda`)
- **Rust toolchain** (`cargo`, `rustc` 2021 edition)
- **C++17 compiler** (`g++` recommended)
- **Linux `x86_64`**

### Build & Test

```bash
git clone https://github.com/barrydeen/ampllm.git
cd ampllm

# Standard release build (defaults to sm_86 / RTX 30-series)
cargo build --release --bin ampllm

# Run full workspace tests (CUDA parity tests automatically serialize GPU access)
cargo test --release --workspace
```

Binary output: `./target/release/ampllm`

### Build-Time Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `NVCC` | auto-detected | Path to `nvcc` (checks `NVCC`, `CUDA_HOME/bin/nvcc`, `CUDA_PATH/bin/nvcc`, `/usr/local/cuda/bin/nvcc`, `/opt/cuda/bin/nvcc`, then `PATH`) |
| `CUDA_HOME` / `CUDA_PATH` | unset | CUDA Toolkit root directory used for `nvcc` and `lib64`/`lib` search paths |
| `AMPLLM_CUDA_ARCH` | `sm_86` | Target CUDA architecture passed to `nvcc -arch=...` (`sm_86` = RTX 30-series Ampere) |
| `CXX` | `g++` | C++17 compiler for CPU AVX-512/AVX2 kernels |
| `AMPLLM_CPU_TUNE` | `generic` | Value passed to `-mtune=...` when compiling CPU C++ kernels (e.g. `znver4` or `native`) |
| `AMPLLM_NVCC_VERBOSE` | unset | Set to `1` to print all `ptxas` register/SMEM info lines during build |

---

## Quick Start by Model

All multi-shard GGUF models load automatically when you pass the **first shard path** (`...-00001-of-0000N.gguf`). Companion files (`mmproj`, `mtp`, `.engram`, `*-maya-indexers.bin`, SafeTensors `dspark` / `vision`) are auto-discovered in the model directory or parent/sibling companion folders.

### 1. Inkling-2-Mini

```bash
# Inspect GGUF metadata & verify golden parity
./target/release/ampllm inspect /path/to/Inkling-2-Mini-IQ4_NL.gguf
./target/release/ampllm verify /path/to/Inkling-2-Mini-IQ4_NL.gguf --golden tests/golden_ref.json

# CLI generation (2× RTX 3090 TP2, or 1× GPU with --expert-cache-mb 8192)
./target/release/ampllm run /path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  -p "Write a Rust function to compute fibonacci:" -n 128 --multi-gpu tp2

# OpenAI-compatible HTTP/SSE server (4 parallel agent slots by default)
./target/release/ampllm serve /path/to/Inkling-2-Mini-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 --port 8080 --max-ctx 32768 --multi-gpu tp2 --kv f16 --parallel 4
```

### 2. Qwen3.8-Flash-Next

Unsloth GGUF weights + optional `mmproj` (`F16`/`BF16`) + optional `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` are auto-discovered.

```bash
# Inspect all shards and companion files
./target/release/ampllm inspect /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf

# Multimodal CLI generation
./target/release/ampllm run /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --image /path/to/photo.png -p "Describe this image concisely." -n 128 --max-ctx 8192 --multi-gpu pipeline

# Multi-slot OpenAI HTTP/SSE server (supports text, tools, vision, and <think> reasoning)
./target/release/ampllm serve /path/to/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --host 127.0.0.1 --port 8080 --max-ctx 32768 --multi-gpu pipeline --parallel 4 --slot-quantum 32
```

- **Thinking effort (`reasoning_effort`):** supports `"none"`/`"off"` (`"enable_thinking": false`), `"low"`, `"medium"`, and `"xhigh"`/`"high"` (default).
- **Vision + Tools:** full OpenAI `image_url` (`data:...;base64,...` or local path) and `<tool_call><function=...><parameter=...>` formatting/parsing.

### 3. GLM-5.3-Flash

```bash
./target/release/ampllm inspect /path/to/GLM-5.3-Flash-IQ2_S-00001-of-00003.gguf

# Dual-RTX 3090 server with 256K context & NextN MTP speculative decoding
./target/release/ampllm serve /path/to/GLM-5.3-Flash-IQ2_S-00001-of-00003.gguf \
  --host 127.0.0.1 --port 8080 --max-ctx 262144 --multi-gpu pipeline \
  --load-profile /path/to/expert-profile-glm.bin
```

- **Thinking effort (`reasoning_effort`):** `"off"`/`"none"`/`"minimal"`/`false` (`"enable_thinking": false`), `"low"`, `"high"`, `"max"` (default).

### 4. Hy4-Flash

```bash
./target/release/ampllm inspect /path/to/Hy4-Flash-Maya-L-IQ4_XS-00001-of-00002.gguf

# Multimodal CLI run
./target/release/ampllm run /path/to/Hy4-Flash-Maya-L-IQ4_XS-00001-of-00002.gguf \
  --image /path/to/photo.png -p "Describe this image in detail." -n 128 --max-ctx 8192 --kv q8_0 --multi-gpu pipeline

# Dual-RTX 3090 server with 131K context, Q8_0 KV, vision & NextN MTP
./target/release/ampllm serve /path/to/Hy4-Flash-Maya-L-IQ4_XS-00001-of-00002.gguf \
  --host 127.0.0.1 --port 8080 --max-ctx 131072 --kv q8_0 --multi-gpu pipeline --parallel 4
```

### 5. MiniMax M3.1 Flash

```bash
./target/release/ampllm inspect /path/to/MiniMax-M3.1-Flash-Maya-S-IQ2_XXS-00001-of-00003.gguf

# Multimodal CLI run
./target/release/ampllm run /path/to/MiniMax-M3.1-Flash-Maya-S-IQ2_XXS-00001-of-00003.gguf \
  --image /path/to/photo.png -p "Describe this image concisely." -n 128 --max-ctx 16384 --kv q8_0 --multi-gpu pipeline

# Multi-slot HTTP/SSE server (auto-discovers indexers + SafeTensors DSpark drafter & Vision tower)
./target/release/ampllm serve /path/to/MiniMax-M3.1-Flash-Maya-S-IQ2_XXS-00001-of-00003.gguf \
  --host 127.0.0.1 --port 8080 --max-ctx 16384 --kv q8_0 --multi-gpu pipeline \
  --load-profile /path/to/expert-profile-mm3.bin --parallel 4 --slot-quantum 32
```

- **Thinking mode (`reasoning_effort`):** supports `"none"`/`"off"` (`"enable_thinking": false`), `"low"`, `"adaptive"` (default), and `"high"`/`"max"`. Streams `<mm:think>...</mm:think>` as `reasoning_content` and parses `<minimax:tool_call><invoke ...>` cleanly into OpenAI `tool_calls`.

---

## HTTP Endpoints, Multi-Slot Subagents & Prometheus Metrics

Both `/v1/*` and root OpenAI aliases (`/models`, `/chat/completions`, `/completions`, `/slots`) are served so OpenAI SDK clients work out of the box by pointing `base_url` at `http://127.0.0.1:8080/v1` (no API key required):

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | JSON readiness check (`status`, `model`, `max_context`) |
| `/v1/slots`, `/slots` | `GET` | Concurrent scheduler slot states (`active`, `paused`, `idle`, `empty`), cached token counts, host-RAM snapshot size, and queue depth |
| `/v1/models`, `/models` | `GET` | OpenAI model list (includes `context_length`) |
| `/v1/chat/completions`, `/chat/completions` | `POST` | OpenAI chat completions (`stream: false` and SSE `stream: true`) with prefix cache reuse, `reasoning_content`, OpenAI `tools`/`tool_choice`/`tool_calls`, `image_url`, and per-request `usage` + `timings` |
| `/v1/completions`, `/completions` | `POST` | Raw prompt completion (`stream: false` and SSE `stream: true`) |
| `/metrics` | `GET` | Prometheus text exposition format |

### Multi-Slot Concurrent Subagent Serving (`--parallel`, `--slot-quantum`)
- Default **`--parallel 4`** (range `1..=16`) and **`--slot-quantum 32`** run multiple concurrent subagents interleaved in fair round-robin quanta.
- **Zero VRAM penalty:** inactive/paused slots offload KV/recurrent state to host RAM (`AMPLLM_SLOT_CACHE_MB`, default `8192` MiB), keeping 100% of VRAM for hot MoE experts.
- **Zero single-stream overhead:** when a single request runs alone, it never yields and reuses resident VRAM directly with 0 D2H/H2D copies.
- **Prefix-aware LRU admission & disconnect cancellation:** multi-turn conversations reuse prefix state even after slot swaps or `<think>`-stripped Turn 2 histories (via secondary post-prefill checkpoints on recurrent/SSM models), and client disconnects abort generation cleanly on the next token.

Exported `/metrics` series:
- `ampllm_requests_total`, `ampllm_requests_cancelled_total`, `ampllm_tokens_prefilled_total`, `ampllm_tokens_generated_total`, `ampllm_prefix_reuse_tokens_total`
- `ampllm_slots_total`, `ampllm_slots_active`, `ampllm_queue_depth`, `ampllm_slot_swaps_total`
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

### 4. Hy4-Flash Throughput (`Hy4-Flash-Maya-L` `IQ4_XS`, 131K Context, **250 W**)

Configured with `max_ctx=131072`, `KV=Q8_0`, `split=23`, `VRAM experts 8636/20992 (41.1%)`, `[22196, 21484] MiB` across 2× RTX 3090:

| Workload | Prompt Tokens | Output Tokens | Prefill | Decode | Notes |
| --- | --- | --- | --- | --- | --- |
| `P4096` (+MTP) | 4,096 | 32 | **676.32 tok/s** | **26.65 tok/s** | `draft=19/27 (70.4%)` |
| `P2048` (+MTP) | 2,048 | 32 | **615.36 tok/s** | **25.53 tok/s** | `draft=19/28 (67.9%)` |
| Short WARM (+MTP) | 15 | 128 | **57.02 tok/s** | **30.32 tok/s** | `draft=81/106 (76.4%)` |
| Short COLD (+MTP) | 15 | 128 | **40.00 tok/s** | **26.81 tok/s** | Cold Engram + disk-backed CPU MoE |
| Short WARM (no-MTP) | 15 | 112 (EOS) | **56.02 tok/s** | **24.70 tok/s** | `--no-spec` / `AMPLLM_HY4_DRAFT=0` |
| Vision (`256×256` PNG/RGB, 4K ctx) | 72 image tokens | — | **78.5 ms warm** (`622.8 ms` cold) | — | Zero permanent VRAM; `cos_sim=0.999987` |

### 5. Qwen3.8-Flash-Next Throughput (`unsloth/Qwen3.8-Flash-Next-GGUF` `UD-Q4_K_XL`, 2× RTX 3090)

Profiled dual-RTX 3090 pipeline (`split=25`, `VRAM experts 12271/24576 (49.9%)`, `mtp_k=4`, CUDA Graphs + pinned cold experts + mmap PLE):

| Workload | Prompt Tokens | Output Tokens | Prefill | Decode | Parity / Notes |
| --- | --- | --- | --- | --- | --- |
| Short (`p=26`) (+MTP) | 26 | 71 | **218.6 tok/s** | **101.0 tok/s** | `32/32` parity, `draft_acc=22/28 (78.6%)` |
| `P4096` (+MTP) | 4,096 | 32 | **1,614.6 tok/s** | **82.7 tok/s** | Pinned host + 2-GPU chunk overlap |
| `P16384` (+MTP) | 16,384 | 32 | **2,350.5 tok/s** | **104.5 tok/s** | Full long-context throughput |
| Vision Encoder E2E | — | — | **Pass** | — | `cos_sim=0.999985` vs reference |

### 6. MiniMax M3.1 Flash Throughput (`MiniMax-M3.1-Flash-Maya-S` `IQ2_XXS`, `--kv q8_0` / `Int8HadV`, 2× RTX 3090)

Configured with `max_ctx=16384` across 2× RTX 3090 pipeline (`dspark` Markov speculative drafting + ViT Vision):

| Workload | Prompt Tokens | Prefill | Mean Decode | Notes |
| --- | --- | --- | --- | --- |
| `dspark=ON` Short topics (`0 / 1 / 2`) | — | — | **24.31 tok/s** (`25.95` / `18.46` / `28.52`) | `32/32` lossless `ON == OFF`, `acc=23/28 (82.1%)` |
| `dspark=ON` `P2048` / `P8192` | 2,048 / 8,192 | **177.0** / **194.8 tok/s** | — | BF16 KV `cos=0.999199`, INT8 KV `cos=0.998338` |
| `dspark=OFF` Short topics (`0 / 1 / 2`) | — | — | **16.48 tok/s** (`16.78` / `16.20` / `16.46`) | `--no-spec` |
| `dspark=OFF` `P2048` / `P8192` | 2,048 / 8,192 | **184.7** / **205.3 tok/s** | — | L03 determinism `cos=1.000000` |
| Vision (`280×280` RGB, `100×6144`) | 100 image tokens | **190.6 ms** | — | `cos=0.999994` vs reference |

---

## Runtime Tuning & Debug Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `AMPLLM_PARALLEL_SLOTS` | `4` | Default concurrent scheduler slot count (`1..=16`; overridden by `--parallel`) |
| `AMPLLM_SLOT_QUANTUM` | `32` | Tokens decoded per round-robin quantum when multiple requests contend (`--slot-quantum`) |
| `AMPLLM_SLOT_CACHE_MB` | `8192` | Max host RAM budget (MiB) for paused/idle sequence snapshots before LRU eviction |
| `AMPLLM_VERIFY_M` | `1` | GLM-5.3 speculative verification batch width (`1..4`) |
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
| `AMPLLM_QWEN4_MTP_K` | auto (`4`) | Override Qwen3.8 MTP speculative draft depth (`0` to disable) |
| `AMPLLM_QWEN4_ADAPT` | `1` | Set `0` to disable Qwen3.8 async GPU/CPU expert swapping during decode |
| `AMPLLM_QWEN4_PROFILE` | unset | Print Qwen3.8 per-stage timing breakdown |
| `AMPLLM_MM3_DSPARK_CONF` | `0.7` | MiniMax M3.1 Flash DSpark draft confidence threshold (`0.0..1.0`) |
| `AMPLLM_HY4_DRAFT` | `3` | Hy4-Flash MTP draft length (`0..3`; `0` disables speculative drafting) |
| `AMPLLM_HY4_DRAFT_FLOOR` | `0.55` | Minimum Hy4-Flash MTP token confidence floor before stopping early |
| `AMPLLM_HY4_KV_Q8` | auto | Override Hy4-Flash KV cache mode (`1` for `Q8_0`, `0` for `F16`) |
| `AMPLLM_PROFILE` | unset | Log per-request decode, MTP acceptance, hit-rate, and CPU timings |
| `AMPLLM_PREFILL_PROFILE` | unset | Log per-stage prefill timing breakdown (`1`/`true`) |
| `AMPLLM_PHASE_PROFILE` | unset | Collect per-layer CUDA event breakdowns |

---

## Workspace Architecture

- [`crates/gguf`](crates/gguf): Zero-copy mmap GGUF v2/v3 parser, SafeTensors reader, GGML block types & reference kernels, and config parsers (`Inkling2`, `Glm5`, `Hy4`, `Qwen4`, `Mm3`).
- [`crates/tokenizer`](crates/tokenizer): GGUF GPT-2 byte-level BPE tokenizer with Inkling-2, GLM-5, Hy4, Qwen3.8 (`qwen35`), and MiniMax (`minimax-m2`) pre-tokenization stages.
- [`crates/cuda-kernels`](crates/cuda-kernels): Standalone `nvcc` CUDA kernels (`sm_86` default for RTX 30-series) across Inkling-2, GLM-5.3, Hy4, Qwen3.8, and MiniMax M3.1 + ViT/SigLIP vision and safe Rust FFI wrappers.
- [`crates/cpu-kernels`](crates/cpu-kernels): Physical-core spin-barrier thread pool and AVX2 / AVX-512 VNNI kernels for offloaded MoE experts (`Q5_K`/`IQ4_XS`/`IQ2_XXS`/`IQ2_XS`/`IQ2_S`/`IQ3_XXS`/`Q8_0`, PLE/Engram lookup).
- [`crates/cache`](crates/cache): Mapped pinned host arenas, VRAM expert LRU/LFU cache planning, and `CacheProfile` / `STRP` v1 persistence.
- [`crates/engine`](crates/engine): Unified `AnyEngine` backends (`Inkling2Engine`, `Glm5Engine`, `Hy4Engine`, `Qwen4Engine`, `Mm3Engine`), `SequenceSnapshot` host-RAM state pooling ([`snapshot.rs`](crates/engine/src/snapshot.rs)), and speculative MTP/DSpark sampling.
- [`crates/server`](crates/server): `axum` OpenAI-compatible HTTP/SSE multimodal & tool-calling server, multi-slot round-robin `SlotScheduler` ([`scheduler.rs`](crates/server/src/scheduler.rs)), image preprocessing ([`vision.rs`](crates/server/src/vision.rs)), and Prometheus `/metrics`.
- [`crates/cli`](crates/cli): AmpLLM CLI (`ampllm` binary: `inspect`, `tokenize`, `verify`, `run`, `serve`).
- [`docs/INKLING2_SPEC.md`](docs/INKLING2_SPEC.md): Exact Inkling-2-Mini GGUF tensor and forward-pass specification.

---

## License

MIT License — see [LICENSE](LICENSE).
