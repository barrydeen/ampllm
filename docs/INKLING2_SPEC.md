# Inkling-2-Mini Exact GGUF & Forward-Pass Specification

Verified verbatim against `ggml-org/llama.cpp` (`src/models/inkling2.cpp`, `src/models/inkling2-loop.cpp`, `src/llama-graph.cpp`, `src/llama-kv-cache.cpp`, `src/llama-vocab.cpp`, `ggml/src/ggml-common.h`, `ggml/src/ggml-quants.c`) and the raw GGUF header of `Inkling-2-Mini-IQ4_NL.gguf`.

---

## 1. Model Metadata (`general.architecture = "inkling2_loop"`)

From `Inkling-2-Mini-IQ4_NL.gguf`:
- `inkling2_loop.block_count`: `24` (physical layers `0..23`)
- `inkling2_loop.loop.start`: `6`, `loop.end`: `18`, `loop.count`: `3`, `loop.gate_rank`: `64`, `loop.varying`: `false`
  - **Logical execution schedule (`48` logical slots total):**
    - Prelude: physical `0..5` -> logical slots `0..5` (`1x`)
    - Core Pass 0: physical `6..17` -> logical slots `6..17` (`1st pass`)
    - Coda: physical `18..23` -> logical slots `18..23` (`1x`)
    - Core Pass 1: physical `6..17` -> logical slots `24..35` (`2nd pass`)
    - Core Pass 2: physical `6..17` -> logical slots `36..47` (`3rd pass`)
    *(Note: each logical slot `0..47` owns its own KV cache and causal short-conv history ring, while sharing physical weights with `layer_phys(slot)`!)*
- `inkling2_loop.embedding_length`: `2304`
- `inkling2_loop.vocab_size`: `201024` (`vocab_size_unpadded`: `200058`)
- `inkling2_loop.attention.head_count`: `24` (all layers), `head_count_kv`: `8` (all layers), `key_length`: `128`, `value_length`: `128`
- `inkling2_loop.attention.layer_norm_rms_epsilon`: `1e-6`
- `inkling2_loop.attention.sliding_window`: `1024`
- `inkling2_loop.attention.sliding_window_pattern`: `true` for all layers except `5, 11, 17, 23` (every 6th layer is global full-attention; 20 SWA physical layers = 40 SWA logical slots, 4 global physical layers = 8 global logical slots).
- `inkling2_loop.attention.relative_length` (`d_rel`): `16`
- `inkling2_loop.attention.relative_extent`: `1024` (SWA layers use `sliding_window = 1024`, global layers use `relative_extent = 1024`)
- `inkling2_loop.attention.qk_clip`: `22.627417` (`16 * sqrt(2)`)
- `inkling2_loop.attention.relative_clip`: `8.0` (`2 * sqrt(16)`)
- `inkling2_loop.attention.log_scale_floor`: `8192`, `log_scale_alpha`: `0.1`
- `inkling2_loop.rope.dimension_count`: `0` (No RoPE!)
- `inkling2_loop.shortconv.l_cache`: `4` (history length `L - 1 = 3`)
- `inkling2_loop.attn_res.block_size`: `6`
- `inkling2_loop.expert_count`: `128`, `expert_used_count`: `8`, `expert_feed_forward_length`: `1152`, `expert_weights_scale`: `8.0`
- `inkling2_loop.activation.situ_beta`: `4.0`, `activation.situ_linear_beta`: `25.0`
- `inkling2_loop.final_logit_softcapping`: `30.0`
- Tokenizer: `tokenizer.ggml.model = "gpt2"`, `tokenizer.ggml.pre = "inkling2"` (GPT-4o regex pre-tokenizer, `clean_spaces = false`, no auto-BOS, `eos_token_id = 200006`).

---

## 2. Exact Tensor Shapes & Quantization (`Inkling-2-Mini-IQ4_NL.gguf`)

GGUF dimensions are listed innermost-contiguous first `[ne0, ne1, ne2]` (meaning linear matmul `y = W @ x` contracts `ne0` to produce `ne1`).

### Global Tensors
- `token_embd.weight`: `[2304, 201024]` (`Q8_0`)
- `token_embd_norm.weight`: `[2304]` (`F32`)
- `output_norm.weight`: `[2304]` (`F32`)
- `output_res_query.weight`: `[2304]` (`F32`)
- `output.weight`: `[2304, 201024]` (`Q8_0`)

### Per-Layer Tensors (`blk.{0..23}.*`)
- `attn_q.weight`: `[2304, 3072]` (`Q8_0`), `attn_k.weight`: `[2304, 1024]` (`Q8_0`), `attn_v.weight`: `[2304, 1024]` (`Q8_0`)
- `attn_gate.weight`: `[2304, 3072]` (`Q8_0`), `attn_output.weight`: `[3072, 2304]` (`Q8_0`)
- `attn_rel.weight`: `[2304, 384]` (`Q8_0`), `attn_rel_bias.weight`: `[16, 1024]` (`F32`)
- `attn_q_clip.weight`: `[128, 24]` (`F32`), `attn_rel_clip.weight`: `[16, 24]` (`F32`)
- `attn_sink_proj.weight`: `[2304, 24]` (`Q8_0`)
- `attn_k_conv.weight`: `[4, 1024]` (`F32`), `attn_v_conv.weight`: `[4, 1024]` (`F32`)
- `attn_post_conv.weight`: `[4, 2304]` (`F32`), `ffn_post_conv.weight`: `[4, 2304]` (`F32`)
- SWA layers (`is_swa == true`): `attn_norm.weight` `[2304]` (`F32`), `attn_res_query.weight` `[2304]` (or `[2304, 1]`, omitted on `blk.0`)
- Global layers (`5, 11, 17, 23`):
  - `attn_q_inp_norm.weight`, `attn_k_inp_norm.weight`, `attn_v_inp_norm.weight`: `[2304]` (`F32`)
  - `attn_k_conv_proj.weight`, `attn_v_conv_proj.weight`: `[2304, 32]` (`Q8_0`) (`32 = 4 * 8 kv_heads`)
  - `attn_res_query.weight`: `[2304, 3]` (`F32`) (separate depth-mix query for Q, K, V)
- Loop Core (`6..17`):
  - `loop_msg_query.weight`: `[2304, 4]` (`F32`) (for `[Q, K, V, FFN]`)
  - `loop_{q,k,v,ffn}_ctrl_down.weight`: `[4608, 64]` (`F32`)
  - `loop_{q,k,v,ffn}_ctrl_gain.weight`: `[64, 2304]` (`F32`)
  - `loop_{q,k,v,ffn}_ctrl_forget.weight`: `[64, 2304]` (`F32`)
- MoE (`0..23`):
  - `ffn_res_query.weight`: `[2304]` (`F32`), `ffn_norm.weight`: `[2304]` (`F32`)
  - `ffn_gate_inp.weight`: `[2304, 128]` (`F32`), `exp_probs_b.bias`: `[128]` (`F32`)
  - `ffn_gate_up_exps.weight`: `[2304, 2304, 128]` (`IQ4_NL` for `0..19`, `Q8_0` for `20..23`; rows `0..1152` are `gate`, rows `1152..2304` are `up`)
  - `ffn_down_exps.weight`: `[1152, 2304, 128]` (`IQ4_NL` for `0..19`, `Q8_0` for `20..23`)

---

## 3. Exact Mathematical Forward Pass

### A. Inputs & Log-Scaling
1. `cur = RMSNorm(token_embd[tok], token_embd_norm, eps=1e-6)`
2. `tau(pos) = 1.0 + 0.1 * log(max(1.0, (pos + 1.0) / 8192.0))`

### B. Depth-Wise Block Attention Residuals (`mix(blocks, cur, queries)`)
Given completed block outputs `blocks = [b_0, b_1, ...]` and current state `cur`:
1. Stack `states = [b_0, b_1, ..., cur]` of shape `[2304, M]` (`M = blocks.len() + 1`).
2. `states_norm = RMSNorm(states, eps=1e-6)` across dim `2304` (no weight).
3. For each query vector `q` in `queries`:
   - `scores[j] = dot(states_norm[j], q) / sqrt(2304.0)` (`1/48.0`)
   - `alpha = softmax(scores)`
   - `out = sum_j alpha[j] * states_norm[j]`

### C. Looped Execution Schedule
- Run Prelude `il = 0..6` -> save `prelude = cur`, `prelude_blocks = blocks`.
- For `pass = 0..3`:
  - If `pass > 0`:
    - `bank = blocks[prelude_blocks.len() + 1 ..]` (the second core block `b_2`, i.e. output of layer 11, excluding the first core block `blocks[prelude_blocks.len()]` which equals `prelude`!)
    - `bank_out = cur` (output of layer 17 from previous pass)
    - `cur = prelude`; `blocks = prelude_blocks`
  - For `ip = 6..18`:
    - `slot = (pass == 0) ? ip : 24 + (pass - 1)*12 + (ip - 6)`
    - If `pass > 0`: compute `messages[0..4] = mix(bank, bank_out, loop_msg_query)` (`[Q, K, V, FFN]`).
    - Run layer `(slot, phys=ip, cur, blocks, messages, loop)`.
- Run Coda `il = 18..24`.
- Output:
  - `cur = RMSNorm(mix(blocks, cur, output_res_query), output_norm, 1e-6)`
  - `logits = output.weight @ cur` (F32 accumulation on `output`)
  - `logits = 30.0 * tanh(logits / 30.0)`, truncated to `200058` unpadded tokens.

### D. Single Layer Execution `(slot, ip, cur, blocks, messages, loop)`
1. `inpSA = cur`
2. **Attention input mixing & control:**
   - Start with `inputs[0..3] = [cur, cur, cur]`.
   - If `attn_res_query` exists (`ip > 0`):
     - `mixed = mix(blocks, cur, attn_res_query)` (`1` vector for SWA, `3` vectors for global).
     - SWA: `inputs[0..3] = [mixed[0], mixed[0], mixed[0]]`.
     - Global: `inputs[0..3] = [mixed[0], mixed[1], mixed[2]]`.
   - If `pass > 0` (`messages` present), for `c in 0..3`:
     - `gate = ctrl_down[c] @ concat(RMSNorm(inputs[c], 1e-6), RMSNorm(messages[c], 1e-6))` (`[64]`)
     - `gain = tanh(ctrl_gain[c] @ gate)`, `forget = tanh(ctrl_forget[c] @ gate)`
     - `inputs[c] = (1 - forget) * inputs[c] + gain * messages[c]`
   - Normalize `inputs`:
     - SWA (`pass == 0` or Prelude/Coda): `inputs[0] = RMSNorm(inputs[0], attn_norm)`; `inputs[1] = inputs[2] = inputs[0]`.
     - SWA (`pass > 0`): each `inputs[j]` independently normalized with `attn_norm`.
     - Global: `inputs[0..3]` normalized with `attn_{q,k,v}_inp_norm`.
3. **Projections & Causal Short Convolutions (`n_conv = 4`, history = last `3` tokens in `slot`'s conv cache):**
   - `Qcur = attn_q @ inputs[0]` (`[24, 128]`)
   - `Kcur = convolve(attn_k @ inputs[1], attn_k_conv, is_swa ? None : attn_k_conv_proj @ inputs[1])`
   - `Vcur = convolve(attn_v @ inputs[2], attn_v_conv, is_swa ? None : attn_v_conv_proj @ inputs[2])`
   - `Rcur = attn_rel @ inputs[0]` (`[24, 16]`)
   - **Exact `convolve(cur, static_w, dyn_proj)`:**
     - Window `x = [h[t-3], h[t-2], h[t-1], cur[t]]` (update conv cache with `cur[t]`).
     - Static conv: `out[c] = sum_{k=0..3} x[k, c] * static_w[k, c]`. (`static_w[0]` is oldest `t-3`, `static_w[3]` is `cur[t]`).
     - Dynamic taps (global only, `dyn_proj: [4, n_heads]`): `for lag in 0..4: out[head, sub] += x[3 - lag, head, sub] * dyn_proj[lag, head]`. (Notice `lag=0` multiplies `cur[t]`!)
     - Output: `(cur + out) / sqrt((has_dynamic ? 2 : 1) * 4 + 1)` (i.e. `/ sqrt(5)` for static, `/ 3.0` for dynamic).
4. **Clip, Value Norm & Log-Scaling:**
   - `clip(x, scale_opt, thresh)`: if `scale_opt`, `x *= scale`; per-head norm `n = sqrt(sum(x^2)) + 1e-6`; `x *= min(1.0, thresh / n)`.
   - `Qcur = clip(Qcur, attn_q_clip, 22.627417)`
   - `Kcur = clip(Kcur, None, 22.627417)`
   - `Vcur = RMSNorm_no_gamma(Vcur, eps=1e-6)` per KV head
   - `Rcur = clip(Rcur, attn_rel_clip, 8.0)`
   - If `!is_swa` (global layers `5, 11, 17, 23`): `Qcur *= tau`, `Rcur *= tau`.
5. **Attention with Relative Bias & Sink (`store Kcur, Vcur into slot's KV cache`):**
   - `sinks = attn_sink_proj @ inputs[0]` (`[24]`)
   - For each query head `h` and visible KV position `j` (causal, and `0 <= pos_q - pos_k < 1024` for SWA):
     - `dist = pos_q - pos_k`; `bias = (0 <= dist < 1024) ? dot(Rcur[h], attn_rel_bias[:, dist]) : 0.0`.
     - Unscaled pre-softmax score: `score_raw = dot(Qcur[h], K[kv_h, j]) + bias - sinks[h]`.
     - Scaled logit: `logit = score_raw / sqrt(128.0)`.
   - Softmax with zero sink entry:
     - `m = max(0.0, max_j logit[j])`
     - `denom = exp(0.0 - m) + sum_j exp(logit[j] - m)`
     - `attn_out[h] = sum_j (exp(logit[j] - m) / denom) * V[kv_h, j]`
6. **Output Gate & Post-Attn Conv & Block Boundary:**
   - `cur = attn_out * sigmoid(attn_gate @ inputs[0])`
   - `cur = convolve(attn_output @ cur, attn_post_conv, None)` (at conv channel offset `2 * 1024`)
   - **Block boundary (`ip % 6 == 0`):** `blocks.push(inpSA)` (and `cur` stays `attn_out`, NO residual add!).
   - **Inside block (`ip % 6 != 0`):** `cur = inpSA + cur`.
7. **MoE Block:**
   - `ffn_inp = cur`
   - `cur = mix(blocks, cur, ffn_res_query)`
   - If `pass > 0`: `cur = control(cur, messages[3], loop.ffn_ctrl)`
   - `cur = RMSNorm(cur, ffn_norm, 1e-6)`
   - Router:
     - `probs = sigmoid(ffn_gate_inp @ cur)` (F32, `[128]`)
     - `sel_scores = probs + exp_probs_b.bias`
     - `top8_ids = topk(sel_scores, 8)`
     - `w[k] = probs[top8_ids[k]]`; `w /= (sum(w) + 1e-20)`; `w *= 8.0`
   - Experts (`e in top8_ids`):
     - `gate = W_gate_e @ cur` (rows `0..1152`), `up = W_up_e @ cur` (rows `1152..2304`)
     - **SiTU activation:**
       `act = (4.0 * tanh(gate / 4.0)) * sigmoid(gate) * (25.0 * tanh(up / 25.0))`
     - `exp_out = W_down_e @ act`
   - `cur = sum_{k=0..7} w[k] * exp_out[k]`
   - `cur = convolve(cur, ffn_post_conv, None)` (at conv channel offset `2 * 1024 + 2304`)
   - `cur = ffn_inp + cur`
