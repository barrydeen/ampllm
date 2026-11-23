#include "common/sm86_common.cuh"

// Store K and V into KV cache (stored as __half to halve VRAM footprint & match llama.cpp F16 default).
// k_cur, v_cur: [M, n_head_kv, head_dim] F32
// k_cache, v_cache: [max_slots, n_head_kv, head_dim] uint16_t (__half)
// For SWA layers, slot = pos % swa_window; for global layers, slot = pos.
__global__ void kv_store_f16_kernel(
    const float* __restrict__ k_cur,
    const float* __restrict__ v_cur,
    uint16_t* __restrict__ k_cache,
    uint16_t* __restrict__ v_cache,
    const int32_t* __restrict__ pos,
    int n_head_kv,
    int head_dim,
    int M,
    int ring_cap // swa_window if SWA, else max_seq
) {
    int total = M * n_head_kv * head_dim;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int d = idx % head_dim;
    int rem = idx / head_dim;
    int kh = rem % n_head_kv;
    int m = rem / n_head_kv;

    int slot = pos[m] % ring_cap;
    size_t dst_idx = ((size_t)slot * n_head_kv + kh) * head_dim + d;
    k_cache[dst_idx] = f32_to_fp16_bits(k_cur[idx]);
    v_cache[dst_idx] = f32_to_fp16_bits(v_cur[idx]);
}

// Store K and V into Q8_0 KV cache (Phase 4 option, 1 block_q8_0 per 32 dims along head_dim).
__global__ void kv_store_q8_0_kernel(
    const float* __restrict__ k_cur,
    const float* __restrict__ v_cur,
    uint8_t* __restrict__ k_cache,
    uint8_t* __restrict__ v_cache,
    const int32_t* __restrict__ pos,
    int n_head_kv,
    int head_dim,
    int M,
    int ring_cap
) {
    int nb_head = head_dim / 32;
    int blk = blockIdx.x;
    int kh = blockIdx.y;
    int m = blockIdx.z;
    if (blk >= nb_head || kh >= n_head_kv || m >= M) return;
    int lane = threadIdx.x;

    int slot = pos[m] % ring_cap;
    size_t src_off = (((size_t)m * n_head_kv + kh) * head_dim) + blk * 32 + lane;
    size_t dst_blk = (((size_t)slot * n_head_kv + kh) * nb_head) + blk;

    // Quantize K block
    {
        float val = k_cur[src_off];
        float amax = warp_reduce_max(fabsf(val));
        float d = amax / 127.0f;
        int q = (d > 0.0f) ? __float2int_rn(val / d) : 0;
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        block_q8_0* kb = reinterpret_cast<block_q8_0*>(k_cache) + dst_blk;
        kb->qs[lane] = (int8_t)q;
        if (lane == 0) kb->d = f32_to_fp16_bits(d);
    }
    // Quantize V block
    {
        float val = v_cur[src_off];
        float amax = warp_reduce_max(fabsf(val));
        float d = amax / 127.0f;
        int q = (d > 0.0f) ? __float2int_rn(val / d) : 0;
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        block_q8_0* vb = reinterpret_cast<block_q8_0*>(v_cache) + dst_blk;
        vb->qs[lane] = (int8_t)q;
        if (lane == 0) vb->d = f32_to_fp16_bits(d);
    }
}

// Precompute relative bias table per token and head:
// rel_bias_table[m, h, dist] = dot(Rcur[m, h, :], attn_rel_bias[dist, :]) - sinks[m, h]
// Rcur: [M, n_head, d_rel=16]
// attn_rel_bias: [rel_extent=1024, d_rel=16]
// sinks: [M, n_head]
// out: [M, n_head, rel_extent + 1] (last entry rel_extent holds -sinks[m, h])
__global__ void precompute_rel_bias_kernel(
    const float* __restrict__ Rcur,
    const float* __restrict__ attn_rel_bias,
    const float* __restrict__ sinks,
    float* __restrict__ rel_table,
    int n_head,
    int d_rel,
    int rel_extent,
    int M
) {
    int m = blockIdx.z;
    int h = blockIdx.y;
    int dist = blockIdx.x * blockDim.x + threadIdx.x;
    if (dist > rel_extent) return;

    const float* r = Rcur + ((size_t)m * n_head + h) * d_rel;
    float sink_val = sinks[(size_t)m * n_head + h];
    float dot = 0.0f;
    if (dist < rel_extent) {
        const float* b = attn_rel_bias + (size_t)dist * d_rel;
        for (int i = 0; i < d_rel; ++i) {
            dot += r[i] * b[i];
        }
    }
    rel_table[((size_t)m * n_head + h) * (rel_extent + 1) + dist] = dot - sink_val;
}

// Fast M=1 decode attention with inline warp-shfl relative bias dot(Rcur, attn_rel_bias) and optional fused sigmoid(attn_gate)!
// Grid: (n_splits, n_head, 1), Block: (32, 4) = 128 threads
__global__ void flash_attn_rel_decode_m1_kernel(
    const float* __restrict__ q_cur,        // [n_head, 128]
    const float* __restrict__ r_cur,        // [n_head, d_rel=16]
    const float* __restrict__ attn_rel_bias,// [rel_extent, d_rel=16]
    const float* __restrict__ sinks,        // [n_head]
    const float* __restrict__ attn_gate,    // nullable [n_head, 128]
    const void* __restrict__ k_cache,
    const void* __restrict__ v_cache,
    const int32_t* __restrict__ q_pos,      // [1]
    float* __restrict__ split_out,          // [n_head, n_splits, 128] or [n_head, 128] when n_splits==1
    float* __restrict__ split_ml,           // [n_head, n_splits, 2]
    int n_head,
    int n_head_kv,
    int head_dim, // 128
    int d_rel,    // 16
    int rel_extent,
    int swa_window,
    int ring_cap,
    int kv_is_q8,
    int n_splits
) {
    int split_id = blockIdx.x;
    int h = blockIdx.y;
    int lane = threadIdx.x; // 0..31
    int wid  = threadIdx.y; // 0..3
    int kv_h = h / (n_head / n_head_kv);
    int pos_q = q_pos[0];

    int k_start = (swa_window > 0 && pos_q + 1 > swa_window) ? (pos_q + 1 - swa_window) : 0;
    int k_end   = pos_q + 1;
    int total_k = k_end - k_start;

    int per_split = (total_k + n_splits - 1) / n_splits;
    int s_begin = k_start + split_id * per_split;
    int s_end   = s_begin + per_split;
    if (s_end > k_end) s_end = k_end;

    const float* q_ptr = q_cur + (size_t)h * head_dim;
    float q0 = q_ptr[lane + 0];
    float q1 = q_ptr[lane + 32];
    float q2 = q_ptr[lane + 64];
    float q3 = q_ptr[lane + 96];

    float r_lane = (lane < d_rel) ? r_cur[(size_t)h * d_rel + lane] : 0.0f;
    float sink_val = sinks[h];
    float inv_sqrt_d = rsqrtf((float)head_dim);

    float m_max = (split_id == 0 && wid == 0) ? 0.0f : -1e30f;
    float l_sum = (split_id == 0 && wid == 0) ? 1.0f : 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    const uint16_t* k_f16 = reinterpret_cast<const uint16_t*>(k_cache);
    const uint16_t* v_f16 = reinterpret_cast<const uint16_t*>(v_cache);
    const block_q8_0* k_q8 = reinterpret_cast<const block_q8_0*>(k_cache);
    const block_q8_0* v_q8 = reinterpret_cast<const block_q8_0*>(v_cache);
    int nb_head = head_dim / 32;

    for (int kp = s_begin + wid; kp < s_end; kp += 4) {
        int slot = kp % ring_cap;
        float k0, k1, k2, k3, v0, v1, v2, v3;

        if (kv_is_q8 == 0) {
            size_t base = ((size_t)slot * n_head_kv + kv_h) * head_dim;
            k0 = fp16_bits_to_f32(k_f16[base + lane + 0]);
            k1 = fp16_bits_to_f32(k_f16[base + lane + 32]);
            k2 = fp16_bits_to_f32(k_f16[base + lane + 64]);
            k3 = fp16_bits_to_f32(k_f16[base + lane + 96]);
            v0 = fp16_bits_to_f32(v_f16[base + lane + 0]);
            v1 = fp16_bits_to_f32(v_f16[base + lane + 32]);
            v2 = fp16_bits_to_f32(v_f16[base + lane + 64]);
            v3 = fp16_bits_to_f32(v_f16[base + lane + 96]);
        } else if (kv_is_q8 == 2) {
            const uint8_t* k_f8 = reinterpret_cast<const uint8_t*>(k_cache);
            const uint8_t* v_f8 = reinterpret_cast<const uint8_t*>(v_cache);
            size_t base = ((size_t)slot * n_head_kv + kv_h) * head_dim;
            k0 = e4m3fn_to_f32(k_f8[base + lane + 0]) * (1.0f / 16.0f);
            k1 = e4m3fn_to_f32(k_f8[base + lane + 32]) * (1.0f / 16.0f);
            k2 = e4m3fn_to_f32(k_f8[base + lane + 64]) * (1.0f / 16.0f);
            k3 = e4m3fn_to_f32(k_f8[base + lane + 96]) * (1.0f / 16.0f);
            v0 = e4m3fn_to_f32(v_f8[base + lane + 0]) * (1.0f / 32.0f);
            v1 = e4m3fn_to_f32(v_f8[base + lane + 32]) * (1.0f / 32.0f);
            v2 = e4m3fn_to_f32(v_f8[base + lane + 64]) * (1.0f / 32.0f);
            v3 = e4m3fn_to_f32(v_f8[base + lane + 96]) * (1.0f / 32.0f);
        } else {
            size_t b_base = ((size_t)slot * n_head_kv + kv_h) * nb_head;
            const block_q8_0* kb0 = k_q8 + b_base + 0;
            const block_q8_0* kb1 = k_q8 + b_base + 1;
            const block_q8_0* kb2 = k_q8 + b_base + 2;
            const block_q8_0* kb3 = k_q8 + b_base + 3;
            k0 = fp16_bits_to_f32(load_u16_unaligned(&kb0->d)) * (float)kb0->qs[lane];
            k1 = fp16_bits_to_f32(load_u16_unaligned(&kb1->d)) * (float)kb1->qs[lane];
            k2 = fp16_bits_to_f32(load_u16_unaligned(&kb2->d)) * (float)kb2->qs[lane];
            k3 = fp16_bits_to_f32(load_u16_unaligned(&kb3->d)) * (float)kb3->qs[lane];
            const block_q8_0* vb0 = v_q8 + b_base + 0;
            const block_q8_0* vb1 = v_q8 + b_base + 1;
            const block_q8_0* vb2 = v_q8 + b_base + 2;
            const block_q8_0* vb3 = v_q8 + b_base + 3;
            v0 = fp16_bits_to_f32(load_u16_unaligned(&vb0->d)) * (float)vb0->qs[lane];
            v1 = fp16_bits_to_f32(load_u16_unaligned(&vb1->d)) * (float)vb1->qs[lane];
            v2 = fp16_bits_to_f32(load_u16_unaligned(&vb2->d)) * (float)vb2->qs[lane];
            v3 = fp16_bits_to_f32(load_u16_unaligned(&vb3->d)) * (float)vb3->qs[lane];
        }

        int dist = pos_q - kp;
        float b_lane = (dist >= 0 && dist < rel_extent && lane < d_rel)
            ? (r_lane * __ldg(attn_rel_bias + (size_t)dist * d_rel + lane))
            : 0.0f;
        float dot = (q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3) * inv_sqrt_d + b_lane;
        float score = warp_reduce_sum(dot) - sink_val;

        float m_new = fmaxf(m_max, score);
        float alpha = (m_max <= -1e20f) ? 0.0f : expf(m_max - m_new);
        float w = expf(score - m_new);
        l_sum = l_sum * alpha + w;
        a0 = a0 * alpha + w * v0;
        a1 = a1 * alpha + w * v1;
        a2 = a2 * alpha + w * v2;
        a3 = a3 * alpha + w * v3;
        m_max = m_new;
    }

    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_a[4][128];

    if (lane == 0) {
        s_m[wid] = m_max;
        s_l[wid] = l_sum;
    }
    s_a[wid][lane + 0]  = a0;
    s_a[wid][lane + 32] = a1;
    s_a[wid][lane + 64] = a2;
    s_a[wid][lane + 96] = a3;
    __syncthreads();

    if (wid == 0) {
        float mg = -1e30f;
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) mg = fmaxf(mg, s_m[w]);
        }
        float lg = 0.0f;
        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) {
                float sc = expf(s_m[w] - mg);
                lg += s_l[w] * sc;
                out0 += s_a[w][lane + 0]  * sc;
                out1 += s_a[w][lane + 32] * sc;
                out2 += s_a[w][lane + 64] * sc;
                out3 += s_a[w][lane + 96] * sc;
            }
        }
        if (n_splits == 1) {
            float inv_l = 1.0f / lg;
            float* dst = split_out + (size_t)h * head_dim;
            float v0 = out0 * inv_l;
            float v1 = out1 * inv_l;
            float v2 = out2 * inv_l;
            float v3 = out3 * inv_l;
            if (attn_gate != nullptr) {
                const float* g_ptr = attn_gate + (size_t)h * head_dim;
                v0 *= 1.0f / (1.0f + expf(-g_ptr[lane + 0]));
                v1 *= 1.0f / (1.0f + expf(-g_ptr[lane + 32]));
                v2 *= 1.0f / (1.0f + expf(-g_ptr[lane + 64]));
                v3 *= 1.0f / (1.0f + expf(-g_ptr[lane + 96]));
            }
            dst[lane + 0]  = v0;
            dst[lane + 32] = v1;
            dst[lane + 64] = v2;
            dst[lane + 96] = v3;
        } else {
            size_t s_off = (size_t)h * n_splits + split_id;
            float* dst = split_out + s_off * head_dim;
            dst[lane + 0]  = out0;
            dst[lane + 32] = out1;
            dst[lane + 64] = out2;
            dst[lane + 96] = out3;
            if (lane == 0) {
                split_ml[s_off * 2 + 0] = mg;
                split_ml[s_off * 2 + 1] = lg;
            }
        }
    }
}

__global__ void flash_attn_rel_reduce_gate_kernel(
    const float* __restrict__ split_out,
    const float* __restrict__ split_ml,
    const float* __restrict__ attn_gate,
    float* __restrict__ out,
    int n_head,
    int head_dim,
    int n_splits
) {
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    size_t base = (size_t)h * n_splits;
    float mg = -1e30f;
    for (int s = 0; s < n_splits; ++s) {
        float l = split_ml[(base + s) * 2 + 1];
        if (l > 0.0f) mg = fmaxf(mg, split_ml[(base + s) * 2 + 0]);
    }
    float lg = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        float ms = split_ml[(base + s) * 2 + 0];
        float ls = split_ml[(base + s) * 2 + 1];
        if (ls > 0.0f) {
            float sc = expf(ms - mg);
            lg += ls * sc;
            acc += split_out[(base + s) * head_dim + d] * sc;
        }
    }
    float v = acc / lg;
    if (attn_gate != nullptr) {
        float g = attn_gate[(size_t)h * head_dim + d];
        v *= 1.0f / (1.0f + expf(-g));
    }
    out[(size_t)h * head_dim + d] = v;
}

// High-parallelism Split-KV FlashDecoding + FlashPrefill for Inkling-2-Mini (GQA 24:8, head_dim=128, no RoPE, rel_table + zero sink).
// Supports both F16 (kv_is_q8 == 0) and Q8_0 (kv_is_q8 == 1) KV caches!
// Grid: (n_splits, n_head, M), Block: (128, 1, 1) -> 1 thread per head dimension (head_dim=128)!
// Wait: 128 threads/block where lane-groups of 4 or warp-shared reductions handle dot(Q, K)!
// Even cleaner: Block = (32, 4) = 128 threads = 4 warps, each warp processes a key step j, and each thread holds 4 dims of head_dim=128 (4 * 32 = 128)!
// Wait: holding 4 dims/thread in 32 threads (1 warp) means `dot(Q, K)` is just 4 FMAs + 1 warp_reduce_sum (5 shfls)!
// Let's make Block = (32, 4): 4 warps cooperate over the KV split range! Each warp accumulates its own (m, l, acc[4]), then warps 0..3 combine via shared memory!
__global__ void flash_attn_rel_split_kernel(
    const float* __restrict__ q_cur,        // [M, n_head, 128]
    const void* __restrict__ k_cache,       // [ring_cap, n_head_kv, ...]
    const void* __restrict__ v_cache,
    const float* __restrict__ rel_table,    // [M, n_head, rel_extent + 1]
    const int32_t* __restrict__ q_pos,      // [M]
    float* __restrict__ split_out,          // [M, n_head, n_splits, 128]
    float* __restrict__ split_ml,           // [M, n_head, n_splits, 2] (m, l)
    int n_head,
    int n_head_kv,
    int head_dim, // 128
    int rel_extent,
    int swa_window, // 0 if global, else 1024
    int ring_cap,
    int kv_is_q8,
    int n_splits,
    int M
) {
    int split_id = blockIdx.x;
    int h = blockIdx.y;
    int m = blockIdx.z;

    int lane = threadIdx.x; // 0..31
    int wid  = threadIdx.y; // 0..3
    int kv_h = h / (n_head / n_head_kv);
    int pos_q = q_pos[m];

    int k_start = (swa_window > 0 && pos_q + 1 > swa_window) ? (pos_q + 1 - swa_window) : 0;
    int k_end   = pos_q + 1;
    int total_k = k_end - k_start;

    int per_split = (total_k + n_splits - 1) / n_splits;
    int s_begin = k_start + split_id * per_split;
    int s_end   = s_begin + per_split;
    if (s_end > k_end) s_end = k_end;

    // Each thread holds 4 elements of Q[h] (head_dim=128 = 32 lanes * 4)
    const float* q_ptr = q_cur + ((size_t)m * n_head + h) * head_dim;
    float q0 = q_ptr[lane + 0];
    float q1 = q_ptr[lane + 32];
    float q2 = q_ptr[lane + 64];
    float q3 = q_ptr[lane + 96];

    const float* rtab = rel_table + ((size_t)m * n_head + h) * (rel_extent + 1);
    float inv_sqrt_d = rsqrtf((float)head_dim);

    // Only split 0, warp 0 starts with the zero-logit attention sink (m=0, l=1)!
    float m_max = (split_id == 0 && wid == 0) ? 0.0f : -1e30f;
    float l_sum = (split_id == 0 && wid == 0) ? 1.0f : 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    const uint16_t* k_f16 = reinterpret_cast<const uint16_t*>(k_cache);
    const uint16_t* v_f16 = reinterpret_cast<const uint16_t*>(v_cache);
    const block_q8_0* k_q8 = reinterpret_cast<const block_q8_0*>(k_cache);
    const block_q8_0* v_q8 = reinterpret_cast<const block_q8_0*>(v_cache);
    int nb_head = head_dim / 32; // 4

    for (int kp = s_begin + wid; kp < s_end; kp += 4) {
        int slot = kp % ring_cap;
        float k0, k1, k2, k3, v0, v1, v2, v3;

        if (kv_is_q8 == 0) {
            size_t base = ((size_t)slot * n_head_kv + kv_h) * head_dim;
            k0 = fp16_bits_to_f32(k_f16[base + lane + 0]);
            k1 = fp16_bits_to_f32(k_f16[base + lane + 32]);
            k2 = fp16_bits_to_f32(k_f16[base + lane + 64]);
            k3 = fp16_bits_to_f32(k_f16[base + lane + 96]);
            v0 = fp16_bits_to_f32(v_f16[base + lane + 0]);
            v1 = fp16_bits_to_f32(v_f16[base + lane + 32]);
            v2 = fp16_bits_to_f32(v_f16[base + lane + 64]);
            v3 = fp16_bits_to_f32(v_f16[base + lane + 96]);
        } else if (kv_is_q8 == 2) {
            const uint8_t* k_f8 = reinterpret_cast<const uint8_t*>(k_cache);
            const uint8_t* v_f8 = reinterpret_cast<const uint8_t*>(v_cache);
            size_t base = ((size_t)slot * n_head_kv + kv_h) * head_dim;
            k0 = e4m3fn_to_f32(k_f8[base + lane + 0]) * (1.0f / 16.0f);
            k1 = e4m3fn_to_f32(k_f8[base + lane + 32]) * (1.0f / 16.0f);
            k2 = e4m3fn_to_f32(k_f8[base + lane + 64]) * (1.0f / 16.0f);
            k3 = e4m3fn_to_f32(k_f8[base + lane + 96]) * (1.0f / 16.0f);
            v0 = e4m3fn_to_f32(v_f8[base + lane + 0]) * (1.0f / 32.0f);
            v1 = e4m3fn_to_f32(v_f8[base + lane + 32]) * (1.0f / 32.0f);
            v2 = e4m3fn_to_f32(v_f8[base + lane + 64]) * (1.0f / 32.0f);
            v3 = e4m3fn_to_f32(v_f8[base + lane + 96]) * (1.0f / 32.0f);
        } else {
            size_t b_base = ((size_t)slot * n_head_kv + kv_h) * nb_head;
            const block_q8_0* kb0 = k_q8 + b_base + 0;
            const block_q8_0* kb1 = k_q8 + b_base + 1;
            const block_q8_0* kb2 = k_q8 + b_base + 2;
            const block_q8_0* kb3 = k_q8 + b_base + 3;
            k0 = fp16_bits_to_f32(load_u16_unaligned(&kb0->d)) * (float)kb0->qs[lane];
            k1 = fp16_bits_to_f32(load_u16_unaligned(&kb1->d)) * (float)kb1->qs[lane];
            k2 = fp16_bits_to_f32(load_u16_unaligned(&kb2->d)) * (float)kb2->qs[lane];
            k3 = fp16_bits_to_f32(load_u16_unaligned(&kb3->d)) * (float)kb3->qs[lane];
            const block_q8_0* vb0 = v_q8 + b_base + 0;
            const block_q8_0* vb1 = v_q8 + b_base + 1;
            const block_q8_0* vb2 = v_q8 + b_base + 2;
            const block_q8_0* vb3 = v_q8 + b_base + 3;
            v0 = fp16_bits_to_f32(load_u16_unaligned(&vb0->d)) * (float)vb0->qs[lane];
            v1 = fp16_bits_to_f32(load_u16_unaligned(&vb1->d)) * (float)vb1->qs[lane];
            v2 = fp16_bits_to_f32(load_u16_unaligned(&vb2->d)) * (float)vb2->qs[lane];
            v3 = fp16_bits_to_f32(load_u16_unaligned(&vb3->d)) * (float)vb3->qs[lane];
        }

        float dot = q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3;
        dot = warp_reduce_sum(dot);

        int dist = pos_q - kp;
        int ridx = (dist >= 0 && dist < rel_extent) ? dist : rel_extent;
        float score = dot * inv_sqrt_d + rtab[ridx];

        float m_new = fmaxf(m_max, score);
        float alpha = (m_max <= -1e20f) ? 0.0f : expf(m_max - m_new);
        float w = expf(score - m_new);
        l_sum = l_sum * alpha + w;
        a0 = a0 * alpha + w * v0;
        a1 = a1 * alpha + w * v1;
        a2 = a2 * alpha + w * v2;
        a3 = a3 * alpha + w * v3;
        m_max = m_new;
    }

    // Combine across the 4 warps in this block using shared memory
    __shared__ float s_m[4];
    __shared__ float s_l[4];
    __shared__ float s_a[4][128];

    if (lane == 0) {
        s_m[wid] = m_max;
        s_l[wid] = l_sum;
    }
    s_a[wid][lane + 0]  = a0;
    s_a[wid][lane + 32] = a1;
    s_a[wid][lane + 64] = a2;
    s_a[wid][lane + 96] = a3;
    __syncthreads();

    if (wid == 0) {
        float mg = -1e30f;
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) mg = fmaxf(mg, s_m[w]);
        }
        float lg = 0.0f;
        float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f, out3 = 0.0f;
        for (int w = 0; w < 4; ++w) {
            if (s_l[w] > 0.0f) {
                float sc = expf(s_m[w] - mg);
                lg += s_l[w] * sc;
                out0 += s_a[w][lane + 0]  * sc;
                out1 += s_a[w][lane + 32] * sc;
                out2 += s_a[w][lane + 64] * sc;
                out3 += s_a[w][lane + 96] * sc;
            }
        }
        if (n_splits == 1) {
            float inv_l = 1.0f / lg;
            float* dst = split_out + ((size_t)m * n_head + h) * head_dim;
            dst[lane + 0]  = out0 * inv_l;
            dst[lane + 32] = out1 * inv_l;
            dst[lane + 64] = out2 * inv_l;
            dst[lane + 96] = out3 * inv_l;
        } else {
            size_t s_off = (((size_t)m * n_head + h) * n_splits + split_id);
            float* dst = split_out + s_off * head_dim;
            dst[lane + 0]  = out0;
            dst[lane + 32] = out1;
            dst[lane + 64] = out2;
            dst[lane + 96] = out3;
            if (lane == 0) {
                split_ml[s_off * 2 + 0] = mg;
                split_ml[s_off * 2 + 1] = lg;
            }
        }
    }
}

// Combine split-KV partial outputs [M, n_head, n_splits, 128] -> [M, n_head, 128]
__global__ void flash_attn_rel_reduce_kernel(
    const float* __restrict__ split_out,
    const float* __restrict__ split_ml,
    float* __restrict__ out,
    int n_head,
    int head_dim,
    int n_splits,
    int M
) {
    int h = blockIdx.x;
    int m = blockIdx.y;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    size_t base = ((size_t)m * n_head + h) * n_splits;
    float mg = -1e30f;
    for (int s = 0; s < n_splits; ++s) {
        float l = split_ml[(base + s) * 2 + 1];
        if (l > 0.0f) mg = fmaxf(mg, split_ml[(base + s) * 2 + 0]);
    }
    float lg = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
        float ms = split_ml[(base + s) * 2 + 0];
        float ls = split_ml[(base + s) * 2 + 1];
        if (ls > 0.0f) {
            float sc = expf(ms - mg);
            lg += ls * sc;
            acc += split_out[(base + s) * head_dim + d] * sc;
        }
    }
    out[((size_t)m * n_head + h) * head_dim + d] = acc / lg;
}

// Fused Q/K/V/R L2-clip + V RMSNorm + KV store (F16 or Q8_0) + optional attn_sink_proj dot product!
// Launches 1 warp (32 threads) per block over `M * (2 * n_head + 2 * n_head_kv + (sink_w ? n_head : 0))` blocks!
__global__ void prep_qkv_r_kv_store_kernel(
    float* __restrict__ q_cur,              // [M, n_head, head_dim]
    const float* __restrict__ q_clip,       // [n_head, head_dim]
    float* __restrict__ k_cur,              // [M, n_head_kv, head_dim]
    float* __restrict__ v_cur,              // [M, n_head_kv, head_dim]
    void* __restrict__ k_cache,
    void* __restrict__ v_cache,
    float* __restrict__ r_cur,              // [M, n_head, d_rel]
    const float* __restrict__ r_clip,       // [n_head, d_rel]
    const float* __restrict__ tau,          // nullable [M]
    const int32_t* __restrict__ pos,        // [M]
    const float* __restrict__ sink_w,       // nullable [n_head, n_embd] F32
    const float* __restrict__ sink_x,       // nullable [M, n_embd] F32
    float* __restrict__ sinks_out,          // nullable [M, n_head] F32
    const float* __restrict__ kp_w,         // nullable [n_head_kv * 4, n_embd] F32
    const float* __restrict__ kp_x,         // nullable [n_embd] F32
    const float* __restrict__ k_conv_w,     // nullable [n_head_kv * head_dim, 4] F32
    const float* __restrict__ vp_w,         // nullable [n_head_kv * 4, n_embd] F32
    const float* __restrict__ vp_x,         // nullable [n_embd] F32
    const float* __restrict__ v_conv_w,     // nullable [n_head_kv * head_dim, 4] F32
    float* __restrict__ conv_history,       // nullable
    int n_conv,                             // 4
    int n_embd,
    int n_head,
    int n_head_kv,
    int head_dim,
    int d_rel,
    int M,
    int ring_cap,
    int kv_is_q8,
    float qk_clip_th,
    float rel_clip_th,
    float eps
) {
    const int gid = blockIdx.x;
    const int lane = threadIdx.x;
    const int per_tok = 2 * n_head + 2 * n_head_kv;

    if (gid >= M * per_tok) {
        int i = gid - M * per_tok;
        int m = i / n_head;
        int h = i % n_head;
        if (!sink_w || m >= M) return;
        const float* w_row = sink_w + (size_t)h * n_embd;
        const float* x_row = sink_x + (size_t)m * n_embd;
        float acc = 0.0f;
        for (int k = lane; k < n_embd; k += 32) {
            acc += w_row[k] * x_row[k];
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) {
            sinks_out[(size_t)m * n_head + h] = acc;
        }
        return;
    }

    const int m = gid / per_tok;
    const int rem = gid % per_tok;
    const float t_mul = tau ? tau[m] : 1.0f;
    const int kv_dim = n_head_kv * head_dim;

    if (rem < n_head) {
        // 1. Q L2-clip + q_clip + tau
        const int h = rem;
        float* qh = q_cur + ((size_t)m * n_head + h) * head_dim;
        const float* sh = q_clip + (size_t)h * head_dim;
        float sum2 = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            float v = qh[d] * sh[d];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        float factor = fminf(1.0f, qk_clip_th / (sqrtf(sum2) + 1e-6f)) * t_mul;
        for (int d = lane; d < head_dim; d += 32) {
            qh[d] = (qh[d] * sh[d]) * factor;
        }
    } else if (rem < n_head + n_head_kv) {
        // 2. K optional dynamic short conv (M=1) + L2-clip + KV store
        const int kh = rem - n_head;
        const int slot = pos[m] % ring_cap;
        float* kh_ptr = k_cur + ((size_t)m * n_head_kv + kh) * head_dim;
        if (M == 1 && kp_w != nullptr && kp_x != nullptr && k_conv_w != nullptr && conv_history != nullptr) {
            float dp0 = 0.0f, dp1 = 0.0f, dp2 = 0.0f, dp3 = 0.0f;
            const float* w0 = kp_w + (size_t)(kh * 4 + 0) * n_embd;
            const float* w1 = kp_w + (size_t)(kh * 4 + 1) * n_embd;
            const float* w2 = kp_w + (size_t)(kh * 4 + 2) * n_embd;
            const float* w3 = kp_w + (size_t)(kh * 4 + 3) * n_embd;
            for (int k = lane; k < n_embd; k += 32) {
                float xv = kp_x[k];
                dp0 += w0[k] * xv;
                dp1 += w1[k] * xv;
                dp2 += w2[k] * xv;
                dp3 += w3[k] * xv;
            }
            dp0 = warp_reduce_sum(dp0);
            dp1 = warp_reduce_sum(dp1);
            dp2 = warp_reduce_sum(dp2);
            dp3 = warp_reduce_sum(dp3);
            float inv_scale = rsqrtf((float)(2 * n_conv + 1));
            for (int d = lane; d < head_dim; d += 32) {
                int ch = kh * head_dim + d;
                float xt = kh_ptr[d];
                float* hist_ch = conv_history + (size_t)ch * 3;
                float h0 = hist_ch[0], h1 = hist_ch[1], h2 = hist_ch[2];
                const float* sw = k_conv_w + (size_t)ch * 4;
                float conv_val = h0 * sw[0] + h1 * sw[1] + h2 * sw[2] + xt * sw[3]
                               + xt * dp0 + h2 * dp1 + h1 * dp2 + h0 * dp3;
                kh_ptr[d] = (xt + conv_val) * inv_scale;
                hist_ch[0] = h1;
                hist_ch[1] = h2;
                hist_ch[2] = xt;
            }
        }
        float sum2 = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            float v = kh_ptr[d];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        float factor = fminf(1.0f, qk_clip_th / (sqrtf(sum2) + 1e-6f));
        if (kv_is_q8 == 0) {
            uint16_t* k_dst = reinterpret_cast<uint16_t*>(k_cache) + ((size_t)slot * n_head_kv + kh) * head_dim;
            for (int d = lane; d < head_dim; d += 32) {
                float v = kh_ptr[d] * factor;
                kh_ptr[d] = v;
                k_dst[d] = f32_to_fp16_bits(v);
            }
        } else if (kv_is_q8 == 2) {
            uint8_t* k_dst = reinterpret_cast<uint8_t*>(k_cache) + ((size_t)slot * n_head_kv + kh) * head_dim;
            for (int d = lane; d < head_dim; d += 32) {
                float v = kh_ptr[d] * factor;
                kh_ptr[d] = v;
                k_dst[d] = f32_to_e4m3fn(v * 16.0f);
            }
        } else {
            int nb_head = head_dim / 32;
            block_q8_0* k_dst_b = reinterpret_cast<block_q8_0*>(k_cache) + ((size_t)slot * n_head_kv + kh) * nb_head;
            for (int b = 0; b < nb_head; ++b) {
                float v = kh_ptr[b * 32 + lane] * factor;
                kh_ptr[b * 32 + lane] = v;
                float amax = warp_reduce_max(fabsf(v));
                float d_sc = amax / 127.0f;
                int q = (d_sc > 0.0f) ? __float2int_rn(v / d_sc) : 0;
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                k_dst_b[b].qs[lane] = (int8_t)q;
                if (lane == 0) k_dst_b[b].d = f32_to_fp16_bits(d_sc);
            }
        }
    } else if (rem < n_head + 2 * n_head_kv) {
        // 3. V optional dynamic short conv (M=1) + RMSNorm (no gamma) + KV store
        const int kh = rem - n_head - n_head_kv;
        const int slot = pos[m] % ring_cap;
        float* vh_ptr = v_cur + ((size_t)m * n_head_kv + kh) * head_dim;
        if (M == 1 && vp_w != nullptr && vp_x != nullptr && v_conv_w != nullptr && conv_history != nullptr) {
            float dp0 = 0.0f, dp1 = 0.0f, dp2 = 0.0f, dp3 = 0.0f;
            const float* w0 = vp_w + (size_t)(kh * 4 + 0) * n_embd;
            const float* w1 = vp_w + (size_t)(kh * 4 + 1) * n_embd;
            const float* w2 = vp_w + (size_t)(kh * 4 + 2) * n_embd;
            const float* w3 = vp_w + (size_t)(kh * 4 + 3) * n_embd;
            for (int k = lane; k < n_embd; k += 32) {
                float xv = vp_x[k];
                dp0 += w0[k] * xv;
                dp1 += w1[k] * xv;
                dp2 += w2[k] * xv;
                dp3 += w3[k] * xv;
            }
            dp0 = warp_reduce_sum(dp0);
            dp1 = warp_reduce_sum(dp1);
            dp2 = warp_reduce_sum(dp2);
            dp3 = warp_reduce_sum(dp3);
            float inv_scale = rsqrtf((float)(2 * n_conv + 1));
            for (int d = lane; d < head_dim; d += 32) {
                int ch = kh * head_dim + d;
                float xt = vh_ptr[d];
                float* hist_ch = conv_history + (size_t)(kv_dim + ch) * 3;
                float h0 = hist_ch[0], h1 = hist_ch[1], h2 = hist_ch[2];
                const float* sw = v_conv_w + (size_t)ch * 4;
                float conv_val = h0 * sw[0] + h1 * sw[1] + h2 * sw[2] + xt * sw[3]
                               + xt * dp0 + h2 * dp1 + h1 * dp2 + h0 * dp3;
                vh_ptr[d] = (xt + conv_val) * inv_scale;
                hist_ch[0] = h1;
                hist_ch[1] = h2;
                hist_ch[2] = xt;
            }
        }
        float sum2 = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            float v = vh_ptr[d];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        float inv_rms = rsqrtf(sum2 / (float)head_dim + eps);
        if (kv_is_q8 == 0) {
            uint16_t* v_dst = reinterpret_cast<uint16_t*>(v_cache) + ((size_t)slot * n_head_kv + kh) * head_dim;
            for (int d = lane; d < head_dim; d += 32) {
                float v = vh_ptr[d] * inv_rms;
                vh_ptr[d] = v;
                v_dst[d] = f32_to_fp16_bits(v);
            }
        } else if (kv_is_q8 == 2) {
            uint8_t* v_dst = reinterpret_cast<uint8_t*>(v_cache) + ((size_t)slot * n_head_kv + kh) * head_dim;
            for (int d = lane; d < head_dim; d += 32) {
                float v = vh_ptr[d] * inv_rms;
                vh_ptr[d] = v;
                v_dst[d] = f32_to_e4m3fn(v * 32.0f);
            }
        } else {
            int nb_head = head_dim / 32;
            block_q8_0* v_dst_b = reinterpret_cast<block_q8_0*>(v_cache) + ((size_t)slot * n_head_kv + kh) * nb_head;
            for (int b = 0; b < nb_head; ++b) {
                float v = vh_ptr[b * 32 + lane] * inv_rms;
                vh_ptr[b * 32 + lane] = v;
                float amax = warp_reduce_max(fabsf(v));
                float d_sc = amax / 127.0f;
                int q = (d_sc > 0.0f) ? __float2int_rn(v / d_sc) : 0;
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                v_dst_b[b].qs[lane] = (int8_t)q;
                if (lane == 0) v_dst_b[b].d = f32_to_fp16_bits(d_sc);
            }
        }
    } else {
        // 4. R L2-clip + r_clip + tau
        const int h = rem - n_head - 2 * n_head_kv;
        float* rh = r_cur + ((size_t)m * n_head + h) * d_rel;
        const float* sh = r_clip + (size_t)h * d_rel;
        float sum2 = 0.0f;
        for (int d = lane; d < d_rel; d += 32) {
            float v = rh[d] * sh[d];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        float factor = fminf(1.0f, rel_clip_th / (sqrtf(sum2) + 1e-6f)) * t_mul;
        for (int d = lane; d < d_rel; d += 32) {
            rh[d] = (rh[d] * sh[d]) * factor;
        }
    }
}

extern "C" {

void launch_kv_store(
    const float* k_cur,
    const float* v_cur,
    void* k_cache,
    void* v_cache,
    const int32_t* pos,
    int n_head_kv,
    int head_dim,
    int M,
    int ring_cap,
    int kv_is_q8,
    cudaStream_t stream
) {
    if (!kv_is_q8) {
        int total = M * n_head_kv * head_dim;
        int threads = 256;
        kv_store_f16_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
            k_cur, v_cur,
            reinterpret_cast<uint16_t*>(k_cache),
            reinterpret_cast<uint16_t*>(v_cache),
            pos, n_head_kv, head_dim, M, ring_cap
        );
    } else {
        dim3 grid(head_dim / 32, n_head_kv, M);
        kv_store_q8_0_kernel<<<grid, 32, 0, stream>>>(
            k_cur, v_cur,
            reinterpret_cast<uint8_t*>(k_cache),
            reinterpret_cast<uint8_t*>(v_cache),
            pos, n_head_kv, head_dim, M, ring_cap
        );
    }
}

void launch_prep_qkv_r_and_kv_store(
    float* q_cur,
    const float* q_clip,
    float* k_cur,
    float* v_cur,
    void* k_cache,
    void* v_cache,
    float* r_cur,
    const float* r_clip,
    const float* tau,
    const int32_t* pos,
    const float* sink_w,
    const float* sink_x,
    float* sinks_out,
    const float* kp_w,
    const float* kp_x,
    const float* k_conv_w,
    const float* vp_w,
    const float* vp_x,
    const float* v_conv_w,
    float* conv_history,
    int n_conv,
    int n_embd,
    int n_head,
    int n_head_kv,
    int head_dim,
    int d_rel,
    int M,
    int ring_cap,
    int kv_is_q8,
    float qk_clip_th,
    float rel_clip_th,
    float eps,
    cudaStream_t stream
) {
    int total_blocks = M * (2 * n_head + 2 * n_head_kv + (sink_w != nullptr ? n_head : 0));
    prep_qkv_r_kv_store_kernel<<<total_blocks, 32, 0, stream>>>(
        q_cur, q_clip, k_cur, v_cur, k_cache, v_cache, r_cur, r_clip, tau, pos,
        sink_w, sink_x, sinks_out,
        kp_w, kp_x, k_conv_w, vp_w, vp_x, v_conv_w, conv_history, n_conv,
        n_embd, n_head, n_head_kv, head_dim, d_rel,
        M, ring_cap, kv_is_q8, qk_clip_th, rel_clip_th, eps
    );
}

void launch_flash_attn_rel_decode_m1(
    const float* q_cur,
    const float* r_cur,
    const float* attn_rel_bias,
    const float* sinks,
    const float* attn_gate,
    const void* k_cache,
    const void* v_cache,
    const int32_t* q_pos,
    float* split_out_scratch,
    float* split_ml_scratch,
    float* out,
    int n_head,
    int n_head_kv,
    int head_dim,
    int d_rel,
    int rel_extent,
    int swa_window,
    int ring_cap,
    int kv_is_q8,
    int max_kv_len,
    cudaStream_t stream
) {
    int n_splits = (max_kv_len > 256) ? 8 : 1;
    dim3 blk(32, 4);
    dim3 grid(n_splits, n_head, 1);
    if (n_splits == 1) {
        flash_attn_rel_decode_m1_kernel<<<grid, blk, 0, stream>>>(
            q_cur, r_cur, attn_rel_bias, sinks, attn_gate, k_cache, v_cache, q_pos,
            out, split_ml_scratch,
            n_head, n_head_kv, head_dim, d_rel, rel_extent, swa_window, ring_cap, kv_is_q8, 1
        );
    } else {
        flash_attn_rel_decode_m1_kernel<<<grid, blk, 0, stream>>>(
            q_cur, r_cur, attn_rel_bias, sinks, nullptr, k_cache, v_cache, q_pos,
            split_out_scratch, split_ml_scratch,
            n_head, n_head_kv, head_dim, d_rel, rel_extent, swa_window, ring_cap, kv_is_q8, n_splits
        );
        flash_attn_rel_reduce_gate_kernel<<<n_head, head_dim, 0, stream>>>(
            split_out_scratch, split_ml_scratch, attn_gate, out, n_head, head_dim, n_splits
        );
    }
}

void launch_flash_attn_rel(
    const float* q_cur,
    const float* r_cur,
    const float* attn_rel_bias,
    const float* sinks,
    const void* k_cache,
    const void* v_cache,
    const int32_t* q_pos,
    float* rel_table_scratch,
    float* split_out_scratch,
    float* split_ml_scratch,
    float* out,
    int n_head,
    int n_head_kv,
    int head_dim,
    int d_rel,
    int rel_extent,
    int swa_window,
    int ring_cap,
    int kv_is_q8,
    int max_kv_len,
    int M,
    cudaStream_t stream
) {
    // 1. Precompute rel_table[M, n_head, rel_extent + 1]
    {
        int threads = 256;
        dim3 grid((rel_extent + 1 + threads - 1) / threads, n_head, M);
        precompute_rel_bias_kernel<<<grid, threads, 0, stream>>>(
            r_cur, attn_rel_bias, sinks, rel_table_scratch, n_head, d_rel, rel_extent, M
        );
    }

    // 2. Choose split count so 24 heads * M * n_splits saturates 82 SMs without redundant splits on short contexts
    int n_splits = 1;
    if (M == 1 && max_kv_len > 256) {
        n_splits = 8;
    } else if (M <= 4 && max_kv_len > 1024) {
        n_splits = 4;
    }

    dim3 blk(32, 4);
    dim3 grid(n_splits, n_head, M);
    if (n_splits == 1) {
        flash_attn_rel_split_kernel<<<grid, blk, 0, stream>>>(
            q_cur, k_cache, v_cache, rel_table_scratch, q_pos,
            out, split_ml_scratch,
            n_head, n_head_kv, head_dim, rel_extent, swa_window, ring_cap, kv_is_q8, 1, M
        );
    } else {
        flash_attn_rel_split_kernel<<<grid, blk, 0, stream>>>(
            q_cur, k_cache, v_cache, rel_table_scratch, q_pos,
            split_out_scratch, split_ml_scratch,
            n_head, n_head_kv, head_dim, rel_extent, swa_window, ring_cap, kv_is_q8, n_splits, M
        );
        dim3 rgrid(n_head, M);
        flash_attn_rel_reduce_kernel<<<rgrid, head_dim, 0, stream>>>(
            split_out_scratch, split_ml_scratch, out, n_head, head_dim, n_splits, M
        );
    }
}

} // extern "C"
