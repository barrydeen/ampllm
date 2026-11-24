#include "common/sm86_common.cuh"
#include <cstdio>
#include <atomic>

template <typename BlockT, int ELEMS_PER_BLOCK, float (*DotFn)(const BlockT*, const block_q8_1*)>
__device__ __forceinline__ float dot_row_blocks(
    const uint8_t* __restrict__ w_row,
    const block_q8_1* __restrict__ x_q8,
    int K,
    int lane
) {
    int nb = K / ELEMS_PER_BLOCK;
    int q8_per_w = ELEMS_PER_BLOCK / 32;
    const BlockT* blocks = reinterpret_cast<const BlockT*>(w_row);
    float acc = 0.0f;
    for (int b = lane; b < nb; b += 32) {
        acc += DotFn(blocks + b, x_q8 + b * q8_per_w);
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float dispatch_dot_row(
    int32_t w_type,
    const uint8_t* w_row,
    const block_q8_1* x_q8,
    int K,
    int lane
) {
    switch (w_type) {
        case GGML_TYPE_Q8_0:    return dot_row_q8_0_sub(reinterpret_cast<const block_q8_0*>(w_row), x_q8, K, lane);
        case GGML_TYPE_Q5_0:    return dot_row_q5_0_sub(reinterpret_cast<const block_q5_0*>(w_row), x_q8, K, lane);
        case GGML_TYPE_Q5_1:    return dot_row_q5_1_sub(reinterpret_cast<const block_q5_1*>(w_row), x_q8, K, lane);
        case GGML_TYPE_Q2_0:    return dot_row_q2_0_sub(reinterpret_cast<const block_q2_0*>(w_row), x_q8, K, lane);
        case GGML_TYPE_Q4_0:    return dot_row_blocks<block_q4_0,   32,  dot_q4_0_q8_1>(w_row, x_q8, K, lane);
        case GGML_TYPE_IQ4_NL:  return dot_row_iq4_nl_sub(reinterpret_cast<const block_iq4_nl*>(w_row), x_q8, K, lane);
        case GGML_TYPE_IQ4_XS:  return dot_row_blocks<block_iq4_xs, 256, dot_iq4_xs_q8_1>(w_row, x_q8, K, lane);
        case GGML_TYPE_Q3_K:    return dot_row_q3_K_sub(reinterpret_cast<const block_q3_K*>(w_row), x_q8, K, lane);
        case GGML_TYPE_Q4_K:    return dot_row_q4_K_sub(reinterpret_cast<const block_q4_K*>(w_row), x_q8, K, lane);
        case GGML_TYPE_Q5_K:    return dot_row_blocks<block_q5_K,   256, dot_q5_K_q8_1>(w_row, x_q8, K, lane);
        case GGML_TYPE_Q6_K:    return dot_row_q6_K_sub(reinterpret_cast<const block_q6_K*>(w_row), x_q8, K, lane);
        case GGML_TYPE_IQ2_S:   return dot_row_iq2_s_sub(reinterpret_cast<const block_iq2_s*>(w_row), x_q8, K, lane);
        case GGML_TYPE_IQ3_XXS: return dot_row_iq3_xxs_sub(reinterpret_cast<const block_iq3_xxs*>(w_row), x_q8, K, lane);
        case GGML_TYPE_IQ3_S:   return dot_row_iq3_s_sub(reinterpret_cast<const block_iq3_s*>(w_row), x_q8, K, lane);
        case GGML_TYPE_NVFP4:   return dot_row_nvfp4_sub(reinterpret_cast<const block_nvfp4*>(w_row), x_q8, K, lane);
        default: return 0.0f;
    }
}

__device__ __forceinline__ void dispatch_dot_pair(
    int32_t w_type,
    const uint8_t* w_row0,
    const uint8_t* w_row1,
    const block_q8_1* x_q8,
    int K,
    int lane,
    float& out0,
    float& out1
) {
    switch (w_type) {
        case GGML_TYPE_NVFP4:
            dot_pair_nvfp4_sub(
                reinterpret_cast<const block_nvfp4*>(w_row0),
                reinterpret_cast<const block_nvfp4*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        case GGML_TYPE_Q8_0:
            dot_pair_q8_0_sub(
                reinterpret_cast<const block_q8_0*>(w_row0),
                reinterpret_cast<const block_q8_0*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        case GGML_TYPE_Q5_1:
            dot_pair_q5_1_sub(
                reinterpret_cast<const block_q5_1*>(w_row0),
                reinterpret_cast<const block_q5_1*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        case GGML_TYPE_Q5_0:
            dot_pair_q5_0_sub(
                reinterpret_cast<const block_q5_0*>(w_row0),
                reinterpret_cast<const block_q5_0*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        case GGML_TYPE_Q2_0:
            dot_pair_q2_0_sub(
                reinterpret_cast<const block_q2_0*>(w_row0),
                reinterpret_cast<const block_q2_0*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        case GGML_TYPE_IQ4_NL:
            dot_pair_iq4_nl_sub(
                reinterpret_cast<const block_iq4_nl*>(w_row0),
                reinterpret_cast<const block_iq4_nl*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        case GGML_TYPE_Q3_K:
            dot_pair_q3_K_sub(
                reinterpret_cast<const block_q3_K*>(w_row0),
                reinterpret_cast<const block_q3_K*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        case GGML_TYPE_IQ2_S:
            dot_pair_iq2_s_sub(
                reinterpret_cast<const block_iq2_s*>(w_row0),
                reinterpret_cast<const block_iq2_s*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        case GGML_TYPE_IQ4_XS:
            dot_pair_iq4_xs_sub(
                reinterpret_cast<const block_iq4_xs*>(w_row0),
                reinterpret_cast<const block_iq4_xs*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        case GGML_TYPE_Q5_K:
            dot_pair_q5_K_sub(
                reinterpret_cast<const block_q5_K*>(w_row0),
                reinterpret_cast<const block_q5_K*>(w_row1),
                x_q8, K, lane, out0, out1
            );
            break;
        default:
            out0 = dispatch_dot_row(w_type, w_row0, x_q8, K, lane);
            out1 = dispatch_dot_row(w_type, w_row1, x_q8, K, lane);
            break;
    }
}

// -----------------------------------------------------------------------------
// GLM-5.3 split gate/up resident MoE kernels (clamped SwiGLU + in-kernel q8_1)
// -----------------------------------------------------------------------------

template <int W_TYPE>
__global__ void __launch_bounds__(256) glm5_moe_gate_up_swiglu_kernel(
    const uint64_t* __restrict__ gate_ptrs, // [n_expert], 0 if on CPU
    const uint64_t* __restrict__ up_ptrs,   // [n_expert]
    size_t row_bytes,
    const block_q8_1* __restrict__ x_q8,    // [M, n_embd / 32]
    const int32_t* __restrict__ expert_ids, // [M, topk]
    block_q8_1* __restrict__ inter_q8,      // [M, topk, n_ff / 32]
    float* __restrict__ inter_f32,          // nullable [M, topk, n_ff]
    int n_embd,
    int n_ff,
    int topk,
    float clamp_lim,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    // 8 warps = 256 threads compute 32 consecutive output rows (1 q8_1 block).
    const int r_base = blockIdx.x * 32;
    const int slot   = blockIdx.y;
    const int m      = blockIdx.z;
    const int tid    = threadIdx.y * 32 + threadIdx.x;
    const int warp   = threadIdx.y;
    const int lane   = threadIdx.x;

    const int e = expert_ids[m * topk + slot];
    const uint8_t* g_ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(gate_ptrs[e]));
    const uint8_t* u_ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(up_ptrs[e]));
    if (g_ptr == nullptr || u_ptr == nullptr) return;

    constexpr int MAX_STAGED_NB = 256; // up to K=8192 (n_embd=4096 uses 128)
    __shared__ __align__(16) uint32_t s_x_words[MAX_STAGED_NB * 9];
    __shared__ __align__(16) uint32_t s_iq2s_u32[W_TYPE == GGML_TYPE_IQ2_S ? 2048 : 4];
    __shared__ float s_h[32];

    const int nb_x = n_embd >> 5;
    const uint32_t* gx_words = reinterpret_cast<const uint32_t*>(x_q8 + (size_t)m * nb_x);
    const int total_words = nb_x * 9;
    if (nb_x <= MAX_STAGED_NB) {
        for (int i = tid; i < total_words; i += 256) {
            s_x_words[i] = gx_words[i];
        }
    }
    if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
        const uint32_t* g_grid = reinterpret_cast<const uint32_t*>(d_iq2s_grid);
        for (int i = tid; i < 2048; i += 256) {
            s_iq2s_u32[i] = g_grid[i];
        }
    }
    __syncthreads();

    const block_q8_1* x_use = (nb_x <= MAX_STAGED_NB) ? reinterpret_cast<const block_q8_1*>(s_x_words) : (x_q8 + (size_t)m * nb_x);

    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        int r = r_base + warp * 4 + step;
        if (r < n_ff) {
            const uint8_t* gr = g_ptr + (size_t)r * row_bytes;
            const uint8_t* ur = u_ptr + (size_t)r * row_bytes;
            float vg = 0.0f, vu = 0.0f;
            if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
                dot_pair_iq2_s_sub(
                    reinterpret_cast<const block_iq2_s*>(gr),
                    reinterpret_cast<const block_iq2_s*>(ur),
                    x_use, n_embd, lane, vg, vu, reinterpret_cast<const uint64_t*>(s_iq2s_u32)
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q3_K) {
                dot_pair_q3_K_sub(
                    reinterpret_cast<const block_q3_K*>(gr),
                    reinterpret_cast<const block_q3_K*>(ur),
                    x_use, n_embd, lane, vg, vu
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q6_K) {
                dot_pair_q6_K_sub(
                    reinterpret_cast<const block_q6_K*>(gr),
                    reinterpret_cast<const block_q6_K*>(ur),
                    x_use, n_embd, lane, vg, vu
                );
            } else if constexpr (W_TYPE == GGML_TYPE_IQ4_NL) {
                dot_pair_iq4_nl_sub(
                    reinterpret_cast<const block_iq4_nl*>(gr),
                    reinterpret_cast<const block_iq4_nl*>(ur),
                    x_use, n_embd, lane, vg, vu
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q5_0) {
                dot_pair_q5_0_sub(
                    reinterpret_cast<const block_q5_0*>(gr),
                    reinterpret_cast<const block_q5_0*>(ur),
                    x_use, n_embd, lane, vg, vu
                );
            } else {
                vg = dispatch_dot_row(W_TYPE, gr, x_use, n_embd, lane);
                vu = dispatch_dot_row(W_TYPE, ur, x_use, n_embd, lane);
            }
            if (lane == 0) {
                float h = glm_swiglu_clamp_val(vg, vu, clamp_lim);
                s_h[warp * 4 + step] = h;
                if (inter_f32 != nullptr) {
                    inter_f32[((size_t)m * topk + slot) * n_ff + r] = h;
                }
            }
        } else if (lane == 0) {
            s_h[warp * 4 + step] = 0.0f;
        }
    }
    __syncthreads();

    if (warp == 0 && (r_base < n_ff)) {
        block_q8_1* dst_blk = inter_q8 + (((size_t)m * topk + slot) * (n_ff >> 5) + blockIdx.x);
        warp_quantize_q8_1_store(s_h[lane], dst_blk, lane);
    }
}

template <int W_TYPE>
__global__ void __launch_bounds__(128) glm5_moe_down_reduce_kernel(
    const uint64_t* __restrict__ down_ptrs, // [n_expert]
    size_t row_bytes,
    const block_q8_1* __restrict__ inter_q8, // [M, topk, n_ff / 32]
    const int32_t* __restrict__ expert_ids,  // [M, topk]
    const float* __restrict__ router_w,      // [M, topk]
    float* __restrict__ out,                 // [M, n_embd]
    int n_embd,
    int n_ff,
    int topk,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    // 4 warps = 128 threads -> 4 output rows per block
    const int r    = blockIdx.x * 4 + threadIdx.y;
    const int m    = blockIdx.y;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;

    const int nb_ff = n_ff >> 5;
    constexpr int MAX_STAGED_TOTAL_BLKS = 512;
    // Stage up to 512 inter_q8 blocks (e.g. 8x2048 or 10x640 = 200 blks -> 7.2 KB) + IQ3 codebook into SMEM
    __shared__ __align__(16) uint32_t s_inter_words[MAX_STAGED_TOTAL_BLKS * 9];
    __shared__ __align__(16) uint32_t s_grid[(W_TYPE == GGML_TYPE_IQ3_S) ? 512 : 256];

    if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
        for (int i = tid; i < 256; i += 128) s_grid[i] = d_iq3xxs_grid[i];
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_S) {
        for (int i = tid; i < 512; i += 128) s_grid[i] = d_iq3s_grid[i];
    }
    const bool staged_inter = (topk * nb_ff <= MAX_STAGED_TOTAL_BLKS);
    if (staged_inter) {
        const uint32_t* src_words = reinterpret_cast<const uint32_t*>(inter_q8 + (size_t)m * topk * nb_ff);
        const int total_words = topk * nb_ff * 9;
        for (int i = tid; i < total_words; i += 128) {
            s_inter_words[i] = src_words[i];
        }
    }
    __syncthreads();

    if (r >= n_embd) return;

    float sum = 0.0f;
    for (int s = 0; s < topk; ++s) {
        const int e = expert_ids[m * topk + s];
        const uint8_t* dp = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(down_ptrs[e]));
        if (dp == nullptr) continue;
        const float w = router_w[m * topk + s];
        if (w == 0.0f) continue;

        const block_q8_1* x_use = staged_inter
            ? reinterpret_cast<const block_q8_1*>(s_inter_words + s * nb_ff * 9)
            : (inter_q8 + ((size_t)m * topk + s) * nb_ff);
        const uint8_t* wr = dp + (size_t)r * row_bytes;
        float dot = 0.0f;
        if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
            dot = dot_row_iq3_xxs_sub(reinterpret_cast<const block_iq3_xxs*>(wr), x_use, n_ff, lane, s_grid);
        } else if constexpr (W_TYPE == GGML_TYPE_IQ3_S) {
            dot = dot_row_iq3_s_sub(reinterpret_cast<const block_iq3_s*>(wr), x_use, n_ff, lane, s_grid);
        } else if constexpr (W_TYPE == GGML_TYPE_Q4_K) {
            dot = dot_row_q4_K_sub(reinterpret_cast<const block_q4_K*>(wr), x_use, n_ff, lane);
        } else if constexpr (W_TYPE == GGML_TYPE_Q5_K) {
            dot = dot_row_q5_K_sub(reinterpret_cast<const block_q5_K*>(wr), x_use, n_ff, lane);
        } else if constexpr (W_TYPE == GGML_TYPE_Q8_0) {
            dot = dot_row_q8_0_sub(reinterpret_cast<const block_q8_0*>(wr), x_use, n_ff, lane);
        } else if constexpr (W_TYPE == GGML_TYPE_Q5_1) {
            dot = dot_row_q5_1_sub(reinterpret_cast<const block_q5_1*>(wr), x_use, n_ff, lane);
        } else if constexpr (W_TYPE == GGML_TYPE_Q5_0) {
            dot = dot_row_q5_0_sub(reinterpret_cast<const block_q5_0*>(wr), x_use, n_ff, lane);
        } else if constexpr (W_TYPE == GGML_TYPE_IQ4_NL) {
            dot = dot_row_iq4_nl_sub(reinterpret_cast<const block_iq4_nl*>(wr), x_use, n_ff, lane);
        } else if constexpr (W_TYPE == GGML_TYPE_Q2_0) {
            dot = dot_row_q2_0_sub(reinterpret_cast<const block_q2_0*>(wr), x_use, n_ff, lane);
        } else if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
            dot = dot_row_iq2_s_sub(reinterpret_cast<const block_iq2_s*>(wr), x_use, n_ff, lane);
        } else {
            dot = dispatch_dot_row(W_TYPE, wr, x_use, n_ff, lane);
        }
        sum += w * dot;
    }

    if (lane == 0) {
        out[(size_t)m * n_embd + r] = sum;
    }
}

template <int W_TYPE>
__global__ void __launch_bounds__(256, 3) glm5_moe_gate_up_plan_kernel(
    const uint64_t* __restrict__ plan_ptrs, // [M, topk] contiguous blob base ptrs, 0 if on CPU
    uint32_t u_off,
    size_t row_bytes,
    const block_q8_1* __restrict__ x_q8,    // [M, n_embd / 32]
    block_q8_1* __restrict__ inter_q8,      // [M, topk, n_ff / 32]
    const void* __restrict__ sh_down_w,     // nullable [n_embd, n_ff_sh]
    int32_t sh_down_type,
    size_t sh_down_row_bytes,
    const block_q8_1* __restrict__ sh_hq,   // nullable [M, n_ff_sh / 32]
    int n_ff_sh,
    float* __restrict__ sh_out_opt,         // nullable [M, n_embd]
    int n_embd,
    int n_ff,
    int topk,
    float clamp_lim,
    const int32_t* __restrict__ skip_dev,
    const uint64_t* __restrict__ fetch_src_filter,
    int filter_mode
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int slot = blockIdx.y;
    const int m    = blockIdx.z;
    const int warp = threadIdx.y; // 0..7
    const int lane = threadIdx.x; // 0..31
    const int tid  = warp * 32 + lane;

    if (slot == topk) {
        if (filter_mode == 2) return;
        const int n_chunks = (int)gridDim.x;
        const int per = (n_embd + n_chunks - 1) / n_chunks;
        const int r0 = (int)blockIdx.x * per;
        const int r1 = min(n_embd, r0 + per);
        if (r0 >= r1) return;
        const uint8_t* sw = reinterpret_cast<const uint8_t*>(sh_down_w);
        const block_q8_1* x_sh = sh_hq + (size_t)m * (n_ff_sh >> 5);
        float* y_sh = sh_out_opt + (size_t)m * n_embd;
        if (sh_down_type == GGML_TYPE_Q6_K) {
            const int n_sh = r1 - r0;
            const int n_quad = n_sh >> 2;
            for (int q = warp; q < n_quad; q += 8) {
                const int r = r0 + (q << 2);
                float o0, o1, o2, o3;
                dot_quad_q6_K_sub(
                    reinterpret_cast<const block_q6_K*>(sw + (size_t)(r + 0) * sh_down_row_bytes),
                    reinterpret_cast<const block_q6_K*>(sw + (size_t)(r + 1) * sh_down_row_bytes),
                    reinterpret_cast<const block_q6_K*>(sw + (size_t)(r + 2) * sh_down_row_bytes),
                    reinterpret_cast<const block_q6_K*>(sw + (size_t)(r + 3) * sh_down_row_bytes),
                    x_sh, n_ff_sh, lane, o0, o1, o2, o3
                );
                if (lane == 0) {
                    y_sh[r + 0] = o0;
                    y_sh[r + 1] = o1;
                    y_sh[r + 2] = o2;
                    y_sh[r + 3] = o3;
                }
            }
            const int tail0 = r0 + (n_quad << 2);
            for (int r = tail0 + warp; r < r1; r += 8) {
                float v = dot_row_q6_K_sub(
                    reinterpret_cast<const block_q6_K*>(sw + (size_t)r * sh_down_row_bytes),
                    x_sh, n_ff_sh, lane
                );
                if (lane == 0) y_sh[r] = v;
            }
        } else if (sh_down_type == GGML_TYPE_Q8_0) {
            const int n_sh = r1 - r0;
            const int n_quad = n_sh >> 2;
            for (int q = warp; q < n_quad; q += 8) {
                const int r = r0 + (q << 2);
                float o0, o1, o2, o3;
                dot_quad_q8_0_sub(
                    reinterpret_cast<const block_q8_0*>(sw + (size_t)(r + 0) * sh_down_row_bytes),
                    reinterpret_cast<const block_q8_0*>(sw + (size_t)(r + 1) * sh_down_row_bytes),
                    reinterpret_cast<const block_q8_0*>(sw + (size_t)(r + 2) * sh_down_row_bytes),
                    reinterpret_cast<const block_q8_0*>(sw + (size_t)(r + 3) * sh_down_row_bytes),
                    x_sh, n_ff_sh, lane, o0, o1, o2, o3
                );
                if (lane == 0) {
                    y_sh[r + 0] = o0;
                    y_sh[r + 1] = o1;
                    y_sh[r + 2] = o2;
                    y_sh[r + 3] = o3;
                }
            }
            const int tail0 = r0 + (n_quad << 2);
            for (int r = tail0 + warp; r < r1; r += 8) {
                float v = dot_row_q8_0_sub(
                    reinterpret_cast<const block_q8_0*>(sw + (size_t)r * sh_down_row_bytes),
                    x_sh, n_ff_sh, lane
                );
                if (lane == 0) y_sh[r] = v;
            }
        } else {
            for (int r = r0 + warp; r < r1; r += 8) {
                float v = dispatch_dot_row(sh_down_type, sw + (size_t)r * sh_down_row_bytes, x_sh, n_ff_sh, lane);
                if (lane == 0) y_sh[r] = v;
            }
        }
        return;
    }

    if (fetch_src_filter != nullptr && filter_mode != 0) {
        const uint64_t fs = fetch_src_filter[(size_t)m * topk + slot];
        if (filter_mode == 1 && fs != 0ULL) return;
        if (filter_mode == 2 && fs == 0ULL) return;
    }

    const uint64_t base = plan_ptrs[(size_t)m * topk + slot];
    if (base == 0ULL) return;
    const uint8_t* g_ptr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(base));

    constexpr int MAX_STAGED_NB = 128; // n_embd=4096 -> 4608 B
    __shared__ __align__(16) uint32_t s_iq2s_u32[W_TYPE == GGML_TYPE_IQ2_S ? 2048 : 4];
    __shared__ __align__(16) uint32_t s_xq8[MAX_STAGED_NB * 9];
    __shared__ float s_h[32];

    const int nb_x = n_embd >> 5;
    if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
        const uint32_t* g_grid = reinterpret_cast<const uint32_t*>(d_iq2s_grid);
        for (int i = tid; i < 2048; i += 256) {
            s_iq2s_u32[i] = g_grid[i];
        }
    }
    const bool staged_x = (nb_x <= MAX_STAGED_NB);
    if (staged_x) {
        const uint32_t* gx_words = reinterpret_cast<const uint32_t*>(x_q8 + (size_t)m * nb_x);
        const int total_words = nb_x * 9;
        for (int i = tid; i < total_words; i += 256) {
            s_xq8[i] = gx_words[i];
        }
    }
    __syncthreads();

    const block_q8_1* x_use = staged_x ? reinterpret_cast<const block_q8_1*>(s_xq8) : (x_q8 + (size_t)m * nb_x);
    const int nb_ff = n_ff >> 5;
    const int rb = (int)blockIdx.x;
    const int r_base = rb * 32;

    #pragma unroll 1
    for (int p = 0; p < 2; ++p) {
        const int r0 = r_base + warp * 4 + p * 2;
        const int r1 = r0 + 1;
        const int r1_c = (r1 < n_ff) ? r1 : r0;

        float vg0 = 0.0f, vu0 = 0.0f, vg1 = 0.0f, vu1 = 0.0f;
        if (r0 < n_ff) {
            const uint8_t* g0 = g_ptr + (size_t)r0 * row_bytes;
            const uint8_t* g1 = g_ptr + (size_t)r1_c * row_bytes;
            const uint8_t* u0 = g0 + u_off;
            const uint8_t* u1 = g1 + u_off;
            if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
                dot_quad_iq2_s_sub(
                    reinterpret_cast<const block_iq2_s*>(g0),
                    reinterpret_cast<const block_iq2_s*>(g1),
                    reinterpret_cast<const block_iq2_s*>(u0),
                    reinterpret_cast<const block_iq2_s*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1,
                    reinterpret_cast<const uint64_t*>(s_iq2s_u32)
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q3_K) {
                dot_quad_q3_K_sub(
                    reinterpret_cast<const block_q3_K*>(g0),
                    reinterpret_cast<const block_q3_K*>(g1),
                    reinterpret_cast<const block_q3_K*>(u0),
                    reinterpret_cast<const block_q3_K*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q4_K) {
                dot_quad_q4_K_sub(
                    reinterpret_cast<const block_q4_K*>(g0),
                    reinterpret_cast<const block_q4_K*>(g1),
                    reinterpret_cast<const block_q4_K*>(u0),
                    reinterpret_cast<const block_q4_K*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q6_K) {
                dot_quad_q6_K_sub(
                    reinterpret_cast<const block_q6_K*>(g0),
                    reinterpret_cast<const block_q6_K*>(g1),
                    reinterpret_cast<const block_q6_K*>(u0),
                    reinterpret_cast<const block_q6_K*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q5_K) {
                dot_quad_q5_K_sub(
                    reinterpret_cast<const block_q5_K*>(g0),
                    reinterpret_cast<const block_q5_K*>(g1),
                    reinterpret_cast<const block_q5_K*>(u0),
                    reinterpret_cast<const block_q5_K*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else if constexpr (W_TYPE == GGML_TYPE_IQ4_XS) {
                dot_quad_iq4_xs_sub(
                    reinterpret_cast<const block_iq4_xs*>(g0),
                    reinterpret_cast<const block_iq4_xs*>(g1),
                    reinterpret_cast<const block_iq4_xs*>(u0),
                    reinterpret_cast<const block_iq4_xs*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q8_0) {
                dot_quad_q8_0_sub(
                    reinterpret_cast<const block_q8_0*>(g0),
                    reinterpret_cast<const block_q8_0*>(g1),
                    reinterpret_cast<const block_q8_0*>(u0),
                    reinterpret_cast<const block_q8_0*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q5_1) {
                dot_quad_q5_1_sub(
                    reinterpret_cast<const block_q5_1*>(g0),
                    reinterpret_cast<const block_q5_1*>(g1),
                    reinterpret_cast<const block_q5_1*>(u0),
                    reinterpret_cast<const block_q5_1*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q2_0) {
                dot_quad_q2_0_sub(
                    reinterpret_cast<const block_q2_0*>(g0),
                    reinterpret_cast<const block_q2_0*>(g1),
                    reinterpret_cast<const block_q2_0*>(u0),
                    reinterpret_cast<const block_q2_0*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else if constexpr (W_TYPE == GGML_TYPE_IQ4_NL) {
                dot_quad_iq4_nl_sub(
                    reinterpret_cast<const block_iq4_nl*>(g0),
                    reinterpret_cast<const block_iq4_nl*>(g1),
                    reinterpret_cast<const block_iq4_nl*>(u0),
                    reinterpret_cast<const block_iq4_nl*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else if constexpr (W_TYPE == GGML_TYPE_Q5_0) {
                dot_quad_q5_0_sub(
                    reinterpret_cast<const block_q5_0*>(g0),
                    reinterpret_cast<const block_q5_0*>(g1),
                    reinterpret_cast<const block_q5_0*>(u0),
                    reinterpret_cast<const block_q5_0*>(u1),
                    x_use, n_embd, lane, vg0, vg1, vu0, vu1
                );
            } else {
                vg0 = dispatch_dot_row(W_TYPE, g0, x_use, n_embd, lane);
                vg1 = dispatch_dot_row(W_TYPE, g1, x_use, n_embd, lane);
                vu0 = dispatch_dot_row(W_TYPE, u0, x_use, n_embd, lane);
                vu1 = dispatch_dot_row(W_TYPE, u1, x_use, n_embd, lane);
            }
        }

        if (lane == 0) {
            s_h[warp * 4 + p * 2 + 0] = (r0 < n_ff) ? glm_swiglu_clamp_val(vg0, vu0, clamp_lim) : 0.0f;
            s_h[warp * 4 + p * 2 + 1] = (r1 < n_ff) ? glm_swiglu_clamp_val(vg1, vu1, clamp_lim) : 0.0f;
        }
    }
    __syncthreads();

    if (warp == 0 && (r_base < n_ff)) {
        block_q8_1* dst_blk = inter_q8 + (((size_t)m * topk + slot) * nb_ff + rb);
        warp_quantize_q8_1_store(s_h[lane], dst_blk, lane);
    }
}

template <int W_TYPE>
__global__ void __launch_bounds__(256) glm5_moe_down_plan_kernel(
    const uint64_t* __restrict__ plan_ptrs, // [M, topk] contiguous blob base ptrs, 0 if on CPU
    uint32_t d_off,
    size_t row_bytes,
    const block_q8_1* __restrict__ inter_q8, // [M, topk, n_ff / 32]
    const float* __restrict__ router_w,      // [M, topk]
    const float* __restrict__ sh_out_opt,    // nullable [M, n_embd]
    float* __restrict__ out,                 // [M, n_embd]
    int n_embd,
    int n_ff,
    int topk,
    const int32_t* __restrict__ skip_dev,
    const uint64_t* __restrict__ fetch_src_filter,
    int filter_mode
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    constexpr int NR = 4;
    const int m       = (int)blockIdx.y;
    const int warp    = threadIdx.y; // 0..7
    const int lane    = threadIdx.x; // 0..31
    const int tid     = warp * 32 + lane;
    const int nb_ff   = n_ff >> 5;
    const int n_quads = (n_embd + NR - 1) / NR;

    __shared__ uint64_t s_act_base[16];
    __shared__ float s_act_rw[16];
    __shared__ int s_act_slot[16];
    __shared__ int s_n_act;
    __shared__ float s_part[8][NR];
    __shared__ __align__(16) uint32_t s_grid[(W_TYPE == GGML_TYPE_IQ3_S) ? 512 : 256];

    if (tid == 0) {
        int na = 0;
        for (int k = 0; k < topk && k < 16; ++k) {
            uint64_t bp = plan_ptrs[(size_t)m * topk + k];
            if (fetch_src_filter != nullptr && filter_mode != 0) {
                const uint64_t fs = fetch_src_filter[(size_t)m * topk + k];
                if (filter_mode == 1 && fs != 0ULL) bp = 0ULL;
                else if (filter_mode == 2 && fs == 0ULL) bp = 0ULL;
            }
            if (bp != 0ULL) {
                s_act_base[na] = bp;
                s_act_rw[na]   = router_w[(size_t)m * topk + k];
                s_act_slot[na] = k;
                ++na;
            }
        }
        s_n_act = na;
    }
    if (filter_mode == 2) {
        __syncthreads();
        if (s_n_act == 0) return;
        if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
            for (int i = tid; i < 256; i += 256) s_grid[i] = d_iq3xxs_grid[i];
        } else if constexpr (W_TYPE == GGML_TYPE_IQ3_S) {
            for (int i = tid; i < 512; i += 256) s_grid[i] = d_iq3s_grid[i];
        }
        __syncthreads();
    } else {
        if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
            for (int i = tid; i < 256; i += 256) s_grid[i] = d_iq3xxs_grid[i];
        } else if constexpr (W_TYPE == GGML_TYPE_IQ3_S) {
            for (int i = tid; i < 512; i += 256) s_grid[i] = d_iq3s_grid[i];
        }
        __syncthreads();
    }

    const int n_act = s_n_act;
    if (filter_mode != 0 && n_act == 0 && sh_out_opt == nullptr) return;

    auto eval_quad = [&](uint64_t base, const block_q8_1* x_use, int r0, float& o0, float& o1, float& o2, float& o3) {
        const uint8_t* dp = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(base + d_off));
        const int rr0 = r0;
        const int rr1 = min(r0 + 1, n_embd - 1);
        const int rr2 = min(r0 + 2, n_embd - 1);
        const int rr3 = min(r0 + 3, n_embd - 1);
        const uint8_t* w0 = dp + (size_t)rr0 * row_bytes;
        const uint8_t* w1 = dp + (size_t)rr1 * row_bytes;
        const uint8_t* w2 = dp + (size_t)rr2 * row_bytes;
        const uint8_t* w3 = dp + (size_t)rr3 * row_bytes;
        if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
            dot_quad_iq3_xxs_sub(
                reinterpret_cast<const block_iq3_xxs*>(w0),
                reinterpret_cast<const block_iq3_xxs*>(w1),
                reinterpret_cast<const block_iq3_xxs*>(w2),
                reinterpret_cast<const block_iq3_xxs*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3, s_grid
            );
        } else if constexpr (W_TYPE == GGML_TYPE_IQ3_S) {
            dot_quad_iq3_s_sub(
                reinterpret_cast<const block_iq3_s*>(w0),
                reinterpret_cast<const block_iq3_s*>(w1),
                reinterpret_cast<const block_iq3_s*>(w2),
                reinterpret_cast<const block_iq3_s*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3, s_grid
            );
        } else if constexpr (W_TYPE == GGML_TYPE_Q4_K) {
            dot_quad_q4_K_sub(
                reinterpret_cast<const block_q4_K*>(w0),
                reinterpret_cast<const block_q4_K*>(w1),
                reinterpret_cast<const block_q4_K*>(w2),
                reinterpret_cast<const block_q4_K*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
            dot_quad_iq2_s_sub(
                reinterpret_cast<const block_iq2_s*>(w0),
                reinterpret_cast<const block_iq2_s*>(w1),
                reinterpret_cast<const block_iq2_s*>(w2),
                reinterpret_cast<const block_iq2_s*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else if constexpr (W_TYPE == GGML_TYPE_Q6_K) {
            dot_quad_q6_K_sub(
                reinterpret_cast<const block_q6_K*>(w0),
                reinterpret_cast<const block_q6_K*>(w1),
                reinterpret_cast<const block_q6_K*>(w2),
                reinterpret_cast<const block_q6_K*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else if constexpr (W_TYPE == GGML_TYPE_Q5_K) {
            dot_quad_q5_K_sub(
                reinterpret_cast<const block_q5_K*>(w0),
                reinterpret_cast<const block_q5_K*>(w1),
                reinterpret_cast<const block_q5_K*>(w2),
                reinterpret_cast<const block_q5_K*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else if constexpr (W_TYPE == GGML_TYPE_IQ4_XS) {
            dot_quad_iq4_xs_sub(
                reinterpret_cast<const block_iq4_xs*>(w0),
                reinterpret_cast<const block_iq4_xs*>(w1),
                reinterpret_cast<const block_iq4_xs*>(w2),
                reinterpret_cast<const block_iq4_xs*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else if constexpr (W_TYPE == GGML_TYPE_Q8_0) {
            dot_quad_q8_0_sub(
                reinterpret_cast<const block_q8_0*>(w0),
                reinterpret_cast<const block_q8_0*>(w1),
                reinterpret_cast<const block_q8_0*>(w2),
                reinterpret_cast<const block_q8_0*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else if constexpr (W_TYPE == GGML_TYPE_Q5_1) {
            dot_quad_q5_1_sub(
                reinterpret_cast<const block_q5_1*>(w0),
                reinterpret_cast<const block_q5_1*>(w1),
                reinterpret_cast<const block_q5_1*>(w2),
                reinterpret_cast<const block_q5_1*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else if constexpr (W_TYPE == GGML_TYPE_Q2_0) {
            dot_quad_q2_0_sub(
                reinterpret_cast<const block_q2_0*>(w0),
                reinterpret_cast<const block_q2_0*>(w1),
                reinterpret_cast<const block_q2_0*>(w2),
                reinterpret_cast<const block_q2_0*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else if constexpr (W_TYPE == GGML_TYPE_IQ4_NL) {
            dot_quad_iq4_nl_sub(
                reinterpret_cast<const block_iq4_nl*>(w0),
                reinterpret_cast<const block_iq4_nl*>(w1),
                reinterpret_cast<const block_iq4_nl*>(w2),
                reinterpret_cast<const block_iq4_nl*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else if constexpr (W_TYPE == GGML_TYPE_Q5_0) {
            dot_quad_q5_0_sub(
                reinterpret_cast<const block_q5_0*>(w0),
                reinterpret_cast<const block_q5_0*>(w1),
                reinterpret_cast<const block_q5_0*>(w2),
                reinterpret_cast<const block_q5_0*>(w3),
                x_use, n_ff, lane, o0, o1, o2, o3
            );
        } else {
            o0 = dispatch_dot_row(W_TYPE, w0, x_use, n_ff, lane);
            o1 = dispatch_dot_row(W_TYPE, w1, x_use, n_ff, lane);
            o2 = dispatch_dot_row(W_TYPE, w2, x_use, n_ff, lane);
            o3 = dispatch_dot_row(W_TYPE, w3, x_use, n_ff, lane);
        }
    };

    const bool two_quad = (n_act <= 4);
    const int rq_base = two_quad ? ((int)blockIdx.x * 2) : (int)blockIdx.x;
    if (rq_base >= n_quads) return;

    const int red_threads = two_quad ? 8 : NR;
    if (two_quad) {
        const int grp = warp >> 2;
        const int w_idx = warp & 3;
        const uint64_t base = (w_idx < n_act) ? s_act_base[w_idx] : 0ULL;
        const int slot_id = (w_idx < n_act) ? s_act_slot[w_idx] : 0;
        const block_q8_1* x_use = inter_q8 + ((size_t)m * topk + slot_id) * nb_ff;
        const int rq = rq_base + grp;
        const int r0 = rq * NR;
        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        if (rq < n_quads && base != 0ULL && r0 < n_embd) {
            eval_quad(base, x_use, r0, o0, o1, o2, o3);
        }
        if (w_idx < n_act && lane == 0) {
            const int ps = grp * 4 + w_idx;
            s_part[ps][0] = o0;
            s_part[ps][1] = o1;
            s_part[ps][2] = o2;
            s_part[ps][3] = o3;
        }
    } else {
        const int r0 = rq_base * NR;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int idx = warp; idx < n_act; idx += 8) {
            const uint64_t base = s_act_base[idx];
            if (base != 0ULL && r0 < n_embd) {
                const int slot_id = s_act_slot[idx];
                const block_q8_1* x_use = inter_q8 + ((size_t)m * topk + slot_id) * nb_ff;
                float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                eval_quad(base, x_use, r0, o0, o1, o2, o3);
                const float rw = s_act_rw[idx];
                a0 += rw * o0;
                a1 += rw * o1;
                a2 += rw * o2;
                a3 += rw * o3;
            }
        }
        if (lane == 0) {
            s_part[warp][0] = a0;
            s_part[warp][1] = a1;
            s_part[warp][2] = a2;
            s_part[warp][3] = a3;
        }
    }
    __syncthreads();

    if (tid < red_threads) {
        const int g = two_quad ? (tid >> 2) : 0;
        const int c = tid & 3;
        const int r = (rq_base + g) * NR + c;
        if (r < n_embd) {
            float sum = 0.0f;
            if (two_quad) {
                for (int i = 0; i < n_act; ++i) {
                    sum += s_act_rw[i] * s_part[g * 4 + i][c];
                }
            } else {
                const int nw = (n_act < 8) ? n_act : 8;
                for (int i = 0; i < nw; ++i) {
                    sum += s_part[i][c];
                }
            }
            if (filter_mode == 2) {
                out[(size_t)m * n_embd + r] += sum;
            } else {
                if (sh_out_opt != nullptr) {
                    sum += sh_out_opt[(size_t)m * n_embd + r];
                }
                out[(size_t)m * n_embd + r] = sum;
            }
        }
    }
}

extern "C" void launch_glm5_moe_plan_filtered(
    int32_t gu_type,
    int32_t d_type,
    const uint64_t* plan_ptrs,
    uint32_t u_off,
    uint32_t d_off,
    size_t gu_row_bytes,
    size_t d_row_bytes,
    const void* x_q8,
    const float* router_w,
    void* inter_q8,
    int32_t sh_down_type,
    const void* sh_down_w,
    size_t sh_down_row_bytes,
    const void* sh_hq,
    int n_ff_sh,
    float* sh_out_opt,
    float* out_f32,
    int n_embd,
    int n_ff,
    int topk,
    int M,
    float clamp_lim,
    const int32_t* skip_dev,
    const uint64_t* fetch_src_filter,
    int filter_mode,
    cudaStream_t stream
) {
    if (M <= 0 || n_ff <= 0 || n_embd <= 0 || topk <= 0) return;
    const bool fuse_sh_down = (filter_mode != 2)
        && (sh_down_w != nullptr && sh_hq != nullptr && sh_out_opt != nullptr && n_ff_sh > 0);
    const int n_row_blocks = (n_ff + 31) / 32;
    const int gu_gx = n_row_blocks;
    dim3 grid_gu(gu_gx, fuse_sh_down ? (topk + 1) : topk, M);
    dim3 block_gu(32, 8);
    const auto* xq = reinterpret_cast<const block_q8_1*>(x_q8);
    auto* iq8 = reinterpret_cast<block_q8_1*>(inter_q8);
    const auto* sh_q8 = reinterpret_cast<const block_q8_1*>(sh_hq);

    switch (gu_type) {
        case GGML_TYPE_IQ2_S:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_IQ2_S><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q3_K:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_Q3_K><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q6_K:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_Q6_K><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q4_K:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_Q4_K><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q5_K:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_Q5_K><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_IQ4_XS:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_IQ4_XS><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q8_0:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_Q8_0><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q5_1:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_Q5_1><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q2_0:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_Q2_0><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_IQ4_NL:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_IQ4_NL><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q5_0:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_Q5_0><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_IQ3_S:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_IQ3_S><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_IQ3_XXS:
            glm5_moe_gate_up_plan_kernel<GGML_TYPE_IQ3_XXS><<<grid_gu, block_gu, 0, stream>>>(
                plan_ptrs, u_off, gu_row_bytes, xq, iq8,
                sh_down_w, sh_down_type, sh_down_row_bytes, sh_q8, n_ff_sh, sh_out_opt,
                n_embd, n_ff, topk, clamp_lim, skip_dev, fetch_src_filter, filter_mode);
            break;
        default: {
            static std::atomic<int> s_plan_gu_unsupported{0};
            if (s_plan_gu_unsupported.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::fprintf(stderr, "[CUDA ERROR] launch_glm5_moe_plan_filtered: unsupported gu_type=%d\n", gu_type);
            }
            break;
        }
    }

    const int n_quads = (n_embd + 3) / 4;
    dim3 grid_d(n_quads, M);
    dim3 block_d(32, 8);
    switch (d_type) {
        case GGML_TYPE_IQ3_XXS:
            glm5_moe_down_plan_kernel<GGML_TYPE_IQ3_XXS><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_IQ3_S:
            glm5_moe_down_plan_kernel<GGML_TYPE_IQ3_S><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q4_K:
            glm5_moe_down_plan_kernel<GGML_TYPE_Q4_K><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q5_K:
            glm5_moe_down_plan_kernel<GGML_TYPE_Q5_K><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_IQ4_XS:
            glm5_moe_down_plan_kernel<GGML_TYPE_IQ4_XS><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q8_0:
            glm5_moe_down_plan_kernel<GGML_TYPE_Q8_0><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q5_1:
            glm5_moe_down_plan_kernel<GGML_TYPE_Q5_1><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q5_0:
            glm5_moe_down_plan_kernel<GGML_TYPE_Q5_0><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_IQ4_NL:
            glm5_moe_down_plan_kernel<GGML_TYPE_IQ4_NL><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q2_0:
            glm5_moe_down_plan_kernel<GGML_TYPE_Q2_0><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_IQ2_S:
            glm5_moe_down_plan_kernel<GGML_TYPE_IQ2_S><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        case GGML_TYPE_Q6_K:
            glm5_moe_down_plan_kernel<GGML_TYPE_Q6_K><<<grid_d, block_d, 0, stream>>>(
                plan_ptrs, d_off, d_row_bytes, iq8, router_w, sh_out_opt, out_f32, n_embd, n_ff, topk,
                skip_dev, fetch_src_filter, filter_mode);
            break;
        default: {
            static std::atomic<int> s_plan_d_unsupported{0};
            if (s_plan_d_unsupported.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::fprintf(stderr, "[CUDA ERROR] launch_glm5_moe_plan_filtered: unsupported d_type=%d\n", d_type);
            }
            break;
        }
    }
}

extern "C" void launch_glm5_moe_plan(
    int32_t gu_type,
    int32_t d_type,
    const uint64_t* plan_ptrs,
    uint32_t u_off,
    uint32_t d_off,
    size_t gu_row_bytes,
    size_t d_row_bytes,
    const void* x_q8,
    const float* router_w,
    void* inter_q8,
    int32_t sh_down_type,
    const void* sh_down_w,
    size_t sh_down_row_bytes,
    const void* sh_hq,
    int n_ff_sh,
    float* sh_out_opt,
    float* out_f32,
    int n_embd,
    int n_ff,
    int topk,
    int M,
    float clamp_lim,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    launch_glm5_moe_plan_filtered(
        gu_type, d_type, plan_ptrs, u_off, d_off, gu_row_bytes, d_row_bytes, x_q8, router_w,
        inter_q8, sh_down_type, sh_down_w, sh_down_row_bytes, sh_hq, n_ff_sh, sh_out_opt,
        out_f32, n_embd, n_ff, topk, M, clamp_lim, skip_dev, nullptr, 0, stream
    );
}

extern "C" void launch_glm5_moe_resident(
    int32_t gu_type,
    int32_t d_type,
    const uint64_t* gate_ptrs,
    const uint64_t* up_ptrs,
    const uint64_t* down_ptrs,
    size_t gu_row_bytes,
    size_t d_row_bytes,
    const void* x_q8,
    const int32_t* expert_ids,
    const float* router_w,
    void* inter_q8,
    float* inter_f32_opt,
    float* out_f32,
    int n_embd,
    int n_ff,
    int topk,
    int M,
    float clamp_lim,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (M <= 0 || n_ff <= 0 || n_embd <= 0) return;
    dim3 grid_gu((n_ff + 31) / 32, topk, M);
    dim3 block_gu(32, 8);
    const auto* xq = reinterpret_cast<const block_q8_1*>(x_q8);
    auto* iq8 = reinterpret_cast<block_q8_1*>(inter_q8);

    switch (gu_type) {
        case GGML_TYPE_IQ2_S:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_IQ2_S><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_Q3_K:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_Q3_K><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_Q6_K:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_Q6_K><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_Q4_K:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_Q4_K><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_Q5_K:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_Q5_K><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_Q8_0:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_Q8_0><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_Q5_1:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_Q5_1><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_Q2_0:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_Q2_0><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_IQ4_NL:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_IQ4_NL><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_Q5_0:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_Q5_0><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_IQ4_XS:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_IQ4_XS><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_IQ3_S:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_IQ3_S><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        case GGML_TYPE_IQ3_XXS:
            glm5_moe_gate_up_swiglu_kernel<GGML_TYPE_IQ3_XXS><<<grid_gu, block_gu, 0, stream>>>(
                gate_ptrs, up_ptrs, gu_row_bytes, xq, expert_ids, iq8, inter_f32_opt, n_embd, n_ff, topk, clamp_lim, skip_dev);
            break;
        default: {
            static std::atomic<int> s_res_gu_unsupported{0};
            if (s_res_gu_unsupported.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::fprintf(stderr, "[CUDA ERROR] launch_glm5_moe_resident: unsupported gu_type=%d\n", gu_type);
            }
            break;
        }
    }

    dim3 grid_d((n_embd + 3) / 4, M);
    dim3 block_d(32, 4);
    switch (d_type) {
        case GGML_TYPE_IQ3_XXS:
            glm5_moe_down_reduce_kernel<GGML_TYPE_IQ3_XXS><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_IQ3_S:
            glm5_moe_down_reduce_kernel<GGML_TYPE_IQ3_S><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_Q4_K:
            glm5_moe_down_reduce_kernel<GGML_TYPE_Q4_K><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_Q5_K:
            glm5_moe_down_reduce_kernel<GGML_TYPE_Q5_K><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_Q8_0:
            glm5_moe_down_reduce_kernel<GGML_TYPE_Q8_0><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_Q5_1:
            glm5_moe_down_reduce_kernel<GGML_TYPE_Q5_1><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_Q5_0:
            glm5_moe_down_reduce_kernel<GGML_TYPE_Q5_0><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_IQ4_NL:
            glm5_moe_down_reduce_kernel<GGML_TYPE_IQ4_NL><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_IQ4_XS:
            glm5_moe_down_reduce_kernel<GGML_TYPE_IQ4_XS><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_Q2_0:
            glm5_moe_down_reduce_kernel<GGML_TYPE_Q2_0><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_IQ2_S:
            glm5_moe_down_reduce_kernel<GGML_TYPE_IQ2_S><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        case GGML_TYPE_Q6_K:
            glm5_moe_down_reduce_kernel<GGML_TYPE_Q6_K><<<grid_d, block_d, 0, stream>>>(
                down_ptrs, d_row_bytes, iq8, expert_ids, router_w, out_f32, n_embd, n_ff, topk, skip_dev);
            break;
        default: {
            static std::atomic<int> s_res_d_unsupported{0};
            if (s_res_d_unsupported.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::fprintf(stderr, "[CUDA ERROR] launch_glm5_moe_resident: unsupported d_type=%d\n", d_type);
            }
            break;
        }
    }
}

// Compute Gate + Up + SiTU for all resident selected experts (slot ptr != 0).
// Grid: (ceil(n_ff / 4), topk, M), Block: (32, 4)
// inter_q8_1: quantized [M, topk, n_ff] output in block_q8_1 OR inter_f32 [M, topk, n_ff].
// Writing inter_f32 [M, topk, n_ff] keeps exact F32 before quantizing for down_proj.
__global__ void moe_gate_up_situ_kernel(
    const uint64_t* __restrict__ gate_up_ptrs, // [n_expert], 0 if not on this GPU
    const float* __restrict__ gate_up_scales,  // nullable [n_expert]
    size_t row_bytes_k,
    int32_t w_type,
    const block_q8_1* __restrict__ x_q8,       // [M, K/32]
    const int32_t* __restrict__ expert_ids,    // [M, topk]
    float* __restrict__ inter_f32,             // [M, topk, n_ff]
    int n_embd,
    int n_ff,
    int topk,
    float situ_beta,
    float situ_linear_beta
) {
    constexpr int MAX_STAGED_NB = 256;
    __shared__ __align__(16) uint32_t s_x_words[MAX_STAGED_NB * 9];

    int r = blockIdx.x * blockDim.y + threadIdx.y;
    int k_slot = blockIdx.y;
    int m = blockIdx.z;
    if (k_slot >= topk) return;

    int e = expert_ids[(size_t)m * topk + k_slot];
    uint64_t base_ptr = (e >= 0) ? gate_up_ptrs[e] : 0ULL;
    if (base_ptr == 0ULL) return;

    int lane = threadIdx.x;
    int nb_x = n_embd / 32;
    const block_q8_1* xm = x_q8 + (size_t)m * nb_x;
    if (nb_x <= MAX_STAGED_NB) {
        int total_words = nb_x * 9;
        const uint32_t* gx = reinterpret_cast<const uint32_t*>(xm);
        int tid = threadIdx.y * 32 + lane;
        int nthreads = blockDim.y * 32;
        for (int i = tid; i < total_words; i += nthreads) {
            s_x_words[i] = gx[i];
        }
        __syncthreads();
        xm = reinterpret_cast<const block_q8_1*>(s_x_words);
    }

    if (r >= n_ff) return;

    const uint8_t* w_base = reinterpret_cast<const uint8_t*>(base_ptr);

    const uint8_t* w_gate_row = w_base + (size_t)r * row_bytes_k;
    const uint8_t* w_up_row   = w_base + (size_t)(n_ff + r) * row_bytes_k;

    float g = 0.0f;
    float u = 0.0f;
    dispatch_dot_pair(w_type, w_gate_row, w_up_row, xm, n_embd, lane, g, u);

    if (lane == 0) {
        if (gate_up_scales != nullptr) {
            float sc = gate_up_scales[e];
            g *= sc;
            u *= sc;
        }
        // SiTU(g, u) = (beta * tanh(g / beta)) * sigmoid(g) * (lb * tanh(u / lb))
        float g_cap = situ_beta * tanhf(g / situ_beta);
        float sig = 1.0f / (1.0f + expf(-g));
        float u_cap = (situ_linear_beta > 0.0f) ? (situ_linear_beta * tanhf(u / situ_linear_beta)) : u;
        inter_f32[((size_t)m * topk + k_slot) * n_ff + r] = g_cap * sig * u_cap;
    }
}

// Quantize resident slots of inter_f32 [M, topk, n_ff] -> inter_q8_1 [M, topk, n_ff/32]
__global__ void moe_quantize_inter_kernel(
    const uint64_t* __restrict__ down_ptrs,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ inter_f32,
    block_q8_1* __restrict__ inter_q8,
    int n_ff,
    int topk
) {
    int blk = blockIdx.x * blockDim.y + threadIdx.y;
    int k_slot = blockIdx.y;
    int m = blockIdx.z;
    int nb = n_ff / 32;
    if (blk >= nb || k_slot >= topk) return;

    int e = expert_ids[(size_t)m * topk + k_slot];
    if (e < 0 || down_ptrs[e] == 0ULL) return;

    int lane = threadIdx.x;
    size_t slot_idx = (size_t)m * topk + k_slot;
    float val = inter_f32[slot_idx * n_ff + blk * 32 + lane];
    float amax = warp_reduce_max(fabsf(val));
    float d = amax / 127.0f;
    int q = (d > 0.0f) ? __float2int_rn(val / d) : 0;
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    int sum_q = q;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum_q += __shfl_xor_sync(0xffffffff, sum_q, off);
    }
    block_q8_1* out = inter_q8 + slot_idx * nb + blk;
    out->qs[lane] = (int8_t)q;
    if (lane == 0) {
        out->d = f32_to_fp16_bits(d);
        out->s = f32_to_fp16_bits(d * (float)sum_q);
    }
}

// Compute Down projection for resident slots into per-slot scratch [M, topk, n_embd] (weighted!).
// Each warp computes TWO adjacent output rows (2*r_pair and 2*r_pair + 1) simultaneously to halve shared-memory X loads!
// Non-resident slots zero their scratch row so final reduction is clean and deterministic.
__global__ void moe_down_slots_kernel(
    const uint64_t* __restrict__ down_ptrs,
    const float* __restrict__ down_scales, // nullable [n_expert]
    size_t row_bytes_ff,
    int32_t w_type,
    const block_q8_1* __restrict__ inter_q8,
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ expert_weights,
    float* __restrict__ slot_out, // [M, topk, n_embd]
    int n_embd,
    int n_ff,
    int topk
) {
    constexpr int MAX_STAGED_NB = 128;
    __shared__ __align__(16) uint32_t s_x_words[MAX_STAGED_NB * 9];

    int r_pair = blockIdx.x * blockDim.y + threadIdx.y;
    int r0 = r_pair * 2;
    int r1 = r0 + 1;
    int k_slot = blockIdx.y;
    int m = blockIdx.z;
    if (k_slot >= topk) return;

    int lane = threadIdx.x;
    size_t slot_idx = (size_t)m * topk + k_slot;
    int e = expert_ids[slot_idx];
    uint64_t base_ptr = (e >= 0) ? down_ptrs[e] : 0ULL;

    if (base_ptr == 0ULL) {
        if (lane == 0) {
            if (r0 < n_embd) slot_out[slot_idx * n_embd + r0] = 0.0f;
            if (r1 < n_embd) slot_out[slot_idx * n_embd + r1] = 0.0f;
        }
        return;
    }

    int nb_ff = n_ff / 32;
    const block_q8_1* xm = inter_q8 + slot_idx * nb_ff;
    if (nb_ff <= MAX_STAGED_NB) {
        int total_words = nb_ff * 9;
        const uint32_t* gx = reinterpret_cast<const uint32_t*>(xm);
        int tid = threadIdx.y * 32 + lane;
        int nthreads = blockDim.y * 32;
        for (int i = tid; i < total_words; i += nthreads) {
            s_x_words[i] = gx[i];
        }
        __syncthreads();
        xm = reinterpret_cast<const block_q8_1*>(s_x_words);
    }

    if (r0 >= n_embd) return;

    const uint8_t* w_row0 = reinterpret_cast<const uint8_t*>(base_ptr) + (size_t)r0 * row_bytes_ff;
    float sc = ((down_scales != nullptr) ? down_scales[e] : 1.0f) * expert_weights[slot_idx];

    if (r1 < n_embd) {
        const uint8_t* w_row1 = w_row0 + row_bytes_ff;
        float val0 = 0.0f, val1 = 0.0f;
        dispatch_dot_pair(w_type, w_row0, w_row1, xm, n_ff, lane, val0, val1);
        if (lane == 0) {
            slot_out[slot_idx * n_embd + r0] = val0 * sc;
            slot_out[slot_idx * n_embd + r1] = val1 * sc;
        }
    } else {
        float val0 = dispatch_dot_row(w_type, w_row0, xm, n_ff, lane);
        if (lane == 0) {
            slot_out[slot_idx * n_embd + r0] = val0 * sc;
        }
    }
}

// Deterministically reduce [M, topk, n_embd] -> [M, n_embd]
__global__ void moe_reduce_slots_kernel(
    const float* __restrict__ slot_out,
    float* __restrict__ out,
    int n_embd,
    int topk,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * n_embd;
    if (idx >= total) return;
    int m = idx / n_embd;
    int r = idx % n_embd;
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        sum += slot_out[((size_t)m * topk + k) * n_embd + r];
    }
    out[idx] = sum;
}

// Fused MoE slot reduction + ffn_post_conv short convolution + residual add(ffn_inp) for M=1!
__global__ void moe_reduce_conv_add_m1_kernel(
    const float* __restrict__ slot_out,
    const float* __restrict__ static_w,
    float* __restrict__ history,
    const float* __restrict__ ffn_inp,
    float* __restrict__ cur_out,
    int n_embd,
    int topk,
    int conv_offset,
    int n_conv
) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_embd) return;
    float xt = 0.0f;
    for (int k = 0; k < topk; ++k) {
        xt += slot_out[(size_t)k * n_embd + r];
    }
    float* hist_ch = history + (size_t)(conv_offset + r) * 3;
    float h0 = hist_ch[0];
    float h1 = hist_ch[1];
    float h2 = hist_ch[2];
    const float* sw = static_w + (size_t)r * 4;
    float conv_val = h0 * sw[0] + h1 * sw[1] + h2 * sw[2] + xt * sw[3];
    float inv_scale = rsqrtf((float)(n_conv + 1));
    cur_out[r] = (xt + conv_val) * inv_scale + ffn_inp[r];
    hist_ch[0] = h1;
    hist_ch[1] = h2;
    hist_ch[2] = xt;
}

// Convert FP32 activation [n_embd] into two FP16 limbs [2, n_embd] (x = hi + lo) for exact NVFP4 Tensor-Core MMA!
__global__ void f32_to_f16_limbs_kernel(
    const float* __restrict__ x,
    uint16_t* __restrict__ x_limbs,
    int n_embd
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float v = x[i];
    uint16_t hi = f32_to_fp16_bits(v);
    uint16_t lo = f32_to_fp16_bits(v - fp16_bits_to_f32(hi));
    x_limbs[i] = hi;
    x_limbs[n_embd + i] = lo;
}

// Tensor-Core mma.sync.m16n8k16 NVFP4 MoE kernel for M=1!
// When Gate == true:
//   Grid: (n_ff / 8, topk, 1), Block: SPLIT * 32 (SPLIT=8 -> 256 threads)
//   Computes 8 Gate rows and 8 Up rows simultaneously across SPLIT=8 warps over K=n_embd,
//   applies SiTU, and writes FP16 hidden directly to `inter_f16` [topk, n_ff]!
// When Gate == false:
//   Grid: (n_embd / 16, topk, 1), Block: SPLIT * 32 (SPLIT=4 -> 128 threads)
//   Computes 16 Down rows simultaneously across SPLIT=4 warps over K=n_ff,
//   multiplies by `down_scales[e] * expert_weights[slot]`, and writes FP32 `slot_out` [topk, n_embd]!
template <bool Gate, int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
moe_nvfp4_mma_m1_kernel(
    const uint64_t* __restrict__ exp_ptrs,      // [n_expert]
    const float* __restrict__ exp_scales,       // nullable [n_expert]
    const int32_t* __restrict__ expert_ids,     // [topk]
    const float* __restrict__ expert_weights,   // [topk] (used when !Gate)
    const uint16_t* __restrict__ x16,           // [2, n_embd] when Gate, [topk, n_ff] when !Gate
    uint16_t* __restrict__ inter_f16,           // [topk, n_ff] when Gate
    float* __restrict__ slot_out,               // [topk, n_embd] when !Gate
    int n_embd,
    int n_ff,
    float situ_beta,
    float situ_linear_beta
) {
    __shared__ float2 part[SPLIT][8];
    const int lane = threadIdx.x & 31;
    const int split = threadIdx.x >> 5;
    const int g = lane >> 2; // 0..7
    const int t = lane & 3;  // 0..3
    const int slot = blockIdx.y;

    const int e = expert_ids[slot];
    const uint64_t base_ptr = (e >= 0) ? exp_ptrs[e] : 0ULL;
    const int r = blockIdx.x * (Gate ? 8 : 16) + g;

    if (base_ptr == 0ULL) {
        if (!Gate && split == 0 && t == 0) {
            slot_out[(size_t)slot * n_embd + r] = 0.0f;
            slot_out[(size_t)slot * n_embd + r + 8] = 0.0f;
        }
        return;
    }

    const int K = Gate ? n_embd : n_ff;
    const size_t rb = (size_t)(K / 64) * 36;
    const size_t up_off = (size_t)n_ff * rb;
    const uint8_t* w_base = reinterpret_cast<const uint8_t*>(base_ptr);
    const uint8_t* a = w_base + (size_t)r * rb;
    const uint8_t* b = Gate ? (a + up_off) : (a + 8 * rb);

    const uint4* xp = reinterpret_cast<const uint4*>(
        x16 + (Gate ? (g == 1 ? (size_t)n_embd : 0) : (size_t)slot * n_ff)
    ) + 2 * t;

    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = K / 64;
    NvMmaStep cur = nv_mma_step_load(a, b, split < nk ? split : 0, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvMmaStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_mma_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu);
        const uint32_t sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1};
        const uint32_t wb[2] = {cur.wb0, cur.wb1};
        const uint4 x0 = __ldg(xp + k * 8);
        const uint4 x1 = __ldg(xp + k * 8 + 1);
        const uint32_t xx[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1));
            const uint32_t qb = wb[m >> 1] >> (16 * (m & 1));
            const uint32_t af[4] = {
                nv_pair_h2(qa, sa),
                nv_pair_h2(qb, sb),
                nv_pair_h2(qa >> 4, sa),
                nv_pair_h2(qb >> 4, sb)
            };
            mma16816_f16(v, af, xx[m], xx[4 + m]);
        }
        cur = nxt;
    }

    if (t == 0) {
        part[split][g] = Gate ? make_float2(v[0] + v[1], v[2] + v[3]) : make_float2(v[0], v[2]);
    }
    __syncthreads();
    if (split != 0 || t != 0) return;

    float2 tot = part[0][g];
    #pragma unroll
    for (int i = 1; i < SPLIT; ++i) {
        tot.x += part[i][g].x;
        tot.y += part[i][g].y;
    }

    const float sc = (exp_scales != nullptr) ? exp_scales[e] : 1.0f;
    if (Gate) {
        const float gg = tot.x * sc;
        const float uu = tot.y * sc;
        const float g_cap = situ_beta * tanhf(gg / situ_beta);
        const float sig = 1.0f / (1.0f + expf(-gg));
        const float u_cap = (situ_linear_beta > 0.0f) ? (situ_linear_beta * tanhf(uu / situ_linear_beta)) : uu;
        inter_f16[(size_t)slot * n_ff + r] = f32_to_fp16_bits(g_cap * sig * u_cap);
    } else {
        const float w_sc = sc * expert_weights[slot];
        slot_out[(size_t)slot * n_embd + r] = tot.x * w_sc;
        slot_out[(size_t)slot * n_embd + r + 8] = tot.y * w_sc;
    }
}

extern "C" {

void launch_moe_grouped_f32(
    const uint64_t* gate_up_ptrs,
    const uint64_t* down_ptrs,
    const float* gate_up_scales,
    const float* down_scales,
    size_t gate_up_row_bytes,
    size_t down_row_bytes,
    int32_t gate_up_type,
    int32_t down_type,
    const void* x_q8,
    const float* x_f32,
    const int32_t* expert_ids,
    const float* expert_weights,
    float* inter_f32,
    void* inter_q8,
    float* slot_out,
    float* out,
    int n_embd,
    int n_ff,
    int topk,
    int M,
    float situ_beta,
    float situ_linear_beta,
    cudaStream_t stream
) {
    if (M == 1 && gate_up_type == GGML_TYPE_NVFP4 && down_type == GGML_TYPE_NVFP4 &&
        x_f32 != nullptr && (n_embd % 64 == 0) && (n_ff % 64 == 0)) {
        uint16_t* x_limbs = reinterpret_cast<uint16_t*>(inter_f32);
        uint16_t* inter_f16 = reinterpret_cast<uint16_t*>(inter_q8);

        dim3 g_gu(n_ff / 8, topk, 1);
        moe_nvfp4_mma_m1_kernel<true, 8><<<g_gu, 8 * 32, 0, stream>>>(
            gate_up_ptrs,
            gate_up_scales,
            expert_ids,
            expert_weights,
            x_limbs,
            inter_f16,
            nullptr,
            n_embd,
            n_ff,
            situ_beta,
            situ_linear_beta
        );

        dim3 g_dn(n_embd / 16, topk, 1);
        moe_nvfp4_mma_m1_kernel<false, 4><<<g_dn, 4 * 32, 0, stream>>>(
            down_ptrs,
            down_scales,
            expert_ids,
            expert_weights,
            inter_f16,
            nullptr,
            slot_out,
            n_embd,
            n_ff,
            situ_beta,
            situ_linear_beta
        );

        int threads = 256;
        if (out != nullptr) {
            moe_reduce_slots_kernel<<<(n_embd + threads - 1) / threads, threads, 0, stream>>>(
                slot_out, out, n_embd, topk, 1
            );
        }
        return;
    }

    const int warps = 4;
    dim3 blk(32, warps);
    dim3 g_gu((n_ff + warps - 1) / warps, topk, M);
    moe_gate_up_situ_kernel<<<g_gu, blk, 0, stream>>>(
        gate_up_ptrs,
        gate_up_scales,
        gate_up_row_bytes,
        gate_up_type,
        reinterpret_cast<const block_q8_1*>(x_q8),
        expert_ids,
        inter_f32,
        n_embd,
        n_ff,
        topk,
        situ_beta,
        situ_linear_beta
    );

    int nb_ff = n_ff / 32;
    dim3 g_q((nb_ff + warps - 1) / warps, topk, M);
    moe_quantize_inter_kernel<<<g_q, blk, 0, stream>>>(
        down_ptrs,
        expert_ids,
        inter_f32,
        reinterpret_cast<block_q8_1*>(inter_q8),
        n_ff,
        topk
    );

    int n_down_pairs = (n_embd + 1) / 2;
    dim3 g_dn((n_down_pairs + warps - 1) / warps, topk, M);
    moe_down_slots_kernel<<<g_dn, blk, 0, stream>>>(
        down_ptrs,
        down_scales,
        down_row_bytes,
        down_type,
        reinterpret_cast<const block_q8_1*>(inter_q8),
        expert_ids,
        expert_weights,
        slot_out,
        n_embd,
        n_ff,
        topk
    );

    if (out != nullptr) {
        int total = M * n_embd;
        int threads = 256;
        moe_reduce_slots_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
            slot_out, out, n_embd, topk, M
        );
    }
}

void launch_moe_reduce_conv_add_m1_f32(
    const float* slot_out,
    const float* static_w,
    float* history,
    const float* ffn_inp,
    float* cur_out,
    int n_embd,
    int topk,
    int conv_offset,
    int n_conv,
    cudaStream_t stream
) {
    int threads = 256;
    moe_reduce_conv_add_m1_kernel<<<(n_embd + threads - 1) / threads, threads, 0, stream>>>(
        slot_out, static_w, history, ffn_inp, cur_out, n_embd, topk, conv_offset, n_conv
    );
}

} // extern "C"
