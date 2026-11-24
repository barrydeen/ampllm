#include "common/sm86_common.cuh"
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdio>

// -----------------------------------------------------------------------------
// Fast 8-lane-per-superblock M=1 GEMV and Multi-Job Batched MV
// -----------------------------------------------------------------------------

template <typename BlockT, float (*DotSubFn)(const BlockT*, const block_q8_1*, int, int)>
__global__ void __launch_bounds__(128) glm5_mmvq_sub256_kernel(
    const uint8_t* __restrict__ W,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y,
    int N,
    int K,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int lane = threadIdx.x;
    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= N) return;

    const size_t row_blocks = (size_t)(K >> 8);
    const BlockT* wr = reinterpret_cast<const BlockT*>(W) + (size_t)row * row_blocks;
    float sum = DotSubFn(wr, X_q8, K, lane);
    if (lane == 0) {
        Y[row] = sum;
    }
}

__device__ __forceinline__ float dot_row_q3_K_def(const block_q3_K* w, const block_q8_1* x, int K, int lane) {
    return dot_row_q3_K_sub(w, x, K, lane);
}
__device__ __forceinline__ float dot_row_q4_K_def(const block_q4_K* w, const block_q8_1* x, int K, int lane) {
    return dot_row_q4_K_sub(w, x, K, lane);
}
__device__ __forceinline__ float dot_row_q5_K_def(const block_q5_K* w, const block_q8_1* x, int K, int lane) {
    return dot_row_q5_K_sub(w, x, K, lane);
}
__device__ __forceinline__ float dot_row_iq4_xs_def(const block_iq4_xs* w, const block_q8_1* x, int K, int lane) {
    return dot_row_iq4_xs_sub(w, x, K, lane);
}
__device__ __forceinline__ float dot_row_q6_K_def(const block_q6_K* w, const block_q8_1* x, int K, int lane) {
    return dot_row_q6_K_sub(w, x, K, lane);
}
__device__ __forceinline__ float dot_row_iq2_s_def(const block_iq2_s* w, const block_q8_1* x, int K, int lane) {
    return dot_row_iq2_s_sub(w, x, K, lane, nullptr);
}
__device__ __forceinline__ float dot_row_iq3_xxs_def(const block_iq3_xxs* w, const block_q8_1* x, int K, int lane) {
    return dot_row_iq3_xxs_sub(w, x, K, lane, nullptr);
}
__device__ __forceinline__ float dot_row_iq3_s_def(const block_iq3_s* w, const block_q8_1* x, int K, int lane) {
    return dot_row_iq3_s_sub(w, x, K, lane, nullptr);
}
__device__ __forceinline__ void dot_pair_q3_K_def(const block_q3_K* w0, const block_q3_K* w1, const block_q8_1* x, int K, int lane, float& o0, float& o1) {
    dot_pair_q3_K_sub(w0, w1, x, K, lane, o0, o1);
}
__device__ __forceinline__ void dot_pair_q4_K_def(const block_q4_K* w0, const block_q4_K* w1, const block_q8_1* x, int K, int lane, float& o0, float& o1) {
    dot_pair_q4_K_sub(w0, w1, x, K, lane, o0, o1);
}
__device__ __forceinline__ void dot_pair_q5_K_def(const block_q5_K* w0, const block_q5_K* w1, const block_q8_1* x, int K, int lane, float& o0, float& o1) {
    dot_pair_q5_K_sub(w0, w1, x, K, lane, o0, o1);
}
__device__ __forceinline__ void dot_pair_iq4_xs_def(const block_iq4_xs* w0, const block_iq4_xs* w1, const block_q8_1* x, int K, int lane, float& o0, float& o1) {
    dot_pair_iq4_xs_sub(w0, w1, x, K, lane, o0, o1);
}
__device__ __forceinline__ void dot_pair_q6_K_def(const block_q6_K* w0, const block_q6_K* w1, const block_q8_1* x, int K, int lane, float& o0, float& o1) {
    dot_pair_q6_K_sub(w0, w1, x, K, lane, o0, o1);
}

template <typename BlockT, void (*DotPairFn)(const BlockT*, const BlockT*, const block_q8_1*, int, int, float&, float&)>
__global__ void __launch_bounds__(128) glm5_mmvq_sub256_pair_kernel(
    const uint8_t* __restrict__ W,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y,
    int N,
    int K,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.x * 4 + threadIdx.y) * 2;
    if (row0 >= N) return;
    const int row1 = row0 + 1;
    const int row1_c = (row1 < N) ? row1 : row0;

    const size_t row_blocks = (size_t)(K >> 8);
    const BlockT* w0 = reinterpret_cast<const BlockT*>(W) + (size_t)row0 * row_blocks;
    const BlockT* w1 = reinterpret_cast<const BlockT*>(W) + (size_t)row1_c * row_blocks;
    float o0 = 0.0f, o1 = 0.0f;
    DotPairFn(w0, w1, X_q8, K, lane, o0, o1);
    if (lane == 0) {
        Y[row0] = o0;
        if (row1 < N) Y[row1] = o1;
    }
}

extern "C" void launch_glm5_mmvq_sub256_skip(
    int32_t w_type,
    const void* W,
    const void* X_q8,
    float* Y,
    int N,
    int K,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    dim3 block(32, 4);
    dim3 grid1((N + 3) / 4);
    dim3 grid2((N + 7) / 8);
    const auto* w8 = reinterpret_cast<const uint8_t*>(W);
    const auto* xq = reinterpret_cast<const block_q8_1*>(X_q8);
    switch (w_type) {
        case GGML_TYPE_Q3_K:    glm5_mmvq_sub256_pair_kernel<block_q3_K, dot_pair_q3_K_def><<<grid2, block, 0, stream>>>(w8, xq, Y, N, K, skip_dev); break;
        case GGML_TYPE_Q4_K:    glm5_mmvq_sub256_pair_kernel<block_q4_K, dot_pair_q4_K_def><<<grid2, block, 0, stream>>>(w8, xq, Y, N, K, skip_dev); break;
        case GGML_TYPE_Q5_K:    glm5_mmvq_sub256_pair_kernel<block_q5_K, dot_pair_q5_K_def><<<grid2, block, 0, stream>>>(w8, xq, Y, N, K, skip_dev); break;
        case GGML_TYPE_Q6_K:    glm5_mmvq_sub256_pair_kernel<block_q6_K, dot_pair_q6_K_def><<<grid2, block, 0, stream>>>(w8, xq, Y, N, K, skip_dev); break;
        case GGML_TYPE_IQ4_XS:  glm5_mmvq_sub256_pair_kernel<block_iq4_xs, dot_pair_iq4_xs_def><<<grid2, block, 0, stream>>>(w8, xq, Y, N, K, skip_dev); break;
        case GGML_TYPE_IQ2_S:   glm5_mmvq_sub256_kernel<block_iq2_s, dot_row_iq2_s_def><<<grid1, block, 0, stream>>>(w8, xq, Y, N, K, skip_dev); break;
        case GGML_TYPE_IQ3_XXS: glm5_mmvq_sub256_kernel<block_iq3_xxs, dot_row_iq3_xxs_def><<<grid1, block, 0, stream>>>(w8, xq, Y, N, K, skip_dev); break;
        case GGML_TYPE_IQ3_S:   glm5_mmvq_sub256_kernel<block_iq3_s, dot_row_iq3_s_def><<<grid1, block, 0, stream>>>(w8, xq, Y, N, K, skip_dev); break;
        default: break;
    }
}

extern "C" void launch_glm5_mmvq_sub256(
    int32_t w_type,
    const void* W,
    const void* X_q8,
    float* Y,
    int N,
    int K,
    cudaStream_t stream
) {
    launch_glm5_mmvq_sub256_skip(w_type, W, X_q8, Y, N, K, nullptr, stream);
}

struct Glm5MvJob {
    const void* w;
    float* y;
    int32_t w_type;
    int32_t n_out;
    float scale;
    int32_t pad;
};

struct Glm5MvBatchArgs {
    Glm5MvJob jobs[6];
    int32_t blk_start[7];
    int32_t n_jobs;
    int32_t K;
    int32_t use_smem_x;
};

__global__ void __launch_bounds__(128) glm5_mv_batch_kernel(
    Glm5MvBatchArgs args,
    const block_q8_1* __restrict__ X_q8,
    const float* __restrict__ X_f32,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;

    __shared__ __align__(16) uint32_t s_xq8[128 * 9];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if (args.use_smem_x) {
        const int total_words = (args.K >> 5) * 9;
        const uint32_t* gx = reinterpret_cast<const uint32_t*>(X_q8);
        for (int i = tid; i < total_words; i += 128) s_xq8[i] = gx[i];
        __syncthreads();
    }
    const block_q8_1* x_use = args.use_smem_x ? reinterpret_cast<const block_q8_1*>(s_xq8) : X_q8;

    int job_idx = 0;
    #pragma unroll
    for (int j = 0; j < 6; ++j) {
        if (j < args.n_jobs && (int)blockIdx.x >= args.blk_start[j + 1]) {
            job_idx = j + 1;
        }
    }
    if (job_idx >= args.n_jobs) return;
    const Glm5MvJob job = args.jobs[job_idx];
    const int row0 = (((int)blockIdx.x - args.blk_start[job_idx]) * 4 + threadIdx.y) * 2;
    if (row0 >= job.n_out) return;
    const int row1 = row0 + 1;
    const int row1_c = (row1 < job.n_out) ? row1 : row0;
    const int lane = threadIdx.x;
    const int K = args.K;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    if (job.w_type == GGML_TYPE_F32) {
        const float* wr0 = reinterpret_cast<const float*>(job.w) + (size_t)row0 * K;
        const float* wr1 = reinterpret_cast<const float*>(job.w) + (size_t)row1_c * K;
        for (int c = lane; c < K; c += 32) {
            float xv = X_f32[c];
            sum0 += wr0[c] * xv;
            sum1 += wr1[c] * xv;
        }
        sum0 = warp_reduce_sum(sum0);
        sum1 = warp_reduce_sum(sum1);
    } else if (job.w_type == GGML_TYPE_BF16) {
        const uint16_t* wr0 = reinterpret_cast<const uint16_t*>(job.w) + (size_t)row0 * K;
        const uint16_t* wr1 = reinterpret_cast<const uint16_t*>(job.w) + (size_t)row1_c * K;
        for (int c = lane; c < K; c += 32) {
            float xv = X_f32[c];
            sum0 += bf16_bits_to_f32(wr0[c]) * xv;
            sum1 += bf16_bits_to_f32(wr1[c]) * xv;
        }
        sum0 = warp_reduce_sum(sum0);
        sum1 = warp_reduce_sum(sum1);
    } else if (job.w_type == GGML_TYPE_F16) {
        const uint16_t* wr0 = reinterpret_cast<const uint16_t*>(job.w) + (size_t)row0 * K;
        const uint16_t* wr1 = reinterpret_cast<const uint16_t*>(job.w) + (size_t)row1_c * K;
        for (int c = lane; c < K; c += 32) {
            float xv = X_f32[c];
            sum0 += fp16_bits_to_f32(wr0[c]) * xv;
            sum1 += fp16_bits_to_f32(wr1[c]) * xv;
        }
        sum0 = warp_reduce_sum(sum0);
        sum1 = warp_reduce_sum(sum1);
    } else if (job.w_type == GGML_TYPE_Q8_0) {
        const block_q8_0* w = reinterpret_cast<const block_q8_0*>(job.w);
        const size_t rb = (size_t)(K >> 5);
        dot_pair_q8_0_sub(w + (size_t)row0 * rb, w + (size_t)row1_c * rb, x_use, K, lane, sum0, sum1);
    } else if (job.w_type == GGML_TYPE_Q6_K) {
        const block_q6_K* w = reinterpret_cast<const block_q6_K*>(job.w);
        const size_t rb = (size_t)(K >> 8);
        dot_pair_q6_K_sub(w + (size_t)row0 * rb, w + (size_t)row1_c * rb, x_use, K, lane, sum0, sum1);
    } else if (job.w_type == GGML_TYPE_Q4_K) {
        const block_q4_K* w = reinterpret_cast<const block_q4_K*>(job.w);
        const size_t rb = (size_t)(K >> 8);
        dot_pair_q4_K_sub(w + (size_t)row0 * rb, w + (size_t)row1_c * rb, x_use, K, lane, sum0, sum1);
    } else if (job.w_type == GGML_TYPE_Q3_K) {
        const block_q3_K* w = reinterpret_cast<const block_q3_K*>(job.w);
        const size_t rb = (size_t)(K >> 8);
        dot_pair_q3_K_sub(w + (size_t)row0 * rb, w + (size_t)row1_c * rb, x_use, K, lane, sum0, sum1);
    }
    if (lane == 0) {
        job.y[row0] = sum0 * job.scale;
        if (row1 < job.n_out) job.y[row1] = sum1 * job.scale;
    }
}

extern "C" void launch_glm5_mv_batch(
    const void* const* w_ptrs,
    float* const* y_ptrs,
    const int32_t* w_types,
    const int32_t* n_outs,
    const float* scales,
    int n_jobs,
    int K,
    const void* X_q8,
    const float* X_f32,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (n_jobs <= 0 || K <= 0) return;
    int off = 0;
    while (off < n_jobs) {
        int cnt = min(6, n_jobs - off);
        Glm5MvBatchArgs args{};
        args.n_jobs = cnt;
        args.K = K;
        bool any_q = false;
        int cur_blk = 0;
        args.blk_start[0] = 0;
        for (int i = 0; i < cnt; ++i) {
            args.jobs[i].w = w_ptrs[off + i];
            args.jobs[i].y = y_ptrs[off + i];
            args.jobs[i].w_type = w_types[off + i];
            args.jobs[i].n_out = n_outs[off + i];
            args.jobs[i].scale = scales ? scales[off + i] : 1.0f;
            int32_t wt = w_types[off + i];
            if (wt != GGML_TYPE_F32 && wt != GGML_TYPE_F16 && wt != GGML_TYPE_BF16) {
                any_q = true;
            }
            cur_blk += (n_outs[off + i] + 7) / 8;
            args.blk_start[i + 1] = cur_blk;
        }
        args.use_smem_x = (any_q && X_q8 != nullptr && K <= 4096 && (K & 31) == 0) ? 1 : 0;
        if (cur_blk > 0) {
            dim3 grid(cur_blk);
            dim3 block(32, 4);
            glm5_mv_batch_kernel<<<grid, block, 0, stream>>>(
                args,
                reinterpret_cast<const block_q8_1*>(X_q8),
                X_f32,
                skip_dev
            );
        }
        off += cnt;
    }
}

// -----------------------------------------------------------------------------
// Hyper-Connections & Stream Reduction / Norm Kernels
// -----------------------------------------------------------------------------

__global__ void glm5_embed_replicate_kernel(
    const float* __restrict__ emb, // [M, n_embd]
    float* __restrict__ R,         // [M, hc, n_embd]
    int n_embd,
    int hc,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = n_embd * hc;
    if (idx >= M * stride) return;
    int t = idx / stride;
    int rem = idx % stride;
    int e = rem % n_embd;
    R[idx] = emb[(size_t)t * n_embd + e];
}

extern "C" void launch_glm5_embed_replicate(
    const float* emb,
    float* R,
    int n_embd,
    int hc,
    int M,
    cudaStream_t stream
) {
    int total = M * hc * n_embd;
    if (total <= 0) return;
    glm5_embed_replicate_kernel<<<(total + 255) / 256, 256, 0, stream>>>(emb, R, n_embd, hc, M);
}

__global__ void __launch_bounds__(256) glm5_hc_pre_norm_q8_kernel(
    const float* __restrict__ R,       // [M, hc, n_embd]
    const void* __restrict__ w_fn,     // [hc_mix_dim, hc * n_embd] (F32 or BF16)
    int32_t w_fn_type,
    const float* __restrict__ w_scale, // [3]
    const float* __restrict__ w_base,  // [hc_mix_dim]
    const float* __restrict__ norm_w,  // [n_embd]
    int n_embd,
    int hc,
    int iters,
    float norm_eps,
    float hc_eps,
    float* __restrict__ post_out,      // [M, hc]
    float* __restrict__ comb_out,      // [M, hc * hc]
    float* __restrict__ x_f32_out,     // [M, n_embd]
    block_q8_1* __restrict__ x_q8_out  // nullable [M, n_embd / 32]
) {
    constexpr int MAX_HC = 8;
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int hc_dim = n_embd * hc;
    const int hc_mix_dim = (2 + hc) * hc;
    const float* Rt = R + (size_t)t * hc_dim;

    __shared__ float s_red[8];
    __shared__ float s_mix[2 * MAX_HC * MAX_HC];
    __shared__ float s_pre[MAX_HC];
    __shared__ float s_inv;

    // 1. Joint RMS over [hc, n_embd]
    float ss = 0.0f;
    if ((hc_dim & 3) == 0) {
        const float4* R4 = reinterpret_cast<const float4*>(Rt);
        const int n4 = hc_dim >> 2;
        for (int i = tid; i < n4; i += 256) {
            float4 v = R4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int i = tid; i < hc_dim; i += 256) {
            float v = Rt[i];
            ss += v * v;
        }
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)hc_dim + norm_eps);
    }
    __syncthreads();
    const float inv = s_inv;

    // 2. Warp-per-row 4-way ILP matvec s_mix[0..hc_mix_dim)
    for (int m = warp; m < hc_mix_dim; m += 8) {
        float acc = 0.0f;
        if (w_fn_type == GGML_TYPE_F32 && (hc_dim & 3) == 0) {
            const float4* R4 = reinterpret_cast<const float4*>(Rt);
            const float4* W4 = reinterpret_cast<const float4*>(reinterpret_cast<const float*>(w_fn) + (size_t)m * hc_dim);
            const int n4 = hc_dim >> 2;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
            int k = lane;
            for (; k + 96 < n4; k += 128) {
                float4 r0 = R4[k],      v0 = W4[k];
                float4 r1 = R4[k + 32], v1 = W4[k + 32];
                float4 r2 = R4[k + 64], v2 = W4[k + 64];
                float4 r3 = R4[k + 96], v3 = W4[k + 96];
                a0 += (r0.x * inv) * v0.x + (r0.y * inv) * v0.y + (r0.z * inv) * v0.z + (r0.w * inv) * v0.w;
                a1 += (r1.x * inv) * v1.x + (r1.y * inv) * v1.y + (r1.z * inv) * v1.z + (r1.w * inv) * v1.w;
                a2 += (r2.x * inv) * v2.x + (r2.y * inv) * v2.y + (r2.z * inv) * v2.z + (r2.w * inv) * v2.w;
                a3 += (r3.x * inv) * v3.x + (r3.y * inv) * v3.y + (r3.z * inv) * v3.z + (r3.w * inv) * v3.w;
            }
            for (; k < n4; k += 32) {
                float4 r = R4[k], v = W4[k];
                a0 += (r.x * inv) * v.x + (r.y * inv) * v.y + (r.z * inv) * v.z + (r.w * inv) * v.w;
            }
            acc = (a0 + a1) + (a2 + a3);
        } else if (w_fn_type == GGML_TYPE_BF16 && (hc_dim & 3) == 0) {
            const float4* R4 = reinterpret_cast<const float4*>(Rt);
            const uint2* W4 = reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(w_fn) + (size_t)m * hc_dim);
            const int n4 = hc_dim >> 2;
            float a0 = 0.0f, a1 = 0.0f;
            int k = lane;
            for (; k + 32 < n4; k += 64) {
                float4 r0 = R4[k];      uint2 u0 = W4[k];
                float4 r1 = R4[k + 32]; uint2 u1 = W4[k + 32];
                a0 += (r0.x * inv) * bf16_bits_to_f32((uint16_t)u0.x)
                    + (r0.y * inv) * bf16_bits_to_f32((uint16_t)(u0.x >> 16))
                    + (r0.z * inv) * bf16_bits_to_f32((uint16_t)u0.y)
                    + (r0.w * inv) * bf16_bits_to_f32((uint16_t)(u0.y >> 16));
                a1 += (r1.x * inv) * bf16_bits_to_f32((uint16_t)u1.x)
                    + (r1.y * inv) * bf16_bits_to_f32((uint16_t)(u1.x >> 16))
                    + (r1.z * inv) * bf16_bits_to_f32((uint16_t)u1.y)
                    + (r1.w * inv) * bf16_bits_to_f32((uint16_t)(u1.y >> 16));
            }
            for (; k < n4; k += 32) {
                float4 r = R4[k]; uint2 u = W4[k];
                a0 += (r.x * inv) * bf16_bits_to_f32((uint16_t)u.x)
                    + (r.y * inv) * bf16_bits_to_f32((uint16_t)(u.x >> 16))
                    + (r.z * inv) * bf16_bits_to_f32((uint16_t)u.y)
                    + (r.w * inv) * bf16_bits_to_f32((uint16_t)(u.y >> 16));
            }
            acc = a0 + a1;
        } else {
            const float* wrow = reinterpret_cast<const float*>(w_fn) + (size_t)m * hc_dim;
            for (int k = lane; k < hc_dim; k += 32) {
                float wv = (w_fn_type == GGML_TYPE_BF16)
                    ? bf16_bits_to_f32(reinterpret_cast<const uint16_t*>(w_fn)[(size_t)m * hc_dim + k])
                    : wrow[k];
                acc += Rt[k] * inv * wv;
            }
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) s_mix[m] = acc;
    }
    __syncthreads();

    // 3. Thread 0 runs exact gates + 20-iter Sinkhorn
    if (tid == 0) {
        float* pt = post_out + (size_t)t * hc;
        float* cb = comb_out + (size_t)t * (hc * hc);
        for (int s = 0; s < hc; ++s) {
            float v = s_mix[s] * w_scale[0] + w_base[s];
            s_pre[s] = 1.0f / (1.0f + expf(-v)) + hc_eps;
        }
        for (int s = 0; s < hc; ++s) {
            float v = s_mix[hc + s] * w_scale[1] + w_base[hc + s];
            pt[s] = 2.0f / (1.0f + expf(-v));
        }
        for (int d = 0; d < hc; ++d) {
            for (int s = 0; s < hc; ++s) {
                cb[d * hc + s] = s_mix[2 * hc + d + hc * s] * w_scale[2] + w_base[2 * hc + d + hc * s];
            }
        }
        float col_max[MAX_HC];
        for (int s = 0; s < hc; ++s) {
            float mx = -INFINITY;
            for (int d = 0; d < hc; ++d) mx = fmaxf(mx, cb[d * hc + s]);
            col_max[s] = mx;
        }
        float denom[MAX_HC];
        for (int s = 0; s < hc; ++s) {
            float e = 0.0f;
            for (int d = 0; d < hc; ++d) e += expf(cb[d * hc + s] - col_max[s]);
            denom[s] = e;
        }
        for (int d = 0; d < hc; ++d) {
            for (int s = 0; s < hc; ++s) {
                cb[d * hc + s] = expf(cb[d * hc + s] - col_max[s]) / denom[s] + hc_eps;
            }
        }
        auto norm_dst = [&]() {
            for (int d = 0; d < hc; ++d) {
                float sum = hc_eps;
                for (int s = 0; s < hc; ++s) sum += cb[d * hc + s];
                for (int s = 0; s < hc; ++s) cb[d * hc + s] /= sum;
            }
        };
        auto norm_src = [&]() {
            for (int s = 0; s < hc; ++s) {
                float sum = hc_eps;
                for (int d = 0; d < hc; ++d) sum += cb[d * hc + s];
                for (int d = 0; d < hc; ++d) cb[d * hc + s] /= sum;
            }
        };
        norm_dst();
        for (int it = 1; it < iters; ++it) {
            norm_src();
            norm_dst();
        }
    }
    __syncthreads();

    // 4. Compute mixed[e] = sum_s s_pre[s] * R[s, e], RMSNorm + optional q8_1 store
    float mss = 0.0f;
    for (int e = tid; e < n_embd; e += 256) {
        float acc = 0.0f;
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * Rt[(size_t)s * n_embd + e];
        mss += acc * acc;
    }
    mss = warp_reduce_sum(mss);
    if (lane == 0) s_red[warp] = mss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)n_embd + norm_eps);
    }
    __syncthreads();
    const float inv_n = s_inv;

    float* xf = x_f32_out + (size_t)t * n_embd;
    block_q8_1* xq = x_q8_out ? (x_q8_out + (size_t)t * (n_embd >> 5)) : nullptr;
    for (int base = 0; base < n_embd; base += 256) {
        int e = base + tid;
        float val = 0.0f;
        if (e < n_embd) {
            float acc = 0.0f;
            for (int s = 0; s < hc; ++s) acc += s_pre[s] * Rt[(size_t)s * n_embd + e];
            val = acc * inv_n * norm_w[e];
            xf[e] = val;
        }
        if (xq != nullptr && (base + warp * 32) < n_embd) {
            warp_quantize_q8_1_store(val, xq + ((base >> 5) + warp), lane);
        }
    }
}

__global__ void __launch_bounds__(256) glm5_hc_matvec_m1_kernel(
    const float* __restrict__ R,       // [hc * n_embd]
    const void* __restrict__ w_fn,     // [hc_mix_dim, hc * n_embd]
    int32_t w_fn_type,
    int hc_dim,
    int hc_mix_dim,
    float* __restrict__ tmp_out        // [hc_mix_dim + 1]
) {
    const int m = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_red[8];

    float acc = 0.0f;
    if (m < hc_mix_dim) {
        const bool aligned16 = (((reinterpret_cast<uintptr_t>(R) | reinterpret_cast<uintptr_t>(w_fn)) & 15u) == 0u);
        if (w_fn_type == GGML_TYPE_F32 && (hc_dim & 3) == 0 && aligned16) {
            const float4* R4 = reinterpret_cast<const float4*>(R);
            const float4* W4 = reinterpret_cast<const float4*>(
                reinterpret_cast<const float*>(w_fn) + (size_t)m * hc_dim
            );
            const int n4 = hc_dim >> 2;
            for (int k = tid; k < n4; k += 256) {
                float4 r = R4[k];
                float4 w = W4[k];
                acc += r.x * w.x + r.y * w.y + r.z * w.z + r.w * w.w;
            }
        } else if (w_fn_type == GGML_TYPE_BF16 && (hc_dim & 3) == 0 && aligned16) {
            const float4* R4 = reinterpret_cast<const float4*>(R);
            const uint2* W4 = reinterpret_cast<const uint2*>(
                reinterpret_cast<const uint16_t*>(w_fn) + (size_t)m * hc_dim
            );
            const int n4 = hc_dim >> 2;
            for (int k = tid; k < n4; k += 256) {
                float4 r = R4[k];
                uint2 u = W4[k];
                acc += r.x * bf16_bits_to_f32((uint16_t)u.x)
                     + r.y * bf16_bits_to_f32((uint16_t)(u.x >> 16))
                     + r.z * bf16_bits_to_f32((uint16_t)u.y)
                     + r.w * bf16_bits_to_f32((uint16_t)(u.y >> 16));
            }
        } else {
            const float* wrow = reinterpret_cast<const float*>(w_fn) + (size_t)m * hc_dim;
            for (int k = tid; k < hc_dim; k += 256) {
                float wv = (w_fn_type == GGML_TYPE_BF16)
                    ? bf16_bits_to_f32(reinterpret_cast<const uint16_t*>(w_fn)[(size_t)m * hc_dim + k])
                    : wrow[k];
                acc += R[k] * wv;
            }
        }
    } else if (m == hc_mix_dim) {
        if ((hc_dim & 3) == 0 && (reinterpret_cast<uintptr_t>(R) & 15u) == 0u) {
            const float4* R4 = reinterpret_cast<const float4*>(R);
            const int n4 = hc_dim >> 2;
            for (int k = tid; k < n4; k += 256) {
                float4 r = R4[k];
                acc += r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
            }
        } else {
            for (int k = tid; k < hc_dim; k += 256) {
                float v = R[k];
                acc += v * v;
            }
        }
    } else {
        return;
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) s_red[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        tmp_out[m] = tot;
    }
}

__global__ void __launch_bounds__(256) glm5_hc_sinkhorn_norm_q8_m1_kernel(
    const float* __restrict__ R,       // [hc, n_embd]
    const float* __restrict__ w_scale, // [3]
    const float* __restrict__ w_base,  // [hc_mix_dim]
    const float* __restrict__ norm_w,  // [n_embd]
    int n_embd,
    int hc,
    int iters,
    float norm_eps,
    float hc_eps,
    float* __restrict__ post_out,      // [hc]
    float* __restrict__ comb_out,      // [hc * hc]
    float* __restrict__ x_f32_out,     // in: [hc_mix_dim + 1] tmp; out: [n_embd]
    block_q8_1* __restrict__ x_q8_out  // nullable [n_embd / 32]
) {
    constexpr int MAX_HC = 8;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int hc_dim = n_embd * hc;
    const int hc_mix_dim = (2 + hc) * hc;

    __shared__ float s_red[8];
    __shared__ float s_mix[2 * MAX_HC * MAX_HC];
    __shared__ float s_pre[MAX_HC];
    __shared__ float s_inv;

    if (tid == 0) {
        float tot_ss = x_f32_out[hc_mix_dim];
        s_inv = rsqrtf(tot_ss / (float)hc_dim + norm_eps);
    }
    if (tid < hc_mix_dim) {
        s_mix[tid] = x_f32_out[tid];
    }
    __syncthreads();

    if (tid == 0) {
        const float inv = s_inv;
        for (int m = 0; m < hc_mix_dim; ++m) s_mix[m] *= inv;
        float* pt = post_out;
        float* cb = comb_out;
        for (int s = 0; s < hc; ++s) {
            float v = s_mix[s] * w_scale[0] + w_base[s];
            s_pre[s] = 1.0f / (1.0f + expf(-v)) + hc_eps;
        }
        for (int s = 0; s < hc; ++s) {
            float v = s_mix[hc + s] * w_scale[1] + w_base[hc + s];
            pt[s] = 2.0f / (1.0f + expf(-v));
        }
        for (int d = 0; d < hc; ++d) {
            for (int s = 0; s < hc; ++s) {
                cb[d * hc + s] = s_mix[2 * hc + d + hc * s] * w_scale[2] + w_base[2 * hc + d + hc * s];
            }
        }
        float col_max[MAX_HC];
        for (int s = 0; s < hc; ++s) {
            float mx = -INFINITY;
            for (int d = 0; d < hc; ++d) mx = fmaxf(mx, cb[d * hc + s]);
            col_max[s] = mx;
        }
        float denom[MAX_HC];
        for (int s = 0; s < hc; ++s) {
            float e = 0.0f;
            for (int d = 0; d < hc; ++d) e += expf(cb[d * hc + s] - col_max[s]);
            denom[s] = e;
        }
        for (int d = 0; d < hc; ++d) {
            for (int s = 0; s < hc; ++s) {
                cb[d * hc + s] = expf(cb[d * hc + s] - col_max[s]) / denom[s] + hc_eps;
            }
        }
        auto norm_dst = [&]() {
            for (int d = 0; d < hc; ++d) {
                float sum = hc_eps;
                for (int s = 0; s < hc; ++s) sum += cb[d * hc + s];
                for (int s = 0; s < hc; ++s) cb[d * hc + s] /= sum;
            }
        };
        auto norm_src = [&]() {
            for (int s = 0; s < hc; ++s) {
                float sum = hc_eps;
                for (int d = 0; d < hc; ++d) sum += cb[d * hc + s];
                for (int d = 0; d < hc; ++d) cb[d * hc + s] /= sum;
            }
        };
        norm_dst();
        for (int it = 1; it < iters; ++it) {
            norm_src();
            norm_dst();
        }
    }
    __syncthreads();

    float mss = 0.0f;
    for (int e = tid; e < n_embd; e += 256) {
        float acc = 0.0f;
        for (int s = 0; s < hc; ++s) acc += s_pre[s] * R[(size_t)s * n_embd + e];
        mss += acc * acc;
    }
    mss = warp_reduce_sum(mss);
    if (lane == 0) s_red[warp] = mss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)n_embd + norm_eps);
    }
    __syncthreads();
    const float inv_n = s_inv;

    for (int base = 0; base < n_embd; base += 256) {
        int e = base + tid;
        float val = 0.0f;
        if (e < n_embd) {
            float acc = 0.0f;
            for (int s = 0; s < hc; ++s) acc += s_pre[s] * R[(size_t)s * n_embd + e];
            val = acc * inv_n * norm_w[e];
            x_f32_out[e] = val;
        }
        if (x_q8_out != nullptr && (base + warp * 32) < n_embd) {
            warp_quantize_q8_1_store(val, x_q8_out + ((base >> 5) + warp), lane);
        }
    }
}

__device__ __forceinline__ uint32_t f32_to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ void mma1688_tf32(
    float (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

// Single-pass TF32 Tensor-Core tiled HC pre-norm + Sinkhorn + RMSNorm + Q8_1 (BM=32, BN=24, BK=64)
__global__ void __launch_bounds__(256) glm5_hc_pre_norm_q8_tc_kernel(
    const float* __restrict__ R,       // [M, 4, n_embd]
    const void* __restrict__ w_fn,     // [24, 4 * n_embd] (F32 or BF16)
    int32_t w_fn_type,
    const float* __restrict__ w_scale, // [3]
    const float* __restrict__ w_base,  // [24]
    const float* __restrict__ norm_w,  // [n_embd]
    int n_embd,
    int iters,
    float norm_eps,
    float hc_eps,
    float* __restrict__ post_out,      // [M, 4]
    float* __restrict__ comb_out,      // [M, 16]
    float* __restrict__ x_f32_out,     // [M, n_embd]
    block_q8_1* __restrict__ x_q8_out, // nullable [M, n_embd / 32]
    int M
) {
    constexpr int HC = 4;
    const int hc_dim = n_embd * HC;
    const int t0 = (int)blockIdx.x * 32;
    if (t0 >= M) return;
    const int bm_act = min(32, M - t0);

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int tid4 = lane & 3;
    const int warp_m = warp >> 2; // 0..1 (rows 0..15 / 16..31)
    const int warp_k = warp & 3;  // 0..3 (k slice 16 of 64)

    union HcTcSmem {
        struct {
            uint32_t R[32][68];
            uint32_t W[24][68];
        } ab;
        float part[4][32][24];
    };
    __shared__ HcTcSmem sm;
    __shared__ float s_inv[32];
    __shared__ float s_pre[32][4];
    __shared__ float s_cb[32][17];

    float acc[3][4] = {{0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f}};
    float ss0 = 0.0f;
    float ss1 = 0.0f;

    const int r_ld = tid >> 4;        // 0..15
    const int c4 = (tid & 15) << 2;   // 0, 4, .., 60

    for (int k0 = 0; k0 < hc_dim; k0 += 64) {
        if (r_ld < bm_act) {
            float4 v0 = *reinterpret_cast<const float4*>(R + (size_t)(t0 + r_ld) * hc_dim + k0 + c4);
            ss0 += v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w;
            sm.ab.R[r_ld][c4 + 0] = f32_to_tf32(v0.x);
            sm.ab.R[r_ld][c4 + 1] = f32_to_tf32(v0.y);
            sm.ab.R[r_ld][c4 + 2] = f32_to_tf32(v0.z);
            sm.ab.R[r_ld][c4 + 3] = f32_to_tf32(v0.w);
        } else {
            sm.ab.R[r_ld][c4 + 0] = 0u;
            sm.ab.R[r_ld][c4 + 1] = 0u;
            sm.ab.R[r_ld][c4 + 2] = 0u;
            sm.ab.R[r_ld][c4 + 3] = 0u;
        }
        const int r1_ld = r_ld + 16;
        if (r1_ld < bm_act) {
            float4 v1 = *reinterpret_cast<const float4*>(R + (size_t)(t0 + r1_ld) * hc_dim + k0 + c4);
            ss1 += v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w;
            sm.ab.R[r1_ld][c4 + 0] = f32_to_tf32(v1.x);
            sm.ab.R[r1_ld][c4 + 1] = f32_to_tf32(v1.y);
            sm.ab.R[r1_ld][c4 + 2] = f32_to_tf32(v1.z);
            sm.ab.R[r1_ld][c4 + 3] = f32_to_tf32(v1.w);
        } else {
            sm.ab.R[r1_ld][c4 + 0] = 0u;
            sm.ab.R[r1_ld][c4 + 1] = 0u;
            sm.ab.R[r1_ld][c4 + 2] = 0u;
            sm.ab.R[r1_ld][c4 + 3] = 0u;
        }

        const int wm0 = r_ld;
        const int wc4 = c4;
        if (w_fn_type == GGML_TYPE_F32) {
            const float* wp0 = reinterpret_cast<const float*>(w_fn) + (size_t)wm0 * hc_dim + k0 + wc4;
            float4 wv0 = *reinterpret_cast<const float4*>(wp0);
            sm.ab.W[wm0][wc4 + 0] = f32_to_tf32(wv0.x);
            sm.ab.W[wm0][wc4 + 1] = f32_to_tf32(wv0.y);
            sm.ab.W[wm0][wc4 + 2] = f32_to_tf32(wv0.z);
            sm.ab.W[wm0][wc4 + 3] = f32_to_tf32(wv0.w);
            if (tid < 128) {
                const int wm1 = 16 + r_ld;
                const float* wp1 = reinterpret_cast<const float*>(w_fn) + (size_t)wm1 * hc_dim + k0 + wc4;
                float4 wv1 = *reinterpret_cast<const float4*>(wp1);
                sm.ab.W[wm1][wc4 + 0] = f32_to_tf32(wv1.x);
                sm.ab.W[wm1][wc4 + 1] = f32_to_tf32(wv1.y);
                sm.ab.W[wm1][wc4 + 2] = f32_to_tf32(wv1.z);
                sm.ab.W[wm1][wc4 + 3] = f32_to_tf32(wv1.w);
            }
        } else {
            const uint16_t* wp0 = reinterpret_cast<const uint16_t*>(w_fn) + (size_t)wm0 * hc_dim + k0 + wc4;
            uint2 u0 = *reinterpret_cast<const uint2*>(wp0);
            sm.ab.W[wm0][wc4 + 0] = f32_to_tf32(bf16_bits_to_f32((uint16_t)u0.x));
            sm.ab.W[wm0][wc4 + 1] = f32_to_tf32(bf16_bits_to_f32((uint16_t)(u0.x >> 16)));
            sm.ab.W[wm0][wc4 + 2] = f32_to_tf32(bf16_bits_to_f32((uint16_t)u0.y));
            sm.ab.W[wm0][wc4 + 3] = f32_to_tf32(bf16_bits_to_f32((uint16_t)(u0.y >> 16)));
            if (tid < 128) {
                const int wm1 = 16 + r_ld;
                const uint16_t* wp1 = reinterpret_cast<const uint16_t*>(w_fn) + (size_t)wm1 * hc_dim + k0 + wc4;
                uint2 u1 = *reinterpret_cast<const uint2*>(wp1);
                sm.ab.W[wm1][wc4 + 0] = f32_to_tf32(bf16_bits_to_f32((uint16_t)u1.x));
                sm.ab.W[wm1][wc4 + 1] = f32_to_tf32(bf16_bits_to_f32((uint16_t)(u1.x >> 16)));
                sm.ab.W[wm1][wc4 + 2] = f32_to_tf32(bf16_bits_to_f32((uint16_t)u1.y));
                sm.ab.W[wm1][wc4 + 3] = f32_to_tf32(bf16_bits_to_f32((uint16_t)(u1.y >> 16)));
            }
        }
        __syncthreads();

        const int k_base = warp_k << 4;
        const int r_base = warp_m << 4;
        #pragma unroll
        for (int k8 = 0; k8 < 2; ++k8) {
            const int kl = k_base + (k8 << 3) + tid4;
            const int kh = kl + 4;
            const uint32_t a0 = sm.ab.R[r_base + gid][kl];
            const uint32_t a1 = sm.ab.R[r_base + gid + 8][kl];
            const uint32_t a2 = sm.ab.R[r_base + gid][kh];
            const uint32_t a3 = sm.ab.R[r_base + gid + 8][kh];
            #pragma unroll
            for (int nt = 0; nt < 3; ++nt) {
                const int col = (nt << 3) + gid;
                const uint32_t b0 = sm.ab.W[col][kl];
                const uint32_t b1 = sm.ab.W[col][kh];
                mma1688_tf32(acc[nt], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    // Reduce ss0 / ss1 across each 16-lane group holding one row's 64-col chunks
    #pragma unroll
    for (int off = 8; off >= 1; off >>= 1) {
        ss0 += __shfl_xor_sync(0xffffffffu, ss0, off);
        ss1 += __shfl_xor_sync(0xffffffffu, ss1, off);
    }
    if ((tid & 15) == 0) {
        s_inv[r_ld] = rsqrtf(ss0 / (float)hc_dim + norm_eps);
        s_inv[r_ld + 16] = rsqrtf(ss1 / (float)hc_dim + norm_eps);
    }

    // Scatter warp partial MMA tiles into sm.part[warp_k][r][c]
    {
        const int r0 = (warp_m << 4) + gid;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int nt = 0; nt < 3; ++nt) {
            const int c0 = (nt << 3) + (tid4 << 1);
            const int c1 = c0 + 1;
            sm.part[warp_k][r0][c0] = acc[nt][0];
            sm.part[warp_k][r0][c1] = acc[nt][1];
            sm.part[warp_k][r1][c0] = acc[nt][2];
            sm.part[warp_k][r1][c1] = acc[nt][3];
        }
    }
    __syncthreads();

    // Parallel exact Sinkhorn per token row (threads 0..bm_act-1)
    if (tid < bm_act) {
        const int t_loc = tid;
        const int tok = t0 + t_loc;
        const float inv = s_inv[t_loc];
        auto get_mix = [&](int m) -> float {
            return (sm.part[0][t_loc][m] + sm.part[1][t_loc][m]
                  + sm.part[2][t_loc][m] + sm.part[3][t_loc][m]) * inv;
        };
        const float ws0 = w_scale[0];
        const float ws1 = w_scale[1];
        const float ws2 = w_scale[2];
        for (int s = 0; s < HC; ++s) {
            const float v = get_mix(s) * ws0 + w_base[s];
            s_pre[t_loc][s] = 1.0f / (1.0f + expf(-v)) + hc_eps;
        }
        float* pt = post_out + (size_t)tok * HC;
        for (int s = 0; s < HC; ++s) {
            const float v = get_mix(HC + s) * ws1 + w_base[HC + s];
            pt[s] = 2.0f / (1.0f + expf(-v));
        }
        float* cb = s_cb[t_loc];
        for (int d = 0; d < HC; ++d) {
            for (int s = 0; s < HC; ++s) {
                const int m_idx = 2 * HC + d + HC * s;
                cb[d * HC + s] = get_mix(m_idx) * ws2 + w_base[m_idx];
            }
        }
        float col_max[HC];
        for (int s = 0; s < HC; ++s) {
            float mx = -INFINITY;
            for (int d = 0; d < HC; ++d) mx = fmaxf(mx, cb[d * HC + s]);
            col_max[s] = mx;
        }
        float denom[HC];
        for (int s = 0; s < HC; ++s) {
            float e = 0.0f;
            for (int d = 0; d < HC; ++d) e += expf(cb[d * HC + s] - col_max[s]);
            denom[s] = e;
        }
        for (int d = 0; d < HC; ++d) {
            for (int s = 0; s < HC; ++s) {
                cb[d * HC + s] = expf(cb[d * HC + s] - col_max[s]) / denom[s] + hc_eps;
            }
        }
        auto norm_dst = [&]() {
            for (int d = 0; d < HC; ++d) {
                const float sum = hc_eps + cb[d * 4 + 0] + cb[d * 4 + 1] + cb[d * 4 + 2] + cb[d * 4 + 3];
                cb[d * 4 + 0] /= sum;
                cb[d * 4 + 1] /= sum;
                cb[d * 4 + 2] /= sum;
                cb[d * 4 + 3] /= sum;
            }
        };
        auto norm_src = [&]() {
            for (int s = 0; s < HC; ++s) {
                const float sum = hc_eps + cb[0 * 4 + s] + cb[1 * 4 + s] + cb[2 * 4 + s] + cb[3 * 4 + s];
                cb[0 * 4 + s] /= sum;
                cb[1 * 4 + s] /= sum;
                cb[2 * 4 + s] /= sum;
                cb[3 * 4 + s] /= sum;
            }
        };
        norm_dst();
        for (int it = 1; it < iters; ++it) {
            norm_src();
            norm_dst();
        }
        float* cb_out = comb_out + (size_t)tok * (HC * HC);
        for (int i = 0; i < HC * HC; ++i) cb_out[i] = cb[i];
    }
    __syncthreads();

    // Pass 4: single-read stream mix + L1-cached RMSNorm + optional Q8_1
    const int n4 = n_embd >> 2;
    for (int t_loc = warp; t_loc < bm_act; t_loc += 8) {
        const int tok = t0 + t_loc;
        const float p0 = s_pre[t_loc][0];
        const float p1 = s_pre[t_loc][1];
        const float p2 = s_pre[t_loc][2];
        const float p3 = s_pre[t_loc][3];
        const float* Rt = R + (size_t)tok * hc_dim;
        const float4* R0 = reinterpret_cast<const float4*>(Rt);
        const float4* R1 = reinterpret_cast<const float4*>(Rt + n_embd);
        const float4* R2 = reinterpret_cast<const float4*>(Rt + 2 * n_embd);
        const float4* R3 = reinterpret_cast<const float4*>(Rt + 3 * n_embd);
        float* xf = x_f32_out + (size_t)tok * n_embd;
        float4* XF4 = reinterpret_cast<float4*>(xf);
        float mss = 0.0f;
        for (int i = lane; i < n4; i += 32) {
            const float4 v0 = R0[i], v1 = R1[i], v2 = R2[i], v3 = R3[i];
            float4 xv;
            xv.x = p0 * v0.x + p1 * v1.x + p2 * v2.x + p3 * v3.x;
            xv.y = p0 * v0.y + p1 * v1.y + p2 * v2.y + p3 * v3.y;
            xv.z = p0 * v0.z + p1 * v1.z + p2 * v2.z + p3 * v3.z;
            xv.w = p0 * v0.w + p1 * v1.w + p2 * v2.w + p3 * v3.w;
            XF4[i] = xv;
            mss += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        }
        mss = warp_reduce_sum(mss);
        const float inv_n = rsqrtf(mss / (float)n_embd + norm_eps);
        block_q8_1* xq = x_q8_out ? (x_q8_out + (size_t)tok * (n_embd >> 5)) : nullptr;
        for (int b = 0; b < n_embd; b += 32) {
            const int e = b + lane;
            const float val = xf[e] * inv_n * norm_w[e];
            xf[e] = val;
            if (xq != nullptr) {
                warp_quantize_q8_1_store(val, xq + (b >> 5), lane);
            }
        }
    }
}

extern "C" void launch_glm5_hc_pre_norm_q8(
    const float* R,
    const void* w_fn,
    int32_t w_fn_type,
    const float* w_scale,
    const float* w_base,
    const float* norm_w,
    int n_embd,
    int hc,
    int iters,
    float norm_eps,
    float hc_eps,
    float* post_out,
    float* comb_out,
    float* x_f32_out,
    void* x_q8_out,
    int M,
    cudaStream_t stream
) {
    if (M <= 0 || n_embd <= 0) return;
    const int hc_mix_dim = (2 + hc) * hc;
    if (M == 1 && (hc_mix_dim + 1) <= n_embd && x_f32_out != nullptr) {
        glm5_hc_matvec_m1_kernel<<<hc_mix_dim + 1, 256, 0, stream>>>(
            R, w_fn, w_fn_type, n_embd * hc, hc_mix_dim, x_f32_out
        );
        glm5_hc_sinkhorn_norm_q8_m1_kernel<<<1, 256, 0, stream>>>(
            R, w_scale, w_base, norm_w, n_embd, hc, iters, norm_eps, hc_eps,
            post_out, comb_out, x_f32_out, reinterpret_cast<block_q8_1*>(x_q8_out)
        );
        return;
    }
    if (M >= 256 && hc == 4 && (n_embd & 255) == 0 && x_f32_out != nullptr
        && (w_fn_type == GGML_TYPE_F32 || w_fn_type == GGML_TYPE_BF16)) {
        glm5_hc_pre_norm_q8_tc_kernel<<<(M + 31) / 32, 256, 0, stream>>>(
            R, w_fn, w_fn_type, w_scale, w_base, norm_w, n_embd, iters,
            norm_eps, hc_eps, post_out, comb_out, x_f32_out,
            reinterpret_cast<block_q8_1*>(x_q8_out), M
        );
    } else {
        glm5_hc_pre_norm_q8_kernel<<<M, 256, 0, stream>>>(
            R, w_fn, w_fn_type, w_scale, w_base, norm_w, n_embd, hc, iters,
            norm_eps, hc_eps, post_out, comb_out, x_f32_out,
            reinterpret_cast<block_q8_1*>(x_q8_out)
        );
    }
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        static std::atomic<int> s_hc_err_cnt{0};
        if (s_hc_err_cnt.fetch_add(1, std::memory_order_relaxed) < 16) {
            int dev = -1;
            cudaGetDevice(&dev);
            std::fprintf(stderr, "[CUDA ERROR] launch_glm5_hc_pre_norm_q8 dev=%d w_fn_type=%d M=%d n_embd=%d err=%d (%s)\n",
                         dev, w_fn_type, M, n_embd, (int)err, cudaGetErrorString(err));
        }
    }
}

__global__ void glm5_hc_post_add_kernel(
    const float* R_in,               // [M, hc, n_embd] (may alias R_out for in-place)
    const float* __restrict__ post,  // [M, hc]
    const float* __restrict__ comb,  // [M, hc * hc]
    const float* __restrict__ a,     // [M, n_embd]
    const float* __restrict__ b_opt, // nullable [M, n_embd]
    const float* __restrict__ c_opt, // nullable [M, n_embd]
    float* R_out,                    // [M, hc, n_embd] (may alias R_in for in-place)
    int n_embd,
    int hc,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * n_embd) return;
    int t = idx / n_embd;
    int e = idx % n_embd;

    float bo = a[idx];
    if (b_opt) bo += b_opt[idx];
    if (c_opt) bo += c_opt[idx];

    const float* Rt = R_in + (size_t)t * hc * n_embd;
    float* Ro = R_out + (size_t)t * hc * n_embd;
    const float* pt = post + (size_t)t * hc;
    const float* cb = comb + (size_t)t * hc * hc;

    if (hc == 4) {
        float r0 = Rt[0 * n_embd + e];
        float r1 = Rt[1 * n_embd + e];
        float r2 = Rt[2 * n_embd + e];
        float r3 = Rt[3 * n_embd + e];
        for (int d = 0; d < 4; ++d) {
            Ro[d * n_embd + e] = bo * pt[d]
                + cb[d * 4 + 0] * r0 + cb[d * 4 + 1] * r1
                + cb[d * 4 + 2] * r2 + cb[d * 4 + 3] * r3;
        }
    } else {
        float r_tmp[8];
        int hc_c = (hc <= 8) ? hc : 8;
        for (int s = 0; s < hc_c; ++s) r_tmp[s] = Rt[s * n_embd + e];
        for (int d = 0; d < hc_c; ++d) {
            float acc = bo * pt[d];
            for (int s = 0; s < hc_c; ++s) acc += cb[d * hc + s] * r_tmp[s];
            Ro[d * n_embd + e] = acc;
        }
    }
}

extern "C" void launch_glm5_hc_post_add(
    const float* R_in,
    const float* post,
    const float* comb,
    const float* a,
    const float* b_opt,
    const float* c_opt,
    float* R_out,
    int n_embd,
    int hc,
    int M,
    cudaStream_t stream
) {
    int total = M * n_embd;
    if (total <= 0) return;
    glm5_hc_post_add_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        R_in, post, comb, a, b_opt, c_opt, R_out, n_embd, hc, M
    );
}

__device__ __forceinline__ void hc_grid_barrier(uint64_t* ctr, uint32_t nb) {
    __syncthreads();
    __shared__ uint64_t s_target;
    if (threadIdx.x == 0) {
        __threadfence();
        unsigned long long prev = atomicAdd(
            reinterpret_cast<unsigned long long*>(ctr),
            (unsigned long long)1ULL
        );
        s_target = (prev / (unsigned long long)nb + 1ULL) * (unsigned long long)nb;
        const uint64_t* p = ctr;
        for (;;) {
            uint64_t cur = 0ULL;
            asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(cur) : "l"(p) : "memory");
            if (cur >= s_target) break;
        }
        __threadfence();
    }
    __syncthreads();
}

// Fused multi-block (32 blocks x 256 threads) M=1 HC-post (optional) + HC-pre + 20-iter warp-shuffle Sinkhorn + RMSNorm + Q8_1
// Note: post_in/post_out and comb_in/comb_out intentionally omit __restrict__ because callers pass in-place sc.post / sc.comb.
__global__ void __launch_bounds__(256) glm5_hc_fused_m1_kernel(
    const float* __restrict__ block_out,   // nullable [4096]
    const float* __restrict__ R_old,       // [4, 4096]
    float* __restrict__ R_new,             // nullable [4, 4096] when block_out != null
    const float* post_in,                  // nullable [4] when block_out != null
    const float* comb_in,                  // nullable [16] when block_out != null
    float* post_out,                       // [4]
    float* comb_out,                       // [16]
    const void* __restrict__ w_fn,         // [24, 16384] BF16
    const float* __restrict__ w_scale,     // [3]
    const float* __restrict__ w_base,      // [24]
    const float* __restrict__ norm_w,      // [4096]
    int iters,
    float norm_eps,
    float hc_eps,
    float* __restrict__ x_f32_out,         // [4096]
    block_q8_1* __restrict__ x_q8_out,     // nullable [4096 / 32]
    float* __restrict__ hc_part,           // [26 * 32]
    uint64_t* __restrict__ hc_ctr,         // [1] monotonic grid barrier counter
    const int32_t* __restrict__ skip_dev   // nullable [1]
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    constexpr int HC = 4;
    constexpr int HC_EB = 128;
    constexpr int HC_MIX = 24;
    constexpr int NB = 32;
    const int b = blockIdx.x;
    const int e0 = b * HC_EB;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    __shared__ float sR[HC][HC_EB];
    __shared__ float sred[8];
    __shared__ float smix[HC_MIX + 1];
    __shared__ float s_pre[HC];
    __shared__ float s_mixed[HC_EB];

    uint2 wv[3][HC];
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int m = warp + 8 * r;
        const uint16_t* wr = reinterpret_cast<const uint16_t*>(w_fn) + (size_t)m * (HC * 4096) + e0 + 4 * lane;
        #pragma unroll
        for (int d = 0; d < HC; ++d) {
            wv[r][d] = *reinterpret_cast<const uint2*>(wr + (size_t)d * 4096);
        }
    }

    float ss = 0.0f;
    for (int i = tid; i < HC * HC_EB; i += 256) {
        const int d = i >> 7;
        const int j = i & (HC_EB - 1);
        const int e = e0 + j;
        float r;
        if (block_out != nullptr) {
            r = block_out[e] * post_in[d];
            #pragma unroll
            for (int s = 0; s < HC; ++s) {
                r += comb_in[d * HC + s] * R_old[(size_t)s * 4096 + e];
            }
            if (R_new != nullptr) {
                R_new[(size_t)d * 4096 + e] = r;
            }
        } else {
            r = R_old[(size_t)d * 4096 + e];
        }
        sR[d][j] = r;
        ss += r * r;
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        const int m = warp + 8 * r;
        float acc = 0.0f;
        #pragma unroll
        for (int d = 0; d < HC; ++d) {
            const uint2 w = wv[r][d];
            const float* x = &sR[d][4 * lane];
            acc += bf16_bits_to_f32((uint16_t)w.x)         * x[0]
                 + bf16_bits_to_f32((uint16_t)(w.x >> 16)) * x[1]
                 + bf16_bits_to_f32((uint16_t)w.y)         * x[2]
                 + bf16_bits_to_f32((uint16_t)(w.y >> 16)) * x[3];
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) hc_part[(size_t)m * NB + b] = acc;
    }

    ss = warp_reduce_sum(ss);
    if (lane == 0) sred[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float bt = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) bt += sred[w];
        hc_part[(size_t)HC_MIX * NB + b] = bt;
    }

    hc_grid_barrier(hc_ctr, NB);

    for (int m = warp; m < HC_MIX + 1; m += 8) {
        float v = __ldcg(&hc_part[(size_t)m * NB + lane]);
        float t = warp_reduce_sum(v);
        if (lane == 0) smix[m] = t;
    }
    __syncthreads();

    const float inv = rsqrtf(smix[HC_MIX] / (float)(HC * 4096) + norm_eps);
    if (warp == 0) {
        if (lane < HC) {
            float v_pre = smix[lane] * inv * w_scale[0] + w_base[lane];
            float pre_v = 1.0f / (1.0f + expf(-v_pre)) + hc_eps;
            s_pre[lane] = pre_v;
            if (b == 0) {
                float v_post = smix[HC + lane] * inv * w_scale[1] + w_base[HC + lane];
                post_out[lane] = 2.0f / (1.0f + expf(-v_post));
            }
        }
        const int l = lane & 15;
        const int d = l >> 2;
        const int s = l & 3;
        float c = smix[2 * HC + d + HC * s] * inv * w_scale[2] + w_base[2 * HC + d + HC * s];
        float mx = fmaxf(c, __shfl_xor_sync(0xFFFFFFFFu, c, 4));
        mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, 8));
        float ex = expf(c - mx);
        float den = ex + __shfl_xor_sync(0xFFFFFFFFu, ex, 4);
        den += __shfl_xor_sync(0xFFFFFFFFu, den, 8);
        c = ex / den + hc_eps;
        {
            float sm = c + __shfl_xor_sync(0xFFFFFFFFu, c, 1);
            sm += __shfl_xor_sync(0xFFFFFFFFu, sm, 2);
            c = c / (hc_eps + sm);
        }
        for (int it = 1; it < iters; ++it) {
            float sm = c + __shfl_xor_sync(0xFFFFFFFFu, c, 4);
            sm += __shfl_xor_sync(0xFFFFFFFFu, sm, 8);
            c = c / (hc_eps + sm);
            sm = c + __shfl_xor_sync(0xFFFFFFFFu, c, 1);
            sm += __shfl_xor_sync(0xFFFFFFFFu, sm, 2);
            c = c / (hc_eps + sm);
        }
        if (b == 0 && lane < 16) {
            comb_out[d * HC + s] = c;
        }
    }
    __syncthreads();

    float ss2 = 0.0f;
    if (tid < HC_EB) {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < HC; ++s) {
            acc += s_pre[s] * sR[s][tid];
        }
        s_mixed[tid] = acc;
        ss2 = acc * acc;
    }
    ss2 = warp_reduce_sum(ss2);
    if (lane == 0) sred[warp] = ss2;
    __syncthreads();
    if (tid == 0) {
        float bt = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) bt += sred[w];
        hc_part[(size_t)(HC_MIX + 1) * NB + b] = bt;
    }

    hc_grid_barrier(hc_ctr, NB);

    float tot = warp_reduce_sum(__ldcg(&hc_part[(size_t)(HC_MIX + 1) * NB + lane]));
    const float inv2 = rsqrtf(tot / 4096.0f + norm_eps);
    if (tid < HC_EB) {
        const int e = e0 + tid;
        const float xv = s_mixed[tid] * inv2 * norm_w[e];
        x_f32_out[e] = xv;
        if (x_q8_out != nullptr) {
            warp_quantize_q8_1_store(xv, x_q8_out + (e >> 5), lane);
        }
    }
}

extern "C" void launch_glm5_hc_fused_m1(
    const float* block_out,
    const float* R_old,
    float* R_new,
    const float* post_in,
    const float* comb_in,
    float* post_out,
    float* comb_out,
    const void* w_fn,
    int32_t w_fn_type,
    const float* w_scale,
    const float* w_base,
    const float* norm_w,
    int n_embd,
    int hc,
    int iters,
    float norm_eps,
    float hc_eps,
    float* x_f32_out,
    void* x_q8_out,
    float* hc_part,
    uint64_t* hc_ctr,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (n_embd == 4096 && hc == 4 && w_fn_type == GGML_TYPE_BF16
        && x_f32_out != nullptr && hc_part != nullptr && hc_ctr != nullptr
        && (reinterpret_cast<uintptr_t>(w_fn) & 7u) == 0u) {
        glm5_hc_fused_m1_kernel<<<32, 256, 0, stream>>>(
            block_out, R_old, R_new, post_in, comb_in, post_out, comb_out, w_fn,
            w_scale, w_base, norm_w, iters, norm_eps, hc_eps, x_f32_out,
            reinterpret_cast<block_q8_1*>(x_q8_out), hc_part, hc_ctr, skip_dev
        );
        return;
    }
    const float* r_src = R_old;
    if (block_out != nullptr && R_new != nullptr) {
        launch_glm5_hc_post_add(
            R_old, post_in, comb_in, block_out, nullptr, nullptr, R_new,
            n_embd, hc, 1, stream
        );
        r_src = R_new;
    }
    launch_glm5_hc_pre_norm_q8(
        r_src, w_fn, w_fn_type, w_scale, w_base, norm_w, n_embd, hc, iters,
        norm_eps, hc_eps, post_out, comb_out, x_f32_out, x_q8_out, 1, stream
    );
}

__global__ void __launch_bounds__(256) glm5_mean_streams_norm_q8_kernel(
    const float* __restrict__ R,       // [M, hc, n_embd]
    const float* __restrict__ norm_w,  // [n_embd]
    float eps,
    int n_embd,
    int hc,
    float* __restrict__ out_f32,
    block_q8_1* __restrict__ out_q8
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* Rt = R + (size_t)t * hc * n_embd;
    const float inv_hc = 1.0f / (float)hc;

    __shared__ float s_red[8];
    __shared__ float s_inv;
    float ss = 0.0f;
    for (int e = tid; e < n_embd; e += 256) {
        float acc = 0.0f;
        for (int s = 0; s < hc; ++s) acc += Rt[(size_t)s * n_embd + e];
        acc *= inv_hc;
        ss += acc * acc;
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)n_embd + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    float* xf = out_f32 + (size_t)t * n_embd;
    block_q8_1* xq = out_q8 ? (out_q8 + (size_t)t * (n_embd >> 5)) : nullptr;
    for (int base = 0; base < n_embd; base += 256) {
        int e = base + tid;
        float val = 0.0f;
        if (e < n_embd) {
            float acc = 0.0f;
            for (int s = 0; s < hc; ++s) acc += Rt[(size_t)s * n_embd + e];
            val = (acc * inv_hc) * inv * norm_w[e];
            xf[e] = val;
        }
        if (xq != nullptr && (base + warp * 32) < n_embd) {
            warp_quantize_q8_1_store(val, xq + ((base >> 5) + warp), lane);
        }
    }
}

extern "C" void launch_glm5_mean_streams_norm_q8(
    const float* R,
    const float* norm_w,
    float eps,
    int n_embd,
    int hc,
    float* out_f32,
    void* out_q8,
    int M,
    cudaStream_t stream
) {
    if (M <= 0 || n_embd <= 0) return;
    glm5_mean_streams_norm_q8_kernel<<<M, 256, 0, stream>>>(
        R, norm_w, eps, n_embd, hc, out_f32, reinterpret_cast<block_q8_1*>(out_q8)
    );
}

__global__ void __launch_bounds__(256) glm5_rms_norm_q8_kernel(
    float* __restrict__ base_res,      // [M, dim]
    const float* __restrict__ add_opt, // nullable [M, dim]
    const float* __restrict__ norm_w,  // nullable [dim]
    float eps,
    int dim,
    int write_back_base,
    float* __restrict__ out_f32,
    block_q8_1* __restrict__ out_q8
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    float* bt = base_res + (size_t)t * dim;
    const float* at = add_opt ? (add_opt + (size_t)t * dim) : nullptr;

    __shared__ float s_red[8];
    __shared__ float s_inv;
    float ss = 0.0f;
    for (int e = tid; e < dim; e += 256) {
        float v = bt[e] + (at ? at[e] : 0.0f);
        if (write_back_base) bt[e] = v;
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)dim + eps);
    }
    __syncthreads();
    const float inv = s_inv;
    float* xf = out_f32 + (size_t)t * dim;
    block_q8_1* xq = out_q8 ? (out_q8 + (size_t)t * (dim >> 5)) : nullptr;
    for (int base = 0; base < dim; base += 256) {
        int e = base + tid;
        float val = 0.0f;
        if (e < dim) {
            float v = (write_back_base || !at) ? bt[e] : (bt[e] + at[e]);
            val = v * inv * (norm_w ? norm_w[e] : 1.0f);
            xf[e] = val;
        }
        if (xq != nullptr && (base + warp * 32) < dim) {
            warp_quantize_q8_1_store(val, xq + ((base >> 5) + warp), lane);
        }
    }
}

extern "C" void launch_glm5_rms_norm_q8(
    float* base_res,
    const float* add_opt,
    const float* norm_w,
    float eps,
    int dim,
    int write_back_base,
    float* out_f32,
    void* out_q8,
    int M,
    cudaStream_t stream
) {
    if (M <= 0 || dim <= 0) return;
    glm5_rms_norm_q8_kernel<<<M, 256, 0, stream>>>(
        base_res, add_opt, norm_w, eps, dim, write_back_base, out_f32,
        reinterpret_cast<block_q8_1*>(out_q8)
    );
}

__global__ void __launch_bounds__(256) glm5_swiglu_clamp_q8_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float lim,
    int n_ff,
    float* __restrict__ out_f32_opt,
    block_q8_1* __restrict__ out_q8
) {
    const int blk = blockIdx.x * 8 + threadIdx.y; // 8 warps = 8 q8_1 blocks
    const int t = blockIdx.y;
    const int lane = threadIdx.x;
    const int n_blocks = n_ff >> 5;
    if (blk >= n_blocks) return;

    int e = (blk << 5) + lane;
    size_t idx = (size_t)t * n_ff + e;
    float h = glm_swiglu_clamp_val(gate[idx], up[idx], lim);
    if (out_f32_opt) out_f32_opt[idx] = h;
    warp_quantize_q8_1_store(h, out_q8 + (size_t)t * n_blocks + blk, lane);
}

extern "C" void launch_glm5_swiglu_clamp_q8(
    const float* gate,
    const float* up,
    float lim,
    int n_ff,
    float* out_f32_opt,
    void* out_q8,
    int M,
    cudaStream_t stream
) {
    if (M <= 0 || n_ff <= 0) return;
    int n_blk = n_ff >> 5;
    dim3 grid((n_blk + 7) / 8, M);
    dim3 block(32, 8);
    glm5_swiglu_clamp_q8_kernel<<<grid, block, 0, stream>>>(
        gate, up, lim, n_ff, out_f32_opt, reinterpret_cast<block_q8_1*>(out_q8)
    );
}

// -----------------------------------------------------------------------------
// KDA (Key-Driven Linear Attention) Kernels
// -----------------------------------------------------------------------------

__global__ void glm5_kda_conv_kernel(
    const float* __restrict__ x_proj,     // [T, d_inner]
    const float* __restrict__ conv_w,     // [d_inner, d_conv]
    float* __restrict__ conv_state,       // [d_inner, d_conv - 1]
    int d_inner,
    int d_conv,
    int T,
    float* __restrict__ out               // [T, d_inner]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_inner * T) return;
    int t = idx / d_inner;
    int c = idx - t * d_inner;
    float acc = 0.0f;
    if (d_conv == 4) {
        const float* w = conv_w + (size_t)c * 4;
        const float* cs = conv_state + (size_t)c * 3;
        float x0 = (t >= 3) ? x_proj[(size_t)(t - 3) * d_inner + c] : cs[t];
        float x1 = (t >= 2) ? x_proj[(size_t)(t - 2) * d_inner + c] : cs[t + 1];
        float x2 = (t >= 1) ? x_proj[(size_t)(t - 1) * d_inner + c] : cs[t + 2];
        float x3 = x_proj[(size_t)t * d_inner + c];
        acc = w[0] * x0 + w[1] * x1 + w[2] * x2 + w[3] * x3;
    } else {
        int hist = d_conv - 1;
        for (int tap = 0; tap < d_conv; ++tap) {
            int src = t - hist + tap;
            float x = (src >= 0) ? x_proj[(size_t)src * d_inner + c]
                                 : conv_state[(size_t)c * hist + (src + hist)];
            acc += conv_w[(size_t)c * d_conv + tap] * x;
        }
    }
    out[idx] = acc / (1.0f + __expf(-acc));
}

__global__ void glm5_kda_conv_slide_kernel(
    const float* __restrict__ x_proj,
    float* __restrict__ conv_state,
    int d_inner,
    int d_conv,
    int T
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d_inner) return;
    int hist = d_conv - 1;
    float nv[16];
    for (int j = 0; j < hist; ++j) {
        int src = T - hist + j;
        nv[j] = (src >= 0) ? x_proj[(size_t)src * d_inner + c]
                           : conv_state[(size_t)c * hist + (src + hist)];
    }
    for (int j = 0; j < hist; ++j) conv_state[(size_t)c * hist + j] = nv[j];
}

__global__ void glm5_kda_conv_m1_inplace_kernel(
    const float* __restrict__ x_proj,
    const float* __restrict__ conv_w,
    float* __restrict__ conv_state,
    int d_inner,
    int d_conv,
    float* __restrict__ out
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d_inner) return;
    int hist = d_conv - 1;
    float acc = 0.0f;
    float s0 = conv_state[(size_t)c * hist + 0];
    float s1 = (hist > 1) ? conv_state[(size_t)c * hist + 1] : 0.0f;
    float s2 = (hist > 2) ? conv_state[(size_t)c * hist + 2] : 0.0f;
    float xn = x_proj[c];
    if (d_conv == 4) {
        acc = conv_w[c * 4 + 0] * s0 + conv_w[c * 4 + 1] * s1 + conv_w[c * 4 + 2] * s2 + conv_w[c * 4 + 3] * xn;
        conv_state[c * 3 + 0] = s1;
        conv_state[c * 3 + 1] = s2;
        conv_state[c * 3 + 2] = xn;
    } else {
        for (int j = 0; j < hist; ++j) acc += conv_w[c * d_conv + j] * conv_state[c * hist + j];
        acc += conv_w[c * d_conv + hist] * xn;
        for (int j = 0; j + 1 < hist; ++j) conv_state[c * hist + j] = conv_state[c * hist + j + 1];
        conv_state[c * hist + hist - 1] = xn;
    }
    out[c] = acc / (1.0f + expf(-acc));
}

__global__ void glm5_kda_l2_heads_kernel(
    float* __restrict__ x, // in-place [T, n_head, head_dim]
    int head_dim,
    int n_head,
    int T
) {
    int ht = blockIdx.x * blockDim.y + threadIdx.y; // 1 warp per head
    if (ht >= n_head * T) return;
    int lane = threadIdx.x;
    float* xh = x + (size_t)ht * head_dim;
    float ss = 0.0f;
    for (int e = lane; e < head_dim; e += 32) {
        float v = xh[e];
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    float inv = rsqrtf(ss + 1e-6f);
    for (int e = lane; e < head_dim; e += 32) {
        xh[e] *= inv;
    }
}

extern "C" void launch_glm5_kda_conv_silu_l2(
    const float* x_proj,
    const float* conv_w,
    float* conv_state,
    int d_inner,
    int d_conv,
    int head_dim,
    int n_head,
    int apply_l2,
    int T,
    float* out,
    cudaStream_t stream
) {
    if (T <= 0 || d_inner <= 0) return;
    if (T == 1) {
        glm5_kda_conv_m1_inplace_kernel<<<(d_inner + 255) / 256, 256, 0, stream>>>(
            x_proj, conv_w, conv_state, d_inner, d_conv, out
        );
    } else {
        int tot = d_inner * T;
        glm5_kda_conv_kernel<<<(tot + 255) / 256, 256, 0, stream>>>(
            x_proj, conv_w, conv_state, d_inner, d_conv, T, out
        );
        glm5_kda_conv_slide_kernel<<<(d_inner + 255) / 256, 256, 0, stream>>>(
            x_proj, conv_state, d_inner, d_conv, T
        );
    }
    if (apply_l2) {
        int total_h = n_head * T;
        dim3 block(32, 4);
        glm5_kda_l2_heads_kernel<<<(total_h + 3) / 4, block, 0, stream>>>(
            out, head_dim, n_head, T
        );
    }
}

__global__ void glm5_kda_gate_beta_kernel(
    const float* __restrict__ g_in,    // [T, d_inner]
    const float* __restrict__ dt_bias, // [d_inner]
    const float* __restrict__ ssm_a,   // [n_head]
    float lower_bound,
    int head_dim,
    int n_head,
    int T,
    float* __restrict__ g1_out,        // [T, n_head, head_dim]
    float* __restrict__ beta_inout     // [T, n_head]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tot = head_dim * n_head * T;
    if (idx < tot) {
        int e = idx % head_dim;
        int h = (idx / head_dim) % n_head;
        int t = idx / (head_dim * n_head);
        int g_idx = e + head_dim * h;
        float v = g_in[(size_t)t * (head_dim * n_head) + g_idx] + dt_bias[g_idx];
        g1_out[idx] = lower_bound / (1.0f + __expf(v * ssm_a[h]));
    }
    if (idx < n_head * T) {
        float b = beta_inout[idx];
        beta_inout[idx] = 1.0f / (1.0f + __expf(-b));
    }
}

extern "C" void launch_glm5_kda_gate_beta(
    const float* g_in,
    const float* dt_bias,
    const float* ssm_a,
    float lower_bound,
    int head_dim,
    int n_head,
    int T,
    float* g1_out,
    float* beta_inout,
    cudaStream_t stream
) {
    int tot = head_dim * n_head * T;
    if (tot <= 0) return;
    glm5_kda_gate_beta_kernel<<<(tot + 255) / 256, 256, 0, stream>>>(
        g_in, dt_bias, ssm_a, lower_bound, head_dim, n_head, T, g1_out, beta_inout
    );
}

// SMEM-resident KDA recurrence: 1 block of 128 threads per head.
// State S[h, i, j] (i=key 0..dk-1, j=val 0..dk-1) loaded into 64KB SMEM once for all T tokens!
__global__ void __launch_bounds__(128) glm5_kda_recurrence_kernel(
    const float* __restrict__ q,    // [T, n_head, dk]
    const float* __restrict__ k,    // [T, n_head, dk]
    const float* __restrict__ v,    // [T, n_head, dk]
    const float* __restrict__ g1,   // [T, n_head, dk]
    const float* __restrict__ beta, // [T, n_head]
    float* __restrict__ state,      // [n_head, dk, dk]
    int n_head,
    int dk,
    int T,
    float* __restrict__ out,        // [T, n_head, dk]
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int h = blockIdx.x;
    if (h >= n_head) return;
    const int j = threadIdx.x; // 0..dk-1 (when dk=128)
    const float scale = rsqrtf((float)dk);

    extern __shared__ float smem[];
    // layout: s_S [dk * dk], s_k [dk], s_q [dk], s_dec [dk]
    float* s_S   = smem;
    float* s_k   = smem + dk * dk;
    float* s_q   = s_k + dk;
    float* s_dec = s_q + dk;

    float* S_g = state + (size_t)h * dk * dk;
    for (int i = 0; i < dk; ++i) {
        s_S[i * dk + j] = S_g[i * dk + j];
    }

    for (int t = 0; t < T; ++t) {
        size_t ht_off = ((size_t)t * n_head + h) * dk;
        s_k[j]   = k[ht_off + j];
        s_q[j]   = q[ht_off + j];
        s_dec[j] = expf(g1[ht_off + j]);
        float vj = v[ht_off + j];
        float b  = beta[(size_t)t * n_head + h];
        __syncthreads();

        float u_j = 0.0f;
        for (int i = 0; i < dk; ++i) {
            float sv = s_S[i * dk + j] * s_dec[i];
            s_S[i * dk + j] = sv;
            u_j += sv * s_k[i];
        }
        float delta_j = b * (vj - u_j);
        float o_j = 0.0f;
        for (int i = 0; i < dk; ++i) {
            float sv = s_S[i * dk + j] + s_k[i] * delta_j;
            s_S[i * dk + j] = sv;
            o_j += sv * s_q[i];
        }
        out[ht_off + j] = o_j * scale;
        __syncthreads();
    }

    for (int i = 0; i < dk; ++i) {
        S_g[i * dk + j] = s_S[i * dk + j];
    }
}

static uint32_t s_kda_max_smem_mask = 0u;

extern "C" void launch_glm5_prepare_kda_smem(int head_dim) {
    if (head_dim <= 0) return;
    size_t smem_bytes = (size_t)(head_dim * head_dim + 3 * head_dim) * sizeof(float);
    if (smem_bytes > 49152) {
        int dev = 0;
        cudaGetDevice(&dev);
        uint32_t bit = 1u << (dev & 31);
        if ((s_kda_max_smem_mask & bit) == 0u) {
            cudaFuncSetAttribute(glm5_kda_recurrence_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
            s_kda_max_smem_mask |= bit;
        }
    }
}

__global__ void __launch_bounds__(32) glm5_kda_recurrence_m1_kernel(
    const float* __restrict__ q,    // [n_head, 128]
    const float* __restrict__ k,    // [n_head, 128]
    const float* __restrict__ v,    // [n_head, 128]
    const float* __restrict__ g1,   // [n_head, 128]
    const float* __restrict__ beta, // [n_head]
    float* __restrict__ state,      // [n_head, 128, 128]
    int n_head,
    float* __restrict__ out,        // [n_head, 128]
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int h = blockIdx.y;
    if (h >= n_head) return;
    const int lane = threadIdx.x; // 0..31
    const int j = blockIdx.x * 32 + lane;

    __shared__ float s_k[128];
    __shared__ float s_q[128];
    __shared__ float s_dec[128];

    const size_t h_off = (size_t)h * 128;
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        int idx = p * 32 + lane;
        s_k[idx] = k[h_off + idx];
        s_q[idx] = q[h_off + idx];
        s_dec[idx] = expf(g1[h_off + idx]);
    }
    const float vj = v[h_off + j];
    const float b = beta[h];
    __syncwarp();

    float* S_g = state + (size_t)h * (128 * 128);
    float u_j = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < 128; ++i) {
        u_j += (S_g[i * 128 + j] * s_dec[i]) * s_k[i];
    }
    const float delta_j = b * (vj - u_j);
    float o_j = 0.0f;
    #pragma unroll 4
    for (int i = 0; i < 128; ++i) {
        float sv = S_g[i * 128 + j] * s_dec[i] + s_k[i] * delta_j;
        S_g[i * 128 + j] = sv;
        o_j += sv * s_q[i];
    }
    out[h_off + j] = o_j * rsqrtf(128.0f);
}

template <int KJ>
__global__ void __launch_bounds__(256) glm5_kda_recurrence_split_kernel(
    const float* __restrict__ q,    // [T, n_head, 128]
    const float* __restrict__ k,    // [T, n_head, 128]
    const float* __restrict__ v,    // [T, n_head, 128]
    const float* __restrict__ g1,   // [T, n_head, 128]
    const float* __restrict__ beta, // [T, n_head]
    float* __restrict__ state,      // [n_head, 128, 128]
    int n_head,
    int T,
    float* __restrict__ out,        // [T, n_head, 128]
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    constexpr int HD = 128;
    constexpr int NG = 256 / KJ;
    constexpr int IPT = HD / NG;
    const int h = blockIdx.x;
    if (h >= n_head) return;
    const int tid = threadIdx.x;
    const int jj = tid & (KJ - 1);
    const int gi = tid / KJ;
    const int j = blockIdx.y * KJ + jj;
    const int DI = n_head * HD;
    const float scale = rsqrtf((float)HD);

    __shared__ float s_q[HD], s_k[HD], s_dec[HD];
    __shared__ float sA[NG][KJ + 1], sO[NG][KJ + 1];

    float reg[IPT];
    float* S_g = state + (size_t)h * HD * HD;
    #pragma unroll
    for (int ii = 0; ii < IPT; ++ii) {
        reg[ii] = S_g[(size_t)(gi * IPT + ii) * HD + j];
    }

    float pq = 0.0f, pk = 0.0f, pd = 0.0f;
    if (tid < HD && T > 0) {
        size_t p0 = (size_t)h * HD + tid;
        pq = q[p0];
        pk = k[p0];
        pd = __expf(g1[p0]);
    }

    for (int t = 0; t < T; ++t) {
        size_t base = (size_t)t * DI + (size_t)h * HD;
        if (tid < HD) {
            s_q[tid] = pq;
            s_k[tid] = pk;
            s_dec[tid] = pd;
            if (t + 1 < T) {
                size_t nb = base + DI + tid;
                pq = q[nb];
                pk = k[nb];
                pd = __expf(g1[nb]);
            }
        }
        const float vj = v[base + j];
        const float b = beta[(size_t)t * n_head + h];
        __syncthreads();

        float A = 0.0f;
        #pragma unroll
        for (int ii = 0; ii < IPT; ++ii) {
            int i = gi * IPT + ii;
            float sv = reg[ii] * s_dec[i];
            reg[ii] = sv;
            A += sv * s_k[i];
        }
        sA[gi][jj] = A;
        __syncthreads();

        float At = 0.0f;
        #pragma unroll
        for (int g = 0; g < NG; ++g) At += sA[g][jj];
        const float dl = b * (vj - At);
        float O = 0.0f;
        #pragma unroll
        for (int ii = 0; ii < IPT; ++ii) {
            int i = gi * IPT + ii;
            float sv = reg[ii] + s_k[i] * dl;
            reg[ii] = sv;
            O += sv * s_q[i];
        }
        sO[gi][jj] = O;
        __syncthreads();

        if (gi == 0) {
            float Ot = 0.0f;
            #pragma unroll
            for (int g = 0; g < NG; ++g) Ot += sO[g][jj];
            out[base + j] = Ot * scale;
        }
    }

    #pragma unroll
    for (int ii = 0; ii < IPT; ++ii) {
        S_g[(size_t)(gi * IPT + ii) * HD + j] = reg[ii];
    }
}

extern "C" void launch_glm5_kda_recurrence(
    const float* q,
    const float* k,
    const float* v,
    const float* g1,
    const float* beta,
    float* state,
    int n_head,
    int head_dim,
    int T,
    float* out,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (n_head <= 0 || head_dim <= 0 || T <= 0) return;
    if (T == 1 && head_dim == 128) {
        dim3 grid(4, n_head);
        glm5_kda_recurrence_m1_kernel<<<grid, 32, 0, stream>>>(
            q, k, v, g1, beta, state, n_head, out, skip_dev
        );
        return;
    }
    if (T > 1 && head_dim == 128) {
        if (n_head <= 32) {
            dim3 grid(n_head, 8);
            glm5_kda_recurrence_split_kernel<16><<<grid, 256, 0, stream>>>(
                q, k, v, g1, beta, state, n_head, T, out, skip_dev
            );
        } else {
            dim3 grid(n_head, 4);
            glm5_kda_recurrence_split_kernel<32><<<grid, 256, 0, stream>>>(
                q, k, v, g1, beta, state, n_head, T, out, skip_dev
            );
        }
        return;
    }
    size_t smem_bytes = (size_t)(head_dim * head_dim + 3 * head_dim) * sizeof(float);
    launch_glm5_prepare_kda_smem(head_dim);
    glm5_kda_recurrence_kernel<<<n_head, head_dim, smem_bytes, stream>>>(
        q, k, v, g1, beta, state, n_head, head_dim, T, out, skip_dev
    );
}

__global__ void __launch_bounds__(128) glm5_kda_out_gate_q8_kernel(
    const float* __restrict__ o,      // [T, n_head, head_dim]
    const float* __restrict__ norm_w, // [head_dim]
    const float* __restrict__ g2,     // [T, n_head, head_dim]
    int head_dim,
    int n_head,
    float eps,
    float* __restrict__ gated_f32,    // nullable [T, n_head, head_dim]
    block_q8_1* __restrict__ gated_q8 // nullable [T, n_head * head_dim / 32]
) {
    const int ht = blockIdx.x; // [0 .. T * n_head - 1]
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ float s_red[4];
    __shared__ float s_inv;

    const float* ot = o + (size_t)ht * head_dim;
    const float* gt = g2 + (size_t)ht * head_dim;
    float ss = 0.0f;
    for (int e = tid; e < head_dim; e += 128) {
        float v = ot[e];
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = s_red[0] + s_red[1] + s_red[2] + s_red[3];
        s_inv = rsqrtf(tot / (float)head_dim + eps);
    }
    __syncthreads();
    const float inv = s_inv;

    float* gf = gated_f32 ? (gated_f32 + (size_t)ht * head_dim) : nullptr;
    block_q8_1* gq = gated_q8 ? (gated_q8 + (size_t)ht * (head_dim >> 5)) : nullptr;
    for (int base = 0; base < head_dim; base += 128) {
        int e = base + tid;
        float val = 0.0f;
        if (e < head_dim) {
            float gv = gt[e];
            val = ot[e] * inv * norm_w[e] * (1.0f / (1.0f + __expf(-gv)));
            if (gf) gf[e] = val;
        }
        if (gq && (base + warp * 32) < head_dim) {
            warp_quantize_q8_1_store(val, gq + ((base >> 5) + warp), lane);
        }
    }
}

extern "C" void launch_glm5_kda_out_gate_q8(
    const float* o,
    const float* norm_w,
    const float* g2,
    int head_dim,
    int n_head,
    int T,
    float eps,
    float* gated_f32,
    void* gated_q8,
    cudaStream_t stream
) {
    int total = n_head * T;
    if (total <= 0) return;
    glm5_kda_out_gate_q8_kernel<<<total, 128, 0, stream>>>(
        o, norm_w, g2, head_dim, n_head, eps, gated_f32, reinterpret_cast<block_q8_1*>(gated_q8)
    );
}

// Fused M=1 KDA prep (grid = [n_head, 11], block = 128):
// roles 0..2: conv1d(d_conv=4) + SiLU + head L2 norm (q, k) -> out[0..2]
// roles 3..6: Q8_1(fa) + Q8_0 f_b + dt_bias + exp gate -> g1
// roles 7..10: Q8_1(ga) + Q8_0 g_b -> g2
__global__ void __launch_bounds__(128) glm5_kda_prep_m1_kernel(
    const float* __restrict__ q_proj,
    const float* __restrict__ k_proj,
    const float* __restrict__ v_proj,
    const float* __restrict__ conv_w_q,
    const float* __restrict__ conv_w_k,
    const float* __restrict__ conv_w_v,
    float* __restrict__ conv_st_q,
    float* __restrict__ conv_st_k,
    float* __restrict__ conv_st_v,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    float* __restrict__ v_out,
    const float* __restrict__ f_a,
    const float* __restrict__ g_a,
    const block_q8_0* __restrict__ f_b,
    const block_q8_0* __restrict__ g_b,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float lower_bound,
    float* __restrict__ g1_out,
    float* __restrict__ g2_out,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    constexpr int HD = 128;
    const int h = blockIdx.x;
    const int role = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    if (role < 3) {
        const float* xp_in = (role == 0) ? q_proj : (role == 1) ? k_proj : v_proj;
        const float* cw    = (role == 0) ? conv_w_q : (role == 1) ? conv_w_k : conv_w_v;
        float* st_base     = (role == 0) ? conv_st_q : (role == 1) ? conv_st_k : conv_st_v;
        float* out_base    = (role == 0) ? q_out : (role == 1) ? k_out : v_out;

        const int c = h * HD + tid;
        float* st = st_base + (size_t)c * 3;
        float s0 = st[0], s1 = st[1], s2 = st[2];
        float xn = xp_in[c];
        float acc = cw[(size_t)c * 4 + 0] * s0
                  + cw[(size_t)c * 4 + 1] * s1
                  + cw[(size_t)c * 4 + 2] * s2
                  + cw[(size_t)c * 4 + 3] * xn;
        st[0] = s1;
        st[1] = s2;
        st[2] = xn;
        float y = acc / (1.0f + expf(-acc));
        if (role < 2) {
            __shared__ float s_red[4];
            float ws = warp_reduce_sum(y * y);
            if (lane == 0) s_red[warp] = ws;
            __syncthreads();
            float tot = s_red[0] + s_red[1] + s_red[2] + s_red[3];
            y *= rsqrtf(tot + 1e-6f);
        }
        out_base[c] = y;
        return;
    }

    const bool is_g1 = role < 7;
    const int quarter = (role - 3) & 3;
    const float* in = is_g1 ? f_a : g_a;
    const block_q8_0* w_mat = is_g1 ? f_b : g_b;
    __shared__ __align__(16) block_q8_1 s_q8[4];
    warp_quantize_q8_1_store(in[tid], s_q8 + warp, lane);
    __syncthreads();

    const int rbase = quarter * 32 + warp * 8;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int c = h * HD + rbase + j;
        const float acc = dot_row_q8_0_sub(w_mat + (size_t)c * 4, s_q8, HD, lane);
        if (lane == 0) {
            if (is_g1) {
                const float v = acc + dt_bias[c];
                g1_out[c] = lower_bound / (1.0f + expf(v * ssm_a[h]));
            } else {
                g2_out[c] = acc;
            }
        }
    }
}

extern "C" void launch_glm5_kda_prep_m1(
    const float* q_proj,
    const float* k_proj,
    const float* v_proj,
    const float* conv_w_q,
    const float* conv_w_k,
    const float* conv_w_v,
    float* conv_st_q,
    float* conv_st_k,
    float* conv_st_v,
    float* q_out,
    float* k_out,
    float* v_out,
    const float* f_a,
    const float* g_a,
    const void* f_b,
    const void* g_b,
    const float* dt_bias,
    const float* ssm_a,
    float lower_bound,
    int n_head,
    float* g1_out,
    float* g2_out,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (n_head <= 0) return;
    dim3 grid(n_head, 11);
    glm5_kda_prep_m1_kernel<<<grid, 128, 0, stream>>>(
        q_proj, k_proj, v_proj, conv_w_q, conv_w_k, conv_w_v,
        conv_st_q, conv_st_k, conv_st_v, q_out, k_out, v_out,
        f_a, g_a,
        reinterpret_cast<const block_q8_0*>(f_b),
        reinterpret_cast<const block_q8_0*>(g_b),
        dt_bias, ssm_a, lower_bound, g1_out, g2_out, skip_dev
    );
}

// Fused single-pass KDA recurrence + RMSNorm + SiLU gate + Q8_1 (grid = n_head, block = 256)
__global__ void __launch_bounds__(256) glm5_kda_rec_gate_q8_m1_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ g1,
    const float* __restrict__ beta_raw,
    float* __restrict__ state,
    const float* __restrict__ g2,
    const float* __restrict__ norm_w,
    float eps,
    int n_head,
    float* __restrict__ gated_f32_opt,
    block_q8_1* __restrict__ gated_q8,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    constexpr int HD = 128;
    const int h = blockIdx.x;
    if (h >= n_head) return;
    const int t = threadIdx.x;
    const int j = t & 127;
    const int half = t >> 7;
    const int lane = t & 31;
    const int warp = t >> 5;

    __shared__ float s_q[HD];
    __shared__ float s_k[HD];
    __shared__ float s_dec[HD];
    __shared__ float s_del[HD];
    __shared__ float s_acc[2][HD];
    __shared__ float s_red[8];

    const size_t h_off = (size_t)h * HD;
    if (t < HD) {
        s_q[t] = q[h_off + t];
        s_k[t] = k[h_off + t];
        s_dec[t] = expf(g1[h_off + t]);
    }
    const float b = 1.0f / (1.0f + expf(-beta_raw[h]));
    __syncthreads();

    float* S_h = state + (size_t)h * (HD * HD);
    float reg[64];
    float A = 0.0f;
    #pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float sv = S_h[i * HD + j] * s_dec[i];
        reg[ii] = sv;
        A += sv * s_k[i];
    }
    s_acc[half][j] = A;
    __syncthreads();
    if (half == 0) {
        s_del[j] = b * (v[h_off + j] - (s_acc[0][j] + s_acc[1][j]));
    }
    __syncthreads();
    const float dl = s_del[j];
    float O = 0.0f;
    #pragma unroll
    for (int ii = 0; ii < 64; ++ii) {
        const int i = half * 64 + ii;
        const float s2 = reg[ii] + s_k[i] * dl;
        S_h[i * HD + j] = s2;
        O += s2 * s_q[i];
    }
    __syncthreads();
    s_acc[half][j] = O;
    __syncthreads();

    const float o = (s_acc[0][j] + s_acc[1][j]) * rsqrtf((float)HD);
    float ss = (half == 0) ? (o * o) : 0.0f;
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    const float tot = s_red[0] + s_red[1] + s_red[2] + s_red[3];
    const float inv = rsqrtf(tot / (float)HD + eps);
    if (half == 0) {
        const float gv = o * inv * norm_w[j] * (1.0f / (1.0f + expf(-g2[h_off + j])));
        if (gated_f32_opt != nullptr) gated_f32_opt[h_off + j] = gv;
        if (gated_q8 != nullptr) warp_quantize_q8_1_store(gv, gated_q8 + (h * 4 + warp), lane);
    }
}

extern "C" void launch_glm5_kda_rec_gate_q8_m1(
    const float* q,
    const float* k,
    const float* v,
    const float* g1,
    const float* beta_raw,
    float* state,
    const float* g2,
    const float* norm_w,
    float eps,
    int n_head,
    float* gated_f32_opt,
    void* gated_q8,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (n_head <= 0) return;
    glm5_kda_rec_gate_q8_m1_kernel<<<n_head, 256, 0, stream>>>(
        q, k, v, g1, beta_raw, state, g2, norm_w, eps, n_head,
        gated_f32_opt, reinterpret_cast<block_q8_1*>(gated_q8), skip_dev
    );
}

// -----------------------------------------------------------------------------
// DSA (Indexer + INT8 lat8 Cache + Parallel Top-K + Absorbed MLA) Kernels
// -----------------------------------------------------------------------------

__device__ __forceinline__ void glm5_dsa_prep_kv_ik_device(
    const float* __restrict__ kv_raw,
    const float* __restrict__ kv_norm_w,
    uint8_t* __restrict__ vram_lat8,
    uint8_t* __restrict__ ram_lat8,
    int res_cap,
    const float* __restrict__ ik_raw,
    const float* __restrict__ ik_norm_w,
    const float* __restrict__ ik_norm_b,
    float* __restrict__ ik_cache,
    const float* __restrict__ ig_raw,
    float* __restrict__ ig_cache,
    int kv_lora,
    int idx_key,
    int t,
    int pos,
    int ring_cap,
    float eps
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ float s_red[8];
    __shared__ float s_inv;

    // 1. kv RMSNorm -> INT8 lat8 pack (kv_lora = 512, 16 blocks of 32)
    const float* kvt = kv_raw + (size_t)t * kv_lora;
    float ss = 0.0f;
    for (int e = tid; e < kv_lora; e += 256) {
        float v = kvt[e];
        ss += v * v;
    }
    ss = warp_reduce_sum(ss);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        for (int w = 0; w < 8; ++w) tot += s_red[w];
        s_inv = rsqrtf(tot / (float)kv_lora + eps);
    }
    __syncthreads();
    const float inv_kv = s_inv;

    size_t lat_stride = (size_t)(kv_lora + kv_lora / 16);
    uint8_t* dst_v = (vram_lat8 && (res_cap <= 0 || pos < res_cap)) ? (vram_lat8 + (size_t)pos * lat_stride) : nullptr;
    uint8_t* dst_r = ram_lat8 ? (ram_lat8 + (size_t)pos * lat_stride) : nullptr;
    for (int base = 0; base < kv_lora; base += 256) {
        int blk = (base >> 5) + warp;
        int e = base + tid;
        float v = (e < kv_lora) ? (kvt[e] * inv_kv * kv_norm_w[e]) : 0.0f;
        float amax = warp_reduce_max(fabsf(v));
        float d = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int q = __float2int_rn(v * inv_d);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        uint16_t d16 = f32_to_fp16_bits(d);
        if (e < kv_lora) {
            if (dst_v) {
                dst_v[e] = (uint8_t)(int8_t)q;
                if (lane == 0) reinterpret_cast<uint16_t*>(dst_v + kv_lora)[blk] = d16;
            }
            if (dst_r) {
                dst_r[e] = (uint8_t)(int8_t)q;
                if (lane == 0) reinterpret_cast<uint16_t*>(dst_r + kv_lora)[blk] = d16;
            }
        }
    }

    // 2. ik LayerNorm + ig copy (when provided)
    if (ik_raw && ik_cache && idx_key > 0) {
        int slot = (ring_cap > 0) ? (pos % ring_cap) : pos;
        const float* ikt = ik_raw + (size_t)t * idx_key;
        float sum = 0.0f, sq = 0.0f;
        if (tid < idx_key) {
            float v = ikt[tid];
            sum = v;
            sq = v * v;
        }
        // idx_key = 128 -> first 4 warps
        sum = warp_reduce_sum(sum);
        sq = warp_reduce_sum(sq);
        if (lane == 0 && warp < 4) {
            s_red[warp] = sum;
            s_red[4 + warp] = sq;
        }
        __syncthreads();
        if (tid < idx_key) {
            float tot_s = s_red[0] + s_red[1] + s_red[2] + s_red[3];
            float tot_q = s_red[4] + s_red[5] + s_red[6] + s_red[7];
            float mu = tot_s / (float)idx_key;
            float inv_ik = rsqrtf(tot_q / (float)idx_key - mu * mu + eps);
            ik_cache[(size_t)slot * idx_key + tid] = (ikt[tid] - mu) * inv_ik * ik_norm_w[tid] + ik_norm_b[tid];
            if (ig_raw && ig_cache) {
                ig_cache[(size_t)slot * idx_key + tid] = ig_raw[(size_t)t * idx_key + tid];
            }
        }
    }
}

__global__ void __launch_bounds__(256) glm5_dsa_prep_kv_ik_kernel(
    const float* __restrict__ kv_raw,     // [T, kv_lora]
    const float* __restrict__ kv_norm_w,  // [kv_lora]
    uint8_t* __restrict__ vram_lat8,      // nullable [res_cap * lat_stride]
    uint8_t* __restrict__ ram_lat8,       // nullable [total_cap * lat_stride]
    int res_cap,
    const float* __restrict__ ik_raw,     // nullable [T, idx_key]
    const float* __restrict__ ik_norm_w,  // nullable [idx_key]
    const float* __restrict__ ik_norm_b,  // nullable [idx_key]
    float* __restrict__ ik_cache,         // [T_or_cap, idx_key]
    const float* __restrict__ ig_raw,     // nullable [T, idx_key]
    float* __restrict__ ig_cache,         // [T_or_cap, idx_key]
    int kv_lora,
    int idx_key,
    int pos0,
    int ring_cap,                         // 0 = linear [pos], >0 = [pos % ring_cap]
    float eps
) {
    const int t = blockIdx.x;
    glm5_dsa_prep_kv_ik_device(
        kv_raw, kv_norm_w, vram_lat8, ram_lat8, res_cap,
        ik_raw, ik_norm_w, ik_norm_b, ik_cache, ig_raw, ig_cache,
        kv_lora, idx_key, t, pos0 + t, ring_cap, eps
    );
}

__global__ void __launch_bounds__(256) glm5_dsa_prep_kv_ik_dyn_kernel(
    const float* __restrict__ kv_raw,
    const float* __restrict__ kv_norm_w,
    uint8_t* __restrict__ vram_lat8,
    uint8_t* __restrict__ ram_lat8,
    int res_cap,
    const float* __restrict__ ik_raw,
    const float* __restrict__ ik_norm_w,
    const float* __restrict__ ik_norm_b,
    float* __restrict__ ik_cache,
    const float* __restrict__ ig_raw,
    float* __restrict__ ig_cache,
    int kv_lora,
    int idx_key,
    const int32_t* __restrict__ pos_arr,
    int ring_cap,
    float eps
) {
    const int t = blockIdx.x;
    glm5_dsa_prep_kv_ik_device(
        kv_raw, kv_norm_w, vram_lat8, ram_lat8, res_cap,
        ik_raw, ik_norm_w, ik_norm_b, ik_cache, ig_raw, ig_cache,
        kv_lora, idx_key, t, pos_arr[t], ring_cap, eps
    );
}

extern "C" void launch_glm5_dsa_prep_kv_ik(
    const float* kv_raw,
    const float* kv_norm_w,
    void* vram_lat8,
    void* ram_lat8,
    int res_cap,
    const float* ik_raw,
    const float* ik_norm_w,
    const float* ik_norm_b,
    float* ik_cache,
    const float* ig_raw,
    float* ig_cache,
    int kv_lora,
    int idx_key,
    int pos0,
    int ring_cap,
    float eps,
    int T,
    cudaStream_t stream
) {
    if (T <= 0) return;
    glm5_dsa_prep_kv_ik_kernel<<<T, 256, 0, stream>>>(
        kv_raw, kv_norm_w,
        reinterpret_cast<uint8_t*>(vram_lat8),
        reinterpret_cast<uint8_t*>(ram_lat8),
        res_cap, ik_raw, ik_norm_w, ik_norm_b, ik_cache,
        ig_raw, ig_cache, kv_lora, idx_key, pos0, ring_cap, eps
    );
}

extern "C" void launch_glm5_dsa_prep_kv_ik_dyn(
    const float* kv_raw,
    const float* kv_norm_w,
    void* vram_lat8,
    void* ram_lat8,
    int res_cap,
    const float* ik_raw,
    const float* ik_norm_w,
    const float* ik_norm_b,
    float* ik_cache,
    const float* ig_raw,
    float* ig_cache,
    int kv_lora,
    int idx_key,
    const int32_t* pos_arr,
    int ring_cap,
    float eps,
    int T,
    cudaStream_t stream
) {
    if (T <= 0) return;
    glm5_dsa_prep_kv_ik_dyn_kernel<<<T, 256, 0, stream>>>(
        kv_raw, kv_norm_w,
        reinterpret_cast<uint8_t*>(vram_lat8),
        reinterpret_cast<uint8_t*>(ram_lat8),
        res_cap, ik_raw, ik_norm_w, ik_norm_b, ik_cache,
        ig_raw, ig_cache, kv_lora, idx_key, pos_arr, ring_cap, eps
    );
}

__device__ __forceinline__ uint16_t dsa_pool_compute_elem(
    const float* __restrict__ ik_cache,
    const float* __restrict__ ig_cache,
    const float* __restrict__ ape,
    int key_dim,
    int kpool,
    int ring_cap,
    int p,
    int e
) {
    float logits[16];
    float mx = -INFINITY;
    for (int m = 0; m < kpool; ++m) {
        int pos = p * kpool + m;
        int slot = (ring_cap > 0) ? (pos % ring_cap) : pos;
        logits[m] = ig_cache[(size_t)slot * key_dim + e] + ape[(size_t)m * key_dim + e];
        mx = fmaxf(mx, logits[m]);
    }
    float denom = 0.0f;
    for (int m = 0; m < kpool; ++m) {
        logits[m] = expf(logits[m] - mx);
        denom += logits[m];
    }
    float acc = 0.0f;
    for (int m = 0; m < kpool; ++m) {
        int pos = p * kpool + m;
        int slot = (ring_cap > 0) ? (pos % ring_cap) : pos;
        acc += (logits[m] / denom) * ik_cache[(size_t)slot * key_dim + e];
    }
    return f32_to_fp16_bits(acc);
}

__global__ void glm5_dsa_pool_kernel(
    const float* __restrict__ ik_cache, // [cap_or_ring, key_dim]
    const float* __restrict__ ig_cache, // [cap_or_ring, key_dim]
    const float* __restrict__ ape,      // [kpool, key_dim]
    int key_dim,
    int kpool,
    int pool0,
    int n_new_pools,
    int ring_cap,
    uint16_t* __restrict__ pooled_f16   // [total_pools, key_dim]
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= key_dim * n_new_pools) return;
    int p = pool0 + idx / key_dim;
    int e = idx % key_dim;
    pooled_f16[(size_t)p * key_dim + e] = dsa_pool_compute_elem(
        ik_cache, ig_cache, ape, key_dim, kpool, ring_cap, p, e
    );
}

__global__ void __launch_bounds__(128) glm5_dsa_pool_m1_dyn_kernel(
    const float* __restrict__ ik_cache,
    const float* __restrict__ ig_cache,
    const float* __restrict__ ape,
    int key_dim,
    int kpool,
    const int32_t* __restrict__ pos_arr,
    int ring_cap,
    uint16_t* __restrict__ pooled_f16
) {
    const int pos = pos_arr[0];
    if (pos < 0 || (pos + 1) % kpool != 0) return;
    const int p = pos / kpool;
    for (int e = threadIdx.x; e < key_dim; e += blockDim.x) {
        pooled_f16[(size_t)p * key_dim + e] = dsa_pool_compute_elem(
            ik_cache, ig_cache, ape, key_dim, kpool, ring_cap, p, e
        );
    }
}

extern "C" void launch_glm5_dsa_pool(
    const float* ik_cache,
    const float* ig_cache,
    const float* ape,
    int key_dim,
    int kpool,
    int pool0,
    int n_new_pools,
    int ring_cap,
    void* pooled_f16,
    cudaStream_t stream
) {
    if (n_new_pools <= 0 || key_dim <= 0) return;
    int tot = key_dim * n_new_pools;
    glm5_dsa_pool_kernel<<<(tot + 255) / 256, 256, 0, stream>>>(
        ik_cache, ig_cache, ape, key_dim, kpool, pool0, n_new_pools, ring_cap,
        reinterpret_cast<uint16_t*>(pooled_f16)
    );
}

extern "C" void launch_glm5_dsa_pool_m1_dyn(
    const float* ik_cache,
    const float* ig_cache,
    const float* ape,
    int key_dim,
    int kpool,
    const int32_t* pos_arr,
    int ring_cap,
    void* pooled_f16,
    cudaStream_t stream
) {
    if (key_dim <= 0) return;
    glm5_dsa_pool_m1_dyn_kernel<<<1, 128, 0, stream>>>(
        ik_cache, ig_cache, ape, key_dim, kpool, pos_arr, ring_cap,
        reinterpret_cast<uint16_t*>(pooled_f16)
    );
}

// Fused M=1 DSA prep (grid = 3, block = 512):
// block 0: qr RMSNorm + q8_1; block 1: kv RMSNorm -> lat8 VRAM+RAM; block 2: ik LN + ig copy + dsa_pool
__global__ void __launch_bounds__(512) glm5_dsa_prep_m1_dyn_kernel(
    float* __restrict__ qr_inout,
    const float* __restrict__ q_norm_w,
    block_q8_1* __restrict__ qr_q8_out,
    int q_lora,
    const float* __restrict__ kv_raw,
    const float* __restrict__ kv_norm_w,
    uint8_t* __restrict__ vram_lat8,
    uint8_t* __restrict__ ram_lat8,
    int res_cap,
    int kv_lora,
    const float* __restrict__ ik_raw,
    const float* __restrict__ ik_norm_w,
    const float* __restrict__ ik_norm_b,
    float* __restrict__ ik_cache,
    const float* __restrict__ ig_raw,
    float* __restrict__ ig_cache,
    const float* __restrict__ ape,
    int idx_key,
    int kpool,
    int ring_cap,
    const int32_t* __restrict__ pos_arr,
    uint16_t* __restrict__ pooled_f16,
    float eps,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_red[32];
    __shared__ float s_inv;

    if (blockIdx.x == 0) {
        if (qr_inout == nullptr || q_lora <= 0) return;
        float ss = 0.0f;
        for (int e = tid; e < q_lora; e += 512) {
            float v = qr_inout[e];
            ss += v * v;
        }
        ss = warp_reduce_sum(ss);
        if (lane == 0) s_red[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < 16; ++w) tot += s_red[w];
            s_inv = rsqrtf(tot / (float)q_lora + eps);
        }
        __syncthreads();
        const float inv = s_inv;
        for (int base = 0; base < q_lora; base += 512) {
            int e = base + tid;
            float val = 0.0f;
            if (e < q_lora) {
                val = qr_inout[e] * inv * q_norm_w[e];
                qr_inout[e] = val;
            }
            if (qr_q8_out != nullptr && (base + warp * 32) < q_lora) {
                warp_quantize_q8_1_store(val, qr_q8_out + ((base >> 5) + warp), lane);
            }
        }
        return;
    }

    const int pos = pos_arr[0];
    if (blockIdx.x == 1) {
        if (kv_raw == nullptr || kv_lora <= 0) return;
        float ss = 0.0f;
        for (int e = tid; e < kv_lora; e += 512) {
            float v = kv_raw[e];
            ss += v * v;
        }
        ss = warp_reduce_sum(ss);
        if (lane == 0) s_red[warp] = ss;
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < 16; ++w) tot += s_red[w];
            s_inv = rsqrtf(tot / (float)kv_lora + eps);
        }
        __syncthreads();
        const float inv_kv = s_inv;
        const size_t lat_stride = (size_t)(kv_lora + kv_lora / 16);
        uint8_t* dst_v = (vram_lat8 && (res_cap <= 0 || pos < res_cap)) ? (vram_lat8 + (size_t)pos * lat_stride) : nullptr;
        uint8_t* dst_r = ram_lat8 ? (ram_lat8 + (size_t)pos * lat_stride) : nullptr;
        for (int base = 0; base < kv_lora; base += 512) {
            int blk = (base >> 5) + warp;
            int e = base + tid;
            if ((base + warp * 32) < kv_lora) {
                float v = (e < kv_lora) ? (kv_raw[e] * inv_kv * kv_norm_w[e]) : 0.0f;
                float amax = warp_reduce_max(fabsf(v));
                float d = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
                float inv_d = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
                int q = __float2int_rn(v * inv_d);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                uint16_t d16 = f32_to_fp16_bits(d);
                if (e < kv_lora) {
                    if (dst_v) {
                        dst_v[e] = (uint8_t)(int8_t)q;
                        if (lane == 0) reinterpret_cast<uint16_t*>(dst_v + kv_lora)[blk] = d16;
                    }
                    if (dst_r) {
                        dst_r[e] = (uint8_t)(int8_t)q;
                        if (lane == 0) reinterpret_cast<uint16_t*>(dst_r + kv_lora)[blk] = d16;
                    }
                }
            }
        }
        return;
    }

    if (ik_raw != nullptr && ik_cache != nullptr && idx_key > 0) {
        const int slot = (ring_cap > 0) ? (pos % ring_cap) : pos;
        float sum = 0.0f, sq = 0.0f;
        if (tid < idx_key) {
            float v = ik_raw[tid];
            sum = v;
            sq = v * v;
        }
        sum = warp_reduce_sum(sum);
        sq = warp_reduce_sum(sq);
        if (lane == 0) {
            s_red[warp] = sum;
            s_red[16 + warp] = sq;
        }
        __syncthreads();
        if (tid < idx_key) {
            float tot_s = 0.0f, tot_q = 0.0f;
            #pragma unroll
            for (int w = 0; w < 16; ++w) {
                tot_s += s_red[w];
                tot_q += s_red[16 + w];
            }
            float mu = tot_s / (float)idx_key;
            float inv_ik = rsqrtf(tot_q / (float)idx_key - mu * mu + eps);
            ik_cache[(size_t)slot * idx_key + tid] = (ik_raw[tid] - mu) * inv_ik * ik_norm_w[tid] + ik_norm_b[tid];
            if (ig_raw && ig_cache) {
                ig_cache[(size_t)slot * idx_key + tid] = ig_raw[tid];
            }
        }
        if (pooled_f16 != nullptr && pos >= 0 && (pos + 1) % kpool == 0) {
            __syncthreads();
            __threadfence_block();
            if (tid < idx_key) {
                const int p = pos / kpool;
                pooled_f16[(size_t)p * idx_key + tid] = dsa_pool_compute_elem(
                    ik_cache, ig_cache, ape, idx_key, kpool, ring_cap, p, tid
                );
            }
        }
    }
}

extern "C" void launch_glm5_dsa_prep_m1_dyn(
    float* qr_inout,
    const float* q_norm_w,
    void* qr_q8_out,
    int q_lora,
    const float* kv_raw,
    const float* kv_norm_w,
    void* vram_lat8,
    void* ram_lat8,
    int res_cap,
    int kv_lora,
    const float* ik_raw,
    const float* ik_norm_w,
    const float* ik_norm_b,
    float* ik_cache,
    const float* ig_raw,
    float* ig_cache,
    const float* ape,
    int idx_key,
    int kpool,
    int ring_cap,
    const int32_t* pos_arr,
    void* pooled_f16,
    float eps,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    glm5_dsa_prep_m1_dyn_kernel<<<3, 512, 0, stream>>>(
        qr_inout, q_norm_w, reinterpret_cast<block_q8_1*>(qr_q8_out), q_lora,
        kv_raw, kv_norm_w,
        reinterpret_cast<uint8_t*>(vram_lat8),
        reinterpret_cast<uint8_t*>(ram_lat8),
        res_cap, kv_lora, ik_raw, ik_norm_w, ik_norm_b, ik_cache, ig_raw, ig_cache,
        ape, idx_key, kpool, ring_cap, pos_arr, reinterpret_cast<uint16_t*>(pooled_f16),
        eps, skip_dev
    );
}

// Coalesced wk_b [n_head, kv_lora, qk_nope] * q [T, n_head, qk_nope] -> q_abs [T, n_head, kv_lora]
// Supports F16 or F32 wk_b. 4 warps/block -> 4 rows of c per block.
__global__ void __launch_bounds__(128) glm5_dsa_q_absorb_kernel(
    const void* __restrict__ wk_b,
    int32_t wk_type,
    const float* __restrict__ q,
    float* __restrict__ q_abs,
    int qk_nope,
    int kv_lora,
    int n_head
) {
    const int c = blockIdx.x * 4 + threadIdx.y;
    const int h = blockIdx.y;
    const int t = blockIdx.z;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;

    __shared__ __align__(16) float s_q[512]; // qk_nope = 256
    const float* qh = q + ((size_t)t * n_head + h) * qk_nope;
    for (int e = tid; e < qk_nope; e += 128) {
        s_q[e] = qh[e];
    }
    __syncthreads();

    if (c >= kv_lora) return;

    float acc = 0.0f;
    size_t row_off = ((size_t)h * kv_lora + c) * qk_nope;
    if (wk_type == GGML_TYPE_F16 && (qk_nope & 7) == 0 && (reinterpret_cast<uintptr_t>(wk_b) & 15u) == 0u) {
        const uint4* wr8 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(wk_b) + row_off);
        const float4* sq4 = reinterpret_cast<const float4*>(s_q);
        const int n8 = qk_nope >> 3;
        for (int e8 = lane; e8 < n8; e8 += 32) {
            uint4 u = wr8[e8];
            float4 q0 = sq4[e8 * 2 + 0];
            float4 q1 = sq4[e8 * 2 + 1];
            acc += fp16_bits_to_f32((uint16_t)u.x)         * q0.x
                 + fp16_bits_to_f32((uint16_t)(u.x >> 16)) * q0.y
                 + fp16_bits_to_f32((uint16_t)u.y)         * q0.z
                 + fp16_bits_to_f32((uint16_t)(u.y >> 16)) * q0.w
                 + fp16_bits_to_f32((uint16_t)u.z)         * q1.x
                 + fp16_bits_to_f32((uint16_t)(u.z >> 16)) * q1.y
                 + fp16_bits_to_f32((uint16_t)u.w)         * q1.z
                 + fp16_bits_to_f32((uint16_t)(u.w >> 16)) * q1.w;
        }
    } else if (wk_type == GGML_TYPE_F16) {
        const uint16_t* wr = reinterpret_cast<const uint16_t*>(wk_b) + row_off;
        for (int e = lane; e < qk_nope; e += 32) {
            acc += fp16_bits_to_f32(wr[e]) * s_q[e];
        }
    } else {
        const float* wr = reinterpret_cast<const float*>(wk_b) + row_off;
        for (int e = lane; e < qk_nope; e += 32) {
            acc += wr[e] * s_q[e];
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        q_abs[((size_t)t * n_head + h) * kv_lora + c] = acc;
    }
}

__device__ __forceinline__ void mma16816_f32_f16_ops(
    float (&c)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
}

__device__ __forceinline__ uint32_t f32x2_to_fp16x2_u32(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    uint32_t u;
    memcpy(&u, &h, 4);
    return u;
}

__device__ __forceinline__ void ldmatrix_x4_f16(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_f16(
    uint32_t& r0, uint32_t& r1, const void* smem_ptr
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}

__device__ __forceinline__ void ldmatrix_x2_trans_f16(
    uint32_t& r0, uint32_t& r1, const void* smem_ptr
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(addr)
    );
}

__global__ void __launch_bounds__(256) glm5_dsa_q_absorb_mma_kernel(
    const uint16_t* __restrict__ wk_b, // [n_head, 512, 256] row-major F16
    const float* __restrict__ q,       // [T, n_head, 256] F32
    float* __restrict__ q_abs,         // [T, n_head, 512] F32
    int n_head,
    int T
) {
    const int t0 = (int)blockIdx.x * 64;
    const int h = (int)blockIdx.y;
    const int c_half = (int)blockIdx.z; // 0 -> [0..256), 1 -> [256..512)
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5; // 0..7
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int k_pair = (lane & 3) * 2;

    constexpr int LD = 256 + 8; // 16B aligned + conflict-free stride
    __shared__ __align__(16) uint16_t s_q[64][LD];
    __shared__ __align__(16) uint16_t s_w[256][24];
    __shared__ float s_inv[64];

    // Stage 64 tokens x 256 dims into FP16 with per-token max-abs scaling
    #pragma unroll 1
    for (int p = 0; p < 8; ++p) {
        const int r = warp * 8 + p;
        const int gt = t0 + r;
        if (gt < T) {
            const float* qr = q + ((size_t)gt * n_head + h) * 256;
            const float4* q4 = reinterpret_cast<const float4*>(qr + lane * 8);
            const float4 a0 = q4[0], a1 = q4[1];
            const float qv[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float mx = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(qv[k]));
            const float amax = warp_reduce_max(mx);
            const float f = (amax > 0.0f) ? (16.0f / amax) : 0.0f;
            if (lane == 0) s_inv[r] = amax * 0.0625f;
            const uint32_t u0 = f32x2_to_fp16x2_u32(qv[0] * f, qv[1] * f);
            const uint32_t u1 = f32x2_to_fp16x2_u32(qv[2] * f, qv[3] * f);
            const uint32_t u2 = f32x2_to_fp16x2_u32(qv[4] * f, qv[5] * f);
            const uint32_t u3 = f32x2_to_fp16x2_u32(qv[6] * f, qv[7] * f);
            *reinterpret_cast<uint4*>(&s_q[r][lane * 8]) = make_uint4(u0, u1, u2, u3);
        } else {
            if (lane == 0) s_inv[r] = 0.0f;
            *reinterpret_cast<uint4*>(&s_q[r][lane * 8]) = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int nj = 0; nj < 8; ++nj)
            acc[mi][nj][0] = acc[mi][nj][1] = acc[mi][nj][2] = acc[mi][nj][3] = 0.0f;

    const int wm = (warp >> 2) * 32;
    const int wn = (warp & 3) * 64;
    const int c_off = c_half * 256;
    const uint16_t* wk_h = wk_b + (size_t)h * 512 * 256;

    #pragma unroll 1
    for (int k0 = 0; k0 < 256; k0 += 16) {
        const uint4* w4 = reinterpret_cast<const uint4*>(wk_h + (size_t)(c_off + tid) * 256 + k0);
        *reinterpret_cast<uint4*>(&s_w[tid][0]) = w4[0];
        *reinterpret_cast<uint4*>(&s_w[tid][8]) = w4[1];
        __syncthreads();

        uint32_t a_frag[2][4];
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            ldmatrix_x4_f16(
                a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                &s_q[wm + mi * 16 + (lane & 15)][k0 + (lane >> 4) * 8]
            );
        }
        #pragma unroll
        for (int nj = 0; nj < 8; ++nj) {
            uint32_t b0, b1;
            ldmatrix_x2_f16(b0, b1, &s_w[wn + nj * 8 + (lane & 7)][((lane >> 3) & 1) * 8]);
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                mma16816_f32_f16_ops(acc[mi][nj], a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3], b0, b1);
            }
        }
        __syncthreads();
    }

    const int c_base = c_off + wn;
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = wm + mi * 16 + gid;
        const int r1 = r0 + 8;
        const int gt0 = t0 + r0;
        const int gt1 = t0 + r1;
        const float inv0 = s_inv[r0];
        const float inv1 = s_inv[r1];
        if (gt0 < T) {
            float* o0 = q_abs + ((size_t)gt0 * n_head + h) * 512;
            #pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                const int c0 = c_base + nj * 8 + k_pair;
                *reinterpret_cast<float2*>(&o0[c0]) = make_float2(acc[mi][nj][0] * inv0, acc[mi][nj][1] * inv0);
            }
        }
        if (gt1 < T) {
            float* o1 = q_abs + ((size_t)gt1 * n_head + h) * 512;
            #pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                const int c0 = c_base + nj * 8 + k_pair;
                *reinterpret_cast<float2*>(&o1[c0]) = make_float2(acc[mi][nj][2] * inv1, acc[mi][nj][3] * inv1);
            }
        }
    }
}

extern "C" void launch_glm5_dsa_q_absorb(
    const void* wk_b,
    int32_t wk_type,
    const float* q,
    float* q_abs,
    int qk_nope,
    int kv_lora,
    int n_head,
    int T,
    cudaStream_t stream
) {
    if (T <= 0 || kv_lora <= 0) return;
    if (T > 1 && wk_type == GGML_TYPE_F16 && qk_nope == 256 && kv_lora == 512
        && (reinterpret_cast<uintptr_t>(wk_b) & 15u) == 0u) {
        dim3 grid((T + 63) / 64, n_head, 2);
        glm5_dsa_q_absorb_mma_kernel<<<grid, 256, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(wk_b), q, q_abs, n_head, T
        );
        return;
    }
    dim3 grid((kv_lora + 3) / 4, n_head, T);
    dim3 block(32, 4);
    glm5_dsa_q_absorb_kernel<<<grid, block, 0, stream>>>(
        wk_b, wk_type, q, q_abs, qk_nope, kv_lora, n_head
    );
}

__device__ __forceinline__ float dsa_score_pool_row(
    const uint16_t* s_iq,
    const float* s_w,
    const uint16_t* pl,
    int idx_heads,
    int lane
) {
    float p0 = fp16_bits_to_f32(pl[lane + 0]);
    float p1 = fp16_bits_to_f32(pl[lane + 32]);
    float p2 = fp16_bits_to_f32(pl[lane + 64]);
    float p3 = fp16_bits_to_f32(pl[lane + 96]);

    float acc = 0.0f;
    for (int h = 0; h < idx_heads; ++h) {
        const uint16_t* iq_h = s_iq + h * 128;
        float dot = fp16_bits_to_f32(iq_h[lane + 0])  * p0
                  + fp16_bits_to_f32(iq_h[lane + 32]) * p1
                  + fp16_bits_to_f32(iq_h[lane + 64]) * p2
                  + fp16_bits_to_f32(iq_h[lane + 96]) * p3;
        dot = warp_reduce_sum(dot);
        if (lane == 0 && dot > 0.0f) {
            acc += dot * s_w[h];
        }
    }
    return acc;
}

// Fast indexer scoring: stages iq [idx_heads, key_dim] + weights in SMEM.
// 4 warps/block score up to 64 consecutive pools per block.
__global__ void __launch_bounds__(128) glm5_dsa_score_kernel(
    const float* __restrict__ iq,         // [T, idx_heads, key_dim]
    const uint16_t* __restrict__ pooled,  // [n_pools, key_dim] F16
    const float* __restrict__ weights,    // [T, idx_heads]
    int key_dim,
    int idx_heads,
    int n_pools,
    float* __restrict__ scores            // [T, n_pools]
) {
    const int t = blockIdx.y;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;

    // idx_heads=32, key_dim=128 -> 4096 half/f16 = 8 KB SMEM!
    __shared__ uint16_t s_iq[32 * 128];
    __shared__ float s_w[32];

    const float* iqt = iq + (size_t)t * idx_heads * key_dim;
    for (int i = tid; i < idx_heads * key_dim; i += 128) {
        s_iq[i] = f32_to_fp16_bits(iqt[i]);
    }
    if (tid < idx_heads) {
        s_w[tid] = weights[(size_t)t * idx_heads + tid];
    }
    __syncthreads();

    const int p_end = min(n_pools, (int)(blockIdx.x + 1) * 64);
    for (int p = (int)blockIdx.x * 64 + (int)threadIdx.y; p < p_end; p += 4) {
        float acc = dsa_score_pool_row(s_iq, s_w, pooled + (size_t)p * key_dim, idx_heads, lane);
        if (lane == 0) {
            scores[(size_t)t * n_pools + p] = acc;
        }
    }
}

__global__ void __launch_bounds__(128) glm5_dsa_score_m1_dyn_kernel(
    const float* __restrict__ iq,         // [1, idx_heads, key_dim]
    const uint16_t* __restrict__ pooled,  // [pool_cap, key_dim] F16
    const float* __restrict__ weights,    // [1, idx_heads]
    int key_dim,
    int idx_heads,
    int kpool,
    const int32_t* __restrict__ pos_arr,  // [1]
    float* __restrict__ scores            // [pool_cap]
) {
    const int pos = pos_arr[0];
    const int n_pools = (pos >= 0) ? ((pos + 1) / kpool) : 0;
    if ((int)blockIdx.x * 4 >= n_pools) return;

    const int tid = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;
    __shared__ uint16_t s_iq[32 * 128];
    __shared__ float s_w[32];

    for (int i = tid; i < idx_heads * key_dim; i += 128) {
        s_iq[i] = f32_to_fp16_bits(iq[i]);
    }
    if (tid < idx_heads) {
        s_w[tid] = weights[tid];
    }
    __syncthreads();

    const int stride = gridDim.x * 4;
    for (int p = (int)blockIdx.x * 4 + threadIdx.y; p < n_pools; p += stride) {
        float acc = dsa_score_pool_row(s_iq, s_w, pooled + (size_t)p * key_dim, idx_heads, lane);
        if (lane == 0) {
            scores[p] = acc;
        }
    }
}

__device__ __forceinline__ uint32_t float_to_ordered_u32(float f) {
    uint32_t bits = __float_as_uint(f);
    uint32_t mask = (bits & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

// Exact O(n_vis) parallel Top-K selector matching Strata's stable descending rank
// (higher score wins; ties broken by LOWER pool index).
// Supports CUDA Graphs by reading pos_arr[t] dynamically from device memory.
__global__ void __launch_bounds__(256) glm5_dsa_select_kernel(
    const float* __restrict__ scores, // [T, n_pools_stride]
    int n_pools_stride,
    int kpool,
    int top_pools_cap,
    int select_tail,
    int n_sel,
    const int32_t* __restrict__ pos_arr, // [T]
    int32_t* __restrict__ cells          // [T, n_sel]
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const int pos = pos_arr[t];
    const int n_vis = (pos + 1) / kpool;
    const int top_pools = min(top_pools_cap, n_vis);
    const float* sc_row = scores + (size_t)t * n_pools_stride;
    int32_t* out_row = cells + (size_t)t * n_sel;

    for (int s = tid; s < n_sel; s += 256) {
        out_row[s] = -1;
    }
    __syncthreads();

    __shared__ uint32_t s_thresh_u;
    __shared__ uint32_t s_gt_count;
    __shared__ float s_cand_v[512];
    __shared__ int32_t s_cand_id[512];
    __shared__ uint32_t s_cand_cnt;

    if (n_vis <= top_pools_cap) {
        for (int p = tid; p < n_vis; p += 256) {
            for (int m = 0; m < kpool; ++m) {
                out_row[p * kpool + m] = p * kpool + m;
            }
        }
        __syncthreads();
    } else if (top_pools > 0) {
        // Binary search over exact 32-bit float order-preserving key to find threshold
        // such that count(score > thresh) <= top_pools and count(score >= thresh) >= top_pools.
        if (tid == 0) {
            s_thresh_u = 0;
            s_gt_count = 0;
            s_cand_cnt = 0;
        }
        for (int i = tid; i < 512; i += 256) {
            s_cand_v[i] = -INFINITY;
            s_cand_id[i] = -1;
        }
        __syncthreads();

        // Exact 4-pass 8-bit radix histogram over ordered 32-bit float key
        __shared__ uint32_t s_hist[256];
        __shared__ uint32_t s_prefix, s_mask, s_krem;
        if (tid == 0) {
            s_prefix = 0u;
            s_mask = 0u;
            s_krem = (uint32_t)top_pools;
        }
        __syncthreads();

        for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            const uint32_t cur_pref = s_prefix;
            const uint32_t cur_mask = s_mask;
            s_hist[tid] = 0u;
            __syncthreads();

            for (int p = tid; p < n_vis; p += 256) {
                uint32_t ku = float_to_ordered_u32(sc_row[p]);
                if ((ku & cur_mask) == cur_pref) {
                    atomicAdd(&s_hist[(ku >> shift) & 0xFFu], 1u);
                }
            }
            __syncthreads();

            if (tid == 0) {
                uint32_t rem = s_krem;
                uint32_t cum = 0u;
                uint32_t win_b = 0u;
                for (int b = 255; b >= 0; --b) {
                    uint32_t c = s_hist[b];
                    if (cum + c >= rem) {
                        win_b = (uint32_t)b;
                        s_krem = rem - cum;
                        break;
                    }
                    cum += c;
                }
                s_prefix = cur_pref | (win_b << shift);
                s_mask = cur_mask | (0xFFu << shift);
            }
            __syncthreads();
        }
        if (tid == 0) s_thresh_u = s_prefix;
        __syncthreads();

        // Gather strict > thresh first, and count exact number of strictly greater
        const uint32_t thr = s_thresh_u;
        uint32_t local_gt = 0;
        for (int p = tid; p < n_vis; p += 256) {
            uint32_t ku = float_to_ordered_u32(sc_row[p]);
            if (ku > thr) {
                ++local_gt;
                uint32_t pos_idx = atomicAdd(&s_cand_cnt, 1u);
                if (pos_idx < 512) {
                    s_cand_v[pos_idx] = sc_row[p];
                    s_cand_id[pos_idx] = p;
                }
            }
        }
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) local_gt += __shfl_xor_sync(0xFFFFFFFFu, local_gt, o);
        if ((tid & 31) == 0 && local_gt > 0) atomicAdd(&s_gt_count, local_gt);
        __syncthreads();

        // Fill remaining (top_pools - s_gt_count) slots with lowest-index ties where key == thr
        int need_eq = top_pools - (int)s_gt_count;
        if (need_eq > 0) {
            if (tid == 0) {
                int filled = 0;
                for (int p = 0; p < n_vis && filled < need_eq; ++p) {
                    if (float_to_ordered_u32(sc_row[p]) == thr) {
                        uint32_t dst = s_cand_cnt++;
                        if (dst < 512) {
                            s_cand_v[dst] = sc_row[p];
                            s_cand_id[dst] = p;
                        }
                        ++filled;
                    }
                }
            }
            __syncthreads();
        }

        // Rank within the <= 512 winners in parallel (exact descending score, tie by lower id)
        // Each thread ranks at most 2 winners (512 / 256 = 2), takes < 1 us!
        const int n_cand = min(top_pools, (int)min(s_cand_cnt, 512u));
        for (int i = tid; i < n_cand; i += 256) {
            float v = s_cand_v[i];
            int id = s_cand_id[i];
            if (id < 0 || id >= n_vis) continue;
            int r = 0;
            for (int j = 0; j < n_cand; ++j) {
                float u = s_cand_v[j];
                int jd = s_cand_id[j];
                if (u > v || (u == v && jd < id)) ++r;
            }
            for (int m = 0; m < kpool; ++m) {
                out_row[r * kpool + m] = id * kpool + m;
            }
        }
    }

    if (select_tail && tid == 0) {
        for (int m = 0; m < kpool - 1; ++m) {
            int cell = n_vis * kpool + m;
            if (cell <= pos) out_row[top_pools * kpool + m] = cell;
        }
    }
}

// Tensor-Core DSA indexer scoring kernel (TS_T=2 tokens x 32 heads = 64 rows, TS_P=64 pools, K=128)
__global__ void __launch_bounds__(256) glm5_dsa_score_tc_kernel(
    const float* __restrict__ iq,         // [T, 32, 128]
    const uint16_t* __restrict__ pooled,  // [n_pools, 128] F16
    const float* __restrict__ weights,    // [T, 32]
    int kpool,
    int T,
    int n_pools,
    const int32_t* __restrict__ pos_arr,  // nullable [T]
    float* __restrict__ scores            // [T, n_pools]
) {
    constexpr int TS_T = 2;
    constexpr int TS_P = 64;
    constexpr int H = 32;
    constexpr int K = 128;
    constexpr int LD = 136;

    const int t0 = (int)blockIdx.y * TS_T;
    const int pbeg = (int)blockIdx.x * TS_P;
    const int t_last = min(T - 1, t0 + TS_T - 1);
    if (pos_arr != nullptr && pbeg >= (pos_arr[t_last] + 1) / kpool) return;

    __shared__ union {
        struct {
            __align__(16) uint16_t sA[64][LD];
            __align__(16) uint16_t sB[64][LD];
        } ab;
        float sC[64][65];
    } sm;
    __shared__ float s_w[TS_T][H];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int k_pair = (lane & 3) * 2;

    // 64 rows x 128 F32 -> F16 = 2048 float4 loads
    for (int i = tid; i < 2048; i += 256) {
        const int r = i >> 5;
        const int k4 = (i & 31) << 2;
        const int tok = t0 + (r >> 5);
        const int hd = r & 31;
        uint32_t u0 = 0u, u1 = 0u;
        if (tok < T) {
            const float4 v = *reinterpret_cast<const float4*>(iq + ((size_t)tok * H + hd) * K + k4);
            u0 = (uint32_t)f32_to_fp16_bits(v.x) | ((uint32_t)f32_to_fp16_bits(v.y) << 16);
            u1 = (uint32_t)f32_to_fp16_bits(v.z) | ((uint32_t)f32_to_fp16_bits(v.w) << 16);
        }
        *reinterpret_cast<uint2*>(&sm.ab.sA[r][k4]) = make_uint2(u0, u1);
    }

    // 64 pools x 128 F16 = 1024 uint4 loads
    for (int i = tid; i < 1024; i += 256) {
        const int pp = i >> 4;
        const int k8 = (i & 15) << 3;
        const int p = pbeg + pp;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (p < n_pools) {
            v = *reinterpret_cast<const uint4*>(pooled + (size_t)p * K + k8);
        }
        *reinterpret_cast<uint4*>(&sm.ab.sB[pp][k8]) = v;
    }

    if (tid < 64) {
        const int ot = tid >> 5;
        const int h = tid & 31;
        const int tok = t0 + ot;
        s_w[ot][h] = (tok < T) ? weights[(size_t)tok * H + h] : 0.0f;
    }
    __syncthreads();

    float acc[4][4];
    #pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
        acc[nj][0] = acc[nj][1] = acc[nj][2] = acc[nj][3] = 0.0f;
    }

    const int wm = (warp >> 1) * 16;
    const int wn = (warp & 1) * 32;
    #pragma unroll
    for (int k0 = 0; k0 < K; k0 += 16) {
        const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&sm.ab.sA[wm + gid + 0][k0 + k_pair]);
        const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&sm.ab.sA[wm + gid + 8][k0 + k_pair]);
        const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&sm.ab.sA[wm + gid + 0][k0 + k_pair + 8]);
        const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&sm.ab.sA[wm + gid + 8][k0 + k_pair + 8]);
        #pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
            const int nr = wn + nj * 8 + gid;
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&sm.ab.sB[nr][k0 + k_pair]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&sm.ab.sB[nr][k0 + k_pair + 8]);
            mma16816_f32_f16_ops(acc[nj], a0, a1, a2, a3, b0, b1);
        }
    }
    __syncthreads();

    #pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
        const int c0 = wn + nj * 8 + k_pair;
        sm.sC[wm + gid + 0][c0 + 0] = acc[nj][0];
        sm.sC[wm + gid + 0][c0 + 1] = acc[nj][1];
        sm.sC[wm + gid + 8][c0 + 0] = acc[nj][2];
        sm.sC[wm + gid + 8][c0 + 1] = acc[nj][3];
    }
    __syncthreads();

    // 256 threads = 2 tokens x 64 pools x 2 half-reducers over 32 heads
    const int ot = tid >> 7;
    const int op = (tid >> 1) & 63;
    const int half = tid & 1;
    const int tok = t0 + ot;
    const int p = pbeg + op;
    float sum = 0.0f;
    const bool active = (tok < T && p < n_pools) && (pos_arr == nullptr || p < (pos_arr[tok] + 1) / kpool);
    if (active) {
        const int h0 = half * 16;
        #pragma unroll
        for (int dh = 0; dh < 16; ++dh) {
            const int h = h0 + dh;
            sum += fmaxf(sm.sC[ot * H + h][op], 0.0f) * s_w[ot][h];
        }
    }
    sum += __shfl_xor_sync(0xFFFFFFFFu, sum, 1);
    if (half == 0 && active) {
        scores[(size_t)tok * n_pools + p] = sum;
    }
}

extern "C" void launch_glm5_dsa_score_select(
    const float* iq,
    const void* pooled_f16,
    const float* weights,
    int key_dim,
    int idx_heads,
    int kpool,
    int top_pools_cap,
    int select_tail,
    int n_sel,
    const int32_t* pos_arr,
    float* scores_buf,
    int32_t* cells_out,
    int T,
    int cur_pool_done,
    cudaStream_t stream
) {
    if (T <= 0) return;
    if (cur_pool_done > 0) {
        if (T >= 2 && key_dim == 128 && idx_heads == 32) {
            dim3 grid((cur_pool_done + 63) / 64, (T + 1) / 2);
            glm5_dsa_score_tc_kernel<<<grid, 256, 0, stream>>>(
                iq, reinterpret_cast<const uint16_t*>(pooled_f16), weights,
                kpool, T, cur_pool_done, pos_arr, scores_buf
            );
        } else {
            dim3 grid((cur_pool_done + 63) / 64, T);
            dim3 block(32, 4);
            glm5_dsa_score_kernel<<<grid, block, 0, stream>>>(
                iq, reinterpret_cast<const uint16_t*>(pooled_f16), weights,
                key_dim, idx_heads, cur_pool_done, scores_buf
            );
        }
    }
    glm5_dsa_select_kernel<<<T, 256, 0, stream>>>(
        scores_buf, max(1, cur_pool_done), kpool, top_pools_cap, select_tail,
        n_sel, pos_arr, cells_out
    );
}

extern "C" void launch_glm5_dsa_score_select_dyn(
    const float* iq,
    const void* pooled_f16,
    const float* weights,
    int key_dim,
    int idx_heads,
    int kpool,
    int top_pools_cap,
    int select_tail,
    int n_sel,
    int pool_cap,
    const int32_t* pos_arr,
    float* scores_buf,
    int32_t* cells_out,
    cudaStream_t stream
) {
    dim3 block(32, 4);
    glm5_dsa_score_m1_dyn_kernel<<<128, block, 0, stream>>>(
        iq, reinterpret_cast<const uint16_t*>(pooled_f16), weights,
        key_dim, idx_heads, kpool, pos_arr, scores_buf
    );
    glm5_dsa_select_kernel<<<1, 256, 0, stream>>>(
        scores_buf, max(1, pool_cap), kpool, top_pools_cap, select_tail,
        n_sel, pos_arr, cells_out
    );
}

// Fast MLA attention reading INT8 lat8 (VRAM window + pinned-RAM streamed extension)
// and F16/F32 wv_b coalesced across c.
__global__ void __launch_bounds__(256) glm5_dsa_mla_attn_kernel(
    const float* __restrict__ q_abs,     // [T, n_head, kv_lora]
    const uint8_t* __restrict__ vram_lat,// nullable [res_cap * lat_stride]
    const uint8_t* __restrict__ ram_lat, // nullable [total_cap * lat_stride]
    int res_cap,
    const int32_t* __restrict__ cells,   // [T, n_sel]
    const void* __restrict__ wv_b,       // [n_head, v_head, kv_lora] F16 or F32
    int32_t wv_type,
    int kv_lora,
    int n_head,
    int v_head,
    int qk_nope,
    int n_sel,
    float* __restrict__ out,             // [T, n_head, v_head]
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int h = blockIdx.x % n_head;
    const int t = blockIdx.x / n_head;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const size_t lat_stride = (size_t)(kv_lora + kv_lora / 16);
    const int32_t* row = cells + (size_t)t * n_sel;
    const float q_scale = rsqrtf((float)qk_nope);

    extern __shared__ float smem[]; // s_prob [padded_4] + s_qa [padded_4] + s_ctx [kv_lora]
    const int n_sel_pad = (n_sel + 3) & ~3;
    const int kv_lora_pad = (kv_lora + 3) & ~3;
    float* s_prob = smem;
    float* s_qa   = smem + n_sel_pad;
    float* s_ctx  = s_qa + kv_lora_pad;
    __shared__ float s_red[8];
    __shared__ int s_nv;
    __shared__ float s_max, s_inv_denom;

    int my_last = -1;
    for (int s = tid; s < n_sel; s += 256) {
        if (row[s] >= 0) my_last = s;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        my_last = max(my_last, __shfl_xor_sync(0xFFFFFFFFu, my_last, o));
    }
    if (lane == 0) s_red[warp] = __int_as_float(my_last);
    __syncthreads();
    if (tid == 0) {
        int mx = -1;
        for (int w = 0; w < 8; ++w) mx = max(mx, __float_as_int(s_red[w]));
        s_nv = mx + 1;
    }
    __syncthreads();
    const int nv = s_nv;
    if (nv <= 0) {
        float* out_h = out + ((size_t)t * n_head + h) * v_head;
        for (int v = tid; v < v_head; v += 256) out_h[v] = 0.0f;
        return;
    }

    const float* qa = q_abs + ((size_t)t * n_head + h) * kv_lora;
    for (int e = tid; e < kv_lora; e += 256) {
        s_qa[e] = qa[e];
        s_ctx[e] = 0.0f;
    }
    __syncthreads();

    // 1. Warp-per-cell dot product over lat8 (16 elements per lane -> 512 elements/warp!)
    float local_mx = -INFINITY;
    for (int s = warp; s < nv; s += 8) {
        int cell = row[s];
        if (cell < 0 || (!vram_lat && !ram_lat) || (res_cap > 0 && cell >= res_cap && ram_lat == nullptr)) {
            if (lane == 0) s_prob[s] = -INFINITY;
            continue;
        }
        const uint8_t* rec = (vram_lat && (res_cap <= 0 || cell < res_cap))
            ? (vram_lat + (size_t)cell * lat_stride)
            : (ram_lat + (size_t)cell * lat_stride);
        float dot = 0.0f;
        if (kv_lora == 512) {
            const uint4 codes = reinterpret_cast<const uint4*>(rec)[lane];
            const float d = fp16_bits_to_f32(reinterpret_cast<const uint16_t*>(rec + 512)[lane >> 1]);
            const int e0 = lane << 4;
            uint32_t ws[4] = {codes.x, codes.y, codes.z, codes.w};
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t w = ws[k];
                int ek = e0 + k * 4;
                float q0 = s_qa[ek + 0], q1 = s_qa[ek + 1], q2 = s_qa[ek + 2], q3 = s_qa[ek + 3];
                dot += ((float)(int8_t)(w & 0xFF)         * d) * q0
                     + ((float)(int8_t)((w >> 8) & 0xFF)  * d) * q1
                     + ((float)(int8_t)((w >> 16) & 0xFF) * d) * q2
                     + ((float)(int8_t)((w >> 24) & 0xFF) * d) * q3;
            }
        } else {
            const uint16_t* sc = reinterpret_cast<const uint16_t*>(rec + kv_lora);
            for (int e = lane; e < kv_lora; e += 32) {
                float d = fp16_bits_to_f32(sc[e >> 5]);
                dot += ((float)(int8_t)rec[e] * d) * s_qa[e];
            }
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            float sc = dot * q_scale;
            s_prob[s] = sc;
            local_mx = fmaxf(local_mx, sc);
        }
    }
    local_mx = warp_reduce_max(local_mx);
    if (lane == 0) s_red[warp] = local_mx;
    __syncthreads();
    if (tid == 0) {
        float mx = -INFINITY;
        for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_red[w]);
        s_max = mx;
    }
    __syncthreads();

    // 2. Parallel softmax across nv
    float mx = s_max;
    float denom_l = 0.0f;
    for (int s = tid; s < nv; s += 256) {
        float v = s_prob[s];
        float ev = (v == -INFINITY) ? 0.0f : expf(v - mx);
        s_prob[s] = ev;
        denom_l += ev;
    }
    denom_l = warp_reduce_sum(denom_l);
    if (lane == 0) s_red[warp] = denom_l;
    __syncthreads();
    if (tid == 0) {
        float dtot = 0.0f;
        for (int w = 0; w < 8; ++w) dtot += s_red[w];
        s_inv_denom = (dtot > 0.0f) ? (1.0f / dtot) : 0.0f;
    }
    __syncthreads();
    const float inv_denom = s_inv_denom;
    for (int s = tid; s < nv; s += 256) {
        s_prob[s] *= inv_denom;
    }
    __syncthreads();

    // 3. Gather ctx[e] = sum_s prob[s] * lat(cell_s)[e]
    if (kv_lora == 512) {
        float acc[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] = 0.0f;
        for (int s = warp; s < nv; s += 8) {
            int cell = row[s];
            if (cell < 0 || (!vram_lat && !ram_lat) || (res_cap > 0 && cell >= res_cap && ram_lat == nullptr)) continue;
            float pr = s_prob[s];
            const uint8_t* rec = (vram_lat && (res_cap <= 0 || cell < res_cap))
                ? (vram_lat + (size_t)cell * lat_stride)
                : (ram_lat + (size_t)cell * lat_stride);
            const uint4 codes = reinterpret_cast<const uint4*>(rec)[lane];
            const float prd = pr * fp16_bits_to_f32(reinterpret_cast<const uint16_t*>(rec + 512)[lane >> 1]);
            const uint32_t ws[4] = {codes.x, codes.y, codes.z, codes.w};
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t w = ws[k];
                acc[k * 4 + 0] += prd * (float)(int8_t)(w & 0xFF);
                acc[k * 4 + 1] += prd * (float)(int8_t)((w >> 8) & 0xFF);
                acc[k * 4 + 2] += prd * (float)(int8_t)((w >> 16) & 0xFF);
                acc[k * 4 + 3] += prd * (float)(int8_t)((w >> 24) & 0xFF);
            }
        }
        float* my_wctx = s_ctx + (size_t)warp * 512 + (size_t)lane * 16;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            my_wctx[k] = acc[k];
        }
        __syncthreads();
        for (int c = tid; c < 512; c += 256) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                sum += s_ctx[(size_t)w * 512 + c];
            }
            s_ctx[c] = sum;
        }
    } else {
        for (int base = 0; base < kv_lora; base += 256) {
            int e = base + tid;
            if (e < kv_lora) {
                float acc = 0.0f;
                int sc_idx = e >> 5;
                for (int s = 0; s < nv; ++s) {
                    int cell = row[s];
                    if (cell < 0 || (!vram_lat && !ram_lat) || (res_cap > 0 && cell >= res_cap && ram_lat == nullptr)) continue;
                    float pr = s_prob[s];
                    const uint8_t* rec = (vram_lat && (res_cap <= 0 || cell < res_cap))
                        ? (vram_lat + (size_t)cell * lat_stride)
                        : (ram_lat + (size_t)cell * lat_stride);
                    float d = fp16_bits_to_f32(reinterpret_cast<const uint16_t*>(rec + kv_lora)[sc_idx]);
                    acc += pr * ((float)(int8_t)rec[e] * d);
                }
                s_ctx[e] = acc;
            }
        }
    }
    __syncthreads();

    // 4. out[t, h, v] = sum_c wv_b[h, v, c] * s_ctx[c] (8 warps across v, vectorized coalesced on c!)
    size_t wv_h_off = (size_t)h * v_head * kv_lora;
    float* out_h = out + ((size_t)t * n_head + h) * v_head;
    for (int v = warp; v < v_head; v += 8) {
        size_t wr_off = wv_h_off + (size_t)v * kv_lora;
        float acc = 0.0f;
        if (wv_type == GGML_TYPE_F16 && (kv_lora & 7) == 0 && (reinterpret_cast<uintptr_t>(wv_b) & 15u) == 0u) {
            const uint4* wr8 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(wv_b) + wr_off);
            const float4* ctx4 = reinterpret_cast<const float4*>(s_ctx);
            const int n8 = kv_lora >> 3;
            for (int c8 = lane; c8 < n8; c8 += 32) {
                uint4 u = wr8[c8];
                float4 a0 = ctx4[c8 * 2 + 0];
                float4 a1 = ctx4[c8 * 2 + 1];
                acc += fp16_bits_to_f32((uint16_t)u.x)         * a0.x
                     + fp16_bits_to_f32((uint16_t)(u.x >> 16)) * a0.y
                     + fp16_bits_to_f32((uint16_t)u.y)         * a0.z
                     + fp16_bits_to_f32((uint16_t)(u.y >> 16)) * a0.w
                     + fp16_bits_to_f32((uint16_t)u.z)         * a1.x
                     + fp16_bits_to_f32((uint16_t)(u.z >> 16)) * a1.y
                     + fp16_bits_to_f32((uint16_t)u.w)         * a1.z
                     + fp16_bits_to_f32((uint16_t)(u.w >> 16)) * a1.w;
            }
        } else if (wv_type == GGML_TYPE_F16) {
            const uint16_t* wr = reinterpret_cast<const uint16_t*>(wv_b) + wr_off;
            for (int c = lane; c < kv_lora; c += 32) {
                acc += fp16_bits_to_f32(wr[c]) * s_ctx[c];
            }
        } else {
            const float* wr = reinterpret_cast<const float*>(wv_b) + wr_off;
            for (int c = lane; c < kv_lora; c += 32) {
                acc += wr[c] * s_ctx[c];
            }
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) out_h[v] = acc;
    }
}

extern "C" void launch_glm5_dsa_mla_attn(
    const float* q_abs,
    const void* vram_lat8,
    const void* ram_lat8,
    int res_cap,
    const int32_t* cells,
    const void* wv_b,
    int32_t wv_type,
    int kv_lora,
    int n_head,
    int v_head,
    int qk_nope,
    int n_sel,
    float* out,
    int T,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (T <= 0 || n_head <= 0 || n_sel <= 0) return;
    const size_t ctx_floats = (kv_lora == 512) ? (8u * 512u) : (size_t)((kv_lora + 3) & ~3);
    size_t smem = ((size_t)((n_sel + 3) & ~3) + (size_t)((kv_lora + 3) & ~3) + ctx_floats) * sizeof(float);
    glm5_dsa_mla_attn_kernel<<<T * n_head, 256, smem, stream>>>(
        q_abs,
        reinterpret_cast<const uint8_t*>(vram_lat8),
        reinterpret_cast<const uint8_t*>(ram_lat8),
        res_cap, cells, wv_b, wv_type, kv_lora, n_head, v_head, qk_nope, n_sel, out, skip_dev
    );
}

__global__ void __launch_bounds__(256) glm5_dsa_mla_prefill_mma_kernel(
    const float* __restrict__ q_abs,     // [T, n_head, 512]
    const uint8_t* __restrict__ vram_lat,// nullable [res_cap * 544]
    const uint8_t* __restrict__ ram_lat, // nullable [total_cap * 544]
    int res_cap,
    const int32_t* __restrict__ cells,   // [T, n_sel]
    int n_head,
    int qk_nope,
    int n_sel,
    float* __restrict__ ctx_out          // [T, n_head, 512] (may alias q_abs)
) {
    const int t = (int)blockIdx.x;
    const int h0 = (int)blockIdx.y * 32;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int k_pair = (lane & 3) * 2;
    const size_t lat_stride = 544;
    const int32_t* row = cells + (size_t)t * n_sel;
    const float q_scale = rsqrtf((float)qk_nope);

    extern __shared__ __align__(16) uint8_t dsmem[];
    constexpr int LD = 520;
    uint16_t (*s_q)[LD] = reinterpret_cast<uint16_t (*)[LD]>(dsmem);
    uint16_t (*s_k)[LD] = reinterpret_cast<uint16_t (*)[LD]>(dsmem + 33280);
    uint16_t (*s_p)[40] = reinterpret_cast<uint16_t (*)[40]>(dsmem + 66560);
    float (*s_score)[32] = reinterpret_cast<float (*)[32]>(dsmem + 69120);

    __shared__ float s_qinv[32];
    __shared__ float s_m[32];
    __shared__ float s_d[32];
    __shared__ float s_rsc[32];
    __shared__ float s_red[8];
    __shared__ int32_t s_cells[32];
    __shared__ int32_t s_nv;

    int my_last = -1;
    for (int s = tid; s < n_sel; s += 256) {
        if (row[s] >= 0) my_last = s;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        my_last = max(my_last, __shfl_xor_sync(0xFFFFFFFFu, my_last, o));
    }
    if (lane == 0) s_red[warp] = __int_as_float(my_last);
    __syncthreads();
    if (tid == 0) {
        int gl = -1;
        for (int w = 0; w < 8; ++w) gl = max(gl, __float_as_int(s_red[w]));
        s_nv = gl + 1;
    }
    __syncthreads();
    const int nv = s_nv;

    #pragma unroll 1
    for (int p = 0; p < 4; ++p) {
        const int r = warp * 4 + p;
        const float* qr = q_abs + ((size_t)t * n_head + (h0 + r)) * 512;
        const float4* q4 = reinterpret_cast<const float4*>(qr + lane * 16);
        const float4 a0 = q4[0], a1 = q4[1], a2 = q4[2], a3 = q4[3];
        const float qv[16] = {
            a0.x, a0.y, a0.z, a0.w,
            a1.x, a1.y, a1.z, a1.w,
            a2.x, a2.y, a2.z, a2.w,
            a3.x, a3.y, a3.z, a3.w
        };
        float mx = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; ++k) mx = fmaxf(mx, fabsf(qv[k]));
        const float amax = warp_reduce_max(mx);
        const float f = (amax > 0.0f) ? (16.0f / amax) : 0.0f;
        if (lane == 0) {
            s_qinv[r] = (amax > 0.0f) ? ((amax * 0.0625f) * q_scale) : 0.0f;
            s_m[r] = -INFINITY;
            s_d[r] = 0.0f;
            s_rsc[r] = 0.0f;
        }
        uint32_t u0[4], u1[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            u0[k] = f32x2_to_fp16x2_u32(qv[k * 2 + 0] * f, qv[k * 2 + 1] * f);
            u1[k] = f32x2_to_fp16x2_u32(qv[8 + k * 2 + 0] * f, qv[8 + k * 2 + 1] * f);
        }
        *reinterpret_cast<uint4*>(&s_q[r][lane * 16 + 0]) = make_uint4(u0[0], u0[1], u0[2], u0[3]);
        *reinterpret_cast<uint4*>(&s_q[r][lane * 16 + 8]) = make_uint4(u1[0], u1[1], u1[2], u1[3]);
    }
    __syncthreads();

    float acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
    }

    const int warp_m = warp >> 2;
    const int warp_n = warp & 3;
    const int h_sub = warp_m * 16;
    const int c_sub = warp_n * 8;
    const int d_sub = warp_n * 128;

    for (int s0 = 0; s0 < nv; s0 += 32) {
        if (s0 > 0) __syncthreads();
        if (tid < 32) {
            s_cells[tid] = (s0 + tid < nv) ? row[s0 + tid] : -1;
        }
        __syncthreads();

        for (int i = tid; i < 1024; i += 256) {
            const int r = i >> 5;
            const int c16 = (i & 31) * 16;
            const int cell = s_cells[r];
            if (cell < 0 || (!vram_lat && !ram_lat) || (res_cap > 0 && cell >= res_cap && ram_lat == nullptr)) {
                *reinterpret_cast<uint4*>(&s_k[r][c16 + 0]) = make_uint4(0u, 0u, 0u, 0u);
                *reinterpret_cast<uint4*>(&s_k[r][c16 + 8]) = make_uint4(0u, 0u, 0u, 0u);
            } else {
                const uint8_t* rec = (vram_lat && (res_cap <= 0 || cell < res_cap))
                    ? (vram_lat + (size_t)cell * lat_stride)
                    : (ram_lat + (size_t)cell * lat_stride);
                const uint4 codes = *reinterpret_cast<const uint4*>(rec + c16);
                const float d = fp16_bits_to_f32(reinterpret_cast<const uint16_t*>(rec + 512)[c16 >> 5]);
                const uint32_t ws[4] = {codes.x, codes.y, codes.z, codes.w};
                uint32_t u0[4], u1[4];
                #pragma unroll
                for (int k = 0; k < 2; ++k) {
                    const uint32_t w0 = ws[k];
                    u0[k * 2 + 0] = f32x2_to_fp16x2_u32(
                        (float)(int8_t)(w0 & 0xFF) * d,
                        (float)(int8_t)((w0 >> 8) & 0xFF) * d
                    );
                    u0[k * 2 + 1] = f32x2_to_fp16x2_u32(
                        (float)(int8_t)((w0 >> 16) & 0xFF) * d,
                        (float)(int8_t)((w0 >> 24) & 0xFF) * d
                    );
                    const uint32_t w1 = ws[k + 2];
                    u1[k * 2 + 0] = f32x2_to_fp16x2_u32(
                        (float)(int8_t)(w1 & 0xFF) * d,
                        (float)(int8_t)((w1 >> 8) & 0xFF) * d
                    );
                    u1[k * 2 + 1] = f32x2_to_fp16x2_u32(
                        (float)(int8_t)((w1 >> 16) & 0xFF) * d,
                        (float)(int8_t)((w1 >> 24) & 0xFF) * d
                    );
                }
                *reinterpret_cast<uint4*>(&s_k[r][c16 + 0]) = make_uint4(u0[0], u0[1], u0[2], u0[3]);
                *reinterpret_cast<uint4*>(&s_k[r][c16 + 8]) = make_uint4(u1[0], u1[1], u1[2], u1[3]);
            }
        }
        __syncthreads();

        float sc_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll 4
        for (int k0 = 0; k0 < 512; k0 += 16) {
            uint32_t a0, a1, a2, a3, b0, b1;
            ldmatrix_x4_f16(a0, a1, a2, a3, &s_q[h_sub + (lane & 15)][k0 + (lane >> 4) * 8]);
            ldmatrix_x2_f16(b0, b1, &s_k[c_sub + (lane & 7)][k0 + ((lane >> 3) & 1) * 8]);
            mma16816_f32_f16_ops(sc_acc, a0, a1, a2, a3, b0, b1);
        }
        const int sc0 = c_sub + k_pair;
        s_score[h_sub + gid + 0][sc0 + 0] = sc_acc[0];
        s_score[h_sub + gid + 0][sc0 + 1] = sc_acc[1];
        s_score[h_sub + gid + 8][sc0 + 0] = sc_acc[2];
        s_score[h_sub + gid + 8][sc0 + 1] = sc_acc[3];
        __syncthreads();

        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int sr = warp * 4 + p;
            const int sc = lane;
            const float vs = (s_cells[sc] < 0) ? -INFINITY : (s_score[sr][sc] * s_qinv[sr]);
            const float mx = warp_reduce_max(vs);
            const float old_m = s_m[sr];
            const float new_m = fmaxf(old_m, mx);
            const float p_val = (vs == -INFINITY || new_m == -INFINITY) ? 0.0f : __expf(vs - new_m);
            const float sum_p = warp_reduce_sum(p_val);
            s_p[sr][sc] = f32_to_fp16_bits(p_val);
            if (lane == 0) {
                const float rsc = (old_m == -INFINITY || new_m == -INFINITY) ? 0.0f : __expf(old_m - new_m);
                s_rsc[sr] = rsc;
                s_d[sr] = s_d[sr] * rsc + sum_p;
                s_m[sr] = new_m;
            }
        }
        __syncthreads();

        const float rsc0 = s_rsc[h_sub + gid];
        const float rsc1 = s_rsc[h_sub + gid + 8];
        #pragma unroll
        for (int nt = 0; nt < 16; ++nt) {
            acc[nt][0] *= rsc0;
            acc[nt][1] *= rsc0;
            acc[nt][2] *= rsc1;
            acc[nt][3] *= rsc1;
        }
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int k_p = kk * 16;
            uint32_t a0, a1, a2, a3;
            ldmatrix_x4_f16(a0, a1, a2, a3, &s_p[h_sub + (lane & 15)][k_p + (lane >> 4) * 8]);
            #pragma unroll
            for (int nt = 0; nt < 16; ++nt) {
                const int col = d_sub + nt * 8;
                uint32_t b0, b1;
                ldmatrix_x2_trans_f16(b0, b1, &s_k[k_p + (lane & 15)][col]);
                mma16816_f32_f16_ops(acc[nt], a0, a1, a2, a3, b0, b1);
            }
        }
    }

    const float inv0 = (s_d[h_sub + gid] > 0.0f) ? (1.0f / s_d[h_sub + gid]) : 0.0f;
    const float inv1 = (s_d[h_sub + gid + 8] > 0.0f) ? (1.0f / s_d[h_sub + gid + 8]) : 0.0f;
    float* out0 = ctx_out + ((size_t)t * n_head + (h0 + h_sub + gid + 0)) * 512;
    float* out1 = ctx_out + ((size_t)t * n_head + (h0 + h_sub + gid + 8)) * 512;
    #pragma unroll
    for (int nt = 0; nt < 16; ++nt) {
        const int c0 = d_sub + nt * 8 + k_pair;
        *reinterpret_cast<float2*>(&out0[c0]) = make_float2(acc[nt][0] * inv0, acc[nt][1] * inv0);
        *reinterpret_cast<float2*>(&out1[c0]) = make_float2(acc[nt][2] * inv1, acc[nt][3] * inv1);
    }
}

__global__ void __launch_bounds__(256) glm5_dsa_vproj_mma_kernel(
    const float* __restrict__ ctx,   // [T, n_head, 512]
    const uint16_t* __restrict__ wv_b, // [n_head, 256, 512]
    int n_head,
    int v_head,
    int T,
    float* __restrict__ out          // [T, n_head, 256]
) {
    const int t0 = (int)blockIdx.x * 64;
    const int h = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int k_pair = (lane & 3) * 2;

    extern __shared__ __align__(16) uint8_t dsmem[];
    constexpr int LD = 520;
    uint16_t (*s_ctx)[LD] = reinterpret_cast<uint16_t (*)[LD]>(dsmem);
    uint16_t (*s_w)[24] = reinterpret_cast<uint16_t (*)[24]>(dsmem + 66560);
    __shared__ float s_inv[64];

    #pragma unroll 1
    for (int p = 0; p < 8; ++p) {
        const int r = warp * 8 + p;
        const int gt = t0 + r;
        if (gt < T) {
            const float* cr = ctx + ((size_t)gt * n_head + h) * 512;
            const float4* c4 = reinterpret_cast<const float4*>(cr + lane * 16);
            const float4 a0 = c4[0], a1 = c4[1], a2 = c4[2], a3 = c4[3];
            const float cv[16] = {
                a0.x, a0.y, a0.z, a0.w,
                a1.x, a1.y, a1.z, a1.w,
                a2.x, a2.y, a2.z, a2.w,
                a3.x, a3.y, a3.z, a3.w
            };
            float mx = 0.0f;
            #pragma unroll
            for (int k = 0; k < 16; ++k) mx = fmaxf(mx, fabsf(cv[k]));
            const float amax = warp_reduce_max(mx);
            const float f = (amax > 0.0f) ? (16.0f / amax) : 0.0f;
            if (lane == 0) s_inv[r] = amax * 0.0625f;
            uint32_t u0[4], u1[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                u0[k] = f32x2_to_fp16x2_u32(cv[k * 2 + 0] * f, cv[k * 2 + 1] * f);
                u1[k] = f32x2_to_fp16x2_u32(cv[8 + k * 2 + 0] * f, cv[8 + k * 2 + 1] * f);
            }
            *reinterpret_cast<uint4*>(&s_ctx[r][lane * 16 + 0]) = make_uint4(u0[0], u0[1], u0[2], u0[3]);
            *reinterpret_cast<uint4*>(&s_ctx[r][lane * 16 + 8]) = make_uint4(u1[0], u1[1], u1[2], u1[3]);
        } else {
            if (lane == 0) s_inv[r] = 0.0f;
            *reinterpret_cast<uint4*>(&s_ctx[r][lane * 16 + 0]) = make_uint4(0u, 0u, 0u, 0u);
            *reinterpret_cast<uint4*>(&s_ctx[r][lane * 16 + 8]) = make_uint4(0u, 0u, 0u, 0u);
        }
    }

    float acc[2][8][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int nj = 0; nj < 8; ++nj)
            acc[mi][nj][0] = acc[mi][nj][1] = acc[mi][nj][2] = acc[mi][nj][3] = 0.0f;

    const int wm = (warp >> 2) * 32;
    const int wn = (warp & 3) * 64;
    const uint16_t* wv_h = wv_b + (size_t)h * v_head * 512;

    #pragma unroll 1
    for (int k0 = 0; k0 < 512; k0 += 16) {
        const uint4* w4 = reinterpret_cast<const uint4*>(wv_h + (size_t)tid * 512 + k0);
        *reinterpret_cast<uint4*>(&s_w[tid][0]) = w4[0];
        *reinterpret_cast<uint4*>(&s_w[tid][8]) = w4[1];
        __syncthreads();

        uint32_t a_frag[2][4];
        #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
            ldmatrix_x4_f16(
                a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                &s_ctx[wm + mi * 16 + (lane & 15)][k0 + (lane >> 4) * 8]
            );
        }
        #pragma unroll
        for (int nj = 0; nj < 8; ++nj) {
            uint32_t b0, b1;
            ldmatrix_x2_f16(b0, b1, &s_w[wn + nj * 8 + (lane & 7)][((lane >> 3) & 1) * 8]);
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                mma16816_f32_f16_ops(acc[mi][nj], a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3], b0, b1);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = wm + mi * 16 + gid;
        const int r1 = r0 + 8;
        const int gt0 = t0 + r0;
        const int gt1 = t0 + r1;
        const float inv0 = s_inv[r0];
        const float inv1 = s_inv[r1];
        if (gt0 < T) {
            float* o0 = out + ((size_t)gt0 * n_head + h) * v_head;
            #pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                const int c0 = wn + nj * 8 + k_pair;
                *reinterpret_cast<float2*>(&o0[c0]) = make_float2(acc[mi][nj][0] * inv0, acc[mi][nj][1] * inv0);
            }
        }
        if (gt1 < T) {
            float* o1 = out + ((size_t)gt1 * n_head + h) * v_head;
            #pragma unroll
            for (int nj = 0; nj < 8; ++nj) {
                const int c0 = wn + nj * 8 + k_pair;
                *reinterpret_cast<float2*>(&o1[c0]) = make_float2(acc[mi][nj][2] * inv1, acc[mi][nj][3] * inv1);
            }
        }
    }
}

static uint32_t s_dsa_mla_smem_mask = 0u;
static uint32_t s_dsa_vproj_smem_mask = 0u;

extern "C" void launch_glm5_dsa_mla_prefill_mma(
    const float* q_abs,
    const void* vram_lat8,
    const void* ram_lat8,
    int res_cap,
    const int32_t* cells,
    const void* wv_b,
    int32_t wv_type,
    int kv_lora,
    int n_head,
    int v_head,
    int qk_nope,
    int n_sel,
    float* ctx_scratch,
    float* out,
    int T,
    cudaStream_t stream
) {
    if (T <= 0 || n_head <= 0 || n_sel <= 0) return;
    if (T > 1 && kv_lora == 512 && (n_head & 31) == 0 && wv_type == GGML_TYPE_F16 && v_head == 256
        && ctx_scratch != nullptr && (reinterpret_cast<uintptr_t>(wv_b) & 15u) == 0u) {
        constexpr int MLA_SMEM = 73216;
        constexpr int VPROJ_SMEM = 78848;
        int dev = 0;
        cudaGetDevice(&dev);
        const uint32_t bit = 1u << (dev & 31);
        if ((s_dsa_mla_smem_mask & bit) == 0u) {
            cudaFuncSetAttribute(glm5_dsa_mla_prefill_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, MLA_SMEM);
            s_dsa_mla_smem_mask |= bit;
        }
        if ((s_dsa_vproj_smem_mask & bit) == 0u) {
            cudaFuncSetAttribute(glm5_dsa_vproj_mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, VPROJ_SMEM);
            s_dsa_vproj_smem_mask |= bit;
        }
        dim3 g1(T, n_head / 32);
        glm5_dsa_mla_prefill_mma_kernel<<<g1, 256, MLA_SMEM, stream>>>(
            q_abs,
            reinterpret_cast<const uint8_t*>(vram_lat8),
            reinterpret_cast<const uint8_t*>(ram_lat8),
            res_cap, cells, n_head, qk_nope, n_sel, ctx_scratch
        );
        dim3 g2((T + 63) / 64, n_head);
        glm5_dsa_vproj_mma_kernel<<<g2, 256, VPROJ_SMEM, stream>>>(
            ctx_scratch,
            reinterpret_cast<const uint16_t*>(wv_b),
            n_head, v_head, T, out
        );
        return;
    }
    launch_glm5_dsa_mla_attn(
        q_abs, vram_lat8, ram_lat8, res_cap, cells, wv_b, wv_type,
        kv_lora, n_head, v_head, qk_nope, n_sel, out, T, nullptr, stream
    );
}

// Split M=1 DSA MLA context kernel (grid = n_head, block = 256)
// Outputs context vector [n_head, kv_lora] (can alias q_abs in-place since s_qa stages it first).
__global__ void __launch_bounds__(256) glm5_dsa_mla_ctx_m1_kernel(
    const float* __restrict__ q_abs,
    const uint8_t* __restrict__ vram_lat,
    const uint8_t* __restrict__ ram_lat,
    int res_cap,
    const int32_t* __restrict__ cells,
    int kv_lora,
    int n_head,
    int qk_nope,
    int n_sel,
    float* __restrict__ ctx_out,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int h = blockIdx.x;
    if (h >= n_head) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const size_t lat_stride = (size_t)(kv_lora + kv_lora / 16);
    const float q_scale = rsqrtf((float)qk_nope);

    extern __shared__ float smem[];
    const int n_sel_pad = (n_sel + 3) & ~3;
    const int kv_lora_pad = (kv_lora + 3) & ~3;
    float* s_prob = smem;
    float* s_qa   = smem + n_sel_pad;
    float* s_ctx  = s_qa + kv_lora_pad;
    __shared__ float s_red[8];
    __shared__ int s_nv;
    __shared__ float s_max, s_inv_denom;

    int my_last = -1;
    for (int s = tid; s < n_sel; s += 256) {
        if (cells[s] >= 0) my_last = s;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        my_last = max(my_last, __shfl_xor_sync(0xFFFFFFFFu, my_last, o));
    }
    if (lane == 0) s_red[warp] = __int_as_float(my_last);
    __syncthreads();
    if (tid == 0) {
        int mx = -1;
        for (int w = 0; w < 8; ++w) mx = max(mx, __float_as_int(s_red[w]));
        s_nv = mx + 1;
    }
    __syncthreads();
    const int nv = s_nv;
    float* ctx_h = ctx_out + (size_t)h * kv_lora;
    if (nv <= 0) {
        for (int c = tid; c < kv_lora; c += 256) ctx_h[c] = 0.0f;
        return;
    }

    const float* qa = q_abs + (size_t)h * kv_lora;
    for (int e = tid; e < kv_lora; e += 256) {
        s_qa[e] = qa[e];
    }
    __syncthreads();

    float local_mx = -INFINITY;
    for (int s = warp; s < nv; s += 8) {
        int cell = cells[s];
        if (cell < 0 || (!vram_lat && !ram_lat) || (res_cap > 0 && cell >= res_cap && ram_lat == nullptr)) {
            if (lane == 0) s_prob[s] = -INFINITY;
            continue;
        }
        const uint8_t* rec = (vram_lat && (res_cap <= 0 || cell < res_cap))
            ? (vram_lat + (size_t)cell * lat_stride)
            : (ram_lat + (size_t)cell * lat_stride);
        float dot = 0.0f;
        if (kv_lora == 512) {
            const uint4 codes = reinterpret_cast<const uint4*>(rec)[lane];
            const float d = fp16_bits_to_f32(reinterpret_cast<const uint16_t*>(rec + 512)[lane >> 1]);
            const int e0 = lane << 4;
            uint32_t ws[4] = {codes.x, codes.y, codes.z, codes.w};
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t w = ws[k];
                int ek = e0 + k * 4;
                dot += ((float)(int8_t)(w & 0xFF)         * d) * s_qa[ek + 0]
                     + ((float)(int8_t)((w >> 8) & 0xFF)  * d) * s_qa[ek + 1]
                     + ((float)(int8_t)((w >> 16) & 0xFF) * d) * s_qa[ek + 2]
                     + ((float)(int8_t)((w >> 24) & 0xFF) * d) * s_qa[ek + 3];
            }
        } else {
            const uint16_t* sc = reinterpret_cast<const uint16_t*>(rec + kv_lora);
            for (int e = lane; e < kv_lora; e += 32) {
                float d = fp16_bits_to_f32(sc[e >> 5]);
                dot += ((float)(int8_t)rec[e] * d) * s_qa[e];
            }
        }
        dot = warp_reduce_sum(dot);
        if (lane == 0) {
            float sc = dot * q_scale;
            s_prob[s] = sc;
            local_mx = fmaxf(local_mx, sc);
        }
    }
    local_mx = warp_reduce_max(local_mx);
    if (lane == 0) s_red[warp] = local_mx;
    __syncthreads();
    if (tid == 0) {
        float mx = -INFINITY;
        for (int w = 0; w < 8; ++w) mx = fmaxf(mx, s_red[w]);
        s_max = mx;
    }
    __syncthreads();

    const float mx = s_max;
    float denom_l = 0.0f;
    for (int s = tid; s < nv; s += 256) {
        float v = s_prob[s];
        float ev = (v == -INFINITY) ? 0.0f : expf(v - mx);
        s_prob[s] = ev;
        denom_l += ev;
    }
    denom_l = warp_reduce_sum(denom_l);
    if (lane == 0) s_red[warp] = denom_l;
    __syncthreads();
    if (tid == 0) {
        float dtot = 0.0f;
        for (int w = 0; w < 8; ++w) dtot += s_red[w];
        s_inv_denom = (dtot > 0.0f) ? (1.0f / dtot) : 0.0f;
    }
    __syncthreads();
    const float inv_denom = s_inv_denom;
    for (int s = tid; s < nv; s += 256) {
        s_prob[s] *= inv_denom;
    }
    __syncthreads();

    if (kv_lora == 512) {
        float acc[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] = 0.0f;
        for (int s = warp; s < nv; s += 8) {
            int cell = cells[s];
            if (cell < 0 || (!vram_lat && !ram_lat) || (res_cap > 0 && cell >= res_cap && ram_lat == nullptr)) continue;
            float pr = s_prob[s];
            const uint8_t* rec = (vram_lat && (res_cap <= 0 || cell < res_cap))
                ? (vram_lat + (size_t)cell * lat_stride)
                : (ram_lat + (size_t)cell * lat_stride);
            const uint4 codes = reinterpret_cast<const uint4*>(rec)[lane];
            const float prd = pr * fp16_bits_to_f32(reinterpret_cast<const uint16_t*>(rec + 512)[lane >> 1]);
            const uint32_t ws[4] = {codes.x, codes.y, codes.z, codes.w};
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t w = ws[k];
                acc[k * 4 + 0] += prd * (float)(int8_t)(w & 0xFF);
                acc[k * 4 + 1] += prd * (float)(int8_t)((w >> 8) & 0xFF);
                acc[k * 4 + 2] += prd * (float)(int8_t)((w >> 16) & 0xFF);
                acc[k * 4 + 3] += prd * (float)(int8_t)((w >> 24) & 0xFF);
            }
        }
        float* my_wctx = s_ctx + (size_t)warp * 512 + (size_t)lane * 16;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            my_wctx[k] = acc[k];
        }
        __syncthreads();
        for (int c = tid; c < 512; c += 256) {
            float sum = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                sum += s_ctx[(size_t)w * 512 + c];
            }
            ctx_h[c] = sum;
        }
    } else {
        for (int e = tid; e < kv_lora; e += 256) {
            float acc = 0.0f;
            int sc_idx = e >> 5;
            for (int s = 0; s < nv; ++s) {
                int cell = cells[s];
                if (cell < 0 || (!vram_lat && !ram_lat) || (res_cap > 0 && cell >= res_cap && ram_lat == nullptr)) continue;
                float pr = s_prob[s];
                const uint8_t* rec = (vram_lat && (res_cap <= 0 || cell < res_cap))
                    ? (vram_lat + (size_t)cell * lat_stride)
                    : (ram_lat + (size_t)cell * lat_stride);
                float d = fp16_bits_to_f32(reinterpret_cast<const uint16_t*>(rec + kv_lora)[sc_idx]);
                acc += pr * ((float)(int8_t)rec[e] * d);
            }
            ctx_h[e] = acc;
        }
    }
}

// High-occupancy headwise wv_b projection + fused Q8_1 quantization (grid = n_head * (v_head / 32), block = 256)
__global__ void __launch_bounds__(256) glm5_dsa_headwise_vproj_q8_kernel(
    const float* __restrict__ ctx,
    const void* __restrict__ wv_b,
    int32_t wv_type,
    int kv_lora,
    int n_head,
    int v_head,
    float* __restrict__ out_f32_opt,
    block_q8_1* __restrict__ out_q8,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int row0 = blockIdx.x * 32;
    const int h = row0 / v_head;
    if (h >= n_head) return;
    const int v0 = row0 - h * v_head;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ __align__(16) float s_ctx[512];
    __shared__ float s_o[32];

    const float* ctx_h = ctx + (size_t)h * kv_lora;
    for (int c = tid; c < kv_lora; c += 256) {
        s_ctx[c] = ctx_h[c];
    }
    __syncthreads();

    const int v_base = v0 + warp * 4;
    if (wv_type == GGML_TYPE_F16 && (kv_lora & 7) == 0 && (reinterpret_cast<uintptr_t>(wv_b) & 15u) == 0u) {
        const size_t head_off = (size_t)h * v_head * kv_lora + (size_t)v_base * kv_lora;
        const uint4* w0 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(wv_b) + head_off + (size_t)0 * kv_lora);
        const uint4* w1 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(wv_b) + head_off + (size_t)1 * kv_lora);
        const uint4* w2 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(wv_b) + head_off + (size_t)2 * kv_lora);
        const uint4* w3 = reinterpret_cast<const uint4*>(reinterpret_cast<const uint16_t*>(wv_b) + head_off + (size_t)3 * kv_lora);
        const float4* ctx4 = reinterpret_cast<const float4*>(s_ctx);
        const int n8 = kv_lora >> 3;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int c8 = lane; c8 < n8; c8 += 32) {
            float4 b0 = ctx4[c8 * 2 + 0];
            float4 b1 = ctx4[c8 * 2 + 1];
            auto dot8 = [&](uint4 u) -> float {
                return fp16_bits_to_f32((uint16_t)u.x)         * b0.x
                     + fp16_bits_to_f32((uint16_t)(u.x >> 16)) * b0.y
                     + fp16_bits_to_f32((uint16_t)u.y)         * b0.z
                     + fp16_bits_to_f32((uint16_t)(u.y >> 16)) * b0.w
                     + fp16_bits_to_f32((uint16_t)u.z)         * b1.x
                     + fp16_bits_to_f32((uint16_t)(u.z >> 16)) * b1.y
                     + fp16_bits_to_f32((uint16_t)u.w)         * b1.z
                     + fp16_bits_to_f32((uint16_t)(u.w >> 16)) * b1.w;
            };
            a0 += dot8(w0[c8]);
            a1 += dot8(w1[c8]);
            a2 += dot8(w2[c8]);
            a3 += dot8(w3[c8]);
        }
        a0 = warp_reduce_sum(a0);
        a1 = warp_reduce_sum(a1);
        a2 = warp_reduce_sum(a2);
        a3 = warp_reduce_sum(a3);
        if (lane == 0) {
            s_o[warp * 4 + 0] = a0;
            s_o[warp * 4 + 1] = a1;
            s_o[warp * 4 + 2] = a2;
            s_o[warp * 4 + 3] = a3;
        }
    } else {
        for (int j = 0; j < 4; ++j) {
            size_t wr_off = ((size_t)h * v_head + (v_base + j)) * kv_lora;
            float acc = 0.0f;
            if (wv_type == GGML_TYPE_F16) {
                const uint16_t* wr = reinterpret_cast<const uint16_t*>(wv_b) + wr_off;
                for (int c = lane; c < kv_lora; c += 32) acc += fp16_bits_to_f32(wr[c]) * s_ctx[c];
            } else {
                const float* wr = reinterpret_cast<const float*>(wv_b) + wr_off;
                for (int c = lane; c < kv_lora; c += 32) acc += wr[c] * s_ctx[c];
            }
            acc = warp_reduce_sum(acc);
            if (lane == 0) s_o[warp * 4 + j] = acc;
        }
    }
    __syncthreads();

    if (warp == 0) {
        const float v = s_o[lane];
        if (out_f32_opt != nullptr) out_f32_opt[row0 + lane] = v;
        if (out_q8 != nullptr) warp_quantize_q8_1_store(v, out_q8 + (row0 >> 5), lane);
    }
}

// FlashMLA Split-KV multi-head M=1 context kernel (kv_lora == 512, n_head % 8 == 0).
// Grid = (32 splits, n_head / 8 head groups), Block = 256 (8 warps = 8 heads sharing SMEM cell loads).
__global__ void __launch_bounds__(256) glm5_dsa_mla_split_m1_kernel(
    const float* __restrict__ q_abs,     // [n_head, 512]
    const uint8_t* __restrict__ vram_lat,// nullable [res_cap * 544]
    const uint8_t* __restrict__ ram_lat, // nullable [total_cap * 544]
    int res_cap,
    const int32_t* __restrict__ cells,   // [n_sel]
    int n_head,
    int qk_nope,
    int n_sel,
    float* __restrict__ split_scratch,   // >= n_head * 32 * 514 floats
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    constexpr int N_SPLITS = 32;
    constexpr size_t LAT_STRIDE = 544; // 512 + 512 / 16
    const int split = (int)blockIdx.x;
    const int h_group = (int)blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int h = h_group * 8 + warp;
    if (h >= n_head) return;

    __shared__ float s_red[8];
    __shared__ int s_nv;
    __shared__ __align__(16) uint8_t s_codes[4][512];
    __shared__ float s_scales[4][16];
    __shared__ int s_valid[4];

    int my_last = -1;
    for (int s = tid; s < n_sel; s += 256) {
        if (cells[s] >= 0) my_last = s;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        my_last = max(my_last, __shfl_xor_sync(0xFFFFFFFFu, my_last, o));
    }
    if (lane == 0) s_red[warp] = __int_as_float(my_last);
    __syncthreads();
    if (tid == 0) {
        int mx = -1;
        for (int w = 0; w < 8; ++w) mx = max(mx, __float_as_int(s_red[w]));
        s_nv = mx + 1;
    }
    __syncthreads();

    const int nv = s_nv;
    const int s_start = (nv * split) / N_SPLITS;
    const int s_end   = (nv * (split + 1)) / N_SPLITS;

    float* part_m = split_scratch;
    float* part_l = part_m + (size_t)n_head * N_SPLITS;
    float* part_ctx = part_l + (size_t)n_head * N_SPLITS;

    if (s_start >= s_end) {
        if (lane == 0) {
            part_m[(size_t)h * N_SPLITS + split] = -INFINITY;
            part_l[(size_t)h * N_SPLITS + split] = 0.0f;
        }
        return;
    }

    const float q_scale = rsqrtf((float)qk_nope);
    const float4* q4p = reinterpret_cast<const float4*>(q_abs + (size_t)h * 512);
    float q_reg[16];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        float4 qv = q4p[lane * 4 + k];
        q_reg[k * 4 + 0] = qv.x;
        q_reg[k * 4 + 1] = qv.y;
        q_reg[k * 4 + 2] = qv.z;
        q_reg[k * 4 + 3] = qv.w;
    }

    float acc[16];
    #pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = 0.0f;
    float m_i = -INFINITY;
    float l_i = 0.0f;

    for (int s0 = s_start; s0 < s_end; s0 += 4) {
        const int nt = min(4, s_end - s0);
        if (tid < 128) {
            const int ts = tid >> 5;
            const int c32 = tid & 31;
            if (ts < nt) {
                const int cell = cells[s0 + ts];
                const bool ok = !(cell < 0 || (!vram_lat && !ram_lat) || (res_cap > 0 && cell >= res_cap && ram_lat == nullptr));
                if (ok) {
                    const uint8_t* rec = (vram_lat && (res_cap <= 0 || cell < res_cap))
                        ? (vram_lat + (size_t)cell * LAT_STRIDE)
                        : (ram_lat + (size_t)cell * LAT_STRIDE);
                    reinterpret_cast<uint4*>(s_codes[ts])[c32] = reinterpret_cast<const uint4*>(rec)[c32];
                }
                if (c32 == 0) s_valid[ts] = ok ? 1 : 0;
            }
        } else if (tid < 192) {
            const int t2 = tid - 128;
            const int ts = t2 >> 4;
            const int sc_i = t2 & 15;
            if (ts < nt) {
                const int cell = cells[s0 + ts];
                const bool ok = !(cell < 0 || (!vram_lat && !ram_lat) || (res_cap > 0 && cell >= res_cap && ram_lat == nullptr));
                if (ok) {
                    const uint8_t* rec = (vram_lat && (res_cap <= 0 || cell < res_cap))
                        ? (vram_lat + (size_t)cell * LAT_STRIDE)
                        : (ram_lat + (size_t)cell * LAT_STRIDE);
                    s_scales[ts][sc_i] = fp16_bits_to_f32(reinterpret_cast<const uint16_t*>(rec + 512)[sc_i]);
                }
            }
        }
        __syncthreads();

        for (int ts = 0; ts < nt; ++ts) {
            if (!s_valid[ts]) continue;
            const uint4 c4 = reinterpret_cast<const uint4*>(s_codes[ts])[lane];
            const float d = s_scales[ts][lane >> 1];
            const uint32_t ws[4] = {c4.x, c4.y, c4.z, c4.w};
            float sub_dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t w = ws[k];
                const int b4 = k * 4;
                sub_dot += (float)(int8_t)(w & 0xFFu)         * q_reg[b4 + 0]
                         + (float)(int8_t)((w >> 8) & 0xFFu)  * q_reg[b4 + 1]
                         + (float)(int8_t)((w >> 16) & 0xFFu) * q_reg[b4 + 2]
                         + (float)(int8_t)((w >> 24) & 0xFFu) * q_reg[b4 + 3];
            }
            const float dot = warp_reduce_sum(sub_dot * d) * q_scale;
            float alpha = 0.0f, p = 0.0f;
            if (lane == 0) {
                const float m_new = fmaxf(m_i, dot);
                alpha = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
                p = expf(dot - m_new);
                l_i = l_i * alpha + p;
                m_i = m_new;
            }
            alpha = __shfl_sync(0xFFFFFFFFu, alpha, 0);
            p     = __shfl_sync(0xFFFFFFFFu, p, 0);
            const float pd = p * d;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t w = ws[k];
                const int b4 = k * 4;
                acc[b4 + 0] = acc[b4 + 0] * alpha + pd * (float)(int8_t)(w & 0xFFu);
                acc[b4 + 1] = acc[b4 + 1] * alpha + pd * (float)(int8_t)((w >> 8) & 0xFFu);
                acc[b4 + 2] = acc[b4 + 2] * alpha + pd * (float)(int8_t)((w >> 16) & 0xFFu);
                acc[b4 + 3] = acc[b4 + 3] * alpha + pd * (float)(int8_t)((w >> 24) & 0xFFu);
            }
        }
        __syncthreads();
    }

    const int has_valid = __shfl_sync(0xFFFFFFFFu, (m_i != -INFINITY) ? 1 : 0, 0);
    if (!has_valid) {
        if (lane == 0) {
            part_m[(size_t)h * N_SPLITS + split] = -INFINITY;
            part_l[(size_t)h * N_SPLITS + split] = 0.0f;
        }
        return;
    }
    if (lane == 0) {
        part_m[(size_t)h * N_SPLITS + split] = m_i;
        part_l[(size_t)h * N_SPLITS + split] = l_i;
    }
    float4* pctx = reinterpret_cast<float4*>(part_ctx + ((size_t)h * N_SPLITS + split) * 512);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        pctx[lane * 4 + k] = make_float4(acc[k * 4 + 0], acc[k * 4 + 1], acc[k * 4 + 2], acc[k * 4 + 3]);
    }
}

__global__ void __launch_bounds__(256) glm5_dsa_mla_split_reduce_m1_kernel(
    const float* __restrict__ split_scratch,
    int n_head,
    float* __restrict__ ctx_out,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    constexpr int N_SPLITS = 32;
    const int h = (int)blockIdx.x;
    if (h >= n_head) return;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const float* part_m = split_scratch;
    const float* part_l = part_m + (size_t)n_head * N_SPLITS;
    const float* part_ctx = part_l + (size_t)n_head * N_SPLITS;

    __shared__ float s_w[32];
    if (warp == 0) {
        const float pm = part_m[(size_t)h * N_SPLITS + lane];
        const float pl = part_l[(size_t)h * N_SPLITS + lane];
        const float m_glob = warp_reduce_max(pm);
        const float scale_sp = (pm == -INFINITY || m_glob == -INFINITY || pl <= 0.0f) ? 0.0f : expf(pm - m_glob);
        const float l_glob = warp_reduce_sum(pl * scale_sp);
        s_w[lane] = (l_glob > 0.0f) ? (scale_sp / l_glob) : 0.0f;
    }
    __syncthreads();

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    const float* pctx_h = part_ctx + (size_t)h * N_SPLITS * 512;
    for (int sp = 0; sp < N_SPLITS; ++sp) {
        const float w = s_w[sp];
        if (w > 0.0f) {
            const float* psp = pctx_h + (size_t)sp * 512;
            sum0 += w * psp[tid];
            sum1 += w * psp[tid + 256];
        }
    }
    float* ctx_h = ctx_out + (size_t)h * 512;
    ctx_h[tid] = sum0;
    ctx_h[tid + 256] = sum1;
}

extern "C" void launch_glm5_dsa_mla_vproj_q8_m1(
    const float* q_abs,
    const void* vram_lat8,
    const void* ram_lat8,
    int res_cap,
    const int32_t* cells,
    const void* wv_b,
    int32_t wv_type,
    int kv_lora,
    int n_head,
    int v_head,
    int qk_nope,
    int n_sel,
    float* ctx_scratch,
    float* split_scratch_opt,
    float* out_f32_opt,
    void* out_q8,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (n_head <= 0 || n_sel <= 0 || kv_lora <= 0 || kv_lora > 512 || (v_head & 31) != 0) return;
    float* ctx_buf = (ctx_scratch != nullptr) ? ctx_scratch : const_cast<float*>(q_abs);
    if (split_scratch_opt != nullptr && kv_lora == 512 && (n_head & 7) == 0) {
        dim3 grid_split(32, n_head >> 3);
        glm5_dsa_mla_split_m1_kernel<<<grid_split, 256, 0, stream>>>(
            q_abs,
            reinterpret_cast<const uint8_t*>(vram_lat8),
            reinterpret_cast<const uint8_t*>(ram_lat8),
            res_cap, cells, n_head, qk_nope, n_sel, split_scratch_opt, skip_dev
        );
        glm5_dsa_mla_split_reduce_m1_kernel<<<n_head, 256, 0, stream>>>(
            split_scratch_opt, n_head, ctx_buf, skip_dev
        );
    } else {
        const int n_sel_pad = (n_sel + 3) & ~3;
        const int kv_lora_pad = (kv_lora + 3) & ~3;
        const size_t ctx_floats = (kv_lora == 512) ? (8u * 512u) : (size_t)kv_lora_pad;
        const size_t smem = ((size_t)n_sel_pad + (size_t)kv_lora_pad + ctx_floats) * sizeof(float);
        glm5_dsa_mla_ctx_m1_kernel<<<n_head, 256, smem, stream>>>(
            q_abs,
            reinterpret_cast<const uint8_t*>(vram_lat8),
            reinterpret_cast<const uint8_t*>(ram_lat8),
            res_cap, cells, kv_lora, n_head, qk_nope, n_sel, ctx_buf, skip_dev
        );
    }
    const int v_blocks = n_head * (v_head >> 5);
    glm5_dsa_headwise_vproj_q8_kernel<<<v_blocks, 256, 0, stream>>>(
        ctx_buf, wv_b, wv_type, kv_lora, n_head, v_head, out_f32_opt,
        reinterpret_cast<block_q8_1*>(out_q8), skip_dev
    );
}

// -----------------------------------------------------------------------------
// Sigmoid Top-K Router + Doorbell CPU Miss Trigger & Wait + HC Post
// -----------------------------------------------------------------------------

struct Glm5Doorbell {
    volatile uint64_t req_seq;
    volatile uint64_t done_seq;
    int32_t layer_idx;
    int32_t miss_count;
    int32_t m_batch;
    int32_t gu_type;
    int32_t d_type;
    int32_t n_embd;
    int32_t n_ff;
    float   clamp_lim;
    volatile int32_t cancel_flag;
    int32_t x_q8k_valid;
    uint32_t u_off;
    uint32_t d_off;
    uint8_t x_q8k[16 * 292];
    int32_t miss_tok[32];
    int32_t miss_ids[32];
    float   miss_weights[32];
    float   x_in[16384];
    float   out[16384];
    int32_t routed_ids[48][32];
    int32_t promo_eid[48][8];
    uint64_t promo_vram_ptr[48][8];
    uint64_t pci_fetch_total;
};

static_assert(offsetof(Glm5Doorbell, layer_idx) == 16, "Glm5Doorbell layer_idx offset");
static_assert(offsetof(Glm5Doorbell, cancel_flag) == 48, "Glm5Doorbell cancel_flag offset");
static_assert(offsetof(Glm5Doorbell, u_off) == 56, "Glm5Doorbell u_off offset");
static_assert(sizeof(int32_t) == 4, "int32_t size");

__global__ void __launch_bounds__(32) glm5_skip_gate_kernel(
    const Glm5Doorbell* __restrict__ doorbell,
    int32_t* __restrict__ skip_dev
) {
    if (threadIdx.x == 0) {
        int32_t flag = 0;
        if (doorbell != nullptr) {
            const int32_t* cf_addr = reinterpret_cast<const int32_t*>(const_cast<const int32_t*>(&doorbell->cancel_flag));
            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(flag) : "l"(cf_addr) : "memory");
        }
        *skip_dev = flag;
    }
}

extern "C" void launch_glm5_skip_gate(
    const void* doorbell_mapped,
    int32_t* skip_dev,
    cudaStream_t stream
) {
    if (skip_dev == nullptr) return;
    glm5_skip_gate_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<const Glm5Doorbell*>(doorbell_mapped), skip_dev
    );
}

__device__ __forceinline__ void glm5_router_select_token_dev(
    const float* __restrict__ lt,
    const float* __restrict__ bias,
    int n_expert,
    int topk,
    float w_scale,
    int norm_w,
    int32_t* __restrict__ io,
    float* __restrict__ ro,
    float* s_p,
    float* s_sel,
    uint8_t* s_used,
    float* s_best_v,
    int32_t* s_best_i
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int e = tid; e < n_expert; e += 256) {
        float p = 1.0f / (1.0f + expf(-lt[e]));
        float s = bias ? (p + bias[e]) : p;
        s_p[e] = p;
        s_sel[e] = s;
        s_used[e] = 0;
    }
    __syncthreads();

    for (int k = 0; k < topk; ++k) {
        float best_s = -INFINITY;
        int best_e = n_expert;
        for (int e = tid; e < n_expert; e += 256) {
            if (!s_used[e]) {
                float s = s_sel[e];
                if (s > best_s || (s == best_s && e < best_e)) {
                    best_s = s;
                    best_e = e;
                }
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float os = __shfl_xor_sync(0xFFFFFFFFu, best_s, off);
            int oe   = __shfl_xor_sync(0xFFFFFFFFu, best_e, off);
            if (os > best_s || (os == best_s && oe < best_e)) {
                best_s = os;
                best_e = oe;
            }
        }
        if (lane == 0) {
            s_best_v[warp] = best_s;
            s_best_i[warp] = best_e;
        }
        __syncthreads();
        if (tid == 0) {
            float gs = -INFINITY;
            int ge = n_expert;
            for (int w = 0; w < 8; ++w) {
                float ws = s_best_v[w];
                int we = s_best_i[w];
                if (ws > gs || (ws == gs && we < ge)) {
                    gs = ws;
                    ge = we;
                }
            }
            s_used[ge] = 1;
            io[k] = ge;
            ro[k] = s_p[ge];
        }
        __syncthreads();
    }

    if (tid == 0) {
        double sum_w = 0.0;
        for (int k = 0; k < topk; ++k) sum_w += (double)ro[k];
        if (norm_w) {
            double denom = sum_w > (1.0 / 16384.0) ? sum_w : (1.0 / 16384.0);
            for (int k = 0; k < topk; ++k) ro[k] = (float)(((double)ro[k] / denom) * (double)w_scale);
        } else {
            for (int k = 0; k < topk; ++k) ro[k] = (float)((double)ro[k] * (double)w_scale);
        }
    }
}

__device__ __forceinline__ void glm5_doorbell_write_x_m1(
    Glm5Doorbell* __restrict__ doorbell,
    const float* __restrict__ x_f32,
    int n_embd,
    uint32_t* s_q8k_words
) {
    const int tid = threadIdx.x;
    if ((n_embd & 255) == 0 && n_embd <= 4096) {
        const int nb = n_embd >> 8;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        for (int b = warp; b < nb; b += 8) {
            const float* xp = x_f32 + (size_t)b * 256 + (size_t)lane * 8;
            const float4 f0 = reinterpret_cast<const float4*>(xp)[0];
            const float4 f1 = reinterpret_cast<const float4*>(xp)[1];
            float amax = fmaxf(
                fmaxf(fmaxf(fabsf(f0.x), fabsf(f0.y)), fmaxf(fabsf(f0.z), fabsf(f0.w))),
                fmaxf(fmaxf(fabsf(f1.x), fabsf(f1.y)), fmaxf(fabsf(f1.z), fabsf(f1.w)))
            );
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
            }
            const float d = amax / 127.0f;
            const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            const float vals[8] = { f0.x, f0.y, f0.z, f0.w, f1.x, f1.y, f1.z, f1.w };
            int qs[8];
            int s8 = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                float scaled = vals[k] * id;
                if (scaled < -127.0f) scaled = -127.0f;
                if (scaled >  127.0f) scaled =  127.0f;
                int q = __float2int_rn(scaled);
                qs[k] = q;
                s8 += q;
            }
            uint32_t* bw = s_q8k_words + (size_t)b * 73;
            if (lane == 0) bw[0] = __float_as_uint(d);
            uint32_t p0 = ((uint32_t)(uint8_t)(int8_t)qs[0])
                        | (((uint32_t)(uint8_t)(int8_t)qs[1]) << 8)
                        | (((uint32_t)(uint8_t)(int8_t)qs[2]) << 16)
                        | (((uint32_t)(uint8_t)(int8_t)qs[3]) << 24);
            uint32_t p1 = ((uint32_t)(uint8_t)(int8_t)qs[4])
                        | (((uint32_t)(uint8_t)(int8_t)qs[5]) << 8)
                        | (((uint32_t)(uint8_t)(int8_t)qs[6]) << 16)
                        | (((uint32_t)(uint8_t)(int8_t)qs[7]) << 24);
            bw[1 + lane * 2 + 0] = p0;
            bw[1 + lane * 2 + 1] = p1;
            int s16 = s8 + __shfl_xor_sync(0xFFFFFFFFu, s8, 1);
            int s16_hi = __shfl_xor_sync(0xFFFFFFFFu, s16, 2);
            if ((lane & 3) == 0) {
                bw[65 + (lane >> 2)] = ((uint32_t)(uint16_t)(int16_t)s16)
                                     | (((uint32_t)(uint16_t)(int16_t)s16_hi) << 16);
            }
        }
        __syncthreads();
        uint32_t* dst_q8k = reinterpret_cast<uint32_t*>(doorbell->x_q8k);
        const int n_words = nb * 73;
        for (int i = tid; i < n_words; i += 256) {
            dst_q8k[i] = s_q8k_words[i];
        }
        if (tid == 0) {
            doorbell->x_q8k_valid = 1;
        }
    } else {
        for (int e = tid; e < n_embd; e += 256) {
            doorbell->x_in[e] = x_f32[e];
        }
        if (tid == 0) {
            doorbell->x_q8k_valid = 0;
        }
    }
}

__device__ __forceinline__ void glm5_doorbell_quant_and_ring_m1(
    Glm5Doorbell* __restrict__ doorbell,
    const float* __restrict__ x_f32,
    int n_embd,
    int s_miss_cnt,
    uint64_t s_exp_seq,
    uint32_t* s_q8k_words
) {
    const int tid = threadIdx.x;
    if (s_miss_cnt > 0 && x_f32 != nullptr && doorbell != nullptr) {
        glm5_doorbell_write_x_m1(doorbell, x_f32, n_embd, s_q8k_words);
        __threadfence_system();
        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            doorbell->req_seq = s_exp_seq;
            __threadfence_system();
        }
    }
}

__global__ void __launch_bounds__(256) glm5_router_sigmoid_probe_kernel(
    const float* __restrict__ logits,       // [M, n_expert]
    const float* __restrict__ bias,         // nullable [n_expert]
    int n_expert,
    int topk,
    float w_scale,
    int norm_w,
    const uint8_t* __restrict__ res_mask,   // nullable [n_expert] (1 = on GPU, 0 = on CPU)
    const float* __restrict__ x_f32,        // nullable [M, n_embd] (for doorbell M==1)
    int n_embd,
    int n_ff,
    int layer_idx,
    int32_t gu_type,
    int32_t d_type,
    float clamp_lim,
    int32_t* __restrict__ ids_out,          // [M, topk]
    float* __restrict__ rw_out,             // [M, topk]
    Glm5Doorbell* __restrict__ doorbell,    // nullable mapped host buffer (when M==1)
    uint64_t* __restrict__ seq_state_dev,   // nullable [1] monotonic seq counter
    uint64_t* __restrict__ expected_seq_out,// nullable [1] 0 if miss_count==0 else seq
    const int32_t* __restrict__ skip_dev    // nullable [1] SPEC_CANCEL gate
) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lt = logits + (size_t)t * n_expert;
    int32_t* io = ids_out + (size_t)t * topk;
    float* ro = rw_out + (size_t)t * topk;

    __shared__ float s_p[512];
    __shared__ float s_sel[512];
    __shared__ uint8_t s_used[512];
    __shared__ float s_best_v[8];
    __shared__ int32_t s_best_i[8];
    __shared__ int32_t s_miss_cnt;
    __shared__ uint64_t s_exp_seq;
    __shared__ int32_t s_abort;
    __shared__ __align__(16) uint32_t s_q8k_words[16 * 73];

    if (tid == 0) {
        int32_t ab = 0;
        if (skip_dev != nullptr && *skip_dev != 0) {
            ab = 1;
        } else if (doorbell != nullptr && t == 0) {
            const int32_t* cf_addr = reinterpret_cast<const int32_t*>(const_cast<const int32_t*>(&doorbell->cancel_flag));
            int32_t cf = 0;
            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(cf) : "l"(cf_addr) : "memory");
            if (cf != 0) ab = 1;
        }
        if (ab != 0 && t == 0 && expected_seq_out) *expected_seq_out = 0ULL;
        s_abort = ab;
    }
    __syncthreads();
    if (s_abort != 0) return;

    glm5_router_select_token_dev(
        lt, bias, n_expert, topk, w_scale, norm_w,
        io, ro, s_p, s_sel, s_used, s_best_v, s_best_i
    );

    if (tid == 0) {
        if (doorbell != nullptr && t == 0) {
            if (layer_idx >= 0 && layer_idx < 48) {
                for (int k = 0; k < topk && k < 8; ++k) {
                    doorbell->routed_ids[layer_idx][k] = io[k];
                }
            }
            int mc = 0;
            for (int k = 0; k < topk; ++k) {
                int eid = io[k];
                if (res_mask && res_mask[eid] == 0 && mc < 32) {
                    doorbell->miss_tok[mc] = 0;
                    doorbell->miss_ids[mc] = eid;
                    doorbell->miss_weights[mc] = ro[k];
                    ++mc;
                }
            }
            s_miss_cnt = mc;
            doorbell->m_batch = 1;
            if (mc > 0) {
                uint64_t next_s = (*seq_state_dev) + 1ULL;
                *seq_state_dev = next_s;
                doorbell->layer_idx = layer_idx;
                doorbell->miss_count = mc;
                doorbell->gu_type = gu_type;
                doorbell->d_type = d_type;
                doorbell->n_embd = n_embd;
                doorbell->n_ff = n_ff;
                doorbell->clamp_lim = clamp_lim;
                s_exp_seq = next_s;
            } else {
                doorbell->miss_count = 0;
                s_exp_seq = 0ULL;
            }
            if (expected_seq_out) *expected_seq_out = s_exp_seq;
        } else {
            s_miss_cnt = 0;
            s_exp_seq = 0ULL;
        }
    }
    __syncthreads();

    if (t == 0) {
        glm5_doorbell_quant_and_ring_m1(doorbell, x_f32, n_embd, s_miss_cnt, s_exp_seq, s_q8k_words);
    }
}

__global__ void __launch_bounds__(256) glm5_router_dual_lane_kernel(
    const float* __restrict__ logits,       // [n_expert]
    const float* __restrict__ bias,         // nullable [n_expert]
    int n_expert,
    int topk,
    float w_scale,
    int norm_w,
    uint64_t* __restrict__ vram_ptrs,       // [n_expert]
    const uint64_t* __restrict__ ram_ptrs,  // nullable [n_expert]
    uint64_t* __restrict__ spares,          // nullable [k_spares]
    int k_spares,
    uint32_t* __restrict__ dcnt,            // nullable [n_expert]
    uint32_t* __restrict__ route_cnt_dev,   // nullable [1]
    const uint64_t* __restrict__ scratch_ptrs, // nullable [8]
    uint64_t cpu_plan,
    int promote_min,
    int skip_from,
    uint64_t* __restrict__ plan_ptrs_out,   // [8]
    uint64_t* __restrict__ fetch_src_out,   // [8]
    const float* __restrict__ x_f32,        // [n_embd]
    int n_embd,
    int n_ff,
    int layer_idx,
    int32_t gu_type,
    int32_t d_type,
    float clamp_lim,
    uint32_t u_off,
    uint32_t d_off,
    const float* __restrict__ sh_gate,      // nullable [n_ff_sh]
    const float* __restrict__ sh_up,        // nullable [n_ff_sh]
    float sh_clamp_lim,
    int n_ff_sh,
    block_q8_1* __restrict__ sh_hq,         // nullable [n_ff_sh / 32]
    int32_t* __restrict__ ids_out,          // [topk]
    float* __restrict__ rw_out,             // [topk]
    Glm5Doorbell* __restrict__ doorbell,    // nullable mapped host buffer
    uint64_t* __restrict__ seq_state_dev,   // nullable [2]: [0]=doorbell seq, [1]=pci_fetch_total
    uint64_t* __restrict__ expected_seq_out,// nullable [1]
    const int32_t* __restrict__ pos_arr_dev,// nullable [1] double-buffered routed_ids slot (pos & 1)
    const int32_t* __restrict__ skip_dev,   // nullable [1]
    const int32_t* __restrict__ pf_in_eids, // nullable [2] incoming prefetched eids for layer l
    const uint64_t* __restrict__ pf_in_ptrs,// nullable [2] incoming prefetched VRAM ptrs for layer l
    const float* __restrict__ pf_logits_next, // nullable [n_expert]
    const float* __restrict__ pf_bias_next,   // nullable [n_expert]
    const uint64_t* __restrict__ vram_ptrs_next, // nullable [n_expert]
    const uint64_t* __restrict__ ram_ptrs_next,  // nullable [n_expert]
    int max_pf,
    uint64_t* __restrict__ pf_out_src,      // nullable [2]
    uint64_t* __restrict__ pf_out_dst,      // nullable [2]
    int32_t* __restrict__ pf_out_eids,      // nullable [2]
    uint64_t* __restrict__ pf_out_ptrs,     // nullable [2]
    int32_t* __restrict__ pf_out_n          // nullable [1]
) {
    const int tid = threadIdx.x;
    if (blockIdx.x > 0) {
        if (skip_dev != nullptr && *skip_dev != 0) return;
        const int warp = tid >> 5;
        const int lane = tid & 31;
        const int blk = ((int)blockIdx.x - 1) * 8 + warp;
        const int n_blocks = n_ff_sh >> 5;
        if (blk >= n_blocks) return;
        const int idx = (blk << 5) + lane;
        float h = glm_swiglu_clamp_val(sh_gate[idx], sh_up[idx], sh_clamp_lim);
        warp_quantize_q8_1_store(h, sh_hq + blk, lane);
        return;
    }

    __shared__ float s_p[512];
    __shared__ float s_sel[512];
    __shared__ uint8_t s_used[512];
    __shared__ float s_best_v[8];
    __shared__ int32_t s_best_i[8];
    __shared__ int32_t s_miss_cnt;
    __shared__ uint64_t s_exp_seq;
    __shared__ int32_t s_abort;
    __shared__ int32_t s_do_age;
    __shared__ int32_t s_routed[8];
    __shared__ int32_t s_rslot;
    __shared__ int32_t s_miss_ids[8];
    __shared__ float s_miss_w[8];
    __shared__ int32_t s_promo_s[8];
    __shared__ int32_t s_promo_e[8];
    __shared__ uint64_t s_promo_p[8];
    __shared__ int32_t s_n_promo;
    __shared__ uint64_t s_hdr_u64[4];
    __shared__ uint64_t s_off_u64;
    __shared__ uint64_t s_next_pci;
    __shared__ int32_t s_n_pci;
    __shared__ int32_t s_pf_ids[8];
    __shared__ float s_pf_rw[8];
    __shared__ __align__(16) uint32_t s_q8k_words[16 * 73];

    if (tid == 0) {
        int32_t ab = 0;
        if (skip_dev != nullptr && *skip_dev != 0) {
            ab = 1;
        } else if (doorbell != nullptr) {
            const int32_t* cf_addr = reinterpret_cast<const int32_t*>(const_cast<const int32_t*>(&doorbell->cancel_flag));
            int32_t cf = 0;
            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(cf) : "l"(cf_addr) : "memory");
            if (cf != 0) ab = 1;
        }
        s_abort = ab;
        if (ab == 0) {
            int32_t ag = 0;
            if (route_cnt_dev != nullptr && dcnt != nullptr) {
                uint32_t rc = *route_cnt_dev + 1u;
                *route_cnt_dev = rc;
                if ((rc & 4095u) == 0u) ag = 1;
            }
            s_do_age = ag;
        } else {
            s_do_age = 0;
        }
    }
    __syncthreads();

    if (s_abort != 0) {
        if (tid == 0) {
            if (expected_seq_out != nullptr) *expected_seq_out = 0ULL;
            if (pf_out_n != nullptr) *pf_out_n = 0;
        }
        if (tid < 8) {
            plan_ptrs_out[tid] = 0ULL;
            fetch_src_out[tid] = 0ULL;
        }
        if (tid < 2) {
            if (pf_out_eids != nullptr) pf_out_eids[tid] = -1;
            if (pf_out_ptrs != nullptr) pf_out_ptrs[tid] = 0ULL;
            if (pf_out_src != nullptr) pf_out_src[tid] = 0ULL;
            if (pf_out_dst != nullptr) pf_out_dst[tid] = 0ULL;
        }
        return;
    }

    if (s_do_age != 0 && dcnt != nullptr) {
        for (int e = tid; e < n_expert; e += 256) {
            dcnt[e] >>= 1;
        }
        __syncthreads();
    }

    glm5_router_select_token_dev(
        logits, bias, n_expert, topk, w_scale, norm_w,
        ids_out, rw_out, s_p, s_sel, s_used, s_best_v, s_best_i
    );
    __syncthreads();

    if (tid == 0) {
        int mc = 0;
        int np = 0;
        int n_pci = 0;
        int f_ram = 0;
        int ram_ks[8];
        uint32_t ram_cnts[8];
        uint64_t pf_hit_ptr[8] = {0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL};
        int sc_idx = 0;
        const int sc_cap = (max_pf > 0 || pf_in_eids != nullptr) ? 4 : 8;
        int32_t in_pf_e[2] = {-1, -1};
        uint64_t in_pf_p[2] = {0ULL, 0ULL};
        if (pf_in_eids != nullptr && pf_in_ptrs != nullptr) {
            in_pf_e[0] = pf_in_eids[0];
            in_pf_e[1] = pf_in_eids[1];
            in_pf_p[0] = pf_in_ptrs[0];
            in_pf_p[1] = pf_in_ptrs[1];
        }
        s_rslot = (pos_arr_dev != nullptr) ? (pos_arr_dev[0] & 1) : 0;

        for (int k = 0; k < topk && k < 8; ++k) {
            int e = ids_out[k];
            if (dcnt != nullptr) dcnt[e] += 1u;
            s_routed[k] = e;
            uint64_t vp = vram_ptrs[e];
            uint64_t rp = (ram_ptrs != nullptr) ? ram_ptrs[e] : 0ULL;
            if (vp == 0ULL && rp != 0ULL && k < skip_from) {
                uint64_t hp = 0ULL;
                if (in_pf_e[0] == e && in_pf_p[0] != 0ULL) hp = in_pf_p[0];
                else if (in_pf_e[1] == e && in_pf_p[1] != 0ULL) hp = in_pf_p[1];
                if (hp != 0ULL) {
                    pf_hit_ptr[k] = hp;
                } else {
                    ram_ks[f_ram] = k;
                    ram_cnts[f_ram] = (dcnt != nullptr) ? dcnt[e] : 1u;
                    ++f_ram;
                }
            }
        }

        int to_cpu = (cpu_plan != 0ULL && f_ram > 0 && doorbell != nullptr)
            ? (int)((cpu_plan >> (4 * f_ram)) & 15ULL)
            : 0;
        if (to_cpu > f_ram) to_cpu = f_ram;

        uint32_t cpu_mask = 0u;
        for (int t = 0; t < to_cpu; ++t) {
            int best_i = -1;
            for (int j = 0; j < f_ram; ++j) {
                int kk = ram_ks[j];
                if ((cpu_mask >> kk) & 1u) continue;
                if (best_i < 0
                    || ram_cnts[j] < ram_cnts[best_i]
                    || (ram_cnts[j] == ram_cnts[best_i] && kk > ram_ks[best_i])) {
                    best_i = j;
                }
            }
            if (best_i >= 0) cpu_mask |= (1u << ram_ks[best_i]);
        }

        for (int k = 0; k < topk && k < 8; ++k) {
            int e = ids_out[k];
            uint64_t vp = vram_ptrs[e];
            uint64_t rp = (ram_ptrs != nullptr) ? ram_ptrs[e] : 0ULL;
            if (vp != 0ULL) {
                plan_ptrs_out[k] = vp;
                fetch_src_out[k] = 0ULL;
            } else if (k >= skip_from) {
                plan_ptrs_out[k] = 0ULL;
                fetch_src_out[k] = 0ULL;
            } else if (pf_hit_ptr[k] != 0ULL) {
                plan_ptrs_out[k] = pf_hit_ptr[k];
                fetch_src_out[k] = 0ULL;
            } else if (rp != 0ULL && ((cpu_mask >> k) & 1u)) {
                plan_ptrs_out[k] = 0ULL;
                fetch_src_out[k] = 0ULL;
                if (doorbell != nullptr && mc < 8) {
                    s_miss_ids[mc] = e;
                    s_miss_w[mc] = rw_out[k];
                    ++mc;
                }
            } else if (rp != 0ULL) {
                uint64_t dst_slot = 0ULL;
                uint32_t rc = (dcnt != nullptr) ? dcnt[e] : 1u;
                if (spares != nullptr && k_spares > 0 && (int)rc >= promote_min) {
                    for (int s = 0; s < k_spares; ++s) {
                        if (spares[s] != 0ULL) {
                            dst_slot = spares[s];
                            spares[s] = 0ULL;
                            vram_ptrs[e] = dst_slot;
                            if (doorbell != nullptr && layer_idx >= 0 && layer_idx < 48 && s < 8 && np < 8) {
                                s_promo_s[np] = s;
                                s_promo_e[np] = e;
                                s_promo_p[np] = dst_slot;
                                ++np;
                            }
                            break;
                        }
                    }
                }
                if (dst_slot == 0ULL && scratch_ptrs != nullptr && sc_idx < sc_cap) {
                    dst_slot = scratch_ptrs[sc_idx++];
                }
                plan_ptrs_out[k] = dst_slot;
                fetch_src_out[k] = (dst_slot != 0ULL) ? rp : 0ULL;
                if (dst_slot != 0ULL) ++n_pci;
            } else {
                plan_ptrs_out[k] = 0ULL;
                fetch_src_out[k] = 0ULL;
            }
        }
        for (int k = topk; k < 8; ++k) {
            plan_ptrs_out[k] = 0ULL;
            fetch_src_out[k] = 0ULL;
        }

        uint64_t next_pci = 0ULL;
        if (n_pci > 0 && seq_state_dev != nullptr) {
            next_pci = seq_state_dev[1] + (uint64_t)n_pci;
            seq_state_dev[1] = next_pci;
        }
        s_next_pci = next_pci;
        s_n_pci = n_pci;
        s_n_promo = np;
        s_miss_cnt = mc;

        if (mc > 0 && doorbell != nullptr && seq_state_dev != nullptr) {
            uint64_t next_s = seq_state_dev[0] + 1ULL;
            seq_state_dev[0] = next_s;
            s_hdr_u64[0] = ((uint64_t)(uint32_t)layer_idx) | (((uint64_t)(uint32_t)mc) << 32);
            s_hdr_u64[1] = 1ULL | (((uint64_t)(uint32_t)gu_type) << 32);
            s_hdr_u64[2] = ((uint64_t)(uint32_t)d_type) | (((uint64_t)(uint32_t)n_embd) << 32);
            s_hdr_u64[3] = ((uint64_t)(uint32_t)n_ff) | (((uint64_t)__float_as_uint(clamp_lim)) << 32);
            s_off_u64 = ((uint64_t)u_off) | (((uint64_t)d_off) << 32);
            s_exp_seq = next_s;
        } else {
            s_exp_seq = 0ULL;
        }
        if (expected_seq_out) *expected_seq_out = s_exp_seq;
    }
    __syncthreads();

    if (doorbell != nullptr) {
        if (tid < topk && tid < 8 && layer_idx >= 0 && layer_idx < 48) {
            doorbell->routed_ids[layer_idx][s_rslot * 8 + tid] = s_routed[tid];
        }
        if (tid < s_n_promo && layer_idx >= 0 && layer_idx < 48) {
            doorbell->promo_vram_ptr[layer_idx][s_promo_s[tid]] = s_promo_p[tid];
        }
        if (s_miss_cnt > 0) {
            if (tid < 4) {
                reinterpret_cast<uint64_t*>(const_cast<int32_t*>(&doorbell->layer_idx))[tid] = s_hdr_u64[tid];
            } else if (tid == 4) {
                *reinterpret_cast<uint64_t*>(const_cast<uint32_t*>(&doorbell->u_off)) = s_off_u64;
            }
            if (tid < s_miss_cnt) {
                doorbell->miss_tok[tid] = 0;
                doorbell->miss_ids[tid] = s_miss_ids[tid];
                doorbell->miss_weights[tid] = s_miss_w[tid];
            }
        }
        if (tid == 5 && s_n_pci > 0 && seq_state_dev != nullptr) {
            doorbell->pci_fetch_total = s_next_pci;
        }
    }

    if (s_miss_cnt > 0 && x_f32 != nullptr && doorbell != nullptr) {
        glm5_doorbell_write_x_m1(doorbell, x_f32, n_embd, s_q8k_words);
    }

    if (doorbell != nullptr && ((s_miss_cnt > 0 && x_f32 != nullptr) || s_n_promo > 0)) {
        if (s_miss_cnt > 0 || tid < 32) {
            __threadfence_system();
        }
        __syncthreads();
        if (tid < s_n_promo && layer_idx >= 0 && layer_idx < 48) {
            doorbell->promo_eid[layer_idx][s_promo_s[tid]] = s_promo_e[tid];
        }
        if (tid == 0 && s_miss_cnt > 0 && x_f32 != nullptr) {
            doorbell->req_seq = s_exp_seq;
        }
        if (tid < 32 && (tid == 0 || tid < s_n_promo)) {
            __threadfence_system();
        }
    }

    if (max_pf > 0 && pf_logits_next != nullptr && vram_ptrs_next != nullptr && ram_ptrs_next != nullptr && pf_out_n != nullptr) {
        __syncthreads();
        glm5_router_select_token_dev(
            pf_logits_next, pf_bias_next, n_expert, topk, w_scale, norm_w,
            s_pf_ids, s_pf_rw, s_p, s_sel, s_used, s_best_v, s_best_i
        );
        __syncthreads();
        if (tid == 0) {
            int npf = 0;
            const int pf_lim = (max_pf < 2) ? max_pf : 2;
            const int pf_slot_base = 4 + (((layer_idx + 1) & 1) * 2);
            int32_t picked0 = -1;
            for (int k = 0; k < topk && k < 8 && npf < pf_lim; ++k) {
                int e_n = s_pf_ids[k];
                if (e_n < 0 || e_n >= n_expert) continue;
                if (vram_ptrs_next[e_n] != 0ULL) continue;
                uint64_t rp_n = ram_ptrs_next[e_n];
                if (rp_n == 0ULL) continue;
                if (npf == 1 && picked0 == e_n) continue;
                uint64_t dst = (scratch_ptrs != nullptr) ? scratch_ptrs[pf_slot_base + npf] : 0ULL;
                if (dst == 0ULL) break;
                if (npf == 0) picked0 = e_n;
                if (pf_out_src != nullptr) pf_out_src[npf] = rp_n;
                if (pf_out_dst != nullptr) pf_out_dst[npf] = dst;
                if (pf_out_eids != nullptr) pf_out_eids[npf] = e_n;
                if (pf_out_ptrs != nullptr) pf_out_ptrs[npf] = dst;
                ++npf;
            }
            for (int u = npf; u < 2; ++u) {
                if (pf_out_src != nullptr) pf_out_src[u] = 0ULL;
                if (pf_out_dst != nullptr) pf_out_dst[u] = 0ULL;
                if (pf_out_eids != nullptr) pf_out_eids[u] = -1;
                if (pf_out_ptrs != nullptr) pf_out_ptrs[u] = 0ULL;
            }
            *pf_out_n = npf;
        }
    } else if (tid == 0 && pf_out_n != nullptr) {
        *pf_out_n = 0;
        if (pf_out_eids != nullptr) { pf_out_eids[0] = -1; pf_out_eids[1] = -1; }
        if (pf_out_ptrs != nullptr) { pf_out_ptrs[0] = 0ULL; pf_out_ptrs[1] = 0ULL; }
    }
}

extern "C" void launch_glm5_router_dual_lane_pf(
    const float* logits,
    const float* bias,
    int n_expert,
    int topk,
    float w_scale,
    int norm_w,
    uint64_t* vram_ptrs,
    const uint64_t* ram_ptrs,
    uint64_t* spares,
    int k_spares,
    uint32_t* dcnt,
    uint32_t* route_cnt_dev,
    const uint64_t* scratch_ptrs,
    uint64_t cpu_plan,
    int promote_min,
    int skip_from,
    uint64_t* plan_ptrs_out,
    uint64_t* fetch_src_out,
    const float* x_f32,
    int n_embd,
    int n_ff,
    int layer_idx,
    int32_t gu_type,
    int32_t d_type,
    float clamp_lim,
    uint32_t u_off,
    uint32_t d_off,
    const float* sh_gate,
    const float* sh_up,
    float sh_clamp_lim,
    int n_ff_sh,
    void* sh_hq,
    int32_t* ids_out,
    float* rw_out,
    void* doorbell_mapped,
    uint64_t* seq_state_dev,
    uint64_t* expected_seq_out,
    const int32_t* pos_arr_dev,
    const int32_t* skip_dev,
    const int32_t* pf_in_eids,
    const uint64_t* pf_in_ptrs,
    const float* pf_logits_next,
    const float* pf_bias_next,
    const uint64_t* vram_ptrs_next,
    const uint64_t* ram_ptrs_next,
    int max_pf,
    uint64_t* pf_out_src,
    uint64_t* pf_out_dst,
    int32_t* pf_out_eids,
    uint64_t* pf_out_ptrs,
    int32_t* pf_out_n,
    cudaStream_t stream
) {
    if (n_expert <= 0 || topk <= 0) return;
    int sh_blocks = (n_ff_sh > 0 && sh_gate != nullptr && sh_up != nullptr && sh_hq != nullptr)
        ? ((n_ff_sh + 255) / 256)
        : 0;
    glm5_router_dual_lane_kernel<<<1 + sh_blocks, 256, 0, stream>>>(
        logits, bias, n_expert, topk, w_scale, norm_w, vram_ptrs, ram_ptrs, spares, k_spares,
        dcnt, route_cnt_dev, scratch_ptrs, cpu_plan, promote_min, skip_from,
        plan_ptrs_out, fetch_src_out, x_f32, n_embd, n_ff, layer_idx, gu_type, d_type,
        clamp_lim, u_off, d_off, sh_gate, sh_up, sh_clamp_lim, n_ff_sh,
        reinterpret_cast<block_q8_1*>(sh_hq), ids_out, rw_out,
        reinterpret_cast<Glm5Doorbell*>(doorbell_mapped), seq_state_dev, expected_seq_out,
        pos_arr_dev, skip_dev, pf_in_eids, pf_in_ptrs, pf_logits_next, pf_bias_next,
        vram_ptrs_next, ram_ptrs_next, max_pf, pf_out_src, pf_out_dst, pf_out_eids,
        pf_out_ptrs, pf_out_n
    );
}

extern "C" void launch_glm5_router_dual_lane(
    const float* logits,
    const float* bias,
    int n_expert,
    int topk,
    float w_scale,
    int norm_w,
    uint64_t* vram_ptrs,
    const uint64_t* ram_ptrs,
    uint64_t* spares,
    int k_spares,
    uint32_t* dcnt,
    uint32_t* route_cnt_dev,
    const uint64_t* scratch_ptrs,
    uint64_t cpu_plan,
    int promote_min,
    int skip_from,
    uint64_t* plan_ptrs_out,
    uint64_t* fetch_src_out,
    const float* x_f32,
    int n_embd,
    int n_ff,
    int layer_idx,
    int32_t gu_type,
    int32_t d_type,
    float clamp_lim,
    uint32_t u_off,
    uint32_t d_off,
    const float* sh_gate,
    const float* sh_up,
    float sh_clamp_lim,
    int n_ff_sh,
    void* sh_hq,
    int32_t* ids_out,
    float* rw_out,
    void* doorbell_mapped,
    uint64_t* seq_state_dev,
    uint64_t* expected_seq_out,
    const int32_t* pos_arr_dev,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    launch_glm5_router_dual_lane_pf(
        logits, bias, n_expert, topk, w_scale, norm_w, vram_ptrs, ram_ptrs, spares, k_spares,
        dcnt, route_cnt_dev, scratch_ptrs, cpu_plan, promote_min, skip_from,
        plan_ptrs_out, fetch_src_out, x_f32, n_embd, n_ff, layer_idx, gu_type, d_type,
        clamp_lim, u_off, d_off, sh_gate, sh_up, sh_clamp_lim, n_ff_sh, sh_hq,
        ids_out, rw_out, doorbell_mapped, seq_state_dev, expected_seq_out,
        pos_arr_dev, skip_dev, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
        nullptr, nullptr, nullptr, nullptr, nullptr, stream
    );
}

// Note: Intentionally does NOT check skip_dev; router_dual_lane zeroes fetch_src when cancelled
// so any in-flight spare-slot transfer completes atomically without leaving partial weights.
__global__ void __launch_bounds__(256) glm5_moe_fetch_kernel(
    const uint64_t* __restrict__ fetch_src,
    const uint64_t* __restrict__ plan_ptrs,
    int topk,
    size_t blob_bytes
) {
    const int bid = (int)blockIdx.x;
    const int nblk = (int)gridDim.x;
    const int ltid = (int)threadIdx.x;
    const int bdim = (int)blockDim.x;
    const int gid = bid * bdim + ltid;
    const int gstride = nblk * bdim;
    const size_t n16 = blob_bytes >> 4;
    const size_t tail0 = n16 << 4;
    const size_t per_blk = (n16 + (size_t)nblk - 1) / (size_t)nblk;
    const size_t c_start = (size_t)bid * per_blk;
    const size_t c_end = (c_start < n16) ? min(n16, c_start + per_blk) : c_start;
    const size_t step4 = (size_t)bdim * 4;

    for (int i = 0; i < topk && i < 8; ++i) {
        const uint64_t s_addr = fetch_src[i];
        if (s_addr == 0ULL) continue;
        const uint64_t d_addr = plan_ptrs[i];
        if (d_addr == 0ULL) continue;
        const uint8_t* sp = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(s_addr));
        uint8_t* dp = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(d_addr));
        if (((s_addr | d_addr) & 15ULL) == 0ULL) {
            const uint4* sv = reinterpret_cast<const uint4*>(sp);
            uint4* dv = reinterpret_cast<uint4*>(dp);
            size_t j = c_start + (size_t)ltid;
            for (; j + (size_t)bdim * 3 < c_end; j += step4) {
                const uint4 v0 = __ldcs(sv + j);
                const uint4 v1 = __ldcs(sv + j + bdim);
                const uint4 v2 = __ldcs(sv + j + (size_t)bdim * 2);
                const uint4 v3 = __ldcs(sv + j + (size_t)bdim * 3);
                __stcs(dv + j, v0);
                __stcs(dv + j + bdim, v1);
                __stcs(dv + j + (size_t)bdim * 2, v2);
                __stcs(dv + j + (size_t)bdim * 3, v3);
            }
            for (; j < c_end; j += bdim) {
                __stcs(dv + j, __ldcs(sv + j));
            }
            for (size_t t = tail0 + gid; t < blob_bytes; t += gstride) {
                dp[t] = sp[t];
            }
        } else {
            for (size_t j = gid; j < blob_bytes; j += gstride) {
                dp[j] = sp[j];
            }
        }
    }
}

extern "C" void launch_glm5_moe_fetch_nblocks(
    const uint64_t* fetch_src,
    const uint64_t* plan_ptrs,
    int topk,
    size_t blob_bytes,
    int n_blocks,
    cudaStream_t stream
) {
    if (topk <= 0 || blob_bytes == 0 || n_blocks <= 0) return;
    glm5_moe_fetch_kernel<<<n_blocks, 256, 0, stream>>>(fetch_src, plan_ptrs, topk, blob_bytes);
}

extern "C" void launch_glm5_moe_fetch(
    const uint64_t* fetch_src,
    const uint64_t* plan_ptrs,
    int topk,
    size_t blob_bytes,
    cudaStream_t stream
) {
    if (topk <= 0 || blob_bytes == 0) return;
    launch_glm5_moe_fetch_nblocks(fetch_src, plan_ptrs, topk, blob_bytes, 64, stream);
}

__global__ void __launch_bounds__(256) glm5_moe_prefetch_kernel(
    const uint64_t* __restrict__ pf_src,
    const uint64_t* __restrict__ pf_dst,
    const int32_t* __restrict__ pf_n,
    size_t blob_bytes
) {
    const int n = *pf_n;
    if (n <= 0) return;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    const size_t n16 = blob_bytes >> 4;
    const size_t tail0 = n16 << 4;
    for (int i = 0; i < n && i < 2; ++i) {
        const uint64_t s_addr = pf_src[i];
        const uint64_t d_addr = pf_dst[i];
        if (s_addr == 0ULL || d_addr == 0ULL) continue;
        const uint8_t* sp = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(s_addr));
        uint8_t* dp = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(d_addr));
        if (((s_addr | d_addr) & 15ULL) == 0ULL) {
            const uint4* sv = reinterpret_cast<const uint4*>(sp);
            uint4* dv = reinterpret_cast<uint4*>(dp);
            for (size_t j = tid; j < n16; j += stride) {
                dv[j] = sv[j];
            }
            for (size_t j = tail0 + tid; j < blob_bytes; j += stride) {
                dp[j] = sp[j];
            }
        } else {
            for (size_t j = tid; j < blob_bytes; j += stride) {
                dp[j] = sp[j];
            }
        }
    }
}

extern "C" void launch_glm5_moe_prefetch(
    const uint64_t* pf_src,
    const uint64_t* pf_dst,
    const int32_t* pf_n,
    size_t blob_bytes,
    cudaStream_t stream
) {
    if (blob_bytes == 0 || pf_src == nullptr || pf_dst == nullptr || pf_n == nullptr) return;
    glm5_moe_prefetch_kernel<<<32, 256, 0, stream>>>(pf_src, pf_dst, pf_n, blob_bytes);
}

__global__ void __launch_bounds__(256) glm5_router_sigmoid_probe_batch_kernel(
    const float* __restrict__ logits,       // [M, n_expert]
    const float* __restrict__ bias,         // nullable [n_expert]
    int n_expert,
    int topk,
    float w_scale,
    int norm_w,
    const uint8_t* __restrict__ res_mask,   // nullable [n_expert]
    const float* __restrict__ x_f32,        // nullable [M, n_embd]
    int n_embd,
    int n_ff,
    int layer_idx,
    int32_t gu_type,
    int32_t d_type,
    float clamp_lim,
    int32_t* __restrict__ ids_out,          // [M, topk]
    float* __restrict__ rw_out,             // [M, topk]
    Glm5Doorbell* __restrict__ doorbell,
    uint64_t* __restrict__ seq_state_dev,
    uint64_t* __restrict__ expected_seq_out,
    const int32_t* __restrict__ skip_dev,
    int M
) {
    if (skip_dev != nullptr && *skip_dev != 0) {
        if (threadIdx.x == 0 && expected_seq_out) *expected_seq_out = 0ULL;
        return;
    }
    const int tid = threadIdx.x;
    __shared__ float s_p[512];
    __shared__ float s_sel[512];
    __shared__ uint8_t s_used[512];
    __shared__ float s_best_v[8];
    __shared__ int32_t s_best_i[8];
    __shared__ int32_t s_miss_cnt;
    __shared__ uint64_t s_exp_seq;

    if (tid == 0) {
        s_miss_cnt = 0;
        s_exp_seq = 0ULL;
    }
    __syncthreads();

    for (int t = 0; t < M; ++t) {
        const float* lt = logits + (size_t)t * n_expert;
        int32_t* io = ids_out + (size_t)t * topk;
        float* ro = rw_out + (size_t)t * topk;
        glm5_router_select_token_dev(
            lt, bias, n_expert, topk, w_scale, norm_w,
            io, ro, s_p, s_sel, s_used, s_best_v, s_best_i
        );
        if (tid == 0) {
            int mc = s_miss_cnt;
            for (int k = 0; k < topk; ++k) {
                int eid = io[k];
                if (layer_idx >= 0 && layer_idx < 48 && t * topk + k < 32) {
                    doorbell->routed_ids[layer_idx][t * topk + k] = eid;
                }
                if (res_mask && res_mask[eid] == 0 && mc < 32) {
                    doorbell->miss_tok[mc] = t;
                    doorbell->miss_ids[mc] = eid;
                    doorbell->miss_weights[mc] = ro[k];
                    ++mc;
                }
            }
            s_miss_cnt = mc;
        }
        __syncthreads();
    }

    const int total_mc = s_miss_cnt;
    if (tid == 0) {
        doorbell->m_batch = M;
        doorbell->miss_count = total_mc;
        doorbell->x_q8k_valid = 0;
        if (total_mc > 0) {
            uint64_t next_s = (*seq_state_dev) + 1ULL;
            *seq_state_dev = next_s;
            doorbell->layer_idx = layer_idx;
            doorbell->gu_type = gu_type;
            doorbell->d_type = d_type;
            doorbell->n_embd = n_embd;
            doorbell->n_ff = n_ff;
            doorbell->clamp_lim = clamp_lim;
            s_exp_seq = next_s;
        } else {
            s_exp_seq = 0ULL;
        }
        if (expected_seq_out) *expected_seq_out = s_exp_seq;
    }
    __syncthreads();

    if (total_mc > 0 && x_f32 != nullptr) {
        const int tot_x = M * n_embd;
        for (int e = tid; e < tot_x; e += 256) {
            doorbell->x_in[e] = x_f32[e];
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            doorbell->req_seq = s_exp_seq;
            __threadfence_system();
        }
    }
}

extern "C" void launch_glm5_router_sigmoid_probe(
    const float* logits,
    const float* bias,
    int n_expert,
    int topk,
    float w_scale,
    int norm_w,
    const uint8_t* res_mask,
    const float* x_f32,
    int n_embd,
    int n_ff,
    int layer_idx,
    int32_t gu_type,
    int32_t d_type,
    float clamp_lim,
    int32_t* ids_out,
    float* rw_out,
    void* doorbell_mapped,
    uint64_t* seq_state_dev,
    uint64_t* expected_seq_out,
    const int32_t* skip_dev,
    int M,
    cudaStream_t stream
) {
    if (M <= 0 || n_expert <= 0) return;
    if (doorbell_mapped != nullptr && M > 1) {
        glm5_router_sigmoid_probe_batch_kernel<<<1, 256, 0, stream>>>(
            logits, bias, n_expert, topk, w_scale, norm_w, res_mask,
            x_f32, n_embd, n_ff, layer_idx, gu_type, d_type, clamp_lim,
            ids_out, rw_out, reinterpret_cast<Glm5Doorbell*>(doorbell_mapped),
            seq_state_dev, expected_seq_out, skip_dev, M
        );
        return;
    }
    glm5_router_sigmoid_probe_kernel<<<M, 256, 0, stream>>>(
        logits, bias, n_expert, topk, w_scale, norm_w, res_mask,
        x_f32, n_embd, n_ff, layer_idx, gu_type, d_type, clamp_lim,
        ids_out, rw_out, reinterpret_cast<Glm5Doorbell*>(doorbell_mapped),
        seq_state_dev, expected_seq_out, skip_dev
    );
}

// Single-block (256 threads) wait + accumulate + HC post for M=1..4 (zero cross-block race)
__global__ void __launch_bounds__(256) glm5_doorbell_wait_add_hc_post_kernel(
    const Glm5Doorbell* __restrict__ doorbell,
    const uint64_t* __restrict__ expected_seq_ptr,
    const float* __restrict__ R_in,   // nullable (null -> plain add R_out[t] += ffn[t])
    const float* __restrict__ post,   // nullable [M, hc]
    const float* __restrict__ comb,   // nullable [M, hc * hc]
    const float* __restrict__ sh_out, // [M, n_embd]
    const float* __restrict__ gpu_moe,// nullable [M, n_embd]
    float* __restrict__ R_out,        // [M, hc, n_embd] or [M, n_embd]
    int n_embd,
    int hc,
    int M
) {
    __shared__ uint64_t s_exp;
    __shared__ float s_post[8];
    __shared__ float s_comb[64];

    if (threadIdx.x == 0) {
        uint64_t exp = expected_seq_ptr ? *expected_seq_ptr : 0ULL;
        s_exp = exp;
        if (exp > 0ULL && doorbell != nullptr) {
            const uint64_t* done_addr = reinterpret_cast<const uint64_t*>(const_cast<const uint64_t*>(&doorbell->done_seq));
            for (;;) {
                uint64_t cur = 0ULL;
                asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(cur) : "l"(done_addr) : "memory");
                if (cur >= exp) break;
                __nanosleep(50);
            }
        }
    }
    __syncthreads();

    const int use_cpu = (s_exp > 0ULL && doorbell != nullptr);
    for (int t = 0; t < M; ++t) {
        if (R_in != nullptr) {
            if (threadIdx.x < hc) s_post[threadIdx.x] = post[(size_t)t * hc + threadIdx.x];
            if (threadIdx.x < hc * hc) s_comb[threadIdx.x] = comb[(size_t)t * hc * hc + threadIdx.x];
            __syncthreads();
        }
        const float* sh_t = sh_out + (size_t)t * n_embd;
        const float* gm_t = gpu_moe ? (gpu_moe + (size_t)t * n_embd) : nullptr;
        const float* db_t = use_cpu ? (doorbell->out + (size_t)t * n_embd) : nullptr;
        const float* ri_t = R_in ? (R_in + (size_t)t * hc * n_embd) : nullptr;
        float* ro_t = R_out + (size_t)t * (R_in ? hc : 1) * n_embd;

        for (int e = threadIdx.x; e < n_embd; e += 256) {
            float ffn = sh_t[e];
            if (gm_t) ffn += gm_t[e];
            if (db_t) ffn += db_t[e];
            if (ri_t == nullptr) {
                ro_t[e] += ffn;
            } else if (hc == 4) {
                float r0 = ri_t[0 * n_embd + e];
                float r1 = ri_t[1 * n_embd + e];
                float r2 = ri_t[2 * n_embd + e];
                float r3 = ri_t[3 * n_embd + e];
                for (int d = 0; d < 4; ++d) {
                    ro_t[d * n_embd + e] = ffn * s_post[d]
                        + s_comb[d * 4 + 0] * r0 + s_comb[d * 4 + 1] * r1
                        + s_comb[d * 4 + 2] * r2 + s_comb[d * 4 + 3] * r3;
                }
            } else {
                for (int d = 0; d < hc; ++d) {
                    float acc = ffn * s_post[d];
                    for (int s = 0; s < hc; ++s) acc += s_comb[d * hc + s] * ri_t[s * n_embd + e];
                    ro_t[d * n_embd + e] = acc;
                }
            }
        }
        if (t + 1 < M) __syncthreads();
    }
}

extern "C" void launch_glm5_doorbell_wait_hc_post(
    const void* doorbell_mapped,
    const uint64_t* expected_seq_ptr,
    const float* R_in,
    const float* post,
    const float* comb,
    const float* sh_out,
    const float* gpu_moe,
    float* R_out,
    int n_embd,
    int hc,
    int M,
    cudaStream_t stream
) {
    if (M <= 0) return;
    glm5_doorbell_wait_add_hc_post_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const Glm5Doorbell*>(doorbell_mapped),
        expected_seq_ptr, R_in, post, comb, sh_out, gpu_moe, R_out, n_embd, hc, M
    );
}

__global__ void __launch_bounds__(256) glm5_doorbell_wait_add_kernel(
    const Glm5Doorbell* __restrict__ doorbell,
    const uint64_t* __restrict__ expected_seq_ptr,
    float* __restrict__ inout_ffn,
    int n_embd
) {
    __shared__ uint64_t s_exp;
    const int tid = threadIdx.x;
    if (tid == 0) {
        uint64_t exp = expected_seq_ptr ? *expected_seq_ptr : 0ULL;
        s_exp = exp;
        if (exp > 0ULL && doorbell != nullptr) {
            const uint64_t* done_addr = reinterpret_cast<const uint64_t*>(const_cast<const uint64_t*>(&doorbell->done_seq));
            for (;;) {
                uint64_t cur = 0ULL;
                asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(cur) : "l"(done_addr) : "memory");
                if (cur >= exp) break;
                __nanosleep(50);
            }
        }
    }
    __syncthreads();
    if (s_exp == 0ULL || doorbell == nullptr) return;
    const float* db_out = doorbell->out;
    const int n4 = n_embd >> 2;
    if ((((uintptr_t)inout_ffn | (uintptr_t)db_out) & 15u) == 0u) {
        float4* dst4 = reinterpret_cast<float4*>(inout_ffn);
        const float4* src4 = reinterpret_cast<const float4*>(db_out);
        for (int i = tid; i < n4; i += 256) {
            float4 a = dst4[i];
            float4 b = src4[i];
            dst4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }
        for (int e = (n4 << 2) + tid; e < n_embd; e += 256) {
            inout_ffn[e] += db_out[e];
        }
    } else {
        for (int e = tid; e < n_embd; e += 256) {
            inout_ffn[e] += db_out[e];
        }
    }
}

extern "C" void launch_glm5_doorbell_wait_add(
    const void* doorbell_mapped,
    const uint64_t* expected_seq_ptr,
    float* inout_ffn,
    int n_embd,
    cudaStream_t stream
) {
    if (n_embd <= 0) return;
    glm5_doorbell_wait_add_kernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<const Glm5Doorbell*>(doorbell_mapped),
        expected_seq_ptr, inout_ffn, n_embd
    );
}

// -----------------------------------------------------------------------------
// MTP Input Quantize & GPU Argmax
// -----------------------------------------------------------------------------

__global__ void __launch_bounds__(256) glm5_mtp_in_q8_kernel(
    const float* __restrict__ emb,    // [n_embd]
    const float* __restrict__ h,      // [n_embd]
    const float* __restrict__ enorm,  // [n_embd]
    const float* __restrict__ hnorm,  // [n_embd]
    float eps,
    int n_embd,
    block_q8_1* __restrict__ cat_q8   // [(2 * n_embd) / 32]
) {
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    __shared__ float s_re[8], s_rh[8], s_ie, s_ih;

    float sse = 0.0f, ssh = 0.0f;
    for (int e = tid; e < n_embd; e += 256) {
        float ve = emb[e], vh = h[e];
        sse += ve * ve;
        ssh += vh * vh;
    }
    sse = warp_reduce_sum(sse);
    ssh = warp_reduce_sum(ssh);
    if (lane == 0) { s_re[warp] = sse; s_rh[warp] = ssh; }
    __syncthreads();
    if (tid == 0) {
        float te = 0.0f, th = 0.0f;
        for (int w = 0; w < 8; ++w) { te += s_re[w]; th += s_rh[w]; }
        s_ie = rsqrtf(te / (float)n_embd + eps);
        s_ih = rsqrtf(th / (float)n_embd + eps);
    }
    __syncthreads();

    float ie = s_ie, ih = s_ih;
    int nb = n_embd >> 5;
    for (int base = 0; base < n_embd; base += 256) {
        int e = base + tid;
        float ve = (e < n_embd) ? (emb[e] * ie * enorm[e]) : 0.0f;
        float vh = (e < n_embd) ? (h[e]   * ih * hnorm[e]) : 0.0f;
        if (base + warp * 32 < n_embd) {
            warp_quantize_q8_1_store(ve, cat_q8 + ((base >> 5) + warp), lane);
            warp_quantize_q8_1_store(vh, cat_q8 + nb + ((base >> 5) + warp), lane);
        }
    }
}

extern "C" void launch_glm5_mtp_in_q8(
    const float* emb,
    const float* h,
    const float* enorm,
    const float* hnorm,
    float eps,
    int n_embd,
    void* cat_q8,
    cudaStream_t stream
) {
    if (n_embd <= 0) return;
    glm5_mtp_in_q8_kernel<<<1, 256, 0, stream>>>(
        emb, h, enorm, hnorm, eps, n_embd, reinterpret_cast<block_q8_1*>(cat_q8)
    );
}

__global__ void __launch_bounds__(256) glm5_argmax_stage1_kernel(
    const float* __restrict__ logits,
    int n_rows,
    float* __restrict__ part_vals,
    int32_t* __restrict__ part_ids
) {
    const int tid = threadIdx.x;
    const int stride = gridDim.x * 256;
    float best_v = -INFINITY;
    int best_i = 0;
    for (int i = blockIdx.x * 256 + tid; i < n_rows; i += stride) {
        float v = logits[i];
        if (v > best_v) { best_v = v; best_i = i; }
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, best_v, o);
        int oi = __shfl_xor_sync(0xFFFFFFFFu, best_i, o);
        if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if (lane == 0) { sv[warp] = best_v; si[warp] = best_i; }
    __syncthreads();
    if (tid == 0) {
        float bv = -INFINITY; int bi = 0;
        for (int w = 0; w < 8; ++w) {
            if (sv[w] > bv || (sv[w] == bv && si[w] < bi)) { bv = sv[w]; bi = si[w]; }
        }
        part_vals[blockIdx.x] = bv;
        part_ids[blockIdx.x] = bi;
    }
}

__global__ void __launch_bounds__(256) glm5_argmax_stage2_kernel(
    const float* __restrict__ part_vals,
    const int32_t* __restrict__ part_ids,
    int n_parts,
    int n_lead,
    const int32_t* __restrict__ extra_ids,
    int32_t* __restrict__ out_tok
) {
    const int tid = threadIdx.x;
    float bv = -INFINITY; int bi = 0;
    for (int i = tid; i < n_parts; i += 256) {
        float v = part_vals[i];
        int id = part_ids[i];
        if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; }
    }
    const int warp = tid >> 5;
    const int lane = tid & 31;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
        int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if (lane == 0) { sv[warp] = bv; si[warp] = bi; }
    __syncthreads();
    if (tid == 0) {
        float fv = -INFINITY; int fi = 0;
        for (int w = 0; w < 8; ++w) {
            if (sv[w] > fv || (sv[w] == fv && si[w] < fi)) { fv = sv[w]; fi = si[w]; }
        }
        if (extra_ids != nullptr && fi >= n_lead) {
            fi = extra_ids[fi - n_lead];
        }
        *out_tok = fi;
        __threadfence_system();
    }
}

extern "C" void launch_glm5_argmax(
    const float* logits,
    int n_rows,
    int n_lead,
    const int32_t* extra_ids,
    float* part_vals_scratch,
    int32_t* part_ids_scratch,
    int32_t* out_tok,
    cudaStream_t stream
) {
    if (n_rows <= 0) return;
    int blocks = min(128, (n_rows + 255) / 256);
    glm5_argmax_stage1_kernel<<<blocks, 256, 0, stream>>>(
        logits, n_rows, part_vals_scratch, part_ids_scratch
    );
    glm5_argmax_stage2_kernel<<<1, 256, 0, stream>>>(
        part_vals_scratch, part_ids_scratch, blocks, n_lead, extra_ids, out_tok
    );
}

__global__ void __launch_bounds__(256) glm5_copy_to_mapped_f32_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst_mapped,
    int n
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    if ((n & 3) == 0 && (((uintptr_t)src | (uintptr_t)dst_mapped) & 15u) == 0u) {
        const float4* __restrict__ src4 = reinterpret_cast<const float4*>(src);
        float4* __restrict__ dst4 = reinterpret_cast<float4*>(dst_mapped);
        const int n4 = n >> 2;
        for (int i = tid; i < n4; i += stride) {
            dst4[i] = src4[i];
        }
    } else {
        for (int i = tid; i < n; i += stride) {
            dst_mapped[i] = src[i];
        }
    }
    __threadfence_system();
}

extern "C" void launch_glm5_copy_to_mapped_f32(
    const float* src,
    float* dst_mapped,
    int n,
    cudaStream_t stream
) {
    if (n <= 0) return;
    int blocks = ((n & 3) == 0 && (((uintptr_t)src | (uintptr_t)dst_mapped) & 15u) == 0u)
        ? min(32, ((n >> 2) + 255) / 256)
        : min(32, (n + 255) / 256);
    if (blocks < 1) blocks = 1;
    glm5_copy_to_mapped_f32_kernel<<<blocks, 256, 0, stream>>>(src, dst_mapped, n);
}

// -----------------------------------------------------------------------------
// Stage B: Batched Small-Token (nt <= 8) Quantized Prefill MoE Kernels
// Avoids dequant_to_f16 + MMA GEMM launch/bandwidth overhead for sparse experts
// by reading each weight row once and contracting against up to 8 Q8_1 tokens.
// -----------------------------------------------------------------------------

struct Glm5SubPackedW {
    int w[8];
    float s0;
    float s1;
};

template <int W_TYPE>
__device__ __forceinline__ void unpack_sub_w_blk32(
    const uint8_t* w_row,
    int blk32,
    Glm5SubPackedW& p
) {
    if constexpr (W_TYPE == GGML_TYPE_Q8_0) {
        const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(w_row) + blk32;
        p.s0 = fp16_bits_to_f32(load_u16_align2(&blk->d));
        p.s1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            p.w[j] = load_i32_align2(blk->qs + 4 * j);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q5_1) {
        const block_q5_1* blk = reinterpret_cast<const block_q5_1*>(w_row) + blk32;
        p.s0 = fp16_bits_to_f32(load_u16_align2(&blk->d));
        p.s1 = -fp16_bits_to_f32(load_u16_align2(&blk->m));
        const uint32_t qh = (uint32_t)load_i32_align2(blk->qh);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4 = (uint32_t)load_i32_align2(blk->qs + 4 * j);
            p.w[j + 0] = (int)((q4 & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (4 * j)));
            p.w[j + 4] = (int)(((q4 >> 4) & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (16 + 4 * j)));
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q5_0) {
        const block_q5_0* blk = reinterpret_cast<const block_q5_0*>(w_row) + blk32;
        p.s0 = fp16_bits_to_f32(load_u16_align2(&blk->d));
        p.s1 = 0.0f;
        const uint32_t qh = (uint32_t)load_i32_align2(blk->qh);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4 = (uint32_t)load_i32_align2(blk->qs + 4 * j);
            const uint32_t lo_u = (q4 & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (4 * j));
            const uint32_t hi_u = ((q4 >> 4) & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (16 + 4 * j));
            p.w[j + 0] = (int)__vsub4(lo_u, 0x10101010u);
            p.w[j + 4] = (int)__vsub4(hi_u, 0x10101010u);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ4_NL) {
        const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(w_row) + blk32;
        p.s0 = fp16_bits_to_f32(load_u16_align2(&blk->d));
        p.s1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4 = (uint32_t)load_i32_align2(blk->qs + 4 * j);
            p.w[j + 0] = lut_iq4nl_x4(q4 & 0x0F0F0F0Fu);
            p.w[j + 4] = lut_iq4nl_x4((q4 >> 4) & 0x0F0F0F0Fu);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q2_0) {
        const block_q2_0* blk = reinterpret_cast<const block_q2_0*>(w_row) + (blk32 >> 1);
        p.s0 = fp16_bits_to_f32(load_u16_align2(&blk->d));
        p.s1 = 0.0f;
        const int half = blk32 & 1;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            p.w[j] = unpack_q2_0_x4(blk->qs[half * 8 + j]);
        }
    }
}

template <int W_TYPE>
__device__ __forceinline__ void unpack_sub_w_row(
    const uint8_t* w_row,
    int b,
    int sub,
    const uint32_t* s_lut,
    Glm5SubPackedW& p
) {
    if constexpr (W_TYPE == GGML_TYPE_Q6_K) {
        const block_q6_K* blk = reinterpret_cast<const block_q6_K*>(w_row) + b;
        float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        int half = sub >> 2;
        int q = sub & 3;
        const uint8_t* ql = blk->ql + 64 * half + 32 * (q & 1);
        const uint8_t* qh = blk->qh + 32 * half;
        int ql_shift = 4 * (q >> 1);
        int qh_shift = 2 * q;
        p.s0 = dw * (float)blk->scales[2 * sub + 0];
        p.s1 = dw * (float)blk->scales[2 * sub + 1];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t lo4 = ((uint32_t)load_i32_align2(ql + 4 * j) >> ql_shift) & 0x0F0F0F0Fu;
            uint32_t hi2 = ((uint32_t)load_i32_align2(qh + 4 * j) >> qh_shift) & 0x03030303u;
            p.w[j] = (int)__vsub4(lo4 | (hi2 << 4), 0x20202020u);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q4_K) {
        const block_q4_K* blk = reinterpret_cast<const block_q4_K*>(w_row) + b;
        float2 dpair = fp16x2_bits_to_f32(&blk->d);
        uint8_t sc = 0, mn = 0;
        get_scale_min_k4_dev(sub, blk->scales, sc, mn);
        p.s0 = dpair.x * (float)sc;
        p.s1 = dpair.y * (float)mn;
        const uint8_t* qs = blk->qs + 32 * (sub >> 1);
        int shift = 4 * (sub & 1);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            p.w[j] = (int)(((uint32_t)load_i32_align4(qs + 4 * j) >> shift) & 0x0F0F0F0Fu);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q5_K) {
        const block_q5_K* blk = reinterpret_cast<const block_q5_K*>(w_row) + b;
        float2 dpair = fp16x2_bits_to_f32(&blk->d);
        uint8_t sc = 0, mn = 0;
        get_scale_min_k4_dev(sub, blk->scales, sc, mn);
        p.s0 = dpair.x * (float)sc;
        p.s1 = dpair.y * (float)mn;
        const uint8_t* qs = blk->qs + 32 * (sub >> 1);
        int shift = 4 * (sub & 1);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t q4 = ((uint32_t)load_i32_align4(qs + 4 * j) >> shift) & 0x0F0F0F0Fu;
            uint32_t raw_qh = (uint32_t)load_i32_align4(blk->qh + 4 * j);
            p.w[j] = (int)(q4 | (((raw_qh >> sub) & 0x01010101u) << 4));
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ4_XS) {
        const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(w_row) + b;
        float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        uint16_t sh = load_u16_align2(&blk->scales_h);
        int ls = ((blk->scales_l[sub >> 1] >> (4 * (sub & 1))) & 0xF) | (((sh >> (2 * sub)) & 3) << 4);
        p.s0 = dw * (float)(ls - 32);
        p.s1 = 0.0f;
        const uint8_t* qs = blk->qs + sub * 16;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t q4 = (uint32_t)load_i32_align2(qs + 4 * j);
            p.w[j + 0] = lut_iq4nl_x4(q4 & 0x0F0F0F0Fu);
            p.w[j + 4] = lut_iq4nl_x4((q4 >> 4) & 0x0F0F0F0Fu);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q8_0 || W_TYPE == GGML_TYPE_Q5_1 || W_TYPE == GGML_TYPE_Q2_0) {
        unpack_sub_w_blk32<W_TYPE>(w_row, b * 8 + sub, p);
    } else if constexpr (W_TYPE == GGML_TYPE_Q3_K) {
        const block_q3_K* blk = reinterpret_cast<const block_q3_K*>(w_row) + b;
        float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        int a_idx = (sub >> 1) & 1;
        uint32_t low4 = ((uint32_t)load_u16_align2(blk->scales + 4 * a_idx + 2 * (sub & 1)) >> (sub & 4)) & 0x0F0Fu;
        uint32_t hi2 = ((uint32_t)load_u16_align2(blk->scales + 8 + 2 * (sub & 1)) >> (2 * (sub >> 1))) & 0x0303u;
        uint32_t sc_pair = low4 | (hi2 << 4);
        int sc0 = (int)(sc_pair & 0xFFu) - 32;
        int sc1 = (int)((sc_pair >> 8) & 0xFFu) - 32;
        p.s0 = dw * (float)sc0;
        p.s1 = dw * (float)sc1;
        int q_shift = 2 * (sub & 3);
        const uint8_t* qp = blk->qs + 32 * (sub >> 2);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint32_t q4 = ((uint32_t)load_i32_align2(qp + 4 * j) >> q_shift) & 0x03030303u;
            uint32_t hm = ((uint32_t)load_i32_align2(blk->hmask + 4 * j) >> sub) & 0x01010101u;
            unsigned sub4 = __vsub4(0x04040404u, hm << 2);
            p.w[j] = (int)__vsub4(q4, sub4);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
        const block_iq2_s* blk = reinterpret_cast<const block_iq2_s*>(w_row) + b;
        float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        uint32_t qs_packed = (uint32_t)load_i32_align2(blk->qs + 4 * sub);
        uint32_t signs_packed = (uint32_t)load_i32_align2(blk->qs + 32 + 4 * sub);
        uint32_t qh = (uint32_t)blk->qh[sub];
        uint8_t sc_byte = blk->scales[sub];
        int ls0 = (int)(sc_byte & 0x0Fu);
        int ls1 = (int)(sc_byte >> 4);
        p.s0 = dw * 0.125f * (float)(2 * ls0 + 1);
        p.s1 = dw * 0.125f * (float)(2 * ls1 + 1);
        const uint64_t* lut = reinterpret_cast<const uint64_t*>(s_lut);
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            int sh = (l0 >> 1) * 8;
            uint32_t q_b = (qs_packed >> sh) & 0xFFu;
            uint32_t sbyte = (signs_packed >> sh) & 0xFFu;
            uint32_t grid_idx = q_b | ((qh << (8 - l0)) & 0x300u);
            uint64_t g64 = lut[grid_idx];
            uint32_t g_lo = (uint32_t)g64;
            uint32_t g_hi = (uint32_t)(g64 >> 32);
            unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
            unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
            p.w[l0]     = (int)__vsub4(g_lo ^ s0, s0);
            p.w[l0 + 1] = (int)__vsub4(g_hi ^ s1, s1);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
        const block_iq3_xxs* blk = reinterpret_cast<const block_iq3_xxs*>(w_row) + b;
        float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        uint32_t q3_words[2];
        q3_words[0] = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
        q3_words[1] = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(q3_words);
        uint32_t aux32 = (uint32_t)load_i32_align2(blk->qs + 64 + 4 * sub);
        int ls = (int)(aux32 >> 28);
        p.s0 = dw * 0.25f * (float)(2 * ls + 1);
        p.s1 = 0.0f;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            uint32_t g0 = s_lut[qs[l0 + 0]];
            uint32_t g1 = s_lut[qs[l0 + 1]];
            uint32_t s7 = (aux32 >> (7 * (l0 >> 1))) & 0x7Fu;
            uint32_t ks = (s7 | ((uint32_t)(__popc(s7) & 1) << 7)) * 0x01010101u;
            unsigned s0 = __vcmpne4(ks & 0x08040201u, 0u);
            unsigned s1 = __vcmpne4(ks & 0x80402010u, 0u);
            p.w[l0]     = (int)__vsub4(g0 ^ s0, s0);
            p.w[l0 + 1] = (int)__vsub4(g1 ^ s1, s1);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_S) {
        const block_iq3_s* blk = reinterpret_cast<const block_iq3_s*>(w_row) + b;
        float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        uint32_t q3_words[2];
        q3_words[0] = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
        q3_words[1] = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
        const uint8_t* qs = reinterpret_cast<const uint8_t*>(q3_words);
        uint32_t qh = (uint32_t)blk->qh[sub];
        uint32_t signs32 = (uint32_t)load_i32_align2(blk->signs + 4 * sub);
        const uint8_t* sb = reinterpret_cast<const uint8_t*>(&signs32);
        int ls = 1 + 2 * (int)((blk->scales[sub >> 1] >> (4 * (sub & 1))) & 0x0Fu);
        p.s0 = dw * (float)ls;
        p.s1 = 0.0f;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            uint32_t g0 = s_lut[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100u)];
            uint32_t g1 = s_lut[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100u)];
            uint32_t sbyte = (uint32_t)sb[l0 >> 1];
            unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
            unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
            p.w[l0]     = (int)__vsub4(g0 ^ s0, s0);
            p.w[l0 + 1] = (int)__vsub4(g1 ^ s1, s1);
        }
    }
}

template <int W_TYPE>
__device__ __forceinline__ float dot_unpacked_w_q8(
    const Glm5SubPackedW& p,
    const int (&u)[8],
    float2 ds8
) {
    if constexpr (W_TYPE == GGML_TYPE_Q4_K || W_TYPE == GGML_TYPE_Q5_K || W_TYPE == GGML_TYPE_Q5_1) {
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) sumi = dp4a_packed(p.w[j], u[j], sumi);
        return p.s0 * ds8.x * (float)sumi - p.s1 * ds8.y;
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS || W_TYPE == GGML_TYPE_IQ3_S || W_TYPE == GGML_TYPE_IQ4_XS || W_TYPE == GGML_TYPE_Q8_0 || W_TYPE == GGML_TYPE_Q2_0 || W_TYPE == GGML_TYPE_Q5_0 || W_TYPE == GGML_TYPE_IQ4_NL) {
        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) sumi = dp4a_packed(p.w[j], u[j], sumi);
        return ds8.x * p.s0 * (float)sumi;
    } else {
        int sum0 = 0, sum1 = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) sum0 = dp4a_packed(p.w[j], u[j], sum0);
        #pragma unroll
        for (int j = 4; j < 8; ++j) sum1 = dp4a_packed(p.w[j], u[j], sum1);
        return ds8.x * (p.s0 * (float)sum0 + p.s1 * (float)sum1);
    }
}

template <int W_TYPE>
__device__ __forceinline__ void stage_small_moe_lut(uint32_t* s_lut, int tid) {
    if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
        const uint32_t* src = reinterpret_cast<const uint32_t*>(d_iq2s_grid);
        for (int i = tid; i < 2048; i += 256) s_lut[i] = src[i];
        __syncthreads();
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
        for (int i = tid; i < 256; i += 256) s_lut[i] = d_iq3xxs_grid[i];
        __syncthreads();
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_S) {
        for (int i = tid; i < 512; i += 256) s_lut[i] = d_iq3s_grid[i];
        __syncthreads();
    }
}

template <int W_TYPE>
__global__ void __launch_bounds__(256, 2) glm5_moe_prefill_small_gu_kernel(
    const uint64_t* __restrict__ exp_ptrs,
    const int32_t* __restrict__ slot_off,
    const int32_t* __restrict__ slot_nt,
    const int32_t* __restrict__ rows_all,
    uint32_t u_off,
    size_t row_bytes,
    const block_q8_1* __restrict__ x_q8,
    block_q8_1* __restrict__ inter_q8,
    int n_embd,
    int n_ff,
    float clamp_lim
) {
    const int r_base = (int)blockIdx.x * 32;
    const int job = (int)blockIdx.y;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;

    const uint64_t base = exp_ptrs[job];
    const int off = slot_off[job];
    const int nt = slot_nt[job];
    if (base == 0ULL || nt <= 0 || r_base >= n_ff) return;

    constexpr int LUT_WORDS = (W_TYPE == GGML_TYPE_IQ2_S) ? 2048 : ((W_TYPE == GGML_TYPE_IQ3_S) ? 512 : ((W_TYPE == GGML_TYPE_IQ3_XXS) ? 256 : 4));
    constexpr bool STAGE_X_OK = (W_TYPE != GGML_TYPE_IQ2_S && W_TYPE != GGML_TYPE_IQ3_S && W_TYPE != GGML_TYPE_IQ3_XXS);
    constexpr int MAX_X_BLKS = STAGE_X_OK ? 80 : 1;
    __shared__ __align__(16) uint32_t s_lut[LUT_WORDS];
    __shared__ float s_h[8][32];
    __shared__ __align__(16) uint32_t s_x[STAGE_X_OK ? 8 : 1][MAX_X_BLKS * 9];
    stage_small_moe_lut<W_TYPE>(s_lut, tid);

    const uint8_t* g_base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(base));
    const uint8_t* u_base = g_base + u_off;
    const int nb_x = n_embd >> 5;
    const int nb_ff = n_ff >> 5;
    const int nb = n_embd >> 8;
    const int sub = lane & 7;
    const int b0 = lane >> 3;
    const bool use_sx = STAGE_X_OK && (nb_x <= MAX_X_BLKS);

    for (int t_base = 0; t_base < nt; t_base += 8) {
        const int t_cnt = min(8, nt - t_base);
        int t_rows[8];
        for (int i = 0; i < t_cnt; ++i) t_rows[i] = rows_all[off + t_base + i];
        if (use_sx) {
            const int wpt = nb_x * 9;
            const int tot_w = t_cnt * wpt;
            uint32_t* dst_sx = &s_x[0][0];
            for (int i = tid; i < tot_w; i += 256) {
                const int ti = i / wpt;
                const int wi = i - ti * wpt;
                dst_sx[ti * (MAX_X_BLKS * 9) + wi] =
                    reinterpret_cast<const uint32_t*>(x_q8 + (size_t)t_rows[ti] * nb_x)[wi];
            }
            __syncthreads();
        }

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int r = r_base + warp * 4 + step;
            if (r < n_ff) {
                float ag[8] = {0}, au[8] = {0};
                const uint8_t* gr = g_base + (size_t)r * row_bytes;
                const uint8_t* ur = u_base + (size_t)r * row_bytes;
                if constexpr (W_TYPE == GGML_TYPE_Q8_0 || W_TYPE == GGML_TYPE_Q5_1 || W_TYPE == GGML_TYPE_Q2_0 || W_TYPE == GGML_TYPE_Q5_0 || W_TYPE == GGML_TYPE_IQ4_NL) {
                    #pragma unroll 1
                    for (int blk32 = lane; blk32 < nb_x; blk32 += 32) {
                        Glm5SubPackedW wg, wu;
                        unpack_sub_w_blk32<W_TYPE>(gr, blk32, wg);
                        unpack_sub_w_blk32<W_TYPE>(ur, blk32, wu);
                        for (int ti = 0; ti < t_cnt; ++ti) {
                            int u[8];
                            float2 ds8;
                            const block_q8_1* x_ptr = use_sx
                                ? (reinterpret_cast<const block_q8_1*>(s_x[ti]) + blk32)
                                : (x_q8 + (size_t)t_rows[ti] * nb_x + blk32);
                            load_q8_1_sub_words(x_ptr, u, ds8);
                            ag[ti] += dot_unpacked_w_q8<W_TYPE>(wg, u, ds8);
                            au[ti] += dot_unpacked_w_q8<W_TYPE>(wu, u, ds8);
                        }
                    }
                } else {
                    #pragma unroll 1
                    for (int b = b0; b < nb; b += 4) {
                        Glm5SubPackedW wg, wu;
                        unpack_sub_w_row<W_TYPE>(gr, b, sub, s_lut, wg);
                        unpack_sub_w_row<W_TYPE>(ur, b, sub, s_lut, wu);
                        for (int ti = 0; ti < t_cnt; ++ti) {
                            int u[8];
                            float2 ds8;
                            const block_q8_1* x_ptr = use_sx
                                ? (reinterpret_cast<const block_q8_1*>(s_x[ti]) + b * 8 + sub)
                                : (x_q8 + (size_t)t_rows[ti] * nb_x + b * 8 + sub);
                            load_q8_1_sub_words(x_ptr, u, ds8);
                            ag[ti] += dot_unpacked_w_q8<W_TYPE>(wg, u, ds8);
                            au[ti] += dot_unpacked_w_q8<W_TYPE>(wu, u, ds8);
                        }
                    }
                }
                for (int ti = 0; ti < t_cnt; ++ti) {
                    float vg = warp_reduce_sum(ag[ti]);
                    float vu = warp_reduce_sum(au[ti]);
                    if (lane == 0) {
                        s_h[ti][warp * 4 + step] = glm_swiglu_clamp_val(vg, vu, clamp_lim);
                    }
                }
            } else if (lane == 0) {
                for (int ti = 0; ti < t_cnt; ++ti) s_h[ti][warp * 4 + step] = 0.0f;
            }
        }
        __syncthreads();

        if (warp < t_cnt) {
            block_q8_1* dst_blk = inter_q8 + ((size_t)(off + t_base + warp) * nb_ff + blockIdx.x);
            warp_quantize_q8_1_store(s_h[warp][lane], dst_blk, lane);
        }
        if (t_base + 8 < nt) __syncthreads();
    }
}

template <int W_TYPE, bool DET_PAIR = false>
__global__ void __launch_bounds__(256, 2) glm5_moe_prefill_small_down_kernel(
    const uint64_t* __restrict__ exp_ptrs,
    const int32_t* __restrict__ slot_off,
    const int32_t* __restrict__ slot_nt,
    const int32_t* __restrict__ rows_all,
    const float* __restrict__ wts_all,
    uint32_t d_off,
    size_t row_bytes,
    const block_q8_1* __restrict__ inter_q8,
    float* __restrict__ out_f32,
    int n_embd,
    int n_ff
) {
    const int r_base = (int)blockIdx.x * 32;
    const int job = (int)blockIdx.y;
    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid = warp * 32 + lane;

    const uint64_t base = exp_ptrs[job];
    const int off = slot_off[job];
    const int nt = slot_nt[job];
    if (base == 0ULL || nt <= 0 || r_base >= n_embd) return;

    constexpr int LUT_WORDS = (W_TYPE == GGML_TYPE_IQ2_S) ? 2048 : ((W_TYPE == GGML_TYPE_IQ3_S) ? 512 : ((W_TYPE == GGML_TYPE_IQ3_XXS) ? 256 : 4));
    constexpr bool STAGE_I_OK = (W_TYPE != GGML_TYPE_IQ2_S && W_TYPE != GGML_TYPE_IQ3_S && W_TYPE != GGML_TYPE_IQ3_XXS);
    constexpr int MAX_I_BLKS = STAGE_I_OK ? 32 : 1;
    __shared__ __align__(16) uint32_t s_lut[LUT_WORDS];
    __shared__ __align__(16) uint32_t s_inter[STAGE_I_OK ? 8 : 1][MAX_I_BLKS * 9];
    stage_small_moe_lut<W_TYPE>(s_lut, tid);

    const uint8_t* d_base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(base + d_off));
    const int nb_ff = n_ff >> 5;
    const int nb = n_ff >> 8;
    const int sub = lane & 7;
    const int b0 = lane >> 3;
    const bool use_si = STAGE_I_OK && (nb_ff <= MAX_I_BLKS);

    for (int t_base = 0; t_base < nt; t_base += 8) {
        const int t_cnt = min(8, nt - t_base);
        int t_rows[8];
        float t_rw[8];
        for (int i = 0; i < t_cnt; ++i) {
            t_rows[i] = rows_all[off + t_base + i];
            t_rw[i] = wts_all[off + t_base + i];
        }
        if (use_si) {
            const int wpt = nb_ff * 9;
            const int tot_w = t_cnt * wpt;
            const uint32_t* gi = reinterpret_cast<const uint32_t*>(inter_q8 + (size_t)(off + t_base) * nb_ff);
            uint32_t* dst_si = &s_inter[0][0];
            for (int i = tid; i < tot_w; i += 256) {
                const int ti = i / wpt;
                const int wi = i - ti * wpt;
                dst_si[ti * (MAX_I_BLKS * 9) + wi] = gi[i];
            }
            __syncthreads();
        }

        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            const int r = r_base + warp * 4 + step;
            if (r >= n_embd) break;
            const uint8_t* dr = d_base + (size_t)r * row_bytes;
            float ad[8] = {0};
            if constexpr (W_TYPE == GGML_TYPE_Q8_0 || W_TYPE == GGML_TYPE_Q5_1 || W_TYPE == GGML_TYPE_Q2_0 || W_TYPE == GGML_TYPE_Q5_0 || W_TYPE == GGML_TYPE_IQ4_NL) {
                #pragma unroll 1
                for (int blk32 = lane; blk32 < nb_ff; blk32 += 32) {
                    Glm5SubPackedW wd;
                    unpack_sub_w_blk32<W_TYPE>(dr, blk32, wd);
                    for (int ti = 0; ti < t_cnt; ++ti) {
                        int u[8];
                        float2 ds8;
                        const block_q8_1* i_ptr = use_si
                            ? (reinterpret_cast<const block_q8_1*>(s_inter[ti]) + blk32)
                            : (inter_q8 + (size_t)(off + t_base + ti) * nb_ff + blk32);
                        load_q8_1_sub_words(i_ptr, u, ds8);
                        ad[ti] += dot_unpacked_w_q8<W_TYPE>(wd, u, ds8);
                    }
                }
            } else {
                #pragma unroll 1
                for (int b = b0; b < nb; b += 4) {
                    Glm5SubPackedW wd;
                    unpack_sub_w_row<W_TYPE>(dr, b, sub, s_lut, wd);
                    for (int ti = 0; ti < t_cnt; ++ti) {
                        int u[8];
                        float2 ds8;
                        const block_q8_1* i_ptr = use_si
                            ? (reinterpret_cast<const block_q8_1*>(s_inter[ti]) + b * 8 + sub)
                            : (inter_q8 + (size_t)(off + t_base + ti) * nb_ff + b * 8 + sub);
                        load_q8_1_sub_words(i_ptr, u, ds8);
                        ad[ti] += dot_unpacked_w_q8<W_TYPE>(wd, u, ds8);
                    }
                }
            }
            for (int ti = 0; ti < t_cnt; ++ti) {
                float v = warp_reduce_sum(ad[ti]);
                if (lane == 0) {
                    if constexpr (DET_PAIR) {
                        out_f32[(size_t)t_rows[ti] * n_embd + r] = t_rw[ti] * v;
                    } else {
                        atomicAdd(&out_f32[(size_t)t_rows[ti] * n_embd + r], t_rw[ti] * v);
                    }
                }
            }
        }
        if (use_si && (t_base + 8 < nt)) __syncthreads();
    }
}

extern "C" void launch_glm5_moe_prefill_small_gu(
    int32_t gu_type,
    const uint64_t* exp_ptrs,
    const int32_t* slot_off,
    const int32_t* slot_nt,
    const int32_t* rows_all,
    uint32_t u_off,
    size_t gu_row_bytes,
    const void* x_q8,
    void* inter_q8,
    int n_embd,
    int n_ff,
    int n_jobs,
    float clamp_lim,
    cudaStream_t stream
) {
    if (n_jobs <= 0 || n_ff <= 0 || n_embd <= 0) return;
    dim3 grid((n_ff + 31) / 32, n_jobs);
    dim3 block(32, 8);
    const auto* xq = reinterpret_cast<const block_q8_1*>(x_q8);
    auto* iq = reinterpret_cast<block_q8_1*>(inter_q8);
    switch (gu_type) {
        case GGML_TYPE_IQ2_S:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_IQ2_S><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_IQ3_XXS:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_IQ3_XXS><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_IQ3_S:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_IQ3_S><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_Q3_K:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_Q3_K><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_Q4_K:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_Q4_K><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_Q5_K:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_Q5_K><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_IQ4_XS:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_IQ4_XS><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_Q8_0:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_Q8_0><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_Q5_1:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_Q5_1><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_Q2_0:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_Q2_0><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_Q5_0:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_Q5_0><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_IQ4_NL:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_IQ4_NL><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        case GGML_TYPE_Q6_K:
            glm5_moe_prefill_small_gu_kernel<GGML_TYPE_Q6_K><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, u_off, gu_row_bytes, xq, iq, n_embd, n_ff, clamp_lim);
            break;
        default: {
            static std::atomic<int> s_small_gu_unsupported{0};
            if (s_small_gu_unsupported.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::fprintf(stderr, "[CUDA ERROR] launch_glm5_moe_prefill_small_gu: unsupported gu_type=%d\n", gu_type);
            }
            break;
        }
    }
}

template <bool DET_PAIR>
static void launch_glm5_moe_prefill_small_down_impl(
    int32_t d_type,
    const uint64_t* exp_ptrs,
    const int32_t* slot_off,
    const int32_t* slot_nt,
    const int32_t* rows_all,
    const float* wts_all,
    uint32_t d_off,
    size_t d_row_bytes,
    const void* inter_q8,
    float* out_f32,
    int n_embd,
    int n_ff,
    int n_jobs,
    cudaStream_t stream
) {
    if (n_jobs <= 0 || n_embd <= 0 || n_ff <= 0) return;
    dim3 grid((n_embd + 31) / 32, n_jobs);
    dim3 block(32, 8);
    const auto* iq = reinterpret_cast<const block_q8_1*>(inter_q8);
    switch (d_type) {
        case GGML_TYPE_IQ2_S:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_IQ2_S, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_IQ3_XXS:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_IQ3_XXS, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_IQ3_S:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_IQ3_S, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_Q3_K:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_Q3_K, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_Q4_K:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_Q4_K, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_Q5_K:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_Q5_K, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_IQ4_XS:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_IQ4_XS, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_Q8_0:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_Q8_0, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_Q5_1:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_Q5_1, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_Q2_0:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_Q2_0, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_Q5_0:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_Q5_0, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_IQ4_NL:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_IQ4_NL, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        case GGML_TYPE_Q6_K:
            glm5_moe_prefill_small_down_kernel<GGML_TYPE_Q6_K, DET_PAIR><<<grid, block, 0, stream>>>(
                exp_ptrs, slot_off, slot_nt, rows_all, wts_all, d_off, d_row_bytes, iq, out_f32, n_embd, n_ff);
            break;
        default: {
            static std::atomic<int> s_small_dn_unsupported{0};
            if (s_small_dn_unsupported.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::fprintf(stderr, "[CUDA ERROR] launch_glm5_moe_prefill_small_down: unsupported d_type=%d\n", d_type);
            }
            break;
        }
    }
}

extern "C" void launch_glm5_moe_prefill_small_down(
    int32_t d_type,
    const uint64_t* exp_ptrs,
    const int32_t* slot_off,
    const int32_t* slot_nt,
    const int32_t* rows_all,
    const float* wts_all,
    uint32_t d_off,
    size_t d_row_bytes,
    const void* inter_q8,
    float* out_f32,
    int n_embd,
    int n_ff,
    int n_jobs,
    cudaStream_t stream
) {
    launch_glm5_moe_prefill_small_down_impl<false>(
        d_type, exp_ptrs, slot_off, slot_nt, rows_all, wts_all,
        d_off, d_row_bytes, inter_q8, out_f32, n_embd, n_ff, n_jobs, stream);
}

extern "C" void launch_glm5_moe_prefill_small_down_det(
    int32_t d_type,
    const uint64_t* exp_ptrs,
    const int32_t* slot_off,
    const int32_t* slot_nt,
    const int32_t* pair_all,
    const float* wts_all,
    uint32_t d_off,
    size_t d_row_bytes,
    const void* inter_q8,
    float* pair_out_f32,
    int n_embd,
    int n_ff,
    int n_jobs,
    cudaStream_t stream
) {
    launch_glm5_moe_prefill_small_down_impl<true>(
        d_type, exp_ptrs, slot_off, slot_nt, pair_all, wts_all,
        d_off, d_row_bytes, inter_q8, pair_out_f32, n_embd, n_ff, n_jobs, stream);
}
