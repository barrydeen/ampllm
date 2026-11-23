#include "common/sm86_common.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ void hy4_mma16816_f32_f16(
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

constexpr int kHy4HcBlocks = 24;
constexpr int kHy4PostBlocks = 12;
constexpr int kHy4MaxM = 16;
constexpr int kHy4KvLora = 512;
constexpr int kHy4NRot = 64;
constexpr int kHy4RowW = (kHy4KvLora + kHy4NRot) / 2; // 288 half2s

__device__ __forceinline__ float hy4_sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float hy4_block_reduce_sum(float v) {
    __shared__ float smem[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw = blockDim.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) smem[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float acc = (lane < nw) ? smem[lane] : 0.0f;
        acc = warp_reduce_sum(acc);
        if (lane == 0) smem[0] = acc;
    }
    __syncthreads();
    return smem[0];
}

__device__ __forceinline__ void hy4_rope_pair(
    float& a,
    float& b,
    float pos,
    int pair_idx,
    int n_rot,
    float base
) {
    const float theta_scale = powf(base, -2.0f / (float)n_rot);
    const float theta = pos * powf(theta_scale, (float)pair_idx);
    float s, c;
    sincosf(theta, &s, &c);
    const float x0 = a;
    const float x1 = b;
    a = x0 * c - x1 * s;
    b = x0 * s + x1 * c;
}

__device__ __forceinline__ float hy4_f8e4m3_to_f32(uint8_t x) {
    const int ax = x & 0x7f;
    if (ax == 0x7f) return __int_as_float(0x7fc00000);
    const int e = (ax >> 3) & 0xf;
    const int m = ax & 7;
    const float v = (e == 0)
        ? ldexpf((float)m, -9)
        : ldexpf(1.0f + (float)m / 8.0f, e - 7);
    return (x & 0x80) ? -v : v;
}

__device__ __forceinline__ float hy4_gelu_tanh(float x) {
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

__device__ __forceinline__ float hy4_gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f));
}

// ---------------------------------------------------------------------------
// 1. iHC-4 Hyper-Connections
// ---------------------------------------------------------------------------

__global__ void hy4_expand_streams_kernel(
    const float* __restrict__ emb,
    int hc,
    int E,
    size_t total,
    float* __restrict__ out
) {
    const size_t stride_t = (size_t)hc * E;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t t = i / stride_t;
        const size_t d = i % (size_t)E;
        out[i] = emb[t * (size_t)E + d];
    }
}

template <bool BYEL>
__global__ void __launch_bounds__(256) hy4_hc_part_kernel(
    const float* __restrict__ x,
    int HE,
    int nm,
    const float* __restrict__ fn,
    float* __restrict__ part
) {
    __shared__ float sh[8][kHy4MaxM + 1];
    const int t = (int)blockIdx.y;
    const int b = (int)blockIdx.x;
    const int per = (HE + kHy4HcBlocks - 1) / kHy4HcBlocks;
    const int i0 = b * per;
    const int i1 = min(HE, i0 + per);
    const float* xt = x + (size_t)t * HE;

    float acc[kHy4MaxM + 1];
    #pragma unroll
    for (int j = 0; j <= kHy4MaxM; ++j) acc[j] = 0.0f;

    for (int i = i0 + (int)threadIdx.x; i < i1; i += (int)blockDim.x) {
        const float v = xt[i];
        acc[kHy4MaxM] = fmaf(v, v, acc[kHy4MaxM]);
        if constexpr (BYEL) {
            const float4* f4 = reinterpret_cast<const float4*>(fn + (size_t)i * nm);
            #pragma unroll
            for (int j = 0; j < kHy4MaxM; j += 4) {
                if (j < nm) {
                    const float4 wv = f4[j >> 2];
                    acc[j + 0] = fmaf(wv.x, v, acc[j + 0]);
                    acc[j + 1] = fmaf(wv.y, v, acc[j + 1]);
                    acc[j + 2] = fmaf(wv.z, v, acc[j + 2]);
                    acc[j + 3] = fmaf(wv.w, v, acc[j + 3]);
                }
            }
        } else {
            #pragma unroll
            for (int j = 0; j < kHy4MaxM; ++j) {
                if (j < nm) acc[j] = fmaf(fn[(size_t)j * HE + i], v, acc[j]);
            }
        }
    }

    const int w = (int)threadIdx.x >> 5;
    const int l = (int)threadIdx.x & 31;
    #pragma unroll
    for (int j = 0; j <= kHy4MaxM; ++j) {
        const float v = (j < nm || j == kHy4MaxM) ? warp_reduce_sum(acc[j]) : 0.0f;
        if (l == 0) sh[w][j] = v;
    }
    __syncthreads();

    if ((int)threadIdx.x <= kHy4MaxM) {
        float s = 0.0f;
        const int nw = (int)blockDim.x >> 5;
        for (int q = 0; q < nw; ++q) s += sh[q][threadIdx.x];
        part[((size_t)t * kHy4HcBlocks + b) * (kHy4MaxM + 1) + threadIdx.x] = s;
    }
}

template <bool HEAD, int KPER>
__global__ void __launch_bounds__(512) hy4_hc_fin_kernel(
    const float* __restrict__ x,
    int hc,
    int E,
    const float* __restrict__ part,
    const float* __restrict__ scale,
    const float* __restrict__ base,
    float eps,
    float hc_eps,
    float mag,
    const float* __restrict__ norm_w,
    float norm_eps,
    float* __restrict__ cur,
    float* __restrict__ post
) {
    __shared__ float s_mix[kHy4MaxM + 1];
    __shared__ float s_pre[8];
    const int t = (int)blockIdx.x;
    const int nm = HEAD ? hc : 2 * hc;
    const int HE = hc * E;

    if ((int)threadIdx.x <= kHy4MaxM) {
        float v = 0.0f;
        for (int b = 0; b < kHy4HcBlocks; ++b) {
            v += part[((size_t)t * kHy4HcBlocks + b) * (kHy4MaxM + 1) + threadIdx.x];
        }
        s_mix[threadIdx.x] = v;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        const float r = rsqrtf(s_mix[kHy4MaxM] / (float)HE + eps);
        for (int j = 0; j < nm; ++j) s_mix[j] *= r;
        for (int j = 0; j < hc; ++j) {
            s_pre[j] = hy4_sigmoidf(s_mix[j] * scale[0] + base[j]) + hc_eps;
        }
        if (!HEAD && post != nullptr) {
            for (int j = 0; j < hc; ++j) {
                post[(size_t)t * hc + j] =
                    hy4_sigmoidf(s_mix[hc + j] * scale[1] + base[hc + j]) * mag + hc_eps;
            }
        }
    }
    __syncthreads();

    const float* xt = x + (size_t)t * HE;
    float vals[KPER];
    float ss = 0.0f;
    #pragma unroll
    for (int q = 0; q < KPER; ++q) {
        const int d = (int)threadIdx.x + q * (int)blockDim.x;
        float a[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            a[j] = (d < E && j < hc) ? xt[(size_t)j * E + d] : 0.0f;
        }
        float v = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            v = fmaf((j < hc) ? s_pre[j] : 0.0f, a[j], v);
        }
        vals[q] = v;
        ss += v * v;
    }

    float r = 1.0f;
    if (norm_w != nullptr) {
        ss = hy4_block_reduce_sum(ss);
        r = rsqrtf(ss / (float)E + norm_eps);
    }

    #pragma unroll
    for (int q = 0; q < KPER; ++q) {
        const int d = (int)threadIdx.x + q * (int)blockDim.x;
        if (d < E) {
            cur[(size_t)t * E + d] = (norm_w != nullptr) ? (vals[q] * r * norm_w[d]) : vals[q];
        }
    }
}

__global__ void __launch_bounds__(256) hy4_hc_post_norm_kernel(
    float* __restrict__ x,
    int hc,
    int E,
    const float* __restrict__ y,
    const float* __restrict__ post,
    const float* __restrict__ w,
    float eps,
    float* __restrict__ y_norm_opt
) {
    const int t = (int)blockIdx.y;
    const int b = (int)blockIdx.x;
    const float* yt = y + (size_t)t * E;

    float ss = 0.0f;
    if ((E & 3) == 0 && (reinterpret_cast<uintptr_t>(yt) & 15u) == 0u) {
        for (int d = (int)threadIdx.x * 4; d < E; d += (int)blockDim.x * 4) {
            const float4 v = *reinterpret_cast<const float4*>(yt + d);
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int d = (int)threadIdx.x; d < E; d += (int)blockDim.x) {
            const float v = yt[d];
            ss += v * v;
        }
    }
    ss = hy4_block_reduce_sum(ss);
    const float r = rsqrtf(ss / (float)E + eps);

    const int per = (E + kHy4PostBlocks - 1) / kHy4PostBlocks;
    const int d0 = b * per;
    const int d1 = min(E, d0 + per);
    float* xt = x + (size_t)t * hc * E;

    __shared__ float s_pw[8];
    if ((int)threadIdx.x < hc) s_pw[threadIdx.x] = post[(size_t)t * hc + threadIdx.x];
    __syncthreads();
    for (int d = d0 + (int)threadIdx.x; d < d1; d += (int)blockDim.x) {
        const float v = yt[d] * r * w[d];
        if (y_norm_opt != nullptr) y_norm_opt[(size_t)t * E + d] = v;
        for (int j = 0; j < hc; ++j) {
            xt[(size_t)j * E + d] += v * s_pw[j];
        }
    }
}

__global__ void hy4_hc_post_add_kernel(
    float* __restrict__ x,
    int hc,
    int E,
    const float* __restrict__ y,
    const float* __restrict__ post,
    size_t total
) {
    const size_t stride_t = (size_t)hc * E;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t t = i / stride_t;
        const size_t s = (i / (size_t)E) % (size_t)hc;
        const size_t d = i % (size_t)E;
        x[i] += y[t * (size_t)E + d] * post[t * (size_t)hc + s];
    }
}

// ---------------------------------------------------------------------------
// 2. RoPE, Store Rows, Elementwise & Small F32/BF16 GEMV
// ---------------------------------------------------------------------------

__global__ void hy4_rope_kernel(
    float* __restrict__ x,
    int T,
    int nh,
    size_t ld_t,
    size_t ld_h,
    int off,
    int n_rot,
    const int32_t* __restrict__ pos,
    float base
) {
    const int half = n_rot >> 1;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)T * nh * half;
    if (idx >= total) return;
    const int p = (int)(idx % (size_t)half);
    const size_t tmp = idx / (size_t)half;
    const int h = (int)(tmp % (size_t)nh);
    const int t = (int)(tmp / (size_t)nh);
    float* v = x + (size_t)t * ld_t + (size_t)h * ld_h + off + 2 * p;
    hy4_rope_pair(v[0], v[1], (float)pos[t], p, n_rot, base);
}

__global__ void hy4_sigmoid_mul_kernel(float* __restrict__ x, const float* __restrict__ g, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        x[i] *= hy4_sigmoidf(g[i]);
    }
}

__global__ void hy4_swiglu_f32_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    size_t ld_gu,
    int rows,
    int n,
    float clamp_lim,
    float* __restrict__ out,
    size_t ld_out
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)rows * n;
    if (idx >= total) return;
    const int r = (int)(idx / (size_t)n);
    const int c = (int)(idx % (size_t)n);
    const float g = gate[(size_t)r * ld_gu + c];
    const float u = up[(size_t)r * ld_gu + c];
    float res;
    if (clamp_lim > 0.0f) {
        res = glm_swiglu_clamp_val(g, u, clamp_lim);
    } else {
        res = (g / (1.0f + expf(-g))) * u;
    }
    out[(size_t)r * ld_out + c] = res;
}

template <typename DstT>
__global__ void hy4_store_rows_kernel(
    const float* __restrict__ src,
    int T,
    int dim,
    DstT* __restrict__ ring,
    int ring_cap,
    const int32_t* __restrict__ pos
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)T * dim;
    if (idx >= total) return;
    const int t = (int)(idx / (size_t)dim);
    const int c = (int)(idx % (size_t)dim);
    const int p = pos[t];
    const int slot = (ring_cap > 0) ? (p % ring_cap) : p;
    const float v = src[(size_t)t * dim + c];
    if constexpr (sizeof(DstT) == 2) {
        ring[(size_t)slot * dim + c] = __float2half_rn(v);
    } else {
        ring[(size_t)slot * dim + c] = v;
    }
}

template <int BM, typename WT>
__global__ void __launch_bounds__(128) hy4_gemv_dense_kernel(
    const WT* __restrict__ W,
    int N,
    int K,
    const float* __restrict__ X,
    int M,
    float* __restrict__ Y,
    float alpha
) {
    const int warp = (int)threadIdx.y;
    const int lane = (int)threadIdx.x;
    const int r = (int)blockIdx.x * 4 + warp;
    const int m0 = (int)blockIdx.y * BM;
    if (r >= N || m0 >= M) return;
    const int mc = min(BM, M - m0);
    const WT* wr = W + (size_t)r * K;

    float acc[BM];
    #pragma unroll
    for (int m = 0; m < BM; ++m) acc[m] = 0.0f;

    if ((K & 3) == 0 && (reinterpret_cast<uintptr_t>(X) & 15u) == 0u) {
        for (int c = lane * 4; c < K; c += 128) {
            float4 wv;
            if constexpr (sizeof(WT) == 4) {
                wv = *reinterpret_cast<const float4*>(wr + c);
            } else {
                const uint2 u = *reinterpret_cast<const uint2*>(wr + c);
                wv.x = bf16_bits_to_f32((uint16_t)(u.x & 0xFFFFu));
                wv.y = bf16_bits_to_f32((uint16_t)(u.x >> 16));
                wv.z = bf16_bits_to_f32((uint16_t)(u.y & 0xFFFFu));
                wv.w = bf16_bits_to_f32((uint16_t)(u.y >> 16));
            }
            #pragma unroll
            for (int m = 0; m < BM; ++m) {
                if (m < mc) {
                    const float4 xv = *reinterpret_cast<const float4*>(X + (size_t)(m0 + m) * K + c);
                    acc[m] = fmaf(wv.x, xv.x, fmaf(wv.y, xv.y, fmaf(wv.z, xv.z, fmaf(wv.w, xv.w, acc[m]))));
                }
            }
        }
    } else {
        for (int c = lane; c < K; c += 32) {
            float wv;
            if constexpr (sizeof(WT) == 4) wv = wr[c];
            else wv = bf16_bits_to_f32(wr[c]);
            #pragma unroll
            for (int m = 0; m < BM; ++m) {
                if (m < mc) acc[m] = fmaf(wv, X[(size_t)(m0 + m) * K + c], acc[m]);
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < BM; ++m) {
        if (m < mc) {
            const float s = warp_reduce_sum(acc[m]);
            if (lane == 0) Y[(size_t)(m0 + m) * N + r] = s * alpha;
        }
    }
}

template <typename WT>
void dispatch_gemv_dense(
    const WT* W,
    int N,
    int K,
    const float* X,
    int M,
    float* Y,
    float alpha,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0 || M <= 0) return;
    dim3 block(32, 4);
    if (M == 1) {
        dim3 grid((N + 3) / 4, 1);
        hy4_gemv_dense_kernel<1, WT><<<grid, block, 0, stream>>>(W, N, K, X, M, Y, alpha);
    } else if (M <= 4) {
        dim3 grid((N + 3) / 4, 1);
        hy4_gemv_dense_kernel<4, WT><<<grid, block, 0, stream>>>(W, N, K, X, M, Y, alpha);
    } else {
        dim3 grid((N + 3) / 4, (M + 7) / 8);
        hy4_gemv_dense_kernel<8, WT><<<grid, block, 0, stream>>>(W, N, K, X, M, Y, alpha);
    }
}

// ---------------------------------------------------------------------------
// 3. Batched Per-Head Q8_0 Linear + Strided Head Q8_1 Quantization
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(128) hy4_quantize_heads_q8_1_kernel(
    const float* __restrict__ X,
    block_q8_1* __restrict__ X_q8,
    int T,
    int H,
    int K,
    size_t ld_t,
    size_t ld_h
) {
    const int nb = K >> 5;
    const int b = (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    const int t = (int)blockIdx.z;
    const int tid = (int)threadIdx.x;
    if (b >= nb || h >= H || t >= T) return;
    // Grid blockIdx.x covers all blocks; each thread handles 1 elem of the 32-elem block.
    // Block size = 32 -> Wait: use 128 threads = 4 blocks per CTA to reduce launch overhead.
    const int blk = b * 4 + (tid >> 5);
    if (blk >= nb) return;
    const int lane = tid & 31;
    const float val = X[(size_t)t * ld_t + (size_t)h * ld_h + blk * 32 + lane];
    block_q8_1* dst = X_q8 + ((size_t)t * H + h) * nb + blk;
    warp_quantize_q8_1_store(val, dst, lane);
}

__global__ void __launch_bounds__(128) hy4_batched_heads_q8_kernel(
    const block_q8_0* __restrict__ W,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y,
    int T,
    int H,
    int N,
    int K,
    size_t x_tok_stride_blk,
    size_t y_tok_stride,
    size_t y_head_stride
) {
    const int warp = (int)threadIdx.y;
    const int lane = (int)threadIdx.x;
    const int tid = warp * 32 + lane;
    const int h = (int)blockIdx.y;
    const int t = (int)blockIdx.z;
    if (h >= H || t >= T) return;

    const int nb = K >> 5;
    __shared__ __align__(16) uint32_t s_xq8[32 * 9]; // up to K=1024 staged in SMEM
    const block_q8_1* x_head = X_q8 + (size_t)t * x_tok_stride_blk + (size_t)h * nb;
    const bool staged = (nb <= 32);
    if (staged) {
        const uint32_t* gx = reinterpret_cast<const uint32_t*>(x_head);
        const int n_words = nb * 9;
        for (int i = tid; i < n_words; i += 128) s_xq8[i] = gx[i];
        __syncthreads();
    }
    const block_q8_1* x_use = staged ? reinterpret_cast<const block_q8_1*>(s_xq8) : x_head;

    const int r0 = ((int)blockIdx.x * 4 + warp) * 4;
    if (r0 >= N) return;

    const block_q8_0* w_head = W + ((size_t)h * N * nb);
    float* y_out = Y + (size_t)t * y_tok_stride + (size_t)h * y_head_stride;

    if (r0 + 3 < N) {
        float o0, o1, o2, o3;
        dot_quad_q8_0_sub(
            w_head + (size_t)(r0 + 0) * nb,
            w_head + (size_t)(r0 + 1) * nb,
            w_head + (size_t)(r0 + 2) * nb,
            w_head + (size_t)(r0 + 3) * nb,
            x_use, K, lane, o0, o1, o2, o3
        );
        if (lane == 0) {
            y_out[r0 + 0] = o0;
            y_out[r0 + 1] = o1;
            y_out[r0 + 2] = o2;
            y_out[r0 + 3] = o3;
        }
    } else {
        for (int dr = 0; dr < 4; ++dr) {
            const int r = r0 + dr;
            if (r < N) {
                const float v = dot_row_q8_0_sub(w_head + (size_t)r * nb, x_use, K, lane);
                if (lane == 0) y_out[r] = v;
            }
        }
    }
}

// Tensor Core batched per-head Q8_0 linear for prefill (`T > 16`):
// Y[t, h, n] = sum_k X[t, h, k] * W_q8_0[h, n, k], reading F32 X directly into F16 MMA tiles.
__global__ void __launch_bounds__(128) hy4_batched_heads_q8_mma_kernel(
    const block_q8_0* __restrict__ W,
    const float* __restrict__ X_f32,
    float* __restrict__ Y,
    int T,
    int H,
    int N,
    int K,
    size_t x_tok_stride,
    size_t x_head_stride,
    size_t y_tok_stride,
    size_t y_head_stride
) {
    __shared__ __align__(16) __half s_A[64][36];
    __shared__ __align__(16) __half s_B[64][36];

    const int n0 = (int)blockIdx.x * 64;
    const int t0 = (int)blockIdx.y * 64;
    const int h = (int)blockIdx.z;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int warp_m = warp >> 1; // 0..1 (32 tokens per warp)
    const int warp_n = warp & 1;  // 0..1 (32 outputs per warp)
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int kp = t4 * 2;
    const int nb = K >> 5;

    float c[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; ++mi)
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni)
            c[mi][ni][0] = c[mi][ni][1] = c[mi][ni][2] = c[mi][ni][3] = 0.0f;

    for (int bk = 0; bk < nb; ++bk) {
        const int k0 = bk * 32;
        #pragma unroll
        for (int r_off = 0; r_off < 64; r_off += 16) {
            const int mr = r_off + (tid >> 3);
            const int kc = (tid & 7) * 4;
            if (t0 + mr < T) {
                const float* xp = X_f32 + (size_t)(t0 + mr) * x_tok_stride + (size_t)h * x_head_stride + k0 + kc;
                *reinterpret_cast<__half2*>(&s_A[mr][kc + 0]) = __floats2half2_rn(xp[0], xp[1]);
                *reinterpret_cast<__half2*>(&s_A[mr][kc + 2]) = __floats2half2_rn(xp[2], xp[3]);
            } else {
                *reinterpret_cast<__half2*>(&s_A[mr][kc + 0]) = __floats2half2_rn(0.0f, 0.0f);
                *reinterpret_cast<__half2*>(&s_A[mr][kc + 2]) = __floats2half2_rn(0.0f, 0.0f);
            }
        }
        #pragma unroll
        for (int r_off = 0; r_off < 64; r_off += 16) {
            const int nr = r_off + (tid >> 3);
            const int kc = (tid & 7) * 4;
            if (n0 + nr < N) {
                const block_q8_0* blk = W + ((size_t)(h * N + n0 + nr) * nb + bk);
                const uint16_t d_bits = blk->d;
                const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
                *reinterpret_cast<__half2*>(&s_B[nr][kc + 0]) = __floats2half2_rn(
                    d * (float)blk->qs[kc + 0],
                    d * (float)blk->qs[kc + 1]
                );
                *reinterpret_cast<__half2*>(&s_B[nr][kc + 2]) = __floats2half2_rn(
                    d * (float)blk->qs[kc + 2],
                    d * (float)blk->qs[kc + 3]
                );
            } else {
                *reinterpret_cast<__half2*>(&s_B[nr][kc + 0]) = __floats2half2_rn(0.0f, 0.0f);
                *reinterpret_cast<__half2*>(&s_B[nr][kc + 2]) = __floats2half2_rn(0.0f, 0.0f);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            uint32_t a[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; ++mi) {
                const int r = warp_m * 32 + mi * 16 + g;
                a[mi][0] = *reinterpret_cast<const uint32_t*>(&s_A[r][kk + kp]);
                a[mi][1] = *reinterpret_cast<const uint32_t*>(&s_A[r + 8][kk + kp]);
                a[mi][2] = *reinterpret_cast<const uint32_t*>(&s_A[r][kk + kp + 8]);
                a[mi][3] = *reinterpret_cast<const uint32_t*>(&s_A[r + 8][kk + kp + 8]);
            }
            #pragma unroll
            for (int ni = 0; ni < 4; ++ni) {
                const int n_row = warp_n * 32 + ni * 8 + g;
                const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&s_B[n_row][kk + kp]);
                const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&s_B[n_row][kk + kp + 8]);
                #pragma unroll
                for (int mi = 0; mi < 2; ++mi) {
                    hy4_mma16816_f32_f16(c[mi][ni], a[mi][0], a[mi][1], a[mi][2], a[mi][3], b0, b1);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
        const int r0 = t0 + warp_m * 32 + mi * 16 + g;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
            const int c0 = n0 + warp_n * 32 + ni * 8 + kp;
            if (r0 < T) {
                float* yp = Y + (size_t)r0 * y_tok_stride + (size_t)h * y_head_stride;
                if (c0 < N) yp[c0] = c[mi][ni][0];
                if (c0 + 1 < N) yp[c0 + 1] = c[mi][ni][1];
            }
            if (r1 < T) {
                float* yp = Y + (size_t)r1 * y_tok_stride + (size_t)h * y_head_stride;
                if (c0 < N) yp[c0] = c[mi][ni][2];
                if (c0 + 1 < N) yp[c0 + 1] = c[mi][ni][3];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. KV Store (F16 & Q8_0)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256) hy4_kv_store_f16_kernel(
    const float* __restrict__ kv,
    int kv_lora,
    int n_rot,
    const float* __restrict__ w,
    float eps,
    const int32_t* __restrict__ pos,
    float base,
    __half* __restrict__ cache,
    int ring_cap
) {
    const int t = (int)blockIdx.x;
    const int row = kv_lora + n_rot;
    const float* x = kv + (size_t)t * row;
    const int p = pos[t];
    const int slot = (ring_cap > 0) ? (p % ring_cap) : p;
    __half* dst = cache + (size_t)slot * row;

    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < kv_lora; i += (int)blockDim.x) ss += x[i] * x[i];
    ss = hy4_block_reduce_sum(ss);
    const float r = rsqrtf(ss / (float)kv_lora + eps);

    for (int i = (int)threadIdx.x; i < kv_lora; i += (int)blockDim.x) {
        dst[i] = __float2half_rn(x[i] * r * w[i]);
    }
    const int half_rot = n_rot >> 1;
    for (int i = (int)threadIdx.x; i < half_rot; i += (int)blockDim.x) {
        float a = x[kv_lora + 2 * i];
        float b = x[kv_lora + 2 * i + 1];
        hy4_rope_pair(a, b, (float)p, i, n_rot, base);
        dst[kv_lora + 2 * i] = __float2half_rn(a);
        dst[kv_lora + 2 * i + 1] = __float2half_rn(b);
    }
}

__global__ void __launch_bounds__(256) hy4_kv_store_q8_kernel(
    const float* __restrict__ kv,
    int kv_lora,
    int n_rot,
    const float* __restrict__ w,
    float eps,
    const int32_t* __restrict__ pos,
    float base,
    uint8_t* __restrict__ cache,
    int ring_cap,
    int rb
) {
    __shared__ float v[1024];
    const int t = (int)blockIdx.x;
    const int row = kv_lora + n_rot;
    const float* x = kv + (size_t)t * row;
    const int p = pos[t];
    const int slot = (ring_cap > 0) ? (p % ring_cap) : p;
    uint8_t* dst = cache + (size_t)slot * rb;

    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < kv_lora; i += (int)blockDim.x) ss += x[i] * x[i];
    ss = hy4_block_reduce_sum(ss);
    const float r = rsqrtf(ss / (float)kv_lora + eps);

    for (int i = (int)threadIdx.x; i < kv_lora; i += (int)blockDim.x) {
        v[i] = x[i] * r * w[i];
    }
    const int half_rot = n_rot >> 1;
    for (int i = (int)threadIdx.x; i < half_rot; i += (int)blockDim.x) {
        float a = x[kv_lora + 2 * i];
        float b = x[kv_lora + 2 * i + 1];
        hy4_rope_pair(a, b, (float)p, i, n_rot, base);
        v[kv_lora + 2 * i] = a;
        v[kv_lora + 2 * i + 1] = b;
    }
    __syncthreads();

    const int wp = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int nw = (int)blockDim.x >> 5;
    __half* sc = reinterpret_cast<__half*>(dst + kv_lora);
    __half* rp = sc + (kv_lora >> 5);

    for (int i = (int)threadIdx.x; i < n_rot; i += (int)blockDim.x) {
        rp[i] = __float2half_rn(v[kv_lora + i]);
    }
    const int n_blk = kv_lora >> 5;
    for (int b = wp; b < n_blk; b += nw) {
        const float x0 = v[32 * b + lane];
        const float am = warp_reduce_max(fabsf(x0));
        const __half dh = __float2half_rn(am / 127.0f);
        const float d = __half2float(dh);
        const int q = (d > 0.0f) ? max(-127, min(127, __float2int_rn(x0 / d))) : 0;
        dst[32 * b + lane] = static_cast<uint8_t>(static_cast<int8_t>(q));
        if (lane == 0) sc[b] = dh;
    }
    const int used_bytes = kv_lora + (kv_lora >> 4) + 2 * n_rot;
    for (int i = used_bytes + (int)threadIdx.x; i < rb; i += (int)blockDim.x) dst[i] = 0;
}

// ---------------------------------------------------------------------------
// 5. MLA Attention (Split-K Decode & Online-Softmax Prefill, F16 & Q8_0 KV, SWA & DSA, Sinks)
// ---------------------------------------------------------------------------

template <bool Q8>
__device__ __forceinline__ void hy4_load_kv_tile_rowwise(
    const void* __restrict__ cache_v,
    int ring_cap,
    int rb,
    const int32_t* __restrict__ keys_opt,
    int ldk,
    int t,
    int lo,
    int j0,
    int nt,
    int tid,
    __half2 (&tile)[32][kHy4RowW]
) {
    if constexpr (!Q8) {
        const uint4* cache_u4 = reinterpret_cast<const uint4*>(cache_v);
        const int n_u4 = nt * 72; // 288 half2 = 72 uint4 per row
        for (int idx = tid; idx < n_u4; idx += 256) {
            const int r = idx / 72;
            const int c4 = idx % 72;
            const int p = keys_opt ? keys_opt[(size_t)t * ldk + (j0 + r)] : (lo + j0 + r);
            const int slot = (ring_cap > 0) ? (p % ring_cap) : p;
            const uint4 v = cache_u4[(size_t)slot * 72 + c4];
            reinterpret_cast<uint4*>(&tile[r][c4 * 4])[0] = v;
        }
    } else {
        const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache_v);
        const int n_pairs = nt * kHy4RowW;
        for (int idx = tid; idx < n_pairs; idx += 256) {
            const int r = idx / kHy4RowW;
            const int c = idx % kHy4RowW;
            const int p = keys_opt ? keys_opt[(size_t)t * ldk + (j0 + r)] : (lo + j0 + r);
            const int slot = (ring_cap > 0) ? (p % ring_cap) : p;
            const uint8_t* rp = cache_u8 + (size_t)slot * rb;
            __half2 val;
            if (c < 256) {
                const char2 qv = *reinterpret_cast<const char2*>(rp + 2 * c);
                const float d = __half2float(*reinterpret_cast<const __half*>(rp + 512 + 2 * (c >> 4)));
                val = __floats2half2_rn(d * (float)qv.x, d * (float)qv.y);
            } else {
                val = *reinterpret_cast<const __half2*>(rp + 512 + 32 + 2 * (2 * (c - 256)));
            }
            tile[r][c] = val;
        }
    }
}

template <bool Q8>
__global__ void __launch_bounds__(256) hy4_mla_decode_split_kernel(
    const float* __restrict__ q,
    int H,
    const void* __restrict__ cache_v,
    int ring_cap,
    const int32_t* __restrict__ keys_opt,
    int ldk,
    const int32_t* __restrict__ lo_opt,
    const int32_t* __restrict__ nk,
    float scale,
    int nchunk,
    int kc,
    float* __restrict__ ws,
    int rb
) {
    __shared__ __half2 tile[32][kHy4RowW];
    const int c = (int)blockIdx.x;
    const int t = (int)blockIdx.z;
    const int w = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int tid = (int)threadIdx.x;
    const int h = (int)blockIdx.y * 8 + w;

    const int n_keys = nk[t];
    const int k0 = c * kc;
    const int k1 = min(n_keys, k0 + kc);

    if (k0 >= k1) {
        if (h < H && lane == 0) {
            float* o = ws + (((size_t)t * H + h) * nchunk + c) * 514;
            o[0] = -INFINITY;
            o[1] = 0.0f;
        }
        return;
    }

    const int lo = lo_opt ? lo_opt[t] : 0;
    float qr[18];
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 18; ++i) qr[i] = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

    if (h < H) {
        const float* qh = q + ((size_t)t * H + h) * 576;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            const int d = 2 * (lane + 32 * i);
            qr[2 * i + 0] = qh[d + 0];
            qr[2 * i + 1] = qh[d + 1];
        }
    }

    float m = -INFINITY;
    float l = 0.0f;
    for (int j0 = k0; j0 < k1; j0 += 32) {
        const int nt = min(32, k1 - j0);
        hy4_load_kv_tile_rowwise<Q8>(cache_v, ring_cap, rb, keys_opt, ldk, t, lo, j0, nt, tid, tile);
        __syncthreads();

        if (h < H) {
            for (int j = 0; j < nt; ++j) {
                float s = 0.0f;
                #pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const float2 kv = __half22float2(tile[j][lane + 32 * i]);
                    s = fmaf(qr[2 * i + 0], kv.x, fmaf(qr[2 * i + 1], kv.y, s));
                }
                s = warp_reduce_sum(s) * scale;
                const float mn = fmaxf(m, s);
                if (mn == m) {
                    const float pj = expf(s - mn);
                    l += pj;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float2 vv = __half22float2(tile[j][lane + 32 * i]);
                        acc[2 * i + 0] = fmaf(pj, vv.x, acc[2 * i + 0]);
                        acc[2 * i + 1] = fmaf(pj, vv.y, acc[2 * i + 1]);
                    }
                } else {
                    const float corr = (m == -INFINITY) ? 0.0f : expf(m - mn);
                    const float pj = expf(s - mn);
                    l = l * corr + pj;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float2 vv = __half22float2(tile[j][lane + 32 * i]);
                        acc[2 * i + 0] = fmaf(pj, vv.x, acc[2 * i + 0] * corr);
                        acc[2 * i + 1] = fmaf(pj, vv.y, acc[2 * i + 1] * corr);
                    }
                }
                m = mn;
            }
        }
        __syncthreads();
    }

    if (h >= H) return;
    float* o = ws + (((size_t)t * H + h) * nchunk + c) * 514;
    if (lane == 0) {
        o[0] = m;
        o[1] = l;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        o[2 + 2 * (lane + 32 * i) + 0] = acc[2 * i + 0];
        o[2 + 2 * (lane + 32 * i) + 1] = acc[2 * i + 1];
    }
}

__global__ void __launch_bounds__(512) hy4_mla_combine_kernel(
    int H,
    int nchunk,
    int kc,
    const int32_t* __restrict__ nk,
    const float* __restrict__ sinks_opt,
    const float* __restrict__ ws,
    float* __restrict__ out
) {
    const int t = (int)blockIdx.y;
    const int h = (int)blockIdx.x;
    const int d = (int)threadIdx.x;
    const int used = min(nchunk, (nk[t] + kc - 1) / kc);
    const float* base = ws + ((size_t)t * H + h) * nchunk * 514;

    float M = (sinks_opt != nullptr) ? sinks_opt[h] : -INFINITY;
    for (int c = 0; c < used; ++c) {
        M = fmaxf(M, base[(size_t)c * 514]);
    }
    float den = (sinks_opt != nullptr) ? expf(sinks_opt[h] - M) : 0.0f;
    float num = 0.0f;
    for (int c = 0; c < used; ++c) {
        const float* o = base + (size_t)c * 514;
        if (o[0] == -INFINITY) continue;
        const float f = expf(o[0] - M);
        den += o[1] * f;
        num += o[2 + d] * f;
    }
    out[((size_t)t * H + h) * 512 + d] = (den > 0.0f) ? (num / den) : 0.0f;
}

template <bool Q8>
__global__ void __launch_bounds__(256) hy4_mla_prefill_kernel(
    const float* __restrict__ q,
    int H,
    const void* __restrict__ cache_v,
    int ring_cap,
    const int32_t* __restrict__ keys_opt,
    int ldk,
    const int32_t* __restrict__ lo_opt,
    const int32_t* __restrict__ nk,
    float scale,
    const float* __restrict__ sinks_opt,
    float* __restrict__ out,
    int rb
) {
    __shared__ __half2 tile[32][kHy4RowW];
    const int t = (int)blockIdx.y;
    const int w = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int tid = (int)threadIdx.x;
    const int h = (int)blockIdx.x * 8 + w;

    const int n_keys = nk[t];
    const int lo = lo_opt ? lo_opt[t] : 0;

    float qr[18];
    float acc[16];
    #pragma unroll
    for (int i = 0; i < 18; ++i) qr[i] = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) acc[i] = 0.0f;

    if (h < H) {
        const float* qh = q + ((size_t)t * H + h) * 576;
        #pragma unroll
        for (int i = 0; i < 9; ++i) {
            const int d = 2 * (lane + 32 * i);
            qr[2 * i + 0] = qh[d + 0];
            qr[2 * i + 1] = qh[d + 1];
        }
    }

    float m = -INFINITY;
    float l = 0.0f;
    for (int j0 = 0; j0 < n_keys; j0 += 32) {
        const int nt = min(32, n_keys - j0);
        hy4_load_kv_tile_rowwise<Q8>(cache_v, ring_cap, rb, keys_opt, ldk, t, lo, j0, nt, tid, tile);
        __syncthreads();

        if (h < H) {
            for (int j = 0; j < nt; ++j) {
                float s = 0.0f;
                #pragma unroll
                for (int i = 0; i < 9; ++i) {
                    const float2 kv = __half22float2(tile[j][lane + 32 * i]);
                    s = fmaf(qr[2 * i + 0], kv.x, fmaf(qr[2 * i + 1], kv.y, s));
                }
                s = warp_reduce_sum(s) * scale;
                const float mn = fmaxf(m, s);
                if (mn == m) {
                    const float pj = expf(s - mn);
                    l += pj;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float2 vv = __half22float2(tile[j][lane + 32 * i]);
                        acc[2 * i + 0] = fmaf(pj, vv.x, acc[2 * i + 0]);
                        acc[2 * i + 1] = fmaf(pj, vv.y, acc[2 * i + 1]);
                    }
                } else {
                    const float corr = (m == -INFINITY) ? 0.0f : expf(m - mn);
                    const float pj = expf(s - mn);
                    l = l * corr + pj;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float2 vv = __half22float2(tile[j][lane + 32 * i]);
                        acc[2 * i + 0] = fmaf(pj, vv.x, acc[2 * i + 0] * corr);
                        acc[2 * i + 1] = fmaf(pj, vv.y, acc[2 * i + 1] * corr);
                    }
                }
                m = mn;
            }
        }
        __syncthreads();
    }

    if (h >= H) return;
    float mult = 0.0f;
    if (sinks_opt != nullptr) {
        const float sink_v = sinks_opt[h];
        const float mn = fmaxf(m, sink_v);
        const float corr = (m == -INFINITY) ? 0.0f : expf(m - mn);
        const float den = l * corr + expf(sink_v - mn);
        mult = (den > 0.0f && m > -INFINITY) ? (corr / den) : 0.0f;
    } else {
        mult = (l > 0.0f) ? (1.0f / l) : 0.0f;
    }
    float* oh = out + ((size_t)t * H + h) * 512;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int idx = 2 * (lane + 32 * i);
        oh[idx + 0] = acc[2 * i + 0] * mult;
        oh[idx + 1] = acc[2 * i + 1] * mult;
    }
}

// Tensor Core MLA Prefill across 16 heads per CTA:
// Computes Q[16, 576] @ K[nt, 576]^T and P[16, nt] @ V[nt, 512] using mma.sync.aligned.m16n8k16.
constexpr int kHy4MlaMmaSmemBytes = 59328;

template <bool Q8>
__global__ void __launch_bounds__(128) hy4_mla_prefill_mma_kernel(
    const float* __restrict__ q,
    int H,
    const void* __restrict__ cache_v,
    int ring_cap,
    const int32_t* __restrict__ keys_opt,
    int ldk,
    const int32_t* __restrict__ lo_opt,
    const int32_t* __restrict__ nk,
    float scale,
    const float* __restrict__ sinks_opt,
    float* __restrict__ out,
    int rb
) {
    extern __shared__ float4 mla_mma_smem[];
    uint8_t* smem_u8 = reinterpret_cast<uint8_t*>(mla_mma_smem);
    __half (*s_Q)[584] = reinterpret_cast<__half (*)[584]>(smem_u8);
    __half (*s_KV)[584] = reinterpret_cast<__half (*)[584]>(smem_u8 + 18688);
    float (*s_S)[32] = reinterpret_cast<float (*)[32]>(smem_u8 + 56064);
    __half (*s_P)[32] = reinterpret_cast<__half (*)[32]>(smem_u8 + 58112);
    float* s_m = reinterpret_cast<float*>(smem_u8 + 59136);
    float* s_l = reinterpret_cast<float*>(smem_u8 + 59200);
    float* s_corr = reinterpret_cast<float*>(smem_u8 + 59264);

    const int h0 = (int)blockIdx.x * 16;
    const int t = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int w = tid >> 5;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int t4 = lane & 3;
    const int kp = t4 * 2;

    const int n_keys = nk[t];
    const int lo = lo_opt ? lo_opt[t] : 0;

    // Load Q[16, 576] F32 -> s_Q[16][576] F16
    for (int idx = tid; idx < 16 * 144; idx += 128) {
        const int hr = idx / 144;
        const int d4 = (idx % 144) * 4;
        if (h0 + hr < H) {
            const float* qp = q + ((size_t)t * H + h0 + hr) * 576 + d4;
            *reinterpret_cast<__half2*>(&s_Q[hr][d4 + 0]) = __floats2half2_rn(qp[0], qp[1]);
            *reinterpret_cast<__half2*>(&s_Q[hr][d4 + 2]) = __floats2half2_rn(qp[2], qp[3]);
        } else {
            *reinterpret_cast<__half2*>(&s_Q[hr][d4 + 0]) = __floats2half2_rn(0.0f, 0.0f);
            *reinterpret_cast<__half2*>(&s_Q[hr][d4 + 2]) = __floats2half2_rn(0.0f, 0.0f);
        }
    }
    if (tid < 16) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
        s_corr[tid] = 1.0f;
    }

    float o_acc[16][4];
    #pragma unroll
    for (int c = 0; c < 16; ++c) {
        o_acc[c][0] = o_acc[c][1] = o_acc[c][2] = o_acc[c][3] = 0.0f;
    }
    __syncthreads();

    for (int j0 = 0; j0 < n_keys; j0 += 32) {
        const int nt = min(32, n_keys - j0);
        if constexpr (!Q8) {
            const uint4* cache_u4 = reinterpret_cast<const uint4*>(cache_v);
            const int n_u4 = nt * 72;
            for (int idx = tid; idx < n_u4; idx += 128) {
                const int r = idx / 72;
                const int c4 = idx % 72;
                const int p = keys_opt ? keys_opt[(size_t)t * ldk + (j0 + r)] : (lo + j0 + r);
                const int slot = (ring_cap > 0) ? (p % ring_cap) : p;
                const uint4 v = cache_u4[(size_t)slot * 72 + c4];
                reinterpret_cast<uint4*>(&s_KV[r][c4 * 8])[0] = v;
            }
        } else {
            const uint8_t* cache_u8 = reinterpret_cast<const uint8_t*>(cache_v);
            const int n_pairs = nt * kHy4RowW;
            for (int idx = tid; idx < n_pairs; idx += 128) {
                const int r = idx / kHy4RowW;
                const int c = idx % kHy4RowW;
                const int p = keys_opt ? keys_opt[(size_t)t * ldk + (j0 + r)] : (lo + j0 + r);
                const int slot = (ring_cap > 0) ? (p % ring_cap) : p;
                const uint8_t* rp = cache_u8 + (size_t)slot * rb;
                __half2 val;
                if (c < 256) {
                    const char2 qv = *reinterpret_cast<const char2*>(rp + 2 * c);
                    const float d = __half2float(*reinterpret_cast<const __half*>(rp + 512 + 2 * (c >> 4)));
                    val = __floats2half2_rn(d * (float)qv.x, d * (float)qv.y);
                } else {
                    val = *reinterpret_cast<const __half2*>(rp + 512 + 32 + 2 * (2 * (c - 256)));
                }
                *reinterpret_cast<__half2*>(&s_KV[r][2 * c]) = val;
            }
        }
        // Zero trailing padded KV rows on partial tile to prevent 0 * NaN inside Tensor Cores
        if (nt < 32) {
            const int rem_u4 = (32 - nt) * 72;
            const uint4 z4 = make_uint4(0u, 0u, 0u, 0u);
            for (int idx = tid; idx < rem_u4; idx += 128) {
                const int r = nt + idx / 72;
                const int c4 = idx % 72;
                reinterpret_cast<uint4*>(&s_KV[r][c4 * 8])[0] = z4;
            }
        }
        __syncthreads();

        // Phase 1: S[16, 32] = Q[16, 576] * KV[32, 576]^T (warp w owns 8 keys)
        const int n_off = w * 8;
        float s_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll 1
        for (int k = 0; k < 576; k += 16) {
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_Q[g][k + kp]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_Q[g + 8][k + kp]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_Q[g][k + kp + 8]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_Q[g + 8][k + kp + 8]);
            const uint32_t b0 = *reinterpret_cast<const uint32_t*>(&s_KV[n_off + g][k + kp]);
            const uint32_t b1 = *reinterpret_cast<const uint32_t*>(&s_KV[n_off + g][k + kp + 8]);
            hy4_mma16816_f32_f16(s_acc, a0, a1, a2, a3, b0, b1);
        }
        s_S[g][n_off + kp + 0] = s_acc[0];
        s_S[g][n_off + kp + 1] = s_acc[1];
        s_S[g + 8][n_off + kp + 0] = s_acc[2];
        s_S[g + 8][n_off + kp + 1] = s_acc[3];
        __syncthreads();

        // Online softmax: 4 rows per warp -> P[16][32] F16 + s_m, s_l, s_corr
        #pragma unroll
        for (int ri = 0; ri < 4; ++ri) {
            const int r = w * 4 + ri;
            const float val = (lane < nt) ? (s_S[r][lane] * scale) : -INFINITY;
            const float mx = warp_reduce_max(val);
            const float prev_m = s_m[r];
            const float mn = fmaxf(prev_m, mx);
            const float corr = (prev_m == -INFINITY) ? 0.0f : expf(prev_m - mn);
            const float p = (lane < nt) ? expf(val - mn) : 0.0f;
            const float rs = warp_reduce_sum(p);
            s_P[r][lane] = __float2half_rn(p);
            if (lane == 0) {
                s_m[r] = mn;
                s_l[r] = s_l[r] * corr + rs;
                s_corr[r] = corr;
            }
        }
        __syncthreads();

        // Phase 2: O[16, 512] = corr * O + P[16, 32] * KV[32, 512] (warp w owns 128 V dims)
        const float corr0 = s_corr[g];
        const float corr1 = s_corr[g + 8];
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            o_acc[c][0] *= corr0;
            o_acc[c][1] *= corr0;
            o_acc[c][2] *= corr1;
            o_acc[c][3] *= corr1;
        }
        #pragma unroll
        for (int kk = 0; kk < 32; kk += 16) {
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_P[g][kk + kp]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_P[g + 8][kk + kp]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_P[g][kk + kp + 8]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_P[g + 8][kk + kp + 8]);
            const int r_b0 = kk + kp;
            const int r_b1 = kk + kp + 8;
            const __half* kv_row0 = &s_KV[r_b0][w * 128 + g];
            const __half* kv_row1 = &s_KV[r_b0 + 1][w * 128 + g];
            const __half* kv_row2 = &s_KV[r_b1][w * 128 + g];
            const __half* kv_row3 = &s_KV[r_b1 + 1][w * 128 + g];
            #pragma unroll
            for (int c = 0; c < 16; ++c) {
                const uint32_t b0 = (uint32_t)(*reinterpret_cast<const uint16_t*>(&kv_row0[c * 8])) |
                                    ((uint32_t)(*reinterpret_cast<const uint16_t*>(&kv_row1[c * 8])) << 16);
                const uint32_t b1 = (uint32_t)(*reinterpret_cast<const uint16_t*>(&kv_row2[c * 8])) |
                                    ((uint32_t)(*reinterpret_cast<const uint16_t*>(&kv_row3[c * 8])) << 16);
                hy4_mma16816_f32_f16(o_acc[c], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }

    auto compute_mult = [&](int r) -> float {
        if (h0 + r >= H) return 0.0f;
        const float mv = s_m[r];
        const float lv = s_l[r];
        if (sinks_opt != nullptr) {
            const float sv = sinks_opt[h0 + r];
            const float mn = fmaxf(mv, sv);
            const float corr = (mv == -INFINITY) ? 0.0f : expf(mv - mn);
            const float den = lv * corr + expf(sv - mn);
            return (den > 0.0f && mv > -INFINITY) ? (corr / den) : 0.0f;
        }
        return (lv > 0.0f) ? (1.0f / lv) : 0.0f;
    };

    const int r0 = g;
    const int r1 = g + 8;
    const float mult0 = compute_mult(r0);
    const float mult1 = compute_mult(r1);
    const int d_base = w * 128 + kp;
    if (h0 + r0 < H) {
        float* oh0 = out + ((size_t)t * H + h0 + r0) * 512 + d_base;
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            oh0[c * 8 + 0] = o_acc[c][0] * mult0;
            oh0[c * 8 + 1] = o_acc[c][1] * mult0;
        }
    }
    if (h0 + r1 < H) {
        float* oh1 = out + ((size_t)t * H + h0 + r1) * 512 + d_base;
        #pragma unroll
        for (int c = 0; c < 16; ++c) {
            oh1[c * 8 + 0] = o_acc[c][2] * mult1;
            oh1[c * 8 + 1] = o_acc[c][3] * mult1;
        }
    }
}

// ---------------------------------------------------------------------------
// 6. Pooled Indexer
// ---------------------------------------------------------------------------

template <typename PT>
__global__ void hy4_ix_pool_kernel(
    const PT* __restrict__ ivg,
    int ring_cap,
    int g0,
    int kpool,
    int dim,
    const float* __restrict__ ape,
    const float* __restrict__ w,
    float eps,
    int n_rot,
    float base,
    PT* __restrict__ pooled
) {
    __shared__ float sv[1024];
    const int g = g0 + (int)blockIdx.x;
    const int c = (int)threadIdx.x;
    const bool valid = c < dim;
    float lg[8];
    float mx = -INFINITY;

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (valid && k < kpool) {
            const int p = g * kpool + k;
            const int slot = (ring_cap > 0) ? (p % ring_cap) : p;
            float gv;
            if constexpr (sizeof(PT) == 2) {
                gv = __half2float(ivg[(size_t)slot * 2 * dim + dim + c]);
            } else {
                gv = ivg[(size_t)slot * 2 * dim + dim + c];
            }
            const float val = gv + ape[(size_t)k * dim + c];
            lg[k] = val;
            mx = fmaxf(mx, val);
        } else {
            lg[k] = -INFINITY;
        }
    }

    float den = 0.0f;
    float num = 0.0f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        if (valid && k < kpool) {
            const int p = g * kpool + k;
            const int slot = (ring_cap > 0) ? (p % ring_cap) : p;
            const float e = expf(lg[k] - mx);
            float kv;
            if constexpr (sizeof(PT) == 2) {
                kv = __half2float(ivg[(size_t)slot * 2 * dim + c]);
            } else {
                kv = ivg[(size_t)slot * 2 * dim + c];
            }
            den += e;
            num += e * kv;
        }
    }

    const float v = (valid && den > 0.0f) ? (num / den) : 0.0f;
    const float ss = hy4_block_reduce_sum(v * v);
    if (valid) sv[c] = v * rsqrtf(ss / (float)dim + eps) * w[c];
    __syncthreads();

    const int nope = dim - n_rot;
    if (valid && c >= nope && ((c - nope) & 1) == 0 && c + 1 < dim) {
        float a = sv[c];
        float b = sv[c + 1];
        hy4_rope_pair(a, b, (float)(g * kpool), (c - nope) >> 1, n_rot, base);
        sv[c] = a;
        sv[c + 1] = b;
    }
    __syncthreads();

    if (valid) {
        if constexpr (sizeof(PT) == 2) {
            pooled[(size_t)g * dim + c] = __float2half_rn(sv[c]);
        } else {
            pooled[(size_t)g * dim + c] = sv[c];
        }
    }
}

__device__ __forceinline__ void hy4_load_pooled8(const float* p, int i4, float4& a, float4& b) {
    a = reinterpret_cast<const float4*>(p)[i4];
    b = reinterpret_cast<const float4*>(p)[i4 + 1];
}

__device__ __forceinline__ void hy4_load_pooled8(const __half* p, int i4, float4& a, float4& b) {
    const uint4 u = reinterpret_cast<const uint4*>(p)[i4 >> 1];
    const float2 a0 = __half22float2(*reinterpret_cast<const __half2*>(&u.x));
    const float2 a1 = __half22float2(*reinterpret_cast<const __half2*>(&u.y));
    const float2 b0 = __half22float2(*reinterpret_cast<const __half2*>(&u.z));
    const float2 b1 = __half22float2(*reinterpret_cast<const __half2*>(&u.w));
    a = make_float4(a0.x, a0.y, a1.x, a1.y);
    b = make_float4(b0.x, b0.y, b1.x, b1.y);
}

template <typename PT>
__global__ void __launch_bounds__(256) hy4_ix_score_kernel(
    const float* __restrict__ q,
    const float* __restrict__ wt,
    int heads,
    int dim,
    const PT* __restrict__ pooled,
    const int32_t* __restrict__ pos,
    int kpool,
    int n_pool,
    float* __restrict__ score,
    size_t ld
) {
    extern __shared__ float sq[]; // [heads][dim] + [heads]
    const int t = (int)blockIdx.y;
    const int g = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    for (int i = (int)threadIdx.x; i < heads * dim; i += (int)blockDim.x) {
        sq[i] = q[(size_t)t * heads * dim + i];
    }
    for (int i = (int)threadIdx.x; i < heads; i += (int)blockDim.x) {
        sq[heads * dim + i] = wt[(size_t)t * heads + i];
    }
    __syncthreads();
    if (g >= n_pool) return;

    const int nv = pos[t] / kpool;
    float out = -INFINITY;
    if (g < nv) {
        float acc[16];
        #pragma unroll
        for (int h = 0; h < 16; ++h) acc[h] = 0.0f;
        const PT* pk = pooled + (size_t)g * dim;
        for (int i2 = 0; i2 < dim / 4; i2 += 2) {
            float4 k0, k1;
            hy4_load_pooled8(pk, i2, k0, k1);
            #pragma unroll
            for (int h = 0; h < 16; ++h) {
                if (h < heads) {
                    const float* qh0 = sq + h * dim + 4 * i2;
                    acc[h] = fmaf(qh0[0], k0.x, fmaf(qh0[1], k0.y, fmaf(qh0[2], k0.z, fmaf(qh0[3], k0.w, acc[h]))));
                    acc[h] = fmaf(qh0[4], k1.x, fmaf(qh0[5], k1.y, fmaf(qh0[6], k1.z, fmaf(qh0[7], k1.w, acc[h]))));
                }
            }
        }
        out = 0.0f;
        #pragma unroll
        for (int h = 0; h < 16; ++h) {
            if (h < heads) out = fmaf(fmaxf(acc[h], 0.0f), sq[heads * dim + h], out);
        }
    }
    score[(size_t)t * ld + g] = out;
}

__device__ __forceinline__ uint32_t hy4_ord_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__global__ void __launch_bounds__(1024) hy4_ix_select_kernel(
    const float* __restrict__ score,
    size_t ld,
    const int32_t* __restrict__ pos,
    int kpool,
    int top,
    int32_t* __restrict__ keys,
    int ldk,
    int32_t* __restrict__ nk
) {
    __shared__ int hist[256];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_need;
    __shared__ int s_gt[32], s_eq[32];

    const int t = (int)blockIdx.x;
    const int nth = (int)blockDim.x;
    const int tid = (int)threadIdx.x;
    const int p = pos[t];
    const int nv = p / kpool;
    const float* sc = score + (size_t)t * ld;
    int32_t* kt = keys + (size_t)t * ldk;
    const int nsel = min(nv, top);

    uint32_t thr = 0;
    int take_eq = 0;
    if (nv > top) {
        if (tid == 0) {
            s_prefix = 0;
            s_mask = 0;
            s_need = top;
        }
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int i = tid; i < 256; i += nth) hist[i] = 0;
            __syncthreads();
            const uint32_t pre = s_prefix;
            const uint32_t msk = s_mask;
            for (int g = tid; g < nv; g += nth) {
                const uint32_t kk = hy4_ord_key(sc[g]);
                if ((kk & msk) == pre) atomicAdd(&hist[(kk >> shift) & 255], 1);
            }
            __syncthreads();
            if (tid == 0) {
                int need = s_need;
                int b = 255;
                for (; b > 0; --b) {
                    if (hist[b] >= need) break;
                    need -= hist[b];
                }
                s_need = need;
                s_prefix = pre | ((uint32_t)b << shift);
                s_mask = msk | (255u << shift);
            }
            __syncthreads();
        }
        thr = s_prefix;
        take_eq = s_need;
    }

    const int per = (nv + nth - 1) / nth;
    const int b0 = min(nv, tid * per);
    const int b1 = min(nv, b0 + per);
    int cg = 0;
    int ce = 0;
    if (nv > top) {
        for (int g = b0; g < b1; ++g) {
            const uint32_t kk = hy4_ord_key(sc[g]);
            cg += (kk > thr) ? 1 : 0;
            ce += (kk == thr) ? 1 : 0;
        }
    } else {
        cg = b1 - b0;
    }

    const int lane = tid & 31;
    const int wp = tid >> 5;
    int ia = cg;
    int ie = ce;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int pa = __shfl_up_sync(0xffffffffu, ia, o);
        const int pe = __shfl_up_sync(0xffffffffu, ie, o);
        if (lane >= o) {
            ia += pa;
            ie += pe;
        }
    }
    if (lane == 31) {
        s_gt[wp] = ia;
        s_eq[wp] = ie;
    }
    __syncthreads();

    if (tid < 32) {
        const bool in = tid < (nth >> 5);
        const int ta = in ? s_gt[tid] : 0;
        const int te = in ? s_eq[tid] : 0;
        int wa = ta;
        int we = te;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int pa = __shfl_up_sync(0xffffffffu, wa, o);
            const int pe = __shfl_up_sync(0xffffffffu, we, o);
            if (lane >= o) {
                wa += pa;
                we += pe;
            }
        }
        if (in) {
            s_gt[tid] = wa - ta;
            s_eq[tid] = we - te;
        }
    }
    __syncthreads();

    int ng = s_gt[wp] + ia - cg;
    int ne = s_eq[wp] + ie - ce;
    for (int g = b0; g < b1; ++g) {
        int slot = -1;
        if (nv <= top) {
            slot = ng++;
        } else {
            const uint32_t kk = hy4_ord_key(sc[g]);
            if (kk > thr) {
                slot = ng + min(ne, take_eq);
                ++ng;
            } else if (kk == thr) {
                if (ne < take_eq) slot = ng + ne;
                ++ne;
            }
        }
        if (slot < 0) continue;
        for (int k = 0; k < kpool; ++k) {
            kt[slot * kpool + k] = g * kpool + k;
        }
    }
    __syncthreads();

    const int ntail = (p % kpool) + 1;
    for (int k = tid; k < ntail; k += nth) {
        kt[nsel * kpool + k] = nv * kpool + k;
    }
    if (tid == 0) nk[t] = nsel * kpool + ntail;
}

// ---------------------------------------------------------------------------
// 7. Sigmoid Router (up to 512 experts, top-k up to 32)
// ---------------------------------------------------------------------------

__device__ __forceinline__ bool hy4_rk_before(float av, int ai, float bv, int bi) {
    return (av > bv) || (av == bv && ai < bi);
}

__device__ __forceinline__ void hy4_rk_step(float& v, int& i, int j, bool up, int lane) {
    const float pv = __shfl_xor_sync(0xffffffffu, v, j);
    const int pi = __shfl_xor_sync(0xffffffffu, i, j);
    const bool lower = (lane & j) == 0;
    const bool pb = hy4_rk_before(pv, pi, v, i);
    if ((lower == up) ? pb : !pb) {
        v = pv;
        i = pi;
    }
}

__global__ void __launch_bounds__(512) hy4_router_bitonic_kernel(
    const float* __restrict__ logits,
    int n_exp,
    const float* __restrict__ bias_opt,
    int n_used,
    int norm,
    float scale,
    int32_t* __restrict__ ids,
    float* __restrict__ wout
) {
    __shared__ float lv[16][32];
    __shared__ int li[16][32];
    const int t = (int)blockIdx.x;
    const int e = (int)threadIdx.x;
    const int w = e >> 5;
    const int lane = e & 31;
    const float* lr = logits + (size_t)t * n_exp;

    float v = (e < n_exp) ? (hy4_sigmoidf(lr[e]) + (bias_opt ? bias_opt[e] : 0.0f)) : -INFINITY;
    int i = (e < n_exp) ? e : 0x7fffffff;

    for (int k = 2; k <= 32; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            hy4_rk_step(v, i, j, (lane & k) == 0, lane);
        }
    }
    lv[w][lane] = v;
    li[w][lane] = i;
    __syncthreads();

    const int n_keep = 32;
    for (int lists = 16; lists > 1; lists >>= 1) {
        float mv = -INFINITY;
        int mi = 0x7fffffff;
        if (w < (lists >> 1)) {
            mv = lv[2 * w][lane];
            mi = li[2 * w][lane];
            // Wait: bitonic merge of 2 sorted descending lists of length <= 32 via SMEM:
            // Compare half 2*w (descending) vs 2*w+1 (ascending): keep the better half of 32 items in one warp!
            const float ov = lv[2 * w + 1][31 - lane];
            const int oi = li[2 * w + 1][31 - lane];
            if (hy4_rk_before(ov, oi, mv, mi)) {
                mv = ov;
                mi = oi;
            }
            for (int j = 16; j > 0; j >>= 1) {
                hy4_rk_step(mv, mi, j, true, lane);
            }
        }
        __syncthreads();
        if (w < (lists >> 1)) {
            lv[w][lane] = mv;
            li[w][lane] = mi;
        }
        __syncthreads();
    }
    (void)n_keep;

    if (w == 0) {
        const int pick = (lane < n_used) ? li[0][lane] : 0;
        const float p = (lane < n_used) ? hy4_sigmoidf(lr[pick]) : 0.0f;
        const float sum = warp_reduce_sum(p);
        const float den = norm ? fmaxf(sum, 6.103515625e-5f) : 1.0f;
        if (lane < n_used) {
            ids[(size_t)t * n_used + lane] = pick;
            wout[(size_t)t * n_used + lane] = (p / den) * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Engram (F8_E4M3 rows + RMSNorm + Mix)
// ---------------------------------------------------------------------------

__global__ void hy4_engram_rows_kernel(
    const uint8_t* __restrict__ rows,
    const float* __restrict__ valid,
    int cols,
    int dim,
    const float* __restrict__ ln,
    float eps,
    const float* __restrict__ scale_opt,
    float* __restrict__ out
) {
    const int col = (int)blockIdx.x;
    const int t = (int)blockIdx.y;
    const int i = (int)threadIdx.x;
    float v = 0.0f;
    if (i < dim) {
        v = hy4_f8e4m3_to_f32(rows[((size_t)t * cols + col) * dim + i]);
        if (scale_opt != nullptr) v *= scale_opt[(size_t)t * cols + col];
    }
    const float ss = hy4_block_reduce_sum(v * v);
    const float rr = rsqrtf(ss / (float)dim + eps);
    if (i < dim) {
        out[((size_t)t * cols + col) * dim + i] = v * rr * ln[(size_t)col * dim + i] * valid[t];
    }
}

__global__ void __launch_bounds__(256) hy4_engram_mix_kernel(
    float* __restrict__ x,
    const float* __restrict__ kv,
    const float* __restrict__ kv_b,
    int hc,
    int E,
    const float* __restrict__ egate,
    const float* __restrict__ gamma,
    const float* __restrict__ hcs,
    const float* __restrict__ valid,
    float eps
) {
    __shared__ float s_gate[8];
    __shared__ float s_rv;
    const int t = (int)blockIdx.x;
    if (valid[t] == 0.0f) return;

    const float* kt = kv + (size_t)t * (hc + 1) * E;
    float* xt = x + (size_t)t * hc * E;

    for (int s = 0; s < hc; ++s) {
        float sx = 0.0f;
        float sk = 0.0f;
        float sd = 0.0f;
        for (int d = (int)threadIdx.x; d < E; d += (int)blockDim.x) {
            const float xv = xt[(size_t)s * E + d];
            const float kvv = kt[(size_t)s * E + d] + kv_b[(size_t)s * E + d];
            sx += xv * xv;
            sk += kvv * kvv;
            sd += xv * kvv * egate[(size_t)s * E + d];
        }
        sx = hy4_block_reduce_sum(sx);
        sk = hy4_block_reduce_sum(sk);
        sd = hy4_block_reduce_sum(sd);
        if (threadIdx.x == 0) {
            const float rx = rsqrtf(sx / (float)E + eps);
            const float rk = rsqrtf(sk / (float)E + eps);
            const float dot = sd * rx * rk * rsqrtf((float)E);
            const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
            s_gate[s] = hy4_sigmoidf((dot < 0.0f) ? -mag : mag);
        }
        __syncthreads();
    }

    float sv = 0.0f;
    for (int d = (int)threadIdx.x; d < E; d += (int)blockDim.x) {
        const float v = kt[(size_t)hc * E + d] + kv_b[(size_t)hc * E + d];
        sv += v * v;
    }
    sv = hy4_block_reduce_sum(sv);
    if (threadIdx.x == 0) s_rv = rsqrtf(sv / (float)E + eps);
    __syncthreads();

    for (int d = (int)threadIdx.x; d < E; d += (int)blockDim.x) {
        const float sig = (kt[(size_t)hc * E + d] + kv_b[(size_t)hc * E + d]) * s_rv * gamma[d];
        for (int s = 0; s < hc; ++s) {
            xt[(size_t)s * E + d] += sig * (s_gate[s] + hcs[(size_t)s * E + d]);
        }
    }
}

// ---------------------------------------------------------------------------
// 9. SigLIP Vision Tower Ops (F16 MMA ready, zero cuBLAS, hd <= 128 including hd=72)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256) hy4_vis_layernorm_f16_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int D,
    float eps,
    uint16_t* __restrict__ y16
) {
    const float* r = x + (size_t)blockIdx.x * D;
    float s = 0.0f;
    for (int i = (int)threadIdx.x; i < D; i += (int)blockDim.x) s += r[i];
    const float mean = hy4_block_reduce_sum(s) / (float)D;
    float q = 0.0f;
    for (int i = (int)threadIdx.x; i < D; i += (int)blockDim.x) {
        const float d = r[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(hy4_block_reduce_sum(q) / (float)D + eps);
    for (int i = (int)threadIdx.x; i < D; i += (int)blockDim.x) {
        y16[(size_t)blockIdx.x * D + i] = f32_to_fp16_bits((r[i] - mean) * inv * w[i] + b[i]);
    }
}

__global__ void __launch_bounds__(256) hy4_vis_rmsnorm_inplace_kernel(
    float* __restrict__ x,
    const float* __restrict__ w,
    int D,
    float eps
) {
    float* r = x + (size_t)blockIdx.x * D;
    float q = 0.0f;
    for (int i = (int)threadIdx.x; i < D; i += (int)blockDim.x) q += r[i] * r[i];
    const float inv = rsqrtf(hy4_block_reduce_sum(q) / (float)D + eps);
    for (int i = (int)threadIdx.x; i < D; i += (int)blockDim.x) r[i] = r[i] * inv * w[i];
}

__global__ void hy4_vis_add_bias_kernel(
    float* __restrict__ y,
    const float* __restrict__ bias,
    size_t n,
    int M
) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        y[i] += bias[i % (size_t)M];
    }
}

__global__ void hy4_vis_bias_act_f16_kernel(
    const float* __restrict__ y,
    const float* __restrict__ bias,
    size_t n,
    int M,
    int act,
    uint16_t* __restrict__ y16
) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float v = y[i] + bias[i % (size_t)M];
        if (act == 1) v = hy4_gelu_tanh(v);
        else if (act == 2) v = hy4_gelu_erf(v);
        y16[i] = f32_to_fp16_bits(v);
    }
}

__global__ void hy4_vis_residual_bias_kernel(
    float* __restrict__ h,
    const float* __restrict__ t,
    const float* __restrict__ bias,
    size_t n,
    int M
) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        h[i] += t[i] + bias[i % (size_t)M];
    }
}

__global__ void __launch_bounds__(256) hy4_vis_merge_gather_f16_kernel(
    const float* __restrict__ xn,
    int gw,
    int D,
    int ow,
    uint16_t* __restrict__ m16
) {
    const int b = (int)blockIdx.x;
    const int by = b / ow;
    const int bx = b % ow;
    uint16_t* dst = m16 + (size_t)b * 4 * D;
    const int total = 4 * D;
    for (int i = (int)threadIdx.x; i < total; i += (int)blockDim.x) {
        const int ic = i >> 2;
        const int ky = (i >> 1) & 1;
        const int kx = i & 1;
        dst[i] = f32_to_fp16_bits(xn[((size_t)(2 * by + ky) * gw + 2 * bx + kx) * D + ic]);
    }
}

__global__ void __launch_bounds__(256) hy4_vis_place_rows_kernel(
    const float* __restrict__ y,
    const float* __restrict__ bias,
    const float* __restrict__ nl,
    int ow,
    int O,
    float* __restrict__ out
) {
    const int r = (int)blockIdx.x;
    const int by = r / (ow + 1);
    const int bx = r % (ow + 1);
    float* dst = out + (size_t)r * O;
    if (bx == ow) {
        for (int c = (int)threadIdx.x; c < O; c += (int)blockDim.x) dst[c] = nl[c];
    } else {
        const float* src = y + ((size_t)by * ow + bx) * O;
        for (int c = (int)threadIdx.x; c < O; c += (int)blockDim.x) dst[c] = src[c] + bias[c];
    }
}

// Non-causal Vision FlashAttention (supports arbitrary hd <= 128 including hd = 72)
// Grid: (ceil(N / 4), heads), Block: 128 (4 warps, 1 warp per query row)
// Dynamic SMEM: 2 * 32 * hd_pad * sizeof(float)
__global__ void __launch_bounds__(128) hy4_vis_flash_attn_kernel(
    const float* __restrict__ qkv,
    int N,
    int heads,
    int hd,
    int hd_pad,
    float scale,
    float* __restrict__ out
) {
    extern __shared__ float smem[];
    float* sK = smem;
    float* sV = smem + 32 * hd_pad;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int q_idx = (int)blockIdx.x * 4 + warp;
    const int h = (int)blockIdx.y;
    const int D = heads * hd;

    float q_reg[4];
    float acc[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * 32;
        q_reg[i] = (q_idx < N && d < hd) ? qkv[(size_t)q_idx * 3 * D + h * hd + d] : 0.0f;
        acc[i] = 0.0f;
    }

    float m = -INFINITY;
    float l = 0.0f;

    for (int k0 = 0; k0 < N; k0 += 32) {
        const int nt = min(32, N - k0);
        for (int kr = warp; kr < nt; kr += 4) {
            const float* kr_k = qkv + (size_t)(k0 + kr) * 3 * D + D + h * hd;
            const float* kr_v = kr_k + D;
            for (int d = lane; d < hd; d += 32) {
                sK[kr * hd_pad + d] = kr_k[d];
                sV[kr * hd_pad + d] = kr_v[d];
            }
        }
        __syncthreads();

        if (q_idx < N) {
            for (int j = 0; j < nt; ++j) {
                const float* kj = sK + j * hd_pad;
                const float* vj = sV + j * hd_pad;
                float s = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int d = lane + i * 32;
                    if (d < hd) s = fmaf(q_reg[i], kj[d], s);
                }
                s = warp_reduce_sum(s) * scale;
                const float mn = fmaxf(m, s);
                if (mn == m) {
                    const float pj = expf(s - mn);
                    l += pj;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int d = lane + i * 32;
                        if (d < hd) acc[i] = fmaf(pj, vj[d], acc[i]);
                    }
                } else {
                    const float corr = (m == -INFINITY) ? 0.0f : expf(m - mn);
                    const float pj = expf(s - mn);
                    l = l * corr + pj;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int d = lane + i * 32;
                        if (d < hd) acc[i] = fmaf(pj, vj[d], acc[i] * corr);
                    }
                }
                m = mn;
            }
        }
        __syncthreads();
    }

    if (q_idx < N) {
        const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
        float* qo = out + (size_t)q_idx * D + h * hd;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int d = lane + i * 32;
            if (d < hd) qo[d] = acc[i] * inv;
        }
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Extern "C" Launchers
// ---------------------------------------------------------------------------

extern "C" void launch_hy4_expand_streams(
    const float* emb,
    int T,
    int hc,
    int E,
    float* x_out,
    cudaStream_t stream
) {
    const size_t total = (size_t)T * hc * E;
    if (total == 0) return;
    const unsigned blocks = (unsigned)std::min<size_t>(4096, (total + 255) / 256);
    hy4_expand_streams_kernel<<<blocks, 256, 0, stream>>>(emb, hc, E, total, x_out);
}

extern "C" void launch_hy4_hc_pre(
    const float* x,
    int T,
    int hc,
    int E,
    const float* fn,
    int fn_by_el,
    const float* scale,
    const float* base,
    float eps,
    float hc_eps,
    float mag,
    const float* norm_w_opt,
    float norm_eps,
    int is_head,
    float* part_buf,
    float* cur,
    float* post_opt,
    cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    const int nm = is_head ? hc : 2 * hc;
    dim3 g_part(kHy4HcBlocks, (unsigned)T);
    if (fn_by_el) {
        hy4_hc_part_kernel<true><<<g_part, 256, 0, stream>>>(x, hc * E, nm, fn, part_buf);
    } else {
        hy4_hc_part_kernel<false><<<g_part, 256, 0, stream>>>(x, hc * E, nm, fn, part_buf);
    }

    if (is_head) {
        if (E <= 4096) {
            hy4_hc_fin_kernel<true, 8><<<T, 512, 0, stream>>>(
                x, hc, E, part_buf, scale, base, eps, hc_eps, 1.0f, norm_w_opt, norm_eps, cur, nullptr);
        } else {
            hy4_hc_fin_kernel<true, 16><<<T, 512, 0, stream>>>(
                x, hc, E, part_buf, scale, base, eps, hc_eps, 1.0f, norm_w_opt, norm_eps, cur, nullptr);
        }
    } else {
        if (E <= 4096) {
            hy4_hc_fin_kernel<false, 8><<<T, 512, 0, stream>>>(
                x, hc, E, part_buf, scale, base, eps, hc_eps, mag, norm_w_opt, norm_eps, cur, post_opt);
        } else {
            hy4_hc_fin_kernel<false, 16><<<T, 512, 0, stream>>>(
                x, hc, E, part_buf, scale, base, eps, hc_eps, mag, norm_w_opt, norm_eps, cur, post_opt);
        }
    }
}

extern "C" void launch_hy4_hc_post_norm(
    float* x,
    int T,
    int hc,
    int E,
    const float* y,
    const float* post,
    const float* w_opt,
    float eps,
    float* y_norm_opt,
    cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    if (w_opt != nullptr) {
        dim3 grid(kHy4PostBlocks, (unsigned)T);
        hy4_hc_post_norm_kernel<<<grid, 256, 0, stream>>>(x, hc, E, y, post, w_opt, eps, y_norm_opt);
    } else {
        const size_t total = (size_t)T * hc * E;
        const unsigned blocks = (unsigned)std::min<size_t>(65535, (total + 255) / 256);
        hy4_hc_post_add_kernel<<<blocks, 256, 0, stream>>>(x, hc, E, y, post, total);
    }
}

extern "C" void launch_hy4_rope(
    float* x,
    int T,
    int nh,
    size_t ld_t,
    size_t ld_h,
    int off,
    int n_rot,
    const int32_t* pos,
    float base,
    cudaStream_t stream
) {
    const size_t total = (size_t)T * nh * (n_rot >> 1);
    if (total == 0) return;
    hy4_rope_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        x, T, nh, ld_t, ld_h, off, n_rot, pos, base);
}

extern "C" void launch_hy4_sigmoid_mul(
    float* x,
    const float* g,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    const unsigned blocks = (unsigned)std::min<size_t>(65535, (n + 255) / 256);
    hy4_sigmoid_mul_kernel<<<blocks, 256, 0, stream>>>(x, g, n);
}

extern "C" void launch_hy4_swiglu_f32(
    const float* gate,
    const float* up,
    size_t ld_gu,
    int rows,
    int n,
    float clamp_lim,
    float* out,
    size_t ld_out,
    cudaStream_t stream
) {
    const size_t total = (size_t)rows * n;
    if (total == 0) return;
    hy4_swiglu_f32_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        gate, up, ld_gu, rows, n, clamp_lim, out, ld_out);
}

extern "C" void launch_hy4_store_rows(
    const float* src,
    int T,
    int dim,
    void* ring,
    int ring_cap,
    const int32_t* pos,
    int is_f16,
    cudaStream_t stream
) {
    const size_t total = (size_t)T * dim;
    if (total == 0) return;
    const unsigned blocks = (unsigned)((total + 255) / 256);
    if (is_f16) {
        hy4_store_rows_kernel<__half><<<blocks, 256, 0, stream>>>(
            src, T, dim, reinterpret_cast<__half*>(ring), ring_cap, pos);
    } else {
        hy4_store_rows_kernel<float><<<blocks, 256, 0, stream>>>(
            src, T, dim, reinterpret_cast<float*>(ring), ring_cap, pos);
    }
}

extern "C" void launch_hy4_gemv_f32(
    const float* W,
    int N,
    int K,
    const float* X,
    int M,
    float* Y,
    float alpha,
    cudaStream_t stream
) {
    dispatch_gemv_dense<float>(W, N, K, X, M, Y, alpha, stream);
}

extern "C" void launch_hy4_gemv_bf16(
    const uint16_t* W,
    int N,
    int K,
    const float* X,
    int M,
    float* Y,
    float alpha,
    cudaStream_t stream
) {
    dispatch_gemv_dense<uint16_t>(W, N, K, X, M, Y, alpha, stream);
}

extern "C" void launch_hy4_quantize_heads_q8_1(
    const float* X,
    void* X_q8_1,
    int T,
    int H,
    int K,
    size_t ld_t,
    size_t ld_h,
    cudaStream_t stream
) {
    const int nb = K >> 5;
    if (T <= 0 || H <= 0 || nb <= 0) return;
    dim3 grid((nb + 3) / 4, H, T);
    hy4_quantize_heads_q8_1_kernel<<<grid, 128, 0, stream>>>(
        X, reinterpret_cast<block_q8_1*>(X_q8_1), T, H, K, ld_t, ld_h);
}

extern "C" void launch_hy4_batched_heads_q8(
    const void* W_q8_0,
    const void* X_q8_1,
    float* Y,
    int T,
    int H,
    int N,
    int K,
    size_t x_tok_stride_blk,
    size_t y_tok_stride,
    size_t y_head_stride,
    cudaStream_t stream
) {
    if (T <= 0 || H <= 0 || N <= 0 || K <= 0) return;
    dim3 grid((N + 15) / 16, H, T);
    dim3 block(32, 4);
    hy4_batched_heads_q8_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const block_q8_0*>(W_q8_0),
        reinterpret_cast<const block_q8_1*>(X_q8_1),
        Y, T, H, N, K, x_tok_stride_blk, y_tok_stride, y_head_stride);
}

extern "C" void launch_hy4_batched_heads_q8_mma(
    const void* W_q8_0,
    const float* X_f32,
    float* Y,
    int T,
    int H,
    int N,
    int K,
    size_t x_tok_stride,
    size_t x_head_stride,
    size_t y_tok_stride,
    size_t y_head_stride,
    cudaStream_t stream
) {
    if (T <= 0 || H <= 0 || N <= 0 || K <= 0) return;
    dim3 grid((N + 63) / 64, (T + 63) / 64, H);
    hy4_batched_heads_q8_mma_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const block_q8_0*>(W_q8_0),
        X_f32,
        Y,
        T,
        H,
        N,
        K,
        x_tok_stride,
        x_head_stride,
        y_tok_stride,
        y_head_stride
    );
}

extern "C" void launch_hy4_kv_store(
    const float* kv,
    int T,
    int kv_lora,
    int n_rot,
    const float* w,
    float eps,
    const int32_t* pos,
    float base,
    void* cache,
    int ring_cap,
    int is_q8,
    int rb,
    cudaStream_t stream
) {
    if (T <= 0) return;
    if (is_q8) {
        hy4_kv_store_q8_kernel<<<T, 256, 0, stream>>>(
            kv, kv_lora, n_rot, w, eps, pos, base, reinterpret_cast<uint8_t*>(cache), ring_cap, rb);
    } else {
        hy4_kv_store_f16_kernel<<<T, 256, 0, stream>>>(
            kv, kv_lora, n_rot, w, eps, pos, base, reinterpret_cast<__half*>(cache), ring_cap);
    }
}

extern "C" void launch_hy4_mla_decode(
    const float* q,
    int T,
    int H,
    const void* cache,
    int ring_cap,
    const int32_t* keys_opt,
    int ldk,
    const int32_t* lo_opt,
    const int32_t* nk,
    int max_keys,
    float scale,
    const float* sinks_opt,
    float* ws,
    float* out,
    int is_q8,
    int rb,
    cudaStream_t stream
) {
    if (T <= 0 || H <= 0) return;
    const int kc = (T <= 4) ? 32 : 128;
    const int nchunk = max(1, (max_keys + kc - 1) / kc);
    dim3 grid(nchunk, (H + 7) / 8, T);
    if (is_q8) {
        hy4_mla_decode_split_kernel<true><<<grid, 256, 0, stream>>>(
            q, H, cache, ring_cap, keys_opt, ldk, lo_opt, nk, scale, nchunk, kc, ws, rb);
    } else {
        hy4_mla_decode_split_kernel<false><<<grid, 256, 0, stream>>>(
            q, H, cache, ring_cap, keys_opt, ldk, lo_opt, nk, scale, nchunk, kc, ws, rb);
    }
    hy4_mla_combine_kernel<<<dim3(H, T), 512, 0, stream>>>(
        H, nchunk, kc, nk, sinks_opt, ws, out);
}

extern "C" void launch_hy4_mla_prefill(
    const float* q,
    int T,
    int H,
    const void* cache,
    int ring_cap,
    const int32_t* keys_opt,
    int ldk,
    const int32_t* lo_opt,
    const int32_t* nk,
    float scale,
    const float* sinks_opt,
    float* out,
    int is_q8,
    int rb,
    cudaStream_t stream
) {
    if (T <= 0 || H <= 0) return;
    if (H >= 16) {
        static std::atomic<uint32_t> attr_mask_f16{0};
        static std::atomic<uint32_t> attr_mask_q8{0};
        int dev = 0;
        cudaGetDevice(&dev);
        const uint32_t bit = 1u << (dev & 31);
        dim3 grid((H + 15) / 16, T);
        if (is_q8) {
            if ((attr_mask_q8.load(std::memory_order_acquire) & bit) == 0) {
                cudaFuncSetAttribute(
                    hy4_mla_prefill_mma_kernel<true>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kHy4MlaMmaSmemBytes
                );
                attr_mask_q8.fetch_or(bit, std::memory_order_release);
            }
            hy4_mla_prefill_mma_kernel<true><<<grid, 128, kHy4MlaMmaSmemBytes, stream>>>(
                q, H, cache, ring_cap, keys_opt, ldk, lo_opt, nk, scale, sinks_opt, out, rb);
        } else {
            if ((attr_mask_f16.load(std::memory_order_acquire) & bit) == 0) {
                cudaFuncSetAttribute(
                    hy4_mla_prefill_mma_kernel<false>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    kHy4MlaMmaSmemBytes
                );
                attr_mask_f16.fetch_or(bit, std::memory_order_release);
            }
            hy4_mla_prefill_mma_kernel<false><<<grid, 128, kHy4MlaMmaSmemBytes, stream>>>(
                q, H, cache, ring_cap, keys_opt, ldk, lo_opt, nk, scale, sinks_opt, out, rb);
        }
        return;
    }
    dim3 grid((H + 7) / 8, T);
    if (is_q8) {
        hy4_mla_prefill_kernel<true><<<grid, 256, 0, stream>>>(
            q, H, cache, ring_cap, keys_opt, ldk, lo_opt, nk, scale, sinks_opt, out, rb);
    } else {
        hy4_mla_prefill_kernel<false><<<grid, 256, 0, stream>>>(
            q, H, cache, ring_cap, keys_opt, ldk, lo_opt, nk, scale, sinks_opt, out, rb);
    }
}

extern "C" void launch_hy4_ix_pool(
    const void* ivg,
    int ring_cap,
    int g0,
    int n_new,
    int kpool,
    int dim,
    const float* ape,
    const float* w,
    float eps,
    int n_rot,
    float base,
    void* pooled,
    int is_f16,
    cudaStream_t stream
) {
    if (n_new <= 0 || dim <= 0) return;
    const int nth = std::min(1024, ((dim + 31) / 32) * 32);
    if (is_f16) {
        hy4_ix_pool_kernel<__half><<<n_new, nth, 0, stream>>>(
            reinterpret_cast<const __half*>(ivg), ring_cap, g0, kpool, dim, ape, w, eps, n_rot, base,
            reinterpret_cast<__half*>(pooled));
    } else {
        hy4_ix_pool_kernel<float><<<n_new, nth, 0, stream>>>(
            reinterpret_cast<const float*>(ivg), ring_cap, g0, kpool, dim, ape, w, eps, n_rot, base,
            reinterpret_cast<float*>(pooled));
    }
}

extern "C" void launch_hy4_ix_score(
    const float* q,
    const float* wt,
    int T,
    int heads,
    int dim,
    const void* pooled,
    const int32_t* pos,
    int kpool,
    int n_pool,
    float* score,
    size_t ld,
    int is_f16,
    cudaStream_t stream
) {
    if (T <= 0 || n_pool <= 0) return;
    dim3 grid((n_pool + 255) / 256, T);
    const size_t sh = (size_t)(heads * dim + heads) * sizeof(float);
    if (is_f16) {
        hy4_ix_score_kernel<__half><<<grid, 256, sh, stream>>>(
            q, wt, heads, dim, reinterpret_cast<const __half*>(pooled), pos, kpool, n_pool, score, ld);
    } else {
        hy4_ix_score_kernel<float><<<grid, 256, sh, stream>>>(
            q, wt, heads, dim, reinterpret_cast<const float*>(pooled), pos, kpool, n_pool, score, ld);
    }
}

extern "C" void launch_hy4_ix_select(
    const float* score,
    size_t ld,
    int T,
    const int32_t* pos,
    int kpool,
    int top,
    int32_t* keys,
    int ldk,
    int32_t* nk,
    cudaStream_t stream
) {
    if (T <= 0) return;
    hy4_ix_select_kernel<<<T, 1024, 0, stream>>>(score, ld, pos, kpool, top, keys, ldk, nk);
}

extern "C" void launch_hy4_router_sigmoid_topk(
    const float* logits,
    int T,
    int n_exp,
    const float* bias_opt,
    int n_used,
    int norm,
    float scale,
    int32_t* ids,
    float* wout,
    cudaStream_t stream
) {
    if (T <= 0 || n_exp <= 0) return;
    hy4_router_bitonic_kernel<<<T, 512, 0, stream>>>(
        logits, n_exp, bias_opt, n_used, norm, scale, ids, wout);
}

extern "C" void launch_hy4_engram_rows(
    const uint8_t* rows,
    const float* valid,
    int T,
    int cols,
    int dim,
    const float* ln,
    float eps,
    const float* scale_opt,
    float* out,
    cudaStream_t stream
) {
    if (T <= 0 || cols <= 0 || dim <= 0) return;
    const int nth = std::min(1024, ((dim + 31) / 32) * 32);
    hy4_engram_rows_kernel<<<dim3(cols, T), nth, 0, stream>>>(
        rows, valid, cols, dim, ln, eps, scale_opt, out);
}

extern "C" void launch_hy4_engram_mix(
    float* x,
    const float* kv,
    const float* kv_b,
    int T,
    int hc,
    int E,
    const float* egate,
    const float* gamma,
    const float* hcs,
    const float* valid,
    float eps,
    cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    hy4_engram_mix_kernel<<<T, 256, 0, stream>>>(
        x, kv, kv_b, hc, E, egate, gamma, hcs, valid, eps);
}

extern "C" void launch_hy4_vis_layernorm_f16(
    const float* x,
    const float* w,
    const float* b,
    int N,
    int D,
    float eps,
    uint16_t* y16,
    cudaStream_t stream
) {
    if (N <= 0 || D <= 0) return;
    hy4_vis_layernorm_f16_kernel<<<N, 256, 0, stream>>>(x, w, b, D, eps, y16);
}

extern "C" void launch_hy4_vis_rmsnorm_inplace(
    float* x,
    const float* w,
    int N,
    int D,
    float eps,
    cudaStream_t stream
) {
    if (N <= 0 || D <= 0) return;
    hy4_vis_rmsnorm_inplace_kernel<<<N, 256, 0, stream>>>(x, w, D, eps);
}

extern "C" void launch_hy4_vis_add_bias(
    float* y,
    const float* bias,
    size_t n,
    int M,
    cudaStream_t stream
) {
    if (n == 0 || M <= 0) return;
    const unsigned blocks = (unsigned)std::min<size_t>(65535, (n + 255) / 256);
    hy4_vis_add_bias_kernel<<<blocks, 256, 0, stream>>>(y, bias, n, M);
}

extern "C" void launch_hy4_vis_bias_act_f16(
    const float* y,
    const float* bias,
    size_t n,
    int M,
    int act,
    uint16_t* y16,
    cudaStream_t stream
) {
    if (n == 0 || M <= 0) return;
    const unsigned blocks = (unsigned)std::min<size_t>(65535, (n + 255) / 256);
    hy4_vis_bias_act_f16_kernel<<<blocks, 256, 0, stream>>>(y, bias, n, M, act, y16);
}

extern "C" void launch_hy4_vis_residual_bias(
    float* h,
    const float* t,
    const float* bias,
    size_t n,
    int M,
    cudaStream_t stream
) {
    if (n == 0 || M <= 0) return;
    const unsigned blocks = (unsigned)std::min<size_t>(65535, (n + 255) / 256);
    hy4_vis_residual_bias_kernel<<<blocks, 256, 0, stream>>>(h, t, bias, n, M);
}

extern "C" void launch_hy4_vis_merge_gather_f16(
    const float* xn,
    int gw,
    int D,
    int ow,
    int oh,
    uint16_t* m16,
    cudaStream_t stream
) {
    const int nb = ow * oh;
    if (nb <= 0 || D <= 0) return;
    hy4_vis_merge_gather_f16_kernel<<<nb, 256, 0, stream>>>(xn, gw, D, ow, m16);
}

extern "C" void launch_hy4_vis_place_rows(
    const float* y,
    const float* bias,
    const float* nl,
    int ow,
    int oh,
    int O,
    float* out,
    cudaStream_t stream
) {
    const int rows = (ow + 1) * oh;
    if (rows <= 0 || O <= 0) return;
    hy4_vis_place_rows_kernel<<<rows, 256, 0, stream>>>(y, bias, nl, ow, O, out);
}

extern "C" void launch_hy4_vis_flash_attn(
    const float* qkv,
    int N,
    int heads,
    int hd,
    float scale,
    float* out,
    cudaStream_t stream
) {
    if (N <= 0 || heads <= 0 || hd <= 0) return;
    const int hd_pad = (hd + 3) & ~3;
    dim3 grid((N + 3) / 4, heads);
    const size_t sh = (size_t)2 * 32 * hd_pad * sizeof(float);
    hy4_vis_flash_attn_kernel<<<grid, 128, sh, stream>>>(qkv, N, heads, hd, hd_pad, scale, out);
}

// ---------------------------------------------------------------------------
// 10. Hy4 Engine Helper Kernels (copy_cols, add_f32, rmsnorm_strided, top1_prob, build_moe_plan)
// ---------------------------------------------------------------------------

__global__ void __launch_bounds__(256) hy4_copy_cols_kernel(
    const float* __restrict__ src,
    size_t ld_src,
    float* __restrict__ dst,
    size_t ld_dst,
    size_t total,
    int width
) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / (size_t)width;
        const size_t c = i % (size_t)width;
        dst[r * ld_dst + c] = src[r * ld_src + c];
    }
}

extern "C" void launch_hy4_copy_cols(
    const float* src,
    size_t ld_src,
    float* dst,
    size_t ld_dst,
    int rows,
    int width,
    cudaStream_t stream
) {
    if (rows <= 0 || width <= 0) return;
    const size_t total = (size_t)rows * (size_t)width;
    const unsigned blocks = (unsigned)std::min<size_t>(65535, (total + 255) / 256);
    hy4_copy_cols_kernel<<<blocks, 256, 0, stream>>>(src, ld_src, dst, ld_dst, total, width);
}

__global__ void __launch_bounds__(256) hy4_add_f32_vec4_kernel(
    float4* __restrict__ dst,
    const float4* __restrict__ src,
    size_t n4
) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        float4 d = dst[i];
        const float4 s = src[i];
        d.x += s.x;
        d.y += s.y;
        d.z += s.z;
        d.w += s.w;
        dst[i] = d;
    }
}

__global__ void __launch_bounds__(256) hy4_add_f32_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    size_t n
) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        dst[i] += src[i];
    }
}

extern "C" void launch_hy4_add_f32(
    float* dst,
    const float* src,
    size_t n,
    cudaStream_t stream
) {
    if (n == 0) return;
    if ((n & 3u) == 0u &&
        (reinterpret_cast<uintptr_t>(dst) & 15u) == 0u &&
        (reinterpret_cast<uintptr_t>(src) & 15u) == 0u) {
        const size_t n4 = n >> 2;
        const unsigned blocks = (unsigned)std::min<size_t>(65535, (n4 + 255) / 256);
        hy4_add_f32_vec4_kernel<<<blocks, 256, 0, stream>>>(
            reinterpret_cast<float4*>(dst), reinterpret_cast<const float4*>(src), n4);
        return;
    }
    const unsigned blocks = (unsigned)std::min<size_t>(65535, (n + 255) / 256);
    hy4_add_f32_kernel<<<blocks, 256, 0, stream>>>(dst, src, n);
}

__global__ void __launch_bounds__(256) hy4_rmsnorm_strided_kernel(
    const float* __restrict__ x,
    size_t ld_x,
    const float* __restrict__ w,
    float* __restrict__ y,
    size_t ld_y,
    int dim,
    float eps
) {
    const float* xr = x + (size_t)blockIdx.x * ld_x;
    float* yr = y + (size_t)blockIdx.x * ld_y;
    float ss = 0.0f;
    for (int i = (int)threadIdx.x; i < dim; i += 256) {
        const float v = xr[i];
        ss += v * v;
    }
    ss = hy4_block_reduce_sum(ss);
    const float inv = rsqrtf(ss / (float)dim + eps);
    for (int i = (int)threadIdx.x; i < dim; i += 256) {
        yr[i] = xr[i] * inv * w[i];
    }
}

extern "C" void launch_hy4_rmsnorm_strided(
    const float* x,
    size_t ld_x,
    const float* w,
    float* y,
    size_t ld_y,
    int dim,
    int rows,
    float eps,
    cudaStream_t stream
) {
    if (rows <= 0 || dim <= 0) return;
    hy4_rmsnorm_strided_kernel<<<rows, 256, 0, stream>>>(x, ld_x, w, y, ld_y, dim, eps);
}

__device__ __forceinline__ void hy4_combine_lse(
    float& m0,
    int& i0,
    float& s0,
    float m1,
    int i1,
    float s1
) {
    if (s1 <= 0.0f) return;
    if (s0 <= 0.0f) {
        m0 = m1;
        i0 = i1;
        s0 = s1;
        return;
    }
    const bool b_better = (m1 > m0) || (m1 == m0 && i1 < i0);
    if (b_better) {
        s0 = s1 + s0 * expf(m0 - m1);
        m0 = m1;
        i0 = i1;
    } else {
        s0 = s0 + s1 * expf(m1 - m0);
    }
}

__global__ void __launch_bounds__(256) hy4_top1_stage1_kernel(
    const float* __restrict__ logits,
    int n_vocab,
    float* __restrict__ scratch_and_out
) {
    __shared__ float s_m[8];
    __shared__ int s_i[8];
    __shared__ float s_s[8];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int stride = 128 * 256;

    float m = -INFINITY;
    int best = 0x7fffffff;
    float s = 0.0f;
    for (int idx = (int)blockIdx.x * 256 + tid; idx < n_vocab; idx += stride) {
        const float v = logits[idx];
        if (!isnan(v)) {
            if (s == 0.0f) {
                m = v;
                best = idx;
                s = 1.0f;
            } else if (v > m || (v == m && idx < best)) {
                s = s * expf(m - v) + 1.0f;
                m = v;
                best = idx;
            } else {
                s += expf(v - m);
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float pm = __shfl_xor_sync(0xffffffffu, m, off);
        const int pi = __shfl_xor_sync(0xffffffffu, best, off);
        const float ps = __shfl_xor_sync(0xffffffffu, s, off);
        if ((lane & off) == 0) {
            hy4_combine_lse(m, best, s, pm, pi, ps);
        }
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_i[warp] = best;
        s_s[warp] = s;
    }
    __syncthreads();

    if (tid == 0) {
        float fm = -INFINITY;
        int fi = 0x7fffffff;
        float fs = 0.0f;
        for (int w = 0; w < 8; ++w) {
            hy4_combine_lse(fm, fi, fs, s_m[w], s_i[w], s_s[w]);
        }
        const int base = 2 + (int)blockIdx.x * 3;
        scratch_and_out[base + 0] = fm;
        scratch_and_out[base + 1] = __int_as_float(fi);
        scratch_and_out[base + 2] = fs;
    }
}

__global__ void __launch_bounds__(128) hy4_top1_stage2_kernel(
    float* __restrict__ scratch_and_out
) {
    __shared__ float s_m[4];
    __shared__ int s_i[4];
    __shared__ float s_s[4];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int base = 2 + tid * 3;
    float m = scratch_and_out[base + 0];
    int best = __float_as_int(scratch_and_out[base + 1]);
    float s = scratch_and_out[base + 2];

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float pm = __shfl_xor_sync(0xffffffffu, m, off);
        const int pi = __shfl_xor_sync(0xffffffffu, best, off);
        const float ps = __shfl_xor_sync(0xffffffffu, s, off);
        if ((lane & off) == 0) {
            hy4_combine_lse(m, best, s, pm, pi, ps);
        }
    }
    if (lane == 0) {
        s_m[warp] = m;
        s_i[warp] = best;
        s_s[warp] = s;
    }
    __syncthreads();

    if (tid == 0) {
        float fm = -INFINITY;
        int fi = 0x7fffffff;
        float fs = 0.0f;
        for (int w = 0; w < 4; ++w) {
            hy4_combine_lse(fm, fi, fs, s_m[w], s_i[w], s_s[w]);
        }
        if (fs <= 0.0f) {
            fi = 0;
        }
        scratch_and_out[0] = __int_as_float(fi);
        scratch_and_out[1] = (fs > 0.0f) ? (1.0f / fs) : 0.0f;
    }
}

extern "C" void launch_hy4_top1_prob(
    const float* logits,
    int n_vocab,
    float* scratch_and_out,
    cudaStream_t stream
) {
    if (n_vocab <= 0) return;
    hy4_top1_stage1_kernel<<<128, 256, 0, stream>>>(logits, n_vocab, scratch_and_out);
    hy4_top1_stage2_kernel<<<1, 128, 0, stream>>>(scratch_and_out);
}

__global__ void __launch_bounds__(256) hy4_build_moe_plan_kernel(
    const int32_t* __restrict__ ids,
    const int32_t* __restrict__ slot_d,
    uint64_t vram_base,
    size_t blob_bytes,
    int count,
    uint64_t* __restrict__ plan_ptrs_out
) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i >= count) return;
    const int s = slot_d[ids[i]];
    plan_ptrs_out[i] = (s >= 0) ? (vram_base + (uint64_t)s * (uint64_t)blob_bytes) : 0ULL;
}

extern "C" void launch_hy4_build_moe_plan(
    const int32_t* ids,
    const int32_t* slot_d,
    uint64_t vram_base,
    size_t blob_bytes,
    int count,
    uint64_t* plan_ptrs_out,
    cudaStream_t stream
) {
    if (count <= 0) return;
    hy4_build_moe_plan_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
        ids, slot_d, vram_base, blob_bytes, count, plan_ptrs_out);
}
