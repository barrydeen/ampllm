#include "common/sm86_common.cuh"
#include <cuda_fp8.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>

namespace {

__device__ __forceinline__ uint16_t f2bf(float f) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(f));
}

__device__ __forceinline__ float bf2f(uint16_t b) {
    return bf16_bits_to_f32(b);
}

__device__ __forceinline__ float swoai_val(float g, float u, float alpha, float limit) {
    if (limit > 0.0f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    return (g / (1.0f + expf(-alpha * g))) * (u + 1.0f);
}

__device__ __forceinline__ float block_reduce_sum(float v, float* red) {
    v = warp_reduce_sum(v);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float s = 0.0f;
    const int nw = (int)(blockDim.x >> 5);
    for (int i = 0; i < nw; ++i) s += red[i];
    __syncthreads();
    return s;
}

__device__ __forceinline__ float block_reduce_max(float v, float* red) {
    v = warp_reduce_max(v);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) red[wid] = v;
    __syncthreads();
    float m = 0.0f;
    const int nw = (int)(blockDim.x >> 5);
    for (int i = 0; i < nw; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    return m;
}

__device__ __forceinline__ void block_quantize_q8_k_store(
    float val,
    block_q8_K* dst,
    int tid,
    float* s_red
) {
    const float amax = block_reduce_max(fabsf(val), s_red);
    const float d = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int q = __float2int_rn(val * id);
    if (q > 127) q = 127;
    if (q < -127) q = -127;
    dst->qs[tid] = (int8_t)q;
    int bsum = q;
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        bsum += __shfl_xor_sync(0xFFFFFFFFu, bsum, off);
    }
    if ((tid & 15) == 0) dst->bsums[tid >> 4] = (int16_t)bsum;
    if (tid == 0) dst->d = d;
}

// Orthonormal self-inverse Walsh-Hadamard transform of 128 elements held 4 per lane across 1 warp.
__device__ __forceinline__ void had128w(float (&v)[4], int lane) {
    const float s0 = v[0] + v[1], d0 = v[0] - v[1];
    const float s1 = v[2] + v[3], d1 = v[2] - v[3];
    v[0] = s0 + s1;
    v[1] = d0 + d1;
    v[2] = s0 - s1;
    v[3] = d0 - d1;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(0xFFFFFFFFu, v[i], m);
            v[i] = (lane & m) ? (o - v[i]) : (v[i] + o);
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) v[i] *= 0.088388347648f;
}

// Orthonormal 128-point WHT across a block of 128 threads (one element per thread).
__device__ __forceinline__ float had128_block(float v, float* sh) {
    const int d = (int)threadIdx.x;
    const int lane = d & 31;
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const float o = __shfl_xor_sync(0xFFFFFFFFu, v, m);
        v = (lane & m) ? (o - v) : (v + o);
    }
    __syncthreads();
    sh[d] = v;
    __syncthreads();
    v = (d & 32) ? (sh[d ^ 32] - v) : (v + sh[d ^ 32]);
    __syncthreads();
    sh[d] = v;
    __syncthreads();
    v = (d & 64) ? (sh[d ^ 64] - v) : (v + sh[d ^ 64]);
    __syncthreads();
    return v * 0.088388347648f;
}

// -----------------------------------------------------------------------------
// 1. RMSNorm / Q8_K / SwiGLU-OAI / QK-Norm-RoPE-KV / Indexer-Store
// -----------------------------------------------------------------------------

__global__ void __launch_bounds__(256) mm3_rms_norm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    int dim,
    float eps,
    int add_one,
    float* __restrict__ y_f32_opt,
    block_q8_1* __restrict__ y_q8_1_opt,
    block_q8_K* __restrict__ y_q8_k_opt,
    uint16_t* __restrict__ y_bf16_opt
) {
    __shared__ float s_red[32];
    const int r = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* xr = x + (size_t)r * dim;

    float ss = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        const float v = xr[i];
        ss += v * v;
    }
    const float inv = rsqrtf(block_reduce_sum(ss, s_red) / (float)dim + eps);

    if (y_q8_k_opt != nullptr && (dim & 255) == 0) {
        const int nb_k = dim >> 8;
        const int nb_1 = dim >> 5;
        for (int b = 0; b < nb_k; ++b) {
            const int i = b * 256 + tid;
            const float wt = add_one ? (1.0f + w[i]) : w[i];
            const float val = xr[i] * inv * wt;
            if (y_f32_opt != nullptr) y_f32_opt[(size_t)r * dim + i] = val;
            if (y_bf16_opt != nullptr) y_bf16_opt[(size_t)r * dim + i] = f2bf(val);
            if (y_q8_1_opt != nullptr) warp_quantize_q8_1_store(val, y_q8_1_opt + (size_t)r * nb_1 + b * 8 + warp, lane);
            block_quantize_q8_k_store(val, y_q8_k_opt + (size_t)r * nb_k + b, tid, s_red);
            __syncthreads();
        }
        return;
    }

    const int nb_1 = dim >> 5;
    const int n_chunks = (dim + 255) >> 8;
    for (int b = 0; b < n_chunks; ++b) {
        const int i = b * 256 + tid;
        const bool ok = (i < dim);
        const float wt = ok ? (add_one ? (1.0f + w[i]) : w[i]) : 0.0f;
        const float val = ok ? (xr[i] * inv * wt) : 0.0f;
        if (ok) {
            if (y_f32_opt != nullptr) y_f32_opt[(size_t)r * dim + i] = val;
            if (y_bf16_opt != nullptr) y_bf16_opt[(size_t)r * dim + i] = f2bf(val);
        }
        if (y_q8_1_opt != nullptr && (dim & 31) == 0 && (b * 8 + warp) < nb_1) {
            warp_quantize_q8_1_store(val, y_q8_1_opt + (size_t)r * nb_1 + b * 8 + warp, lane);
        }
    }
}

__global__ void __launch_bounds__(256) mm3_quantize_q8_k_kernel(
    const float* __restrict__ x,
    block_q8_K* __restrict__ out,
    int K
) {
    __shared__ float s_red[32];
    const int r = (int)blockIdx.y;
    const int b = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int nb = (K + 255) >> 8;
    const int idx = b * 256 + tid;
    const float val = (idx < K) ? x[(size_t)r * K + idx] : 0.0f;
    block_quantize_q8_k_store(val, out + (size_t)r * nb + b, tid, s_red);
}

__global__ void __launch_bounds__(256) mm3_swigluoai_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    int ff,
    float alpha,
    float limit,
    float* __restrict__ y_f32_opt,
    block_q8_1* __restrict__ y_q8_1_opt,
    block_q8_K* __restrict__ y_q8_k_opt,
    uint16_t* __restrict__ y_bf16_opt
) {
    __shared__ float s_red[32];
    const int r = (int)blockIdx.y;
    const int b = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int idx = b * 256 + tid;
    const bool ok = (idx < ff);
    const float val = ok ? swoai_val(gate[(size_t)r * ff + idx], up[(size_t)r * ff + idx], alpha, limit) : 0.0f;
    if (ok) {
        if (y_f32_opt != nullptr) y_f32_opt[(size_t)r * ff + idx] = val;
        if (y_bf16_opt != nullptr) y_bf16_opt[(size_t)r * ff + idx] = f2bf(val);
    }
    if (y_q8_1_opt != nullptr && (ff & 31) == 0 && (b * 8 + warp) < (ff >> 5)) {
        warp_quantize_q8_1_store(val, y_q8_1_opt + (size_t)r * (ff >> 5) + b * 8 + warp, lane);
    }
    if (y_q8_k_opt != nullptr && (ff & 255) == 0) {
        block_quantize_q8_k_store(val, y_q8_k_opt + (size_t)r * (ff >> 8) + b, tid, s_red);
    }
}

__device__ __forceinline__ float norm_rope_elem(
    float x_d,
    const float* __restrict__ w,
    int p,
    int n_rot,
    const float* __restrict__ inv_freq,
    float eps,
    float* sh,
    float* red
) {
    const int d = (int)threadIdx.x;
    const int hd = (int)blockDim.x;
    const float inv = rsqrtf(block_reduce_sum(x_d * x_d, red) / (float)hd + eps);
    const float xn = x_d * inv * (w != nullptr ? w[d] : 1.0f);
    sh[d] = xn;
    __syncthreads();
    if (d >= n_rot) return xn;
    const int half = n_rot >> 1;
    const int partner = (d < half) ? (d + half) : (d - half);
    const int fidx = (d < half) ? d : (d - half);
    float sn, cs;
    sincosf((float)p * inv_freq[fidx], &sn, &cs);
    return (d < half) ? (xn * cs - sh[partner] * sn) : (xn * cs + sh[partner] * sn);
}

__global__ void __launch_bounds__(256) mm3_qk_norm_rope_kv_store_kernel(
    float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ qn,
    const float* __restrict__ kn,
    const int32_t* __restrict__ pos,
    const float* __restrict__ inv_freq,
    int Hq,
    int Hkv,
    int n_rot,
    float eps,
    int kv_fmt,
    int rot_mask,
    int ring,
    uint16_t* __restrict__ kc_bf16,
    uint16_t* __restrict__ vc_bf16,
    int8_t* __restrict__ kc_i8,
    float* __restrict__ ks_i8,
    int8_t* __restrict__ vc_i8,
    float* __restrict__ vs_i8,
    void* __restrict__ host_kc,
    float* __restrict__ host_ks,
    void* __restrict__ host_vc,
    float* __restrict__ host_vs,
    int32_t* __restrict__ host_tag
) {
    __shared__ float red[32], sh[256];
    const int t = (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    const int d = (int)threadIdx.x;
    const int hd = (int)blockDim.x;
    const int p = pos[t];

    if (h < Hq) {
        if (q != nullptr) {
            float* qr = q + ((size_t)t * Hq + h) * hd;
            qr[d] = norm_rope_elem(qr[d], qn, p, n_rot, inv_freq, eps, sh, red);
        }
        return;
    }

    const bool is_k = (h < Hq + Hkv);
    const int kh = is_k ? (h - Hq) : (h - Hq - Hkv);
    const int row = (ring > 0) ? (p % ring) : p;
    float o = is_k
        ? norm_rope_elem(k[((size_t)t * Hkv + kh) * hd + d], kn, p, n_rot, inv_freq, eps, sh, red)
        : v[((size_t)t * Hkv + kh) * hd + d];

    const size_t at = (size_t)row * Hkv + kh;
    const size_t ah = (size_t)p * Hkv + kh;
    const bool has_host = (host_kc != nullptr);

    if (kv_fmt == 0) {
        const uint16_t b = f2bf(o);
        (is_k ? kc_bf16 : vc_bf16)[at * hd + d] = b;
        if (has_host) {
            uint16_t* hk = reinterpret_cast<uint16_t*>(is_k ? host_kc : host_vc);
            if (hk != nullptr) hk[ah * hd + d] = b;
        }
    } else {
        const bool do_rot = is_k ? ((rot_mask & 1) != 0) : ((rot_mask & 2) != 0);
        if (do_rot && hd == 128) o = had128_block(o, sh);
        const float amax = block_reduce_max(fabsf(o), red);
        const float sc = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        int qi = __float2int_rn(o * id);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        const int8_t q8 = (int8_t)qi;
        if (is_k) {
            kc_i8[at * hd + d] = q8;
            if (d == 0) ks_i8[at] = sc;
            if (has_host) {
                reinterpret_cast<int8_t*>(host_kc)[ah * hd + d] = q8;
                if (d == 0 && host_ks != nullptr) host_ks[ah] = sc;
            }
        } else {
            vc_i8[at * hd + d] = q8;
            if (d == 0) vs_i8[at] = sc;
            if (has_host) {
                reinterpret_cast<int8_t*>(host_vc)[ah * hd + d] = q8;
                if (d == 0 && host_vs != nullptr) host_vs[ah] = sc;
            }
        }
    }
    if (has_host && host_tag != nullptr && is_k && kh == 0 && d == 0) {
        host_tag[row] = p;
    }
}

__global__ void __launch_bounds__(256) mm3_index_norm_rope_store_kernel(
    float* __restrict__ iq,
    const float* __restrict__ ik,
    const float* __restrict__ iq_norm_w,
    const float* __restrict__ ik_norm_w,
    const int32_t* __restrict__ pos,
    const float* __restrict__ inv_freq,
    int Hi,
    int n_rot,
    float eps,
    uint16_t* __restrict__ ic_bf16
) {
    __shared__ float red[32], sh[256];
    const int t = (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    const int d = (int)threadIdx.x;
    const int hd = (int)blockDim.x;
    const int p = pos[t];
    if (h < Hi) {
        float* x = iq + ((size_t)t * Hi + h) * hd;
        x[d] = bf2f(f2bf(norm_rope_elem(x[d], iq_norm_w, p, n_rot, inv_freq, eps, sh, red)));
    } else {
        ic_bf16[(size_t)p * hd + d] = f2bf(norm_rope_elem(ik[(size_t)t * hd + d], ik_norm_w, p, n_rot, inv_freq, eps, sh, red));
    }
}

// -----------------------------------------------------------------------------
// 2. MiniMax Sparse Attention (MSA)
// -----------------------------------------------------------------------------

// One block per (query t, key-block b): loops across keys k in [b*blk, min((b+1)*blk, pos[t]+1)),
// evaluating all Hi <= 8 index heads cooperatively. Output bs layout: [Hi, T, nblk].
__global__ void __launch_bounds__(128) mm3_msa_block_scores_kernel(
    const float* __restrict__ iq,      // [T, Hi, 128]
    const uint16_t* __restrict__ ic,   // [ctx, 128] BF16
    const int32_t* __restrict__ pos,   // [T]
    int T,
    int Hi,
    int nblk,
    int blk,
    float* __restrict__ bs             // [Hi, T, nblk]
) {
    __shared__ float s_iq[8][128];
    __shared__ float s_max[8][4];
    const int b = (int)blockIdx.x;
    const int t = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int p = pos[t];
    const int k_start = b * blk;
    if (k_start > p) {
        if (tid < Hi) bs[((size_t)tid * T + t) * nblk + b] = -INFINITY;
        return;
    }

    for (int i = tid; i < Hi * 128; i += 128) {
        const int h = i >> 7;
        const int d = i & 127;
        s_iq[h][d] = iq[((size_t)t * Hi + h) * 128 + d];
    }
    if (tid < 32) s_max[tid >> 2][tid & 3] = -INFINITY;
    __syncthreads();

    const int k_end = min(k_start + blk, p + 1);
    float loc_max[8];
    #pragma unroll
    for (int h = 0; h < 8; ++h) loc_max[h] = -INFINITY;

    for (int k0 = k_start; k0 < k_end; k0 += 4) {
        const int kp = k0 + warp;
        float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (kp < k_end) {
            const uint16_t* kr = ic + (size_t)kp * 128 + lane * 4;
            kv.x = bf2f(kr[0]);
            kv.y = bf2f(kr[1]);
            kv.z = bf2f(kr[2]);
            kv.w = bf2f(kr[3]);
        }
        for (int h = 0; h < Hi; ++h) {
            float dot = 0.0f;
            if (kp < k_end) {
                dot = fmaf(s_iq[h][lane * 4 + 0], kv.x, dot);
                dot = fmaf(s_iq[h][lane * 4 + 1], kv.y, dot);
                dot = fmaf(s_iq[h][lane * 4 + 2], kv.z, dot);
                dot = fmaf(s_iq[h][lane * 4 + 3], kv.w, dot);
            }
            dot = warp_reduce_sum(dot);
            if (lane == 0 && kp < k_end) {
                loc_max[h] = fmaxf(loc_max[h], dot);
            }
        }
    }
    if (lane == 0) {
        for (int h = 0; h < Hi; ++h) s_max[h][warp] = loc_max[h];
    }
    __syncthreads();
    if (tid < Hi) {
        float m = -INFINITY;
        #pragma unroll
        for (int w = 0; w < 4; ++w) m = fmaxf(m, s_max[tid][w]);
        bs[((size_t)tid * T + t) * nblk + b] = m;
    }
}

// Tiled 4-query block for Hi <= 4: reads each ic[kp] BF16 row once across up to 4 query tokens.
__global__ void __launch_bounds__(128) mm3_msa_block_scores_tiled4_kernel(
    const float* __restrict__ iq,      // [T, Hi, 128]
    const uint16_t* __restrict__ ic,   // [ctx, 128] BF16
    const int32_t* __restrict__ pos,   // [T]
    int T,
    int Hi,
    int nblk,
    int blk,
    float* __restrict__ bs             // [Hi, T, nblk]
) {
    __shared__ float s_iq[4][4][128];
    __shared__ float s_max[4][4][4];

    const int b = (int)blockIdx.x;
    const int t0 = (int)blockIdx.y * 4;
    const int nq = min(4, T - t0);
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int k_start = b * blk;

    int p_arr[4] = {-1, -1, -1, -1};
    int max_p = -1;
    #pragma unroll
    for (int qt = 0; qt < 4; ++qt) {
        if (qt < nq) {
            p_arr[qt] = pos[t0 + qt];
            max_p = max(max_p, p_arr[qt]);
        }
    }
    if (k_start > max_p) {
        if (tid < nq * Hi) {
            const int qt = tid / Hi;
            const int h = tid - qt * Hi;
            bs[((size_t)h * T + (t0 + qt)) * nblk + b] = -INFINITY;
        }
        return;
    }

    const int tot = nq * Hi * 128;
    for (int i = tid; i < tot; i += 128) {
        const int qt = i / (Hi * 128);
        const int rem = i - qt * (Hi * 128);
        const int h = rem >> 7;
        const int d = rem & 127;
        s_iq[qt][h][d] = iq[((size_t)(t0 + qt) * Hi + h) * 128 + d];
    }
    __syncthreads();

    const int k_end = min(k_start + blk, max_p + 1);
    float loc_max[4][4];
    #pragma unroll
    for (int qt = 0; qt < 4; ++qt)
        #pragma unroll
        for (int h = 0; h < 4; ++h) loc_max[qt][h] = -INFINITY;

    for (int k0 = k_start; k0 < k_end; k0 += 4) {
        const int kp = k0 + warp;
        const bool k_valid = (kp < k_end);
        float4 kv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (k_valid) {
            const uint16_t* kr = ic + (size_t)kp * 128 + lane * 4;
            kv.x = bf2f(kr[0]);
            kv.y = bf2f(kr[1]);
            kv.z = bf2f(kr[2]);
            kv.w = bf2f(kr[3]);
        }
        for (int h = 0; h < Hi; ++h) {
            #pragma unroll
            for (int qt = 0; qt < 4; ++qt) {
                const bool attend = (qt < nq) && k_valid && (kp <= p_arr[qt]);
                float dot = 0.0f;
                if (attend) {
                    const float4 qv = *reinterpret_cast<const float4*>(&s_iq[qt][h][lane * 4]);
                    dot = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, qv.w * kv.w)));
                }
                dot = warp_reduce_sum(dot);
                if (lane == 0 && attend) {
                    loc_max[qt][h] = fmaxf(loc_max[qt][h], dot);
                }
            }
        }
    }

    if (lane == 0) {
        for (int qt = 0; qt < nq; ++qt)
            for (int h = 0; h < Hi; ++h) s_max[qt][h][warp] = loc_max[qt][h];
    }
    __syncthreads();

    if (tid < nq * Hi) {
        const int qt = tid / Hi;
        const int h = tid - qt * Hi;
        float m = -INFINITY;
        #pragma unroll
        for (int w = 0; w < 4; ++w) m = fmaxf(m, s_max[qt][h][w]);
        bs[((size_t)h * T + (t0 + qt)) * nblk + b] = m;
    }
}

__device__ __forceinline__ bool msa_better(float av, int ai, float bv, int bi) {
    if (isnan(av)) return false;
    if (isnan(bv)) return true;
    return (av > bv) || (av == bv && ai < bi);
}

// One warp per (h, t): selects topk block indices into kept[h, t, topk].
__global__ void __launch_bounds__(256) mm3_msa_select_topk_kernel(
    const float* __restrict__ bs,
    const int32_t* __restrict__ pos,
    int T,
    int Hi,
    int nblk,
    int blk,
    int topk,
    int local_blocks,
    int32_t* __restrict__ kept
) {
    const int idx = (blockIdx.x * 8) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (idx >= T * Hi) return;
    const int h = idx / T;
    const int t = idx - h * T;
    const float* row = bs + ((size_t)h * T + t) * nblk;
    int32_t* out = kept + ((size_t)h * T + t) * topk;

    if (nblk <= 0 || topk <= 0) return;

    const int qb = pos[t] / blk;
    if (qb + 1 <= topk) {
        for (int i = lane; i < topk; i += 32) {
            out[i] = (i <= qb && i < nblk) ? i : -1;
        }
        return;
    }

    const int n_loc = max(0, min(local_blocks, min(topk, qb + 1)));
    for (int j = lane; j < n_loc; j += 32) {
        const int b_loc = qb - j;
        out[j] = (b_loc < nblk) ? b_loc : -1;
    }

    const int search_end = min(nblk, qb + 1 - n_loc);
    float ps = INFINITY;
    int pi = -1;
    for (int r = n_loc; r < topk; ++r) {
        float bv = -INFINITY;
        int bi = 0x7FFFFFFF;
        for (int b = lane; b < search_end; b += 32) {
            const float v = row[b];
            if (v > -INFINITY && msa_better(ps, pi, v, b) && msa_better(v, b, bv, bi)) {
                bv = v;
                bi = b;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, off);
            if (msa_better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (bi == 0x7FFFFFFF || !(bv > -INFINITY)) {
            for (int rem = r + lane; rem < topk; rem += 32) out[rem] = -1;
            break;
        }
        if (lane == 0) out[r] = bi;
        ps = bv;
        pi = bi;
    }
}

constexpr int kAttnTile = 32;
constexpr int kAttnG = 16;
constexpr int kAttnPartStride = 130; // 128 dims + m + l
constexpr int kCombMaxSplits = 32;
constexpr int kWideRow = 130;

// 128-thread block per (t, kv_group g, split z): one thread per dim d in 0..127.
// Computes all G = Hq / Hkv heads sequentially across shared score rows to keep register count <= 48.
template <int KV_FMT, bool CAUSAL>
__global__ void __launch_bounds__(128) mm3_attn_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ kc_bf16,
    const uint16_t* __restrict__ vc_bf16,
    const int8_t* __restrict__ kc_i8,
    const float* __restrict__ ks_i8,
    const int8_t* __restrict__ vc_i8,
    const float* __restrict__ vs_i8,
    const int32_t* __restrict__ kept,  // nullable when !CAUSAL
    const int32_t* __restrict__ lo_arr,// nullable when CAUSAL
    const int32_t* __restrict__ pos,
    int hi_limit,
    int Hq,
    int Hkv,
    int topk,
    int blk,
    int rot_mask,
    int ring,
    float scale,
    int splits,
    float* __restrict__ part_opt,
    float* __restrict__ out
) {
    __shared__ float s_q[kAttnG][128];
    __shared__ float s_k[kAttnTile][128];
    __shared__ float s_v[kAttnTile][128];
    __shared__ float s_sc[kAttnG][kAttnTile];
    __shared__ float s_m[kAttnG], s_l[kAttnG], s_a[kAttnG];
    __shared__ float s_red[kAttnG][4];

    const int t = (int)blockIdx.x;
    const int g = (int)blockIdx.y;
    const int z = (int)blockIdx.z;
    const int d = (int)threadIdx.x;
    const int warp = d >> 5;
    const int lane = d & 31;
    const int p = pos ? pos[t] : hi_limit;
    const int G = Hq / Hkv;

    for (int h = 0; h < G; ++h) {
        s_q[h][d] = q[((size_t)t * Hq + g * G + h) * 128 + d] * scale;
    }
    __syncthreads();
    if constexpr (KV_FMT == 1) {
        if ((rot_mask & 1) != 0) {
            for (int h = warp; h < G; h += 4) {
                float xv[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) xv[i] = s_q[h][lane * 4 + i];
                had128w(xv, lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i) s_q[h][lane * 4 + i] = xv[i];
            }
            __syncthreads();
        }
    }

    float acc[kAttnG];
    #pragma unroll
    for (int h = 0; h < kAttnG; ++h) acc[h] = 0.0f;
    if (d < G) {
        s_m[d] = -INFINITY;
        s_l[d] = 0.0f;
    }

    auto process_tile = [&](int k0, int k_end) {
        const int count = max(0, min(kAttnTile, k_end - k0));
        if (count <= 0) return;
        __syncthreads();
        for (int j = 0; j < count; ++j) {
            const int kp = k0 + j;
            const int row = (ring > 0) ? (kp % ring) : kp;
            const size_t at = (size_t)row * Hkv + g;
            if constexpr (KV_FMT == 0) {
                s_k[j][d] = bf2f(kc_bf16[at * 128 + d]);
                s_v[j][d] = bf2f(vc_bf16[at * 128 + d]);
            } else {
                s_k[j][d] = (float)kc_i8[at * 128 + d] * ks_i8[at];
                s_v[j][d] = (float)vc_i8[at * 128 + d] * vs_i8[at];
            }
        }
        __syncthreads();

        // Warps process 4 keys at a time across heads for minimal register pressure.
        for (int j0 = 0; j0 < count; j0 += 4) {
            const int j = j0 + warp;
            for (int h = 0; h < G; ++h) {
                float dot = 0.0f;
                if (j < count) {
                    const float4 kv = *reinterpret_cast<const float4*>(&s_k[j][lane * 4]);
                    const float4 qv = *reinterpret_cast<const float4*>(&s_q[h][lane * 4]);
                    dot = fmaf(qv.x, kv.x, fmaf(qv.y, kv.y, fmaf(qv.z, kv.z, qv.w * kv.w)));
                }
                dot = warp_reduce_sum(dot);
                if (lane == 0 && j < count) s_sc[h][j] = dot;
            }
        }
        __syncthreads();

        // Head-wise softmax update + V accumulation.
        // Threads compute per-head max across `count` keys in 4-thread-per-head reduction or single pass.
        if (d < G * 4) {
            const int h = d >> 2;
            const int sub = d & 3;
            float m_loc = -INFINITY;
            for (int j = sub; j < count; j += 4) m_loc = fmaxf(m_loc, s_sc[h][j]);
            s_red[h][sub] = m_loc;
        }
        __syncthreads();
        if (d < G) {
            float m_tile = -INFINITY;
            #pragma unroll
            for (int s = 0; s < 4; ++s) m_tile = fmaxf(m_tile, s_red[d][s]);
            const float m_old = s_m[d];
            const float m_new = fmaxf(m_old, m_tile);
            s_a[d] = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
            s_m[d] = m_new;
        }
        __syncthreads();
        for (int i = d; i < G * count; i += 128) {
            const int h = i / count;
            const int j = i - h * count;
            s_sc[h][j] = expf(s_sc[h][j] - s_m[h]);
        }
        __syncthreads();
        if (d < G * 4) {
            const int h = d >> 2;
            const int sub = d & 3;
            float sum_loc = 0.0f;
            for (int j = sub; j < count; j += 4) sum_loc += s_sc[h][j];
            s_red[h][sub] = sum_loc;
        }
        __syncthreads();
        if (d < G) {
            float l_tile = 0.0f;
            #pragma unroll
            for (int s = 0; s < 4; ++s) l_tile += s_red[d][s];
            s_l[d] = s_l[d] * s_a[d] + l_tile;
        }
        __syncthreads();

        for (int h = 0; h < G; ++h) {
            float a = acc[h] * s_a[h];
            for (int j = 0; j < count; ++j) {
                a = fmaf(s_sc[h][j], s_v[j][d], a);
            }
            acc[h] = a;
        }
    };

    if constexpr (CAUSAL) {
        const int bi0 = z * topk / splits;
        const int bi1 = (z + 1) * topk / splits;
        const int* kb = kept + ((size_t)g * (int)gridDim.x + t) * topk;
        for (int bi = bi0; bi < bi1; ++bi) {
            const int b = kb[bi];
            if (b < 0) continue;
            const int b_start = b * blk;
            const int b_end = min((b + 1) * blk, p + 1);
            for (int k0 = b_start; k0 < b_end; k0 += kAttnTile) {
                process_tile(k0, b_end);
            }
        }
    } else {
        const int k_lo = lo_arr[t];
        const int k_hi = hi_limit + 1;
        const int total_tiles = max(0, (k_hi - k_lo + kAttnTile - 1) / kAttnTile);
        const int ti0 = z * total_tiles / splits;
        const int ti1 = (z + 1) * total_tiles / splits;
        for (int ti = ti0; ti < ti1; ++ti) {
            const int k0 = k_lo + ti * kAttnTile;
            process_tile(k0, k_hi);
        }
    }

    __syncthreads();
    if (splits > 1) {
        float* pz = part_opt + (((size_t)t * Hkv + g) * splits + z) * G * kAttnPartStride;
        for (int h = 0; h < G; ++h) pz[h * kAttnPartStride + d] = acc[h];
        if (d < G) {
            pz[d * kAttnPartStride + 128] = s_m[d];
            pz[d * kAttnPartStride + 129] = s_l[d];
        }
        return;
    }

    if constexpr (KV_FMT == 1) {
        if ((rot_mask & 2) != 0) {
            for (int h = 0; h < G; ++h) {
                s_q[h][d] = (s_l[h] > 0.0f) ? (acc[h] / s_l[h]) : 0.0f;
            }
            __syncthreads();
            for (int h = warp; h < G; h += 4) {
                float xv[4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) xv[i] = s_q[h][lane * 4 + i];
                had128w(xv, lane);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    out[((size_t)t * Hq + g * G + h) * 128 + lane * 4 + i] = xv[i];
                }
            }
            return;
        }
    }
    for (int h = 0; h < G; ++h) {
        out[((size_t)t * Hq + g * G + h) * 128 + d] = (s_l[h] > 0.0f) ? (acc[h] / s_l[h]) : 0.0f;
    }
}

// High-occupancy 512-thread wide decode/verify attention for Hq == 16 * Hkv and T <= 16.
// Tile of 32 keys x 16 heads = 512 (head, key) pairs -> 1 thread per pair for QK dots and
// zero-barrier warp softmax, followed by 4 heads/thread V accumulation across kWideRow-padded SMEM.
template <int KV_FMT, bool CAUSAL>
__global__ void __launch_bounds__(512) mm3_attn_wide_kernel(
    const float* __restrict__ q,
    const uint16_t* __restrict__ kc_bf16,
    const uint16_t* __restrict__ vc_bf16,
    const int8_t* __restrict__ kc_i8,
    const float* __restrict__ ks_i8,
    const int8_t* __restrict__ vc_i8,
    const float* __restrict__ vs_i8,
    const int32_t* __restrict__ kept,
    const int32_t* __restrict__ lo_arr,
    const int32_t* __restrict__ pos,
    int hi_limit,
    int Hq,
    int Hkv,
    int topk,
    int blk,
    int rot_mask,
    int ring,
    float scale,
    int splits,
    float* __restrict__ part_opt,
    float* __restrict__ out
) {
    __shared__ float s_q[kAttnG][128];
    __shared__ __align__(4) uint16_t s_k16[kAttnTile][kWideRow];
    __shared__ __align__(4) uint16_t s_v16[kAttnTile][kWideRow];
    __shared__ float s_ksc[kAttnTile];
    __shared__ float s_vsc[kAttnTile];
    __shared__ float s_sc[kAttnG][kAttnTile + 1];
    __shared__ float s_m[kAttnG], s_l[kAttnG], s_a[kAttnG];

    int8_t (*s_k8)[2 * kWideRow] = reinterpret_cast<int8_t (*)[2 * kWideRow]>(s_k16);
    int8_t (*s_v8)[2 * kWideRow] = reinterpret_cast<int8_t (*)[2 * kWideRow]>(s_v16);

    const int t = (int)blockIdx.x;
    const int g = (int)blockIdx.y;
    const int z = (int)blockIdx.z;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int d = tid & 127;
    const int hq = tid >> 7;
    const int p = pos ? pos[t] : hi_limit;

    for (int i = tid; i < kAttnG * 128; i += 512) {
        const int h = i >> 7;
        const int dd = i & 127;
        s_q[h][dd] = q[((size_t)t * Hq + g * kAttnG + h) * 128 + dd] * scale;
    }
    __syncthreads();

    if constexpr (KV_FMT == 1) {
        if ((rot_mask & 1) != 0) {
            float xv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) xv[i] = s_q[warp][lane * 4 + i];
            had128w(xv, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) s_q[warp][lane * 4 + i] = xv[i];
            __syncthreads();
        }
    }

    if (tid < kAttnG) {
        s_m[tid] = -INFINITY;
        s_l[tid] = 0.0f;
    }
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    auto process_tile = [&](int k0, int k_end) {
        const int count = max(0, min(kAttnTile, k_end - k0));
        if (count <= 0) return;
        __syncthreads();

        if constexpr (KV_FMT == 0) {
            for (int i = tid; i < kAttnTile * 64; i += 512) {
                const int j = i >> 6;
                const int c = i & 63;
                uint32_t kw = 0u, vw = 0u;
                if (j < count) {
                    const int kp = k0 + j;
                    const int row = (ring > 0) ? (kp % ring) : kp;
                    const size_t at = ((size_t)row * Hkv + g) * 128 + (size_t)c * 2;
                    kw = *reinterpret_cast<const uint32_t*>(kc_bf16 + at);
                    vw = *reinterpret_cast<const uint32_t*>(vc_bf16 + at);
                }
                *reinterpret_cast<uint32_t*>(&s_k16[j][c * 2]) = kw;
                *reinterpret_cast<uint32_t*>(&s_v16[j][c * 2]) = vw;
            }
        } else {
            for (int i = tid; i < kAttnTile * 32; i += 512) {
                const int j = i >> 5;
                const int c = i & 31;
                uint32_t kw = 0u, vw = 0u;
                if (j < count) {
                    const int kp = k0 + j;
                    const int row = (ring > 0) ? (kp % ring) : kp;
                    const size_t at = ((size_t)row * Hkv + g) * 128 + (size_t)c * 4;
                    kw = *reinterpret_cast<const uint32_t*>(kc_i8 + at);
                    vw = *reinterpret_cast<const uint32_t*>(vc_i8 + at);
                }
                *reinterpret_cast<uint32_t*>(&s_k8[j][c * 4]) = kw;
                *reinterpret_cast<uint32_t*>(&s_v8[j][c * 4]) = vw;
            }
            if (tid < count) {
                const int kp = k0 + tid;
                const int row = (ring > 0) ? (kp % ring) : kp;
                const size_t at = (size_t)row * Hkv + g;
                s_ksc[tid] = ks_i8[at];
                s_vsc[tid] = vs_i8[at];
            }
        }
        __syncthreads();

        float sc = -INFINITY;
        if (lane < count) {
            float dot = 0.0f;
            if constexpr (KV_FMT == 0) {
                #pragma unroll 8
                for (int dd = 0; dd < 128; dd += 2) {
                    const uint32_t kw = *reinterpret_cast<const uint32_t*>(&s_k16[lane][dd]);
                    dot = fmaf(s_q[warp][dd], bf2f(static_cast<uint16_t>(kw & 0xFFFFu)), dot);
                    dot = fmaf(s_q[warp][dd + 1], bf2f(static_cast<uint16_t>(kw >> 16)), dot);
                }
            } else {
                #pragma unroll 8
                for (int dd = 0; dd < 128; dd += 4) {
                    const uint32_t q4 = *reinterpret_cast<const uint32_t*>(&s_k8[lane][dd]);
                    dot = fmaf(s_q[warp][dd + 0], static_cast<float>(static_cast<int8_t>(q4 & 0xFFu)), dot);
                    dot = fmaf(s_q[warp][dd + 1], static_cast<float>(static_cast<int8_t>((q4 >> 8) & 0xFFu)), dot);
                    dot = fmaf(s_q[warp][dd + 2], static_cast<float>(static_cast<int8_t>((q4 >> 16) & 0xFFu)), dot);
                    dot = fmaf(s_q[warp][dd + 3], static_cast<float>(static_cast<int8_t>(q4 >> 24)), dot);
                }
                dot *= s_ksc[lane];
            }
            sc = dot;
        }

        const float m_tile = warp_reduce_max(sc);
        const float m_old = s_m[warp];
        const float m_new = fmaxf(m_old, m_tile);
        const float a_fac = (m_old == -INFINITY) ? 0.0f : expf(m_old - m_new);
        const float e = (sc == -INFINITY) ? 0.0f : expf(sc - m_new);
        const float l_tile = warp_reduce_sum(e);
        s_sc[warp][lane] = (KV_FMT == 1 && lane < count) ? (e * s_vsc[lane]) : e;
        __syncwarp();
        if (lane == 0) {
            s_m[warp] = m_new;
            s_l[warp] = s_l[warp] * a_fac + l_tile;
            s_a[warp] = a_fac;
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int h = hq * 4 + i;
            float s = acc[i] * s_a[h];
            if constexpr (KV_FMT == 1) {
                for (int j = 0; j < count; ++j) {
                    s = fmaf(s_sc[h][j], static_cast<float>(s_v8[j][d]), s);
                }
            } else {
                for (int j = 0; j < count; ++j) {
                    s = fmaf(s_sc[h][j], bf2f(s_v16[j][d]), s);
                }
            }
            acc[i] = s;
        }
    };

    if constexpr (CAUSAL) {
        const int bi0 = z * topk / splits;
        const int bi1 = (z + 1) * topk / splits;
        const int* kb = kept + ((size_t)g * (int)gridDim.x + t) * topk;
        for (int bi = bi0; bi < bi1; ++bi) {
            const int b = kb[bi];
            if (b < 0) continue;
            const int b_start = b * blk;
            const int b_end = min((b + 1) * blk, p + 1);
            for (int k0 = b_start; k0 < b_end; k0 += kAttnTile) {
                process_tile(k0, b_end);
            }
        }
    } else {
        const int k_lo = lo_arr[t];
        const int k_hi = hi_limit + 1;
        const int total_tiles = max(0, (k_hi - k_lo + kAttnTile - 1) / kAttnTile);
        const int ti0 = z * total_tiles / splits;
        const int ti1 = (z + 1) * total_tiles / splits;
        for (int ti = ti0; ti < ti1; ++ti) {
            const int k0 = k_lo + ti * kAttnTile;
            process_tile(k0, k_hi);
        }
    }

    __syncthreads();
    if (splits > 1) {
        float* pz = part_opt + (((size_t)t * Hkv + g) * splits + z) * kAttnG * kAttnPartStride;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            pz[(hq * 4 + i) * kAttnPartStride + d] = acc[i];
        }
        if (tid < kAttnG) {
            pz[tid * kAttnPartStride + 128] = s_m[tid];
            pz[tid * kAttnPartStride + 129] = s_l[tid];
        }
        return;
    }

    if constexpr (KV_FMT == 1) {
        if ((rot_mask & 2) != 0) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int h = hq * 4 + i;
                s_q[h][d] = (s_l[h] > 0.0f) ? (acc[i] / s_l[h]) : 0.0f;
            }
            __syncthreads();
            float xv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) xv[i] = s_q[warp][lane * 4 + i];
            had128w(xv, lane);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                out[((size_t)t * Hq + g * kAttnG + warp) * 128 + lane * 4 + i] = xv[i];
            }
            return;
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int h = hq * 4 + i;
        out[((size_t)t * Hq + g * kAttnG + h) * 128 + d] = (s_l[h] > 0.0f) ? (acc[i] / s_l[h]) : 0.0f;
    }
}

__global__ void __launch_bounds__(128) mm3_attn_combine_kernel(
    const float* __restrict__ part,
    int Hq,
    int Hkv,
    int splits,
    int unrot_v,
    float* __restrict__ out
) {
    __shared__ float s_o[128], s_m[kCombMaxSplits], s_l[kCombMaxSplits];
    const int t = (int)blockIdx.x;
    const int g = (int)blockIdx.y;
    const int h = (int)blockIdx.z;
    const int d = (int)threadIdx.x;
    const int G = Hq / Hkv;
    const float* pt = part + ((size_t)t * Hkv + g) * splits * G * kAttnPartStride;
    if (d < splits) {
        s_m[d] = pt[((size_t)d * G + h) * kAttnPartStride + 128];
        s_l[d] = pt[((size_t)d * G + h) * kAttnPartStride + 129];
    }
    __syncthreads();
    float M = -INFINITY;
    for (int z = 0; z < splits; ++z) M = fmaxf(M, s_m[z]);
    float L = 0.0f, a = 0.0f;
    if (M != -INFINITY) {
        for (int z = 0; z < splits; ++z) {
            const float mz = s_m[z];
            const float f = (mz == -INFINITY) ? 0.0f : expf(mz - M);
            L = fmaf(s_l[z], f, L);
            a = fmaf(pt[((size_t)z * G + h) * kAttnPartStride + d], f, a);
        }
    }
    const float val = (L > 0.0f) ? (a / L) : 0.0f;
    float* orow = out + ((size_t)t * Hq + g * G + h) * 128;
    if (!unrot_v) {
        orow[d] = val;
        return;
    }
    s_o[d] = val;
    __syncthreads();
    if (d < 32) {
        float xv[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) xv[i] = s_o[d * 4 + i];
        had128w(xv, d);
        #pragma unroll
        for (int i = 0; i < 4; ++i) orow[d * 4 + i] = xv[i];
    }
}

// -----------------------------------------------------------------------------
// 3. Router F32 & Independent-Type Q8_K MoE
// -----------------------------------------------------------------------------

__global__ void __launch_bounds__(256) mm3_router_topk_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int T,
    int E,
    int topk,
    float scale,
    int32_t* __restrict__ ids_out,
    float* __restrict__ weights_out
) {
    const int t = ((int)blockIdx.x * 8) + ((int)threadIdx.x >> 5);
    const int lane = (int)threadIdx.x & 31;
    if (t >= T) return;

    float sg[8], ch[8];
    const int per = (E + 31) >> 5;
    for (int i = 0; i < per; ++i) {
        const int e = lane + 32 * i;
        if (e < E) {
            sg[i] = 1.0f / (1.0f + expf(-logits[(size_t)t * E + e]));
            ch[i] = sg[i] + (bias ? bias[e] : 0.0f);
        } else {
            sg[i] = 0.0f;
            ch[i] = -INFINITY;
        }
    }

    float ps = INFINITY;
    int pi = -1;
    float w_sum = 0.0f;
    for (int r = 0; r < topk; ++r) {
        float bv = -INFINITY, bsg = 0.0f;
        int bi = 0x7FFFFFFF;
        for (int i = 0; i < per; ++i) {
            const int e = lane + 32 * i;
            if (e < E && msa_better(ps, pi, ch[i], e) && msa_better(ch[i], e, bv, bi)) {
                bv = ch[i];
                bi = e;
                bsg = sg[i];
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, off);
            const int oi = __shfl_xor_sync(0xFFFFFFFFu, bi, off);
            const float os = __shfl_xor_sync(0xFFFFFFFFu, bsg, off);
            if (msa_better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
                bsg = os;
            }
        }
        if (lane == 0) {
            ids_out[(size_t)t * topk + r] = bi;
            weights_out[(size_t)t * topk + r] = bsg;
        }
        w_sum += bsg;
        ps = bv;
        pi = bi;
    }
    __syncwarp();
    if (lane < topk) {
        const float inv = (w_sum > 0.0f) ? (scale / w_sum) : 0.0f;
        weights_out[(size_t)t * topk + lane] *= inv;
    }
}

__device__ __forceinline__ void load_q8_k_sub_words(
    const block_q8_K* xb,
    int sub,
    int (&u)[8],
    float& d8,
    int& bsum32
) {
    const int8_t* qs = xb->qs + sub * 32;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        u[j] = load_i32_align4(qs + j * 4);
    }
    d8 = xb->d;
    bsum32 = (int)xb->bsums[2 * sub] + (int)xb->bsums[2 * sub + 1];
}

__device__ __forceinline__ float dot_sub_iq1_s_q8k(
    const block_iq1_s* blk,
    const int (&u)[8],
    float d8,
    int bsum32,
    int sub
) {
    const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
    const uint16_t qh = load_u16_align2(&blk->qh[sub]);
    const uint32_t qs4 = (uint32_t)load_i32_align2(blk->qs + 4 * sub);
    const int sc = 2 * (int)((qh >> 12) & 7u) + 1;
    const int sgn = (qh & 0x8000u) ? -1 : 1;
    int sumi = 0;
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t g_idx = ((qs4 >> (8 * l)) & 0xFFu) | ((((uint32_t)qh >> (3 * l)) & 7u) << 8);
        const uint64_t g64 = d_iq1s_grid[g_idx];
        sumi = dp4a_packed((int)(uint32_t)g64, u[2 * l], sumi);
        sumi = dp4a_packed((int)(uint32_t)(g64 >> 32), u[2 * l + 1], sumi);
    }
    return dw * d8 * 0.125f * (float)(sc * (8 * sumi + sgn * bsum32));
}

template <int W_TYPE>
__device__ __forceinline__ float dot_sub_mm3_q8k(
    const uint8_t* w_row,
    int b,
    int sub,
    const int (&u)[8],
    float d8,
    int bsum32
) {
    if constexpr (W_TYPE == GGML_TYPE_IQ2_XXS) {
        return dot_sub_iq2_xxs(reinterpret_cast<const block_iq2_xxs*>(w_row) + b, u, d8, sub);
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_XS) {
        return dot_sub_iq2_xs(reinterpret_cast<const block_iq2_xs*>(w_row) + b, u, d8, sub);
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
        return dot_sub_iq2_s(reinterpret_cast<const block_iq2_s*>(w_row) + b, u, d8, sub, nullptr);
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
        return dot_sub_iq3_xxs(reinterpret_cast<const block_iq3_xxs*>(w_row) + b, u, d8, sub, nullptr);
    } else if constexpr (W_TYPE == GGML_TYPE_IQ1_S) {
        return dot_sub_iq1_s_q8k(reinterpret_cast<const block_iq1_s*>(w_row) + b, u, d8, bsum32, sub);
    } else {
        return 0.0f;
    }
}

__device__ __forceinline__ float dot_sub_mm3_q8k_dyn(
    int w_type,
    const uint8_t* w_row,
    int b,
    int sub,
    const int (&u)[8],
    float d8,
    int bsum32
) {
    switch (w_type) {
        case GGML_TYPE_IQ2_XXS: return dot_sub_mm3_q8k<GGML_TYPE_IQ2_XXS>(w_row, b, sub, u, d8, bsum32);
        case GGML_TYPE_IQ2_XS:  return dot_sub_mm3_q8k<GGML_TYPE_IQ2_XS>(w_row, b, sub, u, d8, bsum32);
        case GGML_TYPE_IQ2_S:   return dot_sub_mm3_q8k<GGML_TYPE_IQ2_S>(w_row, b, sub, u, d8, bsum32);
        case GGML_TYPE_IQ3_XXS: return dot_sub_mm3_q8k<GGML_TYPE_IQ3_XXS>(w_row, b, sub, u, d8, bsum32);
        case GGML_TYPE_IQ1_S:   return dot_sub_mm3_q8k<GGML_TYPE_IQ1_S>(w_row, b, sub, u, d8, bsum32);
        default: return 0.0f;
    }
}

// 1 block of 256 threads computes 256 consecutive n_ff rows (1 block_q8_K of output)
// for a single (m, slot) expert, fusing SwiGLU-OAI and inline Q8_K quantization.
template <int TY_G, int TY_U>
__global__ void __launch_bounds__(256) mm3_moe_gate_up_q8k_kernel(
    int gate_type_dyn,
    int up_type_dyn,
    const uint64_t* __restrict__ gate_ptrs,
    const uint64_t* __restrict__ up_ptrs,
    size_t gate_rb,
    size_t up_rb,
    const block_q8_K* __restrict__ x_q8k,
    const int32_t* __restrict__ expert_ids,
    block_q8_K* __restrict__ inter_q8k,
    float* __restrict__ inter_f32_opt,
    int n_embd,
    int n_ff,
    int topk,
    float alpha,
    float limit,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int r_base = (int)blockIdx.x * 256;
    const int slot = (int)blockIdx.y;
    const int m = (int)blockIdx.z;
    const int e = expert_ids[(size_t)m * topk + slot];
    const uint8_t* gp = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(gate_ptrs[e]));
    const uint8_t* up = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(up_ptrs[e]));
    if (gp == nullptr || up == nullptr) return;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nb_x = n_embd >> 8;
    const int nb_ff = n_ff >> 8;
    const block_q8_K* gx = x_q8k + (size_t)m * nb_x;

    // 24 blocks * 292 bytes = 7008 bytes <= 8 KB SMEM.
    constexpr int MAX_STAGED_Q8K_WORDS = 32 * 73; // supports up to n_embd=8192
    __shared__ __align__(16) uint32_t s_x_words[MAX_STAGED_Q8K_WORDS];
    __shared__ float s_ff[256];
    __shared__ float s_red[32];

    const int n_words = nb_x * 73;
    const bool staged = (nb_x <= 32);
    if (staged) {
        const uint32_t* src_w = reinterpret_cast<const uint32_t*>(gx);
        for (int i = tid; i < n_words; i += 256) s_x_words[i] = src_w[i];
        __syncthreads();
    }
    const block_q8_K* x_use = staged ? reinterpret_cast<const block_q8_K*>(s_x_words) : gx;

    const int sub = lane & 7;
    const int b0 = lane >> 3;
    for (int step = 0; step < 32; ++step) {
        const int r_local = warp * 32 + step;
        const int r = r_base + r_local;
        if (r < n_ff) {
            const uint8_t* gr = gp + (size_t)r * gate_rb;
            const uint8_t* ur = up + (size_t)r * up_rb;
            float ag = 0.0f, au = 0.0f;
            for (int b = b0; b < nb_x; b += 4) {
                int u[8]; float d8 = 0.0f; int bsum32 = 0;
                load_q8_k_sub_words(x_use + b, sub, u, d8, bsum32);
                if constexpr (TY_G > 0 && TY_U > 0) {
                    ag += dot_sub_mm3_q8k<TY_G>(gr, b, sub, u, d8, bsum32);
                    au += dot_sub_mm3_q8k<TY_U>(ur, b, sub, u, d8, bsum32);
                } else {
                    ag += dot_sub_mm3_q8k_dyn(gate_type_dyn, gr, b, sub, u, d8, bsum32);
                    au += dot_sub_mm3_q8k_dyn(up_type_dyn, ur, b, sub, u, d8, bsum32);
                }
            }
            const float vg = warp_reduce_sum(ag);
            const float vu = warp_reduce_sum(au);
            if (lane == 0) {
                const float v = swoai_val(vg, vu, alpha, limit);
                s_ff[r_local] = v;
                if (inter_f32_opt != nullptr) {
                    inter_f32_opt[((size_t)m * topk + slot) * n_ff + r] = v;
                }
            }
        }
    }
    __syncthreads();

    const float val = (r_base + tid < n_ff) ? s_ff[tid] : 0.0f;
    block_quantize_q8_k_store(
        val,
        inter_q8k + ((size_t)m * topk + slot) * nb_ff + blockIdx.x,
        tid,
        s_red
    );
}

template <int TY_D>
__global__ void __launch_bounds__(256) mm3_moe_down_q8k_kernel(
    int down_type_dyn,
    const uint64_t* __restrict__ down_ptrs,
    size_t down_rb,
    const block_q8_K* __restrict__ inter_q8k,
    const int32_t* __restrict__ expert_ids,
    float* __restrict__ expert_out,
    int n_ff,
    int n_embd,
    int topk,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    const int r_base = (int)blockIdx.x * 32;
    const int slot = (int)blockIdx.y;
    const int m = (int)blockIdx.z;
    const int e = expert_ids[(size_t)m * topk + slot];
    const uint8_t* dp = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(down_ptrs[e]));
    if (dp == nullptr) return;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int nb_ff = n_ff >> 8;
    const block_q8_K* gf = inter_q8k + ((size_t)m * topk + slot) * nb_ff;

    constexpr int MAX_STAGED_Q8K_WORDS = 32 * 73;
    __shared__ __align__(16) uint32_t s_f_words[MAX_STAGED_Q8K_WORDS];
    const int n_words = nb_ff * 73;
    const bool staged = (nb_ff <= 32);
    if (staged) {
        const uint32_t* src_w = reinterpret_cast<const uint32_t*>(gf);
        for (int i = tid; i < n_words; i += 256) s_f_words[i] = src_w[i];
        __syncthreads();
    }
    const block_q8_K* f_use = staged ? reinterpret_cast<const block_q8_K*>(s_f_words) : gf;

    const int sub = lane & 7;
    const int b0 = lane >> 3;
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int r = r_base + warp * 4 + step;
        if (r < n_embd) {
            const uint8_t* dr = dp + (size_t)r * down_rb;
            float acc = 0.0f;
            for (int b = b0; b < nb_ff; b += 4) {
                int u[8]; float d8 = 0.0f; int bsum32 = 0;
                load_q8_k_sub_words(f_use + b, sub, u, d8, bsum32);
                if constexpr (TY_D > 0) {
                    acc += dot_sub_mm3_q8k<TY_D>(dr, b, sub, u, d8, bsum32);
                } else {
                    acc += dot_sub_mm3_q8k_dyn(down_type_dyn, dr, b, sub, u, d8, bsum32);
                }
            }
            const float sum = warp_reduce_sum(acc);
            if (lane == 0) {
                expert_out[((size_t)m * topk + slot) * n_embd + r] = sum;
            }
        }
    }
}

__global__ void __launch_bounds__(256) mm3_moe_combine_kernel(
    const float* __restrict__ res_opt,
    const float* __restrict__ expert_out,
    const float* __restrict__ weights,
    const int32_t* __restrict__ expert_ids,
    const uint64_t* __restrict__ vram_ptrs_opt,
    const float* __restrict__ shared_opt,
    int topk,
    int n_embd,
    float* __restrict__ out
) {
    const int t = (int)blockIdx.x;
    for (int d = (int)threadIdx.x; d < n_embd; d += 256) {
        float acc = res_opt ? res_opt[(size_t)t * n_embd + d] : 0.0f;
        for (int s = 0; s < topk; ++s) {
            const int e = expert_ids[(size_t)t * topk + s];
            if (vram_ptrs_opt == nullptr || vram_ptrs_opt[e] != 0ull) {
                acc = fmaf(weights[(size_t)t * topk + s], expert_out[((size_t)t * topk + s) * n_embd + d], acc);
            }
        }
        if (shared_opt != nullptr) acc += shared_opt[(size_t)t * n_embd + d];
        out[(size_t)t * n_embd + d] = acc;
    }
}

struct Mm3DecodePlan {
    uint64_t base_ptrs[32];
    int32_t grp_off[33];
    int32_t entry_tok[32];
    float entry_wt[32];
    int32_t _pad;
};

struct Mm3IqSub {
    int w[8];
    float dw;
    int ls0;
    int ls1;
    int sgn;
};

template <int W_TYPE>
__device__ __forceinline__ void decode_sub_mm3_q8k(
    const uint8_t* w_row,
    int b,
    int sub,
    Mm3IqSub& s
) {
    if constexpr (W_TYPE == GGML_TYPE_IQ2_XXS) {
        const block_iq2_xxs* blk = reinterpret_cast<const block_iq2_xxs*>(w_row) + b;
        s.dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        const uint32_t aux0 = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
        const uint32_t aux1 = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
        s.ls0 = 2 * (int)(aux1 >> 28) + 1;
        s.ls1 = 0;
        s.sgn = 1;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g_idx = (aux0 >> (8 * l)) & 0xFFu;
            const uint32_t s7 = (aux1 >> (7 * l)) & 0x7Fu;
            const uint32_t sbyte = s7 | ((uint32_t)(__popc(s7) & 1) << 7);
            const uint64_t g64 = d_iq2xxs_grid[g_idx];
            const unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
            const unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
            s.w[2 * l]     = (int)__vsub4((uint32_t)g64 ^ s0, s0);
            s.w[2 * l + 1] = (int)__vsub4((uint32_t)(g64 >> 32) ^ s1, s1);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_XS) {
        const block_iq2_xs* blk = reinterpret_cast<const block_iq2_xs*>(w_row) + b;
        s.dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        const uint32_t q01 = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
        const uint32_t q23 = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
        const uint8_t sc_byte = blk->scales[sub];
        s.ls0 = 2 * (int)(sc_byte & 0x0Fu) + 1;
        s.ls1 = 2 * (int)(sc_byte >> 4) + 1;
        s.sgn = 1;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qw = (l < 2) ? q01 : q23;
            const uint32_t q16 = (l & 1) ? (qw >> 16) : (qw & 0xFFFFu);
            const uint32_t g_idx = q16 & 511u;
            const uint32_t s7 = q16 >> 9;
            const uint32_t sbyte = s7 | ((uint32_t)(__popc(s7) & 1) << 7);
            const uint64_t g64 = d_iq2xs_grid[g_idx];
            const unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
            const unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
            s.w[2 * l]     = (int)__vsub4((uint32_t)g64 ^ s0, s0);
            s.w[2 * l + 1] = (int)__vsub4((uint32_t)(g64 >> 32) ^ s1, s1);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
        const block_iq2_s* blk = reinterpret_cast<const block_iq2_s*>(w_row) + b;
        s.dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        const uint32_t qs_packed = (uint32_t)load_i32_align2(blk->qs + 4 * sub);
        const uint32_t signs_packed = (uint32_t)load_i32_align2(blk->qs + 32 + 4 * sub);
        const uint32_t qh = (uint32_t)blk->qh[sub];
        const uint8_t sc_byte = blk->scales[sub];
        s.ls0 = 2 * (int)(sc_byte & 0x0Fu) + 1;
        s.ls1 = 2 * (int)(sc_byte >> 4) + 1;
        s.sgn = 1;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int sh = (l0 >> 1) * 8;
            const uint32_t q_b = (qs_packed >> sh) & 0xFFu;
            const uint32_t sbyte = (signs_packed >> sh) & 0xFFu;
            const uint32_t grid_idx = q_b | ((qh << (8 - l0)) & 0x300u);
            const uint64_t g64 = d_iq2s_grid[grid_idx];
            const unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
            const unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
            s.w[l0]     = (int)__vsub4((uint32_t)g64 ^ s0, s0);
            s.w[l0 + 1] = (int)__vsub4((uint32_t)(g64 >> 32) ^ s1, s1);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
        const block_iq3_xxs* blk = reinterpret_cast<const block_iq3_xxs*>(w_row) + b;
        s.dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        uint32_t q3_words[2];
        q3_words[0] = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
        q3_words[1] = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
        const uint8_t* q3 = reinterpret_cast<const uint8_t*>(q3_words);
        const uint32_t aux32 = (uint32_t)load_i32_align2(blk->qs + 64 + 4 * sub);
        s.ls0 = 2 * (int)(aux32 >> 28) + 1;
        s.ls1 = 0;
        s.sgn = 1;
        #pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t g0 = d_iq3xxs_grid[q3[l0 + 0]];
            const uint32_t g1 = d_iq3xxs_grid[q3[l0 + 1]];
            const uint32_t s7 = (aux32 >> (7 * (l0 >> 1))) & 0x7Fu;
            const uint32_t ks = (s7 | ((uint32_t)(__popc(s7) & 1) << 7)) * 0x01010101u;
            const unsigned s0 = __vcmpne4(ks & 0x08040201u, 0u);
            const unsigned s1 = __vcmpne4(ks & 0x80402010u, 0u);
            s.w[l0]     = (int)__vsub4(g0 ^ s0, s0);
            s.w[l0 + 1] = (int)__vsub4(g1 ^ s1, s1);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ1_S) {
        const block_iq1_s* blk = reinterpret_cast<const block_iq1_s*>(w_row) + b;
        s.dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
        const uint16_t qh = load_u16_align2(&blk->qh[sub]);
        const uint32_t qs4 = (uint32_t)load_i32_align2(blk->qs + 4 * sub);
        s.ls0 = 2 * (int)((qh >> 12) & 7u) + 1;
        s.ls1 = 0;
        s.sgn = (qh & 0x8000u) ? -1 : 1;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t g_idx = ((qs4 >> (8 * l)) & 0xFFu) | ((((uint32_t)qh >> (3 * l)) & 7u) << 8);
            const uint64_t g64 = d_iq1s_grid[g_idx];
            s.w[2 * l]     = (int)(uint32_t)g64;
            s.w[2 * l + 1] = (int)(uint32_t)(g64 >> 32);
        }
    }
}

template <int W_TYPE>
__device__ __forceinline__ float apply_sub_mm3_q8k(
    const Mm3IqSub& s,
    const int (&u)[8],
    float d8,
    int bsum32
) {
    if constexpr (W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_IQ2_S) {
        int sumi0 = 0, sumi1 = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) sumi0 = dp4a_packed(s.w[k], u[k], sumi0);
        #pragma unroll
        for (int k = 4; k < 8; ++k) sumi1 = dp4a_packed(s.w[k], u[k], sumi1);
        return s.dw * d8 * 0.125f * (float)(sumi0 * s.ls0 + sumi1 * s.ls1);
    } else {
        int sumi = 0;
        #pragma unroll
        for (int k = 0; k < 8; ++k) sumi = dp4a_packed(s.w[k], u[k], sumi);
        if constexpr (W_TYPE == GGML_TYPE_IQ2_XXS) {
            return s.dw * d8 * 0.125f * (float)(sumi * s.ls0);
        } else if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
            return s.dw * d8 * 0.25f * (float)(sumi * s.ls0);
        } else if constexpr (W_TYPE == GGML_TYPE_IQ1_S) {
            return s.dw * d8 * 0.125f * (float)(s.ls0 * (8 * sumi + s.sgn * bsum32));
        } else {
            return 0.0f;
        }
    }
}

template <int TY_G, int TY_U, int NT_CAP>
__global__ void __launch_bounds__(256) mm3_moe_grp_gate_up_f32_kernel(
    int gate_type_dyn,
    int up_type_dyn,
    const Mm3DecodePlan* __restrict__ plan,
    size_t u_off,
    size_t gate_rb,
    size_t up_rb,
    const block_q8_K* __restrict__ x_q8k,
    float* __restrict__ inter_f32,
    int n_embd,
    int n_ff,
    float alpha,
    float limit
) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int r = (int)blockIdx.x * 8 + warp;
    if (r >= n_ff) return;

    const int g = (int)blockIdx.y;
    const uint64_t base = plan->base_ptrs[g];
    if (base == 0ull) return;
    const int p_start = plan->grp_off[g];
    const int p_end = plan->grp_off[g + 1];
    if (p_start >= p_end) return;

    const uint8_t* gr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(base)) + (size_t)r * gate_rb;
    const uint8_t* ur = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(base + u_off)) + (size_t)r * up_rb;
    const int nb_x = n_embd >> 8;
    const int sub = lane & 7;
    const int b0 = lane >> 3;

    for (int p0 = p_start; p0 < p_end; p0 += NT_CAP) {
        const int cnt = min(NT_CAP, p_end - p0);
        int toks[NT_CAP];
        #pragma unroll
        for (int k = 0; k < NT_CAP; ++k) {
            toks[k] = (NT_CAP == 1 || k < cnt) ? plan->entry_tok[p0 + k] : 0;
        }
        float ag[NT_CAP], au[NT_CAP];
        #pragma unroll
        for (int k = 0; k < NT_CAP; ++k) {
            ag[k] = 0.0f;
            au[k] = 0.0f;
        }
        if constexpr (NT_CAP > 1 && TY_G > 0 && TY_U > 0) {
            for (int b = b0; b < nb_x; b += 4) {
                {
                    Mm3IqSub sg;
                    decode_sub_mm3_q8k<TY_G>(gr, b, sub, sg);
                    #pragma unroll
                    for (int k = 0; k < NT_CAP; ++k) {
                        if (k < cnt) {
                            int u[8];
                            float d8 = 0.0f;
                            int bsum32 = 0;
                            load_q8_k_sub_words(x_q8k + (size_t)toks[k] * nb_x + b, sub, u, d8, bsum32);
                            ag[k] += apply_sub_mm3_q8k<TY_G>(sg, u, d8, bsum32);
                        }
                    }
                }
                {
                    Mm3IqSub su;
                    decode_sub_mm3_q8k<TY_U>(ur, b, sub, su);
                    #pragma unroll
                    for (int k = 0; k < NT_CAP; ++k) {
                        if (k < cnt) {
                            int u[8];
                            float d8 = 0.0f;
                            int bsum32 = 0;
                            load_q8_k_sub_words(x_q8k + (size_t)toks[k] * nb_x + b, sub, u, d8, bsum32);
                            au[k] += apply_sub_mm3_q8k<TY_U>(su, u, d8, bsum32);
                        }
                    }
                }
            }
        } else {
            for (int b = b0; b < nb_x; b += 4) {
                #pragma unroll
                for (int k = 0; k < NT_CAP; ++k) {
                    if (NT_CAP == 1 || k < cnt) {
                        int u[8];
                        float d8 = 0.0f;
                        int bsum32 = 0;
                        load_q8_k_sub_words(x_q8k + (size_t)toks[k] * nb_x + b, sub, u, d8, bsum32);
                        if constexpr (TY_G > 0 && TY_U > 0) {
                            ag[k] += dot_sub_mm3_q8k<TY_G>(gr, b, sub, u, d8, bsum32);
                            au[k] += dot_sub_mm3_q8k<TY_U>(ur, b, sub, u, d8, bsum32);
                        } else {
                            ag[k] += dot_sub_mm3_q8k_dyn(gate_type_dyn, gr, b, sub, u, d8, bsum32);
                            au[k] += dot_sub_mm3_q8k_dyn(up_type_dyn, ur, b, sub, u, d8, bsum32);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < NT_CAP; ++k) {
            if (NT_CAP == 1 || k < cnt) {
                const float vg = warp_reduce_sum(ag[k]);
                const float vu = warp_reduce_sum(au[k]);
                if (lane == 0) {
                    inter_f32[(size_t)(p0 + k) * n_ff + r] = swoai_val(vg, vu, alpha, limit);
                }
            }
        }
    }
}

template <int TY_D, int NT_CAP>
__global__ void __launch_bounds__(256) mm3_moe_grp_down_q8k_kernel(
    int down_type_dyn,
    const Mm3DecodePlan* __restrict__ plan,
    size_t d_off,
    size_t down_rb,
    const block_q8_K* __restrict__ inter_q8k,
    float* __restrict__ expert_out,
    int n_ff,
    int n_embd
) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int r = (int)blockIdx.x * 8 + warp;
    if (r >= n_embd) return;

    const int g = (int)blockIdx.y;
    const uint64_t base = plan->base_ptrs[g];
    if (base == 0ull) return;
    const int p_start = plan->grp_off[g];
    const int p_end = plan->grp_off[g + 1];
    if (p_start >= p_end) return;

    const uint8_t* dr = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(base + d_off)) + (size_t)r * down_rb;
    const int nb_ff = n_ff >> 8;
    const int sub = lane & 7;
    const int b0 = lane >> 3;

    for (int p0 = p_start; p0 < p_end; p0 += NT_CAP) {
        const int cnt = min(NT_CAP, p_end - p0);
        float acc[NT_CAP];
        #pragma unroll
        for (int k = 0; k < NT_CAP; ++k) acc[k] = 0.0f;
        if constexpr (NT_CAP > 1 && TY_D > 0) {
            for (int b = b0; b < nb_ff; b += 4) {
                Mm3IqSub sd;
                decode_sub_mm3_q8k<TY_D>(dr, b, sub, sd);
                #pragma unroll
                for (int k = 0; k < NT_CAP; ++k) {
                    if (k < cnt) {
                        int u[8];
                        float d8 = 0.0f;
                        int bsum32 = 0;
                        load_q8_k_sub_words(inter_q8k + (size_t)(p0 + k) * nb_ff + b, sub, u, d8, bsum32);
                        acc[k] += apply_sub_mm3_q8k<TY_D>(sd, u, d8, bsum32);
                    }
                }
            }
        } else {
            for (int b = b0; b < nb_ff; b += 4) {
                #pragma unroll
                for (int k = 0; k < NT_CAP; ++k) {
                    if (NT_CAP == 1 || k < cnt) {
                        int u[8];
                        float d8 = 0.0f;
                        int bsum32 = 0;
                        load_q8_k_sub_words(inter_q8k + (size_t)(p0 + k) * nb_ff + b, sub, u, d8, bsum32);
                        if constexpr (TY_D > 0) {
                            acc[k] += dot_sub_mm3_q8k<TY_D>(dr, b, sub, u, d8, bsum32);
                        } else {
                            acc[k] += dot_sub_mm3_q8k_dyn(down_type_dyn, dr, b, sub, u, d8, bsum32);
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int k = 0; k < NT_CAP; ++k) {
            if (NT_CAP == 1 || k < cnt) {
                const float sum = warp_reduce_sum(acc[k]);
                if (lane == 0) {
                    expert_out[(size_t)(p0 + k) * n_embd + r] = sum;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256) mm3_moe_grp_combine_kernel(
    const float* __restrict__ res_opt,
    const float* __restrict__ shared_opt,
    const float* __restrict__ expert_out,
    const Mm3DecodePlan* __restrict__ plan0,
    int n_entries0,
    const Mm3DecodePlan* __restrict__ plan1_opt,
    int n_entries1,
    const float* __restrict__ cpu_out_opt,
    int n_embd,
    float* __restrict__ out
) {
    __shared__ int32_t s_tok[64];
    __shared__ float s_wt[64];
    const int tid = (int)threadIdx.x;
    const int e0 = (plan0 != nullptr) ? min(n_entries0, 32) : 0;
    const int e1 = (plan1_opt != nullptr) ? min(n_entries1, 32) : 0;
    if (tid < e0) {
        s_tok[tid] = plan0->entry_tok[tid];
        s_wt[tid] = plan0->entry_wt[tid];
    }
    if (tid < e1 && e0 + tid < 64) {
        s_tok[e0 + tid] = plan1_opt->entry_tok[tid];
        s_wt[e0 + tid] = plan1_opt->entry_wt[tid];
    }
    __syncthreads();

    const int t = (int)blockIdx.x;
    const int total = min(e0 + e1, 64);
    for (int d = tid; d < n_embd; d += 256) {
        float acc = (res_opt != nullptr) ? res_opt[(size_t)t * n_embd + d] : 0.0f;
        if (shared_opt != nullptr) acc += shared_opt[(size_t)t * n_embd + d];
        if (cpu_out_opt != nullptr) acc += cpu_out_opt[(size_t)t * n_embd + d];
        for (int p = 0; p < total; ++p) {
            if (s_tok[p] == t) {
                acc = fmaf(s_wt[p], expert_out[(size_t)p * n_embd + d], acc);
            }
        }
        out[(size_t)t * n_embd + d] = acc;
    }
}

// -----------------------------------------------------------------------------
// 4. Zero-cuBLAS Custom Ampere (sm_86) BF16 Tensor-Core GEMM & MXFP8 & DSpark
// -----------------------------------------------------------------------------

__device__ __forceinline__ void mma16816_f32_bf16(
    float* c,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
#else
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#endif
}

__device__ __forceinline__ uint32_t mm3_smem_u32(const void* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ int mm3_swz32(int r, int col) {
    return col ^ (((r >> 1) & 3) << 3);
}

__device__ __forceinline__ void mm3_ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr
) {
    const uint32_t addr = mm3_smem_u32(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

template <int ROWS, int THREADS>
__device__ __forceinline__ void mm3_load_tile_bf16_swz32(
    uint16_t* __restrict__ s_dst,
    const uint16_t* __restrict__ g_src,
    int ld,
    int row0,
    int limit,
    int K,
    int k0,
    int tid,
    bool vec16_ok
) {
    constexpr int VECS = ROWS * 4;
    #pragma unroll
    for (int idx = tid; idx < VECS; idx += THREADS) {
        const int r = idx >> 2;
        const int c8 = (idx & 3) * 8;
        const int gr = row0 + r;
        const int gc = k0 + c8;
        uint16_t* sp = &s_dst[r * 32 + mm3_swz32(r, c8)];
        const bool row_ok = (gr < limit);
        if (vec16_ok && row_ok && (gc + 8 <= K)) {
            const uint32_t sp_u = mm3_smem_u32(sp);
            const void* gp = g_src + (size_t)gr * ld + gc;
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"(sp_u), "l"(gp));
        } else if (row_ok && (gc < K)) {
            const uint16_t* gp = g_src + (size_t)gr * ld + gc;
            uint16_t tmp[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            const int n_rem = K - gc;
            const int n_cpy = (n_rem < 8) ? n_rem : 8;
            for (int i = 0; i < n_cpy; ++i) tmp[i] = gp[i];
            *reinterpret_cast<uint4*>(sp) = *reinterpret_cast<const uint4*>(tmp);
        } else {
            *reinterpret_cast<uint4*>(sp) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

template <int MI, int NJ>
__device__ __forceinline__ void mm3_store_mma_epilogue_ld(
    float (&acc)[MI][NJ][4],
    float* __restrict__ C,
    int ldC,
    int M, int N, int bm, int bn, int wm, int wn,
    int gid, int col_pair, float alpha, float beta
) {
    const bool use_vec2 = ((ldC & 1) == 0) && ((reinterpret_cast<uintptr_t>(C) & 7u) == 0u);
    #pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
        const int r0 = bm + wm + mi * 16 + gid;
        const int r1 = r0 + 8;
        #pragma unroll
        for (int nj = 0; nj < NJ; ++nj) {
            const int c0 = bn + wn + nj * 8 + col_pair;
            const int c1 = c0 + 1;
            if (r0 < M) {
                if (use_vec2 && c1 < N) {
                    float2 v;
                    if (beta == 0.0f) {
                        v = make_float2(alpha * acc[mi][nj][0], alpha * acc[mi][nj][1]);
                    } else {
                        const float2 old = *reinterpret_cast<const float2*>(&C[(size_t)r0 * ldC + c0]);
                        v = make_float2(alpha * acc[mi][nj][0] + beta * old.x, alpha * acc[mi][nj][1] + beta * old.y);
                    }
                    *reinterpret_cast<float2*>(&C[(size_t)r0 * ldC + c0]) = v;
                } else {
                    if (c0 < N) C[(size_t)r0 * ldC + c0] = (beta == 0.0f) ? (alpha * acc[mi][nj][0]) : (alpha * acc[mi][nj][0] + beta * C[(size_t)r0 * ldC + c0]);
                    if (c1 < N) C[(size_t)r0 * ldC + c1] = (beta == 0.0f) ? (alpha * acc[mi][nj][1]) : (alpha * acc[mi][nj][1] + beta * C[(size_t)r0 * ldC + c1]);
                }
            }
            if (r1 < M) {
                if (use_vec2 && c1 < N) {
                    float2 v;
                    if (beta == 0.0f) {
                        v = make_float2(alpha * acc[mi][nj][2], alpha * acc[mi][nj][3]);
                    } else {
                        const float2 old = *reinterpret_cast<const float2*>(&C[(size_t)r1 * ldC + c0]);
                        v = make_float2(alpha * acc[mi][nj][2] + beta * old.x, alpha * acc[mi][nj][3] + beta * old.y);
                    }
                    *reinterpret_cast<float2*>(&C[(size_t)r1 * ldC + c0]) = v;
                } else {
                    if (c0 < N) C[(size_t)r1 * ldC + c0] = (beta == 0.0f) ? (alpha * acc[mi][nj][2]) : (alpha * acc[mi][nj][2] + beta * C[(size_t)r1 * ldC + c0]);
                    if (c1 < N) C[(size_t)r1 * ldC + c1] = (beta == 0.0f) ? (alpha * acc[mi][nj][3]) : (alpha * acc[mi][nj][3] + beta * C[(size_t)r1 * ldC + c1]);
                }
            }
        }
    }
}

template <int BM, int BN, int WM_WARPS, int WN_WARPS, int MIN_CTAS>
__global__ void __launch_bounds__(WM_WARPS * WN_WARPS * 32, MIN_CTAS)
mm3_gemm_bf16_mma_kernel(
    const uint16_t* __restrict__ A,
    int ldA,
    const uint16_t* __restrict__ B,
    int ldB,
    float* __restrict__ C,
    int ldC,
    int M, int N, int K, float alpha, float beta
) {
    constexpr int BK = 32;
    constexpr int THREADS = WM_WARPS * WN_WARPS * 32;
    constexpr int MI = (BM / WM_WARPS) / 16;
    constexpr int NJ = (BN / WN_WARPS) / 8;

    __shared__ __align__(16) uint16_t sA[3][BM * BK];
    __shared__ __align__(16) uint16_t sB[3][BN * BK];

    const int bm = (int)blockIdx.y * BM;
    const int bn = (int)blockIdx.x * BN;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = (warp / WN_WARPS) * (MI * 16);
    const int wn = (warp % WN_WARPS) * (NJ * 8);

    const bool a_vec = ((ldA & 7) == 0) && ((reinterpret_cast<uintptr_t>(A) & 15u) == 0u);
    const bool b_vec = ((ldB & 7) == 0) && ((reinterpret_cast<uintptr_t>(B) & 15u) == 0u);

    float acc[MI][NJ][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int j = 0; j < NJ; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    auto load_tile = [&](int stage, int k0) {
        mm3_load_tile_bf16_swz32<BM, THREADS>(sA[stage], A, ldA, bm, M, K, k0, tid, a_vec);
        mm3_load_tile_bf16_swz32<BN, THREADS>(sB[stage], B, ldB, bn, N, K, k0, tid, b_vec);
        asm volatile("cp.async.commit_group;\n" ::);
    };

    const int n_tiles = (K + BK - 1) / BK;
    if (n_tiles <= 0) return;
    load_tile(0, 0);
    if (n_tiles > 1) load_tile(1, BK);

    for (int kt = 0; kt < n_tiles; ++kt) {
        if (kt + 2 < n_tiles) {
            load_tile((kt + 2) % 3, (kt + 2) * BK);
            asm volatile("cp.async.wait_group 2;\n" ::);
        } else if (kt + 1 < n_tiles) {
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();

        const int stage = kt % 3;
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int k_off = kk * 16;
            uint32_t a_frag[MI][4];
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const int row_a = wm + mi * 16 + (lane & 15);
                const int col_a = k_off + ((lane >> 4) * 8);
                mm3_ldmatrix_x4(
                    a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    &sA[stage][row_a * BK + mm3_swz32(row_a, col_a)]
                );
            }
            #pragma unroll
            for (int np = 0; np < NJ / 2; ++np) {
                const int row_b = wn + np * 16 + ((lane >> 4) * 8) + (lane & 7);
                const int col_b = k_off + ((lane >> 3) & 1) * 8;
                uint32_t b0_0, b0_1, b1_0, b1_1;
                mm3_ldmatrix_x4(
                    b0_0, b0_1, b1_0, b1_1,
                    &sB[stage][row_b * BK + mm3_swz32(row_b, col_b)]
                );
                #pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    mma16816_f32_bf16(
                        acc[mi][np * 2 + 0],
                        a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                        b0_0, b0_1
                    );
                    mma16816_f32_bf16(
                        acc[mi][np * 2 + 1],
                        a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                        b1_0, b1_1
                    );
                }
            }
        }
        __syncthreads();
    }

    mm3_store_mma_epilogue_ld<MI, NJ>(acc, C, ldC, M, N, bm, bn, wm, wn, lane >> 2, (lane & 3) * 2, alpha, beta);
}

__device__ __forceinline__ float fp8_e4m3_to_f32_safe(uint8_t b) {
    __nv_fp8_e4m3 f;
    f.__x = b;
    const float v = (float)f;
    return isfinite(v) ? v : 0.0f;
}

__device__ __forceinline__ float ue8m0_to_f32(uint8_t e) {
    return (e == 0u) ? ldexpf(1.0f, -127) : __uint_as_float((uint32_t)e << 23);
}

__global__ void __launch_bounds__(256) mm3_mxfp8_to_bf16_kernel(
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ scale,
    int K,
    uint16_t* __restrict__ out
) {
    const size_t n = (size_t)blockIdx.y;
    const size_t ks = (size_t)(K >> 5);
    for (int k = (int)blockIdx.x * 256 + (int)threadIdx.x; k < K; k += (int)gridDim.x * 256) {
        const uint8_t eb = scale[n * ks + (size_t)(k >> 5)];
        const float sc = ue8m0_to_f32(eb);
        const float v = fp8_e4m3_to_f32_safe(w[n * (size_t)K + k]) * sc;
        out[n * (size_t)K + k] = f2bf(v);
    }
}

template <int R>
__global__ void __launch_bounds__(256) mm3_mx_rows_kernel(
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ sc,
    int N, int K, int kc,
    const float* __restrict__ x, int ldX,
    float* __restrict__ y, int ldY,
    float beta,
    float* __restrict__ part
) {
    __shared__ float s_tab[256];
    __shared__ __align__(16) float s_x[R][512];
    s_tab[threadIdx.x] = fp8_e4m3_to_f32_safe((uint8_t)threadIdx.x);

    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int n0 = (int)blockIdx.x * 32 + warp * 4;
    const int ka = (int)blockIdx.y * kc;
    const int kb = min(K, ka + kc);
    const size_t ks = (size_t)(K >> 5);
    const bool x_vec4 = ((ldX & 3) == 0) && ((reinterpret_cast<uintptr_t>(x) & 15u) == 0u);

    float acc[4][R];
    #pragma unroll
    for (int o = 0; o < 4; ++o)
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[o][r] = 0.0f;

    for (int k0 = ka; k0 < kb; k0 += 512) {
        __syncthreads();
        const int lim = min(512, kb - k0);
        for (int i = (int)threadIdx.x; i < R * 128; i += 256) {
            const int r = i >> 7;
            const int c = (i & 127) * 4;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (x_vec4 && c + 3 < lim) {
                v = *reinterpret_cast<const float4*>(x + (size_t)r * ldX + k0 + c);
            } else if (c < lim) {
                const float* xp = x + (size_t)r * ldX + k0 + c;
                v.x = xp[0];
                if (c + 1 < lim) v.y = xp[1];
                if (c + 2 < lim) v.z = xp[2];
                if (c + 3 < lim) v.w = xp[3];
            }
            *reinterpret_cast<float4*>(&s_x[r][c]) = v;
        }
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int kk = j * 128 + lane * 4;
            if (kk >= lim) continue;
            #pragma unroll
            for (int o = 0; o < 4; ++o) {
                const int nn = n0 + o;
                if (nn >= N) continue;
                const uint32_t b = *reinterpret_cast<const uint32_t*>(w + (size_t)nn * K + k0 + kk);
                const float scale = ue8m0_to_f32(sc[(size_t)nn * ks + (size_t)((k0 + kk) >> 5)]);
                const float w0 = s_tab[b & 0xFFu] * scale;
                const float w1 = s_tab[(b >> 8) & 0xFFu] * scale;
                const float w2 = s_tab[(b >> 16) & 0xFFu] * scale;
                const float w3 = s_tab[b >> 24] * scale;
                #pragma unroll
                for (int r = 0; r < R; ++r) {
                    const float4 xv = *reinterpret_cast<const float4*>(&s_x[r][kk]);
                    acc[o][r] = fmaf(w0, xv.x, fmaf(w1, xv.y, fmaf(w2, xv.z, fmaf(w3, xv.w, acc[o][r]))));
                }
            }
        }
    }

    #pragma unroll
    for (int o = 0; o < 4; ++o) {
        const int nn = n0 + o;
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const float v = warp_reduce_sum(acc[o][r]);
            if (lane == 0 && nn < N) {
                if (part != nullptr) {
                    part[((size_t)blockIdx.y * R + r) * N + nn] = v;
                } else {
                    float* yp = y + (size_t)r * ldY + nn;
                    *yp = (beta != 0.0f) ? (beta * (*yp) + v) : v;
                }
            }
        }
    }
}

__global__ void __launch_bounds__(256) mm3_mx_reduce_kernel(
    const float* __restrict__ part,
    int ks, int rows, int N,
    float* __restrict__ y, int ldY,
    float beta
) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i >= rows * N) return;
    const int r = i / N;
    const int n = i - r * N;
    float v = 0.0f;
    for (int z = 0; z < ks; ++z) v += part[((size_t)z * rows + r) * N + n];
    float* yp = y + (size_t)r * ldY + n;
    *yp = (beta != 0.0f) ? (beta * (*yp) + v) : v;
}

constexpr int kMcKC = 1024;

__device__ __forceinline__ float mm3_mc_raw(uint32_t b) {
    return ((b & 0x7Fu) == 0x7Fu)
        ? 0.0f
        : __uint_as_float(((b & 0x80u) << 24) | ((b & 0x7Fu) << 20));
}

template <int R>
__global__ void __launch_bounds__(256) mm3_mxc_kernel(
    const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ sc,
    int N, int K,
    const float* __restrict__ x, int ldX,
    float* __restrict__ y, int ldY,
    float beta,
    float* __restrict__ part
) {
    __shared__ __align__(16) float s_x[R][kMcKC];
    __shared__ float s_red[8][32][R];
    const int lane = (int)threadIdx.x & 31;
    const int c = (int)threadIdx.x >> 5;
    const int k0 = (int)blockIdx.y * kMcKC;
    const int n = (int)blockIdx.x * 32 + lane;

    for (int i = (int)threadIdx.x; i < R * (kMcKC / 4); i += 256) {
        const int r = i / (kMcKC / 4);
        const int q = (i % (kMcKC / 4)) * 4;
        *reinterpret_cast<float4*>(&s_x[r][q]) = *reinterpret_cast<const float4*>(x + (size_t)r * ldX + k0 + q);
    }
    __syncthreads();

    const bool ok = (n < N);
    const uint8_t* wr = w + (size_t)(ok ? n : 0) * K + k0 + c * 32;
    const uint8_t* sr = sc + (size_t)(ok ? n : 0) * (K / 32) + k0 / 32 + c;
    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    uint4 a0 = *reinterpret_cast<const uint4*>(wr);
    uint4 a1 = *reinterpret_cast<const uint4*>(wr + 16);
    uint32_t e = sr[0];

    #pragma unroll 1
    for (int st = 0; st < kMcKC / 256; ++st) {
        const uint4 b0 = a0;
        const uint4 b1 = a1;
        const uint32_t eb = e;
        if (st + 1 < kMcKC / 256) {
            const uint8_t* wr_n = wr + (st + 1) * 256;
            a0 = *reinterpret_cast<const uint4*>(wr_n);
            a1 = *reinterpret_cast<const uint4*>(wr_n + 16);
            e = sr[(st + 1) * 8];
        }
        const uint32_t wd[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        float p[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) p[r] = 0.0f;
        const int kk = st * 256 + c * 32;
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            const float f0 = mm3_mc_raw(wd[q] & 0xFFu);
            const float f1 = mm3_mc_raw((wd[q] >> 8) & 0xFFu);
            const float f2 = mm3_mc_raw((wd[q] >> 16) & 0xFFu);
            const float f3 = mm3_mc_raw(wd[q] >> 24);
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const float4 xv = *reinterpret_cast<const float4*>(&s_x[r][kk + q * 4]);
                p[r] = fmaf(f0, xv.x, p[r]);
                p[r] = fmaf(f1, xv.y, p[r]);
                p[r] = fmaf(f2, xv.z, p[r]);
                p[r] = fmaf(f3, xv.w, p[r]);
            }
        }
        const float s2 = eb ? __uint_as_float(eb << 23) : __uint_as_float(0x00400000u);
        const float p_scale = __uint_as_float(247u << 23);
        #pragma unroll
        for (int r = 0; r < R; ++r) acc[r] = fmaf(s2, p[r] * p_scale, acc[r]);
    }

    #pragma unroll
    for (int r = 0; r < R; ++r) s_red[c][lane][r] = acc[r];
    __syncthreads();

    for (int i = (int)threadIdx.x; i < 32 * R; i += 256) {
        const int l = i / R;
        const int r = i - l * R;
        const int nn = (int)blockIdx.x * 32 + l;
        if (nn >= N) continue;
        float v = s_red[0][l][r];
        #pragma unroll
        for (int z = 1; z < 8; ++z) v += s_red[z][l][r];
        if (part != nullptr) {
            part[((size_t)blockIdx.y * R + r) * N + nn] = v;
        } else {
            float* yp = y + (size_t)r * ldY + nn;
            *yp = (beta != 0.0f) ? (beta * (*yp) + v) : v;
        }
    }
}

// Coalesced multi-job Q6_K & Q8_0 decode/verify GEMV (M = 1..8) reading packed block_q8_1 safely.
struct Mm3GvJob {
    const uint8_t* w;
    size_t ld;
    int n;
    int k;
    int ncol;
    const block_q8_1* x_q8;
    float* y;
};

struct Mm3GvJobs {
    Mm3GvJob j[3];
};

struct Mm3Chunk8 {
    uint2 v[5];
    __device__ __forceinline__ void load(const uint8_t* a) {
        const uint2* p = reinterpret_cast<const uint2*>(reinterpret_cast<uintptr_t>(a) & ~static_cast<uintptr_t>(7));
        #pragma unroll
        for (int i = 0; i < 5; ++i) v[i] = p[i];
    }
    __device__ __forceinline__ void words(uintptr_t a, uint32_t* w) const {
        const uint32_t h[10] = {v[0].x, v[0].y, v[1].x, v[1].y, v[2].x, v[2].y, v[3].x, v[3].y, v[4].x, v[4].y};
        const bool o = (a & 4u) != 0;
        const uint32_t sh = 8u * static_cast<uint32_t>(a & 3u);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t lo = o ? h[t + 1] : h[t];
            const uint32_t hi = o ? h[t + 2] : h[t + 1];
            w[t] = (sh != 0u) ? __funnelshift_r(lo, hi, sh) : lo;
        }
    }
};

template <int W_TYPE>
struct Mm3GvRaw;

template <>
struct Mm3GvRaw<GGML_TYPE_Q6_K> {
    Mm3Chunk8 l, h;
    uint32_t sc, d;
    uintptr_t al, ah;
    __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int kb) {
        const uint8_t* b = row + static_cast<size_t>(kb >> 3) * 210;
        const int ib = kb & 7;
        const int hf = ib >> 2;
        const int qt = ib & 3;
        const uint8_t* pl = b + 64 * hf + 32 * (qt & 1);
        const uint8_t* ph = b + 128 + 32 * hf;
        al = reinterpret_cast<uintptr_t>(pl);
        ah = reinterpret_cast<uintptr_t>(ph);
        l.load(pl);
        h.load(ph);
        sc = *reinterpret_cast<const uint16_t*>(b + 192 + 8 * hf + 2 * qt);
        d = *reinterpret_cast<const uint16_t*>(b + 208);
    }
    __device__ __forceinline__ void decode(int kb, int (&q)[8], float& dw, int& s0, int& s1) const {
        const int qt = kb & 3;
        uint32_t ql[8], qh[8];
        l.words(al, ql);
        h.words(ah, qh);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const uint32_t v = ((ql[t] >> (4 * (qt >> 1))) & 0x0F0F0F0Fu) |
                               (((qh[t] >> (2 * qt)) & 0x03030303u) << 4);
            const uint32_t lo = (~v) & 0x20202020u;
            q[t] = static_cast<int>((v ^ 0x20202020u) | (lo << 1) | (lo << 2));
        }
        s0 = static_cast<int>(static_cast<int8_t>(sc & 255u));
        s1 = static_cast<int>(static_cast<int8_t>(sc >> 8));
        dw = fp16_bits_to_f32(static_cast<uint16_t>(d));
    }
};

template <>
struct Mm3GvRaw<GGML_TYPE_Q8_0> {
    uint32_t w[9], d;
    uintptr_t a;
    __device__ __forceinline__ void load(const uint8_t* __restrict__ row, int kb) {
        const uint8_t* b = row + static_cast<size_t>(kb) * 34;
        a = reinterpret_cast<uintptr_t>(b + 2);
        const uint32_t* p = reinterpret_cast<const uint32_t*>(a & ~static_cast<uintptr_t>(3));
        #pragma unroll
        for (int i = 0; i < 8; ++i) w[i] = p[i];
        w[8] = (a & 3u) ? p[8] : 0u;
        d = *reinterpret_cast<const uint16_t*>(b);
    }
    __device__ __forceinline__ void decode(int, int (&q)[8], float& dw, int& s0, int& s1) const {
        const uint32_t sh = 8u * static_cast<uint32_t>(a & 3u);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            q[t] = static_cast<int>(sh != 0u ? __funnelshift_r(w[t], w[t + 1], sh) : w[t]);
        }
        s0 = 1;
        s1 = 1;
        dw = fp16_bits_to_f32(static_cast<uint16_t>(d));
    }
};

template <int W_TYPE, int NC, int R, bool PF>
__global__ void __launch_bounds__(256) mm3_gv_kernel(const Mm3GvJobs J) {
    const Mm3GvJob& b = J.j[blockIdx.y];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int row0 = (static_cast<int>(blockIdx.x) * 8 + warp) * R;
    if (row0 >= b.n) return;

    const int nsb = b.k >> 5;
    const uint8_t* rows[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        rows[r] = b.w + static_cast<size_t>(min(row0 + r, b.n - 1)) * b.ld;
    }

    float acc[R][NC];
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int c = 0; c < NC; ++c) acc[r][c] = 0.0f;
    }

    Mm3GvRaw<W_TYPE> nxt[PF ? R : 1];
    if constexpr (PF) {
        if (lane < nsb) {
            #pragma unroll
            for (int r = 0; r < R; ++r) nxt[r].load(rows[r], lane);
        }
    }

    for (int kb = lane; kb < nsb; kb += 32) {
        int q[R][8];
        float dw[R];
        int s0[R], s1[R];
        if constexpr (PF) {
            Mm3GvRaw<W_TYPE> raw[R];
            #pragma unroll
            for (int r = 0; r < R; ++r) raw[r] = nxt[r];
            if (kb + 32 < nsb) {
                #pragma unroll
                for (int r = 0; r < R; ++r) nxt[r].load(rows[r], kb + 32);
            }
            #pragma unroll
            for (int r = 0; r < R; ++r) raw[r].decode(kb, q[r], dw[r], s0[r], s1[r]);
        } else {
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                Mm3GvRaw<W_TYPE> raw;
                raw.load(rows[r], kb);
                raw.decode(kb, q[r], dw[r], s0[r], s1[r]);
            }
        }

        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            const block_q8_1* xb = b.x_q8 + static_cast<size_t>(c) * nsb + kb;
            const float dx = fp16_bits_to_f32(load_u16_align2(&xb->d));
            int xa[8];
            #pragma unroll
            for (int t = 0; t < 8; ++t) xa[t] = load_i32_align4(xb->qs + t * 4);
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                int lo = 0, hi = 0;
                #pragma unroll
                for (int t = 0; t < 4; ++t) lo = dp4a_packed(q[r][t], xa[t], lo);
                #pragma unroll
                for (int t = 4; t < 8; ++t) hi = dp4a_packed(q[r][t], xa[t], hi);
                const float v = dw[r] * static_cast<float>(s0[r] * lo + s1[r] * hi);
                acc[r][c] = fmaf(dx, v, acc[r][c]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = row0 + r;
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            const float sum = warp_reduce_sum(acc[r][c]);
            if (lane == 0 && row < b.n) {
                b.y[static_cast<size_t>(c) * b.n + row] = sum;
            }
        }
    }
}

__global__ void __launch_bounds__(256) mm3_iota_kernel(int32_t* __restrict__ out, int n, int pos0) {
    const int i = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (i < n) out[i] = pos0 + i;
}

__global__ void __launch_bounds__(256) mm3_dspark_draft_block_kernel(
    const float* __restrict__ anc_mapped_f32,
    const float* __restrict__ mask_f32,
    int D,
    int B,
    int pos0,
    int window,
    int32_t anchor_id,
    float* __restrict__ h_out,
    int32_t* __restrict__ pos_out,
    int32_t* __restrict__ lo_out,
    int32_t* __restrict__ prev_out
) {
    const size_t total = static_cast<size_t>(B) * static_cast<size_t>(D);
    for (size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x; i < total; i += static_cast<size_t>(gridDim.x) * 256) {
        const size_t t = i / static_cast<size_t>(D);
        const size_t d = i - t * static_cast<size_t>(D);
        h_out[i] = (t == 0) ? anc_mapped_f32[d] : mask_f32[d];
    }
    if (blockIdx.x == 0 && static_cast<int>(threadIdx.x) < B) {
        const int t = static_cast<int>(threadIdx.x);
        const int pt = pos0 + t;
        pos_out[t] = pt;
        const int lo = pt + 1 - window;
        lo_out[t] = (lo > 0) ? lo : 0;
        if (t == 0) *prev_out = anchor_id;
    }
}

__global__ void __launch_bounds__(32) mm3_wait_u32_kernel(
    const uint32_t* __restrict__ p,
    uint32_t v,
    unsigned long long limit_ns
) {
    if (threadIdx.x != 0) return;
    unsigned long long t0 = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    while (*const_cast<volatile const uint32_t*>(p) != v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        __nanosleep(200);
#endif
        unsigned long long t = 0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if (t - t0 > limit_ns) break;
    }
    __threadfence_system();
}

constexpr int kMkBlocks = 640;
constexpr int kMkR = 256;

__global__ void __launch_bounds__(256) mm3_markov_bias_kernel(
    const float* __restrict__ logits,
    const uint16_t* __restrict__ w2,
    const uint16_t* __restrict__ w1,
    const int32_t* __restrict__ prev,
    int V,
    float* __restrict__ part_v,
    int32_t* __restrict__ part_i,
    float* __restrict__ bias_opt
) {
    __shared__ float s_bv[8];
    __shared__ int32_t s_bi[8];
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;
    const int sub = lane & 7;
    const int rq = lane >> 3;

    float u[32];
    const uint4* up = reinterpret_cast<const uint4*>(w1 + (size_t)(*prev) * kMkR);
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint4 q = up[c * 8 + sub];
        const uint32_t wd[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            u[c * 8 + e * 2]     = __uint_as_float(wd[e] << 16);
            u[c * 8 + e * 2 + 1] = __uint_as_float(wd[e] & 0xFFFF0000u);
        }
    }

    float best = -INFINITY;
    int32_t bid = 0x7FFFFFFF;
    for (int v0 = ((int)blockIdx.x * 8 + warp) * 4; v0 < V; v0 += (int)gridDim.x * 8 * 4) {
        const int v = v0 + rq;
        float acc = 0.0f;
        if (v < V) {
            const uint4* rp = reinterpret_cast<const uint4*>(w2 + (size_t)v * kMkR);
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                const uint4 q = rp[c * 8 + sub];
                const uint32_t wd[4] = {q.x, q.y, q.z, q.w};
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc = fmaf(__uint_as_float(wd[e] << 16), u[c * 8 + e * 2], acc);
                    acc = fmaf(__uint_as_float(wd[e] & 0xFFFF0000u), u[c * 8 + e * 2 + 1], acc);
                }
            }
        }
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) acc += __shfl_xor_sync(0xFFFFFFFFu, acc, o);
        if (v < V && sub == 0) {
            if (bias_opt != nullptr) bias_opt[v] = acc;
            const float x = logits[v] + acc;
            if (x > best || (x == best && v < bid)) {
                best = x;
                bid = v;
            }
        }
    }

    #pragma unroll
    for (int o = 8; o < 32; o <<= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, o);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bid, o);
        if (ov > best || (ov == best && oi < bid)) {
            best = ov;
            bid = oi;
        }
    }
    if (lane == 0) {
        s_bv[warp] = best;
        s_bi[warp] = bid;
    }
    __syncthreads();
    if (warp == 0) {
        best = (lane < 8) ? s_bv[lane] : -INFINITY;
        bid = (lane < 8) ? s_bi[lane] : 0x7FFFFFFF;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, o);
            const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bid, o);
            if (ov > best || (ov == best && oi < bid)) {
                best = ov;
                bid = oi;
            }
        }
        if (lane == 0) {
            part_v[blockIdx.x] = best;
            part_i[blockIdx.x] = bid;
        }
    }
}

__global__ void __launch_bounds__(1024) mm3_markov_final_kernel(
    const float* __restrict__ part_v,
    const int32_t* __restrict__ part_i,
    int nb,
    const uint16_t* __restrict__ w1,
    int32_t* __restrict__ prev,
    int32_t* __restrict__ out,
    const float* __restrict__ cw,
    const float* __restrict__ cb,
    const float* __restrict__ hidden,
    int D,
    int R,
    float* __restrict__ conf
) {
    __shared__ float s_bv[32];
    __shared__ int32_t s_bi[32];
    __shared__ float s_red[32];
    float best = -INFINITY;
    int32_t bid = 0x7FFFFFFF;
    for (int i = (int)threadIdx.x; i < nb; i += (int)blockDim.x) {
        if (part_v[i] > best || (part_v[i] == best && part_i[i] < bid)) {
            best = part_v[i];
            bid = part_i[i];
        }
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, o);
        const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bid, o);
        if (ov > best || (ov == best && oi < bid)) {
            best = ov;
            bid = oi;
        }
    }
    const int lane = (int)threadIdx.x & 31;
    const int wid = (int)threadIdx.x >> 5;
    if (lane == 0) {
        s_bv[wid] = best;
        s_bi[wid] = bid;
    }
    __syncthreads();
    if (wid == 0) {
        best = (lane < 32) ? s_bv[lane] : -INFINITY;
        bid = (lane < 32) ? s_bi[lane] : 0x7FFFFFFF;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xFFFFFFFFu, best, o);
            const int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bid, o);
            if (ov > best || (ov == best && oi < bid)) {
                best = ov;
                bid = oi;
            }
        }
        if (lane == 0) s_bi[0] = bid;
    }
    if (conf != nullptr && cw != nullptr && hidden != nullptr) {
        const uint16_t* er = w1 + (size_t)(*prev) * R;
        float acc = 0.0f;
        for (int i = (int)threadIdx.x; i < D + R; i += (int)blockDim.x) {
            acc = fmaf(cw[i], (i < D) ? hidden[i] : bf2f(er[i - D]), acc);
        }
        const float s = block_reduce_sum(acc, s_red);
        if (threadIdx.x == 0) {
            *conf = 1.0f / (1.0f + expf(-(s + (cb ? *cb : 0.0f))));
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *prev = s_bi[0];
        *out = s_bi[0];
    }
}

// -----------------------------------------------------------------------------
// 5. Zero-cuBLAS Vision Tower Kernels
// -----------------------------------------------------------------------------

__global__ void __launch_bounds__(256) mm3_vision_patch_embed_f32_kernel(
    const float* __restrict__ px,  // [N, P]
    const float* __restrict__ w,   // [D, P]
    int N, int D, int P,
    float* __restrict__ out        // [N, D]
) {
    // Tiled shared-memory F32 GEMM: out[n, d] = sum_p px[n, p] * w[d, p]
    constexpr int BM = 32, BN = 64, BK = 16;
    __shared__ float sA[BM][BK + 1];
    __shared__ float sB[BN][BK + 1];

    const int n0 = (int)blockIdx.y * BM;
    const int d0 = (int)blockIdx.x * BN;
    const int tid = (int)threadIdx.x;
    const int tx = tid & 15;   // 0..15 -> 4 cols of BN=64
    const int ty = tid >> 4;   // 0..15 -> 2 rows of BM=32

    float acc[2][4] = {{0.0f}};
    for (int k0 = 0; k0 < P; k0 += BK) {
        for (int i = tid; i < BM * BK; i += 256) {
            const int r = i >> 4;
            const int c = i & 15;
            const int gn = n0 + r;
            const int gk = k0 + c;
            sA[r][c] = (gn < N && gk < P) ? px[(size_t)gn * P + gk] : 0.0f;
        }
        for (int i = tid; i < BN * BK; i += 256) {
            const int r = i >> 4;
            const int c = i & 15;
            const int gd = d0 + r;
            const int gk = k0 + c;
            sB[r][c] = (gd < D && gk < P) ? w[(size_t)gd * P + gk] : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            const float a0 = sA[ty * 2 + 0][k];
            const float a1 = sA[ty * 2 + 1][k];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float bv = sB[tx * 4 + j][k];
                acc[0][j] = fmaf(a0, bv, acc[0][j]);
                acc[1][j] = fmaf(a1, bv, acc[1][j]);
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i) {
        const int gn = n0 + ty * 2 + i;
        if (gn < N) {
            for (int j = 0; j < 4; ++j) {
                const int gd = d0 + tx * 4 + j;
                if (gd < D) out[(size_t)gn * D + gd] = acc[i][j];
            }
        }
    }
}

__global__ void __launch_bounds__(256) mm3_vision_layernorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    int dim,
    float eps,
    float* __restrict__ y_f32_opt,
    uint16_t* __restrict__ y_bf16_opt
) {
    __shared__ float s_red[32];
    const float* xr = x + (size_t)blockIdx.x * dim;
    float s = 0.0f;
    for (int i = (int)threadIdx.x; i < dim; i += 256) s += xr[i];
    const float mean = block_reduce_sum(s, s_red) / (float)dim;
    float q = 0.0f;
    for (int i = (int)threadIdx.x; i < dim; i += 256) {
        const float d = xr[i] - mean;
        q += d * d;
    }
    const float inv = rsqrtf(block_reduce_sum(q, s_red) / (float)dim + eps);
    __syncthreads();
    for (int i = (int)threadIdx.x; i < dim; i += 256) {
        const float v = (xr[i] - mean) * inv * w[i] + b[i];
        if (y_f32_opt != nullptr) y_f32_opt[(size_t)blockIdx.x * dim + i] = v;
        if (y_bf16_opt != nullptr) y_bf16_opt[(size_t)blockIdx.x * dim + i] = f2bf(v);
    }
}

__global__ void __launch_bounds__(256) mm3_vision_qkv_rope_3d_kernel(
    float* __restrict__ qkv,
    const float* __restrict__ bias,
    const int32_t* __restrict__ pos3d,
    const float* __restrict__ inv_freq,
    int dim, int heads, int hd, int nf
) {
    const int t = (int)blockIdx.x;
    float* r = qkv + (size_t)t * 3 * dim;
    if (bias != nullptr) {
        for (int i = (int)threadIdx.x; i < 3 * dim; i += 256) r[i] += bias[i];
        __syncthreads();
    }
    const int half = 3 * nf;
    for (int j = (int)threadIdx.x; j < 2 * heads * half; j += 256) {
        const int which = j / (heads * half);
        const int rem = j - which * heads * half;
        const int hh = rem / half;
        const int i = rem - hh * half;
        float* v = r + (size_t)which * dim + (size_t)hh * hd;
        const float ang = (float)pos3d[t * 3 + i / nf] * inv_freq[i % nf];
        float sn, cs;
        sincosf(ang, &sn, &cs);
        const float x1 = v[i], x2 = v[i + half];
        v[i] = x1 * cs - x2 * sn;
        v[i + half] = x2 * cs + x1 * sn;
    }
}

// Flash-style online-softmax full attention for Vision tower (arbitrary N, hd <= 128).
// 1 block of 128 threads = 4 warps = 4 queries of head `h`, iterating BK=16 key/value tiles in SMEM.
__global__ void __launch_bounds__(128) mm3_vision_full_attn_kernel(
    const float* __restrict__ qkv,  // [N, 3 * dim]
    int N, int dim, int heads, int hd, float scale,
    float* __restrict__ out         // [N, dim]
) {
    constexpr int BK = 16;
    __shared__ float sK[BK][128];
    __shared__ float sV[BK][128];
    __shared__ float s_sc[4][BK];

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int t = (int)blockIdx.x * 4 + warp;
    const int h = (int)blockIdx.y;

    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int n_chunks = (hd + 31) >> 5;
    if (t < N) {
        const float* qr = qkv + (size_t)t * 3 * dim + (size_t)h * hd;
        for (int c = 0; c < n_chunks; ++c) {
            const int d = c * 32 + lane;
            if (d < hd) qv[c] = qr[d] * scale;
        }
    }

    float m_row = -INFINITY;
    float l_row = 0.0f;

    for (int k0 = 0; k0 < N; k0 += BK) {
        const int count = min(BK, N - k0);
        __syncthreads();
        for (int j = warp; j < count; j += 4) {
            const float* kr = qkv + (size_t)(k0 + j) * 3 * dim + dim + (size_t)h * hd;
            const float* vr = qkv + (size_t)(k0 + j) * 3 * dim + 2 * dim + (size_t)h * hd;
            for (int c = 0; c < n_chunks; ++c) {
                const int d = c * 32 + lane;
                if (d < hd) {
                    sK[j][d] = kr[d];
                    sV[j][d] = vr[d];
                }
            }
        }
        __syncthreads();

        float m_tile = -INFINITY;
        for (int j = 0; j < count; ++j) {
            float dot = 0.0f;
            for (int c = 0; c < n_chunks; ++c) {
                const int d = c * 32 + lane;
                if (d < hd) dot = fmaf(qv[c], sK[j][d], dot);
            }
            dot = warp_reduce_sum(dot);
            if (lane == 0) s_sc[warp][j] = dot;
            if (lane == 0) m_tile = fmaxf(m_tile, dot);
        }
        m_tile = __shfl_sync(0xFFFFFFFFu, m_tile, 0);
        const float m_new = fmaxf(m_row, m_tile);
        const float alpha = (m_row == -INFINITY) ? 0.0f : expf(m_row - m_new);
        float l_tile = 0.0f;
        for (int c = 0; c < n_chunks; ++c) acc[c] *= alpha;
        for (int j = 0; j < count; ++j) {
            const float p = expf(s_sc[warp][j] - m_new);
            l_tile += p;
            for (int c = 0; c < n_chunks; ++c) {
                const int d = c * 32 + lane;
                if (d < hd) acc[c] = fmaf(p, sV[j][d], acc[c]);
            }
        }
        l_row = l_row * alpha + l_tile;
        m_row = m_new;
    }

    if (t < N) {
        float* orow = out + (size_t)t * dim + (size_t)h * hd;
        const float inv = (l_row > 0.0f) ? (1.0f / l_row) : 0.0f;
        for (int c = 0; c < n_chunks; ++c) {
            const int d = c * 32 + lane;
            if (d < hd) orow[d] = acc[c] * inv;
        }
    }
}

__global__ void __launch_bounds__(256) mm3_vision_bias_gelu_kernel(
    const float* __restrict__ y,
    const float* __restrict__ bias,
    size_t n, int M, int apply_gelu,
    float* __restrict__ yf,
    uint16_t* __restrict__ y16
) {
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) {
        float v = y[i] + (bias ? bias[i % (size_t)M] : 0.0f);
        if (apply_gelu) v = 0.5f * v * (1.0f + erff(v * 0.70710678118654752f));
        if (yf != nullptr) yf[i] = v;
        if (y16 != nullptr) y16[i] = f2bf(v);
    }
}

__global__ void __launch_bounds__(256) mm3_vision_residual_bias_kernel(
    float* __restrict__ h,
    const float* __restrict__ t,
    const float* __restrict__ bias,
    size_t n, int M
) {
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n; i += (size_t)gridDim.x * 256) {
        h[i] += t[i] + (bias ? bias[i % (size_t)M] : 0.0f);
    }
}

template <int NT_CAP>
static void dispatch_mm3_moe_grp_gate_up(
    int32_t gate_type,
    int32_t up_type,
    const Mm3DecodePlan* plan,
    size_t u_off,
    size_t gate_rb,
    size_t up_rb,
    const block_q8_K* xq,
    float* inter_f32,
    int n_embd,
    int n_ff,
    int n_grp,
    float alpha,
    float limit,
    cudaStream_t stream
) {
    const dim3 grid((unsigned)((n_ff + 7) >> 3), (unsigned)n_grp);
    const int block = 256;
    if (gate_type == GGML_TYPE_IQ2_XXS && up_type == GGML_TYPE_IQ2_XXS) {
        mm3_moe_grp_gate_up_f32_kernel<GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS, NT_CAP><<<grid, block, 0, stream>>>(gate_type, up_type, plan, u_off, gate_rb, up_rb, xq, inter_f32, n_embd, n_ff, alpha, limit);
    } else if (gate_type == GGML_TYPE_IQ2_XS && up_type == GGML_TYPE_IQ2_XS) {
        mm3_moe_grp_gate_up_f32_kernel<GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, NT_CAP><<<grid, block, 0, stream>>>(gate_type, up_type, plan, u_off, gate_rb, up_rb, xq, inter_f32, n_embd, n_ff, alpha, limit);
    } else if (gate_type == GGML_TYPE_IQ2_XS && up_type == GGML_TYPE_IQ2_XXS) {
        mm3_moe_grp_gate_up_f32_kernel<GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, NT_CAP><<<grid, block, 0, stream>>>(gate_type, up_type, plan, u_off, gate_rb, up_rb, xq, inter_f32, n_embd, n_ff, alpha, limit);
    } else if (gate_type == GGML_TYPE_IQ2_S && up_type == GGML_TYPE_IQ2_S) {
        mm3_moe_grp_gate_up_f32_kernel<GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S, NT_CAP><<<grid, block, 0, stream>>>(gate_type, up_type, plan, u_off, gate_rb, up_rb, xq, inter_f32, n_embd, n_ff, alpha, limit);
    } else if (gate_type == GGML_TYPE_IQ3_XXS && up_type == GGML_TYPE_IQ3_XXS) {
        mm3_moe_grp_gate_up_f32_kernel<GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS, NT_CAP><<<grid, block, 0, stream>>>(gate_type, up_type, plan, u_off, gate_rb, up_rb, xq, inter_f32, n_embd, n_ff, alpha, limit);
    } else if (gate_type == GGML_TYPE_IQ1_S && up_type == GGML_TYPE_IQ1_S) {
        mm3_moe_grp_gate_up_f32_kernel<GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S, NT_CAP><<<grid, block, 0, stream>>>(gate_type, up_type, plan, u_off, gate_rb, up_rb, xq, inter_f32, n_embd, n_ff, alpha, limit);
    } else {
        mm3_moe_grp_gate_up_f32_kernel<0, 0, NT_CAP><<<grid, block, 0, stream>>>(gate_type, up_type, plan, u_off, gate_rb, up_rb, xq, inter_f32, n_embd, n_ff, alpha, limit);
    }
}

template <int NT_CAP>
static void dispatch_mm3_moe_grp_down(
    int32_t down_type,
    const Mm3DecodePlan* plan,
    size_t d_off,
    size_t down_rb,
    const block_q8_K* fq,
    float* expert_out,
    int n_ff,
    int n_embd,
    int n_grp,
    cudaStream_t stream
) {
    const dim3 grid((unsigned)((n_embd + 7) >> 3), (unsigned)n_grp);
    const int block = 256;
    switch (down_type) {
        case GGML_TYPE_IQ2_XXS: mm3_moe_grp_down_q8k_kernel<GGML_TYPE_IQ2_XXS, NT_CAP><<<grid, block, 0, stream>>>(down_type, plan, d_off, down_rb, fq, expert_out, n_ff, n_embd); break;
        case GGML_TYPE_IQ2_XS:  mm3_moe_grp_down_q8k_kernel<GGML_TYPE_IQ2_XS, NT_CAP><<<grid, block, 0, stream>>>(down_type, plan, d_off, down_rb, fq, expert_out, n_ff, n_embd); break;
        case GGML_TYPE_IQ2_S:   mm3_moe_grp_down_q8k_kernel<GGML_TYPE_IQ2_S, NT_CAP><<<grid, block, 0, stream>>>(down_type, plan, d_off, down_rb, fq, expert_out, n_ff, n_embd); break;
        case GGML_TYPE_IQ3_XXS: mm3_moe_grp_down_q8k_kernel<GGML_TYPE_IQ3_XXS, NT_CAP><<<grid, block, 0, stream>>>(down_type, plan, d_off, down_rb, fq, expert_out, n_ff, n_embd); break;
        case GGML_TYPE_IQ1_S:   mm3_moe_grp_down_q8k_kernel<GGML_TYPE_IQ1_S, NT_CAP><<<grid, block, 0, stream>>>(down_type, plan, d_off, down_rb, fq, expert_out, n_ff, n_embd); break;
        default:                mm3_moe_grp_down_q8k_kernel<0, NT_CAP><<<grid, block, 0, stream>>>(down_type, plan, d_off, down_rb, fq, expert_out, n_ff, n_embd); break;
    }
}

template <int R>
__global__ void __launch_bounds__(256) mm3_bf16_gemv_small_m_kernel(
    const uint16_t* __restrict__ A,
    int ldA,
    const uint16_t* __restrict__ B,
    int ldB,
    float* __restrict__ C,
    int ldC,
    int N,
    int K,
    float alpha,
    float beta
) {
    const int warp = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int n = (int)blockIdx.x * 8 + warp;
    if (n >= N) return;

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;

    const uint16_t* b_row = B + (size_t)n * ldB;
    for (int k = lane * 8; k < K; k += 256) {
        const uint4 bw = *reinterpret_cast<const uint4*>(b_row + k);
        const uint32_t bd[4] = {bw.x, bw.y, bw.z, bw.w};
        float bf[8];
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
            bf[2 * e]     = __uint_as_float(bd[e] << 16);
            bf[2 * e + 1] = __uint_as_float(bd[e] & 0xFFFF0000u);
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint4 aw = *reinterpret_cast<const uint4*>(A + (size_t)r * ldA + k);
            const uint32_t ad[4] = {aw.x, aw.y, aw.z, aw.w};
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                acc[r] = fmaf(__uint_as_float(ad[e] << 16), bf[2 * e], acc[r]);
                acc[r] = fmaf(__uint_as_float(ad[e] & 0xFFFF0000u), bf[2 * e + 1], acc[r]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        const float sum = warp_reduce_sum(acc[r]);
        if (lane == 0) {
            float* dst = C + (size_t)r * ldC + n;
            *dst = (beta == 0.0f) ? (alpha * sum) : (alpha * sum + beta * (*dst));
        }
    }
}

}  // namespace

extern "C" {

void launch_linear_q8_1(
    int32_t w_type,
    const void* W,
    const void* X_q8,
    const float* X_f32,
    float* Y,
    int N,
    int K,
    int M,
    cudaStream_t stream
);

void launch_mm3_vision_bias_gelu(
    const float* y,
    const float* bias_opt,
    size_t n,
    int M,
    int apply_gelu,
    float* y_f32_opt,
    uint16_t* y_bf16_opt,
    cudaStream_t stream
);

void launch_mm3_rms_norm_f32(
    const float* x,
    const float* w,
    int rows,
    int dim,
    float eps,
    int add_one,
    float* y_f32_opt,
    void* y_q8_1_opt,
    void* y_q8_k_opt,
    uint16_t* y_bf16_opt,
    cudaStream_t stream
) {
    if (rows <= 0 || dim <= 0) return;
    mm3_rms_norm_kernel<<<rows, 256, 0, stream>>>(
        x, w, dim, eps, add_one, y_f32_opt,
        reinterpret_cast<block_q8_1*>(y_q8_1_opt),
        reinterpret_cast<block_q8_K*>(y_q8_k_opt),
        y_bf16_opt
    );
}

void launch_mm3_quantize_q8_k(
    const float* x,
    void* out_q8k,
    int m,
    int k,
    cudaStream_t stream
) {
    if (m <= 0 || k <= 0) return;
    const int nb = (k + 255) >> 8;
    mm3_quantize_q8_k_kernel<<<dim3((unsigned)nb, (unsigned)m), 256, 0, stream>>>(
        x, reinterpret_cast<block_q8_K*>(out_q8k), k
    );
}

void launch_mm3_swigluoai(
    const float* gate,
    const float* up,
    int rows,
    int ff,
    float alpha,
    float limit,
    float* y_f32_opt,
    void* y_q8_1_opt,
    void* y_q8_k_opt,
    uint16_t* y_bf16_opt,
    cudaStream_t stream
) {
    if (rows <= 0 || ff <= 0) return;
    const int nb = (ff + 255) >> 8;
    mm3_swigluoai_kernel<<<dim3((unsigned)nb, (unsigned)rows), 256, 0, stream>>>(
        gate, up, ff, alpha, limit, y_f32_opt,
        reinterpret_cast<block_q8_1*>(y_q8_1_opt),
        reinterpret_cast<block_q8_K*>(y_q8_k_opt),
        y_bf16_opt
    );
}

void launch_mm3_qk_norm_rope_kv_store(
    float* q,
    const float* k,
    const float* v,
    const float* q_norm_w,
    const float* k_norm_w,
    const int32_t* pos,
    const float* inv_freq,
    int T,
    int Hq,
    int Hkv,
    int hd,
    int rot_dim,
    float eps,
    int kv_fmt,
    int rot_mask,
    int ring,
    uint16_t* k_bf16_opt,
    uint16_t* v_bf16_opt,
    int8_t* k_i8_opt,
    float* k_scale_opt,
    int8_t* v_i8_opt,
    float* v_scale_opt,
    void* host_k_opt,
    float* host_k_scale_opt,
    void* host_v_opt,
    float* host_v_scale_opt,
    int32_t* host_tag_opt,
    cudaStream_t stream
) {
    if (T <= 0 || hd <= 0) return;
    const int total_h = Hq + 2 * Hkv;
    if (total_h <= 0) return;
    mm3_qk_norm_rope_kv_store_kernel<<<dim3((unsigned)T, (unsigned)total_h), (unsigned)hd, 0, stream>>>(
        q, k, v, q_norm_w, k_norm_w, pos, inv_freq, Hq, Hkv, rot_dim, eps,
        kv_fmt, rot_mask, ring, k_bf16_opt, v_bf16_opt, k_i8_opt, k_scale_opt, v_i8_opt, v_scale_opt,
        host_k_opt, host_k_scale_opt, host_v_opt, host_v_scale_opt, host_tag_opt
    );
}

void launch_mm3_index_norm_rope_store(
    float* iq,
    const float* ik,
    const float* iq_norm_w,
    const float* ik_norm_w,
    const int32_t* pos,
    const float* inv_freq,
    int T,
    int Hi,
    int hd,
    int rot_dim,
    float eps,
    uint16_t* ic_bf16,
    cudaStream_t stream
) {
    if (T <= 0 || hd <= 0) return;
    mm3_index_norm_rope_store_kernel<<<dim3((unsigned)T, (unsigned)(Hi + 1)), (unsigned)hd, 0, stream>>>(
        iq, ik, iq_norm_w, ik_norm_w, pos, inv_freq, Hi, rot_dim, eps, ic_bf16
    );
}

void launch_mm3_msa_block_scores(
    const float* iq,
    const uint16_t* ic_bf16,
    const int32_t* pos,
    int T,
    int Hi,
    int nblk,
    int blk,
    float* bs,
    cudaStream_t stream
) {
    if (T <= 0 || Hi <= 0 || nblk <= 0) return;
    if (T >= 16 && Hi <= 4) {
        mm3_msa_block_scores_tiled4_kernel<<<dim3((unsigned)nblk, (unsigned)((T + 3) / 4)), 128, 0, stream>>>(
            iq, ic_bf16, pos, T, Hi, nblk, blk, bs
        );
    } else {
        mm3_msa_block_scores_kernel<<<dim3((unsigned)nblk, (unsigned)T), 128, 0, stream>>>(
            iq, ic_bf16, pos, T, Hi, nblk, blk, bs
        );
    }
}

void launch_mm3_msa_select_topk(
    const float* bs,
    const int32_t* pos,
    int T,
    int Hi,
    int nblk,
    int blk,
    int topk,
    int local_blocks,
    int32_t* kept,
    cudaStream_t stream
) {
    if (T <= 0 || Hi <= 0 || topk <= 0) return;
    const int blocks = (T * Hi + 7) / 8;
    mm3_msa_select_topk_kernel<<<blocks, 256, 0, stream>>>(
        bs, pos, T, Hi, nblk, blk, topk, local_blocks, kept
    );
}

void launch_mm3_msa_sparse_attn(
    const float* q,
    const uint16_t* kc_bf16_opt,
    const uint16_t* vc_bf16_opt,
    const int8_t* kc_i8_opt,
    const float* k_scale_opt,
    const int8_t* vc_i8_opt,
    const float* v_scale_opt,
    const int32_t* kept,
    const int32_t* pos,
    int T,
    int Hq,
    int Hkv,
    int topk,
    int blk,
    int kv_fmt,
    int rot_mask,
    int ring,
    float scale,
    float* part_opt,
    int splits,
    float* out,
    cudaStream_t stream
) {
    if (T <= 0 || Hq <= 0 || Hkv <= 0) return;
    const int S = (part_opt != nullptr && splits > 1) ? min(splits, kCombMaxSplits) : 1;
    dim3 grid((unsigned)T, (unsigned)Hkv, (unsigned)S);
    if (T <= 16 && Hq == 16 * Hkv) {
        if (kv_fmt == 0) {
            mm3_attn_wide_kernel<0, true><<<grid, 512, 0, stream>>>(
                q, kc_bf16_opt, vc_bf16_opt, nullptr, nullptr, nullptr, nullptr,
                kept, nullptr, pos, 0, Hq, Hkv, topk, blk, 0, ring, scale, S, part_opt, out
            );
        } else {
            mm3_attn_wide_kernel<1, true><<<grid, 512, 0, stream>>>(
                q, nullptr, nullptr, kc_i8_opt, k_scale_opt, vc_i8_opt, v_scale_opt,
                kept, nullptr, pos, 0, Hq, Hkv, topk, blk, rot_mask, ring, scale, S, part_opt, out
            );
        }
    } else if (kv_fmt == 0) {
        mm3_attn_kernel<0, true><<<grid, 128, 0, stream>>>(
            q, kc_bf16_opt, vc_bf16_opt, nullptr, nullptr, nullptr, nullptr,
            kept, nullptr, pos, 0, Hq, Hkv, topk, blk, 0, ring, scale, S, part_opt, out
        );
    } else {
        mm3_attn_kernel<1, true><<<grid, 128, 0, stream>>>(
            q, nullptr, nullptr, kc_i8_opt, k_scale_opt, vc_i8_opt, v_scale_opt,
            kept, nullptr, pos, 0, Hq, Hkv, topk, blk, rot_mask, ring, scale, S, part_opt, out
        );
    }
    if (S > 1) {
        const int G = Hq / Hkv;
        const int unrot_v = (kv_fmt != 0) && ((rot_mask & 2) != 0);
        mm3_attn_combine_kernel<<<dim3((unsigned)T, (unsigned)Hkv, (unsigned)G), 128, 0, stream>>>(
            part_opt, Hq, Hkv, S, unrot_v, out
        );
    }
}

void launch_mm3_router_f32(
    const float* x_f32_opt,
    const float* gate_w_opt,
    float* logits_buf,
    const float* bias_opt,
    int T,
    int D,
    int E,
    int topk,
    float routed_scaling_factor,
    int32_t* ids_out,
    float* weights_out,
    cudaStream_t stream
) {
    if (T <= 0 || E <= 0 || topk <= 0) return;
    if (x_f32_opt != nullptr && gate_w_opt != nullptr) {
        launch_linear_q8_1(GGML_TYPE_F32, gate_w_opt, nullptr, x_f32_opt, logits_buf, E, D, T, stream);
    }
    const int blocks = (T + 7) / 8;
    mm3_router_topk_kernel<<<blocks, 256, 0, stream>>>(
        logits_buf, bias_opt, T, E, topk, routed_scaling_factor, ids_out, weights_out
    );
}

void launch_mm3_moe_gate_up_q8k(
    int32_t gate_type,
    int32_t up_type,
    const uint64_t* gate_ptrs,
    const uint64_t* up_ptrs,
    size_t gate_row_bytes,
    size_t up_row_bytes,
    const void* x_q8k,
    const int32_t* expert_ids,
    void* inter_q8k,
    float* inter_f32_opt,
    int M,
    int n_embd,
    int n_ff,
    int topk,
    float swiglu_alpha,
    float swiglu_limit,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (M <= 0 || topk <= 0 || n_ff <= 0 || (n_embd & 255) != 0 || (n_ff & 255) != 0) return;
    const dim3 grid((unsigned)(n_ff >> 8), (unsigned)topk, (unsigned)M);
    const int block = 256;
    const auto* xq = reinterpret_cast<const block_q8_K*>(x_q8k);
    auto* iq = reinterpret_cast<block_q8_K*>(inter_q8k);
    if (gate_type == GGML_TYPE_IQ2_XXS && up_type == GGML_TYPE_IQ2_XXS) {
        mm3_moe_gate_up_q8k_kernel<GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XXS><<<grid, block, 0, stream>>>(gate_type, up_type, gate_ptrs, up_ptrs, gate_row_bytes, up_row_bytes, xq, expert_ids, iq, inter_f32_opt, n_embd, n_ff, topk, swiglu_alpha, swiglu_limit, skip_dev);
    } else if (gate_type == GGML_TYPE_IQ2_XS && up_type == GGML_TYPE_IQ2_XS) {
        mm3_moe_gate_up_q8k_kernel<GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS><<<grid, block, 0, stream>>>(gate_type, up_type, gate_ptrs, up_ptrs, gate_row_bytes, up_row_bytes, xq, expert_ids, iq, inter_f32_opt, n_embd, n_ff, topk, swiglu_alpha, swiglu_limit, skip_dev);
    } else if (gate_type == GGML_TYPE_IQ2_XS && up_type == GGML_TYPE_IQ2_XXS) {
        mm3_moe_gate_up_q8k_kernel<GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS><<<grid, block, 0, stream>>>(gate_type, up_type, gate_ptrs, up_ptrs, gate_row_bytes, up_row_bytes, xq, expert_ids, iq, inter_f32_opt, n_embd, n_ff, topk, swiglu_alpha, swiglu_limit, skip_dev);
    } else if (gate_type == GGML_TYPE_IQ2_S && up_type == GGML_TYPE_IQ2_S) {
        mm3_moe_gate_up_q8k_kernel<GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_S><<<grid, block, 0, stream>>>(gate_type, up_type, gate_ptrs, up_ptrs, gate_row_bytes, up_row_bytes, xq, expert_ids, iq, inter_f32_opt, n_embd, n_ff, topk, swiglu_alpha, swiglu_limit, skip_dev);
    } else if (gate_type == GGML_TYPE_IQ3_XXS && up_type == GGML_TYPE_IQ3_XXS) {
        mm3_moe_gate_up_q8k_kernel<GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS><<<grid, block, 0, stream>>>(gate_type, up_type, gate_ptrs, up_ptrs, gate_row_bytes, up_row_bytes, xq, expert_ids, iq, inter_f32_opt, n_embd, n_ff, topk, swiglu_alpha, swiglu_limit, skip_dev);
    } else if (gate_type == GGML_TYPE_IQ1_S && up_type == GGML_TYPE_IQ1_S) {
        mm3_moe_gate_up_q8k_kernel<GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_S><<<grid, block, 0, stream>>>(gate_type, up_type, gate_ptrs, up_ptrs, gate_row_bytes, up_row_bytes, xq, expert_ids, iq, inter_f32_opt, n_embd, n_ff, topk, swiglu_alpha, swiglu_limit, skip_dev);
    } else {
        mm3_moe_gate_up_q8k_kernel<0, 0><<<grid, block, 0, stream>>>(gate_type, up_type, gate_ptrs, up_ptrs, gate_row_bytes, up_row_bytes, xq, expert_ids, iq, inter_f32_opt, n_embd, n_ff, topk, swiglu_alpha, swiglu_limit, skip_dev);
    }
}

void launch_mm3_moe_down_q8k(
    int32_t down_type,
    const uint64_t* down_ptrs,
    size_t down_row_bytes,
    const void* inter_q8k,
    const int32_t* expert_ids,
    float* expert_out,
    int M,
    int n_ff,
    int n_embd,
    int topk,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    if (M <= 0 || topk <= 0 || n_embd <= 0 || (n_ff & 255) != 0) return;
    const dim3 grid((unsigned)((n_embd + 31) >> 5), (unsigned)topk, (unsigned)M);
    const int block = 256;
    const auto* fq = reinterpret_cast<const block_q8_K*>(inter_q8k);
    switch (down_type) {
        case GGML_TYPE_IQ2_XXS: mm3_moe_down_q8k_kernel<GGML_TYPE_IQ2_XXS><<<grid, block, 0, stream>>>(down_type, down_ptrs, down_row_bytes, fq, expert_ids, expert_out, n_ff, n_embd, topk, skip_dev); break;
        case GGML_TYPE_IQ2_XS:  mm3_moe_down_q8k_kernel<GGML_TYPE_IQ2_XS><<<grid, block, 0, stream>>>(down_type, down_ptrs, down_row_bytes, fq, expert_ids, expert_out, n_ff, n_embd, topk, skip_dev); break;
        case GGML_TYPE_IQ2_S:   mm3_moe_down_q8k_kernel<GGML_TYPE_IQ2_S><<<grid, block, 0, stream>>>(down_type, down_ptrs, down_row_bytes, fq, expert_ids, expert_out, n_ff, n_embd, topk, skip_dev); break;
        case GGML_TYPE_IQ3_XXS: mm3_moe_down_q8k_kernel<GGML_TYPE_IQ3_XXS><<<grid, block, 0, stream>>>(down_type, down_ptrs, down_row_bytes, fq, expert_ids, expert_out, n_ff, n_embd, topk, skip_dev); break;
        case GGML_TYPE_IQ1_S:   mm3_moe_down_q8k_kernel<GGML_TYPE_IQ1_S><<<grid, block, 0, stream>>>(down_type, down_ptrs, down_row_bytes, fq, expert_ids, expert_out, n_ff, n_embd, topk, skip_dev); break;
        default:                mm3_moe_down_q8k_kernel<0><<<grid, block, 0, stream>>>(down_type, down_ptrs, down_row_bytes, fq, expert_ids, expert_out, n_ff, n_embd, topk, skip_dev); break;
    }
}

void launch_mm3_moe_combine(
    const float* res_opt,
    const float* expert_out,
    const float* weights,
    const int32_t* expert_ids,
    const uint64_t* vram_ptrs_opt,
    const float* shared_opt,
    int M,
    int topk,
    int n_embd,
    float* out,
    cudaStream_t stream
) {
    if (M <= 0 || n_embd <= 0) return;
    mm3_moe_combine_kernel<<<M, 256, 0, stream>>>(
        res_opt, expert_out, weights, expert_ids, vram_ptrs_opt, shared_opt, topk, n_embd, out
    );
}

void launch_mm3_moe_grp_gate_up_f32(
    int32_t gate_type,
    int32_t up_type,
    const void* plan_dev,
    size_t u_off,
    size_t gate_row_bytes,
    size_t up_row_bytes,
    const void* x_q8k,
    float* inter_f32,
    int n_embd,
    int n_ff,
    int n_grp,
    int max_nt,
    float swiglu_alpha,
    float swiglu_limit,
    cudaStream_t stream
) {
    if (n_grp <= 0 || n_ff <= 0 || (n_embd & 255) != 0) return;
    const auto* plan = reinterpret_cast<const Mm3DecodePlan*>(plan_dev);
    const auto* xq = reinterpret_cast<const block_q8_K*>(x_q8k);
    if (max_nt <= 1) {
        dispatch_mm3_moe_grp_gate_up<1>(gate_type, up_type, plan, u_off, gate_row_bytes, up_row_bytes, xq, inter_f32, n_embd, n_ff, n_grp, swiglu_alpha, swiglu_limit, stream);
    } else {
        dispatch_mm3_moe_grp_gate_up<4>(gate_type, up_type, plan, u_off, gate_row_bytes, up_row_bytes, xq, inter_f32, n_embd, n_ff, n_grp, swiglu_alpha, swiglu_limit, stream);
    }
}

void launch_mm3_moe_grp_down_q8k(
    int32_t down_type,
    const void* plan_dev,
    size_t d_off,
    size_t down_row_bytes,
    const void* inter_q8k,
    float* expert_out,
    int n_ff,
    int n_embd,
    int n_grp,
    int max_nt,
    cudaStream_t stream
) {
    if (n_grp <= 0 || n_embd <= 0 || (n_ff & 255) != 0) return;
    const auto* plan = reinterpret_cast<const Mm3DecodePlan*>(plan_dev);
    const auto* fq = reinterpret_cast<const block_q8_K*>(inter_q8k);
    if (max_nt <= 1) {
        dispatch_mm3_moe_grp_down<1>(down_type, plan, d_off, down_row_bytes, fq, expert_out, n_ff, n_embd, n_grp, stream);
    } else {
        dispatch_mm3_moe_grp_down<4>(down_type, plan, d_off, down_row_bytes, fq, expert_out, n_ff, n_embd, n_grp, stream);
    }
}

void launch_mm3_moe_grp_combine(
    const float* res_opt,
    const float* shared_opt,
    const float* expert_out,
    const void* plan0_dev_opt,
    int n_entries0,
    const void* plan1_dev_opt,
    int n_entries1,
    const float* cpu_out_opt,
    int M,
    int n_embd,
    float* out,
    cudaStream_t stream
) {
    if (M <= 0 || n_embd <= 0) return;
    mm3_moe_grp_combine_kernel<<<M, 256, 0, stream>>>(
        res_opt,
        shared_opt,
        expert_out,
        reinterpret_cast<const Mm3DecodePlan*>(plan0_dev_opt),
        n_entries0,
        reinterpret_cast<const Mm3DecodePlan*>(plan1_dev_opt),
        n_entries1,
        cpu_out_opt,
        n_embd,
        out
    );
}

void launch_mm3_bf16_gemm_mma(
    const uint16_t* A,
    int ldA,
    const uint16_t* B,
    int ldB,
    float* C,
    int ldC,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    if (M >= 1 && M <= 8 && (K & 7) == 0 && (ldA & 7) == 0 && (ldB & 7) == 0
        && ((reinterpret_cast<uintptr_t>(A) & 15u) == 0u)
        && ((reinterpret_cast<uintptr_t>(B) & 15u) == 0u)) {
        const unsigned grid = static_cast<unsigned>((N + 7) / 8);
        switch (M) {
            case 1: mm3_bf16_gemv_small_m_kernel<1><<<grid, 256, 0, stream>>>(A, ldA, B, ldB, C, ldC, N, K, alpha, beta); break;
            case 2: mm3_bf16_gemv_small_m_kernel<2><<<grid, 256, 0, stream>>>(A, ldA, B, ldB, C, ldC, N, K, alpha, beta); break;
            case 3: mm3_bf16_gemv_small_m_kernel<3><<<grid, 256, 0, stream>>>(A, ldA, B, ldB, C, ldC, N, K, alpha, beta); break;
            case 4: mm3_bf16_gemv_small_m_kernel<4><<<grid, 256, 0, stream>>>(A, ldA, B, ldB, C, ldC, N, K, alpha, beta); break;
            case 5: mm3_bf16_gemv_small_m_kernel<5><<<grid, 256, 0, stream>>>(A, ldA, B, ldB, C, ldC, N, K, alpha, beta); break;
            case 6: mm3_bf16_gemv_small_m_kernel<6><<<grid, 256, 0, stream>>>(A, ldA, B, ldB, C, ldC, N, K, alpha, beta); break;
            case 7: mm3_bf16_gemv_small_m_kernel<7><<<grid, 256, 0, stream>>>(A, ldA, B, ldB, C, ldC, N, K, alpha, beta); break;
            case 8: mm3_bf16_gemv_small_m_kernel<8><<<grid, 256, 0, stream>>>(A, ldA, B, ldB, C, ldC, N, K, alpha, beta); break;
        }
        return;
    }
    if (M <= 32) {
        dim3 grid((unsigned)((N + 63) / 64), (unsigned)((M + 31) / 32));
        mm3_gemm_bf16_mma_kernel<32, 64, 1, 4, 2><<<grid, 128, 0, stream>>>(A, ldA, B, ldB, C, ldC, M, N, K, alpha, beta);
    } else if (M >= 128 && N >= 128) {
        dim3 grid((unsigned)((N + 127) / 128), (unsigned)((M + 127) / 128));
        mm3_gemm_bf16_mma_kernel<128, 128, 2, 4, 1><<<grid, 256, 0, stream>>>(A, ldA, B, ldB, C, ldC, M, N, K, alpha, beta);
    } else if (N >= 128) {
        dim3 grid((unsigned)((N + 127) / 128), (unsigned)((M + 63) / 64));
        mm3_gemm_bf16_mma_kernel<64, 128, 2, 2, 1><<<grid, 128, 0, stream>>>(A, ldA, B, ldB, C, ldC, M, N, K, alpha, beta);
    } else {
        dim3 grid((unsigned)((N + 63) / 64), (unsigned)((M + 63) / 64));
        mm3_gemm_bf16_mma_kernel<64, 64, 2, 2, 2><<<grid, 128, 0, stream>>>(A, ldA, B, ldB, C, ldC, M, N, K, alpha, beta);
    }
}

void launch_mm3_mxfp8_to_bf16(
    const uint8_t* w,
    const uint8_t* scale,
    int N,
    int K,
    uint16_t* out_bf16,
    cudaStream_t stream
) {
    if (N <= 0 || K <= 0) return;
    const int bx = min(64, (K + 255) >> 8);
    mm3_mxfp8_to_bf16_kernel<<<dim3((unsigned)bx, (unsigned)N), 256, 0, stream>>>(w, scale, K, out_bf16);
}

void launch_mm3_mxfp8_linear(
    const uint8_t* w,
    const uint8_t* scale,
    int N,
    int K,
    const float* x,
    int ldX,
    int rows,
    float* y,
    int ldY,
    float beta,
    float* part_opt,
    uint16_t* w_bf16_scratch_opt,
    uint16_t* x_bf16_scratch_opt,
    cudaStream_t stream
) {
    if (rows <= 0 || N <= 0 || K <= 0) return;
    if (rows > 16 && w_bf16_scratch_opt != nullptr && x_bf16_scratch_opt != nullptr) {
        launch_mm3_mxfp8_to_bf16(w, scale, N, K, w_bf16_scratch_opt, stream);
        for (int r = 0; r < rows; ++r) {
            // Convert each row of X[r, 0..K] to contiguous x_bf16_scratch[r, 0..K]
            launch_mm3_vision_bias_gelu(x + (size_t)r * ldX, nullptr, (size_t)K, K, 0, nullptr, x_bf16_scratch_opt + (size_t)r * K, stream);
        }
        launch_mm3_bf16_gemm_mma(x_bf16_scratch_opt, K, w_bf16_scratch_opt, K, y, ldY, rows, N, K, 1.0f, beta, stream);
        return;
    }
    if (rows > 16) {
        for (int r0 = 0; r0 < rows; r0 += 16) {
            launch_mm3_mxfp8_linear(
                w, scale, N, K, x + (size_t)r0 * ldX, ldX, min(16, rows - r0),
                y + (size_t)r0 * ldY, ldY, beta, part_opt, nullptr, nullptr, stream
            );
        }
        return;
    }
    const bool mxc_ok = rows >= 1 && rows <= 8 &&
        (K % kMcKC) == 0 &&
        ((K / kMcKC) == 1 || (part_opt != nullptr && (K / kMcKC) * rows <= 256)) &&
        ((reinterpret_cast<uintptr_t>(w) & 15u) == 0u) &&
        ((reinterpret_cast<uintptr_t>(x) & 15u) == 0u) &&
        ((ldX & 3) == 0);
    if (mxc_ok) {
        const int nb = (N + 31) >> 5;
        const int ks = K / kMcKC;
        float* pp = (ks > 1) ? part_opt : nullptr;
        const dim3 grid((unsigned)nb, (unsigned)ks);
        switch (rows) {
#define MXC_R(R) case R: mm3_mxc_kernel<R><<<grid, 256, 0, stream>>>(w, scale, N, K, x, ldX, y, ldY, beta, pp); break;
            MXC_R(1) MXC_R(2) MXC_R(3) MXC_R(4) MXC_R(5) MXC_R(6) MXC_R(7) MXC_R(8)
#undef MXC_R
        }
        if (ks > 1) {
            mm3_mx_reduce_kernel<<<(rows * N + 255) / 256, 256, 0, stream>>>(part_opt, ks, rows, N, y, ldY, beta);
        }
        return;
    }
    const int nb = (N + 31) >> 5;
    const int tiles = max(1, (K + 511) >> 9);
    int ks = 1;
    if (part_opt != nullptr) {
        while (ks * 2 <= tiles && nb * ks < 160 && (K % (ks * 2 * 512)) == 0) ks *= 2;
    }
    const int kc = (K + ks - 1) / ks;
    const int kc_aligned = ((kc + 511) >> 9) << 9;
    float* pp = (ks > 1) ? part_opt : nullptr;
    const dim3 grid((unsigned)nb, (unsigned)ks);
    switch (rows) {
#define MX_R(R) case R: mm3_mx_rows_kernel<R><<<grid, 256, 0, stream>>>(w, scale, N, K, kc_aligned, x, ldX, y, ldY, beta, pp); break;
        MX_R(1) MX_R(2) MX_R(3) MX_R(4) MX_R(5) MX_R(6) MX_R(7) MX_R(8)
        MX_R(9) MX_R(10) MX_R(11) MX_R(12) MX_R(13) MX_R(14) MX_R(15) MX_R(16)
#undef MX_R
    }
    if (ks > 1) {
        mm3_mx_reduce_kernel<<<(rows * N + 255) / 256, 256, 0, stream>>>(part_opt, ks, rows, N, y, ldY, beta);
    }
}

void launch_mm3_dspark_window_attn(
    const float* q,
    const uint16_t* kc_bf16,
    const uint16_t* vc_bf16,
    int ring,
    const int32_t* lo,
    int hi,
    int T,
    int Hq,
    int Hkv,
    float scale,
    float* part_opt,
    int splits,
    float* out,
    cudaStream_t stream
) {
    if (T <= 0 || Hq <= 0 || Hkv <= 0) return;
    const int S = (part_opt != nullptr && splits > 1) ? min(splits, kCombMaxSplits) : 1;
    dim3 grid((unsigned)T, (unsigned)Hkv, (unsigned)S);
    if (T <= 16 && Hq == 16 * Hkv) {
        mm3_attn_wide_kernel<0, false><<<grid, 512, 0, stream>>>(
            q, kc_bf16, vc_bf16, nullptr, nullptr, nullptr, nullptr,
            nullptr, lo, nullptr, hi, Hq, Hkv, 0, 0, 0, ring, scale, S, part_opt, out
        );
    } else {
        mm3_attn_kernel<0, false><<<grid, 128, 0, stream>>>(
            q, kc_bf16, vc_bf16, nullptr, nullptr, nullptr, nullptr,
            nullptr, lo, nullptr, hi, Hq, Hkv, 0, 0, 0, ring, scale, S, part_opt, out
        );
    }
    if (S > 1) {
        const int G = Hq / Hkv;
        mm3_attn_combine_kernel<<<dim3((unsigned)T, (unsigned)Hkv, (unsigned)G), 128, 0, stream>>>(
            part_opt, Hq, Hkv, S, 0, out
        );
    }
}

void launch_mm3_dspark_markov_greedy(
    const float* logits,
    const uint16_t* w2_bf16,
    const uint16_t* w1_bf16,
    int V,
    int R,
    int32_t* prev,
    int32_t* out,
    const float* cw_opt,
    const float* cb_opt,
    const float* hidden_opt,
    int D,
    float* conf_opt,
    float* part,
    float* bias_opt,
    cudaStream_t stream
) {
    if (V <= 0 || R != kMkR) return;
    float* pv = part;
    int32_t* pi = reinterpret_cast<int32_t*>(part + kMkBlocks);
    mm3_markov_bias_kernel<<<kMkBlocks, 256, 0, stream>>>(logits, w2_bf16, w1_bf16, prev, V, pv, pi, bias_opt);
    mm3_markov_final_kernel<<<1, 1024, 0, stream>>>(pv, pi, kMkBlocks, w1_bf16, prev, out, cw_opt, cb_opt, hidden_opt, D, R, conf_opt);
}

void launch_mm3_vision_patch_embed_f32(
    const float* px,
    const float* w_patch,
    int N,
    int D,
    int P,
    float* out,
    cudaStream_t stream
) {
    if (N <= 0 || D <= 0 || P <= 0) return;
    dim3 grid((unsigned)((D + 63) / 64), (unsigned)((N + 31) / 32));
    mm3_vision_patch_embed_f32_kernel<<<grid, 256, 0, stream>>>(px, w_patch, N, D, P, out);
}

void launch_mm3_vision_layernorm(
    const float* x,
    const float* w,
    const float* b,
    int rows,
    int dim,
    float eps,
    float* y_f32_opt,
    uint16_t* y_bf16_opt,
    cudaStream_t stream
) {
    if (rows <= 0 || dim <= 0) return;
    mm3_vision_layernorm_kernel<<<rows, 256, 0, stream>>>(x, w, b, dim, eps, y_f32_opt, y_bf16_opt);
}

void launch_mm3_vision_qkv_rope_3d(
    float* qkv,
    const float* bias_opt,
    const int32_t* pos3d,
    const float* inv_freq,
    int N,
    int dim,
    int heads,
    int hd,
    int nf,
    cudaStream_t stream
) {
    if (N <= 0 || dim <= 0) return;
    mm3_vision_qkv_rope_3d_kernel<<<N, 256, 0, stream>>>(qkv, bias_opt, pos3d, inv_freq, dim, heads, hd, nf);
}

void launch_mm3_vision_full_attn(
    const float* qkv,
    int N,
    int dim,
    int heads,
    int hd,
    float scale,
    float* out,
    cudaStream_t stream
) {
    if (N <= 0 || dim <= 0 || heads <= 0 || hd <= 0) return;
    dim3 grid((unsigned)((N + 3) / 4), (unsigned)heads);
    mm3_vision_full_attn_kernel<<<grid, 128, 0, stream>>>(qkv, N, dim, heads, hd, scale, out);
}

void launch_mm3_vision_bias_gelu(
    const float* y,
    const float* bias_opt,
    size_t n,
    int M,
    int apply_gelu,
    float* y_f32_opt,
    uint16_t* y_bf16_opt,
    cudaStream_t stream
) {
    if (n == 0 || M <= 0) return;
    const size_t nblk_s = (n + 255) / 256;
    const int blocks = (int)(nblk_s < 65535 ? nblk_s : 65535);
    mm3_vision_bias_gelu_kernel<<<blocks, 256, 0, stream>>>(y, bias_opt, n, M, apply_gelu, y_f32_opt, y_bf16_opt);
}

void launch_mm3_vision_residual_bias(
    float* h,
    const float* t,
    const float* bias_opt,
    size_t n,
    int M,
    cudaStream_t stream
) {
    if (n == 0 || M <= 0) return;
    const size_t nblk_s = (n + 255) / 256;
    const int blocks = (int)(nblk_s < 65535 ? nblk_s : 65535);
    mm3_vision_residual_bias_kernel<<<blocks, 256, 0, stream>>>(h, t, bias_opt, n, M);
}

void launch_mm3_gv_q6k_multi(
    int n_jobs,
    const void* w0, float* y0, int n0,
    const void* w1, float* y1, int n1,
    const void* w2, float* y2, int n2,
    const void* x_q8_1,
    int K,
    int M,
    cudaStream_t stream
) {
    if (n_jobs <= 0 || n_jobs > 3 || M <= 0 || K <= 0) return;
    const void* ws[3] = {w0, w1, w2};
    float* ys[3] = {y0, y1, y2};
    const int ns[3] = {n0, n1, n2};
    const size_t ld = static_cast<size_t>(K / 256) * 210;
    const auto* xq = reinterpret_cast<const block_q8_1*>(x_q8_1);

    Mm3GvJobs J{};
    int act_n = 0;
    int max_n = 0;
    const bool m_ok = (M >= 1 && M <= 8) && ((K % 1024) == 0);
    for (int i = 0; i < n_jobs; ++i) {
        if (ws[i] == nullptr || ys[i] == nullptr || ns[i] <= 0) continue;
        const bool w_ok = m_ok && ((reinterpret_cast<uintptr_t>(ws[i]) & 15u) == 0u);
        if (w_ok) {
            J.j[act_n] = Mm3GvJob{reinterpret_cast<const uint8_t*>(ws[i]), ld, ns[i], K, M, xq, ys[i]};
            max_n = max(max_n, ns[i]);
            ++act_n;
        } else {
            launch_linear_q8_1(GGML_TYPE_Q6_K, ws[i], x_q8_1, nullptr, ys[i], ns[i], K, M, stream);
        }
    }
    if (act_n <= 0) return;
    auto go = [&](auto tag_r, int r_val, auto pf_tag) {
        const unsigned gx = static_cast<unsigned>((max_n + 8 * r_val - 1) / (8 * r_val));
        const dim3 grid(gx, static_cast<unsigned>(act_n));
        constexpr int R_V = decltype(tag_r)::value;
        constexpr bool PF_V = decltype(pf_tag)::value;
        switch (M) {
            case 1: mm3_gv_kernel<GGML_TYPE_Q6_K, 1, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 2: mm3_gv_kernel<GGML_TYPE_Q6_K, 2, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 3: mm3_gv_kernel<GGML_TYPE_Q6_K, 3, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 4: mm3_gv_kernel<GGML_TYPE_Q6_K, 4, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 5: mm3_gv_kernel<GGML_TYPE_Q6_K, 5, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 6: mm3_gv_kernel<GGML_TYPE_Q6_K, 6, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 7: mm3_gv_kernel<GGML_TYPE_Q6_K, 7, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 8: mm3_gv_kernel<GGML_TYPE_Q6_K, 8, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
        }
    };
    if (M == 1) {
        go(std::integral_constant<int, 1>{}, 1, std::integral_constant<bool, true>{});
    } else {
        go(std::integral_constant<int, 2>{}, 2, std::integral_constant<bool, false>{});
    }
}

void launch_mm3_gv_q8_0_multi(
    int n_jobs,
    const void* w0, float* y0, int n0,
    const void* w1, float* y1, int n1,
    const void* w2, float* y2, int n2,
    const void* x_q8_1,
    int K,
    int M,
    cudaStream_t stream
) {
    if (n_jobs <= 0 || n_jobs > 3 || M <= 0 || K <= 0) return;
    const void* ws[3] = {w0, w1, w2};
    float* ys[3] = {y0, y1, y2};
    const int ns[3] = {n0, n1, n2};
    const size_t ld = static_cast<size_t>(K / 32) * 34;
    const auto* xq = reinterpret_cast<const block_q8_1*>(x_q8_1);

    Mm3GvJobs J{};
    int act_n = 0;
    int max_n = 0;
    const bool m_ok = (M >= 1 && M <= 8) && ((K % 128) == 0);
    for (int i = 0; i < n_jobs; ++i) {
        if (ws[i] == nullptr || ys[i] == nullptr || ns[i] <= 0) continue;
        const bool w_ok = m_ok && ((reinterpret_cast<uintptr_t>(ws[i]) & 15u) == 0u);
        if (w_ok) {
            J.j[act_n] = Mm3GvJob{reinterpret_cast<const uint8_t*>(ws[i]), ld, ns[i], K, M, xq, ys[i]};
            max_n = max(max_n, ns[i]);
            ++act_n;
        } else {
            launch_linear_q8_1(GGML_TYPE_Q8_0, ws[i], x_q8_1, nullptr, ys[i], ns[i], K, M, stream);
        }
    }
    if (act_n <= 0) return;
    auto go = [&](auto tag_r, int r_val, auto pf_tag) {
        const unsigned gx = static_cast<unsigned>((max_n + 8 * r_val - 1) / (8 * r_val));
        const dim3 grid(gx, static_cast<unsigned>(act_n));
        constexpr int R_V = decltype(tag_r)::value;
        constexpr bool PF_V = decltype(pf_tag)::value;
        switch (M) {
            case 1: mm3_gv_kernel<GGML_TYPE_Q8_0, 1, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 2: mm3_gv_kernel<GGML_TYPE_Q8_0, 2, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 3: mm3_gv_kernel<GGML_TYPE_Q8_0, 3, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 4: mm3_gv_kernel<GGML_TYPE_Q8_0, 4, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 5: mm3_gv_kernel<GGML_TYPE_Q8_0, 5, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 6: mm3_gv_kernel<GGML_TYPE_Q8_0, 6, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 7: mm3_gv_kernel<GGML_TYPE_Q8_0, 7, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
            case 8: mm3_gv_kernel<GGML_TYPE_Q8_0, 8, R_V, PF_V><<<grid, 256, 0, stream>>>(J); break;
        }
    };
    if (M == 1) {
        go(std::integral_constant<int, 1>{}, 1, std::integral_constant<bool, true>{});
    } else {
        go(std::integral_constant<int, 2>{}, 2, std::integral_constant<bool, false>{});
    }
}

void launch_mm3_iota(int32_t* out, int n, int pos0, cudaStream_t stream) {
    if (n <= 0 || out == nullptr) return;
    mm3_iota_kernel<<<(n + 255) / 256, 256, 0, stream>>>(out, n, pos0);
}

void launch_mm3_dspark_draft_block(
    const float* anc_mapped_f32,
    const float* mask_f32,
    int D,
    int B,
    int pos0,
    int window,
    int32_t anchor_id,
    float* h_out,
    int32_t* pos_out,
    int32_t* lo_out,
    int32_t* prev_out,
    cudaStream_t stream
) {
    if (D <= 0 || B <= 0) return;
    const size_t total = static_cast<size_t>(B) * static_cast<size_t>(D);
    const size_t nblk_s = (total + 255) / 256;
    const unsigned blocks = static_cast<unsigned>(nblk_s < 65535 ? nblk_s : 65535);
    mm3_dspark_draft_block_kernel<<<blocks, 256, 0, stream>>>(
        anc_mapped_f32, mask_f32, D, B, pos0, window, anchor_id, h_out, pos_out, lo_out, prev_out
    );
}

void launch_mm3_wait_u32(
    const uint32_t* wait_dev_ptr,
    uint32_t target_val,
    cudaStream_t stream
) {
    if (wait_dev_ptr == nullptr) return;
    mm3_wait_u32_kernel<<<1, 32, 0, stream>>>(wait_dev_ptr, target_val, 120ull * 1000000000ull);
}

}  // extern "C"
