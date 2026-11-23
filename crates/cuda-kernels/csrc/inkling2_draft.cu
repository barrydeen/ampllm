#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#include "common/sm86_common.cuh"

namespace {

constexpr float kLog2eDraft = 1.44269504088896340736f;
constexpr int kDraftD = 128;
constexpr int kDraftBlockMax = 8;
constexpr int kDraftDaNT = 3;
constexpr int kDraftWaNW = 4;
constexpr int kDraftSplitMax = 32;

__device__ __forceinline__ uint32_t f2_to_u32_d(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ float bf16_to_f32_d(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    return __uint_as_float(u);
}

__device__ __forceinline__ float round_bf16_in_reg(float v) {
    uint32_t u = __float_as_uint(v);
    if ((u & 0x7fffffffu) > 0x7f800000u) return v;
    uint32_t r = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return __uint_as_float(r);
}

// ============================================================================
// 1. Offline-at-load Markov pre-fold kernels (Q8_0 / FP16 lm_head -> int8 fold)
// ============================================================================

__global__ void dspark_unpack_q8p_f32_kernel(
    const int8_t* __restrict__ q,
    const uint16_t* __restrict__ qd,
    size_t n_elems,
    float* __restrict__ out
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    const float sc = fp16_bits_to_f32(__ldg(qd + (i >> 5)));
    out[i] = (float)__ldg(q + i) * sc;
}

__global__ void dspark_unpack_f16_f32_kernel(
    const uint16_t* __restrict__ src,
    size_t n_elems,
    float* __restrict__ out
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    out[i] = fp16_bits_to_f32(__ldg(src + i));
}

// Computes fold_w_f32[r, r0 + v] = sum_c dht[r, c] * dlm[v, c]
// Grid: (ceil(m/64), ceil(R/64)), Block: 256 threads (16x16)
__global__ void __launch_bounds__(256) dspark_fold_matmul_chunk_kernel(
    const float* __restrict__ dht, // [R, C]
    const float* __restrict__ dlm, // [m, C]
    int R,
    int m,
    int C,
    int r0,
    int n_rows,
    float* __restrict__ fold_w_f32 // [R, n_rows]
) {
    __shared__ float sA[16][65];
    __shared__ float sB[16][65];

    const int tid = threadIdx.x;
    const int ty = tid >> 4;   // 0..15 (rows along R in step 4 -> 64)
    const int tx = tid & 15;   // 0..15 (cols along m in step 4 -> 64)

    const int rb0 = (int)blockIdx.y * 64;
    const int cb0 = (int)blockIdx.x * 64;

    float acc[4][4] = {};

    for (int k0 = 0; k0 < C; k0 += 16) {
        // Load sA[16][64] (dht[rb0 + 0..63, k0 + 0..15]) and sB[16][64] (dlm[cb0 + 0..63, k0 + 0..15])
        // 256 threads each load 4 elements -> 1024 elements total = 64 * 16
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int idx = tid + p * 256; // 0..1023
            const int row = idx >> 4;      // 0..63
            const int kk  = idx & 15;      // 0..15
            const int gr = rb0 + row;
            const int gc = cb0 + row;
            const int gk = k0 + kk;
            sA[kk][row] = (gr < R && gk < C) ? __ldg(dht + (size_t)gr * C + gk) : 0.0f;
            sB[kk][row] = (gc < m && gk < C) ? __ldg(dlm + (size_t)gc * C + gk) : 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = sA[kk][ty * 4 + i];
                b[i] = sB[kk][tx * 4 + i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int r = rb0 + ty * 4 + i;
        if (r >= R) continue;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int v = cb0 + tx * 4 + j;
            if (v < m) {
                fold_w_f32[(size_t)r * n_rows + (r0 + v)] = acc[i][j];
            }
        }
    }
}

__global__ void dspark_fold_bias_chunk_kernel(
    const float* __restrict__ dlm, // [m, C]
    const float* __restrict__ dhb, // [C]
    int m,
    int C,
    uint16_t* __restrict__ fold_b_out // [m]
) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= m) return;
    const float* rptr = dlm + (size_t)row * C;
    float acc = 0.0f;
    for (int c = lane; c < C; c += 32) {
        acc = fmaf(__ldg(rptr + c), __ldg(dhb + c), acc);
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
        fold_b_out[row] = f32_to_fp16_bits(acc);
    }
}

__global__ void dspark_fold_quant_row_kernel(
    const float* __restrict__ fold_w_f32, // [R, n_rows]
    int n_rows,
    int8_t* __restrict__ fold_q,
    float* __restrict__ fold_sc
) {
    __shared__ float sh_max[8];
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row = fold_w_f32 + (size_t)r * n_rows;

    float mx = 0.0f;
    for (int v = tid; v < n_rows; v += 256) {
        mx = fmaxf(mx, fabsf(__ldg(row + v)));
    }
    mx = warp_reduce_max(mx);
    if ((tid & 31) == 0) sh_max[tid >> 5] = mx;
    __syncthreads();
    if (tid < 32) {
        float w = (tid < 8) ? sh_max[tid] : 0.0f;
        w = warp_reduce_max(w);
        if (tid == 0) sh_max[0] = (w > 0.0f) ? (w / 127.0f) : 1.0f;
    }
    __syncthreads();

    const float scale = sh_max[0];
    const float inv = 1.0f / scale;
    if (tid == 0) fold_sc[r] = scale;

    int8_t* qrow = fold_q + (size_t)r * n_rows;
    for (int v = tid; v < n_rows; v += 256) {
        float scaled = fmaxf(-127.0f, fminf(127.0f, __ldg(row + v) * inv));
        qrow[v] = (int8_t)lrintf(scaled);
    }
}

// ============================================================================
// 2. Context ingestion: tap mix + dflare RMSNorm + NeoX RoPE -> FP16 / FP8 KV
// ============================================================================

__global__ void dspark_mix_rms_kernel(
    const float* __restrict__ taps, // [n_tap, tap_stride_rows, C] starting at row r0
    int n_tap,
    size_t tap_stride_elems,
    int C,
    const float* __restrict__ alpha, // [n_tap]
    const float* __restrict__ w,     // [C]
    float eps,
    float* __restrict__ out_f32,     // nullable if only f16 wanted
    uint16_t* __restrict__ out_f16   // nullable if only f32 wanted
) {
    __shared__ float sh_sum[8];
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    float ssq = 0.0f;
    for (int c = tid; c < C; c += 256) {
        float m = 0.0f;
        for (int t = 0; t < n_tap; ++t) {
            m = fmaf(__ldg(alpha + t), __ldg(taps + (size_t)t * tap_stride_elems + (size_t)row * C + c), m);
        }
        if (out_f32) out_f32[(size_t)row * C + c] = m;
        ssq = fmaf(m, m, ssq);
    }
    ssq = warp_reduce_sum(ssq);
    if ((tid & 31) == 0) sh_sum[tid >> 5] = ssq;
    __syncthreads();
    if (tid < 32) {
        float wsum = (tid < 8) ? sh_sum[tid] : 0.0f;
        wsum = warp_reduce_sum(wsum);
        if (tid == 0) sh_sum[0] = rsqrtf(wsum / (float)C + eps);
    }
    __syncthreads();

    const float scale = sh_sum[0];
    for (int c = tid; c < C; c += 256) {
        float v = 0.0f;
        if (out_f32) {
            v = out_f32[(size_t)row * C + c];
        } else {
            for (int t = 0; t < n_tap; ++t) {
                v = fmaf(__ldg(alpha + t), __ldg(taps + (size_t)t * tap_stride_elems + (size_t)row * C + c), v);
            }
        }
        v *= scale * __ldg(w + c);
        if (out_f32) out_f32[(size_t)row * C + c] = v;
        if (out_f16) out_f16[(size_t)row * C + c] = f32_to_fp16_bits(v);
    }
}

// Grid: (n_kv, n_rows), Block: 64 threads (i = 0..63 -> handles pair i and i + 64)
__global__ void dspark_ctx_rope_kv_store_kernel(
    const float* __restrict__ k_in,      // [n_rows, n_kv * 128]
    const float* __restrict__ v_in,      // [n_rows, n_kv * 128]
    const float* __restrict__ k_norm_w,  // [128]
    int n_kv,
    int pos0,
    int cap,
    float theta,
    float eps,
    int is_fp8,
    float k_scale,
    float v_scale,
    void* __restrict__ kc_dst,
    void* __restrict__ vc_dst
) {
    __shared__ float sh_warp[2];
    const int g = blockIdx.x;
    const int r = blockIdx.y;
    const int i = threadIdx.x; // 0..63
    const int KD = n_kv * 128;

    const float* kp = k_in + (size_t)r * KD + (size_t)g * 128;
    const float* vp = v_in + (size_t)r * KD + (size_t)g * 128;
    const float ka = kp[i];
    const float kb = kp[i + 64];
    const float va = vp[i];
    const float vb = vp[i + 64];

    float ssq = ka * ka + kb * kb;
    ssq = warp_reduce_sum(ssq);
    if ((i & 31) == 0) sh_warp[i >> 5] = ssq;
    __syncthreads();
    const float rinv = rsqrtf((sh_warp[0] + sh_warp[1]) / 128.0f + eps);

    const float na = ka * rinv * __ldg(k_norm_w + i);
    const float nb = kb * rinv * __ldg(k_norm_w + i + 64);

    const float inv_freq = 1.0f / powf(theta, (float)(2 * i) / 128.0f);
    const float ang = (float)(pos0 + r) * inv_freq;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    const float ra = na * cs - nb * sn;
    const float rb = nb * cs + na * sn;

    int slot = pos0 + r;
    if (cap > 0) slot %= cap;
    const size_t off = (size_t)slot * KD + (size_t)g * 128;
    if (is_fp8) {
        uint8_t* kc8 = reinterpret_cast<uint8_t*>(kc_dst) + off;
        uint8_t* vc8 = reinterpret_cast<uint8_t*>(vc_dst) + off;
        kc8[i] = f32_to_e4m3fn(ra * k_scale);
        kc8[i + 64] = f32_to_e4m3fn(rb * k_scale);
        vc8[i] = f32_to_e4m3fn(va * v_scale);
        vc8[i + 64] = f32_to_e4m3fn(vb * v_scale);
    } else {
        uint16_t* kc16 = reinterpret_cast<uint16_t*>(kc_dst) + off;
        uint16_t* vc16 = reinterpret_cast<uint16_t*>(vc_dst) + off;
        kc16[i] = f32_to_fp16_bits(ra);
        kc16[i + 64] = f32_to_fp16_bits(rb);
        vc16[i] = f32_to_fp16_bits(va);
        vc16[i + 64] = f32_to_fp16_bits(vb);
    }
}

// ============================================================================
// 3. Drafter forward kernels (propose graph)
// ============================================================================

__global__ void dspark_init_block_kernel(
    const float* __restrict__ anchor_h,
    const float* __restrict__ mask,
    float* __restrict__ res,
    float* __restrict__ state,
    int* __restrict__ pos,
    int* __restrict__ toks,
    const int* __restrict__ par, // par[0]=P, par[1]=anchor (nullable)
    int P,
    int anchor,
    int T,
    int C,
    int R
) {
    if (par) {
        P = par[0];
        anchor = par[1];
    }
    const int t = blockIdx.y;
    const int c = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (t == 0 && blockIdx.x == 0) {
        if (threadIdx.x < T) pos[threadIdx.x] = P + threadIdx.x;
        if (threadIdx.x == 0) toks[0] = anchor;
    }
    if (t == 0 && c < R) {
        state[c] = 0.0f;
    }
    if (c < C) {
        res[(size_t)t * C + c] = __ldg((t == 0 ? anchor_h : mask) + c);
    }
}

// Residual add + RMSNorm -> F32 and FP16 outputs (fused: zero extra launches)
// If src_x == nullptr, only normalizes res -> h32, h16 (for layer 0 in-norm).
// If h16_row0_skip != 0 and t == 0, omits h16 row 0 (not needed, harmless if written too).
__global__ void dspark_add_rms_f16_kernel(
    const float* __restrict__ src_x, // nullable
    float* __restrict__ res,         // in/out [T, C]
    const float* __restrict__ w,     // [C]
    float eps,
    int T,
    int C,
    float* __restrict__ h32,         // [T, C]
    uint16_t* __restrict__ h16       // [T, C] (nullable)
) {
    __shared__ float sh_sum[8];
    const int t = blockIdx.x;
    if (t >= T) return;
    const int tid = threadIdx.x;
    const size_t base = (size_t)t * C;

    float ssq = 0.0f;
    for (int c = tid; c < C; c += 256) {
        float v = res[base + c];
        if (src_x) {
            v += src_x[base + c];
            res[base + c] = v;
        }
        ssq = fmaf(v, v, ssq);
    }
    ssq = warp_reduce_sum(ssq);
    if ((tid & 31) == 0) sh_sum[tid >> 5] = ssq;
    __syncthreads();
    if (tid < 32) {
        float wsum = (tid < 8) ? sh_sum[tid] : 0.0f;
        wsum = warp_reduce_sum(wsum);
        if (tid == 0) sh_sum[0] = rsqrtf(wsum / (float)C + eps);
    }
    __syncthreads();

    const float scale = sh_sum[0];
    for (int c = tid; c < C; c += 256) {
        const float v = res[base + c] * scale * __ldg(w + c);
        if (h32) h32[base + c] = v;
        if (h16) h16[base + c] = f32_to_fp16_bits(v);
    }
}

// Fused Q & K per-head RMSNorm + NeoX RoPE in one launch: Grid (H + KV, T), Block 64
__global__ void dspark_rms_neox_qk_kernel(
    float* __restrict__ q,          // [T, H * 128]
    float* __restrict__ k,          // [T, KV * 128]
    const float* __restrict__ qnw,  // [128]
    const float* __restrict__ knw,  // [128]
    const int* __restrict__ pos,    // [T]
    int H,
    int KV,
    float theta,
    float eps
) {
    __shared__ float sh_warp[2];
    const int idx = blockIdx.x;
    const int t = blockIdx.y;
    const int i = threadIdx.x; // 0..63
    const bool is_q = idx < H;
    const int h = is_q ? idx : (idx - H);
    float* p = (is_q ? q : k) + ((size_t)t * (is_q ? H : KV) + h) * 128;
    const float* nw = is_q ? qnw : knw;

    const float a = p[i];
    const float b = p[i + 64];
    float ssq = a * a + b * b;
    ssq = warp_reduce_sum(ssq);
    if ((i & 31) == 0) sh_warp[i >> 5] = ssq;
    __syncthreads();
    const float rinv = rsqrtf((sh_warp[0] + sh_warp[1]) / 128.0f + eps);

    const float na = a * rinv * __ldg(nw + i);
    const float nb = b * rinv * __ldg(nw + i + 64);

    const float inv_freq = 1.0f / powf(theta, (float)(2 * i) / 128.0f);
    const float ang = (float)pos[t] * inv_freq;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    p[i] = na * cs - nb * sn;
    p[i + 64] = nb * cs + na * sn;
}

template <class KVT>
__global__ void __launch_bounds__(kDraftWaNW * 32, 2)
dspark_attn_mma_kernel(
    const float* __restrict__ q,
    const KVT* __restrict__ kc,
    const KVT* __restrict__ vc,
    int k_lo,
    int k_hi,
    int nq,
    int H,
    int Hkv,
    float sc,
    int n_split,
    float* __restrict__ o_part,
    float2* __restrict__ ml_part,
    int cap,
    const int* __restrict__ par,
    int span
) {
    constexpr bool F8 = (sizeof(KVT) == 1);
    constexpr int NT = kDraftDaNT;

    if (par) {
        k_hi = par[0];
        k_lo = max(0, k_hi - span);
    }
    const int nc = k_hi - k_lo;
    const int lane = threadIdx.x & 31;
    const int gr = lane >> 2;
    const int tq = lane & 3;
    const int g = blockIdx.x;
    const int R = H / Hkv;
    const int KD = Hkv * kDraftD;
    const int NC = nq * R;
    const int QH = nq * H;
    const int w = (int)blockIdx.y * kDraftWaNW + (threadIdx.x >> 5);
    const int n_steps = (nc + 15) >> 4;
    const int n_act = min(n_split * kDraftWaNW, n_steps);
    if (w >= n_act) return;

    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = min(n_steps, w * per);
    const int st1 = min(n_steps, st0 + per);

    uint32_t qf[NT][kDraftD / 16][2];
    bool q_ok[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int cq = j * 8 + gr;
        q_ok[j] = cq < NC;
        const int qi = q_ok[j] ? cq / R : 0;
        const int hq = q_ok[j] ? cq % R : 0;
        const float4* qr = reinterpret_cast<const float4*>(q + ((size_t)qi * H + g * R + hq) * kDraftD);
        #pragma unroll
        for (int kk = 0; kk < kDraftD / 16; ++kk) {
            const float4 v = qr[kk * 4 + tq];
            qf[j][kk][0] = q_ok[j] ? f2_to_u32_d(v.x, v.y) : 0u;
            qf[j][kk][1] = q_ok[j] ? f2_to_u32_d(v.z, v.w) : 0u;
        }
    }

    float o[NT][kDraftD / 16][4];
    float ma[NT], mb[NT], la[NT], lb[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        ma[j] = -INFINITY; mb[j] = -INFINITY;
        la[j] = 0.0f; lb[j] = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < kDraftD / 16; ++jj) {
            o[j][jj][0] = 0.0f; o[j][jj][1] = 0.0f; o[j][jj][2] = 0.0f; o[j][jj][3] = 0.0f;
        }
    }

    const KVT* kg = kc + (size_t)g * kDraftD + tq * 4;
    const KVT* vg = vc + (size_t)g * kDraftD + gr * 16;
    const int src = tq * 8 + (gr >> 1);
    const uint32_t psel = (gr & 1) ? 0x7632u : 0x5410u;

    for (int st = st0; st < st1; ++st) {
        const int kt = st * 16;
        const int kmax = nc - 1 - kt;
        const int row0 = (cap > 0) ? ((k_lo + kt) % cap) : (k_lo + kt);
        auto row_off = [&](int key) {
            int rw = row0 + min(key, kmax);
            if (cap > 0 && rw >= cap) rw -= cap;
            return (size_t)rw * KD;
        };

        uint32_t ka[kDraftD / 16][4];
        {
            const KVT* p0 = kg + row_off(gr);
            const KVT* p1 = kg + row_off(gr + 8);
            if constexpr (F8) {
                #pragma unroll
                for (int kk = 0; kk < kDraftD / 16; ++kk) {
                    const uint32_t w0 = __ldg(reinterpret_cast<const uint32_t*>(p0 + kk * 16));
                    const uint32_t w1 = __ldg(reinterpret_cast<const uint32_t*>(p1 + kk * 16));
                    uint32_t t0, t1, t2, t3;
                    e4m3x4_h4(w0, t0, t1);
                    e4m3x4_h4(w1, t2, t3);
                    ka[kk][0] = t0; ka[kk][2] = t1; ka[kk][1] = t2; ka[kk][3] = t3;
                }
            } else {
                #pragma unroll
                for (int kk = 0; kk < kDraftD / 16; ++kk) {
                    const uint2 v0 = __ldg(reinterpret_cast<const uint2*>(p0 + kk * 16));
                    const uint2 v1 = __ldg(reinterpret_cast<const uint2*>(p1 + kk * 16));
                    ka[kk][0] = v0.x; ka[kk][2] = v0.y; ka[kk][1] = v1.x; ka[kk][3] = v1.y;
                }
            }
        }

        const bool v0k = (gr <= kmax), v1k = (gr + 8 <= kmax);
        uint32_t pb0[NT], pb1[NT];
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int kk = 0; kk < kDraftD / 16; ++kk) {
                mma16816_f16((kk & 1) ? sb : sa, ka[kk], qf[j][kk][0], qf[j][kk][1]);
            }
            const bool a_ok = j * 8 + 2 * tq < NC;
            const bool b_ok = j * 8 + 2 * tq + 1 < NC;
            const float c0 = (a_ok && v0k) ? (sa[0] + sb[0]) * sc : -INFINITY;
            const float c1 = (b_ok && v0k) ? (sa[1] + sb[1]) * sc : -INFINITY;
            const float c2 = (a_ok && v1k) ? (sa[2] + sb[2]) * sc : -INFINITY;
            const float c3 = (b_ok && v1k) ? (sa[3] + sb[3]) * sc : -INFINITY;

            float mxa = fmaxf(c0, c2);
            float mxb = fmaxf(c1, c3);
            #pragma unroll
            for (int sh = 4; sh < 32; sh <<= 1) {
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, sh));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, sh));
            }
            if (__ballot_sync(0xffffffffu, mxa > ma[j] || mxb > mb[j])) {
                const float mna = fmaxf(ma[j], mxa);
                const float mnb = fmaxf(mb[j], mxb);
                const float ca = (mna == -INFINITY) ? 1.0f : exp2f(ma[j] - mna);
                const float cb = (mnb == -INFINITY) ? 1.0f : exp2f(mb[j] - mnb);
                ma[j] = mna; mb[j] = mnb;
                la[j] *= ca; lb[j] *= cb;
                #pragma unroll
                for (int jj = 0; jj < kDraftD / 16; ++jj) {
                    o[j][jj][0] *= ca; o[j][jj][1] *= cb; o[j][jj][2] *= ca; o[j][jj][3] *= cb;
                }
            }
            const float ua = (ma[j] == -INFINITY) ? 0.0f : ma[j];
            const float ub = (mb[j] == -INFINITY) ? 0.0f : mb[j];
            const float p0 = exp2f(c0 - ua);
            const float p1 = exp2f(c1 - ub);
            const float p2 = exp2f(c2 - ua);
            const float p3 = exp2f(c3 - ub);
            la[j] += p0 + p2;
            lb[j] += p1 + p3;

            const uint32_t plo = f2_to_u32_d(p0, p1);
            const uint32_t phi = f2_to_u32_d(p2, p3);
            const uint32_t alo = __shfl_sync(0xffffffffu, plo, src);
            const uint32_t blo = __shfl_sync(0xffffffffu, plo, src + 4);
            const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src);
            const uint32_t bhi = __shfl_sync(0xffffffffu, phi, src + 4);
            pb0[j] = q_ok[j] ? __byte_perm(alo, blo, psel) : 0u;
            pb1[j] = q_ok[j] ? __byte_perm(ahi, bhi, psel) : 0u;
        }

        {
            const KVT* v0 = vg + row_off(2 * tq);
            const KVT* v1 = vg + row_off(2 * tq + 1);
            const KVT* v8 = vg + row_off(2 * tq + 8);
            const KVT* v9 = vg + row_off(2 * tq + 9);
            if constexpr (F8) {
                const uint4 r0 = __ldg(reinterpret_cast<const uint4*>(v0));
                const uint4 r1 = __ldg(reinterpret_cast<const uint4*>(v1));
                const uint4 r8 = __ldg(reinterpret_cast<const uint4*>(v8));
                const uint4 r9 = __ldg(reinterpret_cast<const uint4*>(v9));
                const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w};
                const uint32_t x1[4] = {r1.x, r1.y, r1.z, r1.w};
                const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w};
                const uint32_t x9[4] = {r9.x, r9.y, r9.z, r9.w};
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t a[4];
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x5140u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x5140u), a[2], a[3]);
                    #pragma unroll
                    for (int j = 0; j < NT; ++j) mma16816_f16(o[j][2 * i], a, pb0[j], pb1[j]);
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x7362u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x7362u), a[2], a[3]);
                    #pragma unroll
                    for (int j = 0; j < NT; ++j) mma16816_f16(o[j][2 * i + 1], a, pb0[j], pb1[j]);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 r0 = __ldg(reinterpret_cast<const uint4*>(v0) + i);
                    const uint4 r1 = __ldg(reinterpret_cast<const uint4*>(v1) + i);
                    const uint4 r8 = __ldg(reinterpret_cast<const uint4*>(v8) + i);
                    const uint4 r9 = __ldg(reinterpret_cast<const uint4*>(v9) + i);
                    const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w};
                    const uint32_t x1[4] = {r1.x, r1.y, r1.z, r1.w};
                    const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w};
                    const uint32_t x9[4] = {r9.x, r9.y, r9.z, r9.w};
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t a[4] = {
                            __byte_perm(x0[jj], x1[jj], 0x5410u),
                            __byte_perm(x0[jj], x1[jj], 0x7632u),
                            __byte_perm(x8[jj], x9[jj], 0x5410u),
                            __byte_perm(x8[jj], x9[jj], 0x7632u)
                        };
                        #pragma unroll
                        for (int j = 0; j < NT; ++j) mma16816_f16(o[j][4 * i + jj], a, pb0[j], pb1[j]);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        #pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
            la[j] += __shfl_xor_sync(0xffffffffu, la[j], sh);
            lb[j] += __shfl_xor_sync(0xffffffffu, lb[j], sh);
        }
        const int ca = j * 8 + 2 * tq;
        const int cb = ca + 1;
        if (ca < NC) {
            const size_t ix = (size_t)w * QH + (size_t)(ca / R) * H + g * R + (ca % R);
            float* op = o_part + ix * kDraftD + gr * 16;
            #pragma unroll
            for (int jj = 0; jj < kDraftD / 16; ++jj) {
                *reinterpret_cast<float2*>(op + 2 * jj) = make_float2(o[j][jj][0], o[j][jj][2]);
            }
            if (gr == 0) ml_part[ix] = make_float2(ma[j], la[j]);
        }
        if (cb < NC) {
            const size_t ix = (size_t)w * QH + (size_t)(cb / R) * H + g * R + (cb % R);
            float* op = o_part + ix * kDraftD + gr * 16;
            #pragma unroll
            for (int jj = 0; jj < kDraftD / 16; ++jj) {
                *reinterpret_cast<float2*>(op + 2 * jj) = make_float2(o[j][jj][1], o[j][jj][3]);
            }
            if (gr == 0) ml_part[ix] = make_float2(mb[j], lb[j]);
        }
    }
}

__global__ void dspark_attn_mma_merge_kernel(
    const float* __restrict__ o_part,
    const float2* __restrict__ ml_part,
    int n_split,
    int k_lo,
    int k_hi,
    const float* __restrict__ q,
    const float* __restrict__ kb,
    const float* __restrict__ vb,
    int nq,
    int H,
    int Hkv,
    float scale,
    float v_mul,
    uint16_t* __restrict__ out16,
    const int* __restrict__ par,
    int span
) {
    __shared__ float cw[kDraftSplitMax * kDraftWaNW];
    __shared__ float sh_red[4];

    if (par) {
        k_hi = par[0];
        k_lo = max(0, k_hi - span);
    }
    const int nc = k_hi - k_lo;
    const int n_steps = (nc + 15) >> 4;
    const int n_act = min(n_split * kDraftWaNW, n_steps);

    const int h = blockIdx.x;
    const int qi = blockIdx.y;
    const int d = threadIdx.x; // 0..127 (4 warps)
    const int g = h / (H / Hkv);
    const int QH = nq * H;
    const int ix = qi * H + h;
    const int KD = Hkv * kDraftD;
    const int warp = d >> 5;
    const int lane = d & 31;

    const float qd = q[(size_t)ix * kDraftD + d];
    float lb[kDraftBlockMax];
    float m = -INFINITY;

    for (int j = 0; j < nq; ++j) {
        float dot = qd * kb[(size_t)j * KD + (size_t)g * kDraftD + d];
        dot = warp_reduce_sum(dot);
        if (lane == 0) sh_red[warp] = dot;
        __syncthreads();
        const float sum = sh_red[0] + sh_red[1] + sh_red[2] + sh_red[3];
        __syncthreads();
        lb[j] = sum * scale * kLog2eDraft;
        m = fmaxf(m, lb[j]);
    }

    float mp = -INFINITY;
    for (int s = d; s < n_act; s += kDraftD) {
        mp = fmaxf(mp, ml_part[(size_t)s * QH + ix].x);
    }
    mp = warp_reduce_max(mp);
    if (lane == 0) sh_red[warp] = mp;
    __syncthreads();
    if (lane == 0 && warp == 0) {
        sh_red[0] = fmaxf(fmaxf(sh_red[0], sh_red[1]), fmaxf(sh_red[2], sh_red[3]));
    }
    __syncthreads();
    m = fmaxf(m, sh_red[0]);

    float den = 0.0f;
    for (int s = d; s < n_act; s += kDraftD) {
        const float2 ml = ml_part[(size_t)s * QH + ix];
        const float c = (ml.x == -INFINITY) ? 0.0f : exp2f(ml.x - m);
        cw[s] = c;
        den = fmaf(ml.y, c, den);
    }
    den = warp_reduce_sum(den);
    if (lane == 0) sh_red[warp] = den;
    __syncthreads();
    den = sh_red[0] + sh_red[1] + sh_red[2] + sh_red[3];

    float acc = 0.0f;
    for (int s = 0; s < n_act; ++s) {
        acc = fmaf(o_part[((size_t)s * QH + ix) * kDraftD + d], cw[s], acc);
    }
    acc *= v_mul;

    for (int j = 0; j < nq; ++j) {
        const float pj = exp2f(lb[j] - m);
        den += pj;
        acc = fmaf(pj, vb[(size_t)j * KD + (size_t)g * kDraftD + d], acc);
    }
    out16[(size_t)ix * kDraftD + d] = f32_to_fp16_bits(acc / den);
}

__global__ void dspark_silu_mul_f16_kernel(
    const float* __restrict__ gu, // [T, 2*F]
    int T,
    int F,
    uint16_t* __restrict__ out16
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)T * F;
    if (i >= total) return;
    const size_t t = i / F;
    const size_t f = i - t * F;
    const float g = gu[t * 2 * F + f];
    const float u = gu[t * 2 * F + F + f];
    const float s = g / (1.0f + expf(-g));
    out16[i] = f32_to_fp16_bits(s * u);
}

// ============================================================================
// 4. Markov RNN + Int8 Pre-folded Argmax (zero BF16 intermediate write!)
// ============================================================================

// Computes proj[0..3*R] = W_bf16[3*R, 2*R + 2*C] @ round_bf16([state; w1[*prev_tok]; h_step; f_rnn])
// Grid: (3*R + 3) / 4 = 192 blocks, Block: 128 threads (4 warps -> 1 warp per row)
__global__ void dspark_rnn_proj_fused_kernel(
    const uint16_t* __restrict__ w_bf16, // [3*R, 2*R + 2*C]
    const float* __restrict__ state,     // [R]
    const uint16_t* __restrict__ w1,     // [V, R] BF16
    const int32_t* __restrict__ prev_tok,
    const float* __restrict__ h_step,    // [C]
    const float* __restrict__ f_rnn,     // [C]
    int R,
    int C,
    float* __restrict__ proj             // [3*R]
) {
    const int row = (int)blockIdx.x * 4 + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int N = 3 * R;
    if (row >= N) return;
    const int K = 2 * R + 2 * C;
    const int tok = *prev_tok;

    const uint4* w4 = reinterpret_cast<const uint4*>(w_bf16 + (size_t)row * K) + lane;
    const uint4* emb4 = reinterpret_cast<const uint4*>(w1 + (size_t)tok * R);

    float acc = 0.0f;
    // K is a multiple of 256 (5120 = 20 * 256), each chunk processes 256 elements (8 per lane).
    const int n_chunks = K >> 8;
    for (int ch = 0; ch < n_chunks; ++ch) {
        const int k0 = ch * 256 + lane * 8;
        const uint4 wv = __ldg(w4 + ch * 32);
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};

        if (k0 + 7 < R) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float x0 = round_bf16_in_reg(__ldg(state + k0 + 2 * j));
                const float x1 = round_bf16_in_reg(__ldg(state + k0 + 2 * j + 1));
                acc = fmaf(bf16_to_f32_d(ww[j] & 0xffffu), x0, acc);
                acc = fmaf(bf16_to_f32_d(ww[j] >> 16), x1, acc);
            }
        } else if (k0 >= R && k0 + 7 < 2 * R) {
            const uint4 ev = __ldg(emb4 + ((k0 - R) >> 3));
            const uint32_t ee[4] = {ev.x, ev.y, ev.z, ev.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc = fmaf(bf16_to_f32_d(ww[j] & 0xffffu), bf16_to_f32_d(ee[j] & 0xffffu), acc);
                acc = fmaf(bf16_to_f32_d(ww[j] >> 16), bf16_to_f32_d(ee[j] >> 16), acc);
            }
        } else if (k0 >= 2 * R && k0 + 7 < 2 * R + C) {
            const int c0 = k0 - 2 * R;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float x0 = round_bf16_in_reg(__ldg(h_step + c0 + 2 * j));
                const float x1 = round_bf16_in_reg(__ldg(h_step + c0 + 2 * j + 1));
                acc = fmaf(bf16_to_f32_d(ww[j] & 0xffffu), x0, acc);
                acc = fmaf(bf16_to_f32_d(ww[j] >> 16), x1, acc);
            }
        } else if (k0 >= 2 * R + C) {
            const int c0 = k0 - (2 * R + C);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float x0 = round_bf16_in_reg(__ldg(f_rnn + c0 + 2 * j));
                const float x1 = round_bf16_in_reg(__ldg(f_rnn + c0 + 2 * j + 1));
                acc = fmaf(bf16_to_f32_d(ww[j] & 0xffffu), x0, acc);
                acc = fmaf(bf16_to_f32_d(ww[j] >> 16), x1, acc);
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k = k0 + j;
                const uint16_t wb = (j & 1) ? (ww[j >> 1] >> 16) : (ww[j >> 1] & 0xffffu);
                float xv;
                if (k < R) xv = round_bf16_in_reg(__ldg(state + k));
                else if (k < 2 * R) xv = bf16_to_f32_d(__ldg(w1 + (size_t)tok * R + (k - R)));
                else if (k < 2 * R + C) xv = round_bf16_in_reg(__ldg(h_step + (k - 2 * R)));
                else xv = round_bf16_in_reg(__ldg(f_rnn + (k - (2 * R + C))));
                acc = fmaf(bf16_to_f32_d(wb), xv, acc);
            }
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) proj[row] = acc;
}

__global__ void dspark_rnn_gates_scale_kernel(
    const float* __restrict__ proj,    // [3*R]
    const float* __restrict__ bias,    // [3*R]
    const float* __restrict__ fold_sc, // [R]
    int R,
    float* __restrict__ state,         // [R]
    float* __restrict__ fold_xs        // [R]
) {
    const int i = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const float gr = proj[i] + bias[i];
    const float cr = proj[R + i] + bias[R + i];
    const float orw = proj[2 * R + i] + bias[2 * R + i];
    const float gate = 1.0f / (1.0f + expf(-gr));
    state[i] = gate * state[i] + (1.0f - gate) * tanhf(cr);
    fold_xs[i] = tanhf(orw) * fold_sc[i];
}

__device__ __forceinline__ void argmax_better_d(float v, int i, float& best, int& idx) {
    if (v > best || (v == best && i < idx)) {
        best = v;
        idx = i;
    }
}

__device__ __forceinline__ void argmax_block_reduce_d(float& best, int& idx) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        argmax_better_d(ob, oi, best, idx);
    }
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        bv[warp] = best;
        bi[warp] = idx;
    }
    __syncthreads();
    if (warp == 0) {
        const int nw = (blockDim.x + 31) >> 5;
        best = (lane < nw) ? bv[lane] : -INFINITY;
        idx = (lane < nw) ? bi[lane] : 0x7fffffff;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            argmax_better_d(ob, oi, best, idx);
        }
    }
}

__global__ void __launch_bounds__(128, 4) dspark_argmax_gemv_i8_part_kernel(
    const int8_t* __restrict__ W,      // [R, ld]
    int ld,
    const float* __restrict__ xs,      // [R]
    const float* __restrict__ base,    // [n]
    const uint16_t* __restrict__ bf16, // [n]
    int R,
    int n,
    float* __restrict__ part_vals,
    int32_t* __restrict__ part_ids
) {
    extern __shared__ float sx[];
    for (int r = threadIdx.x; r < R; r += blockDim.x) sx[r] = xs[r];
    __syncthreads();

    const int gtid = (int)blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = (int)gridDim.x * blockDim.x;
    float best = -INFINITY;
    int idx = 0x7fffffff;

    const int n8 = n >> 3;
    for (int v8 = gtid; v8 < n8; v8 += gstride) {
        const int v = v8 << 3;
        float d[8] = {};
        const int8_t* wr = W + v;
        for (int r = 0; r < R; ++r, wr += ld) {
            const float xr = sx[r];
            const uint2 w = __ldg(reinterpret_cast<const uint2*>(wr));
            const uint32_t w0 = w.x, w1 = w.y;
            d[0] = fmaf((float)(int8_t)(w0 & 0xffu), xr, d[0]);
            d[1] = fmaf((float)(int8_t)((w0 >> 8) & 0xffu), xr, d[1]);
            d[2] = fmaf((float)(int8_t)((w0 >> 16) & 0xffu), xr, d[2]);
            d[3] = fmaf((float)(int8_t)(w0 >> 24), xr, d[3]);
            d[4] = fmaf((float)(int8_t)(w1 & 0xffu), xr, d[4]);
            d[5] = fmaf((float)(int8_t)((w1 >> 8) & 0xffu), xr, d[5]);
            d[6] = fmaf((float)(int8_t)((w1 >> 16) & 0xffu), xr, d[6]);
            d[7] = fmaf((float)(int8_t)(w1 >> 24), xr, d[7]);
        }
        const float4 b0 = __ldg(reinterpret_cast<const float4*>(base + v));
        const float4 b1 = __ldg(reinterpret_cast<const float4*>(base + v + 4));
        const uint4 bb = __ldg(reinterpret_cast<const uint4*>(bf16 + v));
        const float bvals[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        const uint32_t hwords[4] = {bb.x, bb.y, bb.z, bb.w};
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float f0 = fp16_bits_to_f32(hwords[c] & 0xffffu);
            const float f1 = fp16_bits_to_f32(hwords[c] >> 16);
            argmax_better_d(d[2 * c] + bvals[2 * c] + f0, v + 2 * c, best, idx);
            argmax_better_d(d[2 * c + 1] + bvals[2 * c + 1] + f1, v + 2 * c + 1, best, idx);
        }
    }
    for (int v = (n8 << 3) + gtid; v < n; v += gstride) {
        float d = 0.0f;
        const int8_t* wr = W + v;
        for (int r = 0; r < R; ++r, wr += ld) d = fmaf((float)__ldg(wr), sx[r], d);
        argmax_better_d(d + __ldg(base + v) + fp16_bits_to_f32(__ldg(bf16 + v)), v, best, idx);
    }
    argmax_block_reduce_d(best, idx);
    if (threadIdx.x == 0) {
        part_vals[blockIdx.x] = best;
        part_ids[blockIdx.x] = idx;
    }
}

__global__ void dspark_argmax_final_kernel(
    const float* __restrict__ part_vals,
    const int32_t* __restrict__ part_ids,
    int n_partials,
    int32_t* __restrict__ out_tok
) {
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int j = threadIdx.x; j < n_partials; j += blockDim.x) {
        argmax_better_d(part_vals[j], part_ids[j], best, idx);
    }
    argmax_block_reduce_d(best, idx);
    if (threadIdx.x == 0) *out_tok = idx;
}

static int dspark_attn_mma_blocks(int Hkv) {
    static int blocks[64] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 64) dev = 0;
    if (!blocks[dev]) {
        int n_sm = 0, occ16 = 0, occ8 = 0;
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ16, dspark_attn_mma_kernel<uint16_t>, kDraftWaNW * 32, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ8, dspark_attn_mma_kernel<uint8_t>, kDraftWaNW * 32, 0);
        int occ = occ16 < occ8 ? occ16 : occ8;
        if (occ < 1) occ = 1;
        blocks[dev] = n_sm * occ;
    }
    int ns = blocks[dev] / (Hkv > 0 ? Hkv : 1);
    if (ns < 1) ns = 1;
    if (ns > kDraftSplitMax) ns = kDraftSplitMax;
    return ns;
}

} // namespace

extern "C" {

void launch_dspark_fold_chunk(
    const int8_t* lm_q,     // nullable if lm_f16 != null
    const uint16_t* lm_qd,  // nullable
    const uint16_t* lm_f16, // nullable
    const float* dht,       // [R, C]
    const float* dhb,       // [C]
    float* dlm_tmp,         // [m, C]
    int r0,
    int m,
    int n_rows,
    int R,
    int C,
    float* fold_w_f32,      // [R, n_rows]
    uint16_t* fold_b,       // [n_rows]
    cudaStream_t stream
) {
    const size_t elems = (size_t)m * C;
    if (lm_f16) {
        dspark_unpack_f16_f32_kernel<<<(unsigned)((elems + 255) / 256), 256, 0, stream>>>(
            lm_f16 + (size_t)r0 * C, elems, dlm_tmp
        );
    } else {
        dspark_unpack_q8p_f32_kernel<<<(unsigned)((elems + 255) / 256), 256, 0, stream>>>(
            lm_q + (size_t)r0 * C, lm_qd + (size_t)r0 * (C / 32), elems, dlm_tmp
        );
    }
    dim3 g_mm((unsigned)((m + 63) / 64), (unsigned)((R + 63) / 64));
    dspark_fold_matmul_chunk_kernel<<<g_mm, 256, 0, stream>>>(
        dht, dlm_tmp, R, m, C, r0, n_rows, fold_w_f32
    );
    dspark_fold_bias_chunk_kernel<<<(unsigned)((m + 7) / 8), 256, 0, stream>>>(
        dlm_tmp, dhb, m, C, fold_b + r0
    );
}

void launch_dspark_fold_quant(
    const float* fold_w_f32, // [R, n_rows]
    int R,
    int n_rows,
    int8_t* fold_q,          // [R, n_rows]
    float* fold_sc,          // [R]
    cudaStream_t stream
) {
    dspark_fold_quant_row_kernel<<<(unsigned)R, 256, 0, stream>>>(
        fold_w_f32, n_rows, fold_q, fold_sc
    );
}

void launch_dspark_mix_rms(
    const float* taps,
    int n_tap,
    size_t tap_stride_elems,
    int r0,
    int rows,
    int C,
    const float* alpha,
    const float* w,
    float eps,
    float* out_f32,
    uint16_t* out_f16,
    cudaStream_t stream
) {
    if (rows <= 0) return;
    dspark_mix_rms_kernel<<<(unsigned)rows, 256, 0, stream>>>(
        taps + (size_t)r0 * C, n_tap, tap_stride_elems, C, alpha, w, eps, out_f32, out_f16
    );
}

void launch_dspark_ctx_rope_kv_store(
    const float* k_in,
    const float* v_in,
    const float* k_norm_w,
    int n_rows,
    int n_kv,
    int pos0,
    int cap,
    float theta,
    float eps,
    int is_fp8,
    float k_scale,
    float v_scale,
    void* kc_dst,
    void* vc_dst,
    cudaStream_t stream
) {
    if (n_rows <= 0) return;
    dim3 grid((unsigned)n_kv, (unsigned)n_rows);
    dspark_ctx_rope_kv_store_kernel<<<grid, 64, 0, stream>>>(
        k_in, v_in, k_norm_w, n_kv, pos0, cap, theta, eps, is_fp8, k_scale, v_scale, kc_dst, vc_dst
    );
}

void launch_dspark_init_block(
    const float* anchor_h,
    const float* mask,
    float* res,
    float* state,
    int32_t* pos,
    int32_t* toks,
    const int32_t* par,
    int P,
    int anchor,
    int T,
    int C,
    int R,
    cudaStream_t stream
) {
    dim3 grid((unsigned)((C + 255) / 256), (unsigned)T);
    dspark_init_block_kernel<<<grid, 256, 0, stream>>>(
        anchor_h, mask, res, state, pos, toks, par, P, anchor, T, C, R
    );
}

void launch_dspark_add_rms_f16(
    const float* src_x,
    float* res,
    const float* w,
    float eps,
    int T,
    int C,
    float* h32,
    uint16_t* h16,
    cudaStream_t stream
) {
    dspark_add_rms_f16_kernel<<<(unsigned)T, 256, 0, stream>>>(
        src_x, res, w, eps, T, C, h32, h16
    );
}

void launch_dspark_rms_neox_qk(
    float* q,
    float* k,
    const float* qnw,
    const float* knw,
    const int32_t* pos,
    int T,
    int H,
    int KV,
    float theta,
    float eps,
    cudaStream_t stream
) {
    dim3 grid((unsigned)(H + KV), (unsigned)T);
    dspark_rms_neox_qk_kernel<<<grid, 64, 0, stream>>>(
        q, k, qnw, knw, pos, H, KV, theta, eps
    );
}

void launch_dspark_attn_mma(
    const float* q,
    const void* kc,
    const void* vc,
    const float* kb,
    const float* vb,
    int k_lo,
    int k_hi,
    const int32_t* par,
    int span,
    int cap,
    int is_fp8,
    float k_inv,
    float v_inv,
    int nq,
    int H,
    int Hkv,
    float* scratch,
    uint16_t* out16,
    cudaStream_t stream
) {
    const int ns = dspark_attn_mma_blocks(Hkv);
    const float scale = rsqrtf((float)kDraftD);
    float* o_part = scratch;
    float2* ml_part = reinterpret_cast<float2*>(o_part + (size_t)ns * kDraftWaNW * nq * H * kDraftD);
    const int use_cap = (cap > 0) ? cap : (1 << 30);

    if (is_fp8) {
        dspark_attn_mma_kernel<uint8_t><<<dim3((unsigned)Hkv, (unsigned)ns), kDraftWaNW * 32, 0, stream>>>(
            q,
            reinterpret_cast<const uint8_t*>(kc),
            reinterpret_cast<const uint8_t*>(vc),
            k_lo, k_hi, nq, H, Hkv, scale * k_inv * kLog2eDraft, ns, o_part, ml_part, use_cap, par, span
        );
    } else {
        dspark_attn_mma_kernel<uint16_t><<<dim3((unsigned)Hkv, (unsigned)ns), kDraftWaNW * 32, 0, stream>>>(
            q,
            reinterpret_cast<const uint16_t*>(kc),
            reinterpret_cast<const uint16_t*>(vc),
            k_lo, k_hi, nq, H, Hkv, scale * k_inv * kLog2eDraft, ns, o_part, ml_part, use_cap, par, span
        );
    }
    dspark_attn_mma_merge_kernel<<<dim3((unsigned)H, (unsigned)nq), kDraftD, 0, stream>>>(
        o_part, ml_part, ns, k_lo, k_hi, q, kb, vb, nq, H, Hkv, scale, v_inv, out16, par, span
    );
}

void launch_dspark_silu_mul_f16(
    const float* gu,
    int T,
    int F,
    uint16_t* out16,
    cudaStream_t stream
) {
    const size_t n = (size_t)T * F;
    dspark_silu_mul_f16_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(gu, T, F, out16);
}

void launch_dspark_rnn_step_argmax(
    const uint16_t* rnn_w_bf16,
    const float* rnn_b,
    const uint16_t* w1_bf16,
    const int8_t* fold_q,
    const float* fold_sc,
    const uint16_t* fold_b,
    const float* base,
    const float* h_step,
    const float* f_rnn,
    int R,
    int C,
    int n_rows,
    int n_vocab,
    float* state,
    float* proj_tmp,
    float* fold_xs,
    float* part_vals,
    int32_t* part_ids,
    const int32_t* prev_tok,
    int32_t* out_tok,
    cudaStream_t stream
) {
    const unsigned proj_blocks = (unsigned)((3 * R + 3) / 4);
    dspark_rnn_proj_fused_kernel<<<proj_blocks, 128, 0, stream>>>(
        rnn_w_bf16, state, w1_bf16, prev_tok, h_step, f_rnn, R, C, proj_tmp
    );
    dspark_rnn_gates_scale_kernel<<<(unsigned)((R + 255) / 256), 256, 0, stream>>>(
        proj_tmp, rnn_b, fold_sc, R, state, fold_xs
    );
    const int n8 = (n_vocab >> 3) > 0 ? (n_vocab >> 3) : 1;
    int n_blocks = (n8 + 127) / 128;
    if (n_blocks < 2) n_blocks = 2;
    if (n_blocks > 512) n_blocks = 512;
    const size_t smem = (size_t)R * sizeof(float);
    dspark_argmax_gemv_i8_part_kernel<<<n_blocks, 128, smem, stream>>>(
        fold_q, n_rows, fold_xs, base, fold_b, R, n_vocab, part_vals, part_ids
    );
    dspark_argmax_final_kernel<<<1, 128, 0, stream>>>(part_vals, part_ids, n_blocks, out_tok);
}

} // extern "C"
