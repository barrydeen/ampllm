#include "common/sm86_common.cuh"

namespace {

constexpr int kFaD = 128;
constexpr int kWaNW = 4;
constexpr int kWaMin = 2;
constexpr int kGroupMax = 8;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kFp8KScale = 16.0f;
constexpr float kFp8VScale = 32.0f;
constexpr float kFp8KInv = 1.0f / 16.0f;
constexpr float kFp8VInv = 1.0f / 32.0f;

__device__ __forceinline__ uint32_t f2_to_u32(float a, float b) {
    const __half2 h2 = __float22half2_rn(make_float2(a, b));
    return (uint32_t)__half_as_ushort(__low2half(h2)) |
           ((uint32_t)__half_as_ushort(__high2half(h2)) << 16);
}

__device__ __forceinline__ uint32_t h2_sub(uint32_t a, uint32_t b) {
    uint32_t r;
    asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}

__device__ __forceinline__ uint32_t q8_pair(uint32_t w) {
    const uint32_t u = __byte_perm(w ^ 0x80808080u, 0x64646464u, 0x4140);
    return h2_sub(u, 0x64806480u);
}

__device__ __forceinline__ uint32_t q8_h2(uint32_t w, uint32_t scale) {
    return h2_mul(q8_pair(w), scale);
}

__device__ __forceinline__ float bsum(float v, float* sh) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (t < o) sh[t] += sh[t + o];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

__device__ __forceinline__ float bmax(float v, float* sh) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (t < o) sh[t] = fmaxf(sh[t], sh[t + o]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

// -----------------------------------------------------------------------------
// 1. Fast F32 -> F16 conversion & fused RMSNorm / AttnRes with F16 writeout
// -----------------------------------------------------------------------------

__global__ void f32_to_f16_vec_kernel(const float* __restrict__ x, uint16_t* __restrict__ y, size_t n) {
    const size_t i4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t n4 = n >> 2;
    if (i4 < n4) {
        const float4 v = __ldg(reinterpret_cast<const float4*>(x) + i4);
        reinterpret_cast<uint2*>(y)[i4] = make_uint2(f2_to_u32(v.x, v.y), f2_to_u32(v.z, v.w));
    }
    const size_t tail = (n4 << 2) + i4;
    if (i4 < (n & 3ULL) && tail < n) {
        y[tail] = f32_to_fp16_bits(x[tail]);
    }
}

struct MultiRmsF16Args {
    const float* x[3];
    const float* w[3];
    float* y32[3];
    uint16_t* y16[3];
};

__global__ void __launch_bounds__(256) rmsnorm_f16_multi_kernel(
    MultiRmsF16Args a, int dim, int M, float eps
) {
    __shared__ float sh[256];
    const int m = blockIdx.x;
    const int j = blockIdx.y;
    if (m >= M) return;
    const float* xr = a.x[j] + (size_t)m * dim;
    const float* w = a.w[j];
    float* y32r = a.y32[j] ? (a.y32[j] + (size_t)m * dim) : nullptr;
    uint16_t* y16r = a.y16[j] ? (a.y16[j] + (size_t)m * dim) : nullptr;

    const bool vec4 = (((uintptr_t)xr & 15) == 0) && (!w || ((uintptr_t)w & 15) == 0) &&
                      (!y32r || ((uintptr_t)y32r & 15) == 0) && (!y16r || ((uintptr_t)y16r & 7) == 0) &&
                      ((dim & 3) == 0);
    float acc = 0.0f;
    if (vec4) {
        const int n4 = dim >> 2;
        const float4* x4 = reinterpret_cast<const float4*>(xr);
        for (int d4 = threadIdx.x; d4 < n4; d4 += 256) {
            const float4 v = __ldg(x4 + d4);
            acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float rs = rsqrtf(bsum(acc, sh) / (float)dim + eps);
        const float4* w4 = reinterpret_cast<const float4*>(w);
        float4* y32_4 = reinterpret_cast<float4*>(y32r);
        uint2* y16_4 = reinterpret_cast<uint2*>(y16r);
        for (int d4 = threadIdx.x; d4 < n4; d4 += 256) {
            const float4 v = __ldg(x4 + d4);
            const float4 ww = w4 ? __ldg(w4 + d4) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float r0 = v.x * rs * ww.x, r1 = v.y * rs * ww.y;
            const float r2 = v.z * rs * ww.z, r3 = v.w * rs * ww.w;
            if (y32_4) y32_4[d4] = make_float4(r0, r1, r2, r3);
            if (y16_4) y16_4[d4] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
        }
    } else {
        for (int d = threadIdx.x; d < dim; d += 256) {
            const float v = xr[d];
            acc += v * v;
        }
        const float rs = rsqrtf(bsum(acc, sh) / (float)dim + eps);
        for (int d = threadIdx.x; d < dim; d += 256) {
            const float r = xr[d] * rs * (w ? w[d] : 1.0f);
            if (y32r) y32r[d] = r;
            if (y16r) y16r[d] = f32_to_fp16_bits(r);
        }
    }
}

__global__ void __launch_bounds__(256) attn_res_mix_rms_f16_kernel(
    const float* const* __restrict__ state_ptrs,
    int n_states,
    const float* __restrict__ queries,
    const float* __restrict__ norm_w0,
    const float* __restrict__ norm_w1,
    const float* __restrict__ norm_w2,
    float* __restrict__ norm_y0,
    float* __restrict__ norm_y1,
    float* __restrict__ norm_y2,
    uint16_t* __restrict__ norm_y16_0,
    uint16_t* __restrict__ norm_y16_1,
    uint16_t* __restrict__ norm_y16_2,
    int n_embd,
    int n_q,
    int M,
    float res_eps,
    float rms_eps
) {
    constexpr int NUM_WARPS = 8;
    constexpr int MAX_S = 16;
    __shared__ float sh_red[NUM_WARPS][MAX_S * 5];
    __shared__ float sc[4][MAX_S];
    __shared__ float rs_rms[4];

    const int m = blockIdx.x;
    if (m >= M) return;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c4 = n_embd >> 2;
    const int d4_0 = tid, d4_1 = tid + 256, d4_2 = tid + 512;
    const bool in0 = d4_0 < c4, in1 = d4_1 < c4, in2 = d4_2 < c4;

    float4 q0[4] = {}, q1[4] = {}, q2[4] = {};
    #pragma unroll
    for (int qi = 0; qi < 4; ++qi) {
        if (qi < n_q) {
            const float4* qp = reinterpret_cast<const float4*>(queries + (size_t)qi * n_embd);
            if (in0) q0[qi] = __ldg(qp + d4_0);
            if (in1) q1[qi] = __ldg(qp + d4_1);
            if (in2) q2[qi] = __ldg(qp + d4_2);
        }
    }

    for (int i = 0; i < n_states; ++i) {
        const float4* x4 = reinterpret_cast<const float4*>(state_ptrs[i] + (size_t)m * n_embd);
        float sum_sq = 0.0f;
        float qd[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (in0) {
            const float4 xv = __ldg(x4 + d4_0);
            sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
            #pragma unroll
            for (int qi = 0; qi < 4; ++qi)
                if (qi < n_q) qd[qi] += xv.x * q0[qi].x + xv.y * q0[qi].y + xv.z * q0[qi].z + xv.w * q0[qi].w;
        }
        if (in1) {
            const float4 xv = __ldg(x4 + d4_1);
            sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
            #pragma unroll
            for (int qi = 0; qi < 4; ++qi)
                if (qi < n_q) qd[qi] += xv.x * q1[qi].x + xv.y * q1[qi].y + xv.z * q1[qi].z + xv.w * q1[qi].w;
        }
        if (in2) {
            const float4 xv = __ldg(x4 + d4_2);
            sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
            #pragma unroll
            for (int qi = 0; qi < 4; ++qi)
                if (qi < n_q) qd[qi] += xv.x * q2[qi].x + xv.y * q2[qi].y + xv.z * q2[qi].z + xv.w * q2[qi].w;
        }
        sum_sq = warp_reduce_sum(sum_sq);
        #pragma unroll
        for (int qi = 0; qi < 4; ++qi)
            if (qi < n_q) qd[qi] = warp_reduce_sum(qd[qi]);
        if (lane == 0) {
            sh_red[warp][i * 5 + 0] = sum_sq;
            sh_red[warp][i * 5 + 1] = qd[0];
            sh_red[warp][i * 5 + 2] = qd[1];
            sh_red[warp][i * 5 + 3] = qd[2];
            sh_red[warp][i * 5 + 4] = qd[3];
        }
    }
    __syncthreads();

    if (warp == 0) {
        float ss = 0.0f;
        float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (lane < n_states) {
            #pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) {
                ss   += sh_red[w][lane * 5 + 0];
                d[0] += sh_red[w][lane * 5 + 1];
                d[1] += sh_red[w][lane * 5 + 2];
                d[2] += sh_red[w][lane * 5 + 3];
                d[3] += sh_red[w][lane * 5 + 4];
            }
        }
        const float r = (lane < n_states) ? rsqrtf(ss / (float)n_embd + res_eps) : 0.0f;
        const float inv_sqrt_c = rsqrtf((float)n_embd);
        #pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            if (qi >= n_q) break;
            const float s_i = (lane < n_states) ? (d[qi] * r * inv_sqrt_c) : -1e30f;
            const float mx = warp_reduce_max(s_i);
            const float e_i = (lane < n_states) ? expf(s_i - mx) : 0.0f;
            const float sum = warp_reduce_sum(e_i);
            if (lane < n_states) sc[qi][lane] = (e_i / sum) * r;
        }
    }
    __syncthreads();

    float4 o0[4] = {}, o1[4] = {}, o2[4] = {};
    for (int i = 0; i < n_states; ++i) {
        const float4* x4 = reinterpret_cast<const float4*>(state_ptrs[i] + (size_t)m * n_embd);
        float4 xv0 = in0 ? __ldg(x4 + d4_0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 xv1 = in1 ? __ldg(x4 + d4_1) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 xv2 = in2 ? __ldg(x4 + d4_2) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            if (qi >= n_q) break;
            const float w_qi = sc[qi][i];
            if (in0) {
                o0[qi].x += w_qi * xv0.x; o0[qi].y += w_qi * xv0.y;
                o0[qi].z += w_qi * xv0.z; o0[qi].w += w_qi * xv0.w;
            }
            if (in1) {
                o1[qi].x += w_qi * xv1.x; o1[qi].y += w_qi * xv1.y;
                o1[qi].z += w_qi * xv1.z; o1[qi].w += w_qi * xv1.w;
            }
            if (in2) {
                o2[qi].x += w_qi * xv2.x; o2[qi].y += w_qi * xv2.y;
                o2[qi].z += w_qi * xv2.z; o2[qi].w += w_qi * xv2.w;
            }
        }
    }

    #pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= n_q) break;
        float sq = 0.0f;
        if (in0) sq += o0[qi].x * o0[qi].x + o0[qi].y * o0[qi].y + o0[qi].z * o0[qi].z + o0[qi].w * o0[qi].w;
        if (in1) sq += o1[qi].x * o1[qi].x + o1[qi].y * o1[qi].y + o1[qi].z * o1[qi].z + o1[qi].w * o1[qi].w;
        if (in2) sq += o2[qi].x * o2[qi].x + o2[qi].y * o2[qi].y + o2[qi].z * o2[qi].z + o2[qi].w * o2[qi].w;
        sq = warp_reduce_sum(sq);
        if (lane == 0) sh_red[warp][qi] = sq;
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= n_q) break;
            float s_tot = (lane < NUM_WARPS) ? sh_red[lane][qi] : 0.0f;
            s_tot = warp_reduce_sum(s_tot);
            if (lane == 0) rs_rms[qi] = rsqrtf(s_tot / (float)n_embd + rms_eps);
        }
    }
    __syncthreads();

    const float* nw[3] = {norm_w0, norm_w1, norm_w2};
    float* ny[3] = {norm_y0, norm_y1, norm_y2};
    uint16_t* ny16[3] = {norm_y16_0, norm_y16_1, norm_y16_2};
    #pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= n_q) break;
        const float rs = rs_rms[qi];
        const float4* w4 = reinterpret_cast<const float4*>(nw[qi]);
        float4* y4 = ny[qi] ? reinterpret_cast<float4*>(ny[qi] + (size_t)m * n_embd) : nullptr;
        uint2* y16_4 = ny16[qi] ? reinterpret_cast<uint2*>(ny16[qi] + (size_t)m * n_embd) : nullptr;
        if (in0) {
            const float4 ww = __ldg(w4 + d4_0);
            const float a = o0[qi].x * rs * ww.x, b = o0[qi].y * rs * ww.y;
            const float c = o0[qi].z * rs * ww.z, d = o0[qi].w * rs * ww.w;
            if (y4) y4[d4_0] = make_float4(a, b, c, d);
            if (y16_4) y16_4[d4_0] = make_uint2(f2_to_u32(a, b), f2_to_u32(c, d));
        }
        if (in1) {
            const float4 ww = __ldg(w4 + d4_1);
            const float a = o1[qi].x * rs * ww.x, b = o1[qi].y * rs * ww.y;
            const float c = o1[qi].z * rs * ww.z, d = o1[qi].w * rs * ww.w;
            if (y4) y4[d4_1] = make_float4(a, b, c, d);
            if (y16_4) y16_4[d4_1] = make_uint2(f2_to_u32(a, b), f2_to_u32(c, d));
        }
        if (in2) {
            const float4 ww = __ldg(w4 + d4_2);
            const float a = o2[qi].x * rs * ww.x, b = o2[qi].y * rs * ww.y;
            const float c = o2[qi].z * rs * ww.z, d = o2[qi].w * rs * ww.w;
            if (y4) y4[d4_2] = make_float4(a, b, c, d);
            if (y16_4) y16_4[d4_2] = make_uint2(f2_to_u32(a, b), f2_to_u32(c, d));
        }
    }
}

// -----------------------------------------------------------------------------
// 2. Planar Q8_0 Tensor Core GEMM (m16n8k16), Fused QKVrg, and Fused Output-Argmax
// -----------------------------------------------------------------------------

template <int Splits, int Ahead>
__device__ __forceinline__ void q8_mma_tile_body(
    const int8_t* __restrict__ q,
    const uint16_t* __restrict__ scales,
    int N,
    int K,
    const uint16_t* __restrict__ x,
    int x_ld,
    int T,
    int tile_row,
    int tile_tok0,
    float* __restrict__ out,
    int y_ld,
    float4* sh_part,
    int row_offset_y
) {
    const int lane = threadIdx.x & 31;
    const int split = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int row = 16 * tile_row + g;
    const int token = tile_tok0 + g;
    const int row0 = min(row, N - 1);
    const int row1 = min(row + 8, N - 1);
    const int blocks = K >> 5;

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    struct QStep {
        uint2 a0, a1;
        uint32_t s0, s1;
        uint4 v;
    };
    auto ld = [&](int i) {
        const int col = i * 32 + t * 8;
        QStep r;
        r.a0 = __ldg(reinterpret_cast<const uint2*>(q + (size_t)row0 * K + col));
        r.a1 = __ldg(reinterpret_cast<const uint2*>(q + (size_t)row1 * K + col));
        r.s0 = __ldg(scales + (size_t)row0 * blocks + i);
        r.s1 = __ldg(scales + (size_t)row1 * blocks + i);
        r.v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)min(token, T - 1) * x_ld + col));
        return r;
    };

    QStep pipe[Ahead + 1];
    #pragma unroll
    for (int j = 0; j <= Ahead; ++j) {
        pipe[j] = ld(min(split + j * Splits, blocks - 1));
    }
    for (int i = split; i < blocks; i += Splits) {
        const QStep cur = pipe[0];
        #pragma unroll
        for (int j = 0; j < Ahead; ++j) pipe[j] = pipe[j + 1];
        if (i + (Ahead + 1) * Splits < blocks) {
            pipe[Ahead] = ld(i + (Ahead + 1) * Splits);
        }
        const uint2 a0 = cur.a0, a1 = cur.a1;
        uint32_t s0 = cur.s0, s1 = cur.s1;
        s0 |= s0 << 16;
        s1 |= s1 << 16;
        const uint4 v = cur.v;
        uint32_t a[4] = {q8_h2(a0.x, s0), q8_h2(a1.x, s1), q8_h2(a0.y, s0), q8_h2(a1.y, s1)};
        mma16816_f16(acc, a, v.x, v.z);
        a[0] = q8_h2(a0.x >> 16, s0);
        a[1] = q8_h2(a1.x >> 16, s1);
        a[2] = q8_h2(a0.y >> 16, s0);
        a[3] = q8_h2(a1.y >> 16, s1);
        mma16816_f16(acc, a, v.y, v.w);
    }

    sh_part[split * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    __syncthreads();
    if (split != 0) return;

    float4 v = sh_part[lane];
    #pragma unroll
    for (int i = 1; i < Splits; ++i) {
        const float4 z = sh_part[i * 32 + lane];
        v.x += z.x;
        v.y += z.y;
        v.z += z.z;
        v.w += z.w;
    }

    const int token0 = tile_tok0 + 2 * t;
    const int out_row = row_offset_y + row;
    if (token0 < T) {
        if (row < N) out[(size_t)token0 * y_ld + out_row] = v.x;
        if (row + 8 < N) out[(size_t)token0 * y_ld + out_row + 8] = v.z;
    }
    if (token0 + 1 < T) {
        if (row < N) out[(size_t)(token0 + 1) * y_ld + out_row] = v.y;
        if (row + 8 < N) out[(size_t)(token0 + 1) * y_ld + out_row + 8] = v.w;
    }
}

template <int Splits, int Ahead>
__global__ void __launch_bounds__(Splits * 32) q8_mma_tile_kernel(
    const int8_t* __restrict__ q,
    const uint16_t* __restrict__ scales,
    int N,
    int K,
    const uint16_t* __restrict__ x,
    int x_ld,
    int T,
    float* __restrict__ out,
    int y_ld
) {
    __shared__ float4 part[Splits * 32];
    q8_mma_tile_body<Splits, Ahead>(
        q, scales, N, K, x, x_ld, T,
        (int)blockIdx.x, (int)blockIdx.y * 8,
        out, y_ld, part, 0
    );
}

struct Q8QkvrgSegs {
    const int8_t* q[5];
    const uint16_t* d[5];
    const uint16_t* x[5];
    float* y[5];
    int n[5];
    int t_off[5]; // cumulative tile offset (in 16-row tiles)
};

template <int Splits, int Ahead>
__global__ void __launch_bounds__(Splits * 32) q8_mma_qkvrg_kernel(
    Q8QkvrgSegs s,
    int K,
    int T
) {
    __shared__ float4 part[Splits * 32];
    const int gt = (int)blockIdx.x;
    int seg = 4;
    if (gt < s.t_off[1]) seg = 0;
    else if (gt < s.t_off[2]) seg = 1;
    else if (gt < s.t_off[3]) seg = 2;
    else if (gt < s.t_off[4]) seg = 3;

    const int local_tile = gt - s.t_off[seg];
    const int n_seg = s.n[seg];
    if (n_seg <= 0) return;

    q8_mma_tile_body<Splits, Ahead>(
        s.q[seg], s.d[seg], n_seg, K, s.x[seg], K, T,
        local_tile, (int)blockIdx.y * 8,
        s.y[seg], n_seg, part, 0
    );
}

// Persistent-CTA fused output head (planar Q8_0) + per-block online argmax for T = 1..8!
template <int Splits, int Ahead>
__global__ void __launch_bounds__(Splits * 32) q8_mma_argmax_part_kernel(
    const int8_t* __restrict__ q,
    const uint16_t* __restrict__ scales,
    int N,
    int K,
    const uint16_t* __restrict__ x,
    int x_ld,
    int T,
    float* __restrict__ part_vals,   // [T, gridDim.x]
    int32_t* __restrict__ part_ids   // [T, gridDim.x]
) {
    __shared__ float4 sh_part[Splits * 32];
    __shared__ float s_tile[16][8];

    const int lane = threadIdx.x & 31;
    const int split = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int blocks = K >> 5;
    const int n_tiles = (N + 15) >> 4;

    float best_v[8];
    int32_t best_i[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        best_v[j] = -INFINITY;
        best_i[j] = 0x7fffffff;
    }

    struct QStep {
        uint2 a0, a1;
        uint32_t s0, s1;
        uint4 v;
    };

    for (int tile = (int)blockIdx.x; tile < n_tiles; tile += (int)gridDim.x) {
        const int row = 16 * tile + g;
        const int row0 = min(row, N - 1);
        const int row1 = min(row + 8, N - 1);
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

        auto ld = [&](int i) {
            const int col = i * 32 + t * 8;
            QStep r;
            r.a0 = __ldg(reinterpret_cast<const uint2*>(q + (size_t)row0 * K + col));
            r.a1 = __ldg(reinterpret_cast<const uint2*>(q + (size_t)row1 * K + col));
            r.s0 = __ldg(scales + (size_t)row0 * blocks + i);
            r.s1 = __ldg(scales + (size_t)row1 * blocks + i);
            r.v = __ldg(reinterpret_cast<const uint4*>(x + (size_t)min(g, T - 1) * x_ld + col));
            return r;
        };

        QStep pipe[Ahead + 1];
        #pragma unroll
        for (int j = 0; j <= Ahead; ++j) {
            pipe[j] = ld(min(split + j * Splits, blocks - 1));
        }
        for (int i = split; i < blocks; i += Splits) {
            const QStep cur = pipe[0];
            #pragma unroll
            for (int j = 0; j < Ahead; ++j) pipe[j] = pipe[j + 1];
            if (i + (Ahead + 1) * Splits < blocks) {
                pipe[Ahead] = ld(i + (Ahead + 1) * Splits);
            }
            const uint2 a0 = cur.a0, a1 = cur.a1;
            uint32_t s0 = cur.s0, s1 = cur.s1;
            s0 |= s0 << 16;
            s1 |= s1 << 16;
            const uint4 v = cur.v;
            uint32_t a[4] = {q8_h2(a0.x, s0), q8_h2(a1.x, s1), q8_h2(a0.y, s0), q8_h2(a1.y, s1)};
            mma16816_f16(acc, a, v.x, v.z);
            a[0] = q8_h2(a0.x >> 16, s0);
            a[1] = q8_h2(a1.x >> 16, s1);
            a[2] = q8_h2(a0.y >> 16, s0);
            a[3] = q8_h2(a1.y >> 16, s1);
            mma16816_f16(acc, a, v.y, v.w);
        }

        sh_part[split * 32 + lane] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        __syncthreads();

        if (split == 0) {
            float4 v = sh_part[lane];
            #pragma unroll
            for (int i = 1; i < Splits; ++i) {
                const float4 z = sh_part[i * 32 + lane];
                v.x += z.x; v.y += z.y; v.z += z.z; v.w += z.w;
            }
            const int tok0 = 2 * t;
            s_tile[g][tok0] = v.x;
            s_tile[g][tok0 + 1] = v.y;
            s_tile[g + 8][tok0] = v.z;
            s_tile[g + 8][tok0 + 1] = v.w;
            __syncwarp();
            if (lane < 16) {
                const int r_abs = 16 * tile + lane;
                if (r_abs < N) {
                    #pragma unroll
                    for (int tk = 0; tk < 8; ++tk) {
                        if (tk < T) {
                            const float val = s_tile[lane][tk];
                            if (val > best_v[tk] || (val == best_v[tk] && r_abs < best_i[tk])) {
                                best_v[tk] = val;
                                best_i[tk] = r_abs;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (split == 0) {
        const int stride = (int)gridDim.x;
        for (int tk = 0; tk < T; ++tk) {
            float bv = (lane < 16) ? best_v[tk] : -INFINITY;
            int32_t bi = (lane < 16) ? best_i[tk] : 0x7fffffff;
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
                const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (lane == 0) {
                part_vals[(size_t)tk * stride + blockIdx.x] = bv;
                part_ids[(size_t)tk * stride + blockIdx.x] = bi;
            }
        }
    }
}

__global__ void __launch_bounds__(256) q8_mma_argmax_final_kernel(
    const float* __restrict__ part_vals,
    const int32_t* __restrict__ part_ids,
    int n_parts,
    int T,
    int32_t* __restrict__ out_ids
) {
    const int tk = (int)blockIdx.y;
    if (tk >= T) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;

    float bv = -INFINITY;
    int32_t bi = 0x7fffffff;
    for (int i = tid; i < n_parts; i += 256) {
        const float v = part_vals[(size_t)tk * n_parts + i];
        const int32_t id = part_ids[(size_t)tk * n_parts + i];
        if (v > bv || (v == bv && id < bi)) {
            bv = v;
            bi = id;
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
        const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __shared__ float sv[8];
    __shared__ int32_t si[8];
    if (lane == 0) { sv[wid] = bv; si[wid] = bi; }
    __syncthreads();
    if (wid == 0) {
        bv = (lane < 8) ? sv[lane] : -INFINITY;
        bi = (lane < 8) ? si[lane] : 0x7fffffff;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) out_ids[tk] = bi;
    }
}

// -----------------------------------------------------------------------------
// 3. Short-Conv with Rollback Recording + Single-Launch Rollback Across All Slots
// -----------------------------------------------------------------------------

__global__ void short_conv_rec_kernel(
    const float* __restrict__ cur,
    const float* __restrict__ static_w,
    const float* __restrict__ dyn_proj,
    const float* __restrict__ add_res,
    float* __restrict__ history,
    float* __restrict__ out,
    float* __restrict__ rec_old_hist,
    float* __restrict__ rec_x,
    int n_sconv_channels,
    int slot_idx,
    int n_channels,
    int n_heads,
    int M,
    int conv_offset,
    int n_conv
) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n_channels) return;

    const int head_dim = (n_heads > 0) ? (n_channels / n_heads) : n_channels;
    const int head = (n_heads > 0) ? (ch / head_dim) : 0;

    const int g_ch = conv_offset + ch;
    float* hist_ch = history + (size_t)g_ch * 3;
    float h0 = hist_ch[0];
    float h1 = hist_ch[1];
    float h2 = hist_ch[2];

    if (rec_old_hist != nullptr) {
        float* rh = rec_old_hist + ((size_t)slot_idx * n_sconv_channels + g_ch) * 3;
        rh[0] = h0;
        rh[1] = h1;
        rh[2] = h2;
    }

    const float* sw = static_w + (size_t)ch * 4;
    const float sw0 = sw[0], sw1 = sw[1], sw2 = sw[2], sw3 = sw[3];
    const float inv_scale = rsqrtf((float)((dyn_proj ? 2 : 1) * n_conv + 1));

    for (int t = 0; t < M; ++t) {
        const float xt = cur[(size_t)t * n_channels + ch];
        if (rec_x != nullptr) {
            rec_x[((size_t)slot_idx * 8 + t) * n_sconv_channels + g_ch] = xt;
        }
        float conv_val = h0 * sw0 + h1 * sw1 + h2 * sw2 + xt * sw3;
        if (dyn_proj) {
            const float* dp = dyn_proj + ((size_t)t * n_heads + head) * 4;
            conv_val += xt * dp[0] + h2 * dp[1] + h1 * dp[2] + h0 * dp[3];
        }
        float y = (xt + conv_val) * inv_scale;
        if (add_res) {
            y += add_res[(size_t)t * n_channels + ch];
        }
        out[(size_t)t * n_channels + ch] = y;
        h0 = h1;
        h1 = h2;
        h2 = xt;
    }

    hist_ch[0] = h0;
    hist_ch[1] = h1;
    hist_ch[2] = h2;
}

__global__ void short_conv_kv_pair_rec_kernel(
    const float* __restrict__ k_in,
    const float* __restrict__ k_static_w,
    const float* __restrict__ k_dyn_proj,
    float* __restrict__ k_out,
    const float* __restrict__ v_in,
    const float* __restrict__ v_static_w,
    const float* __restrict__ v_dyn_proj,
    float* __restrict__ v_out,
    float* __restrict__ history,
    float* __restrict__ rec_old_hist,
    float* __restrict__ rec_x,
    int n_sconv_channels,
    int slot_idx,
    int kv_dim,
    int n_heads,
    int M,
    int n_conv
) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= kv_dim) return;
    const bool is_v = (blockIdx.y != 0);
    const float* cur = is_v ? v_in : k_in;
    const float* static_w = is_v ? v_static_w : k_static_w;
    const float* dyn_proj = is_v ? v_dyn_proj : k_dyn_proj;
    float* out = is_v ? v_out : k_out;
    const int conv_offset = is_v ? kv_dim : 0;

    const int head_dim = (n_heads > 0) ? (kv_dim / n_heads) : kv_dim;
    const int head = (n_heads > 0) ? (ch / head_dim) : 0;

    const int g_ch = conv_offset + ch;
    float* hist_ch = history + (size_t)g_ch * 3;
    float h0 = hist_ch[0];
    float h1 = hist_ch[1];
    float h2 = hist_ch[2];

    if (rec_old_hist != nullptr) {
        float* rh = rec_old_hist + ((size_t)slot_idx * n_sconv_channels + g_ch) * 3;
        rh[0] = h0;
        rh[1] = h1;
        rh[2] = h2;
    }

    const float* sw = static_w + (size_t)ch * 4;
    const float sw0 = sw[0], sw1 = sw[1], sw2 = sw[2], sw3 = sw[3];
    const float inv_scale = rsqrtf((float)((dyn_proj ? 2 : 1) * n_conv + 1));

    for (int t = 0; t < M; ++t) {
        const float xt = cur[(size_t)t * kv_dim + ch];
        if (rec_x != nullptr) {
            rec_x[((size_t)slot_idx * 8 + t) * n_sconv_channels + g_ch] = xt;
        }
        float conv_val = h0 * sw0 + h1 * sw1 + h2 * sw2 + xt * sw3;
        if (dyn_proj) {
            const float* dp = dyn_proj + ((size_t)t * n_heads + head) * 4;
            conv_val += xt * dp[0] + h2 * dp[1] + h1 * dp[2] + h0 * dp[3];
        }
        out[(size_t)t * kv_dim + ch] = (xt + conv_val) * inv_scale;
        h0 = h1;
        h1 = h2;
        h2 = xt;
    }

    hist_ch[0] = h0;
    hist_ch[1] = h1;
    hist_ch[2] = h2;
}

// -----------------------------------------------------------------------------
// 3b. Software-Pipelined Exact F32 Planar Q8_0 Batched GEMV (`mv_q8p_colsx`)
// -----------------------------------------------------------------------------

struct Q8QkvrgF32Segs {
    const int8_t* q[5];
    const uint16_t* d[5];
    const float* x[5];
    float* y[5];
    int n[5];
    int cta_off[5];
};

template <int WARPS, int R, int T, int NIT, int DW>
__device__ __forceinline__ void mv_q8p_colsx_body(
    const int8_t* __restrict__ q,
    const uint16_t* __restrict__ d,
    int N,
    int K,
    const float* __restrict__ x,
    int x_ld,
    float* __restrict__ y,
    int y_ld,
    int row0
) {
    const int lane = threadIdx.x & 31;
    const uint32_t* qr[R];
    const uint16_t* dr[R];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        const int row = min(row0 + i, N - 1);
        qr[i] = reinterpret_cast<const uint32_t*>(q + (size_t)row * K);
        dr[i] = d + (size_t)row * (K >> 5);
    }

    uint32_t w[R][DW + 1];
    uint16_t dh[R][DW + 1];
    float4 xv[T][2];
    float acc[R][T];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        #pragma unroll
        for (int t = 0; t < T; ++t) acc[i][t] = 0.0f;
    }

    #pragma unroll
    for (int a = 0; a < DW && a < NIT; ++a) {
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            w[i][a] = __ldg(qr[i] + a * 32 + lane);
            dh[i][a] = __ldg(dr[i] + a * 4 + (lane >> 3));
        }
    }
    if constexpr (NIT > 0) {
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            xv[t][0] = __ldg(reinterpret_cast<const float4*>(x + (size_t)t * x_ld) + lane);
        }
    }

    #pragma unroll
    for (int it = 0; it < NIT; ++it) {
        const int ws = it % (DW + 1);
        const int xs = it & 1;
        if (it + DW < NIT) {
            const int nws = (it + DW) % (DW + 1);
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                w[i][nws] = __ldg(qr[i] + (it + DW) * 32 + lane);
                dh[i][nws] = __ldg(dr[i] + (it + DW) * 4 + (lane >> 3));
            }
        }
        if (it + 1 < NIT) {
            const int nxs = (it + 1) & 1;
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                xv[t][nxs] = __ldg(reinterpret_cast<const float4*>(x + (size_t)t * x_ld) + (it + 1) * 32 + lane);
            }
        }
        float f[R][4], dv[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            const uint32_t ww = w[i][ws];
            dv[i] = fp16_bits_to_f32(dh[i][ws]);
            f[i][0] = (float)(int8_t)(ww & 0xFFu);
            f[i][1] = (float)(int8_t)((ww >> 8) & 0xFFu);
            f[i][2] = (float)(int8_t)((ww >> 16) & 0xFFu);
            f[i][3] = (float)(int8_t)(ww >> 24);
        }
        float p[R][T];
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            #pragma unroll
            for (int i = 0; i < R; ++i) p[i][t] = __fmul_rn(f[i][1], xv[t][xs].y);
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            #pragma unroll
            for (int i = 0; i < R; ++i) p[i][t] = __fmaf_rn(f[i][0], xv[t][xs].x, p[i][t]);
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            #pragma unroll
            for (int i = 0; i < R; ++i) p[i][t] = __fmaf_rn(f[i][2], xv[t][xs].z, p[i][t]);
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            #pragma unroll
            for (int i = 0; i < R; ++i) p[i][t] = __fmaf_rn(f[i][3], xv[t][xs].w, p[i][t]);
        }
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            #pragma unroll
            for (int i = 0; i < R; ++i) acc[i][t] = __fmaf_rn(dv[i], p[i][t], acc[i][t]);
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                acc[i][t] = __fadd_rn(acc[i][t], __shfl_xor_sync(0xffffffffu, acc[i][t], o));
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            if (row0 + i < N) {
                #pragma unroll
                for (int t = 0; t < T; ++t) {
                    y[(size_t)t * y_ld + row0 + i] = acc[i][t];
                }
            }
        }
    }
}

template <int WARPS, int R, int T, int NIT, int DW>
__global__ void __launch_bounds__(WARPS * 32)
mv_q8p_colsx_kernel(
    const int8_t* __restrict__ q,
    const uint16_t* __restrict__ d,
    int N,
    int K,
    const float* __restrict__ x,
    int x_ld,
    float* __restrict__ y,
    int y_ld
) {
    const int row0 = ((int)blockIdx.x * WARPS + (threadIdx.x >> 5)) * R;
    if (row0 >= N) return;
    mv_q8p_colsx_body<WARPS, R, T, NIT, DW>(q, d, N, K, x, x_ld, y, y_ld, row0);
}

template <int WARPS, int R, int T, int NIT, int DW>
__global__ void __launch_bounds__(WARPS * 32)
mv_q8p_colsx_qkvrg_kernel(Q8QkvrgF32Segs s, int K) {
    const int gt = (int)blockIdx.x;
    int seg = 4;
    if (gt < s.cta_off[1]) seg = 0;
    else if (gt < s.cta_off[2]) seg = 1;
    else if (gt < s.cta_off[3]) seg = 2;
    else if (gt < s.cta_off[4]) seg = 3;

    const int n_seg = s.n[seg];
    const int local_cta = gt - s.cta_off[seg];
    const int row0 = (local_cta * WARPS + (threadIdx.x >> 5)) * R;
    if (row0 >= n_seg) return;
    mv_q8p_colsx_body<WARPS, R, T, NIT, DW>(
        s.q[seg], s.d[seg], n_seg, K, s.x[seg], K, s.y[seg], n_seg, row0
    );
}

template <int WARPS, int R, int T, int NIT, int DW>
__global__ void __launch_bounds__(WARPS * 32)
mv_q8p_colsx_argmax_part_kernel(
    const int8_t* __restrict__ q,
    const uint16_t* __restrict__ d,
    int N,
    int K,
    const float* __restrict__ x,
    int x_ld,
    float* __restrict__ part_vals,
    int32_t* __restrict__ part_ids
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_global = (int)blockIdx.x * WARPS + warp;
    const int total_warps = (int)gridDim.x * WARPS;
    const uint32_t* x4_base = reinterpret_cast<const uint32_t*>(x);
    (void)x4_base;

    float best_v[T];
    int32_t best_i[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        best_v[t] = -INFINITY;
        best_i[t] = 0x7fffffff;
    }

    for (int row0 = warp_global * R; row0 < N; row0 += total_warps * R) {
        const uint32_t* qr[R];
        const uint16_t* dr[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            const int row = min(row0 + i, N - 1);
            qr[i] = reinterpret_cast<const uint32_t*>(q + (size_t)row * K);
            dr[i] = d + (size_t)row * (K >> 5);
        }

        uint32_t w[R][DW + 1];
        uint16_t dh[R][DW + 1];
        float4 xv[T][2];
        float acc[R][T];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            #pragma unroll
            for (int t = 0; t < T; ++t) acc[i][t] = 0.0f;
        }

        #pragma unroll
        for (int a = 0; a < DW && a < NIT; ++a) {
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                w[i][a] = __ldg(qr[i] + a * 32 + lane);
                dh[i][a] = __ldg(dr[i] + a * 4 + (lane >> 3));
            }
        }
        if constexpr (NIT > 0) {
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                xv[t][0] = __ldg(reinterpret_cast<const float4*>(x + (size_t)t * x_ld) + lane);
            }
        }

        #pragma unroll
        for (int it = 0; it < NIT; ++it) {
            const int ws = it % (DW + 1);
            const int xs = it & 1;
            if (it + DW < NIT) {
                const int nws = (it + DW) % (DW + 1);
                #pragma unroll
                for (int i = 0; i < R; ++i) {
                    w[i][nws] = __ldg(qr[i] + (it + DW) * 32 + lane);
                    dh[i][nws] = __ldg(dr[i] + (it + DW) * 4 + (lane >> 3));
                }
            }
            if (it + 1 < NIT) {
                const int nxs = (it + 1) & 1;
                #pragma unroll
                for (int t = 0; t < T; ++t) {
                    xv[t][nxs] = __ldg(reinterpret_cast<const float4*>(x + (size_t)t * x_ld) + (it + 1) * 32 + lane);
                }
            }
            float f[R][4], dv[R];
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                const uint32_t ww = w[i][ws];
                dv[i] = fp16_bits_to_f32(dh[i][ws]);
                f[i][0] = (float)(int8_t)(ww & 0xFFu);
                f[i][1] = (float)(int8_t)((ww >> 8) & 0xFFu);
                f[i][2] = (float)(int8_t)((ww >> 16) & 0xFFu);
                f[i][3] = (float)(int8_t)(ww >> 24);
            }
            float p[R][T];
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                #pragma unroll
                for (int i = 0; i < R; ++i) p[i][t] = __fmul_rn(f[i][1], xv[t][xs].y);
            }
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                #pragma unroll
                for (int i = 0; i < R; ++i) p[i][t] = __fmaf_rn(f[i][0], xv[t][xs].x, p[i][t]);
            }
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                #pragma unroll
                for (int i = 0; i < R; ++i) p[i][t] = __fmaf_rn(f[i][2], xv[t][xs].z, p[i][t]);
            }
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                #pragma unroll
                for (int i = 0; i < R; ++i) p[i][t] = __fmaf_rn(f[i][3], xv[t][xs].w, p[i][t]);
            }
            #pragma unroll
            for (int t = 0; t < T; ++t) {
                #pragma unroll
                for (int i = 0; i < R; ++i) acc[i][t] = __fmaf_rn(dv[i], p[i][t], acc[i][t]);
            }
        }

        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                #pragma unroll
                for (int t = 0; t < T; ++t) {
                    acc[i][t] = __fadd_rn(acc[i][t], __shfl_xor_sync(0xffffffffu, acc[i][t], o));
                }
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                const int r = row0 + i;
                if (r < N) {
                    #pragma unroll
                    for (int t = 0; t < T; ++t) {
                        const float v = acc[i][t];
                        if (v > best_v[t] || (v == best_v[t] && r < best_i[t])) {
                            best_v[t] = v;
                            best_i[t] = r;
                        }
                    }
                }
            }
        }
    }

    __shared__ float sh_v[WARPS][T];
    __shared__ int32_t sh_i[WARPS][T];
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            sh_v[warp][t] = best_v[t];
            sh_i[warp][t] = best_i[t];
        }
    }
    __syncthreads();
    if (threadIdx.x < T) {
        const int tk = threadIdx.x;
        float bv = sh_v[0][tk];
        int32_t bi = sh_i[0][tk];
        #pragma unroll
        for (int w_i = 1; w_i < WARPS; ++w_i) {
            const float v = sh_v[w_i][tk];
            const int32_t idx = sh_i[w_i][tk];
            if (v > bv || (v == bv && idx < bi)) {
                bv = v;
                bi = idx;
            }
        }
        part_vals[(size_t)tk * gridDim.x + blockIdx.x] = bv;
        part_ids[(size_t)tk * gridDim.x + blockIdx.x] = bi;
    }
}

// Generic fallbacks for non-2304/3072 K (K % 128 == 0)
__global__ void __launch_bounds__(128) mv_q8p_cols_generic_kernel(
    const int8_t* __restrict__ q,
    const uint16_t* __restrict__ d,
    int N,
    int K,
    const float* __restrict__ x,
    int x_ld,
    int T,
    float* __restrict__ y,
    int y_ld
) {
    const int lane = threadIdx.x & 31;
    const int row0 = ((int)blockIdx.x * 4 + (threadIdx.x >> 5)) * 2;
    if (row0 >= N) return;
    const int nit = K >> 7;
    const uint32_t* qr0 = reinterpret_cast<const uint32_t*>(q + (size_t)row0 * K);
    const uint16_t* dr0 = d + (size_t)row0 * (K >> 5);
    const int row1 = min(row0 + 1, N - 1);
    const uint32_t* qr1 = reinterpret_cast<const uint32_t*>(q + (size_t)row1 * K);
    const uint16_t* dr1 = d + (size_t)row1 * (K >> 5);
    float acc[2][8] = {};
    for (int it = 0; it < nit; ++it) {
        const uint32_t w0 = __ldg(qr0 + it * 32 + lane);
        const uint32_t w1 = __ldg(qr1 + it * 32 + lane);
        const float d0 = fp16_bits_to_f32(__ldg(dr0 + it * 4 + (lane >> 3)));
        const float d1 = fp16_bits_to_f32(__ldg(dr1 + it * 4 + (lane >> 3)));
        const float f00 = (float)(int8_t)(w0 & 0xFFu), f01 = (float)(int8_t)((w0 >> 8) & 0xFFu),
                    f02 = (float)(int8_t)((w0 >> 16) & 0xFFu), f03 = (float)(int8_t)(w0 >> 24);
        const float f10 = (float)(int8_t)(w1 & 0xFFu), f11 = (float)(int8_t)((w1 >> 8) & 0xFFu),
                    f12 = (float)(int8_t)((w1 >> 16) & 0xFFu), f13 = (float)(int8_t)(w1 >> 24);
        for (int t = 0; t < T; ++t) {
            const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (size_t)t * x_ld) + it * 32 + lane);
            float p0 = __fmul_rn(f01, xv.y);
            p0 = __fmaf_rn(f00, xv.x, p0); p0 = __fmaf_rn(f02, xv.z, p0); p0 = __fmaf_rn(f03, xv.w, p0);
            acc[0][t] = __fmaf_rn(d0, p0, acc[0][t]);
            float p1 = __fmul_rn(f11, xv.y);
            p1 = __fmaf_rn(f10, xv.x, p1); p1 = __fmaf_rn(f12, xv.z, p1); p1 = __fmaf_rn(f13, xv.w, p1);
            acc[1][t] = __fmaf_rn(d1, p1, acc[1][t]);
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        for (int i = 0; i < 2; ++i)
            for (int t = 0; t < T; ++t)
                acc[i][t] = __fadd_rn(acc[i][t], __shfl_xor_sync(0xffffffffu, acc[i][t], o));
    }
    if (lane == 0) {
        for (int i = 0; i < 2; ++i)
            if (row0 + i < N)
                for (int t = 0; t < T; ++t) y[(size_t)t * y_ld + row0 + i] = acc[i][t];
    }
}

__global__ void __launch_bounds__(128) mv_q8p_cols_qkvrg_generic_kernel(Q8QkvrgF32Segs s, int K, int T) {
    const int gt = (int)blockIdx.x;
    int seg = 4;
    if (gt < s.cta_off[1]) seg = 0;
    else if (gt < s.cta_off[2]) seg = 1;
    else if (gt < s.cta_off[3]) seg = 2;
    else if (gt < s.cta_off[4]) seg = 3;
    const int n_seg = s.n[seg];
    const int row0 = ((gt - s.cta_off[seg]) * 4 + (threadIdx.x >> 5)) * 2;
    if (row0 >= n_seg) return;
    const int lane = threadIdx.x & 31;
    const int nit = K >> 7;
    const uint32_t* qr0 = reinterpret_cast<const uint32_t*>(s.q[seg] + (size_t)row0 * K);
    const uint16_t* dr0 = s.d[seg] + (size_t)row0 * (K >> 5);
    const int row1 = min(row0 + 1, n_seg - 1);
    const uint32_t* qr1 = reinterpret_cast<const uint32_t*>(s.q[seg] + (size_t)row1 * K);
    const uint16_t* dr1 = s.d[seg] + (size_t)row1 * (K >> 5);
    float acc[2][8] = {};
    for (int it = 0; it < nit; ++it) {
        const uint32_t w0 = __ldg(qr0 + it * 32 + lane);
        const uint32_t w1 = __ldg(qr1 + it * 32 + lane);
        const float d0 = fp16_bits_to_f32(__ldg(dr0 + it * 4 + (lane >> 3)));
        const float d1 = fp16_bits_to_f32(__ldg(dr1 + it * 4 + (lane >> 3)));
        const float f00 = (float)(int8_t)(w0 & 0xFFu), f01 = (float)(int8_t)((w0 >> 8) & 0xFFu),
                    f02 = (float)(int8_t)((w0 >> 16) & 0xFFu), f03 = (float)(int8_t)(w0 >> 24);
        const float f10 = (float)(int8_t)(w1 & 0xFFu), f11 = (float)(int8_t)((w1 >> 8) & 0xFFu),
                    f12 = (float)(int8_t)((w1 >> 16) & 0xFFu), f13 = (float)(int8_t)(w1 >> 24);
        for (int t = 0; t < T; ++t) {
            const float4 xv = __ldg(reinterpret_cast<const float4*>(s.x[seg] + (size_t)t * K) + it * 32 + lane);
            float p0 = __fmul_rn(f01, xv.y);
            p0 = __fmaf_rn(f00, xv.x, p0); p0 = __fmaf_rn(f02, xv.z, p0); p0 = __fmaf_rn(f03, xv.w, p0);
            acc[0][t] = __fmaf_rn(d0, p0, acc[0][t]);
            float p1 = __fmul_rn(f11, xv.y);
            p1 = __fmaf_rn(f10, xv.x, p1); p1 = __fmaf_rn(f12, xv.z, p1); p1 = __fmaf_rn(f13, xv.w, p1);
            acc[1][t] = __fmaf_rn(d1, p1, acc[1][t]);
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        for (int i = 0; i < 2; ++i)
            for (int t = 0; t < T; ++t)
                acc[i][t] = __fadd_rn(acc[i][t], __shfl_xor_sync(0xffffffffu, acc[i][t], o));
    }
    if (lane == 0) {
        for (int i = 0; i < 2; ++i)
            if (row0 + i < n_seg)
                for (int t = 0; t < T; ++t) s.y[seg][(size_t)t * n_seg + row0 + i] = acc[i][t];
    }
}

__global__ void __launch_bounds__(128) mv_q8p_cols_argmax_generic_kernel(
    const int8_t* __restrict__ q,
    const uint16_t* __restrict__ d,
    int N,
    int K,
    const float* __restrict__ x,
    int x_ld,
    int T,
    float* __restrict__ part_vals,
    int32_t* __restrict__ part_ids
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warp_global = (int)blockIdx.x * 4 + warp;
    const int total_warps = (int)gridDim.x * 4;
    const int nit = K >> 7;
    float best_v[8];
    int32_t best_i[8];
    for (int t = 0; t < 8; ++t) { best_v[t] = -INFINITY; best_i[t] = 0x7fffffff; }

    for (int row0 = warp_global * 2; row0 < N; row0 += total_warps * 2) {
        const uint32_t* qr0 = reinterpret_cast<const uint32_t*>(q + (size_t)row0 * K);
        const uint16_t* dr0 = d + (size_t)row0 * (K >> 5);
        const int row1 = min(row0 + 1, N - 1);
        const uint32_t* qr1 = reinterpret_cast<const uint32_t*>(q + (size_t)row1 * K);
        const uint16_t* dr1 = d + (size_t)row1 * (K >> 5);
        float acc[2][8] = {};
        for (int it = 0; it < nit; ++it) {
            const uint32_t w0 = __ldg(qr0 + it * 32 + lane);
            const uint32_t w1 = __ldg(qr1 + it * 32 + lane);
            const float d0 = fp16_bits_to_f32(__ldg(dr0 + it * 4 + (lane >> 3)));
            const float d1 = fp16_bits_to_f32(__ldg(dr1 + it * 4 + (lane >> 3)));
            const float f00 = (float)(int8_t)(w0 & 0xFFu), f01 = (float)(int8_t)((w0 >> 8) & 0xFFu),
                        f02 = (float)(int8_t)((w0 >> 16) & 0xFFu), f03 = (float)(int8_t)(w0 >> 24);
            const float f10 = (float)(int8_t)(w1 & 0xFFu), f11 = (float)(int8_t)((w1 >> 8) & 0xFFu),
                        f12 = (float)(int8_t)((w1 >> 16) & 0xFFu), f13 = (float)(int8_t)(w1 >> 24);
            for (int t = 0; t < T; ++t) {
                const float4 xv = __ldg(reinterpret_cast<const float4*>(x + (size_t)t * x_ld) + it * 32 + lane);
                float p0 = __fmul_rn(f01, xv.y);
                p0 = __fmaf_rn(f00, xv.x, p0); p0 = __fmaf_rn(f02, xv.z, p0); p0 = __fmaf_rn(f03, xv.w, p0);
                acc[0][t] = __fmaf_rn(d0, p0, acc[0][t]);
                float p1 = __fmul_rn(f11, xv.y);
                p1 = __fmaf_rn(f10, xv.x, p1); p1 = __fmaf_rn(f12, xv.z, p1); p1 = __fmaf_rn(f13, xv.w, p1);
                acc[1][t] = __fmaf_rn(d1, p1, acc[1][t]);
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            for (int i = 0; i < 2; ++i)
                for (int t = 0; t < T; ++t)
                    acc[i][t] = __fadd_rn(acc[i][t], __shfl_xor_sync(0xffffffffu, acc[i][t], o));
        }
        if (lane == 0) {
            for (int i = 0; i < 2; ++i) {
                const int r = row0 + i;
                if (r < N) {
                    for (int t = 0; t < T; ++t) {
                        const float v = acc[i][t];
                        if (v > best_v[t] || (v == best_v[t] && r < best_i[t])) {
                            best_v[t] = v;
                            best_i[t] = r;
                        }
                    }
                }
            }
        }
    }
    __shared__ float sh_v[4][8];
    __shared__ int32_t sh_i[4][8];
    if (lane == 0) {
        for (int t = 0; t < T; ++t) { sh_v[warp][t] = best_v[t]; sh_i[warp][t] = best_i[t]; }
    }
    __syncthreads();
    if (threadIdx.x < T) {
        const int tk = threadIdx.x;
        float bv = sh_v[0][tk]; int32_t bi = sh_i[0][tk];
        for (int w_i = 1; w_i < 4; ++w_i) {
            const float v = sh_v[w_i][tk]; const int32_t idx = sh_i[w_i][tk];
            if (v > bv || (v == bv && idx < bi)) { bv = v; bi = idx; }
        }
        part_vals[(size_t)tk * gridDim.x + blockIdx.x] = bv;
        part_ids[(size_t)tk * gridDim.x + blockIdx.x] = bi;
    }
}

template <int T_FIXED>
__global__ void __launch_bounds__(32) mv_f16_batch_t_kernel(
    const uint16_t* __restrict__ w, // [N, K]
    const float* __restrict__ x,    // [T, K]
    float* __restrict__ out,        // [T, N]
    int N,
    int K
) {
    const int lane = threadIdx.x & 31;
    const int row = (int)blockIdx.x;
    if (row >= N) return;

    const uint2* wr4 = reinterpret_cast<const uint2*>(w + (size_t)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int k4_end = K >> 2;
    const int x_stride4 = K >> 2;

    float acc[T_FIXED] = {};
    for (int k4 = lane; k4 < k4_end; k4 += 32) {
        const uint2 wv = __ldg(wr4 + k4);
        const float w0 = fp16_bits_to_f32(wv.x & 0xffffu);
        const float w1 = fp16_bits_to_f32(wv.x >> 16);
        const float w2 = fp16_bits_to_f32(wv.y & 0xffffu);
        const float w3 = fp16_bits_to_f32(wv.y >> 16);
        #pragma unroll
        for (int t = 0; t < T_FIXED; ++t) {
            const float4 xv = __ldg(x4 + (size_t)t * x_stride4 + k4);
            acc[t] = fmaf(w0, xv.x, acc[t]);
            acc[t] = fmaf(w1, xv.y, acc[t]);
            acc[t] = fmaf(w2, xv.z, acc[t]);
            acc[t] = fmaf(w3, xv.w, acc[t]);
        }
    }

    #pragma unroll
    for (int t = 0; t < T_FIXED; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) out[(size_t)t * N + row] = sum;
    }
}

__global__ void __launch_bounds__(32) mv_f16_batch_kernel(
    const uint16_t* __restrict__ w, // [N, K]
    const float* __restrict__ x,    // [T, K]
    float* __restrict__ out,        // [T, N]
    int N,
    int K,
    int T
) {
    const int lane = threadIdx.x & 31;
    const int row = (int)blockIdx.x;
    if (row >= N) return;

    const uint2* wr4 = reinterpret_cast<const uint2*>(w + (size_t)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int k4_end = K >> 2;
    const int x_stride4 = K >> 2;

    float acc[8] = {};
    for (int k4 = lane; k4 < k4_end; k4 += 32) {
        const uint2 wv = __ldg(wr4 + k4);
        const float w0 = fp16_bits_to_f32(wv.x & 0xffffu);
        const float w1 = fp16_bits_to_f32(wv.x >> 16);
        const float w2 = fp16_bits_to_f32(wv.y & 0xffffu);
        const float w3 = fp16_bits_to_f32(wv.y >> 16);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < T) {
                const float4 xv = __ldg(x4 + (size_t)t * x_stride4 + k4);
                acc[t] = fmaf(w0, xv.x, acc[t]);
                acc[t] = fmaf(w1, xv.y, acc[t]);
                acc[t] = fmaf(w2, xv.z, acc[t]);
                acc[t] = fmaf(w3, xv.w, acc[t]);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < T) {
            const float sum = warp_reduce_sum(acc[t]);
            if (lane == 0) out[(size_t)t * N + row] = sum;
        }
    }
}

template <int T_FIXED>
__global__ void __launch_bounds__(32) mv_f16_pair_batch_t_kernel(
    const uint16_t* __restrict__ w0, const float* __restrict__ x0, float* __restrict__ out0,
    const uint16_t* __restrict__ w1, const float* __restrict__ x1, float* __restrict__ out1,
    int N,
    int K
) {
    const bool is1 = (blockIdx.y != 0);
    const uint16_t* w = is1 ? w1 : w0;
    const float* x = is1 ? x1 : x0;
    float* out = is1 ? out1 : out0;

    const int lane = threadIdx.x & 31;
    const int row = (int)blockIdx.x;
    if (row >= N) return;

    const uint2* wr4 = reinterpret_cast<const uint2*>(w + (size_t)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int k4_end = K >> 2;
    const int x_stride4 = K >> 2;

    float acc[T_FIXED] = {};
    for (int k4 = lane; k4 < k4_end; k4 += 32) {
        const uint2 wv = __ldg(wr4 + k4);
        const float f0 = fp16_bits_to_f32(wv.x & 0xffffu);
        const float f1 = fp16_bits_to_f32(wv.x >> 16);
        const float f2 = fp16_bits_to_f32(wv.y & 0xffffu);
        const float f3 = fp16_bits_to_f32(wv.y >> 16);
        #pragma unroll
        for (int t = 0; t < T_FIXED; ++t) {
            const float4 xv = __ldg(x4 + (size_t)t * x_stride4 + k4);
            acc[t] = fmaf(f0, xv.x, acc[t]);
            acc[t] = fmaf(f1, xv.y, acc[t]);
            acc[t] = fmaf(f2, xv.z, acc[t]);
            acc[t] = fmaf(f3, xv.w, acc[t]);
        }
    }

    #pragma unroll
    for (int t = 0; t < T_FIXED; ++t) {
        const float sum = warp_reduce_sum(acc[t]);
        if (lane == 0) out[(size_t)t * N + row] = sum;
    }
}

__global__ void __launch_bounds__(32) mv_f16_pair_batch_kernel(
    const uint16_t* __restrict__ w0, const float* __restrict__ x0, float* __restrict__ out0,
    const uint16_t* __restrict__ w1, const float* __restrict__ x1, float* __restrict__ out1,
    int N,
    int K,
    int T
) {
    const bool is1 = (blockIdx.y != 0);
    const uint16_t* w = is1 ? w1 : w0;
    const float* x = is1 ? x1 : x0;
    float* out = is1 ? out1 : out0;

    const int lane = threadIdx.x & 31;
    const int row = (int)blockIdx.x;
    if (row >= N) return;

    const uint2* wr4 = reinterpret_cast<const uint2*>(w + (size_t)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int k4_end = K >> 2;
    const int x_stride4 = K >> 2;

    float acc[8] = {};
    for (int k4 = lane; k4 < k4_end; k4 += 32) {
        const uint2 wv = __ldg(wr4 + k4);
        const float f0 = fp16_bits_to_f32(wv.x & 0xffffu);
        const float f1 = fp16_bits_to_f32(wv.x >> 16);
        const float f2 = fp16_bits_to_f32(wv.y & 0xffffu);
        const float f3 = fp16_bits_to_f32(wv.y >> 16);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < T) {
                const float4 xv = __ldg(x4 + (size_t)t * x_stride4 + k4);
                acc[t] = fmaf(f0, xv.x, acc[t]);
                acc[t] = fmaf(f1, xv.y, acc[t]);
                acc[t] = fmaf(f2, xv.z, acc[t]);
                acc[t] = fmaf(f3, xv.w, acc[t]);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < T) {
            const float sum = warp_reduce_sum(acc[t]);
            if (lane == 0) out[(size_t)t * N + row] = sum;
        }
    }
}

__global__ void conv_rollback_all_kernel(
    float* const* __restrict__ histories,
    const float* __restrict__ rec_old_hist,
    const float* __restrict__ rec_x,
    int n_slots,
    int n_sconv_channels,
    int accept_n
) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const int slot = blockIdx.y;
    if (slot >= n_slots || ch >= n_sconv_channels) return;

    const float* old_p = rec_old_hist + ((size_t)slot * n_sconv_channels + ch) * 3;
    float h0 = old_p[0], h1 = old_p[1], h2 = old_p[2];
    for (int t = 0; t < accept_n; ++t) {
        const float xt = rec_x[((size_t)slot * 8 + t) * n_sconv_channels + ch];
        h0 = h1;
        h1 = h2;
        h2 = xt;
    }
    float* dst = histories[slot] + (size_t)ch * 3;
    dst[0] = h0;
    dst[1] = h1;
    dst[2] = h2;
}

// -----------------------------------------------------------------------------
// 4. Multi-Token Loop Gate Dots + Prep QKV/R + GQA-Grouped MMA Attention (FP8 & F16)
// -----------------------------------------------------------------------------

struct FastLoopGateArgs {
    const float* x[3];
    const float* m[3];
    const float* D[3];
    const float* F[3];
    const float* G[3];
    float* y[3];
};

template <int NIT>
__global__ void __launch_bounds__(128) loop_gate_dots_toks_fast_kernel(
    FastLoopGateArgs a, int T, int C, int R, float* __restrict__ sc
) {
    __shared__ float sh[8][4][4];
    const int r = blockIdx.x;
    const int j = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const float* dr = a.D[j] + (size_t)r * 2 * C;

    float dx[NIT], dm[NIT];
    #pragma unroll
    for (int i = 0; i < NIT; ++i) {
        dx[i] = __ldg(dr + tid + i * 128);
        dm[i] = __ldg(dr + C + tid + i * 128);
    }

    for (int t = 0; t < T; ++t) {
        const float* xr = a.x[j] + (size_t)t * C;
        const float* mr = a.m[j] + (size_t)t * C;
        float ax = 0.0f, am = 0.0f, sx = 0.0f, sm = 0.0f;
        #pragma unroll
        for (int i = 0; i < NIT; ++i) {
            const float xv = __ldg(xr + tid + i * 128);
            const float mv = __ldg(mr + tid + i * 128);
            ax = __fmaf_rn(dx[i], xv, ax);
            am = __fmaf_rn(dm[i], mv, am);
            if (r == 0) {
                sx = __fmaf_rn(xv, xv, sx);
                sm = __fmaf_rn(mv, mv, sm);
            }
        }
        ax = warp_reduce_sum(ax);
        am = warp_reduce_sum(am);
        if (r == 0) {
            sx = warp_reduce_sum(sx);
            sm = warp_reduce_sum(sm);
        }
        if ((tid & 31) == 0) {
            sh[t][warp][0] = ax;
            sh[t][warp][1] = am;
            sh[t][warp][2] = sx;
            sh[t][warp][3] = sm;
        }
    }
    __syncthreads();
    if (tid < T) {
        const int t = tid;
        float* o = sc + ((size_t)j * T + t) * (2 * R + 2);
        o[2 * r] = sh[t][0][0] + sh[t][1][0] + sh[t][2][0] + sh[t][3][0];
        o[2 * r + 1] = sh[t][0][1] + sh[t][1][1] + sh[t][2][1] + sh[t][3][1];
        if (r == 0) {
            o[2 * R] = sh[t][0][2] + sh[t][1][2] + sh[t][2][2] + sh[t][3][2];
            o[2 * R + 1] = sh[t][0][3] + sh[t][1][3] + sh[t][2][3] + sh[t][3][3];
        }
    }
}

__global__ void __launch_bounds__(128) loop_gate_dots_toks_kernel(
    FastLoopGateArgs a, int T, int C, int R, float* __restrict__ sc
) {
    __shared__ float sh[4][4];
    const int r = blockIdx.x;
    const int j = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const float* dr = a.D[j] + (size_t)r * 2 * C;
    for (int t = 0; t < T; ++t) {
        const float* xr = a.x[j] + (size_t)t * C;
        const float* mr = a.m[j] + (size_t)t * C;
        float ax = 0.0f, am = 0.0f, sx = 0.0f, sm = 0.0f;
        for (int c = tid; c < C; c += 128) {
            const float xv = xr[c], mv = mr[c];
            const float dx = __ldg(dr + c), dm = __ldg(dr + C + c);
            ax += dx * xv;
            am += dm * mv;
            if (r == 0) {
                sx += xv * xv;
                sm += mv * mv;
            }
        }
        ax = warp_reduce_sum(ax);
        am = warp_reduce_sum(am);
        if (r == 0) {
            sx = warp_reduce_sum(sx);
            sm = warp_reduce_sum(sm);
        }
        if ((tid & 31) == 0) {
            sh[warp][0] = ax;
            sh[warp][1] = am;
            sh[warp][2] = sx;
            sh[warp][3] = sm;
        }
        __syncthreads();
        if (tid == 0) {
            float* o = sc + ((size_t)j * T + t) * (2 * R + 2);
            o[2 * r] = sh[0][0] + sh[1][0] + sh[2][0] + sh[3][0];
            o[2 * r + 1] = sh[0][1] + sh[1][1] + sh[2][1] + sh[3][1];
            if (r == 0) {
                o[2 * R] = sh[0][2] + sh[1][2] + sh[2][2] + sh[3][2];
                o[2 * R + 1] = sh[0][3] + sh[1][3] + sh[2][3] + sh[3][3];
            }
        }
        __syncthreads();
    }
}

static constexpr int kFastGateCh = 32;
static constexpr int kFastGatePad = kFastGateCh + 1;
static constexpr int kFastGateWarps = 8;

__global__ void __launch_bounds__(kFastGateWarps * 32) loop_gate_apply_fast_kernel(
    FastLoopGateArgs a,
    int M,
    int C,
    int R,
    float eps,
    const float* __restrict__ sc
) {
    extern __shared__ float sh_lg[];
    float* sF = sh_lg;
    float* sG = sF + (size_t)R * kFastGatePad;
    float* sg = sG + (size_t)R * kFastGatePad;

    const int j = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c0 = (int)blockIdx.x * kFastGateCh;
    const int c = c0 + lane;
    const int t0 = (int)blockIdx.z * kFastGateWarps;
    const int nt = min(kFastGateWarps, M - t0);

    if ((R & 3) == 0 && (c0 + kFastGateCh <= C) &&
        (((uintptr_t)a.F[j] & 15) == 0) && (((uintptr_t)a.G[j] & 15) == 0)) {
        const float4* F4 = reinterpret_cast<const float4*>(a.F[j] + (size_t)c0 * R);
        const float4* G4 = reinterpret_cast<const float4*>(a.G[j] + (size_t)c0 * R);
        const int n4 = (kFastGateCh * R) >> 2;
        for (int i4 = tid; i4 < n4; i4 += kFastGateWarps * 32) {
            const int elem = i4 << 2;
            const int cc = elem / R;
            const int r0 = elem - cc * R;
            const float4 fv = __ldg(F4 + i4);
            const float4 gv = __ldg(G4 + i4);
            sF[(size_t)(r0 + 0) * kFastGatePad + cc] = fv.x;
            sF[(size_t)(r0 + 1) * kFastGatePad + cc] = fv.y;
            sF[(size_t)(r0 + 2) * kFastGatePad + cc] = fv.z;
            sF[(size_t)(r0 + 3) * kFastGatePad + cc] = fv.w;
            sG[(size_t)(r0 + 0) * kFastGatePad + cc] = gv.x;
            sG[(size_t)(r0 + 1) * kFastGatePad + cc] = gv.y;
            sG[(size_t)(r0 + 2) * kFastGatePad + cc] = gv.z;
            sG[(size_t)(r0 + 3) * kFastGatePad + cc] = gv.w;
        }
    } else {
        const float* F_ch = (c < C) ? (a.F[j] + (size_t)c * R) : nullptr;
        const float* G_ch = (c < C) ? (a.G[j] + (size_t)c * R) : nullptr;
        for (int r = warp; r < R; r += kFastGateWarps) {
            const bool in = (c < C);
            sF[(size_t)r * kFastGatePad + lane] = in ? __ldg(F_ch + r) : 0.0f;
            sG[(size_t)r * kFastGatePad + lane] = in ? __ldg(G_ch + r) : 0.0f;
        }
    }
    for (int i = tid; i < nt * R; i += kFastGateWarps * 32) {
        const int tt = i / R;
        const int r = i - tt * R;
        const float* o = sc + ((size_t)j * M + t0 + tt) * (2 * R + 2);
        const float rx = rsqrtf(o[2 * R] / (float)C + eps);
        const float rm = rsqrtf(o[2 * R + 1] / (float)C + eps);
        sg[tt * R + r] = rx * o[2 * r] + rm * o[2 * r + 1];
    }
    __syncthreads();
    if (warp >= nt || c >= C) return;

    const float* g = sg + warp * R;
    float f = 0.0f, gn = 0.0f;
    for (int r = 0; r < R; ++r) {
        const float gv = g[r];
        f  += sF[(size_t)r * kFastGatePad + lane] * gv;
        gn += sG[(size_t)r * kFastGatePad + lane] * gv;
    }
    const size_t idx = (size_t)(t0 + warp) * C + c;
    const float xv = a.x[j][idx];
    const float mv = a.m[j][idx];
    a.y[j][idx] = (1.0f - tanhf(f)) * xv + tanhf(gn) * mv;
}

__global__ void prep_qkv_r_block_kernel(
    float* __restrict__ q_cur,
    uint16_t* __restrict__ q16,
    const float* __restrict__ q_clip,
    float* __restrict__ k_cur,
    void* __restrict__ k_cache,
    float* __restrict__ v_cur,
    void* __restrict__ v_cache,
    float* __restrict__ r_cur,
    const float* __restrict__ r_clip,
    const float* __restrict__ tau,
    const int32_t* __restrict__ pos,
    const float* __restrict__ sink_w,
    const uint16_t* __restrict__ sink_w16,
    const float* __restrict__ sink_x,
    float* __restrict__ sinks_out,
    int n_embd,
    int n_head,
    int n_head_kv,
    int head_dim,
    int d_rel,
    int M,
    int ring_cap,
    int kv_mode, // 0=F16, 1=Q8_0, 2=FP8
    float qk_clip_th,
    float rel_clip_th,
    float eps
) {
    const int gid = blockIdx.x;
    const int lane = threadIdx.x;
    const int per_tok = 2 * n_head + 2 * n_head_kv;

    if (gid >= M * per_tok) {
        const int i = gid - M * per_tok;
        const int m = i / n_head;
        const int h = i % n_head;
        if ((!sink_w && !sink_w16) || m >= M) return;
        const float* x_row = sink_x + (size_t)m * n_embd;
        float acc = 0.0f;
        if (sink_w16) {
            const uint16_t* w_row = sink_w16 + (size_t)h * n_embd;
            if ((n_embd & 3) == 0) {
                const uint2* w4 = reinterpret_cast<const uint2*>(w_row);
                const float4* x4 = reinterpret_cast<const float4*>(x_row);
                const int k4_end = n_embd >> 2;
                for (int k4 = lane; k4 < k4_end; k4 += 32) {
                    const uint2 wv = __ldg(w4 + k4);
                    const float4 xv = __ldg(x4 + k4);
                    acc = fmaf(fp16_bits_to_f32(wv.x & 0xffffu), xv.x, acc);
                    acc = fmaf(fp16_bits_to_f32(wv.x >> 16), xv.y, acc);
                    acc = fmaf(fp16_bits_to_f32(wv.y & 0xffffu), xv.z, acc);
                    acc = fmaf(fp16_bits_to_f32(wv.y >> 16), xv.w, acc);
                }
            } else {
                for (int k = lane; k < n_embd; k += 32) {
                    acc = fmaf(fp16_bits_to_f32(__ldg(w_row + k)), x_row[k], acc);
                }
            }
        } else {
            const float* w_row = sink_w + (size_t)h * n_embd;
            for (int k = lane; k < n_embd; k += 32) {
                acc += __ldg(w_row + k) * x_row[k];
            }
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0) sinks_out[(size_t)m * n_head + h] = acc;
        return;
    }

    const int m = gid / per_tok;
    const int rem = gid % per_tok;
    const float t_mul = tau ? tau[m] : 1.0f;

    if (rem < n_head) {
        const int h = rem;
        float* qh = q_cur + ((size_t)m * n_head + h) * head_dim;
        uint16_t* q16h = q16 ? (q16 + ((size_t)m * n_head + h) * head_dim) : nullptr;
        const float* sh = q_clip + (size_t)h * head_dim;
        float sum2 = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            const float v = qh[d] * sh[d];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        const float factor = fminf(1.0f, qk_clip_th / (sqrtf(sum2) + 1e-6f)) * t_mul;
        for (int d = lane; d < head_dim; d += 32) {
            const float outv = (qh[d] * sh[d]) * factor;
            qh[d] = outv;
            if (q16h) q16h[d] = f32_to_fp16_bits(outv);
        }
    } else if (rem < n_head + n_head_kv) {
        const int kh = rem - n_head;
        const int slot = pos[m] % ring_cap;
        float* kh_ptr = k_cur + ((size_t)m * n_head_kv + kh) * head_dim;
        float sum2 = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            const float v = kh_ptr[d];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        const float factor = fminf(1.0f, qk_clip_th / (sqrtf(sum2) + 1e-6f));
        if (kv_mode == 0) {
            uint16_t* k_dst = reinterpret_cast<uint16_t*>(k_cache) + ((size_t)slot * n_head_kv + kh) * head_dim;
            for (int d = lane; d < head_dim; d += 32) {
                const float v = kh_ptr[d] * factor;
                kh_ptr[d] = v;
                k_dst[d] = f32_to_fp16_bits(v);
            }
        } else if (kv_mode == 2) {
            uint8_t* k_dst = reinterpret_cast<uint8_t*>(k_cache) + ((size_t)slot * n_head_kv + kh) * head_dim;
            const float f_sc = factor * kFp8KScale;
            for (int d = lane; d < head_dim; d += 32) {
                k_dst[d] = f32_to_e4m3fn(kh_ptr[d] * f_sc);
            }
        } else {
            const int nb_head = head_dim / 32;
            block_q8_0* k_dst_b = reinterpret_cast<block_q8_0*>(k_cache) + ((size_t)slot * n_head_kv + kh) * nb_head;
            for (int b = 0; b < nb_head; ++b) {
                const float v = kh_ptr[b * 32 + lane] * factor;
                kh_ptr[b * 32 + lane] = v;
                const float amax = warp_reduce_max(fabsf(v));
                const float d_sc = amax / 127.0f;
                int q = (d_sc > 0.0f) ? __float2int_rn(v / d_sc) : 0;
                q = max(-127, min(127, q));
                k_dst_b[b].qs[lane] = (int8_t)q;
                if (lane == 0) k_dst_b[b].d = f32_to_fp16_bits(d_sc);
            }
        }
    } else if (rem < n_head + 2 * n_head_kv) {
        const int kh = rem - n_head - n_head_kv;
        const int slot = pos[m] % ring_cap;
        float* vh_ptr = v_cur + ((size_t)m * n_head_kv + kh) * head_dim;
        float sum2 = 0.0f;
        for (int d = lane; d < head_dim; d += 32) {
            const float v = vh_ptr[d];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        const float inv_rms = rsqrtf(sum2 / (float)head_dim + eps);
        if (kv_mode == 0) {
            uint16_t* v_dst = reinterpret_cast<uint16_t*>(v_cache) + ((size_t)slot * n_head_kv + kh) * head_dim;
            for (int d = lane; d < head_dim; d += 32) {
                const float v = vh_ptr[d] * inv_rms;
                vh_ptr[d] = v;
                v_dst[d] = f32_to_fp16_bits(v);
            }
        } else if (kv_mode == 2) {
            uint8_t* v_dst = reinterpret_cast<uint8_t*>(v_cache) + ((size_t)slot * n_head_kv + kh) * head_dim;
            const float f_sc = inv_rms * kFp8VScale;
            for (int d = lane; d < head_dim; d += 32) {
                v_dst[d] = f32_to_e4m3fn(vh_ptr[d] * f_sc);
            }
        } else {
            const int nb_head = head_dim / 32;
            block_q8_0* v_dst_b = reinterpret_cast<block_q8_0*>(v_cache) + ((size_t)slot * n_head_kv + kh) * nb_head;
            for (int b = 0; b < nb_head; ++b) {
                const float v = vh_ptr[b * 32 + lane] * inv_rms;
                vh_ptr[b * 32 + lane] = v;
                const float amax = warp_reduce_max(fabsf(v));
                const float d_sc = amax / 127.0f;
                int q = (d_sc > 0.0f) ? __float2int_rn(v / d_sc) : 0;
                q = max(-127, min(127, q));
                v_dst_b[b].qs[lane] = (int8_t)q;
                if (lane == 0) v_dst_b[b].d = f32_to_fp16_bits(d_sc);
            }
        }
    } else {
        const int h = rem - n_head - 2 * n_head_kv;
        float* rh = r_cur + ((size_t)m * n_head + h) * d_rel;
        const float* sh = r_clip + (size_t)h * d_rel;
        float sum2 = 0.0f;
        for (int d = lane; d < d_rel; d += 32) {
            const float v = rh[d] * sh[d];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        const float factor = fminf(1.0f, rel_clip_th / (sqrtf(sum2) + 1e-6f)) * t_mul;
        for (int d = lane; d < d_rel; d += 32) {
            rh[d] = (rh[d] * sh[d]) * factor;
        }
    }
}

__device__ __forceinline__ int wa_active_tok(int qpos, int window, int nw, int& k_lo, int& n_steps) {
    k_lo = (window > 0 && qpos + 1 > window) ? (qpos + 1 - window) : 0;
    n_steps = (qpos + 1 - k_lo + 15) >> 4;
    return min(nw, max(1, (n_steps + kWaMin - 1) / kWaMin));
}

__device__ __forceinline__ bool wa_same_split(int pos0, int T, int window) {
    if (window > 0) return false;
    return ((pos0 + 16) >> 4) == ((pos0 + T + 15) >> 4);
}

__device__ __forceinline__ int wa_active_block(int pos0, int T, int window, int nw, int& k_lo, int& n_steps) {
    k_lo = (window > 0 && pos0 + 1 > window) ? (pos0 + 1 - window) : 0;
    n_steps = (pos0 + T - k_lo + 15) >> 4;
    return min(nw, max(1, (n_steps + kWaMin - 1) / kWaMin));
}

template <class KVT, int NT>
__global__ void __launch_bounds__(kWaNW * 32, 2)
attn_block_mma_kernel(
    const uint16_t* __restrict__ q16,
    const KVT* __restrict__ kc,
    const KVT* __restrict__ vc,
    int kv_cap,
    int H,
    int Hkv,
    const int* __restrict__ pos_d,
    int T,
    float qk_scale,
    const float* __restrict__ r,
    int r_ld,
    const float* __restrict__ proj,
    int d_rel,
    int extent,
    int window,
    int n_split,
    bool only_if_same_split,
    float* __restrict__ o_part,
    float2* __restrict__ ml_part
) {
    constexpr bool F8 = sizeof(KVT) == 1;
    const int pos0 = *pos_d;
    if (only_if_same_split && !wa_same_split(pos0, T, window)) return;
    const int lane = threadIdx.x & 31, gr = lane >> 2, tq = lane & 3;
    const int warp_in_blk = threadIdx.x >> 5;
    const int g = blockIdx.x, R = H / Hkv, KD = Hkv * kFaD, NC = T * R;
    const int w = blockIdx.y * kWaNW + warp_in_blk;
    int k_lo, n_steps;
    const int n_act = wa_active_block(pos0, T, window, n_split * kWaNW, k_lo, n_steps);
    if (w >= n_act) return;
    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = min(n_steps, w * per), st1 = min(n_steps, st0 + per);
    const int qlast = pos0 + T - 1;

    __shared__ uint2 s_qf[kWaNW][(NT == 3) ? 3 : 1][kFaD / 16][32];
    uint32_t qf[(NT < 3) ? NT : 1][kFaD / 16][2];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int cq = j * 8 + gr;
        const bool ok = cq < NC;
        const int tqq = ok ? cq / R : 0, hq = ok ? cq % R : 0;
        const uint2* qr = reinterpret_cast<const uint2*>(q16 + ((size_t)tqq * H + g * R + hq) * kFaD);
        #pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            const uint2 v = __ldg(qr + kk * 4 + tq);
            const uint2 qv = ok ? v : make_uint2(0u, 0u);
            if constexpr (NT == 3) {
                s_qf[warp_in_blk][j][kk][lane] = qv;
            } else {
                qf[j][kk][0] = qv.x;
                qf[j][kk][1] = qv.y;
            }
        }
    }

    float o[NT][kFaD / 16][4];
    float ma[NT], mb[NT], la[NT], lb[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        ma[j] = mb[j] = -INFINITY;
        la[j] = lb[j] = 0.0f;
        #pragma unroll
        for (int jj = 0; jj < kFaD / 16; ++jj) {
            o[j][jj][0] = o[j][jj][1] = o[j][jj][2] = o[j][jj][3] = 0.0f;
        }
    }

    const float sc = qk_scale * kLog2e;
    const KVT* kg = kc + (size_t)g * kFaD + tq * 4;
    const KVT* vg = vc + (size_t)g * kFaD + gr * 16;
    const int src = tq * 8 + (gr >> 1);
    const uint32_t psel = (gr & 1) ? 0x7632u : 0x5410u;
    const bool windowed = window > 0;
    const bool rel4 = ((d_rel & 3) == 0);

    for (int st = st0; st < st1; ++st) {
        const int kt = k_lo + st * 16;
        const int kmax = qlast - kt;
        const int row0 = kv_cap > 0 ? kt % kv_cap : kt;
        auto row = [&](int key) {
            int rw = row0 + min(key, kmax);
            if (kv_cap > 0 && rw >= kv_cap) rw -= kv_cap;
            return (size_t)rw * KD;
        };
        uint32_t ka[kFaD / 16][4];
        {
            const KVT* p0 = kg + row(gr);
            const KVT* p1 = kg + row(gr + 8);
            if constexpr (F8) {
                #pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    const uint32_t w0 = __ldg(reinterpret_cast<const uint32_t*>(p0 + kk * 16));
                    const uint32_t w1 = __ldg(reinterpret_cast<const uint32_t*>(p1 + kk * 16));
                    e4m3x4_h4(w0, ka[kk][0], ka[kk][2]);
                    e4m3x4_h4(w1, ka[kk][1], ka[kk][3]);
                }
            } else {
                #pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    const uint2 v0 = __ldg(reinterpret_cast<const uint2*>(p0 + kk * 16));
                    const uint2 v1 = __ldg(reinterpret_cast<const uint2*>(p1 + kk * 16));
                    ka[kk][0] = v0.x; ka[kk][2] = v0.y; ka[kk][1] = v1.x; ka[kk][3] = v1.y;
                }
            }
        }
        const int k0 = kt + gr, k1 = k0 + 8;
        const bool near = pos0 - (kt + 15) < extent;
        uint32_t pb0[NT], pb1[NT];
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int kk = 0; kk < kFaD / 16; ++kk) {
                uint32_t q0, q1;
                if constexpr (NT == 3) {
                    const uint2 qv = s_qf[warp_in_blk][j][kk][lane];
                    q0 = qv.x; q1 = qv.y;
                } else {
                    q0 = qf[j][kk][0]; q1 = qf[j][kk][1];
                }
                mma16816_f16((kk & 1) ? sb : sa, ka[kk], q0, q1);
            }
            float c0 = (sa[0] + sb[0]) * sc;
            float c1 = (sa[1] + sb[1]) * sc;
            float c2 = (sa[2] + sb[2]) * sc;
            float c3 = (sa[3] + sb[3]) * sc;
            const int ca = j * 8 + 2 * tq, cb = ca + 1;
            const bool a_ok = ca < NC, b_ok = cb < NC;
            const int pa = a_ok ? (pos0 + ca / R) : -1;
            const int pb = b_ok ? (pos0 + cb / R) : -1;
            if (near) {
                const int d0a = pa - k0, d1a = pa - k1, d0b = pb - k0, d1b = pb - k1;
                const bool u0a = a_ok && ((unsigned)d0a < (unsigned)extent);
                const bool u1a = a_ok && ((unsigned)d1a < (unsigned)extent);
                const bool u0b = b_ok && ((unsigned)d0b < (unsigned)extent);
                const bool u1b = b_ok && ((unsigned)d1b < (unsigned)extent);
                const float* rp_a = r + (size_t)(a_ok ? ca / R : 0) * r_ld + (size_t)(g * R + (a_ok ? ca % R : 0)) * d_rel;
                const float* rp_b = r + (size_t)(b_ok ? cb / R : 0) * r_ld + (size_t)(g * R + (b_ok ? cb % R : 0)) * d_rel;
                float b0a = 0.0f, b1a = 0.0f, b0b = 0.0f, b1b = 0.0f;
                const float* p0 = proj + (size_t)(u0a ? d0a : 0) * d_rel;
                const float* p1 = proj + (size_t)(u1a ? d1a : 0) * d_rel;
                const float* p2 = proj + (size_t)(u0b ? d0b : 0) * d_rel;
                const float* p3 = proj + (size_t)(u1b ? d1b : 0) * d_rel;
                if (rel4) {
                    const float4 *r4a = reinterpret_cast<const float4*>(rp_a), *r4b = reinterpret_cast<const float4*>(rp_b);
                    const float4 *q40 = reinterpret_cast<const float4*>(p0), *q41 = reinterpret_cast<const float4*>(p1);
                    const float4 *q42 = reinterpret_cast<const float4*>(p2), *q43 = reinterpret_cast<const float4*>(p3);
                    for (int dd = 0; dd < (d_rel >> 2); ++dd) {
                        const float4 va = __ldg(r4a + dd), vb = __ldg(r4b + dd);
                        if (u0a) { const float4 wv = __ldg(q40 + dd); b0a += va.x*wv.x + va.y*wv.y + va.z*wv.z + va.w*wv.w; }
                        if (u1a) { const float4 wv = __ldg(q41 + dd); b1a += va.x*wv.x + va.y*wv.y + va.z*wv.z + va.w*wv.w; }
                        if (u0b) { const float4 wv = __ldg(q42 + dd); b0b += vb.x*wv.x + vb.y*wv.y + vb.z*wv.z + vb.w*wv.w; }
                        if (u1b) { const float4 wv = __ldg(q43 + dd); b1b += vb.x*wv.x + vb.y*wv.y + vb.z*wv.z + vb.w*wv.w; }
                    }
                } else {
                    for (int dd = 0; dd < d_rel; ++dd) {
                        const float va = __ldg(rp_a + dd), vb = __ldg(rp_b + dd);
                        if (u0a) b0a += va * __ldg(p0 + dd);
                        if (u1a) b1a += va * __ldg(p1 + dd);
                        if (u0b) b0b += vb * __ldg(p2 + dd);
                        if (u1b) b1b += vb * __ldg(p3 + dd);
                    }
                }
                c0 += b0a * kLog2e;
                c2 += b1a * kLog2e;
                c1 += b0b * kLog2e;
                c3 += b1b * kLog2e;
            }
            const bool v0a = a_ok && k0 <= pa && (!windowed || pa - k0 < window);
            const bool v1a = a_ok && k1 <= pa && (!windowed || pa - k1 < window);
            const bool v0b = b_ok && k0 <= pb && (!windowed || pb - k0 < window);
            const bool v1b = b_ok && k1 <= pb && (!windowed || pb - k1 < window);
            if (!v0a) c0 = -INFINITY;
            if (!v1a) c2 = -INFINITY;
            if (!v0b) c1 = -INFINITY;
            if (!v1b) c3 = -INFINITY;

            float mxa = fmaxf(c0, c2), mxb = fmaxf(c1, c3);
            #pragma unroll
            for (int sh = 4; sh < 32; sh <<= 1) {
                mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, sh));
                mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, sh));
            }
            const float mna = fmaxf(ma[j], mxa), mnb = fmaxf(mb[j], mxb);
            const float aa = mna > -INFINITY ? exp2f(ma[j] - mna) : 0.0f;
            const float ab = mnb > -INFINITY ? exp2f(mb[j] - mnb) : 0.0f;
            const float p0a = mna > -INFINITY ? exp2f(c0 - mna) : 0.0f;
            const float p2a = mna > -INFINITY ? exp2f(c2 - mna) : 0.0f;
            const float p1b = mnb > -INFINITY ? exp2f(c1 - mnb) : 0.0f;
            const float p3b = mnb > -INFINITY ? exp2f(c3 - mnb) : 0.0f;
            ma[j] = mna; mb[j] = mnb;
            la[j] = la[j] * aa + p0a + p2a;
            lb[j] = lb[j] * ab + p1b + p3b;

            #pragma unroll
            for (int jj = 0; jj < kFaD / 16; ++jj) {
                o[j][jj][0] *= aa; o[j][jj][2] *= aa;
                o[j][jj][1] *= ab; o[j][jj][3] *= ab;
            }
            const uint32_t plo = f2_to_u32(p0a, p1b), phi = f2_to_u32(p2a, p3b);
            const uint32_t alo = __shfl_sync(0xffffffffu, plo, src);
            const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src);
            const uint32_t blo = __shfl_sync(0xffffffffu, plo, src + 4);
            const uint32_t bhi = __shfl_sync(0xffffffffu, phi, src + 4);
            const bool col_ok = j * 8 + gr < NC;
            pb0[j] = col_ok ? __byte_perm(alo, blo, psel) : 0u;
            pb1[j] = col_ok ? __byte_perm(ahi, bhi, psel) : 0u;
        }

        {
            const KVT* v0 = vg + row(2 * tq);
            const KVT* v1 = vg + row(2 * tq + 1);
            const KVT* v8 = vg + row(2 * tq + 8);
            const KVT* v9 = vg + row(2 * tq + 9);
            if constexpr (F8) {
                const uint4 r0 = __ldg(reinterpret_cast<const uint4*>(v0));
                const uint4 r1 = __ldg(reinterpret_cast<const uint4*>(v1));
                const uint4 r8 = __ldg(reinterpret_cast<const uint4*>(v8));
                const uint4 r9 = __ldg(reinterpret_cast<const uint4*>(v9));
                const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
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
                    const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                    const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t a[4] = {
                            __byte_perm(x0[jj], x1[jj], 0x5410u), __byte_perm(x0[jj], x1[jj], 0x7632u),
                            __byte_perm(x8[jj], x9[jj], 0x5410u), __byte_perm(x8[jj], x9[jj], 0x7632u)
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
        const int ca = j * 8 + 2 * tq, cb = ca + 1;
        if (ca < NC) {
            const size_t ix = (size_t)w * T * H + (size_t)(ca / R) * H + g * R + ca % R;
            float* op = o_part + ix * kFaD + gr * 16;
            #pragma unroll
            for (int jj = 0; jj < kFaD / 16; ++jj) {
                reinterpret_cast<float2*>(op)[jj] = make_float2(o[j][jj][0], o[j][jj][2]);
            }
            if (gr == 0) ml_part[ix] = make_float2(ma[j], la[j]);
        }
        if (cb < NC) {
            const size_t ix = (size_t)w * T * H + (size_t)(cb / R) * H + g * R + cb % R;
            float* op = o_part + ix * kFaD + gr * 16;
            #pragma unroll
            for (int jj = 0; jj < kFaD / 16; ++jj) {
                reinterpret_cast<float2*>(op)[jj] = make_float2(o[j][jj][1], o[j][jj][3]);
            }
            if (gr == 0) ml_part[ix] = make_float2(mb[j], lb[j]);
        }
    }
}

template <class KVT>
__global__ void __launch_bounds__(kWaNW * 32, 2)
attn_decode_mma_toks_kernel(
    const uint16_t* __restrict__ q16,
    const KVT* __restrict__ kc,
    const KVT* __restrict__ vc,
    int kv_cap,
    int H,
    int Hkv,
    const int* __restrict__ pos_d,
    int T,
    float qk_scale,
    const float* __restrict__ r,
    int r_ld,
    const float* __restrict__ proj,
    int d_rel,
    int extent,
    int window,
    int n_split,
    bool skip_if_same_split,
    float* __restrict__ o_part,
    float2* __restrict__ ml_part
) {
    constexpr bool F8 = sizeof(KVT) == 1;
    if (skip_if_same_split && wa_same_split(*pos_d, T, window)) return;
    const int tok = blockIdx.z;
    const int qpos = pos_d[tok];
    const int lane = threadIdx.x & 31, gr = lane >> 2, tq = lane & 3;
    const int warp_in_blk = threadIdx.x >> 5;
    const int g = blockIdx.x, R = H / Hkv, KD = Hkv * kFaD;
    const int w = blockIdx.y * kWaNW + warp_in_blk;
    int k_lo, n_steps;
    const int n_act = wa_active_tok(qpos, window, n_split * kWaNW, k_lo, n_steps);
    if (w >= n_act) return;
    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = min(n_steps, w * per), st1 = min(n_steps, st0 + per);

    uint32_t qf[kFaD / 16][2];
    const bool q_ok = gr < R;
    const uint2* qr = reinterpret_cast<const uint2*>(q16 + ((size_t)tok * H + g * R + (q_ok ? gr : 0)) * kFaD);
    #pragma unroll
    for (int kk = 0; kk < kFaD / 16; ++kk) {
        const uint2 v = __ldg(qr + kk * 4 + tq);
        qf[kk][0] = q_ok ? v.x : 0u;
        qf[kk][1] = q_ok ? v.y : 0u;
    }

    float o[kFaD / 16][4];
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    #pragma unroll
    for (int jj = 0; jj < kFaD / 16; ++jj) {
        o[jj][0] = o[jj][1] = o[jj][2] = o[jj][3] = 0.0f;
    }

    const float sc = qk_scale * kLog2e;
    const KVT* kg = kc + (size_t)g * kFaD + tq * 4;
    const KVT* vg = vc + (size_t)g * kFaD + gr * 16;
    const int src = tq * 8 + (gr >> 1);
    const uint32_t psel = (gr & 1) ? 0x7632u : 0x5410u;
    const bool windowed = window > 0;
    const bool rel4 = ((d_rel & 3) == 0);
    const int ca = 2 * tq, cb = ca + 1;
    const bool a_ok = ca < R, b_ok = cb < R;
    const float* rp_a = r + (size_t)tok * r_ld + (size_t)(g * R + (a_ok ? ca : 0)) * d_rel;
    const float* rp_b = r + (size_t)tok * r_ld + (size_t)(g * R + (b_ok ? cb : 0)) * d_rel;

    for (int st = st0; st < st1; ++st) {
        const int kt = k_lo + st * 16;
        const int kmax = qpos - kt;
        const int row0 = kv_cap > 0 ? kt % kv_cap : kt;
        auto row = [&](int key) {
            int rw = row0 + min(key, kmax);
            if (kv_cap > 0 && rw >= kv_cap) rw -= kv_cap;
            return (size_t)rw * KD;
        };
        uint32_t ka[kFaD / 16][4];
        {
            const KVT* p0 = kg + row(gr);
            const KVT* p1 = kg + row(gr + 8);
            if constexpr (F8) {
                #pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    const uint32_t w0 = __ldg(reinterpret_cast<const uint32_t*>(p0 + kk * 16));
                    const uint32_t w1 = __ldg(reinterpret_cast<const uint32_t*>(p1 + kk * 16));
                    e4m3x4_h4(w0, ka[kk][0], ka[kk][2]);
                    e4m3x4_h4(w1, ka[kk][1], ka[kk][3]);
                }
            } else {
                #pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    const uint2 v0 = __ldg(reinterpret_cast<const uint2*>(p0 + kk * 16));
                    const uint2 v1 = __ldg(reinterpret_cast<const uint2*>(p1 + kk * 16));
                    ka[kk][0] = v0.x; ka[kk][2] = v0.y; ka[kk][1] = v1.x; ka[kk][3] = v1.y;
                }
            }
        }
        const int k0 = kt + gr, k1 = k0 + 8;
        const bool near = qpos - (kt + 15) < extent;
        float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            mma16816_f16((kk & 1) ? sb : sa, ka[kk], qf[kk][0], qf[kk][1]);
        }
        float c0 = (sa[0] + sb[0]) * sc;
        float c1 = (sa[1] + sb[1]) * sc;
        float c2 = (sa[2] + sb[2]) * sc;
        float c3 = (sa[3] + sb[3]) * sc;
        if (near) {
            const int d0 = qpos - k0, d1 = qpos - k1;
            const bool u0 = ((unsigned)d0 < (unsigned)extent);
            const bool u1 = ((unsigned)d1 < (unsigned)extent);
            float b0a = 0.0f, b1a = 0.0f, b0b = 0.0f, b1b = 0.0f;
            const float* p0 = proj + (size_t)(u0 ? d0 : 0) * d_rel;
            const float* p1 = proj + (size_t)(u1 ? d1 : 0) * d_rel;
            if (rel4) {
                const float4 *r4a = reinterpret_cast<const float4*>(rp_a), *r4b = reinterpret_cast<const float4*>(rp_b);
                const float4 *q40 = reinterpret_cast<const float4*>(p0), *q41 = reinterpret_cast<const float4*>(p1);
                for (int dd = 0; dd < (d_rel >> 2); ++dd) {
                    const float4 va = __ldg(r4a + dd), vb = __ldg(r4b + dd);
                    if (u0) {
                        const float4 wv = __ldg(q40 + dd);
                        if (a_ok) b0a += va.x*wv.x + va.y*wv.y + va.z*wv.z + va.w*wv.w;
                        if (b_ok) b0b += vb.x*wv.x + vb.y*wv.y + vb.z*wv.z + vb.w*wv.w;
                    }
                    if (u1) {
                        const float4 wv = __ldg(q41 + dd);
                        if (a_ok) b1a += va.x*wv.x + va.y*wv.y + va.z*wv.z + va.w*wv.w;
                        if (b_ok) b1b += vb.x*wv.x + vb.y*wv.y + vb.z*wv.z + vb.w*wv.w;
                    }
                }
            } else {
                for (int dd = 0; dd < d_rel; ++dd) {
                    const float va = __ldg(rp_a + dd), vb = __ldg(rp_b + dd);
                    if (u0) { const float wv = __ldg(p0 + dd); if (a_ok) b0a += va * wv; if (b_ok) b0b += vb * wv; }
                    if (u1) { const float wv = __ldg(p1 + dd); if (a_ok) b1a += va * wv; if (b_ok) b1b += vb * wv; }
                }
            }
            c0 += b0a * kLog2e;
            c2 += b1a * kLog2e;
            c1 += b0b * kLog2e;
            c3 += b1b * kLog2e;
        }
        const bool v0ok = k0 <= qpos && (!windowed || qpos - k0 < window);
        const bool v1ok = k1 <= qpos && (!windowed || qpos - k1 < window);
        if (!(a_ok && v0ok)) c0 = -INFINITY;
        if (!(a_ok && v1ok)) c2 = -INFINITY;
        if (!(b_ok && v0ok)) c1 = -INFINITY;
        if (!(b_ok && v1ok)) c3 = -INFINITY;

        float mxa = fmaxf(c0, c2), mxb = fmaxf(c1, c3);
        #pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, sh));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, sh));
        }
        const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
        const float aa = mna > -INFINITY ? exp2f(ma - mna) : 0.0f;
        const float ab = mnb > -INFINITY ? exp2f(mb - mnb) : 0.0f;
        const float p0a = mna > -INFINITY ? exp2f(c0 - mna) : 0.0f;
        const float p2a = mna > -INFINITY ? exp2f(c2 - mna) : 0.0f;
        const float p1b = mnb > -INFINITY ? exp2f(c1 - mnb) : 0.0f;
        const float p3b = mnb > -INFINITY ? exp2f(c3 - mnb) : 0.0f;
        ma = mna; mb = mnb;
        la = la * aa + p0a + p2a;
        lb = lb * ab + p1b + p3b;

        #pragma unroll
        for (int jj = 0; jj < kFaD / 16; ++jj) {
            o[jj][0] *= aa; o[jj][2] *= aa;
            o[jj][1] *= ab; o[jj][3] *= ab;
        }
        const uint32_t plo = f2_to_u32(p0a, p1b), phi = f2_to_u32(p2a, p3b);
        const uint32_t alo = __shfl_sync(0xffffffffu, plo, src);
        const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src);
        const uint32_t blo = __shfl_sync(0xffffffffu, plo, src + 4);
        const uint32_t bhi = __shfl_sync(0xffffffffu, phi, src + 4);
        const uint32_t pb0 = q_ok ? __byte_perm(alo, blo, psel) : 0u;
        const uint32_t pb1 = q_ok ? __byte_perm(ahi, bhi, psel) : 0u;

        {
            const KVT* v0 = vg + row(2 * tq);
            const KVT* v1 = vg + row(2 * tq + 1);
            const KVT* v8 = vg + row(2 * tq + 8);
            const KVT* v9 = vg + row(2 * tq + 9);
            if constexpr (F8) {
                const uint4 r0 = __ldg(reinterpret_cast<const uint4*>(v0));
                const uint4 r1 = __ldg(reinterpret_cast<const uint4*>(v1));
                const uint4 r8 = __ldg(reinterpret_cast<const uint4*>(v8));
                const uint4 r9 = __ldg(reinterpret_cast<const uint4*>(v9));
                const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t a[4];
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x5140u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x5140u), a[2], a[3]);
                    mma16816_f16(o[2 * i], a, pb0, pb1);
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x7362u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x7362u), a[2], a[3]);
                    mma16816_f16(o[2 * i + 1], a, pb0, pb1);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 r0 = __ldg(reinterpret_cast<const uint4*>(v0) + i);
                    const uint4 r1 = __ldg(reinterpret_cast<const uint4*>(v1) + i);
                    const uint4 r8 = __ldg(reinterpret_cast<const uint4*>(v8) + i);
                    const uint4 r9 = __ldg(reinterpret_cast<const uint4*>(v9) + i);
                    const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                    const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
                    #pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t a[4] = {
                            __byte_perm(x0[jj], x1[jj], 0x5410u), __byte_perm(x0[jj], x1[jj], 0x7632u),
                            __byte_perm(x8[jj], x9[jj], 0x5410u), __byte_perm(x8[jj], x9[jj], 0x7632u)
                        };
                        mma16816_f16(o[4 * i + jj], a, pb0, pb1);
                    }
                }
            }
        }
    }

    #pragma unroll
    for (int sh = 4; sh < 32; sh <<= 1) {
        la += __shfl_xor_sync(0xffffffffu, la, sh);
        lb += __shfl_xor_sync(0xffffffffu, lb, sh);
    }
    if (a_ok) {
        const size_t ix = (size_t)w * T * H + (size_t)tok * H + g * R + ca;
        float* op = o_part + ix * kFaD + gr * 16;
        #pragma unroll
        for (int jj = 0; jj < kFaD / 16; ++jj) {
            reinterpret_cast<float2*>(op)[jj] = make_float2(o[jj][0], o[jj][2]);
        }
        if (gr == 0) ml_part[ix] = make_float2(ma, la);
    }
    if (b_ok) {
        const size_t ix = (size_t)w * T * H + (size_t)tok * H + g * R + cb;
        float* op = o_part + ix * kFaD + gr * 16;
        #pragma unroll
        for (int jj = 0; jj < kFaD / 16; ++jj) {
            reinterpret_cast<float2*>(op)[jj] = make_float2(o[jj][1], o[jj][3]);
        }
        if (gr == 0) ml_part[ix] = make_float2(mb, lb);
    }
}

__global__ void attn_block_combine_kernel(
    const float* __restrict__ o_part,
    const float2* __restrict__ ml_part,
    int n_split,
    int H,
    int T,
    const float* __restrict__ sink,
    float v_inv,
    const int* __restrict__ pos_d,
    int window,
    const float* __restrict__ gate,
    int gate_ld,
    float* __restrict__ out32,
    uint16_t* __restrict__ out16
) {
    extern __shared__ float cm_sm[];
    float* red = cm_sm;
    float* cw = cm_sm + kFaD;
    const int h = blockIdx.x, t = blockIdx.y, d = threadIdx.x, TH = T * H, ix = t * H + h;
    int k_lo, n_steps;
    const int n_act = wa_active_tok(pos_d[t], window, n_split * kWaNW, k_lo, n_steps);
    const float sk = sink ? (sink[ix] * kLog2e) : -INFINITY;
    float m = (d == 0) ? sk : -INFINITY;
    for (int s = d; s < n_act; s += kFaD) {
        m = fmaxf(m, ml_part[(size_t)s * TH + ix].x);
    }
    m = bmax(m, red);
    float den = (d == 0 && sk > -INFINITY) ? exp2f(sk - m) : 0.0f;
    for (int s = d; s < n_act; s += kFaD) {
        const float2 ml = ml_part[(size_t)s * TH + ix];
        const float c = (ml.x == -INFINITY) ? 0.0f : exp2f(ml.x - m);
        cw[s] = c;
        den += ml.y * c;
    }
    den = bsum(den, red);
    float acc = 0.0f;
    for (int s = 0; s < n_act; ++s) {
        acc += o_part[((size_t)s * TH + ix) * kFaD + d] * cw[s];
    }
    float res = (acc / den) * v_inv;
    if (gate) {
        const float g = gate[(size_t)t * gate_ld + (size_t)h * kFaD + d];
        res *= 1.0f / (1.0f + expf(-g));
    }
    if (out32) out32[(size_t)ix * kFaD + d] = res;
    if (out16) out16[(size_t)ix * kFaD + d] = f32_to_fp16_bits(res);
}

// -----------------------------------------------------------------------------
// 5. Grouped Multi-Token MoE Tensor Core Kernels (T = 1..8, NVFP4 & Q8_0)
// -----------------------------------------------------------------------------

__global__ void group_route_kernel(
    const int32_t* __restrict__ top,
    const float* __restrict__ w,
    int S,
    int K,
    const uint64_t* __restrict__ gu_tab,
    const uint64_t* __restrict__ dn_tab,
    const float* __restrict__ sg_t,
    const float* __restrict__ dn_sc_t,
    uint64_t* __restrict__ grp_gu_ptrs,
    uint64_t* __restrict__ grp_dn_ptrs,
    int32_t* __restrict__ grp_n,
    int32_t* __restrict__ grp_tok,
    int32_t* __restrict__ grp_row,
    float* __restrict__ sg,
    float* __restrict__ su,
    float* __restrict__ wt
) {
    __shared__ int ex[64], lead[64], gid[64];
    const int i = threadIdx.x;
    const int e = (i < S) ? top[i] : -1;
    ex[i] = e;
    if (i < S) grp_n[i] = 0;
    __syncthreads();
    int first = i, m = 0, cnt = 0;
    if (i < S) {
        for (int j = 0; j < S; ++j) {
            if (ex[j] == e) {
                if (j < first) first = j;
                if (j < i) ++m;
                ++cnt;
            }
        }
    }
    lead[i] = (i < S && first == i);
    __syncthreads();
    if (lead[i]) {
        int gx = 0;
        for (int j = 0; j < i; ++j) gx += lead[j];
        gid[i] = gx;
    }
    __syncthreads();
    if (i >= S) return;
    const int g = gid[first];
    grp_tok[g * kGroupMax + m] = i / K;
    grp_row[g * kGroupMax + m] = i;
    if (first == i) {
        grp_gu_ptrs[g] = gu_tab[e];
        grp_dn_ptrs[g] = dn_tab[e];
        grp_n[g] = cnt;
    }
    const float sc_gu = sg_t ? sg_t[e] : 1.0f;
    sg[i] = sc_gu;
    su[i] = sc_gu;
    wt[i] = w[i] * (dn_sc_t ? dn_sc_t[e] : 1.0f);
}

__global__ void f16_limbs_rows_kernel(
    const float* __restrict__ x,
    int T,
    int C,
    uint16_t* __restrict__ out
) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (t >= T || c >= C) return;
    const float v = x[(size_t)t * C + c];
    const uint16_t hi = f32_to_fp16_bits(v);
    const uint16_t lo = f32_to_fp16_bits(v - fp16_bits_to_f32(hi));
    out[((size_t)t * 2 + 0) * C + c] = hi;
    out[((size_t)t * 2 + 1) * C + c] = lo;
}

template <int SPLIT, int NT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_block_gu_nvfp4_kernel(
    const uint64_t* __restrict__ blob,
    const int32_t* __restrict__ n_g,
    const int32_t* __restrict__ tok_g,
    const int32_t* __restrict__ row_g,
    size_t off_up,
    int C,
    int F,
    const uint16_t* __restrict__ limbs,
    const float* __restrict__ sg,
    const float* __restrict__ su,
    float gcap,
    float ucap,
    uint16_t* __restrict__ h
) {
    __shared__ float2 part[SPLIT][NT][32];
    const int grp = blockIdx.y;
    const int n = n_g[grp];
    if (n <= 0 || blob[grp] == 0ULL) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 8 + gr;
    const size_t rb = (size_t)(C / 64) * 36;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(blob[grp])) + (size_t)r * rb;
    const uint8_t* b = a + off_up;
    const uint4* xp[NT];
    bool x_ok[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int mem = (j * 8 + gr) >> 1, limb = gr & 1;
        x_ok[j] = mem < n;
        const int tk = x_ok[j] ? tok_g[grp * kGroupMax + mem] : 0;
        xp[j] = reinterpret_cast<const uint4*>(limbs + ((size_t)tk * 2 + limb) * C) + 2 * t;
    }
    float v[NT][4];
    #pragma unroll
    for (int j = 0; j < NT; ++j) v[j][0] = v[j][1] = v[j][2] = v[j][3] = 0.0f;
    const int nk = C / 64;
    NvMmaStep cur = nv_mma_step_load(a, b, split < nk ? split : 0, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvMmaStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_mma_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu);
        const uint32_t sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        uint32_t af[4][4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1));
            const uint32_t qb = wb[m >> 1] >> (16 * (m & 1));
            af[m][0] = nv_pair_h2(qa, sa);
            af[m][1] = nv_pair_h2(qb, sb);
            af[m][2] = nv_pair_h2(qa >> 4, sa);
            af[m][3] = nv_pair_h2(qb >> 4, sb);
        }
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j && n <= 4 * j) continue;
            uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
            if (x_ok[j]) {
                x0 = __ldg(xp[j] + k * 8);
                x1 = __ldg(xp[j] + k * 8 + 1);
            }
            const uint32_t xx[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            #pragma unroll
            for (int m = 0; m < 4; ++m) mma16816_f16(v[j], af[m], xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        part[split][j][lane] = make_float2(v[j][0] + v[j][1], v[j][2] + v[j][3]);
    }
    __syncthreads();
    if (split != 0) return;
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float2 tot = part[0][j][lane];
        #pragma unroll
        for (int s = 1; s < SPLIT; ++s) {
            tot.x += part[s][j][lane].x;
            tot.y += part[s][j][lane].y;
        }
        const int mem = j * 4 + t;
        if (mem < n) {
            const int slot = row_g[grp * kGroupMax + mem];
            const float gg = tot.x * (sg ? sg[slot] : 1.0f);
            const float uu = tot.y * (su ? su[slot] : 1.0f);
            const float g_cap = gcap * tanhf(gg / gcap);
            const float sig = 1.0f / (1.0f + expf(-gg));
            const float u_cap = (ucap > 0.0f) ? (ucap * tanhf(uu / ucap)) : uu;
            h[(size_t)slot * F + r] = f32_to_fp16_bits(g_cap * sig * u_cap);
        }
    }
}

template <int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_block_down_nvfp4_kernel(
    const uint64_t* __restrict__ blob,
    const int32_t* __restrict__ n_g,
    const int32_t* __restrict__ row_g,
    const float* __restrict__ wt,
    int C,
    int F,
    const uint16_t* __restrict__ h,
    float* __restrict__ y
) {
    __shared__ float4 part[SPLIT][32];
    const int grp = blockIdx.y;
    const int n = n_g[grp];
    if (n <= 0 || blob[grp] == 0ULL) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 16 + gr;
    const size_t rb = (size_t)(F / 64) * 36;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(blob[grp])) + (size_t)r * rb;
    const uint8_t* b = a + 8 * rb;
    const bool x_ok = gr < n;
    const uint4* xp = reinterpret_cast<const uint4*>(h + (size_t)(x_ok ? row_g[grp * kGroupMax + gr] : 0) * F) + 2 * t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = F / 64;
    NvMmaStep cur = nv_mma_step_load(a, b, split < nk ? split : 0, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvMmaStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_mma_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu);
        const uint32_t sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
        if (x_ok) {
            x0 = __ldg(xp + k * 8);
            x1 = __ldg(xp + k * 8 + 1);
        }
        const uint32_t xx[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1));
            const uint32_t qb = wb[m >> 1] >> (16 * (m & 1));
            const uint32_t af[4] = {
                nv_pair_h2(qa, sa), nv_pair_h2(qb, sb),
                nv_pair_h2(qa >> 4, sa), nv_pair_h2(qb >> 4, sb)
            };
            mma16816_f16(v, af, xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
    part[split][lane] = make_float4(v[0], v[1], v[2], v[3]);
    __syncthreads();
    if (split != 0) return;
    float4 tot = part[0][lane];
    #pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        const float4 z = part[s][lane];
        tot.x += z.x; tot.y += z.y; tot.z += z.z; tot.w += z.w;
    }
    if (2 * t < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t];
        const float w0 = wt[slot];
        y[(size_t)slot * C + r] = tot.x * w0;
        y[(size_t)slot * C + r + 8] = tot.z * w0;
    }
    if (2 * t + 1 < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t + 1];
        const float w1 = wt[slot];
        y[(size_t)slot * C + r] = tot.y * w1;
        y[(size_t)slot * C + r + 8] = tot.w * w1;
    }
}

struct Q8ExpStep {
    uint32_t sa, sb;
    uint32_t a[4], b[4];
};
__device__ __forceinline__ Q8ExpStep q8_exp_step_load(const uint8_t* a, const uint8_t* b, int k, int t) {
    const uint16_t* ap = reinterpret_cast<const uint16_t*>(a + (size_t)k * 34);
    const uint16_t* bp = reinterpret_cast<const uint16_t*>(b + (size_t)k * 34);
    Q8ExpStep r;
    r.sa = __ldg(ap);
    r.sb = __ldg(bp);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        r.a[i] = __ldg(ap + 1 + 4 * t + i);
        r.b[i] = __ldg(bp + 1 + 4 * t + i);
    }
    return r;
}

template <int SPLIT, int NT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_block_gu_q8_kernel(
    const uint64_t* __restrict__ blob,
    const int32_t* __restrict__ n_g,
    const int32_t* __restrict__ tok_g,
    const int32_t* __restrict__ row_g,
    size_t off_up,
    int C,
    int F,
    const uint16_t* __restrict__ limbs,
    const float* __restrict__ sg,
    const float* __restrict__ su,
    float gcap,
    float ucap,
    uint16_t* __restrict__ h
) {
    __shared__ float4 part[SPLIT][NT][32];
    const int grp = blockIdx.y;
    const int n = n_g[grp];
    if (n <= 0 || blob[grp] == 0ULL) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 8 + gr;
    const size_t rb = (size_t)(C / 32) * 34;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(blob[grp])) + (size_t)r * rb;
    const uint8_t* b = a + off_up;
    const uint4* xp[NT];
    bool x_ok[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int mem = (j * 8 + gr) >> 1, limb = gr & 1;
        x_ok[j] = mem < n;
        const int tk = x_ok[j] ? tok_g[grp * kGroupMax + mem] : 0;
        xp[j] = reinterpret_cast<const uint4*>(limbs + ((size_t)tk * 2 + limb) * C) + t;
    }
    float v[NT][4];
    #pragma unroll
    for (int j = 0; j < NT; ++j) v[j][0] = v[j][1] = v[j][2] = v[j][3] = 0.0f;
    const int nk = C / 32;
    Q8ExpStep cur = q8_exp_step_load(a, b, split < nk ? split : 0, t);
    for (int k = split; k < nk; k += SPLIT) {
        Q8ExpStep nxt = cur;
        if (k + SPLIT < nk) nxt = q8_exp_step_load(a, b, k + SPLIT, t);
        const float sa = fp16_bits_to_f32((uint16_t)cur.sa);
        const float sb = fp16_bits_to_f32((uint16_t)cur.sb);
        const uint32_t af0[4] = {q8_pair(cur.a[0]), q8_pair(cur.b[0]), q8_pair(cur.a[1]), q8_pair(cur.b[1])};
        const uint32_t af1[4] = {q8_pair(cur.a[2]), q8_pair(cur.b[2]), q8_pair(cur.a[3]), q8_pair(cur.b[3])};
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j && n <= 4 * j) continue;
            uint4 x = make_uint4(0, 0, 0, 0);
            if (x_ok[j]) x = __ldg(xp[j] + k * 4);
            float bs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            mma16816_f16(bs, af0, x.x, x.y);
            mma16816_f16(bs, af1, x.z, x.w);
            v[j][0] += sa * bs[0]; v[j][1] += sa * bs[1];
            v[j][2] += sb * bs[2]; v[j][3] += sb * bs[3];
        }
        cur = nxt;
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) part[split][j][lane] = make_float4(v[j][0], v[j][1], v[j][2], v[j][3]);
    __syncthreads();
    if (split != 0) return;
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float4 tot = part[0][j][lane];
        #pragma unroll
        for (int s = 1; s < SPLIT; ++s) {
            const float4 z = part[s][j][lane];
            tot.x += z.x; tot.y += z.y; tot.z += z.z; tot.w += z.w;
        }
        const int mem = j * 4 + t;
        if (mem < n) {
            const int slot = row_g[grp * kGroupMax + mem];
            const float gg = (tot.x + tot.y) * (sg ? sg[slot] : 1.0f);
            const float uu = (tot.z + tot.w) * (su ? su[slot] : 1.0f);
            const float g_cap = gcap * tanhf(gg / gcap);
            const float sig = 1.0f / (1.0f + expf(-gg));
            const float u_cap = (ucap > 0.0f) ? (ucap * tanhf(uu / ucap)) : uu;
            h[(size_t)slot * F + r] = f32_to_fp16_bits(g_cap * sig * u_cap);
        }
    }
}

template <int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_block_down_q8_kernel(
    const uint64_t* __restrict__ blob,
    const int32_t* __restrict__ n_g,
    const int32_t* __restrict__ row_g,
    const float* __restrict__ wt,
    int C,
    int F,
    const uint16_t* __restrict__ h,
    float* __restrict__ y
) {
    __shared__ float4 part[SPLIT][32];
    const int grp = blockIdx.y;
    const int n = n_g[grp];
    if (n <= 0 || blob[grp] == 0ULL) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 16 + gr;
    const size_t rb = (size_t)(F / 32) * 34;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(blob[grp])) + (size_t)r * rb;
    const uint8_t* b = a + 8 * rb;
    const bool x_ok = gr < n;
    const uint4* xp = reinterpret_cast<const uint4*>(h + (size_t)(x_ok ? row_g[grp * kGroupMax + gr] : 0) * F) + t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = F / 32;
    Q8ExpStep cur = q8_exp_step_load(a, b, split < nk ? split : 0, t);
    for (int k = split; k < nk; k += SPLIT) {
        Q8ExpStep nxt = cur;
        if (k + SPLIT < nk) nxt = q8_exp_step_load(a, b, k + SPLIT, t);
        const float sa = fp16_bits_to_f32((uint16_t)cur.sa);
        const float sb = fp16_bits_to_f32((uint16_t)cur.sb);
        const uint32_t af0[4] = {q8_pair(cur.a[0]), q8_pair(cur.b[0]), q8_pair(cur.a[1]), q8_pair(cur.b[1])};
        const uint32_t af1[4] = {q8_pair(cur.a[2]), q8_pair(cur.b[2]), q8_pair(cur.a[3]), q8_pair(cur.b[3])};
        uint4 x = make_uint4(0, 0, 0, 0);
        if (x_ok) x = __ldg(xp + k * 4);
        float bs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma16816_f16(bs, af0, x.x, x.y);
        mma16816_f16(bs, af1, x.z, x.w);
        v[0] += sa * bs[0]; v[1] += sa * bs[1];
        v[2] += sb * bs[2]; v[3] += sb * bs[3];
        cur = nxt;
    }
    part[split][lane] = make_float4(v[0], v[1], v[2], v[3]);
    __syncthreads();
    if (split != 0) return;
    float4 tot = part[0][lane];
    #pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        const float4 z = part[s][lane];
        tot.x += z.x; tot.y += z.y; tot.z += z.z; tot.w += z.w;
    }
    if (2 * t < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t];
        const float w0 = wt[slot];
        y[(size_t)slot * C + r] = tot.x * w0;
        y[(size_t)slot * C + r + 8] = tot.z * w0;
    }
    if (2 * t + 1 < n) {
        const int slot = row_g[grp * kGroupMax + 2 * t + 1];
        const float w1 = wt[slot];
        y[(size_t)slot * C + r] = tot.y * w1;
        y[(size_t)slot * C + r + 8] = tot.w * w1;
    }
}

__global__ void moe_combine_conv_add_block_kernel(
    const float* __restrict__ slot_out,
    const float* __restrict__ static_w,
    float* __restrict__ history,
    const float* __restrict__ ffn_inp,
    float* __restrict__ cur_out,
    float* __restrict__ rec_old_hist,
    float* __restrict__ rec_x,
    int n_sconv_channels,
    int slot_idx,
    int n_embd,
    int topk,
    int T,
    int conv_offset,
    int n_conv
) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_embd) return;

    const int g_ch = conv_offset + r;
    float* hist_ch = history + (size_t)g_ch * 3;
    float h0 = hist_ch[0], h1 = hist_ch[1], h2 = hist_ch[2];
    if (rec_old_hist != nullptr) {
        float* rh = rec_old_hist + ((size_t)slot_idx * n_sconv_channels + g_ch) * 3;
        rh[0] = h0; rh[1] = h1; rh[2] = h2;
    }

    const float* sw = static_w + (size_t)r * 4;
    const float sw0 = sw[0], sw1 = sw[1], sw2 = sw[2], sw3 = sw[3];
    const float inv_scale = rsqrtf((float)(n_conv + 1));

    for (int t = 0; t < T; ++t) {
        float xt = 0.0f;
        const float* base = slot_out + ((size_t)t * topk) * n_embd + r;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            if (k < topk) xt += base[(size_t)k * n_embd];
        }
        if (rec_x != nullptr) {
            rec_x[((size_t)slot_idx * 8 + t) * n_sconv_channels + g_ch] = xt;
        }
        const float conv_val = h0 * sw0 + h1 * sw1 + h2 * sw2 + xt * sw3;
        cur_out[(size_t)t * n_embd + r] = (xt + conv_val) * inv_scale + ffn_inp[(size_t)t * n_embd + r];
        h0 = h1; h1 = h2; h2 = xt;
    }
    hist_ch[0] = h0;
    hist_ch[1] = h1;
    hist_ch[2] = h2;
}

__global__ void __launch_bounds__(256) route_group_limbs_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int n_expert,
    int topk,
    int T,
    float route_scale,
    const uint64_t* __restrict__ gu_tab,
    const uint64_t* __restrict__ dn_tab,
    const float* __restrict__ sg_t,
    const float* __restrict__ dn_sc_t,
    const float* __restrict__ x_f32,
    int n_embd,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights,
    uint16_t* __restrict__ x_limbs,
    uint64_t* __restrict__ grp_gu_ptrs,
    uint64_t* __restrict__ grp_dn_ptrs,
    int32_t* __restrict__ grp_n,
    int32_t* __restrict__ grp_tok,
    int32_t* __restrict__ grp_row,
    float* __restrict__ slot_sg,
    float* __restrict__ slot_su,
    float* __restrict__ slot_wt
) {
    if (blockIdx.x > 0) {
        const int blocks_per_tok = (n_embd + 255) >> 8;
        const int b = (int)blockIdx.x - 1;
        const int tk = b / blocks_per_tok;
        const int c = (b - tk * blocks_per_tok) * 256 + (int)threadIdx.x;
        if (tk < T && c < n_embd) {
            const float v = x_f32[(size_t)tk * n_embd + c];
            const uint16_t hi = f32_to_fp16_bits(v);
            const uint16_t lo = f32_to_fp16_bits(v - fp16_bits_to_f32(hi));
            x_limbs[((size_t)tk * 2 + 0) * n_embd + c] = hi;
            x_limbs[((size_t)tk * 2 + 1) * n_embd + c] = lo;
        }
        return;
    }

    __shared__ int s_ex[64], s_lead[64], s_gid[64];
    __shared__ float s_wt[64];
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int S = T * topk;

    if (warp < T) {
        const float* l = logits + (size_t)warp * n_expert;
        float sig[4], val[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int e = lane + q * 32;
            if (e < n_expert) {
                const float sl = 1.0f / (1.0f + expf(-__ldg(l + e)));
                sig[q] = sl;
                val[q] = sl + (bias ? __ldg(bias + e) : 0.0f);
            } else {
                sig[q] = 0.0f;
                val[q] = -INFINITY;
            }
        }
        int ch[8];
        float ch_sig[8];
        for (int k = 0; k < topk; ++k) {
            float best = -INFINITY;
            float bsig = 0.0f;
            int idx = 0x7fffffff;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int e = lane + q * 32;
                if (val[q] > best || (val[q] == best && e < idx)) {
                    best = val[q];
                    bsig = sig[q];
                    idx = e;
                }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                const float ob = __shfl_xor_sync(0xffffffffu, best, o);
                const float os = __shfl_xor_sync(0xffffffffu, bsig, o);
                const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
                if (ob > best || (ob == best && oi < idx)) {
                    best = ob;
                    bsig = os;
                    idx = oi;
                }
            }
            ch[k] = idx;
            ch_sig[k] = bsig;
            if (idx < n_expert && (idx & 31) == lane) val[idx >> 5] = -INFINITY;
        }
        if (lane == 0) {
            for (int a = 1; a < topk; ++a) {
                const int ex_k = ch[a];
                const float sg_k = ch_sig[a];
                int b = a - 1;
                while (b >= 0 && ch[b] > ex_k) {
                    ch[b + 1] = ch[b];
                    ch_sig[b + 1] = ch_sig[b];
                    --b;
                }
                ch[b + 1] = ex_k;
                ch_sig[b + 1] = sg_k;
            }
            float sum = 0.0f;
            for (int k = 0; k < topk; ++k) sum += ch_sig[k];
            for (int k = 0; k < topk; ++k) {
                const int slot = warp * topk + k;
                const float wk = ch_sig[k] / (sum + 1e-20f) * route_scale;
                out_ids[slot] = ch[k];
                out_weights[slot] = wk;
                s_ex[slot] = ch[k];
                s_wt[slot] = wk;
            }
        }
    }
    __syncthreads();

    if (T == 1) {
        if (tid < topk) {
            const int e = s_ex[tid];
            grp_gu_ptrs[tid] = gu_tab[e];
            grp_dn_ptrs[tid] = dn_tab[e];
            grp_n[tid] = 1;
            grp_tok[tid * kGroupMax] = 0;
            grp_row[tid * kGroupMax] = tid;
            const float sc_gu = sg_t ? sg_t[e] : 1.0f;
            slot_sg[tid] = sc_gu;
            slot_su[tid] = sc_gu;
            slot_wt[tid] = s_wt[tid] * (dn_sc_t ? dn_sc_t[e] : 1.0f);
        }
        return;
    }

    const int i = tid;
    const int e = (i < S) ? s_ex[i] : -1;
    if (i < S) grp_n[i] = 0;
    int first = i, m = 0, cnt = 0;
    if (i < S) {
        for (int j = 0; j < S; ++j) {
            if (s_ex[j] == e) {
                if (j < first) first = j;
                if (j < i) ++m;
                ++cnt;
            }
        }
    }
    if (i < 64) s_lead[i] = (i < S && first == i);
    __syncthreads();
    if (i < 64 && s_lead[i]) {
        int gx = 0;
        for (int j = 0; j < i; ++j) gx += s_lead[j];
        s_gid[i] = gx;
    }
    __syncthreads();
    if (i >= S) return;
    const int g = s_gid[first];
    grp_tok[g * kGroupMax + m] = i / topk;
    grp_row[g * kGroupMax + m] = i;
    if (first == i) {
        grp_gu_ptrs[g] = gu_tab[e];
        grp_dn_ptrs[g] = dn_tab[e];
        grp_n[g] = cnt;
    }
    const float sc_gu = sg_t ? sg_t[e] : 1.0f;
    slot_sg[i] = sc_gu;
    slot_su[i] = sc_gu;
    slot_wt[i] = s_wt[i] * (dn_sc_t ? dn_sc_t[e] : 1.0f);
}

template <int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_mma_t1_gu_nvfp4_kernel(
    const uint64_t* __restrict__ blob,
    size_t off_up,
    int C,
    int F,
    const uint16_t* __restrict__ limbs,
    const float* __restrict__ sg,
    const float* __restrict__ su,
    float gcap,
    float ucap,
    uint16_t* __restrict__ h
) {
    __shared__ float2 part[SPLIT][8];
    const int slot = blockIdx.y;
    if (blob[slot] == 0ULL) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 8 + gr;
    const size_t rb = (size_t)(C / 64) * 36;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(blob[slot])) + (size_t)r * rb;
    const uint8_t* b = a + off_up;
    const bool x_ok = (gr < 2);
    const uint4* xp = reinterpret_cast<const uint4*>(limbs + (x_ok ? (size_t)gr * C : 0)) + 2 * t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = C / 64;
    NvMmaStep cur = nv_mma_step_load(a, b, split < nk ? split : 0, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvMmaStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_mma_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu);
        const uint32_t sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
        if (x_ok) {
            x0 = __ldg(xp + k * 8);
            x1 = __ldg(xp + k * 8 + 1);
        }
        const uint32_t xx[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1));
            const uint32_t qb = wb[m >> 1] >> (16 * (m & 1));
            const uint32_t af[4] = {
                nv_pair_h2(qa, sa), nv_pair_h2(qb, sb),
                nv_pair_h2(qa >> 4, sa), nv_pair_h2(qb >> 4, sb)
            };
            mma16816_f16(v, af, xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
    if (t == 0) part[split][gr] = make_float2(v[0] + v[1], v[2] + v[3]);
    __syncthreads();
    if (split != 0 || t != 0) return;
    float2 tot = part[0][gr];
    #pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        tot.x += part[s][gr].x;
        tot.y += part[s][gr].y;
    }
    const float gg = tot.x * (sg ? sg[slot] : 1.0f);
    const float uu = tot.y * (su ? su[slot] : 1.0f);
    const float g_cap = gcap * tanhf(gg / gcap);
    const float sig = 1.0f / (1.0f + expf(-gg));
    const float u_cap = (ucap > 0.0f) ? (ucap * tanhf(uu / ucap)) : uu;
    h[(size_t)slot * F + r] = f32_to_fp16_bits(g_cap * sig * u_cap);
}

template <int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_mma_t1_down_nvfp4_kernel(
    const uint64_t* __restrict__ blob,
    const float* __restrict__ wt,
    int C,
    int F,
    const uint16_t* __restrict__ h,
    float* __restrict__ y
) {
    __shared__ float2 part[SPLIT][8];
    const int slot = blockIdx.y;
    if (blob[slot] == 0ULL) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 16 + gr;
    const size_t rb = (size_t)(F / 64) * 36;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(blob[slot])) + (size_t)r * rb;
    const uint8_t* b = a + 8 * rb;
    const bool x_ok = (gr == 0);
    const uint4* xp = reinterpret_cast<const uint4*>(h + (size_t)slot * F) + 2 * t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = F / 64;
    NvMmaStep cur = nv_mma_step_load(a, b, split < nk ? split : 0, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvMmaStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_mma_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu);
        const uint32_t sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        uint4 x0 = make_uint4(0, 0, 0, 0), x1 = x0;
        if (x_ok) {
            x0 = __ldg(xp + k * 8);
            x1 = __ldg(xp + k * 8 + 1);
        }
        const uint32_t xx[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1));
            const uint32_t qb = wb[m >> 1] >> (16 * (m & 1));
            const uint32_t af[4] = {
                nv_pair_h2(qa, sa), nv_pair_h2(qb, sb),
                nv_pair_h2(qa >> 4, sa), nv_pair_h2(qb >> 4, sb)
            };
            mma16816_f16(v, af, xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
    if (t == 0) part[split][gr] = make_float2(v[0], v[2]);
    __syncthreads();
    if (split != 0 || t != 0) return;
    float2 tot = part[0][gr];
    #pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        tot.x += part[s][gr].x;
        tot.y += part[s][gr].y;
    }
    const float w0 = wt[slot];
    y[(size_t)slot * C + r] = tot.x * w0;
    y[(size_t)slot * C + r + 8] = tot.y * w0;
}

template <int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_mma_t1_gu_q8_kernel(
    const uint64_t* __restrict__ blob,
    size_t off_up,
    int C,
    int F,
    const uint16_t* __restrict__ limbs,
    const float* __restrict__ sg,
    const float* __restrict__ su,
    float gcap,
    float ucap,
    uint16_t* __restrict__ h
) {
    __shared__ float2 part[SPLIT][8];
    const int slot = blockIdx.y;
    if (blob[slot] == 0ULL) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 8 + gr;
    const size_t rb = (size_t)(C / 32) * 34;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(blob[slot])) + (size_t)r * rb;
    const uint8_t* b = a + off_up;
    const bool x_ok = (gr < 2);
    const uint4* xp = reinterpret_cast<const uint4*>(limbs + (x_ok ? (size_t)gr * C : 0)) + t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = C / 32;
    Q8ExpStep cur = q8_exp_step_load(a, b, split < nk ? split : 0, t);
    for (int k = split; k < nk; k += SPLIT) {
        Q8ExpStep nxt = cur;
        if (k + SPLIT < nk) nxt = q8_exp_step_load(a, b, k + SPLIT, t);
        const float sa = fp16_bits_to_f32((uint16_t)cur.sa);
        const float sb = fp16_bits_to_f32((uint16_t)cur.sb);
        const uint32_t af0[4] = {q8_pair(cur.a[0]), q8_pair(cur.b[0]), q8_pair(cur.a[1]), q8_pair(cur.b[1])};
        const uint32_t af1[4] = {q8_pair(cur.a[2]), q8_pair(cur.b[2]), q8_pair(cur.a[3]), q8_pair(cur.b[3])};
        uint4 x = make_uint4(0, 0, 0, 0);
        if (x_ok) x = __ldg(xp + k * 4);
        float bs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma16816_f16(bs, af0, x.x, x.y);
        mma16816_f16(bs, af1, x.z, x.w);
        v[0] += sa * bs[0]; v[1] += sa * bs[1];
        v[2] += sb * bs[2]; v[3] += sb * bs[3];
        cur = nxt;
    }
    if (t == 0) part[split][gr] = make_float2(v[0] + v[1], v[2] + v[3]);
    __syncthreads();
    if (split != 0 || t != 0) return;
    float2 tot = part[0][gr];
    #pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        tot.x += part[s][gr].x;
        tot.y += part[s][gr].y;
    }
    const float gg = tot.x * (sg ? sg[slot] : 1.0f);
    const float uu = tot.y * (su ? su[slot] : 1.0f);
    const float g_cap = gcap * tanhf(gg / gcap);
    const float sig = 1.0f / (1.0f + expf(-gg));
    const float u_cap = (ucap > 0.0f) ? (ucap * tanhf(uu / ucap)) : uu;
    h[(size_t)slot * F + r] = f32_to_fp16_bits(g_cap * sig * u_cap);
}

template <int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_mma_t1_down_q8_kernel(
    const uint64_t* __restrict__ blob,
    const float* __restrict__ wt,
    int C,
    int F,
    const uint16_t* __restrict__ h,
    float* __restrict__ y
) {
    __shared__ float2 part[SPLIT][8];
    const int slot = blockIdx.y;
    if (blob[slot] == 0ULL) return;
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, gr = lane >> 2, t = lane & 3;
    const int r = blockIdx.x * 16 + gr;
    const size_t rb = (size_t)(F / 32) * 34;
    const uint8_t* a = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(blob[slot])) + (size_t)r * rb;
    const uint8_t* b = a + 8 * rb;
    const bool x_ok = (gr == 0);
    const uint4* xp = reinterpret_cast<const uint4*>(h + (size_t)slot * F) + t;
    float v[2] = {0.0f, 0.0f};
    const int nk = F / 32;
    Q8ExpStep cur = q8_exp_step_load(a, b, split < nk ? split : 0, t);
    for (int k = split; k < nk; k += SPLIT) {
        Q8ExpStep nxt = cur;
        if (k + SPLIT < nk) nxt = q8_exp_step_load(a, b, k + SPLIT, t);
        const float sa = fp16_bits_to_f32((uint16_t)cur.sa);
        const float sb = fp16_bits_to_f32((uint16_t)cur.sb);
        const uint32_t af0[4] = {q8_pair(cur.a[0]), q8_pair(cur.b[0]), q8_pair(cur.a[1]), q8_pair(cur.b[1])};
        const uint32_t af1[4] = {q8_pair(cur.a[2]), q8_pair(cur.b[2]), q8_pair(cur.a[3]), q8_pair(cur.b[3])};
        uint4 x = make_uint4(0, 0, 0, 0);
        if (x_ok) x = __ldg(xp + k * 4);
        float bs[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        mma16816_f16(bs, af0, x.x, x.y);
        mma16816_f16(bs, af1, x.z, x.w);
        v[0] += sa * bs[0];
        v[1] += sb * bs[2];
        cur = nxt;
    }
    if (t == 0) part[split][gr] = make_float2(v[0], v[1]);
    __syncthreads();
    if (split != 0 || t != 0) return;
    float2 tot = part[0][gr];
    #pragma unroll
    for (int s = 1; s < SPLIT; ++s) {
        tot.x += part[s][gr].x;
        tot.y += part[s][gr].y;
    }
    const float w0 = wt[slot];
    y[(size_t)slot * C + r] = tot.x * w0;
    y[(size_t)slot * C + r + 8] = tot.y * w0;
}

} // namespace

// -----------------------------------------------------------------------------
// External C Launchers
// -----------------------------------------------------------------------------

extern "C" {

void launch_f32_to_f16_fast(const float* x, uint16_t* y, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    const size_t n4 = (n + 3) >> 2;
    f32_to_f16_vec_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, stream>>>(x, y, n);
}

void launch_rmsnorm_f16_multi(
    const float* x0, const float* x1, const float* x2,
    const float* w0, const float* w1, const float* w2,
    float* y32_0, float* y32_1, float* y32_2,
    uint16_t* y16_0, uint16_t* y16_1, uint16_t* y16_2,
    int n_jobs, int dim, int M, float eps, cudaStream_t stream
) {
    MultiRmsF16Args a{};
    a.x[0] = x0; a.x[1] = x1; a.x[2] = x2;
    a.w[0] = w0; a.w[1] = w1; a.w[2] = w2;
    a.y32[0] = y32_0; a.y32[1] = y32_1; a.y32[2] = y32_2;
    a.y16[0] = y16_0; a.y16[1] = y16_1; a.y16[2] = y16_2;
    dim3 grid(M, n_jobs);
    rmsnorm_f16_multi_kernel<<<grid, 256, 0, stream>>>(a, dim, M, eps);
}

void launch_attn_res_mix_rms_f16(
    const float* const* state_ptrs, int n_states, const float* queries,
    const float* norm_w0, const float* norm_w1, const float* norm_w2,
    float* norm_y0, float* norm_y1, float* norm_y2,
    uint16_t* norm_y16_0, uint16_t* norm_y16_1, uint16_t* norm_y16_2,
    int n_embd, int n_q, int M, float res_eps, float rms_eps, cudaStream_t stream
) {
    attn_res_mix_rms_f16_kernel<<<M, 256, 0, stream>>>(
        state_ptrs, n_states, queries,
        norm_w0, norm_w1, norm_w2,
        norm_y0, norm_y1, norm_y2,
        norm_y16_0, norm_y16_1, norm_y16_2,
        n_embd, n_q, M, res_eps, rms_eps
    );
}

void launch_q8_mma_tile(
    const int8_t* q, const uint16_t* d, int N, int K,
    const uint16_t* x16, int x_ld, int T,
    float* out, int y_ld, cudaStream_t stream
) {
    if (N <= 0 || T <= 0) return;
    dim3 grid((unsigned)((N + 15) / 16), (unsigned)((T + 7) / 8));
    q8_mma_tile_kernel<4, 1><<<grid, 128, 0, stream>>>(q, d, N, K, x16, x_ld, T, out, y_ld);
}

void launch_q8_mma_qkvrg(
    const int8_t* q0, const uint16_t* d0, int n0, const uint16_t* x0, float* y0,
    const int8_t* q1, const uint16_t* d1, int n1, const uint16_t* x1, float* y1,
    const int8_t* q2, const uint16_t* d2, int n2, const uint16_t* x2, float* y2,
    const int8_t* q3, const uint16_t* d3, int n3, const uint16_t* x3, float* y3,
    const int8_t* q4, const uint16_t* d4, int n4, const uint16_t* x4, float* y4,
    int K, int T, cudaStream_t stream
) {
    Q8QkvrgSegs s{};
    s.q[0] = q0; s.d[0] = d0; s.x[0] = x0; s.y[0] = y0; s.n[0] = n0;
    s.q[1] = q1; s.d[1] = d1; s.x[1] = x1; s.y[1] = y1; s.n[1] = n1;
    s.q[2] = q2; s.d[2] = d2; s.x[2] = x2; s.y[2] = y2; s.n[2] = n2;
    s.q[3] = q3; s.d[3] = d3; s.x[3] = x3; s.y[3] = y3; s.n[3] = n3;
    s.q[4] = q4; s.d[4] = d4; s.x[4] = x4; s.y[4] = y4; s.n[4] = n4;
    s.t_off[0] = 0;
    int total_tiles = 0;
    for (int i = 0; i < 5; ++i) {
        s.t_off[i] = total_tiles;
        total_tiles += (s.n[i] + 15) / 16;
    }
    if (total_tiles <= 0 || T <= 0) return;
    dim3 grid((unsigned)total_tiles, (unsigned)((T + 7) / 8));
    q8_mma_qkvrg_kernel<4, 2><<<grid, 128, 0, stream>>>(s, K, T);
}

void launch_q8_mma_argmax_batch(
    const int8_t* q, const uint16_t* d, int N, int K,
    const uint16_t* x16, int x_ld, int T,
    float* part_vals, int32_t* part_ids, int32_t* out_ids,
    cudaStream_t stream
) {
    if (N <= 0 || T <= 0) return;
    const int n_tiles = (N + 15) / 16;
    const int ctas = (n_tiles < 328) ? n_tiles : 328;
    q8_mma_argmax_part_kernel<4, 2><<<(unsigned)ctas, 128, 0, stream>>>(
        q, d, N, K, x16, x_ld, T, part_vals, part_ids
    );
    dim3 fg(1, (unsigned)T);
    q8_mma_argmax_final_kernel<<<fg, 256, 0, stream>>>(part_vals, part_ids, ctas, T, out_ids);
}

void launch_short_conv_rec_f32(
    const float* cur, const float* static_w, const float* dyn_proj, const float* add_res,
    float* history, float* out,
    float* rec_old_hist, float* rec_x,
    int n_sconv_channels, int slot_idx,
    int n_channels, int n_heads, int M, int conv_offset, int n_conv,
    cudaStream_t stream
) {
    const int threads = 256;
    short_conv_rec_kernel<<<(n_channels + threads - 1) / threads, threads, 0, stream>>>(
        cur, static_w, dyn_proj, add_res, history, out,
        rec_old_hist, rec_x, n_sconv_channels, slot_idx,
        n_channels, n_heads, M, conv_offset, n_conv
    );
}

void launch_short_conv_kv_pair_rec_f32(
    const float* k_in, const float* k_static_w, const float* k_dyn_proj, float* k_out,
    const float* v_in, const float* v_static_w, const float* v_dyn_proj, float* v_out,
    float* history, float* rec_old_hist, float* rec_x,
    int n_sconv_channels, int slot_idx, int kv_dim, int n_heads, int M, int n_conv,
    cudaStream_t stream
) {
    if (kv_dim <= 0 || M <= 0) return;
    dim3 grid((unsigned)((kv_dim + 127) / 128), 2);
    short_conv_kv_pair_rec_kernel<<<grid, 128, 0, stream>>>(
        k_in, k_static_w, k_dyn_proj, k_out,
        v_in, v_static_w, v_dyn_proj, v_out,
        history, rec_old_hist, rec_x,
        n_sconv_channels, slot_idx, kv_dim, n_heads, M, n_conv
    );
}

void launch_mv_q8p_colsx_qkvrg(
    const int8_t* q0, const uint16_t* d0, int n0, const float* x0, float* y0,
    const int8_t* q1, const uint16_t* d1, int n1, const float* x1, float* y1,
    const int8_t* q2, const uint16_t* d2, int n2, const float* x2, float* y2,
    const int8_t* q3, const uint16_t* d3, int n3, const float* x3, float* y3,
    const int8_t* q4, const uint16_t* d4, int n4, const float* x4, float* y4,
    int K, int T, cudaStream_t stream
) {
    Q8QkvrgF32Segs s{};
    s.q[0] = q0; s.d[0] = d0; s.x[0] = x0; s.y[0] = y0; s.n[0] = n0;
    s.q[1] = q1; s.d[1] = d1; s.x[1] = x1; s.y[1] = y1; s.n[1] = n1;
    s.q[2] = q2; s.d[2] = d2; s.x[2] = x2; s.y[2] = y2; s.n[2] = n2;
    s.q[3] = q3; s.d[3] = d3; s.x[3] = x3; s.y[3] = y3; s.n[3] = n3;
    s.q[4] = q4; s.d[4] = d4; s.x[4] = x4; s.y[4] = y4; s.n[4] = n4;
    if (T <= 0 || K <= 0) return;
    if (K == 2304 && T <= 8) {
        const int rows_per_cta = (T <= 6) ? (4 * 4) : (4 * 2);
        int total_ctas = 0;
        for (int i = 0; i < 5; ++i) {
            s.cta_off[i] = total_ctas;
            total_ctas += (s.n[i] + rows_per_cta - 1) / rows_per_cta;
        }
        if (total_ctas <= 0) return;
        switch (T) {
            case 1: mv_q8p_colsx_qkvrg_kernel<4, 4, 1, 18, 4><<<(unsigned)total_ctas, 128, 0, stream>>>(s, K); break;
            case 2: mv_q8p_colsx_qkvrg_kernel<4, 4, 2, 18, 4><<<(unsigned)total_ctas, 128, 0, stream>>>(s, K); break;
            case 3: mv_q8p_colsx_qkvrg_kernel<4, 4, 3, 18, 4><<<(unsigned)total_ctas, 128, 0, stream>>>(s, K); break;
            case 4: mv_q8p_colsx_qkvrg_kernel<4, 4, 4, 18, 4><<<(unsigned)total_ctas, 128, 0, stream>>>(s, K); break;
            case 5: mv_q8p_colsx_qkvrg_kernel<4, 4, 5, 18, 2><<<(unsigned)total_ctas, 128, 0, stream>>>(s, K); break;
            case 6: mv_q8p_colsx_qkvrg_kernel<4, 4, 6, 18, 2><<<(unsigned)total_ctas, 128, 0, stream>>>(s, K); break;
            case 7: mv_q8p_colsx_qkvrg_kernel<4, 2, 7, 18, 3><<<(unsigned)total_ctas, 128, 0, stream>>>(s, K); break;
            case 8: mv_q8p_colsx_qkvrg_kernel<4, 2, 8, 18, 3><<<(unsigned)total_ctas, 128, 0, stream>>>(s, K); break;
        }
        return;
    }
    int total_ctas = 0;
    for (int i = 0; i < 5; ++i) {
        s.cta_off[i] = total_ctas;
        total_ctas += (s.n[i] + 7) / 8;
    }
    if (total_ctas <= 0) return;
    mv_q8p_cols_qkvrg_generic_kernel<<<(unsigned)total_ctas, 128, 0, stream>>>(s, K, T);
}

void launch_mv_q8p_colsx(
    const int8_t* q, const uint16_t* d, int N, int K,
    const float* x, int x_ld, int T,
    float* y, int y_ld, cudaStream_t stream
) {
    if (N <= 0 || T <= 0 || K <= 0) return;
    if (K == 3072 && T <= 8) {
        const int rows_per_cta = (T <= 6) ? (4 * 4) : (4 * 2);
        const unsigned g = (unsigned)((N + rows_per_cta - 1) / rows_per_cta);
        switch (T) {
            case 1: mv_q8p_colsx_kernel<4, 4, 1, 24, 4><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 2: mv_q8p_colsx_kernel<4, 4, 2, 24, 4><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 3: mv_q8p_colsx_kernel<4, 4, 3, 24, 4><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 4: mv_q8p_colsx_kernel<4, 4, 4, 24, 4><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 5: mv_q8p_colsx_kernel<4, 4, 5, 24, 2><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 6: mv_q8p_colsx_kernel<4, 4, 6, 24, 2><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 7: mv_q8p_colsx_kernel<4, 2, 7, 24, 3><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 8: mv_q8p_colsx_kernel<4, 2, 8, 24, 3><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
        }
        return;
    }
    if (K == 2304 && T <= 8) {
        const int rows_per_cta = (T <= 6) ? (4 * 4) : (4 * 2);
        const unsigned g = (unsigned)((N + rows_per_cta - 1) / rows_per_cta);
        switch (T) {
            case 1: mv_q8p_colsx_kernel<4, 4, 1, 18, 4><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 2: mv_q8p_colsx_kernel<4, 4, 2, 18, 4><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 3: mv_q8p_colsx_kernel<4, 4, 3, 18, 4><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 4: mv_q8p_colsx_kernel<4, 4, 4, 18, 4><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 5: mv_q8p_colsx_kernel<4, 4, 5, 18, 2><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 6: mv_q8p_colsx_kernel<4, 4, 6, 18, 2><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 7: mv_q8p_colsx_kernel<4, 2, 7, 18, 3><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
            case 8: mv_q8p_colsx_kernel<4, 2, 8, 18, 3><<<g, 128, 0, stream>>>(q, d, N, K, x, x_ld, y, y_ld); break;
        }
        return;
    }
    mv_q8p_cols_generic_kernel<<<(unsigned)((N + 7) / 8), 128, 0, stream>>>(q, d, N, K, x, x_ld, T, y, y_ld);
}

void launch_mv_q8p_colsx_argmax(
    const int8_t* q, const uint16_t* d, int N, int K,
    const float* x, int x_ld, int T,
    float* part_vals, int32_t* part_ids, int32_t* out_ids,
    cudaStream_t stream
) {
    if (N <= 0 || T <= 0 || K <= 0) return;
    const int max_ctas = (N + 7) / 8;
    const int ctas = (max_ctas < 656) ? max_ctas : 656;
    if (K == 2304 && T <= 8) {
        switch (T) {
            case 1: mv_q8p_colsx_argmax_part_kernel<4, 2, 1, 18, 3><<<(unsigned)ctas, 128, 0, stream>>>(q, d, N, K, x, x_ld, part_vals, part_ids); break;
            case 2: mv_q8p_colsx_argmax_part_kernel<4, 2, 2, 18, 3><<<(unsigned)ctas, 128, 0, stream>>>(q, d, N, K, x, x_ld, part_vals, part_ids); break;
            case 3: mv_q8p_colsx_argmax_part_kernel<4, 2, 3, 18, 3><<<(unsigned)ctas, 128, 0, stream>>>(q, d, N, K, x, x_ld, part_vals, part_ids); break;
            case 4: mv_q8p_colsx_argmax_part_kernel<4, 2, 4, 18, 3><<<(unsigned)ctas, 128, 0, stream>>>(q, d, N, K, x, x_ld, part_vals, part_ids); break;
            case 5: mv_q8p_colsx_argmax_part_kernel<4, 2, 5, 18, 3><<<(unsigned)ctas, 128, 0, stream>>>(q, d, N, K, x, x_ld, part_vals, part_ids); break;
            case 6: mv_q8p_colsx_argmax_part_kernel<4, 2, 6, 18, 3><<<(unsigned)ctas, 128, 0, stream>>>(q, d, N, K, x, x_ld, part_vals, part_ids); break;
            case 7: mv_q8p_colsx_argmax_part_kernel<4, 2, 7, 18, 3><<<(unsigned)ctas, 128, 0, stream>>>(q, d, N, K, x, x_ld, part_vals, part_ids); break;
            case 8: mv_q8p_colsx_argmax_part_kernel<4, 2, 8, 18, 3><<<(unsigned)ctas, 128, 0, stream>>>(q, d, N, K, x, x_ld, part_vals, part_ids); break;
        }
    } else {
        mv_q8p_cols_argmax_generic_kernel<<<(unsigned)ctas, 128, 0, stream>>>(q, d, N, K, x, x_ld, T, part_vals, part_ids);
    }
    dim3 fg(1, (unsigned)T);
    q8_mma_argmax_final_kernel<<<fg, 256, 0, stream>>>(part_vals, part_ids, ctas, T, out_ids);
}

void launch_mv_f16_batch(
    const uint16_t* w_f16,
    const float* x_f32,
    float* out_f32,
    int n,
    int k,
    int t,
    cudaStream_t stream
) {
    if (t <= 0 || n <= 0) return;
    switch (t) {
        case 1: mv_f16_batch_t_kernel<1><<<(unsigned)n, 32, 0, stream>>>(w_f16, x_f32, out_f32, n, k); break;
        case 2: mv_f16_batch_t_kernel<2><<<(unsigned)n, 32, 0, stream>>>(w_f16, x_f32, out_f32, n, k); break;
        case 3: mv_f16_batch_t_kernel<3><<<(unsigned)n, 32, 0, stream>>>(w_f16, x_f32, out_f32, n, k); break;
        case 4: mv_f16_batch_t_kernel<4><<<(unsigned)n, 32, 0, stream>>>(w_f16, x_f32, out_f32, n, k); break;
        case 5: mv_f16_batch_t_kernel<5><<<(unsigned)n, 32, 0, stream>>>(w_f16, x_f32, out_f32, n, k); break;
        case 6: mv_f16_batch_t_kernel<6><<<(unsigned)n, 32, 0, stream>>>(w_f16, x_f32, out_f32, n, k); break;
        case 7: mv_f16_batch_t_kernel<7><<<(unsigned)n, 32, 0, stream>>>(w_f16, x_f32, out_f32, n, k); break;
        case 8: mv_f16_batch_t_kernel<8><<<(unsigned)n, 32, 0, stream>>>(w_f16, x_f32, out_f32, n, k); break;
        default: mv_f16_batch_kernel<<<(unsigned)n, 32, 0, stream>>>(w_f16, x_f32, out_f32, n, k, t); break;
    }
}

void launch_mv_f16_pair_batch(
    const uint16_t* w0_f16, const float* x0_f32, float* out0_f32,
    const uint16_t* w1_f16, const float* x1_f32, float* out1_f32,
    int n,
    int k,
    int t,
    cudaStream_t stream
) {
    if (t <= 0 || n <= 0) return;
    dim3 grid((unsigned)n, 2);
    switch (t) {
        case 1: mv_f16_pair_batch_t_kernel<1><<<grid, 32, 0, stream>>>(w0_f16, x0_f32, out0_f32, w1_f16, x1_f32, out1_f32, n, k); break;
        case 2: mv_f16_pair_batch_t_kernel<2><<<grid, 32, 0, stream>>>(w0_f16, x0_f32, out0_f32, w1_f16, x1_f32, out1_f32, n, k); break;
        case 3: mv_f16_pair_batch_t_kernel<3><<<grid, 32, 0, stream>>>(w0_f16, x0_f32, out0_f32, w1_f16, x1_f32, out1_f32, n, k); break;
        case 4: mv_f16_pair_batch_t_kernel<4><<<grid, 32, 0, stream>>>(w0_f16, x0_f32, out0_f32, w1_f16, x1_f32, out1_f32, n, k); break;
        case 5: mv_f16_pair_batch_t_kernel<5><<<grid, 32, 0, stream>>>(w0_f16, x0_f32, out0_f32, w1_f16, x1_f32, out1_f32, n, k); break;
        case 6: mv_f16_pair_batch_t_kernel<6><<<grid, 32, 0, stream>>>(w0_f16, x0_f32, out0_f32, w1_f16, x1_f32, out1_f32, n, k); break;
        case 7: mv_f16_pair_batch_t_kernel<7><<<grid, 32, 0, stream>>>(w0_f16, x0_f32, out0_f32, w1_f16, x1_f32, out1_f32, n, k); break;
        case 8: mv_f16_pair_batch_t_kernel<8><<<grid, 32, 0, stream>>>(w0_f16, x0_f32, out0_f32, w1_f16, x1_f32, out1_f32, n, k); break;
        default: mv_f16_pair_batch_kernel<<<grid, 32, 0, stream>>>(w0_f16, x0_f32, out0_f32, w1_f16, x1_f32, out1_f32, n, k, t); break;
    }
}

void launch_conv_rollback_all(
    float* const* histories, const float* rec_old_hist, const float* rec_x,
    int n_slots, int n_sconv_channels, int accept_n, cudaStream_t stream
) {
    dim3 grid((unsigned)((n_sconv_channels + 255) / 256), (unsigned)n_slots);
    conv_rollback_all_kernel<<<grid, 256, 0, stream>>>(
        histories, rec_old_hist, rec_x, n_slots, n_sconv_channels, accept_n
    );
}

void launch_loop_gates_fast_f32(
    const float* x0, const float* x1, const float* x2,
    const float* m0, const float* m1, const float* m2,
    const float* d0, const float* d1, const float* d2,
    const float* f0, const float* f1, const float* f2,
    const float* g0, const float* g1, const float* g2,
    float* y0, float* y1, float* y2,
    float* scratch_sc,
    int n_gates, int M, int C, int R, float eps, cudaStream_t stream
) {
    FastLoopGateArgs a{};
    a.x[0] = x0; a.x[1] = x1; a.x[2] = x2;
    a.m[0] = m0; a.m[1] = m1; a.m[2] = m2;
    a.D[0] = d0; a.D[1] = d1; a.D[2] = d2;
    a.F[0] = f0; a.F[1] = f1; a.F[2] = f2;
    a.G[0] = g0; a.G[1] = g1; a.G[2] = g2;
    a.y[0] = y0; a.y[1] = y1; a.y[2] = y2;
    dim3 g_dots((unsigned)R, (unsigned)n_gates);
    if (C == 2304 && M <= 8) {
        loop_gate_dots_toks_fast_kernel<18><<<g_dots, 128, 0, stream>>>(a, M, C, R, scratch_sc);
    } else {
        loop_gate_dots_toks_kernel<<<g_dots, 128, 0, stream>>>(a, M, C, R, scratch_sc);
    }

    dim3 g_apply(
        (unsigned)((C + kFastGateCh - 1) / kFastGateCh),
        (unsigned)n_gates,
        (unsigned)((M + kFastGateWarps - 1) / kFastGateWarps)
    );
    const size_t smem = (size_t)R * (2 * kFastGatePad + kFastGateWarps) * sizeof(float);
    loop_gate_apply_fast_kernel<<<g_apply, kFastGateWarps * 32, smem, stream>>>(
        a, M, C, R, eps, scratch_sc
    );
}

void launch_prep_qkv_r_block(
    float* q_cur, uint16_t* q16, const float* q_clip,
    float* k_cur, void* k_cache,
    float* v_cur, void* v_cache,
    float* r_cur, const float* r_clip,
    const float* tau, const int32_t* pos,
    const float* sink_w, const uint16_t* sink_w16, const float* sink_x, float* sinks_out,
    int n_embd, int n_head, int n_head_kv, int head_dim, int d_rel,
    int M, int ring_cap, int kv_mode, float qk_clip_th, float rel_clip_th, float eps,
    cudaStream_t stream
) {
    const bool has_sink = (sink_w != nullptr || sink_w16 != nullptr);
    const int total_blocks = M * (2 * n_head + 2 * n_head_kv + (has_sink ? n_head : 0));
    prep_qkv_r_block_kernel<<<total_blocks, 32, 0, stream>>>(
        q_cur, q16, q_clip, k_cur, k_cache, v_cur, v_cache, r_cur, r_clip,
        tau, pos, sink_w, sink_w16, sink_x, sinks_out,
        n_embd, n_head, n_head_kv, head_dim, d_rel,
        M, ring_cap, kv_mode, qk_clip_th, rel_clip_th, eps
    );
}

void launch_attn_block_mma(
    const uint16_t* q16, const float* r_cur, const float* attn_rel_bias, const float* sinks,
    const float* attn_gate, const void* k_cache, const void* v_cache, const int32_t* pos_d,
    float* split_out, float* split_ml, float* out32, uint16_t* out16,
    int n_head, int n_head_kv, int d_rel, int rel_extent, int swa_window, int ring_cap,
    int kv_mode, int T, cudaStream_t stream
) {
    const int n_split = (swa_window > 0) ? 8 : 16;
    const int n_kv_groups = n_head_kv;
    const int kv_cap = (swa_window > 0) ? ring_cap : 0;
    const float qk_scale = rsqrtf((float)kFaD) * (kv_mode == 2 ? kFp8KInv : 1.0f);
    const float v_inv = (kv_mode == 2) ? kFp8VInv : 1.0f;
    const int NC = T * (n_head / n_head_kv);
    const int NT = (NC + 7) / 8; // 1 for T=1..2, 2 for T=3..5, 3 for T=6..8
    dim3 grid((unsigned)n_kv_groups, (unsigned)n_split);
    auto* ml = reinterpret_cast<float2*>(split_ml);

    if (kv_mode == 2) {
        const auto* kc = reinterpret_cast<const uint8_t*>(k_cache);
        const auto* vc = reinterpret_cast<const uint8_t*>(v_cache);
        if (swa_window == 0 && T >= 2) {
            if (NT == 1) {
                attn_block_mma_kernel<uint8_t, 1><<<grid, kWaNW * 32, 0, stream>>>(
                    q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                    r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, true, split_out, ml);
            } else if (NT == 2) {
                attn_block_mma_kernel<uint8_t, 2><<<grid, kWaNW * 32, 0, stream>>>(
                    q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                    r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, true, split_out, ml);
            } else {
                attn_block_mma_kernel<uint8_t, 3><<<grid, kWaNW * 32, 0, stream>>>(
                    q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                    r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, true, split_out, ml);
            }
            dim3 tgrid((unsigned)n_kv_groups, (unsigned)n_split, (unsigned)T);
            attn_decode_mma_toks_kernel<uint8_t><<<tgrid, kWaNW * 32, 0, stream>>>(
                q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, true, split_out, ml);
        } else {
            dim3 tgrid((unsigned)n_kv_groups, (unsigned)n_split, (unsigned)T);
            attn_decode_mma_toks_kernel<uint8_t><<<tgrid, kWaNW * 32, 0, stream>>>(
                q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, false, split_out, ml);
        }
    } else {
        const auto* kc = reinterpret_cast<const uint16_t*>(k_cache);
        const auto* vc = reinterpret_cast<const uint16_t*>(v_cache);
        if (swa_window == 0 && T >= 2) {
            if (NT == 1) {
                attn_block_mma_kernel<uint16_t, 1><<<grid, kWaNW * 32, 0, stream>>>(
                    q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                    r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, true, split_out, ml);
            } else if (NT == 2) {
                attn_block_mma_kernel<uint16_t, 2><<<grid, kWaNW * 32, 0, stream>>>(
                    q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                    r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, true, split_out, ml);
            } else {
                attn_block_mma_kernel<uint16_t, 3><<<grid, kWaNW * 32, 0, stream>>>(
                    q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                    r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, true, split_out, ml);
            }
            dim3 tgrid((unsigned)n_kv_groups, (unsigned)n_split, (unsigned)T);
            attn_decode_mma_toks_kernel<uint16_t><<<tgrid, kWaNW * 32, 0, stream>>>(
                q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, true, split_out, ml);
        } else {
            dim3 tgrid((unsigned)n_kv_groups, (unsigned)n_split, (unsigned)T);
            attn_decode_mma_toks_kernel<uint16_t><<<tgrid, kWaNW * 32, 0, stream>>>(
                q16, kc, vc, kv_cap, n_head, n_head_kv, pos_d, T, qk_scale,
                r_cur, n_head * d_rel, attn_rel_bias, d_rel, rel_extent, swa_window, n_split, false, split_out, ml);
        }
    }

    const size_t smem = (size_t)kFaD * sizeof(float) + (size_t)n_split * kWaNW * sizeof(float);
    dim3 cg((unsigned)n_head, (unsigned)T);
    attn_block_combine_kernel<<<cg, kFaD, smem, stream>>>(
        split_out, ml, n_split, n_head, T, sinks, v_inv, pos_d, swa_window,
        attn_gate, n_head * kFaD, out32, out16
    );
}

__global__ void __launch_bounds__(128) router_f32_batch_kernel(
    const float* __restrict__ w, // [n_expert, n_embd]
    const float* __restrict__ x, // [T, n_embd]
    float* __restrict__ out,     // [T, n_expert]
    int n_expert,
    int n_embd,
    int T
) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row = (int)blockIdx.x * 4 + warp;
    if (row >= n_expert) return;

    const float4* wr4 = reinterpret_cast<const float4*>(w + (size_t)row * n_embd);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int k4_end = n_embd >> 2;
    const int x_stride4 = n_embd >> 2;

    float acc[8] = {};
    for (int k4 = lane; k4 < k4_end; k4 += 32) {
        const float4 wv = __ldg(wr4 + k4);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t < T) {
                const float4 xv = __ldg(x4 + (size_t)t * x_stride4 + k4);
                acc[t] = fmaf(wv.x, xv.x, acc[t]);
                acc[t] = fmaf(wv.y, xv.y, acc[t]);
                acc[t] = fmaf(wv.z, xv.z, acc[t]);
                acc[t] = fmaf(wv.w, xv.w, acc[t]);
            }
        }
    }

    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < T) {
            const float sum = warp_reduce_sum(acc[t]);
            if (lane == 0) out[(size_t)t * n_expert + row] = sum;
        }
    }
}

void launch_router_f32_batch(
    const float* w_f32,
    const float* x_f32,
    float* out_f32,
    int n_expert,
    int n_embd,
    int T,
    cudaStream_t stream
) {
    if (T <= 0 || n_expert <= 0) return;
    router_f32_batch_kernel<<<(unsigned)((n_expert + 3) / 4), 128, 0, stream>>>(
        w_f32, x_f32, out_f32, n_expert, n_embd, T
    );
}

void launch_moe_block_fast(
    const uint64_t* gate_up_ptrs,
    const uint64_t* down_ptrs,
    const float* gate_up_scales,
    const float* down_scales,
    int32_t w_type,
    const float* x_f32,
    const int32_t* expert_ids,
    const float* expert_weights,
    uint16_t* x_limbs,
    uint16_t* inter_f16,
    float* slot_out,
    uint64_t* grp_gu_ptrs,
    uint64_t* grp_dn_ptrs,
    int32_t* grp_n,
    int32_t* grp_tok,
    int32_t* grp_row,
    float* slot_sg,
    float* slot_su,
    float* slot_wt,
    const float* ffn_post_conv_w,
    float* conv_history,
    const float* ffn_inp,
    float* cur_out,
    float* rec_old_hist,
    float* rec_x,
    int n_sconv_channels,
    int slot_idx,
    int conv_offset,
    int n_conv,
    int n_embd,
    int n_ff,
    int topk,
    int T,
    float situ_beta,
    float situ_linear_beta,
    cudaStream_t stream
) {
    const int S = T * topk;
    group_route_kernel<<<1, 64, 0, stream>>>(
        expert_ids, expert_weights, S, topk, gate_up_ptrs, down_ptrs,
        gate_up_scales, down_scales, grp_gu_ptrs, grp_dn_ptrs,
        grp_n, grp_tok, grp_row, slot_sg, slot_su, slot_wt
    );
    dim3 g_limbs((unsigned)((n_embd + 255) / 256), (unsigned)T);
    f16_limbs_rows_kernel<<<g_limbs, 256, 0, stream>>>(x_f32, T, n_embd, x_limbs);

    const size_t gu_row_bytes = (w_type == GGML_TYPE_NVFP4) ? ((size_t)(n_embd / 64) * 36) : ((size_t)(n_embd / 32) * 34);
    const size_t off_up = (size_t)n_ff * gu_row_bytes;
    dim3 g_gu((unsigned)(n_ff / 8), (unsigned)S);
    dim3 g_dn((unsigned)(n_embd / 16), (unsigned)S);

    if (w_type == GGML_TYPE_NVFP4) {
        if (T <= 4) {
            experts_block_gu_nvfp4_kernel<8, 1><<<g_gu, 256, 0, stream>>>(
                grp_gu_ptrs, grp_n, grp_tok, grp_row, off_up, n_embd, n_ff,
                x_limbs, slot_sg, slot_su, situ_beta, situ_linear_beta, inter_f16
            );
        } else {
            experts_block_gu_nvfp4_kernel<8, 2><<<g_gu, 256, 0, stream>>>(
                grp_gu_ptrs, grp_n, grp_tok, grp_row, off_up, n_embd, n_ff,
                x_limbs, slot_sg, slot_su, situ_beta, situ_linear_beta, inter_f16
            );
        }
        experts_block_down_nvfp4_kernel<4><<<g_dn, 128, 0, stream>>>(
            grp_dn_ptrs, grp_n, grp_row, slot_wt, n_embd, n_ff, inter_f16, slot_out
        );
    } else {
        if (T <= 4) {
            experts_block_gu_q8_kernel<8, 1><<<g_gu, 256, 0, stream>>>(
                grp_gu_ptrs, grp_n, grp_tok, grp_row, off_up, n_embd, n_ff,
                x_limbs, slot_sg, slot_su, situ_beta, situ_linear_beta, inter_f16
            );
        } else {
            experts_block_gu_q8_kernel<8, 2><<<g_gu, 256, 0, stream>>>(
                grp_gu_ptrs, grp_n, grp_tok, grp_row, off_up, n_embd, n_ff,
                x_limbs, slot_sg, slot_su, situ_beta, situ_linear_beta, inter_f16
            );
        }
        experts_block_down_q8_kernel<4><<<g_dn, 128, 0, stream>>>(
            grp_dn_ptrs, grp_n, grp_row, slot_wt, n_embd, n_ff, inter_f16, slot_out
        );
    }

    moe_combine_conv_add_block_kernel<<<(unsigned)((n_embd + 63) / 64), 64, 0, stream>>>(
        slot_out, ffn_post_conv_w, conv_history, ffn_inp, cur_out,
        rec_old_hist, rec_x, n_sconv_channels, slot_idx,
        n_embd, topk, T, conv_offset, n_conv
    );
}

void launch_route_and_moe_block_fast(
    const float* router_logits,
    const float* router_bias,
    int n_expert,
    float route_scale,
    const uint64_t* gate_up_ptrs,
    const uint64_t* down_ptrs,
    const float* gate_up_scales,
    const float* down_scales,
    int32_t w_type,
    const float* x_f32,
    int32_t* expert_ids,
    float* expert_weights,
    uint16_t* x_limbs,
    uint16_t* inter_f16,
    float* slot_out,
    uint64_t* grp_gu_ptrs,
    uint64_t* grp_dn_ptrs,
    int32_t* grp_n,
    int32_t* grp_tok,
    int32_t* grp_row,
    float* slot_sg,
    float* slot_su,
    float* slot_wt,
    const float* ffn_post_conv_w,
    float* conv_history,
    const float* ffn_inp,
    float* cur_out,
    float* rec_old_hist,
    float* rec_x,
    int n_sconv_channels,
    int slot_idx,
    int conv_offset,
    int n_conv,
    int n_embd,
    int n_ff,
    int topk,
    int T,
    float situ_beta,
    float situ_linear_beta,
    cudaStream_t stream
) {
    const int blocks_per_tok = (n_embd + 255) >> 8;
    const unsigned rg_blocks = (unsigned)(1 + T * blocks_per_tok);
    route_group_limbs_kernel<<<rg_blocks, 256, 0, stream>>>(
        router_logits, router_bias, n_expert, topk, T, route_scale,
        gate_up_ptrs, down_ptrs, gate_up_scales, down_scales, x_f32, n_embd,
        expert_ids, expert_weights, x_limbs,
        grp_gu_ptrs, grp_dn_ptrs, grp_n, grp_tok, grp_row, slot_sg, slot_su, slot_wt
    );

    const size_t gu_row_bytes = (w_type == GGML_TYPE_NVFP4) ? ((size_t)(n_embd / 64) * 36) : ((size_t)(n_embd / 32) * 34);
    const size_t off_up = (size_t)n_ff * gu_row_bytes;

    if (T == 1) {
        dim3 g_gu((unsigned)(n_ff / 8), (unsigned)topk);
        dim3 g_dn((unsigned)(n_embd / 16), (unsigned)topk);
        if (w_type == GGML_TYPE_NVFP4) {
            experts_mma_t1_gu_nvfp4_kernel<8><<<g_gu, 256, 0, stream>>>(
                grp_gu_ptrs, off_up, n_embd, n_ff, x_limbs, slot_sg, slot_su,
                situ_beta, situ_linear_beta, inter_f16
            );
            experts_mma_t1_down_nvfp4_kernel<4><<<g_dn, 128, 0, stream>>>(
                grp_dn_ptrs, slot_wt, n_embd, n_ff, inter_f16, slot_out
            );
        } else {
            experts_mma_t1_gu_q8_kernel<8><<<g_gu, 256, 0, stream>>>(
                grp_gu_ptrs, off_up, n_embd, n_ff, x_limbs, slot_sg, slot_su,
                situ_beta, situ_linear_beta, inter_f16
            );
            experts_mma_t1_down_q8_kernel<4><<<g_dn, 128, 0, stream>>>(
                grp_dn_ptrs, slot_wt, n_embd, n_ff, inter_f16, slot_out
            );
        }
    } else {
        const int S = T * topk;
        dim3 g_gu((unsigned)(n_ff / 8), (unsigned)S);
        dim3 g_dn((unsigned)(n_embd / 16), (unsigned)S);
        if (w_type == GGML_TYPE_NVFP4) {
            if (T <= 4) {
                experts_block_gu_nvfp4_kernel<8, 1><<<g_gu, 256, 0, stream>>>(
                    grp_gu_ptrs, grp_n, grp_tok, grp_row, off_up, n_embd, n_ff,
                    x_limbs, slot_sg, slot_su, situ_beta, situ_linear_beta, inter_f16
                );
            } else {
                experts_block_gu_nvfp4_kernel<8, 2><<<g_gu, 256, 0, stream>>>(
                    grp_gu_ptrs, grp_n, grp_tok, grp_row, off_up, n_embd, n_ff,
                    x_limbs, slot_sg, slot_su, situ_beta, situ_linear_beta, inter_f16
                );
            }
            experts_block_down_nvfp4_kernel<4><<<g_dn, 128, 0, stream>>>(
                grp_dn_ptrs, grp_n, grp_row, slot_wt, n_embd, n_ff, inter_f16, slot_out
            );
        } else {
            if (T <= 4) {
                experts_block_gu_q8_kernel<8, 1><<<g_gu, 256, 0, stream>>>(
                    grp_gu_ptrs, grp_n, grp_tok, grp_row, off_up, n_embd, n_ff,
                    x_limbs, slot_sg, slot_su, situ_beta, situ_linear_beta, inter_f16
                );
            } else {
                experts_block_gu_q8_kernel<8, 2><<<g_gu, 256, 0, stream>>>(
                    grp_gu_ptrs, grp_n, grp_tok, grp_row, off_up, n_embd, n_ff,
                    x_limbs, slot_sg, slot_su, situ_beta, situ_linear_beta, inter_f16
                );
            }
            experts_block_down_q8_kernel<4><<<g_dn, 128, 0, stream>>>(
                grp_dn_ptrs, grp_n, grp_row, slot_wt, n_embd, n_ff, inter_f16, slot_out
            );
        }
    }

    moe_combine_conv_add_block_kernel<<<(unsigned)((n_embd + 63) / 64), 64, 0, stream>>>(
        slot_out, ffn_post_conv_w, conv_history, ffn_inp, cur_out,
        rec_old_hist, rec_x, n_sconv_channels, slot_idx,
        n_embd, topk, T, conv_offset, n_conv
    );
}

} // extern "C"
