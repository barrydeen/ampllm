#include "common/sm86_common.cuh"

// Fused or standalone RMSNorm across last dimension `dim`.
// gamma may be nullptr (use_value_norm_no_gamma).
__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ y,
    int dim,
    int M,
    float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;
    int tid = threadIdx.x;
    const float* xr = x + (size_t)row * dim;
    float* yr = y + (size_t)row * dim;

    float sum2 = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = xr[i];
        sum2 += v * v;
    }
    sum2 = warp_reduce_sum(sum2);
    __shared__ float warp_sums[32];
    int wid = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) warp_sums[wid] = sum2;
    __syncthreads();
    if (wid == 0) {
        float t = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        t = warp_reduce_sum(t);
        if (lane == 0) warp_sums[0] = rsqrtf(t / (float)dim + eps);
    }
    __syncthreads();
    float inv = warp_sums[0];
    for (int i = tid; i < dim; i += blockDim.x) {
        float g = gamma ? gamma[i] : 1.0f;
        yr[i] = xr[i] * inv * g;
    }
}

// Per-head L2 clip + optional learned per-channel scale + optional tau scaling.
// x: [M, n_head, head_dim], scale: [n_head, head_dim] or nullptr
__global__ void l2_clip_scale_kernel(
    float* __restrict__ x,
    const float* __restrict__ scale,
    const float* __restrict__ tau, // [M] or nullptr
    int head_dim,
    int n_head,
    int M,
    float threshold
) {
    int m = blockIdx.y;
    int h = blockIdx.x * blockDim.y + threadIdx.y;
    if (h >= n_head || m >= M) return;
    int lane = threadIdx.x;
    float* xh = x + ((size_t)m * n_head + h) * head_dim;
    const float* sh = scale ? (scale + (size_t)h * head_dim) : nullptr;
    float t_mul = tau ? tau[m] : 1.0f;

    float sum2 = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
        float v = xh[d];
        if (sh) v *= sh[d];
        sum2 += v * v;
    }
    sum2 = warp_reduce_sum(sum2);
    float norm = sqrtf(sum2) + 1e-6f;
    float factor = fminf(1.0f, threshold / norm) * t_mul;
    for (int d = lane; d < head_dim; d += 32) {
        float v = xh[d];
        if (sh) v *= sh[d];
        xh[d] = v * factor;
    }
}

// Block AttnRes depth mixing:
// states: array of `n_states` pointers to [M, n_embd] tensors (last state is `cur`, earlier are `blocks`).
// queries: [n_q, n_embd] F32 (where n_q = 1, 3, or 4)
// out: [n_q, M, n_embd] (so query q's output for all M tokens is contiguous at out[q * M * n_embd])
__global__ void attn_res_mix_kernel(
    const float* const* __restrict__ state_ptrs, // device ptr array of size n_states
    int n_states,
    const float* __restrict__ queries, // [n_q, n_embd]
    float* __restrict__ out,           // [n_q, M, n_embd]
    int n_embd,
    int n_q,
    int M,
    float eps
) {
    int m = blockIdx.x;
    if (m >= M) return;
    int tid = threadIdx.x;
    const int MAX_S = 16;

    // Shared memory: store inv_rms[MAX_S], scores[4 * MAX_S], alphas[4 * MAX_S]
    __shared__ float s_inv_rms[MAX_S];
    __shared__ float s_scores[4][MAX_S];
    __shared__ float warp_tmp[32];

    float inv_sqrt_d = rsqrtf((float)n_embd);

    // 1. Compute inv_rms for each state j = 0..n_states-1
    for (int j = 0; j < n_states; ++j) {
        const float* sj = state_ptrs[j] + (size_t)m * n_embd;
        float sum2 = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            float v = sj[i];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        int wid = tid >> 5;
        int lane = tid & 31;
        if (lane == 0) warp_tmp[wid] = sum2;
        __syncthreads();
        if (wid == 0) {
            float tot = (lane < (blockDim.x >> 5)) ? warp_tmp[lane] : 0.0f;
            tot = warp_reduce_sum(tot);
            if (lane == 0) s_inv_rms[j] = rsqrtf(tot / (float)n_embd + eps);
        }
        __syncthreads();
    }

    // 2. Compute dot(rms_norm(state[j]), query[q]) * inv_sqrt_d for each q and j
    for (int q = 0; q < n_q; ++q) {
        const float* qq = queries + (size_t)q * n_embd;
        for (int j = 0; j < n_states; ++j) {
            const float* sj = state_ptrs[j] + (size_t)m * n_embd;
            float dot = 0.0f;
            for (int i = tid; i < n_embd; i += blockDim.x) {
                dot += sj[i] * qq[i];
            }
            dot = warp_reduce_sum(dot);
            int wid = tid >> 5;
            int lane = tid & 31;
            if (lane == 0) warp_tmp[wid] = dot;
            __syncthreads();
            if (wid == 0) {
                float tot = (lane < (blockDim.x >> 5)) ? warp_tmp[lane] : 0.0f;
                tot = warp_reduce_sum(tot);
                if (lane == 0) {
                    s_scores[q][j] = tot * s_inv_rms[j] * inv_sqrt_d;
                }
            }
            __syncthreads();
        }
        // Softmax over j = 0..n_states-1 on thread 0
        if (tid == 0) {
            float mx = -1e30f;
            for (int j = 0; j < n_states; ++j) mx = fmaxf(mx, s_scores[q][j]);
            float sum = 0.0f;
            for (int j = 0; j < n_states; ++j) {
                float e = expf(s_scores[q][j] - mx);
                s_scores[q][j] = e;
                sum += e;
            }
            float inv_sum = 1.0f / sum;
            for (int j = 0; j < n_states; ++j) {
                s_scores[q][j] *= inv_sum;
            }
        }
        __syncthreads();
    }

    // 3. Output weighted sum of RMS-normalized states
    for (int i = tid; i < n_embd; i += blockDim.x) {
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int j = 0; j < n_states; ++j) {
            float v_norm = state_ptrs[j][(size_t)m * n_embd + i] * s_inv_rms[j];
            for (int q = 0; q < n_q; ++q) {
                acc[q] += s_scores[q][j] * v_norm;
            }
        }
        for (int q = 0; q < n_q; ++q) {
            out[((size_t)q * M + m) * n_embd + i] = acc[q];
        }
    }
}

// Fast float4-vectorized Block AttnRes depth mixing with optional fused post-mix RMSNorm!
// Launches 1 block per (token m, job_idx). With n_embd=2304 (576 float4s) and 256 threads (8 warps),
// each thread holds 2/3 float4s in registers — zero second global read of states when FUSED_RMS is true!
template <bool FUSED_RMS>
__global__ void __launch_bounds__(256) attn_res_vec4_fast_kernel(
    const float* const* __restrict__ state_ptrs,
    int n_states,
    const float* __restrict__ queries, // [n_jobs * n_q, n_embd]
    float* __restrict__ out,           // [n_jobs * n_q, M, n_embd] (when !FUSED_RMS)
    const float* __restrict__ norm_w0,
    const float* __restrict__ norm_w1,
    const float* __restrict__ norm_w2,
    float* __restrict__ norm_y0,
    float* __restrict__ norm_y1,
    float* __restrict__ norm_y2,
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
    const int job = blockIdx.y;
    if (m >= M) return;

    const float* q_base = queries + (size_t)job * n_q * n_embd;
    float* out_base = out ? (out + (size_t)job * n_q * M * n_embd) : nullptr;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c4 = n_embd >> 2;
    const int d4_0 = tid;
    const int d4_1 = tid + 256;
    const int d4_2 = tid + 512;
    const bool in0 = d4_0 < c4;
    const bool in1 = d4_1 < c4;
    const bool in2 = d4_2 < c4;

    float4 q0[4] = {}, q1[4] = {}, q2[4] = {};
    #pragma unroll
    for (int qi = 0; qi < 4; ++qi) {
        if (qi < n_q) {
            const float4* qp = reinterpret_cast<const float4*>(q_base + (size_t)qi * n_embd);
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
            for (int qi = 0; qi < 4; ++qi) {
                if (qi < n_q) qd[qi] += xv.x * q0[qi].x + xv.y * q0[qi].y + xv.z * q0[qi].z + xv.w * q0[qi].w;
            }
        }
        if (in1) {
            const float4 xv = __ldg(x4 + d4_1);
            sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
            #pragma unroll
            for (int qi = 0; qi < 4; ++qi) {
                if (qi < n_q) qd[qi] += xv.x * q1[qi].x + xv.y * q1[qi].y + xv.z * q1[qi].z + xv.w * q1[qi].w;
            }
        }
        if (in2) {
            const float4 xv = __ldg(x4 + d4_2);
            sum_sq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
            #pragma unroll
            for (int qi = 0; qi < 4; ++qi) {
                if (qi < n_q) qd[qi] += xv.x * q2[qi].x + xv.y * q2[qi].y + xv.z * q2[qi].z + xv.w * q2[qi].w;
            }
        }
        sum_sq = warp_reduce_sum(sum_sq);
        #pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            if (qi < n_q) qd[qi] = warp_reduce_sum(qd[qi]);
        }
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

    if constexpr (!FUSED_RMS) {
        #pragma unroll
        for (int qi = 0; qi < 4; ++qi) {
            if (qi >= n_q) break;
            float4* y4 = reinterpret_cast<float4*>(out_base + ((size_t)qi * M + m) * n_embd);
            if (in0) y4[d4_0] = o0[qi];
            if (in1) y4[d4_1] = o1[qi];
            if (in2) y4[d4_2] = o2[qi];
        }
    } else {
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
        #pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= n_q) break;
            const float rs = rs_rms[qi];
            const float4* w4 = reinterpret_cast<const float4*>(nw[qi]);
            float4* y4 = reinterpret_cast<float4*>(ny[qi] + (size_t)m * n_embd);
            if (in0) {
                const float4 ww = __ldg(w4 + d4_0);
                y4[d4_0] = make_float4(o0[qi].x * rs * ww.x, o0[qi].y * rs * ww.y, o0[qi].z * rs * ww.z, o0[qi].w * rs * ww.w);
            }
            if (in1) {
                const float4 ww = __ldg(w4 + d4_1);
                y4[d4_1] = make_float4(o1[qi].x * rs * ww.x, o1[qi].y * rs * ww.y, o1[qi].z * rs * ww.z, o1[qi].w * rs * ww.w);
            }
            if (in2) {
                const float4 ww = __ldg(w4 + d4_2);
                y4[d4_2] = make_float4(o2[qi].x * rs * ww.x, o2[qi].y * rs * ww.y, o2[qi].z * rs * ww.z, o2[qi].w * rs * ww.w);
            }
        }
    }
}

// Multi-RMSNorm (1..3 vectors in a single launch, 1 CTA per (m, job))
struct MultiRmsArgs {
    const float* x[3];
    const float* w[3];
    float* y[3];
};

__global__ void rmsnorm_multi_kernel(MultiRmsArgs args, int dim, int M, float eps) {
    int m = blockIdx.x;
    int j = blockIdx.y;
    if (m >= M) return;
    int tid = threadIdx.x;
    const float* xr = args.x[j] + (size_t)m * dim;
    const float* gamma = args.w[j];
    float* yr = args.y[j] + (size_t)m * dim;

    float sum2 = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = xr[i];
        sum2 += v * v;
    }
    sum2 = warp_reduce_sum(sum2);
    __shared__ float warp_sums[32];
    int wid = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) warp_sums[wid] = sum2;
    __syncthreads();
    if (wid == 0) {
        float t = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
        t = warp_reduce_sum(t);
        if (lane == 0) warp_sums[0] = rsqrtf(t / (float)dim + eps);
    }
    __syncthreads();
    float inv = warp_sums[0];
    for (int i = tid; i < dim; i += blockDim.x) {
        float g = gamma ? gamma[i] : 1.0f;
        yr[i] = xr[i] * inv * g;
    }
}

// Fused 2-launch loop controller for 1..3 gates (Q, K, V in 2 launches total, or FFN in 2 launches):
// 1. loop_gate_dots_kernel: Grid (R, n_gates, M), Block 128 -> computes dot(D_x[r], x), dot(D_m[r], m) and sum-of-squares(x, m)!
// 2. loop_gate_apply_kernel: Grid (ceil(C/32), n_gates, ceil(M/8)), Block 256 -> loads F[r, c] and G[r, c] into shared memory once across R=64 and computes y = (1 - tanh(F*g))*x + tanh(G*g)*m!
struct FusedLoopGateArgs {
    const float* x[3];
    const float* m[3];
    const float* D[3];
    const float* F[3];
    const float* G[3];
    float* y[3];
};

__global__ void loop_gate_dots_kernel(FusedLoopGateArgs a, int M, int C, int R, float* __restrict__ sc) {
    __shared__ float sh[4][4];
    const int r = blockIdx.x;
    const int j = blockIdx.y;
    const int t = blockIdx.z;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const float* xr = a.x[j] + (size_t)t * C;
    const float* mr = a.m[j] + (size_t)t * C;
    const float* dr = a.D[j] + (size_t)r * 2 * C;
    float ax = 0.0f, am = 0.0f, sx = 0.0f, sm = 0.0f;
    for (int c = tid; c < C; c += 128) {
        const float xv = xr[c];
        const float mv = mr[c];
        ax += dr[c] * xv;
        am += dr[C + c] * mv;
        sx += xv * xv;
        sm += mv * mv;
    }
    ax = warp_reduce_sum(ax);
    am = warp_reduce_sum(am);
    sx = warp_reduce_sum(sx);
    sm = warp_reduce_sum(sm);
    if ((tid & 31) == 0) {
        sh[warp][0] = ax;
        sh[warp][1] = am;
        sh[warp][2] = sx;
        sh[warp][3] = sm;
    }
    __syncthreads();
    if (tid == 0) {
        float* o = sc + ((size_t)j * M + t) * (2 * R + 2);
        o[2 * r]     = sh[0][0] + sh[1][0] + sh[2][0] + sh[3][0];
        o[2 * r + 1] = sh[0][1] + sh[1][1] + sh[2][1] + sh[3][1];
        if (r == 0) {
            o[2 * R]     = sh[0][2] + sh[1][2] + sh[2][2] + sh[3][2];
            o[2 * R + 1] = sh[0][3] + sh[1][3] + sh[2][3] + sh[3][3];
        }
    }
}

constexpr int kGateCh = 32;
constexpr int kGateWarps = 8;
constexpr int kLoopRankMax = 128;
constexpr int kGateRows = kLoopRankMax / kGateWarps;

__global__ void __launch_bounds__(kGateWarps * 32) loop_gate_apply_kernel(
    FusedLoopGateArgs a,
    int M,
    int C,
    int R,
    float eps,
    const float* __restrict__ sc
) {
    extern __shared__ float sh_lg[];
    float* sF = sh_lg;
    float* sG = sF + (size_t)R * kGateCh;
    float* sg = sG + (size_t)R * kGateCh;

    const int j = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c = (int)blockIdx.x * kGateCh + lane;
    const int t0 = (int)blockIdx.z * kGateWarps;
    const int nt = min(kGateWarps, M - t0);

    {
        // Note: in raw GGUF layout, F and G are [n_embd, R] (shape [R, n_embd]),
        // so channel `c`'s R weights are contiguous at `F + c * R + r`!
        const float* F_ch = (c < C) ? (a.F[j] + (size_t)c * R) : nullptr;
        const float* G_ch = (c < C) ? (a.G[j] + (size_t)c * R) : nullptr;
        float vf[kGateRows], vg[kGateRows];
        #pragma unroll
        for (int q = 0; q < kGateRows; ++q) {
            const int r = warp + q * kGateWarps;
            const bool in = (r < R) && (c < C);
            vf[q] = in ? __ldg(F_ch + r) : 0.0f;
            vg[q] = in ? __ldg(G_ch + r) : 0.0f;
        }
        #pragma unroll
        for (int q = 0; q < kGateRows; ++q) {
            const int r = warp + q * kGateWarps;
            if (r < R) {
                sF[r * kGateCh + lane] = vf[q];
                sG[r * kGateCh + lane] = vg[q];
            }
        }
    }
    for (int i = tid; i < nt * R; i += kGateWarps * 32) {
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
        f  += sF[r * kGateCh + lane] * gv;
        gn += sG[r * kGateCh + lane] * gv;
    }
    const size_t idx = (size_t)(t0 + warp) * C + c;
    const float xv = a.x[j][idx];
    const float mv = a.m[j][idx];
    a.y[j][idx] = (1.0f - tanhf(f)) * xv + tanhf(gn) * mv;
}

// Concatenate [rms_norm(cur), rms_norm(msg)] -> [M, 2 * n_embd]
__global__ void concat_two_rmsnorm_kernel(
    const float* __restrict__ cur,
    const float* __restrict__ msg,
    float* __restrict__ out, // [M, 2 * n_embd]
    int n_embd,
    int M,
    float eps
) {
    int m = blockIdx.x;
    if (m >= M) return;
    int tid = threadIdx.x;
    const float* c = cur + (size_t)m * n_embd;
    const float* g = msg + (size_t)m * n_embd;
    float* o = out + (size_t)m * (2 * n_embd);

    float sc = 0.0f, sg = 0.0f;
    for (int i = tid; i < n_embd; i += blockDim.x) {
        float vc = c[i];
        float vg = g[i];
        sc += vc * vc;
        sg += vg * vg;
    }
    sc = warp_reduce_sum(sc);
    sg = warp_reduce_sum(sg);
    __shared__ float wc[32], wg[32];
    int wid = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) { wc[wid] = sc; wg[wid] = sg; }
    __syncthreads();
    if (wid == 0) {
        float tc = (lane < (blockDim.x >> 5)) ? wc[lane] : 0.0f;
        float tg = (lane < (blockDim.x >> 5)) ? wg[lane] : 0.0f;
        tc = warp_reduce_sum(tc);
        tg = warp_reduce_sum(tg);
        if (lane == 0) {
            wc[0] = rsqrtf(tc / (float)n_embd + eps);
            wg[0] = rsqrtf(tg / (float)n_embd + eps);
        }
    }
    __syncthreads();
    float ic = wc[0], ig = wg[0];
    for (int i = tid; i < n_embd; i += blockDim.x) {
        o[i] = c[i] * ic;
        o[n_embd + i] = g[i] * ig;
    }
}

// Loop controller apply: out = (1 - tanh(forget)) * cur + tanh(gain) * msg
__global__ void apply_loop_ctrl_kernel(
    float* __restrict__ cur,
    const float* __restrict__ msg,
    const float* __restrict__ forget_raw,
    const float* __restrict__ gain_raw,
    int n_total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    float f = tanhf(forget_raw[i]);
    float g = tanhf(gain_raw[i]);
    float c = cur[i];
    cur[i] = (c - f * c) + g * msg[i];
}

// Exact inkling2.cpp causal depthwise short convolution with residual & optional dynamic per-head taps.
// cur: [M, n_channels] (overwritten with output OR written to out)
// static_w: [n_channels, 4] F32 (note GGUF shape [4, n_channels] => channel c has 4 contiguous floats at static_w + c * 4!)
// dyn_proj: [M, n_heads * 4] F32 or nullptr (when non-null: for head h, token t, taps are dyn_proj[(t * n_heads + h) * 4 + lag])
// history: [n_channels, 3] F32 (ring/shift state holding last 3 inputs at channel offset)
__global__ void short_conv_f32_kernel(
    const float* __restrict__ cur,
    const float* __restrict__ static_w,
    const float* __restrict__ dyn_proj,
    const float* __restrict__ add_res,
    float* __restrict__ history, // [n_sconv_channels, 3] or [3, n_sconv_channels]? stored per channel: ch * 3 + k
    float* __restrict__ out,
    int n_channels,
    int n_heads,
    int M,
    int conv_offset,
    int n_conv // 4
) {
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n_channels) return;

    int head_dim = (n_heads > 0) ? (n_channels / n_heads) : n_channels;
    int head = (n_heads > 0) ? (ch / head_dim) : 0;

    float* hist_ch = history + (size_t)(conv_offset + ch) * 3;
    float h0 = hist_ch[0]; // t-3
    float h1 = hist_ch[1]; // t-2
    float h2 = hist_ch[2]; // t-1

    const float* sw = static_w + (size_t)ch * 4;
    float sw0 = sw[0], sw1 = sw[1], sw2 = sw[2], sw3 = sw[3];
    float inv_scale = rsqrtf((float)((dyn_proj ? 2 : 1) * n_conv + 1));

    for (int t = 0; t < M; ++t) {
        float xt = cur[(size_t)t * n_channels + ch];
        float conv_val = h0 * sw0 + h1 * sw1 + h2 * sw2 + xt * sw3;
        if (dyn_proj) {
            const float* dp = dyn_proj + ((size_t)t * n_heads + head) * 4;
            // lag=0 -> xt (t), lag=1 -> h2 (t-1), lag=2 -> h1 (t-2), lag=3 -> h0 (t-3)
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

__global__ void sigmoid_mul_kernel(
    float* __restrict__ a,
    const float* __restrict__ gate,
    int n_total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    float g = gate[i];
    float sig = 1.0f / (1.0f + expf(-g));
    a[i] *= sig;
}

__global__ void add_f32_kernel(float* __restrict__ a, const float* __restrict__ b, int n_total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    a[i] += b[i];
}

__global__ void compute_tau_kernel(
    const int32_t* __restrict__ pos,
    float* __restrict__ tau,
    int M,
    float n_floor,
    float alpha
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) return;
    float ratio = fmaxf(1.0f, ((float)pos[i] + 1.0f) / n_floor);
    tau[i] = 1.0f + alpha * logf(ratio);
}

// Compute router top-K selection (sigmoid + exp_probs_b for selection, normalized unbiased probs * scale).
// Parallel 128-thread rank-counting kernel for n_expert <= 128, with optional fused FP32 -> 2x FP16 limbs [2, n_embd] in blockIdx.y == 1!
__global__ void router_topk_128_and_limbs_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights,
    const float* __restrict__ x_f32,
    uint16_t* __restrict__ x_limbs,
    int n_expert,
    int topk,
    int n_embd,
    float route_scale
) {
    if (blockIdx.y > 0) {
        if (!x_f32 || !x_limbs) return;
        int i = (blockIdx.y - 1) * blockDim.x + threadIdx.x;
        if (i < n_embd) {
            float v = x_f32[i];
            uint16_t hi = f32_to_fp16_bits(v);
            uint16_t lo = f32_to_fp16_bits(v - fp16_bits_to_f32(hi));
            x_limbs[i] = hi;
            x_limbs[n_embd + i] = lo;
        }
        return;
    }

    int m = blockIdx.x;
    int e = threadIdx.x; // 0..127
    int lane = e & 31;
    int warp = e >> 5;

    __shared__ float s_val[128];
    __shared__ uint32_t s_picked[4];
    __shared__ float s_sig[17];

    const float* row = logits + (size_t)m * n_expert;
    float sl = (e < n_expert) ? (1.0f / (1.0f + expf(-row[e]))) : 0.0f;
    float v  = (e < n_expert) ? (sl + (bias ? bias[e] : 0.0f)) : -1e30f;
    s_val[e] = v;
    __syncthreads();

    int rank = 0;
    if (e < n_expert) {
        #pragma unroll 8
        for (int j = 0; j < n_expert; ++j) {
            float o = s_val[j];
            rank += (o > v || (o == v && j < e)) ? 1 : 0;
        }
    }
    bool pick = (e < n_expert) && (rank < topk);
    uint32_t mine = __ballot_sync(0xffffffffu, pick);
    if (lane == 0) s_picked[warp] = mine;
    __syncthreads();

    int pos = __popc(mine & ((1u << lane) - 1u));
    for (int ww = 0; ww < warp; ++ww) pos += __popc(s_picked[ww]);
    if (pick) s_sig[pos] = sl;
    __syncthreads();

    if (e == 0) {
        float sum = 0.0f;
        for (int k = 0; k < topk; ++k) sum += s_sig[k];
        s_sig[topk] = sum;
    }
    __syncthreads();

    if (pick) {
        float wk = sl / (s_sig[topk] + 1e-20f) * route_scale;
        out_ids[(size_t)m * topk + pos] = e;
        out_weights[(size_t)m * topk + pos] = wk;
    }
}

// Compute router top-K selection (sigmoid + exp_probs_b for selection, normalized unbiased probs * scale).
// logits: [M, n_expert], bias: [n_expert] or nullptr
// out_ids: [M, k], out_weights: [M, k]
__global__ void router_topk_sigmoid_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int32_t* __restrict__ out_ids,
    float* __restrict__ out_weights,
    int n_expert,
    int topk,
    int M,
    float route_scale
) {
    int m = blockIdx.x;
    if (m >= M) return;
    int lane = threadIdx.x; // 0..31

    const float* row = logits + (size_t)m * n_expert;

    // Supports n_expert up to 512 via shared memory (zero local stack frame!)
    __shared__ float s_sel[512];
    __shared__ float s_prob[512];
    for (int e = lane; e < n_expert; e += 32) {
        float p = 1.0f / (1.0f + expf(-row[e]));
        float b = bias ? bias[e] : 0.0f;
        s_prob[e] = p;
        s_sel[e] = p + b;
    }
    __syncwarp();

    __shared__ int32_t s_ids[16];
    __shared__ float s_w[16];

    for (int k = 0; k < topk; ++k) {
        float best_sel = -1e30f;
        float best_prob = 0.0f;
        int best_e = n_expert;
        for (int e = lane; e < n_expert; e += 32) {
            float v = s_sel[e];
            if (v > best_sel) {
                best_sel = v;
                best_prob = s_prob[e];
                best_e = e;
            }
        }
        // Warp reduction for max (best_sel, tiebreak lower index)
        for (int off = 16; off > 0; off >>= 1) {
            float o_sel = __shfl_xor_sync(0xffffffff, best_sel, off);
            float o_prob = __shfl_xor_sync(0xffffffff, best_prob, off);
            int o_e = __shfl_xor_sync(0xffffffff, best_e, off);
            if (o_sel > best_sel || (o_sel == best_sel && o_e < best_e)) {
                best_sel = o_sel;
                best_prob = o_prob;
                best_e = o_e;
            }
        }
        if (lane == 0) {
            s_ids[k] = best_e;
            s_w[k] = best_prob;
            s_sel[best_e] = -1e30f;
        }
        __syncwarp();
    }

    if (lane < topk) {
        float sum_w = 0.0f;
        for (int k = 0; k < topk; ++k) sum_w += s_w[k];
        float norm_w = s_w[lane] / (sum_w + 1e-20f) * route_scale;
        out_ids[(size_t)m * topk + lane] = s_ids[lane];
        out_weights[(size_t)m * topk + lane] = norm_w;
    }
}

__global__ void softcap_tanh_kernel(float* __restrict__ x, int n_total, float cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_total) return;
    x[i] = cap * tanhf(x[i] / cap);
}

__global__ void argmax_f32_kernel(
    const float* __restrict__ logits,
    int n_vocab,
    float* __restrict__ out_val,
    int32_t* __restrict__ out_id
) {
    __shared__ float s_val[32];
    __shared__ int32_t s_idx[32];
    int tid = threadIdx.x;
    float best_v = -1e30f;
    int32_t best_i = 0;
    for (int i = tid; i < n_vocab; i += blockDim.x) {
        float v = logits[i];
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, best_v, off);
        int32_t oi = __shfl_xor_sync(0xffffffff, best_i, off);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }
    int wid = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) {
        s_val[wid] = best_v;
        s_idx[wid] = best_i;
    }
    __syncthreads();
    if (wid == 0) {
        float v = (lane < (blockDim.x >> 5)) ? s_val[lane] : -1e30f;
        int32_t idx = (lane < (blockDim.x >> 5)) ? s_idx[lane] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, v, off);
            int32_t oi = __shfl_xor_sync(0xffffffff, idx, off);
            if (ov > v || (ov == v && oi < idx)) {
                v = ov;
                idx = oi;
            }
        }
        if (lane == 0) {
            if (out_val) *out_val = v;
            *out_id = idx;
        }
    }
}

extern "C" {

void launch_rmsnorm_f32(
    const float* x,
    const float* gamma,
    float* y,
    int dim,
    int M,
    float eps,
    cudaStream_t stream
) {
    rmsnorm_kernel<<<M, 256, 0, stream>>>(x, gamma, y, dim, M, eps);
}

void launch_l2_clip_scale_f32(
    float* x,
    const float* scale,
    const float* tau,
    int head_dim,
    int n_head,
    int M,
    float threshold,
    cudaStream_t stream
) {
    const int warps = 4;
    dim3 block(32, warps);
    dim3 grid((n_head + warps - 1) / warps, M);
    l2_clip_scale_kernel<<<grid, block, 0, stream>>>(x, scale, tau, head_dim, n_head, M, threshold);
}

void launch_attn_res_mix_f32(
    const float* const* state_ptrs,
    int n_states,
    const float* queries,
    float* out,
    int n_embd,
    int n_q,
    int M,
    float eps,
    cudaStream_t stream
) {
    if (n_embd == 2304 && n_states <= 16 && n_q <= 4 &&
        (((uintptr_t)queries & 15) == 0) && (((uintptr_t)out & 15) == 0)) {
        dim3 grid(M, 1);
        attn_res_vec4_fast_kernel<false><<<grid, 256, 0, stream>>>(
            state_ptrs, n_states, queries, out,
            nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr,
            n_embd, n_q, M, eps, eps
        );
    } else {
        attn_res_mix_kernel<<<M, 256, 0, stream>>>(state_ptrs, n_states, queries, out, n_embd, n_q, M, eps);
    }
}

void launch_attn_res_mix_jobs_f32(
    const float* const* state_ptrs,
    int n_states,
    const float* queries,
    float* out,
    int n_embd,
    int n_q,
    int n_jobs,
    int M,
    float eps,
    cudaStream_t stream
) {
    dim3 grid(M, n_jobs);
    attn_res_vec4_fast_kernel<false><<<grid, 256, 0, stream>>>(
        state_ptrs, n_states, queries, out,
        nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        n_embd, n_q, M, eps, eps
    );
}

void launch_attn_res_mix_rms_f32(
    const float* const* state_ptrs,
    int n_states,
    const float* queries,
    const float* norm_w0,
    const float* norm_w1,
    const float* norm_w2,
    float* norm_y0,
    float* norm_y1,
    float* norm_y2,
    int n_embd,
    int n_q,
    int M,
    float res_eps,
    float rms_eps,
    cudaStream_t stream
) {
    dim3 grid(M, 1);
    attn_res_vec4_fast_kernel<true><<<grid, 256, 0, stream>>>(
        state_ptrs, n_states, queries, nullptr,
        norm_w0, norm_w1, norm_w2,
        norm_y0, norm_y1, norm_y2,
        n_embd, n_q, M, res_eps, rms_eps
    );
}

void launch_rmsnorm_multi_f32(
    const float* x0, const float* x1, const float* x2,
    const float* w0, const float* w1, const float* w2,
    float* y0, float* y1, float* y2,
    int n_jobs,
    int dim,
    int M,
    float eps,
    cudaStream_t stream
) {
    MultiRmsArgs args{};
    args.x[0] = x0; args.x[1] = x1; args.x[2] = x2;
    args.w[0] = w0; args.w[1] = w1; args.w[2] = w2;
    args.y[0] = y0; args.y[1] = y1; args.y[2] = y2;
    dim3 grid(M, n_jobs);
    rmsnorm_multi_kernel<<<grid, 256, 0, stream>>>(args, dim, M, eps);
}

void launch_loop_gates_fused_f32(
    const float* x0, const float* x1, const float* x2,
    const float* m0, const float* m1, const float* m2,
    const float* d0, const float* d1, const float* d2,
    const float* f0, const float* f1, const float* f2,
    const float* g0, const float* g1, const float* g2,
    float* y0, float* y1, float* y2,
    float* scratch_sc,
    int n_gates,
    int M,
    int C,
    int R,
    float eps,
    cudaStream_t stream
) {
    FusedLoopGateArgs a{};
    a.x[0] = x0; a.x[1] = x1; a.x[2] = x2;
    a.m[0] = m0; a.m[1] = m1; a.m[2] = m2;
    a.D[0] = d0; a.D[1] = d1; a.D[2] = d2;
    a.F[0] = f0; a.F[1] = f1; a.F[2] = f2;
    a.G[0] = g0; a.G[1] = g1; a.G[2] = g2;
    a.y[0] = y0; a.y[1] = y1; a.y[2] = y2;

    dim3 g_dots((unsigned)R, (unsigned)n_gates, (unsigned)M);
    loop_gate_dots_kernel<<<g_dots, 128, 0, stream>>>(a, M, C, R, scratch_sc);

    dim3 g_apply(
        (unsigned)((C + kGateCh - 1) / kGateCh),
        (unsigned)n_gates,
        (unsigned)((M + kGateWarps - 1) / kGateWarps)
    );
    size_t smem = (size_t)R * (2 * kGateCh + kGateWarps) * sizeof(float);
    loop_gate_apply_kernel<<<g_apply, kGateWarps * 32, smem, stream>>>(a, M, C, R, eps, scratch_sc);
}

void launch_concat_two_rmsnorm_f32(
    const float* cur,
    const float* msg,
    float* out,
    int n_embd,
    int M,
    float eps,
    cudaStream_t stream
) {
    concat_two_rmsnorm_kernel<<<M, 256, 0, stream>>>(cur, msg, out, n_embd, M, eps);
}

void launch_apply_loop_ctrl_f32(
    float* cur,
    const float* msg,
    const float* forget_raw,
    const float* gain_raw,
    int n_total,
    cudaStream_t stream
) {
    int threads = 256;
    apply_loop_ctrl_kernel<<<(n_total + threads - 1) / threads, threads, 0, stream>>>(
        cur, msg, forget_raw, gain_raw, n_total
    );
}

void launch_short_conv_f32(
    const float* cur,
    const float* static_w,
    const float* dyn_proj,
    float* history,
    float* out,
    int n_channels,
    int n_heads,
    int M,
    int conv_offset,
    int n_conv,
    cudaStream_t stream
) {
    int threads = 256;
    short_conv_f32_kernel<<<(n_channels + threads - 1) / threads, threads, 0, stream>>>(
        cur, static_w, dyn_proj, nullptr, history, out, n_channels, n_heads, M, conv_offset, n_conv
    );
}

void launch_short_conv_add_f32(
    const float* cur,
    const float* static_w,
    const float* dyn_proj,
    const float* add_res,
    float* history,
    float* out,
    int n_channels,
    int n_heads,
    int M,
    int conv_offset,
    int n_conv,
    cudaStream_t stream
) {
    int threads = 256;
    short_conv_f32_kernel<<<(n_channels + threads - 1) / threads, threads, 0, stream>>>(
        cur, static_w, dyn_proj, add_res, history, out, n_channels, n_heads, M, conv_offset, n_conv
    );
}

void launch_sigmoid_mul_f32(float* a, const float* gate, int n_total, cudaStream_t stream) {
    int threads = 256;
    sigmoid_mul_kernel<<<(n_total + threads - 1) / threads, threads, 0, stream>>>(a, gate, n_total);
}

void launch_add_f32(float* a, const float* b, int n_total, cudaStream_t stream) {
    int threads = 256;
    add_f32_kernel<<<(n_total + threads - 1) / threads, threads, 0, stream>>>(a, b, n_total);
}

void launch_compute_tau_f32(const int32_t* pos, float* tau, int M, float n_floor, float alpha, cudaStream_t stream) {
    int threads = 256;
    compute_tau_kernel<<<(M + threads - 1) / threads, threads, 0, stream>>>(pos, tau, M, n_floor, alpha);
}

void launch_router_topk_sigmoid_f32(
    const float* logits,
    const float* bias,
    int32_t* out_ids,
    float* out_weights,
    int n_expert,
    int topk,
    int M,
    float route_scale,
    cudaStream_t stream
) {
    if (n_expert <= 128 && topk <= 16) {
        dim3 grid(M, 1);
        router_topk_128_and_limbs_kernel<<<grid, 128, 0, stream>>>(
            logits, bias, out_ids, out_weights, nullptr, nullptr, n_expert, topk, 0, route_scale
        );
    } else {
        router_topk_sigmoid_kernel<<<M, 32, 0, stream>>>(
            logits, bias, out_ids, out_weights, n_expert, topk, M, route_scale
        );
    }
}

void launch_router_topk_and_f16_limbs_m1(
    const float* logits,
    const float* bias,
    int32_t* out_ids,
    float* out_weights,
    const float* x_f32,
    void* x_limbs,
    int n_expert,
    int topk,
    int n_embd,
    float route_scale,
    cudaStream_t stream
) {
    int limb_blocks = (x_f32 != nullptr && x_limbs != nullptr) ? ((n_embd + 127) / 128) : 0;
    dim3 grid(1, 1 + limb_blocks);
    router_topk_128_and_limbs_kernel<<<grid, 128, 0, stream>>>(
        logits, bias, out_ids, out_weights, x_f32, reinterpret_cast<uint16_t*>(x_limbs),
        n_expert, topk, n_embd, route_scale
    );
}

void launch_softcap_tanh_f32(float* x, int n_total, float cap, cudaStream_t stream) {
    int threads = 256;
    softcap_tanh_kernel<<<(n_total + threads - 1) / threads, threads, 0, stream>>>(x, n_total, cap);
}

void launch_argmax_f32(const float* logits, int n_vocab, int32_t* out_id, cudaStream_t stream) {
    argmax_f32_kernel<<<1, 512, 0, stream>>>(logits, n_vocab, nullptr, out_id);
}

void launch_argmax_val_id_f32(const float* logits, int n_vocab, float* out_val, int32_t* out_id, cudaStream_t stream) {
    argmax_f32_kernel<<<1, 512, 0, stream>>>(logits, n_vocab, out_val, out_id);
}

} // extern "C"
