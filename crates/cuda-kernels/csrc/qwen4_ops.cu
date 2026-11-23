#include "common/sm86_common.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define QWEN4_SM80 1
#else
#define QWEN4_SM80 0
#endif

namespace {

__device__ __forceinline__ float qwen4_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float qwen4_block_reduce_sum_8w(float val, float* s_sums, int tid) {
    val = warp_reduce_sum(val);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) s_sums[warp] = val;
    __syncthreads();
    val = (tid < 8) ? s_sums[tid] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
        if (lane == 0) s_sums[0] = val;
    }
    __syncthreads();
    return s_sums[0];
}

__device__ __forceinline__ float qwen4_block_reduce_sum_4w(float val, float* s_sums, int tid) {
    val = warp_reduce_sum(val);
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) s_sums[warp] = val;
    __syncthreads();
    val = (tid < 4) ? s_sums[tid] : 0.0f;
    if (warp == 0) {
        val = warp_reduce_sum(val);
        if (lane == 0) s_sums[0] = val;
    }
    __syncthreads();
    return s_sums[0];
}

// ============================================================================
// 2a. HC-4 Gated Residual & General RMSNorm
// ============================================================================

__global__ void qwen4_hc_expand_kernel(
    const float* __restrict__ emb,
    int hc,
    int E,
    size_t total,
    float* __restrict__ R_out
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t tc = idx / (size_t)E;
    size_t d = idx - tc * (size_t)E;
    size_t t = tc / (size_t)hc;
    R_out[idx] = emb[t * (size_t)E + d];
}

template <bool WITH_INJ>
__global__ void __launch_bounds__(256) qwen4_hc_rmsnorm_q8_kernel(
    const float* __restrict__ R,
    const float* __restrict__ w_norm,
    const void* __restrict__ w_inject_opt,
    int inject_w_type,
    int hc,
    int E,
    float eps,
    float* __restrict__ xn_out,
    block_q8_1* __restrict__ xn_q8_opt,
    float* __restrict__ inject_partials_out
) {
    const int c = (int)blockIdx.x;
    const int t = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const size_t row = (size_t)t * (size_t)hc + (size_t)c;
    const float* x_row = R + row * (size_t)E;
    const float* g_row = w_norm + (size_t)c * (size_t)E;
    float* y_row = (xn_out != nullptr) ? (xn_out + row * (size_t)E) : nullptr;
    block_q8_1* q8_row = (xn_q8_opt != nullptr) ? (xn_q8_opt + row * (size_t)(E >> 5)) : nullptr;

    __shared__ float s_sums[8];
    float partial = 0.0f;
    if (((reinterpret_cast<uintptr_t>(x_row) & 15u) == 0u) && (E & 3) == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(x_row);
        const int E4 = E >> 2;
        for (int i = tid; i < E4; i += 256) {
            const float4 v = x4[i];
            partial += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    } else {
        for (int d = tid; d < E; d += 256) {
            float v = x_row[d];
            partial += v * v;
        }
    }
    float ss = qwen4_block_reduce_sum_8w(partial, s_sums, tid);
    float scale = rsqrtf(ss / (float)E + eps);

    float inj_p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const size_t c_off = (size_t)c * (size_t)E;
    const size_t stride_co = (size_t)hc * (size_t)E;

    for (int base = 0; base < E; base += 256) {
        int d = base + tid;
        float v = (d < E) ? (x_row[d] * scale * g_row[d]) : 0.0f;
        if (y_row != nullptr && d < E) {
            y_row[d] = v;
        }
        if constexpr (WITH_INJ) {
            if (d < E) {
                if (inject_w_type == GGML_TYPE_F32) {
                    const float* wi = reinterpret_cast<const float*>(w_inject_opt) + c_off + (size_t)d;
                    #pragma unroll
                    for (int co = 0; co < 4; ++co) {
                        if (co < hc) inj_p[co] = fmaf(v, wi[(size_t)co * stride_co], inj_p[co]);
                    }
                } else if (inject_w_type == GGML_TYPE_BF16) {
                    const uint16_t* wi = reinterpret_cast<const uint16_t*>(w_inject_opt) + c_off + (size_t)d;
                    #pragma unroll
                    for (int co = 0; co < 4; ++co) {
                        if (co < hc) inj_p[co] = fmaf(v, bf16_bits_to_f32(wi[(size_t)co * stride_co]), inj_p[co]);
                    }
                }
            }
        }
        if (q8_row != nullptr && (base + warp * 32 + 31 < E)) {
            int blk_idx = (base >> 5) + warp;
            warp_quantize_q8_1_store(v, q8_row + blk_idx, lane);
        }
    }

    if constexpr (WITH_INJ) {
        __shared__ float s_inj[4][8];
        #pragma unroll
        for (int co = 0; co < 4; ++co) {
            if (co < hc) {
                const float r = warp_reduce_sum(inj_p[co]);
                if (lane == 0) s_inj[co][warp] = r;
            }
        }
        __syncthreads();
        if (warp == 0) {
            const int co = tid >> 3;
            const int w = tid & 7;
            float val = (co < hc) ? s_inj[co][w] : 0.0f;
            val += __shfl_xor_sync(0xffffffffu, val, 4);
            val += __shfl_xor_sync(0xffffffffu, val, 2);
            val += __shfl_xor_sync(0xffffffffu, val, 1);
            if (w == 0 && co < hc) {
                inject_partials_out[((size_t)t * (size_t)hc + (size_t)co) * (size_t)hc + (size_t)c] = val;
            }
        }
    }
}

__global__ void __launch_bounds__(32) qwen4_hc_down_silu_q8_kernel(
    const float* __restrict__ lo_in,
    int hc,
    int lr,
    float* __restrict__ lo_out_opt,
    block_q8_1* __restrict__ lo_q8_opt
) {
    const int blk = (int)blockIdx.x;
    const int t = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;
    const int d = blk * 32 + lane;
    const float inv_hc = 1.0f / (float)hc;
    float y = 0.0f;
    if (d < lr) {
        float x = lo_in[(size_t)t * (size_t)lr + (size_t)d] * inv_hc;
        y = x / (1.0f + expf(-x));
        if (lo_out_opt != nullptr) {
            lo_out_opt[(size_t)t * (size_t)lr + (size_t)d] = y;
        }
    }
    if (lo_q8_opt != nullptr && (blk * 32 + 31 < lr)) {
        block_q8_1* dst = lo_q8_opt + (size_t)t * (size_t)(lr >> 5) + blk;
        warp_quantize_q8_1_store(y, dst, lane);
    }
}

__global__ void __launch_bounds__(32) qwen4_hc_mix_q8_kernel(
    const float* __restrict__ xn,
    const float* __restrict__ up_logits,
    int hc,
    int E,
    float* __restrict__ mixed_out,
    block_q8_1* __restrict__ mixed_q8_opt
) {
    const int blk = (int)blockIdx.x;
    const int t = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;
    const int d = blk * 32 + lane;
    float y = 0.0f;
    if (d < E) {
        const size_t t_base = (size_t)t * (size_t)hc * (size_t)E + (size_t)d;
        float sum = 0.0f;
        for (int c = 0; c < hc; ++c) {
            size_t off = t_base + (size_t)c * (size_t)E;
            float w = qwen4_sigmoid(up_logits[off]);
            sum = fmaf(xn[off], w, sum);
        }
        y = (1.0f / (float)hc) * sum;
        if (mixed_out != nullptr) {
            mixed_out[(size_t)t * (size_t)E + (size_t)d] = y;
        }
    }
    if (mixed_q8_opt != nullptr && (blk * 32 + 31 < E)) {
        block_q8_1* dst = mixed_q8_opt + (size_t)t * (size_t)(E >> 5) + blk;
        warp_quantize_q8_1_store(y, dst, lane);
    }
}

__global__ void __launch_bounds__(256) qwen4_hc_inject_dot_kernel(
    const float* __restrict__ xn,
    const void* __restrict__ w_inject,
    int w_type,
    int hc,
    int E,
    float* __restrict__ inject_out
) {
    const int c = (int)blockIdx.x;
    const int t = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int D = hc * E;
    const float* x_t = xn + (size_t)t * (size_t)D;

    __shared__ float s_sums[8];
    float partial = 0.0f;

    if (w_type == GGML_TYPE_F32) {
        const float* w_c = reinterpret_cast<const float*>(w_inject) + (size_t)c * (size_t)D;
        const int D4 = D >> 2;
        if (((reinterpret_cast<uintptr_t>(x_t) | reinterpret_cast<uintptr_t>(w_c)) & 15u) == 0u) {
            const float4* x4 = reinterpret_cast<const float4*>(x_t);
            const float4* w4 = reinterpret_cast<const float4*>(w_c);
            for (int i = tid; i < D4; i += 256) {
                float4 a = x4[i];
                float4 b = w4[i];
                partial += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            for (int d = (D4 << 2) + tid; d < D; d += 256) {
                partial += x_t[d] * w_c[d];
            }
        } else {
            for (int d = tid; d < D; d += 256) {
                partial += x_t[d] * w_c[d];
            }
        }
    } else if (w_type == GGML_TYPE_BF16) {
        const uint16_t* w_c = reinterpret_cast<const uint16_t*>(w_inject) + (size_t)c * (size_t)D;
        const int D8 = D >> 3;
        if (((reinterpret_cast<uintptr_t>(x_t) & 15u) == 0u) && ((reinterpret_cast<uintptr_t>(w_c) & 15u) == 0u)) {
            const float4* x4 = reinterpret_cast<const float4*>(x_t);
            const uint4* w8 = reinterpret_cast<const uint4*>(w_c);
            for (int i = tid; i < D8; i += 256) {
                float4 xlo = x4[2 * i + 0];
                float4 xhi = x4[2 * i + 1];
                uint4 raw = w8[i];
                const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
                float2 f0 = __bfloat1622float2(b2[0]);
                float2 f1 = __bfloat1622float2(b2[1]);
                float2 f2 = __bfloat1622float2(b2[2]);
                float2 f3 = __bfloat1622float2(b2[3]);
                partial += xlo.x * f0.x + xlo.y * f0.y + xlo.z * f1.x + xlo.w * f1.y
                         + xhi.x * f2.x + xhi.y * f2.y + xhi.z * f3.x + xhi.w * f3.y;
            }
            for (int d = (D8 << 3) + tid; d < D; d += 256) {
                partial += x_t[d] * bf16_bits_to_f32(w_c[d]);
            }
        } else {
            for (int d = tid; d < D; d += 256) {
                partial += x_t[d] * bf16_bits_to_f32(w_c[d]);
            }
        }
    }

    float sum = qwen4_block_reduce_sum_8w(partial, s_sums, tid);
    if (tid == 0) {
        inject_out[(size_t)t * (size_t)hc + (size_t)c] = sum;
    }
}

__global__ void qwen4_hc_write_kernel(
    float* __restrict__ R,
    const float* __restrict__ block_out,
    const float* __restrict__ sh_out_opt,
    const float* __restrict__ sh_gate_logit_opt,
    const float* __restrict__ inject,
    int inject_stride,
    int T,
    int hc,
    int E
) {
    const size_t total = (size_t)T * (size_t)hc * (size_t)E;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const size_t tc = idx / (size_t)E;
    const int d = (int)(idx - tc * (size_t)E);
    const int t = (int)(tc / (size_t)hc);
    const int c = (int)(tc - (size_t)t * (size_t)hc);

    const size_t bo_idx = (size_t)t * (size_t)E + (size_t)d;
    float y = (block_out != nullptr) ? block_out[bo_idx] : 0.0f;
    if (sh_out_opt != nullptr) {
        float g = (sh_gate_logit_opt != nullptr) ? qwen4_sigmoid(sh_gate_logit_opt[t]) : 1.0f;
        y += sh_out_opt[bo_idx] * g;
    }
    float inj_val;
    if (inject_stride <= 1) {
        inj_val = inject[(size_t)t * (size_t)hc + (size_t)c];
    } else {
        const float* ip = inject + ((size_t)t * (size_t)hc + (size_t)c) * (size_t)inject_stride;
        float s = 0.0f;
        for (int k = 0; k < hc; ++k) s += ip[k];
        inj_val = s;
    }
    float wc = 2.0f * qwen4_sigmoid(inj_val / (float)hc);
    R[idx] = fmaf(y, wc, R[idx]);
}

__global__ void qwen4_add_streams_broadcast_kernel(
    const float* __restrict__ h,
    const float* __restrict__ e,
    size_t total,
    int hc,
    int E,
    float* __restrict__ R_out
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    size_t tc = idx / (size_t)E;
    size_t d = idx - tc * (size_t)E;
    size_t t = tc / (size_t)hc;
    R_out[idx] = h[idx] + e[t * (size_t)E + d];
}

__global__ void __launch_bounds__(256) qwen4_rmsnorm_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    int cols,
    float eps,
    float* __restrict__ y_out,
    block_q8_1* __restrict__ y_q8_opt
) {
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const float* x_row = x + (size_t)row * (size_t)cols;
    float* y_row = (y_out != nullptr) ? (y_out + (size_t)row * (size_t)cols) : nullptr;
    block_q8_1* q8_row = (y_q8_opt != nullptr) ? (y_q8_opt + (size_t)row * (size_t)(cols >> 5)) : nullptr;

    __shared__ float s_sums[8];
    float partial = 0.0f;
    for (int col = tid; col < cols; col += 256) {
        float v = x_row[col];
        partial += v * v;
    }
    float ss = qwen4_block_reduce_sum_8w(partial, s_sums, tid);
    float scale = rsqrtf(ss / (float)cols + eps);

    for (int base = 0; base < cols; base += 256) {
        int col = base + tid;
        float v = (col < cols) ? (x_row[col] * scale * w[col]) : 0.0f;
        if (y_row != nullptr && col < cols) {
            y_row[col] = v;
        }
        if (q8_row != nullptr && (base + warp * 32 + 31 < cols)) {
            int blk_idx = (base >> 5) + warp;
            warp_quantize_q8_1_store(v, q8_row + blk_idx, lane);
        }
    }
}

// ============================================================================
// 2b. PLE Post-Ops
// ============================================================================

__global__ void __launch_bounds__(256) qwen4_ple_grouped_rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_rep,
    float* __restrict__ y,
    int N,
    int H,
    float eps
) {
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const float* xr = x + (size_t)row * (size_t)N;
    const float* wr = w_rep + (size_t)(row % H) * (size_t)N;
    float* yr = y + (size_t)row * (size_t)N;

    __shared__ float s_sums[8];
    float partial = 0.0f;
    for (int d = tid; d < N; d += 256) {
        float v = xr[d];
        partial += v * v;
    }
    float ss = qwen4_block_reduce_sum_8w(partial, s_sums, tid);
    float scale = rsqrtf(ss / (float)N + eps);
    for (int d = tid; d < N; d += 256) {
        yr[d] = xr[d] * scale * wr[d];
    }
}

__global__ void __launch_bounds__(256) qwen4_ple_gate_kernel(
    const float* __restrict__ key,
    const float* __restrict__ query,
    float* __restrict__ gate,
    int N
) {
    const int row = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const float* kr = key + (size_t)row * (size_t)N;
    const float* qr = query + (size_t)row * (size_t)N;

    __shared__ float s_sums[8];
    float partial = 0.0f;
    for (int d = tid; d < N; d += 256) {
        partial += kr[d] * qr[d];
    }
    float dot = qwen4_block_reduce_sum_8w(partial, s_sums, tid);
    if (tid == 0) {
        float s = dot * rsqrtf((float)N);
        float sign = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
        float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        gate[row] = qwen4_sigmoid(sign * mag);
    }
}

__global__ void qwen4_ple_broadcast_mul_kernel(
    const float* __restrict__ value,
    const float* __restrict__ gate,
    float* __restrict__ gated,
    int T,
    int N,
    int H
) {
    const int D = N * H;
    const size_t total = (size_t)T * (size_t)D;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int t = (int)(idx / (size_t)D);
    const int rem = (int)(idx - (size_t)t * (size_t)D);
    const int c = rem / N;
    const int d = rem - c * N;
    gated[idx] = value[(size_t)t * (size_t)N + (size_t)d] * gate[(size_t)t * (size_t)H + (size_t)c];
}

__global__ void qwen4_ple_conv_residual_kernel(
    const float* __restrict__ history,
    const float* __restrict__ normalized,
    const float* __restrict__ w_conv1d_f32,
    float* __restrict__ hidden_R,
    const float* __restrict__ gated,
    int T,
    int D
) {
    const size_t total = (size_t)T * (size_t)D;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int t = (int)(idx / (size_t)D);
    const int ch = (int)(idx - (size_t)t * (size_t)D);

    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int p = t - 9 + 3 * k;
        float x = (p >= 0)
            ? normalized[(size_t)p * (size_t)D + (size_t)ch]
            : history[(size_t)ch * 9 + (size_t)(9 + p)];
        sum += x * w_conv1d_f32[(size_t)ch * 4 + (size_t)k];
    }
    float act = sum / (1.0f + expf(-sum));
    hidden_R[idx] += gated[idx] + act;
}

__global__ void qwen4_ple_history_advance_kernel(
    float* __restrict__ history,
    const float* __restrict__ normalized,
    int T,
    int D,
    float* __restrict__ snap_opt
) {
    const int ch = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (ch >= D) return;

    float h0[9];
    #pragma unroll
    for (int r = 0; r < 9; ++r) {
        h0[r] = history[(size_t)ch * 9 + (size_t)r];
    }

    if (snap_opt != nullptr) {
        for (int t = 0; t < T; ++t) {
            float* dst = snap_opt + ((size_t)t * (size_t)D + (size_t)ch) * 9;
            #pragma unroll
            for (int r = 0; r < 9; ++r) {
                int p = t - 8 + r;
                dst[r] = (p >= 0) ? normalized[(size_t)p * (size_t)D + (size_t)ch] : h0[9 + p];
            }
        }
    }

    float* hdst = history + (size_t)ch * 9;
    #pragma unroll
    for (int r = 0; r < 9; ++r) {
        int p = T - 9 + r;
        hdst[r] = (p >= 0) ? normalized[(size_t)p * (size_t)D + (size_t)ch] : h0[9 + p];
    }
}

// ============================================================================
// 2c. GDN Linear Attention
// ============================================================================

__global__ void __launch_bounds__(256) qwen4_gdn_ab_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w_alpha,
    const float* __restrict__ w_beta,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    int n_embd,
    int h_v
) {
    const int row = (int)blockIdx.x * 8 + (int)threadIdx.y;
    const int t = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;
    if (row >= 2 * h_v) return;

    const bool is_beta = (row >= h_v);
    const int head = is_beta ? (row - h_v) : row;
    const float* xr = x + (size_t)t * (size_t)n_embd;
    const float* wr = (is_beta ? w_beta : w_alpha) + (size_t)head * (size_t)n_embd;

    float acc = 0.0f;
    const int n4 = n_embd >> 2;
    if (((reinterpret_cast<uintptr_t>(xr) | reinterpret_cast<uintptr_t>(wr)) & 15u) == 0u) {
        const float4* x4 = reinterpret_cast<const float4*>(xr);
        const float4* w4 = reinterpret_cast<const float4*>(wr);
        for (int i = lane; i < n4; i += 32) {
            float4 a = x4[i];
            float4 b = w4[i];
            acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        for (int d = (n4 << 2) + lane; d < n_embd; d += 32) {
            acc += xr[d] * wr[d];
        }
    } else {
        for (int d = lane; d < n_embd; d += 32) {
            acc += xr[d] * wr[d];
        }
    }
    acc = warp_reduce_sum(acc);

    if (lane == 0) {
        const size_t out_idx = (size_t)t * (size_t)h_v + (size_t)head;
        if (is_beta) {
            beta_out[out_idx] = qwen4_sigmoid(acc);
        } else {
            float v = acc + dt_bias[head];
            float sp = (v > 20.0f) ? v : log1pf(expf(v));
            gate_out[out_idx] = sp * ssm_a[head];
        }
    }
}

__global__ void qwen4_gdn_gate_beta_postop_kernel(
    const float* __restrict__ alpha_raw,
    const float* __restrict__ beta_raw,
    const float* __restrict__ dt_bias,
    const float* __restrict__ ssm_a,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    size_t total,
    int h_v
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int h = (int)(idx % (size_t)h_v);
    float v = alpha_raw[idx] + dt_bias[h];
    float sp = (v > 20.0f) ? v : log1pf(expf(v));
    gate_out[idx] = sp * ssm_a[h];
    beta_out[idx] = qwen4_sigmoid(beta_raw[idx]);
}

__global__ void __launch_bounds__(128) qwen4_gdn_conv_l2_t1_kernel(
    float* __restrict__ conv_hist,
    const float* __restrict__ qkv,
    const float* __restrict__ conv_w,
    float* __restrict__ h_out,
    int qk_heads,
    float eps,
    float* __restrict__ hist_snap_opt
) {
    const int c = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int tid = (int)threadIdx.x;
    __shared__ float s_part[4];

    const float v0 = conv_hist[(size_t)c * 3 + 0];
    const float v1 = conv_hist[(size_t)c * 3 + 1];
    const float v2 = conv_hist[(size_t)c * 3 + 2];
    const float x = qkv[c];

    const float* wc = conv_w + (size_t)c * 4;
    float sum = v0 * wc[0] + v1 * wc[1] + v2 * wc[2] + x * wc[3];
    conv_hist[(size_t)c * 3 + 0] = v1;
    conv_hist[(size_t)c * 3 + 1] = v2;
    conv_hist[(size_t)c * 3 + 2] = x;
    if (hist_snap_opt != nullptr) {
        hist_snap_opt[(size_t)c * 3 + 0] = v1;
        hist_snap_opt[(size_t)c * 3 + 1] = v2;
        hist_snap_opt[(size_t)c * 3 + 2] = x;
    }

    float y = sum / (1.0f + expf(-sum));
    if ((int)blockIdx.x < qk_heads) {
        float ss = qwen4_block_reduce_sum_4w(y * y, s_part, tid);
        y *= rsqrtf(ss + eps);
    }
    h_out[c] = y;
}

__global__ void __launch_bounds__(128) qwen4_gdn_conv_l2_batch_kernel(
    const float* __restrict__ conv_hist,
    const float* __restrict__ qkv,
    const float* __restrict__ conv_w,
    float* __restrict__ h_out,
    int channels,
    int qk_heads,
    float eps
) {
    const int c = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int t = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    __shared__ float s_part[4];

    const float* wc = conv_w + (size_t)c * 4;
    float sum = 0.0f;
    #pragma unroll
    for (int tap = 0; tap < 4; ++tap) {
        int p = t - 3 + tap;
        float x = (p >= 0)
            ? qkv[(size_t)p * (size_t)channels + (size_t)c]
            : conv_hist[(size_t)c * 3 + (size_t)(3 + p)];
        sum += x * wc[tap];
    }

    float y = sum / (1.0f + expf(-sum));
    if ((int)blockIdx.x < qk_heads) {
        float ss = qwen4_block_reduce_sum_4w(y * y, s_part, tid);
        y *= rsqrtf(ss + eps);
    }
    h_out[(size_t)t * (size_t)channels + (size_t)c] = y;
}

__global__ void qwen4_gdn_conv_hist_advance_kernel(
    float* __restrict__ conv_hist,
    const float* __restrict__ qkv,
    int T,
    int channels,
    float* __restrict__ hist_snap_opt
) {
    const int c = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (c >= channels) return;

    float h0[3];
    h0[0] = conv_hist[(size_t)c * 3 + 0];
    h0[1] = conv_hist[(size_t)c * 3 + 1];
    h0[2] = conv_hist[(size_t)c * 3 + 2];

    if (hist_snap_opt != nullptr) {
        for (int t = 0; t < T; ++t) {
            float* dst = hist_snap_opt + ((size_t)t * (size_t)channels + (size_t)c) * 3;
            #pragma unroll
            for (int r = 0; r < 3; ++r) {
                int p = t - 2 + r;
                dst[r] = (p >= 0) ? qkv[(size_t)p * (size_t)channels + (size_t)c] : h0[3 + p];
            }
        }
    }

    float* hdst = conv_hist + (size_t)c * 3;
    #pragma unroll
    for (int r = 0; r < 3; ++r) {
        int p = T - 3 + r;
        hdst[r] = (p >= 0) ? qkv[(size_t)p * (size_t)channels + (size_t)c] : h0[3 + p];
    }
}

template <bool MULTI_STEP>
__global__ void __launch_bounds__(512, 1) qwen4_gdn_steps_norm_q8_kernel(
    float* __restrict__ state,
    const float* __restrict__ h_qkv,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    const float* __restrict__ z_gate,
    const float* __restrict__ ssm_norm_w,
    float eps,
    float* __restrict__ y_f32_opt,
    block_q8_1* __restrict__ y_q8_opt,
    float* __restrict__ state_snap_opt,
    int T,
    int h_k,
    int h_v
) {
    constexpr int S = 128;
    constexpr int RG = 4;
    constexpr int RPG = 32;

    const int head = (int)blockIdx.x;
    const int col = (int)threadIdx.x; // 0..127
    const int rg = (int)threadIdx.y;  // 0..3
    const int tid = rg * S + col;
    const int qh = head % h_k;
    const int C = (2 * h_k + h_v) * S;

    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[4];

    float s[RPG];
    float* st_base = state + ((size_t)head * S + (size_t)(rg * RPG)) * S + (size_t)col;
    #pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = st_base[(size_t)r * S];
    }

    const float gamma_col = (rg == 0) ? ssm_norm_w[col] : 0.0f;
    const float out_scale = rsqrtf((float)S);

    const int n_steps = MULTI_STEP ? T : 1;
    for (int t = 0; t < n_steps; ++t) {
        const float* h_t = h_qkv + (size_t)t * (size_t)C;
        if (tid < S) {
            sq[tid] = h_t[(size_t)qh * S + (size_t)tid];
            sk[tid] = h_t[(size_t)(h_k + qh) * S + (size_t)tid];
        }
        __syncthreads();

        const float g = expf(gate[(size_t)t * (size_t)h_v + (size_t)head]);
        const float b = beta[(size_t)t * (size_t)h_v + (size_t)head];
        const float v_in = h_t[(size_t)(2 * h_k + head) * S + (size_t)col];

        float kv = 0.0f;
        #pragma unroll
        for (int r = 0; r < RPG; ++r) {
            kv = fmaf(s[r], sk[rg * RPG + r], kv);
        }
        red_kv[rg][col] = kv;
        __syncthreads();

        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (v_in - g * kv_col) * b;

        float o = 0.0f;
        #pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
            if constexpr (MULTI_STEP) {
                if (state_snap_opt != nullptr && t + 1 < n_steps) {
                    __stcs(&state_snap_opt[(((size_t)t * (size_t)h_v + (size_t)head) * S + (size_t)(rg * RPG + r)) * S + (size_t)col], s[r]);
                }
            }
        }
        red_o[rg][col] = o;
        __syncthreads();

        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * out_scale;
            sq_part = oc * oc;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sq_part += __shfl_xor_sync(0xffffffffu, sq_part, off);
        }
        if ((tid & 31) == 0 && rg == 0) {
            wsum[tid >> 5] = sq_part;
        }
        __syncthreads();

        if (rg == 0) {
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float inv_rms = rsqrtf(ss / (float)S + eps);
            const size_t y_idx = ((size_t)t * (size_t)h_v + (size_t)head) * S + (size_t)col;
            const float zz = z_gate[y_idx];
            const float yv = oc * inv_rms * gamma_col * qwen4_sigmoid(zz);
            if (y_f32_opt != nullptr) {
                y_f32_opt[y_idx] = yv;
            }
            if (y_q8_opt != nullptr) {
                block_q8_1* dst_blk = y_q8_opt + ((size_t)t * (size_t)h_v + (size_t)head) * 4 + (col >> 5);
                warp_quantize_q8_1_store(yv, dst_blk, col & 31);
            }
        }
        if constexpr (MULTI_STEP) {
            if (t + 1 < n_steps) __syncthreads();
        }
    }

    #pragma unroll
    for (int r = 0; r < RPG; ++r) {
        st_base[(size_t)r * S] = s[r];
    }
}

template <bool MULTI_STEP, bool SAVE_SNAP>
__global__ void __launch_bounds__(128, 2) qwen4_gdn_steps_split_kernel(
    float* __restrict__ state,
    const float* __restrict__ h_qkv,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ y_ws_f32,
    float* __restrict__ state_before_opt,
    int T,
    int h_k,
    int h_v
) {
    constexpr int S = 128;
    constexpr int RG = 4;
    constexpr int RPG = 32;

    const int head = (int)blockIdx.x;
    const int col_blk = (int)blockIdx.y; // 0..3
    const int lc = (int)threadIdx.x;      // 0..31
    const int rg = (int)threadIdx.y;      // 0..3
    const int col = col_blk * 32 + lc;    // 0..127
    const int tid = rg * 32 + lc;         // 0..127
    const int qh = head % h_k;
    const int C = (2 * h_k + h_v) * S;

    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][32];
    __shared__ float red_o[RG][32];

    float s[RPG];
    float* st_base = state + ((size_t)head * S + (size_t)(rg * RPG)) * S + (size_t)col;
    #pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = st_base[(size_t)r * S];
    }

    if constexpr (SAVE_SNAP) {
        if (state_before_opt != nullptr) {
            float* sb_base = state_before_opt + ((size_t)head * S + (size_t)(rg * RPG)) * S + (size_t)col;
            #pragma unroll
            for (int r = 0; r < RPG; ++r) {
                __stcs(&sb_base[(size_t)r * S], s[r]);
            }
        }
    }

    const float out_scale = rsqrtf((float)S);
    const int n_steps = MULTI_STEP ? T : 1;
    for (int t = 0; t < n_steps; ++t) {
        const float* h_t = h_qkv + (size_t)t * (size_t)C;
        sq[tid] = h_t[(size_t)qh * S + (size_t)tid];
        sk[tid] = h_t[(size_t)(h_k + qh) * S + (size_t)tid];
        __syncthreads();

        const float g = expf(gate[(size_t)t * (size_t)h_v + (size_t)head]);
        const float b = beta[(size_t)t * (size_t)h_v + (size_t)head];
        const float v_in = h_t[(size_t)(2 * h_k + head) * S + (size_t)col];

        float kv = 0.0f;
        #pragma unroll
        for (int r = 0; r < RPG; ++r) {
            kv = fmaf(s[r], sk[rg * RPG + r], kv);
        }
        red_kv[rg][lc] = kv;
        __syncthreads();

        const float kv_col = red_kv[0][lc] + red_kv[1][lc] + red_kv[2][lc] + red_kv[3][lc];
        const float delta = (v_in - g * kv_col) * b;

        float o = 0.0f;
        #pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][lc] = o;
        __syncthreads();

        if (rg == 0) {
            const float oc = (red_o[0][lc] + red_o[1][lc] + red_o[2][lc] + red_o[3][lc]) * out_scale;
            y_ws_f32[((size_t)t * (size_t)h_v + (size_t)head) * S + (size_t)col] = oc;
        }
        if constexpr (MULTI_STEP) {
            if (t + 1 < n_steps) __syncthreads();
        }
    }

    #pragma unroll
    for (int r = 0; r < RPG; ++r) {
        st_base[(size_t)r * S] = s[r];
    }
}

__global__ void __launch_bounds__(128) qwen4_gdn_norm_gate_q8_kernel(
    const float* __restrict__ y_raw,
    const float* __restrict__ z_gate,
    const float* __restrict__ ssm_norm_w,
    float eps,
    float* __restrict__ y_f32_opt,
    block_q8_1* __restrict__ y_q8_opt,
    int h_v
) {
    constexpr int S = 128;
    const int head = (int)blockIdx.x;
    const int t = (int)blockIdx.y;
    const int col = (int)threadIdx.x; // 0..127
    const size_t y_idx = ((size_t)t * (size_t)h_v + (size_t)head) * S + (size_t)col;

    const float oc = y_raw[y_idx];
    float sq_part = oc * oc;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq_part += __shfl_xor_sync(0xffffffffu, sq_part, off);
    }
    __shared__ float wsum[4];
    if ((col & 31) == 0) {
        wsum[col >> 5] = sq_part;
    }
    __syncthreads();

    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
    const float inv_rms = rsqrtf(ss / (float)S + eps);
    const float yv = oc * inv_rms * ssm_norm_w[col] * qwen4_sigmoid(z_gate[y_idx]);
    if (y_f32_opt != nullptr) {
        y_f32_opt[y_idx] = yv;
    }
    if (y_q8_opt != nullptr) {
        block_q8_1* dst_blk = y_q8_opt + ((size_t)t * (size_t)h_v + (size_t)head) * 4 + (col >> 5);
        warp_quantize_q8_1_store(yv, dst_blk, col & 31);
    }
}

__global__ void __launch_bounds__(128, 2) qwen4_gdn_replay_steps_kernel(
    const float* __restrict__ state_before,
    float* __restrict__ state_out,
    const float* __restrict__ h_qkv,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    int n_steps,
    int h_k,
    int h_v
) {
    constexpr int S = 128;
    constexpr int RG = 4;
    constexpr int RPG = 32;

    const int head = (int)blockIdx.x;
    const int col_blk = (int)blockIdx.y;
    const int lc = (int)threadIdx.x;
    const int rg = (int)threadIdx.y;
    const int col = col_blk * 32 + lc;
    const int tid = rg * 32 + lc;
    const int qh = head % h_k;
    const int C = (2 * h_k + h_v) * S;

    __shared__ float sk[S];
    __shared__ float red_kv[RG][32];

    float s[RPG];
    const float* sb_base = state_before + ((size_t)head * S + (size_t)(rg * RPG)) * S + (size_t)col;
    #pragma unroll
    for (int r = 0; r < RPG; ++r) {
        s[r] = sb_base[(size_t)r * S];
    }

    for (int t = 0; t < n_steps; ++t) {
        const float* h_t = h_qkv + (size_t)t * (size_t)C;
        sk[tid] = h_t[(size_t)(h_k + qh) * S + (size_t)tid];
        __syncthreads();

        const float g = expf(gate[(size_t)t * (size_t)h_v + (size_t)head]);
        const float b = beta[(size_t)t * (size_t)h_v + (size_t)head];
        const float v_in = h_t[(size_t)(2 * h_k + head) * S + (size_t)col];

        float kv = 0.0f;
        #pragma unroll
        for (int r = 0; r < RPG; ++r) {
            kv = fmaf(s[r], sk[rg * RPG + r], kv);
        }
        red_kv[rg][lc] = kv;
        __syncthreads();

        const float kv_col = red_kv[0][lc] + red_kv[1][lc] + red_kv[2][lc] + red_kv[3][lc];
        const float delta = (v_in - g * kv_col) * b;

        #pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
        }
        if (t + 1 < n_steps) __syncthreads();
    }

    float* dst_base = state_out + ((size_t)head * S + (size_t)(rg * RPG)) * S + (size_t)col;
    #pragma unroll
    for (int r = 0; r < RPG; ++r) {
        dst_base[(size_t)r * S] = s[r];
    }
}

// --- Chunked WY Tensor-Core GDN Prefill (adapted from Strata's gdn_tc.cuh for head-major [h_v, 128, 128]) ---

constexpr int GTC_S = 128;
constexpr int GTC_GCH = 32;
constexpr int GTC_GSB = 2048;
constexpr int GTC_NB = GTC_GSB / GTC_GCH; // 64
constexpr int GTC_KP = GTC_S + 8;       // 136
constexpr int GTC_VP = GTC_S + 4;       // 132
constexpr int GTC_PP = GTC_GCH + 8;     // 40
constexpr int GTC_GTP = GTC_GCH + 4;    // 36
constexpr int GTC_CW = GTC_S / 16;      // 8

constexpr size_t kGtcScanSmemSplit =
    2 * (4 * GTC_GCH * GTC_KP * 2 + GTC_GCH * 16) + GTC_GCH * GTC_VP * 4 + 4 * GTC_GCH * GTC_PP * 2 + 16;
constexpr size_t kGtcScanSmemSplitNorm = kGtcScanSmemSplit + GTC_CW * GTC_GCH * 4; // 98,832 B
static_assert(kGtcScanSmemSplitNorm == 98832, "qwen4 gdn_tc SMEM budget");

struct Qwen4GtcScratch {
    uint16_t* ti;
    float* te;
    float4* sc;
    uint16_t* kq;
    uint16_t* pi;
};

__host__ __device__ __forceinline__ size_t align_up_256(size_t n) {
    return (n + 255u) & ~size_t(255u);
}

__host__ __device__ __forceinline__ Qwen4GtcScratch make_gtc_scratch(void* base, int h_k, int h_v) {
    const size_t ti_b = align_up_256((size_t)GTC_NB * h_v * 2 * GTC_GCH * GTC_GCH * 2);
    const size_t te_b = align_up_256((size_t)GTC_NB * h_v * 4);
    const size_t sc_b = align_up_256((size_t)GTC_NB * h_v * GTC_GCH * 16);
    const size_t kq_b = align_up_256((size_t)GTC_NB * h_k * 4 * GTC_GCH * GTC_S * 2);
    uint8_t* p = reinterpret_cast<uint8_t*>(base);
    Qwen4GtcScratch r;
    r.ti = reinterpret_cast<uint16_t*>(p);
    r.te = reinterpret_cast<float*>(p + ti_b);
    r.sc = reinterpret_cast<float4*>(p + ti_b + te_b);
    r.kq = reinterpret_cast<uint16_t*>(p + ti_b + te_b + sc_b);
    r.pi = reinterpret_cast<uint16_t*>(p + ti_b + te_b + sc_b + kq_b);
    return r;
}

__device__ __forceinline__ float gtc_warp_max(float v) {
#if QWEN4_SM80
    return __uint_as_float(__reduce_max_sync(0xffffffffu, __float_as_uint(v)));
#else
    return warp_reduce_max(v);
#endif
}

__device__ __forceinline__ float gtc_pow2(int e) {
    return __int_as_float((e + 127) << 23);
}

__device__ __forceinline__ int gtc_scale_exp(float m) {
    if (m == 0.0f) return 0;
    const int e = 14 - ((__float_as_int(m) >> 23) - 127);
    return (e < -40) ? -40 : ((e > 40) ? 40 : e);
}

__device__ __forceinline__ void gtc_ldsm_x4(uint32_t (&r)[4], const void* smem) {
#if QWEN4_SM80
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned)__cvta_generic_to_shared(smem)));
#else
    r[0] = r[1] = r[2] = r[3] = 0u;
    (void)smem;
#endif
}

__device__ __forceinline__ void gtc_ldsm_x4_trans(uint32_t (&r)[4], const void* smem) {
#if QWEN4_SM80
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"((unsigned)__cvta_generic_to_shared(smem)));
#else
    r[0] = r[1] = r[2] = r[3] = 0u;
    (void)smem;
#endif
}

__device__ __forceinline__ void gtc_mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#if QWEN4_SM80
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    (void)c; (void)a; (void)b0; (void)b1;
#endif
}

__device__ __forceinline__ void gtc_mma3_f16(
    float (&c)[4],
    const uint32_t (&ah)[4],
    const uint32_t (&al)[4],
    uint32_t bh0, uint32_t bh1,
    uint32_t bl0, uint32_t bl1
) {
    gtc_mma_f16(c, al, bh0, bh1);
    gtc_mma_f16(c, ah, bl0, bl1);
    gtc_mma_f16(c, ah, bh0, bh1);
}

__device__ __forceinline__ void gtc_split_f16x2(float x0, float x1, uint32_t& hi, uint32_t& lo) {
    union { __half2 h; uint32_t u; } cv;
    cv.h = __floats2half2_rn(x0, x1);
    hi = cv.u;
    const float2 r = __half22float2(cv.h);
    cv.h = __floats2half2_rn(x0 - r.x, x1 - r.y);
    lo = cv.u;
}

__device__ __forceinline__ void gtc_split_store4(uint16_t* hi, uint16_t* lo, float4 v, float scale) {
    uint32_t h0, l0, h1, l1;
    gtc_split_f16x2(v.x * scale, v.y * scale, h0, l0);
    gtc_split_f16x2(v.z * scale, v.w * scale, h1, l1);
    *reinterpret_cast<uint2*>(hi) = make_uint2(h0, h1);
    *reinterpret_cast<uint2*>(lo) = make_uint2(l0, l1);
}

__device__ __forceinline__ float4 gtc_lds128(const float4* p) {
    float4 v;
#if QWEN4_SM80
    asm volatile("ld.shared.v4.f32 {%0,%1,%2,%3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "r"((unsigned)__cvta_generic_to_shared(p)));
#else
    v = *p;
#endif
    return v;
}

__device__ __forceinline__ void gtc_cp16(void* smem_dst, const void* glob_src) {
#if QWEN4_SM80
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(smem_dst)), "l"(glob_src));
#else
    *reinterpret_cast<uint4*>(smem_dst) = *reinterpret_cast<const uint4*>(glob_src);
#endif
}

__device__ __forceinline__ void gtc_cp4(void* smem_dst, const void* glob_src) {
#if QWEN4_SM80
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                 :: "r"((unsigned)__cvta_generic_to_shared(smem_dst)), "l"(glob_src));
#else
    *reinterpret_cast<uint32_t*>(smem_dst) = *reinterpret_cast<const uint32_t*>(glob_src);
#endif
}

__device__ __forceinline__ void gtc_cp_commit() {
#if QWEN4_SM80
    asm volatile("cp.async.commit_group;\n" :::);
#endif
}

__device__ __forceinline__ void gtc_cp_wait_all() {
#if QWEN4_SM80
    asm volatile("cp.async.wait_all;\n" :::);
#endif
}

__device__ __forceinline__ void gtc_prod_state(
    const float (&s)[16][4],
    float sA,
    const uint16_t* kh,
    const uint16_t* kl,
    const uint16_t* qh,
    const uint16_t* ql,
    float (&xt)[4][4],
    float (&qs)[4][4]
) {
    const int lane = (int)threadIdx.x & 31;
    const int off = (8 * (lane >> 4) + (lane & 7)) * GTC_KP + 8 * ((lane >> 3) & 1);
    #pragma unroll
    for (int kk = 0; kk < GTC_S / 16; ++kk) {
        uint32_t ah[4], al[4];
        gtc_split_f16x2(s[2 * kk][0] * sA, s[2 * kk][1] * sA, ah[0], al[0]);
        gtc_split_f16x2(s[2 * kk][2] * sA, s[2 * kk][3] * sA, ah[1], al[1]);
        gtc_split_f16x2(s[2 * kk + 1][0] * sA, s[2 * kk + 1][1] * sA, ah[2], al[2]);
        gtc_split_f16x2(s[2 * kk + 1][2] * sA, s[2 * kk + 1][3] * sA, ah[3], al[3]);
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            const int o = off + 16 * hf * GTC_KP + 16 * kk;
            uint32_t bh[4], bl[4];
            gtc_ldsm_x4(bh, kh + o);
            gtc_ldsm_x4(bl, kl + o);
            gtc_mma3_f16(xt[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(xt[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
            gtc_ldsm_x4(bh, qh + o);
            gtc_ldsm_x4(bl, ql + o);
            gtc_mma3_f16(qs[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(qs[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}

__device__ __forceinline__ void gtc_prod_dt(
    const float (&dt)[4][4],
    float sD,
    const uint16_t* ph,
    const uint16_t* pl,
    float (&ot)[4][4]
) {
    const int lane = (int)threadIdx.x & 31;
    const int off = (8 * (lane >> 4) + (lane & 7)) * GTC_PP + 8 * ((lane >> 3) & 1);
    #pragma unroll
    for (int ks = 0; ks < GTC_GCH / 16; ++ks) {
        uint32_t ah[4], al[4];
        gtc_split_f16x2(dt[2 * ks][0] * sD, dt[2 * ks][1] * sD, ah[0], al[0]);
        gtc_split_f16x2(dt[2 * ks][2] * sD, dt[2 * ks][3] * sD, ah[1], al[1]);
        gtc_split_f16x2(dt[2 * ks + 1][0] * sD, dt[2 * ks + 1][1] * sD, ah[2], al[2]);
        gtc_split_f16x2(dt[2 * ks + 1][2] * sD, dt[2 * ks + 1][3] * sD, ah[3], al[3]);
        #pragma unroll
        for (int hf = 0; hf < 2; ++hf) {
            if (ks == 1 && hf == 0) continue;
            const int o = off + 16 * hf * GTC_PP + 16 * ks;
            uint32_t bh[4], bl[4];
            gtc_ldsm_x4(bh, ph + o);
            gtc_ldsm_x4(bl, pl + o);
            gtc_mma3_f16(ot[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(ot[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}

__device__ __forceinline__ void gtc_update_state(
    float (&s)[16][4],
    const float (&dd)[4][4],
    const uint16_t* kh,
    const uint16_t* kl
) {
    const int lane = (int)threadIdx.x & 31;
    const int off = (8 * ((lane >> 3) & 1) + (lane & 7)) * GTC_KP + 8 * (lane >> 4);
    #pragma unroll
    for (int ks = 0; ks < GTC_GCH / 16; ++ks) {
        uint32_t ah[4], al[4];
        gtc_split_f16x2(dd[2 * ks][0], dd[2 * ks][1], ah[0], al[0]);
        gtc_split_f16x2(dd[2 * ks][2], dd[2 * ks][3], ah[1], al[1]);
        gtc_split_f16x2(dd[2 * ks + 1][0], dd[2 * ks + 1][1], ah[2], al[2]);
        gtc_split_f16x2(dd[2 * ks + 1][2], dd[2 * ks + 1][3], ah[3], al[3]);
        #pragma unroll
        for (int hf = 0; hf < GTC_S / 16; ++hf) {
            const int o = off + 16 * ks * GTC_KP + 16 * hf;
            uint32_t bh[4], bl[4];
            gtc_ldsm_x4_trans(bh, kh + o);
            gtc_ldsm_x4_trans(bl, kl + o);
            gtc_mma3_f16(s[2 * hf], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(s[2 * hf + 1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
    }
}

__global__ void __launch_bounds__(256) qwen4_gdn_tc_prep_kernel(
    const float* __restrict__ h,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    uint16_t* __restrict__ ti,
    float* __restrict__ te,
    float4* __restrict__ sc,
    uint16_t* __restrict__ kq,
    uint16_t* __restrict__ pi,
    int64_t t0,
    int64_t T,
    int h_k,
    int h_v
) {
    extern __shared__ __align__(16) float gsm[];
    uint16_t* im = reinterpret_cast<uint16_t*>(gsm); // [4][GCH][GTC_KP]
    float* skk = gsm + 4 * GTC_GCH * GTC_KP / 2;     // [GCH][GTC_GTP]
    float* sqk = skk + GTC_GCH * GTC_GTP;            // [GCH][GTC_GTP]
    float* sa = gsm;                                 // [vpk][GCH][GTC_GTP]

    const int qh = (int)blockIdx.y;
    const int ch = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int vpk = h_v / h_k;
    const int C = (2 * h_k + h_v) * GTC_S;
    const int64_t tb = t0 + (int64_t)ch * GTC_GCH;
    const int n = (int)min((int64_t)GTC_GCH, T - tb);

    {
        constexpr int NP = GTC_GCH * GTC_S / 4 / 256; // 4 float4/thread
        float4 kv[NP], qv[NP];
        #pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m;
            const int t = p / (GTC_S / 4);
            const int r = (p % (GTC_S / 4)) * 4;
            kv[m] = (t < n) ? *reinterpret_cast<const float4*>(h + (tb + t) * C + (size_t)(h_k + qh) * GTC_S + r)
                            : make_float4(0.f, 0.f, 0.f, 0.f);
            qv[m] = (t < n) ? *reinterpret_cast<const float4*>(h + (tb + t) * C + (size_t)qh * GTC_S + r)
                            : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        uint16_t* img = kq + ((size_t)ch * (size_t)h_k + (size_t)qh) * 4 * GTC_GCH * GTC_S;
        #pragma unroll
        for (int m = 0; m < NP; ++m) {
            const int p = tid + 256 * m;
            const int t = p / (GTC_S / 4);
            const int r = (p % (GTC_S / 4)) * 4;
            const int o = t * GTC_KP + r;
            gtc_split_store4(im + o, im + GTC_GCH * GTC_KP + o, kv[m], 256.0f);
            gtc_split_store4(im + 2 * GTC_GCH * GTC_KP + o, im + 3 * GTC_GCH * GTC_KP + o, qv[m], 256.0f);
            gtc_split_store4(img + t * GTC_S + r, img + GTC_GCH * GTC_S + t * GTC_S + r, kv[m], 256.0f);
            gtc_split_store4(img + 2 * GTC_GCH * GTC_S + t * GTC_S + r, img + 3 * GTC_GCH * GTC_S + t * GTC_S + r, qv[m], 256.0f);
        }
    }
    __syncthreads();

    {
        const int pr = warp >> 2;
        const int mt = (warp >> 1) & 1;
        const int np = warp & 1;
        const uint16_t* ah_ = im + 2 * pr * GTC_GCH * GTC_KP + (16 * mt + (lane & 7) + 8 * ((lane >> 3) & 1)) * GTC_KP + 8 * (lane >> 4);
        const uint16_t* bh_ = im + (16 * np + 8 * (lane >> 4) + (lane & 7)) * GTC_KP + 8 * ((lane >> 3) & 1);
        float c[2][4] = {};
        #pragma unroll
        for (int kk = 0; kk < GTC_S / 16; ++kk) {
            uint32_t ah[4], al[4], bh[4], bl[4];
            gtc_ldsm_x4(ah, ah_ + 16 * kk);
            gtc_ldsm_x4(al, ah_ + GTC_GCH * GTC_KP + 16 * kk);
            gtc_ldsm_x4(bh, bh_ + 16 * kk);
            gtc_ldsm_x4(bl, bh_ + GTC_GCH * GTC_KP + 16 * kk);
            gtc_mma3_f16(c[0], ah, al, bh[0], bh[1], bl[0], bl[1]);
            gtc_mma3_f16(c[1], ah, al, bh[2], bh[3], bl[2], bl[3]);
        }
        float* g = pr ? sqk : skk;
        const int gid = lane >> 2;
        const int tig = lane & 3;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int t = 16 * mt + gid + 8 * (q >> 1);
                const int i = 16 * np + 8 * j + 2 * tig + (q & 1);
                g[t * GTC_GTP + i] = (i <= t) ? (c[j][q] * gtc_pow2(-16)) : 0.0f;
            }
        }
    }
    __syncthreads();

    if (warp >= 2 * vpk) return;

    const int j = warp % vpk;
    const int vh = qh + j * h_k;
    const int i = lane;
    const bool pv = (i < n);
    const float b = pv ? beta[(tb + i) * (int64_t)h_v + vh] : 0.0f;
    float gam = pv ? gate[(tb + i) * (int64_t)h_v + vh] : 0.0f;

    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float u = __shfl_up_sync(0xffffffffu, gam, o);
        if (lane >= o) gam += u;
    }

    if (warp >= vpk) {
        uint16_t* Ph = pi + ((int64_t)ch * h_v + vh) * 2 * GTC_GCH * GTC_GCH;
        uint16_t* Pl = Ph + GTC_GCH * GTC_GCH;
        #pragma unroll
        for (int t = 0; t < GTC_GCH; ++t) {
            const float gt = __shfl_sync(0xffffffffu, gam, t);
            const float p = (i <= t ? expf(gt - gam) * sqk[t * GTC_GTP + i] : 0.0f) * 256.0f;
            const __half hi = __float2half_rn(p);
            Ph[t * GTC_GCH + i] = __half_as_ushort(hi);
            Pl[t * GTC_GCH + i] = __half_as_ushort(__float2half_rn(p - __half2float(hi)));
        }
        const float gL = __shfl_sync(0xffffffffu, gam, GTC_GCH - 1);
        sc[((int64_t)ch * h_v + vh) * GTC_GCH + i] = make_float4(b, expf(gam), expf(gL - gam), gam);
        return;
    }

    float* a = sa + j * GTC_GCH * GTC_GTP;
    #pragma unroll
    for (int t = 0; t < GTC_GCH; ++t) {
        const float gt = __shfl_sync(0xffffffffu, gam, t);
        const float bt = __shfl_sync(0xffffffffu, b, t);
        a[t * GTC_GTP + i] = (i < t) ? (bt * expf(gt - gam) * skk[t * GTC_GTP + i]) : ((i == t) ? 1.0f : 0.0f);
    }
    __syncwarp();

    float tv[GTC_GCH];
    tv[0] = (i == 0) ? 1.0f : 0.0f;
    #pragma unroll
    for (int t = 1; t < GTC_GCH; ++t) {
        float v = 0.0f;
        #pragma unroll
        for (int k = 0; k < t; ++k) v = fmaf(-a[t * GTC_GTP + k], tv[k], v);
        tv[t] = (i < t) ? v : ((i == t) ? 1.0f : 0.0f);
    }

    float mt4[4] = {};
    #pragma unroll
    for (int t = 0; t < GTC_GCH; ++t) mt4[t & 3] = fmaxf(mt4[t & 3], fabsf(tv[t]));
    const int et = gtc_scale_exp(gtc_warp_max(fmaxf(fmaxf(mt4[0], mt4[1]), fmaxf(mt4[2], mt4[3]))));
    const float sT = gtc_pow2(et);
    uint16_t* Th = ti + ((int64_t)ch * h_v + vh) * 2 * GTC_GCH * GTC_GCH;
    uint16_t* Tl = Th + GTC_GCH * GTC_GCH;
    #pragma unroll
    for (int t = 0; t < GTC_GCH; ++t) {
        const float x = tv[t] * sT;
        const __half hi = __float2half_rn(x);
        Th[t * GTC_GCH + i] = __half_as_ushort(hi);
        Tl[t * GTC_GCH + i] = __half_as_ushort(__float2half_rn(x - __half2float(hi)));
    }
    if (i == 0) te[(int64_t)ch * h_v + vh] = gtc_pow2(-et);
}

template <int OUT_MASK>
__global__ void __launch_bounds__(256, 1) qwen4_gdn_tc_scan_kernel(
    float* __restrict__ state,
    const float* __restrict__ h,
    const uint16_t* __restrict__ tig_,
    const float* __restrict__ teg,
    const float4* __restrict__ scg,
    const uint16_t* __restrict__ kqg,
    const uint16_t* __restrict__ pig,
    const float* __restrict__ zg,
    const float* __restrict__ gamma,
    float eps,
    float* __restrict__ y32,
    uint16_t* __restrict__ y16,
    int64_t t0,
    int nch,
    int64_t T,
    int h_k,
    int h_v
) {
    constexpr int NT = 256;
    extern __shared__ __align__(16) unsigned char gtc_sm[];
    uint16_t* kh = reinterpret_cast<uint16_t*>(gtc_sm);
    uint16_t* kl = kh + 2 * GTC_GCH * GTC_KP;
    uint16_t* qh = kl + 2 * GTC_GCH * GTC_KP;
    uint16_t* ql = qh + 2 * GTC_GCH * GTC_KP;
    float* sv = reinterpret_cast<float*>(ql + 2 * GTC_GCH * GTC_KP);
    uint16_t* th = reinterpret_cast<uint16_t*>(sv + GTC_GCH * GTC_VP);
    uint16_t* tl = th + GTC_GCH * GTC_PP;
    uint16_t* ph = tl + GTC_GCH * GTC_PP;
    uint16_t* pl = ph + GTC_GCH * GTC_PP;
    float4* sc = reinterpret_cast<float4*>(pl + GTC_GCH * GTC_PP);
    float* ste = reinterpret_cast<float*>(sc + 2 * GTC_GCH);
    float* rss = ste + 2;

    const int vh = (int)blockIdx.x;
    const int qhd = vh % h_k;
    const int C = (2 * h_k + h_v) * GTC_S;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const size_t rs = (size_t)h_v * GTC_S;

    auto stage = [&](int ch, int b, int parts) {
        const int64_t c_t0 = t0 + (int64_t)ch * GTC_GCH;
        const int n = (int)min((int64_t)GTC_GCH, T - c_t0);
        if (parts & 1) {
            const uint16_t* img = kqg + ((size_t)ch * (size_t)h_k + (size_t)qhd) * 4 * GTC_GCH * GTC_S;
            #pragma unroll
            for (int m = 0; m < 4 * GTC_GCH * GTC_S / 8 / NT; ++m) {
                const int pc = tid + NT * m;
                const int im = pc / (GTC_GCH * GTC_S / 8);
                const int t = (pc % (GTC_GCH * GTC_S / 8)) / (GTC_S / 8);
                const int r = (pc % (GTC_S / 8)) * 8;
                uint16_t* dst = ((im == 0) ? kh : ((im == 1) ? kl : ((im == 2) ? qh : ql))) + (b * GTC_GCH + t) * GTC_KP + r;
                gtc_cp16(dst, img + (im * GTC_GCH + t) * GTC_S + r);
            }
            if (tid < GTC_GCH) {
                gtc_cp16(sc + b * GTC_GCH + tid, scg + ((int64_t)ch * h_v + vh) * GTC_GCH + tid);
            }
            if (tid == 0) {
                gtc_cp4(ste + b, teg + (int64_t)ch * h_v + vh);
            }
        }
        if (parts & 2) {
            #pragma unroll
            for (int m = 0; m < GTC_GCH * GTC_S / 4 / NT; ++m) {
                const int p = tid + NT * m;
                const int t = p / (GTC_S / 4);
                const int r = (p % (GTC_S / 4)) * 4;
                float* dst = sv + t * GTC_VP + r;
                if (t < n) {
                    gtc_cp16(dst, h + (c_t0 + t) * C + (size_t)(2 * h_k + vh) * GTC_S + r);
                } else {
                    *reinterpret_cast<float4*>(dst) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            }
            #pragma unroll
            for (int m = 0; m < 2 * GTC_GCH * GTC_GCH / 8 / NT; ++m) {
                const int pc = tid + NT * m;
                const int im = pc / (GTC_GCH * GTC_GCH / 8);
                const int t = (pc % (GTC_GCH * GTC_GCH / 8)) / (GTC_GCH / 8);
                const int i = (pc % (GTC_GCH / 8)) * 8;
                const int64_t src_idx = ((((int64_t)ch * h_v + vh) * 2 + im) * GTC_GCH + t) * GTC_GCH + i;
                const int dst_idx = t * GTC_PP + i;
                gtc_cp16((im ? tl : th) + dst_idx, tig_ + src_idx);
                gtc_cp16((im ? pl : ph) + dst_idx, pig + src_idx);
            }
        }
        gtc_cp_commit();
    };

    const int ca = 16 * warp;
    float s[16][4];
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float* p = state + ((size_t)vh * GTC_S + (size_t)(8 * j + 2 * tig)) * GTC_S + ca + gid;
        s[j][0] = p[0];
        s[j][1] = p[GTC_S];
        s[j][2] = p[8];
        s[j][3] = p[GTC_S + 8];
    }

    stage(0, 0, 3);
    const float out_scale = rsqrtf((float)GTC_S);

    for (int ch = 0; ch < nch; ++ch) {
        const int b = ch & 1;
        const int64_t c_t0 = t0 + (int64_t)ch * GTC_GCH;
        const int n = (int)min((int64_t)GTC_GCH, T - c_t0);
        gtc_cp_wait_all();
        __syncthreads();
        if (ch + 1 < nch) stage(ch + 1, b ^ 1, 1);

        const uint16_t* bkh = kh + b * GTC_GCH * GTC_KP;
        const uint16_t* bkl = kl + b * GTC_GCH * GTC_KP;
        const uint16_t* bqh = qh + b * GTC_GCH * GTC_KP;
        const uint16_t* bql = ql + b * GTC_GCH * GTC_KP;
        const float4* bsc = sc + b * GTC_GCH;

        float m8[8] = {};
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                m8[(2 * j + (q >> 1)) & 7] = fmaxf(m8[(2 * j + (q >> 1)) & 7], fabsf(s[j][q]));
            }
        }
        const float m = fmaxf(fmaxf(fmaxf(m8[0], m8[1]), fmaxf(m8[2], m8[3])),
                              fmaxf(fmaxf(m8[4], m8[5]), fmaxf(m8[6], m8[7])));
        const int es = gtc_scale_exp(gtc_warp_max(m));
        float xt[4][4] = {}, qs[4][4] = {};
        gtc_prod_state(s, gtc_pow2(es), bkh, bkl, bqh, bql, xt, qs);

        const float eS = gtc_pow2(-(8 + es));
        float yt[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float4 s0 = gtc_lds128(bsc + 8 * nt + 2 * tig);
            const float4 s1 = gtc_lds128(bsc + 8 * nt + 2 * tig + 1);
            const float* v = sv + (8 * nt + 2 * tig) * GTC_VP + ca + gid;
            const float e0 = s0.y * eS;
            const float e1 = s1.y * eS;
            yt[nt][0] = s0.x * fmaf(-e0, xt[nt][0], v[0]);
            yt[nt][1] = s1.x * fmaf(-e1, xt[nt][1], v[GTC_VP]);
            yt[nt][2] = s0.x * fmaf(-e0, xt[nt][2], v[8]);
            yt[nt][3] = s1.x * fmaf(-e1, xt[nt][3], v[GTC_VP + 8]);
        }

        float my4[4] = {};
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) my4[q] = fmaxf(my4[q], fabsf(yt[nt][q]));
        }
        const int ey = gtc_scale_exp(gtc_warp_max(fmaxf(fmaxf(my4[0], my4[1]), fmaxf(my4[2], my4[3]))));
        float dt[4][4] = {};
        gtc_prod_dt(yt, gtc_pow2(ey), th, tl, dt);
        const float sYT = gtc_pow2(-ey) * ste[b];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) dt[nt][q] *= sYT;
        }

        float md4[4] = {};
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) md4[q] = fmaxf(md4[q], fabsf(dt[nt][q]));
        }
        const int ed = gtc_scale_exp(gtc_warp_max(fmaxf(fmaxf(md4[0], md4[1]), fmaxf(md4[2], md4[3]))));
        const float sD = gtc_pow2(ed);
        const float sO = gtc_pow2(ed + 8);
        const float sOi = gtc_pow2(-(ed + 8));
        const float eO = gtc_pow2(ed - es);

        float ot[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float e0 = gtc_lds128(bsc + 8 * nt + 2 * tig).y * eO;
            const float e1 = gtc_lds128(bsc + 8 * nt + 2 * tig + 1).y * eO;
            ot[nt][0] = qs[nt][0] * e0;
            ot[nt][1] = qs[nt][1] * e1;
            ot[nt][2] = qs[nt][2] * e0;
            ot[nt][3] = qs[nt][3] * e1;
        }
        gtc_prod_dt(dt, sD, ph, pl, ot);

        float zr[4][4] = {};
        const float gm2[2] = {gamma[ca + gid], gamma[ca + gid + 8]};
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                const int t = 8 * nt + 2 * tig + (q & 1);
                if (t < n) zr[nt][q] = zg[(c_t0 + t) * rs + (size_t)vh * GTC_S + ca + gid + 8 * (q >> 1)];
                ot[nt][q] = (ot[nt][q] * sOi) * out_scale;
            }
        }
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                float ps = ot[nt][e] * ot[nt][e] + ot[nt][e + 2] * ot[nt][e + 2];
                ps += __shfl_xor_sync(0xffffffffu, ps, 4);
                ps += __shfl_xor_sync(0xffffffffu, ps, 8);
                ps += __shfl_xor_sync(0xffffffffu, ps, 16);
                if (gid == 0) rss[warp * GTC_GCH + 8 * nt + 2 * tig + e] = ps;
            }
        }

        __syncthreads();
        if (ch + 1 < nch) stage(ch + 1, 0, 2);

        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int t = 8 * nt + 2 * tig + e;
                if (t >= n) continue;
                float ss = 0.0f;
                #pragma unroll
                for (int w = 0; w < GTC_CW; ++w) ss += rss[w * GTC_GCH + t];
                const float rn = rsqrtf(ss / (float)GTC_S + eps);
                #pragma unroll
                for (int hc = 0; hc < 2; ++hc) {
                    const int col = ca + gid + 8 * hc;
                    const float zz = zr[nt][e + 2 * hc];
                    const float v = ot[nt][e + 2 * hc] * rn * gm2[hc] * (1.0f / (1.0f + __expf(-zz)));
                    const size_t out_off = (c_t0 + t) * rs + (size_t)vh * GTC_S + (size_t)col;
                    if constexpr ((OUT_MASK & 1) != 0) y32[out_off] = v;
                    if constexpr ((OUT_MASK & 2) != 0) y16[out_off] = __half_as_ushort(__float2half_rn(v));
                }
            }
        }

        const float sPre = gtc_lds128(bsc + GTC_GCH - 1).y * sO;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) s[j][q] *= sPre;
        }
        float dd[4][4];
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            const float f0 = gtc_lds128(bsc + 8 * nt + 2 * tig).z * sD;
            const float f1 = gtc_lds128(bsc + 8 * nt + 2 * tig + 1).z * sD;
            dd[nt][0] = dt[nt][0] * f0;
            dd[nt][1] = dt[nt][1] * f1;
            dd[nt][2] = dt[nt][2] * f0;
            dd[nt][3] = dt[nt][3] * f1;
        }
        gtc_update_state(s, dd, bkh, bkl);
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            #pragma unroll
            for (int q = 0; q < 4; ++q) s[j][q] *= sOi;
        }
    }

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        float* p = state + ((size_t)vh * GTC_S + (size_t)(8 * j + 2 * tig)) * GTC_S + ca + gid;
        p[0] = s[j][0];
        p[GTC_S] = s[j][1];
        p[8] = s[j][2];
        p[GTC_S + 8] = s[j][3];
    }
}

template <int OUT_MASK>
static inline void launch_qwen4_gdn_tc_scan_configured(
    float* state, const float* h_qkv,
    const Qwen4GtcScratch& sc,
    const float* z_gate, const float* ssm_norm_w, float eps,
    float* y32, uint16_t* y16,
    int64_t t0, int nch, int64_t T, int h_k, int h_v, cudaStream_t stream
) {
    static int s_configured[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64 && !s_configured[dev]) {
        cudaFuncSetAttribute(
            qwen4_gdn_tc_scan_kernel<OUT_MASK>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)kGtcScanSmemSplitNorm
        );
        s_configured[dev] = 1;
    }
    qwen4_gdn_tc_scan_kernel<OUT_MASK><<<h_v, 256, kGtcScanSmemSplitNorm, stream>>>(
        state, h_qkv, sc.ti, sc.te, sc.sc, sc.kq, sc.pi,
        z_gate, ssm_norm_w, eps, y32, y16, t0, nch, T, h_k, h_v
    );
}

// ============================================================================
// 2d. QSA Sparse/Dense Attention, Indexer, IMRoPE & INT8 KV Cache
// ============================================================================

__device__ __forceinline__ int qwen4_mrope_pos(const int32_t* tab, int pos, int pair) {
    return (tab != nullptr) ? __ldg(tab + (size_t)pos * 3 + (pair % 3)) : pos;
}

__global__ void __launch_bounds__(256) qwen4_qsa_norm_imrope_kernel(
    const float* __restrict__ src,
    int in_stride,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    int n_heads,
    int head_dim,
    int n_rot,
    float eps,
    float freq_base,
    const int32_t* __restrict__ pos_dev,
    const int32_t* __restrict__ mrope_pos3_opt
) {
    const int h = (int)blockIdx.x;
    const int t = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const size_t row = (size_t)t * (size_t)n_heads + (size_t)h;
    const float* x_row = src + row * (size_t)in_stride;
    float* y_row = out + row * (size_t)head_dim;

    __shared__ float s_sums[8];
    __shared__ float s_vals[256];

    float partial = 0.0f;
    for (int col = tid; col < head_dim; col += 256) {
        float v = x_row[col];
        partial += v * v;
    }
    float ss = qwen4_block_reduce_sum_8w(partial, s_sums, tid);
    float scale = rsqrtf(ss / (float)head_dim + eps);

    if (tid < head_dim) {
        s_vals[tid] = x_row[tid] * scale * gamma[tid];
    }
    __syncthreads();

    if (tid < head_dim) {
        const int half_rot = n_rot >> 1;
        if (tid < n_rot) {
            const int pair = (tid < half_rot) ? tid : (tid - half_rot);
            const int raw_pos = __ldg(pos_dev + t);
            const int p = qwen4_mrope_pos(mrope_pos3_opt, raw_pos, pair);
            const float theta = (float)p * powf(freq_base, -2.0f * (float)pair / (float)n_rot);
            float sv, cv;
            sincosf(theta, &sv, &cv);
            const float a = s_vals[pair];
            const float b = s_vals[pair + half_rot];
            y_row[tid] = (tid < half_rot) ? (a * cv - b * sv) : (a * sv + b * cv);
        } else {
            y_row[tid] = s_vals[tid];
        }
    }
}

__global__ void __launch_bounds__(128) qwen4_idx_append_steps_kernel(
    const float* __restrict__ raw_k,
    const int32_t* __restrict__ pos_dev,
    int n_steps,
    const float* __restrict__ gamma,
    float eps,
    float freq_base,
    const int32_t* __restrict__ mrope_pos3_opt,
    float* __restrict__ tail,
    float* __restrict__ dead,
    float* __restrict__ pooled,
    int max_cells
) {
    constexpr int D = 128;
    constexpr int R = 4;
    constexpr int ROT = 64;
    const int d = (int)threadIdx.x;
    __shared__ float s_vals[D];
    __shared__ float s_part[4];

    for (int step_idx = 0; step_idx < n_steps; ++step_idx) {
        const int pos = pos_dev[step_idx];
        if (pos >= 0 && pos < max_cells) {
            const float incoming = __half2float(__float2half_rn(raw_k[(size_t)step_idx * D + d]));
            const int slot = pos % R;
            if (slot < R - 1) {
                tail[slot * D + d] = incoming;
            }
            if (pos == 0 || slot == R - 1) {
                float sum = (pos == 0) ? incoming : tail[d];
                #pragma unroll
                for (int j = 1; j < R; ++j) {
                    sum = __fadd_rn(sum, (pos == 0 || j == R - 1) ? incoming : tail[j * D + d]);
                }
                const float mean = __fmaf_rn(0.25f, sum, 0.0f);
                const float ss = qwen4_block_reduce_sum_4w(mean * mean, s_part, d);
                const float scale = rsqrtf(ss / (float)D + eps);
                s_vals[d] = scale * mean * gamma[d];
                __syncthreads();

                const int b = pos / R;
                const int rope_pos = (pos == 0) ? 0 : (R * b);
                float y = s_vals[d];
                if (d < ROT) {
                    const int pair = d % (ROT / 2);
                    const int p = (pos == 0) ? 0 : qwen4_mrope_pos(mrope_pos3_opt, rope_pos, pair);
                    const float theta = (float)p * powf(freq_base, -2.0f * (float)pair / (float)ROT);
                    float sv, cv;
                    sincosf(theta, &sv, &cv);
                    const float a = s_vals[pair];
                    const float z = s_vals[pair + ROT / 2];
                    y = (d < ROT / 2) ? (a * cv - z * sv) : (a * sv + z * cv);
                }
                pooled[(size_t)b * D + d] = y;
                if (pos == 0) dead[d] = y;
                else pooled[(size_t)(b + 1) * D + d] = dead[d];
            }
        }
        if (n_steps > 1) __syncthreads();
    }
}

__global__ void __launch_bounds__(128) qwen4_idx_append_first_kernel(
    const float* __restrict__ raw_k,
    const float* __restrict__ gamma,
    float eps,
    float freq_base,
    float* __restrict__ dead,
    float* __restrict__ pooled
) {
    constexpr int D = 128;
    constexpr int ROT = 64;
    const int d = (int)threadIdx.x;
    __shared__ float s_vals[D];
    __shared__ float s_part[4];

    const float incoming = __half2float(__float2half_rn(raw_k[d]));
    float sum = incoming;
    #pragma unroll
    for (int j = 1; j < 4; ++j) sum = __fadd_rn(sum, incoming);
    const float mean = __fmaf_rn(0.25f, sum, 0.0f);
    const float ss = qwen4_block_reduce_sum_4w(mean * mean, s_part, d);
    const float scale = rsqrtf(ss / (float)D + eps);
    s_vals[d] = scale * mean * gamma[d];
    __syncthreads();

    float y = s_vals[d];
    if (d < ROT) {
        const float a = s_vals[d % (ROT / 2)];
        const float z = s_vals[(d % (ROT / 2)) + ROT / 2];
        (void)freq_base;
        // At pos == 0: theta = 0 -> cos=1, sin=0 -> identity
        y = (d < ROT / 2) ? a : z;
    }
    pooled[d] = y;
    dead[d] = y;
}

__global__ void __launch_bounds__(128) qwen4_idx_append_blocks_kernel(
    const float* __restrict__ raw_k,
    int64_t p0,
    int64_t first_block,
    int64_t last_block,
    const float* __restrict__ gamma,
    float eps,
    float freq_base,
    const int32_t* __restrict__ mrope_pos3_opt,
    const float* __restrict__ tail,
    const float* __restrict__ dead,
    float* __restrict__ pooled
) {
    constexpr int D = 128;
    constexpr int R = 4;
    constexpr int ROT = 64;
    const int64_t b = first_block + (int64_t)blockIdx.x;
    const int d = (int)threadIdx.x;
    __shared__ float s_vals[D];
    __shared__ float s_part[4];

    auto key_at = [&](int j) -> float {
        int64_t cell = b * R + j;
        return (cell >= p0) ? __half2float(__float2half_rn(raw_k[(cell - p0) * D + d])) : tail[j * D + d];
    };
    float sum = key_at(0);
    #pragma unroll
    for (int j = 1; j < R; ++j) sum = __fadd_rn(sum, key_at(j));
    const float mean = __fmaf_rn(0.25f, sum, 0.0f);
    const float ss = qwen4_block_reduce_sum_4w(mean * mean, s_part, d);
    const float scale = rsqrtf(ss / (float)D + eps);
    s_vals[d] = scale * mean * gamma[d];
    __syncthreads();

    const int rope_pos = R * (int)b;
    float y = s_vals[d];
    if (d < ROT) {
        const int pair = d % (ROT / 2);
        const int p = qwen4_mrope_pos(mrope_pos3_opt, rope_pos, pair);
        const float theta = (float)p * powf(freq_base, -2.0f * (float)pair / (float)ROT);
        float sv, cv;
        sincosf(theta, &sv, &cv);
        const float a = s_vals[pair];
        const float z = s_vals[pair + ROT / 2];
        y = (d < ROT / 2) ? (a * cv - z * sv) : (a * sv + z * cv);
    }
    pooled[(size_t)b * D + d] = y;
    if (b == last_block) {
        pooled[(size_t)(b + 1) * D + d] = dead[d];
    }
}

__global__ void __launch_bounds__(128) qwen4_idx_append_tail_kernel(
    const float* __restrict__ raw_k,
    int64_t n_tokens,
    int64_t p0,
    float* __restrict__ tail
) {
    constexpr int D = 128;
    constexpr int R = 4;
    const int s = (int)blockIdx.x;
    const int d = (int)threadIdx.x;
    const int64_t last = p0 + n_tokens - 1;
    int64_t cell = last - (((last % R) - s + R) % R);
    if (cell < p0) return;
    tail[s * D + d] = __half2float(__float2half_rn(raw_k[(cell - p0) * D + d]));
}

__device__ __forceinline__ uint32_t qwen4_tf32_hi(float x) {
#if QWEN4_SM80
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return r;
#else
    return __float_as_uint(x);
#endif
}

__device__ __forceinline__ void qwen4_mma_tf32(float* c, const uint32_t* a, const uint32_t* b) {
#if QWEN4_SM80
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)c; (void)a; (void)b;
#endif
}

__global__ void __launch_bounds__(256) qwen4_idx_score_warp_kernel(
    const float* __restrict__ pooled,
    const float* __restrict__ dead,
    const float* __restrict__ q_idx,
    const int32_t* __restrict__ pos_dev,
    int max_blocks,
    float* __restrict__ scores_ws
) {
    constexpr int D = 128;
    const int qi = (int)blockIdx.y;
    const int n_kv = pos_dev[qi] + 1;
    if (n_kv <= 2051) return;
    const int n_bid = n_kv / 4;
    const int warp = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int b = (int)blockIdx.x * 8 + warp;
    if (b > n_bid || b >= max_blocks) return;

    const float* key = (b == n_bid) ? dead : (pooled + (size_t)b * D);
    const float4 k4 = *reinterpret_cast<const float4*>(key + lane * 4);
    const float* q = q_idx + (size_t)qi * 4 * D + lane * 4;
    float score = 0.0f;
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * D);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
        d = warp_reduce_sum(d);
        score += (d > 0.0f) ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && (n_kv & 3) != 0) score += 1e9f;
        scores_ws[(size_t)qi * (size_t)max_blocks + (size_t)b] = score;
    }
}

__global__ void __launch_bounds__(256) qwen4_idx_score_warp_multi_kernel(
    const float* __restrict__ pooled,
    const float* __restrict__ dead,
    const float* __restrict__ q_idx,
    const int32_t* __restrict__ pos_dev,
    int T,
    int max_blocks,
    float* __restrict__ scores_ws
) {
    constexpr int D = 128;
    const int warp = (int)threadIdx.x >> 5;
    const int lane = (int)threadIdx.x & 31;
    const int b = (int)blockIdx.x * 8 + warp;
    if (b >= max_blocks) return;

    int n_kv_arr[8];
    int n_bid_arr[8];
    int max_nbid = -1;
    bool need_dead = false;
    for (int qi = 0; qi < T; ++qi) {
        const int nkv = __ldg(pos_dev + qi) + 1;
        n_kv_arr[qi] = nkv;
        if (nkv > 2051) {
            const int nb = nkv / 4;
            n_bid_arr[qi] = nb;
            if (nb > max_nbid) max_nbid = nb;
            if (b == nb) need_dead = true;
        } else {
            n_bid_arr[qi] = -1;
        }
    }
    if (b > max_nbid) return;

    const float4 kp = *reinterpret_cast<const float4*>(pooled + (size_t)b * D + lane * 4);
    float4 kd = make_float4(0.f, 0.f, 0.f, 0.f);
    if (need_dead) {
        kd = *reinterpret_cast<const float4*>(dead + lane * 4);
    }

    for (int qi = 0; qi < T; ++qi) {
        const int n_bid = n_bid_arr[qi];
        if (n_bid < 0 || b > n_bid) continue;
        const float4 k4 = (b == n_bid) ? kd : kp;
        const float* q = q_idx + (size_t)qi * 4 * D + lane * 4;
        float score = 0.0f;
        #pragma unroll
        for (int h = 0; h < 4; ++h) {
            const float4 q4 = *reinterpret_cast<const float4*>(q + h * D);
            float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
            d = warp_reduce_sum(d);
            score += (d > 0.0f) ? d : 0.0f;
        }
        if (lane == 0) {
            if (b == n_bid && (n_kv_arr[qi] & 3) != 0) score += 1e9f;
            scores_ws[(size_t)qi * (size_t)max_blocks + (size_t)b] = score;
        }
    }
}

__global__ void __launch_bounds__(128) qwen4_idx_score_tc_kernel(
    const float* __restrict__ pooled,
    const float* __restrict__ q_idx,
    const int32_t* __restrict__ pos_dev,
    int nq,
    int max_blocks,
    float* __restrict__ scores_ws
) {
    constexpr int IDX_HEADS = 4;
    constexpr int IDX_DIM = 128;
    constexpr int TC_QT = 16;
    constexpr int TC_NB = 32;
    constexpr int TC_ITER = 4;
    constexpr int TC_QS = IDX_HEADS * IDX_DIM + 4; // 516
    constexpr int TC_KS = IDX_DIM + 4;             // 132

    extern __shared__ __align__(16) float sm[];
    float* sQ = sm;                  // [16][516]
    float* sK = sm + TC_QT * TC_QS;  // [32][132]
    __shared__ int s_nbid[TC_QT];

    const int t = (int)threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int q0 = (int)blockIdx.y * TC_QT;

    for (int i = t; i < TC_QT * IDX_HEADS * IDX_DIM / 4; i += 128) {
        const int r = i / (IDX_HEADS * IDX_DIM / 4);
        const int c = i % (IDX_HEADS * IDX_DIM / 4);
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (q0 + r < nq) {
            v = reinterpret_cast<const float4*>(q_idx + (size_t)(q0 + r) * IDX_HEADS * IDX_DIM)[c];
        }
        *reinterpret_cast<float4*>(sQ + r * TC_QS + c * 4) = v;
    }
    if (t < TC_QT) {
        int n_kv = (q0 + t < nq) ? (pos_dev[q0 + t] + 1) : 0;
        s_nbid[t] = (n_kv > 2051) ? (n_kv / 4) : 0;
    }
    __syncthreads();

    int hi_nbid = 0;
    for (int i = 0; i < TC_QT; ++i) {
        if (q0 + i >= nq) break;
        hi_nbid = max(hi_nbid, s_nbid[i]);
    }
    if (hi_nbid == 0) return;

    for (int it = 0; it < TC_ITER; ++it) {
        const int b0 = ((int)blockIdx.x * TC_ITER + it) * TC_NB;
        if (b0 >= max_blocks || b0 >= hi_nbid) break;
        __syncthreads();
        for (int i = t; i < TC_NB * IDX_DIM / 4; i += 128) {
            const int r = i / (IDX_DIM / 4);
            const int c = i % (IDX_DIM / 4);
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (b0 + r < hi_nbid && b0 + r < max_blocks) {
                v = reinterpret_cast<const float4*>(pooled + (size_t)(b0 + r) * IDX_DIM)[c];
            }
            *reinterpret_cast<float4*>(sK + r * TC_KS + c * 4) = v;
        }
        __syncthreads();

        float acc[IDX_HEADS][4];
        #pragma unroll
        for (int h = 0; h < IDX_HEADS; ++h) acc[h][0] = acc[h][1] = acc[h][2] = acc[h][3] = 0.0f;

        const float* kr = sK + (warp * 8 + gid) * TC_KS;
        #pragma unroll 4
        for (int k0 = 0; k0 < IDX_DIM; k0 += 8) {
            const float kx0 = kr[k0 + tig];
            const float kx1 = kr[k0 + tig + 4];
            uint32_t bh[2], bl[2];
            bh[0] = qwen4_tf32_hi(kx0);
            bh[1] = qwen4_tf32_hi(kx1);
            bl[0] = qwen4_tf32_hi(kx0 - __uint_as_float(bh[0]));
            bl[1] = qwen4_tf32_hi(kx1 - __uint_as_float(bh[1]));
            #pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float* qa = sQ + h * IDX_DIM + k0 + tig;
                const float x0 = qa[gid * TC_QS];
                const float x1 = qa[(gid + 8) * TC_QS];
                const float x2 = qa[gid * TC_QS + 4];
                const float x3 = qa[(gid + 8) * TC_QS + 4];
                uint32_t ah[4], al[4];
                ah[0] = qwen4_tf32_hi(x0);
                ah[1] = qwen4_tf32_hi(x1);
                ah[2] = qwen4_tf32_hi(x2);
                ah[3] = qwen4_tf32_hi(x3);
                al[0] = qwen4_tf32_hi(x0 - __uint_as_float(ah[0]));
                al[1] = qwen4_tf32_hi(x1 - __uint_as_float(ah[1]));
                al[2] = qwen4_tf32_hi(x2 - __uint_as_float(ah[2]));
                al[3] = qwen4_tf32_hi(x3 - __uint_as_float(ah[3]));
                qwen4_mma_tf32(acc[h], al, bh);
                qwen4_mma_tf32(acc[h], ah, bl);
                qwen4_mma_tf32(acc[h], ah, bh);
            }
        }

        const int bc = b0 + warp * 8 + 2 * tig;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int qr = gid + half * 8;
            const int qi = q0 + qr;
            if (qi >= nq) continue;
            const int nb = s_nbid[qr];
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int b = bc + j;
                if (b >= nb || b >= max_blocks) continue;
                float score = 0.0f;
                #pragma unroll
                for (int h = 0; h < IDX_HEADS; ++h) {
                    const float d = acc[h][half * 2 + j];
                    score += (d > 0.0f) ? d : 0.0f;
                }
                scores_ws[(size_t)qi * (size_t)max_blocks + (size_t)b] = score;
            }
        }
    }
}

__global__ void __launch_bounds__(32) qwen4_idx_score_tail_kernel(
    const float* __restrict__ dead,
    const float* __restrict__ q_idx,
    const int32_t* __restrict__ pos_dev,
    int max_blocks,
    float* __restrict__ scores_ws
) {
    constexpr int D = 128;
    const int qi = (int)blockIdx.x;
    const int n_kv = pos_dev[qi] + 1;
    if (n_kv <= 2051) return;
    const int n_bid = n_kv / 4;
    if (n_bid >= max_blocks) return;

    const int lane = (int)threadIdx.x & 31;
    const float4 k4 = *reinterpret_cast<const float4*>(dead + lane * 4);
    const float* q = q_idx + (size_t)qi * 4 * D + lane * 4;
    float score = 0.0f;
    #pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float4 q4 = *reinterpret_cast<const float4*>(q + h * D);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
        d = warp_reduce_sum(d);
        score += (d > 0.0f) ? d : 0.0f;
    }
    if (lane == 0) {
        if ((n_kv & 3) != 0) score += 1e9f;
        scores_ws[(size_t)qi * (size_t)max_blocks + (size_t)n_bid] = score;
    }
}

__device__ __forceinline__ uint32_t qwen4_order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v)) return 0u;
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__global__ void __launch_bounds__(256) qwen4_idx_radix_topk_kernel(
    const float* __restrict__ scores_ws,
    const int32_t* __restrict__ pos_dev,
    int max_blocks,
    int top_k_cap,
    int32_t* __restrict__ ids_out,
    int32_t* __restrict__ width_out
) {
    constexpr int R = 4;
    constexpr int TOPK_T = 256;
    __shared__ int hist[256];
    __shared__ int s_a[TOPK_T], s_b[TOPK_T];
    __shared__ int s_digit, s_above;

    const int qi = (int)blockIdx.x;
    const int n_kv = pos_dev[qi] + 1;
    const int n_bid = n_kv / R;
    const int width = min(min(n_kv, 2051), top_k_cap);
    const int t = (int)threadIdx.x;

    if (t == 0 && width_out != nullptr) {
        width_out[qi] = width;
    }

    int32_t* out = ids_out + (size_t)qi * (size_t)top_k_cap;
    if (n_kv <= width) {
        for (int j = t; j < width; j += TOPK_T) out[j] = j;
        return;
    }

    const float* sc = scores_ws + (size_t)qi * (size_t)max_blocks;
    const int nb = min(n_bid + 1, max_blocks);
    const int per = (nb + TOPK_T - 1) / TOPK_T;
    const int b0 = t * per;
    const int b1 = min(b0 + per, nb);
    auto weight = [&](int b) -> int {
        return (b < n_bid) ? R : (n_kv - n_bid * R);
    };

    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[t] = 0;
        __syncthreads();
        const uint32_t hi_mask = (shift == 24) ? 0u : (0xffffffffu << (shift + 8));
        for (int b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0) continue;
            const uint32_t k = qwen4_order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask)) atomicAdd(&hist[(k >> shift) & 255], w);
        }
        __syncthreads();
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width) break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        __syncthreads();
        prefix |= (uint32_t)s_digit << shift;
        above = s_above;
        __syncthreads();
    }

    const uint32_t thr = prefix;
    const int eq_budget = width - above;
    int gt = 0, eq = 0;
    for (int b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = qwen4_order_key(sc[b]);
        if (k > thr) gt += w;
        else if (k == thr) eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    __syncthreads();

    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            int g = s_a[i], e = s_b[i];
            s_a[i] = ag;
            s_b[i] = ae;
            ag += g;
            ae += e;
        }
    }
    __syncthreads();

    const int eq_before = s_b[t];
    int my_eq = eq_budget - eq_before;
    if (my_eq < 0) my_eq = 0;
    if (my_eq > eq) my_eq = eq;
    const int sel = gt + my_eq;
    __syncthreads();

    s_a[t] = sel;
    __syncthreads();
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            int c = s_a[i];
            s_a[i] = a;
            a += c;
        }
    }
    __syncthreads();

    int wpos = s_a[t];
    int eq_left = my_eq;
    for (int b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0) continue;
        const uint32_t k = qwen4_order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c) out[wpos++] = b * R + c;
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left) out[wpos++] = b * R + c;
        }
    }
}

__global__ void __launch_bounds__(64) qwen4_kv_append_q8_kernel(
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int32_t* __restrict__ pos_dev,
    int n_kv_heads,
    int head_dim,
    int8_t* __restrict__ k_q,
    int8_t* __restrict__ v_q,
    uint16_t* __restrict__ k_sc,
    uint16_t* __restrict__ v_sc
) {
    const int h = (int)blockIdx.x;
    const int g = (int)blockIdx.y;
    const int t = (int)(blockIdx.z >> 1);
    const int is_v = (int)(blockIdx.z & 1);
    const int d = (int)threadIdx.x;
    const int pos = pos_dev[t];

    const size_t src_off = ((size_t)t * (size_t)n_kv_heads + (size_t)h) * (size_t)head_dim + (size_t)g * 64 + (size_t)d;
    const float val = (is_v ? v : k)[src_off];

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    }
    __shared__ float s_amax[2];
    if ((d & 31) == 0) s_amax[d >> 5] = amax;
    __syncthreads();
    amax = fmaxf(s_amax[0], s_amax[1]);

    const uint16_t sbits = f32_to_fp16_bits(amax / 127.0f);
    const float sf = fp16_bits_to_f32(sbits);
    const float inv_sf = (sf > 0.0f) ? (1.0f / sf) : 0.0f;
    int q = __float2int_rn(val * inv_sf);
    if (q > 127) q = 127;
    if (q < -127) q = -127;

    const size_t row = (size_t)pos * (size_t)n_kv_heads + (size_t)h;
    (is_v ? v_q : k_q)[row * (size_t)head_dim + (size_t)g * 64 + (size_t)d] = (int8_t)q;
    if (d == 0) {
        (is_v ? v_sc : k_sc)[row * (size_t)(head_dim >> 6) + (size_t)g] = sbits;
    }
}

__device__ __forceinline__ void qwen4_load8_kv_q8(
    const int8_t* __restrict__ q_base,
    const uint16_t* __restrict__ sc_base,
    size_t row,
    int d0,
    float* out
) {
    const int8_t* codes = q_base + row * 256 + d0;
    const uint16_t sbits = sc_base[row * 4 + (d0 >> 6)];
    const float sc = fp16_bits_to_f32(sbits);
    const uint2 raw = *reinterpret_cast<const uint2*>(codes);
    const int8_t* c = reinterpret_cast<const int8_t*>(&raw);
    #pragma unroll
    for (int j = 0; j < 8; ++j) out[j] = (float)c[j] * sc;
}

constexpr int QWEN4_QSA_MAX_CHUNKS = 48;

__global__ void __launch_bounds__(256) qwen4_qsa_decode_chunk_kernel(
    const float* __restrict__ q,
    const int8_t* __restrict__ k_q,
    const int8_t* __restrict__ v_q,
    const uint16_t* __restrict__ k_sc,
    const uint16_t* __restrict__ v_sc,
    const int32_t* __restrict__ ids_opt,
    const int32_t* __restrict__ n_ids_dev,
    int id_cap,
    int n_chunks,
    int n_kv_heads,
    float scale,
    float* __restrict__ part_acc,
    float* __restrict__ part_m,
    float* __restrict__ part_l,
    size_t stride_acc,
    size_t stride_m
) {
    constexpr int G = 12;
    constexpr int HD = 256;
    constexpr int CHUNK = 64;
    constexpr int WARPS = 8;

    const int chunk = (int)blockIdx.x;
    const int kvh = (int)blockIdx.y;
    const int qi = (int)blockIdx.z;
    const int t = (int)threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    const int n_ids = __ldg(n_ids_dev + qi);
    const int slot = kvh * n_chunks + chunk;

    part_acc += (size_t)qi * stride_acc;
    part_m   += (size_t)qi * stride_m;
    part_l   += (size_t)qi * stride_m;

    if (chunk * CHUNK >= n_ids || n_ids <= 0) {
        if (t < G) {
            part_m[slot * G + t] = -FLT_MAX;
            part_l[slot * G + t] = 0.0f;
        }
        return;
    }

    __shared__ __align__(16) float sq[G][HD];
    __shared__ float sp[G][CHUNK];
    __shared__ int64_t srow[CHUNK];
    __shared__ float s_m[G];
    __shared__ float s_l[G];
    __shared__ float s_alpha[G];

    const float* q_kvh = q + ((size_t)qi * (size_t)(n_kv_heads * G) + (size_t)(kvh * G)) * HD;
    for (int i = t; i < G * HD; i += 256) {
        sq[i / HD][i % HD] = q_kvh[i];
    }
    if (t < G) {
        s_m[t] = -FLT_MAX;
        s_l[t] = 0.0f;
        s_alpha[t] = 0.0f;
    }

    float acc[G];
    #pragma unroll
    for (int h = 0; h < G; ++h) acc[h] = 0.0f;

    const int n_total_chunks = (n_ids + CHUNK - 1) / CHUNK;
    for (int c_idx = chunk; c_idx < n_total_chunks; c_idx += n_chunks) {
        const int c0 = c_idx * CHUNK;
        const int n_here = min(CHUNK, n_ids - c0);

        if (t < CHUNK) {
            int64_t r = -1;
            if (t < n_here) {
                int cell = (ids_opt != nullptr) ? ids_opt[(size_t)qi * (size_t)id_cap + c0 + t] : (c0 + t);
                if (cell >= 0) r = (int64_t)cell * (int64_t)n_kv_heads + (int64_t)kvh;
            }
            srow[t] = r;
        }
        __syncthreads();

        for (int c = warp; c < CHUNK; c += WARPS) {
            if (c >= n_here || srow[c] < 0) {
                if (lane < G) sp[lane][c] = -FLT_MAX;
                continue;
            }
            float k8[8];
            qwen4_load8_kv_q8(k_q, k_sc, (size_t)srow[c], lane * 8, k8);
            #pragma unroll
            for (int h = 0; h < G; ++h) {
                const float4 qa = *reinterpret_cast<const float4*>(&sq[h][lane * 8]);
                const float4 qb = *reinterpret_cast<const float4*>(&sq[h][lane * 8 + 4]);
                float s = k8[0] * qa.x + k8[1] * qa.y + k8[2] * qa.z + k8[3] * qa.w
                        + k8[4] * qb.x + k8[5] * qb.y + k8[6] * qb.z + k8[7] * qb.w;
                s = warp_reduce_sum(s);
                if (lane == 0) sp[h][c] = s * scale;
            }
        }
        __syncthreads();

        for (int h = warp; h < G; h += WARPS) {
            const float a = (lane < n_here && srow[lane] >= 0) ? sp[h][lane] : -FLT_MAX;
            const float b = (lane + 32 < n_here && srow[lane + 32] >= 0) ? sp[h][lane + 32] : -FLT_MAX;
            const float m_loc = warp_reduce_max(fmaxf(a, b));
            const float ea = (lane < n_here && srow[lane] >= 0 && m_loc > -1e38f) ? __expf(a - m_loc) : 0.0f;
            const float eb = (lane + 32 < n_here && srow[lane + 32] >= 0 && m_loc > -1e38f) ? __expf(b - m_loc) : 0.0f;
            const float l_loc = warp_reduce_sum(ea + eb);
            float beta = 0.0f;
            if (lane == 0) {
                const float m_old = s_m[h];
                const float m_new = fmaxf(m_old, m_loc);
                const float alpha = (m_old > -1e38f) ? __expf(m_old - m_new) : 0.0f;
                beta = (m_loc > -1e38f) ? __expf(m_loc - m_new) : 0.0f;
                s_m[h] = m_new;
                s_l[h] = fmaf(s_l[h], alpha, l_loc * beta);
                s_alpha[h] = alpha;
            }
            beta = __shfl_sync(0xffffffffu, beta, 0);
            sp[h][lane] = ea * beta;
            sp[h][lane + 32] = eb * beta;
        }
        __syncthreads();

        #pragma unroll
        for (int h = 0; h < G; ++h) acc[h] *= s_alpha[h];

        for (int c = 0; c < n_here; ++c) {
            if (srow[c] < 0) continue;
            const float sc = fp16_bits_to_f32(v_sc[(size_t)srow[c] * 4 + (t >> 6)]);
            const float vv = (float)v_q[(size_t)srow[c] * HD + (size_t)t] * sc;
            #pragma unroll
            for (int h = 0; h < G; ++h) acc[h] = fmaf(sp[h][c], vv, acc[h]);
        }
        if (c_idx + n_chunks < n_total_chunks) {
            __syncthreads();
        }
    }

    if (t < G) {
        part_m[slot * G + t] = s_m[t];
        part_l[slot * G + t] = s_l[t];
    }
    #pragma unroll
    for (int h = 0; h < G; ++h) {
        part_acc[((size_t)slot * G + (size_t)h) * HD + (size_t)t] = acc[h];
    }
}

__global__ void __launch_bounds__(256) qwen4_qsa_decode_merge_kernel(
    const float* __restrict__ part_acc,
    const float* __restrict__ part_m,
    const float* __restrict__ part_l,
    int n_chunks,
    const int32_t* __restrict__ n_ids_dev,
    const float* __restrict__ q_full_gate_opt,
    float* __restrict__ out_f32_opt,
    block_q8_1* __restrict__ out_q8_opt,
    size_t stride_acc,
    size_t stride_m
) {
    constexpr int G = 12;
    constexpr int HD = 256;
    const int h = (int)blockIdx.x;
    const int n_heads = (int)gridDim.x;
    const int qi = (int)blockIdx.y;
    const int d = (int)threadIdx.x;
    const int kvh = h / G;
    const int hl = h % G;

    part_acc += (size_t)qi * stride_acc;
    part_m   += (size_t)qi * stride_m;
    part_l   += (size_t)qi * stride_m;

    const int n_ids = __ldg(n_ids_dev + qi);
    const int n_act = (n_ids > 0) ? min(n_chunks, (n_ids + 63) / 64) : 0;

    const size_t base = (size_t)kvh * (size_t)n_chunks * G + (size_t)hl;
    const float* pm = part_m + base;
    const float* pl = part_l + base;
    const float* pa = part_acc + base * HD + (size_t)d;

    float M = -FLT_MAX;
    for (int c = 0; c < n_act; ++c) {
        M = fmaxf(M, pm[(size_t)c * G]);
    }
    float L = 0.0f, acc = 0.0f;
    for (int c = 0; c < n_act; ++c) {
        const float mc = pm[(size_t)c * G];
        if (mc == -FLT_MAX) continue;
        const float w = __expf(mc - M);
        L = fmaf(pl[(size_t)c * G], w, L);
        acc = fmaf(pa[(size_t)c * G * HD], w, acc);
    }

    float val = (L > 0.0f) ? (acc / L) : 0.0f;
    const size_t out_head = (size_t)qi * (size_t)n_heads + (size_t)h;
    if (q_full_gate_opt != nullptr) {
        float g = q_full_gate_opt[out_head * 512 + 256 + (size_t)d];
        val *= qwen4_sigmoid(g);
    }
    if (out_f32_opt != nullptr) {
        out_f32_opt[out_head * HD + (size_t)d] = val;
    }
    if (out_q8_opt != nullptr) {
        warp_quantize_q8_1_store(val, out_q8_opt + out_head * 8 + (d >> 5), d & 31);
    }
}

struct alignas(16) Qwen4QsaPrefillSmem {
    static constexpr int G = 12;
    static constexpr int HD = 256;
    static constexpr int CH = 32;
    static constexpr int QS = HD + 8;     // 264 halves
    static constexpr int KROW = HD + 16;  // 272 int8
    static constexpr int VROW = HD + 16;  // 272 int8
    __half qh[16][QS];
    __half ql[16][QS];
    union {
        int8_t k[CH][KROW];
        int8_t v[CH][VROW];
    } kv;
    float ks[CH][4];
    float vs[CH][4];
    float s[16][CH + 1];
    float qmax[4];
    float alpha[16];
    float lsum[16];
    float mrow[16];
};

__device__ __forceinline__ void qsa_mma16816(float* c, const uint32_t* a, const uint32_t* b) {
#if QWEN4_SM80
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    (void)c; (void)a; (void)b;
#endif
}

__device__ __forceinline__ uint32_t qsa_i8x2_to_h2(uint32_t x) {
    uint32_t y = ((x & 0xffu) | ((x & 0xff00u) << 8)) ^ 0x00800080u;
    y |= 0x64006400u;
    __half2 h = *reinterpret_cast<__half2*>(&y);
    h = __hsub2(h, __halves2half2(__float2half(1152.f), __float2half(1152.f)));
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t qsa_pack_h2(float lo_k, float hi_k) {
    __half2 h = __floats2half2_rn(lo_k, hi_k);
    return *reinterpret_cast<uint32_t*>(&h);
}

__global__ void __launch_bounds__(128) qwen4_qsa_prefill_attn_q8_kernel(
    const float* __restrict__ q,
    const float* __restrict__ q_full_gate_opt,
    const int8_t* __restrict__ k_q,
    const int8_t* __restrict__ v_q,
    const uint16_t* __restrict__ k_sc,
    const uint16_t* __restrict__ v_sc,
    const int32_t* __restrict__ ids_opt,
    const int32_t* __restrict__ n_ids_dev,
    int id_cap,
    int n_kv_heads,
    float scale_log2,
    float* __restrict__ out_f32
) {
    constexpr int G = Qwen4QsaPrefillSmem::G;
    constexpr int HD = Qwen4QsaPrefillSmem::HD;
    constexpr int CH = Qwen4QsaPrefillSmem::CH;
    constexpr int THREADS = 128;
    constexpr int KP = HD / 16; // 16
    constexpr int PPT = (CH * KP) / THREADS; // 4

    __shared__ Qwen4QsaPrefillSmem S;

    const int qi = (int)blockIdx.x;
    const int kvh = (int)blockIdx.y;
    const int n_head = n_kv_heads * G;
    const int n = __ldg(n_ids_dev + qi);
    const int t = (int)threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const int gid = lane >> 2;
    const int tig = lane & 3;

    const float* q_ptr = q + ((size_t)qi * (size_t)n_head + (size_t)kvh * G) * HD;
    float* out_ptr = out_f32 + ((size_t)qi * (size_t)n_head + (size_t)kvh * G) * HD;
    const int32_t* ids = (ids_opt != nullptr) ? (ids_opt + (size_t)qi * (size_t)id_cap) : nullptr;

    float qm = 0.0f;
    for (int i = t; i < G * HD; i += THREADS) qm = fmaxf(qm, fabsf(q_ptr[i]));
    qm = warp_reduce_max(qm);
    if (lane == 0) S.qmax[warp] = qm;
    __syncthreads();
    qm = fmaxf(fmaxf(S.qmax[0], S.qmax[1]), fmaxf(S.qmax[2], S.qmax[3]));

    int qe = 0;
    if (qm > 0.0f) frexpf(qm, &qe);
    const float qup = ldexpf(1.0f, 14 - qe);
    const float qdown = ldexpf(scale_log2, qe - 14);

    for (int i = t; i < 16 * HD; i += THREADS) {
        const int h = i / HD;
        const int d = i % HD;
        const float x = (h < G) ? (q_ptr[(size_t)h * HD + d] * qup) : 0.0f;
        const __half hi = __float2half_rn(x);
        S.qh[h][d] = hi;
        S.ql[h][d] = __float2half_rn(x - __half2float(hi));
    }
    if (t < 16) {
        S.mrow[t] = -INFINITY;
        S.lsum[t] = 0.0f;
    }

    float acc[8][4];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.0f;

    auto row_of = [&](int c0, int c) -> int64_t {
        if (c >= min(CH, n - c0)) return -1;
        const int cell = (ids != nullptr) ? ids[c0 + c] : (c0 + c);
        return (cell >= 0) ? ((int64_t)cell * (int64_t)n_kv_heads + (int64_t)kvh) : -1;
    };

    uint4 pk[PPT], pv[PPT];
    float pa = 0.0f, pb = 0.0f;
    auto fetch = [&](int c0) {
        #pragma unroll
        for (int j = 0; j < PPT; ++j) {
            const int i = t + j * THREADS;
            const int c = i / KP;
            const int pc = i % KP;
            const int64_t r = row_of(c0, c);
            pk[j] = (r >= 0) ? __ldg(reinterpret_cast<const uint4*>(k_q + r * HD) + pc) : make_uint4(0, 0, 0, 0);
            pv[j] = (r >= 0) ? __ldg(reinterpret_cast<const uint4*>(v_q + r * HD) + pc) : make_uint4(0, 0, 0, 0);
        }
        const int c = t >> 2;
        const int g = t & 3;
        const int64_t r = row_of(c0, c);
        pa = (r >= 0) ? fp16_bits_to_f32(k_sc[r * 4 + g]) : 0.0f;
        pb = (r >= 0) ? fp16_bits_to_f32(v_sc[r * 4 + g]) : 0.0f;
    };
    fetch(0);

    for (int c0 = 0; c0 < n; c0 += CH) {
        const int nh = min(CH, n - c0);
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < PPT; ++j) {
            const int i = t + j * THREADS;
            const int c = i / KP;
            const int pc = i % KP;
            *reinterpret_cast<uint4*>(&S.kv.k[c][0] + pc * 16) = pk[j];
        }
        S.ks[t >> 2][t & 3] = pa;
        S.vs[t >> 2][t & 3] = pb;
        __syncthreads();

        const int cb = warp * 8;
        float sc[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            float tg[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                const int k0 = (g * 4 + kk) * 16;
                uint32_t ah[4], al[4], b[2];
                ah[0] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig]);
                ah[1] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig]);
                ah[2] = *reinterpret_cast<const uint32_t*>(&S.qh[gid][k0 + 2 * tig + 8]);
                ah[3] = *reinterpret_cast<const uint32_t*>(&S.qh[gid + 8][k0 + 2 * tig + 8]);
                al[0] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig]);
                al[1] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig]);
                al[2] = *reinterpret_cast<const uint32_t*>(&S.ql[gid][k0 + 2 * tig + 8]);
                al[3] = *reinterpret_cast<const uint32_t*>(&S.ql[gid + 8][k0 + 2 * tig + 8]);
                b[0] = qsa_i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.kv.k[cb + gid][k0 + 2 * tig]));
                b[1] = qsa_i8x2_to_h2(*reinterpret_cast<const uint16_t*>(&S.kv.k[cb + gid][k0 + 2 * tig + 8]));
                qsa_mma16816(tg, ah, b);
                qsa_mma16816(tg, al, b);
            }
            const float s0 = S.ks[cb + 2 * tig][g];
            const float s1 = S.ks[cb + 2 * tig + 1][g];
            sc[0] = fmaf(tg[0], s0, sc[0]);
            sc[1] = fmaf(tg[1], s1, sc[1]);
            sc[2] = fmaf(tg[2], s0, sc[2]);
            sc[3] = fmaf(tg[3], s1, sc[3]);
        }
        const int c = cb + 2 * tig;
        S.s[gid][c]         = (c < nh) ? (sc[0] * qdown) : -INFINITY;
        S.s[gid][c + 1]     = (c + 1 < nh) ? (sc[1] * qdown) : -INFINITY;
        S.s[gid + 8][c]     = (c < nh) ? (sc[2] * qdown) : -INFINITY;
        S.s[gid + 8][c + 1] = (c + 1 < nh) ? (sc[3] * qdown) : -INFINITY;
        __syncthreads();

        #pragma unroll
        for (int j = 0; j < PPT; ++j) {
            const int i = t + j * THREADS;
            const int cc = i / KP;
            const int pc = i % KP;
            *reinterpret_cast<uint4*>(&S.kv.v[cc][0] + pc * 16) = pv[j];
        }
        if (c0 + CH < n) fetch(c0 + CH);

        {
            constexpr int PER = CH / 8; // 4
            const int r = t >> 3;
            const int sub = t & 7;
            float x[PER], mx = -INFINITY;
            #pragma unroll
            for (int j = 0; j < PER; ++j) {
                x[j] = S.s[r][sub * PER + j];
                mx = fmaxf(mx, x[j]);
            }
            #pragma unroll
            for (int o = 1; o < 8; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
            const float m_old = S.mrow[r];
            const float m_new = fmaxf(m_old, mx);
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < PER; ++j) {
                const float e = (x[j] == -INFINITY) ? 0.0f : exp2f(x[j] - m_new);
                S.s[r][sub * PER + j] = e;
                sum += e;
            }
            #pragma unroll
            for (int o = 1; o < 8; o <<= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
            __syncwarp();
            if (sub == 0) {
                const float a = (m_old == -INFINITY) ? 0.0f : exp2f(m_old - m_new);
                S.alpha[r] = a;
                S.lsum[r] = fmaf(S.lsum[r], a, sum);
                S.mrow[r] = m_new;
            }
        }
        __syncthreads();

        {
            float vmax = 0.0f;
            for (int cc = lane; cc < CH; cc += 32) vmax = fmaxf(vmax, fabsf(S.vs[cc][warp]));
            vmax = warp_reduce_max(vmax);
            const float vup = (vmax > 0.0f) ? (16384.0f / vmax) : 0.0f;
            const float vdown = vmax * (1.0f / 16384.0f);

            float tmp[8][4];
            #pragma unroll
            for (int j = 0; j < 8; ++j) tmp[j][0] = tmp[j][1] = tmp[j][2] = tmp[j][3] = 0.0f;

            #pragma unroll
            for (int ks = 0; ks < CH / 16; ++ks) {
                const int cA = ks * 16 + 2 * tig;
                const int cB = cA + 8;
                const float w0 = S.vs[cA][warp] * vup;
                const float w1 = S.vs[cA + 1][warp] * vup;
                const float w2 = S.vs[cB][warp] * vup;
                const float w3 = S.vs[cB + 1][warp] * vup;
                const float p00 = S.s[gid][cA] * w0,     p01 = S.s[gid][cA + 1] * w1;
                const float p10 = S.s[gid + 8][cA] * w0, p11 = S.s[gid + 8][cA + 1] * w1;
                const float p02 = S.s[gid][cB] * w2,     p03 = S.s[gid][cB + 1] * w3;
                const float p12 = S.s[gid + 8][cB] * w2, p13 = S.s[gid + 8][cB + 1] * w3;
                uint32_t ah[4], al[4];
                ah[0] = qsa_pack_h2(p00, p01);
                ah[1] = qsa_pack_h2(p10, p11);
                ah[2] = qsa_pack_h2(p02, p03);
                ah[3] = qsa_pack_h2(p12, p13);
                {
                    const __half2* h2 = reinterpret_cast<const __half2*>(ah);
                    float2 f;
                    f = __half22float2(h2[0]); al[0] = qsa_pack_h2(p00 - f.x, p01 - f.y);
                    f = __half22float2(h2[1]); al[1] = qsa_pack_h2(p10 - f.x, p11 - f.y);
                    f = __half22float2(h2[2]); al[2] = qsa_pack_h2(p02 - f.x, p03 - f.y);
                    f = __half22float2(h2[3]); al[3] = qsa_pack_h2(p12 - f.x, p13 - f.y);
                }
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int d = warp * 64 + j * 8 + gid;
                    uint32_t b[2];
                    const uint32_t x0 = (uint8_t)S.kv.v[cA][d] | ((uint32_t)(uint8_t)S.kv.v[cA + 1][d] << 8);
                    const uint32_t x1 = (uint8_t)S.kv.v[cB][d] | ((uint32_t)(uint8_t)S.kv.v[cB + 1][d] << 8);
                    b[0] = qsa_i8x2_to_h2(x0);
                    b[1] = qsa_i8x2_to_h2(x1);
                    qsa_mma16816(tmp[j], ah, b);
                    qsa_mma16816(tmp[j], al, b);
                }
            }
            const float a0 = S.alpha[gid], a1 = S.alpha[gid + 8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] = fmaf(acc[j][0], a0, tmp[j][0] * vdown);
                acc[j][1] = fmaf(acc[j][1], a0, tmp[j][1] * vdown);
                acc[j][2] = fmaf(acc[j][2], a1, tmp[j][2] * vdown);
                acc[j][3] = fmaf(acc[j][3], a1, tmp[j][3] * vdown);
            }
        }
    }
    __syncthreads();

    const float l0 = S.lsum[gid], l1 = S.lsum[gid + 8];
    const float i0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
    const float i1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
    const size_t gate_head0 = ((size_t)qi * (size_t)n_head + (size_t)kvh * G + (size_t)gid) * 512 + 256;
    const size_t gate_head1 = ((size_t)qi * (size_t)n_head + (size_t)kvh * G + (size_t)(gid + 8)) * 512 + 256;

    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int d = warp * 64 + j * 8 + 2 * tig;
        float o0 = acc[j][0] * i0;
        float o1 = acc[j][1] * i0;
        if (q_full_gate_opt != nullptr) {
            o0 *= qwen4_sigmoid(q_full_gate_opt[gate_head0 + d + 0]);
            o1 *= qwen4_sigmoid(q_full_gate_opt[gate_head0 + d + 1]);
        }
        *reinterpret_cast<float2*>(out_ptr + (size_t)gid * HD + d) = make_float2(o0, o1);
        if (gid + 8 < G) {
            float o2 = acc[j][2] * i1;
            float o3 = acc[j][3] * i1;
            if (q_full_gate_opt != nullptr) {
                o2 *= qwen4_sigmoid(q_full_gate_opt[gate_head1 + d + 0]);
                o3 *= qwen4_sigmoid(q_full_gate_opt[gate_head1 + d + 1]);
            }
            *reinterpret_cast<float2*>(out_ptr + (size_t)(gid + 8) * HD + d) = make_float2(o2, o3);
        }
    }
}

// ============================================================================
// 2e. MoE Router & Shared Expert Scalar Gate
// ============================================================================

__global__ void __launch_bounds__(256) qwen4_router_top10_kernel(
    const float* __restrict__ logits,
    int T,
    int n_exp,
    int topk,
    int32_t* __restrict__ ids_out,
    float* __restrict__ weights_out
) {
    const int t = (int)blockIdx.x * 8 + (int)threadIdx.y;
    if (t >= T) return;
    const int lane = (int)threadIdx.x;
    const float* x = logits + (size_t)t * (size_t)n_exp;
    int32_t* id_dst = ids_out + (size_t)t * (size_t)topk;
    float* wt_dst = weights_out + (size_t)t * (size_t)topk;

    float values[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int idx = lane + i * 32;
        values[i] = (idx < n_exp) ? x[idx] : -INFINITY;
    }

    float maximum = -INFINITY;
    #pragma unroll
    for (int i = 0; i < 16; ++i) maximum = fmaxf(maximum, values[i]);
    maximum = warp_reduce_max(maximum);

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int idx = lane + i * 32;
        float e = (idx < n_exp) ? expf(values[i] - maximum) : 0.0f;
        values[i] = (idx < n_exp) ? e : -INFINITY;
        sum += e;
    }
    const float reciprocal = 1.0f / warp_reduce_sum(sum);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        int idx = lane + i * 32;
        if (idx < n_exp) {
            values[i] *= reciprocal;
            if (isnan(values[i])) values[i] = -FLT_MAX;
        }
    }

    float selected = 0.0f;
    float selected_sum = 0.0f;
    for (int rank = 0; rank < topk; ++rank) {
        float best = values[0];
        int expert = lane;
        #pragma unroll
        for (int i = 1; i < 16; ++i) {
            if (values[i] > best) {
                best = values[i];
                expert = lane + i * 32;
            }
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask);
            if (other > best || (other == best && other_id < expert)) {
                best = other;
                expert = other_id;
            }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -INFINITY;
            id_dst[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }

    selected_sum = fmaxf(warp_reduce_sum(selected_sum), 6.103515625e-5f);
    const float inv = 1.0f / selected_sum;
    if (lane < topk) {
        wt_dst[lane] = selected * inv;
    }
}

__global__ void __launch_bounds__(256) qwen4_shared_gate_dot_kernel(
    const float* __restrict__ x,
    const void* __restrict__ w_sh_gate,
    int w_type,
    int n_embd,
    float* __restrict__ out_logits
) {
    const int t = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;
    const float* xr = x + (size_t)t * (size_t)n_embd;

    __shared__ float s_sums[8];
    float partial = 0.0f;
    if (w_type == GGML_TYPE_F32) {
        const float* w = reinterpret_cast<const float*>(w_sh_gate);
        for (int d = tid; d < n_embd; d += 256) partial += xr[d] * w[d];
    } else if (w_type == GGML_TYPE_BF16) {
        const uint16_t* w = reinterpret_cast<const uint16_t*>(w_sh_gate);
        for (int d = tid; d < n_embd; d += 256) partial += xr[d] * bf16_bits_to_f32(w[d]);
    }
    float sum = qwen4_block_reduce_sum_8w(partial, s_sums, tid);
    if (tid == 0) out_logits[t] = sum;
}

// ============================================================================
// 2f. Qwen3VL Vision Tower Helpers
// ============================================================================

__global__ void __launch_bounds__(64) qwen4_vis_rope2d_kernel(
    float* __restrict__ qkv,
    const int32_t* __restrict__ pos_yx,
    int heads,
    int hd,
    float base
) {
    const int h = (int)blockIdx.x;
    const int n = (int)blockIdx.y;
    const int half = hd >> 1;
    const int quarter = hd >> 2;
    const int D = heads * hd;

    for (int p = (int)threadIdx.x; p < half; p += blockDim.x) {
        const bool is_x = (p >= quarter);
        const int freq_idx = is_x ? (p - quarter) : p;
        const int pos = __ldg(pos_yx + (size_t)n * 2 + (is_x ? 1 : 0));
        const float theta = (float)pos * powf(base, -2.0f * (float)freq_idx / (float)half);
        float sv, cv;
        sincosf(theta, &sv, &cv);

        float* q_head = qkv + (size_t)n * 3 * (size_t)D + (size_t)h * (size_t)hd;
        float* k_head = q_head + (size_t)D;

        const float qa = q_head[p], qb = q_head[p + half];
        q_head[p]        = qa * cv - qb * sv;
        q_head[p + half] = qa * sv + qb * cv;

        const float ka = k_head[p], kb = k_head[p + half];
        k_head[p]        = ka * cv - kb * sv;
        k_head[p + half] = ka * sv + kb * cv;
    }
}

__global__ void qwen4_vis_merge_2x2_f16_kernel(
    const float* __restrict__ x_ln,
    size_t n_elems,
    uint16_t* __restrict__ out_f16
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_elems) return;
    out_f16[idx] = f32_to_fp16_bits(x_ln[idx]);
}

// ============================================================================
// 2g. Phase-4 Optimized Decode/Verify & Fusion Helpers
// ============================================================================

struct alignas(64) Qwen4Doorbell {
    uint64_t req_seq;
    uint64_t done_seq;
    int32_t  layer_idx;
    int32_t  miss_count;
    int32_t  m_batch;
    int32_t  gu_type;
    int32_t  d_type;
    int32_t  n_embd;
    int32_t  n_ff;
    int32_t  _pad;
    int32_t  miss_tok[80];
    int32_t  miss_ids[80];
    float    miss_weights[80];
    float    x_in[8 * 2560];
    float    out[8 * 2560];
    int32_t  routed_ids[48][80];
    int32_t  routed_n_pairs[48];
};
static_assert(sizeof(Qwen4Doorbell) == 180416, "Qwen4Doorbell size mismatch with Rust");

__global__ void __launch_bounds__(128) qwen4_fill_dense_widths_kernel(
    const int32_t* __restrict__ pos_dev,
    int32_t* __restrict__ width_out,
    int T
) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < T) {
        width_out[i] = pos_dev[i] + 1;
    }
}

template <int TM_CAP, bool EXACT_M>
__global__ void __launch_bounds__(512, 2) qwen4_hc_up_mix_q8_lr320_kernel(
    const block_q8_0* __restrict__ w_up_q8_0, // [hc * E, 320]
    const block_q8_1* __restrict__ lo_q8,     // [T, 10]
    const float* __restrict__ xn,             // [T, hc, E]
    int T,
    int hc,
    int E,
    float* __restrict__ mixed_out,
    block_q8_1* __restrict__ mixed_q8_opt
) {
    constexpr int D_BLK = 32;
    constexpr int NB = 10;
    constexpr int MAX_HC = 8;

    const int d0 = (int)blockIdx.x * D_BLK;
    if (d0 >= E || T <= 0) return;

    const int warp = (int)threadIdx.y; // 0..15
    const int lane = (int)threadIdx.x; // 0..31
    const int tid = warp * 32 + lane;  // 0..511

    __shared__ __align__(16) uint32_t s_lo_words[TM_CAP][NB * 9];
    __shared__ __align__(16) float s_up[TM_CAP][MAX_HC][D_BLK];

    const int tot_words = T * (NB * 9);
    const uint32_t* g_lo = reinterpret_cast<const uint32_t*>(lo_q8);
    for (int i = tid; i < tot_words; i += 512) {
        const int tm = i / (NB * 9);
        const int w = i - tm * (NB * 9);
        s_lo_words[tm][w] = g_lo[i];
    }
    __syncthreads();

    const int sub = lane & 7;  // 0..7
    const int blk0 = lane >> 3; // 0..3

    int xa0[TM_CAP], xa1[TM_CAP], xa2[TM_CAP];
    float xd0[TM_CAP], xd1[TM_CAP], xd2[TM_CAP];
    #pragma unroll
    for (int tm = 0; tm < TM_CAP; ++tm) {
        if (EXACT_M || tm < T) {
            const block_q8_1* bx = reinterpret_cast<const block_q8_1*>(s_lo_words[tm]);
            xa0[tm] = *reinterpret_cast<const int*>(bx[blk0].qs + sub * 4);
            xd0[tm] = fp16_bits_to_f32(*reinterpret_cast<const uint16_t*>(&bx[blk0].d));
            xa1[tm] = *reinterpret_cast<const int*>(bx[blk0 + 4].qs + sub * 4);
            xd1[tm] = fp16_bits_to_f32(*reinterpret_cast<const uint16_t*>(&bx[blk0 + 4].d));
            if (blk0 < 2) {
                xa2[tm] = *reinterpret_cast<const int*>(bx[blk0 + 8].qs + sub * 4);
                xd2[tm] = fp16_bits_to_f32(*reinterpret_cast<const uint16_t*>(&bx[blk0 + 8].d));
            } else {
                xa2[tm] = 0;
                xd2[tm] = 0.0f;
            }
        }
    }

    auto ldg_i32_a2 = [&](const void* ptr) -> int {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(ptr);
        return (int)__ldg(p) | ((int)__ldg(p + 1) << 16);
    };

    #pragma unroll
    for (int step = 0; step < 2; ++step) {
        const int d_loc = warp + step * 16;
        const int d = d0 + d_loc;
        if (d < E) {
            for (int c = 0; c < hc; ++c) {
                const block_q8_0* w_row = w_up_q8_0 + ((size_t)c * (size_t)E + (size_t)d) * NB;
                const block_q8_0* wb0 = w_row + blk0;
                const int wa0 = ldg_i32_a2(wb0->qs + sub * 4);
                const float dw0 = fp16_bits_to_f32(__ldg(&wb0->d));

                const block_q8_0* wb1 = w_row + blk0 + 4;
                const int wa1 = ldg_i32_a2(wb1->qs + sub * 4);
                const float dw1 = fp16_bits_to_f32(__ldg(&wb1->d));

                int wa2 = 0;
                float dw2 = 0.0f;
                if (blk0 < 2) {
                    const block_q8_0* wb2 = w_row + blk0 + 8;
                    wa2 = ldg_i32_a2(wb2->qs + sub * 4);
                    dw2 = fp16_bits_to_f32(__ldg(&wb2->d));
                }

                #pragma unroll
                for (int tm = 0; tm < TM_CAP; ++tm) {
                    if (EXACT_M || tm < T) {
                        float acc = (dw0 * xd0[tm]) * (float)dp4a_packed(wa0, xa0[tm], 0)
                                  + (dw1 * xd1[tm]) * (float)dp4a_packed(wa1, xa1[tm], 0);
                        if (blk0 < 2) {
                            acc += (dw2 * xd2[tm]) * (float)dp4a_packed(wa2, xa2[tm], 0);
                        }
                        const float dot = warp_reduce_sum(acc);
                        if (lane == 0) {
                            s_up[tm][c][d_loc] = qwen4_sigmoid(dot);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    const float inv_hc = 1.0f / (float)hc;
    for (int tm = warp; tm < T; tm += 16) {
        const int d = d0 + lane;
        float y = 0.0f;
        if (d < E) {
            const size_t t_base = (size_t)tm * (size_t)hc * (size_t)E + (size_t)d;
            float sum = 0.0f;
            for (int c = 0; c < hc; ++c) {
                sum = fmaf(xn[t_base + (size_t)c * (size_t)E], s_up[tm][c][lane], sum);
            }
            y = inv_hc * sum;
            if (mixed_out != nullptr) {
                mixed_out[(size_t)tm * (size_t)E + (size_t)d] = y;
            }
        }
        if (mixed_q8_opt != nullptr && (d0 + 31 < E)) {
            block_q8_1* dst_blk = mixed_q8_opt + (size_t)tm * (size_t)(E >> 5) + blockIdx.x;
            warp_quantize_q8_1_store(y, dst_blk, lane);
        }
    }
}

template <int TM_CAP>
__global__ void __launch_bounds__(256, 4) qwen4_hc_up_mix_q8_kernel(
    const block_q8_0* __restrict__ w_up_q8_0, // [hc * E, lr] row-major blocks of 32
    const block_q8_1* __restrict__ lo_q8,     // [T, lr / 32]
    const float* __restrict__ xn,             // [T, hc, E]
    int T,
    int hc,
    int E,
    int lr,
    float* __restrict__ mixed_out,
    block_q8_1* __restrict__ mixed_q8_opt
) {
    constexpr int D_BLK = 32;
    constexpr int MAX_NB = 16; // up to lr = 512 (Qwen4 uses lr = 320 -> nb = 10)
    constexpr int MAX_HC = 8;

    const int d0 = (int)blockIdx.x * D_BLK;
    const int t0 = (int)blockIdx.y * TM_CAP;
    const int tm_act = min(TM_CAP, T - t0);
    if (d0 >= E || tm_act <= 0) return;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5; // 0..7
    const int lane = tid & 31; // 0..31
    const int nb = lr >> 5;    // lr / 32

    __shared__ __align__(16) uint32_t s_lo_words[TM_CAP][MAX_NB * 9];
    __shared__ float s_lo_d[TM_CAP][MAX_NB];
    __shared__ float s_up[TM_CAP][MAX_HC][D_BLK];

    const int lo_words_per_tok = nb * 9;
    const uint32_t* g_lo = reinterpret_cast<const uint32_t*>(lo_q8 + (size_t)t0 * nb);
    const int total_lo_words = tm_act * lo_words_per_tok;
    for (int i = tid; i < total_lo_words; i += 256) {
        const int tm = i / lo_words_per_tok;
        const int w = i - tm * lo_words_per_tok;
        s_lo_words[tm][w] = g_lo[(size_t)tm * lo_words_per_tok + w];
    }
    __syncthreads();
    const int total_lo_blks = tm_act * nb;
    for (int i = tid; i < total_lo_blks; i += 256) {
        const int tm = i / nb;
        const int b = i - tm * nb;
        const block_q8_1* xb = reinterpret_cast<const block_q8_1*>(s_lo_words[tm]) + b;
        s_lo_d[tm][b] = fp16_bits_to_f32(load_u16_align2(&xb->d));
    }
    __syncthreads();

    const int sub = lane & 7;  // 0..7 (8 int32 words per block_q8_0 / block_q8_1)
    const int blk0 = lane >> 3; // 0..3

    // 8 warps handle 4 d-columns each -> 32 columns per block
    #pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int d_loc = warp + step * 8;
        const int d = d0 + d_loc;
        if (d < E) {
            for (int c = 0; c < hc; ++c) {
                const block_q8_0* w_row = w_up_q8_0 + ((size_t)c * (size_t)E + (size_t)d) * (size_t)nb;
                float acc[TM_CAP];
                #pragma unroll
                for (int tm = 0; tm < TM_CAP; ++tm) acc[tm] = 0.0f;

                for (int b = blk0; b < nb; b += 4) {
                    const block_q8_0* wb = w_row + b;
                    const int wa = load_i32_align2(wb->qs + sub * 4);
                    const float dw = fp16_bits_to_f32(load_u16_align2(&wb->d));
                    #pragma unroll
                    for (int tm = 0; tm < TM_CAP; ++tm) {
                        if (tm < tm_act) {
                            const block_q8_1* xb = reinterpret_cast<const block_q8_1*>(s_lo_words[tm]) + b;
                            const int xa = load_i32_align4(xb->qs + sub * 4);
                            const int sumi = dp4a_packed(wa, xa, 0);
                            const float sc = dw * s_lo_d[tm][b];
                            acc[tm] += sc * (float)sumi;
                        }
                    }
                }
                #pragma unroll
                for (int tm = 0; tm < TM_CAP; ++tm) {
                    if (tm < tm_act) {
                        const float dot = warp_reduce_sum(acc[tm]);
                        if (lane == 0) {
                            s_up[tm][c][d_loc] = qwen4_sigmoid(dot);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // Wait-free per-token mix + Q8_1 quantize across D_BLK = 32 lanes
    const float inv_hc = 1.0f / (float)hc;
    for (int tm = warp; tm < tm_act; tm += 8) {
        const int t = t0 + tm;
        const int d = d0 + lane;
        float y = 0.0f;
        if (d < E) {
            const size_t t_base = (size_t)t * (size_t)hc * (size_t)E + (size_t)d;
            float sum = 0.0f;
            for (int c = 0; c < hc; ++c) {
                sum = fmaf(xn[t_base + (size_t)c * (size_t)E], s_up[tm][c][lane], sum);
            }
            y = inv_hc * sum;
            if (mixed_out != nullptr) {
                mixed_out[(size_t)t * (size_t)E + (size_t)d] = y;
            }
        }
        if (mixed_q8_opt != nullptr && (d0 + 31 < E)) {
            block_q8_1* dst_blk = mixed_q8_opt + (size_t)t * (size_t)(E >> 5) + blockIdx.x;
            warp_quantize_q8_1_store(y, dst_blk, lane);
        }
    }
}

__global__ void __launch_bounds__(256) qwen4_router_doorbell_plan_kernel(
    const float* __restrict__ logits,     // [T, n_exp]
    int T,
    int n_exp,
    int topk,
    const uint64_t* __restrict__ vram_ptrs, // [n_exp]
    const uint64_t* __restrict__ ram_ptrs,  // nullable [n_exp]
    int gpu_pcie_div,
    int gpu_pcie_max,
    const float* __restrict__ mixed_f32,  // [T, n_embd]
    int n_embd,
    int n_ff,
    int layer_idx,
    int32_t gu_type,
    int32_t d_type,
    int32_t* __restrict__ ids_out,        // [T, topk]
    float* __restrict__ weights_out,      // [T, topk]
    uint64_t* __restrict__ plan_ptrs_out, // [T, topk]
    Qwen4Doorbell* __restrict__ doorbell,
    uint64_t* __restrict__ seq_state_dev,
    uint64_t* __restrict__ expected_seq_out
) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ int32_t s_miss_cnt;
    __shared__ uint64_t s_exp_seq;

    if (warp < T) {
        const float* x = logits + (size_t)warp * (size_t)n_exp;
        int32_t* id_dst = ids_out + (size_t)warp * (size_t)topk;
        float* wt_dst = weights_out + (size_t)warp * (size_t)topk;

        float values[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = lane + i * 32;
            values[i] = (idx < n_exp) ? x[idx] : -INFINITY;
        }

        float maximum = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 16; ++i) maximum = fmaxf(maximum, values[i]);
        maximum = warp_reduce_max(maximum);

        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = lane + i * 32;
            const float e = (idx < n_exp) ? expf(values[i] - maximum) : 0.0f;
            values[i] = (idx < n_exp) ? e : -INFINITY;
            sum += e;
        }
        const float reciprocal = 1.0f / warp_reduce_sum(sum);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = lane + i * 32;
            if (idx < n_exp) {
                values[i] *= reciprocal;
                if (isnan(values[i])) values[i] = -FLT_MAX;
            }
        }

        float selected = 0.0f;
        float selected_sum = 0.0f;
        for (int rank = 0; rank < topk; ++rank) {
            float best = values[0];
            int expert = lane;
            #pragma unroll
            for (int i = 1; i < 16; ++i) {
                if (values[i] > best) {
                    best = values[i];
                    expert = lane + i * 32;
                }
            }
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, best, mask);
                const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask);
                if (other > best || (other == best && other_id < expert)) {
                    best = other;
                    expert = other_id;
                }
            }
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                id_dst[rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }

        selected_sum = fmaxf(warp_reduce_sum(selected_sum), 6.103515625e-5f);
        const float inv = 1.0f / selected_sum;
        if (lane < topk) {
            wt_dst[lane] = selected * inv;
        }
    }
    __syncthreads();

    if (tid == 0) {
        int32_t uniq_eid[80];
        uint8_t uniq_gpu[80];
        int n_uniq = 0;
        const int tot_pairs = T * topk;

        for (int idx = 0; idx < tot_pairs; ++idx) {
            const int eid = ids_out[idx];
            if (layer_idx >= 0 && layer_idx < 48 && idx < 80) {
                doorbell->routed_ids[layer_idx][idx] = eid;
            }
            const uint64_t vp = (eid >= 0 && eid < n_exp) ? vram_ptrs[eid] : 0ULL;
            if (vp == 0ULL && eid >= 0 && eid < n_exp) {
                bool found = false;
                for (int u = 0; u < n_uniq; ++u) {
                    if (uniq_eid[u] == eid) {
                        found = true;
                        break;
                    }
                }
                if (!found && n_uniq < 80) {
                    uniq_eid[n_uniq] = eid;
                    uniq_gpu[n_uniq] = 0;
                    ++n_uniq;
                }
            }
        }
        if (layer_idx >= 0 && layer_idx < 48) {
            doorbell->routed_n_pairs[layer_idx] = (tot_pairs <= 80) ? tot_pairs : 80;
        }

        if (gpu_pcie_div > 0 && ram_ptrs != nullptr && gpu_pcie_max > 0) {
            int max_gpu = n_uniq / gpu_pcie_div;
            if (max_gpu > gpu_pcie_max) max_gpu = gpu_pcie_max;
            int assigned = 0;
            for (int u = 0; u < n_uniq && assigned < max_gpu; ++u) {
                if (ram_ptrs[uniq_eid[u]] != 0ULL) {
                    uniq_gpu[u] = 1;
                    ++assigned;
                }
            }
        }

        int mc = 0;
        for (int idx = 0; idx < tot_pairs; ++idx) {
            const int eid = ids_out[idx];
            const uint64_t vp = (eid >= 0 && eid < n_exp) ? vram_ptrs[eid] : 0ULL;
            if (vp != 0ULL) {
                plan_ptrs_out[idx] = vp;
            } else {
                uint64_t gp = 0ULL;
                if (eid >= 0 && eid < n_exp && ram_ptrs != nullptr) {
                    for (int u = 0; u < n_uniq; ++u) {
                        if (uniq_eid[u] == eid && uniq_gpu[u] != 0) {
                            gp = ram_ptrs[eid];
                            break;
                        }
                    }
                }
                plan_ptrs_out[idx] = gp;
                if (gp == 0ULL && mc < 80) {
                    doorbell->miss_tok[mc] = idx / topk;
                    doorbell->miss_ids[mc] = eid;
                    doorbell->miss_weights[mc] = weights_out[idx];
                    ++mc;
                }
            }
        }

        doorbell->m_batch = T;
        doorbell->miss_count = mc;
        if (mc > 0) {
            const uint64_t next_s = (*seq_state_dev) + 1ULL;
            *seq_state_dev = next_s;
            doorbell->layer_idx = layer_idx;
            doorbell->gu_type = gu_type;
            doorbell->d_type = d_type;
            doorbell->n_embd = n_embd;
            doorbell->n_ff = n_ff;
            s_exp_seq = next_s;
        } else {
            s_exp_seq = 0ULL;
        }
        s_miss_cnt = mc;
        if (expected_seq_out != nullptr) *expected_seq_out = s_exp_seq;
    }
    __syncthreads();

    if (s_miss_cnt > 0 && mixed_f32 != nullptr) {
        const int tot_x = T * n_embd;
        for (int i = tid; i < tot_x; i += 256) {
            doorbell->x_in[i] = mixed_f32[i];
        }
        __syncthreads();
        if (tid == 0) {
            __threadfence_system();
            doorbell->req_seq = s_exp_seq;
            __threadfence_system();
        }
    }
}

__global__ void __launch_bounds__(256) qwen4_moe_fetch_unique_kernel(
    const uint64_t* __restrict__ fetch_src_ptrs,
    const uint64_t* __restrict__ fetch_dst_ptrs,
    size_t blob_bytes,
    int fetch_cap
) {
    const int nblk = (int)gridDim.x;
    const int bid = (int)blockIdx.x;
    const int ltid = (int)threadIdx.x;
    constexpr int bdim = 256;
    const size_t gid = (size_t)bid * bdim + (size_t)ltid;
    const size_t gstride = (size_t)nblk * bdim;

    const size_t n16 = blob_bytes >> 4;
    const size_t tail16 = n16 << 4;
    const size_t per_blk = (n16 + (size_t)nblk - 1) / (size_t)nblk;
    const size_t c_start = (size_t)bid * per_blk;
    const size_t c_end = (c_start < n16) ? min(n16, c_start + per_blk) : c_start;
    const size_t step4 = (size_t)bdim * 4;

    const size_t n4 = blob_bytes >> 2;
    const size_t tail4 = n4 << 2;

    for (int slot = 0; slot < fetch_cap; ++slot) {
        const uint64_t s_addr = fetch_src_ptrs[slot];
        const uint64_t d_addr = fetch_dst_ptrs[slot];
        if (s_addr == 0ULL || d_addr == 0ULL) continue;

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
            for (size_t t = tail16 + gid; t < blob_bytes; t += gstride) {
                dp[t] = sp[t];
            }
        } else if (((s_addr | d_addr) & 3ULL) == 0ULL) {
            const uint32_t* sv = reinterpret_cast<const uint32_t*>(sp);
            uint32_t* dv = reinterpret_cast<uint32_t*>(dp);
            for (size_t j = gid; j < n4; j += gstride) {
                __stcs(dv + j, __ldcs(sv + j));
            }
            for (size_t t = tail4 + gid; t < blob_bytes; t += gstride) {
                dp[t] = sp[t];
            }
        } else {
            for (size_t j = gid; j < blob_bytes; j += gstride) {
                dp[j] = sp[j];
            }
        }
    }
}

__global__ void __launch_bounds__(256) qwen4_router_doorbell_plan_dedup_kernel(
    const float* __restrict__ logits,     // [T, n_exp]
    int T,
    int n_exp,
    int topk,
    const uint64_t* __restrict__ vram_ptrs, // [n_exp]
    const uint64_t* __restrict__ ram_ptrs,  // nullable [n_exp]
    int gpu_pcie_div,
    int fetch_cap,
    uint64_t dma_stage_base,
    size_t slot_stride,
    const float* __restrict__ mixed_f32,  // [T, n_embd]
    int n_embd,
    int n_ff,
    int layer_idx,
    int32_t gu_type,
    int32_t d_type,
    int32_t* __restrict__ ids_out,        // [T, topk]
    float* __restrict__ weights_out,      // [T, topk]
    uint64_t* __restrict__ res_ptrs_out,  // [T * topk]
    int32_t* __restrict__ res_off_out,    // [T * topk]
    int32_t* __restrict__ res_nt_out,     // [T * topk]
    int32_t* __restrict__ res_rows_out,   // [T * topk] (tok idx for small_gu)
    int32_t* __restrict__ res_pair_out,   // [T * topk] (pair idx for deterministic small_down)
    float* __restrict__ res_wts_out,      // [T * topk]
    uint64_t* __restrict__ fetch_src_out, // [fetch_cap]
    uint64_t* __restrict__ fetch_dst_out, // [fetch_cap]
    int32_t* __restrict__ fetch_off_out,  // [fetch_cap]
    int32_t* __restrict__ fetch_nt_out,   // [fetch_cap]
    int32_t* __restrict__ fetch_rows_out, // [T * topk]
    int32_t* __restrict__ fetch_pair_out, // [T * topk]
    float* __restrict__ fetch_wts_out,    // [T * topk]
    uint16_t* __restrict__ gpu_pair_mask_out, // [T] bitmask of GPU-computed pairs per tok
    Qwen4Doorbell* __restrict__ doorbell,
    uint64_t* __restrict__ seq_state_dev,
    uint64_t* __restrict__ expected_seq_out
) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    __shared__ int32_t s_miss_cnt;
    __shared__ uint64_t s_exp_seq;
    __shared__ int16_t s_seen_res[512];
    __shared__ int16_t s_seen_cold[512];
    __shared__ int32_t s_ids[80];
    __shared__ float   s_wt[80];
    __shared__ uint64_t s_vp[80];
    __shared__ int32_t s_miss_tok[80];
    __shared__ int32_t s_miss_ids[80];
    __shared__ float   s_miss_w[80];
    __shared__ int32_t s_res_eid[80];
    __shared__ int32_t s_res_cnt[80];
    __shared__ int32_t s_res_cur[80];
    __shared__ int32_t s_cold_eid[80];
    __shared__ int32_t s_cold_cnt[80];
    __shared__ int32_t s_cold_slot[80];
    __shared__ int32_t s_fetch_cur[8];

    for (int i = tid; i < 512; i += 256) {
        s_seen_res[i] = -1;
        s_seen_cold[i] = -1;
    }

    if (warp < T) {
        const float* x = logits + (size_t)warp * (size_t)n_exp;
        int32_t* id_dst = ids_out + (size_t)warp * (size_t)topk;
        float* wt_dst = weights_out + (size_t)warp * (size_t)topk;

        float values[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = lane + i * 32;
            values[i] = (idx < n_exp) ? x[idx] : -INFINITY;
        }

        float maximum = -INFINITY;
        #pragma unroll
        for (int i = 0; i < 16; ++i) maximum = fmaxf(maximum, values[i]);
        maximum = warp_reduce_max(maximum);

        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = lane + i * 32;
            const float e = (idx < n_exp) ? expf(values[i] - maximum) : 0.0f;
            values[i] = (idx < n_exp) ? e : -INFINITY;
            sum += e;
        }
        const float reciprocal = 1.0f / warp_reduce_sum(sum);
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int idx = lane + i * 32;
            if (idx < n_exp) {
                values[i] *= reciprocal;
                if (isnan(values[i])) values[i] = -FLT_MAX;
            }
        }

        float selected = 0.0f;
        float selected_sum = 0.0f;
        for (int rank = 0; rank < topk; ++rank) {
            float best = values[0];
            int expert = lane;
            #pragma unroll
            for (int i = 1; i < 16; ++i) {
                if (values[i] > best) {
                    best = values[i];
                    expert = lane + i * 32;
                }
            }
            #pragma unroll
            for (int mask = 16; mask > 0; mask >>= 1) {
                const float other = __shfl_xor_sync(0xffffffffu, best, mask);
                const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask);
                if (other > best || (other == best && other_id < expert)) {
                    best = other;
                    expert = other_id;
                }
            }
            if ((expert & 31) == lane) {
                values[expert / 32] = -INFINITY;
                id_dst[rank] = expert;
                selected_sum += best;
            }
            if (rank == lane) selected = best;
        }

        selected_sum = fmaxf(warp_reduce_sum(selected_sum), 6.103515625e-5f);
        const float inv = 1.0f / selected_sum;
        if (lane < topk) {
            wt_dst[lane] = selected * inv;
        }
    }
    __syncthreads();

    const int tot_pairs = T * topk;
    if (tid < tot_pairs && tid < 80) {
        const int eid = ids_out[tid];
        s_ids[tid] = eid;
        s_wt[tid] = weights_out[tid];
        s_vp[tid] = (eid >= 0 && eid < n_exp) ? vram_ptrs[eid] : 0ULL;
    }
    __syncthreads();

    if (tid == 0) {
        int n_res = 0;
        int n_cold = 0;
        const int n_lim = (tot_pairs <= 80) ? tot_pairs : 80;

        for (int idx = 0; idx < n_lim; ++idx) {
            const int eid = s_ids[idx];
            const uint64_t vp = s_vp[idx];
            if (vp != 0ULL) {
                int u = (eid >= 0 && eid < 512) ? (int)s_seen_res[eid] : -1;
                if (u < 0 && n_res < 80) {
                    u = n_res++;
                    if (eid >= 0 && eid < 512) s_seen_res[eid] = (int16_t)u;
                    s_res_eid[u] = eid;
                    s_res_cnt[u] = 1;
                } else if (u >= 0) {
                    ++s_res_cnt[u];
                }
            } else if (eid >= 0 && eid < n_exp) {
                int u = (eid < 512) ? (int)s_seen_cold[eid] : -1;
                if (u < 0 && n_cold < 80) {
                    u = n_cold++;
                    if (eid < 512) s_seen_cold[eid] = (int16_t)u;
                    s_cold_eid[u] = eid;
                    s_cold_cnt[u] = 1;
                    s_cold_slot[u] = -1;
                } else if (u >= 0) {
                    ++s_cold_cnt[u];
                }
            }
        }

        int res_cursor = 0;
        for (int u = 0; u < n_res; ++u) {
            const int eid = s_res_eid[u];
            const int cnt = s_res_cnt[u];
            res_ptrs_out[u] = vram_ptrs[eid];
            res_off_out[u] = res_cursor;
            res_nt_out[u] = cnt;
            s_res_cur[u] = res_cursor;
            res_cursor += cnt;
        }
        for (int u = n_res; u < tot_pairs; ++u) {
            res_ptrs_out[u] = 0ULL;
            res_off_out[u] = 0;
            res_nt_out[u] = 0;
        }

        int n_fetch = 0;
        if (gpu_pcie_div > 0 && ram_ptrs != nullptr && fetch_cap > 0 && dma_stage_base != 0ULL) {
            const int cap_cap = (fetch_cap < 8) ? fetch_cap : 8;
            int max_gpu = n_cold / gpu_pcie_div;
            if (max_gpu > cap_cap) max_gpu = cap_cap;
            int fetch_cursor = 0;
            for (int pick = 0; pick < max_gpu; ++pick) {
                int best_u = -1;
                int best_cnt = 0;
                for (int u = 0; u < n_cold; ++u) {
                    if (s_cold_slot[u] < 0 && ram_ptrs[s_cold_eid[u]] != 0ULL && s_cold_cnt[u] > best_cnt) {
                        best_cnt = s_cold_cnt[u];
                        best_u = u;
                    }
                }
                if (best_u < 0) break;
                const int eid = s_cold_eid[best_u];
                const uint64_t dst_u = dma_stage_base + (uint64_t)n_fetch * (uint64_t)slot_stride;
                s_cold_slot[best_u] = n_fetch;
                fetch_src_out[n_fetch] = ram_ptrs[eid];
                fetch_dst_out[n_fetch] = dst_u;
                fetch_off_out[n_fetch] = fetch_cursor;
                fetch_nt_out[n_fetch] = best_cnt;
                s_fetch_cur[n_fetch] = fetch_cursor;
                fetch_cursor += best_cnt;
                ++n_fetch;
            }
        }
        for (int s = n_fetch; s < fetch_cap; ++s) {
            fetch_src_out[s] = 0ULL;
            fetch_dst_out[s] = 0ULL;
            fetch_off_out[s] = 0;
            fetch_nt_out[s] = 0;
        }

        uint16_t mask[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int mc = 0;
        for (int idx = 0; idx < n_lim; ++idx) {
            const int tok = idx / topk;
            const int rank = idx - tok * topk;
            const int eid = s_ids[idx];
            const float w = s_wt[idx];
            const uint64_t vp = s_vp[idx];
            if (vp != 0ULL) {
                const int u = (eid >= 0 && eid < 512) ? (int)s_seen_res[eid] : -1;
                if (u >= 0) {
                    const int pos = s_res_cur[u]++;
                    res_rows_out[pos] = tok;
                    res_pair_out[pos] = idx;
                    res_wts_out[pos] = w;
                    if (tok < 8 && rank < 16) mask[tok] |= (uint16_t)(1u << rank);
                }
            } else if (eid >= 0 && eid < n_exp) {
                const int u = (eid < 512) ? (int)s_seen_cold[eid] : -1;
                const int fslot = (u >= 0) ? s_cold_slot[u] : -1;
                if (fslot >= 0) {
                    const int pos = s_fetch_cur[fslot]++;
                    fetch_rows_out[pos] = tok;
                    fetch_pair_out[pos] = idx;
                    fetch_wts_out[pos] = w;
                    if (tok < 8 && rank < 16) mask[tok] |= (uint16_t)(1u << rank);
                } else if (mc < 80) {
                    s_miss_tok[mc] = tok;
                    s_miss_ids[mc] = eid;
                    s_miss_w[mc] = w;
                    ++mc;
                }
            }
        }
        for (int t = 0; t < T; ++t) {
            gpu_pair_mask_out[t] = (t < 8) ? mask[t] : 0u;
        }

        if (mc > 0) {
            const uint64_t next_s = (*seq_state_dev) + 1ULL;
            *seq_state_dev = next_s;
            s_exp_seq = next_s;
        } else {
            s_exp_seq = 0ULL;
        }
        s_miss_cnt = mc;
        if (expected_seq_out != nullptr) *expected_seq_out = s_exp_seq;
    }
    __syncthreads();

    if (tid < tot_pairs && tid < 80 && layer_idx >= 0 && layer_idx < 48) {
        doorbell->routed_ids[layer_idx][tid] = s_ids[tid];
    }
    if (tid < s_miss_cnt) {
        doorbell->miss_tok[tid] = s_miss_tok[tid];
        doorbell->miss_ids[tid] = s_miss_ids[tid];
        doorbell->miss_weights[tid] = s_miss_w[tid];
    }
    if (s_miss_cnt > 0 && mixed_f32 != nullptr) {
        const int tot_x = T * n_embd;
        if (((reinterpret_cast<uintptr_t>(mixed_f32) | reinterpret_cast<uintptr_t>(doorbell->x_in)) & 15ULL) == 0ULL && (tot_x & 3) == 0) {
            const float4* sx = reinterpret_cast<const float4*>(mixed_f32);
            float4* dx = reinterpret_cast<float4*>(doorbell->x_in);
            const int tot4 = tot_x >> 2;
            for (int i = tid; i < tot4; i += 256) {
                dx[i] = sx[i];
            }
        } else {
            for (int i = tid; i < tot_x; i += 256) {
                doorbell->x_in[i] = mixed_f32[i];
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        if (layer_idx >= 0 && layer_idx < 48) {
            doorbell->routed_n_pairs[layer_idx] = (tot_pairs <= 80) ? tot_pairs : 80;
        }
        doorbell->m_batch = T;
        doorbell->miss_count = s_miss_cnt;
        if (s_miss_cnt > 0) {
            doorbell->layer_idx = layer_idx;
            doorbell->gu_type = gu_type;
            doorbell->d_type = d_type;
            doorbell->n_embd = n_embd;
            doorbell->n_ff = n_ff;
            __threadfence_system();
            doorbell->req_seq = s_exp_seq;
            __threadfence_system();
        }
    }
}

__global__ void __launch_bounds__(32) qwen4_doorbell_wait_kernel(
    const Qwen4Doorbell* __restrict__ doorbell,
    const uint64_t* __restrict__ expected_seq_ptr
) {
    if (threadIdx.x == 0) {
        const uint64_t exp_seq = *expected_seq_ptr;
        if (exp_seq != 0ULL) {
            const volatile uint64_t* done_p = &doorbell->done_seq;
            while (*done_p < exp_seq) {}
            __threadfence_system();
        }
    }
}

__global__ void __launch_bounds__(128) qwen4_moe_reduce_hc_write_kernel(
    const Qwen4Doorbell* __restrict__ doorbell,
    const uint64_t* __restrict__ expected_seq_ptr,
    float* __restrict__ R,
    const float* __restrict__ block_out_opt,
    const float* __restrict__ pair_out_opt,
    const uint16_t* __restrict__ gpu_pair_mask_opt,
    int topk,
    const float* __restrict__ sh_out_opt,
    const float* __restrict__ sh_gate_logit_opt,
    const float* __restrict__ inject,
    int inject_stride,
    int hc,
    int E
) {
    const int d = (int)blockIdx.x * 128 + (int)threadIdx.x;
    const int t = (int)blockIdx.y;
    if (d >= E) return;

    __shared__ float s_wc[8];
    __shared__ float s_sg;
    __shared__ int s_has_cpu;
    __shared__ uint16_t s_pmask;

    const int tid = (int)threadIdx.x;
    if (tid < hc) {
        float inj_val;
        if (inject_stride <= 1) {
            inj_val = inject[(size_t)t * (size_t)hc + (size_t)tid];
        } else {
            const float* ip = inject + ((size_t)t * (size_t)hc + (size_t)tid) * (size_t)inject_stride;
            float s = 0.0f;
            for (int k = 0; k < hc; ++k) s += ip[k];
            inj_val = s;
        }
        s_wc[tid] = 2.0f * qwen4_sigmoid(inj_val / (float)hc);
    }
    if (tid == 0) {
        s_sg = (sh_gate_logit_opt != nullptr) ? qwen4_sigmoid(sh_gate_logit_opt[t]) : 1.0f;
        s_has_cpu = (*expected_seq_ptr != 0ULL) ? 1 : 0;
        s_pmask = (gpu_pair_mask_opt != nullptr) ? gpu_pair_mask_opt[t] : 0u;
    }
    __syncthreads();

    const size_t bo_idx = (size_t)t * (size_t)E + (size_t)d;
    float y = 0.0f;
    if (pair_out_opt != nullptr) {
        const uint16_t pm = s_pmask;
        const size_t p_base = (size_t)t * (size_t)topk * (size_t)E + (size_t)d;
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            if (k >= topk) break;
            if ((pm >> k) & 1u) {
                y += pair_out_opt[p_base + (size_t)k * (size_t)E];
            }
        }
    } else if (block_out_opt != nullptr) {
        y = block_out_opt[bo_idx];
    }
    if (s_has_cpu) {
        y += doorbell->out[bo_idx];
    }
    if (sh_out_opt != nullptr) {
        y += sh_out_opt[bo_idx] * s_sg;
    }
    for (int c = 0; c < hc; ++c) {
        const size_t r_idx = ((size_t)t * (size_t)hc + (size_t)c) * (size_t)E + (size_t)d;
        R[r_idx] = fmaf(y, s_wc[c], R[r_idx]);
    }
}

__global__ void __launch_bounds__(256) qwen4_argmax_prob_stage1_kernel(
    const float* __restrict__ logits,
    int n_vocab,
    float* __restrict__ partial_max,
    int32_t* __restrict__ partial_idx,
    float* __restrict__ partial_sum
) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int stride = (int)gridDim.x * 256;

    float best_v = -INFINITY;
    int best_i = 0;
    float l_sum = 0.0f;

    for (int i = (int)blockIdx.x * 256 + tid; i < n_vocab; i += stride) {
        const float v = logits[i];
        if (!isnan(v)) {
            if (v > best_v) {
                l_sum = isfinite(best_v) ? (l_sum * expf(best_v - v) + 1.0f) : 1.0f;
                best_v = v;
                best_i = i;
            } else {
                l_sum += expf(v - best_v);
            }
        }
    }

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best_v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
        const float ol = __shfl_xor_sync(0xffffffffu, l_sum, o);
        const float nm = fmaxf(best_v, ov);
        if (isfinite(nm)) {
            const float l1 = isfinite(best_v) ? (l_sum * expf(best_v - nm)) : 0.0f;
            const float l2 = isfinite(ov) ? (ol * expf(ov - nm)) : 0.0f;
            l_sum = l1 + l2;
        } else {
            l_sum = 0.0f;
        }
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    __shared__ float s_v[8];
    __shared__ int32_t s_i[8];
    __shared__ float s_l[8];
    if (lane == 0) {
        s_v[warp] = best_v;
        s_i[warp] = best_i;
        s_l[warp] = l_sum;
    }
    __syncthreads();

    if (tid == 0) {
        float bv = -INFINITY;
        int bi = 0;
        float bl = 0.0f;
        for (int w = 0; w < 8; ++w) {
            const float ov = s_v[w];
            const int oi = s_i[w];
            const float ol = s_l[w];
            const float nm = fmaxf(bv, ov);
            if (isfinite(nm)) {
                const float l1 = isfinite(bv) ? (bl * expf(bv - nm)) : 0.0f;
                const float l2 = isfinite(ov) ? (ol * expf(ov - nm)) : 0.0f;
                bl = l1 + l2;
            }
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        partial_max[blockIdx.x] = bv;
        partial_idx[blockIdx.x] = bi;
        partial_sum[blockIdx.x] = bl;
    }
}

__global__ void __launch_bounds__(128) qwen4_argmax_prob_stage2_kernel(
    const float* __restrict__ partial_max,
    const int32_t* __restrict__ partial_idx,
    const float* __restrict__ partial_sum,
    int n_parts,
    const int32_t* __restrict__ sub_to_id_opt,
    int32_t* __restrict__ out_id,
    float* __restrict__ out_prob
) {
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    float bv = (tid < n_parts) ? partial_max[tid] : -INFINITY;
    int bi = (tid < n_parts) ? partial_idx[tid] : 0;
    float bl = (tid < n_parts) ? partial_sum[tid] : 0.0f;

    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
        const float ol = __shfl_xor_sync(0xffffffffu, bl, o);
        const float nm = fmaxf(bv, ov);
        if (isfinite(nm)) {
            const float l1 = isfinite(bv) ? (bl * expf(bv - nm)) : 0.0f;
            const float l2 = isfinite(ov) ? (ol * expf(ov - nm)) : 0.0f;
            bl = l1 + l2;
        } else {
            bl = 0.0f;
        }
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }

    __shared__ float s_v[4];
    __shared__ int32_t s_i[4];
    __shared__ float s_l[4];
    if (lane == 0) {
        s_v[warp] = bv;
        s_i[warp] = bi;
        s_l[warp] = bl;
    }
    __syncthreads();

    if (tid == 0) {
        float fv = -INFINITY;
        int fi = 0;
        float fl = 0.0f;
        for (int w = 0; w < 4; ++w) {
            const float ov = s_v[w];
            const int oi = s_i[w];
            const float ol = s_l[w];
            const float nm = fmaxf(fv, ov);
            if (isfinite(nm)) {
                const float l1 = isfinite(fv) ? (fl * expf(fv - nm)) : 0.0f;
                const float l2 = isfinite(ov) ? (ol * expf(ov - nm)) : 0.0f;
                fl = l1 + l2;
            }
            if (ov > fv || (ov == fv && oi < fi)) {
                fv = ov;
                fi = oi;
            }
        }
        *out_id = (sub_to_id_opt != nullptr) ? sub_to_id_opt[fi] : fi;
        if (out_prob != nullptr) {
            *out_prob = (fl > 0.0f) ? (1.0f / fl) : 0.0f;
        }
        __threadfence_system();
    }
}

__global__ void __launch_bounds__(256) qwen4_reduce_pairs_f32_kernel(
    const float* __restrict__ pair_out,
    float* __restrict__ block_out,
    int T,
    int topk,
    int n_embd
) {
    const int d = (int)blockIdx.x * 256 + (int)threadIdx.x;
    const int t = (int)blockIdx.y;
    if (t >= T || d >= n_embd) return;
    const size_t base = (size_t)t * (size_t)topk * (size_t)n_embd + (size_t)d;
    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        sum += pair_out[base + (size_t)k * (size_t)n_embd];
    }
    block_out[(size_t)t * (size_t)n_embd + (size_t)d] = sum;
}

__global__ void __launch_bounds__(256) qwen4_patch_u64_kernel(
    const uint64_t* __restrict__ pairs,
    int n
) {
    const int i = (int)blockIdx.x * 256 + (int)threadIdx.x;
    if (i < n) {
        *reinterpret_cast<uint64_t*>(static_cast<uintptr_t>(pairs[2 * i])) = pairs[2 * i + 1];
    }
}

__global__ void __launch_bounds__(256) qwen4_copy_u32_sm_kernel(
    uint32_t* __restrict__ dst,
    const uint32_t* __restrict__ src,
    int n_u32,
    int system_fence
) {
    const int tid = (int)blockIdx.x * 256 + (int)threadIdx.x;
    const int stride = (int)gridDim.x * 256;
    const bool aligned16 = (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15u) == 0u);
    if (aligned16 && n_u32 >= 4) {
        const int n4 = n_u32 >> 2;
        uint4* dst4 = reinterpret_cast<uint4*>(dst);
        const uint4* src4 = reinterpret_cast<const uint4*>(src);
        for (int i = tid; i < n4; i += stride) {
            dst4[i] = src4[i];
        }
        for (int i = (n4 << 2) + tid; i < n_u32; i += stride) {
            dst[i] = src[i];
        }
    } else {
        for (int i = tid; i < n_u32; i += stride) {
            dst[i] = src[i];
        }
    }
    if (system_fence != 0) {
        __threadfence_system();
    }
}

} // namespace

// ============================================================================
// C FFI Entrypoints
// ============================================================================

extern "C" {

void launch_qwen4_hc_expand(
    const float* emb, int T, int hc, int E, float* R_out, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    size_t total = (size_t)T * (size_t)hc * (size_t)E;
    qwen4_hc_expand_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(emb, hc, E, total, R_out);
}

void launch_qwen4_hc_rmsnorm_q8(
    const float* R, const float* w_norm, int T, int hc, int E, float eps,
    float* xn_out, void* xn_q8_opt, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    dim3 grid((unsigned)hc, (unsigned)T);
    qwen4_hc_rmsnorm_q8_kernel<false><<<grid, 256, 0, stream>>>(
        R, w_norm, nullptr, 0, hc, E, eps, xn_out, reinterpret_cast<block_q8_1*>(xn_q8_opt), nullptr
    );
}

void launch_qwen4_hc_rmsnorm_inject_q8(
    const float* R, const float* w_norm, const void* w_inject, int inject_w_type,
    int T, int hc, int E, float eps,
    float* xn_out, void* xn_q8_opt, float* inject_partials_out, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    dim3 grid((unsigned)hc, (unsigned)T);
    qwen4_hc_rmsnorm_q8_kernel<true><<<grid, 256, 0, stream>>>(
        R, w_norm, w_inject, inject_w_type, hc, E, eps, xn_out, reinterpret_cast<block_q8_1*>(xn_q8_opt), inject_partials_out
    );
}

void launch_qwen4_hc_down_silu_q8(
    const float* lo_in, int T, int hc, int lr,
    float* lo_out_opt, void* lo_q8_opt, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || lr <= 0) return;
    dim3 grid((unsigned)((lr + 31) / 32), (unsigned)T);
    qwen4_hc_down_silu_q8_kernel<<<grid, 32, 0, stream>>>(
        lo_in, hc, lr, lo_out_opt, reinterpret_cast<block_q8_1*>(lo_q8_opt)
    );
}

void launch_qwen4_hc_mix_q8(
    const float* xn, const float* up_logits, int T, int hc, int E,
    float* mixed_out, void* mixed_q8_opt, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    dim3 grid((unsigned)((E + 31) / 32), (unsigned)T);
    qwen4_hc_mix_q8_kernel<<<grid, 32, 0, stream>>>(
        xn, up_logits, hc, E, mixed_out, reinterpret_cast<block_q8_1*>(mixed_q8_opt)
    );
}

void launch_qwen4_hc_inject_dot(
    const float* xn, const void* w_inject, int w_type,
    int T, int hc, int E, float* inject_out, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    dim3 grid((unsigned)hc, (unsigned)T);
    qwen4_hc_inject_dot_kernel<<<grid, 256, 0, stream>>>(xn, w_inject, w_type, hc, E, inject_out);
}

void launch_qwen4_hc_write(
    float* R, const float* block_out, const float* sh_out_opt,
    const float* sh_gate_logit_opt, const float* inject,
    int T, int hc, int E, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    size_t total = (size_t)T * (size_t)hc * (size_t)E;
    qwen4_hc_write_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        R, block_out, sh_out_opt, sh_gate_logit_opt, inject, 1, T, hc, E
    );
}

void launch_qwen4_hc_write_strided(
    float* R, const float* block_out, const float* sh_out_opt,
    const float* sh_gate_logit_opt, const float* inject, int inject_stride,
    int T, int hc, int E, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    size_t total = (size_t)T * (size_t)hc * (size_t)E;
    qwen4_hc_write_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        R, block_out, sh_out_opt, sh_gate_logit_opt, inject, inject_stride, T, hc, E
    );
}

void launch_qwen4_add_streams_broadcast(
    const float* h, const float* e, int T, int hc, int E, float* R_out, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    size_t total = (size_t)T * (size_t)hc * (size_t)E;
    qwen4_add_streams_broadcast_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        h, e, total, hc, E, R_out
    );
}

void launch_qwen4_rmsnorm_f32(
    const float* x, const float* w, int rows, int cols, float eps,
    float* y_out, void* y_q8_opt, cudaStream_t stream
) {
    if (rows <= 0 || cols <= 0) return;
    qwen4_rmsnorm_f32_kernel<<<(unsigned)rows, 256, 0, stream>>>(
        x, w, cols, eps, y_out, reinterpret_cast<block_q8_1*>(y_q8_opt)
    );
}

void launch_qwen4_ple_postops(
    float* key, float* hidden_R, const float* value, float* history,
    const float* w_norm_key, const float* w_norm_query, const float* w_norm_conv,
    const float* w_conv1d_f32, float* scratch_qn, float* scratch_gated, float* scratch_gate,
    int T, int N, int H, float eps, float* snap_opt, cudaStream_t stream
) {
    if (T <= 0 || N <= 0 || H <= 0) return;
    const int rows = T * H;
    const int D = N * H;
    const unsigned blocks_td = (unsigned)(((size_t)T * (size_t)D + 255) / 256);

    qwen4_ple_grouped_rmsnorm_kernel<<<(unsigned)rows, 256, 0, stream>>>(key, w_norm_key, key, N, H, eps);
    qwen4_ple_grouped_rmsnorm_kernel<<<(unsigned)rows, 256, 0, stream>>>(hidden_R, w_norm_query, scratch_qn, N, H, eps);
    qwen4_ple_gate_kernel<<<(unsigned)rows, 256, 0, stream>>>(key, scratch_qn, scratch_gate, N);
    qwen4_ple_broadcast_mul_kernel<<<blocks_td, 256, 0, stream>>>(value, scratch_gate, scratch_gated, T, N, H);
    qwen4_ple_grouped_rmsnorm_kernel<<<(unsigned)rows, 256, 0, stream>>>(scratch_gated, w_norm_conv, scratch_qn, N, H, eps);
    qwen4_ple_conv_residual_kernel<<<blocks_td, 256, 0, stream>>>(history, scratch_qn, w_conv1d_f32, hidden_R, scratch_gated, T, D);
    qwen4_ple_history_advance_kernel<<<(unsigned)((D + 255) / 256), 256, 0, stream>>>(history, scratch_qn, T, D, snap_opt);
}

void launch_qwen4_gdn_ab_f32(
    const float* x, const float* w_alpha, const float* w_beta,
    const float* dt_bias, const float* ssm_a,
    float* gate_out, float* beta_out, int T, int n_embd, int h_v, cudaStream_t stream
) {
    if (T <= 0 || n_embd <= 0 || h_v <= 0) return;
    dim3 grid((unsigned)((2 * h_v + 7) / 8), (unsigned)T);
    dim3 block(32, 8);
    qwen4_gdn_ab_f32_kernel<<<grid, block, 0, stream>>>(
        x, w_alpha, w_beta, dt_bias, ssm_a, gate_out, beta_out, n_embd, h_v
    );
}

void launch_qwen4_gdn_gate_beta_postop(
    const float* alpha_raw, const float* beta_raw,
    const float* dt_bias, const float* ssm_a,
    float* gate_out, float* beta_out, int T, int h_v, cudaStream_t stream
) {
    if (T <= 0 || h_v <= 0) return;
    size_t total = (size_t)T * (size_t)h_v;
    qwen4_gdn_gate_beta_postop_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        alpha_raw, beta_raw, dt_bias, ssm_a, gate_out, beta_out, total, h_v
    );
}

void launch_qwen4_gdn_conv_l2(
    float* conv_hist, const float* qkv, const float* conv_w, float* h_out,
    int T, int channels, int qk_heads, float eps, float* hist_snap_opt, cudaStream_t stream
) {
    if (T <= 0 || channels <= 0) return;
    const unsigned heads = (unsigned)(channels / 128);
    if (T == 1) {
        qwen4_gdn_conv_l2_t1_kernel<<<heads, 128, 0, stream>>>(
            conv_hist, qkv, conv_w, h_out, qk_heads, eps, hist_snap_opt
        );
    } else {
        dim3 grid(heads, (unsigned)T);
        qwen4_gdn_conv_l2_batch_kernel<<<grid, 128, 0, stream>>>(
            conv_hist, qkv, conv_w, h_out, channels, qk_heads, eps
        );
        qwen4_gdn_conv_hist_advance_kernel<<<(unsigned)((channels + 255) / 256), 256, 0, stream>>>(
            conv_hist, qkv, T, channels, hist_snap_opt
        );
    }
}

void launch_qwen4_gdn_decode_step_norm_q8(
    float* state, const float* h_qkv, const float* gate, const float* beta,
    const float* z_gate, const float* ssm_norm_w, float eps,
    float* y_f32_opt, void* y_q8_opt, int h_k, int h_v, cudaStream_t stream
) {
    if (h_k <= 0 || h_v <= 0) return;
    dim3 block(128, 4);
    qwen4_gdn_steps_norm_q8_kernel<false><<<(unsigned)h_v, block, 0, stream>>>(
        state, h_qkv, gate, beta, z_gate, ssm_norm_w, eps,
        y_f32_opt, reinterpret_cast<block_q8_1*>(y_q8_opt), nullptr, 1, h_k, h_v
    );
}

void launch_qwen4_gdn_verify_steps_norm_q8(
    float* state, const float* h_qkv, const float* gate, const float* beta,
    const float* z_gate, const float* ssm_norm_w, float eps,
    float* y_f32_opt, void* y_q8_opt, float* state_snap_opt,
    int T, int h_k, int h_v, cudaStream_t stream
) {
    if (T <= 0 || h_k <= 0 || h_v <= 0) return;
    dim3 block(128, 4);
    qwen4_gdn_steps_norm_q8_kernel<true><<<(unsigned)h_v, block, 0, stream>>>(
        state, h_qkv, gate, beta, z_gate, ssm_norm_w, eps,
        y_f32_opt, reinterpret_cast<block_q8_1*>(y_q8_opt), state_snap_opt, T, h_k, h_v
    );
}

void launch_qwen4_gdn_steps_split_norm_q8(
    float* state, const float* h_qkv, const float* gate, const float* beta,
    const float* z_gate, const float* ssm_norm_w, float eps,
    float* y_ws_f32, float* y_f32_opt, void* y_q8_opt, float* state_before_opt,
    int T, int h_k, int h_v, cudaStream_t stream
) {
    if (T <= 0 || h_k <= 0 || h_v <= 0) return;
    dim3 grid_s((unsigned)h_v, 4u);
    dim3 block_s(32, 4);
    if (T == 1) {
        if (state_before_opt != nullptr) {
            qwen4_gdn_steps_split_kernel<false, true><<<grid_s, block_s, 0, stream>>>(
                state, h_qkv, gate, beta, y_ws_f32, state_before_opt, 1, h_k, h_v
            );
        } else {
            qwen4_gdn_steps_split_kernel<false, false><<<grid_s, block_s, 0, stream>>>(
                state, h_qkv, gate, beta, y_ws_f32, nullptr, 1, h_k, h_v
            );
        }
    } else {
        if (state_before_opt != nullptr) {
            qwen4_gdn_steps_split_kernel<true, true><<<grid_s, block_s, 0, stream>>>(
                state, h_qkv, gate, beta, y_ws_f32, state_before_opt, T, h_k, h_v
            );
        } else {
            qwen4_gdn_steps_split_kernel<true, false><<<grid_s, block_s, 0, stream>>>(
                state, h_qkv, gate, beta, y_ws_f32, nullptr, T, h_k, h_v
            );
        }
    }
    dim3 grid_n((unsigned)h_v, (unsigned)T);
    qwen4_gdn_norm_gate_q8_kernel<<<grid_n, 128, 0, stream>>>(
        y_ws_f32, z_gate, ssm_norm_w, eps, y_f32_opt, reinterpret_cast<block_q8_1*>(y_q8_opt), h_v
    );
}

void launch_qwen4_gdn_replay_steps(
    const float* state_before, float* state_out, const float* h_qkv,
    const float* gate, const float* beta,
    int n_steps, int h_k, int h_v, cudaStream_t stream
) {
    if (n_steps <= 0 || h_k <= 0 || h_v <= 0) return;
    dim3 grid((unsigned)h_v, 4u);
    dim3 block(32, 4);
    qwen4_gdn_replay_steps_kernel<<<grid, block, 0, stream>>>(
        state_before, state_out, h_qkv, gate, beta, n_steps, h_k, h_v
    );
}

size_t qwen4_gdn_tc_scratch_bytes(int h_k, int h_v) {
    const size_t ti_b = align_up_256((size_t)GTC_NB * (size_t)h_v * 2 * GTC_GCH * GTC_GCH * 2);
    const size_t te_b = align_up_256((size_t)GTC_NB * (size_t)h_v * 4);
    const size_t sc_b = align_up_256((size_t)GTC_NB * (size_t)h_v * GTC_GCH * 16);
    const size_t kq_b = align_up_256((size_t)GTC_NB * (size_t)h_k * 4 * GTC_GCH * GTC_S * 2);
    const size_t pi_b = align_up_256((size_t)GTC_NB * (size_t)h_v * 2 * GTC_GCH * GTC_GCH * 2);
    return ti_b + te_b + sc_b + kq_b + pi_b;
}

void launch_qwen4_gdn_tc_prefill(
    float* state, const float* h_qkv, const float* gate, const float* beta,
    const float* z_gate, const float* ssm_norm_w, float eps,
    float* y_f32_opt, void* y_f16_opt, void* tc_scratch,
    int T, int h_k, int h_v, cudaStream_t stream
) {
    if (T <= 0 || h_k <= 0 || h_v <= 0) return;
    const Qwen4GtcScratch sc = make_gtc_scratch(tc_scratch, h_k, h_v);
    const size_t prep_smem = std::max(
        (size_t)(4 * GTC_GCH * GTC_KP * 2 + 2 * GTC_GCH * GTC_GTP * 4),
        (size_t)(h_v / h_k) * GTC_GCH * GTC_GTP * 4
    );
    uint16_t* y16 = reinterpret_cast<uint16_t*>(y_f16_opt);
    const int mask = ((y_f32_opt != nullptr) ? 1 : 0) | ((y16 != nullptr) ? 2 : 0);

    for (int t0 = 0; t0 < T; t0 += GTC_GSB) {
        const int nt = min(T - t0, GTC_GSB);
        const int nch = (nt + GTC_GCH - 1) / GTC_GCH;
        qwen4_gdn_tc_prep_kernel<<<dim3((unsigned)nch, (unsigned)h_k), 256, prep_smem, stream>>>(
            h_qkv, gate, beta, sc.ti, sc.te, sc.sc, sc.kq, sc.pi, t0, (int64_t)T, h_k, h_v
        );
        if (mask == 3) {
            launch_qwen4_gdn_tc_scan_configured<3>(
                state, h_qkv, sc, z_gate, ssm_norm_w, eps, y_f32_opt, y16, t0, nch, (int64_t)T, h_k, h_v, stream
            );
        } else if (mask == 2) {
            launch_qwen4_gdn_tc_scan_configured<2>(
                state, h_qkv, sc, z_gate, ssm_norm_w, eps, y_f32_opt, y16, t0, nch, (int64_t)T, h_k, h_v, stream
            );
        } else if (mask == 1) {
            launch_qwen4_gdn_tc_scan_configured<1>(
                state, h_qkv, sc, z_gate, ssm_norm_w, eps, y_f32_opt, y16, t0, nch, (int64_t)T, h_k, h_v, stream
            );
        } else {
            launch_qwen4_gdn_tc_scan_configured<0>(
                state, h_qkv, sc, z_gate, ssm_norm_w, eps, y_f32_opt, y16, t0, nch, (int64_t)T, h_k, h_v, stream
            );
        }
    }
}

void launch_qwen4_qsa_norm_imrope(
    const float* src, int in_stride, const float* gamma, float* out,
    int T, int n_heads, int head_dim, int n_rot, float eps, float freq_base,
    const int32_t* pos_dev, const int32_t* mrope_pos3_opt, cudaStream_t stream
) {
    if (T <= 0 || n_heads <= 0 || head_dim <= 0) return;
    dim3 grid((unsigned)n_heads, (unsigned)T);
    qwen4_qsa_norm_imrope_kernel<<<grid, 256, 0, stream>>>(
        src, in_stride, gamma, out, n_heads, head_dim, n_rot, eps, freq_base, pos_dev, mrope_pos3_opt
    );
}

void launch_qwen4_idx_append_steps(
    const float* raw_k, const int32_t* pos_dev, int n_steps, const float* gamma,
    float eps, float freq_base, const int32_t* mrope_pos3_opt,
    float* tail, float* dead, float* pooled, int max_cells, cudaStream_t stream
) {
    if (n_steps <= 0 || max_cells <= 0) return;
    qwen4_idx_append_steps_kernel<<<1, 128, 0, stream>>>(
        raw_k, pos_dev, n_steps, gamma, eps, freq_base, mrope_pos3_opt, tail, dead, pooled, max_cells
    );
}

void launch_qwen4_idx_append_batch(
    const float* raw_k, int n_tokens, int p0, const float* gamma,
    float eps, float freq_base, const int32_t* mrope_pos3_opt,
    float* tail, float* dead, float* pooled, int max_cells, cudaStream_t stream
) {
    if (n_tokens <= 0 || p0 < 0 || max_cells <= 0) return;
    const int64_t last_pos = min((int64_t)p0 + n_tokens - 1, (int64_t)max_cells - 1);
    const int64_t valid_n = last_pos - p0 + 1;
    if (valid_n <= 0) return;

    if (p0 == 0) {
        qwen4_idx_append_first_kernel<<<1, 128, 0, stream>>>(raw_k, gamma, eps, freq_base, dead, pooled);
    }

    const int64_t first_blk = (p0 <= 3) ? 0 : (int64_t)(p0 / 4);
    const int64_t last_blk = (last_pos < 3) ? -1 : ((last_pos - 3) / 4);
    if (last_blk >= first_blk) {
        qwen4_idx_append_blocks_kernel<<<(unsigned)(last_blk - first_blk + 1), 128, 0, stream>>>(
            raw_k, p0, first_blk, last_blk, gamma, eps, freq_base, mrope_pos3_opt, tail, dead, pooled
        );
    }
    qwen4_idx_append_tail_kernel<<<3, 128, 0, stream>>>(raw_k, valid_n, p0, tail);
}

void launch_qwen4_idx_score_select(
    const float* pooled, const float* dead, const float* q_idx, const int32_t* pos_dev,
    int T, int max_blocks, int top_k_cap, float* scores_ws, int32_t* ids_out, int32_t* width_out,
    cudaStream_t stream
) {
    if (T <= 0 || max_blocks <= 0 || top_k_cap <= 0) return;
    if (T >= 16) {
        constexpr size_t tc_smem = (16 * (4 * 128 + 4) + 32 * (128 + 4)) * sizeof(float); // 49,920 B
        static int s_idx_tc_configured[64] = {};
        int dev = 0;
        if (cudaGetDevice(&dev) == cudaSuccess && dev >= 0 && dev < 64 && !s_idx_tc_configured[dev]) {
            cudaFuncSetAttribute(
                qwen4_idx_score_tc_kernel,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                (int)tc_smem
            );
            s_idx_tc_configured[dev] = 1;
        }
        dim3 grid((unsigned)((max_blocks + 127) / 128), (unsigned)((T + 15) / 16));
        qwen4_idx_score_tc_kernel<<<grid, 128, tc_smem, stream>>>(
            pooled, q_idx, pos_dev, T, max_blocks, scores_ws
        );
        qwen4_idx_score_tail_kernel<<<(unsigned)T, 32, 0, stream>>>(
            dead, q_idx, pos_dev, max_blocks, scores_ws
        );
    } else if (T <= 8) {
        dim3 grid((unsigned)((max_blocks + 7) / 8), 1);
        qwen4_idx_score_warp_multi_kernel<<<grid, 256, 0, stream>>>(
            pooled, dead, q_idx, pos_dev, T, max_blocks, scores_ws
        );
    } else {
        dim3 grid((unsigned)((max_blocks + 7) / 8), (unsigned)T);
        qwen4_idx_score_warp_kernel<<<grid, 256, 0, stream>>>(
            pooled, dead, q_idx, pos_dev, max_blocks, scores_ws
        );
    }

    qwen4_idx_radix_topk_kernel<<<(unsigned)T, 256, 0, stream>>>(
        scores_ws, pos_dev, max_blocks, top_k_cap, ids_out, width_out
    );
}

void launch_qwen4_kv_append_q8(
    const float* k, const float* v, const int32_t* pos_dev,
    int T, int n_kv_heads, int head_dim,
    int8_t* k_q, int8_t* v_q, uint16_t* k_sc, uint16_t* v_sc, cudaStream_t stream
) {
    if (T <= 0 || n_kv_heads <= 0 || head_dim <= 0) return;
    dim3 grid((unsigned)n_kv_heads, (unsigned)(head_dim / 64), (unsigned)(2 * T));
    qwen4_kv_append_q8_kernel<<<grid, 64, 0, stream>>>(
        k, v, pos_dev, n_kv_heads, head_dim, k_q, v_q, k_sc, v_sc
    );
}

size_t qwen4_qsa_decode_ws_floats(int T, int n_kv_heads, int max_n_ids) {
    const int max_chunks = (T == 1) ? 164 : QWEN4_QSA_MAX_CHUNKS;
    int n_chunks = (max_n_ids <= 0) ? 1 : ((max_n_ids + 63) / 64);
    if (n_chunks > max_chunks) n_chunks = max_chunks;
    const size_t stride_m = (size_t)n_kv_heads * (size_t)n_chunks * 12;
    const size_t stride_acc = stride_m * 256;
    return (size_t)T * (stride_acc + 2 * stride_m);
}

void launch_qwen4_qsa_decode_attn_q8(
    const float* q, const float* q_full_gate_opt,
    const int8_t* k_q, const int8_t* v_q, const uint16_t* k_sc, const uint16_t* v_sc,
    const int32_t* ids_opt, const int32_t* n_ids_dev, int max_n_ids, int id_cap,
    int T, int n_kv_heads, float scale,
    float* ws_buf, float* out_f32_opt, void* out_q8_opt, cudaStream_t stream
) {
    if (T <= 0 || n_kv_heads <= 0) return;
    const int max_chunks = (T == 1) ? 164 : QWEN4_QSA_MAX_CHUNKS;
    int n_chunks = (max_n_ids <= 0) ? 1 : ((max_n_ids + 63) / 64);
    if (n_chunks > max_chunks) n_chunks = max_chunks;
    const size_t stride_m = (size_t)n_kv_heads * (size_t)n_chunks * 12;
    const size_t stride_acc = stride_m * 256;
    float* part_acc = ws_buf;
    float* part_m   = ws_buf + (size_t)T * stride_acc;
    float* part_l   = part_m + (size_t)T * stride_m;

    dim3 grid_c((unsigned)n_chunks, (unsigned)n_kv_heads, (unsigned)T);
    qwen4_qsa_decode_chunk_kernel<<<grid_c, 256, 0, stream>>>(
        q, k_q, v_q, k_sc, v_sc, ids_opt, n_ids_dev, id_cap, n_chunks, n_kv_heads, scale,
        part_acc, part_m, part_l, stride_acc, stride_m
    );

    dim3 grid_m((unsigned)(n_kv_heads * 12), (unsigned)T);
    qwen4_qsa_decode_merge_kernel<<<grid_m, 256, 0, stream>>>(
        part_acc, part_m, part_l, n_chunks, n_ids_dev, q_full_gate_opt,
        out_f32_opt, reinterpret_cast<block_q8_1*>(out_q8_opt), stride_acc, stride_m
    );
}

void launch_qwen4_qsa_prefill_attn_q8(
    const float* q, const float* q_full_gate_opt,
    const int8_t* k_q, const int8_t* v_q, const uint16_t* k_sc, const uint16_t* v_sc,
    const int32_t* ids_opt, const int32_t* n_ids_dev, int id_cap,
    int T, int n_kv_heads, float scale, float* out_f32, cudaStream_t stream
) {
    if (T <= 0 || n_kv_heads <= 0) return;
    const float scale_log2 = scale * 1.4426950408889634f;
    dim3 grid((unsigned)T, (unsigned)n_kv_heads);
    qwen4_qsa_prefill_attn_q8_kernel<<<grid, 128, 0, stream>>>(
        q, q_full_gate_opt, k_q, v_q, k_sc, v_sc, ids_opt, n_ids_dev, id_cap,
        n_kv_heads, scale_log2, out_f32
    );
}

void launch_qwen4_router_top10(
    const float* logits, int T, int n_exp, int topk,
    int32_t* ids_out, float* weights_out, cudaStream_t stream
) {
    if (T <= 0 || n_exp <= 0 || topk <= 0) return;
    dim3 block(32, 8);
    qwen4_router_top10_kernel<<<(unsigned)((T + 7) / 8), block, 0, stream>>>(
        logits, T, n_exp, topk, ids_out, weights_out
    );
}

void launch_qwen4_shared_gate_dot(
    const float* x, const void* w_sh_gate, int w_type,
    int T, int n_embd, float* out_logits, cudaStream_t stream
) {
    if (T <= 0 || n_embd <= 0) return;
    qwen4_shared_gate_dot_kernel<<<(unsigned)T, 256, 0, stream>>>(
        x, w_sh_gate, w_type, n_embd, out_logits
    );
}

void launch_qwen4_vis_rope2d(
    float* qkv, const int32_t* pos_yx, int n_patches, int heads, int hd, float base, cudaStream_t stream
) {
    if (n_patches <= 0 || heads <= 0 || hd <= 0) return;
    dim3 grid((unsigned)heads, (unsigned)n_patches);
    qwen4_vis_rope2d_kernel<<<grid, 64, 0, stream>>>(qkv, pos_yx, heads, hd, base);
}

void launch_qwen4_vis_merge_2x2_f16(
    const float* x_ln, int n_merged, int D, uint16_t* out_f16, cudaStream_t stream
) {
    if (n_merged <= 0 || D <= 0) return;
    size_t total = (size_t)n_merged * 4 * (size_t)D;
    qwen4_vis_merge_2x2_f16_kernel<<<(unsigned)((total + 255) / 256), 256, 0, stream>>>(
        x_ln, total, out_f16
    );
}

void launch_qwen4_fill_dense_widths(
    const int32_t* pos_dev, int32_t* width_out, int T, cudaStream_t stream
) {
    if (T <= 0) return;
    qwen4_fill_dense_widths_kernel<<<(unsigned)((T + 127) / 128), 128, 0, stream>>>(
        pos_dev, width_out, T
    );
}

void launch_qwen4_hc_up_mix_q8(
    const void* w_up_q8_0, const void* lo_q8, const float* xn,
    int T, int hc, int E, int lr,
    float* mixed_out, void* mixed_q8_opt, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0 || lr <= 0) return;
    const auto* w = reinterpret_cast<const block_q8_0*>(w_up_q8_0);
    const auto* lo = reinterpret_cast<const block_q8_1*>(lo_q8);
    auto* mq8 = reinterpret_cast<block_q8_1*>(mixed_q8_opt);
    if (lr == 320 && hc <= 8 && T <= 5) {
        dim3 grid((unsigned)((E + 31) / 32), 1);
        dim3 blk(32, 16);
        if (T == 1) {
            qwen4_hc_up_mix_q8_lr320_kernel<1, true><<<grid, blk, 0, stream>>>(
                w, lo, xn, T, hc, E, mixed_out, mq8
            );
        } else if (T <= 4) {
            qwen4_hc_up_mix_q8_lr320_kernel<4, false><<<grid, blk, 0, stream>>>(
                w, lo, xn, T, hc, E, mixed_out, mq8
            );
        } else {
            qwen4_hc_up_mix_q8_lr320_kernel<5, true><<<grid, blk, 0, stream>>>(
                w, lo, xn, T, hc, E, mixed_out, mq8
            );
        }
        return;
    }
    if (T == 1) {
        dim3 grid((unsigned)((E + 31) / 32), 1);
        qwen4_hc_up_mix_q8_kernel<1><<<grid, 256, 0, stream>>>(
            w, lo, xn, T, hc, E, lr, mixed_out, mq8
        );
    } else if (T <= 4) {
        dim3 grid((unsigned)((E + 31) / 32), 1);
        qwen4_hc_up_mix_q8_kernel<4><<<grid, 256, 0, stream>>>(
            w, lo, xn, T, hc, E, lr, mixed_out, mq8
        );
    } else if (T <= 6) {
        dim3 grid((unsigned)((E + 31) / 32), 1);
        qwen4_hc_up_mix_q8_kernel<6><<<grid, 256, 0, stream>>>(
            w, lo, xn, T, hc, E, lr, mixed_out, mq8
        );
    } else {
        dim3 grid((unsigned)((E + 31) / 32), (unsigned)((T + 7) / 8));
        qwen4_hc_up_mix_q8_kernel<8><<<grid, 256, 0, stream>>>(
            w, lo, xn, T, hc, E, lr, mixed_out, mq8
        );
    }
}

void launch_qwen4_router_doorbell_plan(
    const float* logits, int T, int n_exp, int topk,
    const uint64_t* vram_ptrs, const uint64_t* ram_ptrs,
    int gpu_pcie_div, int gpu_pcie_max,
    const float* mixed_f32, int n_embd, int n_ff, int layer_idx,
    int32_t gu_type, int32_t d_type,
    int32_t* ids_out, float* weights_out, uint64_t* plan_ptrs_out,
    void* doorbell_mapped, uint64_t* seq_state_dev, uint64_t* expected_seq_out,
    cudaStream_t stream
) {
    if (T <= 0 || n_exp <= 0 || topk <= 0) return;
    qwen4_router_doorbell_plan_kernel<<<1, 256, 0, stream>>>(
        logits, T, n_exp, topk, vram_ptrs, ram_ptrs,
        gpu_pcie_div, gpu_pcie_max, mixed_f32, n_embd, n_ff, layer_idx,
        gu_type, d_type, ids_out, weights_out, plan_ptrs_out,
        reinterpret_cast<Qwen4Doorbell*>(doorbell_mapped),
        seq_state_dev, expected_seq_out
    );
}

void launch_qwen4_doorbell_wait(
    const void* doorbell_mapped, const uint64_t* expected_seq_ptr, cudaStream_t stream
) {
    qwen4_doorbell_wait_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<const Qwen4Doorbell*>(doorbell_mapped), expected_seq_ptr
    );
}

void launch_qwen4_moe_reduce_hc_write(
    const void* doorbell_mapped, const uint64_t* expected_seq_ptr,
    float* R, const float* block_out_opt, const float* pair_out_opt,
    const uint16_t* gpu_pair_mask_opt, int topk,
    const float* sh_out_opt, const float* sh_gate_logit_opt,
    const float* inject, int inject_stride,
    int T, int hc, int E, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    dim3 grid((unsigned)((E + 127) / 128), (unsigned)T);
    qwen4_moe_reduce_hc_write_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const Qwen4Doorbell*>(doorbell_mapped),
        expected_seq_ptr, R, block_out_opt, pair_out_opt, gpu_pair_mask_opt, topk,
        sh_out_opt, sh_gate_logit_opt, inject, inject_stride, hc, E
    );
}

void launch_qwen4_doorbell_wait_hc_write(
    const void* doorbell_mapped, const uint64_t* expected_seq_ptr,
    float* R, const float* block_out, const float* sh_out_opt,
    const float* sh_gate_logit_opt, const float* inject,
    int T, int hc, int E, cudaStream_t stream
) {
    if (T <= 0 || hc <= 0 || E <= 0) return;
    qwen4_doorbell_wait_kernel<<<1, 32, 0, stream>>>(
        reinterpret_cast<const Qwen4Doorbell*>(doorbell_mapped), expected_seq_ptr
    );
    dim3 grid((unsigned)((E + 127) / 128), (unsigned)T);
    qwen4_moe_reduce_hc_write_kernel<<<grid, 128, 0, stream>>>(
        reinterpret_cast<const Qwen4Doorbell*>(doorbell_mapped),
        expected_seq_ptr, R, block_out, nullptr, nullptr, 0,
        sh_out_opt, sh_gate_logit_opt, inject, 1, hc, E
    );
}

void launch_qwen4_argmax_prob_indexed(
    const float* logits,
    int n_vocab,
    const int32_t* sub_to_id_opt,
    float* partial_max,
    int32_t* partial_idx,
    float* partial_sum,
    int32_t* out_id,
    float* out_prob,
    cudaStream_t stream
) {
    if (n_vocab <= 0) return;
    int n_blocks = (n_vocab + 2047) / 2048;
    if (n_blocks < 1) n_blocks = 1;
    if (n_blocks > 128) n_blocks = 128;
    qwen4_argmax_prob_stage1_kernel<<<(unsigned)n_blocks, 256, 0, stream>>>(
        logits, n_vocab, partial_max, partial_idx, partial_sum
    );
    qwen4_argmax_prob_stage2_kernel<<<1, 128, 0, stream>>>(
        partial_max, partial_idx, partial_sum, n_blocks, sub_to_id_opt, out_id, out_prob
    );
}

void launch_qwen4_argmax_prob(
    const float* logits,
    int n_vocab,
    float* partial_max,
    int32_t* partial_idx,
    float* partial_sum,
    int32_t* out_id,
    float* out_prob,
    cudaStream_t stream
) {
    launch_qwen4_argmax_prob_indexed(
        logits, n_vocab, nullptr, partial_max, partial_idx, partial_sum, out_id, out_prob, stream
    );
}

void launch_qwen4_moe_fetch_unique(
    const uint64_t* fetch_src_ptrs,
    const uint64_t* fetch_dst_ptrs,
    size_t blob_bytes,
    int fetch_cap,
    cudaStream_t stream
) {
    if (fetch_cap <= 0 || blob_bytes == 0) return;
    qwen4_moe_fetch_unique_kernel<<<8u, 256, 0, stream>>>(
        fetch_src_ptrs, fetch_dst_ptrs, blob_bytes, fetch_cap
    );
}

void launch_qwen4_router_doorbell_plan_dedup(
    const float* logits, int T, int n_exp, int topk,
    const uint64_t* vram_ptrs, const uint64_t* ram_ptrs,
    int gpu_pcie_div, int fetch_cap, uint64_t dma_stage_base, size_t slot_stride,
    const float* mixed_f32, int n_embd, int n_ff, int layer_idx,
    int32_t gu_type, int32_t d_type,
    int32_t* ids_out, float* weights_out,
    uint64_t* res_ptrs_out, int32_t* res_off_out, int32_t* res_nt_out,
    int32_t* res_rows_out, int32_t* res_pair_out, float* res_wts_out,
    uint64_t* fetch_src_out, uint64_t* fetch_dst_out, int32_t* fetch_off_out,
    int32_t* fetch_nt_out, int32_t* fetch_rows_out, int32_t* fetch_pair_out, float* fetch_wts_out,
    uint16_t* gpu_pair_mask_out,
    void* doorbell_mapped, uint64_t* seq_state_dev, uint64_t* expected_seq_out,
    cudaStream_t stream
) {
    if (T <= 0 || n_exp <= 0 || topk <= 0) return;
    qwen4_router_doorbell_plan_dedup_kernel<<<1, 256, 0, stream>>>(
        logits, T, n_exp, topk, vram_ptrs, ram_ptrs,
        gpu_pcie_div, fetch_cap, dma_stage_base, slot_stride,
        mixed_f32, n_embd, n_ff, layer_idx, gu_type, d_type,
        ids_out, weights_out,
        res_ptrs_out, res_off_out, res_nt_out, res_rows_out, res_pair_out, res_wts_out,
        fetch_src_out, fetch_dst_out, fetch_off_out, fetch_nt_out, fetch_rows_out, fetch_pair_out, fetch_wts_out,
        gpu_pair_mask_out,
        reinterpret_cast<Qwen4Doorbell*>(doorbell_mapped),
        seq_state_dev, expected_seq_out
    );
}

void launch_qwen4_reduce_pairs_f32(
    const float* pair_out, float* block_out, int T, int topk, int n_embd, cudaStream_t stream
) {
    if (T <= 0 || topk <= 0 || n_embd <= 0) return;
    dim3 grid((unsigned)((n_embd + 255) / 256), (unsigned)T);
    qwen4_reduce_pairs_f32_kernel<<<grid, 256, 0, stream>>>(pair_out, block_out, T, topk, n_embd);
}

void launch_qwen4_patch_u64(
    const uint64_t* mapped_pairs, int n, cudaStream_t stream
) {
    if (n <= 0) return;
    qwen4_patch_u64_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(mapped_pairs, n);
}

void launch_qwen4_copy_u32_sm(
    void* dst, const void* src, int n_u32, int system_fence, cudaStream_t stream
) {
    if (n_u32 <= 0) return;
    const bool aligned16 = (((reinterpret_cast<uintptr_t>(dst) | reinterpret_cast<uintptr_t>(src)) & 15u) == 0u);
    int n_work = (aligned16 && n_u32 >= 4) ? ((n_u32 + 3) / 4) : n_u32;
    unsigned blocks = (unsigned)((n_work + 255) / 256);
    if (blocks < 1u) blocks = 1u;
    if (blocks > 32u) blocks = 32u;
    qwen4_copy_u32_sm_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<uint32_t*>(dst),
        reinterpret_cast<const uint32_t*>(src),
        n_u32,
        system_fence
    );
}

} // extern "C"
