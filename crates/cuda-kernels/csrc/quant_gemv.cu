#include "common/sm86_common.cuh"
#include <type_traits>

// Quantize [n_rows, K] FP32 activations -> [n_rows, K/32] block_q8_1.
// One warp handles one 32-element block.
__global__ void quantize_f32_to_q8_1_kernel(
    const float* __restrict__ x,
    block_q8_1* __restrict__ y,
    int K
) {
    int row = blockIdx.y;
    int blk = blockIdx.x * blockDim.y + threadIdx.y;
    int nb = K / 32;
    if (blk >= nb) return;

    int lane = threadIdx.x; // 0..31
    float val = x[(size_t)row * K + blk * 32 + lane];
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

    block_q8_1* out = y + (size_t)row * nb + blk;
    out->qs[lane] = (int8_t)q;
    if (lane == 0) {
        uint16_t d_bits = f32_to_fp16_bits(d);
        uint16_t s_bits = f32_to_fp16_bits(d * (float)sum_q);
        out->d = d_bits;
        out->s = s_bits;
    }
}

// Fused SWA / Global attention projection GEMV for M=1:
// Concatenates up to 5 row-Ranges (Q, K, V, R, Gate) sharing the same quant type (Q8_0 or IQ4_NL/NVFP4),
// reads from X_q_q8 / X_k_q8 / X_v_q8, and optionally fuses the static 4-tap K/V short convolution in-place!
__device__ __forceinline__ float eval_static_short_conv_m1(
    int ch,
    int conv_offset,
    int n_conv,
    float xt,
    const float* __restrict__ static_w,
    float* __restrict__ history
) {
    float* hist_ch = history + (size_t)(conv_offset + ch) * 3;
    float h0 = hist_ch[0];
    float h1 = hist_ch[1];
    float h2 = hist_ch[2];
    const float* sw = static_w + (size_t)ch * 4;
    float conv_val = h0 * sw[0] + h1 * sw[1] + h2 * sw[2] + xt * sw[3];
    float inv_scale = rsqrtf((float)(n_conv + 1));
    hist_ch[0] = h1;
    hist_ch[1] = h2;
    hist_ch[2] = xt;
    return (xt + conv_val) * inv_scale;
}

__device__ __forceinline__ void apply_static_short_conv_m1(
    int ch,
    int n_channels,
    int conv_offset,
    int n_conv,
    float xt,
    const float* __restrict__ static_w,
    float* __restrict__ history,
    float* __restrict__ out
) {
    (void)n_channels;
    out[ch] = eval_static_short_conv_m1(ch, conv_offset, n_conv, xt, static_w, history);
}

// Planar Q8_0 (int8 `qs` [N, K] + uint16_t `d` [N, K/32]) against float32 `x` [K] for M=1!
// 1 warp per row; each lane reads 4 int8 values (`uint32_t`) and 4 floats (`float4`) coalesced per iteration (128 elements/iter).
__global__ void mv_q8_planar_m1_kernel(
    const int8_t* __restrict__ q0, const uint16_t* __restrict__ d0, int n0, const float* __restrict__ x0, float* __restrict__ y0,
    const int8_t* __restrict__ q1, const uint16_t* __restrict__ d1, int n1, const float* __restrict__ x1, float* __restrict__ y1,
    const float* __restrict__ conv_w1, int conv_off1,
    const int8_t* __restrict__ q2, const uint16_t* __restrict__ d2, int n2, const float* __restrict__ x2, float* __restrict__ y2,
    const float* __restrict__ conv_w2, int conv_off2,
    const int8_t* __restrict__ q3, const uint16_t* __restrict__ d3, int n3, const float* __restrict__ x3, float* __restrict__ y3,
    const int8_t* __restrict__ q4, const uint16_t* __restrict__ d4, int n4, const float* __restrict__ x4, float* __restrict__ y4,
    const float* __restrict__ add_res0,
    float* __restrict__ conv_history,
    int n_conv,
    int K
) {
    int total_rows = n0 + n1 + n2 + n3 + n4;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= total_rows) return;
    int lane = threadIdx.x;

    const int8_t* q_row = nullptr;
    const uint16_t* d_row = nullptr;
    const float* x_vec = nullptr;
    float* dst = nullptr;
    int local_r = 0;
    int seg = 0;
    int nb = K >> 5;

    if (row < n0) {
        seg = 0; local_r = row;
        q_row = q0 + (size_t)local_r * K; d_row = d0 + (size_t)local_r * nb; x_vec = x0; dst = y0;
    } else if (row < n0 + n1) {
        seg = 1; local_r = row - n0;
        q_row = q1 + (size_t)local_r * K; d_row = d1 + (size_t)local_r * nb; x_vec = x1; dst = y1;
    } else if (row < n0 + n1 + n2) {
        seg = 2; local_r = row - (n0 + n1);
        q_row = q2 + (size_t)local_r * K; d_row = d2 + (size_t)local_r * nb; x_vec = x2; dst = y2;
    } else if (row < n0 + n1 + n2 + n3) {
        seg = 3; local_r = row - (n0 + n1 + n2);
        q_row = q3 + (size_t)local_r * K; d_row = d3 + (size_t)local_r * nb; x_vec = x3; dst = y3;
    } else {
        seg = 4; local_r = row - (n0 + n1 + n2 + n3);
        q_row = q4 + (size_t)local_r * K; d_row = d4 + (size_t)local_r * nb; x_vec = x4; dst = y4;
    }

    const uint32_t* qr = reinterpret_cast<const uint32_t*>(q_row);
    const float4* x_vec4 = reinterpret_cast<const float4*>(x_vec);
    float acc = 0.0f;
    int n_it = K >> 7; // K / 128
    for (int it = 0; it < n_it; ++it) {
        uint32_t w = __ldg(qr + it * 32 + lane);
        float4 xv = __ldg(x_vec4 + it * 32 + lane);
        float dv = fp16_bits_to_f32(__ldg(d_row + it * 4 + (lane >> 3)));
        float f0 = (float)(int8_t)(w & 0xFFu);
        float f1 = (float)(int8_t)((w >> 8) & 0xFFu);
        float f2 = (float)(int8_t)((w >> 16) & 0xFFu);
        float f3 = (float)(int8_t)(w >> 24);
        float p = __fmul_rn(f1, xv.y);
        p = __fmaf_rn(f0, xv.x, p);
        p = __fmaf_rn(f2, xv.z, p);
        p = __fmaf_rn(f3, xv.w, p);
        acc = __fmaf_rn(dv, p, acc);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        acc = __fadd_rn(acc, __shfl_xor_sync(0xffffffffu, acc, o));
    }

    if (lane == 0) {
        if (seg == 0 && conv_w1 != nullptr && n1 == 0) {
            // Single matrix with fused conv + optional add_res0 (used for wo!)
            float y = eval_static_short_conv_m1(local_r, conv_off1, n_conv, acc, conv_w1, conv_history);
            if (add_res0 != nullptr) y += add_res0[local_r];
            dst[local_r] = y;
        } else if (seg == 1 && conv_w1 != nullptr) {
            apply_static_short_conv_m1(local_r, n1, conv_off1, n_conv, acc, conv_w1, conv_history, dst);
        } else if (seg == 2 && conv_w2 != nullptr) {
            apply_static_short_conv_m1(local_r, n2, conv_off2, n_conv, acc, conv_w2, conv_history, dst);
        } else {
            dst[local_r] = acc;
        }
    }
}

__global__ void fused_qkvrg_m1_kernel(
    int32_t w_type,
    const uint8_t* __restrict__ w0, int n0, const block_q8_1* __restrict__ x0, float* __restrict__ y0,
    const uint8_t* __restrict__ w1, int n1, const block_q8_1* __restrict__ x1, float* __restrict__ y1,
    const float* __restrict__ conv_w1, int conv_off1,
    const uint8_t* __restrict__ w2, int n2, const block_q8_1* __restrict__ x2, float* __restrict__ y2,
    const float* __restrict__ conv_w2, int conv_off2,
    const uint8_t* __restrict__ w3, int n3, const block_q8_1* __restrict__ x3, float* __restrict__ y3,
    const uint8_t* __restrict__ w4, int n4, const block_q8_1* __restrict__ x4, float* __restrict__ y4,
    float* __restrict__ conv_history,
    int n_conv,
    int K,
    size_t row_bytes
) {
    int total_rows = n0 + n1 + n2 + n3 + n4;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= total_rows) return;
    int lane = threadIdx.x;

    const uint8_t* w_row = nullptr;
    const block_q8_1* x_q8 = nullptr;
    float* dst = nullptr;
    int local_r = 0;
    int seg = 0;

    if (row < n0) {
        seg = 0; local_r = row;
        w_row = w0 + (size_t)local_r * row_bytes; x_q8 = x0; dst = y0;
    } else if (row < n0 + n1) {
        seg = 1; local_r = row - n0;
        w_row = w1 + (size_t)local_r * row_bytes; x_q8 = x1; dst = y1;
    } else if (row < n0 + n1 + n2) {
        seg = 2; local_r = row - (n0 + n1);
        w_row = w2 + (size_t)local_r * row_bytes; x_q8 = x2; dst = y2;
    } else if (row < n0 + n1 + n2 + n3) {
        seg = 3; local_r = row - (n0 + n1 + n2);
        w_row = w3 + (size_t)local_r * row_bytes; x_q8 = x3; dst = y3;
    } else {
        seg = 4; local_r = row - (n0 + n1 + n2 + n3);
        w_row = w4 + (size_t)local_r * row_bytes; x_q8 = x4; dst = y4;
    }

    float sum = 0.0f;
    if (w_type == GGML_TYPE_Q8_0) {
        sum = dot_row_q8_0_sub(reinterpret_cast<const block_q8_0*>(w_row), x_q8, K, lane);
    } else if (w_type == GGML_TYPE_NVFP4) {
        sum = dot_row_nvfp4_sub(reinterpret_cast<const block_nvfp4*>(w_row), x_q8, K, lane);
    } else if (w_type == GGML_TYPE_IQ4_NL) {
        sum = dot_row_iq4_nl_sub(reinterpret_cast<const block_iq4_nl*>(w_row), x_q8, K, lane);
    } else if (w_type == GGML_TYPE_Q4_0) {
        sum = dot_row_q4_0_sub(reinterpret_cast<const block_q4_0*>(w_row), x_q8, K, lane);
    } else if (w_type == GGML_TYPE_Q6_K) {
        int nb = K / 256;
        const block_q6_K* b = reinterpret_cast<const block_q6_K*>(w_row);
        float a = 0.0f;
        for (int i = lane; i < nb; i += 32) a += dot_q6_K_q8_1(b + i, x_q8 + i * 8);
        sum = warp_reduce_sum(a);
    }

    if (lane == 0) {
        if (seg == 1 && conv_w1 != nullptr) {
            apply_static_short_conv_m1(local_r, n1, conv_off1, n_conv, sum, conv_w1, conv_history, dst);
        } else if (seg == 2 && conv_w2 != nullptr) {
            apply_static_short_conv_m1(local_r, n2, conv_off2, n_conv, sum, conv_w2, conv_history, dst);
        } else {
            dst[local_r] = sum;
        }
    }
}

// Generic warp-per-row GEMV / multi-token MMVQ (M = 1..8) for GGML quantized weights.
// W: [N, K], X: [M, K] in block_q8_1, Y: [M, N] in FP32.
template <typename BlockT, int ELEMS_PER_BLOCK, float (*DotFn)(const BlockT*, const block_q8_1*), int MAX_M>
__global__ void mmvq_quant_kernel(
    const uint8_t* __restrict__ W,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y,
    const float* __restrict__ conv_w,
    const float* __restrict__ add_res,
    float* __restrict__ conv_history,
    int conv_offset,
    int n_conv,
    int N,
    int K,
    int M,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    constexpr int MAX_STAGED_NB = 128; // 128 * 36B = 4608B shared, covers K <= 4096 (K=2304, 3072) -> 20 blocks/SM!
    __shared__ __align__(16) uint32_t s_x_words[MAX_M == 1 ? (MAX_STAGED_NB * 9) : 1];

    int row = blockIdx.x * blockDim.y + threadIdx.y;
    int lane = threadIdx.x;
    int nb_w = K / ELEMS_PER_BLOCK;
    int nb_x = K / 32;
    int q8_per_w = ELEMS_PER_BLOCK / 32;

    const block_q8_1* x_staged = X_q8;
    if constexpr (MAX_M == 1) {
        if (nb_x <= MAX_STAGED_NB) {
            int total_words = nb_x * 9;
            const uint32_t* gx = reinterpret_cast<const uint32_t*>(X_q8);
            int tid = threadIdx.y * 32 + lane;
            int nthreads = blockDim.y * 32;
            for (int i = tid; i < total_words; i += nthreads) {
                s_x_words[i] = gx[i];
            }
            __syncthreads();
            x_staged = reinterpret_cast<const block_q8_1*>(s_x_words);
        }
    }

    if (row >= N) return;

    const BlockT* w_row = reinterpret_cast<const BlockT*>(W + (size_t)row * nb_w * sizeof(BlockT));

    if constexpr (MAX_M == 1) {
        float sum = 0.0f;
        if constexpr (std::is_same_v<BlockT, block_nvfp4>) {
            sum = dot_row_nvfp4_sub(reinterpret_cast<const block_nvfp4*>(w_row), x_staged, K, lane);
        } else if constexpr (std::is_same_v<BlockT, block_q8_0>) {
            sum = dot_row_q8_0_sub(reinterpret_cast<const block_q8_0*>(w_row), x_staged, K, lane);
        } else if constexpr (std::is_same_v<BlockT, block_iq4_nl>) {
            sum = dot_row_iq4_nl_sub(reinterpret_cast<const block_iq4_nl*>(w_row), x_staged, K, lane);
        } else if constexpr (std::is_same_v<BlockT, block_q4_0>) {
            sum = dot_row_q4_0_sub(reinterpret_cast<const block_q4_0*>(w_row), x_staged, K, lane);
        } else if constexpr (std::is_same_v<BlockT, block_q5_1>) {
            sum = dot_row_q5_1_sub(reinterpret_cast<const block_q5_1*>(w_row), x_staged, K, lane);
        } else if constexpr (std::is_same_v<BlockT, block_q2_0>) {
            sum = dot_row_q2_0_sub(reinterpret_cast<const block_q2_0*>(w_row), x_staged, K, lane);
        } else {
            float a0 = 0.0f;
            for (int b = lane; b < nb_w; b += 32) {
                a0 += DotFn(w_row + b, x_staged + b * q8_per_w);
            }
            sum = warp_reduce_sum(a0);
        }
        if (lane == 0) {
            if (conv_w != nullptr) {
                float y = eval_static_short_conv_m1(row, conv_offset, n_conv, sum, conv_w, conv_history);
                if (add_res != nullptr) y += add_res[row];
                Y[row] = y;
            } else {
                Y[row] = sum;
            }
        }
    } else {
        float acc[MAX_M];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;
        if constexpr (std::is_same_v<BlockT, block_nvfp4>) {
            const block_nvfp4* wr = reinterpret_cast<const block_nvfp4*>(w_row);
            int sub = lane & 7;
            int blk0 = lane >> 3;
            int sb = sub >> 1;
            int h4 = sub & 1;
            int x_q8_off = sb >> 1;
            int x_byte_lo = (sb & 1) * 16 + h4 * 4;
            int x_byte_hi = x_byte_lo + 8;
            int scale_shift = sb * 8;
            for (int b = blk0; b < nb_w; b += 4) {
                const block_nvfp4* wb = wr + b;
                uint32_t d4 = (uint32_t)load_i32_align4(wb->d);
                uint32_t qw = (uint32_t)load_i32_align4(wb->qs + sub * 4);
                int wlo = lut_nvfp4_x4(qw & 0x0F0F0F0Fu);
                int whi = lut_nvfp4_x4((qw >> 4) & 0x0F0F0F0Fu);
                float wd = ue4m3_to_f32_dev((uint8_t)(d4 >> scale_shift));
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const block_q8_1* xb = X_q8 + (size_t)m * nb_x + b * 2 + x_q8_off;
                        int s = dp4a_packed(wlo, load_i32_align4(xb->qs + x_byte_lo), 0);
                        s = dp4a_packed(whi, load_i32_align4(xb->qs + x_byte_hi), s);
                        acc[m] += (wd * fp16_bits_to_f32(load_u16_align2(&xb->d))) * (float)s;
                    }
                }
            }
        } else if constexpr (std::is_same_v<BlockT, block_q8_0>) {
            const block_q8_0* wr = reinterpret_cast<const block_q8_0*>(w_row);
            int sub = lane & 7;
            int blk0 = lane >> 3;
            for (int b = blk0; b < nb_w; b += 4) {
                const block_q8_0* wb = wr + b;
                int wa = load_i32_align2(wb->qs + sub * 4);
                float wd = fp16_bits_to_f32(load_u16_align2(&wb->d));
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const block_q8_1* xb = X_q8 + (size_t)m * nb_x + b;
                        int s = dp4a_packed(wa, load_i32_align4(xb->qs + sub * 4), 0);
                        acc[m] += (wd * fp16_bits_to_f32(load_u16_align2(&xb->d))) * (float)s;
                    }
                }
            }
        } else if constexpr (std::is_same_v<BlockT, block_iq4_nl>) {
            const block_iq4_nl* wr = reinterpret_cast<const block_iq4_nl*>(w_row);
            int sub = lane & 3;
            int blk0 = lane >> 2;
            for (int b = blk0; b < nb_w; b += 8) {
                const block_iq4_nl* wb = wr + b;
                uint32_t q4 = (uint32_t)load_i32_align2(wb->qs + sub * 4);
                int wlo = lut_iq4nl_x4(q4 & 0x0F0F0F0Fu);
                int whi = lut_iq4nl_x4((q4 >> 4) & 0x0F0F0F0Fu);
                float wd = fp16_bits_to_f32(load_u16_align2(&wb->d));
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const block_q8_1* xb = X_q8 + (size_t)m * nb_x + b;
                        int s = dp4a_packed(wlo, load_i32_align4(xb->qs + sub * 4), 0);
                        s = dp4a_packed(whi, load_i32_align4(xb->qs + 16 + sub * 4), s);
                        acc[m] += (wd * fp16_bits_to_f32(load_u16_align2(&xb->d))) * (float)s;
                    }
                }
            }
        } else if constexpr (std::is_same_v<BlockT, block_q5_1>) {
            const block_q5_1* wr = reinterpret_cast<const block_q5_1*>(w_row);
            int sub = lane & 3;
            int blk0 = lane >> 2;
            for (int b = blk0; b < nb_w; b += 8) {
                const block_q5_1* wb = wr + b;
                uint32_t qh = (uint32_t)load_i32_align2(wb->qh);
                uint32_t q4 = (uint32_t)load_i32_align2(wb->qs + sub * 4);
                int wlo = (int)((q4 & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (sub * 4)));
                int whi = (int)(((q4 >> 4) & 0x0F0F0F0Fu) | expand_q5_1_hi4(qh >> (16 + sub * 4)));
                float wd = fp16_bits_to_f32(load_u16_align2(&wb->d));
                float wm = fp16_bits_to_f32(load_u16_align2(&wb->m));
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const block_q8_1* xb = X_q8 + (size_t)m * nb_x + b;
                        int s = dp4a_packed(wlo, load_i32_align4(xb->qs + sub * 4), 0);
                        s = dp4a_packed(whi, load_i32_align4(xb->qs + 16 + sub * 4), s);
                        float2 ds8 = fp16x2_bits_to_f32(&xb->d);
                        acc[m] += wd * ds8.x * (float)s;
                        if (sub == 0) acc[m] += wm * ds8.y;
                    }
                }
            }
        } else if constexpr (std::is_same_v<BlockT, block_q2_0>) {
            const block_q2_0* wr = reinterpret_cast<const block_q2_0*>(w_row);
            int sub = lane & 3;
            int blk0 = lane >> 2;
            int xoff = (sub & 1) * 16;
            for (int b = blk0; b < nb_w; b += 8) {
                const block_q2_0* wb = wr + b;
                uint32_t q4 = (uint32_t)load_i32_align2(wb->qs + sub * 4);
                int w4[4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) w4[j] = unpack_q2_0_x4((q4 >> (8 * j)) & 0xFFu);
                float wd = fp16_bits_to_f32(load_u16_align2(&wb->d));
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const block_q8_1* xb = X_q8 + (size_t)m * nb_x + b * 2 + (sub >> 1);
                        int s = 0;
                        #pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            s = dp4a_packed(w4[j], load_i32_align4(xb->qs + xoff + j * 4), s);
                        }
                        acc[m] += (wd * fp16_bits_to_f32(load_u16_align2(&xb->d))) * (float)s;
                    }
                }
            }
        } else {
            for (int b = lane; b < nb_w; b += 32) {
                const BlockT* wb = w_row + b;
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < M) {
                        const block_q8_1* xb = X_q8 + (size_t)m * nb_x + b * q8_per_w;
                        acc[m] += DotFn(wb, xb);
                    }
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
                float sum = warp_reduce_sum(acc[m]);
                if (lane == 0) {
                    Y[(size_t)m * N + row] = sum;
                }
            }
        }
    }
}

// Coalesced 8-lane-per-superblock batched GEMV (2 <= M <= MAX_M) for 256-elem K-quants:
// unpacks weight sub-blocks into registers ONCE and applies dp4a across all M tokens.
template <int W_TYPE, int MAX_M>
__global__ void __launch_bounds__(128) glm5_mmvq_sub256_batch_kernel(
    const uint8_t* __restrict__ W,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y,
    int N,
    int K,
    int M
) {
    constexpr int MAX_STAGED_NB = 512; // 18 KB SMEM
    __shared__ __align__(16) uint32_t s_x_words[MAX_STAGED_NB * 9];

    const int nb_x = K >> 5;
    const int nb_w = K >> 8;
    const int tid  = threadIdx.y * 32 + threadIdx.x;
    const int lane = threadIdx.x;
    const int tot_nb = M * nb_x;
    const bool use_smem = (tot_nb <= MAX_STAGED_NB);
    if (use_smem) {
        const uint32_t* gx = reinterpret_cast<const uint32_t*>(X_q8);
        const int nwords = tot_nb * 9;
        for (int i = tid; i < nwords; i += 128) {
            s_x_words[i] = gx[i];
        }
        __syncthreads();
    }

    const int row = blockIdx.x * 4 + threadIdx.y;
    if (row >= N) return;

    const block_q8_1* x_use = use_smem
        ? reinterpret_cast<const block_q8_1*>(s_x_words)
        : X_q8;

    const int sub = lane & 7;
    const int b0  = lane >> 3;
    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    if constexpr (W_TYPE == GGML_TYPE_Q4_K) {
        const block_q4_K* wr = reinterpret_cast<const block_q4_K*>(W) + (size_t)row * nb_w;
        const int shift = 4 * (sub & 1);
        for (int b = b0; b < nb_w; b += 4) {
            const block_q4_K* blk = wr + b;
            const float2 dpair = fp16x2_bits_to_f32(&blk->d);
            uint8_t sc = 0, mn = 0;
            get_scale_min_k4_dev(sub, blk->scales, sc, mn);
            const float ws = dpair.x * (float)sc;
            const float wm = dpair.y * (float)mn;
            const uint8_t* qs = blk->qs + 32 * (sub >> 1);
            int w4[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                w4[j] = (int)(((uint32_t)load_i32_align4(qs + 4 * j) >> shift) & 0x0F0F0F0Fu);
            }
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const block_q8_1* xb = x_use + (size_t)m * nb_x + b * 8 + sub;
                    int sumi = 0;
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        sumi = dp4a_packed(w4[j], load_i32_align4(xb->qs + 4 * j), sumi);
                    }
                    const float2 ds8 = fp16x2_bits_to_f32(&xb->d);
                    acc[m] += ws * ds8.x * (float)sumi - wm * ds8.y;
                }
            }
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q6_K) {
        const block_q6_K* wr = reinterpret_cast<const block_q6_K*>(W) + (size_t)row * nb_w;
        const int half = sub >> 2;
        const int q = sub & 3;
        const int ql_shift = 4 * (q >> 1);
        const int qh_shift = 2 * q;
        for (int b = b0; b < nb_w; b += 4) {
            const block_q6_K* blk = wr + b;
            const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
            const uint8_t* ql = blk->ql + 64 * half + 32 * (q & 1);
            const uint8_t* qh = blk->qh + 32 * half;
            const int sc0 = (int)blk->scales[2 * sub + 0];
            const int sc1 = (int)blk->scales[2 * sub + 1];
            int w4[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t lo4 = ((uint32_t)load_i32_align2(ql + 4 * j) >> ql_shift) & 0x0F0F0F0Fu;
                uint32_t hi2 = ((uint32_t)load_i32_align2(qh + 4 * j) >> qh_shift) & 0x03030303u;
                uint32_t raw6 = lo4 | (hi2 << 4);
                w4[j] = (int)__vsub4(raw6, 0x20202020u);
            }
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const block_q8_1* xb = x_use + (size_t)m * nb_x + b * 8 + sub;
                    int sum0 = 0, sum1 = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        sum0 = dp4a_packed(w4[j],     load_i32_align4(xb->qs + 4 * j), sum0);
                        sum1 = dp4a_packed(w4[j + 4], load_i32_align4(xb->qs + 4 * (j + 4)), sum1);
                    }
                    const float d8 = fp16_bits_to_f32(load_u16_align2(&xb->d));
                    acc[m] += dw * d8 * (float)(sc0 * sum0 + sc1 * sum1);
                }
            }
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q3_K) {
        const block_q3_K* wr = reinterpret_cast<const block_q3_K*>(W) + (size_t)row * nb_w;
        const int a_idx = (sub >> 1) & 1;
        const int q_shift = 2 * (sub & 3);
        for (int b = b0; b < nb_w; b += 4) {
            const block_q3_K* blk = wr + b;
            const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
            uint32_t low4 = ((uint32_t)load_u16_align2(blk->scales + 4 * a_idx + 2 * (sub & 1)) >> (sub & 4)) & 0x0F0Fu;
            uint32_t hi2  = ((uint32_t)load_u16_align2(blk->scales + 8 + 2 * (sub & 1)) >> (2 * (sub >> 1))) & 0x0303u;
            uint32_t sc_pair = low4 | (hi2 << 4);
            const int sc0 = (int)(sc_pair & 0xFFu) - 32;
            const int sc1 = (int)((sc_pair >> 8) & 0xFFu) - 32;
            const uint8_t* qp = blk->qs + 32 * (sub >> 2);
            int w4[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                uint32_t q4 = ((uint32_t)load_i32_align2(qp + 4 * j) >> q_shift) & 0x03030303u;
                uint32_t hm = ((uint32_t)load_i32_align2(blk->hmask + 4 * j) >> sub) & 0x01010101u;
                unsigned sub4 = __vsub4(0x04040404u, hm << 2);
                w4[j] = (int)__vsub4(q4, sub4);
            }
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) {
                    const block_q8_1* xb = x_use + (size_t)m * nb_x + b * 8 + sub;
                    int sum0 = 0, sum1 = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        sum0 = dp4a_packed(w4[j],     load_i32_align4(xb->qs + 4 * j), sum0);
                        sum1 = dp4a_packed(w4[j + 4], load_i32_align4(xb->qs + 4 * (j + 4)), sum1);
                    }
                    const float d8 = fp16_bits_to_f32(load_u16_align2(&xb->d));
                    acc[m] += dw * d8 * (float)(sc0 * sum0 + sc1 * sum1);
                }
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float sum = warp_reduce_sum(acc[m]);
            if (lane == 0) {
                Y[(size_t)m * N + row] = sum;
            }
        }
    }
}

// FP32 / F16 / BF16 GEMV for F32 weights (routers, loop controllers) or F16/BF16 weights.
template <int W_TYPE, int MAX_M>
__global__ void mmvq_dense_kernel(
    const uint8_t* __restrict__ W,
    const float* __restrict__ X,
    float* __restrict__ Y,
    int N,
    int K,
    int M,
    const int32_t* __restrict__ skip_dev
) {
    if (skip_dev != nullptr && *skip_dev != 0) return;
    int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= N) return;
    int lane = threadIdx.x;

    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    if constexpr (W_TYPE == GGML_TYPE_F32) {
        const float* w_row = reinterpret_cast<const float*>(W) + (size_t)row * K;
        for (int k = lane; k < K; k += 32) {
            float wv = w_row[k];
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) acc[m] += wv * X[(size_t)m * K + k];
            }
        }
    } else if constexpr (W_TYPE == GGML_TYPE_F16) {
        const uint16_t* w_row = reinterpret_cast<const uint16_t*>(W) + (size_t)row * K;
        for (int k = lane; k < K; k += 32) {
            float wv = fp16_bits_to_f32(w_row[k]);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) acc[m] += wv * X[(size_t)m * K + k];
            }
        }
    } else if constexpr (W_TYPE == GGML_TYPE_BF16) {
        const uint16_t* w_row = reinterpret_cast<const uint16_t*>(W) + (size_t)row * K;
        for (int k = lane; k < K; k += 32) {
            float wv = bf16_bits_to_f32(w_row[k]);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < M) acc[m] += wv * X[(size_t)m * K + k];
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
            float sum = warp_reduce_sum(acc[m]);
            if (lane == 0) {
                Y[(size_t)m * N + row] = sum;
            }
        }
    }
}

// Embedding lookup + optional RMSNorm (token_embd_norm) for [M] tokens.
__global__ void embed_lookup_rmsnorm_kernel(
    const uint8_t* __restrict__ table,
    int32_t table_type,
    const int32_t* __restrict__ token_ids,
    const float* __restrict__ norm_weight, // nullable
    float* __restrict__ out,
    int n_embd,
    int M,
    float eps
) {
    int tok_idx = blockIdx.x;
    if (tok_idx >= M) return;
    int tid = threadIdx.x;
    int id = token_ids[tok_idx];

    extern __shared__ float smem[];

    size_t row_bytes = 0;
    if (table_type == GGML_TYPE_F32) row_bytes = (size_t)n_embd * 4;
    else if (table_type == GGML_TYPE_F16 || table_type == GGML_TYPE_BF16) row_bytes = (size_t)n_embd * 2;
    else if (table_type == GGML_TYPE_Q8_0) row_bytes = (size_t)(n_embd / 32) * sizeof(block_q8_0);
    else if (table_type == GGML_TYPE_Q4_0) row_bytes = (size_t)(n_embd / 32) * sizeof(block_q4_0);
    else if (table_type == GGML_TYPE_IQ4_NL) row_bytes = (size_t)(n_embd / 32) * sizeof(block_iq4_nl);
    else if (table_type == GGML_TYPE_Q4_K) row_bytes = (size_t)(n_embd / 256) * sizeof(block_q4_K);
    else if (table_type == GGML_TYPE_Q5_K) row_bytes = (size_t)(n_embd / 256) * sizeof(block_q5_K);
    else if (table_type == GGML_TYPE_Q6_K) row_bytes = (size_t)(n_embd / 256) * sizeof(block_q6_K);
    else if (table_type == GGML_TYPE_IQ4_XS) row_bytes = (size_t)(n_embd / 256) * sizeof(block_iq4_xs);
    else if (table_type == GGML_TYPE_NVFP4) row_bytes = (size_t)(n_embd / 64) * sizeof(block_nvfp4);

    const uint8_t* row_ptr = table + (size_t)id * row_bytes;

    for (int i = tid; i < n_embd; i += blockDim.x) {
        float val = 0.0f;
        if (table_type == GGML_TYPE_F32) {
            val = reinterpret_cast<const float*>(row_ptr)[i];
        } else if (table_type == GGML_TYPE_F16) {
            val = fp16_bits_to_f32(reinterpret_cast<const uint16_t*>(row_ptr)[i]);
        } else if (table_type == GGML_TYPE_BF16) {
            val = bf16_bits_to_f32(reinterpret_cast<const uint16_t*>(row_ptr)[i]);
        } else if (table_type == GGML_TYPE_Q8_0) {
            int b = i >> 5;
            int j = i & 31;
            const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(row_ptr) + b;
            val = fp16_bits_to_f32(load_u16_unaligned(&blk->d)) * (float)blk->qs[j];
        } else if (table_type == GGML_TYPE_Q4_0) {
            int b = i >> 5;
            int j = i & 31;
            const block_q4_0* blk = reinterpret_cast<const block_q4_0*>(row_ptr) + b;
            int q = (j < 16) ? (blk->qs[j] & 0xF) : (blk->qs[j - 16] >> 4);
            val = fp16_bits_to_f32(load_u16_unaligned(&blk->d)) * (float)(q - 8);
        } else if (table_type == GGML_TYPE_IQ4_NL) {
            int b = i >> 5;
            int j = i & 31;
            const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(row_ptr) + b;
            int q = (j < 16) ? (blk->qs[j] & 0xF) : (blk->qs[j - 16] >> 4);
            val = fp16_bits_to_f32(load_u16_unaligned(&blk->d)) * (float)KVALUES_IQ4NL_DEV[q];
        } else if (table_type == GGML_TYPE_Q6_K) {
            int b = i >> 8;
            int r = i & 255;
            int half = r >> 7;
            int r128 = r & 127;
            int qgroup = r128 >> 5; // 0..3
            int l = r128 & 31;
            int is = (l >> 4) + qgroup * 2;
            const block_q6_K* blk = reinterpret_cast<const block_q6_K*>(row_ptr) + b;
            const uint8_t* ql = blk->ql + half * 64;
            const uint8_t* qh = blk->qh + half * 32;
            const int8_t* sc = blk->scales + half * 8;
            int ql_idx = (qgroup & 1) ? (l + 32) : l;
            int lo4 = (qgroup < 2) ? (ql[ql_idx] & 0xF) : (ql[ql_idx] >> 4);
            int hi2 = (qh[l] >> (2 * qgroup)) & 3;
            int q = (lo4 | (hi2 << 4)) - 32;
            val = fp16_bits_to_f32(load_u16_unaligned(&blk->d)) * (float)sc[is] * (float)q;
        } else if (table_type == GGML_TYPE_Q4_K) {
            int b = i >> 8;
            int r = i & 255;
            int pair = r >> 6;
            int r64 = r & 63;
            int sub = r64 >> 5;
            int l = r64 & 31;
            const block_q4_K* blk = reinterpret_cast<const block_q4_K*>(row_ptr) + b;
            uint8_t sc, m;
            get_scale_min_k4_dev(pair * 2 + sub, blk->scales, sc, m);
            uint8_t qb = blk->qs[pair * 32 + l];
            int q = (sub == 0) ? (qb & 0xF) : (qb >> 4);
            val = fp16_bits_to_f32(load_u16_unaligned(&blk->d)) * (float)sc * (float)q
                - fp16_bits_to_f32(load_u16_unaligned(&blk->dmin)) * (float)m;
        } else if (table_type == GGML_TYPE_Q5_K) {
            int b = i >> 8;
            int r = i & 255;
            int pair = r >> 6;
            int r64 = r & 63;
            int sub = r64 >> 5;
            int l = r64 & 31;
            const block_q5_K* blk = reinterpret_cast<const block_q5_K*>(row_ptr) + b;
            uint8_t sc, m;
            get_scale_min_k4_dev(pair * 2 + sub, blk->scales, sc, m);
            uint8_t qb = blk->qs[pair * 32 + l];
            uint8_t u = 1u << (pair * 2 + sub);
            int q = ((sub == 0) ? (qb & 0xF) : (qb >> 4)) + ((blk->qh[l] & u) ? 16 : 0);
            val = fp16_bits_to_f32(load_u16_unaligned(&blk->d)) * (float)sc * (float)q
                - fp16_bits_to_f32(load_u16_unaligned(&blk->dmin)) * (float)m;
        } else if (table_type == GGML_TYPE_IQ4_XS) {
            int b = i >> 8;
            int r = i & 255;
            int ib = r >> 5;
            int j = r & 31;
            const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(row_ptr) + b;
            uint16_t sh = load_u16_unaligned(&blk->scales_h);
            int ls = ((blk->scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
            float dl = fp16_bits_to_f32(load_u16_unaligned(&blk->d)) * (float)(ls - 32);
            int q = (j < 16) ? (blk->qs[ib * 16 + j] & 0xF) : (blk->qs[ib * 16 + j - 16] >> 4);
            val = dl * (float)KVALUES_IQ4NL_DEV[q];
        } else if (table_type == GGML_TYPE_NVFP4) {
            int b = i >> 6;
            int r = i & 63;
            int sb = r >> 4;
            int j = r & 15;
            const block_nvfp4* blk = reinterpret_cast<const block_nvfp4*>(row_ptr) + b;
            uint8_t qb = blk->qs[sb * 8 + (j & 7)];
            int q = (j < 8) ? (qb & 0xF) : (qb >> 4);
            val = ue4m3_to_f32_dev(blk->d[sb]) * (float)KVALUES_NVFP4_DEV[q];
        }
        smem[i] = val;
    }
    __syncthreads();

    if (norm_weight != nullptr) {
        float sum2 = 0.0f;
        for (int i = tid; i < n_embd; i += blockDim.x) {
            float v = smem[i];
            sum2 += v * v;
        }
        sum2 = warp_reduce_sum(sum2);
        __shared__ float warp_sums[32];
        int warp_id = tid >> 5;
        int lane = tid & 31;
        if (lane == 0) warp_sums[warp_id] = sum2;
        __syncthreads();
        if (warp_id == 0) {
            float tot = (lane < (blockDim.x >> 5)) ? warp_sums[lane] : 0.0f;
            tot = warp_reduce_sum(tot);
            if (lane == 0) warp_sums[0] = rsqrtf(tot / (float)n_embd + eps);
        }
        __syncthreads();
        float inv_rms = warp_sums[0];
        for (int i = tid; i < n_embd; i += blockDim.x) {
            out[(size_t)tok_idx * n_embd + i] = smem[i] * inv_rms * norm_weight[i];
        }
    } else {
        for (int i = tid; i < n_embd; i += blockDim.x) {
            out[(size_t)tok_idx * n_embd + i] = smem[i];
        }
    }
}

template <int MAX_M, bool EXACT_M = false>
__global__ void __launch_bounds__(256, 2) mmvq_q8_0_splitk_kernel(
    const block_q8_0* __restrict__ W,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y,
    int N,
    int K,
    int M
) {
    const int row = (int)blockIdx.x;
    if (row >= N) return;
    const int warp = (int)threadIdx.y; // 0..7
    const int lane = (int)threadIdx.x; // 0..31
    const int nb = K >> 5;
    const int sub = lane & 7;
    const int b0 = warp * 4 + (lane >> 3);

    const block_q8_0* w_row = W + (size_t)row * (size_t)nb;
    float acc[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) acc[m] = 0.0f;

    for (int b = b0; b < nb; b += 32) {
        const block_q8_0* wb = w_row + b;
        const int wa = load_i32_align2(wb->qs + sub * 4);
        const float wd = fp16_bits_to_f32(load_u16_align2(&wb->d));
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const block_q8_1* xb = X_q8 + (size_t)m * (size_t)nb + b;
                const int s = dp4a_packed(wa, load_i32_align4(xb->qs + sub * 4), 0);
                acc[m] += (wd * fp16_bits_to_f32(load_u16_align2(&xb->d))) * (float)s;
            }
        }
    }

    __shared__ float s_red[8][MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float v = warp_reduce_sum(acc[m]);
            if (lane == 0) s_red[warp][m] = v;
        }
    }
    __syncthreads();

    if (warp == 0 && lane < M) {
        const float tot = s_red[0][lane] + s_red[1][lane] + s_red[2][lane] + s_red[3][lane]
                        + s_red[4][lane] + s_red[5][lane] + s_red[6][lane] + s_red[7][lane];
        Y[(size_t)lane * (size_t)N + (size_t)row] = tot;
    }
}

__device__ __forceinline__ int ldg_i32_align2(const void* ptr) {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(ptr);
    return (int)__ldg(p) | ((int)__ldg(p + 1) << 16);
}

__device__ __forceinline__ uint16_t ldg_u16_align2(const void* ptr) {
    return __ldg(reinterpret_cast<const uint16_t*>(ptr));
}

template <int MAX_M, bool EXACT_M = false>
__global__ void __launch_bounds__(256, 2) mmvq_q8_0_splitk_2row_kernel(
    const block_q8_0* __restrict__ W,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y,
    int N,
    int K,
    int M
) {
    const int row0 = (int)blockIdx.x * 2;
    if (row0 >= N) return;
    const int row1 = row0 + 1;
    const bool has1 = (row1 < N);
    const int warp = (int)threadIdx.y; // 0..7
    const int lane = (int)threadIdx.x; // 0..31
    const int nb = K >> 5;
    const int sub = lane & 7;
    const int b0 = warp * 4 + (lane >> 3);

    const block_q8_0* wr0 = W + (size_t)row0 * (size_t)nb;
    const block_q8_0* wr1 = has1 ? (W + (size_t)row1 * (size_t)nb) : nullptr;
    float acc0[MAX_M];
    float acc1[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int b = b0; b < nb; b += 32) {
        const block_q8_0* wb0 = wr0 + b;
        const int wa0 = load_i32_align2(wb0->qs + sub * 4);
        const float wd0 = fp16_bits_to_f32(load_u16_align2(&wb0->d));
        int wa1 = 0;
        float wd1 = 0.0f;
        if (has1) {
            const block_q8_0* wb1 = wr1 + b;
            wa1 = load_i32_align2(wb1->qs + sub * 4);
            wd1 = fp16_bits_to_f32(load_u16_align2(&wb1->d));
        }
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const block_q8_1* xb = X_q8 + (size_t)m * (size_t)nb + b;
                const int xa = load_i32_align4(xb->qs + sub * 4);
                const float xd = fp16_bits_to_f32(load_u16_align2(&xb->d));
                const int s0 = dp4a_packed(wa0, xa, 0);
                acc0[m] += (wd0 * xd) * (float)s0;
                if (has1) {
                    const int s1 = dp4a_packed(wa1, xa, 0);
                    acc1[m] += (wd1 * xd) * (float)s1;
                }
            }
        }
    }

    __shared__ float s_red[8][2][MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float v0 = warp_reduce_sum(acc0[m]);
            if (lane == 0) s_red[warp][0][m] = v0;
            if (has1) {
                const float v1 = warp_reduce_sum(acc1[m]);
                if (lane == 0) s_red[warp][1][m] = v1;
            }
        }
    }
    __syncthreads();

    if (warp == 0 && lane < M) {
        const float tot0 = s_red[0][0][lane] + s_red[1][0][lane] + s_red[2][0][lane] + s_red[3][0][lane]
                         + s_red[4][0][lane] + s_red[5][0][lane] + s_red[6][0][lane] + s_red[7][0][lane];
        Y[(size_t)lane * (size_t)N + (size_t)row0] = tot0;
        if (has1) {
            const float tot1 = s_red[0][1][lane] + s_red[1][1][lane] + s_red[2][1][lane] + s_red[3][1][lane]
                             + s_red[4][1][lane] + s_red[5][1][lane] + s_red[6][1][lane] + s_red[7][1][lane];
            Y[(size_t)lane * (size_t)N + (size_t)row1] = tot1;
        }
    }
}

template <int MAX_M, bool EXACT_M = false>
__global__ void __launch_bounds__(256, 2) mmvq_q8_0_batch_kernel(
    const block_q8_0* __restrict__ W,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y,
    int N,
    int K,
    int M
) {
    constexpr int KB = 40;
    __shared__ __align__(16) uint32_t s_x[MAX_M * KB * 9];
    __shared__ float s_xd[MAX_M][KB];

    const int nb = K >> 5;
    const int tid = (int)threadIdx.y * 32 + (int)threadIdx.x;
    const int warp = (int)threadIdx.y; // 0..7
    const int lane = (int)threadIdx.x; // 0..31
    const int sub = lane & 7;
    const int b0 = lane >> 3;

    const int row0 = (int)blockIdx.x * 16 + warp * 2;
    const int row1 = row0 + 1;
    const bool has0 = (row0 < N);
    const bool has1 = (row1 < N);

    const block_q8_0* wr0 = has0 ? (W + (size_t)row0 * (size_t)nb) : nullptr;
    const block_q8_0* wr1 = has1 ? (W + (size_t)row1 * (size_t)nb) : nullptr;

    float acc0[MAX_M];
    float acc1[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int b_start = 0; b_start < nb; b_start += KB) {
        const int b_cnt = (nb - b_start < KB) ? (nb - b_start) : KB;
        const int rw = b_cnt * 9;
        for (int m = 0; m < M; ++m) {
            const uint32_t* gxm = reinterpret_cast<const uint32_t*>(
                X_q8 + (size_t)m * (size_t)nb + b_start
            );
            uint32_t* sxm = s_x + m * KB * 9;
            for (int i = tid; i < rw; i += 256) {
                sxm[i] = gxm[i];
            }
        }
        const int tot_b = M * b_cnt;
        for (int i = tid; i < tot_b; i += 256) {
            const int m = i / b_cnt;
            const int b = i - m * b_cnt;
            const block_q8_1* xb = X_q8 + (size_t)m * (size_t)nb + b_start + b;
            s_xd[m][b] = fp16_bits_to_f32(load_u16_align2(&xb->d));
        }
        __syncthreads();

        if (has0) {
            for (int b = b0; b < b_cnt; b += 4) {
                const block_q8_0* wb0 = wr0 + (b_start + b);
                const int wa0 = ldg_i32_align2(wb0->qs + sub * 4);
                const float wd0 = fp16_bits_to_f32(ldg_u16_align2(&wb0->d));
                int wa1 = 0;
                float wd1 = 0.0f;
                if (has1) {
                    const block_q8_0* wb1 = wr1 + (b_start + b);
                    wa1 = ldg_i32_align2(wb1->qs + sub * 4);
                    wd1 = fp16_bits_to_f32(ldg_u16_align2(&wb1->d));
                }
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (EXACT_M || m < M) {
                        const block_q8_1* xb = reinterpret_cast<const block_q8_1*>(
                            s_x + m * KB * 9
                        ) + b;
                        const int xa = load_i32_align4(xb->qs + sub * 4);
                        const float xd = s_xd[m][b];
                        const int s0 = dp4a_packed(wa0, xa, 0);
                        acc0[m] += (wd0 * xd) * (float)s0;
                        if (has1) {
                            const int s1 = dp4a_packed(wa1, xa, 0);
                            acc1[m] += (wd1 * xd) * (float)s1;
                        }
                    }
                }
            }
        }
        if (b_start + KB < nb) __syncthreads();
    }

    if (has0) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const float sum0 = warp_reduce_sum(acc0[m]);
                if (lane == 0) Y[(size_t)m * (size_t)N + (size_t)row0] = sum0;
                if (has1) {
                    const float sum1 = warp_reduce_sum(acc1[m]);
                    if (lane == 0) Y[(size_t)m * (size_t)N + (size_t)row1] = sum1;
                }
            }
        }
    }
}

template <int MAX_M, bool EXACT_M, int N_MATS>
__global__ void __launch_bounds__(256, 2) mmvq_q8_0_multi_batch_kernel(
    const block_q8_0* __restrict__ W0, float* __restrict__ Y0, int N0,
    const block_q8_0* __restrict__ W1, float* __restrict__ Y1, int N1,
    const block_q8_0* __restrict__ W2, float* __restrict__ Y2, int N2,
    const block_q8_1* __restrict__ X_q8,
    int K,
    int M
) {
    constexpr int KB = 40;
    __shared__ __align__(16) uint32_t s_x[MAX_M * KB * 9];
    __shared__ float s_xd[MAX_M][KB];

    const int nb = K >> 5;
    const int tid = (int)threadIdx.y * 32 + (int)threadIdx.x;
    const int warp = (int)threadIdx.y;
    const int lane = (int)threadIdx.x;
    const int sub = lane & 7;
    const int b0 = lane >> 3;

    const int row0 = (int)blockIdx.x * 16 + warp * 2;
    const int row1 = row0 + 1;

    const block_q8_0* wr0 = nullptr;
    const block_q8_0* wr1 = nullptr;
    float* dst0 = nullptr;
    float* dst1 = nullptr;
    int out_n0 = 0, out_n1 = 0;
    int lr0 = 0, lr1 = 0;
    bool has0 = false, has1 = false;

    auto map_row = [&](int r, const block_q8_0*& wr, float*& dst, int& out_n, int& lr, bool& has) {
        if (r < N0) {
            has = true; lr = r; out_n = N0; wr = W0 + (size_t)lr * (size_t)nb; dst = Y0;
        } else if (r < N0 + N1) {
            has = true; lr = r - N0; out_n = N1; wr = W1 + (size_t)lr * (size_t)nb; dst = Y1;
        } else if (N_MATS == 3 && r < N0 + N1 + N2) {
            has = true; lr = r - N0 - N1; out_n = N2; wr = W2 + (size_t)lr * (size_t)nb; dst = Y2;
        }
    };
    map_row(row0, wr0, dst0, out_n0, lr0, has0);
    map_row(row1, wr1, dst1, out_n1, lr1, has1);

    float acc0[MAX_M];
    float acc1[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int b_start = 0; b_start < nb; b_start += KB) {
        const int b_cnt = (nb - b_start < KB) ? (nb - b_start) : KB;
        const int rw = b_cnt * 9;
        for (int m = 0; m < M; ++m) {
            const uint32_t* gxm = reinterpret_cast<const uint32_t*>(
                X_q8 + (size_t)m * (size_t)nb + b_start
            );
            uint32_t* sxm = s_x + m * KB * 9;
            for (int i = tid; i < rw; i += 256) {
                sxm[i] = gxm[i];
            }
        }
        const int tot_b = M * b_cnt;
        for (int i = tid; i < tot_b; i += 256) {
            const int m = i / b_cnt;
            const int b = i - m * b_cnt;
            const block_q8_1* xb = X_q8 + (size_t)m * (size_t)nb + b_start + b;
            s_xd[m][b] = fp16_bits_to_f32(load_u16_align2(&xb->d));
        }
        __syncthreads();

        if (has0) {
            for (int b = b0; b < b_cnt; b += 4) {
                const block_q8_0* wb0 = wr0 + (b_start + b);
                const int wa0 = ldg_i32_align2(wb0->qs + sub * 4);
                const float wd0 = fp16_bits_to_f32(ldg_u16_align2(&wb0->d));
                int wa1 = 0;
                float wd1 = 0.0f;
                if (has1) {
                    const block_q8_0* wb1 = wr1 + (b_start + b);
                    wa1 = ldg_i32_align2(wb1->qs + sub * 4);
                    wd1 = fp16_bits_to_f32(ldg_u16_align2(&wb1->d));
                }
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (EXACT_M || m < M) {
                        const block_q8_1* xb = reinterpret_cast<const block_q8_1*>(
                            s_x + m * KB * 9
                        ) + b;
                        const int xa = load_i32_align4(xb->qs + sub * 4);
                        const float xd = s_xd[m][b];
                        const int s0 = dp4a_packed(wa0, xa, 0);
                        acc0[m] += (wd0 * xd) * (float)s0;
                        if (has1) {
                            const int s1 = dp4a_packed(wa1, xa, 0);
                            acc1[m] += (wd1 * xd) * (float)s1;
                        }
                    }
                }
            }
        }
        if (b_start + KB < nb) __syncthreads();
    }

    if (has0) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const float sum0 = warp_reduce_sum(acc0[m]);
                if (lane == 0) dst0[(size_t)m * (size_t)out_n0 + (size_t)lr0] = sum0;
                if (has1) {
                    const float sum1 = warp_reduce_sum(acc1[m]);
                    if (lane == 0) dst1[(size_t)m * (size_t)out_n1 + (size_t)lr1] = sum1;
                }
            }
        }
    }
}

template <int MAX_M, bool EXACT_M>
__global__ void __launch_bounds__(256, 2) qwen4_sh_gu_swiglu_kernel(
    const block_q8_0* __restrict__ W_gate,
    const block_q8_0* __restrict__ W_up,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y_act,
    int N,
    int K,
    int M
) {
    const int row = (int)blockIdx.x;
    if (row >= N) return;
    const int warp = (int)threadIdx.y; // 0..7
    const int lane = (int)threadIdx.x; // 0..31
    const int nb = K >> 5;
    const int sub = lane & 7;
    const int b0 = warp * 4 + (lane >> 3);

    const block_q8_0* wg_row = W_gate + (size_t)row * (size_t)nb;
    const block_q8_0* wu_row = W_up + (size_t)row * (size_t)nb;

    float acc_g[MAX_M];
    float acc_u[MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        acc_g[m] = 0.0f;
        acc_u[m] = 0.0f;
    }

    for (int b = b0; b < nb; b += 32) {
        const block_q8_0* wgb = wg_row + b;
        const int wag = ldg_i32_align2(wgb->qs + sub * 4);
        const float wdg = fp16_bits_to_f32(ldg_u16_align2(&wgb->d));

        const block_q8_0* wub = wu_row + b;
        const int wau = ldg_i32_align2(wub->qs + sub * 4);
        const float wdu = fp16_bits_to_f32(ldg_u16_align2(&wub->d));

        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (EXACT_M || m < M) {
                const block_q8_1* xb = X_q8 + (size_t)m * (size_t)nb + b;
                const int xa = load_i32_align4(xb->qs + sub * 4);
                const float xd = fp16_bits_to_f32(load_u16_align2(&xb->d));
                const int sg = dp4a_packed(wag, xa, 0);
                const int su = dp4a_packed(wau, xa, 0);
                acc_g[m] += (wdg * xd) * (float)sg;
                acc_u[m] += (wdu * xd) * (float)su;
            }
        }
    }

    __shared__ float s_red[8][2][MAX_M];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (EXACT_M || m < M) {
            const float vg = warp_reduce_sum(acc_g[m]);
            const float vu = warp_reduce_sum(acc_u[m]);
            if (lane == 0) {
                s_red[warp][0][m] = vg;
                s_red[warp][1][m] = vu;
            }
        }
    }
    __syncthreads();

    if (warp == 0 && lane < M) {
        const float g = s_red[0][0][lane] + s_red[1][0][lane] + s_red[2][0][lane] + s_red[3][0][lane]
                      + s_red[4][0][lane] + s_red[5][0][lane] + s_red[6][0][lane] + s_red[7][0][lane];
        const float u = s_red[0][1][lane] + s_red[1][1][lane] + s_red[2][1][lane] + s_red[3][1][lane]
                      + s_red[4][1][lane] + s_red[5][1][lane] + s_red[6][1][lane] + s_red[7][1][lane];
        Y_act[(size_t)lane * (size_t)N + (size_t)row] = (g / (1.0f + expf(-g))) * u;
    }
}

__global__ void __launch_bounds__(256, 2) mmvq_q8_0_indexed_m1_kernel(
    const block_q8_0* __restrict__ W,
    const int32_t* __restrict__ subset_ids,
    const block_q8_1* __restrict__ X_q8,
    float* __restrict__ Y,
    int n_sub,
    int K
) {
    constexpr int MAX_STAGED_NB = 128;
    __shared__ __align__(16) uint32_t s_x_words[MAX_STAGED_NB * 9];

    const int nb = K >> 5;
    const int lane = (int)threadIdx.x;
    const int warp = (int)threadIdx.y;
    const int tid = warp * 32 + lane;

    const block_q8_1* x_staged = X_q8;
    if (nb <= MAX_STAGED_NB) {
        const int total_words = nb * 9;
        const uint32_t* gx = reinterpret_cast<const uint32_t*>(X_q8);
        for (int i = tid; i < total_words; i += 256) {
            s_x_words[i] = gx[i];
        }
        __syncthreads();
        x_staged = reinterpret_cast<const block_q8_1*>(s_x_words);
    }

    const int sub_idx = (int)blockIdx.x * 8 + warp;
    if (sub_idx >= n_sub) return;
    const int src_row = __ldg(subset_ids + sub_idx);
    const block_q8_0* w_row = W + (size_t)src_row * (size_t)nb;
    const float sum = dot_row_q8_0_sub(w_row, x_staged, K, lane);
    if (lane == 0) {
        Y[sub_idx] = sum;
    }
}

extern "C" {

void launch_quantize_q8_1(
    const float* x,
    void* y_q8_1,
    int M,
    int K,
    cudaStream_t stream
) {
    int nb = K / 32;
    const int warps = 4;
    dim3 block(32, warps);
    dim3 grid((nb + warps - 1) / warps, M);
    quantize_f32_to_q8_1_kernel<<<grid, block, 0, stream>>>(
        x, reinterpret_cast<block_q8_1*>(y_q8_1), K
    );
}

void launch_mv_q8_planar_m1(
    const void* q0, const void* d0, int n0, const float* x0, float* y0,
    const void* q1, const void* d1, int n1, const float* x1, float* y1,
    const float* conv_w1, int conv_off1,
    const void* q2, const void* d2, int n2, const float* x2, float* y2,
    const float* conv_w2, int conv_off2,
    const void* q3, const void* d3, int n3, const float* x3, float* y3,
    const void* q4, const void* d4, int n4, const float* x4, float* y4,
    const float* add_res0,
    float* conv_history,
    int n_conv,
    int K,
    cudaStream_t stream
) {
    int total_rows = n0 + n1 + n2 + n3 + n4;
    const int warps = 4;
    dim3 block(32, warps);
    dim3 grid((total_rows + warps - 1) / warps);
    mv_q8_planar_m1_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const int8_t*>(q0), reinterpret_cast<const uint16_t*>(d0), n0, x0, y0,
        reinterpret_cast<const int8_t*>(q1), reinterpret_cast<const uint16_t*>(d1), n1, x1, y1,
        conv_w1, conv_off1,
        reinterpret_cast<const int8_t*>(q2), reinterpret_cast<const uint16_t*>(d2), n2, x2, y2,
        conv_w2, conv_off2,
        reinterpret_cast<const int8_t*>(q3), reinterpret_cast<const uint16_t*>(d3), n3, x3, y3,
        reinterpret_cast<const int8_t*>(q4), reinterpret_cast<const uint16_t*>(d4), n4, x4, y4,
        add_res0,
        conv_history,
        n_conv,
        K
    );
}

void launch_fused_qkvrg_m1(
    int32_t w_type,
    const void* w0, int n0, const void* x0, float* y0,
    const void* w1, int n1, const void* x1, float* y1,
    const float* conv_w1, int conv_off1,
    const void* w2, int n2, const void* x2, float* y2,
    const float* conv_w2, int conv_off2,
    const void* w3, int n3, const void* x3, float* y3,
    const void* w4, int n4, const void* x4, float* y4,
    float* conv_history,
    int n_conv,
    int K,
    size_t row_bytes,
    cudaStream_t stream
) {
    int total_rows = n0 + n1 + n2 + n3 + n4;
    const int warps = 4;
    dim3 block(32, warps);
    dim3 grid((total_rows + warps - 1) / warps);
    fused_qkvrg_m1_kernel<<<grid, block, 0, stream>>>(
        w_type,
        reinterpret_cast<const uint8_t*>(w0), n0, reinterpret_cast<const block_q8_1*>(x0), y0,
        reinterpret_cast<const uint8_t*>(w1), n1, reinterpret_cast<const block_q8_1*>(x1), y1,
        conv_w1, conv_off1,
        reinterpret_cast<const uint8_t*>(w2), n2, reinterpret_cast<const block_q8_1*>(x2), y2,
        conv_w2, conv_off2,
        reinterpret_cast<const uint8_t*>(w3), n3, reinterpret_cast<const block_q8_1*>(x3), y3,
        reinterpret_cast<const uint8_t*>(w4), n4, reinterpret_cast<const block_q8_1*>(x4), y4,
        conv_history,
        n_conv,
        K,
        row_bytes
    );
}

static void launch_mmvq_chunk(
    int32_t w_type,
    const uint8_t* W,
    const block_q8_1* X_q8,
    const float* X_f32,
    float* Y,
    const float* conv_w,
    const float* add_res,
    float* conv_history,
    int conv_offset,
    int n_conv,
    int N,
    int K,
    int M,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    const int warps = 4;
    dim3 block(32, warps);
    dim3 grid_m1((N + warps - 1) / warps);
    dim3 grid_m8((N + warps - 1) / warps);

    if (w_type == GGML_TYPE_F32) {
        if (M == 1) mmvq_dense_kernel<GGML_TYPE_F32, 1><<<grid_m8, block, 0, stream>>>(W, X_f32, Y, N, K, M, skip_dev);
        else mmvq_dense_kernel<GGML_TYPE_F32, 8><<<grid_m8, block, 0, stream>>>(W, X_f32, Y, N, K, M, nullptr);
        return;
    }
    if (w_type == GGML_TYPE_F16) {
        if (M == 1) mmvq_dense_kernel<GGML_TYPE_F16, 1><<<grid_m8, block, 0, stream>>>(W, X_f32, Y, N, K, M, skip_dev);
        else mmvq_dense_kernel<GGML_TYPE_F16, 8><<<grid_m8, block, 0, stream>>>(W, X_f32, Y, N, K, M, nullptr);
        return;
    }
    if (w_type == GGML_TYPE_BF16) {
        if (M == 1) mmvq_dense_kernel<GGML_TYPE_BF16, 1><<<grid_m8, block, 0, stream>>>(W, X_f32, Y, N, K, M, skip_dev);
        else mmvq_dense_kernel<GGML_TYPE_BF16, 8><<<grid_m8, block, 0, stream>>>(W, X_f32, Y, N, K, M, nullptr);
        return;
    }

    if (w_type == GGML_TYPE_Q8_0 && (K & 31) == 0 && conv_w == nullptr && add_res == nullptr && conv_history == nullptr && skip_dev == nullptr) {
        const auto* w_q8 = reinterpret_cast<const block_q8_0*>(W);
        if (N <= 1024 && K >= 2560) {
            dim3 blk_sk(32, 8);
            if (M == 1 || N < 256 || (N <= 512 && K >= 4096)) {
                if (M == 1) {
                    mmvq_q8_0_splitk_kernel<1, true><<<(unsigned)N, blk_sk, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
                    return;
                }
                if (M <= 4) {
                    mmvq_q8_0_splitk_kernel<4, false><<<(unsigned)N, blk_sk, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
                    return;
                }
                if (M == 5) {
                    mmvq_q8_0_splitk_kernel<5, true><<<(unsigned)N, blk_sk, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
                    return;
                }
                mmvq_q8_0_splitk_kernel<8, false><<<(unsigned)N, blk_sk, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
                return;
            }
            const unsigned n2 = (unsigned)((N + 1) / 2);
            if (M <= 4) {
                mmvq_q8_0_splitk_2row_kernel<4, false><<<n2, blk_sk, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
                return;
            }
            if (M == 5) {
                mmvq_q8_0_splitk_2row_kernel<5, true><<<n2, blk_sk, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
                return;
            }
            mmvq_q8_0_splitk_2row_kernel<8, false><<<n2, blk_sk, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
            return;
        }
        if (M >= 2) {
            dim3 grid_q8((unsigned)((N + 15) / 16));
            dim3 blk_q8(32, 8);
            if (M <= 4) {
                mmvq_q8_0_batch_kernel<4, false><<<grid_q8, blk_q8, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
                return;
            }
            if (M == 5) {
                mmvq_q8_0_batch_kernel<5, true><<<grid_q8, blk_q8, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
                return;
            }
            mmvq_q8_0_batch_kernel<8, false><<<grid_q8, blk_q8, 0, stream>>>(w_q8, X_q8, Y, N, K, M);
            return;
        }
    }

    if (M == 1 && (K & 255) == 0 && conv_w == nullptr && add_res == nullptr && conv_history == nullptr) {
        if (w_type == GGML_TYPE_Q3_K || w_type == GGML_TYPE_Q4_K || w_type == GGML_TYPE_Q6_K ||
            w_type == GGML_TYPE_IQ2_S || w_type == GGML_TYPE_IQ3_XXS || w_type == GGML_TYPE_IQ3_S) {
            launch_glm5_mmvq_sub256_skip(w_type, W, X_q8, Y, N, K, skip_dev, stream);
            return;
        }
    }
    if (M >= 2 && M <= 4 && (K & 255) == 0 && conv_w == nullptr && add_res == nullptr && conv_history == nullptr) {
        if (w_type == GGML_TYPE_Q4_K) {
            glm5_mmvq_sub256_batch_kernel<GGML_TYPE_Q4_K, 4><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, N, K, M);
            return;
        }
        if (w_type == GGML_TYPE_Q6_K) {
            glm5_mmvq_sub256_batch_kernel<GGML_TYPE_Q6_K, 4><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, N, K, M);
            return;
        }
        if (w_type == GGML_TYPE_Q3_K) {
            glm5_mmvq_sub256_batch_kernel<GGML_TYPE_Q3_K, 4><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, N, K, M);
            return;
        }
    }
    if (M >= 5 && M <= 8 && (K & 255) == 0 && conv_w == nullptr && add_res == nullptr && conv_history == nullptr) {
        if (w_type == GGML_TYPE_Q4_K) {
            glm5_mmvq_sub256_batch_kernel<GGML_TYPE_Q4_K, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, N, K, M);
            return;
        }
        if (w_type == GGML_TYPE_Q6_K) {
            glm5_mmvq_sub256_batch_kernel<GGML_TYPE_Q6_K, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, N, K, M);
            return;
        }
        if (w_type == GGML_TYPE_Q3_K) {
            glm5_mmvq_sub256_batch_kernel<GGML_TYPE_Q3_K, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, N, K, M);
            return;
        }
    }

    switch (w_type) {
        case GGML_TYPE_Q8_0:
            if (M == 1) mmvq_quant_kernel<block_q8_0, 32, dot_q8_0_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_q8_0, 32, dot_q8_0_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_Q4_0:
            if (M == 1) mmvq_quant_kernel<block_q4_0, 32, dot_q4_0_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_q4_0, 32, dot_q4_0_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_IQ4_NL:
            if (M == 1) mmvq_quant_kernel<block_iq4_nl, 32, dot_iq4_nl_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_iq4_nl, 32, dot_iq4_nl_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_Q5_1:
            if (M == 1) mmvq_quant_kernel<block_q5_1, 32, dot_q5_1_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_q5_1, 32, dot_q5_1_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_Q2_0:
            if (M == 1) mmvq_quant_kernel<block_q2_0, 64, dot_q2_0_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_q2_0, 64, dot_q2_0_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_IQ4_XS:
            if (M == 1) mmvq_quant_kernel<block_iq4_xs, 256, dot_iq4_xs_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_iq4_xs, 256, dot_iq4_xs_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_Q3_K:
            if (M == 1) mmvq_quant_kernel<block_q3_K, 256, dot_q3_K_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_q3_K, 256, dot_q3_K_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_Q4_K:
            if (M == 1) mmvq_quant_kernel<block_q4_K, 256, dot_q4_K_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_q4_K, 256, dot_q4_K_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_Q5_K:
            if (M == 1) mmvq_quant_kernel<block_q5_K, 256, dot_q5_K_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_q5_K, 256, dot_q5_K_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_Q6_K:
            if (M == 1) mmvq_quant_kernel<block_q6_K, 256, dot_q6_K_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_q6_K, 256, dot_q6_K_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_IQ2_XXS:
            if (M == 1) mmvq_quant_kernel<block_iq2_xxs, 256, dot_iq2_xxs_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_iq2_xxs, 256, dot_iq2_xxs_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_IQ2_XS:
            if (M == 1) mmvq_quant_kernel<block_iq2_xs, 256, dot_iq2_xs_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_iq2_xs, 256, dot_iq2_xs_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_IQ1_S:
            if (M == 1) mmvq_quant_kernel<block_iq1_s, 256, dot_iq1_s_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_iq1_s, 256, dot_iq1_s_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_IQ2_S:
            if (M == 1) mmvq_quant_kernel<block_iq2_s, 256, dot_iq2_s_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_iq2_s, 256, dot_iq2_s_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_IQ3_XXS:
            if (M == 1) mmvq_quant_kernel<block_iq3_xxs, 256, dot_iq3_xxs_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_iq3_xxs, 256, dot_iq3_xxs_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_IQ3_S:
            if (M == 1) mmvq_quant_kernel<block_iq3_s, 256, dot_iq3_s_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_iq3_s, 256, dot_iq3_s_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
        case GGML_TYPE_NVFP4:
            if (M == 1) mmvq_quant_kernel<block_nvfp4, 64, dot_nvfp4_q8_1, 1><<<grid_m1, block, 0, stream>>>(W, X_q8, Y, conv_w, add_res, conv_history, conv_offset, n_conv, N, K, M, skip_dev);
            else        mmvq_quant_kernel<block_nvfp4, 64, dot_nvfp4_q8_1, 8><<<grid_m8, block, 0, stream>>>(W, X_q8, Y, nullptr, nullptr, nullptr, 0, 0, N, K, M, nullptr);
            break;
    }
}

void launch_linear_q8_1_skip(
    int32_t w_type,
    const void* W,
    const void* X_q8,
    const float* X_f32,
    float* Y,
    int N,
    int K,
    int M,
    const int32_t* skip_dev,
    cudaStream_t stream
) {
    int nb_x = K / 32;
    for (int m0 = 0; m0 < M; m0 += 8) {
        int chunk = (M - m0 < 8) ? (M - m0) : 8;
        launch_mmvq_chunk(
            w_type,
            reinterpret_cast<const uint8_t*>(W),
            reinterpret_cast<const block_q8_1*>(X_q8) + (size_t)m0 * nb_x,
            X_f32 ? (X_f32 + (size_t)m0 * K) : nullptr,
            Y + (size_t)m0 * N,
            nullptr, nullptr, nullptr, 0, 0,
            N,
            K,
            chunk,
            skip_dev,
            stream
        );
    }
}

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
) {
    launch_linear_q8_1_skip(w_type, W, X_q8, X_f32, Y, N, K, M, nullptr, stream);
}

void launch_linear_q8_1_conv_add_m1(
    int32_t w_type,
    const void* W,
    const void* X_q8,
    float* Y,
    const float* conv_w,
    const float* add_res,
    float* conv_history,
    int conv_offset,
    int n_conv,
    int N,
    int K,
    cudaStream_t stream
) {
    launch_mmvq_chunk(
        w_type,
        reinterpret_cast<const uint8_t*>(W),
        reinterpret_cast<const block_q8_1*>(X_q8),
        nullptr,
        Y,
        conv_w,
        add_res,
        conv_history,
        conv_offset,
        n_conv,
        N,
        K,
        1,
        nullptr,
        stream
    );
}

void launch_mmvq_q8_0_multi2(
    const void* w0, float* y0, int n0,
    const void* w1, float* y1, int n1,
    const void* x_q8,
    int k,
    int m,
    cudaStream_t stream
) {
    if (m <= 0 || m > 8 || (k & 31) != 0) return;
    const int n_tot = n0 + n1;
    if (n_tot <= 0) return;
    const auto* xq = reinterpret_cast<const block_q8_1*>(x_q8);
    const auto* b0 = reinterpret_cast<const block_q8_0*>(w0);
    const auto* b1 = reinterpret_cast<const block_q8_0*>(w1);
    dim3 grid((unsigned)((n_tot + 15) / 16));
    dim3 blk(32, 8);
    if (m == 1) {
        mmvq_q8_0_multi_batch_kernel<1, true, 2><<<grid, blk, 0, stream>>>(
            b0, y0, n0, b1, y1, n1, nullptr, nullptr, 0, xq, k, m);
    } else if (m <= 4) {
        mmvq_q8_0_multi_batch_kernel<4, false, 2><<<grid, blk, 0, stream>>>(
            b0, y0, n0, b1, y1, n1, nullptr, nullptr, 0, xq, k, m);
    } else if (m == 5) {
        mmvq_q8_0_multi_batch_kernel<5, true, 2><<<grid, blk, 0, stream>>>(
            b0, y0, n0, b1, y1, n1, nullptr, nullptr, 0, xq, k, m);
    } else {
        mmvq_q8_0_multi_batch_kernel<8, false, 2><<<grid, blk, 0, stream>>>(
            b0, y0, n0, b1, y1, n1, nullptr, nullptr, 0, xq, k, m);
    }
}

void launch_mmvq_q8_0_multi3(
    const void* w0, float* y0, int n0,
    const void* w1, float* y1, int n1,
    const void* w2, float* y2, int n2,
    const void* x_q8,
    int k,
    int m,
    cudaStream_t stream
) {
    if (m <= 0 || m > 8 || (k & 31) != 0) return;
    const int n_tot = n0 + n1 + n2;
    if (n_tot <= 0) return;
    const auto* xq = reinterpret_cast<const block_q8_1*>(x_q8);
    const auto* b0 = reinterpret_cast<const block_q8_0*>(w0);
    const auto* b1 = reinterpret_cast<const block_q8_0*>(w1);
    const auto* b2 = reinterpret_cast<const block_q8_0*>(w2);
    dim3 grid((unsigned)((n_tot + 15) / 16));
    dim3 blk(32, 8);
    if (m == 1) {
        mmvq_q8_0_multi_batch_kernel<1, true, 3><<<grid, blk, 0, stream>>>(
            b0, y0, n0, b1, y1, n1, b2, y2, n2, xq, k, m);
    } else if (m <= 4) {
        mmvq_q8_0_multi_batch_kernel<4, false, 3><<<grid, blk, 0, stream>>>(
            b0, y0, n0, b1, y1, n1, b2, y2, n2, xq, k, m);
    } else if (m == 5) {
        mmvq_q8_0_multi_batch_kernel<5, true, 3><<<grid, blk, 0, stream>>>(
            b0, y0, n0, b1, y1, n1, b2, y2, n2, xq, k, m);
    } else {
        mmvq_q8_0_multi_batch_kernel<8, false, 3><<<grid, blk, 0, stream>>>(
            b0, y0, n0, b1, y1, n1, b2, y2, n2, xq, k, m);
    }
}

void launch_qwen4_sh_gu_swiglu_q8(
    const void* w_gate,
    const void* w_up,
    const void* x_q8,
    float* y_act,
    int n,
    int k,
    int m,
    cudaStream_t stream
) {
    if (n <= 0 || m <= 0 || m > 8 || (k & 31) != 0) return;
    const auto* wg = reinterpret_cast<const block_q8_0*>(w_gate);
    const auto* wu = reinterpret_cast<const block_q8_0*>(w_up);
    const auto* xq = reinterpret_cast<const block_q8_1*>(x_q8);
    dim3 blk(32, 8);
    if (m == 1) {
        qwen4_sh_gu_swiglu_kernel<1, true><<<(unsigned)n, blk, 0, stream>>>(wg, wu, xq, y_act, n, k, m);
    } else if (m <= 4) {
        qwen4_sh_gu_swiglu_kernel<4, false><<<(unsigned)n, blk, 0, stream>>>(wg, wu, xq, y_act, n, k, m);
    } else if (m == 5) {
        qwen4_sh_gu_swiglu_kernel<5, true><<<(unsigned)n, blk, 0, stream>>>(wg, wu, xq, y_act, n, k, m);
    } else {
        qwen4_sh_gu_swiglu_kernel<8, false><<<(unsigned)n, blk, 0, stream>>>(wg, wu, xq, y_act, n, k, m);
    }
}

void launch_mmvq_q8_0_indexed_m1(
    const void* w,
    const int32_t* subset_ids,
    const void* x_q8,
    float* y,
    int n_sub,
    int k,
    cudaStream_t stream
) {
    if (n_sub <= 0 || (k & 31) != 0) return;
    dim3 grid((unsigned)((n_sub + 7) / 8));
    dim3 blk(32, 8);
    mmvq_q8_0_indexed_m1_kernel<<<grid, blk, 0, stream>>>(
        reinterpret_cast<const block_q8_0*>(w),
        subset_ids,
        reinterpret_cast<const block_q8_1*>(x_q8),
        y,
        n_sub,
        k
    );
}

void launch_embed_lookup_rmsnorm(
    const void* table,
    int32_t table_type,
    const int32_t* token_ids,
    const float* norm_weight,
    float* out,
    int n_embd,
    int M,
    float eps,
    cudaStream_t stream
) {
    embed_lookup_rmsnorm_kernel<<<M, 256, (size_t)n_embd * sizeof(float), stream>>>(
        reinterpret_cast<const uint8_t*>(table),
        table_type,
        token_ids,
        norm_weight,
        out,
        n_embd,
        M,
        eps
    );
}

} // extern "C"
