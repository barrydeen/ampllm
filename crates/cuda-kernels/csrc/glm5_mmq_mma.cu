#include "common/sm86_common.cuh"
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdio>

// -----------------------------------------------------------------------------
// Zero-cuBLAS Custom Ampere (sm_86) INT8 Tensor-Core MMQ GEMM & SwiGLU-Q8
// C[M, N] (F32) = alpha * A_q8[M, K] * B_quant[N, K]^T + beta * C[M, N]
// Optional fused row gather (rows_in_opt) and weighted scatter-add (rows_out_opt, wts_out_opt)
// -----------------------------------------------------------------------------

__device__ __forceinline__ void mma16832_s8_zero(
    int (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "r"(0), "r"(0), "r"(0), "r"(0)
    );
}

__device__ __forceinline__ void mma16816_s8_zero(
    int (&d)[4],
    uint32_t a0, uint32_t a1,
    uint32_t b0
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0),
          "r"(0), "r"(0), "r"(0), "r"(0)
    );
}

__device__ __forceinline__ void ldmatrix_x4_mmq(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    const void* smem_ptr
) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(addr)
    );
}

__device__ __forceinline__ int swz128_b16(int row, int byte_col) {
    return byte_col ^ ((row & 7) << 4);
}

template <int W_TYPE>
__device__ __forceinline__ void stage_mmq_lut(uint32_t* s_lut, int tid, int threads) {
    if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
        const uint32_t* g = reinterpret_cast<const uint32_t*>(d_iq2s_grid);
        for (int i = tid; i < 2048; i += threads) s_lut[i] = g[i];
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_XS) {
        const uint32_t* g = reinterpret_cast<const uint32_t*>(d_iq2xs_grid);
        for (int i = tid; i < 1024; i += threads) s_lut[i] = g[i];
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_XXS) {
        const uint32_t* g = reinterpret_cast<const uint32_t*>(d_iq2xxs_grid);
        for (int i = tid; i < 512; i += threads) s_lut[i] = g[i];
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
        for (int i = tid; i < 256; i += threads) s_lut[i] = d_iq3xxs_grid[i];
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_S) {
        for (int i = tid; i < 512; i += threads) s_lut[i] = d_iq3s_grid[i];
    } else {
        if (tid == 0) s_lut[0] = 0u;
    }
}

template <int W_TYPE>
__device__ __forceinline__ void unpack_w_half16(
    const uint8_t* w_row,
    int k0,
    int k32,
    int half16,
    const uint32_t* s_lut,
    uint4& out16,
    float& sc_out,
    float& sc1_out,
    int16_t& is_out,
    float& blk_d_out,
    bool& blk_d_valid
) {
    is_out = 0;
    blk_d_valid = false;
    if constexpr (W_TYPE == GGML_TYPE_Q8_0) {
        const int blk_idx = k0 >> 5;
        const block_q8_0* blk = reinterpret_cast<const block_q8_0*>(w_row) + blk_idx;
        const uint8_t* qp = reinterpret_cast<const uint8_t*>(blk->qs + half16 * 16);
        out16 = make_uint4(
            (uint32_t)load_i32_align2(qp + 0),
            (uint32_t)load_i32_align2(qp + 4),
            (uint32_t)load_i32_align2(qp + 8),
            (uint32_t)load_i32_align2(qp + 12)
        );
        if (half16 == 0) sc_out = fp16_bits_to_f32(load_u16_align2(&blk->d));
    } else if constexpr (W_TYPE == GGML_TYPE_Q6_K) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_q6_K* blk = reinterpret_cast<const block_q6_K*>(w_row) + b;
        if (k32 == 0 && half16 == 0) {
            blk_d_out = fp16_bits_to_f32(load_u16_align2(&blk->d));
            blk_d_valid = true;
        }
        const int half = sub >> 2;
        const int q = sub & 3;
        const uint8_t* ql = blk->ql + 64 * half + 32 * (q & 1);
        const uint8_t* qh = blk->qh + 32 * half;
        const int ql_shift = 4 * (q >> 1);
        const int qh_shift = 2 * q;
        const int j0 = half16 * 4;
        const bool ql_aligned = ((reinterpret_cast<uintptr_t>(ql) & 3u) == 0u);
        const bool qh_aligned = ((reinterpret_cast<uintptr_t>(qh) & 3u) == 0u);
        uint32_t w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t raw_ql = ql_aligned
                ? (uint32_t)load_i32_align4(ql + 4 * (j0 + j))
                : (uint32_t)load_i32_align2(ql + 4 * (j0 + j));
            const uint32_t raw_qh = qh_aligned
                ? (uint32_t)load_i32_align4(qh + 4 * (j0 + j))
                : (uint32_t)load_i32_align2(qh + 4 * (j0 + j));
            const uint32_t lo4 = (raw_ql >> ql_shift) & 0x0F0F0F0Fu;
            const uint32_t hi2 = (raw_qh >> qh_shift) & 0x03030303u;
            w[j] = __vsub4(lo4 | (hi2 << 4), 0x20202020u);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
        is_out = (int16_t)blk->scales[2 * sub + half16];
    } else if constexpr (W_TYPE == GGML_TYPE_Q4_K) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_q4_K* blk = reinterpret_cast<const block_q4_K*>(w_row) + b;
        const uint8_t* qs = blk->qs + 32 * (sub >> 1);
        const int shift = 4 * (sub & 1);
        const int j0 = half16 * 4;
        uint32_t w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j] = ((uint32_t)load_i32_align4(qs + 4 * (j0 + j)) >> shift) & 0x0F0F0F0Fu;
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
        if (half16 == 0) {
            float2 dpair = fp16x2_bits_to_f32(&blk->d);
            uint8_t sc = 0, mn = 0;
            get_scale_min_k4_dev(sub, blk->scales, sc, mn);
            sc_out  = dpair.x * (float)sc;
            sc1_out = dpair.y * (float)mn;
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q5_K) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_q5_K* blk = reinterpret_cast<const block_q5_K*>(w_row) + b;
        const uint8_t* qs = blk->qs + 32 * (sub >> 1);
        const int shift = 4 * (sub & 1);
        const int j0 = half16 * 4;
        uint32_t w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4 = ((uint32_t)load_i32_align4(qs + 4 * (j0 + j)) >> shift) & 0x0F0F0F0Fu;
            const uint32_t raw_qh = (uint32_t)load_i32_align4(blk->qh + 4 * (j0 + j));
            w[j] = q4 | (((raw_qh >> sub) & 0x01010101u) << 4);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
        if (half16 == 0) {
            float2 dpair = fp16x2_bits_to_f32(&blk->d);
            uint8_t sc = 0, mn = 0;
            get_scale_min_k4_dev(sub, blk->scales, sc, mn);
            sc_out  = dpair.x * (float)sc;
            sc1_out = dpair.y * (float)mn;
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q5_1) {
        const int blk_idx = k0 >> 5;
        const block_q5_1* blk = reinterpret_cast<const block_q5_1*>(w_row) + blk_idx;
        const uint32_t qh = (uint32_t)load_i32_align2(blk->qh);
        const uint32_t qh_half = (half16 == 0) ? qh : (qh >> 16);
        const int qs_shift = half16 * 4;
        uint32_t w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4 = ((uint32_t)load_i32_align2(blk->qs + 4 * j) >> qs_shift) & 0x0F0F0F0Fu;
            w[j] = q4 | expand_q5_1_hi4(qh_half >> (4 * j));
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
        if (half16 == 0) {
            sc_out  =  fp16_bits_to_f32(load_u16_align2(&blk->d));
            sc1_out = -fp16_bits_to_f32(load_u16_align2(&blk->m));
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q5_0) {
        const int blk_idx = k0 >> 5;
        const block_q5_0* blk = reinterpret_cast<const block_q5_0*>(w_row) + blk_idx;
        const uint32_t qh = (uint32_t)load_i32_align2(blk->qh);
        const uint32_t qh_half = (half16 == 0) ? qh : (qh >> 16);
        const int qs_shift = half16 * 4;
        uint32_t w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4 = ((uint32_t)load_i32_align2(blk->qs + 4 * j) >> qs_shift) & 0x0F0F0F0Fu;
            const uint32_t u5 = q4 | expand_q5_1_hi4(qh_half >> (4 * j));
            w[j] = __vsub4(u5, 0x10101010u);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
        if (half16 == 0) {
            sc_out = fp16_bits_to_f32(load_u16_align2(&blk->d));
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ4_NL) {
        const int blk_idx = k0 >> 5;
        const block_iq4_nl* blk = reinterpret_cast<const block_iq4_nl*>(w_row) + blk_idx;
        const uint32_t shift = (uint32_t)(half16 * 4);
        uint32_t w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4 = ((uint32_t)load_i32_align2(blk->qs + 4 * j) >> shift) & 0x0F0F0F0Fu;
            w[j] = (uint32_t)lut_iq4nl_x4(q4);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
        if (half16 == 0) {
            sc_out = fp16_bits_to_f32(load_u16_align2(&blk->d));
        }
    } else if constexpr (W_TYPE == GGML_TYPE_Q2_0) {
        const int blk_idx = k0 >> 6;
        const int sub32 = (k0 >> 5) & 1;
        const block_q2_0* blk = reinterpret_cast<const block_q2_0*>(w_row) + blk_idx;
        const uint8_t* qp = blk->qs + sub32 * 8 + half16 * 4;
        const uint32_t raw4 = (uint32_t)load_i32_align2(qp);
        uint32_t w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j] = (uint32_t)unpack_q2_0_x4((raw4 >> (8 * j)) & 0xFFu);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
        if (half16 == 0) {
            sc_out = fp16_bits_to_f32(load_u16_align2(&blk->d));
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ4_XS) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_iq4_xs* blk = reinterpret_cast<const block_iq4_xs*>(w_row) + b;
        const uint8_t* qs = blk->qs + 16 * sub;
        uint32_t w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t q4 = (uint32_t)load_i32_align2(qs + 4 * j);
            const uint32_t nibs = (half16 == 0) ? (q4 & 0x0F0F0F0Fu) : ((q4 >> 4) & 0x0F0F0F0Fu);
            w[j] = (uint32_t)lut_iq4nl_x4(nibs);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
        if (half16 == 0) {
            const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
            const uint16_t sh = load_u16_align2(&blk->scales_h);
            const int ls = ((blk->scales_l[sub >> 1] >> (4 * (sub & 1))) & 0xF)
                         | (((sh >> (2 * sub)) & 3) << 4);
            sc_out = dw * (float)(ls - 32);
        }
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_XS) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_iq2_xs* blk = reinterpret_cast<const block_iq2_xs*>(w_row) + b;
        if (k32 == 0 && half16 == 0) {
            blk_d_out = fp16_bits_to_f32(load_u16_align2(&blk->d)) * 0.125f;
            blk_d_valid = true;
        }
        const uint8_t sc_byte = blk->scales[sub];
        const int ls = (half16 == 0) ? (int)(sc_byte & 0x0Fu) : (int)(sc_byte >> 4);
        is_out = (int16_t)(2 * ls + 1);
        const uint64_t* lut = reinterpret_cast<const uint64_t*>(s_lut);
        const uint32_t qw = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4 * half16);
        uint32_t w[4];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t q16 = (i == 0) ? (qw & 0xFFFFu) : (qw >> 16);
            const uint64_t g64 = lut[q16 & 511u];
            const uint32_t s7 = q16 >> 9;
            const uint32_t sbyte = s7 | ((uint32_t)(__popc(s7) & 1) << 7);
            const unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
            const unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
            w[2 * i + 0] = __vsub4((uint32_t)g64 ^ s0, s0);
            w[2 * i + 1] = __vsub4((uint32_t)(g64 >> 32) ^ s1, s1);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_XXS) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_iq2_xxs* blk = reinterpret_cast<const block_iq2_xxs*>(w_row) + b;
        const uint32_t aux0 = (uint32_t)load_i32_align2(blk->qs + 8 * sub);
        const uint32_t aux1 = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4);
        if (half16 == 0) {
            const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
            const int ls = (int)(aux1 >> 28);
            sc_out = dw * 0.125f * (float)(2 * ls + 1);
        }
        const uint64_t* lut = reinterpret_cast<const uint64_t*>(s_lut);
        uint32_t w[4];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int l = half16 * 2 + i;
            const uint32_t g_idx = (aux0 >> (8 * l)) & 0xFFu;
            const uint32_t s7 = (aux1 >> (7 * l)) & 0x7Fu;
            const uint32_t sbyte = s7 | ((uint32_t)(__popc(s7) & 1) << 7);
            const uint64_t g64 = lut[g_idx];
            const unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
            const unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
            w[2 * i + 0] = __vsub4((uint32_t)g64 ^ s0, s0);
            w[2 * i + 1] = __vsub4((uint32_t)(g64 >> 32) ^ s1, s1);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
    } else if constexpr (W_TYPE == GGML_TYPE_IQ1_S) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_iq1_s* blk = reinterpret_cast<const block_iq1_s*>(w_row) + b;
        const uint16_t qh = load_u16_align2(&blk->qh[sub]);
        const uint32_t qs4 = (uint32_t)load_i32_align2(blk->qs + 4 * sub);
        if (half16 == 0) {
            const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
            const float dl = dw * (float)(2 * (int)((qh >> 12) & 7u) + 1);
            sc_out = dl;
            sc1_out = (qh & 0x8000u) ? (0.125f * dl) : (-0.125f * dl);
        }
        uint32_t w[4];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int l = half16 * 2 + i;
            const uint32_t g_idx = ((qs4 >> (8 * l)) & 0xFFu) | ((((uint32_t)qh >> (3 * l)) & 7u) << 8);
            const uint64_t g64 = d_iq1s_grid[g_idx];
            w[2 * i + 0] = (uint32_t)g64;
            w[2 * i + 1] = (uint32_t)(g64 >> 32);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
    } else if constexpr (W_TYPE == GGML_TYPE_IQ2_S) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_iq2_s* blk = reinterpret_cast<const block_iq2_s*>(w_row) + b;
        if (k32 == 0 && half16 == 0) {
            blk_d_out = fp16_bits_to_f32(load_u16_align2(&blk->d)) * 0.125f;
            blk_d_valid = true;
        }
        const uint32_t qh = (uint32_t)blk->qh[sub];
        const uint8_t sc_byte = blk->scales[sub];
        const int ls = (half16 == 0) ? (int)(sc_byte & 0x0Fu) : (int)(sc_byte >> 4);
        is_out = (int16_t)(2 * ls + 1);
        const uint64_t* lut = reinterpret_cast<const uint64_t*>(s_lut);
        const uint16_t qs16 = load_u16_align2(blk->qs + 4 * sub + 2 * half16);
        const uint16_t sg16 = load_u16_align2(blk->qs + 32 + 4 * sub + 2 * half16);
        uint32_t w[4];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int l0 = half16 * 4 + i * 2;
            const uint32_t q_b = ((uint32_t)qs16 >> (8 * i)) & 0xFFu;
            const uint32_t sbyte = ((uint32_t)sg16 >> (8 * i)) & 0xFFu;
            const uint32_t grid_idx = q_b | ((qh << (8 - l0)) & 0x300u);
            const uint64_t g64 = lut[grid_idx];
            const uint32_t g_lo = (uint32_t)g64;
            const uint32_t g_hi = (uint32_t)(g64 >> 32);
            const unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
            const unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
            w[2 * i + 0] = __vsub4(g_lo ^ s0, s0);
            w[2 * i + 1] = __vsub4(g_hi ^ s1, s1);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_XXS) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_iq3_xxs* blk = reinterpret_cast<const block_iq3_xxs*>(w_row) + b;
        const uint32_t q4 = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4 * half16);
        const uint32_t aux32 = (uint32_t)load_i32_align2(blk->qs + 64 + 4 * sub);
        if (half16 == 0) {
            const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
            const int ls = (int)(aux32 >> 28);
            sc_out = dw * 0.25f * (float)(2 * ls + 1);
        }
        uint32_t w[4];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t g0 = s_lut[(q4 >> (16 * i + 0)) & 0xFFu];
            const uint32_t g1 = s_lut[(q4 >> (16 * i + 8)) & 0xFFu];
            const uint32_t s7 = (aux32 >> (7 * (half16 * 2 + i))) & 0x7Fu;
            const uint32_t ks = (s7 | ((uint32_t)(__popc(s7) & 1) << 7)) * 0x01010101u;
            const unsigned s0 = __vcmpne4(ks & 0x08040201u, 0u);
            const unsigned s1 = __vcmpne4(ks & 0x80402010u, 0u);
            w[2 * i + 0] = __vsub4(g0 ^ s0, s0);
            w[2 * i + 1] = __vsub4(g1 ^ s1, s1);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
    } else if constexpr (W_TYPE == GGML_TYPE_IQ3_S) {
        const int b = k0 >> 8;
        const int sub = (k0 >> 5) & 7;
        const block_iq3_s* blk = reinterpret_cast<const block_iq3_s*>(w_row) + b;
        const uint32_t q4 = (uint32_t)load_i32_align2(blk->qs + 8 * sub + 4 * half16);
        const uint32_t qh = (uint32_t)blk->qh[sub];
        const uint32_t sg16 = (uint32_t)load_u16_align2(blk->signs + 4 * sub + 2 * half16);
        if (half16 == 0) {
            const float dw = fp16_bits_to_f32(load_u16_align2(&blk->d));
            const int ls = 1 + 2 * (int)((blk->scales[sub >> 1] >> (4 * (sub & 1))) & 0x0Fu);
            sc_out = dw * (float)ls;
        }
        uint32_t w[4];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int l0 = half16 * 4 + i * 2;
            const uint32_t g0 = s_lut[((q4 >> (16 * i + 0)) & 0xFFu) | ((qh << (8 - l0)) & 0x100u)];
            const uint32_t g1 = s_lut[((q4 >> (16 * i + 8)) & 0xFFu) | ((qh << (7 - l0)) & 0x100u)];
            const uint32_t sbyte = (sg16 >> (8 * i)) & 0xFFu;
            const unsigned s0 = __vcmpne4(((sbyte & 0x03u) << 7) | ((sbyte & 0x0Cu) << 21), 0u);
            const unsigned s1 = __vcmpne4(((sbyte & 0x30u) << 3) | ((sbyte & 0xC0u) << 17), 0u);
            w[2 * i + 0] = __vsub4(g0 ^ s0, s0);
            w[2 * i + 1] = __vsub4(g1 ^ s1, s1);
        }
        out16 = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

template <int W_TYPE, int BM, int BN, int THREADS>
__device__ __forceinline__ void unpack_b_tile_k128(
    const uint8_t* __restrict__ B_quant,
    size_t b_row_bytes,
    int bn, int N, int kt128, int nb32,
    int tid,
    const uint32_t* s_lut,
    int8_t* __restrict__ sB,
    float (&sB_sc)[(W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_Q6_K) ? 1 : BN][4][2],
    float (&sB_d)[(W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_Q6_K) ? BN : 1],
    int16_t (&sB_is)[(W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_Q6_K) ? BN : 1][4][2]
) {
    constexpr bool INT_FOLD_SC = (W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_Q6_K);
    constexpr bool HAS_MIN = (W_TYPE == GGML_TYPE_Q4_K || W_TYPE == GGML_TYPE_Q5_K || W_TYPE == GGML_TYPE_Q5_1 || W_TYPE == GGML_TYPE_IQ1_S);
    #pragma unroll 1
    for (int idx = tid; idx < BN * 8; idx += THREADS) {
        const int r = idx >> 3;
        const int k32 = (idx >> 1) & 3;
        const int h16 = idx & 1;
        const int gn = bn + r;
        const int kt_idx = (kt128 << 2) + k32;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        float sc0 = 0.0f, sc1 = 0.0f;
        int16_t is_val = 0;
        float blk_d = 0.0f;
        bool blk_d_valid = false;
        if (gn < N && kt_idx < nb32) {
            const uint8_t* wr = B_quant + (size_t)gn * b_row_bytes;
            unpack_w_half16<W_TYPE>(wr, kt_idx << 5, k32, h16, s_lut, v, sc0, sc1, is_val, blk_d, blk_d_valid);
        }
        *reinterpret_cast<uint4*>(&sB[r * 128 + swz128_b16(r, k32 * 32 + h16 * 16)]) = v;
        if constexpr (INT_FOLD_SC) {
            if (k32 == 0 && h16 == 0) {
                sB_d[r] = blk_d_valid ? blk_d : 0.0f;
            }
            sB_is[r][k32][h16] = is_val;
        } else if constexpr (HAS_MIN) {
            if (h16 == 0) {
                sB_sc[r][k32][0] = sc0;
                sB_sc[r][k32][1] = sc1;
            }
        } else {
            if (h16 == 0) sB_sc[r][k32][0] = sc0;
        }
    }
}

template <int W_TYPE, int MI, int NJ, int BN>
__device__ __forceinline__ void mma_k32_b_pairs(
    const uint32_t (&a_frag)[MI][4],
    const float (&da0)[MI],
    const float (&da1)[MI],
    const float (&sa0)[MI],
    const float (&sa1)[MI],
    const int8_t* __restrict__ sB,
    const float (&sB_sc)[(W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_Q6_K) ? 1 : BN][4][2],
    const float (&sB_d)[(W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_Q6_K) ? BN : 1],
    const int16_t (&sB_is)[(W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_Q6_K) ? BN : 1][4][2],
    int wn, int lane, int col_pair, int k32, int k_byte,
    float (&acc)[MI][NJ][4]
) {
    constexpr bool INT_FOLD_SC = (W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_Q6_K);
    constexpr bool HAS_MIN = (W_TYPE == GGML_TYPE_Q4_K || W_TYPE == GGML_TYPE_Q5_K || W_TYPE == GGML_TYPE_Q5_1 || W_TYPE == GGML_TYPE_IQ1_S);
    #pragma unroll
    for (int np = 0; np < NJ / 2; ++np) {
        const int row_b = wn + np * 16 + ((lane >> 4) * 8) + (lane & 7);
        const int col_b = ((lane >> 3) & 1) * 16;
        uint32_t bf[2][2];
        ldmatrix_x4_mmq(
            bf[0][0], bf[0][1],
            bf[1][0], bf[1][1],
            &sB[row_b * 128 + swz128_b16(row_b, k_byte + col_b)]
        );

        #pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
            const int nj = np * 2 + nh;
            const int c0 = wn + nj * 8 + col_pair;
            if constexpr (INT_FOLD_SC) {
                const float bd0 = sB_d[c0 + 0];
                const float bd1 = sB_d[c0 + 1];
                const int is0_x = (int)sB_is[c0 + 0][k32][0];
                const int is0_y = (int)sB_is[c0 + 0][k32][1];
                const int is1_x = (int)sB_is[c0 + 1][k32][0];
                const int is1_y = (int)sB_is[c0 + 1][k32][1];
                #pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    int d0[4], d1[4];
                    mma16816_s8_zero(d0, a_frag[mi][0], a_frag[mi][1], bf[nh][0]);
                    mma16816_s8_zero(d1, a_frag[mi][2], a_frag[mi][3], bf[nh][1]);
                    acc[mi][nj][0] += (da0[mi] * bd0) * (float)(is0_x * d0[0] + is0_y * d1[0]);
                    acc[mi][nj][1] += (da0[mi] * bd1) * (float)(is1_x * d0[1] + is1_y * d1[1]);
                    acc[mi][nj][2] += (da1[mi] * bd0) * (float)(is0_x * d0[2] + is0_y * d1[2]);
                    acc[mi][nj][3] += (da1[mi] * bd1) * (float)(is1_x * d0[3] + is1_y * d1[3]);
                }
            } else {
                const float sb0_x = sB_sc[c0 + 0][k32][0];
                const float sb1_x = sB_sc[c0 + 1][k32][0];
                float sb0_y = 0.0f, sb1_y = 0.0f;
                if constexpr (HAS_MIN) {
                    sb0_y = sB_sc[c0 + 0][k32][1];
                    sb1_y = sB_sc[c0 + 1][k32][1];
                }
                #pragma unroll
                for (int mi = 0; mi < MI; ++mi) {
                    if constexpr (HAS_MIN) {
                        int d[4];
                        mma16832_s8_zero(d, a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3], bf[nh][0], bf[nh][1]);
                        acc[mi][nj][0] += da0[mi] * sb0_x * (float)d[0] - sa0[mi] * sb0_y;
                        acc[mi][nj][1] += da0[mi] * sb1_x * (float)d[1] - sa0[mi] * sb1_y;
                        acc[mi][nj][2] += da1[mi] * sb0_x * (float)d[2] - sa1[mi] * sb0_y;
                        acc[mi][nj][3] += da1[mi] * sb1_x * (float)d[3] - sa1[mi] * sb1_y;
                    } else {
                        int d[4];
                        mma16832_s8_zero(d, a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3], bf[nh][0], bf[nh][1]);
                        acc[mi][nj][0] += (da0[mi] * sb0_x) * (float)d[0];
                        acc[mi][nj][1] += (da0[mi] * sb1_x) * (float)d[1];
                        acc[mi][nj][2] += (da1[mi] * sb0_x) * (float)d[2];
                        acc[mi][nj][3] += (da1[mi] * sb1_x) * (float)d[3];
                    }
                }
            }
        }
    }
}

template <int W_TYPE, int BM, int BN, int WM_WARPS, int WN_WARPS, int THREADS>
__device__ __forceinline__ void mmq_mma_accum_loop(
    const block_q8_1* __restrict__ A_q8,
    const uint8_t* __restrict__ B_quant,
    size_t b_row_bytes,
    const int32_t* __restrict__ rows_in_opt,
    int row_base_in,
    int M, int N, int K,
    int bm, int bn,
    int tid, int warp, int lane, int wm, int wn, int gid, int col_pair,
    const uint32_t* s_lut,
    float (&acc)[(BM / WM_WARPS) / 16][(BN / WN_WARPS) / 8][4]
) {
    constexpr int MI = (BM / WM_WARPS) / 16;
    constexpr int NJ = (BN / WN_WARPS) / 8;
    constexpr bool INT_FOLD_SC = (W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_Q6_K);
    constexpr bool HAS_MIN = (W_TYPE == GGML_TYPE_Q4_K || W_TYPE == GGML_TYPE_Q5_K || W_TYPE == GGML_TYPE_Q5_1 || W_TYPE == GGML_TYPE_IQ1_S);

    __shared__ __align__(16) int8_t sA[BM * 128];
    __shared__ __align__(16) int8_t sB[BN * 128];
    __shared__ float sA_d[BM][4];
    __shared__ float sA_s[HAS_MIN ? BM : 1][4];
    __shared__ float sB_sc[INT_FOLD_SC ? 1 : BN][4][2];
    __shared__ float sB_d[INT_FOLD_SC ? BN : 1];
    __shared__ int16_t sB_is[INT_FOLD_SC ? BN : 1][4][2];

    const int nb32 = K >> 5;
    if (nb32 <= 0) return;

    const int n_k128 = (nb32 + 3) >> 2;
    for (int kt128 = 0; kt128 < n_k128; ++kt128) {
        #pragma unroll 1
        for (int idx = tid; idx < BM * 8; idx += THREADS) {
            const int r = idx >> 3;
            const int k32 = (idx >> 1) & 3;
            const int h16 = idx & 1;
            const int gr = bm + r;
            const int kt_idx = (kt128 << 2) + k32;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (gr < M && kt_idx < nb32) {
                const int src_r = rows_in_opt ? rows_in_opt[row_base_in + gr] : (row_base_in + gr);
                const block_q8_1* xb = A_q8 + (size_t)src_r * nb32 + kt_idx;
                v = make_uint4(
                    (uint32_t)load_i32_align4(xb->qs + h16 * 16 + 0),
                    (uint32_t)load_i32_align4(xb->qs + h16 * 16 + 4),
                    (uint32_t)load_i32_align4(xb->qs + h16 * 16 + 8),
                    (uint32_t)load_i32_align4(xb->qs + h16 * 16 + 12)
                );
                if (h16 == 0) {
                    const float2 ds = fp16x2_bits_to_f32(&xb->d);
                    sA_d[r][k32] = ds.x;
                    if constexpr (HAS_MIN) {
                        sA_s[r][k32] = ds.y;
                    }
                }
            } else if (h16 == 0) {
                sA_d[r][k32] = 0.0f;
                if constexpr (HAS_MIN) {
                    sA_s[r][k32] = 0.0f;
                }
            }
            *reinterpret_cast<uint4*>(&sA[r * 128 + swz128_b16(r, k32 * 32 + h16 * 16)]) = v;
        }
        unpack_b_tile_k128<W_TYPE, BM, BN, THREADS>(
            B_quant, b_row_bytes, bn, N, kt128, nb32, tid, s_lut,
            sB, sB_sc, sB_d, sB_is
        );
        __syncthreads();

        #pragma unroll 1
        for (int k32 = 0; k32 < 4; ++k32) {
            const int k_byte = k32 * 32;

            uint32_t a_frag[MI][4];
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const int row_a = wm + mi * 16 + (lane & 15);
                const int col_a = (lane >> 4) * 16;
                ldmatrix_x4_mmq(
                    a_frag[mi][0], a_frag[mi][1], a_frag[mi][2], a_frag[mi][3],
                    &sA[row_a * 128 + swz128_b16(row_a, k_byte + col_a)]
                );
            }

            float da0[MI], da1[MI], sa0[MI], sa1[MI];
            #pragma unroll
            for (int mi = 0; mi < MI; ++mi) {
                const int r0 = wm + mi * 16 + gid;
                const int r1 = r0 + 8;
                da0[mi] = sA_d[r0][k32];
                da1[mi] = sA_d[r1][k32];
                if constexpr (HAS_MIN) {
                    sa0[mi] = sA_s[r0][k32];
                    sa1[mi] = sA_s[r1][k32];
                } else {
                    sa0[mi] = 0.0f;
                    sa1[mi] = 0.0f;
                }
            }

            mma_k32_b_pairs<W_TYPE, MI, NJ, BN>(
                a_frag, da0, da1, sa0, sa1,
                sB, sB_sc, sB_d, sB_is,
                wn, lane, col_pair, k32, k_byte, acc
            );
        }

        if (kt128 + 1 < n_k128) {
            __syncthreads();
        }
    }
}

template <int W_TYPE, int BM, int BN, int WM_WARPS, int WN_WARPS, int MIN_CTAS>
__global__ void __launch_bounds__(WM_WARPS * WN_WARPS * 32, MIN_CTAS)
gemm_mmq_mma_kernel(
    const block_q8_1* __restrict__ A_q8,
    const uint8_t* __restrict__ B_quant,
    size_t b_row_bytes,
    float* __restrict__ C,
    const int32_t* __restrict__ rows_in_opt,
    const int32_t* __restrict__ rows_out_opt,
    const float* __restrict__ wts_out_opt,
    int M, int N, int K,
    float alpha, float beta
) {
    constexpr int THREADS = WM_WARPS * WN_WARPS * 32;
    constexpr int MI = (BM / WM_WARPS) / 16;
    constexpr int NJ = (BN / WN_WARPS) / 8;
    constexpr int LUT_WORDS = (W_TYPE == GGML_TYPE_IQ2_S) ? 2048
        : ((W_TYPE == GGML_TYPE_IQ2_XS) ? 1024
        : ((W_TYPE == GGML_TYPE_IQ2_XXS || W_TYPE == GGML_TYPE_IQ3_S) ? 512
        : ((W_TYPE == GGML_TYPE_IQ3_XXS) ? 256 : 4)));

    __shared__ __align__(16) uint32_t s_lut[LUT_WORDS];

    const int bm = (int)blockIdx.y * BM;
    const int bn = (int)blockIdx.x * BN;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = (warp / WN_WARPS) * (MI * 16);
    const int wn = (warp % WN_WARPS) * (NJ * 8);

    stage_mmq_lut<W_TYPE>(s_lut, tid, THREADS);
    if constexpr (W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_IQ2_XXS ||
                  W_TYPE == GGML_TYPE_IQ3_S || W_TYPE == GGML_TYPE_IQ3_XXS) {
        __syncthreads();
    }

    float acc[MI][NJ][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int j = 0; j < NJ; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int gid = lane >> 2;
    const int col_pair = (lane & 3) * 2;
    mmq_mma_accum_loop<W_TYPE, BM, BN, WM_WARPS, WN_WARPS, THREADS>(
        A_q8, B_quant, b_row_bytes, rows_in_opt, 0, M, N, K, bm, bn,
        tid, warp, lane, wm, wn, gid, col_pair, s_lut, acc
    );

    const bool use_vec2 = (N & 1) == 0 && (reinterpret_cast<uintptr_t>(C) & 7u) == 0u;
    const bool scatter_add = (rows_out_opt != nullptr && wts_out_opt != nullptr);

    #pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
        const int r0 = bm + wm + mi * 16 + gid;
        const int r1 = r0 + 8;
        const int dr0 = (r0 < M) ? (scatter_add ? rows_out_opt[r0] : r0) : -1;
        const int dr1 = (r1 < M) ? (scatter_add ? rows_out_opt[r1] : r1) : -1;
        const float w0 = (r0 < M && scatter_add) ? (alpha * wts_out_opt[r0]) : alpha;
        const float w1 = (r1 < M && scatter_add) ? (alpha * wts_out_opt[r1]) : alpha;

        #pragma unroll
        for (int nj = 0; nj < NJ; ++nj) {
            const int c0 = bn + wn + nj * 8 + col_pair;
            const int c1 = c0 + 1;
            if (dr0 >= 0) {
                if (use_vec2 && c1 < N) {
                    float2 v;
                    if (scatter_add) {
                        const float2 old = *reinterpret_cast<const float2*>(&C[(size_t)dr0 * N + c0]);
                        v = make_float2(old.x + w0 * acc[mi][nj][0], old.y + w0 * acc[mi][nj][1]);
                    } else if (beta == 0.0f) {
                        v = make_float2(w0 * acc[mi][nj][0], w0 * acc[mi][nj][1]);
                    } else {
                        const float2 old = *reinterpret_cast<const float2*>(&C[(size_t)dr0 * N + c0]);
                        v = make_float2(w0 * acc[mi][nj][0] + beta * old.x, w0 * acc[mi][nj][1] + beta * old.y);
                    }
                    *reinterpret_cast<float2*>(&C[(size_t)dr0 * N + c0]) = v;
                } else {
                    if (c0 < N) {
                        C[(size_t)dr0 * N + c0] = scatter_add
                            ? (C[(size_t)dr0 * N + c0] + w0 * acc[mi][nj][0])
                            : ((beta == 0.0f) ? (w0 * acc[mi][nj][0]) : (w0 * acc[mi][nj][0] + beta * C[(size_t)dr0 * N + c0]));
                    }
                    if (c1 < N) {
                        C[(size_t)dr0 * N + c1] = scatter_add
                            ? (C[(size_t)dr0 * N + c1] + w0 * acc[mi][nj][1])
                            : ((beta == 0.0f) ? (w0 * acc[mi][nj][1]) : (w0 * acc[mi][nj][1] + beta * C[(size_t)dr0 * N + c1]));
                    }
                }
            }
            if (dr1 >= 0) {
                if (use_vec2 && c1 < N) {
                    float2 v;
                    if (scatter_add) {
                        const float2 old = *reinterpret_cast<const float2*>(&C[(size_t)dr1 * N + c0]);
                        v = make_float2(old.x + w1 * acc[mi][nj][2], old.y + w1 * acc[mi][nj][3]);
                    } else if (beta == 0.0f) {
                        v = make_float2(w1 * acc[mi][nj][2], w1 * acc[mi][nj][3]);
                    } else {
                        const float2 old = *reinterpret_cast<const float2*>(&C[(size_t)dr1 * N + c0]);
                        v = make_float2(w1 * acc[mi][nj][2] + beta * old.x, w1 * acc[mi][nj][3] + beta * old.y);
                    }
                    *reinterpret_cast<float2*>(&C[(size_t)dr1 * N + c0]) = v;
                } else {
                    if (c0 < N) {
                        C[(size_t)dr1 * N + c0] = scatter_add
                            ? (C[(size_t)dr1 * N + c0] + w1 * acc[mi][nj][2])
                            : ((beta == 0.0f) ? (w1 * acc[mi][nj][2]) : (w1 * acc[mi][nj][2] + beta * C[(size_t)dr1 * N + c0]));
                    }
                    if (c1 < N) {
                        C[(size_t)dr1 * N + c1] = scatter_add
                            ? (C[(size_t)dr1 * N + c1] + w1 * acc[mi][nj][3])
                            : ((beta == 0.0f) ? (w1 * acc[mi][nj][3]) : (w1 * acc[mi][nj][3] + beta * C[(size_t)dr1 * N + c1]));
                    }
                }
            }
        }
    }
}

template <int W_TYPE, int BM, int BN, int WM_WARPS, int WN_WARPS, int MIN_CTAS>
__global__ void __launch_bounds__(WM_WARPS * WN_WARPS * 32, MIN_CTAS)
gemm_mmq_mma_grouped_kernel(
    const block_q8_1* __restrict__ A_q8,
    const uint64_t* __restrict__ exp_ptrs,
    size_t b_byte_offset,
    size_t b_row_bytes,
    float* __restrict__ C,
    const int32_t* __restrict__ rows_in_opt,
    const int32_t* __restrict__ rows_out_opt,
    const float* __restrict__ wts_out_opt,
    const int32_t* __restrict__ exp_row_off,
    const int32_t* __restrict__ exp_nt,
    const int32_t* __restrict__ tile_exp_id,
    const int32_t* __restrict__ tile_m_off,
    int N, int K,
    float alpha,
    int use_atomic_scatter
) {
    constexpr int THREADS = WM_WARPS * WN_WARPS * 32;
    constexpr int MI = (BM / WM_WARPS) / 16;
    constexpr int NJ = (BN / WN_WARPS) / 8;
    constexpr int LUT_WORDS = (W_TYPE == GGML_TYPE_IQ2_S) ? 2048
        : ((W_TYPE == GGML_TYPE_IQ2_XS) ? 1024
        : ((W_TYPE == GGML_TYPE_IQ2_XXS || W_TYPE == GGML_TYPE_IQ3_S) ? 512
        : ((W_TYPE == GGML_TYPE_IQ3_XXS) ? 256 : 4)));

    __shared__ __align__(16) uint32_t s_lut[LUT_WORDS];

    const int e = tile_exp_id[blockIdx.y];
    const int bm = tile_m_off[blockIdx.y];
    const int M = exp_nt[e];
    const int row_base = exp_row_off[e];
    const uint8_t* B_quant = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(exp_ptrs[e])) + b_byte_offset;
    const int bn = (int)blockIdx.x * BN;
    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int wm = (warp / WN_WARPS) * (MI * 16);
    const int wn = (warp % WN_WARPS) * (NJ * 8);

    stage_mmq_lut<W_TYPE>(s_lut, tid, THREADS);
    if constexpr (W_TYPE == GGML_TYPE_IQ2_S || W_TYPE == GGML_TYPE_IQ2_XS || W_TYPE == GGML_TYPE_IQ2_XXS ||
                  W_TYPE == GGML_TYPE_IQ3_S || W_TYPE == GGML_TYPE_IQ3_XXS) {
        __syncthreads();
    }

    float acc[MI][NJ][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i)
        #pragma unroll
        for (int j = 0; j < NJ; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

    const int gid = lane >> 2;
    const int col_pair = (lane & 3) * 2;
    mmq_mma_accum_loop<W_TYPE, BM, BN, WM_WARPS, WN_WARPS, THREADS>(
        A_q8, B_quant, b_row_bytes, rows_in_opt, row_base, M, N, K, bm, bn,
        tid, warp, lane, wm, wn, gid, col_pair, s_lut, acc
    );

    const bool use_vec2 = (N & 1) == 0 && (reinterpret_cast<uintptr_t>(C) & 7u) == 0u;
    const bool scatter_add = (rows_out_opt != nullptr && wts_out_opt != nullptr);

    #pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
        const int r0 = bm + wm + mi * 16 + gid;
        const int r1 = r0 + 8;
        const int dr0 = (r0 < M) ? (scatter_add ? rows_out_opt[row_base + r0] : (row_base + r0)) : -1;
        const int dr1 = (r1 < M) ? (scatter_add ? rows_out_opt[row_base + r1] : (row_base + r1)) : -1;
        const float w0 = (r0 < M && scatter_add) ? (alpha * wts_out_opt[row_base + r0]) : alpha;
        const float w1 = (r1 < M && scatter_add) ? (alpha * wts_out_opt[row_base + r1]) : alpha;

        #pragma unroll
        for (int nj = 0; nj < NJ; ++nj) {
            const int c0 = bn + wn + nj * 8 + col_pair;
            const int c1 = c0 + 1;
            if (dr0 >= 0) {
                if (scatter_add && use_atomic_scatter != 0) {
                    if (c0 < N) atomicAdd(&C[(size_t)dr0 * N + c0], w0 * acc[mi][nj][0]);
                    if (c1 < N) atomicAdd(&C[(size_t)dr0 * N + c1], w0 * acc[mi][nj][1]);
                } else if (scatter_add) {
                    if (use_vec2 && c1 < N) {
                        const float2 old = *reinterpret_cast<const float2*>(&C[(size_t)dr0 * N + c0]);
                        *reinterpret_cast<float2*>(&C[(size_t)dr0 * N + c0]) =
                            make_float2(old.x + w0 * acc[mi][nj][0], old.y + w0 * acc[mi][nj][1]);
                    } else {
                        if (c0 < N) C[(size_t)dr0 * N + c0] += w0 * acc[mi][nj][0];
                        if (c1 < N) C[(size_t)dr0 * N + c1] += w0 * acc[mi][nj][1];
                    }
                } else if (use_vec2 && c1 < N) {
                    *reinterpret_cast<float2*>(&C[(size_t)dr0 * N + c0]) =
                        make_float2(w0 * acc[mi][nj][0], w0 * acc[mi][nj][1]);
                } else {
                    if (c0 < N) C[(size_t)dr0 * N + c0] = w0 * acc[mi][nj][0];
                    if (c1 < N) C[(size_t)dr0 * N + c1] = w0 * acc[mi][nj][1];
                }
            }
            if (dr1 >= 0) {
                if (scatter_add && use_atomic_scatter != 0) {
                    if (c0 < N) atomicAdd(&C[(size_t)dr1 * N + c0], w1 * acc[mi][nj][2]);
                    if (c1 < N) atomicAdd(&C[(size_t)dr1 * N + c1], w1 * acc[mi][nj][3]);
                } else if (scatter_add) {
                    if (use_vec2 && c1 < N) {
                        const float2 old = *reinterpret_cast<const float2*>(&C[(size_t)dr1 * N + c0]);
                        *reinterpret_cast<float2*>(&C[(size_t)dr1 * N + c0]) =
                            make_float2(old.x + w1 * acc[mi][nj][2], old.y + w1 * acc[mi][nj][3]);
                    } else {
                        if (c0 < N) C[(size_t)dr1 * N + c0] += w1 * acc[mi][nj][2];
                        if (c1 < N) C[(size_t)dr1 * N + c1] += w1 * acc[mi][nj][3];
                    }
                } else if (use_vec2 && c1 < N) {
                    *reinterpret_cast<float2*>(&C[(size_t)dr1 * N + c0]) =
                        make_float2(w1 * acc[mi][nj][2], w1 * acc[mi][nj][3]);
                } else {
                    if (c0 < N) C[(size_t)dr1 * N + c0] = w1 * acc[mi][nj][2];
                    if (c1 < N) C[(size_t)dr1 * N + c1] = w1 * acc[mi][nj][3];
                }
            }
        }
    }
}

template <typename KernelFn>
static inline void ensure_max_smem_carveout(KernelFn fn, std::atomic<uint32_t>& dev_mask) {
    int dev = 0;
    cudaGetDevice(&dev);
    const uint32_t bit = 1u << (dev & 31);
    if ((dev_mask.load(std::memory_order_acquire) & bit) == 0u) {
        cudaError_t err = cudaFuncSetAttribute(
            fn,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared
        );
        if (err != cudaSuccess) {
            static std::atomic<int> s_attr_err{0};
            cudaError_t clr = cudaGetLastError();
            if (s_attr_err.fetch_add(1, std::memory_order_relaxed) < 16) {
                std::fprintf(stderr, "[CUDA ERROR] cudaFuncSetAttribute carveout dev=%d err=%d (%s)\n",
                             dev, (int)err, cudaGetErrorString(clr != cudaSuccess ? clr : err));
            }
        } else {
            dev_mask.fetch_or(bit, std::memory_order_release);
        }
    }
}

template <int W_TYPE>
static void dispatch_mmq_tile(
    const block_q8_1* A_q8,
    const uint8_t* B_quant,
    size_t b_row_bytes,
    float* C,
    const int32_t* rows_in_opt,
    const int32_t* rows_out_opt,
    const float* wts_out_opt,
    int M, int N, int K,
    float alpha, float beta,
    int force_tile,
    cudaStream_t stream
) {
    static std::atomic<uint32_t> s_carveout_128x128_mask{0u};
    static std::atomic<uint32_t> s_carveout_mask{0u};
    const int g128x128 = ((N + 127) / 128) * ((M + 127) / 128);
    const int g128 = ((N + 63) / 64) * ((M + 127) / 128);
    const int g64  = ((N + 63) / 64) * ((M + 63) / 64);
    const bool use_128x128 = (force_tile == 4)
        || (force_tile == 0 && W_TYPE == GGML_TYPE_Q6_K && M >= 512 && N >= 128 && g128x128 >= 82);
    const bool use_128 = (force_tile == 1)
        || (force_tile == 0 && !use_128x128 && M >= 192 && g128 >= 82);
    const bool use_64  = (force_tile == 2)
        || (force_tile == 0 && !use_128x128 && !use_128 && M > 32 && g64 >= 64);
    if (use_128x128) {
        ensure_max_smem_carveout(gemm_mmq_mma_kernel<W_TYPE, 128, 128, 4, 2, 1>, s_carveout_128x128_mask);
        dim3 grid((N + 127) / 128, (M + 127) / 128);
        gemm_mmq_mma_kernel<W_TYPE, 128, 128, 4, 2, 1><<<grid, 256, 0, stream>>>(
            A_q8, B_quant, b_row_bytes, C, rows_in_opt, rows_out_opt, wts_out_opt, M, N, K, alpha, beta
        );
    } else if (use_128) {
        ensure_max_smem_carveout(gemm_mmq_mma_kernel<W_TYPE, 128, 64, 4, 2, 1>, s_carveout_mask);
        dim3 grid((N + 63) / 64, (M + 127) / 128);
        gemm_mmq_mma_kernel<W_TYPE, 128, 64, 4, 2, 1><<<grid, 256, 0, stream>>>(
            A_q8, B_quant, b_row_bytes, C, rows_in_opt, rows_out_opt, wts_out_opt, M, N, K, alpha, beta
        );
    } else if (use_64) {
        dim3 grid((N + 63) / 64, (M + 63) / 64);
        gemm_mmq_mma_kernel<W_TYPE, 64, 64, 2, 2, 2><<<grid, 128, 0, stream>>>(
            A_q8, B_quant, b_row_bytes, C, rows_in_opt, rows_out_opt, wts_out_opt, M, N, K, alpha, beta
        );
    } else {
        dim3 grid((N + 63) / 64, (M + 31) / 32);
        gemm_mmq_mma_kernel<W_TYPE, 32, 64, 1, 4, 3><<<grid, 128, 0, stream>>>(
            A_q8, B_quant, b_row_bytes, C, rows_in_opt, rows_out_opt, wts_out_opt, M, N, K, alpha, beta
        );
    }
}

template <int W_TYPE>
static void dispatch_mmq_grouped(
    const block_q8_1* A_q8,
    const uint64_t* exp_ptrs,
    size_t b_byte_offset,
    size_t b_row_bytes,
    float* C,
    const int32_t* rows_in_opt,
    const int32_t* rows_out_opt,
    const float* wts_out_opt,
    const int32_t* exp_row_off,
    const int32_t* exp_nt,
    const int32_t* tile_exp_id,
    const int32_t* tile_m_off,
    int n_m_tiles,
    int N, int K,
    float alpha,
    int use_atomic_scatter,
    int tile_mode,
    cudaStream_t stream
) {
    static std::atomic<uint32_t> s_grp_carveout_mask{0u};
    dim3 grid((N + 63) / 64, n_m_tiles);
    if (tile_mode == 1) {
        ensure_max_smem_carveout(gemm_mmq_mma_grouped_kernel<W_TYPE, 128, 64, 4, 2, 1>, s_grp_carveout_mask);
        gemm_mmq_mma_grouped_kernel<W_TYPE, 128, 64, 4, 2, 1><<<grid, 256, 0, stream>>>(
            A_q8, exp_ptrs, b_byte_offset, b_row_bytes, C, rows_in_opt, rows_out_opt, wts_out_opt,
            exp_row_off, exp_nt, tile_exp_id, tile_m_off, N, K, alpha, use_atomic_scatter
        );
    } else if (tile_mode == 2) {
        gemm_mmq_mma_grouped_kernel<W_TYPE, 64, 64, 2, 2, 2><<<grid, 128, 0, stream>>>(
            A_q8, exp_ptrs, b_byte_offset, b_row_bytes, C, rows_in_opt, rows_out_opt, wts_out_opt,
            exp_row_off, exp_nt, tile_exp_id, tile_m_off, N, K, alpha, use_atomic_scatter
        );
    } else {
        gemm_mmq_mma_grouped_kernel<W_TYPE, 32, 64, 1, 4, 3><<<grid, 128, 0, stream>>>(
            A_q8, exp_ptrs, b_byte_offset, b_row_bytes, C, rows_in_opt, rows_out_opt, wts_out_opt,
            exp_row_off, exp_nt, tile_exp_id, tile_m_off, N, K, alpha, use_atomic_scatter
        );
    }
}

extern "C" void launch_gemm_mmq_mma_grouped(
    int32_t w_type,
    const void* a_q8,
    const uint64_t* exp_ptrs,
    size_t b_byte_offset,
    size_t b_row_bytes,
    float* c_f32,
    const int32_t* rows_in_opt,
    const int32_t* rows_out_opt,
    const float* wts_out_opt,
    const int32_t* exp_row_off,
    const int32_t* exp_nt,
    const int32_t* tile_exp_id,
    const int32_t* tile_m_off,
    int n_m_tiles,
    int n, int k,
    float alpha,
    int use_atomic_scatter,
    int tile_mode,
    cudaStream_t stream
) {
    if (n_m_tiles <= 0 || n <= 0 || k <= 0) return;
    const auto* a = reinterpret_cast<const block_q8_1*>(a_q8);
    switch (w_type) {
        case GGML_TYPE_IQ2_XXS:
            dispatch_mmq_grouped<GGML_TYPE_IQ2_XXS>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            dispatch_mmq_grouped<GGML_TYPE_IQ2_XS>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_IQ1_S:
            dispatch_mmq_grouped<GGML_TYPE_IQ1_S>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_IQ2_S:
            dispatch_mmq_grouped<GGML_TYPE_IQ2_S>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            dispatch_mmq_grouped<GGML_TYPE_IQ3_XXS>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_IQ3_S:
            dispatch_mmq_grouped<GGML_TYPE_IQ3_S>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_Q6_K:
            dispatch_mmq_grouped<GGML_TYPE_Q6_K>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_Q8_0:
            dispatch_mmq_grouped<GGML_TYPE_Q8_0>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_Q4_K:
            dispatch_mmq_grouped<GGML_TYPE_Q4_K>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_Q5_K:
            dispatch_mmq_grouped<GGML_TYPE_Q5_K>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            dispatch_mmq_grouped<GGML_TYPE_IQ4_XS>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_Q5_1:
            dispatch_mmq_grouped<GGML_TYPE_Q5_1>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_Q5_0:
            dispatch_mmq_grouped<GGML_TYPE_Q5_0>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            dispatch_mmq_grouped<GGML_TYPE_IQ4_NL>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        case GGML_TYPE_Q2_0:
            dispatch_mmq_grouped<GGML_TYPE_Q2_0>(a, exp_ptrs, b_byte_offset, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, exp_row_off, exp_nt, tile_exp_id, tile_m_off, n_m_tiles, n, k, alpha, use_atomic_scatter, tile_mode, stream);
            break;
        default: {
            static std::atomic<int> s_grp_unsupported{0};
            if (s_grp_unsupported.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::fprintf(stderr, "[CUDA ERROR] launch_gemm_mmq_mma_grouped: unsupported w_type=%d\n", w_type);
            }
            break;
        }
    }
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        static std::atomic<int> s_grp_err_cnt{0};
        if (s_grp_err_cnt.fetch_add(1, std::memory_order_relaxed) < 16) {
            int dev = -1;
            cudaGetDevice(&dev);
            std::fprintf(stderr, "[CUDA ERROR] launch_gemm_mmq_mma_grouped dev=%d w_type=%d n_m_tiles=%d n=%d k=%d err=%d (%s)\n",
                         dev, w_type, n_m_tiles, n, k, (int)err, cudaGetErrorString(err));
        }
    }
}

extern "C" void launch_gemm_mmq_mma_tiled(
    int32_t w_type,
    const void* a_q8,
    const void* b_quant,
    size_t b_row_bytes,
    float* c_f32,
    const int32_t* rows_in_opt,
    const int32_t* rows_out_opt,
    const float* wts_out_opt,
    int m, int n, int k,
    float alpha, float beta,
    int force_tile,
    cudaStream_t stream
) {
    if (m <= 0 || n <= 0 || k <= 0) return;
    const auto* a = reinterpret_cast<const block_q8_1*>(a_q8);
    const auto* b = reinterpret_cast<const uint8_t*>(b_quant);
    switch (w_type) {
        case GGML_TYPE_IQ2_XXS:
            dispatch_mmq_tile<GGML_TYPE_IQ2_XXS>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            dispatch_mmq_tile<GGML_TYPE_IQ2_XS>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_IQ1_S:
            dispatch_mmq_tile<GGML_TYPE_IQ1_S>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_IQ2_S:
            dispatch_mmq_tile<GGML_TYPE_IQ2_S>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            dispatch_mmq_tile<GGML_TYPE_IQ3_XXS>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_IQ3_S:
            dispatch_mmq_tile<GGML_TYPE_IQ3_S>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_Q6_K:
            dispatch_mmq_tile<GGML_TYPE_Q6_K>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_Q8_0:
            dispatch_mmq_tile<GGML_TYPE_Q8_0>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_Q4_K:
            dispatch_mmq_tile<GGML_TYPE_Q4_K>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_Q5_K:
            dispatch_mmq_tile<GGML_TYPE_Q5_K>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            dispatch_mmq_tile<GGML_TYPE_IQ4_XS>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_Q5_1:
            dispatch_mmq_tile<GGML_TYPE_Q5_1>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_Q5_0:
            dispatch_mmq_tile<GGML_TYPE_Q5_0>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            dispatch_mmq_tile<GGML_TYPE_IQ4_NL>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        case GGML_TYPE_Q2_0:
            dispatch_mmq_tile<GGML_TYPE_Q2_0>(a, b, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt, m, n, k, alpha, beta, force_tile, stream);
            break;
        default: {
            static std::atomic<int> s_tiled_unsupported{0};
            if (s_tiled_unsupported.fetch_add(1, std::memory_order_relaxed) < 8) {
                std::fprintf(stderr, "[CUDA ERROR] launch_gemm_mmq_mma_tiled: unsupported w_type=%d\n", w_type);
            }
            break;
        }
    }
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        static std::atomic<int> s_tiled_err_cnt{0};
        if (s_tiled_err_cnt.fetch_add(1, std::memory_order_relaxed) < 16) {
            int dev = -1;
            cudaGetDevice(&dev);
            std::fprintf(stderr, "[CUDA ERROR] launch_gemm_mmq_mma_tiled dev=%d w_type=%d m=%d n=%d k=%d force_tile=%d err=%d (%s)\n",
                         dev, w_type, m, n, k, force_tile, (int)err, cudaGetErrorString(err));
        }
    }
}

extern "C" void launch_gemm_mmq_mma(
    int32_t w_type,
    const void* a_q8,
    const void* b_quant,
    size_t b_row_bytes,
    float* c_f32,
    const int32_t* rows_in_opt,
    const int32_t* rows_out_opt,
    const float* wts_out_opt,
    int m, int n, int k,
    float alpha, float beta,
    cudaStream_t stream
) {
    launch_gemm_mmq_mma_tiled(
        w_type, a_q8, b_quant, b_row_bytes, c_f32, rows_in_opt, rows_out_opt, wts_out_opt,
        m, n, k, alpha, beta, 0, stream
    );
}

__global__ void __launch_bounds__(256) glm5_swiglu_clamp_interleaved_q8_kernel(
    const float* __restrict__ gu,
    float lim,
    int n_ff,
    block_q8_1* __restrict__ out_q8
) {
    const int blk = (int)blockIdx.x * 8 + threadIdx.y;
    const int t = (int)blockIdx.y;
    const int lane = threadIdx.x;
    const int n_blocks = n_ff >> 5;
    if (blk >= n_blocks) return;

    const int c = (blk << 5) + lane;
    const size_t base = (size_t)t * (size_t)(2 * n_ff) + c;
    const float h = glm_swiglu_clamp_val(gu[base], gu[base + n_ff], lim);
    warp_quantize_q8_1_store(h, out_q8 + (size_t)t * n_blocks + blk, lane);
}

extern "C" void launch_glm5_swiglu_clamp_interleaved_q8(
    const float* gu,
    float lim,
    int nt,
    int n_ff,
    void* out_q8,
    cudaStream_t stream
) {
    if (nt <= 0 || n_ff <= 0) return;
    const int n_blk = n_ff >> 5;
    dim3 grid((n_blk + 7) / 8, nt);
    dim3 block(32, 8);
    glm5_swiglu_clamp_interleaved_q8_kernel<<<grid, block, 0, stream>>>(
        gu, lim, n_ff, reinterpret_cast<block_q8_1*>(out_q8)
    );
}
