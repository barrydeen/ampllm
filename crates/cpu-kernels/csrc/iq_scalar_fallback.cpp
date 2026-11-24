#include "iq_tables.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

static constexpr int QK_K = 256;

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K size mismatch");

struct block_q8_1_g8 {
    float d[8];
    float s[8];
    int32_t bsum[8];
    int8_t qs[QK_K];
};
static_assert(sizeof(block_q8_1_g8) == 352, "block_q8_1_g8 size mismatch");

struct block_q5_0 {
    uint16_t d;
    uint8_t qh[4];
    uint8_t qs[16];
};
static_assert(sizeof(block_q5_0) == 22, "block_q5_0 size mismatch");

struct block_q5_1 {
    uint16_t d;
    uint16_t m;
    uint8_t qh[4];
    uint8_t qs[16];
};
static_assert(sizeof(block_q5_1) == 24, "block_q5_1 size mismatch");

struct block_q2_0 {
    uint16_t d;
    uint8_t qs[16];
};
static_assert(sizeof(block_q2_0) == 18, "block_q2_0 size mismatch");

struct block_iq4_nl {
    uint16_t d;
    uint8_t qs[16];
};
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl size mismatch");

struct block_q8_0 {
    uint16_t d;
    int8_t qs[32];
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size mismatch");

struct block_q3_K {
    uint8_t hmask[QK_K / 8];
    uint8_t qs[QK_K / 4];
    uint8_t scales[12];
    uint16_t d;
};
static_assert(sizeof(block_q3_K) == 110, "block_q3_K size mismatch");

struct block_q4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K size mismatch");

struct block_q5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_q5_K) == 176, "block_q5_K size mismatch");

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size mismatch");

struct block_iq4_xs {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[4];
    uint8_t qs[QK_K / 2];
};
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs size mismatch");

struct block_iq2_xxs {
    uint16_t d;
    uint8_t qs[QK_K / 4];
};
static_assert(sizeof(block_iq2_xxs) == 66, "block_iq2_xxs size mismatch");

struct block_iq2_xs {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_xs) == 74, "block_iq2_xs size mismatch");

struct block_iq3_xxs {
    uint16_t d;
    uint8_t qs[3 * QK_K / 8];
};
static_assert(sizeof(block_iq3_xxs) == 98, "block_iq3_xxs size mismatch");

struct block_iq1_s {
    uint16_t d;
    uint8_t qs[QK_K / 8];
    uint16_t qh[QK_K / 32];
};
static_assert(sizeof(block_iq1_s) == 50, "block_iq1_s size mismatch");

struct block_iq3_s {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[QK_K / 64];
};
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s size mismatch");

struct block_iq2_s {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_s) == 82, "block_iq2_s size mismatch");

constexpr int GGML_TYPE_F32 = 0;
constexpr int GGML_TYPE_F16 = 1;
constexpr int GGML_TYPE_Q4_0 = 2;
constexpr int GGML_TYPE_Q5_0 = 6;
constexpr int GGML_TYPE_Q5_1 = 7;
constexpr int GGML_TYPE_Q8_0 = 8;
constexpr int GGML_TYPE_Q3_K = 11;
constexpr int GGML_TYPE_Q4_K = 12;
constexpr int GGML_TYPE_Q5_K = 13;
constexpr int GGML_TYPE_Q6_K = 14;
constexpr int GGML_TYPE_IQ2_XXS = 16;
constexpr int GGML_TYPE_IQ2_XS = 17;
constexpr int GGML_TYPE_IQ3_XXS = 18;
constexpr int GGML_TYPE_IQ1_S = 19;
constexpr int GGML_TYPE_IQ4_NL = 20;
constexpr int GGML_TYPE_IQ3_S = 21;
constexpr int GGML_TYPE_IQ2_S = 22;
constexpr int GGML_TYPE_IQ4_XS = 23;
constexpr int GGML_TYPE_BF16 = 30;
constexpr int GGML_TYPE_Q2_0 = 42;

constexpr int8_t KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

inline float fp16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x03FFu;
    uint32_t f_bits;
    if (exp == 0) {
        if (mant == 0) {
            f_bits = sign;
        } else {
            uint32_t m = mant;
            uint32_t e = 127 - 14;
            while ((m & 0x0400u) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03FFu;
            f_bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f_bits = sign | 0x7F800000u | (mant << 13);
    } else {
        f_bits = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f_bits, 4);
    return out;
}

inline uint16_t f32_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = (uint16_t)((x >> 16) & 0x8000u);
    const uint32_t abs_x = x & 0x7FFFFFFFu;
    if (abs_x >= 0x7F800000u) {
        if (abs_x > 0x7F800000u) return (uint16_t)(sign | 0x7E00u | ((abs_x >> 13) & 0x03FFu));
        return (uint16_t)(sign | 0x7C00u);
    }
    if (abs_x >= 0x47800000u) {
        return (uint16_t)(sign | 0x7C00u);
    }
    if (abs_x < 0x38800000u) {
        if (abs_x < 0x33000000u) return sign;
        const uint32_t exp = abs_x >> 23;
        const uint32_t mant = 0x00800000u | (abs_x & 0x007FFFFFu);
        const uint32_t shift = 126u - exp;
        uint32_t half_m = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t half_way = 1u << (shift - 1u);
        if (rem > half_way || (rem == half_way && (half_m & 1u))) {
            ++half_m;
        }
        return (uint16_t)(sign | half_m);
    }
    const uint32_t exp = (abs_x >> 23) - 112u;
    uint32_t mant = abs_x & 0x007FFFFFu;
    const uint32_t round_bits = mant & 0x1FFFu;
    mant >>= 13;
    if (round_bits > 0x1000u || (round_bits == 0x1000u && (mant & 1u))) {
        ++mant;
        if (mant == 0x0400u) {
            mant = 0;
            return (uint16_t)(sign | ((exp + 1u) << 10));
        }
    }
    return (uint16_t)(sign | (exp << 10) | mant);
}

inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline float clamped_swiglu(float g, float u, float lim) {
    if (lim > 0.0f && std::isfinite(lim)) {
        g = std::min(g, lim);
        u = std::max(-lim, std::min(u, lim));
    }
    return (g / (1.0f + std::exp(-g))) * u;
}

inline float swoai(float g, float u, float alpha, float limit) {
    if (limit > 0.0f && std::isfinite(limit)) {
        g = std::fmin(g, limit);
        u = std::fmin(std::fmax(u, -limit), limit);
    }
    return (g / (1.0f + std::exp(-alpha * g))) * (u + 1.0f);
}

inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

float dot_row_q8k(int32_t ggml_type, const uint8_t* row, int32_t n_in, const block_q8_K* y) {
    switch (ggml_type) {
        case GGML_TYPE_Q8_0: {
            const int nb = n_in / 32;
            const auto* b = reinterpret_cast<const block_q8_0*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 3;
                const int sub = i & 7;
                const int8_t* xq = y[blk].qs + sub * 32;
                int32_t sumi = 0;
                for (int j = 0; j < 32; ++j) sumi += (int32_t)b[i].qs[j] * (int32_t)xq[j];
                acc += fp16_to_f32(b[i].d) * y[blk].d * (float)sumi;
            }
            return acc;
        }
        case GGML_TYPE_Q5_0: {
            const int nb = n_in / 32;
            const auto* b = reinterpret_cast<const block_q5_0*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 3;
                const int sub = i & 7;
                const int8_t* xq = y[blk].qs + sub * 32;
                const uint32_t qh = u32(b[i].qh);
                int32_t sumi = 0;
                for (int j = 0; j < 16; ++j) {
                    const int w0 = (int)((b[i].qs[j] & 0x0F) | (((qh >> j) & 1u) << 4)) - 16;
                    const int w1 = (int)((b[i].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                    sumi += w0 * (int32_t)xq[j] + w1 * (int32_t)xq[j + 16];
                }
                acc += fp16_to_f32(b[i].d) * y[blk].d * (float)sumi;
            }
            return acc;
        }
        case GGML_TYPE_Q5_1: {
            const int nb = n_in / 32;
            const auto* b = reinterpret_cast<const block_q5_1*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 3;
                const int sub = i & 7;
                const int8_t* xq = y[blk].qs + sub * 32;
                const uint32_t qh = u32(b[i].qh);
                int32_t sumi = 0;
                for (int j = 0; j < 16; ++j) {
                    const int w0 = (int)((b[i].qs[j] & 0x0F) | (((qh >> j) & 1u) << 4));
                    const int w1 = (int)((b[i].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4));
                    sumi += w0 * (int32_t)xq[j] + w1 * (int32_t)xq[j + 16];
                }
                const int32_t bsum = (int32_t)y[blk].bsums[2 * sub] + (int32_t)y[blk].bsums[2 * sub + 1];
                acc += fp16_to_f32(b[i].d) * y[blk].d * (float)sumi + fp16_to_f32(b[i].m) * y[blk].d * (float)bsum;
            }
            return acc;
        }
        case GGML_TYPE_IQ4_NL: {
            const int nb = n_in / 32;
            const auto* b = reinterpret_cast<const block_iq4_nl*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 3;
                const int sub = i & 7;
                const int8_t* xq = y[blk].qs + sub * 32;
                int32_t sumi = 0;
                for (int j = 0; j < 16; ++j) {
                    sumi += (int32_t)KVALUES_IQ4NL[b[i].qs[j] & 0xF] * (int32_t)xq[j];
                    sumi += (int32_t)KVALUES_IQ4NL[b[i].qs[j] >> 4] * (int32_t)xq[j + 16];
                }
                acc += fp16_to_f32(b[i].d) * y[blk].d * (float)sumi;
            }
            return acc;
        }
        case GGML_TYPE_Q2_0: {
            const int nb = n_in / 64;
            const auto* b = reinterpret_cast<const block_q2_0*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 2;
                const int sub64 = i & 3;
                const int8_t* xq = y[blk].qs + sub64 * 64;
                int32_t sumi = 0;
                for (int j = 0; j < 16; ++j) {
                    const uint8_t q = b[i].qs[j];
                    for (int s = 0; s < 4; ++s) {
                        const int w = (int)((q >> (2 * s)) & 3u) - 1;
                        sumi += w * (int32_t)xq[4 * j + s];
                    }
                }
                acc += fp16_to_f32(b[i].d) * y[blk].d * (float)sumi;
            }
            return acc;
        }
        case GGML_TYPE_IQ4_XS: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_iq4_xs*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float dw = fp16_to_f32(b[i].d);
                const uint16_t sh = b[i].scales_h;
                int32_t isum = 0;
                for (int ib = 0; ib < 8; ++ib) {
                    const int ls = ((b[i].scales_l[ib / 2] >> (4 * (ib & 1))) & 0xF)
                                 | (((sh >> (2 * ib)) & 3) << 4);
                    const uint8_t* qs = b[i].qs + ib * 16;
                    const int8_t* xq = y[i].qs + ib * 32;
                    int32_t s = 0;
                    for (int j = 0; j < 16; ++j) {
                        s += (int32_t)KVALUES_IQ4NL[qs[j] & 0xF] * (int32_t)xq[j];
                        s += (int32_t)KVALUES_IQ4NL[qs[j] >> 4] * (int32_t)xq[j + 16];
                    }
                    isum += (ls - 32) * s;
                }
                acc += dw * y[i].d * (float)isum;
            }
            return acc;
        }
        case GGML_TYPE_Q4_K: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_q4_K*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float dw = fp16_to_f32(b[i].d) * y[i].d;
                const float dmin = fp16_to_f32(b[i].dmin) * y[i].d;
                int32_t sum_d = 0, sum_m = 0;
                for (int ib = 0; ib < 4; ++ib) {
                    uint8_t sc0, mn0, sc1, mn1;
                    get_scale_min_k4(ib * 2, b[i].scales, sc0, mn0);
                    get_scale_min_k4(ib * 2 + 1, b[i].scales, sc1, mn1);
                    const uint8_t* qs = b[i].qs + ib * 32;
                    const int8_t* x0 = y[i].qs + ib * 64;
                    const int8_t* x1 = x0 + 32;
                    int32_t s0 = 0, s1 = 0;
                    for (int l = 0; l < 32; ++l) {
                        s0 += (int32_t)(qs[l] & 0xF) * (int32_t)x0[l];
                        s1 += (int32_t)(qs[l] >> 4) * (int32_t)x1[l];
                    }
                    sum_d += (int32_t)sc0 * s0 + (int32_t)sc1 * s1;
                    const int32_t b0 = (int32_t)y[i].bsums[ib * 4 + 0] + (int32_t)y[i].bsums[ib * 4 + 1];
                    const int32_t b1 = (int32_t)y[i].bsums[ib * 4 + 2] + (int32_t)y[i].bsums[ib * 4 + 3];
                    sum_m += (int32_t)mn0 * b0 + (int32_t)mn1 * b1;
                }
                acc += dw * (float)sum_d - dmin * (float)sum_m;
            }
            return acc;
        }
        case GGML_TYPE_Q5_K: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_q5_K*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float dw = fp16_to_f32(b[i].d) * y[i].d;
                const float dmin = fp16_to_f32(b[i].dmin) * y[i].d;
                int32_t sum_d = 0, sum_m = 0;
                uint8_t u0 = 1, u1 = 2;
                for (int ib = 0; ib < 4; ++ib) {
                    uint8_t sc0, mn0, sc1, mn1;
                    get_scale_min_k4(ib * 2, b[i].scales, sc0, mn0);
                    get_scale_min_k4(ib * 2 + 1, b[i].scales, sc1, mn1);
                    const uint8_t* qs = b[i].qs + ib * 32;
                    const int8_t* x0 = y[i].qs + ib * 64;
                    const int8_t* x1 = x0 + 32;
                    int32_t s0 = 0, s1 = 0;
                    for (int l = 0; l < 32; ++l) {
                        const int w0 = (int)(qs[l] & 0xF) + ((b[i].qh[l] & u0) ? 16 : 0);
                        const int w1 = (int)(qs[l] >> 4) + ((b[i].qh[l] & u1) ? 16 : 0);
                        s0 += w0 * (int32_t)x0[l];
                        s1 += w1 * (int32_t)x1[l];
                    }
                    sum_d += (int32_t)sc0 * s0 + (int32_t)sc1 * s1;
                    const int32_t b0 = (int32_t)y[i].bsums[ib * 4 + 0] + (int32_t)y[i].bsums[ib * 4 + 1];
                    const int32_t b1 = (int32_t)y[i].bsums[ib * 4 + 2] + (int32_t)y[i].bsums[ib * 4 + 3];
                    sum_m += (int32_t)mn0 * b0 + (int32_t)mn1 * b1;
                    u0 <<= 2;
                    u1 <<= 2;
                }
                acc += dw * (float)sum_d - dmin * (float)sum_m;
            }
            return acc;
        }
        case GGML_TYPE_Q6_K: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_q6_K*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float dw = fp16_to_f32(b[i].d) * y[i].d;
                int32_t isum = 0;
                for (int half = 0; half < 2; ++half) {
                    const uint8_t* ql = b[i].ql + half * 64;
                    const uint8_t* qh = b[i].qh + half * 32;
                    const int8_t* sc = b[i].scales + half * 8;
                    const int8_t* xq = y[i].qs + half * 128;
                    for (int l = 0; l < 32; ++l) {
                        const int is = l >> 4;
                        const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        isum += (int32_t)sc[is + 0] * q1 * (int32_t)xq[l + 0];
                        isum += (int32_t)sc[is + 2] * q2 * (int32_t)xq[l + 32];
                        isum += (int32_t)sc[is + 4] * q3 * (int32_t)xq[l + 64];
                        isum += (int32_t)sc[is + 6] * q4 * (int32_t)xq[l + 96];
                    }
                }
                acc += dw * (float)isum;
            }
            return acc;
        }
        case GGML_TYPE_Q3_K: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_q3_K*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float d = fp16_to_f32(b[i].d) * y[i].d;
                uint32_t aux[4] = {};
                std::memcpy(aux, b[i].scales, 12);
                const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
                const uint32_t t0 = aux[0], t1 = aux[1], t2 = aux[2];
                aux[0] = (t0 & km2) | (((t2 >> 0) & km1) << 4);
                aux[1] = (t1 & km2) | (((t2 >> 2) & km1) << 4);
                aux[2] = ((t0 >> 4) & km2) | (((t2 >> 4) & km1) << 4);
                aux[3] = ((t1 >> 4) & km2) | (((t2 >> 6) & km1) << 4);
                const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
                int32_t isum = 0;
                int is = 0;
                uint8_t m = 1;
                const uint8_t* q = b[i].qs;
                for (int n = 0; n < QK_K; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j) {
                        const int32_t sc0 = (int32_t)scales[is++] - 32;
                        const int8_t* x0 = y[i].qs + n + j * 32;
                        int32_t s0 = 0;
                        for (int l = 0; l < 16; ++l) {
                            const int qv = (int)((q[l] >> shift) & 3) - ((b[i].hmask[l] & m) ? 0 : 4);
                            s0 += qv * (int32_t)x0[l];
                        }
                        const int32_t sc1 = (int32_t)scales[is++] - 32;
                        int32_t s1 = 0;
                        for (int l = 0; l < 16; ++l) {
                            const int qv = (int)((q[l + 16] >> shift) & 3) - ((b[i].hmask[l + 16] & m) ? 0 : 4);
                            s1 += qv * (int32_t)x0[l + 16];
                        }
                        isum += sc0 * s0 + sc1 * s1;
                        shift += 2;
                        m <<= 1;
                    }
                    q += 32;
                }
                acc += d * (float)isum;
            }
            return acc;
        }
        case GGML_TYPE_IQ2_XXS: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_iq2_xxs*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float d = fp16_to_f32(b[i].d) * y[i].d;
                int32_t isum = 0;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint32_t aux0 = u32(b[i].qs + 8 * ib32);
                    const uint32_t aux1 = u32(b[i].qs + 8 * ib32 + 4);
                    const int32_t sc = 2 * (int32_t)(aux1 >> 28) + 1;
                    const int8_t* xq = y[i].qs + ib32 * 32;
                    int32_t s = 0;
                    for (int l = 0; l < 4; ++l) {
                        const uint64_t g = iq2xxs_grid[(aux0 >> (8 * l)) & 0xFFu];
                        const uint8_t sg = ksigns_iq2xs[(aux1 >> (7 * l)) & 0x7Fu];
                        for (int k = 0; k < 8; ++k) {
                            const int8_t gv = (int8_t)((g >> (8 * k)) & 0xFFu);
                            const int32_t w = (sg & (1u << k)) ? -(int32_t)gv : (int32_t)gv;
                            s += w * (int32_t)xq[8 * l + k];
                        }
                    }
                    isum += sc * s;
                }
                acc += d * 0.125f * (float)isum;
            }
            return acc;
        }
        case GGML_TYPE_IQ2_XS: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_iq2_xs*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float d = fp16_to_f32(b[i].d) * y[i].d;
                int32_t isum = 0;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint8_t sb = b[i].scales[ib32];
                    const int32_t sc0 = 2 * (int32_t)(sb & 0xF) + 1;
                    const int32_t sc1 = 2 * (int32_t)(sb >> 4) + 1;
                    const int8_t* xq = y[i].qs + ib32 * 32;
                    int32_t s0 = 0, s1 = 0;
                    for (int l = 0; l < 4; ++l) {
                        const uint16_t q16 = u16(b[i].qs + 8 * ib32 + 2 * l);
                        const uint64_t g = iq2xs_grid[q16 & 511u];
                        const uint8_t sg = ksigns_iq2xs[q16 >> 9];
                        int32_t acc_g = 0;
                        for (int k = 0; k < 8; ++k) {
                            const int8_t gv = (int8_t)((g >> (8 * k)) & 0xFFu);
                            const int32_t w = (sg & (1u << k)) ? -(int32_t)gv : (int32_t)gv;
                            acc_g += w * (int32_t)xq[8 * l + k];
                        }
                        if (l < 2) s0 += acc_g;
                        else s1 += acc_g;
                    }
                    isum += sc0 * s0 + sc1 * s1;
                }
                acc += d * 0.125f * (float)isum;
            }
            return acc;
        }
        case GGML_TYPE_IQ1_S: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_iq1_s*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float d = fp16_to_f32(b[i].d) * y[i].d;
                float bsum_acc = 0.0f;
                for (int ib = 0; ib < 8; ++ib) {
                    const uint16_t qh = b[i].qh[ib];
                    const float dl = (float)(2 * ((qh >> 12) & 7) + 1);
                    const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
                    const int8_t* xq = y[i].qs + ib * 32;
                    int32_t s = 0;
                    for (int l = 0; l < 4; ++l) {
                        const uint64_t g = iq1s_grid[b[i].qs[ib * 4 + l] | (((qh >> (3 * l)) & 7u) << 8)];
                        for (int k = 0; k < 8; ++k) {
                            const int8_t w = (int8_t)((g >> (8 * k)) & 0xFFu);
                            s += (int32_t)w * (int32_t)xq[8 * l + k];
                        }
                    }
                    const int32_t bsum = (int32_t)y[i].bsums[2 * ib] + (int32_t)y[i].bsums[2 * ib + 1];
                    bsum_acc += dl * ((float)s + delta * (float)bsum);
                }
                acc += d * bsum_acc;
            }
            return acc;
        }
        case GGML_TYPE_IQ2_S: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_iq2_s*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float d = fp16_to_f32(b[i].d) * y[i].d;
                int32_t isum = 0;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint8_t sb = b[i].scales[ib32];
                    const int32_t sc0 = 2 * (int32_t)(sb & 0xF) + 1;
                    const int32_t sc1 = 2 * (int32_t)(sb >> 4) + 1;
                    const uint8_t qh = b[i].qh[ib32];
                    const uint8_t* qs = b[i].qs + 4 * ib32;
                    const uint8_t* signs = b[i].qs + 32 + 4 * ib32;
                    const int8_t* xq = y[i].qs + ib32 * 32;
                    int32_t s0 = 0, s1 = 0;
                    for (int l = 0; l < 4; ++l) {
                        const uint32_t gidx = qs[l] | (((uint32_t)qh << (8 - 2 * l)) & 0x300u);
                        const uint64_t g = iq2s_grid[gidx];
                        const uint8_t sg = signs[l];
                        int32_t acc_g = 0;
                        for (int k = 0; k < 8; ++k) {
                            const int8_t gv = (int8_t)((g >> (8 * k)) & 0xFFu);
                            const int32_t w = (sg & (1u << k)) ? -(int32_t)gv : (int32_t)gv;
                            acc_g += w * (int32_t)xq[8 * l + k];
                        }
                        if (l < 2) s0 += acc_g;
                        else s1 += acc_g;
                    }
                    isum += sc0 * s0 + sc1 * s1;
                }
                acc += d * 0.125f * (float)isum;
            }
            return acc;
        }
        case GGML_TYPE_IQ3_XXS: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_iq3_xxs*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float d = fp16_to_f32(b[i].d) * y[i].d;
                int32_t isum = 0;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint8_t* qs = b[i].qs + 8 * ib32;
                    const uint32_t aux = u32(b[i].qs + 64 + 4 * ib32);
                    const int32_t sc = 2 * (int32_t)(aux >> 28) + 1;
                    const int8_t* xq = y[i].qs + ib32 * 32;
                    int32_t s = 0;
                    for (int l = 0; l < 4; ++l) {
                        const uint32_t g0 = iq3xxs_grid[qs[2 * l + 0]];
                        const uint32_t g1 = iq3xxs_grid[qs[2 * l + 1]];
                        const uint8_t sg = ksigns_iq2xs[(aux >> (7 * l)) & 0x7Fu];
                        for (int k = 0; k < 4; ++k) {
                            const int8_t gv0 = (int8_t)((g0 >> (8 * k)) & 0xFFu);
                            const int8_t gv1 = (int8_t)((g1 >> (8 * k)) & 0xFFu);
                            const int32_t w0 = (sg & (1u << k)) ? -(int32_t)gv0 : (int32_t)gv0;
                            const int32_t w1 = (sg & (1u << (k + 4))) ? -(int32_t)gv1 : (int32_t)gv1;
                            s += w0 * (int32_t)xq[8 * l + k] + w1 * (int32_t)xq[8 * l + 4 + k];
                        }
                    }
                    isum += sc * s;
                }
                acc += d * 0.25f * (float)isum;
            }
            return acc;
        }
        case GGML_TYPE_IQ3_S: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_iq3_s*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float d = fp16_to_f32(b[i].d) * y[i].d;
                int32_t isum = 0;
                for (int ib32 = 0; ib32 < 8; ++ib32) {
                    const uint8_t sb = b[i].scales[ib32 >> 1];
                    const int32_t sc = 2 * (int32_t)((ib32 & 1) ? (sb >> 4) : (sb & 0xF)) + 1;
                    const uint8_t qh = b[i].qh[ib32];
                    const uint8_t* qs = b[i].qs + 8 * ib32;
                    const uint8_t* signs = b[i].signs + 4 * ib32;
                    const int8_t* xq = y[i].qs + ib32 * 32;
                    int32_t s = 0;
                    for (int l = 0; l < 4; ++l) {
                        const uint32_t g0 = iq3s_grid[qs[2 * l + 0] | (((uint32_t)qh << (8 - 2 * l)) & 0x100u)];
                        const uint32_t g1 = iq3s_grid[qs[2 * l + 1] | (((uint32_t)qh << (7 - 2 * l)) & 0x100u)];
                        const uint8_t sg = signs[l];
                        for (int k = 0; k < 4; ++k) {
                            const int8_t gv0 = (int8_t)((g0 >> (8 * k)) & 0xFFu);
                            const int8_t gv1 = (int8_t)((g1 >> (8 * k)) & 0xFFu);
                            const int32_t w0 = (sg & (1u << k)) ? -(int32_t)gv0 : (int32_t)gv0;
                            const int32_t w1 = (sg & (1u << (k + 4))) ? -(int32_t)gv1 : (int32_t)gv1;
                            s += w0 * (int32_t)xq[8 * l + k] + w1 * (int32_t)xq[8 * l + 4 + k];
                        }
                    }
                    isum += sc * s;
                }
                acc += d * (float)isum;
            }
            return acc;
        }
        default:
            return 0.0f;
    }
}

float dot_row_q8_1_g8(int32_t ggml_type, const uint8_t* row, int32_t n_in, const block_q8_1_g8* y) {
    switch (ggml_type) {
        case GGML_TYPE_Q8_0: {
            const int nb = n_in / 32;
            const auto* b = reinterpret_cast<const block_q8_0*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 3;
                const int sub = i & 7;
                const int8_t* xq = y[blk].qs + sub * 32;
                int32_t sumi = 0;
                for (int j = 0; j < 32; ++j) sumi += (int32_t)b[i].qs[j] * (int32_t)xq[j];
                acc += fp16_to_f32(b[i].d) * y[blk].d[sub] * (float)sumi;
            }
            return acc;
        }
        case GGML_TYPE_Q5_0: {
            const int nb = n_in / 32;
            const auto* b = reinterpret_cast<const block_q5_0*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 3;
                const int sub = i & 7;
                const int8_t* xq = y[blk].qs + sub * 32;
                const uint32_t qh = u32(b[i].qh);
                int32_t sumi = 0;
                for (int j = 0; j < 16; ++j) {
                    const int w0 = (int)((b[i].qs[j] & 0x0F) | (((qh >> j) & 1u) << 4)) - 16;
                    const int w1 = (int)((b[i].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                    sumi += w0 * (int32_t)xq[j] + w1 * (int32_t)xq[j + 16];
                }
                acc += fp16_to_f32(b[i].d) * y[blk].d[sub] * (float)sumi;
            }
            return acc;
        }
        case GGML_TYPE_Q5_1: {
            const int nb = n_in / 32;
            const auto* b = reinterpret_cast<const block_q5_1*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 3;
                const int sub = i & 7;
                const int8_t* xq = y[blk].qs + sub * 32;
                const uint32_t qh = u32(b[i].qh);
                int32_t sumi = 0;
                for (int j = 0; j < 16; ++j) {
                    const int w0 = (int)((b[i].qs[j] & 0x0F) | (((qh >> j) & 1u) << 4));
                    const int w1 = (int)((b[i].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4));
                    sumi += w0 * (int32_t)xq[j] + w1 * (int32_t)xq[j + 16];
                }
                acc += fp16_to_f32(b[i].d) * y[blk].d[sub] * (float)sumi + fp16_to_f32(b[i].m) * y[blk].s[sub];
            }
            return acc;
        }
        case GGML_TYPE_IQ4_NL: {
            const int nb = n_in / 32;
            const auto* b = reinterpret_cast<const block_iq4_nl*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 3;
                const int sub = i & 7;
                const int8_t* xq = y[blk].qs + sub * 32;
                int32_t sumi = 0;
                for (int j = 0; j < 16; ++j) {
                    sumi += (int32_t)KVALUES_IQ4NL[b[i].qs[j] & 0xF] * (int32_t)xq[j];
                    sumi += (int32_t)KVALUES_IQ4NL[b[i].qs[j] >> 4] * (int32_t)xq[j + 16];
                }
                acc += fp16_to_f32(b[i].d) * y[blk].d[sub] * (float)sumi;
            }
            return acc;
        }
        case GGML_TYPE_Q2_0: {
            const int nb = n_in / 64;
            const auto* b = reinterpret_cast<const block_q2_0*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const int blk = i >> 2;
                const int sub64 = i & 3;
                const float dw = fp16_to_f32(b[i].d);
                const int8_t* x0 = y[blk].qs + (2 * sub64 + 0) * 32;
                const int8_t* x1 = y[blk].qs + (2 * sub64 + 1) * 32;
                int32_t s0 = 0, s1 = 0;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t q0 = b[i].qs[j];
                    const uint8_t q1 = b[i].qs[j + 8];
                    for (int s = 0; s < 4; ++s) {
                        s0 += ((int)((q0 >> (2 * s)) & 3u) - 1) * (int32_t)x0[4 * j + s];
                        s1 += ((int)((q1 >> (2 * s)) & 3u) - 1) * (int32_t)x1[4 * j + s];
                    }
                }
                acc += dw * (y[blk].d[2 * sub64 + 0] * (float)s0 + y[blk].d[2 * sub64 + 1] * (float)s1);
            }
            return acc;
        }
        case GGML_TYPE_Q4_K: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_q4_K*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float dw = fp16_to_f32(b[i].d);
                const float dmin = fp16_to_f32(b[i].dmin);
                for (int ib = 0; ib < 4; ++ib) {
                    uint8_t sc0, mn0, sc1, mn1;
                    get_scale_min_k4(ib * 2, b[i].scales, sc0, mn0);
                    get_scale_min_k4(ib * 2 + 1, b[i].scales, sc1, mn1);
                    const uint8_t* qs = b[i].qs + ib * 32;
                    const int8_t* x0 = y[i].qs + (ib * 2) * 32;
                    const int8_t* x1 = y[i].qs + (ib * 2 + 1) * 32;
                    int32_t s0 = 0, s1 = 0;
                    for (int l = 0; l < 32; ++l) {
                        s0 += (int32_t)(qs[l] & 0xF) * (int32_t)x0[l];
                        s1 += (int32_t)(qs[l] >> 4) * (int32_t)x1[l];
                    }
                    acc += dw * ((float)sc0 * y[i].d[ib * 2] * (float)s0 + (float)sc1 * y[i].d[ib * 2 + 1] * (float)s1)
                         - dmin * ((float)mn0 * y[i].s[ib * 2] + (float)mn1 * y[i].s[ib * 2 + 1]);
                }
            }
            return acc;
        }
        case GGML_TYPE_Q5_K: {
            const int nb = n_in / QK_K;
            const auto* b = reinterpret_cast<const block_q5_K*>(row);
            float acc = 0.0f;
            for (int i = 0; i < nb; ++i) {
                const float dw = fp16_to_f32(b[i].d);
                const float dmin = fp16_to_f32(b[i].dmin);
                uint8_t u0 = 1, u1 = 2;
                for (int ib = 0; ib < 4; ++ib) {
                    uint8_t sc0, mn0, sc1, mn1;
                    get_scale_min_k4(ib * 2, b[i].scales, sc0, mn0);
                    get_scale_min_k4(ib * 2 + 1, b[i].scales, sc1, mn1);
                    const uint8_t* qs = b[i].qs + ib * 32;
                    const int8_t* x0 = y[i].qs + (ib * 2) * 32;
                    const int8_t* x1 = y[i].qs + (ib * 2 + 1) * 32;
                    int32_t s0 = 0, s1 = 0;
                    for (int l = 0; l < 32; ++l) {
                        const int w0 = (int)(qs[l] & 0xF) + ((b[i].qh[l] & u0) ? 16 : 0);
                        const int w1 = (int)(qs[l] >> 4) + ((b[i].qh[l] & u1) ? 16 : 0);
                        s0 += w0 * (int32_t)x0[l];
                        s1 += w1 * (int32_t)x1[l];
                    }
                    acc += dw * ((float)sc0 * y[i].d[ib * 2] * (float)s0 + (float)sc1 * y[i].d[ib * 2 + 1] * (float)s1)
                         - dmin * ((float)mn0 * y[i].s[ib * 2] + (float)mn1 * y[i].s[ib * 2 + 1]);
                    u0 <<= 2;
                    u1 <<= 2;
                }
            }
            return acc;
        }
        default:
            return 0.0f;
    }
}

}  // namespace

extern "C" {

void ampllm_cpu_quantize_q8_k_scalar(const float* x, void* out_q8k, int32_t n) {
    if (n <= 0) return;
    const int nb = n / QK_K;
    const int rem = n - nb * QK_K;
    auto* y = reinterpret_cast<block_q8_K*>(out_q8k);
    for (int i = 0; i < nb; ++i) {
        const float* xp = x + (size_t)i * QK_K;
        float amax = 0.0f;
        for (int j = 0; j < QK_K; ++j) amax = std::fmax(amax, std::fabs(xp[j]));
        const float d = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        y[i].d = d;
        for (int c = 0; c < 16; ++c) {
            int16_t bsum = 0;
            for (int j = 0; j < 16; ++j) {
                const float qf = std::nearbyint(xp[c * 16 + j] * id);
                const int q = (int)std::fmin(127.0f, std::fmax(-127.0f, qf));
                y[i].qs[c * 16 + j] = (int8_t)q;
                bsum += (int16_t)q;
            }
            y[i].bsums[c] = bsum;
        }
    }
    if (rem > 0) {
        const float* xp = x + (size_t)nb * QK_K;
        float amax = 0.0f;
        for (int j = 0; j < rem; ++j) amax = std::fmax(amax, std::fabs(xp[j]));
        const float d = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        y[nb].d = d;
        std::memset(y[nb].qs, 0, QK_K);
        std::memset(y[nb].bsums, 0, sizeof(y[nb].bsums));
        for (int j = 0; j < rem; ++j) {
            const float qf = std::nearbyint(xp[j] * id);
            const int q = (int)std::fmin(127.0f, std::fmax(-127.0f, qf));
            y[nb].qs[j] = (int8_t)q;
            y[nb].bsums[j / 16] += (int16_t)q;
        }
    }
}

void ampllm_cpu_quantize_q8_1_g8_scalar(const float* x, void* out_g8, int32_t n) {
    if (n <= 0) return;
    const int nb = (n + QK_K - 1) / QK_K;
    auto* dst = reinterpret_cast<block_q8_1_g8*>(out_g8);
    for (int b = 0; b < nb; ++b) {
        block_q8_1_g8& out = dst[b];
        const int base_b = b * QK_K;
        for (int k = 0; k < 8; ++k) {
            const int off = base_b + k * 32;
            if (off >= n) {
                out.d[k] = 0.0f;
                out.s[k] = 0.0f;
                out.bsum[k] = 0;
                std::memset(out.qs + k * 32, 0, 32);
                continue;
            }
            const int rem = std::min(32, n - off);
            const float* p = x + off;
            float amax = 0.0f;
            for (int j = 0; j < rem; ++j) amax = std::fmax(amax, std::fabs(p[j]));
            const float d = amax / 127.0f;
            const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
            int32_t sumq = 0;
            for (int j = 0; j < rem; ++j) {
                const float qf = std::nearbyint(p[j] * id);
                const int q = (int)std::fmin(127.0f, std::fmax(-127.0f, qf));
                out.qs[k * 32 + j] = (int8_t)q;
                sumq += q;
            }
            for (int j = rem; j < 32; ++j) out.qs[k * 32 + j] = 0;
            out.d[k] = d;
            out.bsum[k] = sumq;
            out.s[k] = d * (float)sumq;
        }
    }
}

void ampllm_cpu_bf16_to_q8_0_scalar(
    const uint16_t* src_bf16,
    const float* src_f32,
    size_t rows,
    int32_t k,
    uint8_t* out_q8_0
) {
    if (k <= 0 || (k & 31) != 0) return;
    const int nb = k / 32;
    for (size_t r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const size_t at = r * (size_t)k + (size_t)b * 32;
            float vals[32];
            float am = 0.0f;
            for (int j = 0; j < 32; ++j) {
                vals[j] = (src_f32 != nullptr) ? src_f32[at + j] : bf16_to_f32(src_bf16[at + j]);
                am = std::fmax(am, std::fabs(vals[j]));
            }
            const float d = am / 127.0f;
            const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
            uint8_t* dst = out_q8_0 + (r * (size_t)nb + (size_t)b) * sizeof(block_q8_0);
            const uint16_t hd = f32_to_fp16(d);
            std::memcpy(dst, &hd, 2);
            for (int j = 0; j < 32; ++j) {
                const float qf = std::nearbyint(vals[j] * id);
                const int q = (int)std::fmin(127.0f, std::fmax(-127.0f, qf));
                dst[2 + j] = (uint8_t)(int8_t)q;
            }
        }
    }
}

void ampllm_cpu_tok_row_to_f32_scalar(
    int32_t ggml_type,
    const uint8_t* src,
    int32_t k,
    float* dst_f32
) {
    if (k <= 0) return;
    switch (ggml_type) {
        case GGML_TYPE_F32:
            std::memcpy(dst_f32, src, (size_t)k * 4);
            break;
        case GGML_TYPE_F16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            for (int i = 0; i < k; ++i) dst_f32[i] = fp16_to_f32(p[i]);
            break;
        }
        case GGML_TYPE_BF16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            for (int i = 0; i < k; ++i) dst_f32[i] = bf16_to_f32(p[i]);
            break;
        }
        case GGML_TYPE_Q8_0: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_q8_0*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = fp16_to_f32(blks[b].d);
                for (int j = 0; j < 32; ++j) dst_f32[b * 32 + j] = d * (float)blks[b].qs[j];
            }
            break;
        }
        case GGML_TYPE_Q4_0: {
            const int nb = k / 32;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* bp = src + (size_t)b * 18;
                const float d = fp16_to_f32(u16(bp));
                for (int j = 0; j < 16; ++j) {
                    dst_f32[b * 32 + j]      = d * (float)((int)(bp[2 + j] & 0xF) - 8);
                    dst_f32[b * 32 + j + 16] = d * (float)((int)(bp[2 + j] >> 4) - 8);
                }
            }
            break;
        }
        case GGML_TYPE_Q5_0: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_q5_0*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = fp16_to_f32(blks[b].d);
                const uint32_t qh = u32(blks[b].qh);
                for (int j = 0; j < 16; ++j) {
                    const int x0 = (int)((blks[b].qs[j] & 0xF) | (((qh >> j) & 1u) << 4)) - 16;
                    const int x1 = (int)((blks[b].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                    dst_f32[b * 32 + j]      = d * (float)x0;
                    dst_f32[b * 32 + j + 16] = d * (float)x1;
                }
            }
            break;
        }
        case GGML_TYPE_Q5_1: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_q5_1*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = fp16_to_f32(blks[b].d);
                const float m = fp16_to_f32(blks[b].m);
                const uint32_t qh = u32(blks[b].qh);
                for (int j = 0; j < 16; ++j) {
                    const uint32_t x0 = (blks[b].qs[j] & 0xF) | (((qh >> j) & 1u) << 4);
                    const uint32_t x1 = (blks[b].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4);
                    dst_f32[b * 32 + j]      = d * (float)x0 + m;
                    dst_f32[b * 32 + j + 16] = d * (float)x1 + m;
                }
            }
            break;
        }
        case GGML_TYPE_IQ4_NL: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_iq4_nl*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = fp16_to_f32(blks[b].d);
                for (int j = 0; j < 16; ++j) {
                    dst_f32[b * 32 + j]      = d * (float)KVALUES_IQ4NL[blks[b].qs[j] & 0xF];
                    dst_f32[b * 32 + j + 16] = d * (float)KVALUES_IQ4NL[blks[b].qs[j] >> 4];
                }
            }
            break;
        }
        case GGML_TYPE_Q2_0: {
            const int nb = k / 64;
            const auto* blks = reinterpret_cast<const block_q2_0*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = fp16_to_f32(blks[b].d);
                for (int j = 0; j < 16; ++j) {
                    const uint8_t q = blks[b].qs[j];
                    for (int s = 0; s < 4; ++s) {
                        dst_f32[b * 64 + 4 * j + s] = d * (float)((int)((q >> (2 * s)) & 3u) - 1);
                    }
                }
            }
            break;
        }
        case GGML_TYPE_Q3_K: {
            const int nb = k / QK_K;
            const auto* blks = reinterpret_cast<const block_q3_K*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d_all = fp16_to_f32(blks[b].d);
                uint32_t aux[4] = {};
                std::memcpy(aux, blks[b].scales, 12);
                const uint32_t km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
                const uint32_t t0 = aux[0], t1 = aux[1], t2 = aux[2];
                aux[0] = (t0 & km2) | (((t2 >> 0) & km1) << 4);
                aux[1] = (t1 & km2) | (((t2 >> 2) & km1) << 4);
                aux[2] = ((t0 >> 4) & km2) | (((t2 >> 4) & km1) << 4);
                aux[3] = ((t1 >> 4) & km2) | (((t2 >> 6) & km1) << 4);
                const int8_t* scales = reinterpret_cast<const int8_t*>(aux);
                float* y = dst_f32 + b * 256;
                int is = 0;
                uint8_t m = 1;
                const uint8_t* q = blks[b].qs;
                for (int n = 0; n < QK_K; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j) {
                        const float dl0 = d_all * (float)(scales[is++] - 32);
                        for (int l = 0; l < 16; ++l) {
                            y[n + j * 32 + l] = dl0 * (float)((int)((q[l] >> shift) & 3) - ((blks[b].hmask[l] & m) ? 0 : 4));
                        }
                        const float dl1 = d_all * (float)(scales[is++] - 32);
                        for (int l = 0; l < 16; ++l) {
                            y[n + j * 32 + 16 + l] = dl1 * (float)((int)((q[l + 16] >> shift) & 3) - ((blks[b].hmask[l + 16] & m) ? 0 : 4));
                        }
                        shift += 2;
                        m <<= 1;
                    }
                    q += 32;
                }
            }
            break;
        }
        case GGML_TYPE_Q4_K: {
            const int nb = k / QK_K;
            const auto* blks = reinterpret_cast<const block_q4_K*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = fp16_to_f32(blks[b].d);
                const float dm = fp16_to_f32(blks[b].dmin);
                for (int ib = 0; ib < 4; ++ib) {
                    uint8_t sc0, mn0, sc1, mn1;
                    get_scale_min_k4(ib * 2, blks[b].scales, sc0, mn0);
                    get_scale_min_k4(ib * 2 + 1, blks[b].scales, sc1, mn1);
                    const float d0 = d * (float)sc0, m0 = dm * (float)mn0;
                    const float d1 = d * (float)sc1, m1 = dm * (float)mn1;
                    const uint8_t* qs = blks[b].qs + ib * 32;
                    float* dst = dst_f32 + b * 256 + ib * 64;
                    for (int l = 0; l < 32; ++l) {
                        dst[l]      = d0 * (float)(qs[l] & 0xF) - m0;
                        dst[l + 32] = d1 * (float)(qs[l] >> 4)  - m1;
                    }
                }
            }
            break;
        }
        case GGML_TYPE_Q5_K: {
            const int nb = k / QK_K;
            const auto* blks = reinterpret_cast<const block_q5_K*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = fp16_to_f32(blks[b].d);
                const float dm = fp16_to_f32(blks[b].dmin);
                for (int ib = 0; ib < 4; ++ib) {
                    uint8_t sc0, mn0, sc1, mn1;
                    get_scale_min_k4(ib * 2, blks[b].scales, sc0, mn0);
                    get_scale_min_k4(ib * 2 + 1, blks[b].scales, sc1, mn1);
                    const float d0 = d * (float)sc0, m0 = dm * (float)mn0;
                    const float d1 = d * (float)sc1, m1 = dm * (float)mn1;
                    const uint8_t u0 = (uint8_t)(1u << (2 * ib));
                    const uint8_t u1 = (uint8_t)(2u << (2 * ib));
                    const uint8_t* qs = blks[b].qs + ib * 32;
                    float* dst = dst_f32 + b * 256 + ib * 64;
                    for (int l = 0; l < 32; ++l) {
                        dst[l]      = d0 * (float)((qs[l] & 0xF) + ((blks[b].qh[l] & u0) ? 16 : 0)) - m0;
                        dst[l + 32] = d1 * (float)((qs[l] >> 4)  + ((blks[b].qh[l] & u1) ? 16 : 0)) - m1;
                    }
                }
            }
            break;
        }
        case GGML_TYPE_Q6_K: {
            const int nb = k / QK_K;
            const auto* blks = reinterpret_cast<const block_q6_K*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = fp16_to_f32(blks[b].d);
                for (int half = 0; half < 2; ++half) {
                    const uint8_t* ql = blks[b].ql + half * 64;
                    const uint8_t* qh = blks[b].qh + half * 32;
                    const int8_t* sc = blks[b].scales + half * 8;
                    float* dst = dst_f32 + b * 256 + half * 128;
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int q1 = (int)((ql[l] & 0xF)      | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = (int)((ql[l] >> 4)       | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = (int)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                        dst[l +  0] = d * (float)sc[is + 0] * (float)q1;
                        dst[l + 32] = d * (float)sc[is + 2] * (float)q2;
                        dst[l + 64] = d * (float)sc[is + 4] * (float)q3;
                        dst[l + 96] = d * (float)sc[is + 6] * (float)q4;
                    }
                }
            }
            break;
        }
        case GGML_TYPE_IQ4_XS: {
            const int nb = k / QK_K;
            const auto* blks = reinterpret_cast<const block_iq4_xs*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = fp16_to_f32(blks[b].d);
                const uint16_t sh = blks[b].scales_h;
                for (int ib = 0; ib < 8; ++ib) {
                    const int ls = ((blks[b].scales_l[ib / 2] >> (4 * (ib & 1))) & 0xF)
                                 | (((sh >> (2 * ib)) & 3) << 4);
                    const float dl = d * (float)(ls - 32);
                    const uint8_t* qs = blks[b].qs + ib * 16;
                    float* dst = dst_f32 + b * 256 + ib * 32;
                    for (int j = 0; j < 16; ++j) {
                        dst[j]      = dl * (float)KVALUES_IQ4NL[qs[j] & 0xF];
                        dst[j + 16] = dl * (float)KVALUES_IQ4NL[qs[j] >> 4];
                    }
                }
            }
            break;
        }
        default:
            break;
    }
}

void ampllm_cpu_hy4_fused_gu_rows_scalar(
    int32_t ggml_type,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t row_bytes,
    int32_t n_in,
    const void* const* x_q8k,
    int32_t nt,
    float* const* ff_out,
    int32_t r0,
    int32_t r1,
    float swiglu_limit
) {
    if (nt <= 0 || r0 >= r1 || n_in <= 0) return;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate_w + (size_t)r * row_bytes;
        const uint8_t* ur = up_w + (size_t)r * row_bytes;
        for (int t = 0; t < nt; ++t) {
            const auto* y = reinterpret_cast<const block_q8_K*>(x_q8k[t]);
            const float g = dot_row_q8k(ggml_type, gr, n_in, y);
            const float u = dot_row_q8k(ggml_type, ur, n_in, y);
            ff_out[t][r] = clamped_swiglu(g, u, swiglu_limit);
        }
    }
}

void ampllm_cpu_hy4_fused_gu_rows_q8_1_scalar(
    int32_t ggml_type,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t row_bytes,
    int32_t n_in,
    const void* const* x_q8_1,
    int32_t nt,
    float* const* ff_out,
    int32_t r0,
    int32_t r1,
    float swiglu_limit
) {
    if (nt <= 0 || r0 >= r1 || n_in <= 0) return;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate_w + (size_t)r * row_bytes;
        const uint8_t* ur = up_w + (size_t)r * row_bytes;
        for (int t = 0; t < nt; ++t) {
            const auto* y = reinterpret_cast<const block_q8_1_g8*>(x_q8_1[t]);
            const float g = dot_row_q8_1_g8(ggml_type, gr, n_in, y);
            const float u = dot_row_q8_1_g8(ggml_type, ur, n_in, y);
            ff_out[t][r] = clamped_swiglu(g, u, swiglu_limit);
        }
    }
}

void ampllm_cpu_hy4_down_rows_scalar(
    int32_t ggml_type,
    const uint8_t* down_w,
    size_t row_bytes,
    int32_t n_in,
    const void* const* ff_q8k,
    int32_t nt,
    float* const* out,
    int32_t r0,
    int32_t r1,
    float weight_scale,
    int32_t accumulate
) {
    if (nt <= 0 || r0 >= r1 || n_in <= 0) return;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = down_w + (size_t)r * row_bytes;
        for (int t = 0; t < nt; ++t) {
            const float v = weight_scale * dot_row_q8k(ggml_type, dr, n_in, reinterpret_cast<const block_q8_K*>(ff_q8k[t]));
            if (accumulate) out[t][r] += v;
            else out[t][r] = v;
        }
    }
}

void ampllm_cpu_hy4_down_rows_q8_1_scalar(
    int32_t ggml_type,
    const uint8_t* down_w,
    size_t row_bytes,
    int32_t n_in,
    const void* const* ff_q8_1,
    int32_t nt,
    float* const* out,
    int32_t r0,
    int32_t r1,
    float weight_scale,
    int32_t accumulate
) {
    if (nt <= 0 || r0 >= r1 || n_in <= 0) return;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* dr = down_w + (size_t)r * row_bytes;
        for (int t = 0; t < nt; ++t) {
            const float v = weight_scale * dot_row_q8_1_g8(ggml_type, dr, n_in, reinterpret_cast<const block_q8_1_g8*>(ff_q8_1[t]));
            if (accumulate) out[t][r] += v;
            else out[t][r] = v;
        }
    }
}

void ampllm_cpu_iq_fused_gu_rows_scalar(
    int32_t ggml_type,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t row_bytes,
    int32_t n_in,
    const void* const* x_q8k,
    int32_t nt,
    float* const* ff_out,
    int32_t r0,
    int32_t r1,
    float swiglu_limit
) {
    ampllm_cpu_hy4_fused_gu_rows_scalar(
        ggml_type, gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
}

void ampllm_cpu_iq_down_rows_scalar(
    int32_t ggml_type,
    const uint8_t* down_w,
    size_t row_bytes,
    int32_t n_in,
    const void* const* ff_q8k,
    int32_t nt,
    float* const* out,
    int32_t r0,
    int32_t r1,
    float weight_scale,
    int32_t accumulate
) {
    ampllm_cpu_hy4_down_rows_scalar(
        ggml_type, down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, accumulate);
}

void ampllm_cpu_mm3_quantize_q8_k_scalar(const float* x, void* out_q8k, int32_t n) {
    ampllm_cpu_quantize_q8_k_scalar(x, out_q8k, n);
}

void ampllm_cpu_mm3_fused_gu_rows_scalar(
    int32_t gate_type,
    int32_t up_type,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t gate_row_bytes,
    size_t up_row_bytes,
    int32_t n_in,
    const void* const* x_q8k,
    int32_t nt,
    float* const* ff_out,
    int32_t r0,
    int32_t r1,
    float swiglu_alpha,
    float swiglu_limit
) {
    if (nt <= 0 || r0 >= r1 || n_in <= 0) return;
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate_w + (size_t)r * gate_row_bytes;
        const uint8_t* ur = up_w + (size_t)r * up_row_bytes;
        for (int t = 0; t < nt; ++t) {
            const auto* y = reinterpret_cast<const block_q8_K*>(x_q8k[t]);
            const float g = dot_row_q8k(gate_type, gr, n_in, y);
            const float u = dot_row_q8k(up_type, ur, n_in, y);
            ff_out[t][r] = swoai(g, u, swiglu_alpha, swiglu_limit);
        }
    }
}

void ampllm_cpu_mm3_down_rows_scalar(
    int32_t down_type,
    const uint8_t* down_w,
    size_t row_bytes,
    int32_t n_in,
    const void* const* ff_q8k,
    int32_t nt,
    float* const* out,
    int32_t r0,
    int32_t r1,
    float weight_scale,
    int32_t accumulate
) {
    ampllm_cpu_hy4_down_rows_scalar(
        down_type, down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, accumulate);
}

void ampllm_cpu_mm3_q6_k_row_to_f32_scalar(const uint8_t* src, int32_t k, float* dst_f32) {
    ampllm_cpu_tok_row_to_f32_scalar(GGML_TYPE_Q6_K, src, k, dst_f32);
}

}  // extern "C"
