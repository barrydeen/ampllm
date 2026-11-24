#include "iq_tables.hpp"

#include <immintrin.h>

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

alignas(32) constexpr int8_t KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

inline float h2f(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}

inline uint16_t f2h_rne(float f) {
    __m128 v = _mm_set_ss(f);
    __m128i h = _mm_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    return (uint16_t)_mm_cvtsi128_si32(h);
}

inline float bf16_to_f32_scalar(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

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

inline int32_t hsum_i32_256(__m256i v) {
    const __m128i s0 = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    const __m128i s1 = _mm_add_epi32(s0, _mm_shuffle_epi32(s0, 0x4E));
    const __m128i s2 = _mm_add_epi32(s1, _mm_shuffle_epi32(s1, 0xB1));
    return _mm_cvtsi128_si32(s2);
}

inline float hsum_f32_256(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

inline float hmax_f32_256(__m256 x) {
    __m128 r = _mm_max_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    r = _mm_max_ps(r, _mm_movehl_ps(r, r));
    r = _mm_max_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// Saturation-free signed int8 x signed int8 (x clamped to [-127, 127]) -> 8xi32 dot sums.
inline __m256i dot_i8_avx2(__m256i w, __m256i x) {
    const __m256i aw = _mm256_sign_epi8(w, w);
    const __m256i sx = _mm256_sign_epi8(x, w);
    return _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), _mm256_set1_epi16(1));
}

// Saturation-free unsigned (<=63) x signed int8 -> 8xi32 dot sums.
inline __m256i dot_u8_avx2(__m256i w_u8, __m256i x) {
    return _mm256_madd_epi16(_mm256_maddubs_epi16(w_u8, x), _mm256_set1_epi16(1));
}

inline __m256i unpack_nibbles_32(__m128i q) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    const __m128i lo = _mm_and_si128(q, m0f);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(q, 4), m0f);
    return _mm256_set_m128i(hi, lo);
}

inline __m256i expand_32bit_mask_to_bytes(uint32_t mask32) {
    const __m256i shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 3, 3);
    const __m256i bits = _mm256_set1_epi64x((long long)0x8040201008040201ULL);
    const __m256i mb = _mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32((int)mask32), shuf), bits);
    return _mm256_cmpeq_epi8(mb, bits);
}

inline __m256i apply_sign_mask32(__m256i x, uint32_t mask32) {
    const __m256i neg_mask = expand_32bit_mask_to_bytes(mask32);
    return _mm256_sub_epi8(_mm256_xor_si256(x, neg_mask), neg_mask);
}

// Expand 32-bit sign mask into per-byte {-1, +1} vector for single-cycle _mm256_sign_epi8(x, sign_v).
inline __m256i expand_32bit_sign_v(uint32_t mask32) {
    return _mm256_or_si256(expand_32bit_mask_to_bytes(mask32), _mm256_set1_epi8(1));
}

inline __m256i get_scale_shuffle_k4(int i) {
    static const uint8_t k_shuffle[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

inline __m256i get_scale_shuffle_q3k(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

// Per-row dot products over NT tokens simultaneously so weight row is read once.
template <int NT, bool IsQ81>
void dot_q8_0_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / 32;
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_q8_0*>(w + (size_t)r * rb);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            const __m256i wv = _mm256_loadu_si256((const __m256i*)br[i].qs);
            const __m256i aw = _mm256_sign_epi8(wv, wv);
            const int blk = i >> 3;
            const int sub = i & 7;
            for (int t = 0; t < NT; ++t) {
                const int8_t* xp;
                float xd;
                if constexpr (IsQ81) {
                    const auto* xb = reinterpret_cast<const block_q8_1_g8*>(x_ptrs[t]) + blk;
                    xp = xb->qs + sub * 32;
                    xd = xb->d[sub];
                } else {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + blk;
                    xp = xb->qs + sub * 32;
                    xd = xb->d;
                }
                const __m256i xv = _mm256_loadu_si256((const __m256i*)xp);
                const __m256i sx = _mm256_sign_epi8(xv, wv);
                const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones16);
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xd), _mm256_cvtepi32_ps(p32), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]);
    }
}

template <int NT, bool IsQ81>
void dot_iq4_nl_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / 32;
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)KVALUES_IQ4NL));
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_iq4_nl*>(w + (size_t)r * rb);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            const __m256i idx = unpack_nibbles_32(_mm_loadu_si128((const __m128i*)br[i].qs));
            const __m256i wv = _mm256_shuffle_epi8(lut, idx);
            const __m256i aw = _mm256_sign_epi8(wv, wv);
            const int blk = i >> 3;
            const int sub = i & 7;
            for (int t = 0; t < NT; ++t) {
                const int8_t* xp;
                float xd;
                if constexpr (IsQ81) {
                    const auto* xb = reinterpret_cast<const block_q8_1_g8*>(x_ptrs[t]) + blk;
                    xp = xb->qs + sub * 32;
                    xd = xb->d[sub];
                } else {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + blk;
                    xp = xb->qs + sub * 32;
                    xd = xb->d;
                }
                const __m256i xv = _mm256_loadu_si256((const __m256i*)xp);
                const __m256i sx = _mm256_sign_epi8(xv, wv);
                const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(aw, sx), ones16);
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xd), _mm256_cvtepi32_ps(p32), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]);
    }
}

template <int NT, bool IsQ81>
void dot_q5_0_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / 32;
    const __m256i v16 = _mm256_set1_epi8(16);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_q5_0*>(w + (size_t)r * rb);
        __m256 accf[NT];
        float bias[NT];
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_setzero_ps();
            bias[t] = 0.0f;
        }
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            const uint32_t qh = u32(br[i].qh);
            const __m256i w4 = unpack_nibbles_32(_mm_loadu_si128((const __m128i*)br[i].qs));
            const __m256i hi_m = _mm256_and_si256(expand_32bit_mask_to_bytes(qh), v16);
            const __m256i w5 = _mm256_or_si256(w4, hi_m); // 0..31 unsigned
            const int blk = i >> 3;
            const int sub = i & 7;
            for (int t = 0; t < NT; ++t) {
                const int8_t* xp;
                float xd;
                int32_t bsum;
                if constexpr (IsQ81) {
                    const auto* xb = reinterpret_cast<const block_q8_1_g8*>(x_ptrs[t]) + blk;
                    xp = xb->qs + sub * 32;
                    xd = xb->d[sub];
                    bsum = xb->bsum[sub];
                } else {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + blk;
                    xp = xb->qs + sub * 32;
                    xd = xb->d;
                    bsum = (int32_t)xb->bsums[2 * sub] + (int32_t)xb->bsums[2 * sub + 1];
                }
                const __m256i xv = _mm256_loadu_si256((const __m256i*)xp);
                const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(w5, xv), ones16);
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xd), _mm256_cvtepi32_ps(p32), accf[t]);
                bias[t] -= 16.0f * dw * xd * (float)bsum;
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]) + bias[t];
    }
}

template <int NT, bool IsQ81>
void dot_q5_1_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / 32;
    const __m256i v16 = _mm256_set1_epi8(16);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_q5_1*>(w + (size_t)r * rb);
        __m256 accf[NT];
        float bias[NT];
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_setzero_ps();
            bias[t] = 0.0f;
        }
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            const float mw = h2f(br[i].m);
            const uint32_t qh = u32(br[i].qh);
            const __m256i w4 = unpack_nibbles_32(_mm_loadu_si128((const __m128i*)br[i].qs));
            const __m256i hi_m = _mm256_and_si256(expand_32bit_mask_to_bytes(qh), v16);
            const __m256i w5 = _mm256_or_si256(w4, hi_m); // 0..31 unsigned
            const int blk = i >> 3;
            const int sub = i & 7;
            for (int t = 0; t < NT; ++t) {
                const int8_t* xp;
                float xd, xs;
                if constexpr (IsQ81) {
                    const auto* xb = reinterpret_cast<const block_q8_1_g8*>(x_ptrs[t]) + blk;
                    xp = xb->qs + sub * 32;
                    xd = xb->d[sub];
                    xs = xb->s[sub];
                } else {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + blk;
                    xp = xb->qs + sub * 32;
                    xd = xb->d;
                    xs = xd * (float)((int32_t)xb->bsums[2 * sub] + (int32_t)xb->bsums[2 * sub + 1]);
                }
                const __m256i xv = _mm256_loadu_si256((const __m256i*)xp);
                const __m256i p32 = _mm256_madd_epi16(_mm256_maddubs_epi16(w5, xv), ones16);
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xd), _mm256_cvtepi32_ps(p32), accf[t]);
                bias[t] += mw * xs;
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]) + bias[t];
    }
}

template <int NT, bool IsQ81>
void dot_q2_0_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / 64;
    const __m256i ones16 = _mm256_set1_epi16(1);
    const __m256i m03_0 = _mm256_set1_epi32(0x00000003);
    const __m256i m03_1 = _mm256_set1_epi32(0x00000300);
    const __m256i m03_2 = _mm256_set1_epi32(0x00030000);
    const __m256i m03_3 = _mm256_set1_epi32(0x03000000);
    auto unpack32 = [&](const uint8_t* qs8) -> __m256i {
        const __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)qs8));
        const __m256i e0 = _mm256_and_si256(v, m03_0);
        const __m256i e1 = _mm256_and_si256(_mm256_slli_epi32(v, 6), m03_1);
        const __m256i e2 = _mm256_and_si256(_mm256_slli_epi32(v, 12), m03_2);
        const __m256i e3 = _mm256_and_si256(_mm256_slli_epi32(v, 18), m03_3);
        return _mm256_or_si256(_mm256_or_si256(e0, e1), _mm256_or_si256(e2, e3));
    };
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_q2_0*>(w + (size_t)r * rb);
        __m256 accf[NT];
        float bias[NT];
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_setzero_ps();
            bias[t] = 0.0f;
        }
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            const __m256i w0 = unpack32(br[i].qs + 0);
            const __m256i w1 = unpack32(br[i].qs + 8);
            const int blk = i >> 2;
            const int sub64 = i & 3;
            const int s0 = 2 * sub64 + 0, s1 = 2 * sub64 + 1;
            for (int t = 0; t < NT; ++t) {
                if constexpr (IsQ81) {
                    const auto* xb = reinterpret_cast<const block_q8_1_g8*>(x_ptrs[t]) + blk;
                    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(w0, _mm256_loadu_si256((const __m256i*)(xb->qs + s0 * 32))), ones16);
                    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w1, _mm256_loadu_si256((const __m256i*)(xb->qs + s1 * 32))), ones16);
                    accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d[s0]), _mm256_cvtepi32_ps(p0), accf[t]);
                    accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d[s1]), _mm256_cvtepi32_ps(p1), accf[t]);
                    bias[t] -= dw * (xb->d[s0] * (float)xb->bsum[s0] + xb->d[s1] * (float)xb->bsum[s1]);
                } else {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + blk;
                    const int32_t b0 = (int32_t)xb->bsums[2 * s0] + (int32_t)xb->bsums[2 * s0 + 1];
                    const int32_t b1 = (int32_t)xb->bsums[2 * s1] + (int32_t)xb->bsums[2 * s1 + 1];
                    const __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(w0, _mm256_loadu_si256((const __m256i*)(xb->qs + s0 * 32))), ones16);
                    const __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(w1, _mm256_loadu_si256((const __m256i*)(xb->qs + s1 * 32))), ones16);
                    accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d), _mm256_cvtepi32_ps(_mm256_add_epi32(p0, p1)), accf[t]);
                    bias[t] -= dw * xb->d * (float)(b0 + b1);
                }
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]) + bias[t];
    }
}

template <int NT, bool IsQ81>
void dot_q4_k_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    static constexpr uint32_t kmask1 = 0x3f3f3f3fu;
    static constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    static constexpr uint32_t kmask3 = 0x03030303u;
    const int nb = n_in / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_q4_K*>(w + (size_t)r * rb);
        float accf[NT] = {};
        for (int i = 0; i < nb; ++i) {
            if (i + 2 < nb) _mm_prefetch((const char*)(br + i + 2), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            const float dmin = h2f(br[i].dmin);
            uint32_t utmp[4];
            std::memcpy(utmp, br[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
            const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            const __m128i mn128 = _mm256_extracti128_si256(mins_and_scales, 1);
            const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
            const __m256i mn32 = _mm256_cvtepi16_epi32(mn128);
            const __m256 mn_f = _mm256_mul_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(mn32));
            const __m256 sc_f = _mm256_mul_ps(_mm256_set1_ps(dw), _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sc128)));

            __m256i q4l_arr[4], q4h_arr[4];
            for (int j = 0; j < 4; ++j) {
                const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(br[i].qs + j * 32));
                q4l_arr[j] = _mm256_and_si256(q4bits, m4);
                q4h_arr[j] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            }

            if constexpr (IsQ81) {
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_1_g8*>(x_ptrs[t]) + i;
                    alignas(32) int32_t s_arr[8];
                    for (int j = 0; j < 4; ++j) {
                        const __m256i x0 = _mm256_loadu_si256((const __m256i*)(xb->qs + (2 * j + 0) * 32));
                        const __m256i x1 = _mm256_loadu_si256((const __m256i*)(xb->qs + (2 * j + 1) * 32));
                        s_arr[2 * j + 0] = hsum_i32_256(_mm256_madd_epi16(_mm256_maddubs_epi16(q4l_arr[j], x0), ones16));
                        s_arr[2 * j + 1] = hsum_i32_256(_mm256_madd_epi16(_mm256_maddubs_epi16(q4h_arr[j], x1), ones16));
                    }
                    const __m256 sv = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)s_arr));
                    const __m256 dx8 = _mm256_loadu_ps(xb->d);
                    const __m256 sx8 = _mm256_loadu_ps(xb->s);
                    const __m256 prod = _mm256_fmsub_ps(_mm256_mul_ps(sc_f, dx8), sv, _mm256_mul_ps(mn_f, sx8));
                    accf[t] += hsum_f32_256(prod);
                }
            } else {
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                    const __m256i q8sums = _mm256_loadu_si256((const __m256i*)xb->bsums);
                    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                    const __m128i mprod = _mm_madd_epi16(mn128, q8s);
                    const __m128i m1 = _mm_hadd_epi32(mprod, mprod);
                    const __m128i m2 = _mm_hadd_epi32(m1, m1);
                    const int32_t sum_m = _mm_cvtsi128_si32(m2);
                    __m256i sumi = _mm256_setzero_si256();
                    for (int j = 0; j < 4; ++j) {
                        const __m256i x0 = _mm256_loadu_si256((const __m256i*)(xb->qs + (2 * j + 0) * 32));
                        const __m256i x1 = _mm256_loadu_si256((const __m256i*)(xb->qs + (2 * j + 1) * 32));
                        const __m256i sc_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                        const __m256i sc_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sc_l, _mm256_maddubs_epi16(q4l_arr[j], x0)));
                        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sc_h, _mm256_maddubs_epi16(q4h_arr[j], x1)));
                    }
                    accf[t] += xb->d * (dw * (float)hsum_i32_256(sumi) - dmin * (float)sum_m);
                }
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = accf[t];
    }
}

template <int NT, bool IsQ81>
void dot_q5_k_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    static constexpr uint32_t kmask1 = 0x3f3f3f3fu;
    static constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    static constexpr uint32_t kmask3 = 0x03030303u;
    const int nb = n_in / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_q5_K*>(w + (size_t)r * rb);
        float accf[NT] = {};
        for (int i = 0; i < nb; ++i) {
            if (i + 2 < nb) _mm_prefetch((const char*)(br + i + 2), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            const float dmin = h2f(br[i].dmin);
            uint32_t utmp[4];
            std::memcpy(utmp, br[i].scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            const uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;
            const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
            const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            const __m128i mn128 = _mm256_extracti128_si256(mins_and_scales, 1);
            const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
            const __m256 mn_f = _mm256_mul_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(mn128)));
            const __m256 sc_f = _mm256_mul_ps(_mm256_set1_ps(dw), _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(sc128)));

            const __m256i qh = _mm256_loadu_si256((const __m256i*)br[i].qh);
            __m256i w5l[4], w5h[4];
            for (int j = 0; j < 4; ++j) {
                const __m256i q4bits = _mm256_loadu_si256((const __m256i*)(br[i].qs + j * 32));
                const __m256i lo = _mm256_and_si256(q4bits, m4);
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
                const __m256i m_lo = _mm256_set1_epi8((char)(1u << (2 * j)));
                const __m256i m_hi = _mm256_set1_epi8((char)(2u << (2 * j)));
                const __m256i h_lo = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(qh, m_lo), 2 * j), 4);
                const __m256i h_hi = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(qh, m_hi), 2 * j + 1), 4);
                w5l[j] = _mm256_or_si256(lo, h_lo);
                w5h[j] = _mm256_or_si256(hi, h_hi);
            }

            if constexpr (IsQ81) {
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_1_g8*>(x_ptrs[t]) + i;
                    alignas(32) int32_t s_arr[8];
                    for (int j = 0; j < 4; ++j) {
                        const __m256i x0 = _mm256_loadu_si256((const __m256i*)(xb->qs + (2 * j + 0) * 32));
                        const __m256i x1 = _mm256_loadu_si256((const __m256i*)(xb->qs + (2 * j + 1) * 32));
                        s_arr[2 * j + 0] = hsum_i32_256(_mm256_madd_epi16(_mm256_maddubs_epi16(w5l[j], x0), ones16));
                        s_arr[2 * j + 1] = hsum_i32_256(_mm256_madd_epi16(_mm256_maddubs_epi16(w5h[j], x1), ones16));
                    }
                    const __m256 sv = _mm256_cvtepi32_ps(_mm256_load_si256((const __m256i*)s_arr));
                    const __m256 dx8 = _mm256_loadu_ps(xb->d);
                    const __m256 sx8 = _mm256_loadu_ps(xb->s);
                    const __m256 prod = _mm256_fmsub_ps(_mm256_mul_ps(sc_f, dx8), sv, _mm256_mul_ps(mn_f, sx8));
                    accf[t] += hsum_f32_256(prod);
                }
            } else {
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                    const __m256i q8sums = _mm256_loadu_si256((const __m256i*)xb->bsums);
                    const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                    const __m128i mprod = _mm_madd_epi16(mn128, q8s);
                    const __m128i m1 = _mm_hadd_epi32(mprod, mprod);
                    const __m128i m2 = _mm_hadd_epi32(m1, m1);
                    const int32_t sum_m = _mm_cvtsi128_si32(m2);
                    __m256i sumi = _mm256_setzero_si256();
                    for (int j = 0; j < 4; ++j) {
                        const __m256i x0 = _mm256_loadu_si256((const __m256i*)(xb->qs + (2 * j + 0) * 32));
                        const __m256i x1 = _mm256_loadu_si256((const __m256i*)(xb->qs + (2 * j + 1) * 32));
                        const __m256i sc_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
                        const __m256i sc_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
                        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sc_l, _mm256_maddubs_epi16(w5l[j], x0)));
                        sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(sc_h, _mm256_maddubs_epi16(w5h[j], x1)));
                    }
                    accf[t] += xb->d * (dw * (float)hsum_i32_256(sumi) - dmin * (float)sum_m);
                }
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = accf[t];
    }
}

template <int NT>
void dot_q6_k_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / QK_K;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(3);
    const __m256i m32 = _mm256_set1_epi8(32);
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_q6_K*>(w + (size_t)r * rb);
        float accf[NT] = {};
        for (int i = 0; i < nb; ++i) {
            const float dw = h2f(br[i].d);
            const __m256i sc_all = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)br[i].scales));
            __m256i wq[8];
            for (int half = 0; half < 2; ++half) {
                const uint8_t* ql = br[i].ql + half * 64;
                const uint8_t* qh = br[i].qh + half * 32;
                const __m256i ql0 = _mm256_loadu_si256((const __m256i*)(ql + 0));
                const __m256i ql1 = _mm256_loadu_si256((const __m256i*)(ql + 32));
                const __m256i qhv = _mm256_loadu_si256((const __m256i*)qh);
                const __m256i q0 = _mm256_or_si256(_mm256_and_si256(ql0, m4), _mm256_slli_epi16(_mm256_and_si256(qhv, m2), 4));
                const __m256i q1 = _mm256_or_si256(_mm256_and_si256(ql1, m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 2), m2), 4));
                const __m256i q2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 4), m2), 4));
                const __m256i q3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m4), _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qhv, 6), m2), 4));
                wq[half * 4 + 0] = _mm256_sub_epi8(q0, m32);
                wq[half * 4 + 1] = _mm256_sub_epi8(q1, m32);
                wq[half * 4 + 2] = _mm256_sub_epi8(q2, m32);
                wq[half * 4 + 3] = _mm256_sub_epi8(q3, m32);
            }
            const __m128i sc_lo = _mm256_extracti128_si256(sc_all, 0);
            const __m128i sc_hi = _mm256_extracti128_si256(sc_all, 1);
            const __m256i sc_b0 = _mm256_insertf128_si256(_mm256_castsi128_si256(sc_lo), sc_lo, 1);
            const __m256i sc_b1 = _mm256_insertf128_si256(_mm256_castsi128_si256(sc_hi), sc_hi, 1);
            for (int t = 0; t < NT; ++t) {
                const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                __m256i sumi = _mm256_setzero_si256();
                for (int s = 0; s < 4; ++s) {
                    const __m256i xv = _mm256_loadu_si256((const __m256i*)(xb->qs + s * 32));
                    const __m256i aw = _mm256_sign_epi8(wq[s], wq[s]);
                    const __m256i sx = _mm256_sign_epi8(xv, wq[s]);
                    const __m256i p16 = _mm256_maddubs_epi16(aw, sx);
                    const __m256i scv = _mm256_shuffle_epi8(sc_b0, get_scale_shuffle_q3k(s));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scv, p16));
                }
                for (int s = 0; s < 4; ++s) {
                    const __m256i xv = _mm256_loadu_si256((const __m256i*)(xb->qs + 128 + s * 32));
                    const __m256i aw = _mm256_sign_epi8(wq[4 + s], wq[4 + s]);
                    const __m256i sx = _mm256_sign_epi8(xv, wq[4 + s]);
                    const __m256i p16 = _mm256_maddubs_epi16(aw, sx);
                    const __m256i scv = _mm256_shuffle_epi8(sc_b1, get_scale_shuffle_q3k(s));
                    sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(scv, p16));
                }
                accf[t] += dw * xb->d * (float)hsum_i32_256(sumi);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = accf[t];
    }
}

template <int NT>
void dot_q3_k_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / QK_K;
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_q3_K*>(w + (size_t)r * rb);
        float accf[NT] = {};
        for (int i = 0; i < nb; ++i) {
            const float dw = h2f(br[i].d);
            uint32_t aux[3];
            std::memcpy(aux, br[i].scales, 12);
            __m128i scales128 = _mm_set_epi32(
                ((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4),
                ((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4),
                (aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4),
                (aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4));
            scales128 = _mm_sub_epi8(scales128, m32);
            const __m256i all_scales = _mm256_cvtepi8_epi16(scales128);
            const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
            const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
            const __m256i scales[2] = {
                _mm256_insertf128_si256(_mm256_castsi128_si256(l_scales), l_scales, 1),
                _mm256_insertf128_si256(_mm256_castsi128_si256(h_scales), h_scales, 1),
            };
            const __m256i hbits = _mm256_loadu_si256((const __m256i*)br[i].hmask);
            __m256i q3l[8], q3h[8];
            const uint8_t* q3 = br[i].qs;
            int bit = 0;
            for (int j = 0; j < 2; ++j) {
                const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3);
                q3 += 32;
                for (int s = 0; s < 4; ++s) {
                    q3l[j * 4 + s] = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2 * s), m3);
                    q3h[j * 4 + s] = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
                    ++bit;
                }
            }
            for (int t = 0; t < NT; ++t) {
                const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                __m256i sumi = _mm256_setzero_si256();
                for (int j = 0; j < 2; ++j) {
                    for (int s = 0; s < 4; ++s) {
                        const __m256i q8v = _mm256_loadu_si256((const __m256i*)(xb->qs + (j * 4 + s) * 32));
                        const __m256i p16 = _mm256_sub_epi16(
                            _mm256_maddubs_epi16(q3l[j * 4 + s], q8v),
                            _mm256_maddubs_epi16(q3h[j * 4 + s], q8v));
                        const __m256i p32 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(s)), p16);
                        sumi = _mm256_add_epi32(sumi, p32);
                    }
                }
                accf[t] += dw * xb->d * (float)hsum_i32_256(sumi);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = accf[t];
    }
}

template <int NT>
void dot_iq4_xs_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / QK_K;
    const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)KVALUES_IQ4NL));
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_iq4_xs*>(w + (size_t)r * rb);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            if (i + 3 < nb) _mm_prefetch((const char*)(br + i + 3), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            const uint16_t sh = br[i].scales_h;
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib = 0; ib < 8; ++ib) {
                const int ls = ((br[i].scales_l[ib / 2] >> (4 * (ib & 1))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                const __m256i sc16 = _mm256_set1_epi16((int16_t)(ls - 32));
                const __m256i idx = unpack_nibbles_32(_mm_loadu_si128((const __m128i*)(br[i].qs + ib * 16)));
                const __m256i wv = _mm256_shuffle_epi8(lut, idx);
                const __m256i aw = _mm256_sign_epi8(wv, wv);
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                    const __m256i xv = _mm256_loadu_si256((const __m256i*)(xb->qs + ib * 32));
                    const __m256i sx = _mm256_sign_epi8(xv, wv);
                    const __m256i p16 = _mm256_maddubs_epi16(aw, sx);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(p16, sc16));
                }
            }
            for (int t = 0; t < NT; ++t) {
                const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]);
    }
}

template <int NT>
void dot_iq2_xxs_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / QK_K;
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_iq2_xxs*>(w + (size_t)r * rb);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d) * 0.125f;
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib = 0; ib < 8; ++ib) {
                const uint32_t aux0 = u32(br[i].qs + 8 * ib);
                const uint32_t aux1 = u32(br[i].qs + 8 * ib + 4);
                const __m256i sc16 = _mm256_set1_epi16((int16_t)(2 * (int32_t)(aux1 >> 28) + 1));
                const __m256i gv = _mm256_set_epi64x(
                    (long long)iq2xxs_grid[(aux0 >> 24) & 0xFF],
                    (long long)iq2xxs_grid[(aux0 >> 16) & 0xFF],
                    (long long)iq2xxs_grid[(aux0 >> 8) & 0xFF],
                    (long long)iq2xxs_grid[aux0 & 0xFF]);
                const uint32_t sg = (uint32_t)ksigns_iq2xs[(aux1 >> 0) & 0x7F]
                                  | ((uint32_t)ksigns_iq2xs[(aux1 >> 7) & 0x7F] << 8)
                                  | ((uint32_t)ksigns_iq2xs[(aux1 >> 14) & 0x7F] << 16)
                                  | ((uint32_t)ksigns_iq2xs[(aux1 >> 21) & 0x7F] << 24);
                const __m256i sv = expand_32bit_sign_v(sg);
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                    const __m256i xv = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xb->qs + ib * 32)), sv);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(_mm256_maddubs_epi16(gv, xv), sc16));
                }
            }
            for (int t = 0; t < NT; ++t) {
                const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]);
    }
}

template <int NT>
void dot_iq2_xs_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / QK_K;
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_iq2_xs*>(w + (size_t)r * rb);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d) * 0.125f;
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib = 0; ib < 8; ++ib) {
                const uint16_t q0 = u16(br[i].qs + 8 * ib + 0), q1 = u16(br[i].qs + 8 * ib + 2);
                const uint16_t q2 = u16(br[i].qs + 8 * ib + 4), q3 = u16(br[i].qs + 8 * ib + 6);
                const __m256i gv = _mm256_set_epi64x(
                    (long long)iq2xs_grid[q3 & 511], (long long)iq2xs_grid[q2 & 511],
                    (long long)iq2xs_grid[q1 & 511], (long long)iq2xs_grid[q0 & 511]);
                const uint32_t sg = (uint32_t)ksigns_iq2xs[q0 >> 9]
                                  | ((uint32_t)ksigns_iq2xs[q1 >> 9] << 8)
                                  | ((uint32_t)ksigns_iq2xs[q2 >> 9] << 16)
                                  | ((uint32_t)ksigns_iq2xs[q3 >> 9] << 24);
                const __m256i sv = expand_32bit_sign_v(sg);
                const uint8_t sb = br[i].scales[ib];
                const int16_t sc0 = (int16_t)(2 * (int32_t)(sb & 0xF) + 1);
                const int16_t sc1 = (int16_t)(2 * (int32_t)(sb >> 4) + 1);
                const __m256i sc16 = _mm256_set_m128i(_mm_set1_epi16(sc1), _mm_set1_epi16(sc0));
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                    const __m256i xv = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xb->qs + ib * 32)), sv);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(_mm256_maddubs_epi16(gv, xv), sc16));
                }
            }
            for (int t = 0; t < NT; ++t) {
                const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]);
    }
}

template <int NT>
void dot_iq2_s_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / QK_K;
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_iq2_s*>(w + (size_t)r * rb);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d) * 0.125f;
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t* qs = br[i].qs + 4 * ib;
                const uint8_t qh = br[i].qh[ib];
                const __m256i gv = _mm256_set_epi64x(
                    (long long)iq2s_grid[qs[3] | (((uint32_t)qh << 2) & 0x300)],
                    (long long)iq2s_grid[qs[2] | (((uint32_t)qh << 4) & 0x300)],
                    (long long)iq2s_grid[qs[1] | (((uint32_t)qh << 6) & 0x300)],
                    (long long)iq2s_grid[qs[0] | (((uint32_t)qh << 8) & 0x300)]);
                const __m256i sv = expand_32bit_sign_v(u32(br[i].qs + 32 + 4 * ib));
                const uint8_t sb = br[i].scales[ib];
                const int16_t sc0 = (int16_t)(2 * (int32_t)(sb & 0xF) + 1);
                const int16_t sc1 = (int16_t)(2 * (int32_t)(sb >> 4) + 1);
                const __m256i sc16 = _mm256_set_m128i(_mm_set1_epi16(sc1), _mm_set1_epi16(sc0));
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                    const __m256i xv = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xb->qs + ib * 32)), sv);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(_mm256_maddubs_epi16(gv, xv), sc16));
                }
            }
            for (int t = 0; t < NT; ++t) {
                const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]);
    }
}

template <int NT>
void dot_iq3_xxs_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / QK_K;
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_iq3_xxs*>(w + (size_t)r * rb);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d) * 0.25f;
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t* qs = br[i].qs + 8 * ib;
                const uint32_t aux = u32(br[i].qs + 64 + 4 * ib);
                const __m256i sc16 = _mm256_set1_epi16((int16_t)(2 * (int32_t)(aux >> 28) + 1));
                const __m256i gv = _mm256_setr_epi32(
                    (int)iq3xxs_grid[qs[0]], (int)iq3xxs_grid[qs[1]],
                    (int)iq3xxs_grid[qs[2]], (int)iq3xxs_grid[qs[3]],
                    (int)iq3xxs_grid[qs[4]], (int)iq3xxs_grid[qs[5]],
                    (int)iq3xxs_grid[qs[6]], (int)iq3xxs_grid[qs[7]]);
                const uint32_t sg = (uint32_t)ksigns_iq2xs[(aux >> 0) & 0x7F]
                                  | ((uint32_t)ksigns_iq2xs[(aux >> 7) & 0x7F] << 8)
                                  | ((uint32_t)ksigns_iq2xs[(aux >> 14) & 0x7F] << 16)
                                  | ((uint32_t)ksigns_iq2xs[(aux >> 21) & 0x7F] << 24);
                const __m256i sv = expand_32bit_sign_v(sg);
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                    const __m256i xv = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xb->qs + ib * 32)), sv);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(_mm256_maddubs_epi16(gv, xv), sc16));
                }
            }
            for (int t = 0; t < NT; ++t) {
                const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]);
    }
}

template <int NT>
void dot_iq3_s_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / QK_K;
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_iq3_s*>(w + (size_t)r * rb);
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            if (i + 3 < nb) _mm_prefetch((const char*)(br + i + 3), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            __m256i sumi[NT];
            for (int t = 0; t < NT; ++t) sumi[t] = _mm256_setzero_si256();
            for (int ib = 0; ib < 8; ++ib) {
                const uint8_t sb = br[i].scales[ib >> 1];
                const __m256i sc16 = _mm256_set1_epi16((int16_t)(2 * (int32_t)((ib & 1) ? (sb >> 4) : (sb & 0xF)) + 1));
                const uint32_t qh = br[i].qh[ib];
                const uint8_t* qs = br[i].qs + 8 * ib;
                const __m256i gv = _mm256_setr_epi32(
                    (int)iq3s_grid[qs[0] | ((qh << 8) & 0x100)],
                    (int)iq3s_grid[qs[1] | ((qh << 7) & 0x100)],
                    (int)iq3s_grid[qs[2] | ((qh << 6) & 0x100)],
                    (int)iq3s_grid[qs[3] | ((qh << 5) & 0x100)],
                    (int)iq3s_grid[qs[4] | ((qh << 4) & 0x100)],
                    (int)iq3s_grid[qs[5] | ((qh << 3) & 0x100)],
                    (int)iq3s_grid[qs[6] | ((qh << 2) & 0x100)],
                    (int)iq3s_grid[qs[7] | ((qh << 1) & 0x100)]);
                const __m256i sv = expand_32bit_sign_v(u32(br[i].signs + 4 * ib));
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                    const __m256i xv = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i*)(xb->qs + ib * 32)), sv);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(_mm256_maddubs_epi16(gv, xv), sc16));
                }
            }
            for (int t = 0; t < NT; ++t) {
                const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]);
    }
}

template <int NT>
void dot_iq1_s_nt(const uint8_t* w, size_t rb, int n_in, const void* const* x_ptrs, int r0, int r1, float* const* acc_out) {
    const int nb = n_in / QK_K;
    const __m256i one_u8 = _mm256_set1_epi8(1);
    for (int r = r0; r < r1; ++r) {
        const auto* br = reinterpret_cast<const block_iq1_s*>(w + (size_t)r * rb);
        __m256 accf[NT];
        float bias[NT];
        for (int t = 0; t < NT; ++t) {
            accf[t] = _mm256_setzero_ps();
            bias[t] = 0.0f;
        }
        for (int i = 0; i < nb; ++i) {
            if (i + 4 < nb) _mm_prefetch((const char*)(br + i + 4), _MM_HINT_T0);
            const float dw = h2f(br[i].d);
            __m256i sumi[NT];
            int32_t delta_sum[NT];
            for (int t = 0; t < NT; ++t) {
                sumi[t] = _mm256_setzero_si256();
                delta_sum[t] = 0;
            }
            for (int ib = 0; ib < 8; ++ib) {
                const uint16_t qh = br[i].qh[ib];
                const int32_t sc_ib = 2 * (int32_t)((qh >> 12) & 7) + 1;
                const int32_t sgn_ib = (qh & 0x8000u) ? -1 : 1;
                const __m256i sc16 = _mm256_set1_epi16((int16_t)sc_ib);
                const uint8_t* qs = br[i].qs + 4 * ib;
                const __m256i gv = _mm256_set_epi64x(
                    (long long)iq1s_grid[qs[3] | (((qh >> 9) & 7) << 8)],
                    (long long)iq1s_grid[qs[2] | (((qh >> 6) & 7) << 8)],
                    (long long)iq1s_grid[qs[1] | (((qh >> 3) & 7) << 8)],
                    (long long)iq1s_grid[qs[0] | (((qh >> 0) & 7) << 8)]);
                for (int t = 0; t < NT; ++t) {
                    const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                    const __m256i xv = _mm256_loadu_si256((const __m256i*)(xb->qs + ib * 32));
                    const __m256i sx = _mm256_sign_epi8(xv, gv);
                    sumi[t] = _mm256_add_epi32(sumi[t], _mm256_madd_epi16(_mm256_maddubs_epi16(one_u8, sx), sc16));
                    const int32_t bsum = (int32_t)xb->bsums[2 * ib] + (int32_t)xb->bsums[2 * ib + 1];
                    delta_sum[t] += sc_ib * sgn_ib * bsum;
                }
            }
            for (int t = 0; t < NT; ++t) {
                const auto* xb = reinterpret_cast<const block_q8_K*>(x_ptrs[t]) + i;
                accf[t] = _mm256_fmadd_ps(_mm256_set1_ps(dw * xb->d), _mm256_cvtepi32_ps(sumi[t]), accf[t]);
                bias[t] += dw * xb->d * 0.125f * (float)delta_sum[t];
            }
        }
        for (int t = 0; t < NT; ++t) acc_out[t][r - r0] = hsum_f32_256(accf[t]) + bias[t];
    }
}

template <int NT>
void dot_dispatch_nt(
    int32_t ggml_type,
    const uint8_t* w,
    size_t rb,
    int32_t n_in,
    const void* const* x_ptrs,
    bool is_q8_1,
    int r0,
    int r1,
    float* const* acc_out
) {
    switch (ggml_type) {
        case GGML_TYPE_Q8_0:
            if (is_q8_1) dot_q8_0_nt<NT, true>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            else         dot_q8_0_nt<NT, false>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            break;
        case GGML_TYPE_IQ4_NL:
            if (is_q8_1) dot_iq4_nl_nt<NT, true>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            else         dot_iq4_nl_nt<NT, false>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            break;
        case GGML_TYPE_Q5_0:
            if (is_q8_1) dot_q5_0_nt<NT, true>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            else         dot_q5_0_nt<NT, false>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            break;
        case GGML_TYPE_Q5_1:
            if (is_q8_1) dot_q5_1_nt<NT, true>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            else         dot_q5_1_nt<NT, false>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            break;
        case GGML_TYPE_Q2_0:
            if (is_q8_1) dot_q2_0_nt<NT, true>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            else         dot_q2_0_nt<NT, false>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            break;
        case GGML_TYPE_Q4_K:
            if (is_q8_1) dot_q4_k_nt<NT, true>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            else         dot_q4_k_nt<NT, false>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            break;
        case GGML_TYPE_Q5_K:
            if (is_q8_1) dot_q5_k_nt<NT, true>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            else         dot_q5_k_nt<NT, false>(w, rb, n_in, x_ptrs, r0, r1, acc_out);
            break;
        case GGML_TYPE_Q6_K:   dot_q6_k_nt<NT>(w, rb, n_in, x_ptrs, r0, r1, acc_out); break;
        case GGML_TYPE_Q3_K:   dot_q3_k_nt<NT>(w, rb, n_in, x_ptrs, r0, r1, acc_out); break;
        case GGML_TYPE_IQ4_XS: dot_iq4_xs_nt<NT>(w, rb, n_in, x_ptrs, r0, r1, acc_out); break;
        case GGML_TYPE_IQ2_XXS:dot_iq2_xxs_nt<NT>(w, rb, n_in, x_ptrs, r0, r1, acc_out); break;
        case GGML_TYPE_IQ2_XS: dot_iq2_xs_nt<NT>(w, rb, n_in, x_ptrs, r0, r1, acc_out); break;
        case GGML_TYPE_IQ1_S:  dot_iq1_s_nt<NT>(w, rb, n_in, x_ptrs, r0, r1, acc_out); break;
        case GGML_TYPE_IQ3_XXS:dot_iq3_xxs_nt<NT>(w, rb, n_in, x_ptrs, r0, r1, acc_out); break;
        case GGML_TYPE_IQ3_S:  dot_iq3_s_nt<NT>(w, rb, n_in, x_ptrs, r0, r1, acc_out); break;
        case GGML_TYPE_IQ2_S:  dot_iq2_s_nt<NT>(w, rb, n_in, x_ptrs, r0, r1, acc_out); break;
        default: break;
    }
}

template <int NT>
void run_gu_tile(
    int32_t gate_type,
    int32_t up_type,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t gate_rb,
    size_t up_rb,
    int32_t n_in,
    const void* const* x_ptrs,
    bool is_q8_1,
    float* const* ff_out,
    int t_off,
    int32_t r0,
    int32_t r1,
    bool is_oai,
    float alpha,
    float limit
) {
    constexpr int kTile = 64;
    float g_buf[NT][kTile];
    float u_buf[NT][kTile];
    float* gp[NT];
    float* up[NT];
    for (int t = 0; t < NT; ++t) {
        gp[t] = g_buf[t];
        up[t] = u_buf[t];
    }
    for (int rr0 = r0; rr0 < r1; rr0 += kTile) {
        const int rr1 = std::min<int32_t>(r1, rr0 + kTile);
        const int nr = rr1 - rr0;
        dot_dispatch_nt<NT>(gate_type, gate_w, gate_rb, n_in, x_ptrs + t_off, is_q8_1, rr0, rr1, gp);
        dot_dispatch_nt<NT>(up_type, up_w, up_rb, n_in, x_ptrs + t_off, is_q8_1, rr0, rr1, up);
        for (int t = 0; t < NT; ++t) {
            float* dst = ff_out[t_off + t] + rr0;
            if (is_oai) {
                for (int r = 0; r < nr; ++r) dst[r] = swoai(g_buf[t][r], u_buf[t][r], alpha, limit);
            } else {
                for (int r = 0; r < nr; ++r) dst[r] = clamped_swiglu(g_buf[t][r], u_buf[t][r], limit);
            }
        }
    }
}

template <int NT>
void run_dn_tile(
    int32_t down_type,
    const uint8_t* down_w,
    size_t row_bytes,
    int32_t n_in,
    const void* const* ff_ptrs,
    bool is_q8_1,
    float* const* out,
    int t_off,
    int32_t r0,
    int32_t r1,
    float weight_scale,
    int32_t accumulate
) {
    constexpr int kTile = 64;
    float d_buf[NT][kTile];
    float* dp[NT];
    for (int t = 0; t < NT; ++t) dp[t] = d_buf[t];
    for (int rr0 = r0; rr0 < r1; rr0 += kTile) {
        const int rr1 = std::min<int32_t>(r1, rr0 + kTile);
        const int nr = rr1 - rr0;
        dot_dispatch_nt<NT>(down_type, down_w, row_bytes, n_in, ff_ptrs + t_off, is_q8_1, rr0, rr1, dp);
        for (int t = 0; t < NT; ++t) {
            float* dst = out[t_off + t] + rr0;
            if (accumulate) {
                for (int r = 0; r < nr; ++r) dst[r] += weight_scale * d_buf[t][r];
            } else {
                for (int r = 0; r < nr; ++r) dst[r] = weight_scale * d_buf[t][r];
            }
        }
    }
}

void run_gu_common(
    int32_t gate_type,
    int32_t up_type,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t gate_rb,
    size_t up_rb,
    int32_t n_in,
    const void* const* x_ptrs,
    int32_t nt,
    bool is_q8_1,
    float* const* ff_out,
    int32_t r0,
    int32_t r1,
    bool is_oai,
    float alpha,
    float limit
) {
    if (nt <= 0 || r0 >= r1 || n_in <= 0) return;
    int t = 0;
    while (t + 8 <= nt) {
        run_gu_tile<8>(gate_type, up_type, gate_w, up_w, gate_rb, up_rb, n_in, x_ptrs, is_q8_1, ff_out, t, r0, r1, is_oai, alpha, limit);
        t += 8;
    }
    switch (nt - t) {
        case 1: run_gu_tile<1>(gate_type, up_type, gate_w, up_w, gate_rb, up_rb, n_in, x_ptrs, is_q8_1, ff_out, t, r0, r1, is_oai, alpha, limit); break;
        case 2: run_gu_tile<2>(gate_type, up_type, gate_w, up_w, gate_rb, up_rb, n_in, x_ptrs, is_q8_1, ff_out, t, r0, r1, is_oai, alpha, limit); break;
        case 3: run_gu_tile<3>(gate_type, up_type, gate_w, up_w, gate_rb, up_rb, n_in, x_ptrs, is_q8_1, ff_out, t, r0, r1, is_oai, alpha, limit); break;
        case 4: run_gu_tile<4>(gate_type, up_type, gate_w, up_w, gate_rb, up_rb, n_in, x_ptrs, is_q8_1, ff_out, t, r0, r1, is_oai, alpha, limit); break;
        case 5: run_gu_tile<5>(gate_type, up_type, gate_w, up_w, gate_rb, up_rb, n_in, x_ptrs, is_q8_1, ff_out, t, r0, r1, is_oai, alpha, limit); break;
        case 6: run_gu_tile<6>(gate_type, up_type, gate_w, up_w, gate_rb, up_rb, n_in, x_ptrs, is_q8_1, ff_out, t, r0, r1, is_oai, alpha, limit); break;
        case 7: run_gu_tile<7>(gate_type, up_type, gate_w, up_w, gate_rb, up_rb, n_in, x_ptrs, is_q8_1, ff_out, t, r0, r1, is_oai, alpha, limit); break;
        default: break;
    }
}

void run_dn_common(
    int32_t down_type,
    const uint8_t* down_w,
    size_t row_bytes,
    int32_t n_in,
    const void* const* ff_ptrs,
    int32_t nt,
    bool is_q8_1,
    float* const* out,
    int32_t r0,
    int32_t r1,
    float weight_scale,
    int32_t accumulate
) {
    if (nt <= 0 || r0 >= r1 || n_in <= 0) return;
    int t = 0;
    while (t + 8 <= nt) {
        run_dn_tile<8>(down_type, down_w, row_bytes, n_in, ff_ptrs, is_q8_1, out, t, r0, r1, weight_scale, accumulate);
        t += 8;
    }
    switch (nt - t) {
        case 1: run_dn_tile<1>(down_type, down_w, row_bytes, n_in, ff_ptrs, is_q8_1, out, t, r0, r1, weight_scale, accumulate); break;
        case 2: run_dn_tile<2>(down_type, down_w, row_bytes, n_in, ff_ptrs, is_q8_1, out, t, r0, r1, weight_scale, accumulate); break;
        case 3: run_dn_tile<3>(down_type, down_w, row_bytes, n_in, ff_ptrs, is_q8_1, out, t, r0, r1, weight_scale, accumulate); break;
        case 4: run_dn_tile<4>(down_type, down_w, row_bytes, n_in, ff_ptrs, is_q8_1, out, t, r0, r1, weight_scale, accumulate); break;
        case 5: run_dn_tile<5>(down_type, down_w, row_bytes, n_in, ff_ptrs, is_q8_1, out, t, r0, r1, weight_scale, accumulate); break;
        case 6: run_dn_tile<6>(down_type, down_w, row_bytes, n_in, ff_ptrs, is_q8_1, out, t, r0, r1, weight_scale, accumulate); break;
        case 7: run_dn_tile<7>(down_type, down_w, row_bytes, n_in, ff_ptrs, is_q8_1, out, t, r0, r1, weight_scale, accumulate); break;
        default: break;
    }
}

}  // namespace

extern "C" {

void ampllm_cpu_quantize_q8_k_avx2(const float* x, void* out_q8k, int32_t n) {
    if (n <= 0) return;
    const int nb = n / QK_K;
    const int rem = n - nb * QK_K;
    auto* y = reinterpret_cast<block_q8_K*>(out_q8k);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 lo = _mm256_set1_ps(-127.0f);
    const __m256 hi = _mm256_set1_ps(127.0f);

    for (int i = 0; i < nb; ++i) {
        const float* xp = x + (size_t)i * QK_K;
        __m256 mx = _mm256_setzero_ps();
        for (int c = 0; c < 32; ++c) {
            mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(xp + c * 8), abs_mask));
        }
        const float amax = hmax_f32_256(mx);
        const float d = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        y[i].d = d;
        const __m256 mul = _mm256_set1_ps(id);
        for (int c = 0; c < 16; ++c) {
            const __m256 v0 = _mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(xp + c * 16 + 0), mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), lo), hi);
            const __m256 v1 = _mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(_mm256_loadu_ps(xp + c * 16 + 8), mul), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), lo), hi);
            const __m256i i0 = _mm256_cvtps_epi32(v0);
            const __m256i i1 = _mm256_cvtps_epi32(v1);
            const __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1));
            const __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1));
            _mm_storeu_si128((__m128i*)(y[i].qs + c * 16), _mm_packs_epi16(p0, p1));
            const __m128i s16 = _mm_add_epi16(p0, p1);
            const __m128i s32 = _mm_madd_epi16(s16, _mm_set1_epi16(1));
            const __m128i r1 = _mm_hadd_epi32(s32, s32);
            const __m128i r2 = _mm_hadd_epi32(r1, r1);
            y[i].bsums[c] = (int16_t)_mm_cvtsi128_si32(r2);
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

void ampllm_cpu_quantize_q8_1_g8_avx2(const float* x, void* out_g8, int32_t n) {
    if (n <= 0) return;
    const int nb = (n + QK_K - 1) / QK_K;
    auto* dst = reinterpret_cast<block_q8_1_g8*>(out_g8);
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 lo = _mm256_set1_ps(-127.0f);
    const __m256 hi = _mm256_set1_ps(127.0f);

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
            float tmp[32];
            const float* p = x + off;
            if (rem < 32) {
                std::memset(tmp, 0, sizeof(tmp));
                std::memcpy(tmp, p, (size_t)rem * sizeof(float));
                p = tmp;
            }
            const __m256 f0 = _mm256_loadu_ps(p + 0);
            const __m256 f1 = _mm256_loadu_ps(p + 8);
            const __m256 f2 = _mm256_loadu_ps(p + 16);
            const __m256 f3 = _mm256_loadu_ps(p + 24);
            const __m256 m01 = _mm256_max_ps(_mm256_and_ps(f0, abs_mask), _mm256_and_ps(f1, abs_mask));
            const __m256 m23 = _mm256_max_ps(_mm256_and_ps(f2, abs_mask), _mm256_and_ps(f3, abs_mask));
            const float amax = hmax_f32_256(_mm256_max_ps(m01, m23));
            const float d = amax / 127.0f;
            const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
            const __m256 vid = _mm256_set1_ps(id);
            const __m256i i0 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(f0, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), lo), hi));
            const __m256i i1 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(f1, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), lo), hi));
            const __m256i i2 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(f2, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), lo), hi));
            const __m256i i3 = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_max_ps(_mm256_round_ps(_mm256_mul_ps(f3, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC), lo), hi));
            const __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1));
            const __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1));
            const __m128i p2 = _mm_packs_epi32(_mm256_castsi256_si128(i2), _mm256_extracti128_si256(i2, 1));
            const __m128i p3 = _mm_packs_epi32(_mm256_castsi256_si128(i3), _mm256_extracti128_si256(i3, 1));
            _mm_storeu_si128((__m128i*)(out.qs + k * 32 + 0), _mm_packs_epi16(p0, p1));
            _mm_storeu_si128((__m128i*)(out.qs + k * 32 + 16), _mm_packs_epi16(p2, p3));
            const int32_t sumq = hsum_i32_256(_mm256_add_epi32(_mm256_add_epi32(i0, i1), _mm256_add_epi32(i2, i3)));
            out.d[k] = d;
            out.bsum[k] = sumq;
            out.s[k] = d * (float)sumq;
        }
    }
}

void ampllm_cpu_bf16_to_q8_0_avx2(
    const uint16_t* src_bf16,
    const float* src_f32,
    size_t rows,
    int32_t k,
    uint8_t* out_q8_0
) {
    if (k <= 0 || (k & 31) != 0) return;
    const int nb = k / 32;
    const __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
    const __m256 clamp_hi = _mm256_set1_ps(127.0f);
    const __m256 clamp_lo = _mm256_set1_ps(-127.0f);

    for (size_t r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const size_t at = r * (size_t)k + (size_t)b * 32;
            __m256 v0, v1, v2, v3;
            if (src_f32 != nullptr) {
                v0 = _mm256_loadu_ps(src_f32 + at + 0);
                v1 = _mm256_loadu_ps(src_f32 + at + 8);
                v2 = _mm256_loadu_ps(src_f32 + at + 16);
                v3 = _mm256_loadu_ps(src_f32 + at + 24);
            } else {
                v0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(src_bf16 + at + 0))), 16));
                v1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(src_bf16 + at + 8))), 16));
                v2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(src_bf16 + at + 16))), 16));
                v3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(src_bf16 + at + 24))), 16));
            }
            const __m256 m01 = _mm256_max_ps(_mm256_and_ps(v0, abs_mask), _mm256_and_ps(v1, abs_mask));
            const __m256 m23 = _mm256_max_ps(_mm256_and_ps(v2, abs_mask), _mm256_and_ps(v3, abs_mask));
            const float am = hmax_f32_256(_mm256_max_ps(m01, m23));
            const float d = am / 127.0f;
            const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
            uint8_t* dst = out_q8_0 + (r * (size_t)nb + (size_t)b) * sizeof(block_q8_0);
            const uint16_t hd = f2h_rne(d);
            std::memcpy(dst, &hd, 2);

            const __m256 vid = _mm256_set1_ps(id);
            const __m256i i0 = _mm256_cvtps_epi32(_mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, _mm256_mul_ps(v0, vid))));
            const __m256i i1 = _mm256_cvtps_epi32(_mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, _mm256_mul_ps(v1, vid))));
            const __m256i i2 = _mm256_cvtps_epi32(_mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, _mm256_mul_ps(v2, vid))));
            const __m256i i3 = _mm256_cvtps_epi32(_mm256_max_ps(clamp_lo, _mm256_min_ps(clamp_hi, _mm256_mul_ps(v3, vid))));
            const __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1));
            const __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1));
            const __m128i p2 = _mm_packs_epi32(_mm256_castsi256_si128(i2), _mm256_extracti128_si256(i2, 1));
            const __m128i p3 = _mm_packs_epi32(_mm256_castsi256_si128(i3), _mm256_extracti128_si256(i3, 1));
            _mm_storeu_si128((__m128i*)(dst + 2), _mm_packs_epi16(p0, p1));
            _mm_storeu_si128((__m128i*)(dst + 18), _mm_packs_epi16(p2, p3));
        }
    }
}

void ampllm_cpu_tok_row_to_f32_avx2(
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
            int i = 0;
            for (; i + 8 <= k; i += 8) {
                _mm256_storeu_ps(dst_f32 + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)(p + i))));
            }
            for (; i < k; ++i) dst_f32[i] = h2f(p[i]);
            break;
        }
        case GGML_TYPE_BF16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            int i = 0;
            for (; i + 8 <= k; i += 8) {
                __m128i h = _mm_loadu_si128((const __m128i*)(p + i));
                _mm256_storeu_ps(dst_f32 + i, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)));
            }
            for (; i < k; ++i) dst_f32[i] = bf16_to_f32_scalar(p[i]);
            break;
        }
        case GGML_TYPE_Q8_0: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_q8_0*>(src);
            for (int b = 0; b < nb; ++b) {
                const __m256 vd = _mm256_set1_ps(h2f(blks[b].d));
                for (int s = 0; s < 4; ++s) {
                    const __m256 f = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(blks[b].qs + s * 8))));
                    _mm256_storeu_ps(dst_f32 + b * 32 + s * 8, _mm256_mul_ps(vd, f));
                }
            }
            break;
        }
        case GGML_TYPE_IQ4_NL: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_iq4_nl*>(src);
            const __m128i lut = _mm_loadu_si128((const __m128i*)KVALUES_IQ4NL);
            const __m128i m0f = _mm_set1_epi8(0x0F);
            for (int b = 0; b < nb; ++b) {
                const __m256 vd = _mm256_set1_ps(h2f(blks[b].d));
                const __m128i q = _mm_loadu_si128((const __m128i*)blks[b].qs);
                const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m0f));
                const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m0f));
                _mm256_storeu_ps(dst_f32 + b * 32 + 0,  _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo))));
                _mm256_storeu_ps(dst_f32 + b * 32 + 8,  _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)))));
                _mm256_storeu_ps(dst_f32 + b * 32 + 16, _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi))));
                _mm256_storeu_ps(dst_f32 + b * 32 + 24, _mm256_mul_ps(vd, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)))));
            }
            break;
        }
        case GGML_TYPE_Q4_0: {
            const int nb = k / 32;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* bp = src + (size_t)b * 18;
                const float d = h2f(u16(bp));
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
                const float d = h2f(blks[b].d);
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
                const float d = h2f(blks[b].d);
                const float m = h2f(blks[b].m);
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
        case GGML_TYPE_Q2_0: {
            const int nb = k / 64;
            const auto* blks = reinterpret_cast<const block_q2_0*>(src);
            for (int b = 0; b < nb; ++b) {
                const float d = h2f(blks[b].d);
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
                const float d_all = h2f(blks[b].d);
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
                        for (int l = 0; l < 16; ++l) y[n + j * 32 + l] = dl0 * (float)((int)((q[l] >> shift) & 3) - ((blks[b].hmask[l] & m) ? 0 : 4));
                        const float dl1 = d_all * (float)(scales[is++] - 32);
                        for (int l = 0; l < 16; ++l) y[n + j * 32 + 16 + l] = dl1 * (float)((int)((q[l + 16] >> shift) & 3) - ((blks[b].hmask[l + 16] & m) ? 0 : 4));
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
                const float d = h2f(blks[b].d), dm = h2f(blks[b].dmin);
                for (int ib = 0; ib < 4; ++ib) {
                    const int i0 = ib * 2, i1 = ib * 2 + 1;
                    const uint8_t sc0 = i0 < 4 ? (blks[b].scales[i0] & 63) : ((blks[b].scales[i0 + 4] & 0xF) | ((blks[b].scales[i0 - 4] >> 6) << 4));
                    const uint8_t mn0 = i0 < 4 ? (blks[b].scales[i0 + 4] & 63) : ((blks[b].scales[i0 + 4] >> 4) | ((blks[b].scales[i0] >> 6) << 4));
                    const uint8_t sc1 = i1 < 4 ? (blks[b].scales[i1] & 63) : ((blks[b].scales[i1 + 4] & 0xF) | ((blks[b].scales[i1 - 4] >> 6) << 4));
                    const uint8_t mn1 = i1 < 4 ? (blks[b].scales[i1 + 4] & 63) : ((blks[b].scales[i1 + 4] >> 4) | ((blks[b].scales[i1] >> 6) << 4));
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
                const float d = h2f(blks[b].d), dm = h2f(blks[b].dmin);
                for (int ib = 0; ib < 4; ++ib) {
                    const int i0 = ib * 2, i1 = ib * 2 + 1;
                    const uint8_t sc0 = i0 < 4 ? (blks[b].scales[i0] & 63) : ((blks[b].scales[i0 + 4] & 0xF) | ((blks[b].scales[i0 - 4] >> 6) << 4));
                    const uint8_t mn0 = i0 < 4 ? (blks[b].scales[i0 + 4] & 63) : ((blks[b].scales[i0 + 4] >> 4) | ((blks[b].scales[i0] >> 6) << 4));
                    const uint8_t sc1 = i1 < 4 ? (blks[b].scales[i1] & 63) : ((blks[b].scales[i1 + 4] & 0xF) | ((blks[b].scales[i1 - 4] >> 6) << 4));
                    const uint8_t mn1 = i1 < 4 ? (blks[b].scales[i1 + 4] & 63) : ((blks[b].scales[i1 + 4] >> 4) | ((blks[b].scales[i1] >> 6) << 4));
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
                const float d = h2f(blks[b].d);
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
                const float d = h2f(blks[b].d);
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

void ampllm_cpu_hy4_fused_gu_rows_avx2(
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
    run_gu_common(ggml_type, ggml_type, gate_w, up_w, row_bytes, row_bytes, n_in, x_q8k, nt, false, ff_out, r0, r1, false, 1.0f, swiglu_limit);
}

void ampllm_cpu_hy4_fused_gu_rows_q8_1_avx2(
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
    run_gu_common(ggml_type, ggml_type, gate_w, up_w, row_bytes, row_bytes, n_in, x_q8_1, nt, true, ff_out, r0, r1, false, 1.0f, swiglu_limit);
}

void ampllm_cpu_hy4_down_rows_avx2(
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
    run_dn_common(ggml_type, down_w, row_bytes, n_in, ff_q8k, nt, false, out, r0, r1, weight_scale, accumulate);
}

void ampllm_cpu_hy4_down_rows_q8_1_avx2(
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
    run_dn_common(ggml_type, down_w, row_bytes, n_in, ff_q8_1, nt, true, out, r0, r1, weight_scale, accumulate);
}

void ampllm_cpu_iq_fused_gu_rows_avx2(
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
    run_gu_common(ggml_type, ggml_type, gate_w, up_w, row_bytes, row_bytes, n_in, x_q8k, nt, false, ff_out, r0, r1, false, 1.0f, swiglu_limit);
}

void ampllm_cpu_iq_down_rows_avx2(
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
    run_dn_common(ggml_type, down_w, row_bytes, n_in, ff_q8k, nt, false, out, r0, r1, weight_scale, accumulate);
}

void ampllm_cpu_mm3_quantize_q8_k_avx2(const float* x, void* out_q8k, int32_t n) {
    ampllm_cpu_quantize_q8_k_avx2(x, out_q8k, n);
}

void ampllm_cpu_mm3_fused_gu_rows_avx2(
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
    run_gu_common(gate_type, up_type, gate_w, up_w, gate_row_bytes, up_row_bytes, n_in, x_q8k, nt, false, ff_out, r0, r1, true, swiglu_alpha, swiglu_limit);
}

void ampllm_cpu_mm3_down_rows_avx2(
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
    run_dn_common(down_type, down_w, row_bytes, n_in, ff_q8k, nt, false, out, r0, r1, weight_scale, accumulate);
}

void ampllm_cpu_mm3_q6_k_row_to_f32_avx2(const uint8_t* src, int32_t k, float* dst_f32) {
    ampllm_cpu_tok_row_to_f32_avx2(GGML_TYPE_Q6_K, src, k, dst_f32);
}

}  // extern "C"
