#include <immintrin.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "iq_tables.hpp"

namespace {

static constexpr int QK_K = 256;
static constexpr int kPrefetchAhead = 2048;

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K size mismatch");

struct block_q6_K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size mismatch");

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

struct block_iq2_s {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
};
static_assert(sizeof(block_iq2_s) == 82, "block_iq2_s size mismatch");

inline float h2f(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}
inline uint16_t u16(const uint8_t* p) {
    uint16_t v;
    std::memcpy(&v, p, 2);
    return v;
}
inline uint32_t u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint64_t u64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(
        3, 3, 3, 3, 3, 3, 3, 3,
        2, 2, 2, 2, 2, 2, 2, 2,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t)(uint16_t)s0
        | ((uint64_t)(uint16_t)s1 << 16)
        | ((uint64_t)(uint16_t)s2 << 32)
        | ((uint64_t)(uint16_t)s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long)packed));
}

inline __mmask64 sign_mask_from_idx(__m256i si) {
    const __m256i par = _mm256_slli_epi32(
        _mm256_and_si256(_mm256_popcnt_epi32(si), _mm256_set1_epi32(1)), 7);
    return _cvtu64_mask64(
        (uint64_t)_mm_cvtsi128_si64(_mm256_cvtepi32_epi8(_mm256_or_si256(si, par))));
}

inline __m256i sign_idx_xxs(uint32_t lo4, uint32_t hi4) {
    const __m256i w = _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_set1_epi32((int)lo4)),
        _mm_set1_epi32((int)hi4), 1);
    return _mm256_and_si256(
        _mm256_srlv_epi32(w, _mm256_setr_epi32(0, 7, 14, 21, 0, 7, 14, 21)),
        _mm256_set1_epi32(127));
}

inline __m256i fold256(__m512i a) {
    return _mm256_add_epi32(_mm512_castsi512_si256(a), _mm512_extracti64x4_epi64(a, 1));
}

inline __m256i add_delta_lane0(__m256i v, int32_t delta) {
    return _mm256_add_epi32(v, _mm256_castsi128_si256(_mm_cvtsi32_si128(delta)));
}

inline float hsum8_f32(__m256 x) {
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

inline float swoai(float g, float u, float alpha, float limit) {
    if (limit > 0.0f) {
        g = std::fmin(g, limit);
        u = std::fmin(std::fmax(u, -limit), limit);
    }
    return (g / (1.0f + std::exp(-alpha * g))) * (u + 1.0f);
}

template <int TY> struct Mm3Fmt;

template <> struct Mm3Fmt<16> {  // IQ2_XXS
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static constexpr bool has_delta = false;
    static inline int32_t delta_block(const uint8_t*, const block_q8_K&) { return 0; }
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* p = b + 2 + 16 * j;
        const uint32_t aux0_0 = u32(p + 0);
        const uint32_t aux1_0 = u32(p + 4);
        const uint32_t aux0_1 = u32(p + 8);
        const uint32_t aux1_1 = u32(p + 12);
        g = _mm512_set_epi64(
            (long long)iq2xxs_grid[(aux0_1 >> 24) & 0xFF],
            (long long)iq2xxs_grid[(aux0_1 >> 16) & 0xFF],
            (long long)iq2xxs_grid[(aux0_1 >> 8) & 0xFF],
            (long long)iq2xxs_grid[aux0_1 & 0xFF],
            (long long)iq2xxs_grid[(aux0_0 >> 24) & 0xFF],
            (long long)iq2xxs_grid[(aux0_0 >> 16) & 0xFF],
            (long long)iq2xxs_grid[(aux0_0 >> 8) & 0xFF],
            (long long)iq2xxs_grid[aux0_0 & 0xFF]);
        m = sign_mask_from_idx(sign_idx_xxs(aux1_0, aux1_1));
        const int s0 = 2 * (int)(aux1_0 >> 28) + 1;
        const int s1 = 2 * (int)(aux1_1 >> 28) + 1;
        sc = scales4(s0, s0, s1, s1);
    }
};

template <> struct Mm3Fmt<17> {  // IQ2_XS
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static constexpr bool has_delta = false;
    static inline int32_t delta_block(const uint8_t*, const block_q8_K&) { return 0; }
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* p = b + 2 + 16 * j;
        const uint16_t q0 = u16(p + 0), q1 = u16(p + 2), q2 = u16(p + 4), q3 = u16(p + 6);
        const uint16_t q4 = u16(p + 8), q5 = u16(p + 10), q6 = u16(p + 12), q7 = u16(p + 14);
        g = _mm512_set_epi64(
            (long long)iq2xs_grid[q7 & 511], (long long)iq2xs_grid[q6 & 511],
            (long long)iq2xs_grid[q5 & 511], (long long)iq2xs_grid[q4 & 511],
            (long long)iq2xs_grid[q3 & 511], (long long)iq2xs_grid[q2 & 511],
            (long long)iq2xs_grid[q1 & 511], (long long)iq2xs_grid[q0 & 511]);
        const __m256i si = _mm256_setr_epi32(
            q0 >> 9, q1 >> 9, q2 >> 9, q3 >> 9, q4 >> 9, q5 >> 9, q6 >> 9, q7 >> 9);
        m = sign_mask_from_idx(si);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(
            2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1,
            2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Mm3Fmt<18> {  // IQ3_XXS
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static constexpr bool has_delta = false;
    static inline int32_t delta_block(const uint8_t*, const block_q8_K&) { return 0; }
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        g = _mm512_set_epi32(
            (int)iq3xxs_grid[q[15]], (int)iq3xxs_grid[q[14]], (int)iq3xxs_grid[q[13]], (int)iq3xxs_grid[q[12]],
            (int)iq3xxs_grid[q[11]], (int)iq3xxs_grid[q[10]], (int)iq3xxs_grid[q[9]],  (int)iq3xxs_grid[q[8]],
            (int)iq3xxs_grid[q[7]],  (int)iq3xxs_grid[q[6]],  (int)iq3xxs_grid[q[5]],  (int)iq3xxs_grid[q[4]],
            (int)iq3xxs_grid[q[3]],  (int)iq3xxs_grid[q[2]],  (int)iq3xxs_grid[q[1]],  (int)iq3xxs_grid[q[0]]);
        const uint32_t aux0 = u32(b + 66 + 8 * j), aux1 = u32(b + 70 + 8 * j);
        m = sign_mask_from_idx(sign_idx_xxs(aux0, aux1));
        const int s0 = 2 * (int)(aux0 >> 28) + 1, s1 = 2 * (int)(aux1 >> 28) + 1;
        sc = scales4(s0, s0, s1, s1);
    }
};

template <> struct Mm3Fmt<22> {  // IQ2_S
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static constexpr bool has_delta = false;
    static inline int32_t delta_block(const uint8_t*, const block_q8_K&) { return 0; }
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        g = _mm512_set_epi64(
            (long long)iq2s_grid[qs[7] | ((h1 << 2) & 0x300)],
            (long long)iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
            (long long)iq2s_grid[qs[5] | ((h1 << 6) & 0x300)],
            (long long)iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
            (long long)iq2s_grid[qs[3] | ((h0 << 2) & 0x300)],
            (long long)iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
            (long long)iq2s_grid[qs[1] | ((h0 << 6) & 0x300)],
            (long long)iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(
            2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1,
            2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Mm3Fmt<19> {  // IQ1_S
    static constexpr int bytes = 50;
    static constexpr float K = 0.125f;
    static constexpr bool has_delta = true;
    static inline int32_t delta_block(const uint8_t* b, const block_q8_K& y) {
        int32_t acc = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const uint16_t qh = u16(b + 34 + 2 * ib);
            const int32_t sc_ib = 2 * (int32_t)((qh >> 12) & 7) + 1;
            const int32_t sgn = (qh & 0x8000u) ? -1 : 1;
            const int32_t bsum = (int32_t)y.bsums[2 * ib] + (int32_t)y.bsums[2 * ib + 1];
            acc += sc_ib * sgn * bsum;
        }
        return acc;
    }
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint16_t qh0 = u16(b + 34 + 4 * j);
        const uint16_t qh1 = u16(b + 34 + 4 * j + 2);
        const uint8_t* qs0 = b + 2 + 8 * j;
        const __m512i g_s8 = _mm512_set_epi64(
            (long long)iq1s_grid[qs0[7] | (((qh1 >> 9) & 7) << 8)],
            (long long)iq1s_grid[qs0[6] | (((qh1 >> 6) & 7) << 8)],
            (long long)iq1s_grid[qs0[5] | (((qh1 >> 3) & 7) << 8)],
            (long long)iq1s_grid[qs0[4] | (((qh1 >> 0) & 7) << 8)],
            (long long)iq1s_grid[qs0[3] | (((qh0 >> 9) & 7) << 8)],
            (long long)iq1s_grid[qs0[2] | (((qh0 >> 6) & 7) << 8)],
            (long long)iq1s_grid[qs0[1] | (((qh0 >> 3) & 7) << 8)],
            (long long)iq1s_grid[qs0[0] | (((qh0 >> 0) & 7) << 8)]);
        g = _mm512_and_si512(g_s8, _mm512_set1_epi8(1));
        m = _mm512_movepi8_mask(g_s8);
        const int sc0 = 8 * (2 * (int)((qh0 >> 12) & 7) + 1);
        const int sc1 = 8 * (2 * (int)((qh1 >> 12) & 7) + 1);
        sc = scales4(sc0, sc0, sc1, sc1);
    }
};

template <int TY_G, int TY_U, int NT>
void gu_fused_iq(
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t gate_rb,
    size_t up_rb,
    int n,
    const void* const* act,
    float* const* ff,
    int r0,
    int r1,
    float alpha,
    float limit
) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*)act[t];
    const int nb = n / QK_K;
    const __m512i zero = _mm512_setzero_si512();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate_w + (size_t)r * gate_rb;
        const uint8_t* ur = up_w + (size_t)r * up_rb;
        __m256 ag[NT], au[NT];
        for (int t = 0; t < NT; ++t) {
            ag[t] = _mm256_setzero_ps();
            au[t] = _mm256_setzero_ps();
        }
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gr + (size_t)i * Mm3Fmt<TY_G>::bytes;
            const uint8_t* ub = ur + (size_t)i * Mm3Fmt<TY_U>::bytes;
            if (kPrefetchAhead > 0) {
                _mm_prefetch((const char*)gb + kPrefetchAhead, _MM_HINT_T0);
                _mm_prefetch((const char*)ub + kPrefetchAhead, _MM_HINT_T0);
                _mm_prefetch((const char*)gb + kPrefetchAhead + 64, _MM_HINT_T0);
                _mm_prefetch((const char*)ub + kPrefetchAhead + 64, _MM_HINT_T0);
            }
            __m512i ig[NT], iu[NT];
            for (int t = 0; t < NT; ++t) {
                ig[t] = zero;
                iu[t] = zero;
            }
            for (int j = 0; j < 4; ++j) {
                __m512i gg, gs, ug, us;
                __mmask64 gm, um;
                Mm3Fmt<TY_G>::decode(gb, j, gg, gm, gs);
                Mm3Fmt<TY_U>::decode(ub, j, ug, um, us);
                for (int t = 0; t < NT; ++t) {
                    const __m512i yv = _mm512_loadu_si512((const void*)(y[t][i].qs + 64 * j));
                    ig[t] = _mm512_dpwssd_epi32(
                        ig[t], _mm512_maddubs_epi16(gg, _mm512_mask_sub_epi8(yv, gm, zero, yv)), gs);
                    iu[t] = _mm512_dpwssd_epi32(
                        iu[t], _mm512_maddubs_epi16(ug, _mm512_mask_sub_epi8(yv, um, zero, yv)), us);
                }
            }
            const float dg = h2f(u16(gb)), du = h2f(u16(ub));
            for (int t = 0; t < NT; ++t) {
                __m256i fg = fold256(ig[t]);
                __m256i fu = fold256(iu[t]);
                if constexpr (Mm3Fmt<TY_G>::has_delta) {
                    fg = add_delta_lane0(fg, Mm3Fmt<TY_G>::delta_block(gb, y[t][i]));
                }
                if constexpr (Mm3Fmt<TY_U>::has_delta) {
                    fu = add_delta_lane0(fu, Mm3Fmt<TY_U>::delta_block(ub, y[t][i]));
                }
                const float yd = y[t][i].d;
                ag[t] = _mm256_fmadd_ps(_mm256_set1_ps(dg * yd), _mm256_cvtepi32_ps(fg), ag[t]);
                au[t] = _mm256_fmadd_ps(_mm256_set1_ps(du * yd), _mm256_cvtepi32_ps(fu), au[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float g = Mm3Fmt<TY_G>::K * hsum8_f32(ag[t]);
            const float u = Mm3Fmt<TY_U>::K * hsum8_f32(au[t]);
            ff[t][r] = swoai(g, u, alpha, limit);
        }
    }
}

template <int TY_G, int TY_U>
void gu_fused_iq_wide(
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t gate_rb,
    size_t up_rb,
    int n,
    const void* const* act,
    int T,
    float* const* ff,
    int r0,
    int r1,
    float alpha,
    float limit
) {
    constexpr int kTile = 16, kRows = 4;
    const int nb = n / QK_K;
    static thread_local std::vector<uint8_t> ubuf;
    static thread_local std::vector<float> accg, accu;
    ubuf.resize((size_t)(T < kTile ? T : kTile) * n);
    accg.resize((size_t)kRows * kTile * 8);
    accu.resize((size_t)kRows * kTile * 8);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i one8 = _mm512_set1_epi8(1);
    const __m512i flip = _mm512_set1_epi8((char)0x80);
    for (int t0 = 0; t0 < T; t0 += kTile) {
        const int tt = T - t0 < kTile ? T - t0 : kTile;
        const block_q8_K* y[kTile];
        for (int t = 0; t < tt; ++t) {
            y[t] = (const block_q8_K*)act[t0 + t];
            uint8_t* u = ubuf.data() + (size_t)t * n;
            for (int i = 0; i < nb; ++i)
                for (int j = 0; j < 4; ++j)
                    _mm512_storeu_si512(
                        (void*)(u + (size_t)i * 256 + 64 * j),
                        _mm512_xor_si512(
                            _mm512_loadu_si512((const void*)(y[t][i].qs + 64 * j)), flip));
        }
        for (int rg = r0; rg < r1; rg += kRows) {
            const int rr = r1 - rg < kRows ? r1 - rg : kRows;
            for (int k = 0; k < rr * tt; ++k) {
                _mm256_storeu_ps(&accg[(size_t)k * 8], _mm256_setzero_ps());
                _mm256_storeu_ps(&accu[(size_t)k * 8], _mm256_setzero_ps());
            }
            for (int i = 0; i < nb; ++i) {
                for (int q = 0; q < rr; ++q) {
                    const uint8_t* gblk = gate_w + (size_t)(rg + q) * gate_rb + (size_t)i * Mm3Fmt<TY_G>::bytes;
                    const uint8_t* ublk = up_w + (size_t)(rg + q) * up_rb + (size_t)i * Mm3Fmt<TY_U>::bytes;
                    if (kPrefetchAhead > 0) {
                        _mm_prefetch((const char*)gblk + kPrefetchAhead, _MM_HINT_T0);
                        _mm_prefetch((const char*)gblk + kPrefetchAhead + 64, _MM_HINT_T0);
                        _mm_prefetch((const char*)ublk + kPrefetchAhead, _MM_HINT_T0);
                        _mm_prefetch((const char*)ublk + kPrefetchAhead + 64, _MM_HINT_T0);
                    }
                    __m512i gw8[4], gsc[4], uw8[4], usc[4];
                    __m512i gcorr = zero, ucorr = zero;
                    for (int j = 0; j < 4; ++j) {
                        __m512i gg, gs, ug, us;
                        __mmask64 gm, um;
                        Mm3Fmt<TY_G>::decode(gblk, j, gg, gm, gs);
                        Mm3Fmt<TY_U>::decode(ublk, j, ug, um, us);
                        gw8[j] = _mm512_mask_sub_epi8(gg, gm, zero, gg);
                        gsc[j] = gs;
                        gcorr = _mm512_dpwssd_epi32(gcorr, _mm512_maddubs_epi16(one8, gw8[j]), gs);
                        uw8[j] = _mm512_mask_sub_epi8(ug, um, zero, ug);
                        usc[j] = us;
                        ucorr = _mm512_dpwssd_epi32(ucorr, _mm512_maddubs_epi16(one8, uw8[j]), us);
                    }
                    const __m256i gcorr128 = _mm256_slli_epi32(fold256(gcorr), 7);
                    const __m256i ucorr128 = _mm256_slli_epi32(fold256(ucorr), 7);
                    const float dg = h2f(u16(gblk));
                    const float du = h2f(u16(ublk));
                    float* accgq = &accg[(size_t)q * tt * 8];
                    float* accuq = &accu[(size_t)q * tt * 8];
                    for (int t = 0; t < tt; ++t) {
                        const uint8_t* u = ubuf.data() + (size_t)t * n + (size_t)i * 256;
                        __m512i ag = zero, au = zero;
                        for (int j = 0; j < 4; ++j) {
                            const __m512i uv = _mm512_loadu_si512((const void*)(u + 64 * j));
                            ag = _mm512_dpwssd_epi32(ag, _mm512_maddubs_epi16(uv, gw8[j]), gsc[j]);
                            au = _mm512_dpwssd_epi32(au, _mm512_maddubs_epi16(uv, uw8[j]), usc[j]);
                        }
                        __m256i sg = _mm256_sub_epi32(fold256(ag), gcorr128);
                        __m256i su = _mm256_sub_epi32(fold256(au), ucorr128);
                        if constexpr (Mm3Fmt<TY_G>::has_delta) {
                            sg = add_delta_lane0(sg, Mm3Fmt<TY_G>::delta_block(gblk, y[t][i]));
                        }
                        if constexpr (Mm3Fmt<TY_U>::has_delta) {
                            su = add_delta_lane0(su, Mm3Fmt<TY_U>::delta_block(ublk, y[t][i]));
                        }
                        const float yd = y[t][i].d;
                        float* agp = accgq + (size_t)t * 8;
                        float* aup = accuq + (size_t)t * 8;
                        _mm256_storeu_ps(
                            agp, _mm256_fmadd_ps(_mm256_set1_ps(dg * yd), _mm256_cvtepi32_ps(sg), _mm256_loadu_ps(agp)));
                        _mm256_storeu_ps(
                            aup, _mm256_fmadd_ps(_mm256_set1_ps(du * yd), _mm256_cvtepi32_ps(su), _mm256_loadu_ps(aup)));
                    }
                }
            }
            for (int q = 0; q < rr; ++q) {
                for (int t = 0; t < tt; ++t) {
                    const size_t idx = ((size_t)q * tt + t) * 8;
                    const float g = Mm3Fmt<TY_G>::K * hsum8_f32(_mm256_loadu_ps(&accg[idx]));
                    const float u = Mm3Fmt<TY_U>::K * hsum8_f32(_mm256_loadu_ps(&accu[idx]));
                    ff[t0 + t][rg + q] = swoai(g, u, alpha, limit);
                }
            }
        }
    }
}

template <int TY, int NT>
void down_iq(
    const uint8_t* w,
    size_t row_bytes,
    int n,
    const void* const* act,
    float* const* out,
    int r0,
    int r1,
    float weight_scale,
    int accumulate
) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*)act[t];
    const int nb = n / QK_K;
    const __m512i zero = _mm512_setzero_si512();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t)i * Mm3Fmt<TY>::bytes;
            if (kPrefetchAhead > 0) {
                _mm_prefetch((const char*)blk + kPrefetchAhead, _MM_HINT_T0);
                _mm_prefetch((const char*)blk + kPrefetchAhead + 64, _MM_HINT_T0);
            }
            __m512i acci[NT];
            for (int t = 0; t < NT; ++t) acci[t] = zero;
            for (int j = 0; j < 4; ++j) {
                __m512i g, sc;
                __mmask64 m;
                Mm3Fmt<TY>::decode(blk, j, g, m, sc);
                for (int t = 0; t < NT; ++t) {
                    const __m512i yv = _mm512_loadu_si512((const void*)(y[t][i].qs + 64 * j));
                    const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                    acci[t] = _mm512_dpwssd_epi32(acci[t], _mm512_maddubs_epi16(g, ys), sc);
                }
            }
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                __m256i fi = fold256(acci[t]);
                if constexpr (Mm3Fmt<TY>::has_delta) {
                    fi = add_delta_lane0(fi, Mm3Fmt<TY>::delta_block(blk, y[t][i]));
                }
                accf[t] = _mm256_fmadd_ps(
                    _mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fi), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float val = weight_scale * Mm3Fmt<TY>::K * hsum8_f32(accf[t]);
            if (accumulate) out[t][r] += val;
            else out[t][r] = val;
        }
    }
}

template <int TY>
void down_iq_wide(
    const uint8_t* w,
    size_t row_bytes,
    int n,
    const void* const* act,
    int T,
    float* const* out,
    int r0,
    int r1,
    float weight_scale,
    int accumulate
) {
    constexpr int kTile = 16, kRows = 8;
    const int nb = n / QK_K;
    static thread_local std::vector<uint8_t> ubuf;
    static thread_local std::vector<float> accb;
    ubuf.resize((size_t)(T < kTile ? T : kTile) * n);
    accb.resize((size_t)kRows * kTile * 8);
    const __m512i zero = _mm512_setzero_si512();
    const __m512i one8 = _mm512_set1_epi8(1);
    const __m512i flip = _mm512_set1_epi8((char)0x80);
    for (int t0 = 0; t0 < T; t0 += kTile) {
        const int tt = T - t0 < kTile ? T - t0 : kTile;
        const block_q8_K* y[kTile];
        for (int t = 0; t < tt; ++t) {
            y[t] = (const block_q8_K*)act[t0 + t];
            uint8_t* u = ubuf.data() + (size_t)t * n;
            for (int i = 0; i < nb; ++i)
                for (int j = 0; j < 4; ++j)
                    _mm512_storeu_si512(
                        (void*)(u + (size_t)i * 256 + 64 * j),
                        _mm512_xor_si512(
                            _mm512_loadu_si512((const void*)(y[t][i].qs + 64 * j)), flip));
        }
        for (int rg = r0; rg < r1; rg += kRows) {
            const int rr = r1 - rg < kRows ? r1 - rg : kRows;
            for (int k = 0; k < rr * tt; ++k) _mm256_storeu_ps(&accb[(size_t)k * 8], _mm256_setzero_ps());
            for (int i = 0; i < nb; ++i) {
                for (int q = 0; q < rr; ++q) {
                    const uint8_t* blk = w + (size_t)(rg + q) * row_bytes + (size_t)i * Mm3Fmt<TY>::bytes;
                    if (kPrefetchAhead > 0) {
                        _mm_prefetch((const char*)blk + kPrefetchAhead, _MM_HINT_T0);
                        _mm_prefetch((const char*)blk + kPrefetchAhead + 64, _MM_HINT_T0);
                    }
                    __m512i w8[4], sc[4];
                    __m512i corr = zero;
                    for (int j = 0; j < 4; ++j) {
                        __m512i g, s;
                        __mmask64 m;
                        Mm3Fmt<TY>::decode(blk, j, g, m, s);
                        w8[j] = _mm512_mask_sub_epi8(g, m, zero, g);
                        sc[j] = s;
                        corr = _mm512_dpwssd_epi32(corr, _mm512_maddubs_epi16(one8, w8[j]), s);
                    }
                    const __m256i corr128 = _mm256_slli_epi32(fold256(corr), 7);
                    const float dx = h2f(u16(blk));
                    float* accq = &accb[(size_t)q * tt * 8];
                    for (int t = 0; t < tt; ++t) {
                        const uint8_t* u = ubuf.data() + (size_t)t * n + (size_t)i * 256;
                        __m512i a = zero;
                        for (int j = 0; j < 4; ++j)
                            a = _mm512_dpwssd_epi32(
                                a, _mm512_maddubs_epi16(_mm512_loadu_si512((const void*)(u + 64 * j)), w8[j]), sc[j]);
                        __m256i s = _mm256_sub_epi32(fold256(a), corr128);
                        if constexpr (Mm3Fmt<TY>::has_delta) {
                            s = add_delta_lane0(s, Mm3Fmt<TY>::delta_block(blk, y[t][i]));
                        }
                        float* ap = accq + (size_t)t * 8;
                        _mm256_storeu_ps(
                            ap, _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(s), _mm256_loadu_ps(ap)));
                    }
                }
            }
            for (int q = 0; q < rr; ++q) {
                for (int t = 0; t < tt; ++t) {
                    const float val = weight_scale * Mm3Fmt<TY>::K *
                        hsum8_f32(_mm256_loadu_ps(&accb[((size_t)q * tt + t) * 8]));
                    if (accumulate) out[t0 + t][rg + q] += val;
                    else out[t0 + t][rg + q] = val;
                }
            }
        }
    }
}

template <int TY_G, int TY_U>
void gu_fused_nt(
    int nt,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t gate_rb,
    size_t up_rb,
    int n,
    const void* const* act,
    float* const* ff,
    int r0,
    int r1,
    float alpha,
    float limit
) {
    switch (nt) {
        case 1: gu_fused_iq<TY_G, TY_U, 1>(gate_w, up_w, gate_rb, up_rb, n, act, ff, r0, r1, alpha, limit); break;
        case 2: gu_fused_iq<TY_G, TY_U, 2>(gate_w, up_w, gate_rb, up_rb, n, act, ff, r0, r1, alpha, limit); break;
        case 3: gu_fused_iq<TY_G, TY_U, 3>(gate_w, up_w, gate_rb, up_rb, n, act, ff, r0, r1, alpha, limit); break;
        case 4: gu_fused_iq<TY_G, TY_U, 4>(gate_w, up_w, gate_rb, up_rb, n, act, ff, r0, r1, alpha, limit); break;
        case 5: gu_fused_iq<TY_G, TY_U, 5>(gate_w, up_w, gate_rb, up_rb, n, act, ff, r0, r1, alpha, limit); break;
        case 6: gu_fused_iq<TY_G, TY_U, 6>(gate_w, up_w, gate_rb, up_rb, n, act, ff, r0, r1, alpha, limit); break;
        default: gu_fused_iq_wide<TY_G, TY_U>(gate_w, up_w, gate_rb, up_rb, n, act, nt, ff, r0, r1, alpha, limit); break;
    }
}

template <int TY>
void down_nt(
    int nt,
    const uint8_t* w,
    size_t row_bytes,
    int n,
    const void* const* act,
    float* const* out,
    int r0,
    int r1,
    float weight_scale,
    int accumulate
) {
    switch (nt) {
        case 1: down_iq<TY, 1>(w, row_bytes, n, act, out, r0, r1, weight_scale, accumulate); break;
        case 2: down_iq<TY, 2>(w, row_bytes, n, act, out, r0, r1, weight_scale, accumulate); break;
        case 3: down_iq<TY, 3>(w, row_bytes, n, act, out, r0, r1, weight_scale, accumulate); break;
        case 4: down_iq<TY, 4>(w, row_bytes, n, act, out, r0, r1, weight_scale, accumulate); break;
        case 5: down_iq<TY, 5>(w, row_bytes, n, act, out, r0, r1, weight_scale, accumulate); break;
        case 6: down_iq<TY, 6>(w, row_bytes, n, act, out, r0, r1, weight_scale, accumulate); break;
        default: down_iq_wide<TY>(w, row_bytes, n, act, nt, out, r0, r1, weight_scale, accumulate); break;
    }
}

void gu_split_fallback(
    int gate_type,
    int up_type,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t gate_rb,
    size_t up_rb,
    int n_in,
    const void* const* x_q8k,
    int nt,
    float* const* ff_out,
    int r0,
    int r1,
    float alpha,
    float limit
) {
    const int nr = r1 - r0;
    static thread_local std::vector<float> tmp_g, tmp_u;
    tmp_g.resize((size_t)nt * nr);
    tmp_u.resize((size_t)nt * nr);
    std::vector<float*> pg(nt), pu(nt);
    for (int t = 0; t < nt; ++t) {
        pg[t] = tmp_g.data() + (size_t)t * nr - r0;
        pu[t] = tmp_u.data() + (size_t)t * nr - r0;
    }
    auto run_dot = [&](int ty, const uint8_t* w, size_t rb, float* const* dst) {
        switch (ty) {
            case 16: down_nt<16>(nt, w, rb, n_in, x_q8k, dst, r0, r1, 1.0f, 0); break;
            case 17: down_nt<17>(nt, w, rb, n_in, x_q8k, dst, r0, r1, 1.0f, 0); break;
            case 18: down_nt<18>(nt, w, rb, n_in, x_q8k, dst, r0, r1, 1.0f, 0); break;
            case 19: down_nt<19>(nt, w, rb, n_in, x_q8k, dst, r0, r1, 1.0f, 0); break;
            case 22: down_nt<22>(nt, w, rb, n_in, x_q8k, dst, r0, r1, 1.0f, 0); break;
            default: break;
        }
    };
    run_dot(gate_type, gate_w, gate_rb, pg.data());
    run_dot(up_type, up_w, up_rb, pu.data());
    for (int t = 0; t < nt; ++t) {
        for (int r = r0; r < r1; ++r) {
            ff_out[t][r] = swoai(pg[t][r], pu[t][r], alpha, limit);
        }
    }
}

}  // namespace

extern "C" {

void ampllm_cpu_mm3_quantize_q8_k(const float* x, void* out_q8k, int n) {
    const int nb = n / QK_K;
    const int rem = n - nb * QK_K;
    block_q8_K* y = (block_q8_K*)out_q8k;
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
    const __m512 lo = _mm512_set1_ps(-127.0f);
    const __m512 hi = _mm512_set1_ps(127.0f);
    for (int i = 0; i < nb; ++i) {
        const float* xp = x + (size_t)i * QK_K;
        __m512 mx = _mm512_setzero_ps();
        for (int c = 0; c < 16; ++c) {
            const __m512 v = _mm512_loadu_ps(xp + c * 16);
            mx = _mm512_max_ps(mx, _mm512_and_ps(v, abs_mask));
        }
        const float amax = _mm512_reduce_max_ps(mx);
        const float d = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        y[i].d = d;
        const __m512 mul = _mm512_set1_ps(id);
        for (int c = 0; c < 16; ++c) {
            __m512 v = _mm512_roundscale_ps(
                _mm512_mul_ps(_mm512_loadu_ps(xp + c * 16), mul), _MM_FROUND_TO_NEAREST_INT);
            v = _mm512_min_ps(_mm512_max_ps(v, lo), hi);
            const __m512i vi = _mm512_cvtps_epi32(v);
            const __m128i qi = _mm512_cvtepi32_epi8(vi);
            _mm_storeu_si128((__m128i*)(y[i].qs + c * 16), qi);
            y[i].bsums[c] = (int16_t)_mm512_reduce_add_epi32(vi);
        }
    }
    if (rem > 0) {
        const float* xp = x + (size_t)nb * QK_K;
        const int n_sub16 = rem / 16;
        __m512 mx = _mm512_setzero_ps();
        for (int c = 0; c < n_sub16; ++c) {
            const __m512 v = _mm512_loadu_ps(xp + c * 16);
            mx = _mm512_max_ps(mx, _mm512_and_ps(v, abs_mask));
        }
        for (int j = n_sub16 * 16; j < rem; ++j) {
            mx = _mm512_max_ps(mx, _mm512_set1_ps(std::fabs(xp[j])));
        }
        const float amax = _mm512_reduce_max_ps(mx);
        const float d = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        y[nb].d = d;
        const __m512 mul = _mm512_set1_ps(id);
        for (int c = 0; c < n_sub16; ++c) {
            __m512 v = _mm512_roundscale_ps(
                _mm512_mul_ps(_mm512_loadu_ps(xp + c * 16), mul), _MM_FROUND_TO_NEAREST_INT);
            v = _mm512_min_ps(_mm512_max_ps(v, lo), hi);
            const __m512i vi = _mm512_cvtps_epi32(v);
            const __m128i qi = _mm512_cvtepi32_epi8(vi);
            _mm_storeu_si128((__m128i*)(y[nb].qs + c * 16), qi);
            y[nb].bsums[c] = (int16_t)_mm512_reduce_add_epi32(vi);
        }
        if (n_sub16 * 16 < rem) {
            const int rem_tail = rem - n_sub16 * 16;
            int16_t bsum = 0;
            for (int j = 0; j < rem_tail; ++j) {
                const float qf = std::nearbyint(xp[n_sub16 * 16 + j] * id);
                const int q = (int)std::fmin(127.0f, std::fmax(-127.0f, qf));
                y[nb].qs[n_sub16 * 16 + j] = (int8_t)q;
                bsum += (int16_t)q;
            }
            std::memset(y[nb].qs + rem, 0, (size_t)(QK_K - rem));
            y[nb].bsums[n_sub16] = bsum;
            if (n_sub16 + 1 < 16) {
                std::memset(y[nb].bsums + n_sub16 + 1, 0, sizeof(int16_t) * (16 - (n_sub16 + 1)));
            }
        } else {
            std::memset(y[nb].qs + rem, 0, (size_t)(QK_K - rem));
            std::memset(y[nb].bsums + n_sub16, 0, sizeof(int16_t) * (16 - n_sub16));
        }
    }
}

void ampllm_cpu_mm3_fused_gu_rows(
    int gate_type,
    int up_type,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t gate_row_bytes,
    size_t up_row_bytes,
    int n_in,
    const void* const* x_q8k,
    int nt,
    float* const* ff_out,
    int r0,
    int r1,
    float swiglu_alpha,
    float swiglu_limit
) {
    if (r0 >= r1 || nt <= 0) return;
    if (gate_type == up_type) {
        switch (gate_type) {
            case 16: gu_fused_nt<16, 16>(nt, gate_w, up_w, gate_row_bytes, up_row_bytes, n_in, x_q8k, ff_out, r0, r1, swiglu_alpha, swiglu_limit); return;
            case 17: gu_fused_nt<17, 17>(nt, gate_w, up_w, gate_row_bytes, up_row_bytes, n_in, x_q8k, ff_out, r0, r1, swiglu_alpha, swiglu_limit); return;
            case 18: gu_fused_nt<18, 18>(nt, gate_w, up_w, gate_row_bytes, up_row_bytes, n_in, x_q8k, ff_out, r0, r1, swiglu_alpha, swiglu_limit); return;
            case 19: gu_fused_nt<19, 19>(nt, gate_w, up_w, gate_row_bytes, up_row_bytes, n_in, x_q8k, ff_out, r0, r1, swiglu_alpha, swiglu_limit); return;
            case 22: gu_fused_nt<22, 22>(nt, gate_w, up_w, gate_row_bytes, up_row_bytes, n_in, x_q8k, ff_out, r0, r1, swiglu_alpha, swiglu_limit); return;
            default: return;
        }
    }
    if (gate_type == 17 && up_type == 16) {
        gu_fused_nt<17, 16>(
            nt, gate_w, up_w, gate_row_bytes, up_row_bytes, n_in, x_q8k, ff_out,
            r0, r1, swiglu_alpha, swiglu_limit);
        return;
    }
    gu_split_fallback(
        gate_type, up_type, gate_w, up_w, gate_row_bytes, up_row_bytes, n_in, x_q8k,
        nt, ff_out, r0, r1, swiglu_alpha, swiglu_limit);
}

void ampllm_cpu_mm3_down_rows(
    int down_type,
    const uint8_t* down_w,
    size_t row_bytes,
    int n_in,
    const void* const* ff_q8k,
    int nt,
    float* const* out,
    int r0,
    int r1,
    float weight_scale,
    int accumulate
) {
    if (r0 >= r1 || nt <= 0) return;
    switch (down_type) {
        case 16: down_nt<16>(nt, down_w, row_bytes, n_in, ff_q8k, out, r0, r1, weight_scale, accumulate); break;
        case 17: down_nt<17>(nt, down_w, row_bytes, n_in, ff_q8k, out, r0, r1, weight_scale, accumulate); break;
        case 18: down_nt<18>(nt, down_w, row_bytes, n_in, ff_q8k, out, r0, r1, weight_scale, accumulate); break;
        case 19: down_nt<19>(nt, down_w, row_bytes, n_in, ff_q8k, out, r0, r1, weight_scale, accumulate); break;
        case 22: down_nt<22>(nt, down_w, row_bytes, n_in, ff_q8k, out, r0, r1, weight_scale, accumulate); break;
        default: break;
    }
}

void ampllm_cpu_mm3_q6_k_row_to_f32(const uint8_t* src, int32_t k, float* dst_f32) {
    if (k <= 0) return;
    const int nb = k / QK_K;
    const auto* blks = reinterpret_cast<const block_q6_K*>(src);
    for (int b = 0; b < nb; ++b) {
        uint16_t dh;
        std::memcpy(&dh, &blks[b].d, 2);
        const float d = h2f(dh);
        for (int half = 0; half < 2; ++half) {
            const uint8_t* ql = blks[b].ql + half * 64;
            const uint8_t* qh = blks[b].qh + half * 32;
            const int8_t* sc = blks[b].scales + half * 8;
            float* dst = dst_f32 + b * 256 + half * 128;
            for (int l = 0; l < 32; ++l) {
                const int is = l >> 4;
                const int q1 = (int)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                dst[l + 0] = d * (float)sc[is + 0] * (float)q1;
                dst[l + 32] = d * (float)sc[is + 2] * (float)q2;
                dst[l + 64] = d * (float)sc[is + 4] * (float)q3;
                dst[l + 96] = d * (float)sc[is + 6] * (float)q4;
            }
        }
    }
}

}  // extern "C"
