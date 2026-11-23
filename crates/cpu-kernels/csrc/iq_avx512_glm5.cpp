#include "iq_tables.hpp"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

static constexpr int QK_K = 256;

struct block_q8_K {
    float d;
    int8_t qs[QK_K];
    int16_t bsums[QK_K / 16];
};
static_assert(sizeof(block_q8_K) == 292, "block_q8_K size mismatch");

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

inline float h2f(uint16_t h) {
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int)h)));
}
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

constexpr int kPrefetchAhead = 2048;

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
    const __m256i par = _mm256_slli_epi32(_mm256_and_si256(_mm256_popcnt_epi32(si), _mm256_set1_epi32(1)), 7);
    return _cvtu64_mask64((uint64_t)_mm_cvtsi128_si64(_mm256_cvtepi32_epi8(_mm256_or_si256(si, par))));
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

inline float hsum8_ggml(__m256 x) {
    __m128 s = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

template <int TY> struct FmtX;

template <> struct FmtX<22> { // IQ2_S
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
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
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct FmtX<18> { // IQ3_XXS
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        g = _mm512_set_epi32(
            (int)iq3xxs_grid[q[15]], (int)iq3xxs_grid[q[14]], (int)iq3xxs_grid[q[13]], (int)iq3xxs_grid[q[12]],
            (int)iq3xxs_grid[q[11]], (int)iq3xxs_grid[q[10]], (int)iq3xxs_grid[q[9]],  (int)iq3xxs_grid[q[8]],
            (int)iq3xxs_grid[q[7]],  (int)iq3xxs_grid[q[6]],  (int)iq3xxs_grid[q[5]],  (int)iq3xxs_grid[q[4]],
            (int)iq3xxs_grid[q[3]],  (int)iq3xxs_grid[q[2]],  (int)iq3xxs_grid[q[1]],  (int)iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        m = sign_mask_from_idx(sign_idx_xxs(a, c));
        const int sa = 2 * (int)(a >> 28) + 1, sb = 2 * (int)(c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct FmtX<21> { // IQ3_S
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
#define G3(k, h, kk) (int)iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(
            G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4),
            G3(11, h1, 3), G3(10, h1, 2), G3(9, h1, 1),  G3(8, h1, 0),
            G3(7, h0, 7),  G3(6, h0, 6),  G3(5, h0, 5),  G3(4, h0, 4),
            G3(3, h0, 3),  G3(2, h0, 2),  G3(1, h0, 1),  G3(0, h0, 0));
#undef G3
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <int TY, int NT>
void gu_fused_iq(const uint8_t* gate_w, const uint8_t* up_w, size_t row_bytes, int n,
                 const void* const* act, float* const* ff, int r0, int r1, float swiglu_limit) {
    const bool clamp = std::isfinite(swiglu_limit);
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*)act[t];
    const int nb = n / QK_K;
    const __m512i zero = _mm512_setzero_si512();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* gr = gate_w + (size_t)r * row_bytes;
        const uint8_t* ur = up_w + (size_t)r * row_bytes;
        __m256 ag[NT], au[NT];
        for (int t = 0; t < NT; ++t) ag[t] = au[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* gb = gr + (size_t)i * FmtX<TY>::bytes;
            const uint8_t* ub = ur + (size_t)i * FmtX<TY>::bytes;
            if (kPrefetchAhead > 0) {
                _mm_prefetch((const char*)gb + kPrefetchAhead, _MM_HINT_T0);
                _mm_prefetch((const char*)gb + kPrefetchAhead + 64, _MM_HINT_T0);
                _mm_prefetch((const char*)ub + kPrefetchAhead, _MM_HINT_T0);
                _mm_prefetch((const char*)ub + kPrefetchAhead + 64, _MM_HINT_T0);
            }
            __m512i ig[NT], iu[NT];
            for (int t = 0; t < NT; ++t) ig[t] = iu[t] = zero;
            for (int j = 0; j < 4; ++j) {
                __m512i gg, gs, ug, us;
                __mmask64 gm, um;
                FmtX<TY>::decode(gb, j, gg, gm, gs);
                FmtX<TY>::decode(ub, j, ug, um, us);
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
                const float yd = y[t][i].d;
                ag[t] = _mm256_fmadd_ps(_mm256_set1_ps(dg * yd), _mm256_cvtepi32_ps(fold256(ig[t])), ag[t]);
                au[t] = _mm256_fmadd_ps(_mm256_set1_ps(du * yd), _mm256_cvtepi32_ps(fold256(iu[t])), au[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            float g = FmtX<TY>::K * hsum8_ggml(ag[t]);
            float u = FmtX<TY>::K * hsum8_ggml(au[t]);
            if (clamp) {
                g = std::fmin(g, swiglu_limit);
                u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
            }
            ff[t][r] = (g / (1.0f + std::exp(-g))) * u;
        }
    }
}

template <int TY, int NT>
void down_iq(const uint8_t* w, size_t row_bytes, int n, const void* const* act,
             float* const* out, int r0, int r1, float weight_scale, int accumulate) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*)act[t];
    const int nb = n / QK_K;
    const __m512i zero = _mm512_setzero_si512();
    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * row_bytes;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();
        for (int i = 0; i < nb; ++i) {
            const uint8_t* blk = row + (size_t)i * FmtX<TY>::bytes;
            if (kPrefetchAhead > 0) {
                _mm_prefetch((const char*)blk + kPrefetchAhead, _MM_HINT_T0);
                _mm_prefetch((const char*)blk + kPrefetchAhead + 64, _MM_HINT_T0);
            }
            __m512i acci[NT];
            for (int t = 0; t < NT; ++t) acci[t] = zero;
            for (int j = 0; j < 4; ++j) {
                __m512i g, sc;
                __mmask64 m;
                FmtX<TY>::decode(blk, j, g, m, sc);
                for (int t = 0; t < NT; ++t) {
                    const __m512i yv = _mm512_loadu_si512((const void*)(y[t][i].qs + 64 * j));
                    const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                    acci[t] = _mm512_dpwssd_epi32(acci[t], _mm512_maddubs_epi16(g, ys), sc);
                }
            }
            const float dx = h2f(u16(blk));
            for (int t = 0; t < NT; ++t) {
                accf[t] = _mm256_fmadd_ps(
                    _mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(fold256(acci[t])), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            const float val = weight_scale * FmtX<TY>::K * hsum8_ggml(accf[t]);
            if (accumulate) out[t][r] += val;
            else out[t][r] = val;
        }
    }
}

template <int TY>
void down_iq_wide(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int T,
                  float* const* out, int r0, int r1, float weight_scale, int accumulate) {
    constexpr int kTile = 16, kRows = 8;
    const int nb = n / QK_K;
    static thread_local std::vector<uint8_t> ubuf;
    static thread_local std::vector<float> accb;
    ubuf.resize((size_t)(T < kTile ? T : kTile) * n);
    accb.resize((size_t)kRows * kTile * 8);
    const __m512i zero = _mm512_setzero_si512(), one8 = _mm512_set1_epi8(1), flip = _mm512_set1_epi8((char)0x80);
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
                        _mm512_xor_si512(_mm512_loadu_si512((const void*)(y[t][i].qs + 64 * j)), flip));
        }
        for (int rg = r0; rg < r1; rg += kRows) {
            const int rr = r1 - rg < kRows ? r1 - rg : kRows;
            for (int k = 0; k < rr * tt; ++k) _mm256_storeu_ps(&accb[(size_t)k * 8], _mm256_setzero_ps());
            for (int i = 0; i < nb; ++i) {
                for (int q = 0; q < rr; ++q) {
                    const uint8_t* blk = w + (size_t)(rg + q) * row_bytes + (size_t)i * FmtX<TY>::bytes;
                    if (kPrefetchAhead > 0) {
                        _mm_prefetch((const char*)blk + kPrefetchAhead, _MM_HINT_T0);
                        _mm_prefetch((const char*)blk + kPrefetchAhead + 64, _MM_HINT_T0);
                    }
                    __m512i w8[4], sc[4];
                    __m512i corr = zero;
                    for (int j = 0; j < 4; ++j) {
                        __m512i g, s;
                        __mmask64 m;
                        FmtX<TY>::decode(blk, j, g, m, s);
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
                        const __m256i s = _mm256_sub_epi32(fold256(a), corr128);
                        float* ap = accq + (size_t)t * 8;
                        _mm256_storeu_ps(
                            ap, _mm256_fmadd_ps(_mm256_set1_ps(dx * y[t][i].d), _mm256_cvtepi32_ps(s), _mm256_loadu_ps(ap)));
                    }
                }
            }
            for (int q = 0; q < rr; ++q) {
                for (int t = 0; t < tt; ++t) {
                    const float val = weight_scale * FmtX<TY>::K * hsum8_ggml(_mm256_loadu_ps(&accb[((size_t)q * tt + t) * 8]));
                    if (accumulate) out[t0 + t][rg + q] += val;
                    else out[t0 + t][rg + q] = val;
                }
            }
        }
    }
}

// ---- Q3_K (type 11) and Q4_K (type 12) AVX2/VNNI kernels for NextN MTP (blk.45) ----
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

inline float dot_q3_k_q8_k(const block_q3_K* x, const block_q8_K* y, int nb) {
    const uint32_t kmask1 = 0x03030303u;
    const uint32_t kmask2 = 0x0f0f0f0fu;
    const __m256i m3 = _mm256_set1_epi8(3);
    const __m256i mone = _mm256_set1_epi8(1);
    const __m128i m32 = _mm_set1_epi8(32);
    __m256 acc = _mm256_setzero_ps();
    uint32_t aux[3];
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * h2f(x[i].d);
        const uint8_t* q3 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        std::memcpy(aux, x[i].scales, 12);
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
        const __m256i hbits = _mm256_loadu_si256((const __m256i*)x[i].hmask);
        __m256i sumi = _mm256_setzero_si256();
        int bit = 0;
        for (int j = 0; j < 2; ++j) {
            const __m256i q3bits = _mm256_loadu_si256((const __m256i*)q3);
            q3 += 32;
            const __m256i q3l_0 = _mm256_and_si256(q3bits, m3);
            const __m256i q3h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
            const __m256i q3h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
            const __m256i q3h_2 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;
            const __m256i q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
            const __m256i q3h_3 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, bit)), bit), 2);
            ++bit;

            const __m256i q8_0 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_1 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_2 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8_3 = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;

            __m256i p16_0 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_0, q8_0), _mm256_maddubs_epi16(q3h_0, q8_0));
            __m256i p16_1 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_1, q8_1), _mm256_maddubs_epi16(q3h_1, q8_1));
            __m256i p16_2 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_2, q8_2), _mm256_maddubs_epi16(q3h_2, q8_2));
            __m256i p16_3 = _mm256_sub_epi16(_mm256_maddubs_epi16(q3l_3, q8_3), _mm256_maddubs_epi16(q3h_3, q8_3));

            p16_0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(0)), p16_0);
            p16_1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(1)), p16_1);
            p16_2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(2)), p16_2);
            p16_3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], get_scale_shuffle_q3k(3)), p16_3);

            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(_mm256_add_epi32(p16_0, p16_1), _mm256_add_epi32(p16_2, p16_3)));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum8_ggml(acc);
}

inline float dot_q4_k_q8_k(const block_q4_K* x, const block_q8_K* y, int nb) {
    static constexpr uint32_t kmask1 = 0x3f3f3f3fu;
    static constexpr uint32_t kmask2 = 0x0f0f0f0fu;
    static constexpr uint32_t kmask3 = 0x03030303u;
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    __m128 acc_m = _mm_setzero_ps();
    uint32_t utmp[4];
    for (int i = 0; i < nb; ++i) {
        const float d = y[i].d * h2f(x[i].d);
        const float dmin = -y[i].d * h2f(x[i].dmin);
        std::memcpy(utmp, x[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const uint8_t* q4 = x[i].qs;
        const int8_t* q8 = y[i].qs;
        const __m256i mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m256i q8sums = _mm256_loadu_si256((const __m256i*)y[i].bsums);
        const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
        const __m128i prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
        acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

        const __m128i sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales = _mm256_insertf128_si256(_mm256_castsi128_si256(sc128), sc128, 1);
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; ++j) {
            const __m256i scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));
            const __m256i q4bits = _mm256_loadu_si256((const __m256i*)q4); q4 += 32;
            const __m256i q4l = _mm256_and_si256(q4bits, m4);
            const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);
            const __m256i q8l = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            const __m256i q8h = _mm256_loadu_si256((const __m256i*)q8); q8 += 32;
            __m256i p16l = _mm256_madd_epi16(scale_l, _mm256_maddubs_epi16(q4l, q8l));
            __m256i p16h = _mm256_madd_epi16(scale_h, _mm256_maddubs_epi16(q4h, q8h));
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16l, p16h));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
    }
    acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
    acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));
    return hsum8_ggml(acc) + _mm_cvtss_f32(acc_m);
}

template <int TY>
void gu_fused_iq_wide(const uint8_t* gate_w, const uint8_t* up_w, size_t row_bytes, int n,
                      const void* const* act, int T, float* const* ff, int r0, int r1, float swiglu_limit) {
    constexpr int kTile = 16, kRows = 4;
    const bool clamp = std::isfinite(swiglu_limit);
    const int nb = n / QK_K;
    static thread_local std::vector<uint8_t> ubuf;
    static thread_local std::vector<float> accg, accu;
    ubuf.resize((size_t)(T < kTile ? T : kTile) * n);
    accg.resize((size_t)kRows * kTile * 8);
    accu.resize((size_t)kRows * kTile * 8);
    const __m512i zero = _mm512_setzero_si512(), one8 = _mm512_set1_epi8(1), flip = _mm512_set1_epi8((char)0x80);
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
                        _mm512_xor_si512(_mm512_loadu_si512((const void*)(y[t][i].qs + 64 * j)), flip));
        }
        for (int rg = r0; rg < r1; rg += kRows) {
            const int rr = r1 - rg < kRows ? r1 - rg : kRows;
            for (int k = 0; k < rr * tt; ++k) {
                _mm256_storeu_ps(&accg[(size_t)k * 8], _mm256_setzero_ps());
                _mm256_storeu_ps(&accu[(size_t)k * 8], _mm256_setzero_ps());
            }
            for (int i = 0; i < nb; ++i) {
                for (int q = 0; q < rr; ++q) {
                    const uint8_t* gblk = gate_w + (size_t)(rg + q) * row_bytes + (size_t)i * FmtX<TY>::bytes;
                    const uint8_t* ublk = up_w   + (size_t)(rg + q) * row_bytes + (size_t)i * FmtX<TY>::bytes;
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
                        FmtX<TY>::decode(gblk, j, gg, gm, gs);
                        FmtX<TY>::decode(ublk, j, ug, um, us);
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
                        const __m256i sg = _mm256_sub_epi32(fold256(ag), gcorr128);
                        const __m256i su = _mm256_sub_epi32(fold256(au), ucorr128);
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
                    float g = FmtX<TY>::K * hsum8_ggml(_mm256_loadu_ps(&accg[idx]));
                    float u = FmtX<TY>::K * hsum8_ggml(_mm256_loadu_ps(&accu[idx]));
                    if (clamp) {
                        g = std::fmin(g, swiglu_limit);
                        u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
                    }
                    ff[t0 + t][rg + q] = (g / (1.0f + std::exp(-g))) * u;
                }
            }
        }
    }
}

template <int TY>
void gu_fused_nt(int nt, const uint8_t* gate_w, const uint8_t* up_w, size_t row_bytes, int n,
                 const void* const* act, float* const* ff, int r0, int r1, float swiglu_limit) {
    switch (nt) {
        case 1: gu_fused_iq<TY, 1>(gate_w, up_w, row_bytes, n, act, ff, r0, r1, swiglu_limit); break;
        case 2: gu_fused_iq<TY, 2>(gate_w, up_w, row_bytes, n, act, ff, r0, r1, swiglu_limit); break;
        case 3: gu_fused_iq<TY, 3>(gate_w, up_w, row_bytes, n, act, ff, r0, r1, swiglu_limit); break;
        case 4: gu_fused_iq<TY, 4>(gate_w, up_w, row_bytes, n, act, ff, r0, r1, swiglu_limit); break;
        case 5: gu_fused_iq<TY, 5>(gate_w, up_w, row_bytes, n, act, ff, r0, r1, swiglu_limit); break;
        case 6: gu_fused_iq<TY, 6>(gate_w, up_w, row_bytes, n, act, ff, r0, r1, swiglu_limit); break;
        default: gu_fused_iq_wide<TY>(gate_w, up_w, row_bytes, n, act, nt, ff, r0, r1, swiglu_limit); break;
    }
}

template <int TY>
void down_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act,
             float* const* out, int r0, int r1, float weight_scale, int accumulate) {
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

}  // namespace

extern "C" {

void ampllm_cpu_quantize_q8_k(const float* x, void* out_q8k, int n) {
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
            __m512 v = _mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(xp + c * 16), mul), _MM_FROUND_TO_NEAREST_INT);
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
            float av = std::fabs(xp[j]);
            mx = _mm512_max_ps(mx, _mm512_set1_ps(av));
        }
        const float amax = _mm512_reduce_max_ps(mx);
        const float d = amax / 127.0f;
        const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        y[nb].d = d;
        const __m512 mul = _mm512_set1_ps(id);
        for (int c = 0; c < n_sub16; ++c) {
            __m512 v = _mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(xp + c * 16), mul), _MM_FROUND_TO_NEAREST_INT);
            v = _mm512_min_ps(_mm512_max_ps(v, lo), hi);
            const __m512i vi = _mm512_cvtps_epi32(v);
            const __m128i qi = _mm512_cvtepi32_epi8(vi);
            _mm_storeu_si128((__m128i*)(y[nb].qs + c * 16), qi);
            y[nb].bsums[c] = (int16_t)_mm512_reduce_add_epi32(vi);
        }
        if (n_sub16 * 16 < rem) {
            int rem_tail = rem - n_sub16 * 16;
            int16_t bsum = 0;
            for (int j = 0; j < rem_tail; ++j) {
                float qf = std::nearbyint(xp[n_sub16 * 16 + j] * id);
                int q = (int)std::fmin(127.0f, std::fmax(-127.0f, qf));
                y[nb].qs[n_sub16 * 16 + j] = (int8_t)q;
                bsum += (int16_t)q;
            }
            std::memset(y[nb].qs + rem, 0, QK_K - rem);
            y[nb].bsums[n_sub16] = bsum;
            if (n_sub16 + 1 < 16) {
                std::memset(y[nb].bsums + n_sub16 + 1, 0, sizeof(int16_t) * (16 - (n_sub16 + 1)));
            }
        } else {
            std::memset(y[nb].qs + rem, 0, QK_K - rem);
            std::memset(y[nb].bsums + n_sub16, 0, sizeof(int16_t) * (16 - n_sub16));
        }
    }
}

void ampllm_cpu_iq_fused_gu_rows(
    int type,
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t row_bytes,
    int n_in,
    const void* const* x_q8k,
    int nt,
    float* const* ff_out,
    int r0,
    int r1,
    float swiglu_limit
) {
    if (r0 >= r1 || nt <= 0) return;
    switch (type) {
        case 18: gu_fused_nt<18>(nt, gate_w, up_w, row_bytes, n_in, x_q8k, ff_out, r0, r1, swiglu_limit); break;
        case 21: gu_fused_nt<21>(nt, gate_w, up_w, row_bytes, n_in, x_q8k, ff_out, r0, r1, swiglu_limit); break;
        case 22: gu_fused_nt<22>(nt, gate_w, up_w, row_bytes, n_in, x_q8k, ff_out, r0, r1, swiglu_limit); break;
        case 11: {
            const int nb = n_in / QK_K;
            const bool clamp = std::isfinite(swiglu_limit);
            for (int r = r0; r < r1; ++r) {
                const block_q3_K* gr = (const block_q3_K*)(gate_w + (size_t)r * row_bytes);
                const block_q3_K* ur = (const block_q3_K*)(up_w + (size_t)r * row_bytes);
                for (int t = 0; t < nt; ++t) {
                    const block_q8_K* y = (const block_q8_K*)x_q8k[t];
                    float g = dot_q3_k_q8_k(gr, y, nb);
                    float u = dot_q3_k_q8_k(ur, y, nb);
                    if (clamp) {
                        g = std::fmin(g, swiglu_limit);
                        u = std::fmin(std::fmax(u, -swiglu_limit), swiglu_limit);
                    }
                    ff_out[t][r] = (g / (1.0f + std::exp(-g))) * u;
                }
            }
            break;
        }
        default: break;
    }
}

void ampllm_cpu_iq_down_rows(
    int type,
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
    switch (type) {
        case 18: down_nt<18>(nt, down_w, row_bytes, n_in, ff_q8k, out, r0, r1, weight_scale, accumulate); break;
        case 21: down_nt<21>(nt, down_w, row_bytes, n_in, ff_q8k, out, r0, r1, weight_scale, accumulate); break;
        case 22: down_nt<22>(nt, down_w, row_bytes, n_in, ff_q8k, out, r0, r1, weight_scale, accumulate); break;
        case 12: {
            const int nb = n_in / QK_K;
            for (int r = r0; r < r1; ++r) {
                const block_q4_K* dr = (const block_q4_K*)(down_w + (size_t)r * row_bytes);
                for (int t = 0; t < nt; ++t) {
                    const float val = weight_scale * dot_q4_k_q8_k(dr, (const block_q8_K*)ff_q8k[t], nb);
                    if (accumulate) out[t][r] += val;
                    else out[t][r] = val;
                }
            }
            break;
        }
        default: break;
    }
}

}  // extern "C"
