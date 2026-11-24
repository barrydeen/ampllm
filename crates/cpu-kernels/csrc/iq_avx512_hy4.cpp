#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
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
constexpr int GGML_TYPE_IQ4_NL = 20;
constexpr int GGML_TYPE_IQ4_XS = 23;
constexpr int GGML_TYPE_BF16 = 30;
constexpr int GGML_TYPE_Q2_0 = 42;

constexpr int kPrefetchAhead = 2048;

alignas(64) constexpr int8_t KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
    1, 13, 25, 38, 53, 69, 89, 113
};

alignas(64) constexpr uint8_t KVALUES_IQ4NL_U8[16] = {
    1, 24, 45, 63, 79, 93, 106, 118,
    129, 141, 153, 166, 181, 197, 217, 241
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

inline float clamped_swiglu(float g, float u, float lim) {
    if (lim > 0.0f) {
        g = std::min(g, lim);
        u = std::max(-lim, std::min(u, lim));
    }
    return (g / (1.0f + std::exp(-g))) * u;
}

inline int32_t hsum_i32_128(__m128i v) {
    __m128i s1 = _mm_add_epi32(v, _mm_shuffle_epi32(v, 0x4E));
    __m128i s2 = _mm_add_epi32(s1, _mm_shuffle_epi32(s1, 0xB1));
    return _mm_cvtsi128_si32(s2);
}

inline int32_t hsum_i32_256(__m256i v) {
    return hsum_i32_128(_mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1)));
}

inline float hsum_f32_256(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// Collapse each 128-bit lane (4x i32) into its lane-0 dword and pack 4 lanes from a + 4 from b -> 8xi32.
inline __m256i reduce4_pair_to_256(__m512i a, __m512i b) {
    a = _mm512_add_epi32(a, _mm512_shuffle_epi32(a, _MM_PERM_CDAB));
    a = _mm512_add_epi32(a, _mm512_shuffle_epi32(a, _MM_PERM_BADC));
    b = _mm512_add_epi32(b, _mm512_shuffle_epi32(b, _MM_PERM_CDAB));
    b = _mm512_add_epi32(b, _mm512_shuffle_epi32(b, _MM_PERM_BADC));
    const __m512i idx = _mm512_set_epi32(
        0, 0, 0, 0, 0, 0, 0, 0,
        28, 24, 20, 16, 12, 8, 4, 0
    );
    return _mm512_castsi512_si256(_mm512_permutex2var_epi32(a, idx, b));
}

// Reduce each 256-bit half (8x i32) of 4 ZMMs -> 8xi32 [b0..b7]
inline __m256i reduce8_quad_to_256(__m512i a01, __m512i a23, __m512i a45, __m512i a67) {
    a01 = _mm512_add_epi32(a01, _mm512_shuffle_epi32(a01, _MM_PERM_CDAB));
    a01 = _mm512_add_epi32(a01, _mm512_shuffle_epi32(a01, _MM_PERM_BADC));
    a23 = _mm512_add_epi32(a23, _mm512_shuffle_epi32(a23, _MM_PERM_CDAB));
    a23 = _mm512_add_epi32(a23, _mm512_shuffle_epi32(a23, _MM_PERM_BADC));
    a45 = _mm512_add_epi32(a45, _mm512_shuffle_epi32(a45, _MM_PERM_CDAB));
    a45 = _mm512_add_epi32(a45, _mm512_shuffle_epi32(a45, _MM_PERM_BADC));
    a67 = _mm512_add_epi32(a67, _mm512_shuffle_epi32(a67, _MM_PERM_CDAB));
    a67 = _mm512_add_epi32(a67, _mm512_shuffle_epi32(a67, _MM_PERM_BADC));

    // Each ZMM has [lo_a@0, hi_a@4, lo_b@8, hi_b@12].
    const __m512i idx_lo = _mm512_set_epi32(
        0, 0, 0, 0, 0, 0, 0, 0,
        28, 20, 12, 4, 24, 16, 8, 0
    );
    __m512i p03 = _mm512_permutex2var_epi32(a01, idx_lo, a23); // lanes 0..3: lo0..3, lanes 4..7: hi0..3
    __m512i p47 = _mm512_permutex2var_epi32(a45, idx_lo, a67); // lanes 0..3: lo4..7, lanes 4..7: hi4..7
    __m256i y03 = _mm512_castsi512_si256(p03);
    __m256i y47 = _mm512_castsi512_si256(p47);
    __m128i s03 = _mm_add_epi32(_mm256_castsi256_si128(y03), _mm256_extracti128_si256(y03, 1));
    __m128i s47 = _mm_add_epi32(_mm256_castsi256_si128(y47), _mm256_extracti128_si256(y47, 1));
    return _mm256_set_m128i(s47, s03);
}

// Same as reduce8_quad_to_256 but interleaves even/odd pairs: (pe02, po13, pe46, po57) -> [s0, s1, s2, s3, s4, s5, s6, s7]
inline __m256i reduce8_interleaved_to_256(__m512i pe02, __m512i po13, __m512i pe46, __m512i po57) {
    pe02 = _mm512_add_epi32(pe02, _mm512_shuffle_epi32(pe02, _MM_PERM_CDAB));
    pe02 = _mm512_add_epi32(pe02, _mm512_shuffle_epi32(pe02, _MM_PERM_BADC));
    po13 = _mm512_add_epi32(po13, _mm512_shuffle_epi32(po13, _MM_PERM_CDAB));
    po13 = _mm512_add_epi32(po13, _mm512_shuffle_epi32(po13, _MM_PERM_BADC));
    pe46 = _mm512_add_epi32(pe46, _mm512_shuffle_epi32(pe46, _MM_PERM_CDAB));
    pe46 = _mm512_add_epi32(pe46, _mm512_shuffle_epi32(pe46, _MM_PERM_BADC));
    po57 = _mm512_add_epi32(po57, _mm512_shuffle_epi32(po57, _MM_PERM_CDAB));
    po57 = _mm512_add_epi32(po57, _mm512_shuffle_epi32(po57, _MM_PERM_BADC));

    const __m512i idx_eo = _mm512_set_epi32(
        0, 0, 0, 0, 0, 0, 0, 0,
        28, 12, 20, 4, 24, 8, 16, 0
    );
    __m512i p03 = _mm512_permutex2var_epi32(pe02, idx_eo, po13);
    __m512i p47 = _mm512_permutex2var_epi32(pe46, idx_eo, po57);
    __m256i y03 = _mm512_castsi512_si256(p03);
    __m256i y47 = _mm512_castsi512_si256(p47);
    __m128i s03 = _mm_add_epi32(_mm256_castsi256_si128(y03), _mm256_extracti128_si256(y03, 1));
    __m128i s47 = _mm_add_epi32(_mm256_castsi256_si128(y47), _mm256_extracti128_si256(y47, 1));
    return _mm256_set_m128i(s47, s03);
}

// -----------------------------------------------------------------------------
// Prepared Q8_K block representations so NT tokens' activations are unpacked once
// across all output rows in [r0, r1).
// -----------------------------------------------------------------------------

struct PrepIq4Xs {
    float d;
    __m512i xlo03;
    __m512i xhi03;
    __m512i xlo47;
    __m512i xhi47;
    __m256i bias8; // 128 * (bsums[2*sb] + bsums[2*sb+1]) for sb = 0..7
};

inline void prepare_iq4_xs(const block_q8_K* xb, int nb, PrepIq4Xs* dst) {
    const __m512i idx_lo = _mm512_set_epi64(13, 12, 9, 8, 5, 4, 1, 0);
    const __m512i idx_hi = _mm512_set_epi64(15, 14, 11, 10, 7, 6, 3, 2);
    const __m256i v128 = _mm256_set1_epi16(128);
    for (int i = 0; i < nb; ++i) {
        const block_q8_K& b = xb[i];
        dst[i].d = b.d;
        __m512i x01 = _mm512_loadu_si512((const void*)(b.qs + 0));
        __m512i x23 = _mm512_loadu_si512((const void*)(b.qs + 64));
        __m512i x45 = _mm512_loadu_si512((const void*)(b.qs + 128));
        __m512i x67 = _mm512_loadu_si512((const void*)(b.qs + 192));
        dst[i].xlo03 = _mm512_permutex2var_epi64(x01, idx_lo, x23);
        dst[i].xhi03 = _mm512_permutex2var_epi64(x01, idx_hi, x23);
        dst[i].xlo47 = _mm512_permutex2var_epi64(x45, idx_lo, x67);
        dst[i].xhi47 = _mm512_permutex2var_epi64(x45, idx_hi, x67);
        __m256i bs = _mm256_loadu_si256((const __m256i*)b.bsums);
        dst[i].bias8 = _mm256_madd_epi16(bs, v128);
    }
}

struct PrepQ5K {
    float d;
    __m512i xe02;
    __m512i xo13;
    __m512i xe46;
    __m512i xo57;
    __m256i bsum8; // bsums[2*sb] + bsums[2*sb+1] for sb = 0..7
};

inline void prepare_q5_K(const block_q8_K* xb, int nb, PrepQ5K* dst) {
    const __m256i v1 = _mm256_set1_epi16(1);
    for (int i = 0; i < nb; ++i) {
        const block_q8_K& b = xb[i];
        dst[i].d = b.d;
        __m256i xs0 = _mm256_loadu_si256((const __m256i*)(b.qs + 0));
        __m256i xs1 = _mm256_loadu_si256((const __m256i*)(b.qs + 32));
        __m256i xs2 = _mm256_loadu_si256((const __m256i*)(b.qs + 64));
        __m256i xs3 = _mm256_loadu_si256((const __m256i*)(b.qs + 96));
        __m256i xs4 = _mm256_loadu_si256((const __m256i*)(b.qs + 128));
        __m256i xs5 = _mm256_loadu_si256((const __m256i*)(b.qs + 160));
        __m256i xs6 = _mm256_loadu_si256((const __m256i*)(b.qs + 192));
        __m256i xs7 = _mm256_loadu_si256((const __m256i*)(b.qs + 224));
        dst[i].xe02 = _mm512_inserti64x4(_mm512_castsi256_si512(xs0), xs2, 1);
        dst[i].xo13 = _mm512_inserti64x4(_mm512_castsi256_si512(xs1), xs3, 1);
        dst[i].xe46 = _mm512_inserti64x4(_mm512_castsi256_si512(xs4), xs6, 1);
        dst[i].xo57 = _mm512_inserti64x4(_mm512_castsi256_si512(xs5), xs7, 1);
        __m256i bs = _mm256_loadu_si256((const __m256i*)b.bsums);
        dst[i].bsum8 = _mm256_madd_epi16(bs, v1);
    }
}

struct PrepQ6K {
    float d;
    __m512i x0, x1, x2, x3;
    __m512i bias16; // 32 * bsums[0..15]
};

inline void prepare_q6_K(const block_q8_K* xb, int nb, PrepQ6K* dst) {
    const __m512i v32 = _mm512_set1_epi32(32);
    for (int i = 0; i < nb; ++i) {
        const block_q8_K& b = xb[i];
        dst[i].d = b.d;
        dst[i].x0 = _mm512_loadu_si512((const void*)(b.qs + 0));
        dst[i].x1 = _mm512_loadu_si512((const void*)(b.qs + 64));
        dst[i].x2 = _mm512_loadu_si512((const void*)(b.qs + 128));
        dst[i].x3 = _mm512_loadu_si512((const void*)(b.qs + 192));
        __m256i bs = _mm256_loadu_si256((const __m256i*)b.bsums);
        dst[i].bias16 = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(bs), v32);
    }
}

struct PrepQ8_0 {
    float d;
    __m512i x01, x23, x45, x67;
    __m256i bias8; // 128 * (bsums[2*sb] + bsums[2*sb+1])
};

inline void prepare_q8_0(const block_q8_K* xb, int nb, PrepQ8_0* dst) {
    const __m256i v128 = _mm256_set1_epi16(128);
    for (int i = 0; i < nb; ++i) {
        const block_q8_K& b = xb[i];
        dst[i].d = b.d;
        dst[i].x01 = _mm512_loadu_si512((const void*)(b.qs + 0));
        dst[i].x23 = _mm512_loadu_si512((const void*)(b.qs + 64));
        dst[i].x45 = _mm512_loadu_si512((const void*)(b.qs + 128));
        dst[i].x67 = _mm512_loadu_si512((const void*)(b.qs + 192));
        __m256i bs = _mm256_loadu_si256((const __m256i*)b.bsums);
        dst[i].bias8 = _mm256_madd_epi16(bs, v128);
    }
}

struct PrepQ5_1 {
    float d;
    __m512i x01, x23, x45, x67;
    __m256 bsum8f; // (float)(bsums[2*sb] + bsums[2*sb+1])
};

inline void prepare_q5_1(const block_q8_K* xb, int nb, PrepQ5_1* dst) {
    const __m256i v1 = _mm256_set1_epi16(1);
    for (int i = 0; i < nb; ++i) {
        const block_q8_K& b = xb[i];
        dst[i].d = b.d;
        dst[i].x01 = _mm512_loadu_si512((const void*)(b.qs + 0));
        dst[i].x23 = _mm512_loadu_si512((const void*)(b.qs + 64));
        dst[i].x45 = _mm512_loadu_si512((const void*)(b.qs + 128));
        dst[i].x67 = _mm512_loadu_si512((const void*)(b.qs + 192));
        __m256i bs = _mm256_loadu_si256((const __m256i*)b.bsums);
        dst[i].bsum8f = _mm256_cvtepi32_ps(_mm256_madd_epi16(bs, v1));
    }
}

struct PrepQ8_0_Q81 {
    __m256 d8;
    __m512i x01, x23, x45, x67;
    __m256i bias8; // 128 * bsum[sb]
};

inline void prepare_q8_0_q81(const block_q8_1_g8* xb, int nb, PrepQ8_0_Q81* dst) {
    for (int i = 0; i < nb; ++i) {
        const block_q8_1_g8& b = xb[i];
        dst[i].d8 = _mm256_loadu_ps(b.d);
        dst[i].x01 = _mm512_loadu_si512((const void*)(b.qs + 0));
        dst[i].x23 = _mm512_loadu_si512((const void*)(b.qs + 64));
        dst[i].x45 = _mm512_loadu_si512((const void*)(b.qs + 128));
        dst[i].x67 = _mm512_loadu_si512((const void*)(b.qs + 192));
        dst[i].bias8 = _mm256_slli_epi32(_mm256_loadu_si256((const __m256i*)b.bsum), 7);
    }
}

struct PrepQ5_1_Q81 {
    __m256 d8;
    __m256 s8;
    __m512i x01, x23, x45, x67;
};

inline void prepare_q5_1_q81(const block_q8_1_g8* xb, int nb, PrepQ5_1_Q81* dst) {
    for (int i = 0; i < nb; ++i) {
        const block_q8_1_g8& b = xb[i];
        dst[i].d8 = _mm256_loadu_ps(b.d);
        dst[i].s8 = _mm256_loadu_ps(b.s);
        dst[i].x01 = _mm512_loadu_si512((const void*)(b.qs + 0));
        dst[i].x23 = _mm512_loadu_si512((const void*)(b.qs + 64));
        dst[i].x45 = _mm512_loadu_si512((const void*)(b.qs + 128));
        dst[i].x67 = _mm512_loadu_si512((const void*)(b.qs + 192));
    }
}

struct PrepQ5K_Q81 {
    __m256 d8;
    __m256 s8;
    __m512i xe02;
    __m512i xo13;
    __m512i xe46;
    __m512i xo57;
};

inline void prepare_q5_K_q81(const block_q8_1_g8* xb, int nb, PrepQ5K_Q81* dst) {
    for (int i = 0; i < nb; ++i) {
        const block_q8_1_g8& b = xb[i];
        dst[i].d8 = _mm256_loadu_ps(b.d);
        dst[i].s8 = _mm256_loadu_ps(b.s);
        __m256i xs0 = _mm256_loadu_si256((const __m256i*)(b.qs + 0));
        __m256i xs1 = _mm256_loadu_si256((const __m256i*)(b.qs + 32));
        __m256i xs2 = _mm256_loadu_si256((const __m256i*)(b.qs + 64));
        __m256i xs3 = _mm256_loadu_si256((const __m256i*)(b.qs + 96));
        __m256i xs4 = _mm256_loadu_si256((const __m256i*)(b.qs + 128));
        __m256i xs5 = _mm256_loadu_si256((const __m256i*)(b.qs + 160));
        __m256i xs6 = _mm256_loadu_si256((const __m256i*)(b.qs + 192));
        __m256i xs7 = _mm256_loadu_si256((const __m256i*)(b.qs + 224));
        dst[i].xe02 = _mm512_inserti64x4(_mm512_castsi256_si512(xs0), xs2, 1);
        dst[i].xo13 = _mm512_inserti64x4(_mm512_castsi256_si512(xs1), xs3, 1);
        dst[i].xe46 = _mm512_inserti64x4(_mm512_castsi256_si512(xs4), xs6, 1);
        dst[i].xo57 = _mm512_inserti64x4(_mm512_castsi256_si512(xs5), xs7, 1);
    }
}

// -----------------------------------------------------------------------------
// Row kernels: dot(row, prepared_x[t]) across t = 0..NT-1
// -----------------------------------------------------------------------------

template <int NT>
void dot_rows_iq4_xs(
    const uint8_t* w,
    size_t rb,
    int nb,
    const PrepIq4Xs* const* xp,
    int r0,
    int r1,
    float* const* acc_out
) {
    const __m512i m0f = _mm512_set1_epi8(0x0F);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)KVALUES_IQ4NL_U8));
    const __m256i sft4 = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
    const __m256i sft2 = _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14);
    const __m256i v0f = _mm256_set1_epi32(0x0F);
    const __m256i v03 = _mm256_set1_epi32(0x03);
    const __m256i v32 = _mm256_set1_epi32(32);

    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * rb;
        float accf[NT] = {};
        for (int i = 0; i < nb; ++i) {
            const uint8_t* bp = row + (size_t)i * sizeof(block_iq4_xs);
            _mm_prefetch((const char*)bp + kPrefetchAhead, _MM_HINT_T0);
            _mm_prefetch((const char*)bp + kPrefetchAhead + 64, _MM_HINT_T0);
            const auto* blk = reinterpret_cast<const block_iq4_xs*>(bp);
            uint16_t dh, sh;
            uint32_t sl;
            std::memcpy(&dh, &blk->d, 2);
            std::memcpy(&sh, &blk->scales_h, 2);
            std::memcpy(&sl, blk->scales_l, 4);
            const float dw = h2f(dh);

            const __m256i vsl = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)sl), sft4), v0f);
            const __m256i vsh = _mm256_slli_epi32(_mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32((int)sh), sft2), v03), 4);
            const __m256i sc8 = _mm256_sub_epi32(_mm256_or_si256(vsl, vsh), v32);

            const __m512i q0 = _mm512_loadu_si512((const void*)(blk->qs + 0));
            const __m512i q1 = _mm512_loadu_si512((const void*)(blk->qs + 64));
            const __m512i wlo03 = _mm512_shuffle_epi8(lut, _mm512_and_si512(q0, m0f));
            const __m512i whi03 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(q0, 4), m0f));
            const __m512i wlo47 = _mm512_shuffle_epi8(lut, _mm512_and_si512(q1, m0f));
            const __m512i whi47 = _mm512_shuffle_epi8(lut, _mm512_and_si512(_mm512_srli_epi16(q1, 4), m0f));

            for (int t = 0; t < NT; ++t) {
                const PrepIq4Xs& ax = xp[t][i];
                __m512i p03 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), wlo03, ax.xlo03), whi03, ax.xhi03);
                __m512i p47 = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), wlo47, ax.xlo47), whi47, ax.xhi47);
                const __m256i s8 = _mm256_sub_epi32(reduce4_pair_to_256(p03, p47), ax.bias8);
                const int32_t isum = hsum_i32_256(_mm256_mullo_epi32(s8, sc8));
                accf[t] = std::fma(dw * ax.d, (float)isum, accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            acc_out[t][r - r0] = accf[t];
        }
    }
}

template <int NT, typename PrepT = PrepQ5K>
void dot_rows_q4_k(
    const uint8_t* w,
    size_t rb,
    int nb,
    const PrepT* const* xp,
    int r0,
    int r1,
    float* const* acc_out
) {
    const __m512i m0f = _mm512_set1_epi8(0x0F);

    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * rb;
        float accf[NT] = {};
        __m256 accv[NT];
        if constexpr (std::is_same_v<PrepT, PrepQ5K_Q81>) {
            for (int t = 0; t < NT; ++t) accv[t] = _mm256_setzero_ps();
            (void)accf;
        } else {
            (void)accv;
        }
        for (int i = 0; i < nb; ++i) {
            const uint8_t* bp = row + (size_t)i * sizeof(block_q4_K);
            _mm_prefetch((const char*)bp + kPrefetchAhead, _MM_HINT_T0);
            _mm_prefetch((const char*)bp + kPrefetchAhead + 64, _MM_HINT_T0);
            const auto* blk = reinterpret_cast<const block_q4_K*>(bp);
            uint16_t dh, dmh;
            std::memcpy(&dh, &blk->d, 2);
            std::memcpy(&dmh, &blk->dmin, 2);
            const float dw = h2f(dh);
            const float dmin = h2f(dmh);

            alignas(32) int32_t sc_arr[8];
            alignas(32) int32_t mn_arr[8];
            for (int j = 0; j < 4; ++j) {
                sc_arr[j] = blk->scales[j] & 63;
                mn_arr[j] = blk->scales[j + 4] & 63;
                sc_arr[j + 4] = (blk->scales[j + 8] & 0x0F) | ((blk->scales[j] >> 6) << 4);
                mn_arr[j + 4] = (blk->scales[j + 8] >> 4) | ((blk->scales[j + 4] >> 6) << 4);
            }
            const __m256i sc8 = _mm256_load_si256((const __m256i*)sc_arr);
            const __m256i mn8 = _mm256_load_si256((const __m256i*)mn_arr);

            const __m512i raw0 = _mm512_loadu_si512((const void*)(blk->qs + 0));
            const __m512i raw1 = _mm512_loadu_si512((const void*)(blk->qs + 64));

            const __m512i we02 = _mm512_and_si512(raw0, m0f);
            const __m512i wo13 = _mm512_and_si512(_mm512_srli_epi16(raw0, 4), m0f);
            const __m512i we46 = _mm512_and_si512(raw1, m0f);
            const __m512i wo57 = _mm512_and_si512(_mm512_srli_epi16(raw1, 4), m0f);

            if constexpr (std::is_same_v<PrepT, PrepQ5K_Q81>) {
                const __m256 d_sc8 = _mm256_mul_ps(_mm256_set1_ps(dw), _mm256_cvtepi32_ps(sc8));
                const __m256 dm_mn8 = _mm256_mul_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(mn8));
                for (int t = 0; t < NT; ++t) {
                    const PrepT& ax = xp[t][i];
                    const __m512i pe02 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), we02, ax.xe02);
                    const __m512i po13 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wo13, ax.xo13);
                    const __m512i pe46 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), we46, ax.xe46);
                    const __m512i po57 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wo57, ax.xo57);
                    const __m256i s8 = reduce8_interleaved_to_256(pe02, po13, pe46, po57);
                    accv[t] = _mm256_fmadd_ps(_mm256_mul_ps(d_sc8, ax.d8), _mm256_cvtepi32_ps(s8), accv[t]);
                    accv[t] = _mm256_fnmadd_ps(dm_mn8, ax.s8, accv[t]);
                }
            } else {
                for (int t = 0; t < NT; ++t) {
                    const PrepT& ax = xp[t][i];
                    const __m512i pe02 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), we02, ax.xe02);
                    const __m512i po13 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wo13, ax.xo13);
                    const __m512i pe46 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), we46, ax.xe46);
                    const __m512i po57 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wo57, ax.xo57);
                    const __m256i s8 = reduce8_interleaved_to_256(pe02, po13, pe46, po57);
                    const int32_t sum_d = hsum_i32_256(_mm256_mullo_epi32(s8, sc8));
                    const int32_t sum_m = hsum_i32_256(_mm256_mullo_epi32(ax.bsum8, mn8));
                    accf[t] = std::fma(dw * ax.d, (float)sum_d, accf[t]);
                    accf[t] = std::fma(-dmin * ax.d, (float)sum_m, accf[t]);
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            if constexpr (std::is_same_v<PrepT, PrepQ5K_Q81>) {
                acc_out[t][r - r0] = hsum_f32_256(accv[t]);
            } else {
                acc_out[t][r - r0] = accf[t];
            }
        }
    }
}

template <int NT, typename PrepT = PrepQ5K>
void dot_rows_q5_K(
    const uint8_t* w,
    size_t rb,
    int nb,
    const PrepT* const* xp,
    int r0,
    int r1,
    float* const* acc_out
) {
    const __m512i m0f = _mm512_set1_epi8(0x0F);
    const __m512i v16 = _mm512_set1_epi8(16);
    const __m512i qh_m_e02 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8(0x01)), _mm256_set1_epi8(0x04), 1);
    const __m512i qh_m_o13 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8(0x02)), _mm256_set1_epi8(0x08), 1);
    const __m512i qh_m_e46 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8(0x10)), _mm256_set1_epi8(0x40), 1);
    const __m512i qh_m_o57 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_set1_epi8(0x20)), _mm256_set1_epi8((char)0x80), 1);

    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * rb;
        float accf[NT] = {};
        __m256 accv[NT];
        if constexpr (std::is_same_v<PrepT, PrepQ5K_Q81>) {
            for (int t = 0; t < NT; ++t) accv[t] = _mm256_setzero_ps();
            (void)accf;
        } else {
            (void)accv;
        }
        for (int i = 0; i < nb; ++i) {
            const uint8_t* bp = row + (size_t)i * sizeof(block_q5_K);
            _mm_prefetch((const char*)bp + kPrefetchAhead, _MM_HINT_T0);
            _mm_prefetch((const char*)bp + kPrefetchAhead + 64, _MM_HINT_T0);
            const auto* blk = reinterpret_cast<const block_q5_K*>(bp);
            uint16_t dh, dmh;
            std::memcpy(&dh, &blk->d, 2);
            std::memcpy(&dmh, &blk->dmin, 2);
            const float dw = h2f(dh);
            const float dmin = h2f(dmh);

            alignas(32) int32_t sc_arr[8];
            alignas(32) int32_t mn_arr[8];
            for (int j = 0; j < 4; ++j) {
                sc_arr[j] = blk->scales[j] & 63;
                mn_arr[j] = blk->scales[j + 4] & 63;
                sc_arr[j + 4] = (blk->scales[j + 8] & 0x0F) | ((blk->scales[j] >> 6) << 4);
                mn_arr[j + 4] = (blk->scales[j + 8] >> 4) | ((blk->scales[j + 4] >> 6) << 4);
            }
            const __m256i sc8 = _mm256_load_si256((const __m256i*)sc_arr);
            const __m256i mn8 = _mm256_load_si256((const __m256i*)mn_arr);

            const __m256i qh256 = _mm256_loadu_si256((const __m256i*)blk->qh);
            const __m512i qh512 = _mm512_inserti64x4(_mm512_castsi256_si512(qh256), qh256, 1);
            const __m512i raw0 = _mm512_loadu_si512((const void*)(blk->qs + 0));
            const __m512i raw1 = _mm512_loadu_si512((const void*)(blk->qs + 64));

            const __m512i lo0 = _mm512_and_si512(raw0, m0f);
            const __m512i hi0 = _mm512_and_si512(_mm512_srli_epi16(raw0, 4), m0f);
            const __m512i lo1 = _mm512_and_si512(raw1, m0f);
            const __m512i hi1 = _mm512_and_si512(_mm512_srli_epi16(raw1, 4), m0f);

            const __m512i we02 = _mm512_mask_add_epi8(lo0, _mm512_test_epi8_mask(qh512, qh_m_e02), lo0, v16);
            const __m512i wo13 = _mm512_mask_add_epi8(hi0, _mm512_test_epi8_mask(qh512, qh_m_o13), hi0, v16);
            const __m512i we46 = _mm512_mask_add_epi8(lo1, _mm512_test_epi8_mask(qh512, qh_m_e46), lo1, v16);
            const __m512i wo57 = _mm512_mask_add_epi8(hi1, _mm512_test_epi8_mask(qh512, qh_m_o57), hi1, v16);

            if constexpr (std::is_same_v<PrepT, PrepQ5K_Q81>) {
                const __m256 d_sc8 = _mm256_mul_ps(_mm256_set1_ps(dw), _mm256_cvtepi32_ps(sc8));
                const __m256 dm_mn8 = _mm256_mul_ps(_mm256_set1_ps(dmin), _mm256_cvtepi32_ps(mn8));
                for (int t = 0; t < NT; ++t) {
                    const PrepT& ax = xp[t][i];
                    const __m512i pe02 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), we02, ax.xe02);
                    const __m512i po13 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wo13, ax.xo13);
                    const __m512i pe46 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), we46, ax.xe46);
                    const __m512i po57 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wo57, ax.xo57);
                    const __m256i s8 = reduce8_interleaved_to_256(pe02, po13, pe46, po57);
                    accv[t] = _mm256_fmadd_ps(_mm256_mul_ps(d_sc8, ax.d8), _mm256_cvtepi32_ps(s8), accv[t]);
                    accv[t] = _mm256_fnmadd_ps(dm_mn8, ax.s8, accv[t]);
                }
            } else {
                for (int t = 0; t < NT; ++t) {
                    const PrepT& ax = xp[t][i];
                    const __m512i pe02 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), we02, ax.xe02);
                    const __m512i po13 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wo13, ax.xo13);
                    const __m512i pe46 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), we46, ax.xe46);
                    const __m512i po57 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wo57, ax.xo57);
                    const __m256i s8 = reduce8_interleaved_to_256(pe02, po13, pe46, po57);
                    const int32_t sum_d = hsum_i32_256(_mm256_mullo_epi32(s8, sc8));
                    const int32_t sum_m = hsum_i32_256(_mm256_mullo_epi32(ax.bsum8, mn8));
                    accf[t] = std::fma(dw * ax.d, (float)sum_d, accf[t]);
                    accf[t] = std::fma(-dmin * ax.d, (float)sum_m, accf[t]);
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            if constexpr (std::is_same_v<PrepT, PrepQ5K_Q81>) {
                acc_out[t][r - r0] = hsum_f32_256(accv[t]);
            } else {
                acc_out[t][r - r0] = accf[t];
            }
        }
    }
}

template <int NT>
void dot_rows_q6_K(
    const uint8_t* w,
    size_t rb,
    int nb,
    const PrepQ6K* const* xp,
    int r0,
    int r1,
    float* const* acc_out
) {
    const __m512i m0f = _mm512_set1_epi8(0x0F);
    const __m512i m30 = _mm512_set1_epi8(0x30);

    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * rb;
        float accf[NT] = {};
        for (int i = 0; i < nb; ++i) {
            const uint8_t* bp = row + (size_t)i * sizeof(block_q6_K);
            _mm_prefetch((const char*)bp + kPrefetchAhead, _MM_HINT_T0);
            _mm_prefetch((const char*)bp + kPrefetchAhead + 64, _MM_HINT_T0);
            const auto* blk = reinterpret_cast<const block_q6_K*>(bp);
            uint16_t dh;
            std::memcpy(&dh, &blk->d, 2);
            const float dw = h2f(dh);
            const __m512i sc16 = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)blk->scales));

            // Half 0
            const __m512i ql_0 = _mm512_loadu_si512((const void*)(blk->ql + 0));
            const __m256i qh_0_256 = _mm256_loadu_si256((const __m256i*)(blk->qh + 0));
            const __m512i qh_0 = _mm512_inserti64x4(_mm512_castsi256_si512(qh_0_256), qh_0_256, 1);
            const __m512i q12_0_hi = _mm512_and_si512(
                _mm512_mask_mov_epi32(_mm512_slli_epi16(qh_0, 4), 0xFF00u, _mm512_slli_epi16(qh_0, 2)), m30);
            const __m512i q34_0_hi = _mm512_and_si512(
                _mm512_mask_mov_epi32(qh_0, 0xFF00u, _mm512_srli_epi16(qh_0, 2)), m30);
            const __m512i q12_0 = _mm512_or_si512(_mm512_and_si512(ql_0, m0f), q12_0_hi);
            const __m512i q34_0 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(ql_0, 4), m0f), q34_0_hi);

            // Half 1
            const __m512i ql_1 = _mm512_loadu_si512((const void*)(blk->ql + 64));
            const __m256i qh_1_256 = _mm256_loadu_si256((const __m256i*)(blk->qh + 32));
            const __m512i qh_1 = _mm512_inserti64x4(_mm512_castsi256_si512(qh_1_256), qh_1_256, 1);
            const __m512i q12_1_hi = _mm512_and_si512(
                _mm512_mask_mov_epi32(_mm512_slli_epi16(qh_1, 4), 0xFF00u, _mm512_slli_epi16(qh_1, 2)), m30);
            const __m512i q34_1_hi = _mm512_and_si512(
                _mm512_mask_mov_epi32(qh_1, 0xFF00u, _mm512_srli_epi16(qh_1, 2)), m30);
            const __m512i q12_1 = _mm512_or_si512(_mm512_and_si512(ql_1, m0f), q12_1_hi);
            const __m512i q34_1 = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(ql_1, 4), m0f), q34_1_hi);

            for (int t = 0; t < NT; ++t) {
                const PrepQ6K& ax = xp[t][i];
                const __m512i p0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q12_0, ax.x0);
                const __m512i p1 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q34_0, ax.x1);
                const __m512i p2 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q12_1, ax.x2);
                const __m512i p3 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), q34_1, ax.x3);
                const __m256i lo8 = reduce4_pair_to_256(p0, p1);
                const __m256i hi8 = reduce4_pair_to_256(p2, p3);
                const __m512i raw16 = _mm512_sub_epi32(_mm512_inserti64x4(_mm512_castsi256_si512(lo8), hi8, 1), ax.bias16);
                const int32_t isum = _mm512_reduce_add_epi32(_mm512_mullo_epi32(raw16, sc16));
                accf[t] = std::fma(dw * ax.d, (float)isum, accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            acc_out[t][r - r0] = accf[t];
        }
    }
}

template <int NT, typename PrepT = PrepQ8_0>
void dot_rows_q8_0(
    const uint8_t* w,
    size_t rb,
    int n_in,
    const PrepT* const* xp,
    int r0,
    int r1,
    float* const* acc_out
) {
    const __m512i k80 = _mm512_set1_epi8((char)0x80);
    const int nb_full = n_in / QK_K;
    const int rem32 = (n_in - nb_full * QK_K) / 32;

    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();

        for (int i = 0; i < nb_full; ++i) {
            const auto* b = reinterpret_cast<const block_q8_0*>(row + (size_t)(i * 8) * sizeof(block_q8_0));
            _mm_prefetch((const char*)b + kPrefetchAhead, _MM_HINT_T0);
            _mm_prefetch((const char*)b + kPrefetchAhead + 64, _MM_HINT_T0);

            uint16_t dh[8];
            for (int k = 0; k < 8; ++k) std::memcpy(&dh[k], &b[k].d, 2);
            const __m256 dw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));

            auto load_pair = [&](int p) -> __m512i {
                __m256i w0 = _mm256_loadu_si256((const __m256i*)b[2 * p + 0].qs);
                __m256i w1 = _mm256_loadu_si256((const __m256i*)b[2 * p + 1].qs);
                return _mm512_xor_si512(_mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1), k80);
            };
            const __m512i w01 = load_pair(0);
            const __m512i w23 = load_pair(1);
            const __m512i w45 = load_pair(2);
            const __m512i w67 = load_pair(3);

            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][i];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i s8 = _mm256_sub_epi32(reduce8_quad_to_256(p01, p23, p45, p67), ax.bias8);
                __m256 d_ax;
                if constexpr (std::is_same_v<PrepT, PrepQ8_0_Q81>) {
                    d_ax = ax.d8;
                } else {
                    d_ax = _mm256_set1_ps(ax.d);
                }
                const __m256 d_prod = _mm256_mul_ps(dw8, d_ax);
                accf[t] = _mm256_fmadd_ps(d_prod, _mm256_cvtepi32_ps(s8), accf[t]);
            }
        }
        if (rem32 > 0) {
            const auto* bt = reinterpret_cast<const block_q8_0*>(row + (size_t)(nb_full * 8) * sizeof(block_q8_0));
            uint16_t dh[8] = {};
            for (int k = 0; k < rem32; ++k) std::memcpy(&dh[k], &bt[k].d, 2);
            const __m256 dw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));
            auto load_tail_pair = [&](int p) -> __m512i {
                const int k0 = 2 * p, k1 = 2 * p + 1;
                if (k1 < rem32) {
                    __m256i w0 = _mm256_loadu_si256((const __m256i*)bt[k0].qs);
                    __m256i w1 = _mm256_loadu_si256((const __m256i*)bt[k1].qs);
                    return _mm512_xor_si512(_mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1), k80);
                } else if (k0 < rem32) {
                    __m256i w0 = _mm256_loadu_si256((const __m256i*)bt[k0].qs);
                    return _mm512_xor_si512(_mm512_inserti64x4(_mm512_castsi256_si512(w0), _mm256_setzero_si256(), 1), k80);
                }
                return _mm512_setzero_si512();
            };
            const __m512i w01 = load_tail_pair(0);
            const __m512i w23 = load_tail_pair(1);
            const __m512i w45 = load_tail_pair(2);
            const __m512i w67 = load_tail_pair(3);
            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][nb_full];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i s8 = _mm256_sub_epi32(reduce8_quad_to_256(p01, p23, p45, p67), ax.bias8);
                __m256 d_ax;
                if constexpr (std::is_same_v<PrepT, PrepQ8_0_Q81>) {
                    d_ax = ax.d8;
                } else {
                    d_ax = _mm256_set1_ps(ax.d);
                }
                const __m256 d_prod = _mm256_mul_ps(dw8, d_ax);
                accf[t] = _mm256_fmadd_ps(d_prod, _mm256_cvtepi32_ps(s8), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            acc_out[t][r - r0] = hsum_f32_256(accf[t]);
        }
    }
}

template <int NT, typename PrepT = PrepQ5_1>
void dot_rows_q5_1(
    const uint8_t* w,
    size_t rb,
    int n_in,
    const PrepT* const* xp,
    int r0,
    int r1,
    float* const* acc_out
) {
    const __m256i m0f_256 = _mm256_set1_epi8(0x0F);
    const __m512i v16 = _mm512_set1_epi8(16);
    const int nb_full = n_in / QK_K;
    const int rem32 = (n_in - nb_full * QK_K) / 32;

    auto load_q5_1_pair = [&](const block_q5_1* b0, const block_q5_1* b1) -> __m512i {
        const __m128i q0 = _mm_loadu_si128((const __m128i*)b0->qs);
        const __m128i q1 = b1 ? _mm_loadu_si128((const __m128i*)b1->qs) : _mm_setzero_si128();
        const __m256i q01 = _mm256_set_m128i(q1, q0);
        const __m256i lo = _mm256_and_si256(q01, m0f_256);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q01, 4), m0f_256);
        const __m256i wb0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        const __m256i wb1 = _mm256_permute2x128_si256(lo, hi, 0x31);
        const __m512i w4b = _mm512_inserti64x4(_mm512_castsi256_si512(wb0), wb1, 1);
        uint32_t qh0 = 0, qh1 = 0;
        std::memcpy(&qh0, b0->qh, 4);
        if (b1) std::memcpy(&qh1, b1->qh, 4);
        const __mmask64 km = (__mmask64)(((uint64_t)qh1 << 32) | (uint64_t)qh0);
        return _mm512_mask_add_epi8(w4b, km, w4b, v16);
    };

    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();

        for (int i = 0; i < nb_full; ++i) {
            const auto* b = reinterpret_cast<const block_q5_1*>(row + (size_t)(i * 8) * sizeof(block_q5_1));
            _mm_prefetch((const char*)b + kPrefetchAhead, _MM_HINT_T0);
            _mm_prefetch((const char*)b + kPrefetchAhead + 64, _MM_HINT_T0);

            uint16_t dh[8], mh[8];
            for (int k = 0; k < 8; ++k) {
                std::memcpy(&dh[k], &b[k].d, 2);
                std::memcpy(&mh[k], &b[k].m, 2);
            }
            const __m256 dw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));
            const __m256 mw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)mh));

            const __m512i w01 = load_q5_1_pair(&b[0], &b[1]);
            const __m512i w23 = load_q5_1_pair(&b[2], &b[3]);
            const __m512i w45 = load_q5_1_pair(&b[4], &b[5]);
            const __m512i w67 = load_q5_1_pair(&b[6], &b[7]);

            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][i];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i s8 = reduce8_quad_to_256(p01, p23, p45, p67);
                if constexpr (std::is_same_v<PrepT, PrepQ5_1_Q81>) {
                    accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, ax.d8), _mm256_cvtepi32_ps(s8), accf[t]);
                    accf[t] = _mm256_fmadd_ps(mw8, ax.s8, accf[t]);
                } else {
                    const __m256 ax_d = _mm256_set1_ps(ax.d);
                    accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, ax_d), _mm256_cvtepi32_ps(s8), accf[t]);
                    accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(mw8, ax_d), ax.bsum8f, accf[t]);
                }
            }
        }
        if (rem32 > 0) {
            const auto* bt = reinterpret_cast<const block_q5_1*>(row + (size_t)(nb_full * 8) * sizeof(block_q5_1));
            uint16_t dh[8] = {};
            uint16_t mh[8] = {};
            for (int k = 0; k < rem32; ++k) {
                std::memcpy(&dh[k], &bt[k].d, 2);
                std::memcpy(&mh[k], &bt[k].m, 2);
            }
            const __m256 dw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));
            const __m256 mw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)mh));
            auto load_tail = [&](int p) -> __m512i {
                const int k0 = 2 * p, k1 = 2 * p + 1;
                if (k1 < rem32) return load_q5_1_pair(&bt[k0], &bt[k1]);
                if (k0 < rem32) return load_q5_1_pair(&bt[k0], nullptr);
                return _mm512_setzero_si512();
            };
            const __m512i w01 = load_tail(0);
            const __m512i w23 = load_tail(1);
            const __m512i w45 = load_tail(2);
            const __m512i w67 = load_tail(3);
            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][nb_full];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i s8 = reduce8_quad_to_256(p01, p23, p45, p67);
                if constexpr (std::is_same_v<PrepT, PrepQ5_1_Q81>) {
                    accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, ax.d8), _mm256_cvtepi32_ps(s8), accf[t]);
                    accf[t] = _mm256_fmadd_ps(mw8, ax.s8, accf[t]);
                } else {
                    const __m256 ax_d = _mm256_set1_ps(ax.d);
                    accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, ax_d), _mm256_cvtepi32_ps(s8), accf[t]);
                    accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(mw8, ax_d), ax.bsum8f, accf[t]);
                }
            }
        }
        for (int t = 0; t < NT; ++t) {
            acc_out[t][r - r0] = hsum_f32_256(accf[t]);
        }
    }
}

template <int NT, typename PrepT = PrepQ8_0>
void dot_rows_iq4_nl(
    const uint8_t* w,
    size_t rb,
    int n_in,
    const PrepT* const* xp,
    int r0,
    int r1,
    float* const* acc_out
) {
    const __m256i m0f_256 = _mm256_set1_epi8(0x0F);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i*)KVALUES_IQ4NL_U8));
    const int nb_full = n_in / QK_K;
    const int rem32 = (n_in - nb_full * QK_K) / 32;

    auto load_iq4_nl_pair = [&](const block_iq4_nl* b0, const block_iq4_nl* b1) -> __m512i {
        const __m128i q0 = _mm_loadu_si128((const __m128i*)b0->qs);
        const __m128i q1 = b1 ? _mm_loadu_si128((const __m128i*)b1->qs) : _mm_setzero_si128();
        const __m256i q01 = _mm256_set_m128i(q1, q0);
        const __m256i lo = _mm256_and_si256(q01, m0f_256);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q01, 4), m0f_256);
        const __m256i wb0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        const __m256i wb1 = _mm256_permute2x128_si256(lo, hi, 0x31);
        const __m512i w4b = _mm512_inserti64x4(_mm512_castsi256_si512(wb0), wb1, 1);
        return _mm512_shuffle_epi8(lut, w4b);
    };

    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();

        for (int i = 0; i < nb_full; ++i) {
            const auto* b = reinterpret_cast<const block_iq4_nl*>(row + (size_t)(i * 8) * sizeof(block_iq4_nl));
            _mm_prefetch((const char*)b + kPrefetchAhead, _MM_HINT_T0);
            _mm_prefetch((const char*)b + kPrefetchAhead + 64, _MM_HINT_T0);

            uint16_t dh[8];
            for (int k = 0; k < 8; ++k) std::memcpy(&dh[k], &b[k].d, 2);
            const __m256 dw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));

            const __m512i w01 = load_iq4_nl_pair(&b[0], &b[1]);
            const __m512i w23 = load_iq4_nl_pair(&b[2], &b[3]);
            const __m512i w45 = load_iq4_nl_pair(&b[4], &b[5]);
            const __m512i w67 = load_iq4_nl_pair(&b[6], &b[7]);

            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][i];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i s8 = _mm256_sub_epi32(reduce8_quad_to_256(p01, p23, p45, p67), ax.bias8);
                __m256 d_ax;
                if constexpr (std::is_same_v<PrepT, PrepQ8_0_Q81>) {
                    d_ax = ax.d8;
                } else {
                    d_ax = _mm256_set1_ps(ax.d);
                }
                accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, d_ax), _mm256_cvtepi32_ps(s8), accf[t]);
            }
        }
        if (rem32 > 0) {
            const auto* bt = reinterpret_cast<const block_iq4_nl*>(row + (size_t)(nb_full * 8) * sizeof(block_iq4_nl));
            uint16_t dh[8] = {};
            for (int k = 0; k < rem32; ++k) std::memcpy(&dh[k], &bt[k].d, 2);
            const __m256 dw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));
            auto load_tail = [&](int p) -> __m512i {
                const int k0 = 2 * p, k1 = 2 * p + 1;
                if (k1 < rem32) return load_iq4_nl_pair(&bt[k0], &bt[k1]);
                if (k0 < rem32) return load_iq4_nl_pair(&bt[k0], nullptr);
                return _mm512_setzero_si512();
            };
            const __m512i w01 = load_tail(0);
            const __m512i w23 = load_tail(1);
            const __m512i w45 = load_tail(2);
            const __m512i w67 = load_tail(3);
            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][nb_full];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i s8 = _mm256_sub_epi32(reduce8_quad_to_256(p01, p23, p45, p67), ax.bias8);
                __m256 d_ax;
                if constexpr (std::is_same_v<PrepT, PrepQ8_0_Q81>) {
                    d_ax = ax.d8;
                } else {
                    d_ax = _mm256_set1_ps(ax.d);
                }
                accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, d_ax), _mm256_cvtepi32_ps(s8), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            acc_out[t][r - r0] = hsum_f32_256(accf[t]);
        }
    }
}

template <int NT, typename PrepT = PrepQ8_0>
void dot_rows_q5_0(
    const uint8_t* w,
    size_t rb,
    int n_in,
    const PrepT* const* xp,
    int r0,
    int r1,
    float* const* acc_out
) {
    const __m256i m0f_256 = _mm256_set1_epi8(0x0F);
    const __m512i v16 = _mm512_set1_epi8(16);
    const int nb_full = n_in / QK_K;
    const int rem32 = (n_in - nb_full * QK_K) / 32;

    auto load_q5_0_pair = [&](const block_q5_0* b0, const block_q5_0* b1) -> __m512i {
        const __m128i q0 = _mm_loadu_si128((const __m128i*)b0->qs);
        const __m128i q1 = b1 ? _mm_loadu_si128((const __m128i*)b1->qs) : _mm_setzero_si128();
        const __m256i q01 = _mm256_set_m128i(q1, q0);
        const __m256i lo = _mm256_and_si256(q01, m0f_256);
        const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q01, 4), m0f_256);
        const __m256i wb0 = _mm256_permute2x128_si256(lo, hi, 0x20);
        const __m256i wb1 = _mm256_permute2x128_si256(lo, hi, 0x31);
        const __m512i w4b = _mm512_inserti64x4(_mm512_castsi256_si512(wb0), wb1, 1);
        uint32_t qh0 = 0, qh1 = 0;
        std::memcpy(&qh0, b0->qh, 4);
        if (b1) std::memcpy(&qh1, b1->qh, 4);
        const __mmask64 km = (__mmask64)(((uint64_t)qh1 << 32) | (uint64_t)qh0);
        return _mm512_mask_add_epi8(w4b, km, w4b, v16);
    };

    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();

        for (int i = 0; i < nb_full; ++i) {
            const auto* b = reinterpret_cast<const block_q5_0*>(row + (size_t)(i * 8) * sizeof(block_q5_0));
            _mm_prefetch((const char*)b + kPrefetchAhead, _MM_HINT_T0);
            _mm_prefetch((const char*)b + kPrefetchAhead + 64, _MM_HINT_T0);

            uint16_t dh[8];
            for (int k = 0; k < 8; ++k) std::memcpy(&dh[k], &b[k].d, 2);
            const __m256 dw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));

            const __m512i w01 = load_q5_0_pair(&b[0], &b[1]);
            const __m512i w23 = load_q5_0_pair(&b[2], &b[3]);
            const __m512i w45 = load_q5_0_pair(&b[4], &b[5]);
            const __m512i w67 = load_q5_0_pair(&b[6], &b[7]);

            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][i];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i bias16 = _mm256_srai_epi32(ax.bias8, 3);
                const __m256i s8 = _mm256_sub_epi32(reduce8_quad_to_256(p01, p23, p45, p67), bias16);
                __m256 d_ax;
                if constexpr (std::is_same_v<PrepT, PrepQ8_0_Q81>) {
                    d_ax = ax.d8;
                } else {
                    d_ax = _mm256_set1_ps(ax.d);
                }
                accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, d_ax), _mm256_cvtepi32_ps(s8), accf[t]);
            }
        }
        if (rem32 > 0) {
            const auto* bt = reinterpret_cast<const block_q5_0*>(row + (size_t)(nb_full * 8) * sizeof(block_q5_0));
            uint16_t dh[8] = {};
            for (int k = 0; k < rem32; ++k) std::memcpy(&dh[k], &bt[k].d, 2);
            const __m256 dw8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));
            auto load_tail = [&](int p) -> __m512i {
                const int k0 = 2 * p, k1 = 2 * p + 1;
                if (k1 < rem32) return load_q5_0_pair(&bt[k0], &bt[k1]);
                if (k0 < rem32) return load_q5_0_pair(&bt[k0], nullptr);
                return _mm512_setzero_si512();
            };
            const __m512i w01 = load_tail(0);
            const __m512i w23 = load_tail(1);
            const __m512i w45 = load_tail(2);
            const __m512i w67 = load_tail(3);
            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][nb_full];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i bias16 = _mm256_srai_epi32(ax.bias8, 3);
                const __m256i s8 = _mm256_sub_epi32(reduce8_quad_to_256(p01, p23, p45, p67), bias16);
                __m256 d_ax;
                if constexpr (std::is_same_v<PrepT, PrepQ8_0_Q81>) {
                    d_ax = ax.d8;
                } else {
                    d_ax = _mm256_set1_ps(ax.d);
                }
                accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, d_ax), _mm256_cvtepi32_ps(s8), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            acc_out[t][r - r0] = hsum_f32_256(accf[t]);
        }
    }
}

template <int NT, typename PrepT = PrepQ8_0>
void dot_rows_q2_0(
    const uint8_t* w,
    size_t rb,
    int n_in,
    const PrepT* const* xp,
    int r0,
    int r1,
    float* const* acc_out
) {
    const __m512i m03_0 = _mm512_set1_epi32(0x00000003);
    const __m512i m03_1 = _mm512_set1_epi32(0x00000300);
    const __m512i m03_2 = _mm512_set1_epi32(0x00030000);
    const __m512i m03_3 = _mm512_set1_epi32(0x03000000);
    const __m256i dup_idx = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const int nb_full = n_in / QK_K;
    const int rem64 = (n_in - nb_full * QK_K) / 64;

    auto unpack_q2_0_blk = [&](const block_q2_0* b) -> __m512i {
        const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*)b->qs));
        const __m512i e0 = _mm512_and_si512(v, m03_0);
        const __m512i e1 = _mm512_and_si512(_mm512_slli_epi32(v, 6), m03_1);
        const __m512i e2 = _mm512_and_si512(_mm512_slli_epi32(v, 12), m03_2);
        const __m512i e3 = _mm512_and_si512(_mm512_slli_epi32(v, 18), m03_3);
        return _mm512_or_si512(_mm512_or_si512(e0, e1), _mm512_or_si512(e2, e3));
    };

    for (int r = r0; r < r1; ++r) {
        const uint8_t* row = w + (size_t)r * rb;
        __m256 accf[NT];
        for (int t = 0; t < NT; ++t) accf[t] = _mm256_setzero_ps();

        for (int i = 0; i < nb_full; ++i) {
            const auto* b = reinterpret_cast<const block_q2_0*>(row + (size_t)(i * 4) * sizeof(block_q2_0));
            _mm_prefetch((const char*)b + kPrefetchAhead, _MM_HINT_T0);
            _mm_prefetch((const char*)b + kPrefetchAhead + 64, _MM_HINT_T0);

            uint16_t dh[8] = {};
            for (int k = 0; k < 4; ++k) std::memcpy(&dh[k], &b[k].d, 2);
            const __m256 dw4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));
            const __m256 dw8 = _mm256_permutevar8x32_ps(dw4, dup_idx);

            const __m512i w01 = unpack_q2_0_blk(&b[0]);
            const __m512i w23 = unpack_q2_0_blk(&b[1]);
            const __m512i w45 = unpack_q2_0_blk(&b[2]);
            const __m512i w67 = unpack_q2_0_blk(&b[3]);

            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][i];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i bias1 = _mm256_srai_epi32(ax.bias8, 7);
                const __m256i s8 = _mm256_sub_epi32(reduce8_quad_to_256(p01, p23, p45, p67), bias1);
                __m256 d_ax;
                if constexpr (std::is_same_v<PrepT, PrepQ8_0_Q81>) {
                    d_ax = ax.d8;
                } else {
                    d_ax = _mm256_set1_ps(ax.d);
                }
                accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, d_ax), _mm256_cvtepi32_ps(s8), accf[t]);
            }
        }
        if (rem64 > 0) {
            const auto* bt = reinterpret_cast<const block_q2_0*>(row + (size_t)(nb_full * 4) * sizeof(block_q2_0));
            uint16_t dh[8] = {};
            for (int k = 0; k < rem64; ++k) std::memcpy(&dh[k], &bt[k].d, 2);
            const __m256 dw4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*)dh));
            const __m256 dw8 = _mm256_permutevar8x32_ps(dw4, dup_idx);
            const __m512i w01 = (0 < rem64) ? unpack_q2_0_blk(&bt[0]) : _mm512_setzero_si512();
            const __m512i w23 = (1 < rem64) ? unpack_q2_0_blk(&bt[1]) : _mm512_setzero_si512();
            const __m512i w45 = (2 < rem64) ? unpack_q2_0_blk(&bt[2]) : _mm512_setzero_si512();
            const __m512i w67 = (3 < rem64) ? unpack_q2_0_blk(&bt[3]) : _mm512_setzero_si512();
            for (int t = 0; t < NT; ++t) {
                const PrepT& ax = xp[t][nb_full];
                const __m512i p01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, ax.x01);
                const __m512i p23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, ax.x23);
                const __m512i p45 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w45, ax.x45);
                const __m512i p67 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w67, ax.x67);
                const __m256i bias1 = _mm256_srai_epi32(ax.bias8, 7);
                const __m256i s8 = _mm256_sub_epi32(reduce8_quad_to_256(p01, p23, p45, p67), bias1);
                __m256 d_ax;
                if constexpr (std::is_same_v<PrepT, PrepQ8_0_Q81>) {
                    d_ax = ax.d8;
                } else {
                    d_ax = _mm256_set1_ps(ax.d);
                }
                accf[t] = _mm256_fmadd_ps(_mm256_mul_ps(dw8, d_ax), _mm256_cvtepi32_ps(s8), accf[t]);
            }
        }
        for (int t = 0; t < NT; ++t) {
            acc_out[t][r - r0] = hsum_f32_256(accf[t]);
        }
    }
}

template <int NT, typename PrepT, void (*DotFn)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*)>
void dispatch_gu_tile(
    const uint8_t* gate_w,
    const uint8_t* up_w,
    size_t row_bytes,
    int nb,
    int dot_param,
    const std::vector<PrepT>& prepped,
    int t_off,
    float* const* ff_out,
    int r0,
    int r1,
    float swiglu_limit
) {
    const PrepT* xp[NT];
    for (int t = 0; t < NT; ++t) xp[t] = prepped.data() + (size_t)(t_off + t) * nb;

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
        const int rr1 = std::min(r1, rr0 + kTile);
        const int nr = rr1 - rr0;
        DotFn(gate_w, row_bytes, dot_param, xp, rr0, rr1, gp);
        DotFn(up_w, row_bytes, dot_param, xp, rr0, rr1, up);
        for (int t = 0; t < NT; ++t) {
            float* dst = ff_out[t_off + t] + rr0;
            for (int r = 0; r < nr; ++r) {
                dst[r] = clamped_swiglu(g_buf[t][r], u_buf[t][r], swiglu_limit);
            }
        }
    }
}

template <int NT, typename PrepT, void (*DotFn)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*)>
void dispatch_dn_tile(
    const uint8_t* down_w,
    size_t row_bytes,
    int nb,
    int dot_param,
    const std::vector<PrepT>& prepped,
    int t_off,
    float* const* out,
    int r0,
    int r1,
    float weight_scale,
    bool accum
) {
    const PrepT* xp[NT];
    for (int t = 0; t < NT; ++t) xp[t] = prepped.data() + (size_t)(t_off + t) * nb;

    constexpr int kTile = 64;
    float d_buf[NT][kTile];
    float* dp[NT];
    for (int t = 0; t < NT; ++t) dp[t] = d_buf[t];

    for (int rr0 = r0; rr0 < r1; rr0 += kTile) {
        const int rr1 = std::min(r1, rr0 + kTile);
        const int nr = rr1 - rr0;
        DotFn(down_w, row_bytes, dot_param, xp, rr0, rr1, dp);
        for (int t = 0; t < NT; ++t) {
            float* dst = out[t_off + t] + rr0;
            if (accum) {
                for (int r = 0; r < nr; ++r) dst[r] += weight_scale * d_buf[t][r];
            } else {
                for (int r = 0; r < nr; ++r) dst[r] = weight_scale * d_buf[t][r];
            }
        }
    }
}

template <typename InBlkT, typename PrepT, void (*PrepFn)(const InBlkT*, int, PrepT*),
          void (*Dot1)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*),
          void (*Dot2)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*),
          void (*Dot4)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*),
          void (*Dot8)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*),
          bool PassNIn = false>
void run_gu_type(
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
    const int nb = (n_in + QK_K - 1) / QK_K;
    const int dot_param = PassNIn ? n_in : nb;
    thread_local std::vector<PrepT> tls_prep;
    tls_prep.resize((size_t)nt * nb);
    for (int t = 0; t < nt; ++t) {
        PrepFn(reinterpret_cast<const InBlkT*>(x_q8k[t]), nb, tls_prep.data() + (size_t)t * nb);
    }
    int t = 0;
    while (t + 8 <= nt) {
        dispatch_gu_tile<8, PrepT, Dot8>(gate_w, up_w, row_bytes, nb, dot_param, tls_prep, t, ff_out, r0, r1, swiglu_limit);
        t += 8;
    }
    while (t + 4 <= nt) {
        dispatch_gu_tile<4, PrepT, Dot4>(gate_w, up_w, row_bytes, nb, dot_param, tls_prep, t, ff_out, r0, r1, swiglu_limit);
        t += 4;
    }
    while (t + 2 <= nt) {
        dispatch_gu_tile<2, PrepT, Dot2>(gate_w, up_w, row_bytes, nb, dot_param, tls_prep, t, ff_out, r0, r1, swiglu_limit);
        t += 2;
    }
    if (t < nt) {
        dispatch_gu_tile<1, PrepT, Dot1>(gate_w, up_w, row_bytes, nb, dot_param, tls_prep, t, ff_out, r0, r1, swiglu_limit);
    }
}

template <typename InBlkT, typename PrepT, void (*PrepFn)(const InBlkT*, int, PrepT*),
          void (*Dot1)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*),
          void (*Dot2)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*),
          void (*Dot4)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*),
          void (*Dot8)(const uint8_t*, size_t, int, const PrepT* const*, int, int, float* const*),
          bool PassNIn = false>
void run_dn_type(
    const uint8_t* down_w,
    size_t row_bytes,
    int n_in,
    const void* const* ff_q8,
    int nt,
    float* const* out,
    int r0,
    int r1,
    float weight_scale,
    bool accum
) {
    const int nb = (n_in + QK_K - 1) / QK_K;
    const int dot_param = PassNIn ? n_in : nb;
    thread_local std::vector<PrepT> tls_prep;
    tls_prep.resize((size_t)nt * nb);
    for (int t = 0; t < nt; ++t) {
        PrepFn(reinterpret_cast<const InBlkT*>(ff_q8[t]), nb, tls_prep.data() + (size_t)t * nb);
    }
    int t = 0;
    while (t + 8 <= nt) {
        dispatch_dn_tile<8, PrepT, Dot8>(down_w, row_bytes, nb, dot_param, tls_prep, t, out, r0, r1, weight_scale, accum);
        t += 8;
    }
    while (t + 4 <= nt) {
        dispatch_dn_tile<4, PrepT, Dot4>(down_w, row_bytes, nb, dot_param, tls_prep, t, out, r0, r1, weight_scale, accum);
        t += 4;
    }
    while (t + 2 <= nt) {
        dispatch_dn_tile<2, PrepT, Dot2>(down_w, row_bytes, nb, dot_param, tls_prep, t, out, r0, r1, weight_scale, accum);
        t += 2;
    }
    if (t < nt) {
        dispatch_dn_tile<1, PrepT, Dot1>(down_w, row_bytes, nb, dot_param, tls_prep, t, out, r0, r1, weight_scale, accum);
    }
}

} // namespace

extern "C" void ampllm_cpu_quantize_q8_1_g8(
    const float* x,
    void* out_g8,
    int32_t n
) {
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
            float tmp[32];
            const float* p = x + off;
            if (rem < 32) {
                std::memset(tmp, 0, sizeof(tmp));
                std::memcpy(tmp, p, (size_t)rem * sizeof(float));
                p = tmp;
            }
            const __m512 v0 = _mm512_loadu_ps(p);
            const __m512 v1 = _mm512_loadu_ps(p + 16);
            const __m512 av = _mm512_max_ps(_mm512_abs_ps(v0), _mm512_abs_ps(v1));
            const float amax = _mm512_reduce_max_ps(av);
            const float d = amax / 127.0f;
            const float id = (amax != 0.0f) ? (127.0f / amax) : 0.0f;
            const __m512 vid = _mm512_set1_ps(id);
            const __m512i q0 = _mm512_cvt_roundps_epi32(
                _mm512_mul_ps(v0, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const __m512i q1 = _mm512_cvt_roundps_epi32(
                _mm512_mul_ps(v1, vid), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const int32_t sumq = _mm512_reduce_add_epi32(_mm512_add_epi32(q0, q1));
            _mm_storeu_si128((__m128i*)(out.qs + k * 32), _mm512_cvtsepi32_epi8(q0));
            _mm_storeu_si128((__m128i*)(out.qs + k * 32 + 16), _mm512_cvtsepi32_epi8(q1));
            out.d[k] = d;
            out.bsum[k] = sumq;
            out.s[k] = d * (float)sumq;
        }
    }
}

extern "C" void ampllm_cpu_hy4_fused_gu_rows(
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
    switch (ggml_type) {
        case GGML_TYPE_IQ4_XS:
            run_gu_type<block_q8_K, PrepIq4Xs, prepare_iq4_xs, dot_rows_iq4_xs<1>, dot_rows_iq4_xs<2>, dot_rows_iq4_xs<4>, dot_rows_iq4_xs<8>>(
                gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
            break;
        case GGML_TYPE_Q4_K:
            run_gu_type<block_q8_K, PrepQ5K, prepare_q5_K, dot_rows_q4_k<1>, dot_rows_q4_k<2>, dot_rows_q4_k<4>, dot_rows_q4_k<8>>(
                gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
            break;
        case GGML_TYPE_Q5_K:
            run_gu_type<block_q8_K, PrepQ5K, prepare_q5_K, dot_rows_q5_K<1>, dot_rows_q5_K<2>, dot_rows_q5_K<4>, dot_rows_q5_K<8>>(
                gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
            break;
        case GGML_TYPE_Q6_K:
            run_gu_type<block_q8_K, PrepQ6K, prepare_q6_K, dot_rows_q6_K<1>, dot_rows_q6_K<2>, dot_rows_q6_K<4>, dot_rows_q6_K<8>>(
                gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
            break;
        case GGML_TYPE_Q5_1:
            run_gu_type<block_q8_K, PrepQ5_1, prepare_q5_1, dot_rows_q5_1<1>, dot_rows_q5_1<2>, dot_rows_q5_1<4>, dot_rows_q5_1<8>, true>(
                gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
            break;
        case GGML_TYPE_Q8_0:
            run_gu_type<block_q8_K, PrepQ8_0, prepare_q8_0, dot_rows_q8_0<1>, dot_rows_q8_0<2>, dot_rows_q8_0<4>, dot_rows_q8_0<8>, true>(
                gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
            break;
        case GGML_TYPE_IQ4_NL:
            run_gu_type<block_q8_K, PrepQ8_0, prepare_q8_0, dot_rows_iq4_nl<1>, dot_rows_iq4_nl<2>, dot_rows_iq4_nl<4>, dot_rows_iq4_nl<8>, true>(
                gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
            break;
        case GGML_TYPE_Q5_0:
            run_gu_type<block_q8_K, PrepQ8_0, prepare_q8_0, dot_rows_q5_0<1>, dot_rows_q5_0<2>, dot_rows_q5_0<4>, dot_rows_q5_0<8>, true>(
                gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
            break;
        case GGML_TYPE_Q2_0:
            run_gu_type<block_q8_K, PrepQ8_0, prepare_q8_0, dot_rows_q2_0<1>, dot_rows_q2_0<2>, dot_rows_q2_0<4>, dot_rows_q2_0<8>, true>(
                gate_w, up_w, row_bytes, n_in, x_q8k, nt, ff_out, r0, r1, swiglu_limit);
            break;
        default:
            break;
    }
}

extern "C" void ampllm_cpu_hy4_fused_gu_rows_q8_1(
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
    switch (ggml_type) {
        case GGML_TYPE_Q4_K:
            run_gu_type<block_q8_1_g8, PrepQ5K_Q81, prepare_q5_K_q81, dot_rows_q4_k<1, PrepQ5K_Q81>, dot_rows_q4_k<2, PrepQ5K_Q81>, dot_rows_q4_k<4, PrepQ5K_Q81>, dot_rows_q4_k<8, PrepQ5K_Q81>>(
                gate_w, up_w, row_bytes, n_in, x_q8_1, nt, ff_out, r0, r1, swiglu_limit);
            break;
        case GGML_TYPE_Q5_K:
            run_gu_type<block_q8_1_g8, PrepQ5K_Q81, prepare_q5_K_q81, dot_rows_q5_K<1, PrepQ5K_Q81>, dot_rows_q5_K<2, PrepQ5K_Q81>, dot_rows_q5_K<4, PrepQ5K_Q81>, dot_rows_q5_K<8, PrepQ5K_Q81>>(
                gate_w, up_w, row_bytes, n_in, x_q8_1, nt, ff_out, r0, r1, swiglu_limit);
            break;
        default:
            break;
    }
}

extern "C" void ampllm_cpu_hy4_down_rows(
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
    const bool acc = accumulate != 0;
    switch (ggml_type) {
        case GGML_TYPE_IQ4_XS:
            run_dn_type<block_q8_K, PrepIq4Xs, prepare_iq4_xs, dot_rows_iq4_xs<1>, dot_rows_iq4_xs<2>, dot_rows_iq4_xs<4>, dot_rows_iq4_xs<8>>(
                down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q4_K:
            run_dn_type<block_q8_K, PrepQ5K, prepare_q5_K, dot_rows_q4_k<1>, dot_rows_q4_k<2>, dot_rows_q4_k<4>, dot_rows_q4_k<8>>(
                down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q5_K:
            run_dn_type<block_q8_K, PrepQ5K, prepare_q5_K, dot_rows_q5_K<1>, dot_rows_q5_K<2>, dot_rows_q5_K<4>, dot_rows_q5_K<8>>(
                down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q6_K:
            run_dn_type<block_q8_K, PrepQ6K, prepare_q6_K, dot_rows_q6_K<1>, dot_rows_q6_K<2>, dot_rows_q6_K<4>, dot_rows_q6_K<8>>(
                down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q5_1:
            run_dn_type<block_q8_K, PrepQ5_1, prepare_q5_1, dot_rows_q5_1<1>, dot_rows_q5_1<2>, dot_rows_q5_1<4>, dot_rows_q5_1<8>, true>(
                down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q8_0:
            run_dn_type<block_q8_K, PrepQ8_0, prepare_q8_0, dot_rows_q8_0<1>, dot_rows_q8_0<2>, dot_rows_q8_0<4>, dot_rows_q8_0<8>, true>(
                down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_IQ4_NL:
            run_dn_type<block_q8_K, PrepQ8_0, prepare_q8_0, dot_rows_iq4_nl<1>, dot_rows_iq4_nl<2>, dot_rows_iq4_nl<4>, dot_rows_iq4_nl<8>, true>(
                down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q5_0:
            run_dn_type<block_q8_K, PrepQ8_0, prepare_q8_0, dot_rows_q5_0<1>, dot_rows_q5_0<2>, dot_rows_q5_0<4>, dot_rows_q5_0<8>, true>(
                down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q2_0:
            run_dn_type<block_q8_K, PrepQ8_0, prepare_q8_0, dot_rows_q2_0<1>, dot_rows_q2_0<2>, dot_rows_q2_0<4>, dot_rows_q2_0<8>, true>(
                down_w, row_bytes, n_in, ff_q8k, nt, out, r0, r1, weight_scale, acc);
            break;
        default:
            break;
    }
}

extern "C" void ampllm_cpu_hy4_down_rows_q8_1(
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
    const bool acc = accumulate != 0;
    switch (ggml_type) {
        case GGML_TYPE_Q5_1:
            run_dn_type<block_q8_1_g8, PrepQ5_1_Q81, prepare_q5_1_q81, dot_rows_q5_1<1, PrepQ5_1_Q81>, dot_rows_q5_1<2, PrepQ5_1_Q81>, dot_rows_q5_1<4, PrepQ5_1_Q81>, dot_rows_q5_1<8, PrepQ5_1_Q81>, true>(
                down_w, row_bytes, n_in, ff_q8_1, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q8_0:
            run_dn_type<block_q8_1_g8, PrepQ8_0_Q81, prepare_q8_0_q81, dot_rows_q8_0<1, PrepQ8_0_Q81>, dot_rows_q8_0<2, PrepQ8_0_Q81>, dot_rows_q8_0<4, PrepQ8_0_Q81>, dot_rows_q8_0<8, PrepQ8_0_Q81>, true>(
                down_w, row_bytes, n_in, ff_q8_1, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_IQ4_NL:
            run_dn_type<block_q8_1_g8, PrepQ8_0_Q81, prepare_q8_0_q81, dot_rows_iq4_nl<1, PrepQ8_0_Q81>, dot_rows_iq4_nl<2, PrepQ8_0_Q81>, dot_rows_iq4_nl<4, PrepQ8_0_Q81>, dot_rows_iq4_nl<8, PrepQ8_0_Q81>, true>(
                down_w, row_bytes, n_in, ff_q8_1, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q5_0:
            run_dn_type<block_q8_1_g8, PrepQ8_0_Q81, prepare_q8_0_q81, dot_rows_q5_0<1, PrepQ8_0_Q81>, dot_rows_q5_0<2, PrepQ8_0_Q81>, dot_rows_q5_0<4, PrepQ8_0_Q81>, dot_rows_q5_0<8, PrepQ8_0_Q81>, true>(
                down_w, row_bytes, n_in, ff_q8_1, nt, out, r0, r1, weight_scale, acc);
            break;
        case GGML_TYPE_Q2_0:
            run_dn_type<block_q8_1_g8, PrepQ8_0_Q81, prepare_q8_0_q81, dot_rows_q2_0<1, PrepQ8_0_Q81>, dot_rows_q2_0<2, PrepQ8_0_Q81>, dot_rows_q2_0<4, PrepQ8_0_Q81>, dot_rows_q2_0<8, PrepQ8_0_Q81>, true>(
                down_w, row_bytes, n_in, ff_q8_1, nt, out, r0, r1, weight_scale, acc);
            break;
        default:
            break;
    }
}

extern "C" void ampllm_cpu_bf16_to_q8_0(
    const uint16_t* src_bf16,
    const float* src_f32,
    size_t rows,
    int32_t k,
    uint8_t* out_q8_0
) {
    if (k <= 0 || (k & 31) != 0) return;
    const int nb = k / 32;
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7FFFFFFF));
    const __m512 clamp_hi = _mm512_set1_ps(127.0f);
    const __m512 clamp_lo = _mm512_set1_ps(-127.0f);

    for (size_t r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            const size_t at = r * (size_t)k + (size_t)b * 32;
            __m512 v0, v1;
            if (src_f32 != nullptr) {
                v0 = _mm512_loadu_ps(src_f32 + at);
                v1 = _mm512_loadu_ps(src_f32 + at + 16);
            } else {
                const __m256i h0 = _mm256_loadu_si256((const __m256i*)(src_bf16 + at));
                const __m256i h1 = _mm256_loadu_si256((const __m256i*)(src_bf16 + at + 16));
                v0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h0), 16));
                v1 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h1), 16));
            }
            const __m512 vmax = _mm512_max_ps(_mm512_and_ps(v0, abs_mask), _mm512_and_ps(v1, abs_mask));
            const float am = _mm512_reduce_max_ps(vmax);
            const float d = am / 127.0f;
            const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;

            uint8_t* dst = out_q8_0 + (r * (size_t)nb + (size_t)b) * sizeof(block_q8_0);
            const uint16_t hd = f2h_rne(d);
            std::memcpy(dst, &hd, 2);

            const __m512 vid = _mm512_set1_ps(id);
            const __m512 q0f = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, _mm512_mul_ps(v0, vid)));
            const __m512 q1f = _mm512_max_ps(clamp_lo, _mm512_min_ps(clamp_hi, _mm512_mul_ps(v1, vid)));
            // _mm512_cvtps_epi32 uses round-to-nearest-even (mxcsr default), fast and clean; store lower 8 bits of each i32
            const __m512i q0i = _mm512_cvtps_epi32(q0f);
            const __m512i q1i = _mm512_cvtps_epi32(q1f);
            _mm512_mask_cvtepi32_storeu_epi8(dst + 2, 0xFFFFu, q0i);
            _mm512_mask_cvtepi32_storeu_epi8(dst + 18, 0xFFFFu, q1i);
        }
    }
}

extern "C" void ampllm_cpu_tok_row_to_f32(
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
            for (; i + 16 <= k; i += 16) {
                __m256i h = _mm256_loadu_si256((const __m256i*)(p + i));
                _mm512_storeu_ps(dst_f32 + i, _mm512_cvtph_ps(h));
            }
            for (; i < k; ++i) dst_f32[i] = h2f(p[i]);
            break;
        }
        case GGML_TYPE_BF16: {
            const uint16_t* p = reinterpret_cast<const uint16_t*>(src);
            int i = 0;
            for (; i + 16 <= k; i += 16) {
                __m256i h = _mm256_loadu_si256((const __m256i*)(p + i));
                __m512 f = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(h), 16));
                _mm512_storeu_ps(dst_f32 + i, f);
            }
            for (; i < k; ++i) dst_f32[i] = bf16_to_f32_scalar(p[i]);
            break;
        }
        case GGML_TYPE_Q8_0: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_q8_0*>(src);
            for (int b = 0; b < nb; ++b) {
                uint16_t dh;
                std::memcpy(&dh, &blks[b].d, 2);
                const __m512 vd = _mm512_set1_ps(h2f(dh));
                const __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(blks[b].qs + 0))));
                const __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*)(blks[b].qs + 16))));
                _mm512_storeu_ps(dst_f32 + b * 32 + 0, _mm512_mul_ps(vd, f0));
                _mm512_storeu_ps(dst_f32 + b * 32 + 16, _mm512_mul_ps(vd, f1));
            }
            break;
        }
        case GGML_TYPE_Q5_1: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_q5_1*>(src);
            for (int b = 0; b < nb; ++b) {
                uint16_t dh, mh;
                std::memcpy(&dh, &blks[b].d, 2);
                std::memcpy(&mh, &blks[b].m, 2);
                const float d = h2f(dh);
                const float m = h2f(mh);
                uint32_t qh = 0;
                std::memcpy(&qh, blks[b].qh, 4);
                float* dst = dst_f32 + b * 32;
                for (int j = 0; j < 16; ++j) {
                    const uint32_t x0 = (blks[b].qs[j] & 0x0F) | (((qh >> j) & 1u) << 4);
                    const uint32_t x1 = (blks[b].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4);
                    dst[j]      = d * (float)x0 + m;
                    dst[j + 16] = d * (float)x1 + m;
                }
            }
            break;
        }
        case GGML_TYPE_IQ4_NL: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_iq4_nl*>(src);
            const __m128i lut = _mm_loadu_si128((const __m128i*)KVALUES_IQ4NL);
            const __m128i m0f_128 = _mm_set1_epi8(0x0F);
            for (int b = 0; b < nb; ++b) {
                uint16_t dh;
                std::memcpy(&dh, &blks[b].d, 2);
                const __m512 vd = _mm512_set1_ps(h2f(dh));
                const __m128i q = _mm_loadu_si128((const __m128i*)blks[b].qs);
                const __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m0f_128));
                const __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m0f_128));
                const __m512 f0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo));
                const __m512 f1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi));
                _mm512_storeu_ps(dst_f32 + b * 32 + 0, _mm512_mul_ps(vd, f0));
                _mm512_storeu_ps(dst_f32 + b * 32 + 16, _mm512_mul_ps(vd, f1));
            }
            break;
        }
        case GGML_TYPE_Q4_K: {
            const int nb = k / QK_K;
            const auto* blks = reinterpret_cast<const block_q4_K*>(src);
            for (int b = 0; b < nb; ++b) {
                uint16_t dh, dmh;
                std::memcpy(&dh, &blks[b].d, 2);
                std::memcpy(&dmh, &blks[b].dmin, 2);
                const float d = h2f(dh), dm = h2f(dmh);
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
                uint16_t dh, dmh;
                std::memcpy(&dh, &blks[b].d, 2);
                std::memcpy(&dmh, &blks[b].dmin, 2);
                const float d = h2f(dh), dm = h2f(dmh);
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
                uint16_t dh;
                std::memcpy(&dh, &blks[b].d, 2);
                const float d = h2f(dh);
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
                uint16_t dh, sh;
                std::memcpy(&dh, &blks[b].d, 2);
                std::memcpy(&sh, &blks[b].scales_h, 2);
                const float d = h2f(dh);
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
        case GGML_TYPE_Q5_0: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_q5_0*>(src);
            for (int b = 0; b < nb; ++b) {
                uint16_t dh;
                std::memcpy(&dh, &blks[b].d, 2);
                const float d = h2f(dh);
                uint32_t qh = 0;
                std::memcpy(&qh, blks[b].qh, 4);
                float* dst = dst_f32 + b * 32;
                for (int j = 0; j < 16; ++j) {
                    const int x0 = (int)((blks[b].qs[j] & 0x0F) | (((qh >> j) & 1u) << 4)) - 16;
                    const int x1 = (int)((blks[b].qs[j] >> 4) | (((qh >> (j + 16)) & 1u) << 4)) - 16;
                    dst[j]      = d * (float)x0;
                    dst[j + 16] = d * (float)x1;
                }
            }
            break;
        }
        case GGML_TYPE_Q4_0: {
            const int nb = k / 32;
            const auto* blks = reinterpret_cast<const block_q8_0*>(0);
            (void)blks;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* bp = src + (size_t)b * 18;
                uint16_t dh;
                std::memcpy(&dh, bp, 2);
                const float d = h2f(dh);
                float* dst = dst_f32 + b * 32;
                for (int j = 0; j < 16; ++j) {
                    dst[j]      = d * (float)((int)(bp[2 + j] & 0x0F) - 8);
                    dst[j + 16] = d * (float)((int)(bp[2 + j] >> 4) - 8);
                }
            }
            break;
        }
        case GGML_TYPE_Q2_0: {
            const int nb = k / 64;
            const auto* blks = reinterpret_cast<const block_q2_0*>(src);
            for (int b = 0; b < nb; ++b) {
                uint16_t dh;
                std::memcpy(&dh, &blks[b].d, 2);
                const float d = h2f(dh);
                float* dst = dst_f32 + b * 64;
                for (int j = 0; j < 16; ++j) {
                    const uint8_t q = blks[b].qs[j];
                    for (int s = 0; s < 4; ++s) {
                        dst[4 * j + s] = d * (float)((int)((q >> (2 * s)) & 3u) - 1);
                    }
                }
            }
            break;
        }
        case GGML_TYPE_Q3_K: {
            const int nb = k / QK_K;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* bp = src + (size_t)b * 110;
                const uint8_t* hmask = bp;
                const uint8_t* qs = bp + 32;
                const uint8_t* sc_raw = bp + 96;
                uint16_t dh;
                std::memcpy(&dh, bp + 108, 2);
                const float d_all = h2f(dh);
                uint32_t aux[4] = {};
                std::memcpy(aux, sc_raw, 12);
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
                const uint8_t* q = qs;
                for (int n = 0; n < QK_K; n += 128) {
                    int shift = 0;
                    for (int j = 0; j < 4; ++j) {
                        const float dl0 = d_all * (float)(scales[is++] - 32);
                        for (int l = 0; l < 16; ++l) {
                            const int qv = (int)((q[l] >> shift) & 3) - ((hmask[l] & m) ? 0 : 4);
                            y[n + j * 32 + l] = dl0 * (float)qv;
                        }
                        const float dl1 = d_all * (float)(scales[is++] - 32);
                        for (int l = 0; l < 16; ++l) {
                            const int qv = (int)((q[l + 16] >> shift) & 3) - ((hmask[l + 16] & m) ? 0 : 4);
                            y[n + j * 32 + 16 + l] = dl1 * (float)qv;
                        }
                        shift += 2;
                        m <<= 1;
                    }
                    q += 32;
                }
            }
            break;
        }
        default:
            break;
    }
}
