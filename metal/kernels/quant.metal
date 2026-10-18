// metal/kernels/quant.metal - the weight formats as "units": a unit is UNIT consecutive weights of one row that one
// SIMD lane decodes at a time into raw values q[0..UNIT) and one scale s, so that the unit's dot with x is
// s * sum(q[j] * x[j]).  Every GEMV kernel (moe.metal, gemv.metal) is written once over these.
//
// Decode rules transcribed from ggml-quants.c dequantize_row_* (3cf03257) and Strata's Q2_0 contract
// (docs/q2_0-contract.md); the grids and sign tables are ggml-common.h's (prepended by the runtime).  The tables a
// format needs are copied to threadgroup memory once per threadgroup (constant-memory lookups with divergent indices
// serialise on Apple GPUs).
//
// Blocks start at 2-byte aligned addresses (18, 66, 74, 82, 98, 110 bytes), so every multi-byte field is read as
// ushorts.

#define T_Q4_0 2
#define T_Q3_K 11
#define T_BF16 30
#define T_F32 0
#define T_Q5_0 6
#define T_Q5_1 7
#define T_Q8_0 8
#define T_Q4_K 12
#define T_Q5_K 13
#define T_Q6_K 14
#define T_IQ2_XXS 16
#define T_IQ2_XS 17
#define T_IQ3_XXS 18
#define T_IQ4_NL 20
#define T_IQ3_S 21
#define T_IQ2_S 22
#define T_IQ4_XS 23
#define T_Q2_0 42
#define T_Q2_0R 142   // Q2_0 read with raw activations (codes - 1 in the decoder; no prepared x needed)

template <int F> struct fmt;
// QK block elements, BB block bytes, UNIT weights per unit, TAB words of threadgroup table
template <> struct fmt<T_Q2_0>    { enum { QK = 64,  BB = 18,  UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_Q2_0R>   { enum { QK = 64,  BB = 18,  UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_IQ3_XXS> { enum { QK = 256, BB = 98,  UNIT = 16, TAB = 256 + 32 }; };
template <> struct fmt<T_IQ2_XXS> { enum { QK = 256, BB = 66,  UNIT = 16, TAB = 512 + 32 }; };
template <> struct fmt<T_IQ2_XS>  { enum { QK = 256, BB = 74,  UNIT = 16, TAB = 1024 + 32 }; };
template <> struct fmt<T_IQ2_S>   { enum { QK = 256, BB = 82,  UNIT = 16, TAB = 2048 + 32 }; };
template <> struct fmt<T_IQ3_S>   { enum { QK = 256, BB = 110, UNIT = 16, TAB = 512 + 32 }; };
template <> struct fmt<T_IQ4_NL>  { enum { QK = 32,  BB = 18,  UNIT = 32, TAB = 256 }; };
template <> struct fmt<T_IQ4_XS>  { enum { QK = 256, BB = 136, UNIT = 32, TAB = 256 }; };
template <> struct fmt<T_Q3_K>    { enum { QK = 256, BB = 110, UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q4_K>    { enum { QK = 256, BB = 144, UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q5_K>    { enum { QK = 256, BB = 176, UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q6_K>    { enum { QK = 256, BB = 210, UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_BF16>    { enum { QK = 16,  BB = 32,  UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_F32>     { enum { QK = 16,  BB = 64,  UNIT = 16, TAB = 1 }; };
template <> struct fmt<T_Q8_0>    { enum { QK = 32,  BB = 34,  UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q4_0>    { enum { QK = 32,  BB = 18,  UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q5_0>    { enum { QK = 32,  BB = 22,  UNIT = 32, TAB = 1 }; };
template <> struct fmt<T_Q5_1>    { enum { QK = 32,  BB = 24,  UNIT = 32, TAB = 1 }; };

static inline uint rd32(device const uchar* p) {
    device const ushort* s = (device const ushort*)p;
    return uint(s[0]) | (uint(s[1]) << 16);
}
static inline float rdh(device const uchar* p) { return float(*(device const half*)p); }

// ---- threadgroup tables -------------------------------------------------------------------------------------------
// layout in `tab` (uint words): [0, G) the grid, [G, G + 32) ksigns_iq2xs as bytes; IQ4_NL/XS: 16 floats.
template <int F> static inline void load_tab(threadgroup uint* tab, ushort tid, ushort nth);

// QUANT_CTAB (function constant 23): the i-quant grids and the sign table read from ggml-common's constant arrays
// instead of a threadgroup copy (no table load and no barrier wait at the threadgroup's start); the same values.
constant bool QUANT_CTAB [[function_constant(23)]];
constant bool ctab = is_function_constant_defined(QUANT_CTAB) && QUANT_CTAB;
static inline uint ksg(threadgroup const uint* tab, int at, uint i) {
    return ctab ? uint(ksigns_iq2xs[i]) : uint(((threadgroup const uchar*)(tab + at))[i]);
}
static inline uint g_iq3xxs(threadgroup const uint* tab, uint i) { return ctab ? iq3xxs_grid[i] : tab[i]; }
static inline uint g_iq3s(threadgroup const uint* tab, uint i) { return ctab ? iq3s_grid[i] : tab[i]; }
static inline uint g_iq2xxs(threadgroup const uint* tab, uint i, int w) { return ctab ? uint(iq2xxs_grid[i] >> (32 * w)) : tab[2 * i + w]; }
static inline uint g_iq2xs(threadgroup const uint* tab, uint i, int w) { return ctab ? uint(iq2xs_grid[i] >> (32 * w)) : tab[2 * i + w]; }
static inline uint g_iq2s(threadgroup const uint* tab, uint i, int w) { return ctab ? uint(iq2s_grid[i] >> (32 * w)) : tab[2 * i + w]; }

template <> inline void load_tab<T_Q2_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q2_0R>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q8_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q4_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q5_0>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q5_1>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q3_K>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q4_K>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q5_K>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_Q6_K>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_BF16>(threadgroup uint*, ushort, ushort) {}
template <> inline void load_tab<T_F32>(threadgroup uint*, ushort, ushort) {}

static inline void load_signs(threadgroup uint* tab, int at, ushort tid, ushort nth) {
    threadgroup uchar* s = (threadgroup uchar*)(tab + at);
    for (int i = tid; i < 128; i += nth) s[i] = ksigns_iq2xs[i];
}
template <> inline void load_tab<T_IQ3_XXS>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 256; i += nth) tab[i] = iq3xxs_grid[i];
    load_signs(tab, 256, tid, nth);
}
template <> inline void load_tab<T_IQ3_S>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 512; i += nth) tab[i] = iq3s_grid[i];
    load_signs(tab, 512, tid, nth);
}
template <> inline void load_tab<T_IQ2_XXS>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 256; i += nth) { tab[2 * i] = uint(iq2xxs_grid[i]); tab[2 * i + 1] = uint(iq2xxs_grid[i] >> 32); }
    load_signs(tab, 512, tid, nth);
}
template <> inline void load_tab<T_IQ2_XS>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 512; i += nth) { tab[2 * i] = uint(iq2xs_grid[i]); tab[2 * i + 1] = uint(iq2xs_grid[i] >> 32); }
    load_signs(tab, 1024, tid, nth);
}
template <> inline void load_tab<T_IQ2_S>(threadgroup uint* tab, ushort tid, ushort nth) {
    if (ctab) return;
    for (int i = tid; i < 1024; i += nth) { tab[2 * i] = uint(iq2s_grid[i]); tab[2 * i + 1] = uint(iq2s_grid[i] >> 32); }
    load_signs(tab, 2048, tid, nth);
}
// IQ4: a byte -> (codebook[low nibble], codebook[high nibble]) as half2, so one lookup gives two weights
template <> inline void load_tab<T_IQ4_NL>(threadgroup uint* tab, ushort tid, ushort nth) {
    for (int i = tid; i < 256; i += nth)
        tab[i] = as_type<uint>(half2(half(kvalues_iq4nl[i & 15]), half(kvalues_iq4nl[i >> 4])));
}
template <> inline void load_tab<T_IQ4_XS>(threadgroup uint* tab, ushort tid, ushort nth) {
    load_tab<T_IQ4_NL>(tab, tid, nth);
}

// ---- unit decoders: return s, fill q[UNIT] ------------------------------------------------------------------------
// ui = unit index inside the block (0 .. QK/UNIT - 1).
template <int F> static inline float deq(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab);

// Q2_0: 64 = 4 units of 16; code c (2 bits, LSB first) -> c - 1; scale d.
template <> inline float deq<T_Q2_0>(device const uchar* b, int ui, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)(b + 2 + 4 * ui);
    const ushort a = p[0], c = p[1];
    for (int j = 0; j < 8; ++j) {
        q[j] = float((a >> (2 * j)) & 3) - 1.0f;
        q[8 + j] = float((c >> (2 * j)) & 3) - 1.0f;
    }
    return rdh(b);
}

// 8 signed grid bytes from a 4-byte grid word pair helper
static inline void put_signed4(thread float* q, uint g, uint s) {
    const uchar4 v = as_type<uchar4>(g);
    q[0] = (s & 1) ? -float(v[0]) : float(v[0]);
    q[1] = (s & 2) ? -float(v[1]) : float(v[1]);
    q[2] = (s & 4) ? -float(v[2]) : float(v[2]);
    q[3] = (s & 8) ? -float(v[3]) : float(v[3]);
}

// IQ3_XXS: 256 = 8 x 32; unit = half of a 32 (two grid pairs).
template <> inline float deq<T_IQ3_XXS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint aux = rd32(b + 2 + 64 + 4 * ib);
    device const uchar* qs = b + 2 + 8 * ib + 4 * h;
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint s = ksg(tab, 256, (aux >> (7 * l)) & 127);
        put_signed4(q + 8 * l2, g_iq3xxs(tab, qs[2 * l2]), s);
        put_signed4(q + 8 * l2 + 4, g_iq3xxs(tab, qs[2 * l2 + 1]), s >> 4);
    }
    return rdh(b) * (0.5f + float(aux >> 28)) * 0.5f;
}

// IQ2_XXS: per 32: two u32 (4 grid indices, then 4 x 7 sign bits + 4-bit scale).
template <> inline float deq<T_IQ2_XXS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint a0 = rd32(b + 2 + 8 * ib), a1 = rd32(b + 2 + 8 * ib + 4);
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint idx = (a0 >> (8 * l)) & 255;
        const uint s = ksg(tab, 512, (a1 >> (7 * l)) & 127);
        put_signed4(q + 8 * l2, g_iq2xxs(tab, idx, 0), s);
        put_signed4(q + 8 * l2 + 4, g_iq2xxs(tab, idx, 1), s >> 4);
    }
    return rdh(b) * (0.5f + float(a1 >> 28)) * 0.25f;
}

// IQ2_XS: uint16 qs[32] (9-bit grid index + 7-bit sign index), scales[8] nibbles.
template <> inline float deq<T_IQ2_XS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    device const ushort* qs = (device const ushort*)(b + 2);
    for (int l2 = 0; l2 < 2; ++l2) {
        const uint v = qs[4 * ib + 2 * h + l2];
        const uint idx = v & 511, s = ksg(tab, 1024, v >> 9);
        put_signed4(q + 8 * l2, g_iq2xs(tab, idx, 0), s);
        put_signed4(q + 8 * l2 + 4, g_iq2xs(tab, idx, 1), s >> 4);
    }
    const uint sc = b[66 + ib];
    return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15)) * 0.25f;
}

// IQ2_S: qs[32] low 8 bits of the 10-bit index, signs[32], qh[8] (2 high bits per group of 8), scales[8].
template <> inline float deq<T_IQ2_S>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint qh = b[66 + ib];
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint idx = uint(b[2 + 4 * ib + l]) | ((qh << (8 - 2 * l)) & 0x300);
        const uint s = b[2 + 32 + 4 * ib + l];
        put_signed4(q + 8 * l2, g_iq2s(tab, idx, 0), s);
        put_signed4(q + 8 * l2 + 4, g_iq2s(tab, idx, 1), s >> 4);
    }
    const uint sc = b[74 + ib];
    return rdh(b) * (0.5f + float((sc >> (4 * h)) & 15)) * 0.25f;
}

// IQ3_S: qs[64] low 8 bits of 9-bit indices, qh[8], signs[32], scales[4] (nibble per 32).
template <> inline float deq<T_IQ3_S>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    const int ib = ui >> 1, h = ui & 1;
    const uint qh = b[66 + ib];
    for (int l2 = 0; l2 < 2; ++l2) {
        const int l = 2 * h + l2;
        const uint i1 = uint(b[2 + 8 * ib + 2 * l]) | ((qh << (8 - 2 * l)) & 256);
        const uint i2 = uint(b[2 + 8 * ib + 2 * l + 1]) | ((qh << (7 - 2 * l)) & 256);
        const uint s = b[74 + 4 * ib + l];
        put_signed4(q + 8 * l2, g_iq3s(tab, i1), s);
        put_signed4(q + 8 * l2 + 4, g_iq3s(tab, i2), s >> 4);
    }
    const uint sc = b[106 + (ib >> 1)];
    return rdh(b) * float(1 + 2 * ((sc >> (4 * (ib & 1))) & 15));
}

// IQ4_NL: 32 = 1 unit; low nibbles are elements 0..15, high 16..31 (one half2 lookup per byte).
template <> inline float deq<T_IQ4_NL>(device const uchar* b, int, thread float* q, threadgroup const uint* tab) {
    device const ushort* p = (device const ushort*)(b + 2);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const half2 w0 = as_type<half2>(tab[v & 255]), w1 = as_type<half2>(tab[v >> 8]);
        q[2 * j] = float(w0.x);
        q[2 * j + 16] = float(w0.y);
        q[2 * j + 1] = float(w1.x);
        q[2 * j + 17] = float(w1.y);
    }
    return rdh(b);
}

// IQ4_XS: 256 = 8 units of 32; 6-bit scale from scales_l nibble + 2 bits of scales_h, minus 32.
template <> inline float deq<T_IQ4_XS>(device const uchar* b, int ui, thread float* q, threadgroup const uint* tab) {
    device const ushort* p = (device const ushort*)(b + 8 + 16 * ui);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const half2 w0 = as_type<half2>(tab[v & 255]), w1 = as_type<half2>(tab[v >> 8]);
        q[2 * j] = float(w0.x);
        q[2 * j + 16] = float(w0.y);
        q[2 * j + 1] = float(w1.x);
        q[2 * j + 17] = float(w1.y);
    }
    const uint sh = *(device const ushort*)(b + 2);
    const int ls = int((b[4 + (ui >> 1)] >> (4 * (ui & 1))) & 15) | int(((sh >> (2 * ui)) & 3) << 4);
    return rdh(b) * float(ls - 32);
}

template <> inline float deq<T_Q8_0>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)(b + 2);
    for (int j = 0; j < 16; ++j) {
        const ushort v = p[j];
        q[2 * j] = float(as_type<char>(uchar(v & 255)));
        q[2 * j + 1] = float(as_type<char>(uchar(v >> 8)));
    }
    return rdh(b);
}

template <> inline float deq<T_Q4_0>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)(b + 2);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        q[2 * j] = float(v & 15) - 8.0f;
        q[2 * j + 16] = float((v >> 4) & 15) - 8.0f;
        q[2 * j + 1] = float((v >> 8) & 15) - 8.0f;
        q[2 * j + 17] = float(v >> 12) - 8.0f;
    }
    return rdh(b);
}

template <> inline float deq<T_Q5_0>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    const uint hb = rd32(b + 2);
    device const ushort* p = (device const ushort*)(b + 6);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const int j0 = 2 * j, j1 = 2 * j + 1;
        q[j0] = float(int(v & 15) | int(((hb >> j0) & 1) << 4)) - 16.0f;
        q[j0 + 16] = float(int((v >> 4) & 15) | int(((hb >> (j0 + 16)) & 1) << 4)) - 16.0f;
        q[j1] = float(int((v >> 8) & 15) | int(((hb >> j1) & 1) << 4)) - 16.0f;
        q[j1 + 16] = float(int(v >> 12) | int(((hb >> (j1 + 16)) & 1) << 4)) - 16.0f;
    }
    return rdh(b);
}

// Q5_1: 32 = 1 unit; q = the 5-bit code (no offset; the min m is applied through the x sums), returns d
template <> inline float deq<T_Q5_1>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    const uint hb = rd32(b + 4);
    device const ushort* p = (device const ushort*)(b + 8);
    for (int j = 0; j < 8; ++j) {
        const ushort v = p[j];
        const int j0 = 2 * j, j1 = 2 * j + 1;
        q[j0] = float(int(v & 15) | int(((hb >> j0) & 1) << 4));
        q[j0 + 16] = float(int((v >> 4) & 15) | int(((hb >> (j0 + 16)) & 1) << 4));
        q[j1] = float(int((v >> 8) & 15) | int(((hb >> j1) & 1) << 4));
        q[j1 + 16] = float(int(v >> 12) | int(((hb >> (j1 + 16)) & 1) << 4));
    }
    return rdh(b);
}

template <> inline float deq<T_F32>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    device const float* p = (device const float*)b;
    for (int j = 0; j < 16; ++j) {
        q[j] = p[j];
    }
    return 1.0f;
}

template <> inline float deq<T_BF16>(device const uchar* b, int, thread float* q, threadgroup const uint*) {
    device const ushort* p = (device const ushort*)b;
    for (int j = 0; j < 16; ++j) {
        q[j] = as_type<float>((uint)p[j] << 16);
    }
    return 1.0f;
}

static inline void get_scale_min_k4(int j, device const uchar* q, thread uchar* d, thread uchar* m) {
    if (j < 4) {
        *d = q[j]   & 63;
        *m = q[j+4] & 63;
    } else {
        *d = (q[j+4] & 0x0F) | ((q[j-4] & 0xC0) >> 2);
        *m = ((q[j+4] >> 4) & 0x0F) | ((q[j]   & 0xC0) >> 2);
    }
}

template <> inline float deq<T_Q4_K>(device const uchar* b, int ui, thread float* q, threadgroup const uint*) {
    device const block_q4_K* blk = (device const block_q4_K*)b;
    const float d = float(blk->dm[0]);
    const float dmin = float(blk->dm[1]);
    uchar sc, m;
    get_scale_min_k4(ui, blk->scales, &sc, &m);
    const float d_sc = d * float(sc);
    const float m_sc = dmin * float(m);
    device const uchar* q_src = blk->qs + (ui / 2) * 32;
    if ((ui & 1) == 0) {
        for (int l = 0; l < 32; ++l) {
            q[l] = d_sc * float(q_src[l] & 0x0F) - m_sc;
        }
    } else {
        for (int l = 0; l < 32; ++l) {
            q[l] = d_sc * float(q_src[l] >> 4) - m_sc;
        }
    }
    return 1.0f;
}

template <> inline float deq<T_Q5_K>(device const uchar* b, int ui, thread float* q, threadgroup const uint*) {
    device const block_q5_K* blk = (device const block_q5_K*)b;
    const float d = float(blk->dm[0]);
    const float dmin = float(blk->dm[1]);
    uchar sc, m;
    get_scale_min_k4(ui, blk->scales, &sc, &m);
    const float d_sc = d * float(sc);
    const float m_sc = dmin * float(m);
    device const uchar* ql = blk->qs + (ui / 2) * 32;
    device const uchar* qh = blk->qh;
    const uint u_bit = 1u << ui;
    if ((ui & 1) == 0) {
        for (int l = 0; l < 32; ++l) {
            const int val = (ql[l] & 0x0F) + ((qh[l] & u_bit) ? 16 : 0);
            q[l] = d_sc * float(val) - m_sc;
        }
    } else {
        for (int l = 0; l < 32; ++l) {
            const int val = (ql[l] >> 4) + ((qh[l] & u_bit) ? 16 : 0);
            q[l] = d_sc * float(val) - m_sc;
        }
    }
    return 1.0f;
}

template <> inline float deq<T_Q6_K>(device const uchar* b, int ui, thread float* q, threadgroup const uint*) {
    device const block_q6_K* blk = (device const block_q6_K*)b;
    const float d = float(blk->d);
    const int h_part = ui / 4;
    const int sub = ui % 4;
    device const uchar* ql_base = blk->ql + h_part * 64;
    device const uchar* qh_base = blk->qh + h_part * 32;
    device const char* sc_base = (device const char*)blk->scales + h_part * 8;

    const int ql_offset = (sub & 1) ? 32 : 0;
    const int ql_shift = (sub >= 2) ? 4 : 0;
    const int qh_shift = 2 * sub;
    const int sc_offset = 2 * sub;

    for (int l = 0; l < 32; ++l) {
        const int is = l / 16;
        const int low = (ql_base[l + ql_offset] >> ql_shift) & 0x0F;
        const int high = ((qh_base[l] >> qh_shift) & 3) << 4;
        const int q_val = (low | high) - 32;
        q[l] = d * float(sc_base[is + sc_offset]) * float(q_val);
    }
    return 1.0f;
}

template <> inline float deq<T_Q3_K>(device const uchar* b, int ui, thread float* q, threadgroup const uint*) {
    device const block_q3_K* blk = (device const block_q3_K*)b;
    const float d_all = float(blk->d);

    device const uint* sc_p = (device const uint*)blk->scales;
    const uint a0 = sc_p[0];
    const uint a1 = sc_p[1];
    const uint a2 = sc_p[2];

    uint aux_val;
    if (ui < 2) {
        aux_val = (a0 & 0x0f0f0f0f) | (((a2 >> 0) & 0x03030303) << 4);
    } else if (ui < 4) {
        aux_val = (a1 & 0x0f0f0f0f) | (((a2 >> 2) & 0x03030303) << 4);
    } else if (ui < 6) {
        aux_val = ((a0 >> 4) & 0x0f0f0f0f) | (((a2 >> 4) & 0x03030303) << 4);
    } else {
        aux_val = ((a1 >> 4) & 0x0f0f0f0f) | (((a2 >> 6) & 0x03030303) << 4);
    }
    const int sub_pair = ui % 2;
    const int sc0 = as_type<char>(uchar((aux_val >> (16 * sub_pair)) & 0xFF)) - 32;
    const int sc1 = as_type<char>(uchar((aux_val >> (16 * sub_pair + 8)) & 0xFF)) - 32;

    const float dl0 = d_all * float(sc0);
    const float dl1 = d_all * float(sc1);

    const int h_part = ui / 4;
    const int j = ui % 4;
    const int shift = 2 * j;
    const uint m = 1u << ui;
    device const uchar* q_base = blk->qs + h_part * 32;
    device const uchar* hm = blk->hmask;

    for (int l = 0; l < 16; ++l) {
        const int q0 = (q_base[l] >> shift) & 3;
        const int val0 = q0 - ((hm[l] & m) ? 0 : 4);
        q[l] = dl0 * float(val0);

        const int q1 = (q_base[l + 16] >> shift) & 3;
        const int val1 = q1 - ((hm[l + 16] & m) ? 0 : 4);
        q[l + 16] = dl1 * float(val1);
    }
    return 1.0f;
}

// sum over groups of LPR consecutive lanes (LPR = 1, 2, 4, 8, 16 or 32); the result is valid in every lane
template <int LPR>
static inline float lane_sum(float v) {
    if (LPR == 32) return simd_sum(v);
    for (ushort o = LPR / 2; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);
    return v;
}

template <int F>
METAL_FUNC float deq_dot(device const uchar* b, int ui, device const float* xu, threadgroup const uint* tab) {
    constexpr int UNIT = fmt<F>::UNIT;
    thread float q[UNIT];
    const float scale = deq<F>(b, ui, q, tab);
    float udot = 0.0f;
    for (int l = 0; l < UNIT; ++l) {
        udot += q[l] * xu[l];
    }
    return scale * udot;
}

template <>
METAL_FUNC float deq_dot<T_Q4_K>(device const uchar* b, int ui, device const float* xu, threadgroup const uint*) {
    device const block_q4_K* blk = (device const block_q4_K*)b;
    const float d = float(blk->dm[0]);
    const float dmin = float(blk->dm[1]);
    uchar sc, m;
    get_scale_min_k4(ui, blk->scales, &sc, &m);
    const float d_sc = d * float(sc);
    const float m_sc = dmin * float(m);
    device const uint* q_src32 = (device const uint*)(blk->qs + (ui / 2) * 32);
    device const float4* x4 = (device const float4*)xu;

    float acc_dot = 0.0f;
    float acc_xsum = 0.0f;
    const int shift = (ui & 1) ? 4 : 0;

    for (int g = 0; g < 8; ++g) {
        const uint w = q_src32[g];
        const float4 v = float4(float((w >> shift) & 0x0F),
                                float(((w >> 8) >> shift) & 0x0F),
                                float(((w >> 16) >> shift) & 0x0F),
                                float(((w >> 24) >> shift) & 0x0F));
        const float4 x = x4[g];
        acc_dot += dot(v, x);
        acc_xsum += (x.x + x.y) + (x.z + x.w);
    }
    return d_sc * acc_dot - m_sc * acc_xsum;
}

template <>
METAL_FUNC float deq_dot<T_Q5_K>(device const uchar* b, int ui, device const float* xu, threadgroup const uint*) {
    device const block_q5_K* blk = (device const block_q5_K*)b;
    const float d = float(blk->dm[0]);
    const float dmin = float(blk->dm[1]);
    uchar sc, m;
    get_scale_min_k4(ui, blk->scales, &sc, &m);
    const float d_sc = d * float(sc);
    const float m_sc = dmin * float(m);
    device const uint* ql32 = (device const uint*)(blk->qs + (ui / 2) * 32);
    device const uint* qh32 = (device const uint*)(blk->qh);
    device const float4* x4 = (device const float4*)xu;
    const uint u_bit = 1u << ui;
    const int shift = (ui & 1) ? 4 : 0;

    float acc_dot = 0.0f;
    float acc_xsum = 0.0f;

    for (int g = 0; g < 8; ++g) {
        const uint wl = ql32[g];
        const uint wh = qh32[g];
        const float4 v = float4(float(((wl >> shift) & 0x0F) + ((wh & u_bit) ? 16 : 0)),
                                float((((wl >> 8) >> shift) & 0x0F) + (((wh >> 8) & u_bit) ? 16 : 0)),
                                float((((wl >> 16) >> shift) & 0x0F) + (((wh >> 16) & u_bit) ? 16 : 0)),
                                float((((wl >> 24) >> shift) & 0x0F) + (((wh >> 24) & u_bit) ? 16 : 0)));
        const float4 x = x4[g];
        acc_dot += dot(v, x);
        acc_xsum += (x.x + x.y) + (x.z + x.w);
    }
    return d_sc * acc_dot - m_sc * acc_xsum;
}

template <>
METAL_FUNC float deq_dot<T_Q6_K>(device const uchar* b, int ui, device const float* xu, threadgroup const uint*) {
    device const block_q6_K* blk = (device const block_q6_K*)b;
    const float d = float(blk->d);
    const int h_part = ui / 4;
    const int sub = ui % 4;
    device const uchar* ql_base = blk->ql + h_part * 64;
    device const uchar* qh_base = blk->qh + h_part * 32;
    device const char* sc_base = (device const char*)blk->scales + h_part * 8;

    const int ql_offset = (sub & 1) ? 32 : 0;
    const int ql_shift = (sub >= 2) ? 4 : 0;
    const int qh_shift = 2 * sub;
    const int sc_offset = 2 * sub;

    const float sc0 = d * float(sc_base[0 + sc_offset]);
    const float sc1 = d * float(sc_base[1 + sc_offset]);

    float dot0 = 0.0f;
    for (int l = 0; l < 16; ++l) {
        const int low = (ql_base[l + ql_offset] >> ql_shift) & 0x0F;
        const int high = ((qh_base[l] >> qh_shift) & 3) << 4;
        const int q_val = (low | high) - 32;
        dot0 += float(q_val) * xu[l];
    }

    float dot1 = 0.0f;
    for (int l = 16; l < 32; ++l) {
        const int low = (ql_base[l + ql_offset] >> ql_shift) & 0x0F;
        const int high = ((qh_base[l] >> qh_shift) & 3) << 4;
        const int q_val = (low | high) - 32;
        dot1 += float(q_val) * xu[l];
    }

    return sc0 * dot0 + sc1 * dot1;
}

template <typename T> struct vec4_ptr;
template <> struct vec4_ptr<device const float*> { typedef device const float4* type; };
template <> struct vec4_ptr<threadgroup const float*> { typedef threadgroup const float4* type; };

template <int F, typename T_X>
METAL_FUNC float deq_dot(device const uchar* b, int ui, T_X xu, threadgroup const uint* tab) {
    constexpr int UNIT = fmt<F>::UNIT;
    thread float q[UNIT];
    const float scale = deq<F>(b, ui, q, tab);
    float udot = 0.0f;
    for (int l = 0; l < UNIT; ++l) {
        udot += q[l] * xu[l];
    }
    return scale * udot;
}

template <int F, typename T_X>
METAL_FUNC float deq_dot_block(device const uchar* b, T_X x_blk, threadgroup const uint* tab = nullptr) {
    constexpr int UNIT = fmt<F>::UNIT;
    constexpr int QK = fmt<F>::QK;
    constexpr int UPB = QK / UNIT;
    float acc = 0.0f;
    for (int ui = 0; ui < UPB; ++ui) {
        T_X xu = x_blk + ui * UNIT;
        acc += deq_dot<F>(b, ui, xu, tab);
    }
    return acc;
}

template <typename T_X>
METAL_FUNC float deq_dot_block_q4_K(device const uchar* b, T_X x_blk) {
    device const block_q4_K* blk = (device const block_q4_K*)b;
    const float d = float(blk->dm[0]);
    const float dmin = float(blk->dm[1]);

    float d_sc[8];
    float m_sc[8];
    for (int ui = 0; ui < 8; ++ui) {
        uchar sc, m;
        get_scale_min_k4(ui, blk->scales, &sc, &m);
        d_sc[ui] = d * float(sc);
        m_sc[ui] = dmin * float(m);
    }

    device const uint* qs32 = (device const uint*)blk->qs;
    auto x4 = (typename vec4_ptr<T_X>::type)x_blk;

    float acc = 0.0f;

    for (int pair = 0; pair < 4; ++pair) {
        device const uint* q_pair = qs32 + pair * 8;
        auto x4_0 = x4 + (pair * 2) * 8;
        auto x4_1 = x4 + (pair * 2 + 1) * 8;

        float dot0 = 0.0f, xsum0 = 0.0f;
        float dot1 = 0.0f, xsum1 = 0.0f;

        for (int g = 0; g < 8; ++g) {
            const uint w = q_pair[g];
            const float4 v0 = float4(float(w & 0x0F),
                                     float((w >> 8) & 0x0F),
                                     float((w >> 16) & 0x0F),
                                     float((w >> 24) & 0x0F));
            const float4 v1 = float4(float((w >> 4) & 0x0F),
                                     float((w >> 12) & 0x0F),
                                     float((w >> 20) & 0x0F),
                                     float((w >> 28) & 0x0F));
            const float4 val_x0 = x4_0[g];
            const float4 val_x1 = x4_1[g];

            dot0 += dot(v0, val_x0);
            xsum0 += (val_x0.x + val_x0.y) + (val_x0.z + val_x0.w);

            dot1 += dot(v1, val_x1);
            xsum1 += (val_x1.x + val_x1.y) + (val_x1.z + val_x1.w);
        }

        acc += (d_sc[pair * 2] * dot0 - m_sc[pair * 2] * xsum0) +
               (d_sc[pair * 2 + 1] * dot1 - m_sc[pair * 2 + 1] * xsum1);
    }

    return acc;
}

template <>
METAL_FUNC float deq_dot_block<T_Q4_K, device const float*>(device const uchar* b, device const float* x_blk, threadgroup const uint*) {
    return deq_dot_block_q4_K(b, x_blk);
}
template <>
METAL_FUNC float deq_dot_block<T_Q4_K, threadgroup const float*>(device const uchar* b, threadgroup const float* x_blk, threadgroup const uint*) {
    return deq_dot_block_q4_K(b, x_blk);
}

template <typename T_X>
METAL_FUNC float deq_dot_block_q5_K(device const uchar* b, T_X x_blk) {
    device const block_q5_K* blk = (device const block_q5_K*)b;
    const float d = float(blk->dm[0]);
    const float dmin = float(blk->dm[1]);

    float d_sc[8];
    float m_sc[8];
    for (int ui = 0; ui < 8; ++ui) {
        uchar sc, m;
        get_scale_min_k4(ui, blk->scales, &sc, &m);
        d_sc[ui] = d * float(sc);
        m_sc[ui] = dmin * float(m);
    }

    device const uint* ql32 = (device const uint*)blk->qs;
    device const uint* qh32 = (device const uint*)blk->qh;
    auto x4 = (typename vec4_ptr<T_X>::type)x_blk;

    uint wh[8];
    for (int g = 0; g < 8; ++g) {
        wh[g] = qh32[g];
    }

    float acc = 0.0f;

    for (int pair = 0; pair < 4; ++pair) {
        device const uint* q_pair = ql32 + pair * 8;
        auto x4_0 = x4 + (pair * 2) * 8;
        auto x4_1 = x4 + (pair * 2 + 1) * 8;

        const uint u_bit0 = 1u << (pair * 2);
        const uint u_bit1 = 1u << (pair * 2 + 1);

        float dot0 = 0.0f, xsum0 = 0.0f;
        float dot1 = 0.0f, xsum1 = 0.0f;

        for (int g = 0; g < 8; ++g) {
            const uint wl = q_pair[g];
            const uint w_high = wh[g];

            const float4 v0 = float4(float((wl & 0x0F) + ((w_high & u_bit0) ? 16 : 0)),
                                     float(((wl >> 8) & 0x0F) + (((w_high >> 8) & u_bit0) ? 16 : 0)),
                                     float(((wl >> 16) & 0x0F) + (((w_high >> 16) & u_bit0) ? 16 : 0)),
                                     float(((wl >> 24) & 0x0F) + (((w_high >> 24) & u_bit0) ? 16 : 0)));

            const float4 v1 = float4(float(((wl >> 4) & 0x0F) + ((w_high & u_bit1) ? 16 : 0)),
                                     float(((wl >> 12) & 0x0F) + (((w_high >> 8) & u_bit1) ? 16 : 0)),
                                     float(((wl >> 20) & 0x0F) + (((w_high >> 16) & u_bit1) ? 16 : 0)),
                                     float(((wl >> 28) & 0x0F) + (((w_high >> 24) & u_bit1) ? 16 : 0)));

            const float4 val_x0 = x4_0[g];
            const float4 val_x1 = x4_1[g];

            dot0 += dot(v0, val_x0);
            xsum0 += (val_x0.x + val_x0.y) + (val_x0.z + val_x0.w);

            dot1 += dot(v1, val_x1);
            xsum1 += (val_x1.x + val_x1.y) + (val_x1.z + val_x1.w);
        }

        acc += (d_sc[pair * 2] * dot0 - m_sc[pair * 2] * xsum0) +
               (d_sc[pair * 2 + 1] * dot1 - m_sc[pair * 2 + 1] * xsum1);
    }

    return acc;
}

template <>
METAL_FUNC float deq_dot_block<T_Q5_K, device const float*>(device const uchar* b, device const float* x_blk, threadgroup const uint*) {
    return deq_dot_block_q5_K(b, x_blk);
}
template <>
METAL_FUNC float deq_dot_block<T_Q5_K, threadgroup const float*>(device const uchar* b, threadgroup const float* x_blk, threadgroup const uint*) {
    return deq_dot_block_q5_K(b, x_blk);
}

template <typename T_X>
METAL_FUNC float deq_dot_block_q6_K(device const uchar* b, T_X x_blk) {
    device const block_q6_K* blk = (device const block_q6_K*)b;
    const float d = float(blk->d);

    float acc = 0.0f;
    for (int h_part = 0; h_part < 2; ++h_part) {
        device const uchar* ql_base = blk->ql + h_part * 64;
        device const uchar* qh_base = blk->qh + h_part * 32;
        device const char* sc_base = (device const char*)blk->scales + h_part * 8;
        T_X x_h = x_blk + h_part * 128;

        for (int sub = 0; sub < 4; ++sub) {
            const int ql_offset = (sub & 1) ? 32 : 0;
            const int ql_shift = (sub >= 2) ? 4 : 0;
            const int qh_shift = 2 * sub;
            const int sc_offset = 2 * sub;

            const float sc0 = d * float(sc_base[0 + sc_offset]);
            const float sc1 = d * float(sc_base[1 + sc_offset]);
            T_X xu = x_h + sub * 32;

            float dot0 = 0.0f;
            for (int l = 0; l < 16; ++l) {
                const int low = (ql_base[l + ql_offset] >> ql_shift) & 0x0F;
                const int high = ((qh_base[l] >> qh_shift) & 3) << 4;
                const int q_val = (low | high) - 32;
                dot0 += float(q_val) * xu[l];
            }

            float dot1 = 0.0f;
            for (int l = 16; l < 32; ++l) {
                const int low = (ql_base[l + ql_offset] >> ql_shift) & 0x0F;
                const int high = ((qh_base[l] >> qh_shift) & 3) << 4;
                const int q_val = (low | high) - 32;
                dot1 += float(q_val) * xu[l];
            }

            acc += sc0 * dot0 + sc1 * dot1;
        }
    }
    return acc;
}

template <>
METAL_FUNC float deq_dot_block<T_Q6_K, device const float*>(device const uchar* b, device const float* x_blk, threadgroup const uint*) {
    return deq_dot_block_q6_K(b, x_blk);
}
template <>
METAL_FUNC float deq_dot_block<T_Q6_K, threadgroup const float*>(device const uchar* b, threadgroup const float* x_blk, threadgroup const uint*) {
    return deq_dot_block_q6_K(b, x_blk);
}



