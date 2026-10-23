// metal/tests/moe_sim.cpp - host-side parity simulation of the chunk path's routed-expert MoE
// (metal/src/engine/q36_metal.cpp's MoE block), pure C++, no Metal: the kernels are transcribed
// loop for loop from metal/kernels/q36.metal (k_route_chunk, k_tables, k_gather_t, k_expc, k_gemv_c,
// k_gemv_lpr8, k_moe_sum_chunk) and metal/kernels/quant.metal (deq_dot / deq_dot_block for Q8_0 /
// Q4_K / Q5_K / Q6_K); the donor path (q36_dense.metal's act_f16 + dq_f16 + gemm_f16) is transcribed
// at its host-replicable semantics (fp16 operands, fp32 K-ascending accumulation).  One call = one
// dispatch; grid-coordinate loops stand in for threadgroups, per-lane arrays for simdgroup state.
//
// Build + run (Linux sandbox or macOS; on macOS the tests glob builds it with the other parity
// programs):
//   g++ -O2 -std=c++20 -I metal/src -I third_party/ggml metal/src/ref_quant.cpp metal/tests/moe_sim.cpp -o moe_sim && ./moe_sim
//
// What this proves: the routed-expert chain's structure - routing, entry tables, gather, the
// plain/lpr8/donor expert-product forms and the combine - is internally consistent (bit-identical
// wherever two production forms share the accumulation order) and matches an independent
// double-precision reference within the envelopes printed per stage.  What it cannot prove: the
// MSL compile, barrier placement, and Metal's exp/fma contraction (both sides of every comparison
// here run the SAME host primitives, so those cancel; T10's Metal stages re-measure the envelopes).
//
// T10's candidate plug-in (no new infrastructure): add the batched-GEMM transcription as one
// function with the ExpertArmFn signature, register it in k_arms, and every expert-product stage
// picks it up - the arms registry is the only edit.
//
// The rows-per-expert histogram at the end is the DRY-RUN mechanism evidence for T10's
// hybrid-dispatch threshold; the authoritative on-Metal histogram comes from T13's counter lineage
// (STRATA_Q36M_DISPATCH_LOG, landed with T9-r1).  Synthetic router logits only.
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// ---- geometry.  small = the parity test's (q36_chunk_parity.cpp); real = the model's
// ---- (docs/QWEN36_METAL.md section 1: 256 routed + 1 shared, top-8, n_embd 2048, FFN width 640).
// ---- 256-block formats (Q4_K/Q5_K/Q6_K) need n_in % 256 == 0, so the real-geometry down stage
// ---- runs Q8_0 at FF = 640 and brackets Q4_K at 512; the gate/up shape (n_in 2048) takes all.
constexpr int SMALL_N = 256, SMALL_FF = 96, SMALL_FS = 64, SMALL_NEXP = 16, SMALL_K = 4;
constexpr int REAL_N = 2048, REAL_FF = 640, REAL_FS = 640, REAL_NEXP = 256, REAL_K = 8;

static int g_fails = 0, g_checks = 0;

static inline bool feq(float a, float b) {
    return a == b || (std::isnan(a) && std::isnan(b));   // both-NaN counts as equal (the harness's rule)
}
static inline float h16(float v) { return ref::f16_to_f32(ref::f32_to_f16(v)); }
static inline float silu(float v) { return v / (1.0f + std::exp(-v)); }
static inline float sigm(float v) { return 1.0f / (1.0f + std::exp(-v)); }

// the simdgroup sum as a fixed pairwise fold over 32 lanes (the pairing the xor butterfly produces;
// shared by every transcription here so the tree cancels in arm-vs-arm checks)
static float simd_sum32(const float* v32) {
    float t[32];
    memcpy(t, v32, sizeof(t));
    for (int span = 16; span > 0; span /= 2)
        for (int i = 0; i < span; ++i) t[i] = t[i] + t[i + span];
    return t[0];
}
// k_gemv_lpr8's 8-lane butterfly (simd_shuffle_xor 4/2/1)
static float fold8(const float* v8) {
    float t[8];
    memcpy(t, v8, sizeof(t));
    for (int span = 4; span > 0; span /= 2)
        for (int i = 0; i < span; ++i) t[i] = t[i] + t[i + span];
    return t[0];
}

// ---- ggml block layouts (third_party/ggml/ggml-common.h; quant.metal reads the same offsets)
struct BQ4K { uint16_t dm[2]; uint8_t scales[12]; uint8_t qs[128]; };                 // 144 B
struct BQ5K { uint16_t dm[2]; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; }; // 176 B
struct BQ6K { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; };      // 210 B
static_assert(sizeof(BQ4K) == 144 && sizeof(BQ5K) == 176 && sizeof(BQ6K) == 210, "block layout");

static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = (uint8_t)(q[j] & 63);
        *m = (uint8_t)(q[j + 4] & 63);
    } else {
        *d = (uint8_t)((q[j + 4] & 0x0F) | ((q[j - 4] & 0xC0) >> 2));
        *m = (uint8_t)(((q[j + 4] >> 4) & 0x0F) | ((q[j] & 0xC0) >> 2));
    }
}

struct Fmt { int qk, bb, unit, upb; };
static Fmt fmt_of(int t) {
    switch (t) {
        case ref::Q8_0: return {32, 34, 32, 1};
        case ref::Q4_K: return {256, 144, 32, 8};
        case ref::Q5_K: return {256, 176, 32, 8};
        case ref::Q6_K: return {256, 210, 32, 8};
        case ref::IQ2_XS: return {256, 74, 16, 16};   // layout only: the donor arm + refs read its
                                                      // values through ref::dequant; no dot form here
    }
    std::fprintf(stderr, "moe_sim: type %s not transcribed (the sim covers Q8_0/Q4_K/Q5_K/Q6_K;"
                         " IQ2_XS rides the donor arm + refs; IQ3_XXS is the M2 moe2048 stages' format)\n",
                 ref::type_name(t));
    std::exit(2);
}

// ---- dequant-dot transcriptions (quant.metal's deq_dot<F> specializations: the factored forms the
// ---- plain arms run).  dot(float4) = left-assoc serial; the host keeps the same order.
static float dot_unit_q8_0(const uint8_t* b, const float* xu) {
    const float d = ref::f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
    float udot = 0.0f;
    for (int j = 0; j < 16; ++j) {
        udot += (float)(int8_t) b[2 + 2 * j] * xu[2 * j];
        udot += (float)(int8_t) b[2 + 2 * j + 1] * xu[2 * j + 1];
    }
    return d * udot;
}
static float dot_unit_q4k(const uint8_t* b, int ui, const float* xu) {
    const BQ4K* blk = (const BQ4K*)b;
    const float d = ref::f16_to_f32(blk->dm[0]), dmin = ref::f16_to_f32(blk->dm[1]);
    uint8_t sc, m;
    get_scale_min_k4(ui, blk->scales, &sc, &m);
    const float d_sc = d * (float)sc, m_sc = dmin * (float)m;
    const uint32_t* q32 = (const uint32_t*)(blk->qs + (size_t)(ui / 2) * 32);
    const int shift = (ui & 1) ? 4 : 0;
    float adot = 0.0f, axsum = 0.0f;
    for (int g = 0; g < 8; ++g) {
        const uint32_t w = q32[g];
        const float* x = xu + 4 * g;
        adot += (float)((w >> shift) & 0xF) * x[0] + (float)(((w >> 8) >> shift) & 0xF) * x[1] +
                (float)(((w >> 16) >> shift) & 0xF) * x[2] + (float)(((w >> 24) >> shift) & 0xF) * x[3];
        axsum += (x[0] + x[1]) + (x[2] + x[3]);
    }
    return d_sc * adot - m_sc * axsum;
}
static float dot_unit_q5k(const uint8_t* b, int ui, const float* xu) {
    const BQ5K* blk = (const BQ5K*)b;
    const float d = ref::f16_to_f32(blk->dm[0]), dmin = ref::f16_to_f32(blk->dm[1]);
    uint8_t sc, m;
    get_scale_min_k4(ui, blk->scales, &sc, &m);
    const float d_sc = d * (float)sc, m_sc = dmin * (float)m;
    const uint32_t* ql32 = (const uint32_t*)(blk->qs + (size_t)(ui / 2) * 32);
    const uint32_t* qh32 = (const uint32_t*)blk->qh;
    const uint32_t u_bit = 1u << ui;
    const int shift = (ui & 1) ? 4 : 0;
    float adot = 0.0f, axsum = 0.0f;
    for (int g = 0; g < 8; ++g) {
        const uint32_t wl = ql32[g], wh = qh32[g];
        const float* x = xu + 4 * g;
        adot += (float)(((wl >> shift) & 0xF) + ((wh & u_bit) ? 16 : 0)) * x[0] +
                (float)((((wl >> 8) >> shift) & 0xF) + (((wh >> 8) & u_bit) ? 16 : 0)) * x[1] +
                (float)((((wl >> 16) >> shift) & 0xF) + (((wh >> 16) & u_bit) ? 16 : 0)) * x[2] +
                (float)((((wl >> 24) >> shift) & 0xF) + (((wh >> 24) & u_bit) ? 16 : 0)) * x[3];
        axsum += (x[0] + x[1]) + (x[2] + x[3]);
    }
    return d_sc * adot - m_sc * axsum;
}
static float dot_unit_q6k(const uint8_t* b, int ui, const float* xu) {
    const BQ6K* blk = (const BQ6K*)b;
    const float d = ref::f16_to_f32(blk->d);
    const int h_part = ui / 4, sub = ui % 4;
    const uint8_t* ql_base = blk->ql + h_part * 64;
    const uint8_t* qh_base = blk->qh + h_part * 32;
    const int8_t* sc_base = (const int8_t*)blk->scales + h_part * 8;
    const int ql_offset = (sub & 1) ? 32 : 0, ql_shift = (sub >= 2) ? 4 : 0;
    const int qh_shift = 2 * sub, sc_offset = 2 * sub;
    const float sc0 = d * (float)sc_base[0 + sc_offset], sc1 = d * (float)sc_base[1 + sc_offset];
    float dot0 = 0.0f;
    for (int l = 0; l < 16; ++l) {
        const int low = (ql_base[l + ql_offset] >> ql_shift) & 0x0F;
        const int high = ((qh_base[l] >> qh_shift) & 3) << 4;
        dot0 += (float)((low | high) - 32) * xu[l];
    }
    float dot1 = 0.0f;
    for (int l = 16; l < 32; ++l) {
        const int low = (ql_base[l + ql_offset] >> ql_shift) & 0x0F;
        const int high = ((qh_base[l] >> qh_shift) & 3) << 4;
        dot1 += (float)((low | high) - 32) * xu[l];
    }
    return sc0 * dot0 + sc1 * dot1;
}

// deq_dot_block<F>'s whole-256-block factored forms - the lpr8 arm's per-lane unit (one lane dots
// one 256-elem block against one 256-float x span)
static float dot_block_q4k(const uint8_t* b, const float* x256) {
    const BQ4K* blk = (const BQ4K*)b;
    const float d = ref::f16_to_f32(blk->dm[0]), dmin = ref::f16_to_f32(blk->dm[1]);
    float d_sc[8], m_sc[8];
    for (int ui = 0; ui < 8; ++ui) {
        uint8_t sc, m;
        get_scale_min_k4(ui, blk->scales, &sc, &m);
        d_sc[ui] = d * (float)sc;
        m_sc[ui] = dmin * (float)m;
    }
    const uint32_t* qs32 = (const uint32_t*)blk->qs;
    float acc = 0.0f;
    for (int pair = 0; pair < 4; ++pair) {
        const uint32_t* qp = qs32 + pair * 8;
        const float* x0 = x256 + (pair * 2) * 32;
        const float* x1 = x256 + (pair * 2 + 1) * 32;
        float dot0 = 0, xsum0 = 0, dot1 = 0, xsum1 = 0;
        for (int g = 0; g < 8; ++g) {
            const uint32_t w = qp[g];
            const float* xa = x0 + 4 * g;
            const float* xb = x1 + 4 * g;
            dot0 += (float)(w & 0xF) * xa[0] + (float)((w >> 8) & 0xF) * xa[1] +
                    (float)((w >> 16) & 0xF) * xa[2] + (float)((w >> 24) & 0xF) * xa[3];
            xsum0 += (xa[0] + xa[1]) + (xa[2] + xa[3]);
            dot1 += (float)((w >> 4) & 0xF) * xb[0] + (float)((w >> 12) & 0xF) * xb[1] +
                    (float)((w >> 20) & 0xF) * xb[2] + (float)((w >> 28) & 0xF) * xb[3];
            xsum1 += (xb[0] + xb[1]) + (xb[2] + xb[3]);
        }
        acc += (d_sc[pair * 2] * dot0 - m_sc[pair * 2] * xsum0) +
               (d_sc[pair * 2 + 1] * dot1 - m_sc[pair * 2 + 1] * xsum1);
    }
    return acc;
}
static float dot_block_q5k(const uint8_t* b, const float* x256) {
    const BQ5K* blk = (const BQ5K*)b;
    const float d = ref::f16_to_f32(blk->dm[0]), dmin = ref::f16_to_f32(blk->dm[1]);
    float d_sc[8], m_sc[8];
    for (int ui = 0; ui < 8; ++ui) {
        uint8_t sc, m;
        get_scale_min_k4(ui, blk->scales, &sc, &m);
        d_sc[ui] = d * (float)sc;
        m_sc[ui] = dmin * (float)m;
    }
    const uint32_t* ql32 = (const uint32_t*)blk->qs;
    const uint32_t* qh32 = (const uint32_t*)blk->qh;
    uint32_t wh[8];
    for (int g = 0; g < 8; ++g) wh[g] = qh32[g];
    float acc = 0.0f;
    for (int pair = 0; pair < 4; ++pair) {
        const uint32_t* qp = ql32 + pair * 8;
        const float* x0 = x256 + (pair * 2) * 32;
        const float* x1 = x256 + (pair * 2 + 1) * 32;
        const uint32_t u_bit0 = 1u << (pair * 2), u_bit1 = 1u << (pair * 2 + 1);
        float dot0 = 0, xsum0 = 0, dot1 = 0, xsum1 = 0;
        for (int g = 0; g < 8; ++g) {
            const uint32_t wl = qp[g], w_high = wh[g];
            const float* xa = x0 + 4 * g;
            const float* xb = x1 + 4 * g;
            dot0 += (float)((wl & 0xF) + ((w_high & u_bit0) ? 16 : 0)) * xa[0] +
                    (float)(((wl >> 8) & 0xF) + (((w_high >> 8) & u_bit0) ? 16 : 0)) * xa[1] +
                    (float)(((wl >> 16) & 0xF) + (((w_high >> 16) & u_bit0) ? 16 : 0)) * xa[2] +
                    (float)(((wl >> 24) & 0xF) + (((w_high >> 24) & u_bit0) ? 16 : 0)) * xa[3];
            xsum0 += (xa[0] + xa[1]) + (xa[2] + xa[3]);
            dot1 += (float)(((wl >> 4) & 0xF) + ((w_high & u_bit1) ? 16 : 0)) * xb[0] +
                    (float)(((wl >> 12) & 0xF) + (((w_high >> 8) & u_bit1) ? 16 : 0)) * xb[1] +
                    (float)(((wl >> 20) & 0xF) + (((w_high >> 16) & u_bit1) ? 16 : 0)) * xb[2] +
                    (float)(((wl >> 28) & 0xF) + (((w_high >> 24) & u_bit1) ? 16 : 0)) * xb[3];
            xsum1 += (xb[0] + xb[1]) + (xb[2] + xb[3]);
        }
        acc += (d_sc[pair * 2] * dot0 - m_sc[pair * 2] * xsum0) +
               (d_sc[pair * 2 + 1] * dot1 - m_sc[pair * 2 + 1] * xsum1);
    }
    return acc;
}
static float dot_block_q6k(const uint8_t* b, const float* x256) {
    const BQ6K* blk = (const BQ6K*)b;
    const float d = ref::f16_to_f32(blk->d);
    float acc = 0.0f;
    for (int h_part = 0; h_part < 2; ++h_part) {
        const uint8_t* ql_base = blk->ql + h_part * 64;
        const uint8_t* qh_base = blk->qh + h_part * 32;
        const int8_t* sc_base = (const int8_t*)blk->scales + h_part * 8;
        const float* x_h = x256 + h_part * 128;
        for (int sub = 0; sub < 4; ++sub) {
            const int ql_offset = (sub & 1) ? 32 : 0, ql_shift = (sub >= 2) ? 4 : 0;
            const int qh_shift = 2 * sub, sc_offset = 2 * sub;
            const float sc0 = d * (float)sc_base[0 + sc_offset], sc1 = d * (float)sc_base[1 + sc_offset];
            const float* xu = x_h + sub * 32;
            float dot0 = 0.0f;
            for (int l = 0; l < 16; ++l) {
                const int low = (ql_base[l + ql_offset] >> ql_shift) & 0x0F;
                const int high = ((qh_base[l] >> qh_shift) & 3) << 4;
                dot0 += (float)((low | high) - 32) * xu[l];
            }
            float dot1 = 0.0f;
            for (int l = 16; l < 32; ++l) {
                const int low = (ql_base[l + ql_offset] >> ql_shift) & 0x0F;
                const int high = ((qh_base[l] >> qh_shift) & 3) << 4;
                dot1 += (float)((low | high) - 32) * xu[l];
            }
            acc += sc0 * dot0 + sc1 * dot1;
        }
    }
    return acc;
}

static float dot_unit(int type, const uint8_t* b, int ui, const float* xu) {
    switch (type) {
        case ref::Q8_0: return dot_unit_q8_0(b, xu);
        case ref::Q4_K: return dot_unit_q4k(b, ui, xu);
        case ref::Q5_K: return dot_unit_q5k(b, ui, xu);
        case ref::Q6_K: return dot_unit_q6k(b, ui, xu);
    }
    std::fprintf(stderr, "moe_sim: no plain dot form for %s (IQ2_XS/IQ3_XXS are the M2 stages' formats)\n",
                 ref::type_name(type));
    std::exit(2);
}
static float dot_block256(int type, const uint8_t* b, const float* x256) {
    switch (type) {
        case ref::Q4_K: return dot_block_q4k(b, x256);
        case ref::Q5_K: return dot_block_q5k(b, x256);
        case ref::Q6_K: return dot_block_q6k(b, x256);
    }
    std::fprintf(stderr, "moe_sim: no whole-block dot form for %s\n", ref::type_name(type));
    std::exit(2);
}

// ---- the expert-product arms.  Every arm reads the same weights and the same per-entry x values;
// ---- they differ only in accumulation structure, which is what the stage envelopes measure.
struct ArmCtx {
    int type;
    const uint8_t* W;
    size_t row_bytes, expert_bytes;   // expert_bytes = 0 for the dense reuse (the router/sgate form)
    int n_in, n_out;
    const float* x;
    long x_stride;
    const int* ids;      // entry -> expert, (t,k)-ordered (the lpr8 arm)
    const int* srt_in;   // placed row -> x-row index (the expc/donor arms)
    const int* off;      // expert -> first placed row
    const int* cnt;      // expert -> entry count
    int n_expert, n_ent;
};

using ExpertArmFn = void (*)(const ArmCtx&, float* y);

// k_expc / k_gemv_c / k_gemv: the plain order - units strided across the 32 lanes, per-lane accs,
// the simd_sum fold.  y rows are placed-ordered (row = the entry's placed position); with the
// dense-form ctx (one expert, identity placement) this is exactly k_gemv_c's per-pair arithmetic.
static void arm_expc(const ArmCtx& c, float* y) {
    const Fmt f = fmt_of(c.type);
    const int total_units = c.n_in / f.unit;
    for (int e = 0; e < c.n_expert; ++e) {
        for (int o = 0; o < c.n_out; ++o) {
            const uint8_t* row = c.W + (size_t)e * c.expert_bytes + (size_t)o * c.row_bytes;
            for (int r = 0; r < c.cnt[e]; ++r) {
                const int entry = c.off[e] + r;
                const float* xs = c.x + (size_t)c.srt_in[entry] * c.x_stride;
                float lacc[32] = {0};
                for (int u = 0; u < total_units; ++u)
                    lacc[u % 32] += dot_unit(c.type, row + (size_t)(u / f.upb) * f.bb, u % f.upb,
                                             xs + (size_t)u * f.unit);
                y[(size_t)entry * c.n_out + o] = simd_sum32(lacc);
            }
        }
    }
}

// k_gemv_lpr8: the decode order - 8 lanes x one whole 256-block each, the 8-lane butterfly.
// y rows are entry-ordered (row = the (t,k) entry index).  n_in must be 2048 (the kernel's
// hardwired sh_x[2048]; production only dispatches it there).
static void arm_lpr8(const ArmCtx& c, float* y) {
    const Fmt f = fmt_of(c.type);
    if (c.n_in != 2048) {
        std::fprintf(stderr, "moe_sim: the lpr8 arm needs n_in == 2048, got %d\n", c.n_in);
        std::exit(2);
    }
    for (int sel = 0; sel < c.n_ent; ++sel) {
        const int e = c.ids[sel];
        const float* xs = c.x + (size_t)sel * c.x_stride;
        for (int o = 0; o < c.n_out; ++o) {
            const uint8_t* row = c.W + (size_t)e * c.expert_bytes + (size_t)o * c.row_bytes;
            float l8[8];
            for (int li = 0; li < 8; ++li) l8[li] = dot_block256(c.type, row + (size_t)li * f.bb, xs + li * 256);
            y[(size_t)sel * c.n_out + o] = fold8(l8);
        }
    }
}

// the dequant-once donor (q36_dense.metal's semantics at the expert shapes): per expert with rows,
// the matrix dequantized to fp16 once (dq_f16 = ref_quant's values rounded to fp16), the batch's x
// rows staged to half (act_f16), then the M x n_out x n_in product in fp32, K ascending - the MPP
// tile order's host-replicable form.  On Metal a T10 candidate's own transcription is held to
// dense_mm_parity's 1e-5-vs-fp16-reference bound; here the arm IS that fp16-operand reference.
// y rows are placed-ordered.
static void arm_donor(const ArmCtx& c, float* y) {
    std::vector<float> wf((size_t)c.n_out * c.n_in), xh((size_t)c.n_in);
    for (int e = 0; e < c.n_expert; ++e) {
        if (!c.cnt[e]) continue;
        ref::dequant(c.type, c.W + (size_t)e * c.expert_bytes, wf.data(), (size_t)c.n_out * c.n_in);
        for (int r = 0; r < c.cnt[e]; ++r) {
            const int rowp = c.off[e] + r;
            const float* src = c.x + (size_t)c.srt_in[rowp] * c.x_stride;
            for (int k = 0; k < c.n_in; ++k) xh[k] = h16(src[k]);
            for (int o = 0; o < c.n_out; ++o) {
                float acc = 0.0f;
                const float* wr = wf.data() + (size_t)o * c.n_in;
                for (int k = 0; k < c.n_in; ++k) acc += xh[k] * h16(wr[k]);
                y[(size_t)rowp * c.n_out + o] = acc;
            }
        }
    }
}

// ---- chunk dense gemv: k_gemv_c (q36.metal:893).  The tpb token-blocking and the grid split are
// ---- dispatch structure only - no shared state between (token, output) pairs - so the sim computes
// ---- each pair in the kernel's per-pair order (the same plain order arm_expc runs with the
// ---- dense-form ctx; this named wrapper keeps the dense call sites reading as themselves).
static void gemv_c(const ArmCtx& c, float* y, int nt) {
    arm_expc(c, y);
    (void) nt;
}

// ---- THE T10 CANDIDATE SLOT.  Register a batched-GEMM transcription here (one line) and every
// ---- expert-product stage below re-runs against it; the stage envelope tables gain a column.
static const struct { const char* name; ExpertArmFn fn; int entry_ordered; } k_arms[] = {
    {"expc", arm_expc, 0},
    {"lpr8", arm_lpr8, 1},
    {"donor", arm_donor, 0},
    // {"t10_batched_gemm", arm_t10, 0},   <- T10 adds its arm here
};

// ---- routing: k_route_chunk (q36.metal:1015) - one thread per token, serial softmax, top-K with
// ---- the stable tie-break (value desc, index asc), serial weight renorm
static void route_chunk(const float* router, int* ids, float* wts, int n_expert, int K, int T) {
    float p[256];
    for (int t = 0; t < T; ++t) {
        const float* r = router + (size_t)t * n_expert;
        float mx = r[0];
        for (int i = 1; i < n_expert; ++i)
            if (r[i] > mx) mx = r[i];
        float sum = 0.0f;
        for (int i = 0; i < n_expert; ++i) {
            const float val = std::exp(r[i] - mx);
            p[i] = val;
            sum += val;
        }
        const float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
        for (int i = 0; i < n_expert; ++i) p[i] *= inv_sum;
        int top_idx[16];
        float top_val[16];
        for (int k = 0; k < K; ++k) {
            top_val[k] = -1.0f;
            top_idx[k] = -1;
        }
        for (int i = 0; i < n_expert; ++i) {
            const float v = p[i];
            if (v > top_val[K - 1] || (v == top_val[K - 1] && (top_idx[K - 1] < 0 || i < top_idx[K - 1]))) {
                int pos_ = K - 1;
                while (pos_ > 0 && (v > top_val[pos_ - 1] || (v == top_val[pos_ - 1] && i < top_idx[pos_ - 1]))) {
                    top_val[pos_] = top_val[pos_ - 1];
                    top_idx[pos_] = top_idx[pos_ - 1];
                    --pos_;
                }
                top_val[pos_] = v;
                top_idx[pos_] = i;
            }
        }
        float wsum = 0.0f;
        for (int k = 0; k < K; ++k) wsum += top_val[k];
        const float inv_wsum = 1.0f / (wsum > 0.0f ? wsum : 1.0f);
        for (int k = 0; k < K; ++k) {
            ids[(size_t)t * K + k] = top_idx[k];
            wts[(size_t)t * K + k] = top_val[k] * inv_wsum;
        }
    }
}

// the routing reference: the same serial softmax (the production op order), then the top-K picked
// by a full stable sort on (value desc, index asc) - a different algorithm, the same result
static void route_ref(const float* router, int* ids, float* wts, int n_expert, int K, int T) {
    float p[256];
    for (int t = 0; t < T; ++t) {
        const float* r = router + (size_t)t * n_expert;
        float mx = r[0];
        for (int i = 1; i < n_expert; ++i)
            if (r[i] > mx) mx = r[i];
        float sum = 0.0f;
        for (int i = 0; i < n_expert; ++i) {
            const float val = std::exp(r[i] - mx);
            p[i] = val;
            sum += val;
        }
        const float inv_sum = 1.0f / (sum > 0.0f ? sum : 1.0f);
        for (int i = 0; i < n_expert; ++i) p[i] *= inv_sum;
        int order[256];
        for (int i = 0; i < n_expert; ++i) order[i] = i;
        std::stable_sort(order, order + n_expert, [&](int a, int b) {
            if (p[a] != p[b]) return p[a] > p[b];
            return a < b;
        });
        float wsum = 0.0f;
        for (int k = 0; k < K; ++k) wsum += p[order[k]];
        const float inv_wsum = 1.0f / (wsum > 0.0f ? wsum : 1.0f);
        for (int k = 0; k < K; ++k) {
            ids[(size_t)t * K + k] = order[k];
            wts[(size_t)t * K + k] = p[order[k]] * inv_wsum;
        }
    }
}

// ---- entry tables: k_tables (q36.metal:1145) - one threadgroup: zero, count, exclusive offsets,
// ---- placement.  The rank order inside an expert is atomic-driven on Metal; the sim replays one
// ---- legal serialization (tid-strided, the kernel's loop shape); order 1 is the entry-major
// ---- alternative (the naive ranks) used for the placement-invariance check.
struct Tables {
    std::vector<int> cnt_off;   // [0..E) counts, [E..2E) offsets (the kernel's packed layout)
    std::vector<int> srt_gu;    // placed row -> token
    std::vector<int> srt_d;     // placed row -> entry
    std::vector<int> pos;       // entry -> placed row
};
static Tables tables_body(const int* ids, int n_expert, int K, int n_ent, int order) {
    Tables t;
    t.cnt_off.assign(2 * (size_t)n_expert, 0);
    t.srt_gu.assign((size_t)n_ent, 0);
    t.srt_d.assign((size_t)n_ent, 0);
    t.pos.assign((size_t)n_ent, 0);
    for (uint32_t tid = 0; tid < 256; ++tid)               // the count pass, tid-strided as the kernel walks
        for (uint32_t i = tid; i < (uint32_t)n_ent; i += 256) ++t.cnt_off[ids[i]];
    int acc = 0;
    for (int e = 0; e < n_expert; ++e) {
        t.cnt_off[n_expert + e] = acc;
        acc += t.cnt_off[e];
    }
    std::vector<int> run((size_t)n_expert, 0);
    if (order == 0) {                                      // the kernel's placement pass: tid-strided
        for (uint32_t tid = 0; tid < 256; ++tid)
            for (uint32_t i = tid; i < (uint32_t)n_ent; i += 256) {
                const int e = ids[i];
                const int row = t.cnt_off[n_expert + e] + run[e]++;
                t.srt_gu[row] = (int)(i / (uint32_t)K);
                t.srt_d[row] = (int)i;
                t.pos[i] = row;
            }
    } else {                                               // entry-major ranks: the naive alternative
        for (int i = 0; i < n_ent; ++i) {
            const int e = ids[i];
            const int row = t.cnt_off[n_expert + e] + run[e]++;
            t.srt_gu[row] = i / K;
            t.srt_d[row] = i;
            t.pos[i] = row;
        }
    }
    return t;
}
static Tables tables_naive(const int* ids, int n_expert, int K, int n_ent) {
    Tables t;
    t.cnt_off.assign(2 * (size_t)n_expert, 0);
    for (int i = 0; i < n_ent; ++i) ++t.cnt_off[ids[i]];
    int acc = 0;
    for (int e = 0; e < n_expert; ++e) {
        t.cnt_off[n_expert + e] = acc;
        acc += t.cnt_off[e];
    }
    t.srt_gu.assign((size_t)n_ent, 0);
    t.srt_d.assign((size_t)n_ent, 0);
    t.pos.assign((size_t)n_ent, 0);
    std::vector<int> run((size_t)n_expert, 0);
    for (int i = 0; i < n_ent; ++i) {                      // entry-major ranks
        const int e = ids[i];
        const int row = t.cnt_off[n_expert + e] + run[e]++;
        t.srt_gu[row] = i / K;
        t.srt_d[row] = i;
        t.pos[i] = row;
    }
    return t;
}

// k_tables' placement invariant (the harness's check_placement)
static void check_placement(const char* stage, const int* ids, const Tables& t, int n_expert, int K, int n_ent) {
    std::vector<char> seen((size_t)n_ent, 0);
    for (int i = 0; i < n_ent; ++i) {
        const int row = t.pos[i], e = ids[i];
        const bool bad = row < 0 || row >= n_ent || seen[row] || t.srt_d[row] != i ||
                         t.srt_gu[row] != i / K || row < t.cnt_off[n_expert + e] ||
                         row >= t.cnt_off[n_expert + e] + t.cnt_off[e];
        if (bad) {
            std::printf("  %-14s FAIL at entry %d (row %d, expert %d)\n", stage, i, row, e);
            ++g_fails;
            return;
        }
        seen[row] = 1;
    }
    std::printf("  %-14s OK\n", stage);
    ++g_checks;
}

// ---- gather: k_gather_t (q36.metal:1194) - entry i = t*K + k is a pure copy of its token's row
static void gather_t(const float* x, float* y, int width, int K, int n_ent) {
    for (int i = 0; i < n_ent; ++i)
        for (int o = 0; o < width; ++o)
            y[(size_t)i * width + o] = x[(size_t)(i / K) * width + o];
}

// ---- swiglu (k_swiglu) and the combine (k_moe_sum_chunk, q36.metal:1120)
static void swiglu(const float* g, const float* u, float* h, size_t n) {
    for (size_t i = 0; i < n; ++i) h[i] = silu(g[i]) * u[i];
}
static void moe_sum_chunk(float* x, const float* y, const float* w, const float* ys, const float* gs,
                          const int* pos, int k, int n_out, int T) {
    for (int t = 0; t < T; ++t)
        for (int i = 0; i < n_out; ++i) {
            float acc = 0.0f;
            for (int s = 0; s < k; ++s)
                acc += w[(size_t)t * k + s] * y[(size_t)pos[(size_t)t * k + s] * n_out + i];
            x[(size_t)t * n_out + i] += acc + sigm(gs[t]) * ys[(size_t)t * n_out + i];
        }
}

// ---- checks (the harness's conventions)
static void check(const char* stage, const float* a, const float* b, size_t n) {
    size_t first = n, count = 0, nans = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(a[i]) || std::isnan(b[i])) {
            ++nans;
            continue;
        }
        if (!feq(a[i], b[i])) {
            if (first == n) first = i;
            ++count;
        }
    }
    if (count == 0 && nans == 0) {
        std::printf("  %-14s OK\n", stage);
        ++g_checks;
        return;
    }
    std::printf("  %-14s FAIL: %zu real diffs (first at %zu: %.9g vs %.9g), %zu NaN cells\n", stage, count,
                first, first < n ? a[first] : 0.f, first < n ? b[first] : 0.f, nans);
    ++g_fails;
}
static void check_i32(const char* stage, const int* a, const int* b, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (a[i] != b[i]) {
            std::printf("  %-14s FAIL at %zu (%d vs %d)\n", stage, i, a[i], b[i]);
            ++g_fails;
            return;
        }
    std::printf("  %-14s OK\n", stage);
    ++g_checks;
}
// the harness's check_rel metric (RMS-relative, double): sqrt(sum (a-b)^2 / sum b^2)
static double rel_rms(const float* a, const float* b, size_t n) {
    double se = 0, sr = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = (double)a[i] - (double)b[i];
        se += e * e;
        sr += (double)b[i] * (double)b[i];
    }
    return std::sqrt(se / std::max(sr, 1e-30));
}
static void check_rel(const char* stage, const float* a, const float* b, size_t n, double tol) {
    const double r = rel_rms(a, b, n);
    if (r < tol) {
        std::printf("  %-14s OK (rel %.2e)\n", stage, r);
        ++g_checks;
        return;
    }
    std::printf("  %-14s FAIL: rel %.3e >= %.0e\n", stage, r, tol);
    ++g_fails;
}

// synthetic weights: valid random blocks, odd bytes sanitized (finite fp16 scales, the parity
// test's convention).  n_in must be block-aligned (256-multiples for the K-quants, 32 for Q8_0) so
// ref::random_blocks' row stride matches the kernels'.
static void make_weights(int type, uint8_t* W, size_t total_bytes, uint64_t seed) {
    const Fmt f = fmt_of(type);
    for (size_t o = 0; o < total_bytes / (size_t)f.bb; ++o)
        ref::random_blocks(type, W + o * (size_t)f.bb, 1, f.qk, seed + o * 7919u, 0.02f);
    for (size_t i = 1; i < total_bytes; i += 2) W[i] &= 0x83;
}

// the dense-reuse context: one matrix, identity placement, n entries = the rows of x (the form
// push_c_gemv's k_gemv_c branch and the lpr8 dense form (zero_ids) take)
struct DenseCtx {
    std::vector<int> ids, ident, off, cnt;
    explicit DenseCtx(int n) : ids((size_t)n, 0), ident((size_t)n), off(1, 0), cnt(1, n) {
        for (int i = 0; i < n; ++i) ident[(size_t)i] = i;
    }
};

// ---- the chain (both configs).  Builds its own weights + logits, runs route -> tables/gather ->
// ---- gate/up -> swiglu -> down -> shared expert -> combine exactly as the engine's MoE block
// ---- dispatches them (push_c_gemv's branch rule, the gu_lpr8/dn_lpr8 conditions, the combine's
// ---- slot-order sum), and keeps the envelope points (eh/ey/x) plus the inputs the independent
// ---- reference recomputes from.
struct ChainGeom {
    int T, N, FF, FS, NEXP, K;
    int router_type, gu_type, dn_type, shg_type, shd_type, sgt_type;
};
struct ChainOut {
    std::vector<float> x, eh, ey;   // x: T x N; eh/ey: the routed products in their production layouts
    std::vector<float> sg, su, sy, sgate;   // the shared expert's pieces (probe/diagnosis)
    Tables tb;
    std::vector<int> ids;
    std::vector<float> wts, router, x0;
    int N, FF, K, T;
    bool eh_entry_ordered;
};
static ChainOut run_chain(const ChainGeom& g, uint64_t seed, int place_order, bool use_donor) {
    ChainOut out;
    const int T = g.T, N = g.N, FF = g.FF, FS = g.FS, NEXP = g.NEXP, K = g.K;
    const int n_ent = T * K;
    out.N = N;
    out.FF = FF;
    out.K = K;
    out.T = T;
    ref::Rng r(seed);

    const bool gu_lpr8 = (N == 2048);   // the engine's gu_lpr8: n_in == 2048 and an lpr8 pipeline exists
    const Fmt fg = fmt_of(g.gu_type), fd = fmt_of(g.dn_type);
    const size_t rb_g = (size_t)((N + fg.qk - 1) / fg.qk) * (size_t)fg.bb;
    const size_t rb_d = (size_t)((FF + fd.qk - 1) / fd.qk) * (size_t)fd.bb;
    const size_t eb_g = rb_g * (size_t)FF, eb_d = rb_d * (size_t)N;

    std::vector<uint8_t> Wg((size_t)NEXP * eb_g), Wu((size_t)NEXP * eb_g), Wd((size_t)NEXP * eb_d);
    make_weights(g.gu_type, Wg.data(), Wg.size(), seed + 1);
    make_weights(g.gu_type, Wu.data(), Wu.size(), seed + 2);
    make_weights(g.dn_type, Wd.data(), Wd.size(), seed + 3);

    const Fmt fr = fmt_of(g.router_type), fsh = fmt_of(g.shg_type), fshd = fmt_of(g.shd_type),
              fsg = fmt_of(g.sgt_type);
    const size_t rb_r = (size_t)((N + fr.qk - 1) / fr.qk) * (size_t)fr.bb;
    const size_t rb_sh = (size_t)((N + fsh.qk - 1) / fsh.qk) * (size_t)fsh.bb;
    const size_t rb_shd = (size_t)((FS + fshd.qk - 1) / fshd.qk) * (size_t)fshd.bb;
    const size_t rb_sg = (size_t)((N + fsg.qk - 1) / fsg.qk) * (size_t)fsg.bb;
    std::vector<uint8_t> Wr(rb_r * (size_t)NEXP), Wsh(rb_sh * (size_t)FS), Wshu(rb_sh * (size_t)FS),
        Wshd(rb_shd * (size_t)N), Wsg(rb_sg);
    make_weights(g.router_type, Wr.data(), Wr.size(), seed + 4);
    make_weights(g.shg_type, Wsh.data(), Wsh.size(), seed + 5);
    make_weights(g.shg_type, Wshu.data(), Wshu.size(), seed + 8);
    make_weights(g.shd_type, Wshd.data(), Wshd.size(), seed + 6);
    make_weights(g.sgt_type, Wsg.data(), Wsg.size(), seed + 7);

    out.x0.resize((size_t)T * N);
    for (size_t i = 0; i < out.x0.size(); ++i) out.x0[i] = r.sym();
    out.router.resize((size_t)T * NEXP);
    for (size_t i = 0; i < out.router.size(); ++i) out.router[i] = r.sym();

    out.ids.assign((size_t)n_ent, 0);
    out.wts.resize((size_t)n_ent);
    route_chunk(out.router.data(), out.ids.data(), out.wts.data(), NEXP, K, T);

    // the router gemv: push_c_gemv's branch - n_in == 2048 takes the dense lpr8 form (zero_ids),
    // everything else k_gemv_c
    std::vector<float> router_c((size_t)T * NEXP, 0.0f);
    {
        DenseCtx d(T);
        ArmCtx c{g.router_type, Wr.data(), rb_r, 0, N, NEXP, out.x0.data(), N, d.ids.data(),
                 d.ident.data(), d.off.data(), d.cnt.data(), 1, T};
        if (N == 2048) arm_lpr8(c, router_c.data());
        else arm_expc(c, router_c.data());
    }
    out.tb = tables_body(out.ids.data(), NEXP, K, n_ent, place_order);

    // the routed gate/up: gu_lpr8 -> gather + the per-entry decode lpr8 (entry-ordered outputs);
    // otherwise k_expc over the placed rows (token rows via srt_in_gu).  The donor swap replaces
    // the expert products only: per-expert batches over the placed rows, reading the token rows
    // directly (the batched form needs no gather).
    std::vector<float> eg((size_t)n_ent * FF), eu((size_t)n_ent * FF), eh((size_t)n_ent * FF);
    const uint8_t* wg_p = Wg.data();
    const uint8_t* wu_p = Wu.data();
    if (gu_lpr8) {
        std::vector<float> hg((size_t)n_ent * N);
        gather_t(out.x0.data(), hg.data(), N, K, n_ent);
        ArmCtx cg{g.gu_type, wg_p, rb_g, eb_g, N, FF, hg.data(), N, out.ids.data(),
                  out.tb.srt_gu.data(), out.tb.cnt_off.data() + NEXP, out.tb.cnt_off.data(), NEXP, n_ent};
        ArmCtx cu = cg;
        cu.W = wu_p;
        if (use_donor) {
            ArmCtx dg = cg;
            dg.x = out.x0.data();
            dg.srt_in = out.tb.srt_gu.data();   // placed -> token
            arm_donor(dg, eg.data());
            ArmCtx du_dg = dg;
            du_dg.W = wu_p;
            arm_donor(du_dg, eu.data());
        } else {
            arm_lpr8(cg, eg.data());
            arm_lpr8(cu, eu.data());
        }
    } else {
        ArmCtx cg{g.gu_type, wg_p, rb_g, eb_g, N, FF, out.x0.data(), N, out.ids.data(),
                  out.tb.srt_gu.data(), out.tb.cnt_off.data() + NEXP, out.tb.cnt_off.data(), NEXP, n_ent};
        ArmCtx cu = cg;
        cu.W = wu_p;
        if (use_donor) {
            arm_donor(cg, eg.data());
            ArmCtx cu_d = cu;
            arm_donor(cu_d, eu.data());
        } else {
            arm_expc(cg, eg.data());
            arm_expc(cu, eu.data());
        }
    }
    out.eh_entry_ordered = gu_lpr8 && !use_donor;
    swiglu(eg.data(), eu.data(), eh.data(), (size_t)n_ent * FF);
    out.eh = eh;

    // the down: dn_lpr8 is false at FF < 2048 (the lpr8 kernel is 2048-hardwired) -> k_expc.  Its
    // srt_in maps placed rows to the eh rows: entry-ordered eh (the lpr8 gate/up) goes through
    // srt_d, placed-ordered eh (the expc gate/up or either donor form) is the identity.
    std::vector<int> ident((size_t)n_ent);
    for (int i = 0; i < n_ent; ++i) ident[(size_t)i] = i;
    const int* dn_srt = out.eh_entry_ordered ? out.tb.srt_d.data() : ident.data();
    std::vector<float> ey((size_t)n_ent * N);
    {
        ArmCtx cd{g.dn_type, Wd.data(), rb_d, eb_d, FF, N, eh.data(), FF, out.ids.data(), dn_srt,
                  out.tb.cnt_off.data() + NEXP, out.tb.cnt_off.data(), NEXP, n_ent};
        if (use_donor) arm_donor(cd, ey.data());
        else arm_expc(cd, ey.data());
    }
    out.ey = ey;

    // the shared expert (dense products) + its gate, then the combine
    std::vector<float> sg((size_t)T * FS), su((size_t)T * FS), sh((size_t)T * FS), sy((size_t)T * N),
        sgate((size_t)T);
    {
        DenseCtx d(T);
        ArmCtx cg{g.shg_type, Wsh.data(), rb_sh, 0, N, FS, out.x0.data(), N, d.ids.data(),
                  d.ident.data(), d.off.data(), d.cnt.data(), 1, T};
        ArmCtx cu = cg;
        cu.W = Wshu.data();
        if (N == 2048) {
            arm_lpr8(cg, sg.data());
            arm_lpr8(cu, su.data());
        } else {
            arm_expc(cg, sg.data());
            arm_expc(cu, su.data());
        }
        swiglu(sg.data(), su.data(), sh.data(), (size_t)T * FS);   // before the down reads sh
        ArmCtx cd{g.shd_type, Wshd.data(), rb_shd, 0, FS, N, sh.data(), FS, d.ids.data(),
                  d.ident.data(), d.off.data(), d.cnt.data(), 1, T};
        arm_expc(cd, sy.data());
        ArmCtx cs{g.sgt_type, Wsg.data(), rb_sg, 0, N, 1, out.x0.data(), N, d.ids.data(),
                  d.ident.data(), d.off.data(), d.cnt.data(), 1, T};
        if (N == 2048) arm_lpr8(cs, sgate.data());
        else arm_expc(cs, sgate.data());
    }
    out.sg = sg;
    out.su = su;
    out.sy = sy;
    out.sgate = sgate;

    out.x = out.x0;
    moe_sum_chunk(out.x.data(), ey.data(), out.wts.data(), sy.data(), sgate.data(),
                  out.tb.pos.data(), K, N, T);
    return out;
}

// the chain's independent double reference: placement-free, computed straight from ids/wts with
// double-accumulated dots and double silu/sigmoid - the float-exact ideal the envelopes measure.
// Weight regeneration mirrors run_chain's seeds exactly.  The eh/ey fields (entry-ordered, rounded
// to float) let the stage probe attribute a swap mismatch to the gate/up side or the down side.
struct ChainRef {
    std::vector<float> x, eh, ey;
    std::vector<double> sg, su, sy, sgate;   // the shared expert's pieces (probe/diagnosis)
};
static double silu_d(double v) { return v / (1.0 + std::exp(-v)); }
static double sigm_d(double v) { return 1.0 / (1.0 + std::exp(-v)); }
static ChainRef chain_ref(const ChainGeom& g, const ChainOut& c, uint64_t seed) {
    const int T = g.T, N = g.N, FF = g.FF, FS = g.FS, NEXP = g.NEXP, K = g.K;
    const int n_ent = T * K;
    const Fmt fsh = fmt_of(g.shg_type), fshd = fmt_of(g.shd_type), fsg = fmt_of(g.sgt_type);
    const size_t rb_g = (size_t)((N + fmt_of(g.gu_type).qk - 1) / fmt_of(g.gu_type).qk) * (size_t)fmt_of(g.gu_type).bb;
    const size_t rb_d = (size_t)((FF + fmt_of(g.dn_type).qk - 1) / fmt_of(g.dn_type).qk) * (size_t)fmt_of(g.dn_type).bb;
    const size_t eb_g = rb_g * (size_t)FF, eb_d = rb_d * (size_t)N;
    std::vector<uint8_t> Wg((size_t)NEXP * eb_g), Wu((size_t)NEXP * eb_g), Wd((size_t)NEXP * eb_d);
    make_weights(g.gu_type, Wg.data(), Wg.size(), seed + 1);
    make_weights(g.gu_type, Wu.data(), Wu.size(), seed + 2);
    make_weights(g.dn_type, Wd.data(), Wd.size(), seed + 3);
    const size_t rb_sh = (size_t)((N + fsh.qk - 1) / fsh.qk) * (size_t)fsh.bb;
    const size_t rb_shd = (size_t)((FS + fshd.qk - 1) / fshd.qk) * (size_t)fshd.bb;
    const size_t rb_sg = (size_t)((N + fsg.qk - 1) / fsg.qk) * (size_t)fsg.bb;
    std::vector<uint8_t> Wsh(rb_sh * (size_t)FS), Wshu(rb_sh * (size_t)FS), Wshd(rb_shd * (size_t)N),
        Wsg(rb_sg);
    make_weights(g.shg_type, Wsh.data(), Wsh.size(), seed + 5);
    make_weights(g.shg_type, Wshu.data(), Wshu.size(), seed + 8);
    make_weights(g.shd_type, Wshd.data(), Wshd.size(), seed + 6);
    make_weights(g.sgt_type, Wsg.data(), Wsg.size(), seed + 7);

    // shared expert in double: sg (gate), su (up), sh = silu(sg)*su, sy = sh . Wshd, sgate
    std::vector<double> sg_ref((size_t)T * FS), su_ref((size_t)T * FS), sy_ref((size_t)T * N),
        sgate_ref((size_t)T);
    {
        std::vector<float> wf((size_t)FS * N);
        ref::dequant(g.shg_type, Wsh.data(), wf.data(), (size_t)FS * N);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < FS; ++o) {
                double a = 0;
                for (int k = 0; k < N; ++k) a += (double)c.x0[(size_t)t * N + k] * (double)wf[(size_t)o * N + k];
                sg_ref[(size_t)t * FS + o] = a;
            }
        std::vector<float> wu((size_t)FS * N);
        ref::dequant(g.shg_type, Wshu.data(), wu.data(), (size_t)FS * N);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < FS; ++o) {
                double a = 0;
                for (int k = 0; k < N; ++k) a += (double)c.x0[(size_t)t * N + k] * (double)wu[(size_t)o * N + k];
                su_ref[(size_t)t * FS + o] = a;
            }
        std::vector<float> wd((size_t)N * FS);
        ref::dequant(g.shd_type, Wshd.data(), wd.data(), (size_t)N * FS);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < N; ++o) {
                double a = 0;
                for (int k = 0; k < FS; ++k)
                    a += silu_d(sg_ref[(size_t)t * FS + k]) * su_ref[(size_t)t * FS + k] * (double)wd[(size_t)o * FS + k];
                sy_ref[(size_t)t * N + o] = a;
            }
        std::vector<float> w1((size_t)N);
        ref::dequant(g.sgt_type, Wsg.data(), w1.data(), (size_t)N);
        for (int t = 0; t < T; ++t) {
            double a = 0;
            for (int k = 0; k < N; ++k) a += (double)c.x0[(size_t)t * N + k] * (double)w1[k];
            sgate_ref[t] = a;
        }
    }

    // routed experts in double, expert-major (one dequant per matrix), placement-free
    std::vector<double> ehn((size_t)n_ent * FF), ey_ref((size_t)n_ent * N);
    {
        std::vector<float> wfg((size_t)FF * N), wfu((size_t)FF * N), wfd((size_t)N * FF);
        for (int e = 0; e < NEXP; ++e) {
            ref::dequant(g.gu_type, Wg.data() + (size_t)e * eb_g, wfg.data(), (size_t)FF * N);
            ref::dequant(g.gu_type, Wu.data() + (size_t)e * eb_g, wfu.data(), (size_t)FF * N);
            ref::dequant(g.dn_type, Wd.data() + (size_t)e * eb_d, wfd.data(), (size_t)N * FF);
            for (int i = 0; i < n_ent; ++i) {
                if (c.ids[i] != e) continue;
                const int t = i / K;
                for (int o = 0; o < FF; ++o) {
                    double a = 0, b = 0;
                    for (int k = 0; k < N; ++k) {
                        const double xv = (double)c.x0[(size_t)t * N + k];
                        a += xv * (double)wfg[(size_t)o * N + k];
                        b += xv * (double)wfu[(size_t)o * N + k];
                    }
                    ehn[(size_t)i * FF + o] = silu_d(a) * b;
                }
                for (int o = 0; o < N; ++o) {
                    double a = 0;
                    for (int k = 0; k < FF; ++k) a += ehn[(size_t)i * FF + k] * (double)wfd[(size_t)o * FF + k];
                    ey_ref[(size_t)i * N + o] = a;
                }
            }
        }
    }

    // the combine in double, slots ascending as the kernel sums (reads the shared pieces; the
    // moves into the result come after)
    ChainRef out;
    out.eh.resize((size_t)n_ent * FF);
    out.ey.resize((size_t)n_ent * N);
    for (size_t i = 0; i < ehn.size(); ++i) out.eh[i] = (float)ehn[i];
    for (size_t i = 0; i < ey_ref.size(); ++i) out.ey[i] = (float)ey_ref[i];
    out.x.resize((size_t)T * N);
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < N; ++o) {
            double acc = 0;
            for (int s = 0; s < K; ++s)
                acc += (double)c.wts[(size_t)t * K + s] * ey_ref[(size_t)(t * K + s) * N + o];
            out.x[(size_t)t * N + o] =
                (float)((double)c.x0[(size_t)t * N + o] +
                        (acc + sigm_d(sgate_ref[t]) * sy_ref[(size_t)t * N + o]));
        }
    out.sg = std::move(sg_ref);
    out.su = std::move(su_ref);
    out.sy = std::move(sy_ref);
    out.sgate = std::move(sgate_ref);
    return out;
}

// ================= stages =================

// stage 1: routing - the transcription vs the sort-based reference (ids + wts bit-identical),
// uniform and tie-rich logits (the tie-break's index-asc rule is the thing that can silently break)
static void stage_route() {
    std::printf("route\n");
    const int T = 64, E = SMALL_NEXP, K = SMALL_K;
    ref::Rng r(101);
    std::vector<float> logits((size_t)T * E), logits_tie;
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = r.sym();
    logits_tie = logits;
    for (size_t i = 0; i < logits_tie.size(); ++i) logits_tie[i] = std::round(logits_tie[i] * 4.0f) / 4.0f;
    for (int which = 0; which < 2; ++which) {
        const float* lg = which ? logits_tie.data() : logits.data();
        std::vector<int> ids1((size_t)T * K), ids2((size_t)T * K);
        std::vector<float> wts1((size_t)T * K), wts2((size_t)T * K);
        route_chunk(lg, ids1.data(), wts1.data(), E, K, T);
        route_ref(lg, ids2.data(), wts2.data(), E, K, T);
        const char* nm = which ? "route ties" : "route ids";
        check_i32(nm, ids1.data(), ids2.data(), (size_t)T * K);
        check(which ? "route tie wts" : "route wts", wts1.data(), wts2.data(), (size_t)T * K);
    }
}

// stage 2: the entry tables - cnt/off bit-identical to the naive pass, the placement invariant,
// and the numbers' independence from the atomic rank order (the chain re-run on two legal orders)
static void stage_tables() {
    std::printf("tables\n");
    const ChainGeom g{32, SMALL_N, SMALL_FF, SMALL_FS, SMALL_NEXP, SMALL_K,
                      ref::Q4_K, ref::Q4_K, ref::Q8_0, ref::Q4_K, ref::Q8_0, ref::Q4_K};
    ChainOut c = run_chain(g, 202, 0, false);
    const int n_ent = c.T * c.K;
    Tables naive = tables_naive(c.ids.data(), SMALL_NEXP, SMALL_K, n_ent);
    check_i32("tables cnt/off", c.tb.cnt_off.data(), naive.cnt_off.data(), 2 * (size_t)SMALL_NEXP);
    check_placement("tables place", c.ids.data(), c.tb, SMALL_NEXP, SMALL_K, n_ent);
    // the rank order must not move the numbers: the same chain on the entry-major placement
    ChainOut c2 = run_chain(g, 202, 1, false);
    check("tables order-inv", c.x.data(), c2.x.data(), (size_t)c.T * c.N);
}

// stage 3: the gather - pure row copies
static void stage_gather() {
    std::printf("gather\n");
    const int T = 32, K = SMALL_K, N = SMALL_N;
    ref::Rng r(303);
    std::vector<float> x((size_t)T * N), y1((size_t)T * K * N), y2((size_t)T * K * N);
    for (size_t i = 0; i < x.size(); ++i) x[i] = r.sym();
    gather_t(x.data(), y1.data(), N, K, T * K);
    for (int i = 0; i < T * K; ++i)
        for (int o = 0; o < N; ++o) y2[(size_t)i * N + o] = x[(size_t)(i / K) * N + o];
    check("gather", y1.data(), y2.data(), y1.size());
}

// stage 4: the chunk dense gemv (k_gemv_c) vs the grouped expert gemv (k_expc) - the same plain
// order, bit-identical; plus the envelope vs the double reference
static void stage_gemv() {
    std::printf("gemv chunk\n");
    const int T = 32, N = SMALL_N, FF = SMALL_FF;
    for (int type : {ref::Q4_K, ref::Q8_0}) {
        const Fmt f = fmt_of(type);
        const size_t rb = (size_t)((N + f.qk - 1) / f.qk) * (size_t)f.bb;
        std::vector<uint8_t> W(rb * (size_t)FF);
        make_weights(type, W.data(), W.size(), 400 + type);
        ref::Rng r(401);
        std::vector<float> x((size_t)T * N);
        for (size_t i = 0; i < x.size(); ++i) x[i] = r.sym();
        DenseCtx d(T);
        ArmCtx c{type, W.data(), rb, 0, N, FF, x.data(), N, d.ids.data(), d.ident.data(), d.off.data(),
                 d.cnt.data(), 1, T};
        std::vector<float> y1((size_t)T * FF), y2((size_t)T * FF);
        gemv_c(c, y1.data(), T);
        arm_expc(c, y2.data());
        char nm[32];
        std::snprintf(nm, sizeof nm, "gemv chunk %s", ref::type_name(type));
        check(nm, y1.data(), y2.data(), y1.size());
        // the double reference (exact dequant, double accumulation)
        std::vector<float> wf((size_t)FF * N);
        ref::dequant(type, W.data(), wf.data(), (size_t)FF * N);
        std::vector<float> yref((size_t)T * FF);
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < FF; ++o) {
                double a = 0;
                for (int k = 0; k < N; ++k) a += (double)x[(size_t)t * N + k] * (double)wf[(size_t)o * N + k];
                yref[(size_t)t * FF + o] = (float)a;
            }
        check_rel(nm, y1.data(), yref.data(), y1.size(), 1e-5);
    }
}

// stage 5: push_c_gemv's branch rule at n_in = 2048 - the dense lpr8 form vs the plain k_gemv_c
// form (the documented ~1 ulp order difference, measured), plus the branch-selection table
static void stage_gemv2048() {
    std::printf("gemv 2048\n");
    const int T = 16, N = REAL_N, OUT = REAL_FF;
    for (int type : {ref::Q4_K, ref::Q6_K}) {
        const Fmt f = fmt_of(type);
        const size_t rb = (size_t)(N / f.qk) * (size_t)f.bb;
        std::vector<uint8_t> W(rb * (size_t)OUT);
        make_weights(type, W.data(), W.size(), 500 + type);
        ref::Rng r(501);
        std::vector<float> x((size_t)T * N);
        for (size_t i = 0; i < x.size(); ++i) x[i] = r.sym();
        DenseCtx d(T);
        ArmCtx c{type, W.data(), rb, 0, N, OUT, x.data(), N, d.ids.data(), d.ident.data(), d.off.data(),
                 d.cnt.data(), 1, T};
        std::vector<float> y1((size_t)T * OUT), y2((size_t)T * OUT);
        arm_lpr8(c, y1.data());
        arm_expc(c, y2.data());
        char nm[32];
        std::snprintf(nm, sizeof nm, "gemv2048 %s", ref::type_name(type));
        char nm2[64];
        std::snprintf(nm2, sizeof nm2, "%s vs plain", nm);
        check_rel(nm2, y1.data(), y2.data(), y1.size(), 1e-5);
    }
    std::printf("  branch table   (STRATA_Q36M_DENSE_MMA off; lpr8 types Q4_K/Q5_K/Q6_K/IQ2_XS):\n");
    struct M { const char* w; int n_in, n_out; };
    const M ms[] = {{"ffn_gate_inp (router)", REAL_N, REAL_NEXP}, {"attn/ssm dense 2048-ins", REAL_N, 10240},
                    {"ffn_gate/up_shexp", REAL_N, REAL_FS}, {"ffn_gate_inp_shexp", REAL_N, 1},
                    {"experts gate/up (gu_lpr8)", REAL_N, REAL_FF}, {"experts down (n_in 640)", REAL_FF, REAL_N}};
    for (const M& m : ms)
        for (int type : {ref::Q4_K, ref::Q8_0}) {
            const bool lpr8 = m.n_in == 2048 && (type == ref::Q4_K || type == ref::Q5_K || type == ref::Q6_K ||
                                                 type == ref::IQ2_XS);
            std::printf("    %-26s %5d->%5d %-6s -> %s\n", m.w, m.n_in, m.n_out, ref::type_name(type),
                        lpr8 ? "k_gemv_lpr8 (dense form, grid[1]=nt)" : "k_gemv_c");
        }
}

// stage 6: the routed experts - lpr8 vs expc over the same replayed routing at the real gate/up
// shape; the documented order difference, quantified per type
static void stage_experts() {
    std::printf("experts\n");
    const int T = 32, N = REAL_N, FF = REAL_FF, E = REAL_NEXP, K = REAL_K;
    const int n_ent = T * K;
    ref::Rng r(601);
    std::vector<float> logits((size_t)T * E);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = r.sym();
    std::vector<int> ids((size_t)n_ent);
    std::vector<float> wts((size_t)n_ent);
    route_chunk(logits.data(), ids.data(), wts.data(), E, K, T);
    const Tables tb = tables_body(ids.data(), E, K, n_ent, 0);
    std::vector<float> x((size_t)T * N);
    for (size_t i = 0; i < x.size(); ++i) x[i] = r.sym();
    std::vector<float> hg((size_t)n_ent * N);
    gather_t(x.data(), hg.data(), N, K, n_ent);
    for (int type : {ref::Q4_K, ref::Q5_K, ref::Q6_K}) {
        const Fmt f = fmt_of(type);
        const size_t rb = (size_t)(N / f.qk) * (size_t)f.bb, eb = rb * (size_t)FF;
        std::vector<uint8_t> Wg((size_t)E * eb), Wu((size_t)E * eb);
        make_weights(type, Wg.data(), Wg.size(), 610 + type);
        make_weights(type, Wu.data(), Wu.size(), 620 + type);
        ArmCtx cg{type, Wg.data(), rb, eb, N, FF, x.data(), N, ids.data(), tb.srt_gu.data(),
                  tb.cnt_off.data() + E, tb.cnt_off.data(), E, n_ent};   // expc reads the token rows via srt_gu
        ArmCtx cu = cg;
        cu.W = Wu.data();
        std::vector<float> eg_l((size_t)n_ent * FF), eg_p((size_t)n_ent * FF), eu_l((size_t)n_ent * FF),
            eu_p((size_t)n_ent * FF);
        ArmCtx cl = cg;                                    // the lpr8 arm reads the gathered rows
        cl.x = hg.data();
        arm_lpr8(cl, eg_l.data());
        arm_expc(cg, eg_p.data());
        ArmCtx clu = cu;
        clu.x = hg.data();
        arm_lpr8(clu, eu_l.data());
        arm_expc(cu, eu_p.data());
        std::vector<float> ehl((size_t)n_ent * FF), ehp((size_t)n_ent * FF);
        swiglu(eg_l.data(), eu_l.data(), ehl.data(), ehl.size());
        swiglu(eg_p.data(), eu_p.data(), ehp.data(), ehp.size());
        std::vector<float> ehp_mapped((size_t)n_ent * FF);   // expc is placed-ordered: map entry i -> pos[i]
        for (int i = 0; i < n_ent; ++i)
            memcpy(ehp_mapped.data() + (size_t)i * FF, ehp.data() + (size_t)tb.pos[i] * FF,
                   sizeof(float) * (size_t)FF);
        char nm[48];
        std::snprintf(nm, sizeof nm, "experts %s eh lpr8/expc", ref::type_name(type));
        check_rel(nm, ehl.data(), ehp_mapped.data(), ehl.size(), 1e-5);
    }
}

// the fp16-operand and exact double references for one expert matrix against a fixed x batch
// (dense_mm_parity's convention: wf = ref::dequant, the refs accumulate in double)
static void gemm_refs(int type, const uint8_t* Wm, int n_out, int n_in, const std::vector<float>& xrows,
                      int M, std::vector<float>& ref16, std::vector<float>& refex) {
    std::vector<float> wf((size_t)n_out * n_in);
    ref::dequant(type, Wm, wf.data(), (size_t)n_out * n_in);
    ref16.assign((size_t)M * n_out, 0.0f);
    refex.assign((size_t)M * n_out, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int o = 0; o < n_out; ++o) {
            double a16 = 0, ae = 0;
            for (int k = 0; k < n_in; ++k) {
                const float xv = xrows[(size_t)m * n_in + k], wv = wf[(size_t)o * n_in + k];
                a16 += (double)h16(xv) * (double)h16(wv);
                ae += (double)xv * (double)wv;
            }
            ref16[(size_t)m * n_out + o] = (float)a16;
            refex[(size_t)m * n_out + o] = (float)ae;
        }
}

// stage 7: the donor at the expert shapes - the dequant-once MMA form vs the fp16-operand reference
// (<= 1e-5, the dense_mm_parity bound), vs the exact reference (the fp16-rounding envelope), and vs
// the production GEMV arm (the accuracy cost of the swap) across the rows-per-expert ladder
static void stage_donor() {
    std::printf("donor\n");
    struct S { const char* name; int n_in, n_out, type, gemv_type; };
    const S ss[] = {
        {"gu 2048->640", REAL_N, REAL_FF, ref::Q4_K, ref::Q4_K},
        {"gu 2048->640", REAL_N, REAL_FF, ref::IQ2_XS, -1},   // IQ2_XS: no plain arm here (M2's formats)
        {"dn 512->2048", 512, REAL_N, ref::Q4_K, ref::Q4_K},
        {"dn 640->2048", REAL_FF, REAL_N, ref::Q8_0, ref::Q8_0},
    };
    for (const S& s : ss) {
        std::vector<uint8_t> Wm((size_t)s.n_out * (size_t)((s.n_in + fmt_of(s.type).qk - 1) / fmt_of(s.type).qk) *
                                (size_t)fmt_of(s.type).bb);
        make_weights(s.type, Wm.data(), Wm.size(), 700 + s.type + s.n_in);
        ref::Rng r(701 + s.n_in);
        const int MMAX = 64;
        std::vector<float> xrows((size_t)MMAX * s.n_in);
        for (size_t i = 0; i < xrows.size(); ++i) xrows[i] = r.sym();
        for (int M : {1, 2, 4, 8, 16, 32, 64}) {
            DenseCtx d(M);
            ArmCtx cd{s.type, Wm.data(),
                      (size_t)((s.n_in + fmt_of(s.type).qk - 1) / fmt_of(s.type).qk) * (size_t)fmt_of(s.type).bb, 0,
                      s.n_in, s.n_out, xrows.data(), s.n_in, d.ids.data(), d.ident.data(), d.off.data(),
                      d.cnt.data(), 1, M};
            std::vector<float> yd((size_t)M * s.n_out);
            arm_donor(cd, yd.data());
            std::vector<float> r16, rex;
            gemm_refs(s.type, Wm.data(), s.n_out, s.n_in, xrows, M, r16, rex);
            char nm[64];
            std::snprintf(nm, sizeof nm, "donor %s %s M=%d", s.name, ref::type_name(s.type), M);
            check_rel(nm, yd.data(), r16.data(), yd.size(), 1e-5);
            const double re = rel_rms(yd.data(), rex.data(), yd.size());
            if (s.gemv_type >= 0) {
                std::vector<float> yg((size_t)M * s.n_out);
                arm_expc(cd, yg.data());
                std::printf("    vs exact %.2e | vs gemv %.2e\n", re, rel_rms(yd.data(), yg.data(), yd.size()));
            } else {
                std::printf("    vs exact %.2e (no gemv arm for this type)\n", re);
            }
        }
    }
}

// stage 8: the full chains - small (all-plain) and 2048 (the production gu_lpr8 route), each vs the
// independent double reference; the 2048 chain re-run with the donor arms swapped in gives the
// end-to-end envelopes at eh/ey/x (T10's dry-run numbers)
static void stage_chain() {
    std::printf("chain\n");
    {
        const ChainGeom g{32, SMALL_N, SMALL_FF, SMALL_FS, SMALL_NEXP, SMALL_K,
                          ref::Q4_K, ref::Q4_K, ref::Q8_0, ref::Q4_K, ref::Q8_0, ref::Q4_K};
        ChainOut c = run_chain(g, 801, 0, false);
        ChainRef rs = chain_ref(g, c, 801);
        check_rel("chain small x", c.x.data(), rs.x.data(), c.x.size(), 1e-7);
        // the small chain's routed products are placed-ordered (all-plain): map before comparing
        std::vector<float> eh_m((size_t)c.T * c.K * c.FF), ey_m((size_t)c.T * c.K * c.N);
        for (int i = 0; i < c.T * c.K; ++i) {
            memcpy(eh_m.data() + (size_t)i * c.FF, c.eh.data() + (size_t)c.tb.pos[i] * c.FF,
                   sizeof(float) * (size_t)c.FF);
            memcpy(ey_m.data() + (size_t)i * c.N, c.ey.data() + (size_t)c.tb.pos[i] * c.N,
                   sizeof(float) * (size_t)c.N);
        }
        check_rel("chain small eh ref", eh_m.data(), rs.eh.data(), eh_m.size(), 1e-6);
        check_rel("chain small ey ref", ey_m.data(), rs.ey.data(), ey_m.size(), 1e-6);
    }
    {
        const ChainGeom g{32, REAL_N, REAL_FF, REAL_FS, REAL_NEXP, REAL_K,
                          ref::Q4_K, ref::Q4_K, ref::Q8_0, ref::Q4_K, ref::Q8_0, ref::Q4_K};
        ChainOut cg = run_chain(g, 802, 0, false);
        ChainRef rr = chain_ref(g, cg, 802);
        check_rel("chain2048 x gemv", cg.x.data(), rr.x.data(), cg.x.size(), 1e-7);
        check_rel("chain2048 eh ref", cg.eh.data(), rr.eh.data(), cg.eh.size(), 1e-6);
        // the ref's ey is entry-ordered, the chain's placed: map the chain's through pos
        std::vector<float> ey_mapped((size_t)cg.T * cg.K * cg.N);
        for (int i = 0; i < cg.T * cg.K; ++i)
            memcpy(ey_mapped.data() + (size_t)i * cg.N, cg.ey.data() + (size_t)cg.tb.pos[i] * cg.N,
                   sizeof(float) * (size_t)cg.N);
        check_rel("chain2048 ey ref", ey_mapped.data(), rr.ey.data(), ey_mapped.size(), 1e-6);
        ChainOut cd = run_chain(g, 802, 0, true);
        // eh: the gemv chain's is entry-ordered, the donor chain's placed - compare through pos
        std::vector<float> eh_d_mapped((size_t)cg.T * cg.K * cg.FF);
        for (int i = 0; i < cg.T * cg.K; ++i)
            memcpy(eh_d_mapped.data() + (size_t)i * cg.FF,
                   cd.eh.data() + (size_t)cg.tb.pos[i] * cg.FF, sizeof(float) * (size_t)cg.FF);
        // the swap envelopes (the T10 dry-run numbers): the donor's fp16 operands move the routed
        // products by the ~3e-4 class and the final x by ~1e-6 (the routed part is a small share of x)
        check_rel("chain2048 eh swap", eh_d_mapped.data(), cg.eh.data(), eh_d_mapped.size(), 1e-3);
        check_rel("chain2048 ey swap", cd.ey.data(), cg.ey.data(), cd.ey.size(), 1e-3);
        check_rel("chain2048 x swap", cd.x.data(), cg.x.data(), cd.x.size(), 1e-5);
        ChainRef rd = chain_ref(g, cd, 802);
        check_rel("chain2048 x donor-ref", cd.x.data(), rd.x.data(), cd.x.size(), 1e-5);
    }
}

// stage 9: the rows-per-expert histogram (the dry-run).  The routing replay (the k_route_chunk
// transcription) over synthetic router logits; the authoritative on-Metal histogram comes from
// T13's counter lineage.  Distributions: uniform, skewed (a seeded Zipf-favored subset), peaked.
struct HistStats {
    int distinct, zeros, le2;
    int mn, p50, p90, mx;
    double share[5];   // entries in experts with rows >= {4, 8, 16, 32, 64}
    int gcount[5];     // experts with rows >= theta (the hybrid's per-expert GEMM class)
};
static HistStats hist_one(const int* ids, int n_ent, int n_expert) {
    std::vector<int> rows((size_t)n_expert, 0);
    for (int i = 0; i < n_ent; ++i) ++rows[ids[i]];
    HistStats s{};
    std::vector<int> srt = rows;
    std::sort(srt.begin(), srt.end());
    s.mn = srt.front();
    s.mx = srt.back();
    s.p50 = srt[(size_t)(n_expert / 2)];
    s.p90 = srt[(size_t)(n_expert * 9 / 10)];
    s.distinct = 0;
    s.zeros = 0;
    s.le2 = 0;
    for (int v : rows) {
        s.distinct += v > 0;
        s.zeros += v == 0;
        s.le2 += v > 0 && v <= 2;
    }
    const int th[5] = {4, 8, 16, 32, 64};
    for (int j = 0; j < 5; ++j) {
        int above = 0, gcnt = 0;
        for (int v : rows) {
            if (v >= th[j]) {
                above += v;
                ++gcnt;
            }
        }
        s.share[j] = 100.0 * above / n_ent;
        s.gcount[j] = gcnt;
    }
    return s;
}
static void histogram_report() {
    std::printf("histogram (dry-run; the authoritative on-Metal histogram is T13's counter lineage)\n");
    const int E = REAL_NEXP, K = REAL_K;
    const int sizes[4] = {128, 512, 1024, 2048};
    const char* dnames[3] = {"uniform", "skewed", "peaked"};
    const int th[5] = {4, 8, 16, 32, 64};
    for (int si = 0; si < 4; ++si) {
        const int T = sizes[si];
        for (int dist = 0; dist < 3; ++dist) {
            HistStats agg[3];
            for (int sd = 0; sd < 3; ++sd) {
                const uint64_t seed = 900 + (uint64_t)si * 100 + (uint64_t)dist * 10 + (uint64_t)sd;
                ref::Rng r(seed);
                std::vector<float> logits((size_t)T * E);
                if (dist == 0) {
                    for (size_t i = 0; i < logits.size(); ++i) logits[i] = r.sym();
                } else if (dist == 1) {
                    std::vector<int> perm((size_t)E);      // a seeded Zipf-favored subset (soft: the
                    for (int e = 0; e < E; ++e) perm[(size_t)e] = e;   // imbalance regime, not the
                    for (int e = E - 1; e > 0; --e) {      // degenerate one - that is 'peaked')
                        const int j = (int)(r.uni() * (float)(e + 1));
                        std::swap(perm[(size_t)e], perm[(size_t)j]);
                    }
                    std::vector<float> base((size_t)E);
                    for (int e = 0; e < E; ++e) base[(size_t)perm[(size_t)e]] = -0.35f * std::log((float)(e + 1));
                    for (size_t i = 0; i < logits.size(); ++i) logits[i] = base[i % E] + 0.5f * r.sym();
                } else {
                    for (size_t i = 0; i < logits.size(); ++i)
                        logits[i] = ((float)(i % E) < 8.0f ? 4.0f : 0.0f) + 0.25f * r.sym();
                }
                std::vector<int> ids((size_t)T * K);
                std::vector<float> wts((size_t)T * K);
                route_chunk(logits.data(), ids.data(), wts.data(), E, K, T);
                agg[sd] = hist_one(ids.data(), T * K, E);
                const HistStats& s = agg[sd];
                std::printf("  T=%4d %-7s seed%lld: ent %5d hit %3d/%3d rows min %3d p50 %3d p90 %3d max %3d "
                            "| 0-row %3d <=2-row %3d | >=16 %5.1f%%\n",
                            T, dnames[dist], (long long)(sd + 1), T * K, s.distinct, E, s.mn, s.p50, s.p90, s.mx,
                            s.zeros, s.le2, s.share[2]);
            }
            const int mxd = std::max({agg[0].mx, agg[1].mx, agg[2].mx});
            const int mnd = std::min({agg[0].mx, agg[1].mx, agg[2].mx});
            std::printf("  T=%4d %-7s range : max-rows %d..%d; threshold view (theta: GEMM experts G / entries "
                        "above; per-product dispatches hybrid 3G+rem vs today 1):\n", T, dnames[dist], mnd, mxd);
            for (int j = 0; j < 5; ++j) {
                const double share = (agg[0].share[j] + agg[1].share[j] + agg[2].share[j]) / 3.0;
                const int G = agg[1].gcount[j];
                const int rem = share < 99.999 ? 1 : 0;
                std::printf("    theta %2d: G %3d/%3d, entries above %5.1f%% -> hybrid %4d vs today 1 dispatches/product\n",
                            th[j], G, E, share, 3 * G + rem);
            }
        }
    }
}

}  // namespace

int main() {
    std::printf("moe_sim: host-side parity simulation of the chunk MoE expert path (pure C++, no Metal)\n");
    stage_route();
    stage_tables();
    stage_gather();
    stage_gemv();
    stage_gemv2048();
    stage_experts();
    stage_donor();
    stage_chain();
    histogram_report();
    if (g_fails == 0) {
        std::printf("moe_sim: all stages OK (%d checks)\n", g_checks);
        return 0;
    }
    std::printf("moe_sim: %d FAILURES (%d checks passed)\n", g_fails, g_checks);
    return 1;
}
