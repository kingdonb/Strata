// metal/tests/q36_chunk_parity.cpp - the prompt-chunk kernels (q36.metal) against the decode kernels they
// must match bit for bit, stage by stage, on synthetic inputs: the engine's chunked prefill keeps the
// token-by-token path's exact arithmetic, so every stage compares equal with memcmp.
//
//   q36_chunk_parity    embed rows, dense gemv (F32 / Q8_0 / Q4_K / Q5_K / Q6_K), the n_in=2048 lpr8
//                       gemv, top-k routing, the expert entry tables (k_tables vs the old four-dispatch
//                       chain) + grouped expert gemv (Q4_K / IQ2_XS / IQ3_XXS), conv tiles, the delta
//                       rule, gated norm, the MoE combine, the full MoE chain as one graph vs separate
#include "mtl_rt.h"
#include "ref_quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// test geometry (small, but every block/unit boundary the real shapes hit)
constexpr int T = 8;            // tokens per chunk
constexpr int N = 256;          // n_embd
constexpr int FF = 96, FS = 64;
constexpr int NEXP = 16, K = 4;
constexpr int HK = 2, HV = 4, S = 128;
constexpr int C = (2 * HK + HV) * S, ZV = HV * S;
constexpr int H = 4, HKV = 2, HD = 128;

// q36.metal's GemvParam (constant buffer 4 of the gemv kernels)
struct GemvParam {
    unsigned long row_bytes;
    unsigned long expert_bytes;
    int n_in;
    int n_out;
    unsigned long x_stride;
};

int g_fails = 0;

// garbage bytes as half scales give NaN (~1/32 of blocks): clear the exponent bits of every odd byte so
// the dequanted scales stay finite and the NaN-tracking stays meaningful
template <class C> static void sanitize(C& c) {
    for (size_t i = 1; i < c.size(); i += 2) c[i] &= 0x83;
}
static void sanitize(mtl::Buffer& b) {
    unsigned char* p = (unsigned char*) b.ptr;
    for (size_t i = 1; i < b.bytes; i += 2) p[i] &= 0x83;
}

static inline bool feq(float a, float b) {
    return a == b || (std::isnan(a) && std::isnan(b));   // both-NaN counts as equal (bit patterns may differ)
}

void check(const char* stage, const void* a, const void* b, size_t bytes) {
    const float* x = (const float*) a;
    const float* y = (const float*) b;
    const size_t n = bytes / 4;
    size_t first = n, count = 0, nans = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(x[i]) || std::isnan(y[i])) {
            ++nans;
            continue;   // NaN cells are tracked separately: they mean out-of-bounds/garbage reads upstream
        }
        if (!feq(x[i], y[i])) {
            if (first == n) first = i;
            ++count;
        }
    }
    if (count == 0 && nans == 0) {
        std::printf("  %-14s OK\n", stage);
        return;
    }
    std::printf("  %-14s FAIL: %zu real diffs (first at %zu: %.9g vs %.9g), %zu NaN cells\n", stage, count,
                first < n ? first : 0, first < n ? x[first] : 0.0f, first < n ? y[first] : 0.0f, nans);
    ++g_fails;
}

// the expert gemv's outputs live entry-major on the chunk side and per (token, slot) on the decode side
void check_scattered(const char* stage, const float* decode_rows, const float* entry_rows, const int* pos) {
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            const float* a = decode_rows + (size_t)(t * K + k) * FF;
            const float* b = entry_rows + (size_t) pos[t * K + k] * FF;
            for (int o = 0; o < FF; ++o)
                if (!feq(a[o], b[o])) {
                    std::printf("  %-14s FAIL at token %d slot %d dim %d (%.9g vs %.9g)\n", stage, t, k, o, a[o], b[o]);
                    ++g_fails;
                    return;
                }
        }
    }
    std::printf("  %-14s OK\n", stage);
}

// the chunk's entry-major rows against another entry-major set, matched by each (token, slot)'s row (the
// placement rank order can differ between runs; the VALUES must not)
void check_scattered2(const char* stage, const float* a_rows, const int* pos_a, const float* b_rows,
                      const int* pos_b) {
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k) {
            const float* a = a_rows + (size_t) pos_a[t * K + k] * FF;
            const float* b = b_rows + (size_t) pos_b[t * K + k] * FF;
            for (int o = 0; o < FF; ++o)
                if (!feq(a[o], b[o])) {
                    std::printf("  %-14s FAIL at token %d slot %d dim %d (%.9g vs %.9g)\n", stage, t, k, o, a[o],
                                b[o]);
                    ++g_fails;
                    return;
                }
        }
    std::printf("  %-14s OK\n", stage);
}

void check_bytes(const char* stage, const void* a, const void* b, size_t bytes) {
    const unsigned char* x = (const unsigned char*) a;
    const unsigned char* y = (const unsigned char*) b;
    for (size_t i = 0; i < bytes; ++i)
        if (x[i] != y[i]) {
            std::printf("  %-14s FAIL at byte %zu (%02x vs %02x)\n", stage, i, x[i], y[i]);
            ++g_fails;
            return;
        }
    std::printf("  %-14s OK\n", stage);
}

// the WY form reassociates the scan, so its stages compare to rounding, not bits: the relative error of
// a against b (gdn_prompt_parity's scheme)
void check_rel(const char* stage, const void* av, const void* bv, size_t bytes, double tol) {
    const float* a = (const float*) av;
    const float* b = (const float*) bv;
    const size_t n = bytes / 4;
    double se = 0, sr = 0;
    for (size_t i = 0; i < n; ++i) {
        const double e = (double) a[i] - (double) b[i];
        se += e * e;
        sr += (double) b[i] * (double) b[i];
    }
    const double r = std::sqrt(se / std::max(sr, 1e-30));
    if (r < tol) {
        std::printf("  %-14s OK (rel %.2e)\n", stage, r);
        return;
    }
    std::printf("  %-14s FAIL: rel %.3e >= %.0e\n", stage, r, tol);
    ++g_fails;
}

// k_tables' placement invariant: pos is a permutation of the entry rows, each row -> its entry index and
// token, each entry inside its expert's [off, off + cnt) run (the rank order itself varies run to run)
void check_placement(const char* stage, const int* ids, const int* cnt_off, const int* srt_gu, const int* srt_d,
                     const int* pos, int n_ent) {
    std::vector<char> seen((size_t) n_ent, 0);
    for (int i = 0; i < n_ent; ++i) {
        const int row = pos[i], e = ids[i];
        const bool bad = row < 0 || row >= n_ent || seen[row] || srt_d[row] != i || srt_gu[row] != i / K
                         || row < cnt_off[256 + e] || row >= cnt_off[256 + e] + cnt_off[e];
        if (bad) {
            std::printf("  %-14s FAIL at entry %d (row %d, expert %d)\n", stage, i, row, e);
            ++g_fails;
            return;
        }
        seen[row] = 1;
    }
    std::printf("  %-14s OK\n", stage);
}

}  // namespace

// --bench: the WY chain's per-part timings against k_gdn_chunk at the real geometry (HK 16, HV 48,
// S 128, C 10240), T = 512 and 2048, median of 5 after a 0.3 s warm-up - the numbers the engine swap
// decision is checked against (docs/QWEN36_METAL.md's chunk section)
static void run_bench() {
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "q36.metal"});
    const int HK = 16, HV = 48, S = 128, C = (2 * HK + HV) * S, ZV = HV * S;
    const float eps = 1e-6f;
    mtl::Pipeline p_gdnc = cx.pipeline(lib, "k_gdn_chunk"), p_prep = cx.pipeline(lib, "k_gdn_prep_c"),
                  p_cp = cx.pipeline(lib, "k_gdn_cprep_c"), p_cs = cx.pipeline(lib, "k_gdn_cstate_c"),
                  p_co = cx.pipeline(lib, "k_gdn_cout_c");
    ref::Rng r(7);
    auto timeit = [&](const std::vector<mtl::Dispatch>& ds) {
        cx.warm(ds, 0.3);
        std::vector<double> t;
        for (int i = 0; i < 5; ++i) t.push_back(cx.run(ds));
        std::sort(t.begin(), t.end());
        return t[t.size() / 2];
    };
    for (int T : {512, 2048}) {
        const int NC = (T + 31) / 32;
        auto F = [&](size_t n) { return cx.alloc(4 * n); };
        mtl::Buffer conv = F((size_t) T * C), alpha = F(T * HV), beta = F(T * HV), dtb = F(HV), ssa = F(HV);
        mtl::Buffer S1 = F(HV * S * S), S2 = F(HV * S * S), out1 = F((size_t) T * ZV);
        mtl::Buffer qn = F((size_t) T * HK * S), kn = F((size_t) T * HK * S), vc = F((size_t) T * HV * S),
                    gb = cx.alloc(8 * (size_t) T * HV), U = F((size_t) HV * NC * 32 * S), W = F((size_t) HV * NC * 32 * S),
                    lgo = F((size_t) HV * NC * 32), delta = F((size_t) HV * NC * 32 * S);
        for (size_t i = 0; i < (size_t) T * C; ++i) conv.as<float>()[i] = r.sym() * 2.0f;
        for (int i = 0; i < T * HV; ++i) {
            alpha.as<float>()[i] = r.sym() * 2.0f;
            beta.as<float>()[i] = r.sym() * 2.0f;
        }
        for (int i = 0; i < HV; ++i) {
            dtb.as<float>()[i] = r.sym();
            ssa.as<float>()[i] = -std::exp(r.sym());
        }
        for (int i = 0; i < HV * S * S; ++i) S1.as<float>()[i] = r.sym() * 0.1f;
        std::memcpy(S2.ptr, S1.ptr, 4 * HV * S * S);
        mtl::Dispatch dq;
        dq.p = &p_gdnc;
        dq.args = {mtl::Arg::B(conv), mtl::Arg::B(alpha), mtl::Arg::B(dtb), mtl::Arg::B(ssa), mtl::Arg::B(beta),
                   mtl::Arg::B(S1), mtl::Arg::B(out1), mtl::Arg::V(HK), mtl::Arg::V(S), mtl::Arg::V(HV),
                   mtl::Arg::V(C), mtl::Arg::V(eps), mtl::Arg::V(T)};
        dq.grid[0] = HV;
        dq.tg[0] = 128;
        mtl::Dispatch d1;
        d1.p = &p_prep;
        d1.args = {mtl::Arg::B(conv), mtl::Arg::B(alpha), mtl::Arg::B(dtb), mtl::Arg::B(ssa), mtl::Arg::B(beta),
                   mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::V(HK),
                   mtl::Arg::V(S), mtl::Arg::V(C), mtl::Arg::V(eps), mtl::Arg::V(T)};
        d1.grid[0] = HV;
        d1.grid[1] = T;
        d1.tg[0] = 128;
        d1.barrier = true;
        mtl::Dispatch d2;
        d2.p = &p_cp;
        d2.args = {mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(vc), mtl::Arg::B(gb), mtl::Arg::B(U), mtl::Arg::B(W),
                   mtl::Arg::B(lgo), mtl::Arg::V(HK), mtl::Arg::V(HV), mtl::Arg::V(T)};
        d2.grid[0] = HV;
        d2.grid[1] = NC;
        d2.tg[0] = 256;
        d2.barrier = true;
        mtl::Dispatch d3;
        d3.p = &p_cs;
        d3.args = {mtl::Arg::B(S2), mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(U), mtl::Arg::B(W), mtl::Arg::B(lgo),
                   mtl::Arg::B(delta), mtl::Arg::B(out1), mtl::Arg::V(HK), mtl::Arg::V(HV), mtl::Arg::V(T)};
        d3.grid[0] = HV * 4;
        d3.tg[0] = 256;
        d3.barrier = true;
        mtl::Dispatch d4;
        d4.p = &p_co;
        d4.args = {mtl::Arg::B(qn), mtl::Arg::B(kn), mtl::Arg::B(delta), mtl::Arg::B(lgo), mtl::Arg::B(out1),
                   mtl::Arg::V(HK), mtl::Arg::V(HV), mtl::Arg::V(T)};
        d4.grid[0] = HV;
        d4.grid[1] = NC;
        d4.tg[0] = 128;
        d4.barrier = true;
        const double seq = timeit({dq}), prep = timeit({d1}), cp = timeit({d2}), cs = timeit({d3}), co = timeit({d4});
        std::printf("T %4d: k_gdn_chunk %8.3f ms | WY: prep %7.3f + cprep %7.3f + cstate %7.3f + cout %7.3f = %8.3f ms\n",
                    T, seq * 1e3, prep * 1e3, cp * 1e3, cs * 1e3, co * 1e3, (prep + cp + cs + co) * 1e3);
    }

    // the dense projections at the attn_qkv shape (n_out 10240 x n_in 2048): the released chunk GEMV vs
    // act + dq + gemm on the matrix units, Q4_K and Q2_K (UD-Q2_K_XL's dominant dense types)
    {
        const int NIN = 2048, NOUT = 10240;
        const int lib_mma = cx.compile({"quant.metal", "units.metal", "mma.metal", "q36_dense.metal"});
        mtl::Pipeline p_act16 = cx.pipeline(lib_mma, "act_f16"), p_g64 = cx.pipeline(lib_mma, "gemm_f16_64_64_4");
        mtl::Pipeline p_gq4 = cx.pipeline(lib, "k_gemv_c_T_Q4_K"), p_gq2 = cx.pipeline(lib, "k_gemv_c_T_Q2_K");
        struct DqArgsA {
            uint32_t n_in, n_out, row_bytes, perm;
        };
        struct GemmArgsA {
            int M, N, K, pad;
        };
        const int types[] = {12, 10};      // Q4_K, Q2_K
        const int rbs[] = {144, 84};       // bytes per 256-wide block
        for (int T : {512, 2048})
            for (int ti = 0; ti < 2; ++ti) {
                auto F = [&](size_t n) { return cx.alloc(4 * n); };
                mtl::Buffer W = cx.alloc((size_t) rbs[ti] * (NIN / 256) * NOUT);
                for (size_t i = 0; i < W.bytes / 4; ++i) W.as<float>()[i] = r.sym();
                sanitize(W);
                mtl::Buffer x = F((size_t) T * NIN), y1 = F((size_t) T * NOUT), y2 = F((size_t) T * NOUT);
                mtl::Buffer xh = cx.alloc(2 * (size_t) T * NIN), wh = cx.alloc(2 * (size_t) NOUT * NIN);
                GemvParam p{(unsigned long) rbs[ti] * (NIN / 256), 0, NIN, NOUT, (unsigned long) NIN};
                mtl::Dispatch dg;
                dg.p = ti == 0 ? &p_gq4 : &p_gq2;
                dg.args = {mtl::Arg::B(W), mtl::Arg::B(x), mtl::Arg::B(y1), mtl::Arg::V(p), mtl::Arg::V(T),
                           mtl::Arg::V(0), mtl::Arg::V(T)};
                dg.grid[0] = (NOUT + 7) / 8;
                dg.grid[1] = 1;
                dg.tg[0] = 256;
                mtl::Dispatch d0;
                d0.p = &p_act16;
                d0.args = {mtl::Arg::B(x), mtl::Arg::B(xh), mtl::Arg::V((uint32_t) (T * NIN))};
                d0.grid[0] = ((size_t) T * NIN + 255) / 256;
                d0.tg[0] = 256;
                d0.barrier = true;
                DqArgsA da{(uint32_t) NIN, (uint32_t) NOUT, (uint32_t) (rbs[ti] * (NIN / 256)), 0};
                mtl::Pipeline p_dq_t = cx.pipeline(lib_mma, ("dq_f16_" + std::to_string(types[ti])).c_str());
                mtl::Dispatch d1;
                d1.p = &p_dq_t;
                d1.args = {mtl::Arg::B(W), mtl::Arg::B(wh), mtl::Arg::V(da)};
                d1.grid[0] = (NIN / 32 + 63) / 64;
                d1.grid[1] = NOUT;
                d1.tg[0] = 64;
                d1.barrier = true;
                GemmArgsA ga{T, NOUT, NIN, 0};
                mtl::Dispatch d2;
                d2.p = &p_g64;
                d2.args = {mtl::Arg::B(xh), mtl::Arg::B(wh), mtl::Arg::B(y2), mtl::Arg::V(ga)};
                d2.grid[0] = (NOUT + 63) / 64;
                d2.grid[1] = (T + 63) / 64;
                d2.tg[0] = 128;
                d2.barrier = true;
                const double seq = timeit({dg}), act = timeit({d0}), dqt = timeit({d1}), gmt = timeit({d2});
                std::printf("T %4d dense %s: k_gemv_c %8.3f ms | MMA: act %6.3f + dq %6.3f + gemm %7.3f = %8.3f ms\n", T,
                            ti == 0 ? "q4_K" : "q2_K", seq * 1e3, act * 1e3, dqt * 1e3, gmt * 1e3,
                            (act + dqt + gmt) * 1e3);
            }
    }
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        run_bench();
        return 0;
    }
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "q36.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    const float eps = 1e-6f;

    // STRATA_Q36M_KV_F16=1: the attention stages run against the STRATA_KV_F16 lib's kernels (the same
    // seven, half caches) - stages 12-13 adapt their cache buffers to 2 B/elem, and the f16-only cross
    // stage at the end proves both libs bit-identical on half-exact values.  Default mode is untouched.
    const char* kf_env = std::getenv("STRATA_Q36M_KV_F16");
    const bool kf = kf_env && *kf_env && std::atoi(kf_env) != 0;
    int lib_f16 = -1;
    if (kf) {
        lib_f16 = cx.compile({"quant.metal", "q36.metal"}, "#define STRATA_KV_F16 1\n");
        std::printf("kv f16 mode: the STRATA_KV_F16 lib is %d; the attention stages route from it\n", lib_f16);
    }

    auto pipe = [&](const char* n) { return cx.pipeline(lib, n); };
    // the attention stages' pipelines come from the f16 lib in kv f16 mode; p_sk_merge stays on the
    // default lib (it reads the float partials, not K/V)
    auto pipe_kv = [&](const char* n) { return kf ? cx.pipeline(lib_f16, n) : cx.pipeline(lib, n); };
    mtl::Pipeline p_row = pipe("k_row_T_Q8_0"), p_emb = pipe("k_emb_chunk_T_Q8_0");
    mtl::Pipeline p_g_f32 = pipe("k_gemv_T_F32"), p_g_q8 = pipe("k_gemv_T_Q8_0"), p_g_q4k = pipe("k_gemv_T_Q4_K");
    mtl::Pipeline p_g_q5k = pipe("k_gemv_T_Q5_K"), p_g_q6k = pipe("k_gemv_T_Q6_K");
    mtl::Pipeline p_gc_f32 = pipe("k_gemv_c_T_F32"), p_gc_q8 = pipe("k_gemv_c_T_Q8_0"), p_gc_q4k = pipe("k_gemv_c_T_Q4_K");
    mtl::Pipeline p_gc_q5k = pipe("k_gemv_c_T_Q5_K"), p_gc_q6k = pipe("k_gemv_c_T_Q6_K");
    mtl::Pipeline p_gc_q2k = pipe("k_gemv_c_T_Q2_K"), p_gc_iq2 = pipe("k_gemv_c_T_IQ2_XS"),
                  p_gc_iq3 = pipe("k_gemv_c_T_IQ3_XXS");
    mtl::Pipeline p_lpr8 = pipe("k_gemv_lpr8_T_Q4_K");
    mtl::Pipeline p_topk = pipe("k_topk_router"), p_route = pipe("k_route_chunk");
    mtl::Pipeline p_zero = pipe("k_zero"), p_hist = pipe("k_hist"), p_off = pipe("k_exp_off"), p_place = pipe("k_place");
    mtl::Pipeline p_expc = pipe("k_expc_T_Q4_K");
    mtl::Pipeline p_expc_iq2 = pipe("k_expc_T_IQ2_XS"), p_expc_iq3 = pipe("k_expc_T_IQ3_XXS");
    mtl::Pipeline p_g_iq2 = pipe("k_gemv_T_IQ2_XS"), p_g_iq3 = pipe("k_gemv_T_IQ3_XXS");
    mtl::Pipeline p_lpr8_iq2 = pipe("k_gemv_lpr8_T_IQ2_XS"), p_gather = pipe("k_gather_t");
    mtl::Pipeline p_swiglu = pipe("k_swiglu");
    mtl::Pipeline p_rms = pipe("k_rms");
    mtl::Pipeline p_tables = pipe("k_tables");
    mtl::Pipeline p_conv = pipe("k_conv"), p_convc = pipe("k_conv_chunk");
    mtl::Pipeline p_gdn = pipe("k_gdn"), p_gdnc = pipe("k_gdn_chunk");
    mtl::Pipeline p_gdngb = pipe("k_gdn_gb");
    mtl::Pipeline p_prep_wy = pipe("k_gdn_prep_c"), p_cprep = pipe("k_gdn_cprep_c"), p_cstate = pipe("k_gdn_cstate_c"),
                  p_cout = pipe("k_gdn_cout_c");
    mtl::Pipeline p_gn = pipe("k_gated_norm"), p_gnc = pipe("k_gated_norm_chunk");
    mtl::Pipeline p_sum = pipe("k_moe_sum"), p_sumc = pipe("k_moe_sum_chunk");
    mtl::Pipeline p_prep = pipe_kv("k_attn_prep"), p_prep_c = pipe_kv("k_attn_prep_chunk");
    mtl::Pipeline p_sk_part = pipe_kv("k_attn_sk_part"), p_sk_merge = pipe("k_attn_sk_merge");

    ref::Rng r(20260717);
    mtl::Buffer dummy_ids = cx.alloc(sizeof(int));
    *(int*) dummy_ids.ptr = 0;

    // the chunk's per-token rows every stage shares
    mtl::Buffer h_all = cx.alloc(4 * T * N), router_all = cx.alloc(4 * T * NEXP);
    for (int i = 0; i < T * N; ++i) h_all.as<float>()[i] = r.sym();
    for (int i = 0; i < T * NEXP; ++i) router_all.as<float>()[i] = r.sym() * 4.0f;

    // ---- 1. embedding rows (Q8_0) --------------------------------------------------------------------------
    {
        const int VOC = 32;
        const uint64_t RB = (uint64_t) (N / 32) * 34;
        mtl::Buffer W = cx.alloc(VOC * RB);
        for (size_t i = 0; i < VOC * RB / 4; ++i) W.as<float>()[i] = r.sym();
        sanitize(W);
        int toks[T];
        for (int& t : toks) t = (int) (r.uni() * VOC);
        mtl::Buffer toks_b = cx.alloc(sizeof(toks)), y1 = cx.alloc(4 * T * N), y2 = cx.alloc(4 * T * N);
        std::memcpy(toks_b.ptr, toks, sizeof(toks));
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            mtl::Dispatch d;
            d.p = &p_row;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(y1, 4 * (size_t) t * N), mtl::Arg::V(RB), mtl::Arg::V(toks[t]),
                      mtl::Arg::V(N)};
            d.grid[0] = 16;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        mtl::Dispatch d;
        d.p = &p_emb;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(toks_b), mtl::Arg::B(y2), mtl::Arg::V(RB), mtl::Arg::V(N), mtl::Arg::V(N)};
        d.grid[0] = T;
        d.tg[0] = 128;
        cx.run_graph({d});
        check("embed", y1.ptr, y2.ptr, 4 * T * N);
    }

    // ---- 2. dense gemv: the decode kernel per token vs the chunk kernel (tpb = T and tpb = 2) ---------------
    {
        const int NOUT = 64;
        const uint64_t RB = 4ull * N;
        mtl::Buffer W = cx.alloc(RB * NOUT);
        for (size_t i = 0; i < NOUT * N; ++i) W.as<float>()[i] = r.sym();
        sanitize(W);
        mtl::Buffer y1 = cx.alloc(4 * T * NOUT), y2 = cx.alloc(4 * T * NOUT);
        for (int pass = 0; pass < 2; ++pass) {
            const int tpb = pass == 0 ? T : 2;
            std::vector<mtl::Dispatch> ds;
            for (int t = 0; t < T; ++t) {
                GemvParam p{RB, 0, N, NOUT, 0};
                mtl::Dispatch d;
                d.p = &p_g_f32;
                d.args = {mtl::Arg::B(W), mtl::Arg::B(h_all, 4 * (size_t) t * N), mtl::Arg::B(y1, 4 * (size_t) t * NOUT),
                          mtl::Arg::B(dummy_ids), mtl::Arg::V(p)};
                d.grid[0] = (NOUT + 7) / 8;
                d.grid[1] = 1;
                d.tg[0] = 256;
                d.barrier = true;
                ds.push_back(d);
            }
            cx.run_graph(ds);
            GemvParam p{RB, 0, N, NOUT, (unsigned long) N};
            mtl::Dispatch d;
            d.p = &p_gc_f32;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(h_all), mtl::Arg::B(y2), mtl::Arg::V(p), mtl::Arg::V(tpb),
                      mtl::Arg::V(0), mtl::Arg::V(T)};
            d.grid[0] = (NOUT + 7) / 8;
            d.grid[1] = (T + tpb - 1) / tpb;
            d.tg[0] = 256;
            cx.run_graph({d});
            char name[32];
            std::snprintf(name, sizeof(name), "gemv f32 tpb=%d", tpb);
            check(name, y1.ptr, y2.ptr, 4 * T * NOUT);
        }
    }

    // ---- 3. the same for Q8_0 ------------------------------------------------------------------------------
    {
        const int NOUT = 64;
        const uint64_t RB = (uint64_t) (N / 32) * 34;
        mtl::Buffer W = cx.alloc(RB * NOUT);
        for (size_t i = 0; i < W.bytes / 4; ++i) W.as<float>()[i] = r.sym();
        sanitize(W);
        mtl::Buffer y1 = cx.alloc(4 * T * NOUT), y2 = cx.alloc(4 * T * NOUT);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            GemvParam p{RB, 0, N, NOUT, 0};
            mtl::Dispatch d;
            d.p = &p_g_q8;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(h_all, 4 * (size_t) t * N), mtl::Arg::B(y1, 4 * (size_t) t * NOUT),
                      mtl::Arg::B(dummy_ids), mtl::Arg::V(p)};
            d.grid[0] = (NOUT + 7) / 8;
            d.tg[0] = 256;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        GemvParam p{RB, 0, N, NOUT, (unsigned long) N};
        mtl::Dispatch d;
        d.p = &p_gc_q8;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(h_all), mtl::Arg::B(y2), mtl::Arg::V(p), mtl::Arg::V(T),
                  mtl::Arg::V(0), mtl::Arg::V(T)};
        d.grid[0] = (NOUT + 7) / 8;
        d.grid[1] = 1;
        d.tg[0] = 256;
        cx.run_graph({d});
        check("gemv q8_0", y1.ptr, y2.ptr, 4 * T * NOUT);
    }

    // ---- 4. Q4_K (n_in = 512: two 256 blocks) --------------------------------------------------------------
    {
        const int NIN = 512, NOUT = 64;
        const uint64_t RB = 2ull * 144;
        mtl::Buffer hin = cx.alloc(4 * T * NIN);
        for (int i = 0; i < T * NIN; ++i) hin.as<float>()[i] = r.sym();
        mtl::Buffer W = cx.alloc(RB * NOUT);
        for (size_t i = 0; i < W.bytes / 4; ++i) W.as<float>()[i] = r.sym();
        sanitize(W);
        mtl::Buffer y1 = cx.alloc(4 * T * NOUT), y2 = cx.alloc(4 * T * NOUT);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            GemvParam p{RB, 0, NIN, NOUT, 0};
            mtl::Dispatch d;
            d.p = &p_g_q4k;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(hin, 4 * (size_t) t * NIN), mtl::Arg::B(y1, 4 * (size_t) t * NOUT),
                      mtl::Arg::B(dummy_ids), mtl::Arg::V(p)};
            d.grid[0] = (NOUT + 7) / 8;
            d.tg[0] = 256;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        GemvParam p{RB, 0, NIN, NOUT, (unsigned long) NIN};
        mtl::Dispatch d;
        d.p = &p_gc_q4k;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(hin), mtl::Arg::B(y2), mtl::Arg::V(p), mtl::Arg::V(T),
                  mtl::Arg::V(0), mtl::Arg::V(T)};
        d.grid[0] = (NOUT + 7) / 8;
        d.grid[1] = 1;
        d.tg[0] = 256;
        cx.run_graph({d});
        check("gemv q4_K", y1.ptr, y2.ptr, 4 * T * NOUT);
    }

    // ---- 5. the n_in = 2048 lpr8 gemv: the engine's chunk path dispatches the DECODE kernel itself with
    // grid[1] = nt (a token-loop twin differed by ~1 ulp: the serial deq path contracts fma differently
    // inside a loop).  This stage guards that path: decode kernel per token vs decode kernel, grid[1] = T.
    {
        const int NIN = 2048, NOUT = 64;
        const uint64_t RB = 8ull * 144;
        mtl::Buffer hin = cx.alloc(4 * T * NIN);
        for (int i = 0; i < T * NIN; ++i) hin.as<float>()[i] = r.sym();
        mtl::Buffer W = cx.alloc(RB * NOUT);
        for (size_t i = 0; i < W.bytes / 4; ++i) W.as<float>()[i] = r.sym();
        sanitize(W);
        mtl::Buffer zeros = cx.alloc(4 * T);
        mtl::Buffer y1 = cx.alloc(4 * T * NOUT), y2 = cx.alloc(4 * T * NOUT);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            GemvParam p{RB, 0, NIN, NOUT, 0};
            mtl::Dispatch d;
            d.p = &p_lpr8;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(hin, 4 * (size_t) t * NIN), mtl::Arg::B(y1, 4 * (size_t) t * NOUT),
                      mtl::Arg::B(dummy_ids), mtl::Arg::V(p)};
            d.grid[0] = (NOUT + 15) / 16;
            d.grid[1] = 1;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        GemvParam p{RB, 0, NIN, NOUT, (unsigned long) NIN};
        mtl::Dispatch d;
        d.p = &p_lpr8;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(hin), mtl::Arg::B(y2), mtl::Arg::B(zeros), mtl::Arg::V(p)};
        d.grid[0] = (NOUT + 15) / 16;
        d.grid[1] = T;
        d.tg[0] = 128;
        cx.run_graph({d});
        check("lpr8 grid[1]=T", y1.ptr, y2.ptr, 4 * T * NOUT);
    }

    // ---- 4b. Q5_K and Q6_K through k_gemv_c: the serial deq_dot path (the Q4_K/F32 stages above went
    // through the float4-dot specializations or a load-only deq - these are the fma-ambiguous ones)
    for (int which = 0; which < 2; ++which) {
        const int NIN = 512, NOUT = 64;
        const uint64_t RB = which == 0 ? 2ull * 176 : 2ull * 210;   // Q5_K / Q6_K rows of 512
        mtl::Buffer hin = cx.alloc(4 * T * NIN);
        for (int i = 0; i < T * NIN; ++i) hin.as<float>()[i] = r.sym();
        mtl::Buffer W = cx.alloc(RB * NOUT);
        for (size_t i = 0; i < W.bytes; ++i) W.as<unsigned char>()[i] = (unsigned char) (r.next() & 255);
        sanitize(W);
        mtl::Buffer y1 = cx.alloc(4 * T * NOUT), y2 = cx.alloc(4 * T * NOUT);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            GemvParam p{RB, 0, NIN, NOUT, 0};
            mtl::Dispatch d;
            d.p = which == 0 ? &p_g_q5k : &p_g_q6k;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(hin, 4 * (size_t) t * NIN), mtl::Arg::B(y1, 4 * (size_t) t * NOUT),
                      mtl::Arg::B(dummy_ids), mtl::Arg::V(p)};
            d.grid[0] = (NOUT + 7) / 8;
            d.tg[0] = 256;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        GemvParam p{RB, 0, NIN, NOUT, (unsigned long) NIN};
        mtl::Dispatch d;
        d.p = which == 0 ? &p_gc_q5k : &p_gc_q6k;
        d.args = {mtl::Arg::B(W), mtl::Arg::B(hin), mtl::Arg::B(y2), mtl::Arg::V(p), mtl::Arg::V(T),
                  mtl::Arg::V(0), mtl::Arg::V(T)};
        d.grid[0] = (NOUT + 7) / 8;
        d.grid[1] = 1;
        d.tg[0] = 256;
        cx.run_graph({d});
        check(which == 0 ? "gemv q5_K" : "gemv q6_K", y1.ptr, y2.ptr, 4 * T * NOUT);
    }

    // ---- 6. top-k routing ----------------------------------------------------------------------------------
    mtl::Buffer ids1 = cx.alloc(4 * T * K), wts1 = cx.alloc(4 * T * K);
    mtl::Buffer ids2 = cx.alloc(4 * T * K), wts2 = cx.alloc(4 * T * K);
    {
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            mtl::Dispatch d;
            d.p = &p_topk;
            d.args = {mtl::Arg::B(router_all, 4 * (size_t) t * NEXP), mtl::Arg::B(ids1, 4 * (size_t) t * K),
                      mtl::Arg::B(wts1, 4 * (size_t) t * K), mtl::Arg::V(NEXP), mtl::Arg::V(K)};
            d.grid[0] = 1;
            d.tg[0] = 32;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        mtl::Dispatch d;
        d.p = &p_route;
        d.args = {mtl::Arg::B(router_all), mtl::Arg::B(ids2), mtl::Arg::B(wts2), mtl::Arg::V(NEXP), mtl::Arg::V(K)};
        d.grid[0] = T;
        d.tg[0] = 256;
        cx.run_graph({d});
        check("route ids", ids1.ptr, ids2.ptr, 4 * T * K);
        check("route wts", wts1.ptr, wts2.ptr, 4 * T * K);
    }

    // ---- 7. the entry tables + grouped expert gemv (Q4_K) vs per-token gemv with ids ------------------------
    mtl::Buffer pos = cx.alloc(4 * T * K);
    {
        const uint64_t RB = (uint64_t) (N / 256) * 144;   // Q4_K rows of n_in = 256: one 144-byte block
        const uint64_t EB = RB * FF;
        mtl::Buffer W = cx.alloc(NEXP * EB);
        for (size_t i = 0; i < W.bytes / 4; ++i) W.as<float>()[i] = r.sym();
        sanitize(W);
        mtl::Buffer eg1 = cx.alloc(4 * T * K * FF), eg2 = cx.alloc(4 * T * K * FF);
        mtl::Buffer scr = cx.alloc(4 * 3 * 256), srt_gu = cx.alloc(4 * T * K), srt_d = cx.alloc(4 * T * K);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            GemvParam p{RB, EB, N, FF, 0};
            mtl::Dispatch d;
            d.p = &p_g_q4k;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(h_all, 4 * (size_t) t * N), mtl::Arg::B(eg1, 4 * (size_t) t * K * FF),
                      mtl::Arg::B(ids1, 4 * (size_t) t * K), mtl::Arg::V(p)};
            d.grid[0] = (FF + 7) / 8;
            d.grid[1] = K;
            d.tg[0] = 256;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);

        mtl::Dispatch dz;
        dz.p = &p_zero;
        dz.args = {mtl::Arg::B(scr), mtl::Arg::V(3 * 256)};
        dz.grid[0] = (3 * 256 + 127) / 128;
        dz.tg[0] = 128;
        dz.barrier = true;
        mtl::Dispatch dh;
        dh.p = &p_hist;
        dh.args = {mtl::Arg::B(ids2), mtl::Arg::B(scr), mtl::Arg::V(T * K)};
        dh.grid[0] = (T * K + 255) / 256;
        dh.tg[0] = 256;
        dh.barrier = true;
        mtl::Dispatch doff;
        doff.p = &p_off;
        doff.args = {mtl::Arg::B(scr), mtl::Arg::B(scr, 512 * 4), mtl::Arg::V(NEXP)};
        doff.grid[0] = 1;
        doff.tg[0] = 1;
        doff.barrier = true;
        mtl::Dispatch dp;
        dp.p = &p_place;
        dp.args = {mtl::Arg::B(ids2), mtl::Arg::B(scr, 256 * 4), mtl::Arg::B(scr, 512 * 4), mtl::Arg::B(srt_gu),
                   mtl::Arg::B(srt_d), mtl::Arg::B(pos), mtl::Arg::V(K), mtl::Arg::V(T * K)};
        dp.grid[0] = (T * K + 255) / 256;
        dp.tg[0] = 256;
        dp.barrier = true;
        cx.run_graph({dz, dh, doff, dp});

        GemvParam p{RB, EB, N, FF, (unsigned long) N};
        mtl::Dispatch de;
        de.p = &p_expc;
        de.args = {mtl::Arg::B(W), mtl::Arg::B(h_all), mtl::Arg::B(srt_gu), mtl::Arg::B(scr, 512 * 4),
                   mtl::Arg::B(scr), mtl::Arg::B(eg2), mtl::Arg::V(p)};
        de.grid[0] = (FF + 7) / 8;
        de.grid[1] = NEXP;
        de.tg[0] = 256;
        cx.run_graph({de});
        check_scattered("expert gemv", eg1.as<float>(), eg2.as<float>(), pos.as<int>());
    }

    // ---- 7b. the experts through k_expc with the SERIAL-deq types (IQ2_XS / IQ3_XXS: the engine's real
    // expert formats, whose deq_dot has no float4 specialization) vs the decode k_gemv per token
    for (int which = 0; which < 2; ++which) {
        const uint64_t RB = which == 0 ? 74ull : 98ull;   // IQ2_XS / IQ3_XXS rows of n_in = 256
        const uint64_t EB = RB * FF;
        mtl::Pipeline& pe = which == 0 ? p_expc_iq2 : p_expc_iq3;
        mtl::Pipeline& gd = which == 0 ? p_g_iq2 : p_g_iq3;
        mtl::Buffer W = cx.alloc(NEXP * EB);
        for (size_t i = 0; i < W.bytes; ++i) W.as<unsigned char>()[i] = (unsigned char) (r.next() & 255);
        sanitize(W);
        mtl::Buffer eg1 = cx.alloc(4 * T * K * FF), eg2 = cx.alloc(4 * T * K * FF);
        mtl::Buffer scr = cx.alloc(4 * 3 * 256), srt_gu = cx.alloc(4 * T * K), srt_d = cx.alloc(4 * T * K);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            GemvParam p{RB, EB, N, FF, 0};
            mtl::Dispatch d;
            d.p = &gd;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(h_all, 4 * (size_t) t * N), mtl::Arg::B(eg1, 4 * (size_t) t * K * FF),
                      mtl::Arg::B(ids1, 4 * (size_t) t * K), mtl::Arg::V(p)};
            d.grid[0] = (FF + 7) / 8;
            d.grid[1] = K;
            d.tg[0] = 256;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);

        mtl::Dispatch dz;
        dz.p = &p_zero;
        dz.args = {mtl::Arg::B(scr), mtl::Arg::V(3 * 256)};
        dz.grid[0] = (3 * 256 + 127) / 128;
        dz.tg[0] = 128;
        dz.barrier = true;
        mtl::Dispatch dh;
        dh.p = &p_hist;
        dh.args = {mtl::Arg::B(ids2), mtl::Arg::B(scr), mtl::Arg::V(T * K)};
        dh.grid[0] = 1;
        dh.tg[0] = 256;
        dh.barrier = true;
        mtl::Dispatch doff;
        doff.p = &p_off;
        doff.args = {mtl::Arg::B(scr), mtl::Arg::B(scr, 512 * 4), mtl::Arg::V(NEXP)};
        doff.grid[0] = 1;
        doff.tg[0] = 1;
        doff.barrier = true;
        mtl::Dispatch dp;
        dp.p = &p_place;
        dp.args = {mtl::Arg::B(ids2), mtl::Arg::B(scr, 256 * 4), mtl::Arg::B(scr, 512 * 4), mtl::Arg::B(srt_gu),
                   mtl::Arg::B(srt_d), mtl::Arg::B(pos), mtl::Arg::V(K), mtl::Arg::V(T * K)};
        dp.grid[0] = 1;
        dp.tg[0] = 256;
        dp.barrier = true;
        cx.run_graph({dz, dh, doff, dp});

        GemvParam p{RB, EB, N, FF, (unsigned long) N};
        mtl::Dispatch de;
        de.p = &pe;
        de.args = {mtl::Arg::B(W), mtl::Arg::B(h_all), mtl::Arg::B(srt_gu), mtl::Arg::B(scr, 512 * 4),
                   mtl::Arg::B(scr), mtl::Arg::B(eg2), mtl::Arg::V(p)};
        de.grid[0] = (FF + 7) / 8;
        de.grid[1] = NEXP;
        de.tg[0] = 256;
        cx.run_graph({de});
        check_scattered(which == 0 ? "expert iq2_xs" : "expert iq3_xxs", eg1.as<float>(), eg2.as<float>(),
                        pos.as<int>());
    }

    // ---- 8. conv (4-tap) over the chunk vs per token; the state must match too ------------------------------
    mtl::Buffer qkv_all, conv_w, state1, state2, cout1, cout2;
    {
        qkv_all = cx.alloc(4 * T * C);
        conv_w = cx.alloc(4 * C * 4);
        state1 = cx.alloc(4 * 3 * C);
        state2 = cx.alloc(4 * 3 * C);
        cout1 = cx.alloc(4 * T * C);
        cout2 = cx.alloc(4 * T * C);
        for (int i = 0; i < T * C; ++i) qkv_all.as<float>()[i] = r.sym() * 2.0f;
        for (int i = 0; i < C * 4; ++i) conv_w.as<float>()[i] = r.sym() * 0.5f;
        for (int i = 0; i < 3 * C; ++i) state1.as<float>()[i] = r.sym() * 0.1f;
        std::memcpy(state2.ptr, state1.ptr, 4 * 3 * C);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            mtl::Dispatch d;
            d.p = &p_conv;
            d.args = {mtl::Arg::B(qkv_all, 4 * (size_t) t * C), mtl::Arg::B(state1), mtl::Arg::B(conv_w),
                      mtl::Arg::B(cout1, 4 * (size_t) t * C), mtl::Arg::V(C), mtl::Arg::V(4)};
            d.grid[0] = (C + 127) / 128;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        mtl::Dispatch d;
        d.p = &p_convc;
        d.args = {mtl::Arg::B(qkv_all), mtl::Arg::B(state2), mtl::Arg::B(conv_w), mtl::Arg::B(cout2),
                  mtl::Arg::V(C), mtl::Arg::V(4), mtl::Arg::V(T)};
        d.grid[0] = (C + 127) / 128;
        d.grid[1] = (T + 15) / 16;
        d.tg[0] = 128;
        cx.run_graph({d});
        check("conv out", cout1.ptr, cout2.ptr, 4 * T * C);
        check("conv state", state1.ptr, state2.ptr, 4 * 3 * C);
    }

    // ---- 9. the delta rule over the chunk vs per token (input = the conv outputs); S must match -------------
    mtl::Buffer alpha_all = cx.alloc(4 * T * HV), beta_all = cx.alloc(4 * T * HV);
    mtl::Buffer dtb = cx.alloc(4 * HV), ssa = cx.alloc(4 * HV);
    mtl::Buffer S1 = cx.alloc(4 * HV * S * S), S2 = cx.alloc(4 * HV * S * S);
    mtl::Buffer gout1 = cx.alloc(4 * T * ZV), gout2 = cx.alloc(4 * T * ZV);
    for (int i = 0; i < T * HV; ++i) {
        alpha_all.as<float>()[i] = r.sym() * 2.0f;
        beta_all.as<float>()[i] = r.sym() * 2.0f;
    }
    for (int i = 0; i < HV; ++i) {
        dtb.as<float>()[i] = r.sym();
        ssa.as<float>()[i] = -std::exp(r.sym());
    }
    for (int i = 0; i < HV * S * S; ++i) S1.as<float>()[i] = r.sym() * 0.1f;
    std::memcpy(S2.ptr, S1.ptr, 4 * HV * S * S);
    mtl::Buffer S0 = cx.alloc(4 * HV * S * S);   // the pristine initial state, for the WY stage below
    std::memcpy(S0.ptr, S1.ptr, 4 * HV * S * S);
    {
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            mtl::Dispatch d;
            d.p = &p_gdn;
            d.args = {mtl::Arg::B(cout1, 4 * (size_t) t * C), mtl::Arg::B(alpha_all, 4 * (size_t) t * HV),
                      mtl::Arg::B(dtb), mtl::Arg::B(ssa), mtl::Arg::B(beta_all, 4 * (size_t) t * HV),
                      mtl::Arg::B(S1), mtl::Arg::B(gout1, 4 * (size_t) t * ZV), mtl::Arg::V(HK), mtl::Arg::V(S),
                      mtl::Arg::V(eps)};
            d.grid[0] = HV;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        mtl::Dispatch d;
        d.p = &p_gdnc;
        d.args = {mtl::Arg::B(cout2), mtl::Arg::B(alpha_all), mtl::Arg::B(dtb), mtl::Arg::B(ssa),
                  mtl::Arg::B(beta_all), mtl::Arg::B(S2), mtl::Arg::B(gout2), mtl::Arg::V(HK), mtl::Arg::V(S),
                  mtl::Arg::V(HV), mtl::Arg::V(C), mtl::Arg::V(eps), mtl::Arg::V(T)};
        d.grid[0] = HV;
        d.tg[0] = 128;
        cx.run_graph({d});
        check("gdn out", gout1.ptr, gout2.ptr, 4 * T * ZV);
        check("gdn S", S1.ptr, S2.ptr, 4 * HV * S * S);
    }

    // ---- 9b. the WY form (k_gdn_prep_c + k_gdn_cprep_c/cstate_c/cout_c, STRATA_Q36M_GDN_WY=1's kernels)
    // vs k_gdn_chunk on the same chunk inputs and initial state: the reassociated scan matches to
    // rounding, not bits (y and the final state rel < 1e-5, gdn_prompt_parity's scheme).  The prep's v
    // copy and beta gate are bit-exact; the q/k norms share k_gdn_chunk's block_sum and formulas by
    // construction (the same 128-thread reduction per (token, head)) - the composite bound guards them.
    {
        const int NC = (T + 31) / 32;
        mtl::Buffer wy_qn = cx.alloc(4 * T * HK * S), wy_kn = cx.alloc(4 * T * HK * S),
                    wy_vc = cx.alloc(4 * T * HV * S), wy_gb = cx.alloc(8 * T * HV),
                    wy_U = cx.alloc(4 * HV * NC * 32 * S), wy_W = cx.alloc(4 * HV * NC * 32 * S),
                    wy_lgo = cx.alloc(4 * HV * NC * 32), wy_delta = cx.alloc(4 * HV * NC * 32 * S);
        mtl::Buffer S3 = cx.alloc(4 * HV * S * S), gout3 = cx.alloc(4 * T * ZV);
        std::memcpy(S3.ptr, S0.ptr, 4 * HV * S * S);   // the same initial state stage 9's two runs started from
        mtl::Dispatch d1;
        d1.p = &p_prep_wy;
        d1.args = {mtl::Arg::B(cout2), mtl::Arg::B(alpha_all), mtl::Arg::B(dtb), mtl::Arg::B(ssa),
                   mtl::Arg::B(beta_all), mtl::Arg::B(wy_qn), mtl::Arg::B(wy_kn), mtl::Arg::B(wy_vc),
                   mtl::Arg::B(wy_gb), mtl::Arg::V(HK), mtl::Arg::V(S), mtl::Arg::V(C), mtl::Arg::V(eps),
                   mtl::Arg::V(T)};
        d1.grid[0] = HV;
        d1.grid[1] = T;
        d1.tg[0] = 128;
        d1.barrier = true;
        mtl::Dispatch d2;
        d2.p = &p_cprep;
        d2.args = {mtl::Arg::B(wy_qn), mtl::Arg::B(wy_kn), mtl::Arg::B(wy_vc), mtl::Arg::B(wy_gb),
                   mtl::Arg::B(wy_U), mtl::Arg::B(wy_W), mtl::Arg::B(wy_lgo), mtl::Arg::V(HK), mtl::Arg::V(HV),
                   mtl::Arg::V(T)};
        d2.grid[0] = HV;
        d2.grid[1] = NC;
        d2.tg[0] = 256;
        d2.barrier = true;
        mtl::Dispatch d3;
        d3.p = &p_cstate;
        d3.args = {mtl::Arg::B(S3), mtl::Arg::B(wy_qn), mtl::Arg::B(wy_kn), mtl::Arg::B(wy_U), mtl::Arg::B(wy_W),
                   mtl::Arg::B(wy_lgo), mtl::Arg::B(wy_delta), mtl::Arg::B(gout3), mtl::Arg::V(HK), mtl::Arg::V(HV),
                   mtl::Arg::V(T)};
        d3.grid[0] = HV * 4;
        d3.tg[0] = 256;
        d3.barrier = true;
        mtl::Dispatch d4;
        d4.p = &p_cout;
        d4.args = {mtl::Arg::B(wy_qn), mtl::Arg::B(wy_kn), mtl::Arg::B(wy_delta), mtl::Arg::B(wy_lgo),
                   mtl::Arg::B(gout3), mtl::Arg::V(HK), mtl::Arg::V(HV), mtl::Arg::V(T)};
        d4.grid[0] = HV;
        d4.grid[1] = NC;
        d4.tg[0] = 128;
        d4.barrier = true;
        cx.run_graph({d1, d2, d3, d4});
        bool v_ok = true;
        for (int t = 0; t < T && v_ok; ++t)
            for (int i = 0; i < HV * S && v_ok; ++i)
                v_ok = feq(cout2.as<float>()[(size_t) t * C + 2 * HK * S + i],
                           wy_vc.as<float>()[(size_t) t * HV * S + i]);
        if (!v_ok) {
            std::printf("  %-14s FAIL: the v copy differs\n", "wy prep v");
            ++g_fails;
        } else
            std::printf("  %-14s OK\n", "wy prep v");
        {   // the beta half of gb vs k_gdn_gb's transform: the same sigm on the same input, bit for bit
            mtl::Buffer ab = cx.alloc(4 * T * HV), bb = cx.alloc(4 * T * HV);
            std::memcpy(ab.ptr, alpha_all.ptr, 4 * T * HV);
            std::memcpy(bb.ptr, beta_all.ptr, 4 * T * HV);
            std::vector<mtl::Dispatch> ds;
            for (int t = 0; t < T; ++t) {
                mtl::Dispatch d;
                d.p = &p_gdngb;
                d.args = {mtl::Arg::B(ab, 4 * (size_t) t * HV), mtl::Arg::B(dtb), mtl::Arg::B(ssa),
                          mtl::Arg::B(bb, 4 * (size_t) t * HV), mtl::Arg::V(HV)};
                d.grid[0] = HV;
                d.barrier = true;
                ds.push_back(d);
            }
            cx.run_graph(ds);
            bool b_ok = true;
            for (int i = 0; i < T * HV && b_ok; ++i) b_ok = feq(wy_gb.as<float>()[2 * i + 1], bb.as<float>()[i]);
            if (!b_ok) {
                std::printf("  %-14s FAIL: the beta gate differs from k_gdn_gb's\n", "wy prep beta");
                ++g_fails;
            } else
                std::printf("  %-14s OK\n", "wy prep beta");
        }
        check_rel("wy out", gout3.ptr, gout2.ptr, 4 * T * ZV, 1e-5);
        check_rel("wy S", S3.ptr, S2.ptr, 4 * HV * S * S, 1e-5);
    }

    // ---- 9c. the WY form over several chunks (TW = 40: one full 32-token chunk + one ragged) vs
    // k_gdn_chunk on the same inputs - the multi-chunk walk and the ragged tail the engine's last
    // chunk of a prompt hits whenever nt is not a multiple of 32
    {
        const int TW = 40, NC = (TW + 31) / 32;
        mtl::Buffer conv_w = cx.alloc(4 * TW * C), al_w = cx.alloc(4 * TW * HV), be_w = cx.alloc(4 * TW * HV);
        mtl::Buffer S4 = cx.alloc(4 * HV * S * S), S5 = cx.alloc(4 * HV * S * S);
        mtl::Buffer go4 = cx.alloc(4 * TW * ZV), go5 = cx.alloc(4 * TW * ZV);
        mtl::Buffer wy_qn = cx.alloc(4 * TW * HK * S), wy_kn = cx.alloc(4 * TW * HK * S),
                    wy_vc = cx.alloc(4 * TW * HV * S), wy_gb = cx.alloc(8 * TW * HV),
                    wy_U = cx.alloc(4 * HV * NC * 32 * S), wy_W = cx.alloc(4 * HV * NC * 32 * S),
                    wy_lgo = cx.alloc(4 * HV * NC * 32), wy_delta = cx.alloc(4 * HV * NC * 32 * S);
        for (int i = 0; i < TW * C; ++i) conv_w.as<float>()[i] = r.sym() * 2.0f;
        for (int i = 0; i < TW * HV; ++i) {
            al_w.as<float>()[i] = r.sym() * 2.0f;
            be_w.as<float>()[i] = r.sym() * 2.0f;
        }
        for (int i = 0; i < HV * S * S; ++i) S4.as<float>()[i] = r.sym() * 0.1f;
        std::memcpy(S5.ptr, S4.ptr, 4 * HV * S * S);
        mtl::Dispatch dref;
        dref.p = &p_gdnc;
        dref.args = {mtl::Arg::B(conv_w), mtl::Arg::B(al_w), mtl::Arg::B(dtb), mtl::Arg::B(ssa), mtl::Arg::B(be_w),
                     mtl::Arg::B(S5), mtl::Arg::B(go5), mtl::Arg::V(HK), mtl::Arg::V(S), mtl::Arg::V(HV),
                     mtl::Arg::V(C), mtl::Arg::V(eps), mtl::Arg::V(TW)};
        dref.grid[0] = HV;
        dref.tg[0] = 128;
        cx.run_graph({dref});
        mtl::Dispatch d1;
        d1.p = &p_prep_wy;
        d1.args = {mtl::Arg::B(conv_w), mtl::Arg::B(al_w), mtl::Arg::B(dtb), mtl::Arg::B(ssa), mtl::Arg::B(be_w),
                   mtl::Arg::B(wy_qn), mtl::Arg::B(wy_kn), mtl::Arg::B(wy_vc), mtl::Arg::B(wy_gb), mtl::Arg::V(HK),
                   mtl::Arg::V(S), mtl::Arg::V(C), mtl::Arg::V(eps), mtl::Arg::V(TW)};
        d1.grid[0] = HV;
        d1.grid[1] = TW;
        d1.tg[0] = 128;
        d1.barrier = true;
        mtl::Dispatch d2;
        d2.p = &p_cprep;
        d2.args = {mtl::Arg::B(wy_qn), mtl::Arg::B(wy_kn), mtl::Arg::B(wy_vc), mtl::Arg::B(wy_gb),
                   mtl::Arg::B(wy_U), mtl::Arg::B(wy_W), mtl::Arg::B(wy_lgo), mtl::Arg::V(HK), mtl::Arg::V(HV),
                   mtl::Arg::V(TW)};
        d2.grid[0] = HV;
        d2.grid[1] = NC;
        d2.tg[0] = 256;
        d2.barrier = true;
        mtl::Dispatch d3;
        d3.p = &p_cstate;
        d3.args = {mtl::Arg::B(S4), mtl::Arg::B(wy_qn), mtl::Arg::B(wy_kn), mtl::Arg::B(wy_U), mtl::Arg::B(wy_W),
                   mtl::Arg::B(wy_lgo), mtl::Arg::B(wy_delta), mtl::Arg::B(go4), mtl::Arg::V(HK), mtl::Arg::V(HV),
                   mtl::Arg::V(TW)};
        d3.grid[0] = HV * 4;
        d3.tg[0] = 256;
        d3.barrier = true;
        mtl::Dispatch d4;
        d4.p = &p_cout;
        d4.args = {mtl::Arg::B(wy_qn), mtl::Arg::B(wy_kn), mtl::Arg::B(wy_delta), mtl::Arg::B(wy_lgo),
                   mtl::Arg::B(go4), mtl::Arg::V(HK), mtl::Arg::V(HV), mtl::Arg::V(TW)};
        d4.grid[0] = HV;
        d4.grid[1] = NC;
        d4.tg[0] = 128;
        d4.barrier = true;
        cx.run_graph({d1, d2, d3, d4});
        check_rel("wy out 2c", go4.ptr, go5.ptr, 4 * TW * ZV, 1e-5);
        check_rel("wy S 2c", S4.ptr, S5.ptr, 4 * HV * S * S, 1e-5);
    }

    // ---- 9d. the MMA dense (q36_dense.metal, STRATA_Q36M_DENSE_MMA=1's kernels) vs k_gemv_c per type:
    // act_f16 + dq_f16 + gemm_f16 against the released chunk GEMV on the same weight and rows.  The
    // fp16 operands re-round (~5e-4 relative per element), so this compares to rounding (rel < 1e-3);
    // M = 8 rows inside a 64-row tile and NOUT = 96 inside two 64-wide tiles exercise the matmul2d
    // boundary clamp the donor's tensors carry (the reviewer's ragged-boundary note).  BF16 is not in
    // the list: its values round-trip exactly through fp16, so the F32 case covers that path.
    {
        const int lib_mma = cx.compile({"quant.metal", "units.metal", "mma.metal", "q36_dense.metal"});
        mtl::Pipeline p_act16 = cx.pipeline(lib_mma, "act_f16"), p_g64 = cx.pipeline(lib_mma, "gemm_f16_64_64_4");
        const int NOUT = 96;
        struct DqArgsA {
            uint32_t n_in, n_out, row_bytes, perm;
        };
        struct GemmArgsA {
            int M, N, K, pad;
        };
        const int types[] = {0, 12, 10, 17, 18, 14};   // F32, Q4_K, Q2_K, IQ2_XS, IQ3_XXS, Q6_K
        const int units[] = {16, 32, 32, 16, 16, 32};  // fmt<F>::UNIT (quant.metal)
        const int rbs[] = {4 * N, 144, 84, 74, 98, 210};   // one 256-wide block's bytes (N = 256)
        for (int ti = 0; ti < 6; ++ti) {
            const int ty = types[ti];
            mtl::Buffer W = cx.alloc((size_t) rbs[ti] * NOUT);
            if (ty == 0) {   // fp32 weights filled as floats: the byte-sanitize would denormal them past fp16
                for (size_t i = 0; i < W.bytes / 4; ++i) W.as<float>()[i] = r.sym() * 0.5f;
            } else {
                for (size_t i = 0; i < W.bytes / 4; ++i) W.as<float>()[i] = r.sym();
                sanitize(W);
            }
            mtl::Buffer y_ref = cx.alloc(4 * T * NOUT), y_mma = cx.alloc(4 * T * NOUT);
            mtl::Buffer xh = cx.alloc(2 * T * N), wh = cx.alloc(2 * NOUT * N);
            mtl::Pipeline& ref_p = ty == 0 ? p_gc_f32 : ty == 12 ? p_gc_q4k : ty == 10 ? p_gc_q2k
                                                                                 : ty == 17 ? p_gc_iq2
                                                                                 : ty == 18 ? p_gc_iq3 : p_gc_q6k;
            GemvParam p{(unsigned long) rbs[ti], 0, N, NOUT, (unsigned long) N};
            mtl::Dispatch dr;
            dr.p = &ref_p;
            dr.args = {mtl::Arg::B(W), mtl::Arg::B(h_all), mtl::Arg::B(y_ref), mtl::Arg::V(p), mtl::Arg::V(T),
                       mtl::Arg::V(0), mtl::Arg::V(T)};
            dr.grid[0] = (NOUT + 7) / 8;
            dr.grid[1] = 1;
            dr.tg[0] = 256;
            cx.run_graph({dr});
            mtl::Dispatch d0;
            d0.p = &p_act16;
            d0.args = {mtl::Arg::B(h_all), mtl::Arg::B(xh), mtl::Arg::V((uint32_t) (T * N))};
            d0.grid[0] = (T * N + 255) / 256;
            d0.tg[0] = 256;
            d0.barrier = true;
            DqArgsA da{(uint32_t) N, (uint32_t) NOUT, (uint32_t) rbs[ti], 0};
            mtl::Pipeline p_dq_t = cx.pipeline(lib_mma, ("dq_f16_" + std::to_string(ty)).c_str());
            mtl::Dispatch d1m;
            d1m.p = &p_dq_t;
            d1m.args = {mtl::Arg::B(W), mtl::Arg::B(wh), mtl::Arg::V(da)};
            d1m.grid[0] = (N / units[ti] + 63) / 64;
            d1m.grid[1] = NOUT;
            d1m.tg[0] = 64;
            d1m.barrier = true;
            GemmArgsA ga{T, NOUT, N, 0};
            mtl::Dispatch d2m;
            d2m.p = &p_g64;
            d2m.args = {mtl::Arg::B(xh), mtl::Arg::B(wh), mtl::Arg::B(y_mma), mtl::Arg::V(ga)};
            d2m.grid[0] = (NOUT + 63) / 64;
            d2m.grid[1] = (T + 63) / 64;
            d2m.tg[0] = 128;
            d2m.barrier = true;
            cx.run_graph({d0, d1m, d2m});
            char name[24];
            std::snprintf(name, sizeof(name), "mma %d", ty);
            check_rel(name, y_mma.ptr, y_ref.ptr, 4 * T * NOUT, 1e-3);
        }
    }

    // ---- 10. gated norm over the chunk vs per token ---------------------------------------------------------
    {
        mtl::Buffer w_norm = cx.alloc(4 * S), z_all = cx.alloc(4 * T * ZV);
        for (int i = 0; i < S; ++i) w_norm.as<float>()[i] = 1.0f + 0.2f * r.sym();
        for (int i = 0; i < T * ZV; ++i) z_all.as<float>()[i] = r.sym() * 2.0f;
        mtl::Buffer o1 = cx.alloc(4 * T * ZV), o2 = cx.alloc(4 * T * ZV);
        std::memcpy(o1.ptr, gout1.ptr, 4 * T * ZV);
        std::memcpy(o2.ptr, gout2.ptr, 4 * T * ZV);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            mtl::Dispatch d;
            d.p = &p_gn;
            d.args = {mtl::Arg::B(o1, 4 * (size_t) t * ZV), mtl::Arg::B(w_norm), mtl::Arg::B(z_all, 4 * (size_t) t * ZV),
                      mtl::Arg::V(S), mtl::Arg::V(eps)};
            d.grid[0] = HV;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        mtl::Dispatch d;
        d.p = &p_gnc;
        d.args = {mtl::Arg::B(o2), mtl::Arg::B(w_norm), mtl::Arg::B(z_all), mtl::Arg::V(S), mtl::Arg::V(ZV),
                  mtl::Arg::V(eps)};
        d.grid[0] = T;
        d.grid[1] = HV;
        d.tg[0] = 128;
        cx.run_graph({d});
        check("gated norm", o1.ptr, o2.ptr, 4 * T * ZV);
    }

    // ---- 11. the MoE combine over the chunk vs per token ----------------------------------------------------
    {
        mtl::Buffer ey_dec = cx.alloc(4 * T * K * N), ey_ent = cx.alloc(4 * T * K * N);
        mtl::Buffer sy = cx.alloc(4 * T * N), gs = cx.alloc(4 * T);
        for (int i = 0; i < T * K * N; ++i) ey_dec.as<float>()[i] = r.sym();
        for (int i = 0; i < T * N; ++i) sy.as<float>()[i] = r.sym();
        for (int i = 0; i < T; ++i) gs.as<float>()[i] = r.sym() * 3.0f;
        for (int t = 0; t < T; ++t)
            for (int k = 0; k < K; ++k)
                std::memcpy(ey_ent.as<float>() + (size_t) pos.as<int>()[t * K + k] * N,
                            ey_dec.as<float>() + (size_t)(t * K + k) * N, 4 * N);
        mtl::Buffer x1 = cx.alloc(4 * T * N), x2 = cx.alloc(4 * T * N);
        std::memcpy(x1.ptr, h_all.ptr, 4 * T * N);
        std::memcpy(x2.ptr, h_all.ptr, 4 * T * N);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            mtl::Dispatch d;
            d.p = &p_sum;
            d.args = {mtl::Arg::B(x1, 4 * (size_t) t * N), mtl::Arg::B(ey_dec, 4 * (size_t) t * K * N),
                      mtl::Arg::B(wts1, 4 * (size_t) t * K), mtl::Arg::B(sy, 4 * (size_t) t * N),
                      mtl::Arg::B(gs, 4 * (size_t) t), mtl::Arg::V(K), mtl::Arg::V(N)};
            d.grid[0] = (N + 127) / 128;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        mtl::Dispatch d;
        d.p = &p_sumc;
        d.args = {mtl::Arg::B(x2), mtl::Arg::B(ey_ent), mtl::Arg::B(wts2), mtl::Arg::B(sy), mtl::Arg::B(gs),
                  mtl::Arg::B(pos), mtl::Arg::V(K), mtl::Arg::V(N), mtl::Arg::V(N)};
        d.grid[0] = T;
        d.grid[1] = (N + 127) / 128;
        d.tg[0] = 128;
        cx.run_graph({d});
        check("moe combine", x1.ptr, x2.ptr, 4 * T * N);
    }

    // ---- 12. attention prep over the chunk vs per token (q / gate rows, the KV cells) ------------------------
    {
        const int CTX = 32, NROT = HD;
        const float base = 10000.0f;
        mtl::Buffer qg = cx.alloc(4 * T * H * 2 * HD), kx = cx.alloc(4 * T * HKV * HD), vx = cx.alloc(4 * T * HKV * HD);
        mtl::Buffer qw = cx.alloc(4 * HD), kw = cx.alloc(4 * HD);
        for (int i = 0; i < T * H * 2 * HD; ++i) qg.as<float>()[i] = r.sym();
        for (int i = 0; i < T * HKV * HD; ++i) {
            kx.as<float>()[i] = r.sym();
            vx.as<float>()[i] = r.sym();
        }
        for (int i = 0; i < HD; ++i) {
            qw.as<float>()[i] = 1.0f + 0.2f * r.sym();
            kw.as<float>()[i] = 1.0f + 0.2f * r.sym();
        }
        mtl::Buffer q1 = cx.alloc(4 * T * H * HD), gate1 = cx.alloc(4 * T * H * HD);
        mtl::Buffer q2 = cx.alloc(4 * T * H * HD), gate2 = cx.alloc(4 * T * H * HD);
        const size_t kvb = (size_t) HKV * CTX * HD * (kf ? 2 : 4);   // the f16 lib's caches are 2 B/elem
        mtl::Buffer kc1 = cx.alloc(kvb), vc1 = cx.alloc(kvb);
        mtl::Buffer kc2 = cx.alloc(kvb), vc2 = cx.alloc(kvb);
        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            mtl::Dispatch d;
            d.p = &p_prep;
            d.args = {mtl::Arg::B(qg, 4 * (size_t) t * H * 2 * HD), mtl::Arg::B(kx, 4 * (size_t) t * HKV * HD),
                      mtl::Arg::B(vx, 4 * (size_t) t * HKV * HD), mtl::Arg::B(qw), mtl::Arg::B(kw),
                      mtl::Arg::B(q1, 4 * (size_t) t * H * HD), mtl::Arg::B(gate1, 4 * (size_t) t * H * HD),
                      mtl::Arg::B(kc1), mtl::Arg::B(vc1), mtl::Arg::V(H), mtl::Arg::V(HD), mtl::Arg::V(NROT),
                      mtl::Arg::V(t), mtl::Arg::V(CTX), mtl::Arg::V(base), mtl::Arg::V(eps)};
            d.grid[0] = H + HKV;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        cx.run_graph(ds);
        mtl::Dispatch d;
        d.p = &p_prep_c;
        d.args = {mtl::Arg::B(qg), mtl::Arg::B(kx), mtl::Arg::B(vx), mtl::Arg::B(qw), mtl::Arg::B(kw),
                  mtl::Arg::B(q2), mtl::Arg::B(gate2), mtl::Arg::B(kc2), mtl::Arg::B(vc2), mtl::Arg::V(H),
                  mtl::Arg::V(HKV), mtl::Arg::V(HD), mtl::Arg::V(NROT), mtl::Arg::V(0), mtl::Arg::V(CTX),
                  mtl::Arg::V(base), mtl::Arg::V(eps)};
        d.grid[0] = T;
        d.grid[1] = H + HKV;
        d.tg[0] = 128;
        cx.run_graph({d});
        check("prep q", q1.ptr, q2.ptr, 4 * T * H * HD);
        check("prep gate", gate1.ptr, gate2.ptr, 4 * T * H * HD);
        if (kf) {   // half caches compare byte-wise: the f16 stores are deterministic (round-to-nearest-
                    // even), so the two forms' bytes must be identical; the float check() would misread
                    // the pairs as one float
            check_bytes("prep kc", kc1.ptr, kc2.ptr, kvb);
            check_bytes("prep vc", vc1.ptr, vc2.ptr, kvb);
        } else {
            check("prep kc", kc1.ptr, kc2.ptr, kvb);
            check("prep vc", vc1.ptr, vc2.ptr, kvb);
        }
    }

    // ---- 13. split-K attention rows: per-token form vs the row-blocked chunk form (row0 = 0 and 16) ----------
    {
        const int CTX = 64;
        const float scale = 1.0f / std::sqrt((float) 256);
        mtl::Buffer q_a = cx.alloc(4 * T * H * 256);
        for (int i = 0; i < T * H * 256; ++i) q_a.as<float>()[i] = r.sym();
        const size_t kvb = (size_t) HKV * CTX * 256 * (kf ? 2 : 4);
        mtl::Buffer kc = cx.alloc(kvb), vc = cx.alloc(kvb);
        for (int i = 0; i < HKV * CTX * 256; ++i) {
            if (kf) {   // seed through the float->half rounding once, so both row forms read identical halves
                ((uint16_t*) kc.ptr)[i] = ref::f32_to_f16(r.sym());
                ((uint16_t*) vc.ptr)[i] = ref::f32_to_f16(r.sym());
            } else {
                kc.as<float>()[i] = r.sym();
                vc.as<float>()[i] = r.sym();
            }
        }
        mtl::Buffer gate_all = cx.alloc(4 * T * H * 256);
        for (int i = 0; i < T * H * 256; ++i) gate_all.as<float>()[i] = r.sym();
        mtl::Buffer att1 = cx.alloc(4 * T * H * 256), att2 = cx.alloc(4 * T * H * 256);
        mtl::Buffer pacc_s = cx.alloc(4 * 256 * 256), pm_s = cx.alloc(4 * 256), pl_s = cx.alloc(4 * 256);
        const int n_chunk = 1;   // every row's npos <= T <= 256 here: one cell chunk

        std::vector<mtl::Dispatch> ds;
        for (int t = 0; t < T; ++t) {
            mtl::Dispatch dp;
            dp.p = &p_sk_part;
            dp.args = {mtl::Arg::B(q_a, 4 * (size_t) t * H * 256), mtl::Arg::B(kc), mtl::Arg::B(vc),
                       mtl::Arg::V(H), mtl::Arg::V(HKV), mtl::Arg::V(CTX), mtl::Arg::V(scale), mtl::Arg::V(n_chunk),
                       mtl::Arg::V(t), mtl::Arg::V(0), mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s)};
            dp.grid[0] = n_chunk;
            dp.grid[1] = H;
            dp.tg[0] = 256;
            dp.barrier = true;
            ds.push_back(dp);
            mtl::Dispatch dm;
            dm.p = &p_sk_merge;
            dm.args = {mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s),
                       mtl::Arg::B(gate_all, 4 * (size_t) t * H * 256), mtl::Arg::B(att1, 4 * (size_t) t * H * 256),
                       mtl::Arg::V(H), mtl::Arg::V(n_chunk), mtl::Arg::V(t), mtl::Arg::V(0)};
            dm.grid[0] = H;
            dm.tg[0] = 256;
            dm.barrier = true;
            ds.push_back(dm);
        }
        cx.run_graph(ds);

        // the chunk form in two row blocks, exercising row0 = 16 on the second
        std::vector<mtl::Dispatch> ds2;
        for (int pass = 0; pass < 2; ++pass) {
            const int row0 = pass * 16;
            mtl::Dispatch dp;
            dp.p = &p_sk_part;
            dp.args = {mtl::Arg::B(q_a), mtl::Arg::B(kc), mtl::Arg::B(vc), mtl::Arg::V(H), mtl::Arg::V(HKV),
                       mtl::Arg::V(CTX), mtl::Arg::V(scale), mtl::Arg::V(n_chunk), mtl::Arg::V(0), mtl::Arg::V(row0),
                       mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s)};
            dp.grid[0] = n_chunk;
            dp.grid[1] = 16;
            dp.tg[0] = 256;
            dp.barrier = true;
            ds2.push_back(dp);
            mtl::Dispatch dm;
            dm.p = &p_sk_merge;
            dm.args = {mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s), mtl::Arg::B(gate_all),
                       mtl::Arg::B(att2), mtl::Arg::V(H), mtl::Arg::V(n_chunk), mtl::Arg::V(0), mtl::Arg::V(row0)};
            dm.grid[0] = 16;
            dm.tg[0] = 256;
            dm.barrier = true;
            ds2.push_back(dm);
        }
        cx.run_graph(ds2);
        check("sk rows", att1.ptr, att2.ptr, 4 * T * H * 256);
    }

    // ---- 13b. split-K rows, multi-token blocks: the per-token form vs the STRATA_Q36M_ATTN_ROWS row
    // blocks (B token rows per part/merge pair; the block's n_chunk is its LAST row's count) - the
    // new host shapes the engine's blocking loop emits: n_chunk > 1, pos0 > 0, a partial tail block,
    // and the folded part kernel's grid.z token form.  Same per-(row, chunk) arithmetic, so every
    // comparison is bit-equal; the scratch is NaN-poisoned (0xff) so a stale-slot read shows up as
    // NaN cells in the check (row 0 at pos0=255 has an unwritten slot by design and never reads it).
    {
        const int CTX = 600;
        const float scale = 1.0f / std::sqrt((float) 256);
        mtl::Buffer q_a = cx.alloc(4 * T * H * 256), gate_all = cx.alloc(4 * T * H * 256);
        for (int i = 0; i < T * H * 256; ++i) {
            q_a.as<float>()[i] = r.sym();
            gate_all.as<float>()[i] = r.sym();
        }
        const size_t kvb = (size_t) HKV * CTX * 256 * (kf ? 2 : 4);
        mtl::Buffer kc = cx.alloc(kvb), vc = cx.alloc(kvb);
        for (int i = 0; i < HKV * CTX * 256; ++i) {
            if (kf) {   // seed through the float->half rounding once, so both row forms read identical halves
                ((uint16_t*) kc.ptr)[i] = ref::f32_to_f16(r.sym());
                ((uint16_t*) vc.ptr)[i] = ref::f32_to_f16(r.sym());
            } else {
                kc.as<float>()[i] = r.sym();
                vc.as<float>()[i] = r.sym();
            }
        }
        mtl::Pipeline p_sk_part_gqa = pipe_kv("k_attn_sk_part_gqa");
        const int nc_max = (CTX + 255) / 256;
        mtl::Buffer pacc_s = cx.alloc(4 * 256 * nc_max * 256), pm_s = cx.alloc(4 * 256 * nc_max),
                    pl_s = cx.alloc(4 * 256 * nc_max);
        mtl::Buffer att1 = cx.alloc(4 * T * H * 256), att2 = cx.alloc(4 * T * H * 256);

        for (int pass = 0; pass < 2; ++pass) {   // 0: the unfolded part kernel, 1: the folded one
            const bool folded = pass == 1;
            const mtl::Pipeline& p_part = folded ? p_sk_part_gqa : p_sk_part;
            const int part_gy = folded ? HKV : H;   // threadgroups per token: HKV (folded) or H rows
            const uint32_t tgmem = (uint32_t)(folded ? (size_t)(H / HKV) * (256 + 8 + 256) * 4 : 0);
            for (int p0base : {0, 255, 300}) {
                // the per-token form (the released dispatches): one part+merge pair per token
                std::memset(att1.ptr, 0xff, att1.bytes);
                std::memset(pacc_s.ptr, 0xff, pacc_s.bytes);
                std::memset(pm_s.ptr, 0xff, pm_s.bytes);
                std::memset(pl_s.ptr, 0xff, pl_s.bytes);
                std::vector<mtl::Dispatch> ds;
                for (int t = 0; t < T; ++t) {
                    const int npos = p0base + t + 1;
                    const int n_chunk = (npos + 255) / 256;
                    mtl::Dispatch dp;
                    dp.p = &p_part;
                    dp.args = {mtl::Arg::B(q_a, 4 * (size_t) t * H * 256), mtl::Arg::B(kc), mtl::Arg::B(vc),
                               mtl::Arg::V(H), mtl::Arg::V(HKV), mtl::Arg::V(CTX), mtl::Arg::V(scale),
                               mtl::Arg::V(n_chunk), mtl::Arg::V(p0base), mtl::Arg::V(0),
                               mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s)};
                    dp.grid[0] = n_chunk;
                    dp.grid[1] = part_gy;
                    dp.tg[0] = 256;
                    dp.tg_mem = tgmem;
                    dp.barrier = true;
                    ds.push_back(dp);
                    mtl::Dispatch dm;
                    dm.p = &p_sk_merge;
                    dm.args = {mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s),
                               mtl::Arg::B(gate_all, 4 * (size_t) t * H * 256),
                               mtl::Arg::B(att1, 4 * (size_t) t * H * 256), mtl::Arg::V(H),
                               mtl::Arg::V(n_chunk), mtl::Arg::V(p0base), mtl::Arg::V(0)};
                    dm.grid[0] = H;
                    dm.tg[0] = 256;
                    dm.barrier = true;
                    ds.push_back(dm);
                }
                cx.run_graph(ds);

                // the row-block form: B consecutive token rows per part/merge pair (B=3's last block
                // holds 2 rows - the partial tail the engine's final block has whenever nt % B != 0)
                for (int B : {3, 4}) {
                    std::memset(att2.ptr, 0xff, att2.bytes);
                    std::memset(pacc_s.ptr, 0xff, pacc_s.bytes);
                    std::memset(pm_s.ptr, 0xff, pm_s.bytes);
                    std::memset(pl_s.ptr, 0xff, pl_s.bytes);
                    std::vector<mtl::Dispatch> ds2;
                    int t = 0;
                    while (t < T) {
                        int nb = 1;
                        while (nb < B && t + nb < T) ++nb;
                        const int n_chunk = (p0base + t + nb + 255) / 256;
                        mtl::Dispatch dp;
                        dp.p = &p_part;
                        dp.args = {mtl::Arg::B(q_a), mtl::Arg::B(kc), mtl::Arg::B(vc),
                                   mtl::Arg::V(H), mtl::Arg::V(HKV), mtl::Arg::V(CTX), mtl::Arg::V(scale),
                                   mtl::Arg::V(n_chunk), mtl::Arg::V(p0base), mtl::Arg::V(t * H),
                                   mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s)};
                        dp.grid[0] = n_chunk;
                        if (folded) {
                            dp.grid[1] = HKV;
                            dp.grid[2] = nb;
                        } else {
                            dp.grid[1] = nb * H;
                        }
                        dp.tg[0] = 256;
                        dp.tg_mem = tgmem;
                        dp.barrier = true;
                        ds2.push_back(dp);
                        mtl::Dispatch dm;
                        dm.p = &p_sk_merge;
                        dm.args = {mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s),
                                   mtl::Arg::B(gate_all), mtl::Arg::B(att2), mtl::Arg::V(H),
                                   mtl::Arg::V(n_chunk), mtl::Arg::V(p0base), mtl::Arg::V(t * H)};
                        dm.grid[0] = nb * H;
                        dm.tg[0] = 256;
                        dm.barrier = true;
                        ds2.push_back(dm);
                        t += nb;
                    }
                    cx.run_graph(ds2);
                    char name[64];
                    std::snprintf(name, sizeof name, "sk rows mt %s B=%d p0=%d", folded ? "gqa" : "unfold",
                                  B, p0base);
                    check(name, att1.ptr, att2.ptr, 4 * T * H * 256);
                }
            }
        }
    }

    // ---- 14. the full MoE chain as ONE barrier-ordered graph (the engine's structure) vs the same
    // dispatches as separate graphs (the stage tests' structure): embed -> rms -> the router gemv -> route
    // -> k_tables -> grouped experts -> swiglu.  All the kernels are proven exact stage by stage (and the
    // old four-dispatch atomic table chain RACED inside one graph - k_expc read stale tables); this stage
    // asks whether the ONE-GRAPH barrier ordering keeps k_tables + the table-dependent gemvs exact too.
    {
        mtl::Buffer W_e = cx.alloc(32 * (uint64_t) (N / 32) * 34);   // Q8_0 embedding rows, vocab 32
        for (size_t i = 0; i < W_e.bytes; ++i) W_e.as<unsigned char>()[i] = (unsigned char) (r.next() & 255);
        sanitize(W_e);
        const uint64_t RBE = (uint64_t) (N / 32) * 34;
        int toks[T];
        for (int& t : toks) t = (int) (r.uni() * 32);
        mtl::Buffer toks_b = cx.alloc(sizeof(toks));
        std::memcpy(toks_b.ptr, toks, sizeof(toks));

        const uint64_t RBO = 74ull;   // IQ2_XS rows of n_in = N = 256
        const uint64_t EBO = RBO * FF;
        mtl::Buffer W_g = cx.alloc(NEXP * EBO), W_u = cx.alloc(NEXP * EBO);
        for (size_t i = 0; i < W_g.bytes; ++i) W_g.as<unsigned char>()[i] = (unsigned char) (r.next() & 255);
        for (size_t i = 0; i < W_u.bytes; ++i) W_u.as<unsigned char>()[i] = (unsigned char) (r.next() & 255);
        sanitize(W_g);
        sanitize(W_u);
        mtl::Buffer w_norm = cx.alloc(4 * N);

        // the layer's shared inputs: post-norm hidden is produced in-graph; build the router/expert inputs
        // from it.  Run the chain TWICE into separate buffer sets: once as ONE graph, once as N graphs.
        mtl::Buffer xc1 = cx.alloc(4 * T * N), hc1 = cx.alloc(4 * T * N), yc1 = cx.alloc(4 * T * N);
        mtl::Buffer xc2 = cx.alloc(4 * T * N), hc2 = cx.alloc(4 * T * N), yc2 = cx.alloc(4 * T * N);
        mtl::Buffer r1 = cx.alloc(4 * T * NEXP), r2 = cx.alloc(4 * T * NEXP);
        mtl::Buffer id1 = cx.alloc(4 * T * K), wt1 = cx.alloc(4 * T * K);
        mtl::Buffer id2 = cx.alloc(4 * T * K), wt2 = cx.alloc(4 * T * K);
        mtl::Buffer scr1 = cx.alloc(4 * 3 * 256), scr2 = cx.alloc(4 * 3 * 256);
        mtl::Buffer sg1 = cx.alloc(4 * T * K * FF), sg2 = cx.alloc(4 * T * K * FF);
        mtl::Buffer su1 = cx.alloc(4 * T * K * FF), su2 = cx.alloc(4 * T * K * FF);
        mtl::Buffer sh1 = cx.alloc(4 * T * K * FF), sh2 = cx.alloc(4 * T * K * FF);
        mtl::Buffer ey1 = cx.alloc(4 * T * K * N), ey2 = cx.alloc(4 * T * K * N);
        mtl::Buffer pos1 = cx.alloc(4 * T * K), pos2 = cx.alloc(4 * T * K);
        mtl::Buffer sgu1 = cx.alloc(4 * T * K), sgu2 = cx.alloc(4 * T * K);   // srt_in_gu: entry -> token
        mtl::Buffer sd1 = cx.alloc(4 * T * K), sd2 = cx.alloc(4 * T * K);     // srt_in_d: entry -> entry index
        mtl::Pipeline p_emb = pipe("k_emb_chunk_T_Q8_0");
        mtl::Pipeline p_gc_iq2 = pipe("k_gemv_c_T_IQ2_XS");

        auto chain = [&](std::vector<mtl::Dispatch>& ds, mtl::Buffer& xc, mtl::Buffer& hc, mtl::Buffer& yc,
                         mtl::Buffer& ro, mtl::Buffer& ids, mtl::Buffer& wts, mtl::Buffer& scr, mtl::Buffer& eg,
                         mtl::Buffer& eu, mtl::Buffer& eh, mtl::Buffer& ey, mtl::Buffer& pos, mtl::Buffer& sgu,
                         mtl::Buffer& sdt) {
            mtl::Dispatch d0;
            d0.p = &p_emb;
            d0.args = {mtl::Arg::B(W_e), mtl::Arg::B(toks_b), mtl::Arg::B(xc), mtl::Arg::V(RBE), mtl::Arg::V(N), mtl::Arg::V(N)};
            d0.grid[0] = T;
            d0.tg[0] = 128;
            d0.barrier = true;
            ds.push_back(d0);
            mtl::Dispatch d1;
            d1.p = &p_rms;
            d1.args = {mtl::Arg::B(xc), mtl::Arg::B(w_norm), mtl::Arg::B(hc), mtl::Arg::V(N), mtl::Arg::V(eps)};
            d1.grid[0] = T;
            d1.tg[0] = 128;
            d1.barrier = true;
            ds.push_back(d1);
            mtl::Dispatch d2;
            d2.p = &p_gc_iq2;
            d2.args = {mtl::Arg::B(W_g), mtl::Arg::B(hc), mtl::Arg::B(ro),
                       mtl::Arg::V(GemvParam{RBO, 0, N, NEXP, (unsigned long) N}), mtl::Arg::V(1), mtl::Arg::V(0), mtl::Arg::V(T)};
            d2.grid[0] = (NEXP + 7) / 8;
            d2.grid[1] = 1;
            d2.tg[0] = 256;
            d2.barrier = true;
            ds.push_back(d2);
            mtl::Dispatch d3;
            d3.p = &p_route;
            d3.args = {mtl::Arg::B(ro), mtl::Arg::B(ids), mtl::Arg::B(wts), mtl::Arg::V(NEXP), mtl::Arg::V(K)};
            d3.grid[0] = T;
            d3.tg[0] = 256;
            d3.barrier = true;
            ds.push_back(d3);
            mtl::Dispatch dt;
            dt.p = &p_tables;
            dt.args = {mtl::Arg::B(ids), mtl::Arg::B(scr), mtl::Arg::B(sgu), mtl::Arg::B(sdt), mtl::Arg::B(pos),
                       mtl::Arg::V(NEXP), mtl::Arg::V(K), mtl::Arg::V(T * K)};
            dt.grid[0] = 1;
            dt.tg[0] = 256;
            dt.barrier = true;
            ds.push_back(dt);
            GemvParam pe{RBO, EBO, N, FF, (unsigned long) N};
            mtl::Dispatch d4;
            d4.p = &p_expc_iq2;
            d4.args = {mtl::Arg::B(W_g), mtl::Arg::B(hc), mtl::Arg::B(sgu), mtl::Arg::B(scr, 256 * 4),
                       mtl::Arg::B(scr), mtl::Arg::B(eg), mtl::Arg::V(pe)};
            d4.grid[0] = (FF + 7) / 8;
            d4.grid[1] = NEXP;
            d4.tg[0] = 256;
            d4.barrier = true;
            ds.push_back(d4);
            mtl::Dispatch d5;
            d5.p = &p_expc_iq2;
            d5.args = {mtl::Arg::B(W_u), mtl::Arg::B(hc), mtl::Arg::B(sgu), mtl::Arg::B(scr, 256 * 4),
                       mtl::Arg::B(scr), mtl::Arg::B(eu), mtl::Arg::V(pe)};
            d5.grid[0] = (FF + 7) / 8;
            d5.grid[1] = NEXP;
            d5.tg[0] = 256;
            d5.barrier = true;
            ds.push_back(d5);
            mtl::Dispatch dsw;
            dsw.p = &p_swiglu;
            dsw.args = {mtl::Arg::B(eg), mtl::Arg::B(eu), mtl::Arg::B(eh), mtl::Arg::V(T * K * FF)};
            dsw.grid[0] = (T * K * FF + 127) / 128;
            dsw.tg[0] = 128;
            dsw.barrier = true;
            ds.push_back(dsw);
        };
        std::vector<mtl::Dispatch> one, many;
        chain(one, xc1, hc1, yc1, r1, id1, wt1, scr1, sg1, su1, sh1, ey1, pos1, sgu1, sd1);
        chain(many, xc2, hc2, yc2, r2, id2, wt2, scr2, sg2, su2, sh2, ey2, pos2, sgu2, sd2);
        cx.run_graph(one);
        for (const auto& d : many) cx.run_graph({d});
        // the tables' rank order can differ between the two runs (threadgroup scheduling); the VALUES must
        // not - so the placement is checked as the invariant and the expert rows matched through pos
        check("chain router", r1.ptr, r2.ptr, 4 * T * NEXP);
        check("chain route ids", id1.ptr, id2.ptr, 4 * T * K);
        check("chain route wts", wt1.ptr, wt2.ptr, 4 * T * K);
        check("chain scr", scr1.ptr, scr2.ptr, 4 * 2 * 256);
        check_placement("chain place 1", id1.as<int>(), scr1.as<int>(), sgu1.as<int>(), sd1.as<int>(),
                        pos1.as<int>(), T * K);
        check_placement("chain place 2", id2.as<int>(), scr2.as<int>(), sgu2.as<int>(), sd2.as<int>(),
                        pos2.as<int>(), T * K);
        check_scattered2("chain experts", sg1.as<float>(), pos1.as<int>(), sg2.as<float>(), pos2.as<int>());
        check_scattered2("chain combine-in", sh1.as<float>(), pos1.as<int>(), sh2.as<float>(), pos2.as<int>());
    }

    // ---- 15. k_tables (one dispatch, threadgroup atomics) vs the old zero/hist/exp_off/place chain (kept
    // in the MSL file for this): the counts and offsets must be identical, the placement must satisfy the
    // invariant (the old chain's own rank order is free to differ)
    {
        mtl::Buffer ids3 = cx.alloc(4 * T * K);
        std::memcpy(ids3.ptr, ids2.ptr, 4 * T * K);
        mtl::Buffer scr_t = cx.alloc(4 * 2 * 256), scr_o = cx.alloc(4 * 3 * 256);
        mtl::Buffer gu_t = cx.alloc(4 * T * K), gu_o = cx.alloc(4 * T * K);
        mtl::Buffer sd_t = cx.alloc(4 * T * K), sd_o = cx.alloc(4 * T * K);
        mtl::Buffer pos_t = cx.alloc(4 * T * K), pos_o = cx.alloc(4 * T * K);
        mtl::Dispatch dt;
        dt.p = &p_tables;
        dt.args = {mtl::Arg::B(ids3), mtl::Arg::B(scr_t), mtl::Arg::B(gu_t), mtl::Arg::B(sd_t), mtl::Arg::B(pos_t),
                   mtl::Arg::V(NEXP), mtl::Arg::V(K), mtl::Arg::V(T * K)};
        dt.grid[0] = 1;
        dt.tg[0] = 256;
        cx.run_graph({dt});
        mtl::Dispatch dz;
        dz.p = &p_zero;
        dz.args = {mtl::Arg::B(scr_o), mtl::Arg::V(3 * 256)};
        dz.grid[0] = (3 * 256 + 127) / 128;
        dz.tg[0] = 128;
        dz.barrier = true;
        mtl::Dispatch dh;
        dh.p = &p_hist;
        dh.args = {mtl::Arg::B(ids3), mtl::Arg::B(scr_o), mtl::Arg::V(T * K)};
        dh.grid[0] = (T * K + 255) / 256;
        dh.tg[0] = 256;
        dh.barrier = true;
        mtl::Dispatch doff;
        doff.p = &p_off;
        doff.args = {mtl::Arg::B(scr_o), mtl::Arg::B(scr_o, 512 * 4), mtl::Arg::V(NEXP)};
        doff.grid[0] = 1;
        doff.tg[0] = 1;
        doff.barrier = true;
        mtl::Dispatch dp;
        dp.p = &p_place;
        dp.args = {mtl::Arg::B(ids3), mtl::Arg::B(scr_o, 256 * 4), mtl::Arg::B(scr_o, 512 * 4), mtl::Arg::B(gu_o),
                   mtl::Arg::B(sd_o), mtl::Arg::B(pos_o), mtl::Arg::V(K), mtl::Arg::V(T * K)};
        dp.grid[0] = (T * K + 255) / 256;
        dp.tg[0] = 256;
        dp.barrier = true;
        cx.run_graph({dz, dh, doff, dp});
        check("tables cnt", scr_t.ptr, scr_o.ptr, 4 * 256);
        check("tables off", (const char*) scr_t.ptr + 4 * 256, (const char*) scr_o.ptr + 4 * 512, 4 * 256);
        check_placement("tables place", ids3.as<int>(), scr_t.as<int>(), gu_t.as<int>(), sd_t.as<int>(),
                        pos_t.as<int>(), T * K);
    }

    // ---- 16. the full routed MoE at the lpr8 geometry (n_in = 2048 gate/up, the engine's real expert
    // formats): the chunk chain (gather + per-entry decode-lpr8 gate/up -> swiglu -> down expc reading the
    // (t,k)-ordered eh through srt_in_d -> combine through pos) vs the decode chain per token (lpr8
    // gate/up, plain-gemv down, k_moe_sum).  This is the pair round 9's G8 cut caught: k_expc's
    // plain-gemv order differs from lpr8's by ~1 ulp at n_in = 2048.  The shared expert is zeroed on both
    // sides (sgate = 0, sy = 0) so the compare isolates the routed machinery.
    {
        const int N2 = 2048;               // the lpr8 kernel's hardwired input width (the real n_embd)
        const uint64_t RBG = 74ull * 8;    // IQ2_XS rows of n_in = 2048
        const uint64_t RBD = 98ull;        // IQ3_XXS rows of n_in = FF (one 256-dim block, FF < 256)
        const uint64_t EBG = RBG * FF, EBD = RBD * N2;
        mtl::Buffer Wg = cx.alloc(NEXP * EBG), Wu = cx.alloc(NEXP * EBG), Wd = cx.alloc(NEXP * EBD);
        for (size_t i = 0; i < Wg.bytes; ++i) Wg.as<unsigned char>()[i] = (unsigned char) (r.next() & 255);
        for (size_t i = 0; i < Wu.bytes; ++i) Wu.as<unsigned char>()[i] = (unsigned char) (r.next() & 255);
        for (size_t i = 0; i < Wd.bytes; ++i) Wd.as<unsigned char>()[i] = (unsigned char) (r.next() & 255);
        sanitize(Wg);
        sanitize(Wu);
        sanitize(Wd);
        mtl::Buffer hg = cx.alloc(4 * T * N2);
        for (int i = 0; i < T * N2; ++i) hg.as<float>()[i] = r.sym();
        mtl::Buffer eg1 = cx.alloc(4 * T * K * FF), eu1 = cx.alloc(4 * T * K * FF), eh1 = cx.alloc(4 * T * K * FF);
        mtl::Buffer ey1 = cx.alloc(4 * T * K * N2), x1 = cx.alloc(4 * T * N2);
        mtl::Buffer hgc = cx.alloc(4 * T * K * N2), eg2 = cx.alloc(4 * T * K * FF), eu2 = cx.alloc(4 * T * K * FF),
                    eh2 = cx.alloc(4 * T * K * FF), ey2 = cx.alloc(4 * T * K * N2), x2 = cx.alloc(4 * T * N2);
        mtl::Buffer scr2 = cx.alloc(4 * 2 * 256), sgu = cx.alloc(4 * T * K), sd = cx.alloc(4 * T * K),
                    pos2 = cx.alloc(4 * T * K);
        mtl::Buffer sy0 = cx.alloc(4 * T * N2), gs0 = cx.alloc(4 * T);   // cx.alloc zeroes: the shared expert off
        for (int i = 0; i < T * N2; ++i) x1.as<float>()[i] = r.sym();
        std::memcpy(x2.ptr, x1.ptr, 4 * T * N2);
        mtl::Buffer x3 = cx.alloc(4 * T * N2);
        std::memcpy(x3.ptr, x1.ptr, 4 * T * N2);   // the pristine initial: neither combine touches it

        // probe A: what does the GPU see of the host-written x1/x2/x3?  pure copies (k_gather_t with k=1),
        // host-read before any chain runs - decides "the memcpy never reaches the GPU" vs "something later
        // zeroes x2's GPU view".
        {
            mtl::Buffer x1g = cx.alloc(4 * T * N2), x2g = cx.alloc(4 * T * N2), x3g = cx.alloc(4 * T * N2);
            mtl::Dispatch c1, c2, c3;
            c1.p = c2.p = c3.p = &p_gather;
            c1.args = {mtl::Arg::B(x1), mtl::Arg::B(x1g), mtl::Arg::V(T * N2), mtl::Arg::V(1), mtl::Arg::V(T * N2)};
            c2.args = {mtl::Arg::B(x2), mtl::Arg::B(x2g), mtl::Arg::V(T * N2), mtl::Arg::V(1), mtl::Arg::V(T * N2)};
            c3.args = {mtl::Arg::B(x3), mtl::Arg::B(x3g), mtl::Arg::V(T * N2), mtl::Arg::V(1), mtl::Arg::V(T * N2)};
            c1.grid[0] = c2.grid[0] = c3.grid[0] = (T * N2 + 127) / 128;
            c1.tg[0] = c2.tg[0] = c3.tg[0] = 128;
            cx.run_graph({c1, c2, c3});
            std::printf("  moe2048 probeA: gpu x1=%.9g/%.9g x2=%.9g/%.9g x3=%.9g/%.9g | host x1=%.9g x2=%.9g x3=%.9g\n",
                        x1g.as<float>()[0], x1g.as<float>()[T * N2 - 1], x2g.as<float>()[0],
                        x2g.as<float>()[T * N2 - 1], x3g.as<float>()[0], x3g.as<float>()[T * N2 - 1],
                        x1.as<float>()[0], x2.as<float>()[0], x3.as<float>()[0]);
        }

        // the decode chain, per token (the engine's own shapes: gate/up lpr8 with x_stride 0, the down
        // plain gemv at x_stride FF, k_moe_sum on the token's K rows)
        {
            std::vector<mtl::Dispatch> ds;
            for (int t = 0; t < T; ++t) {
                GemvParam pg{RBG, EBG, N2, FF, 0};
                mtl::Dispatch dg;
                dg.p = &p_lpr8_iq2;
                dg.args = {mtl::Arg::B(Wg), mtl::Arg::B(hg, 4 * (size_t) t * N2),
                           mtl::Arg::B(eg1, 4 * (size_t) t * K * FF), mtl::Arg::B(ids2, 4 * (size_t) t * K),
                           mtl::Arg::V(pg)};
                dg.grid[0] = (FF + 15) / 16;
                dg.grid[1] = K;
                dg.tg[0] = 128;
                dg.barrier = true;
                ds.push_back(dg);
                mtl::Dispatch du;
                du.p = &p_lpr8_iq2;
                du.args = {mtl::Arg::B(Wu), mtl::Arg::B(hg, 4 * (size_t) t * N2),
                           mtl::Arg::B(eu1, 4 * (size_t) t * K * FF), mtl::Arg::B(ids2, 4 * (size_t) t * K),
                           mtl::Arg::V(pg)};
                du.grid[0] = (FF + 15) / 16;
                du.grid[1] = K;
                du.tg[0] = 128;
                du.barrier = true;
                ds.push_back(du);
            }
            cx.run_graph(ds);
            mtl::Dispatch dsw;
            dsw.p = &p_swiglu;
            dsw.args = {mtl::Arg::B(eg1), mtl::Arg::B(eu1), mtl::Arg::B(eh1), mtl::Arg::V(T * K * FF)};
            dsw.grid[0] = (T * K * FF + 127) / 128;
            dsw.tg[0] = 128;
            dsw.barrier = true;
            cx.run_graph({dsw});
            std::vector<mtl::Dispatch> ds2;
            for (int t = 0; t < T; ++t) {
                GemvParam pd{RBD, EBD, FF, N2, (unsigned long) FF};
                mtl::Dispatch dd;
                dd.p = &p_g_iq3;
                dd.args = {mtl::Arg::B(Wd), mtl::Arg::B(eh1, 4 * (size_t) t * K * FF),
                           mtl::Arg::B(ey1, 4 * (size_t) t * K * N2), mtl::Arg::B(ids2, 4 * (size_t) t * K),
                           mtl::Arg::V(pd)};
                dd.grid[0] = (N2 + 7) / 8;
                dd.grid[1] = K;
                dd.tg[0] = 256;
                dd.barrier = true;
                ds2.push_back(dd);
                mtl::Dispatch dm;
                dm.p = &p_sum;
                dm.args = {mtl::Arg::B(x1, 4 * (size_t) t * N2), mtl::Arg::B(ey1, 4 * (size_t) t * K * N2),
                           mtl::Arg::B(wts2, 4 * (size_t) t * K), mtl::Arg::B(sy0), mtl::Arg::B(gs0),
                           mtl::Arg::V(K), mtl::Arg::V(N2)};
                dm.grid[0] = (N2 + 127) / 128;
                dm.tg[0] = 128;
                dm.barrier = true;
                ds2.push_back(dm);
            }
            cx.run_graph(ds2);
        }

        // probe B: x2's GPU view again, after the decode chain ran (three command buffers committed since
        // the memcpy) - catches a corruption that happens DURING the decode chain's dispatches.
        {
            mtl::Buffer x2m = cx.alloc(4 * T * N2);
            mtl::Dispatch cm;
            cm.p = &p_gather;
            cm.args = {mtl::Arg::B(x2), mtl::Arg::B(x2m), mtl::Arg::V(T * N2), mtl::Arg::V(1), mtl::Arg::V(T * N2)};
            cm.grid[0] = (T * N2 + 127) / 128;
            cm.tg[0] = 128;
            cx.run_graph({cm});
            std::printf("  moe2048 probeB: gpu x2=%.9g/%.9g (host %.9g)\n", x2m.as<float>()[0],
                        x2m.as<float>()[T * N2 - 1], x2.as<float>()[0]);
        }

        // the chunk chain, one graph (k_tables -> gather -> per-entry lpr8 gate/up -> swiglu -> down expc
        // reading the (t,k)-ordered eh through srt_in_d -> k_moe_sum_chunk through pos)
        {
            mtl::Dispatch dt;
            dt.p = &p_tables;
            dt.args = {mtl::Arg::B(ids2), mtl::Arg::B(scr2), mtl::Arg::B(sgu), mtl::Arg::B(sd), mtl::Arg::B(pos2),
                       mtl::Arg::V(NEXP), mtl::Arg::V(K), mtl::Arg::V(T * K)};
            dt.grid[0] = 1;
            dt.tg[0] = 256;
            dt.barrier = true;
            mtl::Dispatch dga;
            dga.p = &p_gather;
            dga.args = {mtl::Arg::B(hg), mtl::Arg::B(hgc), mtl::Arg::V(N2), mtl::Arg::V(K), mtl::Arg::V(T * K * N2)};
            dga.grid[0] = (T * K * N2 + 127) / 128;
            dga.tg[0] = 128;
            dga.barrier = true;
            mtl::Dispatch dsw;
            dsw.p = &p_swiglu;
            dsw.args = {mtl::Arg::B(eg2), mtl::Arg::B(eu2), mtl::Arg::B(eh2), mtl::Arg::V(T * K * FF)};
            dsw.grid[0] = (T * K * FF + 127) / 128;
            dsw.tg[0] = 128;
            dsw.barrier = true;
            mtl::Dispatch dsum;
            dsum.p = &p_sumc;
            dsum.args = {mtl::Arg::B(x2), mtl::Arg::B(ey2), mtl::Arg::B(wts2), mtl::Arg::B(sy0), mtl::Arg::B(gs0),
                         mtl::Arg::B(pos2), mtl::Arg::V(K), mtl::Arg::V(N2), mtl::Arg::V(N2)};
            dsum.grid[0] = T;
            dsum.grid[1] = (N2 + 127) / 128;
            dsum.tg[0] = 128;
            dsum.barrier = true;
            std::vector<mtl::Dispatch> ds = {dt, dga};
            for (int wi = 0; wi < 2; ++wi) {
                GemvParam p{RBG, EBG, N2, FF, (unsigned long) N2};
                mtl::Dispatch d;
                d.p = &p_lpr8_iq2;
                d.args = {mtl::Arg::B(wi == 0 ? Wg : Wu), mtl::Arg::B(hgc), mtl::Arg::B(wi == 0 ? eg2 : eu2),
                          mtl::Arg::B(ids2), mtl::Arg::V(p)};
                d.grid[0] = (FF + 15) / 16;
                d.grid[1] = T * K;
                d.tg[0] = 128;
                d.barrier = true;
                ds.push_back(d);
            }
            ds.push_back(dsw);
            GemvParam pd{RBD, EBD, FF, N2, (unsigned long) FF};
            mtl::Dispatch dd;
            dd.p = &p_expc_iq3;
            dd.args = {mtl::Arg::B(Wd), mtl::Arg::B(eh2), mtl::Arg::B(sd), mtl::Arg::B(scr2, 256 * 4),
                       mtl::Arg::B(scr2), mtl::Arg::B(ey2), mtl::Arg::V(pd)};
            dd.grid[0] = (N2 + 7) / 8;
            dd.grid[1] = NEXP;
            dd.tg[0] = 256;
            dd.barrier = true;
            ds.push_back(dd);
            ds.push_back(dsum);
            cx.run_graph(ds);
        }
        check("moe2048 eh", eh1.ptr, eh2.ptr, 4 * T * K * FF);
        // the down products: the decode's (t,k) rows vs the chunk's placed rows through the placement -
        // a failure here names the down expc, a pass with a failed x names the combine
        {
            const float* a = ey1.as<float>();
            const float* b = ey2.as<float>();
            const int* pp = pos2.as<int>();
            const char* st = "moe2048 ey";
            bool ok = true;
            for (int t = 0; t < T && ok; ++t)
                for (int k = 0; k < K && ok; ++k) {
                    const float* ar = a + (size_t)(t * K + k) * N2;
                    const float* br = b + (size_t) pp[t * K + k] * N2;
                    for (int o = 0; o < N2; ++o)
                        if (!feq(ar[o], br[o])) {
                            std::printf("  %-14s FAIL at token %d slot %d dim %d (%.9g vs %.9g)\n", st, t, k, o,
                                        ar[o], br[o]);
                            ++g_fails;
                            ok = false;
                            break;
                        }
                }
            if (ok) std::printf("  %-14s OK\n", st);
        }
        check("moe2048 x", x1.ptr, x2.ptr, 4 * T * N2);
        // which combine's delta is missing?  The host recomputes both sides' MoE sums in double from the
        // very buffers the checks above just verified (ey1 == ey2 through pos2, shared wts2), so cpuD ==
        // cpuC exactly; comparing them against the GPU deltas (x - x3) names the combine that never wrote.
        {
            const float* xi = x3.as<float>();
            const float* a = ey1.as<float>();
            const float* b = ey2.as<float>();
            const float* ww = wts2.as<float>();
            const int* pp = pos2.as<int>();
            std::printf("  moe2048 diag: initial[0..2] = %.9g %.9g %.9g\n", xi[0], xi[1], xi[2]);
            for (int t = 0; t < T; t += 7)
                for (int i = 0; i < 2; ++i) {
                    double sd = 0, sc = 0;
                    for (int s = 0; s < K; ++s) {
                        sd += (double) ww[t * K + s] * (double) a[((size_t)(t * K + s)) * N2 + i];
                        sc += (double) ww[t * K + s] * (double) b[((size_t) pp[t * K + s]) * N2 + i];
                    }
                    std::printf("    t%d i%d: d_dec=%.9g d_chk=%.9g cpuD=%.9g cpuC=%.9g (x1=%.9g x2=%.9g)\n", t, i,
                                x1.as<float>()[t * N2 + i] - xi[t * N2 + i],
                                x2.as<float>()[t * N2 + i] - xi[t * N2 + i], sd, sc,
                                x1.as<float>()[t * N2 + i], x2.as<float>()[t * N2 + i]);
                }
        }
        // probe C: the same combine on a fresh memcpy-seeded buffer, as its own command buffer, after the
        // main graphs - separates "the combine + its bindings" from "x2's state at combine time".  x4 ==
        // x1's values means the memcpy-seeded base reached this combine (so the original x2 was zeroed in
        // between); x4 == x2's values means the memcpy base never reaches the GPU.
        {
            mtl::Buffer x4 = cx.alloc(4 * T * N2);
            std::memcpy(x4.ptr, x3.ptr, 4 * T * N2);
            mtl::Dispatch dsum2;
            dsum2.p = &p_sumc;
            dsum2.args = {mtl::Arg::B(x4), mtl::Arg::B(ey2), mtl::Arg::B(wts2), mtl::Arg::B(sy0), mtl::Arg::B(gs0),
                          mtl::Arg::B(pos2), mtl::Arg::V(K), mtl::Arg::V(N2), mtl::Arg::V(N2)};
            dsum2.grid[0] = T;
            dsum2.grid[1] = (N2 + 127) / 128;
            dsum2.tg[0] = 128;
            cx.run_graph({dsum2});
            std::printf("  moe2048 probeC: x4=%.9g/%.9g | x1=%.9g/%.9g | x2=%.9g/%.9g (initial %.9g)\n",
                        x4.as<float>()[0], x4.as<float>()[T * N2 - 1], x1.as<float>()[0],
                        x1.as<float>()[T * N2 - 1], x2.as<float>()[0], x2.as<float>()[T * N2 - 1],
                        x3.as<float>()[0]);
        }
        // the isolation combine, checked (the suite's 38th stage): a fresh buffer carrying the pristine
        // base through the chunk combine must reproduce x1's result bit for bit - same base bytes,
        // term-identical verified inputs (ey2 through pos2 == ey1, shared wts2), same summation order.
        // Permanent guard for the round-11 class: the gather's over-specified grid smashed the bases
        // mid-chain (the combine then read zeros and x failed with the sum-only values).
        {
            mtl::Buffer x5 = cx.alloc(4 * T * N2);
            std::memcpy(x5.ptr, x3.ptr, 4 * T * N2);
            mtl::Dispatch dsum3;
            dsum3.p = &p_sumc;
            dsum3.args = {mtl::Arg::B(x5), mtl::Arg::B(ey2), mtl::Arg::B(wts2), mtl::Arg::B(sy0), mtl::Arg::B(gs0),
                          mtl::Arg::B(pos2), mtl::Arg::V(K), mtl::Arg::V(N2), mtl::Arg::V(N2)};
            dsum3.grid[0] = T;
            dsum3.grid[1] = (N2 + 127) / 128;
            dsum3.tg[0] = 128;
            cx.run_graph({dsum3});
            check("moe2048 iso", x5.ptr, x1.ptr, 4 * T * N2);
        }
    }

    // ---- 39. kv-f16 cross (STRATA_Q36M_KV_F16=1 only): both libs' attention kernels against the same
    // VALUES in the two storages - the fp32 cache the default lib reads, the fp16 cache (the same values
    // as halves) the f16 lib reads.  On a half-exact cache (every value a float->half->float round trip)
    // the f16 kernels' unpack is exact, so k_attn and the split-K pair must produce bit-identical
    // outputs from both libs: the "only storage changed" proof.  On a random (non-representable) cache
    // the outputs differ by the input rounding only; the rel-L1 against the fp32 kernels is printed and
    // bounded at 1e-3 (the CUDA qsa_parity precedent's phase bound for FP16 paths, and its metric
    // sum |a-b| / sum |a|).  The stage prints one line (the suite's 39th in this mode).
    if (kf) {
        const int CTX = 64;
        const float scale = 1.0f / std::sqrt((float) 256);
        mtl::Pipeline p_attn_d = pipe("k_attn"), p_attn_f = cx.pipeline(lib_f16, "k_attn");
        mtl::Pipeline p_sk_d = pipe("k_attn_sk_part"), p_sk_f = cx.pipeline(lib_f16, "k_attn_sk_part");

        mtl::Buffer q_a = cx.alloc(4 * T * H * 256), gate_all = cx.alloc(4 * T * H * 256);
        for (int i = 0; i < T * H * 256; ++i) {
            q_a.as<float>()[i] = r.sym();
            gate_all.as<float>()[i] = r.sym();
        }
        mtl::Buffer ck_f32 = cx.alloc(4 * HKV * CTX * 256), cv_f32 = cx.alloc(4 * HKV * CTX * 256);
        mtl::Buffer ck_f16 = cx.alloc(2 * HKV * CTX * 256), cv_f16 = cx.alloc(2 * HKV * CTX * 256);
        mtl::Buffer att_d = cx.alloc(4 * T * H * 256), att_f = cx.alloc(4 * T * H * 256);
        mtl::Buffer pacc_s = cx.alloc(4 * 256 * 256), pm_s = cx.alloc(4 * 256), pl_s = cx.alloc(4 * 256);

        auto run_attn = [&](mtl::Pipeline& p, const mtl::Buffer& kc, const mtl::Buffer& vc, mtl::Buffer& out) {
            std::vector<mtl::Dispatch> ds;
            for (int t = 0; t < T; ++t) {
                mtl::Dispatch d;
                d.p = &p;
                d.args = {mtl::Arg::B(q_a, 4 * (size_t) t * H * 256), mtl::Arg::B(kc), mtl::Arg::B(vc),
                          mtl::Arg::B(gate_all, 4 * (size_t) t * H * 256), mtl::Arg::B(out, 4 * (size_t) t * H * 256),
                          mtl::Arg::V(H), mtl::Arg::V(HKV), mtl::Arg::V(256), mtl::Arg::V(t + 1), mtl::Arg::V(CTX),
                          mtl::Arg::V(scale)};
                d.grid[0] = H;
                d.tg[0] = 128;
                d.tg_mem = (uint32_t) ((t + 1) * sizeof(float));
                d.barrier = true;
                ds.push_back(d);
            }
            cx.run_graph(ds);
        };
        auto run_sk = [&](mtl::Pipeline& p, const mtl::Buffer& kc, const mtl::Buffer& vc, mtl::Buffer& out) {
            std::vector<mtl::Dispatch> ds;
            for (int t = 0; t < T; ++t) {
                mtl::Dispatch dp;
                dp.p = &p;
                dp.args = {mtl::Arg::B(q_a, 4 * (size_t) t * H * 256), mtl::Arg::B(kc), mtl::Arg::B(vc),
                           mtl::Arg::V(H), mtl::Arg::V(HKV), mtl::Arg::V(CTX), mtl::Arg::V(scale), mtl::Arg::V(1),
                           mtl::Arg::V(t), mtl::Arg::V(0), mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s)};
                dp.grid[0] = 1;
                dp.grid[1] = H;
                dp.tg[0] = 256;
                dp.barrier = true;
                ds.push_back(dp);
                mtl::Dispatch dm;
                dm.p = &p_sk_merge;   // the common merge: float partials, either lib's part can feed it
                dm.args = {mtl::Arg::B(pacc_s), mtl::Arg::B(pm_s), mtl::Arg::B(pl_s),
                           mtl::Arg::B(gate_all, 4 * (size_t) t * H * 256), mtl::Arg::B(out, 4 * (size_t) t * H * 256),
                           mtl::Arg::V(H), mtl::Arg::V(1), mtl::Arg::V(t), mtl::Arg::V(0)};
                dm.grid[0] = H;
                dm.tg[0] = 256;
                dm.barrier = true;
                ds.push_back(dm);
            }
            cx.run_graph(ds);
        };

        bool ok = true;
        const char* detail = "";
        // (a) half-exact values: bit-identity, both kernel pairs
        for (int i = 0; i < HKV * CTX * 256; ++i) {
            const float v = ref::f16_to_f32(ref::f32_to_f16(r.sym()));
            ck_f32.as<float>()[i] = v;
            cv_f32.as<float>()[i] = ref::f16_to_f32(ref::f32_to_f16(r.sym()));
            ((uint16_t*) ck_f16.ptr)[i] = ref::f32_to_f16(ck_f32.as<float>()[i]);
            ((uint16_t*) cv_f16.ptr)[i] = ref::f32_to_f16(cv_f32.as<float>()[i]);
        }
        run_attn(p_attn_d, ck_f32, cv_f32, att_d);
        run_attn(p_attn_f, ck_f16, cv_f16, att_f);
        if (std::memcmp(att_d.ptr, att_f.ptr, 4 * T * H * 256) != 0) {
            ok = false;
            detail = "k_attn outputs differ on the half-exact cache";
        }
        if (ok) {
            run_sk(p_sk_d, ck_f32, cv_f32, att_d);
            run_sk(p_sk_f, ck_f16, cv_f16, att_f);
            if (std::memcmp(att_d.ptr, att_f.ptr, 4 * T * H * 256) != 0) {
                ok = false;
                detail = "split-K outputs differ on the half-exact cache";
            }
        }
        // (b) random values: the cache rounding's own cost, the qsa_parity rel-L1 metric
        double rel = 0.0;
        if (ok) {
            for (int i = 0; i < HKV * CTX * 256; ++i) {
                const float k = r.sym(), v = r.sym();
                ck_f32.as<float>()[i] = k;
                cv_f32.as<float>()[i] = v;
                ((uint16_t*) ck_f16.ptr)[i] = ref::f32_to_f16(k);
                ((uint16_t*) cv_f16.ptr)[i] = ref::f32_to_f16(v);
            }
            for (int pair = 0; pair < 2; ++pair) {
                if (pair == 0) {
                    run_attn(p_attn_d, ck_f32, cv_f32, att_d);
                    run_attn(p_attn_f, ck_f16, cv_f16, att_f);
                } else {
                    run_sk(p_sk_d, ck_f32, cv_f32, att_d);
                    run_sk(p_sk_f, ck_f16, cv_f16, att_f);
                }
                double num = 0, den = 0;
                for (int i = 0; i < T * H * 256; ++i) {
                    num += std::fabs((double) att_f.as<float>()[i] - (double) att_d.as<float>()[i]);
                    den += std::fabs((double) att_d.as<float>()[i]);
                }
                const double r1 = num / (den > 1e-30 ? den : 1e-30);
                if (r1 > rel) rel = r1;
            }
            if (rel > 1e-3) {
                ok = false;
                detail = "random-cache rel-L1 over the 1e-03 bound";
            }
        }
        if (ok) std::printf("  %-14s OK (random-cache rel-L1 %.3g <= 1e-03)\n", "kv f16 cross", rel);
        else {
            std::printf("  %-14s FAIL: %s\n", "kv f16 cross", detail);
            ++g_fails;
        }
    }

    std::printf(g_fails ? "FAILURES: %d\n" : "all stages bit-identical\n", g_fails);
    return g_fails ? 1 : 0;
}
