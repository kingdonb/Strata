// metal/tests/q36_chunk_parity.cpp - the prompt-chunk kernels (q36.metal) against the decode kernels they
// must match bit for bit, stage by stage, on synthetic inputs: the engine's chunked prefill keeps the
// token-by-token path's exact arithmetic, so every stage compares equal with memcmp.
//
//   q36_chunk_parity    embed rows, dense gemv (F32 / Q8_0 / Q4_K, both tpb shapes), the n_in=2048 lpr8
//                       gemv, top-k routing, the expert entry tables + grouped expert gemv, conv tiles,
//                       the delta rule, gated norm, the MoE combine
#include "mtl_rt.h"
#include "ref_quant.h"

#include <cmath>
#include <cstdio>
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

void check(const char* stage, const void* a, const void* b, size_t bytes) {
    if (std::memcmp(a, b, bytes) == 0) {
        std::printf("  %-14s OK\n", stage);
        return;
    }
    const float* x = (const float*) a;
    const float* y = (const float*) b;
    const size_t n = bytes / 4;
    size_t first = n, count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (x[i] != y[i]) {
            if (first == n) first = i;
            ++count;
        }
    }
    std::printf("  %-14s FAIL: %zu/%zu differ, first at %zu (%.9g vs %.9g)", stage, count, n, first,
                first < n ? x[first] : 0.0f, first < n ? y[first] : 0.0f);
    size_t shown = 0;
    for (size_t i = first; i < n && shown < 4; ++i)
        if (x[i] != y[i]) {
            std::printf("; [%zu] %.9g vs %.9g", i, x[i], y[i]);
            ++shown;
        }
    std::printf("\n");
    ++g_fails;
}

// the expert gemv's outputs live entry-major on the chunk side and per (token, slot) on the decode side
void check_scattered(const char* stage, const float* decode_rows, const float* entry_rows, const int* pos) {
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < K; ++k) {
            const float* a = decode_rows + (size_t)(t * K + k) * FF;
            const float* b = entry_rows + (size_t) pos[t * K + k] * FF;
            if (std::memcmp(a, b, 4 * FF) != 0) {
                std::printf("  %-14s FAIL at token %d slot %d\n", stage, t, k);
                ++g_fails;
                return;
            }
        }
    }
    std::printf("  %-14s OK\n", stage);
}

}  // namespace

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"quant.metal", "q36.metal"});
    std::printf("device %s\n", cx.device_name().c_str());
    const float eps = 1e-6f;

    auto pipe = [&](const char* n) { return cx.pipeline(lib, n); };
    mtl::Pipeline p_row = pipe("k_row_T_Q8_0"), p_emb = pipe("k_emb_chunk_T_Q8_0");
    mtl::Pipeline p_g_f32 = pipe("k_gemv_T_F32"), p_g_q8 = pipe("k_gemv_T_Q8_0"), p_g_q4k = pipe("k_gemv_T_Q4_K");
    mtl::Pipeline p_gc_f32 = pipe("k_gemv_c_T_F32"), p_gc_q8 = pipe("k_gemv_c_T_Q8_0"), p_gc_q4k = pipe("k_gemv_c_T_Q4_K");
    mtl::Pipeline p_lpr8 = pipe("k_gemv_lpr8_T_Q4_K"), p_lpr8c = pipe("k_gemv_lpr8_c_T_Q4_K");
    mtl::Pipeline p_topk = pipe("k_topk_router"), p_route = pipe("k_route_chunk");
    mtl::Pipeline p_zero = pipe("k_zero"), p_hist = pipe("k_hist"), p_off = pipe("k_exp_off"), p_place = pipe("k_place");
    mtl::Pipeline p_expc = pipe("k_expc_T_Q4_K");
    mtl::Pipeline p_conv = pipe("k_conv"), p_convc = pipe("k_conv_chunk");
    mtl::Pipeline p_gdn = pipe("k_gdn"), p_gdnc = pipe("k_gdn_chunk");
    mtl::Pipeline p_gn = pipe("k_gated_norm"), p_gnc = pipe("k_gated_norm_chunk");
    mtl::Pipeline p_sum = pipe("k_moe_sum"), p_sumc = pipe("k_moe_sum_chunk");
    mtl::Pipeline p_prep = pipe("k_attn_prep"), p_prep_c = pipe("k_attn_prep_chunk");
    mtl::Pipeline p_sk_part = pipe("k_attn_sk_part"), p_sk_merge = pipe("k_attn_sk_merge");

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

    // ---- 5. the n_in = 2048 lpr8 gemv (Q4_K), tpb = T and tpb = 2 ------------------------------------------
    {
        const int NIN = 2048, NOUT = 64;
        const uint64_t RB = 8ull * 144;
        mtl::Buffer hin = cx.alloc(4 * T * NIN);
        for (int i = 0; i < T * NIN; ++i) hin.as<float>()[i] = r.sym();
        mtl::Buffer W = cx.alloc(RB * NOUT);
        for (size_t i = 0; i < W.bytes / 4; ++i) W.as<float>()[i] = r.sym();
        mtl::Buffer y1 = cx.alloc(4 * T * NOUT), y2 = cx.alloc(4 * T * NOUT), y3 = cx.alloc(4 * T * NOUT);
        for (int pass = 0; pass < 2; ++pass) {
            const int tpb = pass == 0 ? T : 2;
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
            d.p = &p_lpr8c;
            d.args = {mtl::Arg::B(W), mtl::Arg::B(hin), mtl::Arg::B(y2), mtl::Arg::V(p), mtl::Arg::V(tpb),
                      mtl::Arg::V(0), mtl::Arg::V(T)};
            d.grid[0] = (NOUT + 15) / 16;
            d.grid[1] = (T + tpb - 1) / tpb;
            d.tg[0] = 128;
            cx.run_graph({d});
            // run the chunk kernel again into y3: a run-to-run difference would be a race, not arithmetic
            mtl::Dispatch d2 = d;
            d2.args = {mtl::Arg::B(W), mtl::Arg::B(hin), mtl::Arg::B(y3), mtl::Arg::V(p), mtl::Arg::V(tpb),
                       mtl::Arg::V(0), mtl::Arg::V(T)};
            cx.run_graph({d2});
            char name[32];
            std::snprintf(name, sizeof(name), "lpr8 q4_K tpb=%d", tpb);
            check(name, y1.ptr, y2.ptr, 4 * T * NOUT);
            check("lpr8 rerun", y2.ptr, y3.ptr, 4 * T * NOUT);
        }
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
        mtl::Buffer kc1 = cx.alloc(4 * HKV * CTX * HD), vc1 = cx.alloc(4 * HKV * CTX * HD);
        mtl::Buffer kc2 = cx.alloc(4 * HKV * CTX * HD), vc2 = cx.alloc(4 * HKV * CTX * HD);
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
        check("prep kc", kc1.ptr, kc2.ptr, 4 * HKV * CTX * HD);
        check("prep vc", vc1.ptr, vc2.ptr, 4 * HKV * CTX * HD);
    }

    // ---- 13. split-K attention rows: per-token form vs the row-blocked chunk form (row0 = 0 and 16) ----------
    {
        const int CTX = 64;
        const float scale = 1.0f / std::sqrt((float) 256);
        mtl::Buffer q_a = cx.alloc(4 * T * H * 256);
        for (int i = 0; i < T * H * 256; ++i) q_a.as<float>()[i] = r.sym();
        mtl::Buffer kc = cx.alloc(4 * HKV * CTX * 256), vc = cx.alloc(4 * HKV * CTX * 256);
        for (int i = 0; i < HKV * CTX * 256; ++i) {
            kc.as<float>()[i] = r.sym();
            vc.as<float>()[i] = r.sym();
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

    std::printf(g_fails ? "FAILURES: %d\n" : "all stages bit-identical\n", g_fails);
    return g_fails ? 1 : 0;
}
