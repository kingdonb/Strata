// metal/src/engine/q36_metal.cpp - Metal execution engine for Qwen3.6-35B-A3B (qwen35moe).
#include "strata/qwen35/engine.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "mtl_rt.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::qwen35 {

namespace {

enum : int {
    T_F32 = 0,
    T_F16 = 1,
    T_Q8_0 = 8,
    T_Q3_K = 11,
    T_Q4_K = 12,
    T_Q5_K = 13,
    T_Q6_K = 14,
    T_IQ2_XS = 17,
    T_IQ3_XXS = 18,
    T_IQ4_XS = 23,
    T_BF16 = 30
};

struct GemvParam {
    uint64_t row_bytes = 0;
    uint64_t expert_bytes = 0;
    int32_t n_in = 0;
    int32_t n_out = 0;
    uint64_t x_stride = 0;
};

struct Tensor {
    mtl::Buffer b;
    size_t offset = 0;
    int type = -1;
    std::vector<int64_t> shape;
    size_t row_bytes = 0;
};

size_t row_bytes_of(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float h2f_host(uint16_t h) {
    const uint32_t s = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp((float) m, -24);
    const uint32_t f = e == 31 ? (s | 0x7f800000u | (m << 13)) : (s | ((e + 112) << 23) | (m << 13));
    float r;
    std::memcpy(&r, &f, 4);
    return r;
}

struct Model {
    strata::core::ModelGeometry g;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    std::map<std::string, Tensor> t;
    std::map<std::string, mtl::Buffer> f32;

    const Tensor& W(const std::string& n) const {
        auto it = t.find(n);
        if (it == t.end()) {
            std::fprintf(stderr, "q36_metal: missing tensor %s\n", n.c_str());
            std::exit(1);
        }
        return it->second;
    }
    const mtl::Buffer& F(const std::string& n) const {
        auto it = f32.find(n);
        if (it == f32.end()) {
            std::fprintf(stderr, "q36_metal: missing vector %s\n", n.c_str());
            std::exit(1);
        }
        return it->second;
    }
};

struct State {
    int ctx = 4096, pos = 0;
    mtl::Buffer x, h, y, qkv, z, alpha, beta, conv_out, gdn_out, qg, k, v, q, gate, att, router, eg, eu, eh, ey, sg, su, sh, sy, sgate, wts, logits;
    mtl::Buffer ids;
    mtl::Buffer sk_pacc, sk_pm, sk_pl;   // split-K attention partials (k_attn_sk_part / k_attn_sk_merge)
    // prompt-chunk buffers (allocated when --prefill is on; CT = the chunk size)
    mtl::Buffer toks_c, xc, hc, yc, qkv_c, z_c, alpha_c, beta_c, conv_out_c, gdn_out_c, qg_c, k_c, v_c, q_c, gate_c, att_c;
    mtl::Buffer router_c, ids_c, wts_c, scr, srt_in_gu, srt_in_d, pos_c;
    mtl::Buffer eg_c, eu_c, eh_c, ey_c, sg_c, su_c, sh_c, sy_c, sgate_c;
    std::vector<mtl::Buffer> conv_state, S, kc, vc;

    void reset() {
        pos = 0;
        for (auto& b : conv_state) if (b.ptr) std::memset(b.ptr, 0, b.bytes);
        for (auto& b : S) if (b.ptr) std::memset(b.ptr, 0, b.bytes);
        for (auto& b : kc) if (b.ptr) std::memset(b.ptr, 0, b.bytes);
        for (auto& b : vc) if (b.ptr) std::memset(b.ptr, 0, b.bytes);
    }
};

std::string L(int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; }

bool env_flag(const char* name) {
    const char* e = std::getenv(name);
    return e && *e && std::atoi(e) != 0;
}

// Debug-trace sums (STRATA_Q36M_TRACE=1): rows summed in order, each row's sum accumulated in index order.
// The decode path prints one line per (token, cut); tools/qwen35/q36_trace_cmp.py sums those over the
// chunk's token range in token order, which reproduces the chunk path's accumulation exactly - so equal
// states print bit-equal doubles, and the first mismatching cut names the diverging dispatch.
static double trace_rows(const float* p, size_t rows, size_t width) {
    double s = 0.0;
    for (size_t r = 0; r < rows; ++r) {
        double rs = 0.0;
        for (size_t i = 0; i < width; ++i) rs += p[r * width + i];
        s += rs;
    }
    return s;
}
// an entry-major buffer's rows in (token, slot) order through the placement table
static double trace_rows_pos(const float* p, const int* pos, int n_ent, size_t width) {
    double s = 0.0;
    for (int i = 0; i < n_ent; ++i) {
        const float* row = p + (size_t) pos[i] * width;
        double rs = 0.0;
        for (size_t j = 0; j < width; ++j) rs += row[j];
        s += rs;
    }
    return s;
}

// k_attn's scores are threadgroup memory, capped at 32 KB: 8,192 cells.  Above that (or forced by
// STRATA_Q36M_ATTN_SK, the parity knob) the split-K kernels take over; SK_HD must be the head_dim.
constexpr int SK_CELLS = 256;
constexpr int SK_ROWS_MAX = 256;   // query rows per split-K dispatch (decode: n_head; prompt chunks row-block)
constexpr int SK_TG_CAP = 8192;    // the longest context the plain k_attn path serves

struct Pipelines {
    int lib = -1;
    mtl::Buffer dummy_ids;
    mtl::Pipeline p_rms, p_conv, p_gdn_gb, p_gdn, p_gated_norm, p_attn_prep, p_attn, p_swiglu, p_add, p_moe_sum, p_topk_router;
    mtl::Pipeline p_attn_sk_part, p_attn_sk_merge;
    // prompt-chunk pipelines (init_prompt, when --prefill is on)
    std::map<int, mtl::Pipeline> p_emb_c, p_gemv_c, p_expc;
    mtl::Pipeline p_route_c, p_tables, p_moe_sum_c, p_conv_c, p_gdn_c, p_gated_norm_c, p_attn_prep_c;
    mtl::Buffer zero_ids;   // nt zeros: the decode lpr8's per-token dummy expert id (chunk dense products)
    std::map<int, mtl::Pipeline> p_gemv;
    std::map<int, mtl::Pipeline> p_gemv_lpr8;
    std::map<int, mtl::Pipeline> p_row;

    void init(mtl::Context& cx) {
        dummy_ids = cx.alloc(sizeof(int));
        *(dummy_ids.as<int>()) = 0;

        lib = cx.compile({"quant.metal", "q36.metal"});
        p_rms = cx.pipeline(lib, "k_rms");
        p_conv = cx.pipeline(lib, "k_conv");
        p_gdn_gb = cx.pipeline(lib, "k_gdn_gb");
        p_gdn = cx.pipeline(lib, "k_gdn");
        p_gated_norm = cx.pipeline(lib, "k_gated_norm");
        p_attn_prep = cx.pipeline(lib, "k_attn_prep");
        p_attn = cx.pipeline(lib, "k_attn");
        p_attn_sk_part = cx.pipeline(lib, "k_attn_sk_part");
        p_attn_sk_merge = cx.pipeline(lib, "k_attn_sk_merge");
        p_swiglu = cx.pipeline(lib, "k_swiglu");
        p_add = cx.pipeline(lib, "k_add");
        p_moe_sum = cx.pipeline(lib, "k_moe_sum");
        p_topk_router = cx.pipeline(lib, "k_topk_router");

        auto get_gemv = [&](int ty, const char* fn) {
            p_gemv[ty] = cx.pipeline(lib, fn, mtl::Constants().b(23, true));
        };
        auto get_gemv_lpr8 = [&](int ty, const char* fn) {
            p_gemv_lpr8[ty] = cx.pipeline(lib, fn, mtl::Constants().b(23, true));
        };
        auto get_row = [&](int ty, const char* fn) {
            p_row[ty] = cx.pipeline(lib, fn, mtl::Constants().b(23, true));
        };

        get_gemv(T_F32, "k_gemv_T_F32");
        get_gemv(T_BF16, "k_gemv_T_BF16");
        get_gemv(T_Q8_0, "k_gemv_T_Q8_0");
        get_gemv(T_Q3_K, "k_gemv_T_Q3_K");
        get_gemv(T_Q4_K, "k_gemv_T_Q4_K");
        get_gemv(T_Q5_K, "k_gemv_T_Q5_K");
        get_gemv(T_Q6_K, "k_gemv_T_Q6_K");
        get_gemv(T_IQ2_XS, "k_gemv_T_IQ2_XS");
        get_gemv(T_IQ3_XXS, "k_gemv_T_IQ3_XXS");
        get_gemv(T_IQ4_XS, "k_gemv_T_IQ4_XS");

        get_gemv_lpr8(T_Q4_K, "k_gemv_lpr8_T_Q4_K");
        get_gemv_lpr8(T_Q5_K, "k_gemv_lpr8_T_Q5_K");
        get_gemv_lpr8(T_Q6_K, "k_gemv_lpr8_T_Q6_K");
        get_gemv_lpr8(T_IQ2_XS, "k_gemv_lpr8_T_IQ2_XS");

        get_row(T_F32, "k_row_T_F32");
        get_row(T_BF16, "k_row_T_BF16");
        get_row(T_Q8_0, "k_row_T_Q8_0");
        get_row(T_Q3_K, "k_row_T_Q3_K");
        get_row(T_Q4_K, "k_row_T_Q4_K");
        get_row(T_Q5_K, "k_row_T_Q5_K");
        get_row(T_Q6_K, "k_row_T_Q6_K");
        get_row(T_IQ2_XS, "k_row_T_IQ2_XS");
        get_row(T_IQ3_XXS, "k_row_T_IQ3_XXS");
        get_row(T_IQ4_XS, "k_row_T_IQ4_XS");
    }

    // The prompt-chunk kernels (k_*_chunk / k_gemv_c / k_expc): compiled and created only when --prefill
    // turns the chunk path on, so the plain engine's startup does less and its behavior is unchanged.
    void init_prompt(mtl::Context& cx, int ct) {
        zero_ids = cx.alloc((size_t) ct * sizeof(int));   // alloc zeroes it
        auto get_emb_c = [&](int ty, const char* fn) { p_emb_c[ty] = cx.pipeline(lib, fn); };
        auto get_gemv_c = [&](int ty, const char* fn) { p_gemv_c[ty] = cx.pipeline(lib, fn); };
        auto get_expc = [&](int ty, const char* fn) { p_expc[ty] = cx.pipeline(lib, fn); };

        get_emb_c(T_F32, "k_emb_chunk_T_F32");
        get_emb_c(T_BF16, "k_emb_chunk_T_BF16");
        get_emb_c(T_Q8_0, "k_emb_chunk_T_Q8_0");
        get_emb_c(T_Q3_K, "k_emb_chunk_T_Q3_K");
        get_emb_c(T_Q4_K, "k_emb_chunk_T_Q4_K");
        get_emb_c(T_Q5_K, "k_emb_chunk_T_Q5_K");
        get_emb_c(T_Q6_K, "k_emb_chunk_T_Q6_K");
        get_emb_c(T_IQ2_XS, "k_emb_chunk_T_IQ2_XS");
        get_emb_c(T_IQ3_XXS, "k_emb_chunk_T_IQ3_XXS");
        get_emb_c(T_IQ4_XS, "k_emb_chunk_T_IQ4_XS");

        get_gemv_c(T_F32, "k_gemv_c_T_F32");
        get_gemv_c(T_BF16, "k_gemv_c_T_BF16");
        get_gemv_c(T_Q8_0, "k_gemv_c_T_Q8_0");
        get_gemv_c(T_Q3_K, "k_gemv_c_T_Q3_K");
        get_gemv_c(T_Q4_K, "k_gemv_c_T_Q4_K");
        get_gemv_c(T_Q5_K, "k_gemv_c_T_Q5_K");
        get_gemv_c(T_Q6_K, "k_gemv_c_T_Q6_K");
        get_gemv_c(T_IQ2_XS, "k_gemv_c_T_IQ2_XS");
        get_gemv_c(T_IQ3_XXS, "k_gemv_c_T_IQ3_XXS");
        get_gemv_c(T_IQ4_XS, "k_gemv_c_T_IQ4_XS");

        get_expc(T_F32, "k_expc_T_F32");
        get_expc(T_BF16, "k_expc_T_BF16");
        get_expc(T_Q8_0, "k_expc_T_Q8_0");
        get_expc(T_Q3_K, "k_expc_T_Q3_K");
        get_expc(T_Q4_K, "k_expc_T_Q4_K");
        get_expc(T_Q5_K, "k_expc_T_Q5_K");
        get_expc(T_Q6_K, "k_expc_T_Q6_K");
        get_expc(T_IQ2_XS, "k_expc_T_IQ2_XS");
        get_expc(T_IQ3_XXS, "k_expc_T_IQ3_XXS");
        get_expc(T_IQ4_XS, "k_expc_T_IQ4_XS");

        p_route_c = cx.pipeline(lib, "k_route_chunk");
        p_tables = cx.pipeline(lib, "k_tables");
        p_moe_sum_c = cx.pipeline(lib, "k_moe_sum_chunk");
        p_conv_c = cx.pipeline(lib, "k_conv_chunk");
        p_gdn_c = cx.pipeline(lib, "k_gdn_chunk");
        p_gated_norm_c = cx.pipeline(lib, "k_gated_norm_chunk");
        p_attn_prep_c = cx.pipeline(lib, "k_attn_prep_chunk");
    }
};

void push_gemv(std::vector<mtl::Dispatch>& ds, const Pipelines& pipes, const Tensor& w, const mtl::Buffer& x, const mtl::Buffer& y,
               const mtl::Buffer* ids = nullptr, int n_sel = 1, size_t x_stride = 0, bool barrier = true) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    const size_t eb = w.shape.size() == 3 ? w.row_bytes * (size_t) w.shape[1] : 0;
    auto it = pipes.p_gemv.find(w.type);
    if (it == pipes.p_gemv.end()) {
        std::fprintf(stderr, "q36_metal: unhandled GEMV type %d\n", w.type);
        std::exit(1);
    }
    GemvParam p;
    p.row_bytes = w.row_bytes;
    p.expert_bytes = eb;
    p.n_in = n_in;
    p.n_out = n_out;
    p.x_stride = x_stride;

    mtl::Dispatch d;
    auto it_lpr8 = (n_in == 2048) ? pipes.p_gemv_lpr8.find(w.type) : pipes.p_gemv_lpr8.end();
    if (it_lpr8 != pipes.p_gemv_lpr8.end()) {
        d.p = &it_lpr8->second;
        d.grid[0] = (n_out + 15) / 16;
        d.grid[1] = n_sel;
        d.tg[0] = 128;
    } else {
        d.p = &it->second;
        d.grid[0] = (n_out + 7) / 8;
        d.grid[1] = n_sel;
        d.tg[0] = 256;
    }
    d.barrier = barrier;
    d.args = {mtl::Arg::B(w.b, w.offset), mtl::Arg::B(x), mtl::Arg::B(y),
              mtl::Arg::B(ids ? *ids : pipes.dummy_ids), mtl::Arg::V(p)};
    ds.push_back(d);
}

void forward_token(mtl::Context& cx, const Pipelines& pipes, const Model& m, State& s, int tok, std::vector<int>& ids_host, bool sk_force = false, bool prompt_step = false) {
    const auto& g = m.g;
    const int N = (int) g.n_embd, K = (int) m.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const bool tr = prompt_step && env_flag("STRATA_Q36M_TRACE");   // debug: per-layer sub-graphs + checksums

    std::vector<mtl::Dispatch> ds;
    if (tr && s.pos == 0) std::fprintf(stderr, "TRACE BEGIN D\n");

    // Embedding lookup
    {
        const Tensor& e = m.W("token_embd.weight");
        auto it = pipes.p_row.find(e.type);
        if (it == pipes.p_row.end()) {
            std::fprintf(stderr, "q36_metal: unhandled token_embd type %d\n", e.type);
            std::exit(1);
        }
        const uint64_t rb = e.row_bytes;
        const int tok_row = tok;
        const int n_emb = N;
        mtl::Dispatch d;
        d.p = &it->second;
        d.args = {mtl::Arg::B(e.b, e.offset), mtl::Arg::B(s.x), mtl::Arg::V(rb), mtl::Arg::V(tok_row), mtl::Arg::V(n_emb)};
        d.grid[0] = 16;
        d.tg[0] = 128;
        ds.push_back(d);
    }
    if (tr) {   // debug cut G0: the embedding rows (type= prints the token_embd GGUF type id)
        cx.run_graph(ds);
        ds.clear();
        std::fprintf(stderr, "TRACE G0 pos=%d L=-1 x=%.17g embtype=%d\n", (int) s.pos,
                     trace_rows((const float*) s.x.ptr, 1, (size_t) N), (int) m.W("token_embd.weight").type);
    }

    for (int64_t l = 0; l < g.n_layers; ++l) {
        // RMSNorm
        {
            const int n_val = N;
            const float eps_val = m.eps;
            mtl::Dispatch d;
            d.p = &pipes.p_rms;
            d.args = {mtl::Arg::B(s.x), mtl::Arg::B(m.F(L(l, "attn_norm.weight"))), mtl::Arg::B(s.h), mtl::Arg::V(n_val), mtl::Arg::V(eps_val)};
            d.grid[0] = 1;
            d.tg[0] = 128;
            ds.push_back(d);
        }

        if (!strata::core::is_qsa_layer(g, l)) {
            const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, D = (int) g.ssm_state_size;
            const int C = (int) g.ssm_conv_channels;
            push_gemv(ds, pipes, m.W(L(l, "attn_qkv.weight")), s.h, s.qkv, nullptr, 1, 0, true);
            push_gemv(ds, pipes, m.W(L(l, "attn_gate.weight")), s.h, s.z, nullptr, 1, 0, false);
            push_gemv(ds, pipes, m.W(L(l, "ssm_beta.weight")), s.h, s.beta, nullptr, 1, 0, false);
            push_gemv(ds, pipes, m.W(L(l, "ssm_alpha.weight")), s.h, s.alpha, nullptr, 1, 0, false);
            if (tr && l == 0) {   // debug cut G1: the four projections
                cx.run_graph(ds);
                ds.clear();
                std::fprintf(stderr, "TRACE G1 pos=%d L=0 qkv=%.17g z=%.17g al=%.17g be=%.17g\n", (int) s.pos,
                             trace_rows((const float*) s.qkv.ptr, 1, (size_t) C),
                             trace_rows((const float*) s.z.ptr, 1, (size_t) g.ssm_value_dim),
                             trace_rows((const float*) s.alpha.ptr, 1, (size_t) HV),
                             trace_rows((const float*) s.beta.ptr, 1, (size_t) HV));
            }

            // Conv
            {
                const int C_val = C, K_val = (int) g.ssm_d_conv;
                mtl::Dispatch d;
                d.p = &pipes.p_conv;
                d.args = {mtl::Arg::B(s.qkv), mtl::Arg::B(s.conv_state[l]), mtl::Arg::B(m.F(L(l, "ssm_conv1d.weight"))),
                          mtl::Arg::B(s.conv_out), mtl::Arg::V(C_val), mtl::Arg::V(K_val)};
                d.grid[0] = (C + 127) / 128;
                d.tg[0] = 128;
                d.barrier = true;
                ds.push_back(d);
            }
            if (tr && l == 0) {   // debug cut G2: the conv outputs + the conv history it left
                cx.run_graph(ds);
                ds.clear();
                std::fprintf(stderr, "TRACE G2 pos=%d L=0 co=%.17g cst=%.17g\n", (int) s.pos,
                             trace_rows((const float*) s.conv_out.ptr, 1, (size_t) C),
                             trace_rows((const float*) s.conv_state[l].ptr, 1, s.conv_state[l].bytes / 4));
            }
            // GDN step
            {
                const int HK_val = HK, D_val = D;
                const float eps_val = m.eps;
                mtl::Dispatch d;
                d.p = &pipes.p_gdn;
                d.args = {mtl::Arg::B(s.conv_out), mtl::Arg::B(s.alpha), mtl::Arg::B(m.F(L(l, "ssm_dt.bias"))),
                          mtl::Arg::B(m.F(L(l, "ssm_a"))), mtl::Arg::B(s.beta), mtl::Arg::B(s.S[l]),
                          mtl::Arg::B(s.gdn_out), mtl::Arg::V(HK_val), mtl::Arg::V(D_val), mtl::Arg::V(eps_val)};
                d.grid[0] = HV;
                d.tg[0] = 128;
                d.barrier = true;
                ds.push_back(d);
            }
            if (tr && l == 0) {   // debug cut G3: the delta rule's outputs + the state it left
                cx.run_graph(ds);
                ds.clear();
                std::fprintf(stderr, "TRACE G3 pos=%d L=0 go=%.17g sst=%.17g\n", (int) s.pos,
                             trace_rows((const float*) s.gdn_out.ptr, 1, (size_t) g.ssm_value_dim),
                             trace_rows((const float*) s.S[l].ptr, 1, s.S[l].bytes / 4));
            }
            // Gated norm
            {
                const int D_val = D;
                const float eps_val = m.eps;
                mtl::Dispatch d;
                d.p = &pipes.p_gated_norm;
                d.args = {mtl::Arg::B(s.gdn_out), mtl::Arg::B(m.F(L(l, "ssm_norm.weight"))), mtl::Arg::B(s.z),
                          mtl::Arg::V(D_val), mtl::Arg::V(eps_val)};
                d.grid[0] = HV;
                d.tg[0] = 128;
                d.barrier = true;
                ds.push_back(d);
            }
            push_gemv(ds, pipes, m.W(L(l, "ssm_out.weight")), s.gdn_out, s.y, nullptr, 1, 0, true);
            if (tr && l == 0) {   // debug cut G4: the gated norm's z and the mixer output y
                cx.run_graph(ds);
                ds.clear();
                std::fprintf(stderr, "TRACE G4 pos=%d L=0 z=%.17g y=%.17g\n", (int) s.pos,
                             trace_rows((const float*) s.z.ptr, 1, (size_t) g.ssm_value_dim),
                             trace_rows((const float*) s.y.ptr, 1, (size_t) N));
            }
        } else {
            const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
            push_gemv(ds, pipes, m.W(L(l, "attn_q.weight")), s.h, s.qg, nullptr, 1, 0, true);
            push_gemv(ds, pipes, m.W(L(l, "attn_k.weight")), s.h, s.k, nullptr, 1, 0, false);
            push_gemv(ds, pipes, m.W(L(l, "attn_v.weight")), s.h, s.v, nullptr, 1, 0, false);

            // Attn prep
            {
                const int H_val = H, D_val = D, n_rot_val = m.n_rot, pos_val = s.pos, ctx_val = s.ctx;
                const float base_val = m.rope_base, eps_val = m.eps;
                mtl::Dispatch d;
                d.p = &pipes.p_attn_prep;
                d.args = {mtl::Arg::B(s.qg), mtl::Arg::B(s.k), mtl::Arg::B(s.v),
                          mtl::Arg::B(m.F(L(l, "attn_q_norm.weight"))), mtl::Arg::B(m.F(L(l, "attn_k_norm.weight"))),
                          mtl::Arg::B(s.q), mtl::Arg::B(s.gate), mtl::Arg::B(s.kc[l]), mtl::Arg::B(s.vc[l]),
                          mtl::Arg::V(H_val), mtl::Arg::V(D_val), mtl::Arg::V(n_rot_val), mtl::Arg::V(pos_val),
                          mtl::Arg::V(ctx_val), mtl::Arg::V(base_val), mtl::Arg::V(eps_val)};
                d.grid[0] = H + HKV;
                d.tg[0] = 128;
                d.barrier = true;
                ds.push_back(d);
            }
            // Attn.  Above the 8,192-cell threadgroup ceiling (or with STRATA_Q36M_ATTN_SK=1, the parity
            // knob) the split-K kernels run instead: 256-cell partials + merge, q36_engine.cu's math.
            {
                const int H_val = H, HKV_val = HKV, D_val = D, n_pos_val = s.pos + 1, ctx_val = s.ctx;
                const float scale_val = 1.0f / std::sqrt((float) D);
                if (sk_force || n_pos_val > SK_TG_CAP) {
                    if (D_val != 256) {
                        std::fprintf(stderr, "q36_metal: split-K attention needs head_dim 256, model has %d\n", D_val);
                        std::exit(1);
                    }
                    const int n_chunk = (n_pos_val + SK_CELLS - 1) / SK_CELLS;
                    mtl::Dispatch d;
                    d.p = &pipes.p_attn_sk_part;
                    d.args = {mtl::Arg::B(s.q), mtl::Arg::B(s.kc[l]), mtl::Arg::B(s.vc[l]),
                              mtl::Arg::V(H_val), mtl::Arg::V(HKV_val), mtl::Arg::V(ctx_val), mtl::Arg::V(scale_val),
                              mtl::Arg::V(n_chunk), mtl::Arg::V(s.pos), mtl::Arg::V(0),
                              mtl::Arg::B(s.sk_pacc), mtl::Arg::B(s.sk_pm), mtl::Arg::B(s.sk_pl)};
                    d.grid[0] = (uint32_t) n_chunk;
                    d.grid[1] = (uint32_t) H_val;
                    d.tg[0] = SK_CELLS;
                    d.barrier = true;
                    ds.push_back(d);
                    mtl::Dispatch dm;
                    dm.p = &pipes.p_attn_sk_merge;
                    dm.args = {mtl::Arg::B(s.sk_pacc), mtl::Arg::B(s.sk_pm), mtl::Arg::B(s.sk_pl), mtl::Arg::B(s.gate), mtl::Arg::B(s.att),
                               mtl::Arg::V(H_val), mtl::Arg::V(n_chunk), mtl::Arg::V(s.pos), mtl::Arg::V(0)};
                    dm.grid[0] = (uint32_t) H_val;
                    dm.tg[0] = SK_CELLS;
                    dm.barrier = true;
                    ds.push_back(dm);
                } else {
                    mtl::Dispatch d;
                    d.p = &pipes.p_attn;
                    d.args = {mtl::Arg::B(s.q), mtl::Arg::B(s.kc[l]), mtl::Arg::B(s.vc[l]), mtl::Arg::B(s.gate),
                              mtl::Arg::B(s.att), mtl::Arg::V(H_val), mtl::Arg::V(HKV_val), mtl::Arg::V(D_val),
                              mtl::Arg::V(n_pos_val), mtl::Arg::V(ctx_val), mtl::Arg::V(scale_val)};
                    d.grid[0] = H;
                    d.tg[0] = 128;
                    d.tg_mem = (s.pos + 1) * sizeof(float);
                    d.barrier = true;
                    ds.push_back(d);
                }
            }
            push_gemv(ds, pipes, m.W(L(l, "attn_output.weight")), s.att, s.y, nullptr, 1, 0, true);
        }

        // Add
        {
            const int N_val = N;
            mtl::Dispatch d;
            d.p = &pipes.p_add;
            d.args = {mtl::Arg::B(s.x), mtl::Arg::B(s.y), mtl::Arg::V(N_val)};
            d.grid[0] = (N + 127) / 128;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        if (tr && l == 0) {   // debug cut G5: the residual after the mixer's add
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G5 pos=%d L=0 x=%.17g\n", (int) s.pos,
                         trace_rows((const float*) s.x.ptr, 1, (size_t) N));
        }

        // Post norm
        {
            const int n_val = N;
            const float eps_val = m.eps;
            mtl::Dispatch d;
            d.p = &pipes.p_rms;
            d.args = {mtl::Arg::B(s.x), mtl::Arg::B(m.F(L(l, "post_attention_norm.weight"))), mtl::Arg::B(s.h), mtl::Arg::V(n_val), mtl::Arg::V(eps_val)};
            d.grid[0] = 1;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        if (tr && l == 0) {   // debug cut G6: the post-norm rows the MoE reads
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G6 pos=%d L=0 h=%.17g\n", (int) s.pos,
                         trace_rows((const float*) s.h.ptr, 1, (size_t) N));
        }

        // Router
        push_gemv(ds, pipes, m.W(L(l, "ffn_gate_inp.weight")), s.h, s.router, nullptr, 1, 0, true);

        // Top-K routing on GPU
        {
            const int n_exp = (int) g.n_expert;
            const int K_val = K;
            mtl::Dispatch d;
            d.p = &pipes.p_topk_router;
            d.args = {mtl::Arg::B(s.router), mtl::Arg::B(s.ids), mtl::Arg::B(s.wts),
                      mtl::Arg::V(n_exp), mtl::Arg::V(K_val)};
            d.grid[0] = 1;
            d.tg[0] = 32;
            d.barrier = true;
            ds.push_back(d);
        }
        if (tr && l == 0) {   // debug cut G7: the router logits and the top-k pick
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G7 pos=%d L=0 ro=%.17g ids=%.17g wts=%.17g\n", (int) s.pos,
                         trace_rows((const float*) s.router.ptr, 1, (size_t) g.n_expert),
                         trace_rows((const float*) s.ids.ptr, 1, (size_t) K),
                         trace_rows((const float*) s.wts.ptr, 1, (size_t) K));
        }

        // Routed experts + Shared expert
        push_gemv(ds, pipes, m.W(L(l, "ffn_gate_exps.weight")), s.h, s.eg, &s.ids, K, 0, true);
        push_gemv(ds, pipes, m.W(L(l, "ffn_up_exps.weight")), s.h, s.eu, &s.ids, K, 0, false);
        push_gemv(ds, pipes, m.W(L(l, "ffn_gate_shexp.weight")), s.h, s.sg, nullptr, 1, 0, false);
        push_gemv(ds, pipes, m.W(L(l, "ffn_up_shexp.weight")), s.h, s.su, nullptr, 1, 0, false);
        {
            const int n_val = K * FF;
            mtl::Dispatch d;
            d.p = &pipes.p_swiglu;
            d.args = {mtl::Arg::B(s.eg), mtl::Arg::B(s.eu), mtl::Arg::B(s.eh), mtl::Arg::V(n_val)};
            d.grid[0] = (n_val + 127) / 128;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        if (tr && l == 0) {   // debug cut G8: the routed experts' swiglu output
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G8 pos=%d L=0 eh=%.17g\n", (int) s.pos,
                         trace_rows((const float*) s.eh.ptr, (size_t) K, (size_t) FF));
        }
        {
            const int n_val = FS;
            mtl::Dispatch d;
            d.p = &pipes.p_swiglu;
            d.args = {mtl::Arg::B(s.sg), mtl::Arg::B(s.su), mtl::Arg::B(s.sh), mtl::Arg::V(n_val)};
            d.grid[0] = (n_val + 127) / 128;
            d.tg[0] = 128;
            d.barrier = false;
            ds.push_back(d);
        }
        push_gemv(ds, pipes, m.W(L(l, "ffn_down_exps.weight")), s.eh, s.ey, &s.ids, K, FF, true);
        push_gemv(ds, pipes, m.W(L(l, "ffn_down_shexp.weight")), s.sh, s.sy, nullptr, 1, 0, false);
        push_gemv(ds, pipes, m.W(L(l, "ffn_gate_inp_shexp.weight")), s.h, s.sgate, nullptr, 1, 0, false);
        if (tr && l == 0) {   // debug cut G9: the down projections + the shared expert + its gate
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G9 pos=%d L=0 sh=%.17g sy=%.17g sgate=%.17g ey=%.17g\n", (int) s.pos,
                         trace_rows((const float*) s.sh.ptr, 1, (size_t) FS),
                         trace_rows((const float*) s.sy.ptr, 1, (size_t) N),
                         trace_rows((const float*) s.sgate.ptr, 1, 1),
                         trace_rows((const float*) s.ey.ptr, (size_t) K, (size_t) N));
        }

        // MoE sum
        {
            const int K_val = K, N_val = N;
            mtl::Dispatch d;
            d.p = &pipes.p_moe_sum;
            d.args = {mtl::Arg::B(s.x), mtl::Arg::B(s.ey), mtl::Arg::B(s.wts), mtl::Arg::B(s.sy), mtl::Arg::B(s.sgate),
                      mtl::Arg::V(K_val), mtl::Arg::V(N_val)};
            d.grid[0] = (N + 127) / 128;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        if (tr && l == 0) {   // debug cut G10: the residual after the MoE add
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G10 pos=%d L=0 x=%.17g\n", (int) s.pos,
                         trace_rows((const float*) s.x.ptr, 1, (size_t) N));
        }

        if (tr) {   // debug: run this layer alone and checksum the token's rows (row sums in index order)
            cx.run_graph(ds);
            ds.clear();
            double xs = 0.0, hs = 0.0;
            const float* xp = (const float*) s.x.ptr;
            for (int i = 0; i < N; ++i) xs += xp[i];
            const float* hp = (const float*) s.h.ptr;
            for (int i = 0; i < N; ++i) hs += hp[i];
            std::fprintf(stderr, "TRACE D pos=%d L=%d x=%.17g h=%.17g\n", (int) s.pos, (int) l, xs, hs);
        }
    }

    // Output norm & LM head
    {
        const int n_val = N;
        const float eps_val = m.eps;
        mtl::Dispatch d;
        d.p = &pipes.p_rms;
        d.args = {mtl::Arg::B(s.x), mtl::Arg::B(m.F("output_norm.weight")), mtl::Arg::B(s.h), mtl::Arg::V(n_val), mtl::Arg::V(eps_val)};
        d.grid[0] = 1;
        d.tg[0] = 128;
        d.barrier = true;
        ds.push_back(d);
    }
    push_gemv(ds, pipes, m.W("output.weight"), s.h, s.logits, nullptr, 1, 0, true);
    double gpu_sec = cx.run_graph(ds);
    if (s.pos >= 9) {
        std::fprintf(stderr, "[step %d: gpu=%.1f ms] ", s.pos, gpu_sec * 1000.0);
    }
    ++s.pos;
}

// Dense projection of a chunk.  The n_in != 2048 products run through k_gemv_c: the tokens walked inside
// the threadgroup, each weight row read once, each (token, output) summed in the decode kernel's exact
// order.  The n_in == 2048 products run through the DECODE k_gemv_lpr8 with grid[1] = nt instead - its
// serial deq path compiles to different fma contraction inside a token loop (~1 ulp; the parity test), and
// greedy decoding turns a 1-ulp logit difference into different tokens - so the decode kernel's own shape
// is the exact one, and the re-read weights stay L2-resident across the chunk's tokens.
void push_c_gemv(std::vector<mtl::Dispatch>& ds, const Pipelines& pipes, const Tensor& w, const mtl::Buffer& x,
                 const mtl::Buffer& y, int nt, int x_stride) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    if (n_in == 2048) {
        auto it_l = pipes.p_gemv_lpr8.find(w.type);
        if (it_l != pipes.p_gemv_lpr8.end()) {
            GemvParam prm;
            prm.row_bytes = w.row_bytes;
            prm.expert_bytes = 0;
            prm.n_in = n_in;
            prm.n_out = n_out;
            prm.x_stride = (uint64_t) x_stride;
            mtl::Dispatch d;
            d.p = &it_l->second;
            d.grid[0] = (uint32_t)((n_out + 15) / 16);
            d.grid[1] = (uint32_t) nt;
            d.tg[0] = 128;
            d.barrier = true;
            d.args = {mtl::Arg::B(w.b, w.offset), mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::B(pipes.zero_ids),
                      mtl::Arg::V(prm)};
            ds.push_back(d);
            return;
        }
    }
    auto it_c = pipes.p_gemv_c.find(w.type);
    if (it_c == pipes.p_gemv_c.end()) {
        std::fprintf(stderr, "q36_metal: unhandled chunk GEMV type %d\n", w.type);
        std::exit(1);
    }
    const int tpb = (n_out >= 1024) ? nt : std::min(nt, 32);
    GemvParam prm;
    prm.row_bytes = w.row_bytes;
    prm.expert_bytes = 0;
    prm.n_in = n_in;
    prm.n_out = n_out;
    prm.x_stride = (uint64_t) x_stride;

    mtl::Dispatch d;
    d.p = &it_c->second;
    d.grid[0] = (uint32_t)((n_out + 7) / 8);
    d.grid[1] = (uint32_t)((nt + tpb - 1) / tpb);
    d.tg[0] = 256;   // 8 simdgroups: the kernel's row = tg.x * 8 + sgid
    d.barrier = true;
    d.args = {mtl::Arg::B(w.b, w.offset), mtl::Arg::B(x), mtl::Arg::B(y), mtl::Arg::V(prm),
              mtl::Arg::V(tpb), mtl::Arg::V(0), mtl::Arg::V(nt)};
    ds.push_back(d);
}

// Routed experts of a chunk: k_expc over the placed per-expert entry runs (every expert's weights read
// once per chunk); x rows are selected per entry through srt_in; off/cnt are byte offsets into the
// engine's small scratch (the k_tables results: counts at 0, offsets at 256 ints).
void push_c_expc(std::vector<mtl::Dispatch>& ds, const Pipelines& pipes, const Tensor& w, const mtl::Buffer& x,
                 const mtl::Buffer& srt_in, const mtl::Buffer& scr, size_t off_bytes, size_t cnt_bytes,
                 const mtl::Buffer& y, int x_stride) {
    const int n_in = (int) w.shape[0], n_out = (int) w.shape[1];
    const size_t eb = w.shape.size() == 3 ? w.row_bytes * (size_t) w.shape[1] : 0;
    auto it = pipes.p_expc.find(w.type);
    if (it == pipes.p_expc.end()) {
        std::fprintf(stderr, "q36_metal: unhandled chunk expert type %d\n", w.type);
        std::exit(1);
    }
    GemvParam prm;
    prm.row_bytes = w.row_bytes;
    prm.expert_bytes = eb;
    prm.n_in = n_in;
    prm.n_out = n_out;
    prm.x_stride = (uint64_t) x_stride;

    mtl::Dispatch d;
    d.p = &it->second;
    d.grid[0] = (uint32_t)((n_out + 7) / 8);
    d.grid[1] = (uint32_t)(w.shape.size() == 3 ? w.shape[2] : 1);
    d.tg[0] = 256;   // 8 simdgroups: the kernel's row = tg.x * 8 + sgid
    d.barrier = true;
    d.args = {mtl::Arg::B(w.b, w.offset), mtl::Arg::B(x), mtl::Arg::B(srt_in), mtl::Arg::B(scr, off_bytes),
              mtl::Arg::B(scr, cnt_bytes), mtl::Arg::B(y), mtl::Arg::V(prm)};
    ds.push_back(d);
}

// Runs the chunk's nt tokens (absolute positions pos0 + t) through the chunk kernels, updating the same
// recurrent states, conv histories and KV caches the decode path uses.  Every (token, output) is summed
// in the decode kernels' exact order, so the states match a token-by-token run bit for bit.  The prompt's
// LAST token is not chunked - the caller runs it as a decode window, which is also where the logits come
// from.  Returns the GPU seconds.
double forward_prompt_chunk(mtl::Context& cx, const Pipelines& pipes, const Model& m, State& s,
                            const int* toks, int nt, int64_t pos0) {
    const auto& g = m.g;
    const int N = (int) g.n_embd, K = (int) m.top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
    const int C = (int) g.ssm_conv_channels, ZV = (int) g.ssm_value_dim;
    const int HV = (int) g.ssm_v_heads, HK = (int) g.ssm_k_heads, DS = (int) g.ssm_state_size;
    const int H = (int) g.n_head, HKV = (int) g.n_head_kv, D = (int) g.head_dim;
    const int n_ent = nt * K;
    const bool trc = env_flag("STRATA_Q36M_TRACE");   // debug: per-layer sub-graphs + cut checksums

    if (D != 256) {   // k_attn_prep_chunk's threadgroup buffer and the split-K kernels are 256-wide
        std::fprintf(stderr, "q36_metal: chunk prefill needs head_dim 256, model has %d\n", D);
        std::exit(1);
    }
    if (g.ssm_d_conv - 1 > 16) {   // k_conv_chunk's shift register
        std::fprintf(stderr, "q36_metal: chunk prefill needs ssm_d_conv <= 17, model has %d\n", (int) g.ssm_d_conv);
        std::exit(1);
    }
    if (g.n_expert > 256) {   // k_route_chunk's threadgroup score array
        std::fprintf(stderr, "q36_metal: chunk prefill needs n_expert <= 256, model has %d\n", (int) g.n_expert);
        std::exit(1);
    }

    std::vector<mtl::Dispatch> ds;
    std::memcpy(s.toks_c.as<int>(), toks, (size_t) nt * sizeof(int));

    // Embedding rows
    {
        const Tensor& e = m.W("token_embd.weight");
        auto it = pipes.p_emb_c.find(e.type);
        if (it == pipes.p_emb_c.end()) {
            std::fprintf(stderr, "q36_metal: unhandled token_embd type %d\n", e.type);
            std::exit(1);
        }
        mtl::Dispatch d;
        d.p = &it->second;
        d.args = {mtl::Arg::B(e.b, e.offset), mtl::Arg::B(s.toks_c), mtl::Arg::B(s.xc),
                  mtl::Arg::V(e.row_bytes), mtl::Arg::V(N), mtl::Arg::V(N)};
        d.grid[0] = (uint32_t) nt;
        d.tg[0] = 128;
        d.barrier = true;
        ds.push_back(d);
    }
    if (trc) {   // debug cut G0: the chunk's embedding rows (type= prints the token_embd GGUF type id)
        if (pos0 == 0) std::fprintf(stderr, "TRACE BEGIN C\n");
        cx.run_graph(ds);
        ds.clear();
        std::fprintf(stderr, "TRACE G0 pos0=%d L=-1 x=%.17g embtype=%d\n", (int) pos0,
                     trace_rows((const float*) s.xc.ptr, (size_t) nt, (size_t) N), (int) m.W("token_embd.weight").type);
    }

    for (int64_t l = 0; l < g.n_layers; ++l) {
        // RMSNorm (one threadgroup per token row)
        {
            mtl::Dispatch d;
            d.p = &pipes.p_rms;
            d.args = {mtl::Arg::B(s.xc), mtl::Arg::B(m.F(L(l, "attn_norm.weight"))), mtl::Arg::B(s.hc),
                      mtl::Arg::V(N), mtl::Arg::V(m.eps)};
            d.grid[0] = (uint32_t) nt;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }

        if (!strata::core::is_qsa_layer(g, l)) {
            push_c_gemv(ds, pipes, m.W(L(l, "attn_qkv.weight")), s.hc, s.qkv_c, nt, N);
            push_c_gemv(ds, pipes, m.W(L(l, "attn_gate.weight")), s.hc, s.z_c, nt, N);
            push_c_gemv(ds, pipes, m.W(L(l, "ssm_beta.weight")), s.hc, s.beta_c, nt, N);
            push_c_gemv(ds, pipes, m.W(L(l, "ssm_alpha.weight")), s.hc, s.alpha_c, nt, N);
            if (trc && l == 0) {   // debug cut G1: the four projections
                cx.run_graph(ds);
                ds.clear();
                std::fprintf(stderr, "TRACE G1 pos0=%d L=0 qkv=%.17g z=%.17g al=%.17g be=%.17g\n", (int) pos0,
                             trace_rows((const float*) s.qkv_c.ptr, (size_t) nt, (size_t) C),
                             trace_rows((const float*) s.z_c.ptr, (size_t) nt, (size_t) ZV),
                             trace_rows((const float*) s.alpha_c.ptr, (size_t) nt, (size_t) HV),
                             trace_rows((const float*) s.beta_c.ptr, (size_t) nt, (size_t) HV));
            }
            {
                const int Kd = (int) g.ssm_d_conv;
                mtl::Dispatch d;
                d.p = &pipes.p_conv_c;
                d.args = {mtl::Arg::B(s.qkv_c), mtl::Arg::B(s.conv_state[l]), mtl::Arg::B(m.F(L(l, "ssm_conv1d.weight"))),
                          mtl::Arg::B(s.conv_out_c), mtl::Arg::V(C), mtl::Arg::V(Kd), mtl::Arg::V(nt)};
                d.grid[0] = (uint32_t)((C + 127) / 128);
                d.grid[1] = (uint32_t)((nt + 15) / 16);
                d.tg[0] = 128;
                d.barrier = true;
                ds.push_back(d);
            }
            if (trc && l == 0) {   // debug cut G2: the conv outputs + the conv history it left
                cx.run_graph(ds);
                ds.clear();
                std::fprintf(stderr, "TRACE G2 pos0=%d L=0 co=%.17g cst=%.17g\n", (int) pos0,
                             trace_rows((const float*) s.conv_out_c.ptr, (size_t) nt, (size_t) C),
                             trace_rows((const float*) s.conv_state[l].ptr, 1, s.conv_state[l].bytes / 4));
            }
            {
                mtl::Dispatch d;
                d.p = &pipes.p_gdn_c;
                d.args = {mtl::Arg::B(s.conv_out_c), mtl::Arg::B(s.alpha_c), mtl::Arg::B(m.F(L(l, "ssm_dt.bias"))),
                          mtl::Arg::B(m.F(L(l, "ssm_a"))), mtl::Arg::B(s.beta_c), mtl::Arg::B(s.S[l]),
                          mtl::Arg::B(s.gdn_out_c), mtl::Arg::V(HK), mtl::Arg::V(DS), mtl::Arg::V(HV),
                          mtl::Arg::V(C), mtl::Arg::V(m.eps), mtl::Arg::V(nt)};
                d.grid[0] = (uint32_t) HV;
                d.tg[0] = 128;
                d.barrier = true;
                ds.push_back(d);
            }
            if (trc && l == 0) {   // debug cut G3: the delta rule's outputs + the state it left
                cx.run_graph(ds);
                ds.clear();
                std::fprintf(stderr, "TRACE G3 pos0=%d L=0 go=%.17g sst=%.17g\n", (int) pos0,
                             trace_rows((const float*) s.gdn_out_c.ptr, (size_t) nt, (size_t) ZV),
                             trace_rows((const float*) s.S[l].ptr, 1, s.S[l].bytes / 4));
            }
            {
                mtl::Dispatch d;
                d.p = &pipes.p_gated_norm_c;
                d.args = {mtl::Arg::B(s.gdn_out_c), mtl::Arg::B(m.F(L(l, "ssm_norm.weight"))), mtl::Arg::B(s.z_c),
                          mtl::Arg::V(DS), mtl::Arg::V(ZV), mtl::Arg::V(m.eps)};
                d.grid[0] = (uint32_t) nt;
                d.grid[1] = (uint32_t) HV;
                d.tg[0] = 128;
                d.barrier = true;
                ds.push_back(d);
            }
            push_c_gemv(ds, pipes, m.W(L(l, "ssm_out.weight")), s.gdn_out_c, s.yc, nt, ZV);
            if (trc && l == 0) {   // debug cut G4: the gated norm's z and the mixer output y
                cx.run_graph(ds);
                ds.clear();
                std::fprintf(stderr, "TRACE G4 pos0=%d L=0 z=%.17g y=%.17g\n", (int) pos0,
                             trace_rows((const float*) s.z_c.ptr, (size_t) nt, (size_t) ZV),
                             trace_rows((const float*) s.yc.ptr, (size_t) nt, (size_t) N));
            }
        } else {
            push_c_gemv(ds, pipes, m.W(L(l, "attn_q.weight")), s.hc, s.qg_c, nt, N);
            push_c_gemv(ds, pipes, m.W(L(l, "attn_k.weight")), s.hc, s.k_c, nt, N);
            push_c_gemv(ds, pipes, m.W(L(l, "attn_v.weight")), s.hc, s.v_c, nt, N);
            {
                mtl::Dispatch d;
                d.p = &pipes.p_attn_prep_c;
                d.args = {mtl::Arg::B(s.qg_c), mtl::Arg::B(s.k_c), mtl::Arg::B(s.v_c),
                          mtl::Arg::B(m.F(L(l, "attn_q_norm.weight"))), mtl::Arg::B(m.F(L(l, "attn_k_norm.weight"))),
                          mtl::Arg::B(s.q_c), mtl::Arg::B(s.gate_c), mtl::Arg::B(s.kc[l]), mtl::Arg::B(s.vc[l]),
                          mtl::Arg::V(H), mtl::Arg::V(HKV), mtl::Arg::V(D), mtl::Arg::V(m.n_rot), mtl::Arg::V((int) pos0),
                          mtl::Arg::V(s.ctx), mtl::Arg::V(m.rope_base), mtl::Arg::V(m.eps)};
                d.grid[0] = (uint32_t) nt;
                d.grid[1] = (uint32_t)(H + HKV);
                d.tg[0] = 128;
                d.barrier = true;
                ds.push_back(d);
            }
            // Attention over the chunk's rows, SK_ROWS_MAX rows at a time (scratch is bounded by that)
            {
                const float scale_val = 1.0f / std::sqrt((float) D);
                const int rows = nt * H;
                for (int rb = 0; rb < rows; rb += SK_ROWS_MAX) {
                    const int rb_rows = std::min(SK_ROWS_MAX, rows - rb);
                    const int t_last = (rb + rb_rows - 1) / H;
                    const int n_chunk = (int)((pos0 + t_last + 1 + SK_CELLS - 1) / SK_CELLS);
                    mtl::Dispatch d;
                    d.p = &pipes.p_attn_sk_part;
                    d.args = {mtl::Arg::B(s.q_c), mtl::Arg::B(s.kc[l]), mtl::Arg::B(s.vc[l]),
                              mtl::Arg::V(H), mtl::Arg::V(HKV), mtl::Arg::V(s.ctx), mtl::Arg::V(scale_val),
                              mtl::Arg::V(n_chunk), mtl::Arg::V((int) pos0), mtl::Arg::V(rb),
                              mtl::Arg::B(s.sk_pacc), mtl::Arg::B(s.sk_pm), mtl::Arg::B(s.sk_pl)};
                    d.grid[0] = (uint32_t) n_chunk;
                    d.grid[1] = (uint32_t) rb_rows;
                    d.tg[0] = SK_CELLS;
                    d.barrier = true;
                    ds.push_back(d);
                    mtl::Dispatch dm;
                    dm.p = &pipes.p_attn_sk_merge;
                    dm.args = {mtl::Arg::B(s.sk_pacc), mtl::Arg::B(s.sk_pm), mtl::Arg::B(s.sk_pl), mtl::Arg::B(s.gate_c),
                               mtl::Arg::B(s.att_c), mtl::Arg::V(H), mtl::Arg::V(n_chunk), mtl::Arg::V((int) pos0), mtl::Arg::V(rb)};
                    dm.grid[0] = (uint32_t) rb_rows;
                    dm.tg[0] = SK_CELLS;
                    dm.barrier = true;
                    ds.push_back(dm);
                }
            }
            push_c_gemv(ds, pipes, m.W(L(l, "attn_output.weight")), s.att_c, s.yc, nt, H * D);
        }

        // Residual add + post norm
        {
            mtl::Dispatch d;
            d.p = &pipes.p_add;
            d.args = {mtl::Arg::B(s.xc), mtl::Arg::B(s.yc), mtl::Arg::V(nt * N)};
            d.grid[0] = (uint32_t)((nt * N + 127) / 128);
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        if (trc && l == 0) {   // debug cut G5: the residual after the mixer's add
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G5 pos0=%d L=0 x=%.17g\n", (int) pos0,
                         trace_rows((const float*) s.xc.ptr, (size_t) nt, (size_t) N));
        }
        {
            mtl::Dispatch d;
            d.p = &pipes.p_rms;
            d.args = {mtl::Arg::B(s.xc), mtl::Arg::B(m.F(L(l, "post_attention_norm.weight"))), mtl::Arg::B(s.hc),
                      mtl::Arg::V(N), mtl::Arg::V(m.eps)};
            d.grid[0] = (uint32_t) nt;
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        if (trc && l == 0) {   // debug cut G6: the post-norm rows the MoE reads
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G6 pos0=%d L=0 h=%.17g\n", (int) pos0,
                         trace_rows((const float*) s.hc.ptr, (size_t) nt, (size_t) N));
        }

        // Router + top-K
        push_c_gemv(ds, pipes, m.W(L(l, "ffn_gate_inp.weight")), s.hc, s.router_c, nt, N);
        {
            mtl::Dispatch d;
            d.p = &pipes.p_route_c;
            d.args = {mtl::Arg::B(s.router_c), mtl::Arg::B(s.ids_c), mtl::Arg::B(s.wts_c),
                      mtl::Arg::V((int) g.n_expert), mtl::Arg::V(K)};
            d.grid[0] = (uint32_t) nt;
            d.tg[0] = 256;
            d.barrier = true;
            ds.push_back(d);
        }
        if (trc && l == 0) {   // debug cut G7: the router logits and the top-k pick
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G7 pos0=%d L=0 ro=%.17g ids=%.17g wts=%.17g\n", (int) pos0,
                         trace_rows((const float*) s.router_c.ptr, (size_t) nt, (size_t) g.n_expert),
                         trace_rows((const float*) s.ids_c.ptr, (size_t) nt, (size_t) K),
                         trace_rows((const float*) s.wts_c.ptr, (size_t) nt, (size_t) K));
        }

        // Entry tables in one dispatch (k_tables: threadgroup atomics inside; plain stores out), then the
        // grouped experts read them across the ordinary barrier
        {
            mtl::Dispatch d;
            d.p = &pipes.p_tables;
            d.args = {mtl::Arg::B(s.ids_c), mtl::Arg::B(s.scr), mtl::Arg::B(s.srt_in_gu), mtl::Arg::B(s.srt_in_d),
                      mtl::Arg::B(s.pos_c), mtl::Arg::V((int) g.n_expert), mtl::Arg::V(K), mtl::Arg::V(n_ent)};
            d.grid[0] = 1;
            d.tg[0] = 256;
            d.barrier = true;
            ds.push_back(d);
        }

        // Routed experts (grouped, every expert's weights read once per chunk) + shared expert
        push_c_expc(ds, pipes, m.W(L(l, "ffn_gate_exps.weight")), s.hc, s.srt_in_gu, s.scr, 256 * sizeof(int), 0, s.eg_c, N);
        push_c_expc(ds, pipes, m.W(L(l, "ffn_up_exps.weight")), s.hc, s.srt_in_gu, s.scr, 256 * sizeof(int), 0, s.eu_c, N);
        {
            const int n_val = n_ent * FF;
            mtl::Dispatch d;
            d.p = &pipes.p_swiglu;
            d.args = {mtl::Arg::B(s.eg_c), mtl::Arg::B(s.eu_c), mtl::Arg::B(s.eh_c), mtl::Arg::V(n_val)};
            d.grid[0] = (uint32_t)((n_val + 127) / 128);
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        if (trc && l == 0) {   // debug cut G8: the routed experts' swiglu output (pos-ordered rows)
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G8 pos0=%d L=0 eh=%.17g\n", (int) pos0,
                         trace_rows_pos((const float*) s.eh_c.ptr, s.pos_c.as<int>(), n_ent, (size_t) FF));
        }
        push_c_gemv(ds, pipes, m.W(L(l, "ffn_gate_shexp.weight")), s.hc, s.sg_c, nt, N);
        push_c_gemv(ds, pipes, m.W(L(l, "ffn_up_shexp.weight")), s.hc, s.su_c, nt, N);
        {
            const int n_val = nt * FS;
            mtl::Dispatch d;
            d.p = &pipes.p_swiglu;
            d.args = {mtl::Arg::B(s.sg_c), mtl::Arg::B(s.su_c), mtl::Arg::B(s.sh_c), mtl::Arg::V(n_val)};
            d.grid[0] = (uint32_t)((n_val + 127) / 128);
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        push_c_expc(ds, pipes, m.W(L(l, "ffn_down_exps.weight")), s.eh_c, s.srt_in_d, s.scr, 256 * sizeof(int), 0, s.ey_c, FF);
        push_c_gemv(ds, pipes, m.W(L(l, "ffn_down_shexp.weight")), s.sh_c, s.sy_c, nt, FS);
        push_c_gemv(ds, pipes, m.W(L(l, "ffn_gate_inp_shexp.weight")), s.hc, s.sgate_c, nt, N);
        if (trc && l == 0) {   // debug cut G9: the down projections + the shared expert + its gate (pos-ordered ey)
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G9 pos0=%d L=0 sh=%.17g sy=%.17g sgate=%.17g ey=%.17g\n", (int) pos0,
                         trace_rows((const float*) s.sh_c.ptr, (size_t) nt, (size_t) FS),
                         trace_rows((const float*) s.sy_c.ptr, (size_t) nt, (size_t) N),
                         trace_rows((const float*) s.sgate_c.ptr, (size_t) nt, 1),
                         trace_rows_pos((const float*) s.ey_c.ptr, s.pos_c.as<int>(), n_ent, (size_t) N));
        }
        {
            mtl::Dispatch d;
            d.p = &pipes.p_moe_sum_c;
            d.args = {mtl::Arg::B(s.xc), mtl::Arg::B(s.ey_c), mtl::Arg::B(s.wts_c), mtl::Arg::B(s.sy_c),
                      mtl::Arg::B(s.sgate_c), mtl::Arg::B(s.pos_c), mtl::Arg::V(K), mtl::Arg::V(N), mtl::Arg::V(N)};
            d.grid[0] = (uint32_t) nt;
            d.grid[1] = (uint32_t)((N + 127) / 128);
            d.tg[0] = 128;
            d.barrier = true;
            ds.push_back(d);
        }
        if (trc && l == 0) {   // debug cut G10: the residual after the MoE add
            cx.run_graph(ds);
            ds.clear();
            std::fprintf(stderr, "TRACE G10 pos0=%d L=0 x=%.17g\n", (int) pos0,
                         trace_rows((const float*) s.xc.ptr, (size_t) nt, (size_t) N));
        }

        if (trc) {   // debug: run this layer alone and checksum every row
            cx.run_graph(ds);
            ds.clear();
            double xs = 0.0, hs = 0.0;
            const float* xp = (const float*) s.xc.ptr;
            const float* hp = (const float*) s.hc.ptr;
            for (int t = 0; t < nt; ++t) {
                double rx = 0.0, rh = 0.0;
                for (int i = 0; i < N; ++i) rx += xp[(size_t) t * N + i];
                for (int i = 0; i < N; ++i) rh += hp[(size_t) t * N + i];
                xs += rx;
                hs += rh;
            }
            std::fprintf(stderr, "TRACE C nt=%d pos0=%lld L=%lld x=%.17g h=%.17g\n", nt, (long long) pos0,
                         (long long) l, xs, hs);
        }
    }

    if (trc) return 0.0;   // the per-layer sub-graphs already ran
    return cx.run_graph(ds);
}

std::vector<int> parse_ids(const std::string& s) {
    std::vector<int> v;
    size_t i = 0;
    while (i < s.size()) {
        while (i < s.size() && (s[i] == ',' || s[i] == ' ' || s[i] == '\n' || s[i] == '\r')) ++i;
        if (i >= s.size()) break;
        v.push_back(std::atoi(s.c_str() + i));
        while (i < s.size() && s[i] != ',' && s[i] != ' ' && s[i] != '\n') ++i;
    }
    return v;
}

}  // namespace

int run(int argc, char** argv) {
    std::string model_path;
    std::string tokens_str;
    int n_gen = 64;
    int ctx = 4096;
    int prefill_chunk = 0;   // --prefill N: the prompt-chunk size (0 = token-by-token, as before)
    bool serve_mode = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--serve") {
            serve_mode = true;
        } else if ((a == "--native" || a == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--tokens" && i + 1 < argc) {
            tokens_str = argv[++i];
        } else if ((a == "--n" || a == "--max-new") && i + 1 < argc) {
            n_gen = std::atoi(argv[++i]);
        } else if ((a == "--ctx" || a == "--max-context") && i + 1 < argc) {
            ctx = std::atoi(argv[++i]);
        } else if (a == "--prefill" && i + 1 < argc) {
            prefill_chunk = std::atoi(argv[++i]);
        }
    }
    if (prefill_chunk < 0) prefill_chunk = 0;
    if (prefill_chunk == 0 && env_flag("STRATA_Q36M_PREFILL")) prefill_chunk = 512;

    if (model_path.empty()) {
        std::fprintf(stderr, "strata metal q36: specify --model <path.gguf>\n");
        return 2;
    }

    mtl::Context cx;
    std::fprintf(stderr, "strata metal q36: GPU %s (recommended working set: %.1f GB)\n",
                 cx.device_name().c_str(), cx.recommended_working_set() / (1024.0 * 1024.0 * 1024.0));

    Pipelines pipes;
    pipes.init(cx);

    Model m;
    std::string err;
    const auto t0 = std::chrono::steady_clock::now();
    const strata::GgufModel gm = strata::GgufModel::open(model_path);
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) {
        std::fprintf(stderr, "strata metal q36: geometry error: %s\n", err.c_str());
        return 1;
    }
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        std::fprintf(stderr, "strata metal q36: expected qwen35moe architecture\n");
        return 1;
    }

    const std::string p = "qwen35moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    const bool sk_force = env_flag("STRATA_Q36M_ATTN_SK");

    // Wrap GGUF tensors no-copy directly into Metal buffers
    uint64_t total_weight_bytes = 0;
    std::vector<mtl::Buffer> wrapped_shards;
    for (size_t s = 0; s < gm.size(); ++s) {
        const strata::GgufFile& f = gm.shard(s);
        // Each shard is already mmapped; wrap the mapped region no-copy
        mtl::Buffer sb = cx.wrap((void*) f.data(), f.file_size(), cx.max_buffer_length());
        wrapped_shards.push_back(sb);

        for (const strata::TensorInfo& ti : f.tensors()) {
            if (ti.name.rfind("blk." + std::to_string(m.g.n_layers) + ".", 0) == 0) continue; // skip MTP
            const uint64_t nb = strata::tensor_payload_bytes(ti);
            const uint8_t* src = f.tensor_data(ti);
            const bool conv = ti.name.size() > 17 && ti.name.compare(ti.name.size() - 17, 17, "ssm_conv1d.weight") == 0;

            if (ti.shape.size() == 1 || conv) {
                // Convert 1D vectors to F32 in Metal shared buffer
                std::vector<float> h(ti.elements());
                if (ti.type == T_F32) std::memcpy(h.data(), src, h.size() * 4);
                else if (ti.type == T_BF16)
                    for (size_t i = 0; i < h.size(); ++i) {
                        const uint32_t u = (uint32_t) (src[2 * i] | (src[2 * i + 1] << 8)) << 16;
                        std::memcpy(&h[i], &u, 4);
                    }
                else if (ti.type == T_F16)
                    for (size_t i = 0; i < h.size(); ++i) h[i] = h2f_host((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));

                mtl::Buffer b = cx.alloc(h.size() * sizeof(float));
                std::memcpy(b.ptr, h.data(), h.size() * sizeof(float));
                m.f32[ti.name] = b;
                total_weight_bytes += h.size() * 4;

                if (ti.name.find("ffn_gate_inp_shexp.weight") != std::string::npos) {
                    Tensor w;
                    w.type = T_F32;
                    w.shape = {(int64_t) h.size(), 1};
                    w.row_bytes = h.size() * 4;
                    w.b = b;
                    w.offset = 0;
                    m.t[ti.name] = w;
                }
            } else {
                Tensor w;
                w.type = (int) ti.type;
                for (uint64_t d : ti.shape) w.shape.push_back((int64_t) d);
                w.row_bytes = row_bytes_of(w.type, w.shape[0]);
                w.b = sb;
                w.offset = (size_t) (src - f.data());
                m.t[ti.name] = w;
                total_weight_bytes += nb;
            }
        }
    }

    std::fprintf(stderr, "strata metal q36: model loaded in %.2f s (%.2f GB weights wrapped no-copy)\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(),
                 total_weight_bytes / (1024.0 * 1024.0 * 1024.0));

    // Allocate runtime state buffers
    State s;
    s.ctx = ctx;
    const auto& g = m.g;
    const int N = (int) g.n_embd, FF = (int) g.n_ff, K = (int) m.top_k;
    s.x = cx.alloc(N * sizeof(float));
    s.h = cx.alloc(N * sizeof(float));
    s.y = cx.alloc(N * sizeof(float));
    s.qkv = cx.alloc(g.ssm_conv_channels * sizeof(float));
    s.z = cx.alloc(g.ssm_value_dim * sizeof(float));
    s.alpha = cx.alloc(g.ssm_v_heads * sizeof(float));
    s.beta = cx.alloc(g.ssm_v_heads * sizeof(float));
    s.conv_out = cx.alloc(g.ssm_conv_channels * sizeof(float));
    s.gdn_out = cx.alloc(g.ssm_value_dim * sizeof(float));
    s.qg = cx.alloc(2 * g.n_head * g.head_dim * sizeof(float));
    s.k = cx.alloc(g.n_head_kv * g.head_dim * sizeof(float));
    s.v = cx.alloc(g.n_head_kv * g.head_dim * sizeof(float));
    s.q = cx.alloc(g.n_head * g.head_dim * sizeof(float));
    s.gate = cx.alloc(g.n_head * g.head_dim * sizeof(float));
    s.att = cx.alloc(g.n_head * g.head_dim * sizeof(float));
    s.router = cx.alloc(g.n_expert * sizeof(float));
    s.eg = cx.alloc((size_t) K * FF * sizeof(float));
    s.eu = cx.alloc((size_t) K * FF * sizeof(float));
    s.eh = cx.alloc((size_t) K * FF * sizeof(float));
    s.ey = cx.alloc((size_t) K * N * sizeof(float));
    s.sg = cx.alloc(g.n_ff_shexp * sizeof(float));
    s.su = cx.alloc(g.n_ff_shexp * sizeof(float));
    s.sh = cx.alloc(g.n_ff_shexp * sizeof(float));
    s.sy = cx.alloc(N * sizeof(float));
    s.sgate = cx.alloc(sizeof(float));
    s.wts = cx.alloc(K * sizeof(float));
    s.logits = cx.alloc(g.n_vocab * sizeof(float));
    s.ids = cx.alloc(K * sizeof(int));

    // Split-K attention scratch: one partial per (query row, 256-cell chunk), up to SK_ROWS_MAX rows per
    // dispatch (decode uses n_head of them; the prompt chunks row-block to the same bound).
    {
        const int n_chunk_max = (ctx + SK_CELLS - 1) / SK_CELLS;
        s.sk_pacc = cx.alloc((size_t) SK_ROWS_MAX * n_chunk_max * SK_CELLS * sizeof(float));
        s.sk_pm = cx.alloc((size_t) SK_ROWS_MAX * n_chunk_max * sizeof(float));
        s.sk_pl = cx.alloc((size_t) SK_ROWS_MAX * n_chunk_max * sizeof(float));
    }

    // Prompt-chunk buffers (--prefill N): sized for this run's largest chunk.  The prompt's last token is
    // never chunked - it runs as a decode window, which is also where the logits come from.
    if (prefill_chunk > 0) {
        const size_t CT = (size_t) prefill_chunk;
        s.toks_c = cx.alloc(CT * sizeof(int));
        s.xc = cx.alloc(CT * N * sizeof(float));
        s.hc = cx.alloc(CT * N * sizeof(float));
        s.yc = cx.alloc(CT * N * sizeof(float));
        s.qkv_c = cx.alloc(CT * g.ssm_conv_channels * sizeof(float));
        s.conv_out_c = cx.alloc(CT * g.ssm_conv_channels * sizeof(float));
        s.z_c = cx.alloc(CT * g.ssm_value_dim * sizeof(float));
        s.gdn_out_c = cx.alloc(CT * g.ssm_value_dim * sizeof(float));
        s.alpha_c = cx.alloc(CT * g.ssm_v_heads * sizeof(float));
        s.beta_c = cx.alloc(CT * g.ssm_v_heads * sizeof(float));
        s.qg_c = cx.alloc(CT * 2 * g.n_head * g.head_dim * sizeof(float));
        s.k_c = cx.alloc(CT * g.n_head_kv * g.head_dim * sizeof(float));
        s.v_c = cx.alloc(CT * g.n_head_kv * g.head_dim * sizeof(float));
        s.q_c = cx.alloc(CT * g.n_head * g.head_dim * sizeof(float));
        s.gate_c = cx.alloc(CT * g.n_head * g.head_dim * sizeof(float));
        s.att_c = cx.alloc(CT * g.n_head * g.head_dim * sizeof(float));
        s.router_c = cx.alloc(CT * g.n_expert * sizeof(float));
        s.ids_c = cx.alloc(CT * K * sizeof(int));
        s.wts_c = cx.alloc(CT * K * sizeof(float));
        s.scr = cx.alloc(2 * 256 * sizeof(int));
        s.srt_in_gu = cx.alloc(CT * K * sizeof(int));
        s.srt_in_d = cx.alloc(CT * K * sizeof(int));
        s.pos_c = cx.alloc(CT * K * sizeof(int));
        s.eg_c = cx.alloc(CT * K * FF * sizeof(float));
        s.eu_c = cx.alloc(CT * K * FF * sizeof(float));
        s.eh_c = cx.alloc(CT * K * FF * sizeof(float));
        s.ey_c = cx.alloc(CT * K * N * sizeof(float));
        s.sg_c = cx.alloc(CT * g.n_ff_shexp * sizeof(float));
        s.su_c = cx.alloc(CT * g.n_ff_shexp * sizeof(float));
        s.sh_c = cx.alloc(CT * g.n_ff_shexp * sizeof(float));
        s.sy_c = cx.alloc(CT * N * sizeof(float));
        s.sgate_c = cx.alloc(CT * sizeof(float));
        pipes.init_prompt(cx, prefill_chunk);
        std::fprintf(stderr, "strata metal q36: prompt chunks of %d tokens on (chunk buffers %.2f GB)\n",
                     prefill_chunk, (double) (CT * K * (size_t) (3 * FF + N + 4)) / (1024.0 * 1024.0 * 1024.0)
                         + (double) (CT * (size_t) (6 * N + 2 * g.ssm_conv_channels + 2 * g.ssm_value_dim
                                                    + 2 * g.ssm_v_heads + 2 * g.n_head * g.head_dim + 2 * g.n_head_kv * g.head_dim
                                                    + 3 * g.n_head * g.head_dim + g.n_expert + 3 * g.n_ff_shexp + 1) * 4
                                    + CT * K * 16 + 3 * 256 * 4) / (1024.0 * 1024.0 * 1024.0));
    }

    // Working-set refusal with the numbers (the CUDA engine refuses a count that cannot fit, with the
    // numbers): everything allocated above (weights, KV cache, recurrent state, split-K and chunk scratch)
    // plus a reserve for sampling and transients.  Only enforced above the 8,192-token context the
    // threadgroup attention could reach - at or below it every setup runs exactly as before - and
    // STRATA_Q36M_MEM_OK=1 overrides it.
    {
        const uint64_t est = cx.allocated() + (256ull << 20);
        const uint64_t rec = cx.recommended_working_set();
        if ((ctx > SK_TG_CAP || prefill_chunk > 0) && est > rec && !env_flag("STRATA_Q36M_MEM_OK")) {
            std::fprintf(stderr,
                         "strata metal q36: ctx %d takes ~%.2f GB allocated (+0.25 reserve) past the "
                         "recommended working set %.2f GB - lower --ctx (or --prefill) or set "
                         "STRATA_Q36M_MEM_OK=1\n",
                         ctx, est / 1e9, rec / 1e9);
            return 1;
        }
    }

    for (int64_t l = 0; l < g.n_layers; ++l) {
        if (strata::core::is_qsa_layer(g, l)) {
            s.conv_state.push_back({});
            s.S.push_back({});
            const size_t cache_bytes = (size_t) g.n_head_kv * s.ctx * g.head_dim * sizeof(float);
            s.kc.push_back(cx.alloc(cache_bytes));
            s.vc.push_back(cx.alloc(cache_bytes));
            std::memset(s.kc.back().ptr, 0, cache_bytes);
            std::memset(s.vc.back().ptr, 0, cache_bytes);
        } else {
            const size_t conv_bytes = (size_t) (g.ssm_d_conv - 1) * g.ssm_conv_channels * sizeof(float);
            const size_t s_bytes = (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * sizeof(float);
            s.conv_state.push_back(cx.alloc(conv_bytes));
            s.S.push_back(cx.alloc(s_bytes));
            std::memset(s.conv_state.back().ptr, 0, conv_bytes);
            std::memset(s.S.back().ptr, 0, s_bytes);
            s.kc.push_back({});
            s.vc.push_back({});
        }
    }

    auto sample_next = [&](float temp, float top_p) -> int {
        const float* lg = s.logits.as<float>();
        const int V = (int) g.n_vocab;
        if (temp <= 0.01f) {
            int best_id = 0;
            float best_val = -1e30f;
            for (int i = 0; i < V; ++i) {
                if (std::isnan(lg[i])) continue;
                if (lg[i] > best_val) {
                    best_val = lg[i];
                    best_id = i;
                }
            }
            return best_id;
        }
        float mx = -1e30f;
        for (int i = 0; i < V; ++i) if (!std::isnan(lg[i]) && lg[i] > mx) mx = lg[i];
        std::vector<std::pair<float, int>> probs;
        probs.reserve(V);
        float sum = 0.0f;
        for (int i = 0; i < V; ++i) {
            if (std::isnan(lg[i])) continue;
            float p = std::exp((lg[i] - mx) / temp);
            probs.push_back({p, i});
            sum += p;
        }
        if (sum <= 0.0f || probs.empty()) return 0;
        for (auto& pr : probs) pr.first /= sum;
        if (top_p < 1.0f) {
            std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            float cum = 0.0f;
            size_t cutoff = 0;
            for (; cutoff < probs.size(); ++cutoff) {
                cum += probs[cutoff].first;
                if (cum >= top_p) { ++cutoff; break; }
            }
            probs.resize(std::min(cutoff, probs.size()));
            float nsum = 0.0f;
            for (const auto& pr : probs) nsum += pr.first;
            for (auto& pr : probs) pr.first /= nsum;
        }
        float r = ((float) std::rand()) / (float) RAND_MAX;
        for (const auto& pr : probs) {
            r -= pr.first;
            if (r <= 0.0f) return pr.second;
        }
        return probs.back().second;
    };

    std::vector<int> ids_host(K);

    if (serve_mode) {
        std::printf("READY %d stop\n", s.ctx);
        std::fflush(stdout);

        std::atomic<bool> stop_flag{false};
        std::mutex mu;
        std::condition_variable cv;
        std::deque<std::string> lines;
        bool eof = false;

        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") {
                    stop_flag.store(true);
                    continue;
                }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();

        std::vector<int> live_tokens;

        for (;;) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (lines.empty() && eof) break;
                line = lines.front();
                lines.pop_front();
            }

            if (line.empty()) continue;
            if (line.rfind("GEN", 0) != 0) {
                std::printf("ERR expected GEN <max_new> [key=value ...] <ids>\n");
                std::fflush(stdout);
                continue;
            }

            stop_flag.store(false);
            const char* p = line.c_str() + 3;
            while (*p == ' ') ++p;
            int max_new = std::atoi(p);
            while (*p && *p != ' ') ++p;

            float temperature = 0.0f;
            float top_p = 1.0f;
            for (;;) {
                while (*p == ' ') ++p;
                const char* st0 = p;
                while (*p && *p != ' ') ++p;
                std::string tk(st0, (size_t)(p - st0));
                size_t eq = tk.find('=');
                if (tk.empty() || eq == std::string::npos) {
                    p = st0;
                    break;
                }
                std::string key = tk.substr(0, eq);
                std::string val = tk.substr(eq + 1);
                if (key == "temperature") temperature = std::strtof(val.c_str(), nullptr);
                else if (key == "top_p") top_p = std::strtof(val.c_str(), nullptr);
            }

            std::vector<int> ids = parse_ids(p);
            if (ids.empty() || max_new < 1) {
                std::printf("ERR bad request: max_new or ids\n");
                std::fflush(stdout);
                continue;
            }

            // Prefix matching
            size_t reuse = 0;
            while (reuse < live_tokens.size() && reuse < ids.size() && live_tokens[reuse] == ids[reuse]) {
                ++reuse;
            }

            if (reuse < live_tokens.size() || reuse == 0) {
                s.reset();
                reuse = 0;
                live_tokens.clear();
            }

            const auto t_prompt0 = std::chrono::steady_clock::now();
            size_t pi = reuse;
            if (prefill_chunk > 0 && ids.size() >= 2) {
                const size_t last = ids.size() - 1;   // the last prompt token runs as a decode window
                while (pi < last) {
                    const int nt = (int) std::min<size_t>((size_t) prefill_chunk, last - pi);
                    forward_prompt_chunk(cx, pipes, m, s, ids.data() + pi, nt, s.pos);
                    s.pos += nt;
                    pi += nt;
                }
            }
            for (; pi < ids.size(); ++pi) {
                forward_token(cx, pipes, m, s, ids[pi], ids_host, sk_force, true);
            }
            live_tokens = ids;
            const double prompt_ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prompt0).count() * 1000.0;

            const auto t_dec0 = std::chrono::steady_clock::now();
            long long produced = 0;
            const char* finish = "length";

            int cur_tok = sample_next(temperature, top_p);
            for (;;) {
                std::printf("T %d\n", cur_tok);
                std::fflush(stdout);
                ++produced;
                live_tokens.push_back(cur_tok);

                if (cur_tok == 248046 || cur_tok == 248044 || cur_tok == 151645 || cur_tok == 151643) {
                    finish = "stop";
                    break;
                }
                if (produced >= max_new) {
                    finish = "length";
                    break;
                }
                if (stop_flag.load()) {
                    finish = "cancel";
                    break;
                }
                if (s.pos + 1 >= s.ctx) {
                    finish = "length";
                    break;
                }

                forward_token(cx, pipes, m, s, cur_tok, ids_host, sk_force);
                cur_tok = sample_next(temperature, top_p);
            }

            const double decode_ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_dec0).count() * 1000.0;
            std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %zu 0\n",
                        produced, ids.size(), prompt_ms, decode_ms, finish, reuse, ids.size() - reuse);
            std::fflush(stdout);
        }
        return 0;
    }

    // Measure kernel breakdown for layer 0
    {
        std::fprintf(stderr, "strata metal q36: profiling layer 0 kernel breakdown (20 reps):\n");
        auto bench = [&](const char* tag, auto fn) {
            std::vector<mtl::Dispatch> ds;
            fn(ds);
            double t = (cx.run(ds, 20, false, false) / 20.0) * 1e6;
            std::fprintf(stderr, "  %-25s: %7.1f us\n", tag, t);
        };
        bench("attn_qkv (Q5_K)", [&](auto& ds) { push_gemv(ds, pipes, m.W(L(0, "attn_qkv.weight")), s.h, s.qkv); });
        bench("attn_gate (Q5_K)", [&](auto& ds) { push_gemv(ds, pipes, m.W(L(0, "attn_gate.weight")), s.h, s.z); });
        bench("ssm_out (Q6_K)", [&](auto& ds) { push_gemv(ds, pipes, m.W(L(0, "ssm_out.weight")), s.gdn_out, s.y); });
        bench("router (F32)", [&](auto& ds) { push_gemv(ds, pipes, m.W(L(0, "ffn_gate_inp.weight")), s.h, s.router); });
        bench("topk_router", [&](auto& ds) {
            const int n_exp = (int)g.n_expert, K_val = K;
            mtl::Dispatch d;
            d.p = &pipes.p_topk_router;
            d.args = {mtl::Arg::B(s.router), mtl::Arg::B(s.ids), mtl::Arg::B(s.wts), mtl::Arg::V(n_exp), mtl::Arg::V(K_val)};
            d.grid[0] = 1; d.tg[0] = 32; ds.push_back(d);
        });
        bench("ffn_gate_exps (IQ2_XS)", [&](auto& ds) { push_gemv(ds, pipes, m.W(L(0, "ffn_gate_exps.weight")), s.h, s.eg, &s.ids, K, 0); });
        bench("ffn_up_exps (IQ2_XS)", [&](auto& ds) { push_gemv(ds, pipes, m.W(L(0, "ffn_up_exps.weight")), s.h, s.eu, &s.ids, K, 0); });
        bench("swiglu", [&](auto& ds) {
            const int n_val = K * FF;
            mtl::Dispatch d;
            d.p = &pipes.p_swiglu;
            d.args = {mtl::Arg::B(s.eg), mtl::Arg::B(s.eu), mtl::Arg::B(s.eh), mtl::Arg::V(n_val)};
            d.grid[0] = (n_val + 127) / 128; d.tg[0] = 128; ds.push_back(d);
        });
        bench("ffn_down_exps (IQ3_XXS)", [&](auto& ds) { push_gemv(ds, pipes, m.W(L(0, "ffn_down_exps.weight")), s.eh, s.ey, &s.ids, K, FF); });
        bench("ffn_shexp_down (Q6_K)", [&](auto& ds) { push_gemv(ds, pipes, m.W(L(0, "ffn_down_shexp.weight")), s.sh, s.sy); });
        bench("output.weight (Q4_K)", [&](auto& ds) { push_gemv(ds, pipes, m.W("output.weight"), s.h, s.logits); });
    }

    std::vector<int> prompt = parse_ids(tokens_str);
    if (prompt.empty()) prompt = {248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198}; // Default Qwen 3.6 ChatML prompt

    std::fprintf(stderr, "strata metal q36: evaluating prompt (%zu tokens) ...\n", prompt.size());
    const auto t_prompt0 = std::chrono::steady_clock::now();
    {
        size_t pi = 0;
        if (prefill_chunk > 0 && prompt.size() >= 2) {
            const size_t last = prompt.size() - 1;   // the last prompt token runs as a decode window
            while (pi < last) {
                const int nt = (int) std::min<size_t>((size_t) prefill_chunk, last - pi);
                forward_prompt_chunk(cx, pipes, m, s, prompt.data() + pi, nt, s.pos);
                s.pos += nt;
                pi += nt;
            }
        }
        for (; pi < prompt.size(); ++pi) {
            forward_token(cx, pipes, m, s, prompt[pi], ids_host, sk_force, true);
        }
    }
    const double prompt_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_prompt0).count();
    std::fprintf(stderr, "strata metal q36: prompt evaluated in %.2f s (%.1f tok/s)\n",
                 prompt_time, prompt.size() / prompt_time);

    std::fprintf(stderr, "strata metal q36: generating %d tokens (greedy) ...\n", n_gen);
    std::printf("gen:");
    std::fflush(stdout);

    const auto t_gen0 = std::chrono::steady_clock::now();
    int cur_tok = sample_next(0.0f, 1.0f);
    std::printf(" %d", cur_tok);
    std::fflush(stdout);

    for (int step = 1; step < n_gen; ++step) {
        forward_token(cx, pipes, m, s, cur_tok, ids_host, sk_force);
        cur_tok = sample_next(0.0f, 1.0f);
        std::printf(" %d", cur_tok);
        std::fflush(stdout);
    }
    std::printf("\n");
    const double gen_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen0).count();
    std::fprintf(stderr, "strata metal q36: generated %d tokens in %.2f s (%.1f tok/s decode)\n",
                 n_gen, gen_time, n_gen / gen_time);

    return 0;
}

}  // namespace strata::qwen35
