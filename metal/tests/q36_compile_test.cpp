// metal/tests/q36_compile_test.cpp - verify compilation of q36.metal (decode + split-K + prompt chunks)
#include "mtl_rt.h"
#include <cstdio>

int main() {
    mtl::Context cx;
    std::printf("Metal device: %s\n", cx.device_name().c_str());
    int lib = cx.compile({"quant.metal", "q36.metal"});
    std::printf("q36.metal compiled successfully, lib index: %d\n", lib);
    auto p_rms = cx.pipeline(lib, "k_rms");
    std::printf("pipeline k_rms created: %s (max threads: %d)\n", p_rms.name.c_str(), p_rms.max_threads);
    auto p_conv = cx.pipeline(lib, "k_conv");
    std::printf("pipeline k_conv created: %s (max threads: %d)\n", p_conv.name.c_str(), p_conv.max_threads);
    auto p_gdn = cx.pipeline(lib, "k_gdn");
    std::printf("pipeline k_gdn created: %s (max threads: %d)\n", p_gdn.name.c_str(), p_gdn.max_threads);
    auto p_attn_prep = cx.pipeline(lib, "k_attn_prep");
    std::printf("pipeline k_attn_prep created: %s (max threads: %d)\n", p_attn_prep.name.c_str(), p_attn_prep.max_threads);
    auto p_attn_sk_part = cx.pipeline(lib, "k_attn_sk_part");
    std::printf("pipeline k_attn_sk_part created: %s (max threads: %d)\n", p_attn_sk_part.name.c_str(), p_attn_sk_part.max_threads);
    auto p_attn_sk_merge = cx.pipeline(lib, "k_attn_sk_merge");
    std::printf("pipeline k_attn_sk_merge created: %s (max threads: %d)\n", p_attn_sk_merge.name.c_str(), p_attn_sk_merge.max_threads);
    auto p_attn_gqa_part = cx.pipeline(lib, "k_attn_gqa_part");
    std::printf("pipeline k_attn_gqa_part created: %s (max threads: %d)\n", p_attn_gqa_part.name.c_str(), p_attn_gqa_part.max_threads);
    auto p_attn_gqa_merge = cx.pipeline(lib, "k_attn_gqa_merge");
    std::printf("pipeline k_attn_gqa_merge created: %s (max threads: %d)\n", p_attn_gqa_merge.name.c_str(), p_attn_gqa_merge.max_threads);
    auto p_attn_sk_part_gqa = cx.pipeline(lib, "k_attn_sk_part_gqa");
    std::printf("pipeline k_attn_sk_part_gqa created: %s (max threads: %d)\n", p_attn_sk_part_gqa.name.c_str(), p_attn_sk_part_gqa.max_threads);
    auto p_gemv_q8 = cx.pipeline(lib, "k_gemv_T_Q8_0");
    std::printf("pipeline k_gemv_T_Q8_0 created: %s (max threads: %d)\n", p_gemv_q8.name.c_str(), p_gemv_q8.max_threads);
    auto p_gemv_q4k = cx.pipeline(lib, "k_gemv_T_Q4_K");
    std::printf("pipeline k_gemv_T_Q4_K created: %s (max threads: %d)\n", p_gemv_q4k.name.c_str(), p_gemv_q4k.max_threads);

    // prompt chunks
    const char* emb_types[] = {"T_F32", "T_BF16", "T_Q8_0", "T_Q3_K", "T_Q4_K", "T_Q5_K", "T_Q6_K", "T_IQ2_XS", "T_IQ3_XXS", "T_IQ4_XS"};
    for (const char* t : emb_types) {
        auto p = cx.pipeline(lib, (std::string("k_emb_chunk_") + t).c_str());
        std::printf("pipeline k_emb_chunk_%s created: (max threads: %d)\n", t, p.max_threads);
    }
    for (const char* t : emb_types) {
        auto p = cx.pipeline(lib, (std::string("k_gemv_c_") + t).c_str());
        std::printf("pipeline k_gemv_c_%s created: (max threads: %d)\n", t, p.max_threads);
    }
    const char* lpr8_types[] = {"T_Q4_K", "T_Q5_K", "T_Q6_K", "T_IQ2_XS"};
    for (const char* t : lpr8_types) {
        auto p = cx.pipeline(lib, (std::string("k_gemv_lpr8_") + t).c_str());
        std::printf("pipeline k_gemv_lpr8_%s created: (max threads: %d)\n", t, p.max_threads);
    }
    for (const char* t : emb_types) {
        auto p = cx.pipeline(lib, (std::string("k_expc_") + t).c_str());
        std::printf("pipeline k_expc_%s created: (max threads: %d)\n", t, p.max_threads);
    }
    const char* fixed_kernels[] = {"k_route_chunk", "k_tables", "k_gather_t", "k_zero", "k_hist",
                                   "k_exp_off", "k_place", "k_moe_sum_chunk", "k_conv_chunk", "k_gdn_chunk",
                                   "k_gated_norm_chunk", "k_attn_prep_chunk"};
    for (const char* k : fixed_kernels) {
        auto p = cx.pipeline(lib, k);
        std::printf("pipeline %s created: (max threads: %d)\n", k, p.max_threads);
    }

    // --kv-f16's second runtime compile: the STRATA_KV_F16 lib (the prelude define) and its seven KV
    // pipelines.  A q36.metal syntax error under the define surfaces here (or at engine startup),
    // which is the only check MSL gets on a Mac round - Linux cannot compile MSL.
    int lib_f16 = cx.compile({"quant.metal", "q36.metal"}, "#define STRATA_KV_F16 1\n");
    std::printf("q36.metal compiled with STRATA_KV_F16, lib index: %d\n", lib_f16);
    auto p_attn_prep_f16 = cx.pipeline(lib_f16, "k_attn_prep");
    std::printf("pipeline k_attn_prep (f16 lib) created: %s (max threads: %d)\n", p_attn_prep_f16.name.c_str(),
                p_attn_prep_f16.max_threads);
    auto p_attn_f16 = cx.pipeline(lib_f16, "k_attn");
    std::printf("pipeline k_attn (f16 lib) created: %s (max threads: %d)\n", p_attn_f16.name.c_str(),
                p_attn_f16.max_threads);
    auto p_attn_sk_part_f16 = cx.pipeline(lib_f16, "k_attn_sk_part");
    std::printf("pipeline k_attn_sk_part (f16 lib) created: %s (max threads: %d)\n", p_attn_sk_part_f16.name.c_str(),
                p_attn_sk_part_f16.max_threads);
    auto p_attn_gqa_part_f16 = cx.pipeline(lib_f16, "k_attn_gqa_part");
    std::printf("pipeline k_attn_gqa_part (f16 lib) created: %s (max threads: %d)\n", p_attn_gqa_part_f16.name.c_str(),
                p_attn_gqa_part_f16.max_threads);
    auto p_attn_gqa_merge_f16 = cx.pipeline(lib_f16, "k_attn_gqa_merge");
    std::printf("pipeline k_attn_gqa_merge (f16 lib) created: %s (max threads: %d)\n", p_attn_gqa_merge_f16.name.c_str(),
                p_attn_gqa_merge_f16.max_threads);
    auto p_attn_sk_part_gqa_f16 = cx.pipeline(lib_f16, "k_attn_sk_part_gqa");
    std::printf("pipeline k_attn_sk_part_gqa (f16 lib) created: %s (max threads: %d)\n",
                p_attn_sk_part_gqa_f16.name.c_str(), p_attn_sk_part_gqa_f16.max_threads);
    auto p_attn_prep_c_f16 = cx.pipeline(lib_f16, "k_attn_prep_chunk");
    std::printf("pipeline k_attn_prep_chunk (f16 lib) created: %s (max threads: %d)\n", p_attn_prep_c_f16.name.c_str(),
                p_attn_prep_c_f16.max_threads);
    std::printf("all q36 f16 pipelines created\n");
    std::printf("all q36 pipelines created\n");
    return 0;
}
