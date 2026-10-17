// metal/tests/q36_compile_test.cpp - verify compilation of q36.metal
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
    auto p_gemv_q8 = cx.pipeline(lib, "k_gemv_T_Q8_0");
    std::printf("pipeline k_gemv_T_Q8_0 created: %s (max threads: %d)\n", p_gemv_q8.name.c_str(), p_gemv_q8.max_threads);
    auto p_gemv_q4k = cx.pipeline(lib, "k_gemv_T_Q4_K");
    std::printf("pipeline k_gemv_T_Q4_K created: %s (max threads: %d)\n", p_gemv_q4k.name.c_str(), p_gemv_q4k.max_threads);
    return 0;
}
