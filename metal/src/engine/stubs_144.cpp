// metal/src/engine/stubs_144.cpp - the 0.1.44 PC entry points the Metal engine does not implement (merge of the Mac branch
// with the PC release): new overloads / new functions that stubs.cpp (generated) does not know.  Each either stops the engine
// with its name (like stubs.cpp) or, where the caller has a documented fallback, answers "not available".  None is reached
// with the Metal defaults: STRATA_COMMIT_BATCH is off on Metal, the NVFP4 CPU rows need AVX2 / AVX-512, and the K-quant /
// NVFP4 prompt paths are CUDA / HIP only.

#include <cstddef>
#include <cstdint>

#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/prefill/kernels.hpp"

namespace strata::metal_engine {
void missing(const char* name);
}

namespace strata::kernels {
void iq_dequant_f16(int, const void*, int64_t, uint16_t*, void*, const float*) { strata::metal_engine::missing("strata::kernels::iq_dequant_f16(scale)"); }
void iq_dequant_gu_f16(int, const void*, const void*, int64_t, int64_t, uint16_t*, void*, const float*) { strata::metal_engine::missing("strata::kernels::iq_dequant_gu_f16(tail)"); }
bool gdn_commit_layers(float*, int64_t, int64_t, const float*, int64_t, const float*, int64_t, const float*, const float*, int64_t,
                       int, int, int, int, int, const int32_t*, void*) { return false; }   // "nothing launched: keep the per-layer loop"
void native_qsa_indexer_append_steps_layers(const float*, int64_t, const int32_t*, int, int, int32_t, float, const QsaIndexerLayers&, int,
                                            const QsaShapes&, int64_t, const RopeScaling&, void*) {
    strata::metal_engine::missing("strata::kernels::native_qsa_indexer_append_steps_layers");
}
}  // namespace strata::kernels

namespace strata::prefill {
void moe_combine(const float*, const int32_t*, const float*, const float*, const float*, float*, int64_t, void*, const float*) {
    strata::metal_engine::missing("strata::prefill::moe_combine(row_sd)");
}
}  // namespace strata::prefill

namespace strata::kernels::cpu {
bool nvfp4_256_fits(int) { return false; }
bool nvfp4_512_fits(int) { return false; }
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_256_gu_rows");
}
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_256_rows");
}
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_512_gu_rows");
}
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_512_rows");
}
}  // namespace strata::kernels::cpu

// STRATA_QSA_SLOTS (PR #2364): the one-launch-over-the-rows QSA launchers have no Metal kernel; false = "run the per-row calls",
// which is what Verifier::record_window does on a decline.
namespace strata::kernels {
bool qsa_block_scores_slots(const float* const*, const float* const*, const float*, const int32_t*, int64_t, int64_t,
                            const QsaShapes&, float*, void*) { return false; }
bool qsa_decode_attn_slots(const float*, const QsaAttnSlotPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&,
                           float*, float*, int64_t, void*) { return false; }

void fused_gr_persist_init() {}
bool fused_gr_read_persist(const FusedGrArgs*, int, void*) { return false; }
void hc_q8_to_bf16(const void*, const float*, uint16_t*, long long, void*) {
    strata::metal_engine::missing("strata::kernels::hc_q8_to_bf16");
}
void qsa_block_scores_rows(const QsaScoreRows&, const float*, const int32_t*, int64_t, int64_t, const QsaShapes&, float*, void*) {
    strata::metal_engine::missing("strata::kernels::qsa_block_scores_rows");
}
bool qsa_decode_attn_split(const float*, const QsaAttnPools&, const int32_t*, const int32_t*, int64_t, const QsaShapes&, float*, unsigned long long, float*, int64_t, void*) {
    return false;
}
void qsa_decode_attn_batch_rows(const float*, const QsaAttnPoolsRows&, const int32_t*, const int32_t*, int64_t, const QsaShapes&, float*, float*, int64_t, void*) {
    strata::metal_engine::missing("strata::kernels::qsa_decode_attn_batch_rows");
}
void native_quantize_q8_1_il(const float*, void*, void*, int, int, void*) {
    strata::metal_engine::missing("strata::kernels::native_quantize_q8_1_il");
}
void wait_copy_i32_from_mapped(const uint32_t*, uint32_t, int32_t*, const int32_t*, int64_t, void*) {
    strata::metal_engine::missing("strata::kernels::wait_copy_i32_from_mapped");
}
void res_patch(int32_t*, const int32_t*, int, void*) {
    strata::metal_engine::missing("strata::kernels::res_patch");
}
void copy_rows_from_mapped_after(const uint32_t*, uint32_t, float*, const float*, int64_t, int64_t, const int32_t*, const int32_t*, void*) {
    strata::metal_engine::missing("strata::kernels::copy_rows_from_mapped_after");
}
void gdn_step_norm_groups(const GdnGroups&, const float*, int, const float*, const float*, const float*, const float*, float, float*, int, int, bool, void*) {
    strata::metal_engine::missing("strata::kernels::gdn_step_norm_groups");
}
void handoff_publish(float*, const float*, int64_t, const float*, int64_t, const float*, int64_t, uint32_t*, uint32_t*, void*, const uint32_t*) {
    strata::metal_engine::missing("strata::kernels::handoff_publish");
}
void wait_handoff(const uint32_t*, uint32_t, uint32_t*, unsigned long long, void*) {
    strata::metal_engine::missing("strata::kernels::wait_handoff");
}
void wait_flag_ge_stamped(const uint32_t*, uint32_t, unsigned long long*, unsigned long long*, void*) {
    strata::metal_engine::missing("strata::kernels::wait_flag_ge_stamped");
}
}  // namespace strata::kernels

namespace strata::kernels::cpu {
void iq256_gu_rows_exact_one(int, const uint8_t*, size_t, size_t, int, const void*, float*, int, int) {
    strata::metal_engine::missing("strata::kernels::cpu::iq256_gu_rows_exact_one");
}
}  // namespace strata::kernels::cpu

namespace strata::prefill {
bool gemm_f16_range_read(float[2], unsigned long long[2], bool) { return false; }
bool f16_range_read(float&, unsigned long long&, bool) { return false; }
void gr_norm(const float*, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*, bool) {
    strata::metal_engine::missing("strata::prefill::gr_norm");
}
void gr_norm_rs(const float*, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*, int64_t, bool) {
    strata::metal_engine::missing("strata::prefill::gr_norm_rs");
}
void gr_write_norm_rs(float*, const float*, const float*, int64_t, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*, int64_t, bool) {
    strata::metal_engine::missing("strata::prefill::gr_write_norm_rs");
}
void gr_write_cvec_norm_rs(float*, const float*, const float*, int64_t, const float*, const float*, const int32_t*, int, const float*, float, float*, uint16_t*, int64_t, void*, uint16_t*, int64_t, bool) {
    strata::metal_engine::missing("strata::prefill::gr_write_cvec_norm_rs");
}
}  // namespace strata::prefill
