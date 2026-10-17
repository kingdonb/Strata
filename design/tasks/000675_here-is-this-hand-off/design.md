# Design: Qwen3.6 Metal Long Context, Chunked Prefill and MTP

## Current state (verified in repo)

- `metal/src/engine/q36_metal.cpp` (883 lines): one-token `forward_token` graph per step.
  - Attention dispatch at L324-338 sets `d.tg_mem = (s.pos + 1) * sizeof(float)` — threadgroup scores, the
    32 KB / 8,192-token ceiling. No friendly validation above 8K today.
  - KV cache fp32 at L609-613: `n_head_kv * ctx * head_dim * 4` bytes per layer (2 × 256 × 4 = 4 KB/layer/token;
    10 attention layers → 40 KB/token; 1.34 GB at 32K).
  - MTP weights skipped at L523 (`blk.<n_layers>.*` is dropped when wrapping the GGUF).
  - Host sampling at L626-673 scans ~248K logits per token (top-p sorts the full vocabulary).
  - Serve IPC (`READY <ctx> stop`, `GEN`, `T <id>`, `DONE`, `STOP`) and live-token prefix cache at L755-771.
- `metal/kernels/q36.metal`: `k_rms k_conv k_gdn k_gated_norm k_attn_prep k_attn k_swiglu k_add k_moe_sum
  k_topk_router`; `metal/kernels/quant.metal`: dequant GEMVs (`k_gemv_T_*`, `k_gemv_lpr8_T_*`, `k_row_T_*`).
- Build: `metal/CMakeLists.txt` globs `src/engine/*.cpp` into `strata_metal_kernels`, builds a Metal `strata`
  binary from `src/program/generate.cpp` with `STRATA_HAVE_QWEN35=1`; `--native` routes to `qwen35::run`.
  Kernels are runtime-compiled from `metal/kernels/*.metal` via `cx.compile({...})`.
- Geometry (`include/strata/core/{geometry,family_layout}.hpp`, read from GGUF): 24 query heads, 2 KV heads,
  head_dim 256, 40 trunk layers + `blk.40` MTP (`nextn_predict_layers`), GDN layers where `l % 4 != 3`.

## Prior art this design copies (do not re-invent)

1. **CUDA/HIP q36 engine** `src/qwen35/q36_engine.cu` (3,976 lines) + `docs/QWEN36.md`:
   - Split-K attention: per (query row, 256-cell chunk) partials `pacc / pm / pl`, then a merge kernel; used
     for both prompt rows and (short-context) decode. ~L298-355.
   - Chunked prefill through `strata::prefill::Gemm` / `moe_mmq` (`include/strata/prefill/`); MoE prompt
     kernels read every expert once per chunk; chunk ladder 512 → 4096 picked by a memory planner;
     measured pp4096 6,220 tok/s on R9700. `--prefill 512` restores the old text bit-for-bit.
   - GDN conv in tiles of 16 tokens side by side; delta rule sequential per token, one token ahead, two
     barriers — "both exact". Chunked (WY) GDN / associative scan: not attempted there.
   - MTP: `nextn.enorm/hnorm/shared_head_norm` + `eh_proj` over `[enorm(e) | hnorm(h)]`, draft-verify with
     solo-path gating; measured 1.60x solo on R9700; greedy text identical with and without drafts.
   - Verification culture: `STRATA_Q36_VERIFY=1` runs old beside new kernels bit-compared on real activations;
     10 parity prompts greedy 128; gates in `tools/qwen35/` speak the engine's serve protocol directly.
2. **Metal Flash-Next window** `metal/src/engine/window.cpp` + `prompt_chain.inc`: chunks of C rows through
   prompt kernels (`gdn_prompt.metal`, `qsa_prompt.metal`, `moe_mm.metal`), `STRATA_METAL_PROMPT=window`
   fallback and `STRATA_METAL_PROMPT_MIN` short-prompt fallback pattern, kernel parity tests
   (`metal/tests/*_parity.cpp`). Its MoE chunk kernel uses Metal 4 `matmul2d` — M5 only, not usable on M2.

## Key decisions

### A. Split-K attention (replaces both hand-off options for Priority 1)
The hand-off offered "online softmax" or a `[H, max_ctx]` scratch. The repo's proven answer is better: the
CUDA engine's split-K partials, which the Flash-Next Metal window already mirrors (`qsa_prompt.metal`: "the
same split-K partials as qsa_attn"). One kernel pair serves decode above 8K **and** prompt chunks (Phase B),
and its math order matches the CUDA engine (cross-backend parity).

- `k_attn_part`: grid (chunk-of-256 cells, query-row block); each block computes scores over its cells,
  keeps running max `m`, sum `l`, accumulator `acc[256]` (fp32) → device partials.
- `k_attn_merge`: combines chunks, normalizes, applies the sigmoid gate → `s.att`.
- Scratch = query-row-block × ceil(n_pos/256) × (256 + 2) floats. Decode (24 rows): ~0.8 MB at 8K, ~12.7 MB
  at 128K. Prompt: row-block the chunk (≤256 rows per block) so partials stay ≤ ~1 GB at 32K.
- The existing `k_attn` **stays the default for `n_pos ≤ 8192`** (byte-identical default path, per AGENTS.md);
  split-K engages above it, plus a test-only env knob to force it at short ctx for parity checks.
- Startup validation: refuse ctx that KV + S + conv state + chunk scratch cannot hold, printing the numbers
  (CUDA refusal style). `READY <ctx> stop` protocol unchanged.

### B. Chunked prompt prefill (Priority 2, shaped by what CUDA measured)
- `forward_prompt_chunk(T)` in `q36_metal.cpp`, mirroring the CUDA engine's chunk planner:
  - Dense projections (attn q/k/v/o, shared expert, router, `ssm_out`, alpha/beta small projections,
    embedding rows) become batched GEMMs over T rows that read each weight **once** per chunk (new
    `k_gemm_*` kernels; not T× GEMV re-reads — that would multiply weight traffic by T).
  - MoE: group the chunk's T × top-8 (token, expert) entries by expert (sort, per-expert runs, padding),
    one fp32-accum GEMM per expert run reading each expert once per chunk; weighted combine + shared expert
    after. The `moe_mm.metal` plan without `matmul2d` (M2 has no MPP); revisit an M5 path later.
  - GDN layers: conv kernel tiled 16 tokens side by side; delta rule runs token-by-token within the chunk
    (exact, same as CUDA). No associative scan.
  - QSA layers: Phase A split-K over the chunk's rows (row-blocked partials).
  - LM head and logits only for the chunk's last row; intermediate rows only update states.
- Chunk size: default 512 (the bit-friendliest baseline; CUDA's `--prefill 512` equals the old text), ladder
  1024/2048/4096 via `--prefill N` + env. Prompts shorter than the chunk fall back to token-by-token.
  Flip the default only on measured numbers (see requirements AC 2).
- Opt-in first (`STRATA_Q36M_PREFILL=1` or equivalent), so the released decode path is untouched until gates
  pass.

### C. MTP speculative decoding (Priority 3, direct CUDA port)
- Load `blk.40` nextn weights when `--mtp > 0` (keep the skip when off): `nextn.enorm.weight`,
  `nextn.hnorm.weight`, `nextn.shared_head_norm.weight`, the block's layer weights, `eh_proj`.
- Draft step: one MTP-layer forward (norms + eh_proj + the block's attention/MoE graph reusing existing
  pipelines) proposes up to 2 tokens.
- Verify window: up to 3 rows (1 held + 2 drafts) through the Phase B chunk machinery; accept the matching
  prefix, resample at the first mismatch from the target logits. Solo path only.
- Opt-in; default `--mtp 0` keeps the binary's behavior byte-identical.

### D. Deferred on numbers (hand-off Priorities 4-5)
- **FP16 KV cache**: CUDA ships fp32 KV; at 32K the fp32 cache is 1.34 GB on a 24 GB box — affordable. fp16
  changes rounding → text differs, so it can never be a silent default. Keep as an opt-in follow-up
  (`--kv-f16`, unpack in the attention kernels) behind a measurement.
- **simdgroup_matrix / AMX GEMVs**: decode is bandwidth-bound (CUDA profiling: the big products run at
  530 GB/s = the card's bandwidth; M2 ~100 GB/s). A compute-side rewrite does not move decode. Out of scope.
- **Sampling**: measure host cost first; a GPU argmax kernel (pattern: `sample.metal` in the Flash-Next
  window) only if the profile justifies it, preserving the RNG stream and chosen tokens.

## Verification plan
- Build on the Mac: `/opt/homebrew/bin/cmake --build build-metal`; `./build-metal/q36_compile_test` (extended
  to the new pipelines).
- Parity: the 10 parity prompts greedy 128 through `GEN` against the pre-change build (default paths); forced
  split-K vs `k_attn` at 4K; chunked vs token-by-token; `--mtp 2` vs `--mtp 0`. Gates: `tools/qwen35/
  q36_solo_gate.py` / `q36_cmp_gate.py` already speak the serve protocol and point at the Metal engine.
- Numbers: per-step GPU ms at 4K/16K/32K; prefill tok/s per chunk size; decode ratio with/without MTP —
  same build, same Mac, same day, recorded in docs/QWEN36_METAL.md.
- Only `metal/` files and docs change; the CUDA/HIP/SYCL builds are untouched (if a shared header under
  `include/strata/` ever needs a touch, rebuild all backends it compiles into, per AGENTS.md).

## Risks
- 24 GB pressure at 32K ctx with a 512+ chunk (partials + KV + 11.4 GB weights): the chunk planner must
  budget like CUDA's; refuse with numbers rather than swap-thrash.
- The grouped MoE chunk GEMM is the one genuinely new kernel on M2 — the biggest perf unknown. Fallback:
  token-by-token prefill stays (the feature is opt-in).
- Split-K summation order differs from `k_attn`: token-level parity is the gate; a divergence keeps the
  two-path design (default kernel ≤8K) and the case is investigated before merge.

## Learnings (for future agents)
- Truth for model shape lives in `include/strata/core/geometry.hpp` / `family_layout.hpp`; family defaults
  are Flash-Next's (24/2/256, 48 trunk layers); qwen35moe overrides come from the GGUF (40 trunk + 1 MTP).
- `.metal` files are runtime-compiled by filename (`cx.compile({"quant.metal", "q36.metal"})`) from the
  kernels dir wired as `STRATA_METAL_ENGINE_KERNELS_DEFAULT`; adding kernels = drop-in + `cx.pipeline(lib, ...)`.
- Engine entry: `metal/CMakeLists.txt` → `strata` binary; `src/program/generate.cpp` `--native` →
  `qwen35::run(argc, argv)`; the Python server (`serve/server.py`) spawns this process and parses
  `READY/T/DONE` — engine-local flags ride through argv, no server change expected.
- The same serve protocol means the CUDA gates in `tools/qwen35/` double as Metal gates — reuse them before
  writing new harnesses.
