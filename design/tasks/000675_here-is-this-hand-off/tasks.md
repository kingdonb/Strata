# Implementation Tasks: Qwen3.6 Metal Long Context, Chunked Prefill and MTP

All work on branch `feat/qwen36-metal`, built and tested on the Mac (`/opt/homebrew/bin/cmake --build
build-metal`). No CUDA/HIP/SYCL files are touched.

Reviewer answers (approved design): M2 is the only test Mac (user runs intermediary tests on pushed
intermediates); do what upstream did on the prefill default (flip after measured gates — default flip lands
only after the Mac run passes); solo-path MTP with fixed `--mtp 2` is fine; include the opt-in FP16 KV cache;
host-sampling check in scope (GPU argmax / top-k prefilter only if profiling shows ≥2 ms/token).

NOTE (sandbox constraint): this sandbox is Linux — nothing Metal compiles or runs here. Every phase ends with
a push to `feat/qwen36-metal` and a request to the user to build + run the phase's gates on the M2 before the
next phase proceeds. Host C++ gets a `g++ -fsyntax-only` pass where headers allow it; MSL is checked by
careful mirroring of proven kernels and by the Mac compile test.

## Phase A — Split-K attention, uncap the context window

- [x] Add `k_attn_part` + `k_attn_merge` to `metal/kernels/q36.metal`, copying the partial layout
      (256-cell chunks, acc/m/l) from `src/qwen35/q36_engine.cu` (~L298-355) — `k_attn_sk_part` /
      `k_attn_sk_merge` pushed in 65037a91
- [x] Wire dispatch in `metal/src/engine/q36_metal.cpp`: split-K only when `n_pos > 8192`; keep `k_attn`
      default below; add a test-only env knob to force split-K at short ctx — `STRATA_Q36M_ATTN_SK=1`
- [x] Allocate partials/merge buffers in `State`; row-block the partials so prompt use stays bounded —
      `SK_ROWS_MAX = 256` rows per dispatch, decode uses `n_head`
- [x] Startup ctx validation: refuse a context that KV + S + conv state + scratch cannot hold, printing the
      numbers; leave `READY <ctx> stop` and `--ctx` parsing unchanged — enforced only above 8,192;
      `STRATA_Q36M_MEM_OK=1` overrides
- [x] Extend `metal/tests/q36_compile_test.cpp` with the new pipelines
- [~] Verify on the Mac (user runs): 10 parity prompts greedy 128 unchanged at ctx ≤ 8192; forced split-K
      token-identical to `k_attn` at 4K; 2K-token prompt + 128 gen at ctx 16384 and 32768; record per-step
      GPU ms at 4K/16K/32K
- [ ] Verify on the Mac: 10 parity prompts greedy 128 unchanged at ctx ≤ 8192; forced split-K token-identical
      to `k_attn` at 4K; 2K-token prompt + 128 gen at ctx 16384 and 32768; record per-step GPU ms at 4K/16K/32K

## Phase B — Chunked prompt prefill

- [x] Batched dense GEMM kernels (T rows, weights read once per chunk) — implemented as weight-stationary
      batched GEMVs (`k_gemv_c`, `k_gemv_lpr8_c` for n_in=2048) with the token loop inside the threadgroup:
      simpler than a tiled GEMM and each (token, output) keeps the decode GEMV's exact summation order, so
      chunk results are bit-identical to token-by-token rather than merely close
- [x] Grouped MoE chunk GEMM: entry tables on GPU (`k_zero`/`k_hist`/`k_exp_off`/`k_place`, device atomics),
      per-expert runs walked by `k_expc` (each expert read once per chunk), combine via `k_moe_sum_chunk`
      reading through the per-token slot table (fixed slot order, so placement order cannot change digits)
- [x] GDN chunk kernels: `k_conv_chunk` (16-token tiles side by side, tile 0 stores the history — the CUDA
      engine's k_conv_seq tile mode, k_conv's math) + `k_gdn_chunk` (delta rule sequential per token in one
      launch, k_gdn's math, states L1-hot) + `k_gated_norm_chunk`
- [x] QSA chunk attention: `k_attn_prep_chunk` (rows in the split-K kernels' row order) + the Phase A
      split-K kernels row-blocked (SK_ROWS_MAX rows per dispatch); split-K kernels gained a `row0` arg
- [x] `forward_prompt_chunk(T)` + `--prefill N` / `STRATA_Q36M_PREFILL=1` (512); prompts shorter than the
      chunk run as one partial chunk; without the flag the engine behaves exactly as before (the default
      flip lands only after the Mac parity + speed gates pass — reviewer: do what upstream did)
- [x] Head/logits only via the last prompt token's decode window; prefix-cache state reuse untouched
- [~] Parity + measurement (user, on the Mac): chunked vs token-by-token greedy text on the 10 parity
      prompts + ≥3 real documents (expected identical by construction; log KL if not); prefill tok/s per
      chunk size vs token-by-token, recorded in docs/QWEN36_METAL.md

## Phase C — MTP speculative decoding

- [ ] Load `blk.40` nextn weights (`nextn.enorm/hnorm/shared_head_norm`, `eh_proj`, block layer weights) when
      `--mtp > 0`; keep the current skip when off
- [ ] MTP draft step: one MTP-layer forward proposing up to 2 tokens
- [ ] Verify window (≤3 rows) through the Phase B chunk machinery: accept prefix, resample at first mismatch;
      stop tokens, `STOP` cancel, length and ctx-end guards respected
- [ ] Gate: `--mtp 2` greedy text identical to `--mtp 0` on the 10 parity prompts; measured decode tok/s ratio
      on the same Mac/day; keep opt-in unless ≥1.2x measured

## Phase D — Measured follow-ups (decide on numbers)

- [ ] Profile host sampling per token (greedy vs temp+top-p, ~248K vocab); if ≥2 ms/token, add GPU argmax /
      top-k prefilter preserving the RNG stream and chosen tokens, re-verify sampled parity with fixed seed
- [ ] Optional opt-in FP16 KV cache (`--kv-f16`) with fp32 unpack in the attention kernels; document the text
      difference and measured memory numbers; never a default
- [ ] Fix or restore the `./run-q36.sh` reference in docs/QWEN36_METAL.md §5

## Wrap-up

- [ ] Update docs/QWEN36_METAL.md: measured Metal numbers (what, hardware, date) replacing the hand-off's
      projections; note the associative-scan and AMX out-of-scope rationale
- [ ] Full gate run on the Mac green (build, `q36_compile_test`, parity gates through `tools/qwen35/`) before
      review; confirm CUDA/HIP builds untouched (or rebuilt if any shared header was edited)
