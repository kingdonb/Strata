# Qwen3.6 Metal — delivery plan

The Metal work lands as a sequence of one-change pull requests (the project's contributing rule), each
verified on an Apple M2 before it moves. This file is the plan the follow-up tickets work from.
docs/QWEN36_METAL.md §4's priorities map onto the tickets (1 → T1, 2 → T2/T3, 3 → T4, 4 → T5; 5, the
SIMD-group matrix ops, stays out of scope — see the report's concession notes). The work-in-progress
implementation lives on the delivery branch `feat/qwen36-metal`, which this PR is the first slice of.

## T0 — the Metal port (this PR; delivered)

The engine (`metal/src/engine/q36_metal.cpp`), the kernels (`metal/kernels/q36.metal`,
`metal/kernels/quant.metal`), the build integration, the download tool, and docs/QWEN36_METAL.md.
Measured on the M2: 12.5–14.5 tok/s decode, token-by-token prefill, 8,192-cell context cap. Every
non-Metal backend untouched; the one shared-header change is a read-only accessor in
`include/strata/artifact/gguf_reader.hpp` (`data()`), with no call-site changes.

## T1 — Split-K attention: context past the 8K ceiling

State: built and M2-verified on the delivery branch (commit `65037a91` plus the MSL enum fix); staged as
branch `feat/qwen36-splitk`.
Scope: `k_attn_sk_part` / `k_attn_sk_merge` (one partial per (query row, 256-cell chunk), a merge that
folds and gates), `SK_ROWS_MAX` row-blocking, auto-engage above 8,192 cells, the `STRATA_Q36M_ATTN_SK=1`
parity knob, and the `--ctx` working-set refusal with the numbers (`STRATA_Q36M_MEM_OK=1` overrides).
Gates: the compile test; split-K forced 10/10 token-identical vs the default at 4K; a 32K context loads
and generates; the default path unchanged at 8,192 and below.

## T2 — Chunked prompt prefill: parity green

State: all chunk kernels plus `forward_prompt_chunk` and the `--prefill` knob are built on the delivery
branch (commit `d93247da` onward); the parity harness (`metal/tests/q36_chunk_parity.cpp`) stands at
36/37 stages bit-exact on the M2; the routed-expert root cause is fixed and verified (the gather plus
per-entry decode-lpr8 gate/up: the eh and ey stages green). The one failing stage (`moe2048 x`) is pinned
to a GPU-visibility anomaly on a memcpy-seeded TEST buffer — the decode side is proven correct (the
host-recomputed sum equals the GPU delta) — and the engine's chunk ladder is still 0/10.
Scope: close the combine question (the probes are designed: GPU-side copies of the host-written buffers
before and after the decode chain, plus an isolation re-run of the combine on a freshly seeded buffer),
bring the parity harness to all-OK, then make the engine ladder (the 10 parity prompts, plain vs
`--prefill 512`) token-identical.
Gates: `q36_chunk_parity` all-OK on the M2; the ladder 10/10 with the json compare.

## T3 — Chunked prefill: measured speed and the default flip

State: blocked on T2.
Scope: TTFT and tok/s, chunked vs token-by-token, on 2K+-token prompts, same Mac, same day, same build;
flip the `--prefill` default only with the numbers in hand (the project rule: change a default only where
measured faster); re-run the 10-prompt parity at the new default; document the numbers in
docs/QWEN36_METAL.md.

## T4 — MTP speculative decoding (`--mtp 2`)

State: not started; the design is settled (the CUDA engine's spec_step mirrored: a redo-row draft call, a
chained draft seeded by the block's own output, a verify window with per-row head and argmax,
accept-prefix rewind by checkpointing S/conv and re-running the kept rows through the chunk path).
Scope: load the `blk.40` nextn weights when `--mtp > 0`; the draft step; the verify window; greedy-only,
opt-in, off by default.
Gates: `--mtp 2` greedy text identical to `--mtp 0` on the 10 parity prompts; the measured decode ratio
on the same Mac and day; it stays opt-in unless ≥1.2x is measured. Depends on T2.

## T5 — FP16 KV cache and the sampling profile

Scope: `--kv-f16` (opt-in, never a default) with fp32 unpack in the attention kernels; the measured
memory difference at 8K and 32K; the documented text difference. Plus: profile host sampling per token
(greedy vs temp+top-p, the ~248K vocab); add a GPU argmax / top-k prefilter only if the profile shows
≥2 ms/token, preserving the RNG stream and the chosen tokens.
Gates: the memory and profile numbers measured; sampled parity with a fixed seed.

## T6 — Wrap-up: docs, the batch upstream PR, maintenance

Scope: docs/QWEN36_METAL.md final numbers (what, hardware, date) replacing the hand-off's projections;
every gate green on the M2 (build, `q36_compile_test`, the tools/qwen35 gates); confirm the CUDA/HIP
builds (the one shared-header accessor); assemble the batch upstream PR from the merged ticket PRs; offer
to maintain the new sections (`metal/kernels/`, `metal/src/engine/q36_metal.cpp`, the qwen35 q36 gates)
following the contributing conventions — docs/TEST_REQUESTS.md and docs/COMMUNITY_BENCHMARKS.md are the
follow-through paths for hardware the maintainers do not have.
