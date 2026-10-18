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

State: green on the delivery branch (through `36b394c8`, Apple M2, 2026-10-18). `q36_chunk_parity` stands
at all 38 stages bit-identical (the 38th = the MoE isolation combine). The engine ladder (the 10 parity
prompts, token-by-token vs `--prefill 512`, greedy) is 10/10 twice — the chunk path self-consistent across
runs and token-identical to the reference json — and the default path stayed byte-identical across the
whole chunk effort (the base gate on the fixed build vs the pre-chunk build's base json, 10/10). Getting
there surfaced and fixed four real defects, each localized with the stage harness and the per-layer
FNV-hash trace (`f54f90ad`): the chunk gather over-dispatched 128x and wrote past its buffers
(`eac5e9a7`); the whole-chunk single ~600-dispatch graph and `MTLDispatchTypeConcurrent` encoding raced on
the M2 (`f6f57bd7`, `62eba361`); the chunk's attention ran the split-K arithmetic where the decode runs
per-row `k_attn` — both paths now run the identical per-row dispatch sequence (`02d7aeec`); and
`block_sum`'s broadcast read raced the next call's slot reuse — an intra-dispatch race invisible to every
commit structure, the harness's small grids and every float-sum check until the bit hashes pointed at
(layer 0, token 19) (`36b394c8`).
Scope: close the combine question, bring the parity harness to all-OK, then make the engine ladder (the
10 parity prompts, plain vs `--prefill 512`) token-identical — closed.
Gates: `q36_chunk_parity` all-OK on the M2 (38/38, 2026-10-18); the ladder 10/10 with the json compare
(twice, 2026-10-18, build `36b394c8`); the default path byte-identical (base vs base, 10/10). The chunked
prefill stays opt-in (`--prefill N`, default 0 = token-by-token; the chunk pipelines compile only when
the flag is on) — the measured default flip is T3's.

## T3 — Chunked prefill: measured speed and the default flip

State: unblocked — T2 green (M2, 2026-10-18). First measured points (M2, UD-Q2_K_XL, 2026-10-18):
short prompts (~20-40 tokens): TTFT ~30% faster with `--prefill 512`; a 4,098-token prompt at
`--ctx 16384`: chunked 512 ≈ token-by-token (243 vs 245 s to the first token) — at depth the read is
bound by the attention's KV reads (every query head of a KV group re-reads the same cells, ~8x), a term
the parity contract keeps identical on both paths, and it grows with position (per-chunk 22 → 40 s).
The lever this points at: a GQA-folded attention pass (one K/V read per KV-head group — same FLOPs,
~1/8 the KV traffic at depth) would move the chunked read and the deep decode together; that is kernel
work for its own ticket. The `--prefill` default stays token-by-token until a measurement says faster;
`--prefill auto` resolves to 512 (never worse: ~20% better on short prompts, a wash at depth).
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
