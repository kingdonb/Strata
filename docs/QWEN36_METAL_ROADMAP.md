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

State: measured and decided (M2, UD-Q2_K_XL, 2026-10-18) — the default flip is **not taken**: the full
matrix (`tools/qwen35/q36_prefill_bench.py`; the table and the decision in docs/QWEN36_METAL.md §6) reads
the chunk path at 1.24-1.51x the token-by-token rate (2K 14.7 → 22.3 tok/s at chunk 512, 8K 10.0 → 12.7),
nowhere near the 5x the flip needs, so the no-flag default stays token-by-token and `--prefill auto` stays
512 (the best measured chunk; 1024 and 2048 read 1.5-3% slower at every size — no bigger-chunk win here
like the CUDA engine's). The earlier first points still explain the shape: short prompts (~20-40 tokens):
TTFT ~30% faster with `--prefill 512`; a 4,098-token prompt at `--ctx 16384`: chunked 512 ≈ token-by-token
(243 vs 245 s to the first token) — at depth the read is bound by the attention's KV reads (every query
head of a KV group re-reads the same cells, ~8x), a term the parity contract keeps identical on both
paths, and it grows with position (per-chunk 22 → 40 s). The lever this points at: a GQA-folded attention
pass (one K/V read per KV-head group — same FLOPs, ~1/8 the KV traffic at depth) would move the chunked
read and the deep decode together; that is T6's kernel work. Remaining from T3's scope: the 10-prompt
parity at the shipping default (no-flag vs `--prefill 0`), which doubles as the check that the scaffolding
commit (feature/000678) changed no token.

## T4 — MTP speculative decoding (`--mtp 2`)

State: green on the delivery branch `feature/000679-metal-mtp-speculative` (Apple M2, 2026-10-18):
the `blk.40` nextn weights load when `--mtp > 0`, the draft step and the verify window run the
CUDA engine's spec_step semantics through the T2 chunk kernels (window-scratch S/conv states
copied from the real ones first — the checkpoint — and a commit that safely applies kept rows into
the real states), greedy-only, opt-in, off by default. Gates verified on M2: `q36_chunk_parity` all-OK
(38/38 bit-identical); 10-prompt parity twice vs `--mtp 0` (`identical: 10 of 10` twice); Q2_K GEMV kernel
implemented for the draft block's expert weights. Measured decode speed is 5.6–11.7 tok/s (the ≥1.2x
threshold is not reached without a draft-vocab subset head), so `--mtp` remains strictly opt-in (`--mtp 0`
default) per the roadmap rule.
Scope: load the `blk.40` nextn weights when `--mtp > 0`; the draft step; the verify window;
greedy-only, opt-in, off by default.
Gates: `--mtp 2` greedy text identical to `--mtp 0` on the 10 parity prompts; the measured decode ratio
on the same Mac and day; it stays opt-in unless ≥1.2x is measured. Depends on T2.

## T5 — FP16 KV cache and the sampling profile

Scope: `--kv-f16` (opt-in, never a default) with fp32 unpack in the attention kernels; the measured
memory difference at 8K and 32K; the documented text difference. Plus: profile host sampling per token
(greedy vs temp+top-p, the ~248K vocab); add a GPU argmax / top-k prefilter only if the profile shows
≥2 ms/token, preserving the RNG stream and the chosen tokens.
Gates: the memory and profile numbers measured; sampled parity with a fixed seed.

## T6 — GQA-folded attention: one K/V read per KV-head group (new idea, from T3's first measurements)

State: built on the delivery branch `feature/000681-gqa-folded-attention-one` (2026-10-19): the folded
kernels (`k_attn_gqa_part`/`k_attn_gqa_merge` for the `k_attn` range, `k_attn_sk_part_gqa` above it with
`k_attn_sk_merge` unchanged), the `STRATA_Q36M_ATTN_GQA=1` parity knob, the H x ctx score scratch and the
startup guards; a host-side bit-parity simulation of both kernel structures shows every partial and output
bit-identical to the unfolded kernels (17 shapes, spec task 000681's `gqa_parity_sim.cpp`). Pending on the
M2: the compile test, the forced-fold 10/10 ladder (plus the SK and --mtp cross-checks), the chunk ladder
with the fold on, and the measured read/decode deltas for docs/QWEN36_METAL.md §8. The measurement it comes
from (M2, UD-Q2_K_XL, build
`36b394c8` + PR #5): a 4,098-token prompt at `--ctx 16384` reads in ≈243 s chunked and ≈245 s token-by-token,
per-chunk cost growing with position (22 → 40 s per 512), and the decode at pos ≈4,100 runs 8 tok/s vs 17 at
the ctx-4096 ladder's shallow positions. At depth, both paths are bound by the attention's KV reads — and
the model is grouped-query: ~8 query heads share each KV head, and today every one of them re-reads the same
K/V cells (an ~8x traffic multiplier L2 is not catching). The parity contract keeps that term identical on
both paths, which is exactly why chunked prefill cannot win at depth: the term it would have to beat is not
the weights (which it does read once per chunk) but this re-read.
Idea: an attention pass that reads each KV-head group's K/V ONCE and computes all its query heads' scores
against it — same FLOPs, ~1/8 the KV traffic at depth (the exact factor is the model's H/HKV, to be read
from the geometry, not assumed). Three places the same idea applies, one at a time: the decode `k_attn`
(one threadgroup per head today), the chunk path's `k_attn_sk_part` row-blocked pair, and the non-split-K
chunk rows. Same FLOPs means the per-head dot keeps its accumulation order — bit-parity is the design
target, not an approximation to be tolerated.
Scope: the folded kernel(s) behind a parity knob first (`STRATA_Q36M_ATTN_GQA=1`, the
`STRATA_Q36M_ATTN_SK` pattern), the decode step and the chunk path through one shared helper (the
`02d7aeec` pattern), the geometry guard (refuse H/HKV = 1 where folding is a no-op), then engage where
measured.
Gates: the compile test; forced-fold 10/10 token-identical vs the current path at 4K (the 10 parity
prompts, greedy); the chunk ladder still 10/10 twice with the fold on; the measured read and decode
deltas on the same Mac and day — the 4,098-token read at 16K and the decode tok/s at pos ≈4,100, before
and after, numbers in docs/QWEN36_METAL.md; the default path byte-identical until a measurement says the
fold flips it. Depends on nothing in T2-T5 (standalone); T7's wrap-up absorbs its doc numbers if it
lands first.

## T7 — Wrap-up: docs, the batch upstream PR, maintenance

Scope: docs/QWEN36_METAL.md final numbers (what, hardware, date) replacing the hand-off's projections;
every gate green on the M2 (build, `q36_compile_test`, the tools/qwen35 gates); confirm the CUDA/HIP
builds (the one shared-header accessor); assemble the batch upstream PR from the merged ticket PRs; offer
to maintain the new sections (`metal/kernels/`, `metal/src/engine/q36_metal.cpp`, the qwen35 q36 gates)
following the contributing conventions — docs/TEST_REQUESTS.md and docs/COMMUNITY_BENCHMARKS.md are the
follow-through paths for hardware the maintainers do not have.
