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

## T4.5 — The draft-vocab subset head (`--draft-vocab N`, ticket 000682)

State: done — all gates green on the M2 (Apple M2, 2026-10-19, build `acd8c8a0`, UD-Q2_K_XL):
build + `q36_compile_test` clean; `q36_chunk_parity` 38/38; the subset's parity 10/10 vs `--mtp 0`
and vs `--mtp 2` (full head) across {4096, 8192, 16384}, and 10/10 with the GQA fold on
(`STRATA_Q36M_ATTN_GQA=1`); the traffic line verified (4.7–18.9 MB vs 286 MB full head, up to 60.6× cut).
Measured 5.0–11.1 tok/s against the ~17.5–21.4 base: the ≥1.2x rule is **not met** — with the head cut,
the verify window (three rows through 40 layers at a ~15–25% accept rate) is the round's dominant term — so
`--draft-vocab` ships strictly opt-in with the numbers recorded (docs/QWEN36_METAL.md §7). T4's
verdict named this as the missing piece; the piece is now in, and the M2's verdict is that the
window, not the head, is what keeps MTP net-negative there.
Scope: `--draft-vocab N` (the env `STRATA_Q36_DRAFT_VOCAB` as the fallback, the flag wins — the
CUDA engine's flag and env, `src/qwen35/q36_engine.cu`): the draft pick runs over the head's first
N rows in place (a shrunken view of the same tensor, row i = token id i — no gather, no map, no new
kernel); the startup stderr line prints the subset's rows and MB per draft step against the full
head's. Parity by construction (drafts only propose), so the T4 gate stands: `--mtp 2
--draft-vocab N` greedy text identical to `--mtp 0` and to `--mtp 2` on the 10 parity prompts,
twice. Strictly opt-in; 0 or absent = the whole vocabulary, as before.
Gates: the build + `q36_compile_test` green; the two 10/10 ladder compares; the no-flag base json
byte-identical; the measured `{4096, 8192, 16384}` table (tok/s, accept %, English/code and CJK)
recorded in docs/QWEN36_METAL.md §7; the ≥1.2x rule decides any docs recommendation, never a
default flip. Depends on T4.

## T5 — FP16 KV cache and the sampling profile

Scope: `--kv-f16` (opt-in, never a default) with fp32 unpack in the attention kernels; the measured
memory difference at 8K and 32K; the documented text difference. Plus: profile host sampling per token
(greedy vs temp+top-p, the ~248K vocab); add a GPU argmax / top-k prefilter only if the profile shows
≥2 ms/token, preserving the RNG stream and the chosen tokens.
Gates: the memory and profile numbers measured; sampled parity with a fixed seed.

State (2026-10-19): the KV half is implemented on `feature/000680-fp16-kv-cache-kv-f16-for` — the
`KV_T` typedef and the seven kernels' half caches under a `STRATA_KV_F16` prelude define (a second
runtime compile only flagged runs pay for), the engine's `--kv-f16` flag, selection, allocation math
(2 B/elem), the flagged working-set `est` term, and the parity harness's f16 mode with the cross
stage (both libs bit-identical on half-exact caches; the random-cache rel-L1 bounded 1e-3, the CUDA
`qsa_parity` precedent). Mechanism, kernel table and memory math: docs/QWEN36_METAL.md §9. The M2
gates all ran green the same day (2026-10-19, build `5bcf2485`): the compile test creates the f16
lib's seven pipelines; `q36_chunk_parity` is all-OK in both modes (the f16 cross stage bit-identical
on half-exact caches, the random-cache rel-L1 measured at 1.75e-4 vs the 1e-3 bound); the no-flag
base json is 10/10 against the pre-change build's (the byte-identical default path); two flagged
ladder runs are 10/10 identical, and the fp32-vs-f16 text difference is 7 of 10 prompts
token-identical, first divergence at generated tokens 46–100 of ≤128 (the positions in §9); the KV
lines read 0.17 GB (fp32 0.34) at 8K and 0.67 GB (fp32 1.34) at 32K; the SK/GQA/MTP cross-checks and
the `--draft-vocab` composition are all 10/10. The KV half is green — delivered, strictly opt-in.
The sampling
half (the profile, the ≥2 ms/token GPU argmax prefilter gate, the fixed-seed parity gate) moves to
its own card; the hours-class benchmark matrix (the deep-decode tok/s A/B the halved KV traffic
should buy) is deliberately deferred until the dimensional-improvements showcase — its ready-to-run
command is in §9.

## T6 — GQA-folded attention: one K/V read per KV-head group (new idea, from T3's first measurements)

State: built on the delivery branch `feature/000681-gqa-folded-attention-one` (2026-10-19): the folded
kernels (`k_attn_gqa_part`/`k_attn_gqa_merge` for the `k_attn` range, `k_attn_sk_part_gqa` above it with
`k_attn_sk_merge` unchanged), the `STRATA_Q36M_ATTN_GQA=1` parity knob, the H x ctx score scratch and the
startup guards. M2-run results: compile test green; forced-fold ladder 10/10; SK+GQA and MTP+GQA cross-checks
10/10; `q36_chunk_parity` 38/38. The first merge-kernel revision collapsed deep-decode occupancy (2 threadgroups,
per-head V re-walks past a too-small L1: 2.7-3.2 tok/s vs the 8 tok/s baseline at pos ~4,100) - fixed by hoisting
the head loop into the grid (merge grid (HKV, HG), the score pass position-tiled; bit-identity re-proven on the
host sim); the fixed pair's deep-decode and read deltas are the pending M2 measurement (docs/QWEN36_METAL.md §8's
commands), and the V-shared merge variant (one thread owning a dim's HG chains, V traffic 1x, occupancy capped
at HKV x D threads by the parity contract) is the follow-up experiment. The measurement it comes
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

---

## Phase 2: Post-0.2.0 Milestones — Measured Prefill Parity and Predictable Context

> **Author**: Gemini-3.8-Flash (original draft); reviewed and revised by Qwen-3.8-Flash-Next, 2026-10-19.  
> **Status**: Reviewed — the claim-by-claim audit, the corrected numbers and the evidence requests are in docs/QWEN36_METAL_REVIEW.md (its §5 is this section's review table).  
> - [x] **Review Status**: Reviewed by Qwen-3.8-Flash-Next; the tickets below are the post-review versions.

On 2026-10-19, empirical benchmarks were conducted on an Apple M2 (24 GB) comparing Strata's custom Metal Qwen3.6 engine against Unsloth's `Qwen3.6-35B-A3B-MTP-GGUF` (`UD-Q2_K_XL`). The post-review state of the evidence (every number sourced in the review doc):

- **The engine is identified**: the Unsloth Desktop build is `b11724-mix-c208e5f` — upstream llama.cpp `b11724` plus 20 pinned commits, none of which touch Metal compute. On the M2, the comparison ran against upstream ggml-metal kernels (the fork's compute pins are CUDA/HIP/Vulkan only; #264's +16.4–17.8% MoE prefill gain on Qwen3.6 is CUDA-only — on Metal a good batched MoE GEMM is a differentiator, not catch-up).
- **Unsloth highlights (measured, screenshot-derived)**: ~263.6 tok/s prefill on bulk prompts (5,862 tokens in 22.24 s) and ~260 tok/s at a 32k-window session — prefill is depth-independent and batch-size-dependent (136 tok/s on ~447 fresh tokens). Decode: 31.2 tok/s shallow, 23.8 tok/s sustained over 2,328 tokens, 21.2 tok/s sustained 6.3k→11.3k depth on the follow-up. MTP on/off in the benchmark session is unverified (the card's recipe is `-np 1 --spec-type draft-mtp`; the API page showed 4 slots) — one model-load log line settles it.
- **Context allocation**: Auto budgeted 8,192 tokens under desktop load; a manual 32.8k load succeeded on the same machine (one yellow warning, no failure) — 8.2k was a conservative auto-budget, not an engine ceiling. Strata's differentiator is the explicit working-set gate, not a bigger ceiling.
- **Strata's recorded side**: the prefill baseline is stale (~25 tok/s is the bringup token-by-token number; the pre-lever matrix reads 22.3 tok/s at 2K, §6) — T12 re-baselines it before any target is set. Split-K + `--kv-f16` sustain 16k+ today; the GQA fold's A/B (T6) is the pending deep-decode lever.

### Execution order (measurement-first; the review doc's §6)

**T12 → T13 → T6's pending A/B → T8 → T9 → T10 → T11.** The kernel tickets' targets are set against T12's numbers, not against prose. Every measurement block in the review doc's §7 carries a **Decides** line — what each outcome changes. The kernel tickets' risk register and fallback rungs are the review's §13; the scoreboard that tracks every comparative number (measured, recorded, projected, pending) is its §11.

### T12 — The honest re-baseline (new, first)
- **Scope**: run the §6 matrix with every lever on (`tools/qwen35/q36_prefill_bench.py`, sizes 2048/4098, chunks 0/512, `--prefill auto --kv-f16`, `STRATA_Q36M_GDN_WY=1 STRATA_Q36M_DENSE_MMA=1 STRATA_Q36M_ATTN_GQA=1`), fresh model load per cell; record prefill and decode with hardware, build, date, and machine state. The full hours-class ablation matrix — levers added one at a time — is the review's §7 block 6, the stretch that ends the which-lever-bought-what question.
- **Gate**: the numbers land in docs/QWEN36_METAL.md; the targets below are restated against them. No default flips in this ticket.

### T13 — Dispatch-count instrumentation (new, second)
- **Scope**: a one-off counter for Metal dispatches per chunk, per path (per-row attention, MoE GEMV, chunked) — counted at the single dispatch choke point (the command-encoder wrapper), one stderr line per chunk, no behavior change. The static ~20,480 estimate it replaces sits labeled in the review's §11 scoreboard.
- **Gate**: the measured per-chunk dispatch counts recorded per path; replaces the unverified "20,000+ dispatches" estimate in the comparison narrative; T9/T10 cite before/after counts as their evidence.

### T8 — Prefix cache: measure the shipped feature, keep the common prefix on divergence (revised)
- **Observation (corrected)**: token-prefix matching and state reuse shipped in T0 (`q36_metal.cpp` L2419–2429; docs/QWEN36_METAL.md §2.2). The comparison article's "Cache Hits: 0 (Fresh prefill)" row was a fresh request, not a missing feature.
- **Scope**: (a) measure the existing reuse — one two-request same-prefix session through the serve protocol (expected: delta-only evaluation); (b) keep the common prefix on divergence — today a divergence resets the state fully; (c) optional, its own decision: persistence across restarts.
- **Gate**: the measured reuse numbers recorded; the divergence case keeps the common prefix and re-evaluates only the divergent tail; the 10-prompt parity unchanged.

### T9 — Batched FlashAttention for QSA chunk prefill (targets revised)
- **Observation (kept)**: `push_qsa_attention` dispatches the decode path's per-row attention sequence `nt` times per chunk per layer (`q36_metal.cpp` L1274–1278) — the dispatch storm T13 counts.
- **Scope**: a fused causal 2D attention kernel for the chunk path behind an opt-in knob (`STRATA_Q36M_ATTN_FLASH=1`, the env pattern read once per process so the MTP verify window never mixes paths), with a measured rel-bound envelope.
- **Parity contract (the honest version)**: fused online-softmax attention cannot preserve the T2 bit-identity contract — the accumulation order differs by design. The gate set follows the §10 MMA precedent: rel-bound envelope, 10-prompt ladder token-identity (greedy, 128 tokens), byte-identical no-flag default, and the **measured ≥5x-on-every-size bar** (§6) before any default flip.
- **Target**: set by T12's re-baseline and the ≥5x bar — not prose. The mechanism direction is confirmed from both sides: batch structure is where prefill lives (our T3 numbers; their pinned #264 on CUDA). The risk register and the fallback rung (multi-row blocks through the row-blocked SK pair — dispatches ÷ B at the same envelope class) are the review's §13.

### T10 — Batched MoE GEMM for routed experts (donor-reuse)
- **Observation (kept)**: routed experts run gathered GEMV (`push_c_gemv` / `push_c_lpr8`).
- **Scope**: group the chunk's tokens by routed expert ID and run a batched GEMM per active expert, reusing the `q36_dense.metal` donor (`push_c_dense`'s dequant-once, matrix-unit path). Same knob + rel-bound envelope + ladder gates as T9; strictly opt-in until the measured bar clears.
- **Context**: the fork's pinned #264 (+16.4–17.8% pp on Qwen3.6, decode tie) is CUDA-only — Metal takes the existing path, so this is a differentiator opportunity, not catch-up. The imbalance case (tail experts at 1–2 rows, ~64 rows/expert at a 2k chunk but ~16 at 512) keeps the existing GEMV below a measured row threshold — the threshold comes from T13's rows-per-expert histogram (review §13).

### T11 — Predictable context budgets (reframed)
- **Observation (corrected)**: there is no measured 8.2k wall — the 32k load succeeded on 24 GB on both engines. The original "paged & dynamic context" premise is refuted; vLLM-style paging is speculative infrastructure at these sizes.
- **Scope**: keep and document the explicit working-set gate as the differentiator (refusal with override, already built); decide the `--kv-f16` default on the measured record (§9's numbers are in; the flip decision rides on T6's A/B matrix); run the controlled 32k comparison (cool machine, Chrome closed, decode sampled at matched ~2k/6k/11k positions, both engines). Paging only if a measured need appears (e.g. fragmentation under long multi-session use).
- **Gate**: zero truncation at the gate-approved budget on 24 GB; sustained decode > 6 tok/s at 16k depth measured (T6's A/B feeds this); the comparison table recorded in docs/QWEN36_METAL.md.

### Decode-at-depth workstream (owned by T6's pending A/B; tracked here for ordering)

The comparison's heaviest number is 21.2 tok/s sustained at 6.3k→11.3k depth. The bandwidth arithmetic (review doc §4.1): with the fold + `--kv-f16` the 11.3k-depth ceiling rises to ~76–89 tok/s; at the ~25–35% efficiency class their engine demonstrates at both depths, that is ~19–31 tok/s — parity, *if* the efficiency class carries over to the folded path (the labeled assumption the A/B tests; Strata's recorded deep numbers today run ~14–22% of ceiling at ≤4k depth). Ticket: T6 (Phase 1) — its A/B is Phase 2's third gate; T5's deferred deep-decode matrix folds into the same run. The optional tokens-per-joule experiment (energy at depth, `powermetrics`, cool machine) rides on the same A/B — review doc §4.3.

