# Design: GQA-Folded Attention — One K/V Read per KV-Head Group

## Where attention runs today

One shared helper serves both the decode step and the chunk path — `push_qsa_attention`
(`metal/src/engine/q36_metal.cpp:356`, the `02d7aeec` pattern: shared, not mirrored):

- n_pos <= 8,192 (`SK_TG_CAP`): one `k_attn` dispatch per query row — grid (H), 128 threads,
  scores in dynamic threadgroup memory (`d.tg_mem = n_pos * 4`), one-pass structure:
  per-thread scores over cells p = tid (mod 128) -> global max (per-thread max, simd_max,
  4-simdgroup tree) -> exp + per-thread partial sums reduced by `block_sum` (128 threads,
  4 simdgroups) -> P·V: each thread owns output dims i = tid (mod 128), walking p ascending -> 
  `out = (acc * inv_sum) * sigm(gate)`.
- n_pos > 8,192, or `STRATA_Q36M_ATTN_SK=1`: `k_attn_sk_part` (grid (n_chunk, H), 256 threads,
  one partial per (query row, 256-cell chunk) into `sk_pacc`/`sk_pm`/`sk_pl`) + `k_attn_sk_merge`
  (grid (H), folds the chunks online-softmax style, `out = num / den * sigm(gate)`).

Geometry (runtime, from the GGUF via `read_geometry`; cross-checked against the upstream config):
H = 16, HKV = 2, so HG = H/HKV = 8 query heads per KV head; D = 256; 10 QSA layers;
`kvh = h / HG`. KV traffic per attention layer per token is 2 (K+V) x H x n_pos x D floats — every
query head of a group re-reads the same cells. Folded, it is 2 x HKV x n_pos x D plus a score-scratch
round trip (HG x n_pos, ~256x smaller than the KV term): ~1/7.7 of today at depth (8x on the KV term).

## New kernels (`metal/kernels/q36.metal`)

### Path A — the `k_attn` range (n_pos <= 8,192)

A folded kernel cannot keep k_attn's "all scores in threadgroup memory" structure: HG x n_pos
scores (8 x 8,192 floats = 256 KB) is past the 32 KB threadgroup ceiling. The fold splits k_attn's
one pass into two dispatches over one score scratch of HG x ctx floats + HG floats of per-head max
(allocated knob-gated at startup next to the split-K scratch, so `cx.allocated()` counts it in the
working-set refusal; reused per row — dispatch barriers separate rows):

1. `k_attn_gqa_part` — grid (HKV), 128 threads, per KV head kvh:
   - Cooperatively stage the group's HG q rows in dynamic threadgroup memory (HG x D floats).
   - Thread tid handles cells p = tid (mod 128): read the K row once, compute the HG dot products
     (i ascending 0..D-1, one fp chain per (head, cell)), scores to scratch.
   - Per-head max with k_attn's tree (any tree gives the same value — max is exact); mx[h] to scratch.
2. `k_attn_gqa_merge` — grid (HKV), 128 threads, per kvh:
   - Thread tid walks p = tid (mod 128): e = exp(score - mx) written back to scratch; per-thread
     partial sums in the same order, reduced by the same `block_sum` tree (fp addition order is
     load-bearing — see the contract).
   - P·V: for each output dim i = tid (mod 128) and each h, one accumulator chain walking p
     ascending; each loaded V element shared across the group's HG chains. Stage e in per-tile
     threadgroup buffers (HG x SK_CELLS floats) so the chains read threadgroup, not device.
   - `out[h][i] = (acc * (1/sum)) * sigm(gate[h][i])` — k_attn's exact final expression (a reciprocal
     multiply, NOT the split-K merge's `num/den` division).

Occupancy note: grid (HKV) = 2 threadgroups on this model. Variants that preserve the contract are
allowed where they help: split the output dims across more threadgroups (each (h, i) chain stays in
one thread; each dim's V cells still read once per kvh), and/or compute the max in the merge kernel
from the score scratch (order-free) to keep the part kernel position-parallel. Every variant must
pass the gate-2 ladder.

### Path B — the split-K range (n_pos > 8,192, or `STRATA_Q36M_ATTN_SK=1`)

`k_attn_sk_part_gqa` — grid (n_chunk, HKV), 256 threads. The threadgroup (chunk c, kvh) covers the
group's HG rows (row = row0 + g.y*HG + h_local; contiguous inside the row block, so the row
arithmetic and the `q`/`att` indexing are unchanged). Per cell j: read the K row once, HG dots in
the part kernel's exact chain (i ascending 0..SK_HD-1). Per head h: the same `sk_max`/`sk_sum` trees
over the chunk's 256 scores; the P·V partial per (h, dim j) walks the chunk's cells ascending, each
V element loaded once and shared across the HG chains. Partials go to the same
`sk_pacc`/`sk_pm`/`sk_pl` slots as today ((row - row0) * n_chunk + c), so **`k_attn_sk_merge` runs
unchanged** and sees bit-identical partials.

Threadgroup staging: qs (HG x SK_HD) + scores (HG x SK_CELLS) = HG x 512 x 4 B — 16 KB at HG=8.
Path A's staging (HG x D qs; HG x SK_CELLS e-tiles) fits the same bound.

## The bit-parity contract

Per query head h, the folded path must produce the same floats as the path it replaces:

| quantity  | contract |
|-----------|----------|
| score     | sum over i ascending 0..D-1 of q[h][i]*k[kvh][p][i], one fp chain (same FMA contraction), then *scale. q may come from threadgroup staging — same values. |
| mx        | max over p — exact, any reduction tree. |
| e         | exp(score - mx), same operands, computed once per (h, p). |
| sum       | per-thread partials over p = tid (mod 128) ascending, then the same reduction tree the path uses (path A: 128-thread `block_sum`; path B: per-chunk 256-thread `sk_sum`, then the unchanged merge's weighted fold). |
| P·V       | one fp chain per (h, i), p ascending 0..n_pos-1; the e and v operands may be staged/shared, the chain may not be split or tiled across threads (that re-rounds). |
| out       | path A: `(acc * (1/sum)) * sigm(gate)`; path B: whatever the unchanged merge does. |

FLOPs stay exact: no score recompute, no exp recompute — the only new traffic is the score-scratch
write/read. Q staged in threadgroup, e written back to the score scratch in the sum phase and read
from staged tiles in the P·V phase.

## Engagement, guards, plumbing

- `const bool gqa = env_flag("STRATA_Q36M_ATTN_GQA")` beside the SK knob
  (`q36_metal.cpp:1287`); threaded into `forward_token` (the `sk_force` pattern — 4 call sites) and
  into `forward_prompt_chunk` (which currently hardcodes `sk_force=false`; the fold MUST reach the
  chunk rows — the read acceleration is a measured gate). The selection lives inside
  `push_qsa_attention`: fold on -> path A pair replaces `k_attn`, `k_attn_sk_part_gqa` replaces
  `k_attn_sk_part` (merge common).
- Startup guards, knob-gated, printed like the `head_dim 256` refusals: HG < 2 (the fold is a
  no-op); HG x (D + SK_CELLS) x 4 B + slack vs the documented 32,768 B `maxThreadgroupMemoryLength`
  (no accessor in `mtl_rt.h` — hardcode with a comment). HG=8 needs 16 KB. Refuse, don't fall back.
- Pipeline creation unconditional (the SK pattern) so the compile test covers the new MSL.
- Scratch: HG x ctx floats of scores + HG floats of max, allocated beside the split-K scratch.

## Known gotchas (from this codebase's history)

- **Reduction-slot reuse race**: `sk_max`/`sk_sum` end with a barrier BEFORE the `sh[0]` read;
  back-to-back calls can race the next call's `sh[sgid]` write against the previous read (the
  `block_sum` round-20 race shape, fixed only in `block_sum`). The current part kernel calls each
  once; the folded part kernel calls them HG times back-to-back — give each head its own slots
  (`threadgroup float sh[HG][8]`, ~64 B) or interpose a barrier. Do NOT modify `sk_max`/`sk_sum`
  themselves: the unfolded kernels must stay byte-identical. (The same latent shape exists in the
  current part kernel's sk_max->sk_sum pair; out of scope here.)
- **Dispatch count**: path A doubles the per-row dispatches (2 vs 1). At 512-token chunks that is
  ~512 extra small dispatches per chunk — noise next to the 22-40 s chunk cost, but worth a look in
  the decode profile (10 extra dispatches per token).
- **Default path byte-identity**: knob off -> no new dispatches, no scratch, only dead
  pipeline-creation and guard branches. The ladder proves it (gate 5).
- MSL program-scope constants are `enum`, not `constexpr`.

## Out of scope

Default flips (the fold, `--prefill`); the CUDA engine's same redundancy (`src/qwen35/q36_engine.cu`,
a later ticket); FP16 KV (T5); MTP (T4); the QSA indexer (not implemented on Metal — attention is
dense over all cells there).

## Notes for the implementer

- Tooling: `tools/qwen35/q36_solo_gate.py` (greedy tokens of the 10 parity prompts -> json),
  `q36_cmp_gate.py` (compare two jsons, exit 1 on mismatch), `q36_chunk_parity` (38-stage
  bit-identity harness). The 4K ladder is the oracle; verify against the same-day base json, not a
  stored one.
- Depth benchmark: no dedicated tool exists — build a ~4,098-token ids file (repeated filler ids,
  the `q36_proto_edge.py` long_prompt pattern), run `q36_solo_gate.py --prompts <file> --args
  "--native $Q36_MODEL --ctx 16384 --prefill 512" --n 128` fold off and on, same build, same day;
  TTFT = the read, `len(toks)/(wall-ttft)` = the decode rate at pos ~4,100; `--log` captures the
  per-chunk `PP` lines for the doc.
- Cheap extra check: `STRATA_Q36M_ATTN_SK=1` with and without the fold should be 10/10 identical
  (the folded part reproduces the part kernel's partials exactly).
- Files this task touches: `metal/kernels/q36.metal`, `metal/src/engine/q36_metal.cpp`,
  `metal/tests/q36_compile_test.cpp`, `docs/QWEN36_METAL.md`,
  `docs/QWEN36_METAL_ROADMAP.md` (T6 State line). No shared headers -> the CUDA/HIP/SYCL builds
  are untouched; say so in the PR description (the contributing rule).
