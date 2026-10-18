# Requirements: Split-K Attention Uncap Context Past 8K (32K-64K)

## Background

The Qwen3.6 Metal engine (`metal/src/engine/q36_metal.cpp`, kernels in `metal/kernels/q36.metal`)
keeps attention scores in threadgroup memory. Metal's default threadgroup budget is 32 KB, which at
4 bytes per score caps the served context at 8,192 cells — `--ctx` beyond 8K is impossible today.

The fix already exists and was verified on an M2 Mac: commit `65037a91` (split-K attention, ported
from the CUDA engine's `k_attn_part` / `k_attn_merge` in `src/qwen35/q36_engine.cu`), the MSL
program-scope fix `5f84e2e5` (`constexpr` → `enum`), and docs §6 in `docs/QWEN36_METAL.md`. These
commits sit on `origin/feat/qwen36-metal`; `65037a91`'s parent is on `main`, so it cherry-picks
cleanly. The M2 gates were recorded as passed (branch commits `44b4f8e0`: compile gate; `c7a66ff3`:
split-K 10/10 exact, 32K runs).

This task lands the split-K scope on `main`. The chunked-prefill work that shares the branch
(`d4ee3dbe` and later) is explicitly OUT of scope.

## User Stories

1. As a Mac user, I want to run the Qwen3.6 Metal engine with `--ctx 32768` so that I can work with
   long documents instead of being capped at 8K.
2. As a user, I want the default decode path at ≤ 8,192 cells to stay byte-identical to the last
   release so that my existing setup behaves exactly as before.
3. As a maintainer, I want startup to refuse a context size that cannot fit the machine's unified
   memory, with the actual numbers in the message, so failures are diagnosable instead of mysterious.
4. As a maintainer, I want a parity knob (`STRATA_Q36M_ATTN_SK=1`) that forces split-K at short
   context so I can prove the new path matches the default path token-for-token.

## Acceptance Criteria

1. `./build-metal/q36_compile_test` creates all pipelines, including the new `k_attn_sk_part` and
   `k_attn_sk_merge`. (Verified on M2, branch commit `44b4f8e0`.)
2. Parity: with `STRATA_Q36M_ATTN_SK=1` at `--ctx 4096`, generation is 10/10 token-identical to the
   default `k_attn` path. (Verified on M2, branch commit `c7a66ff3`.)
3. Long context: `--ctx 32768` boots (`READY 32768`) and generates without crashes. (Verified on M2,
   branch commit `c7a66ff3`.)
4. Default path unchanged: at `n_pos <= 8192` the plain `k_attn` dispatch is used, unchanged, and no
   memory refusal is ever triggered (≤ 8,192 is never refused).
5. Memory gate: for `--ctx` above 8,192 whose estimated working set (weights + KV + SSM state + conv
   state + split-K scratch + 0.25 GB reserve) exceeds `recommended_working_set()`, startup exits 1
   with a per-component breakdown in GB and names the override `STRATA_Q36M_MEM_OK=1`; setting that
   env var lets the run proceed.
6. Split-K engages automatically on the decode step as soon as `n_pos > 8192` — no flag needed.
7. The host refuses split-K (clear error, exit 1) if `head_dim != 256`, since the kernels are written
   for `SK_HD = 256` (the qwen35moe head_dim).

## Out of Scope

- Chunked prompt prefill (`k_emb_chunk`, `k_gemv_c`, `k_expc`, `k_conv_chunk`, `k_gdn_chunk`, ...):
  separate task; its commits stay off the landed branch.
- MTP/draft work and trace tooling on `feat/qwen36-metal`.
- 64K context is a stretch goal of the title, not a gate: only 32K is gated. KV at 64K is limited by
  unified memory and must pass gate 5's arithmetic on the target machine.

## Open Questions

1. Delivery shape: is landing via cherry-pick onto a branch + PR to `main` the intended flow, or
   should this task only re-verify the existing `feat/qwen36-metal` state? (Spec assumes cherry-pick
   + PR, per the fork's one-change-per-PR convention.)
2. Compile-test change: `65037a91` already adds the two split-K pipeline checks (4 lines). The later
   `5c367647` reorganizes the test and also adds chunk-prefill pipeline checks. Spec assumes we keep
   only the 4-line version and leave `5c367647` to the chunked-prefill task. Correct?
3. Where should the recorded M2 gate results live — a short "measured on" note in
   `docs/QWEN36_METAL.md` §6 (per the docs' style of numbers with what they were measured on), or
   `docs/TEST_REQUESTS.md`?
