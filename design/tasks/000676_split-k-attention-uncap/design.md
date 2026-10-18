# Design: Split-K Attention Uncap Context Past 8K (32K-64K)

## Root cause

`k_attn` writes one score per (query head, context cell) into threadgroup memory
(`d.tg_mem = (s.pos + 1) * sizeof(float)`). Metal's default threadgroup budget is 32 KB, so
`s.pos + 1` tops out at 8,192 floats. Nothing else in the engine depends on that limit — the KV
cache buffers are already sized from `--ctx` — so attention is the single bottleneck.

## Approach: split-K partials (already implemented in 65037a91)

Port of the CUDA engine's design (`src/qwen35/q36_engine.cu`, `k_attn_part` / `k_attn_merge`), same
math and the same order, so long-context attention agrees with the CUDA path:

1. `k_attn_sk_part` — one threadgroup per (query row, 256-cell chunk). Each threadgroup computes the
   chunk's scores, running max `m` (via `simd_max` reductions in `sk_max`), sum `l` (`sk_sum`), and
   the chunk's partial accumulator `P·V`, and writes the triple `{acc[256], m, l}` to device scratch
   (`sk_pacc` / `sk_pm` / `sk_pl`). Threadgroup memory use is constant (~1 KB: `qs[256]`, `sc[256]`,
   `sh[8]`) regardless of context. Query rows are `t * H + h`; the decode step passes one token
   (`t = 0`, `H` rows).
2. `k_attn_sk_merge` — one threadgroup per query row: folds the chunks with the online-softmax
   rescale (`M = max m`, `num += w * acc`, `den += w * l`, output `num / den`) and applies the
   sigmoid gate, writing to `s.att` exactly where `k_attn` left it. Downstream dispatches are
   untouched.

### Key decisions

- **Constants as MSL `enum`, not `constexpr`.** Metal Shading Language rejected program-scope
  `constexpr int SK_HD/SK_CELLS`; commit `5f84e2e5` switched to `enum { SK_HD = 256 }`. Any new
  program-scope constant in `q36.metal` must follow this.
- **Engagement rule** (in `forward_token`): split-K runs when `sk_force || n_pos > SK_TG_CAP(=8192)`;
  otherwise the original `k_attn` dispatch, byte-identical. `sk_force` comes from
  `STRATA_Q36M_ATTN_SK` (nonzero = force) and is the parity knob.
- **Scratch bounded, allocated once at startup**: `SK_ROWS_MAX = 256` rows × `ceil(ctx/256)` chunks
  × (256 + 2) floats. Decode uses `H` rows (24 on qwen35moe); the bound exists so a later prompt
  row-block path can reuse the same buffers. Three `mtl::Buffer`s on `State`.
- **head_dim guard**: kernels are specialized for `SK_HD = 256`; the host checks
  `head_dim == 256` before dispatching split-K and exits with a clear message otherwise.
- **Memory gate at startup** (only for `ctx > 8192`, never for smaller ctx):
  `est = weights + kv + ssm_state + conv_state + scratch + 0.25 GB reserve`, compared against
  `cx.recommended_working_set()`. On refusal: exit 1, printing every component in GB plus the
  override name (`STRATA_Q36M_MEM_OK=1`). Same shape as the CUDA engine's refusal.
- **Compile test**: `metal/tests/q36_compile_test.cpp` creates both new pipelines from the library.
  Keep the minimal 4-line version from `65037a91` (the chunk-pipeline coverage in `5c367647` belongs
  to the chunked-prefill task). Confirmed by reviewer 2026-10-18.

## Landing plan

- `65037a91`'s parent (`f9168964`) is on `main` → cherry-pick `65037a91` onto a task branch off
  `main`, then apply the split-K hunks of `5f84e2e5` (the `constexpr` → `enum` change; its merge
  kernel hunk depends on chunk-era code and is N/A here — only the unused-var removal matters if the
  context differs).
- Docs: add the split-K bullet of §6 (knobs `STRATA_Q36M_ATTN_SK`, `STRATA_Q36M_MEM_OK`, gate
  commands) to `docs/QWEN36_METAL.md`, keeping the docs' plain-words, measured-numbers style, and
  include the M2 gate numbers with hardware attribution ("measured on M2"). Also add a
  `docs/TEST_REQUESTS.md` entry requesting same-build comparisons on other Apple Silicon (M1/M3),
  following `docs/COMMUNITY_BENCHMARKS.md` (compare against a same-day run of the build under test).
- Do NOT cherry-pick `d4ee3dbe` / `5c367647` / later chunk and MTP commits.
- Push branch, open PR to `main` (one change per PR). The task is complete only when the PR merges;
  `feat/qwen36-metal` is abandoned per reviewer decision — rebasing it later is a separate follow-up
  task, not part of this one.

## Verification notes / gotchas

- This sandbox is Linux without Metal: the engine cannot be compiled or run here. Gate 1-3 runs are
  M2 (Apple Silicon) work. Branch commits `44b4f8e0` (compile gate passed) and `c7a66ff3` (split-K
  10/10 exact at 4K; 32K boots and generates) already record the M2 results for exactly this code;
  the implementation phase should re-run the gates only if fresh Mac hardware is available, otherwise
  cite the recorded runs.
- Gate commands (drive the engine directly via the serve protocol):
  ```bash
  # base reference
  python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
    --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-base.json
  # forced split-K parity: tokens must equal base
  STRATA_Q36M_ATTN_SK=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
    --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-sk.json
  # long context
  python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
    --args "--native $Q36_MODEL --ctx 32768" --out /tmp/q36-32k.json
  ```
- `65037a91` ships with `constexpr int SK_HD/SK_CELLS` which fails Mac compile — the enum fix MUST be
  applied in the same PR, otherwise gate 1 fails on M2.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims
  without a measurement (per `Strata/AGENTS.md`).

## Implementation Notes (2026-10-18)

- Branch executed as: `feature/000676-split-k-attention-uncap` = `main` (`e4fe2def`) + cherry-pick
  `65037a91` (`e7a2de1d`, clean, hook kept the original message) + MSL enum fix (`9fbd3769`) + docs
  (`d496736e`). The reviewer's branch name `feat/qwen36-splitk` was already claimed by another task in
  this project (the proposal tool refused it), so the identical commits ride the task branch instead.
- The MSL fix from `5f84e2e5` has TWO hunks, both needed: (1) `constexpr int SK_HD/SK_CELLS` →
  `enum { ... }`; (2) drop the unused `h` in `k_attn_sk_merge` (`const int t = (int) row / H, h = ...` →
  `const int t = (int) row / H;`). The part kernel's `t, h, kvh` line (L562) is NOT touched — `h` and
  `kvh` are used there.
- `5f84e2e5`'s version of the kernels is the chunk-era one (`row0` buffer arg, row-block merge). When
  diffing our decode-only branch against that commit you WILL see those chunk differences — they are
  expected and belong to the chunked-prefill task. The correct scope check is `git diff main HEAD`:
  exactly 5 files (+39 docs §6, +1 test-request row, +124 kernels, +113/-16 engine, +4 compile test).
- Host-side check that works on the Linux sandbox: `g++ -fsyntax-only -std=c++17 -Iinclude -Imetal/include
  -Imetal/src metal/src/engine/q36_metal.cpp` passes clean with `-Iinclude` (the engine includes
  `strata/qwen35/engine.hpp` from the main include dir). Metal kernels cannot be compiled here — gate 1
  on the Mac covers them.
- Gate citations: the M2 runs are recorded in the (abandoned) branch's spec commits — compile gate at
  `44b4f8e0` ("all 30 pipelines, macOS 26.6.2"), parity 10/10 + READY 32768 at `c7a66ff3`. The recorded
  detail lives in `git show c7a66ff3:design/tasks/000675_here-is-this-hand-off/tasks.md`.
- `docs/TEST_REQUESTS.md` is a 3-column table (| Wanted | Why | Example of an earlier answer |); add a
  row, not a section. `docs/QWEN36_METAL.md` had sections 1-5 on `main`; §6 is appended after §5's last
  bullet (`./run-q36.sh 8082`).
- Commit hook: conventional-commit format enforced in both repos; subjects over 72 chars only warn.
