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
  to the chunked-prefill task).

## Landing plan

- `65037a91`'s parent (`f9168964`) is on `main` → cherry-pick `65037a91` onto a task branch off
  `main`, then apply the split-K hunks of `5f84e2e5` (the `constexpr` → `enum` change; its merge
  kernel hunk depends on chunk-era code and is N/A here — only the unused-var removal matters if the
  context differs).
- Docs: add the split-K bullet of §6 (knobs `STRATA_Q36M_ATTN_SK`, `STRATA_Q36M_MEM_OK`, gate
  commands) to `docs/QWEN36_METAL.md`, keeping the docs' plain-words, measured-numbers style.
- Do NOT cherry-pick `d4ee3dbe` / `5c367647` / later chunk and MTP commits.
- Push branch, open PR to `main` (one change per PR).

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
