# Implementation Tasks: Split-K Attention Uncap Context Past 8K (32K-64K)

- [ ] Create task branch off `main` in Strata; cherry-pick `65037a91` (split-K kernels `k_attn_sk_part` / `k_attn_sk_merge` in `metal/kernels/q36.metal`, engine dispatch + memory gate + scratch buffers in `metal/src/engine/q36_metal.cpp`, split-K pipeline checks in `metal/tests/q36_compile_test.cpp`)
- [ ] Apply the MSL fix from `5f84e2e5`: `constexpr int SK_HD/SK_CELLS` → `enum { ... }` in `q36.metal` (required for gate 1 on M2)
- [ ] Confirm the cherry-picked `q36_compile_test.cpp` keeps the two split-K pipeline checks and does NOT pull in `5c367647`'s chunk-pipeline additions
- [ ] Add the split-K bullet of docs §6 (knobs `STRATA_Q36M_ATTN_SK`, `STRATA_Q36M_MEM_OK`, engagement rule, gate commands) to `docs/QWEN36_METAL.md`, in the repo's plain-words/measured-numbers style
- [ ] Verify the branch contains ONLY split-K scope: no `d4ee3dbe`/`5c367647`/MTP/trace commits, and `git diff` vs `65037a91`+enum-fix is empty apart from the docs bullet
- [ ] Cite or re-run the M2 acceptance gates and record results with hardware noted:
  - [ ] Gate 1: `./build-metal/q36_compile_test` creates all pipelines incl. `k_attn_sk_part` / `k_attn_sk_merge` (recorded passed at `44b4f8e0`)
  - [ ] Gate 2: `STRATA_Q36M_ATTN_SK=1` at `--ctx 4096` is 10/10 token-identical vs base via `q36_solo_gate.py` + `q36_cmp_gate.py` (recorded passed at `c7a66ff3`)
  - [ ] Gate 3: `--ctx 32768` reaches `READY 32768` and generates without crashes (recorded passed at `c7a66ff3`)
  - [ ] Gate 4 (default-path invariance): confirm by code inspection that ≤ 8192 dispatches the original `k_attn` and startup never refuses `ctx <= 8192`
- [ ] Push the task branch and open a PR to `main` (one change per PR; note in the PR description what was not touched — chunked prefill, MTP, trace tooling)
