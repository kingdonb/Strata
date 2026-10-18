# Implementation Tasks: Chunked Prompt Prefill — Parity Green (38/38 Stages)

## Step 1 — Probe run (Mac, evidence only) — DONE, root cause found

- [x] Mac round 11 (user, tip `9d4c0aee`): probeA/probeB saw the seeded bases intact; `moe2048 x`
      still failed; `initial[0..2] = 0 0 0` (x3's HOST view zeroed by diag time); probeC's x4 == x2's
      values. Full console block recorded in the ticket chat.
- [x] Interpretation (supersedes the §B tree's "memcpy never reaches" row): both combine kernels are
      correct (`x[...] += ...`); the bases were zeroed mid-chain by a THIRD party. Root cause: every
      `k_gather_t` dispatch passed the ELEMENT count as the threadgroup count (grid = E, tg = 128 →
      128x the threads), and the kernel had no bound check — the extra threads read past the source
      (fresh arena pages read as zeros) and wrote ~64 MB (harness dga) / ~4 GB (engine, per chunk)
      past the destination, smashing the live buffers above it: stage 16's x2/x3 (allocated after
      hgc; x1 survived below it), and in the engine everything above `s.hg_c` on every chunk — the
      ladder's 0/10.

## Step 2 — Fix (pushed: `eac5e9a7` + merge `f6a7d081` + `87877024` on the task branch)

- [x] `metal/kernels/q36.metal`: `k_gather_t` takes the element count (`buffer(4)`) and bounds its
      threads (`if (i >= count) return`)
- [x] All five dispatch sites pass ceil-div threadgroup grids + the count: the engine's chunk gather
      (`q36_metal.cpp`), the harness's dga, probeA's c1/c2/c3, probeB's cm
- [x] The isolation combine is the suite's 38th checked stage: `moe2048 iso` = the pristine base
      (memcpy'd from x3) through the chunk combine must equal x1's result bit for bit
- [x] `g++ -fsyntax-only` passes for the harness and the engine; no CUDA/HIP/SYCL file touched
- [x] Mac round 12 (user, 2026-10-18): the build FAILED on my merge's duplicate `p_attn_sk_part/merge`
      declarations in `q36_compile_test.cpp`; ninja stopped and the parity ran the **stale `9d4c0aee`
      binary**: its 4-arg gather dispatches met the new 5-arg MSL (compiled at run time from the source
      tree), the unbound `count` slot read 0, and every gather returned on its first check — nothing
      written (probeA/B zeros, `hgc` → `eh2`/`ey2` zeros, `cpuC = 0`, `x2 = base + 0`, x3 survived,
      `moe2048 iso` absent). NOT a fix failure — the round-11 smash's footprint is gone because
      nothing was written at all; the decode side stayed correct.
- [x] Dedupe the compile test's sk declarations + add the under-binding guard: every pipeline
      reflects its max `[[buffer(N)]]` index + 1 at creation; a dispatch binding fewer args exits
      loudly with a rebuild hint instead of silently reading unbound slots (`0c0608da`); the
      reflection API corrected for the real SDK — `MTLFunction.functionArguments` does not exist,
      the list comes from `MTLComputePipelineReflection.arguments` via the pipeline-creation call's
      out-param (`76ba259d`); site sweep (engine q36 + q35 + harness): zero under-bound dispatches,
      the guard cannot false-fire
- [x] **Mac round 13 (user, 2026-10-18): GATE 1 PASSED — `./build-metal/q36_chunk_parity` reports
      all 38/38 stages bit-identical, 0 failures.** Tells all textbook: probeA/B see the seeded
      values (the gathers work); the diag's `d_dec == d_chk` to the last digit; x3 survives (the
      smash is gone); probeC's `x4 == x1` (the round-10c "GPU saw the base as zeros" question
      resolved at the root); `moe2048 iso` (the 38th stage) OK. The arg-count guard is live:
      `refl.arguments` populated `arg_min` (deprecation warnings only — `MTLArgument`/`.arguments`/
      `.type` are macOS-13-deprecated with `MTLBinding`/`bindings`/`MTLBindingType` as the
      replacements; the house tolerates warnings).

## Step 3 — Engine audit — done alongside the fix

- [x] The audit's biggest find was the engine's own gather dispatch (fixed above). Remaining
      host→GPU memcpys on live paths: `s.toks_c` token ids (≤ 2 KB per 512-token chunk), mrope rows
      (≤ 6 KB), image/last-norm rows (small) — all in the proven-reaching small class; the load-time
      weight memcpys complete before any dispatch is encoded. No further engine change.
- [x] Kernel-bound sweep (round-11 lesson, done 2026-10-18): every grid-indexed kernel in
      `metal/kernels/*.metal` bounds its threads (`if (i < n)` / `>=` return) or runs an exact-fit
      grid — `k_gather_t` was the only unbounded one. No other smasher; the remaining dispatch sites
      are safe (bounded kernels make grid overshoot harmless, and undershoots would fail the
      value checks).

## Step 4 — Engine ladder (Mac)

- [~] **Mac round 13 (user, 2026-10-18): the ladder is 5/10** (`cmp_gate`: p02/p04/p07/p08/p09
      identical; p00 differs at 100, p01 at 3, p03 at 21, p05 at 121, p06 at 1 — len 81 vs 86). Up
      from 0/10 — the gather-smash fix moved it — and the residual pattern is ULP-class drift
      snowballing, not garbage. Incidental: the chunk path's TTFT is ~30% faster on every prompt
      (p00 19.6 s → 8.9 s incl. the first-request JIT warmup) — T3's speed story, noted.
- [x] **Round 14 (Mac, same build, ladder rerun): the divergence is a RACE** — the failing set and
      depths changed between runs (p00 at 100 then 26; p01/p05 passed then; p02/p04 fail now;
      p03 at 21 then 46; p06 at 1 then 8). Static read found the mechanism: outside trace mode
      `forward_prompt_chunk` accumulated ALL layers' dispatches (~600, barrier'd) into ONE command
      buffer committed at return — the whole-chunk one-graph (round 8b's structural hazard). The
      harness's small chains (≤ 12 dispatches) are the stable class; ~600 barriers on one concurrent
      encoder race on the M2.
- [x] Fix (`f6f57bd7`): commit the chunk graph **per layer** and **per split-K part/merge pair**
      (the partials scratch is reused across pairs); GPU-time return sums the per-layer graphs;
      trace mode unchanged (already per-cut). Every intra-layer dependency remains the barrier'd
      pattern the harness verifies; cross-command-buffer work is queue-ordered.
- [~] **Round 15 (Mac, post per-layer flush): the ladder improved to 8/10 (p00@100, p01@69) but the
      race SURVIVED** — chunk-vs-chunk2 of the same binary differs (p04 stops at 18 vs 19, p08 at
      18). The racing unit = the dispatch TYPE: `run_graph` encoded `MTLDispatchTypeConcurrent`
      (dispatches may overlap) while `memoryBarrierWithScope` does not order overlapping dispatches
      on macOS — the `k_tables` race was the same class at 4 dispatches, and the harness's small
      chains only looked stable.
- [x] Fix (`62eba361`): `run_graph` encodes `MTLDispatchTypeSerial` — hardware in-order execution,
      every barrier'd dependency holds by construction. The per-layer/per-pair flushes stay (small
      graphs, less encode overhead); the bench paths keep their explicit concurrent choice.
- [ ] Round 16 (user): rebuild, then (0) `cmp_gate /tmp/q36-base2.json /tmp/q36-base.json` (the
      decode side's stability across the round-13b/15 builds — if it differs, forward_token gets
      the same treatment); (1) the ladder twice + chunk-vs-chunk2 as in round 15 — want 10/10 ×3;
      (2) rerun `q36_chunk_parity` (the 38/38 gate re-check on the new runtime)

## Step 5 — Land

- [ ] Update `docs/QWEN36_METAL_ROADMAP.md` T2 State: green (stage count, ladder 10/10, Mac/date)
- [ ] Confirm no CUDA/HIP/SYCL file changed; confirm `--prefill` stays opt-in (default flip is T3)
- [ ] Propose the PR (task branch → main), then request the final Mac verification and mark done
