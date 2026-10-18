# Implementation Tasks: Chunked Prompt Prefill — Parity Green (38/38 Stages)

## Step 1 — Probe run (Mac, evidence only)

- [ ] Ask the user to pull tip `9d4c0aee` on `feat/qwen36-metal`, `cmake --build build-metal`, and
      run `./build-metal/q36_chunk_parity`; paste back the full moe2048 block (diag + probeA/B/C
      lines) and the FAILURES line
- [ ] Interpret the prints against the decision tree in design.md §B and name the mechanism

## Step 2 — Fix the failing stage (test file only)

- [ ] Apply the mapped fix in `metal/tests/q36_chunk_parity.cpp`: coherent seeding helper for the
      memcpy-seeded GPU-read buffers (stage 16 `x2/x3/x4`, stage 11 `x1/x2` + `ey_ent`), or the
      overlap fix, per the probe evidence
- [ ] Promote the isolation combine to a checked stage `moe2048 iso` (38th stage; check it equals
      `x1`'s combine result); keep probeA/B/C prints
- [ ] Host pass: `g++ -fsyntax-only` where headers allow; confirm `git diff --stat` touches only
      `metal/tests/q36_chunk_parity.cpp` (+ `q36_compile_test.cpp` only if pipelines changed)
- [ ] Push and request Mac round: `q36_chunk_parity` must report 38/38 OK, 0 failures; verify
      stage 11 still passes with real bases

## Step 3 — Engine audit

- [ ] List every host `memcpy` into a GPU-read shared buffer in the engine
      (`metal/src/engine/q36_metal.cpp`, `window.cpp`) with sizes; confirm all are in the
      proven-reaching small class (`toks_c` ≤ 2 KB per 512-token chunk, mrope rows) or fix GPU-side
- [ ] Record the audit result in the ticket so T3 (default flip) inherits it

## Step 4 — Engine ladder (Mac)

- [ ] Mac round: `q36_solo_gate.py` plain → `/tmp/q36-base.json`; `--prefill 512` →
      `/tmp/q36-chunk.json`; `q36_cmp_gate.py` must print `identical: 10 of 10`
- [ ] If red: same round, rerun both trace gates with `--log` + `q36_trace_cmp.py`; name the first
      diverging layer/cut and apply the mapped fix (design.md §E order: G0 embtype, G8 re-check,
      one-graph hazard); one fix per push, repeat until 10/10

## Step 5 — Land

- [ ] Update `docs/QWEN36_METAL_ROADMAP.md` T2 State: green (stage count, ladder 10/10, Mac/date)
- [ ] Confirm no CUDA/HIP/SYCL file changed; confirm `--prefill` stays opt-in (default flip is T3)
- [ ] Request the final Mac verification of both gates and mark the ticket done
