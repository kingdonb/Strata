# Design: Chunked Prompt Prefill — Parity Green (38/38 Stages)

## Context & resume point

Work continues on `feat/qwen36-metal` (T2 of `docs/QWEN36_METAL_ROADMAP.md`). Round 10c proved both
MoE combines compute the correct sum; the failing `moe2048 x` stage fails because the GPU read the
memcpy-seeded base of `x2` as zeros (loop-written `x1` reaches; a 32-byte memcpy in stage 1
reaches). Tip `9d4c0aee` already adds three probes to stage 16 of
`metal/tests/q36_chunk_parity.cpp`:

- **probeA** — `k_gather_t` (k=1) copies `x1/x2/x3` into `x1g/x2g/x3g` in one graph immediately
  after the host writes: does the GPU ever see the memcpy?
- **probeB** — re-copies `x2` into `x2m` after the decode chain ran: was `x2` zeroed *between*?
- **probeC** — the same combine (`p_sumc`) on a fresh memcpy-seeded `x4`, as its own command buffer:
  `x4 == x1`'s values ⇒ the memcpy base reached *this* combine (so `x2` was zeroed in between);
  `x4 == x2`'s values ⇒ the memcpy base never reaches the GPU.

The decision rule below consumes those prints; no code changes before the first Mac probe run.

## Key decisions

**A. Probes first.** Round 11 on the Mac is pure evidence: build tip, run
`./build-metal/q36_chunk_parity`, paste the moe2048 block. Nothing is "fixed" until the prints pick
a mechanism.

**B. Decision tree for the moe2048 x fix (all inside the TEST file unless stated).**

| Probe evidence | Mechanism | Fix |
|---|---|---|
| probeA: `x2g == 0` (host shows values); probeC: `x4 == x2` values | the 64 KB memcpy never reaches the GPU | Replace memcpy seeding with the proven-coherent pattern: write the row values once into a `std::vector<float>`, then loop-write `x1`, `x2`, `x3` (and `x4`) from it — identical bytes, the loop-write class proven to reach. Mirrors `x1`'s existing seeding. |
| probeA: `x2g == host`; probeB: `x2m == 0` | something zeroes `x2`'s GPU view during the decode chain | Dump the decode chain's dispatch bindings for an overlapping buffer (allocation collision with untracked hazard tracking); fix the overlap, not the kernel. |
| All copies reach; `moe2048 x` still fails | mechanism not the seeding | Keep the printed sums (`cpuD`/`cpuC` vs GPU deltas), add the missing discriminating probe in the next push (one hypothesis per round), fall back to the round-9 cut methodology inside the stage. |

**C. Promote the isolation combine to a checked stage.** After the fix, probeC becomes stage
`moe2048 iso`: run `p_sumc` on a fresh buffer seeded the (fixed) coherent way and check it equals
`x1`'s combine result. Suite prints 38 stages; the probe prints stay as documentation. This is the
38th stage the ticket's gate expects.

**D. Seeding audit.** Enumerate every host `memcpy` into a GPU-read buffer in the harness (stage
16's `x2/x3/x4`, stage 11's `x1/x2` + `ey_ent`) and route them through the same coherent seeding
helper. Stage 11 passes today only because both sides zeroed equally; it must still pass with real
bases. **Engine audit:** the engine's per-request host→GPU memcpys are small (`s.toks_c` token ids
≤ 2 KB per 512-token chunk, mrope rows) — the proven-reaching class; weight memcpys (`q36_metal.cpp`
~L1258) are load-time, before any GPU read. If the audit finds a large memcpy into a GPU-read
buffer on a live engine path, fix it GPU-side in the same change; otherwise no engine file changes.
Rationale: the anomaly so far reproduces only on harness-style mid-run seeding; the engine's
small-copy class is proven reachable (stage 1 `toks_b`).

**E. Ladder procedure and fallbacks.** With the harness green, run the documented ladder
(`docs/QWEN36_METAL.md` gates): plain `q36_solo_gate.py --out /tmp/q36-base.json`, then
`--prefill 512 --out /tmp/q36-chunk.json`, then `q36_cmp_gate.py` → 10/10 required. If red, on the
same Mac round run both trace gates **with `--log`** (the harness gate sends engine stderr to
/dev/null without it — round 8a gotcha) plus `q36_trace_cmp.py`, which names the first diverging
layer/cut. Known candidates, in order:
1. **G0 embedding** — round 9's unresolved paradox (G0 diverges while G1–G7 match; r0/rl row sums
   and the embtype printout settle rows-vs-comparison-artifact). If rows genuinely differ, the
   real-model `token_embd` type is the suspect: only Q8_0 is stage-tested in stage 1; add a
   stage-1 pass for the real type.
2. **G8 re-check** — the round-10 gather + per-entry decode-lpr8 fix is verified by `moe2048 eh/ey`;
   the ladder re-run is its first end-to-end test.
3. **One-graph structural hazard** — round 8b saw trace-mode (per-layer sub-graphs) produce
   different texts than one-graph runs; the `k_tables` race was found exactly this way. If the
   ladder fails only in one-graph mode, bisect the chunk graph by committing prefixes as separate
   command buffers.

**F. Guardrails.** One change per commit; no CUDA/HIP/SYCL files; update
`metal/tests/q36_compile_test.cpp` only if pipelines change (not expected); update the T2 State
paragraph of `docs/QWEN36_METAL_ROADMAP.md` with the measured green (stage count, ladder result,
Mac/date) as the last step.

## What success looks like

```
$ ./build-metal/q36_chunk_parity
  ... 38 stages, each "OK" ...
all stages bit-identical          # exit 0
$ python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-chunk.json
identical: 10 of 10               # exit 0
```

## Learnings for future agents

- The harness names stages as `check("name", ...)` calls inside `// ---- N. ...` blocks; runtime
  count = call sites × loop passes (37 at tip `9d4c0aee`).
- Ladder parity always means: `q36_solo_gate.py` twice (plain / `--prefill 512`) + `q36_cmp_gate.py`
  — never eyeball engine logs.
- Trace gates must pass `--log` to `q36_solo_gate.py` or the engine's stderr (all TRACE output)
  is discarded; `q36_trace_cmp.py` prints LAYERS:/CUTS: verdicts.
- This codebase keeps bit-exact parity by construction (the chunk kernels reuse the decode kernels'
  summation order); any "close enough" arithmetic-order change is a bug, not a tolerance question.
- Mac rounds are user-run; each push should end with the exact commands to run and what to paste
  back (see prior ticket 000675's Phase B rounds for the cadence).
