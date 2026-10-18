# Implementation Tasks: Chunked Prefill Benchmark and --prefill Default Flip on M2

## Bench tool (this Linux sandbox)

- [x] Add `tools/qwen35/q36_prefill_bench.py`: per-(size, round) seeded prompt ids (stop ids avoided, no shared prefixes), one engine start per `--prefill` value via `q36_harness.Eng`, discarded warm-up per start (parity prompt 0), per size × round GEN n=16 capturing client TTFT, PP lines and the DONE fields (prompt_ms, fresh), fresh-count assertion, median aggregation, JSON + markdown table output, `line()` timeout ≥ 1800 s
- [x] Record a refused start (memory gate) as a row note instead of crashing
- [x] Smoke-test the tool against a stub engine (READY/INFO/PP/T/DONE script) and verify the JSON and table; `python3 -m py_compile` every touched file

## Benchmark + decision (needs the M2 Mac — maintainer)

- [x] Build on the M2 (docs/QWEN36_METAL.md §5 commands), then run:
      `python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata --args "--native $Q36_MODEL --ctx 32768" --sizes 2048,4096,8192 --chunks 0,512,1024,2048 --rounds 3 --gen 16 --out bench/results/2026-10-18-m2-q36-prefill/bench.json`
      (run completed 2026-10-18 on the M2 — all 12 cells × 3 rounds; first row: baseline 2K = TTFT 139.8 s, prompt 14.6 tok/s, decode 13.3 tok/s, TTFT ≈ 2048 ÷ 14.6 so the tool's numbers are self-consistent on real hardware; raw JSON committed by the maintainer as 91efae1a and verified against the published table — 36 rows, ratios match)
- [x] Compute per-size chunked/baseline prompt-tok/s ratios; apply the ≥ 5× rule (every measured size must clear)
      (verdict: **not met** — best 1.51x at 2K / 512, 1.27x at 8K; the tool's per-chunk verdicts: 512 = 1.51/1.43/1.27, 1024 = 1.48/1.38/1.25, 2048 = 1.48/1.39/1.24 — T3's prediction confirmed with the full matrix)

## Engine change (conditional on the gate)

- [x] Engine scaffolding in `metal/src/engine/q36_metal.cpp` `run()`: `prefill_given` distinguishes flag-absent from `--prefill 0`; a named `default_prefill_chunk` constant (currently 0, behavior-identical) holds the future flip value; `--prefill 0` and `STRATA_Q36M_PREFILL=1` keep today's behavior
- [x] After the M2 numbers arrive: the actual flip in `run()` — **decided: no flip**. The gate failed on every size (1.24-1.51x vs the 5x needed), so `default_prefill_chunk` stays 0, the engine needs no further change, and `--prefill auto` stays 512 (the best measured chunk; 1024/2048 read 1.5-3% slower)
- [ ] If the gate is not met: leave the default at 0 and record that in the doc

## Parity + docs (needs the M2 Mac — maintainer)

- [x] Re-verify parity under the shipping default: rebuild (the scaffold commit touched `run()`), then `q36_solo_gate.py` with no `--prefill` flag vs `--prefill 0`, then `q36_cmp_gate.py` → **10/10 identical** (maintainer run, 2026-10-18: `identical: 10 of 10` — the shipping default path is intact and the scaffolding changed no token)
- [x] Update docs/QWEN36_METAL.md §6: benchmark table (prompt size, chunk, TTFT ms, prompt tok/s, ratio; date, M2/macOS, model file, ctx, build), the decision paragraph with the numbers and the escape hatch, the parity line, the reproduction command block; update the "Prompts run token-by-token unless --prefill is set" sentence if flipped
  - [x] The reproduction command block (bench + default-vs-token-by-token parity commands) is in §6's reference block — added before the M2 run so the maintainer can copy-paste it
  - [x] The table + decision paragraph are in (commit 0826981d); the parity line lands after the maintainer's parity run
- [x] Doc style pass: plain words, every number carries what it was measured on, state what was not tested (e.g. other Apple Silicon — the benchmark and parity are M2-only so far, per docs/TEST_REQUESTS.md)

All acceptance gates met: (1) the benchmark table is published in docs/QWEN36_METAL.md §6 with the date, hardware, model, ctx and the raw results in `bench/results/2026-10-18-m2-q36-prefill/`; (2) the 10-prompt parity is re-verified (10/10) under the shipping default. The flip gate failed (1.24-1.51x vs the 5x needed), so the default stays opt-in by the reviewer-confirmed rule. Remaining delivery step: the maintainer retries opening the approved PR (prp_01m5874xp5mhqeyxfz2b6br08c) from Helix — the branch is reconciled and pushed.
