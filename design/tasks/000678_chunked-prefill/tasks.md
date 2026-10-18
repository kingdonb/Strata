# Implementation Tasks: Chunked Prefill Benchmark and --prefill Default Flip on M2

## Bench tool (this Linux sandbox)

- [x] Add `tools/qwen35/q36_prefill_bench.py`: per-(size, round) seeded prompt ids (stop ids avoided, no shared prefixes), one engine start per `--prefill` value via `q36_harness.Eng`, discarded warm-up per start (parity prompt 0), per size × round GEN n=16 capturing client TTFT, PP lines and the DONE fields (prompt_ms, fresh), fresh-count assertion, median aggregation, JSON + markdown table output, `line()` timeout ≥ 1800 s
- [x] Record a refused start (memory gate) as a row note instead of crashing
- [x] Smoke-test the tool against a stub engine (READY/INFO/PP/T/DONE script) and verify the JSON and table; `python3 -m py_compile` every touched file

## Benchmark + decision (needs the M2 Mac — maintainer)

- [ ] Build on the M2 (docs/QWEN36_METAL.md §5 commands), then run:
      `python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata --args "--native $Q36_MODEL --ctx 32768" --sizes 2048,4096,8192 --chunks 0,512,1024,2048 --rounds 3 --gen 16 --out bench/results/<date>-m2-q36-prefill/bench.json`
- [ ] Compute per-size chunked/baseline prompt-tok/s ratios; apply the ≥ 5× rule (every measured size must clear)

## Engine change (conditional on the gate)

- [x] Engine scaffolding in `metal/src/engine/q36_metal.cpp` `run()`: `prefill_given` distinguishes flag-absent from `--prefill 0`; a named `default_prefill_chunk` constant (currently 0, behavior-identical) holds the future flip value; `--prefill 0` and `STRATA_Q36M_PREFILL=1` keep today's behavior
- [ ] After the M2 numbers arrive: the actual flip in `run()` — set `default_prefill_chunk` to the parity-verified winner only if every measured size cleared ≥ 5×, add the before-allocation memory downgrade for a defaulted chunk (explicit `--prefill N` and `ctx > SK_TG_CAP` keep today's refuse), and re-point `--prefill auto` only if the winning size differs from 512 and passed parity (expected outcome from the T3 data: no flip, the default stays 0)
- [ ] If the gate is not met: leave the default at 0 and record that in the doc

## Parity + docs (needs the M2 Mac — maintainer)

- [ ] Re-verify parity under the shipping default: `q36_solo_gate.py` with no `--prefill` flag vs `--prefill 0`, then `q36_cmp_gate.py` → 10/10 identical
- [ ] Update docs/QWEN36_METAL.md §6: benchmark table (prompt size, chunk, TTFT ms, prompt tok/s, ratio; date, M2/macOS, model file, ctx, build), the decision paragraph with the numbers and the escape hatch, the parity line, the reproduction command block; update the "Prompts run token-by-token unless --prefill is set" sentence if flipped
  - [x] The reproduction command block (bench + default-vs-token-by-token parity commands) is in §6's reference block — added before the M2 run so the maintainer can copy-paste it
- [ ] Doc style pass: plain words, every number carries what it was measured on, state what was not tested (e.g. other Apple Silicon)
