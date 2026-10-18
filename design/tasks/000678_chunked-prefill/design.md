# Design: Chunked Prefill Benchmark and --prefill Default Flip on M2

## Current state (verified in the repo)

- `metal/src/engine/q36_metal.cpp` `run()`: `int prefill_chunk = 0` — token-by-token
  unless the flag is given. `--prefill auto` (prefix match) maps to 512 with a stderr
  note; `STRATA_Q36M_PREFILL=1` also forces 512 when the flag is absent/0. The flag is
  a startup arg, so each chunk size needs its own engine start.
- The chunk loop (`run()`, serve branch ~L1648) reads `ids.size() - 1` tokens in
  `prefill_chunk`-wide chunks, the last token runs as a decode window (the logits
  source), and `PP <read> <total> <ms> <fresh tok/s>` prints once per chunk.
- `DONE <produced> <prompt_len> <prompt_ms> <decode_ms> <finish> 0 0 <reuse> 0 0 0 0
  0.0 <fresh> 0`: field 3 = prompt_ms (the fresh read, includes the last-token decode
  window), field 8 = reused prefix tokens, field 14 = fresh tokens. This is the same
  convention the server's request line reports as `prefill X tok/s`.
- Startup memory gate: when `ctx > SK_TG_CAP (8192)` or `prefill_chunk > 0`, refuses
  if `cx.allocated() + 0.25 GB > recommended_working_set()` unless
  `STRATA_Q36M_MEM_OK=1`. The chunk buffers are allocated *before* the gate, so they
  count in the estimate; `Context::alloc` `exit(4)`s on failure.
- setup already writes `--prefill auto` for this family (`setup.py` `install_own_path`,
  `metal/setup_mac.py` `engine_args`), so configured installs already opt in. The
  opt-in surface this task decides is the engine's no-flag default.
- `tools/qwen35/q36_harness.py` `Eng`: speak-the-protocol driver. `Eng.solo` returns
  `(tokens, DONE line, ttft, wall)`; it collects PP lines but drops them. Ten parity
  prompts in `parity_prompts.ids`; `q36_cmp_gate.py` compares two solo-gate JSONs
  token-for-token. `q36_par_bench.py` is the pattern for a bench tool (args, JSON
  lines, results folder).

## Benchmark design — `tools/qwen35/q36_prefill_bench.py`

New tool beside the q36 gates, mirroring `q36_par_bench.py`'s interface. It does not
touch `q36_harness.py`; it reuses `Eng`'s public `send`/`line`/`info` and runs its own
GEN loop so it can keep the PP lines.

- **Prompts**: deterministic seeded ids per (size, round), drawn from a safe id range,
  stop ids (248046, 248044, 151645, 151643) avoided. Different seed per round means no
  shared prefix — the engine's prefix matcher would otherwise reuse the whole prompt on
  round 2 and report fresh=0. The tool asserts each run's DONE fresh count equals the
  prompt size and flags the row otherwise.
- **Matrix**: `--prefill` 0 (baseline) / 512 / 1024 / 2048; sizes 2048 / 4096 / 8192;
  `--ctx 32768` fixed (the §6-verified long-context state; an 8K prompt plus
  generation needs more than 8192 anyway). One engine start per chunk value.
- **Protocol**: per start, one discarded warm-up request (parity prompt 0, n=8 — the
  documented 10.1 s cold first request is skipped cheaply); then each size × rounds,
  GEN n=16 (TTFT only needs the first token; decode tok/s is recorded for sanity).
- **Metrics per run**: client TTFT (send → first `T` line), engine `prompt_ms`, fresh
  tokens, `decode_ms`; prompt tok/s = fresh ÷ prompt_ms (end-of-prompt, DONE-based —
  the number comparable with the server's request line). Aggregation: median of
  rounds. `Eng.line()`'s default 600 s timeout is too short for an 8K token-by-token
  read (~8 min at ~17 tok/s) — pass a ≥ 1800 s timeout.
- **Output**: JSON with per-run raws (reproducible, recomputable) + a printed markdown
  table; results land in `bench/results/<date>-m2-q36-prefill/` following the repo's
  folder convention. A refused start (memory gate at chunk 2048, say) is recorded as
  a row note, not silently dropped.
- **Ratio and gate**: per size, chunked prompt tok/s ÷ baseline prompt tok/s. The flip
  requires ≥ 5× on every measured size. TTFT is reported but the gate keys on prompt
  tok/s (TTFT moves with it here — the Metal serve loop has no decode overlap, so
  TTFT ≈ prompt read + one decode step).

## The flip (conditional, in `q36_metal.cpp` `run()`)

- Track whether `--prefill` was given. Absent → the default chunk (the measured,
  parity-verified winner; 512 unless the table says otherwise and that size passed
  parity). Explicit `--prefill 0` → token-by-token, the old default path, unchanged.
  `STRATA_Q36M_PREFILL=1` behavior unchanged.
- **Memory-safety downgrade**: because chunk buffers are allocated before the memory
  gate, a defaulted chunk must be checked *before* that allocation — reuse the gate's
  existing estimate plus the chunk-buffer formula the startup log already prints. If a
  defaulted (not explicit) chunk would push the estimate past the working set and
  `ctx <= SK_TG_CAP`, drop to token-by-token with a stderr note instead of refusing;
  explicit `--prefill N` and `ctx > SK_TG_CAP` keep today's hard refuse. This keeps
  "starts that used to start still start" while the measured fast path stays the
  default on machines that fit it (the M2 24 GB target does: ~11.4 GB weights,
  ~0.5 GB chunk buffers at 512).
- `--prefill auto` mapping: stays 512 unless the winning size differs and passed
  parity; then `auto` re-points at the winner and `--prefill 512` remains the
  documented old size (mirrors docs/QWEN36.md's CUDA-engine flip, where `--prefill
  512` / `STRATA_Q36_CHUNK=512` is the escape hatch).
- setup and the written configs do not change (they already pass `--prefill auto`).

## Doc changes — docs/QWEN36_METAL.md §6

New subsection "Chunked-prefill speed on M2 (measured <date>)" after the §6 gates:

- One hardware/method line: Apple M2 (macOS 26.6.2), the §5 model file
  (UD-Q2_K_XL), the engine build (commit/date), `--ctx 32768`, greedy, n=16, median
  of 3, warm-up discarded, deterministic seeded prompts.
- The table: Prompt tokens × `--prefill` → TTFT ms, prompt tok/s (+ a ratio column).
- The decision paragraph, in the style of docs/QWEN36.md's "Prompt chunk (the
  default changed)": the numbers, the ≥ 5× verdict, the new default (or "stays
  opt-in"), the `--prefill 0` escape hatch, and the parity line ("the 10 parity
  prompts under the new default: 10/10 token-identical to `--prefill 0`").
- The reproduction command block (solo-gate/cmp-gate style, like §5's).
- If flipped, update §6's sentence "Prompts run token-by-token unless `--prefill` is
  set" and the §3.1 concession text pointer.

## Validation

- Here (Linux, no Metal): stub-engine smoke test of the bench tool (a small script
  that speaks READY/INFO/PP/T/DONE), JSON + table rendering checked;
  `python3 -m py_compile` on touched files. The engine flip compiles only on macOS —
  the M2 build + gates (§5 commands) are the validation there, listed in tasks.md.
- On the M2: run the bench, apply the flip per the numbers, run the two solo gates +
  cmp gate (10/10), paste the table into the doc.

## Notes for implementers (learnings)

- Engine protocol: `Eng(exe, args)` starts `exe --serve <args>`; lines are
  `READY/INFO/PP/T/DONE/ERR`; TTFT is client-measured (first `T` after send).
- Repeated identical prompts hit the prefix cache — vary ids per round or the
  fresh/prompt_ms numbers collapse to zero.
- First request after engine load is cold (documented 10.1 s for ~30 tokens) —
  always warm up and discard.
- `Context::alloc` exits(4) on failure and the memory gate sits after chunk-buffer
  allocation — any defaulted-chunk downgrade must be decided before that allocation.
- Docs style: plain words, every number carries what it was measured on; results
  folders follow `bench/results/YYYY-MM-DD-<topic>/`; one change per PR, opt-in
  defaults until measured.

## Implementation notes (2026-10-18, what actually happened)

- **Ticket #000677's T3 data was already on main** (docs/QWEN36_METAL_ROADMAP.md):
  chunked 512 ≈ token-by-token at depth (243 vs 245 s for a 4,098-token prompt at
  --ctx 16384) because both paths re-read the same KV cells ~8x (GQA); short prompts
  gain ~30% TTFT. So the expected gate outcome for this benchmark is **no flip** —
  the tool exists to confirm it with the full 2K/4K/8K × 512/1024/2048 table, and
  the data feeds T6 (GQA-folded attention). The reviewer confirmed this reading.
- **Engine scaffolding** landed behavior-identical: `prefill_given` (flag-absent vs
  `--prefill 0`) + `const int default_prefill_chunk = 0` in `run()`. The flip, if
  the numbers ever say ≥5× everywhere, is a one-constant edit plus the
  before-allocation memory downgrade — written then, not blind now (q36_metal.cpp
  compiles only on macOS, so every blind edit risks breaking the maintainer's M2
  build; the bench must run on an unmodified engine anyway).
- **DONE-line precision**: the engine prints `prompt_ms` with %.1f, so ratios carry
  ~1e-5 relative drift; the stub test compares with 1e-3 tolerance. Same drift will
  appear between table rows on the M2 — harmless at a 5× gate.
- **Stub-test gotchas found**: the stub must apply its speed multiplier only to the
  chunked rows (the first version made the baseline fast too and the gate verdict
  read 1.00x); the ctx-guard case needs a size that genuinely exceeds --ctx
  (32000 + 16 still fits 32768 — use 33000).
- The tool rewrites its JSON after every cell, so a killed run keeps its finished
  rows; a refused start is a row note + its stderr log (engine-prefill-<chunk>.log).
- Order of changes that worked: tool → stub test → scaffolding → doc command block,
  one commit each; the M2 run, flip decision, parity re-run and doc table follow
  when the maintainer pastes the bench output.
