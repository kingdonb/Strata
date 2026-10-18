# Requirements: Chunked Prefill Benchmark and --prefill Default Flip on M2

## Background

The Metal Qwen3.6-35B-A3B engine (`metal/src/engine/q36_metal.cpp`) reads prompts
token-by-token unless `--prefill N` is passed; `--prefill auto` maps to the
parity-verified chunk of 512, which is what setup writes for this family
(docs/QWEN36_METAL.md §6). The chunked path exists and passed the 10-prompt parity
gate at 512, but no speed numbers for it have been published: §3.1 of the same doc
still describes prompt reading at ~15-20 tok/s and §4 Priority 2 predicted
1,500-3,000+ tok/s. Strata's convention is that a default changes only where it was
measured faster, with the numbers shown.

Goal: benchmark TTFT and prompt tok/s on 2K / 4K / 8K-token prompts across chunk
sizes 512 / 1024 / 2048 against the token-by-token baseline, on the M2. Flip the
engine's `--prefill` default to active (chunked) only if measured ≥ 5× faster;
otherwise keep it opt-in. Publish the numbers in docs/QWEN36_METAL.md.

## User stories

1. As a Mac (M2) user sending multi-thousand-token prompts, I want prompt reading to
   be fast by default, so time-to-first-token drops from minutes toward seconds.
2. As a maintainer, I want the default flip decided by a published measurement
   table against the ≥ 5× gate, not by expectation.
3. As a user who needs the previous behavior exactly (byte-identical token-by-token
   reads, or debugging), I can pass `--prefill 0` and get the last release's path.
4. As a maintainer reading docs/QWEN36_METAL.md, I want the benchmark table (prompt
   size, chunk size, TTFT, tok/s, date, M2 hardware) and the parity result in the
   repo's doc style.

## Acceptance criteria

- AC1 — A bench tool (`tools/qwen35/q36_prefill_bench.py`) measures, for prompt sizes
  2048 / 4096 / 8192 × `--prefill` 0 / 512 / 1024 / 2048: client TTFT (ms from request
  sent to first token) and prompt tok/s (fresh tokens ÷ the engine DONE line's
  `prompt_ms`), median of 3 rounds after a discarded warm-up, writing JSON plus a
  printed markdown table.
- AC2 — The table is published in docs/QWEN36_METAL.md with the date, the M2
  hardware/macOS line, the model file, `--ctx`, and the engine build — the same style
  as the §6 gate numbers.
- AC3 — The flip decision follows the table strictly: flip (no-flag start reads
  chunked) only if every measured prompt size reads ≥ 5× the token-by-token
  baseline's prompt tok/s on the same build; otherwise the engine keeps
  token-by-token as the no-flag default and the doc states that with the numbers.
- AC4 — If flipped: `--prefill 0` still produces the last release's token-by-token
  path unchanged, and a defaulted (not explicitly requested) chunk that the startup
  memory gate would refuse downgrades to token-by-token with a stderr note instead
  of refusing a start that used to start. Explicit `--prefill N` keeps today's
  refuse behavior.
- AC5 — The 10 parity prompts are re-verified under whatever default ships:
  `q36_solo_gate.py` with no `--prefill` flag vs `--prefill 0`, and
  `q36_cmp_gate.py` reports 10/10 identical.
- AC6 — The bench tool is smoke-tested on the Linux sandbox against a stub engine
  (no Metal needed), and the exact M2 command block that reproduces the table is in
  docs/QWEN36_METAL.md.

## Open Questions

1. The ≥ 5× gate: must every measured prompt size (2K, 4K, 8K) clear it, or is an
   aggregate (e.g. geometric mean) enough? Proposed: every size, the conservative
   reading — confirm.
2. If 1024 or 2048 wins the benchmark but 512 is close, does the default flip to the
   winner even though only 512 has been parity-gated so far? Proposed: flip only to
   a chunk size that passes the 10-prompt parity gate; gate the winner on the M2
   before flipping to it.
3. The benchmark and the parity re-run need the M2 Mac (this sandbox is Linux and
   cannot build the Metal engine). Confirm the maintainer runs those steps from the
   command block in the doc, with the flip patch written here against the measured
   winner.
4. After a flip, should `--prefill auto` stay hardwired to 512, or map to the
   measured winner? Proposed: re-point `auto` at the winner only if that winner
   passed parity.
