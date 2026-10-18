# Requirements: Chunked Prompt Prefill — Parity Green (38/38 Stages)

## Background

Strata's Qwen3.6 Metal engine (`metal/src/engine/q36_metal.cpp`) has chunked prompt prefill
implemented (`forward_prompt_chunk`, `--prefill 512`): dense projections batch over the chunk's T
rows, MoE runs grouped per-expert, and the last token's decode window produces logits. The parity
harness `metal/tests/q36_chunk_parity.cpp` compares every chunk kernel against its token-by-token
decode kernel bit-for-bit. Mac round 10c resolved the last open combine question: both combines'
sums are correct — the GPU saw the harness's memcpy-seeded test buffer `x2` (64 KB, shared storage)
as **zeros** at combine time, while loop-written buffers and a 32-byte memcpy reach fine. GPU-copy
isolation probes are already pushed at branch tip `9d4c0aee` (`feat/qwen36-metal`) but not yet run.
The engine's chunk ladder (the 10 parity prompts, plain vs `--prefill 512`) is still 0/10
token-identical.

## User Stories

**US-1 — Harness green (verifier, on the M2).** As the developer verifying on the Mac, I run
`./build-metal/q36_chunk_parity` and it reports every stage OK with 0 failures, so the chunked
prefill path is proven bit-exact stage by stage and the ticket's first gate is closed.
- AC: the moe2048 block prints the probe lines (A/B/C + diag), the `moe2048 x` stage passes, and a
  new checked isolation-combine stage (`moe2048 iso`) passes; the suite exits with
  `all stages bit-identical` and exit code 0.
- AC: stage 11 (the MoE combine over the chunk vs per token) still passes after any seeding change.

**US-2 — Regression guard.** As a maintainer, I want the isolation combine to stay in the harness
as a checked stage (not just a print), so the host-memcpy-visibility failure class cannot silently
return.
- AC: the suite reports 38 stages OK (37 named checks at tip `9d4c0aee` + the promoted isolation
  stage); the probe prints stay in the output.

**US-3 — Engine ladder parity (user, on the M2).** As a user of the Metal engine, `--prefill 512`
must produce exactly the same tokens as the default token-by-token prefill, so chunked prefill is
safe to make the default later (T3 owns that flip, not this ticket).
- AC: `q36_solo_gate.py` runs plain and `--prefill 512` over the 10 parity prompts (greedy, 128
  tokens); `q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-chunk.json` reports `identical: 10 of 10`.

## Acceptance Gates

1. `./build-metal/q36_chunk_parity` reports **38/38 stages OK, 0 failures** on the M2.
2. Parity ladder: **10/10** parity prompts greedy 128 identical, plain vs `--prefill 512`, via the
   documented `q36_solo_gate.py` + `q36_cmp_gate.py` json compare.

## Non-Goals

- Prefill speed measurement and the `--prefill` default flip (T3); MTP (T4); FP16 KV cache (T5);
  docs wrap-up and the upstream batch PR (T6).
- No CUDA/HIP/SYCL file is touched; no engine behavior change when `--prefill` is off.
- No MSL pipeline changes are expected. If a fix turns out to need a new/renamed kernel, the change
  grows `metal/tests/q36_compile_test.cpp` in the same commit.

## Constraints

- Sandbox is Linux: nothing Metal compiles or runs here. Every push to `feat/qwen36-metal` ends
  with a request to build (`cmake --build build-metal`) and run the gates on the M2. Host C++ gets
  a `g++ -fsyntax-only` pass where headers allow.
- One fix per Mac round; evidence (pasted console blocks) decides the next step. Bundled probe
  batteries are the default (reviewer decision); the round count scales with the number of
  unresolved hypotheses — no one-round guarantee.

## Open Questions

1. **Stage count (assumption — no objection at review).** The ticket's gate says 38/38; the harness
   at tip `9d4c0aee` prints 37 named checks. Plan: promote the isolation combine (probe C) to a
   checked stage `moe2048 iso` → 38. Say so before Step 2 lands if you'd rather name a different
   38th stage.
