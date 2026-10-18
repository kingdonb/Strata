# Requirements: Metal MTP Speculative Decoding (--mtp 2)

Add opt-in MTP (Multi-Token Prediction) speculative decoding to the Metal engine for
Qwen3.6-35B-A3B (`qwen35moe`): `metal/src/engine/q36_metal.cpp`. This is roadmap item **T4** of
`docs/QWEN36_METAL_ROADMAP.md` (depends on T2, green). The contract is the CUDA engine's
`spec_step` in `src/qwen35/q36_engine.cu` — the Metal engine mirrors it. Greedy-only, off by
default (`--mtp 0` / absent flag = today's behavior, byte-identical).

## User stories

- **Mac user**: I pass `--mtp 2` and greedy decode gets measurably faster (target ≥ 1.2×, toward
  20–25 tok/s on the M2) while producing exactly the same tokens as without it.
- **Maintainer**: the default path (no `--mtp`) stays byte-identical; the chunked-prefill path
  (T2) and the split-K attention path keep their parity; the CUDA engine remains the reference
  this port is checked against.

## Scope (from the ticket)

- Ingest the `blk.40` MTP block (`qwen35moe.nextn_predict_layers`) weights at startup when
  `--mtp > 0`: the block is a full-attention layer (attn_q with fused output gate, attn_k/v/output,
  q/k norms) + MoE, plus `nextn.eh_proj.weight` (2N×N), `nextn.enorm/hnorm/shared_head_norm` (N).
- Mirror the CUDA `spec_step`: a redo-row draft call, a chained draft seeded by the block's own
  output, a verify window (2 draft tokens → a 3-row window) with per-row head and argmax.
- Support accept-prefix rewinds: the window must not consume the real recurrent state S / Conv1D
  state of the GDN layers; the kept prefix's states are (re)built by a commit step.

## Acceptance criteria

1. **Text parity (gate 1)**: with `--mtp 2`, greedy output is token-identical to `--mtp 0` on all
   10 parity prompts (`tools/qwen35/parity_prompts.ids`, 128 tokens, via `q36_solo_gate.py` +
   `q36_cmp_gate.py`), verified twice on the same build.
2. **Measured speedup (gate 2)**: decode tok/s base vs `--mtp 2` measured on the same Mac, same
   day, same build, and documented in `docs/QWEN36_METAL.md` with hardware and date. Target
   ≥ 1.2× (≈ 20–25 tok/s). The flag remains opt-in either way — this PR never flips a default.
3. **Default path untouched**: with `--mtp` absent or `--mtp 0`, the 10-prompt token output is
   byte-identical to the pre-change build's base run, and `metal/tests/q36_chunk_parity.cpp`
   stays all-OK (the chunk path is not regressed).
4. **Flag behavior**: `--mtp N` parses on the Metal engine (integer drafts); `--mtp 0` / absent =
   off; `--mtp N > 0` on a model whose GGUF has no `nextn_predict_layers` block is refused with a
   clear message; drafts are only offered to greedy requests (serve: temperature ≤ 0.01) and only
   while `pos + N + 1 ≤ ctx`.
5. **State contract**: after every step, the engine's state (recurrent S, conv history, KV cache)
   corresponds exactly to the tokens that were emitted/shown — including when a window is cut
   short by EOS, cancel, `max_new`, or context end — so serve prefix caching stays exact.
6. **Build**: `q36_compile_test` passes (including the MTP-enabled pipeline/scratch init).

## Non-goals

- Non-greedy (sampled) speculative decoding; batched multi-slot MTP windows (CUDA's
  `--batch-mtp`); the draft-vocab subset head (CUDA's `STRATA_Q36_DRAFT_VOCAB`, see Open
  Questions); FP16 KV cache (T5); GQA-folded attention (T6); any change to files outside
  `metal/` plus the one docs section; changing the serve protocol's `T`/`DONE` line format
  (acceptance stats go to stderr, see Open Questions).

## Open Questions

1. **Acceptance reporting**: keep the serve `DONE` line byte-for-byte as today and print the
   offered/accepted draft counts on stderr only (like the CUDA CLI does), or add fields at the
   end of `DONE`? Default assumption: stderr only, protocol unchanged.
2. **Draft head subset**: CUDA has an opt-in draft head over the most frequent tokens
   (`STRATA_Q36_DRAFT_VOCAB`, the `draft_vocab.bin` file). In scope for T4, or deferred to a
   follow-up if the full-vocab head's cost hurts the measured ratio? Default assumption: defer.
3. **Max draft count**: accept any `--mtp N ≥ 1` (CUDA allows up to its picks[16] array) or cap
   at 2 for now? Default assumption: parse N ≥ 1 (cap 16), verify gates at N=2.
