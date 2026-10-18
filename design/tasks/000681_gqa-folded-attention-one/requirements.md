# Requirements: GQA-Folded Attention — One K/V Read per KV-Head Group

## Background

Qwen3.6-35B-A3B (`qwen35moe`) on the Metal engine uses Grouped Query Attention: 16 query heads
share 2 KV heads (H/HKV = 8), head_dim 256, 10 full-attention layers. Today every query head reads
the same K/V cells independently — in `k_attn` (decode and chunk rows at n_pos ≤ 8,192 cells) and in
`k_attn_sk_part` (rows above that, or `STRATA_Q36M_ATTN_SK=1`) — an ~8x KV memory-traffic
multiplier that L2 is not catching.

Measured on the M2 (UD-Q2_K_XL, 2026-10-18, build `36b394c8` + PR #5, from the T3 data in
docs/QWEN36_METAL_ROADMAP.md): a 4,098-token prompt at `--ctx 16384` reads in ~243 s chunked
vs ~245 s token-by-token — a wash, because the parity contract keeps the KV re-read term identical
on both paths (per-512-token chunk cost grows 22 -> 40 s with position), and decode at pos ~4,100
runs 8 tok/s vs 17 at shallow ctx-4096 positions. At depth the attention's KV reads bound both paths.

The geometry (read at runtime from the GGUF; cross-checked against the upstream config:
`num_attention_heads: 16`, `num_key_value_heads: 2`, `head_dim: 256`) gives the fold factor 8 on this
model. The kernels must read H/HKV from the geometry, never hardcode it.

## User Stories

1. As an operator reading long prompts on Apple Silicon, I want the attention pass to read each
   KV-head group's K and V once and score all its query heads together, so prompt reads and
   deep-context decode stop paying the ~8x redundant KV traffic.
2. As a maintainer, I want the folded kernels behind the parity knob `STRATA_Q36M_ATTN_GQA=1`
   (the `STRATA_Q36M_ATTN_SK` pattern) with proven token parity, so the default path stays
   byte-identical until a measurement justifies flipping it.

## Acceptance Criteria

Gates 1-4 are the task's acceptance gates; 5-6 are the project's own rules applied to this change.
All measured on the same M2, same day, same build.

1. **Compilation**: `q36_compile_test` passes and creates all pipelines, including the three new
   folded ones.
2. **Parity**: forced fold (`STRATA_Q36M_ATTN_GQA=1`) is 10/10 token-identical vs the default path
   on the 10 parity prompts at `--ctx 4096` (`tools/qwen35/q36_solo_gate.py` +
   `q36_cmp_gate.py`). The T2 history (row-blocked split-K vs `k_attn` ULP differences flipped
   greedy tokens) means this gate in practice demands bit-parity of the attention arithmetic, not
   near-equality.
3. **Chunk parity**: with the fold active, the chunked ladder (plain vs `--prefill 512`) remains
   10/10 identical (twice), and `q36_chunk_parity` stays all-OK.
4. **Deep-context acceleration, documented**: the measured speed delta on M2 for a 4,098-token
   read at `--ctx 16384 --prefill 512` (TTFT, the ~243 s baseline) and decode tok/s at depth
   (pos ~4,100, the 8 tok/s baseline), fold off vs on, written into `docs/QWEN36_METAL.md` with
   what was measured, on what hardware, and the date.
5. **Default unchanged**: knob unset -> no new dispatches, kernels, or scratch on the default path;
   the base json matches the pre-change build's base json 10/10.
6. **Guards**: with the knob set, the engine refuses geometries where folding is impossible or
   pointless — H/HKV < 2 (no-op) or threadgroup staging past the 32 KB threadgroup ceiling —
   printing the numbers (the `head_dim 256` refusal pattern).

## Non-Goals

- No default flip of the fold, and no `--prefill` default flip (the measurements this task produces
  feed that later decision).
- No CUDA/HIP engine changes: the same redundancy exists in `src/qwen35/q36_engine.cu`, but this
  task is Metal-only and touches no shared headers.
- No FP16 KV cache (T5), MTP (T4), or online-softmax rewrite of the default kernels.

## Open Questions

None. Geometry verified against the upstream config (H=16, HKV=2, head_dim 256 -> HG=8); the
engagement, knob, and default-path rules are all settled by docs/QWEN36_METAL_ROADMAP.md T6.
