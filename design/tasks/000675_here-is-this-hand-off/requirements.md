# Requirements: Qwen3.6 Metal Long Context, Chunked Prefill and MTP

## Background

The merged Metal work (branch `feat/qwen36-metal`, `metal/src/engine/q36_metal.cpp`) runs Qwen3.6-35B-A3B
token-by-token: ~15-20 tok/s prefill, 12.5-14.5 tok/s sustained decode on an M2 24 GB, context hard-capped at
8,192 tokens because `k_attn` keeps attention scores in threadgroup memory (32 KB ceiling), KV cache in fp32
(40 KB/token), and the GGUF's MTP layer (`blk.40`) is skipped at load. The hand-off (docs/QWEN36_METAL.md §4)
lists five follow-up priorities. This spec covers them, reshaped against what the repo has already proven on
CUDA/HIP (src/qwen35/q36_engine.cu + docs/QWEN36.md): split-K attention, chunked prefill and MTP all exist
there with measured numbers and parity gates — Metal ports them instead of re-inventing.

## User Stories

### 1. Long prompts (context beyond 8K)
As a Mac user, I want context windows of 16K-64K so that long documents and long conversations fit.
- Engine starts with `--ctx 16384` and `--ctx 32768` on the 24 GB reference Mac, ingests a ≥2,000-token
  prompt and generates 128 greedy tokens with no Metal errors and no silent truncation.
- At `n_pos ≤ 8192` the decode path is unchanged: the 10 parity prompts (tools/qwen35/parity_prompts.ids),
  greedy 128, give exactly the pre-change build's tokens.
- A test-only knob that forces the new attention path at short context gives token-identical greedy output
  to the default kernel on the 10 parity prompts.
- A requested context that KV + recurrent state cannot hold is refused at startup with the numbers (bytes
  needed vs available), in the CUDA engine's refusal style — never a runtime crash or silent cap.

### 2. Fast prompt processing (chunked prefill)
As a user, I want a 3K-token prompt answered in seconds, not minutes.
- A chunked prompt path (chunk sizes 512 / 1024 / 2048 / 4096) is implemented, opt-in first.
- Chunked prefill gives the same greedy 128 continuation as the token-by-token path on the 10 parity prompts
  (single-chunk) and on ≥3 real documents of 1.5-3K tokens (multi-chunk); logits KL per chunk size is logged.
- Measured prefill tok/s per chunk size vs the token-by-token path, same Mac, same day, recorded in
  docs/QWEN36_METAL.md. The default flips to a larger chunk only if measured ≥5x for multi-K prompts;
  otherwise it stays opt-in with the numbers published.
- Multi-turn prefix-cache reuse (held recurrent state, Conv1D history, KV cache) still works when the prompt
  is read in chunks.

### 3. Faster decode (MTP speculative decoding)
As a user, I want higher sustained tok/s from the MTP draft layer the file already carries.
- `--mtp 2` (default off) loads `blk.40`'s nextn weights and runs draft-verify decoding.
- Greedy text with `--mtp 2` is identical to `--mtp 0` on the 10 parity prompts (the CUDA engine's guarantee).
- Measured decode ratio ≥1.2x on the reference Mac before any default change; feature stays opt-in.
- Stop tokens, `STOP` cancel, length limit, context-end guard and prefix-cache reuse all behave the same with
  drafts active.

### 4. Honest numbers and docs
As a maintainer, I want the hand-off's unmeasured claims replaced with measurements.
- docs/QWEN36_METAL.md gains a measured-numbers section (what, which Mac, which date) and drops or marks
  estimates (e.g. the "1,500-3,000+ tok/s" prefill and "22-28 tok/s" decode figures were projections).
- The stale `./run-q36.sh` reference in docs/QWEN36_METAL.md §5 is fixed or the script restored.

## Non-goals
- O(log T) associative parallel scan for the GDN delta rule — the CUDA engine ships the exact
  sequential-per-token-in-chunk scheme and lists chunked (WY) GDN as "not attempted".
- `simdgroup_matrix` / Apple AMX rewrite of the dequant GEMVs — decode is memory-bandwidth bound; the
  MetalPerformancePrimitives path is M5-only (used by the Flash-Next Metal window).
- Multi-request batching (`--batch`) on Metal, vision/multimodal, FP16 KV as a default.

## Open Questions
1. Which Macs must pass the gates — M2 24 GB only, or also M4/M5? (moe_mm.metal's matmul2d grouped-GEMM is an
   M5 path; on M4/M5 a faster MoE chunk kernel could follow later.)
2. Once the parity gates pass, should the chunked-prefill default flip in this same task (the CUDA engine
   flipped after measuring), or stay opt-in until a later round?
3. Is solo-path-only MTP with a fixed draft count (`--mtp 2`) acceptable for the first Metal round (no
   `--batch-mtp`, no batch windows — batching itself is out of scope)?
4. Should the opt-in FP16 KV cache (`--kv-f16`, halves 40 → 20 KB/token) be part of this round or deferred?
5. Host sampling loops ~248K logits per generated token (top-p sorts them); if profiling shows ≥2 ms/token,
   may a GPU argmax / host top-k prefilter (tokens unchanged, RNG stream preserved) be included?
