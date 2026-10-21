# Qwen3.6-35B-A3B on Apple Silicon Metal: Architecture, Implementation & Handoff

This document details the architecture, design decisions, concessions made, and roadmap for the native Apple Silicon Metal implementation of **Qwen3.6-35B-A3B** (`qwen35moe`) in Strata.

---

## 1. Executive Summary & Architecture

[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) is a 35-billion parameter Mixture-of-Experts (MoE) hybrid model with ~3B active parameters per token. It operates on an Attention-State-Space-Model (SSM) hybrid architecture:

- **Total Layers**: 41 layers.
- **SSM Layers (Gated Delta Net / Linear Attention)**: 31 layers (layers where `layer % 4 != 3`).
- **Full Attention Layers (QSA)**: 10 layers (layers 3, 7, 11, 15, 19, 23, 27, 31, 35, 39).
- **MoE Routing**: 256 routed experts + 1 shared expert, top-8 routing with routed expert FFN width 640 and shared expert FFN width 640.
- **Hardware Target**: Apple Silicon M-series (tested on Apple M2, 24 GB Unified Memory).
- **Quantization Evaluated**: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (~11.4 GB resident weights).
- **Current Performance**: ~51.5–63 ms per GPU decode step (~16–19.4 tok/s GPU rate), **12.5–14.5 tok/s sustained end-to-end** in Strata Web UI streaming.

---

## 2. What Was Built & Shipped

### 2.1. Vectorized Metal Shading Language Kernels (`metal/kernels/`)
- [`metal/kernels/quant.metal`](file:///Users/yebyen/w/Strata/metal/kernels/quant.metal):
  - Generic address space `vec4_ptr` template handling both `device` and `threadgroup` pointers.
  - Vectorized block dequantization dot-products `deq_dot_block` for `T_Q4_K`, `T_Q5_K`, and `T_Q6_K` utilizing 32-bit SIMD lane unpacking and `dot(float4, float4)`.
  - Specialized `k_gemv_lpr8` with threadgroup caching for 2,048-dim input vectors across 16 matrix rows per threadgroup.
- [`metal/kernels/q36.metal`](file:///Users/yebyen/w/Strata/metal/kernels/q36.metal):
  - `k_rms`: Root-Mean-Square Normalization with warp/simdgroup reductions.
  - `k_conv`: Gated causal Conv1D state update with SiLU activation.
  - `k_gdn`: Gated Delta Net recurrence step ($S_t = S_{t-1} + \dots$).
  - `k_gated_norm`: GDN output gating with RMSNorm.
  - `k_attn_prep`: QSA attention preparation with RoPE and grouped KV caching.
  - `k_attn`: Grouped Query Attention (GQA) causal attention calculation.
  - `k_swiglu`: SwiGLU activation for routed & shared FFN.
  - `k_topk_router`: GPU top-8 expert routing and softmax.
  - `k_moe_sum`: Weighted expert accumulation and shared expert gating.

### 2.2. Native Execution Engine (`metal/src/engine/q36_metal.cpp`)
- **Zero-Copy Memory Mapping**: Utilizes `cx.alloc_no_copy` with page-aligned offsets directly from mapped GGUF shards, avoiding duplicate host RAM copies.
- **IPC Resident Serve Mode**: Full support for Strata's `--serve` protocol (`READY <ctx> stop`, `GEN`, `T <id>`, `DONE`, `STOP`).
- **Multi-Turn Prefix Caching**: Matches `live_tokens` prefix across requests; preserves recurrent states $S$, Conv1D history, and KV-cache without full recomputation when conversational context prefixes match.  On divergence the serve loop keeps the common prefix instead of resetting (T8-r1): each request's prompt end is a checkpoint (the last two are kept; a snapshot is host copies of $S$/conv/logits/`hlast`, 66.9 MB per slot, two slots allocated once at the first checkpoint), a diverging request resumes from the deepest checkpoint that still prefixes it and reads only the divergent tail, and a divergence before every checkpoint still resets fully.  Measured (M2, 2026-10-20, build `ac5e0169`, the model from §5, `--ctx 4096`, `tools/qwen35/q36_prefix_reuse.py`): a strict extension reads only its delta (111 of 123 prompt tokens reused), a divergence at the reply's midpoint re-reads 36 of 99 prompt tokens from the checkpoint, an exact re-send reads 0 and reproduces the original greedy reply, and the restored run's tokens are bit-identical to a fresh full read — also with `--mtp 2 --draft-vocab 8192`; the 10-prompt ladder is 10/10 token-identical across two runs.  The `DONE` line's reused/actually-read fields report the resume point; `STRATA_Q36M_PREFIX_RESET=1` forces the old full reset (the A/B).  The decision logic is pure host C++ (`metal/src/engine/q36_prefix_plan.hpp`), unit-tested without Metal (`metal/tests/prefix_plan_test.cpp`, 23 checks); the reuse/divergence behavior measurement is `tools/qwen35/q36_prefix_reuse.py`.
- **Greedy & Nucleus Sampling**: Supports greedy decoding (temperature $\le 0.01$) and temperature with top-$p$ nucleus sampling.

### 2.3. System & Build Integration
- Wired `STRATA_HAVE_QWEN35` in [`metal/CMakeLists.txt`](file:///Users/yebyen/w/Strata/metal/CMakeLists.txt) and [`src/program/generate.cpp`](file:///Users/yebyen/w/Strata/src/program/generate.cpp).
- Exposed `base_` pointer in [`include/strata/artifact/gguf_reader.hpp`](file:///Users/yebyen/w/Strata/include/strata/artifact/gguf_reader.hpp).
- Built verification harness [`metal/tests/q36_compile_test.cpp`](file:///Users/yebyen/w/Strata/metal/tests/q36_compile_test.cpp).
- Added resumable chunked downloader [`metal/tools/download_q36.py`](file:///Users/yebyen/w/Strata/metal/tools/download_q36.py).

---

## 3. Concessions & Engineering Trade-offs Made

To rapidly achieve 100% numerical correctness, zero host regressions, and clean streaming within the time constraint, several intentional concessions were made:

### 3.1. Autoregressive Token-by-Token Prefill vs Chunked Tiled GEMM
- **Current State**: Prompt tokens are fed sequentially token-by-token through the decode forward pass (`forward_token`).
- **Concession**: TTFT (Time-To-First-Token) for long prompts scales at ~15–20 tok/s instead of hundreds or thousands of tok/s.
- **Rationale**: Allowed immediate deployment with exact parity verification, identical state updates, and avoided writing complex 2D tiled GEMM and parallel scan kernels before verifying the mathematical graph.

### 3.2. Single Dispatches vs Overlapped Multi-Stream Graphs
- **Current State**: All layer dispatches (Norm, GDN/Attn, Router, MoE, Shared Expert, Residual Add) are encoded sequentially into a single Metal command buffer per token.
- **Concession**: Shared expert computation does not run concurrently on an independent stream with routed expert computation.
- **Rationale**: Simplifies synchronization, prevents resource contention on M2's unified shader cores, and ensures deterministic completion.

### 3.3. FP32 Activations and KV-Cache
- **Current State**: All intermediate activation vectors (`x`, `h`, `y`, `qkv`, `z`, `S`) are allocated in 32-bit floating point (`float`), and the KV cache defaults to fp32.
- **Concession**: KV cache and activations use 2x the memory compared to FP16/BF16.
- **Rationale**: Eliminates underflow/overflow risks during the GDN delta-rule recurrence ($S_t = S_{t-1} + \dots$) and norm scaling, ensuring identical output quality.
- **Update (2026-10-19)**: the KV-cache half of this concession is now opt-in — `--kv-f16` stores K and V as fp16 with the fp32 unpack inside the kernels (§9). The activations and the GDN's S/conv states stay fp32, and the default path is unchanged.

### 3.4. Threadgroup Memory Ceiling in `k_attn`
- **Current State**: In `k_attn`, attention scores `sc` are allocated in dynamic threadgroup memory (`d.tg_mem = (pos + 1) * sizeof(float)`).
- **Concession**: Apple Silicon GPUs cap `maxThreadgroupMemoryLength` at 32 KB (32,768 bytes), which limits context length to **8,192 tokens** before this kernel exceeds hardware threadgroup limits.
- **Rationale**: 8K is more than double the initial 4K default and allowed zero-overhead scratch sharing without extra buffer allocations.

---

## 4. What To Do Next (Recommendations for Qwen-3.8-Flash-Next)

If you have unlimited time, compute, and optimization resources, here is the prioritized roadmap to push this implementation to peak performance:

### Priority 1: Uncap Context Window to 32K–128K (Online Softmax / Device Scratch)
- **Problem**: `k_attn` threadgroup memory caps at 8,192 tokens (32 KB limit).
- **Solution**:
  - Replace the threadgroup score allocation with either:
    1. **Online Softmax** (FlashAttention-style): Maintain running max $m$, normalizer $l$, and accumulator in SIMD registers over chunks of $K$ and $V$. Requires $O(1)$ scratch memory!
    2. **Device Scratch Buffer**: Allocate a global scratch buffer `s.att_scores` of shape `[H, max_ctx]`. At 32K context and 24 heads, this consumes only **3.0 MB** of VRAM!
  - **Impact**: Enables 16K, 32K, and 64K context windows immediately on 24 GB Macs.

### Priority 2: Tiled GEMM & Associative Scan for Fast Prefill
- **Problem**: Prompt evaluation is currently single-token autoregressive.
- **Solution**:
  - Implement tiled GEMM kernels for prompt batches ($B \in [512, 1024, 2048]$).
  - Implement parallel associative scan for GDN linear attention:
    $$\Delta \text{-Net recurrence: } S_t = S_{t-1} (I - \beta_t k_t k_t^T) + \beta_t v_t k_t^T$$
    This can be evaluated in $O(\log T)$ parallel GPU dispatches via prefix scans rather than $O(T)$ sequential steps.
  - **Impact**: Boosts prompt prefill speed from ~20 tok/s to **1,500–3,000+ tok/s**.

### Priority 3: Speculative Decoding via MTP (Multi-Token Prediction) Layer
- **Problem**: Autoregressive decode is memory-bandwidth bound (~14 tok/s).
- **Solution**:
  - The model weights include `blk.40` (the MTP draft layer).
  - Implement MTP draft proposal generation: generate 2 draft tokens per step and verify them in a single batch-2 window.
  - **Impact**: Expected speedup of **1.4x–1.8x**, reaching **22–28 tok/s** sustained generation on M2.

### Priority 4: FP16 / BF16 KV-Cache Storage
- **Problem**: FP32 KV cache takes 40 KB / token.
- **Solution**:
  - Store `kc` and `vc` as FP16 (or BF16).
  - Unpack to FP32 during dot-product in `k_attn`.
  - **Impact**: Halves KV cache memory to **20 KB / token**. A 64K context window will take only **1.28 GB** instead of 2.56 GB!
- **Delivered (2026-10-19)**: `--kv-f16`, FP16 only, strictly opt-in — the mechanism, kernels, memory math and gates are §9. (BF16 remains a possible follow-up if a measured need appears.)

### Priority 5: SIMD-Group Matrix Operations (AMX / Apple Neural Accelerators)
- **Problem**: GEMV dequantization executes via scalar SIMD lanes.
- **Solution**:
  - Re-express the dequantization GEMV and GEMM using Metal shading language `simdgroup_matrix` primitives (available on Apple Silicon M-series).
  - **Impact**: Substantial increase in compute-bound kernel efficiency, particularly for shared expert and FFN down-projections.

---

## 5. Verification & Testing Reference

- **Compile Test**:
  ```bash
  /opt/homebrew/bin/cmake --build build-metal
  ./build-metal/q36_compile_test
  ```
- **CLI Generation Run**:
  ```bash
  ./build-metal/strata --native /Users/yebyen/w/Strata-data/models/qwen36/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf --tokens "1, 2, 3" --n 64 --ctx 4096
  ```
- **Web UI & Streaming Server**:
  ```bash
  ./run-q36.sh 8082
  ```

## 6. Long context (split-K attention)

`k_attn` keeps the attention scores in threadgroup memory, and Metal's threadgroup budget is 32 KB —
8,192 score cells — so `--ctx` stopped at 8K. The split-K kernels lift that ceiling: the context is now
limited by the KV cache and unified memory only. The default decode path is unchanged at 8,192 cells and
below.

- **Split-K attention** (`k_attn_sk_part` / `k_attn_sk_merge` in `metal/kernels/q36.metal`, ported from
  q36_engine.cu's `k_attn_part` / `k_attn_merge`): one partial per (query row, 256-cell chunk) in device
  scratch — the chunk's scores, running max `m`, sum `l` and `P·V` accumulator — and a merge kernel that
  folds the chunks (the online-softmax rescale) and applies the sigmoid gate. The math and its order
  match the CUDA engine's kernels. The decode step passes the head's query rows; scratch is bounded by
  `SK_ROWS_MAX = 256` rows per dispatch, sized once at startup from `--ctx`. The chunked-prefill path
  reuses the same kernels row-blocked (`row0` per dispatch).
- **Engagement**: automatic on the decode step as soon as `n_pos` passes 8,192 cells; at or below it the
  original `k_attn` runs unchanged. `STRATA_Q36M_ATTN_SK=1` forces split-K at any length — the parity
  knob against `k_attn` at short context — on the decode step, the MTP draft/verify paths, and (since
  the multi-row change below) the chunked prefill too, which previously forced split-K only above
  8,192. The knob-unset paths keep the released behavior exactly.
- **Multi-row blocks on the chunk path** (`STRATA_Q36M_ATTN_ROWS=B`, read once per process): the
  chunked prefill's split-K pair dispatches B token rows per part/merge pair instead of one — the same
  kernels, the same per-row arithmetic, bit-identical outputs (parity suite stage 13b, and the
  10-prompt ladders 10/10 with `--prefill 512`, unfolded and GQA-folded). The engine refuses B with
  `B * H` past `SK_ROWS_MAX` and prints the numbers (max B = 16 at H = 16). Measured on the M2
  (2026-10-20, the 000685 session, `--prefill 512`, split-K forced): one 512-token chunk's dispatches
  go from 11,210 to 1,610 — the attention bucket 10,250 → 650 (512 rows × 2 dispatches × 10 QSA layers
  + 10 prep → 32 blocks × 2 × 10 + 10), the MoE GEMV (749) and the rest (211) unchanged. The chunk
  cost itself barely moved at this depth (2,048-token read: TTFT 80.4 s → 79.4 s, 25.5 → 25.8 tok/s) —
  the dispatch overhead is a small term next to the KV reads; the counts, not this delta, are the
  mechanism evidence.
- **Dispatch counter** (`STRATA_Q36M_DISPATCH_LOG=1`): one stderr line per chunk —
  `q36 dispatches: pos0=… nt=… attn=… gemv=… other=… total=…` — counted at the single encode point
  (`Context::run_graph`), classified by kernel name (attn: every attention kernel; gemv: the GEMV and
  routed-expert family; other: norms, conv, GDN, the rest). Diagnostic only; unset, nothing is counted
  or printed.
- **Memory gate at startup**: before the per-layer caches allocate, the engine refuses a `--ctx` above
  8,192 (and `--prefill`/`--mtp` runs at any context) when its working-set estimate exceeds
  `recommended_working_set()`: everything allocated so far (the wrapped weights, the decode and chunk
  scratch) + the MTP draft block's cache when `--mtp` + a 0.25 GB reserve (`STRATA_Q36M_MEM_OK=1`
  overrides; a token-by-token run at 8,192 or below is never refused, so existing setups behave exactly
  as before). The per-layer KV and S/conv caches allocate after this gate, so the fp32 path's estimate
  has never counted the main KV term — that fp32 refusal behavior is frozen; with `--kv-f16` the
  estimate adds the (halved) main-KV projection it is about to allocate, so a flagged run's refusal
  counts the thing the flag halves (§9).
- The kernels are written for `SK_HD = 256` (the qwen35moe head_dim); the host refuses anything else.
  Program-scope constants in `q36.metal` are `enum`, not `constexpr` — MSL rejects the latter.
- **Chunked prompt prefill** (`--prefill N`; `--prefill auto` = 512, the parity-verified chunk — what setup
  writes for this family and what the CUDA engine's flag means by pick-for-me; `STRATA_Q36M_PREFILL=1` for
  512 without the flag): the chunk of T tokens runs as T-wide dispatches — embedding rows in one
  dispatch, dense projections with the tokens walked inside the threadgroup (`k_gemv_c` /
  `k_gemv_lpr8_c`: every weight row read once per chunk, each (token, output) summed in the decode
  kernels' exact order), routed experts grouped per expert (`k_tables` builds the entry tables in one
  dispatch; `k_expc` walks each expert's entries once; the lpr8-typed gate/up run the decode kernel per
  entry through `k_gather_t`), the conv in 16-token tiles and the delta rule sequential per token
  (`k_conv_chunk` / `k_gdn_chunk`, the CUDA engine's tile mode with `k_conv` / `k_gdn`'s math), and
  attention through the split-K kernels row by row-block. Every state the decode path keeps (recurrent
  S, conv history, KV cache) is updated by the same arithmetic in the same order, and the prompt's last
  token still runs as a decode window, which is where the logits come from. Prompts run token-by-token
  unless `--prefill` is set.  The serve loop reports `PP <read> <total> <ms> <tok/s>` once per chunk
  (every 512 tokens on the token-by-token path) — the server's prompt percent and its read watchdog run
  off those lines; this engine never sent them before, so the percent stayed hidden and a read longer
  than engine_silence_s (300 s default) could be killed as engine-silent.

Gates, measured on the M2 (macOS 26.6.2) with the model file from §5, on this exact build (2026-10-18):
the compile test creates all pipelines including the two split-K ones; forced split-K at 4K is 10/10
token-identical to the default `k_attn` path (128-token greedy; the prompts that hit a stop condition
agree on every token they share); `--ctx 32768` loads (READY 32768) and the 10 parity prompts generate
without crashes, decode 16.9–19.3 tok/s. Prompt processing at 32K stays sequential per token and costs
~1.1–2.3 s even for the ~30-token parity prompts (10.1 s on the first request after load) — the
chunked-prefill work targets exactly that.

```bash
# the reference run
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-base.json
# split-K forced at short context: tokens must equal the default path's
STRATA_Q36M_ATTN_SK=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-sk.json
# long context: 32K boots and generates
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 32768" --out /tmp/q36-32k.json
# chunked prefill: tokens must equal the token-by-token run's (the chunk kernels keep its exact order)
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --prefill 512" --out /tmp/q36-chunk.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-chunk.json
# long context with chunked prefill: a 2K-token prompt read, 128 generated, at 32K
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 32768 --prefill 512" --out /tmp/q36-32k.json
# the chunk-size benchmark (TTFT + prompt tok/s, sizes 2K/4K/8K x chunks 512/1024/2048 against the
# token-by-token baseline, medians of 3, warm-up discarded, seeds per cell so the prefix matcher
# never hides a read; prints the table and the >=5x gate verdict, writes every raw run to the JSON).
# The no-flag default flips only if every measured size clears 5x AND that chunk passed the 10-prompt
# parity gate above - until then prompts stay token-by-token without --prefill.  Full matrix is a
# couple of hours on the M2 (the token-by-token 8K rows alone run ~8 min each): cut --rounds or the
# size list for a first look.
python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 32768" --sizes 2048,4096,8192 --chunks 0,512,1024,2048 \
  --rounds 3 --gen 16 --out bench/results/2026-XX-XX-m2-q36-prefill/bench.json
# parity under the shipping default vs the token-by-token path (rerun after any default change:
# the no-flag run must give the --prefill 0 run's tokens, 10/10)
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 32768" --out /tmp/q36-default.json
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 32768 --prefill 0" --out /tmp/q36-tbt.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-tbt.json /tmp/q36-default.json
```

The chunk-size benchmark that decides the `--prefill` default (2026-10-18, the M2 (macOS 26.6.2) with the
model file from §5, `--ctx 32768`, greedy, 16 generated tokens per run, median of 3 rounds per cell, one
discarded warm-up per engine start, deterministic seeded prompt ids per size and round so the prefix
matcher never hides a read; `tools/qwen35/q36_prefill_bench.py`, the raw runs in
`bench/results/2026-10-18-m2-q36-prefill/`):

| Prompt tokens | --prefill | TTFT | prompt tok/s | vs token-by-token |
|---:|---:|---:|---:|---:|
| 2048 | 0 (token-by-token) | 139 s | 14.7 | 1.00x |
| 2048 | 512 | 92 s | 22.3 | 1.51x |
| 2048 | 1024 | 94 s | 21.8 | 1.48x |
| 2048 | 2048 | 94 s | 21.8 | 1.48x |
| 4096 | 0 (token-by-token) | 321 s | 12.8 | 1.00x |
| 4096 | 512 | 225 s | 18.2 | 1.43x |
| 4096 | 1024 | 233 s | 17.6 | 1.38x |
| 4096 | 2048 | 231 s | 17.7 | 1.39x |
| 8192 | 0 (token-by-token) | 823 s | 10.0 | 1.00x |
| 8192 | 512 | 647 s | 12.7 | 1.27x |
| 8192 | 1024 | 658 s | 12.5 | 1.25x |
| 8192 | 2048 | 663 s | 12.4 | 1.24x |

The flip gate is not met: the chunk path reads 1.24-1.51x the token-by-token rate — best at 2K, the gain
shrinking with depth as the attention's KV reads grow — nowhere near the 5x a default change needs, so the
no-flag default stays token-by-token (`default_prefill_chunk = 0` in `q36_metal.cpp`; `--prefill auto`
stays 512). This is the full-matrix version of T3's first points (docs/QWEN36_METAL_ROADMAP.md): at depth
both paths pay the same ~8x GQA KV re-read, and that term grows with position; a GQA-folded attention pass
(T6) is the lever that would move the chunked read and the deep decode together. 512 is the best measured
chunk on Metal; 1024 and 2048 read 1.5-3% slower at every size, so there is no bigger-chunk win here like
the CUDA engine's. What the benchmark does show in the chunk path's favor: a 2K prompt's TTFT drops from
139 s to 92 s and an 8K one from 823 s to 647 s — real, just not a default-changing factor of five.

Parity under the shipping default re-verified the same day on the same build: the no-flag run and the
`--prefill 0` run are 10/10 token-identical (`q36_solo_gate.py` at `--ctx 32768`, 128-token greedy,
`q36_cmp_gate.py`: `identical: 10 of 10`) — with the flip not taken, this is the check that the default
path (and the scaffolding commit that touched its arg parsing) changed no token.

## 7. MTP speculative decoding (--mtp N)

The GGUF carries its own MTP draft block, `blk.40` (`qwen35moe.nextn_predict_layers`): a
full-attention layer + MoE plus `nextn.eh_proj` / `enorm` / `hnorm` / `shared_head_norm`, which
proposes the next token from the main model's last hidden state. `--mtp N` (an integer number of
drafts; 0 or absent = off, the default; Flash-Next's `--mtp <dir>` means nothing for this family)
turns it on for greedy requests.  The design mirrors the CUDA engine's `spec_step`
(`src/qwen35/q36_engine.cu`) dispatch for dispatch:

- **The draft call** re-runs the last window's accepted drafts through the block at their
  positions (each seeded by that window's main-model hidden row) and then the row of the next
  decided token (seeded by the last kept row's hidden); the chained drafts seed from the block's
  own output.  The block's row is the decode path's exact QSA dispatch sequence against the
  block's own KV cache.
- **The verify window** runs `[x, d1..dD]` (D+1 rows) through the main model as one chunk pass
  (the T2 chunk kernels — bitwise the decode path), with the GDN/conv dispatches on
  window-scratch states **copied from the real S/conv first** (the checkpoint: the real states
  stay at the accept point whatever the window accepts), then a commit that re-runs the kept
  rows' conv history and recurrence into the real states over the window's saved inputs.
- **Kept = emitted**: a window cut short by EOS, cancel, `max_new` or the context end commits
  only its shown prefix, so the state ends where the shown tokens end.  The engine prints the
  offered/accepted draft counts on stderr; the serve `DONE` line is unchanged.

Greedy only: the window's picks are argmaxes, so the kept text is the plain greedy text whatever
the drafts proposed — with the parity gate green, `--mtp 2`'s tokens are the `--mtp 0` tokens.
No new Metal kernels: `eh_proj` runs the type-generic GEMV (the dims travel in the dispatch
params), the two RMSNorm halves land in one 2N buffer via arg offsets, the checkpoint copies are
`k_gather_t` flat copies (K=1), and the picks are host argmaxes matching `sample_next`'s tie
rule.  The block weights, its cache and the window scratch load only when `--mtp > 0`, and their
bytes join the startup working-set refusal.

Gates (same Mac, same day, same build):

```bash
# the base run and the MTP run over the 10 parity prompts (serve protocol, greedy, 128 tokens)
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-base.json
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --mtp 2" --out /tmp/q36-mtp.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-mtp.json   # 10/10 identical, twice
# the decode ratio: the two runs' per-prompt tok/s (the solo_gate lines / the DONE decode_ms),
# plus the accept rate from the engine's stderr MTP line
```

Measured on the M2 (Apple M2, 2026-10-18, UD-Q2_K_XL):
- **Parity**: 10/10 token-identical twice (`q36_cmp_gate.py` pass 1 and pass 2 vs `/tmp/q36-default.json`: `identical: 10 of 10`). `q36_chunk_parity` remains all-OK (38/38 bit-identical).
- **Speed**: decode tok/s across the 10 prompts measured at 5.6–11.7 tok/s (base without MTP: ~17.5–21.4 tok/s); draft accept rate measured at ~15–25%.
- **Verdict**: The target $\ge 1.2\times$ speedup is not met on Apple Silicon unified memory without a draft-vocab subset head (CUDA's `STRATA_Q36_DRAFT_VOCAB`). Per the ticket scope, the flag remains strictly opt-in (`--mtp 0` / flag-absent default); no default flip is taken.

### The draft-vocab subset head (`--draft-vocab N`, ticket 000682)

T4's verdict named the missing piece, and this is it. Without it, every draft step's head GEMV
reads the whole `output.weight` — all 248,320 rows — and the host argmax scans all 248,320 logits;
`--mtp 2` runs two such passes per round, and on the M2's 100 GB/s bus that is the draft round's
largest term. `--draft-vocab N` (an integer; the env `STRATA_Q36_DRAFT_VOCAB` is the fallback, the
flag wins — the CUDA engine, `src/qwen35/q36_engine.cu`, takes the same flag and env) restricts the
draft pick to the head's **first N rows**, read in place: a shrunken view of the same tensor (same
buffer, same offset, same row bytes, `n_out = N`), so no new kernel, no gather and no id map — row
*i* is token id *i*. The lowest ids are the most frequent ones (the CUDA engine's own rationale for
the same knob); a token past row N is a rejected draft and nothing else: drafts only propose, the
verify window's full-head pick decides, and the kept text is the same greedy text whatever N is.
0 or an absent flag = the whole vocabulary, as before; a value at or past the vocabulary falls back
to the full head with a stderr note, and `--draft-vocab` without `--mtp N` notes that it does
nothing. The engine prints the subset's rows and MB per draft step beside the full head's at
startup. Operators turn it on from the config's `args`: `"--draft-vocab", "8192"` beside
`"--mtp", "2"`.

Per draft step the subset reads `N × row_bytes(head)` instead of `248,320 × row_bytes(head)`. Measured
on the M2 at startup (Apple M2, 2026-10-19, build `acd8c8a0`, UD-Q2_K_XL): the head is **Q4_K** there,
1,152 B/row, so the full head streams 286 MB per draft step and

    strata metal q36: --draft-vocab 8192: the draft pick reads the head's first 8192 rows (9.4 MB per draft step against the full head's 286 MB)

— 8,192 rows = 9.4 MB (30.4×), 4,096 ≈ 4.7 MB (61×), 16,384 ≈ 18.9 MB (15×). (UD-Q4_K_XL's head is
Q8_0 at 2,304 B/row per docs/QWEN36.md's rocprof note — the line reads the file's own bytes, whatever
they are.) The verify window's own full-head pass stays and becomes the round's remaining head term.

Gates (same Mac, same day, same build):

```bash
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --mtp 2 --draft-vocab 8192" --out /tmp/q36-dv.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-dv.json   # 10/10 identical, twice
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-mtp.json  /tmp/q36-dv.json   # 10/10 identical
# the measurement: the same ladder at --draft-vocab {4096, 8192, 16384}; per-prompt tok/s and the
# engine's accept-rate stderr line; tools/qwen35/q36_multilingual_prompts.py prints each prompt's
# share of token ids above each draft-vocab limit (the coverage predictor per language)
```

M2 results (Apple M2, 2026-10-19, build `acd8c8a0`, UD-Q2_K_XL):

- **Build**: `cmake --build build-metal` and `./build-metal/q36_compile_test` clean; `q36_chunk_parity` 38/38 bit-identical.
- **Parity**: `--mtp 2 --draft-vocab N` is token-identical to `--mtp 0` (**10 of 10**) across all tested values:
  - $N=4,096$: `identical: 10 of 10`
  - $N=8,192$: `identical: 10 of 10` (and `10 of 10` vs full-head `--mtp 2`)
  - $N=16,384$: `identical: 10 of 10`
  - Composed with folded attention (`STRATA_Q36M_ATTN_GQA=1 --mtp 2 --draft-vocab 8192`): `identical: 10 of 10` vs `--mtp 0`.
  - Baseline no-flag compare: `identical: 10 of 10` vs pre-patch base.
- **Traffic Reduction**:
  - $N=4,096$: 4.7 MB per draft step against the full head's 286 MB (**60.6× cut**).
  - $N=8,192$: 9.4 MB per draft step against the full head's 286 MB (**30.4× cut**).
  - $N=16,384$: 18.9 MB per draft step against the full head's 286 MB (**15.1× cut**).
- **Speed & Acceptance**:
  - $N=4,096$: 5.0–8.8 tok/s across prompts.
  - $N=8,192$: 5.0–8.7 tok/s across prompts.
  - $N=16,384$: 5.5–11.1 tok/s across prompts (draft accept rate improves with 16k coverage, reaching 11.1 tok/s on code).
  - Baseline greedy decode without MTP (`--mtp 0`): ~17.5–21.4 tok/s.
- **Verdict**: the ≥1.2× threshold is **not met** on the M2. Even with the draft head's memory traffic reduced by up to 60.6×, the draft head is no longer the bottleneck — the 3-token chunk verify window through all 40 layers of the main model dominates when draft acceptance sits at ~15–25%. The bandwidth cut is real and output parity is exact across all prompts (English, code, and CJK), but MTP remains slower than pure baseline decode on Apple Silicon unified memory. Both flags remain strictly opt-in (`draft_n = 0` and `--mtp 0` defaults untouched).

The CUDA engine's own data point for the same knob (docs/QWEN36.md, "Tried and dropped") reads the
same way from the other side: there the head pass was small enough that the subset moved tok/s by
single-digit percents and cost 10–15% on other scripts — on the M2 the head pass was large, but the
round's binding term was the window all along.

## 8. GQA-folded attention (`STRATA_Q36M_ATTN_GQA=1`)

The model is grouped-query: 16 query heads share 2 KV heads (H/HKV = 8, read from the geometry,
never assumed), and every query head of a group reads the same K/V cells — an ~8x KV-traffic
multiplier at depth that L1/L2 are not catching (§6's chunk table shows the term growing with
position; it is why chunked prefill cannot win at depth). The folded kernels give one threadgroup
the whole KV-head group and compute its heads' scores against each K/V cell once. Opt-in behind
the parity knob `STRATA_Q36M_ATTN_GQA=1` (the `STRATA_Q36M_ATTN_SK` pattern); the default path is
unchanged until a measurement says otherwise.

- **Path A — the `k_attn` range (n_pos <= 8,192 cells)**: `k_attn_gqa_part` (grid (HKV,
  ceil(n_pos/128)) — one threadgroup per (KV head, 128-cell tile), one cell per thread; the K row
  read once, all HG dots of the group chaining against it) and `k_attn_gqa_merge` (grid (HKV, HG) —
  one threadgroup per (KV head, query head): k_attn's own one-pass sequence for that head over the
  staged scores — max, exp into the tile, `block_sum`, the P·V chains per output dim, the V cells
  read once per head exactly as `k_attn` itself reads them).  Scores travel through a device scratch
  (`[head][cell]`, H x ctx floats, allocated only when the knob is on) because HG x n_pos scores
  cannot fit the 32 KB threadgroup ceiling.  (An intermediate revision looped the heads inside one
  (HKV)-grid threadgroup — bit-identical output, but the per-head V re-walks streamed the whole
  slice through a too-small L1 once per head and the dispatch ran on 2 threadgroups: measured deep
  decode collapsed to 2.7-3.2 tok/s vs the 8 tok/s baseline at pos ~4,100.  Hoisting the head loop
  into the grid is the fix; bit-identity re-proven on the host sim.)
- **Path B — the split-K range (n_pos > 8,192, or `STRATA_Q36M_ATTN_SK=1`)**:
  `k_attn_sk_part_gqa` (one threadgroup per (256-cell chunk, KV head) covers the group's HG rows —
  `k_attn_sk_part`'s body per head) writes the same (row, chunk) partial slots, so
  **`k_attn_sk_merge` runs unchanged**.
- **Bit-parity contract** (the design target, not a tolerated approximation): per (head, cell) the
  score is the same fp chain (i ascending, one accumulator, then *scale); the max is over the same
  values (exact, any tree); exp and the per-thread partial sums keep `k_attn`'s thread<->cell map
  and reduction trees; P·V is one fp chain per (head, output dim) walking p ascending, the e and v
  operands shared but never re-rounded; path A keeps `k_attn`'s `(acc * (1/sum)) * sigm(gate)`
  ending, path B keeps the merge's. Exact FLOPs: no score or exp recompute — the only new traffic
  is the score-scratch round trip (~256x smaller than the KV term it replaces). The safe-math
  build (no reassociation) is what makes the chain-by-chain argument hold. A host-side simulation
  of both kernel structures (transcribed loop for loop, sharing only the reduction primitives)
  shows every partial and output bit-identical to the unfolded kernels across decode and
  chunk-row shapes, n_pos 1..8,192 and the split-K range (spec task 000681's
  `gqa_parity_sim.cpp`; 17 shapes, all bit-identical).
- **Engagement**: the fold replaces whichever kernels the row's length selects — the `k_attn` pair
  below the cap, the split-K part above it (the merge is common). `push_qsa_attention` reads the
  knob at the point of use (the `STRATA_Q36M_TRACE` pattern), so the decode step, the chunk path's
  rows and the MTP draft rows/verify windows (§7) all fold through the one shared helper; with
  `STRATA_Q36M_ATTN_SK=1` too, path B runs at every length.
- **Guards at startup (knob-gated, refused with the numbers, no fallback)**: H/HKV < 2 (the fold
  is a no-op) or threadgroup staging past the 32 KB ceiling — the folded split-K part takes
  HG x (256 + 8 + 256) floats (16,640 B at HG = 8; HG <= 15 at head_dim 256), the folded `k_attn`
  part HG x head_dim floats.

Gates (same Mac, same day, same build):

```bash
# gate 1: the compile test creates all pipelines, the three folded ones included
cmake --build build-metal && ./build-metal/q36_compile_test
# gate 2: the fold forced at 4K is 10/10 token-identical to the default path
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-base.json
STRATA_Q36M_ATTN_GQA=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-gqa.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-gqa.json
# cross-checks, each 10/10: split-K forced, with and without the fold; --mtp 2, with and without it
STRATA_Q36M_ATTN_SK=1 STRATA_Q36M_ATTN_GQA=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-sk-gqa.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-sk.json /tmp/q36-sk-gqa.json
STRATA_Q36M_ATTN_GQA=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --mtp 2" --out /tmp/q36-mtp-gqa.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-mtp.json /tmp/q36-mtp-gqa.json
# gate 3: the chunk ladder with the fold on (twice), and q36_chunk_parity
STRATA_Q36M_ATTN_GQA=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --prefill 512" --out /tmp/q36-gqa-chunk.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-gqa.json /tmp/q36-gqa-chunk.json
# gate 4: the deep-context delta - 4,098-token read at 16K and decode tok/s at pos ~4,100,
# fold off vs on (the tool appends --prefill itself; the knob travels through the environment)
python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 16384" --sizes 2048,4098 --chunks 0,512 --rounds 3 --gen 128 \
  --out bench/results/<date>-m2-q36-gqa/bench-off.json
STRATA_Q36M_ATTN_GQA=1 python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 16384" --sizes 2048,4098 --chunks 0,512 --rounds 3 --gen 128 \
  --out bench/results/<date>-m2-q36-gqa/bench-on.json
```

Measured on the M2: **pending** — the numbers (the 4,098-token read at 16K, fold off vs on; decode
tok/s at pos ~4,100; the same-day gate 2/3 results) land here when the gates run on the M2, in this
section's format: what, hardware, date. Until then the knob stays opt-in and the default path is
byte-identical (the fold-off base json vs the pre-change build's base json, 10/10, is part of gate 2).

## 9. FP16 KV cache (`--kv-f16`)

The QSA layers' KV cache (§6's geometry: 10 full-attention layers, 16 query heads sharing 2 KV heads,
head_dim 256) stores every past token's K and V in fp32 — 10 layers × 2 (K,V) × 2 heads × 256 dims ×
4 B = **40,960 B ≈ 40 KB per generated token**, resident for the whole run in unified memory the
weights and the expert cache also need (0.34 GB at an 8K context, 1.34 GB at 32K, 2.68 GB at 64K).
`--kv-f16` is the opt-in flag that halves it: K and V store as fp16 (`half`), and every kernel load
unpacks to fp32 at the load site — every dot product, softmax and P·V chain keeps its exact fp32
arithmetic. **20,480 B/token** over the QSA layers (0.17 GB at 8K, 0.67 GB at 32K, 1.34 GB at 64K).
The MTP draft block's own cache (`s.mtp_kc`/`s.mtp_vc`, one full-attention layer) halves with it.
FP16, not BF16: MSL has a native `half`; BF16 would need manual bit-twiddling in every kernel for a
range win this model does not need (K and V are norm-scaled, |x| ≈ O(1)).

**Mechanism — one extra runtime compile, selected pipelines.** MSL cannot switch a buffer's pointee
type at runtime, so `metal/kernels/q36.metal` compiles twice: the default lib exactly as before, and
— only when the flag is passed — a second lib built with `#define STRATA_KV_F16 1` through the
prelude hole `mtl::Context::compile` already has (`Pipelines::init_kv_f16`, called from `run()` right
after `pipes.init(cx)`). The flagged lib's seven KV kernels take `device (const) KV_T*` caches
(`KV_T` = `half` under the define, `float` otherwise — an identity cast in the default build); the
engine selects them at the dispatch sites only when the flag is set. Buffer binding is positional
into typeless buffers, so no argument list moves. The default path's source diff is casts-only and
its startup does nothing extra — the second compile (~1–2 s) is flagged-runs-only.

| Kernel | K/V access | f16 change |
|---|---|---|
| `k_attn_prep` | writes `kc`, `vc` (the RoPE'd K row, the V row) | `(KV_T)v` at the two stores |
| `k_attn_prep_chunk` | writes both, T rows at `pos0 + t` | `(KV_T)v` at the two stores |
| `k_attn` | reads both (score loop, P·V loop) | `float(kp[i])` / `float(vc[...])` at the loads |
| `k_attn_sk_part` | reads both | same at the loads |
| `k_attn_gqa_part` | reads `kc` (scores to the scratch) | same at the loads |
| `k_attn_gqa_merge` | reads `vc` | same at the loads |
| `k_attn_sk_part_gqa` | reads both | same at the loads |

`k_attn_sk_merge` stays on the default lib in every mode: it reads the float partials, not K/V. The
same selective compile is what keeps a half cache from ever meeting an fp32 reader or vice versa —
nothing in the dispatch plumbing could catch that mix, the gates below are the catch.

**Memory.** The allocations halve with the flag (2 B/elem instead of 4 — the host never reads or
writes KV values, the memsets only; zero bytes are zero half):

| Context | fp32 (default) | fp16 (`--kv-f16`) | draft block fp32 → fp16 |
|---|---|---|---|
| 8,192 | 0.34 GB (335,544,320 B) | 0.17 GB (167,772,160 B) | 0.03 GB → 0.02 GB |
| 32,768 | 1.34 GB (1,342,177,280 B) | 0.67 GB (671,088,640 B) | 0.13 GB → 0.07 GB |
| 65,536 | 2.68 GB (2,684,354,560 B) | 1.34 GB (1,342,177,280 B) | 0.27 GB → 0.13 GB |

The engine prints the real numbers once the caches are up, e.g. at `--ctx 32768 --kv-f16`:

```
strata metal q36: --kv-f16: KV caches fp16: 10 QSA layers x 2 (K,V) x 2 KV heads x 32768 ctx x 256 dims x 2 B = 0.67 GB (fp32: 1.34 GB)
```

(with `--mtp` the line continues `; draft block 0.07 GB (fp32: 0.13 GB)`). Measured on the M2
(2026-10-19, branch build `5bcf2485`): both gate-5 lines matched the table exactly — `--ctx 8192`:
`... x 8192 ctx x 256 dims x 2 B = 0.17 GB (fp32: 0.34 GB)`; `--ctx 32768`: `... x 32768 ctx x
256 dims x 2 B = 0.67 GB (fp32: 1.34 GB)`.

The startup refusal's estimate changes only under the flag: `--kv-f16` adds the main-KV projection
it is about to allocate to `est` (the fp32 path's estimate has never counted the per-layer KV — they
allocate after the gate — and that fp32 behavior is frozen; §6's memory-gate bullet has the reading).

**Text difference (expected, documented — not a bug).** The fp16 cache rounds K and V once at write
(round-to-nearest-even); runs with the flag are deterministic (two flagged runs are 10/10 identical
to each other) but differ from the fp32 runs' text. Measured on the M2 (2026-10-19, the 10 parity
prompts, ≤128 generated tokens each): **7 of 10 prompts are token-identical across the whole run**;
the three that diverge do so at generated token 100 (p00), 69 (p01) and 46 (p03) — never at the
first tokens (the garbage-class signature of a half cache meeting an fp32 pipeline), and every
prompt is identical up to its divergence point. The parity harness's f16 mode
(`STRATA_Q36M_KV_F16=1 ./build-metal/q36_chunk_parity`) measures the rounding's own cost directly:
on a cache of half-exact values both libs' `k_attn` and split-K pair produce bit-identical outputs
(the "only storage changed" proof), and on a random cache the rel-L1 against the fp32 kernels is
printed and bounded at 1e-3 (the CUDA `qsa_parity.cpp` precedent's metric and phase bound for FP16
paths). Measured (M2, 2026-10-19): the half-exact cross is bit-identical on both kernel pairs, and
the random-cache rel-L1 is **1.75e-4** vs the 1e-3 bound (`kv f16 cross   OK (random-cache rel-L1
0.000175 <= 1e-03)`).

**Interactions.** Works with `--prefill` (the chunk writer `k_attn_prep_chunk` has an f16 pipeline;
the MTP verify window rides it), with `--mtp N` (the draft cache halves), and with either attention
knob (`STRATA_Q36M_ATTN_SK=1`, `STRATA_Q36M_ATTN_GQA=1` — all four reader variants have f16
pipelines). Never mix: a half cache with an fp32 pipeline (or the reverse) reads garbage — the
cross-checks below exist to catch a missed selection site.

Gates (same Mac, same day, same build):

```bash
# gate 1: the compile test also compiles the STRATA_KV_F16 lib and lists its seven pipelines
cmake --build build-metal && ./build-metal/q36_compile_test
# gate 2: parity both modes - default 38/38, f16 mode all-OK incl. the kv f16 cross stage
./build-metal/q36_chunk_parity
STRATA_Q36M_KV_F16=1 ./build-metal/q36_chunk_parity
# gate 3: the default path byte-identical - the no-flag base json vs the pre-change build's
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-base2.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-base2.json      # 10/10 required
# gate 4: the f16 ladder is deterministic - two flagged runs 10/10 identical to each other; the
# fp32-vs-f16 cmp's first-divergence positions are the documented text difference (recorded above)
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --kv-f16" --out /tmp/q36-f16a.json
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --kv-f16" --out /tmp/q36-f16b.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-f16a.json /tmp/q36-f16b.json       # 10/10
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base2.json /tmp/q36-f16a.json      # divergence positions
# gate 5: the KV line at both contexts (the memory table above reads off these)
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 32768 --kv-f16" --out /tmp/q36-32kf16.json
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 8192 --kv-f16" --out /tmp/q36-8kf16.json
# gate 6: cross-checks under the flag, each twice and 10/10 self-identical (a missed selection site
# - a half cache meeting an fp32 pipeline - reads garbage; these catch it)
STRATA_Q36M_ATTN_SK=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --kv-f16" --out /tmp/q36-skf16a.json    # ...then run b, cmp 10/10
STRATA_Q36M_ATTN_GQA=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --kv-f16" --out /tmp/q36-gqaf16a.json   # ...then run b, cmp 10/10
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --mtp 2 --kv-f16" --out /tmp/q36-mtpf16a.json   # ...then b, cmp 10/10
```

Measured on the M2 (Apple M2, macOS 26.6.2, 2026-10-19, branch build `5bcf2485`, UD-Q2_K_XL — every
gate from this list, same build, same day):

- **gate 1**: green — the f16 lib (index 1) compiles and its seven pipelines create (`all q36 f16
  pipelines created`), on top of the default lib's full list.
- **gate 2**: default mode all stages bit-identical (the 38-check suite); f16 mode all-OK plus the
  cross stage (rel-L1 1.75e-4, above).
- **gate 3**: the no-flag base json vs the pre-change build's base json — `identical: 10 of 10`.
  This is the byte-identical proof the contributing rule asks for.
- **gate 4**: two flagged runs `identical: 10 of 10` (deterministic); the fp32-vs-f16 cmp is 7 of 10
  with the divergence positions above.
- **gate 5**: the two KV lines match the memory table exactly (0.17/0.67 GB f16, fp32 would-be
  0.34/1.34 GB).
- **gate 6**: split-K-forced, GQA-fold and `--mtp 2` under the flag — each pair twice 10/10
  self-identical, and each 10/10 token-identical to the plain flagged run (the knob paths and the
  speculative path agree with the default attention under the flag). The composition
  `--mtp 2 --kv-f16 --draft-vocab 8192` twice 10/10 self-identical and 10/10 vs the plain flagged
  run — the draft-vocab head slicing (§7) and the halved caches compose as the review predicted.

**Speed (ready to run, not run yet).** The halved KV traffic should show up as a deep-decode
tok/s gain; the A/B is documented here and deliberately not a gate (the hours-class benchmark
matrix waits for the dimensional-improvements showcase):

```bash
python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 16384" --sizes 2048,4098 --chunks 0,512 --rounds 3 --gen 128 \
  --out bench/results/<date>-m2-q36-kvf16/bench-off.json
python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 16384 --kv-f16" --sizes 2048,4098 --chunks 0,512 --rounds 3 --gen 128 \
  --out bench/results/<date>-m2-q36-kvf16/bench-on.json
```

## 10. The chunk's parallel GDN scan and matrix-unit GEMMs (`STRATA_Q36M_GDN_WY=1`, `STRATA_Q36M_DENSE_MMA=1`)

§6's chunk path batches everything except two components, both kept token-sequential on purpose by
the T2 bit-parity contract: the GDN recurrence (`k_gdn_chunk`: one dispatch, 48 × 128 threads, each
thread owning one state row and walking all T tokens serially — two threadgroup barriers and ~384
MACs per thread per token) and the dense chunk GEMVs (`k_gemv_c`: every weight row re-dequantized
once per token row, so the dequant arithmetic — not the MACs — scales with T). §4's Priority 2 asked
for exactly these two levers ("tiled GEMM kernels for prompt batches" and a "parallel associative
scan" for the delta-rule recurrence). Both now exist as opt-in env knobs on top of the chunk path,
ported from the mac-kern chain's proven donors (`gdn_prompt.metal`'s chunked form, `dense_mm.metal`'s
fp16-then-matmul2d pattern) — with the parity contract deliberately relaxed from bit-identity to
measured error bounds, the same trade the CUDA engine already ships for its chunked prefill
(docs/QWEN36.md's KL/top-1 methodology). The knobs are read once per process, so the MTP verify
window and its commit re-run can never take different paths.

**`STRATA_Q36M_GDN_WY=1` — the WY/UT chunked scan.** Four kernels in `q36.metal`. `k_gdn_prep_c`
computes the q/k RMS norms and gates per (token, head) in parallel — the same `block_sum` over the
same 128 threads and the same formulas as `k_gdn_chunk`'s inline arithmetic, bit-identical by
construction, with the 1/√D query scale folded into `qn` (the decode kernel's `sh_q` convention).
Then the WY form over 32-token chunks (`GDN_GC`): with G_t the chunk's gate prefix product (log
space), (I + A)·D = B with A_ts = β_t (k_t·k_s)·G_t/G_s (s < t) solved by forward substitution per
column gives U (drives the state) and W (state coefficients); the inter-chunk walk applies
delta = U − W·S0 per chunk and updates S_L = G_L·S0 + Σ_s (G_L/G_s)·δ_s k_sᵀ; the intra-chunk output
adds Σ_{s≤t} (q_t·k_s)·(G_t/G_s)·δ_s. This is the standard parallel-scan decomposition for gated
linear attention: everything inside a 32-token block is parallel, and the sequential part is the
inter-chunk walk — T/32 chunk-steps instead of T token-steps, three dispatches instead of one. The
sequential `k_gdn_chunk` stays in the tree and runs whenever the knob is off — the permanent fallback.

**`STRATA_Q36M_DENSE_MMA=1` — dequant-once, matrix-unit GEMMs.** `metal/kernels/q36_dense.metal`
(copied from the mac-kern chain's `dense_mm.metal`; three more `dq_f16` types — Q2_K, IQ2_XS,
IQ3_XXS — through the generic `DW<F>` path): per chunk, each eligible dense matrix dequantizes to
fp16 once (`dq_f16_<type>`), the chunk's X rows stage to half (`act_f16`), and the T × n_out × n_in
product runs on MPP `matmul2d` 64 × 64 tiles with fp32 output (`gemm_f16_64_64_4`). Eligible: the
large per-layer matrices (attn_qkv, attn_gate, ssm_out, the QSA q/k/v/attn_output, the shared
expert's gate/up/down); the router, alpha, beta and the routed experts keep the parity GEMVs (tiny or
parity-owned; the experts are the next lever if the numbers ask for it). Matrices whose type has no
dq pipeline fall back to `k_gemv_c` per matrix, so a mixed-quant model engages where it can. The
fp16 operand rounding costs ~5e-4 relative — hence this knob's 1e-3 envelope against the 1e-5-class
bounds the reassociation-only GDN stages get. One fp16 weight arena (40 MB at the qkv shape) plus
the WY scratch (~59 MB at chunk 512, ~236 MB at 2048) allocate inside the chunk-buffer block, before
the §6 working-set refusal counts them.

Gates (Apple M2, macOS 26.6.2, UD-Q2_K_XL, 2026-10-18/19):

```bash
cmake --build build-metal && ./build-metal/q36_compile_test     # the WY + MMA lib pipelines in the list
./build-metal/q36_chunk_parity                                   # all stages OK incl. wy * and mma * (rel bounds)
./build-metal/q36_chunk_parity --bench                           # the per-part kernel table
# byte-identity: the no-flag base json vs the pre-change build (10/10), then per-knob ladders:
STRATA_Q36M_GDN_WY=1 python3 tools/qwen35/q36_solo_gate.py ...   # twice, 10/10 vs the sequential path
STRATA_Q36M_DENSE_MMA=1 python3 tools/qwen35/q36_solo_gate.py ... # twice, 10/10
STRATA_Q36M_GDN_WY=1 STRATA_Q36M_DENSE_MMA=1 ... --mtp 2 ...     # 10/10 vs --mtp 0
```

Measured (M2, 2026-10-18, build `4836ba99` — the WY slice's full gate list, all green): parity all
stages OK — `wy out` rel 2.51e-07, `wy S` rel 1.21e-07, the two-chunk T=40 stage 2.99e-07/1.26e-07
(the 1e-5 bound holds with ~40x margin), prep v/beta bit-exact; byte-identity 10/10; the WY ladder
10/10 twice; `--mtp 2` with the knob 10/10 vs base. Kernel bench (median of 5 after warm-up):

| GDN stage | T = 512 | T = 2048 |
|---|---:|---:|
| `k_gdn_chunk` (sequential) | 167.730 ms | 670.919 ms |
| prep + cprep + cstate + cout | 0.834 + 2.172 + 6.742 + 1.729 = **11.476 ms** | 3.580 + 8.680 + 26.007 + 6.674 = **44.942 ms** |
| | **14.6x** | **14.9x** |

cstate is the largest WY part at both sizes (~59%) — the GC=64 / segmented-combine follow-up lever if
it ever matters. Reported from the same machine's live-session round (build `3dca4b47`, the dense
slice included): the dense rows ran 110.2 → 7.2 ms per chunk at the attn_qkv shape (**15.3x**) —
combined with the GDN scan, the two kernel classes dropped from ~280 ms to ~18.7 ms per 512-token
chunk. Short-chunk TTFT with the WY knob measured −40-44% (p01 1917→1090 ms, p02 1952→1118 ms,
p08 2247→1258 ms).

**Ad-hoc live validation in place of the TTFT matrix (deliberate, the §9 pattern).** The hours-class
`q36_prefill_bench.py` matrix was skipped this round in favor of a live server session with every
lever on (`--ctx 16384 --prefill auto --kv-f16`, `STRATA_METAL_STREAM=1`, `STRATA_Q36M_PREFILL=1`,
`STRATA_Q36M_ATTN_GQA=1`, `STRATA_Q36M_GDN_WY=1`, `STRATA_Q36M_DENSE_MMA=1`, ~11.8 GB resident):
six requests, 1,051 cumulative tokens, decode 12.2-16.3 tok/s client-side and 57-66 ms per step
throughout — including at depth, where the pre-ticket build collapsed to 2.7-3.2 tok/s. The
attribution: the depth stability is §8's fold and §9's halved caches (both merged); this ticket's
contribution is the prefill kernel paths above. **The default path does not flip**: the no-flag
engine is byte-identical to the release and the knobs stay opt-in — the measured ≥5x-on-every-size
bar a default change needs (§6) has not been run since the levers landed, and the ad-hoc TTFT
deltas, while large on short chunks, are not that matrix. The remaining pre-merge gates are the
quick ones on the current tip: the compile test, the six `mma <type>` parity stages, the MMA-knob
ladder and the both-knobs `--mtp 2` ladder.
