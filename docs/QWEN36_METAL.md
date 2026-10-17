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
- **Multi-Turn Prefix Caching**: Matches `live_tokens` prefix across requests; preserves recurrent states $S$, Conv1D history, and KV-cache without full recomputation when conversational context prefixes match.
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
- **Current State**: All intermediate activation vectors (`x`, `h`, `y`, `qkv`, `z`, `S`) and the KV cache are allocated in 32-bit floating point (`float`).
- **Concession**: KV cache and activations use 2x the memory compared to FP16/BF16.
- **Rationale**: Eliminates underflow/overflow risks during the GDN delta-rule recurrence ($S_t = S_{t-1} + \dots$) and norm scaling, ensuring identical output quality.

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

## 6. Long context, chunked prefill (in progress on this branch)

Two additions land on top of the sections above (commit 65037a91 onward). Both leave the default paths
unchanged: attention stays in the threadgroup `k_attn` at 8,192 cells and below, and prompts run
token-by-token unless `--prefill` is set.

- **Split-K attention** (q36_engine.cu's `k_attn_part` / `k_attn_merge` on Metal): one partial per (query
  row, 256-cell chunk) in device scratch, a merge kernel folds the chunks and applies the gate. Engages
  automatically above 8,192 cells, or at any length with `STRATA_Q36M_ATTN_SK=1` (the parity knob against
  `k_attn`). Context is now limited by the KV cache, and the engine refuses a `--ctx` whose weights + KV +
  state + scratch do not fit the recommended working set, with the numbers (`STRATA_Q36M_MEM_OK=1`
  overrides; 8,192 and below is never refused).
- **Chunked prompt prefill** (`--prefill N`, or `STRATA_Q36M_PREFILL=1` for 512): the chunk of T tokens
  runs as T-wide dispatches — embedding rows in one dispatch, dense projections with the tokens walked
  inside the threadgroup (`k_gemv_c` / `k_gemv_lpr8_c`: every weight row read once per chunk, each
  (token, output) summed in the decode kernels' exact order), routed experts grouped per expert
  (`k_hist` / `k_exp_off` / `k_place` entry tables on device atomics, `k_expc` walking each expert's
  entries once), the conv in 16-token tiles and the delta rule sequential per token (`k_conv_chunk` /
  `k_gdn_chunk`, the CUDA engine's tile mode with `k_conv` / `k_gdn`'s math), and attention through the
  split-K kernels row by row-block. Every state the decode path keeps (recurrent S, conv history, KV
  cache) is updated by the same arithmetic in the same order, and the prompt's last token still runs as a
  decode window, which is where the logits come from.

Gates (the tools/qwen35 gates speak the serve protocol and drive this engine directly):

```bash
# the reference run (the build before these commits, if it is still around: same commands without them)
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-base.json
# split-K forced at short context: tokens must equal the default path's
STRATA_Q36M_ATTN_SK=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-sk.json
# chunked prefill: tokens must equal the token-by-token run's (the chunk kernels keep its exact order)
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096 --prefill 512" --out /tmp/q36-chunk.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-chunk.json
# long context: a 2K-token prompt read, 128 generated, at 16K and 32K
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 32768 --prefill 512" --out /tmp/q36-32k.json
```
