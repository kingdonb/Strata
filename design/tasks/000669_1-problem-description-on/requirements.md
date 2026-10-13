# Requirements: Fix Metal Prompt Arena Fallback on Small-Memory Macs

## Background

On Apple Silicon Macs with tight unified memory (the reported case: M2, 24 GB, macOS 26), Strata in streaming mode
(`--mmap-experts`, `STRATA_METAL_STREAM=1`) fails every prompt with:

```
metal prompt: no arena region for the prompt buffers
```

Two verified defects cause this (both confirmed by code inspection of this repo at `main`):

1. **`metal/src/engine/stream.cpp:716` — `ExpertStream::region()` returns `nullptr` when no arena chunk is big enough.**
   The prompt's scratch buffers need ~1.60 GB (`need`, chunk C = 1024). The loop at L717–751 skips every chunk smaller
   than `bytes + 64 MB` and falls out to `return nullptr;` at L752. The standalone-mmap fallback ("the prompt's buffers
   are their own allocation", added in `f5a6fb47` for the 32 GB M4) is nested *inside* the loop body (L729–744), so it is
   only reached when a chunk *is* big enough but stealing its slots would starve a slot class ("tight"). When the arena
   shrank at startup to ~0.8–1.0 GB per chunk (macOS had little to give at commit), no chunk qualifies and the fallback
   is unreachable. The caller (`metal/src/engine/window.cpp:2095–2096`, `Window::prompt_gemm_init`) then aborts the prompt.
2. **`metal/kernels/qsa_prompt.metal:234` — `auto tot = acc;` fails to compile at runtime on macOS 15+ / 26 SDKs.**
   Copy-constructing `metal::cooperative_tensor` hits Apple's `MPPTensorOpsMatMul2dImpl.h`
   `Layout::copy_construct(thread void *, thread void *)`, which is missing a `const` on its source parameter:
   `error: cannot initialize a parameter of type 'void *' with an rvalue of type 'const void *'`. The runtime Metal
   compile of `qsa_prompt.metal` exits 3 and the QSA flash prompt path never comes up. Reported and locally fixed by
   the user on the M2 (decode then ran 4.7–5.2 tok/s streaming off the SSD).

## User Stories

- As a user on a 24 GB Apple Silicon Mac running Strata in streaming mode, a prompt succeeds even when the expert arena
  is too small to carve a region out of: the engine logs that the prompt's buffers are their own allocation, and the
  prompt completes.
- As a user on macOS 15/26 with Xcode 26 CLT, the QSA flash prompt kernel compiles at runtime (no `void*`/`const void*`
  error), so prompts run on the fast path instead of aborting.

## Acceptance Criteria

1. `ExpertStream::region(bytes)`: when no chunk in `chunks_` satisfies `bytes + (64 MB)`, the function performs the
   standalone anonymous `mmap` fallback (the same allocation, message, and `region_`/`region_bytes_ = 0` state as the
   existing tight path at L729–744) instead of returning `nullptr`. It returns `nullptr` only if that `mmap` fails.
2. Existing behavior is unchanged in every other case: a chunk that fits still carves a region of its slots; the
   "tight" slot-class fallback still triggers where it did before.
3. `qsa_prompt.metal:234`: `tot` is initialized via
   `opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>,
   __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>()` (identical template arguments to the `acc`
   initialization at L232–233). The existing zero-fill loop over `tot` and all other kernel code are unchanged.
4. The Metal engine builds cleanly (no errors, no new warnings) with
   `cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release` + `ninja -C build-metal` on macOS.
5. On the 24 GB M2: the engine starts in streaming mode, the first prompt logs
   `strata metal stream: the arena is small: the prompt's ... buffers are their own allocation ...` and completes;
   decode speed is unchanged from the pre-fix run.
6. Work lands on one branch `fix/metal-prompt-arena-fallback`, pushed to origin, with a PR against `main`.

## Non-Goals

- The chat-template `reasoning_effort: none` cutoff (`tokenizer/chat_template.jinja`) is a separate defect; not fixed here.
- No change to arena sizing, `STRATA_METAL_PROMPT_CHUNK` defaults, decode paths, or the CUDA/HIP/SYCL backends
  (`metal/` is its own macOS-only CMake project; the two files touched are Metal-only).

## Open Questions

1. **On-Mac validation:** this sandbox is Linux; `metal/` refuses to configure on non-Apple (`metal/CMakeLists.txt`
   fatals), so acceptance criteria 4–5 can only be checked on your M2 Mac. Is running the build + live-prompt check on
   your machine after merge the plan, or should the PR be held open until you report a run?
2. **One PR, two fixes:** the repo's contributing guide says one change per PR, but both fixes were requested on one
   branch (`fix/metal-prompt-arena-fallback`). Confirm one PR containing both is intended (they come from the same
   Mac-testing session).
