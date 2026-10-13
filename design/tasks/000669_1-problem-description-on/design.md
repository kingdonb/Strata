# Design: Fix Metal Prompt Arena Fallback on Small-Memory Macs

## Context

The Metal engine's SSD-streaming mode keeps routed experts in a resident arena (`ExpertStream`, `metal/src/engine/stream.cpp`):
at most 4 `mmap`'d anonymous buffers ("chunks", `chunks_`), each capped by the device's `maxBufferLength`, filled in
1.5 GB steps until macOS's memory floor is reached; if macOS has less than planned at commit time the arena shrinks
(`shrink_to_` retry in `Window`, `window.cpp:1195–1208`). Prompt scratch buffers (~1.60 GB at chunk C = 1024, measured
by the `AB` lambda in `Window::prompt_gemm_init`, `window.cpp:2050–2090`) are placed *once* by `ExpertStream::region()`
at the end of one chunk, whose slots are held (state 3) while a prompt runs (`hold_region`). If no chunk is big enough,
`region()` returned `nullptr` and `prompt_gemm_init` aborted the prompt with
`metal prompt: no arena region for the prompt buffers` (`window.cpp:2096`).

## Fix 1 — after-loop standalone mmap fallback in `ExpertStream::region()`

**Root cause (verified at `main`):** `stream.cpp:716–753` — the chunk loop `continue`s past every chunk with
`bytes < bytes + (64 << 20)` and exits to `return nullptr;` at L752. The standalone-mmap fallback (L729–744) is nested
in the loop body behind the `tight` slot-class check, so it runs only when a chunk *is* big enough. A 24 GB Mac whose
arena shrank to ~0.8–1.0 GB per chunk (observed in the user's logs) skips every chunk and fails.

**Change:** replace the bare `return nullptr;` at `stream.cpp:752` with the standalone mmap fallback (the user's
verified local fix, mirroring the `tight` path's body at L731–744):

```cpp
    const uint64_t sz = (bytes + PAGE - 1) / PAGE * PAGE;
    void* m = mmap(nullptr, (size_t) sz, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (m != MAP_FAILED) {
        region_slots_.clear();
        region_ = (uint8_t*) m;
        region_bytes_ = 0;   // fresh zero pages; nothing in the arena to empty
        std::fprintf(stderr, "strata metal stream: the arena is small: the prompt's %.2f GB of buffers are their own "
                             "allocation, not a region of its slots\n", bytes / 1e9);
        return region_;
    }
    return nullptr;
```

**Why this is safe:**
- `region()` runs once per engine (`prompt_gemm_init` early-returns when `pg_` exists, `window.cpp:2019`), so the extra
  ~1.6 GB allocation lives exactly as long as the prompt buffers themselves — same lifetime as the existing tight path.
- With `region_bytes_ = 0`, `hold_region`'s `memset(region_, 0, region_bytes_)` (`stream.cpp:776`) is a no-op and no
  arena slots are stolen (empty `region_slots_`), so decoding keeps the full arena while a prompt runs.
- The `tight` path inside the loop is untouched: a fitting chunk still carves a region, and the slot-class guard still
  applies there. The after-loop fallback steals nothing, so it needs no `tight` check.
- No `munmap` today for either standalone path (verified: none in `stream.cpp`) — same behavior before and after; the
  region lives with `pg_` for the engine's lifetime.

**Alternative rejected:** auto-shrinking the prompt chunk C until `region()` fits. More complex, changes prompt
throughput defaults, and `STRATA_METAL_PROMPT_CHUNK` already exists as the manual knob. The fallback matches the
pattern upstream chose in `f5a6fb47`.

## Fix 2 — `tot` via `get_destination_cooperative_tensor` in `qsa_prompt.metal`

**Root cause (user-verified on the M2):** `qsa_prompt.metal:234` `auto tot = acc;` copy-constructs a
`metal::cooperative_tensor`. In Apple's `MPPTensorOpsMatMul2dImpl.h` (macOS 15+ / 26 SDKs),
`Layout::copy_construct(thread void *, thread void *)` lacks `const` on its source parameter, so the runtime Metal
compile of `qsa_prompt.metal` fails with exit 3:
`cannot initialize a parameter of type 'void *' with an rvalue of type 'const void *'`.

**Change:** initialize `tot` from the same op instead of copying (identical template arguments to the `acc` call at
L232–233):

```cpp
    auto tot = opO.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>,
                                                               __tensor_ops_detail::__remove_addrspace_t<decltype(rv)>, float>();
```

**Semantics unchanged:** `acc` (per-chunk destination) and `tot` (running output, `tot[i] = fma(tot[i], fac[...], acc[i])`)
are already distinct storages today (the copy constructor deep-copies; the kernel's correctness relies on it — the L234
comment). A fresh destination tensor is also distinct storage; the existing zero-fill loop (L237–238) initializes it.
Register usage is unchanged (two tensors existed before, two after). This is the only copy-construction of a
cooperative tensor in `metal/kernels/` (verified by grep); `qsa_prompt.metal` already uses this exact call twice
elsewhere, so it is proven in this kernel.

## Constraints & Notes for the Implementer

- **Metal builds only on macOS** (`metal/CMakeLists.txt` fatals on non-APPLE). This Linux sandbox cannot compile
  `stream.cpp`/`qsa_prompt.metal`; the build check (acceptance 4) and the live-prompt check (acceptance 5) must run on
  the user's M2. Do not attempt to "fix" this by touching the root CMake build — `metal/` is deliberately separate.
- `metal/src/engine/window.cpp` needs no change: once `region()` stops returning `nullptr`, the error at L2096 simply
  stops firing.
- The two fallbacks (tight in-loop, no-fit after-loop) share the same log line on purpose — one story in the log for
  "the prompt's buffers are their own allocation".
- Docs: `docs/MAC.md:141` already says "a small arena now gives the prompt's 1.6 GB of buffers their own allocation";
  extend that sentence to cover the no-chunk-fits case (behavior description, no numbers).
- Commit message style in this repo: lowercase, states the behavior and where it was seen (see `f5a6fb47` for the shape).
- Process: branch `fix/metal-prompt-arena-fallback` off `main`; one PR with both fixes (user's explicit ask; note the
  repo's one-change-per-PR preference in the PR description).
