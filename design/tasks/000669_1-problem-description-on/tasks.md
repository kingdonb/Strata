# Implementation Tasks: Fix Metal Prompt Arena Fallback on Small-Memory Macs

- [ ] Create branch `fix/metal-prompt-arena-fallback` off `main`
- [ ] `metal/src/engine/stream.cpp` (in `ExpertStream::region`, replacing the bare `return nullptr;` at L752): add the standalone anonymous-mmap fallback after the chunk loop (allocation, `region_`/`region_bytes_ = 0` state, "the arena is small ..." log line, and `nullptr` on `MAP_FAILED` exactly as in `design.md` Fix 1)
- [ ] `metal/kernels/qsa_prompt.metal` (L234): replace `auto tot = acc;` with `opO.template get_destination_cooperative_tensor<...>(...)` using the same template arguments as the `acc` call at L232–233; keep the zero-fill loop and comment intact
- [ ] `docs/MAC.md` (~L141): extend the small-arena sentence to say the prompt's buffers also get their own allocation when no arena chunk is big enough for them
- [ ] Commit with a repo-style message (lowercase; states both behaviors and where they were seen, cf. `f5a6fb47`)
- [ ] Push `fix/metal-prompt-arena-fallback` to origin and open a PR against `main` (kingdonb/Strata), noting in the description that it is two fixes from one Mac-testing session
- [ ] On the user's M2 Mac (cannot be done in this Linux sandbox — `metal/` builds on macOS only): configure + build `metal/` cleanly, start the server in streaming mode, confirm the first prompt logs the fallback line and completes, and confirm decode speed matches the pre-fix run
