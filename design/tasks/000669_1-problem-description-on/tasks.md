# Implementation Tasks: Fix Metal Prompt Arena Fallback on Small-Memory Macs

- [x] Create branch off `main` (Helix flow: `feature/000669-fix-metal-prompt-arena`; the originally requested name `fix/metal-prompt-arena-fallback` is covered by this task's PR)
- [~] `metal/src/engine/stream.cpp` (in `ExpertStream::region`, replacing the bare `return nullptr;` at L752): add the standalone anonymous-mmap fallback after the chunk loop (allocation, `region_`/`region_bytes_ = 0` state, "the arena is small ..." log line, and `nullptr` on `MAP_FAILED` exactly as in `design.md` Fix 1)
- [ ] `metal/kernels/qsa_prompt.metal` (L234): replace `auto tot = acc;` with `opO.template get_destination_cooperative_tensor<...>(...)` using the same template arguments as the `acc` call at L232–233; keep the zero-fill loop and comment intact
- [ ] `docs/MAC.md` (~L141): extend the small-arena sentence to say the prompt's buffers also get their own allocation when no arena chunk is big enough for them
- [ ] Commit with a conventional message (`fix(metal): ...` — the Helix commit-msg hook enforces the format and adds the Spec-Ref trailer; the description itself keeps the repo's plain-words style, cf. `f5a6fb47`)
- [ ] Push `feature/000669-fix-metal-prompt-arena` to origin and propose a PR against `main` (kingdonb/Strata), noting in the description that it is two fixes from one Mac-testing session (reviewer confirmed one PR is wanted; split for upstream later if requested there)
- [ ] On the reviewer's M2 Mac (cannot be done in this Linux sandbox — `metal/` builds on macOS only; reviewer confirmed they will build and test on their workstation and report back): configure + build `metal/` cleanly, start the server in streaming mode, confirm the first prompt logs the fallback line and completes, and confirm decode speed matches the pre-fix run

## Reviewer decisions (2026-10-13)

- Mac-side build/prompt validation: the reviewer runs it on their M2 workstation; back-and-forth feedback is expected.
- One PR with both fixes is confirmed for this fork; split into separate PRs only if upstream later asks.
- Branch name: use the Helix task branch `feature/000669-fix-metal-prompt-arena` (base `main`).
