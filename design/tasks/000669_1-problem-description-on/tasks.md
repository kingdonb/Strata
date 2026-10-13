# Implementation Tasks: Fix Metal Prompt Arena Fallback on Small-Memory Macs

- [x] Create branch off `main` (Helix flow: `feature/000669-fix-metal-prompt-arena`; the originally requested name `fix/metal-prompt-arena-fallback` is covered by this task's PR)
- [x] `metal/src/engine/stream.cpp` (in `ExpertStream::region`, replacing the bare `return nullptr;` at L752): add the standalone anonymous-mmap fallback after the chunk loop (allocation, `region_`/`region_bytes_ = 0` state, "the arena is small ..." log line, and `nullptr` on `MAP_FAILED` exactly as in `design.md` Fix 1)
- [x] `metal/kernels/qsa_prompt.metal` (L234): replace `auto tot = acc;` with `opO.template get_destination_cooperative_tensor<...>(...)` using the same template arguments as the `acc` call at L232–233; keep the zero-fill loop and comment intact
- [x] `docs/MAC.md` (~L141): extend the small-arena sentence to say the prompt's buffers also get their own allocation when no arena chunk is big enough for them
- [x] Commit with a conventional message (`fix(metal): ...` — the Helix commit-msg hook enforces the format and adds the Spec-Ref trailer; the description itself keeps the repo's plain-words style, cf. `f5a6fb47`) → Strata `56a86b3a`
- [x] Push `feature/000669-fix-metal-prompt-arena` to origin and propose a PR against `main` (kingdonb/Strata), noting in the description that it is two fixes from one Mac-testing session (reviewer confirmed one PR is wanted; split for upstream later if requested there)
- [ ] On the reviewer's M2 Mac (cannot be done in this Linux sandbox — `metal/` builds on macOS only; reviewer confirmed they will build and test on their workstation and report back): configure + build `metal/` cleanly, start the server in streaming mode, confirm the first prompt logs the fallback line and completes, and confirm decode speed matches the pre-fix run

## Implementation Notes (2026-10-13)

- Commit `56a86b3a` on `feature/000669-fix-metal-prompt-arena` (base `main` at `ed4773eb`, merged/mergeable) contains all three edits; the branch is pushed to origin.
- The stream.cpp fallback sits at function-body level after the loop; its `sz`/`m` locals do not collide with the loop's inner `tight`-block ones (different scopes).
- The kernel comment was kept and extended with the why (Apple's non-const `Layout::copy_construct`), since the old comment's M4 rationale still applies.
- Remaining: the M2 build + live-prompt check on the reviewer's workstation (their action); no Metal build was possible in this Linux sandbox, as expected and accepted by the reviewer.

## Reviewer decisions (2026-10-13)

- Mac-side build/prompt validation: the reviewer runs it on their M2 workstation; back-and-forth feedback is expected.
- One PR with both fixes is confirmed for this fork; split into separate PRs only if upstream later asks.
- Branch name: use the Helix task branch `feature/000669-fix-metal-prompt-arena` (base `main`).
