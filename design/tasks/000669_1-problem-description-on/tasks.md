# Implementation Tasks: Fix Metal Prompt Arena Fallback on Small-Memory Macs

- [x] Create branch off `main` (Helix flow: `feature/000669-fix-metal-prompt-arena`; the originally requested name `fix/metal-prompt-arena-fallback` is covered by this task's PR)
- [x] `metal/src/engine/stream.cpp` (in `ExpertStream::region`, replacing the bare `return nullptr;` at L752): add the standalone anonymous-mmap fallback after the chunk loop (allocation, `region_`/`region_bytes_ = 0` state, "the arena is small ..." log line, and `nullptr` on `MAP_FAILED` exactly as in `design.md` Fix 1)
- [x] `metal/kernels/qsa_prompt.metal` (L234): replace `auto tot = acc;` with `opO.template get_destination_cooperative_tensor<...>(...)` using the same template arguments as the `acc` call at L232–233; keep the zero-fill loop and comment intact
- [x] `docs/MAC.md` (~L141): extend the small-arena sentence to say the prompt's buffers also get their own allocation when no arena chunk is big enough for them
- [x] Commit with a conventional message (`fix(metal): ...` — the Helix commit-msg hook enforces the format and adds the Spec-Ref trailer; the description itself keeps the repo's plain-words style, cf. `f5a6fb47`) → Strata `56a86b3a`
- [x] Push `feature/000669-fix-metal-prompt-arena` to origin and propose a PR against `main` (kingdonb/Strata), noting in the description that it is two fixes from one Mac-testing session (reviewer confirmed one PR is wanted; split for upstream later if requested there)
- [x] On the reviewer's M2 Mac (cannot be done in this Linux sandbox — `metal/` builds on macOS only; reviewer confirmed they will build and test on their workstation and report back): configure + build `metal/` cleanly, start the server in streaming mode, confirm the first prompt logs the fallback line and completes, and confirm decode speed matches the pre-fix run → **verified by the reviewer on the M2 (2026-10-13)**: arena chunk was ~0.80 GB against the prompt's 1.60 GB, the fallback fired with the expected log line, prompts completed (no "no arena region" error), and 3.7–4.1 tok/s steady generation in web chat passed end-to-end (streaming, thought generation, multi-turn context caching; no crashes or memory corruption). PR #1 open for merge.

- [x] Verify Gemini's upstream claims against the actual repo (fetched `Niko1221/Strata` directly): hotfix `5ce99945` (#2323) fixes `qsa_prompt.metal` **identically** to ours (plus a sharper comment: "tot is zeroed right below, so the copy was only there for its type"), adds `require_fit` dispatch checks (`mtl_rt`, `runtime.cpp`, `gdn.metal`); `metal/src/engine/stream.cpp` and `docs/MAC.md` are **untouched** — the small-arena crash is still live upstream on 16-24 GB Macs; 4-dot tags are house style (v0.1.43.1, v0.1.44.1 exist)
- [x] Rebase `feature/000669-fix-metal-prompt-arena` onto upstream `v0.1.44.1` (= upstream/main tip `296b59d2`), dropping our `qsa_prompt.metal` hunk (upstream's is identical)
- [x] DRY refactor per reviewer request: both standalone-mmap sites now share `ExpertStream::standalone_region(bytes)` (stream.cpp helper + private declaration in stream.h); behavior identical incl. the `MAP_FAILED` case
- [x] One atomic commit `1b221316` (stream.cpp + stream.h + MAC.md, message references #2323), force-pushed — PR #1 updated; branch remains landable (fork main is an ancestor; PR diff shrinks to just this fix once the fork's main syncs to v0.1.44.1)
- [ ] Upstream submission to Niko1221/Strata — user's decision; suggested framing: "completes #2323 for small-memory Macs (the hotfix's reporter ran a 96 GB M2 Max where chunks are several GB; 16-24 GB Macs still stop at the first prompt)". Cross-repo PR mechanics need the user's GitHub flow; also re-run the M2 build on the refactored commit first.
- [ ] Defect B (chat template `reasoning_effort: none` cutoff, `tokenizer/chat_template.jinja`) is still open upstream: the hotfix's reasoning_effort work (5d08c753, Responses/Codex) is API plumbing, not the jinja empty-thought issue. Candidate follow-up PR.

- [x] Final pre-submission review of the branch (2026-10-14): diff re-read as upstream would see it (1 commit, 3 files, +20/−13); one fix applied — the commit message's "verified on the reporter's M2" misattributed the 24 GB verification to #2323's reporter (Excavatory, 96 GB M2 Max); amended to "verified on a 24 GB M2" and force-pushed (`5f158604`)
- [x] Upstream etiquette check (AGENTS.md / COMMUNITY_BENCHMARKS.md): one change per PR ✓ (docs sentence rides with the fix); "say what you did not test" → listed in the PR body draft; no matching open entry in TEST_REQUESTS.md to cite — #2323 itself is the reference; only `metal/` builds touched ✓
- [ ] Before submitting upstream: rebuild + run one prompt on the M2 at `5f158604` (the verified run predates the `standalone_region()` refactor — statement-for-statement identical, but the etiquette is to test what you ship); then merge PR #1 in the fork and submit upstream with the paste-ready body in the reply

## Upstream review notes (2026-10-13, self-review after M2 verification)

- Fix 1 is an extension of the accepted f5a6fb47 design decision to the case it missed, not a new mechanism — the strongest upstream argument. Diff is 12 lines; no behavior change for any arena that already fits.
- Fix 2 removes the only cooperative-tensor copy construction in the tree (verified: kernels and src); the direct `get_*_cooperative_tensor` call is the established pattern (22 uses in moe_mm.metal). Upstream will hit the same Apple-header compile failure as soon as they build with the Xcode 26 CLT / macOS 26 SDK, regardless of Mac size — this fix is future-proofing for them, not just small-Mac-only.
- Known gaps a maintainer may flag, deliberate non-blockers:
  1. The fallback's ~1.6 GB standalone allocation is taken beside a shrunken arena and is not counted by the engine's footprint accounting (proc_mem/footprint_cap run at arena-fill time, window.cpp:1153) nor logged with a memory figure. Cheap follow-up: log available/footprint alongside the fallback line.
  2. The prompt-chunk knob is not mentioned at the fallback: smaller STRATA_METAL_PROMPT_CHUNK (C=512) shrinks the standalone allocation roughly in half (fixed weights ~0.5-0.6 GB do not scale with C), which lowers pressure on exactly these machines — though on this user's 0.80 GB chunks it does not make the buffers fit the arena again. Worth a MAC.md sentence or a log hint in a follow-up.
  3. The 10-line fallback body now exists twice (tight path + after-loop). A static helper could deduplicate; upstream's style tolerates local duplication, so left as-is for a minimal diff.
- Rejected for this PR (unchanged from design): auto-shrinking the prompt chunk C to refit the arena (changes prompt-throughput behavior, more complex); shimming Apple's header in metal_compat (bigger, out of scope).
- Recommendation recorded: merge PR #1 as the verified crash fix; do any polish as a separate small follow-up PR, keeping one change per PR for upstream.

## Verification on the M2 (reviewer, 2026-10-13)

- Hardware/software: Apple M2, 24 GB unified memory, macOS 26.6.2, Qwen3.8-Flash-Next-GSQ-RCO-Q2_0 (31.64 GiB `.experts` streaming file).
- Measured: arena chunk ~0.80 GB against the prompt's 1.60 GB of buffers — the after-loop fallback fired: "strata metal stream: the arena is small: the prompt's 1.60 GB of buffers are their own allocation"; prompt evaluation succeeded where every prompt used to stop with "metal prompt: no arena region for the prompt buffers".
- Live run: 3.7–4.1 tok/s steady generation in web chat; streaming, thought generation (\langle think\rangle), and multi-turn context caching all passed end-to-end, no crashes or memory corruption.

## Implementation Notes (2026-10-13)

- Commit `56a86b3a` on `feature/000669-fix-metal-prompt-arena` (base `main` at `ed4773eb`, merged/mergeable) contains all three edits; the branch is pushed to origin.
- The stream.cpp fallback sits at function-body level after the loop; its `sz`/`m` locals do not collide with the loop's inner `tight`-block ones (different scopes).
- The kernel comment was kept and extended with the why (Apple's non-const `Layout::copy_construct`), since the old comment's M4 rationale still applies.
- Remaining: the M2 build + live-prompt check on the reviewer's workstation (their action); no Metal build was possible in this Linux sandbox, as expected and accepted by the reviewer.

## Reviewer decisions (2026-10-13)

- Mac-side build/prompt validation: the reviewer runs it on their M2 workstation; back-and-forth feedback is expected.
- One PR with both fixes is confirmed for this fork; split into separate PRs only if upstream later asks.
- Branch name: use the Helix task branch `feature/000669-fix-metal-prompt-arena` (base `main`).
