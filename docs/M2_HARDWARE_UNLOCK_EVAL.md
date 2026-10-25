# The M2 hardware-unlock evaluation

**The question.** Can the owner's M2 stop being a scarce, owner-mediated device in the development
loop — and what would that cost, in money, ops burden and agreement changes? The working agreement
(rev. 2) is built around one hard constraint: the Helix sandbox cannot compile Metal kernels, so
every Metal-touching ticket stops at the compile wall and waits for the owner to run the M2 step
interactively. This ticket only **proposes**; the owner decides. Nothing here changes
infrastructure.

**What does NOT change under any option:** Benchmark Day stays gated (tally 4/24); no benchmark
numbers are published from this work; production launch paths stay byte-identical. The M2's role as
the *validation instrument for runtime gates* changes only if the owner rewrites the agreement —
each option below states what that rewrite would look like.

## The week's evidence (what changed since the spike was written)

- **The runbook pattern works.** Three guided sessions completed (T9-r1 multi-row; the T6 fold A/B
  and T8 checks; the new-UI walkthrough with gates 1–3 passed). Every session produced its numbers
  and its findings; session #3's two fork bugs (the empty-`STRATA_API_KEY` spawn bug, the catalog
  filter omitting Q36 on Mac) were found by reading engine-log lines against gate expectations.
- **The interactive loop's cost is measured, not estimated.** The runbooks' own clocks: session #2
  ≈ 75–90 min wall clock, session #3 ≈ 150 min; owner hands-on ≈ 15 min per sitting (the pastes and
  one screenshot); the rest is watching. The real cost is not the owner's minutes — it is the
  **round-trip latency**: a compile error found in a paste costs a full evening turnaround.
- **The sim pattern covers host-verifiable semantics.** `metal/tests/moe_sim.cpp` (and the
  `q36_chunk_parity` family) run on the plain Linux sandbox — verified 2026-10-25
  (`moe_sim: all stages OK (55 checks)`, seconds to build and run, see
  `docs/ENVIRONMENT.md`). Kernel *semantics* no longer need the M2. What the M2 keeps:
  the runtime build, pipeline creation, dispatch behavior, the gates that need real Metal, and
  the feel.
- **The archive exists now** (`docs/m2-sessions/`), so whichever option lands, artifacts flow back
  into the repo the agents work in.

## Toolchain facts (posture; not verifiable from the Linux sandbox — verify at decision time)

- The Metal compiler (`metal` / `xcrun metal`, plus `metallib`) ships inside Apple's Xcode
  toolchain — macOS-only, closed source. Apple's EULA restricts macOS use to Apple-branded
  hardware. There is no open-source reimplementation that emits valid `.air`/`.metallib` today —
  treat "OSS Metal compiler" as a watch item, not a plan.
- Compiling is not running: on Apple Silicon, Virtualization.framework provides no GPU
  acceleration to guests — a macOS VM can host the compiler but never the Metal runtime. Runtime
  gates need real Apple hardware wherever compilation happens.

## The options

### A. Always-on Apple node as a Helix server asset

A used Mac mini (M-series; market estimate ~$400–600, not a measurement) in the bunker, macOS +
Xcode, reachable over SSH from the agent sandboxes. Because it is real Apple hardware with a GPU,
it compiles **and** runs the parity/ladder gates autonomously.

- **Loop shape:** the interactive loop shrinks to near zero for routine development — agents
  compile and gate on the node; the owner's M2 retreats to product-level checks (the feel blocks,
  and Benchmark Day when it opens).
- **Helix mechanism:** the SSH credential rides as a project secret (the same
  `helix secret create -p prj_…` path as the `GITHUB_TOKEN` recipe — a private key value, write-only);
  optionally, if the node also *serves*, it attaches as a Helix provider endpoint
  (`helix provider create -u https://<node>/v1 …`). No new Helix machinery needed.
- **Artifact flow-back:** the node holds a clone of the fork; build and gate outputs are committed
  to `docs/m2-sessions/<session>/` and pushed (or fetched back via the WS5 token). The
  session-archive conventions apply unchanged.
- **Costs / ops:** hardware + power + care-and-feeding (macOS updates, disk, security patching);
  an always-on internet-reachable Mac is an attack surface — keep it LAN-only behind the owner's
  network, credentials in the Helix secret store, and treat the node as semi-trusted (never a
  model-serving public endpoint).
- **Agreement implications (the owner's rewrite, if adopted):** runtime gates on routine tickets
  could run agent-driven on the node; the owner's minutes-scale contact stays for the feel blocks
  and anything the agreement still names as owner-run.

### B. Compile-only microservice on an existing always-on Mac

A tiny authenticated HTTP endpoint (on the owner's M2 or another always-on Mac) that takes
`.metal` sources, returns `.air`/`.metallib` artifacts and the compiler's stderr. Smaller ops
footprint than A; no GPU work accepted.

- **Loop shape:** compile-check moves off the owner's evening; **runtime gates stay interactive** —
  the owner's M2 sessions remain the gate instrument (the measured 75–150-minute sittings
  continue).
- **Helix mechanism:** a Helix secret carrying the service's URL + token; the sandbox curls it
  from the ticket's scripts. (A trigger-driven queue is the same thing with buffering — not worth
  separate machinery until more than one agent contends.)
- **Costs / ops:** near-zero hardware (reuses an existing Mac); the service is ~100 lines behind an
  auth token; the owner still owns the sittings.
- **Agreement implications:** none — the validation instrument is unchanged; only the compile wall
  moves.

### C. macOS VM on bunker hardware

Compile-only (see the GPU fact above), and license-clean **only if the host is Apple hardware
running macOS** — on non-Apple bunker hardware a macOS guest violates the EULA outright. Where an
Apple host exists, a VM buys little over option A (same hardware; more layers, less performance,
same patching).

- **Verdict:** dominated by A or B in every scenario; record it so the option is on paper, not as
  a candidate to develop.

### D. Status quo

The measured baseline every option competes with: owner-run interactive M2 sessions per the
runbooks, ~15 min hands-on per sitting, evening-scale latency on compile errors, and the session
archive capturing the evidence. It works — it just does not scale past one owner's attention.

## Recommendation

**Adopt A (the always-on Apple node) when the next Metal-heavy phase justifies the hardware; adopt
B meanwhile** — it reuses hardware the owner already runs, removes the compile wall, and keeps the
agreement untouched. Reasons, from the evidence: the sim pattern already took semantics off the M2,
so the M2's remaining interactive work is compile + runtime gates; B removes the compile half this
week for ~100 lines and no hardware; A removes the other half and retires the sittings — the
measured loop cost (75–150 min wall clock, evening latency) is exactly what A's autonomy
eliminates. C stays dominated. D stays the fallback if neither is wanted.

**Proof-of-concept plan for A (one compile + one parity gate end-to-end, when the owner approves):**

1. **Acquire:** a used Mac mini (M-series, ≥16 GB), macOS + Xcode toolchain, LAN-only.
2. **Expose:** `helix secret create -n M2_NODE_SSH -v "<key>" -p prj_…`; the sandbox reaches the
   node with plain `ssh` from the terminal (no new Helix machinery).
3. **Probe (the acceptance test):** from a sandbox — `ssh <node> "cd <fork clone> && git pull &&
   cmake --build build-metal && ./build-metal/q36_compile_test"` (the compile wall crossed), then
   `ssh <node> "cd <fork clone> && ./build-metal/q36_chunk_parity"` (a real runtime gate, no owner
   paste), and one `moe_sim` run stays local to show what remains host-side.
4. **Flow back:** the node's outputs land in `docs/m2-sessions/<date>-session-4-poc/` per the
   archive conventions; the PoC's verdict — measured latency per gate, what still needs the owner —
   is what the owner's agreement rewrite would be written from.

**The owner decides; this ticket only proposes.** Until a decision, the working agreement stands as
it is: the M2 remains the validation instrument, the runbooks and the session archive carry the
interactive loop, and the environment carries everything that does not need Apple silicon.
