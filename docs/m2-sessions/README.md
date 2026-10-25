# M2 session archive

Every guided M2 session lands its artifacts here, in the repo, so agents mine prior compile
errors and gate numbers instead of rediscovering them. The runbooks' capture steps (the
session's S10) write into `docs/m2-sessions/<this-session>/`; the owner pastes the same evidence
into the task chat as before — the archive is the durable copy, the chat is the conversation.

## Conventions

- One directory per session, named `YYYY-MM-DD-session-N-<slug>/`.
- Contents:
  - `RESULTS.md` — the filled results table, copied from the runbook (the session's verdicts
    and numbers, traceable to the raw artifacts).
  - the engine log — the run config's `log` file (`serve/server.py` spawns the engine with
    `stderr=self.log`); gzip it when it is over ~256 KiB and keep the `.gz` name.
  - `app-diagnostics.txt` — the app's Diagnostics → View logs → **Save logs** output (when the
    session uses the app).
  - `gates/` — the gate outputs: solo/cmp gate JSONs, `q36_chunk_parity` output, compile-test
    output.
  - `telemetry.txt` — the dispatch lines, memory/swap lines, machine-state captures (the S0/S8
    pastes).
- Text only, no binaries. Never an API key (`docs/TROUBLESHOOTING.md`'s rule).
- Every artifact is traceable to the runbook step that produced it — nothing invented after the
  fact; a step that was skipped says so.

## The index

| Session | Date | Runbook / card | Headline | Artifacts |
| --- | --- | --- | --- | --- |
| #1 (validation, T9-r1 multi-row + the pre-lever matrix) | 2026-10-18 | helix-specs `design/tasks/000685_t9-r1-multi-row` (its M2 run; PR #13 carried the numbers into `docs/QWEN36_METAL.md` §6) | dispatches per 512-token chunk 11,210 → 1,610 with `ATTN_ROWS=16`; the pre-lever prefill matrix recorded | `bench/results/2026-10-18-m2-q36-prefill/bench.json` (in-repo); the rest in the ticket record |
| #2 (validation, T6 fold A/B + T8 divergence checks) | 2026-10-19 | `docs/M2_VALIDATION_SESSION_RUNBOOK.md`; card helix-specs `design/tasks/000690_validation-session-1` | the fold A/B and gate-2 parity per the runbook's results table (§5); T8-r1's ticket-level ladder already 10/10 ×2 (build `ac5e0169`) | the runbook's §5 table (filled at session time); raw logs owner-side — awaiting archive |
| #3 (the new app UI walkthrough + the four merged engine features) | 2026-10-23 | `docs/M2_UI_SESSION_3_RUNBOOK.md`; card helix-specs `design/tasks/000697_guided-m2-session-3-new` | Gates 1–3 passed (12.6 tok/s sustained; `reused: 105` on the follow-up; dispatch line plausible); two fork bugs found and patched en route: the empty-`STRATA_API_KEY` spawn bug and the catalog model filter omitting Q36 on Mac | the runbook's §6 table (filled at session time); engine log + Save-logs file owner-side — awaiting archive |
| #4 (validation, the three merged fixes on post-toolchain main: H1 4K+ prompt, H2 rootless thermal, H3 lifeline watcher) | 2026-10-25 | `2026-10-25-session-4/RUNBOOK.md`; card helix-specs `design/tasks/000711_m2-validation-session-4` | anchored on main `af38e089`; sandbox pre-gates recorded (lifeline 6/6, MacTelemetry 5/5, fix-in-tree static); the three hardware gates at the sitting | the session dir — runbook + results table + pre-gates in it; the sitting's artifacts land beside them |

The value of the archive is already proven: session #3's two fixes (the API-key spawn bug, the
catalog filter) came out of reading the engine log against the gate expectations — archived, the
next agent reads them in seconds instead of asking the owner.

Session #4's directory is already in place (runbook, results table, pre-gates) — its sitting lands the
hardware artifacts beside them. Later sessions land here directly — the runbooks' capture steps point at
`docs/m2-sessions/<this-session>/`; create the session dir (named and laid out per the
conventions above) at capture time.
