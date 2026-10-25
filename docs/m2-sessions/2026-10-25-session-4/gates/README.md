# gates/ — what lands here at session time

The gate outputs of `../RUNBOOK.md`'s steps, named as the runbook and `../RESULTS.md` reference them.
Text only, no binaries; the `docs/m2-sessions/README.md` conventions apply (a log over ~256 KiB is
gzipped and keeps the `.gz` name; never an API key — `../RUNBOOK.md` S5's sweep runs before anything
lands).

| File | Produced by | Contents |
|---|---|---|
| `pregate-h1-fix-in-tree-static.txt` | recorded 2026-10-25, before the sitting | the PR #22 guard's presence at `generate.cpp`'s loan block on `af38e089` (static analysis, labeled) |
| `pregate-h2-telemetry-unit-linux.txt` | recorded 2026-10-25, before the sitting | `serve.test_server.MacTelemetry` 5/5 on Linux (the reader's plumbing, fake reader + IOREG fixture) |
| `pregate-h3-lifeline-unit-linux.txt` | recorded 2026-10-25, before the sitting | `serve.test_lifeline` 6/6 on Linux, portable-watcher path included (the mechanism macOS uses in production) |
| `h1-bench.json` | S2 (the sitting) | the 4,201-token bench run's raw JSON (copied from `bench/results/2026-10-25-m2-q36-4k-prompt/`) |
| `h1-engine-prefill-512.log` | S2 (the sitting) | that run's engine stderr log — the zero-abort grep target (gzip if > ~256 KiB) |
| `h3-hard-kill.txt` | S4 (the sitting) | the before / kill / after pastes with timestamps: the three PIDs, the `kill -9`, the +5 s and +15 s checks, host + port state |

The serving session's `engine.log`, `server.log` and `telemetry.txt` land one level up (the session dir's
top level) per the archive conventions.
