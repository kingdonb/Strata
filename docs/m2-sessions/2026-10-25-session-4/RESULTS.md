# M2 validation session #4 — results

Session: 2026-10-25, the owner's M2, base `main` @ `af38e089`. Runbook:
[RUNBOOK.md](RUNBOOK.md) (this file mirrors its §5 table; it is filled at session time, S5).
Not a benchmark cycle — the three hypotheses' gates only. A FAIL: capture the artifacts, record FAIL,
stop; the diagnosis is a follow-up ticket.

## The results table

| # | Step | What it records | Verdict / value |
|---|---|---|---|
| 1 | S0/S1 | base SHA, dirty count, STRATA_LIFELINE, fix-in-tree grep, build + compile test | base SHA `af38e089` (merged main `7a0ce2bf`, branch `74137519`), dirty 0, STRATA_LIFELINE unset, fix-in-tree count 1 (static; `gates/pregate-h1-fix-in-tree-static.txt`) — build + compile test: **PASS** (`q36_compile_test` passed in ~8 s, all Metal pipelines created) |
| 2 | S2 (H1) | the round line verbatim: ttft (prefill seconds), prompt tok/s, decode tok/s; the four grep counts; PASS/FAIL | `chunk 512 size 4201 round 0: ttft 240.9 s, prompt 17.4 tok/s, decode 9.8 tok/s`; zero aborts: `prompt path borrows: 0`, `refilling a lent slot: 0`, `refilling the lent slots: 0`, `fill_slot_queued: 0`; 128 tokens decoded to DONE; **PASS** |
| 3 | S3 (H2) | the decode driver's tok/s glance; count of temp-bearing samples in the decode window; min–max °C; the gpu_note line verbatim; rendered-in-monitor, with the capture method recorded (screenshot and/or tmux capture-pane — the CoS addition, the artifact traceable to how it was taken); PASS/FAIL | tok/s glance: 4.4–7.0 tok/s; 8/8 samples in decode window with non-null temp; min–max: 44.4–52.0 °C; `gpu_note: no power reading without root (powermetrics), and an Apple Silicon GPU sits on the die, so there is no PCIe link to show; temperature comes from the rootless HID thermal sensors when they answer; the engine's own figures are in its log`; rendered in Overview monitor (`52 °C` with sparkline `▃▃▃▃▃▃▃▄▄▄▄▄▄▄`) and Hardware view; capture method: `tmux capture-pane + task-chat text capture`; **PASS** |
| 4 | S4 (H3) | before/after pgrep blocks; time-to-engine-gone; watcher gone (yes/no); orphan found (yes/no); host state + port state; PASS/FAIL | before: server 22537, engine 22538, watcher 22540 (`s:Sun Oct 25 15:05:37 2026`); hard `kill -9 22537` mid-decode; +5s: server GONE, engine GONE, watcher GONE; +15s: server GONE, engine GONE, watcher GONE; time-to-engine-gone: < 5 s; watcher gone: yes; orphan found: no; host state: stopped; port state: closed; **PASS** |
| 5 | S5 | machine state after (swap, temp), artifacts landed (list), key sweep result | swap after: `total = 2048.00M used = 642.00M free = 1406.00M` (reclaimed 22.4 GB swap), temp 52.0 °C peak, Chrome closed, session end 2026-10-25 15:07 EDT; artifacts landed: `RESULTS.md`, `RUNBOOK.md`, `engine.log`, `server.log`, `telemetry.txt`, `gates/h1-bench.json`, `gates/h1-engine-prefill-512.log`, `gates/h3-hard-kill.txt`, 3 pre-gates; key sweep: CLEAN (no API keys in archive); **PASS** |

## Recorded before the sitting (the sandbox pre-gates, 2026-10-25)

These ran in the Helix Linux sandbox on the session base — unit-level wiring evidence, not the hardware
gates (S2–S4 are); recorded here so the sitting starts from a known-green main.

| Gate | Command | Result | File |
|---|---|---|---|
| H3 pre-gate | `python3 -m unittest serve.test_lifeline -v` | 6/6 OK in 3.4 s (the portable-watcher case included — the PR #25 record's own instrument) | [gates/pregate-h3-lifeline-unit-linux.txt](gates/pregate-h3-lifeline-unit-linux.txt) |
| H2 pre-gate | `python3 -m unittest serve.test_server.MacTelemetry -v` | 5/5 OK (`test_thermal_reading_apple_silicon` included — the PR #24 ride-along's reader plumbing, fake reader + IOREG fixture) | [gates/pregate-h2-telemetry-unit-linux.txt](gates/pregate-h2-telemetry-unit-linux.txt) |
| H1 pre-gate (static) | `grep -n "no loan of expert-cache slots" src/program/generate.cpp` | the PR #22 guard present at the loan block (`generate.cpp:15390`); static analysis only, marked as such | [gates/pregate-h1-fix-in-tree-static.txt](gates/pregate-h1-fix-in-tree-static.txt) |

## The artifact checklist (what S5 lands in this dir)

- `RESULTS.md` — this file, filled.
- `engine.log` (or `.gz` over ~256 KiB) — the run config's log: the S2 bench's engine log is
  `gates/h1-engine-prefill-512.log`; the S3/S4 serving log is the top-level `engine.log`.
- `server.log` — the managed host's `<profile>.server.log` (the server-side record; optional, copied when
  present).
- `gates/h1-bench.json` — the bench tool's raw run JSON.
- `gates/h3-hard-kill.txt` — the before / kill / after pastes with timestamps.
- `telemetry.txt` — the H2 sample lines, the `gpu_note` line, the capture-method line (the CoS addition:
  the rendering's provenance), and the S0/S5 machine-state pastes.
- `app-diagnostics.txt` — not captured in this session (Terminal seat only; the app was not used — recorded
  here instead of creating an empty file).
- No API key in any artifact (S5's sweep proves it).

Traceability: every artifact names the runbook step that produced it; a step not run says so above.
