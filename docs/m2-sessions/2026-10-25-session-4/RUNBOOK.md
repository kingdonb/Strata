# M2 validation session #4 runbook — the three merged fixes on post-toolchain main

One sitting, one paste per step, every verdict and number lands in the results table (§5), not in prose.
Source of record: the spec card helix-specs `design/tasks/000711_m2-validation-session-4` (spt_01m5t40q75qb14r5jvfe10dpdy).
Base: `main` @ `af38e089` — the first main carrying all of this week's fixes together (PR #22, #24, #25, #26, #27).
Process: `docs/M2_VALIDATION_SESSION_RUNBOOK.md`'s #26-amended form — the capture steps write into
`docs/m2-sessions/2026-10-25-session-4/` (create the dir with `mkdir -p` first), gzip a log over ~256 KiB
and keep the `.gz` name, never an API key in an artifact (`docs/TROUBLESHOOTING.md`'s rule).

What this session is: three hypotheses, each with its gate —

- **H1** — the 4K+ prompt Metal fix (PR #22; M2-validated at 4,201 tokens pre-toolchain) still holds on
  current main.
- **H2** — the live rootless Apple-Silicon GPU/SoC thermal telemetry (PR #24 ride-along) renders in the
  interactive monitor during a sustained decode.
- **H3** — the stabilized lifeline watcher (PR #25) ends engine children under a hard server kill.

What it is not: a benchmark cycle — no re-baselines, no new matrices, no scoreboard edits. The
pre-toolchain validation numbers are already recorded (the card); this sitting re-proves the three fixes
on the post-toolchain main and archives the evidence. A FAIL: capture the failure artifacts, record FAIL,
stop — the diagnosis is a follow-up ticket, not this sitting's work.

Seats: the owner's hands are on the M2 (this card's agent guided the steps through the task chat; gemini/agy
may follow along — the steps do not change). All three hypotheses run in the **Terminal seat** — the TUI
monitor `strata launch` opens is the interactive monitor of H2; the server it starts is the process H3 kills.

## 1. The hypotheses and their gates

| # | Hypothesis | Gate (all three must hold) | Where the evidence lands |
|---|---|---|---|
| 1 | H1: PR #22's Metal prompt-path loan fix holds on `af38e089` | a 4,201-token prompt (> 4,096) runs live prefill **and** decode to DONE; the engine log carries the fix's own line `prompt path allocates its own buffers (no cache slots to borrow)` and **zero** of: `prompt path borrows`, `refilling a lent slot`, `refilling the lent slots`, `fill_slot_queued` | `gates/h1-bench.json`, `gates/h1-engine-prefill-512.log`; numbers in §5 |
| 2 | H2: the rootless HID thermal reading (PR #24 ride-along) renders in the interactive monitor during sustained decode | during a streaming decode: ≥ 3 `/v1/telemetry` samples ≥ 1 s apart with a non-null `devices[0].temp`; the monitor shows it (Overview's GPU-temperature tile / the Hardware view's `temp C`); the truthful `gpu_note` line captured | `telemetry.txt` (the sample lines + the note), screenshots in the task chat |
| 3 | H3: the lifeline watcher (PR #25) ends the engine child under a hard server kill | `kill -9` the `serve.server` process mid-decode: engine gone ≤ 15 s, watcher gone, no orphan (no `strata` engine and no `serve.server` process left), the managed host stopped, the port closed | `gates/h3-hard-kill.txt`; engine log tail in `engine.log` |

## 2. Preconditions — machine state, before (S0, ~5 min)

One paste:

```bash
export Q36_MODEL=/Users/yebyen/w/Strata-data/models/qwen36/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
export REPO=$HOME/w/Strata
date '+session start: %Y-%m-%d %H:%M %Z'; sysctl vm.swapusage
pgrep -xl "Google Chrome" || echo "Chrome: closed"
git -C $REPO rev-parse HEAD; git -C $REPO status -s | wc -l
printenv STRATA_LIFELINE || echo "STRATA_LIFELINE: unset (macOS mechanism: watcher)"
grep -c "no loan of expert-cache slots" $REPO/src/program/generate.cpp
ls -l $REPO/build-metal/strata 2>/dev/null; ls -l "$Q36_MODEL"
```

Expected: a SHA equal to `af38e089…` and a dirty-file count of 0; `Chrome: closed`; `STRATA_LIFELINE:
unset` (if it is set, unset it — the watcher must be the mechanism macOS gets by default); the fix-in-tree
grep count `1`; the model file present (the filename carries the quant: UD-Q2_K_XL).
Record: date; swap before; SHA + dirty count; the grep count.
Static, not a run: the fix-in-tree grep is reading code, not measuring (marked so in the results table);
the live gate is S2.
Abort: none — this is a record.

## 3. The steps

### S1 — Build + compile test (the recorded form; both earlier sessions' S1) — ~10–20 min

```bash
cmake --build $REPO/build-metal && $REPO/build-metal/q36_compile_test
```

Expected: the build completes (first build after the toolchain unlock may take longer — record the
minutes); the compile test creates all pipelines (split-K, the folded pair, the f16 lib's).
Abort: any compile error or a failed pipeline creation → stop the whole session; record the failing line
(a post-toolchain build failure is itself a finding for the toolchain ticket).

### S2 — H1: the 4,201-token prompt through live prefill/decode (gate 1) — ~6 min

One paste (the tool form `docs/M2_VALIDATION_SESSION_RUNBOOK.md` §3's S2 uses; one size, one round — this
is the fix's gate, not a benchmark):

```bash
mkdir -p bench/results/2026-10-25-m2-q36-4k-prompt
python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 16384" --sizes 4201 --chunks 512 --rounds 1 --gen 128 \
  --out bench/results/2026-10-25-m2-q36-4k-prompt/bench.json
```

Expected: `chunk 512: warm-up done (load X.Xs)`; the round line `chunk 512 size 4201 round 1: ttft ... s,
prompt ... tok/s, decode ... tok/s`; the markdown table; exit 0. Sanity bands from the recorded bases (not
gates — the gate is the zero-abort): chunk-512 read ≈ 225 s at 4,098 tokens (≈ 18 prompt tok/s), decode
≈ 8–16 tok/s at pos ≈ 4,200 — so TTFT ≈ 225–235 s and a wild number gets one retry, then a note.
Then the abort-gate greps (second paste):

```bash
L=bench/results/2026-10-25-m2-q36-4k-prompt/engine-prefill-512.log
for p in "prompt path borrows" "refilling a lent slot" "refilling the lent slots" "fill_slot_queued"; do
  printf '%s: %s\n' "$p" "$(grep -c "$p" $L)"; done
grep -c "prompt path allocates its own buffers" $L
```

Expected: every listed pattern counts **0**; the fix's own line counts **≥ 1**. Any nonzero count on the
listed patterns, or a mid-request engine death (the tool's `engine died mid-request`), is H1 FAIL: copy
the log + JSON to `gates/`, record, stop.

### S3 — H2: the rootless thermal reading in the interactive monitor during a sustained decode (gate 2) — ~15 min

Start serving (one paste; the TUI that opens is the interactive monitor):

```bash
cd $REPO && strata launch --model q2_0 --yes
```

Expected: the engine starts and the monitor opens on the Overview. Record the address it prints (also in
`strata launch --status`); below it is `$URL` (the session #3 form was `http://127.0.0.1:18500` — use what
this launch prints). Second terminal, one paste for the driver's context:

```bash
export STRATA_KEY="$(strata launch --show-key)"; export URL=http://127.0.0.1:18500
export REPO=$HOME/w/Strata
MODEL="$(curl -s -H "Authorization: Bearer $STRATA_KEY" "$URL/v1/status" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("model") or "q2_0")')"
```

Start a sustained decode (one paste; ~512 tokens ≈ 40–60 s at the recorded 8–16 tok/s):

```bash
curl -sN "$URL/v1/chat/completions" -H "Authorization: Bearer $STRATA_KEY" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"stream\":true,\"max_tokens\":512,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Summarize the plot of Moby-Dick in five paragraphs, with detail.\"}]}" | cat
```

While it streams, in the monitor: press `5` (Hardware) — the Metal card row shows `temp <N> C` — then `1`
(Overview) — the GPU-temperature tile and its sparkline move. Screenshot both (task chat). Capture the
telemetry lines (third terminal, one paste; run during the decode):

```bash
mkdir -p $REPO/docs/m2-sessions/2026-10-25-session-4
for i in 1 2 3 4 5 6 7 8; do
  curl -s -H "Authorization: Bearer $STRATA_KEY" "$URL/v1/telemetry" | python3 -c '
import datetime, json, sys
d = json.load(sys.stdin)
dev = (d.get("devices") or [{}])[0]
print(datetime.datetime.now().strftime("%H:%M:%S"), "temp=", dev.get("temp"), "util=", dev.get("util"),
      "mem_used=", dev.get("mem_used"))'
  sleep 1
done | tee -a $REPO/docs/m2-sessions/2026-10-25-session-4/telemetry.txt
```

And once, the note (the truthful rootless wording from PR #24's `bb3e2304`):

```bash
curl -s -H "Authorization: Bearer $STRATA_KEY" "$URL/metrics" | python3 -c '
import json, sys
print("gpu_note:", (json.load(sys.stdin).get("hardware_static") or {}).get("gpu_note"))' \
  | tee -a $REPO/docs/m2-sessions/2026-10-25-session-4/telemetry.txt
```

Expected: ≥ 3 `temp=` lines inside the decode window with a number (10–125 °C is the reader's plausibility
band), not `None`; the note says temperature comes from the rootless HID sensors. Optional one-time
cross-check (a machine-state capture, not an engine command; needs sudo):
`sudo powermetrics --samplers smc -n 1 -i 1000 | grep -i die` — the two readings should sit within a few °C.
Abort: every sample `None` during a decode that demonstrably ran (the request completed) → H2 FAIL: the
lines + the note + the engine log go to the archive, record, stop.

### S4 — H3: the hard server kill mid-decode (gate 3) — ~10 min

The server from S3 is still running. Second terminal, one paste to record the living tree (all three
processes, with start times):

```bash
pgrep -fl "serve.server"; pgrep -fl "build-metal/strata"; pgrep -fl "import os, signal"
```

Expected: one `serve.server` line (the server), one `build-metal/strata` line (the engine), one
`import os, signal, subprocess` line (the watcher holding the engine's pipe). Record the PIDs; copy this
paste's output into `gates/h3-hard-kill.txt` (the "before" block). Start the same sustained decode as S3
in a third terminal (same two pastes), and once tokens are streaming, kill the server the hard way (one
paste, the recorded server PID):

```bash
kill -9 <server-pid>
```

Then the after-checks (one paste at ~+5 s and again at ~+15 s):

```bash
pgrep -fl "serve.server"; pgrep -fl "build-metal/strata"; pgrep -fl "import os, signal"
strata launch --status
python3 -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',18500))==0 else 1)" && echo "port: OPEN" || echo "port: closed"
```

Expected at +15 s (the watcher's grace: SIGTERM, 10 s, SIGKILL): no `serve.server`, no `build-metal/strata`,
no `import os, signal` process — the engine child is gone, no orphan, the watcher exited clean; the managed
host stopped with its server (`strata launch --status`: stopped); the port closed. The TUI may stay open
showing a stopped instance — leave it. Copy every after-paste into `gates/h3-hard-kill.txt` (the "after"
blocks, with timestamps).
Abort: the engine still alive at +15 s, or any survivor of the three patterns → H3 FAIL: the before/after
pastes + `ps -o pid,ppid,lstart,command` of the survivor + the engine log tail go to the archive, record, stop.

### S5 — State capture (after) + results fill + the archive — ~15 min

One paste (S0's fields, after): date/time, swap after, a hot temperature reading
(`sudo powermetrics --samplers smc -n 1 -i 1000 | grep -i die` — machine state, not an engine command),
Chrome state. Fill the results table (§5) into `RESULTS.md`. Then the capture — the #26 amendment's
`mkdir -p` first:

```bash
cd $REPO && mkdir -p docs/m2-sessions/2026-10-25-session-4/gates
cp bench/results/2026-10-25-m2-q36-4k-prompt/bench.json docs/m2-sessions/2026-10-25-session-4/gates/h1-bench.json
cp bench/results/2026-10-25-m2-q36-4k-prompt/engine-prefill-512.log docs/m2-sessions/2026-10-25-session-4/gates/h1-engine-prefill-512.log
cp /Users/yebyen/w/Strata-data/strata-q2_0.log docs/m2-sessions/2026-10-25-session-4/engine.log
cp <runtime-dir>/<profile>.server.log docs/m2-sessions/2026-10-25-session-4/server.log 2>/dev/null || true
```

(`<runtime-dir>` prints with `python3 -c "from strata_control.runtime import runtime_dir; print(runtime_dir())"`;
the server log is the managed host's `<profile>.server.log` beside its state files.) Then:

- gzip any log over ~256 KiB and keep the `.gz` name (`gzip -k` first, drop the original only after the
  copy reads back).
- the key sweep before anything lands: `grep -rl "$STRATA_KEY" docs/m2-sessions/2026-10-25-session-4/`
  must print nothing — a hit means strip that file and re-copy; the launch's key never enters the archive.
- the app is not part of this session's seats (Terminal only) — record "app not used; app-diagnostics
  not captured" in RESULTS.md rather than creating an empty file.
Hard-stop check against start + 120 min: if the clock runs out, drop in order S3's optional cross-check →
S4 → S2's number-recording (never the S2 gate greps, never the table) — a dropped hypothesis is recorded
as not-run, never silent.

## 4. The clock

S0 5 + S1 10–20 + S2 ~6 + S3 ~15 + S4 ~10 + S5 15 ≈ **60–75 min wall clock** (the model load itself is
seconds; the bulk is the 4,201-token read and the two decodes). Hands-on owner time ≈ 15 min (the pastes
and two screenshots); the rest is watching. Hard stop: **start + 120 min**.

## 5. Results table — fill at session time (S5)

`RESULTS.md` beside this runbook mirrors this table and is the archived copy. Every verdict is traceable
to a file in the session dir.

| # | Step | What it records | Verdict / value |
|---|---|---|---|
| 1 | S0/S1 | base SHA, dirty count, STRATA_LIFELINE, fix-in-tree grep, build + compile test | |
| 2 | S2 (H1) | the round line verbatim: ttft (prefill seconds), prompt tok/s, decode tok/s; the four grep counts; PASS/FAIL | |
| 3 | S3 (H2) | the decode driver's tok/s glance; the count of temp-bearing samples in the decode window; the min–max °C; the gpu_note line verbatim; rendered-in-monitor (screenshot yes/no); PASS/FAIL | |
| 4 | S4 (H3) | before/after pgrep blocks; time-to-engine-gone; watcher gone (yes/no); orphan found (yes/no); host state + port state; PASS/FAIL | |
| 5 | S5 | machine state after (swap, temp), artifacts landed (list), key sweep result | |

## 6. Notes

- The sandbox pre-gates recorded beside this runbook (`gates/pregate-*.txt`, 2026-10-25, checkout
  `af38e089`): the lifeline suite 6/6 (`serve.test_lifeline`, the portable-watcher path included), the
  Mac-telemetry suite 5/5 (`serve.test_server.MacTelemetry`, `test_thermal_reading_apple_silicon`
  included), and the fix-in-tree static check. They prove the wiring on main; they do not prove the
  hardware — S2–S4 do.
- The mechanism map (serve/lifeline.py): Windows job object, Linux PR_SET_PDEATHSIG, macOS and every other
  POSIX the portable watcher — the `python -S -c` process holding the pipe. On the M2 the watcher is the
  production path; PR #25 fixed its identity token (start time only — the state letter raced R↔S and made
  the watcher exit without killing anything, the exact orphan H3 must not show).
- H2's reading path on the M2 (one GPU): `serve/telemetry.py` `_AppleSiliconThermal` (IOHIDEventSystemClient,
  rootless) → `_Metal.read()` `temp` → the sampler's `gpu_temp` (1 s cadence) → `/v1/telemetry`'s
  single-device row (`devices[0].temp`) → the monitor's tile/Hardware row. The Overview's temperature tile
  keeps a 60-sample history (the sparkline).
- The monitor is the client; the sampler is the server's — a temp value can render between requests too.
  The gate samples during a decode so the "live during sustained decode" half of H2 is real, and the GPU
  tile's utilization is the witness.
- No API key in an artifact, ever; the curl pastes take the key through `$STRATA_KEY` only.
- Session #3's findings (the empty-`STRATA_API_KEY` spawn bug, the catalog filter) are merged — this
  session runs on their fixes; nothing here re-opens them.
