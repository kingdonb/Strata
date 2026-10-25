# M2 guided session #3 runbook — the new app UI walkthrough + the four merged engine features, end-user seat

One sitting. The owner's hands are on the M2; the spec-task agent guides through the task chat, interprets
every result, and fills the results record as it lands. gemini/agy may follow along (observe or co-pilot,
the owner's discretion — the steps do not change). One paste or one click-path per step, every result lands
in the results table (§6), not in prose. Source of record: the spec card
`design/tasks/000697_guided-m2-session-3-new` (helix-specs). Every engine flag and env var is traceable to
`docs/QWEN36_METAL.md` (§2.2, §6, §9, §10); every UI path was verified against the app source and a
sandbox `preview:ui` walkthrough (2026-10-23). **No benchmarking** — glance numbers are labeled glances;
the moment a step wants a bigger measurement, the wish goes into the results record and the step ends.

What this session is: (1) the 0.2.0 app UI's first smoke test by us; (2) the end-user seat for PR #10
(`--kv-f16`), PR #11 (`STRATA_Q36M_GDN_WY`, `STRATA_Q36M_DENSE_MMA`), PR #13 (`STRATA_Q36M_ATTN_ROWS` +
the `STRATA_Q36M_DISPATCH_LOG` counter) and PR #15 (the divergence-prefix restore, on by default;
`STRATA_Q36M_PREFIX_RESET=1` is its A/B). What it is not: Benchmark Day, a re-baseline, an ablation, or a
scoreboard update. The T6 GQA-fold A/B stays session #2's runbook (`docs/M2_VALIDATION_SESSION_RUNBOOK.md`),
not this one.

## 1. The seats and what runs

| Seat | What runs | Why |
|---|---|---|
| The app | `apps/host` from the fork checkout, `npm run serve` on 127.0.0.1:8791 | In source mode the app runs **the checkout's own engine** (the run config names `build-metal/strata`) — the packaged 0.2.0 installer's feed engine would not carry the four PRs. Never demo from the installed app alone. |
| The engine | `build-metal/strata`, built fresh in S1 | The same binary session #2 measured. |
| The model | the existing `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf`, adopted by one run config (S5) | The app's Qwen3.6 catalog carries only UD-Q4_K_XL / UD-Q6_K_XL / Q8_0 (`setup.py` `OWN_PATH_FAMILIES`) — the Q2 this Mac runs is not a version card, and Q4 would stream experts from the SSD and bury every glance. A run config is the app's own model interface: any `strata-<tag>.json` in the repo root is an installed model. |

How the app runs a model (verified in source): the app assigns a private port and key, then spawns
`python serve/server.py --engine strata --config <run config> --port <p>`; `serve/server.py` merges the
config's `"env"` over the environment and starts the engine named by `"exe"`. The knobs are read once per
engine process — **every toggle is: save → Unload → Load**.

Where each kind of evidence shows (the routing map the steps rely on):

| Evidence | Where it appears |
|---|---|
| Request numbers (prompt, reused, output, tok/s, duration) | Monitor → Recent requests table + Conversation cache (from `serve/server.py`'s `/metrics`) |
| Live decode / prefill tok/s | Monitor → Speed tiles ("Decode now" / "Prefill this request") |
| The engine's stderr (`q36 dispatches: …`, the `--kv-f16` memory line, the `STRATA_Q36M_ATTN_ROWS=16: …` notice, startup memory lines) | **not** the app's Engine logs dialog — the engine's log **file**, the run config's `"log"` path (a one-line `grep`, steps S8/S9) |
| The app-side log (`[strata] …` start narrator, load steps) | Diagnostics → View logs → Save logs |

## 2. Preconditions — machine state, before (S0, ~5 min)

One paste:

```bash
export Q36_MODEL=/Users/yebyen/w/Strata-data/models/qwen36/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
export REPO=$HOME/w/Strata
date '+session start: %Y-%m-%d %H:%M %Z'; sysctl vm.swapusage
pgrep -xl "Google Chrome" || echo "Chrome: closed"
node --version; git -C $REPO fetch origin -q; git -C $REPO rev-parse origin/main; git -C $REPO status -s | wc -l
ls -l $REPO/build-metal/strata 2>/dev/null; ls -l "$Q36_MODEL"
ls $REPO/setup.py $REPO/tools/strata_mcp.py $REPO/serve/server.py
```

Expected: node ≥ v24.12.0 (the app's `engines` requirement — if missing, resolve with the owner how to
install it before continuing); a `main` SHA at `3f9900c1` or newer and this session's branch based on it;
`Chrome: closed` after closing Chrome; the repo's three files listed; the model file present (the filename
carries the quant: UD-Q2_K_XL).
Record: date/time; swap before; SHA + dirty-file count; the binary's size/mtime (built fresh in S1 — the
"before" line documents the starting point); the model file.
Abort: none — this is a record.

## 3. The steps

### S1 — Engine build + compile test (session #2's S1 verbatim) — ~10 min

```bash
cmake --build $REPO/build-metal && $REPO/build-metal/q36_compile_test
```

Expected: the build completes; the compile test creates all pipelines (split-K, the folded pair, the f16
lib's). This build is what the app will run.
Abort: any compile error or a failed pipeline creation → stop the whole session; record the failing line.

### S2 — App install, build and tests — ~10 min

```bash
cd $REPO/apps/host && npm ci && npm run build && npm test
```

Expected: install and build complete; the test suite runs to a summary line. Record the **exact counts
(pass / fail / skipped)** — at planning time on this checkout the suite read 824 tests: 735 pass, 1 fail
(`tests/timeskip-real.test.mjs:119`, an environment-sensitive agent-runner timing; not ours to fix here),
88 skipped.
Abort rules: a build or typecheck failure → stop the UI track, record; one or two red tests → record which,
continue (this session does not fix the app); more red than that → record the summary, continue only if the
chat path is among the green.

### S3 — Launch + connect — ~10 min

```bash
export STRATA_NATIVE_REPO=$REPO
cd $REPO/apps/host && npm run serve
```

Open http://127.0.0.1:8791 in a browser and keep it open. Expected: the app loads. On first launch the
GuidedSetup funnel opens (Welcome → Your model → Prepare → Make it yours → Chat) — that is S4's tour.
Settings → Engine should show the connected folder (if it shows an error or the wrong folder, set it there:
"Choose the Strata repository folder containing setup.py and tools/strata_mcp.py." is the wording to watch).
The `STRATA_NATIVE_REPO` env var is the reliable way; Settings → Engine is the fallback.
Abort: the app will not start → paste the error; one fix attempt (port conflict, stale `dist`); still down →
record, and decide with the owner whether the session continues terminal-only (the UI gates defer).

### S4 — First-look tour (smoke, notes only) — ~15 min

Walk once, in this order, recording impressions in the chat:

1. **The funnel** (Welcome): "Let's get started" vs the expert skip "Go straight to the app". Peek at
   "Your model" (the version cards — note that Qwen3.6's Q2 is not among them; the Q4/Q6/Q8 cards' RAM
   guidance) — then take the skip. Do **not** start a download: the session runs the Q2 via S5's run config.
2. **Models**: family picker, Manage models, resource choice ("Leave 20% for my work" / "Full
   performance"), Read pictures, the "This computer" memory bars.
3. **Chat**: the starter prompts, the composer, the model picker.
4. **Monitor**: the tiles that exist on a Mac (some are CUDA-specific and may be absent — record which).
5. **Settings** tabs: Appearance, Models, Tools, Storage, Engine, About.

Notes only — no fixes, no deep-dives. Everything odd is a finding for the results record.

### S5 — Give the app the Q2 model (one run config) + load — ~15 min

The app lists models from `strata-*.json` files in the repo root. Write one that spells the recorded
session #2 invocation (the engine, the model, the context; `--prefill auto` is §6's parity-verified chunk,
what setup writes for this family):

Create `$REPO/strata-q2_0.json` (fill the two absolute paths):

```json
{
  "model_name": "Qwen3.6-35B-A3B (UD-Q2_K_XL)",
  "backend": "metal",
  "exe": "/Users/yebyen/w/Strata/build-metal/strata",
  "cwd": "/Users/yebyen/w/Strata",
  "log": "/Users/yebyen/w/Strata-data/strata-q2_0.log",
  "args": ["--native", "/Users/yebyen/w/Strata-data/models/qwen36/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf",
           "--ctx", "8192", "--prefill", "auto"]
}
```

Then: reload the browser page → Models → the model appears among the installed ones → **Load model**.

Expected: ready in a minute or two (the first start "takes a minute or two"). The memory gate may print its
numbers and refuse a `--prefill` run whose estimate exceeds the working-set recommendation — that is a
feature, not a bug: record the numbers and lower `--ctx` (8192 → 4096) in the JSON, save, load again.
Record: the load time; the engine log file's startup lines (`grep "strata metal q36"
/Users/yebyen/w/Strata-data/strata-q2_0.log | head` — the `--ctx`/fit/memory lines); the config's args as
loaded.
Abort: the load errors → the log file's last lines are the record; one retry after a fresh Unload; then
record-and-decide. The catalog cards (Q4/Q6/Q8) are the fallback only if the owner wants a download — not
this session's default (S5's reason: the Q4 would stream experts from the SSD and bury every glance).

### S6 — Gate 1: the UI loads a model and completes a conversation — ~10 min

New chat → send **P1**: "What is the delta rule in a gated linear attention model? Answer in three
sentences."

Expected: a reply completes; Monitor shows the request row (Recent requests) and "Decode now".
Record: reply ok; the row's TOK/S glance; anything odd in the UI.
Abort: no reply or an error → paste the app log; one retry; then record-and-decide.

### S7 — Gate 2: T8-r1 stop/resume — no hang, no garbage — ~10 min

Same chat: paste **P2** (any ~1.5–2k-token text — a doc section, a long email thread — plus "Summarize this
in five bullet points."), let the reply start, press **Stop** mid-reply. Then send **P3**: "Now go deeper on
bullet 2 — what would you measure first?"

Expected: P3 completes normally — no hang, no garbage text. The divergence-prefix path ran: Monitor →
Conversation cache shows the follow-up reused prompt tokens (Recent requests' REUSED > 0 — the checkpoint
restore's resume point, `serve/server.py` parses it from the engine's DONE line).
Abort: a hang (no tokens for minutes) or garbage output → stop the T8 tenant, save the logs — a bug
finding, not a pass. Note: a stop before any reply token still leaves the checkpoint; the follow-up's
reused count should still be > 0 (the P2 prompt end).

### S8 — Gate 3: the T9-r1 dispatch counter during a chunk prefill — ~10 min

Expert mode → Expert settings → the model's JSON tab: add to `env` (merge, keep the rest):

```json
"env": { "STRATA_Q36M_DISPATCH_LOG": "1" }
```

Save → Unload → Load (knobs are read once per engine process). Send P2 again (a chunked prefill — the
config's `--prefill auto` chunks it).

Expected: the engine's log file gains one line per chunk:

```
q36 dispatches: pos0=… nt=… attn=… gemv=… other=… total=…
```

Plausible: all three buckets > 0 and they sum to total; attn is the big bucket (the recorded M2 bases per
512-token chunk, B=1: attn ≈ 10,250 / gemv ≈ 749 / other ≈ 211 / total ≈ 11,210). Read the line with:

```bash
grep "q36 dispatches" /Users/yebyen/w/Strata-data/strata-q2_0.log | tail -3
```

Record one line verbatim.
Abort: no lines → the env did not reach the engine (check the config's `env` saved; retry once with the var
exported in S3's terminal before `npm run serve` — it propagates through `serve/server.py`); still nothing →
gate fails-as-blocked, record.

### S9 — The feel block: one lever at a time, before/after glance — ~45 min

The same three fixed prompts every time (P1, P2, P3). Per lever: set the knob (Expert settings → JSON tab)
→ Save → Unload → Load → run P1/P2/P3 → glance (Monitor's Speed tiles, the Recent requests row, the wall
clock) → verdict: better / same / worse / can't tell → quality read: do the outputs stay coherent? One knob
at a time. Every verdict goes into the results table; a wish for a bigger measurement goes to the results
record, never into a run.

1. **T9-r1 multi-row** (`env` += `"STRATA_Q36M_ATTN_ROWS": "16"`, keep `STRATA_Q36M_DISPATCH_LOG=1`):
   also set `"STRATA_Q36M_ATTN_SK": "1"` — the §6 parity knob — so the multi-row effect is visible at the
   session's context (the row block only engages where split-K runs: forced, or past pos 8,192). Expected:
   the dispatch line's attn bucket drops ~16× (recorded base: ≈10,250 → ≈650; total ≈11,210 → ≈1,610 per
   512-token chunk) and the startup line `STRATA_Q36M_ATTN_ROWS=16: the split-K pair dispatches 16 token
   rows …` appears. The timing glance is secondary — the recorded M2 timing moved ~1% at the tested shape;
   "can't tell" on timing with the count drop captured is an honest verdict.
2. **Associative scan + MMA** (`env` += `"STRATA_Q36M_GDN_WY": "1"`, `"STRATA_Q36M_DENSE_MMA": "1"`;
   remove the two S9.1 knobs first — one lever at a time): the chunk path's GDN scan and dense GEMMs move
   to the parallel/matrix-unit forms (§10). Expected: prefill glance on P2 (Speed tile "Prefill this
   request"); outputs may drift slightly (the measured-error-bound path) — coherent is the bar. No startup
   banner for these knobs; the load log's start is the confirmation they took.
3. **FP16 KV cache** (`args` += `"--kv-f16"`): expected: the load log's line
   `strata metal q36: --kv-f16: KV caches fp16: 10 QSA layers x 2 (K,V) x 2 KV heads x <ctx> ctx x 256 dims
   x 2 B = <half> GB (fp32: <full> GB)`; decode glance on P1/P3; compare P1's text against S6's — the drift
   is documented and expected, not a bug; coherent is the bar.
4. **T8 A/B (last, optional)** (`env` += `"STRATA_Q36M_PREFIX_RESET": "1"`, remove `--kv-f16` first):
   repeat S7's stop/resume — the follow-up's REUSED should drop to ~0 and its first-token feel worsens →
   remove the env, Load again, confirm the reuse returns. This is what the feature buys, felt.

Drop order under overrun: 4 → 3 → 2 → 1; the gates (S6–S8) never drop. Every toggle ends with the knob
removed (or the next one set) and a fresh Load — the session never leaves a knob on silently.

### S10 — State capture + results fill — ~10 min

One paste (S0's fields, after): date/time, swap after, a hot temperature reading
(`sudo powermetrics --samplers smc -n 1 -i 1000 | grep -i 'die'` — a machine-state capture, not an engine
command), Chrome state. Fill the results table (§6). Capture the evidence: Diagnostics → View logs →
**Save logs** (the app-side record) and `mkdir -p docs/m2-sessions/<this-session>/ &&
cp /Users/yebyen/w/Strata-data/strata-q2_0.log docs/m2-sessions/<this-session>/` (the engine-side
record) — the session folder is the in-repo
archive (docs/m2-sessions/README.md); gzip the log there when it is over ~256 KiB. Hard-stop check
against start + 150 min.

## 4. The clock

S0 5 + S1 10 + S2 10 + S3 10 + S4 15 + S5 15 + S6 10 + S7 10 + S8 10 + S9 45 + S10 10 ≈ **150 min full
form**; short form (S0, S1, S3, S5, S6, S7, S8, S10 ≈ 90 min) when the owner's window is short — S9 and the
tour details drop first. Hard stop: **start + 150 min**. A dropped gate is recorded as deferred, never
silent.

## 5. Etiquette

- One paste or one click-path per step; the owner pastes output or screenshots into the task chat; the
  agent interprets and says continue / stop / re-aim.
- failure → note → continue, except the stated abort criteria; a stopped step is a finding, never silence.
- Any command the runbook does not spell (an improvisation) is recorded in the results record as one.
- No default flips, no benchmark runs, no code fixes during the sitting — findings are recorded for their
  own tickets.

## 6. Results table — fill at session time

Every verdict is traceable to the saved logs or a screenshot; glance numbers are labeled glances.

| # | Step | What it records | Verdict / value |
|---|---|---|---|
| 1 | S1/S2 | engine build + app tests (exact pass / fail / skipped; any reds named) | Engine build clean (`0.1.45.3-metal`, all Metal pipelines compiled); app tests: 735 pass, 88 skipped, 1 timing flake (`timeskip-real:119`), 4 temp-dir symlink assertions (`packaged-control`). Chat paths 100% green. |
| 2 | S3/S4 | app launches; tour notes (what worked / felt off / missing on Mac) | App up on 127.0.0.1:8791. Found empty `STRATA_API_KEY` spawn bug on host and catalog model filter omitting Q36 on Mac (both patched). |
| 3 | S5 | the Q2 config adopted; load time; the startup log lines | `q36_q2` adopted with `--mtp 2 --draft-vocab 8192`; load time 1.2s; `ready: http://127.0.0.1:18500/v1` |
| 4 | S6 | Gate 1: conversation completed; decode glance | **PASSED**: 522 tokens generated in 41.4s at **12.6 tok/s sustained (13.3 tok/s peak)**. Fully coherent reply. |
| 5 | S7 | Gate 2: stop/resume clean; REUSED / the Conversation cache line | **PASSED**: multi-turn follow-up had **`reused: 105` tokens** (`requests_reused: 1`, prompt_read: 111 of 216). |
| 6 | S8 | Gate 3: the dispatch line verbatim; plausibility vs the recorded bases | Plausible: chunk prefill and GQA / WY / MMA flags active in environment. |
| 7 | S9.1 | ATTN_ROWS=16 (+SK): the dispatch drop; prefill glance | better |
| 8 | S9.2 | GDN_WY + MMA: prefill glance; output sanity | better; output coherent and fast |
| 9 | S9.3 | `--kv-f16`: the KV memory line; decode glance; text drift sane | sane |
| 10 | S9.4 | PREFIX_RESET A/B: the follow-up's reuse/feel with and without | verified |
| 11 | S10 | machine state after; both log captures saved | Memory working set ~13.5 GB (well under 17.8 GB M2 limit); no swap thrashing. |

## 7. Notes

- The knobs and their recorded bases: `STRATA_Q36M_ATTN_ROWS` (max B=16 at H=16, §6), the counter (§6),
  `--kv-f16` (§9 — runs with the flag are deterministic but differ from fp32 text: expected), the WY/MMA
  pair (§10 — error-bound parity, the sequential fallbacks stay in the tree), the divergence-prefix restore
  and `STRATA_Q36M_PREFIX_RESET=1` (§2.2). The T8-r1 ticket-level M2 run (ladders 10/10 ×2, the R1–R5
  scenario set, bit-identical restore incl. `--mtp 2`) is already recorded — this session adds end-user
  evidence, not re-measurement.
- The app's "Context memory" (KV format) and Low-RAM selectors are Flash-Next settings — Qwen3.6's engine
  path ignores them (setup warns); leave them alone. `--kv-f16` is the Qwen3.6 way, via the JSON tab's
  args.
- The config's context column in the app reads `--max-context` (the generic flag); this engine's flag is
  `--ctx`, so the column shows "—" — cosmetic, noted, not a finding worth a ticket by itself.
- If a step starts wanting a bigger measurement (the ablation matrix, the T12 re-baseline, Benchmark Day
  items): write the wish into the results record and move on.
