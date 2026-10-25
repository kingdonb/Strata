# M2 validation session runbook — session #2 (T6 fold A/B, T8 divergence checks)

One sitting, one paste per step, every measured number lands in the results table (§5), not in prose.
Source of record: the spec card `spt_01m5bgj85gewt1qgve81w25r8z` (helix-specs
`design/tasks/000690_validation-session-1/design.md`). Every command here is the recorded one —
`docs/QWEN36_METAL.md` §5 (build, model path), §6 (ladder, bench tool, memory gate), §8 (the T6 gate
commands) — plus ticket 000688's interactive-M2 list for T8. No invented engine flags; the only argument
trims are the ones §6's bench note blesses ("cut --rounds or the size list for a first look"), and each
trimmed run is labeled with its fidelity. Scoreboard cells are `docs/QWEN36_METAL_REVIEW.md` §11's; the
machine-state fields are §3's (the confound lesson: 14.4 tok/s was a hot, swapping, contention number —
state is recorded in both states so no number is ever mislabeled again).

## 1. The tenants and what runs

| # | Tenant | Condition | Decides (review §7) |
|---|---|---|---|
| 1 | T6's GQA-fold A/B — the headline | always (the fold is merged, opt-in knob) | the fold's flip decision (§7 block 2) |
| 2 | Gate-2 fold parity | always — the record must exist even on a "no flip" | required green for any scheduled flip |
| 3 | T8-r1 ladder ×2 + `q36_prefix_reuse.py` | only if T8-r1 is merged (the tool landed on its branch 2026-10-20, `ac5e0169`) | the divergence-prefix change proceeds iff parity holds twice (§7 block 3) |
| 4 | T10-r0 rows-per-expert histogram | only if T10-r0 landed (its delivery note carries the command) | the T10 hybrid threshold's data (§13) |
| 5 | T9-r1's before/after | done — cited, no re-run (PR #13) | — |
| 6 | E1, the Unsloth model-load log | owner's screenshot, last | §11's "MTP in the benchmark session" cell |

E2 (the clean 32k rerun) is **not** this session — Benchmark Day content (§7 block 4).

## 2. Preconditions — machine state, before (S0, ~5 min)

One paste:

```bash
export Q36_MODEL=/Users/yebyen/w/Strata-data/models/qwen36/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf
date '+session start: %Y-%m-%d %H:%M %Z'; sysctl vm.swapusage
pgrep -xl "Google Chrome" || echo "Chrome: closed"
git rev-parse HEAD; git status -s | wc -l
ls -l build-metal/strata; ls -l "$Q36_MODEL"
```

Record here: date; swap before (`allocated`/`used`); Chrome open/closed (close it if open); git SHA +
dirty-file count; the binary's size/mtime; model file + quant (the filename carries it: UD-Q2_K_XL).
Temperature, cool reading: `sudo powermetrics --samplers smc -n 1 -i 1000 | grep -i 'die'` or the usual
reader — a machine-state capture, not an engine command; repeat after the runs (§4).

Expected: the date line; a `swapusage` line (record it either way); `Chrome: closed` after closing; a SHA
equal to the day's `origin/main` merge; the file sizes.
Abort: none — this is a record, and the A/B is self-paired (both arms run in this same machine state, so
contention shifts both arms together). Only a swap already deep in use is an owner's-call reschedule
suggestion, recorded, not a stop.

## 3. The steps

### S1 — Build + compile test (§5's recorded form; §8 gate 1) — ~10 min

```bash
cmake --build build-metal && ./build-metal/q36_compile_test
```

Expected: the build completes; the compile test creates all pipelines, the three folded ones included.
This one build serves every tenant.
Abort: any compile error or a failed pipeline creation → stop the whole session (no tenant runs on a broken
build); record the failing line.

### S2 — T6 GQA-fold A/B, fold off then on (§8 gate 4's commands) — ~30 min

The §8 gate-4 command records `--sizes 2048,4098 --chunks 0,512 --rounds 3`. The session's primary run
trims it to the headline cells exactly as §6 blesses ("cut --rounds or the size list for a first look"):
`--sizes 4098 --chunks 0,512 --rounds 1` — the 4,098-token read at 16K and decode tok/s at pos ≈4,100,
fold off vs on. The §8-verbatim full form is S2d below, run only if the verdict needs it. Arms run
off-then-on (the recorded order); this ordering thermally biases against the fold-on arm (it runs hotter),
so a positive ratio is robust and an ambiguous one gets deepened.

```bash
mkdir -p bench/results/<date>-m2-q36-gqa
python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 16384" --sizes 4098 --chunks 0,512 --rounds 1 --gen 128 \
  --out bench/results/<date>-m2-q36-gqa/bench-off.json
```

```bash
STRATA_Q36M_ATTN_GQA=1 python3 tools/qwen35/q36_prefill_bench.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 16384" --sizes 4098 --chunks 0,512 --rounds 1 --gen 128 \
  --out bench/results/<date>-m2-q36-gqa/bench-on.json
```

Expected per arm: `chunk C: warm-up done (load X.Xs)` (the tool prints the model load time — record it);
round lines `chunk C size 4098 round 1: ttft ... s, prompt ... tok/s, decode ... tok/s`; a markdown table
(TTFT + prompt tok/s) and the `>=5x` gate print — read the tok/s numbers, not the 5x verdict (that gate is
the prefill-default question, not this A/B); every raw run lands in the JSON. The decode tok/s at
pos ≈4,100 is the round line's `decode` field (128 generated after the 4,098 read). Recorded bases for
sanity: tbt read ≈321 s / chunk-512 ≈225 s (§6's 4,096 rows), decode ≈8 tok/s pre-fold (§11), so decode
≈16 s per run.
Abort: `chunk C: REFUSED before READY` (unexpected at 16K on 24 GB — the 2026-10-18 matrix ran 32K) →
stop, record the log line; the engine dies mid-request → stop; the fold-on start hits §8's startup guards
(H/HKV or threadgroup ceiling — they print their numbers and refuse) → stop the tenant, that is a finding,
not a measurement; the tool's reused-prefix refusal → stop (seeded ids should prevent it). A clean run with
a wild number (decode >40 or <2 tok/s) → one retry of that arm, then record whatever stands with a note.
Continue to S3 either way — the verdict is recorded with its fidelity label, never silently.

### S2d — conditional deepen (only if the S2 ratio lands in the 1.2–1.5× band AND ≥2 h remain) — ~110 min

The §8 gate-4 commands verbatim, both arms (`--sizes 2048,4098 --chunks 0,512 --rounds 3 --gen 128`, out
`bench-off-r3.json` / `bench-on-r3.json`). Otherwise skip and record: "rounds-1 fidelity, band ambiguous,
deepen scheduled" — the conservative default (§8: the default stays byte-identical until a measurement says
otherwise).

### S3 — Gate-2 fold parity (always; §8 gate 2 verbatim) — ~10 min

```bash
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-base.json
STRATA_Q36M_ATTN_GQA=1 python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-gqa.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-base.json /tmp/q36-gqa.json
```

Expected: `identical: 10 of 10` (the fold forced at 4K token-identical to the default path), on this
session's build and day (§8's same-Mac/same-day/same-build rule).
Abort: a miss → record and stop: the flip defers (the knob stays opt-in — §7 block 2's rule requires all
gates green to schedule one), and the miss itself is a finding for the fold's ticket.

### S4 — T8-r1 ladder parity ×2 (only if T8-r1 is merged; 000688 AC 6) — ~10 min

Run early at the ticket level 2026-10-20 (build `ac5e0169`, results in §5 row 6): **10/10 twice, green** —
session #2 may skip S4/S5 or re-confirm post-merge.

```bash
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-t8-a.json
python3 tools/qwen35/q36_solo_gate.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out /tmp/q36-t8-b.json
python3 tools/qwen35/q36_cmp_gate.py /tmp/q36-t8-a.json /tmp/q36-t8-b.json
```

Expected: two fresh-process ladders; `identical: 10 of 10`. No pre-change reference JSON exists under
`bench/results/` (only `2026-10-18-m2-q36-prefill/bench.json`), so the same-build ×2 comparison is the
check.
Abort: not 10/10 → the divergence-prefix change does not proceed (§7 block 3's gate: it proceeds "if the
10-prompt parity holds twice") — record and stop the T8 tenant.

### S5 — T8-r1 reuse measurement (same precondition; 000688 AC 8) — ~10 min

The tool landed on the T8-r1 branch 2026-10-20 (`q36_prefix_reuse.py`, CLI `--exe --args --out [--n 48]
[--mtp]`). Confirm with `--help` at session time (the paste is pre-filled from the landed CLI, not
invented), then one paste:

```bash
mkdir -p bench/results/<date>-m2-q36-prefix
python3 tools/qwen35/q36_prefix_reuse.py --exe ./build-metal/strata \
  --args "--native $Q36_MODEL --ctx 4096" --out bench/results/<date>-m2-q36-prefix/reuse.json --mtp
```

Expected: the tool drives three engine processes itself and prints per-scenario counts + JSON. Base set —
R1 `reused 0`; R2 (strict extension) `reused = |P|+|G1|`, the T0 delta-only path; R3 (divergence restore)
`reused = |P|`; R4 (exact re-send) `reused = |P|`, read 0, tokens == R1's; R5 (early divergence)
`reused 0`; a fresh-process R3 byte-matches the in-session R3 (the restore rebuilds exact state). `--mtp`
set — the kept tokens still match the fresh full evaluation, and R4 still reproduces R1. RESET A/B
(`STRATA_Q36M_PREFIX_RESET=1`, the tool's third process) — R3/R4/R5 read fully again (`reused 0`; R2 keeps
the T0 extension) and its R3 matches the fresh R3 (reset and restore agree on the tokens). R2's `reused`
count carries the phantom stop token's signature — recorded as the follow-up ticket's evidence, not fixed
here.
Abort: tool missing → skip the T8 tenant, record "not landed"; a count pattern that contradicts the above
→ stop — it is a bug finding, not a measurement to record as pass.

### S6 — T10-r0 rows-per-expert histogram (conditional) — ~5 min

No command is written here — the histogram's command comes from T10-r0's delivery note when it lands
(nothing is invented). Expected: per-expert row counts for one 512-token chunk (and a 2,048-token chunk if
its note says so) — the T10 hybrid threshold's authoritative data (review §13).
Abort: not landed → skip, record "T10-r0 not landed".

### S7 — E1, the owner-side quick capture (last, after all timed runs) — ~2 min

One screenshot of the Unsloth app's model-load log (review §8 E1): the `spec-type`/`n_draft` lines and the
`n_ctx`/fit lines visible. Fills §11's "MTP in the benchmark session" cell ("open by one capture"). Done
last because loading the other engine heats and contends the machine — the §3 lesson applied to the sitting
itself.

### S8 — State capture (after) + results table fill — ~10 min

One paste (same as S0's minus the export): date/time, swap after, temperature (hot reading), Chrome state.
Then fill the results table below. Capture the evidence: `mkdir -p
docs/m2-sessions/<this-session>/ && cp` the engine log (the run config's `log` file)
and the gate outputs (the solo/cmp gate JSONs, the chunk-parity output) into
`docs/m2-sessions/<this-session>/` — the in-repo archive (docs/m2-sessions/README.md). Hard-stop check: if
the projected end passes start + 90 min, drop the
remaining steps in the order S2d → S6 → S5 → S4 → S3 (the A/B and this table are never dropped; gate-2 S3
survives longest — if even it must drop, record "gate-2 parity deferred" explicitly, never silently).

## 4. The clock

S0 5 + S1 10 + S2 ~30 + S3 10 + S4 10 + S5 10 + S6 5 + S7 2 + S8 10 ≈ **75–90 min wall clock** with the
conditional tenants (S4–S6) run and S2d skipped; ~55–65 min with them skipped (the short form: S0 → S2 →
S3 → S8 — gate-2 stays in every form); the S2d deepen adds ~110 min and only runs when ≥2 h remain.
Hands-on owner time ≈15 min (the pastes and one screenshot); the rest is watching — the agreement's
"minutes-scale contact". Hard stop: **start + 90 min**.

## 5. Results table — fill at session time (S8)

Every number is traceable to a raw JSON in `bench/results/`. After the session, each tenant's own PR copies
its cells into the recorded docs (T6's numbers → `docs/QWEN36_METAL.md` §8's "Measured on the M2: pending"
paragraph; T8's counts → §2.2; scoreboard edits → review §11), naming this card as the source.

| # | Step | Number it produces (exact field) | Scoreboard cell it fills (review §11) | Value |
|---|---|---|---|---|
| 1 | T6 fold-off (S2) | prompt tok/s on the 4,098-token read @16K; decode tok/s @pos≈4,100; TTFT | "Decode, 16k depth" — ours (updates the `8 @ ~4.1k (pre-fold)` entry; labeled with build + date + fidelity) | |
| 2 | T6 fold-on (S2) | same fields, fold on | same cell (fold-on arm); annotates "QSA KV bytes" (its A/B-pending note → the fold half measured) and "Bus efficiency class" (the measured bandwidth fraction at ~4.1k) | |
| 3 | A/B verdict (S2/S2d) | fold-on ÷ fold-off decode ratio; band per review §7 block 2 (≥1.5× schedule the flip behind the soak / 1.2–1.5× keep opt-in, record / <1.2× kill the KV-traffic hypothesis, §12.2) | the recorded decision line **feeding** the headline "Decode, 11.3k depth" cell (that cell stays pending E2 — the A/B is its decision input, not its measurement) | |
| 4 | Gate-2 parity (S3) | `identical: 10 of 10` (or the miss) | the fold's §8 gate record on this build/day — the flip decision's required-green line | |
| 5 | T9-r1 (cited — no re-run) | dispatches 11,210 → 1,610; TTFT 80.4 → 79.4 s | "Dispatches per 512-token chunk" — **already filled** (PR #13; `docs/QWEN36_METAL.md` §6's multi-row bullet) | 11,210 → 1,610 |
| 6 | T8 ladder + reuse (S4/S5) | reused/read per scenario; parity verdicts; checkpoint bytes | not a §11 row — lands in `docs/QWEN36_METAL.md` §2.2's measured record + ticket 000688 (the proceed/stop decision is review §7 block 3's) | **run early at the ticket level 2026-10-20** (build `ac5e0169`, ticket chat paste): ladder `identical: 10 of 10` ×2; checkpoint slot 66.9 MB; R2 reused 111/123 (delta-only, read 12 — carries the phantom's count signature), R3 reused 63 / read 36 and **bit-identical to fresh** (plain and `--mtp 2 --draft-vocab 8192`), R4 read 0 and == R1's reply, R5 full reset; A/B reads fully again, its R3 == fresh R3. Session #2 may skip S4/S5 or re-confirm post-merge |
| 7 | T10 histogram (S6) | rows per expert, one 512-token chunk | review §13's T10-threshold data (not a §11 cell) | |
| 8 | E1 (S7) | screenshot: spec-type/n_draft, n_ctx/fit | "MTP in the benchmark session" — the one-capture cell | |

## 6. Notes

- `q36_prefill_bench.py`: one engine start per chunk value; one discarded warm-up per start; per-(size,
  round) seeded ids so the prefix matcher never hides a read (it refuses a reused-prefix cell); the round
  lines are the only place decode tok/s is printed (the medians table has TTFT + prompt tok/s only); every
  raw run lands in the JSON; refusals are recorded as row notes; load time prints per start.
- The A/B is self-paired, so pre-existing machine contention shifts both arms — that is why S0 has no hard
  abort; the one real ordering bias (fold-on runs hotter) is recorded in S2.
- Machine-state captures are not engine commands — `sysctl vm.swapusage`, `pgrep`, `powermetrics` are
  standard macOS state reads; the no-invented-flags rule binds the engine's commands.
- The fold is merged (opt-in `STRATA_Q36M_ATTN_GQA=1`); §8's "Measured on the M2: pending" is what this
  session's S2/S3 numbers fill.
- Session #1's lesson is already merged (§6's multi-row bullet carries the 000685 numbers, PR #13) — cited,
  not re-run.
