# Proposal: a quick tuner for Inkling-2-Mini in `strata launch`

Status: proposal for the fork's owner. Nothing here is built. No benchmark was run for this text.
Written 2026-11-26 against fork `main` at `beda9ff42` (Strata 0.2.22, engine 0.1.69).

Every number below is labelled:

- **measured** = from the owner's hand sweep of 2026-11-23 and 2026-11-24 (ticket T-0036 in the owner's
  private notes). One machine: RTX 5060 Ti 16 GB, i5-10400, PCIe gen3 x8, 61.7 GiB RAM, Linux, Inkling-2-Mini
  SQ-10, engine 0.1.66.1 and 0.1.67.1.
- **estimate** = worked out from those numbers. Not measured.
- **read** = read from the code. Not run.

## The problem

`strata launch --tune quick` does nothing for Inkling-2-Mini. The tune step returns at once for a native
`.strata` engine (`terminal/strata_cli/launch/runner.py:2375`, "no Qwen tuning knobs"). That is true of the
Qwen knobs. But `strata-inkling` has its own: the number of request slots, the K/V memory format, the drafter
and how many experts live on the card. On the test machine the hand-picked settings were about twice as fast
in total as what a launch picks by itself.

| Setting, SQ-10, 80K window | Total tok/s, slots full | One request alone, prose | 53k-token prompt read |
|---|---|---|---|
| 1 slot, fp8 K/V | 75-78 | 80-84 tok/s | 1,372 tok/s |
| 4 slots, fp8 K/V | 149-151 | 71-74 tok/s | 1,142 tok/s |
| **4 slots, k8v4 K/V** | **148-150** | **82-86 tok/s** | **1,273 tok/s** |
| 8 slots, k8v4 K/V | 223 | not recorded | 966 tok/s |

All rows measured, drafting (DSpark) on. The hand sweep took 32 settings and about two hours, at about two
minutes a setting, because every setting was a full relaunch plus an HTTP benchmark.

What a launch picks by itself on that card today (read): one slot, because the table `PARALLEL_BY_VRAM` gives
one slot below 20 GiB (`terminal/strata_cli/launch/settings.py`), and the fp16 K/V, because `auto_fp8_kv`
keeps fp16 below 16 slots (`terminal/strata_control/inkling_fit.py:160`). The fp16 K/V read prompts about 8%
slower than fp8 at one slot (measured).

## What upstream has, and what it plans

Checked on 2026-11-26: tags `v0.2.21`, `v0.2.21.1`, `v0.2.22`, the one commit after it on upstream `main`,
`docs/DETAILS.md`, and a search of upstream's issues for "inkling tune" and "inkling bench parallel".

- **No Inkling tuner, and no plan for one that we could find.** The commits between 0.2.21 and 0.2.22 that
  touch Inkling are about thinking budgets, pictures, sampling and a card-split fix. `docs/DETAILS.md` says
  nothing on tuning Inkling. The issue searches returned nothing.
- **The Qwen tuner** (`strata tune auto`, `terminal/strata_control/tune_*.py`, `autotune.py`) drives the Qwen
  engine's knobs and saves a preset. It does not know `strata-inkling`.
- **`engine calibrate`** (`terminal/strata_control/calibration.py`) measures six Qwen settings
  (`--pcie-frac`, `--spec-min-p`, `--pool-workers`, `--adapt-every`, `--adapt-swaps`, `--adapt-decay`). Its
  check refuses any other setting, and it needs `tools/calibrate.py`, which takes 15 to 30 minutes.
- **The calibrated layer is general.** `terminal/strata_control/arg_layers.py` puts four sources over each
  other: this command line, the person's standing options, calibrated values, Auto. A calibrated record is
  `{"settings": {flag: value}, "model_name", "max_context", "images"}` in the run config's `"calibrated"`
  entry. `layer()` applies any flags it holds over Auto, never over the person's own. It is dropped when the
  model, the context or the image setting changes. Nothing in it is Qwen-only.
- **The drafting probe** (0.2.21, `terminal/strata_control/inkling_probe.py`) is the one thing a launch
  already measures for Inkling. Before the first start of a placement it starts the engine twice, plain and
  drafting, decodes a code prompt and a chat prompt, and turns drafting on only for a gain of 1.10x. It
  writes `"drafting_probe"` into the run config with a key for the placement. It has a small engine harness
  on the `--serve` line protocol that this proposal reuses.

- **Auto is careful about slots on purpose.** `fit_parallel_for` (`terminal/strata_cli/launch/runner.py:472`)
  gives Inkling no more slots than keep the K/V on the card, and cites upstream's own measurement: 3 slots at
  32K on a 24 GB laptop card, decode 86 tok/s falling to 44 after 30K tokens. A tuner does not argue with
  that default. It measures the same trade on the person's own card, when asked.

So the tuner would be the fork's own work. Under the fork's rule it is kept only while upstream has none.
If upstream ships one, ours goes.

## What the engine offers

`strata-inkling --bench --native FILE --max-context N --parallel P --kv F --prompt-len L --gen G`
(`engine/src/inkling2/ink_main.cpp:131`). It loads the model, fills **all** P slots with a made-up prompt of
L tokens, decodes G steps in all P slots, and prints one line:

```text
[bench] backend=… format=… parallel=4 gpus=1 kv=… prompt=… gen=… | prefill=… ppts (…s) | decode=… tps agg (… tps/slot, …s)
```

It needs no server, no tokenizer and no HTTP. Model load was 4.6 s on the test machine (measured, SQ-10).

Three limits, all read from the code:

1. **Every slot is busy.** `--bench` cannot measure one request alone beside idle slots, or one long prompt
   beside idle slots. Those two were the deciding numbers of the hand sweep.
2. **The prompt is made-up tokens.** Draft acceptance on them says nothing about real text. The drafting
   probe uses real prompts for that reason.
3. **The output is one text line**, and the K/V split between the card and host RAM is on a separate
   `[InkCUDA]` line on stderr.

## The proposal

### What the person sees

`--tune` already exists on `strata launch` and its default is `skip`. For Inkling-2-Mini:

- `strata launch … --tune quick` runs the quick sweep below, records the winner and starts with it.
- `--tune thorough` runs the wider sweep.
- No `--tune`: nothing changes. See "The default path".
- `--retune` measures again. `--tune-allow-small` (it exists: the "answer-changing" tick) lets the sweep
  also try K/V formats that change answers slightly.

One line before it starts, in the tuner's usual words, with the time it expects. One line after it, with the
gain and what was chosen. A sweep that fails never stops the launch: Auto runs, and the line says why.

The tuner needs the whole card. The launch already stops its own server for tuning
(`stop_own_server_for_tuning`), so the model is away for the length of the sweep. That is why it only runs
when asked.

### What it sweeps, and what it leaves alone

| Knob | Swept? | Why |
|---|---|---|
| Slots (`--parallel`) | Yes, first | The largest gain measured: 75 to 150 to 223 to 312 tok/s total at 1, 4, 8, 16 slots. |
| K/V format (`--kv`) | Yes, second, only with `--tune-allow-small` | It decides how much K/V stays on the card once slots share it. k8v4 against fp8 at 4 slots: 50,944 against 38,144 tokens on the card, the 53k prompt 11% faster. It changes answers slightly, so the person has to allow it. |
| Drafting (`--draft`) | No | The drafting probe owns it and uses real prompts. Measured: drafting only helps a request that runs alone, and the totals at 2 and 4 slots were the same with it on or off. The probe is run again when the K/V format changes, because its key holds `--kv`. |
| Experts on the card (`--vram-experts-gb`) | Thorough only | Measured: 6 GB instead of the automatic 8.06 GB cost far more than it gained (97 tok/s total at 4 slots), and 10 and 11 GB were no better. |
| Context (`--max-context`) | Never | The person's choice. A result is valid for one context. |
| `--kv-offload`, `--ram-experts-gb`, `--ssd-experts`, `--cpu-threads`, card split | Never in the first version | Automatic today. `--kv-offload off` cannot reach the engine through the launcher at all (see follow-ups). |

### What "best" means

More slots always gave more total speed on the test machine. They cost long prompts, because each slot keeps
only a window of its K/V on the card and reads the rest from host RAM over PCIe. So the tuner cannot simply
take the largest number. The rule proposed:

> Take the most slots, and then the K/V format, for which **one long prompt reads no more than 10% slower**
> and **one request alone decodes no more than 5% slower** than with one slot. A `--parallel N` on the
> command line is never changed: then only the K/V format is tuned for N slots.

Against the measured table this rule picks 4 slots with k8v4 when answer-changing formats are allowed
(1,273 against 1,372, 7% slower; no dip alone). It refuses 8 slots (966, 30% slower). With fp8 only, it stops
at 2 slots (1,282, 7% slower; 4 slots is 17% slower and dips alone). That matches what the owner chose by
hand. The two limits are open question 1.

A winner replaces Auto only if it is at least 5% better in total speed in two measurements. Otherwise the
tuner records "the standard settings are the fastest", as the Qwen tuner does (`tuned_at`).

### The quick sweep, in order

Each step is a new engine process, so one failed setting cannot harm the next. Each is started the way the
launch starts the engine: `inkling_kind.engine_args`, `child_env(cfg)`, the engine's own folder for its
libraries, stderr kept.

0. **Plan, no engine.** From `inkling_fit`: the slot ladder 1, 2, 4, 8, 16, cut where `parallel_fits` says
   no. Steps 3, 5 and 6 are left out of the quick sweep: 5 and 6 slots gave no more than 4 (measured), and
   3 gave 113. The K/V formats this backend has (`inkling_kv_modes`: k8v4 and k8v8 are CUDA only).
1. **Baseline.** `--bench` with exactly what a launch would start, `--draft off`, 1,024-token prompt, a
   decode length chosen so the decode takes about 8 s. This is the reference, and its speed sizes the
   later steps.
2. **Slot ladder.** `--bench` at 2, 4, 8, 16 slots with the baseline's K/V format. Read from each: total
   decode tok/s, and the tokens of K/V on the card from the `[InkCUDA]` line. Stop climbing when the total
   stops growing by 5%, when a setting does not start, or when the engine shrinks the context
   (`Auto-sizing max_ctx`): a setting that cannot hold the context asked for is out.
3. **K/V formats** (only with `--tune-allow-small`). `--bench` with fp8 and k8v4 at the two best slot counts
   of step 2. k8v8 and fp16 are left to the thorough sweep: at one slot k8v8 decoded like the others and fp16
   read prompts about 8% slower (measured).
4. **The two checks `--bench` cannot make**, for the baseline and the two best candidates, on the `--serve`
   line protocol with the probe's harness (`inkling_probe.Engine`), drafting off:
   - one request alone, 128 tokens, with all slots allocated: the lone-request speed;
   - one long prompt, about two thirds of the context and no more than 53k tokens, one token out: the long
     read.
   Candidates are checked in order of total speed, and the first that passes the rule wins.
5. **Confirm.** The winner and the baseline once more with `--bench`, in the other order. The hand sweep's
   rounds differed by about 1% for prompt reading and a few percent for decode (measured), so 5% is the gate.
6. **Record and apply** (next section). If the K/V format changed, the drafting probe runs at the start
   that follows, as it does today for any new placement.

The thorough sweep adds slots 3, 5, 6, 12, 32 where they fit, k8v8 and fp16, the experts one step up and
down from automatic, and a second round in reverse order for every setting.

### How the result is recorded

The same way `engine calibrate --apply-result` records, so the existing layering does the rest.

- **A report file** next to the launch config, `<name>.inkling-tune.json`, in the shape of the calibration
  report: `kind: "strata.inkling.tuning"`, `suite: "inkling-bench-v1"`, an identity (engine file hash, model
  file size and time, backend, cards, context, images), every step's raw `[bench]` line and timings, the
  baseline, the winner and the rule's numbers. This is the evidence, as `"drafting_probe"` is for drafting.
- **The run config**: `arg_layers.record_calibration(candidate, before, settings)` writes
  `"calibrated": {"settings": {"--kv": "k8v4"}, "model_name", "max_context", "images"}` and keeps the
  person's hand edits apart. Before that, a backup of the config and one smoke generation with the
  candidate, as `apply_result` does. A launch that builds its config again puts these values over Auto,
  and never over the person's own flags. `strata status` shows them with the source "calibrated".
- **Two small additions to the record**: `"suite"` and `"placement"` (the probe's `placement_key` idea:
  engine version, card and VRAM, backend, expert flags). `_calibration_fits` would also compare the
  placement when a record has one. A record without it, upstream's, behaves as today.
- **The slot count** is not an engine argument in a launch: it is a profile value, filled in by
  `ensure_profile` from `--parallel` or from `catalog_parallel()`. The record carries `"parallel": N`, and
  `ensure_profile` reads it between the two: the person's `--parallel`, then the tuned value, then Auto.
  This is the one new precedence rule (open question 4).
- `tuned_at` is set in the launch settings, so the next start does not measure again and the preset list can
  say "tuned" where it says "built-in" today.

The record is dropped, with one line, when the model, context, image setting or placement is another. It is
not measured again unless `--tune` is given.

### The default path

A new feature is opt-in and the default path stays byte-identical (`.github/CONTRIBUTING.md`).

- `--tune` defaults to `skip`, so `tune()` returns before any new code.
- With no `"calibrated"` record in the run config, `arg_layers.layer` and `ensure_profile` take the same
  branches as today.
- The check to ship with the change: for a fixed set of Inkling launch settings, the written launch config
  and profile are byte-identical before and after the change. As a test, in a `/tmp` worktree.

No engine source changes in the first version. The tuner reads the `[bench]` line and the `[InkCUDA]` line
as text. That keeps upstream's engine code as it is.

## Estimated cost

All estimates for the test machine, SQ-10, 80K window. None is measured. One `--bench` step is the model load
(4.6 s measured), a 1,024-token prompt in every slot, about 8 s of decode and the exit: about 15 to 25 s.

| Part | Steps | Estimate |
|---|---|---|
| Baseline and slot ladder (1, 2, 4, 8, 16) | 5 | about 1.5 to 2 min |
| K/V formats (fp8, k8v4 at two slot counts) | up to 4 | about 1 to 1.5 min |
| Lone request and long prompt, baseline and up to two candidates | 2 to 3 | about 2 to 3 min; a 53k prompt takes 41 to 46 s to read (measured) |
| Confirm | 2 | about 40 s |
| Drafting probe at the next start, if the K/V changed | 2 engine starts | about 1 min |
| **Quick, total** | | **about 5 to 8 min with the model away** |
| Thorough | about 25 settings, twice | about 20 to 30 min |

For comparison: the hand sweep took about 2 hours for 32 settings (measured). On a slow placement the time
grows: the drafting probe's notes give a 4060 with SQ-10 at 2 tok/s plain decode. The decode length is sized
from the baseline for that reason, and the sweep needs a total time limit (open question 7).

Work to build it, as a guess: the sweep and its report in one new module beside `inkling_probe.py`, the hook
in `tune()`, the two additions to the record, tests with a fake engine as the probe's tests have, and one
page of docs. About 600 to 900 lines with tests. It depends on the probe's library fix (branch
`fix/inkling-drafting-probe-libs`, `71cc5fb26`, not on `main` yet): without it an engine started outside the
server exits with 127 on the test machine (`libcudart.so.13` not found, measured 2026-11-24) and the failure
is silent.

## What is not known

Stated plainly, because none of it is measured:

- **Whether `--bench` ranks settings the way real serving does.** Its prompt is made-up tokens, it decodes
  greedy, and it fills every slot. The hand sweep measured through the HTTP server with real text. The first
  thing to do with a built tuner is to run it on the test machine and compare its ranking with the hand
  sweep's table. That needs the owner's yes: about 10 minutes with the model away.
- **How `--bench` reads prompts when all slots read at once**, against one prompt alone. Unknown.
- **Anything on the Mac, on AMD or on Intel.** The measurements are from one NVIDIA card. Metal, HIP and
  SYCL have no k8v4, so there only the slots would be tuned.
- **Answer quality with k8v4** beyond one test: a planted six-digit code was recalled 9 of 9 times at up to
  75k tokens, the same as fp8 (measured, 4 slots). Recall of a planted fact is one test of quality, not all.

## Open questions for the owner

1. **The rule.** Are "long prompt no more than 10% slower, lone request no more than 5% slower" the right
   limits? Or should the person state a goal instead (one agent with long contexts, a few agents, many short
   requests) and the tuner pick per goal: 1, 4 or 16 slots on the test machine?
2. **K/V format behind `--tune-allow-small`.** Is that the right gate? Without it the quick sweep tunes
   slots only, and on the test machine it would stop at 2 slots with fp8. Should fp8 itself count as
   answer-changing against Auto's fp16?
3. **The engine.** The first version reads text lines and uses the `--serve` protocol for two checks. A
   `--bench` option for "K of P slots busy" and a JSON result line would be cleaner and faster, but they are
   changes to upstream's engine source. Keep to no engine changes?
4. **Where the slot count lives.** In the calibrated record and read by `ensure_profile`, as proposed, or
   as a saved preset the way the Qwen tuner does (`tuned_preset`)? The preset way shows up in the preset list
   without new code, but a preset is the person's own object, and the hand sweep already left 32 of them to
   clean up.
5. **How much of `calibration.py` to reuse.** The full path there has a lease, a review step, a backup, a
   smoke test and a rollback. Proposed: the record, the backup and the smoke test, not the job machinery.
   Enough?
6. **Backends in the first version.** CUDA only, where the measurements are, with the others refusing in one
   sentence? Or slots-only tuning everywhere from the start, unmeasured?
7. **The time limit.** Stop the quick sweep at 10 minutes and apply the best so far, or apply nothing?
8. **Slot choices.** The launch screens offer 1, 4, 8, 32, 64 (`PARALLEL_CHOICES`); the engine takes 1 to
   64. May the tuner choose 2 or 16, which the screens cannot show as a choice?
9. **Hy4-Flash.** It is also skipped by the same line, but it has no slots and no drafter choice. Left out
   here. Agreed?
10. **Several cards.** Left out of the first version: the launcher drops the expert caps for a card split
    and the hand sweep has one card. Agreed?

## Follow-ups found on the way

Not part of this proposal. Each is its own change.

- `--kv-offload off` cannot reach the engine through the launcher: a flag whose value is "off" is dropped
  unless it is in `WORD_VALUE_FLAGS`, which is empty (`terminal/strata_cli/launch/settings.py:389`; read,
  and seen in the hand sweep).
- The engine can shrink the context to fit free VRAM (`engine/src/inkling2/cuda/ink_runner.cuh`,
  `Auto-sizing max_ctx`) without telling the server, which keeps the context asked for. It did not fire on
  the test machine. Unconfirmed static analysis.
- A lone request decoded at 71 to 75 tok/s instead of about 80 with fp8 at 3, 4 and 6 slots, in both rounds,
  and not at 1, 2, 5, 8 or 16 (measured, unexplained). The tuner's lone-request check would catch it; it
  would not explain it.
- The drafting probe's library fix (`71cc5fb26`) is not on `main`.
