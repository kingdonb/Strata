# PR #24 review: the monitor bundle fix, the Apple Silicon thermal sensor, and the pre-release punch list

PR #24 (https://github.com/kingdonb/Strata/pull/24) started as one CI bugfix and grew, by the
owner's decision (2026-10-25), into a mixed slice: bugfixes ride along with the story that touches
them instead of waiting for a separate patch release (the fork ships no releases; see FORK.md).
This file records the review of everything the branch carries and the list of things to check
before the next release or sync-forward.

## What the branch carries

| Commit | Concern | Review |
| --- | --- | --- |
| `1ff83498` | fix(packaging): the thin monitor bundle packs the interactive Monitor's runtime imports (bundle schema 6, adopted verbatim from upstream main) | reviewed by the spec agent; see the PR body |
| `1710a845` | feat(telemetry): rootless Apple Silicon thermal sensor readings for the Metal backend | reviewed below |
| `68d293ba` | test(serve): stabilize the sampler thread-count test against lingering previous threads | reviewed below |
| (this commit) | fix(telemetry): two review fixes + this file | — |

## The bundle fix (schema 6) — verified

The v0.2.3.1 sync made the packaged interactive Monitor import `strata_cli/consoleinfo.py` and
`strata_cli/overview.py` at run time; the schema-5 bundle packed neither, so the py3.11
"Smoke test packaged interactive Overview" step crashed with
`ImportError: cannot import name 'consoleinfo' from 'strata_cli'`. Upstream main already carries
the schema-6 fix (builder + bundle tests). The branch adopts both files verbatim; the fork's
copies were exactly upstream minus that delta (checked by diff before adopting).

Measured 2026-10-25: CI green on the PR — all four "Portable monitor client" jobs on push and
pull_request events, including the two previously-red py3.11 jobs; the "Minimal serving runtime"
suite was already green on main (run 16) and is not triggered by this PR's paths.

## The thermal sensor — reviewed; hardware claims marked unverified

`_AppleSiliconThermal` (serve/telemetry.py) loads CoreFoundation/IOKit with ctypes on darwin only
(`sys.platform != "darwin"` returns before any framework load), builds an IOHIDEventSystemClient
with a PrimaryUsagePage 0xff00 / PrimaryUsage 5 match, and per poll reads each service's
`Product` name, keeps the `tdie*`/`tdev*` sensors, and returns the peak event-15 temperature
(selector `15 << 16`) in a 10–125 °C sanity range, rounded to 0.1. `_Metal.read()` puts it in
`out["temp"]`, which the existing pipeline already consumes (`Telemetry.sample()` takes the max
into `gpu_temp`; the 60-second history and both UIs read `gpu_temp`/`history.gpu_temp`).

Verified on Linux (this sandbox has no Apple hardware):

- `python -m py_compile serve/telemetry.py`; the off-darwin early return means Linux/Windows
  behavior is unchanged — `read()` gains `temp` only when a thermal reader answers.
- `python -m unittest serve.test_server.MacTelemetry serve.test_responses.NoLeakedSampler` —
  6 tests OK, run twice consecutively (stable).
- The pre-fix `NoLeakedSampler` code FAILED on this machine (`2 != 3`, the count race) — the
  `is_alive()` filter + bounded settle loop in `68d293ba` fixes a real flake, not a test
  weakening: a leaked sampler is still a live `(_loop)` thread and is still caught.
- The wiring test (`test_thermal_reading_apple_silicon`, mock-based) passes on Linux.

Unverified (needs the owner's Mac): that a real M2 answers the query (the "38 °C" reading in the
PR notes is the author's run, not re-measured here), and long-run stability of the per-poll
`CopyServices` call. Treat the hardware claims as the author's report until re-measured.

### Findings and fixes applied in this branch

1. **Use-after-free risk in the matching dictionary** (static analysis, unconfirmed on hardware):
   `CFDictionaryCreate(None, keys, vals, 2, None, None)` — with NULL callbacks the dictionary
   retains neither keys nor values, but the constructor released the four CF objects immediately;
   `SetMatching`/`CopyServices` read them on every poll. Fixed by keeping the references for the
   object's lifetime (`self._matching_refs`). Leaking four small CF objects once per process is
   harmless; dangling them is not.
2. **The user-facing `note` was stale**: it still claimed no temperature exists without root
   (`gpu_note` surfaces in the sample), while the reading now arrives. Reworded truthfully
   (power still root-only via `powermetrics`; temperature rootless when the sensors answer);
   `test_note_reaches_the_monitor`'s assertions still hold.

## Punch list for the next release / sync-forward

Pick at least the item that matches the code a story touches; do not expand a story's scope
further than that.

1. [ ] Verify the thermal reader on the owner's Mac over a long serving session (hours): the
       temperature is present and stable, nothing crashes, and the per-poll IOHID cost is measured
       (the ioreg path is 22 ms — put the HID cost next to it in docs/DETAILS.md if it is not
       negligible; cache the service list if it is).
2. [ ] "GPU temp" label semantics: the figure is the peak across SoC-die (`tdie*`) and device
       (`tdev*`) sensors. Either derive distinct SoC vs GPU figures or document the mapping next
       to the tile (upstream has the same question — a good upstream-contribution candidate).
3. [ ] Make `gpu_note` dynamic: when a Metal reading carries a temperature, the note should not
       repeat the rootless-temperature sentence as if it were missing.
4. [ ] Extend the import-closure guard pattern: an AST test that `strata_cli/handover.py` (which
       `strata.py` imports before the preflight) stays standard-library-only, so a future sync
       cannot quietly break `python -S` runs. (The bundle's closure guard in
       `tests/terminal/test_monitor_bundle.py` is the template.)
5. [ ] Bundle-test parity: when a future sync bumps the bundle schema, adopt upstream's
       `tests/terminal/test_monitor_bundle.py` in the same commit — the fork's copy must never be
       the stale one again.
6. [ ] After every sync-forward: run the two matrix smokes locally before opening the terminal PR
       (minimal CLI smoke on a bare package-less interpreter; minimal server smoke in a
       requirements.txt-only venv). Both suites were born red on 2026-10-19 and stayed red for six
       days because no local run happened first.
7. [ ] Mixed-concern PRs are allowed now (owner decision, 2026-10-25, recorded in FORK.md), but
       each concern gets its own commit and a line in the PR body, so a revert stays possible.
