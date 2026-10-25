# About this fork (kingdonb/Strata)

This fork is a **development line**, not a release. It is Strata 0.2.0 with a divergent Apple-Silicon
Metal/Qwen3.6 kernel line on top (`metal/`, documented in [QWEN36_METAL.md](QWEN36_METAL.md)). It is built
from source on the owner's Mac; it ships no installers, no engine packages and no update feed of its own.

- App version: **0.2.0-beta** (`apps/host/package.json`, `apps/desktop`) — kept deliberately; see "Versioning".
- Engine version: **0.1.45.3-metal** (`metal/CMakeLists.txt` `STRATA_VERSION`).
- The host's update check still points at upstream's release feed (`api.github.com/repos/Niko1221/Strata`,
  inherited from 0.2.0). "Update" would offer upstream engines; installing one would replace this fork's
  engine. Do not use it on this fork. (A standalone decision on the fork's update story is pending.)

## Upstream syncs

| Date | Upstream synced through | Merge-base | Slices |
| --- | --- | --- | --- |
| 2026-10-25 | **v0.2.3.1** (`7ddc17a3`), range `011c9e60..7ddc17a3` (131 commits, 294 files, +14,913/−661) | `011c9e60` ("Release notes 0.2.2") | Host context handling, Terminal steadiness, safe drafting wiring (serve/ deltas skipped — engine version gates) |
| 2026-10-23 | **v0.2.2** (`011c9e60`), range `3d1fb480..011c9e60` (404 commits, 590 files, +52,967/−2,271) | `3d1fb480` ("Release notes 0.2.0") | Terminal/Control/serve; host app; docs + this file |

Syncs are **path-scoped and deliberate**, one PR per slice — never a blind merge:

```
git remote add upstream https://github.com/Niko1221/Strata.git
git fetch upstream --tags
git diff 3d1fb480 v0.2.2 -- <paths> | git apply      # per slice, on a fresh branch from main
```

`scripts/upstream_survey.sh <from> <to>` wraps this method (the survey, ledger-row generation and
skip-list application; run it with `--help`). The skip list's machine-readable twin is
`scripts/upstream_skip.txt` — update it in the same PR whenever this table changes.

Then apply the skip-list below (revert modified files to the merge-base, delete upstream-new ones), run the
suites, and record what was skipped and why in the PR.

## Skip-list as of the v0.2.2 sync (with reasons)

| Skipped | Why |
| --- | --- |
| `strata_control/feed.py`, `strata_cli/update_cmd.py` | The signed update feed / `strata update` rework — the fork keeps the 0.2.0 update behavior (pending standalone decision) |
| `strata_cli/bench_facts.py`, `bench_standard.py` (the `standard-v1` suite) | Benchmarks stay gated (Benchmark Day); the plain serving-bench commands keep working; `bench run --suite standard-v1` fails with a plain fork message |
| `strata_control/calibration_jobs.py`, `autotune.py`, `benchmarking.py` | Bench-gated measurement machinery (the `calibration.py`/`components.py`/`bench.py` hardening fixes ARE taken — API-supersets) |
| `tools/test_setup_*.py`, `setup.py`, `START-HERE.bat`, `cmake/`, root `CMakeLists.txt` | Upstream's source-installer and engine-package machinery (CUDA/HIP/SYCL package selection, gfx lists); the fork builds the Metal engine |
| `src/` (CUDA/HIP engine), `sycl/` (Intel) | The fork builds no CUDA/HIP/SYCL engine; keeping them unsynced keeps the fork's delta small. The shared `include/strata/` headers stay synced as one block with them, never piecemeal (deferred engine-adjacent slice, interactive per the working agreement) |
| `metal/` upstream changes (Gyro trellis kernels, dense tiles) | Engine behavior — deferred behind an owner decision (a Metal Gyro port would let the fork's engine load Gyro files); the overlap with our `quant.metal` is section-disjoint and mergeable later |
| `packaging/`, `docker/` (except nothing), `.github/`, `Dockerfile`, `installer.nsi` | Upstream release/CI/image machinery the fork does not run |
| `serve/` upstream changes in v0.2.3.1 | Gated on upstream's 0.1.48 engine packaging, version gates, and DFlash2 routing; the fork runs 0.1.45.3-metal. Skipped to prevent advertising unsupported engine features. Terminal/CLI connection TLS trust was ported via `strata_control/tls.py`. |
| `tools/dflash2/`, `strata_control/drafting_fetch.py` (DFlash2 drafter) | Surveyed, not adopted — targets Qwen3.8-Flash-Next (CUDA/SYCL); the fork's q36 line uses its own native MTP speculative decoding |
| `bench/**`, `bench.sh`, `BENCH.bat` | Reference material only — no claims, no measurement scope |
| README translations, `AGENTS.md` upstream changes | They advertise features the fork does not take; the fork's README/AGENTS stay fork-specific |
| Gyro/RX-6000 model entries | The fork's Metal engine does not load Gyro yet (host references are engine-report-guarded and stay graceful) |

**Versioning (deliberate):** the fork keeps its own version line (`0.2.0-beta` app, `0.1.45.3-metal` engine)
and records upstream provenance here instead of adopting upstream's bumps (`0.2.2` / `0.1.46` / `0.1.47`).
Revisit only if the fork starts shipping releases.

## Notes for the next sync

1. The survey method is four commands (fetch tags; `diff --name-only` per area; commit subjects per area; read
   the release notes) — see the spec task's design doc (helix-specs `design/tasks/000696_upstream-sync-survey/`).
2. Upstream test files embed product-version strings and Windows path assumptions. Validate with a
   **three-way comparison** (merge-base worktree, full-upstream worktree, synced tree) before trusting any
   single red test; the sandbox lacks `regex`/`textual` (many environmental errors) and its `sh` is dash.
3. `tests/data/` carries test fixtures ported tests need (`gpu_roles_vectors.json`) — sync it with the tests,
   not with the skipped areas.
4. Upstream's terminal tests that read `%APPDATA%/Strata/settings.json` fail identically on the pristine
   upstream tree on Linux; the ported code honors the same setting written to the platform-correct path.
5. The two engine-tree overlap files (`metal/CMakeLists.txt`, `metal/kernels/quant.metal`) and
   `docs/TEST_REQUESTS.md` are the only places both sides have touched — resolve them as unions when the
   engine-adjacent slice (if approved) lands.
6. Owner decision (2026-10-25): small bugfixes ride along inline with the story that touches them —
   no separate patch release per fix. Each concern still gets its own commit and a line in the PR body;
   the review of a mixed PR lands as a docs file in the same branch (see docs/PR24_REVIEW.md and its
   punch list for the pre-release checks).
