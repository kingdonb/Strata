# About this fork (kingdonb/Strata)

This fork is upstream Strata plus a small set of additions. It is built from source on the owner's Mac. It does not
publish installers, engine packages or an update feed of its own yet; that is planned (see "Releasing from the fork").

- Base: upstream **v0.2.20**, merged on 2026-11-22 with an ordinary `git merge` (22 files in conflict: the fork's
  additions beside upstream's changes, both sides kept; and the Inkling engine, where upstream's 0.2.20 rewrite
  replaced two of the fork's fixes). 0.2.16 had moved the top level into `app/`, `engine/`, `terminal/`, `docs/` and
  `tools/`; the fork's own scripts are in `tools/fork/`. The fork's five Settings strings are `s9900`-`s9904`, far
  from upstream's numbering, which took the fork's earlier numbers twice.
- Upstream 0.2.14 removed DeepSeek V4.1-Flash from the tree. The fork follows: it has no DeepSeek either.
- Versions: upstream's (app and Terminal 0.2.20, Control 0.1.66, engine 0.1.66). The fork no longer keeps a version
  line of its own. A fork release will add a suffix to upstream's number (owner decision, 2026-11-14).
- Until the fork has its own feed, the built-in update address is still upstream's
  (`github.com/Niko1221/Strata/releases/latest/download`). `strata update` and the app's Update button would install
  upstream's engine, which does not have the Qwen3.6 Metal path. Do not use them on a machine that runs this fork's
  engine.

## What the fork adds

Everything else in the tree is upstream's file, byte for byte. `git diff v0.2.20 HEAD --stat` lists the whole
difference.

| Addition | Where |
| --- | --- |
| Qwen3.6-35B-A3B on Apple Silicon (Metal) | `engine/metal/kernels/q36*.metal`, the K-quant kernels in `engine/metal/kernels/quant.metal`, `engine/metal/src/engine/q36_metal.cpp`, `engine/metal/src/engine/q36_prefix_plan.hpp`, `engine/metal/tests/q36_*`, `engine/metal/tests/moe_sim.cpp`, `terminal/tools/qwen35/q36_*`; documented in [QWEN36_METAL.md](QWEN36_METAL.md) |
| Serial GPU dispatch for the Qwen3.6 graphs, the dispatch tally and the argument-count guard | `engine/metal/src/mtl_rt.h`, `engine/metal/src/mtl_rt.mm` (`set_serial_graphs`; off for every other engine) |
| The direct chat asks before a connected tool runs (Manual and Plan ask, Bypass runs) | `terminal/serve/server.py` (`GET /v1/mcp/tools`, `POST /v1/mcp/call`), `app/host/server/tool-run.mjs`, `app/host/server/index.mjs` (`/api/tools/run`), `app/host/server/inference.mjs`, `app/host/src/lib/store.ts`, `app/host/src/app/ContextPanel.tsx`, `app/host/src/components/ComposerControls.tsx` |
| The built-in Strata tools in the agent (`mcp__strata__*`) | `app/host/server/native-control.mjs` (`strataToolSpec`), `app/host/server/agent/` (`profile.mjs`, `tools-settings.mjs`, `plugins/strata-gate.mjs`, `agent-bridge.mjs`), `app/host/src/components/SettingsTools.tsx`, strings `s9039`-`s9042` in `app/host/src/locales/` |
| The built-in Strata tools in the engine's own chat | `terminal/serve/mcp.py` (`builtin_strata_server`, `"mcp": {"builtin_strata": false}` turns it off). The server tells the tools its own port, config and key (`STRATA_MCP_HOST_*`), so `terminal/tools/strata_mcp.py` marks it as the chat's own server (`this_chat`), reads its config and engine log, and refuses to stop it; `terminal/tools/test_strata_mcp_host.py`. The model table counts a size as installed when a config `strata launch` wrote holds it (its `.choices.json`, or its `model_name`), not only a config named after the size: upstream's table listed every model `strata launch` set up as not installed (`installed_index`, `installed_row`; `LaunchConfigsInTheModelTable`). `strata_status` counts a model folder of `.strata` files (Hy4-Flash, Inkling-2-Mini) as a finished download: upstream counted `.gguf` files only and listed both as incomplete (`download_state`; `DownloadsInStatus`) |
| The Qwen3.6 size the Mac runs (UD-Q2_K_XL, pinned by size and SHA-256), and the Qwen3.6 rows in the Mac model list | `terminal/strata_control/mac_q36.py`, `terminal/tools/strata_mcp.py` (`_mac_qwen36`) |
| Qwen3.6 setup on a Mac (`./setup.sh --family qwen36`) | `engine/metal/setup_mac_q36.py`, the hand-over in `engine/metal/setup_mac.py` (`qwen36_setup`), `terminal/tools/test_setup_mac_q36.py` |
| The Qwen3.6 engine's settings in the register, its result folders in the bench index, the guarded check script | `terminal/data/settings-register.json` (13 `STRATA_Q36M_*` entries), `docs/bench-results/README.md` (two rows), `tools/fork/check_guarded.sh` |
| The M2 validation records, the Metal roadmap and review, the sandbox notes | `docs/QWEN36_METAL*.md`, `docs/m2-sessions/`, `docs/M2_*.md`, `docs/ENVIRONMENT.md`, `docs/blog/`, `docs/bench-results/2026-10-*-m2-q36-*` |
| A frozen engine is ended on the `--batch` path too (every Inkling request takes it, even at `--parallel 1`): the reader, a slot and the drain after a disconnect call the stall check, so a silent engine with no CPU, I/O or GPU work is ended after `ENGINE_STALL_S` (90 s) instead of never | `terminal/serve/server.py` (`_control`, the slot reader in `generate_batched`, `_drain_control`), four tests in `terminal/serve/test_server.py` (`SilentEngine`) |
| Inkling engine (CUDA): a CUDA or cuBLAS error ends the process with `_Exit` after flushing stdout and stderr, not `std::exit`, which could hang holding the card (seen on chudmax, RTX 5060 Ti 16 GB, 2026-11-22). In v0.2.19-mbp.rtx.2 the fork also shipped a graph fallback and a half-size prompt workspace; upstream 0.2.20 fixed both its own way (graph builds fall back to eager mode; the workspace is a fixed 2,048-token tile, about 114 MiB, its #6605 / #6609), and the merge took upstream's | `engine/src/inkling2/cuda/ink_cuda.cuh` (`INK_DIE`) |
| Inkling engine (CUDA): pictures are read on the card. Upstream's CUDA backend had no picture reader, so on NVIDIA every picture was read on the CPU inside the only request slot (a 2,000x1,088 screenshot: 468+ s on chudmax; on the card 0.9 s). The reader never ends the engine: without free card memory it hands the picture to the CPU, which first scales it down to `--vision-cpu-max-tokens` (default 256, about 30 s on an i5-10400) | `engine/src/inkling2/cuda/ink_cuda_vision.cuh`, `cuda/ink_cuda_backend.cu` (`encode_image`), `ink_serve.h` (`shrink_for_cpu`), `cuda/ink_vision_check.cu` (`strata-inkling-vision-check`), [DETAILS.md](DETAILS.md#inkling-2-minis-picture-reader) |

## Syncing with upstream

Merge a release tag. Do not apply path-scoped diffs by hand again.

```
git fetch origin --tags
git merge v0.2.16          # the next upstream tag
```

Conflicts can only appear in the files of the table above. For each one, take upstream's change and keep the fork's
addition beside it. When upstream fixes something the fork also fixed, take upstream's fix and drop the fork's.

After a merge:

1. Upstream's fast checks (`tools/check`: the Python suites, the app's typecheck, tests and build, the docs checks).
   On Linux run `python tools/check`. On a Mac run `tools/fork/check_guarded.sh`, never `tools/check` in a checkout you
   serve models from. On a Mac `setup.py` hands its whole command line to `engine/metal/setup_mac.py`, so a test that calls
   setup for real runs a real setup: on 2026-11-14 that rewrote a run config in the live checkout and started its
   model, and in a clean worktree it began downloading a model (58 GB). The two tests found then are fixed; upstream
   runs the Python suites on Linux and Windows only, so another can appear. The script runs the checks in a scratch
   worktree with a scratch `HOME`, no outside network and no write access to the checkout or the model folders:

   ```
   STRATA_GUARD_READONLY=$HOME/w/Strata-data:$HOME/Documents/Strata tools/fork/check_guarded.sh
   ```

   Many tests fail on a Mac on pristine upstream too. Compare with the same run of the upstream tag
   (`tools/fork/check_guarded.sh v0.2.20`) before reading a failure as the fork's.
   After adding a `STRATA_*` setting to the engine, add it to `terminal/data/settings-register.json` (run
   `python tools/settings_register.py`, keep only the new entries, fill in `status`, `owner`, `note`); after adding a
   folder under `docs/bench-results`, run `python tools/bench_results_index.py` in a clean checkout.
2. Build the Metal engine (`cmake -S metal -B build-metal -G Ninja -DMETAL_CPP_DIR=third_party/metal-cpp`, then
   `cmake --build build-metal --target strata`) and the Qwen3.6 tests (`q36_compile_test`, `q36_chunk_parity`,
   `prefix_plan_test`, `moe_sim`).
3. Run the Qwen3.6 gates on the Mac with the model loaded. The commands are in [QWEN36_METAL.md](QWEN36_METAL.md).

Push the upstream tags the fork merged (`v0.2.2`, `v0.2.3.1`, `v0.2.13.1`, `v0.2.14`, `v0.2.14.1`, `v0.2.15`, `v0.2.16`, `v0.2.17`, `v0.2.18`, `v0.2.19`, `v0.2.20` so far) to the fork, not every upstream tag.

### Names the two sides share

Upstream's model tables are keyed by the bare size name, and Unsloth uses the same size names for different models.
`UD-Q2_K_XL`, `UD-Q4_K_XL` and `UD-Q6_K_XL` are Qwen3.8-Flash-Next sets in `setup.py` `MODELS` and in
`terminal/strata_control/mac_fit.py` `MAC_MODELS`, and they are also Qwen3.6 sizes in `OWN_PATH_FAMILIES["qwen36"]`. Never add a
Qwen3.6 size to `MODELS`, `MAC_MODELS` or a label map under its bare name: it would replace or relabel the Flash-Next
set. Reach Qwen3.6 through its family (`own_size("qwen36", ...)`, `--family qwen36`).

`setup.py` is upstream's file, byte for byte. Its `OWN_PATH_FAMILIES["qwen36"]` lists the sizes upstream's CUDA and
HIP path runs; a size added there is offered on every PC and, for a PC with little RAM, recommended. The size the Mac
runs is in `terminal/strata_control/mac_q36.py` and nowhere else.

## Open work after the 2026-11-14 merge

- **Qwen3.6 setup on a Mac: ported, not yet run end to end on the Mac.** `./setup.sh --setup --family qwen36` goes to
  `engine/metal/setup_mac_q36.py` (a file of its own; `engine/metal/setup_mac.py` only hands the run over). It writes the run config
  the way `setup.py` does for this family on a PC: `"arch": "qwen35moe"`, `--max-context`, `--mtp` as a draft count.
  A run config written before the merge has no `"arch"`, so the rest of Strata reads it as a Flash-Next config: run
  setup again (the model file is found where it is and not downloaded again), or add `"arch": "qwen35moe"` by hand.
  `strata launch` on a Mac installs upstream's packaged Metal engine, which has no Qwen3.6 path: use `./setup.sh`
  from this checkout until the fork publishes its own engine package.
- **Qwen3.6 names in the app.** Settled (task 000861, 2026-11-19): the sizes are labelled through their family
  (`QWEN36_LABELS` in `app/host/src/lib/native-catalog.ts`, resolved by `sourceFamily`, a test in
  `app/host/tests/qwen36-labels.test.mjs`); the bare-name map stays Flash-Next's.
- **The Metal gates on the merged engine.** See the merge's pull request for what was run and what was not.

## Releasing from the fork

Upstream has no release workflow. It builds on build boxes without a key and signs on one release PC
([tools/packaging/UPDATES.md](../tools/packaging/UPDATES.md)). The fork follows the same split, with three choices of its own:

- **The tree keeps upstream's addresses and upstream's key.** The fork's release address and public feed key
  (`tools/packaging/fork/feed-key.pub.json`, key id `e52bcd0bc55781a9`) are written into the sources when a package is
  built, by `tools/fork/fork_release.py bake`. So upstream's tests that pin its addresses still hold, and a merge does
  not conflict there. A source checkout therefore still reads upstream's feed: only an installed fork release reads
  the fork's.
- **Upstream's engines and drafters are not built or uploaded again.** `tools/fork/fork_release.py carry` checks
  upstream's signed feed against upstream's key and copies its engine and drafter entries, with upstream's
  addresses, sizes and SHA-256, into the fork's feed. A fork Control trusts both keys: the fork's for the feed, and
  upstream's because each engine package carries upstream's signature. Upstream's Metal engine is left out (it has
  no Qwen3.6 path); the fork's own Metal engine package is open work. There is no DeepSeek member.
- **One exception: the Inkling 2 Mini engine for Linux CUDA 13** (`strata-inkling`, shape `inkling2`) is the fork's own
  build when the fork has engine changes upstream lacks: `0.1.65.1` in v0.2.19-mbp.rtx.2, and `0.1.66.1` from
  v0.2.20-mbp.rtx.3 (the CUDA picture reader and the `_Exit` fix). It is signed with the fork's key, and its entry
  replaces upstream's carried `strata-inkling` linux-x86_64 cuda13 entry in `RELEASE/shapes` before `build-feed`. Built
  with `tools/fork/inkling_release.sh` in `nvidia/cuda:13.0.2-devel-ubuntu24.04`. Every other engine is upstream's.
- **The feed sequence is upstream's times 1000, plus N** (`tools/fork/fork_release.py sequence`). A PC refuses a feed
  older than the last one it accepted, so a PC that ran upstream needs a higher number. A PC that took a fork feed
  refuses upstream's feed afterwards.

The steps of a release `vX-mbp.rtx.N`:

1. Push the commit to a branch named `fork-release/...`, or run the "Fork release build" workflow by hand. It bakes,
   builds the Terminal for Linux x86_64 and macOS arm64, smoke-tests it and packs it as the feed's Control package.
   Download the two `fork-control-*` artifacts into one folder `RELEASE` (`control/`, `install.sh`, `install.ps1`).
2. Download upstream's `strata-feed.json` and `strata-feed.json.sig` of the same version, then:

   ```
   python tools/fork/fork_release.py carry --upstream-feed strata-feed.json --upstream-sig strata-feed.json.sig \
       --upstream-tag vX --out RELEASE
   ```

3. On the owner's machine, with the key (the only step that reads it):

   ```
   python tools/packaging/feed_tool.py build-feed --out RELEASE --key KEYDIR/feed-key.json \
       --sequence $(python tools/fork/fork_release.py sequence --upstream-feed strata-feed.json -n N) \
       --engine RELEASE/engine --drafters RELEASE/drafters --shapes RELEASE/shapes \
       --control RELEASE/control --no-dsv41-carry
   python tools/packaging/feed_tool.py verify-feed RELEASE --pub tools/packaging/fork/feed-key.pub.json
   ```

4. Tag the commit `vX-mbp.rtx.N` and publish a normal (not pre-release) GitHub release on `kingdonb/Strata` with
   `strata-feed.json`, `strata-feed.json.sig`, `install-manifest.txt`, `install.sh`, `install.ps1` and the two
   Control archives as flat assets. The one-line installers and `strata update` read `releases/latest/download`.
5. On a PC: `curl -fsSL https://github.com/kingdonb/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh`.

The desktop app is not part of a fork release yet: it would need a Tauri updater key and its own addresses
(`app/desktop/src-tauri/tauri.conf.json`, `app/host/server/updates.mjs`).

### The two machines a fork release is for (owner, 2026-11-14)

| Machine | What it runs | What a release needs for it |
| --- | --- | --- |
| `mbp`: a MacBook with an M2 and 24 GB (about 18 GB of it wired while the model runs). The owner's own computer; it has to work offline. The signing keys are made and kept here. | Qwen3.6-35B-A3B on this fork's Metal engine | The fork's own Metal engine package (`macos-arm64`, `metal`), built on this Mac, and the fork's Control |
| `rtx`: `chudmax`, Ubuntu on x86_64 (kernel 7.0), 64 GB of RAM, one NVIDIA RTX 5060 Ti with 16 GB | Qwen3.8-Flash-Next on upstream's CUDA engine | The fork's Control for `linux-x86_64`. The engine is upstream's `strata-engine-<version>-linux-x86_64-cuda-cuda13` package, signed again with the fork's key (`feed_tool.py sign-package`) or trusted through upstream's key |

### Names and version numbers

- A fork release is tagged with upstream's version and the suffix `-mbp.rtx.N`: `v0.2.14-mbp.rtx.1` was the first
  release, on upstream v0.2.14; `v0.2.14.1-mbp.rtx.1` followed the same day (the suffix names the two machines above; `N` counts the fork's releases on that base).
- The suffix belongs to the git tag and the GitHub release only. Versions inside the signed feed must be plain numbers
  with three or four parts: `terminal/strata_control/feed.py` `version_key` refuses anything else ("is not a version number"),
  and where a suffix is allowed (the installed app's version, `terminal/strata_cli/handover.py`) it reads as a pre-release and
  sorts below the same number. So the packages of a fork release carry upstream's numbers (Control 0.1.60.1,
  engine 0.1.60).
- A second fork release on the same upstream base (a `.2` after a `.1`) carries the same package versions, because
  they are upstream's numbers; `strata update` refuses it (it only installs a higher version). The convention the
  releases follow (written out in the v0.2.15-mbp.rtx.2 notes): such a machine is updated by running the fork's
  installer again — it installs the Terminal beside the old one and keeps the models and the settings.
  `strata update --force` does not help here: it only tries again a version that failed its test start or was rolled
  back from (`terminal/strata_control/updater.py`), never the one already current. A release on a newer base updates
  with `strata update` as usual
  (v0.2.16-mbp.rtx.1 did: Control 0.1.62 and engine 0.1.62 are both newer than 0.1.61).

## Owner decisions on record

- 2026-11-14: nothing from this fork is offered upstream. It lives in the kingdonb tree only. Upstream chose not to
  support Qwen3.6 on Metal, for good reason (a laptop runs it hot and slowly); the fork is for people who want to try
  it anyway. A bug found in upstream's code is fixed here and noted in this file, not reported there.
- 2026-11-14: the Metal work ships in the fork's own releases. If a signed feed is published, its versions are
  upstream's, unmodified; the fork's `-mbp.rtx.N` tag is added beside upstream's tag so that it is plain the release
  is not upstream's. The fork does not rebuild upstream's engine unmodified: where upstream's engine is right for a
  machine, upstream's package is used.
- 2026-11-14: on the owner's MacBook, full test suites, engine builds and model runs happen in a window the owner
  names (overnight as a rule). Unscheduled work stays light.
- 2026-11-14: when the fork and upstream both changed something, follow upstream unless the fork's work is expressly
  new. Adopt upstream's version numbers. Publish only the upstream tags the fork merged. A fork release carries a
  suffix on upstream's number.
- 2026-11-22: v0.2.19-mbp.rtx.2 ships the fork's own build of the Inkling engine (Linux, CUDA 13) with the fork's
  fixes, version `0.1.65.1`: newer than upstream's 0.1.65, so `strata update` installs it beside the old one. This
  departs from "versions are upstream's, unmodified" for that one package; the reason is that a package with
  upstream's number would never reach a machine that has upstream's.
- 2026-10-25: a small bugfix rides along with the story that touches it; there is no separate patch release per fix.
  Each concern still gets its own commit and a line in the pull request, and the review of a mixed pull request lands
  as a docs file in the same branch (see [PR24_REVIEW.md](PR24_REVIEW.md)).

## History

- 2026-10-17 to 2026-11-09: the fork was developed on upstream 0.2.0 and synced twice by hand-applied, path-scoped
  diffs (v0.2.2 on 2026-10-23, v0.2.3.1 on 2026-10-25) with a skip-list. The 2026-11-14 merge retired that method and
  the skip-list: every skipped area is upstream's again.
- Fork changes that upstream later made in its own way, and that the merge dropped in favour of upstream's: the
  lifeline watcher identity (upstream #3590), the empty `STRATA_API_KEY` of a source-run server (#3851), ending the
  Time-skip process group (#4529, #4570), the Metal prompt path's expert-cache loan, the Gyro exclusion on the Mac,
  the consolidated Metal stubs file, and the test skips that existed only for the skip-list.
