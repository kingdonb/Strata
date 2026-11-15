# About this fork (kingdonb/Strata)

This fork is upstream Strata plus a small set of additions. It is built from source on the owner's Mac. It does not
publish installers, engine packages or an update feed of its own yet; that is planned (see "Releasing from the fork").

- Base: upstream **v0.2.14.1**, merged on 2026-11-15 with an ordinary `git merge` (v0.2.14 first: six files
  conflicted, all of them files of the table below; then the Control-only update v0.2.14.1, no conflicts).
- Upstream 0.2.14 removed DeepSeek V4.1-Flash from the tree. The fork follows: it has no DeepSeek either.
- Versions: upstream's (app 0.2.14, Terminal 0.2.14.1, Control 0.1.60.1, engine 0.1.60). The fork no longer keeps a version
  line of its own. A fork release will add a suffix to upstream's number (owner decision, 2026-11-14).
- Until the fork has its own feed, the built-in update address is still upstream's
  (`github.com/Niko1221/Strata/releases/latest/download`). `strata update` and the app's Update button would install
  upstream's engine, which does not have the Qwen3.6 Metal path. Do not use them on a machine that runs this fork's
  engine.

## What the fork adds

Everything else in the tree is upstream's file, byte for byte. `git diff v0.2.14.1 HEAD --stat` lists the whole
difference.

| Addition | Where |
| --- | --- |
| Qwen3.6-35B-A3B on Apple Silicon (Metal) | `metal/kernels/q36*.metal`, the K-quant kernels in `metal/kernels/quant.metal`, `metal/src/engine/q36_metal.cpp`, `metal/src/engine/q36_prefix_plan.hpp`, `metal/tests/q36_*`, `metal/tests/moe_sim.cpp`, `tools/qwen35/q36_*`; documented in [QWEN36_METAL.md](QWEN36_METAL.md) |
| Serial GPU dispatch for the Qwen3.6 graphs, the dispatch tally and the argument-count guard | `metal/src/mtl_rt.h`, `metal/src/mtl_rt.mm` (`set_serial_graphs`; off for every other engine) |
| The direct chat asks before a connected tool runs (Manual and Plan ask, Bypass runs) | `serve/server.py` (`GET /v1/mcp/tools`, `POST /v1/mcp/call`), `apps/host/server/tool-run.mjs`, `apps/host/server/index.mjs` (`/api/tools/run`), `apps/host/server/inference.mjs`, `apps/host/src/lib/store.ts`, `apps/host/src/app/ContextPanel.tsx`, `apps/host/src/components/ComposerControls.tsx` |
| The built-in Strata tools in the agent (`mcp__strata__*`) | `apps/host/server/native-control.mjs` (`strataToolSpec`), `apps/host/server/agent/` (`profile.mjs`, `tools-settings.mjs`, `plugins/strata-gate.mjs`, `agent-bridge.mjs`), `apps/host/src/components/SettingsTools.tsx`, strings `s9039`-`s9042` in `apps/host/src/locales/` |
| The built-in Strata tools in the engine's own chat | `serve/mcp.py` (`builtin_strata_server`, `"mcp": {"builtin_strata": false}` turns it off). The server tells the tools its own port, config and key (`STRATA_MCP_HOST_*`), so `tools/strata_mcp.py` marks it as the chat's own server (`this_chat`), reads its config and engine log, and refuses to stop it; `tools/test_strata_mcp_host.py` |
| Apple Silicon temperature without root | `serve/telemetry.py` (`_AppleSiliconThermal`) |
| The Qwen3.6 size the Mac runs (UD-Q2_K_XL, pinned by size and SHA-256), and the Qwen3.6 rows in the Mac model list | `strata_control/mac_q36.py`, `tools/strata_mcp.py` (`_mac_qwen36`) |
| Qwen3.6 setup on a Mac (`./setup.sh --family qwen36`) | `metal/setup_mac_q36.py`, the hand-over in `metal/setup_mac.py` (`qwen36_setup`), `tools/test_setup_mac_q36.py` |
| The Qwen3.6 engine's settings in the register, its result folders in the bench index, the guarded check script | `data/settings-register.json` (13 `STRATA_Q36M_*` entries), `bench/results/README.md` (two rows), `scripts/check_guarded.sh` |
| The M2 validation records, the Metal roadmap and review, the sandbox notes | `docs/QWEN36_METAL*.md`, `docs/m2-sessions/`, `docs/M2_*.md`, `docs/ENVIRONMENT.md`, `docs/blog/`, `bench/results/2026-10-*-m2-q36-*` |

## Syncing with upstream

Merge a release tag. Do not apply path-scoped diffs by hand again.

```
git fetch origin --tags
git merge v0.2.15          # the next upstream tag
```

Conflicts can only appear in the files of the table above. For each one, take upstream's change and keep the fork's
addition beside it. When upstream fixes something the fork also fixed, take upstream's fix and drop the fork's.

After a merge:

1. Upstream's fast checks (`tools/check`: the Python suites, the app's typecheck, tests and build, the docs checks).
   On Linux run `python tools/check`. On a Mac run `scripts/check_guarded.sh`, never `tools/check` in a checkout you
   serve models from. On a Mac `setup.py` hands its whole command line to `metal/setup_mac.py`, so a test that calls
   setup for real runs a real setup: on 2026-11-14 that rewrote a run config in the live checkout and started its
   model, and in a clean worktree it began downloading a model (58 GB). The two tests found then are fixed; upstream
   runs the Python suites on Linux and Windows only, so another can appear. The script runs the checks in a scratch
   worktree with a scratch `HOME`, no outside network and no write access to the checkout or the model folders:

   ```
   STRATA_GUARD_READONLY=$HOME/w/Strata-data:$HOME/Documents/Strata scripts/check_guarded.sh
   ```

   Many tests fail on a Mac on pristine upstream too. Compare with the same run of the upstream tag
   (`scripts/check_guarded.sh v0.2.14.1`) before reading a failure as the fork's.
   After adding a `STRATA_*` setting to the engine, add it to `data/settings-register.json` (run
   `python tools/settings_register.py`, keep only the new entries, fill in `status`, `owner`, `note`); after adding a
   folder under `bench/results`, run `python tools/bench_results_index.py` in a clean checkout.
2. Build the Metal engine (`cmake -S metal -B build-metal -G Ninja -DMETAL_CPP_DIR=third_party/metal-cpp`, then
   `cmake --build build-metal --target strata`) and the Qwen3.6 tests (`q36_compile_test`, `q36_chunk_parity`,
   `prefix_plan_test`, `moe_sim`).
3. Run the Qwen3.6 gates on the Mac with the model loaded. The commands are in [QWEN36_METAL.md](QWEN36_METAL.md).

Push the upstream tags the fork merged (`v0.2.2`, `v0.2.3.1`, `v0.2.13.1`, `v0.2.14`, `v0.2.14.1` so far) to the fork, not every upstream tag.

### Names the two sides share

Upstream's model tables are keyed by the bare size name, and Unsloth uses the same size names for different models.
`UD-Q2_K_XL`, `UD-Q4_K_XL` and `UD-Q6_K_XL` are Qwen3.8-Flash-Next sets in `setup.py` `MODELS` and in
`strata_control/mac_fit.py` `MAC_MODELS`, and they are also Qwen3.6 sizes in `OWN_PATH_FAMILIES["qwen36"]`. Never add a
Qwen3.6 size to `MODELS`, `MAC_MODELS` or a label map under its bare name: it would replace or relabel the Flash-Next
set. Reach Qwen3.6 through its family (`own_size("qwen36", ...)`, `--family qwen36`).

`setup.py` is upstream's file, byte for byte. Its `OWN_PATH_FAMILIES["qwen36"]` lists the sizes upstream's CUDA and
HIP path runs; a size added there is offered on every PC and, for a PC with little RAM, recommended. The size the Mac
runs is in `strata_control/mac_q36.py` and nowhere else.

## Open work after the 2026-11-14 merge

- **Qwen3.6 setup on a Mac: ported, not yet run end to end on the Mac.** `./setup.sh --setup --family qwen36` goes to
  `metal/setup_mac_q36.py` (a file of its own; `metal/setup_mac.py` only hands the run over). It writes the run config
  the way `setup.py` does for this family on a PC: `"arch": "qwen35moe"`, `--max-context`, `--mtp` as a draft count.
  A run config written before the merge has no `"arch"`, so the rest of Strata reads it as a Flash-Next config: run
  setup again (the model file is found where it is and not downloaded again), or add `"arch": "qwen35moe"` by hand.
  `strata launch` on a Mac installs upstream's packaged Metal engine, which has no Qwen3.6 path: use `./setup.sh`
  from this checkout until the fork publishes its own engine package.
- **Qwen3.6 names in the app.** The labels "Compact (MTP)" and so on were dropped from
  `apps/host/src/lib/native-catalog.ts` for the same reason; the sizes show their plain names.
- **The Metal gates on the merged engine.** See the merge's pull request for what was run and what was not.

## Releasing from the fork

Upstream has no release workflow. It builds on build boxes without a key and signs on one release PC
([packaging/UPDATES.md](../packaging/UPDATES.md)). The fork follows the same split, with three choices of its own:

- **The tree keeps upstream's addresses and upstream's key.** The fork's release address and public feed key
  (`packaging/fork/feed-key.pub.json`, key id `e52bcd0bc55781a9`) are written into the sources when a package is
  built, by `scripts/fork_release.py bake`. So upstream's tests that pin its addresses still hold, and a merge does
  not conflict there. A source checkout therefore still reads upstream's feed: only an installed fork release reads
  the fork's.
- **Upstream's engines and drafters are not built or uploaded again.** `scripts/fork_release.py carry` checks
  upstream's signed feed against upstream's key and copies its engine and drafter entries, with upstream's
  addresses, sizes and SHA-256, into the fork's feed. A fork Control trusts both keys: the fork's for the feed, and
  upstream's because each engine package carries upstream's signature. Upstream's Metal engine is left out (it has
  no Qwen3.6 path); the fork's own Metal engine package is open work. There is no DeepSeek member.
- **The feed sequence is upstream's times 1000, plus N** (`scripts/fork_release.py sequence`). A PC refuses a feed
  older than the last one it accepted, so a PC that ran upstream needs a higher number. A PC that took a fork feed
  refuses upstream's feed afterwards.

The steps of a release `vX-mbp.rtx.N`:

1. Push the commit to a branch named `fork-release/...`, or run the "Fork release build" workflow by hand. It bakes,
   builds the Terminal for Linux x86_64 and macOS arm64, smoke-tests it and packs it as the feed's Control package.
   Download the two `fork-control-*` artifacts into one folder `RELEASE` (`control/`, `install.sh`, `install.ps1`).
2. Download upstream's `strata-feed.json` and `strata-feed.json.sig` of the same version, then:

   ```
   python scripts/fork_release.py carry --upstream-feed strata-feed.json --upstream-sig strata-feed.json.sig \
       --upstream-tag vX --out RELEASE
   ```

3. On the owner's machine, with the key (the only step that reads it):

   ```
   python packaging/feed_tool.py build-feed --out RELEASE --key KEYDIR/feed-key.json \
       --sequence $(python scripts/fork_release.py sequence --upstream-feed strata-feed.json -n N) \
       --engine RELEASE/engine --drafters RELEASE/drafters --control RELEASE/control --no-dsv41-carry
   python packaging/feed_tool.py verify-feed RELEASE --pub packaging/fork/feed-key.pub.json
   ```

4. Tag the commit `vX-mbp.rtx.N` and publish a normal (not pre-release) GitHub release on `kingdonb/Strata` with
   `strata-feed.json`, `strata-feed.json.sig`, `install-manifest.txt`, `install.sh`, `install.ps1` and the two
   Control archives as flat assets. The one-line installers and `strata update` read `releases/latest/download`.
5. On a PC: `curl -fsSL https://github.com/kingdonb/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh`.

The desktop app is not part of a fork release yet: it would need a Tauri updater key and its own addresses
(`apps/desktop/src-tauri/tauri.conf.json`, `apps/host/server/updates.mjs`).

### The two machines a fork release is for (owner, 2026-11-14)

| Machine | What it runs | What a release needs for it |
| --- | --- | --- |
| `mbp`: a MacBook with an M2 and 24 GB (about 18 GB of it wired while the model runs). The owner's own computer; it has to work offline. The signing keys are made and kept here. | Qwen3.6-35B-A3B on this fork's Metal engine | The fork's own Metal engine package (`macos-arm64`, `metal`), built on this Mac, and the fork's Control |
| `rtx`: `chudmax`, Ubuntu on x86_64 (kernel 7.0), 64 GB of RAM, one NVIDIA RTX 5060 Ti with 16 GB | Qwen3.8-Flash-Next on upstream's CUDA engine | The fork's Control for `linux-x86_64`. The engine is upstream's `strata-engine-<version>-linux-x86_64-cuda-cuda13` package, signed again with the fork's key (`feed_tool.py sign-package`) or trusted through upstream's key |

### Names and version numbers

- A fork release is tagged with upstream's version and the suffix `-mbp.rtx.N`: `v0.2.14-mbp.rtx.1` was the first
  release, on upstream v0.2.14; `v0.2.14.1-mbp.rtx.1` followed the same day (the suffix names the two machines above; `N` counts the fork's releases on that base).
- The suffix belongs to the git tag and the GitHub release only. Versions inside the signed feed must be plain numbers
  with three or four parts: `strata_control/feed.py` `version_key` refuses anything else ("is not a version number"),
  and where a suffix is allowed (the installed app's version, `strata_cli/handover.py`) it reads as a pre-release and
  sorts below the same number. So the packages of a fork release carry upstream's numbers (Control 0.1.60.1,
  engine 0.1.60).
- Not decided: how a second fork release on the same upstream base reaches a machine that already has the first.
  `strata update` only installs a higher version, and upstream owns the next numbers. Until that is settled, such a
  machine is updated with `strata update --force` or installed again.

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
