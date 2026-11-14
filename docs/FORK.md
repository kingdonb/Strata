# About this fork (kingdonb/Strata)

This fork is upstream Strata plus a small set of additions. It is built from source on the owner's Mac. It does not
publish installers, engine packages or an update feed of its own yet; that is planned (see "Releasing from the fork").

- Base: upstream **v0.2.13.1** (`628f7495`), merged on 2026-11-14 with an ordinary `git merge`.
- Versions: upstream's (app and Terminal 0.2.13, Control 0.1.59.1, engine 0.1.59). The fork no longer keeps a version
  line of its own. A fork release will add a suffix to upstream's number (owner decision, 2026-11-14).
- Until the fork has its own feed, the built-in update address is still upstream's
  (`github.com/Niko1221/Strata/releases/latest/download`). `strata update` and the app's Update button would install
  upstream's engine, which does not have the Qwen3.6 Metal path. Do not use them on a machine that runs this fork's
  engine.

## What the fork adds

Everything else in the tree is upstream's file, byte for byte. `git diff v0.2.13.1 HEAD --stat` lists the whole
difference.

| Addition | Where |
| --- | --- |
| Qwen3.6-35B-A3B on Apple Silicon (Metal) | `metal/kernels/q36*.metal`, the K-quant kernels in `metal/kernels/quant.metal`, `metal/src/engine/q36_metal.cpp`, `metal/src/engine/q36_prefix_plan.hpp`, `metal/tests/q36_*`, `metal/tests/moe_sim.cpp`, `tools/qwen35/q36_*`; documented in [QWEN36_METAL.md](QWEN36_METAL.md) |
| Serial GPU dispatch for the Qwen3.6 graphs, the dispatch tally and the argument-count guard | `metal/src/mtl_rt.h`, `metal/src/mtl_rt.mm` (`set_serial_graphs`; off for every other engine) |
| The direct chat asks before a connected tool runs (Manual and Plan ask, Bypass runs) | `serve/server.py` (`GET /v1/mcp/tools`, `POST /v1/mcp/call`), `apps/host/server/tool-run.mjs`, `apps/host/server/index.mjs` (`/api/tools/run`), `apps/host/server/inference.mjs`, `apps/host/src/lib/store.ts`, `apps/host/src/app/ContextPanel.tsx`, `apps/host/src/components/ComposerControls.tsx` |
| The built-in Strata tools in the agent (`mcp__strata__*`) | `apps/host/server/native-control.mjs` (`strataToolSpec`), `apps/host/server/agent/` (`profile.mjs`, `tools-settings.mjs`, `plugins/strata-gate.mjs`, `agent-bridge.mjs`), `apps/host/src/components/SettingsTools.tsx`, strings `s9039`-`s9042` in `apps/host/src/locales/` |
| The built-in Strata tools in the engine's own chat | `serve/mcp.py` (`builtin_strata_server`, `"mcp": {"builtin_strata": false}` turns it off) |
| Apple Silicon temperature without root | `serve/telemetry.py` (`_AppleSiliconThermal`) |
| Qwen3.6 UD-Q2_K_XL in the model tables, and the Qwen3.6 rows in the Mac model list | `setup.py` (`QWEN36_FILES`, `OWN_PATH_FAMILIES`), `tools/strata_mcp.py` |
| The M2 validation records, the Metal roadmap and review, the sandbox notes | `docs/QWEN36_METAL*.md`, `docs/m2-sessions/`, `docs/M2_*.md`, `docs/ENVIRONMENT.md`, `docs/blog/`, `bench/results/2026-10-*-m2-q36-*` |

## Syncing with upstream

Merge a release tag. Do not apply path-scoped diffs by hand again.

```
git fetch origin --tags
git merge v0.2.14          # the next upstream tag
```

Conflicts can only appear in the files of the table above. For each one, take upstream's change and keep the fork's
addition beside it. When upstream fixes something the fork also fixed, take upstream's fix and drop the fork's.

After a merge:

1. `python tools/check` (upstream's fast checks: the Python suites, the app's typecheck, tests and build).
2. Build the Metal engine (`cmake -S metal -B build-metal -G Ninja -DMETAL_CPP_DIR=third_party/metal-cpp`, then
   `cmake --build build-metal --target strata`) and the Qwen3.6 tests (`q36_compile_test`, `q36_chunk_parity`,
   `prefix_plan_test`, `moe_sim`).
3. Run the Qwen3.6 gates on the Mac with the model loaded. The commands are in [QWEN36_METAL.md](QWEN36_METAL.md).

Push the upstream tags the fork merged (`v0.2.2`, `v0.2.3.1`, `v0.2.13.1` so far) to the fork, not every upstream tag.

### Names the two sides share

Upstream's model tables are keyed by the bare size name, and Unsloth uses the same size names for different models.
`UD-Q2_K_XL`, `UD-Q4_K_XL` and `UD-Q6_K_XL` are Qwen3.8-Flash-Next sets in `setup.py` `MODELS` and in
`strata_control/mac_fit.py` `MAC_MODELS`, and they are also Qwen3.6 sizes in `OWN_PATH_FAMILIES["qwen36"]`. Never add a
Qwen3.6 size to `MODELS`, `MAC_MODELS` or a label map under its bare name: it would replace or relabel the Flash-Next
set. Reach Qwen3.6 through its family (`own_size("qwen36", ...)`, `--family qwen36`).

## Open work after the 2026-11-14 merge

- **Qwen3.6 setup on a Mac.** `metal/setup_mac.py` and `strata_control/mac_fit.py` are upstream's files, so
  `./setup.sh --family qwen36` does not exist yet on this base. The fork's earlier version of that flow keyed
  `UD-Q2_K_XL` by its bare name (see above) and has to be ported onto upstream's family tables. A model that is
  already set up (an existing `strata-*.json` run config) starts as before.
- **Qwen3.6 names in the app.** The labels "Compact (MTP)" and so on were dropped from
  `apps/host/src/lib/native-catalog.ts` for the same reason; the sizes show their plain names.
- **The Metal gates on the merged engine.** See the merge's pull request for what was run and what was not.
- **`UD-Q2_K_XL` for Qwen3.6 is listed on every platform** because `OWN_PATH_FAMILIES` is shared. It was only run on
  the Metal engine.

## Releasing from the fork

Upstream's release process is in [packaging/UPDATES.md](../packaging/UPDATES.md) and nothing in it is tied to
upstream's GitHub account except three things the fork has to change before its first release:

1. **Keys.** The owner makes a feed key (`python packaging/feed_tool.py keygen`) and, for the desktop app, a Tauri
   updater key. The private halves stay outside git. The public halves go into `strata_control/update_keys.json`
   and `apps/desktop/src-tauri/tauri.conf.json`.
2. **Addresses.** `strata_cli/update_cmd.py` (`DEFAULT_FEED`), `packaging/install.sh`, `packaging/install.ps1`,
   `apps/desktop/src-tauri/tauri.conf.json` (the updater endpoint) and `apps/host/server/updates.mjs` name
   `Niko1221/Strata`.
3. **A release.** Signed engine and Control packages, `strata-feed.json` and its `.sig`, `install-manifest.txt`,
   `install.sh` and `install.ps1`, uploaded as the assets of a normal (not pre-release) GitHub release.

Outside `metal/` the fork's engine sources are upstream's, so on NVIDIA, AMD and Intel hardware a fork release only
needs its own Control package; the engine package can be upstream's.

## Owner decisions on record

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
