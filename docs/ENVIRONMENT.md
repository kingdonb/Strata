# Sandbox and environment facts

Written for the agents who work on this fork in a Helix sandbox. One section per fact, so a new
ticket stops rediscovering the terrain. Every entry carries a **status** and a **source**:

- `verified` — the check was run in a sandbox and passed, on the date given.
- `reported` — the fact comes from a session or ticket record; the owning place is named (and, where
  it could not be confirmed from here, that is said).

Rule: this page never states more than its evidence. When you re-verify an entry, update its date
and note what you ran. When a fact moves (a version floor, a reachable host), fix it here in the
same PR that discovers the change.

## Python — no pip, no ensurepip on the system interpreter

The sandbox's system `python3` is **3.13 with no pip and no ensurepip** (`python3 -m pip` and
`python3 -m ensurepip` both fail: "No module named …"). A working pip comes from bootstrapping into
a venv:

```sh
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o .scratch/get-pip.py
.venv/bin/python .scratch/get-pip.py --quiet
```

Runtime deps come from `requirements.txt` (pinned); build deps (cmake, ninja) from
`requirements-build.txt`, only when a local compile needs them. `scripts/sandbox_setup.sh` runs this
whole sequence for you.

- Status: **verified** 2026-10-25 (3.13.7, both modules absent; the bootstrap produced pip 26.2.2).
- Source: probe on this repo; first recorded in helix-specs `design/tasks/000709_upstream-ci-bundled/design.md`.

## `/tmp` — treat as scratch that may vanish

The week's session records say `/tmp` is **ephemeral across terminal invocations** in these sandboxes.
A probe on 2026-10-25 in this sandbox found `/tmp` persisting across two terminal invocations, so the
behavior is not universal — but the standing rule costs nothing to honor:

**Scratch lives under the project directory and is deleted after use.** Nothing long-lived goes in
`/tmp` (test reports, venvs, fetched artifacts).

- Status: **reported** (helix-specs `design/tasks/000709_upstream-ci-bundled/design.md`), with the
  2026-10-25 counter-observation above.
- Source: session records; this page's scripts follow the rule regardless.

## Node — ≥ 24.12.0 for `apps/host`

Sandboxes carry Node ≥ 24.12. Verified today: `node v24.22.0`, `npm 11.19.1`. The floor is not
conventional — `apps/host/package.json` declares `"engines": { "node": ">=24.12.0" }`. Verify the
version, don't assume it; npm warns (and some tooling refuses) below the floor.

- Status: **verified** 2026-10-25 (`node --version` → v24.22.0 ≥ the engines floor).
- Source: `apps/host/package.json`; probe on this repo.

## `SQLX_OFFLINE=true` for worker builds

Where a ticket builds the Rust worker that uses sqlx, set `SQLX_OFFLINE=true` — the prepared query
metadata lives in the repo's `.sqlx/` directory, so builds need no live database.

- Status: **reported** — the owning repo is not this one; `sqlx` appears nowhere in the Strata or
  mecris trees (checked 2026-10-25). The fact comes from the week's session records; confirm the
  owning repo and the exact crate set on the ticket that needs it.
- Source: task 000687's charter (the owner's re-scope note, 2026-10-25).

## `wit-bindgen` — the CLI is a different crate

`cargo install wit-bindgen` installs the **library** crate — the command-line generator is a
separate crate, `wit-bindgen-cli`. If a Makefile (or a doc) says `cargo install wit-bindgen` and the
next step calls `wit-bindgen …` and fails with "command not found", that is why.

- Status: **reported** — no `cargo install wit-bindgen` line exists in the Strata or mecris trees
  (checked 2026-10-25); the gotcha was hit in a session whose repo is not reachable from here.
  Confirm the exact Makefile/target on the ticket that needs it.
- Source: task 000687's charter (the owner's re-scope note, 2026-10-25).

## GitHub — reachable from sandboxes; Actions logs need a token

`api.github.com` and `raw.githubusercontent.com` are reachable from the sandboxes (both returned 200
on 2026-10-25). Two rules:

- **Reading the repo, releases and metadata anonymously works.** Tags, file contents, workflow runs
  and job listings are all public reads.
- **Downloading Actions job logs 403s without a token.** Anonymous `GET …/actions/jobs/<id>/logs`
  returns `403 Must have admin rights to Repository.` (probed 2026-10-25, run 40213677056, job
  121230110062). For log diagnosis use the token recipe below.

### The token recipe (CI-log diagnosis without the owner)

The project's Helix secret `GITHUB_TOKEN` is a **fine-grained personal access token with Actions:
read and Metadata: read — nothing else** (no write scope; the optional Issues/PR-read extension is
an owner decision, tracked as an open question in ticket 000687). The operator creates and stores
it with the Helix CLI — values are write-only, listings return names only:

```sh
helix secret create -n GITHUB_TOKEN -v github_pat_… -p prj_01m4vp6ttsaertqw6zed8ew8vm
```

A project-scoped secret is injected into this project's spec-task sessions when the container is
created — adding it does not reach already-running sessions; the next session picks it up.

Diagnosing a red CI run, with plain curl (no `gh` in the sandboxes):

```sh
# 1. the failed run's jobs (public read; the run id comes from the PR's checks or the run URL)
curl -sS "https://api.github.com/repos/kingdonb/Strata/actions/runs/<run-id>/jobs"

# 2. the failing job's log (needs the token; follows one redirect)
curl -sSL -H "Authorization: Bearer $GITHUB_TOKEN" \
  "https://api.github.com/repos/kingdonb/Strata/actions/jobs/<job-id>/logs"

# 3. paste the failing section into the ticket record — the diagnosis lives with the ticket
```

- Status: **verified** 2026-10-25 (the reachability, the anonymous 403, and the token path — the
  first secret-backed fetch succeeded that day: run 40210736015's job list read, failing job
  121221201459's log fetched with the token, HTTP 200, the failure signature
  `test_the_portable_watcher_ends_the_child_when_the_server_is_killed` present in the log). The
  token stays read-only (Actions: read + Metadata: read) and expires ~90 days from 2026-10-25 —
  after that, log fetches 403 again until the operator rotates the secret.
- Source: probe on this repo; the lesson from the post-merge red diagnosed in
  helix-specs `design/tasks/000709_upstream-ci-bundled/`.

## `python -S` / `-I` — site-packages go invisible

Running Python with `-S` (or `-I`) skips the `site` machinery: pip-installed packages stay installed
but become invisible to imports and to `from importlib.util import find_spec`. Verified 2026-10-25
in a venv with pip installed: under `-S`, `find_spec('pip')` is `None`; without it, it is found.

**Never gate behavior on a package-visibility probe inside a site-less process** — the probe answers
"not installed" for packages that are installed. The CI lesson behind this: the smoke that runs under
`-S` must assert behavior (exit codes, output), not importability.

- Status: **verified** 2026-10-25.
- Source: helix-specs `design/tasks/000709_upstream-ci-bundled/design.md` (the bare-interpreter
  preflight work); probe on this repo.

## Public RPC endpoints — rate limits, rotation, backoff

Public RPC endpoints rate-limit aggressively. The rotation list and the backoff pattern are
documented in the **nocknock** repo (`urmanac/nocknock`) — cross-referenced here only; this fork has
no RPC traffic. When a ticket here touches anything RPC-shaped, read that repo's pattern first
instead of inventing a retry loop.

- Status: **reported** (external; noted for cross-reference in task 000687's charter).
- Source: `urmanac/nocknock`, per the charter note 2026-10-25.

## The sim pattern — host-verify kernel semantics without the M2

The standing way to verify Metal kernel **semantics** on a plain Linux sandbox (no macOS, no GPU):
the pure-C++ host sims in `metal/tests/`. They transcribe the kernels loop-for-loop and check the
arithmetic against reference code:

```sh
g++ -O2 -std=c++20 -I metal/src -I third_party/ggml \
  metal/src/ref_quant.cpp metal/tests/moe_sim.cpp -o .scratch/moe_sim && .scratch/moe_sim
```

Verified 2026-10-25 on this repo: builds clean with g++ 15 and prints `moe_sim: all stages OK
(55 checks)` in seconds. `metal/tests/q36_chunk_parity.cpp` is the same family (the M2's tests glob
builds it with the other parity programs).

What the sim covers and what it does not: **semantics** (routing, dequant, GEMV, the expert chain)
are host-verifiable; **runtime** (pipeline creation, dispatch behavior, tok/s) needs the M2 —
that line is the working agreement's, and this page only records it.

- Status: **verified** 2026-10-25.
- Source: `metal/tests/moe_sim.cpp` header; `docs/QWEN36_METAL.md` §11; probe on this repo.
