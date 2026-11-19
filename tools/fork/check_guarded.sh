#!/bin/sh
# tools/fork/check_guarded.sh - run upstream's fast checks (tools/check) on a Mac without touching this checkout,
# the installed models or the network.
#
#   tools/fork/check_guarded.sh [REV] [-- tools/check options]
#   tools/fork/check_guarded.sh                      # HEAD, every step
#   tools/fork/check_guarded.sh v0.2.13.1 -- --only py-tests,py-serve
#
# Why this exists: on a Mac setup.py hands its whole command line to engine/metal/setup_mac.py before anything else, so a
# test that calls setup for real runs a real, unattended setup.  On 2026-11-14 that rewrote a run config in the
# checkout the suite ran in and started its model; in a clean worktree it began downloading a model (58 GB).  The
# two tests found then are fixed, but upstream's CI runs the Python suites on Linux and Windows only, so nothing
# keeps the next one from appearing.
#
# What it does:
#   * checks REV out as a detached worktree under a scratch folder (this checkout is not switched or written);
#   * gives the run a scratch HOME, so no settings file points it at real model folders;
#   * runs it under sandbox-exec: outbound network only to this machine (no download can start), and no writes to
#     this checkout or to the folders named in STRATA_GUARD_READONLY (colon-separated; name your model folders).
#
# The scratch folder is kept (logs, result.json) and printed at the end.  macOS only.
set -eu

[ "$(uname -s)" = "Darwin" ] || { echo "check_guarded.sh: macOS only (on Linux run python tools/check)" >&2; exit 2; }
command -v sandbox-exec >/dev/null 2>&1 || { echo "check_guarded.sh: sandbox-exec not found" >&2; exit 2; }

REV=HEAD
if [ $# -gt 0 ] && [ "$1" != "--" ]; then REV=$1; shift; fi
[ $# -gt 0 ] && [ "$1" = "--" ] && shift

ROOT=$(git rev-parse --show-toplevel)
PY=${STRATA_CHECK_PYTHON:-python3}
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/strata-check-XXXXXX")
TREE=$SCRATCH/tree
HOME_DIR=$SCRATCH/home
mkdir -p "$HOME_DIR"

git -C "$ROOT" worktree add --quiet --detach "$TREE" "$REV"
# the app's packages are large and identical for one lockfile: share this checkout's when it has them
if [ -d "$ROOT/app/host/node_modules" ] && cmp -s "$ROOT/app/host/package-lock.json" "$TREE/app/host/package-lock.json"; then
    ln -s "$ROOT/app/host/node_modules" "$TREE/app/host/node_modules"
fi

PROFILE=$SCRATCH/guard.sb
{
    echo '(version 1)'
    echo '(allow default)'
    echo '(deny network-outbound)'
    echo '(allow network-outbound (remote ip "localhost:*"))'
    echo '(allow network-outbound (remote unix-socket))'
    printf '(deny file-write* (subpath "%s")' "$ROOT"
    OLD_IFS=$IFS; IFS=:
    for folder in ${STRATA_GUARD_READONLY:-}; do
        [ -n "$folder" ] && printf ' (subpath "%s")' "$folder"
    done
    IFS=$OLD_IFS
    echo ')'
    # a worktree's git state lives in this checkout's .git; the shared node_modules keeps a build cache
    printf '(allow file-write* (subpath "%s/.git") (subpath "%s/app/host/node_modules"))\n' "$ROOT" "$ROOT"
} > "$PROFILE"

status=0
( cd "$TREE" && HOME="$HOME_DIR" sandbox-exec -f "$PROFILE" "$PY" tools/check --keep --scratch "$SCRATCH/check" \
    --json "$SCRATCH/result.json" "$@" ) || status=$?

echo
echo "check_guarded.sh: logs and result.json are in $SCRATCH"
echo "check_guarded.sh: remove the worktree with: git worktree remove --force $TREE"
exit $status
