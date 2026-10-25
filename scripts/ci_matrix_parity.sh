#!/bin/sh
# ci_matrix_parity.sh — reproduce the two minimal-matrix CI jobs locally, pre-push.
#
# Catches exactly the class that reached CI once: a test that passes in a full dev
# environment but fails in the requirements-only one (the test_lifeline lesson —
# helix-specs design/tasks/000710_testlifeline-watcher). Nothing local ran the full
# server suite before; this script always does.
#
# Usage:
#   scripts/ci_matrix_parity.sh [--full]
#
# Legs, in CI's order (terminal-minimal-server.yml, then terminal-portable-monitor.yml):
#   A. requirements-only venv:
#        tools/minimal_server_smoke.py --report <scratch>/minimal-server-smoke.json
#        "$VENV/bin/python" -m unittest discover -s serve -p 'test_*.py'   (CI "step 6")
#   B. the bare system interpreter, no optional packages (the rich/textual-invisible
#      assertions — site-packages are simply absent):
#        tools/minimal_cli_smoke.py --report <scratch>/minimal-cli-smoke.json
#        packaging/monitor_bundle.py build <scratch>/strata-monitor.pyz
#        tools/portable_monitor_smoke.py --bundle ... --report ...
#      --full adds CI's textual legs (install requirements-monitor.txt into the leg-A
#      venv, the interactive smoke, and the POSIX pty smoke). Default-off keeps the
#      pre-push run fast; they exercise packaging, not the parity class.
#
# Known skew, printed, never hidden: CI runs Python 3.11 (leg A) and 3.10/3.11
# (leg B); locally this is whatever python3 is (3.13 in today's sandboxes). The pins
# carry per-version markers, so the run is the same commands and the same dependency
# class, not byte-identical deps.
#
# The venv bootstrap duplicates scripts/sandbox_setup.sh's on purpose — two scripts,
# ten lines, no shared helper to couple them.
#
# Scratch (venv, bundle, reports) lives under the project dir and is deleted after a
# green run; a red run keeps it for debugging (both paths are git-ignored). Stops at
# the first red with the failing command echoed. Exit 0 only when every leg ran green.

set -u

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_ROOT" || exit 1

FULL=0
[ "${1:-}" = "--full" ] && FULL=1

SCRATCH="$REPO_ROOT/.scratch"
VENV="$REPO_ROOT/.venv-parity"
SYSTEM_PY=python3

SUMMARY=""
FAIL_CMD=""
fail() { # fail <leg> <what>  — echoes the failing command and stops
    echo ""
    echo "PARITY RED: $1 — $2"
    echo "failing command: $FAIL_CMD"
    echo "scratch kept for debugging: $VENV, $SCRATCH"
    exit 1
}
note() { SUMMARY="$SUMMARY$1
"; }

rm -rf "$VENV"
mkdir -p "$SCRATCH"

PYVER=$("$SYSTEM_PY" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)
echo "== CI-matrix parity — local python $PYVER (CI: 3.11 leg A, 3.10/3.11 leg B) =="

# ---- the requirements-only venv (same recipe as sandbox_setup.sh) -----------
echo ""
echo "--- leg A: requirements-only venv ---"
if ! "$SYSTEM_PY" -m venv --without-pip "$VENV" >/dev/null 2>&1; then
    FAIL_CMD="$SYSTEM_PY -m venv --without-pip $VENV"
    fail "venv creation" "could not create $VENV"
fi
if ! "$VENV/bin/python" -m ensurepip >/dev/null 2>&1; then
    # sandbox pythons carry no pip/ensurepip (docs/ENVIRONMENT.md): bootstrap get-pip
    if ! curl -fsSL -m 60 https://bootstrap.pypa.io/get-pip.py -o "$SCRATCH/get-pip.py" \
       || ! "$VENV/bin/python" "$SCRATCH/get-pip.py" --quiet >/dev/null 2>&1; then
        FAIL_CMD="curl bootstrap.pypa.io/get-pip.py + venv python get-pip.py"
        fail "pip bootstrap" "could not provision pip into the venv"
    fi
fi
FAIL_CMD="\"$VENV/bin/python\" -m pip install -r requirements.txt"
"$VENV/bin/python" -m pip install -r requirements.txt >/dev/null 2>&1 \
    || fail "leg A setup" "requirements.txt install failed"

FAIL_CMD="\"$VENV/bin/python\" tools/minimal_server_smoke.py --report $SCRATCH/minimal-server-smoke.json"
"$VENV/bin/python" tools/minimal_server_smoke.py --report "$SCRATCH/minimal-server-smoke.json" \
    || fail "leg A smoke" "minimal_server_smoke.py failed (see its output above)"
echo "smoke report (CI prints this too):"
cat "$SCRATCH/minimal-server-smoke.json"

# the suite's status must be read from the interpreter, not from a pipe's last stage
FAIL_CMD="\"$VENV/bin/python\" -m unittest discover -s serve -p 'test_*.py'"
if ! "$VENV/bin/python" -m unittest discover -s serve -p 'test_*.py' >"$SCRATCH/serve-suite.log" 2>&1; then
    tail -15 "$SCRATCH/serve-suite.log"
    fail "leg A suite" "the full server suite failed in the requirements-only venv"
fi
tail -3 "$SCRATCH/serve-suite.log"
PYV_A=$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")
note "leg A (requirements-only venv, python $PYV_A): GREEN"

# ---- leg B: the bare system interpreter -------------------------------------
echo ""
echo "--- leg B: bare interpreter, no optional packages ---"
FAIL_CMD="$SYSTEM_PY tools/minimal_cli_smoke.py --report $SCRATCH/minimal-cli-smoke.json"
"$SYSTEM_PY" tools/minimal_cli_smoke.py --report "$SCRATCH/minimal-cli-smoke.json" \
    || fail "leg B smoke" "minimal_cli_smoke.py failed on the bare interpreter"
echo "smoke report (CI prints this too):"
cat "$SCRATCH/minimal-cli-smoke.json"

BUNDLE="$SCRATCH/strata-monitor.pyz"
FAIL_CMD="$SYSTEM_PY packaging/monitor_bundle.py build $BUNDLE"
"$SYSTEM_PY" packaging/monitor_bundle.py build "$BUNDLE" >/dev/null 2>&1 \
    || fail "leg B bundle" "monitor_bundle.py build failed"
FAIL_CMD="$SYSTEM_PY tools/portable_monitor_smoke.py --bundle $BUNDLE --report $SCRATCH/monitor-stdlib-smoke.json"
"$SYSTEM_PY" tools/portable_monitor_smoke.py --bundle "$BUNDLE" --report "$SCRATCH/monitor-stdlib-smoke.json" \
    || fail "leg B smoke" "the stdlib bundle smoke failed"
echo "stdlib bundle smoke report:"
cat "$SCRATCH/monitor-stdlib-smoke.json"
note "leg B (bare $SYSTEM_PY $PYVER, no optional packages): GREEN"

# ---- optional: CI's textual legs --------------------------------------------
if [ "$FULL" -eq 1 ]; then
    echo ""
    echo "--- leg B --full: the textual-installed legs (CI python 3.11 only) ---"
    FAIL_CMD="\"$VENV/bin/python\" -m pip install -r requirements-monitor.txt"
    "$VENV/bin/python" -m pip install -r requirements-monitor.txt >/dev/null 2>&1 \
        || fail "leg B --full" "requirements-monitor.txt install failed"
    FAIL_CMD="\"$VENV/bin/python\" tools/portable_monitor_smoke.py --bundle $BUNDLE --interactive --report $SCRATCH/monitor-interactive-smoke.json"
    "$VENV/bin/python" tools/portable_monitor_smoke.py --bundle "$BUNDLE" --interactive --report "$SCRATCH/monitor-interactive-smoke.json" \
        || fail "leg B --full" "the interactive Overview smoke failed"
    FAIL_CMD="\"$VENV/bin/python\" tools/portable_monitor_pty_smoke.py --bundle $BUNDLE --report $SCRATCH/monitor-pty-smoke.json"
    "$VENV/bin/python" tools/portable_monitor_pty_smoke.py --bundle "$BUNDLE" --report "$SCRATCH/monitor-pty-smoke.json" \
        || fail "leg B --full" "the POSIX pty restoration smoke failed"
    PYV_F=$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "?")
    note "leg B --full (textual installed, python $PYV_F): GREEN"
fi

# ---- summary + cleanup --------------------------------------------------------
rm -rf "$VENV" "$SCRATCH"
echo ""
echo "== parity summary ============================================================"
printf '%s' "$SUMMARY"
echo "interpreter: $SYSTEM_PY $PYVER (CI leg A: 3.11, leg B: 3.10/3.11 — same commands,"
echo "same dependency class; version skew is recorded, never hidden)"
echo "scratch cleaned: $VENV, $SCRATCH"
echo "==============================================================================="
echo "PARITY GREEN — safe to push."
