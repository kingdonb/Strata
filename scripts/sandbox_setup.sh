#!/bin/sh
# sandbox_setup.sh — one-shot provisioning for a fresh Strata sandbox ticket.
#
# Idempotent: safe to re-run on a prepared checkout; completed steps are skipped or
# cheaply re-verified, and the capabilities report prints either way.
#
# Usage:
#   scripts/sandbox_setup.sh [--build-requirements] [--venv DIR]
#
# What it does, in order (see docs/ENVIRONMENT.md for the facts behind each probe):
#   1. python >= 3.10 probe + venv (bootstrapped with get-pip.py when pip is absent)
#   2. runtime deps from requirements.txt (pinned)
#   3. optional build deps from requirements-build.txt   (--build-requirements)
#   4. node/npm vs the apps/host engines floor (>= 24.12.0)
#   5. SQLX_OFFLINE state (reported; owning repo to confirm)
#   6. GitHub reachability (api.github.com, raw.githubusercontent.com)
#   7. g++ — the host-sim toolchain (metal/tests/moe_sim.cpp)
#
# Scratch lives under the project dir (.scratch/), never /tmp, and is deleted after
# the run. The venv persists (git-ignored). Exit 0 when every required probe passes,
# exit 1 naming the missing ones.

set -u

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_ROOT" || exit 1

BUILD_REQS=0
VENV="$REPO_ROOT/.venv"
while [ $# -gt 0 ]; do
    case "$1" in
        --build-requirements) BUILD_REQS=1 ;;
        --venv) VENV="$2"; shift ;;
        *) echo "unknown argument: $1 (usage: scripts/sandbox_setup.sh [--build-requirements] [--venv DIR])" >&2; exit 2 ;;
    esac
    shift
done

SCRATCH="$REPO_ROOT/.scratch"
REPORT=""

# ---- tiny report helpers ----------------------------------------------------
row() { # row <probe> <result> <required?> <status>
    REPORT="$REPORT$(printf '%-28s %-34s %-10s %s\n' "$1" "$2" "$3" "$4")
"
}
REQUIRED_MISSING=""

# ---- 1. python + venv -------------------------------------------------------
PY_OK=0
PYTHON_BIN=""
PYVER=""
if command -v python3 >/dev/null 2>&1; then
    PYVER=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo unknown)
    PYMAJ=$(printf '%s' "$PYVER" | cut -d. -f1)
    PYMIN=$(printf '%s' "$PYVER" | cut -d. -f2)
    if [ "${PYMAJ:-0}" -eq 3 ] && [ "${PYMIN:-0}" -ge 10 ]; then
        PY_OK=1
    fi
fi

VENV_STATUS="absent"
if [ "$PY_OK" -eq 1 ]; then
    if [ ! -x "$VENV/bin/python" ]; then
        # The sandbox python has no pip/ensurepip (docs/ENVIRONMENT.md): create the
        # venv without pip, then bootstrap pip into it from get-pip.py.
        if ! python3 -m venv --without-pip "$VENV" >/dev/null 2>&1; then
            PY_OK=0
            VENV_STATUS="venv creation failed"
        else
            mkdir -p "$SCRATCH"
            if curl -fsSL -m 60 https://bootstrap.pypa.io/get-pip.py -o "$SCRATCH/get-pip.py" \
               && "$VENV/bin/python" "$SCRATCH/get-pip.py" --quiet >/dev/null 2>&1; then
                VENV_STATUS="created + pip bootstrapped"
            else
                PY_OK=0
                VENV_STATUS="created, pip bootstrap FAILED"
            fi
        fi
    else
        VENV_STATUS="exists (reused)"
    fi
fi

if [ "$PY_OK" -eq 1 ]; then
    row "python >= 3.10" "$PYVER" required OK
    row "venv ($VENV)" "$VENV_STATUS" required OK
else
    if [ -z "$PYVER" ] || [ "$PYVER" = "unknown" ]; then
        row "python >= 3.10" "python3 missing or unreadable" required FAIL
        REQUIRED_MISSING="$REQUIRED_MISSING python>=3.10"
    elif [ "$VENV_STATUS" = "created, pip bootstrap FAILED" ]; then
        row "python >= 3.10" "$PYVER" required OK
        row "venv ($VENV)" "$VENV_STATUS" required FAIL
        REQUIRED_MISSING="$REQUIRED_MISSING venv/pip"
    elif [ "$VENV_STATUS" = "venv creation failed" ]; then
        row "python >= 3.10" "$PYVER" required OK
        row "venv ($VENV)" "creation failed" required FAIL
        REQUIRED_MISSING="$REQUIRED_MISSING venv"
    fi
fi

# ---- 2. runtime requirements ------------------------------------------------
if [ "$PY_OK" -eq 1 ]; then
    if "$VENV/bin/python" -m pip install -r requirements.txt >/dev/null 2>&1; then
        PIPVER=$("$VENV/bin/python" -m pip --version 2>/dev/null | awk '{print $2}')
        NUMPYVER=$("$VENV/bin/python" -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo "?")
        row "requirements.txt" "installed (pip $PIPVER, numpy $NUMPYVER)" required OK
    else
        row "requirements.txt" "install FAILED" required FAIL
        REQUIRED_MISSING="$REQUIRED_MISSING requirements.txt"
    fi
else
    row "requirements.txt" "skipped (no working venv)" required FAIL
    REQUIRED_MISSING="$REQUIRED_MISSING requirements.txt"
fi

# ---- 3. optional build requirements -----------------------------------------
if [ "$BUILD_REQS" -eq 1 ]; then
    if [ "$PY_OK" -eq 1 ] && "$VENV/bin/python" -m pip install -r requirements-build.txt >/dev/null 2>&1; then
        row "requirements-build.txt" "installed" optional OK
    else
        row "requirements-build.txt" "install FAILED" optional FAIL
    fi
else
    row "requirements-build.txt" "not requested" optional SKIP
fi

# ---- 4. node / npm vs the apps/host engines floor ---------------------------
NODE_FLOOR="24.12.0"
if command -v node >/dev/null 2>&1; then
    NODEVER=$(node -p 'process.versions.node' 2>/dev/null || echo unknown)
    LOWEST=$(printf '%s\n%s\n' "$NODE_FLOOR" "$NODEVER" | sort -V | head -1)
    if [ "$LOWEST" = "$NODE_FLOOR" ]; then
        row "node >= $NODE_FLOOR" "$NODEVER" "only host work" OK
    else
        row "node >= $NODE_FLOOR" "$NODEVER (below floor)" "only host work" FAIL
    fi
    if command -v npm >/dev/null 2>&1; then
        row "npm" "$(npm --version 2>/dev/null)" "only host work" OK
    else
        row "npm" "missing" "only host work" FAIL
    fi
else
    row "node >= $NODE_FLOOR" "missing" "only host work" FAIL
fi

# ---- 5. SQLX_OFFLINE (reported fact; owning repo to confirm) ----------------
row "SQLX_OFFLINE" "env: ${SQLX_OFFLINE:-<unset>} (reported; see docs/ENVIRONMENT.md)" "only worker builds" INFO

# ---- 6. GitHub reachability --------------------------------------------------
if curl -fsS -m 10 https://api.github.com/rate_limit >/dev/null 2>&1 \
   && curl -fsS -m 10 -o /dev/null --head https://raw.githubusercontent.com/kingdonb/Strata/main/README.md 2>/dev/null; then
    row "github reachability" "api.github.com + raw.githubusercontent.com" "only sync/log work" OK
else
    row "github reachability" "unreachable (network grants are per-sandbox)" "only sync/log work" FAIL
fi

# ---- 7. g++ (the host-sim toolchain) ----------------------------------------
if command -v g++ >/dev/null 2>&1; then
    row "g++ (host sims)" "$(g++ --version 2>/dev/null | head -1 | awk '{print $1, $NF}')" "only sim work" OK
else
    row "g++ (host sims)" "missing" "only sim work" FAIL
fi

# ---- report ------------------------------------------------------------------
rm -rf "$SCRATCH"
echo ""
echo "== Strata sandbox capabilities ==============================================="
echo "probe                         result                             required?  status"
printf '%s' "$REPORT"
echo ""
echo "venv:      $VENV"
echo "scratch:   $SCRATCH (deleted after this run)"
echo "details:   docs/ENVIRONMENT.md"
echo "==============================================================================="

if [ -n "$REQUIRED_MISSING" ]; then
    echo "MISSING (required):$REQUIRED_MISSING"
    exit 1
fi
echo "All required capabilities ready."
