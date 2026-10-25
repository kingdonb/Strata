#!/bin/sh
# upstream_survey.sh — the fork's recurring upstream sync, made cheap and complete.
#
# Wraps the four-command survey recorded in docs/FORK.md (fetch tags; per-area name-only
# diff; per-area commit subjects; release-notes pointer), applies the skip list on
# demand, and generates the FORK.md ledger row. It never merges, rebases or pushes:
# the sync stays path-scoped and deliberate, one PR per slice (FORK.md's standing rule).
#
# Usage:
#   scripts/upstream_survey.sh <from> <to> [--no-fetch] [--skip-list] [--ledger] [--apply]
#                              [--skip-file FILE]
#
#   <from> <to>   upstream refs — normally tags (e.g. v0.2.2 v0.2.3.1). <from> is the
#                 previous synced-through (the merge base), <to> the new one.
#   --no-fetch    skip `git fetch upstream --tags`
#   --skip-list   print the effective skip list (paths only)
#   --ledger      print a ready-to-paste docs/FORK.md ledger row for the range
#   --apply       apply the skip list to the worktree: files upstream modified since
#                 <from> are reverted to <from>'s version; files upstream added are
#                 deleted (FORK.md's recorded method). Refuses on main and on a dirty
#                 tree. Commit the result on your sync branch yourself.
#   --skip-file   use FILE instead of scripts/upstream_skip.txt (a per-sync list: the
#                 standing list plus that sync's uncommented amendments)
#
# Validation anchor (2026-10-25): --ledger reproduces both recorded rows exactly —
#   3d1fb480..011c9e60  = 404 commits, 590 files, +52967/−2271  (the v0.2.2 row)
#   011c9e60..7ddc17a3  = 131 commits, 294 files, +14913/−661   (the v0.2.3.1 row)

set -u

UPSTREAM_REMOTE=upstream
DEFAULT_SKIP_FILE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/upstream_skip.txt"
AREAS="apps bench bin cmake data docker docs include integrations metal packaging ref serve src strata_cli strata_control sycl tests third_party tools .github"

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

FROM=""
TO=""
FETCH=1
SKIP_LIST=0
LEDGER=0
APPLY=0
SKIP_FILE="$DEFAULT_SKIP_FILE"
while [ $# -gt 0 ]; do
    case "$1" in
        --no-fetch)  FETCH=0 ;;
        --skip-list) SKIP_LIST=1 ;;
        --ledger)    LEDGER=1 ;;
        --apply)     APPLY=1 ;;
        --skip-file) SKIP_FILE="$2"; shift ;;
        -h|--help)   usage ;;
        *) if [ -z "$FROM" ]; then FROM="$1"; elif [ -z "$TO" ]; then TO="$1";
           else echo "unknown argument: $1" >&2; usage; fi ;;
    esac
    shift
done
[ -n "$FROM" ] && [ -n "$TO" ] || usage

cd_to_root() {
    ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "not inside a git repository" >&2; exit 2; }
    cd "$ROOT" || exit 1
}
cd_to_root

# ---- resolve refs, fetch tags -------------------------------------------------
for ref in "$FROM" "$TO"; do
    git rev-parse -q --verify "$ref^{commit}" >/dev/null 2>&1 \
        || { echo "error: '$ref' does not resolve (fetch upstream first: git remote add $UPSTREAM_REMOTE https://github.com/Niko1221/Strata.git && git fetch $UPSTREAM_REMOTE --tags)" >&2; exit 2; }
done

if [ "$FETCH" -eq 1 ]; then
    if ! git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
        echo "note: no '$UPSTREAM_REMOTE' remote — skipping the fetch (add it per docs/FORK.md:"
        echo "  git remote add $UPSTREAM_REMOTE https://github.com/Niko1221/Strata.git)"
    else
        git fetch "$UPSTREAM_REMOTE" --tags >/dev/null 2>&1 && echo "fetched $UPSTREAM_REMOTE --tags" || echo "note: fetch failed — working from already-fetched refs"
    fi
fi

# ---- the survey (the four commands, wrapped) -----------------------------------
echo "== upstream survey: $FROM..$TO ================================================="
ALL_FILES=$(git diff --name-only "$FROM" "$TO")
for area in $AREAS; do
    files=$(printf '%s\n' "$ALL_FILES" | grep -E "^$area/" | wc -l)
    [ "$files" -eq 0 ] && continue
    echo ""
    echo "--- $area/ — $files file(s) ---"
    # per-area commit subjects, capped
    total=$(git rev-list --count "$FROM..$TO" -- "$area")
    git log --oneline --no-decorate -12 "$FROM..$TO" -- "$area"
    [ "$total" -gt 12 ] && echo "... and $((total - 12)) more commits in $area/"
done
root_files=$(printf '%s\n' "$ALL_FILES" | grep -vE '^([a-zA-Z0-9_]+|\.github)/' | wc -l)
if [ "$root_files" -gt 0 ]; then
    echo ""
    echo "--- root files — $root_files file(s) ---"
    printf '%s\n' "$ALL_FILES" | grep -vE '^([a-zA-Z0-9_]+|\.github)/'
fi
release_notes=$(printf '%s\n' "$ALL_FILES" | grep -E '^docs/release-notes.*\.md$' || true)
if [ -n "$release_notes" ]; then
    echo ""
    echo "--- release notes (read these — the fourth survey command) ---"
    printf '%s\n' "$release_notes"
fi

# ---- skip list ------------------------------------------------------------------
effective_entries() {
    grep -vE '^[[:space:]]*#' "$SKIP_FILE" 2>/dev/null | sed 's/#.*//' | sed 's/[[:space:]]*$//' | grep -vE '^$'
}
if [ "$SKIP_LIST" -eq 1 ]; then
    echo ""
    echo "== effective skip list ($SKIP_FILE) =="
    n=0
    while IFS= read -r e; do [ -n "$e" ] && { echo "$e"; n=$((n + 1)); }; done <<EOF
$(effective_entries)
EOF
    echo "($n entries; reasons inline in $SKIP_FILE)"
fi

# ---- ledger row -----------------------------------------------------------------
if [ "$LEDGER" -eq 1 ]; then
    tosha=$(git rev-parse --short "$TO")
    fromsha=$(git rev-parse --short "$FROM")
    commits=$(git rev-list --count "$FROM..$TO")
    files=$(printf '%s\n' "$ALL_FILES" | wc -l)
    shortstat=$(git diff --shortstat "$FROM" "$TO")
    ins=$(printf '%s' "$shortstat" | sed -n 's/.* \([0-9,]*\) insertion.*/\1/p')
    del=$(printf '%s' "$shortstat" | sed -n 's/.* \([0-9,]*\) deletion.*/\1/p')
    fromsub=$(git log -1 --format=%s "$FROM")
    slices=$(for area in $AREAS; do
        c=$(git rev-list --count "$FROM..$TO" -- "$area"); [ "$c" -gt 0 ] && printf '%s(%s) ' "$area" "$c"
    done)
    echo ""
    echo "== FORK.md ledger row (paste, then replace the slices cell with the real decisions) =="
    echo "| $(date +%Y-%m-%d) | **$TO** (\`$tosha\`), range \`$fromsha..$tosha\` ($commits commits, $files files, +$ins/−$del) | \`$fromsha\` (\"$fromsub\") | PROPOSED slices: $slices — the operator picks |"
fi

# ---- apply the skip list --------------------------------------------------------
if [ "$APPLY" -eq 1 ]; then
    branch=$(git branch --show-current)
    [ "$branch" != "main" ] || { echo "error: --apply refuses to run on main — do it on a sync branch" >&2; exit 2; }
    git diff --quiet && git diff --cached --quiet \
        || { echo "error: --apply refuses a dirty tree — commit or stash first" >&2; exit 2; }
    echo ""
    echo "== applying the skip list (revert modified to $FROM, drop upstream-new leftovers) =="
    treverted=0; tdropped=0
    while IFS= read -r entry; do
        [ -n "$entry" ] || continue
        r=0; d=0
        # AMRT: modifications/additions/renames/types the range introduces. Pure upstream
        # deletions are declined implicitly — the fork keeps its copy untouched.
        changed=$(git diff --name-only --diff-filter=AMRT "$FROM" "$TO" -- "$entry")
        for f in $changed; do
            if git ls-files --error-unmatch -- "$f" >/dev/null 2>&1; then
                # the fork tracks it (HEAD, or staged-new from the recorded git-apply step)
                if git cat-file -e "$FROM:$f" 2>/dev/null; then
                    # worktree-only: never stage (git checkout <sha> -- <path> would)
                    git restore -q --source="$FROM" --worktree -- "$f" && r=$((r + 1))
                fi
                # else: the fork's own file, upstream has no merge-base copy — leave it
            elif [ -e "$f" ]; then
                # upstream-new, present as an untracked leftover (e.g. after the
                # recorded `git diff | git apply` step) — the fork does not carry it
                rm -f "$f" && d=$((d + 1))
            fi
            # else: absent from a clean fork tree — a past skip decision already stands
        done
        treverted=$((treverted + r)); tdropped=$((tdropped + d))
        [ $((r + d)) -gt 0 ] && echo "$entry: $r reverted, $d dropped" || echo "$entry: nothing to do in range"
    done <<EOF
$(effective_entries)
EOF
    echo ""
    echo "done: $treverted reverted, $tdropped dropped (worktree-only; nothing staged)."
    echo "Next, per docs/FORK.md: run the suites, record what was skipped and why in the PR,"
    echo "and keep scripts/upstream_skip.txt and FORK.md's table in the same PR."
fi
