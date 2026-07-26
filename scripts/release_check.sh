#!/usr/bin/env bash
#
# Pre-commit release gate. FAIL-CLOSED.
#
# Replaces an ad-hoc one-liner that had three defects, all found on review:
#
#   1. THE ALARM DID NOT LOCK THE DOOR.
#        if grep ...; then echo "HITS"; else echo "clean"; fi && git commit
#      Both branches end in `echo`, which exits 0, so `&& git commit` ran in
#      BOTH cases. A detected leak printed a warning and committed anyway.
#
#   2. PART OF THE PATTERN WAS INERT.
#      `grep -E '(?i)term|...'` uses a PCRE inline flag that ERE does not
#      support. Verified: it returns exit 1 ("no match") against a file that
#      does contain the term. It did not error -- it silently never matched,
#      so the first alternative was dead in every scan. Use `grep -i`.
#
#   3. THE TERM LIST WAS IN THE SCANNER.
#      This script's first version listed the guarded strings in its own
#      source, so committing the guard would have published the list of
#      things it guards. The gate caught itself. Terms now live in a
#      gitignored file; this script contains none of them.
#
# Exits non-zero on any hit AND on any scanner error, so the caller cannot
# proceed. Scans the STAGED SNAPSHOT, not the textual diff, so a sensitive
# line in an unchanged region is still caught.
#
# Usage:   bash scripts/release_check.sh
#          (then commit, inspect, and push as separate deliberate steps)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

TERMS_FILE=".leakpatterns"
if [ ! -f "$TERMS_FILE" ]; then
    echo ">>> $TERMS_FILE missing. Cannot scan. NOTHING COMMITTED."
    echo "    It is intentionally gitignored; recreate it locally."
    exit 1
fi

PATTERN="$(grep -v -e '^\s*#' -e '^\s*$' "$TERMS_FILE" | paste -sd'|' -)"
if [ -z "$PATTERN" ]; then
    echo ">>> term list is empty. NOTHING COMMITTED."
    exit 1
fi

echo "=== STAGED FILES ==="
if git diff --cached --quiet; then
    echo "nothing staged -- aborting"
    exit 1
fi
git diff --cached --name-status
git diff --cached --check

echo
echo "=== PUBLIC COMMIT IDENTITY ==="
git var GIT_AUTHOR_IDENT
git var GIT_COMMITTER_IDENT
git config --get commit.gpgsign || echo "commit.gpgsign: unset"
git remote -v

echo
echo "=== LOCAL-ONLY FILES MUST BE IGNORED ==="
for f in TEARRL-0-*.md CLAUDE.md "$TERMS_FILE"; do
    [ -e "$f" ] || continue
    if git check-ignore -q "$f"; then
        echo "  ignored : $f"
    else
        echo "  *** TRACKED: $f"
        echo ">>> LOCAL-ONLY FILE IS NOT IGNORED. NOTHING COMMITTED."
        exit 1
    fi
done

echo
echo "=== SCANNER SELF-TEST ==="
# A scanner that always reports "clean" is worse than none. Prove the pattern
# matches known-positives, drawn from the term list itself, before trusting a
# negative result. Also proves case-insensitivity is live -- defect 2 above.
probe_lower="$(head -1 <(grep -v -e '^\s*#' -e '^\s*$' "$TERMS_FILE"))"
probe_upper="$(printf '%s' "$probe_lower" | tr '[:lower:]' '[:upper:]')"
matched=0
printf '%s\n' "$probe_lower" | grep -q -i -E "$PATTERN" && matched=$((matched + 1))
printf '%s\n' "$probe_upper" | grep -q -i -E "$PATTERN" && matched=$((matched + 1))
if [ "$matched" -lt 2 ]; then
    echo ">>> SELF-TEST FAILED (matched $matched/2, case-insensitivity broken)."
    echo ">>> NOTHING COMMITTED."
    exit 1
fi
echo "self-test: matched 2/2 known-positives, case-insensitive"

echo
echo "=== LEAK SCAN (staged snapshot, case-insensitive) ==="
set +e
hits="$(git grep --cached -I -n -i -E "$PATTERN" -- . 2>&1)"
scan_rc=$?
set -e

case "$scan_rc" in
    0)
        printf '%s\n' "$hits"
        echo ">>> LEAK SCAN FAILED. NOTHING COMMITTED."
        exit 1
        ;;
    1)
        echo "leak scan: CLEAN"
        ;;
    *)
        printf '%s\n' "$hits"
        echo ">>> LEAK SCAN ERROR (rc=$scan_rc). NOTHING COMMITTED."
        exit "$scan_rc"
        ;;
esac

echo
echo "=== FULL TEST SUITE ==="
./.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --no-header

echo
echo "=== GATE PASSED ==="
echo "Now commit, inspect with 'git show --stat HEAD', then push as a"
echo "separate deliberate step. Do not chain commit and push."
