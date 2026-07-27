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
for f in TEARRL-0-*.md CLAUDE.md GOAL-*.md .claude .codex .agents "$TERMS_FILE"; do
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
echo "=== LEAK SCAN (raw staged blobs, including bytecode) ==="
./.venv/Scripts/python.exe scripts/scan_staged_snapshot.py --terms "$TERMS_FILE"
echo
echo "=== FULL TEST SUITE ==="
./.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider --no-header

echo
echo "=== GATE PASSED ==="
echo "Now commit, inspect with 'git show --stat HEAD', then push as a"
echo "separate deliberate step. Do not chain commit and push."
