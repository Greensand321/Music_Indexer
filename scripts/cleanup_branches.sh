#!/usr/bin/env bash
# Delete stale remote branches on Greensand321/Music_Indexer.
# git-only - no gh CLI or extra auth needed, just your existing git credentials.
#
# Keeps: main, whatever branch you have checked out, and any branch starting
# with "claude/". Deletes everything else, regardless of open PR status.
#
# Usage:
#   ./scripts/cleanup_branches.sh            # dry run - just prints what WOULD be deleted
#   ./scripts/cleanup_branches.sh --execute  # actually deletes (after typing DELETE to confirm)

set -euo pipefail

BATCH_SIZE=30

EXECUTE=false
if [[ "${1:-}" == "--execute" ]]; then
  EXECUTE=true
fi

echo "Fetching branches ..."
git fetch origin --prune --quiet

DEFAULT_BRANCH=$(git remote show origin | awk '/HEAD branch/ {print $NF}')
CURRENT_BRANCH=$(git branch --show-current || true)

echo "Default branch: $DEFAULT_BRANCH"
echo "Current branch: ${CURRENT_BRANCH:-<detached>}"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

git for-each-ref --format='%(refname:short)' refs/remotes/origin \
  | sed 's#^origin/##' \
  | grep -v '^HEAD$' \
  | sort -u > "$TMPDIR/all_branches.txt"

grep -vE "^(${DEFAULT_BRANCH}|${CURRENT_BRANCH})\$|^claude/" "$TMPDIR/all_branches.txt" > "$TMPDIR/to_delete.txt" || true

TOTAL=$(wc -l < "$TMPDIR/all_branches.txt")
DELETE=$(wc -l < "$TMPDIR/to_delete.txt")
KEEP=$((TOTAL - DELETE))

echo ""
echo "Total branches:  $TOTAL"
echo "Keeping:         $KEEP"
echo "To delete:       $DELETE"
echo ""

if [[ "$DELETE" -eq 0 ]]; then
  echo "Nothing to delete."
  exit 0
fi

echo "Branches slated for deletion (first 30 shown):"
head -30 "$TMPDIR/to_delete.txt"
[[ "$DELETE" -gt 30 ]] && echo "... and $((DELETE - 30)) more"
echo ""

if [[ "$EXECUTE" != true ]]; then
  echo "Dry run only. Re-run with --execute to actually delete these branches."
  exit 0
fi

read -r -p "Type DELETE to permanently delete these $DELETE branches: " CONFIRM
if [[ "$CONFIRM" != "DELETE" ]]; then
  echo "Aborted."
  exit 1
fi

split -l "$BATCH_SIZE" "$TMPDIR/to_delete.txt" "$TMPDIR/batch_"

deleted=0
for f in "$TMPDIR"/batch_*; do
  mapfile -t branches < "$f"
  refspecs=()
  for b in "${branches[@]}"; do
    refspecs+=(":$b")
  done
  if git push origin "${refspecs[@]}"; then
    deleted=$((deleted + ${#branches[@]}))
  else
    echo "WARNING: batch failed, some branches in this batch may not be deleted: ${branches[*]}" >&2
  fi
done

echo ""
echo "Done. Deleted approximately $deleted branches."
