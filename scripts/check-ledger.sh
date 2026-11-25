#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:-${REPO_ROOT}}"

LEDGER="${ROOT}/docs/pallas-model-ledger.md"
LEAN_DIR="${ROOT}/TpuKernelLean"

if [[ ! -f "$LEDGER" ]]; then
  echo "error: ledger file not found: $LEDGER" >&2
  exit 1
fi
if [[ ! -d "$LEAN_DIR" ]]; then
  echo "error: Lean source directory not found: $LEAN_DIR" >&2
  exit 1
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

# Extract every `### PAL-...` entry heading from the ledger.
{ grep -oE '^### PAL-[A-Z0-9-]+$' "$LEDGER" || true; } | sed 's/^### //' | sort > "$tmpdir/ledger_entries_raw.txt"
sort -u "$tmpdir/ledger_entries_raw.txt" > "$tmpdir/ledger_entries.txt"

if ! cmp -s "$tmpdir/ledger_entries_raw.txt" "$tmpdir/ledger_entries.txt"; then
  echo "error: duplicate PAL- entry headings in $LEDGER:" >&2
  uniq -d "$tmpdir/ledger_entries_raw.txt" >&2
  exit 1
fi

# Extract every PAL- ID referenced anywhere in the ledger.
{ grep -oE 'PAL-[A-Z0-9-]+' "$LEDGER" || true; } | sort -u > "$tmpdir/ledger_refs.txt"

# Extract every PAL- ID mentioned under TpuKernelLean/.
{ grep -rohE 'PAL-[A-Z0-9-]+' "$LEAN_DIR" || true; } | sort -u > "$tmpdir/lean_ids.txt"

status=0

missing_in_ledger=$(comm -23 "$tmpdir/lean_ids.txt" "$tmpdir/ledger_entries.txt")
if [[ -n "$missing_in_ledger" ]]; then
  echo "error: PAL- IDs mentioned under TpuKernelLean/ but missing a '### <ID>' entry in $LEDGER:" >&2
  echo "$missing_in_ledger" >&2
  status=1
fi

missing_in_lean=$(comm -13 "$tmpdir/lean_ids.txt" "$tmpdir/ledger_entries.txt")
if [[ -n "$missing_in_lean" ]]; then
  echo "error: PAL- entries in $LEDGER not mentioned under TpuKernelLean/:" >&2
  echo "$missing_in_lean" >&2
  status=1
fi

dangling_in_ledger=$(comm -23 "$tmpdir/ledger_refs.txt" "$tmpdir/ledger_entries.txt")
if [[ -n "$dangling_in_ledger" ]]; then
  echo "error: PAL- IDs referenced in $LEDGER without a '### <ID>' entry:" >&2
  echo "$dangling_in_ledger" >&2
  status=1
fi

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

count=$(wc -l < "$tmpdir/ledger_entries.txt" | tr -d ' ')
echo "OK: $count PAL- ledger entries match TpuKernelLean/."
