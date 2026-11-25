#!/usr/bin/env bash
# Verify every theorem name (or glob pattern) listed in backticks in the "Theorems"
# column of README.md's "What is verified" table is declared under TpuKernelLean/.
set -euo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:-${REPO_ROOT}}"

README="${ROOT}/README.md"
LEAN_DIR="${ROOT}/TpuKernelLean"

if [[ ! -f "${README}" ]]; then
  echo "error: README file not found: ${README}" >&2
  exit 1
fi
if [[ ! -d "${LEAN_DIR}" ]]; then
  echo "error: Lean source directory not found: ${LEAN_DIR}" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

# Extract backticked identifiers from the "Theorems" column of the table under "## What is verified".
awk '
  /^## What is verified[[:space:]]*$/ { in_sec = 1; next }
  in_sec && /^## / { exit }
  in_sec && /^\|/ {
    n = split($0, cols, "|")
    if (!in_table) {
      for (i = 2; i < n; i++) {
        cell = cols[i]
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", cell)
        if (cell == "Theorems") {
          thm_col = i
          in_table = 1
          break
        }
      }
      next
    }
    if ($0 ~ /^\|([[:space:]]*:?-+:?[[:space:]]*\|)+[[:space:]]*$/) {
      next
    }
    if (thm_col > 0 && thm_col < n) {
      print cols[thm_col]
    }
    next
  }
  in_table { exit }
' "${README}" | { grep -oE '`[^`]+`' || true; } | sed 's/^`//; s/`$//' > "${tmpdir}/readme_theorems.txt"

if [[ ! -s "${tmpdir}/readme_theorems.txt" ]]; then
  echo "error: no theorem entries found in the 'What is verified' table of ${README}" >&2
  exit 1
fi

# Collect every theorem name declared at the start of a declaration under TpuKernelLean/.
{ grep -rhE '^(@\[[^]]*\][[:space:]]+)*((private|protected)[[:space:]]+)?theorem[[:space:]]+[A-Za-z_][A-Za-z0-9_.'\'']*' \
    "${LEAN_DIR}" --include='*.lean' || true; } \
  | sed -E \
      -e 's/^(@\[[^]]*\][[:space:]]+)*//' \
      -e 's/^(private|protected)[[:space:]]+//' \
      -e 's/^theorem[[:space:]]+([A-Za-z_][A-Za-z0-9_.'\'']*).*/\1/' \
  | sort -u > "${tmpdir}/declared_theorems.txt"

mapfile -t declared < "${tmpdir}/declared_theorems.txt"

missing=()
total=0
while IFS= read -r entry; do
  [[ -z "${entry}" ]] && continue
  total=$((total + 1))
  matched=0
  if (( ${#declared[@]} > 0 )); then
    if [[ "${entry}" == *'*'* ]]; then
      for decl in "${declared[@]}"; do
        # Unquoted RHS performs glob matching for patterns containing `*`.
        if [[ "${decl}" == ${entry} || "${decl}" == *.${entry} ]]; then
          matched=1
          break
        fi
      done
    else
      for decl in "${declared[@]}"; do
        if [[ "${decl}" == "${entry}" || "${decl}" == *."${entry}" ]]; then
          matched=1
          break
        fi
      done
    fi
  fi
  if (( matched == 0 )); then
    missing+=("${entry}")
  fi
done < "${tmpdir}/readme_theorems.txt"

if (( ${#missing[@]} > 0 )); then
  echo "error: README theorem(s) not declared under ${LEAN_DIR}:" >&2
  for m in "${missing[@]}"; do
    echo "  ${m}" >&2
  done
  exit 1
fi

echo "OK: ${total} README theorem entries verified under TpuKernelLean/."
