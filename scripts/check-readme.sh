#!/usr/bin/env bash
# Verify README.md's "What is verified" table against manifest.json's theorem registry:
# - Every backticked theorem name or glob in each row's "Theorems" column must match at least one
#   theorem in `correct` + `other` of that row's kernel entry in manifest.json.
# - Every `correct` theorem in manifest.json must appear (directly or via glob) in its README row.
# - Public-source rows are tied to their manifest entry by the backticked file path in the "Kernel"
#   column; unreferenced rows must map bijectively to `source: "unreferenced"` manifest entries.
# Does not grep Lean sources; compiled-environment existence and types are audited by CheckAxioms.lean.
set -euo pipefail
export LC_ALL=C

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${1:-${REPO_ROOT}}"

README="${ROOT}/README.md"
MANIFEST="${ROOT}/manifest.json"

if [[ ! -f "${README}" ]]; then
  echo "error: README file not found: ${README}" >&2
  exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
  echo "error: manifest file not found: ${MANIFEST}" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "error: required tool 'jq' not found in PATH" >&2
  exit 1
fi

tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

# Extract "<kernel_cell>\t<theorems_cell>" for each data row of "## What is verified".
awk '
  /^## What is verified[[:space:]]*$/ { in_sec = 1; next }
  in_sec && /^## / { exit }
  in_sec && /^\|/ {
    n = split($0, cols, "|")
    if (!in_table) {
      for (i = 2; i < n; i++) {
        cell = cols[i]
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", cell)
        if (cell == "Kernel") kern_col = i
        if (cell == "Theorems") thm_col = i
      }
      if (kern_col > 0 && thm_col > 0) in_table = 1
      next
    }
    if ($0 ~ /^\|([[:space:]]*:?-+:?[[:space:]]*\|)+[[:space:]]*$/) {
      next
    }
    if (kern_col > 0 && thm_col > 0 && kern_col < n && thm_col < n) {
      printf "%s\t%s\n", cols[kern_col], cols[thm_col]
    }
    next
  }
  in_table { exit }
' "${README}" > "${tmpdir}/rows.tsv"

if [[ ! -s "${tmpdir}/rows.tsv" ]]; then
  echo "error: no rows found in the 'What is verified' table of ${README}" >&2
  exit 1
fi

# Load manifest kernel metadata.
mapfile -t all_keys < <(jq -er '.kernels | keys[]' "${MANIFEST}")
mapfile -t unref_keys < <(jq -er '.kernels | to_entries[] | select((.value.source // "") == "unreferenced") | .key' "${MANIFEST}")

declare -A matched_keys=()
had_error=0
total_entries=0

pattern_matches_list() {
  local pat="$1"
  shift
  local decl
  if [[ "${pat}" == *'*'* ]]; then
    for decl in "$@"; do
      # Unquoted RHS performs glob matching for patterns containing `*`.
      if [[ "${decl}" == ${pat} ]]; then
        return 0
      fi
    done
  else
    for decl in "$@"; do
      if [[ "${decl}" == "${pat}" ]]; then
        return 0
      fi
    done
  fi
  return 1
}

row_num=0
while IFS=$'\t' read -r kern_cell thm_cell; do
  row_num=$((row_num + 1))
  mapfile -t thms < <(printf '%s\n' "${thm_cell}" | { grep -oE '`[^`]+`' || true; } | sed 's/^`//; s/`$//')
  if (( ${#thms[@]} == 0 )); then
    echo "error: README row ${row_num} has no backticked theorems in 'Theorems' column" >&2
    had_error=1
    continue
  fi

  # Check if the Kernel column mentions a backticked source path (contains '/' or ends in '.py').
  row_path=""
  while IFS= read -r tok; do
    [[ -z "${tok}" ]] && continue
    if [[ "${tok}" == */* || "${tok}" == *.py ]]; then
      row_path="${tok}"
      break
    fi
  done < <(printf '%s\n' "${kern_cell}" | { grep -oE '`[^`]+`' || true; } | sed 's/^`//; s/`$//')

  kernel_key=""
  if [[ -n "${row_path}" ]]; then
    mapfile -t path_matches < <(
      jq -r --arg p "${row_path}" \
        '.kernels | to_entries[] | select(.value.path == $p) | .key' "${MANIFEST}"
    )
    if (( ${#path_matches[@]} == 0 )); then
      echo "error: README row ${row_num} path '${row_path}' has no matching entry in ${MANIFEST}" >&2
      had_error=1
      continue
    elif (( ${#path_matches[@]} > 1 )); then
      echo "error: README row ${row_num} path '${row_path}' matches multiple manifest entries: ${path_matches[*]}" >&2
      had_error=1
      continue
    fi
    kernel_key="${path_matches[0]}"
  else
    # Unreferenced kernel row: match against an unmatched `source: "unreferenced"` entry whose
    # registry contains every theorem/glob listed in this row.
    candidates=()
    for uk in "${unref_keys[@]}"; do
      [[ -n "${matched_keys[${uk}]+x}" ]] && continue
      mapfile -t reg < <(jq -r --arg k "${uk}" '.kernels[$k] | ((.correct // []) + (.other // []))[]' "${MANIFEST}")
      all_ok=1
      for t in "${thms[@]}"; do
        if ! pattern_matches_list "${t}" "${reg[@]}"; then
          all_ok=0
          break
        fi
      done
      if (( all_ok == 1 )); then
        candidates+=("${uk}")
      fi
    done
    if (( ${#candidates[@]} == 0 )); then
      echo "error: README row ${row_num} theorems (${thms[*]}) do not match any unreferenced manifest entry" >&2
      had_error=1
      continue
    elif (( ${#candidates[@]} > 1 )); then
      echo "error: README row ${row_num} ambiguously matches multiple unreferenced manifest entries: ${candidates[*]}" >&2
      had_error=1
      continue
    fi
    kernel_key="${candidates[0]}"
  fi

  if [[ -n "${matched_keys[${kernel_key}]+x}" ]]; then
    echo "error: manifest kernel '${kernel_key}' appears in more than one README row" >&2
    had_error=1
    continue
  fi
  matched_keys["${kernel_key}"]=1

  mapfile -t reg_correct < <(jq -r --arg k "${kernel_key}" '.kernels[$k].correct[]? // empty' "${MANIFEST}")
  mapfile -t reg_all < <(jq -r --arg k "${kernel_key}" '.kernels[$k] | ((.correct // []) + (.other // []))[]' "${MANIFEST}")

  # Every backticked theorem/glob in this row must be in this kernel's manifest registry.
  for t in "${thms[@]}"; do
    total_entries=$((total_entries + 1))
    if ! pattern_matches_list "${t}" "${reg_all[@]}"; then
      echo "error: README theorem '${t}' is not in manifest registry for '${kernel_key}'" >&2
      had_error=1
    fi
  done

  # Every `correct` theorem in the manifest registry for this kernel must appear in this row.
  for c in "${reg_correct[@]}"; do
    [[ -z "${c}" ]] && continue
    covered=0
    for t in "${thms[@]}"; do
      if [[ "${t}" == *'*'* ]]; then
        if [[ "${c}" == ${t} ]]; then
          covered=1
          break
        fi
      else
        if [[ "${c}" == "${t}" ]]; then
          covered=1
          break
        fi
      fi
    done
    if (( covered == 0 )); then
      echo "error: manifest 'correct' theorem '${c}' for '${kernel_key}' is missing from README" >&2
      had_error=1
    fi
  done
done < "${tmpdir}/rows.tsv"

# Ensure every manifest kernel with `correct` theorems (and every kernel overall) has a README row.
for k in "${all_keys[@]}"; do
  if [[ -z "${matched_keys[${k}]+x}" ]]; then
    mapfile -t reg_correct < <(jq -r --arg k "${k}" '.kernels[$k].correct[]? // empty' "${MANIFEST}")
    if (( ${#reg_correct[@]} > 0 )); then
      echo "error: manifest kernel '${k}' has 'correct' theorem(s) (${reg_correct[*]}) missing from README" >&2
      had_error=1
    else
      echo "error: manifest kernel '${k}' has no row in README 'What is verified' table" >&2
      had_error=1
    fi
  fi
done

if (( had_error != 0 )); then
  exit 1
fi

echo "OK: ${total_entries} README theorem entries verified against manifest.json (${#all_keys[@]} kernels)."
