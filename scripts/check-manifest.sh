#!/usr/bin/env bash
# Verify manifest.json offline structure/files and (without --offline) against GitHub Contents API:
# - Checks JSON shape, required fields (including `correct`, `other`, `witness`), 40-hex commit/blob
#   SHAs, level values, module existence, recursive coverage of TpuKernelLean/Examples/**/*.lean,
#   uniqueness of (repository, path) pins, e2e/test_<key>.py presence for public pallas_call entries,
#   an explicit allowlist for `source: "unreferenced"`, and that unreferenced entries carry no
#   repository, path, or commit.
# - Online mode also checks each pinned commit's git blob SHA equals manifest blob_sha (exit 1)
#   and each tracked ref's git blob SHA equals manifest blob_sha (difference -> STALE, exit 2).
set -euo pipefail

# Explicit allowlist of manifest keys permitted to use `"source": "unreferenced"`.
ALLOWED_UNREFERENCED_KEYS=(
  "block_scaled_mm"
  "grouped_scaled_mm"
)

is_allowed_unreferenced() {
  local key="$1" allowed
  for allowed in "${ALLOWED_UNREFERENCED_KEYS[@]}"; do
    if [[ "${allowed}" == "${key}" ]]; then
      return 0
    fi
  done
  return 1
}

OFFLINE=0
TARGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --offline)
      OFFLINE=1
      shift
      ;;
    -*)
      echo "ERROR: unknown option: $1" >&2
      exit 1
      ;;
    *)
      if [[ -n "${TARGET}" ]]; then
        echo "ERROR: unexpected extra argument: $1" >&2
        exit 1
      fi
      TARGET="$1"
      shift
      ;;
  esac
done

SCRIPT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${TARGET}" ]]; then
  REPO_ROOT="${SCRIPT_REPO_ROOT}"
  MANIFEST="${REPO_ROOT}/manifest.json"
elif [[ -d "${TARGET}" ]]; then
  REPO_ROOT="$(cd "${TARGET}" && pwd)"
  MANIFEST="${REPO_ROOT}/manifest.json"
else
  MANIFEST="${TARGET}"
  REPO_ROOT="$(cd "$(dirname "${TARGET}")" && pwd)"
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: required tool 'jq' not found in PATH" >&2
  exit 1
fi
if (( OFFLINE == 0 )) && ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: required tool 'curl' not found in PATH" >&2
  exit 1
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest file not found: ${MANIFEST}" >&2
  exit 1
fi

TOKEN="${GITHUB_TOKEN:-}"

fetch_blob_sha() {
  local repo="$1"
  local path="$2"
  local ref="$3"
  local url="https://api.github.com/repos/${repo}/contents/${path}"
  local resp http_code body sha msg

  local -a headers=(
    -H "Accept: application/vnd.github+json"
    -H "X-GitHub-Api-Version: 2022-11-28"
    -H "User-Agent: tpu-kernel-lean-check-manifest"
  )
  if [[ -n "${TOKEN}" ]]; then
    headers+=(-H "Authorization: Bearer ${TOKEN}")
  fi

  resp="$(curl -sS -G "${headers[@]}" --data-urlencode "ref=${ref}" -w "\n%{http_code}" "${url}")" || {
    echo "ERROR: curl failed for ${repo}/${path}@${ref}" >&2
    return 1
  }
  http_code="${resp##*$'\n'}"
  body="${resp%$'\n'*}"

  if [[ "${http_code}" != "200" ]]; then
    msg="$(printf '%s' "${body}" | jq -r 'if type == "object" then (.message // empty) else empty end' 2>/dev/null || true)"
    echo "ERROR: GitHub API returned HTTP ${http_code} for ${repo}/${path}@${ref}${msg:+: ${msg}}" >&2
    return 1
  fi

  sha="$(printf '%s' "${body}" | jq -r 'if type == "object" and .type == "file" then (.sha // empty) else empty end' 2>/dev/null || true)"
  if [[ -z "${sha}" ]]; then
    echo "ERROR: GitHub API response missing file .sha for ${repo}/${path}@${ref}" >&2
    return 1
  fi

  printf '%s\n' "${sha}"
}

entries="$(jq -er '
  if (type != "object") then
    error("manifest.json: top-level value must be an object")
  elif (.schema_version | type != "number") then
    error("manifest.json: schema_version must be a number")
  elif (.kernels | type != "object" or length == 0) then
    error("manifest.json: kernels must be a non-empty object")
  elif ([.kernels[] | type == "object"] | all | not) then
    error("manifest.json: every kernel entry must be an object")
  else
    .kernels | to_entries[] |
    [
      .key,
      (if (.value.lean_module | type == "string") then .value.lean_module else "" end),
      (if (.value.level | type == "string") then .value.level else "" end),
      (if (.value.repository | type == "string") then .value.repository else "" end),
      (if (.value.ref | type == "string") then .value.ref else "" end),
      (if (.value.commit | type == "string") then .value.commit else "" end),
      (if (.value.path | type == "string") then .value.path else "" end),
      (if (.value.blob_sha | type == "string") then .value.blob_sha else "" end),
      (if (.value.source | type == "string") then .value.source else "" end),
      (if (.value | has("repository")) then "1" else "0" end),
      (if (.value | has("path")) then "1" else "0" end),
      (if (.value | has("commit")) then "1" else "0" end),
      (if (.value.correct | type == "array" and ([.[] | type == "string"] | all)) then (.value.correct | length | tostring) else "BAD" end),
      (if (.value.other | type == "array" and ([.[] | type == "string"] | all)) then (.value.other | length | tostring) else "BAD" end),
      (if (.value | has("witness") | not) then "MISSING"
       elif (.value.witness == null) then "NULL"
       elif (.value.witness | type == "string" and length > 0) then "STR"
       else "BAD" end)
    ] | join("\u001f")
  end
' "${MANIFEST}")" || {
  echo "ERROR: failed to parse ${MANIFEST}" >&2
  exit 1
}

had_error=0
had_stale=0

# 1. Verify every .lean file under TpuKernelLean/Examples/ (recursively) has a manifest entry.
if [[ ! -d "${REPO_ROOT}/TpuKernelLean/Examples" ]]; then
  echo "ERROR: examples directory not found: ${REPO_ROOT}/TpuKernelLean/Examples" >&2
  had_error=1
else
  while IFS= read -r lean_file; do
    [[ -f "${lean_file}" ]] || continue
    rel_file="${lean_file#"${REPO_ROOT}/"}"
    expected_mod="$(printf '%s' "${rel_file%.lean}" | tr '/' '.')"
    if ! jq -e --arg mod "${expected_mod}" '[.kernels[].lean_module] | index($mod) != null' "${MANIFEST}" >/dev/null; then
      echo "ERROR: ${expected_mod} (${rel_file}) has no entry in ${MANIFEST}" >&2
      had_error=1
    fi
  done < <(find "${REPO_ROOT}/TpuKernelLean/Examples" -type f -name '*.lean' | LC_ALL=C sort)
fi

# 2. Reject duplicate (repository, path) pins across distinct entries.
dup_repo_paths="$(jq -r '
  [
    .kernels | to_entries[] |
    select((.value.source // "") != "unreferenced" and (.value.repository | type == "string") and (.value.path | type == "string")) |
    { key: .key, rp: "\(.value.repository)::\(.value.path)" }
  ]
  | group_by(.rp)
  | map(select(length > 1))
  | .[] | "\(.[0].rp) pinned by: \(map(.key) | join(", "))"
' "${MANIFEST}")"
if [[ -n "${dup_repo_paths}" ]]; then
  while IFS= read -r dup_line; do
    [[ -z "${dup_line}" ]] && continue
    echo "ERROR: duplicate (repository, path) pin: ${dup_line}" >&2
    had_error=1
  done <<< "${dup_repo_paths}"
fi

while IFS=$'\x1f' read -r kernel lean_module level repo ref commit path blob_sha source has_repo has_path has_commit correct_len other_len witness_kind; do
  [[ -z "${kernel}" ]] && continue

  if [[ -z "${lean_module}" ]]; then
    echo "ERROR: ${kernel}: missing required field (lean_module)" >&2
    had_error=1
    continue
  fi
  if [[ "${level}" != "pallas_call" && "${level}" != "algorithm_only" ]]; then
    echo "ERROR: ${kernel}: invalid level '${level}' (expected pallas_call or algorithm_only)" >&2
    had_error=1
    continue
  fi
  if [[ "${correct_len}" == "BAD" ]]; then
    echo "ERROR: ${kernel}: 'correct' must be an array of theorem names" >&2
    had_error=1
    continue
  fi
  if [[ "${other_len}" == "BAD" ]]; then
    echo "ERROR: ${kernel}: 'other' must be an array of theorem names" >&2
    had_error=1
    continue
  fi
  if [[ "${witness_kind}" == "MISSING" || "${witness_kind}" == "BAD" ]]; then
    echo "ERROR: ${kernel}: 'witness' must be a non-empty theorem name string or null" >&2
    had_error=1
    continue
  fi
  if [[ "${level}" == "pallas_call" ]]; then
    if [[ "${correct_len}" == "0" ]]; then
      echo "ERROR: ${kernel}: pallas_call entry must declare at least one 'correct' theorem" >&2
      had_error=1
      continue
    fi
    if [[ "${witness_kind}" != "STR" ]]; then
      echo "ERROR: ${kernel}: pallas_call entry must declare a non-null 'witness' theorem" >&2
      had_error=1
      continue
    fi
  fi

  mod_rel="$(printf '%s' "${lean_module}" | tr '.' '/').lean"
  if [[ ! -f "${REPO_ROOT}/${mod_rel}" ]]; then
    echo "ERROR: ${kernel}: lean_module file not found: ${mod_rel}" >&2
    had_error=1
    continue
  fi

  # An entry whose kernel source is not public must be explicitly allowlisted and omit repository/path/commit.
  if [[ "${source}" == "unreferenced" ]]; then
    if ! is_allowed_unreferenced "${kernel}"; then
      echo "ERROR: ${kernel}: source 'unreferenced' is not in the allowlist (${ALLOWED_UNREFERENCED_KEYS[*]})" >&2
      had_error=1
      continue
    fi
    if [[ "${has_repo}" != "0" || "${has_path}" != "0" || "${has_commit}" != "0" ]]; then
      echo "ERROR: ${kernel}: unreferenced entry must not carry repository, path, or commit" >&2
      had_error=1
      continue
    fi
    echo "SKIP: ${kernel} (source not referenced)"
    continue
  fi

  if [[ -n "${source}" ]]; then
    echo "ERROR: ${kernel}: invalid source '${source}'" >&2
    had_error=1
    continue
  fi
  if [[ -z "${repo}" || -z "${ref}" || -z "${path}" ]]; then
    echo "ERROR: ${kernel}: missing required field (repository/ref/path)" >&2
    had_error=1
    continue
  fi
  if [[ ! "${commit}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: ${kernel}: commit must be a 40-char lowercase hex SHA, got '${commit}'" >&2
    had_error=1
    continue
  fi
  if [[ ! "${blob_sha}" =~ ^[0-9a-f]{40}$ ]]; then
    echo "ERROR: ${kernel}: blob_sha must be a 40-char lowercase hex SHA, got '${blob_sha}'" >&2
    had_error=1
    continue
  fi

  if [[ "${level}" == "pallas_call" ]]; then
    e2e_test="${REPO_ROOT}/e2e/test_${kernel}.py"
    if [[ ! -f "${e2e_test}" ]]; then
      echo "ERROR: ${kernel}: missing required e2e test file e2e/test_${kernel}.py" >&2
      had_error=1
      continue
    fi
  fi

  if (( OFFLINE != 0 )); then
    echo "OK (offline): ${kernel} ${path}"
    continue
  fi

  if ! commit_sha="$(fetch_blob_sha "${repo}" "${path}" "${commit}")"; then
    had_error=1
    continue
  fi
  if [[ "${commit_sha}" != "${blob_sha}" ]]; then
    echo "ERROR: ${kernel} (${path}@${commit}): manifest blob_sha ${blob_sha} != GitHub ${commit_sha}" >&2
    had_error=1
    continue
  fi

  if ! ref_sha="$(fetch_blob_sha "${repo}" "${path}" "${ref}")"; then
    had_error=1
    continue
  fi
  if [[ "${ref_sha}" != "${blob_sha}" ]]; then
    echo "STALE: ${kernel} ${path}"
    had_stale=1
  else
    echo "OK: ${kernel} ${path}"
  fi
done <<< "${entries}"

if (( had_error != 0 )); then
  exit 1
fi
if (( had_stale != 0 )); then
  exit 2
fi
exit 0
