#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${REPO_ROOT}/manifest.json"
VENV="${REPO_ROOT}/.e2e-venv"

for cmd in python3 curl jq git; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "ERROR: required tool '${cmd}' not found in PATH" >&2
    exit 1
  fi
done

if [[ ! -f "${MANIFEST}" ]]; then
  echo "ERROR: manifest file not found: ${MANIFEST}" >&2
  exit 1
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  python3 -m venv "${VENV}"
fi
"${VENV}/bin/pip" install --quiet --disable-pip-version-check -r "${REPO_ROOT}/e2e/requirements.txt"

TMPDIR_KERNELS="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_KERNELS}"' EXIT

entries="$(jq -er '
  .kernels | to_entries[] |
  select(.value.level == "pallas_call" and (.value.source // "") != "unreferenced") |
  [.key, .value.repository, .value.commit, .value.path, .value.blob_sha] | join("\u001f")
' "${MANIFEST}")"

while IFS=$'\x1f' read -r kernel repo commit path blob_sha; do
  [[ -z "${kernel}" ]] && continue
  dest="${TMPDIR_KERNELS}/${path}"
  mkdir -p "$(dirname "${dest}")"
  url="https://raw.githubusercontent.com/${repo}/${commit}/${path}"
  curl -sSfL "${url}" -o "${dest}"
  actual_sha="$(git hash-object "${dest}")"
  if [[ "${actual_sha}" != "${blob_sha}" ]]; then
    echo "ERROR: ${kernel} (${path}@${commit}): git blob SHA ${actual_sha} != manifest ${blob_sha}" >&2
    exit 1
  fi
done <<< "${entries}"

# Fetch helper module imported at runtime by true_on_policy/{norm,matmul}_kernel.py
# at the exact commit pinned for them in manifest.json.
top_repo="$(jq -er '.kernels.true_on_policy_norm.repository' "${MANIFEST}")"
top_commit="$(jq -er '.kernels.true_on_policy_norm.commit' "${MANIFEST}")"
mm_repo="$(jq -er '.kernels.true_on_policy_matmul.repository' "${MANIFEST}")"
mm_commit="$(jq -er '.kernels.true_on_policy_matmul.commit' "${MANIFEST}")"
if [[ "${top_repo}" != "${mm_repo}" || "${top_commit}" != "${mm_commit}" ]]; then
  echo "ERROR: true_on_policy_norm and true_on_policy_matmul are pinned to different commits" >&2
  exit 1
fi
helper_path="sglang_tpu/rl/true_on_policy/common.py"
helper_dest="${TMPDIR_KERNELS}/${helper_path}"
mkdir -p "$(dirname "${helper_dest}")"
curl -sSfL "https://raw.githubusercontent.com/${top_repo}/${top_commit}/${helper_path}" -o "${helper_dest}"

export JAX_PLATFORMS=cpu
export TKL_E2E_KERNELS_DIR="${TMPDIR_KERNELS}"
"${VENV}/bin/pytest" "${REPO_ROOT}/e2e" -q
