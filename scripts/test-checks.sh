#!/usr/bin/env bash
# Positive and negative fixture tests for check-ledger.sh, check-readme.sh, and check-manifest.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

tmp=""
cleanup() {
  if [[ -n "$tmp" && -d "$tmp" ]]; then
    rm -rf "$tmp"
  fi
}
trap cleanup EXIT

make_copy() {
  local dest="$1"
  mkdir -p "${dest}/docs"
  cp "${REPO_ROOT}/docs/pallas-model-ledger.md" "${dest}/docs/pallas-model-ledger.md"
  cp "${REPO_ROOT}/README.md" "${dest}/README.md"
  cp "${REPO_ROOT}/manifest.json" "${dest}/manifest.json"
  cp -R "${REPO_ROOT}/TpuKernelLean" "${dest}/TpuKernelLean"
}

assert_rejects() {
  local name="$1" expect="$2"
  shift 2
  set +e
  local out
  out="$("$@" 2>&1)"
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "FAIL: case ${name} unexpectedly PASSED" >&2
    echo "${out}" >&2
    exit 1
  fi
  if ! grep -Fq "${expect}" <<<"${out}"; then
    echo "FAIL: case ${name} failed for wrong reason (expected: ${expect})" >&2
    echo "${out}" >&2
    exit 1
  fi
  echo "PASS (rejected as expected): ${name}"
}

# 1. Unmodified copy MUST pass all three checks.
tmp="$(mktemp -d)"
make_copy "${tmp}"
bash "${REPO_ROOT}/scripts/check-ledger.sh" "${tmp}" >/dev/null
bash "${REPO_ROOT}/scripts/check-readme.sh" "${tmp}" >/dev/null
bash "${REPO_ROOT}/scripts/check-manifest.sh" --offline "${tmp}" >/dev/null
rm -rf "${tmp}"
tmp=""
echo "PASS (unmodified copy): all three checks"

# 2. Ledger with one PAL- entry heading removed.
tmp="$(mktemp -d)"
make_copy "${tmp}"
sed '/^### PAL-GRID-ORDER$/d' "${tmp}/docs/pallas-model-ledger.md" > "${tmp}/docs/pallas-model-ledger.md.tmp"
mv "${tmp}/docs/pallas-model-ledger.md.tmp" "${tmp}/docs/pallas-model-ledger.md"
assert_rejects "ledger-missing-entry" "PAL-GRID-ORDER" \
  bash "${REPO_ROOT}/scripts/check-ledger.sh" "${tmp}"
rm -rf "${tmp}"
tmp=""

# 3. README with a non-existent theorem name.
tmp="$(mktemp -d)"
make_copy "${tmp}"
sed 's/`writePages_correct`/`writePages_correct`, `nonexistent_theorem_xyz`/' "${tmp}/README.md" > "${tmp}/README.md.tmp"
mv "${tmp}/README.md.tmp" "${tmp}/README.md"
assert_rejects "readme-missing-theorem" "nonexistent_theorem_xyz" \
  bash "${REPO_ROOT}/scripts/check-readme.sh" "${tmp}"
rm -rf "${tmp}"
tmp=""

# 4. README with a non-matching glob pattern.
tmp="$(mktemp -d)"
make_copy "${tmp}"
sed 's/`\*_elem_independent`/`*_nonexistent_glob_xyz`/' "${tmp}/README.md" > "${tmp}/README.md.tmp"
mv "${tmp}/README.md.tmp" "${tmp}/README.md"
assert_rejects "readme-missing-glob" "*_nonexistent_glob_xyz" \
  bash "${REPO_ROOT}/scripts/check-readme.sh" "${tmp}"
rm -rf "${tmp}"
tmp=""

# 5. Manifest with one example's entry removed.
tmp="$(mktemp -d)"
make_copy "${tmp}"
jq 'del(.kernels.kv_page_write)' "${tmp}/manifest.json" > "${tmp}/manifest.json.tmp"
mv "${tmp}/manifest.json.tmp" "${tmp}/manifest.json"
assert_rejects "manifest-missing-example" "TpuKernelLean.Examples.KvPageWrite" \
  bash "${REPO_ROOT}/scripts/check-manifest.sh" --offline "${tmp}"
rm -rf "${tmp}"
tmp=""

# 6. Manifest with an unreferenced entry carrying a repository field.
tmp="$(mktemp -d)"
make_copy "${tmp}"
jq '.kernels.block_scaled_mm.repository = "example/repo"' "${tmp}/manifest.json" > "${tmp}/manifest.json.tmp"
mv "${tmp}/manifest.json.tmp" "${tmp}/manifest.json"
assert_rejects "manifest-unreferenced-with-repo" "unreferenced entry must not carry repository" \
  bash "${REPO_ROOT}/scripts/check-manifest.sh" --offline "${tmp}"
rm -rf "${tmp}"
tmp=""

# 7. Manifest with a malformed blob SHA.
tmp="$(mktemp -d)"
make_copy "${tmp}"
jq '.kernels.kv_page_write.blob_sha = "not-a-sha"' "${tmp}/manifest.json" > "${tmp}/manifest.json.tmp"
mv "${tmp}/manifest.json.tmp" "${tmp}/manifest.json"
assert_rejects "manifest-bad-sha" "40-char lowercase hex SHA" \
  bash "${REPO_ROOT}/scripts/check-manifest.sh" --offline "${tmp}"
rm -rf "${tmp}"
tmp=""
