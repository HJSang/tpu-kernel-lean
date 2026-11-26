#!/usr/bin/env bash
# Negative fixture tests proving that the hardened verification pipeline rejects
# the red-team attacks addressed by this task (A4, A5, A7, A8, A9, A10, A11, A12b),
# while the unmodified library passes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIX_DIR="${REPO_ROOT}/scripts/attack-fixtures"

tmp=""
cleanup() {
  if [[ -n "$tmp" && -d "$tmp" ]]; then
    rm -rf "$tmp"
  fi
}
trap cleanup EXIT

make_repo_copy() {
  local dest="$1"
  mkdir -p "${dest}/docs"
  cp "${REPO_ROOT}/docs/pallas-model-ledger.md" "${dest}/docs/pallas-model-ledger.md"
  cp "${REPO_ROOT}/README.md" "${dest}/README.md"
  cp "${REPO_ROOT}/manifest.json" "${dest}/manifest.json"
  cp -R "${REPO_ROOT}/TpuKernelLean" "${dest}/TpuKernelLean"
  cp -R "${REPO_ROOT}/e2e" "${dest}/e2e"
}

# Compile a single attack fixture module against the already-built TpuKernelLean library
# (without rebuilding Mathlib or any real TpuKernelLean module) and build an AttackRoot
# wrapper importing TpuKernelLean plus the fixture module.
compile_attack_mod() {
  local dest="$1" src_file="$2" mod_name="$3"
  local build_lib="${REPO_ROOT}/.lake/build/lib/lean"
  local mod_rel
  mod_rel="$(printf '%s' "${mod_name}" | tr '.' '/')"
  # Overlay prebuilt TpuKernelLean artifacts via symlinks so Lean's single-root SearchPath
  # resolves both the real library and the newly compiled fixture under one TpuKernelLean root.
  mkdir -p "${dest}/TpuKernelLean"
  (
    cd "${build_lib}"
    for f in TpuKernelLean.*; do
      [[ -e "${f}" ]] && ln -s "${build_lib}/${f}" "${dest}/${f}"
    done
    find TpuKernelLean -type d -exec mkdir -p "${dest}/{}" \;
    find TpuKernelLean -type f -exec ln -s "${build_lib}/{}" "${dest}/{}" \;
  )
  mkdir -p "${dest}/$(dirname "${mod_rel}")"
  cp "${src_file}" "${dest}/${mod_rel}.lean"
  lake env sh -c 'LEAN_PATH="$1:$LEAN_PATH" exec lean -R "$1" -o "$1/$2.olean" "$1/$2.lean"' \
    _ "${dest}" "${mod_rel}"
  cat > "${dest}/AttackRoot.lean" <<EOF
import TpuKernelLean
import ${mod_name}
EOF
  lake env sh -c 'LEAN_PATH="$1:$LEAN_PATH" exec lean -R "$1" -o "$1/AttackRoot.olean" "$1/AttackRoot.lean"' \
    _ "${dest}"
}

run_check_env() {
  local dest="$1" manifest="$2"
  lake env sh -c 'LEAN_PATH="$1:$LEAN_PATH" exec lean --run "$2/scripts/CheckAxioms.lean" --check-env --base "$1" --root AttackRoot --audit-root TpuKernelLean --manifest "$3"' \
    _ "${dest}" "${REPO_ROOT}" "${manifest}"
}

run_scan_only() {
  local dest="$1" root="$2"
  lake env lean --run "${REPO_ROOT}/scripts/CheckAxioms.lean" --scan-only --base "${dest}" --root "${root}"
}

assert_rejects() {
  local name="$1"
  shift
  local -a expects=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    expects+=("$1")
    shift
  done
  shift # consume "--"
  set +e
  local out
  out="$("$@" 2>&1)"
  local rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    echo "FAIL: attack ${name} unexpectedly PASSED" >&2
    echo "${out}" >&2
    exit 1
  fi
  local exp
  for exp in "${expects[@]}"; do
    if ! grep -Fq "${exp}" <<<"${out}"; then
      echo "FAIL: attack ${name} failed for wrong reason (expected: ${exp})" >&2
      echo "${out}" >&2
      exit 1
    fi
  done
  echo "PASS (rejected as expected): ${name}"
}

# 0. Unmodified library MUST pass.
bash "${REPO_ROOT}/scripts/check-readme.sh" "${REPO_ROOT}" >/dev/null
bash "${REPO_ROOT}/scripts/check-manifest.sh" --offline "${REPO_ROOT}" >/dev/null
echo "PASS (unmodified library): check-readme.sh and check-manifest.sh"

# 1. A4 — Non-Pallas `run` (headline theorem does not reach TpuKernelLean.Pallas.pallasCall).
tmp="$(mktemp -d)"
compile_attack_mod "${tmp}" "${FIX_DIR}/a4-non-pallas-run/AttackA4.lean" "TpuKernelLean.Examples.AttackA4"
jq '.kernels.attack_a4 = {
  lean_module: "TpuKernelLean.Examples.AttackA4",
  level: "pallas_call",
  correct: ["pallasRelu_correct"],
  other: [],
  witness: "TpuKernelLean.Tests.EndToEnd.kvPageWrite_endToEnd"
}' "${REPO_ROOT}/manifest.json" > "${tmp}/manifest.json"
assert_rejects "A4-non-pallas-run" \
  "does not reach \`TpuKernelLean.Pallas.pallasCall\`" \
  -- run_check_env "${tmp}" "${tmp}/manifest.json"
rm -rf "${tmp}"; tmp=""

# 2. A5 — Namespace shadowing of `Pallas.pallasCall`.
tmp="$(mktemp -d)"
compile_attack_mod "${tmp}" "${FIX_DIR}/a5-shadowing/AttackA5.lean" "TpuKernelLean.Examples.AttackA5"
jq '.kernels.attack_a5 = {
  lean_module: "TpuKernelLean.Examples.AttackA5",
  level: "pallas_call",
  correct: ["vectorCopy_correct"],
  other: [],
  witness: "TpuKernelLean.Tests.EndToEnd.kvPageWrite_endToEnd"
}' "${REPO_ROOT}/manifest.json" > "${tmp}/manifest.json"
assert_rejects "A5-shadowing" \
  "protected name component \`pallasCall\`" \
  "does not reach \`TpuKernelLean.Pallas.pallasCall\`" \
  -- run_check_env "${tmp}" "${tmp}/manifest.json"
rm -rf "${tmp}"; tmp=""

# 3. A7 — Undefined behaviour dodged via truncating `ℕ` subtraction in `BlockSpec.indexMap`.
tmp="$(mktemp -d)"
compile_attack_mod "${tmp}" "${FIX_DIR}/a7-ub-dodged/AttackA7.lean" "TpuKernelLean.Examples.AttackA7"
assert_rejects "A7-blockspec-nat-sub" \
  "indexMap uses truncating natural-number subtraction" \
  -- run_check_env "${tmp}" "${REPO_ROOT}/manifest.json"
rm -rf "${tmp}"; tmp=""

# 3b. A7 — Truncating `ℕ` subtraction in `ReductionKernel.outIndexMap`.
tmp="$(mktemp -d)"
compile_attack_mod "${tmp}" "${FIX_DIR}/a7-ub-dodged/AttackA7Reduction.lean" "TpuKernelLean.Examples.AttackA7Reduction"
assert_rejects "A7b-reduction-outIndexMap-nat-sub" \
  "indexMap uses truncating natural-number subtraction" \
  -- run_check_env "${tmp}" "${REPO_ROOT}/manifest.json"
rm -rf "${tmp}"; tmp=""

# 4. A8 — Provenance spoofing: duplicate (repository, path) pin, missing e2e test, mismatched witness.
tmp="$(mktemp -d)"
make_repo_copy "${tmp}"
cp "${FIX_DIR}/a8-provenance-spoof/AttackA8.lean" "${tmp}/TpuKernelLean/Examples/AttackA8.lean"
jq '.kernels.attack_a8 = (.kernels.kv_page_write + {
  lean_module: "TpuKernelLean.Examples.AttackA8",
  correct: ["provenanceSpoof_correct"],
  other: [],
  witness: "TpuKernelLean.Tests.EndToEnd.kvPageWrite_endToEnd"
})' "${REPO_ROOT}/manifest.json" > "${tmp}/manifest.json"
assert_rejects "A8-duplicate-repo-path-and-missing-e2e" \
  "duplicate (repository, path) pin" \
  "missing required e2e test file e2e/test_attack_a8.py" \
  -- bash "${REPO_ROOT}/scripts/check-manifest.sh" --offline "${tmp}"
rm -rf "${tmp}"; tmp=""

tmp="$(mktemp -d)"
compile_attack_mod "${tmp}" "${FIX_DIR}/a8-provenance-spoof/AttackA8.lean" "TpuKernelLean.Examples.AttackA8"
jq '.kernels.attack_a8 = {
  lean_module: "TpuKernelLean.Examples.AttackA8",
  level: "pallas_call",
  correct: ["provenanceSpoof_correct"],
  other: [],
  witness: "TpuKernelLean.Tests.EndToEnd.kvPageWrite_endToEnd"
}' "${REPO_ROOT}/manifest.json" > "${tmp}/manifest.json"
assert_rejects "A8-mismatched-witness" \
  "does not share any" \
  -- run_check_env "${tmp}" "${tmp}/manifest.json"
rm -rf "${tmp}"; tmp=""

# 5. A9 — Proof-checker bypass (`import Lean.*` + `elab` + `debug.skipKernelTC` + `addDecl`).
tmp="$(mktemp -d)"
mkdir -p "${tmp}/TpuKernelLean/Examples"
cp "${FIX_DIR}/a9-proof-checker-bypass/AttackA9.lean" "${tmp}/TpuKernelLean/Examples/AttackA9.lean"
assert_rejects "A9-source-scan-import-and-elab" \
  "forbidden import \`Lean.Elab.Command\`" \
  "forbidden \`elab\` command" \
  -- run_scan_only "${tmp}" "TpuKernelLean"
lake env lean -R "${tmp}" -o "${tmp}/TpuKernelLean/Examples/AttackA9.olean" \
  "${tmp}/TpuKernelLean/Examples/AttackA9.lean"
set +e
lc_out="$(lake env sh -c 'LEAN_PATH="$1:$LEAN_PATH" exec leanchecker TpuKernelLean.Examples.AttackA9' _ "${tmp}" 2>&1)"
lc_rc=$?
set -e
if [[ $lc_rc -eq 0 ]]; then
  echo "FAIL: attack A9-leanchecker unexpectedly PASSED leanchecker" >&2
  echo "${lc_out}" >&2
  exit 1
fi
echo "PASS (rejected as expected): A9-leanchecker"
rm -rf "${tmp}"; tmp=""

# 5b. A9-initialize — `initialize` must be rejected BEFORE any library module is loaded.
tmp="$(mktemp -d)"
mkdir -p "${tmp}/TpuKernelLean/Examples"
cp "${FIX_DIR}/a9-initialize/AttackA9Init.lean" "${tmp}/TpuKernelLean/Examples/AttackA9Init.lean"
lake env lean -R "${tmp}" -o "${tmp}/TpuKernelLean/Examples/AttackA9Init.olean" \
  "${tmp}/TpuKernelLean/Examples/AttackA9Init.lean"
cat > "${tmp}/TpuKernelLean.lean" <<'EOF'
import TpuKernelLean.Examples.AttackA9Init
EOF
lake env sh -c 'LEAN_PATH="$1:$LEAN_PATH" exec lean -R "$1" -o "$1/TpuKernelLean.olean" "$1/TpuKernelLean.lean"' \
  _ "${tmp}"
set +e
init_out="$(lake env sh -c 'LEAN_PATH="$1:$LEAN_PATH" exec lean --run "$2/scripts/CheckAxioms.lean" --base "$1" --root TpuKernelLean --parse-root Init' \
  _ "${tmp}" "${REPO_ROOT}" 2>&1)"
init_rc=$?
set -e
if [[ $init_rc -eq 0 ]]; then
  echo "FAIL: attack A9-initialize unexpectedly PASSED" >&2
  echo "${init_out}" >&2
  exit 1
fi
if grep -Fq "INITIALIZER_EXECUTED" <<<"${init_out}"; then
  echo "FAIL: attack A9-initialize executed its initializer before being rejected!" >&2
  echo "${init_out}" >&2
  exit 1
fi
if ! grep -Fq "forbidden \`initialize\`" <<<"${init_out}"; then
  echo "FAIL: attack A9-initialize failed for wrong reason" >&2
  echo "${init_out}" >&2
  exit 1
fi
echo "PASS (rejected before loading): A9-initialize"
rm -rf "${tmp}"; tmp=""

# 6. A10 — Check-script gaps (comment phantom, `: True`, cross-file README, subdir glob, unreferenced).
# 6a. Comment-only phantom in README and in manifest.
tmp="$(mktemp -d)"
make_repo_copy "${tmp}"
sed 's/`writePages_correct`/`writePages_correct`, `phantomKernel_correct`/' "${tmp}/README.md" > "${tmp}/README.md.tmp"
mv "${tmp}/README.md.tmp" "${tmp}/README.md"
assert_rejects "A10a-readme-phantom" \
  "phantomKernel_correct" \
  -- bash "${REPO_ROOT}/scripts/check-readme.sh" "${tmp}"
rm -rf "${tmp}"; tmp=""

tmp="$(mktemp -d)"
compile_attack_mod "${tmp}" "${FIX_DIR}/a10-check-script-gaps/AttackA10.lean" "TpuKernelLean.Examples.AttackA10"
jq '.kernels.attack_a10_phantom = {
  lean_module: "TpuKernelLean.Examples.AttackA10",
  level: "algorithm_only",
  correct: [],
  other: ["phantomKernel_correct"],
  witness: null
}' "${REPO_ROOT}/manifest.json" > "${tmp}/manifest.json"
assert_rejects "A10a-env-phantom" \
  "not found in compiled environment" \
  -- run_check_env "${tmp}" "${tmp}/manifest.json"
rm -rf "${tmp}"; tmp=""

# 6b. Vacuous `: True` theorem in manifest.
tmp="$(mktemp -d)"
compile_attack_mod "${tmp}" "${FIX_DIR}/a10-check-script-gaps/AttackA10.lean" "TpuKernelLean.Examples.AttackA10"
jq '.kernels.attack_a10_trivial = {
  lean_module: "TpuKernelLean.Examples.AttackA10",
  level: "algorithm_only",
  correct: [],
  other: ["trivialClaim_correct"],
  witness: null
}' "${REPO_ROOT}/manifest.json" > "${tmp}/manifest.json"
assert_rejects "A10b-trivial-true" \
  "trivially \`True\` statement" \
  -- run_check_env "${tmp}" "${tmp}/manifest.json"
rm -rf "${tmp}"; tmp=""

# 6c. Cross-file README hijack.
tmp="$(mktemp -d)"
make_repo_copy "${tmp}"
awk '
  /^\| `sglang_tpu\/kv_page_write\.py`/ {
    print
    print "| `sglang_tpu/fake_flash_decode.py` | `writePages_correct` | Fake claim |"
    next
  }
  { print }
' "${tmp}/README.md" > "${tmp}/README.md.tmp"
mv "${tmp}/README.md.tmp" "${tmp}/README.md"
assert_rejects "A10c-cross-file-readme" \
  "sglang_tpu/fake_flash_decode.py" \
  -- bash "${REPO_ROOT}/scripts/check-readme.sh" "${tmp}"
rm -rf "${tmp}"; tmp=""

# 6d. Subdirectory file under TpuKernelLean/Examples/ missing from manifest.
tmp="$(mktemp -d)"
make_repo_copy "${tmp}"
mkdir -p "${tmp}/TpuKernelLean/Examples/SubDir"
cp "${FIX_DIR}/a10-check-script-gaps/AttackA10.lean" "${tmp}/TpuKernelLean/Examples/SubDir/AttackA10.lean"
assert_rejects "A10d-subdir-missing-from-manifest" \
  "TpuKernelLean.Examples.SubDir.AttackA10" \
  -- bash "${REPO_ROOT}/scripts/check-manifest.sh" --offline "${tmp}"
rm -rf "${tmp}"; tmp=""

# 6e. Unauthorized `"source": "unreferenced"` entry.
tmp="$(mktemp -d)"
make_repo_copy "${tmp}"
cp "${FIX_DIR}/a10-check-script-gaps/AttackA10.lean" "${tmp}/TpuKernelLean/Examples/AttackA10.lean"
jq '.kernels.attack_a10_unref = {
  lean_module: "TpuKernelLean.Examples.AttackA10",
  level: "algorithm_only",
  correct: [],
  other: ["trivialClaim_correct"],
  witness: null,
  source: "unreferenced"
}' "${REPO_ROOT}/manifest.json" > "${tmp}/manifest.json"
assert_rejects "A10e-unauthorized-unreferenced" \
  "not in the allowlist" \
  -- bash "${REPO_ROOT}/scripts/check-manifest.sh" --offline "${tmp}"
rm -rf "${tmp}"; tmp=""

# 7. A11 — `local notation` hijacking `=`.
tmp="$(mktemp -d)"
mkdir -p "${tmp}/TpuKernelLean/Examples"
cp "${FIX_DIR}/a11-local-notation/AttackA11.lean" "${tmp}/TpuKernelLean/Examples/AttackA11.lean"
assert_rejects "A11-local-notation" \
  "forbidden \`notation\` command" \
  -- run_scan_only "${tmp}" "TpuKernelLean"
rm -rf "${tmp}"; tmp=""

# 8. A12b — Unicode homoglyph `.Cοrrect` (Greek omicron) in `TpuKernelLean.KernelSpec`.
tmp="$(mktemp -d)"
mkdir -p "${tmp}/TpuKernelLean/Examples"
cp "${FIX_DIR}/a12b-homoglyph/AttackA12b.lean" "${tmp}/TpuKernelLean/Examples/AttackA12b.lean"
assert_rejects "A12b-source-scan-non-ascii" \
  "non-ASCII character in declaration name" \
  -- run_scan_only "${tmp}" "TpuKernelLean"
rm -rf "${tmp}"; tmp=""

tmp="$(mktemp -d)"
compile_attack_mod "${tmp}" "${FIX_DIR}/a12b-homoglyph/AttackA12b.lean" "TpuKernelLean.Examples.AttackA12b"
jq '.kernels.attack_a12b = {
  lean_module: "TpuKernelLean.Examples.AttackA12b",
  level: "pallas_call",
  correct: ["homoglyphRelu_correct"],
  other: [],
  witness: "TpuKernelLean.Tests.EndToEnd.kvPageWrite_endToEnd"
}' "${REPO_ROOT}/manifest.json" > "${tmp}/manifest.json"
assert_rejects "A12b-env-protected-ns-and-non-ascii" \
  "non-ASCII character in declaration name" \
  "protected namespace \`TpuKernelLean.KernelSpec\`" \
  "trivially \`True\` statement" \
  -- run_check_env "${tmp}" "${tmp}/manifest.json"
rm -rf "${tmp}"; tmp=""

echo "OK: all attack fixtures rejected as expected."
