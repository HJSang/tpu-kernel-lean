#!/usr/bin/env bash
set -euo pipefail

# 1. Independent kernel re-check of every module under TpuKernelLean/ before the axiom gate.
mapfile -t tkl_modules < <(
  find TpuKernelLean -name '*.lean' -type f | LC_ALL=C sort | sed 's/\.lean$//; s|/|.|g'
)
if [[ -f TpuKernelLean.lean ]]; then
  tkl_modules=("TpuKernelLean" "${tkl_modules[@]}")
fi
lake env leanchecker "${tkl_modules[@]}"

# 2. Two-phase axiom/syntax/statement gate (source scan BEFORE loading library with initializers).
lake env lean --run scripts/CheckAxioms.lean --scan-only
lake env lean --run scripts/CheckAxioms.lean --check-env
bash scripts/test-gate.sh

# 3. Ledger, README registry, manifest checks, and fixture suites.
bash scripts/check-ledger.sh
bash scripts/check-readme.sh
bash scripts/check-manifest.sh --offline
bash scripts/test-checks.sh
bash scripts/test-attacks.sh
