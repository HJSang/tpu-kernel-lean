#!/usr/bin/env bash
set -euo pipefail

lake env lean --run scripts/CheckAxioms.lean
bash scripts/test-gate.sh

bash scripts/check-ledger.sh
bash scripts/check-readme.sh
bash scripts/check-manifest.sh --offline
bash scripts/test-checks.sh
