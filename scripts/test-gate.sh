#!/usr/bin/env bash
set -euo pipefail

tmp=""
cleanup() {
  if [[ -n "$tmp" && -d "$tmp" ]]; then
    rm -rf "$tmp"
  fi
}
trap cleanup EXIT

compile_fixture() {
  local dir="$1" out_dir="$2"
  lake env lean -R "$dir" -o "$out_dir/GateFixture.olean" "$dir/GateFixture.lean"
}

run_gate() {
  local dir="$1" out_dir="$2"
  lake env lean --run scripts/CheckAxioms.lean --scan-only --parse-root Init --base "$dir" --root GateFixture &&
  lake env sh -c 'LEAN_PATH="$1:$LEAN_PATH" exec lean --run scripts/CheckAxioms.lean --check-env --base "$2" --root GateFixture' \
    _ "$out_dir" "$dir"
}

# 1. Clean control fixture MUST pass.
tmp="$(mktemp -d)"
compile_fixture "scripts/gate-fixtures/clean" "$tmp"
run_gate "scripts/gate-fixtures/clean" "$tmp"
rm -rf "$tmp"
tmp=""

# 2. Negative fixtures MUST compile cleanly and MUST be rejected with the expected reason.
cases=(
  "bad-axiom|disallowed axiom"
  "bad-private-outside-ns|disallowed axiom"
  "bad-unsafe|unsafe"
  "bad-example-native|forbidden \`example\`"
  "bad-example-sorry|forbidden \`example\`"
  "bad-lexer-evasion|GateFixture.lean:4: forbidden token \`sorry\`"
  "bad-skip-kernel-tc|skipKernelTC"
  "bad-unimported|not imported"
  "bad-opaque|\`opaque\` declaration is forbidden"
  "bad-implemented-by|implemented_by"
)

for entry in "${cases[@]}"; do
  name="${entry%%|*}"
  expect="${entry#*|}"
  dir="scripts/gate-fixtures/$name"
  tmp="$(mktemp -d)"
  compile_fixture "$dir" "$tmp"
  set +e
  out="$(run_gate "$dir" "$tmp" 2>&1)"
  rc=$?
  set -e
  rm -rf "$tmp"
  tmp=""
  if [[ $rc -eq 0 ]]; then
    echo "FAIL: fixture $name unexpectedly PASSED the gate" >&2
    echo "$out" >&2
    exit 1
  fi
  if ! grep -Fq "$expect" <<<"$out"; then
    echo "FAIL: fixture $name failed for wrong reason (expected: $expect)" >&2
    echo "$out" >&2
    exit 1
  fi
  echo "PASS (rejected as expected): $name"
done
