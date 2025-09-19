#!/usr/bin/env bash
set -euo pipefail

TARGETS=(
  aarch64-apple-darwin
  aarch64-pc-windows-msvc
  aarch64-unknown-linux-gnu
  armv7-unknown-linux-gnueabihf
  i686-unknown-linux-gnu
  powerpc64le-unknown-linux-gnu
  riscv64gc-unknown-linux-gnu
  s390x-unknown-linux-gnu
  x86_64-apple-darwin
  x86_64-pc-windows-msvc
  x86_64-unknown-linux-gnu
)

CARGO_ARGS=(--workspace --all-targets)
FAILED=()

for t in "${TARGETS[@]}"; do
  echo "===== cargo clippy --target $t ====="
  if ! cargo clippy "${CARGO_ARGS[@]}" --target "$t" -- -D warnings; then
    FAILED+=("$t")
  fi
done

if ((${#FAILED[@]})); then
  echo "clippy failed for targets: ${FAILED[*]}" >&2
  exit 1
fi

echo "clippy passed for all targets."
