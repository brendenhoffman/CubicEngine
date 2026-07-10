#!/usr/bin/env bash
set -euo pipefail

# aarch64-apple-darwin and x86_64-apple-darwin are deliberately excluded:
# cross-compiling to macOS from Linux needs Apple's SDK, which can't be
# fetched by a package manager, so there's no way to make these pass locally.
# CI checks them on an actual macos-latest runner instead (see ci.yml).
TARGETS=(
  aarch64-pc-windows-msvc
  aarch64-unknown-linux-gnu
  armv7-unknown-linux-gnueabihf
  i686-unknown-linux-gnu
  powerpc64-unknown-linux-gnu
  riscv64gc-unknown-linux-gnu
  x86_64-pc-windows-msvc
  x86_64-unknown-linux-gnu
)

CARGO_ARGS=(--workspace --all-targets)
FAILED=()

# gilrs's udev feature (libudev-sys) probes for libudev via pkg-config at
# build-script time, which refuses to run at all when cross-compiling unless
# this is set — it's a safety guard, not an actual missing-library error.
# clippy only type-checks and never links, so it's fine that the target's
# libudev isn't actually present; PKG_CONFIG_ALLOW_CROSS just lets the
# build script proceed instead of aborting outright.
export PKG_CONFIG_ALLOW_CROSS=1

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
