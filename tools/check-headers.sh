#!/usr/bin/env bash
set -euo pipefail

# Find all tracked Rust files
files=$(git ls-files '*.rs' || true)

missing=()

# Check top 10 lines for BOTH:
# 1) // SPDX-License-Identifier:
# 2) #![deny(unsafe_op_in_unsafe_fn)]
while IFS= read -r f; do
  [ -z "$f" ] && continue
  head10=$(head -n 10 "$f" || true)

  if ! printf '%s\n' "$head10" | grep -qE '^// SPDX-License-Identifier:'; then
    missing+=("$f")
    continue
  fi

  if ! printf '%s\n' "$head10" | grep -qE '^#!\[deny\(unsafe_op_in_unsafe_fn\)\]'; then
    missing+=("$f")
    continue
  fi
done <<<"$files"

if [ ${#missing[@]} -ne 0 ]; then
  echo "❌ Missing SPDX or unsafe lint in these files (must appear within first 10 lines):"
  printf '%s\n' "${missing[@]}"
  exit 1
fi

echo "✅ All Rust files have SPDX + unsafe lint within the first 10 lines."
