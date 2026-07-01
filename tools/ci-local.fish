#!/usr/bin/env fish
# ci-local.fish — run every check the GitHub CI pipeline runs, locally.
# Usage: fish tools/ci-local.fish [--full]
#
# Default: host-target clippy only (fast).
# --full  runs the complete cross-target clippy matrix, same as CI.

set -l fast 1
for arg in $argv
    if test "$arg" = "--full"
        set fast 0
    end
end

set -l root (git rev-parse --show-toplevel)
cd $root

set -l failed

function section
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $argv"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
end

# ── 1. rustfmt ────────────────────────────────────────────────────────────────
section "1/4  rustfmt"
if cargo fmt --all -- --check
    echo "✅  fmt"
else
    echo "❌  fmt"
    set -a failed "fmt"
end

# ── 2. clippy ─────────────────────────────────────────────────────────────────
section "2/4  clippy"
if test $fast -eq 1
    echo "(host target only — run with --full for the complete CI matrix)"
    if cargo clippy --workspace --all-targets -- -D warnings
        echo "✅  clippy (host)"
    else
        echo "❌  clippy (host)"
        set -a failed "clippy"
    end
else
    echo "(full cross-target matrix)"
    if bash tools/runclippy.sh
        echo "✅  clippy (all targets)"
    else
        echo "❌  clippy (all targets)"
        set -a failed "clippy"
    end
end

# ── 3. cargo-deny ─────────────────────────────────────────────────────────────
section "3/4  cargo-deny"
if command -q cargo-deny
    if cargo deny check
        echo "✅  cargo-deny"
    else
        echo "❌  cargo-deny"
        set -a failed "cargo-deny"
    end
else
    echo "⚠️   cargo-deny not installed — skipping"
    echo "    install with: cargo install cargo-deny"
end

# ── 4. header check ───────────────────────────────────────────────────────────
section "4/4  check-headers"
if bash tools/check-headers.sh
    echo "✅  check-headers"
else
    echo "❌  check-headers"
    set -a failed "check-headers"
end

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════════════════"
if test (count $failed) -eq 0
    echo "  ✅  All checks passed"
    exit 0
else
    echo "  ❌  Failed: $failed"
    exit 1
end
