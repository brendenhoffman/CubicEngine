#!/usr/bin/env bash
set -euo pipefail
GLSLC=${GLSLC:-glslc} # or glslangValidator -V (adjust flags if you switch)

SRC_DIR="assets/shaders"
OUT_DIR="assets/shaders"
TARGET_ENV="--target-env=vulkan1.2"

$GLSLC "$SRC_DIR/tri.vert" -o "$OUT_DIR/tri.vert.spv" $TARGET_ENV -O
$GLSLC "$SRC_DIR/tri.frag" -o "$OUT_DIR/tri.frag.spv" $TARGET_ENV -O
echo "Shaders built to $OUT_DIR"
