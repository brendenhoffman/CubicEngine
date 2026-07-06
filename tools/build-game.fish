#!/usr/bin/env fish
# Build the cubic-game WASM plugin and copy it to games/

set -l root (git rev-parse --show-toplevel)

cargo build \
    --manifest-path $root/crates/cubic-game/Cargo.toml \
    --target wasm32-unknown-unknown \
    --release

set -l wasm $root/target/wasm32-unknown-unknown/release/cubic_game.wasm

mkdir -p $root/games/cubic-game
cp $wasm $root/games/cubic-game/game.wasm
echo "Built: games/cubic-game/game.wasm"
