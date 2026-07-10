// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! The currently loaded WASM game plugin and the world generator(s) wrapping
//! it — grouped together since load_world() always sets all three at once
//! (there's no way to change the seed on an existing WasmPlugin, so a
//! relaunch rebuilds the plugin and both generator handles fresh).

use cubic_wasm::{WasmPlugin, WasmWorldGenerator};
use cubic_world::WorldGenerator;
use std::sync::Arc;

/// Empty (all `None`) until `App::load_world()` constructs a fresh plugin
/// for the current launch's seed.
#[derive(Default)]
pub(crate) struct GuestPlugin {
    pub(crate) plugin: Option<Arc<WasmPlugin>>,
    // Same generator as `wasm_game`, but type-erased to `dyn WorldGenerator`
    // for the streaming pipeline, which doesn't need to call `tick`.
    pub(crate) generator: Option<Arc<dyn WorldGenerator>>,
    // Same generator as `generator`, but concretely typed so on_tick can be
    // called — `Arc<dyn WorldGenerator>` doesn't expose `tick`.
    pub(crate) wasm_game: Option<Arc<WasmWorldGenerator>>,
}
