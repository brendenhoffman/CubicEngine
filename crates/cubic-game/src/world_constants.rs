// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

//! World-scale constants and latitude helpers. Must match
//! `cubic-world::world_constants` exactly -- duplicated here rather than
//! taken as a dependency because `cubic-world` pulls in `cubic-render`
//! (egui, raw-window-handle, ...), which isn't buildable for this crate's
//! `wasm32-unknown-unknown` guest target. See the CHUNK_SIZE/VOXEL_SIZE constants
//! above for the same pattern already in use in this crate.

/// Distance from equator to pole in meters. 1/4 Earth scale.
pub const EQUATOR_TO_POLE_M: f64 = 2_500_000.0;
/// Full N/S tile in meters (south pole to north pole).
pub const NS_TILE_M: f64 = 5_000_000.0;
/// Sea level in world Y meters. Terrain below this is ocean floor.
pub const SEA_LEVEL_M: f64 = 0.0;
/// Maximum terrain height above sea level in meters.
pub const MAX_TERRAIN_HEIGHT_M: f64 = 12_000.0;
/// Minimum ocean floor depth below sea level in meters.
pub const MIN_OCEAN_FLOOR_M: f64 = -2_500.0;
/// Moisture grid resolution in meters per cell.
pub const MOISTURE_GRID_CELL_M: f64 = 1_000.0;
/// Temperature lapse rate in degrees C per meter of elevation.
pub const TEMP_LAPSE_RATE_C_PER_M: f64 = -0.0065;

/// Convert world Z coordinate to normalized latitude [-1.0, 1.0].
/// 0.0 = equator, ±1.0 = poles.
#[inline]
pub fn world_z_to_latitude(z: f64) -> f64 {
    (z / EQUATOR_TO_POLE_M).clamp(-1.0, 1.0)
}

/// Convert normalized latitude to baseline temperature in Celsius at sea
/// level. Quadratic falloff from equator (+30°C) to poles (-35°C).
#[inline]
pub fn latitude_to_base_temp_c(lat: f64) -> f64 {
    let abs_lat = lat.abs();
    30.0 - 65.0 * abs_lat * abs_lat
}
