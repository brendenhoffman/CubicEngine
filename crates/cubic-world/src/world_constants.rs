// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

//! World-scale constants and latitude helpers shared by the host
//! (`cubic-world`, `cubic-app`) and guest (`cubic-game`) worldgen code, so
//! both sides agree on world size, sea level, and climate scale without
//! duplicating magic numbers.

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
/// level. Quadratic falloff from equator (+30°C) to poles (-35°C) --
/// coefficients solved so both endpoints land exactly on those targets.
#[inline]
pub fn latitude_to_base_temp_c(lat: f64) -> f64 {
    let abs_lat = lat.abs();
    30.0 - 65.0 * abs_lat * abs_lat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temp_matches_equator_and_pole_targets() {
        assert!((latitude_to_base_temp_c(0.0) - 30.0).abs() < 1e-9);
        assert!((latitude_to_base_temp_c(1.0) - (-35.0)).abs() < 1e-9);
        assert!((latitude_to_base_temp_c(-1.0) - (-35.0)).abs() < 1e-9);
    }

    #[test]
    fn temp_symmetric_across_hemispheres() {
        assert_eq!(latitude_to_base_temp_c(0.4), latitude_to_base_temp_c(-0.4));
    }

    #[test]
    fn latitude_clamped_at_poles() {
        assert_eq!(world_z_to_latitude(EQUATOR_TO_POLE_M * 10.0), 1.0);
        assert_eq!(world_z_to_latitude(-EQUATOR_TO_POLE_M * 10.0), -1.0);
    }

    #[test]
    fn latitude_zero_at_equator() {
        assert_eq!(world_z_to_latitude(0.0), 0.0);
    }
}
