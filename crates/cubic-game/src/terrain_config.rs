// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

//! Live-tunable worldgen parameters, loaded from the datapack's
//! `world/terrain.toml` at `on_load` time. Every field falls back to the
//! value that used to be a hardcoded constant if the file, a whole
//! section, or an individual key is missing -- so an empty or absent file
//! reproduces the exact terrain those constants used to produce, and a
//! partial override file only changes the keys it specifies.
//!
//! Deliberately excluded (see the card): world seed (`world.toml`), world
//! scale constants like `EQUATOR_TO_POLE_M` (compile-time in `cubic-world`
//! -- changing them would break saves), block ids (registered dynamically),
//! and player physics (`cubic.toml`).

use serde::Deserialize;

/// Generates `fn $name() -> $ty { $val }` for each entry -- used as
/// `#[serde(default = "...")]` targets so missing individual TOML keys
/// fall back to the pre-config hardcoded value for that key, not `T::default()`.
macro_rules! default_fns {
    ($($name:ident: $ty:ty = $val:expr;)+) => {
        $(fn $name() -> $ty { $val })+
    };
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(default)]
pub struct WorldCfg {
    #[serde(default = "default_plate_count")]
    pub plate_count: usize,
    #[serde(default = "default_continental_fraction")]
    pub continental_fraction: f64,
}

default_fns! {
    default_plate_count: usize = 28;
    default_continental_fraction: f64 = 0.40;
}

impl Default for WorldCfg {
    fn default() -> Self {
        Self {
            plate_count: default_plate_count(),
            continental_fraction: default_continental_fraction(),
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(default)]
pub struct UpliftCfg {
    #[serde(default = "default_continental_base_m")]
    pub continental_base_m: f32,
    #[serde(default = "default_continental_density_scale")]
    pub continental_density_scale: f32,
    #[serde(default = "default_oceanic_base_m")]
    pub oceanic_base_m: f32,
    #[serde(default = "default_oceanic_density_scale")]
    pub oceanic_density_scale: f32,
    #[serde(default = "default_boundary_falloff_m")]
    pub boundary_falloff_m: f64,
    #[serde(default = "default_max_convergent_uplift_m")]
    pub max_convergent_uplift_m: f32,
    #[serde(default = "default_max_divergent_subsidence_m")]
    pub max_divergent_subsidence_m: f32,
}

default_fns! {
    default_continental_base_m: f32 = 500.0;
    default_continental_density_scale: f32 = 4000.0;
    default_oceanic_base_m: f32 = -500.0;
    default_oceanic_density_scale: f32 = 3000.0;
    default_boundary_falloff_m: f64 = 200_000.0;
    default_max_convergent_uplift_m: f32 = 12_000.0;
    default_max_divergent_subsidence_m: f32 = 3_000.0;
}

impl Default for UpliftCfg {
    fn default() -> Self {
        Self {
            continental_base_m: default_continental_base_m(),
            continental_density_scale: default_continental_density_scale(),
            oceanic_base_m: default_oceanic_base_m(),
            oceanic_density_scale: default_oceanic_density_scale(),
            boundary_falloff_m: default_boundary_falloff_m(),
            max_convergent_uplift_m: default_max_convergent_uplift_m(),
            max_divergent_subsidence_m: default_max_divergent_subsidence_m(),
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(default)]
pub struct NoiseCfg {
    #[serde(default = "default_regional_frequency")]
    pub regional_frequency: f64,
    #[serde(default = "default_regional_amplitude")]
    pub regional_amplitude: f32,
    #[serde(default = "default_regional_octaves")]
    pub regional_octaves: u32,

    #[serde(default = "default_local_frequency")]
    pub local_frequency: f64,
    #[serde(default = "default_local_amplitude_ocean")]
    pub local_amplitude_ocean: f32,
    #[serde(default = "default_local_amplitude_plain")]
    pub local_amplitude_plain: f32,
    #[serde(default = "default_local_amplitude_mountain")]
    pub local_amplitude_mountain: f32,
    #[serde(default = "default_local_octaves")]
    pub local_octaves: u32,

    #[serde(default = "default_ridge_frequency")]
    pub ridge_frequency: f64,
    #[serde(default = "default_ridge_amplitude_factor")]
    pub ridge_amplitude_factor: f32,
    #[serde(default = "default_ridge_octaves")]
    pub ridge_octaves: u32,
    #[serde(default = "default_ridge_boundary_type_threshold")]
    pub ridge_boundary_type_threshold: f32,
    #[serde(default = "default_ridge_uplift_threshold_m")]
    pub ridge_uplift_threshold_m: f32,

    #[serde(default = "default_coastal_frequency")]
    pub coastal_frequency: f64,
    #[serde(default = "default_coastal_amplitude")]
    pub coastal_amplitude: f32,
    #[serde(default = "default_coastal_blend_zone_m")]
    pub coastal_blend_zone_m: f32,
    #[serde(default = "default_coastal_octaves")]
    pub coastal_octaves: u32,
}

default_fns! {
    default_regional_frequency: f64 = 2e-6;
    default_regional_amplitude: f32 = 500.0;
    default_regional_octaves: u32 = 4;

    default_local_frequency: f64 = 5e-5;
    default_local_amplitude_ocean: f32 = 200.0;
    default_local_amplitude_plain: f32 = 300.0;
    default_local_amplitude_mountain: f32 = 500.0;
    default_local_octaves: u32 = 6;

    default_ridge_frequency: f64 = 5e-4;
    default_ridge_amplitude_factor: f32 = 0.3;
    default_ridge_octaves: u32 = 8;
    default_ridge_boundary_type_threshold: f32 = 0.3;
    default_ridge_uplift_threshold_m: f32 = 2000.0;

    default_coastal_frequency: f64 = 1e-4;
    default_coastal_amplitude: f32 = 200.0;
    default_coastal_blend_zone_m: f32 = 300.0;
    default_coastal_octaves: u32 = 3;
}

impl Default for NoiseCfg {
    fn default() -> Self {
        Self {
            regional_frequency: default_regional_frequency(),
            regional_amplitude: default_regional_amplitude(),
            regional_octaves: default_regional_octaves(),
            local_frequency: default_local_frequency(),
            local_amplitude_ocean: default_local_amplitude_ocean(),
            local_amplitude_plain: default_local_amplitude_plain(),
            local_amplitude_mountain: default_local_amplitude_mountain(),
            local_octaves: default_local_octaves(),
            ridge_frequency: default_ridge_frequency(),
            ridge_amplitude_factor: default_ridge_amplitude_factor(),
            ridge_octaves: default_ridge_octaves(),
            ridge_boundary_type_threshold: default_ridge_boundary_type_threshold(),
            ridge_uplift_threshold_m: default_ridge_uplift_threshold_m(),
            coastal_frequency: default_coastal_frequency(),
            coastal_amplitude: default_coastal_amplitude(),
            coastal_blend_zone_m: default_coastal_blend_zone_m(),
            coastal_octaves: default_coastal_octaves(),
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(default)]
pub struct ClimateCfg {
    #[serde(default = "default_moisture_iterations")]
    pub moisture_iterations: usize,
    #[serde(default = "default_moisture_tile_cells")]
    pub moisture_tile_cells: usize,
    #[serde(default = "default_orographic_drop_per_1000m")]
    pub orographic_drop_per_1000m: f32,
    #[serde(default = "default_ocean_evaporation_rate")]
    pub ocean_evaporation_rate: f32,
    #[serde(default = "default_ocean_base_moisture")]
    pub ocean_base_moisture: f32,
    #[serde(default = "default_land_base_moisture")]
    pub land_base_moisture: f32,
}

default_fns! {
    default_moisture_iterations: usize = 40;
    default_moisture_tile_cells: usize = 300;
    default_orographic_drop_per_1000m: f32 = 15.0;
    default_ocean_evaporation_rate: f32 = 0.30;
    default_ocean_base_moisture: f32 = 90.0;
    default_land_base_moisture: f32 = 20.0;
}

impl Default for ClimateCfg {
    fn default() -> Self {
        Self {
            moisture_iterations: default_moisture_iterations(),
            moisture_tile_cells: default_moisture_tile_cells(),
            orographic_drop_per_1000m: default_orographic_drop_per_1000m(),
            ocean_evaporation_rate: default_ocean_evaporation_rate(),
            ocean_base_moisture: default_ocean_base_moisture(),
            land_base_moisture: default_land_base_moisture(),
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, PartialEq)]
#[serde(default)]
pub struct BiomesCfg {
    #[serde(default = "default_glacier_temp_c")]
    pub glacier_temp_c: f32,
    #[serde(default = "default_snow_surface_temp_c")]
    pub snow_surface_temp_c: f32,
    #[serde(default = "default_alpine_elevation_m")]
    pub alpine_elevation_m: f64,
    #[serde(default = "default_beach_elevation_max_m")]
    pub beach_elevation_max_m: f64,
    #[serde(default = "default_desert_temp_c")]
    pub desert_temp_c: f32,
    #[serde(default = "default_desert_moisture_pct")]
    pub desert_moisture_pct: f32,
    #[serde(default = "default_dirt_depth_blocks")]
    pub dirt_depth_blocks: i32,
}

default_fns! {
    default_glacier_temp_c: f32 = -8.0;
    default_snow_surface_temp_c: f32 = -5.0;
    default_alpine_elevation_m: f64 = 3500.0;
    default_beach_elevation_max_m: f64 = 3.0;
    default_desert_temp_c: f32 = 20.0;
    default_desert_moisture_pct: f32 = 25.0;
    default_dirt_depth_blocks: i32 = 3;
}

impl Default for BiomesCfg {
    fn default() -> Self {
        Self {
            glacier_temp_c: default_glacier_temp_c(),
            snow_surface_temp_c: default_snow_surface_temp_c(),
            alpine_elevation_m: default_alpine_elevation_m(),
            beach_elevation_max_m: default_beach_elevation_max_m(),
            desert_temp_c: default_desert_temp_c(),
            desert_moisture_pct: default_desert_moisture_pct(),
            dirt_depth_blocks: default_dirt_depth_blocks(),
        }
    }
}

#[derive(Deserialize, Clone, Copy, Debug, Default, PartialEq)]
#[serde(default)]
pub struct TerrainCfg {
    pub world: WorldCfg,
    pub uplift: UpliftCfg,
    pub noise: NoiseCfg,
    pub climate: ClimateCfg,
    pub biomes: BiomesCfg,
}

/// Parse `terrain.toml` contents. Returns `TerrainCfg::default()` (which
/// reproduces the previously-hardcoded worldgen constants exactly) if
/// `toml_str` is empty (file missing) or fails to parse -- a typo'd config
/// falls back quietly rather than panicking or blocking world load.
pub fn parse_terrain_config(toml_str: &str) -> TerrainCfg {
    if toml_str.is_empty() {
        return TerrainCfg::default();
    }
    toml::from_str(toml_str).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_gives_defaults() {
        assert_eq!(parse_terrain_config(""), TerrainCfg::default());
    }

    #[test]
    fn invalid_toml_falls_back_to_defaults() {
        assert_eq!(
            parse_terrain_config("not valid toml {{{"),
            TerrainCfg::default()
        );
    }

    /// The shipped datapack file documents every default value inline --
    /// keep it honest: it must actually parse to `TerrainCfg::default()`,
    /// not silently drift from the values it claims.
    #[test]
    fn shipped_datapack_file_matches_defaults() {
        let shipped = include_str!("../../../games/cubic-game/data/world/terrain.toml");
        assert_eq!(parse_terrain_config(shipped), TerrainCfg::default());
    }

    #[test]
    fn partial_section_override_keeps_other_fields_at_default() {
        let cfg = parse_terrain_config(
            r#"
            [world]
            plate_count = 12
            "#,
        );
        assert_eq!(cfg.world.plate_count, 12);
        // continental_fraction wasn't specified -- still the default.
        assert_eq!(
            cfg.world.continental_fraction,
            WorldCfg::default().continental_fraction
        );
    }

    #[test]
    fn missing_section_falls_back_entirely() {
        let cfg = parse_terrain_config(
            r#"
            [world]
            plate_count = 12
            "#,
        );
        assert_eq!(cfg.biomes, BiomesCfg::default());
    }

    #[test]
    fn full_override_round_trips() {
        let cfg = parse_terrain_config(
            r#"
            [world]
            plate_count = 12
            continental_fraction = 0.5

            [biomes]
            glacier_temp_c = -10.0
            dirt_depth_blocks = 5
            "#,
        );
        assert_eq!(cfg.world.plate_count, 12);
        assert_eq!(cfg.world.continental_fraction, 0.5);
        assert_eq!(cfg.biomes.glacier_temp_c, -10.0);
        assert_eq!(cfg.biomes.dirt_depth_blocks, 5);
        // Untouched fields in a partially-overridden section stay default.
        assert_eq!(
            cfg.biomes.snow_surface_temp_c,
            BiomesCfg::default().snow_surface_temp_c
        );
    }
}
