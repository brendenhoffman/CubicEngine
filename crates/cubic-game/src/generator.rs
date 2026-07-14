// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

//! Per-column block placement: turns a terrain height (from `heightmap.rs`)
//! and a climate sample (from `climate.rs`) into actual block types --
//! surface material, subsurface fill, ocean water, and glacier/sea ice.
//! Pure and chunk-agnostic: given a chunk's Y index and the column's
//! already-computed surface height/climate, it returns that chunk's slice
//! of the column. Kept separate from the WASM glue in `lib.rs` so it's
//! directly unit-testable.

use crate::climate::ClimateSample;
use crate::terrain_config::BiomesCfg;
use crate::world_constants::SEA_LEVEL_M;
use crate::{CHUNK_SIZE, VOXEL_SIZE};

/// Registered block type ids this generator places, resolved once in
/// `on_load` from the scanned `blocks/` datapack.
pub struct BlockIds {
    pub stone: u32,
    pub dirt: u32,
    pub grass: u32,
    pub sand: u32,
    pub gravel: u32,
    pub snow: u32,
    pub ice: u32,
    pub water: u32,
}

/// Below this elevation, "cold" terrain still gets gravel rather than snow
/// (cold, low-lying shores). Not yet part of the live config schema.
const COLD_GRAVEL_ELEVATION_M: f64 = 500.0;
/// Surface temperature below which ocean water at y=0 freezes into sea ice.
/// Not yet part of the live config schema.
const SEA_ICE_TEMP_C: f32 = -2.0;
/// Additional glacier ice depth (in blocks) per degree colder than
/// `cfg.glacier_temp_c`. Not yet part of the live config schema.
const GLACIER_DEPTH_BLOCKS_PER_DEGREE: f32 = 2.0;

/// Generate one chunk's worth (`CHUNK_SIZE` voxels) of a single XZ column.
/// `chunk_y` is the chunk's Y index (chunk-local voxel 0 sits at world Y
/// `chunk_y * CHUNK_SIZE * VOXEL_SIZE`). `surface_m` is this column's
/// terrain height from `heightmap::sample_heightmap`. Returns block ids
/// indexed by local Y (0 = air).
pub fn generate_column(
    chunk_y: i32,
    surface_m: f64,
    climate: &ClimateSample,
    ids: &BlockIds,
    cfg: &BiomesCfg,
) -> [u32; CHUNK_SIZE] {
    let surface_blocks = (surface_m / VOXEL_SIZE as f64).round() as i32;

    let is_beach = surface_m > SEA_LEVEL_M && surface_m < cfg.beach_elevation_max_m;
    let is_desert =
        climate.temp_c > cfg.desert_temp_c && climate.moisture_pct < cfg.desert_moisture_pct;
    let is_cold = climate.temp_c < cfg.snow_surface_temp_c;
    let is_high_alpine = surface_m > cfg.alpine_elevation_m;

    let surface_block = if surface_m < SEA_LEVEL_M {
        ids.stone // underwater -- water fills above it separately
    } else if is_beach || is_desert {
        ids.sand
    } else if is_high_alpine || (is_cold && surface_m > COLD_GRAVEL_ELEVATION_M) {
        ids.gravel
    } else if is_cold {
        ids.snow
    } else {
        ids.grass
    };

    let dirt_depth = if surface_m < SEA_LEVEL_M || is_beach || is_desert || is_high_alpine {
        0
    } else {
        cfg.dirt_depth_blocks
    };

    // Glacier: land colder than cfg.glacier_temp_c grows ice beneath the
    // snow cap, deeper the colder it gets. 0 outside a glacier zone.
    let glacier_depth_blocks = if climate.surface_temp_c < cfg.glacier_temp_c
        && surface_m > SEA_LEVEL_M
    {
        (((cfg.glacier_temp_c - climate.surface_temp_c) * GLACIER_DEPTH_BLOCKS_PER_DEGREE) as i32)
            .max(0)
    } else {
        0
    };

    let mut blocks = [0u32; CHUNK_SIZE];
    for (local_y, block) in blocks.iter_mut().enumerate() {
        let world_y_blocks = chunk_y * CHUNK_SIZE as i32 + local_y as i32;
        let world_y_m = world_y_blocks as f64 * VOXEL_SIZE as f64;

        *block = if world_y_blocks == surface_blocks && surface_m >= SEA_LEVEL_M {
            surface_block
        } else if world_y_blocks < surface_blocks {
            let depth_from_surface = surface_blocks - world_y_blocks;
            if glacier_depth_blocks > 0 && depth_from_surface <= glacier_depth_blocks {
                ids.ice
            } else if depth_from_surface <= dirt_depth {
                if surface_block == ids.grass {
                    ids.dirt
                } else {
                    ids.stone
                }
            } else {
                ids.stone
            }
        } else if world_y_m <= SEA_LEVEL_M && surface_m < SEA_LEVEL_M {
            // Ocean: water up to sea level, sea ice replacing just the
            // topmost layer (world_y_blocks == 0) when cold enough.
            if world_y_blocks == 0 && climate.surface_temp_c < SEA_ICE_TEMP_C && surface_m < 0.5 {
                ids.ice
            } else {
                ids.water
            }
        } else {
            0 // air
        };
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids() -> BlockIds {
        BlockIds {
            stone: 1,
            dirt: 2,
            grass: 3,
            sand: 4,
            gravel: 5,
            snow: 6,
            ice: 7,
            water: 8,
        }
    }

    fn climate(surface_temp_c: f32, temp_c: f32, moisture_pct: f32) -> ClimateSample {
        ClimateSample {
            surface_temp_c,
            temp_c,
            moisture_pct,
            wind_dir: [0.0, 0.0],
        }
    }

    #[test]
    fn temperate_grass_column_has_grass_over_dirt_over_stone() {
        let ids = ids();
        // Surface at world Y 8m -> chunk_y=0 covers it (0..16m).
        let blocks = generate_column(
            0,
            8.0,
            &climate(15.0, 15.0, 50.0),
            &ids,
            &BiomesCfg::default(),
        );
        let surface_index = (8.0 / VOXEL_SIZE as f64).round() as usize;
        assert_eq!(blocks[surface_index], ids.grass);
        assert_eq!(blocks[surface_index - 1], ids.dirt);
        assert_eq!(blocks[surface_index - 4], ids.stone); // past dirt_depth=3
        assert_eq!(blocks[surface_index + 1], 0); // air above surface
    }

    #[test]
    fn ocean_column_fills_water_to_sea_level_over_stone_floor() {
        let ids = ids();
        // Ocean floor at -20m, chunk_y=0 spans world Y [0, 16).
        // Use chunk_y=-1 to cover the floor and sea level (world Y [-16, 0)).
        let blocks = generate_column(
            -1,
            -20.0,
            &climate(15.0, 15.0, 90.0),
            &ids,
            &BiomesCfg::default(),
        );
        // local_y=31 -> world_y_blocks = -1*32+31 = -1 -> world_y_m=-0.5 (water)
        assert_eq!(blocks[31], ids.water);
        // Floor is at -20m = -40 blocks, not reached at chunk_y=-1 (covers
        // -32..-1 blocks = -16..-0.5m), so everything in range is water.
        assert_eq!(blocks[0], ids.water);
    }

    #[test]
    fn beach_column_has_sand_directly_over_stone() {
        let ids = ids();
        let blocks = generate_column(
            0,
            1.5,
            &climate(22.0, 22.0, 50.0),
            &ids,
            &BiomesCfg::default(),
        );
        let surface_index = (1.5 / VOXEL_SIZE as f64).round() as usize;
        assert_eq!(blocks[surface_index], ids.sand);
        assert_eq!(blocks[surface_index - 1], ids.stone); // no dirt on a beach
    }

    #[test]
    fn desert_column_has_sand_even_at_moderate_elevation() {
        let ids = ids();
        let blocks = generate_column(
            0,
            10.0,
            &climate(30.0, 30.0, 10.0),
            &ids,
            &BiomesCfg::default(),
        );
        let surface_index = (10.0 / VOXEL_SIZE as f64).round() as usize;
        assert_eq!(blocks[surface_index], ids.sand);
    }

    #[test]
    fn high_alpine_column_has_gravel_regardless_of_climate() {
        let ids = ids();
        let surface_m = 4000.0;
        let surface_blocks = (surface_m / VOXEL_SIZE as f64).round() as i32;
        let chunk_y = surface_blocks.div_euclid(CHUNK_SIZE as i32);
        let surface_index = surface_blocks.rem_euclid(CHUNK_SIZE as i32) as usize;
        let blocks = generate_column(
            chunk_y,
            surface_m,
            &climate(10.0, -15.0, 50.0),
            &ids,
            &BiomesCfg::default(),
        );
        assert_eq!(blocks[surface_index], ids.gravel);
    }

    #[test]
    fn cold_low_elevation_column_has_snow_surface() {
        let ids = ids();
        let blocks = generate_column(
            0,
            10.0,
            &climate(-10.0, -10.0, 50.0),
            &ids,
            &BiomesCfg::default(),
        );
        let surface_index = (10.0 / VOXEL_SIZE as f64).round() as usize;
        assert_eq!(blocks[surface_index], ids.snow);
    }

    #[test]
    fn glacier_column_has_ice_beneath_snow_cap() {
        let ids = ids();
        let blocks = generate_column(
            0,
            10.0,
            &climate(-10.0, -10.0, 60.0),
            &ids,
            &BiomesCfg::default(),
        );
        let surface_index = (10.0 / VOXEL_SIZE as f64).round() as usize;
        assert_eq!(blocks[surface_index], ids.snow); // surface still snow
        assert_eq!(blocks[surface_index - 1], ids.ice); // glacier body below
    }

    #[test]
    fn sea_ice_replaces_top_water_layer_when_very_cold() {
        let ids = ids();
        // Ocean floor well below sea level; chunk_y=-1 covers world Y
        // [-16, 0), so local_y=31 is world Y block -1 (world_y_m=-0.5,
        // just under sea level) and world_y_blocks==0 is out of this
        // chunk's range -- use chunk_y=0 to include the y=0 block itself.
        let blocks = generate_column(
            0,
            -20.0,
            &climate(-10.0, -10.0, 60.0),
            &ids,
            &BiomesCfg::default(),
        );
        assert_eq!(blocks[0], ids.ice, "y=0 water should freeze when very cold");
        assert_eq!(blocks[1], 0, "above sea level is open air, not water");
    }

    #[test]
    fn warm_ocean_has_no_sea_ice() {
        let ids = ids();
        let blocks = generate_column(
            0,
            -20.0,
            &climate(15.0, 15.0, 60.0),
            &ids,
            &BiomesCfg::default(),
        );
        assert_eq!(blocks[0], ids.water);
    }

    #[test]
    fn air_above_surface_on_land() {
        let ids = ids();
        let blocks = generate_column(
            0,
            5.0,
            &climate(15.0, 15.0, 50.0),
            &ids,
            &BiomesCfg::default(),
        );
        let surface_index = (5.0 / VOXEL_SIZE as f64).round() as usize;
        for b in &blocks[surface_index + 1..] {
            assert_eq!(*b, 0);
        }
    }
}
