// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

//! Regional climate: per-column temperature (pure math, no cache needed)
//! and a wind-driven moisture simulation. Moisture requires knowing upwind
//! terrain for hundreds of km, so it's computed as 2D grid tiles, cached in
//! memory, and recomputed on demand deterministically from the world seed.

use crate::tectonics::TectonicPlates;
use crate::terrain_config::ClimateCfg;
use crate::world_constants::{
    MIN_OCEAN_FLOOR_M, MOISTURE_GRID_CELL_M, TEMP_LAPSE_RATE_C_PER_M, latitude_to_base_temp_c,
    world_z_to_latitude,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

pub struct ClimateSample {
    /// Surface temperature in Celsius (before elevation lapse).
    pub surface_temp_c: f32,
    /// Temperature at the given elevation in Celsius.
    pub temp_c: f32,
    /// Moisture 0.0-100.0 percent.
    pub moisture_pct: f32,
    /// Base wind direction at this latitude (normalized-ish XZ vector).
    /// Not yet consumed by block placement -- reserved for wind-dependent
    /// effects (e.g. biome refinement, F3 debug readout) added by later cards.
    #[allow(dead_code)]
    pub wind_dir: [f32; 2],
}

/// Per-column temperature: sea-level baseline from latitude, then lapsed
/// down/up to `elevation_m`. Pure function, no cache needed.
pub fn sample_temperature(_x: f64, z: f64, elevation_m: f64) -> (f32, f32) {
    let lat = world_z_to_latitude(z);
    let surface_temp = latitude_to_base_temp_c(lat) as f32;
    let temp_at_elevation = surface_temp + (elevation_m * TEMP_LAPSE_RATE_C_PER_M) as f32;
    (surface_temp, temp_at_elevation)
}

/// Base wind direction by latitude band -- a simplified model of Earth's
/// major circulation cells (trade winds, westerlies, polar easterlies).
pub fn wind_direction(lat: f64) -> [f32; 2] {
    let abs_lat = lat.abs();
    let lat_sign = lat.signum() as f32;
    if abs_lat < 0.23 {
        // Trade winds: blow toward the equator from ~23°.
        [-1.0_f32, -lat_sign * 0.3]
    } else if abs_lat < 0.66 {
        // Westerlies.
        [1.0_f32, lat_sign * 0.2]
    } else {
        // Polar easterlies.
        [-1.0_f32, lat_sign * 0.1]
    }
}

// ---------------------------------------------------------------------------
// Moisture tile
// ---------------------------------------------------------------------------

pub struct MoistureTile {
    /// World XZ origin of this tile in meters.
    pub origin: [f64; 2],
    /// Width and height in cells.
    pub size: [usize; 2],
    /// Cell size in meters.
    pub cell_m: f64,
    /// Moisture values 0.0-100.0, row-major `[z][x]`.
    pub data: Vec<f32>,
}

impl MoistureTile {
    /// Bilinear-interpolated moisture at a world position. Positions
    /// outside the tile clamp to the nearest edge.
    pub fn sample(&self, x: f64, z: f64) -> f32 {
        let gx = ((x - self.origin[0]) / self.cell_m).clamp(0.0, (self.size[0] - 1) as f64);
        let gz = ((z - self.origin[1]) / self.cell_m).clamp(0.0, (self.size[1] - 1) as f64);
        let x0 = gx.floor() as usize;
        let z0 = gz.floor() as usize;
        let x1 = (x0 + 1).min(self.size[0] - 1);
        let z1 = (z0 + 1).min(self.size[1] - 1);
        let fx = (gx - x0 as f64) as f32;
        let fz = (gz - z0 as f64) as f32;

        let v00 = self.data[z0 * self.size[0] + x0];
        let v10 = self.data[z0 * self.size[0] + x1];
        let v01 = self.data[z1 * self.size[0] + x0];
        let v11 = self.data[z1 * self.size[0] + x1];

        let top = v00 * (1.0 - fx) + v10 * fx;
        let bottom = v01 * (1.0 - fx) + v11 * fx;
        top * (1.0 - fz) + bottom * fz
    }
}

/// Compute a moisture tile (`cfg.moisture_tile_cells` square) at `origin`
/// by seeding moisture from ocean/land and then running wind-driven
/// advection with an orographic (rain shadow) effect. Deterministic given
/// `origin` and `tectonic` -- `seed` is accepted for API symmetry with the
/// rest of worldgen but isn't currently needed (the simulation has no
/// stochastic component of its own).
pub fn compute_moisture_tile(
    origin: [f64; 2],
    _seed: u64,
    tectonic: &TectonicPlates,
    cfg: &ClimateCfg,
) -> MoistureTile {
    let size = [cfg.moisture_tile_cells, cfg.moisture_tile_cells];
    let cell_m = MOISTURE_GRID_CELL_M;
    let mut moisture = vec![0.0f32; size[0] * size[1]];

    // --- Pass 1: seed moisture from ocean/land ---
    for zi in 0..size[1] {
        for xi in 0..size[0] {
            let wx = origin[0] + xi as f64 * cell_m;
            let wz = origin[1] + zi as f64 * cell_m;
            let tec = tectonic.sample(wx, wz);
            moisture[zi * size[0] + xi] = if tec.uplift_m < 0.0 {
                cfg.ocean_base_moisture + (tec.uplift_m / MIN_OCEAN_FLOOR_M as f32) * 10.0
            } else {
                cfg.land_base_moisture
            };
        }
    }

    // --- Pass 2: wind advection + orographic effect ---
    for _iter in 0..cfg.moisture_iterations {
        let prev = moisture.clone();
        for zi in 0..size[1] {
            for xi in 0..size[0] {
                let wx = origin[0] + xi as f64 * cell_m;
                let wz = origin[1] + zi as f64 * cell_m;
                let lat = world_z_to_latitude(wz);
                let wind = wind_direction(lat);

                let upwind_xi = (xi as f64 - wind[0] as f64).clamp(0.0, (size[0] - 1) as f64);
                let upwind_zi = (zi as f64 - wind[1] as f64).clamp(0.0, (size[1] - 1) as f64);
                let upwind_xi = upwind_xi as usize;
                let upwind_zi = upwind_zi as usize;
                let upwind_moisture = prev[upwind_zi * size[0] + upwind_xi];

                let tec = tectonic.sample(wx, wz);
                let is_ocean = tec.uplift_m < 0.0;

                let uwx = origin[0] + upwind_xi as f64 * cell_m;
                let uwz = origin[1] + upwind_zi as f64 * cell_m;
                let upwind_tec = tectonic.sample(uwx, uwz);

                // Orographic lift: moisture drops when terrain rises from
                // the upwind cell to this one; descending doesn't refund it.
                let height_diff = (tec.uplift_m - upwind_tec.uplift_m).max(0.0);
                let orographic_drop = (height_diff / 1000.0) * cfg.orographic_drop_per_1000m;

                let advected = upwind_moisture - orographic_drop;

                // Ocean evaporation: ocean tiles slowly restore moisture.
                let restored = if is_ocean {
                    advected + (cfg.ocean_base_moisture - advected) * cfg.ocean_evaporation_rate
                } else {
                    advected
                };

                moisture[zi * size[0] + xi] = restored.clamp(0.0, 100.0);
            }
        }
    }

    MoistureTile {
        origin,
        size,
        cell_m,
        data: moisture,
    }
}

// ---------------------------------------------------------------------------
// Regional climate cache
// ---------------------------------------------------------------------------

pub struct ClimateCache {
    /// Active moisture tiles, keyed by tile origin quantized to tile size.
    tiles: HashMap<[i64; 2], MoistureTile>,
    /// LRU order for eviction; back = most recently used.
    lru: VecDeque<[i64; 2]>,
    /// Max tiles kept in memory (each ~360KB, so 20 tiles = ~7MB).
    max_tiles: usize,
    /// Tile edge length in cells -- must match whatever `ClimateCfg` this
    /// cache is used with, so tile keys/origins line up with tiles actually
    /// computed by `compute_moisture_tile`.
    tile_cells: usize,
}

impl ClimateCache {
    pub fn new(max_tiles: usize, tile_cells: usize) -> Self {
        Self {
            tiles: HashMap::new(),
            lru: VecDeque::new(),
            max_tiles,
            tile_cells,
        }
    }

    pub fn sample_moisture(
        &mut self,
        x: f64,
        z: f64,
        seed: u64,
        tectonic: &TectonicPlates,
        cfg: &ClimateCfg,
    ) -> f32 {
        let tile_key = self.tile_key(x, z);
        if !self.tiles.contains_key(&tile_key) {
            self.evict_if_needed();
            let origin = self.tile_origin(tile_key);
            let tile = compute_moisture_tile(origin, seed, tectonic, cfg);
            self.tiles.insert(tile_key, tile);
            self.lru.push_back(tile_key);
        } else {
            self.lru.retain(|k| k != &tile_key);
            self.lru.push_back(tile_key);
        }
        self.tiles[&tile_key].sample(x, z)
    }

    /// Number of tiles currently resident -- exposed for tests/diagnostics.
    #[allow(dead_code)]
    pub fn resident_tiles(&self) -> usize {
        self.tiles.len()
    }

    fn evict_if_needed(&mut self) {
        if self.tiles.len() >= self.max_tiles
            && let Some(oldest) = self.lru.pop_front()
        {
            self.tiles.remove(&oldest);
        }
    }

    fn tile_span_m(&self) -> f64 {
        self.tile_cells as f64 * MOISTURE_GRID_CELL_M
    }

    fn tile_key(&self, x: f64, z: f64) -> [i64; 2] {
        let tile_m = self.tile_span_m();
        [(x / tile_m).floor() as i64, (z / tile_m).floor() as i64]
    }

    fn tile_origin(&self, key: [i64; 2]) -> [f64; 2] {
        let tile_m = self.tile_span_m();
        [key[0] as f64 * tile_m, key[1] as f64 * tile_m]
    }
}

/// Full climate sample at a world position: temperature (direct) plus
/// moisture (via the cache, computing a tile on first access).
pub fn sample_climate(
    x: f64,
    z: f64,
    elevation_m: f64,
    seed: u64,
    tectonic: &TectonicPlates,
    cache: &Mutex<ClimateCache>,
    cfg: &ClimateCfg,
) -> ClimateSample {
    let (surface_temp, temp_c) = sample_temperature(x, z, elevation_m);
    let moisture_pct = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .sample_moisture(x, z, seed, tectonic, cfg);
    let wind_dir = wind_direction(world_z_to_latitude(z));
    ClimateSample {
        surface_temp_c: surface_temp,
        temp_c,
        moisture_pct,
        wind_dir,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonics::Plate;

    fn plates(entries: &[(f64, f64, f32)]) -> TectonicPlates {
        TectonicPlates {
            plates: entries
                .iter()
                .enumerate()
                .map(|(i, &(cx, cz, density))| Plate {
                    center: [cx, cz],
                    velocity: [0.0, 0.0],
                    density,
                    id: i as u32,
                })
                .collect(),
            uplift: crate::terrain_config::UpliftCfg::default(),
        }
    }

    // --- temperature ---

    #[test]
    fn temperature_matches_latitude_curve_at_sea_level() {
        let (surface, at_elevation) = sample_temperature(0.0, 0.0, 0.0);
        assert!((surface - 30.0).abs() < 1e-3);
        assert!((at_elevation - surface).abs() < 1e-6, "no lapse at 0m");
    }

    #[test]
    fn temperature_cools_with_elevation() {
        let (surface, at_elevation) = sample_temperature(0.0, 0.0, 1000.0);
        assert!(
            at_elevation < surface,
            "1000m up should be cooler than sea level: {at_elevation} vs {surface}"
        );
        // TEMP_LAPSE_RATE_C_PER_M = -0.0065, so 1000m should cool ~6.5C.
        assert!((surface - at_elevation - 6.5).abs() < 1e-3);
    }

    // --- wind ---

    #[test]
    fn wind_bands_match_circulation_cells() {
        assert_eq!(wind_direction(0.1)[0], -1.0); // trade winds
        assert_eq!(wind_direction(0.4)[0], 1.0); // westerlies
        assert_eq!(wind_direction(0.8)[0], -1.0); // polar easterlies
    }

    // --- moisture tile bilinear sampling ---

    #[test]
    fn moisture_tile_sample_interpolates_between_cells() {
        let size = [2usize, 2];
        let tile = MoistureTile {
            origin: [0.0, 0.0],
            size,
            cell_m: 10.0,
            data: vec![0.0, 100.0, 0.0, 100.0], // right column is 100, left is 0
        };
        assert!((tile.sample(0.0, 0.0) - 0.0).abs() < 1e-3);
        assert!((tile.sample(10.0, 0.0) - 100.0).abs() < 1e-3);
        assert!((tile.sample(5.0, 0.0) - 50.0).abs() < 1e-3);
    }

    #[test]
    fn moisture_tile_sample_clamps_outside_bounds() {
        let tile = MoistureTile {
            origin: [0.0, 0.0],
            size: [2, 2],
            cell_m: 10.0,
            data: vec![5.0, 5.0, 5.0, 5.0],
        };
        assert!((tile.sample(-1000.0, -1000.0) - 5.0).abs() < 1e-3);
        assert!((tile.sample(1000.0, 1000.0) - 5.0).abs() < 1e-3);
    }

    // --- moisture simulation ---

    #[test]
    fn ocean_tile_ends_up_wetter_than_land_tile() {
        let origin = [-150_000.0, -150_000.0];
        let ocean = plates(&[(-5_000_000.0, 0.0, 0.8), (-4_000_000.0, 0.0, 0.85)]);
        let land = plates(&[(-5_000_000.0, 0.0, 0.3), (-4_000_000.0, 0.0, 0.35)]);

        let ocean_tile = compute_moisture_tile(origin, 1, &ocean, &ClimateCfg::default());
        let land_tile = compute_moisture_tile(origin, 1, &land, &ClimateCfg::default());

        let mean = |t: &MoistureTile| t.data.iter().sum::<f32>() / t.data.len() as f32;
        let ocean_mean = mean(&ocean_tile);
        let land_mean = mean(&land_tile);

        assert!(
            ocean_mean > 60.0,
            "ocean tile should be predominantly wet, got mean {ocean_mean}"
        );
        assert!(
            ocean_mean > land_mean,
            "ocean tile ({ocean_mean}) should be wetter than land tile ({land_mean})"
        );
    }

    #[test]
    fn moisture_depletes_progressively_while_climbing_windward_slope() {
        // Two continental plates converging head-on at x=0 build a
        // symmetric mountain ridge (see the equivalent test in
        // tectonics.rs). At the equator (z=0), wind blows in -X (from
        // high x toward low x), so the +X foot of the ridge is windward.
        //
        // Each advection pass only shifts information by one grid cell
        // (the wind vector's components are exactly +/-1), so after
        // MOISTURE_ADVECTION_ITERATIONS (40) passes a cell's moisture only
        // reflects the climb over the *nearest* ~40 cells upwind of it --
        // not the full ~150-cell climb from tile edge to peak. That's too
        // short a memory to carry a signal all the way across this ridge
        // and back down the far (leeward) side within one tile, so this
        // test checks the directly observable effect instead: moisture
        // drops monotonically as a windward point gets closer to the peak
        // (i.e. has more climbing behind it within reach of the sim).
        let tectonic = TectonicPlates {
            plates: vec![
                Plate {
                    center: [-1_000_000.0, 0.0],
                    velocity: [10.0, 0.0],
                    density: 0.3,
                    id: 0,
                },
                Plate {
                    center: [1_000_000.0, 0.0],
                    velocity: [-10.0, 0.0],
                    density: 0.3,
                    id: 1,
                },
            ],
            uplift: crate::terrain_config::UpliftCfg::default(),
        };
        let origin = [-150_000.0, -150_000.0];
        let tile = compute_moisture_tile(origin, 1, &tectonic, &ClimateCfg::default());

        let far = tile.sample(140_000.0, 0.0); // near the source edge, least climbed
        let mid = tile.sample(120_000.0, 0.0);
        let near = tile.sample(100_000.0, 0.0); // closer to the peak, most climbed

        assert!(
            far > mid && mid > near,
            "moisture should deplete monotonically approaching the peak: far={far} mid={mid} near={near}"
        );
    }

    // --- cache ---

    #[test]
    fn cache_reuses_tile_for_repeated_queries_in_same_region() {
        let tectonic = plates(&[(-5_000_000.0, 0.0, 0.3), (-4_000_000.0, 0.0, 0.35)]);
        let mut cache = ClimateCache::new(4, ClimateCfg::default().moisture_tile_cells);
        let a = cache.sample_moisture(1000.0, 1000.0, 1, &tectonic, &ClimateCfg::default());
        assert_eq!(cache.resident_tiles(), 1);
        let b = cache.sample_moisture(2000.0, 2000.0, 1, &tectonic, &ClimateCfg::default());
        assert_eq!(cache.resident_tiles(), 1, "same tile, no new computation");
        assert!((a - b).abs() < 50.0); // same tile region, not wildly different
    }

    #[test]
    fn cache_evicts_oldest_tile_past_capacity() {
        let tectonic = plates(&[(-5_000_000.0, 0.0, 0.3), (-4_000_000.0, 0.0, 0.35)]);
        let mut cache = ClimateCache::new(1, ClimateCfg::default().moisture_tile_cells);
        cache.sample_moisture(0.0, 0.0, 1, &tectonic, &ClimateCfg::default());
        assert_eq!(cache.resident_tiles(), 1);
        // Far enough away to land in a different tile (tile span is 300km).
        cache.sample_moisture(
            10_000_000.0,
            10_000_000.0,
            1,
            &tectonic,
            &ClimateCfg::default(),
        );
        assert_eq!(
            cache.resident_tiles(),
            1,
            "capacity 1 should evict the first tile, not grow"
        );
    }

    #[test]
    fn sample_climate_combines_temperature_moisture_and_wind() {
        let tectonic = plates(&[(-5_000_000.0, 0.0, 0.3), (-4_000_000.0, 0.0, 0.35)]);
        let cache = Mutex::new(ClimateCache::new(
            4,
            ClimateCfg::default().moisture_tile_cells,
        ));
        let sample = sample_climate(0.0, 0.0, 0.0, 1, &tectonic, &cache, &ClimateCfg::default());
        assert!((sample.surface_temp_c - 30.0).abs() < 1e-3);
        assert!((0.0..=100.0).contains(&sample.moisture_pct));
        assert_eq!(sample.wind_dir, wind_direction(0.0));
    }
}
