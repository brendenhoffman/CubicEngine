// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

//! Terrain heightmap: a pure function of (x, z, seed, tectonic sample) that
//! combines the tectonic envelope from `tectonics.rs` with multi-scale
//! noise to produce a plausible terrain height in meters. No caching --
//! it's fast and deterministic enough to call directly per column.

use crate::tectonics::TectonicSample;
use crate::terrain_config::NoiseCfg;
use crate::world_constants::{MAX_TERRAIN_HEIGHT_M, MIN_OCEAN_FLOOR_M};
use noise::{NoiseFn, OpenSimplex};

/// Target elevation mid-ocean ridges pull the ocean floor toward, at the
/// crest of a divergent boundary.
const MID_OCEAN_RIDGE_HEIGHT_M: f32 = -200.0;
/// Distance over which the mid-ocean ridge's influence falls off.
const MID_OCEAN_RIDGE_FALLOFF_M: f64 = 100_000.0;

/// Minimum additional subsidence at a convergent oceanic boundary (subduction trench).
const ABYSSAL_TRENCH_MIN_DEPTH_M: f32 = -500.0;
/// Additional subsidence on top of the minimum at full convergence strength,
/// so a strong convergent oceanic boundary reaches -1000m total.
const ABYSSAL_TRENCH_EXTRA_DEPTH_M: f32 = -500.0;
/// Distance over which a trench's influence falls off -- narrower than the
/// mid-ocean ridge since subduction trenches are tight features.
const ABYSSAL_TRENCH_FALLOFF_M: f64 = 50_000.0;

/// Seeded, deterministic multi-octave noise. Wraps the `noise` crate's
/// OpenSimplex implementation rather than hand-rolling permutation tables
/// (the card allows either).
pub struct NoiseState {
    seed: u64,
}

impl NoiseState {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Single-octave simplex noise in roughly [-1.0, 1.0], seeded directly
    /// from the world seed (no decorrelation offset). Not yet called by
    /// `sample_heightmap` (which only uses `fbm`), but part of the API the
    /// card specifies for future direct use.
    #[allow(dead_code)]
    pub fn simplex(&self, x: f64, z: f64) -> f32 {
        OpenSimplex::new(self.seed as u32).get([x, z]) as f32
    }

    /// Fractal Brownian motion: `octaves` layers of the same noise source
    /// at doubling frequency and halving amplitude, normalized back to
    /// roughly [-1.0, 1.0]. `seed_xor` decorrelates this call's noise field
    /// from other `fbm`/`simplex` calls sharing the same `NoiseState`.
    pub fn fbm(&self, x: f64, z: f64, seed_xor: u64, octaves: u32) -> f32 {
        let noise = OpenSimplex::new((self.seed ^ seed_xor) as u32);
        let mut sum = 0.0f32;
        let mut amplitude = 1.0f32;
        let mut max_amplitude = 0.0f32;
        let mut frequency = 1.0f64;
        for _ in 0..octaves {
            sum += noise.get([x * frequency, z * frequency]) as f32 * amplitude;
            max_amplitude += amplitude;
            amplitude *= 0.5;
            frequency *= 2.0;
        }
        sum / max_amplitude.max(1e-9)
    }
}

/// Combine the tectonic envelope with three noise layers (regional, local,
/// mountain-ridge detail), a coastal irregularity blend, and ocean-floor
/// detail (mid-ocean ridges, abyssal trenches) into a terrain height.
pub fn sample_heightmap(
    x: f64,
    z: f64,
    seed: u64,
    tec: &TectonicSample,
    noise: &NoiseState,
    cfg: &NoiseCfg,
) -> f32 {
    let mut tec_base = tec
        .uplift_m
        .clamp(MIN_OCEAN_FLOOR_M as f32, MAX_TERRAIN_HEIGHT_M as f32);

    // Coastline shaping: within the blend zone of sea level, domain-warp
    // the boundary so it's an irregular coastline (bays, peninsulas)
    // rather than a straight Voronoi edge.
    if tec_base.abs() < cfg.coastal_blend_zone_m {
        let coastal_noise = noise.fbm(
            x * cfg.coastal_frequency,
            z * cfg.coastal_frequency,
            seed ^ 0x4444,
            cfg.coastal_octaves,
        ) * cfg.coastal_amplitude;
        tec_base += coastal_noise;
    }

    // Ocean floor detail: mid-ocean ridges at divergent boundaries, abyssal
    // trenches at convergent oceanic boundaries. Not yet part of the live
    // config schema (the card's terrain.toml doesn't list ocean-floor-detail
    // keys) -- stays hardcoded.
    if tec_base < 0.0 {
        let boundary_distance = (tec.boundary_distance_m as f64).max(0.0);
        if tec.boundary_type < -0.3 {
            let divergence_strength = ((-tec.boundary_type - 0.3) / 0.7).clamp(0.0, 1.0);
            let falloff = (-boundary_distance / MID_OCEAN_RIDGE_FALLOFF_M).exp() as f32;
            let ridge_strength = divergence_strength * falloff;
            tec_base += (MID_OCEAN_RIDGE_HEIGHT_M - tec_base) * ridge_strength;
        } else if tec.boundary_type > 0.3 {
            let convergence_strength = ((tec.boundary_type - 0.3) / 0.7).clamp(0.0, 1.0);
            let falloff = (-boundary_distance / ABYSSAL_TRENCH_FALLOFF_M).exp() as f32;
            let trench_depth =
                ABYSSAL_TRENCH_MIN_DEPTH_M + ABYSSAL_TRENCH_EXTRA_DEPTH_M * convergence_strength;
            tec_base += trench_depth * falloff;
        }
    }

    // Layer 1: regional variation -- highlands, basins, plateaus within a plate.
    let regional = noise.fbm(
        x * cfg.regional_frequency,
        z * cfg.regional_frequency,
        seed ^ 0x1111,
        cfg.regional_octaves,
    ) * cfg.regional_amplitude;

    // Layer 2: local terrain, amplitude scaling with tectonic context.
    let local_amp = if tec_base < 0.0 {
        cfg.local_amplitude_ocean
    } else if tec_base < 500.0 {
        cfg.local_amplitude_plain
    } else {
        cfg.local_amplitude_mountain * (1.0 + tec.boundary_type.max(0.0))
    };
    let local = noise.fbm(
        x * cfg.local_frequency,
        z * cfg.local_frequency,
        seed ^ 0x2222,
        cfg.local_octaves,
    ) * local_amp;

    // Layer 3: mountain ridges, only in high-uplift convergent zones.
    let mountain = if tec.boundary_type > cfg.ridge_boundary_type_threshold
        && tec_base > cfg.ridge_uplift_threshold_m
    {
        let ridge_amp = tec_base * cfg.ridge_amplitude_factor;
        let boundary_factor = ((tec.boundary_type - cfg.ridge_boundary_type_threshold)
            / (1.0 - cfg.ridge_boundary_type_threshold))
            .clamp(0.0, 1.0);
        // Ridged noise: 1 - |fbm|, creates sharp peaks instead of smooth hills.
        let ridged = 1.0
            - noise
                .fbm(
                    x * cfg.ridge_frequency,
                    z * cfg.ridge_frequency,
                    seed ^ 0x3333,
                    cfg.ridge_octaves,
                )
                .abs();
        ridged * ridge_amp * boundary_factor
    } else {
        0.0
    };

    (tec_base + regional + local + mountain)
        .clamp(MIN_OCEAN_FLOOR_M as f32, MAX_TERRAIN_HEIGHT_M as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_sample(uplift_m: f32, boundary_type: f32, boundary_distance_m: f32) -> TectonicSample {
        TectonicSample {
            uplift_m,
            plate_id: 0,
            boundary_distance_m,
            boundary_type,
        }
    }

    #[test]
    fn fbm_is_deterministic() {
        let noise = NoiseState::new(42);
        let a = noise.fbm(123.4, 567.8, 0x1111, 4);
        let b = noise.fbm(123.4, 567.8, 0x1111, 4);
        assert_eq!(a, b);
    }

    #[test]
    fn fbm_seed_xor_decorrelates_layers() {
        let noise = NoiseState::new(42);
        let a = noise.fbm(100.0, 100.0, 0x1111, 4);
        let b = noise.fbm(100.0, 100.0, 0x2222, 4);
        assert_ne!(
            a, b,
            "different seed_xor should give a different noise field"
        );
    }

    #[test]
    fn fbm_stays_roughly_normalized() {
        let noise = NoiseState::new(7);
        for i in 0..50 {
            let v = noise.fbm(i as f64 * 137.0, i as f64 * 911.0, 0x2222, 6);
            assert!(
                (-1.5..=1.5).contains(&v),
                "fbm value {v} out of expected range"
            );
        }
    }

    #[test]
    fn result_always_within_valid_terrain_range() {
        let noise = NoiseState::new(99);
        let cases = [
            flat_sample(-2000.0, 0.0, 500_000.0),
            flat_sample(1500.0, 0.0, 500_000.0),
            flat_sample(9000.0, 0.9, 10_000.0),
            flat_sample(-1800.0, -0.9, 5_000.0),
            flat_sample(-1200.0, 0.9, 5_000.0),
            flat_sample(50.0, 0.0, 1_000.0), // right at the coast
        ];
        for (i, tec) in cases.iter().enumerate() {
            for j in 0..20 {
                let x = i as f64 * 10_000.0 + j as f64 * 777.0;
                let z = j as f64 * 333.0;
                let h = sample_heightmap(x, z, 99, tec, &noise, &NoiseCfg::default());
                assert!(
                    (MIN_OCEAN_FLOOR_M as f32..=MAX_TERRAIN_HEIGHT_M as f32).contains(&h),
                    "height {h} out of valid range for case {i}"
                );
            }
        }
    }

    #[test]
    fn mountain_zone_has_larger_amplitude_than_flat_zone() {
        let noise = NoiseState::new(1234);
        let mountain_tec = flat_sample(6000.0, 0.9, 5_000.0);
        let flat_tec = flat_sample(6000.0, 0.0, 500_000.0);

        // Sample several points and compare deviation from the tectonic
        // baseline -- the mountain case adds a whole extra ridged-noise
        // layer with amplitude tec_base*0.3, so its spread should dwarf
        // the flat case's (which only has the regional+local layers).
        let mut mountain_spread = 0.0f32;
        let mut flat_spread = 0.0f32;
        for i in 0..30 {
            let x = i as f64 * 401.0;
            let z = i as f64 * 233.0;
            mountain_spread +=
                (sample_heightmap(x, z, 1234, &mountain_tec, &noise, &NoiseCfg::default())
                    - mountain_tec.uplift_m)
                    .abs();
            flat_spread += (sample_heightmap(x, z, 1234, &flat_tec, &noise, &NoiseCfg::default())
                - flat_tec.uplift_m)
                .abs();
        }

        assert!(
            mountain_spread > flat_spread * 2.0,
            "mountain zone spread ({mountain_spread}) should far exceed flat zone spread ({flat_spread})"
        );
    }

    #[test]
    fn mid_ocean_ridge_raises_floor_near_divergent_boundary() {
        let noise = NoiseState::new(55);
        let ridge = flat_sample(-2500.0, -0.9, 0.0);
        let no_boundary = flat_sample(-2500.0, 0.0, 500_000.0);

        let ridge_h =
            sample_heightmap(10_000.0, 20_000.0, 55, &ridge, &noise, &NoiseCfg::default());
        let baseline_h = sample_heightmap(
            10_000.0,
            20_000.0,
            55,
            &no_boundary,
            &noise,
            &NoiseCfg::default(),
        );

        assert!(
            ridge_h > baseline_h,
            "mid-ocean ridge ({ridge_h}) should be shallower than baseline ocean floor ({baseline_h})"
        );
    }

    #[test]
    fn abyssal_trench_deepens_floor_near_convergent_oceanic_boundary() {
        let noise = NoiseState::new(55);
        let trench = flat_sample(-2000.0, 0.9, 0.0);
        let no_boundary = flat_sample(-2000.0, 0.0, 500_000.0);

        let trench_h = sample_heightmap(
            10_000.0,
            20_000.0,
            55,
            &trench,
            &noise,
            &NoiseCfg::default(),
        );
        let baseline_h = sample_heightmap(
            10_000.0,
            20_000.0,
            55,
            &no_boundary,
            &noise,
            &NoiseCfg::default(),
        );

        assert!(
            trench_h < baseline_h,
            "abyssal trench ({trench_h}) should be deeper than baseline ocean floor ({baseline_h})"
        );
    }
}
