// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

//! Tectonic plate system: drives large-scale continental/oceanic structure
//! and mountain range placement. A fixed set of plates is generated once
//! from the world seed and reused for the lifetime of the session; each
//! column is evaluated against its two nearest plates to derive uplift or
//! subsidence and, at plate boundaries, mountain-building potential.

use crate::terrain_config::{UpliftCfg, WorldCfg};
use crate::world_constants::{NS_TILE_M, world_z_to_latitude};

/// Lloyd relaxation passes applied to plate centers so they spread out
/// roughly evenly instead of clumping, as pure random placement would.
const LLOYD_ITERATIONS: usize = 4;

/// Random sample points drawn per plate per relaxation pass -- a coarse
/// Monte-Carlo approximation of each Voronoi cell's centroid. Exact
/// centroid computation isn't worth it for only 28 sites.
const RELAXATION_SAMPLES_PER_PLATE: usize = 400;

/// Plate speed range in m/My (meters per million years -- an arbitrary
/// simulation unit, not real-time).
const PLATE_SPEED_MIN: f64 = 5.0;
const PLATE_SPEED_MAX: f64 = 50.0;

pub struct Plate {
    /// Voronoi center in world XZ meters.
    pub center: [f64; 2],
    /// Velocity vector in m/My.
    pub velocity: [f64; 2],
    /// Density: low = continental crust (floats high), high = oceanic (sinks).
    pub density: f32,
    pub id: u32,
}

pub struct TectonicPlates {
    pub plates: Vec<Plate>,
    pub(crate) uplift: UpliftCfg,
}

pub struct TectonicSample {
    /// Uplift in meters above sea level baseline. Negative = subsidence.
    pub uplift_m: f32,
    /// The plate this column sits on.
    pub plate_id: u32,
    /// Distance to nearest plate boundary in meters.
    pub boundary_distance_m: f32,
    /// Boundary type: convergent uplift factor (-1.0 to +1.0).
    /// +1.0 = head-on collision (max mountains), 0.0 = transform, -1.0 =
    /// divergent (rift).
    pub boundary_type: f32,
}

// ---------------------------------------------------------------------------
// Seeded RNG — splitmix64. Deterministic, fast, good enough statistical
// quality for plate placement (not used for anything cryptographic).
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0.0, 1.0).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

// ---------------------------------------------------------------------------
// Plate generation
// ---------------------------------------------------------------------------

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dz = a[1] - b[1];
    (dx * dx + dz * dz).sqrt()
}

fn nearest_center(centers: &[[f64; 2]], p: [f64; 2]) -> usize {
    centers
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| dist(**a, p).partial_cmp(&dist(**b, p)).unwrap())
        .map(|(i, _)| i)
        .expect("centers is never empty")
}

impl TectonicPlates {
    pub fn new(seed: u64, world_cfg: WorldCfg, uplift_cfg: UpliftCfg) -> Self {
        let mut rng = Rng(seed ^ 0xFEED_5EED_FEED_5EED);
        let plate_count = world_cfg.plate_count;

        // Plate centers are distributed across a square world tile;
        // Voronoi assignment naturally extends the outermost plates'
        // cells to infinity for columns queried beyond it.
        let half = NS_TILE_M * 0.5;
        let mut centers: Vec<[f64; 2]> = (0..plate_count)
            .map(|_| [rng.range(-half, half), rng.range(-half, half)])
            .collect();

        for _ in 0..LLOYD_ITERATIONS {
            let mut sums = vec![[0.0_f64; 2]; plate_count];
            let mut counts = vec![0u32; plate_count];
            for _ in 0..(plate_count * RELAXATION_SAMPLES_PER_PLATE) {
                let sample = [rng.range(-half, half), rng.range(-half, half)];
                let nearest = nearest_center(&centers, sample);
                sums[nearest][0] += sample[0];
                sums[nearest][1] += sample[1];
                counts[nearest] += 1;
            }
            for i in 0..plate_count {
                if counts[i] > 0 {
                    centers[i] = [sums[i][0] / counts[i] as f64, sums[i][1] / counts[i] as f64];
                }
            }
        }

        let continental_count =
            (plate_count as f64 * world_cfg.continental_fraction).round() as usize;
        let plates = centers
            .into_iter()
            .enumerate()
            .map(|(i, center)| {
                let speed = rng.range(PLATE_SPEED_MIN, PLATE_SPEED_MAX);
                let angle = rng.range(0.0, std::f64::consts::TAU);
                let velocity = [speed * angle.cos(), speed * angle.sin()];
                let density = if i < continental_count {
                    rng.range(0.2, 0.4) as f32
                } else {
                    rng.range(0.6, 0.9) as f32
                };
                Plate {
                    center,
                    velocity,
                    density,
                    id: i as u32,
                }
            })
            .collect();

        Self {
            plates,
            uplift: uplift_cfg,
        }
    }

    pub fn sample(&self, x: f64, z: f64) -> TectonicSample {
        let p = [x, z];

        // Nearest and second-nearest plate by distance to center.
        let mut own_idx = 0;
        let mut own_dist = f64::MAX;
        let mut neighbor_idx = 0;
        let mut neighbor_dist = f64::MAX;
        for (i, plate) in self.plates.iter().enumerate() {
            let d = dist(plate.center, p);
            if d < own_dist {
                neighbor_idx = own_idx;
                neighbor_dist = own_dist;
                own_idx = i;
                own_dist = d;
            } else if d < neighbor_dist {
                neighbor_idx = i;
                neighbor_dist = d;
            }
        }

        let own = &self.plates[own_idx];
        let neighbor = &self.plates[neighbor_idx];

        // Distance to the Voronoi boundary between the two nearest sites:
        // for point sites the bisector sits halfway along the distance gap.
        let boundary_distance_m = ((neighbor_dist - own_dist) * 0.5) as f32;

        // Boundary normal: direction from own plate's center toward the
        // neighbor's, i.e. pointing across the boundary.
        let dir = [
            neighbor.center[0] - own.center[0],
            neighbor.center[1] - own.center[1],
        ];
        let dir_len = dist([0.0, 0.0], dir).max(1e-9);
        let normal = [dir[0] / dir_len, dir[1] / dir_len];

        let rel_vel = [
            own.velocity[0] - neighbor.velocity[0],
            own.velocity[1] - neighbor.velocity[1],
        ];
        let rel_speed = dist([0.0, 0.0], rel_vel).max(1e-9);
        let boundary_type = ((rel_vel[0] * normal[0] + rel_vel[1] * normal[1]) / rel_speed) as f32;

        let base_uplift = if own.density < 0.5 {
            // Continental: sits above sea level.
            self.uplift.continental_base_m
                + (0.5 - own.density) * self.uplift.continental_density_scale
        } else {
            // Oceanic: sits below sea level.
            self.uplift.oceanic_base_m - (own.density - 0.5) * self.uplift.oceanic_density_scale
        };

        let boundary_influence =
            (-boundary_distance_m as f64 / self.uplift.boundary_falloff_m).exp() as f32;
        let convergent_uplift =
            boundary_type.max(0.0) * boundary_influence * self.uplift.max_convergent_uplift_m;
        let divergent_subsidence =
            boundary_type.min(0.0) * boundary_influence * self.uplift.max_divergent_subsidence_m;

        let uplift_m = base_uplift + convergent_uplift + divergent_subsidence;

        TectonicSample {
            uplift_m,
            plate_id: own.id,
            boundary_distance_m,
            boundary_type,
        }
    }
}

/// Deterministic linear congruential step for the spawn search below --
/// fast and seeded, no need for cryptographic quality, just decorrelated
/// candidates.
fn lcg_next(x: u64) -> u64 {
    x.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

/// Returns a random XZ coordinate biased toward mid-latitude continental
/// areas (normalized latitude 0.15-0.65, on a continental plate). Used to
/// place the player on first world creation.
pub fn random_spawn(plates: &TectonicPlates, seed: u64) -> [f64; 2] {
    let mut rng = seed ^ 0xC0FF_EEC0_FFEE_0000;
    for _ in 0..1000 {
        rng = lcg_next(rng);
        let x = (rng as f64 / u64::MAX as f64 - 0.5) * NS_TILE_M;
        rng = lcg_next(rng);
        let z = (rng as f64 / u64::MAX as f64 - 0.5) * NS_TILE_M * 0.5;
        let sample = plates.sample(x, z);
        let lat = world_z_to_latitude(z).abs();
        if sample.uplift_m > 0.0 && lat > 0.15 && lat < 0.65 {
            return [x, z];
        }
    }
    [0.0, 0.0] // fallback: equator/prime meridian
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_requested_plate_count() {
        let plates = TectonicPlates::new(42, WorldCfg::default(), UpliftCfg::default());
        assert_eq!(plates.plates.len(), WorldCfg::default().plate_count);
    }

    #[test]
    fn plate_ids_are_unique_and_sequential() {
        let plates = TectonicPlates::new(7, WorldCfg::default(), UpliftCfg::default());
        for (i, plate) in plates.plates.iter().enumerate() {
            assert_eq!(plate.id, i as u32);
        }
    }

    #[test]
    fn continental_and_oceanic_split_matches_target_fraction() {
        let world_cfg = WorldCfg::default();
        let plates = TectonicPlates::new(1234, world_cfg, UpliftCfg::default());
        let continental = plates.plates.iter().filter(|p| p.density < 0.5).count();
        let expected =
            (world_cfg.plate_count as f64 * world_cfg.continental_fraction).round() as usize;
        assert_eq!(continental, expected);
    }

    #[test]
    fn density_ranges_are_respected() {
        let plates = TectonicPlates::new(99, WorldCfg::default(), UpliftCfg::default());
        for plate in &plates.plates {
            assert!(
                (0.2..=0.9).contains(&plate.density) && !(0.4..0.6).contains(&plate.density),
                "density {} outside expected continental/oceanic bands",
                plate.density
            );
        }
    }

    /// Two hand-built plates far enough apart that sampling near their own
    /// centers isolates each one's baseline uplift from boundary effects.
    fn two_plate_world(continental_density: f32, oceanic_density: f32) -> TectonicPlates {
        TectonicPlates {
            plates: vec![
                Plate {
                    center: [-1_000_000.0, 0.0],
                    velocity: [10.0, 0.0],
                    density: continental_density,
                    id: 0,
                },
                Plate {
                    center: [1_000_000.0, 0.0],
                    velocity: [-10.0, 0.0],
                    density: oceanic_density,
                    id: 1,
                },
            ],
            uplift: UpliftCfg::default(),
        }
    }

    #[test]
    fn continental_plate_center_sits_above_sea_level() {
        let plates = two_plate_world(0.3, 0.8);
        let sample = plates.sample(-1_000_000.0, 0.0);
        assert_eq!(sample.plate_id, 0);
        assert!(
            sample.uplift_m > 0.0,
            "continental plate center should be above sea level, got {}",
            sample.uplift_m
        );
    }

    #[test]
    fn oceanic_plate_center_sits_below_sea_level() {
        let plates = two_plate_world(0.3, 0.8);
        let sample = plates.sample(1_000_000.0, 0.0);
        assert_eq!(sample.plate_id, 1);
        assert!(
            sample.uplift_m < 0.0,
            "oceanic plate center should be below sea level, got {}",
            sample.uplift_m
        );
    }

    #[test]
    fn head_on_convergent_boundary_uplifts_and_decays_with_distance() {
        // Plates moving directly toward each other along the X axis: at
        // their shared boundary (x=0) this is a head-on collision
        // (boundary_type ~= 1.0), and uplift should fall off moving away
        // from that boundary.
        let plates = two_plate_world(0.3, 0.35); // both continental-ish, isolates the boundary term
        let at_boundary = plates.sample(0.0, 0.0);
        let near_boundary = plates.sample(50_000.0, 0.0);
        let far_from_boundary = plates.sample(900_000.0, 0.0);

        assert!(
            at_boundary.boundary_type > 0.9,
            "expected near-head-on convergence, got {}",
            at_boundary.boundary_type
        );
        assert!(
            at_boundary.uplift_m > near_boundary.uplift_m,
            "uplift should peak at the boundary: {} vs {}",
            at_boundary.uplift_m,
            near_boundary.uplift_m
        );
        assert!(
            near_boundary.uplift_m > far_from_boundary.uplift_m,
            "uplift should fall off away from the boundary: {} vs {}",
            near_boundary.uplift_m,
            far_from_boundary.uplift_m
        );
    }

    #[test]
    fn divergent_boundary_subsides() {
        // Plates moving directly away from each other.
        let plates = TectonicPlates {
            plates: vec![
                Plate {
                    center: [-1_000_000.0, 0.0],
                    velocity: [-10.0, 0.0],
                    density: 0.3,
                    id: 0,
                },
                Plate {
                    center: [1_000_000.0, 0.0],
                    velocity: [10.0, 0.0],
                    density: 0.3,
                    id: 1,
                },
            ],
            uplift: UpliftCfg::default(),
        };
        let at_boundary = plates.sample(0.0, 0.0);
        assert!(
            at_boundary.boundary_type < -0.9,
            "expected near-divergent boundary, got {}",
            at_boundary.boundary_type
        );
        // Base continental uplift alone (no boundary) would be positive;
        // divergent subsidence should be able to pull it down.
        let no_boundary_baseline = 500.0 + (0.5 - 0.3) * 4000.0;
        assert!(
            (at_boundary.uplift_m as f64) < no_boundary_baseline,
            "divergent boundary should subside relative to baseline: {} vs {}",
            at_boundary.uplift_m,
            no_boundary_baseline
        );
    }

    #[test]
    fn random_spawn_lands_on_mid_latitude_continental_ground() {
        let plates = TectonicPlates::new(2026, WorldCfg::default(), UpliftCfg::default());
        let [x, z] = random_spawn(&plates, 2026);
        let sample = plates.sample(x, z);
        let lat = world_z_to_latitude(z).abs();
        assert!(sample.uplift_m > 0.0, "spawn should be on land");
        assert!(
            (0.15..0.65).contains(&lat),
            "spawn latitude {lat} outside mid-latitude band"
        );
    }
}
