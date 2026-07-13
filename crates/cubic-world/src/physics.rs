// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use crate::{BlockTypeId, Chunk, ChunkLocalPos, ChunkPos, CHUNK_SIZE, VOXEL_SIZE};
use cubic_math::{DVec3, Vec3};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ChunkQuery trait
// ---------------------------------------------------------------------------

pub trait ChunkQuery {
    fn get_block_at(&self, wx: f64, wy: f64, wz: f64) -> BlockTypeId;
    fn is_solid(&self, wx: f64, wy: f64, wz: f64) -> bool {
        self.get_block_at(wx, wy, wz) != BlockTypeId(0)
    }
}

/// Which chunk a world-space position falls in, and its position within
/// that chunk in voxel-local coordinates — shared by `get_block_at` (read)
/// and `AsyncWorldStream::set_block_at` (write, see stream_pool.rs) so both
/// use the exact same world->chunk mapping. f64 input so the chunk index
/// stays exact at any distance from the origin — the local voxel index
/// (0..CHUNK_SIZE) only ever needs a few bits of precision, but the chunk
/// index itself must not drift.
pub fn world_to_chunk_local(wx: f64, wy: f64, wz: f64) -> (ChunkPos, ChunkLocalPos) {
    let s = CHUNK_SIZE as f64 * VOXEL_SIZE as f64;
    let cp = ChunkPos {
        x: (wx / s).floor() as i32,
        y: (wy / s).floor() as i32,
        z: (wz / s).floor() as i32,
    };
    let lx = ((wx / VOXEL_SIZE as f64).floor() as i64).rem_euclid(CHUNK_SIZE as i64) as u8;
    let ly = ((wy / VOXEL_SIZE as f64).floor() as i64).rem_euclid(CHUNK_SIZE as i64) as u8;
    let lz = ((wz / VOXEL_SIZE as f64).floor() as i64).rem_euclid(CHUNK_SIZE as i64) as u8;
    (cp, ChunkLocalPos::new(lx, ly, lz))
}

impl ChunkQuery for HashMap<ChunkPos, Chunk> {
    fn get_block_at(&self, wx: f64, wy: f64, wz: f64) -> BlockTypeId {
        let (cp, lp) = world_to_chunk_local(wx, wy, wz);
        match self.get(&cp) {
            Some(chunk) => chunk.get(lp),
            None => BlockTypeId(0),
        }
    }
}

// ---------------------------------------------------------------------------
// AABB sweep
// ---------------------------------------------------------------------------

pub struct SweepResult {
    pub pos: DVec3,
    pub hit_x: bool,
    pub hit_y: bool,
    pub hit_z: bool,
}

/// Check whether an AABB at `pos` (center-bottom) overlaps any solid voxel.
/// `pos` is f64 (absolute world position); `hw`/`height`/`hd` stay f32 —
/// they're AABB half-extents, always small regardless of world position.
fn overlaps_solid(world: &dyn ChunkQuery, pos: DVec3, hw: f32, height: f32, hd: f32) -> bool {
    let voxel = VOXEL_SIZE as f64;
    let min_x = ((pos.x - hw as f64) / voxel).floor() as i64;
    let max_x = ((pos.x + hw as f64) / voxel).ceil() as i64;
    let min_y = (pos.y / voxel).floor() as i64;
    let max_y = ((pos.y + height as f64) / voxel).ceil() as i64;
    let min_z = ((pos.z - hd as f64) / voxel).floor() as i64;
    let max_z = ((pos.z + hd as f64) / voxel).ceil() as i64;

    for vx in min_x..max_x {
        for vy in min_y..max_y {
            for vz in min_z..max_z {
                let wx = vx as f64 * voxel + voxel * 0.5;
                let wy = vy as f64 * voxel + voxel * 0.5;
                let wz = vz as f64 * voxel + voxel * 0.5;
                if world.is_solid(wx, wy, wz) {
                    return true;
                }
            }
        }
    }
    false
}

/// Cap on de-penetration steps in `sweep_aabb`, in half-voxel increments
/// (10m of headroom) — generous for spawn-above-terrain overshoot or a
/// slope-corner nudge, but bounded so a genuinely buried AABB (e.g. deep
/// underground) can't loop forever.
const MAX_UNSTICK_STEPS: u32 = 40;

/// `origin` is f64 (absolute world position, precise at any distance from
/// the origin); `delta`/`hw`/`height`/`hd` stay f32 — one tick's movement
/// and the AABB's half-extents are always small regardless of where in the
/// world this sweep happens.
pub fn sweep_aabb(
    world: &dyn ChunkQuery,
    origin: DVec3,
    delta: Vec3,
    hw: f32,
    height: f32,
    hd: f32,
) -> SweepResult {
    let mut pos = origin;
    let mut hit_x = false;
    let mut hit_y = false;
    let mut hit_z = false;
    let step_size = VOXEL_SIZE * 0.5;

    // De-penetration: if the AABB starts this sweep already overlapping
    // solid geometry — spawned inside terrain, or nudged into a slope
    // corner by a previous tick's horizontal resolution — push it straight
    // up until clear before resolving this tick's movement. Without this, a
    // sweep that starts embedded can never escape: the loops below only
    // ever *refuse* to move further into solid ground, they never back out
    // of ground they're already in.
    for _ in 0..MAX_UNSTICK_STEPS {
        if !overlaps_solid(world, pos, hw, height, hd) {
            break;
        }
        pos.y += step_size as f64;
    }

    // Resolve Y first so gravity settles before horizontal movement
    let steps_y = (delta.y.abs() / step_size).ceil() as u32 + 1;
    let dy = delta.y / steps_y as f32;
    for _ in 0..steps_y {
        let next = DVec3::new(pos.x, pos.y + dy as f64, pos.z);
        if overlaps_solid(world, next, hw, height, hd) {
            hit_y = true;
            break;
        }
        pos.y = next.y;
    }

    // X
    let steps_x = (delta.x.abs() / step_size).ceil() as u32 + 1;
    let dx = delta.x / steps_x as f32;
    for _ in 0..steps_x {
        let next = DVec3::new(pos.x + dx as f64, pos.y, pos.z);
        if overlaps_solid(world, next, hw, height, hd) {
            hit_x = true;
            break;
        }
        pos.x = next.x;
    }

    // Z
    let steps_z = (delta.z.abs() / step_size).ceil() as u32 + 1;
    let dz = delta.z / steps_z as f32;
    for _ in 0..steps_z {
        let next = DVec3::new(pos.x, pos.y, pos.z + dz as f64);
        if overlaps_solid(world, next, hw, height, hd) {
            hit_z = true;
            break;
        }
        pos.z = next.z;
    }

    SweepResult {
        pos,
        hit_x,
        hit_y,
        hit_z,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Solid below `ground_y`, air at or above it — enough to exercise
    /// sweep_aabb's de-penetration and normal collision behavior without
    /// needing real chunk data.
    struct FlatGround {
        ground_y: f32,
    }

    impl ChunkQuery for FlatGround {
        fn get_block_at(&self, _wx: f64, wy: f64, _wz: f64) -> BlockTypeId {
            if wy < self.ground_y as f64 {
                BlockTypeId(1)
            } else {
                BlockTypeId(0)
            }
        }
    }

    const HW: f32 = 0.3;
    const HEIGHT: f32 = 1.8;
    const HD: f32 = 0.3;

    #[test]
    fn sweep_aabb_unsticks_from_embedded_start() {
        // Origin is 5m below the surface — as if spawned inside terrain, or
        // shoved into a slope corner by a previous tick's horizontal
        // resolution. With zero requested movement, a naive sweep would
        // never move at all (nothing "refuses" to move if it's not moving);
        // de-penetration must act even when delta is zero.
        let ground = FlatGround { ground_y: 10.0 };
        let origin = DVec3::new(0.0, 5.0, 0.0);
        let result = sweep_aabb(&ground, origin, Vec3::ZERO, HW, HEIGHT, HD);
        assert!(
            result.pos.y >= ground.ground_y as f64,
            "expected to be pushed clear of the ground (y >= {}), got y = {}",
            ground.ground_y,
            result.pos.y
        );
    }

    #[test]
    fn sweep_aabb_falling_lands_on_surface_without_penetrating() {
        // Falling from well above the surface should stop right at it, not
        // inside it.
        let ground = FlatGround { ground_y: 10.0 };
        let origin = DVec3::new(0.0, 20.0, 0.0);
        let result = sweep_aabb(&ground, origin, Vec3::new(0.0, -50.0, 0.0), HW, HEIGHT, HD);
        assert!(result.hit_y);
        assert!(
            result.pos.y >= ground.ground_y as f64,
            "landed pos.y = {} should not be below ground_y = {}",
            result.pos.y,
            ground.ground_y
        );
    }

    #[test]
    fn sweep_aabb_free_movement_unobstructed() {
        // No ground anywhere below the AABB — horizontal movement should
        // apply in full, matching the intent of a normal, uncollided tick.
        let ground = FlatGround {
            ground_y: f32::NEG_INFINITY,
        };
        let origin = DVec3::new(0.0, 20.0, 0.0);
        let delta = Vec3::new(1.0, 0.0, 2.0);
        let result = sweep_aabb(&ground, origin, delta, HW, HEIGHT, HD);
        assert!(!result.hit_x && !result.hit_z);
        assert!((result.pos.x - 1.0).abs() < 1e-4);
        assert!((result.pos.z - 2.0).abs() < 1e-4);
    }
}
