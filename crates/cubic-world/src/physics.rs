// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use crate::{BlockTypeId, Chunk, ChunkLocalPos, ChunkPos, CHUNK_SIZE, VOXEL_SIZE};
use cubic_math::Vec3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// ChunkQuery trait
// ---------------------------------------------------------------------------

pub trait ChunkQuery {
    fn get_block_at(&self, wx: f32, wy: f32, wz: f32) -> BlockTypeId;
    fn is_solid(&self, wx: f32, wy: f32, wz: f32) -> bool {
        self.get_block_at(wx, wy, wz) != BlockTypeId(0)
    }
}

impl ChunkQuery for HashMap<ChunkPos, Chunk> {
    fn get_block_at(&self, wx: f32, wy: f32, wz: f32) -> BlockTypeId {
        let s = CHUNK_SIZE as f32 * VOXEL_SIZE;
        let cp = ChunkPos {
            x: (wx / s).floor() as i32,
            y: (wy / s).floor() as i32,
            z: (wz / s).floor() as i32,
        };
        let chunk = match self.get(&cp) {
            Some(c) => c,
            None => return BlockTypeId(0),
        };
        let lx = ((wx / VOXEL_SIZE).floor() as i32).rem_euclid(CHUNK_SIZE as i32) as u8;
        let ly = ((wy / VOXEL_SIZE).floor() as i32).rem_euclid(CHUNK_SIZE as i32) as u8;
        let lz = ((wz / VOXEL_SIZE).floor() as i32).rem_euclid(CHUNK_SIZE as i32) as u8;
        chunk.get(ChunkLocalPos::new(lx, ly, lz))
    }
}

// ---------------------------------------------------------------------------
// AABB sweep
// ---------------------------------------------------------------------------

pub struct SweepResult {
    pub pos: Vec3,
    pub hit_x: bool,
    pub hit_y: bool,
    pub hit_z: bool,
}

/// Check whether an AABB at `pos` (center-bottom) overlaps any solid voxel.
fn overlaps_solid(world: &dyn ChunkQuery, pos: Vec3, hw: f32, height: f32, hd: f32) -> bool {
    let min_x = ((pos.x - hw) / VOXEL_SIZE).floor() as i32;
    let max_x = ((pos.x + hw) / VOXEL_SIZE).ceil() as i32;
    let min_y = (pos.y / VOXEL_SIZE).floor() as i32;
    let max_y = ((pos.y + height) / VOXEL_SIZE).ceil() as i32;
    let min_z = ((pos.z - hd) / VOXEL_SIZE).floor() as i32;
    let max_z = ((pos.z + hd) / VOXEL_SIZE).ceil() as i32;

    for vx in min_x..max_x {
        for vy in min_y..max_y {
            for vz in min_z..max_z {
                let wx = vx as f32 * VOXEL_SIZE + VOXEL_SIZE * 0.5;
                let wy = vy as f32 * VOXEL_SIZE + VOXEL_SIZE * 0.5;
                let wz = vz as f32 * VOXEL_SIZE + VOXEL_SIZE * 0.5;
                if world.is_solid(wx, wy, wz) {
                    return true;
                }
            }
        }
    }
    false
}

pub fn sweep_aabb(
    world: &dyn ChunkQuery,
    origin: Vec3,
    delta: Vec3,
    hw: f32,
    height: f32,
    hd: f32,
) -> SweepResult {
    let mut pos = origin;
    let mut hit_x = false;
    let mut hit_y = false;
    let mut hit_z = false;

    // Resolve Y first so gravity settles before horizontal movement
    let step_size = VOXEL_SIZE * 0.5;
    let steps_y = (delta.y.abs() / step_size).ceil() as u32 + 1;
    let dy = delta.y / steps_y as f32;
    for _ in 0..steps_y {
        let next = Vec3::new(pos.x, pos.y + dy, pos.z);
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
        let next = Vec3::new(pos.x + dx, pos.y, pos.z);
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
        let next = Vec3::new(pos.x, pos.y, pos.z + dz);
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
