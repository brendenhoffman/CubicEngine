// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use crate::{Chunk, ChunkPos, WorldGenerator, CHUNK_SIZE, VOXEL_SIZE};
use cubic_math::Vec3;
use std::collections::HashMap;

/// Convert a world-space position to the chunk that contains it.
pub fn world_pos_to_chunk(world: Vec3) -> ChunkPos {
    let s = CHUNK_SIZE as f32 * VOXEL_SIZE;
    ChunkPos {
        x: (world.x / s).floor() as i32,
        y: (world.y / s).floor() as i32,
        z: (world.z / s).floor() as i32,
    }
}

/// Chunks newly loaded or unloaded in a single `WorldStream::update` call.
pub struct StreamDelta {
    pub loaded: Vec<ChunkPos>,
    pub unloaded: Vec<ChunkPos>,
}

/// Manages the set of loaded chunks around a moving camera.
/// Synchronous — generation happens on the calling thread.
/// See `AsyncWorldStream` for the threaded version.
pub struct WorldStream {
    pub chunks: HashMap<ChunkPos, Chunk>,
    pub radius_xz: i32,
    pub radius_y: i32,
}

impl WorldStream {
    pub fn new(radius_xz: i32, radius_y: i32) -> Self {
        Self {
            chunks: HashMap::new(),
            radius_xz,
            radius_y,
        }
    }

    /// Update the loaded set given the current camera chunk position.
    /// Generates newly needed chunks synchronously and unloads out-of-range
    /// ones. Returns what changed so the caller can upload/free GPU meshes.
    pub fn update(
        &mut self,
        center: ChunkPos,
        generator: &dyn WorldGenerator,
        seed: u64,
    ) -> StreamDelta {
        let mut to_load = Vec::new();
        let mut to_unload = Vec::new();

        // Compute desired set and find what needs loading
        for x in (center.x - self.radius_xz)..=(center.x + self.radius_xz) {
            for y in (center.y - self.radius_y)..=(center.y + self.radius_y) {
                for z in (center.z - self.radius_xz)..=(center.z + self.radius_xz) {
                    let pos = ChunkPos { x, y, z };
                    if !self.chunks.contains_key(&pos) {
                        to_load.push(pos);
                    }
                }
            }
        }

        // Find what needs unloading
        for pos in self.chunks.keys().copied() {
            if (pos.x - center.x).abs() > self.radius_xz
                || (pos.y - center.y).abs() > self.radius_y
                || (pos.z - center.z).abs() > self.radius_xz
            {
                to_unload.push(pos);
            }
        }

        // Apply unloads
        for pos in &to_unload {
            self.chunks.remove(pos);
        }

        // Apply loads
        for pos in &to_load {
            let chunk = generator.generate(*pos, seed);
            self.chunks.insert(*pos, chunk);
        }

        StreamDelta {
            loaded: to_load,
            unloaded: to_unload,
        }
    }

    /// Neighbor references for the mesher, in [-X,+X,-Y,+Y,-Z,+Z] order.
    pub fn neighbors(&self, pos: ChunkPos) -> [Option<&Chunk>; 6] {
        [
            self.chunks.get(&ChunkPos {
                x: pos.x - 1,
                ..pos
            }),
            self.chunks.get(&ChunkPos {
                x: pos.x + 1,
                ..pos
            }),
            self.chunks.get(&ChunkPos {
                y: pos.y - 1,
                ..pos
            }),
            self.chunks.get(&ChunkPos {
                y: pos.y + 1,
                ..pos
            }),
            self.chunks.get(&ChunkPos {
                z: pos.z - 1,
                ..pos
            }),
            self.chunks.get(&ChunkPos {
                z: pos.z + 1,
                ..pos
            }),
        ]
    }
}
