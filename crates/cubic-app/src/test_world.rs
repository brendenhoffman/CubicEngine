// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Minimal flat-plane test harness for the world system: 8×8 chunks of
//! stone (bottom half) and air (top half), ready to mesh and upload.
//! No game logic, no saving, no interaction — just geometry.

use cubic_render::Vertex;
use cubic_world::{
    mesh_chunk, BlockRegistry, Chunk, ChunkLocalPos, ChunkPos, CHUNK_SIZE, VOXEL_SIZE,
};
use std::collections::HashMap;

/// Width and depth of the test grid in chunks.
pub const GRID_W: usize = 64;
pub const GRID_D: usize = 64;

pub struct FlatWorld {
    chunks: HashMap<ChunkPos, Chunk>,
}

impl FlatWorld {
    pub fn new() -> Self {
        let mut registry = BlockRegistry::new();
        let stone = registry.register("stone");
        let fill_y = CHUNK_SIZE / 2;
        let mut chunks = HashMap::new();

        for cx in 0..GRID_W as i32 {
            for cz in 0..GRID_D as i32 {
                let mut chunk = Chunk::new();
                for x in 0..CHUNK_SIZE as u8 {
                    for y in 0..fill_y as u8 {
                        for z in 0..CHUNK_SIZE as u8 {
                            chunk.set(ChunkLocalPos::new(x, y, z), stone);
                        }
                    }
                }
                chunks.insert(ChunkPos { x: cx, y: 0, z: cz }, chunk);
            }
        }
        Self { chunks }
    }

    pub fn mesh(&self, pos: ChunkPos) -> Option<(Vec<Vertex>, Vec<u32>)> {
        let chunk = self.chunks.get(&pos)?;
        let neighbors = [
            self.chunks
                .get(&ChunkPos {
                    x: pos.x - 1,
                    ..pos
                })
                .map(|c| c as &Chunk), // -X
            self.chunks
                .get(&ChunkPos {
                    x: pos.x + 1,
                    ..pos
                })
                .map(|c| c as &Chunk), // +X
            self.chunks
                .get(&ChunkPos {
                    y: pos.y - 1,
                    ..pos
                })
                .map(|c| c as &Chunk), // -Y
            self.chunks
                .get(&ChunkPos {
                    y: pos.y + 1,
                    ..pos
                })
                .map(|c| c as &Chunk), // +Y
            self.chunks
                .get(&ChunkPos {
                    z: pos.z - 1,
                    ..pos
                })
                .map(|c| c as &Chunk), // -Z
            self.chunks
                .get(&ChunkPos {
                    z: pos.z + 1,
                    ..pos
                })
                .map(|c| c as &Chunk), // +Z
        ];
        let (verts, idxs) = mesh_chunk(chunk, neighbors);
        if verts.is_empty() {
            None
        } else {
            Some((verts, idxs))
        }
    }

    pub fn positions(&self) -> impl Iterator<Item = ChunkPos> + '_ {
        self.chunks.keys().copied()
    }
}

pub fn chunk_model(pos: ChunkPos) -> [[f32; 4]; 4] {
    let o = pos.to_world_origin();
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [o.x, o.y, o.z, 1.0],
    ]
}

/// Y-coordinate (in metres) of the voxel surface the camera should start above.
pub const SURFACE_Y: f32 = (CHUNK_SIZE / 2) as f32 * VOXEL_SIZE;
