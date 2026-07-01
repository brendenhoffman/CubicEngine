// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Minimal flat-plane test harness for the world system: 8×8 chunks of
//! stone (bottom half) and air (top half), ready to mesh and upload.
//! No game logic, no saving, no interaction — just geometry.

use cubic_render::Vertex;
use cubic_world::{mesh_chunk, BlockRegistry, Chunk, ChunkLocalPos, CHUNK_SIZE, VOXEL_SIZE};

/// Width and depth of the test grid in chunks.
pub const GRID_W: usize = 64;
pub const GRID_D: usize = 64;

/// A flat 8×8 grid of chunks with stone filling the bottom half of each.
pub struct FlatWorld {
    /// chunks[cx][cz]
    chunks: Vec<Vec<Chunk>>,
}

impl FlatWorld {
    /// Build the world: all chunks filled with stone below y = CHUNK_SIZE/2,
    /// air above. Uses a local registry; "air" is id 0 and "stone" is id 1.
    pub fn new() -> Self {
        let mut registry = BlockRegistry::new();
        let stone = registry.register("stone");
        let fill_y = CHUNK_SIZE / 2; // bottom 16 voxels

        let chunks: Vec<Vec<Chunk>> = (0..GRID_W)
            .map(|_| {
                (0..GRID_D)
                    .map(|_| {
                        let mut chunk = Chunk::new();
                        for x in 0..CHUNK_SIZE as u8 {
                            for y in 0..fill_y as u8 {
                                for z in 0..CHUNK_SIZE as u8 {
                                    chunk.set(ChunkLocalPos::new(x, y, z), stone);
                                }
                            }
                        }
                        chunk
                    })
                    .collect()
            })
            .collect();

        Self { chunks }
    }

    /// Mesh a single chunk, providing correct neighbor references so
    /// cross-boundary faces are culled rather than doubled up.
    /// Returns `None` if the chunk produces no geometry (shouldn't happen
    /// for stone chunks but handles the empty case cleanly).
    pub fn mesh(&self, cx: usize, cz: usize) -> Option<(Vec<Vertex>, Vec<u32>)> {
        let chunk = &self.chunks[cx][cz];
        let neighbors = [
            if cx > 0 {
                Some(&self.chunks[cx - 1][cz])
            } else {
                None
            }, // −X
            if cx + 1 < GRID_W {
                Some(&self.chunks[cx + 1][cz])
            } else {
                None
            }, // +X
            None, // −Y (no chunk below)
            None, // +Y (no chunk above)
            if cz > 0 {
                Some(&self.chunks[cx][cz - 1])
            } else {
                None
            }, // −Z
            if cz + 1 < GRID_D {
                Some(&self.chunks[cx][cz + 1])
            } else {
                None
            }, // +Z
        ];
        let (verts, idxs) = mesh_chunk(chunk, neighbors);
        if verts.is_empty() {
            None
        } else {
            Some((verts, idxs))
        }
    }

    /// Iterate over all (cx, cz) chunk positions in the grid.
    pub fn positions(&self) -> impl Iterator<Item = (usize, usize)> {
        (0..GRID_W).flat_map(|cx| (0..GRID_D).map(move |cz| (cx, cz)))
    }
}

/// Column-major 4×4 translation matrix positioning chunk (cx, cz) in world
/// space. Each chunk is `CHUNK_SIZE * VOXEL_SIZE` metres wide.
pub fn chunk_model(cx: usize, cz: usize) -> [[f32; 4]; 4] {
    let tx = cx as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE;
    let tz = cz as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE;
    // Column-major: columns are stored as rows in the [[f32;4];4].
    [
        [1.0, 0.0, 0.0, 0.0], // col 0
        [0.0, 1.0, 0.0, 0.0], // col 1
        [0.0, 0.0, 1.0, 0.0], // col 2
        [tx, 0.0, tz, 1.0],   // col 3 — translation
    ]
}

/// Y-coordinate (in metres) of the voxel surface the camera should start above.
pub const SURFACE_Y: f32 = (CHUNK_SIZE / 2) as f32 * VOXEL_SIZE;
