// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Greedy meshing: converts a Chunk's voxel data into a triangle mesh.
//!
//! For each of the 6 face directions the algorithm:
//! 1. Builds a 2D mask of visible faces for one layer (slice perpendicular to
//!    the current axis).
//! 2. Sweeps the mask greedily: picks the first unmarked face, extends it into
//!    the largest rectangle of identical block type, emits one quad, marks
//!    those cells consumed.
//! 3. Advances to the next layer.
//!
//! This crate imports `Vertex` from `cubic-render` rather than from
//! `cubic-render-vk`, so meshing is completely backend-agnostic.

use crate::{BlockTypeId, Chunk, ChunkLocalPos, CHUNK_SIZE, VOXEL_SIZE};
use cubic_render::Vertex;

const CS: usize = CHUNK_SIZE;
const AIR: BlockTypeId = BlockTypeId(0);

/// Greedy-merged quads of differing sizes meet at T-junctions (a vertex of
/// one quad falling mid-edge on a neighbor's edge, not a shared vertex) —
/// GPU rasterizers don't guarantee gap-free coverage across those, so at
/// certain view angles a stray pixel falls in the crack and shows whatever
/// is behind the mesh. Inflating every quad's edges outward by this much
/// (sub-millimeter, far below a texel or a screen pixel at any sane view
/// distance) turns any such crack into a coplanar micro-overlap instead —
/// invisible, but leaves no gap for the rasterizer to drop a pixel into.
/// Applied uniformly, including at chunk boundaries, where the same
/// T-junction issue can occur between two independently-meshed chunks.
const CRACK_EPS: f32 = VOXEL_SIZE * 0.001;

// ---------------------------------------------------------------------------
// BlockFaceTextures
// ---------------------------------------------------------------------------

/// Pre-built per-block face texture index table, indexed by `BlockTypeId.0`.
/// Entry `6*id+dir` gives the bindless texture array index for that face.
/// Dir order matches the mesher: 0=-X 1=+X 2=-Y 3=+Y 4=-Z 5=+Z
pub struct BlockFaceTextures {
    /// Flat array: [block0_neg_x, block0_pos_x, block0_neg_y, block0_pos_y, block0_neg_z, block0_pos_z, block1_neg_x, ...]
    data: Vec<u32>,
}

impl Default for BlockFaceTextures {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockFaceTextures {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn push(&mut self, faces: [u32; 6]) {
        self.data.extend_from_slice(&faces);
    }

    #[inline]
    pub fn get(&self, id: BlockTypeId, dir: u8) -> u32 {
        let base = id.0 as usize * 6;
        self.data.get(base + dir as usize).copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Convert a chunk's voxel data to a triangle mesh via greedy meshing.
///
/// `neighbors` provides the 6 adjacent chunks for cross-boundary face
/// visibility, in the order **[−X, +X, −Y, +Y, −Z, +Z]**. Pass `None` for
/// a missing neighbor to treat that boundary as exposed (face is always
/// generated).
///
/// `face_textures` supplies the bindless texture index for each block/face
/// combination (see `BlockFaceTextures`).
///
/// Returns `(vertices, indices)` ready to hand directly to `upload_mesh`.
pub fn mesh_chunk(
    chunk: &Chunk,
    neighbors: [Option<&Chunk>; 6],
    face_textures: &BlockFaceTextures,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts: Vec<Vertex> = Vec::new();
    let mut idxs: Vec<u32> = Vec::new();

    // Iterate all 6 face directions.
    // dir layout: 0=-X  1=+X  2=-Y  3=+Y  4=-Z  5=+Z
    for dir in 0u8..6 {
        let axis = (dir >> 1) as usize; // 0=X 1=Y 2=Z
        let positive = (dir & 1) == 1; // true = facing in positive axis direction

        // Winding flip keeps CCW front-face convention for all 6 directions.
        // Derived by checking (v1-v0)×(v2-v0) for each direction; the Y axis
        // needs the opposite flip relative to X and Z.
        let flip = ((dir & 1) == 0) ^ (axis == 1);

        let normal: [f32; 3] = {
            let mut n = [0.0f32; 3];
            n[axis] = if positive { 1.0 } else { -1.0 };
            n
        };

        let mut mask = [None::<BlockTypeId>; CS * CS];

        for layer in 0..CS {
            // ----- Build the 2D visibility mask for this layer -----
            for u in 0..CS {
                for v in 0..CS {
                    let (cx, cy, cz) = layer_uvw(axis, layer, u, v);
                    // The "other" voxel across the face boundary.
                    let (ox, oy, oz) = {
                        let (mut x, mut y, mut z) = (cx as i32, cy as i32, cz as i32);
                        match axis {
                            0 => x += if positive { 1 } else { -1 },
                            1 => y += if positive { 1 } else { -1 },
                            _ => z += if positive { 1 } else { -1 },
                        }
                        (x, y, z)
                    };

                    let cur = chunk.get(ChunkLocalPos::new(cx as u8, cy as u8, cz as u8));
                    let other = sample(chunk, &neighbors, ox, oy, oz);

                    mask[u * CS + v] = if is_opaque(cur) && !is_opaque(other) {
                        Some(cur)
                    } else {
                        None
                    };
                }
            }

            // ----- Greedy merge over the 2D mask -----
            let mut consumed = [false; CS * CS];

            for u0 in 0..CS {
                for v0 in 0..CS {
                    let i0 = u0 * CS + v0;
                    if consumed[i0] || mask[i0].is_none() {
                        continue;
                    }
                    let block = mask[i0].unwrap();

                    // Extend in the u direction.
                    let mut w = 1;
                    while u0 + w < CS {
                        let i = (u0 + w) * CS + v0;
                        if consumed[i] || mask[i] != Some(block) {
                            break;
                        }
                        w += 1;
                    }

                    // Extend in the v direction, keeping width w fixed.
                    let mut h = 1;
                    'grow_v: while v0 + h < CS {
                        for du in 0..w {
                            let i = (u0 + du) * CS + (v0 + h);
                            if consumed[i] || mask[i] != Some(block) {
                                break 'grow_v;
                            }
                        }
                        h += 1;
                    }

                    // Mark the rectangle consumed.
                    for du in 0..w {
                        for dv in 0..h {
                            consumed[(u0 + du) * CS + (v0 + dv)] = true;
                        }
                    }

                    // ----- Emit the quad -----
                    //
                    // World-space face coordinate along the meshing axis.
                    let face_coord = if positive { layer + 1 } else { layer };
                    let af = face_coord as f32 * VOXEL_SIZE;

                    let u0f = u0 as f32 * VOXEL_SIZE;
                    let v0f = v0 as f32 * VOXEL_SIZE;
                    let wf = w as f32 * VOXEL_SIZE;
                    let hf = h as f32 * VOXEL_SIZE;

                    // Four corners: BL, BR, TR, TL (in u-v space), inflated
                    // by CRACK_EPS on every edge to close greedy-mesh
                    // T-junction rasterization cracks (see CRACK_EPS's doc
                    // comment) — geometry only, UVs below are left at the
                    // true w/h so texturing is unaffected.
                    let corners = [
                        world_pos(axis, af, u0f - CRACK_EPS, v0f - CRACK_EPS), // 0 = BL
                        world_pos(axis, af, u0f + wf + CRACK_EPS, v0f - CRACK_EPS), // 1 = BR
                        world_pos(axis, af, u0f + wf + CRACK_EPS, v0f + hf + CRACK_EPS), // 2 = TR
                        world_pos(axis, af, u0f - CRACK_EPS, v0f + hf + CRACK_EPS), // 3 = TL
                    ];
                    // UVs corrected per face direction so textures appear upright on all faces.
                    // Corners are BL, BR, TR, TL in u-v space — the mapping differs per axis
                    // to compensate for how world_pos lays out the quad geometry.
                    // dir: 0=-X  1=+X  2=-Y  3=+Y  4=-Z  5=+Z
                    let uvs: [[f32; 2]; 4] = match dir {
                        0 => [
                            [0.0, w as f32],
                            [0.0, 0.0],
                            [h as f32, 0.0],
                            [h as f32, w as f32],
                        ],
                        1 => [
                            [h as f32, w as f32],
                            [h as f32, 0.0],
                            [0.0, 0.0],
                            [0.0, w as f32],
                        ],
                        2 => [
                            [0.0, h as f32],
                            [w as f32, h as f32],
                            [w as f32, 0.0],
                            [0.0, 0.0],
                        ],
                        3 => [
                            [0.0, 0.0],
                            [w as f32, 0.0],
                            [w as f32, h as f32],
                            [0.0, h as f32],
                        ],
                        4 => [
                            [w as f32, h as f32],
                            [0.0, h as f32],
                            [0.0, 0.0],
                            [w as f32, 0.0],
                        ],
                        _ => [
                            [0.0, h as f32],
                            [w as f32, h as f32],
                            [w as f32, 0.0],
                            [0.0, 0.0],
                        ],
                    };
                    let tex_index = face_textures.get(block, dir);

                    let base = verts.len() as u32;
                    for (pos, uv) in corners.iter().zip(uvs.iter()) {
                        verts.push(Vertex {
                            pos: *pos,
                            color: [1.0, 1.0, 1.0],
                            uv: *uv,
                            normal,
                            tex_index,
                        });
                    }

                    if !flip {
                        // Standard: (0,1,2) then (0,2,3)
                        idxs.extend_from_slice(&[
                            base,
                            base + 1,
                            base + 2,
                            base,
                            base + 2,
                            base + 3,
                        ]);
                    } else {
                        // Flipped: (0,3,2) then (0,2,1)
                        idxs.extend_from_slice(&[
                            base,
                            base + 3,
                            base + 2,
                            base,
                            base + 2,
                            base + 1,
                        ]);
                    }
                }
            }
        }
    }

    (verts, idxs)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Is this block type opaque (non-transparent, generates faces on its surface)?
/// For now every non-air block is opaque.
#[inline]
fn is_opaque(id: BlockTypeId) -> bool {
    id != AIR
}

/// Map (axis, layer, u, v) → (x, y, z) voxel coordinates.
/// u and v are the two axes that span the 2D slice perpendicular to `axis`.
#[inline]
fn layer_uvw(axis: usize, layer: usize, u: usize, v: usize) -> (usize, usize, usize) {
    match axis {
        0 => (layer, u, v), // X-slices: u=Y v=Z
        1 => (u, layer, v), // Y-slices: u=X v=Z
        _ => (u, v, layer), // Z-slices: u=X v=Y
    }
}

/// Map (axis, face_coord, u_world, v_world) → 3D world position.
#[inline]
fn world_pos(axis: usize, face: f32, u: f32, v: f32) -> [f32; 3] {
    match axis {
        0 => [face, u, v],
        1 => [u, face, v],
        _ => [u, v, face],
    }
}

/// Sample a voxel that may lie outside this chunk's bounds, consulting the
/// appropriate neighbor. Returns air when a neighbor is absent (treating the
/// world edge as open air, which generates boundary faces).
fn sample(chunk: &Chunk, neighbors: &[Option<&Chunk>; 6], x: i32, y: i32, z: i32) -> BlockTypeId {
    let cs = CS as i32;
    if x >= 0 && x < cs && y >= 0 && y < cs && z >= 0 && z < cs {
        return chunk.get(ChunkLocalPos::new(x as u8, y as u8, z as u8));
    }
    // One component is out of range — find which neighbor and the local pos.
    let (ni, lx, ly, lz) = if x < 0 {
        (0usize, cs - 1, y, z) // −X neighbor
    } else if x >= cs {
        (1, 0, y, z) // +X neighbor
    } else if y < 0 {
        (2, x, cs - 1, z) // −Y neighbor
    } else if y >= cs {
        (3, x, 0, z) // +Y neighbor
    } else if z < 0 {
        (4, x, y, cs - 1) // −Z neighbor
    } else {
        (5, x, y, 0) // +Z neighbor
    };
    neighbors[ni]
        .map(|n| n.get(ChunkLocalPos::new(lx as u8, ly as u8, lz as u8)))
        .unwrap_or(AIR)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BlockRegistry, CHUNK_SIZE};

    fn solid_chunk(reg: &mut BlockRegistry) -> Chunk {
        let stone = reg.register("stone");
        let mut c = Chunk::new();
        for x in 0..CHUNK_SIZE {
            for y in 0..CHUNK_SIZE {
                for z in 0..CHUNK_SIZE {
                    c.set(ChunkLocalPos::new(x as u8, y as u8, z as u8), stone);
                }
            }
        }
        c
    }

    #[test]
    fn empty_chunk_no_geometry() {
        let chunk = Chunk::new();
        let (v, i) = mesh_chunk(&chunk, [None; 6], &BlockFaceTextures::new());
        assert!(v.is_empty(), "empty chunk should produce no vertices");
        assert!(i.is_empty());
    }

    #[test]
    fn single_voxel_six_faces() {
        let mut reg = BlockRegistry::new();
        let stone = reg.register("stone");
        let mut chunk = Chunk::new();
        chunk.set(ChunkLocalPos::new(1, 1, 1), stone);
        let (v, i) = mesh_chunk(&chunk, [None; 6], &BlockFaceTextures::new());
        // 6 faces × 4 vertices = 24 verts, 6 faces × 6 indices = 36 indices
        assert_eq!(v.len(), 24, "single voxel: 6 faces × 4 verts");
        assert_eq!(i.len(), 36, "single voxel: 6 faces × 6 indices");
    }

    #[test]
    fn solid_chunk_with_solid_neighbors_no_geometry() {
        let mut reg = BlockRegistry::new();
        let c = solid_chunk(&mut reg);
        let (v, _) = mesh_chunk(&c, [Some(&c); 6], &BlockFaceTextures::new());
        assert!(v.is_empty(), "fully-buried solid chunk: no visible faces");
    }

    #[test]
    fn solid_chunk_no_neighbors_only_boundary_faces() {
        let mut reg = BlockRegistry::new();
        let c = solid_chunk(&mut reg);
        let (v, _) = mesh_chunk(&c, [None; 6], &BlockFaceTextures::new());
        // 6 faces, each a single CHUNK_SIZE×CHUNK_SIZE greedy quad → 4 verts each
        assert_eq!(
            v.len(),
            6 * 4,
            "solid chunk with no neighbors: 6 boundary quads"
        );
    }

    #[test]
    fn normals_point_outward() {
        let mut reg = BlockRegistry::new();
        let stone = reg.register("stone");
        let mut chunk = Chunk::new();
        // Put one voxel at the centre; verify all 6 normals are unit vectors
        // pointing along a single axis.
        chunk.set(ChunkLocalPos::new(0, 0, 0), stone);
        let (verts, _) = mesh_chunk(&chunk, [None; 6], &BlockFaceTextures::new());
        let unique_normals: std::collections::HashSet<[i32; 3]> = verts
            .iter()
            .map(|v| [v.normal[0] as i32, v.normal[1] as i32, v.normal[2] as i32])
            .collect();
        // Exactly 6 distinct normals: ±X ±Y ±Z
        assert_eq!(unique_normals.len(), 6);
    }

    #[test]
    fn greedy_merges_row() {
        // A 4-voxel horizontal row along X should produce one merged top quad
        // of width 4 for the +Y face.
        let mut reg = BlockRegistry::new();
        let stone = reg.register("stone");
        let mut chunk = Chunk::new();
        for x in 0..4u8 {
            chunk.set(ChunkLocalPos::new(x, 0, 0), stone);
        }
        let (verts, idxs) = mesh_chunk(&chunk, [None; 6], &BlockFaceTextures::new());
        // The +Y face should be one quad (4 verts, 6 indices) — the four voxels
        // are adjacent, same type, same slice → greedy merges them.
        // Count how many quads have a +Y normal.
        let plus_y_verts: Vec<_> = verts
            .iter()
            .filter(|v| v.normal == [0.0, 1.0, 0.0])
            .collect();
        // 4 merged into 1 quad = 4 verts for the +Y face.
        assert_eq!(plus_y_verts.len(), 4, "+Y face should be one merged quad");
        let _ = idxs; // silence unused warning
    }
}
