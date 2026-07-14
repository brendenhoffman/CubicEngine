// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use crate::{Chunk, ChunkPos, CHUNK_SIZE, VOXEL_SIZE};

/// Contract for procedural chunk generation. Implementations live in game
/// crates, not here — the engine only defines the interface.
///
/// # Requirements for implementors
/// - **Pure and deterministic**: identical inputs must produce identical chunks.
///   The streaming system may call this multiple times for the same position.
/// - **`Send + Sync`**: implementations are shared across worker threads via
///   `Arc<dyn WorldGenerator>`.
/// - **No registry access**: the generator owns whatever block type mappings
///   it needs internally. Do not take a `&BlockRegistry` parameter.
pub trait WorldGenerator: Send + Sync {
    fn generate(&self, pos: ChunkPos, seed: u64) -> Chunk;

    /// Return the approximate terrain surface height in meters at world
    /// coordinate (x, z). Used by LOD systems, air skipping, and the far
    /// heightmap renderer. Does not need to match `generate` exactly --
    /// it's a fast approximation. Default returns 0.0 (sea level).
    fn sample_height(&self, _x: f64, _z: f64, _seed: u64) -> f32 {
        0.0
    }

    /// Returns true if this chunk is guaranteed to contain only air.
    /// Used by the streaming system to skip generation entirely.
    /// Default impl samples the surface height at the chunk's XZ center and
    /// checks whether the chunk's Y range sits entirely above it (with a
    /// one-chunk margin to avoid false positives near the surface).
    /// Override with a cheaper heuristic if `sample_height` is expensive.
    fn is_definitely_air(&self, pos: ChunkPos, seed: u64) -> bool {
        let chunk_m = CHUNK_SIZE as f64 * VOXEL_SIZE as f64;
        let cx = pos.x as f64 * chunk_m + chunk_m * 0.5;
        let cz = pos.z as f64 * chunk_m + chunk_m * 0.5;
        let chunk_min_y = pos.y as f64 * chunk_m;
        let surface = self.sample_height(cx, cz, seed) as f64;
        chunk_min_y > surface + chunk_m
    }
}
