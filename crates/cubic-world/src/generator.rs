// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use crate::{Chunk, ChunkPos};

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
    /// Returns true if this chunk is guaranteed to contain only air.
    /// Used by the streaming system to skip generation entirely.
    /// Default impl returns false — conservative, always generates.
    /// Override with a cheap height check for significant performance gains.
    fn is_definitely_air(&self, _pos: ChunkPos) -> bool {
        false
    }
}
