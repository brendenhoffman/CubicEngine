// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
pub mod mesher;
pub use mesher::mesh_chunk;
pub mod generator;
pub use generator::WorldGenerator;

use cubic_math::Vec3;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Constants — single source of truth for world scale.
// ---------------------------------------------------------------------------

/// Side length of one voxel in metres.
pub const VOXEL_SIZE: f32 = 0.5;

/// Side length of one chunk in voxels. A chunk occupies
/// CHUNK_SIZE × VOXEL_SIZE = 16 m per axis.
pub const CHUNK_SIZE: usize = 32;

/// Total number of voxels in one chunk (32³ = 32768).
pub const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

// ---------------------------------------------------------------------------
// BlockTypeId and BlockRegistry
// ---------------------------------------------------------------------------

/// Indexes into a `BlockRegistry`. Newtypes u32 so the type system prevents
/// mixing block ids with other integers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BlockTypeId(pub u32);

/// String-keyed registry of block types. `BlockTypeId(0)` is always "air"
/// and is registered automatically on construction.
pub struct BlockRegistry {
    map: HashMap<String, BlockTypeId>,
    next_id: u32,
}

impl Default for BlockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl BlockRegistry {
    /// Create a new registry. Air (id = 0) is registered immediately.
    pub fn new() -> Self {
        let mut reg = Self {
            map: HashMap::new(),
            next_id: 0,
        };
        reg.register("air"); // always BlockTypeId(0)
        reg
    }

    /// Register a block type by name and return its id. Idempotent: calling
    /// this twice with the same name returns the same id both times.
    pub fn register(&mut self, name: &str) -> BlockTypeId {
        if let Some(&id) = self.map.get(name) {
            return id;
        }
        let id = BlockTypeId(self.next_id);
        self.next_id += 1;
        self.map.insert(name.to_owned(), id);
        id
    }

    /// Look up a block type by name without registering it.
    pub fn get(&self, name: &str) -> Option<BlockTypeId> {
        self.map.get(name).copied()
    }

    /// The id for air (always 0).
    pub fn air() -> BlockTypeId {
        BlockTypeId(0)
    }
}

// ---------------------------------------------------------------------------
// ChunkLocalPos
// ---------------------------------------------------------------------------

/// A position within a single chunk, in voxel coordinates. Each component is
/// in the range `[0, CHUNK_SIZE)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkLocalPos {
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl ChunkLocalPos {
    /// Construct a position, clamping each component to `[0, CHUNK_SIZE - 1]`.
    pub fn new(x: u8, y: u8, z: u8) -> Self {
        let max = (CHUNK_SIZE - 1) as u8;
        Self {
            x: x.min(max),
            y: y.min(max),
            z: z.min(max),
        }
    }

    /// Flat array index into a CHUNK_VOLUME-length slice.
    /// Layout: x varies fastest, then z, then y — so a horizontal Y slice
    /// is contiguous in memory (good for meshing passes).
    #[inline]
    pub fn to_index(self) -> usize {
        self.x as usize + self.z as usize * CHUNK_SIZE + self.y as usize * CHUNK_SIZE * CHUNK_SIZE
    }
}

// ---------------------------------------------------------------------------
// BlockData  (per-instance data store)
// ---------------------------------------------------------------------------

/// A simple recursive value type for per-block instance data (signs, chests,
/// custom properties, etc.). Keys are always strings; values are heterogeneous.
/// Not serialised yet — that comes later when I/O is needed.
#[derive(Debug, Clone)]
pub enum BlockData {
    Int(i64),
    Float(f64),
    String(String),
    List(Vec<BlockData>),
    Map(HashMap<String, BlockData>),
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

/// One 32×32×32 block of voxel data.
///
/// Storage uses a **local palette**: `data` holds u16 indices into `palette`,
/// so the chunk always carries the minimal set of block types actually present.
/// `instance_data` is a sparse side-channel for per-voxel metadata that
/// most blocks will never need (signs, containers, etc.).
pub struct Chunk {
    /// Block types present in this chunk. Index 0 is always air.
    pub palette: Vec<BlockTypeId>,

    /// Flat CHUNK_VOLUME array of palette indices (u16 → up to 65536 distinct
    /// block types per chunk, far more than any real workload). Boxed to avoid
    /// a 64 KiB stack allocation.
    pub data: Box<[u16; CHUNK_VOLUME]>,

    /// Sparse per-voxel data for blocks that need it (e.g. sign text,
    /// inventory contents). Absent entries mean "no extra data".
    pub instance_data: HashMap<ChunkLocalPos, BlockData>,
}

impl Default for Chunk {
    fn default() -> Self {
        Self::new()
    }
}

impl Chunk {
    /// Create an empty chunk filled with air (palette index 0).
    pub fn new() -> Self {
        Self {
            palette: vec![BlockTypeId(0)], // slot 0 = air
            data: Box::new([0u16; CHUNK_VOLUME]),
            instance_data: HashMap::new(),
        }
    }

    /// Read the block type at `pos`.
    #[inline]
    pub fn get(&self, pos: ChunkLocalPos) -> BlockTypeId {
        self.palette[self.data[pos.to_index()] as usize]
    }

    /// Write the block type at `pos`. If `id` is not yet in the palette it is
    /// appended; the palette index (up to u16::MAX) is stored in `data`.
    pub fn set(&mut self, pos: ChunkLocalPos, id: BlockTypeId) {
        let palette_idx = self
            .palette
            .iter()
            .position(|&p| p == id)
            .unwrap_or_else(|| {
                self.palette.push(id);
                self.palette.len() - 1
            });
        self.data[pos.to_index()] = palette_idx as u16;
    }
}

/// Signed world-space chunk coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkPos {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl ChunkPos {
    /// World-space position of this chunk's (0,0,0) corner in metres.
    pub fn to_world_origin(self) -> Vec3 {
        let s = CHUNK_SIZE as f32 * VOXEL_SIZE;
        Vec3::new(self.x as f32 * s, self.y as f32 * s, self.z as f32 * s)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_air_is_zero() {
        let reg = BlockRegistry::new();
        assert_eq!(reg.get("air"), Some(BlockTypeId(0)));
    }

    #[test]
    fn registry_idempotent() {
        let mut reg = BlockRegistry::new();
        let a = reg.register("stone");
        let b = reg.register("stone");
        assert_eq!(a, b);
    }

    #[test]
    fn chunk_defaults_to_air() {
        let chunk = Chunk::new();
        let pos = ChunkLocalPos::new(5, 10, 15);
        assert_eq!(chunk.get(pos), BlockTypeId(0));
    }

    #[test]
    fn chunk_set_get_roundtrip() {
        let mut chunk = Chunk::new();
        let stone = BlockTypeId(1);
        let pos = ChunkLocalPos::new(1, 2, 3);
        chunk.set(pos, stone);
        assert_eq!(chunk.get(pos), stone);
        // Surrounding voxels still air.
        assert_eq!(chunk.get(ChunkLocalPos::new(0, 0, 0)), BlockTypeId(0));
    }

    #[test]
    fn chunk_palette_expands() {
        let mut chunk = Chunk::new();
        chunk.set(ChunkLocalPos::new(0, 0, 0), BlockTypeId(1));
        chunk.set(ChunkLocalPos::new(1, 0, 0), BlockTypeId(2));
        assert_eq!(chunk.palette.len(), 3); // air + 2 new types
    }

    #[test]
    fn local_pos_clamped() {
        let pos = ChunkLocalPos::new(255, 255, 255);
        assert_eq!(pos.x, (CHUNK_SIZE - 1) as u8);
        assert_eq!(pos.to_index(), CHUNK_VOLUME - 1);
    }

    #[test]
    fn local_pos_index_roundtrip() {
        // All corners map to distinct indices.
        let a = ChunkLocalPos::new(0, 0, 0).to_index();
        let b = ChunkLocalPos::new(31, 0, 0).to_index();
        let c = ChunkLocalPos::new(0, 31, 0).to_index();
        let d = ChunkLocalPos::new(0, 0, 31).to_index();
        let e = ChunkLocalPos::new(31, 31, 31).to_index();
        let indices = [a, b, c, d, e];
        let unique: std::collections::HashSet<_> = indices.iter().collect();
        assert_eq!(unique.len(), 5);
        assert_eq!(e, CHUNK_VOLUME - 1);
    }
}
