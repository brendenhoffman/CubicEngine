// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use cubic_world::{BlockTypeId, Chunk, ChunkLocalPos, ChunkPos, WorldGenerator, CHUNK_SIZE};

pub struct FlatGenerator {
    pub stone: BlockTypeId,
    pub fill_y: usize,
}

impl FlatGenerator {
    pub fn new() -> Self {
        Self {
            stone: BlockTypeId(1),
            fill_y: CHUNK_SIZE / 2,
        }
    }
}

impl WorldGenerator for FlatGenerator {
    fn generate(&self, pos: ChunkPos, _seed: u64) -> Chunk {
        if pos.y != 0 {
            return Chunk::new();
        }
        let mut chunk = Chunk::new();
        for x in 0..CHUNK_SIZE as u8 {
            for y in 0..self.fill_y as u8 {
                for z in 0..CHUNK_SIZE as u8 {
                    chunk.set(ChunkLocalPos::new(x, y, z), self.stone);
                }
            }
        }
        chunk
    }
}
