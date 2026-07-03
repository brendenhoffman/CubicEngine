// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use cubic_world::{
    BlockTypeId, Chunk, ChunkLocalPos, ChunkPos, WorldGenerator, CHUNK_SIZE, VOXEL_SIZE,
};
use noise::{NoiseFn, OpenSimplex};

pub struct NoiseLayer {
    pub frequency: f32,
    pub amplitude: f32,
}

pub struct NoiseGenerator {
    pub _sea_level: f32,
    pub base_height: f32,
    pub layers: Vec<NoiseLayer>,
    pub stone: BlockTypeId,
    noise: OpenSimplex,
}

impl NoiseGenerator {
    fn surface_height(&self, world_x: f32, world_z: f32) -> f32 {
        let mut h = self.base_height;
        for layer in &self.layers {
            h += self.noise.get([
                // use self.noise, not a new instance
                (world_x * layer.frequency) as f64,
                (world_z * layer.frequency) as f64,
            ]) as f32
                * layer.amplitude;
        }
        h
    }

    fn max_possible_height(&self) -> f32 {
        self.base_height + self.layers.iter().map(|l| l.amplitude).sum::<f32>()
    }

    pub fn new(
        sea_level: f32,
        base_height: f32,
        layers: Vec<NoiseLayer>,
        stone: BlockTypeId,
        seed: u64,
    ) -> Self {
        Self {
            _sea_level: sea_level,
            base_height,
            layers,
            stone,
            noise: OpenSimplex::new(seed as u32),
        }
    }
}

impl WorldGenerator for NoiseGenerator {
    fn generate(&self, pos: ChunkPos, _seed: u64) -> Chunk {
        let _chunk_world_size = CHUNK_SIZE as f32 * VOXEL_SIZE;
        let origin = pos.to_world_origin();
        let mut chunk = Chunk::new();

        for x in 0..CHUNK_SIZE as u8 {
            for z in 0..CHUNK_SIZE as u8 {
                let world_x = origin.x + x as f32 * VOXEL_SIZE;
                let world_z = origin.z + z as f32 * VOXEL_SIZE;
                let surface = self.surface_height(world_x, world_z);

                for y in 0..CHUNK_SIZE as u8 {
                    let world_y = origin.y + y as f32 * VOXEL_SIZE;
                    if world_y < surface {
                        chunk.set(ChunkLocalPos::new(x, y, z), self.stone);
                    }
                }
            }
        }
        chunk
    }

    fn is_definitely_air(&self, pos: ChunkPos) -> bool {
        let chunk_y_min = pos.y as f32 * CHUNK_SIZE as f32 * VOXEL_SIZE;
        let chunk_y_max = chunk_y_min + CHUNK_SIZE as f32 * VOXEL_SIZE;
        chunk_y_min > self.max_possible_height() || chunk_y_max <= 0.0
    }
}
