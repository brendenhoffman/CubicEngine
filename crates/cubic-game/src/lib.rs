// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

wit_bindgen::generate!({
    world: "game",
    path: "../cubic-wasm/wit/game.wit",
});

use exports::cubic::game::world_gen::Guest;
use noise::{NoiseFn, OpenSimplex};
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Noise config
// ---------------------------------------------------------------------------

struct NoiseLayer {
    frequency: f32,
    amplitude: f32,
}

struct NoiseGenerator {
    base_height: f32,
    layers: Vec<NoiseLayer>,
    noise: OpenSimplex,
    stone_id: u32,
}

impl NoiseGenerator {
    fn surface_height(&self, world_x: f32, world_z: f32) -> f32 {
        let mut h = self.base_height;
        for layer in &self.layers {
            h += self.noise.get([
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
}

// ---------------------------------------------------------------------------
// Global state — safe because WASM is single-threaded per instance
// ---------------------------------------------------------------------------

static GENERATOR: OnceLock<NoiseGenerator> = OnceLock::new();

fn generator() -> &'static NoiseGenerator {
    GENERATOR
        .get()
        .expect("generator not initialized — on_load not called")
}

// ---------------------------------------------------------------------------
// Constants — must match cubic-world
// ---------------------------------------------------------------------------

const CHUNK_SIZE: usize = 32;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const VOXEL_SIZE: f32 = 0.5;

// ---------------------------------------------------------------------------
// WIT guest implementation
// ---------------------------------------------------------------------------

struct GamePlugin;

impl Guest for GamePlugin {
    fn on_load(seed: u64) -> u32 {
        let stone_id = cubic::game::block_registry::register_block("stone");
        GENERATOR
            .set(NoiseGenerator {
                base_height: 16.0,
                layers: vec![
                    NoiseLayer {
                        frequency: 0.01,
                        amplitude: 16.0,
                    },
                    NoiseLayer {
                        frequency: 0.05,
                        amplitude: 4.0,
                    },
                    NoiseLayer {
                        frequency: 0.1,
                        amplitude: 1.0,
                    },
                ],
                noise: OpenSimplex::new(seed as u32),
                stone_id,
            })
            .unwrap_or_else(|_| panic!("on_load called twice"));
        0
    }

    fn generate(_handle: u32, cx: i32, cy: i32, cz: i32, out_ptr: u32) -> u32 {
        let noise_gen = generator();

        let origin_x = (cx as f32) * (CHUNK_SIZE as f32) * VOXEL_SIZE;
        let origin_y = (cy as f32) * (CHUNK_SIZE as f32) * VOXEL_SIZE;
        let origin_z = (cz as f32) * (CHUNK_SIZE as f32) * VOXEL_SIZE;

        // Write voxels into shared memory at out_ptr.
        // Layout: index = x + z*CHUNK_SIZE + y*CHUNK_SIZE*CHUNK_SIZE
        // Safety: out_ptr is a valid offset into WASM linear memory,
        // pre-allocated by the host for this worker's output buffer.
        let mem_ptr = out_ptr as *mut u32;
        for y in 0..CHUNK_SIZE {
            let world_y = origin_y + y as f32 * VOXEL_SIZE;
            for z in 0..CHUNK_SIZE {
                let world_z = origin_z + z as f32 * VOXEL_SIZE;
                for x in 0..CHUNK_SIZE {
                    let world_x = origin_x + x as f32 * VOXEL_SIZE;
                    let surface = noise_gen.surface_height(world_x, world_z);
                    let index = x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE;
                    let id = if world_y < surface {
                        noise_gen.stone_id
                    } else {
                        0
                    };
                    // Safety: index < CHUNK_VOLUME, buffer pre-allocated by host
                    unsafe {
                        mem_ptr.add(index).write(id);
                    }
                }
            }
        }

        CHUNK_VOLUME as u32
    }

    fn is_definitely_air(_handle: u32, _cx: i32, cy: i32, _cz: i32) -> bool {
        let noise_gen = generator();
        let chunk_y_min = (cy as f32) * (CHUNK_SIZE as f32) * VOXEL_SIZE;
        let chunk_y_max = chunk_y_min + (CHUNK_SIZE as f32) * VOXEL_SIZE;
        chunk_y_min > noise_gen.max_possible_height() || chunk_y_max <= 0.0
    }
}

export!(GamePlugin);
