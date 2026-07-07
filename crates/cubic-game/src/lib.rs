// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
wit_bindgen::generate!({
    world: "game",
    path: "../cubic-wasm/wit/game.wit",
});

mod player;

use cubic::game::block_registry::{FaceDef, register_block_with_faces};
use exports::cubic::game::world_gen::Guest;
use noise::{NoiseFn, OpenSimplex};
use player::{InputState, Player};
use serde::Deserialize;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// Noise config
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct NoiseLayer {
    frequency: f32,
    amplitude: f32,
}

struct ResolvedSurfaceRule {
    block_id: u32,
    max_depth: Option<u32>,
}

struct NoiseGenerator {
    base_height: f32,
    layers: Vec<NoiseLayer>,
    noise: OpenSimplex,
    surface_rules: Vec<ResolvedSurfaceRule>,
    fallback_id: u32, // used if no rules match (shouldn't happen with a catch-all)
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
// Terrain TOML schema
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct TerrainConfig {
    terrain: TerrainInner,
}

#[derive(Deserialize)]
struct TerrainInner {
    base_height: f32,
    #[allow(dead_code)]
    sea_level: f32,
    #[serde(default)]
    layers: Vec<NoiseLayer>,
    #[serde(default)]
    surface_rules: Vec<SurfaceRule>,
}

#[derive(Deserialize)]
struct SurfaceRule {
    block: String,
    #[serde(default)]
    max_depth: Option<u32>, // None = no limit (catch-all)
}

// ---------------------------------------------------------------------------
// Block textures
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct BlockConfig {
    block: BlockInner,
}

#[derive(Deserialize)]
struct BlockInner {
    name: String,
    faces: BlockFaces,
}

#[derive(Deserialize)]
struct BlockFaces {
    // shorthands
    #[serde(default)]
    all: Option<String>,
    #[serde(default)]
    sides: Option<String>,
    // per-face overrides
    #[serde(default)]
    top: Option<String>,
    #[serde(default)]
    bottom: Option<String>,
    #[serde(default)]
    front: Option<String>,
    #[serde(default)]
    back: Option<String>,
    #[serde(default)]
    left: Option<String>,
    #[serde(default)]
    right: Option<String>,
}

impl BlockFaces {
    fn resolve(&self) -> [String; 6] {
        // Priority: per-face > sides > all > empty
        let all = self.all.as_deref().unwrap_or("");
        let sides = self.sides.as_deref().unwrap_or(all);
        [
            self.left.as_deref().unwrap_or(sides).to_string(), // -X
            self.right.as_deref().unwrap_or(sides).to_string(), // +X
            self.bottom.as_deref().unwrap_or(all).to_string(), // -Y
            self.top.as_deref().unwrap_or(all).to_string(),    // +Y
            self.front.as_deref().unwrap_or(sides).to_string(), // -Z
            self.back.as_deref().unwrap_or(sides).to_string(), // +Z
        ]
    }
}

fn load_block(path: &str) -> Option<(String, [String; 6])> {
    let mut buf = vec![0u8; 65536];
    let len = cubic::game::data::read_file(path, buf.as_mut_ptr() as u32, buf.len() as u32);
    if len == 0 {
        return None;
    }
    buf.truncate(len as usize);
    let cfg: BlockConfig = toml::from_str(std::str::from_utf8(&buf).ok()?).ok()?;
    let faces = cfg.block.faces.resolve();
    Some((cfg.block.name, faces))
}

// ---------------------------------------------------------------------------
// Global state — safe because WASM is single-threaded per instance
// ---------------------------------------------------------------------------

static GENERATOR: OnceLock<NoiseGenerator> = OnceLock::new();

// RefCell isn't Sync, so it can't sit in a OnceLock static as-is even
// though this module only ever runs on one thread (no wasm threads
// proposal in use here). Wrap it so the Sync bound OnceLock's static
// storage requires is satisfiable.
struct PlayerCell(RefCell<Player>);
// Safety: this WASM guest instance is only ever driven from a single
// thread (see the section comment above), so RefCell's lack of Sync is
// never actually exercised.
unsafe impl Sync for PlayerCell {}

static PLAYER: OnceLock<PlayerCell> = OnceLock::new();

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
        // Scan blocks/ directory and register all block definitions
        let mut block_ids: HashMap<String, u32> = HashMap::new();
        let mut buf = vec![0u8; 65536];
        let len = cubic::game::data::list_dir("blocks", buf.as_mut_ptr() as u32, buf.len() as u32);
        buf.truncate(len as usize);

        for filename in std::str::from_utf8(&buf).unwrap_or("").lines() {
            if !filename.ends_with(".toml") {
                continue;
            }
            let path = format!("blocks/{filename}");
            if let Some((name, faces)) = load_block(&path) {
                let id = register_block_with_faces(
                    &name,
                    &FaceDef {
                        top: faces[3].clone(),
                        bottom: faces[2].clone(),
                        front: faces[4].clone(),
                        back: faces[5].clone(),
                        left: faces[0].clone(),
                        right: faces[1].clone(),
                    },
                );
                block_ids.insert(name, id);
            }
        }

        // Read terrain config from datapack
        let mut buf = vec![0u8; 65536];
        let len = cubic::game::data::read_file(
            "world/terrain.toml",
            buf.as_mut_ptr() as u32,
            buf.len() as u32,
        );
        buf.truncate(len as usize);
        let cfg: TerrainConfig =
            toml::from_str(std::str::from_utf8(&buf).expect("terrain.toml is not valid utf8"))
                .expect("failed to parse terrain.toml");

        // Resolve surface rules: map block names to registered IDs
        let surface_rules: Vec<ResolvedSurfaceRule> = cfg
            .terrain
            .surface_rules
            .iter()
            .map(|r| ResolvedSurfaceRule {
                block_id: block_ids.get(&r.block).copied().unwrap_or(0),
                max_depth: r.max_depth,
            })
            .collect();

        let fallback_id = block_ids.get("stone").copied().unwrap_or(0);

        GENERATOR
            .set(NoiseGenerator {
                base_height: cfg.terrain.base_height,
                layers: cfg.terrain.layers,
                noise: OpenSimplex::new(seed as u32),
                surface_rules,
                fallback_id,
            })
            .unwrap_or_else(|_| panic!("on_load called twice"));

        PLAYER
            .set(PlayerCell(RefCell::new(Player::new())))
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
                        let depth_voxels = ((surface - world_y) / VOXEL_SIZE).ceil() as u32;
                        noise_gen
                            .surface_rules
                            .iter()
                            .find(|r| r.max_depth.is_none_or(|d| depth_voxels <= d))
                            .map(|r| r.block_id)
                            .unwrap_or(noise_gen.fallback_id)
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

impl exports::cubic::game::tick::Guest for GamePlugin {
    fn on_tick(dt: f32) {
        // Read input from host — out-ptr pattern, 32 bytes. Stack-allocated
        // is fine; see the matching note on the sweep-aabb buffer in
        // player.rs.
        let mut buf = [0u8; 32];
        cubic::game::input::get_input(buf.as_mut_ptr() as u32);

        let input_state = InputState {
            move_forward: i32::from_le_bytes(buf[0..4].try_into().unwrap()) != 0,
            move_back: i32::from_le_bytes(buf[4..8].try_into().unwrap()) != 0,
            move_left: i32::from_le_bytes(buf[8..12].try_into().unwrap()) != 0,
            move_right: i32::from_le_bytes(buf[12..16].try_into().unwrap()) != 0,
            jump: i32::from_le_bytes(buf[16..20].try_into().unwrap()) != 0,
            sneak: i32::from_le_bytes(buf[20..24].try_into().unwrap()) != 0,
            look_dx: f32::from_le_bytes(buf[24..28].try_into().unwrap()),
            look_dy: f32::from_le_bytes(buf[28..32].try_into().unwrap()),
        };

        if let Some(player_cell) = PLAYER.get() {
            let mut player = player_cell.0.borrow_mut();
            player.tick(dt, &input_state);
        }
    }
}

export!(GamePlugin);
