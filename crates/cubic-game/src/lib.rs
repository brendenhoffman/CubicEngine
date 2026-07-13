// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
wit_bindgen::generate!({
    world: "game",
    path: "../cubic-wasm/wit/game.wit",
});

mod player;

use cubic::game::block_registry::{FaceDef, register_block_with_faces};
use cubic::game::commands;
use exports::cubic::game::world_gen::Guest;
use noise::{NoiseFn, OpenSimplex};
use player::{EYE_HEIGHT, InputState, Player};
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
    /// `world_x`/`world_z` are f64 — X/Z are unbounded (per the
    /// f64-world-coordinates card), and f32 loses adjacent-voxel precision
    /// past ~16.78M metres (2^24), causing many consecutive voxel columns to
    /// round to the same noise sample and the terrain to visibly alias/flatten
    /// at extreme distance. `h` (a terrain height, always small — capped
    /// ~12,000m by the generator) stays f32.
    fn surface_height(&self, world_x: f64, world_z: f64) -> f32 {
        let mut h = self.base_height;
        for layer in &self.layers {
            h += self.noise.get([
                world_x * layer.frequency as f64,
                world_z * layer.frequency as f64,
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

static GIVE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Block name -> registered id, persisted from on_load's local scan so
/// on_command (which runs later, per-invocation) can resolve a /give
/// argument like "stone" back to a BlockTypeId.
static BLOCK_IDS: OnceLock<HashMap<String, u32>> = OnceLock::new();

// ---------------------------------------------------------------------------
// Constants — must match cubic-world
// ---------------------------------------------------------------------------

const CHUNK_SIZE: usize = 32;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
const VOXEL_SIZE: f32 = 0.5;

/// Clearance above the queried surface height used for the player's spawn
/// point (see `on_load`) — one voxel is enough to start in air rather than
/// exactly on the solid/air boundary, given a negligible, near-instant fall
/// onto the real surface (sweep_aabb's de-penetration step covers any
/// residual voxel-quantization overlap).
const SPAWN_MARGIN: f32 = VOXEL_SIZE;

/// break_block/place_block/pick_block InputEvents, deferred until after
/// player.tick() runs (see on_tick) since they need this tick's raycast
/// target, which tick() computes internally.
enum BlockAction {
    Break,
    Place,
    Pick,
}

/// `physics::request-set-block`/`get-block` want a point *inside* the
/// target voxel, not its min corner (`RayHit::block`/`place`) — floating
/// point could otherwise land exactly on a boundary and floor into the
/// wrong voxel. Shifting to the voxel's center is a cheap, exact fix.
fn voxel_center(min_corner: [f64; 3]) -> [f64; 3] {
    let half_voxel = VOXEL_SIZE as f64 * 0.5;
    [
        min_corner[0] + half_voxel,
        min_corner[1] + half_voxel,
        min_corner[2] + half_voxel,
    ]
}

/// Bytes per entry in the flat completion-values buffer `commands.
/// register-completion` expects — see that WIT function's doc comment:
/// `vals-len` values, each a null-padded `val-stride`-byte slot.
const COMPLETION_STRIDE: usize = 32;

/// Register `names` as the completion candidates for /give's first
/// (block-name) argument.
fn register_completions_for_give(names: &[&str]) {
    let mut buf = vec![0u8; names.len() * COMPLETION_STRIDE];
    for (i, name) in names.iter().enumerate() {
        let bytes = name.as_bytes();
        let len = bytes.len().min(COMPLETION_STRIDE - 1);
        let start = i * COMPLETION_STRIDE;
        buf[start..start + len].copy_from_slice(&bytes[..len]);
        // Remaining bytes in this slot (including the rest of `buf`'s
        // initial zero-fill) are already the null padding register-completion
        // expects.
    }
    let cmd = "give";
    commands::register_completion(
        cmd.as_ptr() as u32,
        cmd.len() as u32,
        0, // arg-index: /give's first argument
        buf.as_ptr() as u32,
        names.len() as u32,
        COMPLETION_STRIDE as u32,
    );
}

/// `/give <block>` — sets the player's selected block (what break/place's
/// "place" action uses) to the named block type.
fn cmd_give(args: &[&str]) -> String {
    let Some(name) = args.first() else {
        return "Usage: /give <block>".to_string();
    };
    let Some(ids) = BLOCK_IDS.get() else {
        return "Block registry not ready".to_string();
    };
    let Some(&id) = ids.get(*name) else {
        return format!("Unknown block '{name}'");
    };
    let Some(player_cell) = PLAYER.get() else {
        return "Player not ready".to_string();
    };
    player_cell.0.borrow_mut().selected_block = id;
    format!("Selected block set to {name}")
}

// ---------------------------------------------------------------------------
// WIT guest implementation
// ---------------------------------------------------------------------------

static PLAYER_MESH: OnceLock<u32> = OnceLock::new();
static PLAYER_TEX: OnceLock<u32> = OnceLock::new();
static HIGHLIGHT_MESH: OnceLock<u32> = OnceLock::new();
static HIGHLIGHT_TEX: OnceLock<u32> = OnceLock::new();

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

        // Persist the name -> id map (built above while scanning blocks/)
        // for on_command to resolve /give's block-name argument later --
        // on_load only runs once, but on_command runs per-invocation.
        BLOCK_IDS
            .set(block_ids.clone())
            .unwrap_or_else(|_| panic!("on_load called twice"));

        // Register /give command
        let name = "give";
        let usage = "/give <block>";
        let desc = "Set your selected block";
        let give_id = commands::register_command(
            name.as_ptr() as u32,
            name.len() as u32,
            usage.as_ptr() as u32,
            usage.len() as u32,
            desc.as_ptr() as u32,
            desc.len() as u32,
        );
        GIVE_ID.store(give_id, std::sync::atomic::Ordering::Relaxed);

        // Register block name completions for /give arg 0
        let names: Vec<&str> = block_ids.keys().map(|s| s.as_str()).collect();
        register_completions_for_give(&names);

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

        // Load player model
        let mesh_path = b"assets/models/player.obj";
        let mesh_handle =
            cubic::game::render::load_mesh(mesh_path.as_ptr() as u32, mesh_path.len() as u32);
        let tex_path = b"assets/textures/player.png";
        let tex_index =
            cubic::game::render::load_texture(tex_path.as_ptr() as u32, tex_path.len() as u32);
        PLAYER_MESH.set(mesh_handle).ok();
        PLAYER_TEX.set(tex_index).ok();

        // Block-highlight wireframe — see tools/gen_highlight_obj.py for
        // how the mesh itself was generated.
        let hl_mesh_path = b"assets/models/highlight.obj";
        let hl_mesh_handle =
            cubic::game::render::load_mesh(hl_mesh_path.as_ptr() as u32, hl_mesh_path.len() as u32);
        let hl_tex_path = b"assets/textures/highlight.png";
        let hl_tex_index = cubic::game::render::load_texture(
            hl_tex_path.as_ptr() as u32,
            hl_tex_path.len() as u32,
        );
        HIGHLIGHT_MESH.set(hl_mesh_handle).ok();
        HIGHLIGHT_TEX.set(hl_tex_index).ok();

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

        // Spawn at the actual terrain height under (0, 0) — surface_height
        // is the same continuous height function generate() already uses to
        // decide solid-vs-air, and it needs no chunks loaded, so we can
        // query it directly instead of guessing a fixed height (which can
        // land below the real surface on tall-terrain seeds) or dropping
        // the player from way above and waiting for gravity. SPAWN_MARGIN
        // is just enough clearance that the spawn AABB starts in air, not
        // exactly on the solid/air boundary; sweep_aabb's de-penetration
        // step (see cubic-world/src/physics.rs) covers the rest if this
        // column's voxel quantization puts solid ground fractionally above
        // that estimate.
        let spawn_y = generator().surface_height(0.0, 0.0) + SPAWN_MARGIN;
        PLAYER
            .set(PlayerCell(RefCell::new(Player::new(spawn_y, fallback_id))))
            .unwrap_or_else(|_| panic!("on_load called twice"));

        0
    }

    fn generate(_handle: u32, cx: i32, cy: i32, cz: i32, out_ptr: u32) -> u32 {
        let noise_gen = generator();
        // X/Z origins are f64 — see surface_height's doc comment for why.
        // Y stays f32: height is capped ~12,000m by the generator, nowhere
        // near f32's ~16.78M precision cliff.
        let origin_x = (cx as f64) * (CHUNK_SIZE as f64) * VOXEL_SIZE as f64;
        let origin_y = (cy as f32) * (CHUNK_SIZE as f32) * VOXEL_SIZE;
        let origin_z = (cz as f64) * (CHUNK_SIZE as f64) * VOXEL_SIZE as f64;

        // Write voxels into shared memory at out_ptr.
        // Layout: index = x + z*CHUNK_SIZE + y*CHUNK_SIZE*CHUNK_SIZE
        // Safety: out_ptr is a valid offset into WASM linear memory,
        // pre-allocated by the host for this worker's output buffer.
        let mem_ptr = out_ptr as *mut u32;
        for y in 0..CHUNK_SIZE {
            let world_y = origin_y + y as f32 * VOXEL_SIZE;
            for z in 0..CHUNK_SIZE {
                let world_z = origin_z + z as f64 * VOXEL_SIZE as f64;
                for x in 0..CHUNK_SIZE {
                    let world_x = origin_x + x as f64 * VOXEL_SIZE as f64;
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
        let mut buf = [0u8; 52];
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
            walk_speed: f32::from_le_bytes(buf[32..36].try_into().unwrap()),
            fly_speed: f32::from_le_bytes(buf[36..40].try_into().unwrap()),
            jump_velocity: f32::from_le_bytes(buf[40..44].try_into().unwrap()),
            gravity: f32::from_le_bytes(buf[44..48].try_into().unwrap()),
            sprint_multiplier: f32::from_le_bytes(buf[48..52].try_into().unwrap()),
        };

        // Buffer: 64 events × max entry size ~300 bytes
        let mut evt_buf = [0u8; 19200];
        let evt_bytes =
            cubic::game::input::get_events(evt_buf.as_mut_ptr() as u32, evt_buf.len() as u32)
                as usize;

        if let Some(player_cell) = PLAYER.get() {
            let mut player = player_cell.0.borrow_mut();

            // break/place/pick need this tick's raycast target, which is
            // only computed inside player.tick() (it depends on the
            // yaw/pitch mouse-look updates that happen there) — so unlike
            // the toggles below, which must apply *before* tick (e.g.
            // "flying" needs to be set before this tick's physics runs),
            // these are only recorded here and resolved after tick returns.
            let mut block_action: Option<BlockAction> = None;

            // Process discrete events before tick. The host (InputTracker)
            // has already decided *whether* each control's configured
            // trigger (tap vs double-tap) was satisfied this tick — it only
            // ever forwards an event once that's true — so the guest reacts
            // to the action name alone, not a specific kind, for the
            // toggle-style actions. kind==1 (Released) is still delivered
            // for cases that do care, like "sprint" below.
            let mut i = 0;
            while i < evt_bytes {
                if i + 38 > evt_bytes {
                    break;
                }
                let name_bytes = &evt_buf[i..i + 32];
                let name = std::str::from_utf8(name_bytes)
                    .unwrap_or("")
                    .trim_end_matches('\0');
                let kind = u32::from_le_bytes(evt_buf[i + 32..i + 36].try_into().unwrap());
                let payload_len =
                    u16::from_le_bytes(evt_buf[i + 36..i + 38].try_into().unwrap()) as usize;
                let payload = if payload_len > 0 && i + 38 + payload_len <= evt_bytes {
                    std::str::from_utf8(&evt_buf[i + 38..i + 38 + payload_len]).unwrap_or("")
                } else {
                    ""
                };
                // Advance by padded entry size
                let raw = 32 + 4 + 2 + payload_len;
                let entry_size = (raw + 7) & !7;
                i += entry_size;

                match name {
                    "teleport" if kind != 1 => {
                        let coords: Vec<f64> = payload
                            .split_whitespace()
                            .filter_map(|s| s.parse().ok())
                            .collect();
                        if coords.len() == 3 {
                            player.pos = [coords[0], coords[1] - EYE_HEIGHT as f64, coords[2]];
                            player.vel = [0.0; 3];
                            player.grounded = false;
                        }
                    }
                    "teleport_spectator" if kind != 1 => {
                        let coords: Vec<f64> = payload
                            .split_whitespace()
                            .filter_map(|s| s.parse().ok())
                            .collect();
                        if coords.len() == 3 {
                            player.spectator_pos = [coords[0], coords[1], coords[2]];
                        }
                    }
                    "fly" if kind != 1 => {
                        player.flying = !player.flying;
                        player.vel[1] = 0.0;
                    }
                    "spectate" if kind != 1 => {
                        if !player.spectating {
                            if !payload.is_empty() {
                                let coords: Vec<f64> = payload
                                    .split_whitespace()
                                    .filter_map(|s| s.parse().ok())
                                    .collect();
                                if coords.len() == 3 {
                                    player.spectator_pos = [coords[0], coords[1], coords[2]];
                                } else {
                                    player.spectator_pos = player.eye_pos();
                                }
                            } else {
                                player.spectator_pos = player.eye_pos();
                            }
                        }
                        player.spectating = !player.spectating;
                    }
                    "toggle_third_person" if kind != 1 => {
                        player.third_person = !player.third_person;
                    }
                    // Registered by this game's game_overrides.toml, not a
                    // built-in engine control (see main.rs's custom-control
                    // registration) — defaults to double-tap-forward.
                    // Unlike the toggles above, sprint tracks Released too:
                    // it lasts only as long as the key stays held after the
                    // qualifying double-tap, not a persistent on/off flip.
                    "sprint" => {
                        player.sprinting = kind != 1;
                    }
                    // Also game-registered (mouse buttons by default — see
                    // game_overrides.toml). Resolved below, after tick.
                    "break_block" if kind != 1 => block_action = Some(BlockAction::Break),
                    "place_block" if kind != 1 => block_action = Some(BlockAction::Place),
                    "pick_block" if kind != 1 => block_action = Some(BlockAction::Pick),
                    _ => {}
                }
                // `i` was already advanced by entry_size above -- this
                // event's actual (padded) size, not a fixed stride. A
                // second, unconditional `i += 64` here used to double-skip
                // every iteration, corrupting any tick with more than one
                // input event.
            }

            player.tick(dt, &input_state);

            if let (Some(action), Some(target)) = (block_action, player.target) {
                match action {
                    BlockAction::Break => {
                        let c = voxel_center(target.block);
                        cubic::game::physics::request_set_block(c[0], c[1], c[2], 0);
                    }
                    BlockAction::Place => {
                        let c = voxel_center(target.place);
                        cubic::game::physics::request_set_block(
                            c[0],
                            c[1],
                            c[2],
                            player.selected_block,
                        );
                    }
                    BlockAction::Pick => {
                        let c = voxel_center(target.block);
                        let id = cubic::game::physics::get_block(c[0], c[1], c[2]);
                        if id != 0 {
                            player.selected_block = id;
                        }
                    }
                }
            }

            if player.third_person
                && let (Some(&mesh), Some(&tex)) = (PLAYER_MESH.get(), PLAYER_TEX.get())
            {
                cubic::game::render::draw_mesh(
                    mesh,
                    tex,
                    player.pos[0],
                    player.pos[1],
                    player.pos[2],
                    player.yaw,
                );
            }

            if let (Some(target), Some(&mesh), Some(&tex)) =
                (player.target, HIGHLIGHT_MESH.get(), HIGHLIGHT_TEX.get())
            {
                cubic::game::render::draw_mesh(
                    mesh,
                    tex,
                    target.block[0],
                    target.block[1],
                    target.block[2],
                    0.0,
                );
            }
        }
    }
}

impl exports::cubic::game::command_dispatch::Guest for GamePlugin {
    fn on_command(command_id: u32) {
        let give_id = GIVE_ID.load(std::sync::atomic::Ordering::Relaxed);

        let mut arg_buf = [0u8; 1024];
        let args_len =
            commands::get_command_args(arg_buf.as_mut_ptr() as u32, arg_buf.len() as u32) as usize;

        let args: Vec<&str> = arg_buf[..args_len]
            .split(|&b| b == 0)
            .filter_map(|s| std::str::from_utf8(s).ok())
            .filter(|s| !s.is_empty())
            .collect();

        let result = if command_id == give_id {
            cmd_give(&args)
        } else {
            String::new()
        };

        commands::set_command_result(result.as_ptr() as u32, result.len() as u32);
    }
}

export!(GamePlugin);
