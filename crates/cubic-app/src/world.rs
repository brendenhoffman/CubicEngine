// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! World (re)loading and the per-frame guest tick / chunk streaming /
//! upload / remesh / draw pipeline driven from RedrawRequested.

use cubic_math::Vec3;
use cubic_render::{MeshHandle, PushData};
use cubic_wasm::{
    clear_tick_query, set_tick_input, set_tick_query, take_camera_update, InputSnapshot,
    WasmPlugin, WasmWorldGenerator,
};
use cubic_world::{
    mesh_chunk, world_pos_to_chunk, AsyncWorldStream, BlockFaceTextures, WorldGenerator,
    CHUNK_SIZE, VOXEL_SIZE,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::error;

use cubic_world::ChunkPos;

use crate::backend::{Backend, RendererBackend};
use crate::frustum::Frustum;
use crate::App;

/// Renderer-facing world state: chunk/entity mesh handles, the bindless
/// texture-index lookups meshing needs, and the streaming pipeline that
/// drives them. Grouped here (rather than flat on `App`) because it's all
/// renderer-adjacent data — a mesh handle or a texture index means nothing
/// without the `Backend` it was uploaded to — populated by `load_world`
/// and consumed every frame by `world_tick_and_draw`.
pub(crate) struct WorldRenderer {
    pub(crate) stream: AsyncWorldStream,
    pub(crate) chunk_meshes: HashMap<ChunkPos, MeshHandle>,
    // Path (relative to the game's data dir) -> bindless texture index,
    // populated by load_world() from the WASM plugin's block registry.
    // Consumed by the mesher to assign tex_index per face.
    pub(crate) tex_map: HashMap<String, u32>,
    // Per-block-per-face bindless texture index lookup built from tex_map
    // in load_world(); Arc'd so streaming worker threads can share it.
    pub(crate) face_textures: Arc<BlockFaceTextures>,
    pub(crate) entity_meshes: HashMap<u32, MeshHandle>,
    pub(crate) next_entity_mesh_id: u32,
    pub(crate) remesh_scratch: HashSet<ChunkPos>,
    pub(crate) seed: u64,
}

impl WorldRenderer {
    pub(crate) fn new(stream_radius: i32, stream_radius_y: i32) -> Self {
        Self {
            stream: AsyncWorldStream::new(
                stream_radius,
                stream_radius_y,
                Some(Arc::new(cubic_wasm::set_worker_id as fn(usize))),
            ),
            chunk_meshes: HashMap::new(),
            tex_map: HashMap::new(),
            face_textures: Arc::new(BlockFaceTextures::new()),
            entity_meshes: HashMap::new(),
            next_entity_mesh_id: 1,
            remesh_scratch: HashSet::new(),
            seed: 0,
        }
    }
}

impl App {
    /// Load block-face textures into the bindless array and (re)start world
    /// streaming from scratch. Called from handle_launch() once the user
    /// clicks Launch — NOT from resumed(), so the launcher screen can be
    /// shown without loading (or generating) any world data yet.
    pub(crate) fn load_world(&mut self) {
        // Reset state from any previous world so re-launching works
        // cleanly (no supported way to trigger that yet, but load_world()
        // shouldn't assume it only ever runs once).
        self.world.chunk_meshes.clear();
        self.world.face_textures = Arc::new(BlockFaceTextures::new());
        self.world.tex_map = HashMap::new();

        // Reinitialize seed
        let seed = if self.cfg.world.seed == 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos() as u64
        } else {
            self.cfg.world.seed
        };
        self.world.seed = seed;

        // Construct the WASM plugin fresh, with this launch's seed baked
        // in — there's no way to change the seed on an existing WasmPlugin
        // (it's used internally by make_instance()), so making the
        // launcher's seed field actually affect generated terrain means
        // rebuilding the plugin (and the generator, which just wraps it)
        // on every launch rather than reusing one built once in main().
        let worker_count = std::thread::available_parallelism()
            .map_or(4, |n| n.get())
            .saturating_sub(1)
            .max(1);
        let plugin = Arc::new(
            WasmPlugin::load(
                &self.cfg.game.path,
                worker_count,
                self.cfg.game.wasm_memory_mb,
                seed,
            )
            .expect("failed to load game plugin"),
        );

        // Set up asset loading callbacks before warm_up so on_load can call
        // load-mesh/load-texture synchronously during the guest's on_load.
        // Safety: warm_up() is synchronous and returns before these closures
        // go out of scope. The pointers are valid for the duration of the call.
        {
            let backend_ptr = self.backend.as_mut().unwrap() as *mut Backend;
            let entity_meshes_ptr = &mut self.world.entity_meshes as *mut HashMap<u32, MeshHandle>;
            let next_id_ptr = &mut self.world.next_entity_mesh_id as *mut u32;
            let game_dir = std::path::Path::new(&self.cfg.game.path)
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .to_path_buf();
            let game_dir2 = game_dir.clone();

            cubic_wasm::set_load_fns(
                move |path: &str| {
                    let full = game_dir.join(path);
                    let backend = unsafe { &mut *backend_ptr };
                    let entity_meshes = unsafe { &mut *entity_meshes_ptr };
                    let next_id = unsafe { &mut *next_id_ptr };
                    match crate::loader::load_obj_mesh(&full) {
                        Ok((verts, idxs)) => match backend.upload_mesh(&verts, &idxs) {
                            Ok(handle) => {
                                let id = *next_id;
                                *next_id += 1;
                                entity_meshes.insert(id, handle);
                                tracing::info!("loaded mesh: {path} -> handle {id}");
                                id
                            }
                            Err(e) => {
                                tracing::error!("load-mesh upload failed: {e}");
                                0
                            }
                        },
                        Err(e) => {
                            tracing::error!("load-mesh failed for {path}: {e}");
                            0
                        }
                    }
                },
                move |path: &str| {
                    let full = game_dir2.join(path);
                    let backend = unsafe { &mut *backend_ptr };
                    match image::open(&full) {
                        Ok(img) => {
                            let rgba = img.to_rgba8();
                            let (w, h) = rgba.dimensions();
                            match backend.upload_texture(rgba.as_raw(), w, h) {
                                Ok(idx) => {
                                    tracing::info!("loaded texture: {path} -> index {idx}");
                                    idx
                                }
                                Err(e) => {
                                    tracing::error!("load-texture upload failed: {e}");
                                    0
                                }
                            }
                        }
                        Err(e) => {
                            tracing::error!("load-texture failed for {path}: {e}");
                            0
                        }
                    }
                },
            );
        }

        // Warm up (runs on_load, populates block registry) so the texture
        // loading below sees a populated registry immediately.
        plugin.warm_up();
        self.guest.plugin = Some(Arc::clone(&plugin));
        let wasm_gen = Arc::new(WasmWorldGenerator::new(Arc::clone(&plugin)));
        self.guest.generator = Some(Arc::clone(&wasm_gen) as Arc<dyn WorldGenerator>);
        self.guest.wasm_game = Some(wasm_gen);

        // Load textures
        if let Some(backend) = &mut self.backend {
            let unique_paths: HashSet<String> = {
                let registry_arc = plugin.block_registry();
                let registry = registry_arc.lock().unwrap();
                registry
                    .all_defs()
                    .flat_map(|def| {
                        [
                            def.faces.top.clone(),
                            def.faces.bottom.clone(),
                            def.faces.front.clone(),
                            def.faces.back.clone(),
                            def.faces.left.clone(),
                            def.faces.right.clone(),
                        ]
                    })
                    .filter(|p| !p.is_empty())
                    .collect()
            };

            let game_dir = std::path::Path::new(&self.cfg.game.path)
                .parent()
                .unwrap_or(std::path::Path::new("."));

            let mut tex_map: HashMap<String, u32> = HashMap::new();
            for path in unique_paths {
                let full = game_dir.join(&path);
                match image::open(&full) {
                    Ok(img) => {
                        let rgba = img.to_rgba8();
                        let (w, h) = rgba.dimensions();
                        match backend.upload_texture(rgba.as_raw(), w, h) {
                            Ok(index) => {
                                tex_map.insert(path, index);
                            }
                            Err(e) => error!("texture upload failed {full:?}: {e}"),
                        }
                    }
                    Err(e) => error!("failed to load texture {full:?}: {e}"),
                }
            }
            self.world.tex_map = tex_map;

            // Build the per-block-per-face texture lookup the mesher
            // indexes by BlockTypeId, now that tex_map has the path ->
            // bindless index mapping.
            let registry_arc = plugin.block_registry();
            let registry = registry_arc.lock().unwrap();
            let mut face_textures = BlockFaceTextures::new();
            for def in registry.all_defs() {
                // dir order: -X, +X, -Y, +Y, -Z, +Z
                // face mapping: left/right=sides, bottom=-Y, top=+Y, front/back=sides
                let get = |path: &str| self.world.tex_map.get(path).copied().unwrap_or(0);
                face_textures.push([
                    get(&def.faces.left),   // -X
                    get(&def.faces.right),  // +X
                    get(&def.faces.bottom), // -Y
                    get(&def.faces.top),    // +Y
                    get(&def.faces.front),  // -Z
                    get(&def.faces.back),   // +Z
                ]);
            }
            self.world.face_textures = Arc::new(face_textures);
        }

        // Initialize streaming using the current (possibly launcher-edited)
        // radius settings, not whatever main() built App with.
        self.world.stream = AsyncWorldStream::new(
            self.cfg.world.stream_radius,
            self.cfg.world.stream_radius_y,
            Some(Arc::new(cubic_wasm::set_worker_id as fn(usize))),
        );
        self.world.chunk_meshes.clear();
    }

    /// Advance the guest tick, chunk streaming, mesh upload/remesh, and
    /// submit this frame's chunk draws. Called from RedrawRequested once
    /// per frame while InGame/Paused; `now`/`dt` are the frame's
    /// already-computed instant/delta so the upload budget and physics
    /// tick stay consistent with the rest of the frame (egui, present).
    pub(crate) fn world_tick_and_draw(
        &mut self,
        backend: &mut Backend,
        now: std::time::Instant,
        dt: f32,
    ) {
        // --- Physics tick ---
        // Bracket on_tick with a chunk-query view borrowed from
        // self.world.stream: queries happen on the main thread, sequentially,
        // before the streaming update below mutates chunks, so no locking
        // or copying is needed — just a borrow scoped to this call.
        let view = self.world.stream.query_view();
        set_tick_query(&view);

        // take_mouse_delta() is consumed here for the game tick —
        // apply_input() skips its own yaw/pitch update whenever wasm_game
        // is active (see its doc comment) so the delta isn't
        // double-applied.
        let (look_dx, look_dy) = self.input.take_mouse_delta();
        let snap = InputSnapshot {
            move_forward: self.input.binding_active(&self.controls.forward),
            move_back: self.input.binding_active(&self.controls.back),
            move_left: self.input.binding_active(&self.controls.left),
            move_right: self.input.binding_active(&self.controls.right),
            jump: self.input.binding_active(&self.controls.jump),
            sneak: self.input.binding_active(&self.controls.sneak),
            look_dx: look_dx * self.cfg.camera.mouse_sensitivity,
            look_dy: look_dy * self.cfg.camera.mouse_sensitivity,
            walk_speed: self.cfg.player.walk_speed,
            fly_speed: self.cfg.player.fly_speed,
            jump_velocity: self.cfg.player.jump_velocity,
            gravity: self.cfg.player.gravity,
            sprint_multiplier: self.cfg.player.sprint_multiplier,
        };
        // toggle_diagnostics is host-only (no guest round trip needed) —
        // InputTracker still applies its configured trigger gating
        // (tap/double-tap/hold) the same as
        // toggle_third_person/spectate/fly, just acted on directly here
        // instead of via InputEvent.
        if self
            .input_tracker
            .update(&mut self.input, dt)
            .iter()
            .any(|name| name == "toggle_diagnostics")
        {
            self.show_diagnostics = !self.show_diagnostics;
        }
        set_tick_input(snap);

        if let Some(game) = &self.guest.wasm_game {
            game.tick(dt);
        }

        if let Some(cam) = take_camera_update() {
            self.camera.position = Vec3::new(cam.x, cam.y, cam.z);
            self.camera.yaw = cam.yaw;
            self.camera.pitch = cam.pitch;
        }

        clear_tick_query();

        // Apply any block edits (break/place) the guest requested this
        // tick — deferred until after the chunk-query borrow above ends,
        // since it aliases the same chunk data (see BlockEditRequest's doc
        // comment). set_block_at pushes into self.world.stream.remesh_queue,
        // which the boundary remesh pass below already drains — no
        // separate "upload this edit's mesh" step needed.
        for edit in cubic_wasm::take_block_edits() {
            self.world.stream.set_block_at(
                edit.x,
                edit.y,
                edit.z,
                cubic_world::BlockTypeId(edit.block_id),
            );
        }

        // Flush entity draw queue from game tick
        let cam_pos = self.camera.position;
        for req in cubic_wasm::take_draw_queue() {
            if let Some(&handle) = self.world.entity_meshes.get(&req.mesh_handle) {
                let relative = Vec3::new(req.x, req.y, req.z) - cam_pos;
                let cos_y = req.yaw.cos();
                // Negated (not req.yaw + PI): at yaw=0 this matrix already
                // maps the model's -Z front (see player.obj) to world -Z
                // correctly, so adding PI would just turn the model to
                // face backward. The bug is a mirroring, not a 180 offset
                // — cubic_math::Camera::forward() (and player.rs's own
                // forward/right vectors) define yaw such that local -Z
                // rotates to world (-sin(yaw), 0, -cos(yaw)), but this
                // matrix without the negation rotates it to (sin(yaw), 0,
                // -cos(yaw)) instead — correct only at yaw = 0 or PI,
                // mirrored everywhere else. Negating sin_y (equivalent to
                // using -yaw) matches the engine's convention for every
                // yaw, not just the yaw=0 case a single visual check at
                // spawn would catch.
                let sin_y = -req.yaw.sin();
                let push = PushData {
                    model: [
                        [cos_y, 0.0, sin_y, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [-sin_y, 0.0, cos_y, 0.0],
                        [relative.x, relative.y, relative.z, 1.0],
                    ],
                    tint: [1.0, 1.0, 1.0, 1.0],
                    tex_index: req.tex_index,
                    _pad: [0; 3],
                };
                backend.draw_mesh(handle, push);
            }
        }

        // --- Stream update ---
        let center = world_pos_to_chunk(self.camera.position);
        let delta = self.world.stream.update(
            center,
            self.guest.generator.as_ref().unwrap(),
            self.world.seed,
            &self.world.face_textures,
        );

        for pos in delta.unloaded {
            if let Some(handle) = self.world.chunk_meshes.remove(&pos) {
                backend.free_mesh(handle);
            }
        }

        // Compute this frame's mesh budget
        let frame_budget_ms = (dt * 1000.0).min(33.3);
        let upload_ms = if self.cfg.world.upload_budget_ms == 0.0 {
            (frame_budget_ms * 0.25).max(self.cfg.world.upload_budget_min_ms)
        } else {
            self.cfg.world.upload_budget_ms
        };
        let budget_deadline = now + std::time::Duration::from_secs_f32(upload_ms / 1000.0);

        // Upload new chunks
        while std::time::Instant::now() < budget_deadline {
            let Some((pos, verts, idxs)) = self.world.stream.ready_meshes.pop() else {
                break;
            };
            match backend.upload_mesh(&verts, &idxs) {
                Ok(handle) => {
                    self.world.chunk_meshes.insert(pos, handle);
                }
                Err(e) => error!("chunk {pos:?} upload failed: {e}"),
            }
        }

        // Boundary remesh — shares the same deadline
        self.world.remesh_scratch.clear();
        self.world
            .remesh_scratch
            .extend(self.world.stream.remesh_queue.drain(..));
        let mut deferred = Vec::new();
        for &pos in &self.world.remesh_scratch {
            if std::time::Instant::now() >= budget_deadline {
                deferred.push(pos);
                continue;
            }
            let neighbors = self.world.stream.neighbors(pos);
            if neighbors.iter().all(Option::is_none) {
                continue;
            }
            let chunk = match self.world.stream.chunks().get(&pos) {
                Some(c) => c,
                None => continue,
            };
            let (verts, idxs) = mesh_chunk(chunk, neighbors, &self.world.face_textures);
            if let Some(old) = self.world.chunk_meshes.remove(&pos) {
                backend.free_mesh(old);
            }
            if !verts.is_empty() {
                match backend.upload_mesh(&verts, &idxs) {
                    Ok(handle) => {
                        self.world.chunk_meshes.insert(pos, handle);
                        self.world.stream.mark_remeshed(pos);
                    }
                    Err(e) => error!("remesh {pos:?} failed: {e}"),
                }
            }
        }
        self.world.stream.remesh_queue.extend(deferred);

        // --- Draw ---
        backend.set_camera(self.camera);

        let aspect = self.render_size.width as f32 / self.render_size.height as f32;
        let view_proj =
            self.camera.projection_matrix(aspect) * self.camera.view_matrix_no_translation();
        let frustum = Frustum::from_view_proj(&view_proj);
        let chunk_world_size = CHUNK_SIZE as f32 * VOXEL_SIZE;
        let cam_pos = self.camera.position; // snapshot once

        for (&pos, &handle) in &self.world.chunk_meshes {
            let world_origin = pos.to_world_origin();
            let relative = world_origin - cam_pos; // camera-relative translation
            let min = relative;
            let max = relative + Vec3::splat(chunk_world_size);
            if frustum.contains_aabb(min, max) {
                let push = PushData {
                    model: [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [relative.x, relative.y, relative.z, 1.0],
                    ],
                    tint: [1.0, 1.0, 1.0, 1.0],
                    tex_index: 0,
                    _pad: [0; 3],
                };
                backend.draw_mesh(handle, push);
            }
        }
    }
}
