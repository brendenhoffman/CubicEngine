// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
#[cfg(debug_assertions)]
mod flat_generator;
mod frustum;

use anyhow::Result;
use clap::Parser;
use cubic_core::init_tracing;
use cubic_math::{Camera, Vec3};
use cubic_platform::winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{CursorGrabMode, Window, WindowId},
};
use cubic_render::{MeshHandle, PushData, RenderSize, Renderer, Vertex};
use cubic_render_gl::GlRenderer;
use cubic_render_vk::{HdrFlavor, VkRenderer, VkVsyncMode};
use cubic_wasm::{WasmPlugin, WasmWorldGenerator};
use cubic_world::{
    mesh_chunk, world_pos_to_chunk, AsyncWorldStream, BlockFaceTextures, ChunkPos, WorldGenerator,
    CHUNK_SIZE, VOXEL_SIZE,
};
use frustum::Frustum;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Arc;
use tracing::{error, info};

// ---------------------------------------------------------------------------
// Backend abstraction
// ---------------------------------------------------------------------------

trait RendererBackend {
    fn resize(&mut self, size: RenderSize) -> Result<()>;
    fn set_clear_color(&mut self, rgba: [f32; 4]);
    fn set_vsync(&mut self, on: bool);
    fn configure_advanced(&mut self, cfg: &RenderCfg);
    fn upload_mesh(&mut self, verts: &[Vertex], idxs: &[u32]) -> Result<MeshHandle>;
    fn set_camera(&mut self, camera: Camera);
    fn draw_mesh(&mut self, handle: MeshHandle, push: PushData);
    fn render(&mut self) -> Result<()>;
    fn free_mesh(&mut self, _handle: MeshHandle) {} // default no-op
    fn upload_texture(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<u32>;
}

enum Backend {
    Gl(Box<GlRenderer>),
    Vk(Box<VkRenderer>),
}

impl RendererBackend for Backend {
    fn resize(&mut self, size: RenderSize) -> Result<()> {
        match self {
            Backend::Gl(r) => r.resize(size),
            Backend::Vk(r) => r.resize(size),
        }
    }

    fn set_clear_color(&mut self, rgba: [f32; 4]) {
        match self {
            Backend::Gl(r) => r.set_clear_color(rgba),
            Backend::Vk(r) => r.set_clear_color(rgba),
        }
    }

    fn set_vsync(&mut self, on: bool) {
        match self {
            Backend::Gl(r) => r.set_vsync(on),
            Backend::Vk(r) => r.set_vsync(on),
        }
    }

    fn configure_advanced(&mut self, cfg: &RenderCfg) {
        // GL has no advanced knobs yet.
        if let Backend::Vk(r) = self {
            let mode = match cfg.vsync_mode {
                VsyncMode::Fifo => VkVsyncMode::Fifo,
                VsyncMode::Mailbox => VkVsyncMode::Mailbox,
            };
            r.set_vsync_mode(mode);
            r.set_hdr_enabled(cfg.hdr);
            let flavor = match cfg.hdr_flavor {
                HdrFlavorCfg::PreferScrgb => HdrFlavor::PreferScrgb,
                HdrFlavorCfg::PreferHdr10 => HdrFlavor::PreferHdr10,
            };
            r.set_hdr_flavor(flavor);
        }
    }

    fn upload_mesh(&mut self, verts: &[Vertex], idxs: &[u32]) -> Result<MeshHandle> {
        match self {
            // GL mesh API not yet implemented; uploaded meshes are silently
            // dropped until the GL backend card is complete.
            Backend::Gl(_) => Ok(MeshHandle(u32::MAX)),
            Backend::Vk(r) => r.upload_mesh(verts, idxs),
        }
    }

    fn set_camera(&mut self, camera: Camera) {
        match self {
            Backend::Gl(_) => {} // GL camera via uniforms — not yet implemented.
            Backend::Vk(r) => r.set_camera(camera),
        }
    }

    fn draw_mesh(&mut self, handle: MeshHandle, push: PushData) {
        match self {
            Backend::Gl(_) => {} // GL draw_mesh — not yet implemented.
            Backend::Vk(r) => r.draw_mesh(handle, push),
        }
    }

    fn free_mesh(&mut self, handle: MeshHandle) {
        match self {
            Backend::Gl(_) => {}
            Backend::Vk(r) => r.free_mesh(handle),
        }
    }

    fn render(&mut self) -> Result<()> {
        match self {
            Backend::Gl(r) => r.render(),
            Backend::Vk(r) => r.render(),
        }
    }

    fn upload_texture(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<u32> {
        match self {
            // GL texture API not yet implemented.
            Backend::Gl(_) => Ok(0),
            Backend::Vk(r) => r.upload_texture(pixels, width, height),
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
struct AppCfg {
    #[serde(default)]
    render: RenderCfg,
    #[serde(default)]
    world: WorldCfg,
    #[serde(default)]
    camera: CameraCfg,
    #[serde(default)]
    game: GameCfg,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Choose renderer backend: gl | vk
    #[arg(long, default_value = "vk")]
    backend: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum VsyncMode {
    Fifo,
    #[default]
    Mailbox,
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum UnfocusedPolicy {
    None,
    #[default]
    VsyncOn,
    Throttle,
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum HdrFlavorCfg {
    #[default]
    PreferScrgb,
    PreferHdr10,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct RenderCfg {
    #[serde(default = "default_clear")]
    clear_color: [f32; 4],
    #[serde(default = "default_vsync")]
    vsync: bool,
    #[serde(default)]
    vsync_mode: VsyncMode,
    #[serde(default)]
    unfocused: UnfocusedPolicy,
    #[serde(default)]
    unfocused_fps: u32,
    #[serde(default)]
    fps_when_vsync_off: u32,
    #[serde(default)]
    hdr: bool,
    #[serde(default)]
    hdr_flavor: HdrFlavorCfg,
}

impl Default for RenderCfg {
    fn default() -> Self {
        RenderCfg {
            clear_color: default_clear(),
            vsync: true,
            vsync_mode: VsyncMode::Mailbox,
            unfocused: UnfocusedPolicy::Throttle,
            unfocused_fps: 30,
            fps_when_vsync_off: 0,
            hdr: false,
            hdr_flavor: HdrFlavorCfg::PreferScrgb,
        }
    }
}

fn default_clear() -> [f32; 4] {
    [0.02, 0.02, 0.04, 1.0]
}
fn default_vsync() -> bool {
    true
}
fn load_cfg() -> AppCfg {
    match fs::read_to_string("cubic.toml") {
        Ok(s) => toml::from_str::<AppCfg>(&s).unwrap_or_default(),
        Err(_) => AppCfg::default(),
    }
}

fn default_stream_radius() -> i32 {
    8
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct WorldCfg {
    #[serde(default = "default_stream_radius")]
    stream_radius: i32,
    #[serde(default)]
    seed: u64,
    #[serde(default)] // 0.0 = auto
    upload_budget_ms: f32,
    #[serde(default = "default_upload_budget_min_ms")]
    upload_budget_min_ms: f32,
    #[serde(default = "default_stream_radius_y")]
    stream_radius_y: i32,
}

impl Default for WorldCfg {
    fn default() -> Self {
        WorldCfg {
            stream_radius: default_stream_radius(),
            seed: 0,
            upload_budget_ms: 0.0,
            upload_budget_min_ms: default_upload_budget_min_ms(),
            stream_radius_y: default_stream_radius_y(),
        }
    }
}

fn default_move_speed() -> f32 {
    3.0
}

fn default_mouse_sensitivity() -> f32 {
    0.0025
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct CameraCfg {
    #[serde(default = "default_move_speed")]
    move_speed: f32,
    #[serde(default = "default_mouse_sensitivity")]
    mouse_sensitivity: f32,
}

impl Default for CameraCfg {
    fn default() -> Self {
        CameraCfg {
            move_speed: default_move_speed(),
            mouse_sensitivity: default_mouse_sensitivity(),
        }
    }
}

fn default_upload_budget_min_ms() -> f32 {
    0.5
}

fn default_stream_radius_y() -> i32 {
    2
}

#[derive(Debug, Deserialize, Clone)]
struct GameCfg {
    #[serde(default = "default_game_path")]
    path: String,
    #[serde(default = "default_wasm_memory_mb")]
    wasm_memory_mb: usize,
}

fn default_game_path() -> String {
    "games/cubic-game/game.wasm".to_string()
}

fn default_wasm_memory_mb() -> usize {
    16
}

impl Default for GameCfg {
    fn default() -> Self {
        GameCfg {
            path: default_game_path(),
            wasm_memory_mb: default_wasm_memory_mb(),
        }
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

#[derive(Default)]
struct InputState {
    held_keys: HashSet<KeyCode>,
    mouse_delta: (f32, f32),
}

impl InputState {
    fn set_key(&mut self, code: KeyCode, pressed: bool) {
        if pressed {
            self.held_keys.insert(code);
        } else {
            self.held_keys.remove(&code);
        }
    }

    fn is_held(&self, code: KeyCode) -> bool {
        self.held_keys.contains(&code)
    }

    fn accumulate_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Returns the accumulated delta and resets it to zero.
    fn take_mouse_delta(&mut self) -> (f32, f32) {
        std::mem::take(&mut self.mouse_delta)
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

struct App {
    backend_choice: String,
    window: Option<Window>,
    backend: Option<Backend>,
    render_size: RenderSize,

    cfg: AppCfg,
    exiting: bool,
    frames: u32,
    last_fps_instant: std::time::Instant,

    paused: bool,
    focused: bool,
    next_frame_deadline: Option<std::time::Instant>,

    stream: AsyncWorldStream,
    generator: Arc<dyn WorldGenerator>,
    plugin: Arc<WasmPlugin>,
    chunk_meshes: HashMap<ChunkPos, MeshHandle>,
    seed: u64,
    camera: Camera,
    input: InputState,
    last_frame_instant: std::time::Instant,
    detected_refresh_hz: f32,
    remesh_scratch: HashSet<ChunkPos>,
    // Path (relative to the game's data dir) -> bindless texture index,
    // populated once in resumed() from the WASM plugin's block registry.
    // Consumed by the mesher to assign tex_index per face.
    tex_map: HashMap<String, u32>,
    // Per-block-per-face bindless texture index lookup built from tex_map
    // once in resumed(); Arc'd so streaming worker threads can share it.
    face_textures: Arc<BlockFaceTextures>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let window = event_loop
            .create_window(Window::default_attributes().with_title("cubic"))
            .expect("create_window");

        self.detected_refresh_hz = event_loop
            .primary_monitor()
            .and_then(|m| m.refresh_rate_millihertz())
            .map(|mhz| mhz as f32 / 1000.0)
            .unwrap_or(60.0);

        let size = window.inner_size();
        self.render_size = RenderSize {
            width: size.width.max(1),
            height: size.height.max(1),
        };

        let wh = window.window_handle().expect("window_handle");
        let dh = window.display_handle().expect("display_handle");

        // --- 1. Construct backend ---
        let mut backend: Backend = match self.backend_choice.as_str() {
            "gl" => Backend::Gl(Box::new(
                GlRenderer::new(&wh, &dh, self.render_size).expect("GL init"),
            )),
            _ => match VkRenderer::new(&wh, &dh, self.render_size) {
                Ok(vk) => Backend::Vk(Box::new(vk)),
                Err(e) => {
                    error!("vk init failed: {e}; falling back to gl");
                    Backend::Gl(Box::new(
                        GlRenderer::new(&wh, &dh, self.render_size).expect("GL init"),
                    ))
                }
            },
        };

        // --- 2. Configure backend (agnostic then advanced) ---
        backend.set_clear_color(self.cfg.render.clear_color);
        backend.set_vsync(self.cfg.render.vsync);
        backend.configure_advanced(&self.cfg.render);

        // --- 2b. Load block face textures into the bindless array ---
        let unique_paths: HashSet<String> = {
            let registry_arc = self.plugin.block_registry();
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
        self.tex_map = tex_map;

        // Build the per-block-per-face texture lookup the mesher indexes by
        // BlockTypeId, now that tex_map has the path -> bindless index
        // mapping.
        {
            let registry_arc = self.plugin.block_registry();
            let registry = registry_arc.lock().unwrap();
            let mut face_textures = BlockFaceTextures::new();
            for def in registry.all_defs() {
                // dir order: -X, +X, -Y, +Y, -Z, +Z
                // face mapping: left/right=sides, bottom=-Y, top=+Y, front/back=sides
                let get = |path: &str| self.tex_map.get(path).copied().unwrap_or(0);
                face_textures.push([
                    get(&def.faces.left),   // -X
                    get(&def.faces.right),  // +X
                    get(&def.faces.bottom), // -Y
                    get(&def.faces.top),    // +Y
                    get(&def.faces.front),  // -Z
                    get(&def.faces.back),   // +Z
                ]);
            }
            self.face_textures = Arc::new(face_textures);
        }

        info!(
            "backend = {}",
            match &backend {
                Backend::Gl(_) => "gl",
                Backend::Vk(_) => "vk",
            }
        );
        info!("vsync cfg = {}", self.cfg.render.vsync);

        self.window = Some(window);
        self.backend = Some(backend);

        event_loop.set_control_flow(if self.cfg.render.vsync {
            ControlFlow::Wait
        } else {
            ControlFlow::Poll
        });

        self.paused = self.render_size.width == 0 || self.render_size.height == 0;
        info!("resumed → paused={}", self.paused);

        if !self.paused {
            if let Some(w) = &self.window {
                w.request_redraw();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(window) = &self.window {
            if window_id != window.id() {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                info!("CloseRequested");
                self.exiting = true;
                self.backend = None;
                self.window = None;
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                self.render_size = RenderSize {
                    width: new_size.width,
                    height: new_size.height,
                };
                let now_paused = self.render_size.width == 0 || self.render_size.height == 0;

                if self.paused != now_paused {
                    self.paused = now_paused;
                    info!(
                        "Resized → {}x{} (paused={})",
                        self.render_size.width, self.render_size.height, self.paused
                    );
                } else {
                    info!(
                        "Resized → {}x{} (paused unchanged={})",
                        self.render_size.width, self.render_size.height, self.paused
                    );
                }

                if !self.paused {
                    if let Some(backend) = &mut self.backend {
                        let _ = backend.resize(self.render_size);
                    }
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }

            WindowEvent::Occluded(occluded) => {
                let now_paused =
                    occluded || self.render_size.width == 0 || self.render_size.height == 0;
                if self.paused != now_paused {
                    self.paused = now_paused;
                    info!("Occluded={} → paused={}", occluded, self.paused);
                } else {
                    info!("Occluded={} (paused unchanged={})", occluded, self.paused);
                }
            }

            WindowEvent::Focused(focused) => {
                if self.focused != focused {
                    self.focused = focused;
                    info!("Focused({})", focused);

                    if let Some(backend) = &mut self.backend {
                        match (focused, self.cfg.render.unfocused) {
                            (false, UnfocusedPolicy::VsyncOn) => {
                                backend.set_vsync(true);
                                // Force Fifo (lowest-power vsync) while unfocused.
                                backend.configure_advanced(&RenderCfg {
                                    vsync_mode: VsyncMode::Fifo,
                                    ..self.cfg.render
                                });
                            }
                            (true, UnfocusedPolicy::VsyncOn) => {
                                backend.set_vsync(self.cfg.render.vsync);
                                backend.configure_advanced(&self.cfg.render);
                            }
                            _ => {}
                        }
                    }

                    if let Some(window) = &self.window {
                        if focused {
                            let _ = window
                                .set_cursor_grab(CursorGrabMode::Locked)
                                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                            window.set_cursor_visible(false);
                        } else {
                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        }
                    }

                    if focused {
                        self.next_frame_deadline = None;
                    } else {
                        // Can't reliably observe key-up events while unfocused;
                        // clear held keys so movement doesn't get stuck on alt-tab.
                        self.input.held_keys.clear();
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input
                        .set_key(code, event.state == ElementState::Pressed);
                }
            }

            WindowEvent::RedrawRequested => {
                if self.exiting || self.paused {
                    return;
                }

                let now = std::time::Instant::now();
                let dt = now.duration_since(self.last_frame_instant).as_secs_f32();
                self.last_frame_instant = now;
                self.apply_input(dt);

                if let Some(backend) = &mut self.backend {
                    // --- Stream update ---
                    let center = world_pos_to_chunk(self.camera.position);
                    let delta =
                        self.stream
                            .update(center, &self.generator, self.seed, &self.face_textures);

                    for pos in delta.unloaded {
                        if let Some(handle) = self.chunk_meshes.remove(&pos) {
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
                    let budget_deadline =
                        now + std::time::Duration::from_secs_f32(upload_ms / 1000.0);

                    // Upload new chunks
                    while std::time::Instant::now() < budget_deadline {
                        let Some((pos, verts, idxs)) = self.stream.ready_meshes.pop() else {
                            break;
                        };
                        match backend.upload_mesh(&verts, &idxs) {
                            Ok(handle) => {
                                self.chunk_meshes.insert(pos, handle);
                            }
                            Err(e) => error!("chunk {pos:?} upload failed: {e}"),
                        }
                    }

                    // Boundary remesh — shares the same deadline
                    self.remesh_scratch.clear();
                    self.remesh_scratch
                        .extend(self.stream.remesh_queue.drain(..));
                    let mut deferred = Vec::new();
                    for &pos in &self.remesh_scratch {
                        if std::time::Instant::now() >= budget_deadline {
                            deferred.push(pos);
                            continue;
                        }
                        let neighbors = self.stream.neighbors(pos);
                        if neighbors.iter().all(Option::is_none) {
                            continue;
                        }
                        let chunk = match self.stream.chunks().get(&pos) {
                            Some(c) => c,
                            None => continue,
                        };
                        let (verts, idxs) = mesh_chunk(chunk, neighbors, &self.face_textures);
                        if let Some(old) = self.chunk_meshes.remove(&pos) {
                            backend.free_mesh(old);
                        }
                        if !verts.is_empty() {
                            match backend.upload_mesh(&verts, &idxs) {
                                Ok(handle) => {
                                    self.chunk_meshes.insert(pos, handle);
                                    self.stream.mark_remeshed(pos);
                                }
                                Err(e) => error!("remesh {pos:?} failed: {e}"),
                            }
                        }
                    }
                    self.stream.remesh_queue.extend(deferred);

                    // --- Draw ---
                    backend.set_camera(self.camera);

                    let aspect = self.render_size.width as f32 / self.render_size.height as f32;
                    let view_proj = self.camera.projection_matrix(aspect)
                        * self.camera.view_matrix_no_translation();
                    let frustum = Frustum::from_view_proj(&view_proj);
                    let chunk_world_size = CHUNK_SIZE as f32 * VOXEL_SIZE;
                    let cam_pos = self.camera.position; // snapshot once

                    for (&pos, &handle) in &self.chunk_meshes {
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

                    match backend.render() {
                        Ok(()) => self.frames = self.frames.saturating_add(1),
                        Err(e) => error!("render error: {e}"),
                    }
                }
            }

            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if self.focused {
                self.input
                    .accumulate_mouse_delta(delta.0 as f32, delta.1 as f32);
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.exiting {
            return;
        }

        if self.paused {
            event_loop.set_control_flow(ControlFlow::Wait);
            self.frames = 0;
            return;
        }

        let mut target_fps: u32 = 0;

        if !self.focused {
            match self.cfg.render.unfocused {
                UnfocusedPolicy::Throttle => target_fps = self.cfg.render.unfocused_fps,
                UnfocusedPolicy::VsyncOn => {} // vsync handles pacing
                UnfocusedPolicy::None => {}
            }
        }

        if target_fps == 0 {
            if self.cfg.render.vsync {
                event_loop.set_control_flow(ControlFlow::Wait);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            } else {
                target_fps = self.cfg.render.fps_when_vsync_off;
                if target_fps == 0 {
                    event_loop.set_control_flow(ControlFlow::Poll);
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }
        }

        if target_fps > 0 {
            let now = std::time::Instant::now();
            let frame_dt =
                std::time::Duration::from_nanos(1_000_000_000u64 / target_fps.max(1) as u64);
            let need_redraw = match self.next_frame_deadline {
                None => true,
                Some(t) => now >= t,
            };
            if need_redraw {
                let next = now + frame_dt;
                self.next_frame_deadline = Some(next);
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            } else {
                event_loop
                    .set_control_flow(ControlFlow::WaitUntil(self.next_frame_deadline.unwrap()));
            }
        }

        // FPS counter
        let now = std::time::Instant::now();
        if now.duration_since(self.last_fps_instant).as_secs_f32() >= 1.0 {
            info!("fps ~ {} | loaded={}", self.frames, self.chunk_meshes.len());
            self.frames = 0;
            self.last_fps_instant = now;
        }
    }
}

impl App {
    fn apply_input(&mut self, dt: f32) {
        let (dx, dy) = self.input.take_mouse_delta();
        self.camera.yaw -= dx * self.cfg.camera.mouse_sensitivity;
        self.camera.pitch = (self.camera.pitch - dy * self.cfg.camera.mouse_sensitivity)
            .clamp(-MAX_PITCH, MAX_PITCH);

        let forward = self.camera.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let mut movement = Vec3::ZERO;

        if self.input.is_held(KeyCode::KeyW) {
            movement += forward;
        }
        if self.input.is_held(KeyCode::KeyS) {
            movement -= forward;
        }
        if self.input.is_held(KeyCode::KeyD) {
            movement += right;
        }
        if self.input.is_held(KeyCode::KeyA) {
            movement -= right;
        }
        if self.input.is_held(KeyCode::Space) {
            movement += Vec3::Y;
        }
        if self.input.is_held(KeyCode::ShiftLeft) {
            movement -= Vec3::Y;
        }

        self.camera.position += movement.normalize_or_zero() * self.cfg.camera.move_speed * dt;
    }
}

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let event_loop: EventLoop<()> = EventLoop::new()?;
    let cfg = load_cfg();

    let seed = if cfg.world.seed == 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u64
    } else {
        cfg.world.seed
    };

    let worker_count = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .saturating_sub(1)
        .max(1);

    let plugin = Arc::new(WasmPlugin::load(
        &cfg.game.path,
        worker_count,
        cfg.game.wasm_memory_mb,
        seed,
    )?);
    plugin.warm_up(); // populates block_registry via on_load
    let generator =
        Arc::new(WasmWorldGenerator::new(Arc::clone(&plugin))) as Arc<dyn WorldGenerator>;

    let mut app = App {
        backend_choice: args.backend,
        window: None,
        backend: None,
        render_size: RenderSize {
            width: 1,
            height: 1,
        },
        stream: AsyncWorldStream::new(
            cfg.world.stream_radius,
            cfg.world.stream_radius_y,
            Some(Arc::new(cubic_wasm::set_worker_id as fn(usize))),
        ),
        generator,
        plugin,
        chunk_meshes: HashMap::new(),
        seed,
        cfg,
        exiting: false,
        frames: 0,
        last_fps_instant: std::time::Instant::now(),
        paused: false,
        focused: true,
        next_frame_deadline: None,
        camera: Camera {
            position: Vec3::new(0.0, (CHUNK_SIZE / 2) as f32 * VOXEL_SIZE + 12.0, 0.0),
            pitch: -0.3,
            ..Camera::default()
        },
        input: InputState::default(),
        last_frame_instant: std::time::Instant::now(),
        detected_refresh_hz: 60.0, // overwritten in resumed()
        remesh_scratch: HashSet::new(),
        tex_map: HashMap::new(),
        face_textures: Arc::new(BlockFaceTextures::new()), // populated in resumed()
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
