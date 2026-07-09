// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
#[cfg(debug_assertions)]
mod flat_generator;
mod frustum;
mod game_override;
mod loader;
mod profile;

use anyhow::Result;
use clap::Parser;
use cubic_core::init_tracing;
use cubic_math::{Camera, Vec3};
use cubic_platform::winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{CursorGrabMode, Fullscreen, Window, WindowId},
};
use cubic_render::{MeshHandle, PushData, RenderSize, Renderer, Vertex};
use cubic_render_gl::GlRenderer;
use cubic_render_vk::{Filter, HdrFlavor, SamplerMipmapMode, VkRenderer, VkVsyncMode};
use cubic_wasm::{
    clear_tick_query, set_tick_input, set_tick_query, take_camera_update, InputSnapshot,
    WasmPlugin, WasmWorldGenerator,
};
use cubic_world::{
    mesh_chunk, world_pos_to_chunk, AsyncWorldStream, BlockFaceTextures, ChunkPos, WorldGenerator,
    CHUNK_SIZE, VOXEL_SIZE,
};
use egui::{ClippedPrimitive, TexturesDelta};
use frustum::Frustum;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::sync::Arc;
use tracing::{error, info};

// ---------------------------------------------------------------------------
// App state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum AppState {
    Launcher, // egui launcher shown, no world loaded, cursor free
    InGame,   // world running, cursor locked, no egui (except diagnostics)
    Paused,   // world paused, cursor free, egui pause menu shown
}

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
    fn queue_egui(
        &mut self,
        textures_delta: TexturesDelta,
        paint_jobs: Vec<ClippedPrimitive>,
        w: u32,
        h: u32,
        ppp: f32,
    );
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

            let filter = match cfg.texture_filter {
                TextureFilter::Nearest => Filter::NEAREST,
                TextureFilter::Linear => Filter::LINEAR,
            };
            let mipmap_mode = match cfg.mipmap_mode {
                MipmapMode::Nearest => SamplerMipmapMode::NEAREST,
                MipmapMode::Linear => SamplerMipmapMode::LINEAR,
            };
            r.set_sampler_config(filter, filter, mipmap_mode, cfg.anisotropy, cfg.lod_bias);
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

    fn queue_egui(
        &mut self,
        textures_delta: TexturesDelta,
        paint_jobs: Vec<ClippedPrimitive>,
        w: u32,
        h: u32,
        ppp: f32,
    ) {
        match self {
            Backend::Gl(_) => {}
            Backend::Vk(r) => r.queue_egui(textures_delta, paint_jobs, w, h, ppp),
        }
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, Default)]
struct AppCfg {
    #[serde(default)]
    render: RenderCfg,
    #[serde(default)]
    world: WorldCfg,
    #[serde(default)]
    camera: CameraCfg,
    #[serde(default)]
    player: PlayerCfg,
    #[serde(default)]
    game: GameCfg,
    #[serde(default)]
    controls: ControlsCfg,
    #[serde(default)]
    launcher: LauncherCfg,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Choose renderer backend: gl | vk
    #[arg(long, default_value = "vk")]
    backend: String,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
enum VsyncMode {
    Fifo,
    #[default]
    Mailbox,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
enum UnfocusedPolicy {
    None,
    #[default]
    VsyncOn,
    Throttle,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
enum HdrFlavorCfg {
    #[default]
    PreferScrgb,
    PreferHdr10,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
enum TextureFilter {
    Nearest,
    #[default]
    Linear,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
enum MipmapMode {
    Nearest,
    #[default]
    Linear,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
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
    #[serde(default)]
    texture_filter: TextureFilter,
    #[serde(default)]
    mipmap_mode: MipmapMode,
    #[serde(default = "default_anisotropy")]
    anisotropy: f32,
    #[serde(default)]
    lod_bias: f32,
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
            texture_filter: TextureFilter::Linear,
            mipmap_mode: MipmapMode::Linear,
            anisotropy: default_anisotropy(),
            lod_bias: 0.0,
        }
    }
}

fn default_clear() -> [f32; 4] {
    [0.02, 0.02, 0.04, 1.0]
}
fn default_vsync() -> bool {
    true
}
fn default_anisotropy() -> f32 {
    0.0
}
fn load_cfg() -> AppCfg {
    match fs::read_to_string("cubic.toml") {
        Ok(s) => toml::from_str::<AppCfg>(&s).unwrap_or_default(),
        Err(_) => AppCfg::default(),
    }
}

/// Deserialize a bare string as `T` (e.g. an enum with
/// `#[serde(rename_all = "snake_case")]`), the same way it would deserialize
/// out of a TOML value — used to parse profile override strings (like
/// `texture_filter`) with the exact same rules as the main config.
fn parse_cfg_str<T: for<'de> Deserialize<'de>>(s: &str) -> Option<T> {
    toml::Value::String(s.to_string()).try_into().ok()
}

/// Apply a sparse profile-override onto a resolved `KeyBinding`, one part
/// (key/modifier/trigger) at a time — same "only touch what's actually
/// overridden" contract as every other profile override in this file.
fn apply_key_binding_override(binding: &mut KeyBinding, ov: &profile::KeyBindingOverride) {
    if let Some(k) = &ov.key {
        binding.key = if k.is_empty() { None } else { Some(k.clone()) };
    }
    if let Some(m) = &ov.modifier {
        match parse_cfg_str::<ModifierKey>(m) {
            Some(modifier) => binding.modifier = modifier,
            None => tracing::warn!("unknown modifier in profile: {m}"),
        }
    }
    if let Some(t) = &ov.trigger {
        match parse_cfg_str::<TriggerKind>(t) {
            Some(trigger) => binding.trigger = trigger,
            None => tracing::warn!("unknown trigger kind in profile: {t}"),
        }
    }
}

/// Apply game_overrides.toml onto a resolved AppCfg (which already has
/// global cubic.toml applied). Kept here rather than in game_override.rs,
/// which only owns disk I/O and the sparse override schema —
/// game_override.rs has no reason to know about AppCfg or its enum types,
/// and giving it that dependency risks a cycle since main.rs is what
/// defines AppCfg.
fn apply_game_override(mut cfg: AppCfg, overrides: &game_override::GameOverrideCfg) -> AppCfg {
    if let Some(r) = &overrides.render {
        if let Some(v) = r.vsync {
            cfg.render.vsync = v;
        }
        if let Some(v) = r.anisotropy {
            cfg.render.anisotropy = v;
        }
        if let Some(v) = r.lod_bias {
            cfg.render.lod_bias = v;
        }
        if let Some(v) = r.clear_color {
            cfg.render.clear_color = v;
        }
        if let Some(v) = &r.texture_filter {
            cfg.render.texture_filter = parse_cfg_str(v).unwrap_or(cfg.render.texture_filter);
        }
        if let Some(v) = &r.mipmap_mode {
            cfg.render.mipmap_mode = parse_cfg_str(v).unwrap_or(cfg.render.mipmap_mode);
        }
    }
    if let Some(w) = &overrides.world {
        if let Some(v) = w.stream_radius {
            cfg.world.stream_radius = v;
        }
        if let Some(v) = w.stream_radius_y {
            cfg.world.stream_radius_y = v;
        }
        if let Some(v) = w.upload_budget_ms {
            cfg.world.upload_budget_ms = v;
        }
    }
    if let Some(c) = &overrides.camera {
        if let Some(v) = c.move_speed {
            cfg.camera.move_speed = v;
        }
        if let Some(v) = c.mouse_sensitivity {
            cfg.camera.mouse_sensitivity = v;
        }
    }
    if let Some(p) = &overrides.player {
        if let Some(v) = p.walk_speed {
            cfg.player.walk_speed = v;
        }
        if let Some(v) = p.fly_speed {
            cfg.player.fly_speed = v;
        }
        if let Some(v) = p.jump_velocity {
            cfg.player.jump_velocity = v;
        }
        if let Some(v) = p.gravity {
            cfg.player.gravity = v;
        }
        if let Some(v) = p.sprint_multiplier {
            cfg.player.sprint_multiplier = v;
        }
    }
    cfg
}

/// Apply profile overrides onto a resolved AppCfg (which already has global
/// cubic.toml and game_overrides.toml applied). Kept here rather than in
/// profile.rs, which only owns disk I/O and the sparse override schema —
/// profile.rs has no reason to know about AppCfg or its enum types, and
/// giving it that dependency risks a cycle since main.rs is what defines
/// AppCfg.
fn apply_profile(mut cfg: AppCfg, profile: &profile::ProfileCfg) -> AppCfg {
    if let Some(r) = &profile.render {
        if let Some(v) = r.vsync {
            cfg.render.vsync = v;
        }
        if let Some(v) = r.anisotropy {
            cfg.render.anisotropy = v;
        }
        if let Some(v) = r.lod_bias {
            cfg.render.lod_bias = v;
        }
        if let Some(v) = &r.texture_filter {
            match parse_cfg_str::<TextureFilter>(v) {
                Some(tf) => cfg.render.texture_filter = tf,
                None => tracing::warn!("unknown texture_filter in profile: {v}"),
            }
        }
        if let Some(v) = &r.mipmap_mode {
            match parse_cfg_str::<MipmapMode>(v) {
                Some(mm) => cfg.render.mipmap_mode = mm,
                None => tracing::warn!("unknown mipmap_mode in profile: {v}"),
            }
        }
    }
    if let Some(w) = &profile.world {
        if let Some(v) = w.stream_radius {
            cfg.world.stream_radius = v;
        }
        if let Some(v) = w.stream_radius_y {
            cfg.world.stream_radius_y = v;
        }
        if let Some(v) = w.seed {
            cfg.world.seed = v;
        }
        if let Some(v) = w.upload_budget_ms {
            cfg.world.upload_budget_ms = v;
        }
    }
    if let Some(c) = &profile.camera {
        if let Some(v) = c.move_speed {
            cfg.camera.move_speed = v;
        }
        if let Some(v) = c.mouse_sensitivity {
            cfg.camera.mouse_sensitivity = v;
        }
    }
    if let Some(p) = &profile.player {
        if let Some(v) = p.walk_speed {
            cfg.player.walk_speed = v;
        }
        if let Some(v) = p.fly_speed {
            cfg.player.fly_speed = v;
        }
        if let Some(v) = p.jump_velocity {
            cfg.player.jump_velocity = v;
        }
        if let Some(v) = p.gravity {
            cfg.player.gravity = v;
        }
        if let Some(v) = p.sprint_multiplier {
            cfg.player.sprint_multiplier = v;
        }
    }
    if let Some(ctrl) = &profile.controls {
        if let Some(v) = &ctrl.forward {
            apply_key_binding_override(&mut cfg.controls.forward, v);
        }
        if let Some(v) = &ctrl.back {
            apply_key_binding_override(&mut cfg.controls.back, v);
        }
        if let Some(v) = &ctrl.left {
            apply_key_binding_override(&mut cfg.controls.left, v);
        }
        if let Some(v) = &ctrl.right {
            apply_key_binding_override(&mut cfg.controls.right, v);
        }
        if let Some(v) = &ctrl.jump {
            apply_key_binding_override(&mut cfg.controls.jump, v);
        }
        if let Some(v) = &ctrl.sneak {
            apply_key_binding_override(&mut cfg.controls.sneak, v);
        }
        if let Some(v) = &ctrl.toggle_diagnostics {
            apply_key_binding_override(&mut cfg.controls.toggle_diagnostics, v);
        }
        if let Some(v) = &ctrl.toggle_third_person {
            apply_key_binding_override(&mut cfg.controls.toggle_third_person, v);
        }
        if let Some(v) = &ctrl.spectate {
            apply_key_binding_override(&mut cfg.controls.spectate, v);
        }
        if let Some(v) = &ctrl.fly {
            apply_key_binding_override(&mut cfg.controls.fly, v);
        }
    }
    cfg
}

/// A control the currently loaded game registered itself, via
/// game_overrides.toml's `[[controls]]` (see
/// `game_override::CustomControlDef`) — not one of the engine's fixed
/// `ControlsCfg` fields. The engine has no idea what the named action
/// *does* (that's entirely up to the guest's on_tick matching on the same
/// name, e.g. cubic-game's "sprint"); it only knows how to bind/display/
/// persist it, exactly like a built-in control — see control_binding_mut,
/// control_override_mut, and InputTracker::new, all of which treat this
/// list as a dynamic extension of the built-in ones.
#[derive(Clone)]
struct CustomControl {
    name: String,
    label: String,
    binding: KeyBinding,
}

/// Resolve `overrides`' declared custom controls (defaults) against
/// `profile`'s sparse per-control overrides, the same two-layer resolution
/// built-in controls get from cubic.toml + profile.toml — minus a global
/// cubic.toml layer, since the engine has no default for a control it
/// doesn't know exists.
fn build_custom_controls(
    overrides: &game_override::GameOverrideCfg,
    profile: &profile::ProfileCfg,
) -> Vec<CustomControl> {
    overrides
        .controls
        .iter()
        .map(|def| {
            let mut binding = KeyBinding {
                key: def.key.clone(),
                modifier: def
                    .modifier
                    .as_deref()
                    .and_then(parse_cfg_str::<ModifierKey>)
                    .unwrap_or(ModifierKey::None),
                trigger: def
                    .trigger
                    .as_deref()
                    .and_then(parse_cfg_str::<TriggerKind>)
                    .unwrap_or(TriggerKind::Tap),
            };
            if let Some(ov) = profile
                .controls
                .as_ref()
                .and_then(|c| c.custom.get(&def.name))
            {
                apply_key_binding_override(&mut binding, ov);
            }
            CustomControl {
                name: def.name.clone(),
                label: def.label.clone().unwrap_or_else(|| def.name.clone()),
                binding,
            }
        })
        .collect()
}

fn default_stream_radius() -> i32 {
    8
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
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

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
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

fn default_walk_speed() -> f32 {
    4.5
}
fn default_fly_speed() -> f32 {
    10.0
}
fn default_jump_velocity() -> f32 {
    8.0
}
fn default_gravity() -> f32 {
    -20.0
}
fn default_sprint_multiplier() -> f32 {
    1.5
}

/// In-game player movement/physics — distinct from `CameraCfg`, which only
/// drives the free-fly debug camera shown before a game is loaded. Sent to
/// the guest every tick via `InputSnapshot` (see RedrawRequested), so
/// Settings-tab edits apply immediately without a relaunch. `sprint_multiplier`
/// is a generic movement-speed multiplier — the engine has no concept of
/// "sprint" itself, that's entirely a guest-side feature (see cubic-game's
/// player.rs), this just carries the configured number through the same way
/// walk_speed etc. already do.
#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
struct PlayerCfg {
    #[serde(default = "default_walk_speed")]
    walk_speed: f32,
    #[serde(default = "default_fly_speed")]
    fly_speed: f32,
    #[serde(default = "default_jump_velocity")]
    jump_velocity: f32,
    #[serde(default = "default_gravity")]
    gravity: f32,
    #[serde(default = "default_sprint_multiplier")]
    sprint_multiplier: f32,
}

impl Default for PlayerCfg {
    fn default() -> Self {
        PlayerCfg {
            walk_speed: default_walk_speed(),
            fly_speed: default_fly_speed(),
            jump_velocity: default_jump_velocity(),
            gravity: default_gravity(),
            sprint_multiplier: default_sprint_multiplier(),
        }
    }
}

fn default_upload_budget_min_ms() -> f32 {
    0.5
}

fn default_stream_radius_y() -> i32 {
    2
}

#[derive(Debug, Deserialize, Serialize, Clone)]
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

fn default_launcher_width() -> u32 {
    800
}
fn default_launcher_height() -> u32 {
    600
}

/// The launcher screen's own window size — fixed, global, not part of the
/// per-profile "settings" a player can tweak in-app (no Settings-tab
/// control, no game/profile override layer). Distinct from the *game's*
/// windowed-launch size, which lives in LauncherState/ControlsCfg and is
/// only ever applied to the window in handle_launch().
#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
struct LauncherCfg {
    #[serde(default = "default_launcher_width")]
    width: u32,
    #[serde(default = "default_launcher_height")]
    height: u32,
}

impl Default for LauncherCfg {
    fn default() -> Self {
        LauncherCfg {
            width: default_launcher_width(),
            height: default_launcher_height(),
        }
    }
}

/// Optional modifier layered on top of a control's base key (e.g. "F6" +
/// Shift). Deliberately side-agnostic (not ShiftLeft-vs-ShiftRight) — unlike
/// a control's own base key, which can legitimately be bound to a specific
/// physical modifier key (sneak defaults to ShiftLeft), a *combo* modifier
/// just means "shift is down, either side", matching how most games treat
/// Ctrl/Shift/Alt combos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
enum ModifierKey {
    #[default]
    None,
    Shift,
    Control,
    Alt,
}

impl ModifierKey {
    fn label(self) -> &'static str {
        match self {
            ModifierKey::None => "none",
            ModifierKey::Shift => "Shift",
            ModifierKey::Control => "Control",
            ModifierKey::Alt => "Alt",
        }
    }

    /// Config-string form — must match the `#[serde(rename_all =
    /// "snake_case")]` on this enum, since it's round-tripped through
    /// `parse_cfg_str` (profile.toml overrides store plain strings, not
    /// this enum directly — see KeyBindingOverride's doc comment).
    fn cfg_str(self) -> &'static str {
        match self {
            ModifierKey::None => "none",
            ModifierKey::Shift => "shift",
            ModifierKey::Control => "control",
            ModifierKey::Alt => "alt",
        }
    }
}

/// How a control's key press turns into activation. Only meaningfully
/// distinct for the discrete-event path (see InputTracker::update): Tap
/// fires immediately on press (and additionally reports a DoubleTap-kind
/// event if the press really was rapid enough, same as always) — this is
/// also what movement-style controls use under the hood, since those are
/// read continuously via `InputState::binding_active` and never consult
/// this enum at all. DoubleTap is different in kind, not just labeling: a
/// lone tap is suppressed rather than forwarded, so the bound action only
/// ever fires on a genuine rapid double-press. There used to be a separate
/// Hold variant, but it fired identically to Tap (both act on press, not on
/// release) and every control that actually cares about held-vs-not reads
/// input continuously instead of through this trigger system — so it was
/// removed rather than kept as a confusing no-op option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
enum TriggerKind {
    #[default]
    Tap,
    DoubleTap,
}

impl TriggerKind {
    fn label(self) -> &'static str {
        match self {
            TriggerKind::Tap => "tap",
            TriggerKind::DoubleTap => "double tap",
        }
    }

    /// Config-string form — see ModifierKey::cfg_str's doc comment.
    fn cfg_str(self) -> &'static str {
        match self {
            TriggerKind::Tap => "tap",
            TriggerKind::DoubleTap => "double_tap",
        }
    }
}

/// A control's full binding: base input source (None = unbound), optional
/// modifier, and trigger kind. `key` is a misnomer kept for config-file
/// compatibility — despite the name it can hold a keyboard key, mouse
/// button, or gamepad button name (see `str_to_input_source`). Deserializes
/// from either the legacy plain-string form (`forward = "KeyW"`, still
/// written by older cubic.toml/profile.toml files — treated as that key
/// with no modifier and Tap trigger) or the full table form (`forward =
/// { key = "KeyW", modifier = "shift" }`), so existing configs keep loading
/// unchanged.
#[derive(Debug, Clone, Serialize)]
struct KeyBinding {
    #[serde(skip_serializing_if = "Option::is_none")]
    key: Option<String>,
    #[serde(default, skip_serializing_if = "is_default_modifier")]
    modifier: ModifierKey,
    #[serde(default, skip_serializing_if = "is_default_trigger")]
    trigger: TriggerKind,
}

fn is_default_modifier(m: &ModifierKey) -> bool {
    *m == ModifierKey::None
}
fn is_default_trigger(t: &TriggerKind) -> bool {
    *t == TriggerKind::Tap
}

impl KeyBinding {
    fn unbound(trigger: TriggerKind) -> Self {
        KeyBinding {
            key: None,
            modifier: ModifierKey::None,
            trigger,
        }
    }
    fn key(key: &str) -> Self {
        KeyBinding {
            key: Some(key.to_string()),
            modifier: ModifierKey::None,
            trigger: TriggerKind::Tap,
        }
    }
}

impl<'de> Deserialize<'de> for KeyBinding {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Legacy(String),
            Full {
                #[serde(default)]
                key: Option<String>,
                #[serde(default)]
                modifier: ModifierKey,
                #[serde(default)]
                trigger: TriggerKind,
            },
        }
        Ok(match Repr::deserialize(deserializer)? {
            Repr::Legacy(s) if s.is_empty() => KeyBinding::unbound(TriggerKind::Tap),
            Repr::Legacy(s) => KeyBinding::key(&s),
            Repr::Full {
                key,
                modifier,
                trigger,
            } => KeyBinding {
                key,
                modifier,
                trigger,
            },
        })
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
struct ControlsCfg {
    #[serde(default = "default_forward")]
    forward: KeyBinding,
    #[serde(default = "default_back")]
    back: KeyBinding,
    #[serde(default = "default_left")]
    left: KeyBinding,
    #[serde(default = "default_right")]
    right: KeyBinding,
    #[serde(default = "default_jump")]
    jump: KeyBinding,
    #[serde(default = "default_sneak")]
    sneak: KeyBinding,
    #[serde(default = "default_toggle_diagnostics")]
    toggle_diagnostics: KeyBinding,
    #[serde(default = "default_toggle_third_person")]
    toggle_third_person: KeyBinding,
    // Deliberately unbound by default — a nice-to-have utility mode, not
    // something every player needs a key eaten for out of the box. The
    // pause-menu "Toggle Spectate" button works regardless.
    #[serde(default = "default_spectate")]
    spectate: KeyBinding,
    // Same physical key as jump by default, and that's fine — jump's own
    // ResolvedBinding is checked independently for the continuous
    // ascend/grounded-jump boolean; this is a separate DoubleTap-triggered
    // binding on top, not a conflict.
    #[serde(default = "default_fly")]
    fly: KeyBinding,
}

fn default_forward() -> KeyBinding {
    KeyBinding::key("KeyW")
}
fn default_back() -> KeyBinding {
    KeyBinding::key("KeyS")
}
fn default_left() -> KeyBinding {
    KeyBinding::key("KeyA")
}
fn default_right() -> KeyBinding {
    KeyBinding::key("KeyD")
}
fn default_jump() -> KeyBinding {
    KeyBinding::key("Space")
}
fn default_sneak() -> KeyBinding {
    KeyBinding::key("ShiftLeft")
}
fn default_toggle_diagnostics() -> KeyBinding {
    KeyBinding::key("F3")
}
fn default_toggle_third_person() -> KeyBinding {
    KeyBinding::key("F5")
}
fn default_spectate() -> KeyBinding {
    KeyBinding::unbound(TriggerKind::Tap)
}
fn default_fly() -> KeyBinding {
    KeyBinding {
        key: Some("Space".to_string()),
        modifier: ModifierKey::None,
        trigger: TriggerKind::DoubleTap,
    }
}

impl Default for ControlsCfg {
    fn default() -> Self {
        ControlsCfg {
            forward: default_forward(),
            back: default_back(),
            left: default_left(),
            right: default_right(),
            jump: default_jump(),
            sneak: default_sneak(),
            toggle_diagnostics: default_toggle_diagnostics(),
            toggle_third_person: default_toggle_third_person(),
            spectate: default_spectate(),
            fly: default_fly(),
        }
    }
}

// Note: pause (Escape) is intentionally not bindable here — it's hardcoded
// engine behavior for the app state machine, not a remappable control.
fn str_to_keycode(s: &str) -> Option<KeyCode> {
    match s {
        "KeyA" => Some(KeyCode::KeyA),
        "KeyB" => Some(KeyCode::KeyB),
        "KeyC" => Some(KeyCode::KeyC),
        "KeyD" => Some(KeyCode::KeyD),
        "KeyE" => Some(KeyCode::KeyE),
        "KeyF" => Some(KeyCode::KeyF),
        "KeyG" => Some(KeyCode::KeyG),
        "KeyH" => Some(KeyCode::KeyH),
        "KeyI" => Some(KeyCode::KeyI),
        "KeyJ" => Some(KeyCode::KeyJ),
        "KeyK" => Some(KeyCode::KeyK),
        "KeyL" => Some(KeyCode::KeyL),
        "KeyM" => Some(KeyCode::KeyM),
        "KeyN" => Some(KeyCode::KeyN),
        "KeyO" => Some(KeyCode::KeyO),
        "KeyP" => Some(KeyCode::KeyP),
        "KeyQ" => Some(KeyCode::KeyQ),
        "KeyR" => Some(KeyCode::KeyR),
        "KeyS" => Some(KeyCode::KeyS),
        "KeyT" => Some(KeyCode::KeyT),
        "KeyU" => Some(KeyCode::KeyU),
        "KeyV" => Some(KeyCode::KeyV),
        "KeyW" => Some(KeyCode::KeyW),
        "KeyX" => Some(KeyCode::KeyX),
        "KeyY" => Some(KeyCode::KeyY),
        "KeyZ" => Some(KeyCode::KeyZ),
        "Digit0" => Some(KeyCode::Digit0),
        "Digit1" => Some(KeyCode::Digit1),
        "Digit2" => Some(KeyCode::Digit2),
        "Digit3" => Some(KeyCode::Digit3),
        "Digit4" => Some(KeyCode::Digit4),
        "Digit5" => Some(KeyCode::Digit5),
        "Digit6" => Some(KeyCode::Digit6),
        "Digit7" => Some(KeyCode::Digit7),
        "Digit8" => Some(KeyCode::Digit8),
        "Digit9" => Some(KeyCode::Digit9),
        "Space" => Some(KeyCode::Space),
        "ShiftLeft" => Some(KeyCode::ShiftLeft),
        "ShiftRight" => Some(KeyCode::ShiftRight),
        "ControlLeft" => Some(KeyCode::ControlLeft),
        "ControlRight" => Some(KeyCode::ControlRight),
        "AltLeft" => Some(KeyCode::AltLeft),
        "AltRight" => Some(KeyCode::AltRight),
        "CapsLock" => Some(KeyCode::CapsLock),
        "Insert" => Some(KeyCode::Insert),
        "F1" => Some(KeyCode::F1),
        "F2" => Some(KeyCode::F2),
        "F3" => Some(KeyCode::F3),
        "F4" => Some(KeyCode::F4),
        "F5" => Some(KeyCode::F5),
        "F6" => Some(KeyCode::F6),
        "F7" => Some(KeyCode::F7),
        "F8" => Some(KeyCode::F8),
        "F9" => Some(KeyCode::F9),
        "F10" => Some(KeyCode::F10),
        "F11" => Some(KeyCode::F11),
        "F12" => Some(KeyCode::F12),
        "ArrowUp" => Some(KeyCode::ArrowUp),
        "ArrowDown" => Some(KeyCode::ArrowDown),
        "ArrowLeft" => Some(KeyCode::ArrowLeft),
        "ArrowRight" => Some(KeyCode::ArrowRight),
        // winit's variant is `Enter`, not `Return`.
        "Enter" => Some(KeyCode::Enter),
        "Tab" => Some(KeyCode::Tab),
        "Backspace" => Some(KeyCode::Backspace),
        "Delete" => Some(KeyCode::Delete),
        "Home" => Some(KeyCode::Home),
        "End" => Some(KeyCode::End),
        "PageUp" => Some(KeyCode::PageUp),
        "PageDown" => Some(KeyCode::PageDown),
        _ => {
            tracing::warn!("unknown key name in config: {s}");
            None
        }
    }
}

/// Reverse of `str_to_keycode` — used by the Controls tab's remap capture
/// (see `WindowEvent::KeyboardInput`) to turn a captured `KeyCode` back into
/// the string stored in config. Deliberately exhaustive over the same set
/// `str_to_keycode` accepts, including modifiers (ShiftLeft/ControlLeft/
/// AltLeft/etc): unlike egui's `Key` enum, winit's `KeyCode` reports bare
/// modifier presses as ordinary keys with a precise left/right physical
/// side, which is what makes capturing them for a binding possible at all.
fn keycode_to_str(code: KeyCode) -> Option<&'static str> {
    Some(match code {
        KeyCode::KeyA => "KeyA",
        KeyCode::KeyB => "KeyB",
        KeyCode::KeyC => "KeyC",
        KeyCode::KeyD => "KeyD",
        KeyCode::KeyE => "KeyE",
        KeyCode::KeyF => "KeyF",
        KeyCode::KeyG => "KeyG",
        KeyCode::KeyH => "KeyH",
        KeyCode::KeyI => "KeyI",
        KeyCode::KeyJ => "KeyJ",
        KeyCode::KeyK => "KeyK",
        KeyCode::KeyL => "KeyL",
        KeyCode::KeyM => "KeyM",
        KeyCode::KeyN => "KeyN",
        KeyCode::KeyO => "KeyO",
        KeyCode::KeyP => "KeyP",
        KeyCode::KeyQ => "KeyQ",
        KeyCode::KeyR => "KeyR",
        KeyCode::KeyS => "KeyS",
        KeyCode::KeyT => "KeyT",
        KeyCode::KeyU => "KeyU",
        KeyCode::KeyV => "KeyV",
        KeyCode::KeyW => "KeyW",
        KeyCode::KeyX => "KeyX",
        KeyCode::KeyY => "KeyY",
        KeyCode::KeyZ => "KeyZ",
        KeyCode::Digit0 => "Digit0",
        KeyCode::Digit1 => "Digit1",
        KeyCode::Digit2 => "Digit2",
        KeyCode::Digit3 => "Digit3",
        KeyCode::Digit4 => "Digit4",
        KeyCode::Digit5 => "Digit5",
        KeyCode::Digit6 => "Digit6",
        KeyCode::Digit7 => "Digit7",
        KeyCode::Digit8 => "Digit8",
        KeyCode::Digit9 => "Digit9",
        KeyCode::Space => "Space",
        KeyCode::ShiftLeft => "ShiftLeft",
        KeyCode::ShiftRight => "ShiftRight",
        KeyCode::ControlLeft => "ControlLeft",
        KeyCode::ControlRight => "ControlRight",
        KeyCode::AltLeft => "AltLeft",
        KeyCode::AltRight => "AltRight",
        KeyCode::CapsLock => "CapsLock",
        KeyCode::Insert => "Insert",
        KeyCode::F1 => "F1",
        KeyCode::F2 => "F2",
        KeyCode::F3 => "F3",
        KeyCode::F4 => "F4",
        KeyCode::F5 => "F5",
        KeyCode::F6 => "F6",
        KeyCode::F7 => "F7",
        KeyCode::F8 => "F8",
        KeyCode::F9 => "F9",
        KeyCode::F10 => "F10",
        KeyCode::F11 => "F11",
        KeyCode::F12 => "F12",
        KeyCode::ArrowUp => "ArrowUp",
        KeyCode::ArrowDown => "ArrowDown",
        KeyCode::ArrowLeft => "ArrowLeft",
        KeyCode::ArrowRight => "ArrowRight",
        KeyCode::Enter => "Enter",
        KeyCode::Tab => "Tab",
        KeyCode::Backspace => "Backspace",
        KeyCode::Delete => "Delete",
        KeyCode::Home => "Home",
        KeyCode::End => "End",
        KeyCode::PageUp => "PageUp",
        KeyCode::PageDown => "PageDown",
        _ => return None,
    })
}

/// Any physical input that can be captured as a control's base binding —
/// keyboard key, mouse button, or gamepad button. A binding's modifier
/// (ModifierKey::Shift/Control/Alt) is always keyboard-only regardless of
/// which source this is, same as before.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum InputSource {
    Key(KeyCode),
    Mouse(MouseButton),
    Gamepad(gilrs::Button),
}

fn mouse_button_to_string(button: MouseButton) -> String {
    match button {
        MouseButton::Left => "MouseLeft".to_string(),
        MouseButton::Right => "MouseRight".to_string(),
        MouseButton::Middle => "MouseMiddle".to_string(),
        MouseButton::Back => "MouseBack".to_string(),
        MouseButton::Forward => "MouseForward".to_string(),
        MouseButton::Other(n) => format!("MouseOther{n}"),
    }
}

fn str_to_mouse_button(s: &str) -> Option<MouseButton> {
    Some(match s {
        "MouseLeft" => MouseButton::Left,
        "MouseRight" => MouseButton::Right,
        "MouseMiddle" => MouseButton::Middle,
        "MouseBack" => MouseButton::Back,
        "MouseForward" => MouseButton::Forward,
        other => {
            let n: u16 = other.strip_prefix("MouseOther")?.parse().ok()?;
            MouseButton::Other(n)
        }
    })
}

/// gilrs::Button::Unknown deliberately has no string form — a binding can't
/// meaningfully be restored to "some unrecognized button", so it's simply
/// not capturable/round-trippable, same as an unrecognized KeyCode.
fn gamepad_button_to_str(button: gilrs::Button) -> Option<&'static str> {
    use gilrs::Button::*;
    Some(match button {
        South => "GamepadSouth",
        East => "GamepadEast",
        North => "GamepadNorth",
        West => "GamepadWest",
        C => "GamepadC",
        Z => "GamepadZ",
        LeftTrigger => "GamepadLeftTrigger",
        LeftTrigger2 => "GamepadLeftTrigger2",
        RightTrigger => "GamepadRightTrigger",
        RightTrigger2 => "GamepadRightTrigger2",
        Select => "GamepadSelect",
        Start => "GamepadStart",
        Mode => "GamepadMode",
        LeftThumb => "GamepadLeftThumb",
        RightThumb => "GamepadRightThumb",
        DPadUp => "GamepadDPadUp",
        DPadDown => "GamepadDPadDown",
        DPadLeft => "GamepadDPadLeft",
        DPadRight => "GamepadDPadRight",
        Unknown => return None,
    })
}

fn str_to_gamepad_button(s: &str) -> Option<gilrs::Button> {
    use gilrs::Button::*;
    Some(match s {
        "GamepadSouth" => South,
        "GamepadEast" => East,
        "GamepadNorth" => North,
        "GamepadWest" => West,
        "GamepadC" => C,
        "GamepadZ" => Z,
        "GamepadLeftTrigger" => LeftTrigger,
        "GamepadLeftTrigger2" => LeftTrigger2,
        "GamepadRightTrigger" => RightTrigger,
        "GamepadRightTrigger2" => RightTrigger2,
        "GamepadSelect" => Select,
        "GamepadStart" => Start,
        "GamepadMode" => Mode,
        "GamepadLeftThumb" => LeftThumb,
        "GamepadRightThumb" => RightThumb,
        "GamepadDPadUp" => DPadUp,
        "GamepadDPadDown" => DPadDown,
        "GamepadDPadLeft" => DPadLeft,
        "GamepadDPadRight" => DPadRight,
        _ => return None,
    })
}

/// Config-string form of an `InputSource`, for writing a freshly captured
/// binding back into cubic.toml/profile.toml (see `apply_control_remap`).
fn input_source_to_string(source: InputSource) -> Option<String> {
    match source {
        InputSource::Key(code) => keycode_to_str(code).map(str::to_string),
        InputSource::Mouse(button) => Some(mouse_button_to_string(button)),
        InputSource::Gamepad(button) => gamepad_button_to_str(button).map(str::to_string),
    }
}

/// Reverse of `input_source_to_string`, tried in Mouse/Gamepad/Key order
/// since those prefixes are mutually exclusive with keyboard key names.
fn str_to_input_source(s: &str) -> Option<InputSource> {
    if s.starts_with("Mouse") {
        return str_to_mouse_button(s).map(InputSource::Mouse);
    }
    if s.starts_with("Gamepad") {
        return str_to_gamepad_button(s).map(InputSource::Gamepad);
    }
    str_to_keycode(s).map(InputSource::Key)
}

/// A control's binding resolved once at startup (see `resolve_controls`), so
/// the hot path doesn't re-parse strings every frame. `source: None` means
/// unbound — the control simply never activates, rather than silently
/// falling back to some default key (which would hide a broken/cleared
/// binding instead of surfacing it).
#[derive(Copy, Clone)]
struct ResolvedBinding {
    source: Option<InputSource>,
    modifier: ModifierKey,
    trigger: TriggerKind,
}

fn resolve_binding(b: &KeyBinding) -> ResolvedBinding {
    ResolvedBinding {
        source: b.key.as_deref().and_then(str_to_input_source),
        modifier: b.modifier,
        trigger: b.trigger,
    }
}

#[derive(Copy, Clone)]
struct ResolvedControls {
    forward: ResolvedBinding,
    back: ResolvedBinding,
    left: ResolvedBinding,
    right: ResolvedBinding,
    jump: ResolvedBinding,
    sneak: ResolvedBinding,
    toggle_diagnostics: ResolvedBinding,
    toggle_third_person: ResolvedBinding,
    spectate: ResolvedBinding,
    fly: ResolvedBinding,
}

fn resolve_controls(cfg: &AppCfg) -> ResolvedControls {
    ResolvedControls {
        forward: resolve_binding(&cfg.controls.forward),
        back: resolve_binding(&cfg.controls.back),
        left: resolve_binding(&cfg.controls.left),
        right: resolve_binding(&cfg.controls.right),
        jump: resolve_binding(&cfg.controls.jump),
        sneak: resolve_binding(&cfg.controls.sneak),
        toggle_diagnostics: resolve_binding(&cfg.controls.toggle_diagnostics),
        toggle_third_person: resolve_binding(&cfg.controls.toggle_third_person),
        spectate: resolve_binding(&cfg.controls.spectate),
        fly: resolve_binding(&cfg.controls.fly),
    }
}

// ---------------------------------------------------------------------------
// Input
// ---------------------------------------------------------------------------

const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

#[derive(Default)]
struct InputState {
    held: HashSet<InputSource>,
    mouse_delta: (f32, f32),
}

impl InputState {
    fn set_source(&mut self, source: InputSource, pressed: bool) {
        if pressed {
            self.held.insert(source);
        } else {
            self.held.remove(&source);
        }
    }

    fn is_held(&self, source: InputSource) -> bool {
        self.held.contains(&source)
    }

    /// Whether the given modifier is currently held, side-agnostic (either
    /// ShiftLeft or ShiftRight counts as "Shift"). `ModifierKey::None`
    /// trivially always holds — a binding with no modifier configured
    /// shouldn't require one. Modifiers are always keyboard keys regardless
    /// of what source the binding's own base key uses.
    fn modifier_held(&self, modifier: ModifierKey) -> bool {
        match modifier {
            ModifierKey::None => true,
            ModifierKey::Shift => {
                self.is_held(InputSource::Key(KeyCode::ShiftLeft))
                    || self.is_held(InputSource::Key(KeyCode::ShiftRight))
            }
            ModifierKey::Control => {
                self.is_held(InputSource::Key(KeyCode::ControlLeft))
                    || self.is_held(InputSource::Key(KeyCode::ControlRight))
            }
            ModifierKey::Alt => {
                self.is_held(InputSource::Key(KeyCode::AltLeft))
                    || self.is_held(InputSource::Key(KeyCode::AltRight))
            }
        }
    }

    /// Whether a resolved binding is currently "held": its base input
    /// source is down (if any — unbound bindings never activate) and its
    /// configured modifier, if any, is also down.
    fn binding_active(&self, binding: &ResolvedBinding) -> bool {
        match binding.source {
            Some(source) => self.is_held(source) && self.modifier_held(binding.modifier),
            None => false,
        }
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

struct ActionTracker {
    was_held: bool,
    last_press_time: f32,
}

/// Tracks the purely discrete/toggle-style controls only (movement's
/// forward/back/left/right/jump/sneak are read continuously via
/// `InputState::binding_active` directly into `InputSnapshot` instead —
/// they have no use for tap/double-tap/hold gating, so pushing events for
/// them would just be wasted WASM-boundary traffic).
struct InputTracker {
    // (action_name, binding, state)
    actions: Vec<(String, ResolvedBinding, ActionTracker)>,
    elapsed: f32,
}

impl InputTracker {
    /// `custom` are controls the currently loaded game registered itself
    /// (see CustomControl) — tracked exactly like the four built-in
    /// discrete controls, just resolved from their own KeyBinding instead
    /// of a ResolvedControls field, since the set of names isn't known at
    /// compile time.
    fn new(controls: &ResolvedControls, custom: &[CustomControl]) -> Self {
        let mut actions = vec![
            (
                "toggle_diagnostics".into(),
                controls.toggle_diagnostics,
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ),
            (
                "toggle_third_person".into(),
                controls.toggle_third_person,
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ),
            (
                "spectate".into(),
                controls.spectate,
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ),
            (
                "fly".into(),
                controls.fly,
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ),
        ];
        for c in custom {
            actions.push((
                c.name.clone(),
                resolve_binding(&c.binding),
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            ));
        }
        Self {
            actions,
            elapsed: 0.0,
        }
    }

    /// Advances edge/double-tap detection for every tracked action and
    /// returns the names of any whose configured trigger condition (see
    /// TriggerKind) was satisfied this tick — a lone tap on a
    /// DoubleTap-configured control is *not* included, only a genuine rapid
    /// double-press is. Every fired action is also forwarded to the guest
    /// as an InputEvent (kind 0, or 2 if it happened to be a double-tap);
    /// callers that only care about something host-side (toggle_diagnostics)
    /// use the returned names instead of waiting on a guest round-trip.
    fn update(&mut self, input: &InputState, dt: f32) -> Vec<String> {
        self.elapsed += dt;
        let mut fired = Vec::new();
        for (name, binding, state) in &mut self.actions {
            let is_held = input.binding_active(binding);
            if is_held && !state.was_held {
                let is_double_tap = self.elapsed - state.last_press_time < 0.3;
                state.last_press_time = self.elapsed;
                // DoubleTap is the one case that actually changes *whether*
                // the action fires, not just which kind is reported: a lone
                // tap is swallowed here rather than forwarded, so a
                // DoubleTap-configured control (unlike Hold/Tap) never
                // activates on a single press.
                let should_fire = match binding.trigger {
                    TriggerKind::DoubleTap => is_double_tap,
                    TriggerKind::Tap => true,
                };
                if should_fire {
                    cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                        name: name.clone(),
                        kind: if is_double_tap { 2 } else { 0 },
                    });
                    fired.push(name.clone());
                }
            } else if !is_held && state.was_held {
                cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                    name: name.clone(),
                    kind: 1, // Released
                });
            }
            state.was_held = is_held;
        }
        fired
    }
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

/// Transient launcher UI state — not persisted directly; committed to
/// cfg/profile.toml as the user interacts (see handle_launch,
/// apply_control_remap).
struct LauncherState {
    selected_game: String,
    available_games: Vec<GameEntry>,
    selected_profile: String,
    available_profiles: Vec<String>,
    seed_str: String, // text field buffer for seed input
    window_mode: WindowMode,
    window_width_str: String,
    window_height_str: String,
    // Not yet read anywhere: reserved for a future collapsible/overlay
    // settings affordance within the launcher itself, distinct from the
    // Settings tab already wired up below.
    #[allow(dead_code)]
    settings_open: bool,
    // Which control is being remapped, if any, plus when capture started —
    // capture accepts a key, mouse button, or gamepad button (see
    // window_event's pre-egui interception and App::poll_gamepads) and
    // auto-cancels after REMAP_TIMEOUT if nothing is pressed.
    remapping: Option<(String, std::time::Instant)>,
}

const REMAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);

#[derive(Clone)]
struct GameEntry {
    name: String,         // directory name, used as game_name key
    display_name: String, // from game.toml metadata if present, else same as name
    // Not yet read anywhere: selecting a game in the launcher only updates
    // `selected_game`/`available_profiles` today (see build_game_tab) —
    // actually switching games at Launch time (reconstructing the WASM
    // plugin from this path) is future work.
    #[allow(dead_code)]
    path: std::path::PathBuf, // full path to game directory
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum WindowMode {
    Windowed,
    Maximized,
    Fullscreen,
}

/// Tracks an in-flight maximize→unmaximize dance used to unstick
/// `request_inner_size()` on Wayland, where a plain custom-size request is
/// silently ignored once the compositor's last reported configure marks
/// the window maximized/fullscreen/tiled — and windows can apparently come
/// up "tiled" under some KDE/KWin configs with zero user action. Explicit
/// maximize/unmaximize *are* real protocol state-change requests the
/// compositor must honor (unlike a bare size suggestion), and empirically,
/// cycling through one clears the stuck state and unlocks free resizing
/// again.
///
/// Each step waits for the *previous* one's resize to be confirmed via
/// `WindowEvent::Resized` (Wayland round trips are asynchronous, so firing
/// all three requests back-to-back in one call does not work). That alone
/// wasn't enough, though — reacting to a confirmed Resized by immediately
/// sending the next request, still inside that same event callback, was
/// *also* unreliable in practice: winit had apparently not finished
/// updating its internal last-configure bookkeeping (which gates whether
/// request_inner_size takes effect) by the time the very next request
/// checked it. So each "Confirmed" step below is a deliberate pause,
/// acted on the *following* RedrawRequested rather than inline in the
/// Resized handler that observed it, giving a full event-loop turn to
/// settle before the next request goes out.
enum PendingWindowedResize {
    AwaitingMaximizeConfirm { width: u32, height: u32 },
    MaximizeConfirmed { width: u32, height: u32 },
    AwaitingUnmaximizeConfirm { width: u32, height: u32 },
    UnmaximizeConfirmed { width: u32, height: u32 },
}

fn window_mode_to_str(mode: WindowMode) -> &'static str {
    match mode {
        WindowMode::Windowed => "windowed",
        WindowMode::Maximized => "maximized",
        WindowMode::Fullscreen => "fullscreen",
    }
}

fn str_to_window_mode(s: &str) -> Option<WindowMode> {
    match s {
        "windowed" => Some(WindowMode::Windowed),
        "maximized" => Some(WindowMode::Maximized),
        "fullscreen" => Some(WindowMode::Fullscreen),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq)]
enum LauncherTab {
    Game,
    Profile,
    Settings,
    Controls,
}

fn scan_games() -> Vec<GameEntry> {
    let mut games = vec![];
    // Bundled games dir
    for entry in std::fs::read_dir("games").into_iter().flatten().flatten() {
        if entry.path().is_dir() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let display_name = read_game_display_name(&entry.path()).unwrap_or(name.clone());
            games.push(GameEntry {
                name,
                display_name,
                path: entry.path(),
            });
        }
    }
    // User-installed games from XDG
    for entry in std::fs::read_dir(profile::user_games_dir())
        .into_iter()
        .flatten()
        .flatten()
    {
        if entry.path().is_dir() {
            let name = entry.file_name().to_string_lossy().into_owned();
            let display_name = read_game_display_name(&entry.path()).unwrap_or(name.clone());
            games.push(GameEntry {
                name,
                display_name,
                path: entry.path(),
            });
        }
    }
    games.sort_by(|a, b| a.name.cmp(&b.name));
    games
}

fn read_game_display_name(game_dir: &std::path::Path) -> Option<String> {
    // Try to read [game] name = "..." from game.toml if it exists
    let s = std::fs::read_to_string(game_dir.join("game.toml")).ok()?;
    let v: toml::Value = toml::from_str(&s).ok()?;
    v.get("game")?.get("name")?.as_str().map(|s| s.to_owned())
}

/// Merge `new` into `existing` in place, preserving `existing`'s comments
/// and formatting ("decor" in toml_edit terms) wherever possible. Naively
/// assigning whole sections (`doc[k] = v.clone()`) or even whole leaf
/// values (`doc[k][field] = value(x)`) replaces the target `Item` outright,
/// decor and all — verified empirically, not just assumed — so this walks
/// the structure and only ever swaps in new *scalar* values, transplanting
/// the old decor onto each one, leaving every comment untouched.
///
/// Note: `serde`-derived nested structs always serialize as inline tables
/// (`Item::Value(Value::InlineTable)`), never `Item::Table` — even for
/// what's conceptually a top-level `[section]` — so both sides of this
/// merge are checked via `as_table_like()`/`as_table_like_mut()`, which
/// treats `Table` and `InlineTable` uniformly, rather than matching on
/// `Item::Table` directly.
fn merge_preserving_decor(existing: &mut toml_edit::Item, new: &toml_edit::Item) {
    if let Some(new_table) = new.as_table_like() {
        let Some(existing_table) = existing.as_table_like_mut() else {
            *existing = promote_new_table(new);
            return;
        };
        for (k, v) in new_table.iter() {
            match existing_table.get_mut(k) {
                Some(child) => merge_preserving_decor(child, v),
                None => {
                    existing_table.insert(k, promote_new_table(v));
                }
            }
        }
        return;
    }

    // Leaf scalar: keep the existing decor (indentation/inline comment),
    // just swap in the new value.
    let old_decor = existing.as_value().map(|v| v.decor().clone());
    let mut new_item = new.clone();
    if let (Some(decor), Some(v)) = (old_decor, new_item.as_value_mut()) {
        *v.decor_mut() = decor;
    }
    *existing = new_item;
}

/// A brand-new key (nothing to merge into, no decor to preserve) that
/// serializes as an inline table gets promoted to a real `Table` instead.
/// Two reasons: it should look like cubic.toml's other `[section]` blocks
/// rather than a `key = { ... }` one-liner, and TOML syntax requires bare
/// `key = value` root entries to precede every `[header]` table — leaving
/// it inline would force a brand-new top-level section before all of
/// cubic.toml's existing ones instead of appending it at the end.
fn promote_new_table(item: &toml_edit::Item) -> toml_edit::Item {
    if let toml_edit::Item::Value(toml_edit::Value::InlineTable(it)) = item {
        toml_edit::Item::Table(it.clone().into_table())
    } else {
        item.clone()
    }
}

/// Serialize the current in-memory AppCfg back to cubic.toml, preserving
/// existing comments/formatting via a structural merge (see
/// merge_preserving_decor) instead of a wholesale rewrite. Still "bakes in"
/// every layer's currently active value (global+game_override+profile),
/// not just the one knob that changed — full sparse-diff-only writing
/// would need per-field provenance tracking, a bigger feature than this
/// adds.
fn save_global_cfg(cfg: &AppCfg) {
    let existing = std::fs::read_to_string("cubic.toml").unwrap_or_default();
    let mut doc = existing
        .parse::<toml_edit::DocumentMut>()
        .unwrap_or_default();

    match toml_edit::ser::to_document(cfg) {
        Ok(new_doc) => {
            for (k, v) in new_doc.iter() {
                if doc.contains_key(k) {
                    merge_preserving_decor(&mut doc[k], v);
                } else {
                    doc[k] = promote_new_table(v);
                }
            }
            if let Err(e) = std::fs::write("cubic.toml", doc.to_string()) {
                tracing::warn!("failed to write cubic.toml: {e}");
            }
        }
        Err(e) => tracing::warn!("failed to serialize config: {e}"),
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
    // The profile actively in use — apply_control_remap() updates and saves
    // this (see current_profile_name/current_game_name below) whenever a
    // control is rebound in the launcher/pause Controls tab.
    current_profile: profile::ProfileCfg,
    current_profile_name: String,
    current_game_name: String,
    // Not yet read anywhere: needed by the launcher settings tab to show
    // "(game override)" labels next to affected knobs (future card).
    #[allow(dead_code)]
    game_overrides: game_override::GameOverrideCfg,
    // Controls the currently loaded game registered itself (see
    // CustomControl/build_custom_controls) — resolved once at startup from
    // game_overrides + current_profile, same as `controls`/`input_tracker`.
    custom_controls: Vec<CustomControl>,
    // Transient launcher UI state (selected game/profile, seed field,
    // window-mode radio, remap-in-progress, ...).
    launcher: LauncherState,
    launcher_tab: LauncherTab,
    // Toggled by the pause menu's Settings button; shows the same content
    // as the launcher's Settings tab in a floating egui::Window.
    pause_settings_open: bool,
    // Same idea, for the Controls tab — so bindings (including a game's own
    // custom_controls) can be changed mid-game without quitting to the
    // launcher; persist_control_change already applies changes live.
    pause_controls_open: bool,
    // See PendingWindowedResize doc comment. None when no dance is in
    // flight (the common case — only set by handle_launch's Windowed arm).
    pending_windowed_resize: Option<PendingWindowedResize>,
    exiting: bool,
    // Set by the pause menu's Quit button; event_loop.exit() is only
    // callable from ApplicationHandler methods that receive an
    // &ActiveEventLoop (build_pause_ui doesn't), so the actual exit is
    // deferred to about_to_wait.
    quit_requested: bool,
    frames: u32,
    // Snapshot of `frames` taken once per completed second (see
    // about_to_wait); `frames` itself is a live in-progress counter that
    // resets every second, so UI reading it directly saw a 0→N sawtooth.
    last_fps: u32,
    last_fps_instant: std::time::Instant,

    paused: bool,
    focused: bool,
    next_frame_deadline: Option<std::time::Instant>,

    state: AppState,
    egui_ctx: egui::Context,
    // Option because it's initialized in resumed(), once the window exists.
    egui_winit: Option<egui_winit::State>,
    show_diagnostics: bool,
    // Resolved once at startup from cfg.controls (see resolve_controls).
    controls: ResolvedControls,

    stream: AsyncWorldStream,
    // Both None until load_world() (called from handle_launch()) actually
    // constructs the WASM plugin with that launch's seed baked in — the
    // seed can't be changed on an existing WasmPlugin, so getting a
    // launcher-chosen seed to actually affect terrain means rebuilding the
    // plugin (and generator, which just wraps it) fresh on every launch.
    generator: Option<Arc<dyn WorldGenerator>>,
    plugin: Option<Arc<WasmPlugin>>,
    // Same generator as `generator`, but concretely typed so on_tick can be
    // called — `Arc<dyn WorldGenerator>` doesn't expose `tick`. Both are set
    // together in load_world().
    wasm_game: Option<Arc<WasmWorldGenerator>>,
    chunk_meshes: HashMap<ChunkPos, MeshHandle>,
    seed: u64,
    camera: Camera,
    input: InputState,
    last_frame_instant: std::time::Instant,
    last_frame_dt: f32,
    detected_refresh_hz: f32,
    remesh_scratch: HashSet<ChunkPos>,
    // Path (relative to the game's data dir) -> bindless texture index,
    // populated once in resumed() from the WASM plugin's block registry.
    // Consumed by the mesher to assign tex_index per face.
    tex_map: HashMap<String, u32>,
    // Per-block-per-face bindless texture index lookup built from tex_map
    // once in resumed(); Arc'd so streaming worker threads can share it.
    face_textures: Arc<BlockFaceTextures>,
    entity_meshes: HashMap<u32, MeshHandle>,
    next_entity_mesh_id: u32,
    input_tracker: InputTracker,
    // None if no gamepad backend is available on this platform (Gilrs::new
    // can fail, e.g. no udev) — gamepad support is then simply absent
    // rather than a hard error, same spirit as backend/render fallbacks
    // elsewhere in this file.
    gilrs: Option<gilrs::Gilrs>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // The launcher itself always opens at a fixed size from cubic.toml
        // (not the remembered game window_mode/size in self.launcher —
        // that's applied to the *game's* window only, in handle_launch()).
        let attrs = Window::default_attributes()
            .with_title("cubic")
            .with_inner_size(PhysicalSize::new(
                self.cfg.launcher.width,
                self.cfg.launcher.height,
            ));
        let window = event_loop.create_window(attrs).expect("create_window");

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

        let egui_winit = egui_winit::State::new(
            self.egui_ctx.clone(),
            self.egui_ctx.viewport_id(),
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        self.egui_winit = Some(egui_winit);

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

        // While the Controls tab is capturing a new binding (see
        // build_controls_tab), intercept keyboard/mouse presses here,
        // before egui or the normal match below ever sees them. This has
        // to happen pre-egui because otherwise most mouse clicks would
        // already be consumed by whatever panel/button is under the
        // cursor (egui covers the whole window in Launcher/Paused state),
        // and it has to use the raw winit event streams rather than
        // egui's because egui's `Key` enum has no variants for bare
        // modifier presses and its event stream doesn't expose mouse
        // buttons the same way — see keycode_to_str's doc comment for the
        // same reasoning applied to keyboard alone. Gamepad button
        // capture is handled separately in poll_gamepads, since gilrs
        // events don't arrive as WindowEvents.
        if let Some((binding, _)) = self.launcher.remapping.clone() {
            match &event {
                WindowEvent::KeyboardInput {
                    event: key_event, ..
                } if key_event.state == ElementState::Pressed => {
                    if let PhysicalKey::Code(code) = key_event.physical_key {
                        if code == KeyCode::Escape {
                            self.launcher.remapping = None;
                        } else {
                            self.complete_remap(&binding, InputSource::Key(code));
                        }
                    }
                    return;
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button,
                    ..
                } => {
                    self.complete_remap(&binding, InputSource::Mouse(*button));
                    return;
                }
                _ => {}
            }
        }

        // Feed event to egui first
        if let Some(egui_winit) = &mut self.egui_winit {
            if let Some(window) = &self.window {
                let response = egui_winit.on_window_event(window, &event);
                // Only consume the event if egui wants it AND we are not in
                // InGame state (in InGame, the cursor is locked and egui is
                // not shown, so don't consume).
                if response.consumed && self.state != AppState::InGame {
                    return;
                }
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                info!("CloseRequested");
                self.exiting = true;
                self.backend = None;
                // Drop before the window: its clipboard wraps a raw pointer
                // into the window's Wayland display, and destroying that
                // clipboard after the display is gone segfaults on Wayland.
                self.egui_winit = None;
                self.window = None;
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                self.apply_resized(new_size);

                // Note this resize as confirmed, if it's one step of the
                // maximize/unmaximize dance (see PendingWindowedResize) —
                // the *next* request goes out on the following
                // RedrawRequested, not from inside this handler.
                self.pending_windowed_resize = match self.pending_windowed_resize.take() {
                    Some(PendingWindowedResize::AwaitingMaximizeConfirm { width, height }) => {
                        Some(PendingWindowedResize::MaximizeConfirmed { width, height })
                    }
                    Some(PendingWindowedResize::AwaitingUnmaximizeConfirm { width, height }) => {
                        Some(PendingWindowedResize::UnmaximizeConfirmed { width, height })
                    }
                    other => other,
                };
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

                    self.apply_cursor_state();

                    if focused {
                        self.next_frame_deadline = None;
                    } else {
                        // Can't reliably observe key-up events while unfocused;
                        // clear held keys so movement doesn't get stuck on alt-tab.
                        self.input.held.clear();
                    }
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                // A press that started/completed a remap capture is
                // already handled (and consumed) above, before egui and
                // before this match — this only ever sees ordinary clicks.
                self.input
                    .set_source(InputSource::Mouse(button), state == ElementState::Pressed);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input
                        .set_source(InputSource::Key(code), event.state == ElementState::Pressed);
                }

                if event.state == ElementState::Pressed {
                    if let PhysicalKey::Code(code) = event.physical_key {
                        // toggle_diagnostics used to be special-cased here
                        // too, but that bypassed trigger-kind gating
                        // entirely (any press toggled it, regardless of
                        // Tap/DoubleTap) — it's now handled generically
                        // through InputTracker instead, same as
                        // toggle_third_person/spectate/fly. See its
                        // RedrawRequested call site.
                        if code == KeyCode::Escape {
                            match self.state {
                                AppState::InGame => {
                                    self.state = AppState::Paused;
                                    self.apply_cursor_state();
                                }
                                AppState::Paused => {
                                    self.state = AppState::InGame;
                                    self.apply_cursor_state();
                                }
                                AppState::Launcher => {} // egui handles escape
                            }
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                if self.exiting || self.paused {
                    return;
                }

                let now = std::time::Instant::now();
                let dt = now.duration_since(self.last_frame_instant).as_secs_f32();
                self.last_frame_instant = now;
                self.last_frame_dt = dt;

                self.poll_gamepads();

                // Auto-cancel an in-progress remap capture that's gone
                // unanswered too long — without this, clicking the capture
                // button and walking away would leave the Controls tab
                // stuck showing "Press a key..." forever.
                if let Some((_, started)) = &self.launcher.remapping {
                    if started.elapsed() > REMAP_TIMEOUT {
                        self.launcher.remapping = None;
                    }
                }

                // Advance the maximize/unmaximize dance one step, if the
                // previous step's resize was confirmed on a prior
                // WindowEvent::Resized (see PendingWindowedResize) — done
                // here, a full event-loop turn later, rather than inline
                // in that handler; see the type's doc comment for why.
                match self.pending_windowed_resize.take() {
                    Some(PendingWindowedResize::MaximizeConfirmed { width, height }) => {
                        if let Some(window) = &self.window {
                            window.set_maximized(false);
                        }
                        self.pending_windowed_resize =
                            Some(PendingWindowedResize::AwaitingUnmaximizeConfirm {
                                width,
                                height,
                            });
                    }
                    Some(PendingWindowedResize::UnmaximizeConfirmed { width, height }) => {
                        // request_inner_size's return is the authoritative
                        // result here — winit's Wayland backend never
                        // synthesizes a WindowEvent::Resized for a resize
                        // *the client itself* requested (only for
                        // compositor-initiated ones), so without applying
                        // this directly, render_size/the swapchain would
                        // never learn about a resize that actually worked.
                        let result = self
                            .window
                            .as_ref()
                            .and_then(|w| w.request_inner_size(PhysicalSize::new(width, height)));
                        if let Some(size) = result {
                            self.apply_resized(size);
                        }
                    }
                    other => self.pending_windowed_resize = other,
                }

                // Game input and streaming only when world is active
                if self.state == AppState::InGame {
                    self.apply_input(dt);
                }

                // Build this frame's egui output before borrowing
                // `self.backend` mutably below — build_ui() needs `&mut
                // self`, so it can't run while any other field is already
                // borrowed. take_egui_input/handle_platform_output are kept
                // in their own scopes (rather than spanning the run() call)
                // for the same reason.
                let raw_input = match (&mut self.egui_winit, &self.window) {
                    (Some(egui_winit), Some(window)) => Some(egui_winit.take_egui_input(window)),
                    _ => None,
                };
                let egui_frame = raw_input.map(|raw_input| {
                    // Context is a cheap Arc handle to shared state; clone it
                    // so `run`'s receiver borrow doesn't overlap with the
                    // closure's need for `&mut self` (build_ui).
                    let egui_ctx = self.egui_ctx.clone();
                    let full_output = egui_ctx.run(raw_input, |ctx| {
                        self.build_ui(ctx);
                    });
                    if let (Some(egui_winit), Some(window)) = (&mut self.egui_winit, &self.window) {
                        egui_winit.handle_platform_output(window, full_output.platform_output);
                    }
                    let paint_jobs =
                        egui_ctx.tessellate(full_output.shapes, full_output.pixels_per_point);
                    (
                        full_output.textures_delta,
                        paint_jobs,
                        full_output.pixels_per_point,
                    )
                });

                if let Some(backend) = &mut self.backend {
                    // Scene render only when world is active
                    if self.state == AppState::InGame || self.state == AppState::Paused {
                        // --- Physics tick ---
                        // Bracket on_tick with a chunk-query view borrowed
                        // from self.stream: queries happen on the main
                        // thread, sequentially, before the streaming update
                        // below mutates chunks, so no locking or copying is
                        // needed — just a borrow scoped to this call.
                        let view = self.stream.query_view();
                        set_tick_query(&view);

                        // take_mouse_delta() is consumed here for the game
                        // tick — apply_input() skips its own yaw/pitch
                        // update whenever wasm_game is active (see its doc
                        // comment) so the delta isn't double-applied.
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
                        // toggle_diagnostics is host-only (no guest round
                        // trip needed) — InputTracker still applies its
                        // configured trigger gating (tap/double-tap/hold)
                        // the same as toggle_third_person/spectate/fly, just
                        // acted on directly here instead of via InputEvent.
                        if self
                            .input_tracker
                            .update(&self.input, dt)
                            .iter()
                            .any(|name| name == "toggle_diagnostics")
                        {
                            self.show_diagnostics = !self.show_diagnostics;
                        }
                        set_tick_input(snap);

                        if let Some(game) = &self.wasm_game {
                            game.tick(dt);
                        }

                        if let Some(cam) = take_camera_update() {
                            self.camera.position = Vec3::new(cam.x, cam.y, cam.z);
                            self.camera.yaw = cam.yaw;
                            self.camera.pitch = cam.pitch;
                        }

                        clear_tick_query();

                        // Apply any block edits (break/place) the guest
                        // requested this tick — deferred until after the
                        // chunk-query borrow above ends, since it aliases
                        // the same chunk data (see BlockEditRequest's doc
                        // comment). set_block_at pushes into
                        // self.stream.remesh_queue, which the boundary
                        // remesh pass below already drains — no separate
                        // "upload this edit's mesh" step needed.
                        for edit in cubic_wasm::take_block_edits() {
                            self.stream.set_block_at(
                                edit.x,
                                edit.y,
                                edit.z,
                                cubic_world::BlockTypeId(edit.block_id),
                            );
                        }

                        // Flush entity draw queue from game tick
                        let cam_pos = self.camera.position;
                        for req in cubic_wasm::take_draw_queue() {
                            if let Some(&handle) = self.entity_meshes.get(&req.mesh_handle) {
                                let relative = Vec3::new(req.x, req.y, req.z) - cam_pos;
                                let cos_y = req.yaw.cos();
                                // Negated (not req.yaw + PI): at yaw=0 this
                                // matrix already maps the model's -Z front
                                // (see player.obj) to world -Z correctly, so
                                // adding PI would just turn the model to
                                // face backward. The bug is a mirroring, not
                                // a 180 offset — cubic_math::Camera::forward()
                                // (and player.rs's own forward/right vectors)
                                // define yaw such that local -Z rotates to
                                // world (-sin(yaw), 0, -cos(yaw)), but this
                                // matrix without the negation rotates it to
                                // (sin(yaw), 0, -cos(yaw)) instead — correct
                                // only at yaw = 0 or PI, mirrored everywhere
                                // else. Negating sin_y (equivalent to using
                                // -yaw) matches the engine's convention for
                                // every yaw, not just the yaw=0 case a single
                                // visual check at spawn would catch.
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
                        let delta = self.stream.update(
                            center,
                            self.generator.as_ref().unwrap(),
                            self.seed,
                            &self.face_textures,
                        );

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
                    }

                    // egui -- runs every frame regardless of state
                    if let Some((textures_delta, paint_jobs, pixels_per_point)) = egui_frame {
                        backend.queue_egui(
                            textures_delta,
                            paint_jobs,
                            self.render_size.width,
                            self.render_size.height,
                            pixels_per_point,
                        );
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
            if self.focused && self.state == AppState::InGame {
                self.input
                    .accumulate_mouse_delta(delta.0 as f32, delta.1 as f32);
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.exiting {
            return;
        }

        if self.quit_requested {
            self.exiting = true;
            self.backend = None;
            // See the CloseRequested handler above for why this must come
            // before self.window = None.
            self.egui_winit = None;
            self.window = None;
            event_loop.exit();
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
            self.last_fps = self.frames;
            info!(
                "fps ~ {} | loaded={}",
                self.last_fps,
                self.chunk_meshes.len()
            );
            self.frames = 0;
            self.last_fps_instant = now;
        }
    }
}

impl App {
    /// Free-fly camera controls, used only while no WASM game is loaded
    /// (`wasm_game.is_none()`) — once one is, RedrawRequested's tick handler
    /// feeds input/mouse-look into the guest via on-tick instead, and the
    /// guest owns the camera via set-camera. Skipping both blocks here below
    /// avoids double-applying the same mouse delta to the camera.
    fn apply_input(&mut self, dt: f32) {
        if self.wasm_game.is_none() {
            let (dx, dy) = self.input.take_mouse_delta();
            self.camera.yaw -= dx * self.cfg.camera.mouse_sensitivity;
            self.camera.pitch = (self.camera.pitch - dy * self.cfg.camera.mouse_sensitivity)
                .clamp(-MAX_PITCH, MAX_PITCH);
        }

        if self.wasm_game.is_none() {
            let forward = self.camera.forward();
            let right = forward.cross(Vec3::Y).normalize_or_zero();
            let mut movement = Vec3::ZERO;

            if self.input.binding_active(&self.controls.forward) {
                movement += forward;
            }
            if self.input.binding_active(&self.controls.back) {
                movement -= forward;
            }
            if self.input.binding_active(&self.controls.right) {
                movement += right;
            }
            if self.input.binding_active(&self.controls.left) {
                movement -= right;
            }
            if self.input.binding_active(&self.controls.jump) {
                movement += Vec3::Y;
            }
            if self.input.binding_active(&self.controls.sneak) {
                movement -= Vec3::Y;
            }

            self.camera.position += movement.normalize_or_zero() * self.cfg.camera.move_speed * dt;
        }
    }

    /// Apply a new window size to render_size/paused/backend — shared by
    /// the WindowEvent::Resized handler and the tail of the
    /// maximize/unmaximize dance (see PendingWindowedResize). The latter
    /// needs this because winit's Wayland backend only ever synthesizes
    /// WindowEvent::Resized from a *compositor*-initiated configure, never
    /// from a client's own successful request_inner_size() call — so
    /// without calling this directly, a resize that actually succeeded
    /// would still leave render_size/the swapchain stuck at the old size.
    fn apply_resized(&mut self, new_size: PhysicalSize<u32>) {
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

    fn apply_cursor_state(&self) {
        let Some(window) = &self.window else { return };
        let should_lock = self.focused && self.state == AppState::InGame;
        if should_lock {
            let _ = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            window.set_cursor_visible(false);
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
    }

    /// Load block-face textures into the bindless array and (re)start world
    /// streaming from scratch. Called from handle_launch() once the user
    /// clicks Launch — NOT from resumed(), so the launcher screen can be
    /// shown without loading (or generating) any world data yet.
    fn load_world(&mut self) {
        // Reset state from any previous world so re-launching works
        // cleanly (no supported way to trigger that yet, but load_world()
        // shouldn't assume it only ever runs once).
        self.chunk_meshes.clear();
        self.face_textures = Arc::new(BlockFaceTextures::new());
        self.tex_map = HashMap::new();

        // Reinitialize seed
        let seed = if self.cfg.world.seed == 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos() as u64
        } else {
            self.cfg.world.seed
        };
        self.seed = seed;

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
            let entity_meshes_ptr = &mut self.entity_meshes as *mut HashMap<u32, MeshHandle>;
            let next_id_ptr = &mut self.next_entity_mesh_id as *mut u32;
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
        self.plugin = Some(Arc::clone(&plugin));
        let wasm_gen = Arc::new(WasmWorldGenerator::new(Arc::clone(&plugin)));
        self.generator = Some(Arc::clone(&wasm_gen) as Arc<dyn WorldGenerator>);
        self.wasm_game = Some(wasm_gen);

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
            self.tex_map = tex_map;

            // Build the per-block-per-face texture lookup the mesher
            // indexes by BlockTypeId, now that tex_map has the path ->
            // bindless index mapping.
            let registry_arc = plugin.block_registry();
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

        // Initialize streaming using the current (possibly launcher-edited)
        // radius settings, not whatever main() built App with.
        self.stream = AsyncWorldStream::new(
            self.cfg.world.stream_radius,
            self.cfg.world.stream_radius_y,
            Some(Arc::new(cubic_wasm::set_worker_id as fn(usize))),
        );
        self.chunk_meshes.clear();
    }

    /// Called by the launcher's Launch button (and nothing else yet — the
    /// selected game/profile in `self.launcher` isn't re-resolved into
    /// `self.cfg`/`self.plugin` here; switching games/profiles at runtime is
    /// future work, per the `current_profile`/`current_game_name` fields).
    fn handle_launch(&mut self) {
        // Parse seed
        let seed = self.launcher.seed_str.parse::<u64>().unwrap_or(0);
        self.cfg.world.seed = seed;

        // Apply window mode
        if let Some(window) = &self.window {
            match self.launcher.window_mode {
                WindowMode::Windowed => {
                    window.set_fullscreen(None);
                    if let (Ok(width), Ok(height)) = (
                        self.launcher.window_width_str.parse::<u32>(),
                        self.launcher.window_height_str.parse::<u32>(),
                    ) {
                        // Don't request the size directly — see
                        // PendingWindowedResize. Kick off the maximize
                        // dance instead; the actual request_inner_size
                        // happens once both steps are confirmed, in the
                        // WindowEvent::Resized handler.
                        window.set_maximized(true);
                        self.pending_windowed_resize =
                            Some(PendingWindowedResize::AwaitingMaximizeConfirm { width, height });
                    }
                }
                WindowMode::Maximized => {
                    window.set_fullscreen(None);
                    window.set_maximized(true);
                }
                WindowMode::Fullscreen => {
                    window.set_fullscreen(Some(Fullscreen::Borderless(None)));
                }
            }
        }

        // Remember this launch's window choice for next time. Tied to the
        // Launch click (not each widget edit) so browsing the window-mode
        // options without launching doesn't churn profile.toml.
        self.persist_window_prefs();

        // Load world -- the world loading code formerly in resumed()
        self.load_world();

        // Transition to InGame
        self.state = AppState::InGame;
        self.apply_cursor_state();
    }

    /// Save the launcher's current window mode/size into the active
    /// profile so it's remembered next time this profile is used.
    fn persist_window_prefs(&mut self) {
        let prefs = self
            .current_profile
            .window
            .get_or_insert_with(Default::default);
        prefs.mode = Some(window_mode_to_str(self.launcher.window_mode).to_string());
        prefs.width = self.launcher.window_width_str.parse().ok();
        prefs.height = self.launcher.window_height_str.parse().ok();
        if let Err(e) = profile::save(
            &self.current_profile,
            &self.current_game_name,
            &self.current_profile_name,
        ) {
            tracing::warn!("failed to save profile window prefs: {e}");
        }
    }

    fn build_ui(&mut self, ctx: &egui::Context) {
        match self.state {
            AppState::Launcher => self.build_launcher_ui(ctx),
            AppState::Paused => self.build_pause_ui(ctx),
            AppState::InGame => {
                if self.show_diagnostics {
                    self.build_diagnostics_ui(ctx);
                }
            }
        }
    }

    fn build_launcher_ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("CubicEngine");
            ui.add_space(8.0);

            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.launcher_tab, LauncherTab::Game, "Game");
                ui.selectable_value(&mut self.launcher_tab, LauncherTab::Profile, "Profile");
                ui.selectable_value(&mut self.launcher_tab, LauncherTab::Settings, "Settings");
                ui.selectable_value(&mut self.launcher_tab, LauncherTab::Controls, "Controls");
            });
            ui.separator();

            match self.launcher_tab {
                LauncherTab::Game => self.build_game_tab(ui),
                LauncherTab::Profile => self.build_profile_tab(ui),
                LauncherTab::Settings => self.build_settings_tab(ui),
                LauncherTab::Controls => self.build_controls_tab(ui),
            }

            // Window mode/size lives here (not in a tab) since it's a
            // launch-time choice, not a per-game/profile config knob — it
            // should stay visible next to Launch no matter which tab is
            // open. Persisted in handle_launch(), not on every widget
            // change (see persist_window_prefs).
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                if ui
                    .add_sized([200.0, 40.0], egui::Button::new("Launch"))
                    .clicked()
                {
                    self.handle_launch();
                }

                ui.add_space(8.0);

                if self.launcher.window_mode == WindowMode::Windowed {
                    ui.horizontal(|ui| {
                        ui.label("Size:");
                        ui.text_edit_singleline(&mut self.launcher.window_width_str);
                        ui.label("x");
                        ui.text_edit_singleline(&mut self.launcher.window_height_str);
                    });
                }

                ui.horizontal(|ui| {
                    ui.label("Window:");
                    ui.selectable_value(
                        &mut self.launcher.window_mode,
                        WindowMode::Windowed,
                        "Windowed",
                    );
                    ui.selectable_value(
                        &mut self.launcher.window_mode,
                        WindowMode::Maximized,
                        "Maximized",
                    );
                    ui.selectable_value(
                        &mut self.launcher.window_mode,
                        WindowMode::Fullscreen,
                        "Fullscreen",
                    );
                });
            });
        });
    }

    fn build_game_tab(&mut self, ui: &mut egui::Ui) {
        ui.label("Select game:");
        for game in &self.launcher.available_games.clone() {
            let selected = self.launcher.selected_game == game.name;
            if ui.selectable_label(selected, &game.display_name).clicked()
                && self.launcher.selected_game != game.name
            {
                self.launcher.selected_game = game.name.clone();
                // Reload profiles for new game
                self.launcher.available_profiles = profile::list_profiles(&game.name);
                self.launcher.selected_profile = self
                    .launcher
                    .available_profiles
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "default".to_string());
            }
        }
    }

    fn build_profile_tab(&mut self, ui: &mut egui::Ui) {
        ui.label("Profile:");
        for p in &self.launcher.available_profiles.clone() {
            let selected = self.launcher.selected_profile == *p;
            if ui.selectable_label(selected, p).clicked() {
                self.launcher.selected_profile = p.clone();
            }
        }

        ui.horizontal(|ui| {
            if ui.button("New profile").clicked() {
                // For now just create "new_profile", rename support is future
                let name = "new_profile".to_string();
                let _ = profile::load_or_create(&self.launcher.selected_game, &name);
                self.launcher.available_profiles =
                    profile::list_profiles(&self.launcher.selected_game);
            }
        });

        ui.separator();
        ui.label("World seed (0 = random):");
        ui.text_edit_singleline(&mut self.launcher.seed_str);
    }

    fn build_settings_tab(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.collapsing("Render", |ui| {
                let mut changed = false;

                let mut vsync = self.cfg.render.vsync;
                if ui.checkbox(&mut vsync, "VSync").changed() {
                    self.cfg.render.vsync = vsync;
                    changed = true;
                }

                ui.horizontal(|ui| {
                    ui.label("Texture filter");
                    self.game_override_label(ui, "render.texture_filter");
                    egui::ComboBox::from_id_salt("tex_filter")
                        .selected_text(format!("{:?}", self.cfg.render.texture_filter))
                        .show_ui(ui, |ui| {
                            changed |= ui
                                .selectable_value(
                                    &mut self.cfg.render.texture_filter,
                                    TextureFilter::Nearest,
                                    "nearest",
                                )
                                .changed();
                            changed |= ui
                                .selectable_value(
                                    &mut self.cfg.render.texture_filter,
                                    TextureFilter::Linear,
                                    "linear",
                                )
                                .changed();
                        });
                });

                ui.horizontal(|ui| {
                    ui.label("Anisotropy");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.render.anisotropy,
                            0.0..=16.0,
                        ))
                        .changed();
                });

                ui.horizontal(|ui| {
                    ui.label("LOD bias");
                    changed |= ui
                        .add(egui::Slider::new(&mut self.cfg.render.lod_bias, -2.0..=2.0))
                        .changed();
                });

                // Apply live (not just on next restart) and persist,
                // mirroring the same set_vsync + configure_advanced pair
                // the Focused-event handler already uses.
                if changed {
                    if let Some(backend) = &mut self.backend {
                        backend.set_vsync(self.cfg.render.vsync);
                        backend.configure_advanced(&self.cfg.render);
                    }
                    save_global_cfg(&self.cfg);
                }
            });

            ui.collapsing("World", |ui| {
                let mut changed = false;
                ui.horizontal(|ui| {
                    ui.label("Stream radius XZ");
                    changed |= ui
                        .add(egui::Slider::new(&mut self.cfg.world.stream_radius, 1..=32))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Stream radius Y");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.world.stream_radius_y,
                            1..=16,
                        ))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Upload budget (ms, 0=auto)");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.world.upload_budget_ms,
                            0.0..=16.0,
                        ))
                        .changed();
                });
                if changed {
                    save_global_cfg(&self.cfg);
                }
            });

            ui.collapsing("Camera", |ui| {
                let mut changed = false;
                ui.horizontal(|ui| {
                    ui.label("Move speed (m/s)");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.camera.move_speed,
                            0.5..=100.0,
                        ))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Mouse sensitivity");
                    changed |= ui
                        .add(
                            egui::Slider::new(
                                &mut self.cfg.camera.mouse_sensitivity,
                                0.0001..=0.01,
                            )
                            .logarithmic(true),
                        )
                        .changed();
                });
                if changed {
                    save_global_cfg(&self.cfg);
                }
            });

            ui.collapsing("Player", |ui| {
                let mut changed = false;
                ui.horizontal(|ui| {
                    ui.label("Walk speed (m/s)");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.player.walk_speed,
                            0.5..=20.0,
                        ))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Fly speed (m/s)");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.player.fly_speed,
                            0.5..=50.0,
                        ))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Jump velocity (m/s)");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.player.jump_velocity,
                            1.0..=20.0,
                        ))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Gravity (m/s\u{b2})");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.player.gravity,
                            -40.0..=-1.0,
                        ))
                        .changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Sprint multiplier");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.player.sprint_multiplier,
                            1.0..=3.0,
                        ))
                        .changed();
                });
                if changed {
                    save_global_cfg(&self.cfg);
                }
            });
        });
    }

    /// Show a small "(game)" label if this key is overridden by
    /// game_overrides.toml. Placeholder for now — full detection requires
    /// comparing the resolved value against what the override would have
    /// produced, which needs per-field plumbing this card doesn't add yet.
    fn game_override_label(&self, _ui: &mut egui::Ui, key: &str) {
        let _ = key;
    }

    fn build_controls_tab(&mut self, ui: &mut egui::Ui) {
        let controls = [
            ("Forward", "forward", self.cfg.controls.forward.clone()),
            ("Back", "back", self.cfg.controls.back.clone()),
            ("Left", "left", self.cfg.controls.left.clone()),
            ("Right", "right", self.cfg.controls.right.clone()),
            ("Jump", "jump", self.cfg.controls.jump.clone()),
            ("Sneak", "sneak", self.cfg.controls.sneak.clone()),
            (
                "Toggle diagnostics",
                "toggle_diagnostics",
                self.cfg.controls.toggle_diagnostics.clone(),
            ),
            (
                "Toggle third person",
                "toggle_third_person",
                self.cfg.controls.toggle_third_person.clone(),
            ),
            ("Spectate", "spectate", self.cfg.controls.spectate.clone()),
            ("Fly", "fly", self.cfg.controls.fly.clone()),
        ];

        for (label, action, current) in &controls {
            // Trigger kind only matters for controls actually routed
            // through InputTracker (toggle_diagnostics/toggle_third_person/
            // spectate/fly); movement controls are read continuously via
            // InputState::binding_active and never consult it, so the
            // dropdown would just be a confusing no-op there.
            let show_trigger = matches!(
                *action,
                "toggle_diagnostics" | "toggle_third_person" | "spectate" | "fly"
            );
            self.control_row(ui, label, action, current, show_trigger);
        }

        // Controls the currently loaded game registered itself (see
        // game_override::CustomControlDef) — the engine has no idea what
        // these *do* (that's entirely up to the guest's on_tick matching on
        // the same name), it only knows how to bind/display/persist them,
        // same as any built-in control. Always InputTracker-routed (there's
        // no continuously-read variant for a game-defined control), so the
        // trigger dropdown always applies.
        if !self.custom_controls.is_empty() {
            ui.separator();
            ui.label("Game controls");
            let custom = self.custom_controls.clone();
            for c in &custom {
                self.control_row(ui, &c.label, &c.name, &c.binding, true);
            }
        }
    }

    /// One row of the Controls tab: a capture button (left-click to bind
    /// any key/mouse/gamepad button, right-click to clear), a modifier
    /// combo, and — if `show_trigger` — a trigger-kind combo. Shared by
    /// both the fixed built-in control list and the dynamic per-game
    /// `custom_controls` list, which differ only in where their KeyBinding
    /// lives (see control_binding_mut/control_override_mut).
    fn control_row(
        &mut self,
        ui: &mut egui::Ui,
        label: &str,
        action: &str,
        current: &KeyBinding,
        show_trigger: bool,
    ) {
        ui.horizontal(|ui| {
            ui.label(label);

            // Base binding: captured by pressing/clicking it, via the raw
            // winit event streams intercepted at the top of window_event
            // (see that comment) rather than egui's own event stream —
            // that's what makes modifier keys (ShiftLeft, ControlLeft,
            // AltLeft, ...) usable as a control's own base key at all, and
            // what lets a mouse button or gamepad button be captured too:
            // egui's `Key` enum has no variants for bare modifier presses
            // (they only ever show up via a `Modifiers` bitset alongside
            // some other key, which can't tell ShiftLeft from ShiftRight
            // anyway) and has no concept of mouse/gamepad buttons as
            // bindable at all. This is unambiguous for a *base* key
            // (there's only ever one, and it's the whole point of pressing
            // it) — unlike the *combo* modifier below, which must not be
            // captured this way (see its comment).
            let capturing = matches!(&self.launcher.remapping, Some((a, _)) if a == action);
            let btn_label = if capturing {
                "Press a key, click, or button... (Esc to cancel)".to_string()
            } else {
                current.key.clone().unwrap_or_else(|| "unbound".to_string())
            };
            let btn = ui
                .button(&btn_label)
                .on_hover_text("Click to bind, right-click to clear");
            if btn.clicked() {
                self.launcher.remapping = Some((action.to_string(), std::time::Instant::now()));
            }
            if btn.secondary_clicked() {
                self.clear_control_binding(action);
            }

            // Combo modifier: an explicit dropdown, not press-to-capture —
            // "wait for the next key" can't distinguish "the user wants
            // Shift+F6" from "the user wants to bind Shift itself," since
            // the modifier's own key-down event arrives first either way.
            // Selecting it here instead means nothing is reserved: any key
            // can still be the base key (above), and any of Shift/Control/
            // Alt can independently gate it.
            let mut modifier = current.modifier;
            egui::ComboBox::from_id_salt(format!("modifier_{action}"))
                .selected_text(if modifier == ModifierKey::None {
                    "+ modifier".to_string()
                } else {
                    modifier.label().to_string()
                })
                .show_ui(ui, |ui| {
                    for m in [
                        ModifierKey::None,
                        ModifierKey::Shift,
                        ModifierKey::Control,
                        ModifierKey::Alt,
                    ] {
                        ui.selectable_value(&mut modifier, m, m.label());
                    }
                });
            if modifier != current.modifier {
                self.set_control_modifier(action, modifier);
            }

            // Trigger kind — see TriggerKind's doc comment for what each
            // option actually changes.
            if show_trigger {
                let mut trigger = current.trigger;
                egui::ComboBox::from_id_salt(format!("trigger_{action}"))
                    .selected_text(trigger.label())
                    .show_ui(ui, |ui| {
                        for t in [TriggerKind::Tap, TriggerKind::DoubleTap] {
                            ui.selectable_value(&mut trigger, t, t.label());
                        }
                    });
                if trigger != current.trigger {
                    self.set_control_trigger(action, trigger);
                }
            }
        });
    }

    /// The live `KeyBinding` for a control named the same way the Controls
    /// tab / InputTracker / InputEvent system already names actions
    /// ("forward", "spectate", ...). Centralizing this name->field lookup
    /// here is what lets remap/modifier/trigger changes share one small
    /// generic setter each instead of three parallel match statements. Any
    /// name not matching a built-in engine control falls back to
    /// `custom_controls` — a control the currently loaded game registered
    /// itself (see CustomControl/build_custom_controls).
    fn control_binding_mut(&mut self, name: &str) -> Option<&mut KeyBinding> {
        match name {
            "forward" => Some(&mut self.cfg.controls.forward),
            "back" => Some(&mut self.cfg.controls.back),
            "left" => Some(&mut self.cfg.controls.left),
            "right" => Some(&mut self.cfg.controls.right),
            "jump" => Some(&mut self.cfg.controls.jump),
            "sneak" => Some(&mut self.cfg.controls.sneak),
            "toggle_diagnostics" => Some(&mut self.cfg.controls.toggle_diagnostics),
            "toggle_third_person" => Some(&mut self.cfg.controls.toggle_third_person),
            "spectate" => Some(&mut self.cfg.controls.spectate),
            "fly" => Some(&mut self.cfg.controls.fly),
            _ => self
                .custom_controls
                .iter_mut()
                .find(|c| c.name == name)
                .map(|c| &mut c.binding),
        }
    }

    /// The profile-override sparse entry for a control, creating it (and
    /// its parent `ControlsOverride`) if this is the first time any part of
    /// this control has been overridden. Custom (game-registered) controls
    /// share one sparse `HashMap<String, KeyBindingOverride>` instead of
    /// their own named field, since the set of names isn't known statically.
    fn control_override_mut(&mut self, name: &str) -> Option<&mut profile::KeyBindingOverride> {
        let ctrl = self
            .current_profile
            .controls
            .get_or_insert_with(Default::default);
        match name {
            "forward" => Some(ctrl.forward.get_or_insert_with(Default::default)),
            "back" => Some(ctrl.back.get_or_insert_with(Default::default)),
            "left" => Some(ctrl.left.get_or_insert_with(Default::default)),
            "right" => Some(ctrl.right.get_or_insert_with(Default::default)),
            "jump" => Some(ctrl.jump.get_or_insert_with(Default::default)),
            "sneak" => Some(ctrl.sneak.get_or_insert_with(Default::default)),
            "toggle_diagnostics" => {
                Some(ctrl.toggle_diagnostics.get_or_insert_with(Default::default))
            }
            "toggle_third_person" => Some(
                ctrl.toggle_third_person
                    .get_or_insert_with(Default::default),
            ),
            "spectate" => Some(ctrl.spectate.get_or_insert_with(Default::default)),
            "fly" => Some(ctrl.fly.get_or_insert_with(Default::default)),
            _ => Some(ctrl.custom.entry(name.to_string()).or_default()),
        }
    }

    /// Save the profile and rebuild `self.controls` *and* `self.input_tracker`
    /// so a binding change (key/modifier/trigger, from any of the setters
    /// below) applies immediately and survives restart — shared tail of all
    /// of them. Rebuilding the tracker is essential, not just tidy: it caches
    /// its own copy of every ResolvedBinding it watches (toggle_diagnostics/
    /// toggle_third_person/spectate/fly), and without refreshing it here a
    /// control's key/modifier/trigger could be changed in the UI and saved
    /// to disk while runtime behavior kept using whatever was resolved at
    /// startup — indistinguishable from the change doing nothing at all.
    fn persist_control_change(&mut self) {
        if let Err(e) = profile::save(
            &self.current_profile,
            &self.current_game_name,
            &self.current_profile_name,
        ) {
            tracing::warn!("failed to save profile after control change: {e}");
        }
        self.controls = resolve_controls(&self.cfg);
        self.input_tracker = InputTracker::new(&self.controls, &self.custom_controls);
    }

    fn apply_control_remap(&mut self, binding: &str, key_name: &str) {
        if let Some(b) = self.control_binding_mut(binding) {
            b.key = Some(key_name.to_string());
        }
        if let Some(ov) = self.control_override_mut(binding) {
            ov.key = Some(key_name.to_string());
        }
        self.persist_control_change();
    }

    /// Finish an in-progress remap capture with a newly-pressed input
    /// source, converting it to config-string form. On failure (an
    /// unrecognized source, e.g. a gilrs `Button::Unknown`), the capture is
    /// deliberately left in progress rather than cancelled, so a stray
    /// unsupported press doesn't kick the user out of capture mode — same
    /// as the old keyboard-only behavior.
    fn complete_remap(&mut self, binding: &str, source: InputSource) {
        match input_source_to_string(source) {
            Some(key_name) => {
                self.apply_control_remap(binding, &key_name);
                self.launcher.remapping = None;
            }
            None => tracing::warn!("unsupported input source for remapping: {source:?}"),
        }
    }

    /// Unbind a control entirely (right-click on its key button in the
    /// Controls tab) — also cancels any in-progress capture, so right-
    /// clicking works as an unambiguous "clear" regardless of state.
    fn clear_control_binding(&mut self, binding: &str) {
        self.launcher.remapping = None;
        if let Some(b) = self.control_binding_mut(binding) {
            b.key = None;
        }
        if let Some(ov) = self.control_override_mut(binding) {
            ov.key = Some(String::new());
        }
        self.persist_control_change();
    }

    /// Drain pending gilrs events: updates continuous held-state for
    /// gamepad buttons the same way keyboard/mouse do, and — if the
    /// Controls tab is capturing a new binding — completes it on the first
    /// button press seen. Called once per rendered frame (RedrawRequested),
    /// which keeps running in Launcher/Paused state, not just InGame.
    fn poll_gamepads(&mut self) {
        let Some(gilrs) = &mut self.gilrs else {
            return;
        };
        let mut events = Vec::new();
        while let Some(gilrs::Event { event, .. }) = gilrs.next_event() {
            events.push(event);
        }
        for event in events {
            match event {
                gilrs::EventType::ButtonPressed(button, _) => {
                    self.input.set_source(InputSource::Gamepad(button), true);
                    if let Some((binding, _)) = self.launcher.remapping.clone() {
                        self.complete_remap(&binding, InputSource::Gamepad(button));
                    }
                }
                gilrs::EventType::ButtonReleased(button, _) => {
                    self.input.set_source(InputSource::Gamepad(button), false);
                }
                _ => {}
            }
        }
    }

    fn set_control_modifier(&mut self, binding: &str, modifier: ModifierKey) {
        if let Some(b) = self.control_binding_mut(binding) {
            b.modifier = modifier;
        }
        if let Some(ov) = self.control_override_mut(binding) {
            ov.modifier = Some(modifier.cfg_str().to_string());
        }
        self.persist_control_change();
    }

    fn set_control_trigger(&mut self, binding: &str, trigger: TriggerKind) {
        if let Some(b) = self.control_binding_mut(binding) {
            b.trigger = trigger;
        }
        if let Some(ov) = self.control_override_mut(binding) {
            ov.trigger = Some(trigger.cfg_str().to_string());
        }
        self.persist_control_change();
    }
    fn build_pause_ui(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(egui::Color32::from_black_alpha(180)))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(ui.available_height() / 3.0);

                    ui.heading("Paused");
                    ui.add_space(16.0);

                    let btn_size = egui::vec2(160.0, 32.0);

                    if ui
                        .add_sized(btn_size, egui::Button::new("Continue"))
                        .clicked()
                    {
                        self.state = AppState::InGame;
                        self.apply_cursor_state();
                    }

                    ui.add_space(8.0);

                    if ui
                        .add_sized(btn_size, egui::Button::new("Settings"))
                        .clicked()
                    {
                        self.pause_settings_open = !self.pause_settings_open;
                    }

                    ui.add_space(8.0);

                    if ui
                        .add_sized(btn_size, egui::Button::new("Controls"))
                        .clicked()
                    {
                        self.pause_controls_open = !self.pause_controls_open;
                    }

                    ui.add_space(8.0);

                    if ui
                        .add_sized(btn_size, egui::Button::new("Toggle Spectate"))
                        .clicked()
                    {
                        // Same discrete-event path a "spectate" keybind press
                        // would take (see InputTracker::update) — the guest
                        // treats a UI click and a key tap identically, both
                        // just a Pressed event for the "spectate" action.
                        cubic_wasm::push_input_event(cubic_wasm::InputEvent {
                            name: "spectate".to_string(),
                            kind: 0, // Pressed
                        });
                    }

                    ui.add_space(8.0);

                    if ui.add_sized(btn_size, egui::Button::new("Quit")).clicked() {
                        self.quit_requested = true;
                    }
                });
            });

        if self.pause_settings_open {
            egui::Window::new("Settings")
                .collapsible(false)
                .show(ctx, |ui| {
                    self.build_settings_tab(ui);
                });
        }

        if self.pause_controls_open {
            egui::Window::new("Controls")
                .collapsible(false)
                .show(ctx, |ui| {
                    self.build_controls_tab(ui);
                });
        }

        // Diagnostics overlay still visible while paused
        if self.show_diagnostics {
            self.build_diagnostics_ui(ctx);
        }
    }

    fn build_diagnostics_ui(&mut self, ctx: &egui::Context) {
        egui::Window::new("diagnostics")
            .title_bar(false)
            .resizable(false)
            .anchor(egui::Align2::LEFT_TOP, egui::vec2(8.0, 8.0))
            .frame(
                egui::Frame::new()
                    .fill(egui::Color32::from_black_alpha(160))
                    .inner_margin(8.0),
            )
            .show(ctx, |ui| {
                ui.style_mut().visuals.override_text_color = Some(egui::Color32::WHITE);

                // FPS and frame time
                let fps = self.last_fps;
                let frame_ms = self.last_frame_dt * 1000.0;
                ui.label(format!("{fps} fps  {frame_ms:.2}ms"));

                // Position — feet, not the camera, when a WASM game is
                // driving: third-person orbit moves the camera away from
                // the player, so camera.position alone would show orbit
                // position instead of where the player actually is. In
                // free-fly mode (no game loaded) there's no feet position
                // to report, so fall back to the camera as before.
                let p = if self.wasm_game.is_some() {
                    let feet = cubic_wasm::get_player_feet();
                    Vec3::new(feet.x, feet.y, feet.z)
                } else {
                    self.camera.position
                };
                ui.label(format!("XYZ: {:.1} / {:.1} / {:.1}", p.x, p.y, p.z));

                // Facing
                let yaw_deg = self.camera.yaw.to_degrees().rem_euclid(360.0);
                let pitch_deg = self.camera.pitch.to_degrees();
                let cardinal = match yaw_deg as u32 {
                    315..=360 | 0..=44 => "N",
                    45..=134 => "E",
                    135..=224 => "S",
                    _ => "W",
                };
                ui.label(format!(
                    "Facing: {cardinal} ({yaw_deg:.1} / {pitch_deg:.1})"
                ));

                // Chunk stats
                let loaded = self.chunk_meshes.len();
                let pending = self.stream.ready_meshes.len();
                ui.label(format!("Chunks: {loaded} loaded  {pending} pending"));

                // Block position (which voxel the camera is in)
                let voxel_x = (p.x / cubic_world::VOXEL_SIZE).floor() as i32;
                let voxel_y = (p.y / cubic_world::VOXEL_SIZE).floor() as i32;
                let voxel_z = (p.z / cubic_world::VOXEL_SIZE).floor() as i32;
                ui.label(format!("Block: {voxel_x} {voxel_y} {voxel_z}"));
            });
    }
}

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let event_loop: EventLoop<()> = EventLoop::new()?;

    let game_name = "cubic-game".to_string();
    let profile_name = "default".to_string();
    let current_profile = profile::load_or_create(&game_name, &profile_name).unwrap_or_default();
    tracing::info!(
        "profile: {}",
        profile::profile_toml_path(&game_name, &profile_name).display()
    );
    // Create the XDG directory structure at startup so it exists even if
    // empty; the user-mods layer lands in a future card.
    let _ = std::fs::create_dir_all(profile::user_games_dir());
    let _ = std::fs::create_dir_all(profile::user_mods_dir());

    // Resolution chain: cubic.toml (global) -> game_overrides.toml (game) ->
    // profile.toml (user). game.path only ever comes from cubic.toml, so
    // it's already known from this same load_cfg() call — no need to read
    // and parse cubic.toml a second time just to find game_dir.
    let base_cfg = load_cfg();
    let game_dir = std::path::Path::new(&base_cfg.game.path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let game_overrides = game_override::load(&game_dir);

    let cfg = apply_profile(
        apply_game_override(base_cfg, &game_overrides),
        &current_profile,
    );
    let controls = resolve_controls(&cfg);
    let custom_controls = build_custom_controls(&game_overrides, &current_profile);

    // Placeholder until load_world() (called from handle_launch()) computes
    // the real one and rebuilds the plugin/generator with it baked in — see
    // the comment on App::plugin. Not used for anything before then.
    let seed = if cfg.world.seed == 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos() as u64
    } else {
        cfg.world.seed
    };

    // Remembered from a previous launch, if this profile has ever saved one
    // (see handle_launch/persist_window_prefs); otherwise sensible defaults.
    let remembered_window = current_profile.window.as_ref();
    let window_mode = remembered_window
        .and_then(|w| w.mode.as_deref())
        .and_then(str_to_window_mode)
        .unwrap_or(WindowMode::Windowed);
    let window_width_str = remembered_window
        .and_then(|w| w.width)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "1280".to_string());
    let window_height_str = remembered_window
        .and_then(|w| w.height)
        .map(|v| v.to_string())
        .unwrap_or_else(|| "720".to_string());

    let launcher = LauncherState {
        selected_game: game_name.clone(),
        available_games: scan_games(),
        selected_profile: profile_name.clone(),
        available_profiles: profile::list_profiles(&game_name),
        seed_str: cfg.world.seed.to_string(),
        window_mode,
        window_width_str,
        window_height_str,
        settings_open: false,
        remapping: None,
    };

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
        generator: None,
        plugin: None,
        wasm_game: None,
        chunk_meshes: HashMap::new(),
        seed,
        cfg,
        current_profile,
        current_profile_name: profile_name,
        current_game_name: game_name,
        game_overrides,
        custom_controls: custom_controls.clone(),
        launcher,
        launcher_tab: LauncherTab::Game,
        pause_settings_open: false,
        pause_controls_open: false,
        pending_windowed_resize: None,
        exiting: false,
        quit_requested: false,
        frames: 0,
        last_fps: 0,
        last_fps_instant: std::time::Instant::now(),
        paused: false,
        focused: true,
        next_frame_deadline: None,
        state: AppState::Launcher,
        egui_ctx: egui::Context::default(),
        egui_winit: None,
        show_diagnostics: false,
        controls,
        camera: Camera {
            position: Vec3::new(0.0, (CHUNK_SIZE / 2) as f32 * VOXEL_SIZE + 12.0, 0.0),
            pitch: -0.3,
            ..Camera::default()
        },
        input: InputState::default(),
        last_frame_instant: std::time::Instant::now(),
        last_frame_dt: 0.0,
        detected_refresh_hz: 60.0, // overwritten in resumed()
        remesh_scratch: HashSet::new(),
        tex_map: HashMap::new(),
        face_textures: Arc::new(BlockFaceTextures::new()), // populated in resumed()
        entity_meshes: HashMap::new(),
        next_entity_mesh_id: 1,
        input_tracker: InputTracker::new(&controls, &custom_controls),
        gilrs: gilrs::Gilrs::new()
            .inspect_err(|e| tracing::warn!("gamepad support unavailable: {e}"))
            .ok(),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(key: KeyCode, modifier: ModifierKey, trigger: TriggerKind) -> ResolvedBinding {
        ResolvedBinding {
            source: Some(InputSource::Key(key)),
            modifier,
            trigger,
        }
    }

    fn tracker_for(name: &str, binding: ResolvedBinding) -> InputTracker {
        InputTracker {
            actions: vec![(
                name.to_string(),
                binding,
                ActionTracker {
                    was_held: false,
                    last_press_time: -1.0,
                },
            )],
            elapsed: 0.0,
        }
    }

    fn press(input: &mut InputState, key: KeyCode, pressed: bool) {
        input.set_source(InputSource::Key(key), pressed);
    }

    #[test]
    fn tap_trigger_fires_once_on_press_not_while_held() {
        let mut tracker = tracker_for(
            "x",
            binding(KeyCode::KeyF, ModifierKey::None, TriggerKind::Tap),
        );
        let mut input = InputState::default();

        press(&mut input, KeyCode::KeyF, true);
        assert_eq!(tracker.update(&input, 0.016), vec!["x".to_string()]);

        // Still held on the next tick — must not re-fire.
        assert!(tracker.update(&input, 0.016).is_empty());
    }

    #[test]
    fn double_tap_trigger_suppresses_a_lone_tap() {
        let mut tracker = tracker_for(
            "x",
            binding(KeyCode::Space, ModifierKey::None, TriggerKind::DoubleTap),
        );
        let mut input = InputState::default();

        press(&mut input, KeyCode::Space, true);
        assert!(
            tracker.update(&input, 0.05).is_empty(),
            "a lone tap must not fire a DoubleTap-configured control \
             (this is the exact bug being fixed: toggle_third_person set to \
             DoubleTap used to fire on a single press)"
        );
    }

    #[test]
    fn double_tap_trigger_fires_on_a_genuine_rapid_double_press() {
        let mut tracker = tracker_for(
            "x",
            binding(KeyCode::Space, ModifierKey::None, TriggerKind::DoubleTap),
        );
        let mut input = InputState::default();

        press(&mut input, KeyCode::Space, true);
        assert!(tracker.update(&input, 0.05).is_empty()); // elapsed 0.05

        press(&mut input, KeyCode::Space, false);
        tracker.update(&input, 0.05); // elapsed 0.10, Released

        press(&mut input, KeyCode::Space, true); // second press 0.10s after the first
        assert_eq!(
            tracker.update(&input, 0.05), // elapsed 0.15
            vec!["x".to_string()],
            "a rapid second press within the 0.3s window must fire"
        );
    }

    #[test]
    fn double_tap_trigger_does_not_fire_on_two_slow_taps() {
        let mut tracker = tracker_for(
            "x",
            binding(KeyCode::Space, ModifierKey::None, TriggerKind::DoubleTap),
        );
        let mut input = InputState::default();

        press(&mut input, KeyCode::Space, true);
        tracker.update(&input, 0.05); // elapsed 0.05

        press(&mut input, KeyCode::Space, false);
        tracker.update(&input, 0.05); // elapsed 0.10

        press(&mut input, KeyCode::Space, true); // second press ~1s after the first
        assert!(
            tracker.update(&input, 1.0).is_empty(), // elapsed 1.10
            "two taps more than 0.3s apart must not count as a double-tap"
        );
    }

    #[test]
    fn modifier_gates_activation_and_is_side_agnostic() {
        let b = binding(KeyCode::F6, ModifierKey::Shift, TriggerKind::Tap);
        let mut input = InputState::default();

        press(&mut input, KeyCode::F6, true);
        assert!(
            !input.binding_active(&b),
            "key alone shouldn't activate a Shift-gated binding"
        );

        press(&mut input, KeyCode::ShiftLeft, true);
        assert!(
            input.binding_active(&b),
            "key + ShiftLeft should activate it"
        );

        press(&mut input, KeyCode::ShiftLeft, false);
        press(&mut input, KeyCode::ShiftRight, true);
        assert!(
            input.binding_active(&b),
            "either shift side should satisfy a generic Shift modifier"
        );
    }

    #[test]
    fn unbound_binding_never_activates() {
        let b = ResolvedBinding {
            source: None,
            modifier: ModifierKey::None,
            trigger: TriggerKind::Tap,
        };
        let mut input = InputState::default();
        press(&mut input, KeyCode::F6, true); // holding some unrelated key
        assert!(!input.binding_active(&b));
    }

    #[test]
    fn mouse_button_activates_a_binding_the_same_as_a_key() {
        let b = ResolvedBinding {
            source: Some(InputSource::Mouse(MouseButton::Right)),
            modifier: ModifierKey::None,
            trigger: TriggerKind::Tap,
        };
        let mut input = InputState::default();
        assert!(!input.binding_active(&b));

        input.set_source(InputSource::Mouse(MouseButton::Right), true);
        assert!(input.binding_active(&b));

        input.set_source(InputSource::Mouse(MouseButton::Right), false);
        assert!(!input.binding_active(&b));
    }

    #[test]
    fn gamepad_button_activates_a_binding_the_same_as_a_key() {
        let b = ResolvedBinding {
            source: Some(InputSource::Gamepad(gilrs::Button::South)),
            modifier: ModifierKey::None,
            trigger: TriggerKind::Tap,
        };
        let mut input = InputState::default();
        input.set_source(InputSource::Gamepad(gilrs::Button::South), true);
        assert!(input.binding_active(&b));
    }

    #[test]
    fn input_source_round_trips_through_config_strings() {
        for source in [
            InputSource::Key(KeyCode::KeyW),
            InputSource::Mouse(MouseButton::Left),
            InputSource::Mouse(MouseButton::Other(7)),
            InputSource::Gamepad(gilrs::Button::South),
            InputSource::Gamepad(gilrs::Button::DPadUp),
        ] {
            let s = input_source_to_string(source).expect("source should stringify");
            assert_eq!(
                str_to_input_source(&s),
                Some(source),
                "round trip failed for {s}"
            );
        }
    }

    #[test]
    fn trigger_kind_has_no_hold_variant() {
        // Regression guard for the removed Hold trigger: legacy/empty
        // bindings and KeyBinding::key's default must land on Tap instead
        // (Hold and Tap fired identically anyway, so this is a pure label
        // change, not a behavior change).
        assert_eq!(KeyBinding::key("KeyW").trigger, TriggerKind::Tap);
        assert_eq!(
            KeyBinding::unbound(TriggerKind::Tap).trigger,
            TriggerKind::Tap
        );
    }

    #[test]
    fn custom_control_resolves_game_default_with_no_profile_override() {
        let overrides = game_override::GameOverrideCfg {
            controls: vec![game_override::CustomControlDef {
                name: "sprint".to_string(),
                label: Some("Sprint".to_string()),
                key: Some("KeyW".to_string()),
                modifier: None,
                trigger: Some("double_tap".to_string()),
            }],
            ..Default::default()
        };
        let profile = profile::ProfileCfg::default();

        let custom = build_custom_controls(&overrides, &profile);
        assert_eq!(custom.len(), 1);
        assert_eq!(custom[0].name, "sprint");
        assert_eq!(custom[0].label, "Sprint");
        assert_eq!(custom[0].binding.key.as_deref(), Some("KeyW"));
        assert_eq!(custom[0].binding.trigger, TriggerKind::DoubleTap);
        assert_eq!(custom[0].binding.modifier, ModifierKey::None);
    }

    #[test]
    fn custom_control_profile_override_takes_precedence_over_game_default() {
        let overrides = game_override::GameOverrideCfg {
            controls: vec![game_override::CustomControlDef {
                name: "sprint".to_string(),
                label: None,
                key: Some("KeyW".to_string()),
                modifier: None,
                trigger: Some("double_tap".to_string()),
            }],
            ..Default::default()
        };
        let mut custom_override = std::collections::HashMap::new();
        custom_override.insert(
            "sprint".to_string(),
            profile::KeyBindingOverride {
                key: Some("KeyR".to_string()),
                modifier: None,
                trigger: Some("tap".to_string()),
            },
        );
        let profile = profile::ProfileCfg {
            controls: Some(profile::ControlsOverride {
                custom: custom_override,
                ..Default::default()
            }),
            ..Default::default()
        };

        let custom = build_custom_controls(&overrides, &profile);
        assert_eq!(custom.len(), 1);
        // Falls back to the control's own name when no label is declared.
        assert_eq!(custom[0].label, "sprint");
        assert_eq!(custom[0].binding.key.as_deref(), Some("KeyR"));
        assert_eq!(custom[0].binding.trigger, TriggerKind::Tap);
    }
}
