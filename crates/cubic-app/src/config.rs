// SPDX-License-Identifier: CEPL-1.0
//! cubic.toml config structs, profile/game-override resolution, and
//! persistence (save_global_cfg).

use serde::{Deserialize, Serialize};
use std::fs;

use crate::{game_override, profile};

#[derive(Debug, Deserialize, Serialize, Default)]
pub(crate) struct AppCfg {
    #[serde(default)]
    pub(crate) render: RenderCfg,
    #[serde(default)]
    pub(crate) world: WorldCfg,
    #[serde(default)]
    pub(crate) camera: CameraCfg,
    #[serde(default)]
    pub(crate) player: PlayerCfg,
    #[serde(default)]
    pub(crate) game: GameCfg,
    #[serde(default)]
    pub(crate) controls: ControlsCfg,
    #[serde(default)]
    pub(crate) launcher: LauncherCfg,
    #[serde(default)]
    pub(crate) ui: UiCfg,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum VsyncMode {
    Fifo,
    #[default]
    Mailbox,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum UnfocusedPolicy {
    None,
    #[default]
    VsyncOn,
    Throttle,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub(crate) enum HdrFlavorCfg {
    #[default]
    PreferScrgb,
    PreferHdr10,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TextureFilter {
    Nearest,
    #[default]
    Linear,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Default, PartialEq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum MipmapMode {
    Nearest,
    #[default]
    Linear,
}

#[derive(Debug, Deserialize, Serialize, Clone, Copy)]
pub(crate) struct RenderCfg {
    #[serde(default = "default_clear")]
    pub(crate) clear_color: [f32; 4],
    #[serde(default = "default_vsync")]
    pub(crate) vsync: bool,
    #[serde(default)]
    pub(crate) vsync_mode: VsyncMode,
    #[serde(default)]
    pub(crate) unfocused: UnfocusedPolicy,
    #[serde(default)]
    pub(crate) unfocused_fps: u32,
    #[serde(default)]
    pub(crate) fps_when_vsync_off: u32,
    #[serde(default)]
    pub(crate) hdr: bool,
    #[serde(default)]
    pub(crate) hdr_flavor: HdrFlavorCfg,
    #[serde(default)]
    pub(crate) texture_filter: TextureFilter,
    #[serde(default)]
    pub(crate) mipmap_mode: MipmapMode,
    #[serde(default = "default_anisotropy")]
    pub(crate) anisotropy: f32,
    #[serde(default)]
    pub(crate) lod_bias: f32,
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
pub(crate) fn load_cfg() -> AppCfg {
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
pub(crate) fn apply_game_override(
    mut cfg: AppCfg,
    overrides: &game_override::GameOverrideCfg,
) -> AppCfg {
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
pub(crate) fn apply_profile(mut cfg: AppCfg, profile: &profile::ProfileCfg) -> AppCfg {
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
    if let Some(u) = &profile.ui {
        if let Some(v) = &u.crosshair_path {
            cfg.ui.crosshair_path = v.clone();
        }
        if let Some(v) = u.crosshair_size {
            cfg.ui.crosshair_size = v;
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
pub(crate) struct CustomControl {
    pub(crate) name: String,
    pub(crate) label: String,
    pub(crate) binding: KeyBinding,
}

/// Resolve `overrides`' declared custom controls (defaults) against
/// `profile`'s sparse per-control overrides, the same two-layer resolution
/// built-in controls get from cubic.toml + profile.toml — minus a global
/// cubic.toml layer, since the engine has no default for a control it
/// doesn't know exists.
pub(crate) fn build_custom_controls(
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
pub(crate) struct WorldCfg {
    #[serde(default = "default_stream_radius")]
    pub(crate) stream_radius: i32,
    #[serde(default)]
    pub(crate) seed: u64,
    #[serde(default)] // 0.0 = auto
    pub(crate) upload_budget_ms: f32,
    #[serde(default = "default_upload_budget_min_ms")]
    pub(crate) upload_budget_min_ms: f32,
    #[serde(default = "default_stream_radius_y")]
    pub(crate) stream_radius_y: i32,
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
pub(crate) struct CameraCfg {
    #[serde(default = "default_move_speed")]
    pub(crate) move_speed: f32,
    #[serde(default = "default_mouse_sensitivity")]
    pub(crate) mouse_sensitivity: f32,
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
pub(crate) struct PlayerCfg {
    #[serde(default = "default_walk_speed")]
    pub(crate) walk_speed: f32,
    #[serde(default = "default_fly_speed")]
    pub(crate) fly_speed: f32,
    #[serde(default = "default_jump_velocity")]
    pub(crate) jump_velocity: f32,
    #[serde(default = "default_gravity")]
    pub(crate) gravity: f32,
    #[serde(default = "default_sprint_multiplier")]
    pub(crate) sprint_multiplier: f32,
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

fn default_crosshair_path() -> String {
    "assets/ui/crosshair.png".to_string()
}
fn default_crosshair_size() -> f32 {
    32.0
}

/// In-game HUD appearance. `crosshair_path` is resolved relative to the
/// engine's working directory (same convention as `game.path`) — swapping
/// in a different image is the entire "make your own crosshair" story for
/// now, no in-engine editor. See tools/gen_crosshair.py for how the bundled
/// default was made, including why it's shipped at a much higher
/// resolution (256x256) than its default on-screen size.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct UiCfg {
    #[serde(default = "default_crosshair_path")]
    pub(crate) crosshair_path: String,
    #[serde(default = "default_crosshair_size")]
    pub(crate) crosshair_size: f32,
}

impl Default for UiCfg {
    fn default() -> Self {
        UiCfg {
            crosshair_path: default_crosshair_path(),
            crosshair_size: default_crosshair_size(),
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
pub(crate) struct GameCfg {
    #[serde(default = "default_game_path")]
    pub(crate) path: String,
    #[serde(default = "default_wasm_memory_mb")]
    pub(crate) wasm_memory_mb: usize,
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
pub(crate) struct LauncherCfg {
    #[serde(default = "default_launcher_width")]
    pub(crate) width: u32,
    #[serde(default = "default_launcher_height")]
    pub(crate) height: u32,
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
pub(crate) enum ModifierKey {
    #[default]
    None,
    Shift,
    Control,
    Alt,
}

impl ModifierKey {
    pub(crate) fn label(self) -> &'static str {
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
    pub(crate) fn cfg_str(self) -> &'static str {
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
pub(crate) enum TriggerKind {
    #[default]
    Tap,
    DoubleTap,
}

impl TriggerKind {
    pub(crate) fn label(self) -> &'static str {
        match self {
            TriggerKind::Tap => "tap",
            TriggerKind::DoubleTap => "double tap",
        }
    }

    /// Config-string form — see ModifierKey::cfg_str's doc comment.
    pub(crate) fn cfg_str(self) -> &'static str {
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
pub(crate) struct KeyBinding {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) key: Option<String>,
    #[serde(default, skip_serializing_if = "is_default_modifier")]
    pub(crate) modifier: ModifierKey,
    #[serde(default, skip_serializing_if = "is_default_trigger")]
    pub(crate) trigger: TriggerKind,
}

fn is_default_modifier(m: &ModifierKey) -> bool {
    *m == ModifierKey::None
}
fn is_default_trigger(t: &TriggerKind) -> bool {
    *t == TriggerKind::Tap
}

impl KeyBinding {
    pub(crate) fn unbound(trigger: TriggerKind) -> Self {
        KeyBinding {
            key: None,
            modifier: ModifierKey::None,
            trigger,
        }
    }
    pub(crate) fn key(key: &str) -> Self {
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
pub(crate) struct ControlsCfg {
    #[serde(default = "default_forward")]
    pub(crate) forward: KeyBinding,
    #[serde(default = "default_back")]
    pub(crate) back: KeyBinding,
    #[serde(default = "default_left")]
    pub(crate) left: KeyBinding,
    #[serde(default = "default_right")]
    pub(crate) right: KeyBinding,
    #[serde(default = "default_jump")]
    pub(crate) jump: KeyBinding,
    #[serde(default = "default_sneak")]
    pub(crate) sneak: KeyBinding,
    #[serde(default = "default_toggle_diagnostics")]
    pub(crate) toggle_diagnostics: KeyBinding,
    #[serde(default = "default_toggle_third_person")]
    pub(crate) toggle_third_person: KeyBinding,
    // Deliberately unbound by default — a nice-to-have utility mode, not
    // something every player needs a key eaten for out of the box. The
    // pause-menu "Toggle Spectate" button works regardless.
    #[serde(default = "default_spectate")]
    pub(crate) spectate: KeyBinding,
    // Same physical key as jump by default, and that's fine — jump's own
    // ResolvedBinding is checked independently for the continuous
    // ascend/grounded-jump boolean; this is a separate DoubleTap-triggered
    // binding on top, not a conflict.
    #[serde(default = "default_fly")]
    pub(crate) fly: KeyBinding,
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
pub(crate) fn save_global_cfg(cfg: &AppCfg) {
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
