// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use serde::Deserialize;
use std::path::Path;

// Schema — same sparse Option<T> pattern as ProfileCfg. `controls` is the
// one exception to "sparse override of an existing value": games still
// cannot remap forward/jump/etc, but they can *register new* controls of
// their own (see CustomControlDef).

#[derive(Debug, Deserialize, Default, Clone)]
pub struct GameOverrideCfg {
    #[serde(default)]
    pub render: Option<GameRenderOverride>,
    #[serde(default)]
    pub world: Option<GameWorldOverride>,
    #[serde(default)]
    pub camera: Option<GameCameraOverride>,
    #[serde(default)]
    pub player: Option<GamePlayerOverride>,
    // Controls a game *defines itself* (e.g. "sprint"), not an override of
    // an existing engine control — games still cannot remap forward/jump/
    // etc, there's just no field for that here at all.
    #[serde(default)]
    pub controls: Vec<CustomControlDef>,
}

/// A single custom control this game wants registered, declared as
/// `[[controls]]` entries in game_overrides.toml. `key`/`modifier`/
/// `trigger` are plain strings (not main.rs's `ModifierKey`/`TriggerKind`
/// enums) to keep this crate decoupled from main.rs's types, same
/// reasoning as `profile::KeyBindingOverride` — parsed via main.rs's
/// parse_cfg_str in `build_custom_controls`. Unset `key` means the control
/// starts out unbound rather than defaulting to some key that might already
/// be taken.
#[derive(Debug, Deserialize, Clone)]
pub struct CustomControlDef {
    /// Also the InputEvent name forwarded to the guest — must match
    /// whatever the game's own on_tick matches on.
    pub name: String,
    /// Controls-tab display label; falls back to `name` if unset.
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub key: Option<String>,
    #[serde(default)]
    pub modifier: Option<String>,
    #[serde(default)]
    pub trigger: Option<String>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct GameRenderOverride {
    #[serde(default)]
    pub vsync: Option<bool>,
    #[serde(default)]
    pub texture_filter: Option<String>,
    #[serde(default)]
    pub mipmap_mode: Option<String>,
    #[serde(default)]
    pub anisotropy: Option<f32>,
    #[serde(default)]
    pub lod_bias: Option<f32>,
    #[serde(default)]
    pub clear_color: Option<[f32; 4]>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct GameWorldOverride {
    #[serde(default)]
    pub stream_radius: Option<i32>,
    #[serde(default)]
    pub stream_radius_y: Option<i32>,
    #[serde(default)]
    pub upload_budget_ms: Option<f32>,
    // seed intentionally omitted — seed is user/profile territory
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct GameCameraOverride {
    #[serde(default)]
    pub move_speed: Option<f32>,
    #[serde(default)]
    pub mouse_sensitivity: Option<f32>,
}

#[derive(Debug, Deserialize, Default, Clone)]
pub struct GamePlayerOverride {
    #[serde(default)]
    pub walk_speed: Option<f32>,
    #[serde(default)]
    pub fly_speed: Option<f32>,
    #[serde(default)]
    pub jump_velocity: Option<f32>,
    #[serde(default)]
    pub gravity: Option<f32>,
    #[serde(default)]
    pub sprint_multiplier: Option<f32>,
}

/// Load game override config from {game_dir}/game_overrides.toml.
/// Returns empty override (no-op) if the file doesn't exist.
/// The file is optional — games only ship it if they need non-default settings.
pub fn load(game_dir: &Path) -> GameOverrideCfg {
    let path = game_dir.join("game_overrides.toml");
    if !path.exists() {
        return GameOverrideCfg::default();
    }
    match std::fs::read_to_string(&path) {
        Ok(s) => toml::from_str(&s).unwrap_or_else(|e| {
            tracing::warn!("failed to parse game_overrides.toml: {e}");
            GameOverrideCfg::default()
        }),
        Err(e) => {
            tracing::warn!("failed to read game_overrides.toml: {e}");
            GameOverrideCfg::default()
        }
    }
}
