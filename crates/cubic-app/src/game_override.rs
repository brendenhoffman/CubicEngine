// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use serde::Deserialize;
use std::path::Path;

// Schema — same sparse Option<T> pattern as ProfileCfg but no controls
// section (games cannot override controls).

#[derive(Debug, Deserialize, Default, Clone)]
pub struct GameOverrideCfg {
    #[serde(default)]
    pub render: Option<GameRenderOverride>,
    #[serde(default)]
    pub world: Option<GameWorldOverride>,
    #[serde(default)]
    pub camera: Option<GameCameraOverride>,
    // no controls field — games cannot remap engine controls
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
