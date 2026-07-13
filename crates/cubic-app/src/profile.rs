// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Directory layout
// ---------------------------------------------------------------------------

/// Returns $XDG_DATA_HOME/CubicEngine or equivalent on each platform.
pub fn data_root() -> std::path::PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("CubicEngine")
}

pub fn profiles_dir(game_name: &str) -> std::path::PathBuf {
    data_root().join("profiles").join(game_name)
}

pub fn profile_dir(game_name: &str, profile_name: &str) -> std::path::PathBuf {
    profiles_dir(game_name).join(profile_name)
}

pub fn profile_toml_path(game_name: &str, profile_name: &str) -> std::path::PathBuf {
    profile_dir(game_name, profile_name).join("profile.toml")
}

pub fn user_games_dir() -> std::path::PathBuf {
    data_root().join("games")
}

pub fn user_mods_dir() -> std::path::PathBuf {
    data_root().join("mods")
}

pub fn worlds_dir(game_name: &str, profile_name: &str) -> std::path::PathBuf {
    profile_dir(game_name, profile_name).join("worlds")
}

pub fn world_dir(game_name: &str, profile_name: &str, world_name: &str) -> std::path::PathBuf {
    worlds_dir(game_name, profile_name).join(world_name)
}

pub fn world_toml_path(
    game_name: &str,
    profile_name: &str,
    world_name: &str,
) -> std::path::PathBuf {
    world_dir(game_name, profile_name, world_name).join("world.toml")
}

// ---------------------------------------------------------------------------
// Profile TOML schema — sparse override, every field Option<T>
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct ProfileCfg {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render: Option<RenderOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub world: Option<WorldOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera: Option<CameraOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub player: Option<PlayerOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub controls: Option<ControlsOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui: Option<UiOverride>,
    // Not an AppCfg override like the sections above — window mode/size are
    // launcher-only UI state with no corresponding engine config field —
    // but profile.toml is the natural place to remember them per profile.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<WindowPrefs>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct WindowPrefs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>, // "windowed" | "maximized" | "fullscreen"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct RenderOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vsync: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub texture_filter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mipmap_mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anisotropy: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lod_bias: Option<f32>,
    // add other RenderCfg fields as Option here
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct WorldOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_radius: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_radius_y: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upload_budget_ms: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_world: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct CameraOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub move_speed: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mouse_sensitivity: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct PlayerOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub walk_speed: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fly_speed: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jump_velocity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gravity: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sprint_multiplier: Option<f32>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct UiOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crosshair_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crosshair_size: Option<f32>,
}

/// Sparse override for one control's binding. All three parts are
/// independently optional so e.g. changing just the trigger kind in the
/// launcher doesn't need to also know/rewrite the current key — main.rs's
/// apply_profile applies whichever parts are present onto the resolved
/// cubic.toml binding. Plain strings (not main.rs's `ModifierKey`/
/// `TriggerKind` enums) to keep this crate decoupled from AppCfg's types,
/// same reasoning as the rest of this sparse-override schema; parsed via
/// main.rs's parse_cfg_str, same as texture_filter/mipmap_mode overrides.
#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct KeyBindingOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub modifier: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Default, Clone)]
pub struct ControlsOverride {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forward: Option<KeyBindingOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub back: Option<KeyBindingOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub left: Option<KeyBindingOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub right: Option<KeyBindingOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub jump: Option<KeyBindingOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sneak: Option<KeyBindingOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toggle_diagnostics: Option<KeyBindingOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toggle_third_person: Option<KeyBindingOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spectate: Option<KeyBindingOverride>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fly: Option<KeyBindingOverride>,
    /// Overrides for controls a game registers itself (see
    /// game_override::CustomControlDef), keyed by control name — a sparse
    /// map instead of named fields since the set of names isn't known
    /// statically the way the built-ins above are.
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub custom: std::collections::HashMap<String, KeyBindingOverride>,
}

/// Immutable metadata written once when a world is created and never
/// overwritten. Read back to show seed/date in the world picker UI.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorldToml {
    pub seed: u64,
    pub generator: String,
    pub created_at: String, // RFC 3339
    pub engine_version: String,
}

// ---------------------------------------------------------------------------
// Profile loading and creation
// ---------------------------------------------------------------------------

/// Load profile from disk. Creates the directory and an empty profile.toml
/// if they don't exist yet. Returns the loaded (possibly empty) ProfileCfg.
pub fn load_or_create(game_name: &str, profile_name: &str) -> anyhow::Result<ProfileCfg> {
    let dir = profile_dir(game_name, profile_name);
    std::fs::create_dir_all(&dir)?;

    let path = profile_toml_path(game_name, profile_name);
    if !path.exists() {
        std::fs::write(&path, "")?;
        tracing::info!("created new profile at {}", path.display());
        return Ok(ProfileCfg::default());
    }

    let s = std::fs::read_to_string(&path)?;
    let cfg = toml::from_str(&s)?;
    Ok(cfg)
}

/// Save profile to disk. Only writes fields that are Some (sparse).
pub fn save(profile: &ProfileCfg, game_name: &str, profile_name: &str) -> anyhow::Result<()> {
    let path = profile_toml_path(game_name, profile_name);
    std::fs::create_dir_all(path.parent().unwrap())?;
    let s = toml::to_string_pretty(profile)?;
    std::fs::write(&path, s)?;
    Ok(())
}

/// List profile names for a game. Returns empty vec if no profiles exist yet.
pub fn list_profiles(game_name: &str) -> Vec<String> {
    let dir = profiles_dir(game_name);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return vec![];
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

pub fn write_world_toml(
    game_name: &str,
    profile_name: &str,
    world_name: &str,
    meta: &WorldToml,
) -> anyhow::Result<()> {
    let path = world_toml_path(game_name, profile_name, world_name);
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, toml::to_string_pretty(meta)?)?;
    Ok(())
}

pub fn read_world_toml(
    game_name: &str,
    profile_name: &str,
    world_name: &str,
) -> anyhow::Result<WorldToml> {
    let path = world_toml_path(game_name, profile_name, world_name);
    let s = std::fs::read_to_string(&path)?;
    Ok(toml::from_str(&s)?)
}

/// List world names for a profile. Returns empty vec if none exist yet.
pub fn list_worlds(game_name: &str, profile_name: &str) -> Vec<String> {
    let dir = worlds_dir(game_name, profile_name);
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return vec![];
    };
    let mut names: Vec<String> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

/// Formats a Unix timestamp as a bare RFC 3339 UTC string without pulling
/// in chrono. Accurate for dates from 1970 through ~2099.
pub(crate) fn format_unix_as_rfc3339(secs: u64) -> String {
    // Days since epoch
    let mut days = secs / 86400;
    let time_of_day = secs % 86400;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;

    // Gregorian calendar from day count
    let mut year = 1970u64;
    loop {
        let leap =
            year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_per_month = [
        31u64,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u64;
    for &dim in &days_per_month {
        if days < dim {
            break;
        }
        days -= dim;
        month += 1;
    }
    let day = days + 1;

    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}
