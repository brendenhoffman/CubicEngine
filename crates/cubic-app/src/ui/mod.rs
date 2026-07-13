// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! egui UI: launcher/pause/diagnostics screens and the shared launcher
//! state/types they operate on.

mod launcher;
mod pause;

pub(crate) use launcher::scan_games;
pub(crate) mod chat;
pub(crate) use chat::{ChatMessage, ChatMessageKind};
pub(crate) mod input_bar;

use crate::{profile, App};

/// Transient launcher UI state — not persisted directly; committed to
/// cfg/profile.toml as the user interacts (see handle_launch,
/// apply_control_remap).
pub(crate) struct LauncherState {
    pub(crate) selected_game: String,
    pub(crate) available_games: Vec<GameEntry>,
    pub(crate) selected_profile: String,
    pub(crate) available_profiles: Vec<String>,
    pub(crate) window_mode: WindowMode,
    pub(crate) window_width_str: String,
    pub(crate) window_height_str: String,
    // Not yet read anywhere: reserved for a future collapsible/overlay
    // settings affordance within the launcher itself, distinct from the
    // Settings tab already wired up below.
    #[allow(dead_code)]
    pub(crate) settings_open: bool,
    // Which control is being remapped, if any, plus when capture started —
    // capture accepts a key, mouse button, or gamepad button (see
    // window_event's pre-egui interception and App::poll_gamepads) and
    // auto-cancels after REMAP_TIMEOUT if nothing is pressed.
    pub(crate) remapping: Option<(String, std::time::Instant)>,
    // Worlds tab
    pub(crate) world_list: Vec<WorldEntry>,
    pub(crate) new_world_name: String,
    pub(crate) new_world_seed_str: String,
    pub(crate) renaming: Option<(String, String)>, // (old_name, draft_name)
    pub(crate) pending_delete: Option<String>,     // world name awaiting confirm
    pub(crate) worlds_error: Option<String>,       // inline validation error
}

pub(crate) const REMAP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(8);

#[derive(Clone)]
pub(crate) struct GameEntry {
    pub(crate) name: String,         // directory name, used as game_name key
    pub(crate) display_name: String, // from game.toml metadata if present, else same as name
    // Not yet read anywhere: selecting a game in the launcher only updates
    // `selected_game`/`available_profiles` today (see build_game_tab) —
    // actually switching games at Launch time (reconstructing the WASM
    // plugin from this path) is future work.
    #[allow(dead_code)]
    pub(crate) path: std::path::PathBuf, // full path to game directory
}

#[derive(Clone)]
pub(crate) struct WorldEntry {
    pub(crate) name: String,
    /// None if world.toml is missing or malformed
    pub(crate) meta: Option<profile::WorldToml>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum WindowMode {
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
pub(crate) enum PendingWindowedResize {
    AwaitingMaximizeConfirm { width: u32, height: u32 },
    MaximizeConfirmed { width: u32, height: u32 },
    AwaitingUnmaximizeConfirm { width: u32, height: u32 },
    UnmaximizeConfirmed { width: u32, height: u32 },
}

pub(crate) fn window_mode_to_str(mode: WindowMode) -> &'static str {
    match mode {
        WindowMode::Windowed => "windowed",
        WindowMode::Maximized => "maximized",
        WindowMode::Fullscreen => "fullscreen",
    }
}

pub(crate) fn str_to_window_mode(s: &str) -> Option<WindowMode> {
    match s {
        "windowed" => Some(WindowMode::Windowed),
        "maximized" => Some(WindowMode::Maximized),
        "fullscreen" => Some(WindowMode::Fullscreen),
        _ => None,
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum LauncherTab {
    Game,
    Profile,
    Settings,
    Controls,
    Worlds,
}

impl App {
    pub(crate) fn build_ui(&mut self, ui: &mut egui::Ui) {
        match self.state {
            crate::AppState::Launcher => self.build_launcher_ui(ui),
            crate::AppState::Paused => {
                self.build_pause_ui(ui);
                self.build_chat_ui(ui.ctx());
                if self.chat_submit_pending {
                    self.chat_submit_pending = false;
                    self.submit_chat();
                }
            }
            crate::AppState::InGame => {
                self.build_crosshair_ui(ui.ctx());
                if self.show_diagnostics {
                    self.build_diagnostics_ui(ui.ctx());
                }
                self.build_chat_ui(ui.ctx());
                if self.chat_submit_pending {
                    self.chat_submit_pending = false;
                    self.submit_chat();
                }
            }
        }
    }

    /// (Re)load the crosshair image from `cfg.ui.crosshair_path` into an
    /// egui texture — called once from resumed(), and again by the
    /// Settings tab whenever the path/size is edited, so swapping in a
    /// custom crosshair.png takes effect immediately without a relaunch.
    /// On failure, logs a warning and leaves `crosshair_tex` as None —
    /// build_crosshair_ui just skips drawing rather than the app crashing
    /// over a missing/corrupt HUD asset.
    pub(crate) fn load_crosshair_texture(&mut self) {
        match image::open(&self.cfg.ui.crosshair_path) {
            Ok(img) => {
                let rgba = img.to_rgba8();
                let (w, h) = rgba.dimensions();
                let color_image =
                    egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &rgba);
                self.crosshair_tex = Some(self.egui_ctx.load_texture(
                    "crosshair",
                    color_image,
                    egui::TextureOptions::LINEAR,
                ));
            }
            Err(e) => {
                tracing::warn!(
                    "failed to load crosshair image {}: {e}",
                    self.cfg.ui.crosshair_path
                );
                self.crosshair_tex = None;
            }
        }
    }

    /// Paints the crosshair centered on the viewport, InGame only. Uses the
    /// raw layer painter rather than an egui::Window/Area, so it's pure
    /// drawing — no hit-testing, no risk of swallowing the mouse clicks
    /// break/place/pick rely on during InGame.
    fn build_crosshair_ui(&self, ctx: &egui::Context) {
        let Some(tex) = &self.crosshair_tex else {
            return;
        };
        let size = self.cfg.ui.crosshair_size;
        let center = ctx.content_rect().center();
        let rect = egui::Rect::from_center_size(center, egui::vec2(size, size));
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("crosshair"),
        ));
        painter.image(
            tex.id(),
            rect,
            egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
            egui::Color32::WHITE,
        );
    }

    pub(crate) fn build_diagnostics_ui(&mut self, ctx: &egui::Context) {
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
                let p = if self.guest.wasm_game.is_some() {
                    let feet = cubic_wasm::get_player_feet();
                    cubic_math::DVec3::new(feet.x, feet.y, feet.z)
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
                let loaded = self.world.chunk_meshes.len();
                let pending = self.world.stream.ready_meshes.len();
                ui.label(format!("Chunks: {loaded} loaded  {pending} pending"));

                // Block position (which voxel the camera is in)
                let voxel_x = (p.x / cubic_world::VOXEL_SIZE as f64).floor() as i32;
                let voxel_y = (p.y / cubic_world::VOXEL_SIZE as f64).floor() as i32;
                let voxel_z = (p.z / cubic_world::VOXEL_SIZE as f64).floor() as i32;
                ui.label(format!("Block: {voxel_x} {voxel_y} {voxel_z}"));
                ui.label(format!("Seed: {}", self.world.seed));
            });
    }
}
