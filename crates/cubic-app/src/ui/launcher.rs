// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Launcher screen: game/profile/settings/controls tabs, remap capture, and
//! the Launch button's transition into InGame.

use crate::backend::RendererBackend;
use crate::config::{save_global_cfg, KeyBinding, ModifierKey, TextureFilter, TriggerKind};
use crate::input::{input_source_to_string, resolve_controls, InputSource, InputTracker};
use crate::profile;
use crate::{App, AppState};
use cubic_platform::winit::window::Fullscreen;

use super::{GameEntry, LauncherTab, PendingWindowedResize, WindowMode};

pub(crate) fn scan_games() -> Vec<GameEntry> {
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

impl App {
    /// Called by the launcher's Launch button (and nothing else yet — the
    /// selected game/profile in `self.launcher` isn't re-resolved into
    /// `self.cfg`/`self.guest.plugin` here; switching games/profiles at runtime is
    /// future work, per the `current_profile`/`current_game_name` fields).
    pub(crate) fn handle_launch(&mut self) {
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

        self.persist_world_prefs();

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
        prefs.mode = Some(super::window_mode_to_str(self.launcher.window_mode).to_string());
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

    fn persist_world_prefs(&mut self) {
        let world = self
            .current_profile
            .world
            .get_or_insert_with(Default::default);
        world.last_world = Some(self.current_world_name.clone());
        if let Err(e) = profile::save(
            &self.current_profile,
            &self.current_game_name,
            &self.current_profile_name,
        ) {
            tracing::warn!("failed to save profile after world launch: {e}");
        }
    }

    pub(crate) fn build_launcher_ui(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show(ui, |ui| {
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

    pub(crate) fn build_settings_tab(&mut self, ui: &mut egui::Ui) {
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

            ui.collapsing("Crosshair", |ui| {
                let mut changed = false;
                ui.horizontal(|ui| {
                    ui.label("Image path");
                    let resp = ui.text_edit_singleline(&mut self.cfg.ui.crosshair_path);
                    if resp.lost_focus() {
                        changed = true;
                        self.load_crosshair_texture();
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Size (px)");
                    changed |= ui
                        .add(egui::Slider::new(
                            &mut self.cfg.ui.crosshair_size,
                            4.0..=128.0,
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

    pub(crate) fn build_controls_tab(&mut self, ui: &mut egui::Ui) {
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
    pub(crate) fn control_binding_mut(&mut self, name: &str) -> Option<&mut KeyBinding> {
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

    pub(crate) fn apply_control_remap(&mut self, binding: &str, key_name: &str) {
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
    pub(crate) fn complete_remap(&mut self, binding: &str, source: InputSource) {
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
    pub(crate) fn clear_control_binding(&mut self, binding: &str) {
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
    pub(crate) fn poll_gamepads(&mut self) {
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

    pub(crate) fn set_control_modifier(&mut self, binding: &str, modifier: ModifierKey) {
        if let Some(b) = self.control_binding_mut(binding) {
            b.modifier = modifier;
        }
        if let Some(ov) = self.control_override_mut(binding) {
            ov.modifier = Some(modifier.cfg_str().to_string());
        }
        self.persist_control_change();
    }

    pub(crate) fn set_control_trigger(&mut self, binding: &str, trigger: TriggerKind) {
        if let Some(b) = self.control_binding_mut(binding) {
            b.trigger = trigger;
        }
        if let Some(ov) = self.control_override_mut(binding) {
            ov.trigger = Some(trigger.cfg_str().to_string());
        }
        self.persist_control_change();
    }
}
