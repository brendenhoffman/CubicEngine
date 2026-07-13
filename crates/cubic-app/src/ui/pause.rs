// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Pause menu overlay.

use crate::App;
use crate::AppState;

impl App {
    pub(crate) fn build_pause_ui(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(egui::Color32::from_black_alpha(180)))
            .show(ui, |ui| {
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
                            kind: 0,
                            payload: String::new(),
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
                .show(ui.ctx(), |ui| {
                    self.build_settings_tab(ui);
                });
        }

        if self.pause_controls_open {
            egui::Window::new("Controls")
                .collapsible(false)
                .show(ui.ctx(), |ui| {
                    self.build_controls_tab(ui);
                });
        }

        // Diagnostics overlay still visible while paused
        if self.show_diagnostics {
            self.build_diagnostics_ui(ui.ctx());
        }
    }
}
