// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! In-game chat overlay: painter-drawn message history, egui input bar,
//! and append-only disk log.

use crate::ui::input_bar::INPUT_HISTORY_CAP;
use crate::App;
use std::io::Write;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum messages kept in memory.
pub(crate) const HISTORY_MEMORY: usize = 200;
/// Maximum messages shown at once in the history panel.
const HISTORY_VISIBLE: usize = 20;
/// Seconds history stays visible after a message arrives (before fading).
const FADE_SECS: f32 = 5.0;
/// Width of the chat history panel in logical pixels.
const PANEL_WIDTH: f32 = 480.0;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub(crate) struct ChatMessage {
    pub(crate) timestamp: String, // HH:MM:SS
    pub(crate) text: String,
    pub(crate) kind: ChatMessageKind,
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum ChatMessageKind {
    Chat,
    CommandOutput,
    Error,
}

impl ChatMessageKind {
    fn color(self) -> egui::Color32 {
        match self {
            Self::Chat => egui::Color32::WHITE,
            Self::CommandOutput => egui::Color32::YELLOW,
            Self::Error => egui::Color32::RED,
        }
    }
}

// ---------------------------------------------------------------------------
// App impl
// ---------------------------------------------------------------------------

impl App {
    /// Push a message to the in-memory log, reset the fade timer, and
    /// append to the disk log if a world is loaded. Splits `text` on
    /// embedded newlines (e.g. /help's multi-line output) into one entry
    /// per line: the history panel paints one `ChatMessage` per visual
    /// line with no-wrap text (see `build_chat_ui`), so a single message
    /// containing '\n' would render as one long unwrapped line overflowing
    /// the panel instead of stacking normally. Keeping the disk log one
    /// physical line per message here too keeps `load_chat_log` (which
    /// reads it back line-by-line) symmetric with what was written.
    pub(crate) fn push_chat_message(&mut self, text: String, kind: ChatMessageKind) {
        let ts = wall_time_hms();
        let date = wall_date();

        let mut lines = text.lines().peekable();
        if lines.peek().is_none() {
            // text was empty (no lines at all) -- still record one blank entry.
            self.push_chat_line("", &ts, &date, kind);
            return;
        }
        for line in lines {
            self.push_chat_line(line, &ts, &date, kind);
        }
    }

    fn push_chat_line(&mut self, line: &str, ts: &str, date: &str, kind: ChatMessageKind) {
        if self.chat_messages.len() >= HISTORY_MEMORY {
            self.chat_messages.pop_front();
        }
        self.chat_messages.push_back(ChatMessage {
            timestamp: ts.to_string(),
            text: line.to_string(),
            kind,
        });
        self.chat_fade_timer = Some(std::time::Instant::now());

        if let Some(path) = &self.chat_log_path.clone() {
            let entry = format!("[{date} {ts}] {line}\n");
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .truncate(false)
                .open(path)
            {
                let _ = f.write_all(entry.as_bytes());
            }
        }
    }

    /// Open the chat input bar, optionally pre-filling it (e.g. `"/"` for
    /// command mode). Releases cursor grab while chat is open.
    pub(crate) fn open_chat(&mut self, prefill: &str) {
        self.chat_open = true;
        self.input_bar.text = prefill.to_string();
        self.input_bar.history_cursor = None;
        self.input_bar.ghost = None;
        self.input_bar.popup_open = false;
        self.input_bar.ctrl_r_mode = false;
        self.apply_cursor_state();
    }

    /// Close the chat bar without submitting (Escape) — discards the draft
    /// and re-locks the cursor, same cleanup `submit_chat` does on its way
    /// out, just without dispatching anything.
    pub(crate) fn close_chat(&mut self) {
        self.input_bar.text.clear();
        self.input_bar.ghost = None;
        self.input_bar.popup_open = false;
        self.input_bar.completions.clear();
        self.input_bar.history_cursor = None;
        self.input_bar.ctrl_r_mode = false;
        self.input_bar.ctrl_r_query.clear();
        self.chat_open = false;
        self.apply_cursor_state();
    }

    /// Submit the current input, dispatch commands or push plain messages,
    /// then close the bar and re-lock the cursor.
    pub(crate) fn submit_chat(&mut self) {
        let input = self.input_bar.text.trim().to_string();
        self.input_bar.text.clear();
        self.input_bar.ghost = None;
        self.input_bar.popup_open = false;
        self.chat_open = false;
        self.apply_cursor_state();

        if input.is_empty() {
            return;
        }

        self.input_bar.push_history(input.clone());
        if let Some(path) = self
            .chat_log_path
            .as_ref()
            .and_then(|p| p.parent())
            .map(|p| p.to_owned())
        {
            self.save_input_history(&path);
        }

        if input.starts_with('/') {
            crate::commands::dispatch(self, &input);
        } else {
            self.push_chat_message(input, ChatMessageKind::Chat);
        }
    }

    /// Build the chat overlay. Call from `build_ui` while InGame or Paused.
    /// Uses a raw painter for the history panel (no egui window chrome) and
    /// a bare egui Area for the TextEdit so keyboard input still routes
    /// through egui's event system.
    pub(crate) fn build_chat_ui(&mut self, ctx: &egui::Context) {
        let screen = ctx.content_rect();
        let margin = 8.0;
        let line_height = 16.0;
        let font_size = 14.0;
        let pad = 4.0;
        let input_height = 22.0;

        // How opaque should the history be right now?
        let history_alpha: f32 = if self.chat_open {
            1.0
        } else {
            match self.chat_fade_timer {
                Some(t) => {
                    let e = t.elapsed().as_secs_f32();
                    if e >= FADE_SECS {
                        0.0
                    } else if e > FADE_SECS - 1.0 {
                        1.0 - (e - (FADE_SECS - 1.0))
                    } else {
                        1.0
                    }
                }
                None => 0.0,
            }
        };

        let font = egui::FontId::monospace(font_size);
        let painter = ctx.layer_painter(egui::LayerId::new(
            egui::Order::Foreground,
            egui::Id::new("chat"),
        ));

        // --- History panel ---
        if history_alpha > 0.0 && !self.chat_messages.is_empty() {
            let count = self.chat_messages.len().min(HISTORY_VISIBLE);
            let start = self.chat_messages.len() - count;
            let panel_height = count as f32 * line_height + pad * 2.0;

            // Sit just above the input bar when open, otherwise at the
            // bottom of the screen.
            let panel_bottom = screen.max.y
                - margin
                - if self.chat_open {
                    input_height + pad
                } else {
                    0.0
                };
            let panel_rect = egui::Rect::from_min_size(
                egui::pos2(margin, panel_bottom - panel_height),
                egui::vec2(PANEL_WIDTH, panel_height),
            );

            let bg_alpha = ((if self.chat_open { 160.0 } else { 80.0 }) * history_alpha) as u8;
            painter.rect_filled(panel_rect, 2.0, egui::Color32::from_black_alpha(bg_alpha));

            for (i, msg) in self.chat_messages.range(start..).enumerate() {
                let y = panel_rect.min.y + pad + i as f32 * line_height;
                let a = (255.0 * history_alpha) as u8;

                // Timestamp
                let ts_text = format!("[{}] ", msg.timestamp);
                let ts_color = egui::Color32::from_rgba_unmultiplied(160, 160, 160, a);
                let ts_galley =
                    ctx.fonts_mut(|f| f.layout_no_wrap(ts_text.clone(), font.clone(), ts_color));
                let ts_width = ts_galley.size().x;
                painter.galley(egui::pos2(panel_rect.min.x + pad, y), ts_galley, ts_color);

                // Message
                let base = msg.kind.color();
                let msg_color =
                    egui::Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), a);
                let msg_galley =
                    ctx.fonts_mut(|f| f.layout_no_wrap(msg.text.clone(), font.clone(), msg_color));
                painter.galley(
                    egui::pos2(panel_rect.min.x + pad + ts_width, y),
                    msg_galley,
                    msg_color,
                );
            }
        }

        // ctrl-r uses a separate buffer so the TextEdit
        // doesn't write into self.input_bar.text while searching.
        let mut ctrl_r_buf = if self.input_bar.ctrl_r_mode {
            self.input_bar.ctrl_r_query.clone()
        } else {
            String::new()
        };

        // Input bar
        if self.chat_open {
            let bar_rect = egui::Rect::from_min_size(
                egui::pos2(margin, screen.max.y - margin - input_height),
                egui::vec2(screen.width() - margin * 2.0, input_height),
            );

            // Background + left accent bar
            painter.rect_filled(bar_rect, 2.0, egui::Color32::from_black_alpha(200));
            painter.rect_filled(
                egui::Rect::from_min_size(bar_rect.min, egui::vec2(2.0, input_height)),
                0.0,
                egui::Color32::WHITE,
            );

            // Paint syntax highlighting, ghost, ctrl-r prompt, popup. Game-
            // registered commands (e.g. /give) are just as valid as the
            // built-ins (see commands::dispatch's fallback) and need to be
            // in this list too, or they'd always paint red as "unknown".
            let mut known: Vec<&str> = vec!["tp", "set", "help", "locate"];
            known.extend(self.guest.registered_commands.iter().map(|c| c.name.as_str()));
            self.input_bar.paint(ctx, &painter, bar_rect, &font, &known);

            // Invisible TextEdit for keyboard routing only --
            // in ctrl-r mode it targets ctrl_r_buf instead of text so
            // typed characters feed the search query, not the command.
            egui::Area::new(egui::Id::new("chat_input_area"))
                .fixed_pos(egui::pos2(bar_rect.min.x + 6.0, bar_rect.min.y + 3.0))
                .order(egui::Order::Foreground)
                .show(ctx, |ui| {
                    ui.style_mut().visuals.extreme_bg_color = egui::Color32::TRANSPARENT;
                    ui.style_mut().visuals.widgets.inactive.bg_fill = egui::Color32::TRANSPARENT;
                    ui.style_mut().visuals.widgets.active.bg_fill = egui::Color32::TRANSPARENT;
                    // Transparent text -- painter draws the visible text
                    ui.style_mut().visuals.override_text_color = Some(egui::Color32::TRANSPARENT);

                    let edit_target = if self.input_bar.ctrl_r_mode {
                        &mut ctrl_r_buf
                    } else {
                        &mut self.input_bar.text
                    };

                    let mut output = egui::TextEdit::singleline(edit_target)
                        .desired_width(bar_rect.width() - 12.0)
                        .frame(egui::Frame::NONE)
                        .font(font.clone())
                        .show(ui);

                    if !output.response.has_focus() {
                        output.response.request_focus();
                    }

                    // Force cursor to end after history navigation or tab complete
                    if self.input_bar.force_cursor_end {
                        self.input_bar.force_cursor_end = false;
                        let len = self.input_bar.text.chars().count();
                        let cursor = egui::text::CCursor::new(len);
                        let cursor_range = egui::text::CCursorRange::one(cursor);
                        output.state.cursor.set_char_range(Some(cursor_range));
                        output.state.store(ui.ctx(), output.response.id);
                    }

                    if output.response.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter))
                    {
                        self.chat_submit_pending = true;
                    }

                    if self.input_bar.ctrl_r_mode {
                        // Sync query back and apply search on every change
                        if output.response.changed() {
                            self.input_bar.ctrl_r_query = ctrl_r_buf.clone();
                            self.input_bar.ctrl_r_apply();
                        }
                    } else {
                        // Normal mode: update ghost and auto-completions
                        self.input_bar.update_ghost();

                        if output.response.changed() && self.input_bar.text.starts_with('/') {
                            let cur = self.input_bar.text.len();
                            let text = self.input_bar.text.clone();
                            let candidates = crate::commands::completions(self, &text, cur);
                            if !candidates.is_empty() {
                                self.input_bar.completions = candidates;
                                if self.input_bar.completions.len() > 1 {
                                    self.input_bar.popup_open = true;
                                }
                                if self.input_bar.completion_cursor
                                    >= self.input_bar.completions.len()
                                {
                                    self.input_bar.completion_cursor = 0;
                                }
                            } else {
                                self.input_bar.popup_open = false;
                                self.input_bar.completions.clear();
                            }
                        }

                        // Clear completions if text no longer starts with /
                        if output.response.changed() && !self.input_bar.text.starts_with('/') {
                            self.input_bar.popup_open = false;
                            self.input_bar.completions.clear();
                        }
                    }
                });
        }
    }

    /// Read the last `HISTORY_MEMORY` lines of `chat.log` from `world_dir`
    /// into the in-memory history. Sets `chat_log_path` for future writes.
    /// Called from `load_world()`.
    pub(crate) fn load_chat_log(&mut self, world_dir: &std::path::Path) {
        let path = world_dir.join("chat.log");
        self.chat_log_path = Some(path.clone());
        self.chat_messages.clear();

        let Ok(content) = std::fs::read_to_string(&path) else {
            return;
        };

        // Take last HISTORY_MEMORY lines, preserving order
        let lines: Vec<&str> = content.lines().collect();
        let start = lines.len().saturating_sub(HISTORY_MEMORY);
        for line in &lines[start..] {
            // Format on disk: [YYYY-MM-DD HH:MM:SS] text
            let (timestamp, text) = if let Some(rest) = line.strip_prefix('[') {
                if let Some(close) = rest.find("] ") {
                    let ts_field = &rest[..close]; // "YYYY-MM-DD HH:MM:SS"
                    let hms = if ts_field.len() >= 19 {
                        ts_field[11..19].to_string()
                    } else {
                        String::new()
                    };
                    (hms, rest[close + 2..].to_string())
                } else {
                    (String::new(), line.to_string())
                }
            } else {
                (String::new(), line.to_string())
            };

            self.chat_messages.push_back(ChatMessage {
                timestamp,
                text,
                kind: ChatMessageKind::Chat,
            });
        }
    }

    pub(crate) fn load_input_history(&mut self, world_dir: &std::path::Path) {
        let path = world_dir.join("input_history.txt");
        self.input_bar.history.clear();
        let Ok(content) = std::fs::read_to_string(&path) else {
            return;
        };
        for line in content.lines().take(INPUT_HISTORY_CAP) {
            self.input_bar.history.push_back(line.to_string());
        }
    }

    pub(crate) fn save_input_history(&self, world_dir: &std::path::Path) {
        let path = world_dir.join("input_history.txt");
        let content: String = self
            .input_bar
            .history
            .iter()
            .map(|h| format!("{h}\n"))
            .collect();
        let _ = std::fs::write(&path, content);
    }
}

// ---------------------------------------------------------------------------
// Time helpers (no chrono dependency)
// ---------------------------------------------------------------------------

fn wall_time_hms() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    format!("{h:02}:{m:02}:{s:02}")
}

fn wall_date() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    crate::profile::format_unix_as_rfc3339(secs)[..10].to_string()
}
