// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Shell-like command input bar with history, fish-style ghost completion,
//! tab-complete popup, and syntax highlighting.

use std::collections::VecDeque;

pub(crate) const INPUT_HISTORY_CAP: usize = 200;
/// Max completions shown in the popup at once.
const POPUP_MAX_VISIBLE: usize = 8;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

#[derive(Default)]
pub(crate) struct CommandInputBar {
    /// Current text in the input field.
    pub(crate) text: String,
    /// Submitted command history, most recent last.
    pub(crate) history: VecDeque<String>,
    /// Index into history while navigating (None = live input).
    pub(crate) history_cursor: Option<usize>,
    /// Saved live draft while navigating history.
    pub(crate) draft: String,
    /// Fish-style ghost: suffix from history to show greyed-out after cursor.
    pub(crate) ghost: Option<String>,
    /// Current tab-complete candidates.
    pub(crate) completions: Vec<String>,
    /// Which completion is highlighted.
    pub(crate) completion_cursor: usize,
    /// Whether the completion popup is visible.
    pub(crate) popup_open: bool,
    /// Ctrl-R reverse history search mode.
    pub(crate) ctrl_r_mode: bool,
    /// Query string while in ctrl-r mode.
    pub(crate) ctrl_r_query: String,
    /// How many matches (from most recent) the current match has skipped
    /// past — advanced by repeated Ctrl-R presses (bash-style cycling to
    /// progressively older matches for the same query), reset to 0 whenever
    /// the query itself changes.
    pub(crate) ctrl_r_skip: usize,
    pub(crate) force_cursor_end: bool,
}

impl CommandInputBar {
    /// Called after each keystroke to update ghost completion from history.
    pub(crate) fn update_ghost(&mut self) {
        if self.text.is_empty() {
            self.ghost = None;
            return;
        }
        self.ghost = self
            .history
            .iter()
            .rev()
            .find(|h| h.starts_with(&self.text) && h.as_str() != self.text)
            .map(|h| h[self.text.len()..].to_string());
    }

    /// Accept the ghost completion (right arrow / End).
    pub(crate) fn accept_ghost(&mut self) {
        if let Some(ghost) = self.ghost.take() {
            self.text.push_str(&ghost);
            self.history_cursor = None;
            self.force_cursor_end = true;
        }
    }

    /// Navigate history up (older).
    pub(crate) fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let next = match self.history_cursor {
            None => {
                self.draft = self.text.clone();
                self.history.len() - 1
            }
            Some(0) => 0,
            Some(i) => i - 1,
        };
        self.history_cursor = Some(next);
        self.text = self.history[next].clone();
        self.update_ghost();
        self.force_cursor_end = true;
    }

    /// Navigate history down (newer).
    pub(crate) fn history_down(&mut self) {
        match self.history_cursor {
            None => {}
            Some(i) if i + 1 >= self.history.len() => {
                self.history_cursor = None;
                self.text = self.draft.clone();
                self.update_ghost();
                self.force_cursor_end = true;
            }
            Some(i) => {
                self.history_cursor = Some(i + 1);
                self.text = self.history[i + 1].clone();
                self.update_ghost();
                self.force_cursor_end = true;
            }
        }
    }

    /// Push a submitted entry into history.
    pub(crate) fn push_history(&mut self, entry: String) {
        // Don't push duplicates of the most recent entry
        if self.history.back().map(|h| h == &entry).unwrap_or(false) {
            return;
        }
        if self.history.len() >= INPUT_HISTORY_CAP {
            self.history.pop_front();
        }
        self.history.push_back(entry);
        self.history_cursor = None;
        self.ghost = None;
    }

    /// Update completions from the dispatcher and open popup if multiple.
    pub(crate) fn refresh_completions(&mut self, completions: Vec<String>) {
        self.completions = completions;
        self.completion_cursor = 0;
        self.popup_open = self.completions.len() > 1;
        // If exactly one, complete inline
        if self.completions.len() == 1 {
            let c = self.completions[0].clone();
            // Replace the last token with the completion
            let last_space = self.text.rfind(' ').map(|i| i + 1).unwrap_or(0);
            self.text = format!("{}{} ", &self.text[..last_space], c);
            self.completions.clear();
        }
    }

    /// Accept the currently highlighted completion.
    pub(crate) fn accept_completion(&mut self) {
        if self.completions.is_empty() {
            return;
        }
        let c = self.completions[self.completion_cursor].clone();
        let last_space = self.text.rfind(' ').map(|i| i + 1).unwrap_or(0);
        self.text = format!("{}{} ", &self.text[..last_space], c);
        self.completions.clear();
        self.popup_open = false;
        self.update_ghost();
        self.force_cursor_end = true;
    }

    pub(crate) fn completion_up(&mut self) {
        if self.completion_cursor > 0 {
            self.completion_cursor -= 1;
        }
    }

    pub(crate) fn completion_down(&mut self) {
        if self.completion_cursor + 1 < self.completions.len() {
            self.completion_cursor += 1;
        }
    }

    // --- Ctrl-R search ---

    /// The `skip`-th match (0 = most recent), searching from the newest
    /// history entry backward. Returns an owned copy to avoid holding a
    /// borrow of `self.history` across the caller's `self.text` write.
    fn ctrl_r_nth_match(&self, skip: usize) -> Option<String> {
        self.history
            .iter()
            .rev()
            .filter(|h| h.contains(&self.ctrl_r_query))
            .nth(skip)
            .cloned()
    }

    /// Re-run the search from the most recent match — called whenever the
    /// query itself changes (typing or backspacing), which resets any
    /// progress `ctrl_r_next` had cycled through.
    pub(crate) fn ctrl_r_apply(&mut self) {
        self.ctrl_r_skip = 0;
        if let Some(hit) = self.ctrl_r_nth_match(0) {
            self.text = hit;
        }
    }

    /// Cycle to the next older match for the same query — bash's
    /// reverse-i-search behavior for repeated Ctrl-R. A no-op (stays on the
    /// current match) once there's nothing further back, rather than
    /// wrapping around to the newest match again.
    pub(crate) fn ctrl_r_next(&mut self) {
        let next_skip = self.ctrl_r_skip + 1;
        if let Some(hit) = self.ctrl_r_nth_match(next_skip) {
            self.ctrl_r_skip = next_skip;
            self.text = hit;
        }
    }

    pub(crate) fn ctrl_r_accept(&mut self) {
        self.ctrl_r_mode = false;
        self.ctrl_r_query.clear();
        self.ctrl_r_skip = 0;
        self.update_ghost();
    }

    pub(crate) fn ctrl_r_cancel(&mut self, restore: String) {
        self.ctrl_r_mode = false;
        self.ctrl_r_query.clear();
        self.ctrl_r_skip = 0;
        self.text = restore;
        self.update_ghost();
        self.force_cursor_end = true;
    }

    // --- Rendering ---

    /// Paint the input bar background, syntax-highlighted text, ghost, and
    /// completion popup. The egui TextEdit is handled separately in
    /// `build_chat_ui` -- this method is pure painter output.
    pub(crate) fn paint(
        &self,
        ctx: &egui::Context,
        painter: &egui::Painter,
        bar_rect: egui::Rect,
        font: &egui::FontId,
        known_commands: &[&str],
    ) {
        let pad = 8.0;

        // Determine display text and prompt
        let (prompt, display_text) = if self.ctrl_r_mode {
            (
                format!("(reverse-search `{}`): ", self.ctrl_r_query),
                self.text.clone(),
            )
        } else {
            (String::new(), self.text.clone())
        };

        // Paint prompt in cyan if ctrl-r
        let mut x = bar_rect.min.x + pad;
        let y = bar_rect.min.y + (bar_rect.height() - font_height(ctx, font)) * 0.5;

        if !prompt.is_empty() {
            let color = egui::Color32::from_rgb(100, 220, 220);
            let galley = ctx.fonts_mut(|f| f.layout_no_wrap(prompt.clone(), font.clone(), color));
            let w = galley.size().x;
            painter.galley(egui::pos2(x, y), galley, color);
            x += w;
        }

        // Tokenize and paint with syntax colors
        x = paint_tokens(ctx, painter, &display_text, x, y, font, known_commands);

        // Ghost completion (greyed-out suffix)
        if let Some(ghost) = &self.ghost {
            if !self.ctrl_r_mode {
                let ghost_color = egui::Color32::from_gray(100);
                let galley =
                    ctx.fonts_mut(|f| f.layout_no_wrap(ghost.clone(), font.clone(), ghost_color));
                painter.galley(egui::pos2(x, y), galley, ghost_color);
            }
        }

        // Completion popup
        if self.popup_open && !self.completions.is_empty() {
            let line_h = font_height(ctx, font) + 4.0;
            let visible = self.completions.len().min(POPUP_MAX_VISIBLE);
            let popup_height = visible as f32 * line_h + 4.0;
            let popup_width = 300.0f32;
            let popup_rect = egui::Rect::from_min_size(
                egui::pos2(bar_rect.min.x, bar_rect.min.y - popup_height - 4.0),
                egui::vec2(popup_width, popup_height),
            );

            painter.rect_filled(popup_rect, 2.0, egui::Color32::from_black_alpha(220));

            let scroll_start = if self.completion_cursor >= POPUP_MAX_VISIBLE {
                self.completion_cursor - POPUP_MAX_VISIBLE + 1
            } else {
                0
            };

            for (i, entry) in self
                .completions
                .iter()
                .enumerate()
                .skip(scroll_start)
                .take(POPUP_MAX_VISIBLE)
            {
                let row = i - scroll_start;
                let row_rect = egui::Rect::from_min_size(
                    egui::pos2(
                        popup_rect.min.x,
                        popup_rect.min.y + 2.0 + row as f32 * line_h,
                    ),
                    egui::vec2(popup_width, line_h),
                );
                if i == self.completion_cursor {
                    painter.rect_filled(
                        row_rect,
                        0.0,
                        egui::Color32::from_rgba_unmultiplied(100, 160, 255, 60),
                    );
                }
                let color = if i == self.completion_cursor {
                    egui::Color32::WHITE
                } else {
                    egui::Color32::from_gray(200)
                };
                let galley =
                    ctx.fonts_mut(|f| f.layout_no_wrap(entry.clone(), font.clone(), color));
                painter.galley(
                    egui::pos2(row_rect.min.x + 4.0, row_rect.min.y + 2.0),
                    galley,
                    color,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Syntax highlighting
// ---------------------------------------------------------------------------

fn paint_tokens(
    ctx: &egui::Context,
    painter: &egui::Painter,
    text: &str,
    start_x: f32,
    y: f32,
    font: &egui::FontId,
    known_commands: &[&str],
) -> f32 {
    let mut x = start_x;
    let space_w = ctx.fonts_mut(|f| f.glyph_width(font, ' '));

    // Split preserving spaces for correct x advance
    let mut tokens: Vec<(String, bool)> = vec![]; // (token, is_space)
    let mut cur = String::new();
    for ch in text.chars() {
        if ch == ' ' {
            if !cur.is_empty() {
                tokens.push((cur.clone(), false));
                cur.clear();
            }
            tokens.push((" ".to_string(), true));
        } else {
            cur.push(ch);
        }
    }
    if !cur.is_empty() {
        tokens.push((cur, false));
    }

    let mut word_idx = 0usize; // index of non-space tokens
    for (token, is_space) in &tokens {
        if *is_space {
            x += space_w;
            continue;
        }
        let color = token_color(token, word_idx, known_commands);
        let galley = ctx.fonts_mut(|f| f.layout_no_wrap(token.clone(), font.clone(), color));
        let w = galley.size().x;
        painter.galley(egui::pos2(x, y), galley, color);
        x += w;
        word_idx += 1;
    }
    x
}

fn token_color(token: &str, word_idx: usize, known_commands: &[&str]) -> egui::Color32 {
    if word_idx == 0 {
        // Command name
        let name = token.trim_start_matches('/');
        if known_commands.contains(&name) {
            egui::Color32::WHITE
        } else {
            egui::Color32::RED
        }
    } else if token == "@p" || token == "@c" {
        egui::Color32::from_rgb(100, 160, 255) // blue
    } else if token == "~" {
        egui::Color32::from_rgb(100, 220, 220) // cyan
    } else if (token.starts_with('~') && token[1..].parse::<f64>().is_ok())
        || token.parse::<f64>().is_ok()
    {
        egui::Color32::from_rgb(100, 220, 100) // green: plain number or ~offset
    } else {
        egui::Color32::from_rgb(220, 200, 100) // yellow for string args
    }
}

fn font_height(ctx: &egui::Context, font: &egui::FontId) -> f32 {
    ctx.fonts_mut(|f| f.row_height(font))
}
