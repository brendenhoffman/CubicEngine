// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
mod backend;
mod commands;
mod config;
#[cfg(debug_assertions)]
mod flat_generator;
mod frustum;
mod game_override;
mod guest;
mod input;
mod loader;
mod profile;
mod ui;
mod world;

use anyhow::Result;
use backend::{Backend, RendererBackend};
use clap::Parser;
use config::{
    apply_game_override, apply_profile, build_custom_controls, load_cfg, AppCfg, CustomControl,
    RenderCfg, UnfocusedPolicy, VsyncMode,
};
use cubic_core::init_tracing;
use cubic_math::{Camera, DVec3, Vec3};
use cubic_platform::winit::{
    application::ApplicationHandler,
    dpi::PhysicalSize,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, ModifiersState, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{CursorGrabMode, Window, WindowId},
};
use cubic_render::{RenderSize, Renderer};
use cubic_render_gl::GlRenderer;
use cubic_render_vk::VkRenderer;
use cubic_world::{RegionCache, CHUNK_SIZE, VOXEL_SIZE};
use input::{resolve_controls, InputSource, InputState, InputTracker, ResolvedControls, MAX_PITCH};
use std::sync::{Arc, Mutex};
use tracing::{error, info};
use ui::{
    scan_games, str_to_window_mode, LauncherState, LauncherTab, PendingWindowedResize, WindowMode,
    REMAP_TIMEOUT,
};

// ---------------------------------------------------------------------------
// App state machine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum AppState {
    Launcher, // egui launcher shown, no world loaded, cursor free
    InGame,   // world running, cursor locked, no egui (except diagnostics)
    Paused,   // world paused, cursor free, egui pause menu shown
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Choose renderer backend: gl | vk
    #[arg(long, default_value = "vk")]
    backend: String,
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
    // Loaded once in resumed() from cfg.ui.crosshair_path (see
    // load_crosshair_texture) — None if that image failed to load, in
    // which case the crosshair is just silently skipped rather than
    // crashing the app over a missing/bad HUD asset.
    crosshair_tex: Option<egui::TextureHandle>,
    // Resolved once at startup from cfg.controls (see resolve_controls).
    controls: ResolvedControls,

    // Empty until load_world() (called from handle_launch()) actually
    // constructs the WASM plugin with that launch's seed baked in — see
    // GuestPlugin's doc comment.
    guest: guest::GuestPlugin,
    // Renderer-facing world state (chunk/entity meshes, bindless texture
    // lookups, streaming) — see WorldRenderer's doc comment.
    world: world::WorldRenderer,
    camera: Camera,
    input: InputState,
    // Tracked from WindowEvent::ModifiersChanged rather than InputState's
    // held-key tracking, which is deliberately suppressed while chat has
    // focus (see window_event's chat-input block) — a Ctrl-R detection
    // that relied on InputState would see Ctrl as never-held, since its
    // own key-down event never reaches set_source while chat is open.
    modifiers: ModifiersState,
    last_frame_instant: std::time::Instant,
    last_frame_dt: f32,
    detected_refresh_hz: f32,
    input_tracker: InputTracker,
    // None if no gamepad backend is available on this platform (Gilrs::new
    // can fail, e.g. no udev) — gamepad support is then simply absent
    // rather than a hard error, same spirit as backend/render fallbacks
    // elsewhere in this file.
    gilrs: Option<gilrs::Gilrs>,

    current_world_name: String,
    region_cache: Option<Arc<Mutex<RegionCache>>>,
    autosave_timer: std::time::Instant,

    // Chat
    chat_open: bool,
    input_bar: ui::input_bar::CommandInputBar,
    chat_messages: std::collections::VecDeque<ui::ChatMessage>,
    chat_log_path: Option<std::path::PathBuf>,
    chat_fade_timer: Option<std::time::Instant>,
    chat_submit_pending: bool,
    player_spectating: bool,
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
        self.load_crosshair_texture();

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

        // Refresh world list
        self.refresh_world_list();
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

        // Tracked unconditionally, before egui or any other branch below can
        // consume/return early — Ctrl-R detection in the chat input block
        // needs reliable modifier state even while chat suppresses ordinary
        // key events from reaching InputState (see `modifiers`'s doc comment
        // on the App struct).
        if let WindowEvent::ModifiersChanged(mods) = &event {
            self.modifiers = mods.state();
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
                        self.input.clear_held();
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
                // Chat intercepts first — suppress game input while open,
                // and handle T / / / Escape for opening and closing.
                if matches!(self.state, AppState::InGame | AppState::Paused)
                    && event.state == ElementState::Pressed
                {
                    if self.chat_open {
                        if self.input_bar.ctrl_r_mode {
                            if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                                let restore = self.input_bar.draft.clone();
                                self.input_bar.ctrl_r_cancel(restore);
                                return;
                            }
                            if let PhysicalKey::Code(KeyCode::Enter) = event.physical_key {
                                self.input_bar.ctrl_r_accept();
                                return;
                            }
                            // Repeated Ctrl-R while already searching cycles
                            // to the next older match for the same query,
                            // like bash's reverse-i-search.
                            if let PhysicalKey::Code(KeyCode::KeyR) = event.physical_key {
                                if self.modifiers.control_key() {
                                    self.input_bar.ctrl_r_next();
                                    return;
                                }
                            }
                            // Deliberately NOT handling Backspace or
                            // event.text here: this raw event was already
                            // fed to egui_winit above (unconditionally,
                            // before this match), so the TextEdit bound to
                            // ctrl_r_buf in build_chat_ui will process
                            // typing/backspacing on its own next frame, and
                            // its "sync back to ctrl_r_query on change"
                            // logic there is the single source of truth —
                            // same as normal (non-ctrl-r) mode, which never
                            // pushes into self.input_bar.text manually
                            // either. Doing it here too double-applies every
                            // keystroke (query ends up "tt" from one "t").
                        }

                        // Esc: ctrl-r (above) takes priority when active —
                        // one press cancels just the search, restoring the
                        // draft; a second press (now that ctrl_r_mode is
                        // false) reaches here and closes chat entirely.
                        if let PhysicalKey::Code(KeyCode::Escape) = event.physical_key {
                            self.close_chat();
                            return;
                        }

                        match event.physical_key {
                            PhysicalKey::Code(KeyCode::ArrowUp) => {
                                if self.input_bar.popup_open {
                                    self.input_bar.completion_up();
                                } else if self.input_bar.ctrl_r_mode {
                                    // no-op in ctrl-r
                                } else {
                                    self.input_bar.history_up();
                                }
                                return;
                            }
                            PhysicalKey::Code(KeyCode::ArrowDown) => {
                                if self.input_bar.popup_open {
                                    self.input_bar.completion_down();
                                } else {
                                    self.input_bar.history_down();
                                }
                                return;
                            }
                            PhysicalKey::Code(KeyCode::ArrowRight)
                            | PhysicalKey::Code(KeyCode::End) => {
                                if self.input_bar.popup_open {
                                    self.input_bar.accept_completion();
                                } else {
                                    self.input_bar.accept_ghost();
                                }
                                return;
                            }
                            PhysicalKey::Code(KeyCode::Tab) => {
                                if self.input_bar.popup_open {
                                    self.input_bar.accept_completion();
                                } else {
                                    let cur = self.input_bar.text.len();
                                    let candidates = crate::commands::completions(
                                        self,
                                        &self.input_bar.text.clone(),
                                        cur,
                                    );
                                    self.input_bar.refresh_completions(candidates);
                                }
                                return;
                            }
                            _ => {}
                        }

                        // Ctrl-R. Checked via `self.modifiers` (tracked from
                        // WindowEvent::ModifiersChanged), not
                        // InputState::is_held — Ctrl's own key-down event
                        // never reaches InputState::set_source while chat is
                        // open (see the early `return`s throughout this
                        // block), so is_held(ControlLeft/Right) would always
                        // read stale/false here. Only entered when not
                        // already searching — ctrl_r_mode's own block above
                        // handles a held Ctrl during an active search (no
                        // `event.text` is produced for the Ctrl+R combo
                        // itself, so it just falls through to here), and
                        // re-triggering would wipe the in-progress query.
                        if !self.input_bar.ctrl_r_mode {
                            if let PhysicalKey::Code(KeyCode::KeyR) = event.physical_key {
                                if self.modifiers.control_key() {
                                    self.input_bar.draft = self.input_bar.text.clone();
                                    self.input_bar.ctrl_r_mode = true;
                                    self.input_bar.ctrl_r_query.clear();
                                    return;
                                }
                            }
                        }

                        // Suppress all other keys from reaching the game while chat is open
                        return;
                    }
                    if self.state == AppState::InGame {
                        match event.physical_key {
                            PhysicalKey::Code(KeyCode::KeyT) => {
                                self.open_chat("");
                                return;
                            }
                            PhysicalKey::Code(KeyCode::Slash) => {
                                self.open_chat("/");
                                return;
                            }
                            _ => {}
                        }
                    }
                }

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
                    let full_output = egui_ctx.run_ui(raw_input, |ctx| {
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

                // Taken out of `self` (rather than borrowed) for the
                // duration of this block so world_tick_and_draw can take
                // `&mut self` without aliasing a live `&mut self.backend`
                // borrow — put back before returning either way.
                if let Some(mut backend) = self.backend.take() {
                    // Scene render only when world is active
                    if self.state == AppState::InGame || self.state == AppState::Paused {
                        self.world_tick_and_draw(&mut backend, now, dt);
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

                    self.backend = Some(backend);
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
            // Raw motion deltas arrive independent of cursor grab (see
            // apply_cursor_state's should_lock, which already excludes
            // chat_open for the grab decision) — without also excluding it
            // here, moving the mouse over the open chat bar still turns the
            // camera underneath it.
            if self.focused && self.state == AppState::InGame && !self.chat_open {
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
            self.world.stream.flush_dirty();
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
                self.world.chunk_meshes.len()
            );
            self.frames = 0;
            self.last_fps_instant = now;
        }
    }
}

impl App {
    /// (Re)load the crosshair image from `cfg.ui.crosshair_path` into an
    /// egui texture — called once from resumed(), and again by the
    /// Settings tab whenever the path/size is edited, so swapping in a
    /// custom crosshair.png takes effect immediately without a relaunch.
    /// On failure, logs a warning and leaves `crosshair_tex` as None —
    /// build_crosshair_ui just skips drawing rather than the app crashing
    /// over a missing/corrupt HUD asset.
    /// Free-fly camera controls, used only while no WASM game is loaded
    /// (`wasm_game.is_none()`) — once one is, RedrawRequested's tick handler
    /// feeds input/mouse-look into the guest via on-tick instead, and the
    /// guest owns the camera via set-camera. Skipping both blocks here below
    /// avoids double-applying the same mouse delta to the camera.
    fn apply_input(&mut self, dt: f32) {
        if self.guest.wasm_game.is_none() {
            let (dx, dy) = self.input.take_mouse_delta();
            self.camera.yaw -= dx * self.cfg.camera.mouse_sensitivity;
            self.camera.pitch = (self.camera.pitch - dy * self.cfg.camera.mouse_sensitivity)
                .clamp(-MAX_PITCH, MAX_PITCH);
        }

        if self.guest.wasm_game.is_none() {
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

            self.camera.position +=
                (movement.normalize_or_zero() * self.cfg.camera.move_speed * dt).as_dvec3();
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
        let should_lock = self.focused && self.state == AppState::InGame && !self.chat_open;
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
    let _ = std::fs::create_dir_all(profile::worlds_dir(&game_name, &profile_name));

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
        window_mode,
        window_width_str,
        window_height_str,
        settings_open: false,
        remapping: None,
        world_list: vec![],
        new_world_name: String::new(),
        new_world_seed_str: "0".to_string(),
        renaming: None,
        pending_delete: None,
        worlds_error: None,
    };

    let current_world_name = current_profile
        .world
        .as_ref()
        .and_then(|w| w.last_world.clone())
        .unwrap_or_else(|| "New World".to_string());

    let mut app = App {
        backend_choice: args.backend,
        window: None,
        backend: None,
        render_size: RenderSize {
            width: 1,
            height: 1,
        },
        world: world::WorldRenderer::new(cfg.world.stream_radius, cfg.world.stream_radius_y),
        guest: guest::GuestPlugin::default(),
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
        crosshair_tex: None, // loaded in resumed(), once egui_ctx/window exist
        controls,
        camera: Camera {
            position: DVec3::new(
                0.0,
                ((CHUNK_SIZE / 2) as f32 * VOXEL_SIZE + 12.0) as f64,
                0.0,
            ),
            pitch: -0.3,
            ..Camera::default()
        },
        input: InputState::default(),
        modifiers: ModifiersState::empty(),
        last_frame_instant: std::time::Instant::now(),
        last_frame_dt: 0.0,
        detected_refresh_hz: 60.0, // overwritten in resumed()
        input_tracker: InputTracker::new(&controls, &custom_controls),
        gilrs: gilrs::Gilrs::new()
            .inspect_err(|e| tracing::warn!("gamepad support unavailable: {e}"))
            .ok(),
        current_world_name,
        region_cache: None,
        autosave_timer: std::time::Instant::now(),
        chat_open: false,
        input_bar: ui::input_bar::CommandInputBar::default(),
        chat_messages: std::collections::VecDeque::new(),
        chat_log_path: None,
        chat_fade_timer: None,
        chat_submit_pending: false,
        player_spectating: false,
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
