// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::Result;
use clap::Parser;
use cubic_core::init_tracing;
use cubic_math::{Camera, Vec3};
use cubic_render::{RenderSize, Renderer};
use cubic_render_gl::GlRenderer;
use cubic_render_vk::{MeshHandle, VkRenderer};
use tracing::{error, info};

use cubic_platform::winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{CursorGrabMode, Window, WindowId},
};

use serde::Deserialize;
use std::collections::HashSet;
use std::fs;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Choose renderer backend: gl | vk
    #[arg(long, default_value = "vk")]
    backend: String,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct RenderCfg {
    #[serde(default = "default_clear")]
    clear_color: [f32; 4],
    #[serde(default = "default_vsync")]
    vsync: bool,
    #[serde(default)]
    vsync_mode: VsyncMode,
    #[serde(default)]
    unfocused: UnfocusedPolicy,
    #[serde(default)]
    unfocused_fps: u32,
    #[serde(default)]
    fps_when_vsync_off: u32,
    #[serde(default)]
    hdr: bool,
    #[serde(default)]
    hdr_flavor: HdrFlavorCfg,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum VsyncMode {
    Fifo,
    #[default]
    Mailbox,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum UnfocusedPolicy {
    None,
    #[default]
    VsyncOn,
    Throttle,
}

#[derive(Debug, Clone, Copy, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
enum HdrFlavorCfg {
    #[default]
    PreferScrgb,
    PreferHdr10,
}

#[derive(Debug, Deserialize, Default)]
struct AppCfg {
    #[serde(default)]
    render: RenderCfg,
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
        }
    }
}

fn default_clear() -> [f32; 4] {
    [0.02, 0.02, 0.04, 1.0]
}
fn default_vsync() -> bool {
    true
}
fn load_cfg() -> AppCfg {
    match fs::read_to_string("cubic.toml") {
        Ok(s) => toml::from_str::<AppCfg>(&s).unwrap_or_default(),
        Err(_) => AppCfg::default(),
    }
}

// Temporary test scene: the same two-triangle test geometry that used to be
// hardcoded inside cubic-render-vk, now supplied by the app through the
// upload_mesh/draw_mesh API.
mod test_scene {
    use cubic_render_vk::{PushData, Vertex};

    const UV_TILE: f32 = 1.0;

    pub const TRI_VERTS: &[Vertex] = &[
        // FRONT triangle (closer)
        Vertex {
            pos: [0.0, 0.5, -0.988],
            color: [1.0, 0.2, 0.2],
            uv: [0.5 * UV_TILE, 1.0 * UV_TILE], // top
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [-0.3, -0.4, -0.788],
            color: [1.0, 0.2, 0.2],
            uv: [0.0 * UV_TILE, 0.0 * UV_TILE], // bottom-left
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [0.3, -0.4, -0.788],
            color: [1.0, 0.2, 0.2],
            uv: [1.0 * UV_TILE, 0.0 * UV_TILE], // bottom-right
            normal: [0.0, 0.0, 1.0],
        },
        // BACK triangle (farther)
        Vertex {
            pos: [0.0, 0.4, -0.888],
            color: [0.2, 0.8, 1.0],
            uv: [0.5 * UV_TILE, 1.0 * UV_TILE], // top
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [-0.5, -0.4, -0.888],
            color: [0.2, 0.8, 1.0],
            uv: [0.0 * UV_TILE, 0.0 * UV_TILE], // bottom-left
            normal: [0.0, 0.0, 1.0],
        },
        Vertex {
            pos: [0.5, -0.4, -0.888],
            color: [0.2, 0.8, 1.0],
            uv: [1.0 * UV_TILE, 0.0 * UV_TILE], // bottom-right
            normal: [0.0, 0.0, 1.0],
        },
    ];

    pub const TRI_IDXS: &[u32] = &[0, 1, 2, 3, 4, 5];

    pub const IDENTITY_PUSH: PushData = PushData {
        model: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        tint: [1.0, 1.0, 1.0, 1.0],
    };
}

const MOVE_SPEED: f32 = 3.0; // units/sec
const MOUSE_SENSITIVITY: f32 = 0.0025; // radians/pixel
const MAX_PITCH: f32 = std::f32::consts::FRAC_PI_2 - 0.01;

/// Held keys and accumulated mouse motion since the last time they were
/// read, driven from WindowEvent::KeyboardInput and DeviceEvent::MouseMotion.
#[derive(Default)]
struct InputState {
    held_keys: HashSet<KeyCode>,
    mouse_delta: (f32, f32),
}

impl InputState {
    fn set_key(&mut self, code: KeyCode, pressed: bool) {
        if pressed {
            self.held_keys.insert(code);
        } else {
            self.held_keys.remove(&code);
        }
    }

    fn is_held(&self, code: KeyCode) -> bool {
        self.held_keys.contains(&code)
    }

    fn accumulate_mouse_delta(&mut self, dx: f32, dy: f32) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Returns the accumulated delta and resets it to zero.
    fn take_mouse_delta(&mut self) -> (f32, f32) {
        std::mem::take(&mut self.mouse_delta)
    }
}

enum Backend {
    Gl(Box<GlRenderer>),
    Vk(Box<VkRenderer>),
}

struct App {
    backend_choice: String,
    window: Option<Window>,
    backend: Option<Backend>,
    render_size: RenderSize,

    cfg: AppCfg,
    exiting: bool,
    frames: u32,
    last_fps_instant: std::time::Instant,

    paused: bool,
    focused: bool,
    next_frame_deadline: Option<std::time::Instant>,

    vk_mesh: Option<MeshHandle>,
    camera: Camera,
    input: InputState,
    last_frame_instant: std::time::Instant,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            let window = event_loop
                .create_window(Window::default_attributes().with_title("cubic"))
                .expect("create_window");

            let size = window.inner_size();

            self.render_size = RenderSize {
                width: size.width.max(1),
                height: size.height.max(1),
            };

            let wh = window.window_handle().expect("window_handle");
            let dh = window.display_handle().expect("display_handle");

            // Backend choice
            let mut backend = match self.backend_choice.as_str() {
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

            // Apply clear color from config
            match &mut backend {
                Backend::Gl(r) => {
                    r.as_mut().set_clear_color(self.cfg.render.clear_color);
                    r.as_mut().set_vsync(self.cfg.render.vsync);
                }
                Backend::Vk(r) => {
                    r.as_mut().set_clear_color(self.cfg.render.clear_color);
                    r.as_mut().set_vsync(self.cfg.render.vsync);
                    let mode = match self.cfg.render.vsync_mode {
                        VsyncMode::Fifo => cubic_render_vk::VkVsyncMode::Fifo,
                        VsyncMode::Mailbox => cubic_render_vk::VkVsyncMode::Mailbox,
                    };
                    r.as_mut().set_vsync_mode(mode);
                    r.as_mut().set_hdr_enabled(self.cfg.render.hdr);
                    let flavor = match self.cfg.render.hdr_flavor {
                        HdrFlavorCfg::PreferScrgb => cubic_render_vk::HdrFlavor::PreferScrgb,
                        HdrFlavorCfg::PreferHdr10 => cubic_render_vk::HdrFlavor::PreferHdr10,
                    };
                    r.as_mut().set_hdr_flavor(flavor);

                    match r
                        .as_mut()
                        .upload_mesh(test_scene::TRI_VERTS, test_scene::TRI_IDXS)
                    {
                        Ok(handle) => self.vk_mesh = Some(handle),
                        Err(e) => error!("upload_mesh failed: {e}"),
                    }
                }
            }

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
        }

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

        match event {
            WindowEvent::CloseRequested => {
                info!("CloseRequested");
                self.exiting = true;
                self.backend = None;
                self.window = None;
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
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
                        let _ = match backend {
                            Backend::Gl(r) => r.as_mut().resize(self.render_size),
                            Backend::Vk(r) => r.as_mut().resize(self.render_size),
                        };
                    }
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
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
                                if let Backend::Vk(r) = backend {
                                    r.as_mut().set_vsync(true);
                                    r.as_mut()
                                        .set_vsync_mode(cubic_render_vk::VkVsyncMode::Fifo);
                                }
                                if let Backend::Gl(r) = backend {
                                    r.as_mut().set_vsync(true);
                                }
                            }
                            (true, UnfocusedPolicy::VsyncOn) => match backend {
                                Backend::Vk(r) => {
                                    r.as_mut().set_vsync(self.cfg.render.vsync);
                                    let mode = match self.cfg.render.vsync_mode {
                                        VsyncMode::Fifo => cubic_render_vk::VkVsyncMode::Fifo,
                                        VsyncMode::Mailbox => cubic_render_vk::VkVsyncMode::Mailbox,
                                    };
                                    r.as_mut().set_vsync_mode(mode);
                                }
                                Backend::Gl(r) => {
                                    r.as_mut().set_vsync(self.cfg.render.vsync);
                                }
                            },
                            _ => {}
                        }
                    }

                    if let Some(window) = &self.window {
                        if focused {
                            let _ = window
                                .set_cursor_grab(CursorGrabMode::Locked)
                                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
                            window.set_cursor_visible(false);
                        } else {
                            let _ = window.set_cursor_grab(CursorGrabMode::None);
                            window.set_cursor_visible(true);
                        }
                    }

                    if focused {
                        self.next_frame_deadline = None;
                    } else {
                        // Can't reliably observe key-up events that happen
                        // while the window isn't focused, so don't leave
                        // movement keys stuck held across an alt-tab.
                        self.input.held_keys.clear();
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input
                        .set_key(code, event.state == ElementState::Pressed);
                }
            }

            WindowEvent::RedrawRequested => {
                if self.exiting || self.paused {
                    return;
                }

                let now = std::time::Instant::now();
                let dt = now.duration_since(self.last_frame_instant).as_secs_f32();
                self.last_frame_instant = now;
                self.apply_input(dt);

                if let Some(backend) = &mut self.backend {
                    if let Backend::Vk(r) = &mut *backend {
                        r.set_camera(self.camera);
                    }
                    if let (Backend::Vk(r), Some(handle)) = (&mut *backend, self.vk_mesh) {
                        r.draw_mesh(handle, test_scene::IDENTITY_PUSH);
                    }

                    let res = match backend {
                        Backend::Gl(r) => r.render(),
                        Backend::Vk(r) => r.render(),
                    };

                    match res {
                        Ok(()) => {
                            // count only frames that were actually rendered
                            self.frames = self.frames.saturating_add(1);
                        }
                        Err(e) => {
                            error!("render error: {e}");
                        }
                    }
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
            // Raw input is global, not window-scoped; only track it while
            // we actually own the cursor.
            if self.focused {
                self.input
                    .accumulate_mouse_delta(delta.0 as f32, delta.1 as f32);
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.exiting {
            return;
        }

        // 1) Decide target FPS for this frame (0 means "no cap here")
        let mut target_fps: u32 = 0;

        if self.paused {
            // window-size=0 or occluded → sleep
            event_loop.set_control_flow(ControlFlow::Wait);
            self.frames = 0;
            return;
        }

        if !self.focused {
            match self.cfg.render.unfocused {
                UnfocusedPolicy::Throttle => {
                    target_fps = self.cfg.render.unfocused_fps;
                }
                UnfocusedPolicy::VsyncOn => {
                    target_fps = 0; /* rely on vsync */
                }
                UnfocusedPolicy::None => { /* fall through to focused policy */ }
            }
        }

        if target_fps == 0 {
            if self.cfg.render.vsync {
                // Vsync: block until events, then redraw once
                event_loop.set_control_flow(ControlFlow::Wait);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            } else {
                // Optional cap when vsync off
                target_fps = self.cfg.render.fps_when_vsync_off;
                if target_fps == 0 {
                    // Uncapped: poll and keep drawing
                    event_loop.set_control_flow(ControlFlow::Poll);
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                }
            }
        }

        // 2) Throttled path (focused or unfocused)
        if target_fps > 0 {
            let now = std::time::Instant::now();
            let frame_dt =
                std::time::Duration::from_nanos(1_000_000_000u64 / target_fps.max(1) as u64);

            // Only request a redraw when it's time (or when there's no deadline yet).
            let need_redraw_now = match self.next_frame_deadline {
                None => true,
                Some(t) => now >= t,
            };

            if need_redraw_now {
                // Schedule the next deadline first, then ask for one redraw.
                let next = now + frame_dt;
                self.next_frame_deadline = Some(next);
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            } else {
                // Not time yet: just sleep until the stored deadline. Do NOT request redraw.
                event_loop
                    .set_control_flow(ControlFlow::WaitUntil(self.next_frame_deadline.unwrap()));
            }
        }

        // 3) FPS counter (unchanged)
        let now = std::time::Instant::now();
        if now.duration_since(self.last_fps_instant).as_secs_f32() >= 1.0 {
            info!("fps ~ {}", self.frames);
            self.frames = 0;
            self.last_fps_instant = now;
        }
    }
}

impl App {
    /// Apply accumulated mouse look and held-key movement to the camera.
    fn apply_input(&mut self, dt: f32) {
        let (dx, dy) = self.input.take_mouse_delta();
        self.camera.yaw -= dx * MOUSE_SENSITIVITY;
        self.camera.pitch =
            (self.camera.pitch - dy * MOUSE_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);

        let forward = self.camera.forward();
        let right = forward.cross(Vec3::Y).normalize_or_zero();

        let mut movement = Vec3::ZERO;
        if self.input.is_held(KeyCode::KeyW) {
            movement += forward;
        }
        if self.input.is_held(KeyCode::KeyS) {
            movement -= forward;
        }
        if self.input.is_held(KeyCode::KeyD) {
            movement += right;
        }
        if self.input.is_held(KeyCode::KeyA) {
            movement -= right;
        }
        if self.input.is_held(KeyCode::Space) {
            movement += Vec3::Y;
        }
        if self.input.is_held(KeyCode::ShiftLeft) {
            movement -= Vec3::Y;
        }

        self.camera.position += movement.normalize_or_zero() * MOVE_SPEED * dt;
    }
}

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let event_loop: EventLoop<()> = EventLoop::new()?;

    let mut app = App {
        backend_choice: args.backend,
        window: None,
        backend: None,
        render_size: RenderSize {
            width: 1,
            height: 1,
        },
        cfg: load_cfg(),
        exiting: false,
        frames: 0,
        last_fps_instant: std::time::Instant::now(),
        paused: false,
        focused: true,
        next_frame_deadline: None,
        vk_mesh: None,
        camera: Camera::default(),
        input: InputState::default(),
        last_frame_instant: std::time::Instant::now(),
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}
