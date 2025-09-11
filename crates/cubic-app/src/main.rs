use anyhow::Result;
use clap::Parser;
use cubic_core::init_tracing;
use cubic_render::{RenderSize, Renderer};
use cubic_render_gl::GlRenderer;
use cubic_render_vk::VkRenderer;
use tracing::{error, info};

use cubic_platform::winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

use serde::Deserialize;
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

    // shutdown guard
    exiting: bool,

    // fps
    frames: u32,
    last_fps_instant: std::time::Instant,
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
        if let Some(w) = &self.window {
            w.request_redraw();
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
                self.exiting = true; // stop rendering new frames
                self.backend = None;
                self.window = None;
                event_loop.exit();
            }
            WindowEvent::Resized(new_size) => {
                let w = new_size.width.max(1);
                let h = new_size.height.max(1);
                if (w, h) == (self.render_size.width, self.render_size.height) {
                    return; // nothing to do
                }
                self.render_size = RenderSize {
                    width: w,
                    height: h,
                };

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

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.exiting {
            return;
        }
        // Render
        if let Some(backend) = &mut self.backend {
            let res = match backend {
                Backend::Gl(r) => r.render(),
                Backend::Vk(r) => r.render(),
            };

            match res {
                Ok(()) => {
                    self.frames += 1;
                }
                Err(e) => {
                    error!("render error: {e}");
                }
            }
        }
        // FPS counter
        let now = std::time::Instant::now();

        if now.duration_since(self.last_fps_instant).as_secs_f32() >= 1.0 {
            info!("fps ~ {}", self.frames);
            self.frames = 0;
            self.last_fps_instant = now;
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
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
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}
