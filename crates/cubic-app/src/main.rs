use anyhow::Result;
use clap::Parser;
use tracing::{error, info};

use cubic_core::init_tracing;
use cubic_render::{RenderSize, Renderer};
use cubic_render_gl::GlRenderer;
use cubic_render_vk::VkRenderer;

use cubic_platform::winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Choose renderer backend: gl | vk
    #[arg(long, default_value = "vk")]
    backend: String,
}

enum Backend {
    Gl(GlRenderer),
    Vk(VkRenderer),
}

struct App {
    backend_choice: String,
    window: Option<Window>,
    backend: Option<Backend>,
    render_size: RenderSize,
    exiting: bool,
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

            // Raw handles for renderer creation
            let wh = window.window_handle().expect("window_handle");
            let dh = window.display_handle().expect("display_handle");

            // Pick backend
            let backend = match self.backend_choice.as_str() {
                "gl" => Backend::Gl(GlRenderer::new(&wh, &dh, self.render_size).expect("GL init")),
                _ => match VkRenderer::new(&wh, &dh, self.render_size) {
                    Ok(vk) => Backend::Vk(vk),
                    Err(e) => {
                        error!("vk init failed: {e}; falling back to gl");
                        Backend::Gl(GlRenderer::new(&wh, &dh, self.render_size).expect("GL init"))
                    }
                },
            };

            info!(
                "backend = {}",
                match &backend {
                    Backend::Gl(_) => "gl",
                    Backend::Vk(_) => "vk",
                }
            );

            self.window = Some(window);
            self.backend = Some(backend);
        }

        // Continuous rendering
        event_loop.set_control_flow(ControlFlow::Poll);
        if let Some(w) = &self.window {
            w.request_redraw(); // make Wayland map the window immediately
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
                if let Some(w) = &self.window {
                    self.backend = None;
                    self.window = None;
                }
                event_loop.exit();
            }

            // In winit 0.30, this also covers scale-factor/DPI changes
            WindowEvent::Resized(new_size) => {
                self.render_size = RenderSize {
                    width: new_size.width.max(1),
                    height: new_size.height.max(1),
                };
                if let Some(backend) = &mut self.backend {
                    let _ = match backend {
                        Backend::Gl(r) => r.resize(self.render_size),
                        Backend::Vk(r) => r.resize(self.render_size),
                    };
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }

            // There is no RedrawRequested callback method on ApplicationHandler in 0.30.
            // We render in about_to_wait instead (continuous).
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.exiting {
            return;
        }
        // Render one frame every tick; keeps things simple and guarantees first frame on Wayland.
        if let Some(backend) = &mut self.backend {
            let res = match backend {
                Backend::Gl(r) => r.render(),
                Backend::Vk(r) => r.render(),
            };
            if let Err(e) = res {
                error!("render error: {e}");
            }
        }
        if let Some(w) = &self.window {
            w.request_redraw(); // keep frames going
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
        exiting: false,
    };

    event_loop.run_app(&mut app)?;
    Ok(())
}
