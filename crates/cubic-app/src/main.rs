use anyhow::Result;
use tracing::{error, info};
use clap::Parser;

use cubic_core::init_tracing;
use cubic_render::{RenderSize, Renderer};
use cubic_render_gl::GlRenderer;
use cubic_render_vk::VkRenderer;

use cubic_platform::winit::{
  application::ApplicationHandler,
  event::{ElementState, KeyEvent, WindowEvent},
  event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
  keyboard::{Key, NamedKey},
  raw_window_handle::{HasDisplayHandle, HasWindowHandle},
  window::{Window, WindowId},
  dpi::PhysicalSize,
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
  /// Choose renderer backend: gl | vk
  #[arg(long, default_value = "gl")]
  backend: String,
}

enum Backend {
  Gl(GlRenderer),
  Vk(VkRenderer),
}

struct App {
  args: Args,
  window: Option<Window>,
  window_id: Option<WindowId>,
  backend: Option<Backend>,
  render_size: RenderSize,
}

impl App {
  fn new(args: Args) -> Self {
    Self {
      args,
      window: None,
      window_id: None,
      backend: None,
      render_size: RenderSize { width: 1280, height: 720 },
    }
  }
}

impl ApplicationHandler for App {
  fn resumed(&mut self, event_loop: &ActiveEventLoop) {
    // Create the window when the app is active (Wayland-friendly)
    if self.window.is_none() {
      let attrs = Window::default_attributes()
        .with_title("cubic")
        .with_inner_size(PhysicalSize::new(self.render_size.width, self.render_size.height));
      let window = event_loop.create_window(attrs).expect("create window");
      self.window_id = Some(window.id());

      // Grab initial size from compositor
      let size = window.inner_size();
      self.render_size = RenderSize { width: size.width.max(1), height: size.height.max(1) };

      // Raw handles for backends
      let wh = window.window_handle().expect("window handle");
      let dh = window.display_handle().expect("display handle");

      // Choose backend
      let backend = match self.args.backend.as_str() {
        "vk" => match VkRenderer::new(&wh, &dh, self.render_size) {
          Ok(vk) => Backend::Vk(vk),
          Err(e) => {
            error!("vk backend failed: {e}; falling back to gl");
            Backend::Gl(GlRenderer::new(&wh, &dh, self.render_size).expect("gl renderer"))
          }
        },
        _ => Backend::Gl(GlRenderer::new(&wh, &dh, self.render_size).expect("gl renderer")),
      };

      info!(
        "backend = {}",
        match &backend { Backend::Gl(_) => "gl", Backend::Vk(_) => "vk" }
      );

      self.backend = Some(backend);
      self.window = Some(window);
    }
  }

  fn window_event(&mut self, _event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
    match event {
      WindowEvent::CloseRequested => {
        // Mark for exit; actual exit is set in `about_to_wait`
        // (some platforms prefer exit from there)
        if let Some(w) = &self.window { w.request_redraw(); }
      }

      WindowEvent::KeyboardInput {
        event: KeyEvent {
          logical_key: Key::Named(NamedKey::Escape),
          state: ElementState::Pressed,
          ..
        },
        ..
      } => {
        if let Some(w) = &self.window { w.request_redraw(); }
        // Let about_to_wait handle exit uniformly
      }

      WindowEvent::Resized(new_size) => {
        self.render_size = RenderSize { width: new_size.width.max(1), height: new_size.height.max(1) };
        if let Some(backend) = &mut self.backend {
          let _ = match backend {
            Backend::Gl(r) => r.resize(self.render_size),
            Backend::Vk(r) => r.resize(self.render_size),
          };
        }
      }

      WindowEvent::RedrawRequested => {
        if let Some(backend) = &mut self.backend {
          let res = match backend {
            Backend::Gl(r) => r.render(),
            Backend::Vk(r) => r.render(),
          };
          if let Err(e) = res { error!("render error: {e}"); }
        }
      }

      _ => {}
    }
  }

  fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
    // Drive frames
    if let Some(w) = &self.window {
      w.request_redraw();
    }
    // Exit conditions
    if let Some(w) = &self.window {
      // If user asked to close (CloseRequested), winit will send it; we can exit here:
      // But to keep it simple, holding ESC to quit:
      // (Alternatively, track a flag set in CloseRequested)
    }
    event_loop.set_control_flow(ControlFlow::Poll);
  }
}

fn main() -> Result<()> {
  init_tracing();
  let args = Args::parse();
  let event_loop = EventLoop::new()?;

  let mut app = App::new(args);
  event_loop.run_app(&mut app)?;

  Ok(())
}
