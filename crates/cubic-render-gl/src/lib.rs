use anyhow::{anyhow, Context, Result};
use tracing::info;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use cubic_render::{RenderSize, Renderer};

use glow::HasContext as _;
use glutin::{
  config::{ConfigTemplateBuilder, ConfigSurfaceTypes},
  context::{ContextApi, ContextAttributesBuilder, NotCurrentContext, PossiblyCurrentContext, Version},
  display::{Display, DisplayApiPreference},
  prelude::*,
  surface::{Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface},
};
use std::num::NonZeroU32;

pub struct GlRenderer {
  display: Display,
  context: PossiblyCurrentContext,
  surface: Surface<WindowSurface>,
  gl: glow::Context,
  size: RenderSize,
}

impl GlRenderer {
  fn make_current(
    display: &Display,
    window_handle: RawWindowHandle,
    size: RenderSize,
  ) -> Result<(PossiblyCurrentContext, Surface<WindowSurface>, glow::Context)> {
    // Helper that tries with a specific template
    fn make_with_template(
      display: &Display,
      window_handle: RawWindowHandle,
      size: RenderSize,
      template: ConfigTemplateBuilder,
    ) -> Result<(PossiblyCurrentContext, Surface<WindowSurface>, glow::Context)> {
      // Find a config
      let mut configs = unsafe { display.find_configs(template.build()) }.context("find_configs")?;
      let config = configs
        .next()
        .ok_or_else(|| anyhow!("no GL configs returned by display"))?;

      // Non-zero dims (Wayland requirement)
      let w = NonZeroU32::new(size.width.max(1)).unwrap();
      let h = NonZeroU32::new(size.height.max(1)).unwrap();

      // Create a window surface
      let sattrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(window_handle, w, h);
      let surface = unsafe { display.create_window_surface(&config, &sattrs) }
        .context("create_window_surface")?;

      // Make a 3.3 core context current
      let ctx_attrs = ContextAttributesBuilder::new()
        .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
        .build(Some(window_handle));
      let not_current: NotCurrentContext =
        unsafe { display.create_context(&config, &ctx_attrs) }.context("create_context")?;
      let context = not_current.make_current(&surface).context("make_current")?;

      // Load GL procs
      let gl = unsafe {
        glow::Context::from_loader_function(|s| {
          display.get_proc_address(&std::ffi::CString::new(s).unwrap()) as *const _
        })
      };

      // Try vsync (ok if it fails)
      let _ = surface.set_swap_interval(&context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()));
      Ok((context, surface, gl))
    }

    // First try: window-capable + "compatible with native window" hint
    let strict = ConfigTemplateBuilder::new()
      .with_surface_type(ConfigSurfaceTypes::WINDOW)
      .compatible_with_native_window(window_handle);

    if let Ok(ok) = make_with_template(display, window_handle, size, strict) {
      return Ok(ok);
    }

    // Retry: window-capable without the compatibility hint (some stacks are picky)
    let relaxed = ConfigTemplateBuilder::new().with_surface_type(ConfigSurfaceTypes::WINDOW);
    if let Ok(ok) = make_with_template(display, window_handle, size, relaxed) {
      return Ok(ok);
    }

    // Last resort: minimal ask (no depth/stencil)
    let fallback = ConfigTemplateBuilder::new()
      .with_surface_type(ConfigSurfaceTypes::WINDOW)
      .with_depth_size(0)
      .with_stencil_size(0);
    make_with_template(display, window_handle, size, fallback)
  }
}

impl Renderer for GlRenderer {
  fn new(
    window: &dyn HasWindowHandle,
    display_handle: &dyn HasDisplayHandle,
    size: RenderSize,
  ) -> Result<Self> {
    // Raw handles
    let wh: RawWindowHandle = window.window_handle()?.as_raw();
    let dh: RawDisplayHandle = display_handle.display_handle()?.as_raw();

    // Use EGL (works on Wayland and X11 with Mesa)
    let display = unsafe { Display::new(dh, DisplayApiPreference::Egl) }
      .context("Display::new(EGL)")?;

    let (context, surface, gl) = Self::make_current(&display, wh, size)?;
    info!("GL (EGL) context created; window should be visible now");

    Ok(Self {
      display,
      context,
      surface,
      gl,
      size,
    })
  }

  fn resize(&mut self, size: RenderSize) -> Result<()> {
    self.size = size;
    let w = NonZeroU32::new(size.width.max(1)).unwrap();
    let h = NonZeroU32::new(size.height.max(1)).unwrap();
    self.surface.resize(&self.context, w, h);
    Ok(())
  }

  fn render(&mut self) -> Result<()> {
    unsafe {
      self.gl.viewport(0, 0, self.size.width as i32, self.size.height as i32);
      self.gl.clear_color(0.02, 0.02, 0.04, 1.0);
      self.gl.clear(glow::COLOR_BUFFER_BIT);
    }
    self.surface.swap_buffers(&self.context).context("swap_buffers")?;
    Ok(())
  }
}
