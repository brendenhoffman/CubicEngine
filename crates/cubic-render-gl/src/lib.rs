// SPDX-License-Identifier: CEPL-1.0
use anyhow::{anyhow, Context, Result};
use cubic_render::{RenderSize, Renderer};
use glow::HasContext as _;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};

use glutin::{
    config::ConfigTemplateBuilder,
    context::{
        ContextApi, ContextAttributesBuilder, NotCurrentContext, PossiblyCurrentContext, Version,
    },
    display::{Display, DisplayApiPreference},
    prelude::*,
    surface::{Surface, SurfaceAttributesBuilder, SwapInterval, WindowSurface},
};

use std::num::NonZeroU32;

pub struct GlRenderer {
    //display: Display,
    context: PossiblyCurrentContext,
    surface: Surface<WindowSurface>,
    gl: glow::Context,
    size: RenderSize,
    clear: [f32; 4],
    program: glow::Program,
    vao: glow::VertexArray,
    vsync: bool,
}

fn compile_program(gl: &glow::Context) -> Result<glow::Program> {
    unsafe {
        let vs = gl
            .create_shader(glow::VERTEX_SHADER)
            .map_err(anyhow::Error::msg)?;
        let fs = gl
            .create_shader(glow::FRAGMENT_SHADER)
            .map_err(anyhow::Error::msg)?;

        let vert_src = r#"#version 330 core
        out vec3 vColor;
        void main() {
          vec2 pos[3] = vec2[3](
            vec2( 0.0,  0.6),
            vec2(-0.5, -0.4),
            vec2( 0.5, -0.4)
          );
          vec3 col[3] = vec3[3](
            vec3(1,0,0),
            vec3(0,1,0),
            vec3(0,0,1)
          );
          gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
          vColor = col[gl_VertexID];
        }"#;

        let frag_src = r#"#version 330 core
        in vec3 vColor;
        out vec4 outColor;
        void main(){ outColor = vec4(vColor, 1.0); }"#;

        gl.shader_source(vs, vert_src);
        gl.compile_shader(vs);

        if !gl.get_shader_compile_status(vs) {
            return Err(anyhow!("GL vert compile: {}", gl.get_shader_info_log(vs)));
        }

        gl.shader_source(fs, frag_src);
        gl.compile_shader(fs);

        if !gl.get_shader_compile_status(fs) {
            return Err(anyhow!("GL frag compile: {}", gl.get_shader_info_log(fs)));
        }

        let program = gl.create_program().map_err(anyhow::Error::msg)?;

        gl.attach_shader(program, vs);
        gl.attach_shader(program, fs);
        gl.link_program(program);

        if !gl.get_program_link_status(program) {
            return Err(anyhow::anyhow!(
                "GL link: {}",
                gl.get_program_info_log(program)
            ));
        }

        gl.detach_shader(program, vs);
        gl.detach_shader(program, fs);
        gl.delete_shader(vs);
        gl.delete_shader(fs);

        Ok(program)
    }
}

impl GlRenderer {
    fn make_current(
        display: &Display,
        window_handle: RawWindowHandle,
        size: RenderSize,
    ) -> Result<(
        PossiblyCurrentContext,
        Surface<WindowSurface>,
        glow::Context,
    )> {
        let template = ConfigTemplateBuilder::new().build();
        let mut configs = unsafe { display.find_configs(template) }.context("find_configs")?;
        let config = configs.next().ok_or_else(|| anyhow!("no GL configs"))?;
        let w = NonZeroU32::new(size.width.max(1)).unwrap();
        let h = NonZeroU32::new(size.height.max(1)).unwrap();

        let sattrs = SurfaceAttributesBuilder::<WindowSurface>::new().build(window_handle, w, h);
        let surface = unsafe { display.create_window_surface(&config, &sattrs) }
            .context("create_window_surface")?;
        let ctx_attrs = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(3, 3))))
            .build(Some(window_handle));
        let not_current: NotCurrentContext =
            unsafe { display.create_context(&config, &ctx_attrs) }.context("create_context")?;

        let context = not_current.make_current(&surface).context("make_current")?;

        let gl = unsafe {
            glow::Context::from_loader_function(|s| {
                display.get_proc_address(&std::ffi::CString::new(s).unwrap()) as *const _
            })
        };

        let _ =
            surface.set_swap_interval(&context, SwapInterval::Wait(NonZeroU32::new(1).unwrap()));

        Ok((context, surface, gl))
    }
}

impl Renderer for GlRenderer {
    fn new(
        window: &dyn HasWindowHandle,
        display_handle: &dyn HasDisplayHandle,
        size: RenderSize,
    ) -> Result<Self> {
        let wh = window
            .window_handle()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .as_raw();
        let dh = display_handle
            .display_handle()
            .map_err(|e| anyhow::anyhow!("{e}"))?
            .as_raw();

        let display =
            unsafe { Display::new(dh, DisplayApiPreference::Egl) }.context("Display::new")?;

        let (context, surface, gl) = Self::make_current(&display, wh, size)?;
        let program = compile_program(&gl)?;
        let vao = unsafe { gl.create_vertex_array().map_err(anyhow::Error::msg)? };

        unsafe {
            gl.bind_vertex_array(Some(vao));
            gl.bind_vertex_array(None);
            gl.enable(glow::FRAMEBUFFER_SRGB);
            gl.enable(glow::CULL_FACE);
            gl.front_face(glow::CCW);
            gl.cull_face(glow::BACK);
            gl.bind_vertex_array(None);
            gl.disable(glow::DEPTH_TEST);
        }

        let initial_vsync = true;

        let _ = surface.set_swap_interval(
            &context,
            if initial_vsync {
                SwapInterval::Wait(NonZeroU32::new(1).unwrap())
            } else {
                SwapInterval::DontWait
            },
        );

        Ok(Self {
            //display,
            context,
            surface,
            gl,
            size,
            clear: [0.02, 0.02, 0.04, 1.0],
            program,
            vao,
            vsync: initial_vsync,
        })
    }

    fn resize(&mut self, size: RenderSize) -> Result<()> {
        self.size = size;

        let w = NonZeroU32::new(size.width).unwrap();
        let h = NonZeroU32::new(size.height).unwrap();

        self.surface.resize(&self.context, w, h);
        self.set_vsync(self.vsync);

        Ok(())
    }
    fn set_clear_color(&mut self, rgba: [f32; 4]) {
        self.clear = rgba;
    }
    fn render(&mut self) -> Result<()> {
        if self.size.width == 0 || self.size.height == 0 {
            return Ok(());
        }

        unsafe {
            self.gl
                .viewport(0, 0, self.size.width as i32, self.size.height as i32);
            self.gl
                .clear_color(self.clear[0], self.clear[1], self.clear[2], self.clear[3]);

            self.gl.clear(glow::COLOR_BUFFER_BIT);
            self.gl.use_program(Some(self.program));
            self.gl.bind_vertex_array(Some(self.vao));
            self.gl.draw_arrays(glow::TRIANGLES, 0, 3);
            self.gl.bind_vertex_array(None);
            self.gl.use_program(None);
        }

        self.surface
            .swap_buffers(&self.context)
            .context("swap_buffers")?;

        Ok(())
    }
}
