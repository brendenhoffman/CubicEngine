// SPDX-License-Identifier: CEPL-1.0
//! Renderer-backend abstraction: a small trait over the concrete GL/Vulkan
//! renderers so the rest of the app doesn't match on which one is active.

use crate::config::{HdrFlavorCfg, MipmapMode, RenderCfg, TextureFilter, VsyncMode};
use anyhow::Result;
use cubic_math::Camera;
use cubic_render::{MeshHandle, PushData, RenderSize, Renderer, Vertex};
use cubic_render_gl::GlRenderer;
use cubic_render_vk::{Filter, HdrFlavor, SamplerMipmapMode, VkRenderer, VkVsyncMode};
use egui::{ClippedPrimitive, TexturesDelta};

pub(crate) trait RendererBackend {
    fn resize(&mut self, size: RenderSize) -> Result<()>;
    fn set_clear_color(&mut self, rgba: [f32; 4]);
    fn set_vsync(&mut self, on: bool);
    fn configure_advanced(&mut self, cfg: &RenderCfg);
    fn upload_mesh(&mut self, verts: &[Vertex], idxs: &[u32]) -> Result<MeshHandle>;
    fn set_camera(&mut self, camera: Camera);
    fn draw_mesh(&mut self, handle: MeshHandle, push: PushData);
    fn render(&mut self) -> Result<()>;
    fn free_mesh(&mut self, _handle: MeshHandle) {} // default no-op
    fn upload_texture(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<u32>;
    fn queue_egui(
        &mut self,
        textures_delta: TexturesDelta,
        paint_jobs: Vec<ClippedPrimitive>,
        w: u32,
        h: u32,
        ppp: f32,
    );
}

pub(crate) enum Backend {
    Gl(Box<GlRenderer>),
    Vk(Box<VkRenderer>),
}

impl RendererBackend for Backend {
    fn resize(&mut self, size: RenderSize) -> Result<()> {
        match self {
            Backend::Gl(r) => r.resize(size),
            Backend::Vk(r) => r.resize(size),
        }
    }

    fn set_clear_color(&mut self, rgba: [f32; 4]) {
        match self {
            Backend::Gl(r) => r.set_clear_color(rgba),
            Backend::Vk(r) => r.set_clear_color(rgba),
        }
    }

    fn set_vsync(&mut self, on: bool) {
        match self {
            Backend::Gl(r) => r.set_vsync(on),
            Backend::Vk(r) => r.set_vsync(on),
        }
    }

    fn configure_advanced(&mut self, cfg: &RenderCfg) {
        // GL has no advanced knobs yet.
        if let Backend::Vk(r) = self {
            let mode = match cfg.vsync_mode {
                VsyncMode::Fifo => VkVsyncMode::Fifo,
                VsyncMode::Mailbox => VkVsyncMode::Mailbox,
            };
            r.set_vsync_mode(mode);
            r.set_hdr_enabled(cfg.hdr);
            let flavor = match cfg.hdr_flavor {
                HdrFlavorCfg::PreferScrgb => HdrFlavor::PreferScrgb,
                HdrFlavorCfg::PreferHdr10 => HdrFlavor::PreferHdr10,
            };
            r.set_hdr_flavor(flavor);

            let filter = match cfg.texture_filter {
                TextureFilter::Nearest => Filter::NEAREST,
                TextureFilter::Linear => Filter::LINEAR,
            };
            let mipmap_mode = match cfg.mipmap_mode {
                MipmapMode::Nearest => SamplerMipmapMode::NEAREST,
                MipmapMode::Linear => SamplerMipmapMode::LINEAR,
            };
            r.set_sampler_config(filter, filter, mipmap_mode, cfg.anisotropy, cfg.lod_bias);
        }
    }

    fn upload_mesh(&mut self, verts: &[Vertex], idxs: &[u32]) -> Result<MeshHandle> {
        match self {
            // GL mesh API not yet implemented; uploaded meshes are silently
            // dropped until the GL backend card is complete.
            Backend::Gl(_) => Ok(MeshHandle(u32::MAX)),
            Backend::Vk(r) => r.upload_mesh(verts, idxs),
        }
    }

    fn set_camera(&mut self, camera: Camera) {
        match self {
            Backend::Gl(_) => {} // GL camera via uniforms — not yet implemented.
            Backend::Vk(r) => r.set_camera(camera),
        }
    }

    fn draw_mesh(&mut self, handle: MeshHandle, push: PushData) {
        match self {
            Backend::Gl(_) => {} // GL draw_mesh — not yet implemented.
            Backend::Vk(r) => r.draw_mesh(handle, push),
        }
    }

    fn free_mesh(&mut self, handle: MeshHandle) {
        match self {
            Backend::Gl(_) => {}
            Backend::Vk(r) => r.free_mesh(handle),
        }
    }

    fn render(&mut self) -> Result<()> {
        match self {
            Backend::Gl(r) => r.render(),
            Backend::Vk(r) => r.render(),
        }
    }

    fn upload_texture(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<u32> {
        match self {
            // GL texture API not yet implemented.
            Backend::Gl(_) => Ok(0),
            Backend::Vk(r) => r.upload_texture(pixels, width, height),
        }
    }

    fn queue_egui(
        &mut self,
        textures_delta: TexturesDelta,
        paint_jobs: Vec<ClippedPrimitive>,
        w: u32,
        h: u32,
        ppp: f32,
    ) {
        match self {
            Backend::Gl(_) => {}
            Backend::Vk(r) => r.queue_egui(textures_delta, paint_jobs, w, h, ppp),
        }
    }
}
