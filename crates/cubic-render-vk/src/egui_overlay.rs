// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
//! Egui overlay integration (GPU plumbing only — no egui::Context or input
//! handling here; that lives in cubic-app).

use anyhow::Result;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::sync::{Arc, Mutex};

use crate::VkRenderer;

pub(crate) struct EguiFrame {
    pub(crate) textures_delta: egui::TexturesDelta,
    pub(crate) paint_jobs: Vec<egui::ClippedPrimitive>,
    pub(crate) screen_width: u32,
    pub(crate) screen_height: u32,
    pub(crate) pixels_per_point: f32,
}

/// True if `format` needs `Options::srgb_framebuffer = true` for egui:
/// egui always outputs linear color, so the sRGB conversion must happen
/// either via the swapchain image's sRGB view (B8G8R8A8/R8G8B8A8_SRGB) or
/// be treated as already-linear by the shader (R16G16B16A16_SFLOAT scRGB).
#[inline]
fn format_needs_srgb_egui(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::B8G8R8A8_SRGB | vk::Format::R8G8B8A8_SRGB | vk::Format::R16G16B16A16_SFLOAT
    )
}

/// Builds the egui overlay renderer. Uses its own private gpu-allocator
/// instance (rather than sharing the caller's) so wiring this in doesn't
/// require touching any of that allocator's many existing call sites; the
/// only owner of the returned Arc is the renderer itself once
/// `with_gpu_allocator` returns, so it's torn down when the renderer is
/// (see the explicit `egui_renderer.take()` in VkRenderer's Drop).
pub(crate) fn build_egui_renderer(
    instance: &ash::Instance,
    device: &ash::Device,
    phys: vk::PhysicalDevice,
    depth_format: vk::Format,
    color_format: vk::Format,
    in_flight_frames: usize,
) -> Result<egui_ash_renderer::Renderer> {
    let egui_gpu_allocator = Arc::new(Mutex::new(Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device: phys,
        debug_settings: Default::default(),
        buffer_device_address: false,
        allocation_sizes: Default::default(),
    })?));
    Ok(egui_ash_renderer::Renderer::with_gpu_allocator(
        egui_gpu_allocator,
        device.clone(),
        egui_ash_renderer::DynamicRendering {
            color_attachment_format: color_format,
            // egui draws inside the same cmd_begin_rendering scope as the
            // scene (see record_egui), which always binds the real depth
            // attachment — Vulkan requires the bound pipeline's declared
            // depthAttachmentFormat to match whenever one is bound, even if
            // this pipeline doesn't test/write depth (both disabled via
            // Options below).
            depth_attachment_format: Some(depth_format),
        },
        egui_ash_renderer::Options {
            srgb_framebuffer: format_needs_srgb_egui(color_format),
            in_flight_frames: in_flight_frames.max(1),
            ..Default::default()
        },
    )?)
}

impl VkRenderer {
    /// Stage an egui frame (tessellated paint jobs + texture updates) to be
    /// drawn on top of the scene during the next render() call. Consumed
    /// and cleared by that call; call again each frame you want an overlay.
    pub fn queue_egui(
        &mut self,
        textures_delta: egui::TexturesDelta,
        paint_jobs: Vec<egui::ClippedPrimitive>,
        screen_width: u32,
        screen_height: u32,
        pixels_per_point: f32,
    ) {
        self.egui_pending = Some(EguiFrame {
            textures_delta,
            paint_jobs,
            screen_width,
            screen_height,
            pixels_per_point,
        });
    }

    /// Draw a staged egui frame (see queue_egui) into `cmd`. Must be called
    /// while `cmd` is inside an active dynamic-rendering scope (between
    /// begin_rendering and cmd_end_rendering) targeting the same color
    /// attachment format the egui renderer was built with, and before that
    /// image transitions to PRESENT_SRC_KHR. No-op if nothing is staged.
    pub(crate) fn record_egui(&mut self, cmd: vk::CommandBuffer) -> Result<()> {
        let Some(frame) = self.egui_pending.take() else {
            return Ok(());
        };
        let Some(renderer) = self.egui_renderer.as_mut() else {
            return Ok(());
        };
        // Must run before cmd_draw uploads/binds the textures it references.
        renderer.set_textures(
            self.queue,
            self.cmd_pool,
            frame.textures_delta.set.as_slice(),
        )?;
        renderer.cmd_draw(
            cmd,
            vk::Extent2D {
                width: frame.screen_width,
                height: frame.screen_height,
            },
            frame.pixels_per_point,
            &frame.paint_jobs,
        )?;
        // Safe to free immediately: cmd_draw only records commands (it
        // doesn't submit), and set_textures's uploads are already
        // synchronously complete (queue_wait_idle) by the time it returns.
        renderer.free_textures(frame.textures_delta.free.as_slice())?;
        Ok(())
    }
}
