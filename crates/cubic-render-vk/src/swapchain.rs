// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::Result;
use ash::khr::{surface, swapchain};
use ash::vk;
use cubic_render::RenderSize;

use crate::pipeline::{create_pipeline, PipelineConfig};
use crate::resources::{
    create_depth_resources, create_frame_uniforms_and_sets, create_indirect_draw_resources,
};
use crate::sync::FrameSync;
use crate::{DeferredDrop, GpuResource, VkRenderer};

#[derive(Clone, Copy, Debug)]
pub enum VkVsyncMode {
    Fifo,    // Target monitor refresh rate
    Mailbox, // Smart Vsync, fps uncapped
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrFlavor {
    PreferScrgb, // FP16 scRGB first, then HDR10
    PreferHdr10, // HDR10 first, then scRGB
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct SwapchainConfig {
    pub(crate) hint: RenderSize,
    pub(crate) vsync: bool,
    pub(crate) vsync_mode: VkVsyncMode,
    pub(crate) want_hdr: bool,
    pub(crate) allow_extended_colorspace: bool,
    pub(crate) hdr_flavor: HdrFlavor,
}

pub(crate) struct SwapchainBundle {
    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) format: vk::Format,
    pub(crate) extent: vk::Extent2D,
    pub(crate) images: Vec<vk::Image>,
    pub(crate) image_views: Vec<vk::ImageView>,
    pub(crate) color_space: vk::ColorSpaceKHR,
}

#[inline]
fn fmt_name(f: ash::vk::Format) -> &'static str {
    match f {
        ash::vk::Format::B8G8R8A8_UNORM => "B8G8R8A8_UNORM",
        ash::vk::Format::B8G8R8A8_SRGB => "B8G8R8A8_SRGB",
        ash::vk::Format::R8G8B8A8_SRGB => "R8G8B8A8_SRGB",
        ash::vk::Format::R8G8B8A8_UNORM => "R8G8B8A8_UNORM",
        vk::Format::A2B10G10R10_UNORM_PACK32 => "A2B10G10R10_UNORM",
        vk::Format::A2R10G10B10_UNORM_PACK32 => "A2R10G10B10_UNORM",
        vk::Format::R16G16B16A16_SFLOAT => "R16G16B16A16_SFLOAT",
        _ => "OTHER",
    }
}

#[inline]
fn cs_name(cs: ash::vk::ColorSpaceKHR) -> &'static str {
    match cs {
        ash::vk::ColorSpaceKHR::SRGB_NONLINEAR => "SRGB_NONLINEAR",
        ash::vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT => "DISPLAY_P3_NONLINEAR",
        ash::vk::ColorSpaceKHR::HDR10_ST2084_EXT => "HDR10_ST2084",
        ash::vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => "EXTENDED_SRGB_LINEAR",
        ash::vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT => "EXTENDED_SRGB_NONLINEAR",
        _ => "OTHER",
    }
}

#[inline]
fn pm_name(m: ash::vk::PresentModeKHR) -> &'static str {
    match m {
        ash::vk::PresentModeKHR::FIFO => "FIFO",
        ash::vk::PresentModeKHR::MAILBOX => "MAILBOX",
        ash::vk::PresentModeKHR::IMMEDIATE => "IMMEDIATE",
        ash::vk::PresentModeKHR::FIFO_RELAXED => "FIFO_RELAXED",
        _ => "OTHER",
    }
}

#[inline]
fn choose_present_mode(
    modes: &[vk::PresentModeKHR],
    vsync: bool,
    mode: VkVsyncMode,
) -> vk::PresentModeKHR {
    if !vsync {
        return [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::FIFO,
        ]
        .into_iter()
        .find(|m| modes.contains(m))
        .unwrap_or(vk::PresentModeKHR::FIFO);
    }

    match mode {
        VkVsyncMode::Mailbox => [vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::FIFO]
            .into_iter()
            .find(|m| modes.contains(m))
            .unwrap_or(vk::PresentModeKHR::FIFO),
        VkVsyncMode::Fifo => [vk::PresentModeKHR::FIFO, vk::PresentModeKHR::MAILBOX]
            .into_iter()
            .find(|m| modes.contains(m))
            .unwrap_or(vk::PresentModeKHR::FIFO),
    }
}

#[inline]
fn extent_from_caps(caps: &vk::SurfaceCapabilitiesKHR, want: RenderSize) -> vk::Extent2D {
    if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: want
                .width
                .clamp(caps.min_image_extent.width, caps.max_image_extent.width),
            height: want
                .height
                .clamp(caps.min_image_extent.height, caps.max_image_extent.height),
        }
    }
}

fn pick_surface_format(
    formats: &[vk::SurfaceFormatKHR],
    want_hdr: bool,
    allow_extended: bool,
    flavor: HdrFlavor,
) -> (vk::SurfaceFormatKHR, &'static str) {
    if want_hdr && allow_extended {
        let try_hdr10 = || {
            formats
                .iter()
                .copied()
                .find(|f| {
                    f.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT
                        && (f.format == vk::Format::A2B10G10R10_UNORM_PACK32
                            || f.format == vk::Format::A2R10G10B10_UNORM_PACK32
                            || f.format == vk::Format::R16G16B16A16_SFLOAT)
                })
                .map(|f| (f, "hdr10_pq"))
        };
        let try_scrgb = || {
            formats
                .iter()
                .copied()
                .find(|f| {
                    (f.color_space == vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT
                        || f.color_space == vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT)
                        && f.format == vk::Format::R16G16B16A16_SFLOAT
                })
                .map(|f| (f, "scrgb_fp16"))
        };

        return match flavor {
            HdrFlavor::PreferScrgb => try_scrgb().or_else(try_hdr10),
            HdrFlavor::PreferHdr10 => try_hdr10().or_else(try_scrgb),
        }
        .unwrap_or_else(|| (formats[0], "driver_default_hdr"));
    }

    // SDR fallbacks
    if let Some(f) = formats
        .iter()
        .copied()
        .find(|f| f.format == vk::Format::B8G8R8A8_SRGB)
    {
        return (f, "sdr_bgra8_srgb");
    }
    if let Some(f) = formats
        .iter()
        .copied()
        .find(|f| f.format == vk::Format::R8G8B8A8_SRGB)
    {
        return (f, "sdr_rgba8_srgb");
    }
    if let Some(f) = formats.iter().copied().find(|f| {
        f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
    }) {
        return (f, "sdr_bgra8_unorm_srgbcs");
    }

    (formats[0], "driver_default")
}

fn make_color_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
) -> anyhow::Result<vk::ImageView> {
    let sub = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let iv = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        components: vk::ComponentMapping::default(),
        subresource_range: sub,
        ..Default::default()
    };
    Ok(unsafe { device.create_image_view(&iv, None)? })
}

// ORDER NOTE: must be called AFTER creating the (new) swapchain and BEFORE first present.
// Scope: only HDR10/PQ swapchains need metadata; scRGB doesn't use VK_EXT_hdr_metadata.
pub(crate) fn create_hdr_metadata_if_needed(
    instance: &ash::Instance,
    device: &ash::Device,
    has_hdr_meta: bool,
    color_space: vk::ColorSpaceKHR,
    swapchain: vk::SwapchainKHR,
) {
    // Fast bailouts: no extension, or not an HDR10 PQ surface
    if !has_hdr_meta || color_space != vk::ColorSpaceKHR::HDR10_ST2084_EXT {
        return;
    }

    let hdr = ash::ext::hdr_metadata::Device::new(instance, device);

    // Basic BT.2020 primaries + D65 white and typical luminance values.
    // Adjust later if you want per-display calibration or content-driven values.
    let metadata = vk::HdrMetadataEXT {
        s_type: vk::StructureType::HDR_METADATA_EXT,
        display_primary_red: vk::XYColorEXT { x: 0.708, y: 0.292 },
        display_primary_green: vk::XYColorEXT { x: 0.170, y: 0.797 },
        display_primary_blue: vk::XYColorEXT { x: 0.131, y: 0.046 },
        white_point: vk::XYColorEXT {
            x: 0.3127,
            y: 0.3290,
        },
        max_luminance: 1000.0,
        min_luminance: 0.001,
        max_content_light_level: 1000.0,
        max_frame_average_light_level: 400.0,
        ..Default::default()
    };

    // Apply to the current swapchain. Safe to reapply on recreate.
    unsafe { hdr.set_hdr_metadata(&[swapchain], std::slice::from_ref(&metadata)) };
}

pub(crate) fn create_swapchain_bundle(
    device: &ash::Device,
    surf_i: &surface::Instance,
    swap_d: &swapchain::Device,
    phys: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    old_swapchain: vk::SwapchainKHR,
    cfg: SwapchainConfig,
) -> Result<SwapchainBundle> {
    // --- Query surface capabilities / formats / present modes ---
    // capabilities: image counts, transforms, current extent (or UINT_MAX for free-size)
    let caps = unsafe { surf_i.get_physical_device_surface_capabilities(phys, surface)? };
    // (format, colorspace) pairs exposed by WSI; must choose one
    let formats = unsafe { surf_i.get_physical_device_surface_formats(phys, surface)? };
    // present modes: FIFO is always available; MAILBOX/IMMEDIATE are optional
    let modes = unsafe { surf_i.get_physical_device_surface_present_modes(phys, surface)? };

    tracing::info!(
        "hdr_request={} allow_extended_colorspace={}",
        cfg.want_hdr,
        cfg.allow_extended_colorspace,
    );

    // --- Choose (format, colorspace) and present mode based on config ---
    // Note: pick_surface_format encodes your HDR flavor policy (HDR10 vs scRGB preference).
    let (surf_format, pick_reason) = pick_surface_format(
        &formats,
        cfg.want_hdr,
        cfg.allow_extended_colorspace,
        cfg.hdr_flavor,
    );
    // Prefer MAILBOX if vsync==true && mode==Mailbox (& available), else FIFO fallback
    let present_mode = choose_present_mode(&modes, cfg.vsync, cfg.vsync_mode);
    // Resolve desired extent respecting min/max if current_extent is UINT_MAX (free-size)
    let extent = extent_from_caps(&caps, cfg.hint);

    tracing::info!(
        "reason: {}, format: {} / {}, present_mode: {}, vsync={}, mode={:?}, extent: {}x{}, images(min={} → picked={})",
        pick_reason,
        fmt_name(surf_format.format),
        cs_name(surf_format.color_space),
        pm_name(present_mode),
        cfg.vsync,
        cfg.vsync_mode,
        extent.width, extent.height,
        caps.min_image_count,
        if caps.max_image_count == 0 { caps.min_image_count + 1 }
        else { (caps.min_image_count + 1).min(caps.max_image_count) }
    );

    // --- Decide image count ---
    let want_images = if present_mode == vk::PresentModeKHR::MAILBOX {
        (caps.min_image_count + 1).max(3)
    } else {
        caps.min_image_count + 1
    };
    let min_count = if caps.max_image_count == 0 {
        want_images
    } else {
        want_images.min(caps.max_image_count)
    };

    // --- Surface transform ---
    // Prefer IDENTITY if supported (common), otherwise use current to avoid extra blits.
    let pre_transform = if caps
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        caps.current_transform
    };

    // PIck supported alpha flag
    let composite_alpha = [
        vk::CompositeAlphaFlagsKHR::OPAQUE,
        vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
        vk::CompositeAlphaFlagsKHR::INHERIT,
    ]
    .iter()
    .copied()
    .find(|f| caps.supported_composite_alpha.contains(*f))
    .unwrap_or(vk::CompositeAlphaFlagsKHR::OPAQUE);

    // --- Swapchain create info ---
    // IMPORTANT: image_usage must match how you use the images; here we only render to them.
    // If you later add post-processing blits/reads, include TRANSFER_DST/SRC as needed.
    let swap_info = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        surface,
        min_image_count: min_count,
        image_format: surf_format.format,
        image_color_space: surf_format.color_space,
        image_extent: extent,
        image_array_layers: 1, // non-stereo
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE, // single graphics queue family
        pre_transform,
        composite_alpha,
        present_mode,
        clipped: vk::TRUE, // don't care about obscured pixels
        old_swapchain,     // enables seamless re-creation w/ resource reuse
        ..Default::default()
    };

    // --- Create swapchain + fetch images ---
    let new_swapchain = unsafe { swap_d.create_swapchain(&swap_info, None)? };
    let images = unsafe { swap_d.get_swapchain_images(new_swapchain)? };

    // --- Create image views (one per swapchain image) ---
    // View format MUST match swapchain image format for direct rendering.
    let mut views = Vec::new();
    for &img in &images {
        let view = make_color_view(device, img, surf_format.format)?;
        views.push(view);
    }

    // --- Return the bundle used by higher-level code (recording, present, etc.) ---
    Ok(SwapchainBundle {
        swapchain: new_swapchain,
        format: surf_format.format,
        extent,
        images,
        image_views: views,
        color_space: surf_format.color_space,
    })
}

impl VkRenderer {
    // STRICT ORDER (recreate):
    // 1) Wait all in-flight image fences + acquire fences (no work using old sc)
    // 2) device_wait_idle() to avoid destroying in-use views/pipelines
    // 3) Destroy per-image views + per-image sync tied to OLD swapchain
    // 4) Create NEW swapchain + images + views
    // 5) Recreate per-image sync objects
    // 6) Recreate pipeline ONLY if format changed
    // 7) Resize command buffers if image count changed
    // (No re-record step here: render() records each frame's command
    // buffer fresh for whichever image it just acquired.)
    // Any deviation can cause sporadic DEVICE_LOST or image-in-use errors.
    pub(crate) fn recreate_swapchain(&mut self, size: RenderSize) -> Result<()> {
        // Guard min size window
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        // 1) Wait for GPU to reach the last signaled timeline value (flush all prior work)
        if self.timeline_value > 0 {
            let wait_info = vk::SemaphoreWaitInfo {
                s_type: vk::StructureType::SEMAPHORE_WAIT_INFO,
                flags: vk::SemaphoreWaitFlags::empty(),
                semaphore_count: 1,
                p_semaphores: &self.timeline,
                p_values: &self.timeline_value,
                ..Default::default()
            };
            unsafe { self.device.wait_semaphores(&wait_info, u64::MAX).ok() };
        }

        // 2) device_wait_idle() to avoid destroying in-use views/pipelines
        unsafe { self.device.device_wait_idle().ok() };

        // 3) Destroy per-image views + per-image sync tied to OLD swapchain
        for &iv in &self.image_views {
            unsafe { self.device.destroy_image_view(iv, None) };
        }
        for f in &self.frames {
            unsafe { self.device.destroy_semaphore(f.render_finished, None) };
        }
        self.frames.clear();

        // 3b) Retire per-image UBOs + descriptor pool tied to OLD swapchain.
        // gpu-allocator persistently maps CpuToGpu allocations, so no
        // explicit unmap is needed. device_wait_idle() above already makes
        // these safe to destroy immediately, but route them through the
        // trash queue anyway for consistency with the rest of the renderer.
        for (buffer, alloc) in self.ubufs.drain(..).zip(self.umems.drain(..)) {
            self.trash.push(DeferredDrop {
                value: self.timeline_value,
                resource: GpuResource::Buffer { buffer, alloc },
            });
        }
        self.ubo_ptrs.clear();
        self.ubo_size = 0;

        if self.desc_pool != vk::DescriptorPool::null() {
            unsafe { self.device.destroy_descriptor_pool(self.desc_pool, None) };
            self.desc_pool = vk::DescriptorPool::null();
        }
        self.desc_sets.clear();

        // 3c) Retire per-image indirect draw buffers.
        for (buffer, alloc) in self
            .candidate_bufs
            .drain(..)
            .zip(self.candidate_allocs.drain(..))
        {
            self.trash.push(DeferredDrop {
                value: self.timeline_value,
                resource: GpuResource::Buffer { buffer, alloc },
            });
        }
        self.candidate_ptrs.clear();
        for (buffer, alloc) in self
            .indirect_bufs
            .drain(..)
            .zip(self.indirect_allocs.drain(..))
        {
            self.trash.push(DeferredDrop {
                value: self.timeline_value,
                resource: GpuResource::Buffer { buffer, alloc },
            });
        }
        for (buffer, alloc) in self
            .draw_count_bufs
            .drain(..)
            .zip(self.draw_count_allocs.drain(..))
        {
            self.trash.push(DeferredDrop {
                value: self.timeline_value,
                resource: GpuResource::Buffer { buffer, alloc },
            });
        }
        if self.indirect_desc_pool != vk::DescriptorPool::null() {
            unsafe {
                self.device
                    .destroy_descriptor_pool(self.indirect_desc_pool, None)
            };
            self.indirect_desc_pool = vk::DescriptorPool::null();
        }
        self.indirect_compute_desc_sets.clear();
        self.indirect_graphics_desc_sets.clear();

        // 4a) cfg for new swapchain (hdr/vsync/flavor/extent)
        let cfg = self.cfg.to_swapchain_config(size);

        // 4b) create NEW swapchain + images + views
        let bundle = create_swapchain_bundle(
            &self.device,
            &self.surface_loader,
            &self.swapchain_loader,
            self.phys,
            self.surface,
            self.swapchain,
            cfg,
        )?;
        unsafe {
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None)
        };
        let SwapchainBundle {
            swapchain,
            format,
            extent,
            images,
            image_views,
            color_space,
        } = bundle;

        // 4c) HDR metadata
        create_hdr_metadata_if_needed(
            &self.instance,
            &self.device,
            self.has_hdr_metadata_ext,
            color_space,
            swapchain,
        );

        // 4d) Swap in new data
        let old_format = self.format;
        self.swapchain = swapchain;
        self.format = format;
        self.extent = extent;
        self.images = images;
        self.image_views = image_views;

        // 4e) Recreate depth resources for the NEW extent (using same depth format)
        if self.depth_view != vk::ImageView::null() {
            self.trash.push(DeferredDrop {
                value: self.timeline_value,
                resource: GpuResource::ImageView(self.depth_view),
            });
        }
        if self.depth_image != vk::Image::null() {
            self.trash.push(DeferredDrop {
                value: self.timeline_value,
                resource: GpuResource::Image {
                    image: self.depth_image,
                    alloc: std::mem::take(&mut self.depth_alloc),
                },
            });
        } else {
            // No image to pair the allocation with (shouldn't happen in
            // practice; image/alloc/view are always set together), but
            // don't leak the allocation if it ever does.
            let old_alloc = std::mem::take(&mut self.depth_alloc);
            let _ = self
                .allocator
                .as_mut()
                .expect("allocator missing")
                .free(old_alloc);
        }
        let (dimg, dalloc, dview) = create_depth_resources(
            &self.device,
            self.allocator.as_mut().expect("allocator missing"),
            self.extent,
            self.depth_format,
        )?;
        self.depth_image = dimg;
        self.depth_alloc = dalloc;
        self.depth_view = dview;

        // 5) Recreate per-image UBOs + descriptor sets
        let (ubufs, umems, ubo_ptrs, ubo_size, desc_pool, desc_sets) =
            create_frame_uniforms_and_sets(
                &self.instance,
                &self.device,
                self.phys,
                self.allocator.as_mut().expect("allocator missing"),
                self.desc_set_layout_camera,
                self.images.len(),
            )?;
        self.ubufs = ubufs;
        self.umems = umems;
        self.ubo_ptrs = ubo_ptrs;
        self.ubo_size = ubo_size;
        self.desc_pool = desc_pool;
        self.desc_sets = desc_sets;

        // 5b) Recreate per-image indirect draw resources.
        let indirect = create_indirect_draw_resources(
            &self.device,
            self.allocator.as_mut().expect("allocator missing"),
            self.desc_set_layout_indirect_compute,
            self.desc_set_layout_indirect_graphics,
            self.images.len(),
        )?;
        self.candidate_bufs = indirect.candidate_bufs;
        self.candidate_allocs = indirect.candidate_allocs;
        self.candidate_ptrs = indirect.candidate_ptrs;
        self.indirect_bufs = indirect.indirect_bufs;
        self.indirect_allocs = indirect.indirect_allocs;
        self.draw_count_bufs = indirect.draw_count_bufs;
        self.draw_count_allocs = indirect.draw_count_allocs;
        self.indirect_desc_pool = indirect.desc_pool;
        self.indirect_compute_desc_sets = indirect.compute_desc_sets;
        self.indirect_graphics_desc_sets = indirect.graphics_desc_sets;

        // 5c) Recreate per-image sync
        let image_count = self.images.len();
        let sem_info = vk::SemaphoreCreateInfo::default();
        for _ in 0..image_count {
            let rf = unsafe { self.device.create_semaphore(&sem_info, None)? };
            self.frames.push(FrameSync {
                render_finished: rf,
            });
        }

        // 6) Recreate pipeline only if COLOR format changed
        if self.format != old_format {
            let (new_layout, new_pipeline) = create_pipeline(
                &self.device,
                self.pipeline_cache,
                &PipelineConfig {
                    color_format: self.format,
                    depth_format: self.depth_format,
                    set_layout_camera: self.desc_set_layout_camera,
                    set_layout_material: self.desc_set_layout_material,
                    set_layout_indirect_graphics: self.desc_set_layout_indirect_graphics,
                },
            )?;
            self.trash.push(DeferredDrop {
                value: self.timeline_value,
                resource: GpuResource::Pipeline(self.pipeline),
            });
            self.trash.push(DeferredDrop {
                value: self.timeline_value,
                resource: GpuResource::PipelineLayout(self.pipeline_layout),
            });
            self.pipeline_layout = new_layout;
            self.pipeline = new_pipeline;

            // The egui pipeline is built against a fixed color format too
            // (see build_renderer); left stale here, cmd_begin_rendering's
            // new-format attachment wouldn't match it and every egui draw
            // would hit VUID-vkCmdDrawIndexed-dynamicRenderingUnusedAttachments-08910.
            // NB: this doesn't update Options::srgb_framebuffer (baked in at
            // construction), so if a format change ever crosses the
            // sRGB-view/HDR10-PQ/UNORM boundary (see format_needs_srgb_egui)
            // rather than just changing bit layout, egui's colors would be
            // off (not a crash) until the renderer is fully reconstructed —
            // not a case this engine's flavor selection hits today.
            if let Some(egui_renderer) = self.egui_renderer.as_mut() {
                let _ = egui_renderer.set_dynamic_rendering(egui_ash_renderer::DynamicRendering {
                    color_attachment_format: self.format,
                    depth_attachment_format: Some(self.depth_format),
                    stencil_attachment_format: None, //added for egui 0.35 compat
                });
            }
        }

        // 7) Resize CBs if image count changed
        if self.cmd_bufs.len() != self.images.len() {
            unsafe {
                self.device
                    .free_command_buffers(self.cmd_pool, &self.cmd_bufs)
            };
            let alloc_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                command_pool: self.cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: self.images.len() as u32,
                ..Default::default()
            };
            self.cmd_bufs = unsafe { self.device.allocate_command_buffers(&alloc_info)? };
        }

        self.acq_index = 0;

        Ok(())
    }
}
