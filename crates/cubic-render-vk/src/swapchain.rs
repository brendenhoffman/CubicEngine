// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::Result;
use ash::khr::{surface, swapchain};
use ash::vk;
use cubic_render::RenderSize;

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
