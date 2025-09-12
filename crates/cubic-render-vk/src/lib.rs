use anyhow::{anyhow, Context, Result};
#[cfg(debug_assertions)]
use ash::ext::debug_utils as ext_debug;
use ash::khr::{surface, swapchain};
use ash::util::read_spv;
use ash::{vk, Entry, Instance};
use cubic_render::{RenderSize, Renderer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::io::Cursor;
use tracing::info;

pub struct VkRenderer {
    instance: ash::Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,

    phys: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,

    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,

    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,

    cmd_pool: vk::CommandPool,
    cmd_bufs: Vec<vk::CommandBuffer>,
    frames: Vec<FrameSync>,

    clear: vk::ClearValue,
    vsync: bool,
    paused: bool,
    vsync_mode: VkVsyncMode,

    #[allow(dead_code)]
    path: RenderPath,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
    acq_slots: Vec<AcquireSlot>,
    acq_index: usize,
    hdr: bool,
    have_swapchain_colorspace_ext: bool,
}

struct FrameSync {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight: vk::Fence,
}

struct AcquireSlot {
    sem: vk::Semaphore,
    fence: vk::Fence,
}

#[derive(Clone, Copy, Debug)]
struct SwapchainConfig {
    hint: RenderSize,
    vsync: bool,
    vsync_mode: VkVsyncMode,
    want_hdr: bool,
    allow_extended_colorspace: bool,
}

struct SwapchainBundle {
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    color_space: vk::ColorSpaceKHR,
}

#[derive(Clone, Copy, Debug)]
pub enum VkVsyncMode {
    Fifo,
    Mailbox,
}

#[derive(Clone, Copy, Debug)]
enum RenderPath {
    Core13, // Vulkan 1.3 core dynamic rendering + sync2
    KhrExt, // Vulkan 1.2 + VK_KHR_dynamic_rendering + VK_KHR_synchronization2
    Legacy, // No dynamic rendering: would need render pass/framebuffer path
}

#[derive(Clone, Copy, Debug)]
pub enum HdrMode {
    Off,
    Auto,
}

#[cfg(debug_assertions)]
unsafe extern "system" fn debug_callback(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    if !data.is_null() {
        let msg = std::ffi::CStr::from_ptr((*data).p_message);
        eprintln!("[Vulkan] {:?}", msg);
    }
    vk::FALSE
}

impl Drop for VkRenderer {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        {
            let entry = Entry::linked();
            let dbg = ext_debug::Instance::new(&entry, &self.instance);

            unsafe {
                dbg.destroy_debug_utils_messenger(self.debug_messenger, None);
            }
        }
        unsafe {
            let d = &self.device;
            let acq_fences: Vec<_> = self.acq_slots.iter().map(|s| s.fence).collect();

            if !acq_fences.is_empty() {
                let _ = self.device.wait_for_fences(&acq_fences, true, u64::MAX);
            }

            d.device_wait_idle().ok();
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);

            for &iv in &self.image_views {
                d.destroy_image_view(iv, None);
            }

            if !self.cmd_bufs.is_empty() {
                d.free_command_buffers(self.cmd_pool, &self.cmd_bufs);
            }

            d.destroy_command_pool(self.cmd_pool, None);

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            for f in &self.frames {
                d.destroy_fence(f.in_flight, None);
                d.destroy_semaphore(f.render_finished, None);
                d.destroy_semaphore(f.image_available, None);
            }
            for s in &self.acq_slots {
                self.device.destroy_fence(s.fence, None);
                self.device.destroy_semaphore(s.sem, None);
            }
            d.destroy_device(None);

            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

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

fn cs_name(cs: ash::vk::ColorSpaceKHR) -> &'static str {
    match cs {
        ash::vk::ColorSpaceKHR::SRGB_NONLINEAR => "SRGB_NONLINEAR",
        ash::vk::ColorSpaceKHR::DISPLAY_P3_NONLINEAR_EXT => "DISPLAY_P3_NONLINEAR",
        ash::vk::ColorSpaceKHR::HDR10_ST2084_EXT => "HDR10_ST2084",
        ash::vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT => "EXTENDED_SRGB_LINEAR",
        _ => "OTHER",
    }
}

fn pm_name(m: ash::vk::PresentModeKHR) -> &'static str {
    match m {
        ash::vk::PresentModeKHR::FIFO => "FIFO",
        ash::vk::PresentModeKHR::MAILBOX => "MAILBOX",
        ash::vk::PresentModeKHR::IMMEDIATE => "IMMEDIATE",
        ash::vk::PresentModeKHR::FIFO_RELAXED => "FIFO_RELAXED",
        _ => "OTHER",
    }
}

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

unsafe fn create_instance(entry: &Entry, display_raw: RawDisplayHandle) -> Result<Instance> {
    let app = std::ffi::CString::new("CubicEngine").unwrap();

    let app_info = vk::ApplicationInfo {
        s_type: vk::StructureType::APPLICATION_INFO,
        p_application_name: app.as_ptr(),
        application_version: 0,
        p_engine_name: app.as_ptr(),
        engine_version: 0,
        api_version: vk::API_VERSION_1_3,
        ..Default::default()
    };

    let ext_slice = ash_window::enumerate_required_extensions(display_raw)
        .context("enumerate_required_extensions")?;

    let inst_exts = entry
        .enumerate_instance_extension_properties(None)
        .context("enumerate_instance_extension_properties(instance)")?;
    let has_swapchain_cs = inst_exts.iter().any(|e| unsafe {
        std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::swapchain_colorspace::NAME
    });

    // Build the final list we’ll enable
    #[cfg(debug_assertions)]
    let ext_vec = {
        let mut v = ext_slice.to_vec();
        if has_swapchain_cs {
            v.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
        }
        v.push(ash::ext::debug_utils::NAME.as_ptr());
        v
    };
    #[cfg(not(debug_assertions))]
    let ext_vec = {
        let mut v = ext_slice.to_vec();
        if has_swapchain_cs {
            v.push(ash::ext::swapchain_colorspace::NAME.as_ptr());
        }
        v
    };

    #[cfg(debug_assertions)]
    let layers = [std::ffi::CString::new("VK_LAYER_KHRONOS_validation").unwrap()];

    let (enabled_layer_count, pp_enabled_layer_names) = {
        #[cfg(debug_assertions)]
        {
            (layers.len() as u32, layers.as_ptr() as *const *const i8)
        }
        #[cfg(not(debug_assertions))]
        {
            (0u32, std::ptr::null())
        }
    };

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_application_info: &app_info,
        enabled_extension_count: ext_vec.len() as u32,
        pp_enabled_extension_names: ext_vec.as_ptr(),
        enabled_layer_count,
        pp_enabled_layer_names,
        ..Default::default()
    };

    Ok(entry.create_instance(&create_info, None)?)
}

unsafe fn pick_device_and_queue(
    instance: &Instance,
    surf_i: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    for phys in instance.enumerate_physical_devices()? {
        let qprops = instance.get_physical_device_queue_family_properties(phys);

        for (i, q) in qprops.iter().enumerate() {
            if q.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && surf_i
                    .get_physical_device_surface_support(phys, i as u32, surface)
                    .unwrap_or(false)
            {
                return Ok((phys, i as u32));
            }
        }
    }
    Err(anyhow!("no suitable physical device/queue family"))
}

fn pick_surface_format(
    formats: &[vk::SurfaceFormatKHR],
    want_hdr: bool,
    allow_extended: bool,
) -> (vk::SurfaceFormatKHR, &'static str) {
    if want_hdr && allow_extended {
        // Prefer true HDR10 (PQ) on 10/16 bit
        if let Some(f) = formats.iter().copied().find(|f| {
            f.color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT
                && (f.format == vk::Format::R16G16B16A16_SFLOAT
                    || f.format == vk::Format::A2B10G10R10_UNORM_PACK32
                    || f.format == vk::Format::A2R10G10B10_UNORM_PACK32)
        }) {
            return (f, "hdr10_pq");
        }
        // Next best: scRGB linear FP16 (wide gamut, HDR-ish path on many desktops)
        if let Some(f) = formats.iter().copied().find(|f| {
            (f.color_space == vk::ColorSpaceKHR::EXTENDED_SRGB_LINEAR_EXT
             /* some WSI also expose NONLINEAR variant */ ||
             f.color_space == vk::ColorSpaceKHR::EXTENDED_SRGB_NONLINEAR_EXT)
                && f.format == vk::Format::R16G16B16A16_SFLOAT
        }) {
            return (f, "scrgb_fp16");
        }
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

unsafe fn create_swapchain_bundle(
    device: &ash::Device,
    surf_i: &surface::Instance,
    swap_d: &swapchain::Device,
    phys: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    cfg: SwapchainConfig,
) -> Result<SwapchainBundle> {
    let caps = surf_i.get_physical_device_surface_capabilities(phys, surface)?;
    let formats = surf_i.get_physical_device_surface_formats(phys, surface)?;
    //dump_surface_formats("enumerate", &formats);
    tracing::info!(
        "hdr_request={} allow_extended_colorspace={} have_swapchain_colorspace_ext={}",
        cfg.want_hdr,
        cfg.allow_extended_colorspace,
        cfg.allow_extended_colorspace
    );
    let modes = surf_i.get_physical_device_surface_present_modes(phys, surface)?;

    let (surf_format, pick_reason) =
        pick_surface_format(&formats, cfg.want_hdr, cfg.allow_extended_colorspace);

    let present_mode = choose_present_mode(&modes, cfg.vsync, cfg.vsync_mode);
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

        if caps.max_image_count == 0 { caps.min_image_count + 1
        } else {
          (caps.min_image_count + 1).min(caps.max_image_count)
        }
    );

    let min_count = if caps.max_image_count == 0 {
        caps.min_image_count + 1
    } else {
        (caps.min_image_count + 1).min(caps.max_image_count)
    };
    let pre_transform = if caps
        .supported_transforms
        .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
    {
        vk::SurfaceTransformFlagsKHR::IDENTITY
    } else {
        caps.current_transform
    };
    let swap_info = vk::SwapchainCreateInfoKHR {
        s_type: vk::StructureType::SWAPCHAIN_CREATE_INFO_KHR,
        surface,
        min_image_count: min_count,
        image_format: surf_format.format,
        image_color_space: surf_format.color_space,
        image_extent: extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        image_sharing_mode: vk::SharingMode::EXCLUSIVE,
        pre_transform,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        ..Default::default()
    };

    let swapchain = swap_d.create_swapchain(&swap_info, None)?;
    let images = swap_d.get_swapchain_images(swapchain)?;
    let mut views = Vec::with_capacity(images.len());

    for &img in &images {
        let sub = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };
        let iv_info = vk::ImageViewCreateInfo {
            s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
            image: img,
            view_type: vk::ImageViewType::TYPE_2D,
            format: surf_format.format,
            subresource_range: sub,
            ..Default::default()
        };

        views.push(device.create_image_view(&iv_info, None)?);
    }

    Ok(SwapchainBundle {
        swapchain,
        format: surf_format.format,
        extent,
        images,
        image_views: views,
        color_space: surf_format.color_space,
    })
}

unsafe fn create_pipeline(
    device: &ash::Device,
    color_format: vk::Format,
    _extent: vk::Extent2D,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vs_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/tri.vert.spv"));
    let fs_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/tri.frag.spv"));
    let vs_code = read_spv(&mut Cursor::new(&vs_bytes[..]))?;
    let fs_code = read_spv(&mut Cursor::new(&fs_bytes[..]))?;

    let vs_ci = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_code: vs_code.as_ptr(),
        code_size: vs_code.len() * 4,
        ..Default::default()
    };
    let fs_ci = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_code: fs_code.as_ptr(),
        code_size: fs_code.len() * 4,
        ..Default::default()
    };

    let vs = device.create_shader_module(&vs_ci, None)?;
    let fs = device.create_shader_module(&fs_ci, None)?;
    let entry = std::ffi::CString::new("main").unwrap();

    let stages = [
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::VERTEX,
            module: vs,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            s_type: vk::StructureType::PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: fs,
            p_name: entry.as_ptr(),
            ..Default::default()
        },
    ];

    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        ..Default::default()
    };
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };

    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

    let dynamic_state = vk::PipelineDynamicStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        dynamic_state_count: dyn_states.len() as u32,
        p_dynamic_states: dyn_states.as_ptr(),
        ..Default::default()
    };
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        viewport_count: 1,
        p_viewports: std::ptr::null(), // dynamic
        scissor_count: 1,
        p_scissors: std::ptr::null(), // dynamic
        ..Default::default()
    };
    let raster = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        ..Default::default()
    };
    let multisample = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    let color_blend_att = vk::PipelineColorBlendAttachmentState {
        color_write_mask: vk::ColorComponentFlags::R
            | vk::ColorComponentFlags::G
            | vk::ColorComponentFlags::B
            | vk::ColorComponentFlags::A,
        blend_enable: vk::FALSE,
        ..Default::default()
    };
    let color_blend = vk::PipelineColorBlendStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        attachment_count: 1,
        p_attachments: &color_blend_att,
        ..Default::default()
    };
    let layout_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        ..Default::default()
    };

    let layout = device.create_pipeline_layout(&layout_info, None)?;

    let rendering = vk::PipelineRenderingCreateInfo {
        s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
        color_attachment_count: 1,
        p_color_attachment_formats: &color_format,
        ..Default::default()
    };
    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        p_next: (&rendering as *const _) as *const _,
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vertex_input,
        p_input_assembly_state: &input_assembly,
        p_viewport_state: &viewport_state,
        p_rasterization_state: &raster,
        p_multisample_state: &multisample,
        p_color_blend_state: &color_blend,
        p_dynamic_state: &dynamic_state,
        layout,
        ..Default::default()
    };

    let pipelines = match device.create_graphics_pipelines(
        vk::PipelineCache::null(),
        std::slice::from_ref(&pipeline_info),
        None,
    ) {
        Ok(p) => p,
        Err((_, err)) => {
            return Err(anyhow!("create_graphics_pipelines failed: {:?}", err));
        }
    };

    device.destroy_shader_module(vs, None);
    device.destroy_shader_module(fs, None);

    Ok((layout, pipelines[0]))
}

unsafe fn build_renderer(
    window: &dyn HasWindowHandle,
    display: &dyn HasDisplayHandle,
    size: RenderSize,
) -> Result<VkRenderer> {
    let entry = Entry::linked();

    let dh: RawDisplayHandle = display
        .display_handle()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .as_raw();
    let wh: RawWindowHandle = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .as_raw();

    let instance = create_instance(&entry, dh)?;

    #[cfg(debug_assertions)]
    let debug_messenger = {
        let debug_loader = ext_debug::Instance::new(&entry, &instance);
        let ci = vk::DebugUtilsMessengerCreateInfoEXT {
            s_type: vk::StructureType::DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            pfn_user_callback: Some(debug_callback),
            ..Default::default()
        };

        unsafe {
            debug_loader
                .create_debug_utils_messenger(&ci, None)
                .unwrap()
        }
    };

    let surface =
        ash_window::create_surface(&entry, &instance, dh, wh, None).context("create_surface")?;

    let surface_loader = surface::Instance::new(&entry, &instance);
    let (phys, queue_family) = pick_device_and_queue(&instance, &surface_loader, surface)?;
    let priorities = [1.0_f32];

    let qinfo = vk::DeviceQueueCreateInfo {
        s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
        queue_family_index: queue_family,
        queue_count: 1,
        p_queue_priorities: priorities.as_ptr(),
        ..Default::default()
    };

    let dev_props = instance.get_physical_device_properties(phys);
    let dev_api = dev_props.api_version;
    let _dev_major = vk::api_version_major(dev_api);
    let _dev_minor = vk::api_version_minor(dev_api);

    let ext_props = instance
        .enumerate_device_extension_properties(phys)
        .context("enumerate_device_extension_properties")?;
    let has = |name: &std::ffi::CStr| -> bool {
        ext_props
            .iter()
            .any(|e| unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == name })
    };

    let mut device_exts: Vec<*const i8> = vec![swapchain::NAME.as_ptr()];

    let has_sync2_khr = has(ash::khr::synchronization2::NAME);
    let has_dynren_khr = has(ash::khr::dynamic_rendering::NAME);
    let has_hdr_meta = has(ash::ext::hdr_metadata::NAME);

    let inst_exts = entry
        .enumerate_instance_extension_properties(None)
        .unwrap_or_default();
    let have_swapchain_colorspace_ext = inst_exts.iter().any(|e| unsafe {
        std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::swapchain_colorspace::NAME
    });

    let mut feats12 = vk::PhysicalDeviceVulkan12Features {
        s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
        ..Default::default()
    };
    let mut feats13 = vk::PhysicalDeviceVulkan13Features {
        s_type: vk::StructureType::PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        ..Default::default()
    };
    let mut feats_sync2_khr = vk::PhysicalDeviceSynchronization2FeaturesKHR {
        s_type: vk::StructureType::PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES_KHR,
        ..Default::default()
    };
    let mut feats_dr_khr = vk::PhysicalDeviceDynamicRenderingFeaturesKHR {
        s_type: vk::StructureType::PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES_KHR,
        ..Default::default()
    };

    // Optional HDR metadata if present (device extension only if you use set_hdr_metadataEXT later)
    if has_hdr_meta {
        device_exts.push(ash::ext::hdr_metadata::NAME.as_ptr());
    }

    // ---- Decide path and build pNext chain (point into the long-lived structs above) ----
    let mut feats2 = vk::PhysicalDeviceFeatures2 {
        s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
        ..Default::default()
    };

    let force_khr = std::env::var("CUBIC_FORCE_KHR").ok().as_deref() == Some("1");

    // IMPORTANT: don't mix 1.3 core features with KHR feature structs.
    let (path, pnext): (RenderPath, *const std::ffi::c_void) = if !force_khr {
        let dev_api = instance.get_physical_device_properties(phys).api_version;
        let maj = vk::api_version_major(dev_api);
        let min = vk::api_version_minor(dev_api);

        if maj > 1 || (maj == 1 && min >= 3) {
            // ---- Core 1.3: ONLY 12 -> 13 ----
            feats13.synchronization2 = vk::TRUE;
            feats13.dynamic_rendering = vk::TRUE;

            feats12.p_next = (&mut feats13 as *mut _) as *mut _;
            feats2.p_next = (&mut feats12 as *mut _) as *mut _;

            (RenderPath::Core13, (&mut feats2 as *mut _) as *const _)
        } else if has_sync2_khr && has_dynren_khr {
            // ---- Vulkan 1.2 + KHR: ONLY 12 -> sync2KHR -> dynRenderingKHR ----
            device_exts.push(ash::khr::synchronization2::NAME.as_ptr());
            device_exts.push(ash::khr::dynamic_rendering::NAME.as_ptr());

            feats_sync2_khr.synchronization2 = vk::TRUE;
            feats_dr_khr.dynamic_rendering = vk::TRUE;

            feats_sync2_khr.p_next = (&mut feats_dr_khr as *mut _) as *mut _;
            feats12.p_next = (&mut feats_sync2_khr as *mut _) as *mut _;
            feats2.p_next = (&mut feats12 as *mut _) as *mut _;

            (RenderPath::KhrExt, (&mut feats2 as *mut _) as *const _)
        } else {
            (RenderPath::Legacy, std::ptr::null())
        }
    } else {
        // ---- Forced 1.2 + KHR on 1.3 hardware for testing ----
        device_exts.push(ash::khr::synchronization2::NAME.as_ptr());
        device_exts.push(ash::khr::dynamic_rendering::NAME.as_ptr());

        feats_sync2_khr.synchronization2 = vk::TRUE;
        feats_dr_khr.dynamic_rendering = vk::TRUE;

        feats_sync2_khr.p_next = (&mut feats_dr_khr as *mut _) as *mut _;
        feats12.p_next = (&mut feats_sync2_khr as *mut _) as *mut _;
        feats2.p_next = (&mut feats12 as *mut _) as *mut _;

        (RenderPath::KhrExt, (&mut feats2 as *mut _) as *const _)
    };

    // ---- DeviceCreateInfo (use selected pNext) ----
    let dinfo = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
        p_next: pnext,
        queue_create_info_count: 1,
        p_queue_create_infos: &qinfo,
        enabled_extension_count: device_exts.len() as u32,
        pp_enabled_extension_names: device_exts.as_ptr(),
        ..Default::default()
    };

    let device = instance
        .create_device(phys, &dinfo, None)
        .context("create_device")?;

    let queue = device.get_device_queue(queue_family, 0);

    if let RenderPath::Legacy = path {
        return Err(anyhow!(
            "No dynamic rendering available on this device; \
         legacy render pass path is required (not compiled here)."
        ));
    }

    let swapchain_loader = swapchain::Device::new(&instance, &device);
    let initial_vsync = true;
    let initial_mode = VkVsyncMode::Mailbox;
    let initial_hdr = std::env::var("CUBIC_HDR").ok().as_deref() == Some("1");

    let cfg = SwapchainConfig {
        hint: size,
        vsync: initial_vsync,
        vsync_mode: initial_mode,
        want_hdr: initial_hdr,
        allow_extended_colorspace: have_swapchain_colorspace_ext,
    };

    let bundle = create_swapchain_bundle(
        &device,
        &surface_loader,
        &swapchain_loader,
        phys,
        surface,
        cfg,
    )?;

    let SwapchainBundle {
        swapchain,
        format,
        extent,
        images,
        image_views,
        color_space,
    } = bundle;

    if has_hdr_meta && color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT {
        let hdr = ash::ext::hdr_metadata::Device::new(&instance, &device);
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
        unsafe {
            hdr.set_hdr_metadata(&[swapchain], std::slice::from_ref(&metadata));
        }
    }

    let pool_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        queue_family_index: queue_family,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER, // <-- required for vkResetCommandBuffer
        ..Default::default()
    };

    let cmd_pool = device.create_command_pool(&pool_info, None)?;

    let alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool: cmd_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: image_views.len() as u32, // was framebuffers.len()
        ..Default::default()
    };

    let cmd_bufs = device.allocate_command_buffers(&alloc_info)?;
    let (pipeline_layout, pipeline) = create_pipeline(&device, format, extent)?;
    let image_count = image_views.len();
    let mut frames = Vec::with_capacity(image_count);
    let mut acq_slots = Vec::with_capacity(2);
    let acq_sem_info = vk::SemaphoreCreateInfo::default();

    let acq_fence_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };

    for _ in 0..2 {
        let sem = device.create_semaphore(&acq_sem_info, None)?;
        let fence = device.create_fence(&acq_fence_info, None)?;
        acq_slots.push(AcquireSlot { sem, fence });
    }

    let sem_info = vk::SemaphoreCreateInfo::default();

    let fence_info = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };

    for _ in 0..image_count {
        let ia = device.create_semaphore(&sem_info, None)?;
        let rf = device.create_semaphore(&sem_info, None)?;
        let inflight = device.create_fence(&fence_info, None)?;

        frames.push(FrameSync {
            image_available: ia,
            render_finished: rf,
            in_flight: inflight,
        });
    }

    let mut r = VkRenderer {
        instance,
        surface_loader,
        surface,

        phys,
        device,
        queue,

        swapchain_loader,
        swapchain,
        format,
        extent,

        images,
        image_views,

        pipeline,
        pipeline_layout,
        cmd_pool,
        cmd_bufs,

        frames,
        clear: vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.02, 0.02, 0.04, 1.0],
            },
        },

        vsync: initial_vsync,
        paused: false,
        vsync_mode: VkVsyncMode::Mailbox,
        path,

        #[cfg(debug_assertions)]
        debug_messenger,
        acq_slots,
        acq_index: 0,
        hdr: initial_hdr,
        have_swapchain_colorspace_ext,
    };

    r.record_commands()?;

    Ok(r)
}

impl VkRenderer {
    pub fn set_vsync_mode(&mut self, mode: VkVsyncMode) {
        if self.vsync_mode as u8 == mode as u8 {
            return;
        }

        self.vsync_mode = mode;

        let want = RenderSize {
            width: self.extent.width,
            height: self.extent.height,
        };

        let _ = unsafe { self.recreate_swapchain(want) };
    }
    pub fn set_hdr_enabled(&mut self, on: bool) {
        if self.hdr == on {
            return;
        }
        self.hdr = on;
        let want = RenderSize {
            width: self.extent.width,
            height: self.extent.height,
        };
        let _ = unsafe { self.recreate_swapchain(want) };
    }

    unsafe fn record_commands(&mut self) -> Result<()> {
        for (i, &cmd) in self.cmd_bufs.iter().enumerate() {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            let begin = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                ..Default::default()
            };

            self.device.begin_command_buffer(cmd, &begin)?;

            let image = self.images[i];

            let subrange = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            let pre_barrier = vk::ImageMemoryBarrier2 {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
                src_access_mask: vk::AccessFlags2::empty(),
                dst_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,

                dst_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags2::COLOR_ATTACHMENT_READ,

                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                image,
                subresource_range: subrange,
                ..Default::default()
            };
            let dep_pre = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                image_memory_barrier_count: 1,
                p_image_memory_barriers: &pre_barrier,
                ..Default::default()
            };

            self.device.cmd_pipeline_barrier2(cmd, &dep_pre);

            let color_att = vk::RenderingAttachmentInfo {
                s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
                image_view: self.image_views[i],
                image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: self.clear,
                ..Default::default()
            };
            let render_area = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            };
            let rendering_info = vk::RenderingInfo {
                s_type: vk::StructureType::RENDERING_INFO,
                render_area,
                layer_count: 1,
                color_attachment_count: 1,
                p_color_attachments: &color_att,
                ..Default::default()
            };

            self.device.cmd_begin_rendering(cmd, &rendering_info);

            if self.pipeline == vk::Pipeline::null() {
                return Err(anyhow!("pipeline is VK_NULL_HANDLE at record time"));
            }

            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            let vp = vk::Viewport {
                x: 0.0,
                y: self.extent.height as f32,
                width: self.extent.width as f32,
                height: -(self.extent.height as f32), // flip Y to match GL-style
                min_depth: 0.0,
                max_depth: 1.0,
            };

            self.device
                .cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));

            let sc = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            };

            self.device
                .cmd_set_scissor(cmd, 0, std::slice::from_ref(&sc));

            self.device.cmd_draw(cmd, 3, 1, 0, 0);
            self.device.cmd_end_rendering(cmd);

            let post_barrier = vk::ImageMemoryBarrier2 {
                s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
                src_stage_mask: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                src_access_mask: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                dst_stage_mask: vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
                dst_access_mask: vk::AccessFlags2::empty(),
                old_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                image,
                subresource_range: subrange,
                ..Default::default()
            };
            let dep_post = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                image_memory_barrier_count: 1,
                p_image_memory_barriers: &post_barrier,
                ..Default::default()
            };

            self.device.cmd_pipeline_barrier2(cmd, &dep_post);
            self.device.end_command_buffer(cmd)?;
        }

        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, size: RenderSize) -> Result<()> {
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        let image_fences: Vec<_> = self.frames.iter().map(|f| f.in_flight).collect();

        if !image_fences.is_empty() {
            let _ = self.device.wait_for_fences(&image_fences, true, u64::MAX);
        }

        let acq_fences: Vec<_> = self.acq_slots.iter().map(|s| s.fence).collect();

        if !acq_fences.is_empty() {
            let _ = self.device.wait_for_fences(&acq_fences, true, u64::MAX);
        }

        self.device.device_wait_idle().ok();

        for &iv in &self.image_views {
            self.device.destroy_image_view(iv, None);
        }

        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);

        for f in &self.frames {
            self.device.destroy_fence(f.in_flight, None);
            self.device.destroy_semaphore(f.render_finished, None);
            self.device.destroy_semaphore(f.image_available, None);
        }

        self.frames.clear();

        let cfg = SwapchainConfig {
            hint: size,
            vsync: self.vsync,
            vsync_mode: self.vsync_mode,
            want_hdr: self.hdr,
            allow_extended_colorspace: self.have_swapchain_colorspace_ext,
        };

        let bundle = create_swapchain_bundle(
            &self.device,
            &self.surface_loader,
            &self.swapchain_loader,
            self.phys,
            self.surface,
            cfg,
        )?;

        let SwapchainBundle {
            swapchain,
            format,
            extent,
            images,
            image_views,
            color_space,
        } = bundle;

        let has_hdr_meta = {
            let ext_props = self
                .instance
                .enumerate_device_extension_properties(self.phys)
                .unwrap_or_default();
            ext_props.iter().any(|e| unsafe {
                std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == ash::ext::hdr_metadata::NAME
            })
        };

        if has_hdr_meta && color_space == vk::ColorSpaceKHR::HDR10_ST2084_EXT {
            let hdr = ash::ext::hdr_metadata::Device::new(&self.instance, &self.device);
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
            unsafe {
                hdr.set_hdr_metadata(&[swapchain], std::slice::from_ref(&metadata));
            }
        }

        let old_format = self.format;

        self.swapchain = swapchain;
        self.format = format;
        self.extent = extent;
        self.images = images;
        self.image_views = image_views;

        let image_count = self.images.len();
        let sem_info = vk::SemaphoreCreateInfo::default();

        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };

        for _ in 0..image_count {
            let ia = self.device.create_semaphore(&sem_info, None)?;
            let rf = self.device.create_semaphore(&sem_info, None)?;
            let inflight = self.device.create_fence(&fence_info, None)?;

            self.frames.push(FrameSync {
                image_available: ia,
                render_finished: rf,
                in_flight: inflight,
            });
        }

        if self.format != old_format {
            let (new_layout, new_pipeline) =
                create_pipeline(&self.device, self.format, self.extent)?;

            self.device.destroy_pipeline(self.pipeline, None);

            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);

            self.pipeline_layout = new_layout;
            self.pipeline = new_pipeline;
        }

        if self.cmd_bufs.len() != self.images.len() {
            self.device
                .free_command_buffers(self.cmd_pool, &self.cmd_bufs);

            let alloc_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                command_pool: self.cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: self.images.len() as u32,
                ..Default::default()
            };

            self.cmd_bufs = self.device.allocate_command_buffers(&alloc_info)?;
        }

        self.record_commands()?;

        Ok(())
    }
}

impl Renderer for VkRenderer {
    fn new(
        window: &dyn HasWindowHandle,
        display: &dyn HasDisplayHandle,
        size: RenderSize,
    ) -> Result<Self> {
        unsafe { build_renderer(window, display, size) }
    }

    fn set_vsync(&mut self, on: bool) {
        if self.vsync == on {
            return;
        }

        self.vsync = on;

        let want = RenderSize {
            width: self.extent.width,
            height: self.extent.height,
        };

        let _ = unsafe { self.recreate_swapchain(want) };
    }

    fn resize(&mut self, size: RenderSize) -> Result<()> {
        if size.width == 0 || size.height == 0 {
            if !self.paused {
                info!("vk: resize to 0x0 → paused=true");
            }

            self.paused = true;
            return Ok(());
        }

        if self.paused {
            info!(
                "vk: resize to {}x{} → paused=false",
                size.width, size.height
            );
        }

        self.paused = false;

        unsafe { self.recreate_swapchain(size) }
    }

    fn set_clear_color(&mut self, rgba: [f32; 4]) {
        self.clear = vk::ClearValue {
            color: vk::ClearColorValue { float32: rgba },
        };

        unsafe {
            let _ = self.record_commands();
        }
    }

    fn render(&mut self) -> Result<()> {
        if self.paused {
            return Ok(());
        }

        unsafe {
            match self
                .surface_loader
                .get_physical_device_surface_capabilities(self.phys, self.surface)
            {
                Ok(caps) => {
                    if caps.current_extent.width == 0 || caps.current_extent.height == 0 {
                        if !self.paused {
                            self.paused = true;
                            info!("vk: current_extent is 0x0 → paused=true");
                        }
                        return Ok(());
                    }
                }
                Err(e) => {
                    if !self.paused {
                        self.paused = true;
                        info!("vk: surface caps error {:?} → paused=true", e);
                    }
                    return Ok(());
                }
            }

            let s = &self.acq_slots[self.acq_index];

            self.device
                .wait_for_fences(&[s.fence], true, u64::MAX)
                .context("wait_for_fences(acquire slot)")?;
            self.device.reset_fences(&[s.fence])?;

            let (image_index, _) = match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                s.sem,
                vk::Fence::null(),
            ) {
                Ok(pair) => pair,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    if !self.paused {
                        self.paused = true;
                        info!("vk: acquire → OUT_OF_DATE/SUBOPTIMAL → paused=true");
                    }
                    return Ok(());
                }
                Err(e) => return Err(anyhow::anyhow!("acquire_next_image: {e:?}")),
            };

            let img = image_index as usize;
            let f_img = &self.frames[img];
            let cmd = self.cmd_bufs[img];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

            let submit = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                wait_semaphore_count: 1,
                p_wait_semaphores: &s.sem,
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                signal_semaphore_count: 1,
                p_signal_semaphores: &f_img.render_finished,
                ..Default::default()
            };

            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit), s.fence)
                .context("queue_submit")?;

            let present = vk::PresentInfoKHR {
                s_type: vk::StructureType::PRESENT_INFO_KHR,
                wait_semaphore_count: 1,
                p_wait_semaphores: &f_img.render_finished,
                swapchain_count: 1,
                p_swapchains: &self.swapchain,
                p_image_indices: &image_index,
                ..Default::default()
            };

            match self.swapchain_loader.queue_present(self.queue, &present) {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    let want = RenderSize {
                        width: self.extent.width,
                        height: self.extent.height,
                    };

                    let _ = self.recreate_swapchain(want);

                    if !self.paused {
                        self.paused = true;
                        info!("vk: present → OUT_OF_DATE/SUBOPTIMAL → paused=true");
                    }
                }
                Err(e) => return Err(anyhow::anyhow!("queue_present: {e:?}")),
            }

            self.acq_index = (self.acq_index + 1) % self.acq_slots.len();

            Ok(())
        }
    }
}
