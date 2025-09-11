use anyhow::{anyhow, Context, Result};
use ash::khr::{surface, swapchain};
use ash::util::read_spv;
use ash::{vk, Entry, Instance};
use cubic_render::{RenderSize, Renderer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::io::Cursor;
use tracing::info;

pub struct VkRenderer {
    //entry: ash::Entry,
    instance: ash::Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,

    phys: vk::PhysicalDevice,
    device: ash::Device,
    //queue_family: u32,
    queue: vk::Queue,

    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,

    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,

    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    cmd_pool: vk::CommandPool,
    cmd_bufs: Vec<vk::CommandBuffer>,

    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    clear: vk::ClearValue,
    vsync: bool,
}

struct SwapchainBundle {
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
}

impl Drop for VkRenderer {
    fn drop(&mut self) {
        unsafe {
            let d = &self.device;

            d.device_wait_idle().ok();
            d.destroy_semaphore(self.render_finished, None);
            d.destroy_semaphore(self.image_available, None);
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);

            for &fb in &self.framebuffers {
                d.destroy_framebuffer(fb, None);
            }
            for &iv in &self.image_views {
                d.destroy_image_view(iv, None);
            }

            d.destroy_render_pass(self.render_pass, None);

            if !self.cmd_bufs.is_empty() {
                d.free_command_buffers(self.cmd_pool, &self.cmd_bufs);
            }

            d.destroy_command_pool(self.cmd_pool, None);

            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

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

fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    // Prefer true sRGB if available
    formats
        .iter()
        .copied()
        .find(|f| f.format == vk::Format::B8G8R8A8_SRGB)
        .or_else(|| {
            formats
                .iter()
                .copied()
                .find(|f| f.format == vk::Format::R8G8B8A8_SRGB)
        })
        .or_else(|| {
            formats.iter().copied().find(|f| {
                f.format == vk::Format::B8G8R8A8_UNORM
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
        })
        .unwrap_or_else(|| formats[0])
}
fn choose_present_mode(modes: &[vk::PresentModeKHR], vsync: bool) -> vk::PresentModeKHR {
    // Priority order: with vsync prefer MAILBOX>FIFO; without vsync prefer IMMEDIATE>MAILBOX>FIFO
    let order: &[vk::PresentModeKHR] = if vsync {
        &[vk::PresentModeKHR::MAILBOX, vk::PresentModeKHR::FIFO]
    } else {
        &[
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::MAILBOX,
            vk::PresentModeKHR::FIFO,
        ]
    };

    order
        .iter()
        .copied()
        .find(|m| modes.contains(m))
        .unwrap_or(vk::PresentModeKHR::FIFO)
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
        api_version: vk::API_VERSION_1_0,
        ..Default::default()
    };

    let ext_slice = ash_window::enumerate_required_extensions(display_raw)
        .context("enumerate_required_extensions")?;

    let ext_vec = ext_slice.to_vec();

    let create_info = vk::InstanceCreateInfo {
        s_type: vk::StructureType::INSTANCE_CREATE_INFO,
        p_application_info: &app_info,
        enabled_extension_count: ext_vec.len() as u32,
        pp_enabled_extension_names: ext_vec.as_ptr(),
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

unsafe fn create_swapchain_bundle(
    //instance: &Instance,
    device: &ash::Device,
    surf_i: &surface::Instance,
    swap_d: &swapchain::Device,
    phys: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    hint: RenderSize,
    vsync: bool,
) -> Result<SwapchainBundle> {
    let caps = surf_i.get_physical_device_surface_capabilities(phys, surface)?;
    let formats = surf_i.get_physical_device_surface_formats(phys, surface)?;
    let modes = surf_i.get_physical_device_surface_present_modes(phys, surface)?;
    let surf_format = choose_surface_format(&formats);
    let present_mode = choose_present_mode(&modes, vsync);
    let extent = extent_from_caps(&caps, hint);

    tracing::info!(
        "swapchain: fmt={}, cs={}, present={}, extent={}x{}",
        fmt_name(surf_format.format),
        cs_name(surf_format.color_space),
        pm_name(present_mode),
        extent.width,
        extent.height
    );

    info!(
    "swapchain choose → format: {} / {}, present_mode: {}, extent: {}x{}, images(min={} → picked={})",
    fmt_name(surf_format.format),
    cs_name(surf_format.color_space),
    pm_name(present_mode),
    extent.width, extent.height,
    caps.min_image_count,
    if caps.max_image_count == 0 { caps.min_image_count + 1 } else { (caps.min_image_count + 1).min(caps.max_image_count) }
);

    let min_count = if caps.max_image_count == 0 {
        caps.min_image_count + 1
    } else {
        (caps.min_image_count + 1).min(caps.max_image_count)
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
        pre_transform: caps.current_transform,
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

    // Render pass
    let color_att = vk::AttachmentDescription {
        format: surf_format.format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::UNDEFINED,
        final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
        ..Default::default()
    };
    let att_ref = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    let subpass = vk::SubpassDescription {
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        color_attachment_count: 1,
        p_color_attachments: &att_ref,
        ..Default::default()
    };
    let rp_info = vk::RenderPassCreateInfo {
        s_type: vk::StructureType::RENDER_PASS_CREATE_INFO,
        attachment_count: 1,
        p_attachments: &color_att,
        subpass_count: 1,
        p_subpasses: &subpass,
        ..Default::default()
    };

    let render_pass = device.create_render_pass(&rp_info, None)?;
    let mut framebuffers = Vec::with_capacity(views.len());

    for &view in &views {
        let fb_info = vk::FramebufferCreateInfo {
            s_type: vk::StructureType::FRAMEBUFFER_CREATE_INFO,
            render_pass,
            attachment_count: 1,
            p_attachments: &view,
            width: extent.width,
            height: extent.height,
            layers: 1,
            ..Default::default()
        };
        framebuffers.push(device.create_framebuffer(&fb_info, None)?);
    }

    Ok(SwapchainBundle {
        swapchain,
        format: surf_format.format,
        extent,
        images,
        image_views: views,
        render_pass,
        framebuffers,
    })
}

unsafe fn create_pipeline(
    device: &ash::Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
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
    let viewport = vk::Viewport {
        x: 0.0,
        y: extent.height as f32,
        width: extent.width as f32,
        height: -(extent.height as f32),
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent,
    };
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        viewport_count: 1,
        p_viewports: &viewport,
        scissor_count: 1,
        p_scissors: &scissor,
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

    let pipeline_info = vk::GraphicsPipelineCreateInfo {
        s_type: vk::StructureType::GRAPHICS_PIPELINE_CREATE_INFO,
        stage_count: stages.len() as u32,
        p_stages: stages.as_ptr(),
        p_vertex_input_state: &vertex_input,
        p_input_assembly_state: &input_assembly,
        p_viewport_state: &viewport_state,
        p_rasterization_state: &raster,
        p_multisample_state: &multisample,
        p_color_blend_state: &color_blend,
        layout,
        render_pass,
        subpass: 0,
        ..Default::default()
    };

    let pipelines = device
        .create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
        .map_err(|e| anyhow!("create_graphics_pipelines: {:?}", e.1))?;

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

    let device_exts = [swapchain::NAME.as_ptr()];

    let dinfo = vk::DeviceCreateInfo {
        s_type: vk::StructureType::DEVICE_CREATE_INFO,
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
    let swapchain_loader = swapchain::Device::new(&instance, &device);
    let initial_vsync = true;

    let bundle = create_swapchain_bundle(
        //&instance,
        &device,
        &surface_loader,
        &swapchain_loader,
        phys,
        surface,
        size,
        initial_vsync,
    )?;

    let SwapchainBundle {
        swapchain,
        format,
        extent,
        images,
        image_views,
        render_pass,
        framebuffers,
    } = bundle;

    //let (pipeline_layout, pipeline) = create_pipeline(&device, render_pass, extent)?;

    let pool_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        queue_family_index: queue_family,
        ..Default::default()
    };

    let cmd_pool = device.create_command_pool(&pool_info, None)?;

    let alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool: cmd_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: framebuffers.len() as u32,
        ..Default::default()
    };

    let cmd_bufs = device.allocate_command_buffers(&alloc_info)?;
    let sem_info = vk::SemaphoreCreateInfo::default();
    let image_available = device.create_semaphore(&sem_info, None)?;
    let render_finished = device.create_semaphore(&sem_info, None)?;
    let (pipeline_layout, pipeline) = create_pipeline(&device, render_pass, extent)?;

    let mut r = VkRenderer {
        //entry,
        instance,
        surface_loader,
        surface,

        phys,
        device,
        //queue_family,
        queue,

        swapchain_loader,
        swapchain,
        format,
        extent,

        images,
        image_views,
        render_pass,
        framebuffers,

        pipeline,
        pipeline_layout,
        cmd_pool,
        cmd_bufs,

        image_available,
        render_finished,

        clear: vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.02, 0.02, 0.04, 1.0],
            },
        },
        vsync: initial_vsync,
    };

    r.record_commands()?;

    Ok(r)
}

impl VkRenderer {
    unsafe fn record_commands(&mut self) -> Result<()> {
        for (i, &cmd) in self.cmd_bufs.iter().enumerate() {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            let begin = vk::CommandBufferBeginInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
                ..Default::default()
            };

            self.device.begin_command_buffer(cmd, &begin)?;

            let clears = [self.clear];

            let rp_begin = vk::RenderPassBeginInfo {
                s_type: vk::StructureType::RENDER_PASS_BEGIN_INFO,
                render_pass: self.render_pass,
                framebuffer: self.framebuffers[i],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                },
                clear_value_count: clears.len() as u32,
                p_clear_values: clears.as_ptr(),
                ..Default::default()
            };

            self.device
                .cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);

            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            self.device.cmd_draw(cmd, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(cmd);
            self.device.end_command_buffer(cmd)?;
        }

        Ok(())
    }

    unsafe fn recreate_swapchain(&mut self, size: RenderSize) -> Result<()> {
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        self.device.device_wait_idle().ok();

        for &fb in &self.framebuffers {
            self.device.destroy_framebuffer(fb, None);
        }
        for &iv in &self.image_views {
            self.device.destroy_image_view(iv, None);
        }

        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);

        self.device.destroy_pipeline(self.pipeline, None);

        let bundle = create_swapchain_bundle(
            //&self.instance,
            &self.device,
            &self.surface_loader,
            &self.swapchain_loader,
            self.phys,
            self.surface,
            size,
            self.vsync,
        )?;

        let SwapchainBundle {
            swapchain,
            format,
            extent,
            images,
            image_views,
            render_pass,
            framebuffers,
        } = bundle;

        self.device.destroy_render_pass(self.render_pass, None);
        self.swapchain = swapchain;
        self.format = format;
        self.extent = extent;
        self.images = images;
        self.image_views = image_views;
        self.render_pass = render_pass;
        self.framebuffers = framebuffers;

        let (new_layout, new_pipeline) =
            create_pipeline(&self.device, self.render_pass, self.extent)?;

        self.device
            .destroy_pipeline_layout(self.pipeline_layout, None);

        self.pipeline_layout = new_layout;
        self.pipeline = new_pipeline;

        if self.cmd_bufs.len() != self.framebuffers.len() {
            self.device
                .free_command_buffers(self.cmd_pool, &self.cmd_bufs);

            let alloc_info = vk::CommandBufferAllocateInfo {
                s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
                command_pool: self.cmd_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: self.framebuffers.len() as u32,
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
        if self.extent.width == 0 || self.extent.height == 0 {
            return Ok(()); // minimized
        }

        unsafe {
            // 1) Acquire image
            let acquire = self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available,
                vk::Fence::null(),
            );

            let (image_index, _) = match acquire {
                Ok(pair) => pair,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    let want = RenderSize {
                        width: self.extent.width,
                        height: self.extent.height,
                    };

                    let _ = self.recreate_swapchain(want);

                    return Ok(());
                }

                Err(e) => return Err(anyhow::anyhow!("acquire_next_image: {e:?}")),
            };

            let cmd = self.cmd_bufs[image_index as usize];

            // 2) Submit
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

            let submit = vk::SubmitInfo {
                s_type: vk::StructureType::SUBMIT_INFO,
                wait_semaphore_count: 1,
                p_wait_semaphores: &self.image_available,
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                signal_semaphore_count: 1,
                p_signal_semaphores: &self.render_finished,
                ..Default::default()
            };

            self.device
                .queue_submit(self.queue, std::slice::from_ref(&submit), vk::Fence::null())
                .context("queue_submit")?;

            // 3) Present
            let present = vk::PresentInfoKHR {
                s_type: vk::StructureType::PRESENT_INFO_KHR,
                wait_semaphore_count: 1,
                p_wait_semaphores: &self.render_finished,
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
                }

                Err(e) => return Err(anyhow::anyhow!("queue_present: {e:?}")),
            }

            Ok(())
        }
    }
}
