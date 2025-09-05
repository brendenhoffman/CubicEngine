use anyhow::{anyhow, Context, Result};
use tracing::info;

use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use cubic_render::{RenderSize, Renderer};

use ash::{vk, Entry, Instance};
use ash::khr::{surface, swapchain}; // <-- 0.38 module paths
use ash_window;

pub struct VkRenderer {
  _entry: Entry,
  instance: Instance,
  surface_loader: surface::Instance,
  surface: vk::SurfaceKHR,

  phys: vk::PhysicalDevice,
  device: ash::Device,
  queue_family: u32,
  queue: vk::Queue,

  swapchain_loader: swapchain::Device,
  swapchain: vk::SwapchainKHR,
  format: vk::Format,
  extent: vk::Extent2D,
  images: Vec<vk::Image>,
  image_views: Vec<vk::ImageView>,

  render_pass: vk::RenderPass,
  framebuffers: Vec<vk::Framebuffer>,

  cmd_pool: vk::CommandPool,
  cmd_bufs: Vec<vk::CommandBuffer>,

  clear: vk::ClearValue,
}

impl Drop for VkRenderer {
  fn drop(&mut self) {
    unsafe {
      let d = &self.device;
      d.device_wait_idle().ok();

      for &fb in &self.framebuffers { d.destroy_framebuffer(fb, None); }
      for &iv in &self.image_views { d.destroy_image_view(iv, None); }
      d.destroy_render_pass(self.render_pass, None);

      self.swapchain_loader.destroy_swapchain(self.swapchain, None);

      d.destroy_command_pool(self.cmd_pool, None);
      d.destroy_device(None);

      self.surface_loader.destroy_surface(self.surface, None);
      self.instance.destroy_instance(None);
    }
  }
}

fn choose_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
  formats.iter().copied().find(|f|
    f.format == vk::Format::B8G8R8A8_UNORM &&
    f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
  ).unwrap_or_else(|| formats[0])
}

fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
  if modes.iter().any(|&m| m == vk::PresentModeKHR::MAILBOX) {
    vk::PresentModeKHR::MAILBOX
  } else {
    vk::PresentModeKHR::FIFO
  }
}

fn extent_from_caps(caps: &vk::SurfaceCapabilitiesKHR, want: RenderSize) -> vk::Extent2D {
  if caps.current_extent.width != u32::MAX {
    caps.current_extent
  } else {
    vk::Extent2D {
      width:  want.width.clamp(caps.min_image_extent.width,  caps.max_image_extent.width),
      height: want.height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
    }
  }
}

unsafe fn create_instance(entry: &Entry, display_raw: RawDisplayHandle) -> Result<Instance> {
  let app_name = std::ffi::CString::new("CubicEngine").unwrap();

  let app_info = vk::ApplicationInfo {
    s_type: vk::StructureType::APPLICATION_INFO,
    p_application_name: app_name.as_ptr(),
    application_version: 0,
    p_engine_name: app_name.as_ptr(),
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
  surface_loader: &surface::Instance,
  surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
  for phys in instance.enumerate_physical_devices()? {
    let qprops = instance.get_physical_device_queue_family_properties(phys);
    for (i, q) in qprops.iter().enumerate() {
      if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) &&
         surface_loader.get_physical_device_surface_support(phys, i as u32, surface).unwrap_or(false)
      {
        return Ok((phys, i as u32));
      }
    }
  }
  Err(anyhow!("no suitable physical device/queue family"))
}

unsafe fn create_swapchain_bundle(
  instance: &Instance,
  device: &ash::Device,
  surface_loader: &surface::Instance,
  swapchain_loader: &swapchain::Device,
  phys: vk::PhysicalDevice,
  surface: vk::SurfaceKHR,
  extent_hint: RenderSize,
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, Vec<vk::Image>, Vec<vk::ImageView>, vk::RenderPass, Vec<vk::Framebuffer>)> {

  let caps = surface_loader.get_physical_device_surface_capabilities(phys, surface)?;
  let formats = surface_loader.get_physical_device_surface_formats(phys, surface)?;
  let modes = surface_loader.get_physical_device_surface_present_modes(phys, surface)?;

  let surf_format = choose_surface_format(&formats);
  let present_mode = choose_present_mode(&modes);
  let extent = extent_from_caps(&caps, extent_hint);

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

  let swapchain = swapchain_loader.create_swapchain(&swap_info, None)?;
  let images = swapchain_loader.get_swapchain_images(swapchain)?;

  let mut views = Vec::with_capacity(images.len());
  for &img in &images {
    let sub = vk::ImageSubresourceRange {
      aspect_mask: vk::ImageAspectFlags::COLOR,
      base_mip_level: 0, level_count: 1,
      base_array_layer: 0, layer_count: 1,
      ..Default::default()
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

  // Render pass: single color attachment → present
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
  let att_ref = vk::AttachmentReference { attachment: 0, layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL };

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

  Ok((swapchain, surf_format.format, extent, images, views, render_pass, framebuffers))
}

unsafe fn build_renderer(
  window: &dyn HasWindowHandle,
  display: &dyn HasDisplayHandle,
  size: RenderSize,
) -> Result<VkRenderer> {
  let entry = Entry::linked();

  let dh: RawDisplayHandle = display.display_handle()?.as_raw();
  let wh: RawWindowHandle  = window.window_handle()?.as_raw();

  let instance = create_instance(&entry, dh)?;

  let surface = ash_window::create_surface(&entry, &instance, dh, wh, None)
    .context("create_surface")?;
  let surface_loader = surface::Instance::new(&entry, &instance);

  let (phys, queue_family) = pick_device_and_queue(&instance, &surface_loader, surface)?;

  // Device + queue
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

  let device = instance.create_device(phys, &dinfo, None).context("create_device")?;
  let queue = device.get_device_queue(queue_family, 0);

  // Swapchain bundle
  let swapchain_loader = swapchain::Device::new(&instance, &device);
  let (swapchain, format, extent, images, image_views, render_pass, framebuffers) =
    create_swapchain_bundle(&instance, &device, &surface_loader, &swapchain_loader, phys, surface, size)?;

  // Command pool + buffers
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

  let mut r = VkRenderer {
    _entry: entry,
    instance,
    surface_loader, surface,
    phys, device, queue_family, queue,
    swapchain_loader, swapchain, format, extent, images, image_views,
    render_pass, framebuffers,
    cmd_pool, cmd_bufs,
    clear: vk::ClearValue { color: vk::ClearColorValue { float32: [0.02, 0.02, 0.04, 1.0] } },
  };

  r.record_commands()?;
  Ok(r)
}

impl VkRenderer {
  unsafe fn record_commands(&mut self) -> Result<()> {
    for (i, &cmd) in self.cmd_bufs.iter().enumerate() {
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
        render_area: vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: self.extent },
        clear_value_count: clears.len() as u32,
        p_clear_values: clears.as_ptr(),
        ..Default::default()
      };

      self.device.cmd_begin_render_pass(cmd, &rp_begin, vk::SubpassContents::INLINE);
      self.device.cmd_end_render_pass(cmd);

      self.device.end_command_buffer(cmd)?;
    }
    Ok(())
  }

  unsafe fn recreate_swapchain(&mut self, size: RenderSize) -> Result<()> {
    self.device.device_wait_idle().ok();

    for &fb in &self.framebuffers { self.device.destroy_framebuffer(fb, None); }
    for &iv in &self.image_views { self.device.destroy_image_view(iv, None); }
    self.swapchain_loader.destroy_swapchain(self.swapchain, None);

    let (swapchain, format, extent, images, image_views, render_pass, framebuffers) =
      create_swapchain_bundle(&self.instance, &self.device, &self.surface_loader, &self.swapchain_loader, self.phys, self.surface, size)?;

    self.swapchain = swapchain;
    self.format = format;
    self.extent = extent;
    self.images = images;
    self.image_views = image_views;
    self.device.destroy_render_pass(self.render_pass, None);
    self.render_pass = render_pass;
    self.framebuffers = framebuffers;

    self.record_commands()?;
    Ok(())
  }
}

impl Renderer for VkRenderer {
  fn new(window: &dyn HasWindowHandle, display: &dyn HasDisplayHandle, size: RenderSize) -> Result<Self> {
    unsafe {
      let r = build_renderer(window, display, size)?;
      info!(
        "Vulkan swapchain ready ({}x{}, fmt 0x{:x})",
        r.extent.width, r.extent.height, r.format.as_raw()
      );
      Ok(r)
    }
  }

  fn resize(&mut self, size: RenderSize) -> Result<()> {
    unsafe { self.recreate_swapchain(size) }
  }

  fn render(&mut self) -> Result<()> {
    unsafe {
      // (no sync objects yet; just clear & present)
      let (image_index, _suboptimal) = self.swapchain_loader.acquire_next_image(
        self.swapchain, u64::MAX, vk::Semaphore::null(), vk::Fence::null()
      ).context("acquire_next_image")?;

      let cmd = self.cmd_bufs[image_index as usize];

      let submit = vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        ..Default::default()
      };
      self.device.queue_submit(self.queue, std::slice::from_ref(&submit), vk::Fence::null())
        .context("queue_submit")?;

      let present = vk::PresentInfoKHR {
        s_type: vk::StructureType::PRESENT_INFO_KHR,
        swapchain_count: 1,
        p_swapchains: &self.swapchain,
        p_image_indices: &image_index,
        ..Default::default()
      };
      self.swapchain_loader.queue_present(self.queue, &present)
        .context("queue_present")?;

      self.device.queue_wait_idle(self.queue).ok();
      Ok(())
    }
  }
}
