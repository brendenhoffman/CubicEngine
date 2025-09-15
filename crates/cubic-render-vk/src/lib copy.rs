use anyhow::{anyhow, Context, Result};
#[cfg(debug_assertions)]
use ash::ext::debug_utils as ext_debug;
use ash::khr::{surface, swapchain};
use ash::util::read_spv;
use ash::{vk, Entry, Instance};
use cubic_render::{RenderSize, Renderer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle};
use std::io::Cursor;
use tracing::info;

#[cfg(debug_assertions)]
type DebugState = vk::DebugUtilsMessengerEXT;
#[cfg(not(debug_assertions))]
type DebugState = ();

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
    paused: bool,

    #[allow(dead_code)]
    path: RenderPath,
    #[cfg(debug_assertions)]
    debug_messenger: vk::DebugUtilsMessengerEXT,
    acq_slots: Vec<AcquireSlot>,
    acq_index: usize,
    has_hdr_metadata_ext: bool,
    cfg: RuntimeConfig,

    depth_image: vk::Image,
    depth_mem: vk::DeviceMemory,
    depth_view: vk::ImageView,
    depth_format: vk::Format,
}

struct FrameSync {
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
    hdr_flavor: HdrFlavor,
}

struct SwapchainBundle {
    swapchain: vk::SwapchainKHR,
    format: vk::Format,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    color_space: vk::ColorSpaceKHR,
}

struct CommandResources {
    pool: vk::CommandPool,
    bufs: Vec<vk::CommandBuffer>,
}

#[derive(Clone, Copy, Debug)]
struct RuntimeConfig {
    vsync: bool,
    vsync_mode: VkVsyncMode,
    hdr: bool,
    hdr_flavor: HdrFlavor,
    allow_extended_colorspace: bool,
}
impl RuntimeConfig {
    /// Build from environment (CUBIC_HDR, CUBIC_HDR_FLAVOR), plus a flag
    /// detected at instance creation time.
    fn from_env(allow_extended_colorspace: bool) -> Self {
        let hdr = std::env::var("CUBIC_HDR").ok().as_deref() == Some("1");
        let hdr_flavor = match std::env::var("CUBIC_HDR_FLAVOR").ok().as_deref() {
            Some(s) if s.eq_ignore_ascii_case("hdr10") => HdrFlavor::PreferHdr10,
            _ => HdrFlavor::PreferScrgb,
        };

        Self {
            vsync: true,
            vsync_mode: VkVsyncMode::Mailbox,
            hdr,
            hdr_flavor,
            allow_extended_colorspace,
        }
    }

    /// Convert to the swapchain’s creation config for a given target size.
    fn to_swapchain_config(&self, hint: RenderSize) -> SwapchainConfig {
        SwapchainConfig {
            hint,
            vsync: self.vsync,
            vsync_mode: self.vsync_mode,
            want_hdr: self.hdr,
            allow_extended_colorspace: self.allow_extended_colorspace,
            hdr_flavor: self.hdr_flavor,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum VkVsyncMode {
    Fifo,    // Target monitor refresh rate
    Mailbox, // Smart Vsync, fps uncapped
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HdrFlavor {
    PreferScrgb, // FP16 scRGB first, then HDR10
    PreferHdr10, // HDR10 first, then scRGB
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
#[cfg(debug_assertions)]
unsafe fn create_debug_messenger(
    entry: &ash::Entry,
    instance: &ash::Instance,
) -> Result<DebugState> {
    let debug_loader = ext_debug::Instance::new(entry, instance);
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
    Ok(debug_loader.create_debug_utils_messenger(&ci, None)?)
}

#[cfg(not(debug_assertions))]
unsafe fn create_debug_messenger(
    _entry: &ash::Entry,
    _instance: &ash::Instance,
) -> Result<DebugState> {
    Ok(())
}

#[cfg(debug_assertions)]
unsafe fn destroy_debug_messenger(entry: &ash::Entry, instance: &ash::Instance, dbg: DebugState) {
    let loader = ext_debug::Instance::new(entry, instance);
    loader.destroy_debug_utils_messenger(dbg, None);
}

#[cfg(not(debug_assertions))]
unsafe fn destroy_debug_messenger(
    _entry: &ash::Entry,
    _instance: &ash::Instance,
    _dbg: DebugState,
) {
    // no-op
}

// STRICT TEARDOWN ORDER:
// - Wait all fences (acquire + in-flight)
// - device_wait_idle()
// - Destroy pipelines/layouts BEFORE swapchain
// - Destroy image views BEFORE swapchain
// - Free command buffers BEFORE destroying their pool
// - Destroy swapchain BEFORE device
// - Destroy per-frame semaphores/fences BEFORE device
// - Destroy surface AFTER device; instance last.
impl Drop for VkRenderer {
    fn drop(&mut self) {
        #[cfg(debug_assertions)]
        unsafe {
            let entry = Entry::linked();
            destroy_debug_messenger(&entry, &self.instance, self.debug_messenger);
        }

        unsafe {
            let d = &self.device;

            // 1) WAIT ALL FENCES (both per-image in-flight and acquire-slot)
            //    Ensures no submits are still referencing old resources.
            if !self.frames.is_empty() {
                let img_fences: Vec<_> = self.frames.iter().map(|f| f.in_flight).collect();
                let _ = d.wait_for_fences(&img_fences, true, u64::MAX);
            }
            if !self.acq_slots.is_empty() {
                let acq_fences: Vec<_> = self.acq_slots.iter().map(|s| s.fence).collect();
                let _ = d.wait_for_fences(&acq_fences, true, u64::MAX);
            }

            // 2) QUIESCE DEVICE (covers any remaining queue work)
            d.device_wait_idle().ok();

            // 3) PIPELINE & LAYOUTS BEFORE SWAPCHAIN (pipelines can depend on sc format)
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);

            // 4) IMAGE VIEWS BEFORE SWAPCHAIN (views are created from sc images)
            for &iv in &self.image_views {
                d.destroy_image_view(iv, None);
            }

            // 5) FREE COMMAND BUFFERS BEFORE DESTROYING THEIR POOL
            if !self.cmd_bufs.is_empty() {
                d.free_command_buffers(self.cmd_pool, &self.cmd_bufs);
            }
            d.destroy_command_pool(self.cmd_pool, None);

            // 6) DESTROY SWAPCHAIN BEFORE DEVICE
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);

            // 7) DESTROY PER-FRAME SYNCS (render-finished, in-flight) BEFORE DEVICE
            for f in &self.frames {
                d.destroy_fence(f.in_flight, None);
                d.destroy_semaphore(f.render_finished, None);
            }
            //    Also destroy acquire-slot syncs (sems + fences)
            for s in &self.acq_slots {
                d.destroy_fence(s.fence, None);
                d.destroy_semaphore(s.sem, None);
            }
            // Destroy depth
            d.destroy_image_view(self.depth_view, None);
            d.destroy_image(self.depth_image, None);
            d.free_memory(self.depth_mem, None);

            // 8) DESTROY DEVICE, THEN SURFACE, THEN INSTANCE
            d.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

// Info only
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

unsafe fn pick_depth_format(instance: &ash::Instance, phys: vk::PhysicalDevice) -> vk::Format {
    // Prefer 32f → 24+S8 → 16f
    let candidates = [
        vk::Format::D32_SFLOAT,
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D16_UNORM,
    ];
    for &fmt in &candidates {
        let props = instance.get_physical_device_format_properties(phys, fmt);
        let feats = props.optimal_tiling_features;
        if feats.contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
            return fmt;
        }
    }
    vk::Format::D32_SFLOAT // fallback; most desktop GPUs support it
}

unsafe fn create_depth_resources(
    instance: &ash::Instance,
    device: &ash::Device,
    phys: vk::PhysicalDevice,
    extent: vk::Extent2D,
    depth_format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let img_ci = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: vk::ImageType::TYPE_2D,
        format: depth_format,
        extent: vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let image = device.create_image(&img_ci, None)?;

    let mem_req = device.get_image_memory_requirements(image);
    let mem_props = instance.get_physical_device_memory_properties(phys);
    let mut mem_type_idx = 0;
    for i in 0..mem_props.memory_type_count {
        let is_type_ok = (mem_req.memory_type_bits & (1 << i)) != 0;
        let is_device_local = mem_props.memory_types[i as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        if is_type_ok && is_device_local {
            mem_type_idx = i;
            break;
        }
    }
    let alloc = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: mem_req.size,
        memory_type_index: mem_type_idx,
        ..Default::default()
    };
    let memory = device.allocate_memory(&alloc, None)?;
    device.bind_image_memory(image, memory, 0)?;

    let sub = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };
    let view_ci = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format: depth_format,
        subresource_range: sub,
        ..Default::default()
    };
    let view = device.create_image_view(&view_ci, None)?;
    Ok((image, memory, view))
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

unsafe fn init_instance_and_surface(
    window: &dyn HasWindowHandle,
    display: &dyn HasDisplayHandle,
) -> Result<(
    ash::Entry,
    ash::Instance,
    surface::Instance,
    vk::SurfaceKHR,
    DebugState,
    bool,
)> {
    // STRICT ORDER:
    // 1) Create VkInstance (enables platform WSI + debug ext)
    // 2) Create VkSurfaceKHR FROM THIS INSTANCE
    // 3) (Later) Query physical devices/queues AGAINST THIS SURFACE (present support)
    // 4) (Later) Create VkDevice using features/exts chosen for the selected phys
    // Changing this order => surface may be incompatible with chosen device/queue.

    // --- Platform handles (can fail on some backends) ---
    let dh = display
        .display_handle()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .as_raw();
    let wh = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .as_raw();

    // --- Instance (with required WSI + optional debug/swapchain-colorspace) ---
    let entry = Entry::linked();
    let instance = create_instance(&entry, dh)
        .context("create_instance (with WSI + optional debug/colorspace exts)")?;

    // --- Surface loader bound to THIS instance, then create surface from window/display ---
    let surface_loader = surface::Instance::new(&entry, &instance);
    let surface = ash_window::create_surface(&entry, &instance, dh, wh, None)
        .context("ash_window::create_surface")?;

    // --- Debug messenger (instance-scoped) ---
    let debug_state = create_debug_messenger(&entry, &instance)?;

    // --- Detect if the INSTANCE has the swapchain colorspace extension available ---
    // We record this bool to steer HDR surface format selection later.
    let have_swapchain_colorspace_ext = entry
        .enumerate_instance_extension_properties(None)
        .unwrap_or_default()
        .iter()
        .any(|e| {
            std::ffi::CStr::from_ptr(e.extension_name.as_ptr())
                == ash::ext::swapchain_colorspace::NAME
        });

    Ok((
        entry,
        instance,
        surface_loader,
        surface,
        debug_state,
        have_swapchain_colorspace_ext,
    ))
}

unsafe fn select_device_and_queue(
    instance: &ash::Instance,
    surf_i: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    pick_device_and_queue(instance, surf_i, surface)
}

unsafe fn decide_path_and_create_device(
    _entry: &ash::Entry,
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    queue_family: u32,
) -> Result<(
    ash::Device,
    vk::Queue,
    RenderPath,
    bool, /*has_hdr_metadata*/
)> {
    // STRICT ORDER (feature pNext chain):
    // Core 1.3 path: feats13 -> chained after feats12 -> chained after feats2
    // KHR path:      feats_sync2_khr -> feats_dr_khr -> feats12 -> feats2
    // DO NOT MIX core 1.3 structs with KHR equivalents in the same chain.
    // Wrong chain = undefined features; validation won't always catch it.

    // --- Queue we want on this device ---
    let priorities = [1.0_f32];
    let qinfo = vk::DeviceQueueCreateInfo {
        s_type: vk::StructureType::DEVICE_QUEUE_CREATE_INFO,
        queue_family_index: queue_family,
        queue_count: 1,
        p_queue_priorities: priorities.as_ptr(),
        ..Default::default()
    };

    // --- One shot device extension query ---
    let ext_props = instance
        .enumerate_device_extension_properties(phys)
        .context("enumerate_device_extension_properties(device)")?;
    let has = |name: &std::ffi::CStr| -> bool {
        ext_props
            .iter()
            .any(|e| std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == name)
    };

    let mut device_exts: Vec<*const i8> = vec![swapchain::NAME.as_ptr()];
    let has_sync2_khr = has(ash::khr::synchronization2::NAME);
    let has_dynren_khr = has(ash::khr::dynamic_rendering::NAME);
    let has_hdr_meta = has(ash::ext::hdr_metadata::NAME);
    if has_hdr_meta {
        device_exts.push(ash::ext::hdr_metadata::NAME.as_ptr());
    }

    // --- Feature structs (must outlive create_device); build the correct pNext chain ---
    let force_khr = std::env::var("CUBIC_FORCE_KHR").ok().as_deref() == Some("1");

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
    let mut feats2 = vk::PhysicalDeviceFeatures2 {
        s_type: vk::StructureType::PHYSICAL_DEVICE_FEATURES_2,
        ..Default::default()
    };

    let (path, pnext): (RenderPath, *const std::ffi::c_void) = if !force_khr {
        let dev_api = instance.get_physical_device_properties(phys).api_version;
        let maj = vk::api_version_major(dev_api);
        let min = vk::api_version_minor(dev_api);

        if maj > 1 || (maj == 1 && min >= 3) {
            // Core 1.3: enable core features only
            feats13.synchronization2 = vk::TRUE;
            feats13.dynamic_rendering = vk::TRUE;

            feats12.p_next = (&mut feats13) as *mut _ as *mut _;
            feats2.p_next = (&mut feats12) as *mut _ as *mut _;
            (RenderPath::Core13, (&mut feats2) as *mut _ as *const _)
        } else if has_sync2_khr && has_dynren_khr {
            // Vulkan 1.2 + KHR
            device_exts.push(ash::khr::synchronization2::NAME.as_ptr());
            device_exts.push(ash::khr::dynamic_rendering::NAME.as_ptr());

            feats_sync2_khr.synchronization2 = vk::TRUE;
            feats_dr_khr.dynamic_rendering = vk::TRUE;

            feats_sync2_khr.p_next = (&mut feats_dr_khr) as *mut _ as *mut _;
            feats12.p_next = (&mut feats_sync2_khr) as *mut _ as *mut _;
            feats2.p_next = (&mut feats12) as *mut _ as *mut _;
            (RenderPath::KhrExt, (&mut feats2) as *mut _ as *const _)
        } else {
            (RenderPath::Legacy, std::ptr::null())
        }
    } else {
        // Forced KHR path on 1.3 hardware (for testing)
        device_exts.push(ash::khr::synchronization2::NAME.as_ptr());
        device_exts.push(ash::khr::dynamic_rendering::NAME.as_ptr());

        feats_sync2_khr.synchronization2 = vk::TRUE;
        feats_dr_khr.dynamic_rendering = vk::TRUE;

        feats_sync2_khr.p_next = (&mut feats_dr_khr) as *mut _ as *mut _;
        feats12.p_next = (&mut feats_sync2_khr) as *mut _ as *mut _;
        feats2.p_next = (&mut feats12) as *mut _ as *mut _;
        (RenderPath::KhrExt, (&mut feats2) as *mut _ as *const _)
    };

    // IMPORTANT: if we’re on Legacy path, bail out BEFORE creating the device
    if let RenderPath::Legacy = path {
        return Err(anyhow!(
            "Dynamic rendering not available on this device; legacy render-pass path not compiled"
        ));
    }

    // --- Create device with our queue and the chosen feature chain ---
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
    Ok((device, queue, path, has_hdr_meta))
}

// ORDER NOTE: must be called AFTER creating the (new) swapchain and BEFORE first present.
// Scope: only HDR10/PQ swapchains need metadata; scRGB doesn't use VK_EXT_hdr_metadata.
unsafe fn create_hdr_metadata_if_needed(
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
    hdr.set_hdr_metadata(&[swapchain], std::slice::from_ref(&metadata));
}

unsafe fn create_command_resources(
    device: &ash::Device,
    queue_family: u32,
    image_count: usize,
) -> Result<CommandResources> {
    let pool_info = vk::CommandPoolCreateInfo {
        s_type: vk::StructureType::COMMAND_POOL_CREATE_INFO,
        queue_family_index: queue_family,
        flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
        ..Default::default()
    };
    let pool = device.create_command_pool(&pool_info, None)?;
    let alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: image_count as u32,
        ..Default::default()
    };
    let bufs = device.allocate_command_buffers(&alloc_info)?;
    Ok(CommandResources { pool, bufs })
}

unsafe fn create_sync_objects(
    device: &ash::Device,
    image_count: usize,
) -> Result<(Vec<AcquireSlot>, Vec<FrameSync>)> {
    let mut acq_slots = Vec::with_capacity(2);
    let mut frames = Vec::with_capacity(image_count);

    let sem_ci = vk::SemaphoreCreateInfo::default();
    let acq_fence_ci = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };
    for _ in 0..2 {
        let sem = device.create_semaphore(&sem_ci, None)?;
        let fence = device.create_fence(&acq_fence_ci, None)?;
        acq_slots.push(AcquireSlot { sem, fence });
    }

    let fence_ci = vk::FenceCreateInfo {
        s_type: vk::StructureType::FENCE_CREATE_INFO,
        flags: vk::FenceCreateFlags::SIGNALED,
        ..Default::default()
    };
    for _ in 0..image_count {
        let rf = device.create_semaphore(&sem_ci, None)?;
        let inflight = device.create_fence(&fence_ci, None)?;
        frames.push(FrameSync {
            render_finished: rf,
            in_flight: inflight,
        });
    }
    Ok((acq_slots, frames))
}

unsafe fn make_initial_swapchain_resources(
    device: &ash::Device,
    instance: &ash::Instance,
    surf_i: &surface::Instance,
    swap_d: &swapchain::Device,
    phys: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    cfg: SwapchainConfig,
    queue_family: u32,
    has_hdr_meta: bool,
    depth_format: vk::Format,
) -> Result<(
    SwapchainBundle,
    CommandResources,
    (vk::PipelineLayout, vk::Pipeline),
    Vec<AcquireSlot>,
    Vec<FrameSync>,
)> {
    let bundle = create_swapchain_bundle(
        device,
        surf_i,
        swap_d,
        phys,
        surface,
        vk::SwapchainKHR::null(),
        cfg,
    )?;

    create_hdr_metadata_if_needed(
        instance,
        device,
        has_hdr_meta,
        bundle.color_space,
        bundle.swapchain,
    );

    let image_count = bundle.image_views.len();
    let cmds = create_command_resources(device, queue_family, image_count)?;
    let pipe = create_pipeline(device, bundle.format, depth_format, bundle.extent)?;
    let (acq, frames) = create_sync_objects(device, image_count)?;
    Ok((bundle, cmds, pipe, acq, frames))
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

unsafe fn create_swapchain_bundle(
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
    let caps = surf_i.get_physical_device_surface_capabilities(phys, surface)?;
    // (format, colorspace) pairs exposed by WSI; must choose one
    let formats = surf_i.get_physical_device_surface_formats(phys, surface)?;
    // present modes: FIFO is always available; MAILBOX/IMMEDIATE are optional
    let modes = surf_i.get_physical_device_surface_present_modes(phys, surface)?;

    tracing::info!(
        "hdr_request={} allow_extended_colorspace={} have_swapchain_colorspace_ext={}",
        cfg.want_hdr,
        cfg.allow_extended_colorspace,
        cfg.allow_extended_colorspace
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
    // Heuristic: one more than minimum (for better overlap) but capped by max (0 == "no max").
    let min_count = if caps.max_image_count == 0 {
        caps.min_image_count + 1
    } else {
        (caps.min_image_count + 1).min(caps.max_image_count)
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
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE, // ignore window alpha
        present_mode,
        clipped: vk::TRUE, // don't care about obscured pixels
        old_swapchain,     // enables seamless re-creation w/ resource reuse
        ..Default::default()
    };

    // --- Create swapchain + fetch images ---
    let new_swapchain = swap_d.create_swapchain(&swap_info, None)?;
    let images = swap_d.get_swapchain_images(new_swapchain)?;

    // --- Create image views (one per swapchain image) ---
    // View format MUST match swapchain image format for direct rendering.
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

unsafe fn create_pipeline(
    device: &ash::Device,
    color_format: vk::Format,
    depth_format: vk::Format,
    _extent: vk::Extent2D,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    // STRICT: color_attachment_formats MUST match current swapchain image format.
    // On swapchain format change, pipeline must be rebuilt before recording.

    // --- Load + create shader modules (destroyed before return) ---
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

    // --- Shader stage infos ---
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

    // --- Fixed-function pipeline states ---
    // Vertex input (none → hardcoded triangle in shader)
    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        ..Default::default()
    };
    // Input assembly (triangles)
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };
    // Dynamic state (viewport/scissor set later)
    let dyn_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state = vk::PipelineDynamicStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        dynamic_state_count: dyn_states.len() as u32,
        p_dynamic_states: dyn_states.as_ptr(),
        ..Default::default()
    };
    // Viewport/scissor (placeholders, actual set at draw time)
    let viewport_state = vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        viewport_count: 1,
        p_viewports: std::ptr::null(), // dynamic
        scissor_count: 1,
        p_scissors: std::ptr::null(), // dynamic
        ..Default::default()
    };
    // Rasterization
    let raster = vk::PipelineRasterizationStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        polygon_mode: vk::PolygonMode::FILL,
        cull_mode: vk::CullModeFlags::BACK,
        front_face: vk::FrontFace::COUNTER_CLOCKWISE,
        line_width: 1.0,
        ..Default::default()
    };
    // Multisampling (disabled → 1 sample)
    let multisample = vk::PipelineMultisampleStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        rasterization_samples: vk::SampleCountFlags::TYPE_1,
        ..Default::default()
    };
    // Depth-stencil: enable depth test/write
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        depth_test_enable: vk::TRUE,
        depth_write_enable: vk::TRUE,
        depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
        // leave stencil disabled for now
        ..Default::default()
    };
    // Color blend (no blending; write all RGBA)
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

    // --- Pipeline layout (no descriptors/push constants yet) ---
    let layout_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        ..Default::default()
    };
    let layout = device.create_pipeline_layout(&layout_info, None)?;

    // --- Dynamic rendering info (ext / core 1.3 replacement for render passes) ---
    let rendering = vk::PipelineRenderingCreateInfo {
        s_type: vk::StructureType::PIPELINE_RENDERING_CREATE_INFO,
        color_attachment_count: 1,
        p_color_attachment_formats: &color_format,
        depth_attachment_format: depth_format,
        ..Default::default()
    };

    // --- Graphics pipeline create info (glues everything together) ---
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
        p_depth_stencil_state: &depth_stencil,
        p_color_blend_state: &color_blend,
        p_dynamic_state: &dynamic_state,
        layout,
        ..Default::default()
    };

    // --- Create pipeline; destroy shader modules afterwards ---
    let pipelines = match device.create_graphics_pipelines(
        vk::PipelineCache::null(),
        std::slice::from_ref(&pipeline_info),
        None,
    ) {
        Ok(p) => p,
        Err((_, err)) => return Err(anyhow!("create_graphics_pipelines failed: {:?}", err)),
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
    // 1) Instance + surface (and record whether colorspace ext exists)
    let (entry, instance, surface_loader, surface, debug_state, have_swapchain_colorspace_ext) =
        init_instance_and_surface(window, display)?;

    // 2) Pick device/queue family
    let (phys, queue_family) = select_device_and_queue(&instance, &surface_loader, surface)?;

    // 3) Create device + choose render path, detect HDR metadata support
    let (device, queue, path, has_hdr_meta) =
        decide_path_and_create_device(&entry, &instance, phys, queue_family)?;

    // 4) WSI device wrapper
    let swapchain_loader = swapchain::Device::new(&instance, &device);

    // 5) Initial runtime knobs
    let initial_cfg = RuntimeConfig::from_env(have_swapchain_colorspace_ext);
    let cfg = initial_cfg.to_swapchain_config(size);
    let depth_format = pick_depth_format(&instance, phys);

    // 6) Build all swapchain-scoped resources in one place
    let (sc, cmd, (pipeline_layout, pipeline), acq_slots, frames) =
        make_initial_swapchain_resources(
            &device,
            &instance,
            &surface_loader,
            &swapchain_loader,
            phys,
            surface,
            cfg,
            queue_family,
            has_hdr_meta,
            depth_format,
        )?;

    let depth_format = pick_depth_format(&instance, phys);
    let (depth_image, depth_mem, depth_view) =
        create_depth_resources(&instance, &device, phys, sc.extent, depth_format)?;

    // 7) Assemble VkRenderer
    let mut r = VkRenderer {
        instance,
        surface_loader,
        surface,

        phys,
        device,
        queue,

        swapchain_loader,
        swapchain: sc.swapchain,
        format: sc.format,
        extent: sc.extent,

        images: sc.images,
        image_views: sc.image_views,

        pipeline,
        pipeline_layout,
        cmd_pool: cmd.pool,
        cmd_bufs: cmd.bufs,

        frames,
        clear: vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.02, 0.02, 0.04, 1.0],
            },
        },
        paused: false,
        path,

        #[cfg(debug_assertions)]
        debug_messenger: debug_state,
        acq_slots,
        acq_index: 0,
        has_hdr_metadata_ext: has_hdr_meta,
        cfg: initial_cfg,
        depth_image,
        depth_mem,
        depth_view,
        depth_format,
    };

    // 8) Record per-image command buffers once
    r.record_commands()?;

    Ok(r)
}

impl VkRenderer {
    // Set cfg options
    pub fn set_vsync_mode(&mut self, mode: VkVsyncMode) {
        if self.cfg.vsync_mode as u8 == mode as u8 {
            return;
        }
        self.cfg.vsync_mode = mode;
        let want = RenderSize {
            width: self.extent.width,
            height: self.extent.height,
        };
        let _ = unsafe { self.recreate_swapchain(want) };
    }
    pub fn set_hdr_enabled(&mut self, on: bool) {
        if self.cfg.hdr == on {
            return;
        }
        self.cfg.hdr = on;
        let want = RenderSize {
            width: self.extent.width,
            height: self.extent.height,
        };
        let _ = unsafe { self.recreate_swapchain(want) };
    }
    pub fn set_hdr_flavor(&mut self, flavor: HdrFlavor) {
        if self.cfg.hdr_flavor == flavor {
            return;
        }
        self.cfg.hdr_flavor = flavor;
        let want = RenderSize {
            width: self.extent.width,
            height: self.extent.height,
        };
        let _ = unsafe { self.recreate_swapchain(want) };
    }

    #[inline]
    unsafe fn transition_to_color(&self, cmd: vk::CommandBuffer, image: vk::Image) {
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
    }

    #[inline]
    unsafe fn begin_rendering(&self, cmd: vk::CommandBuffer, image_view: vk::ImageView) {
        let color_att = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view,
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
    }

    #[inline]
    unsafe fn bind_draw_fullscreen_triangle(&self, cmd: vk::CommandBuffer) -> Result<()> {
        if self.pipeline == vk::Pipeline::null() {
            return Err(anyhow!("pipeline is VK_NULL_HANDLE at record time"));
        }

        self.device
            .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

        // dynamic viewport/scissor
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
        Ok(())
    }

    #[inline]
    unsafe fn transition_to_present(&self, cmd: vk::CommandBuffer, image: vk::Image) {
        let subrange = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

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
    }

    #[inline]
    unsafe fn record_one_command(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        image_view: vk::ImageView,
    ) -> Result<()> {
        // reset + begin
        self.device
            .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            ..Default::default()
        };
        self.device.begin_command_buffer(cmd, &begin)?;

        // body
        self.transition_to_color(cmd, image);
        self.begin_rendering(cmd, image_view);
        self.bind_draw_fullscreen_triangle(cmd)?;
        self.device.cmd_end_rendering(cmd);
        self.transition_to_present(cmd, image);

        // end
        self.device.end_command_buffer(cmd)?;
        Ok(())
    }

    // --- Public-ish: record all per-swapchain-image CBs ----------------------

    unsafe fn record_commands(&mut self) -> Result<()> {
        for (i, &cmd) in self.cmd_bufs.iter().enumerate() {
            self.record_one_command(cmd, self.images[i], self.image_views[i])?;
        }
        Ok(())
    }

    // STRICT ORDER (recreate):
    // 1) Wait all in-flight image fences + acquire fences (no work using old sc)
    // 2) device_wait_idle() to avoid destroying in-use views/pipelines
    // 3) Destroy per-image views + per-image sync tied to OLD swapchain
    // 4) Create NEW swapchain + images + views
    // 5) Recreate per-image sync objects
    // 6) Recreate pipeline ONLY if format changed
    // 7) Resize command buffers if image count changed
    // 8) Re-record commands for ALL images
    // Any deviation can cause sporadic DEVICE_LOST or image-in-use errors.
    unsafe fn recreate_swapchain(&mut self, size: RenderSize) -> Result<()> {
        // Guard min size window
        if size.width == 0 || size.height == 0 {
            return Ok(());
        }

        // 1) Wait all in-flight image fences + acquire fences (no work using old sc)
        let image_fences: Vec<_> = self.frames.iter().map(|f| f.in_flight).collect();
        if !image_fences.is_empty() {
            let _ = self.device.wait_for_fences(&image_fences, true, u64::MAX);
        }
        let acq_fences: Vec<_> = self.acq_slots.iter().map(|s| s.fence).collect();
        if !acq_fences.is_empty() {
            let _ = self.device.wait_for_fences(&acq_fences, true, u64::MAX);
        }

        // 2) device_wait_idle() to avoid destroying in-use views/pipelines
        self.device.device_wait_idle().ok();

        // 3) Destroy per-image views + per-image sync tied to OLD swapchain
        for &iv in &self.image_views {
            self.device.destroy_image_view(iv, None);
        }
        for f in &self.frames {
            self.device.destroy_fence(f.in_flight, None);
            self.device.destroy_semaphore(f.render_finished, None);
        }
        self.frames.clear();

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
        self.swapchain_loader
            .destroy_swapchain(self.swapchain, None);
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

        // 5) Recreate per-image syn
        let image_count = self.images.len();
        let sem_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo {
            s_type: vk::StructureType::FENCE_CREATE_INFO,
            flags: vk::FenceCreateFlags::SIGNALED,
            ..Default::default()
        };
        for _ in 0..image_count {
            let rf = self.device.create_semaphore(&sem_info, None)?;
            let inflight = self.device.create_fence(&fence_info, None)?;
            self.frames.push(FrameSync {
                render_finished: rf,
                in_flight: inflight,
            });
        }

        // 6) Recreate pipeline only if COLOR format changed
        if self.format != old_format {
            let (new_layout, new_pipeline) =
                create_pipeline(&self.device, self.format, self.depth_format, self.extent)?;
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.pipeline_layout = new_layout;
            self.pipeline = new_pipeline;
        }

        // 7) Resize CBs if image count changed
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

        // 8) Record
        self.acq_index = 0;
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
        if self.cfg.vsync == on {
            return;
        }
        self.cfg.vsync = on;
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

    // STRICT PER-FRAME ORDER:
    // 1) acquire_next_image (waits on acquire semaphore)
    // 2) queue_submit (signals render-finished for THIS image)
    // 3) queue_present (waits on render-finished)
    // Each swapchain image has its own FrameSync; do not cross-use semaphores.
    fn render(&mut self) -> Result<()> {
        // Guard on pause
        if self.paused {
            return Ok(());
        }

        unsafe {
            // Guard for min surface
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

            // Use current acquire slot
            let s = &self.acq_slots[self.acq_index];

            // Make sure this slot is free (completed last submit that used it)
            self.device
                .wait_for_fences(&[s.fence], true, u64::MAX)
                .context("wait_for_fences(acquire slot)")?;
            self.device.reset_fences(&[s.fence])?;

            // 1) Acquire
            let (image_index, _) = match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                s.sem,
                vk::Fence::null(),
            ) {
                Ok(pair) => pair,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    let want = RenderSize {
                        width: self.extent.width,
                        height: self.extent.height,
                    };
                    let _ = self.recreate_swapchain(want);
                    // Skip this frame after recreate; try again next tick.
                    return Ok(());
                }
                Err(e) => return Err(anyhow::anyhow!("acquire_next_image: {e:?}")),
            };

            let img = image_index as usize;
            let f_img = &self.frames[img];
            let cmd = self.cmd_bufs[img];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];

            // 2) Submit (wait on acquire sem; signal render-finished; track with this slot's fence)
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

            // 3) Present (wait on render-finished)
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
                Ok(_) => { /* all good */ }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    // Recreate and continue next frame
                    let want = RenderSize {
                        width: self.extent.width,
                        height: self.extent.height,
                    };
                    let _ = self.recreate_swapchain(want);
                }
                Err(e) => return Err(anyhow::anyhow!("queue_present: {e:?}")),
            }

            // Rotate acquire slot
            self.acq_index = (self.acq_index + 1) % self.acq_slots.len();

            Ok(())
        }
    }
}
