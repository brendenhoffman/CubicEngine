// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::{anyhow, Context, Result};
#[cfg(debug_assertions)]
use ash::ext::debug_utils as ext_debug;
use ash::khr::{surface, swapchain};
use ash::util::read_spv;
use ash::{vk, Entry, Instance};
use bytemuck::{Pod, Zeroable};
use cubic_render::{RenderSize, Renderer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::io::Cursor;
#[cfg(debug_assertions)]
use std::time::SystemTime;
use std::{fs, path::Path, path::PathBuf};
use tracing::info;

// 1) Public api / constants
const UV_TILE: f32 = 1.0;

const TRI_VERTS: &[Vertex] = &[
    // FRONT triangle (closer)
    Vertex {
        pos: [0.0, 0.5, -0.988],
        color: [1.0, 0.2, 0.2],
        uv: [0.5 * UV_TILE, 1.0 * UV_TILE], // top
    },
    Vertex {
        pos: [-0.3, -0.4, -0.788],
        color: [1.0, 0.2, 0.2],
        uv: [0.0 * UV_TILE, 0.0 * UV_TILE], // bottom-left
    },
    Vertex {
        pos: [0.3, -0.4, -0.788],
        color: [1.0, 0.2, 0.2],
        uv: [1.0 * UV_TILE, 0.0 * UV_TILE], // bottom-right
    },
    // BACK triangle (farther)
    Vertex {
        pos: [0.0, 0.4, -0.888],
        color: [0.2, 0.8, 1.0],
        uv: [0.5 * UV_TILE, 1.0 * UV_TILE], // top
    },
    Vertex {
        pos: [-0.5, -0.4, -0.888],
        color: [0.2, 0.8, 1.0],
        uv: [0.0 * UV_TILE, 0.0 * UV_TILE], // bottom-left
    },
    Vertex {
        pos: [0.5, -0.4, -0.888],
        color: [0.2, 0.8, 1.0],
        uv: [1.0 * UV_TILE, 0.0 * UV_TILE], // bottom-right
    },
];

const TRI_IDXS: &[u32] = &[0, 1, 2, 3, 4, 5];

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
// END Public api / constants

// 2) Debug wiring
#[cfg(debug_assertions)]
type DebugState = vk::DebugUtilsMessengerEXT;
#[cfg(not(debug_assertions))]
type DebugState = ();

#[cfg(debug_assertions)]
unsafe extern "system" fn debug_callback(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    if !data.is_null() {
        let msg = unsafe { std::ffi::CStr::from_ptr((*data).p_message) };
        eprintln!("[Vulkan] {:?}", msg);
    }
    vk::FALSE
}
#[cfg(debug_assertions)]
fn create_debug_messenger(entry: &ash::Entry, instance: &ash::Instance) -> Result<DebugState> {
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
    Ok(unsafe { debug_loader.create_debug_utils_messenger(&ci, None)? })
}
#[cfg(not(debug_assertions))]
fn create_debug_messenger(_entry: &ash::Entry, _instance: &ash::Instance) -> Result<DebugState> {
    Ok(())
}

#[cfg(debug_assertions)]
fn destroy_debug_messenger(entry: &ash::Entry, instance: &ash::Instance, dbg: DebugState) {
    let loader = ext_debug::Instance::new(entry, instance);
    unsafe { loader.destroy_debug_utils_messenger(dbg, None) };
}

#[cfg(debug_assertions)]
struct ShaderDev {
    vert_spv: PathBuf,
    frag_spv: PathBuf,
    vert_mtime: SystemTime,
    frag_mtime: SystemTime,
}
//END Debug wiring

// 3) Renderer data model
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
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    acq_slots: Vec<AcquireSlot>,
    acq_index: usize,
    has_hdr_metadata_ext: bool,
    cfg: RuntimeConfig,

    depth_image: vk::Image,
    depth_mem: vk::DeviceMemory,
    depth_view: vk::ImageView,
    depth_format: vk::Format,
    vbuf: vk::Buffer,
    vbuf_mem: vk::DeviceMemory,
    ibuf: vk::Buffer,
    ibuf_mem: vk::DeviceMemory,
    index_count: u32,
    desc_pool: vk::DescriptorPool,
    desc_set_layout_camera: vk::DescriptorSetLayout,
    desc_set_layout_material: vk::DescriptorSetLayout,
    desc_sets: Vec<vk::DescriptorSet>,
    ubufs: Vec<vk::Buffer>,
    umems: Vec<vk::DeviceMemory>,
    ubo_ptrs: Vec<*mut std::ffi::c_void>,
    ubo_size: vk::DeviceSize,
    pipeline_cache: vk::PipelineCache,
    timeline: vk::Semaphore,
    timeline_value: u64,
    display_raw: RawDisplayHandle,
    window_raw: RawWindowHandle,
    backoff_frames: u32,
    #[cfg(debug_assertions)]
    shader_dev: Option<ShaderDev>,
    material_desc_pool: vk::DescriptorPool,
    material_desc_set: vk::DescriptorSet,
    tex_image: vk::Image,
    tex_mem: vk::DeviceMemory,
    tex_view: vk::ImageView,
    tex_sampler: vk::Sampler,
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
        {
            let entry = Entry::linked();
            if let Some(dbg) = self.debug_messenger {
                destroy_debug_messenger(&entry, &self.instance, dbg);
            }
        }

        unsafe {
            let d = &self.device;

            // 1) Wait GPU to finish last work we submitted via timeline
            if self.timeline_value > 0 {
                let wait_info = vk::SemaphoreWaitInfo {
                    s_type: vk::StructureType::SEMAPHORE_WAIT_INFO,
                    flags: vk::SemaphoreWaitFlags::empty(),
                    semaphore_count: 1,
                    p_semaphores: &self.timeline,
                    p_values: &self.timeline_value,
                    ..Default::default()
                };
                let _ = d.wait_semaphores(&wait_info, u64::MAX);
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
                d.destroy_semaphore(f.render_finished, None);
            }
            //    Also destroy acquire-slot syncs (sems + fences)
            for s in &self.acq_slots {
                d.destroy_semaphore(s.sem, None);
            }
            // Destroy timeline semaphore
            d.destroy_semaphore(self.timeline, None);

            // Destroy depth
            d.destroy_image_view(self.depth_view, None);
            d.destroy_image(self.depth_image, None);
            d.free_memory(self.depth_mem, None);

            // Destroy vertex/image buffers
            d.destroy_buffer(self.vbuf, None);
            d.free_memory(self.vbuf_mem, None);
            d.destroy_buffer(self.ibuf, None);
            d.free_memory(self.ibuf_mem, None);

            // Destroy frame resources
            for (i, &m) in self.umems.iter().enumerate() {
                let p = self
                    .ubo_ptrs
                    .get(i)
                    .copied()
                    .unwrap_or(std::ptr::null_mut());
                if !p.is_null() {
                    self.device.unmap_memory(m);
                }
            }
            for &b in &self.ubufs {
                self.device.destroy_buffer(b, None);
            }
            for &m in &self.umems {
                self.device.free_memory(m, None);
            }
            self.ubufs.clear();
            self.umems.clear();
            self.ubo_ptrs.clear();
            self.ubo_size = 0;
            if self.desc_pool != vk::DescriptorPool::null() {
                d.destroy_descriptor_pool(self.desc_pool, None);
            }
            if self.desc_set_layout_material != vk::DescriptorSetLayout::null() {
                d.destroy_descriptor_set_layout(self.desc_set_layout_material, None);
            }
            if self.desc_set_layout_camera != vk::DescriptorSetLayout::null() {
                d.destroy_descriptor_set_layout(self.desc_set_layout_camera, None);
            }

            // Material descriptor pool (set is freed with pool)
            d.destroy_descriptor_pool(self.material_desc_pool, None);

            // Texture + sampler
            d.destroy_sampler(self.tex_sampler, None);
            d.destroy_image_view(self.tex_view, None);
            d.destroy_image(self.tex_image, None);
            d.free_memory(self.tex_mem, None);

            // Save and destroy pipeline cache
            let props = self.instance.get_physical_device_properties(self.phys);
            let cache_path = pipeline_cache_path(&props);
            let _ = save_pipeline_cache(d, self.pipeline_cache, &cache_path);
            d.destroy_pipeline_cache(self.pipeline_cache, None);

            // 8) DESTROY DEVICE, THEN SURFACE, THEN INSTANCE
            d.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
// END Renderer data model

// 4) Structs
struct FrameSync {
    render_finished: vk::Semaphore,
}

struct AcquireSlot {
    sem: vk::Semaphore,
    last_signal_value: u64,
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

#[repr(C)]
#[derive(Clone, Copy, Default, Zeroable, Pod)]
struct CameraUbo {
    mvp: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct PushData {
    model: [[f32; 4]; 4],
    tint: [f32; 4],
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
    fn to_swapchain_config(self, hint: RenderSize) -> SwapchainConfig {
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

struct SwapchainInitInput<'a> {
    device: &'a ash::Device,
    instance: &'a ash::Instance,
    surf_i: &'a surface::Instance,
    swap_d: &'a swapchain::Device,
    phys: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    cfg: SwapchainConfig,
    queue_family: u32,
    has_hdr_meta: bool,
    pipeline_cache: vk::PipelineCache,
    depth_format: vk::Format,
    desc_set_layout_camera: vk::DescriptorSetLayout,
    desc_set_layout_material: vk::DescriptorSetLayout,
}

struct DeviceCtx<'a> {
    instance: &'a ash::Instance,
    device: &'a ash::Device,
    phys: vk::PhysicalDevice,
}

struct ImageAllocInfo {
    extent: vk::Extent2D,
    mip_levels: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    tiling: vk::ImageTiling,
}

struct LayoutTransition {
    image: vk::Image,
    sub: vk::ImageSubresourceRange,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    old_layout: vk::ImageLayout,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
    new_layout: vk::ImageLayout,
}
// END Structs

// 5) Enums
#[derive(Clone, Copy, Debug)]
enum RenderPath {
    Core13, // Vulkan 1.3 core dynamic rendering + sync2
    KhrExt, // Vulkan 1.2 + VK_KHR_dynamic_rendering + VK_KHR_synchronization2
    Legacy, // No dynamic rendering: would need render pass/framebuffer path
}
// END Enums

// 6) Types
type InitRet = (
    ash::Entry,
    ash::Instance,
    surface::Instance,
    vk::SurfaceKHR,
    Option<DebugState>,
    bool,
);
type FrameUniforms = (
    Vec<vk::Buffer>,
    Vec<vk::DeviceMemory>,
    Vec<*mut std::ffi::c_void>,
    vk::DeviceSize,
    vk::DescriptorPool,
    Vec<vk::DescriptorSet>,
);

type SwapchainInit = (
    SwapchainBundle,
    CommandResources,
    (vk::PipelineLayout, vk::Pipeline),
    Vec<AcquireSlot>,
    Vec<FrameSync>,
);
// END Types

// 7) Inline helper functions
#[inline]
fn find_memory_type(
    instance: &ash::Instance,
    phys: vk::PhysicalDevice,
    type_bits: u32,
    req: vk::MemoryPropertyFlags,
) -> anyhow::Result<u32> {
    let mem = unsafe { instance.get_physical_device_memory_properties(phys) };

    for i in 0..mem.memory_type_count {
        let type_ok = (type_bits & (1 << i)) != 0;
        let props_ok = mem.memory_types[i as usize].property_flags.contains(req);
        if type_ok && props_ok {
            return Ok(i);
        }
    }

    Err(anyhow!(
        "no suitable memory type: type_bits=0x{type_bits:08x}, required_flags={req:?}"
    ))
}

#[inline]
fn stage_flags2_from_legacy(stage: vk::PipelineStageFlags) -> vk::PipelineStageFlags2 {
    let mut out = vk::PipelineStageFlags2::empty();
    if stage.contains(vk::PipelineStageFlags::TOP_OF_PIPE) {
        out |= vk::PipelineStageFlags2::TOP_OF_PIPE;
    }
    if stage.contains(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT) {
        out |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
    }
    if stage.contains(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS) {
        out |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS;
    }
    if stage.contains(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS) {
        out |= vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
    }
    out
}

#[inline]
fn semaphore_submit_info_wait(
    sem: vk::Semaphore,
    value: u64,
    stage: vk::PipelineStageFlags2,
) -> vk::SemaphoreSubmitInfo<'static> {
    vk::SemaphoreSubmitInfo {
        s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
        p_next: std::ptr::null(),
        semaphore: sem,
        value,
        stage_mask: stage,
        device_index: 0,
        ..Default::default()
    }
}

#[inline]
fn semaphore_submit_info_signal(
    sem: vk::Semaphore,
    value: u64,
    stage: vk::PipelineStageFlags2,
) -> vk::SemaphoreSubmitInfo<'static> {
    vk::SemaphoreSubmitInfo {
        s_type: vk::StructureType::SEMAPHORE_SUBMIT_INFO,
        p_next: std::ptr::null(),
        semaphore: sem,
        value,
        stage_mask: stage,
        device_index: 0,
        ..Default::default()
    }
}

#[inline]
fn is_swapchain_out_of_date(e: vk::Result) -> bool {
    matches!(
        e,
        vk::Result::ERROR_OUT_OF_DATE_KHR | vk::Result::SUBOPTIMAL_KHR
    )
}

#[inline]
fn is_surface_lost(e: vk::Result) -> bool {
    e == vk::Result::ERROR_SURFACE_LOST_KHR
}

#[inline]
fn is_device_lost(e: vk::Result) -> bool {
    e == vk::Result::ERROR_DEVICE_LOST
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
#[inline]
fn depth_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    let mut aspect = vk::ImageAspectFlags::DEPTH;
    if has_stencil(format) {
        aspect |= vk::ImageAspectFlags::STENCIL;
    }
    aspect
}

#[inline]
fn depth_attachment_layout(format: vk::Format) -> vk::ImageLayout {
    if has_stencil(format) {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
    }
}

#[inline]
fn transition_color_to_transfer_dst(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    mips: u32,
) {
    let sub = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: mips,
        base_array_layer: 0,
        layer_count: 1,
    };
    transition_image_layout2(
        device,
        cmd,
        &LayoutTransition {
            image,
            sub,
            src_stage: vk::PipelineStageFlags2::TOP_OF_PIPE,
            src_access: vk::AccessFlags2::empty(),
            old_layout: vk::ImageLayout::UNDEFINED,
            dst_stage: vk::PipelineStageFlags2::TRANSFER,
            dst_access: vk::AccessFlags2::TRANSFER_WRITE,
            new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        },
    );
}

#[inline]
fn transition_color_to_shader_read(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    mips: u32,
) {
    let sub = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: mips,
        base_array_layer: 0,
        layer_count: 1,
    };
    transition_image_layout2(
        device,
        cmd,
        &LayoutTransition {
            image,
            sub,
            src_stage: vk::PipelineStageFlags2::TRANSFER,
            src_access: vk::AccessFlags2::TRANSFER_WRITE,
            old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            dst_stage: vk::PipelineStageFlags2::FRAGMENT_SHADER,
            dst_access: vk::AccessFlags2::SHADER_READ,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        },
    );
}
// END Inline helper functions

// 8) Helper functions
fn load_spv_file(path: &Path) -> Result<Vec<u32>> {
    let bytes = fs::read(path).with_context(|| format!("read {:?}", path))?;
    read_spv(&mut Cursor::new(&bytes[..])).with_context(|| format!("read_spv {:?}", path))
}

fn load_spv_bytes(bytes: &[u8]) -> Result<Vec<u32>> {
    read_spv(&mut Cursor::new(bytes)).context("read_spv from embedded bytes")
}

fn hex_bytes(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        use std::fmt::Write as _;
        let _ = write!(&mut s, "{:02x}", x);
    }
    s
}

fn pipeline_cache_path(props: &vk::PhysicalDeviceProperties) -> PathBuf {
    // Keep it simple: local file next to the binary.
    // You can switch to a platform cache dir later.
    let uuid = hex_bytes(&props.pipeline_cache_uuid);
    PathBuf::from(format!(
        "vk_pipeline_cache_{:04x}_{:04x}_{:08x}_{}.bin",
        props.vendor_id, props.device_id, props.driver_version, uuid
    ))
}

fn create_or_load_pipeline_cache(
    device: &ash::Device,
    path: &PathBuf,
) -> Result<vk::PipelineCache> {
    let (p_initial_data, initial_size);
    let data = fs::read(path).ok();
    if let Some(ref bytes) = data {
        p_initial_data = bytes.as_ptr() as *const std::ffi::c_void;
        initial_size = bytes.len();
    } else {
        p_initial_data = std::ptr::null();
        initial_size = 0;
    }

    let ci = vk::PipelineCacheCreateInfo {
        s_type: vk::StructureType::PIPELINE_CACHE_CREATE_INFO,
        initial_data_size: initial_size,
        p_initial_data,
        ..Default::default()
    };
    let cache = unsafe { device.create_pipeline_cache(&ci, None)? };
    Ok(cache)
}

fn save_pipeline_cache(
    device: &ash::Device,
    cache: vk::PipelineCache,
    path: &PathBuf,
) -> Result<()> {
    let bytes = match unsafe { device.get_pipeline_cache_data(cache) } {
        Ok(b) => b,
        Err(_) => return Ok(()),
    };

    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(path, &bytes)?;
    Ok(())
}

// Prefer pure depth formats only: D32F -> D16
fn pick_depth_format(instance: &ash::Instance, phys: vk::PhysicalDevice) -> vk::Format {
    for &fmt in &[vk::Format::D32_SFLOAT, vk::Format::D16_UNORM] {
        let props = unsafe { instance.get_physical_device_format_properties(phys, fmt) };
        if props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
            return fmt;
        }
    }
    vk::Format::D32_SFLOAT
}

fn has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT
    )
}

fn make_depth_view(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
) -> anyhow::Result<vk::ImageView> {
    let mut aspect = vk::ImageAspectFlags::DEPTH;
    if has_stencil(format) {
        aspect |= vk::ImageAspectFlags::STENCIL;
    }
    let sub = vk::ImageSubresourceRange {
        aspect_mask: aspect,
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

fn create_depth_resources(
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
    let image = unsafe { device.create_image(&img_ci, None) }.with_context(|| {
        format!(
            "create_image depth format={depth_format:?} extent={:?}",
            extent
        )
    })?;

    let mem_req = unsafe { device.get_image_memory_requirements(image) };
    let mem_type_idx = find_memory_type(
        instance,
        phys,
        mem_req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .with_context(|| {
        format!(
            "depth image memory selection: req_bits=0x{:08x}, size={}",
            mem_req.memory_type_bits, mem_req.size
        )
    })?;

    let alloc = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: mem_req.size,
        memory_type_index: mem_type_idx,
        ..Default::default()
    };
    let memory = unsafe { device.allocate_memory(&alloc, None) }.with_context(|| {
        format!(
            "allocate_memory (depth) size={} mem_type_index={}",
            mem_req.size, mem_type_idx
        )
    })?;

    unsafe { device.bind_image_memory(image, memory, 0) }
        .with_context(|| "bind_image_memory (depth)")?;

    let depth_view = make_depth_view(device, image, depth_format)?;
    Ok((image, memory, depth_view))
}

fn create_instance(entry: &Entry, display_raw: RawDisplayHandle) -> Result<(Instance, bool)> {
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

    let inst_exts = unsafe {
        entry
            .enumerate_instance_extension_properties(None)
            .context("enumerate_instance_extension_properties(instance)")?
    };
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

    let instance = unsafe { entry.create_instance(&create_info, None)? };
    Ok((instance, has_swapchain_cs))
}

fn init_instance_and_surface(
    window: &dyn HasWindowHandle,
    display: &dyn HasDisplayHandle,
) -> anyhow::Result<InitRet> {
    let dh = display
        .display_handle()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .as_raw();
    let wh = window
        .window_handle()
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .as_raw();

    let entry = Entry::linked();

    let (instance, have_swapchain_colorspace_ext) = create_instance(&entry, dh)?;

    let surface_loader = surface::Instance::new(&entry, &instance);

    let surface = unsafe {
        ash_window::create_surface(&entry, &instance, dh, wh, None)
            .context("ash_window::create_surface")?
    };

    let debug_state = if cfg!(debug_assertions) {
        Some(create_debug_messenger(&entry, &instance)?)
    } else {
        None
    };

    Ok((
        entry,
        instance,
        surface_loader,
        surface,
        debug_state,
        have_swapchain_colorspace_ext,
    ))
}

fn select_device_and_queue(
    instance: &ash::Instance,
    surf_i: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    pick_device_and_queue(instance, surf_i, surface)
}

// ORDER NOTE: must be called AFTER creating the (new) swapchain and BEFORE first present.
// Scope: only HDR10/PQ swapchains need metadata; scRGB doesn't use VK_EXT_hdr_metadata.
fn create_hdr_metadata_if_needed(
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

fn create_command_resources(
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
    let pool = unsafe { device.create_command_pool(&pool_info, None)? };
    let alloc_info = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: image_count as u32,
        ..Default::default()
    };
    let bufs = unsafe { device.allocate_command_buffers(&alloc_info)? };
    Ok(CommandResources { pool, bufs })
}

fn create_buffer_and_memory(
    instance: &ash::Instance,
    device: &ash::Device,
    phys: vk::PhysicalDevice,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    props: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let bci = vk::BufferCreateInfo {
        s_type: vk::StructureType::BUFFER_CREATE_INFO,
        size,
        usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let buf = unsafe { device.create_buffer(&bci, None) }
        .with_context(|| format!("create_buffer usage={usage:?} size={size}"))?;

    let req = unsafe { device.get_buffer_memory_requirements(buf) };
    let mem_type = find_memory_type(instance, phys, req.memory_type_bits, props)
        .with_context(|| format!("buffer memory selection for usage={usage:?}, props={props:?}, size={size}, req_bits=0x{:08x}", req.memory_type_bits))?;

    let mai = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index: mem_type,
        ..Default::default()
    };
    let mem = unsafe { device.allocate_memory(&mai, None) }.with_context(|| {
        format!(
            "allocate_memory size={} mem_type_index={}",
            req.size, mem_type
        )
    })?;

    unsafe { device.bind_buffer_memory(buf, mem, 0) }.with_context(|| "bind_buffer_memory")?;

    Ok((buf, mem))
}

fn create_host_visible_ubo(
    instance: &ash::Instance,
    device: &ash::Device,
    phys: vk::PhysicalDevice,
    size: vk::DeviceSize,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    create_buffer_and_memory(
        instance,
        device,
        phys,
        size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
}

fn create_camera_desc_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    let binding = vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::VERTEX,
        ..Default::default()
    };
    let ci = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        binding_count: 1,
        p_bindings: &binding,
        ..Default::default()
    };
    Ok(unsafe { device.create_descriptor_set_layout(&ci, None)? })
}

fn create_timeline_semaphore(device: &ash::Device, initial: u64) -> Result<vk::Semaphore> {
    let type_ci = vk::SemaphoreTypeCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_TYPE_CREATE_INFO,
        semaphore_type: vk::SemaphoreType::TIMELINE,
        initial_value: initial,
        ..Default::default()
    };
    let sem_ci = vk::SemaphoreCreateInfo {
        s_type: vk::StructureType::SEMAPHORE_CREATE_INFO,
        p_next: (&type_ci as *const _) as *const _,
        ..Default::default()
    };
    Ok(unsafe { device.create_semaphore(&sem_ci, None)? })
}

/// One-shot staging upload: host->staging, then staging->dst (device-local).
/// Uses the graphics queue and a one-time command buffer; waits until done.
fn upload_via_staging(
    instance: &ash::Instance,
    device: &ash::Device,
    phys: vk::PhysicalDevice,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    dst: vk::Buffer,
    src_data: &[u8],
) -> Result<()> {
    // 1) Create HOST_VISIBLE|COHERENT staging buffer
    let size = src_data.len() as vk::DeviceSize;
    let (staging, staging_mem) = create_buffer_and_memory(
        instance,
        device,
        phys,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;

    // Map + copy into staging
    let mapped = unsafe {
        std::slice::from_raw_parts_mut(
            device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())? as *mut u8,
            src_data.len(),
        )
    };
    mapped.copy_from_slice(src_data);
    unsafe { device.unmap_memory(staging_mem) };

    // 2) One-time copy staging -> dst
    let ai = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool: cmd_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    let cmd = unsafe { device.allocate_command_buffers(&ai)?[0] };
    let bi = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    unsafe { device.begin_command_buffer(cmd, &bi)? };
    let region = vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size,
    };
    unsafe { device.cmd_copy_buffer(cmd, staging, dst, std::slice::from_ref(&region)) };
    unsafe { device.end_command_buffer(cmd)? };

    // 3) Submit + wait
    let si = vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        ..Default::default()
    };
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
    unsafe { device.queue_submit(queue, std::slice::from_ref(&si), fence)? };
    unsafe { device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)? };
    unsafe { device.destroy_fence(fence, None) };

    // 4) Cleanup
    unsafe { device.free_command_buffers(cmd_pool, std::slice::from_ref(&cmd)) };
    unsafe { device.destroy_buffer(staging, None) };
    unsafe { device.free_memory(staging_mem, None) };
    Ok(())
}

fn create_sync_objects(
    device: &ash::Device,
    image_count: usize,
) -> Result<(Vec<AcquireSlot>, Vec<FrameSync>)> {
    let mut acq_slots = Vec::with_capacity(2);
    let mut frames = Vec::with_capacity(image_count);

    let sem_ci = vk::SemaphoreCreateInfo::default();

    // Two acquire slots (binary semaphores), tracked by timeline values
    for _ in 0..2 {
        let sem = unsafe { device.create_semaphore(&sem_ci, None)? };
        acq_slots.push(AcquireSlot {
            sem,
            last_signal_value: 0,
        });
    }

    // Per-image present wait semaphores (binary)
    for _ in 0..image_count {
        let rf = unsafe { device.create_semaphore(&sem_ci, None)? };
        frames.push(FrameSync {
            render_finished: rf,
        });
    }
    Ok((acq_slots, frames))
}

fn create_frame_uniforms_and_sets(
    instance: &ash::Instance,
    device: &ash::Device,
    phys: vk::PhysicalDevice,
    set_layout: vk::DescriptorSetLayout,
    image_count: usize,
) -> Result<FrameUniforms> {
    let limits = unsafe { instance.get_physical_device_properties(phys).limits };
    let a = limits.min_uniform_buffer_offset_alignment.max(1);
    let sz = std::mem::size_of::<CameraUbo>() as u64;
    let ubo_size = sz.div_ceil(a) * a;

    let mut ubufs = Vec::with_capacity(image_count);
    let mut umems = Vec::with_capacity(image_count);
    let mut ubo_ptrs = Vec::with_capacity(image_count);

    for _ in 0..image_count {
        let (b, m) = create_host_visible_ubo(instance, device, phys, ubo_size)?;
        let ptr = unsafe { device.map_memory(m, 0, ubo_size, vk::MemoryMapFlags::empty())? };
        ubufs.push(b);
        umems.push(m);
        ubo_ptrs.push(ptr);
    }

    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: image_count as u32,
    }];
    let pool_ci = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
        max_sets: image_count as u32,
        pool_size_count: 1,
        p_pool_sizes: pool_sizes.as_ptr(),
        ..Default::default()
    };
    let pool = unsafe { device.create_descriptor_pool(&pool_ci, None)? };

    let layouts = vec![set_layout; image_count];
    let alloc = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptor_pool: pool,
        descriptor_set_count: image_count as u32,
        p_set_layouts: layouts.as_ptr(),
        ..Default::default()
    };
    let sets = unsafe { device.allocate_descriptor_sets(&alloc)? };

    let mut writes = Vec::with_capacity(image_count);
    let mut infos: Vec<vk::DescriptorBufferInfo> = Vec::with_capacity(image_count);
    for i in 0..image_count {
        infos.push(vk::DescriptorBufferInfo {
            buffer: ubufs[i],
            offset: 0,
            range: ubo_size,
        });
        writes.push(vk::WriteDescriptorSet {
            s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
            dst_set: sets[i],
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            p_buffer_info: &infos[i],
            ..Default::default()
        });
    }
    unsafe { device.update_descriptor_sets(&writes, &[]) };

    Ok((ubufs, umems, ubo_ptrs, ubo_size, pool, sets))
}

fn recreate_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    surf_i: &surface::Instance,
    old_surface: &mut vk::SurfaceKHR,
    display_raw: RawDisplayHandle,
    window_raw: raw_window_handle::RawWindowHandle,
) -> Result<vk::SurfaceKHR> {
    let new_surface =
        unsafe { ash_window::create_surface(entry, instance, display_raw, window_raw, None) }
            .context("recreate_surface: ash_window::create_surface")?;
    if *old_surface != vk::SurfaceKHR::null() {
        unsafe { surf_i.destroy_surface(*old_surface, None) };
    }
    *old_surface = new_surface;
    Ok(new_surface)
}

fn make_initial_swapchain_resources(inp: &SwapchainInitInput) -> Result<SwapchainInit> {
    let bundle = create_swapchain_bundle(
        inp.device,
        inp.surf_i,
        inp.swap_d,
        inp.phys,
        inp.surface,
        vk::SwapchainKHR::null(),
        inp.cfg,
    )?;

    create_hdr_metadata_if_needed(
        inp.instance,
        inp.device,
        inp.has_hdr_meta,
        bundle.color_space,
        bundle.swapchain,
    );

    let image_count = bundle.image_views.len();
    let cmds = create_command_resources(inp.device, inp.queue_family, image_count)?;
    let pipe = create_pipeline(
        inp.device,
        inp.pipeline_cache,
        bundle.format,
        inp.depth_format,
        bundle.extent,
        inp.desc_set_layout_camera,
        inp.desc_set_layout_material,
    )?;
    let (acq, frames) = create_sync_objects(inp.device, image_count)?;
    Ok((bundle, cmds, pipe, acq, frames))
}

fn pick_device_and_queue(
    instance: &Instance,
    surf_i: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    let phys_devs = unsafe { instance.enumerate_physical_devices()? };

    for phys in phys_devs {
        let qprops = unsafe { instance.get_physical_device_queue_family_properties(phys) };

        for (i, q) in qprops.iter().enumerate() {
            if q.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                let supports_surface =
                    unsafe { surf_i.get_physical_device_surface_support(phys, i as u32, surface) }
                        .unwrap_or(false);

                if supports_surface {
                    return Ok((phys, i as u32));
                }
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

fn create_material_desc_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout> {
    // set = 1, binding = 0  (convention; set index is decided by pipeline layout order)
    let binding = vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        ..Default::default()
    };
    let ci = vk::DescriptorSetLayoutCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        binding_count: 1,
        p_bindings: &binding,
        ..Default::default()
    };
    Ok(unsafe { device.create_descriptor_set_layout(&ci, None)? })
}

fn create_image_and_memory(
    ctx: &DeviceCtx,
    info: &ImageAllocInfo,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let ci = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        image_type: vk::ImageType::TYPE_2D,
        format: info.format,
        extent: vk::Extent3D {
            width: info.extent.width,
            height: info.extent.height,
            depth: 1,
        },
        mip_levels: info.mip_levels,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: info.tiling,
        usage: info.usage,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };
    let image = unsafe { ctx.device.create_image(&ci, None) }.with_context(|| {
        format!(
            "create_image fmt={:?} extent={:?}",
            info.format, info.extent
        )
    })?;

    let req = unsafe { ctx.device.get_image_memory_requirements(image) };
    let mem_type_idx = find_memory_type(
        ctx.instance,
        ctx.phys,
        req.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let ai = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        allocation_size: req.size,
        memory_type_index: mem_type_idx,
        ..Default::default()
    };
    let mem = unsafe { ctx.device.allocate_memory(&ai, None) }
        .with_context(|| format!("allocate_memory (image) size={}", req.size))?;
    unsafe { ctx.device.bind_image_memory(image, mem, 0) }?;
    Ok((image, mem))
}

fn make_image_view_2d_color(
    device: &ash::Device,
    image: vk::Image,
    format: vk::Format,
    base_mip_level: u32,
    level_count: u32,
) -> Result<vk::ImageView> {
    let sub = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level,
        level_count,
        base_array_layer: 0,
        layer_count: 1,
    };
    let ci = vk::ImageViewCreateInfo {
        s_type: vk::StructureType::IMAGE_VIEW_CREATE_INFO,
        image,
        view_type: vk::ImageViewType::TYPE_2D,
        format,
        components: vk::ComponentMapping::default(),
        subresource_range: sub,
        ..Default::default()
    };
    Ok(unsafe { device.create_image_view(&ci, None)? })
}

// sync2 layout transition (generic helper)
fn transition_image_layout2(device: &ash::Device, cmd: vk::CommandBuffer, t: &LayoutTransition) {
    let b = vk::ImageMemoryBarrier2 {
        s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
        src_stage_mask: t.src_stage,
        src_access_mask: t.src_access,
        dst_stage_mask: t.dst_stage,
        dst_access_mask: t.dst_access,
        old_layout: t.old_layout,
        new_layout: t.new_layout,
        image: t.image,
        subresource_range: t.sub,
        ..Default::default()
    };
    let dep = vk::DependencyInfo {
        s_type: vk::StructureType::DEPENDENCY_INFO,
        image_memory_barrier_count: 1,
        p_image_memory_barriers: &b,
        ..Default::default()
    };
    unsafe { device.cmd_pipeline_barrier2(cmd, &dep) };
}

fn copy_buffer_to_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    buffer: vk::Buffer,
    image: vk::Image,
    extent: vk::Extent2D,
) {
    let sub = vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
    };
    let region = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,   // tightly packed
        buffer_image_height: 0, // tightly packed
        image_subresource: sub,
        image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        image_extent: vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        },
    };
    unsafe {
        device.cmd_copy_buffer_to_image(
            cmd,
            buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&region),
        )
    };
}

fn create_sampler(device: &ash::Device, mip_levels: u32) -> Result<vk::Sampler> {
    // No anisotropy yet (you didn’t enable it on device features). Safe defaults.
    let ci = vk::SamplerCreateInfo {
        s_type: vk::StructureType::SAMPLER_CREATE_INFO,
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        address_mode_u: vk::SamplerAddressMode::REPEAT,
        address_mode_v: vk::SamplerAddressMode::REPEAT,
        address_mode_w: vk::SamplerAddressMode::REPEAT,
        min_lod: 0.0,
        max_lod: mip_levels as f32,
        ..Default::default()
    };
    Ok(unsafe { device.create_sampler(&ci, None)? })
}

fn create_material_desc_pool_and_set(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<(vk::DescriptorPool, vk::DescriptorSet)> {
    let pool_sizes = [vk::DescriptorPoolSize {
        ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        descriptor_count: 1,
    }];
    let pool_ci = vk::DescriptorPoolCreateInfo {
        s_type: vk::StructureType::DESCRIPTOR_POOL_CREATE_INFO,
        max_sets: 1,
        pool_size_count: pool_sizes.len() as u32,
        p_pool_sizes: pool_sizes.as_ptr(),
        ..Default::default()
    };
    let pool = unsafe { device.create_descriptor_pool(&pool_ci, None)? };

    let alloc = vk::DescriptorSetAllocateInfo {
        s_type: vk::StructureType::DESCRIPTOR_SET_ALLOCATE_INFO,
        descriptor_pool: pool,
        descriptor_set_count: 1,
        p_set_layouts: &set_layout,
        ..Default::default()
    };
    let set = unsafe { device.allocate_descriptor_sets(&alloc)?[0] };
    Ok((pool, set))
}

fn write_material_descriptors(
    device: &ash::Device,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let image_info = vk::DescriptorImageInfo {
        sampler,
        image_view: view,
        image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    let write = vk::WriteDescriptorSet {
        s_type: vk::StructureType::WRITE_DESCRIPTOR_SET,
        dst_set: set,
        dst_binding: 0,
        descriptor_count: 1,
        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        p_image_info: &image_info,
        ..Default::default()
    };
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

fn create_dummy_texture_and_sampler(
    instance: &ash::Instance,
    device: &ash::Device,
    phys: vk::PhysicalDevice,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler)> {
    // 2x2 checkerboard RGBA
    let extent = vk::Extent2D {
        width: 2,
        height: 2,
    };
    let pixels: [u8; 16] = [
        255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255,
    ];

    // Create device-local image
    let ctx = DeviceCtx {
        instance,
        device,
        phys,
    };
    let info = ImageAllocInfo {
        extent,
        mip_levels: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        tiling: vk::ImageTiling::OPTIMAL,
    };
    let (image, memory) = create_image_and_memory(&ctx, &info)?;

    // Create staging buffer and copy pixels into it
    let size = pixels.len() as vk::DeviceSize;
    let (staging, staging_mem) = create_buffer_and_memory(
        instance,
        device,
        phys,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )?;
    unsafe {
        let mapped =
            device.map_memory(staging_mem, 0, size, vk::MemoryMapFlags::empty())? as *mut u8;
        std::ptr::copy_nonoverlapping(pixels.as_ptr(), mapped, pixels.len());
        device.unmap_memory(staging_mem);
    }

    // One-time command buffer to do the transitions + copy
    let ai = vk::CommandBufferAllocateInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_ALLOCATE_INFO,
        command_pool: cmd_pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    let cmd = unsafe { device.allocate_command_buffers(&ai)?[0] };
    let bi = vk::CommandBufferBeginInfo {
        s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    unsafe { device.begin_command_buffer(cmd, &bi)? };

    transition_color_to_transfer_dst(device, cmd, image, 1);
    copy_buffer_to_image(device, cmd, staging, image, extent);
    transition_color_to_shader_read(device, cmd, image, 1);

    unsafe { device.end_command_buffer(cmd)? };
    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
    let si = vk::SubmitInfo {
        s_type: vk::StructureType::SUBMIT_INFO,
        command_buffer_count: 1,
        p_command_buffers: &cmd,
        ..Default::default()
    };
    unsafe {
        device.queue_submit(queue, std::slice::from_ref(&si), fence)?;
        device.wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)?;
        device.destroy_fence(fence, None);
        device.free_command_buffers(cmd_pool, std::slice::from_ref(&cmd));
        device.destroy_buffer(staging, None);
        device.free_memory(staging_mem, None);
    }

    let view = make_image_view_2d_color(device, image, vk::Format::R8G8B8A8_UNORM, 0, 1)?;
    let sampler = create_sampler(device, 1)?;

    Ok((image, memory, view, sampler))
}
// END Helper functions

// 9) BIG BAD IMPORTANT STUFF
fn decide_path_and_create_device(
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
    let ext_props = unsafe {
        instance
            .enumerate_device_extension_properties(phys)
            .context("enumerate_device_extension_properties(device)")?
    };
    let has = unsafe {
        |name: &std::ffi::CStr| -> bool {
            ext_props
                .iter()
                .any(|e| std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) == name)
        }
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

    // Enable timeline semaphore
    feats12.timeline_semaphore = vk::TRUE;

    let (path, pnext): (RenderPath, *const std::ffi::c_void) = if !force_khr {
        let dev_api = unsafe { instance.get_physical_device_properties(phys).api_version };
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

    let device = unsafe {
        instance
            .create_device(phys, &dinfo, None)
            .context("create_device")?
    };

    let queue = unsafe { device.get_device_queue(queue_family, 0) };
    Ok((device, queue, path, has_hdr_meta))
}

fn create_swapchain_bundle(
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

fn create_pipeline(
    device: &ash::Device,
    cache: vk::PipelineCache,
    color_format: vk::Format,
    depth_format: vk::Format,
    _extent: vk::Extent2D,
    set_layout_camera: vk::DescriptorSetLayout,
    set_layout_material: vk::DescriptorSetLayout,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    // STRICT: color_attachment_formats MUST match current swapchain image format.
    // On swapchain format change, pipeline must be rebuilt before recording.

    // --- Load + create shader modules (destroyed before return) ---
    // Try CUBIC_SHADER_DIR override first (e.g., for mods or dev drops),
    // otherwise fall back to embedded SPIR-V from build.rs.
    let (vs_words, fs_words): (Vec<u32>, Vec<u32>) = {
        if let Ok(dir) = std::env::var("CUBIC_SHADER_DIR") {
            let vs_path = std::path::Path::new(&dir).join("tri.vert.spv");
            let fs_path = std::path::Path::new(&dir).join("tri.frag.spv");
            if vs_path.exists() && fs_path.exists() {
                (load_spv_file(&vs_path)?, load_spv_file(&fs_path)?)
            } else {
                let vs_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/tri.vert.spv"));
                let fs_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/tri.frag.spv"));
                (
                    load_spv_bytes(&vs_bytes[..])?,
                    load_spv_bytes(&fs_bytes[..])?,
                )
            }
        } else {
            let vs_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/tri.vert.spv"));
            let fs_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/tri.frag.spv"));
            (
                load_spv_bytes(&vs_bytes[..])?,
                load_spv_bytes(&fs_bytes[..])?,
            )
        }
    };

    let vs_ci = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_code: vs_words.as_ptr(),
        code_size: vs_words.len() * 4,
        ..Default::default()
    };
    let fs_ci = vk::ShaderModuleCreateInfo {
        s_type: vk::StructureType::SHADER_MODULE_CREATE_INFO,
        p_code: fs_words.as_ptr(),
        code_size: fs_words.len() * 4,
        ..Default::default()
    };
    let vs = unsafe { device.create_shader_module(&vs_ci, None)? };
    let fs = unsafe { device.create_shader_module(&fs_ci, None)? };
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
    // Vertex input layout: binding 0 with Vertex { pos, color }
    let vb = vk::VertexInputBindingDescription {
        binding: 0,
        stride: std::mem::size_of::<Vertex>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    };
    let va = [
        vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        },
        vk::VertexInputAttributeDescription {
            location: 1,
            binding: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: std::mem::size_of::<[f32; 3]>() as u32,
        },
        vk::VertexInputAttributeDescription {
            location: 2,
            binding: 0,
            format: vk::Format::R32G32_SFLOAT,
            offset: (std::mem::size_of::<[f32; 3]>() * 2) as u32,
        },
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        vertex_binding_description_count: 1,
        p_vertex_binding_descriptions: &vb,
        vertex_attribute_description_count: va.len() as u32,
        p_vertex_attribute_descriptions: va.as_ptr(),
        ..Default::default()
    };
    // Input assembly (triangles)
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        ..Default::default()
    };
    // Dynamic state
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
        depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL, // reverse-z
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
    let layouts = [set_layout_camera, set_layout_material];
    let layout_info = vk::PipelineLayoutCreateInfo {
        s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
        set_layout_count: layouts.len() as u32,
        p_set_layouts: layouts.as_ptr(),
        ..Default::default()
    };
    let layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

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
    let pipelines = unsafe {
        device.create_graphics_pipelines(cache, std::slice::from_ref(&pipeline_info), None)
    }
    .map_err(|(_, err)| anyhow!("create_graphics_pipelines failed: {:?}", err))?;

    unsafe {
        device.destroy_shader_module(vs, None);
        device.destroy_shader_module(fs, None);
    }

    Ok((layout, pipelines[0]))
}

fn build_renderer(
    window: &dyn HasWindowHandle,
    display: &dyn HasDisplayHandle,
    size: RenderSize,
) -> Result<VkRenderer> {
    // 1) Instance + surface (and record whether colorspace ext exists)
    #[cfg(debug_assertions)]
    let (entry, instance, surface_loader, surface, debug_state, have_swapchain_colorspace_ext) =
        init_instance_and_surface(window, display)?;
    #[cfg(not(debug_assertions))]
    let (entry, instance, surface_loader, surface, _debug_state, have_swapchain_colorspace_ext) =
        init_instance_and_surface(window, display)?;

    let display_raw = display
        .display_handle()
        .map_err(|e| anyhow!("{e}"))?
        .as_raw();
    let window_raw = window.window_handle().map_err(|e| anyhow!("{e}"))?.as_raw();

    // 2) Pick device/queue family
    let (phys, queue_family) = select_device_and_queue(&instance, &surface_loader, surface)?;

    // 3) Create device + choose render path, detect HDR metadata support
    let (device, queue, path, has_hdr_meta) =
        decide_path_and_create_device(&entry, &instance, phys, queue_family)?;
    let props = unsafe { instance.get_physical_device_properties(phys) };
    let cache_path = pipeline_cache_path(&props);
    let pipeline_cache = create_or_load_pipeline_cache(&device, &cache_path)?;

    // Create timeline semaphore
    let timeline = create_timeline_semaphore(&device, 0)?;
    let timeline_value: u64 = 0;

    // 4) WSI device wrapper
    let swapchain_loader = swapchain::Device::new(&instance, &device);

    // 5) Initial runtime knobs
    let initial_cfg = RuntimeConfig::from_env(have_swapchain_colorspace_ext);
    let cfg = initial_cfg.to_swapchain_config(size);
    #[cfg(debug_assertions)]
    let shader_dev = {
        if let Ok(dir) = std::env::var("CUBIC_SHADER_DIR") {
            let dir = PathBuf::from(dir);
            let vp = dir.join("tri.vert.spv");
            let fp = dir.join("tri.frag.spv");
            if vp.exists() && fp.exists() {
                if let (Ok(vm), Ok(fm)) = (
                    fs::metadata(&vp).and_then(|m| m.modified()),
                    fs::metadata(&fp).and_then(|m| m.modified()),
                ) {
                    Some(ShaderDev {
                        vert_spv: vp,
                        frag_spv: fp,
                        vert_mtime: vm,
                        frag_mtime: fm,
                    })
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        }
    };

    // Create depth buffers
    let depth_format = pick_depth_format(&instance, phys);
    let desc_set_layout_camera = create_camera_desc_set_layout(&device)?;
    let desc_set_layout_material = create_material_desc_set_layout(&device)?;

    // 6) Build all swapchain-scoped resources in one place
    let init_inp = SwapchainInitInput {
        device: &device,
        instance: &instance,
        surf_i: &surface_loader,
        swap_d: &swapchain_loader,
        phys,
        surface,
        cfg,
        queue_family,
        has_hdr_meta,
        pipeline_cache,
        depth_format,
        desc_set_layout_camera,
        desc_set_layout_material,
    };
    let (sc, cmd, (pipeline_layout, pipeline), acq_slots, frames) =
        make_initial_swapchain_resources(&init_inp)?;
    let (depth_image, depth_mem, depth_view) =
        create_depth_resources(&instance, &device, phys, sc.extent, depth_format)?;

    // Global material set (swapchain-invariant)
    let (material_desc_pool, material_desc_set) =
        create_material_desc_pool_and_set(&device, desc_set_layout_material)?;

    // Tiny 2×2 texture and sampler, then write the descriptor
    let (tex_image, tex_mem, tex_view, tex_sampler) =
        create_dummy_texture_and_sampler(&instance, &device, phys, queue, cmd.pool)?;
    write_material_descriptors(&device, material_desc_set, tex_view, tex_sampler);

    let (ubufs, umems, ubo_ptrs, ubo_size, desc_pool, desc_sets) = create_frame_uniforms_and_sets(
        &instance,
        &device,
        phys,
        desc_set_layout_camera,
        sc.image_views.len(),
    )?;

    // --- Create device-local vertex/index buffers and upload data ---
    let vsize = std::mem::size_of_val(TRI_VERTS) as vk::DeviceSize;
    let isize = std::mem::size_of_val(TRI_IDXS) as vk::DeviceSize;

    // Create destination (device-local) buffers
    let (vbuf, vmem) = create_buffer_and_memory(
        &instance,
        &device,
        phys,
        vsize,
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    let (ibuf, imem) = create_buffer_and_memory(
        &instance,
        &device,
        phys,
        isize,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    // Upload via staging
    let vbytes = bytemuck::cast_slice(TRI_VERTS);
    let ibytes = bytemuck::cast_slice(TRI_IDXS);

    upload_via_staging(&instance, &device, phys, queue, cmd.pool, vbuf, vbytes)?;
    upload_via_staging(&instance, &device, phys, queue, cmd.pool, ibuf, ibytes)?;

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
        vbuf,
        vbuf_mem: vmem,
        ibuf,
        ibuf_mem: imem,
        index_count: TRI_IDXS.len() as u32,
        desc_pool,
        desc_set_layout_camera,
        desc_set_layout_material,
        desc_sets,
        ubufs,
        umems,
        ubo_ptrs,
        ubo_size,
        pipeline_cache,
        timeline,
        timeline_value,
        display_raw,
        window_raw,
        backoff_frames: 0,
        #[cfg(debug_assertions)]
        shader_dev,
        material_desc_pool,
        material_desc_set,
        tex_image,
        tex_mem,
        tex_view,
        tex_sampler,
    };

    // 8) Record per-image command buffers once
    r.record_commands()?;

    Ok(r)
}

impl VkRenderer {
    /// RH camera, forward = -Z, Vulkan ZO (0..1), reverse-Z, infinite far plane.
    /// `flip_y` should be false while you're using a negative viewport height.
    fn perspective_rh_zo_reverse_infinite(
        fovy: f32,
        aspect: f32,
        near: f32,
        flip_y: bool,
    ) -> [[f32; 4]; 4] {
        let f = 1.0 / (0.5 * fovy).tan();
        let c0 = [f / aspect, 0.0, 0.0, 0.0];
        let mut c1 = [0.0, f, 0.0, 0.0];
        let c2 = [0.0, 0.0, 0.0, -1.0];
        let c3 = [0.0, 0.0, near, 0.0];
        if flip_y {
            c1[1] = -c1[1];
        }
        [c0, c1, c2, c3] // columns
    }

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
        let _ = self.recreate_swapchain(want);
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
        let _ = self.recreate_swapchain(want);
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
        let _ = self.recreate_swapchain(want);
    }

    #[inline]
    fn should_skip_for_backoff(&mut self) -> bool {
        if self.backoff_frames > 0 {
            self.backoff_frames -= 1;
            true
        } else {
            false
        }
    }

    #[cfg(debug_assertions)]
    fn hot_reload_shaders_if_changed(&mut self) -> Result<()> {
        let Some(dev) = self.shader_dev.as_mut() else {
            return Ok(());
        };

        let vm = fs::metadata(&dev.vert_spv).and_then(|m| m.modified()).ok();
        let fm = fs::metadata(&dev.frag_spv).and_then(|m| m.modified()).ok();

        let vert_changed = vm.is_some() && vm.unwrap() > dev.vert_mtime;
        let frag_changed = fm.is_some() && fm.unwrap() > dev.frag_mtime;

        if !(vert_changed || frag_changed) {
            return Ok(());
        }

        tracing::info!("vk: .spv change detected → rebuilding pipeline");

        // Update mtimes first to avoid tight loops if rebuild fails.
        if let Some(t) = vm {
            dev.vert_mtime = t;
        }
        if let Some(t) = fm {
            dev.frag_mtime = t;
        }

        // Ensure no in-flight use of old pipeline while swapping.
        unsafe {
            self.device.device_wait_idle().ok();
        }

        // Rebuild using the same loader (it prefers CUBIC_SHADER_DIR/*.spv if present)
        let (new_layout, new_pipeline) = create_pipeline(
            &self.device,
            self.pipeline_cache,
            self.format,
            self.depth_format,
            self.extent,
            self.desc_set_layout_camera,
            self.desc_set_layout_material,
        )?;

        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
        self.pipeline_layout = new_layout;
        self.pipeline = new_pipeline;

        // Re-record CBs because pipeline handle changed.
        self.record_commands()?;
        Ok(())
    }

    fn update_camera_ubo_for_image(
        &self,
        image_index: usize,
        data: &CameraUbo,
    ) -> anyhow::Result<()> {
        let dst = self.ubo_ptrs[image_index];
        if dst.is_null() {
            return Err(anyhow::anyhow!("UBO memory not mapped"));
        }
        let src = bytemuck::bytes_of(data);

        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst as *mut u8, src.len());
        }
        Ok(())
    }

    #[inline]
    fn transition_to_color(&self, cmd: vk::CommandBuffer, image: vk::Image) {
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
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep_pre) };
    }

    #[inline]
    fn transition_depth_to_attachment(&self, cmd: vk::CommandBuffer, image: vk::Image) {
        let subrange = vk::ImageSubresourceRange {
            aspect_mask: depth_aspect_mask(self.depth_format),
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        };

        let pre = vk::ImageMemoryBarrier2 {
            s_type: vk::StructureType::IMAGE_MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::TOP_OF_PIPE,
            src_access_mask: vk::AccessFlags2::empty(),
            dst_stage_mask: vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            dst_access_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: depth_attachment_layout(self.depth_format),
            image,
            subresource_range: subrange,
            ..Default::default()
        };
        let dep = vk::DependencyInfo {
            s_type: vk::StructureType::DEPENDENCY_INFO,
            image_memory_barrier_count: 1,
            p_image_memory_barriers: &pre,
            ..Default::default()
        };
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep) };
    }

    #[inline]
    fn begin_rendering(&self, cmd: vk::CommandBuffer, image_view: vk::ImageView) {
        let color_att = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view,
            image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            clear_value: self.clear,
            ..Default::default()
        };

        let depth_att = vk::RenderingAttachmentInfo {
            s_type: vk::StructureType::RENDERING_ATTACHMENT_INFO,
            image_view: self.depth_view,
            image_layout: depth_attachment_layout(self.depth_format),
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            clear_value: vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            },
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
            p_depth_attachment: &depth_att,
            ..Default::default()
        };

        unsafe { self.device.cmd_begin_rendering(cmd, &rendering_info) };
    }

    #[inline]
    fn bind_draw_geometry(&self, cmd: vk::CommandBuffer, image_index: usize) -> Result<()> {
        if self.pipeline == vk::Pipeline::null() {
            return Err(anyhow!("pipeline is VK_NULL_HANDLE at record time"));
        }

        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline)
        };

        // dynamic viewport/scissor
        let vp = vk::Viewport {
            // Try positive flip for 3D
            x: 0.0,
            y: self.extent.height as f32, //0
            width: self.extent.width as f32,
            height: -(self.extent.height as f32), //self.extent.height as f32
            min_depth: 0.0,
            max_depth: 1.0,
        };
        unsafe {
            self.device
                .cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp))
        };
        let sc = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };
        unsafe {
            self.device
                .cmd_set_scissor(cmd, 0, std::slice::from_ref(&sc))
        };

        // Bind per-image descriptor set (set = 0)
        let set = [self.desc_sets[image_index], self.material_desc_set];
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0, // firstSet -> set 0 = camera, set 1 = material
                &set,
                &[], // no dynamic offsets
            );
        }

        // bind vertex + index buffers
        let offsets = [0_u64];
        unsafe {
            self.device
                .cmd_bind_vertex_buffers(cmd, 0, std::slice::from_ref(&self.vbuf), &offsets);
            self.device
                .cmd_bind_index_buffer(cmd, self.ibuf, 0, vk::IndexType::UINT32);

            self.device
                .cmd_draw_indexed(cmd, self.index_count, 1, 0, 0, 0)
        };
        Ok(())
    }

    #[inline]
    fn transition_to_present(&self, cmd: vk::CommandBuffer, image: vk::Image) {
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
            dst_stage_mask: vk::PipelineStageFlags2::NONE,
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
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep_post) };
    }

    #[inline]
    fn record_one_command(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        image_view: vk::ImageView,
        image_index: usize,
    ) -> Result<()> {
        // reset + begin
        unsafe {
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?
        };
        let begin = vk::CommandBufferBeginInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_BEGIN_INFO,
            ..Default::default()
        };
        unsafe { self.device.begin_command_buffer(cmd, &begin)? };

        // body
        self.transition_to_color(cmd, image);
        self.transition_depth_to_attachment(cmd, self.depth_image);
        self.begin_rendering(cmd, image_view);
        self.bind_draw_geometry(cmd, image_index)?;
        unsafe { self.device.cmd_end_rendering(cmd) };
        self.transition_to_present(cmd, image);

        // end
        unsafe { self.device.end_command_buffer(cmd)? };
        Ok(())
    }

    // --- Record all per-swapchain-image CBs ----------------------
    fn record_commands(&mut self) -> Result<()> {
        for (i, &cmd) in self.cmd_bufs.iter().enumerate() {
            self.record_one_command(cmd, self.images[i], self.image_views[i], i)?;
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
    fn recreate_swapchain(&mut self, size: RenderSize) -> Result<()> {
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

        // 3b) Destroy per-image UBOs + descriptor pool tied to OLD swapchain
        for (i, &m) in self.umems.iter().enumerate() {
            let p = self
                .ubo_ptrs
                .get(i)
                .copied()
                .unwrap_or(std::ptr::null_mut());
            if !p.is_null() {
                unsafe { self.device.unmap_memory(m) };
            }
        }
        for &b in &self.ubufs {
            unsafe { self.device.destroy_buffer(b, None) };
        }
        for &m in &self.umems {
            unsafe { self.device.free_memory(m, None) };
        }
        self.ubufs.clear();
        self.umems.clear();
        self.ubo_ptrs.clear();
        self.ubo_size = 0;

        if self.desc_pool != vk::DescriptorPool::null() {
            unsafe { self.device.destroy_descriptor_pool(self.desc_pool, None) };
            self.desc_pool = vk::DescriptorPool::null();
        }
        self.desc_sets.clear();

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
            unsafe { self.device.destroy_image_view(self.depth_view, None) };
        }
        if self.depth_image != vk::Image::null() {
            unsafe { self.device.destroy_image(self.depth_image, None) };
        }
        if self.depth_mem != vk::DeviceMemory::null() {
            unsafe { self.device.free_memory(self.depth_mem, None) };
        }
        let (dimg, dmem, dview) = create_depth_resources(
            &self.instance,
            &self.device,
            self.phys,
            self.extent,
            self.depth_format,
        )?;
        self.depth_image = dimg;
        self.depth_mem = dmem;
        self.depth_view = dview;

        // 5) Recreate per-image UBOs + descriptor sets
        let (ubufs, umems, ubo_ptrs, ubo_size, desc_pool, desc_sets) =
            create_frame_uniforms_and_sets(
                &self.instance,
                &self.device,
                self.phys,
                self.desc_set_layout_camera,
                self.images.len(),
            )?;
        self.ubufs = ubufs;
        self.umems = umems;
        self.ubo_ptrs = ubo_ptrs;
        self.ubo_size = ubo_size;
        self.desc_pool = desc_pool;
        self.desc_sets = desc_sets;

        // 5b) Recreate per-image sync
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
                self.format,
                self.depth_format, // ensure dynamic rendering knows the depth format
                self.extent,
                self.desc_set_layout_camera,
                self.desc_set_layout_material,
            )?;
            unsafe { self.device.destroy_pipeline(self.pipeline, None) };
            unsafe {
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None)
            };
            self.pipeline_layout = new_layout;
            self.pipeline = new_pipeline;
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
        build_renderer(window, display, size)
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
        let _ = self.recreate_swapchain(want);
    }

    fn resize(&mut self, size: RenderSize) -> Result<()> {
        // Handle minimized / 0×0 and pause
        if size.width == 0 || size.height == 0 {
            if !self.paused {
                info!("vk: resize to 0x0 → paused=true");
            }
            self.paused = true;
            return Ok(());
        }

        // Coming back from pause
        if self.paused {
            info!(
                "vk: resize to {}x{} → paused=false",
                size.width, size.height
            );
        }
        self.paused = false;

        // Try to recreate the swapchain; if the surface was lost, rebuild it once and retry
        match self.recreate_swapchain(size) {
            Ok(()) => Ok(()),
            Err(e) => {
                // If we can peel out a vk::Result and it's SURFACE_LOST, rebuild the surface
                if let Some(vkerr) = e.downcast_ref::<vk::Result>() {
                    if *vkerr == vk::Result::ERROR_SURFACE_LOST_KHR {
                        let entry = Entry::linked();
                        // requires: self.display_raw / self.window_raw fields and recreate_surface() helper
                        recreate_surface(
                            &entry,
                            &self.instance,
                            &self.surface_loader,
                            &mut self.surface,
                            self.display_raw,
                            self.window_raw,
                        )?;
                        // retry swapchain on the new surface
                        return self.recreate_swapchain(size);
                    }
                }
                Err(e)
            }
        }
    }

    fn set_clear_color(&mut self, rgba: [f32; 4]) {
        self.clear = vk::ClearValue {
            color: vk::ClearColorValue { float32: rgba },
        };

        let _ = self.record_commands();
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
        // Backoff check
        if self.should_skip_for_backoff() {
            return Ok(());
        }
        #[cfg(debug_assertions)]
        self.hot_reload_shaders_if_changed()?;

        // Query caps
        let caps = match unsafe {
            self.surface_loader
                .get_physical_device_surface_capabilities(self.phys, self.surface)
        } {
            Ok(caps) => caps,
            Err(e) => {
                if !self.paused {
                    self.paused = true;
                    info!("vk: surface caps error {:?} → paused=true", e);
                }
                return Ok(());
            }
        };
        if caps.current_extent.width == 0 || caps.current_extent.height == 0 {
            if !self.paused {
                self.paused = true;
                info!("vk: current_extent is 0x0 → paused=true");
            }
            return Ok(());
        }

        // 1) Acquire
        let s = &self.acq_slots[self.acq_index];
        if s.last_signal_value > 0 {
            let wait_info = vk::SemaphoreWaitInfo {
                s_type: vk::StructureType::SEMAPHORE_WAIT_INFO,
                flags: vk::SemaphoreWaitFlags::empty(),
                semaphore_count: 1,
                p_semaphores: &self.timeline,
                p_values: &s.last_signal_value,
                ..Default::default()
            };
            unsafe {
                self.device.wait_semaphores(&wait_info, u64::MAX)?;
            }
        }

        let (image_index, _) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                s.sem,
                vk::Fence::null(),
            )
        } {
            // same match arms…
            Ok(pair) => pair,
            Err(e) if is_swapchain_out_of_date(e) => {
                self.backoff_frames = 2;
                let want = RenderSize {
                    width: caps.current_extent.width,
                    height: caps.current_extent.height,
                };
                let _ = self.recreate_swapchain(want);
                return Ok(());
            }
            Err(e) if is_surface_lost(e) => {
                self.backoff_frames = 2;
                let entry = Entry::linked();
                if recreate_surface(
                    &entry,
                    &self.instance,
                    &self.surface_loader,
                    &mut self.surface,
                    self.display_raw,
                    self.window_raw,
                )
                .is_ok()
                {
                    let want = RenderSize {
                        width: caps.current_extent.width,
                        height: caps.current_extent.height,
                    };
                    let _ = self.recreate_swapchain(want);
                } else {
                    self.paused = true;
                }
                return Ok(());
            }
            Err(e) if is_device_lost(e) => return Err(anyhow!("vk: device lost during acquire")),
            Err(e) => return Err(anyhow!("acquire_next_image: {e:?}")),
        };

        let img = image_index as usize;
        let f_img = &self.frames[img];
        let cmd = self.cmd_bufs[img];
        let aspect = self.extent.width as f32 / self.extent.height as f32;
        let fovy = std::f32::consts::FRAC_PI_3; // 60°
        let near = 0.1_f32; // 0.05–0.5 is a good range
        let flip_y = false; // you're using a negative viewport height right now
        let proj = VkRenderer::perspective_rh_zo_reverse_infinite(fovy, aspect, near, flip_y);
        let mvp = CameraUbo { mvp: proj };
        self.update_camera_ubo_for_image(img, &mvp)?;

        // 2) Submit (wait on acquire sem; signal render-finished; bump timeline)
        let next_value = self.timeline_value.wrapping_add(1);

        let stage_color = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let stage2_color = stage_flags2_from_legacy(stage_color);

        // Build the semaphore infos
        let wait_acquire = semaphore_submit_info_wait(s.sem, 0, stage2_color);
        let signal_present = semaphore_submit_info_signal(f_img.render_finished, 0, stage2_color);
        let signal_timeline = semaphore_submit_info_signal(self.timeline, next_value, stage2_color);

        // IMPORTANT: store in locals so the pointers in SubmitInfo2 stay valid
        let waits = [wait_acquire];
        let signals = [signal_present, signal_timeline];

        let cmd_info = vk::CommandBufferSubmitInfo {
            s_type: vk::StructureType::COMMAND_BUFFER_SUBMIT_INFO,
            command_buffer: cmd,
            device_mask: 0,
            ..Default::default()
        };

        let submit2 = vk::SubmitInfo2 {
            s_type: vk::StructureType::SUBMIT_INFO_2,
            wait_semaphore_info_count: waits.len() as u32,
            p_wait_semaphore_infos: waits.as_ptr(),
            command_buffer_info_count: 1,
            p_command_buffer_infos: &cmd_info,
            signal_semaphore_info_count: signals.len() as u32,
            p_signal_semaphore_infos: signals.as_ptr(),
            ..Default::default()
        };

        // Submit with robust error handling
        let submit_res = unsafe {
            self.device.queue_submit2(
                self.queue,
                std::slice::from_ref(&submit2),
                vk::Fence::null(),
            )
        };

        match submit_res {
            Ok(()) => {
                self.timeline_value = next_value;
                self.acq_slots[self.acq_index].last_signal_value = next_value;
            }
            Err(vk::Result::ERROR_DEVICE_LOST) => {
                return Err(anyhow!("vk: device lost during submit"));
            }
            Err(e) => {
                return Err(anyhow!("queue_submit2: {e:?}"));
            }
        }

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

        match unsafe { self.swapchain_loader.queue_present(self.queue, &present) } {
            Ok(_) => {}
            Err(e) if is_swapchain_out_of_date(e) => {
                self.backoff_frames = 2;
                let want = RenderSize {
                    width: caps.current_extent.width,
                    height: caps.current_extent.height,
                };
                let _ = self.recreate_swapchain(want);
                return Ok(());
            }
            Err(e) if is_surface_lost(e) => {
                self.backoff_frames = 2;
                let entry = Entry::linked();
                if recreate_surface(
                    &entry,
                    &self.instance,
                    &self.surface_loader,
                    &mut self.surface,
                    self.display_raw,
                    self.window_raw,
                )
                .is_ok()
                {
                    let want = RenderSize {
                        width: caps.current_extent.width,
                        height: caps.current_extent.height,
                    };
                    let _ = self.recreate_swapchain(want);
                } else {
                    self.paused = true;
                }
                return Ok(());
            }
            Err(e) if is_device_lost(e) => return Err(anyhow!("vk: device lost during present")),
            Err(e) => return Err(anyhow!("queue_present: {e:?}")),
        }

        // Rotate acquire slot
        self.acq_index = (self.acq_index + 1) % self.acq_slots.len();

        Ok(())
    }
}
