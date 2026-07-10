// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

mod device;
mod egui_overlay;
mod frame;
mod instance;
mod pipeline;
mod resources;
mod swapchain;
mod sync;

use anyhow::{anyhow, Result};
use ash::khr::surface;
use ash::{vk, Entry};
use cubic_math::Camera;
use cubic_render::{RenderSize, Renderer};
use gpu_allocator::vulkan::{Allocation, Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use tracing::info;

use device::{decide_path_and_create_device, select_device_and_queue, RenderPath};
#[cfg(debug_assertions)]
use instance::destroy_debug_messenger;
use instance::{init_instance_and_surface, recreate_surface};
#[cfg(debug_assertions)]
use pipeline::ShaderDev;
use pipeline::{
    create_compute_pipeline, create_or_load_pipeline_cache, create_pipeline, load_spv_file,
    pipeline_cache_path, save_pipeline_cache, shader_dir, PipelineConfig,
};
use resources::{
    create_buffer_and_memory, create_camera_desc_set_layout, create_depth_resources,
    create_dummy_texture_and_sampler, create_frame_uniforms_and_sets,
    create_indirect_compute_desc_set_layout, create_indirect_draw_resources,
    create_indirect_graphics_desc_set_layout, create_material_desc_pool_and_set,
    create_material_desc_set_layout, pick_depth_format, upload_via_staging,
    write_material_descriptors, RangeAlloc, SamplerConfig, MAX_SHARED_INDICES, MAX_SHARED_VERTICES,
};
// Vertex, PushData, and MeshHandle are now defined in cubic-render so that
// cubic-world can use them without depending on Vulkan. Re-export them from
// here so existing callers (cubic-app etc.) import from cubic-render-vk
// without any changes.
pub use cubic_render::{MeshHandle, PushData, Vertex};
use swapchain::{
    create_hdr_metadata_if_needed, create_swapchain_bundle, SwapchainBundle, SwapchainConfig,
};
pub use swapchain::{HdrFlavor, VkVsyncMode};
// Re-exported so callers (cubic-app's set_sampler_config plumbing) can build
// sampler settings without depending on `ash` directly. These two are plain,
// trivially-constructible enums (unlike e.g. vsync/HDR, which need custom
// wrapper types for fallback logic), so re-exporting as-is is simplest.
pub use ash::vk::{Filter, SamplerMipmapMode};
use sync::{
    create_command_resources, create_sync_objects, create_timeline_semaphore, AcquireSlot,
    CommandResources, FrameSync,
};

/// Offsets into the shared vertex/index buffers (see
/// `MAX_SHARED_VERTICES`/`MAX_SHARED_INDICES`) rather than owning dedicated
/// buffers: one `cmd_draw_indexed_indirect_count` call can only bind a
/// single vertex/index buffer pair, so every mesh that might be drawn
/// together in one indirect call has to live in the same buffers.
struct GpuMesh {
    first_vertex: i32,
    first_index: u32,
    index_count: u32,
    vertex_count: u32,
}

/// A GPU object retired while it might still be in use, destroyed once the
/// timeline semaphore reaches `value` (see `VkRenderer::drain_trash`).
enum GpuResource {
    Buffer {
        buffer: vk::Buffer,
        alloc: Allocation,
    },
    Image {
        image: vk::Image,
        alloc: Allocation,
    },
    ImageView(vk::ImageView),
    Pipeline(vk::Pipeline),
    PipelineLayout(vk::PipelineLayout),
    MeshSlot {
        first_vertex: u32,
        vertex_count: u32,
        first_index: u32,
        index_count: u32,
    },
}

struct DeferredDrop {
    value: u64,
    resource: GpuResource,
}

// 3) Renderer data model
pub struct VkRenderer {
    instance: ash::Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,

    phys: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    // Option so Drop can `.take()` it and drop it explicitly before the
    // device is destroyed (Allocator::drop frees any remaining cached
    // memory blocks via its own device handle).
    allocator: Option<Allocator>,

    swapchain_loader: ash::khr::swapchain::Device,
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
    camera: Camera,

    depth_image: vk::Image,
    depth_alloc: Allocation,
    depth_view: vk::ImageView,
    depth_format: vk::Format,
    // Shared by every mesh (see GpuMesh); bump-allocated, never freed
    // individually since there's no free_mesh API yet.
    shared_vbuf: vk::Buffer,
    shared_vbuf_alloc: Allocation,
    shared_ibuf: vk::Buffer,
    shared_ibuf_alloc: Allocation,
    vert_alloc: RangeAlloc,
    idx_alloc: RangeAlloc,
    meshes: Vec<GpuMesh>,
    // Draws queued by draw_mesh() for the next render() call; consumed and
    // cleared each time a frame's command buffer is recorded.
    pending_draws: Vec<(MeshHandle, PushData)>,
    // GPU resources retired while possibly still in use; reclaimed once the
    // timeline semaphore catches up (see drain_trash).
    trash: Vec<DeferredDrop>,
    desc_pool: vk::DescriptorPool,
    desc_set_layout_camera: vk::DescriptorSetLayout,
    desc_set_layout_material: vk::DescriptorSetLayout,
    // Graphics-side read-only view of the candidates buffer (set = 2); see
    // indirect_compute_desc_set_layout for the compute-side write access.
    desc_set_layout_indirect_graphics: vk::DescriptorSetLayout,
    desc_set_layout_indirect_compute: vk::DescriptorSetLayout,
    desc_sets: Vec<vk::DescriptorSet>,
    ubufs: Vec<vk::Buffer>,
    umems: Vec<Allocation>,
    ubo_ptrs: Vec<*mut std::ffi::c_void>,
    ubo_size: vk::DeviceSize,
    // GPU-driven indirect draw path: per-image candidate/indirect-command/
    // draw-count buffers + descriptor sets (see resources::IndirectDrawResources).
    indirect_cull_pipeline: vk::Pipeline,
    indirect_cull_pipeline_layout: vk::PipelineLayout,
    candidate_bufs: Vec<vk::Buffer>,
    candidate_allocs: Vec<Allocation>,
    candidate_ptrs: Vec<*mut std::ffi::c_void>,
    indirect_bufs: Vec<vk::Buffer>,
    indirect_allocs: Vec<Allocation>,
    draw_count_bufs: Vec<vk::Buffer>,
    draw_count_allocs: Vec<Allocation>,
    indirect_desc_pool: vk::DescriptorPool,
    indirect_compute_desc_sets: Vec<vk::DescriptorSet>,
    indirect_graphics_desc_sets: Vec<vk::DescriptorSet>,
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
    tex_alloc: Allocation,
    tex_view: vk::ImageView,
    tex_sampler: vk::Sampler,
    // Bindless texture array bookkeeping for upload_texture(). Index 0 is
    // permanently the dummy texture above; uploads start at 1.
    next_tex_index: u32,
    tex_store: Vec<(vk::Image, Allocation, vk::ImageView, vk::Sampler)>,
    // Filter/mipmap/anisotropy settings applied to every texture uploaded
    // via upload_texture(). Starts at a sensible default (used for the
    // dummy texture, created before cubic-app's configure_advanced() can
    // run); set_sampler_config() overrides it with the real cubic.toml
    // values immediately after construction, before any real textures load.
    sampler_config: SamplerConfig,

    // egui overlay support (GPU plumbing only — no egui::Context or input
    // handling here; that lives in cubic-app). Option because it's created
    // after the swapchain exists, not at struct-literal time.
    egui_renderer: Option<egui_ash_renderer::Renderer>,
    // Staged by queue_egui(), consumed by the next render() call.
    egui_pending: Option<egui_overlay::EguiFrame>,
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
        }

        // Drop the egui overlay renderer before the device: it owns a
        // pipeline, descriptor pool/layout, and any managed textures
        // (allocated via its own private gpu-allocator instance), all of
        // which must be torn down against a still-valid device.
        self.egui_renderer.take();

        // Device is fully idle, so every trashed resource is now safe to
        // destroy regardless of its retirement value.
        self.drain_trash();

        unsafe {
            let d = &self.device;

            // 3) PIPELINE & LAYOUTS BEFORE SWAPCHAIN (pipelines can depend on sc format)
            d.destroy_pipeline(self.pipeline, None);
            d.destroy_pipeline_layout(self.pipeline_layout, None);
            d.destroy_pipeline(self.indirect_cull_pipeline, None);
            d.destroy_pipeline_layout(self.indirect_cull_pipeline_layout, None);

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

            // Take ownership of the allocator so every allocation can be
            // freed below, and so it can be explicitly dropped before the
            // device is destroyed (Allocator::drop frees any remaining
            // cached memory blocks via its own device handle, which must
            // still be valid at that point).
            let mut allocator = self.allocator.take().expect("allocator missing in Drop");

            // Destroy depth
            d.destroy_image_view(self.depth_view, None);
            d.destroy_image(self.depth_image, None);
            let _ = allocator.free(std::mem::take(&mut self.depth_alloc));

            // Destroy the shared vertex/index buffers every upload_mesh call
            // bump-allocates from (meshes themselves own no buffers).
            self.meshes.clear();
            d.destroy_buffer(self.shared_vbuf, None);
            d.destroy_buffer(self.shared_ibuf, None);
            let _ = allocator.free(std::mem::take(&mut self.shared_vbuf_alloc));
            let _ = allocator.free(std::mem::take(&mut self.shared_ibuf_alloc));

            // Destroy GPU-driven indirect draw resources
            for &b in &self.candidate_bufs {
                d.destroy_buffer(b, None);
            }
            for alloc in self.candidate_allocs.drain(..) {
                let _ = allocator.free(alloc);
            }
            for &b in &self.indirect_bufs {
                d.destroy_buffer(b, None);
            }
            for alloc in self.indirect_allocs.drain(..) {
                let _ = allocator.free(alloc);
            }
            for &b in &self.draw_count_bufs {
                d.destroy_buffer(b, None);
            }
            for alloc in self.draw_count_allocs.drain(..) {
                let _ = allocator.free(alloc);
            }
            d.destroy_descriptor_pool(self.indirect_desc_pool, None);
            d.destroy_descriptor_set_layout(self.desc_set_layout_indirect_compute, None);
            d.destroy_descriptor_set_layout(self.desc_set_layout_indirect_graphics, None);

            // Destroy frame resources (gpu-allocator persistently maps
            // CpuToGpu allocations, so no explicit unmap is needed)
            for &b in &self.ubufs {
                self.device.destroy_buffer(b, None);
            }
            for alloc in self.umems.drain(..) {
                let _ = allocator.free(alloc);
            }
            self.ubufs.clear();
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
            let _ = allocator.free(std::mem::take(&mut self.tex_alloc));

            // Uploaded textures (upload_texture)
            for (image, alloc, view, sampler) in self.tex_store.drain(..) {
                d.destroy_sampler(sampler, None);
                d.destroy_image_view(view, None);
                d.destroy_image(image, None);
                let _ = allocator.free(alloc);
            }

            // Save and destroy pipeline cache
            let props = self.instance.get_physical_device_properties(self.phys);
            let cache_path = pipeline_cache_path(&props);
            let _ = save_pipeline_cache(d, self.pipeline_cache, &cache_path);
            d.destroy_pipeline_cache(self.pipeline_cache, None);

            // Explicitly drop the allocator now, while the device is still
            // alive, and before destroying the device.
            drop(allocator);

            // 8) DESTROY DEVICE, THEN SURFACE, THEN INSTANCE
            d.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}

// 4) Structs (remaining)
#[derive(Clone, Copy)]
struct SwapchainInitInput<'a> {
    device: &'a ash::Device,
    instance: &'a ash::Instance,
    surf_i: &'a surface::Instance,
    swap_d: &'a ash::khr::swapchain::Device,
    phys: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
    cfg: SwapchainConfig,
    queue_family: u32,
    has_hdr_meta: bool,
    pipeline_cache: vk::PipelineCache,
    pipeline_cfg: PipelineConfig,
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

    /// Convert to the swapchain's creation config for a given target size.
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

// 6) Types
type SwapchainInit = (
    SwapchainBundle,
    CommandResources,
    (vk::PipelineLayout, vk::Pipeline),
    Vec<AcquireSlot>,
    Vec<FrameSync>,
);

// 7) Inline helper functions
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

// 8) Orchestration helpers
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
        &PipelineConfig {
            color_format: bundle.format,
            ..inp.pipeline_cfg
        },
    )?;
    let (acq, frames) = create_sync_objects(inp.device, image_count)?;
    Ok((bundle, cmds, pipe, acq, frames))
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

    // 3b) GPU memory sub-allocator, replaces raw vkAllocateMemory/vkFreeMemory
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device: phys,
        debug_settings: Default::default(),
        buffer_device_address: false,
        allocation_sizes: Default::default(),
    })?;

    // Create timeline semaphore
    let timeline = create_timeline_semaphore(&device, 0)?;
    let timeline_value: u64 = 0;

    // 4) WSI device wrapper
    let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);

    // 5) Initial runtime knobs
    let initial_cfg = RuntimeConfig::from_env(have_swapchain_colorspace_ext);
    let cfg = initial_cfg.to_swapchain_config(size);
    #[cfg(debug_assertions)]
    let shader_dev = {
        let dir = shader_dir();
        let vp = dir.join("tri.vert.spv");
        let fp = dir.join("tri.frag.spv");
        if vp.exists() && fp.exists() {
            if let (Ok(vm), Ok(fm)) = (
                std::fs::metadata(&vp).and_then(|m| m.modified()),
                std::fs::metadata(&fp).and_then(|m| m.modified()),
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
    };

    // Create depth buffers
    let depth_format = pick_depth_format(&instance, phys);
    let desc_set_layout_camera = create_camera_desc_set_layout(&device)?;
    let desc_set_layout_material = create_material_desc_set_layout(&device)?;
    let desc_set_layout_indirect_compute = create_indirect_compute_desc_set_layout(&device)?;
    let desc_set_layout_indirect_graphics = create_indirect_graphics_desc_set_layout(&device)?;

    // GPU-driven indirect draw: a no-real-culling-yet compute shader that
    // expands this frame's candidate list into VkDrawIndexedIndirectCommand
    // entries (see indirect_cull.comp).
    let indirect_cull_pipeline_layout = unsafe {
        let push_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            offset: 0,
            size: std::mem::size_of::<u32>() as u32, // candidate_count
        };
        let layouts = [desc_set_layout_indirect_compute];
        let ci = vk::PipelineLayoutCreateInfo {
            s_type: vk::StructureType::PIPELINE_LAYOUT_CREATE_INFO,
            set_layout_count: layouts.len() as u32,
            p_set_layouts: layouts.as_ptr(),
            push_constant_range_count: 1,
            p_push_constant_ranges: &push_range,
            ..Default::default()
        };
        device.create_pipeline_layout(&ci, None)?
    };
    let indirect_cull_words = load_spv_file(&shader_dir().join("indirect_cull.comp.spv"))?;
    let indirect_cull_pipeline = create_compute_pipeline(
        &device,
        pipeline_cache,
        indirect_cull_pipeline_layout,
        &indirect_cull_words,
    )?;

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
        pipeline_cfg: PipelineConfig {
            color_format: vk::Format::UNDEFINED, // filled in from swapchain in make_initial_swapchain_resources
            depth_format,
            set_layout_camera: desc_set_layout_camera,
            set_layout_material: desc_set_layout_material,
            set_layout_indirect_graphics: desc_set_layout_indirect_graphics,
        },
    };
    let (sc, cmd, (pipeline_layout, pipeline), acq_slots, frames) =
        make_initial_swapchain_resources(&init_inp)?;

    let egui_renderer = Some(egui_overlay::build_egui_renderer(
        &instance,
        &device,
        phys,
        depth_format,
        sc.format,
        sc.image_views.len(),
    )?);

    let (depth_image, depth_alloc, depth_view) =
        create_depth_resources(&device, &mut allocator, sc.extent, depth_format)?;

    // Shared vertex/index buffers every upload_mesh call bump-allocates
    // from (see GpuMesh).
    let (shared_vbuf, shared_vbuf_alloc) = create_buffer_and_memory(
        &device,
        &mut allocator,
        MAX_SHARED_VERTICES * std::mem::size_of::<Vertex>() as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
        "shared mesh vertex buffer",
    )?;
    let (shared_ibuf, shared_ibuf_alloc) = create_buffer_and_memory(
        &device,
        &mut allocator,
        MAX_SHARED_INDICES * std::mem::size_of::<u32>() as u64,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        MemoryLocation::GpuOnly,
        "shared mesh index buffer",
    )?;

    // Global material set (swapchain-invariant)
    let (material_desc_pool, material_desc_set) =
        create_material_desc_pool_and_set(&device, desc_set_layout_material)?;

    // Sensible default until cubic-app's configure_advanced() pushes the
    // real cubic.toml values via set_sampler_config() — see the field doc
    // on VkRenderer::sampler_config. Only the dummy texture below ever
    // actually uses this default, since configure_advanced() always runs
    // before any real texture is uploaded.
    let sampler_config = SamplerConfig {
        mag_filter: vk::Filter::LINEAR,
        min_filter: vk::Filter::LINEAR,
        mipmap_mode: vk::SamplerMipmapMode::LINEAR,
        max_anisotropy: 0.0,
        lod_bias: 0.0,
    };

    // Tiny 2×2 texture and sampler, registered at bindless index 0 (the
    // fallback every draw uses until real texture loading exists).
    let (tex_image, tex_alloc, tex_view, tex_sampler) = create_dummy_texture_and_sampler(
        &device,
        &mut allocator,
        queue,
        cmd.pool,
        &sampler_config,
    )?;
    write_material_descriptors(&device, material_desc_set, 0, tex_view, tex_sampler);

    let (ubufs, umems, ubo_ptrs, ubo_size, desc_pool, desc_sets) = create_frame_uniforms_and_sets(
        &instance,
        &device,
        phys,
        &mut allocator,
        desc_set_layout_camera,
        sc.image_views.len(),
    )?;

    let indirect = create_indirect_draw_resources(
        &device,
        &mut allocator,
        desc_set_layout_indirect_compute,
        desc_set_layout_indirect_graphics,
        sc.image_views.len(),
    )?;

    // 7) Assemble VkRenderer
    let r = VkRenderer {
        instance,
        surface_loader,
        surface,

        phys,
        device,
        queue,
        allocator: Some(allocator),

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
        camera: Camera::default(),
        depth_image,
        depth_alloc,
        depth_view,
        depth_format,
        shared_vbuf,
        shared_vbuf_alloc,
        shared_ibuf,
        shared_ibuf_alloc,
        vert_alloc: RangeAlloc::new(MAX_SHARED_VERTICES as u32),
        idx_alloc: RangeAlloc::new(MAX_SHARED_INDICES as u32),
        meshes: Vec::new(),
        pending_draws: Vec::new(),
        trash: Vec::new(),
        desc_pool,
        desc_set_layout_camera,
        desc_set_layout_material,
        desc_set_layout_indirect_graphics,
        desc_set_layout_indirect_compute,
        desc_sets,
        ubufs,
        umems,
        ubo_ptrs,
        ubo_size,
        indirect_cull_pipeline,
        indirect_cull_pipeline_layout,
        candidate_bufs: indirect.candidate_bufs,
        candidate_allocs: indirect.candidate_allocs,
        candidate_ptrs: indirect.candidate_ptrs,
        indirect_bufs: indirect.indirect_bufs,
        indirect_allocs: indirect.indirect_allocs,
        draw_count_bufs: indirect.draw_count_bufs,
        draw_count_allocs: indirect.draw_count_allocs,
        indirect_desc_pool: indirect.desc_pool,
        indirect_compute_desc_sets: indirect.compute_desc_sets,
        indirect_graphics_desc_sets: indirect.graphics_desc_sets,
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
        tex_alloc,
        tex_view,
        tex_sampler,
        next_tex_index: 1,
        tex_store: Vec::new(),
        sampler_config,
        egui_renderer,
        egui_pending: None,
    };

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

    pub fn set_camera(&mut self, camera: Camera) {
        self.camera = camera;
    }

    /// Upload vertex/index data into the shared buffers via bump allocation
    /// and return an opaque handle. All meshes share one vertex buffer and
    /// one index buffer so the entire scene can be drawn with one
    /// cmd_draw_indexed_indirect_count call (GPU-driven indirect path).
    pub fn upload_mesh(&mut self, vertices: &[Vertex], indices: &[u32]) -> Result<MeshHandle> {
        let vc = vertices.len() as u32;
        let ic = indices.len() as u32;

        let vstart = self
            .vert_alloc
            .alloc(vc)
            .ok_or_else(|| anyhow!("upload_mesh: shared vertex buffer full"))?;
        let istart = self
            .idx_alloc
            .alloc(ic)
            .ok_or_else(|| anyhow!("upload_mesh: shared index buffer full"))?;

        let vbyte_offset = vstart as u64 * std::mem::size_of::<Vertex>() as u64;
        let ibyte_offset = istart as u64 * std::mem::size_of::<u32>() as u64;

        upload_via_staging(
            &self.device,
            self.allocator.as_mut().expect("allocator missing"),
            self.queue,
            self.cmd_pool,
            self.shared_vbuf,
            vbyte_offset,
            bytemuck::cast_slice(vertices),
        )?;
        upload_via_staging(
            &self.device,
            self.allocator.as_mut().expect("allocator missing"),
            self.queue,
            self.cmd_pool,
            self.shared_ibuf,
            ibyte_offset,
            bytemuck::cast_slice(indices),
        )?;

        let handle = MeshHandle(self.meshes.len() as u32);
        self.meshes.push(GpuMesh {
            first_vertex: vstart as i32,
            first_index: istart,
            index_count: ic,
            vertex_count: vc,
        });
        Ok(handle)
    }

    /// Queue a draw of a previously uploaded mesh for the next render()
    /// call, with the given per-object push constants. Call once per frame
    /// per object; the queue is consumed and cleared when that frame's
    /// command buffer is recorded.
    pub fn draw_mesh(&mut self, handle: MeshHandle, push: PushData) {
        self.pending_draws.push((handle, push));
    }

    pub fn free_mesh(&mut self, handle: MeshHandle) {
        let mesh = &self.meshes[handle.0 as usize];
        self.trash.push(DeferredDrop {
            value: self.timeline_value,
            resource: GpuResource::MeshSlot {
                first_vertex: mesh.first_vertex as u32,
                vertex_count: mesh.vertex_count,
                first_index: mesh.first_index,
                index_count: mesh.index_count,
            },
        });
        // Tombstone so draw_mesh on a freed handle panics in debug
        self.meshes[handle.0 as usize] = GpuMesh {
            first_vertex: -1,
            first_index: 0,
            index_count: 0,
            vertex_count: 0,
        };
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
    }

    // STRICT PER-FRAME ORDER:
    // 1) acquire_next_image (waits on acquire semaphore)
    // 2) record this frame's draws into the acquired image's command buffer
    //    (acquire_next_image only returns an image once the GPU is done
    //    with its previous use, so resetting its command buffer here is safe)
    // 3) queue_submit (signals render-finished for THIS image)
    // 4) queue_present (waits on render-finished)
    // Each swapchain image has its own FrameSync; do not cross-use semaphores.
    fn render(&mut self) -> Result<()> {
        self.render_frame()
    }
}
