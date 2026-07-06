// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]

mod device;
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
    create_material_desc_set_layout, create_texture_and_sampler, depth_aspect_mask,
    depth_attachment_layout, pick_depth_format, upload_via_staging, write_material_descriptors,
    CameraUbo, DrawCandidate, RangeAlloc, MAX_INDIRECT_DRAWS, MAX_SHARED_INDICES,
    MAX_SHARED_VERTICES, MAX_TEXTURES,
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

    // Tiny 2×2 texture and sampler, registered at bindless index 0 (the
    // fallback every draw uses until real texture loading exists).
    let (tex_image, tex_alloc, tex_view, tex_sampler) =
        create_dummy_texture_and_sampler(&device, &mut allocator, queue, cmd.pool)?;
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

    #[inline]
    fn should_skip_for_backoff(&mut self) -> bool {
        if self.backoff_frames > 0 {
            self.backoff_frames -= 1;
            true
        } else {
            false
        }
    }

    /// Destroy every trashed resource whose retirement value the timeline
    /// semaphore has already reached. Non-blocking: queries the semaphore's
    /// current counter rather than waiting on it.
    fn drain_trash(&mut self) {
        if self.trash.is_empty() {
            return;
        }
        // On query failure, fall back to 0 (i.e. drain nothing this round)
        // rather than risk destroying a resource still in use.
        let signaled =
            unsafe { self.device.get_semaphore_counter_value(self.timeline) }.unwrap_or(0);

        let mut i = 0;
        while i < self.trash.len() {
            if self.trash[i].value > signaled {
                i += 1;
                continue;
            }
            let item = self.trash.swap_remove(i);
            match item.resource {
                GpuResource::Buffer { buffer, alloc } => unsafe {
                    self.device.destroy_buffer(buffer, None);
                    let _ = self
                        .allocator
                        .as_mut()
                        .expect("allocator missing")
                        .free(alloc);
                },
                GpuResource::Image { image, alloc } => unsafe {
                    self.device.destroy_image(image, None);
                    let _ = self
                        .allocator
                        .as_mut()
                        .expect("allocator missing")
                        .free(alloc);
                },
                GpuResource::ImageView(view) => unsafe {
                    self.device.destroy_image_view(view, None);
                },
                GpuResource::Pipeline(p) => unsafe {
                    self.device.destroy_pipeline(p, None);
                },
                GpuResource::PipelineLayout(l) => unsafe {
                    self.device.destroy_pipeline_layout(l, None);
                },
                GpuResource::MeshSlot {
                    first_vertex,
                    vertex_count,
                    first_index,
                    index_count,
                } => {
                    self.vert_alloc.free(first_vertex, vertex_count);
                    self.idx_alloc.free(first_index, index_count);
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    fn hot_reload_shaders_if_changed(&mut self) -> Result<()> {
        let Some(dev) = self.shader_dev.as_mut() else {
            return Ok(());
        };

        let vm = std::fs::metadata(&dev.vert_spv)
            .and_then(|m| m.modified())
            .ok();
        let fm = std::fs::metadata(&dev.frag_spv)
            .and_then(|m| m.modified())
            .ok();

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

        // Rebuild using the same loader (reads from shader_dir(), i.e.
        // CUBIC_SHADER_DIR if set, else assets/shaders/)
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

        // No re-record needed here: render() records each frame's command
        // buffer fresh against whatever self.pipeline currently is.
        Ok(())
    }

    fn update_camera_ubo_for_image(
        &self,
        image_index: usize,
        camera: &Camera,
        aspect: f32,
    ) -> anyhow::Result<()> {
        let view_proj = camera.projection_matrix(aspect) * camera.view_matrix_no_translation();
        let data = CameraUbo {
            view_proj: view_proj.to_cols_array_2d(),
        };

        let dst = self.ubo_ptrs[image_index];
        if dst.is_null() {
            return Err(anyhow::anyhow!("UBO memory not mapped"));
        }
        let src = bytemuck::bytes_of(&data);

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

    /// Phase 1 of the GPU-driven draw: write candidates, dispatch indirect-cull
    /// compute, and leave the indirect/count buffers ready for the draw call.
    /// Must run OUTSIDE the render pass (before vkCmdBeginRendering).
    fn cull_compute_prepass(&self, cmd: vk::CommandBuffer, image_index: usize) {
        let candidate_count = self.pending_draws.len() as u32;

        // Write this frame's DrawCandidate array to the host-mapped buffer.
        if candidate_count > 0 {
            let ptr = self.candidate_ptrs[image_index] as *mut DrawCandidate;
            for (i, (handle, push)) in self.pending_draws.iter().enumerate() {
                let mesh = match self.meshes.get(handle.0 as usize) {
                    Some(m) => m,
                    None => continue,
                };
                unsafe {
                    std::ptr::write(
                        ptr.add(i),
                        DrawCandidate {
                            model: push.model,
                            tint: push.tint,
                            first_vertex: mesh.first_vertex as u32,
                            first_index: mesh.first_index,
                            index_count: mesh.index_count,
                            tex_index: push.tex_index,
                        },
                    );
                }
            }
        }

        // --- Compute dispatch: expand candidates → indirect commands ---
        // Zero the draw-count atomics before the compute shader writes them.
        // TRANSFER_DST ensures vkCmdFillBuffer completes before COMPUTE reads.
        let fill_to_compute = vk::MemoryBarrier2 {
            s_type: vk::StructureType::MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::TRANSFER,
            src_access_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            dst_access_mask: vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
            ..Default::default()
        };
        let compute_to_indirect = vk::MemoryBarrier2 {
            s_type: vk::StructureType::MEMORY_BARRIER_2,
            src_stage_mask: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_access_mask: vk::AccessFlags2::SHADER_WRITE,
            dst_stage_mask: vk::PipelineStageFlags2::DRAW_INDIRECT
                | vk::PipelineStageFlags2::VERTEX_SHADER,
            dst_access_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ
                | vk::AccessFlags2::SHADER_READ,
            ..Default::default()
        };
        unsafe {
            self.device
                .cmd_fill_buffer(cmd, self.draw_count_bufs[image_index], 0, 4, 0);
            let dep = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                memory_barrier_count: 1,
                p_memory_barriers: &fill_to_compute,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier2(cmd, &dep);

            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.indirect_cull_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.indirect_cull_pipeline_layout,
                0,
                std::slice::from_ref(&self.indirect_compute_desc_sets[image_index]),
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.indirect_cull_pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&candidate_count),
            );
            let groups = candidate_count.div_ceil(64).max(1);
            self.device.cmd_dispatch(cmd, groups, 1, 1);

            let dep2 = vk::DependencyInfo {
                s_type: vk::StructureType::DEPENDENCY_INFO,
                memory_barrier_count: 1,
                p_memory_barriers: &compute_to_indirect,
                ..Default::default()
            };
            self.device.cmd_pipeline_barrier2(cmd, &dep2);
        }
    }

    /// Phase 2: the actual indirect draw call. Must run INSIDE the render pass
    /// (between vkCmdBeginRendering and vkCmdEndRendering).
    fn record_indirect_draws(&self, cmd: vk::CommandBuffer, image_index: usize) -> Result<()> {
        if self.pipeline == vk::Pipeline::null() {
            return Err(anyhow!("pipeline is VK_NULL_HANDLE at record time"));
        }
        let vp = vk::Viewport {
            x: 0.0,
            y: self.extent.height as f32,
            width: self.extent.width as f32,
            height: -(self.extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let sc = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };
        let sets = [
            self.desc_sets[image_index],                   // set 0: camera
            self.material_desc_set,                        // set 1: bindless textures
            self.indirect_graphics_desc_sets[image_index], // set 2: candidates
        ];
        let offsets = [0_u64];
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device
                .cmd_set_viewport(cmd, 0, std::slice::from_ref(&vp));
            self.device
                .cmd_set_scissor(cmd, 0, std::slice::from_ref(&sc));
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &sets,
                &[],
            );
            // One shared vertex/index buffer pair for all meshes.
            self.device.cmd_bind_vertex_buffers(
                cmd,
                0,
                std::slice::from_ref(&self.shared_vbuf),
                &offsets,
            );
            self.device
                .cmd_bind_index_buffer(cmd, self.shared_ibuf, 0, vk::IndexType::UINT32);
            // GPU populates the indirect buffer and count; CPU has no per-draw
            // involvement beyond writing the candidate array above.
            self.device.cmd_draw_indexed_indirect_count(
                cmd,
                self.indirect_bufs[image_index],
                0,
                self.draw_count_bufs[image_index],
                0,
                MAX_INDIRECT_DRAWS,
                std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
            );
        }
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

    /// Dispatches the no-op compute pipeline, bracketed by compute<->graphics
    /// barriers, purely to validate the compute pipeline/dispatch/sync path
    /// (see "Add compute pipeline infrastructure"). Does no real work yet;
    #[inline]
    // Records draws queued via draw_mesh() into the given image's command
    // buffer. Called fresh every frame for the just-acquired image (see
    // render()) — safe to reset because acquire_next_image only returns an
    // image index once the GPU is done with its previous use.
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
        // Phase 1: compute cull — MUST happen outside the render pass.
        self.cull_compute_prepass(cmd, image_index);
        self.transition_to_color(cmd, image);
        self.transition_depth_to_attachment(cmd, self.depth_image);
        self.begin_rendering(cmd, image_view);
        // Phase 2: indirect draw — inside the render pass.
        self.record_indirect_draws(cmd, image_index)?;
        unsafe { self.device.cmd_end_rendering(cmd) };
        self.transition_to_present(cmd, image);

        // end
        unsafe { self.device.end_command_buffer(cmd)? };
        Ok(())
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

    /// Upload an RGBA8 texture and register it into the bindless descriptor
    /// array, returning its index (see `PushData::tex_index`). Index 0 is
    /// permanently the dummy texture created in `build_renderer`; this
    /// starts handing out indices at 1.
    pub fn upload_texture(&mut self, pixels: &[u8], width: u32, height: u32) -> Result<u32> {
        if self.next_tex_index >= MAX_TEXTURES {
            return Err(anyhow!(
                "upload_texture: bindless texture array full (MAX_TEXTURES = {MAX_TEXTURES})"
            ));
        }

        let (image, alloc, view, sampler) = create_texture_and_sampler(
            &self.device,
            self.allocator.as_mut().expect("allocator missing"),
            self.queue,
            self.cmd_pool,
            pixels,
            width,
            height,
        )?;

        let index = self.next_tex_index;
        write_material_descriptors(&self.device, self.material_desc_set, index, view, sampler);

        self.tex_store.push((image, alloc, view, sampler));
        self.next_tex_index += 1;

        Ok(index)
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

        // 1) Acquire
        let acq_sem = self.acq_slots[self.acq_index].sem;
        let acq_last_signal_value = self.acq_slots[self.acq_index].last_signal_value;
        if acq_last_signal_value > 0 {
            let wait_info = vk::SemaphoreWaitInfo {
                s_type: vk::StructureType::SEMAPHORE_WAIT_INFO,
                flags: vk::SemaphoreWaitFlags::empty(),
                semaphore_count: 1,
                p_semaphores: &self.timeline,
                p_values: &acq_last_signal_value,
                ..Default::default()
            };
            unsafe {
                self.device.wait_semaphores(&wait_info, u64::MAX)?;
            }
        }

        self.drain_trash();

        let (image_index, _) = match unsafe {
            self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                acq_sem,
                vk::Fence::null(),
            )
        } {
            Ok(pair) => pair,
            Err(e) if is_swapchain_out_of_date(e) => {
                self.backoff_frames = 2;
                let want = RenderSize {
                    width: self.extent.width,
                    height: self.extent.height,
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
                        width: self.extent.width,
                        height: self.extent.height,
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
        self.update_camera_ubo_for_image(img, &self.camera, aspect)?;

        // Record this frame's draws (queued via draw_mesh()) into the
        // image we just acquired, then clear the queue for the next frame.
        self.record_one_command(cmd, self.images[img], self.image_views[img], img)?;
        self.pending_draws.clear();

        // 2) Submit (wait on acquire sem; signal render-finished; bump timeline)
        let next_value = self.timeline_value.wrapping_add(1);

        let stage_color = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
        let stage2_color = stage_flags2_from_legacy(stage_color);

        // Build the semaphore infos
        let wait_acquire = semaphore_submit_info_wait(acq_sem, 0, stage2_color);
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
                    width: self.extent.width,
                    height: self.extent.height,
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
                        width: self.extent.width,
                        height: self.extent.height,
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
