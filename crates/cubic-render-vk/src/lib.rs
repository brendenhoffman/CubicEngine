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
use cubic_render::{RenderSize, Renderer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use tracing::info;

use device::{decide_path_and_create_device, select_device_and_queue, RenderPath};
#[cfg(debug_assertions)]
use instance::destroy_debug_messenger;
use instance::{init_instance_and_surface, recreate_surface};
#[cfg(debug_assertions)]
use pipeline::ShaderDev;
use pipeline::{
    create_or_load_pipeline_cache, create_pipeline, pipeline_cache_path, save_pipeline_cache,
};
use resources::{
    create_buffer_and_memory, create_camera_desc_set_layout, create_depth_resources,
    create_dummy_texture_and_sampler, create_frame_uniforms_and_sets,
    create_material_desc_pool_and_set, create_material_desc_set_layout, depth_aspect_mask,
    depth_attachment_layout, pick_depth_format, upload_via_staging, write_material_descriptors,
    CameraUbo, Vertex,
};
use swapchain::{
    create_hdr_metadata_if_needed, create_swapchain_bundle, SwapchainBundle, SwapchainConfig,
};
pub use swapchain::{HdrFlavor, VkVsyncMode};
use sync::{
    create_command_resources, create_sync_objects, create_timeline_semaphore, AcquireSlot,
    CommandResources, FrameSync,
};

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

// 3) Renderer data model
pub struct VkRenderer {
    instance: ash::Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,

    phys: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,

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
    depth_format: vk::Format,
    desc_set_layout_camera: vk::DescriptorSetLayout,
    desc_set_layout_material: vk::DescriptorSetLayout,
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
        bundle.format,
        inp.depth_format,
        bundle.extent,
        inp.desc_set_layout_camera,
        inp.desc_set_layout_material,
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
        if let Ok(dir) = std::env::var("CUBIC_SHADER_DIR") {
            let dir = std::path::PathBuf::from(dir);
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
