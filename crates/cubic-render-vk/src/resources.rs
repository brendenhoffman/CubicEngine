// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::{anyhow, Context, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use gpu_allocator::MemoryLocation;

// Convention: this holds the combined view*proj matrix only; the model
// transform is supplied separately via PushData and applied in the vertex
// shader, so this is not a true "MVP" matrix.
#[repr(C)]
#[derive(Clone, Copy, Default, Zeroable, Pod)]
pub(crate) struct CameraUbo {
    pub(crate) view_proj: [[f32; 4]; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
    pub uv: [f32; 2],
    pub normal: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
pub struct PushData {
    pub model: [[f32; 4]; 4],
    pub tint: [f32; 4],
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

pub(crate) type FrameUniforms = (
    Vec<vk::Buffer>,
    Vec<Allocation>,
    Vec<*mut std::ffi::c_void>,
    vk::DeviceSize,
    vk::DescriptorPool,
    Vec<vk::DescriptorSet>,
);

fn has_stencil(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT
    )
}

#[inline]
pub(crate) fn depth_aspect_mask(format: vk::Format) -> vk::ImageAspectFlags {
    let mut aspect = vk::ImageAspectFlags::DEPTH;
    if has_stencil(format) {
        aspect |= vk::ImageAspectFlags::STENCIL;
    }
    aspect
}

#[inline]
pub(crate) fn depth_attachment_layout(format: vk::Format) -> vk::ImageLayout {
    if has_stencil(format) {
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
    }
}

// Prefer pure depth formats only: D32F -> D16
pub(crate) fn pick_depth_format(instance: &ash::Instance, phys: vk::PhysicalDevice) -> vk::Format {
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

pub(crate) fn create_depth_resources(
    device: &ash::Device,
    allocator: &mut Allocator,
    extent: vk::Extent2D,
    depth_format: vk::Format,
) -> Result<(vk::Image, Allocation, vk::ImageView)> {
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
    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name: "depth image",
            requirements: mem_req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::DedicatedImage(image),
        })
        .with_context(|| format!("allocate (depth) size={}", mem_req.size))?;

    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }
        .with_context(|| "bind_image_memory (depth)")?;

    let depth_view = make_depth_view(device, image, depth_format)?;
    Ok((image, allocation, depth_view))
}

// Buffers are sub-allocated (GpuAllocatorManaged) rather than given a
// dedicated VkDeviceMemory each: many short-lived/small buffers (UBOs,
// staging, mesh data) would otherwise burn through the driver's discrete
// allocation cap (~4096) fast.
pub(crate) fn create_buffer_and_memory(
    device: &ash::Device,
    allocator: &mut Allocator,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    location: MemoryLocation,
    name: &str,
) -> Result<(vk::Buffer, Allocation)> {
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
    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements: req,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })
        .with_context(|| format!("allocate buffer name={name} usage={usage:?} size={size}"))?;

    unsafe { device.bind_buffer_memory(buf, allocation.memory(), allocation.offset()) }
        .with_context(|| "bind_buffer_memory")?;

    Ok((buf, allocation))
}

fn create_host_visible_ubo(
    device: &ash::Device,
    allocator: &mut Allocator,
    size: vk::DeviceSize,
) -> Result<(vk::Buffer, Allocation)> {
    create_buffer_and_memory(
        device,
        allocator,
        size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        MemoryLocation::CpuToGpu,
        "camera ubo",
    )
}

pub(crate) fn create_camera_desc_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout> {
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

pub(crate) fn create_material_desc_set_layout(
    device: &ash::Device,
) -> Result<vk::DescriptorSetLayout> {
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
    device: &ash::Device,
    allocator: &mut Allocator,
    info: &ImageAllocInfo,
    name: &str,
) -> Result<(vk::Image, Allocation)> {
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
    let image = unsafe { device.create_image(&ci, None) }.with_context(|| {
        format!(
            "create_image fmt={:?} extent={:?}",
            info.format, info.extent
        )
    })?;

    let req = unsafe { device.get_image_memory_requirements(image) };
    let allocation = allocator
        .allocate(&AllocationCreateDesc {
            name,
            requirements: req,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::DedicatedImage(image),
        })
        .with_context(|| format!("allocate (image) name={name} size={}", req.size))?;
    unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }?;
    Ok((image, allocation))
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
    // No anisotropy yet (you didn't enable it on device features). Safe defaults.
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

pub(crate) fn create_material_desc_pool_and_set(
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

pub(crate) fn write_material_descriptors(
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

pub(crate) fn create_dummy_texture_and_sampler(
    device: &ash::Device,
    allocator: &mut Allocator,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
) -> Result<(vk::Image, Allocation, vk::ImageView, vk::Sampler)> {
    // 2x2 checkerboard RGBA
    let extent = vk::Extent2D {
        width: 2,
        height: 2,
    };
    let pixels: [u8; 16] = [
        255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255,
    ];

    // Create device-local image
    let info = ImageAllocInfo {
        extent,
        mip_levels: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
        tiling: vk::ImageTiling::OPTIMAL,
    };
    let (image, memory) = create_image_and_memory(device, allocator, &info, "dummy texture")?;

    // Create staging buffer and copy pixels into it
    let size = pixels.len() as vk::DeviceSize;
    let (staging, mut staging_alloc) = create_buffer_and_memory(
        device,
        allocator,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
        "dummy texture staging",
    )?;
    {
        let mapped = staging_alloc
            .mapped_slice_mut()
            .ok_or_else(|| anyhow!("dummy texture staging allocation not host-mapped"))?;
        mapped[..pixels.len()].copy_from_slice(&pixels);
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
    }
    allocator.free(staging_alloc)?;

    let view = make_image_view_2d_color(device, image, vk::Format::R8G8B8A8_UNORM, 0, 1)?;
    let sampler = create_sampler(device, 1)?;

    Ok((image, memory, view, sampler))
}

/// One-shot staging upload: host->staging, then staging->dst (device-local).
/// Uses the graphics queue and a one-time command buffer; waits until done.
pub(crate) fn upload_via_staging(
    device: &ash::Device,
    allocator: &mut Allocator,
    queue: vk::Queue,
    cmd_pool: vk::CommandPool,
    dst: vk::Buffer,
    src_data: &[u8],
) -> Result<()> {
    // 1) Create CPU-to-GPU staging buffer
    let size = src_data.len() as vk::DeviceSize;
    let (staging, mut staging_alloc) = create_buffer_and_memory(
        device,
        allocator,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
        "upload staging",
    )?;

    // Map + copy into staging
    let mapped = staging_alloc
        .mapped_slice_mut()
        .ok_or_else(|| anyhow!("upload staging allocation not host-mapped"))?;
    mapped[..src_data.len()].copy_from_slice(src_data);

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
    allocator.free(staging_alloc)?;
    Ok(())
}

pub(crate) fn create_frame_uniforms_and_sets(
    instance: &ash::Instance,
    device: &ash::Device,
    phys: vk::PhysicalDevice,
    allocator: &mut Allocator,
    set_layout: vk::DescriptorSetLayout,
    image_count: usize,
) -> Result<FrameUniforms> {
    let limits = unsafe { instance.get_physical_device_properties(phys).limits };
    let a = limits.min_uniform_buffer_offset_alignment.max(1);
    let sz = std::mem::size_of::<CameraUbo>() as u64;
    let ubo_size = sz.div_ceil(a) * a;

    let mut ubufs = Vec::with_capacity(image_count);
    let mut uallocs = Vec::with_capacity(image_count);
    let mut ubo_ptrs = Vec::with_capacity(image_count);

    for _ in 0..image_count {
        let (b, alloc) = create_host_visible_ubo(device, allocator, ubo_size)?;
        let ptr = alloc
            .mapped_ptr()
            .ok_or_else(|| anyhow!("UBO allocation not host-mapped"))?
            .as_ptr();
        ubufs.push(b);
        uallocs.push(alloc);
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

    Ok((ubufs, uallocs, ubo_ptrs, ubo_size, pool, sets))
}
