// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::Result;
use ash::vk;

pub(crate) struct FrameSync {
    pub(crate) render_finished: vk::Semaphore,
}

pub(crate) struct AcquireSlot {
    pub(crate) sem: vk::Semaphore,
    pub(crate) last_signal_value: u64,
}

pub(crate) struct CommandResources {
    pub(crate) pool: vk::CommandPool,
    pub(crate) bufs: Vec<vk::CommandBuffer>,
}

pub(crate) fn create_command_resources(
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

pub(crate) fn create_timeline_semaphore(
    device: &ash::Device,
    initial: u64,
) -> Result<vk::Semaphore> {
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

/// Generic sync2 execution/memory barrier, with no buffer or image tied to
/// it. Suitable for compute<->graphics ordering around dispatches that
/// don't (yet) read/write a specific resource; barriers scoped to an
/// actual buffer or image should use VkBufferMemoryBarrier2 /
/// VkImageMemoryBarrier2 instead once there's a real resource to name.
#[allow(dead_code)]
fn memory_barrier2(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let barrier = vk::MemoryBarrier2 {
        s_type: vk::StructureType::MEMORY_BARRIER_2,
        src_stage_mask: src_stage,
        src_access_mask: src_access,
        dst_stage_mask: dst_stage,
        dst_access_mask: dst_access,
        ..Default::default()
    };
    let dep = vk::DependencyInfo {
        s_type: vk::StructureType::DEPENDENCY_INFO,
        memory_barrier_count: 1,
        p_memory_barriers: &barrier,
        ..Default::default()
    };
    unsafe { device.cmd_pipeline_barrier2(cmd, &dep) };
}

/// Ensures compute shader writes are visible to subsequent vertex/fragment
/// shader reads (e.g. a compute pass writing chunk/culling data that the
/// graphics pipeline then reads).
#[allow(dead_code)]
pub(crate) fn barrier_compute_to_graphics(device: &ash::Device, cmd: vk::CommandBuffer) {
    memory_barrier2(
        device,
        cmd,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_READ,
    );
}

/// Ensures vertex/fragment shader writes are visible to a subsequent
/// compute dispatch's reads (the reverse direction of
/// `barrier_compute_to_graphics`).
#[allow(dead_code)]
pub(crate) fn barrier_graphics_to_compute(device: &ash::Device, cmd: vk::CommandBuffer) {
    memory_barrier2(
        device,
        cmd,
        vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_READ,
    );
}

pub(crate) fn create_sync_objects(
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
