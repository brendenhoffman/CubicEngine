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
