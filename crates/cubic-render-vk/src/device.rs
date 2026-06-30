// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::{anyhow, Context, Result};
use ash::khr::{surface, swapchain};
use ash::{vk, Instance};
use std::ffi::c_char;

#[derive(Clone, Copy, Debug)]
pub(crate) enum RenderPath {
    Core13, // Vulkan 1.3 core dynamic rendering + sync2
    KhrExt, // Vulkan 1.2 + VK_KHR_dynamic_rendering + VK_KHR_synchronization2
    Legacy, // No dynamic rendering: would need render pass/framebuffer path
}

pub(crate) fn select_device_and_queue(
    instance: &ash::Instance,
    surf_i: &surface::Instance,
    surface: vk::SurfaceKHR,
) -> Result<(vk::PhysicalDevice, u32)> {
    pick_device_and_queue(instance, surf_i, surface)
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

pub(crate) fn decide_path_and_create_device(
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

    let mut device_exts: Vec<*const c_char> = vec![swapchain::NAME.as_ptr()];
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
    // Enable buffer device address (VK_KHR_buffer_device_address, core in
    // 1.2). Not used yet, but must be enabled at device creation time so
    // ray tracing / bindless buffer patterns don't require a device
    // recreation later.
    feats12.buffer_device_address = vk::TRUE;

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

    // IMPORTANT: if we're on Legacy path, bail out BEFORE creating the device
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
