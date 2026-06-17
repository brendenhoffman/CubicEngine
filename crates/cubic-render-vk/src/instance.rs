// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::{Context, Result};
#[cfg(debug_assertions)]
use ash::ext::debug_utils as ext_debug;
use ash::khr::surface;
use ash::{vk, Entry, Instance};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use std::ffi::c_char;

#[cfg(debug_assertions)]
pub(crate) type DebugState = vk::DebugUtilsMessengerEXT;
#[cfg(not(debug_assertions))]
pub(crate) type DebugState = ();

type InitRet = (
    ash::Entry,
    ash::Instance,
    surface::Instance,
    vk::SurfaceKHR,
    Option<DebugState>,
    bool,
);

#[cfg(debug_assertions)]
unsafe extern "system" fn debug_callback(
    _severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if !data.is_null() {
        let msg = unsafe { std::ffi::CStr::from_ptr((*data).p_message) };
        eprintln!("[Vulkan] {:?}", msg);
    }
    vk::FALSE
}

#[cfg(debug_assertions)]
pub(crate) fn create_debug_messenger(
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
    Ok(unsafe { debug_loader.create_debug_utils_messenger(&ci, None)? })
}

#[cfg(not(debug_assertions))]
pub(crate) fn create_debug_messenger(
    _entry: &ash::Entry,
    _instance: &ash::Instance,
) -> Result<DebugState> {
    Ok(())
}

#[cfg(debug_assertions)]
pub(crate) fn destroy_debug_messenger(
    entry: &ash::Entry,
    instance: &ash::Instance,
    dbg: DebugState,
) {
    let loader = ext_debug::Instance::new(entry, instance);
    unsafe { loader.destroy_debug_utils_messenger(dbg, None) };
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
    #[cfg(debug_assertions)]
    let layer_ptrs: [*const c_char; 1] = [layers[0].as_ptr()];
    let (enabled_layer_count, pp_enabled_layer_names) = {
        #[cfg(debug_assertions)]
        {
            (layer_ptrs.len() as u32, layer_ptrs.as_ptr())
        }
        #[cfg(not(debug_assertions))]
        {
            (0u32, std::ptr::null::<*const c_char>())
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

pub(crate) fn init_instance_and_surface(
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

pub(crate) fn recreate_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    surf_i: &surface::Instance,
    old_surface: &mut vk::SurfaceKHR,
    display_raw: RawDisplayHandle,
    window_raw: RawWindowHandle,
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
