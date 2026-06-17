// SPDX-License-Identifier: CEPL-1.0
#![deny(unsafe_op_in_unsafe_fn)]
use anyhow::{anyhow, Context, Result};
use ash::util::read_spv;
use ash::vk;
use std::io::Cursor;
#[cfg(debug_assertions)]
use std::time::SystemTime;
use std::{fs, path::Path, path::PathBuf};

#[cfg(debug_assertions)]
pub(crate) struct ShaderDev {
    pub(crate) vert_spv: PathBuf,
    pub(crate) frag_spv: PathBuf,
    pub(crate) vert_mtime: SystemTime,
    pub(crate) frag_mtime: SystemTime,
}

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

pub(crate) fn pipeline_cache_path(props: &vk::PhysicalDeviceProperties) -> PathBuf {
    // Keep it simple: local file next to the binary.
    // You can switch to a platform cache dir later.
    let uuid = hex_bytes(&props.pipeline_cache_uuid);
    PathBuf::from(format!(
        "vk_pipeline_cache_{:04x}_{:04x}_{:08x}_{}.bin",
        props.vendor_id, props.device_id, props.driver_version, uuid
    ))
}

pub(crate) fn create_or_load_pipeline_cache(
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

pub(crate) fn save_pipeline_cache(
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

pub(crate) fn create_pipeline(
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
        stride: std::mem::size_of::<super::resources::Vertex>() as u32,
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
