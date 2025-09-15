use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Vertex shader: consume vertex buffer (pos, color) and pass color through.
    // NOTE: Matches your Rust pipeline vertex layout:
    //   - binding 0, location 0: R32G32B32_SFLOAT (pos)
    //   - binding 0, location 1: R32G32B32_SFLOAT (color)
    let vs_src = r#"
#version 450
layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;

layout(set = 0, binding = 0) uniform Camera { mat4 mvp; } u;

layout(location = 0) out vec3 vColor;

void main() {
    vColor = inColor;
    gl_Position = u.mvp * vec4(inPos, 1.0);
}
"#;

    // Fragment shader: just write the color (tonemap/sRGB later).
    let fs_src = r#"
#version 450
layout(location = 0) in vec3 vColor;
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(vColor, 1.0);
}
"#;

    let comp = shaderc::Compiler::new().unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();

    // Vulkan 1.0 is fine for this; you can bump to 1.2 later if you want.
    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_0 as u32,
    );
    // Mild optimization
    opts.set_optimization_level(shaderc::OptimizationLevel::Performance);

    let vs_spv = comp
        .compile_into_spirv(
            vs_src,
            shaderc::ShaderKind::Vertex,
            "tri.vert",
            "main",
            Some(&opts),
        )
        .unwrap();

    let fs_spv = comp
        .compile_into_spirv(
            fs_src,
            shaderc::ShaderKind::Fragment,
            "tri.frag",
            "main",
            Some(&opts),
        )
        .unwrap();

    fs::write(out.join("tri.vert.spv"), vs_spv.as_binary_u8()).unwrap();
    fs::write(out.join("tri.frag.spv"), fs_spv.as_binary_u8()).unwrap();

    // Re-run if this file changes (inline sources live here)
    println!("cargo:rerun-if-changed=build.rs");
}
