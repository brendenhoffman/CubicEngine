use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let vs_src = r#"

    #version 450
    layout(location=0) out vec3 vColor;
    void main() {
      vec2 pos[3] = vec2[3](vec2(0.0, 0.6), vec2(-0.5, -0.4), vec2(0.5, -0.4));
      vec3 col[3] = vec3[3](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));
      gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
      vColor = col[gl_VertexIndex];
    }"#;

    let fs_src = r#"

    #version 450
    layout(location=0) in vec3 vColor;
    layout(location=0) out vec4 outColor;
    void main(){ outColor = vec4(vColor, 1.0); }"#;

    let comp = shaderc::Compiler::new().unwrap();
    let mut opts = shaderc::CompileOptions::new().unwrap();

    opts.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_0 as u32,
    );

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

    println!("cargo:rerun-if-changed=build.rs");
}
