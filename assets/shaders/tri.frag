#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_uv;

// Bindless texture array: one global set (no per-draw descriptor set
// switching), indexed per-draw via push.tex_index.
layout(set = 1, binding = 0) uniform sampler2D textures[];

layout(push_constant) uniform Push {
    mat4 model;
    vec4 tint;
    uint tex_index;
} push;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 texel = texture(textures[nonuniformEXT(push.tex_index)], v_uv);
    outColor = texel * vec4(v_color, 1.0);
}
