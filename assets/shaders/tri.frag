#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 3) flat in uint v_tex_index;

// Bindless texture array (set = 1), indexed per-draw via v_tex_index
// (forwarded from the vertex shader, which reads it from the candidates
// SSBO indexed by gl_InstanceIndex).
layout(set = 1, binding = 0) uniform sampler2D textures[];

layout(location = 0) out vec4 outColor;

void main() {
    vec4 texel = texture(textures[nonuniformEXT(v_tex_index)], v_uv);
    outColor = texel * vec4(v_color, 1.0);
}
