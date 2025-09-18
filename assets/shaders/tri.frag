#version 460

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_uv;

layout(set = 1, binding = 0) uniform sampler2D tex0;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 texel = texture(tex0, v_uv);
    outColor = texel * vec4(v_color, 1.0);
}
