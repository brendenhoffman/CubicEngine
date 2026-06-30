#version 460

layout(set = 0, binding = 0) uniform Camera {
    mat4 mvp;
} ubo;

layout(push_constant) uniform Push {
    mat4 model;
    vec4 tint;
} push;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_normal;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec2 v_uv;
layout(location = 2) out vec3 v_normal;

// Optional compile-time knobs:
#ifndef UV_TILE
#define UV_TILE vec2(1.0, 1.0)   // e.g. set to vec2(2.0) to see more checkers
#endif
#ifndef FLIP_V
#define FLIP_V 0                 // set to 1 if your texture appears upside-down
#endif

void main() {
    gl_Position = ubo.mvp * push.model * vec4(in_pos, 1.0);

    v_color = in_color * push.tint.rgb;

    vec2 uv = in_uv * UV_TILE;
#if FLIP_V
    uv.y = 1.0 - uv.y;
#endif
    v_uv = uv;

    // World-space normal. Assumes uniform scale; revisit with a proper
    // normal matrix (inverse-transpose) once non-uniform scaling or real
    // lighting shows up. Unused downstream for now.
    v_normal = mat3(push.model) * in_normal;
}
