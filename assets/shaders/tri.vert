#version 460

layout(set = 0, binding = 0) uniform Camera {
    mat4 view_proj;
} ubo;

// Per-draw data from the GPU-driven indirect cull compute shader, indexed
// by gl_InstanceIndex which equals first_instance set by that shader (the
// original candidate index). This replaces push constants, which can't vary
// per indirect-draw-buffer entry.
struct Candidate {
    mat4 model;
    vec4 tint;
    uint first_vertex; // unused in vertex shader (draw handles buffer offset)
    uint first_index;
    uint index_count;
    uint tex_index;
};
layout(std430, set = 2, binding = 0) readonly buffer Candidates {
    Candidate candidates[];
};

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_normal;
layout(location = 4) in uint in_tex_index;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec2 v_uv;
layout(location = 2) out vec3 v_normal;
layout(location = 3) flat out uint v_tex_index;

// Optional compile-time knobs:
#ifndef UV_TILE
#define UV_TILE vec2(1.0, 1.0)   // e.g. set to vec2(2.0) to see more checkers
#endif
#ifndef FLIP_V
#define FLIP_V 0                 // set to 1 if your texture appears upside-down
#endif

void main() {
    Candidate c = candidates[gl_InstanceIndex];

    gl_Position = ubo.view_proj * c.model * vec4(in_pos, 1.0);

    v_color = in_color * c.tint.rgb;

    vec2 uv = in_uv * UV_TILE;
#if FLIP_V
    uv.y = 1.0 - uv.y;
#endif
    v_uv = uv;

    // World-space normal. Assumes uniform scale; revisit with a proper
    // normal matrix (inverse-transpose) once non-uniform scaling or real
    // lighting shows up. Unused downstream for now.
    v_normal = mat3(c.model) * in_normal;

    // Per-vertex texture index (assigned per block face by the mesher) takes
    // precedence over the per-draw candidate value, except when unset (0 —
    // the bindless dummy/checkerboard slot): OBJ-loaded entity meshes (see
    // loader::load_obj_mesh) have no per-face texture concept and always
    // bake 0 into every vertex, so they fall through to the per-draw value
    // instead — that's how draw-mesh's separate tex-index argument is meant
    // to control a single-textured entity's appearance. Chunk draws already
    // pass 0 as their own per-draw tex_index too, so this is a no-op for
    // untextured block faces, which still render the dummy as before.
    v_tex_index = in_tex_index != 0u ? in_tex_index : c.tex_index;
}
