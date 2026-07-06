#version 460
#extension GL_EXT_nonuniform_qualifier : require

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec2 v_uv;
layout(location = 2) in vec3 v_normal;
layout(location = 3) flat in uint v_tex_index;

layout(set = 1, binding = 0) uniform sampler2D textures[];

layout(location = 0) out vec4 outColor;

void main() {
    vec4 texel = texture(textures[nonuniformEXT(v_tex_index)], v_uv);

    vec3 sun_dir = normalize(vec3(0.5, 1.0, 0.3));
    float diffuse = max(dot(normalize(v_normal), sun_dir), 0.0);
    float ambient = 0.4;
    float light = ambient + (1.0 - ambient) * diffuse;

    outColor = texel * vec4(v_color * light, 1.0);
}
