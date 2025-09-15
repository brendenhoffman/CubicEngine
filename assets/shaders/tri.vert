#version 450
layout(set=0, binding=0) uniform CameraUbo { mat4 uMVP; };

layout(location=0) in vec3 inPos;
layout(location=1) in vec3 inColor;

layout(location=0) out vec3 vColor;

void main() {
    // Small horizontal nudge so position changes are obvious
    vec3 p = inPos + vec3(0.09, 0.1, 0.2);
    gl_Position = uMVP * vec4(p, 1.0);
    vColor = inColor;
}
