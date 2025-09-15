#version 450
layout(location=0) in vec3 vColor;
layout(location=0) out vec4 outColor;

void main() {
    // RGB → GRB so colors visibly “rotate”
    outColor = vec4(vColor.g, vColor.r, vColor.b, 1.0);
}
