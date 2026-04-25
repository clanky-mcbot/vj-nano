#version 130

// Fullscreen pass-through vertex shader for post-processing

in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;

out vec2 texcoord;

void main() {
    gl_Position = p3d_Vertex;
    texcoord = p3d_MultiTexCoord0;
}
