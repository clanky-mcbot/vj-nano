[VERTEX]
#version 130
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 texcoord;
void main() {
    gl_Position = p3d_Vertex;
    texcoord = p3d_MultiTexCoord0;
}

[FRAGMENT]
#version 130
in vec2 texcoord;
out vec4 p3d_FragColor;
void main() {
    p3d_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
