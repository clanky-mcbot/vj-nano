#version 130

// PS1-style vertex shader v2
// Simulates low-precision fixed-point vertex processing with visible snap.

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform float ps1_time;
uniform float ps1_snap_resolution;
uniform float ps1_wobble_intensity;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;

out vec4 v_color;
out vec3 v_normal;
out vec3 v_view_pos;
out float v_depth;

float quantize(float x, float steps) {
    return floor(x * steps + 0.5) / steps;
}

void main() {
    vec4 pos = p3d_Vertex;

    // Stronger vertex wobble for PS1 instability feel
    if (ps1_wobble_intensity > 0.0) {
        float wobble = sin(ps1_time * 4.0 + pos.x * 10.0 + pos.y * 7.0) *
                       cos(ps1_time * 3.0 + pos.z * 6.0) *
                       ps1_wobble_intensity * 0.025;
        pos.x += wobble;
        pos.y += wobble * 0.8;
        pos.z += wobble * 0.3;
    }

    vec4 clip_pos = p3d_ModelViewProjectionMatrix * pos;

    // VERTEX SNAP: quantize in NDC to simulate low subpixel precision
    // Lower resolution = more visible snap. PS1 was ~256x224 screen with
    // limited precision, so snap to a coarse grid.
    if (ps1_snap_resolution > 0.0) {
        vec3 ndc = clip_pos.xyz / max(clip_pos.w, 0.0001);
        float snap = ps1_snap_resolution;
        ndc.x = quantize(ndc.x, snap);
        ndc.y = quantize(ndc.y, snap);
        clip_pos.xy = ndc.xy * clip_pos.w;
    }

    gl_Position = clip_pos;

    v_color = p3d_Color;
    v_normal = mat3(p3d_ModelViewMatrix) * p3d_Normal;

    vec4 view_pos = p3d_ModelViewMatrix * p3d_Vertex;
    v_view_pos = view_pos.xyz;
    v_depth = -view_pos.z;
}
