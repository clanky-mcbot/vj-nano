#version 130

// PS1-style vertex shader
// Simulates low-precision fixed-point vertex processing, vertex snap,
// and subtle affine wobble.

uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelViewMatrix;
uniform float ps1_time;
uniform float ps1_snap_resolution;
uniform float ps1_wobble_intensity;

in vec4 p3d_Vertex;
in vec3 p3d_Normal;
in vec4 p3d_Color;

out vec4 v_color;
out vec3 v_view_pos;
out float v_depth;

// Quantize a value to simulate fixed-point precision
float quantize(float x, float steps) {
    return floor(x * steps + 0.5) / steps;
}

vec3 quantize_vec3(vec3 v, float steps) {
    return vec3(quantize(v.x, steps),
                quantize(v.y, steps),
                quantize(v.z, steps));
}

void main() {
    // Start with original vertex
    vec4 pos = p3d_Vertex;

    // Apply subtle vertex wobble based on time and position
    // This simulates the unstable vertex processing of the PS1 GTE
    if (ps1_wobble_intensity > 0.0) {
        float wobble = sin(ps1_time * 3.0 + pos.x * 8.0 + pos.y * 6.0) *
                       cos(ps1_time * 2.5 + pos.z * 5.0) *
                       ps1_wobble_intensity * 0.015;
        pos.x += wobble;
        pos.y += wobble * 0.7;
    }

    // Transform to clip space
    vec4 clip_pos = p3d_ModelViewProjectionMatrix * pos;

    // Vertex snap: quantize clip-space XY to simulate low subpixel precision
    // PS1 had ~12-bit subpixel precision in the rasterizer
    if (ps1_snap_resolution > 0.0) {
        // Quantize in normalized device coordinates (after perspective divide)
        // but before the final viewport mapping
        vec3 ndc = clip_pos.xyz / max(clip_pos.w, 0.0001);
        float snap = ps1_snap_resolution;
        ndc.x = quantize(ndc.x, snap);
        ndc.y = quantize(ndc.y, snap);
        // Reconstruct clip space
        clip_pos.xy = ndc.xy * clip_pos.w;
    }

    gl_Position = clip_pos;

    // Pass color
    v_color = p3d_Color;

    // View-space position for fog
    vec4 view_pos = p3d_ModelViewMatrix * p3d_Vertex;
    v_view_pos = view_pos.xyz;
    v_depth = -view_pos.z; // positive depth
}
