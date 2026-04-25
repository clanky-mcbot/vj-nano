#version 130

// Post-process mega-shader: dither + scanlines + pixelate + vignette

uniform sampler2D tex;
uniform vec2 resolution;
uniform float enable_dither;
uniform float enable_scanlines;
uniform float enable_pixelate;
uniform float enable_vignette;

in vec2 texcoord;
out vec4 p3d_FragColor;

// 4x4 Bayer ordered dither matrix
float bayer_4x4(vec2 coord) {
    int x = int(mod(coord.x, 4.0));
    int y = int(mod(coord.y, 4.0));
    int idx = x + y * 4;

    if (idx == 0)  return 0.0 / 16.0;
    if (idx == 1)  return 8.0 / 16.0;
    if (idx == 2)  return 2.0 / 16.0;
    if (idx == 3)  return 10.0 / 16.0;
    if (idx == 4)  return 12.0 / 16.0;
    if (idx == 5)  return 4.0 / 16.0;
    if (idx == 6)  return 14.0 / 16.0;
    if (idx == 7)  return 6.0 / 16.0;
    if (idx == 8)  return 3.0 / 16.0;
    if (idx == 9)  return 11.0 / 16.0;
    if (idx == 10) return 1.0 / 16.0;
    if (idx == 11) return 9.0 / 16.0;
    if (idx == 12) return 15.0 / 16.0;
    if (idx == 13) return 7.0 / 16.0;
    if (idx == 14) return 13.0 / 16.0;
    return 5.0 / 16.0;
}

void main() {
    vec2 uv = texcoord;

    // --- Pixelate ---
    if (enable_pixelate > 0.5) {
        float px = 6.0;
        uv = floor(uv * resolution / px) * px / resolution + 0.5 / resolution;
    }

    vec4 color = texture2D(tex, uv);
    // Force opaque — some embedded drivers return alpha=0 for RGB buffers.
    color.a = 1.0;

    // --- Ordered dither to limited palette ---
    if (enable_dither > 0.5) {
        float threshold = bayer_4x4(gl_FragCoord.xy);
        vec3 quantized = floor(color.rgb * 6.0 + threshold) / 6.0;
        color.rgb = quantized;
    }

    // --- CRT scanlines ---
    if (enable_scanlines > 0.5) {
        float line = sin(gl_FragCoord.y * 1.2) * 0.12 + 0.88;
        color.rgb *= line;
        // Slight chromatic offset per scanline
        float offset = sin(gl_FragCoord.y * 0.6) * 0.003;
        color.r = texture2D(tex, uv + vec2(offset, 0.0)).r;
        color.b = texture2D(tex, uv - vec2(offset, 0.0)).b;
    }

    // --- Vignette (subtle, always on when any filter active) ---
    if (enable_vignette > 0.5 || enable_dither > 0.5 || enable_scanlines > 0.5 || enable_pixelate > 0.5) {
        vec2 center = uv - 0.5;
        float vignette = 1.0 - dot(center, center) * 0.6;
        color.rgb *= clamp(vignette, 0.3, 1.0);
    }

    p3d_FragColor = color;
}
