#version 130

// PS1-style fragment shader
// Simulates 16-bit color banding, Bayer dithering, and distance fog.

uniform float ps1_banding_steps;
uniform float ps1_dither_amount;
uniform float ps1_fog_start;
uniform float ps1_fog_end;
uniform vec3 ps1_fog_color;
uniform float ps1_time;

in vec4 v_color;
in vec3 v_view_pos;
in float v_depth;

out vec4 p3d_FragColor;

// 4x4 Bayer matrix for ordered dithering
float bayer_4x4(vec2 coord) {
    int x = int(mod(coord.x, 4.0));
    int y = int(mod(coord.y, 4.0));

    int index = x + y * 4;

    float bayer[16];
    bayer[0]  = 0.0 / 16.0;  bayer[1]  = 8.0 / 16.0;
    bayer[2]  = 2.0 / 16.0;  bayer[3]  = 10.0 / 16.0;
    bayer[4]  = 12.0 / 16.0; bayer[5]  = 4.0 / 16.0;
    bayer[6]  = 14.0 / 16.0; bayer[7]  = 6.0 / 16.0;
    bayer[8]  = 3.0 / 16.0;  bayer[9]  = 11.0 / 16.0;
    bayer[10] = 1.0 / 16.0;  bayer[11] = 9.0 / 16.0;
    bayer[12] = 15.0 / 16.0; bayer[13] = 7.0 / 16.0;
    bayer[14] = 13.0 / 16.0; bayer[15] = 5.0 / 16.0;

    return bayer[index];
}

// Quantize color to simulate 16-bit 5-6-5 RGB
vec3 band_color(vec3 c, float steps) {
    return floor(c * steps + 0.5) / steps;
}

void main() {
    // Start with vertex color + any color scale
    vec3 col = v_color.rgb;

    // Color banding: quantize to simulate limited color depth
    float steps = ps1_banding_steps;
    if (steps > 0.0) {
        vec3 banded = band_color(col, steps);

        // Apply Bayer dithering to reduce visible banding artifacts
        // while keeping the low-color-depth feel
        if (ps1_dither_amount > 0.0) {
            float dither = bayer_4x4(gl_FragCoord.xy) - 0.5;
            vec3 dithered = banded + dither * (1.0 / steps) * ps1_dither_amount;
            col = clamp(dithered, 0.0, 1.0);
        } else {
            col = banded;
        }
    }

    // Distance fog / depth cue
    float fog = 0.0;
    if (ps1_fog_end > ps1_fog_start) {
        fog = clamp((v_depth - ps1_fog_start) / (ps1_fog_end - ps1_fog_start), 0.0, 1.0);
    }
    col = mix(col, ps1_fog_color, fog);

    p3d_FragColor = vec4(col, v_color.a);
}
