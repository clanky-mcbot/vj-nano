#version 130

// PS1-style fragment shader v2
// Color banding, Bayer dithering, distance fog, + simple lighting for gradients.

uniform float ps1_banding_steps;
uniform float ps1_dither_amount;
uniform float ps1_fog_start;
uniform float ps1_fog_end;
uniform vec3 ps1_fog_color;
uniform float ps1_time;

in vec4 v_color;
in vec3 v_normal;
in vec3 v_view_pos;
in float v_depth;

out vec4 p3d_FragColor;

// 4x4 Bayer matrix
float bayer_4x4(vec2 coord) {
    int x = int(mod(coord.x, 4.0));
    int y = int(mod(coord.y, 4.0));
    int idx = x + y * 4;
    float vals[16];
    vals[0]  = 0.0/16.0;  vals[1]  = 8.0/16.0;
    vals[2]  = 2.0/16.0;  vals[3]  = 10.0/16.0;
    vals[4]  = 12.0/16.0; vals[5]  = 4.0/16.0;
    vals[6]  = 14.0/16.0; vals[7]  = 6.0/16.0;
    vals[8]  = 3.0/16.0;  vals[9]  = 11.0/16.0;
    vals[10] = 1.0/16.0;  vals[11] = 9.0/16.0;
    vals[12] = 15.0/16.0; vals[13] = 7.0/16.0;
    vals[14] = 13.0/16.0; vals[15] = 5.0/16.0;
    return vals[idx];
}

vec3 band_color(vec3 c, float steps) {
    return floor(c * steps + 0.5) / steps;
}

void main() {
    // Simple directional lighting to create gradients across faces
    // This makes color banding visible!
    vec3 light_dir = normalize(vec3(0.5, 0.8, 1.0));
    vec3 normal = normalize(v_normal);
    float ndotl = max(dot(normal, light_dir), 0.0);
    float ambient = 0.35;
    float lighting = ambient + ndotl * 0.65;

    // Apply lighting to vertex color
    vec3 col = v_color.rgb * lighting;

    // COLOR BANDING: quantize to simulate 16-bit color depth
    float steps = ps1_banding_steps;
    if (steps > 0.0) {
        vec3 banded = band_color(col, steps);

        // BAYER DITHERING
        if (ps1_dither_amount > 0.0) {
            float dither = bayer_4x4(gl_FragCoord.xy) - 0.5;
            vec3 dithered = banded + dither * (1.0 / steps) * ps1_dither_amount;
            col = clamp(dithered, 0.0, 1.0);
        } else {
            col = banded;
        }
    }

    // Distance fog
    float fog = 0.0;
    if (ps1_fog_end > ps1_fog_start) {
        fog = clamp((v_depth - ps1_fog_start) / (ps1_fog_end - ps1_fog_start), 0.0, 1.0);
    }
    col = mix(col, ps1_fog_color, fog);

    p3d_FragColor = vec4(col, v_color.a);
}
