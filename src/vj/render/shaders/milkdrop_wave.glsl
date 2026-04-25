#version 130

/**
 * milkdrop_wave.glsl
 *
 * Audio waveform interference patterns — like watching sound waves
 * ripple across the screen. Bass creates concentric rings, mid creates
 * grid interference, treble creates fine detail ripples.
 */

uniform float u_time;
uniform float u_bass;
uniform float u_mid;
uniform float u_treble;
uniform float u_volume;
uniform float u_energy;
uniform float u_onset;
uniform vec2  u_resolution;

in vec2 texcoord;
out vec4 p3d_FragColor;

void main() {
    vec2 uv = texcoord * 2.0 - 1.0;
    float aspect = u_resolution.x / u_resolution.y;
    uv.x *= aspect;

    float dist = length(uv);
    float angle = atan(uv.y, uv.x);

    // --- Bass: concentric wave rings ---
    float bass_speed = u_time * (0.5 + u_bass * 0.8);
    float ring1 = sin(dist * 12.0 - bass_speed * 4.0);
    float ring2 = sin(dist * 20.0 - bass_speed * 6.0 + u_bass * 2.0);
    float bass_wave = (ring1 * 0.6 + ring2 * 0.4) * (0.5 + u_bass * 0.5);

    // Onset: rings explode outward
    if (u_onset > 0.5) {
        float blast = sin(dist * 30.0 - u_time * 20.0);
        bass_wave += blast * 0.5 * (1.0 - dist * 0.5);
    }

    // --- Mid: grid interference ---
    float mid_speed = u_time * (0.3 + u_mid * 0.4);
    float grid_x = sin(uv.x * 15.0 + mid_speed * 3.0 + dist * 2.0);
    float grid_y = sin(uv.y * 15.0 + mid_speed * 2.0 + dist * 3.0);
    float grid_wave = (grid_x + grid_y) * 0.5 * (0.3 + u_mid * 0.7);

    // --- Treble: fine ripple detail ---
    float treb_speed = u_time * (1.0 + u_treble * 1.0);
    float ripple = sin(uv.x * 30.0 + angle * 6.0 + treb_speed * 5.0);
    ripple += cos(uv.y * 30.0 - angle * 4.0 + treb_speed * 4.0);
    ripple *= 0.5 * (0.2 + u_treble * 0.8);

    // Combine waves
    float wave = bass_wave * 0.5 + grid_wave * 0.3 + ripple * 0.2;
    wave = wave * 0.5 + 0.5;  // normalize to ~0..1

    // Colour: audio-locked HSV
    float hue = wave * 0.5 + dist * 0.15 + u_time * (0.02 + u_mid * 0.04);
    float sat = 0.6 + u_bass * 0.4 + u_volume * 0.2;
    float val = 0.4 + wave * 0.4 + u_energy * 0.3;

    // HSV to RGB
    float h = fract(hue);
    float s2 = clamp(sat, 0.0, 1.0);
    float v2 = clamp(val, 0.2, 1.0);

    vec3 rgb;
    float fi = floor(h * 6.0);
    float f = h * 6.0 - fi;
    float p = v2 * (1.0 - s2);
    float q = v2 * (1.0 - s2 * f);
    float t = v2 * (1.0 - s2 * (1.0 - f));

    if (fi < 1.0)      rgb = vec3(v2, t, p);
    else if (fi < 2.0) rgb = vec3(q, v2, p);
    else if (fi < 3.0) rgb = vec3(p, v2, t);
    else if (fi < 4.0) rgb = vec3(p, q, v2);
    else if (fi < 5.0) rgb = vec3(t, p, v2);
    else               rgb = vec3(v2, p, q);

    // Extra bright rings on bass peaks
    float ring_bright = abs(ring1) * u_bass * 0.3;
    rgb += vec3(ring_bright);

    // Vignette
    float vig = 1.0 - dist * 0.35;
    rgb *= clamp(vig, 0.2, 1.0);

    p3d_FragColor = vec4(rgb, 1.0);
}
