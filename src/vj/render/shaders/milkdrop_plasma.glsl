#version 130

/**
 * milkdrop_plasma.glsl
 *
 * Classic sin/cos plasma interference with FFT-driven color cycling.
 * Replaces CPU-based starfield + waveform ring + radial spectrum.
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
    // UV in [-1, 1] with aspect correction
    vec2 uv = texcoord * 2.0 - 1.0;
    float aspect = u_resolution.x / u_resolution.y;
    uv.x *= aspect;

    // Bass-driven zoom pulse
    float zoom = 1.0 + u_bass * 2.0 + u_onset * 1.5;
    uv *= zoom;

    // Time scales per frequency band
    float t_bass = u_time * (0.15 + u_bass * 0.3);
    float t_mid  = u_time * (0.30 + u_mid  * 0.5);
    float t_treb = u_time * (0.50 + u_treble * 0.8);

    // Three overlapping plasma waves
    float p1 = sin(uv.x * 3.0 + t_bass) + cos(uv.y * 4.0 + t_bass * 0.7);
    float p2 = sin((uv.x + uv.y) * 5.0 + t_mid) + cos((uv.y - uv.x) * 6.0 + t_mid * 0.8);
    float p3 = sin(uv.x * 7.0 + uv.y * 3.0 + t_treb) + cos(uv.x * 9.0 - uv.y * 5.0 + t_treb * 0.6);

    // Combine with energy weighting
    float plasma = p1 * (0.5 + u_bass * 0.3) + p2 * (0.3 + u_mid * 0.2) + p3 * (0.2 + u_treble * 0.15);
    plasma = plasma * 0.33 + 0.5;  // normalize to ~0..1

    // Bass drives saturation, mid drives hue speed, treble drives value
    float hue = plasma + u_time * (0.04 + u_mid * 0.06);
    float sat = 0.6 + u_bass * 0.4 + u_onset * 0.3;
    float val = 0.5 + u_volume * 0.5 + u_energy * 0.3;

    // HSV to RGB (fast 6-point lookup)
    float h = fract(hue);
    float s = clamp(sat, 0.0, 1.0);
    float v = clamp(val, 0.0, 1.0);

    vec3 rgb;
    float i = floor(h * 6.0);
    float f = h * 6.0 - i;
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));

    if (i < 1.0)      rgb = vec3(v, t, p);
    else if (i < 2.0) rgb = vec3(q, v, p);
    else if (i < 3.0) rgb = vec3(p, v, t);
    else if (i < 4.0) rgb = vec3(p, q, v);
    else if (i < 5.0) rgb = vec3(t, p, v);
    else              rgb = vec3(v, p, q);

    // Onset flash: white burst
    if (u_onset > 0.5) {
        float flash = 1.0 - length(uv) * 0.5;
        rgb = mix(rgb, vec3(1.0), flash * 0.4);
    }

    // Subtle vignette
    float vig = 1.0 - length(uv) * 0.3;
    rgb *= clamp(vig, 0.3, 1.0);

    p3d_FragColor = vec4(rgb, 1.0);
}
