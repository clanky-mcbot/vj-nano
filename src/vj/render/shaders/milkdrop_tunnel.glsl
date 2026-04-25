#version 130

/**
 * milkdrop_tunnel.glsl
 *
 * Perspective tunnel effect with FFT-driven speed and colour.
 * Like flying through a hyperspace tube that pulses to the beat.
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

    // Bass-driven tunnel speed — jumps on beat
    float speed = 1.0 + u_bass * 2.0 + u_onset * 4.0;
    float z = u_time * speed + dist * 4.0;

    // Mid drives ring density, treble drives wave distortion
    float density = 6.0 + u_mid * 4.0;
    float warp = 1.0 + u_treble * 0.5;

    // Main tunnel pattern — concentric rings with angular distortion
    float pattern = sin(dist * density * warp + u_time * speed * 0.5);
    pattern += sin(angle * 8.0 + z * 0.3 + u_bass * 2.0);
    pattern += cos(dist * 10.0 + u_time * 1.3);
    pattern = pattern * 0.33 + 0.5;  // normalize

    // Onset: tunnel contracts sharply
    float d2 = dist;
    if (u_onset > 0.5) {
        d2 *= 0.5 + u_onset * 0.5;
    }

    // Colour: HSV from angle + distance + audio
    float hue = angle / 6.2832 + u_time * (0.02 + u_treble * 0.05) + d2 * 0.2;
    float sat = 0.5 + u_bass * 0.5 + u_volume * 0.2;
    float val = 0.4 + pattern * 0.4 + u_energy * 0.3;

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

    // Brighten center (hyperspace star effect)
    float centre_glow = exp(-dist * 3.0);
    rgb += vec3(centre_glow * 0.5 * (1.0 + u_bass));

    // Vignette
    float vig = 1.0 - dist * 0.4;
    rgb *= clamp(vig, 0.1, 1.0);

    p3d_FragColor = vec4(rgb, 1.0);
}
