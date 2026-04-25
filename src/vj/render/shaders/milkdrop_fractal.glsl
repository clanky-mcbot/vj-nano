#version 130

/**
 * milkdrop_fractal.glsl
 *
 * FFT-driven fractal zoom — like watching a Mandelbrot set orbit to
 * the beat. Bass controls zoom speed, mid controls rotation, treble
 * controls colour cycling.
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

    // Center offset — bass pulses move the fractal centre
    vec2 centre = vec2(
        0.5 + 0.3 * sin(u_time * 0.2 + u_bass * 3.0),
        0.3 * cos(u_time * 0.17 + u_mid * 2.0)
    );

    // Bass-driven zoom — accelerates on beat
    float zoom_speed = 0.3 + u_bass * 0.7 + u_onset * 1.5;
    float zoom = exp(u_time * zoom_speed);
    zoom = min(zoom, 100.0);  // clamp to avoid float overflow

    // Mid-driven rotation
    float rot = u_time * (0.1 + u_mid * 0.2);
    float c = cos(rot);
    float s = sin(rot);
    vec2 z = vec2(
        uv.x * c - uv.y * s,
        uv.x * s + uv.y * c
    ) / zoom + centre;

    // Julia-style iteration with audio-driven seed
    vec2 c_julia = vec2(
        0.7885 + u_bass * 0.3 * sin(u_time * 0.1),
        0.2885 + u_treble * 0.3 * cos(u_time * 0.13)
    );

    float iter = 0.0;
    const float MAX_ITER = 64.0;

    for (float i = 0.0; i < MAX_ITER; i += 1.0) {
        float zx2 = z.x * z.x - z.y * z.y;
        float zy2 = 2.0 * z.x * z.y;
        z = vec2(zx2 + c_julia.x, zy2 + c_julia.y);
        if (dot(z, z) > 4.0) {
            iter = i;
            break;
        }
    }

    float n = iter / MAX_ITER;

    // Treble-driven colour cycling
    float hue = n + u_time * (0.03 + u_treble * 0.08);
    float sat = 0.7 + u_bass * 0.3;
    float val = 0.6 + (1.0 - n) * 0.4;

    // HSV to RGB
    float h = fract(hue);
    float s2 = clamp(sat, 0.0, 1.0);
    float v2 = clamp(val, 0.0, 1.0);

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

    // Glow around edges of fractal
    float glow = sin(n * 10.0) * 0.5 + 0.5;
    rgb += vec3(glow * 0.15 * u_bass);

    // Onset pulse: centre flash
    if (u_onset > 0.5) {
        float d = length(uv / zoom);
        rgb = mix(rgb, vec3(1.0, 0.8, 0.6), exp(-d * 8.0) * 0.5);
    }

    // Vignette
    float vig = 1.0 - length(uv) * 0.35;
    rgb *= clamp(vig, 0.2, 1.0);

    p3d_FragColor = vec4(rgb, 1.0);
}
