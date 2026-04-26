"""Retro 2000s Windows-XP-style visualizer effects for vj-nano.

Lightweight Panda3D Geom-based effects that react to audio features.
Designed for the Jetson Nano's Maxwell GPU — minimal overdraw, no shaders.

Python 3.6 compatible.
"""

from __future__ import print_function

import math
import random

import numpy as np

from panda3d.core import (
    Geom, GeomLinestrips, GeomLines, GeomPoints, GeomTriangles, GeomNode,
    GeomVertexData, GeomVertexFormat, GeomVertexWriter, NodePath,
    TextureStage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hsv_to_rgb(h, s, v):
    """Convert HSV 0..1 to RGB 0..1. Python 3.6 compatible (no colorsys needed)."""
    if s == 0.0:
        return v, v, v
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    return v, p, q


def _lerp(a, b, t):
    """Linear interpolate."""
    return a + (b - a) * t


def _blend_hsv(base_h, base_s, base_v, tint_rgb, strength):
    """Shift HSV toward the tint colour."""
    if tint_rgb is None or strength <= 0.0:
        return base_h, base_s, base_v
    import colorsys
    tr, tg, tb = tint_rgb
    th, ts, tv = colorsys.rgb_to_hsv(float(tr), float(tg), float(tb))
    h = _lerp(base_h, th, strength)
    s = min(1.0, _lerp(base_s, ts, strength * 0.5) + 0.1)
    v = min(1.0, _lerp(base_v, tv, strength * 0.3) + 0.05)
    return h, s, v


# ---------------------------------------------------------------------------
# 1. Waveform Ring — 3D oscilloscope line orbiting the character
# ---------------------------------------------------------------------------

class WaveformRing(object):
    """A 3D ring of audio waveform points that orbits the actor."""

    def __init__(self, parent, radius=6.0, n_points=128,
                 color_cycle_speed=0.3):
        # type: (NodePath, float, int, float) -> None
        self._radius = radius
        self._n = n_points
        self._color_speed = color_cycle_speed
        self._hue = 0.0

        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("wfring", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(self._n):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(1.0, 1.0, 1.0, 1.0)

        prim = GeomLinestrips(Geom.UHDynamic)
        prim.addConsecutiveVertices(0, self._n)

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("wfring")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setPos(0, 8, 2.8)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 0)
        self._bounds_frame = 0
        self._np.setRenderModeThickness(5)
        self._np.setAntialias(0)

    def update(self, waveform, dt, bpm=120.0, energy=0.5, onset=False, tint=None, intensity=1.0):
        # type: (np.ndarray, float, float, float, bool, Optional[Tuple[float,float,float]]) -> None
        self._hue = (self._hue + dt * self._color_speed) % 1.0

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        n = min(self._n, waveform.shape[0])
        rot_speed = (bpm / 60.0) * 0.5 if bpm > 0 else 1.0
        base_angle = self._hue * 4.0 * math.pi
        pulse = 1.0 + (3.0 if onset else 0.0)

        for i in range(n):
            frac = i / float(n)
            angle = base_angle + frac * 2.0 * math.pi

            amp = float(waveform[i]) * (3.0 + energy * 5.0) * pulse * intensity
            r = self._radius + amp

            x = r * math.cos(angle)
            y = r * math.sin(angle) * 0.3
            z = math.sin(angle * 3.0 + base_angle) * 0.6 * energy * pulse

            vwrite.setData3(x, y, z)

            h = (self._hue + frac) % 1.0
            h, s, v = _blend_hsv(h, 0.95, 1.0, tint, 0.4)
            r_col, g_col, b_col = _hsv_to_rgb(h, s, v)
            alpha = (0.7 + 0.3 * min(abs(amp) / 3.0, 1.0)) * intensity
            cwrite.setData4(r_col, g_col, b_col, min(alpha, 1.0))



        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()
# ---------------------------------------------------------------------------
# 2. Spectrum Bars 2D — bottom-screen equalizer (filled WMP look)
# ---------------------------------------------------------------------------

class SpectrumBars2D(object):
    """Filled vertical spectrum bars across the bottom of the screen.

    Rendered in aspect2d so they're always visible regardless of 3D camera.
    """

    def __init__(self, aspect2d, n_bars=32, height=0.30):
        # type: (NodePath, int, float) -> None
        self._n = n_bars
        self._height = height
        self._heights = [0.0] * n_bars
        self._peaks = [0.0] * n_bars
        self._fall = 2.0
        self._peak_fall = 1.0
        self._hue = 0.0

        # Each bar is a filled quad = 4 vertices, 2 triangles
        n_verts = n_bars * 4
        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("spec2d", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_verts):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(1.0, 1.0, 1.0, 1.0)

        prim = GeomTriangles(Geom.UHDynamic)
        for i in range(n_bars):
            b = i * 4
            prim.addVertices(b, b + 1, b + 2)
            prim.addVertices(b, b + 2, b + 3)

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("spec2d")
        node.addGeom(geom)
        self._np = aspect2d.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 5)

    @staticmethod
    def _xp_gradient(h_frac):
        # type: (float) -> tuple
        """Classic WMP green->yellow->orange->red gradient."""
        if h_frac < 0.20:
            r = 0.0
            g = 0.5 + h_frac * 2.5
            b = 0.0
        elif h_frac < 0.45:
            r = (h_frac - 0.20) * 4.0
            g = 1.0
            b = 0.0
        elif h_frac < 0.70:
            r = 1.0
            g = 1.0 - (h_frac - 0.45) * 4.0
            b = 0.0
        elif h_frac < 0.90:
            r = 1.0
            g = (h_frac - 0.70) * 2.0
            b = 0.0
        else:
            r = 1.0
            g = 0.4 - (h_frac - 0.90) * 4.0
            b = (h_frac - 0.90) * 5.0
        return (max(0.0, min(1.0, r)), max(0.0, min(1.0, g)),
                max(0.0, min(1.0, b)))

    def update(self, bass, mid, treble, dt, onset=False, tint=None, intensity=1.0):
        # type: (float, float, float, float, bool, Optional[Tuple[float,float,float]]) -> None
        self._hue = (self._hue + dt * 0.06) % 1.0

        # Noise gate: zero out when total energy is near silence
        total = bass + mid + treble
        if total < 0.003:
            bass = mid = treble = 0.0

        targets = []
        for i in range(self._n):
            frac = i / float(self._n - 1) if self._n > 1 else 0.5
            if frac < 0.33:
                t = bass * (1.0 + (0.33 - frac) * 2.0)
            elif frac < 0.66:
                t = mid
            else:
                t = treble * (1.0 + (frac - 0.66) * 2.0)
            if onset:
                t *= 1.8
            targets.append(min(t * self._height * 2.2 * intensity, self._height))

        for i in range(self._n):
            if targets[i] > self._heights[i]:
                self._heights[i] = targets[i]
            else:
                self._heights[i] = max(
                    0.0, self._heights[i] - dt * self._fall)
            # Peak caps
            if self._heights[i] >= self._peaks[i]:
                self._peaks[i] = self._heights[i]
            else:
                self._peaks[i] = max(
                    self._heights[i],
                    self._peaks[i] - dt * self._peak_fall)

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        x_start = -1.0
        x_step = 2.0 / self._n
        y_base = -0.92
        bar_w = x_step * 0.80

        for i in range(self._n):
            x = x_start + i * x_step + x_step * 0.5
            h = self._heights[i]

            # Quad: bottom-left, bottom-right, top-right, top-left
            vwrite.setData3(x - bar_w * 0.5, 0.0, y_base)
            vwrite.setData3(x + bar_w * 0.5, 0.0, y_base)
            vwrite.setData3(x + bar_w * 0.5, 0.0, y_base + h)
            vwrite.setData3(x - bar_w * 0.5, 0.0, y_base + h)

            h_frac = h / self._height if self._height > 0 else 0.0
            r, g, b = self._xp_gradient(h_frac)
            # Horizontal rainbow tint so adjacent bars have different colours
            bar_hue = (i / float(self._n)) * 0.35
            rh, gh, bh = _hsv_to_rgb(bar_hue, 0.6, 1.0)
            r = r * 0.7 + rh * 0.3
            g = g * 0.7 + gh * 0.3
            b = b * 0.7 + bh * 0.3
            # Blend with webcam tint
            if tint is not None:
                r, g, b = _lerp(r, tint[0], 0.25), _lerp(g, tint[1], 0.25), _lerp(b, tint[2], 0.25)
            alpha = 0.55 + min(h_frac, 1.0) * 0.45

            # Bottom darker
            cwrite.setData4(r * 0.3, g * 0.3, b * 0.3, alpha)
            cwrite.setData4(r * 0.3, g * 0.3, b * 0.3, alpha)
            # Top bright
            cwrite.setData4(r, g, b, min(alpha + 0.15, 1.0))
            cwrite.setData4(r, g, b, min(alpha + 0.15, 1.0))


# ---------------------------------------------------------------------------
# 3. Waveform Scope 2D — top oscilloscope line
# ---------------------------------------------------------------------------

class WaveformScope2D(object):
    """A 2D oscilloscope line at the top of the screen, WMP-style."""

    def __init__(self, aspect2d, n_points=256):
        # type: (NodePath, int) -> None
        self._n = n_points
        self._hue = 0.0

        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("scope2d", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_points):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(0.0, 1.0, 0.5, 1.0)

        prim = GeomLinestrips(Geom.UHDynamic)
        prim.addConsecutiveVertices(0, n_points)

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("scope2d")
        node.addGeom(geom)
        self._np = aspect2d.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 6)
        self._np.setRenderModeThickness(2)
        self._np.setAntialias(0)

    def update(self, waveform, dt, energy=0.0, tint=None, intensity=1.0):
        # type: (np.ndarray, float, float, Optional[Tuple[float,float,float]]) -> None
        self._hue = (self._hue + dt * 0.1) % 1.0

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        n = min(self._n, waveform.shape[0])
        y_base = 0.82
        y_amp = (0.18 + energy * 0.15) * intensity

        for i in range(n):
            x = -0.95 + 1.9 * (i / float(n - 1)) if n > 1 else 0.0
            amp = float(waveform[i])
            y = y_base + amp * y_amp
            vwrite.setData3(x, 0.0, y)

            # Cyan/green cycling color, blended with tint
            h = (self._hue + i * 0.002) % 1.0
            h, s, v = _blend_hsv(0.3 + h * 0.2, 0.8, 1.0, tint, 0.35)
            r, g, b = _hsv_to_rgb(h, s, v)
            alpha = (0.7 + 0.3 * abs(amp)) * intensity
            cwrite.setData4(r, g, b, min(alpha, 1.0))


# ---------------------------------------------------------------------------
# 4. Radial Spectrum — circular equalizer in top-right corner
# ---------------------------------------------------------------------------

class RadialSpectrum(object):
    """A circular spectrum visualizer drawn with thick radial lines."""

    def __init__(self, aspect2d, n_bars=24, radius=0.10):
        # type: (NodePath, int, float) -> None
        self._n = n_bars
        self._radius = radius
        self._heights = [0.0] * n_bars
        self._hue = 0.0

        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("radial", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_bars * 2):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(1.0, 1.0, 1.0, 1.0)

        prim = GeomLinestrips(Geom.UHDynamic)
        for i in range(n_bars):
            prim.addVertices(i * 2, i * 2 + 1)
            prim.closePrimitive()

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("radial")
        node.addGeom(geom)
        self._np = aspect2d.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 7)
        self._np.setRenderModeThickness(3)
        self._np.setAntialias(0)

    def update(self, bass, mid, treble, dt, onset=False, energy=0.0, tint=None, intensity=1.0):
        # type: (float, float, float, float, bool, float, Optional[Tuple[float,float,float]]) -> None
        self._hue = (self._hue + dt * 0.15) % 1.0
        cx, cz = 0.78, 0.68  # top-right center in aspect2d

        # Noise gate
        total = bass + mid + treble
        if total < 0.003:
            bass = mid = treble = 0.0

        targets = []
        for i in range(self._n):
            frac = i / float(self._n)
            if frac < 0.33:
                t = bass * (1.0 + (0.33 - frac) * 2.0)
            elif frac < 0.66:
                t = mid * (1.0 + abs(0.5 - frac) * 1.5)
            else:
                t = treble * (1.0 + (frac - 0.66) * 2.0)
            if onset:
                t *= 1.6
            targets.append(min(t * 0.30 * intensity, 0.25))

        for i in range(self._n):
            if targets[i] > self._heights[i]:
                self._heights[i] = targets[i]
            else:
                self._heights[i] = max(0.0, self._heights[i] - dt * 4.0)

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        for i in range(self._n):
            angle = i * 2.0 * math.pi / self._n + self._hue * 0.5
            h = self._heights[i] + self._radius
            # inner point
            vwrite.setData3(cx + self._radius * math.cos(angle), 0.0,
                           cz + self._radius * math.sin(angle))
            # outer point
            vwrite.setData3(cx + h * math.cos(angle), 0.0,
                           cz + h * math.sin(angle))

            hue = (self._hue + i / float(self._n)) % 1.0
            hue, s, v = _blend_hsv(hue, 0.9, 1.0, tint, 0.3)
            r, g, b = _hsv_to_rgb(hue, s, v)
            alpha = 0.5 + min(self._heights[i] * 6.0, 0.5)
            if onset:
                alpha = min(1.0, alpha + 0.3)
            cwrite.setData4(r, g, b, alpha)
            cwrite.setData4(r, g, b, alpha * 0.6)


# ---------------------------------------------------------------------------
# 5. Starfield — floating particles that pulse on beat
# ---------------------------------------------------------------------------

class Starfield(object):
    """A cloud of point particles floating around the scene."""

    def __init__(self, parent, n_stars=256, bounds=15.0):
        # type: (NodePath, int, float) -> None
        self._n = n_stars
        self._bounds = bounds
        self._stars = []

        for _ in range(n_stars):
            self._stars.append({
                "x": random.uniform(-bounds, bounds),
                "y": random.uniform(-bounds, bounds),
                "z": random.uniform(-2.0, 6.0),
                "vx": random.uniform(-0.3, 0.3),
                "vy": random.uniform(-0.3, 0.3),
                "vz": random.uniform(-0.1, 0.1),
                "base_alpha": random.uniform(0.3, 0.8),
                "phase": random.uniform(0.0, 2.0 * math.pi),
            })

        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("stars", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_stars):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(1.0, 1.0, 1.0, 1.0)

        prim = GeomPoints(Geom.UHDynamic)
        prim.addConsecutiveVertices(0, n_stars)

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("stars")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setPos(0, 8, 0.0)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 2)
        self._bounds_frame = 0
        self._np.setRenderModeThickness(3)

    def update(self, dt, onset=False, bpm=120.0, energy=0.0, tint=None, intensity=1.0):
        # type: (float, bool, float, float, Optional[Tuple[float,float,float]]) -> None
        pulse = 1.0
        if onset:
            pulse = 5.0
        speed_mul = 1.0 + energy * 5.0 * intensity

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        beat_freq = (bpm / 60.0) * 2.0 * math.pi if bpm > 0 else 4.0

        for s in self._stars:
            s["x"] += s["vx"] * dt * speed_mul
            s["y"] += s["vy"] * dt * speed_mul
            s["z"] += s["vz"] * dt * speed_mul
            s["phase"] += dt * beat_freq * 0.5

            for key in ("x", "y", "z"):
                if s[key] > self._bounds:
                    s[key] -= self._bounds * 2.0
                elif s[key] < -self._bounds:
                    s[key] += self._bounds * 2.0

            vwrite.setData3(s["x"], s["y"], s["z"])

            twinkle = 0.7 + 0.3 * math.sin(s["phase"])
            alpha = min(s["base_alpha"] * twinkle * pulse * intensity, 1.0)
            brightness = 0.6 + 0.4 * twinkle * pulse
            # Slight color variation per star, blended with tint
            hue = (s["phase"] * 0.1) % 1.0
            hue, sat, val = _blend_hsv(hue, 0.4, 1.0, tint, 0.25)
            r, g, b = _hsv_to_rgb(hue, sat, val)
            cwrite.setData4(
                brightness * (0.8 + 0.2 * r),
                brightness * (0.8 + 0.2 * g),
                brightness * (0.8 + 0.2 * b),
                alpha,
            )



        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()
# ---------------------------------------------------------------------------
# 5. Grid Floor — retro wireframe plane that pulses on kick
# ---------------------------------------------------------------------------

class GridFloor(object):
    """A perspective grid on the floor, like Tron or old demoscene intros."""

    def __init__(self, parent, size=20.0, divisions=20):
        # type: (NodePath, float, int) -> None
        self._size = size
        self._divs = divisions
        self._pulse = 0.0

        n_lines = (divisions + 1) * 2
        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("grid", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_lines * 2):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(0.2, 0.3, 0.4, 0.5)

        prim = GeomLinestrips(Geom.UHDynamic)
        for i in range(n_lines):
            prim.addVertices(i * 2, i * 2 + 1)
            prim.closePrimitive()

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("grid")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setPos(0, 8, -2.0)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 3)
        self._bounds_frame = 0
        self._np.setRenderModeThickness(2)

    def update(self, dt, bass=0.0, onset=False, bpm=120.0, tint=None, intensity=1.0):
        # type: (float, float, bool, float, Optional[Tuple[float,float,float]]) -> None
        if onset:
            self._pulse = 1.0 + bass * 2.0
        self._pulse = max(0.0, self._pulse - dt * 2.5)

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        half = self._size / 2.0
        pulse_r = 0.15 + self._pulse * 0.9 * intensity
        pulse_g = 0.25 + self._pulse * 1.0 * intensity
        pulse_b = 0.35 + self._pulse * 1.2 * intensity

        # Blend pulse colour with webcam tint
        if tint is not None:
            pulse_r = _lerp(pulse_r, tint[0], 0.35)
            pulse_g = _lerp(pulse_g, tint[1], 0.35)
            pulse_b = _lerp(pulse_b, tint[2], 0.35)

        for i in range(self._divs + 1):
            y = -half + i * (self._size / self._divs)
            z_far = -1.5 + (y + half) / self._size * 0.5 + self._pulse * 0.5
            z_near = -2.0 - self._pulse * 0.3

            vwrite.setData3(-half, y, z_near)
            dist = abs(y) / half
            alpha = (0.3 + (1.0 - dist) * 0.3 + self._pulse * 0.4) * intensity
            cwrite.setData4(pulse_r * (1.0 - dist * 0.5),
                           pulse_g * (1.0 - dist * 0.3),
                           pulse_b, min(alpha, 1.0))

            vwrite.setData3(half, y, z_far)
            cwrite.setData4(pulse_r, pulse_g, pulse_b, min(alpha * 0.7, 1.0))

        for i in range(self._divs + 1):
            x = -half + i * (self._size / self._divs)
            vwrite.setData3(x, -half, -2.0 - self._pulse * 0.3)
            dist = abs(x) / half
            alpha = (0.2 + (1.0 - dist) * 0.2 + self._pulse * 0.3) * intensity
            cwrite.setData4(pulse_r * 0.8, pulse_g, pulse_b * 1.2,
                           min(alpha, 1.0))

            vwrite.setData3(x, half, -1.5 + self._pulse * 0.5)
            cwrite.setData4(pulse_r, pulse_g, pulse_b, min(alpha * 0.6, 1.0))



        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()
# ---------------------------------------------------------------------------
# 6. Spectrum Cylinder 3D — circular equalizer surrounding the actor
#    (inspired by TempoVis 3D spectrum display)
# ---------------------------------------------------------------------------

class SpectrumCylinder3D(object):
    """3D spectrum bars arranged in a rotating cylinder around the actor."""

    def __init__(self, parent, n_bars=40, radius=5.0, height=4.0):
        # type: (NodePath, int, float, float) -> None
        self._n = n_bars
        self._radius = radius
        self._height = height
        self._heights = [0.0] * n_bars
        self._angle = 0.0
        self._hue = 0.0

        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("cyl", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        # Each bar = 2 vertices (line)
        for _ in range(n_bars * 2):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(1.0, 1.0, 1.0, 1.0)

        prim = GeomLinestrips(Geom.UHDynamic)
        for i in range(n_bars):
            prim.addVertices(i * 2, i * 2 + 1)
            prim.closePrimitive()

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("cyl")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setPos(0, 8, 1.5)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 1)
        self._bounds_frame = 0
        self._np.setRenderModeThickness(3)

    def update(self, bass, mid, treble, dt, bpm=120.0, onset=False, energy=0.0, tint=None, intensity=1.0):
        # type: (float, float, float, float, float, bool, float, Optional[Tuple[float,float,float]]) -> None
        self._hue = (self._hue + dt * 0.08) % 1.0
        rot_speed = (bpm / 60.0) * 0.3 if bpm > 0 else 0.5
        self._angle += dt * rot_speed

        total = bass + mid + treble
        if total < 0.003:
            bass = mid = treble = 0.0

        targets = []
        for i in range(self._n):
            frac = i / float(self._n)
            if frac < 0.33:
                t = bass * (1.0 + (0.33 - frac) * 2.0)
            elif frac < 0.66:
                t = mid * (1.0 + abs(0.5 - frac) * 1.5)
            else:
                t = treble * (1.0 + (frac - 0.66) * 2.0)
            if onset:
                t *= 1.7
            targets.append(min(t * self._height * 1.5 * intensity, self._height))

        for i in range(self._n):
            if targets[i] > self._heights[i]:
                self._heights[i] = targets[i]
            else:
                self._heights[i] = max(0.0, self._heights[i] - dt * 3.0)

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        for i in range(self._n):
            theta = i * 2.0 * math.pi / self._n + self._angle
            h = self._heights[i]
            # Base point on cylinder floor
            bx = self._radius * math.cos(theta)
            by = self._radius * math.sin(theta)
            # Top point = amplitude upward
            tx = bx
            ty = by
            tz = h

            vwrite.setData3(bx, by, 0.0)
            vwrite.setData3(tx, ty, tz)

            hue = (self._hue + i / float(self._n)) % 1.0
            hue, s, v = _blend_hsv(hue, 0.85, 1.0, tint, 0.4)
            r, g, b = _hsv_to_rgb(hue, s, v)
            alpha = 0.55 + min(h * 1.2, 0.45)
            if onset:
                alpha = min(1.0, alpha + 0.25)
            cwrite.setData4(r, g, b, alpha)
            cwrite.setData4(r, g, b, alpha * 0.5)



        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()
# ---------------------------------------------------------------------------
# 7. Waveform Helix 3D — DNA-like spiral oscilloscope
#    (inspired by AVS superscope / parametric oscilloscope)
# ---------------------------------------------------------------------------

class WaveformHelix3D(object):
    """A 3D helix whose radius is modulated by the audio waveform."""

    def __init__(self, parent, n_points=160, radius=4.0, coils=2.0):
        # type: (NodePath, int, float, float) -> None
        self._n = n_points
        self._radius = radius
        self._coils = coils
        self._hue = 0.0

        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("helix", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_points):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(1.0, 1.0, 1.0, 1.0)

        prim = GeomLinestrips(Geom.UHDynamic)
        prim.addConsecutiveVertices(0, n_points)

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("helix")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setPos(0, 8, 1.8)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 1)
        self._bounds_frame = 0
        self._np.setRenderModeThickness(5)

    def update(self, waveform, dt, bpm=120.0, energy=0.5, onset=False, tint=None, intensity=1.0):
        # type: (np.ndarray, float, float, float, bool, Optional[Tuple[float,float,float]]) -> None
        self._hue = (self._hue + dt * 0.12) % 1.0
        rot_speed = (bpm / 60.0) * 0.2 if bpm > 0 else 0.2
        base_angle = self._hue * 2.0 * math.pi

        n = min(self._n, waveform.shape[0])
        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        pulse = 1.0 + (2.5 if onset else 0.0)
        amp_scale = (2.0 + energy * 4.0) * intensity

        for i in range(n):
            frac = i / float(n)
            angle = base_angle + frac * self._coils * 2.0 * math.pi
            amp = float(waveform[i]) * amp_scale * pulse
            r = self._radius + amp

            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = (frac - 0.5) * 5.0 + amp * 0.5

            vwrite.setData3(x, y, z)

            h = (self._hue + frac * 0.5) % 1.0
            h, s, v = _blend_hsv(h, 0.9, 1.0, tint, 0.35)
            r_col, g_col, b_col = _hsv_to_rgb(h, s, v)
            alpha = 0.75 + 0.25 * min(abs(amp) / 2.0, 1.0)
            cwrite.setData4(r_col, g_col, b_col, min(alpha, 1.0))



        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()
# ---------------------------------------------------------------------------
# 8. Beat Burst — radial lines explode from center on onset (AVS style)
# ---------------------------------------------------------------------------

class BeatBurst(object):
    """Pre-allocated pool of radial lines that burst outward on beat onsets."""

    def __init__(self, parent, n_rays=48):
        # type: (NodePath, int) -> None
        self._n = n_rays
        self._rays = []
        for _ in range(n_rays):
            self._rays.append({
                "age": 999.0,
                "max_age": random.uniform(0.4, 0.9),
                "angle": random.uniform(0.0, 2.0 * math.pi),
                "speed": random.uniform(4.0, 10.0),
                "length": random.uniform(0.5, 2.0),
            })

        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("burst", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_rays * 2):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(1.0, 1.0, 1.0, 0.0)

        prim = GeomLinestrips(Geom.UHDynamic)
        for i in range(n_rays):
            prim.addVertices(i * 2, i * 2 + 1)
            prim.closePrimitive()

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("burst")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setPos(0, 8, 1.0)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 0)
        self._bounds_frame = 0
        self._np.setRenderModeThickness(4)

    def _trigger(self, energy):
        # type: (float) -> None
        for r in self._rays:
            r["age"] = 0.0
            r["angle"] = random.uniform(0.0, 2.0 * math.pi)
            r["speed"] = random.uniform(4.0, 10.0) * (1.0 + energy * 2.0)
            r["length"] = random.uniform(0.5, 2.5) * (1.0 + energy)
            r["max_age"] = random.uniform(0.35, 0.8)

    def update(self, dt, onset=False, energy=0.0, tint=None, intensity=1.0):
        # type: (float, bool, float, Optional[Tuple[float,float,float]]) -> None
        if onset:
            self._trigger(energy)

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        for r in self._rays:
            r["age"] += dt
            age_frac = r["age"] / r["max_age"] if r["max_age"] > 0 else 1.0
            if age_frac >= 1.0 or r["age"] < 0.0:
                # Invisible / dead
                vwrite.setData3(0.0, 0.0, 0.0)
                vwrite.setData3(0.0, 0.0, 0.0)
                cwrite.setData4(0.0, 0.0, 0.0, 0.0)
                cwrite.setData4(0.0, 0.0, 0.0, 0.0)
                continue

            dist = r["age"] * r["speed"]
            ang = r["angle"]
            lx = r["length"] * math.cos(ang)
            ly = r["length"] * math.sin(ang)

            x0 = dist * math.cos(ang)
            y0 = dist * math.sin(ang)
            x1 = x0 + lx
            y1 = y0 + ly

            vwrite.setData3(x0, y0, 0.0)
            vwrite.setData3(x1, y1, 0.0)

            # Fade out with age
            alpha = (1.0 - age_frac) * 0.9 * intensity
            hue = (r["angle"] / (2.0 * math.pi) + 0.2) % 1.0
            hue, s, v = _blend_hsv(hue, 0.95, 1.0, tint, 0.45)
            r_col, g_col, b_col = _hsv_to_rgb(hue, s, v)
            cwrite.setData4(r_col, g_col, b_col, alpha)
            cwrite.setData4(r_col, g_col, b_col, alpha * 0.4)



        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()
# ---------------------------------------------------------------------------
# 9. Flash Overlay — full-screen colour flash on beat onset (classic AVS)
# ---------------------------------------------------------------------------

class FlashOverlay(object):
    """A full-screen quad that flashes brightly on onset, then decays."""

    def __init__(self, aspect2d):
        # type: (NodePath) -> None
        self._intensity = 0.0
        from panda3d.core import CardMaker
        cm = CardMaker("flash")
        cm.setFrame(-1.0, 1.0, -1.0, 1.0)
        self._np = aspect2d.attachNewNode(cm.generate())
        self._np.setColor(1.0, 1.0, 1.0, 0.0)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 20)
        # Start invisible
        self._np.hide()

    def update(self, dt, onset=False, energy=0.0, tint=None):
        # type: (float, bool, float, Optional[Tuple[float,float,float]]) -> None
        if onset:
            self._intensity = 0.35 + min(energy * 0.8, 0.5)
            self._np.show()
        else:
            self._intensity = max(0.0, self._intensity - dt * 3.5)

        if self._intensity <= 0.001:
            self._np.hide()
            return

        # Blend white flash with tint so the flash matches room colour
        if tint is not None:
            r = _lerp(1.0, tint[0], 0.35)
            g = _lerp(1.0, tint[1], 0.35)
            b = _lerp(1.0, tint[2], 0.35)
        else:
            r = g = b = 1.0
        self._np.setColor(r, g, b, self._intensity)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 11. Plasma Background -- animated sin/cos interference (classic AVS)
# ---------------------------------------------------------------------------

class PlasmaBackground(object):
    """A low-res animated plasma texture displayed behind the actor."""

    def __init__(self, parent, width=64, height=64):
        self._w = width
        self._h = height
        self._t = 0.0

        from panda3d.core import Texture, CardMaker
        self._tex = Texture("plasma")
        self._tex.setup2dTexture(width, height, Texture.TUnsignedByte, Texture.FRgb)

        cm = CardMaker("plasma_plane")
        cm.setFrame(-40, 40, -25, 25)
        self._np = parent.attachNewNode(cm.generate())
        self._np.setTexture(self._tex)
        self._np.setPos(0, 30, 2)
        self._np.setBillboardPointEye()
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", -20)
        self._np.setColor(1, 1, 1, 0.15)  # subtle
        # Flip V for OpenCV->OpenGL coordinate mismatch
        self._np.setTexScale(TextureStage.getDefault(), 1, -1)

    def update(self, dt, energy=0.0, onset=False, tint=None, intensity=1.0):
        self._t += dt * (1.0 + energy * 3.0)
        t = self._t
        w, h = self._w, self._h

        xs = np.linspace(0, 4.0 * np.pi, w)
        ys = np.linspace(0, 4.0 * np.pi, h)
        xg, yg = np.meshgrid(xs, ys)

        v1 = np.sin(xg * 0.5 + t * 1.2)
        v2 = np.sin(yg * 0.7 - t * 0.9)
        v3 = np.sin((xg + yg) * 0.3 + t * 0.6)
        v4 = np.sin(np.sqrt(xg**2 + yg**2 + 1.0) + t * 1.5)
        plasma = (v1 + v2 + v3 + v4) * 0.25

        if onset:
            cx, cy = w // 2, h // 2
            yy, xx = np.ogrid[-cx:h-cx, -cy:w-cy]
            ring = np.exp(-(xx**2 + yy**2) / (w * 0.15) ** 2)
            plasma += ring * 0.8

        hue = (plasma + 1.0) * 0.5
        if tint is not None:
            import colorsys
            th, ts, tv = colorsys.rgb_to_hsv(float(tint[0]), float(tint[1]), float(tint[2]))
            hue = (hue * 0.7 + th * 0.3) % 1.0

        sat = 0.6 + energy * 0.4
        val = 0.4 + (plasma + 1.0) * 0.25 * intensity
        val = np.clip(val, 0, 1)

        h6 = hue * 6.0
        i = np.floor(h6).astype(np.int32) % 6
        f = h6 - np.floor(h6)
        p = val * (1.0 - sat)
        q = val * (1.0 - sat * f)
        t_col = val * (1.0 - sat * (1.0 - f))

        r = np.choose(i, [val, q, p, p, t_col, val])
        g = np.choose(i, [t_col, val, val, q, p, p])
        b = np.choose(i, [p, p, t_col, val, val, q])

        rgb = np.stack([r, g, b], axis=2).astype(np.float32)
        rgb = np.power(rgb, 0.85)
        rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        self._tex.setRamImage(rgb_uint8.tobytes())


# ---------------------------------------------------------------------------
# 12. Superscope 3D -- parametric waveform curve (AVS classic)
# ---------------------------------------------------------------------------

class Superscope3D(object):
    """A 3D rotating parametric curve driven by the audio waveform."""

    def __init__(self, parent, n_points=200):
        self._n = n_points
        self._hue = 0.0
        self._angle = 0.0

        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("superscope", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_points):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(1.0, 1.0, 1.0, 1.0)

        prim = GeomLinestrips(Geom.UHDynamic)
        prim.addConsecutiveVertices(0, n_points)

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("superscope")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setPos(0, 8, 2.0)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 0)
        self._bounds_frame = 0
        self._np.setRenderModeThickness(4)

    def update(self, waveform, dt, bpm=120.0, energy=0.5, onset=False, tint=None, intensity=1.0):
        self._hue = (self._hue + dt * 0.15) % 1.0
        rot_speed = (bpm / 60.0) * 0.4 if bpm > 0 else 0.4
        self._angle += dt * rot_speed

        n = min(self._n, waveform.shape[0])
        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        pulse = 1.0 + (3.0 if onset else 0.0)
        amp_scale = (2.5 + energy * 5.0) * intensity * pulse
        a = self._angle

        for i in range(n):
            frac = i / float(n)
            amp = float(waveform[i]) * amp_scale

            x = (4.0 + amp * 0.6) * math.cos(frac * 4.0 * math.pi + a)
            y = (3.0 + amp * 0.4) * math.sin(frac * 4.0 * math.pi + a)
            z = amp * 1.5 + math.sin(frac * 8.0 * math.pi + self._hue * 6.0) * 0.8

            vwrite.setData3(x, y, z)

            h = (self._hue + frac * 0.6 + (1.0 if onset else 0.0) * 0.3) % 1.0
            h, s, v = _blend_hsv(h, 0.9, 1.0, tint, 0.4)
            r, g, b = _hsv_to_rgb(h, s, v)
            alpha = (0.6 + 0.4 * abs(amp) / amp_scale if amp_scale > 0 else 0.6) * intensity
            cwrite.setData4(r, g, b, min(alpha, 1.0))


        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()
# ---------------------------------------------------------------------------
# 10. Spectrum Waterfall 3D — TempoVis-style topographic frequency grid
# ---------------------------------------------------------------------------

class SpectrumWaterfall3D(object):
    """A 3D wireframe waterfall where spectrum history recedes into the scene.

    Each frame pushes a new spectrum slice; older slices move backward in Y
    and fade out.  Rainbow-coloured across frequency — the signature TempoVis look.
    """

    def __init__(self, parent, n_bins=32, n_history=20, width=14.0, depth=10.0, height=4.0):
        # type: (NodePath, int, int, float, float, float) -> None
        self._n_bins = n_bins
        self._n_hist = n_history
        self._width = width
        self._depth = depth
        self._height = height
        self._history = [np.zeros(n_bins, dtype=np.float32) for _ in range(n_history)]
        self._write_idx = 0
        self._hue = 0.0

        n_verts = n_history * n_bins
        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("waterfall", fmt, Geom.UHDynamic)
        self._vwrite_init = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite_init = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_verts):
            self._vwrite_init.addData3(0.0, 0.0, 0.0)
            self._cwrite_init.addData4(1.0, 1.0, 1.0, 1.0)

        prim = GeomLinestrips(Geom.UHDynamic)
        for i in range(n_history):
            prim.addConsecutiveVertices(i * n_bins, n_bins)

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("waterfall")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        self._np.setPos(0, 8, -2.0)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 1)
        self._bounds_frame = 0
        self._np.setRenderModeThickness(2)
        self._np.setAntialias(0)

    def update(self, bass, mid, treble, dt, bpm=120.0, onset=False, energy=0.0, tint=None, intensity=1.0):
        # type: (float, float, float, float, float, bool, float, Optional[Tuple[float,float,float]]) -> None
        self._hue = (self._hue + dt * 0.06) % 1.0

        spec = np.zeros(self._n_bins, dtype=np.float32)
        for i in range(self._n_bins):
            frac = i / float(self._n_bins - 1) if self._n_bins > 1 else 0.5
            if frac < 0.33:
                spec[i] = bass * (1.0 + (0.33 - frac) * 2.0)
            elif frac < 0.66:
                spec[i] = mid * (1.0 + abs(0.5 - frac) * 1.5)
            else:
                spec[i] = treble * (1.0 + (frac - 0.66) * 2.0)
            if onset:
                spec[i] *= 1.5

        self._history[self._write_idx] = spec
        self._write_idx = (self._write_idx + 1) % self._n_hist

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        for h in range(self._n_hist):
            idx = (self._write_idx - 1 - h) % self._n_hist
            hist = self._history[idx]
            y = h * (self._depth / self._n_hist)
            alpha = (1.0 - h / self._n_hist) * 0.75 * intensity

            for i in range(self._n_bins):
                frac = i / float(self._n_bins - 1) if self._n_bins > 1 else 0.5
                x = (frac - 0.5) * self._width
                z = hist[i] * self._height

                vwrite.setData3(x, y, z)

                hue = (frac + self._hue) % 1.0
                hue, s, v = _blend_hsv(hue, 0.9, 1.0, tint, 0.3)
                r, g, b = _hsv_to_rgb(hue, s, v)
                cwrite.setData4(r, g, b, alpha)

        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()

# ---------------------------------------------------------------------------
# 11. Scanline Overlay — subtle CRT horizontal lines (aspect2d)
# ---------------------------------------------------------------------------

class ScanlineOverlay(object):
    """Horizontal dark lines across the screen for a retro CRT feel."""

    def __init__(self, aspect2d, n_lines=72):
        # type: (NodePath, int) -> None
        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("scanlines", fmt, Geom.UHDynamic)
        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        for i in range(n_lines):
            y = -1.05 + 2.1 * (i + 0.5) / n_lines
            vwrite.addData3(-1.2, 0.0, y)
            vwrite.addData3(1.2, 0.0, y)
            cwrite.addData4(0.0, 0.0, 0.0, 0.10)
            cwrite.addData4(0.0, 0.0, 0.0, 0.10)

        prim = GeomLinestrips(Geom.UHDynamic)
        for i in range(n_lines):
            prim.addVertices(i * 2, i * 2 + 1)
            prim.closePrimitive()

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("scanlines")
        node.addGeom(geom)
        self._np = aspect2d.attachNewNode(node)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 50)

    def update(self, dt=0.0):
        # type: (float) -> None
        pass

# 12. RetroVisualizer — manager that owns all effects
# ---------------------------------------------------------------------------

class RetroVisualizer(object):
    """Container for all 2000s-style background effects."""

    def __init__(self, render_root, aspect2d):
        # type: (NodePath, NodePath) -> None
        self._root = render_root.attachNewNode("retro-fx")

        self._ring = WaveformRing(self._root, radius=6.0, n_points=128)
        self._stars = Starfield(self._root, n_stars=120, bounds=12.0)
        self._grid = GridFloor(self._root, size=24.0, divisions=16)

        # New 3D effects inspired by TempoVis & AVS
        self._cylinder = SpectrumCylinder3D(self._root, n_bars=24, radius=5.0, height=3.5)
        self._helix = WaveformHelix3D(self._root, n_points=100, radius=4.0, coils=2.5)
        self._burst = BeatBurst(self._root, n_rays=32)

        # 2D overlays \u2014 always visible
        self._spec2d = SpectrumBars2D(aspect2d, n_bars=24, height=0.28)
        self._scope2d = WaveformScope2D(aspect2d, n_points=160)
        self._radial = RadialSpectrum(aspect2d, n_bars=20, radius=0.10)

        # Motion-diff background plane
        self._motion = MotionDiff(render_root, width=160, height=120)

        # New AVS-inspired effects
        self._plasma = PlasmaBackground(render_root, width=64, height=64)
        self._superscope = Superscope3D(self._root, n_points=120)

        # Demoscene water + vortex
        self._water = WaterPlane(self._root, size=30.0, divisions=20)
        self._vortex = VortexParticles(self._root, n_particles=100)

        # TempoVis-inspired + retro CRT
        self._waterfall = SpectrumWaterfall3D(self._root, n_bins=28, n_history=18, width=12.0, depth=8.0, height=3.5)
        self._scanlines = ScanlineOverlay(aspect2d, n_lines=64)

        # Effect toggles & intensities
        self.enabled = {
            "ring": False,
            "stars": True,    # nice background feel
            "grid": True,     # ground reference
            "cylinder": False,
            "helix": False,
            "burst": False,
            "spec2d": False,
            "scope2d": False,
            "radial": False,
            "motion": False,
            "plasma": False,
            "superscope": False,
            "water": False,
            "vortex": False,
            "waterfall": False,
            "scanlines": False,
            "trail": False,
            "glow": False,
        }
        self.intensity = {k: 1.0 for k in self.enabled}

        # Mapping for NodePath show/hide
        self._fx_map = {
            "ring": self._ring,
            "stars": self._stars,
            "grid": self._grid,
            "cylinder": self._cylinder,
            "helix": self._helix,
            "burst": self._burst,
            "spec2d": self._spec2d,
            "scope2d": self._scope2d,
            "radial": self._radial,
            "motion": self._motion,
            "plasma": self._plasma,
            "superscope": self._superscope,
            "water": self._water,
            "vortex": self._vortex,
            "waterfall": self._waterfall,
            "scanlines": self._scanlines,
        }

    def set_enabled(self, key, state):
        # type: (str, bool) -> None
        self.enabled[key] = state
        fx = self._fx_map.get(key)
        if fx is not None and hasattr(fx, "_np"):
            # Use scale-based hiding -- hide()/show() is unreliable on
            # the Jetson's older Panda3D build (same issue as GUI panel).
            if state:
                fx._np.setScale(1)
            else:
                fx._np.setScale(0.001)

    def set_intensity(self, key, val):
        # type: (str, float) -> None
        self.intensity[key] = float(val)

    def update(self, feat, waveform, dt, tint=None):
        # type: (object, np.ndarray, float, Optional[Tuple[float,float,float]]) -> None
        bpm = feat.bpm if feat.bpm > 0 else 120.0
        energy = getattr(feat, "rms", 0.0)
        onset = getattr(feat, "onset", False)
        bass = getattr(feat, "bass", 0.0)
        mid = getattr(feat, "mid", 0.0)
        treble = getattr(feat, "treble", 0.0)

        # Ensure tint is a plain tuple for fast access
        tint_tuple = None  # type: Optional[Tuple[float,float,float]]
        if tint is not None and len(tint) >= 3:
            tint_tuple = (float(tint[0]), float(tint[1]), float(tint[2]))

        if self.enabled.get("ring", True):
            self._ring.update(waveform, dt, bpm=bpm, energy=energy,
                             onset=onset, tint=tint_tuple, intensity=self.intensity.get("ring", 1.0))
        if self.enabled.get("stars", True):
            self._stars.update(dt, onset=onset, bpm=bpm, energy=energy,
                              tint=tint_tuple, intensity=self.intensity.get("stars", 1.0))
        if self.enabled.get("grid", True):
            self._grid.update(dt, bass=bass, onset=onset, bpm=bpm,
                             tint=tint_tuple, intensity=self.intensity.get("grid", 1.0))
        if self.enabled.get("cylinder", True):
            self._cylinder.update(bass, mid, treble, dt, bpm=bpm, onset=onset,
                                  energy=energy, tint=tint_tuple,
                                  intensity=self.intensity.get("cylinder", 1.0))
        if self.enabled.get("helix", True):
            self._helix.update(waveform, dt, bpm=bpm, energy=energy,
                              onset=onset, tint=tint_tuple,
                              intensity=self.intensity.get("helix", 1.0))
        if self.enabled.get("burst", True):
            self._burst.update(dt, onset=onset, energy=energy,
                              tint=tint_tuple, intensity=self.intensity.get("burst", 1.0))

        if self.enabled.get("spec2d", True):
            self._spec2d.update(bass, mid, treble, dt, onset=onset,
                               tint=tint_tuple, intensity=self.intensity.get("spec2d", 1.0))
        if self.enabled.get("scope2d", True):
            self._scope2d.update(waveform, dt, energy=energy,
                                tint=tint_tuple, intensity=self.intensity.get("scope2d", 1.0))
        if self.enabled.get("radial", True):
            self._radial.update(bass, mid, treble, dt, onset=onset, energy=energy,
                               tint=tint_tuple, intensity=self.intensity.get("radial", 1.0))

        if self.enabled.get("plasma", True):
            self._plasma.update(dt, energy=energy, onset=onset,
                               tint=tint_tuple, intensity=self.intensity.get("plasma", 1.0))

        if self.enabled.get("superscope", True):
            self._superscope.update(waveform, dt, bpm=bpm, energy=energy,
                                   onset=onset, tint=tint_tuple,
                                   intensity=self.intensity.get("superscope", 1.0))

        if self.enabled.get("water", True):
            self._water.update(dt, bass=bass, onset=onset, bpm=bpm,
                              energy=energy, tint=tint_tuple,
                              intensity=self.intensity.get("water", 1.0))

        if self.enabled.get("vortex", True):
            self._vortex.update(dt, energy=energy, onset=onset, bpm=bpm,
                               tint=tint_tuple, intensity=self.intensity.get("vortex", 1.0))

        if self.enabled.get("waterfall", True):
            self._waterfall.update(bass, mid, treble, dt, bpm=bpm, onset=onset,
                                   energy=energy, tint=tint_tuple,
                                   intensity=self.intensity.get("waterfall", 1.0))

        if self.enabled.get("scanlines", True):
            self._scanlines.update(dt)
# ---------------------------------------------------------------------------
# 10. MotionDiff — background plane showing only pixels that changed
#     between consecutive webcam frames (classic frame-difference motion)
# ---------------------------------------------------------------------------

class MotionDiff(object):
    """A fullscreen-ish plane behind the actor that displays only moving
    pixels from the webcam feed. Static pixels are blacked out.
    """

    def __init__(self, parent, width=160, height=120):
        # type: (NodePath, int, int) -> None
        self._w = width
        self._h = height
        self._prev = None  # type: Optional[np.ndarray]

        from panda3d.core import Texture, CardMaker
        self._tex = Texture("motion_diff")
        self._tex.setup2dTexture(width, height, Texture.TUnsignedByte, Texture.FRgba)

        cm = CardMaker("motion_plane")
        # Large card to fill camera view when billboarded behind actor
        cm.setFrame(-35, 35, -22, 22)
        self._np = parent.attachNewNode(cm.generate())
        self._np.setTexture(self._tex)
        self._np.setPos(0, 25, 2)
        self._np.setBillboardPointEye()
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", -10)
        self._np.setColor(1, 1, 1, 1)
        # OpenCV images are top-to-bottom; OpenGL textures sample bottom-to-top.
        # Flip V so the webcam image appears right-side up.
        self._np.setTexScale(TextureStage.getDefault(), 1, -1)

    def update(self, frame_rgb, dt, intensity=1.0):
        # type: (Optional[np.ndarray], float, float) -> None
        if frame_rgb is None:
            return

        h, w = frame_rgb.shape[:2]
        # Resize to our texture size if needed
        if w != self._w or h != self._h:
            import cv2
            frame_rgb = cv2.resize(frame_rgb, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
            h, w = self._h, self._w

        if self._prev is None or self._prev.shape != (h, w, 3):
            self._prev = frame_rgb.copy()
            # First frame: show nothing (all black)
            black = np.zeros((h, w, 4), dtype=np.uint8)
            self._tex.setRamImage(black.tobytes())
            return

        # Absolute diff in grayscale
        diff = np.abs(frame_rgb.astype(np.int16) - self._prev.astype(np.int16))
        diff_gray = diff.mean(axis=2)  # HxW

        # Threshold: lower threshold = more pixels shown when intensity is high
        base_thresh = 12
        threshold = max(3, int(base_thresh / max(intensity, 0.05)))
        mask = diff_gray > threshold

        # Build RGBA: original colour where motion, else black/transparent
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = frame_rgb
        # Alpha: full where motion, else zero
        rgba[:, :, 3] = (mask.astype(np.uint8)) * 255

        self._tex.setRamImage(rgba.tobytes())
        self._prev = frame_rgb.copy()


# ---------------------------------------------------------------------------
# 13. Water Plane — rippling perspective grid driven by bass (demoscene classic)
# ---------------------------------------------------------------------------

class WaterPlane(object):
    """A perspective grid that undulates like water, driven by audio."""

    def __init__(self, parent, size=30.0, divisions=32):
        self._size = size
        self._divs = divisions
        self._t = 0.0
        self._ripple_phase = 0.0

        n_verts = (divisions + 1) * (divisions + 1)
        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("water", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_verts):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(0.2, 0.4, 0.6, 0.4)

        prim = GeomTriangles(Geom.UHDynamic)
        for row in range(divisions):
            for col in range(divisions):
                i = row * (divisions + 1) + col
                prim.addVertices(i, i + 1, i + divisions + 1)
                prim.addVertices(i + 1, i + divisions + 2, i + divisions + 1)

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("water")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setPos(0, 8, -3.0)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 3)
        self._bounds_frame = 0
        self._np.setTwoSided(True)

    def update(self, dt, bass=0.0, onset=False, bpm=120.0, energy=0.0, tint=None, intensity=1.0):
        self._t += dt
        self._ripple_phase += dt * (1.5 + energy * 4.0)
        t = self._t
        phase = self._ripple_phase

        # On kick: inject extra ripple energy
        ripple_amp = (0.06 + bass * 0.5) * intensity  # 5x smaller
        if onset:
            ripple_amp += 1.5

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        half = self._size / 2.0
        step = self._size / self._divs

        # Base colour — deep cyan/blue
        base_r, base_g, base_b = 0.1, 0.35, 0.55
        if tint is not None:
            base_r = _lerp(base_r, tint[0], 0.3)
            base_g = _lerp(base_g, tint[1], 0.3)
            base_b = _lerp(base_b, tint[2], 0.3)

        for row in range(self._divs + 1):
            z_world = -half + row * step
            z_dist = abs(z_world) / half
            for col in range(self._divs + 1):
                x_world = -half + col * step
                x_dist = abs(x_world) / half

                # Distance from center for radial ripple
                dist = math.sqrt(x_world * x_world + z_world * z_world)

                # Multiple interfering waves
                h = (
                    math.sin(dist * 0.8 - phase * 2.0) * 0.5 +
                    math.sin(x_world * 0.5 + phase * 1.3) * 0.3 +
                    math.sin(z_world * 0.4 - phase * 0.9) * 0.3 +
                    math.sin(dist * 1.5 - phase * 3.0) * 0.2
                ) * ripple_amp

                # Height falls off at edges
                h *= (1.0 - max(x_dist, z_dist) * 0.4)

                vwrite.setData3(x_world, z_world, h)

                # Colour: brighter at peaks, darker at troughs
                h_norm = (h / ripple_amp + 1.0) * 0.5 if ripple_amp > 0.01 else 0.5
                r = base_r * (0.5 + h_norm * 0.8)
                g = base_g * (0.6 + h_norm * 0.7)
                b = base_b * (0.8 + h_norm * 0.5)
                alpha = (0.25 + h_norm * 0.35) * intensity
                cwrite.setData4(r, g, b, min(alpha, 0.75))



        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()
# ---------------------------------------------------------------------------
# 14. Vortex Particles — spiralling swarm driven by treble / energy
# ---------------------------------------------------------------------------

class VortexParticles(object):
    """A swarm of particles spiralling in a vortex, AVS-style."""

    def __init__(self, parent, n_particles=200):
        self._n = n_particles
        self._particles = []
        for i in range(n_particles):
            angle = random.uniform(0.0, 2.0 * math.pi)
            radius = random.uniform(0.5, 6.0)
            self._particles.append({
                "angle": angle,
                "radius": radius,
                "speed": random.uniform(0.5, 3.0),
                "rise": random.uniform(0.2, 1.5),
                "base_alpha": random.uniform(0.3, 0.9),
                "hue_offset": random.uniform(0.0, 1.0),
            })
        self._vortex_angle = 0.0
        self._hue = 0.0

        fmt = GeomVertexFormat.getV3c4()
        self._vdata = GeomVertexData("vortex", fmt, Geom.UHDynamic)
        self._vwrite = GeomVertexWriter(self._vdata, "vertex")
        self._cwrite = GeomVertexWriter(self._vdata, "color")

        for _ in range(n_particles):
            self._vwrite.addData3(0.0, 0.0, 0.0)
            self._cwrite.addData4(1.0, 1.0, 1.0, 1.0)

        prim = GeomPoints(Geom.UHDynamic)
        prim.addConsecutiveVertices(0, n_particles)

        geom = Geom(self._vdata)
        geom.addPrimitive(prim)
        node = GeomNode("vortex")
        node.addGeom(geom)
        self._np = parent.attachNewNode(node)
        # self._np.node().set_bounds(OmniBoundingVolume())  # TEMP DISABLED
        self._np.setPos(0, 8, 0.0)
        self._np.setTransparency(1)
        self._np.setDepthWrite(False)
        self._np.setBin("transparent", 2)
        self._bounds_frame = 0
        self._np.setRenderModeThickness(4)

    def update(self, dt, energy=0.0, onset=False, bpm=120.0, tint=None, intensity=1.0):
        self._hue = (self._hue + dt * 0.08) % 1.0
        rot_speed = (bpm / 60.0) * 1.5 if bpm > 0 else 1.5
        self._vortex_angle += dt * rot_speed

        vwrite = GeomVertexWriter(self._vdata, "vertex")
        cwrite = GeomVertexWriter(self._vdata, "color")

        pulse = 1.0
        if onset:
            pulse = 2.5
        speed_mul = (1.0 + energy * 6.0) * pulse * intensity

        for p in self._particles:
            p["angle"] += dt * p["speed"] * speed_mul * 0.3
            # Spiral inward when energy is high, outward when low
            target_r = 2.0 + (1.0 - energy) * 4.0
            p["radius"] += (target_r - p["radius"]) * dt * 0.5

            a = p["angle"] + self._vortex_angle
            r = p["radius"]
            x = r * math.cos(a)
            y = r * math.sin(a)
            z = p["rise"] * 3.0 + math.sin(a * 3.0 + self._hue * 10.0) * 0.5

            vwrite.setData3(x, y, z)

            hue = (self._hue + p["hue_offset"] + r * 0.1) % 1.0
            hue, s, v = _blend_hsv(hue, 0.85, 1.0, tint, 0.35)
            r_col, g_col, b_col = _hsv_to_rgb(hue, s, v)
            alpha = p["base_alpha"] * (0.5 + energy * 0.5) * pulse * intensity
            cwrite.setData4(r_col, g_col, b_col, min(alpha, 1.0))




        self._bounds_frame += 1
        if self._bounds_frame % 3 == 0:
            self._np.node().markInternalBoundsStale()
# ---------------------------------------------------------------------------
# 15. MotionTrail — frame feedback with fade (classic AVS "fadeout")
# ---------------------------------------------------------------------------
