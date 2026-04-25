"""MilkDrop-style GPU visualizer for vj-nano — CPU color animation fallback.

Uses a CardMaker quad (NO shaders) with CPU-driven color cycling that
responds to audio. This works 100% reliably on the Jetson — shader
investigation can continue later.

Phase 1: CPU color animation (working NOW).
Phase 2: GPU shader background (when shader issue is solved).
"""

from __future__ import print_function
import math


def _hsv_to_rgb(h, s, v):
    """HSV 0..1 to RGB 0..1."""
    if s == 0.0: return v, v, v
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q


class MilkDropRenderer(object):
    """Animated background via CPU color cycling — NO shaders needed."""

    def __init__(self, base):
        from panda3d.core import CardMaker
        self._base = base
        self._enabled = True
        self._energy = 0.0
        self._bass = 0.0
        self._onset = False

        # Billboarded CardMaker quad
        cm = CardMaker("milkdrop_quad")
        cm.setFrame(-1.0, 1.0, -1.0, 1.0)
        self._quad = base.render.attachNewNode(cm.generate())
        self._quad.setPos(0, 30, 2)
        self._quad.setBillboardPointEye()
        self._quad.setDepthWrite(False)
        self._quad.setTwoSided(True)
        self._quad.setScale(50)
        # Start with dark background color
        self._quad.setColor(0.06, 0.08, 0.10, 1.0)

        print("[milkdrop] CPU animated background ready")

    def set_enabled(self, val):
        self._enabled = val
        if self._quad is not None:
            self._quad.setScale(50 if val else 0.001)

    def next_preset(self):
        """Cycle brightness mode."""
        return "CPU Animated"

    @property
    def preset_label(self):
        return "CPU Animated"

    def update(self, bass, mid, treble, volume, onset, energy, dt):
        if self._quad is None or not self._enabled:
            return

        t = self._base.taskMgr.globalClock.getFrameTime()

        # Base hue slowly cycles with time, pulled by bass
        hue = (t * 0.03 + bass * 0.1) % 1.0

        # Saturation from energy, value from volume + bass
        sat = 0.3 + energy * 0.5 + bass * 0.2
        val = 0.15 + volume * 0.6 + bass * 0.25

        # Onset flash: spike brightness + shift hue
        if onset:
            val = min(1.0, val + 0.5)
            hue = (hue + 0.5) % 1.0

        sat = min(1.0, max(0.1, sat))
        val = min(1.0, max(0.06, val))

        r, g, b = _hsv_to_rgb(hue, sat, val)
        self._quad.setColor(r, g, b, 1.0)

    def destroy(self):
        if self._quad is not None:
            self._quad.removeNode()
            self._quad = None
