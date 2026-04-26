"""Moses shield-face — SVG rendered via Cairo + Rsvg to Panda3D texture.

The SVG is rendered to a PNG surface using librsvg, uploaded as a
Panda3D Texture, and displayed on a CardMaker quad.

For audio-reactivity: the SVG XML is modified (eye/mouth params),
re-rendered each frame. Simple static quad for eyes/mouth as fallback.
"""

from __future__ import print_function

import math
import os
import tempfile

import numpy as np
import time

import cairo
import gi
gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg

from panda3d.core import (
    CardMaker, NodePath, TransparencyAttrib,
    Texture, GeomVertexFormat, GeomVertexData, GeomVertexWriter,
    Geom, GeomNode, GeomTriangles,
    Shader, ShaderAttrib,
)


# ═══════════════════════════════════════════════════════════════════════
# SVG TEMPLATE  —  {eye_angle} and {mouth_open} are substituted per frame
# ═══════════════════════════════════════════════════════════════════════

_SVG_TEMPLATE = r"""<svg width="655" height="755" viewBox="0 0 655 755" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="mosesGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#FF4D00;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#FF9900;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#FFD700;stop-opacity:1" />
    </linearGradient>
  </defs>

  <rect width="655" height="755" fill="#060610"/>
  <path d="M546.572 511.5L315.572 754L81.0724 511.5C59.9058 379.667 14.6725 111.1 3.07245 91.5C-8.52755 71.9 21.9058 58.3333 38.5725 54L305.572 0.5H387.072C454.406 16.1667 596.872 48.8 628.072 54C659.272 59.2 656.406 81.1667 651.072 91.5L546.572 511.5Z" 
        fill="url(#mosesGrad)"/>

  <g fill="#331100" fill-opacity="0.7" stroke="#331100" stroke-width="1" stroke-opacity="0.5">
    <path d="M366.072 195.5C444.072 171.5 595.072 115 595.072 115C546.105 108.802 457.572 131.5 366.072 195.5Z" />
    <path d="M27.5724 87.5C105.072 120.5 270.572 218 257.072 205.5C210.636 162.504 95.0724 80.5 27.5724 87.5Z" />
    
    <path d="M474.572 239.5C472.572 206.5 526.172 210.6 524.572 245" fill="none" stroke-width="3"/>
    <path d="M490.572 242C489.072 231 508.772 227.3 505.572 244.5" />
    <path d="M143.572 246.5C139.572 231.5 162.072 229.5 159.072 244.5" />
    <path d="M126.572 248C119.072 218.5 171.072 203.5 175.072 240.5" fill="none" stroke-width="3"/>
    <path d="M53.0724 239C53.4726 239.121 53.8755 239.242 54.2811 239.361M54.2811 239.361C104.052 253.994 194.667 253.422 243.072 207C212.545 193.241 132.048 180.449 54.2811 239.361Z" />
    
    <path d="M396.572 208C444.072 234 515.072 261 580.072 234C537.572 194 462.172 187.2 396.572 208Z" />
    
    <path d="M389.572 374.5L367.572 348.5C369.406 347.5 378.372 346.7 399.572 351.5C416.372 369.9 399.072 377 389.572 374.5Z" />
    <path d="M239.572 374L261.572 353C258.406 349.833 246.872 345.4 226.072 353C215.272 377.4 226.072 379.5 239.572 374Z" />
    
    <path d="M480.572 489.5C410.072 468.5 231.472 461 153.072 477C249.572 506.5 439.372 505.5 480.572 489.5Z" />
    <path d="M102.572 479C344.072 538 299.072 532.5 514.072 495C451.072 445.5 213.372 436.2 102.572 479Z" />
  </g>
</svg>"""


# ═══════════════════════════════════════════════════════════════════════
# MOSES FACE  (SVG-texture-based)
# ═══════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════
# HUE-ROTATION SHADER (GLSL 130 for Tegra X1)
# ═══════════════════════════════════════════════════════════════════════

_HUE_VERT = """#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec2 p3d_MultiTexCoord0;
out vec2 v_texcoord;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_texcoord = p3d_MultiTexCoord0;
}
"""

_HUE_FRAG = """#version 130
uniform sampler2D p3d_Texture0;
uniform float u_hue_shift;
in vec2 v_texcoord;
out vec4 p3d_FragColor;

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec4 tex = texture2D(p3d_Texture0, v_texcoord);
    vec3 hsv = rgb2hsv(tex.rgb);
    hsv.x = fract(hsv.x + u_hue_shift);
    p3d_FragColor = vec4(hsv2rgb(hsv), tex.a);
}
"""


class MosesFace(object):
    """Moses shield-face: SVG rendered to texture via Cairo+Rsvg."""

    TEX_W, TEX_H = 400, 460

    def __init__(self, render_parent):
        # type: (NodePath) -> None
        self._render_parent = render_parent
        self._svg_data = None
        self._dirty = True

        # Pre-render the SVG to bytes once (static base)
        try:
            self._base_rgba = self._render_svg_to_rgba(eye_angle=0.35, mouth_open=0.4)
            print("[moses] SVG rendered to texture OK")
        except Exception as e:
            print("[moses] SVG render FAILED:", e)
            import traceback; traceback.print_exc()
            raise

        # Create Panda3D texture
        self._tex = Texture("moses_svg")
        self._tex.setup2dTexture(
            self.TEX_W, self.TEX_H,
            Texture.T_unsigned_byte, Texture.F_rgba,
        )
        self._tex.setMinfilter(Texture.FT_linear)
        self._tex.setMagfilter(Texture.FT_linear)
        self._upload_texture(self._base_rgba)

        # CardMaker quad
        cm = CardMaker("moses_card")
        cm.setFrame(-3.25, 3.25, -3.75, 3.75)  # 6.5×7.5 world
        self._root = render_parent.attachNewNode(cm.generate())
        self._root.setTexture(self._tex)
        self._root.setTransparency(TransparencyAttrib.MPremultipliedAlpha)
        self._root.setPos(0, 10, 0.0)
        self._root.setScale(0.67)
        self._root.setTwoSided(True)
        self._root.setColor(1, 1, 1, 1)  # full white, texture provides color
        self._root.setAttrib(TransparencyAttrib.make(TransparencyAttrib.MPremultipliedAlpha))
        self._root.setBin("transparent", 0)

        # --- Glow aura (larger card behind face, beat-reactive) ---
        cm_glow = CardMaker("moses_aura")
        cm_glow.setFrame(-4.0, 4.0, -4.5, 4.5)
        self._glow = render_parent.attachNewNode(cm_glow.generate())
        self._glow.setPos(0, 10, 0.0)
        self._glow.setScale(0.67)
        self._glow.setY(-0.05)
        self._glow.setColor(1.0, 0.45, 0.08, 0.12)
        self._glow.setTransparency(TransparencyAttrib.MAlpha)
        self._glow.setDepthWrite(False)
        self._glow.setDepthTest(False)
        self._glow.setTwoSided(True)
        self._glow.setBin("transparent", -1)
        self._glow.hide()

        # --- Compile hue-rotation shader (safe: won't crash on failure) ---
        self._shader_ok = False
        self._hue_shift = 0.0
        try:
            s = Shader.make(Shader.SL_GLSL, _HUE_VERT, _HUE_FRAG)
            if s is not None:
                self._root.setShader(s)
                self._root.setShaderInput("u_hue_shift", 0.0)
                self._shader_ok = True
                self._shader = s
                print("[moses] hue-rotation shader compiled")
            else:
                print("[moses] Shader.make returned None (expected on Tegra GLES)")
        except Exception as e:
            print("[moses] shader skipped:", e)

        # --- Dynamic eye overlays (simple filled ellipses) ---
        self._build_eye_overlays()

        # Start hidden (hide root, pupils follow)
        self._root.hide()

    # ── SVG → RGBA bytes ──────────────────────────────────────────────
    def _render_svg_to_rgba(self, eye_angle=0.35, mouth_open=0.4):
        """Render the SVG to a W×H RGBA numpy array using Cairo+Rsvg."""
        svg_str = _SVG_TEMPLATE  # static template for now

        # Create Cairo image surface
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32,
                                     self.TEX_W, self.TEX_H)
        ctx = cairo.Context(surface)

        # Scale to fit
        ctx.scale(self.TEX_W / 655.0, self.TEX_H / 755.0)

        # Render SVG via Rsvg
        handle = Rsvg.Handle.new_from_data(svg_str.encode("utf-8"))
        handle.render_cairo(ctx)

        # Extract ARGB32 → RGBA numpy
        buf = surface.get_data()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(
            self.TEX_H, self.TEX_W, 4
        )
        # Cairo ARGB32 is BGRA in memory — Panda3D F_rgba expects this order
        surface.finish()

        return np.ascontiguousarray(arr[::-1, :, :])  # flip: Cairo top-down → Panda3D bottom-up

    def _upload_texture(self, rgba):
        """Upload RGBA numpy array to Panda3D texture."""
        self._tex.setRamImage(rgba.tobytes())

    # ── EYE OVERLAYS (dynamic Panda3D geometry on top of texture) ─────
    def _build_eye_overlays(self):
        """Create two simple card quads for golden pupils."""
        fmt = GeomVertexFormat.getV3c4()
        self._eye_vdata = GeomVertexData("moses_pupils", fmt, Geom.UHDynamic)
        vw = GeomVertexWriter(self._eye_vdata, "vertex")
        cw = GeomVertexWriter(self._eye_vdata, "color")
        # 2 quads × 4 verts = 8
        for _ in range(8):
            vw.addData3(0, 0.01, 0)
            cw.addData4f(0, 0, 0, 0)

        tris = GeomTriangles(Geom.UHDynamic)
        for q in range(2):
            b = q * 4
            tris.addVertices(b, b + 1, b + 2)
            tris.addVertices(b, b + 2, b + 3)

        geom = Geom(self._eye_vdata)
        geom.addPrimitive(tris)
        node = GeomNode("moses_pupils")
        node.addGeom(geom)
        self._pupil_np = self._root.attachNewNode(node)
        self._pupil_np.setTransparency(TransparencyAttrib.MAlpha)
        self._pupil_np.setDepthWrite(False)
        self._pupil_np.setDepthTest(False)
        self._pupil_np.setTwoSided(True)
        self._pupil_np.setBin("transparent", 5)

    def _update_pupils(self, pupil_scale):
        """Reposition golden pupil quads (floating above texture)."""
        # SVG eye centers: left ~(150, 244), right ~(500, 242)
        # Map to world coords on the CardMaker quad
        # CardMaker frame: -3.25..3.25 (x), -3.75..3.75 (z)
        # SVG: 0..655 (x), 0..755 (y with y=0 at top)
        # World: fu = svg_x/655, fv = 1 - svg_y/755
        # x_world = (fu - 0.5) * 6.5
        # z_world = fv * 7.5 - 7.5*0.43

        W, H = 6.5, 7.5

        eye_positions = [
            (150.0, 244.0),   # left eye center in SVG px
            (500.0, 242.0),   # right eye center
        ]

        vw = GeomVertexWriter(self._eye_vdata, "vertex")
        cw = GeomVertexWriter(self._eye_vdata, "color")
        vw.setRow(0)
        cw.setRow(0)

        # Pupil size in world units (small golden dot)
        pw = 0.08 * pupil_scale * W / 6.5
        ph = 0.06 * pupil_scale * H / 7.5

        for svg_x, svg_y in eye_positions:
            fu = svg_x / 655.0
            fv = 1.0 - svg_y / 755.0
            cx = (fu - 0.5) * W
            cz = fv * H - H * 0.5
            y = 0.05  # float in front of card

            for sx, sz in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
                vw.addData3(cx + sx * pw, y, cz + sz * ph)
                cw.addData4f(1.0, 0.94, 0.10, 0.92)

    # ── SHOW / HIDE ───────────────────────────────────────────────────
    def show(self):
        print("[moses] show() called")
        self._root.show()
        self._pupil_np.show()
        self._glow.show()
        print("[moses] show() done")

    def hide(self):
        self._root.hide()
        self._pupil_np.hide()
        self._glow.hide()

    def isHidden(self):
        return self._root.isHidden()

    def node(self):
        return self._root

    # ── UPDATE ─────────────────────────────────────────────────────────
    def update(self, dt, features=None):
        """Animate from audio features."""
        if features is None:
            return

        rms = float(getattr(features, 'rms', 0.0))
        bass = float(getattr(features, 'bass', 0.0))
        treble = float(getattr(features, 'treble', 0.0))
        onset = bool(getattr(features, 'onset', False))
        beat_phase = float(getattr(features, 'beat_phase', 0.0))

        # --- Hue shift (beat-reactive gradient cycling) ---
        if self._shader_ok:
            base_speed = 0.003 + bass * 0.02 + treble * 0.01
            self._hue_shift += dt * base_speed
            if onset:
                self._hue_shift += 0.04 + bass * 0.06
            self._hue_shift = self._hue_shift % 1.0
            self._root.setShaderInput("u_hue_shift", self._hue_shift)

        # --- Gentle hover oscillation ---
        hover_z = 0.0 + 0.12 * math.sin(time.time() * 0.7)
        hover_x = 0.06 * math.sin(time.time() * 0.5 + 1.3)
        self._root.setZ(hover_z)
        self._root.setX(hover_x)
        if hasattr(self, '_glow'):
            self._glow.setZ(hover_z)
            self._glow.setX(hover_x)

        # --- Glow pulse (beat-reactive) ---
        if hasattr(self, '_glow'):
            ga = 0.05 + rms * 0.10 + bass * 0.04
            if onset:
                ga = min(0.22, ga + 0.08)
            gs = 1.0 + rms * 0.12 + bass * 0.06
            self._glow.setColor(1.0, 0.45, 0.08, ga)
            self._glow.setScale(0.67 * gs)

        # Pupil dilation from bass
        pupil_scale = 0.4 + min(bass, 0.5) * 0.7
        if onset:
            pupil_scale = min(1.2, pupil_scale + 0.3)
        self._update_pupils(pupil_scale)

    def destroy(self):
        if self._root is not None:
            self._root.removeNode()
            self._root = None
