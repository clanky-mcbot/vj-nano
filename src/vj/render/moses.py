"""High-quality Moses shield-face for vj-nano.

360x360 dome-shaded shield with brows, nostrils, animated eyes and mouth,
hue-rotating gradient, and beat-reactive glow. Designed for Jetson Nano.
"""

from __future__ import print_function

import math
import numpy as np

from panda3d.core import (
    CardMaker, Geom, GeomNode, GeomTriangles, GeomVertexData,
    GeomVertexFormat, GeomVertexWriter, GeomLines, GeomLinestrips,
    NodePath, TextureStage, TransparencyAttrib, Shader, ShaderAttrib,
)


# ═══════════════════════════════════════════════════════════════════════
# HUE-ROTATION SHADER (GLSL 130 for Tegra X1)
# ═══════════════════════════════════════════════════════════════════════

_HUE_SHADER_VERT = """#version 130
uniform mat4 p3d_ModelViewProjectionMatrix;
in vec4 p3d_Vertex;
in vec4 p3d_Color;
out vec4 v_color;
void main() {
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
    v_color = p3d_Color;
}
"""

_HUE_SHADER_FRAG = """#version 130
uniform float u_hue_shift;
in vec4 v_color;
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
    vec3 hsv = rgb2hsv(v_color.rgb);
    hsv.x = fract(hsv.x + u_hue_shift);
    p3d_FragColor = vec4(hsv2rgb(hsv), v_color.a);
}
"""

# ═══════════════════════════════════════════════════════════════════════
# SHIELD PROFILE
# ═══════════════════════════════════════════════════════════════════════

def _half_width(fv):
    """fv=0 (tip) .. fv=1 (crown). Returns half-width in fu-space [0, 0.5]."""
    if fv < 0.09:
        t = fv / 0.09
        return t * t * (3.0 - 2.0 * t) * 0.34
    elif fv < 0.21:
        t = (fv - 0.09) / 0.12
        return 0.34 + t * 0.135
    elif fv < 0.86:
        t = (fv - 0.21) / 0.65
        return 0.475 + math.sin(t * math.pi) * 0.007
    else:
        t = (fv - 0.86) / 0.14
        return 0.475 + 0.007 - t * t * 0.022


def _shield_alpha(fu, fv, hw):
    """Smooth shield mask: 1 inside, 0 outside, soft edges."""
    dcx = abs(fu - 0.5)
    if hw < 1e-4:
        return 0.0
    nx = dcx / hw
    if nx >= 1.0:
        return 0.0
    h_alpha = min(1.0, (1.0 - nx) / 0.07)
    t_alpha = min(1.0, fv / 0.055)
    return h_alpha * t_alpha


def _in_ellipse(fu, fv, cx, cy, rw, rh, angle):
    """Rotated ellipse hit-test."""
    dx =  (fu - cx) * math.cos(angle) + (fv - cy) * math.sin(angle)
    dy = -(fu - cx) * math.sin(angle) + (fv - cy) * math.cos(angle)
    return (dx / rw) ** 2 + (dy / rh) ** 2 < 1.0


# ═══════════════════════════════════════════════════════════════════════
# MOSES FACE
# ═══════════════════════════════════════════════════════════════════════

class MosesFace(object):
    """Moses shield-face: static base + animated eyes/mouth + beat glow."""

    def __init__(self, render_parent):
        # type: (NodePath) -> None
        self._parent = render_parent
        self._W = 6.2
        self._H = 7.6
        self._ROWS = 360
        self._COLS = 360
        self._hue_shift = 0.0
        self._shader = None

        # Root node
        self._root = render_parent.attachNewNode("moses")
        self._root.setPos(0, 10, 1.5)
        self._root.setTwoSided(True)
        self._root.setTransparency(TransparencyAttrib.MAlpha)

        # --- Compile shared shader ---
        self._shader = Shader.make(
            Shader.SL_GLSL, _HUE_SHADER_VERT, _HUE_SHADER_FRAG
        )

        # --- Layer 1: Glow (behind, pulsing) ---
        self._build_glow()

        # --- Layer 2: Static shield base (gradient + dome + brows + nostrils) ---
        self._build_base()

        # --- Layer 3: Dynamic eyes ---
        self._build_eyes()

        # --- Layer 4: Dynamic mouth ---
        self._build_mouth()

        # Start hidden
        self._root.hide()


    # ── GLOW ──────────────────────────────────────────────────────────
    def _build_glow(self):
        cm = CardMaker("moses_glow")
        cm.setFrame(-6.5, 6.5, -5.5, 5.5)
        self._glow = self._root.attachNewNode(cm.generate())
        self._glow.setY(-0.15)
        self._glow.setColor(1.0, 0.55, 0.05, 0.12)
        self._glow.setTransparency(TransparencyAttrib.MAlpha)
        self._glow.setDepthWrite(False)
        self._glow.setBin("transparent", -1)
        self._glow.setTwoSided(True)

    # ── STATIC BASE ───────────────────────────────────────────────────
    def _build_base(self):
        """Build full 360x360 shield face once — NO eyes/mouth."""
        fmt = GeomVertexFormat.getV3c4()
        vdata = GeomVertexData('moses_base', fmt, Geom.UHStatic)
        vtx = GeomVertexWriter(vdata, 'vertex')
        col = GeomVertexWriter(vdata, 'color')

        ROWS, COLS = self._ROWS, self._COLS
        W, H = self._W, self._H

        for ri in range(ROWS):
            fv = ri / (ROWS - 1)
            z = fv * H - H * 0.43
            hw = _half_width(fv)

            for si in range(COLS):
                fu = si / (COLS - 1)
                alpha = _shield_alpha(fu, fv, hw)
                if alpha <= 0.0:
                    vtx.addData3((fu - 0.5) * W * hw * 2, 0, z)
                    col.addData4f(0, 0, 0, 0)
                    continue

                dcx = abs(fu - 0.5)
                nx = (dcx / hw) if hw > 1e-4 else 1.0
                x = (fu - 0.5) * W * hw * 2

                # Dome depth (hot-spot slightly right)
                cx_shift = 0.05
                nx_s = abs(fu - (0.5 + cx_shift)) / hw if hw > 1e-4 else 1.0
                nx_s = min(1.0, nx_s)
                dome = max(0.0, 1.0 - nx_s ** 2)
                y = dome * 1.30 * (0.22 + hw * 2 * 0.78)

                # Base gradient: orange left → yellow centre → yellow-green right
                rc = 1.0
                gc = 0.32 + fu * 0.56 + fv * 0.15
                if fu > 0.55 and fv < 0.40:
                    gc = min(1.0, gc + (fu - 0.55) * 0.50 + (0.40 - fv) * 0.40)
                gc = min(1.0, gc)
                bc = 0.0

                # 3D shading: bright centre, dark edges
                shade = max(0.03, 1.0 - nx ** 1.30 * 0.82)
                if fv < 0.22:
                    shade *= max(0.04, 0.25 + fv * 3.5)
                rc *= shade
                gc *= shade

                # Eyebrows (diagonal slashes)
                # Left brow
                lbx0, lby0 = 0.155, 0.760
                lbx1, lby1 = 0.448, 0.700
                if lbx0 < fu < lbx1:
                    t = (fu - lbx0) / (lbx1 - lbx0)
                    by = lby0 + t * (lby1 - lby0)
                    thk = 0.026 - t * 0.007
                    if abs(fv - by) < thk:
                        rc, gc, bc = 0.68 * shade, 0.26 * shade, 0.0

                # Right brow (mirror)
                rbx0, rby0 = 0.552, 0.700
                rbx1, rby1 = 0.845, 0.760
                if rbx0 < fu < rbx1:
                    t = (fu - rbx0) / (rbx1 - rbx0)
                    by = rby0 + t * (rby1 - rby0)
                    thk = 0.019 + t * 0.009
                    if abs(fv - by) < thk:
                        rc, gc, bc = 0.68 * shade, 0.26 * shade, 0.0

                # Nostrils (static, subtle)
                if 0.508 < fv < 0.527:
                    if (0.453 < fu < 0.472) or (0.528 < fu < 0.547):
                        rc = 0.50 * shade
                        gc = 0.19 * shade
                        bc = 0.0

                # NO eyes, NO mouth in base layer

                vtx.addData3(x, y, z)
                col.addData4f(rc, gc, bc, alpha)

        # Triangles
        tris = GeomTriangles(Geom.UHStatic)
        for ri in range(ROWS - 1):
            for si in range(COLS - 1):
                v0 = ri * COLS + si
                tris.addVertices(v0, v0 + 1, v0 + COLS)
                tris.addVertices(v0 + 1, v0 + COLS + 1, v0 + COLS)

        geom = Geom(vdata)
        geom.addPrimitive(tris)
        node = GeomNode("moses_base")
        node.addGeom(geom)

        self._base_np = self._root.attachNewNode(node)
        if self._shader is not None:
            self._base_np.setShader(self._shader)
            self._base_np.setShaderInput("u_hue_shift", 0.0)
        self._base_np.setTransparency(TransparencyAttrib.MAlpha)
        # Apply to this node and children
        self._shader_attrib = ShaderAttrib.make(self._shader)
        self._base_np.setAttrib(self._shader_attrib)

    # ── EYES (dynamic) ────────────────────────────────────────────────
    def _build_eyes(self):
        """Create eye slit quads and pupil quads — rebuilt each frame."""
        # Two dark slit quads + two golden pupil quads
        n_verts = 4 * 4  # 4 quads x 4 verts = 16 total
        fmt = GeomVertexFormat.getV3c4()
        self._eye_vdata = GeomVertexData("moses_eyes", fmt, Geom.UHDynamic)
        vw = GeomVertexWriter(self._eye_vdata, "vertex")
        cw = GeomVertexWriter(self._eye_vdata, "color")
        for _ in range(n_verts):
            vw.addData3(0, 0.01, 0)
            cw.addData4f(0, 0, 0, 0)

        # 4 quads, each 2 triangles = 8 tris total
        tris = GeomTriangles(Geom.UHDynamic)
        for q in range(4):
            b = q * 4
            tris.addVertices(b, b + 1, b + 2)
            tris.addVertices(b, b + 2, b + 3)

        geom = Geom(self._eye_vdata)
        geom.addPrimitive(tris)
        node = GeomNode("moses_eyes")
        node.addGeom(geom)
        self._eyes_np = self._root.attachNewNode(node)
        self._eyes_np.setY(0.02)
        self._eyes_np.setTransparency(TransparencyAttrib.MAlpha)
        self._eyes_np.setDepthWrite(False)
        self._eyes_np.setBin("transparent", 1)

    def _update_eyes(self, eye_angle, pupil_scale):
        """Rebuild eye geometry based on tilt angle and pupil size."""
        EY = 0.644
        ELH = 0.316
        ERH = 0.684
        EW = 0.075
        EH = 0.021

        la = -eye_angle * 0.44
        ra = eye_angle * 0.44

        W, H = self._W, self._H

        vw = GeomVertexWriter(self._eye_vdata, "vertex")
        cw = GeomVertexWriter(self._eye_vdata, "color")
        vw.setRow(0)
        cw.setRow(0)

        for (cx_fu, angle, side) in [(ELH, la, -1), (ERH, ra, 1)]:
            fv_ey = EY
            hw = _half_width(fv_ey)
            cx = (cx_fu - 0.5) * W * hw * 2
            z_base = fv_ey * H - H * 0.43

            # Dome depth at eye position
            nx_s = abs(cx_fu - 0.55) / hw if hw > 1e-4 else 1.0
            nx_s = min(1.0, nx_s)
            dome = max(0.0, 1.0 - nx_s ** 2)
            y_dome = dome * 1.30 * (0.22 + hw * 2 * 0.78)

            # Eye dimensions in world units
            ew = EW * W * hw * 2
            eh = EH * H * 0.5
            y = y_dome + 0.015

            # Build rotated quad corners
            ca = math.cos(angle)
            sa = math.sin(angle)
            corners = [
                (-ew, -eh), (ew, -eh), (ew, eh), (-ew, eh)
            ]
            # Dark slit
            for dx, dz in corners:
                rx = cx + dx * ca - dz * sa
                rz = z_base + dx * sa + dz * ca
                vw.addData3(rx, y, rz)
                cw.addData4f(0.07, 0.02, 0.0, 0.95)

            # Golden pupil (centered, smaller)
            ps = pupil_scale * 0.28
            for dx, dz in corners:
                rx = cx + dx * ps * ca - dz * ps * sa
                rz = z_base + dx * ps * sa + dz * ps * ca
                vw.addData3(rx, y + 0.002, rz)
                cw.addData4f(1.0, 0.94, 0.10, 0.9)

    # ── MOUTH (dynamic) ───────────────────────────────────────────────
    def _build_mouth(self):
        """Upper and lower lip line segments."""
        fmt = GeomVertexFormat.getV3c4()
        self._mouth_vdata = GeomVertexData("moses_mouth", fmt, Geom.UHDynamic)
        vw = GeomVertexWriter(self._mouth_vdata, "vertex")
        cw = GeomVertexWriter(self._mouth_vdata, "color")
        # 2 line segments (upper + lower), 2 verts each = 4 verts
        for _ in range(4):
            vw.addData3(0, 0.02, 0)
            cw.addData4f(0.5, 0.2, 0.0, 0.85)

        lines = GeomLines(Geom.UHDynamic)
        lines.addVertices(0, 1)
        lines.addVertices(2, 3)

        geom = Geom(self._mouth_vdata)
        geom.addPrimitive(lines)
        node = GeomNode("moses_mouth")
        node.addGeom(geom)
        self._mouth_np = self._root.attachNewNode(node)
        self._mouth_np.setY(0.025)
        self._mouth_np.setRenderModeThickness(2.5)
        self._mouth_np.setTransparency(TransparencyAttrib.MAlpha)
        self._mouth_np.setDepthWrite(False)
        self._mouth_np.setBin("transparent", 2)

    def _update_mouth(self, mouth_open):
        """Reposition lip lines based on mouth openness."""
        MCV = 0.398
        MHW = 0.190
        lip_gap = 0.010 + mouth_open * 0.058

        W, H = self._W, self._H
        fv_mid = MCV
        hw = _half_width(fv_mid)
        z_mid = fv_mid * H - H * 0.43

        # Dome depth at mouth
        cx_shift = 0.05
        nx_s = abs(0.5 - (0.5 + cx_shift)) / hw if hw > 1e-4 else 1.0
        nx_s = min(1.0, nx_s)
        dome = max(0.0, 1.0 - nx_s ** 2)
        y_dome = dome * 1.30 * (0.22 + hw * 2 * 0.78)

        half_w = MHW * W * hw * 2
        sag = MHW ** 2 * 0.030 * H

        vw = GeomVertexWriter(self._mouth_vdata, "vertex")
        cw = GeomVertexWriter(self._mouth_vdata, "color")
        vw.setRow(0)
        cw.setRow(0)

        y = y_dome + 0.02
        # Upper lip
        zu = z_mid - sag
        vw.addData3(-half_w, y, zu)
        cw.addData4f(0.66, 0.26, 0.0, 0.85)
        vw.addData3(half_w, y, zu)
        cw.addData4f(0.66, 0.26, 0.0, 0.85)
        # Lower lip
        zl = zu - lip_gap
        vw.addData3(-half_w, y, zl)
        cw.addData4f(0.58, 0.22, 0.0, 0.85)
        vw.addData3(half_w, y, zl)
        cw.addData4f(0.58, 0.22, 0.0, 0.85)

    # ── SHOW/HIDE ─────────────────────────────────────────────────────
    def show(self):
        self._root.show()

    def hide(self):
        self._root.hide()

    @property
    def isHidden(self):
        return self._root.isHidden()

    @property
    def node(self):
        return self._root

    # ── UPDATE ─────────────────────────────────────────────────────────
    def update(self, dt, features=None):
        """
        Animate the face from audio features.

        features should have: rms, bass, mid, treble, onset, bpm, beat_phase
        """
        if features is None:
            return

        rms = float(getattr(features, 'rms', 0.0))
        bass = float(getattr(features, 'bass', 0.0))
        treble = float(getattr(features, 'treble', 0.0))
        onset = bool(getattr(features, 'onset', False))
        beat_phase = float(getattr(features, 'beat_phase', 0.0))
        bpm = float(getattr(features, 'bpm', 120.0))

        # --- Gradient hue rotation ---
        speed = 0.02 + bass * 0.08 + treble * 0.04
        self._hue_shift += dt * speed
        self._hue_shift %= 1.0
        self._base_np.setShaderInput("u_hue_shift", self._hue_shift)

        # --- Eye tilt: driven by treble (wider/more aggressive on highs) ---
        eye_angle = 0.25 + treble * 0.55
        # Pupil dilation: driven by bass (bigger on kicks)
        pupil_scale = 0.6 + bass * 2.0
        if onset:
            pupil_scale = min(2.5, pupil_scale + 1.0)
        self._update_eyes(eye_angle, pupil_scale)

        # --- Mouth open: driven by rms/energy ---
        mouth_open = 0.05 + rms * 1.8
        if onset:
            mouth_open = min(1.0, mouth_open + 0.5)
        self._update_mouth(mouth_open)

        # --- Glow pulse: beat-reactive ---
        # Brighter on beat, decays between beats
        beat_pulse = 1.0 - beat_phase  # 1 at beat start, 0 right before next
        glow_alpha = 0.06 + rms * 0.30 + beat_pulse * 0.12 * bass
        if onset:
            glow_alpha = min(0.5, glow_alpha + 0.2)
        self._glow.setColor(1.0, 0.55, 0.05, glow_alpha)
        # Pulse glow size slightly
        glow_scale = 1.0 + rms * 0.25 + (0.15 if onset else 0.0)
        self._glow.setScale(glow_scale)

    def destroy(self):
        if self._root is not None:
            self._root.removeNode()
            self._root = None
