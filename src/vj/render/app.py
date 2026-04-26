"""Panda3D renderer module.

Opens a window, loads a PS1-style low-poly humanoid, and drives its
animations with audio-band energies and webcam palette colours.

Renderer runs on the same process as the audio analyzer + webcam capture
so they can share numpy arrays with zero serialization. The Nano's Maxwell
GPU (Tegra X1) has plenty of fixed-function rendering headroom for our
low-poly targets.

Target environment:
  - NVIDIA Tegra X1 GPU (sm_53 Maxwell)
  - OpenGL 4.6 via GLX, GLSL 5.0+
  - Panda3D 1.10.13
"""

# Python 3.6 compatible.

import math
import os
from typing import Optional

import numpy as np

from vj.render.actor import PS1Humanoid
from vj.render.animator import BeatAnimator

# Optional glTF path
try:
    from vj.render.gltf_actor import GltfActor
    from vj.render.gltf_animator import GltfAnimator
    _GLTF_AVAILABLE = True
except Exception:
    _GLTF_AVAILABLE = False

# Optional retro effects
try:
    from vj.render.effects import RetroVisualizer
    _FX_AVAILABLE = True
except Exception as exc:
    print("[render] effects not available:", exc)
    _FX_AVAILABLE = False

# GUI menu
try:
    from vj.render.gui import EffectMenu
    _GUI_AVAILABLE = True
except Exception as exc:
    print("[render] gui not available:", exc)
    _GUI_AVAILABLE = False

# Post-process filters
try:
    from vj.render.filters import PostProcessFilters
    _FILTERS_AVAILABLE = True
except Exception as exc:
    print("[render] filters not available:", exc)
    _FILTERS_AVAILABLE = False

# MilkDrop GPU visualizer
try:
    from vj.render.milkdrop import MilkDropRenderer
    _MILKDROP_AVAILABLE = True
except Exception as exc:
    print("[render] milkdrop not available:", exc)
    _MILKDROP_AVAILABLE = False

# We expose a config path relative to the repo root so callers can find it.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
PRC_CONFIG = os.path.join(REPO_ROOT, "config", "panda3d.prc")


def _load_prc():
    # type: () -> None
    """Load our PRC config before importing ShowBase."""
    from panda3d.core import loadPrcFile
    if os.path.isfile(PRC_CONFIG):
        loadPrcFile(PRC_CONFIG)


class VJApp(object):
    """The renderer.

    Decoupled from ShowBase so we can inject state from the main loop
    without subclassing. We still *use* ShowBase internally because it
    gives us a task manager, window, camera, and scene graph for free.
    """

    def __init__(self, window_title="vj-nano", win_size=(960, 540), debug=False, model="procedural", flip_webcam=False):
        # type: (str, tuple, bool, str, bool) -> None
        _load_prc()
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "window-title {}".format(window_title))
        loadPrcFileData(
            "", "win-size {} {}".format(win_size[0], win_size[1]))

        from direct.showbase.ShowBase import ShowBase
        self.base = ShowBase()
        # Dark slate background — matches the Nous palette.
        self.base.setBackgroundColor(0.06, 0.08, 0.10, 1.0)

        # Create character (procedural or glTF)
        self._character = self.base.render.attachNewNode("character-root")
        self._model_path = model
        if model != "procedural" and _GLTF_AVAILABLE and os.path.isfile(model):
            model_abs = os.path.abspath(model)
            self._actor = GltfActor(self._character, model_abs)
            self._actor.root.setPos(0, 0, 0)
            self._actor.setScale(0.8)
            self._animator = GltfAnimator(self._actor)
            print("[render] Loaded glTF model: {}".format(model_abs))
            print("[render] Available anims:", self._actor.list_anims())
        else:
            self._actor = PS1Humanoid(self._character)
            self._actor.root.setPos(0, 0, 0)
            self._animator = BeatAnimator(self._actor)
            if model != "procedural":
                print("[render] warning: model '{}' not found, using procedural".format(model))
        # Position character in front of camera
        self._character.setPos(0, 8, 0)

        # Load PS1-style shader onto the whole character hierarchy
        self._setup_ps1_shader()

        # Camera pulled back for a wider view; actor stays center-framed.
        self.base.camera.setPos(0, -7, 2.0)
        if hasattr(self._actor, 'actor'):  # GLB actor
            self.base.camera.lookAt(0, 8, 0.0)
        else:
            self.base.camera.lookAt(0, 8, 1.24)

        # Disable default mouse camera controls so clicks don't move the view.
        self.base.disableMouse()

        # --- retro 2000s visualizer effects ---
        self._fx = None  # type: Optional[RetroVisualizer]
        if _FX_AVAILABLE:
            self._fx = RetroVisualizer(self.base.render, self.base.aspect2d)
            print("[render] Retro visualizer effects loaded")

        # --- post-process filters (disabled by default — heavy on Maxwell) ---
        self._filters = None
        if _FILTERS_AVAILABLE and os.environ.get("VJ_FILTERS", "0") == "1":
            try:
                self._filters = PostProcessFilters(self.base)
                print("[render] Post-process filters ready")
            except Exception as exc:
                print("[render] filter init failed:", exc)
        else:
            print("[render] Post-process filters skipped (set VJ_FILTERS=1 to enable)")

        # --- MilkDrop GPU visualizer (always on — ultra cheap by design) ---
        self._milkdrop = None  # type: Optional[MilkDropRenderer]
        if _MILKDROP_AVAILABLE:
            try:
                self._milkdrop = MilkDropRenderer(self.base)
                print("[render] MilkDrop GPU visualizer ready")
            except Exception as exc:
                print("[render] milkdrop init failed:", exc)
                self._milkdrop = None

        # --- debug overlay state ---
        self._debug_nodes = []  # populated below if debug=True
        self._debug_visible = True

        # --- effect control menu ---
        self._menu = None  # type: Optional[EffectMenu]

        # State fed from outside each frame.
        self._rotation = 0.0
        self._energy = 0.0
        self._tint = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # RGB 0..1
        self._features = None  # type: Optional[object]
        self._waveform = np.zeros(512, dtype=np.float32)
        self._ps1_shader_loaded = False

        self._debug = debug
        self._flip_webcam = flip_webcam
        self._webcam_frame = None  # type: Optional[np.ndarray]
        self._robot_visible = True
        self._robot_orig_scale = None  # set lazily
        if self._debug:
            self._build_debug_overlay()

        self._make_orb()
        self._make_moses()

        # Create menu AFTER debug nodes are populated so it gets the real list
        if _GUI_AVAILABLE and self._fx is not None:
            self._menu = EffectMenu(self.base, self._fx, self._filters, self._debug_nodes, milkdrop=self._milkdrop)
            print("[render] Effect menu loaded")
            # Bind 'H' to toggle all debug overlays
            self.base.accept("h", self._toggle_debug)
            self.base.accept("H", self._toggle_debug)
            # Bind 'R' to toggle robot visibility
            self.base.accept("r", self.toggle_robot)
            self.base.accept("R", self.toggle_robot)

        # Drive rotation/tint every frame via a task.
        self.base.taskMgr.add(self._update_task, "vj-update")

    # ------------------------------------------------------------------
    def _setup_ps1_shader(self):
        # type: () -> None
        """Load PS1-style GLSL shader onto the character."""
        from panda3d.core import Shader
        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        vert = os.path.join(shader_dir, "ps1_vertex_v2.glsl")
        frag = os.path.join(shader_dir, "ps1_fragment_v2.glsl")
        if os.path.isfile(vert) and os.path.isfile(frag):
            shader = Shader.load(Shader.SL_GLSL, vert, frag)
            if shader:
                self._character.setShader(shader)
                self._character.setShaderInput("ps1_time", 0.0)
                self._character.setShaderInput("ps1_snap_resolution", 96.0)
                self._character.setShaderInput("ps1_wobble_intensity", 1.5)
                self._character.setShaderInput("ps1_banding_steps", 8.0)
                self._character.setShaderInput("ps1_dither_amount", 0.3)  # reduced: was 1.5
                self._character.setShaderInput("ps1_fog_start", 6.0)
                self._character.setShaderInput("ps1_fog_end", 20.0)
                self._character.setShaderInput("ps1_fog_color", (0.06, 0.08, 0.10))
                self._ps1_shader_loaded = True
                print("[render] PS1 shader loaded")
            else:
                print("[render] warning: shader compile failed, using fixed-function")
        else:
            print("[render] warning: PS1 shader files not found, using fixed-function")

    # ------------------------------------------------------------------
    # Debug overlay
    # ------------------------------------------------------------------
    def _build_debug_overlay(self):
        # type: () -> None
        """Create 2D debug HUD: waveform + text stats."""
        from panda3d.core import (
            GeomNode, Geom, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
            GeomLinestrips, NodePath,
        )
        from direct.gui.OnscreenText import OnscreenText

        self._debug_original_scales = {}  # node -> original scale

        # --- waveform oscilloscope (line strip in aspect2d) ---
        self._scope_n = 512
        fmt = GeomVertexFormat.getV3()
        self._scope_vdata = GeomVertexData("scope", fmt, Geom.UHDynamic)
        vwrite = GeomVertexWriter(self._scope_vdata, "vertex")
        for _ in range(self._scope_n):
            vwrite.addData3(0.0, 0.0, 0.0)

        prim = GeomLinestrips(Geom.UHDynamic)
        prim.addConsecutiveVertices(0, self._scope_n)
        geom = Geom(self._scope_vdata)
        geom.addPrimitive(prim)
        node = GeomNode("scope")
        node.addGeom(geom)
        self._scope_np = NodePath(node)
        self._scope_np.reparentTo(self.base.aspect2d)
        self._scope_np.setColor(0.0, 1.0, 0.5, 0.8)
        self._scope_np.setAntialias(0)  # crisp pixels
        self._debug_nodes.append(self._scope_np)
        self._debug_original_scales[self._scope_np] = self._scope_np.getScale()

        # --- text labels ---
        self._txt_bpm = OnscreenText(
            text="BPM: --  FPS: --",
            pos=(-0.95, 0.90),
            scale=0.05,
            fg=(0.0, 1.0, 0.5, 1.0),
            align=0,  # left
            parent=self.base.aspect2d,
        )
        self._debug_nodes.append(self._txt_bpm)
        self._debug_original_scales[self._txt_bpm] = self._txt_bpm.getScale()

        self._txt_rms = OnscreenText(
            text="RMS: --",
            pos=(-0.95, 0.84),
            scale=0.04,
            fg=(0.7, 0.7, 0.7, 1.0),
            align=0,
            parent=self.base.aspect2d,
        )
        self._debug_nodes.append(self._txt_rms)
        self._debug_original_scales[self._txt_rms] = self._txt_rms.getScale()

        self._txt_onset = OnscreenText(
            text="ONSET",
            pos=(-0.95, 0.78),
            scale=0.04,
            fg=(0.2, 0.2, 0.2, 1.0),
            align=0,
            parent=self.base.aspect2d,
        )
        self._debug_nodes.append(self._txt_onset)
        self._debug_original_scales[self._txt_onset] = self._txt_onset.getScale()

        # --- webcam preview (top-left) ---
        from panda3d.core import Texture, CardMaker, TextureStage
        self._webcam_tex = Texture("webcam")
        self._webcam_tex.setup2dTexture(160, 120, Texture.TUnsignedByte, Texture.FRgb)
        cm = CardMaker("webcam")
        cm.setFrame(-0.98, -0.62, 0.55, 0.85)
        self._webcam_card = self.base.aspect2d.attachNewNode(cm.generate())
        self._webcam_card.setTexture(self._webcam_tex)
        # OpenGL texture V coord goes bottom-to-top, OpenCV images are top-to-bottom.
        # If the webcam frame is NOT flipped, we need to flip the texture to show it right-side up.
        # If --flip-webcam is used, the frame data is already right-side up, so no texture flip needed.
        if not getattr(self, '_flip_webcam', False):
            self._webcam_card.setTexScale(TextureStage.getDefault(), 1, -1)
        self._webcam_card.setTransparency(1)
        self._webcam_card.setBin("transparent", 10)
        self._debug_nodes.append(self._webcam_card)
        self._debug_original_scales[self._webcam_card] = self._webcam_card.getScale()

    def _update_debug(self, feat, wf):
        # type: (object, np.ndarray) -> None
        if not self._debug:
            return

        # Update text
        bpm = feat.bpm if feat.bpm > 0 else 0.0
        fps = self.base.taskMgr.globalClock.getAverageFrameRate()
        self._txt_bpm.setText("BPM: {:.1f}  FPS: {:.0f}".format(bpm, fps))
        self._txt_rms.setText("RMS: {:.3f}  B:{:.2f} M:{:.2f} T:{:.2f}".format(
            feat.rms, feat.bass, feat.mid, feat.treble))
        if feat.onset:
            self._txt_onset.setFg((1.0, 0.2, 0.2, 1.0))
            self._txt_onset.setText("ONSET")
        else:
            self._txt_onset.setFg((0.2, 0.2, 0.2, 1.0))
            self._txt_onset.setText("     ")

        # Update waveform geometry
        from panda3d.core import GeomVertexWriter
        vwrite = GeomVertexWriter(self._scope_vdata, "vertex")
        n = min(self._scope_n, wf.shape[0])
        y_base = -0.70
        y_height = 0.12
        for i in range(n):
            x = -0.90 + 1.80 * (i / float(n - 1)) if n > 1 else 0.0
            amp = float(wf[i])
            y = y_base + amp * y_height
            vwrite.setData3(x, 0.0, y)

    # ------------------------------------------------------------------
    def _toggle_debug(self):
        # type: () -> None
        """Toggle all debug overlay nodes on/off (bound to 'H' key)."""
        self._debug_visible = not self._debug_visible
        for node in self._debug_nodes:
            if self._debug_visible:
                orig = self._debug_original_scales.get(node, (1, 1, 1))
                node.setScale(orig)
            else:
                node.setScale(0.001)

    def set_audio_energy(self, energy):
        # type: (float) -> None
        """0..1-ish overall energy from the audio analyzer."""
        self._energy = float(energy)

    def set_tint(self, rgb):
        # type: (np.ndarray) -> None
        """RGB 0..1 triple to tint the character."""
        self._tint = np.asarray(rgb, dtype=np.float32).reshape(3).clip(0.0, 1.0)

    def set_beat_phase(self, phase):
        # type: (float) -> None
        """0..1 position in the current beat; drives rotation target."""
        self._beat_phase = float(phase)

    def set_features(self, feat):
        # type: (object) -> None
        """Feed a full AudioFeatures snapshot for animator-driven motion."""
        self._features = feat

    def set_waveform(self, wf):
        # type: (np.ndarray) -> None
        """Feed raw audio waveform for debug oscilloscope."""
        self._waveform = np.asarray(wf, dtype=np.float32)

    def set_webcam_frame(self, frame_rgb):
        # type: (np.ndarray) -> None
        """Feed a small RGB frame (HxWx3 uint8) for debug preview."""
        self._webcam_frame = frame_rgb

    # ------------------------------------------------------------------
    def _update_task(self, task):
        dt = self.base.taskMgr.globalClock.getDt()
        if self._features is not None:
            # Animator drives the actor directly via BPM-locked clips
            _hpr, scale, _pos = self._animator.update(self._features, dt)
            # Only set scale if robot is visible (R key toggle)
            if getattr(self, '_robot_visible', True):
                self._character.setScale(scale)
            if self._debug:
                self._update_debug(self._features, self._waveform)
            # Update retro effects
            if self._fx is not None:
                self._fx.update(self._features, self._waveform, dt, tint=self._tint)
        else:
            # Legacy manual path (smoke test, etc.)
            self._rotation += dt * (30.0 + 240.0 * self._energy)
            phase = getattr(self, "_beat_phase", 0.0)
            pulse = 1.0 + 0.06 * math.cos(phase * 2.0 * math.pi)
            self._character.setHpr(self._rotation, 15.0, 0.0)
            self._character.setScale(pulse)
            self._character.setPos(0, 8, 0)

        # Update PS1 shader time uniform
        if getattr(self, "_ps1_shader_loaded", False):
            t = self.base.taskMgr.globalClock.getFrameTime()
            self._character.setShaderInput("ps1_time", t)

        # Update post-process filters (e.g. ASCII art)
        if self._filters is not None:
            self._filters.update(dt)

        # Update MilkDrop GPU visualizer with audio state
        if self._milkdrop is not None:
            if self._features is not None:
                self._milkdrop.update(
                    bass=float(self._features.bass),
                    mid=float(self._features.mid),
                    treble=float(self._features.treble),
                    volume=float(self._features.rms),
                    onset=bool(self._features.onset),
                    energy=float(self._energy),
                    dt=dt,
                )
            else:
                # Legacy/smoke-test path: synthetic audio from energy + time
                self._milkdrop.update(
                    bass=self._energy * 0.8,
                    mid=0.2,
                    treble=0.1,
                    volume=self._energy,
                    onset=False,
                    energy=self._energy,
                    dt=dt,
                )

        # Subtle background colour shift driven by webcam tint + energy
        br = 0.06 + self._tint[0] * 0.04 * self._energy
        bg = 0.08 + self._tint[1] * 0.04 * self._energy
        bb = 0.10 + self._tint[2] * 0.04 * self._energy
        self.base.setBackgroundColor(br, bg, bb, 1.0)
        # Also propagate to the FilterManager offscreen buffer so it clears
        # to the same colour (prevents ghosting on Jetson).
        if self._filters is not None:
            self._filters.set_clear_color(br, bg, bb, 1.0)

        # Update webcam preview texture & motion diff background
        if getattr(self, '_webcam_tex', None) is not None and self._webcam_frame is not None:
            # Feed motion diff before consuming frame (only if enabled)
            if self._fx is not None and hasattr(self._fx, "_motion"):
                if self._fx.enabled.get("motion", True):
                    self._fx._motion.update(
                        self._webcam_frame,
                        dt,
                        intensity=self._fx.intensity.get("motion", 1.0),
                    )
            self._webcam_tex.setRamImage(self._webcam_frame.tobytes())
            self._webcam_frame = None  # consume

        # --- Orb beat morph ---
        if hasattr(self, '_orb') and not self._orb.isHidden() and self._features is not None:
            if not hasattr(self, '_orb_morph_t'): self._orb_morph_t = 1.0
            self._orb_morph_t = min(1.0, self._orb_morph_t + dt * 3.0)
            # Detect beat boundaries from phase wrapping (0.0..1.0)
            # This never misses a beat because phase updates every hop
            phase = float(self._features.beat_phase)
            prev = getattr(self, '_orb_last_phase', 0.5)
            if prev > 0.5 and phase < 0.5:
                self._cycle_orb()
            self._orb_last_phase = phase
            # Smooth morph toward target shape
            if self._orb_target_verts is not None and self._orb_morph < 1.0:
                self._orb_morph = min(1.0, self._orb_morph + dt * 7.0)
                from panda3d.core import GeomVertexWriter
                vw = GeomVertexWriter(self._orb_vdata, "vertex")
                vw.setRow(0)
                for i in range(self._orb_n):
                    tx, ty, tz = self._orb_target_verts[i]
                    cx, cy, cz = self._orb_verts[i]
                    t = self._orb_morph
                    nx = cx + (tx - cx) * t
                    ny = cy + (ty - cy) * t
                    nz = cz + (tz - cz) * t
                    vw.setData3f(nx, ny, nz)
                    self._orb_verts[i] = (nx, ny, nz)
            self._orb.setScale(1.0 + 0.04 * float(self._features.rms))
            self._orb.setH(self._orb.getH() + dt * 30.0)

        # --- Moses face spin ---
        if hasattr(self, '_moses') and not self._moses.isHidden() and self._features is not None:
            self._moses.setH(self._moses.getH() + dt * 15.0)
            self._moses.setScale(1.5 + 0.08 * float(self._features.bass) * 3.0)

        # Vertex-color tint
        r, g, b = float(self._tint[0]), float(self._tint[1]), float(self._tint[2])
        self._character.setColorScale(r, g, b, 1.0)
        from direct.task import Task
        return Task.cont

    def run(self):
        """Block on the Panda3D main loop. Press ESC or close window to exit."""
        self.base.run()


    # --- Cycle dispatcher ---
    def toggle_robot(self):
        self._cycle_state = (getattr(self, "_cycle_state", 1) + 1) % 4
        self._character.setScale(0.001)
        if hasattr(self, "_orb"): self._orb.hide()
        if hasattr(self, "_moses"): self._moses.hide()
        self._robot_visible = False

        if self._cycle_state == 1:
            if self._robot_orig_scale is None:
                self._robot_orig_scale = self._character.getScale()
            self._character.setScale(self._robot_orig_scale)
            self._robot_visible = True
        elif self._cycle_state == 2 and hasattr(self, "_orb"):
            self._orb.show()
            self._orb_segments = 16
            self._orb_target_segments = 16
            self._orb_morph_t = 1.0
            self._orb_beat_count = -1  # reset bar so next beat is beat 0
            self._orb_last_phase = 0.5
        elif self._cycle_state == 3 and hasattr(self, "_moses"):
            self._moses.show()
        return self._cycle_state


    # --- Wireframe Orb (IBM green, beat-reactive) ---
    def _make_orb(self):
        from panda3d.core import (
            Geom, GeomLines, GeomNode, GeomVertexData,
            GeomVertexFormat, GeomVertexWriter, NodePath,
        )
        import math, random

        # Shape selection now handled by bar-aware _cycle_orb (random picks on beats 3-4)
        self._orb_beat_count = 0
        self._orb_morph = 1.0
        self._orb_last_phase = 0.5
        self._orb_target_verts = None
        self._orb_verts = None
        self._orb_n = 0

        rg, gg, bg = 0.20, 0.95, 0.25
        r = 3.0

        # Generate base sphere vertices
        verts = []
        n_rings, n_segs = 8, 16
        for ring in range(n_rings + 1):
            phi = math.pi * ring / n_rings
            n = n_segs if ring > 0 and ring < n_rings else 1
            for pt in range(n):
                th = 2 * math.pi * pt / n
                verts.append((r * math.sin(phi) * math.cos(th),
                              r * math.sin(phi) * math.sin(th),
                              r * math.cos(phi)))

        self._orb_n = len(verts)
        self._orb_base = verts[:]
        self._orb_verts = [(x,y,z) for x,y,z in verts]

        # Pre-compute target vertices for each shape
        self._orb_targets = {}
        self._orb_targets["sphere"] = [(x,y,z) for x,y,z in verts]

        # Pyramid targets
        pyr_v = [(0,0,3.5),(-3.5,3.5,-3.5),(3.5,3.5,-3.5),(3.5,-3.5,-3.5),(-3.5,-3.5,-3.5)]
        pyr_e = [(0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)]
        pyr_t = []
        for vx,vy,vz in verts:
            best = (vx,vy,vz)
            best_d = 1e9
            for e0,e1 in pyr_e:
                p0,p1 = pyr_v[e0],pyr_v[e1]
                dx,dy,dz = p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2]
                t = max(0,min(1,((vx-p0[0])*dx+(vy-p0[1])*dy+(vz-p0[2])*dz)/(dx*dx+dy*dy+dz*dz+1e-12)))
                px,py,pz = p0[0]+t*dx,p0[1]+t*dy,p0[2]+t*dz
                d = (vx-px)**2+(vy-py)**2+(vz-pz)**2
                if d < best_d:
                    best_d = d
                    best = (px,py,pz)
            pyr_t.append(best)
        self._orb_targets["pyramid"] = pyr_t

        # Dodecahedron targets (regular dodecahedron, 20 vertices)
        phi = (1+math.sqrt(5))/2
        iphi = 1/phi
        s = 2.5
        dod_v = []
        # Generate all 20 dodecahedron vertices
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    dod_v.append((sx*s, sy*s, sz*s))
        for a, b, c in [(0, phi, iphi), (iphi, 0, phi), (phi, iphi, 0)]:
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    dod_v.append((s1*a*s, s2*b*s, c*s))
                    dod_v.append((c*s, s1*a*s, s2*b*s))
                    dod_v.append((s2*b*s, c*s, s1*a*s))
        # Deduplicate and take exactly 20
        seen = {}
        unique = []
        for v in dod_v:
            key = (round(v[0], 3), round(v[1], 3), round(v[2], 3))
            if key not in seen:
                seen[key] = len(unique)
                unique.append(v)
        # Compute edge length (distance between adjacent vertices)
        edge_len = 0
        for i in range(1, len(unique)):
            d = math.sqrt((unique[0][0]-unique[i][0])**2 + (unique[0][1]-unique[i][1])**2 + (unique[0][2]-unique[i][2])**2)
            if d > 1 and (edge_len == 0 or d < edge_len):
                edge_len = d
        # Build adjacency
        edges = set()
        for i in range(len(unique)):
            for j in range(i+1, len(unique)):
                d = math.sqrt((unique[i][0]-unique[j][0])**2 + (unique[i][1]-unique[j][1])**2 + (unique[i][2]-unique[j][2])**2)
                if abs(d - edge_len) < 0.3:
                    edges.add((i, j))
                    edges.add((j, i))
        # Project sphere vertices onto nearest dodecahedron edge/face
        dod_t = []
        for vx, vy, vz in verts:
            best = (vx, vy, vz)
            best_d = 1e9
            for ei, ej in edges:
                p0, p1 = unique[ei], unique[ej]
                dx, dy, dz = p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]
                t = max(0, min(1, ((vx-p0[0])*dx+(vy-p0[1])*dy+(vz-p0[2])*dz)/(dx*dx+dy*dy+dz*dz+1e-12)))
                px, py, pz = p0[0]+t*dx, p0[1]+t*dy, p0[2]+t*dz
                d = (vx-px)**2+(vy-py)**2+(vz-pz)**2
                if d < best_d:
                    best_d = d
                    best = (px, py, pz)
            dod_t.append(best)
        self._orb_targets["dodeca"] = dod_t
        # Spiky targets
        spk_t = []
        for vx,vy,vz in verts:
            mag = math.sqrt(vx*vx+vy*vy+vz*vz)+1e-12
            spike = 0.5+1.5*random.random()
            spk_t.append((vx/mag*r*spike, vy/mag*r*spike, vz/mag*r*spike))
        self._orb_targets["spiky"] = spk_t

        # Wavy targets
        wav_t = []
        for vx,vy,vz in verts:
            mag = math.sqrt(vx*vx+vy*vy+vz*vz)+1e-12
            nx,ny,nz = vx/mag,vy/mag,vz/mag
            wave = 1.0+0.3*math.sin(ny*5)*math.cos(nx*4)*math.cos(nz*3)
            wav_t.append((vx*wave, vy*wave, vz*wave))
        self._orb_targets["wavy"] = wav_t

        # Build dynamic geometry
        vdata = GeomVertexData("orb", GeomVertexFormat.getV3cp(), Geom.UHDynamic)
        vdata.setNumRows(self._orb_n)
        vw = GeomVertexWriter(vdata, "vertex")
        cw = GeomVertexWriter(vdata, "color")
        for v in verts:
            vw.addData3f(*v)
            cw.addData4f(rg, gg, bg, 0.85)

        geom = Geom(vdata)
        rs = [0]
        for ring in range(n_rings+1):
            n = n_segs if ring>0 and ring<n_rings else 1
            rs.append(rs[-1]+n)
        for ring in range(n_rings):
            nc = n_segs if ring>0 and ring<n_rings else 1
            nn = n_segs if ring+1>0 and ring+1<n_rings else 1
            oc,on_ = rs[ring],rs[ring+1]
            if nc>1:
                for i in range(nc):
                    ls=GeomLines(Geom.UHStatic)
                    ls.addVertex(oc+i);ls.addVertex(oc+(i+1)%nc)
                    geom.addPrimitive(ls)
            for i in range(nc):
                j = i*nn//max(1,nc)
                if j<nn:
                    ls=GeomLines(Geom.UHStatic)
                    ls.addVertex(oc+i);ls.addVertex(on_+j)
                    geom.addPrimitive(ls)

        self._orb_geom = geom
        self._orb_vdata = vdata
        node = GeomNode("orb")
        node.addGeom(geom)
        self._orb = self.base.render.attachNewNode(node)
        self._orb.setPos(0, 10, 0)
        self._orb.setRenderModeWireframe()
        self._orb.setRenderModeThickness(1.2)
        self._orb.hide()

        def _cycle_orb():
            import random as _orb_random
            self._orb_beat_count = (self._orb_beat_count + 1) % 4  # 4-beat bars
            if self._orb_beat_count == 0:
                shape = "sphere"
            elif self._orb_beat_count in (2, 3):
                shape = _orb_random.choice(["pyramid", "dodeca", "spiky", "wavy"])
            else:
                return
            self._orb_target_verts = self._orb_targets[shape]
            self._orb_morph = 0.0

        self._cycle_orb = _cycle_orb

    def _make_moses(self):
        """Build a stylised shield-face - PS1-era low-poly icon."""
        from panda3d.core import (
            Geom, GeomTriangles, GeomLines, GeomNode, GeomVertexData,
            GeomVertexFormat, GeomVertexWriter, TransparencyAttrib,
        )

        # Shield vertices in XZ plane (Y=0 faces camera)
        points = [
            (0, -1.0),      # 0: Bottom Tip
            (0.9, 0.4),     # 1: Mid Right
            (0.8, 1.0),     # 2: Top Right
            (-0.8, 1.0),    # 3: Top Left
            (-0.9, 0.4),    # 4: Mid Left
        ]

        # --- Helper to build one shield triangle-fan layer ---
        def _make_layer(name, scale, left_rgba, right_rgba, center_rgba):
            fmt = GeomVertexFormat.getV3c4()
            vd = GeomVertexData(name, fmt, Geom.UHStatic)
            vw = GeomVertexWriter(vd, "vertex")
            cw = GeomVertexWriter(vd, "color")
            for x, z in points:
                vw.addData3(x * scale, 0, z * scale)
                if x < -0.01:
                    cw.addData4f(*left_rgba)
                elif x > 0.01:
                    cw.addData4f(*right_rgba)
                else:
                    cw.addData4f(*center_rgba)
            tris = GeomTriangles(Geom.UHStatic)
            tris.addVertices(0, 1, 2)
            tris.addVertices(0, 2, 3)
            tris.addVertices(0, 3, 4)
            g = Geom(vd)
            g.addPrimitive(tris)
            n = GeomNode(name)
            n.addGeom(g)
            return n

        # --- Build face features (line eyes + mouth) ---
        fmt = GeomVertexFormat.getV3c4()
        vd = GeomVertexData("moses_features", fmt, Geom.UHStatic)
        vw = GeomVertexWriter(vd, "vertex")
        cw = GeomVertexWriter(vd, "color")
        fc = (0.35, 0.18, 0.0, 0.85)  # dark brown

        # Left eye, right eye, mouth
        for (x0, z0), (x1, z1) in [((-0.5, 0.5), (-0.2, 0.45)),
                                     ((0.5, 0.5), (0.2, 0.45)),
                                     ((-0.4, -0.25), (0.4, -0.25))]:
            vw.addData3(x0, 0, z0); cw.addData4f(*fc)
            vw.addData3(x1, 0, z1); cw.addData4f(*fc)

        lines = GeomLines(Geom.UHStatic)
        for i in range(0, 6, 2):
            lines.addVertices(i, i + 1)
        g = Geom(vd)
        g.addPrimitive(lines)
        feat_node = GeomNode("moses_features")
        feat_node.addGeom(g)

        # --- Assemble parent ---
        self._moses = self.base.render.attachNewNode("moses")

        # Glow layer (behind, larger, translucent)
        glow_np = self._moses.attachNewNode(
            _make_layer("moses_glow", 1.18,
                        (1.0, 0.4, 0.0, 0.25),
                        (1.0, 0.9, 0.0, 0.25),
                        (1.0, 0.65, 0.0, 0.25)))
        glow_np.setY(-0.05)

        # Main face layer
        face_np = self._moses.attachNewNode(
            _make_layer("moses_face", 1.0,
                        (1.0, 0.4, 0.0, 1.0),
                        (1.0, 0.9, 0.0, 1.0),
                        (1.0, 0.65, 0.0, 1.0)))

        # Face features (slightly in front)
        feat_np = self._moses.attachNewNode(feat_node)
        feat_np.setY(0.01)
        feat_np.setRenderModeThickness(3)

        # Position, transparency, hide
        self._moses.setPos(0, 10, 1.5)
        self._moses.setTwoSided(True)
        self._moses.hide()


    def destroy(self):
        if self._milkdrop is not None:
            self._milkdrop.destroy()
        self.base.destroy()


# ---------------------------------------------------------------------------
# CLI: smoke test — open window, make actor react to synthetic audio + palette
# ---------------------------------------------------------------------------

def _cli():
    # type: () -> None
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Renderer smoke test.")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--demo", choices=["synthetic", "webcam"], default="synthetic",
                    help="Drive with a sine-wave demo or real webcam palette.")
    ap.add_argument("--debug", action="store_true",
                    help="Show debug overlay.")
    ap.add_argument("--model", type=str, default="procedural",
                    help="Path to glTF/glb model, or 'procedural' for built-in humanoid.")
    args = ap.parse_args()

    app = VJApp(window_title="vj-nano renderer smoke test", debug=args.debug, model=args.model)

    if args.demo == "webcam":
        from vj.vision.webcam import Webcam
        from vj.vision.palette import PaletteTracker
        cam = Webcam(); cam.open()
        tracker = PaletteTracker(k=5, alpha=0.3, update_every=6)

    t0 = time.monotonic()

    def step_task(task):
        t = time.monotonic() - t0
        if t >= args.seconds:
            app.base.taskMgr.remove("vj-drive")
            app.base.userExit()
            return task.done
        # Synthetic drive: sine wave energy + hue-shifting tint + mock audio features.
        energy = 0.5 + 0.5 * math.sin(t * 2.0)
        beat_phase = (t * 2.0) % 1.0
        app.set_audio_energy(energy)
        app.set_beat_phase(beat_phase)
        # Feed mock AudioFeatures so the BPM-locked animation system is active.
        mock_onset = beat_phase < 0.1 and (int(t * 2.0) > int((t - task.dt) * 2.0))
        from vj.audio.analyzer import AudioFeatures
        feat = AudioFeatures(
            rms=energy,
            bass=energy * 0.8,
            mid=0.2,
            treble=0.1,
            flux=0.5 if mock_onset else 0.05,
            onset=mock_onset,
            beat_phase=beat_phase,
            bpm=120.0,
        )
        app.set_features(feat)
        if args.demo == "webcam":
            frame, _ = cam.read()
            pal = tracker.update(frame)
            mid = pal[2]
            app.set_tint(np.array([mid[2] / 255.0, mid[1] / 255.0, mid[0] / 255.0]))
        else:
            hue = (t * 0.3) % 1.0
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
            app.set_tint(np.array([r, g, b]))
        from direct.task import Task
        return Task.cont

    app.base.taskMgr.add(step_task, "vj-drive")
    try:
        app.run()
    finally:
        if args.demo == "webcam":
            cam.close()


if __name__ == "__main__":
    _cli()