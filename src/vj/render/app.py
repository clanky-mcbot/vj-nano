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
        if self._debug:
            self._build_debug_overlay()

        # Create menu AFTER debug nodes are populated so it gets the real list
        if _GUI_AVAILABLE and self._fx is not None:
            self._menu = EffectMenu(self.base, self._fx, self._filters, self._debug_nodes)
            print("[render] Effect menu loaded")
            # Bind 'H' to toggle all debug overlays
            self.base.accept("h", self._toggle_debug)
            self.base.accept("H", self._toggle_debug)

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
                self._character.setShaderInput("ps1_dither_amount", 1.5)
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
            text="BPM: --",
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
            pos=(-0.95, 0.82),
            scale=0.04,
            fg=(0.7, 0.7, 0.7, 1.0),
            align=0,
            parent=self.base.aspect2d,
        )
        self._debug_nodes.append(self._txt_rms)
        self._debug_original_scales[self._txt_rms] = self._txt_rms.getScale()

        self._txt_onset = OnscreenText(
            text="ONSET",
            pos=(-0.95, 0.74),
            scale=0.05,
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
        self._txt_bpm.setText("BPM: {:.1f}".format(bpm))
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

        # Vertex-color tint
        r, g, b = float(self._tint[0]), float(self._tint[1]), float(self._tint[2])
        self._character.setColorScale(r, g, b, 1.0)
        from direct.task import Task
        return Task.cont

    def run(self):
        """Block on the Panda3D main loop. Press ESC or close window to exit."""
        self.base.run()

    def destroy(self):
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