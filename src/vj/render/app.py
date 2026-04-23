"""Panda3D renderer module.

v1 scope: open a window, load/display a placeholder PS1-style cube, drive
its rotation with audio band energies and its tint with webcam palette
colors.

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

    def __init__(self, window_title="vj-nano", win_size=(1280, 720)):
        # type: (str, tuple) -> None
        _load_prc()
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "window-title {}".format(window_title))
        loadPrcFileData(
            "", "win-size {} {}".format(win_size[0], win_size[1]))

        from direct.showbase.ShowBase import ShowBase
        self.base = ShowBase()
        # Dark slate background — matches the Nous palette.
        from panda3d.core import ClearableRegion
        self.base.setBackgroundColor(0.06, 0.08, 0.10, 1.0)

        self._character = self._make_placeholder_character()
        self._character.reparentTo(self.base.render)
        self._character.setPos(0, 8, 0)

        # Tilt camera slightly down so the cube is center-framed.
        self.base.camera.setPos(0, 0, 1.5)
        self.base.camera.lookAt(0, 8, 0)

        # State fed from outside each frame.
        self._rotation = 0.0
        self._energy = 0.0
        self._tint = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # RGB 0..1

        # Drive rotation/tint every frame via a task.
        self.base.taskMgr.add(self._update_task, "vj-update")

    # ------------------------------------------------------------------
    def _make_placeholder_character(self):
        """Build a tiny procedural PS1-style cube.

        We'll replace this with a proper rigged model later. For now it
        proves the pipeline: geometry on screen, color uniforms reachable.
        """
        from panda3d.core import (
            GeomNode, Geom, GeomVertexData, GeomVertexFormat, GeomVertexWriter,
            GeomTriangles, NodePath,
        )
        fmt = GeomVertexFormat.getV3n3c4()
        vdata = GeomVertexData("cube", fmt, Geom.UHStatic)
        vwrite = GeomVertexWriter(vdata, "vertex")
        nwrite = GeomVertexWriter(vdata, "normal")
        cwrite = GeomVertexWriter(vdata, "color")

        # Six faces × 4 vertices × (xyz + normal + rgba)
        faces = [
            # (normal, 4 corners)
            ((0, 0, 1),  [(-1, -1, 1), (1, -1, 1),  (1, 1, 1),  (-1, 1, 1)]),   # +Z
            ((0, 0, -1), [(-1, 1, -1), (1, 1, -1),  (1, -1, -1), (-1, -1, -1)]),# -Z
            ((1, 0, 0),  [(1, -1, -1), (1, 1, -1),  (1, 1, 1),  (1, -1, 1)]),   # +X
            ((-1, 0, 0), [(-1, -1, 1), (-1, 1, 1),  (-1, 1, -1), (-1, -1, -1)]),# -X
            ((0, 1, 0),  [(-1, 1, -1), (-1, 1, 1),  (1, 1, 1),  (1, 1, -1)]),   # +Y
            ((0, -1, 0), [(1, -1, -1), (1, -1, 1),  (-1, -1, 1), (-1, -1, -1)]),# -Y
        ]
        tri = GeomTriangles(Geom.UHStatic)
        vi = 0
        for normal, corners in faces:
            for corner in corners:
                vwrite.addData3(*corner)
                nwrite.addData3(*normal)
                cwrite.addData4(1.0, 1.0, 1.0, 1.0)  # will be tinted later
            tri.addVertices(vi, vi + 1, vi + 2)
            tri.addVertices(vi, vi + 2, vi + 3)
            vi += 4

        geom = Geom(vdata)
        geom.addPrimitive(tri)
        node = GeomNode("placeholder-character")
        node.addGeom(geom)
        return NodePath(node)

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _update_task(self, task):
        dt = self.base.taskMgr.globalClock.getDt()
        # Rotate at a rate proportional to audio energy.
        self._rotation += dt * (30.0 + 240.0 * self._energy)
        self._character.setHpr(self._rotation, 15.0, 0.0)
        # Scale breathing on beat phase (if set).
        phase = getattr(self, "_beat_phase", 0.0)
        pulse = 1.0 + 0.06 * math.cos(phase * 2.0 * math.pi)
        self._character.setScale(pulse)
        # Vertex-color tint (cheap placeholder; proper shader comes later).
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
# CLI: smoke test — open window, make cube react to synthetic audio + palette
# ---------------------------------------------------------------------------

def _cli():
    # type: () -> None
    import argparse
    import time

    ap = argparse.ArgumentParser(description="Renderer smoke test.")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--demo", choices=["synthetic", "webcam"], default="synthetic",
                    help="Drive with a sine-wave demo or real webcam palette.")
    args = ap.parse_args()

    app = VJApp(window_title="vj-nano renderer smoke test")

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
        # Synthetic drive: sine wave energy + hue-shifting tint.
        energy = 0.5 + 0.5 * math.sin(t * 2.0)
        app.set_audio_energy(energy)
        app.set_beat_phase((t * 2.0) % 1.0)  # fake 120 BPM
        if args.demo == "webcam":
            frame, _ = cam.read()
            pal = tracker.update(frame)
            # Use the mid-brightness color as tint (idx 2 of 5, sorted by lum).
            mid = pal[2]
            # Palette is BGR; Panda3D tint is RGB.
            app.set_tint(np.array([mid[2] / 255.0, mid[1] / 255.0, mid[0] / 255.0]))
        else:
            # Rotate tint through hue for visual demo.
            hue = (t * 0.3) % 1.0
            # Simple HSV->RGB (S=V=1).
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
    import math
    _cli()
