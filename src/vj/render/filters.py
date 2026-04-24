"""Post-processing filter system for vj-nano.

Uses Panda3D FilterManager for shader-based effects (dither, scanlines,
pixelate) and a CPU-based ASCII-art overlay updated every N frames.

Python 3.6 compatible.
"""

from __future__ import print_function

import os
import numpy as np

# ASCII character ramp from dark to light
_ASCII_RAMP = " .:-=+*#%@"
_ASCII_W = 80
_ASCII_H = 45


class PostProcessFilters(object):
    """Manages post-processing effects via FilterManager + ASCII overlay."""

    def __init__(self, base):
        # type: (ShowBase) -> None
        self._base = base
        self._enabled = {
            "dither": False,
            "scanlines": False,
            "pixelate": False,
            "vignette": True,
        }
        self._ascii_enabled = False

        # --- Shader-based filters via FilterManager ---
        from direct.filter.FilterManager import FilterManager
        from panda3d.core import Texture, Shader

        self._fm = FilterManager(base.win, base.cam)
        self._tex = Texture()
        self._quad = self._fm.renderSceneInto(colortex=self._tex)

        # The Jetson's older Panda3D build sometimes fails to auto-clear the
        # offscreen buffer created by FilterManager.  Force it here.
        if self._quad is not None and self._fm.buffers:
            buf = self._fm.buffers[0]
            # Keep FilterManager's default buffer clear (it clears before
            # render). Just sync the clear colour to our background.
            buf.setClearColorActive(True)
            buf.setClearDepthActive(True)
            buf.setClearColor(base.win.getClearColor())
            print("[render] FilterManager buffer clear configured")

        # FilterManager disables clears on the main window assuming the
        # fullscreen opaque quad covers everything. Re-enable them so the
        # window clears even if the quad has alpha issues.
        if self._quad is not None:
            base.win.setClearColorActive(True)
            base.win.setClearDepthActive(True)
            print("[render] Main window clear restored")

        shader_dir = os.path.join(os.path.dirname(__file__), "shaders")
        vert = os.path.join(shader_dir, "postprocess_vert.glsl")
        frag = os.path.join(shader_dir, "postprocess_frag.glsl")

        if os.path.isfile(vert) and os.path.isfile(frag):
            shader = Shader.load(Shader.SL_GLSL, vert, frag)
            if shader:
                self._quad.setShader(shader)
                self._quad.setShaderInput("tex", self._tex)
                self._quad.setShaderInput("resolution", (base.win.getXSize(), base.win.getYSize()))
                self._update_shader_uniforms()
                print("[render] Post-process filters loaded")
            else:
                print("[render] warning: post-process shader compile failed")
        else:
            print("[render] warning: post-process shader files not found")

        # --- ASCII overlay (CPU-based) ---
        self._ascii_text = None  # type: Optional[OnscreenText]
        self._ascii_frame = 0
        self._build_ascii_overlay()

    def _build_ascii_overlay(self):
        # type: () -> None
        from direct.gui.OnscreenText import OnscreenText
        # Monospace block of text that covers the screen
        self._ascii_text = OnscreenText(
            text="",
            pos=(0, 0),
            scale=0.018,
            fg=(0.8, 0.9, 0.7, 1.0),
            align=1,  # center
            font=None,
            parent=self._base.aspect2d,
            wordwrap=200,
        )
        self._ascii_text.hide()

    def _update_shader_uniforms(self):
        # type: () -> None
        if self._quad is None:
            return
        self._quad.setShaderInput("enable_dither", 1.0 if self._enabled["dither"] else 0.0)
        self._quad.setShaderInput("enable_scanlines", 1.0 if self._enabled["scanlines"] else 0.0)
        self._quad.setShaderInput("enable_pixelate", 1.0 if self._enabled["pixelate"] else 0.0)
        self._quad.setShaderInput("enable_vignette", 1.0 if self._enabled["vignette"] else 0.0)

    def set_enabled(self, name, val):
        # type: (str, bool) -> None
        if name == "ascii":
            self._ascii_enabled = val
            self._update_ascii_visibility()
            return
        if name in self._enabled:
            self._enabled[name] = val
            self._update_shader_uniforms()

    def set_clear_color(self, r, g, b, a):
        # type: (float, float, float, float) -> None
        """Propagate background colour changes to the FilterManager buffer."""
        if self._fm.buffers:
            self._fm.buffers[0].setClearColor((r, g, b, a))

    def _update_ascii_visibility(self):
        # type: () -> None
        if self._ascii_enabled:
            self._quad.setScale(0.01)  # hide normal render
            self._ascii_text.show()
        else:
            self._quad.setScale(1)
            self._ascii_text.hide()

    def update(self, dt):
        # type: (float) -> None
        if not self._ascii_enabled:
            return
        self._ascii_frame += 1
        # Update ASCII art every 6 frames (~10 Hz at 60 fps)
        if self._ascii_frame % 6 != 0:
            return
        self._regen_ascii()

    def _regen_ascii(self):
        # type: () -> None
        """Read the scene texture and convert to ASCII art."""
        try:
            # Get texture data
            tex = self._tex
            x_size = tex.getXSize()
            y_size = tex.getYSize()
            if x_size == 0 or y_size == 0:
                return

            # Read pixels as RGB
            data = tex.getRamImageAs("RGB")
            if data is None or data.getNumRows() == 0:
                return

            arr = np.frombuffer(data, dtype=np.uint8)
            arr = arr.reshape((y_size, x_size, 3))

            # Downsample to ASCII grid
            h, w = _ASCII_H, _ASCII_W
            # Simple block averaging
            block_y = max(1, y_size // h)
            block_x = max(1, x_size // w)

            lines = []
            for row in range(h):
                y0 = row * block_y
                y1 = min(y0 + block_y, y_size)
                line_chars = []
                for col in range(w):
                    x0 = col * block_x
                    x1 = min(x0 + block_x, x_size)
                    block = arr[y0:y1, x0:x1]
                    lum = np.mean(block) / 255.0
                    idx = int(lum * (len(_ASCII_RAMP) - 1))
                    idx = max(0, min(idx, len(_ASCII_RAMP) - 1))
                    line_chars.append(_ASCII_RAMP[idx])
                lines.append("".join(line_chars))

            ascii_str = "\n".join(lines)
            self._ascii_text.setText(ascii_str)

        except Exception as exc:
            # Don't crash the render loop for ASCII errors
            print("[render] ASCII filter error:", exc)
