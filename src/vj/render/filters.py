"""CPU post-process filters — framebuffer capture via getScreenshot() + numpy."""

from __future__ import print_function
import numpy as np

_ASCII_RAMP = " .:-=+*#%@"
_ASCII_W, _ASCII_H = 80, 45

_BAYER = np.array([
    [ 0,  8,  2, 10], [12,  4, 14,  6],
    [ 3, 11,  1,  9], [15,  7, 13,  5],
], dtype=np.float32) / 16.0


class PostProcessFilters(object):
    def __init__(self, base):
        from panda3d.core import Texture, CardMaker
        self._base = base
        self._enabled = {
            "dither": False, "scanlines": False,
            "pixelate": False, "vignette": True, "ascii": False,
        }
        self._ascii_enabled = False
        self._frame = 0

        win = base.win
        self._w, self._h = win.getXSize(), win.getYSize()
        self._pw, self._ph = self._w // 2, self._h // 2

        self._tex = Texture("cpu-pp")
        self._tex.setup2dTexture(self._pw, self._ph, Texture.TUnsignedByte, Texture.FRgb)

        cm = CardMaker("pp-quad")
        cm.setFrameFullscreenQuad()
        self._quad = base.render2d.attachNewNode(cm.generate())
        self._quad.setTexture(self._tex)
        self._quad.setBin("fixed", 0)
        self._quad.setDepthTest(False)
        self._quad.setDepthWrite(False)
        self._quad.setTransparency(0)

        # Precompute vignette
        ys, xs = np.ogrid[:self._ph, :self._pw]
        dist = np.sqrt((xs - self._pw/2.)**2 + (ys - self._ph/2.)**2) / max(self._pw/2., self._ph/2.)
        self._vignette = np.clip(1.0 - dist * 0.55, 0.25, 1.0).astype(np.float32)

        tile_y, tile_x = self._ph // 4 + 1, self._pw // 4 + 1
        self._bayer_tile = np.tile(_BAYER, (tile_y, tile_x))[:self._ph, :self._pw]

        print("[render] CPU post-process ready ({}x{})".format(self._pw, self._ph))

        self._ascii_text = None
        self._build_ascii_overlay()

    def _build_ascii_overlay(self):
        from direct.gui.OnscreenText import OnscreenText
        self._ascii_text = OnscreenText(
            text="", pos=(0, 0), scale=0.018,
            fg=(0.8, 0.9, 0.7, 1.0), align=1, font=None,
            parent=self._base.aspect2d, wordwrap=200,
        )
        self._ascii_text.hide()

    def set_enabled(self, name, val):
        if name == "ascii":
            self._ascii_enabled = bool(val)
            self._update_ascii_visibility()
            return
        if name in self._enabled:
            self._enabled[name] = bool(val)

    def set_clear_color(self, r, g, b, a):
        pass

    def _update_ascii_visibility(self):
        if self._ascii_enabled:
            self._ascii_text.show()
        else:
            self._ascii_text.hide()

    def update(self, dt):
        self._frame += 1
        if self._frame % 6 != 0:
            return

        any_fx = any(self._enabled[k] for k in ["dither", "pixelate", "scanlines", "vignette"])
        if not any_fx:
            return

        try:
            self._process()
        except Exception:
            pass

    def _process(self):
        from panda3d.core import PNMImage

        # Capture framebuffer via getScreenshot (no disk I/O)
        ss = PNMImage()
        if not self._base.win.getScreenshot(ss):
            return

        src_w, src_h = ss.getXSize(), ss.getYSize()

        # Downsample to half-res and convert to numpy
        rgb = np.zeros((self._ph, self._pw, 3), dtype=np.float32)
        x_ratio = float(src_w) / self._pw
        y_ratio = float(src_h) / self._ph

        for py in range(self._ph):
            sy = int(py * y_ratio)
            for px in range(self._pw):
                sx = int(px * x_ratio)
                c = ss.getXel(sx, sy)
                rgb[py, px, 0] = c[0]
                rgb[py, px, 1] = c[1]
                rgb[py, px, 2] = c[2]

        # --- Apply effects ---
        if self._enabled["pixelate"]:
            px_sz = 6
            sh, sw = max(1, self._ph // px_sz), max(1, self._pw // px_sz)
            small = np.zeros((sh, sw, 3), dtype=np.float32)
            for sy in range(sh):
                y0, y1 = sy * px_sz, min((sy + 1) * px_sz, self._ph)
                for sx in range(sw):
                    x0, x1 = sx * px_sz, min((sx + 1) * px_sz, self._pw)
                    small[sy, sx] = rgb[y0:y1, x0:x1].mean(axis=(0, 1))
            for sy in range(sh):
                y0, y1 = sy * px_sz, min((sy + 1) * px_sz, self._ph)
                for sx in range(sw):
                    x0, x1 = sx * px_sz, min((sx + 1) * px_sz, self._pw)
                    rgb[y0:y1, x0:x1] = small[sy, sx]

        if self._enabled["dither"]:
            threshold = (self._bayer_tile - 0.5) / 6.0
            rgb = np.floor(rgb * 6.0 + threshold[:, :, None]) / 6.0
            rgb = np.clip(rgb, 0.0, 1.0)

        if self._enabled["scanlines"]:
            line = 0.88 + 0.12 * np.sin(np.arange(self._ph)[:, None] * 1.2)
            rgb *= line[:, :, None]

        if self._enabled["vignette"] or any(
            self._enabled[k] for k in ["dither", "scanlines", "pixelate"]
        ):
            rgb *= self._vignette[:, :, None]

        out = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        self._tex.setRamImage(out.tobytes())
