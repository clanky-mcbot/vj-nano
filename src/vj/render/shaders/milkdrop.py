"""MilkDrop-style GPU visualizer for vj-nano."""

from __future__ import print_function
import os

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SHADER_DIR = os.path.join(_THIS_DIR, "shaders")


class MilkDropRenderer(object):
    """MilkDrop-style background via billboarded shader quad."""

    PRESETS = [
        ("test_red", "Test Red", "test_red.glsl"),
        ("plasma",  "Plasma",   "milkdrop_plasma.glsl"),
        ("fractal", "Fractal",  "milkdrop_fractal.glsl"),
        ("tunnel",  "Tunnel",   "milkdrop_tunnel.glsl"),
        ("wave",    "Wave",     "milkdrop_wave.glsl"),
    ]

    def __init__(self, base):
        from panda3d.core import CardMaker, Shader
        self._base = base
        self._enabled = True
        self._active_preset = 0
        self._shaders = {}
        self._quad = None

        vert_path = os.path.join(_SHADER_DIR, "milkdrop_vert.glsl")

        for name, _label, filename in self.PRESETS:
            frag_path = os.path.join(_SHADER_DIR, filename)
            if os.path.isfile(vert_path) and os.path.isfile(frag_path):
                # Use makeFragmentShader for maximum compatibility
                with open(frag_path, "r") as f:
                    src = f.read()
                shader = Shader.makeFragmentShader(src)
                if shader:
                    self._shaders[name] = shader
                    print("[milkdrop] loaded preset '{}' ({})".format(name, filename))
                else:
                    print("[milkdrop] warning: '{}' compile failed".format(filename))
                    self._shaders[name] = None
            else:
                print("[milkdrop] warning: shader files not found")
                self._shaders[name] = None

        # Billboarded quad in 3D space
        cm = CardMaker("milkdrop_quad")
        cm.setFrame(-1.0, 1.0, -1.0, 1.0)
        self._quad = base.render.attachNewNode(cm.generate())
        self._quad.setPos(0, 30, 2)
        self._quad.setBillboardPointEye()
        self._quad.setDepthWrite(False)
        self._quad.setTwoSided(True)
        self._quad.setScale(50)
        # NO setColor — shader controls output directly
        self._apply_active_shader()

        print("[milkdrop] MilkDrop visualizer ready")

    def _apply_active_shader(self):
        if self._quad is None: return
        name = self.PRESETS[self._active_preset][0]
        shader = self._shaders.get(name)
        if shader is not None:
            self._quad.setShader(shader)
        else:
            self._quad.setShaderOff(1)

    @property
    def preset_name(self): return self.PRESETS[self._active_preset][0]
    @property
    def preset_label(self): return self.PRESETS[self._active_preset][1]

    def set_enabled(self, val):
        self._enabled = val
        if self._quad is not None:
            self._quad.setScale(50 if val else 0.001)

    def next_preset(self):
        self._active_preset = (self._active_preset + 1) % len(self.PRESETS)
        self._apply_active_shader()
        return self.preset_label

    def update(self, bass, mid, treble, volume, onset, energy, dt):
        if self._quad is None or not self._enabled: return
        t = self._base.taskMgr.globalClock.getFrameTime()
        win = self._base.win
        self._quad.setShaderInput("u_time", t)
        self._quad.setShaderInput("u_bass", bass)
        self._quad.setShaderInput("u_mid", mid)
        self._quad.setShaderInput("u_treble", treble)
        self._quad.setShaderInput("u_volume", volume)
        self._quad.setShaderInput("u_energy", energy)
        self._quad.setShaderInput("u_onset", 1.0 if onset else 0.0)
        self._quad.setShaderInput("u_dt", dt)
        self._quad.setShaderInput("u_resolution", (float(win.getXSize()), float(win.getYSize())))

    def destroy(self):
        if self._quad is not None:
            self._quad.removeNode()
            self._quad = None
