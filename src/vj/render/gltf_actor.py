"""Wrapper for glTF animated characters loaded via panda3d-gltf.

Provides a uniform interface so the renderer and animator can treat
external GLB actors the same way as our procedural PS1Humanoid.
"""

import os

import gltf  # noqa: F401 — registers loader with Panda3D
from direct.actor.Actor import Actor
from panda3d.core import NodePath


class GltfActor(object):
    """Low-poly GLB character with animation state machine + cross-fading."""

    def __init__(self, parent, glb_path, crossfade_duration=0.12):
        # type: (NodePath, str, float) -> None
        if not os.path.isfile(glb_path):
            raise FileNotFoundError("GLB not found: {}".format(glb_path))

        self.root = parent.attachNewNode("gltf-actor-root")
        self.actor = Actor(glb_path)
        self.actor.reparentTo(self.root)
        self.actor.enableBlend()          # ← enable animation blending

        self._anims = list(self.actor.getAnimNames())
        self._target_anim = None          # anim we want at full weight
        self._active_weights = {}         # anim_name -> current_weight
        self._crossfade_duration = float(crossfade_duration)
        self._play_rate = 1.0

    # ------------------------------------------------------------------
    # Animation control
    # ------------------------------------------------------------------
    def list_anims(self):
        # type: () -> list
        return self._anims[:]

    def play_anim(self, name, loop=True, restart=False):
        # type: (str, bool, bool) -> None
        """Set *name* as the target animation.  Actual weight ramp happens
        inside update_blend()."""
        if name not in self._anims:
            return
        if not restart and name == self._target_anim:
            # Just keep play-rate up to date
            self.actor.setPlayRate(self._play_rate, name)
            return

        self._target_anim = name

        # Start the new animation at weight 0 if not already tracked
        if name not in self._active_weights:
            if loop:
                self.actor.loop(name)
            else:
                self.actor.play(name)
            self.actor.setControlEffect(name, 0.0)
            self._active_weights[name] = 0.0

        self.actor.setPlayRate(self._play_rate, name)

    def update_blend(self, dt):
        # type: (float) -> None
        """Advance cross-fade weights by one frame.  Call from the
        animator's update() each frame."""
        if not self._active_weights:
            return

        speed = dt / self._crossfade_duration

        for anim_name in list(self._active_weights.keys()):
            target = 1.0 if anim_name == self._target_anim else 0.0
            current = self._active_weights[anim_name]

            if abs(target - current) <= speed:
                new_weight = target
            else:
                new_weight = current + speed * (1.0 if target > current else -1.0)

            self._active_weights[anim_name] = new_weight
            self.actor.setControlEffect(anim_name, new_weight)

            # Stop animations that have fully faded out
            if new_weight <= 0.0 and anim_name != self._target_anim:
                self.actor.stop(anim_name)
                del self._active_weights[anim_name]

    def stop(self):
        # type: () -> None
        self._target_anim = None
        for anim_name in list(self._active_weights.keys()):
            self.actor.stop(anim_name)
        self._active_weights.clear()

    def set_play_rate(self, rate):
        # type: (float) -> None
        """Scale animation speed (1.0 = default)."""
        self._play_rate = float(rate)
        if self._target_anim and self._target_anim in self._active_weights:
            self.actor.setPlayRate(self._play_rate, self._target_anim)

    # ------------------------------------------------------------------
    # Passthroughs so the renderer can treat this like a NodePath
    # ------------------------------------------------------------------
    def setPos(self, *args):
        self.root.setPos(*args)

    def setHpr(self, *args):
        self.root.setHpr(*args)

    def setScale(self, *args):
        self.root.setScale(*args)

    def setColorScale(self, *args):
        self.root.setColorScale(*args)

    def setShader(self, *args, **kwargs):
        self.root.setShader(*args, **kwargs)

    def setShaderInput(self, *args, **kwargs):
        self.root.setShaderInput(*args, **kwargs)
