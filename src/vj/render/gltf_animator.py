"""Audio-reactive animator for glTF Actors.

Single looping dance animation with BPM-synced playback rate
so the loop length equals an integer number of beats.
"""

import math

import numpy as np


class GltfAnimator(object):
    """Drive a GltfActor with a single looping dance animation."""

    def __init__(self, actor, base_bpm=120.0, dance_anim="Dance",
                 beats_per_loop=4.0):
        # type: (object, float, str, float) -> None
        self.actor = actor
        self._base_bpm = float(base_bpm)
        self._bpm = self._base_bpm
        self._dance_anim = dance_anim
        self._beats_per_loop = float(beats_per_loop)

        # Query true animation duration (seconds) from the Actor
        self._anim_duration = self._query_duration(dance_anim)
        if self._anim_duration <= 0.0:
            self._anim_duration = 3.33  # fallback for RobotExpressive

        # Compute play-rate so loop = N beats at base BPM
        base_loop_period = 60.0 / self._base_bpm * self._beats_per_loop
        self._base_rate = self._anim_duration / base_loop_period

        # Start the dance loop
        self.actor.play_anim(self._dance_anim, loop=True)
        self.actor.set_play_rate(self._base_rate)

        self._last_onset_t = -1.0
        self._last_sync_t = -1.0

        # Root motion state
        self._bounce = 0.0
        self._sway = 0.0
        self._pulse = 0.0
        self._last_scale = 1.0

    def _query_duration(self, anim_name):
        # type: (str) -> float
        try:
            a = self.actor.actor
            nf = float(a.getNumFrames(anim_name))
            fr = float(a.getFrameRate(anim_name))
            if fr > 0.0:
                return nf / fr
        except Exception:
            pass
        return 0.0

    def _anim_time(self):
        # type: () -> float
        """Current time (seconds) into the dance animation."""
        try:
            a = self.actor.actor
            return float(a.getCurrentFrame(self._dance_anim)) / float(
                a.getFrameRate(self._dance_anim))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def update(self, feat, dt):
        # type: (object, float) -> tuple
        """Process one frame of audio features.

        Returns (hpr, scale, pos) for the *root* node.
        """
        self.actor.update_blend(dt)

        # --- BPM-sync play rate ----------------------------------------
        if feat.bpm > 0:
            self._bpm = feat.bpm
        loop_period = 60.0 / self._bpm * self._beats_per_loop
        play_rate = self._anim_duration / loop_period
        self.actor.set_play_rate(play_rate)

        # --- soft beat-sync: nudge animation on onset ------------------
        if feat.onset and feat.t - self._last_onset_t > 0.15:
            self._last_onset_t = feat.t
            self._pulse = 1.0

            # Only hard-snap if drift is > 0.15s (avoids frozen look)
            try:
                t_anim = self._anim_time()
                if t_anim > 0.15:
                    self.actor.actor.setTime(0.0, self._dance_anim)
            except Exception:
                pass

        # Decay pulse
        self._pulse = max(0.0, self._pulse - dt * 3.5)

        # --- gentle beat-phase bounce & sway ---------------------------
        phase = feat.beat_phase
        self._bounce = 0.04 * math.sin(phase * 2.0 * math.pi)

        # Slow continuous rotation driven by BPM
        self._sway += dt * 20.0 * (self._bpm / 120.0)

        # On-beat rotation impulse
        rot_impulse = 6.0 * self._pulse * math.sin(phase * math.pi)

        self.actor.root.setHpr(
            self._sway + rot_impulse,
            0.0,
            0.0,
        )
        self.actor.root.setZ(self._bounce + 0.03 * self._pulse)

        scale = 1.0 + 0.03 * self._pulse + 0.02 * math.sin(
            phase * 2.0 * math.pi)
        self._last_scale = scale

        return (0.0, 0.0, 0.0), scale, (0.0, 0.0, self._bounce)

    @property
    def scale(self):
        # type: () -> float
        return self._last_scale
