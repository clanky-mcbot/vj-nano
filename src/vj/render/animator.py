"""Beat-reactive choreography engine v2.

Drives a PS1Humanoid via BPM-locked animation clips.  Replaces the raw
transform math from v1 with a proper animation mixer that plays clips at
rates determined by detected tempo.

Design:
  * BpmClock        — converts real-time to beat-time from detected BPM.
  * AnimationMixer  — blends looping clips (idle, groove) + one-shots (jump).
  * BeatAnimator    — maps audio features to mixer weights & root motion.
"""

import math

import numpy as np

from vj.render.actor import PS1Humanoid
from vj.render.animation import (
    AnimationMixer, BpmClock,
    make_idle_clip, make_groove_clip, make_jump_clip,
)


class BeatAnimator(object):
    """Stateless-ish animator: feed it AudioFeatures, it drives the actor."""

    def __init__(self, actor):
        # type: (PS1Humanoid) -> None
        self.actor = actor
        self.clock = BpmClock(default_bpm=120.0)
        self.mixer = AnimationMixer(self.clock)
        self.mixer.add_clip(make_idle_clip())
        self.mixer.add_clip(make_groove_clip())
        self.mixer.add_clip(make_jump_clip())
        self.mixer.set_weight("idle", 1.0)
        self.mixer.set_weight("groove", 0.0)
        self.mixer.set_weight("jump", 0.0)

        # Root-motion impulse state
        self._beat_hpr = np.zeros(3, dtype=np.float32)
        self._beat_pos = np.zeros(3, dtype=np.float32)
        self._onset_cooldown = 0.0
        self._spin = 0.0

        # Cached pose for external readers
        self._last_pose = {}  # type: dict
        self._last_scale = 1.0

    # ------------------------------------------------------------------
    def update(self, feat, dt):
        # type: (object, float) -> tuple
        """Process one frame of audio features.

        Args:
            feat: AudioFeatures instance (rms, bass, mid, treble,
                  onset, beat_phase, bpm).
            dt:   Frame delta-time in seconds.

        Returns:
            (hpr, scale, pos) for the *root* node — the caller can still
            apply additional transform if desired.
        """
        # --- update BPM clock ------------------------------------------
        if feat.bpm > 0:
            self.clock.set_bpm(feat.bpm)

        # --- blend weights based on energy -----------------------------
        energy = min(feat.rms * 2.5, 1.0)
        self.mixer.set_weight("idle", 1.0 - energy * 0.85)
        self.mixer.set_weight("groove", energy * 0.85)

        # --- trigger one-shot on onset ---------------------------------
        if feat.onset and self._onset_cooldown <= 0.0:
            self._onset_cooldown = 0.18
            intensity = 0.6 + 0.4 * min(feat.rms * 3.0, 1.0)

            if feat.bass > feat.mid and feat.bass > feat.treble:
                # Bass drop: big jump
                self.mixer.trigger_one_shot("jump", duration_beats=0.75)
                self._beat_pos[2] = 0.25 * intensity
                self._beat_hpr[1] += 15.0 * intensity
            elif feat.treble > feat.mid:
                # Treble hit: fast spin burst
                self._spin += 90.0 * intensity
                self._beat_hpr[0] += 20.0 * intensity
            else:
                # Mid-range groove: gentle hop
                self.mixer.trigger_one_shot("jump", duration_beats=0.5)
                self._beat_pos[2] = 0.10 * intensity
                self._beat_hpr[0] += 10.0 * intensity

        # Decay impulses
        decay = math.exp(-10.0 * dt)
        self._beat_hpr *= decay
        self._beat_pos *= decay
        self._onset_cooldown = max(0.0, self._onset_cooldown - dt)
        self._spin *= math.exp(-2.0 * dt)

        # --- sample animation pose -------------------------------------
        pose = self.mixer.update(dt)
        self._last_pose = pose
        self.actor.apply_pose(pose)

        # --- add root motion on top ------------------------------------
        # Spin accumulates on heading
        h, p, r = self.actor.hips.getHpr()
        self.actor.hips.setHpr(h + self._spin * dt + self._beat_hpr[0], p + self._beat_hpr[1], r + self._beat_hpr[2])

        # Vertical bounce from beat phase (subtle)
        phase = feat.beat_phase
        base_bounce = 0.02 * math.cos(phase * 2.0 * math.pi)
        z = self.actor.hips.getZ()
        self.actor.hips.setZ(z + base_bounce + self._beat_pos[2])

        # Scale pulse
        scale = 1.0 + 0.04 * math.cos(phase * 2.0 * math.pi)
        self._last_scale = scale

        return (self.actor.hips.getHpr()[0], 0.0, 0.0), scale, self._beat_pos.copy()

    @property
    def scale(self):
        # type: () -> float
        return self._last_scale
