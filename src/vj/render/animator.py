"""Beat-reactive choreography engine.

Drives NodePath transforms from AudioFeatures.  Model-agnostic: works on a
procedural cube today and on a rigged character tomorrow.

Design:
  * Base layer    — continuous idle sway locked to beat_phase.
  * Beat layer    — impulse responses triggered by onset detection.
  * Energy layer  — slow modulation tied to overall RMS.

The three layers are composited each frame into (HPR, scale, position).
"""

# Python 3.6 compatible — no f-strings, no walrus.

import math

import numpy as np


class BeatAnimator(object):
    """Stateless-ish animator: feed it AudioFeatures, read transforms."""

    def __init__(self):
        # type: () -> None
        # Beat-layer impulse state (decays each frame).
        self._beat_hpr = np.zeros(3, dtype=np.float32)
        self._beat_scale = 1.0
        self._beat_pos = np.zeros(3, dtype=np.float32)
        self._onset_cooldown = 0.0

        # Persistent rotation (spin accumulates across beats).
        self._spin = 0.0

    # ------------------------------------------------------------------
    def update(self, feat, dt):
        # type: (object, float) -> tuple
        """Process one frame of audio features.

        Args:
            feat: AudioFeatures instance (rms, bass, mid, treble,
                  onset, beat_phase, bpm).
            dt:   Frame delta-time in seconds (from renderer clock).

        Returns:
            (hpr, scale, pos) where
                hpr   = (heading, pitch, roll) in degrees
                scale = uniform scale multiplier
                pos   = (x, y, z) offset
        """
        phase = feat.beat_phase

        # --- base layer: idle sway locked to beat --------------------
        base_h = 15.0 * math.sin(phase * 2.0 * math.pi)
        base_p = 8.0 * math.sin(phase * 4.0 * math.pi)
        base_r = 5.0 * math.cos(phase * 3.0 * math.pi)
        base_hpr = np.array([base_h, base_p, base_r], dtype=np.float32)

        # --- beat layer: impulse on onset ----------------------------
        if feat.onset and self._onset_cooldown <= 0.0:
            self._onset_cooldown = 0.18  # ~3 frames @ 60 Hz debounce

            intensity = 0.6 + 0.4 * min(feat.rms * 3.0, 1.0)

            if feat.bass > feat.mid and feat.bass > feat.treble:
                # Bass drop: heavy pulse + upward kick
                self._beat_scale = 1.0 + 0.35 * intensity
                self._beat_pos[2] = 0.6 * intensity
                self._beat_hpr[1] += 20.0 * intensity  # pitch nod

            elif feat.treble > feat.mid:
                # Treble hit: fast spin burst
                self._spin += 90.0 * intensity
                self._beat_scale = 1.0 + 0.15 * intensity

            else:
                # Mid-range groove: gentle scale breathe
                self._beat_scale = 1.0 + 0.20 * intensity
                self._beat_hpr[0] += 25.0 * intensity

        # Decay beat impulses with exponential damping.
        decay = math.exp(-10.0 * dt)
        self._beat_hpr *= decay
        self._beat_pos *= decay
        self._beat_scale = 1.0 + (self._beat_scale - 1.0) * decay
        self._onset_cooldown = max(0.0, self._onset_cooldown - dt)

        # Spin is persistent but damped slightly so it doesn't run away.
        self._spin *= math.exp(-2.0 * dt)
        base_hpr[0] += self._spin

        # --- energy layer: slow RMS modulation -----------------------
        energy_scale = 1.0 + 0.08 * feat.rms

        # --- composite ------------------------------------------------
        hpr = base_hpr + self._beat_hpr
        scale = self._beat_scale * energy_scale
        pos = self._beat_pos.copy()

        return hpr, scale, pos
