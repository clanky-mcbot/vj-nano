"""BPM-locked animation system for PS1Humanoid.

Supports looping clips played at variable rates driven by detected BPM.
All clips are defined as Python keyframe data — no external files needed.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


class Keyframe(object):
    """Single keyframe for one joint."""

    def __init__(self, t, hpr, pos=None):
        # type: (float, tuple, Optional[tuple]) -> None
        self.t = float(t)
        self.hpr = tuple(float(x) for x in hpr)
        self.pos = tuple(float(x) for x in pos) if pos is not None else None


class AnimationClip(object):
    """Looping animation clip: keyframes per joint."""

    def __init__(self, name, duration, keyframes):
        # type: (str, float, Dict[str, List[Keyframe]]) -> None
        self.name = name
        self.duration = float(duration)
        self.keyframes = keyframes  # {joint: [Keyframe, ...]}

    def sample(self, joint, t):
        # type: (str, float) -> Tuple[Tuple[float, float, float], Optional[Tuple[float, float, float]]]
        """Sample HPR and optional POS for a joint at local time t (seconds, looped)."""
        frames = self.keyframes.get(joint)
        if not frames:
            return (0.0, 0.0, 0.0), None

        t = t % self.duration
        if t <= frames[0].t:
            return frames[0].hpr, frames[0].pos
        if t >= frames[-1].t:
            return frames[-1].hpr, frames[-1].pos

        for i in range(len(frames) - 1):
            a, b = frames[i], frames[i + 1]
            if a.t <= t <= b.t:
                if abs(b.t - a.t) < 1e-9:
                    return a.hpr, a.pos
                u = (t - a.t) / (b.t - a.t)
                h = a.hpr[0] + u * (b.hpr[0] - a.hpr[0])
                p = a.hpr[1] + u * (b.hpr[1] - a.hpr[1])
                r = a.hpr[2] + u * (b.hpr[2] - a.hpr[2])
                hpr = (h, p, r)
                if a.pos is not None and b.pos is not None:
                    px = a.pos[0] + u * (b.pos[0] - a.pos[0])
                    py = a.pos[1] + u * (b.pos[1] - a.pos[1])
                    pz = a.pos[2] + u * (b.pos[2] - a.pos[2])
                    return hpr, (px, py, pz)
                return hpr, a.pos

        return frames[-1].hpr, frames[-1].pos


class BpmClock(object):
    """Converts real-time to beat-time based on detected BPM."""

    def __init__(self, default_bpm=120.0):
        # type: (float) -> None
        self._bpm = float(default_bpm)
        self._beat_duration = 60.0 / default_bpm
        self._elapsed_beats = 0.0

    def set_bpm(self, bpm):
        # type: (float) -> None
        if bpm > 0 and abs(bpm - self._bpm) > 0.5:
            self._elapsed_beats *= self._beat_duration / (60.0 / bpm)
            self._bpm = float(bpm)
            self._beat_duration = 60.0 / bpm

    def update(self, dt):
        # type: (float) -> float
        """Advance by dt seconds; returns current total beat count."""
        self._elapsed_beats += dt / self._beat_duration
        return self._elapsed_beats

    @property
    def elapsed_beats(self):
        # type: () -> float
        return self._elapsed_beats

    @property
    def beat_phase(self):
        # type: () -> float
        return self._elapsed_beats % 1.0

    @property
    def bar_phase(self):
        # type: () -> float
        return (self._elapsed_beats % 4.0) / 4.0


class AnimationMixer(object):
    """Manages multiple animation clips, blends them, and drives them via BpmClock."""

    def __init__(self, clock):
        # type: (BpmClock) -> None
        self.clock = clock
        self.clips = {}  # type: Dict[str, AnimationClip]
        self.weights = {}  # type: Dict[str, float]
        self._one_shots = []  # type: List[Tuple[str, float, float]]

    def add_clip(self, clip):
        # type: (AnimationClip) -> None
        self.clips[clip.name] = clip
        self.weights[clip.name] = 0.0

    def set_weight(self, name, w):
        # type: (str, float) -> None
        if name in self.weights:
            self.weights[name] = float(w)

    def trigger_one_shot(self, name, duration_beats=1.0):
        # type: (str, float) -> None
        """Fire a one-shot clip that auto-fades over duration_beats."""
        self._one_shots.append((name, self.clock.elapsed_beats, float(duration_beats)))

    def update(self, dt):
        # type: (float) -> Dict[str, dict]
        """Return blended pose dict {joint: {"hpr": (h,p,r), "pos": (x,y,z)}}."""
        self.clock.update(dt)

        # Fade out one-shots
        active = []
        for name, start_beat, dur in self._one_shots:
            elapsed = self.clock.elapsed_beats - start_beat
            if elapsed < dur:
                active.append((name, elapsed, dur))
        self._one_shots = active

        # Accumulate per-joint
        pose = {}  # type: Dict[str, List[float]]
        # Each entry: [h, p, r, px, py, pz, w_hpr, w_pos]

        for name, clip in self.clips.items():
            w = self.weights.get(name, 0.0)
            if w <= 0.0:
                continue
            t = self.clock.elapsed_beats * self.clock._beat_duration
            t = t % clip.duration
            for joint in clip.keyframes:
                hpr, pos = clip.sample(joint, t)
                if joint not in pose:
                    pose[joint] = [0.0] * 8
                pacc = pose[joint]
                for i in range(3):
                    pacc[i] += w * hpr[i]
                pacc[6] += w
                if pos is not None:
                    for i in range(3):
                        pacc[3 + i] += w * pos[i]
                    pacc[7] += w

        # Add one-shots
        for name, elapsed, dur in active:
            clip = self.clips.get(name)
            if clip is None:
                continue
            w = 1.0 - (elapsed / dur)
            if w <= 0.0:
                continue
            t = elapsed * self.clock._beat_duration
            t = t % clip.duration
            for joint in clip.keyframes:
                hpr, pos = clip.sample(joint, t)
                if joint not in pose:
                    pose[joint] = [0.0] * 8
                pacc = pose[joint]
                for i in range(3):
                    pacc[i] += w * hpr[i]
                pacc[6] += w
                if pos is not None:
                    for i in range(3):
                        pacc[3 + i] += w * pos[i]
                    pacc[7] += w

        # Normalize
        result = {}  # type: Dict[str, dict]
        for joint, acc in pose.items():
            entry = {}
            if acc[6] > 0.0:
                entry["hpr"] = (acc[0] / acc[6], acc[1] / acc[6], acc[2] / acc[6])
            else:
                entry["hpr"] = (0.0, 0.0, 0.0)
            if acc[7] > 0.0:
                entry["pos"] = (acc[3] / acc[7], acc[4] / acc[7], acc[5] / acc[7])
            else:
                entry["pos"] = None
            result[joint] = entry

        return result


# ---------------------------------------------------------------------------
# Built-in dance clips
# ---------------------------------------------------------------------------

def make_idle_clip():
    # type: () -> AnimationClip
    return AnimationClip("idle", 2.0, {
        "spine": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(1.0, (0, 2, 0)),
            Keyframe(2.0, (0, 0, 0)),
        ],
        "chest": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(1.0, (0, -1, 0)),
            Keyframe(2.0, (0, 0, 0)),
        ],
        "head": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(1.0, (0, 1, 0)),
            Keyframe(2.0, (0, 0, 0)),
        ],
        "left_shoulder": [
            Keyframe(0.0, (0, 0, 5)),
            Keyframe(1.0, (0, 0, -5)),
            Keyframe(2.0, (0, 0, 5)),
        ],
        "right_shoulder": [
            Keyframe(0.0, (0, 0, -5)),
            Keyframe(1.0, (0, 0, 5)),
            Keyframe(2.0, (0, 0, -5)),
        ],
    })


def make_groove_clip():
    # type: () -> AnimationClip
    return AnimationClip("groove", 1.0, {
        "hips": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(0.25, (0, 0, 3)),
            Keyframe(0.5, (0, 0, 0)),
            Keyframe(0.75, (0, 0, -3)),
            Keyframe(1.0, (0, 0, 0)),
        ],
        "spine": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(0.5, (0, 5, 0)),
            Keyframe(1.0, (0, 0, 0)),
        ],
        "chest": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(0.5, (0, -3, 0)),
            Keyframe(1.0, (0, 0, 0)),
        ],
        "head": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(0.25, (5, 0, 0)),
            Keyframe(0.75, (-5, 0, 0)),
            Keyframe(1.0, (0, 0, 0)),
        ],
        "left_shoulder": [
            Keyframe(0.0, (0, 0, 10)),
            Keyframe(0.5, (0, 0, 30)),
            Keyframe(1.0, (0, 0, 10)),
        ],
        "right_shoulder": [
            Keyframe(0.0, (0, 0, -10)),
            Keyframe(0.5, (0, 0, -30)),
            Keyframe(1.0, (0, 0, -10)),
        ],
        "left_elbow": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(0.5, (0, 0, 20)),
            Keyframe(1.0, (0, 0, 0)),
        ],
        "right_elbow": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(0.5, (0, 0, -20)),
            Keyframe(1.0, (0, 0, 0)),
        ],
    })


def make_jump_clip():
    # type: () -> AnimationClip
    return AnimationClip("jump", 0.5, {
        "hips": [
            Keyframe(0.0, (0, 0, 0), pos=(0, 0, 0)),
            Keyframe(0.125, (0, 0, 0), pos=(0, 0, 0.15)),
            Keyframe(0.25, (0, 0, 0), pos=(0, 0, 0.30)),
            Keyframe(0.375, (0, 0, 0), pos=(0, 0, 0.15)),
            Keyframe(0.5, (0, 0, 0), pos=(0, 0, 0)),
        ],
        "left_shoulder": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(0.25, (0, 0, 45)),
            Keyframe(0.5, (0, 0, 0)),
        ],
        "right_shoulder": [
            Keyframe(0.0, (0, 0, 0)),
            Keyframe(0.25, (0, 0, -45)),
            Keyframe(0.5, (0, 0, 0)),
        ],
    })
