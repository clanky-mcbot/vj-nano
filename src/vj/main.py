"""vj-nano main entry point.

Spawns three concurrent work-streams:
  1. Audio thread    — captures / reads audio, runs analysis, publishes features.
  2. Webcam thread   — captures frames, extracts palette, publishes tint.
  3. Render thread   — Panda3D main loop (this thread), reads shared state each
                       frame to drive the scene.

Python 3.6 compatible (Jetson Nano JetPack 4.6.4).
"""

from __future__ import print_function

import argparse
import sys
import threading
import time
from typing import Optional

import numpy as np

from vj.audio.analyzer import AudioAnalyzer
from vj.audio.sources import FileSource, LineInSource
from vj.render.app import VJApp
from vj.vision.palette import PaletteTracker
from vj.vision.webcam import Webcam


# ---------------------------------------------------------------------------
# Thread-safe state bucket
# ---------------------------------------------------------------------------

class SharedState(object):
    """Simple locked container for cross-thread data.

    AudioFeatures and palette arrays are replaced wholesale each update,
    so readers always see a consistent snapshot.
    """

    def __init__(self):
        # type: () -> None
        self._lock = threading.Lock()
        self.features = None   # type: Optional[AudioFeatures]
        self.palette = None    # type: Optional[np.ndarray]
        self.tint = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.running = True

    def set_features(self, feat):
        # type: (AudioFeatures) -> None
        with self._lock:
            self.features = feat

    def get_features(self):
        # type: () -> Optional[AudioFeatures]
        with self._lock:
            return self.features

    def set_palette(self, pal):
        # type: (np.ndarray) -> None
        with self._lock:
            self.palette = pal

    def get_palette(self):
        # type: () -> Optional[np.ndarray]
        with self._lock:
            return self.palette


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

def _audio_worker(state, source, analyzer):
    # type: (SharedState, object, AudioAnalyzer) -> None
    """Consume audio chunks, analyze, publish features."""
    try:
        for chunk in source:
            if not state.running:
                break
            feat = analyzer.process(chunk)
            state.set_features(feat)
    except Exception as exc:
        print("[audio] fatal error:", exc, file=sys.stderr)
        state.running = False
        raise


def _webcam_worker(state, cam, tracker, update_every):
    # type: (SharedState, Webcam, PaletteTracker, int) -> None
    """Consume webcam frames, extract palette, publish tint."""
    try:
        frame_i = 0
        while state.running:
            frame, ok = cam.read()
            if not ok:
                time.sleep(0.005)
                continue
            frame_i += 1
            if frame_i % update_every == 0:
                pal = tracker.update(frame)
                state.set_palette(pal)
    except Exception as exc:
        print("[webcam] fatal error:", exc, file=sys.stderr)
        state.running = False
        raise
    finally:
        cam.close()


# ---------------------------------------------------------------------------
# Panda3D task — runs on the main (render) thread
# ---------------------------------------------------------------------------

def _drive_task(task, app, state):
    # type: (object, VJApp, SharedState) -> int
    """Copy latest audio + vision state into the renderer.

    Called once per frame by Panda3D's task manager.
    """
    feat = state.get_features()
    if feat is not None:
        app.set_audio_energy(feat.rms)
        app.set_beat_phase(feat.beat_phase)

    pal = state.get_palette()
    if pal is not None:
        # Palette returns BGR uint8, 5 colors sorted by luminance.
        # Use the mid-brightness colour (index 2) for a balanced tint.
        mid = pal[2]
        app.set_tint(
            np.array(
                [mid[2] / 255.0, mid[1] / 255.0, mid[0] / 255.0],
                dtype=np.float32,
            )
        )

    from direct.task import Task
    return Task.cont


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    # type: () -> None
    ap = argparse.ArgumentParser(
        description="vj-nano: PS1-style audio-reactive VJ rig for Jetson Nano"
    )
    ap.add_argument(
        "--audio",
        default="line",
        help=(
            "Audio source. 'line' = default sounddevice input (gig mode). "
            "Any other string is treated as a file path (dev mode)."
        ),
    )
    ap.add_argument(
        "--no-webcam",
        action="store_true",
        help="Disable webcam / palette tinting.",
    )
    ap.add_argument(
        "--win-size",
        default="1280x720",
        help="Window size, e.g. 1280x720 or 640x480.",
    )
    ap.add_argument(
        "--fps",
        action="store_true",
        help="Show Panda3D frame-rate meter.",
    )
    ap.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Auto-exit after N seconds (0 = run until window closed).",
    )
    args = ap.parse_args()

    # Parse window size
    try:
        w_str, h_str = args.win_size.split("x")
        win_size = (int(w_str), int(h_str))
    except ValueError:
        print(
            "error: --win-size must be WxH, e.g. 1280x720",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- audio -----------------------------------------------------------
    sr = 44100
    hop = 1024
    if args.audio.lower() == "line":
        print("[main] audio source: line-in")
        source = LineInSource(sr=sr, hop=hop)
    else:
        print("[main] audio source: file '{}'".format(args.audio))
        source = FileSource(
            path=args.audio,
            sr=sr,
            hop=hop,
            realtime=True,
            loop=True,
        )
    analyzer = AudioAnalyzer(sr=sr, hop=hop)

    # --- webcam ----------------------------------------------------------
    cam = None      # type: Optional[Webcam]
    tracker = None  # type: Optional[PaletteTracker]
    if not args.no_webcam:
        print("[main] webcam: opening /dev/video0 ...")
        cam = Webcam()
        cam.open()
        tracker = PaletteTracker(k=5, alpha=0.3, update_every=6)
        print("[main] webcam: ok")

    # --- renderer --------------------------------------------------------
    # fps meter must be set *before* ShowBase is instantiated.
    if args.fps:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "show-frame-rate-meter 1")

    app = VJApp(window_title="vj-nano", win_size=win_size)

    # --- shared state & threads ------------------------------------------
    state = SharedState()

    t_audio = threading.Thread(
        target=_audio_worker,
        args=(state, source, analyzer),
        daemon=True,
    )
    t_audio.start()

    t_webcam = None  # type: Optional[threading.Thread]
    if cam is not None:
        t_webcam = threading.Thread(
            target=_webcam_worker,
            args=(state, cam, tracker, 6),
            daemon=True,
        )
        t_webcam.start()

    # Hook the renderer into Panda3D's task manager.
    app.base.taskMgr.add(
        lambda task: _drive_task(task, app, state),
        "vj-drive",
    )

    # Optional timed exit (handy for headless benchmarking).
    if args.duration > 0.0:
        def _quit_task(task):
            app.base.userExit()
            return task.done
        app.base.taskMgr.doMethodLater(args.duration, _quit_task, "vj-quit")

    # --- run -------------------------------------------------------------
    print("[main] starting render loop (close window or Ctrl+C to stop)")
    try:
        app.run()
    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("\n[main] interrupted")
    finally:
        print("[main] shutting down ...")
        state.running = False
        if t_webcam is not None:
            t_webcam.join(timeout=1.0)
        t_audio.join(timeout=1.0)
        app.destroy()
        print("[main] done")


if __name__ == "__main__":
    main()
