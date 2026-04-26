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
from vj.audio.sources import FileSource, LineInSource, NullSource, PulseAudioSource
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
        self.webcam_frame = None  # type: Optional[np.ndarray]  # 160x120 RGB uint8
        self.tint = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.waveform = np.zeros(512, dtype=np.float32)
        self.running = True

    def set_waveform(self, wf):
        # type: (np.ndarray) -> None
        with self._lock:
            n = min(512, wf.shape[0])
            self.waveform[:n] = wf[:n].astype(np.float32)

    def get_waveform(self):
        # type: () -> np.ndarray
        with self._lock:
            return self.waveform.copy()

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

    def set_webcam_frame_small(self, frame):
        # type: (np.ndarray) -> None
        with self._lock:
            self.webcam_frame = frame

    def get_webcam_frame_small(self):
        # type: () -> Optional[np.ndarray]
        with self._lock:
            return self.webcam_frame


# ---------------------------------------------------------------------------
# Worker threads
# ---------------------------------------------------------------------------

def _audio_worker(state, source, analyzer):
    # type: (SharedState, object, AudioAnalyzer) -> None
    """Consume audio chunks, analyze, publish features + waveform."""
    try:
        for chunk in source:
            if not state.running:
                break
            feat = analyzer.process(chunk)
            state.set_features(feat)
            state.set_waveform(chunk)
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
            # Cheap: always send a preview frame (160x120 RGB resize)
            import cv2
            small = cv2.resize(frame, (160, 120))
            small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            state.set_webcam_frame_small(small_rgb)
            # Expensive: palette extraction every N frames (k-means on full res)
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
        app.set_features(feat)

    wf = state.get_waveform()
    if wf is not None:
        app.set_waveform(wf)

    wf_small = state.get_webcam_frame_small()
    if wf_small is not None:
        app.set_webcam_frame(wf_small)

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
            "Audio source. 'line' = sounddevice (broken), 'pulse' = PulseAudio, 'null' = silent. "
            "'net' = TCP network stream (run audio_sender.py on your PC). "
            "Any other string is treated as a file path (dev mode)."
        ),
    )
    ap.add_argument(
        "--audio-device",
        type=int,
        default=None,
        help="sounddevice input index (default = system default). "
             "Use --list-devices to see indices.",
    )
    ap.add_argument(
        "--audio-channels",
        type=int,
        default=1,
        help="Input channels (1=mono, 2=stereo). For stereo sources like CI1, use 2 (Left channel extracted).",
    )
    ap.add_argument(
        "--net-port",
        type=int,
        default=5000,
        help="TCP port for network audio source (default 5000).",
    )
    ap.add_argument(
        "--list-devices",
        action="store_true",
        help="Print audio devices and exit.",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Show debug overlay: BPM, waveform, sensitivity stats.",
    )
    ap.add_argument(
        "--sensitivity",
        type=float,
        default=2.5,
        help="Onset detection sensitivity multiplier (higher = less sensitive). "
             "Default 2.5 for room mics; try 1.5 for clean line-in.",
    )
    ap.add_argument(
        "--debounce",
        type=float,
        default=0.18,
        help="Minimum seconds between onsets. Default 0.18.",
    )
    ap.add_argument(
        "--no-webcam",
        action="store_true",
        help="Disable webcam / palette tinting.",
    )
    ap.add_argument(
        "--webcam-device",
        default="auto",
        help="Webcam device path or 'auto' to detect (default: auto).",
    )
    ap.add_argument(
        "--flip-webcam",
        action="store_true",
        help="Flip webcam frames vertically (use when camera is mounted upside-down).",
    )
    ap.add_argument(
        "--model",
        default="assets/models/RobotExpressive.glb",
        help="Model path or 'procedural' (default: RobotExpressive.glb).",
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

    if args.list_devices:
        import sounddevice as sd
        print("Audio devices:")
        for i, d in enumerate(sd.query_devices()):
            print("  {}: {}  (in={}, out={})".format(
                i, d["name"], d["max_input_channels"], d["max_output_channels"]))
        sys.exit(0)

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
    hop = 512
    source = None  # type: Optional[object]
    analyzer = None  # type: Optional[AudioAnalyzer]
    try:
        if args.audio.lower() == "line":
            print("[main] audio source: line-in (device={})".format(args.audio_device))
            source = LineInSource(sr=sr, hop=hop, device=args.audio_device, latency=0.02, channels=args.audio_channels)
        elif args.audio.lower() == "pulse":
            print("[main] audio source: PulseAudio (parec)")
            source = PulseAudioSource(sr=sr, hop=hop, channels=args.audio_channels)
        elif args.audio.lower() == "null":
            print("[main] audio source: null (silent)")
            source = NullSource(sr=sr, hop=hop)
        elif args.audio.lower() == "net":
            from vj.audio.net_source import NetworkAudioSource
            print("[main] audio source: network stream on port {}".format(args.net_port))
            source = NetworkAudioSource(port=args.net_port, sr=sr, hop=hop)
        else:
            print("[main] audio source: file '{}'".format(args.audio))
            source = FileSource(
                path=args.audio,
                sr=sr,
                hop=hop,
                realtime=True,
                loop=True,
            )
        analyzer = AudioAnalyzer(
            sr=sr,
            hop=hop,
            onset_thresh_mul=args.sensitivity,
            onset_min_interval_sec=args.debounce,
        )
    except Exception as exc:
        print("[main] audio: unavailable ({}) — continuing without audio reactivity".format(exc))
        source = None
        analyzer = None

    # --- webcam ----------------------------------------------------------
    cam = None      # type: Optional[Webcam]
    tracker = None  # type: Optional[PaletteTracker]
    if not args.no_webcam:
        try:
            print("[main] webcam: opening {} ...".format(args.webcam_device))
            cam = Webcam(
                device=args.webcam_device if args.webcam_device != "auto" else "/dev/video0",
                flip=args.flip_webcam,
            )
            cam.open()
            tracker = PaletteTracker(k=5, alpha=0.3, update_every=6)
            print("[main] webcam: ok  (mode={})".format(cam.mode))
        except Exception as exc:
            print("[main] webcam: unavailable ({}) — continuing without tint".format(exc))
            cam = None
            tracker = None

    # --- renderer --------------------------------------------------------
    # fps meter must be set *before* ShowBase is instantiated.
    if args.fps:
        from panda3d.core import loadPrcFileData
        loadPrcFileData("", "show-frame-rate-meter 1")

    app = VJApp(window_title="vj-nano", win_size=win_size, debug=args.debug, model=args.model, flip_webcam=args.flip_webcam)

    # --- shared state & threads ------------------------------------------
    state = SharedState()

    t_audio = None  # type: Optional[threading.Thread]
    if source is not None and analyzer is not None:
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
        if t_audio is not None:
            t_audio.join(timeout=1.0)
        app.destroy()
        print("[main] done")


if __name__ == "__main__":
    main()
