"""Webcam capture via gstreamer pipeline on Jetson Nano.

Uses opencv's gstreamer backend to decode MJPEG via the hardware-accelerated
path when available (`nvjpegdec`), falling back to software `jpegdec` if
nvjpeg isn't in the pipeline graph.

Why gstreamer and not plain cv2.VideoCapture(0)?
    On JetPack 4.6.4 the CAP_V4L2 backend does software MJPEG decode and
    spends 30%+ CPU on a 640x480/30fps stream. The gstreamer path is 3-5x
    cheaper because the C270's MJPEG frames go through the hardware JPEG
    decoder. We need every CPU cycle for Panda3D + audio DSP.
"""

# Python 3.6 compatible — no PEP 585/604 syntax.

import time
from typing import Iterator, Optional, Tuple

import numpy as np


def _gst_pipeline(device, width, height, fps, hw_decode):
    # type: (str, int, int, int, bool) -> str
    """Build a gstreamer pipeline string for cv2.VideoCapture.

    Output: BGR frames at the requested size/fps.
    """
    # C270 and most UVC webcams only do MJPEG at >= 640x480. We ask v4l2src
    # for image/jpeg and decode to BGR. nvjpegdec is the hardware decoder on
    # Jetson; jpegdec is the software fallback.
    decoder = "nvjpegdec" if hw_decode else "jpegdec"
    # After decode we need to ensure BGR output for OpenCV. nvjpegdec emits
    # I420 by default; convert via videoconvert (CPU, but small).
    return (
        "v4l2src device={dev} io-mode=2 ! "
        "image/jpeg,width={w},height={h},framerate={fps}/1 ! "
        "{dec} ! videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=2 sync=false"
    ).format(dev=device, w=width, h=height, fps=fps, dec=decoder)


class Webcam:
    """Iterable webcam source yielding (frame_bgr, capture_time) pairs.

    Example:
        with Webcam(width=640, height=480, fps=30) as cam:
            for frame, t in cam:
                # frame: HxWx3 uint8 BGR
                ...

    The iterator is infinite; break out of the loop to stop.
    """

    def __init__(
        self,
        device="/dev/video0",
        width=640,
        height=480,
        fps=30,
        hw_decode=True,
        fallback_to_plain_v4l2=True,
    ):
        # type: (str, int, int, int, bool, bool) -> None
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.hw_decode = hw_decode
        self.fallback_to_plain_v4l2 = fallback_to_plain_v4l2
        self._cap = None  # type: Optional["cv2.VideoCapture"]
        self._mode = None  # type: Optional[str]

    # -- context manager ------------------------------------------------
    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    # -- lifecycle ------------------------------------------------------
    def open(self):
        # type: () -> None
        import cv2

        # Attempt 1: hardware-accelerated gstreamer.
        if self.hw_decode:
            pipeline = _gst_pipeline(
                self.device, self.width, self.height, self.fps, hw_decode=True
            )
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened() and cap.read()[0]:
                self._cap = cap
                self._mode = "gst-nvjpeg"
                return
            cap.release()

        # Attempt 2: software gstreamer (useful on dev laptops with gstreamer).
        pipeline = _gst_pipeline(
            self.device, self.width, self.height, self.fps, hw_decode=False
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened() and cap.read()[0]:
            self._cap = cap
            self._mode = "gst-jpegdec"
            return
        cap.release()

        # Attempt 3: plain V4L2 (dev laptops without gstreamer, or bad Nano).
        if self.fallback_to_plain_v4l2:
            # On some Linux boxes cv2.CAP_V4L2 wants an integer index.
            idx = int(self.device.rsplit("video", 1)[-1]) if "video" in self.device else 0
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
                if cap.read()[0]:
                    self._cap = cap
                    self._mode = "v4l2-sw"
                    return
                cap.release()

        raise RuntimeError(
            "Could not open {} at {}x{}@{}fps via any backend.".format(
                self.device, self.width, self.height, self.fps
            )
        )

    def close(self):
        # type: () -> None
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- iteration ------------------------------------------------------
    @property
    def mode(self):
        # type: () -> Optional[str]
        """Which backend ended up opening the device ('gst-nvjpeg', etc.)."""
        return self._mode

    def read(self):
        # type: () -> Tuple[np.ndarray, float]
        """Block until the next frame is available, return (frame_bgr, t)."""
        if self._cap is None:
            raise RuntimeError("Webcam not opened; use `with Webcam(...)` or .open()")
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError("Webcam read failed — device disconnected?")
        return frame, time.monotonic()

    def __iter__(self):
        # type: () -> Iterator[Tuple[np.ndarray, float]]
        while True:
            yield self.read()


# ---------------------------------------------------------------------------
# CLI: benchmark + live format info
# ---------------------------------------------------------------------------

def _cli():
    # type: () -> None
    import argparse

    ap = argparse.ArgumentParser(description="Webcam capture benchmark.")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--no-hw", action="store_true", help="Force software decode")
    args = ap.parse_args()

    with Webcam(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        hw_decode=not args.no_hw,
    ) as cam:
        print("opened: mode={}  size={}x{}  fps={}".format(
            cam.mode, args.width, args.height, args.fps))
        t0 = time.monotonic()
        n = 0
        for frame, t in cam:
            n += 1
            if t - t0 >= args.seconds:
                break
        elapsed = time.monotonic() - t0
        print("captured {} frames in {:.2f}s  ->  {:.1f} fps  "
              "(frame shape: {})".format(n, elapsed, n / elapsed, frame.shape))


if __name__ == "__main__":
    _cli()
