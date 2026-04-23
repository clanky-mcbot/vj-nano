"""Grab one webcam frame + extract palette + save a side-by-side preview PNG.

Use for quick verification: `python -m vj.vision.snapshot out.png`
"""

import argparse
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="Output PNG path")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10,
                    help="Discard N frames before sampling (UVC auto-exposure)")
    args = ap.parse_args()

    import cv2
    from vj.vision.webcam import Webcam
    from vj.vision.palette import palette_from_frame

    with Webcam(device=args.device, width=args.width, height=args.height,
                fps=args.fps) as cam:
        for _ in range(args.warmup):
            cam.read()
        frame, _ = cam.read()
    print("captured {}x{} via {}".format(frame.shape[1], frame.shape[0], cam.mode))

    pal = palette_from_frame(frame, k=args.k)
    hex_colors = ["#{:02x}{:02x}{:02x}".format(
        int(c[2]), int(c[1]), int(c[0])) for c in pal]
    print("palette:", "  ".join(hex_colors))

    # Build composite: original on top, palette strip on bottom.
    strip_h = 60
    strip = np.empty((strip_h, frame.shape[1], 3), dtype=np.uint8)
    seg = frame.shape[1] // args.k
    for i, c in enumerate(pal):
        x0 = i * seg
        x1 = (i + 1) * seg if i < args.k - 1 else frame.shape[1]
        strip[:, x0:x1] = c.astype(np.uint8)
    composite = np.vstack([frame, strip])
    cv2.imwrite(args.out, composite)
    print("saved {}  ({}x{})".format(args.out, composite.shape[1], composite.shape[0]))


if __name__ == "__main__":
    main()
