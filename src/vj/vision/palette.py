"""Palette extraction from webcam frames.

Given a BGR frame, return K dominant colors. The colors drive a PS1-style
palette-remap shader so the 3D character visually re-tints to match the
room lighting picked up by the webcam.

Design choices for Nano-friendliness:
    - Downscale to 64x48 before clustering -> ~3000 pixels, not ~300k.
    - Custom tiny k-means (no sklearn): 5 iterations usually enough,
      doesn't drag in a 50MB dep and start warmup costs are near zero.
    - Temporal smoothing via EMA on the returned palette (via
      PaletteTracker) so the character doesn't flicker frame-to-frame
      when a pixel happens to land on a different cluster.

Typical Nano cost at 64x48: ~2-3 ms/frame. Plenty of headroom for 20 fps.
"""

# Python 3.6 compatible.

import time
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core k-means
# ---------------------------------------------------------------------------

def kmeans_palette(pixels, k=5, iters=5, seed=0):
    # type: (np.ndarray, int, int, int) -> np.ndarray
    """Lloyd's k-means on a (N, 3) uint8 or float32 pixel array.

    Returns (k, 3) float32 centroids in the same color space as input,
    sorted ascending by luminance (Rec. 709).

    Performance tricks used here matter on the Nano (Cortex-A57 / slow
    NumPy broadcast):
      - Squared-distance via ||a||^2 + ||b||^2 - 2 a.b,  computed as a
        single matrix multiply. ~10x faster than the 3-d broadcast form
        `((px[:,None,:] - c[None,:,:])**2).sum(2)` at our sizes.
      - np.einsum for the per-pixel norm (avoids a temporary).
      - argmin in-place, labels as int8 (k<=127).
    """
    px = pixels.reshape(-1, 3).astype(np.float32, copy=False)
    n = px.shape[0]
    if n == 0:
        return np.zeros((k, 3), dtype=np.float32)

    rng = np.random.default_rng(seed)

    # --- k-means++ seeding ---
    centroids = np.empty((k, 3), dtype=np.float32)
    idx0 = int(rng.integers(n))
    centroids[0] = px[idx0]
    px_norm = np.einsum("ij,ij->i", px, px)  # (n,) cached for reuse below

    # Distance-to-nearest-seed is maintained incrementally.
    # Start with distance to the single first seed.
    diff = px - centroids[0]
    d2_nearest = np.einsum("ij,ij->i", diff, diff)

    for i in range(1, k):
        total = float(d2_nearest.sum())
        if total <= 0:
            centroids[i] = px[int(rng.integers(n))]
        else:
            probs = d2_nearest / total
            centroids[i] = px[int(rng.choice(n, p=probs))]
        # Update d2_nearest to include the newly added centroid.
        diff = px - centroids[i]
        d2_new = np.einsum("ij,ij->i", diff, diff)
        np.minimum(d2_nearest, d2_new, out=d2_nearest)

    # --- Lloyd iterations using ||a - b||^2 = a.a + b.b - 2 a.b ---
    for _ in range(iters):
        c_norm = np.einsum("ij,ij->i", centroids, centroids)      # (k,)
        # (n, k) = |px|^2  +  |c|^2  -  2 * px @ c.T
        d2 = px_norm[:, None] + c_norm[None, :] - 2.0 * (px @ centroids.T)
        labels = d2.argmin(axis=1).astype(np.int8)
        for j in range(k):
            m = labels == j
            if m.any():
                centroids[j] = px[m].mean(axis=0)

    # Sort by Rec.709 luminance (assumes input is BGR; convert inline).
    # If input is already RGB, results are still valid — just a sort order.
    lum = 0.0722 * centroids[:, 0] + 0.7152 * centroids[:, 1] + 0.2126 * centroids[:, 2]
    order = np.argsort(lum)
    return centroids[order]


# ---------------------------------------------------------------------------
# Frame-level convenience: downscale -> palette
# ---------------------------------------------------------------------------

def palette_from_frame(frame_bgr, k=5, downscale=(64, 48), iters=5, seed=0):
    # type: (np.ndarray, int, Tuple[int, int], int, int) -> np.ndarray
    """Extract a k-color palette from a BGR frame.

    `downscale` is (width, height). Returns (k, 3) float32, BGR, sorted
    by luminance.
    """
    import cv2
    small = cv2.resize(frame_bgr, downscale, interpolation=cv2.INTER_AREA)
    return kmeans_palette(small, k=k, iters=iters, seed=seed)


# ---------------------------------------------------------------------------
# Temporal smoothing via Hungarian-free greedy matching
# ---------------------------------------------------------------------------

class PaletteTracker:
    """EMA-smoothed palette over time.

    The tricky bit: k-means returns centroids in arbitrary order, so we
    need to match them to the previous frame's centroids before
    smoothing. We use a greedy nearest-match (good enough for k=5).

    Because the palette changes slowly relative to frame rate (lighting
    in a room doesn't really swing at 30Hz), `update_every` lets callers
    run k-means only every Nth frame and return the EMA-cached value on
    the others. This keeps the per-frame cost near zero while still
    converging quickly (~0.5s at update_every=6, alpha=0.3, 30fps).

    Usage:
        tracker = PaletteTracker(k=5, alpha=0.2, update_every=6)
        for frame, _ in cam:
            p = tracker.update(frame)       # (5, 3) smoothed BGR palette
    """

    def __init__(self, k=5, alpha=0.2, downscale=(64, 48), update_every=1):
        # type: (int, float, Tuple[int, int], int) -> None
        self.k = k
        self.alpha = alpha
        self.downscale = downscale
        self.update_every = max(1, int(update_every))
        self._state = None  # type: Optional[np.ndarray]
        self._counter = 0

    def update(self, frame_bgr):
        # type: (np.ndarray) -> np.ndarray
        # Skip the heavy recompute unless it's time.
        if self._state is not None and (self._counter % self.update_every) != 0:
            self._counter += 1
            return self._state
        self._counter += 1

        new_pal = palette_from_frame(
            frame_bgr, k=self.k, downscale=self.downscale
        )
        if self._state is None:
            self._state = new_pal.copy()
            return self._state
        # Greedy match: for each previous centroid, pick the closest unused new.
        prev = self._state
        used = np.zeros(self.k, dtype=bool)
        matched = np.empty_like(prev)
        for i in range(self.k):
            d = ((new_pal - prev[i]) ** 2).sum(axis=1)
            d[used] = np.inf
            j = int(d.argmin())
            matched[i] = new_pal[j]
            used[j] = True
        self._state = (1.0 - self.alpha) * prev + self.alpha * matched
        return self._state

    def reset(self):
        # type: () -> None
        self._state = None
        self._counter = 0


# ---------------------------------------------------------------------------
# CLI: live palette preview (terminal-friendly, ANSI blocks)
# ---------------------------------------------------------------------------

def _ansi_block(bgr):
    # type: (np.ndarray) -> str
    r, g, b = int(bgr[2]), int(bgr[1]), int(bgr[0])
    return "\x1b[48;2;{};{};{}m  \x1b[0m".format(r, g, b)


def _cli():
    # type: () -> None
    import argparse

    ap = argparse.ArgumentParser(description="Live palette preview from webcam.")
    ap.add_argument("--device", default="/dev/video0")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--no-hw", action="store_true")
    args = ap.parse_args()

    from vj.vision.webcam import Webcam

    tracker = PaletteTracker(k=args.k, alpha=args.alpha)
    with Webcam(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        hw_decode=not args.no_hw,
    ) as cam:
        print("webcam: mode={} size={}x{}".format(cam.mode, args.width, args.height))
        t0 = time.monotonic()
        n_frames = 0
        total_pal_ms = 0.0
        last_print = 0.0
        for frame, t in cam:
            tA = time.monotonic()
            pal = tracker.update(frame)
            total_pal_ms += (time.monotonic() - tA) * 1000
            n_frames += 1
            # Print every ~200ms to not flood the terminal.
            if t - last_print > 0.2:
                blocks = "".join(_ansi_block(c) for c in pal)
                hex_ = "  ".join("#{:02x}{:02x}{:02x}".format(
                    int(c[2]), int(c[1]), int(c[0])) for c in pal)
                avg_ms = total_pal_ms / n_frames
                print("\r{}  {}  ({:.1f} ms/palette)".format(
                    blocks, hex_, avg_ms), end="", flush=True)
                last_print = t
            if t - t0 > args.seconds:
                break
    elapsed = time.monotonic() - t0
    print("\n--- {} frames in {:.2f}s ({:.1f} fps camera, "
          "{:.2f} ms avg palette) ---".format(
              n_frames, elapsed, n_frames / elapsed,
              total_pal_ms / max(1, n_frames)))


if __name__ == "__main__":
    _cli()
