"""Tests for vj.vision.palette — synthetic images, no webcam needed."""

import numpy as np

from vj.vision.palette import kmeans_palette, palette_from_frame


def _synthetic_3color(h=48, w=64):
    """Image with three clear BGR regions: red, green, blue."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 3] = (0, 0, 255)          # red (BGR)
    img[:, w // 3 : 2 * w // 3] = (0, 255, 0)  # green
    img[:, 2 * w // 3 :] = (255, 0, 0)      # blue
    return img


def test_kmeans_recovers_three_distinct_clusters():
    img = _synthetic_3color()
    pal = kmeans_palette(img, k=3, iters=10, seed=0)
    assert pal.shape == (3, 3)
    # Each centroid should be close to one of the three pure colors.
    pure = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.float32)
    for c in pal:
        d = ((pure - c) ** 2).sum(axis=1)
        assert d.min() < 25.0, "centroid {} not close to any pure color".format(c)


def test_palette_sorted_by_luminance():
    img = _synthetic_3color()
    pal = kmeans_palette(img, k=3, iters=10, seed=0)
    # Rec.709 luminance ordering: blue (low) < red (mid) < green (high).
    lum = 0.0722 * pal[:, 0] + 0.7152 * pal[:, 1] + 0.2126 * pal[:, 2]
    assert (np.diff(lum) >= 0).all()


def test_palette_from_frame_works_with_cv2_resize():
    # 1280x960 solid-black frame -> palette should be 5 clusters of ~black.
    img = np.full((960, 1280, 3), 12, dtype=np.uint8)
    pal = palette_from_frame(img, k=5)
    assert pal.shape == (5, 3)
    assert (pal < 25).all()


def test_empty_like_input_does_not_crash():
    # Zero-pixel input would be pathological; tiny one still should work.
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    pal = kmeans_palette(img, k=3, iters=3, seed=0)
    assert pal.shape == (3, 3)
    # All centroids should land on ~200 since input is uniform.
    assert (np.abs(pal - 200) < 5).all()
