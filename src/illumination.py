"""Phase 4: estimate how the page is lit.

Key trick: text is DARK and THIN. A grayscale *closing* (dilate then erode) with a
kernel wider than the strokes erases the text and leaves only the paper
brightness, i.e. the illumination / background map.
"""
import cv2
import numpy as np

from .preprocessing import odd


def estimate_background(gray, kernel=None):
    """Smooth map of paper brightness (same size as `gray`, float32, 0..255)."""
    # light smoothing first: the closing below takes local maxima, which noise would inflate
    g = cv2.GaussianBlur(np.clip(gray, 0, 255).astype(np.float32), (0, 0), 2.0)
    g = np.clip(g, 0, 255).astype(np.uint8)
    h, w = g.shape
    if kernel is None:
        kernel = odd(max(21, 0.025 * max(h, w)))
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, se)
    bg = cv2.medianBlur(bg, odd(kernel))
    bg = cv2.GaussianBlur(bg.astype(np.float32), (0, 0), kernel / 2.0)
    return bg


def illumination_ratio(bg):
    """Local brightness relative to the well-lit part of the page.

    ref = 95th percentile of the background (the "good light" level).
    ratio ~ 1.0 in normal areas, < 1.0 in shadows.
    """
    ref = float(np.percentile(bg, 95))
    return bg / max(ref, 1.0), ref


def normalize_by_background(gray, bg, ref):
    """Divide out the illumination: paper -> `ref`, text keeps its relative contrast."""
    out = gray.astype(np.float32) / np.maximum(bg, 10.0) * ref
    return np.clip(out, 0, 255)
