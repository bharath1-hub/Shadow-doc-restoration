"""Phase 4: shadow detection and shadow removal (background correction)."""
import cv2
import numpy as np

from .illumination import normalize_by_background


def detect_shadow_mask(ratio, thr=0.85):
    """Pixel mask (uint8 0/1) of areas whose background is darker than thr * reference."""
    mask = (ratio < thr).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def correct_shadow(gray, bg, ref):
    """Remove shading by dividing by the estimated background (see illumination.py)."""
    return normalize_by_background(gray, bg, ref)
