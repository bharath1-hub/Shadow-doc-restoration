"""Basic helpers: grayscale, resizing, odd kernel sizes."""
import cv2
import numpy as np


def to_gray(img):
    """BGR -> grayscale (a 2-D image is returned unchanged)."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_max_side(img, max_side=1600):
    """Shrink so the longer side equals max_side. Never enlarges."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA)


def odd(n):
    """Return n rounded up to the nearest odd integer (OpenCV kernels must be odd)."""
    n = int(n)
    return n if n % 2 == 1 else n + 1
