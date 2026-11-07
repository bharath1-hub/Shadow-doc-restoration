"""Phase 9: evaluation metrics.

Image metrics need a clean reference image (full-reference):
  MSE, MAE, PSNR, SSIM.
Text metrics need the true text: CER, WER.
Without a reference use no_reference_quality() (only relative, never absolute truth).
"""
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def mae(a, b):
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def psnr(a, b, data_range=255.0):
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / m))


def ssim(a, b):
    """Structural similarity (1.0 = identical). Works for gray or colour images."""
    if a.ndim == 3:
        return float(structural_similarity(a, b, data_range=255, channel_axis=2))
    return float(structural_similarity(a, b, data_range=255))


def levenshtein(a, b):
    """Edit distance (insert / delete / substitute) between two sequences."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def normalize_text(s):
    return " ".join(s.split())


def cer(reference, hypothesis):
    """Character error rate = edit distance / number of reference characters."""
    r, h = normalize_text(reference), normalize_text(hypothesis)
    return levenshtein(r, h) / max(1, len(r))


def wer(reference, hypothesis):
    """Word error rate = word-level edit distance / number of reference words."""
    r, h = reference.split(), hypothesis.split()
    return levenshtein(r, h) / max(1, len(r))


class Timer:
    """with Timer() as t: ...; t.seconds"""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._t0


def no_reference_quality(img):
    """Simple statistics usable without ground truth (compare before vs after only)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = gray.astype(np.float32)
    from .illumination import estimate_background, illumination_ratio
    bg = estimate_background(g)
    ratio, _ = illumination_ratio(bg)
    return {
        "sharpness_laplacian_var": float(cv2.Laplacian(g, cv2.CV_32F).var()),
        "contrast_std": float(g.std()),
        "illumination_nonuniformity": float(ratio.std()),   # lower = more even lighting
    }
