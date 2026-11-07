"""Phase 5: local blur estimation (per patch).

Sharp text has steep edges; blur makes them gentle. For each text patch we
measure the steepest gradients relative to a contrast reference C:

    s  = p99(|gradient|)   / (4 * C)      overall edge sharpness
    sx = p99(|d/dx|)       / (4 * C)      sharpness across vertical edges
    sy = p99(|d/dy|)       / (4 * C)      sharpness across horizontal edges

(For an ideal step edge the 3x3 Sobel response is 4 * contrast, so a perfect
edge scores about 1 and blurred edges score less.)

C = max(patch contrast, 0.7 * page brightness). The floor matters: thin
strokes lose their OWN contrast when blurred, which would otherwise hide the blur.

 * Blur score  = min(sx, sy). Defocus lowers both; motion lowers one direction.
 * Type        = log(sx / sy) compared with crisp text of the same page.
 * Defocus sigma and motion length are rough estimates calibrated on synthetic
   data (see docs/experiment_log.md). Motion is handled for horizontal/vertical
   directions only.
"""
import cv2
import numpy as np


def has_text(patch, cfg):
    """True if the patch has enough clearly dark ink on a lighter background."""
    white = np.percentile(patch, 99)
    black = np.percentile(patch, 1)
    contrast = white - black
    if contrast < cfg.text_min_contrast:
        return False
    dark_frac = float((patch < white - 0.4 * contrast).mean())
    return dark_frac >= cfg.text_min_frac


def edge_metrics(patch, ref, pre_sigma=0.8):
    """Return (s, sx, sy) for an illumination-normalised text patch.

    `ref` is the page brightness level (used for the contrast floor).
    """
    g = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), pre_sigma)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    contrast = float(np.percentile(patch, 99) - np.percentile(patch, 1))
    denom = 4.0 * max(contrast, 0.7 * ref) + 1e-6
    s = float(np.percentile(mag, 99)) / denom
    sx = float(np.percentile(np.abs(gx), 99)) / denom
    sy = float(np.percentile(np.abs(gy), 99)) / denom
    return s, sx, sy


def sigma_from_sharpness(s, s_ref, sigma0=1.0, max_sigma=4.0):
    """Estimate Gaussian blur sigma from overall sharpness.

    Edge slope of a Gaussian-blurred edge scales like 1/sqrt(sigma0^2 + sigma^2),
    so sigma = sigma0 * sqrt((s_ref / s)^2 - 1).
    """
    ratio = max(s_ref / max(s, 1e-6), 1.0)
    return float(min(sigma0 * np.sqrt(ratio * ratio - 1.0), max_sigma))


def motion_length(s_dir, s_ref, k=3.0, max_len=25):
    """Rough motion-blur length in pixels from the sharpness lost in the blur direction."""
    ratio = max(s_ref / max(s_dir, 1e-6), 1.0)
    return int(np.clip(round(k * ratio), 3, max_len))


def classify_blur_type(log_ratio, log_ratio_ref, delta_thr=0.5):
    """'motion_h', 'motion_v' or 'defocus' from log(sx/sy) relative to crisp text."""
    delta = log_ratio - log_ratio_ref
    if delta < -delta_thr:
        return "motion_h"      # sx collapsed => blur runs horizontally
    if delta > delta_thr:
        return "motion_v"
    return "defocus"
