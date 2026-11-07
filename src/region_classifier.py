"""Phase 6: split the page into patches, measure each one, and decide what is wrong.

Every patch gets three independent flags (shadow, blur, noise). The label is the
combination, e.g. "Normal", "Shadow", "Blur", "Shadow+Blur", "Noisy".
"""
from dataclasses import dataclass
from math import ceil

import numpy as np

from .blur_estimation import (classify_blur_type, edge_metrics, has_text,
                              motion_length, sigma_from_sharpness)
from .config import Config
from .illumination import estimate_background, illumination_ratio, normalize_by_background
from .noise_estimation import estimate_noise_sigma


@dataclass
class RegionAnalysis:
    patch: int
    img_shape: tuple
    bg: np.ndarray            # illumination (background) map
    ratio: np.ndarray         # local / reference brightness, per pixel
    ref: float                # reference brightness level
    norm: np.ndarray          # illumination-normalised image (used for blur measurement)
    ratio_grid: np.ndarray
    text_grid: np.ndarray     # bool
    sharp_grid: np.ndarray    # overall sharpness s (NaN where no text)
    smin_grid: np.ndarray     # min(sx, sy) = blur score (NaN where no text)
    noise_grid: np.ndarray    # NaN where unknown
    shadow: np.ndarray        # bool grids
    blur: np.ndarray
    noise: np.ndarray
    blur_type: np.ndarray     # object grid: '', 'defocus', 'motion_h', 'motion_v'
    blur_sigma: np.ndarray
    motion_len: np.ndarray
    labels: np.ndarray        # object grid of strings
    sharp_ref: float

    def label_counts(self):
        names, counts = np.unique(self.labels, return_counts=True)
        return {str(n): int(c) for n, c in zip(names, counts)}


def _label(shadow, blur, noise):
    parts = []
    if shadow:
        parts.append("Shadow")
    if blur:
        parts.append("Blur")
    if noise:
        parts.append("Noisy")
    return "+".join(parts) if parts else "Normal"


def analyze_regions(L, cfg=None):
    """Analyse a lightness image L (float, 0..255) patch by patch."""
    cfg = cfg or Config()
    L = L.astype(np.float32)
    h, w = L.shape
    p = cfg.patch

    bg = estimate_background(L)
    ratio, ref = illumination_ratio(bg)
    norm = normalize_by_background(L, bg, ref)

    rows, cols = ceil(h / p), ceil(w / p)
    shape = (rows, cols)
    ratio_grid = np.zeros(shape, np.float32)
    text_grid = np.zeros(shape, bool)
    sharp_grid = np.full(shape, np.nan, np.float32)
    sx_grid = np.full(shape, np.nan, np.float32)
    sy_grid = np.full(shape, np.nan, np.float32)
    smin_grid = np.full(shape, np.nan, np.float32)
    lr_grid = np.full(shape, np.nan, np.float32)
    noise_grid = np.full(shape, np.nan, np.float32)
    shadow = np.zeros(shape, bool)

    for r in range(rows):
        for c in range(cols):
            ys, ye = r * p, min(h, (r + 1) * p)
            xs, xe = c * p, min(w, (c + 1) * p)
            rp = ratio[ys:ye, xs:xe]
            ratio_grid[r, c] = rp.mean()
            shadow[r, c] = (rp < cfg.shadow_ratio_thr).mean() > cfg.shadow_min_frac
            npatch = norm[ys:ye, xs:xe]
            if has_text(npatch, cfg):
                text_grid[r, c] = True
                s, sx, sy = edge_metrics(npatch, ref)
                sharp_grid[r, c], sx_grid[r, c], sy_grid[r, c] = s, sx, sy
                smin_grid[r, c] = min(sx, sy)
                lr_grid[r, c] = np.log(max(sx, 1e-6) / max(sy, 1e-6))
            noise_grid[r, c] = estimate_noise_sigma(L[ys:ye, xs:xe])

    # reference values from the crisp patches of THIS document (if there are enough)
    crisp = text_grid & (smin_grid >= cfg.blur_sharp_thr)
    if crisp.sum() >= 3:
        s_ref = float(np.nanmedian(sharp_grid[crisp]))
        smin_ref = float(np.nanmedian(smin_grid[crisp]))
        lr_ref = float(np.nanmedian(lr_grid[crisp]))
    else:
        s_ref, smin_ref, lr_ref = cfg.sharp_ref_default, cfg.smin_ref_default, 0.07

    blur = text_grid & (smin_grid < cfg.blur_sharp_thr)
    noise = np.nan_to_num(noise_grid, nan=0.0) > cfg.noise_sigma_thr

    # blur type per patch, then a 5x5 majority vote so a stray patch cannot get a wrong type
    raw_type = np.full(shape, "", dtype=object)
    for r, c in zip(*np.nonzero(blur)):
        raw_type[r, c] = classify_blur_type(lr_grid[r, c], lr_ref, cfg.motion_delta)
    blur_type = np.full(shape, "", dtype=object)
    blur_sigma = np.zeros(shape, np.float32)
    motion_len = np.zeros(shape, np.float32)
    for r, c in zip(*np.nonzero(blur)):
        window = raw_type[max(r - 2, 0):r + 3, max(c - 2, 0):c + 3].ravel()
        names, counts = np.unique(window[window != ""], return_counts=True)
        t = str(names[np.argmax(counts)])
        blur_type[r, c] = t
        if t == "defocus":
            blur_sigma[r, c] = sigma_from_sharpness(sharp_grid[r, c], s_ref,
                                                    cfg.sigma0, cfg.max_sigma)
        else:
            motion_len[r, c] = motion_length(smin_grid[r, c], smin_ref, cfg.len_k)

    labels = np.empty(shape, dtype=object)
    for r in range(rows):
        for c in range(cols):
            labels[r, c] = _label(shadow[r, c], blur[r, c], noise[r, c])

    return RegionAnalysis(p, (h, w), bg, ratio, ref, norm, ratio_grid, text_grid,
                          sharp_grid, smin_grid, noise_grid, shadow, blur, noise, blur_type,
                          blur_sigma, motion_len, labels, s_ref)
