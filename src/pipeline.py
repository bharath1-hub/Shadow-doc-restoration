"""The whole system: photo in -> region-aware restored page out."""
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .blending import blend, grid_to_mask
from .config import Config
from .document_detection import detect_and_warp
from .preprocessing import resize_max_side
from .region_classifier import RegionAnalysis, analyze_regions
from .restoration import deblur_rl, denoise_nlm, gaussian_psf, motion_psf
from .shadow import correct_shadow


@dataclass
class PipelineResult:
    original: np.ndarray
    working: np.ndarray        # flattened + resized image that was analysed
    restored: np.ndarray
    analysis: RegionAnalysis
    info: dict


def restore_document(img_bgr, cfg=None, detect=True):
    """Run detection, analysis and adaptive restoration. Returns a PipelineResult."""
    cfg = cfg or Config()
    timings, applied = {}, []
    t_all = time.perf_counter()

    # 1. document detection + perspective correction
    t = time.perf_counter()
    corners = None
    work = img_bgr
    if detect:
        work, corners = detect_and_warp(img_bgr)
    work = resize_max_side(work, cfg.max_side)
    timings["detection"] = time.perf_counter() - t

    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    shape = L.shape

    # 2. region analysis
    t = time.perf_counter()
    an = analyze_regions(L, cfg)
    timings["analysis"] = time.perf_counter() - t

    # 3. adaptive restoration (only where flagged)
    t = time.perf_counter()
    out = L.copy()
    touched = np.zeros(shape, np.float32)
    params = {}

    if an.noise.any():                                           # denoise first
        m = grid_to_mask(an.noise, cfg.patch, shape, cfg.feather)
        sigma = float(np.nanmedian(an.noise_grid[an.noise]))
        out = blend(out, denoise_nlm(out, sigma), m)
        touched = np.maximum(touched, m)
        applied.append("denoise")
        params["noise_sigma"] = round(sigma, 2)

    if an.shadow.any():                                          # then shadow correction
        m = grid_to_mask(an.shadow, cfg.patch, shape, cfg.feather)
        out = blend(out, correct_shadow(out, an.bg, an.ref), m)
        touched = np.maximum(touched, m)
        applied.append("shadow_correction")

    for btype in ("defocus", "motion_h", "motion_v"):            # finally deblur, per blur type
        grid = an.blur & (an.blur_type == btype)
        if not grid.any():
            continue
        m = grid_to_mask(grid, cfg.patch, shape, cfg.feather)
        if btype == "defocus":
            sig = float(np.median(an.blur_sigma[grid]))
            psf = gaussian_psf(max(sig, 0.5))
            params["defocus_sigma"] = round(sig, 2)
        else:
            length = int(np.median(an.motion_len[grid]))
            psf = motion_psf(length, "h" if btype == "motion_h" else "v")
            params[f"{btype}_length"] = length
        out = blend(out, deblur_rl(out, psf, m, cfg.rl_iterations), m)
        touched = np.maximum(touched, m)
        applied.append(f"deblur_{btype}")
    timings["restoration"] = time.perf_counter() - t

    lab[..., 0] = np.clip(out, 0, 255).astype(np.uint8)
    restored = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    timings["total"] = time.perf_counter() - t_all

    info = {
        "document_detected": corners is not None,
        "corners": None if corners is None else corners.round(1).tolist(),
        "working_size_hw": list(shape),
        "region_counts": an.label_counts(),
        "regions_total": int(an.labels.size),
        "untouched_pixel_fraction": round(float((touched < 0.05).mean()), 4),
        "methods_applied": applied,
        "parameters": params,
        "reference_brightness": round(an.ref, 1),
        "sharpness_reference": round(an.sharp_ref, 3),
        "timings_s": {k: round(v, 3) for k, v in timings.items()},
    }
    return PipelineResult(img_bgr, work, restored, an, info)
