import cv2
import numpy as np

from src.blur_estimation import classify_blur_type, edge_metrics, motion_length, sigma_from_sharpness
from src.illumination import estimate_background, illumination_ratio, normalize_by_background
from src.noise_estimation import estimate_noise_sigma
from tools.make_synthetic import blur_region, render_document


def _patch(img, r=6, c=7):
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    bg = estimate_background(L)
    _, ref = illumination_ratio(bg)
    n = normalize_by_background(L, bg, ref)
    return n[r * 64:(r + 1) * 64, c * 64:(c + 1) * 64], ref


def test_sharp_text_scores_high_blurred_scores_low():
    clean, _ = render_document(0)
    blurred = blur_region(clean, (0, 0, 1000, 1400), "defocus", 2.0)
    s_clean = edge_metrics(*_patch(clean))
    s_blur = edge_metrics(*_patch(blurred))
    assert min(s_clean[1], s_clean[2]) > 0.55
    assert min(s_blur[1], s_blur[2]) < 0.40


def test_motion_vs_defocus_type():
    clean, _ = render_document(0)
    _, sx_c, sy_c = edge_metrics(*_patch(clean))
    lr_ref = np.log(sx_c / sy_c)
    mot = blur_region(clean, (0, 0, 1000, 1400), "motion", 15)
    dfo = blur_region(clean, (0, 0, 1000, 1400), "defocus", 2.0)
    _, sx, sy = edge_metrics(*_patch(mot))
    assert classify_blur_type(np.log(sx / sy), lr_ref) == "motion_h"
    _, sx, sy = edge_metrics(*_patch(dfo))
    assert classify_blur_type(np.log(sx / sy), lr_ref) == "defocus"


def test_parameter_estimates_are_in_the_right_range():
    clean, _ = render_document(0)
    s_ref = edge_metrics(*_patch(clean))[0]
    s_blur = edge_metrics(*_patch(blur_region(clean, (0, 0, 1000, 1400), "defocus", 2.0)))[0]
    est = sigma_from_sharpness(s_blur, s_ref)
    assert 1.4 < est < 3.0                     # true sigma is 2.0
    assert 3 <= motion_length(0.14, 0.66) <= 25


def test_noise_estimate_close_to_truth():
    rng = np.random.default_rng(0)
    flat = np.full((64, 64), 128, np.float32) + rng.normal(0, 8, (64, 64)).astype(np.float32)
    est = estimate_noise_sigma(flat)
    assert abs(est - 8) / 8 < 0.2


def test_noise_estimate_near_zero_on_clean_flat_patch():
    assert estimate_noise_sigma(np.full((64, 64), 200, np.float32)) < 0.5
