import cv2
import numpy as np

from src.illumination import estimate_background, illumination_ratio
from src.metrics import psnr
from src.shadow import correct_shadow, detect_shadow_mask
from tools.make_synthetic import make_sample


def _L(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)


def test_clean_page_has_no_shadow():
    clean, _, _, _ = make_sample("clean", 0)
    ratio, _ = illumination_ratio(estimate_background(_L(clean)))
    assert detect_shadow_mask(ratio).mean() < 0.01


def test_shadow_is_detected_where_it_was_added():
    clean, deg, _, _ = make_sample("shadow", 0)
    ratio, _ = illumination_ratio(estimate_background(_L(deg)))
    mask = detect_shadow_mask(ratio)
    inside = mask[800:1150, 50:450].mean()      # well inside the synthetic shadow box
    outside = mask[0:500, :].mean()
    assert inside > 0.9 and outside < 0.02


def test_shadow_correction_improves_psnr():
    clean, deg, _, _ = make_sample("shadow", 0)
    L, Lc = _L(deg), _L(clean)
    bg = estimate_background(L)
    _, ref = illumination_ratio(bg)
    fixed = correct_shadow(L, bg, ref)
    assert psnr(fixed, Lc) > psnr(L, Lc) + 8
