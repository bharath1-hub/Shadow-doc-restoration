import pytest

from src.metrics import no_reference_quality, psnr
from src.pipeline import restore_document
from tools.make_synthetic import make_photo, make_sample


def _run(scenario):
    clean, deg, _, _ = make_sample(scenario, 0)
    res = restore_document(deg, detect=False)
    return clean, deg, res


def test_clean_page_is_left_alone():
    clean, deg, res = _run("clean")
    assert res.info["methods_applied"] == []
    assert res.info["untouched_pixel_fraction"] > 0.99
    assert psnr(res.restored, clean) > 50


@pytest.mark.parametrize("scenario", ["shadow", "defocus", "motion", "shadow_blur", "noise", "combined"])
def test_adaptive_never_worse_than_input(scenario):
    clean, deg, res = _run(scenario)
    assert psnr(res.restored, clean) >= psnr(deg, clean) - 0.1


def test_shadow_scenario_gets_shadow_correction_only():
    clean, deg, res = _run("shadow")
    assert "shadow_correction" in res.info["methods_applied"]
    assert not any(m.startswith("deblur") for m in res.info["methods_applied"])
    assert psnr(res.restored, clean) > psnr(deg, clean) + 8


def test_shadow_blur_gets_both_treatments():
    _, _, res = _run("shadow_blur")
    assert "Shadow+Blur" in res.info["region_counts"]
    assert "shadow_correction" in res.info["methods_applied"]
    assert "deblur_defocus" in res.info["methods_applied"]


def test_motion_is_typed_as_motion():
    _, _, res = _run("motion")
    assert "deblur_motion_h" in res.info["methods_applied"]


def test_noise_scenario_triggers_denoise():
    _, _, res = _run("noise")
    assert "denoise" in res.info["methods_applied"]


def test_end_to_end_on_perspective_photo():
    clean, deg, _, _ = make_sample("shadow", 1)
    photo, _ = make_photo(deg, seed=7)
    res = restore_document(photo, detect=True)
    assert res.info["document_detected"]
    before = no_reference_quality(res.working)["illumination_nonuniformity"]
    after = no_reference_quality(res.restored)["illumination_nonuniformity"]
    assert after < before
