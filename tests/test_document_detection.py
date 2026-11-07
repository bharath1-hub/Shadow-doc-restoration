import numpy as np

from src.document_detection import detect_and_warp, find_document_corners, order_points
from tools.make_synthetic import make_photo, make_sample


def test_order_points():
    pts = np.array([[100, 300], [0, 0], [100, 0], [0, 300]], np.float32)
    o = order_points(pts)
    assert o.tolist() == [[0, 0], [100, 0], [100, 300], [0, 300]]


def test_detects_page_on_table_within_tolerance():
    clean, _, _, _ = make_sample("clean", 0)
    photo, true_corners = make_photo(clean, seed=3)
    found = find_document_corners(photo)
    assert found is not None
    err = np.linalg.norm(found - order_points(true_corners), axis=1)
    assert err.max() < 20, f"corner error too large: {err}"


def test_warp_output_has_page_aspect_ratio():
    clean, _, _, _ = make_sample("clean", 0)
    photo, _ = make_photo(clean, seed=5)
    warped, corners = detect_and_warp(photo)
    assert corners is not None
    h, w = warped.shape[:2]
    assert abs(w / h - 1000 / 1400) < 0.08


def test_no_page_returns_original():
    blank = np.full((400, 300, 3), 120, np.uint8)
    out, corners = detect_and_warp(blank)
    assert corners is None and out is blank
