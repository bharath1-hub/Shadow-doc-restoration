import numpy as np
import pytest
from src.io_utils import load_image, save_image, resize_keep_aspect


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_image("data/raw/does_not_exist.jpg")


def test_save_and_load_roundtrip(tmp_path):
    img = np.full((20, 30, 3), 128, dtype=np.uint8)
    p = str(tmp_path / "x.png")
    save_image(p, img)
    out = load_image(p)
    assert out.shape == (20, 30, 3)


def test_resize_never_enlarges():
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    assert resize_keep_aspect(img, 1000).shape == img.shape
    assert max(resize_keep_aspect(img, 50).shape[:2]) == 50
