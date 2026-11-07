import numpy as np

from src.metrics import cer, levenshtein, mae, mse, psnr, ssim, wer


def test_identical_images():
    a = np.full((16, 16, 3), 100, np.uint8)
    assert mse(a, a) == 0 and mae(a, a) == 0
    assert psnr(a, a) == float("inf")
    assert ssim(a, a) == 1.0


def test_psnr_known_value():
    a = np.zeros((8, 8), np.uint8)
    b = np.full((8, 8), 10, np.uint8)       # MSE = 100
    assert abs(psnr(a, b) - 10 * np.log10(255 ** 2 / 100)) < 1e-9


def test_levenshtein_classic_example():
    assert levenshtein("kitten", "sitting") == 3


def test_cer_and_wer():
    assert abs(cer("kitten", "sitting") - 3 / 6) < 1e-9
    assert wer("the cat sat", "the dog sat") == 1 / 3
    assert cer("a  b", "a b") == 0            # whitespace is normalised
