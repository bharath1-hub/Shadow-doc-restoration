"""Phase 5/6: noise level estimate (std of the noise in grey levels).

Method: use only FLAT pixels (no text edges). There, (pixel - smoothed pixel) is
almost pure noise. Its robust spread (MAD) gives sigma, corrected for the
variance that the smoothing filter removes from the residual.
"""
import cv2
import numpy as np

_SMOOTH_SIGMA = 1.5


def _residual_gain():
    k = cv2.getGaussianKernel(int(6 * _SMOOTH_SIGMA) | 1, _SMOOTH_SIGMA)
    h = np.outer(k, k)
    h0 = h[h.shape[0] // 2, h.shape[1] // 2]
    return float(np.sqrt(1.0 - 2.0 * h0 + np.sum(h * h)))


_GAIN = _residual_gain()


def estimate_noise_sigma(patch, flat_thr=10.0, min_pixels=200):
    """Noise std of a patch in grey levels, or NaN if there are too few flat pixels."""
    p = patch.astype(np.float32)
    sm = cv2.GaussianBlur(p, (0, 0), _SMOOTH_SIGMA)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
    nonflat = (np.sqrt(gx * gx + gy * gy) >= flat_thr).astype(np.uint8)
    nonflat = cv2.dilate(nonflat, np.ones((5, 5), np.uint8))   # drop the halo around edges
    flat = nonflat == 0
    if int(flat.sum()) < min_pixels:
        return float("nan")
    resid = (p - sm)[flat]
    mad = np.median(np.abs(resid - np.median(resid)))
    return float(1.4826 * mad / _GAIN)
