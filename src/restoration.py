"""Phase 6: the actual restoration operations (each works on a lightness image 0..255).

They are applied to the WHOLE image here; the pipeline then blends each result
back only inside the regions that need it.
"""
import cv2
import numpy as np
from skimage.restoration import richardson_lucy

from .preprocessing import odd


def denoise_nlm(L, sigma):
    """Non-local-means denoising; strength follows the estimated noise sigma."""
    g = np.clip(L, 0, 255).astype(np.uint8)
    h = float(np.clip(0.9 * sigma + 1.0, 3.0, 20.0))
    out = cv2.fastNlMeansDenoising(g, None, h=h, templateWindowSize=7, searchWindowSize=21)
    return out.astype(np.float32)


def gaussian_psf(sigma):
    """Point-spread function of defocus blur (approximated as Gaussian)."""
    size = odd(max(3, int(np.ceil(6 * sigma)) + 1))
    k = cv2.getGaussianKernel(size, sigma)
    return np.outer(k, k).astype(np.float64)


def motion_psf(length, axis="h"):
    """Straight-line motion blur kernel (horizontal or vertical)."""
    length = odd(max(int(length), 3))      # must be odd: an even kernel has no centre and RL diverges
    psf = np.zeros((length, length), np.float64)
    if axis == "h":
        psf[length // 2, :] = 1.0
    else:
        psf[:, length // 2] = 1.0
    return psf / psf.sum()


def deblur_rl(L, psf, mask, iterations=12):
    """Richardson-Lucy deconvolution, computed only inside the mask's bounding box."""
    ys, xs = np.nonzero(mask > 0.01)
    if len(ys) == 0:
        return L
    h, w = L.shape
    pad = int(max(psf.shape)) + 2
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
    crop = (np.clip(L[y0:y1, x0:x1], 0, 255) / 255.0).astype(np.float64)
    padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    out = richardson_lucy(padded, psf, num_iter=iterations, clip=True)
    out = out[pad:-pad, pad:-pad] * 255.0
    full = L.copy()
    full[y0:y1, x0:x1] = out.astype(np.float32)
    return full


def unsharp(L, sigma=1.5, amount=1.0):
    """Classic global sharpening (used as a comparison baseline)."""
    blur = cv2.GaussianBlur(L, (0, 0), sigma)
    return np.clip(L + amount * (L - blur), 0, 255)
