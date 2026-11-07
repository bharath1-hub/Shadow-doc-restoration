"""Phase 6: turn patch flags into smooth masks and blend processed regions back."""
import cv2
import numpy as np


def grid_to_mask(grid, patch, shape, feather=16.0):
    """Upsample a boolean patch grid to a pixel mask in [0, 1] with soft borders."""
    h, w = shape
    up = np.kron(grid.astype(np.float32), np.ones((patch, patch), np.float32))[:h, :w]
    if feather > 0:
        up = cv2.GaussianBlur(up, (0, 0), feather)
    return np.clip(up, 0.0, 1.0)


def blend(base, processed, mask):
    """out = base * (1 - mask) + processed * mask."""
    return base * (1.0 - mask) + processed * mask
