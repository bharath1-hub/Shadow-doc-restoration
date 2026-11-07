"""Colour overlay showing what the region classifier decided."""
import cv2
import numpy as np

COLORS = {          # BGR
    "Shadow": (255, 128, 0),
    "Blur": (0, 0, 255),
    "Shadow+Blur": (255, 0, 255),
    "Noisy": (0, 215, 255),
}
OTHER = (128, 128, 128)


def label_overlay(img, analysis, alpha=0.40):
    """Return a copy of `img` with every non-Normal patch tinted by its label colour."""
    out = img.copy()
    layer = img.copy()
    p = analysis.patch
    h, w = img.shape[:2]
    rows, cols = analysis.labels.shape
    for r in range(rows):
        for c in range(cols):
            lab = analysis.labels[r, c]
            if lab == "Normal":
                continue
            col = COLORS.get(lab, OTHER)
            cv2.rectangle(layer, (c * p, r * p), (min(w, (c + 1) * p) - 1, min(h, (r + 1) * p) - 1),
                          col, -1)
    return cv2.addWeighted(layer, alpha, out, 1 - alpha, 0)


def put_title(img, text):
    """Draw a title bar on top of an image."""
    bar = np.full((36, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack([bar, img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)])
