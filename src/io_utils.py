"""Reading, saving and displaying images (Phase 1/2)."""
import os
import cv2


def load_image(path):
    """Read an image in BGR. Raises FileNotFoundError instead of returning None."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def save_image(path, img):
    """Save an image, creating the folder if needed."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if not cv2.imwrite(path, img):
        raise IOError(f"Could not write image: {path}")


def resize_keep_aspect(img, max_side=1200):
    """Shrink so the longer side equals max_side (never enlarges)."""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
