"""Phase 8: OCR with Tesseract (needs the Tesseract program installed, see README)."""
import shutil

import cv2


def ocr_available():
    if shutil.which("tesseract") is None:
        return False
    try:
        import pytesseract  # noqa: F401
        return True
    except ImportError:
        return False


def extract_text(img, lang="eng", psm=6):
    """Return the text found in a BGR/gray image, or '' if OCR is unavailable.

    psm 6 = assume one uniform block of text, which suits a flattened page.
    """
    if not ocr_available():
        return ""
    import pytesseract
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img.ndim == 3 else img
    return pytesseract.image_to_string(rgb, lang=lang, config=f"--psm {psm}")
