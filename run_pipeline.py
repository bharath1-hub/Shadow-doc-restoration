"""Restore one document photo from the command line.

Examples:
    python run_pipeline.py data/raw/test1.jpg
    python run_pipeline.py data/raw/test1.jpg --out results/restored --ocr
    python run_pipeline.py data/synthetic/combined_0_degraded.png --no-detect
"""
import argparse
import json
import os

import cv2

from src.config import Config
from src.io_utils import load_image, save_image
from src.metrics import no_reference_quality
from src.ocr_engine import extract_text, ocr_available
from src.pipeline import restore_document
from src.visualize import label_overlay


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--out", default="results/restored")
    ap.add_argument("--no-detect", action="store_true", help="skip page detection / perspective correction")
    ap.add_argument("--ocr", action="store_true", help="also run Tesseract before and after")
    a = ap.parse_args()

    img = load_image(a.image)
    res = restore_document(img, Config(), detect=not a.no_detect)
    name = os.path.splitext(os.path.basename(a.image))[0]
    save_image(os.path.join(a.out, f"{name}_restored.png"), res.restored)
    save_image(os.path.join(a.out, f"{name}_regions.png"), label_overlay(res.working, res.analysis))

    info = dict(res.info)
    info["quality_before"] = no_reference_quality(res.working)
    info["quality_after"] = no_reference_quality(res.restored)
    if a.ocr:
        if ocr_available():
            info["ocr_before"] = extract_text(res.working)
            info["ocr_after"] = extract_text(res.restored)
        else:
            info["ocr"] = "Tesseract / pytesseract not available"
    with open(os.path.join(a.out, f"{name}_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: v for k, v in info.items() if not k.startswith("ocr_")}, indent=2))
    print("saved to", a.out)


if __name__ == "__main__":
    main()
