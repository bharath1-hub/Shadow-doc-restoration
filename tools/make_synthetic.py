"""Synthetic test documents with KNOWN ground truth.

Why synthetic? To measure a restoration method you need the clean original.
We render a clean text page, then add shadows / blur / noise in chosen regions.
Results on these images prove the code does what it claims on controlled
degradations. They do NOT prove it works on real phone photos: test those too.

Usage:  python tools/make_synthetic.py --out data/synthetic --n 3
"""
import argparse
import os

import cv2
import numpy as np

WORDS = ("the quick brown fox jumps over lazy dog document image restoration shadow "
         "region adaptive local illumination blur noise smartphone capture text quality "
         "page paper reading engineer project final year system method result analysis "
         "pixel contrast edge filter algorithm detect estimate correct measure output "
         "student office report letter number total amount date name address").split()
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
SCENARIOS = ["clean", "shadow", "defocus", "motion", "shadow_blur", "noise", "combined"]


def render_document(seed=0, width=1000, height=1400):
    """Return (clean BGR page, ground-truth text)."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 245, np.uint8)
    font, scale, thick = FONTS[seed % len(FONTS)], 0.8, 2
    margin, y, line_h = 60, 90, 46
    lines = []
    while y < height - 60:
        line = ""
        while True:
            cand = (line + " " + str(rng.choice(WORDS))).strip()
            (tw, _), _ = cv2.getTextSize(cand, font, scale, thick)
            if tw > width - 2 * margin:
                break
            line = cand
        cv2.putText(img, line, (margin, y), font, scale, (25, 25, 25), thick, cv2.LINE_AA)
        lines.append(line)
        y += line_h
    return img, "\n".join(lines)


def _soft_box(shape, box, feather=30):
    m = np.zeros(shape[:2], np.float32)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 1.0
    return cv2.GaussianBlur(m, (0, 0), feather)[..., None]


def add_shadow(img, box, strength=0.5, softness=40):
    """Darken a (soft) rectangular area to `strength` of its brightness."""
    m = _soft_box(img.shape, box, softness)
    return (img.astype(np.float32) * (1 - (1 - strength) * m)).clip(0, 255).astype(np.uint8)


def blur_region(img, box, kind="defocus", param=2.0, feather=20):
    """Blur only inside `box` (spatially varying blur)."""
    if kind == "defocus":
        blurred = cv2.GaussianBlur(img, (0, 0), param)
    else:  # horizontal motion blur of length `param`
        k = np.zeros((int(param), int(param)), np.float32)
        k[int(param) // 2, :] = 1.0 / int(param)
        blurred = cv2.filter2D(img, -1, k)
    m = _soft_box(img.shape, box, feather)
    return (img * (1 - m) + blurred * m).clip(0, 255).astype(np.uint8)


def add_noise(img, box, sigma=12, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape).astype(np.float32)
    m = _soft_box(img.shape, box, 20)
    return (img.astype(np.float32) + noise * m).clip(0, 255).astype(np.uint8)


def make_sample(scenario, seed=0):
    """Return (clean, degraded, truth_text, meta)."""
    clean, text = render_document(seed)
    h, w = clean.shape[:2]
    d = clean.copy()
    meta = {"scenario": scenario, "seed": seed}
    if scenario == "clean":
        pass
    elif scenario == "shadow":
        d = add_shadow(d, (0, 700, 560, 1250), 0.5)
    elif scenario == "defocus":
        d = blur_region(d, (0, 0, w, 650), "defocus", 2.0)
    elif scenario == "motion":
        d = blur_region(d, (0, 0, w, 650), "motion", 15)
    elif scenario == "shadow_blur":
        d = blur_region(d, (0, 700, w, 1400), "defocus", 2.0)
        d = add_shadow(d, (0, 700, 560, 1250), 0.5)
    elif scenario == "noise":
        d = add_noise(d, (0, 0, w, 650), 12, seed)
    elif scenario == "combined":
        d = blur_region(d, (0, 0, w, 650), "defocus", 2.0)
        d = add_shadow(d, (450, 300, w, 900), 0.55)
        d = add_noise(d, (0, 800, w, 1400), 12, seed)
    else:
        raise ValueError(scenario)
    return clean, d, text, meta


def make_photo(page, seed=0, canvas=(1500, 2000)):
    """Put a page on a dark table with a random perspective. Returns (photo, corners tl,tr,br,bl)."""
    rng = np.random.default_rng(seed)
    cw, ch = canvas
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    cx, cy = cw / 2, ch / 2
    base = np.float32([[cx - 480, cy - 650], [cx + 480, cy - 650],
                       [cx + 480, cy + 650], [cx - 480, cy + 650]])
    dst = base + rng.uniform(-60, 60, base.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    table = np.full((ch, cw, 3), (40, 55, 70), np.uint8)
    table = np.clip(table + rng.normal(0, 4, table.shape), 0, 255).astype(np.uint8)
    warped = cv2.warpPerspective(page, M, (cw, ch), flags=cv2.INTER_CUBIC)
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (cw, ch))
    photo = np.where(mask[..., None] > 127, warped, table)
    return photo, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--n", type=int, default=3, help="seeds per scenario")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for sc in SCENARIOS:
        for s in range(a.n):
            clean, deg, text, _ = make_sample(sc, s)
            base = os.path.join(a.out, f"{sc}_{s}")
            cv2.imwrite(base + "_clean.png", clean)
            cv2.imwrite(base + "_degraded.png", deg)
            with open(base + "_truth.txt", "w") as f:
                f.write(text)
            if s == 0:   # also a phone-style photo (page on a table, perspective) for the detector
                photo, _ = make_photo(deg, seed=s + 11)
                cv2.imwrite(os.path.join(a.out, f"photo_{sc}.jpg"), photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("written to", a.out)


if __name__ == "__main__":
    main()
