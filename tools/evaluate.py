"""Evaluate the adaptive pipeline against the clean originals of synthetic pages.

For each scenario/seed it reports (image vs clean ground truth):
  degraded      - the input as it is
  global_filter - a typical one-size-fits-all pipeline (shadow normalise + sharpen + denoise on EVERYTHING)
  adaptive      - our region-aware pipeline
Metrics: PSNR, SSIM, MAE, MSE, processing time, and (with --ocr) CER / WER.

Usage:  python tools/evaluate.py --n 3 --ocr --out results/metrics
Everything printed or saved comes from this run. Nothing is hard-coded.
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config                                    # noqa: E402
from src.illumination import estimate_background, illumination_ratio  # noqa: E402
from src.metrics import Timer, cer, mae, mse, psnr, ssim, wer   # noqa: E402
from src.ocr_engine import extract_text, ocr_available          # noqa: E402
from src.pipeline import restore_document                        # noqa: E402
from src.restoration import denoise_nlm, unsharp                 # noqa: E402
from src.shadow import correct_shadow                            # noqa: E402
from src.visualize import label_overlay, put_title               # noqa: E402
from tools.make_synthetic import SCENARIOS, make_sample          # noqa: E402


def global_filter(img):
    """Baseline: apply every fix to every pixel, no analysis."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    bg = estimate_background(L)
    _, ref = illumination_ratio(bg)
    L = correct_shadow(L, bg, ref)
    L = denoise_nlm(L, 5.0)
    L = unsharp(L, 1.5, 1.0)
    lab[..., 0] = np.clip(L, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def score(name, out, clean, truth, do_ocr, seconds):
    row = {"method": name, "PSNR": psnr(out, clean), "SSIM": ssim(out, clean),
           "MAE": mae(out, clean), "MSE": mse(out, clean), "time_s": seconds}
    if do_ocr:
        txt = extract_text(out)
        row["CER"], row["WER"] = cer(truth, txt), wer(truth, txt)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="seeds per scenario")
    ap.add_argument("--ocr", action="store_true")
    ap.add_argument("--out", default="results/metrics")
    a = ap.parse_args()
    do_ocr = a.ocr and ocr_available()
    if a.ocr and not do_ocr:
        print("OCR requested but Tesseract/pytesseract not available: skipping CER/WER")
    os.makedirs(a.out, exist_ok=True)
    fig_dir = os.path.join(os.path.dirname(a.out.rstrip("/")) or ".", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cfg = Config()
    rows = []
    for sc in SCENARIOS:
        for seed in range(a.n):
            clean, deg, truth, _ = make_sample(sc, seed)
            rows.append({"scenario": sc, "seed": seed, **score("degraded", deg, clean, truth, do_ocr, 0.0)})
            with Timer() as t:
                g = global_filter(deg)
            rows.append({"scenario": sc, "seed": seed, **score("global_filter", g, clean, truth, do_ocr, t.seconds)})
            with Timer() as t:
                res = restore_document(deg, cfg, detect=False)
            rows.append({"scenario": sc, "seed": seed, **score("adaptive", res.restored, clean, truth, do_ocr, t.seconds)})
            if seed == 0:
                sm = lambda im: cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2))
                panels = [put_title(sm(deg), "degraded"), put_title(sm(g), "global filter"),
                          put_title(sm(res.restored), "adaptive"),
                          put_title(sm(label_overlay(res.working, res.analysis)), "region labels")]
                cv2.imwrite(os.path.join(fig_dir, f"eval_{sc}.png"), np.hstack(panels))
            print(f"{sc:12s} seed {seed}: {res.info['region_counts']} applied={res.info['methods_applied']}")

    cols = ["scenario", "seed", "method", "PSNR", "SSIM", "MAE", "MSE", "time_s"] + (["CER", "WER"] if do_ocr else [])
    with open(os.path.join(a.out, "evaluation.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # summary table (mean over seeds)
    lines = ["| scenario | method | PSNR (dB) | SSIM | MAE | time (s) |" + (" CER | WER |" if do_ocr else ""),
             "|---|---|---|---|---|---|" + ("---|---|" if do_ocr else "")]
    for sc in SCENARIOS:
        for m in ("degraded", "global_filter", "adaptive"):
            sel = [r for r in rows if r["scenario"] == sc and r["method"] == m]
            mean = lambda k: float(np.mean([r[k] for r in sel]))
            line = f"| {sc} | {m} | {mean('PSNR'):.2f} | {mean('SSIM'):.4f} | {mean('MAE'):.2f} | {mean('time_s'):.2f} |"
            if do_ocr:
                line += f" {mean('CER'):.3f} | {mean('WER'):.3f} |"
            lines.append(line)
    summary = "\n".join(lines)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(f"Seeds per scenario: {a.n}. Synthetic pages only. OCR: {'yes' if do_ocr else 'no'}\n\n" + summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
