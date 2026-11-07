# Shadow Aware Image Quality Restoration for Smartphone Captured Text
### Using Local Illumination and Blur Adaptive Processing

A region-aware restoration pipeline for phone-captured document photos. The page is cut into
patches; each patch is diagnosed (shadow? blur? noise?) and gets only the fix it needs.
Clean regions are left untouched.

| Region condition | Action |
|---|---|
| Clean | Preserve |
| Shadow | Illumination correction (divide by estimated background) |
| Blur | Richardson-Lucy deblurring (Gaussian kernel for defocus, line kernel for motion) |
| Shadow + blur | Both, in the right order |
| Noise | Non-local-means denoising |

## Setup
```
python -m venv venv
venv\Scripts\activate              # Windows PowerShell:  venv\Scripts\Activate.ps1
source venv/bin/activate           # Linux / macOS
pip install -r requirements.txt
python -m pytest -q                # should print: 31 passed
```
**OCR needs the Tesseract program** (separate from `pip install pytesseract`):
- Windows: install from the UB Mannheim Tesseract builds, then add its folder to PATH.
- Ubuntu/Debian: `sudo apt install tesseract-ocr`
- macOS: `brew install tesseract`

Without Tesseract everything else still works; OCR is skipped.

## Use it
```
python run_pipeline.py data/raw/test1.jpg --ocr                 # one photo -> results/restored/
python run_pipeline.py data/synthetic/photo_combined.jpg --ocr  # included sample photo
streamlit run app/streamlit_app.py                              # web UI
python tools/make_synthetic.py --out data/synthetic --n 3       # regenerate synthetic pages
python tools/evaluate.py --n 3 --ocr --out results/metrics      # evaluation vs clean originals
```
Outputs per photo: `*_restored.png`, `*_regions.png` (what was decided where), `*_info.json`.

## Layout
```
src/            pipeline stages (one module each), config.py holds every threshold
tools/          synthetic data generator, evaluation script
tests/          31 automated tests
app/            Streamlit UI
data/synthetic  sample pages (clean, degraded, truth text) and phone-style photos
results/        evaluation.csv, summary.md, comparison figures
docs/           experiment log, code walkthrough + viva Q&A, outlines, Phase 7 decision
lessons/        your study notes
```
Start reading with `docs/code_walkthrough.md`.

## What was measured (synthetic pages, 3 seeds each; full table in docs/experiment_log.md)
| scenario | PSNR degraded | PSNR global filter | PSNR adaptive |
|---|---|---|---|
| clean | inf | 25.75 | 67.30 |
| shadow | 14.21 | 25.61 | 37.36 |
| defocus | 20.37 | 20.68 | 21.94 |
| motion | 18.85 | 18.17 | 20.40 |
| shadow + blur | 13.18 | 20.35 | 21.46 |
| noise | 31.08 | 25.35 | 31.92 |
| combined | 13.56 | 20.55 | 21.43 |

## Read this before you claim anything
- **All results are on synthetic documents.** Nothing has been tested on real phone photos yet.
- The big wins are shadow removal and *not damaging clean regions*. Deblurring gains are modest (about 1 to 2 dB).
- OCR gains appear for shadows; Tesseract already handles mild defocus well.
- Thresholds in `src/config.py` come from synthetic data and must be re-checked on your photos.
- Motion blur is handled for horizontal/vertical directions only.
- The literature survey and public-dataset search are not done; the "research gap" is unverified.
- Deep learning is intentionally not used yet (docs/phase7_decision.md).

See `PROGRESS_TRACKER.md` for status and the to-do list.
