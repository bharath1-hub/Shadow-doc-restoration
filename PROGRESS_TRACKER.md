# Progress Tracker

Rule: a task is marked done only if it was implemented AND tested. "Tested" below means
tested on SYNTHETIC pages unless stated otherwise.

| Phase | Topic | Code | Tested | Your learning |
|---|---|---|---|---|
| 1 | Python and environment | done | You ran `check_setup.py`: OpenCV 5.0.0, NumPy 2.5.4, image (1599, 899, 3) uint8 | Lesson 1 exercises NOT answered |
| 2 | Image processing fundamentals | used throughout (`preprocessing.py`, gradients, thresholding, morphology) | via later phases | Lessons NOT taught yet |
| 3 | Document detection | `document_detection.py` | 4 unit tests + perspective-photo test | Not yet studied |
| 4 | Illumination and shadow | `illumination.py`, `shadow.py` | 3 unit tests, E006 | Not yet studied |
| 5 | Blur estimation | `blur_estimation.py`, `noise_estimation.py` | 5 unit tests, E001-E002 | Not yet studied |
| 6 | Adaptive restoration | `region_classifier.py`, `restoration.py`, `blending.py`, `pipeline.py` | 12 pipeline tests, E003-E006 | Not yet studied |
| 7 | AI/ML and deep learning | intentionally not built | n/a | See docs/phase7_decision.md |
| 8 | OCR | `ocr_engine.py` (Tesseract) | used in evaluation, CER/WER measured | Not yet studied |
| 9 | Evaluation | `metrics.py`, `tools/evaluate.py` | 4 metric tests, E006 | Not yet studied |
| 10 | Application | `app/streamlit_app.py`, `run_pipeline.py` | CLI run end to end; app loads without errors. **Upload flow not tested by a human click-through** | Not yet studied |
| 11 | Paper, report, viva | outlines + `docs/code_walkthrough.md` (viva Q&A) | n/a | Literature survey NOT started |

Automated tests: `python -m pytest -q` (31 tests).

## Still to do (important)
1. **Test on your own real phone photos.** Nothing has been run on real images yet.
   Put photos in `data/raw/`, run `python run_pipeline.py data/raw/yourphoto.jpg --ocr`, look at the
   `_regions.png` output, and retune `src/config.py` if the labels look wrong. Log it in `docs/experiment_log.md`.
2. **Literature survey** (`docs/literature_survey.md`): no papers have been collected or verified yet.
   The "research gap" is still a hypothesis until you read them.
3. **Public datasets** (`docs/dataset_notes.md`): none chosen or downloaded yet.
4. **Learn the code**: read `docs/code_walkthrough.md`, then do the Lesson exercises when you have time.
5. **Real paired data** (same page photographed with and without shadow/blur) for honest evaluation.

## Known limitations
- Thresholds were calibrated on synthetic pages (one text size, white paper).
- Deblurring gains are modest; a wrong blur type can hurt.
- Motion blur handled only horizontally / vertically.
- Page detection needs a visible contrast between page and background.
- Large dark graphics on the page may be mistaken for shadow.
