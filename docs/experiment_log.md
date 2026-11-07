# Experiment Log

Every number here was produced by running the code in this repository on
**synthetic** pages (`tools/make_synthetic.py`). Nothing was run on real phone
photos yet. Add your own real-photo experiments below as new entries.

## E001: first measurement of crisp vs blurred text (initial metric)
- Metric: sharpness = p99(gradient) / (4 x patch contrast).
- Result: crisp text scored 0.72 (very stable, p5 0.71 to p95 0.73). But a sigma = 2.0
  defocus blur still gave a median of 0.70 (!) because thin strokes lose their *own*
  contrast when blurred, which cancelled the signal. Motion blur (L = 15) was also
  invisible to the overall gradient: horizontal strokes keep their steepness.
- Outcome: **metric rejected**, redesigned (E002).

## E002: redesigned blur metric (contrast floor + directional sharpness)
- Change: contrast reference C = max(patch contrast, 0.7 x page brightness); blur score
  = min(sx, sy) (weaker of horizontal / vertical edge sharpness); type from log(sx / sy).
- Results (median over ~120 text patches, 64 px):

| condition | min(sx, sy) | log(sx/sy) |
|---|---|---|
| crisp | 0.66 | +0.07 |
| defocus sigma 1.0 | 0.48 | +0.09 |
| defocus sigma 1.5 | 0.40 | +0.04 |
| defocus sigma 2.0 | 0.28 | -0.04 |
| defocus sigma 3.0 | 0.16 | -0.17 |
| horizontal motion L = 7 | 0.24 | -0.90 |
| horizontal motion L = 11 | 0.19 | -1.17 |
| horizontal motion L = 15 | 0.14 | -1.47 |
| horizontal motion L = 21 | 0.10 | -1.76 |

- Outcome: threshold 0.50 on min(sx, sy) separates crisp from blurred; log-ratio separates
  motion from defocus. Parameter estimates computed from these medians (verified numerically):
  defocus sigma true -> estimated: 1.0 -> 0.88 (-12 %), 1.5 -> 1.30 (-14 %), 2.0 -> 2.02 (+1 %),
  3.0 -> 3.66 (+22 %). Motion length true -> estimated: 7 -> 8 (+14 %), 11 -> 10 (-9 %),
  15 -> 14 (-7 %), 21 -> 20 (-5 %). These are medians on synthetic data, not a guarantee per patch.

## E003: Richardson-Lucy deconvolution with known kernels (top half of page)
- Defocus sigma 2.0, correct kernel, 25 iterations: region PSNR 16.46 -> 18.48 dB.
- Horizontal motion L = 15, correct kernel, 25 iterations: 15.29 -> 17.36 dB.
- Wrong direction (vertical kernel on horizontal blur): 15.29 -> 11.90 dB (worse).
- **Bug found:** even kernel length (14 or 16) made RL diverge (0.83 dB, pixel values down to
  -255, black smear in the output). Fixed by forcing odd kernel sizes (`motion_psf`).
- Lesson: deblurring gives modest gains at best, and a wrong kernel is destructive. This is
  why the classifier must be right about blur type and why a majority vote was added.

## E004: false "Blur" flags on a clean page
- 4 of 352 patches of a perfectly clean page were flagged blurred. They held only 2-4 % ink
  (a few stray strokes) versus ~12 % for real text lines, so their gradient statistics were unreliable.
- Fix: `text_min_frac` 0.02 -> 0.06, plus a 5 x 5 majority vote for blur type.
- Result: clean page -> 0 flagged patches, output identical to input (PSNR 66 to 67 dB).

## E005: noise inflating the illumination estimate
- Grayscale closing takes local maxima, so noise pushed the estimated page brightness to 255
  instead of 246. Fix: Gaussian smoothing (sigma 2) before the closing.

## E006: final evaluation (3 seeds per scenario, 1000 x 1400 synthetic pages, Tesseract OCR)
Full per-image data: `results/metrics/evaluation.csv`. Figures: `results/figures/`.
Comparison methods: *degraded* (no processing), *global_filter* (shadow normalisation +
non-local-means + unsharp mask on EVERY pixel), *adaptive* (this project).

| scenario | method | PSNR (dB) | SSIM | MAE | time (s) | CER | WER |
|---|---|---|---|---|---|---|---|
| clean | degraded | inf | 1.0000 | 0.00 | 0.00 | 0.000 | 0.002 |
| clean | global_filter | 25.75 | 0.9680 | 6.14 | 0.75 | 0.000 | 0.001 |
| clean | adaptive | 67.30 | 1.0000 | 0.01 | 0.28 | 0.000 | 0.002 |
| shadow | degraded | 14.21 | 0.9450 | 23.52 | 0.00 | 0.247 | 0.270 |
| shadow | global_filter | 25.61 | 0.9670 | 6.66 | 0.66 | 0.000 | 0.001 |
| shadow | adaptive | 37.36 | 0.9982 | 1.40 | 0.29 | 0.000 | 0.002 |
| defocus | degraded | 20.37 | 0.9060 | 8.19 | 0.00 | 0.001 | 0.007 |
| defocus | global_filter | 20.68 | 0.9099 | 10.20 | 0.60 | 0.001 | 0.006 |
| defocus | adaptive | 21.94 | 0.9248 | 6.72 | 0.83 | 0.001 | 0.004 |
| motion | degraded | 18.85 | 0.9017 | 8.75 | 0.00 | 0.332 | 0.455 |
| motion | global_filter | 18.17 | 0.8811 | 12.04 | 0.60 | 0.385 | 0.464 |
| motion | adaptive | 20.40 | 0.9239 | 7.36 | 0.82 | 0.259 | 0.408 |
| shadow_blur | degraded | 13.18 | 0.8533 | 29.39 | 0.00 | 0.273 | 0.276 |
| shadow_blur | global_filter | 20.35 | 0.9029 | 11.01 | 0.67 | 0.002 | 0.011 |
| shadow_blur | adaptive | 21.46 | 0.9173 | 8.04 | 0.90 | 0.001 | 0.005 |
| noise | degraded | 31.08 | 0.8016 | 3.98 | 0.00 | 0.000 | 0.002 |
| noise | global_filter | 25.35 | 0.8136 | 8.01 | 0.62 | 0.000 | 0.002 |
| noise | adaptive | 31.92 | 0.8429 | 3.42 | 0.89 | 0.000 | 0.002 |
| combined | degraded | 13.56 | 0.6835 | 32.67 | 0.00 | 0.256 | 0.280 |
| combined | global_filter | 20.55 | 0.7614 | 12.22 | 0.69 | 0.001 | 0.005 |
| combined | adaptive | 21.43 | 0.7810 | 10.65 | 1.37 | 0.001 | 0.004 |

### How to read E006 honestly
- **Where adaptive clearly wins:** clean pages (untouched, 67 dB vs 26 dB for the global
  filter, which damages good areas), shadows (37 vs 26 dB), and noise (the global filter
  *lowers* PSNR below the input; adaptive does not).
- **Where the gain is small:** defocus, shadow + blur, combined. Classical deblurring is
  limited; adaptive beats global by roughly 1 to 1.5 dB, not a dramatic margin.
- **Where OCR does not show a difference:** Tesseract already reads sigma = 2 defocus almost
  perfectly (CER 0.001 before processing), and shadows are fixed equally well by the global
  filter. OCR improvement is clear only for shadow scenarios (CER 0.25 -> 0.00) and motion
  blur (0.33 -> 0.26, still poor).
- **Limits of this evidence:** synthetic pages with one font family each, white paper, regions
  with sharp rectangular boundaries, degradations generated by the same models the estimators
  assume. Real photos will be harder. Re-run this evaluation on your own paired photos
  before claiming anything in your paper.

## Template for your own experiments
| ID | Date | Phase | Image(s) | Method / parameters | Result | Saved at | Notes |
|---|---|---|---|---|---|---|---|
| E007 | | | | | | results/ | |
