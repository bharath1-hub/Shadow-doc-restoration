# Code Walkthrough (read this to understand and defend the project)

The system is a chain. Each module does one job and passes its output on.

```
photo -> document_detection -> region analysis -> adaptive restoration -> restored page
                               (illumination, blur, noise,   (denoise -> shadow -> deblur,
                                per 64x64 patch)               each only where flagged)
```
Entry point: `src/pipeline.py :: restore_document`.

## 1. document_detection.py  (Phase 3)
**Goal:** find the page and make it a flat rectangle.
1. Shrink the photo (speed), blur it slightly, make binary masks with Otsu thresholding
   (page brighter, then page darker) and Canny edges.
2. Take the largest *external* contour. A contour is a closed outline.
3. `cv2.convexHull` + `cv2.approxPolyDP` simplify it; if it has 4 corners, that is the page.
4. `order_points` sorts corners as top-left, top-right, bottom-right, bottom-left
   (top-left has the smallest x+y; bottom-right the largest; top-right has the smallest y-x).
5. `cv2.getPerspectiveTransform` + `warpPerspective` map the 4 corners to a rectangle.
**Fails when:** the page is the whole frame, the background is as bright as the page, or a corner is hidden.

## 2. illumination.py + shadow.py  (Phase 4)
**Model:** a pixel = paper reflectance x illumination, so illumination = pixel / reflectance.
**Trick:** text is dark and thin, so a grayscale *closing* (dilate then erode) with a kernel
wider than the strokes erases the text and leaves a map of the paper brightness: `bg`.
- `ratio = bg / ref`, where `ref` is the 95th percentile of `bg` ("good light").
  Normal areas have ratio about 1; shadows have ratio below 1.
- Shadow flag: ratio < 0.85 over a quarter of a patch.
- **Correction:** `corrected = pixel / bg * ref`. Paper in a shadow becomes as bright as lit paper,
  and the text's relative contrast is kept.
**Fails when:** large dark objects (photos, black bars) are mistaken for shadow.

## 3. blur_estimation.py  (Phase 5)
Sharp edges have steep gradients; blur flattens them. Using Sobel gradients on the
illumination-normalised patch:
`sx = p99(|dI/dx|) / (4C)`, `sy` likewise, with contrast reference `C = max(patch contrast, 0.7 x page brightness)`.
(An ideal step edge of contrast C gives Sobel response 4C, so crisp edges score about 1.)
- **Blur score = min(sx, sy).** Defocus lowers both; motion lowers one direction.
- **Type:** `log(sx/sy)` compared to crisp text of the same page: strongly negative means
  horizontal motion, strongly positive means vertical motion, near zero means defocus.
- **Defocus sigma:** a Gaussian-blurred edge has slope proportional to 1/sqrt(sigma0^2 + sigma^2),
  so `sigma = sigma0 * sqrt((s_ref/s)^2 - 1)`.
- **Motion length:** proportional to the sharpness lost along the blur direction.
These formulas were calibrated on synthetic blur (E002 in the experiment log).

## 4. noise_estimation.py  (Phase 5/6)
Use only flat pixels (no text edges). There, `pixel - smoothed pixel` is almost pure noise.
Its robust spread, MAD x 1.4826, estimates sigma; it is divided by a gain that accounts for
the noise variance the smoothing removed. Flag a patch as noisy if sigma > 4.

## 5. region_classifier.py  (Phase 6)
Cuts the page into 64 x 64 patches. Each patch gets three independent flags (shadow, blur, noise)
and a label such as `Normal`, `Shadow`, `Blur`, `Shadow+Blur`, `Noisy`.
A 5 x 5 majority vote stabilises the blur type. Patches without enough ink are never called blurred.

## 6. restoration.py + blending.py + pipeline.py  (Phase 6)
Order matters: **denoise -> shadow correction -> deblur** (deblurring amplifies noise, so
denoise first; blur should be measured and removed after lighting is even).
- Denoise: non-local means, strength from the measured sigma.
- Deblur: **Richardson-Lucy** deconvolution. Starting from the blurred image y and the kernel h,
  repeat `x <- x * ( (y / (x * h)) * h_flipped )`. Gaussian kernel for defocus, line kernel for motion.
  Kernels must have odd size (an even kernel diverges, E003).
- Each method is computed on the whole image, but `blend(out, processed, mask)` keeps it only
  inside the flagged patches. Masks are blurred (feathered) so borders do not show seams.
- Work happens on the L (lightness) channel of the Lab colour space, so colours are untouched.

## 7. metrics.py + ocr_engine.py  (Phases 8, 9)
| metric | meaning | limitation |
|---|---|---|
| MSE, MAE | average pixel error | ignores structure; a 1-pixel shift of sharp text is a big error |
| PSNR = 10 log10(255^2 / MSE) | error in dB, higher is better | does not match human or OCR judgement well |
| SSIM | similarity of local structure, 1 = identical | needs a clean reference; insensitive to some artefacts |
| CER = (S + D + I) / N | character edit distance / reference length | needs true text; depends on the OCR engine |
| WER | same on words | one wrong character spoils a whole word |
| time | seconds per image | machine dependent |
All of these except time need a reference. For real photos without a reference, only
`no_reference_quality` (sharpness, contrast, lighting evenness) can be used, and only as before/after indicators.

## Viva questions and answers
1. **Why not apply one filter to the whole image?** A global fix over-processes good areas
   (the global baseline scored 26 dB on a clean page; ours left it at 67 dB) and cannot fit
   different degradations at once.
2. **How do you find shadows?** Closing removes thin dark text, leaving the paper brightness;
   regions much darker than the well-lit reference are shadows.
3. **Why divide by the background to remove a shadow?** Pixel = reflectance x illumination, so dividing by illumination leaves reflectance.
4. **How do you tell blur from sharp?** Steepness of edges relative to contrast: blur flattens edges.
5. **Motion vs defocus?** Motion removes edge steepness in one direction only (sx and sy become unbalanced); defocus affects both equally.
6. **Why deblur last?** Lighting must be even first and noise removed, because deconvolution amplifies noise.
7. **What is Richardson-Lucy and why use it?** An iterative deconvolution assuming Poisson noise; it keeps pixel values non-negative and is simpler and more stable than inverse filtering.
8. **Why odd kernel sizes?** An even kernel has no centre pixel, so every iteration shifts the image and RL diverges (we observed this, E003).
9. **Why Lab / the L channel?** Brightness and blur live in L; colour stays unchanged.
10. **What is a Gaussian blur for?** Smoothing before gradients and noise estimation; it also models defocus.
11. **Why PSNR and SSIM both?** PSNR measures pixel error, SSIM structure; neither equals readability, so we add OCR CER/WER.
12. **Why is OCR improvement small on defocus?** Tesseract already reads mild blur well (CER 0.001 before processing).
13. **What are the limits of your results?** Synthetic pages only; classical deblurring gives modest gains; axis-aligned motion only; thresholds tuned on synthetic data.
14. **Why no deep learning?** Not justified yet; see docs/phase7_decision.md.
15. **How would you improve it?** Test on real paired photos, retune thresholds, add arbitrary-angle motion estimation, consider a learned module where classical methods fail.
