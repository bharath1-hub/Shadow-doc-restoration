# Phase 7 Decision: is deep learning justified yet?

**Decision (so far): no deep learning in this version. This is a deliberate choice, not an omission.**

## Why
- The classical pipeline already detects and corrects the degradations we can generate, with
  measured gains (docs/experiment_log.md, E006). Training a network before knowing where the
  classical method fails would be unjustified and would need data we do not have.
- A learned model needs many paired (degraded, clean) real document photos. We have none yet.
- Everything is explainable line by line, which matters for the viva.

## Where classical methods are expected to fail (verify on real photos first)
1. **Shadows with texture or coloured paper:** division by the background assumes a uniform
   paper colour. Photos, tables, coloured backgrounds break this.
2. **Motion blur at arbitrary angles:** we handle horizontal and vertical only.
3. **Non-uniform / spatially complex blur**, e.g. camera rotation or depth variation.
4. **Severe blur:** Richardson-Lucy gains are modest (E003) and amplify noise.
5. **Small or non-Latin text:** the sharpness references are calibrated on one text size.

## When to revisit
Run `tools/evaluate.py`-style experiments on your own real photos. If failure cases in
points 1 to 4 dominate, then (and only then) consider a small trained model for ONE module
(shadow detection or deblurring), train it on Google Colab, and compare it fairly against
the classical module using the same metrics. Candidate research directions to read about
(no specific papers are cited here because the literature survey is not done yet):
learned shadow removal for documents, blind/non-uniform deblurring networks, and learned
blur-map estimation. Find the actual papers in docs/literature_survey.md before citing any.
