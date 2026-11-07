"""All tunable thresholds in one place.

Every number here is a starting value that was calibrated on SYNTHETIC
documents (see tools/make_synthetic.py). Real phone photos can differ, so
re-check these on your own images and log changes in docs/experiment_log.md.
"""
from dataclasses import dataclass


@dataclass
class Config:
    # ---- general ----
    max_side: int = 1600          # longest image side used for processing
    patch: int = 64               # region (patch) size in pixels

    # ---- illumination / shadow ----
    shadow_ratio_thr: float = 0.85   # local background / reference below this => shadow
    shadow_min_frac: float = 0.25    # fraction of dark-background pixels needed to flag a patch

    # ---- "does this patch contain text?" ----
    text_min_contrast: float = 60.0  # (p99 - p1) intensity range needed
    text_min_frac: float = 0.06      # fraction of clearly dark pixels needed (sparse patches are unreliable)

    # ---- blur ----
    blur_sharp_thr: float = 0.50     # min(sx, sy) below this => blurred (crisp text ~0.66)
    sharp_ref_default: float = 0.72  # overall sharpness of crisp text (used if no crisp patch)
    smin_ref_default: float = 0.66   # min(sx, sy) of crisp text (used if no crisp patch)
    motion_delta: float = 0.5        # |change in log(sx/sy)| that separates motion from defocus
    sigma0: float = 1.0              # intrinsic edge softness (pixels) in the defocus model
    len_k: float = 3.0               # scale in the motion-length estimate

    # ---- noise ----
    noise_sigma_thr: float = 4.0     # estimated noise std (grey levels) above this => noisy

    # ---- restoration ----
    feather: float = 16.0            # Gaussian sigma used to blend region borders
    rl_iterations: int = 12          # Richardson-Lucy iterations
    max_sigma: float = 4.0           # cap on deblur strength
