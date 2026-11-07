"""Phase 10: web UI.   Run with:  streamlit run app/streamlit_app.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2          # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import Config                                  # noqa: E402
from src.metrics import no_reference_quality                   # noqa: E402
from src.ocr_engine import extract_text, ocr_available         # noqa: E402
from src.pipeline import restore_document                      # noqa: E402
from src.visualize import label_overlay                        # noqa: E402

st.set_page_config(page_title="Shadow-Aware Document Restoration", layout="wide")
st.title("Shadow-Aware Image Quality Restoration for Smartphone-Captured Text")
st.caption("Region-aware restoration: each part of the page gets only the fix it needs.")

with st.sidebar:
    st.header("Settings")
    detect = st.checkbox("Detect page and fix perspective", value=True)
    run_ocr = st.checkbox("Extract text (OCR)", value=True)
    st.subheader("Thresholds")
    cfg = Config()
    cfg.shadow_ratio_thr = st.slider("Shadow: brightness ratio below", 0.5, 0.98, cfg.shadow_ratio_thr, 0.01)
    cfg.blur_sharp_thr = st.slider("Blur: sharpness score below", 0.2, 0.65, cfg.blur_sharp_thr, 0.01)
    cfg.noise_sigma_thr = st.slider("Noise: sigma above", 1.0, 15.0, cfg.noise_sigma_thr, 0.5)
    if run_ocr and not ocr_available():
        st.warning("Tesseract is not installed, so OCR is disabled. See README.")
        run_ocr = False

up = st.file_uploader("Upload a document photo", type=["jpg", "jpeg", "png"])
if up is None:
    st.info("Upload a phone photo of a document to begin. Sample images are in data/synthetic/.")
    st.stop()

img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
if img is None:
    st.error("Could not read this file as an image.")
    st.stop()

with st.spinner("Analysing and restoring..."):
    res = restore_document(img, cfg, detect=detect)

rgb = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
c1, c2 = st.columns(2)
c1.subheader("Original (after page detection)")
c1.image(rgb(res.working), use_container_width=True)
c2.subheader("Restored")
c2.image(rgb(res.restored), use_container_width=True)

with st.expander("What the system decided for each region", expanded=True):
    st.image(rgb(label_overlay(res.working, res.analysis)), use_container_width=True)
    st.write("Blue = shadow, red = blur, magenta = shadow + blur, yellow = noise, uncoloured = left untouched.")
    st.json(res.info["region_counts"])

st.subheader("Image-quality comparison (no reference image, so only before vs after)")
qb, qa = no_reference_quality(res.working), no_reference_quality(res.restored)
st.table({"metric": list(qb.keys()),
          "before": [round(v, 4) for v in qb.values()],
          "after": [round(v, 4) for v in qa.values()]})
st.caption("Higher sharpness and contrast, and LOWER illumination non-uniformity, usually mean a cleaner page. "
           "These are indicators, not proof of better readability.")

if run_ocr:
    st.subheader("Extracted text")
    t1, t2 = st.columns(2)
    with st.spinner("Running OCR..."):
        before, after = extract_text(res.working), extract_text(res.restored)
    t1.markdown("**Before restoration**")
    t1.text_area("ocr_before", before, height=250, label_visibility="collapsed")
    t2.markdown("**After restoration**")
    t2.text_area("ocr_after", after, height=250, label_visibility="collapsed")
    st.caption(f"Characters recognised: before {len(before.strip())}, after {len(after.strip())}. "
               "Without the true text we cannot compute CER/WER here.")

st.subheader("Processing information")
st.json({k: v for k, v in res.info.items() if k != "region_counts"})

ok, png = cv2.imencode(".png", res.restored)
if ok:
    st.download_button("Download restored image", png.tobytes(), "restored.png", "image/png")
