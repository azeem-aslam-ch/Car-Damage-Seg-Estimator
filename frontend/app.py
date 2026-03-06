"""Streamlit frontend for Car Damage Segmentation Estimator."""

import io
import os
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from PIL import Image

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚗 Car Damage Estimator",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #e0e0e0; }

    /* Cards */
    .metric-card {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-card h3 { margin: 0; font-size: 2rem; font-weight: 700; }
    .metric-card p  { margin: 4px 0 0; font-size: 0.8rem; opacity: 0.7; }

    /* Severity badges */
    .sev-low    { background: #27ae60; color: white; border-radius: 6px; padding: 2px 10px; font-weight: 600; }
    .sev-medium { background: #f39c12; color: white; border-radius: 6px; padding: 2px 10px; font-weight: 600; }
    .sev-high   { background: #e74c3c; color: white; border-radius: 6px; padding: 2px 10px; font-weight: 600; }
    .sev-none   { background: #7f8c8d; color: white; border-radius: 6px; padding: 2px 10px; font-weight: 600; }

    /* Header */
    .hero-title { font-size: 2.4rem; font-weight: 700; background: linear-gradient(90deg, #e94560, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-sub   { opacity: 0.7; font-size: 1rem; margin-top: -8px; }

    /* Sidebar */
    [data-testid="stSidebar"] { background: rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.1); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Header ──────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<div class="hero-title">🚗 Car Damage Estimator</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Segmentation-based damage detection & Cost estimation in PKR · Developed by Azeem Aslam</div>', unsafe_allow_html=True)

with col_status:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3)
        if resp.ok and resp.json().get("model_loaded"):
            st.success("✅ API Online")
        else:
            st.warning("⚠️ API Warming Up")
    except Exception:
        st.error("❌ API Offline")

st.divider()

# ─── Sidebar: User Inputs ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Assessment Settings")
    car_model_input = st.text_input("Car Model (optional)", placeholder="e.g. Toyota Corolla 2022")
    panel_location = st.selectbox(
        "Panel Location",
        ["unknown", "bumper", "door", "fender", "hood", "roof"],
        index=0,
    )
    notes = st.text_area("Additional Notes", placeholder="Inspector remarks, VIN, etc.")
    st.markdown("---")
    st.markdown("**Currency:** PKR 🇵🇰")
    st.markdown("**Model:** YOLOv8-seg")
    st.markdown("**Classes:** dent · scratch · peel_paint · broken")

# ─── Main: Upload ────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a car image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload a clear photo of the damaged car.",
)

if uploaded_file is not None:
    col_orig, col_overlay = st.columns(2, gap="medium")

    with col_orig:
        st.markdown("#### 📷 Original Image")
        orig_img = Image.open(uploaded_file)
        st.image(orig_img, use_column_width="auto")

    # ─── Run Prediction ───────────────────────────────────────────────────────
    predict_btn = st.button("🔍 Analyse Damage", type="primary", use_container_width=True)

    if predict_btn or "last_prediction" in st.session_state:
        if predict_btn:
            with st.spinner("Running segmentation inference…"):
                uploaded_file.seek(0)
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/api/v1/predict",
                        files={"image": (uploaded_file.name, uploaded_file, uploaded_file.type)},
                        data={"panel_location": panel_location},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    prediction = resp.json()
                    st.session_state["last_prediction"] = prediction
                    st.session_state["last_car_model"] = car_model_input
                    st.session_state["last_location"] = panel_location
                    st.session_state["last_notes"] = notes
                except requests.exceptions.RequestException as exc:
                    st.error(f"❌ API Error: {exc}")
                    st.stop()

        prediction = st.session_state.get("last_prediction", {})

        # ── Overlay image ───────────────────────────────────────────────────────
        with col_overlay:
            st.markdown("#### 🎯 Damage Overlay")
            overlay_path = prediction.get("artifacts", {}).get("overlay_image_path")
            if overlay_path:
                # Try fetching from backend static endpoint
                filename = os.path.basename(overlay_path)
                img_url = f"{BACKEND_URL}/outputs/{filename}"
                try:
                    img_resp = requests.get(img_url, timeout=10)
                    if img_resp.ok:
                        overlay_img = Image.open(io.BytesIO(img_resp.content))
                        st.image(overlay_img, use_column_width="auto")
                    else:
                        st.info("Overlay file not accessible via URL.")
                except Exception:
                    st.info("Overlay not available in this environment.")
            else:
                st.info("No overlay generated (no damages detected or inference error).")

        st.divider()

        # ── Summary metrics ─────────────────────────────────────────────────────
        damage_detected = prediction.get("damage_detected", False)
        overall_sev = prediction.get("overall_severity", "none")
        sev_class = f"sev-{overall_sev}"
        summary = prediction.get("summary", {})
        cost = summary.get("estimated_cost_pkr", {})

        if not damage_detected:
            st.success("✅ No significant damage detected in this image.")
        else:
            st.markdown("### 📊 Analysis Summary")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'<div class="metric-card"><h3>{summary.get("total_instances", 0)}</h3><p>Damage Instances</p></div>',
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f'<div class="metric-card"><h3>{summary.get("total_damage_percent", 0):.2f}%</h3><p>Total Area Affected</p></div>',
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f'<div class="metric-card"><h3><span class="{sev_class}">{overall_sev.upper()}</span></h3><p>Overall Severity</p></div>',
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f'<div class="metric-card"><h3>PKR {cost.get("min", 0):,.0f}–{cost.get("max", 0):,.0f}</h3><p>Estimated Repair Cost</p></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # ── Detections table ────────────────────────────────────────────────
            st.markdown("### 🔬 Detection Details")
            detections = prediction.get("detections", [])
            if detections:
                rows = []
                for d in detections:
                    c = d.get("cost_pkr", {})
                    rows.append(
                        {
                            "Class": d.get("class", d.get("class_name", "?")),
                            "Confidence": f"{d.get('confidence', 0):.2f}",
                            "Mask Area (px)": d.get("mask_area_px", 0),
                            "Damage %": f"{d.get('damage_area_percent', 0):.3f}%",
                            "Severity": d.get("severity", "?").upper(),
                            "Cost Min (PKR)": f"{c.get('min', 0):,.0f}",
                            "Cost Max (PKR)": f"{c.get('max', 0):,.0f}",
                        }
                    )
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

        # ── PDF Export ──────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📄 Export Report")
        export_col, _ = st.columns([1, 3])
        with export_col:
            if st.button("📥 Download PDF Report", use_container_width=True):
                with st.spinner("Generating PDF…"):
                    try:
                        report_payload = {
                            "prediction": prediction,
                            "car_model": st.session_state.get("last_car_model") or None,
                            "panel_location": st.session_state.get("last_location", "unknown"),
                            "notes": st.session_state.get("last_notes") or None,
                            "currency": "PKR",
                        }
                        pdf_resp = requests.post(
                            f"{BACKEND_URL}/api/v1/report",
                            json=report_payload,
                            timeout=60,
                        )
                        pdf_resp.raise_for_status()
                        st.download_button(
                            label="⬇️ Save Report PDF",
                            data=pdf_resp.content,
                            file_name="car_damage_report.pdf",
                            mime="application/pdf",
                        )
                        st.success("Report ready!")
                    except Exception as exc:
                        st.error(f"❌ Could not generate PDF: {exc}")

else:
    # Landing placeholder
    st.markdown(
        """
        <div style="text-align:center;padding:60px 20px;opacity:0.5;">
            <h2>📤 Upload a car image to begin</h2>
            <p>Supports JPG, PNG, WEBP. The system will detect dents, scratches, paint peeling, and broken parts.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
