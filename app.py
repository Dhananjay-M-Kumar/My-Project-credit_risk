"""
Streamlit Web Application
=========================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This is the interactive UI for demonstrating the full end-to-end decision support system:
- Quantitative ML Default Risk Assessment (Logistic Regression)
- Conformal Prediction Confidence Calibration (90% prediction intervals on held-out calibration data)
- Multi-Agent Decision-Support Collaboration:
    * Agent 1: Qualitative Financial Analyst (Gemini API)
    * Agent 2: Deterministic Risk Assessment Agent
    * Agent 3: Review Routing Recommendation Agent
"""

import os
import joblib
import pandas as pd
import streamlit as st

# Import our existing modular components directly (no code duplication)
from agents.analysis_agent import analyze_applicant, EXPECTED_FEATURES
from agents.risk_agent import assess_risk
from agents.recommendation_agent import generate_recommendation
from conformal.conformal_calibration import (
    compute_calibration_quantile,
    calibrate_prediction,
)

# ---------------------------------------------------------
# Streamlit Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Confidence-Calibrated Multi-Agent AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, clean, and professional appearance
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.25rem;
        font-weight: 500;
        color: #475569;
        margin-bottom: 0.8rem;
    }
    .desc-text {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
    }
    .metric-box {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .risk-low {
        background-color: #ECFDF5;
        border: 1px solid #10B981;
        color: #065F46;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .risk-mod {
        background-color: #FFFBEB;
        border: 1px solid #F59E0B;
        color: #92400E;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .risk-high {
        background-color: #FEF2F2;
        border: 1px solid #EF4444;
        color: #991B1B;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.2rem;
        text-align: center;
    }
    .disclaimer-box {
        background-color: #F1F5F9;
        border-left: 4px solid #64748B;
        padding: 14px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #475569;
        margin-top: 25px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Resource Loaders with Streamlit Caching
# ---------------------------------------------------------
@st.cache_resource
def load_ml_model():
    """Loads the trained scikit-learn model pipeline."""
    model_path = os.path.join("models", "risk_model.joblib")
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please run 'python models/risk_model.py' first.")
        st.stop()
    return joblib.load(model_path)


@st.cache_data
def load_dataset():
    """Loads the credit dataset for applicant selection."""
    data_path = os.path.join("data", "cs-training.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "credit_data.csv")
    if not os.path.exists(data_path):
        st.error("Dataset not found. Please ensure data/cs-training.csv exists.")
        st.stop()
    
    df = pd.read_csv(data_path)
    unnamed = [c for c in df.columns if "Unnamed" in c or c.strip() == ""]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


@st.cache_data
def get_calibration_quantile():
    """Calculates the 90th percentile calibration score from the held-out dataset."""
    model_path = os.path.join("models", "risk_model.joblib")
    calib_path = os.path.join("data", "calibration_data.csv")
    return compute_calibration_quantile(
        model_path=model_path,
        calib_path=calib_path,
        confidence_level=0.90
    )


# ---------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------
def main():
    # Load backend resources
    model = load_ml_model()
    df_data = load_dataset()
    quantile_score = get_calibration_quantile()

    # ---------------------------------------------------------
    # 1. Title & Header
    # ---------------------------------------------------------
    st.markdown('<div class="main-title">Confidence-Calibrated Multi-Agent AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">High-Stakes Decision Support</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="desc-text">AI-assisted credit risk analysis combining machine learning, '
        'qualitative AI analysis, risk assessment, and confidence calibration.</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # ---------------------------------------------------------
    # Sidebar: Applicant Selection & Controls
    # ---------------------------------------------------------
    with st.sidebar:
        st.header("👤 Applicant Selection")
        st.write("Select an applicant from the dataset to analyze (Applicant 1 to 10):")

        applicant_options = [f"Applicant #{i+1}" for i in range(10)]
        selected_option = st.selectbox("Choose Applicant:", applicant_options, index=0)
        selected_index = int(selected_option.split("#")[1]) - 1

        st.markdown("---")
        st.write("### 🚀 Execution Trigger")
        st.write("Click below to perform end-to-end multi-agent evaluation:")
        run_button = st.button("▶ Run Risk Analysis", type="primary", use_container_width=True)

        st.markdown("---")
        st.caption("ℹ️ **Note:** Gemini API calls are only made when you press the button above.")

    # ---------------------------------------------------------
    # Session State Initialization for Results Caching
    # ---------------------------------------------------------
    if "analysis_cache" not in st.session_state:
        st.session_state["analysis_cache"] = {}

    # Extract selected applicant record (10 feature columns only; Target SeriousDlqin2yrs is EXCLUDED)
    selected_row = df_data.iloc[selected_index]
    applicant_data = {feat: selected_row[feat] for feat in EXPECTED_FEATURES}
    applicant_df = pd.DataFrame([applicant_data])

    # ---------------------------------------------------------
    # 2. Section: Applicant Information
    # ---------------------------------------------------------
    st.subheader(f"📋 Applicant Information ({selected_option})")
    
    # Create clean formatted table for displaying the 10 features
    features_display = []
    for feat in EXPECTED_FEATURES:
        val = applicant_data[feat]
        if isinstance(val, float):
            formatted_val = f"{val:.4f}" if abs(val) < 10 else f"{val:,.2f}"
        else:
            formatted_val = str(val)
        features_display.append({"Feature Name": feat, "Value": formatted_val})
    
    df_display = pd.DataFrame(features_display)
    
    # Display in 2 clean columns
    col_feat_1, col_feat_2 = st.columns(2)
    with col_feat_1:
        st.dataframe(df_display.iloc[:5], use_container_width=True, hide_index=True)
    with col_feat_2:
        st.dataframe(df_display.iloc[5:], use_container_width=True, hide_index=True)

    st.markdown("---")

    # ---------------------------------------------------------
    # Trigger or Retrieve Pipeline Execution
    # ---------------------------------------------------------
    # Check if analysis has been run for this applicant
    cache_key = f"applicant_{selected_index}"

    if run_button:
        with st.spinner("Invoking Gemini Analysis Agent and synthesizing multi-agent decision support..."):
            # Step 1: Calculate raw default probability using ML model
            raw_prob = float(model.predict_proba(applicant_df)[0, 1])

            # Step 2: Conformal prediction calibration at 90% confidence level
            calib_res = calibrate_prediction(
                default_probability=raw_prob,
                quantile_score=quantile_score,
                confidence_level=0.90
            )

            # Step 3: Run Gemini Qualitative Analysis Agent (1 API Call)
            analysis_text = analyze_applicant(applicant_data)

            # Step 4: Run Risk Assessment Agent
            risk_res = assess_risk(default_probability=raw_prob, analysis=analysis_text)

            # Step 5: Run Recommendation Agent
            rec_res = generate_recommendation(
                default_probability=raw_prob,
                risk_level=risk_res["risk_level"],
                analysis=analysis_text
            )

            # Cache the complete result in session state
            st.session_state["analysis_cache"][cache_key] = {
                "raw_probability": raw_prob,
                "calibration": calib_res,
                "analysis": analysis_text,
                "risk": risk_res,
                "recommendation": rec_res
            }

    # ---------------------------------------------------------
    # Render Analysis Results if Available
    # ---------------------------------------------------------
    if cache_key in st.session_state["analysis_cache"]:
        results = st.session_state["analysis_cache"][cache_key]
        raw_prob = results["raw_probability"]
        calib_res = results["calibration"]
        analysis_text = results["analysis"]
        risk_res = results["risk"]
        rec_res = results["recommendation"]

        # ---------------------------------------------------------
        # 3 & 4. Quantitative ML Assessment & Confidence Calibration
        # ---------------------------------------------------------
        col_ml, col_calib = st.columns(2)

        with col_ml:
            st.subheader("📊 Quantitative ML Assessment")
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size: 0.9rem; color: #64748B; font-weight: 500;">ESTIMATED DEFAULT PROBABILITY</div>
                <div style="font-size: 2.2rem; font-weight: 700; color: #1E293B; margin: 8px 0;">
                    {raw_prob * 100:.2f}%
                </div>
                <div style="font-size: 0.85rem; color: #64748B;">
                    Generated by Scikit-Learn Logistic Regression (<code>predict_proba</code>)
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_calib:
            st.subheader("🎯 Confidence Calibration")
            st.markdown(f"""
            <div class="metric-box" style="border-left: 4px solid #3B82F6;">
                <div style="font-size: 0.9rem; color: #1E40AF; font-weight: 600;">CALIBRATION LEVEL: 90%</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #1D4ED8; margin: 8px 0;">
                    {calib_res['lower_bound'] * 100:.2f}% — {calib_res['upper_bound'] * 100:.2f}%
                </div>
                <div style="font-size: 0.83rem; color: #475569;">
                    The interval represents uncertainty estimated using the held-out calibration dataset (30,000 samples).
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 5. Agent 1 — AI Financial Analysis (Gemini)
        # ---------------------------------------------------------
        st.subheader("🤖 Agent 1 — AI Financial Analysis (Gemini)")
        st.caption("Qualitative decision-support analysis generated by Google Gemini API.")
        
        with st.expander("🔍 View Detailed Qualitative Analysis", expanded=True):
            st.markdown(analysis_text)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 6 & 7. Agent 2 (Risk Assessment) & Agent 3 (Recommendation)
        # ---------------------------------------------------------
        col_risk, col_rec = st.columns(2)

        with col_risk:
            st.subheader("🛡️ Agent 2 — Risk Assessment")
            risk_lvl = risk_res["risk_level"]
            if risk_lvl == "LOW RISK":
                css_class = "risk-low"
            elif risk_lvl == "MODERATE RISK":
                css_class = "risk-mod"
            else:
                css_class = "risk-high"

            st.markdown(f'<div class="{css_class}">{risk_lvl}</div>', unsafe_allow_html=True)
            st.caption("Deterministic threshold mapping from calibrated default probability.")

        with col_rec:
            st.subheader("📌 Agent 3 — Recommendation")
            st.markdown(f"""
            <div class="card">
                <div style="font-size: 1.2rem; font-weight: 700; color: #0F172A; margin-bottom: 6px;">
                    {rec_res['recommendation']}
                </div>
                <div style="font-size: 0.92rem; color: #475569;">
                    <strong>Reason:</strong> {rec_res['reason']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # 8. Final Decision-Support Summary
        # ---------------------------------------------------------
        st.subheader("📑 Final Decision-Support Summary")
        st.markdown(f"""
        <div class="card" style="background-color: #F8FAFC; border: 2px solid #CBD5E1; padding: 20px;">
            <table style="width: 100%; border-collapse: collapse; font-size: 1rem;">
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600; color: #334155; width: 45%;">Default Probability:</td>
                    <td style="font-weight: 700; color: #0F172A;">{raw_prob * 100:.2f}%</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600; color: #334155;">90% Calibrated Prediction Interval:</td>
                    <td style="font-weight: 700; color: #1D4ED8;">{calib_res['lower_bound'] * 100:.2f}% — {calib_res['upper_bound'] * 100:.2f}%</td>
                </tr>
                <tr style="border-bottom: 1px solid #E2E8F0; height: 35px;">
                    <td style="font-weight: 600; color: #334155;">Risk Level:</td>
                    <td style="font-weight: 700;">{risk_res['risk_level']}</td>
                </tr>
                <tr style="height: 35px;">
                    <td style="font-weight: 600; color: #334155;">Decision-Support Recommendation:</td>
                    <td style="font-weight: 700; color: #0F172A;">{rec_res['recommendation']}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Shown when the user hasn't clicked "Run Risk Analysis" yet for this applicant
        st.info("👈 Click **'▶ Run Risk Analysis'** in the sidebar to execute the full multi-agent pipeline and generate decision support.")

    # ---------------------------------------------------------
    # 9. Disclaimer
    # ---------------------------------------------------------
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Decision-Support Disclaimer:</strong><br>
        This system provides AI-assisted decision support only.
        It does not make automated loan approval or rejection decisions.
        Final decisions require qualified human review.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
