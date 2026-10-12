"""
Main End-to-End Decision Support Pipeline
=========================================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This is the primary entry point connecting all completed project components:
1. Baseline ML Model (Logistic Regression Pipeline)
2. Confidence Calibration Layer (Conformal Prediction on Held-Out Calibration Set)
3. Agent 1: Qualitative Financial Analysis Agent (Google Gemini SDK)
4. Agent 2: Quantitative Risk Assessment Agent (Rule-Based Risk Tiering)
5. Agent 3: Decision-Support Recommendation Agent (Workflow Routing)
"""

import os
import joblib
import pandas as pd
from agents.analysis_agent import analyze_applicant, EXPECTED_FEATURES
from agents.risk_agent import assess_risk
from agents.recommendation_agent import generate_recommendation
from conformal.conformal_calibration import (
    compute_calibration_quantile,
    calibrate_prediction,
)


def run_pipeline():
    # ---------------------------------------------------------
    # Step 1 & 2: Load Dataset and Select Applicant #1
    # ---------------------------------------------------------
    data_path = os.path.join("data", "cs-training.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "credit_data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found at '{data_path}'.")

    df = pd.read_csv(data_path)

    # Clean index column if present
    unnamed_cols = [c for c in df.columns if "Unnamed" in c or c.strip() == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    first_row = df.iloc[0]

    # ---------------------------------------------------------
    # Step 3: Extract the 10 Feature Columns (Exclude Target)
    # ---------------------------------------------------------
    applicant_data = {feat: first_row[feat] for feat in EXPECTED_FEATURES}
    applicant_df = pd.DataFrame([applicant_data])

    # ---------------------------------------------------------
    # Step 4 & 5: Load ML Model & Calculate Raw Default Probability
    # ---------------------------------------------------------
    model_path = os.path.join("models", "risk_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python models/risk_model.py' first."
        )

    model = joblib.load(model_path)
    # class 1 = default in 2 years
    raw_default_prob = float(model.predict_proba(applicant_df)[0, 1])

    # ---------------------------------------------------------
    # Step 6: Run Agent 1 (Qualitative Financial Analysis via Gemini)
    # ---------------------------------------------------------
    # SeriousDlqin2yrs is never included in applicant_data
    analysis_text = analyze_applicant(applicant_data)

    # ---------------------------------------------------------
    # Step 7, 8 & 9: Conformal Prediction / Confidence Calibration
    # ---------------------------------------------------------
    calib_data_path = os.path.join("data", "calibration_data.csv")
    quantile_score = compute_calibration_quantile(
        model_path=model_path,
        calib_path=calib_data_path,
        confidence_level=0.90
    )

    calibration_result = calibrate_prediction(
        default_probability=raw_default_prob,
        quantile_score=quantile_score,
        confidence_level=0.90
    )

    # ---------------------------------------------------------
    # Step 10: Run Agent 2 (Risk Assessment)
    # ---------------------------------------------------------
    risk_result = assess_risk(
        default_probability=raw_default_prob,
        analysis=analysis_text
    )

    # ---------------------------------------------------------
    # Step 11: Run Agent 3 (Decision-Support Recommendation)
    # ---------------------------------------------------------
    recommendation_result = generate_recommendation(
        default_probability=raw_default_prob,
        risk_level=risk_result["risk_level"],
        analysis=analysis_text
    )

    # ---------------------------------------------------------
    # Step 12: Print Final Unified Decision-Support Report
    # ---------------------------------------------------------
    print("=" * 60)
    print("CONFIDENCE-CALIBRATED MULTI-AGENT AI")
    print("DECISION-SUPPORT SYSTEM")
    print("=" * 60)
    print()

    print("APPLICANT PROFILE")
    print("-" * 60)
    for feat in EXPECTED_FEATURES:
        val = applicant_data[feat]
        if isinstance(val, float):
            val_str = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
        else:
            val_str = str(val)
        print(f"  • {feat:36}: {val_str}")
    print("-" * 60)
    print()

    print("QUANTITATIVE ML ASSESSMENT")
    print("-" * 60)
    print("Default Probability:")
    print(f"{raw_default_prob * 100:.2f}% ({raw_default_prob:.4f})")
    print("-" * 60)
    print()

    print("CONFIDENCE CALIBRATION")
    print("-" * 60)
    print("Calibration Confidence Level:")
    print(f"{int(calibration_result['confidence_level'] * 100)}%")
    print()
    print("Calibrated Prediction Interval:")
    print(f"{calibration_result['lower_bound'] * 100:.2f}% - {calibration_result['upper_bound'] * 100:.2f}%")
    print(f"[{calibration_result['lower_bound']:.4f}, {calibration_result['upper_bound']:.4f}]")
    print("-" * 60)
    print()

    print("AGENT 1 — AI FINANCIAL ANALYSIS")
    print("-" * 60)
    print(analysis_text.strip())
    print("-" * 60)
    print()

    print("AGENT 2 — RISK ASSESSMENT")
    print("-" * 60)
    print("Risk Level:")
    print(risk_result["risk_level"])
    print("-" * 60)
    print()

    print("AGENT 3 — DECISION-SUPPORT RECOMMENDATION")
    print("-" * 60)
    print("Recommendation:")
    print(recommendation_result["recommendation"])
    print()
    print("Reason:")
    print(recommendation_result["reason"])
    print("-" * 60)
    print()

    print("FINAL DECISION-SUPPORT SUMMARY")
    print("-" * 60)
    print("Default Probability:")
    print(f"{raw_default_prob * 100:.2f}%")
    print()
    print("90% Calibrated Prediction Interval:")
    print(f"{calibration_result['lower_bound'] * 100:.2f}% - {calibration_result['upper_bound'] * 100:.2f}%")
    print()
    print("Risk Level:")
    print(risk_result["risk_level"])
    print()
    print("Recommended Review:")
    print(recommendation_result["recommendation"])
    print("-" * 60)
    print()

    print("=" * 60)
    print("DISCLAIMER:")
    print("This system provides AI-assisted decision support only.")
    print("It does not make automated loan approval or rejection decisions.")
    print("Final decisions require qualified human review.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
