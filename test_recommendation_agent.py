"""
Test Script for Recommendation Agent
====================================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This script executes the complete integrated pipeline:
1. Loads the trained ML model pipeline (Phase 1).
2. Loads data/cs-training.csv and selects Applicant #1 (row 0).
3. Computes the default probability with model.predict_proba().
4. Executes the Gemini Analysis Agent (Phase 2).
5. Executes the Risk Assessment Agent (Phase 3).
6. Executes the Recommendation Agent (Phase 4).
7. Prints a unified, comprehensive Decision-Support Report.
"""

import os
import joblib
import pandas as pd
from agents.analysis_agent import analyze_applicant, EXPECTED_FEATURES
from agents.risk_agent import assess_risk
from agents.recommendation_agent import generate_recommendation


def main():
    print("==================================================================")
    print("   Testing Phase 4 Recommendation Agent (Full Multi-Agent Flow)   ")
    print("==================================================================\n")

    # 1. Load trained ML Model Pipeline from Phase 1
    model_path = os.path.join("models", "risk_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Please run 'python models/risk_model.py' first."
        )
    print(f"[1/6] Loading trained ML model from: {model_path}")
    model = joblib.load(model_path)

    # 2. Load Dataset
    data_path = os.path.join("data", "cs-training.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "credit_data.csv")

    print(f"[2/6] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Remove index column if present
    unnamed_cols = [c for c in df.columns if "Unnamed" in c or c.strip() == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # 3. Select Applicant #1 (row 0)
    print(f"[3/6] Selecting Applicant #1 (row 0)...")
    first_row = df.iloc[0]

    # Convert the 10 features to a dictionary and DataFrame (SeriousDlqin2yrs excluded)
    applicant_data = {feat: first_row[feat] for feat in EXPECTED_FEATURES}
    applicant_df = pd.DataFrame([applicant_data])

    # 4. Generate quantitative default probability using ML model (Phase 1)
    default_prob = float(model.predict_proba(applicant_df)[0, 1])
    print(f"      Calculated ML Default Probability: {default_prob:.4f} ({default_prob * 100:.2f}%)")

    # 5. Run Qualitative Analysis Agent (Phase 2)
    print(f"[4/6] Running Qualitative Analysis Agent (Gemini API)...")
    analysis_text = analyze_applicant(applicant_data)

    # 6. Run Risk Assessment Agent (Phase 3)
    print(f"[5/6] Running Risk Assessment Agent...")
    risk_assessment = assess_risk(default_prob, analysis_text)

    # 7. Run Recommendation Agent (Phase 4)
    print(f"[6/6] Generating Decision-Support Recommendation...")
    recommendation_result = generate_recommendation(
        default_probability=risk_assessment["default_probability"],
        risk_level=risk_assessment["risk_level"],
        analysis=risk_assessment["analysis"]
    )

    # 8. Print Complete Decision-Support Report
    print("\n" + "=" * 65)
    print("          COMPREHENSIVE DECISION-SUPPORT REPORT                  ")
    print("=" * 65)

    print("\n[APPLICANT FINANCIAL PROFILE]")
    print("-" * 50)
    for feat, val in applicant_data.items():
        if isinstance(val, float):
            val_str = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
        else:
            val_str = str(val)
        print(f"  • {feat:36}: {val_str}")
    print("-" * 50)

    print("\n[QUANTITATIVE RISK & RECOMMENDATION]")
    print(f"  • Default Probability : {risk_assessment['default_probability_percent']} ({risk_assessment['default_probability']})")
    print(f"  • Risk Level          : {recommendation_result['risk_level']}")
    print(f"  • Recommendation      : {recommendation_result['recommendation']}")
    print(f"  • Reason              : {recommendation_result['reason']}")

    print("\n[QUALITATIVE ANALYSIS REPORT]")
    print("-" * 65)
    print(risk_assessment['analysis'])
    print("-" * 65)
    print("=" * 65)

    print("\n[SUCCESS] Recommendation Agent test passed successfully!")


if __name__ == "__main__":
    main()
