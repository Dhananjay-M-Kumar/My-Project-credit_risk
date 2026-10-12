"""
Test Script for Risk Assessment Agent
=====================================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This script integrates Phase 1 (ML Model), Phase 2 (Analysis Agent), and Phase 3 (Risk Agent):
1. Loads the trained ML model pipeline from 'models/risk_model.joblib'.
2. Loads 'data/cs-training.csv' using pandas.
3. Selects the first applicant (row index 0) with the 10 credit risk features.
4. Predicts default probability using model.predict_proba().
5. Generates qualitative analysis using the Gemini Analysis Agent (Phase 2).
6. Combines quantitative probability and qualitative analysis using assess_risk() (Phase 3).
7. Displays the complete integrated decision-support output.
"""

import os
import joblib
import pandas as pd
from agents.analysis_agent import analyze_applicant, EXPECTED_FEATURES
from agents.risk_agent import assess_risk


def main():
    print("==================================================================")
    print("      Testing Phase 3 Risk Agent (Integrated Multi-Agent Flow)    ")
    print("==================================================================\n")

    # 1. Load trained ML Model Pipeline from Phase 1
    model_path = os.path.join("models", "risk_model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Please run 'python models/risk_model.py' first."
        )
    print(f"[1/5] Loading trained ML model from: {model_path}")
    model = joblib.load(model_path)

    # 2. Load Dataset
    data_path = os.path.join("data", "cs-training.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "credit_data.csv")

    print(f"[2/5] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Clean index column if present
    unnamed_cols = [c for c in df.columns if "Unnamed" in c or c.strip() == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # 3. Extract the first applicant
    print(f"[3/5] Selecting Applicant #1 (row 0)...")
    first_row = df.iloc[0]

    # Convert the 10 features to a DataFrame for the model and dict for the agent
    # (SeriousDlqin2yrs is strictly excluded)
    applicant_data = {feat: first_row[feat] for feat in EXPECTED_FEATURES}
    applicant_df = pd.DataFrame([applicant_data])

    # 4. Generate quantitative default probability using ML model (Phase 1)
    # class 1 = default in 2 years
    default_prob = float(model.predict_proba(applicant_df)[0, 1])
    print(f"      Calculated ML Default Probability: {default_prob:.4f} ({default_prob * 100:.2f}%)")

    # 5. Generate qualitative analysis using Gemini Analysis Agent (Phase 2)
    print(f"[4/5] Running Qualitative Analysis Agent (Gemini API)...")
    analysis_text = analyze_applicant(applicant_data)

    # 6. Assess risk using Risk Agent (Phase 3)
    print(f"[5/5] Synthesizing final risk assessment...")
    assessment = assess_risk(default_prob, analysis_text)

    # 7. Print integrated decision-support results clearly
    print("\n" + "=" * 65)
    print("               CREDIT RISK DECISION SUPPORT REPORT               ")
    print("=" * 65)
    
    print("\n[APPLICANT FINANCIAL FEATURES]")
    print("-" * 50)
    for feat, val in applicant_data.items():
        if isinstance(val, float):
            val_str = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
        else:
            val_str = str(val)
        print(f"  • {feat:36}: {val_str}")
    print("-" * 50)

    print("\n[QUANTITATIVE RISK ESTIMATION]")
    print(f"  • Default Probability : {assessment['default_probability_percent']} ({assessment['default_probability']})")
    print(f"  • Risk Level          : {assessment['risk_level']}")

    print("\n[QUALITATIVE ANALYSIS AGENT REPORT]")
    print("-" * 65)
    print(assessment['analysis'])
    print("-" * 65)
    print("=" * 65)

    print("\n[SUCCESS] Risk Agent test passed successfully!")


if __name__ == "__main__":
    main()
