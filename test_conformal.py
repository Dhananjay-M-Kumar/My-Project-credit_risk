"""
Test Script for Conformal Prediction / Calibration Layer
========================================================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This script tests the Conformal Calibration module:
1. Loads the trained ML model pipeline (models/risk_model.joblib).
2. Loads data/calibration_data.csv (30,000 held-out samples).
3. Computes the nonconformity scores and the 90th percentile calibration quantile.
4. Selects Applicant #1 from data/cs-training.csv.
5. Computes the applicant's raw ML default probability.
6. Calibrates the prediction using calibrate_prediction().
7. Prints the 90% Calibrated Prediction Interval.
"""

import os
import joblib
import pandas as pd
from conformal.conformal_calibration import (
    compute_calibration_quantile,
    calibrate_prediction,
)
from agents.analysis_agent import EXPECTED_FEATURES


def main():
    # 1. Paths
    model_path = os.path.join("models", "risk_model.joblib")
    calib_path = os.path.join("data", "calibration_data.csv")
    training_data_path = os.path.join("data", "cs-training.csv")
    if not os.path.exists(training_data_path):
        training_data_path = os.path.join("data", "credit_data.csv")

    # 2. Load trained model
    model = joblib.load(model_path)

    # 3. Compute 90th percentile calibration score from calibration dataset
    quantile_score = compute_calibration_quantile(
        model_path=model_path,
        calib_path=calib_path,
        confidence_level=0.90
    )

    # 4. Load Applicant #1 from original dataset
    df = pd.read_csv(training_data_path)
    unnamed_cols = [c for c in df.columns if "Unnamed" in c or c.strip() == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    first_row = df.iloc[0]
    applicant_data = {feat: first_row[feat] for feat in EXPECTED_FEATURES}
    applicant_df = pd.DataFrame([applicant_data])

    # 5. Calculate applicant's raw default probability
    raw_default_prob = float(model.predict_proba(applicant_df)[0, 1])

    # 6. Perform Conformal Calibration at 90% confidence level
    calib_result = calibrate_prediction(
        default_probability=raw_default_prob,
        quantile_score=quantile_score,
        confidence_level=0.90
    )

    # 7. Print result in the requested format
    print("========================================")
    print("CONFORMAL CALIBRATION RESULT")
    print("========================================")
    print()
    print("Raw Default Probability:")
    print(f"{calib_result['raw_probability']:.4f} ({calib_result['raw_probability'] * 100:.2f}%)")
    print()
    print("Calibration Quantile:")
    print(f"{quantile_score:.4f} ({quantile_score * 100:.2f}%)")
    print()
    print("90% Calibrated Prediction Interval:")
    print(f"Lower Bound: {calib_result['lower_bound']:.4f} ({calib_result['lower_bound'] * 100:.2f}%)")
    print(f"Upper Bound: {calib_result['upper_bound']:.4f} ({calib_result['upper_bound'] * 100:.2f}%)")
    print()
    print("========================================")


if __name__ == "__main__":
    main()
