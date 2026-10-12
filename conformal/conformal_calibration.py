"""
Conformal Prediction / Confidence Calibration Layer
===================================================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This module implements a lightweight prototype of a Conformal-Style
Confidence Calibration Layer for binary credit risk scoring.

Core Concepts:
1. Held-Out Calibration Set:
   - Uses 30,000 separate samples from 'data/calibration_data.csv' that were
     NEVER seen during model training.
2. Nonconformity Score:
   - Measures how "surprising" an outcome is relative to model confidence:
     * If True Class = 1 (Default):     score = 1 - P(default)
     * If True Class = 0 (Non-Default): score = 1 - P(non-default) = P(default)
3. Empirical 90% Quantile:
   - The 90th percentile of calibration scores represents the threshold error
     margin capturing 90% of calibration instances.
4. Calibrated Prediction Interval:
   - Converts point probability into a bounded interval [lower_bound, upper_bound]
     at a 90% confidence level.

Disclaimer:
This module provides an empirical prototype conformal-style calibration layer
for academic decision-support demonstration.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Constants & Default Paths
TARGET_COLUMN = "SeriousDlqin2yrs"
DEFAULT_MODEL_PATH = os.path.join("models", "risk_model.joblib")
DEFAULT_CALIB_PATH = os.path.join("data", "calibration_data.csv")


def compute_calibration_quantile(model_path=DEFAULT_MODEL_PATH, calib_path=DEFAULT_CALIB_PATH, confidence_level=0.90):
    """
    Computes the empirical nonconformity quantile score on the calibration dataset.

    Steps:
    1. Loads trained model and calibration dataset.
    2. Computes predicted probabilities for all calibration instances.
    3. Calculates nonconformity scores: 1 - P(true_class).
    4. Finds the (confidence_level * 100)th percentile (e.g. 90th percentile).

    Args:
        model_path (str): Path to trained model pipeline (.joblib).
        calib_path (str): Path to calibration CSV dataset.
        confidence_level (float): Target confidence level (default 0.90 for 90%).

    Returns:
        float: The 90th percentile nonconformity score (quantile_score).
    """
    # 1. Load trained model pipeline
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'.")
    model = joblib.load(model_path)

    # 2. Load calibration dataset
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration dataset not found at '{calib_path}'.")
    df_calib = pd.read_csv(calib_path)

    # Remove any index column if present
    unnamed_cols = [c for c in df_calib.columns if "Unnamed" in c or c.strip() == ""]
    if unnamed_cols:
        df_calib = df_calib.drop(columns=unnamed_cols)

    if TARGET_COLUMN not in df_calib.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in calibration data.")

    # 3. Separate features and true labels
    X_calib = df_calib.drop(columns=[TARGET_COLUMN])
    y_calib = df_calib[TARGET_COLUMN].values.astype(int)

    # 4. Generate predicted probabilities for calibration samples
    # probs[:, 1] = P(default), probs[:, 0] = P(non-default)
    probs = model.predict_proba(X_calib)
    p_default = probs[:, 1]
    p_non_default = probs[:, 0]

    # 5. Compute nonconformity score = 1 - P(true_class)
    # If y == 1: score = 1 - P(default)
    # If y == 0: score = 1 - P(non-default) = P(default)
    nonconformity_scores = np.where(y_calib == 1, 1.0 - p_default, 1.0 - p_non_default)

    # 6. Calculate the 90th percentile calibration score
    percentile_value = confidence_level * 100.0
    quantile_score = float(np.percentile(nonconformity_scores, percentile_value))

    return quantile_score


def calibrate_prediction(default_probability, quantile_score, confidence_level=0.90):
    """
    Calibrates a single predicted default probability by constructing an empirical
    90% prediction interval based on the calibration quantile score.

    Args:
        default_probability (float): Raw default probability from ML model.
        quantile_score (float): The 90th percentile nonconformity score from calibration set.
        confidence_level (float): Confidence level (default 0.90 for 90%).

    Returns:
        dict: Calibrated prediction output containing:
            - raw_probability (float)
            - lower_bound (float): Clamped to [0.0, 1.0]
            - upper_bound (float): Clamped to [0.0, 1.0]
            - confidence_level (float): 0.90
    """
    raw_prob = float(default_probability)
    margin = float(quantile_score)

    # Calculate calibrated lower and upper bounds using the uncertainty margin
    lower_bound = max(0.0, raw_prob - margin)
    upper_bound = min(1.0, raw_prob + margin)

    return {
        "raw_probability": round(raw_prob, 4),
        "lower_bound": round(lower_bound, 4),
        "upper_bound": round(upper_bound, 4),
        "confidence_level": confidence_level
    }
