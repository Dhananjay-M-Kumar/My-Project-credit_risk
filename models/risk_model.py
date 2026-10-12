"""
Credit Risk Baseline Machine Learning Model
===========================================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This module implements the core baseline Machine Learning pipeline:
1. Loads the Kaggle 'Give Me Some Credit' dataset (cs-training.csv).
2. Cleans and separates features from the target ('SeriousDlqin2yrs').
3. Performs a stratified 3-way split: Train (60%), Calibration (20%), and Test (20%).
4. Builds a scikit-learn Pipeline with:
   - SimpleImputer (median strategy to handle missing values without data leakage)
   - StandardScaler (normalizes features for Logistic Regression)
   - LogisticRegression (calibrated probability estimator for credit risk)
5. Generates default probabilities using predict_proba().
6. Evaluates performance using ROC-AUC, Precision, Recall, and F1 Score.
7. Saves the trained model pipeline and split datasets for downstream agents & conformal calibration.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
TARGET_COLUMN = "SeriousDlqin2yrs"
RANDOM_STATE = 42

# Default dataset path (falls back to credit_data.csv if cs-training.csv is not present)
DEFAULT_DATA_PATHS = [
    os.path.join("data", "cs-training.csv"),
    os.path.join("data", "credit_data.csv"),
]

MODEL_SAVE_PATH = os.path.join("models", "risk_model.joblib")
CALIBRATION_DATA_PATH = os.path.join("data", "calibration_data.csv")
TEST_DATA_PATH = os.path.join("data", "test_data.csv")


def find_data_file(custom_path=None):
    """Locate the dataset file."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    for path in DEFAULT_DATA_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find dataset. Looked in: {DEFAULT_DATA_PATHS}. "
        f"Please ensure data/cs-training.csv exists."
    )


def load_and_split_data(data_path=None, test_size=0.2, calib_size=0.2, random_state=RANDOM_STATE):
    """
    Loads dataset, handles index column, and performs a stratified 3-way split:
    - Train Set: (1 - test_size - calib_size) e.g., 60%
    - Calibration Set: calib_size e.g., 20% (Crucial for Confidence Calibration / Conformal Prediction)
    - Test Set: test_size e.g., 20%
    
    Returns:
        X_train, y_train, X_calib, y_calib, X_test, y_test, feature_names
    """
    file_path = find_data_file(data_path)
    print(f"[1/5] Loading dataset from: {file_path}")
    df = pd.read_csv(file_path)

    # Drop unnamed index column if present (common in Kaggle exports)
    unnamed_cols = [c for c in df.columns if "Unnamed" in c or c.strip() == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {df.columns.tolist()}")

    # Separate features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    feature_names = list(X.columns)

    print(f"      Total samples: {len(df):,}")
    print(f"      Features ({len(feature_names)}): {feature_names}")
    print(f"      Class balance: Default=1 ({y.mean():.2%}), Non-Default=0 ({(1-y.mean()):.2%})")

    # Step 1: Split into temporary Train+Calib set (80%) and Test set (20%) with stratification
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    # Step 2: Split temporary set into Train (60% total) and Calibration (20% total)
    # calib_ratio relative to X_temp is calib_size / (1 - test_size) = 0.2 / 0.8 = 0.25
    calib_ratio = calib_size / (1.0 - test_size)
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp,
        test_size=calib_ratio,
        stratify=y_temp,
        random_state=random_state
    )

    print(f"[2/5] Stratified 3-way split completed:")
    print(f"      - Training set:    {len(X_train):,} samples ({len(X_train)/len(df):.0%}) | Defaults: {y_train.sum():,}")
    print(f"      - Calibration set: {len(X_calib):,} samples ({len(X_calib)/len(df):.0%}) | Defaults: {y_calib.sum():,}")
    print(f"      - Test set:        {len(X_test):,} samples ({len(X_test)/len(df):.0%}) | Defaults: {y_test.sum():,}")

    return X_train, y_train, X_calib, y_calib, X_test, y_test, feature_names


def create_risk_pipeline(random_state=RANDOM_STATE):
    """
    Creates an end-to-end scikit-learn pipeline:
    1. Imputation: SimpleImputer(strategy='median') to handle missing values (MonthlyIncome, NumberOfDependents)
    2. Scaling: StandardScaler() to normalize feature magnitudes
    3. Model: LogisticRegression() to output calibrated probability of default
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=1000,
            solver='lbfgs',
            random_state=random_state
        ))
    ])
    return pipeline


def train_risk_model(pipeline, X_train, y_train):
    """Fits the pipeline on the training data."""
    print("[3/5] Training Logistic Regression model pipeline...")
    pipeline.fit(X_train, y_train)
    print("      Model training complete.")
    return pipeline


def evaluate_risk_model(pipeline, X_test, y_test):
    """
    Evaluates the model on test data:
    - Computes default probabilities using predict_proba()
    - Calculates ROC-AUC, Precision, Recall, and F1 Score
    """
    print("[4/5] Evaluating model performance on unseen Test Set...")

    # Predict probabilities for class 1 (default within 2 years)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    # Standard 0.5 decision threshold predictions
    predictions = pipeline.predict(X_test)

    # Compute key metrics
    roc_auc = roc_auc_score(y_test, probabilities)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    print("\n" + "=" * 50)
    print("           BASELINE MODEL EVALUATION")
    print("=" * 50)
    print(f"  ROC-AUC Score : {roc_auc:.4f}  (Measures ranking capability)")
    print(f"  Precision     : {precision:.4f}  (True defaults / Predicted defaults)")
    print(f"  Recall        : {recall:.4f}  (Detected defaults / Total actual defaults)")
    print(f"  F1 Score      : {f1:.4f}  (Harmonic mean of precision & recall)")
    print("=" * 50)

    print("\nClassification Report (Default 0.5 Decision Threshold):")
    print(classification_report(y_test, predictions, target_names=["Non-Default (0)", "Default (1)"], digits=4))

    cm = confusion_matrix(y_test, predictions)
    print(f"Confusion Matrix (Test set: {len(y_test):,} samples):")
    print(f"  True Negatives (TN): {cm[0, 0]:>6} | False Positives (FP): {cm[0, 1]:>6}")
    print(f"  False Negatives (FN): {cm[1, 0]:>6} | True Positives (TP):  {cm[1, 1]:>6}")

    metrics = {
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "probabilities": probabilities
    }
    return metrics


def save_model_artifacts(pipeline, X_calib, y_calib, X_test, y_test):
    """
    Saves:
    1. Trained pipeline to 'models/risk_model.joblib'
    2. Calibration set to 'data/calibration_data.csv' (for Conformal Prediction)
    3. Test set to 'data/test_data.csv'
    """
    print("\n[5/5] Saving model and datasets...")
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Save trained scikit-learn pipeline
    joblib.dump(pipeline, MODEL_SAVE_PATH)
    print(f"      Saved model pipeline -> {MODEL_SAVE_PATH}")

    # Save calibration set with target for conformal prediction calibration
    calib_df = X_calib.copy()
    calib_df[TARGET_COLUMN] = y_calib
    calib_df.to_csv(CALIBRATION_DATA_PATH, index=False)
    print(f"      Saved calibration set -> {CALIBRATION_DATA_PATH} ({len(calib_df):,} rows)")

    # Save test set for downstream multi-agent evaluation
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"      Saved test set        -> {TEST_DATA_PATH} ({len(test_df):,} rows)")


def predict_single_applicant(pipeline, applicant_dict):
    """
    Helper utility for beginner testing / single applicant inference.
    
    Args:
        pipeline: Trained scikit-learn pipeline
        applicant_dict: Dictionary containing feature values for one applicant
        
    Returns:
        dict with default probability, predicted risk class, and interpretation
    """
    df_applicant = pd.DataFrame([applicant_dict])
    prob_default = float(pipeline.predict_proba(df_applicant)[0, 1])
    
    return {
        "default_probability": round(prob_default, 4),
        "default_probability_percent": f"{prob_default * 100:.2f}%",
        "binary_prediction": int(prob_default >= 0.5),
        "risk_level": "High Risk" if prob_default >= 0.3 else ("Moderate Risk" if prob_default >= 0.1 else "Low Risk")
    }


def main():
    """Main execution function."""
    print("==================================================================")
    print(" Confidence-Calibrated Multi-Agent AI - Baseline ML Risk Model   ")
    print("==================================================================\n")

    # 1. Load data and split
    X_train, y_train, X_calib, y_calib, X_test, y_test, feature_names = load_and_split_data()

    # 2. Build pipeline
    pipeline = create_risk_pipeline()

    # 3. Train model
    pipeline = train_risk_model(pipeline, X_train, y_train)

    # 4. Evaluate model
    metrics = evaluate_risk_model(pipeline, X_test, y_test)

    # 5. Save model and splits
    save_model_artifacts(pipeline, X_calib, y_calib, X_test, y_test)

    # 6. Run a quick beginner demo prediction
    print("\n--- Example Single Applicant Inference Demo ---")
    sample_applicant = {
        "RevolvingUtilizationOfUnsecuredLines": 0.75,
        "age": 42,
        "NumberOfTime30-59DaysPastDueNotWorse": 1,
        "DebtRatio": 0.45,
        "MonthlyIncome": 5000.0,
        "NumberOfOpenCreditLinesAndLoans": 6,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60-89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2.0
    }
    result = predict_single_applicant(pipeline, sample_applicant)
    print(f"Sample Applicant Features: {sample_applicant}")
    print(f"Result: {result}")
    print("\n[SUCCESS] Baseline ML model training and evaluation finished successfully!")


if __name__ == "__main__":
    main()
