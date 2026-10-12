"""
Test Script for Analysis Agent
==============================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This script tests the Gemini-powered analyze_applicant function by:
1. Loading data/cs-training.csv with pandas.
2. Selecting the 10 credit-risk features (excluding the target SeriousDlqin2yrs).
3. Extracting the first applicant record.
4. Converting the record into a Python dictionary.
5. Making exactly ONE call to analyze_applicant().
6. Printing the structured Gemini analysis clearly.
"""

import os
import pandas as pd
from agents.analysis_agent import analyze_applicant, EXPECTED_FEATURES


def main():
    print("==================================================================")
    print("      Testing Phase 2 Analysis Agent (Single Gemini API Call)     ")
    print("==================================================================\n")

    # 1. Locate dataset
    data_path = os.path.join("data", "cs-training.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "credit_data.csv")

    print(f"[1/4] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Remove Kaggle index column if present
    unnamed_cols = [c for c in df.columns if "Unnamed" in c or c.strip() == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # 2. Extract first applicant (row index 0)
    print(f"[2/4] Extracting applicant #1 (first record)...")
    first_row = df.iloc[0]

    # 3. Create dictionary containing ONLY the 10 credit risk features
    # (SeriousDlqin2yrs is strictly NOT included)
    applicant_data = {feat: first_row[feat] for feat in EXPECTED_FEATURES}

    print("\nApplicant Features Sent to Agent:")
    print("-" * 50)
    for feat, val in applicant_data.items():
        print(f"  • {feat:36}: {val}")
    print("-" * 50)

    # 4. Invoke the Analysis Agent (single API call)
    print("\n[3/4] Calling Gemini Analysis Agent via google-genai SDK...")
    analysis_text = analyze_applicant(applicant_data)

    # 5. Display the structured response
    print("\n[4/4] Gemini Financial Analysis Agent Response:")
    print("=" * 65)
    print(analysis_text)
    print("=" * 65)

    print("\n[SUCCESS] Analysis Agent test passed successfully with 1 Gemini API call.")


if __name__ == "__main__":
    main()
