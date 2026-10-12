"""
Analysis Agent for Credit Risk Assessment
=========================================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This module implements the qualitative Financial Analysis Agent using Google's
official Gemini Python SDK (google-genai).

Key Responsibilities:
1. Receives raw credit risk features for an applicant.
2. Prompts Gemini to generate a structured qualitative report:
   - Financial profile summary
   - Important risk factors
   - Positive factors
   - Important observations
3. Adheres strictly to safety constraints:
   - Does NOT approve or reject loans.
   - Does NOT invent numerical risk probabilities (reserved for ML model).
   - Does NOT claim absolute certainty.
   - Acts as decision-support advisory only.
"""

import os
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# The 10 standard credit-risk applicant features
EXPECTED_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

# Candidate models to handle temporary capacity fluctuations seamlessly
DEFAULT_MODELS = [
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
    "gemini-2.5-flash",
]


def _get_gemini_client():
    """
    Initializes and returns the Google GenAI client using the GEMINI_API_KEY from .env.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment or .env file. "
            "Please ensure your GEMINI_API_KEY is configured in the .env file."
        )
    return genai.Client(api_key=api_key)


def build_analysis_prompt(applicant_data):
    """
    Constructs the prompt for Gemini given applicant feature dictionary.
    Formats the 10 features clearly and instructs Gemini on safety boundaries.
    """
    formatted_features = []
    for feat in EXPECTED_FEATURES:
        val = applicant_data.get(feat, "N/A")
        # Format floating-point numbers cleanly if applicable
        if isinstance(val, float):
            val_str = f"{val:.4f}" if abs(val) < 10 else f"{val:.2f}"
        else:
            val_str = str(val)
        formatted_features.append(f"- {feat}: {val_str}")

    features_block = "\n".join(formatted_features)

    prompt = f"""You are an expert AI Financial Analyst Agent supporting a credit risk evaluation committee.
Your objective is to provide a structured qualitative analysis of an applicant's credit profile based strictly on the financial data provided below.

APPLICANT FINANCIAL DATA:
{features_block}

INSTRUCTIONS & STRUCTURE:
Please provide your analysis structured into the following 4 sections:
1. Financial Profile Summary: Overview of the applicant's age, income level, debt load, and general credit utilization.
2. Important Risk Factors: Highlight elevated risk metrics (such as past delinquencies, high revolving credit utilization, heavy debt-to-income ratio).
3. Positive Factors: Highlight stabilizing metrics (such as on-time payment track record, established credit lines, stable income, or manageable debt).
4. Important Observations: Note any context-specific anomalies, missing data, or patterns relevant to human credit underwriters.

CRITICAL CONSTRAINTS:
- Do NOT make a final loan approval or loan rejection decision.
- Do NOT invent, guess, or calculate a numerical default probability (the quantitative ML model handles this).
- Do NOT claim certainty or pretend to be an automated bank decision authority.
- Maintain an objective, professional, and balanced decision-support tone.
"""
    return prompt


def analyze_applicant(applicant_data, model_name=None):
    """
    Analyzes an applicant's financial features using Gemini.

    Args:
        applicant_data (dict): Dictionary containing the 10 applicant features.
        model_name (str, optional): Gemini model to use. Defaults to GEMINI_MODEL env var or 'gemini-flash-latest'.

    Returns:
        str: Gemini's qualitative financial analysis text.
    """
    if not isinstance(applicant_data, dict):
        raise TypeError(f"applicant_data must be a dictionary, got {type(applicant_data)}")

    # Initialize client from .env API key
    client = _get_gemini_client()

    # Construct the structured prompt
    prompt = build_analysis_prompt(applicant_data)

    # Determine model list to try (user-specified / env var / default list)
    env_model = os.getenv("GEMINI_MODEL")
    models_to_try = [model_name] if model_name else ([env_model] if env_model else DEFAULT_MODELS)

    last_error = None
    for model in models_to_try:
        if not model:
            continue
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt
            )
            if response and response.text:
                return response.text
        except Exception as e:
            last_error = e
            # Small delay before trying next fallback model
            time.sleep(1)
            continue

    # If all models failed, raise the last encountered error
    raise RuntimeError(f"Failed to generate analysis with Gemini. Last error: {last_error}")
