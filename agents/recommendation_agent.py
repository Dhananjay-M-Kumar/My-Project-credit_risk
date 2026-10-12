"""
Recommendation Agent
====================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This module implements the decision-support Recommendation Agent.

Key Principles:
1. Pure Rule-Based Logic (NO LLM / NO Gemini used in this agent).
2. Decision-Support Terminology:
   - Does NOT automatically approve or reject loans.
   - Recommends structured human review workflows based on calibrated risk tiers:
     * LOW RISK      -> "Normal Review"
     * MODERATE RISK -> "Further Review"
     * HIGH RISK     -> "Enhanced Review"
3. Safe, accountable, and transparent advisory for human credit underwriters.

Disclaimer:
This prototype provides advisory recommendations for committee review and does
not constitute an automated final credit decision.
"""


def generate_recommendation(default_probability, risk_level, analysis=None):
    """
    Generates a structured decision-support recommendation based on the applicant's
    risk classification and estimated default probability.

    Review Tier Mapping:
    - LOW RISK      -> 'Normal Review'
    - MODERATE RISK -> 'Further Review'
    - HIGH RISK     -> 'Enhanced Review'

    Args:
        default_probability (float): Numerical probability of default from the ML model.
        risk_level (str): Risk tier ('LOW RISK', 'MODERATE RISK', 'HIGH RISK') from Risk Agent.
        analysis (str, optional): Qualitative analysis text from Analysis Agent.

    Returns:
        dict: Structured recommendation containing:
            - default_probability (float)
            - risk_level (str)
            - recommendation (str)
            - reason (str)
    """
    # Normalize risk level string for reliable comparison
    normalized_risk = str(risk_level).strip().upper()

    if "LOW" in normalized_risk:
        recommendation = "Normal Review"
        reason = (
            "Applicant has a relatively low estimated default probability "
            "and therefore requires normal review."
        )
    elif "MODERATE" in normalized_risk:
        recommendation = "Further Review"
        reason = (
            "Applicant has a moderate estimated default probability "
            "and should receive further review of the financial profile."
        )
    elif "HIGH" in normalized_risk:
        recommendation = "Enhanced Review"
        reason = (
            "Applicant has a high estimated default probability "
            "and should receive enhanced review."
        )
    else:
        # Fallback based directly on numerical probability
        prob = float(default_probability)
        if prob < 0.10:
            recommendation = "Normal Review"
            reason = "Applicant has a relatively low estimated default probability and therefore requires normal review."
        elif prob < 0.20:
            recommendation = "Further Review"
            reason = "Applicant has a moderate estimated default probability and should receive further review of the financial profile."
        else:
            recommendation = "Enhanced Review"
            reason = "Applicant has a high estimated default probability and should receive enhanced review."

    return {
        "default_probability": round(float(default_probability), 4),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "reason": reason
    }
