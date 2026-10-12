"""
Risk Assessment Agent
=====================
Project: Confidence-Calibrated Multi-Agent AI for High-Stakes Decision Support

This module implements the quantitative Risk Assessment Agent.

Design Principles:
1. NO LLMs / NO Gemini used in this agent:
   - Numerical default probabilities are calculated objectively by the trained
     scikit-learn Logistic Regression model (Phase 1).
2. Pure deterministic decision-support logic:
   - Maps calibrated ML default probabilities into standard risk tiers.
3. Integrates with the qualitative Analysis Agent (Phase 2):
   - Combines numerical probability, calibrated risk category, and the qualitative
     financial analysis into a unified assessment payload.

Disclaimer:
This system is an academic prototype designed for high-stakes decision support.
The risk classifications generated here are advisory indicators, not legal
or automated banking credit determinations.
"""


def assess_risk(default_probability, analysis):
    """
    Assesses credit risk by classifying the ML-predicted default probability
    into standardized risk levels and pairing it with qualitative analysis.

    Threshold Logic:
    - Default Probability < 0.10 (10%)         -> LOW RISK
    - Default Probability >= 0.10 and < 0.20   -> MODERATE RISK
    - Default Probability >= 0.20 (20%)        -> HIGH RISK

    Args:
        default_probability (float): Numerical probability of default produced
                                     by the trained ML model's predict_proba().
        analysis (str): Qualitative narrative produced by the Analysis Agent.

    Returns:
        dict: Structured assessment containing:
            - default_probability (float)
            - default_probability_percent (str)
            - risk_level (str)
            - analysis (str)
    """
    # Ensure default_probability is a float
    prob = float(default_probability)

    # 1. Apply risk classification thresholds
    if prob < 0.10:
        risk_level = "LOW RISK"
    elif prob < 0.20:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "HIGH RISK"

    # 2. Format percentage string for human-readable reporting
    prob_percent = f"{prob * 100:.2f}%"

    # 3. Assemble and return the structured assessment dictionary
    assessment = {
        "default_probability": round(prob, 4),
        "default_probability_percent": prob_percent,
        "risk_level": risk_level,
        "analysis": analysis
    }

    return assessment
