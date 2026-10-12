"""
Conformal Calibration Module Shortcut
=====================================
Re-exports conformal calibration functions from conformal_calibration.py.
"""

from conformal.conformal_calibration import (
    compute_calibration_quantile,
    calibrate_prediction,
)

__all__ = ["compute_calibration_quantile", "calibrate_prediction"]
