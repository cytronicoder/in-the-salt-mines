"""
Chemistry-specific models for acid-base titration analysis.

This subpackage implements the Henderson-Hasselbalch regression model for
extracting apparent pKa values from titration data within chemically valid
buffer regions.

Modules:
    hh_model:
        Henderson-Hasselbalch regression with interpretation guardrails.
        Implements the Stage 2 refined regression of the two-stage pKa_app
        extraction protocol.

    buffer_region:
        Buffer region selection enforcing |pH − pKa_app| ≤ 1 constraint.
        Ensures Henderson-Hasselbalch approximation validity.

Interpretation Guardrails:
    The extracted pKa_app is an operational, concentration-based parameter.
    It does NOT represent thermodynamic acid strength because:
    1. Activity coefficients vary with ionic strength
    2. NaCl addition alters the Debye-Hückel ionic atmosphere

    All conclusions must be comparative across ionic strengths, not absolute.

Design Principle:
    This subpackage has no dependencies on plotting/ or matplotlib.
    It provides pure chemistry models that can be independently tested.
"""

from .buffer_region import select_buffer_region
from .hh_model import fit_henderson_hasselbalch

__all__ = ["select_buffer_region", "fit_henderson_hasselbalch"]
