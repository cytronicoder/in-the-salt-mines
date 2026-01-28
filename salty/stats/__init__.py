"""
Statistical utilities for titration analysis.

This subpackage provides numerical routines for regression analysis and
uncertainty propagation. All functions operate on arrays and primitive types;
no chemistry-specific logic is included.

Modules:
    regression:
        Linear regression with standard error and confidence interval
        computation. Includes slope uncertainty estimation from endpoint
        error boxes.

    uncertainty:
        IB DP worst-case uncertainty propagation for addition, subtraction,
        multiplication, division, and powers. Equipment uncertainty lookup
        and rounding utilities.

Design Principle:
    This subpackage has no dependencies on chemistry/ or plotting/ modules.
    It provides pure numerical utilities that can be independently tested.
"""

from .regression import linear_regression, slope_uncertainty_from_endpoints
from .uncertainty import (
    add_subtract,
    burette_delivered_uncertainty,
    combine_uncertainties,
    concentration_uncertainty,
    format_value_with_uncertainty,
    mul_div,
    power,
    round_value_to_uncertainty,
    uncertainty_for_equipment,
)

__all__ = [
    "linear_regression",
    "slope_uncertainty_from_endpoints",
    "add_subtract",
    "burette_delivered_uncertainty",
    "combine_uncertainties",
    "concentration_uncertainty",
    "format_value_with_uncertainty",
    "mul_div",
    "power",
    "round_value_to_uncertainty",
    "uncertainty_for_equipment",
]
