"""Numerical regression and systematic uncertainty utilities.

This subpackage provides array-based regression diagnostics and worst-case
uncertainty propagation. Statistical diagnostics (e.g., standard errors) are
reported separately from systematic uncertainty bounds, which are intended as
conservative experimental limits rather than probabilistic confidence levels.
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
