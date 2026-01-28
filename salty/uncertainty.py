"""Compatibility wrapper for uncertainty utilities."""

from .stats.uncertainty import (
    Quantity,
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

# Backwards-compatible alias for legacy code
get_equipment_uncertainty = uncertainty_for_equipment

__all__ = [
    "Quantity",
    "add_subtract",
    "burette_delivered_uncertainty",
    "combine_uncertainties",
    "concentration_uncertainty",
    "format_value_with_uncertainty",
    "get_equipment_uncertainty",
    "mul_div",
    "power",
    "round_value_to_uncertainty",
    "uncertainty_for_equipment",
]
