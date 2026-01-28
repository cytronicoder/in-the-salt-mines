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

# Backwards-compatible wrapper for legacy code
def get_equipment_uncertainty(equipment: str, value: float | None = None) -> float | None:
    """
    Return absolute uncertainty for named lab equipment (legacy API).
    
    Returns None for unknown equipment (instead of raising KeyError).
    Returns None for percent-based equipment if value is omitted (instead of raising ValueError).
    
    For new code, use uncertainty_for_equipment() which raises explicit exceptions.
    """
    try:
        return uncertainty_for_equipment(equipment, value)
    except KeyError:
        # Legacy behavior: return None for unknown equipment
        return None
    except ValueError:
        # Legacy behavior: return None for percent-based equipment without value
        return None

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
