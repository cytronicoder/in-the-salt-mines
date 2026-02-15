"""Centralized unit conversion utilities."""

from __future__ import annotations

CM3_PER_DM3: float = 1000.0


def cm3_to_dm3(volume_cm3: float) -> float:
    """Convert a delivered volume from cm^3 to dm^3.

    Args:
        volume_cm3 (float): Volume in cubic centimeters (cm^3, numerically
            equal to mL).

    Returns:
        float: Volume in cubic decimeters (dm^3, numerically equal to L).

    Note:
        Use this conversion in concentration-preparation uncertainty calculations
        so concentration units remain mol dm^-3 throughout the analysis.

    References:
        SI derived-unit relation: 1 dm^3 = 1000 cm^3.
    """
    return float(volume_cm3) / CM3_PER_DM3
