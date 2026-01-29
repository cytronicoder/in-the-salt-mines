"""Select chemically valid buffer regions for Henderson-Hasselbalch analysis."""

from __future__ import annotations

import numpy as np


def select_buffer_region(pH: np.ndarray, pKa_app: float) -> np.ndarray:
    """Return a boolean mask for the chemically valid buffer region.

    The buffer region is defined as ``|pH - pKa_app| ≤ 1``. This operational
    criterion corresponds to ``0.1 ≤ [A⁻]/[HA] ≤ 10`` and is where the
    Henderson-Hasselbalch approximation is considered chemically defensible.
    The pKa_app argument must be the Stage 1 (half-equivalence) estimate used
    to define the Stage 2 regression window.

    Args:
        pH: Array of measured pH values.
        pKa_app: Apparent pKa estimate from the half-equivalence point.

    Returns:
        Boolean NumPy array indicating which points lie in the buffer region.

    Raises:
        ValueError: If ``pKa_app`` is not finite.
    """
    pH_arr = np.asarray(pH, dtype=float)
    if not np.isfinite(pKa_app):
        raise ValueError("pKa_app must be finite to select buffer region.")
    return np.abs(pH_arr - float(pKa_app)) <= 1.0
