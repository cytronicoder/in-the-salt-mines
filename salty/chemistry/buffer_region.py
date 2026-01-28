"""Buffer-region selection utilities for acid-base titrations."""

from __future__ import annotations

import numpy as np


def select_buffer_region(pH: np.ndarray, pKa_app: float) -> np.ndarray:
    """
    Return boolean mask for chemically valid buffer region:
    |pH − pKa_app| ≤ 1
    Equivalent to 0.1 ≤ [A−]/[HA] ≤ 10.
    """
    pH_arr = np.asarray(pH, dtype=float)
    if not np.isfinite(pKa_app):
        raise ValueError("pKa_app must be finite to select buffer region.")
    return np.abs(pH_arr - float(pKa_app)) <= 1.0
