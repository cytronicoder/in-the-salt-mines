"""Buffer-region selection utilities for acid-base titrations."""

from __future__ import annotations

import numpy as np


def select_buffer_region(pH: np.ndarray, pKa_app: float) -> np.ndarray:
    r"""
    Return boolean mask for chemically valid buffer region.

    BUFFER REGION DEFINITION:
    =========================
    $\lvert \mathrm{pH} - \mathrm{p}K_{a,\mathrm{app}} \rvert \le 1$

    This corresponds to the range where $0.1 \le \dfrac{[\ce{A-}]}{[\ce{HA}]} \le 10$,
    ensuring Henderson-Hasselbalch approximation validity.

    USAGE IN TWO-STAGE pKa_app EXTRACTION:
    =======================================
    This function is called with pKa_app_initial (half-equivalence pH estimate)
    to define the buffer region for Stage 2 regression.

    Parameters:
    -----------
    pH : np.ndarray
        Measured pH values
    pKa_app : float
        Initial pKa_app estimate (from half-equivalence point)

    Returns:
    --------
    np.ndarray
        Boolean mask for buffer region points

    Raises:
    -------
    ValueError
        If pKa_app is not finite
    """
    pH_arr = np.asarray(pH, dtype=float)
    if not np.isfinite(pKa_app):
        raise ValueError("pKa_app must be finite to select buffer region.")
    return np.abs(pH_arr - float(pKa_app)) <= 1.0
