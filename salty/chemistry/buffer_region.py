"""Select chemically valid buffer regions for Henderson-Hasselbalch analysis.

Experimental Context:
    In this investigation, ethanoic acid (0.10 M) is titrated with NaOH (0.10 M)
    in the presence of varying NaCl concentrations (0.00-0.80 M). The buffer region
    is where the Henderson-Hasselbalch equation provides a chemically valid
    description of the pH.

Buffer Region Definition:
    The operational criterion |pH - pKa_app| ≤ 1 corresponds to:
        0.1 ≤ [A⁻]/[HA] ≤ 10

    This range ensures that both the acid and conjugate base are present in
    significant amounts, making the solution an effective buffer. Outside this
    region, the log term dominates and small volume uncertainties cause large
    pH changes, reducing the reliability of pKa_app determination.
"""

from __future__ import annotations

import numpy as np


def select_buffer_region(pH: np.ndarray, pKa_app: float) -> np.ndarray:
    """Return a boolean mask for the chemically valid buffer region.

    The buffer region is defined as ``|pH - pKa_app| ≤ 1``. This operational
    criterion corresponds to ``0.1 ≤ [A⁻]/[HA] ≤ 10`` and is where the
    Henderson-Hasselbalch approximation is considered chemically defensible.

    In the context of this IA, the pKa_app argument must be the Stage 1
    (half-equivalence) estimate, which is then used to define the Stage 2
    regression window. Because pKa_app varies with [NaCl] due to ionic strength
    effects on activity coefficients, each titration at a different [NaCl]
    will have a different buffer region centered around its respective pKa_app.

    Args:
        pH (numpy.ndarray): Measured pH values for one titration run (pH
            units).
        pKa_app (float): Stage-1 apparent pKa estimate (dimensionless) used to
            center the buffer window.

    Returns:
        numpy.ndarray: Boolean mask selecting points satisfying
        ``|pH - pKa_app| <= 1``; shape matches ``pH``.

    Raises:
        ValueError: If ``pKa_app`` is non-finite.

    Note:
        The returned mask is shape-aligned with input ``pH`` and can be applied
        directly to transformed regression arrays.
        IA correspondence: this is the explicit Stage-2 inclusion-window rule.

    References:
        Henderson-Hasselbalch validity region, equivalent to
        ``0.1 <= [A-]/[HA] <= 10``.
    """
    pH_arr = np.asarray(pH, dtype=float)
    if not np.isfinite(pKa_app):
        raise ValueError("pKa_app must be finite to select buffer region.")
    return np.abs(pH_arr - float(pKa_app)) <= 1.0
