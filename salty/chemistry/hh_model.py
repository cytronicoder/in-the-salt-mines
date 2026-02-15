"""Henderson-Hasselbalch regression utilities for apparent pKa extraction.

This module implements the Stage 2 regression step in the two-stage pKa_app
workflow. It performs no I/O and relies exclusively on chemically validated
inputs supplied by the analysis pipeline.

Experimental Context (from IA):
    - Ethanoic acid (CH₃COOH) is a weak Brønsted-Lowry acid
    - Titrated with 0.10 M NaOH at 26 ± 1°C
    - NaCl concentrations: 0.00-0.80 M (ionic strength modifier)
    - pH measured using Vernier pH Sensor (±0.3 pH, measures H⁺ activity)

Theoretical Framework:
    At the half-equivalence point, [CH₃COOH] ≈ [CH₃COO⁻], so:
        pH ≈ pKa + log₁₀([A⁻]/[HA]) ≈ pKa  (when ratio ≈ 1)

    However, pH probes measure H⁺ activity (a_H+), not concentration. In solutions
    with added NaCl, ionic strength (μ) causes activity coefficients (γ) to deviate
    from unity. The measured pH reflects an apparent pKa (pKa_app) that varies with
    ionic strength:

        pH = pKa_app + log₁₀([A⁻]/[HA])

    where pKa_app incorporates non-ideal behavior at each [NaCl] condition.
"""

from __future__ import annotations

import warnings
from typing import Dict

import numpy as np
import pandas as pd

from salty.chemistry.buffer_region import select_buffer_region
from salty.stats.regression import linear_regression


def fit_henderson_hasselbalch(
    step_df: pd.DataFrame, veq: float, pka_app_guess: float
) -> Dict[str, object]:
    r"""Fit the Henderson-Hasselbalch model within the buffer region.

    This function implements stage 2 (refined regression) of the two-stage
    pKa_app extraction workflow. Stage 1 provides a coarse estimate from the
    half-equivalence point; Stage 2 uses that estimate to define the buffer
    region (``|pH - pKa_app| ≤ 1``) and performs the regression only within that
    chemically valid window.

    Args:
        step_df (pandas.DataFrame): Step-aggregated titration table with
            ``Volume (cm^3)`` and ``pH_step``.
        veq (float): Equivalence volume in cm^3.
        pka_app_guess (float): Stage-1 apparent pKa estimate from
            half-equivalence interpolation (dimensionless).

    Returns:
        dict[str, object]: Fit output containing ``pka_app`` (intercept,
        dimensionless), ``slope_reg`` (dimensionless), ``r2_reg``
        (dimensionless), uncertainty diagnostics, and the buffer-region
        dataframe used for fitting.

    Raises:
        ValueError: If required columns are absent, ``veq`` is invalid, or
            fewer than three valid buffer-region points are available.

    Note:
        Model form is ``pH = m * log10(V / (V_eq - V)) + b`` where ``b`` is
        apparent ``pKa_app`` and ``m`` is expected near 1.0 in an ideal buffer.
        Numerical stability is enforced by pre-equivalence filtering (``V < V_eq``)
        before evaluating ``log10(V / (V_eq - V))``. Interpret fitted
        ``pKa_app`` as an operational, concentration-based parameter for
        comparative ionic-strength analysis rather than as a thermodynamic
        constant. IA correspondence: this is the Stage-2 regression method.

        Failure modes include too few buffer points, unstable endpoint inputs,
        and strong slope deviation from unity.

    References:
        Henderson-Hasselbalch linear form for weak-acid/conjugate-base buffers.
    """
    if (
        step_df.empty
        or "Volume (cm^3)" not in step_df.columns
        or "pH_step" not in step_df.columns
    ):
        raise ValueError(
            "Step data must include Volume (cm^3) and pH_step for regression."
        )
    if not np.isfinite(veq) or veq <= 0:
        raise ValueError("Equivalence volume must be positive and finite.")
    if not np.isfinite(pka_app_guess):
        raise ValueError("pKa_app guess must be finite for buffer selection.")

    volumes = pd.to_numeric(step_df["Volume (cm^3)"], errors="coerce").to_numpy(
        dtype=float
    )
    pH_values = pd.to_numeric(step_df["pH_step"], errors="coerce").to_numpy(dtype=float)

    valid = np.isfinite(volumes) & np.isfinite(pH_values)
    volumes = volumes[valid]
    pH_values = pH_values[valid]

    pre_eq = (volumes > 0) & (volumes < veq)
    volumes = volumes[pre_eq]
    pH_values = pH_values[pre_eq]

    if len(volumes) == 0:
        raise ValueError("No pre-equivalence data available for regression.")

    log_ratio = np.log10(volumes / (veq - volumes))
    finite = np.isfinite(log_ratio)
    log_ratio = log_ratio[finite]
    pH_values = pH_values[finite]
    volumes = volumes[finite]

    buffer_mask = select_buffer_region(pH_values, pka_app_guess)
    log_ratio = log_ratio[buffer_mask]
    pH_values = pH_values[buffer_mask]
    volumes = volumes[buffer_mask]

    if len(log_ratio) < 3:
        raise ValueError(
            f"Insufficient valid buffer points for regression. "
            f"Found {len(log_ratio)} points, minimum 3 required."
        )

    reg = linear_regression(log_ratio, pH_values, min_points=3)
    slope = float(reg["m"])
    intercept = float(reg["b"])

    if abs(slope - 1.0) > 0.1:
        warnings.warn(
            f"HH slope ({slope:.3f}) deviates significantly from unity; "
            f"model assumptions may be violated.",
            UserWarning,
            stacklevel=2,
        )

    pH_fit = slope * log_ratio + intercept

    buffer_df = pd.DataFrame(
        {
            "Volume (cm^3)": volumes,
            "log10_ratio": log_ratio,
            "pH_step": pH_values,
            "pH_fit": pH_fit,
        }
    )

    return {
        "pka_app": intercept,
        "slope_reg": slope,
        "r2_reg": float(reg["r2"]),
        "n_points": int(len(log_ratio)),
        "se_intercept": float(reg["se_b"]),
        "ci95_intercept": float(reg["ci95_b"]),
        "buffer_df": buffer_df,
    }
