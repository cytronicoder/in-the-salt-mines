"""Henderson-Hasselbalch regression utilities for apparent pKa extraction.

This module implements the Stage 2 regression step in the two-stage pKa_app
workflow. It performs no I/O and relies exclusively on chemically validated
inputs supplied by the analysis pipeline.

Experimental Context (from IA):
    - Ethanoic acid (CH₃COOH) is a weak Brønsted-Lowry acid
    - Titrated with 0.10 M NaOH at 26 ± 1°C
    - NaCl concentrations: 0.00-1.00 M (ionic strength modifier)
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

    Scientific interpretation:
        The fitted intercept is an apparent pKa_app, not a thermodynamic pKa.
        Because ionic strength (from added NaCl) alters activity coefficients,
        pKa_app is an operational, concentration-based parameter that varies with
        [NaCl]. The pH sensor measures H⁺ activity, so at [NaCl] > 0, the measured
        pH reflects both the true equilibrium and non-ideal solution behavior.

        All conclusions must be comparative across ionic strength conditions
        rather than absolute. The IA research question investigates how pKa_app
        changes as [NaCl] is varied from 0.00 to 1.00 M.

    Model form:
        ``pH = m * log10(V / (V_eq - V)) + b``, where ``b`` is pKa_app and ``m``
        is expected to be close to 1.0 for an ideal buffer. Deviations in slope
        indicate non-ideality or experimental artifacts.

    Args:
        step_df: Step-aggregated data with ``Volume (cm^3)`` and ``pH_step``.
        veq: Equivalence volume in cm^3. Expected to be ~25 cm^3 for 0.10 M
            acid titrated with 0.10 M base starting from 25.00 cm^3.
        pka_app_guess: Stage 1 pKa_app estimate from the half-equivalence point.

    Returns:
        A dictionary containing the fitted pKa_app, slope, R^2, confidence
        interval information, and the buffer-region DataFrame used for the fit.

    Raises:
        ValueError: If required columns are missing, V_eq is invalid, or the
            buffer region contains fewer than three valid points.
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
