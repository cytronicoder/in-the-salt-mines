"""Henderson-Hasselbalch regression utilities for apparent pKa extraction."""

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
    r"""
    Perform Henderson–Hasselbalch regression in the chemically valid buffer region.

    TWO-STAGE pKa_app EXTRACTION PROTOCOL:
    ========================================
    Stage 1 — Coarse estimate (performed externally in analysis.py):
        - Estimate pKa_app_initial using half-equivalence pH (pH at V ≈ 0.5·V_eq)
        - No regression allowed at this stage

    Stage 2 — Refined regression (this function):
        - Uses pKa_app_initial (pka_app_guess) to define buffer region
        - Performs Henderson–Hasselbalch regression only inside valid buffer region
        - Buffer region: $\lvert \mathrm{pH} - \mathrm{p}K_{a,\mathrm{app}} \rvert \le 1$

    CRITICAL INTERPRETATION GUARDRAILS:
    ====================================
    The extracted pKa_app is an operational, concentration-based parameter.
    Because NaCl alters ionic strength, activity coefficients (γ) vary between trials.
    Therefore:
    - pKa_app does NOT represent thermodynamic acid strength.
    - Observed shifts reflect combined effects of dissociation equilibrium and
      activity coefficient changes.
    All conclusions must be comparative across ionic strengths, not absolute.

    Failure to recognize this limitation invalidates chemical interpretation.

    REGRESSION EQUATION:
    ====================
    $\displaystyle \mathrm{pH} = m\,\log_{10}\!\left(\dfrac{V}{V_{\mathrm{eq}} - V}\right) + b$

    Where:
    - $b \to \mathrm{p}K_{a,\mathrm{app}}$ (apparent pK_{a})
    - $m$ → Henderson–Hasselbalch slope (expected $\approx 1.0$ for ideal buffer)
    """
    if (
        step_df.empty
        or "Volume (cm³)" not in step_df.columns
        or "pH_step" not in step_df.columns
    ):
        raise ValueError(
            "Step data must include Volume (cm³) and pH_step for regression."
        )
    if not np.isfinite(veq) or veq <= 0:
        raise ValueError("Equivalence volume must be positive and finite.")
    if not np.isfinite(pka_app_guess):
        raise ValueError("pKa_app guess must be finite for buffer selection.")

    volumes = pd.to_numeric(step_df["Volume (cm³)"], errors="coerce").to_numpy(
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

    # SLOPE EXPECTATION CHECK (DIAGNOSTIC)
    # Henderson-Hasselbalch theory predicts slope ≈ 1.0 for ideal buffer
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
            "Volume (cm³)": volumes,
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
