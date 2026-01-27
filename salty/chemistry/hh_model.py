"""Henderson–Hasselbalch regression utilities for apparent pKa extraction."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from salty.chemistry.buffer_region import select_buffer_region
from salty.stats.regression import linear_regression


def fit_henderson_hasselbalch(
    step_df: pd.DataFrame, veq: float, pka_app_guess: float
) -> Dict[str, object]:
    """
    Perform Henderson–Hasselbalch regression in the chemically valid buffer region.

    NOTE:
    The extracted pKa is an *apparent pKa* (pKa_app), not a thermodynamic pKa.
    Because NaCl alters ionic strength, activity coefficients are not constant.
    Henderson–Hasselbalch is used here as an operational model to compare
    systematic shifts in pKa_app with ionic strength.
    """
    if step_df.empty or "Volume (cm³)" not in step_df.columns:
        raise ValueError("Step data must include Volume (cm³) for regression.")
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
        raise ValueError("Insufficient valid data for regression.")

    reg = linear_regression(log_ratio, pH_values, min_points=3)
    slope = float(reg["m"])
    intercept = float(reg["b"])

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
        "r2": float(reg["r2"]),
        "n_points": int(len(log_ratio)),
        "se_intercept": float(reg["se_b"]),
        "ci95_intercept": float(reg["ci95_b"]),
        "buffer_df": buffer_df,
    }
