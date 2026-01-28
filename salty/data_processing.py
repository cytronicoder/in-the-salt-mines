"""
Data processing utilities for Logger Pro titration CSV exports.

This module handles the conversion of raw titration data (continuous pH readings
during NaOH addition) into the aggregated step-wise format required for
equivalence point detection and pKa_app analysis.

Data Flow:
    1. load_titration_data: Read CSV file into DataFrame
    2. extract_runs: Identify and separate individual experimental runs
    3. aggregate_volume_steps: Convert continuous readings to step-wise data
       with representative pH values per volume increment

Step Aggregation:
    At each volume step, the pH stabilizes over time as the system equilibrates.
    The aggregate_volume_steps function takes the median of the final readings
    ("tail") at each volume to obtain a representative equilibrium pH.

    This addresses:
    - Transient mixing effects immediately after base addition
    - Electrode response time
    - Minor drift during stabilization

Interpolation Support:
    The processed step data feeds into PCHIP interpolation for smooth curve
    generation and derivative computation. Strictly increasing, unique x-values
    are enforced.

Logger Pro Format:
    Expected columns follow Logger Pro naming conventions:
    - 'Run 1: Volume (cm³)', 'Run 1: pH', etc.
    - Or aliased columns that will be renamed during extraction
"""

from __future__ import annotations

import importlib.util
from typing import Dict, Optional

import numpy as np
import pandas as pd

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
if HAVE_SCIPY:
    from scipy.interpolate import PchipInterpolator


DEFAULT_VOLUME_BIN: Optional[float] = None


def _round_to_resolution(x: pd.Series, res: Optional[float]) -> pd.Series:
    """Round values to a specified resolution (bin width)."""
    if res is None or res <= 0:
        return x
    return (np.round(x / res) * res).astype(float)


def _infer_recorded_resolution(x: pd.Series) -> Optional[float]:
    """
    Infer the recording resolution from observed step sizes.

    Examines the distribution of volume increments to identify the
    burette resolution used during data collection.

    Args:
        x: Series of volume values.

    Returns:
        Inferred resolution (e.g., 0.05 cm³) or None if inference fails.
    """
    xv = pd.to_numeric(x, errors="coerce")
    xv = xv[np.isfinite(xv)]
    if len(xv) < 3:
        return None
    xv = np.sort(xv.to_numpy(dtype=float))
    dv = np.diff(xv)
    dv = dv[np.isfinite(dv)]
    dv = dv[dv > 0]
    if len(dv) < 3:
        return None
    q = float(np.quantile(dv, 0.5))
    candidates = [0.01, 0.02, 0.05, 0.10, 0.20]
    best = min(candidates, key=lambda c: abs(c - q))
    if abs(best - q) / best <= 0.35:
        return best
    return None


def _ensure_strictly_increasing_unique(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Ensure x-values are strictly increasing and unique.

    PCHIP interpolation requires strictly increasing x-values. This function
    sorts the data, removes NaN x-values, and merges duplicate x-values by
    averaging their y-values.

    Args:
        df: Input DataFrame.
        x_col: Name of the x-column (independent variable).
        y_cols: Names of y-columns to aggregate when merging duplicates.

    Returns:
        DataFrame with sorted, unique x-values.
    """
    if y_cols is None:
        y_cols = []

    out = df.copy()
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    out = out[np.isfinite(out[x_col])].copy()
    if out.empty:
        return out

    out = out.sort_values(x_col).reset_index(drop=True)

    if out[x_col].duplicated().any():
        agg: Dict[str, str] = {x_col: "first"}
        for c in y_cols:
            if c not in out.columns:
                continue

            cl = c.lower()
            if cl.startswith("ph"):
                agg[c] = "median"
            elif cl.startswith("n_") or cl.endswith("_n") or cl in {"n_step", "n_tail"}:
                agg[c] = "sum"
            else:
                agg[c] = "mean"

        out = out.groupby(x_col, as_index=False).agg(agg)

    out = out.sort_values(x_col).reset_index(drop=True)
    return out


def extract_runs(
    df: pd.DataFrame,
    allow_time_fallback: bool = False,
) -> Dict[str, Dict]:
    """
    Extract individual experimental runs from a multi-run Logger Pro export.

    Logger Pro exports multiple runs in a single CSV with columns named
    'Run 1: Volume (cm³)', 'Run 1: pH', 'Run 2: Volume (cm³)', etc.
    This function separates them into individual DataFrames.

    Args:
        df: Raw DataFrame from load_titration_data().
        allow_time_fallback: If True, use time as x-axis when volume is
            unavailable (for drift analysis).

    Returns:
        Dictionary mapping run names to dictionaries containing:
            - 'df': Tidy DataFrame with 'Volume (cm³)', 'pH', 'Time (min)'
            - 'x_col': Name of the x-column to use for analysis
    """
    runs: Dict[str, Dict] = {}

    prefixes = {
        col.split(":")[0].strip()
        for col in df.columns
        if ":" in col and col.lower().startswith("run")
    }

    for prefix in prefixes:
        run_cols = [col for col in df.columns if col.startswith(prefix)]
        run_df = df[run_cols].copy()
        run_df.columns = [
            col.split(": ", 1)[1] if ": " in col else col.split(":", 1)[1]
            for col in run_cols
        ]

        rename_map = {
            "Volume of NaOH Added (cm³)": "Volume (cm³)",
            "Time (min)": "Time (min)",
            "pH": "pH",
            "Temperature (°C)": "Temperature (°C)",
        }
        run_df = run_df.rename(columns=rename_map)
        run_df = run_df.dropna(how="all")

        if "pH" not in run_df.columns:
            continue

        run_df["pH"] = pd.to_numeric(run_df["pH"], errors="coerce")

        if "Volume (cm³)" in run_df.columns:
            run_df["Volume (cm³)"] = pd.to_numeric(
                run_df["Volume (cm³)"], errors="coerce"
            ).ffill()

        if "Time (min)" in run_df.columns:
            run_df["Time (min)"] = pd.to_numeric(run_df["Time (min)"], errors="coerce")

        has_volume = (
            "Volume (cm³)" in run_df.columns
            and run_df["Volume (cm³)"].notna().any()
            and run_df["Volume (cm³)"].nunique(dropna=True) > 1
        )

        if has_volume:
            tidy = run_df.dropna(subset=["pH", "Volume (cm³)"]).reset_index(drop=True)
            runs[prefix] = {"df": tidy, "x_col": "Volume (cm³)"}
            continue

        if (
            allow_time_fallback
            and ("Time (min)" in run_df.columns)
            and run_df["Time (min)"].notna().any()
        ):
            tidy = run_df.dropna(subset=["pH", "Time (min)"]).reset_index(drop=True)
            if not tidy.empty and tidy["Time (min)"].nunique(dropna=True) > 1:
                runs[prefix] = {"df": tidy, "x_col": "Time (min)"}

    return runs


def aggregate_volume_steps(
    df: pd.DataFrame,
    volume_col: str = "Volume (cm³)",
    ph_col: str = "pH",
    volume_bin: Optional[float] = DEFAULT_VOLUME_BIN,
    auto_bin_if_needed: bool = False,
    time_col: Optional[str] = "Time (min)",
    tail_max: int = 10,
    tail_min: int = 3,
) -> pd.DataFrame:
    """
    Aggregate continuous pH readings into step-wise equilibrium values.

    At each volume increment during a titration, multiple pH readings are
    recorded as the system equilibrates. This function groups readings by
    volume and extracts representative equilibrium pH values.

    Equilibration Strategy:
        For each volume step, the function:
        1. Identifies all readings at that volume
        2. Takes the "tail" (final readings) to represent equilibrated pH
        3. Computes median of tail as the representative pH_step value

        The tail approach avoids transient mixing effects and electrode
        response delays that affect early readings at each step.

    Args:
        df: Raw titration data with continuous pH readings.
        volume_col: Name of volume column.
        ph_col: Name of pH column.
        volume_bin: If specified, rounds volumes to this resolution before
            grouping. Useful for noisy volume sensors.
        auto_bin_if_needed: If True, automatically infer binning resolution.
        time_col: Name of time column (used for drift analysis).
        tail_max: Maximum number of readings to include in tail.
        tail_min: Minimum number of readings to include in tail.

    Returns:
        DataFrame with columns:
            - 'Volume (cm³)': Unique volume values
            - 'pH_step': Representative equilibrium pH at each volume
            - 'pH_step_sd': Standard deviation of tail pH values
            - 'pH_drift_step': pH change during equilibration
            - 'pH_slope_step': Rate of pH change vs. time (if available)
            - 'n_step': Total number of readings at this volume
            - 'n_tail': Number of readings in the tail

    Note:
        Output is sorted by volume and guaranteed to have strictly
        increasing, unique volume values (required for PCHIP interpolation).
    """
    if volume_col not in df.columns or ph_col not in df.columns:
        return pd.DataFrame(
            columns=[
                volume_col,
                "pH_step",
                "pH_step_sd",
                "pH_drift_step",
                "pH_slope_step",
                "n_step",
                "n_tail",
            ]
        )

    keep_cols = [volume_col, ph_col]
    if time_col and time_col in df.columns:
        keep_cols.append(time_col)
    working = df[keep_cols].copy()
    working[volume_col] = pd.to_numeric(working[volume_col], errors="coerce").ffill()
    working[ph_col] = pd.to_numeric(working[ph_col], errors="coerce")
    if time_col and time_col in working.columns:
        working[time_col] = pd.to_numeric(working[time_col], errors="coerce")
    working = working.dropna(subset=[volume_col, ph_col])
    if working.empty:
        return pd.DataFrame(
            columns=[
                volume_col,
                "pH_step",
                "pH_step_sd",
                "pH_drift_step",
                "pH_slope_step",
                "n_step",
                "n_tail",
            ]
        )

    if volume_bin is None and auto_bin_if_needed:
        vol_res = _infer_recorded_resolution(working[volume_col])
        if vol_res is not None:
            volume_bin = vol_res

    working[volume_col] = _round_to_resolution(working[volume_col], volume_bin)

    records = []
    for vol, grp in working.groupby(volume_col, sort=True):
        ph_vals = grp[ph_col].dropna()
        if ph_vals.empty:
            continue

        n_total = int(len(ph_vals))
        n_tail = int(min(tail_max, max(tail_min, n_total // 3)))
        third = max(1, n_total // 3)
        first_third = ph_vals.head(third)
        tail = ph_vals.tail(n_tail)
        last_third = ph_vals.tail(third)

        ph_step = float(np.median(tail))
        ph_step_sd = float(np.std(tail, ddof=1)) if len(tail) > 1 else 0.0
        ph_drift = float(np.median(last_third) - np.median(first_third))

        ph_slope = np.nan
        if time_col and time_col in grp.columns:
            tvals = pd.to_numeric(grp[time_col], errors="coerce").to_numpy(dtype=float)
            pvals = ph_vals.to_numpy(dtype=float)
            mask = np.isfinite(tvals) & np.isfinite(pvals)
            if np.sum(mask) >= 2:
                tt = tvals[mask]
                pp = pvals[mask]
                try:
                    ph_slope = float(np.polyfit(tt, pp, 1)[0])
                except Exception:
                    ph_slope = np.nan

        records.append(
            {
                volume_col: float(vol),
                "pH_step": ph_step,
                "pH_step_sd": ph_step_sd,
                "pH_drift_step": ph_drift,
                "pH_slope_step": ph_slope,
                "n_step": n_total,
                "n_tail": n_tail,
            }
        )

    step_df = pd.DataFrame.from_records(records)
    if step_df.empty:
        return pd.DataFrame(
            columns=[
                volume_col,
                "pH_step",
                "pH_step_sd",
                "pH_drift_step",
                "pH_slope_step",
                "n_step",
                "n_tail",
            ]
        )

    step_df = step_df.sort_values(volume_col).reset_index(drop=True)

    step_df = _ensure_strictly_increasing_unique(
        step_df,
        volume_col,
        y_cols=[
            "pH_step",
            "pH_step_sd",
            "pH_drift_step",
            "pH_slope_step",
            "n_step",
            "n_tail",
        ],
    )
    return step_df


def load_titration_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)
