"""Prepare Logger Pro titration exports for two-stage pKa_app analysis.

This module converts raw, time-resolved titration measurements into the
step-wise equilibrium format required by the two-stage apparent pKa protocol.
It enforces explicit volume-based data handling and provides deterministic,
traceable aggregation rules for chemically meaningful processing.

The workflow is:
    1) Load the CSV export into a DataFrame.
    2) Split multi-run exports into individual runs with explicit volume axes.
    3) Aggregate each run into equilibrium pH values per volume step.

These utilities are strictly data preparation tools. They do not perform
chemistry, regression, or plotting, and they raise explicit exceptions when
required experimental metadata are missing.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_VOLUME_BIN: Optional[float] = None


def _round_to_resolution(x: pd.Series, res: Optional[float]) -> pd.Series:
    """Round values to a specified resolution (bin width)."""
    if res is None or res <= 0:
        return x
    return (np.round(x / res) * res).astype(float)


def _infer_recorded_resolution(x: pd.Series) -> Optional[float]:
    """Infer the recorded volume resolution from observed step sizes."""
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
    """Return a DataFrame with strictly increasing, unique x-values."""
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


def extract_runs(df: pd.DataFrame) -> Dict[str, Dict]:
    """Extract per-run titration data using explicit volume axes.

    Logger Pro exports multiple runs in a single CSV with columns named
    ``Run 1: Volume (cm³)``, ``Run 1: pH``, and analogous columns for each run.
    This function returns a tidy DataFrame per run and enforces the presence
    of volume data for downstream chemical analysis.

    Args:
        df: Raw DataFrame obtained from ``load_titration_data``.

    Returns:
        A mapping from run identifiers to dictionaries containing:
            - ``df``: DataFrame with ``Volume (cm³)`` and ``pH`` columns.
            - ``x_col``: The explicit x-axis column name (always ``Volume (cm³)``).

    Raises:
        ValueError: If a run contains pH data without a valid volume axis.
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

        if run_df["pH"].notna().any():
            raise ValueError(
                f"Run '{prefix}' contains pH data without a valid Volume (cm³) axis."
            )

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
    """Aggregate continuous pH readings into equilibrium step values.

    The raw Logger Pro export records multiple pH readings while the solution
    equilibrates at each burette volume. This function groups those readings
    by volume and summarizes the equilibrium pH using the median of the final
    ("tail") measurements, which mitigates transient mixing and electrode lag.

    Args:
        df: Raw titration data containing continuous pH readings.
        volume_col: Column name for the delivered volume (independent variable).
        ph_col: Column name for the measured pH values.
        volume_bin: Optional rounding resolution applied before grouping. This
            should match the recording resolution of the burette.
        auto_bin_if_needed: Whether to infer a sensible bin width from the
            recorded volume increments when no explicit bin is provided.
        time_col: Optional time column used to estimate equilibration drift.
        tail_max: Maximum number of readings in the tail window per step.
        tail_min: Minimum number of readings required in the tail window.

    Returns:
        A DataFrame containing one row per volume step with equilibrium pH
        statistics and metadata.

    Raises:
        ValueError: If the required volume or pH columns are absent.
    """
    if volume_col not in df.columns or ph_col not in df.columns:
        raise ValueError(
            f"Required columns missing for aggregation: {volume_col}, {ph_col}."
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
    """Load a Logger Pro CSV export into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file on disk.

    Returns:
        A DataFrame containing the raw Logger Pro export.

    Raises:
        FileNotFoundError: If the file path does not exist.
        pandas.errors.ParserError: If the CSV cannot be parsed.
    """
    return pd.read_csv(filepath)
