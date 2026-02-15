"""Prepare raw titration exports for IA-ready analytical processing.

This module maps directly to the front end of the IA method:
- import raw Logger Pro CSV exports,
- normalize headers and units,
- extract run-specific tables, and
- aggregate repeated readings into step-level equilibrium summaries.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


DEFAULT_VOLUME_BIN: Optional[float] = None


def _strip_uncertainty_from_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove uncertainty annotations from column headers.

    Column headers may contain uncertainty information in parentheses,
    e.g., 'Run 1: pH (±0.3 pH)'. This function strips that annotation
    to restore the original column names expected by the processing pipeline.

    Args:
        df (pandas.DataFrame): Table with possibly annotated column headers.

    Returns:
        pandas.DataFrame: Table with uncertainty suffixes removed from headers.
    """
    cleaned_columns = {}
    for col in df.columns:
        cleaned = re.sub(r"\s*\(±[^)]+\)\s*$", "", col).strip()
        if cleaned != col:
            cleaned_columns[col] = cleaned
            logger.debug("Stripped uncertainty from column: '%s' -> '%s'", col, cleaned)

    if cleaned_columns:
        df = df.rename(columns=cleaned_columns)

    return df


def _round_to_resolution(x: pd.Series, res: Optional[float]) -> pd.Series:
    """Round numeric values to a specified measurement resolution.

    Args:
        x (pandas.Series): Numeric values to round (typically volume in cm^3).
        res (float | None): Resolution/bin width in the same unit as ``x``.

    Returns:
        pandas.Series: Rounded values. Return unchanged values when ``res`` is
        ``None`` or non-positive.
    """
    if res is None or res <= 0:
        return x
    return (np.round(x / res) * res).astype(float)


def _infer_recorded_resolution(x: pd.Series) -> Optional[float]:
    """Infer recording resolution from observed positive step increments.

    Args:
        x (pandas.Series): Observed recorded values (typically volume in cm^3).

    Returns:
        float | None: Best-matching candidate resolution in cm^3, or ``None``
        when inference is not reliable.

    Note:
        Candidate set is ``[0.01, 0.02, 0.05, 0.10, 0.20]`` cm^3 and requires
        relative mismatch <= 35%.
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
    """Enforce strictly increasing unique x-values for downstream analysis.

    Args:
        df (pandas.DataFrame): Input dataframe.
        x_col (str): Independent-variable column (for example, ``Volume (cm^3)``).
        y_cols (list[str] | None): Dependent/metadata columns to aggregate when
            duplicate ``x_col`` values are found.

    Returns:
        pandas.DataFrame: Sorted dataframe with unique ``x_col`` values.

    Note:
        Apply median aggregation for pH-like columns, sum for count-like columns,
        and mean for other columns.
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


def extract_runs(df: pd.DataFrame) -> Dict[str, Dict]:
    """Extract per-run titration data using explicit volume axes.

    Logger Pro exports multiple runs in a single CSV with columns named
    ``Run 1: Volume (cm^3)``, ``Run 1: pH``, and analogous columns for each run.
    This function returns a tidy DataFrame per run and enforces the presence
    of volume data for downstream chemical analysis.

    Args:
        df (pandas.DataFrame): Raw Logger Pro table from
            ``load_titration_data``.

    Returns:
        dict[str, dict]: Mapping from run prefix (for example ``Run 1``) to a
        dictionary with ``df`` (tidy run table) and ``x_col``
        (independent-variable label, usually ``Volume (cm^3)``).

    Raises:
        ValueError: If pH values exist for a run but no valid volume axis
            (cm^3) is available.

    Note:
        Uncertainty annotations in column headers are stripped before run parsing.
        IA correspondence: this function enforces the requirement that each run
        has paired volume and pH data before chemical interpretation.

    References:
        Logger Pro multi-run CSV structure conventions.
    """
    df = _strip_uncertainty_from_columns(df)

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
            "Volume of NaOH Added (cm^3)": "Volume (cm^3)",
            "Time (min)": "Time (min)",
            "pH": "pH",
            "Temperature (°C)": "Temperature (°C)",
        }
        run_df = run_df.rename(columns=rename_map)
        run_df = run_df.dropna(how="all")

        if "pH" not in run_df.columns:
            continue

        run_df["pH"] = pd.to_numeric(run_df["pH"], errors="coerce")

        if "Volume (cm^3)" in run_df.columns:
            run_df["Volume (cm^3)"] = pd.to_numeric(
                run_df["Volume (cm^3)"], errors="coerce"
            ).ffill()

        if "Time (min)" in run_df.columns:
            run_df["Time (min)"] = pd.to_numeric(run_df["Time (min)"], errors="coerce")

        has_volume = (
            "Volume (cm^3)" in run_df.columns
            and run_df["Volume (cm^3)"].notna().any()
            and run_df["Volume (cm^3)"].nunique(dropna=True) > 1
        )

        if has_volume:
            tidy = run_df.dropna(subset=["pH", "Volume (cm^3)"]).reset_index(drop=True)
            if tidy.empty:
                n_vol = int(run_df["Volume (cm^3)"].notna().sum())
                msg = (
                    "Run '%s' contains a Volume (cm^3) axis but no paired "
                    "pH readings; skipping (%d volume entries)."
                )
                logger.warning(msg, prefix, n_vol)
                continue
            runs[prefix] = {"df": tidy, "x_col": "Volume (cm^3)"}
            continue

        if run_df["pH"].notna().any():
            raise ValueError(
                f"Run '{prefix}' contains pH data without a valid Volume (cm^3) axis."
            )

    return runs


def aggregate_volume_steps(
    df: pd.DataFrame,
    volume_col: str = "Volume (cm^3)",
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
        df (pandas.DataFrame): Raw run table containing repeated pH
            measurements at each delivered volume.
        volume_col (str, optional): Delivered-volume column name in cm^3.
            Defaults to ``"Volume (cm^3)"``.
        ph_col (str, optional): pH measurement column name (pH units).
            Defaults to ``"pH"``.
        volume_bin (float | None, optional): Optional volume rounding/binning
            width in cm^3. Defaults to ``DEFAULT_VOLUME_BIN``.
        auto_bin_if_needed (bool, optional): If ``True`` and ``volume_bin`` is
            ``None``, infer a bin width from observed volume spacing.
            Defaults to ``False``.
        time_col (str | None, optional): Optional time column in minutes used
            for per-step drift slope estimation. Defaults to ``"Time (min)"``.
        tail_max (int, optional): Maximum number of trailing readings per step
            used for equilibrium summary. Defaults to ``10``.
        tail_min (int, optional): Minimum number of trailing readings per step
            used for equilibrium summary. Defaults to ``3``.

    Returns:
        pandas.DataFrame: Step-aggregated table with one row per volume step
        and columns including ``pH_step`` (pH units), ``pH_step_sd`` (pH
        units), ``pH_drift_step`` (pH units), ``pH_slope_step`` (pH min^-1),
        ``n_step`` (count), and ``n_tail`` (count).

    Raises:
        ValueError: If required input columns are missing.

    Note:
        ``pH_step`` is computed as the median of a tail window to reduce mixing
        transients and electrode lag effects.
        IA correspondence: this represents the equilibrium-reading reduction
        step applied before derivative endpoint detection and buffer modeling.

    References:
        Equilibration-tail summarization for potentiometric titration traces.
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

    Column headers may contain uncertainty annotations (e.g., '±0.3 pH').
    These annotations are preserved at the loading stage and will be
    stripped during extraction if present.

    Args:
        filepath (str): CSV file path.

    Returns:
        pandas.DataFrame: Raw Logger Pro table.

    Raises:
        FileNotFoundError: If the path does not exist.
        pandas.errors.ParserError: If CSV parsing fails.

    Note:
        Header normalization and uncertainty-annotation stripping are performed in
        downstream extraction utilities, not during file load.
        IA correspondence: this is the raw data import boundary for provenance.

    References:
        pandas CSV I/O behavior for tabular scientific exports.
    """
    return pd.read_csv(filepath)
