"""
Handles CSV parsing, run extraction, and derivative calculations.
"""

# Algorithm summary: parse wide Logger Pro CSVs, forward-fill cumulative
# volume, aggregate step-wise median pH values, optionally smooth with
# Savitzky–Golay when available, and compute derivatives for equivalence
# detection and pKa estimation.

import importlib.util

import numpy as np
import pandas as pd

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
if HAVE_SCIPY:
    from scipy.signal import savgol_filter


def extract_runs(df):
    """Extract individual titration runs from the Logger Pro export.

    The raw CSVs are in a wide format with columns such as
    ``Run 1: Time (min)``, ``Run 1: pH`` and ``Run 1: Volume of NaOH Added (cm³)``.
    We reshape each run into a tidy DataFrame with monotonically increasing
    cumulative volume so downstream calculations (derivatives with respect to
    volume, interpolation at half-equivalence, etc.) behave properly.

    For runs lacking volume data, we fall back to using time as the independent
    variable for derivative calculations.

    Args:
        df: Raw wide-format :class:`pandas.DataFrame` loaded directly from the CSV.

    Returns:
        dict[str, dict]: Mapping of run name to dict with 'df' (DataFrame) and
        'x_col' (str, either "Volume (cm³)" or "Time (min)").
    """

    runs = {}
    prefixes = {
        col.split(":")[0].strip()
        for col in df.columns
        if ":" in col and col.lower().startswith("run")
    }

    for prefix in prefixes:
        run_cols = [col for col in df.columns if col.startswith(prefix)]
        run_df = df[run_cols].copy()
        run_df.columns = [col.split(": ")[1] for col in run_cols]

        # Normalise column names we care about.
        rename_map = {
            "Volume of NaOH Added (cm³)": "Volume (cm³)",
            "Temperature (°C)": "Temperature (°C)",
            "Time (min)": "Time (min)",
            "pH": "pH",
        }
        run_df = run_df.rename(columns=rename_map)

        # Drop rows that are entirely empty, then forward-fill the manually
        # entered cumulative volume so each pH reading has a corresponding
        # volume value.
        run_df = run_df.dropna(how="all")
        if "Volume (cm³)" in run_df.columns:
            run_df["Volume (cm³)"] = (
                pd.to_numeric(run_df["Volume (cm³)"], errors="coerce").ffill()
            )

        # Coerce numeric for pH and time.
        run_df["pH"] = pd.to_numeric(run_df["pH"], errors="coerce")
        run_df["Time (min)"] = pd.to_numeric(run_df["Time (min)"], errors="coerce")

        # Determine the independent variable: prefer volume, fall back to time.
        if ("Volume (cm³)" in run_df.columns and 
            run_df["Volume (cm³)"].notna().any() and 
            run_df["Volume (cm³)"].nunique() > 1):
            x_col = "Volume (cm³)"
            subset_cols = ["pH", "Volume (cm³)"]
        else:
            x_col = "Time (min)"
            subset_cols = ["pH", "Time (min)"]

        run_df = run_df.dropna(subset=subset_cols)

        if not run_df.empty:
            runs[prefix] = {"df": run_df.reset_index(drop=True), "x_col": x_col}

    return runs


def _choose_savgol_window(n_points, min_window=5):
    """Choose a Savitzky–Golay window length based on dataset size."""

    if n_points < min_window:
        return None
    candidate = max(min_window, int(n_points // 3))
    if candidate % 2 == 0:
        candidate += 1
    max_window = max(min_window, int(n_points // 2))
    if max_window % 2 == 0:
        max_window -= 1
    if candidate > max_window:
        candidate = max_window
    return candidate if candidate >= min_window else None


def aggregate_volume_steps(df, volume_col="Volume (cm³)", ph_col="pH"):
    """Aggregate raw readings into volume steps with robust equilibrium pH.

    For each constant (forward-filled) volume segment, compute:
      - pH_step: median of the last N readings (N adapts with segment length)
      - pH_step_sd: standard deviation over those last N readings
      - n_step: number of raw readings in the segment
    """

    if volume_col not in df.columns or ph_col not in df.columns:
        return pd.DataFrame(columns=[volume_col, "pH_step", "pH_step_sd", "n_step"])

    working = df[[volume_col, ph_col]].copy()
    working[volume_col] = pd.to_numeric(working[volume_col], errors="coerce").ffill()
    working[ph_col] = pd.to_numeric(working[ph_col], errors="coerce")
    working = working.dropna(subset=[volume_col, ph_col])

    records = []
    for volume, group in working.groupby(volume_col, sort=False):
        ph_values = group[ph_col].dropna()
        if ph_values.empty:
            continue
        n_total = len(ph_values)
        n_tail = min(10, max(3, n_total // 3))
        tail_values = ph_values.tail(n_tail)
        ph_step = float(np.median(tail_values))
        # For a single reading, the sample standard deviation with ddof=1 would be undefined (NaN),
        # so we treat its uncertainty as 0.0 (equivalent to the population std with ddof=0).
        ph_step_sd = float(np.std(tail_values, ddof=1)) if len(tail_values) > 1 else 0.0
        records.append(
            {
                volume_col: float(volume),
                "pH_step": ph_step,
                "pH_step_sd": ph_step_sd,
                "n_step": int(n_total),
            }
        )

    step_df = pd.DataFrame.from_records(records)
    if not step_df.empty:
        step_df = step_df.sort_values(volume_col).reset_index(drop=True)
    return step_df


def calculate_derivatives(
    df,
    x_col="Volume (cm³)",
    ph_col="pH_step",
    smooth=True,
    polyorder=2,
):
    """Compute derivatives with respect to the independent variable.

    Using cumulative volume as the independent variable keeps the calculated
    equivalence point aligned with the experimentally determined Veq (~25 mL)
    and supports direct interpolation of the pH at half-equivalence volume.

    Args:
        df: DataFrame containing at least the columns specified by ``x_col``
            (default ``"Volume (cm³)"``) and ``ph_col`` (default ``"pH"``).
        x_col: Column name for the independent variable.
        ph_col: Column name for the measured pH values.
        polyorder: Polynomial order for the Savitzky–Golay filter.

    The method respects uneven spacing in ``x_col`` and uses optional
    Savitzky–Golay smoothing with an adaptive window length chosen
    automatically based on the dataset size.

    Returns:
        pd.DataFrame: The original DataFrame with added ``pH_smooth``,
        ``dpH/dx`` and ``d2pH/dx2`` columns.
    """

    x = df[x_col].values
    ph = df[ph_col].values

    window_length = _choose_savgol_window(len(ph))
    if smooth and HAVE_SCIPY and window_length is not None and window_length > polyorder:
        ph_smooth = savgol_filter(ph, window_length, polyorder)
    else:
        ph_smooth = ph

    dpH = np.gradient(ph_smooth, x)
    d2pH = np.gradient(dpH, x)

    df["pH_smooth"] = ph_smooth
    df["dpH/dx"] = dpH
    df["d2pH/dx2"] = d2pH

    return df


def load_titration_data(filepath):
    """
    Load titration data from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filepath)
