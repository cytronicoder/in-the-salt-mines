"""
Handles CSV parsing, run extraction, and derivative calculations.
"""

import numpy as np
import pandas as pd
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
        if "Volume (cm³)" in run_df.columns and run_df["Volume (cm³)"].notna().any():
            x_col = "Volume (cm³)"
            subset_cols = ["pH", "Volume (cm³)"]
        else:
            x_col = "Time (min)"
            subset_cols = ["pH", "Time (min)"]

        run_df = run_df.dropna(subset=subset_cols)

        # Sort by the independent variable and drop duplicates.
        run_df = run_df.sort_values(x_col)
        run_df = run_df.drop_duplicates(subset=x_col, keep="last")

        if not run_df.empty:
            runs[prefix] = {"df": run_df.reset_index(drop=True), "x_col": x_col}

    return runs


def calculate_derivatives(
    df,
    x_col="Volume (cm³)",
    ph_col="pH",
    window_length=15,
    polyorder=3,
):
    """Smooth the pH trace and compute derivatives with respect to volume.

    Using cumulative volume as the independent variable keeps the calculated
    equivalence point aligned with the experimentally determined Veq (~25 mL)
    and supports direct interpolation of the pH at half-equivalence volume.

    Args:
        df: DataFrame containing at least the columns specified by ``x_col``
            (default ``"Volume (cm³)"``) and ``ph_col`` (default ``"pH"``).
        x_col: Column name for the independent variable.
        ph_col: Column name for the measured pH values.
        window_length: Savitzky–Golay window length for smoothing.
        polyorder: Polynomial order for the Savitzky–Golay filter.

    Returns:
        pd.DataFrame: The original DataFrame with added ``pH_smooth``,
        ``dpH/dx`` and ``d2pH/dx2`` columns.
    """

    x = df[x_col].values
    ph = df[ph_col].values

    # Ensure the smoothing window is valid for the data length.
    if window_length >= len(ph):
        window_length = len(ph) - 1 if len(ph) % 2 == 0 else len(ph)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    ph_smooth = savgol_filter(ph, window_length, polyorder) if len(ph) > window_length else ph

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
