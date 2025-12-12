"""
Handles CSV parsing, run extraction, and derivative calculations.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def extract_runs(df):
    """
    Extracts individual runs from the wide-format CSV.

    Args:
        df (pd.DataFrame): Wide-format DataFrame with columns like 'Run 1: Time (min)', 'Run 1: pH', etc.

    Returns:
        dict: Dictionary mapping run names to DataFrames with 'Time (min)' and 'pH' columns.
    """
    runs = {}
    columns = df.columns
    prefixes = set([col.split(":")[0] for col in columns if ":" in col])

    for prefix in prefixes:
        run_cols = [col for col in columns if col.startswith(prefix)]
        run_df = df[run_cols].copy()

        run_df.columns = [col.split(": ")[1] for col in run_cols]
        run_df = run_df.dropna(how="all")

        if "Time (min)" in run_df.columns and "pH" in run_df.columns:
            run_df = run_df.dropna(subset=["Time (min)", "pH"])
            run_df = run_df.sort_values("Time (min)")

            runs[prefix] = run_df

    return runs


def calculate_derivatives(
    df, time_col="Time (min)", ph_col="pH", window_length=15, polyorder=3
):
    """
    Calculates 1st and 2nd derivatives of pH w.r.t Time using Savitzky-Golay smoothing.

    Args:
        df (pd.DataFrame): DataFrame with time and pH data.
        time_col (str, optional): Name of the time column (default: 'Time (min)').
        ph_col (str, optional): Name of the pH column (default: 'pH').
        window_length (int, optional): Window length for Savitzky-Golay filter (default: 15).
        polyorder (int, optional): Polynomial order for Savitzky-Golay filter (default: 3).

    Returns:
        pd.DataFrame: Input DataFrame with added 'pH_smooth', 'dpH/dt', and 'd2pH/dt2' columns.
    """
    t = df[time_col].values
    ph = df[ph_col].values

    if window_length >= len(ph):
        window_length = len(ph) - 1 if len(ph) % 2 == 0 else len(ph)
    if window_length % 2 == 0:
        window_length -= 1
    if window_length < polyorder + 2:
        window_length = polyorder + 2
        if window_length % 2 == 0:
            window_length += 1

    if len(ph) > window_length:
        ph_smooth = savgol_filter(ph, window_length, polyorder)
    else:
        ph_smooth = ph

    dpH = np.gradient(ph_smooth, t)
    d2pH = np.gradient(dpH, t)

    df["pH_smooth"] = ph_smooth
    df["dpH/dt"] = dpH
    df["d2pH/dt2"] = d2pH

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
