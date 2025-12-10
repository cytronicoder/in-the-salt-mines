"""
Data processing module for titration analysis.
Handles CSV parsing, run extraction, and derivative calculations.

Functions:
    extract_runs(df): Extracts individual runs from the wide-format CSV.
    calculate_derivatives(df, time_col="Time (min)", ph_col="pH"): Calculates 1st and 2nd derivatives of pH w.r.t Time.
    load_titration_data(filepath): Loads titration data from a CSV file.
"""

import numpy as np
import pandas as pd


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


def calculate_derivatives(df, time_col="Time (min)", ph_col="pH"):
    """
    Calculates 1st and 2nd derivatives of pH w.r.t Time.

    Args:
        df (pd.DataFrame): DataFrame with time and pH data.
        time_col (str, optional): Name of the time column (default: 'Time (min)').
        ph_col (str, optional): Name of the pH column (default: 'pH').

    Returns:
        pd.DataFrame: Input DataFrame with added 'dpH/dt' and 'd2pH/dt2' columns.
    """
    t = df[time_col].values
    ph = df[ph_col].values
    dpH = np.gradient(ph, t)
    d2pH = np.gradient(dpH, t)

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
