"""
Output utilities for saving titration analysis results to CSV files.

Provides functions for persisting individual run results and statistical
summaries to disk in a standardized format suitable for further analysis
or import into spreadsheet applications.

Output Files:
    individual_results.csv:
        One row per experimental run containing pKa_app, V_eq, uncertainties,
        QC status, and regression diagnostics.

    statistical_summary.csv:
        One row per NaCl concentration containing mean pKa_app with
        systematic (half-range) uncertainty.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def save_data_to_csv(
    results_df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: str = "output"
) -> Tuple[str, str]:
    """
    Save analysis results and statistical summary to CSV files.

    Args:
        results_df: Individual run results from create_results_dataframe().
        stats_df: Statistical summary from calculate_statistics().
        output_dir: Directory for output files (created if necessary).

    Returns:
        Tuple of (individual_results_path, statistical_summary_path).

    Note:
        Prints confirmation messages to stdout with file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "individual_results.csv")
    stats_path = os.path.join(output_dir, "statistical_summary.csv")

    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved individual results to {results_path}")
    print(f"Saved statistical summary to {stats_path}")

    return results_path, stats_path
