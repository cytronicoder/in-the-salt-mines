"""Persist titration analysis results to structured CSV outputs.

This module contains simple file-output utilities that serialize validated
analysis results and statistical summaries. It performs no chemistry or
regression and assumes that inputs already comply with the two-stage pKa_app
framework.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def save_data_to_csv(
    results_df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: str = "output"
) -> Tuple[str, str]:
    """Save per-run results and summary statistics to CSV files.

    Args:
        results_df: Output from ``create_results_dataframe``.
        stats_df: Output from ``calculate_statistics``.
        output_dir: Directory in which to write the CSV files.

    Returns:
        A tuple containing the paths to ``individual_results.csv`` and
        ``statistical_summary.csv``.
    """
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "individual_results.csv")
    stats_path = os.path.join(output_dir, "statistical_summary.csv")

    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved individual results to {results_path}")
    print(f"Saved statistical summary to {stats_path}")

    return results_path, stats_path
