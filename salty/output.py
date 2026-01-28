"""Output helpers for saving analysis results."""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def save_data_to_csv(
    results_df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: str = "output"
) -> Tuple[str, str]:
    """Save processed data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "individual_results.csv")
    stats_path = os.path.join(output_dir, "statistical_summary.csv")

    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved individual results to {results_path}")
    print(f"Saved statistical summary to {stats_path}")

    return results_path, stats_path
