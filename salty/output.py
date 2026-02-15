"""Write analysis outputs and provenance tables to reproducible CSV files.

This module is the reporting/output boundary between in-memory analysis and
submission-ready tabular artifacts.
"""

from __future__ import annotations

import os
from typing import Tuple

import pandas as pd

from .reporting import add_formatted_reporting_columns, add_uncertainty_form_columns


def _build_provenance_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build processed-to-raw provenance mapping table.

    Args:
        results_df (pandas.DataFrame): Run-level results table that may include
            ``Run``, ``Source File``, and ``NaCl Concentration (M)`` columns.

    Returns:
        pandas.DataFrame: Provenance table with columns ``Processed Run``,
        ``Raw Source File``, and ``NaCl Concentration (M)`` when available.
    """
    cols = ["Run", "Source File", "NaCl Concentration (M)"]
    available = [c for c in cols if c in results_df.columns]
    if not available:
        return pd.DataFrame(columns=["Run", "Source File", "NaCl Concentration (M)"])
    return (
        results_df[available]
        .copy()
        .rename(
            columns={
                "Run": "Processed Run",
                "Source File": "Raw Source File",
                "NaCl Concentration (M)": "NaCl Concentration (M)",
            }
        )
    )


def save_data_to_csv(
    results_df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: str = "output"
) -> Tuple[str, str]:
    """Save per-run results and summary statistics to CSV files.

    Args:
        results_df (pandas.DataFrame): Output from
            ``create_results_dataframe``.
        stats_df (pandas.DataFrame): Output from ``calculate_statistics``.
        output_dir (str): Directory where CSV outputs are written.

    Returns:
        tuple[str, str]: Paths to ``individual_results.csv`` and
        ``statistical_summary.csv``.

    Note:
        Also write ``provenance_map.csv`` in the same output directory.
        IA correspondence: these exports are the canonical tables used in the
        processed-data and appendix sections.
    """
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "individual_results.csv")
    stats_path = os.path.join(output_dir, "statistical_summary.csv")

    results_report = add_formatted_reporting_columns(
        results_df,
        [
            ("Apparent pKa", "Uncertainty in Apparent pKa"),
            ("Veq (used)", "Veq uncertainty (ΔVeq)"),
        ],
    )
    results_report = add_uncertainty_form_columns(
        results_report,
        [
            ("Apparent pKa", "Uncertainty in Apparent pKa"),
            ("Veq (used)", "Veq uncertainty (ΔVeq)"),
        ],
    )

    stats_report = add_formatted_reporting_columns(
        stats_df,
        [("Mean Apparent pKa", "Uncertainty")],
    )
    stats_report = add_uncertainty_form_columns(
        stats_report,
        [("Mean Apparent pKa", "Uncertainty")],
    )

    results_report.to_csv(results_path, index=False)
    stats_report.to_csv(stats_path, index=False)

    provenance_path = os.path.join(output_dir, "provenance_map.csv")
    _build_provenance_table(results_df).to_csv(provenance_path, index=False)

    print(f"Saved individual results to {results_path}")
    print(f"Saved statistical summary to {stats_path}")
    print(f"Saved provenance mapping to {provenance_path}")

    return results_path, stats_path
