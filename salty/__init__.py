"""Public API for the apparent pKa analysis package.

Research Question:
    How does [NaCl], varied from 0.00 to 0.80 mol dm^-3 in 0.20 mol dm^-3 intervals,
    affect pH at the half-equivalence point (and thus pKa_app) for 0.10 mol dm^-3
    ethanoic acid titrated with 0.10 mol dm^-3 NaOH, measured at 26 ± 1°C?

Key Concepts:
    - Ionic strength (μ): For NaCl, μ ≈ [NaCl] (1:1 electrolyte)
    - Activity coefficients (γ): Deviate from unity at higher μ
    - Apparent pKa (pKa_app): pH measured at half-equivalence point
    - Two-stage analysis: (1) Coarse estimate, (2) H-H regression
"""

__version__ = "1.0.0"
__author__ = "Zeyu (Peter) Yao"

from .analysis import (
    analyze_titration,
    build_summary_plot_data,
    calculate_statistics,
    create_results_dataframe,
    process_all_files,
)
from .output import save_data_to_csv
from .plotting import plot_statistical_summary, plot_titration_curves, setup_plot_style

__all__ = [
    "analyze_titration",
    "process_all_files",
    "create_results_dataframe",
    "calculate_statistics",
    "build_summary_plot_data",
    "plot_titration_curves",
    "plot_statistical_summary",
    "save_data_to_csv",
    "setup_plot_style",
]
