"""
A Python package for analyzing weak acid-strong base titration data.

Determines equivalence points and apparent pKa values under varying ionic strength conditions.

Modules:
    - data_processing: Loads and processes titration data from CSV files.
    - analysis: Analyzes titration runs, calculates pKa, and estimates uncertainties.
    - plotting: Creates publication-quality plots and saves results.
    - uncertainty: Handles uncertainty propagation using IB DP rules.
"""

__version__ = "1.0.0"
__author__ = "Zeyu (Peter) Yao"

from .analysis import (
    analyze_titration,
    calculate_statistics,
    create_results_dataframe,
    detect_equivalence_point,
    print_statistics,
    process_all_files,
)
from .data_processing import extract_runs, load_titration_data
from .plotting import (
    plot_statistical_summary,
    plot_titration_curves,
    save_data_to_csv,
    setup_plot_style,
)
from .uncertainty import (
    burette_delivered_uncertainty,
    combine_uncertainties,
    round_value_to_uncertainty,
)

__all__ = [
    # Data processing
    "extract_runs",
    "load_titration_data",
    # Analysis
    "detect_equivalence_point",
    "analyze_titration",
    "process_all_files",
    "create_results_dataframe",
    "calculate_statistics",
    # Plotting
    "setup_plot_style",
    "plot_titration_curves",
    "plot_statistical_summary",
    "save_data_to_csv",
]
