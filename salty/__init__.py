"""
A Python package for analyzing titration data to determine half-equivalence points
and apparent pKa values of acids under different ionic strength conditions. I
created this package to streamline the analysis of my titration experiments
for my DP Chemistry SL IA project.

Modules:
    - data_processing: Functions for loading and processing titration data
    - analysis: Functions for analyzing titration runs and calculating pKa
    - plotting: Functions for creating professional figures and saving results
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
from .data_processing import calculate_derivatives, extract_runs, load_titration_data
from .plotting import (
    plot_statistical_summary,
    plot_titration_curves,
    save_data_to_csv,
    setup_plot_style,
)

__all__ = [
    # Data processing
    "extract_runs",
    "calculate_derivatives",
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
