"""Public API for the apparent pKa analysis package."""

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
