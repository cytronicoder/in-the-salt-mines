"""Apparent pKa analysis under controlled ionic strength."""

__version__ = "1.0.0"
__author__ = "Zeyu (Peter) Yao"

from .analysis import (
    analyze_titration,
    build_summary_plot_data,
    calculate_statistics,
    create_results_dataframe,
    detect_equivalence_point,
    print_statistics,
    process_all_files,
)
from .data_processing import extract_runs, load_titration_data
from .output import save_data_to_csv
from .plotting import plot_statistical_summary, plot_titration_curves, setup_plot_style
from .uncertainty import (
    burette_delivered_uncertainty,
    combine_uncertainties,
    round_value_to_uncertainty,
)

__all__ = [
    "extract_runs",
    "load_titration_data",
    "detect_equivalence_point",
    "analyze_titration",
    "process_all_files",
    "create_results_dataframe",
    "calculate_statistics",
    "build_summary_plot_data",
    "setup_plot_style",
    "plot_titration_curves",
    "plot_statistical_summary",
    "save_data_to_csv",
]
