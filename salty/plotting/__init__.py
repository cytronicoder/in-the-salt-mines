"""Publication-quality plotting utilities for titration analysis.

This subpackage renders black-and-white figures suitable for academic reports
and manuscripts. Plotting functions accept validated outputs from the analysis
pipeline and do not perform chemistry or regression internally.
"""

from .summary_plots import plot_statistical_summary
from .titration_plots import plot_titration_curves, setup_plot_style

__all__ = ["plot_titration_curves", "plot_statistical_summary", "setup_plot_style"]
