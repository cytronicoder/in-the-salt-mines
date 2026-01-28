"""
Publication-quality plotting utilities for titration analysis.

This subpackage generates black-and-white figures suitable for academic
reports and publications. All plotting functions accept precomputed results
and do not perform chemistry calculations.

Modules:
    titration_plots:
        Three-panel figures for individual titration runs:
        (1) pH vs. Volume with interpolation curve
        (2) First derivative (d(pH)/dV) for equivalence detection
        (3) Henderson-Hasselbalch diagnostic plot

    summary_plots:
        Statistical summary showing pKa_app vs. NaCl concentration
        with error bars representing systematic uncertainties.

Design Principles:
    1. No chemistry calculations in plotting code. Functions receive
       precomputed values and simply render them.

    2. Black-and-white output for journal compatibility.

    3. Input validation with explicit KeyError for missing required fields.

Styling:
    Uses serif fonts (Times New Roman family), 300 DPI output, and
    suppressed top/right spines for clean academic appearance.
"""

from .summary_plots import plot_statistical_summary
from .titration_plots import plot_titration_curves, setup_plot_style

__all__ = ["plot_titration_curves", "plot_statistical_summary", "setup_plot_style"]
