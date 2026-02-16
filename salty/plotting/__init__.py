"""Plotting utilities for titration analysis."""

from .ia_figures import generate_ia_figure_set
from .qc_plots import (
    generate_all_qc_plots,
    plot_buffer_region_coverage,
    plot_equivalence_volumes,
    plot_hh_slope_diagnostics,
    plot_initial_ph_by_concentration,
    plot_initial_ph_scatter,
    plot_pka_precision,
    plot_residuals_analysis,
    plot_temperature_and_calibration_qc,
    plot_temperature_boxplots,
)
from .summary_plots import (
    plot_hh_linearization_and_diagnostics,
    plot_pka_app_vs_nacl_and_I,
    plot_statistical_summary,
)
from .titration_plots import (
    plot_derivative_equivalence_by_nacl,
    plot_titration_curves,
    plot_titration_overlays_by_nacl,
    setup_plot_style,
)

__all__ = [
    "generate_ia_figure_set",
    "plot_titration_curves",
    "plot_titration_overlays_by_nacl",
    "plot_derivative_equivalence_by_nacl",
    "plot_statistical_summary",
    "plot_pka_app_vs_nacl_and_I",
    "plot_hh_linearization_and_diagnostics",
    "setup_plot_style",
    "generate_all_qc_plots",
    "plot_initial_ph_by_concentration",
    "plot_initial_ph_scatter",
    "plot_temperature_boxplots",
    "plot_temperature_and_calibration_qc",
    "plot_equivalence_volumes",
    "plot_hh_slope_diagnostics",
    "plot_pka_precision",
    "plot_buffer_region_coverage",
    "plot_residuals_analysis",
]
