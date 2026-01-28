"""
Salty: A Python package for acid-base titration analysis under ionic strength variation.

This package implements a rigorous two-stage protocol for extracting apparent
dissociation constants (pKa_app) from weak acid-strong base titration data,
with explicit treatment of ionic strength effects on acid-base equilibria.

Scientific Framework:
    The package addresses the question: How does increasing ionic strength
    (via NaCl addition) affect the measured pKa of a weak acid?

    Key insight: In solutions of non-zero ionic strength, the thermodynamic
    acid dissociation constant (Ka) relates to concentrations through activity
    coefficients (γ). The apparent constant pKa_app measured from concentration
    data therefore reflects both the intrinsic dissociation equilibrium and
    the activity coefficient ratio (γ_HA / (γ_H+ · γ_A-)).

Methodological Approach:
    Stage 1 — Coarse pKa_app estimate:
        Uses the half-equivalence point (V = 0.5·V_eq) where pH ≈ pKa_app
        by Henderson-Hasselbalch approximation.

    Stage 2 — Refined regression:
        Performs Henderson-Hasselbalch regression (pH vs. log₁₀(V/(V_eq−V)))
        only within the chemically valid buffer region (|pH − pKa_app| ≤ 1).

Error Handling Philosophy:
    The package fails loudly on invalid science:
    - Insufficient buffer points raise ValueError (minimum 3 required)
    - Invalid V_eq or pKa_app inputs raise explicit exceptions
    - Henderson-Hasselbalch slope deviations (|m − 1| > 0.1) trigger warnings

Uncertainty Treatment:
    All uncertainties are systematic (worst-case bounds), not statistical:
    - Equipment uncertainties propagated through calculations
    - Trial-to-trial variation reported as half-range (max − min)/2
    - Consistent with IB Chemistry Data Processing methodology

Subpackages:
    chemistry:
        Henderson-Hasselbalch regression (hh_model) and buffer region
        selection (buffer_region). No plotting dependencies.

    stats:
        Linear regression (regression) and uncertainty propagation
        (uncertainty). Pure numerical utilities with no chemistry logic.

    plotting:
        Publication-quality figure generation for titration curves
        (titration_plots) and statistical summaries (summary_plots).
        Accepts precomputed results; no chemistry calculations.

See Also:
    - salty.analysis: Main orchestration functions
    - salty.data_processing: CSV parsing and run extraction
    - salty.chemistry: Henderson-Hasselbalch model implementation
"""

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
    # Data processing
    "extract_runs",
    "load_titration_data",
    # Analysis
    "detect_equivalence_point",
    "analyze_titration",
    "process_all_files",
    "create_results_dataframe",
    "calculate_statistics",
    "build_summary_plot_data",
    # Plotting
    "setup_plot_style",
    "plot_titration_curves",
    "plot_statistical_summary",
    "save_data_to_csv",
]
