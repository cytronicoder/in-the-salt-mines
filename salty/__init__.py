"""
Titration Analysis Package

A Python package for analyzing titration data to determine half-equivalence points
and apparent pKa values of acids under different ionic strength conditions.

Modules:
    - data_processing: Functions for loading and processing titration data
    - analysis: Functions for analyzing titration runs and calculating pKa
    - plotting: Functions for creating professional figures and saving results
"""

__version__ = "1.0.0"
__author__ = "Cytronicoder"

from .data_processing import (
    extract_runs,
    calculate_derivatives,
    load_titration_data
)

from .analysis import (
    find_equivalence_point,
    analyze_titration,
    process_all_files,
    create_results_dataframe,
    calculate_statistics,
    print_statistics
)

from .plotting import (
    setup_plot_style,
    plot_titration_curves,
    plot_statistical_summary,
    save_data_to_csv
)

__all__ = [
    # Data processing
    'extract_runs',
    'calculate_derivatives',
    'load_titration_data',
    
    # Analysis
    'find_equivalence_point',
    'analyze_titration',
    'process_all_files',
    'create_results_dataframe',
    'calculate_statistics',
    'print_statistics',
    
    # Plotting
    'setup_plot_style',
    'plot_titration_curves',
    'plot_statistical_summary',
    'save_data_to_csv',
]
