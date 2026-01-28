import os

import numpy as np
import pandas as pd
import pytest

from salty.analysis import build_summary_plot_data
from salty.plotting import plot_statistical_summary, plot_titration_curves
from salty.schema import ResultColumns


def make_dummy_results():
    step_df = pd.DataFrame(
        {
            "Volume (cm³)": [0.0, 1.0, 2.0, 3.0],
            "pH_step": [3.0, 3.5, 4.5, 7.0],
            "pH_step_sd": [0.05, 0.05, 0.05, 0.05],
            "dpH/dx": [0.1, 0.3, 1.2, 0.5],
        }
    )
    dense_df = pd.DataFrame(
        {"Volume (cm³)": np.linspace(0, 3, 50), "pH_interp": np.linspace(3, 7, 50)}
    )
    res = {
        "data": pd.DataFrame(
            {"Volume (cm³)": [0, 1, 2, 3], "pH": [3.0, 3.5, 4.5, 7.0]}
        ),
        "step_data": step_df,
        "dense_curve": dense_df,
        "run_name": "Test Run",
        "veq_used": 2.5,
        "veq_uncertainty": 0.1,
        "half_eq_x": 1.25,
        "half_eq_pH": 4.0,
        "buffer_region": pd.DataFrame(
            {
                "log10_ratio": [-0.3, 0.0, 0.3],
                "pH_step": [3.1, 3.6, 4.1],
                "pH_fit": [3.05, 3.6, 4.15],
            }
        ),
        "slope_reg": 1.0,
        "pka_app": 3.6,
        "r2_reg": 0.99,
    }
    return [res]


def test_plot_titration_curves(tmp_path):
    results = make_dummy_results()
    out = plot_titration_curves(results, output_dir=str(tmp_path))
    assert len(out) == 1


def test_plot_statistical_summary(tmp_path):
    cols = ResultColumns()
    stats_df = pd.DataFrame(
        {
            cols.nacl: [0.0, 0.1],
            "Mean Apparent pKa": [5.0, 4.9],
            "Uncertainty": [0.05, 0.07],
            "n": [3, 3],
        }
    )
    results_df = pd.DataFrame(
        {
            "Run": ["r1", "r2"],
            cols.nacl: [0.0, 0.1],
            cols.pka_app: [5.0, 4.9],
            cols.pka_unc: [0.05, 0.07],
        }
    )
    summary = build_summary_plot_data(stats_df, results_df)
    out = plot_statistical_summary(summary, output_dir=str(tmp_path))
    assert out.endswith("statistical_summary.png")
    assert os.path.exists(out)


def test_build_summary_plot_data_missing_results_df_columns():
    """Test that build_summary_plot_data raises KeyError when results_df is missing required columns."""
    cols = ResultColumns()
    stats_df = pd.DataFrame(
        {
            cols.nacl: [0.0, 0.1],
            "Mean Apparent pKa": [5.0, 4.9],
            "Uncertainty": [0.05, 0.07],
        }
    )
    
    # Missing pka_app column
    results_df_missing_pka = pd.DataFrame(
        {
            cols.nacl: [0.0, 0.1],
            "SomeOtherColumn": [1, 2],
        }
    )
    
    with pytest.raises(KeyError, match="results_df is missing required columns"):
        build_summary_plot_data(stats_df, results_df_missing_pka)
    
    # Missing nacl column
    results_df_missing_nacl = pd.DataFrame(
        {
            cols.pka_app: [5.0, 4.9],
            "SomeOtherColumn": [1, 2],
        }
    )
    
    with pytest.raises(KeyError, match="results_df is missing required columns"):
        build_summary_plot_data(stats_df, results_df_missing_nacl)

