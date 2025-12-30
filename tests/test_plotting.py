import os

import numpy as np
import pandas as pd

from salty.plotting import plot_statistical_summary, plot_titration_curves


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
        "pka_reg": 3.6,
    }
    return [res]


def test_plot_titration_curves(tmp_path):
    results = make_dummy_results()
    out = plot_titration_curves(results, output_dir=str(tmp_path))
    assert len(out) == 1


def test_plot_statistical_summary(tmp_path):
    stats_df = pd.DataFrame(
        {
            "NaCl Concentration (M)": [0.0, 0.1],
            "Mean pKa": [5.0, 4.9],
            "Uncertainty": [0.05, 0.07],
            "n": [3, 3],
        }
    )
    results_df = pd.DataFrame(
        {
            "Run": ["r1", "r2"],
            "NaCl Concentration (M)": [0.0, 0.1],
            "pKa (buffer regression)": [5.0, 4.9],
            "pKa uncertainty (ΔpKa)": [0.05, 0.07],
        }
    )
    out = plot_statistical_summary(stats_df, results_df, output_dir=str(tmp_path))
    assert out.endswith("statistical_summary.png")
    assert os.path.exists(out)
