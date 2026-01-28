import copy

import numpy as np
import pandas as pd
import pandas.testing as pdt

from salty.plotting import plot_titration_curves


def _make_results():
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


def test_plot_does_not_mutate_results(tmp_path):
    results = _make_results()
    snapshot = copy.deepcopy(results)

    plot_titration_curves(results, output_dir=str(tmp_path))

    pdt.assert_frame_equal(results[0]["data"], snapshot[0]["data"])
    pdt.assert_frame_equal(results[0]["step_data"], snapshot[0]["step_data"])
    pdt.assert_frame_equal(results[0]["dense_curve"], snapshot[0]["dense_curve"])
    pdt.assert_frame_equal(results[0]["buffer_region"], snapshot[0]["buffer_region"])
    assert results[0]["pka_app"] == snapshot[0]["pka_app"]
