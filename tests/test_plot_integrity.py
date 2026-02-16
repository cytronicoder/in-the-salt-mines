"""Verify plotting functions do not mutate analysis results."""

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas.testing as pdt

from salty.plotting import plot_titration_curves
from salty.plotting.style import add_panel_label, save_figure_bundle


def _make_results():
    """Create a minimal, validated results payload for plotting tests."""
    step_df = pd.DataFrame(
        {
            "Volume (cm^3)": [0.0, 1.0, 2.0, 3.0],
            "pH_step": [3.0, 3.5, 4.5, 7.0],
            "pH_step_sd": [0.05, 0.05, 0.05, 0.05],
            "dpH/dx": [0.1, 0.3, 1.2, 0.5],
        }
    )
    dense_df = pd.DataFrame(
        {"Volume (cm^3)": np.linspace(0, 3, 50), "pH_interp": np.linspace(3, 7, 50)}
    )
    res = {
        "data": pd.DataFrame(
            {"Volume (cm^3)": [0, 1, 2, 3], "pH": [3.0, 3.5, 4.5, 7.0]}
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
    """Ensure plot generation does not mutate input result dictionaries.

    Args:
        tmp_path: Pytest fixture providing a temporary directory.

    Returns:
        None.
    """
    results = _make_results()
    snapshot = copy.deepcopy(results)

    plot_titration_curves(results, output_dir=str(tmp_path))

    pdt.assert_frame_equal(results[0]["data"], snapshot[0]["data"])
    pdt.assert_frame_equal(results[0]["step_data"], snapshot[0]["step_data"])
    pdt.assert_frame_equal(results[0]["dense_curve"], snapshot[0]["dense_curve"])
    pdt.assert_frame_equal(results[0]["buffer_region"], snapshot[0]["buffer_region"])
    assert results[0]["pka_app"] == snapshot[0]["pka_app"]


def test_save_figure_bundle_uses_tight_bounding(monkeypatch, tmp_path):
    """Ensure multi-format exports use tight bounding and expected padding."""
    fig, _ = plt.subplots()
    calls = []

    def _fake_savefig(path, **kwargs):
        calls.append((Path(path).suffix, kwargs))

    monkeypatch.setattr(fig, "savefig", _fake_savefig)

    out_png = tmp_path / "integrity_plot.png"
    saved = save_figure_bundle(fig, str(out_png))

    assert saved.endswith("integrity_plot.png")
    assert [ext for ext, _ in calls] == [".png", ".pdf", ".svg"]
    for ext, kwargs in calls:
        assert kwargs.get("bbox_inches") == "tight"
        assert kwargs.get("pad_inches") == 0.12
        if ext == ".png":
            assert kwargs.get("dpi") == 300
        else:
            assert kwargs.get("dpi") is None

    plt.close(fig)


def test_add_panel_label_defaults_and_bbox_behavior():
    """Verify panel-label defaults and boxless rendering policy."""
    fig, ax = plt.subplots()
    add_panel_label(ax, "(a)")
    txt_default = ax.texts[-1]
    assert txt_default.get_position() == (0.02, 0.98)
    assert txt_default.get_bbox_patch() is None

    add_panel_label(ax, "(b)", bbox=True)
    txt_with_bbox = ax.texts[-1]
    assert txt_with_bbox.get_position() == (0.02, 0.98)
    assert txt_with_bbox.get_bbox_patch() is None

    plt.close(fig)
