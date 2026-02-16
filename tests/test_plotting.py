"""Test plotting utilities using validated analysis inputs."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from salty.analysis import build_summary_plot_data, create_results_dataframe
from salty.plotting import (
    generate_ia_figure_set,
    plot_derivative_equivalence_by_nacl,
    plot_pka_app_vs_nacl_and_I,
    plot_statistical_summary,
    plot_titration_curves,
    plot_titration_overlays_by_nacl,
)
from salty.plotting.style import MATH_LABELS
from salty.schema import ResultColumns


def make_dummy_results():
    """Construct a minimal results payload for plotting tests."""
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


def make_ia_results():
    """Create synthetic multi-condition run payloads for IA figure tests."""
    results = []
    rng = np.random.default_rng(7)
    for nacl in (0.0, 0.2, 0.4, 0.6, 0.8):
        for run_id in (1, 2):
            vol = np.array([0, 2, 4, 6, 8, 10, 11, 12, 13, 14, 16, 18, 20], dtype=float)
            baseline = 2.85 + 0.04 * nacl + 0.02 * run_id
            ph = baseline + 0.09 * vol + 2.5 / (1.0 + np.exp(-(vol - 12.2) / 0.9))
            ph += rng.normal(0.0, 0.03, size=len(vol))
            veq = 12.5 + 0.1 * nacl + 0.04 * (run_id - 1)
            half = veq / 2.0
            half_ph = float(np.interp(half, vol, ph))

            ratio = vol / (veq - vol)
            mask = np.isfinite(ratio) & (ratio > 0.1) & (ratio < 10)
            x_hh = np.log10(ratio[mask])
            y_hh = ph[mask]
            if len(x_hh) >= 2:
                m, b = np.polyfit(x_hh, y_hh, 1)
                y_fit = m * x_hh + b
                ss_res = float(np.sum((y_hh - y_fit) ** 2))
                ss_tot = float(np.sum((y_hh - np.mean(y_hh)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.99
            else:
                m, b, r2, y_fit = 1.0, half_ph, 0.99, y_hh

            temp_mean = 27.4 if (nacl == 0.8 and run_id == 2) else 26.0 + 0.1 * run_id
            temp_trace = temp_mean + 0.08 * np.sin(np.linspace(0, 2 * np.pi, len(vol)))

            step_df = pd.DataFrame({"Volume (cm^3)": vol, "pH_step": ph})
            dense_vol = np.linspace(float(np.min(vol)), float(np.max(vol)), 200)
            dense_df = pd.DataFrame(
                {
                    "Volume (cm^3)": dense_vol,
                    "pH_interp": np.interp(dense_vol, vol, ph),
                }
            )
            raw_df = pd.DataFrame(
                {
                    "Volume (cm^3)": vol,
                    "pH": ph,
                    "Time (min)": np.linspace(0, 12, len(vol)),
                    "Temperature (°C)": temp_trace,
                }
            )
            buffer_df = pd.DataFrame(
                {
                    "Volume (cm^3)": vol[mask],
                    "log10_ratio": x_hh,
                    "pH_step": y_hh,
                    "pH_fit": y_fit,
                }
            )

            results.append(
                {
                    "run_name": f"{nacl:.1f}M - Run {run_id}",
                    "nacl_conc": nacl,
                    "source_file": f"{nacl:.1f}M synthetic.csv",
                    "pka_app": float(b),
                    "pka_method": "buffer_regression",
                    "pka_app_uncertainty": 0.04,
                    "eq_qc_pass": True,
                    "veq_used": veq,
                    "veq_uncertainty": 0.10,
                    "veq_method": "discrete_midpoint",
                    "half_eq_x": half,
                    "half_eq_pH": half_ph,
                    "slope_reg": float(m),
                    "r2_reg": float(r2),
                    "data": raw_df,
                    "step_data": step_df,
                    "dense_curve": dense_df,
                    "buffer_region": buffer_df,
                }
            )
    return results


def test_plot_titration_curves(tmp_path):
    """Confirm that titration plots are written to disk.

    Args:
        tmp_path: Pytest fixture providing a temporary directory.

    Returns:
        None.
    """
    results = make_dummy_results()
    out = plot_titration_curves(results, output_dir=str(tmp_path))
    assert len(out) == 1
    assert os.path.exists(out[0].replace(".png", ".pdf"))
    assert os.path.exists(out[0].replace(".png", ".svg"))
    png = out[0]
    assert os.path.exists(png)
    assert os.path.exists(png.replace(".png", ".pdf"))
    assert os.path.exists(png.replace(".png", ".svg"))


def test_plot_statistical_summary(tmp_path):
    """Confirm that the summary plot is written to disk.

    Args:
        tmp_path: Pytest fixture providing a temporary directory.

    Returns:
        None.
    """
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
    assert os.path.exists(out.replace(".png", ".pdf"))
    assert os.path.exists(out.replace(".png", ".svg"))
    assert os.path.exists(out.replace(".png", ".pdf"))
    assert os.path.exists(out.replace(".png", ".svg"))


def test_build_summary_plot_data_missing_results_df_columns():
    """Raise errors when required results columns are missing.

    Returns:
        None.
    """
    cols = ResultColumns()
    stats_df = pd.DataFrame(
        {
            cols.nacl: [0.0, 0.1],
            "Mean Apparent pKa": [5.0, 4.9],
            "Uncertainty": [0.05, 0.07],
        }
    )

    results_df_missing_pka = pd.DataFrame(
        {
            cols.nacl: [0.0, 0.1],
            "SomeOtherColumn": [1, 2],
        }
    )

    with pytest.raises(KeyError, match="results_df is missing required columns"):
        build_summary_plot_data(stats_df, results_df_missing_pka)

    results_df_missing_nacl = pd.DataFrame(
        {
            cols.pka_app: [5.0, 4.9],
            "SomeOtherColumn": [1, 2],
        }
    )
    with pytest.raises(KeyError, match="results_df is missing required columns"):
        build_summary_plot_data(stats_df, results_df_missing_nacl)


def test_generate_ia_figure_set_outputs_and_captions(tmp_path):
    """Generate Figure 1-5 files and caption boilerplates for IA output."""
    results = make_ia_results()
    results_df = create_results_dataframe(results)

    ia_dir = tmp_path / "output" / "ia"
    iter_dir = tmp_path / "output" / "iterations" / "all_valid"
    ia_dir.mkdir(parents=True, exist_ok=True)

    summary_df = (
        results_df.groupby("NaCl Concentration (M)", as_index=False)["Apparent pKa"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "NaCl Concentration (M)": "NaCl Concentration (M)",
                "mean": "Mean pKa_app",
                "std": "SD pKa_app",
                "count": "n",
            }
        )
    )
    summary_df["SEM pKa_app"] = summary_df["SD pKa_app"] / np.sqrt(summary_df["n"])
    summary_df["Combined uncertainty"] = summary_df["SEM pKa_app"].fillna(0.04)
    summary_path = ia_dir / "processed_summary_with_sd.csv"
    summary_df.to_csv(summary_path, index=False)

    generate_ia_figure_set(
        results=results,
        results_df=results_df,
        output_dir=str(ia_dir),
        iteration_output_dir=str(iter_dir),
        summary_csv_path=str(summary_path),
    )

    stems = [
        "titration_overlays_by_nacl",
        "derivative_equivalence_by_nacl",
        "pka_app_vs_nacl_and_I",
        "hh_linearization_and_diagnostics",
        "temperature_and_calibration_qc",
    ]
    for stem in stems:
        for ext in ("png", "pdf", "svg"):
            assert (ia_dir / f"{stem}.{ext}").exists()
            assert (iter_dir / f"{stem}.{ext}").exists()
        assert (ia_dir / f"{stem}_caption.txt").exists()

    caption = (ia_dir / "titration_overlays_by_nacl_caption.txt").read_text(
        encoding="utf-8"
    )
    assert "V_eq" in caption
    assert "V_{1/2}" in caption
    assert "activity" in caption


def test_new_plot_axis_labels_match_spec(tmp_path):
    """Validate axis labels for Figures 1-3 against the publication spec."""
    results = make_ia_results()
    results_df = create_results_dataframe(results)
    out_dir = tmp_path / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    p1, fig1 = plot_titration_overlays_by_nacl(
        results, output_dir=str(out_dir), return_figure=True
    )
    assert os.path.exists(p1)
    labels1_x = {ax.get_xlabel() for ax in fig1.axes}
    labels1_y = {ax.get_ylabel() for ax in fig1.axes}
    assert MATH_LABELS["x_volume"] in labels1_x
    assert MATH_LABELS["ph"] in labels1_y
    plt.close(fig1)

    p2, fig2 = plot_derivative_equivalence_by_nacl(
        results, output_dir=str(out_dir), return_figure=True
    )
    assert os.path.exists(p2)
    labels2_x = {ax.get_xlabel() for ax in fig2.axes}
    labels2_y = {ax.get_ylabel() for ax in fig2.axes}
    assert MATH_LABELS["x_volume"] in labels2_x
    assert MATH_LABELS["y_derivative"] in labels2_y
    plt.close(fig2)

    summary_path = out_dir / "processed_summary_with_sd.csv"
    summary_df = (
        results_df.groupby("NaCl Concentration (M)", as_index=False)["Apparent pKa"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "NaCl Concentration (M)": "NaCl Concentration (M)",
                "mean": "Mean pKa_app",
                "std": "SD pKa_app",
                "count": "n",
            }
        )
    )
    summary_df["SEM pKa_app"] = summary_df["SD pKa_app"] / np.sqrt(summary_df["n"])
    summary_df.to_csv(summary_path, index=False)

    p3, fig3 = plot_pka_app_vs_nacl_and_I(
        results_df=results_df,
        results=results,
        summary_csv_path=str(summary_path),
        output_dir=str(out_dir),
        return_figure=True,
    )
    assert os.path.exists(p3)
    assert fig3.axes[0].get_xlabel() == MATH_LABELS["x_nacl"]
    assert fig3.axes[1].get_xlabel() == MATH_LABELS["x_ionic"]
    assert MATH_LABELS["pka_app"] in fig3.axes[0].get_ylabel()
    plt.close(fig3)
