"""Tests for IA error diagnosis analysis pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from salty.error_diagnosis import (
    compute_lack_of_fit,
    compute_level_summary,
    compute_pure_error,
    fit_polynomial_model,
    linear_vs_quadratic_f_test,
    run_error_diagnosis_pipeline,
)

LEVELS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=float)


def _make_long_df(
    *, curvature: bool = False, heteroscedastic: bool = False
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for conc in LEVELS:
        mean = 4.80 - 0.25 * conc + (0.35 * conc**2 if curvature else 0.0)
        spread = 0.01 if not heteroscedastic else (0.005 + 0.18 * conc)
        offsets = (-spread, 0.0, spread)
        for idx, offset in enumerate(offsets, start=1):
            rows.append(
                {
                    "concentration": float(conc),
                    "replicate": idx,
                    "y": float(mean + offset),
                }
            )
    return pd.DataFrame(rows)


def test_level_summary_stats_known_dataset():
    df = _make_long_df(curvature=False, heteroscedastic=False)
    summary = compute_level_summary(df)

    assert summary["concentration"].to_list() == list(LEVELS)
    baseline = summary.loc[np.isclose(summary["concentration"], 0.0)].iloc[0]
    assert int(baseline["n"]) == 3
    assert np.isclose(float(baseline["mean_y"]), 4.80)
    assert np.isclose(float(baseline["sd_y"]), 0.01)
    assert np.isclose(float(baseline["var_y"]), 0.0001)
    assert np.isclose(float(baseline["cv_pct"]), 100.0 * 0.01 / 4.80)


def test_pure_error_matches_hand_computation():
    df = _make_long_df(curvature=False, heteroscedastic=False)
    pure = compute_pure_error(df)

    expected_ss_per_group = 2 * (0.01**2)
    expected_ss_total = 6 * expected_ss_per_group
    expected_df = 6 * (3 - 1)
    expected_ms = expected_ss_total / expected_df

    assert np.isclose(float(pure["SS_pure"]), expected_ss_total)
    assert int(pure["df_pure"]) == expected_df
    assert np.isclose(float(pure["MS_pure"]), expected_ms)


def test_lack_of_fit_df_matches_regression_minus_pure():
    df = _make_long_df(curvature=False, heteroscedastic=False)
    x = df["concentration"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    linear_fit = fit_polynomial_model(x=x, y=y, degree=1, model_name="linear")
    pure = compute_pure_error(df)
    lof = compute_lack_of_fit(linear_fit=linear_fit, pure_error=pure)

    assert int(lof["df_LOF"]) == int(linear_fit.df_res - int(pure["df_pure"]))


def test_nested_f_test_matches_formula_for_curved_data():
    df = _make_long_df(curvature=True, heteroscedastic=False)
    x = df["concentration"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)

    linear_fit = fit_polynomial_model(
        x=x,
        y=y,
        degree=1,
        model_name="linear_unweighted",
    )
    quadratic_fit = fit_polynomial_model(
        x=x,
        y=y,
        degree=2,
        model_name="quadratic_unweighted",
    )
    nested = linear_vs_quadratic_f_test(
        linear_fit=linear_fit,
        quadratic_fit=quadratic_fit,
        n_obs=len(df),
    )

    expected_f = ((linear_fit.ss_res - quadratic_fit.ss_res) / 1.0) / (
        quadratic_fit.ss_res / (len(df) - 3)
    )
    assert np.isclose(float(nested["F"]), expected_f)
    assert float(nested["pvalue"]) < 0.05


def test_pipeline_creates_expected_output_files(tmp_path):
    df = _make_long_df(curvature=True, heteroscedastic=True)
    input_csv = tmp_path / "replicates.csv"
    outdir = tmp_path / "error_outputs"
    df.to_csv(input_csv, index=False)

    run_error_diagnosis_pipeline(input_path=input_csv, outdir=outdir, pka_lit=4.76)

    expected = [
        "level_summary.csv",
        "pure_error.csv",
        "tests.csv",
        "model_fits.csv",
        "point_residuals.csv",
        "regression_with_errorbars.png",
        "residuals_vs_fitted_linear.png",
        "residuals_vs_concentration_linear.png",
        "sd_or_var_vs_concentration.png",
        "group_spread.png",
    ]
    for name in expected:
        assert (outdir / name).exists(), f"Missing output: {name}"

    model_fits = pd.read_csv(outdir / "model_fits.csv")
    assert {"linear_unweighted", "quadratic_unweighted"} <= set(model_fits["model"])
