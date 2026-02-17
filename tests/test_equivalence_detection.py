"""Unit tests for robust equivalence-point peak selection."""

import numpy as np
import pandas as pd
import pytest

from salty.analysis import (
    _ensure_derivative,
    build_ph_interpolator,
    detect_equivalence_point,
)


def _make_step_df(volumes: np.ndarray, ph_values: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Volume (cm^3)": np.asarray(volumes, dtype=float),
            "pH_step": np.asarray(ph_values, dtype=float),
        }
    )
    df["pH_smooth"] = df["pH_step"]
    return _ensure_derivative(df, use_smooth=True)


@pytest.fixture()
def synthetic_edge_spike_curve() -> tuple[pd.DataFrame, float]:
    """Curve with a valid central jump and an artificial end-tail spike."""
    vol = np.linspace(0.0, 40.0, 81)
    ph = 3.0 + 0.06 * vol
    ph += 5.4 / (1.0 + np.exp(-(vol - 23.0) / 1.3))
    ph += 0.8 / (1.0 + np.exp(-(vol - 36.0) / 0.9))
    ph[-4] += -0.05
    ph[-3] += 0.50
    ph[-2] += -0.05
    ph[-1] += 0.45
    return _make_step_df(vol, ph), 23.0


@pytest.fixture()
def synthetic_clean_curve() -> tuple[pd.DataFrame, float]:
    """Single equivalence jump with no pathological edge behavior."""
    vol = np.linspace(0.0, 40.0, 81)
    ph = 3.1 + 0.055 * vol
    ph += 5.2 / (1.0 + np.exp(-(vol - 24.0) / 1.5))
    return _make_step_df(vol, ph), 24.0


@pytest.fixture()
def synthetic_two_peak_curve() -> tuple[pd.DataFrame, float]:
    """Two-peak curve where derivative max is not the true equivalence jump."""
    vol = np.linspace(0.0, 40.0, 81)
    ph = 3.0 + 0.05 * vol
    ph += 5.0 / (1.0 + np.exp(-(vol - 19.5) / 1.9))
    ph += 1.0 / (1.0 + np.exp(-(vol - 30.0) / 0.18))
    ph += 0.03 * np.sin(0.9 * vol) + 0.02 * np.cos(1.7 * vol)
    return _make_step_df(vol, ph), 20.0


def test_detect_equivalence_ignores_pchip_edge_spike(synthetic_edge_spike_curve):
    step_df, expected_veq = synthetic_edge_spike_curve
    interp = build_ph_interpolator(step_df, method="pchip")

    vol = step_df["Volume (cm^3)"].to_numpy(dtype=float)
    d_pchip = np.asarray(interp["deriv_func"](vol), dtype=float)
    pchip_peak_idx = int(np.nanargmax(d_pchip))
    assert pchip_peak_idx >= len(vol) - 3

    eq = detect_equivalence_point(step_df, interpolator=interp)

    assert eq["qc_pass"] is True
    assert eq["qc_diagnostics"]["derivative_source"] == "step_gradient"
    assert abs(float(eq["eq_x"]) - expected_veq) <= 1.0
    assert int(eq["qc_diagnostics"]["peak_index"]) != pchip_peak_idx
    assert int(eq["qc_diagnostics"]["peak_index"]) > 2
    assert int(eq["qc_diagnostics"]["peak_index"]) < len(step_df) - 3


def test_detect_equivalence_clean_curve_no_regression(synthetic_clean_curve):
    step_df, expected_veq = synthetic_clean_curve
    interp = build_ph_interpolator(step_df, method="pchip")

    vol = step_df["Volume (cm^3)"].to_numpy(dtype=float)
    d_step = step_df["dpH/dx"].to_numpy(dtype=float)
    d_pchip = np.asarray(interp["deriv_func"](vol), dtype=float)
    assert (
        abs(
            float(vol[int(np.nanargmax(d_step))])
            - float(vol[int(np.nanargmax(d_pchip))])
        )
        <= 1.0
    )

    eq = detect_equivalence_point(step_df, interpolator=interp)

    assert eq["qc_pass"] is True
    assert abs(float(eq["eq_x"]) - expected_veq) <= 1.0
    assert int(eq["qc_diagnostics"]["peak_index"]) > 2
    assert int(eq["qc_diagnostics"]["peak_index"]) < len(step_df) - 3


def test_detect_equivalence_prefers_peak_with_larger_ph_jump(synthetic_two_peak_curve):
    step_df, expected_veq = synthetic_two_peak_curve

    naive_idx = int(np.nanargmax(step_df["dpH/dx"].to_numpy(dtype=float)))
    naive_veq = float(step_df.loc[naive_idx, "Volume (cm^3)"])
    assert naive_veq > 27.0

    eq = detect_equivalence_point(step_df, interpolator=None, min_post_points=4)

    assert eq["qc_pass"] is True
    assert abs(float(eq["eq_x"]) - expected_veq) <= 1.2
    assert float(eq["eq_x"]) < naive_veq

    candidates = eq["qc_diagnostics"]["candidate_peaks"]
    qc_candidates = [c for c in candidates if c["qc_pass"]]
    selected_idx = int(eq["qc_diagnostics"]["peak_index"])
    selected = next(c for c in candidates if int(c["peak_index"]) == selected_idx)

    assert len(qc_candidates) >= 2
    assert (
        float(selected["local_jump_pH"])
        >= max(float(c["local_jump_pH"]) for c in qc_candidates) - 1e-6
    )
    assert selected_idx > 2
    assert selected_idx < len(step_df) - 3
