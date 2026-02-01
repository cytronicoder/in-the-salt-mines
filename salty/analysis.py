"""Orchestrate two-stage analysis for weak acid titration data.

This module implements the complete analysis pipeline for the IA investigation:
'How NaCl Concentration Affects the Half-Equivalence pH in an Ethanoic Acid Titration'

Experimental Design:
    - Ethanoic acid (CH₃COOH): 0.10 mol dm⁻^3, 25.00 cm^3 sample
    - Titrant (NaOH): 0.10 mol dm⁻^3
    - NaCl concentrations: 0.00, 0.20, 0.40, 0.60, 0.80, 1.00 M
    - Temperature: 26 ± 1°C
    - pH measurement: Vernier pH Sensor (±0.3 pH units)
    - Three replicate titrations per [NaCl] condition

Analysis Strategy (Two-Stage):
    Stage 1: Coarse pKa_app estimation
        - Locate equivalence point (V_eq) from maximum dpH/dV
        - Calculate half-equivalence volume: V_half = V_eq / 2
        - Interpolate pH at V_half to obtain pKa_app (first estimate)

    Stage 2: Refined Henderson-Hasselbalch regression
        - Select buffer region: |pH - pKa_app| ≤ 1
        - Fit pH = m·log₁₀(V/(V_eq - V)) + b within buffer region
        - Extract refined pKa_app as intercept (b)
        - Slope (m) should be ≈1.0; deviations indicate non-ideality

Theoretical Context:
    At higher [NaCl], ionic strength (μ) increases, causing activity coefficients
    to deviate from unity. Since pH probes measure H⁺ activity rather than
    concentration, the measured pH reflects an apparent pKa (pKa_app) that varies
    with ionic strength. This investigation examines how pKa_app changes as a
    function of [NaCl].
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .chemistry.hh_model import fit_henderson_hasselbalch
from .data_processing import aggregate_volume_steps, extract_runs, load_titration_data
from .schema import ResultColumns
from .stats.regression import linear_regression, slope_uncertainty_from_endpoints
from .stats.uncertainty import (
    burette_delivered_uncertainty,
    combine_uncertainties,
    concentration_uncertainty,
    round_value_to_uncertainty,
)

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
HAVE_SAVGOL = HAVE_SCIPY and (importlib.util.find_spec("scipy.signal") is not None)
if HAVE_SCIPY:
    from scipy.interpolate import PchipInterpolator
if HAVE_SAVGOL:
    from scipy.signal import savgol_filter
else:
    savgol_filter = None

# Default uncertainties from IA equipment specifications
DEFAULT_BURETTE_UNC = 0.05  # 50.0 cm^3 burette: ±0.05 cm^3 (total volume)
DEFAULT_BURETTE_READING_UNC = 0.02  # Individual burette graduations: ±0.02 cm^3
DEFAULT_PH_METER_SYS = 0.3  # Vernier pH Sensor: ±0.3 pH units (measures activity)


def _prepare_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare x-y arrays for interpolation by sorting and deduplicating.

    Args:
        x: Independent-variable values (e.g., delivered volume).
        y: Dependent-variable values (e.g., pH).

    Returns:
        A tuple of ``(x, y)`` arrays with NaN values removed, sorted in
        ascending x order, and deduplicated by averaging repeated x values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if len(np.unique(x)) < len(x):
        df = pd.DataFrame({"x": x, "y": y}).groupby("x", as_index=False).mean()
        x = df["x"].to_numpy(dtype=float)
        y = df["y"].to_numpy(dtype=float)
    return x, y


def build_ph_interpolator(step_df: pd.DataFrame, method: str | None = None) -> Dict:
    """Construct a pH interpolation model from step-aggregated data.

    The interpolation is used to evaluate pH at half-equivalence and to
    generate smooth titration curves for visualization. When available, the
    piecewise cubic Hermite interpolating polynomial (PCHIP) is preferred
    because it preserves monotonicity and avoids overshoot in sigmoidal data.

    Args:
        step_df: DataFrame containing ``Volume (cm^3)`` and ``pH_step`` columns.
        method: Interpolation method (``"pchip"`` or ``"linear"``). When
            ``None``, PCHIP is selected if SciPy is available.

    Returns:
        A dictionary containing:
            - ``method``: The interpolation method used.
            - ``func``: Callable mapping volume to pH.
            - ``deriv_func``: Callable for the first derivative (PCHIP only).
            - ``x_min`` and ``x_max``: Bounds of the interpolation domain.
    """
    if (
        step_df.empty
        or "Volume (cm^3)" not in step_df.columns
        or "pH_step" not in step_df.columns
    ):
        return {
            "method": "linear",
            "func": lambda x: np.full_like(np.asarray(x, dtype=float), np.nan),
        }

    x = step_df["Volume (cm^3)"].to_numpy(dtype=float)
    y = step_df["pH_step"].to_numpy(dtype=float)
    x, y = _prepare_xy(x, y)

    if len(x) < 2:
        return {
            "method": "linear",
            "func": lambda xq: np.full_like(np.asarray(xq, dtype=float), np.nan),
        }

    x_min = float(np.min(x))
    x_max = float(np.max(x))

    if method is None:
        method = "pchip" if HAVE_SCIPY else "linear"

    if method == "pchip" and HAVE_SCIPY and len(x) >= 3:
        interp = PchipInterpolator(x, y, extrapolate=False)
        d_interp = interp.derivative()

        def f(xq):
            xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
            yq = np.asarray(interp(xq_arr), dtype=float)
            out = (xq_arr < x_min) | (xq_arr > x_max)
            yq[out] = np.nan
            return float(yq[0]) if np.ndim(xq) == 0 else yq

        def df(xq):
            xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
            yq = np.asarray(d_interp(xq_arr), dtype=float)
            out = (xq_arr < x_min) | (xq_arr > x_max)
            yq[out] = np.nan
            return float(yq[0]) if np.ndim(xq) == 0 else yq

        return {
            "method": "pchip",
            "func": f,
            "deriv_func": df,
            "x_min": x_min,
            "x_max": x_max,
        }

    def f(xq):
        xq_arr = np.atleast_1d(np.asarray(xq, dtype=float))
        yq = np.interp(xq_arr, x, y)
        out = (xq_arr < x_min) | (xq_arr > x_max)
        yq = np.asarray(yq, dtype=float)
        yq[out] = np.nan
        return float(yq[0]) if np.ndim(xq) == 0 else yq

    return {"method": "linear", "func": f, "x_min": x_min, "x_max": x_max}


def generate_dense_curve(interpolator: Dict, n_points: int = 1200) -> pd.DataFrame:
    """Evaluate an interpolation model on a dense grid for plotting.

    Args:
        interpolator: Dictionary returned by ``build_ph_interpolator``.
        n_points: Number of points used to sample the interpolation domain.

    Returns:
        A DataFrame with ``Volume (cm^3)`` and ``pH_interp`` columns. Returns an
        empty DataFrame if interpolation bounds are unavailable.
    """
    if "x_min" not in interpolator or "x_max" not in interpolator:
        return pd.DataFrame(columns=["Volume (cm^3)", "pH_interp"])
    x_dense = np.linspace(interpolator["x_min"], interpolator["x_max"], n_points)
    y_dense = interpolator["func"](x_dense)
    return pd.DataFrame({"Volume (cm^3)": x_dense, "pH_interp": y_dense})


def _smooth_ph_for_derivative(
    step_df: pd.DataFrame, window: int = 7, poly: int = 2
) -> pd.DataFrame:
    if step_df.empty:
        return step_df

    df = step_df.copy()
    v = pd.to_numeric(df["Volume (cm^3)"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(df["pH_step"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v) & np.isfinite(p)

    p_smooth = np.full_like(p, np.nan, dtype=float)
    pp = p[mask]

    if len(pp) < 5:
        p_smooth[mask] = pp
    else:
        w = int(window)
        if w % 2 == 0:
            w += 1
        w = min(w, len(pp) if len(pp) % 2 == 1 else len(pp) - 1)
        w = max(w, 5)

        if HAVE_SCIPY and HAVE_SAVGOL and w >= 5 and w <= len(pp):
            try:
                p_sg = savgol_filter(
                    pp, window_length=w, polyorder=min(poly, w - 2), mode="interp"
                )
                p_smooth[mask] = p_sg
            except Exception:
                p_smooth[mask] = pp
        else:
            ser = pd.Series(pp)
            p_rm = (
                ser.rolling(window=w, center=True, min_periods=1)
                .mean()
                .to_numpy(dtype=float)
            )
            p_smooth[mask] = p_rm

    df["pH_smooth"] = p_smooth
    return df


def _ensure_derivative(step_df: pd.DataFrame, use_smooth: bool = True) -> pd.DataFrame:
    if step_df.empty:
        return step_df

    df = step_df.copy()
    if "dpH/dx" in df.columns and df["dpH/dx"].notna().any():
        return df

    v = pd.to_numeric(df["Volume (cm^3)"], errors="coerce").to_numpy(dtype=float)
    col = "pH_smooth" if (use_smooth and "pH_smooth" in df.columns) else "pH_step"
    p = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(v) & np.isfinite(p)
    dp = np.full_like(p, np.nan, dtype=float)

    if np.sum(mask) >= 3:
        vv = v[mask]
        pp = p[mask]
        dp_local = np.gradient(pp, vv)
        dp[mask] = dp_local

    df["dpH/dx"] = dp
    return df


def _qc_equivalence(
    step_df: pd.DataFrame,
    peak_index: int,
    edge_buffer: int = 2,
    min_post_points: int = 3,
) -> Tuple[bool, str, Dict[str, float | int | bool]]:
    n = int(len(step_df))
    reasons = []
    diagnostics: Dict[str, float | int | bool] = {
        "edge_buffer": int(edge_buffer),
        "min_post_points": int(min_post_points),
        "post_points": int(n - peak_index - 1),
    }

    if peak_index <= edge_buffer or peak_index >= n - 1 - edge_buffer:
        reasons.append("Peak too close to data edge")
        diagnostics["edge_proximity_flag"] = True
    else:
        diagnostics["edge_proximity_flag"] = False
    if (n - peak_index - 1) < min_post_points:
        reasons.append("Insufficient post-equivalence coverage")
        diagnostics["post_coverage_flag"] = True
    else:
        diagnostics["post_coverage_flag"] = False

    pH_vals = pd.to_numeric(step_df["pH_step"], errors="coerce").to_numpy(dtype=float)
    pH_f = pH_vals[np.isfinite(pH_vals)]
    if len(pH_f) >= 3:
        total_span = float(np.max(pH_f) - np.min(pH_f))
        min_jump = max(0.5, 0.10 * total_span)
        before_idx = max(int(peak_index) - 2, 0)
        after_idx = min(int(peak_index) + 2, n - 1)
        delta_region = float(
            step_df.loc[after_idx, "pH_step"] - step_df.loc[before_idx, "pH_step"]
        )
        diagnostics["steepness_delta_pH"] = float(delta_region)
        diagnostics["steepness_threshold_pH"] = float(min_jump)
        if not np.isfinite(delta_region) or delta_region < min_jump:
            reasons.append("Steep-region pH change too small for a clear equivalence")
            diagnostics["steepness_flag"] = True
        else:
            diagnostics["steepness_flag"] = False
    else:
        diagnostics["steepness_delta_pH"] = np.nan
        diagnostics["steepness_threshold_pH"] = np.nan
        diagnostics["steepness_flag"] = True

    ok = len(reasons) == 0
    return ok, ("OK" if ok else "; ".join(reasons)), diagnostics


def detect_equivalence_point(
    step_df: pd.DataFrame,
    interpolator: Dict | None = None,
    edge_buffer: int = 2,
    min_post_points: int = 3,
    gate_on_qc: bool = False,
) -> Dict:
    """Detect the equivalence point using derivative maxima and QC checks.

    The equivalence point occurs when the moles of NaOH added equal the initial
    moles of ethanoic acid. For 0.10 M acid (25.00 cm^3) titrated with 0.10 M NaOH,
    the expected equivalence volume is ~25 cm^3.

    Detection strategy:
        1. Compute dpH/dV from step-aggregated data (numerical gradient)
        2. Locate the maximum of dpH/dV (steepest pH rise)
        3. Apply QC checks: edge proximity, post-equivalence coverage, pH jump
        4. If QC passes, record V_eq and pH_eq

    The equivalence point is used to calculate the half-equivalence volume:
        V_half = V_eq / 2

    At V_half, [CH₃COOH] ≈ [CH₃COO⁻], so pH ≈ pKa_app (Stage 1 estimate).

    Args:
        step_df: Step-aggregated DataFrame containing ``Volume (cm^3)`` and
            ``pH_step`` columns from Logger Pro exports.
        interpolator: Optional interpolation dictionary from
            ``build_ph_interpolator``. When provided, a derivative of the
            interpolator is used for peak detection.
        edge_buffer: Minimum number of points away from each data edge for
            a valid equivalence point. Default: 2.
        min_post_points: Minimum number of points required after the peak
            to ensure adequate post-equivalence coverage. Default: 3.
        gate_on_qc: Whether to discard the peak if QC checks fail. If True,
            returns NaN for failed QC; if False, returns the peak regardless.

    Returns:
        A dictionary with equivalence volume (cm^3), pH at equivalence, QC status,
        and diagnostic metadata. If no valid equivalence point is found, the
        returned values are NaN and QC is marked as failed.
    """
    if step_df.empty:
        return {
            "eq_x": np.nan,
            "eq_pH": np.nan,
            "qc_pass": False,
            "qc_reason": "No step data",
            "method": None,
            "qc_diagnostics": {},
        }

    df = _ensure_derivative(step_df, use_smooth=True)

    volumes = pd.to_numeric(df["Volume (cm^3)"], errors="coerce").to_numpy(dtype=float)
    deriv_source = "step_gradient"
    d = pd.to_numeric(df["dpH/dx"], errors="coerce")
    if interpolator and "deriv_func" in interpolator:
        d_eval = interpolator["deriv_func"](volumes)
        d = pd.Series(d_eval, index=df.index)
        deriv_source = "pchip_step"

    if d.dropna().empty:
        step_ok, step_reason = False, "Derivative all NaN"
        eq_x_step = np.nan
        peak_idx = None
    else:
        peak_idx = int(d.idxmax())

        d_vals = d.dropna()
        if len(d_vals) > 0:
            max_deriv = float(d_vals.max())
            threshold = 0.9 * max_deriv
            n_comparable = int(np.sum(d_vals >= threshold))
            if n_comparable > 1:
                msg = (
                    "Multiple possible equivalence points detected "
                    f"({n_comparable} peaks); result may be unstable."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)

        step_ok, step_reason, qc_metrics = _qc_equivalence(
            df, peak_idx, edge_buffer=edge_buffer, min_post_points=min_post_points
        )
        eq_x_step = float(df.loc[peak_idx, "Volume (cm^3)"])

        v_min, v_max = float(np.nanmin(volumes)), float(np.nanmax(volumes))
        v_range = v_max - v_min
        if v_range > 0:
            if eq_x_step < (v_min + 0.05 * v_range) or eq_x_step > (
                v_max - 0.05 * v_range
            ):
                msg = (
                    f"Equivalence point V_eq={eq_x_step:.2f} is near data edge; "
                    "extend volume range for more reliable detection."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
        if gate_on_qc and not step_ok:
            eq_x_step = np.nan

    qc_diagnostics = {
        "edge_buffer": edge_buffer,
        "min_post_points": min_post_points,
        "derivative_source": deriv_source,
    }
    if peak_idx is not None:
        qc_diagnostics["peak_index"] = int(peak_idx)
    if "qc_metrics" in locals():
        qc_diagnostics.update(qc_metrics)

    if np.isfinite(eq_x_step):
        eq_pH = (
            float(interpolator["func"](eq_x_step))
            if (interpolator and "func" in interpolator)
            else float(df.loc[peak_idx, "pH_step"])
        )
        return {
            "eq_x": eq_x_step,
            "eq_pH": eq_pH,
            "qc_pass": bool(step_ok),
            "qc_reason": step_reason,
            "method": "derivative_step",
            "qc_diagnostics": qc_diagnostics,
        }

    if interpolator and ("x_min" in interpolator) and ("x_max" in interpolator):
        x_dense = np.linspace(interpolator["x_min"], interpolator["x_max"], 2500)
        if "deriv_func" in interpolator:
            d_dense = interpolator["deriv_func"](x_dense)
        else:
            y_dense = interpolator["func"](x_dense)
            d_dense = np.gradient(y_dense, x_dense)

        d_dense = np.asarray(d_dense, dtype=float)
        mask = np.isfinite(d_dense)
        if np.any(mask):
            idx = int(np.nanargmax(d_dense))
            eq_x = float(x_dense[idx])
            eq_pH = float(interpolator["func"](eq_x))
            bound_buffer = 0.05 * (interpolator["x_max"] - interpolator["x_min"])
            reasons = []
            if (eq_x - interpolator["x_min"]) < bound_buffer or (
                interpolator["x_max"] - eq_x
            ) < bound_buffer:
                reasons.append("Dense peak too close to bounds")
            if not step_df.empty:
                nearest_idx = int(
                    np.nanargmin(
                        np.abs(step_df["Volume (cm^3)"].to_numpy(dtype=float) - eq_x)
                    )
                )
                ok2, reason2, qc2 = _qc_equivalence(
                    step_df,
                    nearest_idx,
                    edge_buffer=edge_buffer,
                    min_post_points=min_post_points,
                )
                if not ok2 and reason2 != "OK":
                    reasons.append(reason2)
            qc_pass = len(reasons) == 0
            return {
                "eq_x": eq_x,
                "eq_pH": eq_pH,
                "qc_pass": qc_pass,
                "qc_reason": ("OK" if qc_pass else "; ".join(reasons)),
                "method": "derivative_dense",
                "qc_diagnostics": qc_diagnostics,
            }

    return {
        "eq_x": np.nan,
        "eq_pH": np.nan,
        "qc_pass": False,
        "qc_reason": "Equivalence not found",
        "method": None,
        "qc_diagnostics": qc_diagnostics,
    }


def _veq_uncertainty(
    step_df: pd.DataFrame,
    burette_unc: float = DEFAULT_BURETTE_UNC,
    method: str = "worst_case",
) -> float:
    if step_df.empty or "Volume (cm^3)" not in step_df.columns:
        return float(burette_delivered_uncertainty(burette_unc))

    volumes = pd.to_numeric(step_df["Volume (cm^3)"], errors="coerce").to_numpy(
        dtype=float
    )
    volumes = volumes[np.isfinite(volumes)]
    if len(volumes) < 2:
        return float(burette_delivered_uncertainty(burette_unc))

    dv = np.diff(volumes)
    dv = np.abs(dv[np.isfinite(dv)])
    if len(dv) == 0:
        return float(burette_delivered_uncertainty(burette_unc))

    median_step = float(np.median(dv))
    res_term = 0.5 * median_step
    burette_term = burette_delivered_uncertainty(burette_unc)
    return float(combine_uncertainties([res_term, burette_term], method=method))


def _pka_app_unc_from_buffer_fit(
    step_df: pd.DataFrame,
    veq: float,
    veq_unc: float,
    buffer_fit: Dict,
    ph_sys: float,
    method: str = "worst_case",
) -> float:
    """Compute systematic uncertainty in pKa_app from buffer regression.

    This combines three systematic contributions: the regression intercept
    uncertainty, sensitivity of pKa_app to V_eq uncertainty, and the pH meter
    systematic limit. The result is a worst-case bound on the apparent pKa.

    Args:
        step_df: Aggregated volume-step data.
        veq: Equivalence volume in cm^3.
        veq_unc: Systematic uncertainty in V_eq in cm^3.
        buffer_fit: Output dictionary from ``fit_henderson_hasselbalch``.
        ph_sys: Systematic pH meter uncertainty.
        method: Uncertainty combination method (``"worst_case"`` or
            ``"quadrature"``).

    Returns:
        The combined systematic uncertainty in pKa_app. Returns NaN when
        insufficient information is available.
    """
    if step_df.empty or buffer_fit is None:
        return np.nan

    pka0 = buffer_fit.get("pka_app", np.nan)
    if not (
        np.isfinite(pka0) and np.isfinite(veq) and np.isfinite(veq_unc) and veq > 0
    ):
        return np.nan

    ci95 = buffer_fit.get("ci95_intercept", np.nan)
    se = buffer_fit.get("se_intercept", np.nan)
    reg_term = (
        float(ci95) if np.isfinite(ci95) else (float(se) if np.isfinite(se) else np.nan)
    )

    try:
        fit_plus = fit_henderson_hasselbalch(step_df, veq + veq_unc, pka_app_guess=pka0)
        pka_plus = fit_plus.get("pka_app", np.nan)
    except ValueError:
        pka_plus = np.nan
    try:
        fit_minus = fit_henderson_hasselbalch(
            step_df, max(veq - veq_unc, 0.1), pka_app_guess=pka0
        )
        pka_minus = fit_minus.get("pka_app", np.nan)
    except ValueError:
        pka_minus = np.nan

    sens = np.nan
    if np.isfinite(pka_plus) and np.isfinite(pka_minus):
        sens = 0.5 * abs(float(pka_plus) - float(pka_minus))
    elif np.isfinite(pka_plus):
        sens = abs(float(pka_plus) - float(pka0))
    elif np.isfinite(pka_minus):
        sens = abs(float(pka_minus) - float(pka0))

    terms = []
    for t in (reg_term, sens, ph_sys):
        if np.isfinite(t) and t > 0:
            terms.append(float(t))

    return float(combine_uncertainties(terms, method=method)) if terms else np.nan


def analyze_titration(
    df: pd.DataFrame,
    run_name: str,
    x_col: str = "Volume (cm^3)",
    burette_unc: float = DEFAULT_BURETTE_UNC,
    ph_sys: float = DEFAULT_PH_METER_SYS,
    uncertainty_method: str = "worst_case",
    smooth_for_derivative: bool = True,
    savgol_window: int = 7,
    polyorder: int = 2,
) -> Dict:
    """Perform a complete two-stage pKa_app analysis for one titration run.

    Stage 1 determines the equivalence volume from ``d(pH)/dV`` and reads the
    pH at half-equivalence to obtain a coarse apparent pKa estimate. Stage 2
    applies Henderson-Hasselbalch regression within the chemically defined
    buffer region (|pH - pKa_app,initial| ≤ 1) to refine pKa_app.

    Args:
        df: Raw titration data with ``Volume (cm^3)`` and ``pH`` columns.
        run_name: Human-readable identifier for the experimental run.
        x_col: Column name for the independent variable.
        burette_unc: Systematic burette reading uncertainty in cm^3.
        ph_sys: Systematic pH meter uncertainty.
        uncertainty_method: Uncertainty combination rule (``"worst_case"`` or
            ``"quadrature"``).
        smooth_for_derivative: Whether to apply Savitzky-Golay smoothing before
            computing the derivative.
        savgol_window: Window length for Savitzky-Golay smoothing.
        polyorder: Polynomial order for Savitzky-Golay smoothing.

    Returns:
        A dictionary containing pKa_app, V_eq, uncertainties, and diagnostic
        metadata required for plotting and reporting.

    Raises:
        ValueError: If the dataset cannot support the two-stage protocol or if
            the buffer-region regression fails.
    """
    step_df = aggregate_volume_steps(df)
    if step_df.empty:
        raise ValueError("No valid volume/pH data after aggregation.")

    if smooth_for_derivative:
        step_df = _smooth_ph_for_derivative(
            step_df, window=savgol_window, poly=polyorder
        )
    step_df = _ensure_derivative(step_df, use_smooth=smooth_for_derivative)

    interpolator = build_ph_interpolator(step_df, method=None)
    dense_curve = generate_dense_curve(interpolator, n_points=1200)

    eq_info = detect_equivalence_point(step_df, interpolator=interpolator)

    veq_used = float(eq_info.get("eq_x", np.nan))
    veq_method = eq_info.get("method", None)

    veq_unc = _veq_uncertainty(
        step_df, burette_unc=burette_unc, method=uncertainty_method
    )

    half_eq_x = veq_used / 2.0 if np.isfinite(veq_used) and veq_used > 0 else np.nan
    half_eq_pH = (
        float(interpolator["func"](half_eq_x))
        if (np.isfinite(half_eq_x) and "func" in interpolator)
        else np.nan
    )

    if not np.isfinite(half_eq_pH):
        raise ValueError("Half-equivalence pH required for buffer-region selection.")

    buffer_fit = fit_henderson_hasselbalch(step_df, veq_used, pka_app_guess=half_eq_pH)

    buffer_df = buffer_fit.get("buffer_df", pd.DataFrame())

    pka_app = float(buffer_fit.get("pka_app", np.nan))
    pka_method = "buffer_regression"
    pka_unc = _pka_app_unc_from_buffer_fit(
        step_df,
        veq_used,
        veq_unc,
        buffer_fit,
        ph_sys=ph_sys,
        method=uncertainty_method,
    )

    veq_round, veq_unc_round = round_value_to_uncertainty(veq_used, veq_unc)
    pka_round, pka_unc_round = round_value_to_uncertainty(pka_app, pka_unc)

    return {
        "run_name": run_name,
        "eq_x": eq_info.get("eq_x", np.nan),
        "eq_pH": eq_info.get("eq_pH", np.nan),
        "eq_qc_pass": bool(eq_info.get("qc_pass", False)),
        "eq_qc_reason": eq_info.get("qc_reason", ""),
        "veq_used": veq_used,
        "veq_method": veq_method,
        "veq_uncertainty": veq_unc,
        "veq_used_rounded": veq_round,
        "veq_uncertainty_rounded": veq_unc_round,
        "half_eq_x": half_eq_x,
        "half_eq_pH": half_eq_pH,
        "pka_app": pka_app,
        "pka_method": pka_method,
        "pka_app_uncertainty": pka_unc,
        "pka_app_rounded": pka_round,
        "pka_app_uncertainty_rounded": pka_unc_round,
        "slope_reg": buffer_fit.get("slope_reg", np.nan),
        "r2_reg": buffer_fit.get("r2_reg", np.nan),
        "x_col": x_col,
        "data": df,
        "step_data": step_df,
        "dense_curve": dense_curve,
        "buffer_region": buffer_df,
        "diagnostics": {
            "interpolator_method": interpolator.get("method"),
            "step_points": int(len(step_df)),
            "buffer_points": int(buffer_fit.get("n_points", 0)),
            "equivalence_qc": eq_info.get("qc_diagnostics", {}),
        },
    }


def process_all_files(file_list):
    """Process multiple titration files with strict scientific validation.

    Args:
        file_list: Iterable of ``(filepath, nacl_concentration)`` tuples.
            Each filepath must point to a Logger Pro CSV export and each
            concentration must be expressed in mol/L.

    Returns:
        A list of analysis result dictionaries from ``analyze_titration`` with
        ``nacl_conc`` and ``source_file`` fields attached.

    Raises:
        ValueError: If any run lacks required volume data or if analysis fails.
        FileNotFoundError: If any input file does not exist.
    """
    results = []

    for filepath, nacl_conc in file_list:
        print(f"Processing {filepath} (NaCl: {nacl_conc} M)...")
        df_raw = load_titration_data(filepath)
        runs = extract_runs(df_raw)

        for run_name, run_info in runs.items():
            run_df = run_info["df"]
            x_col = run_info["x_col"]

            if "Volume (cm^3)" not in run_df.columns or len(run_df) < 10:
                import logging

                logger = logging.getLogger(__name__)
                msg = (
                    "Skipping run '%s' in file '%s': only %d paired "
                    "pH/volume rows (need >= 10)."
                )
                logger.warning(msg, run_name, os.path.basename(filepath), len(run_df))
                continue

            analysis = analyze_titration(
                run_df, f"{nacl_conc}M - {run_name}", x_col=x_col
            )
            analysis["nacl_conc"] = nacl_conc
            analysis["source_file"] = os.path.basename(filepath)
            results.append(analysis)

    return results


def create_results_dataframe(results):
    """Convert analysis results into a standardized results DataFrame.

    Args:
        results: List of dictionaries returned by ``process_all_files``.

    Returns:
        A DataFrame with standardized column names for pKa_app values,
        uncertainties, equivalence diagnostics, and provenance metadata.

    Raises:
        KeyError: If required result fields are missing from any entry.
    """
    rows = []
    cols = ResultColumns()
    required_keys = {
        "run_name",
        "nacl_conc",
        "pka_app",
        "pka_method",
        "pka_app_uncertainty",
        "eq_qc_pass",
        "veq_used",
        "slope_reg",
        "r2_reg",
        "source_file",
    }
    for res in results:
        missing = required_keys - set(res.keys())
        if missing:
            raise KeyError(f"Result entry missing required keys: {missing}.")
        rows.append(
            {
                "Run": res.get("run_name"),
                cols.nacl: res.get("nacl_conc", np.nan),
                cols.pka_app: res.get("pka_app", np.nan),
                "pKa_app method": res.get("pka_method", ""),
                cols.pka_unc: res.get("pka_app_uncertainty", np.nan),
                "Apparent pKa (rounded)": res.get("pka_app_rounded", np.nan),
                "Apparent pKa uncertainty (rounded)": res.get(
                    "pka_app_uncertainty_rounded", np.nan
                ),
                "Equivalence QC Pass": bool(res.get("eq_qc_pass", False)),
                "Veq (used)": res.get("veq_used", np.nan),
                "Veq uncertainty (ΔVeq)": res.get("veq_uncertainty", np.nan),
                "Veq (used, rounded)": res.get("veq_used_rounded", np.nan),
                "Veq uncertainty (rounded)": res.get("veq_uncertainty_rounded", np.nan),
                "Veq method": res.get("veq_method", ""),
                "Slope (buffer fit)": res.get("slope_reg", np.nan),
                "R2 (buffer fit)": res.get("r2_reg", np.nan),
                "Source File": res.get("source_file", ""),
            }
        )
    return pd.DataFrame(rows)


def calculate_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean pKa_app per ionic strength with systematic uncertainty.

    The statistical summary reports the mean pKa_app for each NaCl
    concentration and expresses uncertainty as a systematic half-range
    (max - min)/2 across trials. This is a worst-case estimate and is not a
    statistical standard deviation.

    Args:
        results_df: DataFrame produced by ``create_results_dataframe``.

    Returns:
        A DataFrame with mean pKa_app, systematic uncertainty, and trial counts.

    Raises:
        KeyError: If required result columns are missing.
    """
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                ResultColumns().nacl,
                "Mean Apparent pKa",
                "Uncertainty",
                "n",
            ]
        )

    cols = ResultColumns()
    required = {cols.nacl, cols.pka_app}
    missing = required - set(results_df.columns)
    if missing:
        raise KeyError(f"results_df missing required columns: {missing}.")

    rows = []
    grouped = results_df.groupby(cols.nacl)

    for conc, group in grouped:
        vals = pd.to_numeric(group[cols.pka_app], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(len(vals))
        mean_pka_app = float(np.mean(vals)) if n else np.nan

        if n >= 2:
            unc = 0.5 * float(np.max(vals) - np.min(vals))
        elif n == 1:
            u = pd.to_numeric(group[cols.pka_unc], errors="coerce").to_numpy(
                dtype=float
            )
            u = u[np.isfinite(u)]
            unc = float(u[0]) if len(u) else np.nan
        else:
            unc = np.nan

        rows.append(
            {
                cols.nacl: conc,
                "Mean Apparent pKa": mean_pka_app,
                "Uncertainty": unc,
                "n": n,
            }
        )

    return (
        pd.DataFrame.from_records(rows)
        .sort_values(ResultColumns().nacl)
        .reset_index(drop=True)
    )


def build_summary_plot_data(
    stats_df: pd.DataFrame, results_df: pd.DataFrame
) -> Dict[str, object]:
    """Prepare validated plotting inputs for summary figures.

    This function performs no plotting and no chemistry. It simply validates
    required columns, computes helper arrays, and packages the data for
    ``plot_statistical_summary``.

    Args:
        stats_df: Output of ``calculate_statistics``.
        results_df: Output of ``create_results_dataframe``.

    Returns:
        A dictionary of NumPy arrays and metadata suitable for plotting.

    Raises:
        KeyError: If required columns are missing.
    """
    cols = ResultColumns()
    required_cols = [cols.nacl, "Mean Apparent pKa", "Uncertainty"]
    missing = [c for c in required_cols if c not in stats_df.columns]
    if missing:
        raise KeyError(
            "stats_df is missing required columns: " + ", ".join(map(str, missing))
        )

    required_results_cols = [cols.nacl, cols.pka_app]
    missing_results = [c for c in required_results_cols if c not in results_df.columns]
    if missing_results:
        raise KeyError(
            "results_df is missing required columns: "
            + ", ".join(map(str, missing_results))
        )

    x = pd.to_numeric(stats_df[cols.nacl], errors="coerce").to_numpy(dtype=float)
    y_mean = pd.to_numeric(stats_df["Mean Apparent pKa"], errors="coerce").to_numpy(
        dtype=float
    )
    yerr = pd.to_numeric(stats_df["Uncertainty"], errors="coerce").to_numpy(dtype=float)
    yerr = np.where(np.isfinite(yerr), yerr, 0.0)
    xerr = np.array(
        [concentration_uncertainty(c) if np.isfinite(c) else 0.0 for c in x],
        dtype=float,
    )

    individual = []
    conc_values = np.unique(x[np.isfinite(x)])
    for conc in conc_values:
        subset = results_df[results_df[cols.nacl] == conc]
        if subset.empty:
            continue

        vals = pd.to_numeric(subset[cols.pka_app], errors="coerce").to_numpy(
            dtype=float
        )
        mask = np.isfinite(vals)
        vals = vals[mask]
        if len(vals) == 0:
            continue

        n = len(vals)
        jitter = 0.012 if n > 1 else 0.0
        xs = conc + np.linspace(-jitter, jitter, n)

        this_xerr = concentration_uncertainty(float(conc))

        yerrs_ind = None
        if cols.pka_unc in subset.columns:
            tmp = pd.to_numeric(subset[cols.pka_unc], errors="coerce").to_numpy(
                dtype=float
            )[mask]
            if np.any(np.isfinite(tmp)) and np.any(tmp > 0):
                yerrs_ind = tmp

        individual.append({"x": xs, "y": vals, "xerr": this_xerr, "yerr": yerrs_ind})

    fit = {"m": np.nan, "b": np.nan, "r2": np.nan}
    finite = np.isfinite(x) & np.isfinite(y_mean)
    if np.sum(finite) >= 2:
        try:
            reg = linear_regression(x[finite], y_mean[finite], min_points=2)
            fit = {"m": reg["m"], "b": reg["b"], "r2": reg["r2"]}
        except ValueError:
            fit = {"m": np.nan, "b": np.nan, "r2": np.nan}

    slope_uncertainty = None
    if np.sum(finite) >= 2:
        try:
            slope_uncertainty = slope_uncertainty_from_endpoints(
                x[finite],
                y_mean[finite],
                xerr[finite],
                yerr[finite],
            )
        except ValueError:
            slope_uncertainty = None

    return {
        "x": x,
        "y_mean": y_mean,
        "xerr": xerr,
        "yerr": yerr,
        "individual": individual,
        "fit": fit,
        "slope_uncertainty": slope_uncertainty,
    }


def print_statistics(stats_df: pd.DataFrame, results_df: pd.DataFrame):
    """Print a human-readable summary of pKa_app statistics.

    Args:
        stats_df: Summary statistics DataFrame.
        results_df: Per-run results DataFrame.
    """
    print("\nStatistical summary by NaCl concentration (IB-style):")
    if stats_df.empty:
        print("  (no data)")
        return

    cols = ResultColumns()
    for _, row in stats_df.iterrows():
        conc = row[cols.nacl]
        mean = row["Mean Apparent pKa"]
        unc = row.get("Uncertainty")
        n = int(row["n"])

        if pd.notna(unc):
            print(f" - {conc} M: Mean apparent pKa = {mean:.3f} ± {unc:.3f} (n={n})")
        else:
            base = f" - {conc} M: Mean apparent pKa = {mean:.3f}"
            extras = f" (uncertainty N/A) (n={n})"
            print(base + extras)

        subset = results_df[results_df[cols.nacl] == conc]
        for _, r in subset.iterrows():
            pka = r.get(cols.pka_app)
            pka_unc = r.get(cols.pka_unc)
            qc = r.get("Equivalence QC Pass")
            veq = r.get("Veq (used)")
            method = r.get("Veq method", "")
            pka_method = r.get("pKa_app method", "")
            if (
                "Apparent pKa (rounded)" in r
                and "Apparent pKa uncertainty (rounded)" in r
                and pd.notna(r.get("Apparent pKa uncertainty (rounded)"))
            ):
                pka = r.get("Apparent pKa (rounded)")
                pka_unc = r.get("Apparent pKa uncertainty (rounded)")
            if "Veq (used, rounded)" in r and pd.notna(r.get("Veq (used, rounded)")):
                veq = r.get("Veq (used, rounded)")
            if pd.notna(pka_unc):
                base = f"     {r['Run']}: pKa_app={pka:.3f} ± {pka_unc:.3f}"
                extras = f" ({pka_method}) | Veq={veq:.3f} ({method}) | QC: {qc}"
                print(base + extras)
            else:
                base = f"     {r['Run']}: pKa_app={pka:.3f} ({pka_method})"
                extras = f" | Veq={veq:.3f} ({method}) | QC: {qc}"
                print(base + extras)
