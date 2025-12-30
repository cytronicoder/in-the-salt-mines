"""
DP Chemistry SL titration analysis.

This module analyzes weak acid-strong base titration data to estimate:
- Equivalence volume (V_eq) from the maximum of d(pH)/dV (inflection point).
- Apparent pK_a from Henderson-Hasselbalch buffer-region linear regression:
    pH = m * log10(V / (V_eq - V)) + b, where b ≈ pK_a.
  The regression is performed for every run (no slope/R^2 gating). Slope and R^2 are reported
  as diagnostics only.

Interpolation (PCHIP if available; otherwise linear) is used to:
- Produce a smooth curve for plotting.
- Estimate pH at half-equivalence (reported as an additional DP reference value).

Uncertainties (IB-defensible, simple):
- ΔV_eq from delivered burette uncertainty (two readings) and half the median volume step
  using worst-case addition (quadrature available only if explicitly requested).
- ΔpK_a from regression intercept uncertainty (95% CI if available, else SE), V_eq sensitivity,
  and optional pH systematic offset using worst-case addition. If regression cannot be performed,
  ΔpK_a is estimated from half-equivalence sensitivity plus pH meter terms.

All reported pK_a values should be treated as apparent pK_a (ionic strength/activity effects can shift values).
"""

from __future__ import annotations

# CHANGELOG:
# - Standardized uncertainty propagation to IB worst-case rules with optional quadrature.
# - Added configurable derivative smoothing and step-derivative evaluation at step volumes.
# - Reported buffer regression diagnostics consistently and avoided equivalence gating by QC.

import importlib.util
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .data_processing import aggregate_volume_steps, extract_runs, load_titration_data
from .uncertainty import (
    burette_delivered_uncertainty,
    combine_uncertainties,
    round_value_to_uncertainty,
)

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
if HAVE_SCIPY:
    from scipy.interpolate import PchipInterpolator
    from scipy.stats import t as student_t

    try:
        from scipy.signal import savgol_filter

        _HAVE_SAVGOL = True
    except Exception:
        savgol_filter = None
        _HAVE_SAVGOL = False


DEFAULT_BURETTE_UNC = 0.05
DEFAULT_PH_METER_SYS = 0.00
DEFAULT_PH_METER_RAND = 0.20


def _linear_regression_with_uncertainty(
    x: np.ndarray, y: np.ndarray
) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))
    if n < 3:
        return {
            "m": np.nan,
            "b": np.nan,
            "r2": np.nan,
            "se_m": np.nan,
            "se_b": np.nan,
            "ci95_b": np.nan,
        }

    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    resid = y - yhat

    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    dof = n - 2
    if dof <= 0:
        return {
            "m": float(m),
            "b": float(b),
            "r2": float(r2),
            "se_m": np.nan,
            "se_b": np.nan,
            "ci95_b": np.nan,
        }

    s2 = sse / dof
    xbar = float(np.mean(x))
    ssxx = float(np.sum((x - xbar) ** 2))
    if ssxx <= 0:
        return {
            "m": float(m),
            "b": float(b),
            "r2": float(r2),
            "se_m": np.nan,
            "se_b": np.nan,
            "ci95_b": np.nan,
        }

    se_m = float(np.sqrt(s2 / ssxx))
    se_b = float(np.sqrt(s2 * (1.0 / n + (xbar**2) / ssxx)))

    ci95_b = np.nan
    if HAVE_SCIPY:
        try:
            tcrit = float(student_t.ppf(0.975, dof))
            ci95_b = float(tcrit * se_b)
        except Exception:
            ci95_b = np.nan

    return {
        "m": float(m),
        "b": float(b),
        "r2": float(r2),
        "se_m": se_m,
        "se_b": se_b,
        "ci95_b": ci95_b,
    }


def _prepare_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    if (
        step_df.empty
        or "Volume (cm³)" not in step_df.columns
        or "pH_step" not in step_df.columns
    ):
        return {
            "method": "linear",
            "func": lambda x: np.full_like(np.asarray(x, dtype=float), np.nan),
        }

    x = step_df["Volume (cm³)"].to_numpy(dtype=float)
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
    if "x_min" not in interpolator or "x_max" not in interpolator:
        return pd.DataFrame(columns=["Volume (cm³)", "pH_interp"])
    x_dense = np.linspace(interpolator["x_min"], interpolator["x_max"], n_points)
    y_dense = interpolator["func"](x_dense)
    return pd.DataFrame({"Volume (cm³)": x_dense, "pH_interp": y_dense})


def _smooth_ph_for_derivative(
    step_df: pd.DataFrame, window: int = 7, poly: int = 2
) -> pd.DataFrame:
    if step_df.empty:
        return step_df

    df = step_df.copy()
    v = pd.to_numeric(df["Volume (cm³)"], errors="coerce").to_numpy(dtype=float)
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

        if HAVE_SCIPY and _HAVE_SAVGOL and w >= 5 and w <= len(pp):
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

    v = pd.to_numeric(df["Volume (cm³)"], errors="coerce").to_numpy(dtype=float)
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

    volumes = pd.to_numeric(df["Volume (cm³)"], errors="coerce").to_numpy(dtype=float)
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
        step_ok, step_reason, qc_metrics = _qc_equivalence(
            df, peak_idx, edge_buffer=edge_buffer, min_post_points=min_post_points
        )
        eq_x_step = float(df.loc[peak_idx, "Volume (cm³)"])
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
                        np.abs(step_df["Volume (cm³)"].to_numpy(dtype=float) - eq_x)
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


def fit_buffer_region(
    step_df: pd.DataFrame, veq: float, window: Tuple[float, float] = (0.2, 0.8)
) -> Dict:
    out = {
        "pka_reg": np.nan,
        "slope_reg": np.nan,
        "r2": np.nan,
        "n_points": 0,
        "se_intercept": np.nan,
        "ci95_intercept": np.nan,
        "buffer_df": pd.DataFrame(
            columns=["Volume (cm³)", "log10_ratio", "pH_step", "pH_fit"]
        ),
        "window": window,
    }

    if not np.isfinite(veq) or veq <= 0 or step_df.empty:
        return out

    volumes = step_df["Volume (cm³)"].to_numpy(dtype=float)
    ph_values = step_df["pH_step"].to_numpy(dtype=float)

    lo, hi = window
    mask = (
        (volumes > lo * veq)
        & (volumes < hi * veq)
        & (volumes < 0.95 * veq)
        & (volumes > 0)
        & np.isfinite(ph_values)
    )
    v = volumes[mask]
    y = ph_values[mask]
    out["n_points"] = int(len(v))
    if len(v) < 4:
        return out

    x = np.log10(v / (veq - v))
    reg = _linear_regression_with_uncertainty(x, y)

    m = reg["m"]
    b = reg["b"]
    r2 = reg["r2"]

    if np.isfinite(m) and np.isfinite(b):
        y_pred = m * x + b
    else:
        y_pred = np.full_like(y, np.nan)

    out.update(
        {
            "pka_reg": float(b),
            "slope_reg": float(m),
            "r2": float(r2),
            "se_intercept": float(reg["se_b"]),
            "ci95_intercept": float(reg["ci95_b"]),
            "buffer_df": pd.DataFrame(
                {"Volume (cm³)": v, "log10_ratio": x, "pH_step": y, "pH_fit": y_pred}
            ),
        }
    )
    return out


def select_best_buffer_fit(step_df: pd.DataFrame, veq: float) -> Dict:
    windows = [(0.25, 0.75), (0.2, 0.8), (0.3, 0.7)]
    fits = [fit_buffer_region(step_df, veq, window=w) for w in windows]
    fits.sort(key=lambda f: (f.get("r2", -np.inf), f.get("n_points", 0)), reverse=True)
    return fits[0] if fits else fit_buffer_region(step_df, veq, window=(0.2, 0.8))


def _veq_uncertainty(
    step_df: pd.DataFrame,
    burette_unc: float = DEFAULT_BURETTE_UNC,
    method: str = "worst_case",
) -> float:
    if step_df.empty or "Volume (cm³)" not in step_df.columns:
        return float(burette_delivered_uncertainty(burette_unc))

    volumes = pd.to_numeric(step_df["Volume (cm³)"], errors="coerce").to_numpy(
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


def _pka_unc_from_half_eq(
    interpolator: Dict,
    veq: float,
    veq_unc: float,
    ph_rand: float,
    ph_sys: float,
    method: str = "worst_case",
) -> float:
    if not (
        interpolator
        and "func" in interpolator
        and np.isfinite(veq)
        and np.isfinite(veq_unc)
        and veq > 0
    ):
        terms = [t for t in [ph_rand, ph_sys] if np.isfinite(t) and t > 0]
        return float(combine_uncertainties(terms, method=method)) if terms else np.nan

    v0 = veq / 2.0
    v_plus = (veq + veq_unc) / 2.0
    v_minus = max((veq - veq_unc) / 2.0, 0.0)

    p0 = float(interpolator["func"](v0))
    pp = float(interpolator["func"](v_plus))
    pm = float(interpolator["func"](v_minus))

    sens = np.nan
    if np.isfinite(pp) and np.isfinite(pm):
        sens = 0.5 * abs(pp - pm)
    elif np.isfinite(pp) and np.isfinite(p0):
        sens = abs(pp - p0)
    elif np.isfinite(pm) and np.isfinite(p0):
        sens = abs(pm - p0)

    terms = []
    for t in (sens, ph_rand, ph_sys):
        if np.isfinite(t) and t > 0:
            terms.append(float(t))
    return float(combine_uncertainties(terms, method=method)) if terms else np.nan


def _pka_unc_from_buffer_fit(
    step_df: pd.DataFrame,
    veq: float,
    veq_unc: float,
    buffer_fit: Dict,
    ph_sys: float,
    method: str = "worst_case",
) -> float:
    if step_df.empty or buffer_fit is None:
        return np.nan

    pka0 = buffer_fit.get("pka_reg", np.nan)
    if not (
        np.isfinite(pka0) and np.isfinite(veq) and np.isfinite(veq_unc) and veq > 0
    ):
        return np.nan

    ci95 = buffer_fit.get("ci95_intercept", np.nan)
    se = buffer_fit.get("se_intercept", np.nan)
    reg_term = (
        float(ci95) if np.isfinite(ci95) else (float(se) if np.isfinite(se) else np.nan)
    )

    window = buffer_fit.get("window", (0.2, 0.8))
    fit_plus = fit_buffer_region(step_df, veq + veq_unc, window=window)
    fit_minus = fit_buffer_region(step_df, max(veq - veq_unc, 0.1), window=window)
    pka_plus = fit_plus.get("pka_reg", np.nan)
    pka_minus = fit_minus.get("pka_reg", np.nan)

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
    x_col: str = "Volume (cm³)",
    burette_unc: float = DEFAULT_BURETTE_UNC,
    ph_sys: float = DEFAULT_PH_METER_SYS,
    ph_rand: float = DEFAULT_PH_METER_RAND,
    uncertainty_method: str = "worst_case",
    smooth_for_derivative: bool = True,
    savgol_window: int = 7,
    polyorder: int = 2,
    buffer_window: Tuple[float, float] = (0.2, 0.8),
) -> Dict:
    step_df = aggregate_volume_steps(df)
    if step_df.empty:
        return {
            "run_name": run_name,
            "skip_reason": "No valid volume/pH data after aggregation",
            "x_col": x_col,
            "data": df,
            "step_data": step_df,
            "dense_curve": pd.DataFrame(),
            "buffer_region": pd.DataFrame(),
        }

    if smooth_for_derivative:
        # Mild smoothing improves derivative stability without overfitting.
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

    buffer_fit = (
        fit_buffer_region(step_df, veq_used, window=buffer_window)
        if np.isfinite(veq_used)
        else None
    )
    buffer_df = (
        buffer_fit.get("buffer_df", pd.DataFrame()) if buffer_fit else pd.DataFrame()
    )

    pka_reg = buffer_fit.get("pka_reg", np.nan) if buffer_fit else np.nan
    if np.isfinite(pka_reg):
        pka_used = float(pka_reg)
        pka_method = "buffer_regression"
        pka_unc = _pka_unc_from_buffer_fit(
            step_df,
            veq_used,
            veq_unc,
            buffer_fit,
            ph_sys=ph_sys,
            method=uncertainty_method,
        )
    else:
        pka_used = float(half_eq_pH) if np.isfinite(half_eq_pH) else np.nan
        pka_method = "half_equivalence" if np.isfinite(half_eq_pH) else "unknown"
        pka_unc = _pka_unc_from_half_eq(
            interpolator,
            veq_used,
            veq_unc,
            ph_rand=ph_rand,
            ph_sys=ph_sys,
            method=uncertainty_method,
        )

    veq_round, veq_unc_round = round_value_to_uncertainty(veq_used, veq_unc)
    pka_round, pka_unc_round = round_value_to_uncertainty(pka_used, pka_unc)

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
        "pka_used": pka_used,
        "pka_method": pka_method,
        "pka_uncertainty": pka_unc,
        "pka_used_rounded": pka_round,
        "pka_uncertainty_rounded": pka_unc_round,
        "pka_reg": pka_reg,
        "slope_reg": buffer_fit.get("slope_reg", np.nan) if buffer_fit else np.nan,
        "r2_reg": buffer_fit.get("r2", np.nan) if buffer_fit else np.nan,
        "buffer_window": buffer_fit.get("window", None) if buffer_fit else None,
        "x_col": x_col,
        "data": df,
        "step_data": step_df,
        "dense_curve": dense_curve,
        "buffer_region": buffer_df,
        "diagnostics": {
            "interpolator_method": interpolator.get("method"),
            "step_points": int(len(step_df)),
            "buffer_points": int(buffer_fit.get("n_points", 0)) if buffer_fit else 0,
            "equivalence_qc": eq_info.get("qc_diagnostics", {}),
        },
    }


def process_all_files(file_list):
    results = []

    for filepath, nacl_conc in file_list:
        print(f"Processing {filepath} (NaCl: {nacl_conc} M)...")
        try:
            df_raw = load_titration_data(filepath)
            runs = extract_runs(df_raw)

            for run_name, run_info in runs.items():
                run_df = run_info["df"]
                x_col = run_info["x_col"]

                if "Volume (cm³)" not in run_df.columns or len(run_df) < 10:
                    continue

                analysis = analyze_titration(
                    run_df, f"{nacl_conc}M - {run_name}", x_col=x_col
                )
                if analysis.get("skip_reason"):
                    continue

                analysis["nacl_conc"] = nacl_conc
                analysis["source_file"] = os.path.basename(filepath)
                results.append(analysis)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    return results


def create_results_dataframe(results):
    rows = []
    for res in results:
        rows.append(
            {
                "Run": res.get("run_name"),
                "NaCl Concentration (M)": res.get("nacl_conc", np.nan),
                "pKa (used)": res.get("pka_used", np.nan),
                "pKa method": res.get("pka_method", ""),
                "pKa uncertainty (ΔpKa)": res.get("pka_uncertainty", np.nan),
                "pKa (used, rounded)": res.get("pka_used_rounded", np.nan),
                "pKa uncertainty (rounded)": res.get(
                    "pka_uncertainty_rounded", np.nan
                ),
                "Equivalence QC Pass": bool(res.get("eq_qc_pass", False)),
                "Veq (used)": res.get("veq_used", np.nan),
                "Veq uncertainty (ΔVeq)": res.get("veq_uncertainty", np.nan),
                "Veq (used, rounded)": res.get("veq_used_rounded", np.nan),
                "Veq uncertainty (rounded)": res.get(
                    "veq_uncertainty_rounded", np.nan
                ),
                "Veq method": res.get("veq_method", ""),
                "pKa (buffer regression)": res.get("pka_reg", np.nan),
                "Slope (buffer fit)": res.get("slope_reg", np.nan),
                "R2 (buffer fit)": res.get("r2_reg", np.nan),
                "Buffer window": res.get("buffer_window", None),
                "Source File": res.get("source_file", ""),
            }
        )
    return pd.DataFrame(rows)


def calculate_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(
            columns=["NaCl Concentration (M)", "Mean pKa", "Uncertainty", "n"]
        )

    rows = []
    grouped = results_df.groupby("NaCl Concentration (M)")

    for conc, group in grouped:
        vals = pd.to_numeric(group["pKa (used)"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        n = int(len(vals))
        mean_pka = float(np.mean(vals)) if n else np.nan

        if n >= 2:
            unc = 0.5 * float(np.max(vals) - np.min(vals))
        elif n == 1:
            u = pd.to_numeric(
                group["pKa uncertainty (ΔpKa)"], errors="coerce"
            ).to_numpy(dtype=float)
            u = u[np.isfinite(u)]
            unc = float(u[0]) if len(u) else np.nan
        else:
            unc = np.nan

        rows.append(
            {
                "NaCl Concentration (M)": conc,
                "Mean pKa": mean_pka,
                "Uncertainty": unc,
                "n": n,
            }
        )

    return (
        pd.DataFrame.from_records(rows)
        .sort_values("NaCl Concentration (M)")
        .reset_index(drop=True)
    )


def print_statistics(stats_df: pd.DataFrame, results_df: pd.DataFrame):
    print("\nStatistical summary by NaCl concentration (IB-style):")
    if stats_df.empty:
        print("  (no data)")
        return

    for _, row in stats_df.iterrows():
        conc = row["NaCl Concentration (M)"]
        mean = row["Mean pKa"]
        unc = row.get("Uncertainty")
        n = int(row["n"])

        if pd.notna(unc):
            print(f" - {conc} M: Mean pKa = {mean:.3f} ± {unc:.3f} (n={n})")
        else:
            print(
                f" - {conc} M: Mean pKa = {mean:.3f} (uncertainty not available) (n={n})"
            )

        subset = results_df[results_df["NaCl Concentration (M)"] == conc]
        for _, r in subset.iterrows():
            pka = r.get("pKa (used)")
            pka_unc = r.get("pKa uncertainty (ΔpKa)")
            qc = r.get("Equivalence QC Pass")
            veq = r.get("Veq (used)")
            method = r.get("Veq method", "")
            pka_method = r.get("pKa method", "")
            if (
                "pKa (used, rounded)" in r
                and "pKa uncertainty (rounded)" in r
                and pd.notna(r.get("pKa uncertainty (rounded)"))
            ):
                pka = r.get("pKa (used, rounded)")
                pka_unc = r.get("pKa uncertainty (rounded)")
            if "Veq (used, rounded)" in r and pd.notna(
                r.get("Veq (used, rounded)")
            ):
                veq = r.get("Veq (used, rounded)")
            if pd.notna(pka_unc):
                print(
                    f"     {r['Run']}: pKa={pka:.3f} ± {pka_unc:.3f} ({pka_method}) | Veq={veq:.3f} ({method}) | QC: {qc}"
                )
            else:
                print(
                    f"     {r['Run']}: pKa={pka:.3f} ({pka_method}) | Veq={veq:.3f} ({method}) | QC: {qc}"
                )
