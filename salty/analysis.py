"""
Analyzes weak acid-strong base titration data to estimate equivalence volumes and
apparent pKa (pKa_app) values.

Detects V_eq from the inflection point (max d(pH)/dV) and pKa_app from
buffer-region regression.

Uses PCHIP interpolation for smooth curves and half-equivalence pH estimation.

Estimates uncertainties using IB DP rules, including burette and pH meter contributions.
"""

from __future__ import annotations

import importlib.util
import os
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
if HAVE_SCIPY:
    from scipy.interpolate import PchipInterpolator

    try:
        from scipy.signal import savgol_filter

        _HAVE_SAVGOL = True
    except Exception:
        savgol_filter = None
        _HAVE_SAVGOL = False


DEFAULT_BURETTE_UNC = 0.05
DEFAULT_PH_METER_SYS = 0.00
DEFAULT_PH_METER_RAND = 0.20


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


def _pka_app_unc_from_half_eq(
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


def _pka_app_unc_from_buffer_fit(
    step_df: pd.DataFrame,
    veq: float,
    veq_unc: float,
    buffer_fit: Dict,
    ph_sys: float,
    method: str = "worst_case",
) -> float:
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
        fit_plus = fit_henderson_hasselbalch(
            step_df, veq + veq_unc, pka_app_guess=pka0
        )
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

    buffer_fit = fit_henderson_hasselbalch(
        step_df, veq_used, pka_app_guess=half_eq_pH
    )
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
        "r2_reg": buffer_fit.get("r2", np.nan),
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
    cols = ResultColumns()
    for res in results:
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
                "Apparent pKa (buffer regression)": res.get("pka_app", np.nan),
                "Slope (buffer fit)": res.get("slope_reg", np.nan),
                "R2 (buffer fit)": res.get("r2_reg", np.nan),
                "Source File": res.get("source_file", ""),
            }
        )
    return pd.DataFrame(rows)


def calculate_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean apparent pKa with IB-style uncertainty.

    Uncertainty estimated as half-range to represent systematic variation
    between trials, consistent with IB Chemistry methodology.
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

    rows = []
    cols = ResultColumns()
    grouped = results_df.groupby(cols.nacl)

    for conc, group in grouped:
        vals = pd.to_numeric(group[cols.pka_app], errors="coerce").to_numpy(
            dtype=float
        )
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
    """
    Build plotting inputs without computing statistics inside plotting functions.
    """
    cols = ResultColumns()
    if cols.nacl not in stats_df.columns or "Mean Apparent pKa" not in stats_df.columns:
        raise KeyError(
            f"stats_df must contain '{cols.nacl}' and 'Mean Apparent pKa' columns."
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

        individual.append(
            {"x": xs, "y": vals, "xerr": this_xerr, "yerr": yerrs_ind}
        )

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
            print(
                f" - {conc} M: Mean apparent pKa = {mean:.3f} ± {unc:.3f} (n={n})"
            )
        else:
            print(
                f" - {conc} M: Mean apparent pKa = {mean:.3f} (uncertainty not available) (n={n})"
            )

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
                print(
                    f"     {r['Run']}: pKa_app={pka:.3f} ± {pka_unc:.3f} ({pka_method}) | Veq={veq:.3f} ({method}) | QC: {qc}"
                )
            else:
                print(
                    f"     {r['Run']}: pKa_app={pka:.3f} ({pka_method}) | Veq={veq:.3f} ({method}) | QC: {qc}"
                )
