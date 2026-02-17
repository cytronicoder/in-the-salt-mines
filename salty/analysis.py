"""Run the core two-stage IB IA titration analysis workflow.

This module corresponds to the central IA processing sequence:
1. Detect equivalence behavior from derivative information.
2. Estimate half-equivalence pH for a Stage-1 apparent pKa anchor.
3. Fit a buffer-region Henderson-Hasselbalch linear model for Stage-2
   apparent pKa reporting.

Key outputs include run-level apparent pKa, equivalence diagnostics, and
uncertainty terms required for IB-style reporting and QC subset filtering.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from typing import Dict, List, Tuple

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
)

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
HAVE_SAVGOL = HAVE_SCIPY and (importlib.util.find_spec("scipy.signal") is not None)
if HAVE_SCIPY:
    from scipy.interpolate import PchipInterpolator
if HAVE_SAVGOL:
    from scipy.signal import find_peaks, savgol_filter
else:
    find_peaks = None
    savgol_filter = None

DEFAULT_BURETTE_UNC = 0.10
DEFAULT_BURETTE_READING_UNC = 0.02
DEFAULT_PH_METER_SYS = 0.3


def _prepare_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare x-y arrays for interpolation by sorting and deduplicating.

    Args:
        x (numpy.ndarray): Independent-variable values (cm^3).
        y (numpy.ndarray): Dependent-variable values (pH units).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Cleaned ``(x, y)`` arrays with
        finite pairs only, sorted by ``x``, and deduplicated by averaging
        repeated ``x`` values.

    Note:
        This preprocessing avoids interpolation failures caused by NaNs,
        unsorted points, or duplicate volume entries.
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
        step_df (pandas.DataFrame): Step-level table containing
            ``Volume (cm^3)`` and ``pH_step``.
        method (str | None): Interpolation method (``\"pchip\"`` or ``\"linear\"``).
            If ``None``, select PCHIP when SciPy is available.

    Returns:
        dict: Interpolator payload with callable ``func`` and domain metadata.
        When PCHIP is used, include ``deriv_func`` for first-derivative
        evaluation in units of pH per cm^3.

    Note:
        Prefer PCHIP for monotone, shape-preserving interpolation that avoids
        spline overshoot near the equivalence region.

    References:
        Fritsch-Carlson monotone cubic interpolation (PCHIP).
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
        interpolator (dict): Dictionary returned by ``build_ph_interpolator``.
        n_points (int): Number of grid points over the interpolation domain.

    Returns:
        pandas.DataFrame: Dense interpolation table with columns
        ``Volume (cm^3)`` and ``pH_interp``. Return an empty table if bounds
        are unavailable.

    Note:
        Use this dense table for smooth plotting and reproducible
        half-equivalence pH lookup.

    References:
        Numerical interpolation sampling for curve diagnostics.
    """
    if "x_min" not in interpolator or "x_max" not in interpolator:
        return pd.DataFrame(columns=["Volume (cm^3)", "pH_interp"])
    x_dense = np.linspace(interpolator["x_min"], interpolator["x_max"], n_points)
    y_dense = interpolator["func"](x_dense)
    return pd.DataFrame({"Volume (cm^3)": x_dense, "pH_interp": y_dense})


def _smooth_ph_for_derivative(
    step_df: pd.DataFrame, window: int = 7, poly: int = 2
) -> pd.DataFrame:
    """Smooth stepwise pH values before derivative estimation.

    Args:
        step_df (pandas.DataFrame): Step-level table with ``pH_step`` and
            ``Volume (cm^3)`` columns.
        window (int): Smoothing window length in points. Use odd values.
        poly (int): Polynomial order for Savitzky-Golay smoothing when enabled.

    Returns:
        pandas.DataFrame: Copy of ``step_df`` with added ``pH_smooth`` column
        (pH units).

    Note:
        Fall back to rolling-mean smoothing when SciPy Savitzky-Golay is
        unavailable or fails.
    """
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
    """Ensure derivative column exists for equivalence-point detection.

    Args:
        step_df (pandas.DataFrame): Step-level dataframe containing volume and pH.
        use_smooth (bool): Compute gradient from ``pH_smooth`` when available.

    Returns:
        pandas.DataFrame: Copy with ``dpH/dx`` column in pH per cm^3.

    Note:
        Compute derivatives via ``numpy.gradient`` on finite points only.
    """
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


def _find_peak_candidates(
    derivative: np.ndarray, edge_buffer: int = 2
) -> Tuple[np.ndarray, Dict[int, float], Dict[str, float]]:
    """Find candidate maxima in ``dpH/dx`` for equivalence selection.

    Args:
        derivative (numpy.ndarray): First-derivative series at step points.
        edge_buffer (int): Minimum index distance from each boundary.

    Returns:
        tuple[numpy.ndarray, dict[int, float], dict[str, float]]: Candidate
        indices, per-index prominence lookup, and peak-detection settings.

    Note:
        Prefer SciPy peak finding with adaptive prominence thresholds and fall
        back to local-maxima logic when SciPy is unavailable.
    """
    d = np.asarray(derivative, dtype=float)
    n = int(len(d))
    if n == 0:
        return np.array([], dtype=int), {}, {}

    finite = np.isfinite(d)
    if not np.any(finite):
        return np.array([], dtype=int), {}, {}

    d_f = d[finite]
    d_range = float(np.nanmax(d_f) - np.nanmin(d_f))
    q75, q25 = np.nanpercentile(d_f, [75.0, 25.0])
    iqr = float(max(q75 - q25, 0.0))
    min_prominence = float(max(0.05, 0.10 * d_range, 0.50 * iqr))

    settings = {
        "min_prominence": min_prominence,
        "min_width_points": 1.0,
        "edge_buffer": float(edge_buffer),
    }

    prominence_lookup: Dict[int, float] = {}
    candidate_indices: np.ndarray

    if find_peaks is not None:
        work = d.copy()
        floor = float(np.nanmin(d_f) - max(1e-6, 0.05 * max(abs(np.nanmax(d_f)), 1.0)))
        work[~np.isfinite(work)] = floor
        peaks, props = find_peaks(work, prominence=min_prominence, width=1)
        if len(peaks) == 0:
            peaks, props = find_peaks(work, width=1)
        prominences = props.get("prominences", np.full(len(peaks), np.nan))
        prominence_lookup = {
            int(idx): float(prominences[i]) for i, idx in enumerate(peaks)
        }
        candidate_indices = np.asarray(peaks, dtype=int)
    else:
        if n < 3:
            candidate_indices = np.array([], dtype=int)
        else:
            is_peak = (d[1:-1] >= d[:-2]) & (d[1:-1] > d[2:])
            candidate_indices = np.where(is_peak)[0] + 1

    if candidate_indices.size:
        valid = np.isfinite(d[candidate_indices])
        candidate_indices = candidate_indices[valid]

    if candidate_indices.size:
        interior = (candidate_indices > edge_buffer) & (
            candidate_indices < (n - 1 - edge_buffer)
        )
        candidate_indices = candidate_indices[interior]

    return candidate_indices, prominence_lookup, settings


def _candidate_peak_metrics(
    step_df: pd.DataFrame,
    derivative: np.ndarray,
    peak_index: int,
    prominence: float = np.nan,
) -> Dict[str, float]:
    """Compute explainable selection metrics for one equivalence candidate."""
    volumes = pd.to_numeric(step_df["Volume (cm^3)"], errors="coerce").to_numpy(
        dtype=float
    )
    pH_vals = pd.to_numeric(step_df["pH_step"], errors="coerce").to_numpy(dtype=float)
    n = int(len(step_df))

    before_idx = max(int(peak_index) - 2, 0)
    after_idx = min(int(peak_index) + 2, n - 1)
    local_jump = float(pH_vals[after_idx] - pH_vals[before_idx])
    deriv_val = (
        float(derivative[peak_index]) if np.isfinite(derivative[peak_index]) else np.nan
    )
    post_points = int(n - peak_index - 1)

    post_tail_slope = np.nan
    post_monotonic_fraction = np.nan
    post_slice = slice(int(peak_index) + 1, n)
    v_post = volumes[post_slice]
    p_post = pH_vals[post_slice]
    post_mask = np.isfinite(v_post) & np.isfinite(p_post)
    v_post = v_post[post_mask]
    p_post = p_post[post_mask]

    if len(p_post) >= 2:
        diffs = np.diff(p_post)
        post_monotonic_fraction = float(np.mean(diffs >= -0.05))

    if len(p_post) >= 4:
        tail_n = int(min(6, len(p_post)))
        try:
            post_tail_slope = float(
                np.polyfit(v_post[-tail_n:], p_post[-tail_n:], 1)[0]
            )
        except Exception:
            post_tail_slope = np.nan

    score = float(local_jump)
    if np.isfinite(prominence):
        score += 0.35 * float(prominence)
    elif np.isfinite(deriv_val):
        score += 0.15 * float(max(deriv_val, 0.0))
    if np.isfinite(post_monotonic_fraction):
        score += 0.20 * float(post_monotonic_fraction)
    if np.isfinite(post_tail_slope):
        score -= 0.30 * abs(float(post_tail_slope))
    score += 0.10 * float(min(post_points / 6.0, 1.0))

    return {
        "local_jump_pH": float(local_jump),
        "derivative_value": float(deriv_val) if np.isfinite(deriv_val) else np.nan,
        "prominence": float(prominence) if np.isfinite(prominence) else np.nan,
        "post_points": float(post_points),
        "post_monotonic_fraction": (
            float(post_monotonic_fraction)
            if np.isfinite(post_monotonic_fraction)
            else np.nan
        ),
        "post_tail_slope_pH_per_cm3": (
            float(post_tail_slope) if np.isfinite(post_tail_slope) else np.nan
        ),
        "selection_score": float(score),
    }


def _qc_equivalence(
    step_df: pd.DataFrame,
    peak_index: int,
    edge_buffer: int = 2,
    min_post_points: int = 3,
) -> Tuple[bool, str, Dict[str, float | int | bool]]:
    """Run QC checks for a candidate equivalence-point peak index.

    Args:
        step_df (pandas.DataFrame): Step-level dataframe with ``pH_step``.
        peak_index (int): Index of candidate derivative maximum.
        edge_buffer (int): Minimum point distance from each edge.
        min_post_points (int): Minimum number of points required after peak.

    Returns:
        tuple[bool, str, dict[str, float | int | bool]]: QC pass flag, reason
        string, and diagnostics dictionary.

    Note:
        Steepness check uses local pH jump threshold
        ``max(0.5, 0.10 * total_span)`` in pH units.
    """
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
    v_vals = pd.to_numeric(step_df["Volume (cm^3)"], errors="coerce").to_numpy(
        dtype=float
    )
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

    post_slice = slice(int(peak_index) + 1, n)
    post_pH = pH_vals[post_slice]
    post_v = v_vals[post_slice]
    post_mask = np.isfinite(post_pH) & np.isfinite(post_v)
    post_pH = post_pH[post_mask]
    post_v = post_v[post_mask]

    monotonic_threshold = 0.60
    if len(post_pH) >= 2:
        diffs = np.diff(post_pH)
        monotonic_fraction = float(np.mean(diffs >= -0.05))
        diagnostics["post_monotonic_fraction"] = monotonic_fraction
        diagnostics["post_monotonic_threshold"] = float(monotonic_threshold)
        if monotonic_fraction < monotonic_threshold:
            reasons.append("Post-equivalence region not monotonic/stable")
            diagnostics["post_monotonic_flag"] = True
        else:
            diagnostics["post_monotonic_flag"] = False
    else:
        diagnostics["post_monotonic_fraction"] = np.nan
        diagnostics["post_monotonic_threshold"] = float(monotonic_threshold)
        diagnostics["post_monotonic_flag"] = True

    if len(post_pH) >= 4:
        tail_n = int(min(6, len(post_pH)))
        diagnostics["post_tail_points"] = tail_n
        try:
            tail_slope = float(np.polyfit(post_v[-tail_n:], post_pH[-tail_n:], 1)[0])
        except Exception:
            tail_slope = np.nan
        diagnostics["post_tail_slope_pH_per_cm3"] = tail_slope
    else:
        diagnostics["post_tail_points"] = int(len(post_pH))
        diagnostics["post_tail_slope_pH_per_cm3"] = np.nan

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
        1. Compute stepwise ``dpH/dV`` from smoothed step data (numerical gradient).
        2. Detect candidate derivative peaks (prominence-based with edge exclusion).
        3. Apply QC checks per candidate (edge distance, post-coverage, pH jump,
           post-equivalence monotonic behavior).
        4. Score candidates using local pH-jump strength and post-peak behavior,
           then select the best QC-passing candidate.

    The equivalence point is used to calculate the half-equivalence volume:
        V_half = V_eq / 2

    At V_half, [CH₃COOH] ≈ [CH₃COO⁻], so pH ≈ pKa_app (Stage 1 estimate).

    Args:
        step_df (pandas.DataFrame): Step-aggregated table with
            ``Volume (cm^3)`` and ``pH_step``.
        interpolator (dict | None): Optional interpolation payload from
            ``build_ph_interpolator``.
        edge_buffer (int): Minimum point distance from each edge for a valid peak.
        min_post_points (int): Minimum number of post-peak points.
        gate_on_qc (bool): If ``True``, return ``NaN`` for QC-rejected peaks.

    Returns:
        dict: Equivalence result dictionary with ``eq_x`` (cm^3), ``eq_pH``
        (pH units), detection method, QC status, and diagnostics.

    Note:
        IA correspondence: this function implements the endpoint-identification
        step used to define equivalence before half-equivalence and buffer
        analysis. Peak selection is performed on the step-gradient derivative.
        Interpolation is used for pH lookup and dense fallback only. If an
        interval-midpoint endpoint convention is required for final IA text,
        document that as a reporting-layer convention and keep provenance explicit.

        Failure modes include boundary peaks, weak local steepness, or multiple
        comparable maxima; these are exposed through QC flags and warnings.

    References:
        First-derivative equivalence detection for acid-base titration curves.
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
    d_series = pd.to_numeric(df["dpH/dx"], errors="coerce")
    d = d_series.to_numpy(dtype=float)

    step_ok = False
    step_reason = "Derivative all NaN"
    eq_x_step = np.nan
    peak_idx = None
    qc_metrics: Dict[str, float | int | bool] = {}
    candidate_rows: List[Dict] = []
    peak_settings: Dict[str, float] = {}
    selected_row: Dict | None = None

    if np.any(np.isfinite(d)):
        candidate_indices, prominence_lookup, peak_settings = _find_peak_candidates(
            d, edge_buffer=edge_buffer
        )

        if candidate_indices.size == 0:
            finite_idx = np.where(np.isfinite(d))[0]
            interior_idx = finite_idx[
                (finite_idx > edge_buffer) & (finite_idx < (len(d) - 1 - edge_buffer))
            ]
            pool = interior_idx if len(interior_idx) else finite_idx
            if len(pool):
                fallback_idx = int(pool[int(np.nanargmax(d[pool]))])
                candidate_indices = np.array([fallback_idx], dtype=int)

        for idx in candidate_indices:
            prom = float(prominence_lookup.get(int(idx), np.nan))
            qc_pass, qc_reason, cand_qc_metrics = _qc_equivalence(
                df, int(idx), edge_buffer=edge_buffer, min_post_points=min_post_points
            )
            metrics = _candidate_peak_metrics(df, d, int(idx), prominence=prom)
            candidate = {
                "peak_index": int(idx),
                "volume_cm3": float(df.loc[int(idx), "Volume (cm^3)"]),
                "qc_pass": bool(qc_pass),
                "qc_reason": str(qc_reason),
                "qc_metrics": cand_qc_metrics,
            }
            candidate.update(metrics)
            candidate_rows.append(candidate)

        valid_candidates = [c for c in candidate_rows if c["qc_pass"]]
        ranking_pool = valid_candidates if valid_candidates else candidate_rows

        if ranking_pool:
            center_idx = 0.5 * float(len(df) - 1)

            def _rank_key(cand: Dict) -> Tuple[float, float, float, float]:
                prominence = cand["prominence"]
                prominence = float(prominence) if np.isfinite(prominence) else -np.inf
                return (
                    float(cand["selection_score"]),
                    float(cand["local_jump_pH"]),
                    prominence,
                    -abs(float(cand["peak_index"]) - center_idx),
                )

            selected_row = max(ranking_pool, key=_rank_key)
            peak_idx = int(selected_row["peak_index"])
            step_ok = bool(selected_row["qc_pass"])
            step_reason = str(selected_row["qc_reason"])
            qc_metrics = selected_row["qc_metrics"]
            eq_x_step = float(df.loc[peak_idx, "Volume (cm^3)"])

            if len(valid_candidates) == 0:
                msg = (
                    "No equivalence candidate passed QC; selected the highest-"
                    "scoring candidate but QC is not satisfied."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)

            if len(ranking_pool) > 1:
                scores = np.asarray(
                    [c["selection_score"] for c in ranking_pool], dtype=float
                )
                scores = scores[np.isfinite(scores)]
                if len(scores) >= 2:
                    scores = np.sort(scores)[::-1]
                    if scores[1] >= 0.90 * scores[0]:
                        msg = (
                            "Multiple comparable equivalence candidates remain after "
                            "QC/score filtering; result may be unstable."
                        )
                        warnings.warn(msg, UserWarning, stacklevel=2)

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
        "peak_selection": "qc_pass_then_composite_score",
        "candidate_count": int(len(candidate_rows)),
        "peak_detection": peak_settings,
    }
    if peak_idx is not None:
        qc_diagnostics["peak_index"] = int(peak_idx)
    if selected_row is not None:
        qc_diagnostics["selected_peak_score"] = float(selected_row["selection_score"])
        qc_diagnostics["selected_peak_local_jump_pH"] = float(
            selected_row["local_jump_pH"]
        )
    if qc_metrics:
        qc_diagnostics.update(qc_metrics)
    if candidate_rows:
        qc_diagnostics["candidate_peaks"] = [
            {
                "peak_index": int(c["peak_index"]),
                "volume_cm3": float(c["volume_cm3"]),
                "derivative": float(c["derivative_value"]),
                "prominence": (
                    float(c["prominence"]) if np.isfinite(c["prominence"]) else np.nan
                ),
                "local_jump_pH": float(c["local_jump_pH"]),
                "post_points": int(c["post_points"]),
                "post_monotonic_fraction": (
                    float(c["post_monotonic_fraction"])
                    if np.isfinite(c["post_monotonic_fraction"])
                    else np.nan
                ),
                "post_tail_slope_pH_per_cm3": (
                    float(c["post_tail_slope_pH_per_cm3"])
                    if np.isfinite(c["post_tail_slope_pH_per_cm3"])
                    else np.nan
                ),
                "selection_score": float(c["selection_score"]),
                "qc_pass": bool(c["qc_pass"]),
                "qc_reason": str(c["qc_reason"]),
            }
            for c in candidate_rows
        ]

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
        y_dense = interpolator["func"](x_dense)
        d_dense = np.gradient(y_dense, x_dense)
        qc_diagnostics["dense_derivative_source"] = "dense_gradient"

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
    """Estimate systematic uncertainty for equivalence volume.

    Args:
        step_df (pandas.DataFrame): Step-level dataframe containing
            ``Volume (cm^3)``.
        burette_unc (float): Delivered-volume uncertainty term (cm^3).
        method (str): Uncertainty combination mode passed to
            ``combine_uncertainties``.

    Returns:
        float: Estimated ``ΔVeq`` in cm^3.

    Note:
        IA correspondence: this function provides the software-side Veq
        uncertainty term for propagated pKa uncertainty. The current algorithm
        combines half the median step size with the burette delivered-volume
        term. If interval-midpoint half-width uncertainty is reported in the IA
        narrative, reconcile that convention in documentation and final tables.
    """
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
        step_df (pandas.DataFrame): Step-aggregated run data.
        veq (float): Equivalence volume in cm^3.
        veq_unc (float): Equivalence-volume uncertainty in cm^3.
        buffer_fit (dict): Output from ``fit_henderson_hasselbalch``.
        ph_sys (float): pH systematic uncertainty term (pH units).
        method (str): Uncertainty combination method (``\"worst_case\"`` or
            ``\"quadrature\"``).

    Returns:
        float: Combined systematic uncertainty in apparent pKa (pH units).
        Return ``nan`` if inputs are insufficient.
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
        df (pandas.DataFrame): Raw run data with ``Volume (cm^3)`` and ``pH``.
        run_name (str): Human-readable run identifier.
        x_col (str): Independent-variable label stored in outputs.
        burette_unc (float): Burette uncertainty contribution (cm^3) for ``ΔVeq``.
        ph_sys (float): pH systematic contribution (pH units) for ``ΔpKa_app``.
        uncertainty_method (str): Uncertainty combiner mode.
        smooth_for_derivative (bool): Smooth pH before derivative computation.
        savgol_window (int): Savitzky-Golay window size (points).
        polyorder (int): Savitzky-Golay polynomial order.

    Returns:
        dict: Run-level analysis payload including ``veq_used`` (cm^3),
        ``pka_app`` (dimensionless), uncertainty terms, and diagnostic
        dataframes.

    Raises:
        ValueError: If data are insufficient for aggregation, half-equivalence
            interpolation, or Stage-2 buffer-region regression.

    Note:
        IA correspondence: this is the primary single-run method pipeline used
        to generate Veq, half-equivalence pH, regression-based apparent pKa,
        and run-level uncertainty/QC metadata.

        Common failure modes are insufficient valid step points, failed
        half-equivalence interpolation, or too few valid buffer-region points
        for regression; these raise ``ValueError``.

    References:
        Henderson-Hasselbalch regression and first-derivative equivalence detection.
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

    return {
        "run_name": run_name,
        "eq_x": eq_info.get("eq_x", np.nan),
        "eq_pH": eq_info.get("eq_pH", np.nan),
        "eq_qc_pass": bool(eq_info.get("qc_pass", False)),
        "eq_qc_reason": eq_info.get("qc_reason", ""),
        "veq_used": veq_used,
        "veq_method": veq_method,
        "veq_uncertainty": veq_unc,
        "half_eq_x": half_eq_x,
        "half_eq_pH": half_eq_pH,
        "pka_app": pka_app,
        "pka_method": pka_method,
        "pka_app_uncertainty": pka_unc,
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
        file_list (list[tuple[str, float]]): Sequence of
            ``(filepath, nacl_concentration_M)`` pairs.

    Returns:
        list[dict]: Run-level analysis dictionaries with added
        concentration and source-file provenance metadata.

    Raises:
        FileNotFoundError: If an input path does not exist.
        ValueError: If a non-skipped run fails analysis.

    Note:
        Skip runs with fewer than 10 paired volume/pH rows.

    References:
        Batch processing for replicated titration conditions.
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
        results (list[dict]): Run-level payloads from ``process_all_files``.

    Returns:
        pandas.DataFrame: Standardized run table with chemistry outputs,
        uncertainties, fit diagnostics, and provenance columns.

    Raises:
        KeyError: If required keys are missing from any result payload.

    Note:
        Keep standardized column names aligned with ``ResultColumns``.

    References:
        Internal reporting schema in ``salty.schema.ResultColumns``.
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
        eq_diag = (
            res.get("diagnostics", {}).get("equivalence_qc", {})
            if isinstance(res.get("diagnostics", {}), dict)
            else {}
        )
        rows.append(
            {
                "Run": res.get("run_name"),
                cols.nacl: res.get("nacl_conc", np.nan),
                cols.pka_app: res.get("pka_app", np.nan),
                "pKa_app method": res.get("pka_method", ""),
                cols.pka_unc: res.get("pka_app_uncertainty", np.nan),
                "Equivalence QC Pass": bool(res.get("eq_qc_pass", False)),
                "Equivalence QC Reason": res.get("eq_qc_reason", ""),
                "Equivalence Derivative Source": eq_diag.get("derivative_source", ""),
                "Equivalence Peak Selection": eq_diag.get("peak_selection", ""),
                "Veq (used)": res.get("veq_used", np.nan),
                "Veq uncertainty (ΔVeq)": res.get("veq_uncertainty", np.nan),
                "Veq method": res.get("veq_method", ""),
                "V_half (cm^3)": res.get("half_eq_x", np.nan),
                "Slope (buffer fit)": res.get("slope_reg", np.nan),
                "R2 (buffer fit)": res.get("r2_reg", np.nan),
                "Source File": res.get("source_file", ""),
            }
        )
    return pd.DataFrame(rows)


def calculate_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean pKa_app per ionic strength with systematic uncertainty.

    The statistical summary reports mean pKa_app with explicit IB-style
    uncertainty separation:
    - Random uncertainty from repeats: ± 1/2(range)
    - Instrument/propgated uncertainty: mean per-run propagated uncertainty
    - Combined uncertainty: IB convention - max(random, instrument)

    Following IB guidelines, the absolute uncertainty for processed data is
    the larger of the random uncertainty and the instrumental uncertainty.

    Args:
        results_df (pandas.DataFrame): Run-level results table.

    Returns:
        pandas.DataFrame: Condition-level summary with mean apparent pKa,
        random uncertainty, instrument uncertainty, combined uncertainty,
        and replicate count.

    Raises:
        KeyError: If required columns are missing.

    Note:
        Use the reporting convention
        ``combined_uncertainty = max(random, instrument)``.

    References:
        IB-style processed-data uncertainty convention for replicated trials.
    """
    if results_df.empty:
        return pd.DataFrame(
            columns=[
                ResultColumns().nacl,
                "Mean Apparent pKa",
                "Random Uncertainty (±1/2 range)",
                "Instrument Uncertainty (mean propagated)",
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
            random_unc = 0.5 * float(np.max(vals) - np.min(vals))
        elif n == 1:
            random_unc = 0.0
        else:
            random_unc = np.nan

        inst_u = pd.to_numeric(group[cols.pka_unc], errors="coerce").to_numpy(
            dtype=float
        )
        inst_u = inst_u[np.isfinite(inst_u)]
        instrument_unc = float(np.mean(inst_u)) if len(inst_u) else np.nan

        if np.isfinite(random_unc) and np.isfinite(instrument_unc):
            unc = float(max(random_unc, instrument_unc))
        elif np.isfinite(instrument_unc):
            unc = instrument_unc
        elif np.isfinite(random_unc):
            unc = random_unc
        else:
            unc = np.nan

        rows.append(
            {
                cols.nacl: conc,
                "Mean Apparent pKa": mean_pka_app,
                "Random Uncertainty (±1/2 range)": random_unc,
                "Instrument Uncertainty (mean propagated)": instrument_unc,
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
        stats_df (pandas.DataFrame): Condition-level statistics table.
        results_df (pandas.DataFrame): Run-level results table.

    Returns:
        dict[str, object]: Validated plotting payload with arrays
        (`x`, `y_mean`, `xerr`, `yerr`) and trend metadata (`fit`,
        `slope_uncertainty`).

    Raises:
        KeyError: If required columns are missing.

    Note:
        Exclude individual points from summary payload by design; plot those in
        run-level figures.

    References:
        Linear-trend diagnostics with endpoint-based slope bounds.
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

    fit = {"m": np.nan, "b": np.nan, "r2": np.nan}
    finite = np.isfinite(x) & np.isfinite(y_mean)
    if np.sum(finite) >= 2:
        try:
            reg = linear_regression(x[finite], y_mean[finite], min_points=2)
            fit = reg
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
    """Print a human-readable condition and run summary.

    Args:
        stats_df (pandas.DataFrame): Condition-level summary statistics.
        results_df (pandas.DataFrame): Run-level result records.

    Returns:
        None: Print a console audit summary.

    Note:
        Treat this as a convenience audit view; use CSV outputs as canonical
        reporting artifacts.

    References:
        Human-readable analytical workflow audit summaries.
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
        rand_unc = row.get("Random Uncertainty (±1/2 range)")
        inst_unc = row.get("Instrument Uncertainty (mean propagated)")
        n = int(row["n"])

        if pd.notna(unc):
            print(
                f" - {conc} M: Mean apparent pKa = {mean:.3f} ± {unc:.3f} "
                f"(random={rand_unc:.3f}, instrument={inst_unc:.3f}, n={n})"
            )
        else:
            base = f" - {conc} M: Mean apparent pKa = {mean:.3f}"
            extras = f" (uncertainty N/A) (n={n})"
            print(base + extras)

        subset = results_df[results_df[cols.nacl] == conc]
        for _, r in subset.iterrows():
            pka = r.get(cols.pka_app)
            pka_unc = r.get(cols.pka_unc)
            qc = r.get("Equivalence QC Pass")
            qc_reason = str(r.get("Equivalence QC Reason", "") or "")
            deriv_source = str(r.get("Equivalence Derivative Source", "") or "")
            veq = r.get("Veq (used)")
            method = r.get("Veq method", "")
            pka_method = r.get("pKa_app method", "")
            qc_txt = f"QC: {qc}"
            if qc_reason:
                qc_txt += f" ({qc_reason})"
            if deriv_source:
                qc_txt += f", d-source={deriv_source}"
            if pd.notna(pka_unc):
                base = f"     {r['Run']}: pKa_app={pka:.3f} ± {pka_unc:.3f}"
                extras = f" ({pka_method}) | Veq={veq:.3f} ({method}) | {qc_txt}"
                print(base + extras)
            else:
                base = f"     {r['Run']}: pKa_app={pka:.3f} ({pka_method})"
                extras = f" | Veq={veq:.3f} ({method}) | {qc_txt}"
                print(base + extras)
