"""Provide regression utilities used in IA trend and fit diagnostics.

This module supports:
- run-level Henderson-Hasselbalch linear fits,
- condition-level linear trend summaries, and
- conservative slope-uncertainty bounds used in reporting plots.
"""

from __future__ import annotations

import importlib.util
import math
from typing import Dict

import numpy as np

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
if HAVE_SCIPY:
    from scipy.stats import t as student_t


def linear_regression(
    x: np.ndarray, y: np.ndarray, min_points: int = 3
) -> Dict[str, float]:
    """Fit an ordinary least-squares straight line to finite data pairs.

    Args:
        x (numpy.ndarray): Independent variable array (dimensionless or with
            consistent units).
        y (numpy.ndarray): Dependent variable array (for example, pH units).
        min_points (int, optional): Minimum number of finite paired
            observations required. Defaults to ``3``.

    Returns:
        dict[str, float]: Regression diagnostics with keys ``m`` (slope),
        ``b`` (intercept), ``r2`` (coefficient of determination), ``se_m``,
        ``se_b``, ``ci95_m``, ``ci95_b`` (95% half-widths), and ``p_m``
        (p-value for slope).

    Raises:
        ValueError: If there are insufficient valid points or insufficient x/y
            variance.

    Note:
        ``r2`` and standard errors describe statistical scatter only and do not
        include systematic instrument uncertainty.
        IA correspondence: this regression is the computational core of the
        Stage-2 buffer-region fit and grouped trend diagnostics.

    References:
        Ordinary least squares linear regression.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = int(len(x_arr))
    if n < min_points:
        raise ValueError("Insufficient valid data for regression.")

    m, b = np.polyfit(x_arr, y_arr, 1)
    yhat = m * x_arr + b
    resid = y_arr - yhat

    sse = float(np.sum(resid**2))
    sst = float(np.sum((y_arr - y_arr.mean()) ** 2))
    if sst <= 0:
        raise ValueError("Insufficient variance for regression.")
    r2 = 1.0 - sse / sst

    dof = n - 2
    xbar = float(np.mean(x_arr))
    ssxx = float(np.sum((x_arr - xbar) ** 2))
    mse = sse / dof if dof > 0 else np.inf

    se_m = math.nan
    se_b = math.nan
    ci95_m = math.nan
    ci95_b = math.nan
    p_m = math.nan

    if dof > 0 and ssxx > 0:
        se_m = float(np.sqrt(mse / ssxx))
        se_b = float(np.sqrt(mse * (1.0 / n + (xbar**2) / ssxx)))

        if HAVE_SCIPY:
            try:
                t_stat = m / se_m if se_m > 0 else np.inf
                p_m = float(2 * (1 - student_t.cdf(abs(t_stat), dof)))
                t_crit = float(student_t.ppf(0.975, dof))
                ci95_m = t_crit * se_m
                ci95_b = t_crit * se_b
            except Exception:
                pass

    return {
        "m": float(m),
        "b": float(b),
        "r2": float(r2),
        "se_m": se_m,
        "se_b": se_b,
        "ci95_m": ci95_m,
        "ci95_b": ci95_b,
        "p_m": p_m,
        "n": n,
        "dof": dof,
        "mse": mse,
        "ssxx": ssxx,
        "xbar": xbar,
    }


def slope_uncertainty_from_endpoints(
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
) -> Dict[str, float]:
    """Estimate worst-case slope uncertainty from endpoint error bounds.

    Args:
        x (numpy.ndarray): Independent-variable values.
        y (numpy.ndarray): Dependent-variable values.
        xerr (numpy.ndarray): Absolute systematic uncertainty for ``x`` in the
            same units as ``x``.
        yerr (numpy.ndarray): Absolute systematic uncertainty for ``y`` in the
            same units as ``y``.

    Returns:
        dict[str, float]: Dictionary with extreme compatible slopes
        (``m_max``, ``m_min``), half-range uncertainty (``slope_unc``), and
        endpoint bound coordinates used to compute those limits.

    Raises:
        ValueError: If fewer than two finite points are available or if
            endpoint bounds produce non-positive slope denominators.

    Note:
        This is a conservative geometric bound, not a probabilistic confidence
        interval. IA correspondence: this method supports uncertainty-band
        visualization for concentration-trend slopes.

    References:
        Worst-case endpoint bounding for slope uncertainty propagation.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    xerr_arr = np.asarray(xerr, dtype=float)
    yerr_arr = np.asarray(yerr, dtype=float)

    finite = (
        np.isfinite(x_arr)
        & np.isfinite(y_arr)
        & np.isfinite(xerr_arr)
        & np.isfinite(yerr_arr)
    )
    if np.sum(finite) < 2:
        raise ValueError("Insufficient data for slope uncertainty estimate.")

    idx = np.argsort(x_arr[finite])
    xf = x_arr[finite][idx]
    yf = y_arr[finite][idx]
    xef = xerr_arr[finite][idx]
    yef = yerr_arr[finite][idx]

    x1, y1, dx1, dy1 = float(xf[0]), float(yf[0]), float(xef[0]), float(yef[0])
    x2, y2, dx2, dy2 = float(xf[-1]), float(yf[-1]), float(xef[-1]), float(yef[-1])

    x1_max = x1 + dx1
    y1_min = y1 - dy1
    x2_min = x2 - dx2
    y2_max = y2 + dy2
    denom_max = x2_min - x1_max

    x1_min = x1 - dx1
    y1_max = y1 + dy1
    x2_max = x2 + dx2
    y2_min = y2 - dy2
    denom_min = x2_max - x1_min

    if denom_max <= 0 or denom_min <= 0:
        raise ValueError("Invalid endpoint configuration for slope uncertainty.")

    m_max = (y2_max - y1_min) / denom_max
    m_min = (y2_min - y1_max) / denom_min

    slope_unc = 0.5 * (max(m_max, m_min) - min(m_max, m_min))

    return {
        "m_max": float(m_max),
        "m_min": float(m_min),
        "slope_unc": float(slope_unc),
        "x1_max": float(x1_max),
        "x2_min": float(x2_min),
        "y1_min": float(y1_min),
        "y2_max": float(y2_max),
        "x1_min": float(x1_min),
        "x2_max": float(x2_max),
        "y1_max": float(y1_max),
        "y2_min": float(y2_min),
    }
