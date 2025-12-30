"""
Plotting module for titration analysis (IB IA–standard, interpretability-first).

Updates in this revision:
- Any units containing powers/subscripts/superscripts are LaTeX-wrapped so Matplotlib renders them correctly:
  * cm^3, dm^-3, cm^-3, min^-1, pK_a, V_eq, etc.
- Error bars are 50% opacity everywhere (raw curves, individual runs, and summary means).
  * Markers remain opaque; only the error-bar artists are semi-transparent.
- Titration curve interpolation remains scientifically appropriate: PCHIP (shape-preserving) when available.

No captions/footnotes and no PDF output.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd

# Prefer PCHIP if available
try:
    from scipy.interpolate import PchipInterpolator  # type: ignore

    _HAS_PCHIP = True
except Exception:
    PchipInterpolator = None
    _HAS_PCHIP = False


# ----------------------------
# Uncertainty helpers (IB-style)
# ----------------------------


def _round_sigfig(x: float, sig: int = 1) -> float:
    """Round x to sig significant figures."""
    if x == 0 or not np.isfinite(x):
        return float(x)
    return float(round(x, sig - int(np.floor(np.log10(abs(x)))) - 1))


def _unc_sigfig(unc: float) -> float:
    """
    IB convention:
    - uncertainties usually 1 s.f.
    - exception: if leading digit is 1, use 2 s.f.
    """
    if unc == 0 or not np.isfinite(unc):
        return float(unc)
    lead = int(abs(unc) / (10 ** np.floor(np.log10(abs(unc)))))
    return _round_sigfig(unc, sig=2 if lead == 1 else 1)


def _concentration_uncertainty(c: float) -> float:
    """
    Absolute uncertainty in NaCl concentration based on balance + flask uncertainties.

    c in mol dm^-3 (M).
    """
    if c == 0.0 or not np.isfinite(c):
        return 0.0

    mw = 58.44  # g/mol
    v = 0.1  # L
    m = c * v * mw  # g

    delta_m = 0.01  # g
    delta_v = 0.0001  # L (0.10 cm^3)

    rel_unc_m = delta_m / m
    rel_unc_v = delta_v / v
    rel_unc_c = (rel_unc_m**2 + rel_unc_v**2) ** 0.5
    return _unc_sigfig(float(c * rel_unc_c))


# ----------------------------
# Regression helper
# ----------------------------


def _linear_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Return (slope, intercept, R^2) for y = m x + b."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(m), float(b), float(r2)


# ----------------------------
# Scientifically sensible titration interpolation (PCHIP)
# ----------------------------


def _prepare_xy_for_interp(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clean and sort x/y for interpolation:
    - remove non-finite
    - sort by x
    - collapse duplicate x values by averaging y (prevents interpolator failure)
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


def _titration_interpolated_curve(
    x: np.ndarray, y: np.ndarray, n_points: int = 900
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth titration curve using PCHIP (shape-preserving monotonic cubic).
    Falls back to dense linear interpolation if SciPy/PCHIP is unavailable.

    Returns: (x_dense, y_dense)
    """
    x, y = _prepare_xy_for_interp(x, y)
    if len(x) < 3:
        if len(x) >= 2:
            x_dense = np.linspace(x.min(), x.max(), max(200, n_points // 4))
            y_dense = np.interp(x_dense, x, y)
            return x_dense, y_dense
        return np.array([]), np.array([])

    x_dense = np.linspace(x.min(), x.max(), n_points)

    if _HAS_PCHIP:
        f = PchipInterpolator(x, y, extrapolate=False)
        y_dense = f(x_dense)
        mask = np.isfinite(y_dense)
        return x_dense[mask], y_dense[mask]

    y_dense = np.interp(x_dense, x, y)
    return x_dense, y_dense


# ----------------------------
# Plot style: Times/serif + mathtext for LaTeX fragments
# ----------------------------


def setup_plot_style():
    """High-legibility style for black-and-white report figures."""
    try:
        if "seaborn-v0_8-whitegrid" in plt.style.available:
            plt.style.use("seaborn-v0_8-whitegrid")
        elif "seaborn-whitegrid" in plt.style.available:
            plt.style.use("seaborn-whitegrid")
        else:
            plt.style.use("default")
    except Exception:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 16,
            "axes.titlesize": 22,
            "axes.labelsize": 18,
            "legend.fontsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.2,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


# ----------------------------
# Main plots
# ----------------------------


def plot_titration_curves(results: List[Dict], output_dir: str = "output") -> List[str]:
    """
    For each run, saves ONE black-and-white figure with 3 panels:
    (1) pH vs Volume (with xerr and yerr) + PCHIP interpolation
    (2) First derivative vs Volume
    (3) Henderson–Hasselbalch diagnostic (scatter + best-fit + eqn/R² bottom-right)

    Returns list of PNG paths.
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    out_paths: List[str] = []

    # Instrument uncertainties (adjust to your apparatus)
    burette_unc = 0.05  # cm^3
    ph_unc = 0.2  # pH

    # Error bar opacity (50% for bars/caps) everywhere
    bar_alpha = 0.50
    ecolor_bar = (0, 0, 0, bar_alpha)  # RGBA

    for i, res in enumerate(results):
        raw_df: pd.DataFrame = res["data"]
        step_df: pd.DataFrame = res["step_data"]
        buffer_df: pd.DataFrame = res.get("buffer_region", pd.DataFrame())

        run_name = str(res.get("run_name", f"Run {i+1}"))
        x_col = res.get("x_col", "Volume (cm³)")
        is_volume = (x_col == "Volume (cm³)") or ("Volume" in x_col)

        # Labels: only wrap units with exponents/sub/superscripts in LaTeX
        x_label = r"Volume of NaOH added / cm$^3$" if is_volume else r"Time / min"
        deriv_label = (
            r"$\mathrm{d\,pH}/\mathrm{d}V$ / (pH cm$^{-3}$)"
            if is_volume
            else r"$\mathrm{d\,pH}/\mathrm{d}t$ / (pH min$^{-1}$)"
        )

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.8))
        fig.suptitle(run_name, fontweight="bold", fontsize=32, y=0.965)

        # -------------------------
        # (1) Titration curve
        # -------------------------
        if x_col not in raw_df.columns or "pH" not in raw_df.columns:
            plt.close(fig)
            continue

        x_raw = pd.to_numeric(raw_df[x_col], errors="coerce").to_numpy(dtype=float)
        y_raw = pd.to_numeric(raw_df["pH"], errors="coerce").to_numpy(dtype=float)

        ax1.errorbar(
            x_raw,
            y_raw,
            xerr=(burette_unc if is_volume else None),
            yerr=ph_unc,
            fmt="o",
            markersize=6,
            markerfacecolor="white",
            markeredgecolor="black",
            ecolor=ecolor_bar,
            elinewidth=1.6,
            capsize=4,
            alpha=1.0,  # markers opaque; bars/caps use RGBA ecolor for transparency
            label="Measurements",
        )

        x_smooth, y_smooth = _titration_interpolated_curve(
            x_raw, y_raw, n_points=900 if is_volume else 600
        )
        if len(x_smooth) > 0:
            ax1.plot(
                x_smooth,
                y_smooth,
                linewidth=2.6,
                color="black",
                label="PCHIP interpolation",
            )

        veq = res.get("veq_used", np.nan)
        if np.isfinite(veq):
            ax1.axvline(
                veq,
                color="black",
                linestyle="--",
                linewidth=1.8,
                label="Equivalence point",
            )

        ax1.set_title("Titration curve", fontweight="bold")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("pH")

        ax1.yaxis.set_major_locator(MaxNLocator(nbins=8))
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if is_volume:
            ax1.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        xvals = x_raw[np.isfinite(x_raw)]
        if len(xvals) > 0:
            xmin, xmax = float(np.min(xvals)), float(np.max(xvals))
            span = (xmax - xmin) if xmax != xmin else 1.0
            ax1.set_xlim(xmin - 0.03 * span, xmax + 0.03 * span)

        ax1.legend(loc="best")

        # -------------------------
        # (2) First derivative
        # -------------------------
        if not step_df.empty and "dpH/dx" in step_df.columns:
            x_step = (
                "Volume (cm³)"
                if ("Volume (cm³)" in step_df.columns and is_volume)
                else x_col
            )
            if x_step in step_df.columns:
                xs = pd.to_numeric(step_df[x_step], errors="coerce").to_numpy(
                    dtype=float
                )
                ys = pd.to_numeric(step_df["dpH/dx"], errors="coerce").to_numpy(
                    dtype=float
                )
                mask = np.isfinite(xs) & np.isfinite(ys)
                ax2.plot(
                    xs[mask],
                    ys[mask],
                    linewidth=2.4,
                    color="black",
                    label="First derivative",
                )
                if np.isfinite(veq):
                    ax2.axvline(
                        veq,
                        color="black",
                        linestyle="--",
                        linewidth=1.6,
                        label="Equivalence point",
                    )
                ax2.axhline(0, color="black", linewidth=1.0, alpha=0.6)

        ax2.set_title("First derivative", fontweight="bold")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(deriv_label)

        ax2.yaxis.set_major_locator(MaxNLocator(nbins=7))
        if is_volume:
            ax2.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        if len(xvals) > 0:
            ax2.set_xlim(ax1.get_xlim())
        ax2.legend(loc="best")

        # -------------------------
        # (3) Henderson–Hasselbalch diagnostic
        # -------------------------
        if not buffer_df.empty and {"log10_ratio", "pH_step"}.issubset(
            buffer_df.columns
        ):
            xhh = pd.to_numeric(buffer_df["log10_ratio"], errors="coerce").to_numpy(
                dtype=float
            )
            yhh = pd.to_numeric(buffer_df["pH_step"], errors="coerce").to_numpy(
                dtype=float
            )
            mask = np.isfinite(xhh) & np.isfinite(yhh)
            xhh = xhh[mask]
            yhh = yhh[mask]

            ax3.scatter(
                xhh,
                yhh,
                s=60,
                edgecolor="black",
                facecolor="white",
                linewidth=1.2,
                label="Buffer region points",
            )

            m, b, r2 = _linear_fit(xhh, yhh)
            if np.isfinite(m) and np.isfinite(b) and len(xhh) >= 2:
                xgrid = np.linspace(float(np.min(xhh)), float(np.max(xhh)), 120)
                ax3.plot(
                    xgrid,
                    m * xgrid + b,
                    color="black",
                    linewidth=2.2,
                    label="Best-fit line",
                )
                ax3.text(
                    0.98,
                    0.02,
                    rf"$\mathrm{{pH}} = {m:.3f}\,x + {b:.3f}$"
                    + "\n"
                    + rf"$R^2 = {r2:.3f}$",
                    transform=ax3.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=14,
                )

        ax3.set_title("Henderson–Hasselbalch", fontweight="bold")
        ax3.set_xlabel(r"$\log_{10}\!\left(\frac{V}{V_{eq}-V}\right)$")
        ax3.set_ylabel("pH")
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax3.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax3.legend(loc="best")

        # Reduce vertical gap between suptitle and subplots
        fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.955], w_pad=2.5)

        source_file = str(res.get("source_file", ""))
        source_base = os.path.splitext(source_file)[0] if source_file else ""
        combined_name = f"{run_name}_{source_base}" if source_base else run_name
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", combined_name).strip("_")

        out_path = os.path.join(output_dir, f"titration_{sanitized}.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def plot_statistical_summary(
    stats_df: pd.DataFrame, results_df: pd.DataFrame, output_dir: str = "output"
) -> str:
    """
    Black-and-white plot: apparent pK_a vs NaCl concentration.

    - Means with xerr (prep) and yerr (IB-style mean uncertainty).
    - Individual runs with xerr (prep) and optional yerr if available.
    - Best-fit line through means.
    - Max/min lines (50% opacity) using BOTH xerr and yerr, for slope-uncertainty.
    - Slope uncertainty: Δm = (m_max - m_min)/2, included in equation.
    - Equation + R^2 bottom-left (math wrapped).
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    # 50% opacity for error bars in the summary plot as well
    bar_alpha = 0.50
    ecolor_bar = (0, 0, 0, bar_alpha)

    # Max/min line opacity (50%)
    line_alpha = 0.50
    maxmin_color = (0, 0, 0, line_alpha)

    fig, ax = plt.subplots(figsize=(10.8, 6.8))

    if (
        "NaCl Concentration (M)" not in stats_df.columns
        or "Mean pKa" not in stats_df.columns
    ):
        plt.close(fig)
        raise KeyError(
            "stats_df must contain 'NaCl Concentration (M)' and 'Mean pKa' columns."
        )

    x = pd.to_numeric(stats_df["NaCl Concentration (M)"], errors="coerce").to_numpy(
        dtype=float
    )
    y_mean = pd.to_numeric(stats_df["Mean pKa"], errors="coerce").to_numpy(dtype=float)

    def _mean_uncertainty(conc: float) -> float:
        subset = results_df[results_df["NaCl Concentration (M)"] == conc]
        if "pKa (buffer regression)" in subset.columns:
            vals = pd.to_numeric(
                subset["pKa (buffer regression)"], errors="coerce"
            ).to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                return _unc_sigfig(0.5 * (np.max(vals) - np.min(vals)))
            if len(vals) == 1 and "pKa uncertainty (ΔpKa)" in subset.columns:
                v = subset["pKa uncertainty (ΔpKa)"].iloc[0]
                if pd.notna(v) and np.isfinite(v):
                    return _unc_sigfig(float(v))
        return 0.0

    yerr = np.array(
        [_mean_uncertainty(c) if np.isfinite(c) else 0.0 for c in x], dtype=float
    )
    xerr = np.array(
        [_concentration_uncertainty(c) if np.isfinite(c) else 0.0 for c in x],
        dtype=float,
    )

    # Means
    ax.errorbar(
        x,
        y_mean,
        xerr=xerr,
        yerr=yerr,
        fmt="D",
        markersize=9,
        markerfacecolor="white",
        markeredgecolor="black",
        ecolor=ecolor_bar,
        elinewidth=1.8,
        capsize=4,
        alpha=1.0,  # markers opaque; bars/caps use RGBA ecolor
        label="Mean",
        zorder=10,
    )

    # Individual runs (also 50% opacity error bars via ecolor)
    if "pKa (buffer regression)" in results_df.columns:
        first_individual_label = True
        conc_values = np.unique(x[np.isfinite(x)])
        for conc in conc_values:
            subset = results_df[results_df["NaCl Concentration (M)"] == conc]
            if subset.empty:
                continue

            vals = pd.to_numeric(
                subset["pKa (buffer regression)"], errors="coerce"
            ).to_numpy(dtype=float)
            mask = np.isfinite(vals)
            vals = vals[mask]
            if len(vals) == 0:
                continue

            n = len(vals)
            jitter = 0.012 if n > 1 else 0.0
            xs = conc + np.linspace(-jitter, jitter, n)

            this_xerr = _concentration_uncertainty(float(conc))

            yerrs_ind = None
            if "pKa uncertainty (ΔpKa)" in subset.columns:
                tmp = pd.to_numeric(
                    subset["pKa uncertainty (ΔpKa)"], errors="coerce"
                ).to_numpy(dtype=float)[mask]
                if np.any(np.isfinite(tmp)) and np.any(tmp > 0):
                    yerrs_ind = tmp

            ax.errorbar(
                xs,
                vals,
                xerr=this_xerr,
                yerr=yerrs_ind,
                fmt="o",
                markersize=7,
                markerfacecolor="white",
                markeredgecolor="black",
                ecolor=ecolor_bar,
                elinewidth=1.2,
                capsize=3,
                alpha=1.0,  # markers opaque; error bars are RGBA
                label="Individual runs" if first_individual_label else None,
                zorder=5,
            )
            first_individual_label = False

    # Best fit through means
    finite = np.isfinite(x) & np.isfinite(y_mean)
    m_best, b_best, r2 = np.nan, np.nan, np.nan
    if np.sum(finite) >= 2:
        m_best, b_best, r2 = _linear_fit(x[finite], y_mean[finite])
        xgrid = np.linspace(float(np.min(x[finite])), float(np.max(x[finite])), 200)
        ax.plot(
            xgrid,
            m_best * xgrid + b_best,
            color="black",
            linewidth=2.2,
            label="Best fit (means)",
        )

    # Max/min lines and slope uncertainty
    slope_unc = np.nan
    if np.sum(finite) >= 2:
        idx = np.argsort(x[finite])
        xf = x[finite][idx]
        yf = y_mean[finite][idx]
        xef = xerr[finite][idx]
        yef = yerr[finite][idx]

        x1, y1, dx1, dy1 = float(xf[0]), float(yf[0]), float(xef[0]), float(yef[0])
        x2, y2, dx2, dy2 = float(xf[-1]), float(yf[-1]), float(xef[-1]), float(yef[-1])

        # Max slope: maximize numerator, minimize denominator
        x1_max = x1 + dx1
        y1_min = y1 - dy1
        x2_min = x2 - dx2
        y2_max = y2 + dy2
        denom_max = x2_min - x1_max

        # Min slope: minimize numerator, maximize denominator
        x1_min = x1 - dx1
        y1_max = y1 + dy1
        x2_max = x2 + dx2
        y2_min = y2 - dy2
        denom_min = x2_max - x1_min

        if denom_max > 0 and denom_min > 0:
            m_max = (y2_max - y1_min) / denom_max
            m_min = (y2_min - y1_max) / denom_min

            ax.plot(
                [x1_max, x2_min],
                [y1_min, y2_max],
                color=maxmin_color,
                linestyle="--",
                linewidth=2.0,
                label="Max slope",
                zorder=2,
            )
            ax.plot(
                [x1_min, x2_max],
                [y1_max, y2_min],
                color=maxmin_color,
                linestyle=":",
                linewidth=2.2,
                label="Min slope",
                zorder=2,
            )

            if np.isfinite(m_max) and np.isfinite(m_min):
                slope_unc = 0.5 * (max(m_max, m_min) - min(m_max, m_min))

    # Equation text bottom-left (math wrapped)
    if np.isfinite(m_best) and np.isfinite(b_best) and np.isfinite(r2):
        if np.isfinite(slope_unc):
            eq = (
                rf"$pK_a = ({m_best:.3f} \pm {slope_unc:.3f})\,c + {b_best:.3f}$"
                + "\n"
                + rf"$R^2 = {r2:.3f}$"
            )
        else:
            eq = (
                rf"$pK_a = {m_best:.3f}\,c + {b_best:.3f}$"
                + "\n"
                + rf"$R^2 = {r2:.3f}$"
            )

        ax.text(
            0.02, 0.02, eq, transform=ax.transAxes, ha="left", va="bottom", fontsize=14
        )

    ax.set_title(r"Effect of NaCl concentration on apparent $pK_a$", fontweight="bold")
    ax.set_xlabel(r"NaCl concentration / mol dm$^{-3}$")
    ax.set_ylabel(r"$pK_a$")

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

    ax.legend(loc="best")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "statistical_summary.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def save_data_to_csv(
    results_df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: str = "output"
) -> Tuple[str, str]:
    """Save processed data to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "individual_results.csv")
    stats_path = os.path.join(output_dir, "statistical_summary.csv")

    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved individual results to {results_path}")
    print(f"Saved statistical summary to {stats_path}")

    return results_path, stats_path
