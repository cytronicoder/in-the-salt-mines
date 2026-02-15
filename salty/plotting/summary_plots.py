"""Render condition-level summary plots for ionic-strength trend evaluation.

This module visualizes grouped apparent pKa outcomes and fit bounds used in
the IA conclusion section.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator

from .titration_plots import save_figure_bundle, setup_plot_style


def plot_statistical_summary(summary: Dict, output_dir: str = "output") -> str:
    """Render the condition-level pKa summary figure.

    Args:
        summary (dict): Plot payload from
            ``salty.analysis.build_summary_plot_data`` containing ``x`` (NaCl,
            mol dm^-3), ``y_mean`` (apparent pKa, dimensionless), ``xerr``
            (mol dm^-3), ``yerr`` (pKa units), and fit metadata.
        output_dir (str, optional): Directory where PNG/PDF/SVG outputs are
            written. Defaults to ``"output"``.

    Returns:
        str: Path to the saved PNG file.

    Raises:
        KeyError: If required fields are missing from ``summary``.

    Note:
        Figure validates whether the reported ionic-strength trend is supported by
        uncertainty-aware means. Failure modes include slope bounds that are too
        wide, poor linear fit (low ``R^2``), or non-overlapping uncertainty
        patterns that suggest heteroscedasticity/outlier influence. IA
        correspondence: this figure supports the final trend claim.

    References:
        Ordinary least squares trend visualization with endpoint-based slope bounds.
    """
    required_fields = {"x", "y_mean", "xerr", "yerr", "individual", "fit"}
    missing = required_fields - set(summary.keys())
    if missing:
        raise KeyError(
            f"summary dict missing required fields: {missing}. "
            f"Expected fields: {required_fields}"
        )

    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    title_fs = 12
    label_fs = 9
    tick_fs = 8
    legend_fs = 7
    eq_fs = 8

    fig, ax = plt.subplots(figsize=(3.8, 3.4), constrained_layout=False)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.86, bottom=0.50)

    x = np.asarray(summary["x"], dtype=float)
    y_mean = np.asarray(summary["y_mean"], dtype=float)
    xerr = np.asarray(summary["xerr"], dtype=float)
    yerr = np.asarray(summary["yerr"], dtype=float)

    slope_info = summary.get("slope_uncertainty")
    finite = np.isfinite(x) & np.isfinite(y_mean)

    def _line_from_two_points(
        x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return float(m), float(b)

    if slope_info and np.sum(finite) >= 2:
        xmin, xmax = float(np.min(x[finite])), float(np.max(x[finite]))
        xgrid = np.linspace(xmin, xmax, 200)

        m_max, b_max = _line_from_two_points(
            float(slope_info["x1_max"]),
            float(slope_info["y1_min"]),
            float(slope_info["x2_min"]),
            float(slope_info["y2_max"]),
        )
        m_min, b_min = _line_from_two_points(
            float(slope_info["x1_min"]),
            float(slope_info["y1_max"]),
            float(slope_info["x2_max"]),
            float(slope_info["y2_min"]),
        )

        y_max = m_max * xgrid + b_max
        y_min = m_min * xgrid + b_min
        y_lo = np.minimum(y_min, y_max)
        y_hi = np.maximum(y_min, y_max)

        ax.fill_between(
            xgrid,
            y_lo,
            y_hi,
            facecolor="0.90",
            alpha=0.14,
            linewidth=0,
            label="Slope uncertainty region",
            zorder=0,
        )
        ax.plot(
            xgrid,
            y_max,
            linestyle="--",
            linewidth=1.0,
            color="0.45",
            alpha=0.55,
            label="Maximum slope bound",
            zorder=1,
        )
        ax.plot(
            xgrid,
            y_min,
            linestyle=":",
            linewidth=1.0,
            color="0.45",
            alpha=0.55,
            label="Minimum slope bound",
            zorder=1,
        )

    fit = summary.get("fit", {})
    m_best = fit.get("m", np.nan)
    b_best = fit.get("b", np.nan)
    r2 = fit.get("r2", np.nan)

    if np.sum(finite) >= 2 and np.isfinite(m_best) and np.isfinite(b_best):
        xmin, xmax = float(np.min(x[finite])), float(np.max(x[finite]))
        xgrid_fit = np.linspace(xmin, xmax, 200)
        ax.plot(
            xgrid_fit,
            m_best * xgrid_fit + b_best,
            color="black",
            linewidth=1.6,
            alpha=0.92,
            label="Best-fit line (fit to means)",
            zorder=3,
        )

    ax.errorbar(
        x,
        y_mean,
        xerr=xerr,
        yerr=yerr,
        fmt="D",
        markersize=6.2,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.25,
        ecolor="black",
        elinewidth=0.85,
        capsize=2.0,
        alpha=1.0,
        label="Mean ± uncertainty",
        zorder=5,
    )

    ax.set_title(
        r"Effect of NaCl concentration on $pK_{a,\mathrm{app}}$ of ethanoic acid",
        fontweight="bold",
        fontsize=title_fs,
        pad=6,
    )
    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$ (±0.01 M)",
        fontsize=label_fs,
        labelpad=4,
    )
    ax.set_ylabel(r"$pK_{a,\mathrm{app}}$ (±0.3)", fontsize=label_fs, labelpad=6)

    ax.set_xlim(-0.05, 0.85)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.tick_params(axis="x", which="major", pad=2)

    ax.yaxis.grid(True, which="major", linewidth=0.6, alpha=0.10)
    ax.xaxis.grid(False)

    eq = None
    if np.isfinite(m_best) and np.isfinite(b_best):
        slope_unc = np.nan
        if slope_info:
            slope_unc = float(slope_info.get("slope_unc", np.nan))

        m_txt = f"{m_best:.3f}"
        b_txt = f"{b_best:.3f}"
        if np.isfinite(slope_unc):
            s_txt = f"{slope_unc:.3f}"
            eq = rf"$pK_{{a,\mathrm{{app}}}} = ({m_txt}\pm{s_txt})\,c + {b_txt}$"
        else:
            eq = rf"$pK_{{a,\mathrm{{app}}}} = {m_txt}\,c + {b_txt}$"
        if np.isfinite(r2):
            eq = eq + rf"  ($R^2={r2:.3f}$)"

    handles, labels = ax.get_legend_handles_labels()
    preferred = [
        "Mean ± uncertainty",
        "Best-fit line (fit to means)",
        "Slope uncertainty region",
        "Maximum slope bound",
        "Minimum slope bound",
    ]
    order = [labels.index(k) for k in preferred if k in labels]
    order += [i for i in range(len(labels)) if i not in order]

    if eq is not None:
        fig.text(
            0.55,
            0.35,
            eq,
            ha="center",
            va="center",
            fontsize=eq_fs,
        )

    fig.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.15),
        ncol=2,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.4,
        handletextpad=0.6,
        labelspacing=0.8,
        fontsize=legend_fs,
    )

    out_path = os.path.join(output_dir, "statistical_summary.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path
