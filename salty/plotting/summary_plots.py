"""
Creates statistical summary plots from precomputed data.

Plotting functions accept fully computed results and do not compute statistics.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .titration_plots import setup_plot_style


def plot_statistical_summary(summary: Dict, output_dir: str = "output") -> str:
    """
    Black-and-white plot: apparent pKa vs NaCl concentration.

    summary fields:
      - x, y_mean, xerr, yerr
      - individual: list of dicts with x, y, xerr, yerr
      - fit: dict with m, b, r2
      - slope_uncertainty: dict with line endpoints and slope_unc (optional),
        representing a conservative systematic estimate (not a statistical σ).
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    bar_alpha = 0.50
    ecolor_bar = (0, 0, 0, bar_alpha)

    line_alpha = 0.50
    maxmin_color = (0, 0, 0, line_alpha)

    fig, ax = plt.subplots(figsize=(10.8, 6.8))

    x = np.asarray(summary["x"], dtype=float)
    y_mean = np.asarray(summary["y_mean"], dtype=float)
    xerr = np.asarray(summary["xerr"], dtype=float)
    yerr = np.asarray(summary["yerr"], dtype=float)

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
        alpha=1.0,
        label="Mean",
        zorder=10,
    )

    individual: List[Dict] = summary.get("individual", [])
    first_individual_label = True
    for entry in individual:
        ax.errorbar(
            entry["x"],
            entry["y"],
            xerr=entry.get("xerr", None),
            yerr=entry.get("yerr", None),
            fmt="o",
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            ecolor=ecolor_bar,
            elinewidth=1.2,
            capsize=3,
            alpha=1.0,
            label="Individual runs" if first_individual_label else None,
            zorder=5,
        )
        first_individual_label = False

    fit = summary.get("fit", {})
    m_best = fit.get("m", np.nan)
    b_best = fit.get("b", np.nan)
    r2 = fit.get("r2", np.nan)

    finite = np.isfinite(x) & np.isfinite(y_mean)
    if np.sum(finite) >= 2 and np.isfinite(m_best) and np.isfinite(b_best):
        xgrid = np.linspace(float(np.min(x[finite])), float(np.max(x[finite])), 200)
        ax.plot(
            xgrid,
            m_best * xgrid + b_best,
            color="black",
            linewidth=2.2,
            label="Best fit (means)",
        )

    slope_info = summary.get("slope_uncertainty")
    slope_unc = np.nan
    if slope_info:
        ax.plot(
            [slope_info["x1_max"], slope_info["x2_min"]],
            [slope_info["y1_min"], slope_info["y2_max"]],
            color=maxmin_color,
            linestyle="--",
            linewidth=2.0,
            label="Max slope",
            zorder=2,
        )
        ax.plot(
            [slope_info["x1_min"], slope_info["x2_max"]],
            [slope_info["y1_max"], slope_info["y2_min"]],
            color=maxmin_color,
            linestyle=":",
            linewidth=2.2,
            label="Min slope",
            zorder=2,
        )
        slope_unc = slope_info.get("slope_unc", np.nan)

    if np.isfinite(m_best) and np.isfinite(b_best) and np.isfinite(r2):
        if np.isfinite(slope_unc):
            eq = (
                rf"$pK_{{a,\mathrm{{app}}}} = ({m_best:.3f} \pm {slope_unc:.3f})\,c + {b_best:.3f}$"
                + "\n"
                + rf"$R^2 = {r2:.3f}$"
            )
        else:
            eq = (
                rf"$pK_{{a,\mathrm{{app}}}} = {m_best:.3f}\,c + {b_best:.3f}$"
                + "\n"
                + rf"$R^2 = {r2:.3f}$"
            )

        ax.text(
            0.02, 0.02, eq, transform=ax.transAxes, ha="left", va="bottom", fontsize=14
        )

    ax.set_title(
        r"Effect of NaCl concentration on apparent $pK_{a,\mathrm{app}}$",
        fontweight="bold",
    )
    ax.set_xlabel(r"NaCl concentration / mol dm$^{-3}$")
    ax.set_ylabel(r"$pK_{a,\mathrm{app}}$")

    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))

    ax.legend(loc="best")
    fig.tight_layout()

    out_path = os.path.join(output_dir, "statistical_summary.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
