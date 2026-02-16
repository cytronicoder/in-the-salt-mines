"""Render condition-level summary plots for ionic-strength trend evaluation.

This module visualizes grouped apparent pKa outcomes and fit bounds used in
the IA conclusion section.
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from scipy.stats import t as student_t

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
        Figure is formatted for IA presentation with a single OLS line,
        95% confidence band for mean response, in-axes fit summary, and
        right-side legend.

    References:
        Ordinary least squares trend visualization with confidence band.
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

    title_fs = 16
    label_fs = 13
    tick_fs = 12
    legend_fs = 12
    data_color = "#004371"
    line_color = "#a50f15"
    ci_color = "#f1c4c1"

    fig = plt.figure(figsize=(7.8, 3.9), constrained_layout=False)
    gs = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[3.6, 1.4],
        wspace=0.06,
        left=0.08,
        right=0.98,
        top=0.83,
        bottom=0.16,
    )
    ax = fig.add_subplot(gs[0, 0])
    gutter_ax = fig.add_subplot(gs[0, 1])
    gutter_ax.axis("off")

    x = np.asarray(summary["x"], dtype=float)
    y_mean = np.asarray(summary["y_mean"], dtype=float)
    yerr = np.asarray(summary["yerr"], dtype=float)

    finite = np.isfinite(x) & np.isfinite(y_mean)
    fit = summary.get("fit", {})
    m_best = fit.get("m", np.nan)
    b_best = fit.get("b", np.nan)
    r2 = fit.get("r2", np.nan)
    dof = fit.get("dof", np.nan)
    n_fit = int(fit.get("n", np.sum(finite))) if np.sum(finite) else 0
    r2_adj = np.nan
    if np.isfinite(r2) and n_fit > 2:
        r2_adj = 1.0 - (1.0 - float(r2)) * ((n_fit - 1.0) / (n_fit - 2.0))

    x_pad = 0.05
    x_lo = -x_pad
    x_hi = 0.8 + x_pad

    if np.sum(finite) >= 2 and np.isfinite(m_best) and np.isfinite(b_best):
        xgrid_fit = np.linspace(x_lo, x_hi, 300)

        # 95% confidence band for mean predicted y
        mse = fit.get("mse", np.nan)
        xbar = fit.get("xbar", np.nan)
        ssxx = fit.get("ssxx", np.nan)

        if (
            np.isfinite(mse)
            and np.isfinite(dof)
            and np.isfinite(xbar)
            and np.isfinite(ssxx)
            and dof > 0
            and ssxx > 0
        ):
            n = dof + 2
            t_crit = float(student_t.ppf(0.975, dof))

            y_hat = m_best * xgrid_fit + b_best
            se_fit = np.sqrt(mse * (1 / n + (xgrid_fit - xbar) ** 2 / ssxx))
            ci_half = t_crit * se_fit

            ax.fill_between(
                xgrid_fit,
                y_hat - ci_half,
                y_hat + ci_half,
                facecolor=ci_color,
                alpha=0.45,
                linewidth=0,
                label="95% CI (mean)",
                zorder=1,
                edgecolor="none",
            )

        ax.plot(
            xgrid_fit,
            m_best * xgrid_fit + b_best,
            color=line_color,
            linewidth=1.8,
            alpha=0.95,
            label="Linear fit",
            zorder=3,
        )

    ax.errorbar(
        x,
        y_mean,
        yerr=yerr,
        fmt="none",
        ecolor=data_color,
        elinewidth=1.1,
        capsize=2.4,
        alpha=0.95,
        label="Mean ± propagated u (k=1)",
        zorder=3,
    )
    ax.scatter(
        x,
        y_mean,
        s=38,
        color=data_color,
        edgecolors="black",
        linewidth=0.6,
        zorder=5,
    )

    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=label_fs,
        labelpad=6,
    )
    ax.set_ylabel(r"$pK_{a,\mathrm{app}}$", fontsize=label_fs, labelpad=8)

    ax.set_xlim(x_lo, x_hi)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    y_min_data = (
        np.nanmin(y_mean - yerr) if np.any(np.isfinite(y_mean - yerr)) else 4.75
    )
    y_max_data = (
        np.nanmax(y_mean + yerr) if np.any(np.isfinite(y_mean + yerr)) else 5.65
    )
    y_lo = min(4.75, float(y_min_data) - 0.03)
    y_hi = max(5.65, float(y_max_data) + 0.03)
    ax.set_ylim(y_lo, y_hi)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.tick_params(axis="x", which="major", pad=2)

    ax.yaxis.grid(True, which="major", linewidth=0.5, alpha=0.12)
    ax.xaxis.grid(False)

    ci95_m = fit.get("ci95_m", np.nan)
    ci95_b = fit.get("ci95_b", np.nan)
    p_m = fit.get("p_m", np.nan)

    def _fmt_sigfig(value: float, sigfigs: int = 4) -> str:
        if not np.isfinite(value):
            return "n/a"
        return f"{value:.{sigfigs}g}"

    def _fmt_pvalue(value: float) -> str:
        if not np.isfinite(value):
            return "n/a"
        if value < 1e-3:
            return "<0.001"
        return f"{value:.5f}".rstrip("0").rstrip(".")

    if np.isfinite(ci95_m):
        m_line = (
            f"m = {_fmt_sigfig(m_best)}; 95% CI [{_fmt_sigfig(m_best - ci95_m)}, "
            f"{_fmt_sigfig(m_best + ci95_m)}]"
        )
    else:
        m_line = f"m = {_fmt_sigfig(m_best)}"

    if np.isfinite(ci95_b):
        b_line = (
            f"b = {_fmt_sigfig(b_best)}; 95% CI [{_fmt_sigfig(b_best - ci95_b)}, "
            f"{_fmt_sigfig(b_best + ci95_b)}]"
        )
    else:
        b_line = f"b = {_fmt_sigfig(b_best)}"

    r2_display = f"{_fmt_sigfig(r2, 3)}" if np.isfinite(r2) else "n/a"
    r2_adj_display = f"{_fmt_sigfig(r2_adj, 3)}" if np.isfinite(r2_adj) else "n/a"
    r2p_line = f"R² = {r2_display}; adj. R² = {r2_adj_display}; p = {_fmt_pvalue(p_m)}"

    summary_heading = "Fit summary (OLS)"
    summary_body = "\n".join([m_line, b_line, r2p_line])

    eq_line = r"Regression equation: $y = m\,x + b$"
    defs_line = r"$y \equiv pK_{a,\mathrm{app}},\; x \equiv [\mathrm{NaCl}]$"

    # draw a single rounded box that contains the heading, equation and values
    box_x, box_y = 0.01, 0.5
    box_w, box_h = 1.5, 0.45
    box = FancyBboxPatch(
        (box_x, box_y),
        box_w,
        box_h,
        transform=gutter_ax.transAxes,
        boxstyle="round,pad=0.02",
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=1,
        clip_on=False,
    )
    gutter_ax.add_patch(box)

    gutter_ax.text(
        0.03,
        0.95,
        summary_heading,
        transform=gutter_ax.transAxes,
        ha="left",
        va="top",
        fontsize=label_fs,
        weight="bold",
        color="black",
        zorder=2,
    )
    gutter_ax.text(
        0.03,
        0.87,
        eq_line,
        transform=gutter_ax.transAxes,
        ha="left",
        va="top",
        fontsize=legend_fs,
        color="black",
        zorder=2,
    )
    gutter_ax.text(
        0.03,
        0.79,
        defs_line,
        transform=gutter_ax.transAxes,
        ha="left",
        va="top",
        fontsize=legend_fs - 0,
        color="black",
        zorder=2,
    )
    gutter_ax.text(
        0.03,
        0.69,
        summary_body,
        transform=gutter_ax.transAxes,
        ha="left",
        va="top",
        fontsize=legend_fs,
        color="black",
        zorder=2,
    )

    data_proxy = Line2D(
        [],
        [],
        color=data_color,
        marker="o",
        linestyle="-",
        linewidth=1.0,
        markersize=6,
        markerfacecolor=data_color,
        markeredgecolor="black",
    )
    line_proxy = Line2D([], [], color=line_color, linewidth=1.8)
    ci_proxy = Patch(facecolor=ci_color, edgecolor="black", linewidth=0.6, alpha=0.45)

    gutter_ax.legend(
        [data_proxy, line_proxy, ci_proxy],
        ["Mean ± propagated u (k=1)", "Linear fit", "95% CI (mean)"],
        loc="upper left",
        bbox_to_anchor=(-0.075, 0.35),
        ncol=1,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        labelspacing=0.8,
        fontsize=legend_fs,
    )

    fig.suptitle(
        "Apparent pKₐ of ethanoic acid vs [NaCl] (26 ± 1 °C)",
        x=0.55,
        y=0.9,
        fontsize=title_fs,
        fontweight="bold",
    )

    se_m = fit.get("se_m", np.nan)
    se_b = fit.get("se_b", np.nan)

    if np.isfinite(ci95_m):
        slope_txt = (
            f"m = {_fmt_sigfig(m_best)}; 95% CI [{_fmt_sigfig(m_best - ci95_m)}, "
            f"{_fmt_sigfig(m_best + ci95_m)}]"
        )
    elif np.isfinite(se_m):
        slope_txt = f"m = {_fmt_sigfig(m_best)} ± {se_m:.3g} (SE)"
    else:
        slope_txt = f"m = {_fmt_sigfig(m_best)}"

    if np.isfinite(ci95_b):
        intercept_txt = (
            f"b = {_fmt_sigfig(b_best)}; 95% CI [{_fmt_sigfig(b_best - ci95_b)}, "
            f"{_fmt_sigfig(b_best + ci95_b)}]"
        )
    elif np.isfinite(se_b):
        intercept_txt = f"b = {_fmt_sigfig(b_best)} ± {se_b:.3g} (SE)"
    else:
        intercept_txt = f"b = {_fmt_sigfig(b_best)}"

    p_txt = "n/a"
    if np.isfinite(p_m):
        p_txt = f"{p_m:.3g}" if p_m >= 1e-3 else "<0.001"

    r2_txt = f"R² = {_fmt_sigfig(r2, 3)}" if np.isfinite(r2) else "R² = n/a"
    r2_adj_txt = (
        f"adjusted R² = {_fmt_sigfig(r2_adj, 3)}"
        if np.isfinite(r2_adj)
        else "adjusted R² = n/a"
    )

    caption = (
        "Mean points show group mean apparent pKₐ with vertical error bars "
        "as propagated uncertainty u (k=1)"
        " per concentration condition; n = 3 runs per condition. "
        "NaCl concentration prepared to ±0.01 mol dm⁻³ (x-uncertainty not drawn). "
        "OLS linear regression to mean points: "
        f"{slope_txt}; {intercept_txt}; {r2_txt}; {r2_adj_txt}; "
        f"two-sided p-value for slope = {p_txt}."
    )

    caption_path = os.path.join(output_dir, "statistical_summary_caption.txt")
    with open(caption_path, "w", encoding="utf-8") as fh:
        fh.write(caption + "\n")

    out_path = os.path.join(output_dir, "statistical_summary.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path
