"""Render condition-level summary plots for ionic-strength trend evaluation.

This module visualizes grouped apparent pKa outcomes and fit bounds used in
the IA conclusion section.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
from scipy.stats import t as student_t

from ..schema import ResultColumns
from ..stats.regression import linear_regression
from .style import (
    ALPHAS,
    FONT_SIZES,
    MARKER_SIZES,
    MATH_LABELS,
    NACL_LEVELS,
    add_info_box,
    add_panel_label,
    color_for_nacl,
    fig_size,
    figure_base_path,
    figure_legend,
    finalize_figure,
    marker_for_run,
    new_figure,
    place_fig_legend,
    safe_annotate,
    save_figure_all_formats,
    set_axes_style,
    set_axis_labels,
    set_naCl_axis,
    set_sensible_ticks,
    set_ticks,
)
from .titration_plots import save_figure_bundle, setup_plot_style


def _initial_ph_by_nacl(results: List[Dict]) -> dict[float, list[float]]:
    """Collect initial pH values per NaCl concentration."""
    grouped: dict[float, list[float]] = {}
    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        if not np.isfinite(nacl):
            continue
        raw_df = res.get("data", pd.DataFrame())
        if not isinstance(raw_df, pd.DataFrame) or "pH" not in raw_df.columns:
            continue
        ph = pd.to_numeric(raw_df["pH"], errors="coerce").to_numpy(dtype=float)
        ph = ph[np.isfinite(ph)]
        if len(ph) == 0:
            continue
        grouped.setdefault(float(np.round(nacl, 1)), []).append(float(ph[0]))
    return grouped


def plot_initial_ph_by_nacl(
    results: List[Dict],
    output_dir: str | None = None,
    file_stem: str = "initial_ph_by_nacl",
) -> str:
    """Plot boxplot + jittered initial pH values by NaCl concentration."""
    setup_plot_style()
    grouped = _initial_ph_by_nacl(results)
    if not grouped:
        return ""

    concentrations = [c for c in (0.0, 0.2, 0.4, 0.6, 0.8) if c in grouped]
    values = [grouped[c] for c in concentrations]
    means = [float(np.mean(v)) for v in values]
    rng = np.random.default_rng(11)

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=False)
    ax.boxplot(
        values,
        positions=concentrations,
        widths=0.10,
        patch_artist=True,
        showmeans=False,
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        boxprops={"facecolor": "0.90", "edgecolor": "black", "linewidth": 1.0},
    )

    for index, (nacl, vals) in enumerate(zip(concentrations, values)):
        jitter = rng.normal(0.0, 0.008, size=len(vals))
        ax.scatter(
            nacl + jitter,
            vals,
            s=34,
            alpha=0.72,
            marker=marker_for_run(index),
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    ax.scatter(
        concentrations,
        means,
        s=62,
        marker="D",
        facecolor="black",
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
        label="Condition mean",
    )

    expected = 2.88
    ax.axhline(
        expected,
        linestyle="--",
        color="0.2",
        linewidth=1.2,
        label="Expected initial pH (reference)",
    )

    set_naCl_axis(ax)
    set_axes_style(
        ax,
        xlabel=r"$[\mathrm{NaCl}]\ /\ \mathrm{mol\,dm^{-3}}$",
        ylabel=r"Initial\ pH\ (\mathrm{pH}_0)",
        xticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        xfmt="%.1f",
    )
    ax.set_title("Initial pH vs [NaCl]", fontsize=FONT_SIZES["title"], pad=8)
    ax.grid(True, axis="y", alpha=0.14, linestyle=":", linewidth=0.8)

    all_vals = np.asarray([item for row in values for item in row], dtype=float)
    if len(all_vals):
        ax.set_ylim(float(np.min(all_vals) - 0.05), float(np.max(all_vals) + 0.05))

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="Individual runs",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="black",
            markeredgecolor="white",
            markersize=7,
            label="Condition mean",
        ),
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.2",
            linewidth=1.2,
            label="Expected initial pH (reference)",
        ),
    ]
    figure_legend(
        fig,
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )

    if output_dir:
        out_base = os.path.join(output_dir, file_stem)
    else:
        out_base = str(figure_base_path(fig_key=file_stem, kind="summary"))
    out_path = str(
        finalize_figure(
            fig,
            savepath=out_base,
            legend_height=0.10,
            tight=True,
            pad_inches=0.12,
        )
    )
    plt.close(fig)
    return out_path


def plot_initial_ph_scatter_with_errorbar(
    results: List[Dict],
    output_dir: str | None = None,
    file_stem: str = "initial_ph_scatter_with_errorbar",
) -> str:
    """Plot condition means ±1 SD of initial pH by NaCl concentration."""
    setup_plot_style()
    grouped = _initial_ph_by_nacl(results)
    if not grouped:
        return ""

    concentrations = [c for c in (0.0, 0.2, 0.4, 0.6, 0.8) if c in grouped]
    means = np.asarray([np.mean(grouped[c]) for c in concentrations], dtype=float)
    sds = np.asarray(
        [
            np.std(grouped[c], ddof=1) if len(grouped[c]) > 1 else 0.0
            for c in concentrations
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=False)
    ax.errorbar(
        concentrations,
        means,
        yerr=sds,
        fmt="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.0,
        ecolor="0.25",
        elinewidth=1.2,
        capsize=4,
        label="Mean ± 1 SD",
        zorder=3,
    )

    overall = float(np.mean(means))
    ax.axhline(
        overall,
        linestyle="--",
        color="0.35",
        linewidth=1.1,
        label="Overall mean (reference)",
    )
    ax.axhline(
        2.88,
        linestyle=":",
        color="0.2",
        linewidth=1.2,
        label="Expected initial pH (reference)",
    )

    set_naCl_axis(ax)
    set_axes_style(
        ax,
        xlabel=r"$[\mathrm{NaCl}]\ /\ \mathrm{mol\,dm^{-3}}$",
        ylabel=r"Initial\ pH\ (\mathrm{pH}_0)",
        xticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        xfmt="%.1f",
    )
    ax.set_title("Initial pH (mean ± SD)", fontsize=FONT_SIZES["title"], pad=8)
    ax.grid(True, axis="y", alpha=0.14, linestyle=":", linewidth=0.8)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Mean ± 1 SD",
        ),
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.35",
            linewidth=1.1,
            label="Overall mean (reference)",
        ),
        Line2D(
            [],
            [],
            linestyle=":",
            color="0.2",
            linewidth=1.2,
            label="Expected initial pH (reference)",
        ),
    ]
    figure_legend(
        fig,
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )

    if output_dir:
        out_base = os.path.join(output_dir, file_stem)
    else:
        out_base = str(figure_base_path(fig_key=file_stem, kind="summary"))
    out_path = str(
        finalize_figure(
            fig,
            savepath=out_base,
            legend_height=0.10,
            tight=True,
            pad_inches=0.12,
        )
    )
    plt.close(fig)
    return out_path


def plot_temperature_control_by_nacl(
    results: List[Dict],
    output_dir: str | None = None,
    file_stem: str = "temperature_control_by_nacl",
) -> str:
    """Plot temperature control as boxplots plus random subsample scatter."""
    setup_plot_style()
    grouped: dict[float, list[float]] = {}
    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        if not np.isfinite(nacl):
            continue
        raw_df = res.get("data", pd.DataFrame())
        if (
            not isinstance(raw_df, pd.DataFrame)
            or "Temperature (°C)" not in raw_df.columns
        ):
            continue
        vals = pd.to_numeric(raw_df["Temperature (°C)"], errors="coerce").to_numpy(
            dtype=float
        )
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        grouped.setdefault(float(np.round(nacl, 1)), []).extend(vals.tolist())

    if not grouped:
        return ""

    concentrations = [c for c in (0.0, 0.2, 0.4, 0.6, 0.8) if c in grouped]
    values = [grouped[c] for c in concentrations]
    rng = np.random.default_rng(29)

    fig, ax = plt.subplots(figsize=fig_size("wide"), constrained_layout=False)
    ax.axhspan(25.0, 27.0, color="0.93", zorder=0)
    ax.axhline(26.0, color="0.15", linestyle="--", linewidth=1.2)

    ax.boxplot(
        values,
        positions=concentrations,
        widths=0.10,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "black",
            "markersize": 6,
        },
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
        boxprops={"facecolor": "0.87", "edgecolor": "black", "linewidth": 1.0},
    )

    for nacl, vals in zip(concentrations, values):
        arr = np.asarray(vals, dtype=float)
        if len(arr) > 200:
            idx = rng.choice(len(arr), size=200, replace=False)
            arr = arr[idx]
        jitter = rng.normal(0.0, 0.008, size=len(arr))
        ax.scatter(
            nacl + jitter,
            arr,
            s=14,
            alpha=0.22,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=0.5,
            zorder=2,
        )

    set_naCl_axis(ax)
    set_axes_style(
        ax,
        xlabel=r"$[\mathrm{NaCl}]\ /\ \mathrm{mol\,dm^{-3}}$",
        ylabel=r"Temperature\ /\ ^\circ\mathrm{C}",
        xticks=(0.0, 0.2, 0.4, 0.6, 0.8),
        xfmt="%.1f",
    )
    ax.set_title("Temperature control by [NaCl]", fontsize=FONT_SIZES["title"], pad=8)
    safe_annotate(
        ax,
        "Random subsample for visibility",
        xy=(0.98, 0.04),
        xytext=(0, 0),
        textcoords="axes fraction",
        ha="right",
        va="bottom",
    )
    ax.grid(True, axis="y", alpha=0.14, linestyle=":", linewidth=0.8)

    handles = [
        Patch(facecolor="0.93", edgecolor="none", label="Tolerance band (±1.0 °C)"),
        Line2D(
            [], [], linestyle="--", color="0.15", linewidth=1.2, label="Target 26.0 °C"
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=5,
            label="Random subsample for visibility",
        ),
    ]

    figure_legend(
        fig,
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
    )

    if output_dir:
        out_base = os.path.join(output_dir, file_stem)
    else:
        out_base = str(figure_base_path(fig_key=file_stem, kind="summary"))
    out_path = str(
        finalize_figure(
            fig, savepath=out_base, legend_height=0.10, tight=True, pad_inches=0.12
        )
    )
    plt.close(fig)
    return out_path


def plot_statistical_summary(summary: Dict, output_dir: str | None = None) -> str:
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
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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

    gutter_ax.text(
        0.03,
        0.97,
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
        0.88,
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
        0.80,
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
        0.70,
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

    place_fig_legend(
        fig,
        [data_proxy, line_proxy, ci_proxy],
        ["Mean ± propagated u (k=1)", "Linear fit", "95% CI (mean)"],
        where="bottom",
        ncol=3,
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

    if output_dir:
        caption_path = os.path.join(output_dir, "statistical_summary_caption.txt")
        with open(caption_path, "w", encoding="utf-8") as fh:
            fh.write(caption + "\n")

        out_path = os.path.join(output_dir, "statistical_summary.png")
        save_figure_bundle(fig, out_path)
    else:
        out_path = str(
            save_figure_all_formats(
                fig,
                figure_base_path("statistical_summary", kind="supplemental"),
            )
        )
    plt.close(fig)
    return out_path


def _mean_ci95(values: np.ndarray) -> tuple[float, float, float]:
    """Return mean, standard error, and 95% CI half-width."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(arr))
    if n == 1:
        return mean, np.nan, 0.0
    sd = float(np.std(arr, ddof=1))
    se = sd / np.sqrt(n)
    t_crit = float(student_t.ppf(0.975, n - 1))
    return mean, float(se), float(t_crit * se)


def _fit_line_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    se: np.ndarray | None = None,
) -> Dict[str, np.ndarray | float]:
    """Fit linear trend and return slope/intercept plus 95% CI band."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 2:
        return {
            "slope": np.nan,
            "intercept": np.nan,
            "slope_ci_low": np.nan,
            "slope_ci_high": np.nan,
            "x_grid": np.array([], dtype=float),
            "y_hat": np.array([], dtype=float),
            "ci_low": np.array([], dtype=float),
            "ci_high": np.array([], dtype=float),
            "fit_basis": "condition means (insufficient points)",
            "weighted": False,
        }

    weighted = False
    if se is not None:
        se_arr = np.asarray(se, dtype=float)[finite]
        good = np.isfinite(se_arr) & (se_arr > 0)
        if np.sum(good) == len(se_arr):
            w = 1.0 / (se_arr**2)
            weighted = True
        else:
            w = np.ones_like(x, dtype=float)
    else:
        w = np.ones_like(x, dtype=float)

    X = np.column_stack([np.ones_like(x), x])
    W = np.diag(w)
    xtwx = X.T @ W @ X
    xtwx_inv = np.linalg.pinv(xtwx)
    beta = xtwx_inv @ (X.T @ W @ y)
    intercept, slope = float(beta[0]), float(beta[1])
    y_hat_obs = X @ beta
    resid = y - y_hat_obs
    dof = max(len(x) - 2, 1)
    rss = float(np.sum(w * (resid**2)))
    sigma2 = rss / dof if dof > 0 else np.nan
    cov_beta = sigma2 * xtwx_inv if np.isfinite(sigma2) else np.full((2, 2), np.nan)

    x_grid = np.linspace(float(np.min(x)), float(np.max(x)), 240)
    Xg = np.column_stack([np.ones_like(x_grid), x_grid])
    y_hat = Xg @ beta
    pred_var = np.sum((Xg @ cov_beta) * Xg, axis=1)
    pred_var = np.maximum(pred_var, 0.0)
    t_crit = float(student_t.ppf(0.975, dof)) if dof > 0 else np.nan
    ci_half = (
        t_crit * np.sqrt(pred_var) if np.isfinite(t_crit) else np.zeros_like(x_grid)
    )

    slope_se = np.sqrt(float(cov_beta[1, 1])) if np.isfinite(cov_beta[1, 1]) else np.nan
    slope_ci_half = (
        t_crit * slope_se if np.isfinite(t_crit) and np.isfinite(slope_se) else np.nan
    )

    fit_basis = (
        "condition means (weighted by inverse SE^2)"
        if weighted
        else "condition means (unweighted, equal variance assumption)"
    )
    return {
        "slope": slope,
        "intercept": intercept,
        "slope_ci_low": slope - slope_ci_half if np.isfinite(slope_ci_half) else np.nan,
        "slope_ci_high": (
            slope + slope_ci_half if np.isfinite(slope_ci_half) else np.nan
        ),
        "x_grid": x_grid,
        "y_hat": y_hat,
        "ci_low": y_hat - ci_half,
        "ci_high": y_hat + ci_half,
        "fit_basis": fit_basis,
        "weighted": weighted,
    }


def _build_condition_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """Build condition means and CI table from replicate-level results."""
    cols = ResultColumns()
    table_rows: List[Dict[str, float]] = []
    for nacl, group in results_df.groupby(cols.nacl):
        vals = pd.to_numeric(group[cols.pka_app], errors="coerce").to_numpy(dtype=float)
        mean, se, ci95 = _mean_ci95(vals)
        table_rows.append(
            {
                cols.nacl: float(nacl),
                "mean_pka": mean,
                "se_pka": se,
                "ci95": ci95,
                "n": int(np.sum(np.isfinite(vals))),
            }
        )
    out = pd.DataFrame(table_rows)
    if out.empty:
        return out
    return out.sort_values(cols.nacl).reset_index(drop=True)


def _load_results_for_pka_plot(
    results_df: pd.DataFrame | None,
    summary_csv_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load replicate-level and condition-level tables for Figure 3."""
    cols = ResultColumns()
    replicate_df = pd.DataFrame()
    summary_df = pd.DataFrame()

    if results_df is not None and not results_df.empty:
        replicate_df = results_df.copy()

    if replicate_df.empty:
        companion = os.path.join(
            os.path.dirname(summary_csv_path), "individual_results_rawfiles.csv"
        )
        if os.path.exists(companion):
            replicate_df = pd.read_csv(companion)

    if os.path.exists(summary_csv_path):
        summary_df = pd.read_csv(summary_csv_path)

    if (
        not replicate_df.empty
        and cols.nacl in replicate_df.columns
        and cols.pka_app in replicate_df.columns
    ):
        condition_df = _build_condition_table(replicate_df)
    elif not summary_df.empty:
        needed = {"NaCl Concentration (M)", "Mean pKa_app"}
        if needed.issubset(summary_df.columns):
            condition_df = summary_df.rename(
                columns={
                    "NaCl Concentration (M)": cols.nacl,
                    "Mean pKa_app": "mean_pka",
                    "SEM pKa_app": "se_pka",
                }
            ).copy()
            if "ci95" not in condition_df.columns:
                if "n" in condition_df.columns and "se_pka" in condition_df.columns:
                    ci_vals = []
                    for _, row in condition_df.iterrows():
                        n = int(row.get("n", 0))
                        se = float(row.get("se_pka", np.nan))
                        if n > 1 and np.isfinite(se):
                            t_crit = float(student_t.ppf(0.975, n - 1))
                            ci_vals.append(float(t_crit * se))
                        else:
                            ci_vals.append(np.nan)
                    condition_df["ci95"] = ci_vals
                else:
                    condition_df["ci95"] = np.nan
        else:
            condition_df = pd.DataFrame()
    else:
        condition_df = pd.DataFrame()

    if not replicate_df.empty and cols.nacl in replicate_df.columns:
        replicate_df[cols.nacl] = pd.to_numeric(
            replicate_df[cols.nacl], errors="coerce"
        )
    if not condition_df.empty and cols.nacl in condition_df.columns:
        condition_df[cols.nacl] = pd.to_numeric(
            condition_df[cols.nacl], errors="coerce"
        )

    return replicate_df, condition_df


def _temperature_outlier_map(results: List[Dict] | None) -> dict[tuple[str, str], bool]:
    """Map (run, source_file) to temperature-outlier status."""
    outlier_map: dict[tuple[str, str], bool] = {}
    if not results:
        return outlier_map
    for res in results:
        run_key = str(res.get("run_name", ""))
        src_key = str(res.get("source_file", ""))
        raw_df = res.get("data", pd.DataFrame())
        temp_outlier = False
        if isinstance(raw_df, pd.DataFrame) and "Temperature (°C)" in raw_df.columns:
            temp = pd.to_numeric(raw_df["Temperature (°C)"], errors="coerce").to_numpy(
                dtype=float
            )
            temp = temp[np.isfinite(temp)]
            if len(temp) > 0:
                t_mean = float(np.mean(temp))
                temp_outlier = bool(t_mean < 25.0 or t_mean > 27.0)
        outlier_map[(run_key, src_key)] = temp_outlier
    return outlier_map


def plot_pka_app_vs_nacl_and_I(
    results_df: pd.DataFrame | None = None,
    results: List[Dict] | None = None,
    summary_csv_path: str = "output/ia/processed_summary_with_sd.csv",
    output_dir: str | None = None,
    file_stem: str = "pka_app_vs_nacl_and_I",
    return_figure: bool = False,
    return_metadata: bool = False,
):
    """Figure 3: apparent pKa trend against NaCl concentration."""
    setup_plot_style()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cols = ResultColumns()
    rep_df, cond_df = _load_results_for_pka_plot(results_df, summary_csv_path)
    if cond_df.empty:
        raise ValueError("No condition-level pK_a,app data available for Figure 3.")

    outlier_map = _temperature_outlier_map(results)

    fig, left_ax = new_figure(1, 1, figsize=fig_size("single"))
    rng = np.random.default_rng(42)

    left_ax.grid(True, axis="y")
    left_ax.set_ylabel(MATH_LABELS["pka_app"])

    if not rep_df.empty and {cols.nacl, cols.pka_app}.issubset(rep_df.columns):
        for _, row in rep_df.iterrows():
            x0 = float(row[cols.nacl])
            y0 = float(row[cols.pka_app])
            if not np.isfinite(x0) or not np.isfinite(y0):
                continue
            jitter = float(rng.normal(0.0, 0.012))
            run_key = str(row.get("Run", ""))
            src_key = str(row.get("Source File", ""))
            is_outlier = outlier_map.get((run_key, src_key), False)
            face = "0.75" if is_outlier else color_for_nacl(x0)
            edge = "black" if is_outlier else "white"
            left_ax.scatter(
                x0 + jitter,
                y0,
                s=MARKER_SIZES["replicate"],
                facecolor=face,
                edgecolor=edge,
                linewidth=0.8,
                alpha=ALPHAS["replicate"],
                zorder=3,
            )

    x_mean = pd.to_numeric(cond_df[cols.nacl], errors="coerce").to_numpy(dtype=float)
    y_mean = pd.to_numeric(cond_df["mean_pka"], errors="coerce").to_numpy(dtype=float)
    y_ci = pd.to_numeric(cond_df.get("ci95", np.nan), errors="coerce").to_numpy(
        dtype=float
    )

    excluded_outliers = 0
    fit_cond_df = cond_df.copy()
    if (
        outlier_map
        and not rep_df.empty
        and "Run" in rep_df.columns
        and "Source File" in rep_df.columns
        and cols.nacl in rep_df.columns
        and cols.pka_app in rep_df.columns
    ):
        keep_mask = []
        for _, row in rep_df.iterrows():
            flagged = outlier_map.get(
                (str(row.get("Run", "")), str(row.get("Source File", ""))),
                False,
            )
            excluded_outliers += int(flagged)
            keep_mask.append(not flagged)
        fit_rep_df = rep_df[np.array(keep_mask, dtype=bool)].copy()
        unique_nacl = fit_rep_df[cols.nacl].nunique() if not fit_rep_df.empty else 0
        if unique_nacl >= 2:
            fit_cond_df = _build_condition_table(fit_rep_df)

    left_ax.errorbar(
        x_mean,
        y_mean,
        yerr=np.where(np.isfinite(y_ci), y_ci, 0.0),
        fmt="o",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        markersize=8,
        capsize=3,
        linewidth=1.1,
        zorder=4,
    )

    fit_x = pd.to_numeric(fit_cond_df[cols.nacl], errors="coerce").to_numpy(dtype=float)
    fit_y = pd.to_numeric(fit_cond_df["mean_pka"], errors="coerce").to_numpy(
        dtype=float
    )
    fit_se = pd.to_numeric(fit_cond_df.get("se_pka", np.nan), errors="coerce").to_numpy(
        dtype=float
    )
    fit_mask = np.isfinite(fit_x) & np.isfinite(fit_y)
    fit_info = _fit_line_with_ci(fit_x[fit_mask], fit_y[fit_mask], fit_se[fit_mask])
    if len(fit_info["x_grid"]) > 0:
        left_ax.fill_between(
            fit_info["x_grid"],
            fit_info["ci_low"],
            fit_info["ci_high"],
            color="#B6D6F2",
            alpha=ALPHAS["ci_band"],
            zorder=1,
        )
        left_ax.plot(
            fit_info["x_grid"],
            fit_info["y_hat"],
            color="#0B4F8C",
            linewidth=1.5,
            zorder=2,
        )

    slope = float(fit_info.get("slope", np.nan))
    slope_low = float(fit_info.get("slope_ci_low", np.nan))
    slope_high = float(fit_info.get("slope_ci_high", np.nan))
    slope_units = r"\mathrm{pH\cdot(mol\ dm^{-3})^{-1}}"
    slope_text = (
        rf"$\mathrm{{Linear\ fit\ slope}}={slope:.4f}\ {slope_units}$"
        + "\n"
        + rf"$95\%\ \mathrm{{CI}}\ [{slope_low:.4f},\ {slope_high:.4f}]$"
        if np.isfinite(slope) and np.isfinite(slope_low) and np.isfinite(slope_high)
        else r"$\mathrm{Linear\ fit\ slope}=\mathrm{n/a}$"
    )
    add_info_box(left_ax, slope_text, loc="upper right", fontsize=11)
    set_axis_labels(left_ax, MATH_LABELS["x_nacl"], MATH_LABELS["pka_app"])

    y_mask = np.isfinite(y_mean)
    yerr_plot = np.where(np.isfinite(y_ci), y_ci, 0.0)
    yerr_mask = np.isfinite(yerr_plot)
    valid_bounds = y_mask & yerr_mask
    if np.any(valid_bounds):
        y_lower = y_mean[valid_bounds] - yerr_plot[valid_bounds]
        y_upper = y_mean[valid_bounds] + yerr_plot[valid_bounds]
        span = max(float(np.max(y_upper) - np.min(y_lower)), 1e-6)
        pad = max(0.03, 0.10 * span)
        left_ax.set_ylim(float(np.min(y_lower) - pad), float(np.max(y_upper) + pad))

    set_ticks(left_ax, xstep=0.2, ystep=0.1, yfmt="%.1f")
    set_sensible_ticks(left_ax, x=5, y=5)
    add_panel_label(left_ax, "(a)", loc="upper left", pad=0.03)

    if output_dir:
        out_base = os.path.join(output_dir, file_stem)
    else:
        out_base = str(figure_base_path(fig_key=file_stem, kind="main_results"))
    out_path = str(
        finalize_figure(
            fig,
            savepath=out_base,
            tight=True,
            pad_inches=0.14,
        )
    )

    metadata = {
        "fit_basis": str(fit_info.get("fit_basis", "condition means")),
        "slope": slope,
        "slope_ci_low": slope_low,
        "slope_ci_high": slope_high,
        "n_reps": int(np.nanmedian(cond_df["n"])) if "n" in cond_df.columns else 0,
        "n_reps_range": (
            f"{int(np.nanmin(cond_df['n']))}-{int(np.nanmax(cond_df['n']))}"
            if "n" in cond_df.columns and len(cond_df)
            else "n/a"
        ),
        "excluded_outliers": int(excluded_outliers),
    }

    if return_figure and return_metadata:
        return out_path, fig, metadata
    if return_figure:
        return out_path, fig
    if return_metadata:
        plt.close(fig)
        return out_path, metadata
    plt.close(fig)
    return out_path


def _hh_points_for_run(res: Dict) -> tuple[np.ndarray, np.ndarray]:
    """Return HH x/y arrays for one run using 0.1 < [A-]/[HA] < 10."""
    step_df = res.get("step_data", pd.DataFrame())
    if step_df.empty or "Volume (cm^3)" not in step_df.columns:
        return np.array([], dtype=float), np.array([], dtype=float)
    p_col = "pH_step" if "pH_step" in step_df.columns else "pH"
    if p_col not in step_df.columns:
        return np.array([], dtype=float), np.array([], dtype=float)

    veq = float(res.get("veq_used", np.nan))
    if not np.isfinite(veq) or veq <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    v = pd.to_numeric(step_df["Volume (cm^3)"], errors="coerce").to_numpy(dtype=float)
    ph = pd.to_numeric(step_df[p_col], errors="coerce").to_numpy(dtype=float)
    ratio = v / (veq - v)
    valid = (
        np.isfinite(v)
        & np.isfinite(ph)
        & np.isfinite(ratio)
        & (ratio > 0.1)
        & (ratio < 10.0)
    )
    x = np.log10(ratio[valid])
    y = ph[valid]
    finite = np.isfinite(x) & np.isfinite(y)
    return x[finite], y[finite]


def plot_hh_linearization_and_diagnostics(
    results: List[Dict],
    output_dir: str | None = None,
    file_stem: str = "hh_linearization_and_diagnostics",
    return_figure: bool = False,
):
    """Figure 4: HH linearization, residual diagnostics, slope CI, and RMSE.

    Curves are grouped by NaCl concentration.
    """
    setup_plot_style()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = new_figure(2, 2, figsize=fig_size("panel_2x2"))
    ax_a, ax_b, ax_c, ax_d = axes.flatten()

    slope_rows: List[tuple[float, float]] = []
    rmse_rows: List[tuple[float, float]] = []
    hh_x_values: List[float] = []
    hh_by_nacl: dict[float, dict[str, list[float]]] = {}

    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        slope = float(res.get("slope_reg", np.nan))
        float(res.get("r2_reg", np.nan))
        if np.isfinite(nacl) and np.isfinite(slope):
            slope_rows.append((nacl, slope))
        x_all, y_all = _hh_points_for_run(res)
        if np.isfinite(nacl) and len(x_all) >= 3:
            reg_all = linear_regression(x_all, y_all, min_points=3)
            if np.isfinite(reg_all.get("m", np.nan)) and np.isfinite(
                reg_all.get("b", np.nan)
            ):
                residuals_all = y_all - (reg_all["m"] * x_all + reg_all["b"])
                rmse = (
                    float(np.sqrt(np.mean(residuals_all**2)))
                    if len(residuals_all)
                    else np.nan
                )
                if np.isfinite(rmse):
                    rmse_rows.append((nacl, rmse))

    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        if not np.isfinite(nacl):
            continue
        step_df = res.get("step_data", pd.DataFrame())
        if step_df.empty or "Volume (cm^3)" not in step_df.columns:
            continue
        p_col = "pH_step" if "pH_step" in step_df.columns else "pH"
        if p_col not in step_df.columns:
            continue
        veq = float(res.get("veq_used", np.nan))
        if not np.isfinite(veq) or veq <= 0:
            continue

        v = pd.to_numeric(step_df["Volume (cm^3)"], errors="coerce").to_numpy(
            dtype=float
        )
        ph = pd.to_numeric(step_df[p_col], errors="coerce").to_numpy(dtype=float)
        ratio = v / (veq - v)
        valid = np.isfinite(v) & np.isfinite(ph) & np.isfinite(ratio) & (ratio > 0)
        if not np.any(valid):
            continue

        x_all = np.log10(ratio[valid])
        y_all = ph[valid]
        included = (ratio[valid] > 0.1) & (ratio[valid] < 10.0)
        x = x_all[included]
        y = y_all[included]
        x_ex = x_all[~included]
        y_ex = y_all[~included]
        if len(x) < 3:
            continue
        hh_x_values.extend(x.tolist())
        reg = linear_regression(x, y, min_points=3)
        color = color_for_nacl(nacl)
        order = np.argsort(x)
        y_fit = reg["m"] * x + reg["b"]
        residuals = y - y_fit

        bucket = hh_by_nacl.setdefault(
            float(np.round(nacl, 1)),
            {"x": [], "y": [], "x_ex": [], "y_ex": [], "resid": []},
        )
        bucket["x"].extend(x.tolist())
        bucket["y"].extend(y.tolist())
        bucket["x_ex"].extend(x_ex.tolist())
        bucket["y_ex"].extend(y_ex.tolist())
        bucket["resid"].extend(residuals.tolist())

        ax_a.plot(x[order], y_fit[order], color=color, linewidth=1.4, alpha=0.9)

    legend_handles = []
    for nacl in sorted(hh_by_nacl):
        color = color_for_nacl(nacl)
        x_arr = np.asarray(hh_by_nacl[nacl]["x"], dtype=float)
        y_arr = np.asarray(hh_by_nacl[nacl]["y"], dtype=float)
        x_ex_arr = np.asarray(hh_by_nacl[nacl]["x_ex"], dtype=float)
        y_ex_arr = np.asarray(hh_by_nacl[nacl]["y_ex"], dtype=float)
        resid_arr = np.asarray(hh_by_nacl[nacl]["resid"], dtype=float)

        if len(x_ex_arr):
            ax_a.scatter(
                x_ex_arr,
                y_ex_arr,
                s=max(12, int(0.65 * MARKER_SIZES["replicate"])),
                marker="o",
                facecolor="none",
                edgecolor="0.75",
                linewidth=0.8,
                alpha=0.3,
            )

        if len(x_arr):
            ax_a.scatter(
                x_arr,
                y_arr,
                s=MARKER_SIZES["replicate"],
                marker="o",
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.45,
            )
            ax_b.scatter(
                x_arr,
                resid_arr,
                s=MARKER_SIZES["replicate"],
                marker="o",
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.45,
            )
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    marker="o",
                    linestyle="none",
                    markerfacecolor=color,
                    markeredgecolor="white",
                    markersize=6,
                    label=rf"$[\mathrm{{NaCl}}]={nacl:.1f}\ \mathrm{{mol\,dm^{{-3}}}}$",
                )
            )

    set_axis_labels(ax_a, MATH_LABELS["hh_x"], MATH_LABELS["ph"])
    set_ticks(ax_a, xstep=0.5, ystep=0.2)
    ax_a.set_title("HH linearization with fit lines")
    ax_a.grid(True)

    ax_b.axhline(0.0, color="0.25", linestyle="--", linewidth=1.1)
    set_axis_labels(ax_b, MATH_LABELS["hh_x"], MATH_LABELS["residual_ph"])
    set_ticks(ax_b, xstep=0.5, ystep=0.05)
    ax_b.set_title("Residuals vs log term")
    ax_b.grid(True)

    ax_a.xaxis.set_major_locator(MultipleLocator(0.5))
    ax_b.xaxis.set_major_locator(MultipleLocator(0.5))

    if hh_x_values:
        hh_min = float(np.min(hh_x_values))
        hh_max = float(np.max(hh_x_values))
        hh_pad = max(0.05, 0.05 * (hh_max - hh_min))
        ax_a.set_xlim(hh_min - hh_pad, hh_max + hh_pad)
        ax_b.set_xlim(hh_min - hh_pad, hh_max + hh_pad)

    slope_df = pd.DataFrame(slope_rows, columns=["nacl", "slope"])
    if not slope_df.empty:
        means_x = []
        means_y = []
        ci_y = []
        for nacl, grp in slope_df.groupby("nacl"):
            vals = grp["slope"].to_numpy(dtype=float)
            m, _, ci = _mean_ci95(vals)
            means_x.append(float(nacl))
            means_y.append(m)
            ci_y.append(ci)
            jitter = np.random.default_rng(int(round(nacl * 1000))).normal(
                0, 0.008, size=len(vals)
            )
            ax_c.scatter(
                float(nacl) + jitter,
                vals,
                s=MARKER_SIZES["replicate"],
                color=color_for_nacl(float(nacl)),
                alpha=0.6,
                edgecolor="white",
                linewidth=0.5,
            )
        ax_c.errorbar(
            means_x,
            means_y,
            yerr=np.array(ci_y, dtype=float),
            fmt="o",
            color="black",
            markerfacecolor="white",
            markeredgecolor="black",
            elinewidth=0.9,
            capsize=3,
            zorder=2,
        )
        for nacl_val, grp in slope_df.groupby("nacl"):
            vals = grp["slope"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                y_top = float(np.max(vals))
                ax_c.text(
                    float(nacl_val),
                    y_top + 0.02,
                    rf"$n={len(vals)}$",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="0.35",
                )
    ax_c.axhline(
        1.0, color="0.2", linestyle="--", linewidth=1.2, label="ideal HH slope"
    )
    set_axis_labels(ax_c, MATH_LABELS["x_nacl"], "HH slope m")
    set_ticks(ax_c, xstep=0.2, ystep=0.05)
    ax_c.set_title("Fitted slope m vs [NaCl]")
    add_info_box(
        ax_c,
        r"$\mathrm{Error\ bars}:\ 95\%\ \mathrm{CI}$",
        loc="upper left",
        fontsize=11,
    )
    ax_c.grid(True, axis="y")
    set_sensible_ticks(ax_c, x=5, y=5)

    rmse_df = pd.DataFrame(rmse_rows, columns=["nacl", "rmse"])
    if not rmse_df.empty:
        for nacl, grp in rmse_df.groupby("nacl"):
            vals = grp["rmse"].to_numpy(dtype=float)
            jitter = np.random.default_rng(int(round(nacl * 1000 + 7))).normal(
                0, 0.008, size=len(vals)
            )
            ax_d.scatter(
                float(nacl) + jitter,
                vals,
                s=MARKER_SIZES["replicate"],
                color=color_for_nacl(float(nacl)),
                alpha=ALPHAS["replicate"],
                edgecolor="white",
                linewidth=0.5,
            )
            vals_finite = vals[np.isfinite(vals)]
            if len(vals_finite):
                mean_rmse = float(np.mean(vals_finite))
                sd_rmse = (
                    float(np.std(vals_finite, ddof=1)) if len(vals_finite) > 1 else 0.0
                )
                ax_d.errorbar(
                    [float(nacl)],
                    [mean_rmse],
                    yerr=[sd_rmse],
                    fmt="o",
                    color="black",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=7,
                    capsize=3,
                    zorder=4,
                )
    set_axis_labels(ax_d, MATH_LABELS["x_nacl"], r"HH fit RMSE / $\mathrm{pH}$")
    set_ticks(ax_d, xstep=0.2, ystep=0.01)
    ax_d.set_title("Fit RMSE vs [NaCl]")
    ax_d.grid(True, axis="y")
    set_sensible_ticks(ax_d, x=5, y=5)

    for ax, label in zip((ax_a, ax_b, ax_c, ax_d), ("(a)", "(b)", "(c)", "(d)")):
        add_panel_label(ax, label, y=0.95)

    if legend_handles:
        concentration_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=color_for_nacl(float(nacl)),
                markeredgecolor="white",
                markersize=6,
                label=(
                    rf"$[\mathrm{{NaCl}}]={float(nacl):.1f}" r"\ \mathrm{mol\,dm^{-3}}$"
                ),
            )
            for nacl in NACL_LEVELS
        ]
        legend_handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="none",
                markeredgecolor="0.75",
                markersize=6,
                alpha=0.6,
                label="excluded",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor="0.3",
                markeredgecolor="white",
                markersize=6,
                alpha=0.45,
                label="included buffer region",
            ),
        ] + concentration_handles
        figure_legend(
            fig,
            legend_handles,
            [h.get_label() for h in legend_handles],
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.95),
            frameon=False,
        )

    if output_dir:
        out_base = os.path.join(output_dir, file_stem)
    else:
        out_base = str(figure_base_path(fig_key=file_stem, kind="diagnostics"))
    out_path = str(
        finalize_figure(
            fig,
            savepath=out_base,
            legend_height=0.14 if legend_handles else 0.0,
            tight=True,
            pad_inches=0.14,
        )
    )
    if return_figure:
        return out_path, fig
    plt.close(fig)
    return out_path
