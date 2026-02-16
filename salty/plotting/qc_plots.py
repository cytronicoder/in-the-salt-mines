"""Generate QC figures that justify run inclusion and interpretation strength.

These figures correspond to IA quality arguments:
- endpoint plausibility,
- model validity,
- controlled-variable stability, and
- repeatability across ionic-strength conditions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from ..schema import ResultColumns
from .style import (
    ALPHAS,
    FONT_SIZES,
    LABEL_HH_SLOPE,
    LABEL_LOG_RATIO,
    LABEL_NACL,
    LABEL_R2,
    LABEL_RESIDUAL,
    MATH_LABELS,
    NACL_LEVELS,
    apply_subplot_padding,
    clean_axis,
    color_for_nacl,
    fig_size,
    figure_base_path,
    figure_legend,
    finalize_figure,
    safe_set_lims,
    set_axis_labels,
    set_ticks,
    should_plot_distribution,
)
from .summary_plots import plot_initial_ph_by_nacl as _plot_initial_ph_by_nacl
from .summary_plots import (
    plot_initial_ph_scatter_with_errorbar as _plot_initial_ph_scatter_with_errorbar,
)
from .summary_plots import (
    plot_temperature_control_by_nacl as _plot_temperature_control_by_nacl,
)
from .titration_plots import setup_plot_style


def _save_qc_figure(
    fig: plt.Figure,
    fig_key: str,
    *,
    output_dir: str | None,
    kind: str = "qc",
    legend_height: float = 0.0,
) -> str:
    """Save QC figures either to legacy output_dir or taxonomy path."""
    if output_dir:
        base = Path(output_dir) / fig_key
    else:
        base = figure_base_path(fig_key=fig_key, kind=kind)
    saved = finalize_figure(
        fig,
        savepath=base,
        legend_height=legend_height,
        tight=True,
        pad_inches=0.14,
    )
    return str(saved) if saved is not None else ""


def _rng(seed: int = 0) -> np.random.Generator:
    """Create a deterministic random generator for plotting noise.

    Args:
        seed (int): Integer seed for deterministic jitter and subsampling.

    Returns:
        numpy.random.Generator: Seeded random generator instance.
    """
    return np.random.default_rng(seed)


def _marker_for_nacl(nacl: float) -> str:
    """Select a marker style for a NaCl concentration level.

    Args:
        nacl (float): NaCl concentration in mol dm^-3.

    Returns:
        str: Matplotlib marker code chosen for grayscale-safe discrimination.
    """
    mapping = {0.0: "o", 0.2: "s", 0.4: "^", 0.6: "D", 0.8: "X"}
    return mapping.get(float(np.round(nacl, 1)), "o")


def _set_nacl_axis(ax: plt.Axes) -> None:
    """Apply standard NaCl axis ticks and limits for QC plots.

    Args:
        ax (matplotlib.axes.Axes): Target axes object.

    Returns:
        None: Modify ``ax`` in place.
    """
    ax.set_xlim(-0.1, 0.9)
    ax.set_xticks(NACL_LEVELS)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def _qc_grid(ax: plt.Axes, y_only: bool = True) -> None:
    """Apply a light grid style for QC readability.

    Args:
        ax (matplotlib.axes.Axes): Target axes object.
        y_only (bool): If ``True``, draw gridlines on y-axis only.

    Returns:
        None: Modify ``ax`` in place.
    """
    ax.grid(False)
    if y_only:
        ax.yaxis.grid(True, alpha=0.14, linestyle=":", linewidth=0.8)
    else:
        ax.grid(True, alpha=0.14, linestyle=":", linewidth=0.8)


def _apply_qc_typography(ax: plt.Axes, title: str) -> None:
    """Apply QC title and tick typography defaults.

    Args:
        ax (matplotlib.axes.Axes): Target axes object.
        title (str): Plot title text.

    Returns:
        None: Modify ``ax`` in place.
    """
    ax.set_title(title, fontweight="bold", fontsize=FONT_SIZES["title"], pad=10)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick"])


def _jitter(x: float, n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Generate jittered x-positions around one central value.

    Args:
        x (float): Central x-position (for example NaCl concentration).
        n (int): Number of jittered points.
        scale (float): Standard deviation of jitter in x-axis units.
        rng (numpy.random.Generator): Random generator for deterministic output.

    Returns:
        numpy.ndarray: Jittered x-positions with shape ``(n,)``.
    """
    return x + rng.normal(0.0, scale, size=n)


def _categorical_positions(
    concentrations: Sequence[float],
) -> Tuple[np.ndarray, List[str]]:
    """Map concentration levels to categorical axis positions and labels.

    Args:
        concentrations (Sequence[float]): Concentration levels in mol dm^-3.

    Returns:
        tuple[numpy.ndarray, list[str]]: Numeric positions and formatted tick
        labels.
    """
    concs = [float(c) for c in concentrations]
    pos = np.arange(len(concs), dtype=float)
    labels = [f"{c:.1f}" for c in concs]
    return pos, labels


def _reserve_bottom(fig: plt.Figure, bottom: float) -> None:
    """Reserve bottom figure margin for external legends.

    Args:
        fig (matplotlib.figure.Figure): Target figure.
        bottom (float): Bottom margin fraction in figure coordinates.

    Returns:
        None: Modify ``fig`` layout in place.
    """
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        rect_bottom = min(max(float(bottom), 0.0), 0.35)
        engine.set(rect=(0.0, rect_bottom, 1.0, 1.0))
    else:
        fig.subplots_adjust(bottom=bottom)


def _legend_below(
    fig: plt.Figure,
    handles: Sequence,
    labels: Sequence[str],
    ncol: int,
    y: float = 0.05,
) -> None:
    """Place a figure-level legend at top center for consistency.

    Args:
        fig (matplotlib.figure.Figure): Figure object.
        handles (Sequence): Legend handle objects.
        labels (Sequence[str]): Legend labels.
        ncol (int): Number of legend columns.
        y (float): Vertical anchor in figure coordinates.

    Returns:
        None: Add legend artist to ``fig``.
    """
    if not handles:
        return
    offset = float(max(0.0, min(0.4, y)))
    anchor_y = max(0.65, min(0.99, 0.995 - offset))
    ncol_use = max(1, int(ncol))

    figure_legend(
        fig,
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=ncol_use,
        frameon=False,
    )


def plot_initial_ph_by_concentration(
    results: List[Dict], output_dir: str | None = None
) -> str:
    """Plot initial pH distributions by NaCl concentration.

    Args:
        results (list[dict]): Per-run analysis payloads containing raw run
            data.
        output_dir (str, optional): Directory for output figure bundle.
            Defaults to ``"output/qc"``.

    Returns:
        str: PNG path, or ``""`` when no usable initial-pH data are available.

    Note:
        The reference line at pH 2.88 provides a baseline check. Failure modes:
        strong condition-dependent offsets or very wide within-condition spread,
        both of which suggest calibration or preparation inconsistency.

    References:
        QC visualization for replicate baseline measurements.
    """
    return _plot_initial_ph_by_nacl(
        results,
        output_dir=output_dir,
        file_stem="initial_ph_by_nacl",
    )


def plot_initial_ph_scatter(results: List[Dict], output_dir: str | None = None) -> str:
    """Plot mean initial pH (plus/minus SD) across NaCl conditions.

    Args:
        results (list[dict]): Per-run analysis payloads containing raw run
            data.
        output_dir (str, optional): Directory for output figure bundle.
            Defaults to ``"output/qc"``.

    Returns:
        str: PNG path, or ``""`` when no valid initial-pH data exist.

    Note:
        Failure modes include monotonic drift in mean initial pH with NaCl or
        unusually large SD at one condition, both of which can confound pKa
        trends.

    References:
        Descriptive mean-plus-variability QC diagnostics.
    """
    return _plot_initial_ph_scatter_with_errorbar(
        results,
        output_dir=output_dir,
        file_stem="initial_ph_scatter_with_errorbar",
    )


def plot_temperature_boxplots(
    results: List[Dict], output_dir: str | None = None
) -> str:
    """Plot temperature distributions by NaCl with target and tolerance band.

    Args:
        results (list[dict]): Per-run analysis payloads with raw temperature
            channels in degrees Celsius.
        output_dir (str, optional): Directory for output figure bundle.
            Defaults to ``"output/qc"``.

    Returns:
        str: PNG path, or ``""`` when no temperature data are available.

    Note:
        Uses deterministic subsampling for overplotted points while preserving full
        boxplot statistics. Failure modes: frequent values outside the
        26.0 +/- 1.0 deg C band or condition-dependent thermal drift.

    References:
        Temperature-control verification in wet-chemistry experiments.
    """
    return _plot_temperature_control_by_nacl(
        results,
        output_dir=output_dir,
        file_stem="temperature_control_by_nacl",
    )


def plot_equivalence_volumes(
    results_df: pd.DataFrame, output_dir: str | None = None
) -> str:
    """Plot per-run equivalence volumes by NaCl with uncertainty bars."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if results_df.empty or "Veq (used)" not in results_df.columns:
        return ""

    setup_plot_style()
    rng = _rng(2)

    cols = ResultColumns()
    data = results_df[[cols.nacl, "Veq (used)", "Veq uncertainty (ΔVeq)"]].copy()
    data = data[data["Veq (used)"].notna()]
    if data.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6.2))
    _reserve_bottom(fig, bottom=0.34)

    for nacl in sorted(data[cols.nacl].unique()):
        subset = data[data[cols.nacl] == nacl]
        veq = subset["Veq (used)"].to_numpy(dtype=float)
        unc = subset["Veq uncertainty (ΔVeq)"].fillna(0.0).to_numpy(dtype=float)
        xs = _jitter(float(nacl), len(veq), scale=0.010, rng=rng)
        ax.errorbar(
            xs,
            veq,
            yerr=unc,
            fmt=_marker_for_nacl(float(nacl)),
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            ecolor="0.45",
            elinewidth=1.1,
            capsize=3,
            linestyle="none",
            alpha=0.90,
        )

    ax.axhline(25.0, color="0.15", linestyle="--", linewidth=1.5)
    _set_nacl_axis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=FONT_SIZES["axis_label"],
        labelpad=6,
    )
    ax.set_ylabel(
        r"Equivalence volume / $\mathrm{cm^3}$",
        fontsize=FONT_SIZES["axis_label"],
        labelpad=6,
    )
    _apply_qc_typography(ax, "Equivalence Volume Verification Across Ionic Strength")
    _qc_grid(ax, y_only=True)

    handles: List = [
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.15",
            linewidth=1.5,
            label="Expected equivalence volume (25.0 cm$^3$)",
        ),
    ]
    labels: List[str] = [handles[0].get_label()]
    for nacl in NACL_LEVELS:
        handles.append(
            Line2D(
                [],
                [],
                marker=_marker_for_nacl(float(nacl)),
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=7,
                label=f"{float(nacl):.2f} mol dm$^{{-3}}$",
            )
        )
        labels.append(handles[-1].get_label())
    _legend_below(fig, handles, labels, ncol=3, y=0.05)

    out_path = _save_qc_figure(
        fig, "equivalence_volumes_by_nacl", output_dir=output_dir, kind="methods"
    )
    plt.close(fig)
    return out_path


def plot_hh_slope_diagnostics(
    results_df: pd.DataFrame, output_dir: str | None = None
) -> str:
    """Plot H-H slope and R^2 diagnostics by NaCl concentration."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if results_df.empty or "Slope (buffer fit)" not in results_df.columns:
        return ""

    setup_plot_style()
    rng = _rng(3)

    cols = ResultColumns()
    data = results_df[[cols.nacl, "Slope (buffer fit)", "R2 (buffer fit)"]].copy()
    data = data[data["Slope (buffer fit)"].notna()]
    if data.empty:
        return ""

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=fig_size("diagnostic"), constrained_layout=True
    )
    apply_subplot_padding(fig, wspace=0.24, hspace=0.18, w_pad=0.06, h_pad=0.08)

    for nacl in sorted(data[cols.nacl].unique()):
        subset = data[data[cols.nacl] == nacl]
        slopes = subset["Slope (buffer fit)"].to_numpy(dtype=float)
        xs = _jitter(float(nacl), len(slopes), scale=0.010, rng=rng)
        ax1.scatter(
            xs,
            slopes,
            s=62,
            marker=_marker_for_nacl(float(nacl)),
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.90,
        )

    ax1.axhspan(0.95, 1.00, color="0.92", alpha=1.0, zorder=0)
    ax1.axhline(
        1.0, color="0.15", linestyle="--", linewidth=1.3, zorder=1, label="Ideal slope"
    )
    _set_nacl_axis(ax1)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    set_axis_labels(ax1, rf"${LABEL_NACL}$", rf"${LABEL_HH_SLOPE}$")
    set_ticks(ax1, xstep=0.2, ystep=0.01, yfmt="%.2f")
    ax1.set_title("H-H slope validation")
    clean_axis(ax1, grid_axis="y", nbins_x=5, nbins_y=5)

    for nacl in sorted(data[cols.nacl].unique()):
        subset = data[data[cols.nacl] == nacl]
        r2_vals = subset["R2 (buffer fit)"].to_numpy(dtype=float)
        xs = _jitter(float(nacl), len(r2_vals), scale=0.010, rng=rng)
        ax2.scatter(
            xs,
            r2_vals,
            s=62,
            marker=_marker_for_nacl(float(nacl)),
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.90,
        )

    ax2.axhline(
        0.99,
        color="0.15",
        linestyle="--",
        linewidth=1.3,
        label="Excellent fit threshold",
    )
    _set_nacl_axis(ax2)
    r2_min = float(np.nanmin(data["R2 (buffer fit)"].to_numpy(dtype=float)))
    r2_max = float(np.nanmax(data["R2 (buffer fit)"].to_numpy(dtype=float)))
    safe_set_lims(ax2, y=(min(0.90, r2_min), max(1.005, r2_max)), pad_frac=0.02)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    set_axis_labels(ax2, rf"${LABEL_NACL}$", rf"${LABEL_R2}$")
    set_ticks(ax2, xstep=0.2, ystep=0.01, yfmt="%.2f")
    ax2.set_title(r"Goodness-of-fit diagnostics")
    clean_axis(ax2, grid_axis="y", nbins_x=5, nbins_y=5)

    marker_handles: List = [
        Line2D(
            [],
            [],
            marker=_marker_for_nacl(float(nacl)),
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=f"{float(nacl):.1f} mol dm$^{{-3}}$",
        )
        for nacl in sorted(data[cols.nacl].unique())
    ]

    line_handles: List = [
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.15",
            linewidth=1.3,
            label="Ideal slope (1.00)",
        ),
        Patch(facecolor="0.92", edgecolor="none", label="Acceptable range (±5%)"),
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.15",
            linewidth=1.3,
            label=r"Excellent fit threshold ($R^2 \geq 0.99$)",
        ),
    ]
    _legend_below(
        fig,
        line_handles + marker_handles,
        [h.get_label() for h in line_handles + marker_handles],
        ncol=4,
        y=0.06,
    )

    out_path = _save_qc_figure(
        fig,
        "hh_slope_and_r2_diagnostics",
        output_dir=output_dir,
        kind="diagnostics",
        legend_height=0.16,
    )
    plt.close(fig)
    return out_path


def plot_pka_precision(results_df: pd.DataFrame, output_dir: str | None = None) -> str:
    """Plot per-run apparent pKa precision by NaCl.

    Args:
        results_df (pandas.DataFrame): Consolidated run-level results with
            pKa and uncertainty columns (dimensionless).
        output_dir (str, optional): Directory for output figure bundle.
            Defaults to ``"output/qc"``.

    Returns:
        str: PNG path, or ``""`` when no valid pKa values are available.

    Note:
        Failure modes include condition-specific spread inflation or uncertainty bars
        that are inconsistent with observed replicate scatter.

    References:
        Precision and repeatability visualization for replicate chemical assays.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if results_df.empty:
        return ""

    setup_plot_style()
    rng = _rng(4)

    cols = ResultColumns()
    data = results_df[[cols.nacl, cols.pka_app, cols.pka_unc]].copy()
    data = data[data[cols.pka_app].notna()]
    if data.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6.2))
    _reserve_bottom(fig, bottom=0.34)

    for nacl in sorted(data[cols.nacl].unique()):
        subset = data[data[cols.nacl] == nacl]
        pkas = subset[cols.pka_app].to_numpy(dtype=float)
        uncs = subset[cols.pka_unc].fillna(0.0).to_numpy(dtype=float)
        xs = _jitter(float(nacl), len(pkas), scale=0.010, rng=rng)

        ax.errorbar(
            xs,
            pkas,
            yerr=uncs,
            fmt=_marker_for_nacl(float(nacl)),
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            ecolor="0.45",
            elinewidth=1.1,
            capsize=3,
            linestyle="none",
            alpha=0.90,
        )

    _set_nacl_axis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=FONT_SIZES["axis_label"],
        labelpad=6,
    )
    ax.set_ylabel(r"Apparent $pK_a$", fontsize=FONT_SIZES["axis_label"], labelpad=6)
    _apply_qc_typography(ax, r"Apparent $pK_a$ Precision: Trial-to-Trial Variability")
    _qc_grid(ax, y_only=True)

    handles = [
        Line2D(
            [],
            [],
            marker=_marker_for_nacl(float(n)),
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=f"{float(n):.2f} mol dm$^{{-3}}$",
        )
        for n in NACL_LEVELS
    ]
    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=3, y=0.15)

    out_path = _save_qc_figure(
        fig,
        "pka_precision_by_nacl",
        output_dir=output_dir,
        kind="supplemental",
    )
    plt.close(fig)
    return out_path


def plot_buffer_region_coverage(
    results: List[Dict], output_dir: str | None = None
) -> str:
    """Plot buffer-region point coverage by NaCl concentration.

    Args:
        results (list[dict]): Per-run analysis payloads containing
            ``buffer_region`` tables.
        output_dir (str, optional): Directory for output figure bundle.
            Defaults to ``"output/qc"``.

    Returns:
        str: PNG path, or ``""`` when no runs provide buffer-region metadata.

    Note:
        Reference line at 10 points marks a practical minimum for stable fit
        diagnostics. Failure mode: repeated runs below this threshold.

    References:
        Minimum-data-density guidance for linear regression stability.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    data_by_nacl: Dict[float, List[int]] = {}
    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        if not np.isfinite(nacl):
            continue
        buffer_df = res.get("buffer_region", pd.DataFrame())
        data_by_nacl.setdefault(nacl, []).append(int(len(buffer_df)))

    if not data_by_nacl:
        return ""

    setup_plot_style()
    rng = _rng(5)

    concentrations = [c for c in NACL_LEVELS if c in data_by_nacl]
    values_list = [data_by_nacl[c] for c in concentrations]
    means = np.array([np.mean(v) for v in values_list], dtype=float)
    stds = np.array(
        [np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in values_list], dtype=float
    )

    fig, ax = plt.subplots(figsize=(10, 6.2))
    _reserve_bottom(fig, bottom=0.34)

    ax.bar(
        concentrations,
        means,
        width=0.12,
        color="0.88",
        edgecolor="black",
        linewidth=1.0,
        zorder=1,
    )
    ax.errorbar(
        concentrations,
        means,
        yerr=stds,
        fmt="none",
        ecolor="0.15",
        elinewidth=1.1,
        capsize=4,
        zorder=3,
    )

    for conc, vals in zip(concentrations, values_list):
        xs = _jitter(float(conc), len(vals), scale=0.012, rng=rng)
        ax.scatter(
            xs,
            vals,
            s=48,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
            zorder=4,
        )

    ax.axhline(10, color="0.20", linestyle="--", linewidth=1.4, zorder=2)

    _set_nacl_axis(ax)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=FONT_SIZES["axis_label"],
        labelpad=6,
    )
    ax.set_ylabel(
        "Number of points in buffer region",
        fontsize=FONT_SIZES["axis_label"],
        labelpad=6,
    )
    _apply_qc_typography(ax, r"Buffer Region Coverage ($|\mathrm{pH} - pK_a| \leq 1$)")
    _qc_grid(ax, y_only=True)

    handles = [
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.20",
            linewidth=1.4,
            label="Minimum recommended (10 points)",
        ),
        Patch(facecolor="0.88", edgecolor="black", label="Mean number of points"),
        Line2D(
            [],
            [],
            color="0.15",
            marker="_",
            linestyle="none",
            markersize=16,
            label="± one standard deviation",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Individual trials",
        ),
    ]
    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=2, y=0.05)

    out_path = _save_qc_figure(
        fig,
        "buffer_region_coverage",
        output_dir=output_dir,
        kind="methods",
    )
    plt.close(fig)
    return out_path


def plot_residuals_analysis(results: List[Dict], output_dir: str | None = None) -> str:
    """Plot residual structure of H-H fits (scatter plus histogram).

    Args:
        results (list[dict]): Per-run analysis payloads containing
            ``buffer_region`` fit outputs.
        output_dir (str, optional): Directory for output figure bundle.
            Defaults to ``"output/qc"``.

    Returns:
        str: PNG path, or ``""`` when residuals cannot be computed.

    Note:
        Residuals are ``pH_step - pH_fit`` for buffer-region points only.
        Failure modes: curved residual trend versus log ratio, skewed
        histogram, or outlier clusters at specific conditions.

    References:
        Regression residual diagnostics (pattern and distribution checks).
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    all_residuals: List[float] = []
    all_x_values: List[float] = []
    all_nacl: List[float] = []

    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        if not np.isfinite(nacl):
            continue
        buffer_df = res.get("buffer_region", pd.DataFrame())
        if (
            buffer_df.empty
            or "pH_step" not in buffer_df.columns
            or "pH_fit" not in buffer_df.columns
        ):
            continue

        ph_obs = pd.to_numeric(buffer_df["pH_step"], errors="coerce").to_numpy(
            dtype=float
        )
        ph_fit = pd.to_numeric(buffer_df["pH_fit"], errors="coerce").to_numpy(
            dtype=float
        )
        x_vals = pd.to_numeric(
            buffer_df.get("log10_ratio", np.nan), errors="coerce"
        ).to_numpy(dtype=float)

        mask = np.isfinite(ph_obs) & np.isfinite(ph_fit) & np.isfinite(x_vals)
        residuals = ph_obs[mask] - ph_fit[mask]

        all_residuals.extend(residuals.tolist())
        all_x_values.extend(x_vals[mask].tolist())
        all_nacl.extend([nacl] * int(np.sum(mask)))

    if not all_residuals:
        return ""

    setup_plot_style()

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=fig_size("diagnostic"), constrained_layout=True
    )
    apply_subplot_padding(fig, wspace=0.25, hspace=0.18, w_pad=0.06, h_pad=0.08)

    for nacl in sorted(set(all_nacl)):
        m = np.array(all_nacl) == nacl
        x = np.array(all_x_values)[m]
        y = np.array(all_residuals)[m]
        ax1.scatter(
            x,
            y,
            s=24,
            marker=_marker_for_nacl(float(nacl)),
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.55,
        )

    ax1.axhline(0, color="0.15", linestyle="--", linewidth=1.2)
    set_axis_labels(ax1, rf"${LABEL_LOG_RATIO}$", rf"${LABEL_RESIDUAL}$")
    set_ticks(ax1, xstep=0.5, ystep=0.05)
    ax1.set_title("Residuals vs buffer ratio")
    clean_axis(ax1, grid_axis="both", nbins_x=6, nbins_y=6)

    if should_plot_distribution(len(all_residuals)):
        bins = max(8, min(20, int(np.sqrt(len(all_residuals)) * 2)))
        ax2.hist(
            all_residuals, bins=bins, color="0.82", edgecolor="black", linewidth=0.8
        )
        ax2.axvline(0, color="0.15", linestyle="--", linewidth=1.2)
        set_axis_labels(ax2, rf"${LABEL_RESIDUAL}$", "Frequency")
        set_ticks(ax2, xstep=0.05, ystep=None, ynbins=5)
        ax2.set_title("Residual distribution")
        clean_axis(ax2, grid_axis="y", nbins_x=6, nbins_y=5)
    else:
        y_positions = np.zeros(len(all_residuals), dtype=float)
        ax2.scatter(
            np.asarray(all_residuals, dtype=float),
            y_positions,
            s=24,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            alpha=0.65,
        )
        ax2.axvline(0, color="0.15", linestyle="--", linewidth=1.2)
        set_axis_labels(ax2, rf"${LABEL_RESIDUAL}$", "")
        ax2.set_yticks([])
        ax2.set_title("Residual strip view (n < 20)")
        clean_axis(ax2, grid_axis="x", nbins_x=6, nbins_y=3)

    resid = np.asarray(all_residuals, dtype=float)
    finite = np.isfinite(resid)
    if np.any(finite):
        rmin, rmax = float(np.min(resid[finite])), float(np.max(resid[finite]))
        lim = max(abs(rmin), abs(rmax), 1e-6)
        ax1.set_ylim(-1.05 * lim, 1.05 * lim)
        safe_set_lims(ax2, x=(rmin, rmax), pad_frac=0.08)

    handles = [
        Line2D(
            [],
            [],
            marker=_marker_for_nacl(float(n)),
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label=f"{float(n):.2f} mol dm$^{{-3}}$",
        )
        for n in NACL_LEVELS
    ]
    handles.append(
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.15",
            linewidth=1.2,
            label="Zero residual reference",
        )
    )
    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=3, y=0.06)

    out_path = _save_qc_figure(
        fig,
        "hh_residuals_analysis",
        output_dir=output_dir,
        kind="diagnostics",
        legend_height=0.16,
    )
    plt.close(fig)
    return out_path


def plot_half_equivalence_check(
    results_df: pd.DataFrame, output_dir: str | None = None
) -> str:
    """Plot ``V_half / V_eq`` ratio by NaCl concentration.

    Args:
        results_df (pandas.DataFrame): Consolidated run-level results with
            ``V_half (cm^3)`` and ``Veq (used)`` columns.
        output_dir (str, optional): Directory for output figure bundle.
            Defaults to ``"output/qc"``.

    Returns:
        str: PNG path, or ``""`` when required columns/data are unavailable.

    Note:
        Theoretical ratio is 0.5. Failure modes include systematic drift from 0.5 or
    broad scatter, indicating unstable equivalence detection or interpolation.

    References:
        Half-equivalence geometry for monoprotic weak-acid titration.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    required_cols = ["V_half (cm^3)", "Veq (used)"]
    if results_df.empty or not all(col in results_df.columns for col in required_cols):
        return ""

    setup_plot_style()
    rng = _rng(6)

    cols = ResultColumns()
    data = results_df[[cols.nacl, "V_half (cm^3)", "Veq (used)"]].copy()

    v_half = pd.to_numeric(data["V_half (cm^3)"], errors="coerce")
    v_eq = pd.to_numeric(data["Veq (used)"], errors="coerce")
    data["ratio"] = v_half / v_eq
    data = data[data["ratio"].notna()]
    if data.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6.2))
    _reserve_bottom(fig, bottom=0.34)

    for nacl in sorted(data[cols.nacl].unique()):
        subset = data[data[cols.nacl] == nacl]
        ratios = subset["ratio"].to_numpy(dtype=float)
        xs = _jitter(float(nacl), len(ratios), scale=0.010, rng=rng)
        ax.scatter(
            xs,
            ratios,
            s=62,
            marker=_marker_for_nacl(float(nacl)),
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.90,
        )

    ax.axhspan(0.49, 0.51, color="0.92", alpha=1.0, zorder=0)
    ax.axhline(0.5, color="0.15", linestyle="--", linewidth=1.5, zorder=1)

    _set_nacl_axis(ax)
    ax.set_ylim(0.45, 0.55)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=FONT_SIZES["axis_label"],
        labelpad=6,
    )
    ax.set_ylabel(
        r"$V_{\mathrm{half}} / V_{\mathrm{eq}}$ (dimensionless)",
        fontsize=FONT_SIZES["axis_label"],
        labelpad=6,
    )
    _apply_qc_typography(
        ax,
        r"Half-Equivalence Verification: $V_{\mathrm{half}} / V_{\mathrm{eq}}$ Ratio",
    )
    _qc_grid(ax, y_only=True)

    handles: List = [
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.15",
            linewidth=1.5,
            label=r"Theoretical value ($V_{\mathrm{half}} / V_{\mathrm{eq}} = 0.50$)",
        ),
        Patch(facecolor="0.92", edgecolor="none", label="Acceptable range (±2%)"),
    ]
    labels: List[str] = [h.get_label() for h in handles]
    for nacl in NACL_LEVELS:
        handles.append(
            Line2D(
                [],
                [],
                marker=_marker_for_nacl(float(nacl)),
                linestyle="none",
                markerfacecolor="white",
                markeredgecolor="black",
                markersize=7,
                label=f"{float(nacl):.2f} mol dm$^{{-3}}$",
            )
        )
        labels.append(handles[-1].get_label())

    _legend_below(fig, handles, labels, ncol=3, y=0.05)

    out_path = _save_qc_figure(
        fig,
        "half_equivalence_verification",
        output_dir=output_dir,
        kind="methods",
    )
    plt.close(fig)
    return out_path


def plot_temperature_and_calibration_qc(
    results: List[Dict],
    results_df: pd.DataFrame | None = None,
    output_dir: str | None = None,
    file_stem: str = "temperature_and_calibration_qc",
    return_figure: bool = False,
    return_metadata: bool = False,
):
    """Figure 5: temperature stability and calibration checkpoint diagnostics."""
    setup_plot_style()
    _ = results_df
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    temp_outlier_count = 0
    temp_orders: List[float] = []
    temp_means: List[float] = []
    temp_sds: List[float] = []
    temp_colors: List[str] = []

    calibration_orders: List[float] = []
    calibration_values: List[float] = []

    for order, res in enumerate(results, start=1):
        nacl = float(res.get("nacl_conc", np.nan))
        raw_df = res.get("data", pd.DataFrame())
        if not isinstance(raw_df, pd.DataFrame):
            continue

        if "Temperature (°C)" in raw_df.columns:
            temp = pd.to_numeric(raw_df["Temperature (°C)"], errors="coerce").to_numpy(
                dtype=float
            )
            temp = temp[np.isfinite(temp)]
            if len(temp):
                t_mean = float(np.mean(temp))
                t_sd = float(np.std(temp, ddof=1)) if len(temp) > 1 else 0.0
                is_outlier = bool(t_mean < 25.0 or t_mean > 27.0)
                temp_outlier_count += int(is_outlier)
                temp_orders.append(float(order))
                temp_means.append(t_mean)
                temp_sds.append(t_sd)
                temp_colors.append("0.75" if is_outlier else color_for_nacl(nacl))

        for candidate in (
            "Calibration pH 7.00",
            "Calibration pH",
            "Buffer Check pH",
            "pH 7 Buffer",
        ):
            if candidate in raw_df.columns:
                vals = pd.to_numeric(raw_df[candidate], errors="coerce").to_numpy(
                    dtype=float
                )
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    calibration_orders.append(float(order))
                    calibration_values.append(float(np.mean(vals)))
                break

    has_calibration = len(calibration_values) > 0
    if has_calibration:
        fig, (ax_temp, ax_cal) = plt.subplots(
            1, 2, figsize=(12.0, 4.0), constrained_layout=False
        )
    else:
        fig, ax_temp = plt.subplots(1, 1, figsize=(10.0, 4.0), constrained_layout=False)
        ax_cal = None

    ax_temp.axhspan(25.0, 27.0, color="0.94", zorder=0)
    ax_temp.axhline(26.0, color="0.2", linestyle="--", linewidth=1.2)
    for x, y, yerr, color in zip(temp_orders, temp_means, temp_sds, temp_colors):
        ax_temp.errorbar(
            [x],
            [y],
            yerr=[yerr],
            fmt="o",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=0.6,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.4,
            alpha=ALPHAS["replicate"],
            zorder=3,
        )

    ax_temp.set_xlabel("Run order")
    ax_temp.set_ylabel(MATH_LABELS["temperature_c"])
    ax_temp.set_title("Temperature control", fontsize=FONT_SIZES["title"], pad=8)
    ax_temp.grid(True, axis="y", alpha=0.14, linestyle=":", linewidth=0.8)
    if temp_orders:
        ax_temp.set_xlim(0.5, float(max(temp_orders) + 0.5))
    else:
        ax_temp.text(
            0.02,
            0.96,
            "No temperature readings found",
            transform=ax_temp.transAxes,
            ha="left",
            va="top",
            fontsize=FONT_SIZES["annotation"],
            color="0.35",
        )

    if has_calibration and ax_cal is not None:
        vals = np.asarray(calibration_values, dtype=float)
        orders = np.asarray(calibration_orders, dtype=float)
        ax_cal.axhspan(6.95, 7.05, color="0.94", zorder=0)
        ax_cal.axhline(7.0, color="0.2", linestyle="--", linewidth=1.2)
        ax_cal.scatter(
            orders,
            vals,
            s=34,
            facecolor="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax_cal.set_xlabel("Run order")
        ax_cal.set_ylabel(MATH_LABELS["calibration_ph"])
        ax_cal.set_title("Calibration check", fontsize=FONT_SIZES["title"], pad=8)
        ax_cal.grid(True, axis="y", alpha=0.14, linestyle=":", linewidth=0.8)
    else:
        ax_temp.text(
            0.98,
            0.96,
            "Calibration readings not recorded",
            transform=ax_temp.transAxes,
            ha="right",
            va="top",
            fontsize=FONT_SIZES["annotation"],
            color="0.35",
        )

    legend_handles = [
        Patch(facecolor="0.94", edgecolor="none", label="Tolerance band (±1.0 °C)"),
        Line2D(
            [],
            [],
            color="0.2",
            linestyle="--",
            linewidth=1.2,
            label=r"Target $26.0\,^\circ\mathrm{C}$",
        ),
    ]
    for nacl in sorted(
        {
            float(r.get("nacl_conc", np.nan))
            for r in results
            if np.isfinite(float(r.get("nacl_conc", np.nan)))
        }
    ):
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markerfacecolor=color_for_nacl(nacl),
                markeredgecolor="black",
                markersize=6,
                label=rf"$[\mathrm{{NaCl}}]={nacl:.1f}\ \mathrm{{mol\,dm^{{-3}}}}$",
            )
        )
    figure_legend(
        fig,
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="upper center",
        ncol=min(max(len(legend_handles), 1), 5),
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
    )

    out_path = _save_qc_figure(
        fig, file_stem, output_dir=output_dir, kind="qc", legend_height=0.08
    )
    metadata = {
        "temperature_runs": int(len(temp_orders)),
        "temperature_outliers": int(temp_outlier_count),
        "calibration_points": int(len(calibration_values)),
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


def generate_all_qc_plots(
    results: List[Dict],
    results_df: pd.DataFrame,
    output_dir: str | None = None,
) -> List[str]:
    """Generate the full QC figure suite for one analysis batch.

    Args:
        results (list[dict]): Per-run analysis payloads.
        results_df (pandas.DataFrame): Consolidated run-level results
            dataframe.
        output_dir (str, optional): Destination directory for QC figure
            bundles. Defaults to ``"output/qc"``.

    Returns:
        list[str]: PNG paths for all successfully generated QC figures.

    Note:
        Figure generation is deterministic because jitter/subsampling helpers use
        fixed RNG seeds. IA correspondence: this is the consolidated QC evidence
        bundle used to justify all-valid, qc-pass, and strict-fit subsets.

    References:
        Deterministic QC artifact generation for reproducible reporting.
    """
    setup_plot_style()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plot_paths: List[str] = []

    for fn in (
        plot_initial_ph_by_concentration,
        plot_initial_ph_scatter,
        plot_temperature_boxplots,
    ):
        path = fn(results, output_dir)
        if path:
            plot_paths.append(path)

    for fn in (
        plot_equivalence_volumes,
        plot_hh_slope_diagnostics,
        plot_pka_precision,
        plot_half_equivalence_check,
    ):
        path = fn(results_df, output_dir)
        if path:
            plot_paths.append(path)

    path = plot_residuals_analysis(results, output_dir)
    if path:
        plot_paths.append(path)

    path = plot_buffer_region_coverage(results, output_dir)
    if path:
        plot_paths.append(path)

    return plot_paths
