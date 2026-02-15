"""Generate QC figures that justify run inclusion and interpretation strength.

These figures correspond to IA quality arguments:
- endpoint plausibility,
- model validity,
- controlled-variable stability, and
- repeatability across ionic-strength conditions.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from ..schema import ResultColumns
from .titration_plots import save_figure_bundle, setup_plot_style

_NACL_LEVELS = np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=float)
_QC_TITLE_FONTSIZE = 14
_QC_LABEL_FONTSIZE = 12
_QC_TICK_FONTSIZE = 11
_QC_LEGEND_FONTSIZE = 10


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
    ax.set_xticks(_NACL_LEVELS)
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
    ax.set_title(title, fontweight="bold", fontsize=_QC_TITLE_FONTSIZE, pad=10)
    ax.tick_params(axis="both", which="major", labelsize=_QC_TICK_FONTSIZE)


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
    fig.subplots_adjust(bottom=bottom)


def _legend_below(
    fig: plt.Figure,
    handles: Sequence,
    labels: Sequence[str],
    ncol: int,
    y: float = 0.15,
) -> None:
    """Place a figure-level legend below axes.

    Args:
        fig (matplotlib.figure.Figure): Figure object.
        handles (Sequence): Legend handle objects.
        labels (Sequence[str]): Legend labels.
        ncol (int): Number of legend columns.
        y (float): Vertical anchor in figure coordinates.

    Returns:
        None: Add legend artist to ``fig``.
    """
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=_QC_LEGEND_FONTSIZE,
        handlelength=2.6,
        columnspacing=1.6,
        handletextpad=0.7,
        labelspacing=0.8,
    )


def plot_initial_ph_by_concentration(
    results: List[Dict], output_dir: str = "output/qc"
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
    os.makedirs(output_dir, exist_ok=True)

    data_by_nacl: Dict[float, List[float]] = {}
    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        if not np.isfinite(nacl):
            continue
        raw_df = res.get("data", pd.DataFrame())
        if raw_df.empty or "pH" not in raw_df.columns:
            continue
        ph_vals = pd.to_numeric(raw_df["pH"], errors="coerce").to_numpy(dtype=float)
        ph_vals = ph_vals[np.isfinite(ph_vals)]
        if ph_vals.size == 0:
            continue
        data_by_nacl.setdefault(nacl, []).append(float(ph_vals[0]))

    if not data_by_nacl:
        return ""

    setup_plot_style()
    rng = _rng(0)

    concentrations = [c for c in _NACL_LEVELS if c in data_by_nacl]
    box_data = [data_by_nacl[c] for c in concentrations]
    pos, labels = _categorical_positions(concentrations)

    fig, ax = plt.subplots(figsize=(10, 6.2))
    _reserve_bottom(fig, bottom=0.28)

    bp = ax.boxplot(
        box_data,
        positions=pos,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6
        ),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        boxprops=dict(edgecolor="black", linewidth=1.0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("0.88")
        patch.set_alpha(1.0)

    for i, values in enumerate(box_data):
        xs = pos[i] + rng.normal(0.0, 0.08, size=len(values))
        ax.scatter(
            xs,
            values,
            s=42,
            facecolor="white",
            edgecolor="black",
            linewidth=1.0,
            alpha=0.85,
            zorder=3,
        )

    expected_ph0 = 2.88
    ax.axhline(expected_ph0, color="0.20", linestyle="--", linewidth=1.4)

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=_QC_TICK_FONTSIZE)
    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax.set_ylabel(
        r"Initial pH ($\mathrm{pH}_0$)", fontsize=_QC_LABEL_FONTSIZE, labelpad=6
    )
    _apply_qc_typography(ax, "Initial pH Verification Across Ionic Strength Conditions")

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    _qc_grid(ax, y_only=True)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Individual trials (initial pH)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Mean (boxplot marker)",
        ),
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.20",
            linewidth=1.4,
            label="Expected initial pH (ideal)",
        ),
        Patch(facecolor="0.88", edgecolor="black", label="Box-and-whisker summary"),
    ]
    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=2, y=0.1)

    out_path = os.path.join(output_dir, "initial_ph_by_nacl.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path


def plot_initial_ph_scatter(results: List[Dict], output_dir: str = "output/qc") -> str:
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
    os.makedirs(output_dir, exist_ok=True)

    data_by_nacl: Dict[float, List[float]] = {}
    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        if not np.isfinite(nacl):
            continue
        raw_df = res.get("data", pd.DataFrame())
        if raw_df.empty or "pH" not in raw_df.columns:
            continue
        ph_vals = pd.to_numeric(raw_df["pH"], errors="coerce").to_numpy(dtype=float)
        ph_vals = ph_vals[np.isfinite(ph_vals)]
        if ph_vals.size == 0:
            continue
        data_by_nacl.setdefault(nacl, []).append(float(ph_vals[0]))

    if not data_by_nacl:
        return ""

    setup_plot_style()

    concentrations = [c for c in _NACL_LEVELS if c in data_by_nacl]
    means = np.array([np.mean(data_by_nacl[c]) for c in concentrations], dtype=float)
    stds = np.array(
        [
            np.std(data_by_nacl[c], ddof=1) if len(data_by_nacl[c]) > 1 else 0.0
            for c in concentrations
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 6.0))
    _reserve_bottom(fig, bottom=0.32)

    ax.errorbar(
        concentrations,
        means,
        yerr=stds,
        fmt="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=1.0,
        ecolor="0.25",
        elinewidth=1.2,
        capsize=4,
        linestyle="none",
    )

    overall_mean = float(np.mean(means)) if means.size else np.nan
    expected_ph0 = 2.88
    if np.isfinite(overall_mean):
        ax.axhline(overall_mean, color="0.35", linestyle="--", linewidth=1.2)
    ax.axhline(expected_ph0, color="0.20", linestyle=":", linewidth=1.2)

    _set_nacl_axis(ax)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax.set_ylabel(
        r"Initial pH ($\mathrm{pH}_0$)", fontsize=_QC_LABEL_FONTSIZE, labelpad=6
    )
    _apply_qc_typography(
        ax, "Initial pH: Mean and Standard Deviation vs. Ionic Strength"
    )
    _qc_grid(ax, y_only=True)

    handles: List[Line2D] = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
            label="Mean initial pH ± one standard deviation",
        ),
        Line2D(
            [],
            [],
            linestyle="--",
            color="0.35",
            linewidth=1.2,
            label="Overall mean (across conditions)",
        ),
        Line2D(
            [],
            [],
            linestyle=":",
            color="0.20",
            linewidth=1.2,
            label="Expected initial pH (ideal)",
        ),
    ]
    if not np.isfinite(overall_mean):
        handles = [handles[0], handles[2]]

    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=1, y=0.1)

    out_path = os.path.join(output_dir, "initial_ph_scatter_with_errorbar.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path


def plot_temperature_boxplots(
    results: List[Dict], output_dir: str = "output/qc"
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
    os.makedirs(output_dir, exist_ok=True)

    data_by_nacl: Dict[float, List[float]] = {}
    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        if not np.isfinite(nacl):
            continue
        raw_df = res.get("data", pd.DataFrame())
        if raw_df.empty or "Temperature (°C)" not in raw_df.columns:
            continue
        temps = pd.to_numeric(raw_df["Temperature (°C)"], errors="coerce").to_numpy(
            dtype=float
        )
        temps = temps[np.isfinite(temps)]
        if temps.size == 0:
            continue
        data_by_nacl.setdefault(nacl, []).extend(temps.tolist())

    if not data_by_nacl:
        return ""

    setup_plot_style()
    rng = _rng(1)

    concentrations = [c for c in _NACL_LEVELS if c in data_by_nacl]
    box_data = [data_by_nacl[c] for c in concentrations]
    pos, labels = _categorical_positions(concentrations)

    target_temp = 26.0
    tol = 1.0
    max_points_per_group = 450

    fig, ax = plt.subplots(figsize=(10, 6.2))
    _reserve_bottom(fig, bottom=0.32)

    ax.axhspan(target_temp - tol, target_temp + tol, color="0.92", alpha=1.0, zorder=0)
    ax.axhline(target_temp, color="0.15", linewidth=1.6, zorder=1)

    bp = ax.boxplot(
        box_data,
        positions=pos,
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(
            marker="s", markerfacecolor="white", markeredgecolor="black", markersize=6
        ),
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        boxprops=dict(edgecolor="black", linewidth=1.0),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("0.80")
        patch.set_alpha(0.90)

    for i, values in enumerate(box_data):
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        if vals.size > max_points_per_group:
            idx = rng.choice(vals.size, size=max_points_per_group, replace=False)
            vals_plot = vals[idx]
        else:
            vals_plot = vals
        xs = pos[i] + rng.normal(0.0, 0.10, size=vals_plot.size)
        ax.scatter(
            xs,
            vals_plot,
            s=18,
            facecolor="white",
            edgecolor="black",
            linewidth=0.7,
            alpha=0.30,
            zorder=2,
        )

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=_QC_TICK_FONTSIZE)
    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax.set_ylabel(
        r"Temperature / $^\circ\mathrm{C}$", fontsize=_QC_LABEL_FONTSIZE, labelpad=6
    )
    _apply_qc_typography(ax, "Temperature Control Verification (All Measurements)")

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    _qc_grid(ax, y_only=True)

    y_all = np.concatenate([np.asarray(v, dtype=float) for v in box_data if len(v) > 0])
    y_all = y_all[np.isfinite(y_all)]
    if y_all.size > 0:
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
        pad = 0.08 * max(1e-6, (y_max - y_min))
        ax.set_ylim(y_min - pad, y_max + pad)

    handles = [
        Patch(facecolor="0.92", edgecolor="none", label="Tolerance band (±1 °C)"),
        Line2D(
            [], [], color="0.15", linewidth=1.6, label="Target temperature (26.0 °C)"
        ),
        Patch(facecolor="0.80", edgecolor="black", label="Box-and-whisker summary"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="Subsampled individual readings (for visibility)",
        ),
    ]
    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=2, y=0.125)

    out_path = os.path.join(output_dir, "temperature_control_by_nacl.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path


def plot_equivalence_volumes(
    results_df: pd.DataFrame, output_dir: str = "output/qc"
) -> str:
    """Plot per-run equivalence volumes by NaCl with uncertainty bars.

    Args:
        results_df (pandas.DataFrame): Consolidated run-level results
            containing ``Veq (used)`` and ``Veq uncertainty (ΔVeq)`` in cm^3.
        output_dir (str, optional): Directory for output figure bundle.
            Defaults to ``"output/qc"``.

    Returns:
        str: PNG path, or ``""`` when required data are unavailable.

    Note:
        Reference line at 25.0 cm^3 reflects the nominal stoichiometric expectation.
        Failure modes: systematic offsets from 25.0 cm^3 or large
        within-condition scatter indicating unstable equivalence detection.

    References:
        Derivative-based equivalence diagnostics for acid-base titration.
    """
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
        veqs = subset["Veq (used)"].to_numpy(dtype=float)
        uncs = subset["Veq uncertainty (ΔVeq)"].fillna(0.0).to_numpy(dtype=float)
        xs = _jitter(float(nacl), len(veqs), scale=0.010, rng=rng)

        ax.errorbar(
            xs,
            veqs,
            yerr=uncs,
            fmt=_marker_for_nacl(float(nacl)),
            markersize=7,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.0,
            ecolor="0.35",
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
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax.set_ylabel(
        r"Equivalence volume / $\mathrm{cm^3}$", fontsize=_QC_LABEL_FONTSIZE, labelpad=6
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

    for nacl in _NACL_LEVELS:
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

    _legend_below(fig, handles, labels, ncol=3, y=0.15)

    out_path = os.path.join(output_dir, "equivalence_volumes_by_nacl.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path


def plot_hh_slope_diagnostics(
    results_df: pd.DataFrame, output_dir: str = "output/qc"
) -> str:
    """Plot H-H slope and R^2 diagnostics by NaCl concentration.

    Args:
        results_df (pandas.DataFrame): Consolidated run-level results
            containing slope and R^2 columns.
        output_dir (str, optional): Directory for output figure bundle.
            Defaults to ``"output/qc"``.

    Returns:
        str: PNG path, or ``""`` when required data are missing.

    Note:
        Slope near 1 supports H-H assumptions. Low R^2 or strong slope drift are
        typical failure modes and should trigger residual inspection.

    References:
        Henderson-Hasselbalch linear-fit diagnostics.
    """
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0))
    _reserve_bottom(fig, bottom=0.30)

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

    ax1.axhspan(0.95, 1.05, color="0.92", alpha=1.0, zorder=0)
    ax1.axhline(1.0, color="0.15", linestyle="--", linewidth=1.3, zorder=1)
    _set_nacl_axis(ax1)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax1.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax1.set_ylabel(
        "Henderson–Hasselbalch slope", fontsize=_QC_LABEL_FONTSIZE, labelpad=6
    )
    _apply_qc_typography(ax1, "Henderson–Hasselbalch Slope Validation")
    _qc_grid(ax1, y_only=True)

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

    ax2.axhline(0.99, color="0.15", linestyle="--", linewidth=1.3)
    _set_nacl_axis(ax2)
    ax2.set_ylim(0.90, 1.005)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax2.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol}\,\mathrm{dm^{-3}}$",
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax2.set_ylabel(
        r"$R^2$ (buffer-region regression)", fontsize=_QC_LABEL_FONTSIZE, labelpad=6
    )
    _apply_qc_typography(ax2, r"Goodness of Fit ($R^2$) for H–H Regression")
    _qc_grid(ax2, y_only=True)

    handles: List = [
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
    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=2, y=0.1)

    out_path = os.path.join(output_dir, "hh_slope_and_r2_diagnostics.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path


def plot_pka_precision(results_df: pd.DataFrame, output_dir: str = "output/qc") -> str:
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
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax.set_ylabel(r"Apparent $pK_a$", fontsize=_QC_LABEL_FONTSIZE, labelpad=6)
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
        for n in _NACL_LEVELS
    ]
    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=3, y=0.15)

    out_path = os.path.join(output_dir, "pka_precision_by_nacl.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path


def plot_buffer_region_coverage(
    results: List[Dict], output_dir: str = "output/qc"
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

    concentrations = [c for c in _NACL_LEVELS if c in data_by_nacl]
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
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax.set_ylabel(
        "Number of points in buffer region", fontsize=_QC_LABEL_FONTSIZE, labelpad=6
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
    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=2, y=0.15)

    out_path = os.path.join(output_dir, "buffer_region_coverage.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path


def plot_residuals_analysis(results: List[Dict], output_dir: str = "output/qc") -> str:
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 6.0))
    _reserve_bottom(fig, bottom=0.36)

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
    ax1.set_xlabel(
        r"$\log_{10}\!\left(\dfrac{[A^-]}{[HA]}\right)$",
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax1.set_ylabel(
        r"Residual ($\mathrm{pH}_{obs} - \mathrm{pH}_{fit}$)",
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    _apply_qc_typography(ax1, "Residuals vs. Buffer Ratio")
    _qc_grid(ax1, y_only=False)

    ax2.hist(all_residuals, bins=25, color="0.80", edgecolor="black", linewidth=0.8)
    ax2.axvline(0, color="0.15", linestyle="--", linewidth=1.2)
    ax2.set_xlabel(
        r"Residual ($\mathrm{pH}_{obs} - \mathrm{pH}_{fit}$)",
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax2.set_ylabel("Frequency", fontsize=_QC_LABEL_FONTSIZE, labelpad=6)
    _apply_qc_typography(ax2, "Residuals Distribution")
    _qc_grid(ax2, y_only=True)

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
        for n in _NACL_LEVELS
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
    _legend_below(fig, handles, [h.get_label() for h in handles], ncol=3, y=0.15)

    out_path = os.path.join(output_dir, "hh_residuals_analysis.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path


def plot_half_equivalence_check(
    results_df: pd.DataFrame, output_dir: str = "output/qc"
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
        fontsize=_QC_LABEL_FONTSIZE,
        labelpad=6,
    )
    ax.set_ylabel(
        r"$V_{\mathrm{half}} / V_{\mathrm{eq}}$ (dimensionless)",
        fontsize=_QC_LABEL_FONTSIZE,
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
    for nacl in _NACL_LEVELS:
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

    _legend_below(fig, handles, labels, ncol=3, y=0.1)

    out_path = os.path.join(output_dir, "half_equivalence_verification.png")
    save_figure_bundle(fig, out_path)
    plt.close(fig)
    return out_path


def generate_all_qc_plots(
    results: List[Dict],
    results_df: pd.DataFrame,
    output_dir: str = "output/qc",
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
