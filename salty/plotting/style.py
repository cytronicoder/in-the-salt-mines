"""Shared plotting style and nomenclature helpers for publication figures."""

from __future__ import annotations

import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")
FIGURE_DPI = 300

FONT_SIZES = {
    "base": 12,
    "title": 14,
    "axis_label": 13,
    "tick": 11,
    "legend": 11,
    "panel": 15,
}

LINE_WIDTHS = {
    "replicate": 1.0,
    "mean": 2.0,
    "guide": 1.4,
    "band_edge": 0.0,
}

MARKER_SIZES = {
    "replicate": 26,
    "mean": 54,
    "diagnostic": 26,
}

ALPHAS = {
    "replicate": 0.60,
    "ci_band": 0.15,
    "sd_band": 0.12,
}

NACL_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
NACL_COLOR_MAP = {
    0.0: "#1f77b4",
    0.2: "#1b9e77",
    0.4: "#2ca02c",
    0.6: "#ff7f0e",
    0.8: "#9467bd",
}
RUN_MARKERS = ("o", "s", "^", "D", "v", "P", "X")

MATH_LABELS = {
    "veq": r"$V_{\mathrm{eq}}$",
    "vhalf": r"$V_{1/2}$",
    "ph_vhalf": r"$\mathrm{pH}(V_{1/2})$",
    "pka_app": r"$pK_{a,\mathrm{app}}$",
    "dph_dv": r"$\Delta \mathrm{pH}/\Delta V$",
    "x_volume": r"$V_{\mathrm{NaOH}}\ \mathrm{added}\ /\ \mathrm{cm^3}$",
    "x_nacl": r"$[\mathrm{NaCl}]\ /\ (\mathrm{mol\,dm^{-3}})$",
    "x_ionic": r"$I\ /\ (\mathrm{mol\,dm^{-3}})$",
    "y_derivative": r"$\Delta\mathrm{pH}/\Delta V\ /\ (\mathrm{pH\,cm^{-3}})$",
    "hh_x": r"$\log_{10}\!\left(\frac{V}{V_{\mathrm{eq}}-V}\right)$",
    "ph": r"$\mathrm{pH}$",
    "residual_ph": r"$\mathrm{Residual}\ /\ \mathrm{pH}$",
}


def apply_publication_style() -> None:
    """Apply a consistent Matplotlib-only style for all IA figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZES["base"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.labelsize": FONT_SIZES["axis_label"],
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "legend.fontsize": FONT_SIZES["legend"],
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.12,
            "grid.linewidth": 0.5,
            "grid.linestyle": ":",
            "lines.linewidth": LINE_WIDTHS["mean"],
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "figure.dpi": 120,
        }
    )


def apply_rcparams() -> None:
    """Alias for unified plot-style initialization."""
    apply_publication_style()


def _round_nacl(nacl: float) -> float:
    return float(np.round(float(nacl), 1))


def color_for_nacl(nacl: float) -> str:
    """Return a stable color for one NaCl concentration."""
    return NACL_COLOR_MAP.get(_round_nacl(nacl), "#4A4A4A")


def marker_for_run(run_index: int) -> str:
    """Return a marker code for a replicate index."""
    return RUN_MARKERS[int(run_index) % len(RUN_MARKERS)]


def panel_tag(index: int) -> str:
    """Return panel label text as (a), (b), ..."""
    return f"({chr(ord('a') + int(index))})"


def add_panel_label(ax: plt.Axes, label: str) -> None:
    """Render a bold panel label at the top-left of an axes."""
    ax.text(
        0.02,
        0.97,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONT_SIZES["panel"],
        fontweight="bold",
    )


def fmt_nacl(nacl: float) -> str:
    """Return standardized mathtext label for one NaCl concentration."""
    nacl_val = _round_nacl(nacl)
    return rf"$[\mathrm{{NaCl}}]={nacl_val:.1f}\ \mathrm{{mol\,dm^{{-3}}}}$"


def draw_equivalence_guides(
    ax: plt.Axes,
    veq_mean: float,
    veq_sd: float,
    vhalf_mean: float,
    vhalf_sd: float,
    color: str,
) -> None:
    """Draw V_eq and V_{1/2} guides with ±1 SD bands."""
    if np.isfinite(veq_mean):
        if np.isfinite(veq_sd) and veq_sd > 0:
            ax.axvspan(
                veq_mean - veq_sd,
                veq_mean + veq_sd,
                color=color,
                alpha=ALPHAS["sd_band"],
                linewidth=LINE_WIDTHS["band_edge"],
            )
        ax.axvline(veq_mean, color=color, linewidth=LINE_WIDTHS["guide"], linestyle="-")

    if np.isfinite(vhalf_mean):
        if np.isfinite(vhalf_sd) and vhalf_sd > 0:
            ax.axvspan(
                vhalf_mean - vhalf_sd,
                vhalf_mean + vhalf_sd,
                color=color,
                alpha=0.08,
                linewidth=LINE_WIDTHS["band_edge"],
            )
        ax.axvline(
            vhalf_mean,
            color=color,
            linewidth=LINE_WIDTHS["guide"],
            linestyle="--",
        )


def save_figure_bundle(fig: plt.Figure, png_path: str) -> str:
    """Save synchronized PNG, PDF, and SVG files for a figure."""
    base, _ = os.path.splitext(png_path)
    os.makedirs(os.path.dirname(png_path), exist_ok=True)
    for ext in OUTPUT_FORMATS:
        target = f"{base}.{ext}"
        if ext == "png":
            fig.savefig(target, dpi=FIGURE_DPI)
        else:
            fig.savefig(target)
    return f"{base}.png"


def save_to_multiple_dirs(fig: plt.Figure, file_stem: str, dirs: Iterable[str]) -> str:
    """Save one figure bundle to multiple directories and return primary PNG."""
    primary = ""
    for idx, directory in enumerate(dirs):
        out_png = os.path.join(directory, f"{file_stem}.png")
        saved = save_figure_bundle(fig, out_png)
        if idx == 0:
            primary = saved
    return primary
