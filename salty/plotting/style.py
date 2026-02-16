"""Centralized plotting style, labels, legends, and save helpers."""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, ScalarFormatter

OUTPUT_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")
FIGURE_DPI = 300
FIGURE_ROOT = Path("output") / "figures"
_STYLE_STATE = {"initialized": False}

_KIND_TO_FOLDER = {
    "main_results": "summary",
    "methods": "methods_or_derivations",
    "qc": "qc",
    "summary": "summary",
    "diagnostics": "diagnostics",
    "individual": "titration",
    "supplemental": "summary",
}

QQ_MIN_N = 20


@dataclass(frozen=True)
class StyleConfig:
    BASE_FONTSIZE: float = 12.0
    TITLE_FONTSIZE: float = 14.0
    LABEL_FONTSIZE: float = 12.0
    TICK_FONTSIZE: float = 11.0
    LEGEND_FONTSIZE: float = 11.0
    PANEL_FONTSIZE: float = 13.0
    ANNOTATION_FONTSIZE: float = 11.0
    LINEWIDTH: float = 2.0
    LINEWIDTH_THIN: float = 1.2
    LINEWIDTH_THICK: float = 2.5
    MARKERSIZE: float = 6.0
    ALPHA_REPLICATE: float = 0.55
    ALPHA_BAND: float = 0.12
    GRID_ALPHA: float = 0.20
    FIGSIZE_SINGLE: tuple[float, float] = (7.0, 4.2)
    FIGSIZE_WIDE: tuple[float, float] = (9.5, 4.2)
    FIGSIZE_2x2: tuple[float, float] = (9.5, 7.2)


STYLE = StyleConfig()

FIGSIZE_SINGLE = STYLE.FIGSIZE_SINGLE
FIGSIZE_WIDE = STYLE.FIGSIZE_WIDE
FIGSIZE_GRID_2x2 = STYLE.FIGSIZE_2x2
DEFAULT_WSPACE = 0.18
DEFAULT_HSPACE = 0.24
LEGEND_TOP_PAD = 0.14
LEGEND_BOTTOM_PAD = 0.16

FIG_SIZES: dict[str, tuple[float, float]] = {
    "single": STYLE.FIGSIZE_SINGLE,
    "wide": STYLE.FIGSIZE_WIDE,
    "tall": (7.0, 5.8),
    "panel_2x2": STYLE.FIGSIZE_2x2,
    "grid_2x2": STYLE.FIGSIZE_2x2,
    "panel_2x3": (13.0, 7.2),
    "grid_2x3": (13.0, 7.2),
    "qc_wide": (12.0, 4.8),
    "diagnostic": (12.0, 4.8),
}

FONT_SIZES = {
    "base": STYLE.BASE_FONTSIZE,
    "title": STYLE.TITLE_FONTSIZE,
    "axis_label": STYLE.LABEL_FONTSIZE,
    "tick": STYLE.TICK_FONTSIZE,
    "legend": STYLE.LEGEND_FONTSIZE,
    "panel": STYLE.PANEL_FONTSIZE,
    "annotation": STYLE.ANNOTATION_FONTSIZE,
}

LINE_WIDTHS = {
    "replicate": STYLE.LINEWIDTH_THIN,
    "mean": STYLE.LINEWIDTH,
    "guide": STYLE.LINEWIDTH_THIN,
    "band_edge": 0.0,
}

MARKER_SIZES = {
    "replicate": 26,
    "mean": 54,
    "diagnostic": 26,
}

ALPHAS = {
    "replicate": STYLE.ALPHA_REPLICATE,
    "ci_band": STYLE.ALPHA_BAND,
    "sd_band": 0.08,
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
    "pka_app_from_vhalf": r"$pK_{a,\mathrm{app}}\ \approx\ \mathrm{pH}(V_{1/2})$",
    "dph_dv": r"$\Delta \mathrm{pH}/\Delta V$",
    "x_volume": r"$V_{\mathrm{NaOH}}\ \mathrm{added}\ /\ \mathrm{cm^3}$",
    "x_nacl": r"$[\mathrm{NaCl}]\ /\ (\mathrm{mol\,dm^{-3}})$",
    "x_ionic": r"$I\ /\ (\mathrm{mol\,dm^{-3}})$",
    "y_derivative": r"$\Delta \mathrm{pH}/\Delta V\ /\ \mathrm{cm^{-3}}$",
    "hh_x": r"$\log_{10}\!\left(\frac{V}{V_{\mathrm{eq}}-V}\right)$",
    "ph": r"$\mathrm{pH}$",
    "residual_ph": r"$\mathrm{Residual}\ /\ \mathrm{pH}$",
    "temperature_c": r"$T\ /\ ^\circ\mathrm{C}$",
    "calibration_ph": r"$\mathrm{Calibration\ buffer\ reading}\ /\ \mathrm{pH}$",
    "residuals_title": r"$\mathrm{Residuals}\ (\mathrm{buffer\ region\ only})$",
    "sd_band": r"$\pm 1\ \mathrm{SD}$",
}

UNIT_MOLAR = r"\mathrm{mol\,dm^{-3}}"
UNIT_VOL = r"\mathrm{cm^{3}}"
UNIT_PH = r"\mathrm{pH}"

LABEL_NACL = rf"[\mathrm{{NaCl}}]\;/\;{UNIT_MOLAR}"
LABEL_I = rf"I\;/\;{UNIT_MOLAR}"
LABEL_V_NAOH = rf"V_{{\mathrm{{NaOH}}}}\;\mathrm{{added}}\;/\;{UNIT_VOL}"
LABEL_DPH_DV = r"\Delta pH/\Delta V\;/\;\mathrm{pH\,cm^{-3}}"
LABEL_RESIDUAL = rf"\mathrm{{Residual}}\;/\;{UNIT_PH}"
LABEL_LOG_RATIO = r"\log_{10}\!\left(\dfrac{[A^-]}{[HA]}\right)"
LABEL_HH_SLOPE = r"m_{\mathrm{HH}}\;/\;\mathrm{dimensionless}"
LABEL_R2 = r"R^2\;/\;\mathrm{dimensionless}"


def label_nacl() -> str:
    """Return standardized x-label for sodium chloride concentration."""
    return LABEL_NACL


def label_I() -> str:
    """Return standardized x-label for ionic strength."""
    return LABEL_I


def label_v_naoh() -> str:
    """Return standardized x-label for NaOH volume added."""
    return LABEL_V_NAOH


def label_dph_dv() -> str:
    """Return standardized y-label for discrete derivative signal."""
    return LABEL_DPH_DV


def label_residual() -> str:
    """Return standardized y-label for model residual values."""
    return LABEL_RESIDUAL


def apply_global_style(font_scale: float = 1.0, context: str = "paper") -> None:
    """Apply global Matplotlib style once, scaled by context and font scale."""
    ctx_scale = {
        "paper": 1.0,
        "notebook": 1.05,
        "slides": 1.12,
        "talk": 1.12,
        "poster": 1.22,
    }
    scale = float(font_scale) * ctx_scale.get(context, 1.0)
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": STYLE.BASE_FONTSIZE * scale,
            "axes.titlesize": STYLE.TITLE_FONTSIZE * scale,
            "figure.titlesize": STYLE.TITLE_FONTSIZE * scale,
            "axes.labelsize": STYLE.LABEL_FONTSIZE * scale,
            "xtick.labelsize": STYLE.TICK_FONTSIZE * scale,
            "ytick.labelsize": STYLE.TICK_FONTSIZE * scale,
            "legend.fontsize": STYLE.LEGEND_FONTSIZE * scale,
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "axes.titlepad": 8,
            "axes.labelpad": 6,
            "axes.linewidth": STYLE.LINEWIDTH_THIN,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "grid.alpha": STYLE.GRID_ALPHA,
            "grid.linestyle": ":",
            "grid.linewidth": 0.7,
            "axes.grid": False,
            "legend.frameon": False,
            "legend.handlelength": 2.2,
            "legend.handletextpad": 0.6,
            "legend.borderaxespad": 0.6,
            "lines.linewidth": STYLE.LINEWIDTH,
            "lines.markersize": STYLE.MARKERSIZE,
            "errorbar.capsize": 3.0,
            "figure.constrained_layout.use": False,
            "figure.dpi": 120,
            "savefig.dpi": FIGURE_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.12,
        }
    )


def apply_style(font_scale: float = 1.0, context: str = "paper") -> None:
    """Backward-compatible alias for apply_global_style."""
    apply_global_style(font_scale=font_scale, context=context)


def apply_publication_style() -> None:
    """Backward-compatible alias for historical style entrypoint."""
    apply_global_style(font_scale=1.0, context="paper")


def set_global_style() -> None:
    """Apply global plotting style once per process."""
    if not _STYLE_STATE["initialized"]:
        apply_global_style(font_scale=1.0, context="paper")
        _STYLE_STATE["initialized"] = True


def apply_rcparams() -> None:
    """Alias for unified plot-style initialization."""
    set_global_style()


def _round_nacl(nacl: float) -> float:
    return float(np.round(float(nacl), 1))


def color_for_nacl(nacl: float) -> str:
    """Return a stable color for one NaCl concentration."""
    return NACL_COLOR_MAP.get(_round_nacl(nacl), "#4A4A4A")


def fig_size(kind: str = "single") -> tuple[float, float]:
    """Return standardized figure size tuple for a named figure kind."""
    aliases = {
        "panel": "single",
        "grid": "panel_2x2",
        "panel_2x2": "grid_2x2",
        "panel_2x3": "grid_2x3",
    }
    canonical = aliases.get(kind, kind)
    return FIG_SIZES.get(canonical, FIG_SIZES["single"])


def marker_for_run(run_index: int) -> str:
    """Return a marker code for a replicate index."""
    return RUN_MARKERS[int(run_index) % len(RUN_MARKERS)]


def panel_tag(index: int) -> str:
    """Return panel label text as (a), (b), ..."""
    return f"({chr(ord('a') + int(index))})"


def add_panel_label(
    ax: Axes,
    label: str,
    x: float = 0.02,
    y: float = 0.98,
    *,
    loc: str = "upper left",
    pad: float = 0.02,
    dx: float = 0.0,
    dy: float = 0.0,
    fontsize: float | None = None,
    bbox: bool = False,
) -> None:
    """Render a panel label in axes coordinates with collision-safe offsets."""
    _ = bbox
    anchor = {
        "upper left": ("left", "top", 0.02, 0.98),
        "upper right": ("right", "top", 0.98, 0.98),
        "lower left": ("left", "bottom", 0.02, 0.02),
        "lower right": ("right", "bottom", 0.98, 0.02),
    }
    ha, va, x0, y0 = anchor.get(loc, anchor["upper left"])
    if loc == "upper left":
        x0, y0 = pad, 1.0 - pad
    elif loc == "upper right":
        x0, y0 = 1.0 - pad, 1.0 - pad
    elif loc == "lower left":
        x0, y0 = pad, pad
    elif loc == "lower right":
        x0, y0 = 1.0 - pad, pad
    x_use = x if x != 0.02 else x0
    y_use = y if y != 0.98 else y0
    ax.text(
        x_use + dx,
        y_use + dy,
        label,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize or FONT_SIZES["panel"],
        fontweight="bold",
        color="0.20",
    )


def add_info_box(
    ax: Axes,
    text: str,
    loc: str = "upper left",
    pad: float = 0.25,
    fontsize: float = FONT_SIZES["annotation"],
    *,
    bbox: bool = False,
) -> None:
    """Add a consistently styled annotation box anchored to one corner."""
    _ = (pad, bbox)
    anchor_map = {
        "upper left": (0.02, 0.90, "left", "top"),
        "upper right": (0.98, 0.90, "right", "top"),
        "lower left": (0.02, 0.08, "left", "bottom"),
        "lower right": (0.98, 0.08, "right", "bottom"),
    }
    x, y, ha, va = anchor_map.get(loc, anchor_map["upper left"])
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color="0.35",
    )


def annotate_offset(
    ax: Axes,
    text: str,
    xy: tuple[float, float],
    dx: float = 10,
    dy: float = 10,
    **kwargs,
):
    """Annotate text with offset points so labels avoid overlapping data."""
    defaults = {
        "xycoords": "data",
        "textcoords": "offset points",
        "ha": "left",
        "va": "bottom",
        "fontsize": FONT_SIZES["annotation"],
        "color": "0.25",
    }
    defaults.update(kwargs)
    return ax.annotate(text, xy=xy, xytext=(dx, dy), **defaults)


def place_label(
    ax: Axes,
    text: str,
    xy_data: tuple[float, float],
    *,
    dx_pts: float = 0,
    dy_pts: float = 0,
    ha: str = "left",
    va: str = "bottom",
    fontsize: float = FONT_SIZES["annotation"],
    color: str = "0.25",
    **kwargs,
):
    """Place an annotation in data coordinates with point offsets."""
    return ax.annotate(
        text,
        xy=xy_data,
        xytext=(dx_pts, dy_pts),
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        **kwargs,
    )


def panel_label(
    ax: Axes,
    label: str = "(a)",
    *,
    dx_pts: float = 0,
    dy_pts: float = 0,
    bbox: bool = False,
) -> None:
    """Add a panel label at top-left with optional point offsets."""
    add_panel_label(
        ax,
        label=label,
        x=0.02 + dx_pts / 500.0,
        y=0.98 + dy_pts / 500.0,
        **{"bbox": bbox},
    )


def clean_axis(
    ax: Axes,
    *,
    grid_axis: str = "y",
    nbins_x: int = 6,
    nbins_y: int = 6,
) -> None:
    """Apply consistent ticks, grid, and spine/tick formatting to one axis."""
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick"], width=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbins_x, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nbins_y, min_n_ticks=4))
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(STYLE.LINEWIDTH_THIN)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if grid_axis == "both":
        ax.grid(True, axis="both", alpha=STYLE.GRID_ALPHA, linestyle=":", linewidth=0.7)
    elif grid_axis in {"x", "y"}:
        ax.grid(
            True, axis=grid_axis, alpha=STYLE.GRID_ALPHA, linestyle=":", linewidth=0.7
        )


def set_axis_labels(ax: Axes, x: str | None = None, y: str | None = None) -> None:
    """Apply standardized axis labels with project typography."""
    if x is not None:
        ax.set_xlabel(x, fontsize=FONT_SIZES["axis_label"], labelpad=6)
    if y is not None:
        ax.set_ylabel(y, fontsize=FONT_SIZES["axis_label"], labelpad=6)


def set_naCl_axis(ax: Axes) -> None:
    """Apply standardized NaCl x-axis label and canonical tick locations."""
    ax.set_xlim(-0.05, 0.85)
    ax.set_xticks(NACL_LEVELS)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.set_xlabel(
        r"$[\mathrm{NaCl}]\ /\ \mathrm{mol\,dm^{-3}}$",
        fontsize=FONT_SIZES["axis_label"],
        labelpad=6,
    )


def format_axis(
    ax: Axes,
    xlabel: str,
    ylabel: str,
    x_unit: str | None = None,
    y_unit: str | None = None,
    x_ticks: Sequence[float] | None = None,
    y_ticks: Sequence[float] | None = None,
) -> None:
    """Apply standardized labels, units, and optional explicit major ticks."""
    x_label = xlabel if x_unit is None else f"{xlabel} / {x_unit}"
    y_label = ylabel if y_unit is None else f"{ylabel} / {y_unit}"
    set_axis_labels(ax, x=x_label, y=y_label)
    if x_ticks is not None:
        ax.set_xticks(list(x_ticks))
    if y_ticks is not None:
        ax.set_yticks(list(y_ticks))


def set_axes_style(
    ax: Axes,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    xticks: Sequence[float] | None = None,
    yticks: Sequence[float] | None = None,
    xfmt: str | None = None,
    yfmt: str | None = None,
) -> None:
    """Apply standardized labels, limits, ticks, and readable tick formatting."""
    if xlabel is not None or ylabel is not None:
        set_axis_labels(ax, x=xlabel, y=ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xticks is not None:
        ax.set_xticks(list(xticks))
    if yticks is not None:
        ax.set_yticks(list(yticks))
    if xfmt:
        ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))
    if yfmt:
        ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZES["tick"], width=1.0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))


def set_ticks(
    ax: Axes,
    xstep: float | None = None,
    ystep: float | None = None,
    *,
    x: Mapping[str, float | int | str | Sequence[float]] | None = None,
    y: Mapping[str, float | int | str | Sequence[float]] | None = None,
    xfmt: str | None = None,
    yfmt: str | None = None,
    xnbins: int = 5,
    ynbins: int = 5,
) -> None:
    """Apply deterministic major ticks/formatting for publication plots."""

    def _apply_policy(
        axis: str, policy: Mapping[str, float | int | str | Sequence[float]]
    ) -> None:
        locator_axis = ax.xaxis if axis == "x" else ax.yaxis
        set_ticks_axis = ax.set_xticks if axis == "x" else ax.set_yticks
        set_formatter = (
            ax.xaxis.set_major_formatter
            if axis == "x"
            else ax.yaxis.set_major_formatter
        )

        if "ticks" in policy:
            ticks = policy.get("ticks")
            if isinstance(ticks, Sequence):
                set_ticks_axis(list(ticks))
                if "format" in policy and isinstance(policy["format"], str):
                    set_formatter(FormatStrFormatter(str(policy["format"])))
                return

        if "step" in policy:
            step = float(policy["step"])
            if np.isfinite(step) and step > 0:
                start, end = ax.get_xlim() if axis == "x" else ax.get_ylim()
                ticks = np.arange(
                    np.floor(start / step) * step,
                    end + step * 0.5,
                    step,
                )
                set_ticks_axis(ticks)

        nbins = int(policy.get("nbins", 5))
        min_n_ticks = int(policy.get("min_n_ticks", 4))
        locator_axis.set_major_locator(
            MaxNLocator(nbins=nbins, min_n_ticks=min_n_ticks)
        )

        if "format" in policy and isinstance(policy["format"], str):
            set_formatter(FormatStrFormatter(str(policy["format"])))

    if x is not None:
        _apply_policy("x", x)
        if xfmt:
            ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))
    else:
        if xstep is not None and np.isfinite(float(xstep)) and float(xstep) > 0:
            start, end = ax.get_xlim()
            ticks = np.arange(np.floor(start / xstep) * xstep, end + xstep * 0.5, xstep)
            ax.set_xticks(ticks)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=int(xnbins), min_n_ticks=4))

        if xfmt:
            ax.xaxis.set_major_formatter(FormatStrFormatter(xfmt))

    if y is not None:
        _apply_policy("y", y)
        if yfmt:
            ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))
    else:
        if ystep is not None and np.isfinite(float(ystep)) and float(ystep) > 0:
            start, end = ax.get_ylim()
            ticks = np.arange(np.floor(start / ystep) * ystep, end + ystep * 0.5, ystep)
            ax.set_yticks(ticks)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=int(ynbins), min_n_ticks=4))

        if yfmt:
            ax.yaxis.set_major_formatter(FormatStrFormatter(yfmt))


def safe_set_lims(
    ax: Axes,
    x: tuple[float, float] | None = None,
    y: tuple[float, float] | None = None,
    *,
    pad_frac: float = 0.05,
) -> None:
    """Set axis limits with small symmetric padding to avoid clipping."""
    if x is not None:
        x0, x1 = x
        if np.isfinite(x0) and np.isfinite(x1):
            span = max(float(x1 - x0), 1e-9)
            pad = span * pad_frac
            ax.set_xlim(float(x0 - pad), float(x1 + pad))
    if y is not None:
        y0, y1 = y
        if np.isfinite(y0) and np.isfinite(y1):
            span = max(float(y1 - y0), 1e-9)
            pad = span * pad_frac
            ax.set_ylim(float(y0 - pad), float(y1 + pad))


def collect_legend_items(axes: Sequence[Axes]) -> tuple[list, list[str]]:
    """Collect and dedupe legend handles/labels from axes in order."""
    seen: set[str] = set()
    handles: list = []
    labels: list[str] = []
    for ax in axes:
        axis_handles, axis_labels = ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if not label or label.startswith("_"):
                continue
            if label in seen:
                continue
            seen.add(label)
            handles.append(handle)
            labels.append(label)
    return handles, labels


def place_figure_legend(
    fig: Figure,
    handles: Sequence,
    labels: Sequence[str],
    *,
    where: str = "top",
    ncol: int | str = "auto",
) -> None:
    """Add a figure-level legend at standardized top/bottom center anchors."""
    if not handles:
        return
    if ncol == "auto":
        ncol_use = min(max(1, int(len(handles))), 5)
    else:
        ncol_use = max(1, int(ncol))
    if where == "bottom":
        loc = "lower center"
        anchor = (0.5, 0.01)
    else:
        loc = "upper center"
        anchor = (0.5, 0.95)
    fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=ncol_use,
        frameon=False,
        fontsize=FONT_SIZES["legend"],
    )


def figure_legend(
    fig: Figure,
    handles: Sequence,
    labels: Sequence[str],
    *,
    loc: str = "upper center",
    ncol: int | None = None,
    bbox_to_anchor: tuple[float, float] = (0.5, 0.95),
    frameon: bool = False,
) -> None:
    """Single standardized figure-level legend entrypoint."""
    if not handles:
        return
    ncol_use = ncol if ncol is not None else min(max(1, int(len(handles))), 5)
    fig.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=max(1, int(ncol_use)),
        frameon=frameon,
        fontsize=FONT_SIZES["legend"],
        handlelength=2.6,
        columnspacing=1.25,
        handletextpad=0.65,
    )


def place_fig_legend(
    fig: Figure,
    handles: Sequence,
    labels: Sequence[str],
    *,
    where: str = "top",
    ncol: int | str = "auto",
) -> None:
    """Backward-compatible alias for place_figure_legend."""
    place_figure_legend(fig, handles, labels, where=where, ncol=ncol)


def should_plot_qq(n: int, min_n: int = 20) -> bool:
    """Return whether Q-Q plot is meaningful for sample size n."""
    floor = max(int(min_n), int(QQ_MIN_N))
    return int(n) >= floor


def should_plot_distribution(n: int, min_n: int = 20) -> bool:
    """Return whether histogram-style distribution diagnostics are meaningful."""
    floor = max(int(min_n), int(QQ_MIN_N))
    return int(n) >= floor


def should_plot(kind: str, n: int, min_n: int = 20) -> bool:
    """Return whether a diagnostic kind should be rendered for sample size n."""
    kind_norm = str(kind).strip().lower()
    if kind_norm in {"qq", "q-q", "q_q"}:
        return should_plot_qq(n=n, min_n=min_n)
    if kind_norm in {"distribution", "hist", "histogram", "density"}:
        return should_plot_distribution(n=n, min_n=min_n)
    return True


def fallback_note(kind: str, n: int, min_n: int = 20) -> str:
    """Return standardized explanatory note when a diagnostic is omitted."""
    required = max(int(min_n), int(QQ_MIN_N))
    return (
        f"{kind} omitted (n={int(n)} < {required}); "
        "insufficient sample size for a reliable shape diagnostic."
    )


def safe_annotate(
    ax: Axes,
    text: str,
    xy: tuple[float, float],
    *,
    xytext: tuple[float, float] = (10, 10),
    textcoords: str = "offset points",
    ha: str = "left",
    va: str = "center",
    arrow: bool = False,
    fontsize: float = FONT_SIZES["annotation"],
    color: str = "0.25",
):
    """Annotate with offset text and optional arrow without any text box."""
    arrowprops = {"arrowstyle": "->", "lw": 0.9, "color": color} if arrow else None
    return ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords=textcoords,
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        arrowprops=arrowprops,
    )


def set_sensible_ticks(
    ax: plt.Axes,
    x: int | None = 5,
    y: int | None = 5,
) -> None:
    """Apply readable major ticks and scalar formatting."""
    if x is not None:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=int(x), min_n_ticks=4))
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_scientific(False)
        ax.xaxis.set_major_formatter(xfmt)
    if y is not None:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=int(y), min_n_ticks=4))
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(False)
        ax.yaxis.set_major_formatter(yfmt)


def new_figure(*args, **kwargs):
    """Create subplots with constrained layout and consistent padding."""
    kwargs = dict(kwargs)
    set_global_style()
    fig, axes = plt.subplots(*args, **kwargs)
    apply_subplot_padding(fig)
    return fig, axes


def apply_subplot_padding(
    fig: Figure,
    wspace: float = DEFAULT_WSPACE,
    hspace: float = DEFAULT_HSPACE,
    w_pad: float = 0.02,
    h_pad: float = 0.02,
) -> None:
    """Apply consistent inter-subplot spacing on standard tight-layout figures."""
    left = 0.08 + min(max(w_pad, 0.0), 0.12)
    right = 0.98 - min(max(w_pad, 0.0), 0.08)
    bottom = 0.10 + min(max(h_pad, 0.0), 0.10)
    top = 0.92 - min(max(h_pad, 0.0), 0.12)
    engine = fig.get_layout_engine()
    if engine is not None and hasattr(engine, "set"):
        engine.set(rect=(left, bottom, right, top), wspace=wspace, hspace=hspace)
    else:
        fig.subplots_adjust(
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=wspace,
            hspace=hspace,
        )


def fmt_nacl(nacl: float) -> str:
    """Return standardized mathtext label for one NaCl concentration."""
    nacl_val = _round_nacl(nacl)
    return rf"$[\mathrm{{NaCl}}]={nacl_val:.1f}\ \mathrm{{mol\,dm^{{-3}}}}$"


def draw_equivalence_guides(
    ax: Axes,
    veq_mean: float,
    veq_sd: float,
    vhalf_mean: float,
    vhalf_sd: float,
    color: str,
    show_veq_band: bool = True,
    show_vhalf_band: bool = True,
) -> None:
    """Draw V_eq and V_{1/2} guides with ±1 SD bands."""
    if np.isfinite(veq_mean):
        if show_veq_band and np.isfinite(veq_sd) and veq_sd > 0:
            ax.axvspan(
                veq_mean - veq_sd,
                veq_mean + veq_sd,
                color=color,
                alpha=ALPHAS["sd_band"],
                linewidth=LINE_WIDTHS["band_edge"],
            )
        ax.axvline(veq_mean, color=color, linewidth=LINE_WIDTHS["guide"], linestyle="-")

    if np.isfinite(vhalf_mean):
        if show_vhalf_band and np.isfinite(vhalf_sd) and vhalf_sd > 0:
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


def save_figure_bundle(fig: Figure, png_path: str) -> str:
    """Save synchronized PNG, PDF, and SVG files for a figure."""
    base = Path(os.path.splitext(png_path)[0])
    save_figure_all_formats(fig, base)
    return str(base.with_suffix(".png"))


def save_figure(
    fig: Figure,
    savepath_base: str | Path,
    formats: Sequence[str] = OUTPUT_FORMATS,
    dpi: int = FIGURE_DPI,
    *,
    bbox_inches: str = "tight",
    pad_inches: float = 0.12,
) -> Path:
    """Save a figure to multiple formats using one extensionless base path."""
    base = Path(savepath_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        target = base.with_suffix(f".{ext}")
        fig.savefig(
            str(target),
            dpi=dpi if ext == "png" else None,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
    return base.with_suffix(".png")


def save_figure_all_formats(
    fig: Figure,
    path_base: Path,
    *,
    bbox_inches: str = "tight",
    pad_inches: float = 0.12,
) -> Path:
    """Backward-compatible wrapper over save_figure."""
    return save_figure(
        fig,
        savepath_base=path_base,
        formats=OUTPUT_FORMATS,
        dpi=FIGURE_DPI,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )


def finalize_figure(
    fig: Figure,
    *,
    title: str | None = None,
    suptitle: str | None = None,
    subtitle: str | None = None,
    legend: dict | None = None,
    pad: float = 0.12,
    legend_height: float = 0.0,
    tight: bool = True,
    w_pad: float | None = None,
    h_pad: float | None = None,
    rect: tuple[float, float, float, float] | None = None,
    pad_inches: float | None = None,
    savepath: str | Path | None = None,
    outpath: str | Path | None = None,
    dpi: int = FIGURE_DPI,
    formats: Sequence[str] = OUTPUT_FORMATS,
) -> str | None:
    """Finalize layout/legend/title and optionally save a figure bundle."""
    if title is None and suptitle is not None:
        title = suptitle

    if title:
        fig.suptitle(title, fontsize=FONT_SIZES["title"])
    if subtitle:
        fig.text(
            0.5,
            0.99,
            subtitle,
            ha="center",
            va="top",
            fontsize=FONT_SIZES["annotation"],
        )

    legend_where = "top"
    if legend:
        legend_where = str(legend.get("where", "top"))
        place_figure_legend(
            fig,
            legend.get("handles", []),
            legend.get("labels", []),
            where=legend_where,
            ncol=legend.get("ncol", "auto"),
        )

    if rect is not None:
        rect_use = rect
    elif legend_height > 0:
        h = min(max(float(legend_height), 0.0), 0.30)
        if legend_where == "bottom":
            rect_use = (0.0, h, 1.0, 1.0)
        else:
            rect_use = (0.0, 0.0, 1.0, 1.0 - h)
    else:
        rect_use = (0.0, 0.0, 1.0, 1.0)

    if w_pad is not None or h_pad is not None:
        apply_subplot_padding(
            fig,
            w_pad=0.02 if w_pad is None else float(w_pad),
            h_pad=0.02 if h_pad is None else float(h_pad),
        )

    if tight:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.tight_layout(rect=rect_use, pad=1.2)

    save_target = savepath if savepath is not None else outpath
    if save_target is None:
        return None

    pad_use = float(pad if pad_inches is None else pad_inches)
    savebase = Path(save_target)
    if savebase.suffix:
        savebase = savebase.with_suffix("")
    png_path = save_figure(
        fig,
        savepath_base=savebase,
        formats=formats,
        dpi=dpi,
        pad_inches=pad_use,
    )
    return str(png_path)


def sanitize_filename(name: str) -> str:
    """Normalize a filename component into a stable, filesystem-safe token."""
    text = re.sub(r"\s+", "_", str(name).strip())
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "figure"


def figure_path(
    fig_key: str,
    kind: str,
    ext: str,
    *,
    iteration: str | None = None,
) -> Path:
    """Build a stable figure output path under output/figures taxonomy."""
    if ext not in OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported extension '{ext}'. Expected one of {OUTPUT_FORMATS}."
        )
    if kind not in _KIND_TO_FOLDER:
        raise ValueError(
            (
                f"Unsupported figure kind '{kind}'. "
                f"Expected one of {tuple(_KIND_TO_FOLDER)}."
            )
        )

    folder = FIGURE_ROOT / _KIND_TO_FOLDER[kind]
    if iteration:
        folder = folder / "iterations" / sanitize_filename(iteration)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{sanitize_filename(fig_key)}.{ext}"


def figure_base_path(fig_key: str, kind: str, *, iteration: str | None = None) -> Path:
    """Build an extensionless output path for multi-format figure saving."""
    return figure_path(
        fig_key=fig_key, kind=kind, ext="png", iteration=iteration
    ).with_suffix("")


def save_to_multiple_dirs(fig: plt.Figure, file_stem: str, dirs: Iterable[str]) -> str:
    """Save one figure bundle to multiple directories and return primary PNG."""
    primary = ""
    for idx, directory in enumerate(dirs):
        out_png = os.path.join(directory, f"{file_stem}.png")
        saved = save_figure_bundle(fig, out_png)
        if idx == 0:
            primary = saved
    return primary


def warn_skipped_qq(n: int, min_n: int = 20) -> None:
    """Emit a standardized warning for skipped Q-Q diagnostics."""
    warnings.warn(
        (
            "Q-Q plot skipped: "
            f"n={int(n)} is below minimum n={int(min_n)} "
            "for meaningful normality assessment."
        ),
        RuntimeWarning,
        stacklevel=2,
    )
