"""Render run-level titration figures for method transparency and diagnostics.

Each figure combines curve shape, derivative endpoint behavior, and
buffer-region fit context for one run, supporting IA method defense.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from .style import (
    ALPHAS,
    LINE_WIDTHS,
    MARKER_SIZES,
    MATH_LABELS,
    NACL_LEVELS,
    add_panel_label,
    apply_rcparams,
    color_for_nacl,
    draw_equivalence_guides,
    fmt_nacl,
    panel_tag,
)
from .style import save_figure_bundle as _save_figure_bundle


def save_figure_bundle(fig: plt.Figure, png_path: str) -> str:
    """Save one figure as synchronized PNG, PDF, and SVG artifacts.

    Args:
        fig (matplotlib.figure.Figure): Figure object to serialize.
        png_path (str): Output path for the PNG file; PDF and SVG use the same
            basename.

    Returns:
        str: PNG output path.

    Note:
        Multi-format export preserves vector versions for publication while keeping
        PNG convenience for quick review.

    References:
        Reproducible figure-bundle export practice for scientific reporting.
    """
    return _save_figure_bundle(fig, png_path)


def setup_plot_style():
    """Apply the project plotting style for publication-ready grayscale figures.

    Returns:
        None: Update global matplotlib ``rcParams`` in-place.

    Note:
        A serif, high-contrast style is used to keep figures legible in print and
        monochrome exports.

    References:
        Matplotlib rcParams styling for publication-quality scientific graphics.
    """
    apply_rcparams()


def plot_titration_curves(
    results: List[Dict], output_dir: str = "output", show_raw_pH: bool = False
) -> List[str]:
    """Render per-run three-panel titration diagnostic figures.

    Args:
        results (list[dict]): Run-level analysis payloads from
            ``salty.analysis.analyze_titration``.
        output_dir (str, optional): Directory for figure bundles. Defaults to
            ``"output"``.
        show_raw_pH (bool, optional): If ``True``, overlay raw pH points in
            panel 1. Defaults to ``False``.

    Returns:
        list[str]: PNG paths for each generated run figure.

    Raises:
        ValueError: If ``results`` is empty.
        KeyError: If required payload keys are missing in any run dictionary.

    Note:
        Typical failure modes include broad/multiple derivative peaks, implausible
        ``V_eq`` placement near boundaries, and visibly non-linear H-H panel
        points. Such runs should be reviewed with QC outputs before trend
        interpretation. IA correspondence: these are the run-level evidence
        figures that connect raw behavior to reported pKa outputs.

    References:
        First-derivative titration analysis and Henderson-Hasselbalch regression.
    """
    if not results:
        raise ValueError("results list is empty; nothing to plot")

    required_keys = {"data", "step_data", "dense_curve", "buffer_region"}
    for idx, result in enumerate(results):
        missing = required_keys - set(result.keys())
        if missing:
            raise KeyError(
                f"Result entry {idx} missing required keys: {missing}. "
                f"Expected keys: {required_keys}"
            )

    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    out_paths: List[str] = []

    for i, res in enumerate(results):
        raw_df: pd.DataFrame = res["data"]
        step_df: pd.DataFrame = res["step_data"]
        buffer_df: pd.DataFrame = res.get("buffer_region", pd.DataFrame())
        dense_df: pd.DataFrame = res.get("dense_curve", pd.DataFrame())

        run_name = str(res.get("run_name", f"Run {i+1}"))
        x_col = res.get("x_col", "Volume (cm^3)")
        is_volume = (x_col == "Volume (cm^3)") or ("Volume" in x_col)

        x_label = (
            r"Volume of NaOH added / $\mathrm{cm^3}$" if is_volume else r"Time / min"
        )
        deriv_label = (
            r"$\mathrm{d}(\mathrm{pH})/\mathrm{d}V$ / $\mathrm{cm^{-3}}$"
            if is_volume
            else r"$\mathrm{d}(\mathrm{pH})/\mathrm{d}t$ / $\mathrm{min^{-1}}$"
        )

        fig, (ax1, ax2, ax3) = plt.subplots(
            1, 3, figsize=(17.5, 5.6), constrained_layout=True
        )
        fig.suptitle(run_name, fontweight="bold", fontsize=20)

        if x_col not in raw_df.columns or "pH" not in raw_df.columns:
            plt.close(fig)
            continue

        x_raw = pd.to_numeric(raw_df[x_col], errors="coerce").to_numpy(dtype=float)
        y_raw = pd.to_numeric(raw_df["pH"], errors="coerce").to_numpy(dtype=float)

        if show_raw_pH:
            temp_df = pd.DataFrame({"x": x_raw, "y": y_raw})
            temp_df["x"] = pd.to_numeric(temp_df["x"], errors="coerce")
            temp_df["y"] = pd.to_numeric(temp_df["y"], errors="coerce")
            temp_df = temp_df[
                temp_df["x"].notna() & temp_df["y"].notna()
            ].drop_duplicates()
            x_raw = temp_df["x"].to_numpy(dtype=float)
            y_raw = temp_df["y"].to_numpy(dtype=float)

            ax1.plot(
                x_raw,
                y_raw,
                linestyle="none",
                marker="o",
                markersize=3.5,
                markerfacecolor="white",
                markeredgecolor="black",
                alpha=0.45,
                label="Raw measurements",
                zorder=2,
            )

        if not dense_df.empty and {"Volume (cm^3)", "pH_interp"}.issubset(
            dense_df.columns
        ):
            x_smooth = pd.to_numeric(
                dense_df["Volume (cm^3)"], errors="coerce"
            ).to_numpy(dtype=float)
            y_smooth = pd.to_numeric(dense_df["pH_interp"], errors="coerce").to_numpy(
                dtype=float
            )
            mask = np.isfinite(x_smooth) & np.isfinite(y_smooth)
            x_smooth = x_smooth[mask]
            y_smooth = y_smooth[mask]
            if len(x_smooth) > 0:
                ax1.plot(
                    x_smooth,
                    y_smooth,
                    linewidth=2.2,
                    color="black",
                    label="Interpolation (precomputed)",
                    zorder=3,
                )

        veq = res.get("veq_used", np.nan)
        ph_at_veq = np.nan
        ph_at_half = np.nan

        if (
            np.isfinite(veq)
            and not dense_df.empty
            and {"Volume (cm^3)", "pH_interp"}.issubset(dense_df.columns)
        ):
            x_smooth = pd.to_numeric(
                dense_df["Volume (cm^3)"], errors="coerce"
            ).to_numpy(dtype=float)
            y_smooth = pd.to_numeric(dense_df["pH_interp"], errors="coerce").to_numpy(
                dtype=float
            )
            mask = np.isfinite(x_smooth) & np.isfinite(y_smooth)
            x_smooth = x_smooth[mask]
            y_smooth = y_smooth[mask]

            if len(x_smooth) > 0:
                idx_veq = int(np.nanargmin(np.abs(x_smooth - veq)))
                ph_at_veq = y_smooth[idx_veq]
                idx_half = int(np.nanargmin(np.abs(x_smooth - veq / 2)))
                ph_at_half = y_smooth[idx_half]

                ax1.axvline(
                    veq,
                    color="black",
                    linestyle="--",
                    linewidth=1.6,
                    label="Equivalence point",
                )
                ax1.axvline(
                    veq / 2,
                    color="black",
                    linestyle=":",
                    linewidth=1.6,
                    label="Half-equivalence point",
                )

                if np.isfinite(ph_at_veq):
                    ax1.text(
                        veq,
                        0.98,
                        f"pH {ph_at_veq:.2f}",
                        transform=ax1.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=11,
                        bbox=dict(facecolor="white", edgecolor="0.85", pad=0.2),
                    )
                if np.isfinite(ph_at_half):
                    ax1.text(
                        veq / 2,
                        0.98,
                        f"pH {ph_at_half:.2f}",
                        transform=ax1.get_xaxis_transform(),
                        ha="center",
                        va="top",
                        fontsize=11,
                        bbox=dict(facecolor="white", edgecolor="0.85", pad=0.2),
                    )

        ax1.set_title("Titration curve", fontweight="bold")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(r"$\mathrm{pH}$")
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax1.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        if is_volume:
            ax1.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        xvals = x_raw[np.isfinite(x_raw)]
        if len(xvals) > 0:
            xmin, xmax = float(np.min(xvals)), float(np.max(xvals))
            span = (xmax - xmin) if xmax != xmin else 1.0
            ax1.set_xlim(xmin - 0.03 * span, xmax + 0.03 * span)

        ax1.legend(loc="best")

        if not step_df.empty and "dpH/dx" in step_df.columns:
            x_step = (
                "Volume (cm^3)"
                if ("Volume (cm^3)" in step_df.columns and is_volume)
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
                    linewidth=2.0,
                    color="black",
                    label="First derivative",
                )

                if np.isfinite(veq):
                    ax2.axvline(
                        veq,
                        color="black",
                        linestyle="--",
                        linewidth=1.4,
                        label="Equivalence point",
                    )
                    ax2.axvline(
                        veq / 2,
                        color="black",
                        linestyle=":",
                        linewidth=1.4,
                        label="Half-equivalence point",
                    )

                    if np.isfinite(ph_at_veq):
                        ax2.text(
                            veq,
                            0.98,
                            f"pH {ph_at_veq:.2f}",
                            transform=ax2.get_xaxis_transform(),
                            ha="center",
                            va="top",
                            fontsize=11,
                            bbox=dict(facecolor="white", edgecolor="0.85", pad=0.2),
                        )
                    if np.isfinite(ph_at_half):
                        ax2.text(
                            veq / 2,
                            0.98,
                            f"pH {ph_at_half:.2f}",
                            transform=ax2.get_xaxis_transform(),
                            ha="center",
                            va="top",
                            fontsize=11,
                            bbox=dict(facecolor="white", edgecolor="0.85", pad=0.2),
                        )

                ax2.axhline(0, color="black", linewidth=1.0, alpha=0.6)

        ax2.set_title("First derivative", fontweight="bold")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(deriv_label)
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=6))

        if is_volume:
            ax2.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        if len(xvals) > 0:
            ax2.set_xlim(ax1.get_xlim())

        ax2.legend(loc="best")

        if not buffer_df.empty and {"log10_ratio", "pH_step", "pH_fit"}.issubset(
            buffer_df.columns
        ):
            xhh = pd.to_numeric(buffer_df["log10_ratio"], errors="coerce").to_numpy(
                dtype=float
            )
            yhh = pd.to_numeric(buffer_df["pH_step"], errors="coerce").to_numpy(
                dtype=float
            )
            yfit = pd.to_numeric(buffer_df["pH_fit"], errors="coerce").to_numpy(
                dtype=float
            )

            mask = np.isfinite(xhh) & np.isfinite(yhh)
            xhh = xhh[mask]
            yhh = yhh[mask]
            yfit = yfit[mask]

            ax3.scatter(
                xhh,
                yhh,
                s=40,
                edgecolor="black",
                facecolor="white",
                linewidth=1.0,
                label="Buffer region points",
            )

            if len(xhh) > 1 and np.any(np.isfinite(yfit)):
                order = np.argsort(xhh)
                ax3.plot(
                    xhh[order],
                    yfit[order],
                    color="black",
                    linewidth=1.8,
                    label="Best-fit line",
                )

                slope = res.get("slope_reg", np.nan)
                intercept = res.get("pka_app", np.nan)
                r2 = res.get("r2_reg", np.nan)

                if np.isfinite(slope) and np.isfinite(intercept):
                    label_r2 = f"$R^2 = {r2:.3f}$" if np.isfinite(r2) else ""
                    ax3.text(
                        0.02,
                        0.98,
                        rf"$\mathrm{{pH}} = {slope:.3f}\,x + {intercept:.3f}$"
                        + "\n"
                        + rf"$pK_{{a,\mathrm{{app}}}} = {intercept:.3f}$"
                        + (("\n" + label_r2) if label_r2 else ""),
                        transform=ax3.transAxes,
                        ha="left",
                        va="top",
                        fontsize=12,
                        bbox=dict(facecolor="white", edgecolor="0.85", pad=0.25),
                    )

        ax3.set_title("Henderson-Hasselbalch (apparent $pK_a$)", fontweight="bold")
        ax3.set_xlabel(r"$\log_{10}\!\left(\frac{V}{V_{eq}-V}\right)$")
        ax3.set_ylabel(r"$\mathrm{pH}$")
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax3.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax3.legend(loc="best")

        source_file = str(res.get("source_file", ""))
        source_base = os.path.splitext(source_file)[0] if source_file else ""
        combined_name = f"{run_name}_{source_base}" if source_base else run_name
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", combined_name).strip("_")

        out_path = os.path.join(output_dir, f"titration_{sanitized}.png")
        save_figure_bundle(fig, out_path)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths


def _temperature_mean_from_result(res: Dict) -> float:
    """Return mean recorded run temperature in °C if available."""
    raw_df = res.get("data", pd.DataFrame())
    if not isinstance(raw_df, pd.DataFrame) or "Temperature (°C)" not in raw_df.columns:
        return np.nan
    vals = pd.to_numeric(raw_df["Temperature (°C)"], errors="coerce").to_numpy(
        dtype=float
    )
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def _is_temperature_outlier(temp_c: float) -> bool:
    """Flag runs outside 26 ± 1 °C."""
    return bool(np.isfinite(temp_c) and (temp_c < 25.0 or temp_c > 27.0))


def _clean_step_xy(step_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract sorted, unique volume/pH arrays from step data."""
    if step_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)
    if "Volume (cm^3)" not in step_df.columns:
        return np.array([], dtype=float), np.array([], dtype=float)
    p_col = "pH_step" if "pH_step" in step_df.columns else "pH"
    if p_col not in step_df.columns:
        return np.array([], dtype=float), np.array([], dtype=float)

    vol = pd.to_numeric(step_df["Volume (cm^3)"], errors="coerce").to_numpy(dtype=float)
    ph = pd.to_numeric(step_df[p_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(vol) & np.isfinite(ph)
    vol = vol[mask]
    ph = ph[mask]
    if len(vol) < 2:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(vol)
    vol = vol[order]
    ph = ph[order]
    grouped = pd.DataFrame({"v": vol, "p": ph}).groupby("v", as_index=False).mean()
    return grouped["v"].to_numpy(dtype=float), grouped["p"].to_numpy(dtype=float)


def _linear_interp_between_brackets(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """Linearly interpolate y(xq) between the measured points bracketing xq."""
    if len(x) < 2 or not np.isfinite(xq):
        return np.nan
    if xq < x[0] or xq > x[-1]:
        return np.nan
    if np.isclose(xq, x).any():
        return float(y[int(np.argmin(np.abs(x - xq)))])

    right = int(np.searchsorted(x, xq, side="right"))
    left = right - 1
    if left < 0 or right >= len(x):
        return np.nan
    x0, x1 = float(x[left]), float(x[right])
    y0, y1 = float(y[left]), float(y[right])
    if not np.isfinite(x0) or not np.isfinite(x1) or np.isclose(x1, x0):
        return np.nan
    frac = (xq - x0) / (x1 - x0)
    return float(y0 + frac * (y1 - y0))


def _discrete_equivalence_metrics(
    step_df: pd.DataFrame,
) -> Dict[str, float | np.ndarray]:
    """Compute discrete-derivative endpoint metrics for one run."""
    vol, ph = _clean_step_xy(step_df)
    if len(vol) < 2:
        return {
            "vol": vol,
            "ph": ph,
            "mid": np.array([], dtype=float),
            "dp_dv": np.array([], dtype=float),
            "peak_idx": -1,
            "vi": np.nan,
            "vip1": np.nan,
            "veq": np.nan,
            "sigma_veq": np.nan,
            "vhalf": np.nan,
            "ph_vhalf": np.nan,
        }

    dv = np.diff(vol)
    dp = np.diff(ph)
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = dp / dv
    mid = 0.5 * (vol[:-1] + vol[1:])

    valid = np.isfinite(slope) & np.isfinite(mid) & np.isfinite(dv) & (dv > 0)
    if not np.any(valid):
        return {
            "vol": vol,
            "ph": ph,
            "mid": mid,
            "dp_dv": slope,
            "peak_idx": -1,
            "vi": np.nan,
            "vip1": np.nan,
            "veq": np.nan,
            "sigma_veq": np.nan,
            "vhalf": np.nan,
            "ph_vhalf": np.nan,
        }

    valid_idx = np.where(valid)[0]
    best_local = int(np.argmax(slope[valid]))
    peak_idx = int(valid_idx[best_local])
    vi = float(vol[peak_idx])
    vip1 = float(vol[peak_idx + 1])
    veq = 0.5 * (vi + vip1)
    sigma_veq = 0.5 * (vip1 - vi)
    vhalf = 0.5 * veq
    ph_vhalf = _linear_interp_between_brackets(vol, ph, vhalf)

    return {
        "vol": vol,
        "ph": ph,
        "mid": mid,
        "dp_dv": slope,
        "peak_idx": peak_idx,
        "vi": vi,
        "vip1": vip1,
        "veq": veq,
        "sigma_veq": sigma_veq,
        "vhalf": vhalf,
        "ph_vhalf": ph_vhalf,
    }


def _interp_curve_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Interpolate one replicate curve onto a shared grid with NaN outside support."""
    if len(x) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    yy = np.interp(grid, x, y)
    yy[(grid < x[0]) | (grid > x[-1])] = np.nan
    return yy


def _mean_sd(values: List[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan
    if len(arr) == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def plot_titration_overlays_by_nacl(
    results: List[Dict],
    output_dir: str = "output/ia",
    file_stem: str = "titration_overlays_by_nacl",
    return_figure: bool = False,
):
    """Figure 1: pH vs NaOH volume overlays by [NaCl] with endpoint guides."""
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    all_vol: List[float] = []
    all_ph: List[float] = []

    for idx, nacl in enumerate(NACL_LEVELS):
        ax = axes_flat[idx]
        add_panel_label(ax, panel_tag(idx))
        color = color_for_nacl(nacl)

        subset = [
            r
            for r in results
            if np.isfinite(float(r.get("nacl_conc", np.nan)))
            and abs(float(r.get("nacl_conc", np.nan)) - nacl) < 1e-9
        ]
        if not subset:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title(fmt_nacl(nacl))
            ax.grid(True)
            continue

        veq_vals: List[float] = []
        vhalf_vals: List[float] = []
        ph_half_vals: List[float] = []
        temp_vals: List[float] = []
        x_curves: List[np.ndarray] = []
        y_curves: List[np.ndarray] = []

        for res in subset:
            step_df = res.get("step_data", pd.DataFrame())
            metrics = _discrete_equivalence_metrics(step_df)
            x = metrics["vol"]
            y = metrics["ph"]
            if len(x) < 2:
                continue

            all_vol.extend(x.tolist())
            all_ph.extend(y.tolist())

            run_temp = _temperature_mean_from_result(res)
            temp_vals.append(run_temp)
            is_outlier = _is_temperature_outlier(run_temp)

            run_color = "0.75" if is_outlier else color
            ax.plot(
                x,
                y,
                color=run_color,
                alpha=ALPHAS["replicate"],
                linewidth=LINE_WIDTHS["replicate"],
            )

            x_curves.append(x)
            y_curves.append(y)
            veq_vals.append(float(metrics["veq"]))
            vhalf_vals.append(float(metrics["vhalf"]))
            ph_half_vals.append(float(metrics["ph_vhalf"]))

        if x_curves:
            x_min = min(float(np.min(x)) for x in x_curves)
            x_max = max(float(np.max(x)) for x in x_curves)
            grid = np.linspace(x_min, x_max, 220)
            curve_matrix = np.vstack(
                [_interp_curve_to_grid(x, y, grid) for x, y in zip(x_curves, y_curves)]
            )
            mean_curve = np.nanmean(curve_matrix, axis=0)
            ax.plot(
                grid,
                mean_curve,
                color=color,
                linewidth=LINE_WIDTHS["mean"],
                alpha=0.95,
            )

        veq_mean, veq_sd = _mean_sd(veq_vals)
        vhalf_mean, vhalf_sd = _mean_sd(vhalf_vals)
        ph_half_mean, ph_half_sd = _mean_sd(ph_half_vals)
        draw_equivalence_guides(ax, veq_mean, veq_sd, vhalf_mean, vhalf_sd, color=color)

        if np.isfinite(vhalf_mean) and np.isfinite(ph_half_mean):
            ax.errorbar(
                [vhalf_mean],
                [ph_half_mean],
                yerr=[ph_half_sd if np.isfinite(ph_half_sd) else 0.0],
                fmt="o",
                color=color,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.8,
                markersize=6.0,
                capsize=3,
                zorder=5,
            )

        t_mean, _ = _mean_sd(temp_vals)
        ax.text(
            0.98,
            0.95,
            f"n={len(veq_vals)}\nT≈{t_mean:.1f} °C"
            if np.isfinite(t_mean)
            else f"n={len(veq_vals)}\nT≈n/a",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="0.85", pad=0.2),
        )
        ax.set_title(fmt_nacl(nacl))
        ax.grid(True)

        if idx in (0, 3):
            ax.set_ylabel(MATH_LABELS["ph"])
        if idx >= 3:
            ax.set_xlabel(MATH_LABELS["x_volume"])

    legend_ax = axes_flat[5]
    add_panel_label(legend_ax, panel_tag(5))
    legend_ax.axis("off")
    legend_handles = [
        Line2D(
            [],
            [],
            color="0.35",
            linewidth=LINE_WIDTHS["replicate"],
            label="Replicate curve",
        ),
        Line2D([], [], color="0.1", linewidth=LINE_WIDTHS["mean"], label="Mean curve"),
        Line2D(
            [],
            [],
            color="0.1",
            linestyle="-",
            linewidth=LINE_WIDTHS["guide"],
            label=MATH_LABELS["veq"],
        ),
        Line2D(
            [],
            [],
            color="0.1",
            linestyle="--",
            linewidth=LINE_WIDTHS["guide"],
            label=MATH_LABELS["vhalf"],
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="0.2",
            markeredgecolor="black",
            markersize=6,
            label=rf"{MATH_LABELS['ph_vhalf']} ± 1 SD",
        ),
        Patch(facecolor="0.6", alpha=0.12, edgecolor="none", label="±1 SD band"),
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.85),
        frameon=False,
    )
    legend_ax.text(
        0.02,
        0.25,
        (
            f"Solid: {MATH_LABELS['veq']}\n"
            f"Dashed: {MATH_LABELS['vhalf']}\n"
            "Gray traces: temperature outliers"
        ),
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
    )

    if all_vol:
        xmin, xmax = float(np.min(all_vol)), float(np.max(all_vol))
        span = max(xmax - xmin, 1e-6)
        for ax in axes_flat[:5]:
            ax.set_xlim(xmin - 0.02 * span, xmax + 0.02 * span)
    if all_ph:
        ymin, ymax = float(np.min(all_ph)), float(np.max(all_ph))
        span = max(ymax - ymin, 1e-6)
        for ax in axes_flat[:5]:
            ax.set_ylim(ymin - 0.04 * span, ymax + 0.06 * span)

    fig.tight_layout()
    out_path = save_figure_bundle(fig, os.path.join(output_dir, f"{file_stem}.png"))
    if return_figure:
        return out_path, fig
    plt.close(fig)
    return out_path


def plot_derivative_equivalence_by_nacl(
    results: List[Dict],
    output_dir: str = "output/ia",
    file_stem: str = "derivative_equivalence_by_nacl",
    return_figure: bool = False,
):
    """Figure 2: discrete-derivative endpoint identification by [NaCl]."""
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.2), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    all_mid: List[float] = []
    all_deriv: List[float] = []

    for idx, nacl in enumerate(NACL_LEVELS):
        ax = axes_flat[idx]
        add_panel_label(ax, panel_tag(idx))
        color = color_for_nacl(nacl)

        subset = [
            r
            for r in results
            if np.isfinite(float(r.get("nacl_conc", np.nan)))
            and abs(float(r.get("nacl_conc", np.nan)) - nacl) < 1e-9
        ]
        if not subset:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
            )
            ax.set_title(fmt_nacl(nacl))
            ax.grid(True)
            continue

        veq_vals: List[float] = []
        vi_vals: List[float] = []
        vip1_vals: List[float] = []
        sigma_vals: List[float] = []
        mid_curves: List[np.ndarray] = []
        deriv_curves: List[np.ndarray] = []

        for res in subset:
            step_df = res.get("step_data", pd.DataFrame())
            metrics = _discrete_equivalence_metrics(step_df)
            mid = np.asarray(metrics["mid"], dtype=float)
            dp_dv = np.asarray(metrics["dp_dv"], dtype=float)
            peak_idx = int(metrics["peak_idx"])
            if len(mid) == 0:
                continue

            run_temp = _temperature_mean_from_result(res)
            is_outlier = _is_temperature_outlier(run_temp)
            run_color = "0.75" if is_outlier else color

            valid = np.isfinite(mid) & np.isfinite(dp_dv)
            ax.plot(
                mid[valid],
                dp_dv[valid],
                color=run_color,
                linewidth=LINE_WIDTHS["replicate"],
                alpha=ALPHAS["replicate"],
            )
            mid_curves.append(mid[valid])
            deriv_curves.append(dp_dv[valid])

            if 0 <= peak_idx < len(mid) and np.isfinite(dp_dv[peak_idx]):
                ax.scatter(
                    [mid[peak_idx]],
                    [dp_dv[peak_idx]],
                    s=MARKER_SIZES["diagnostic"],
                    color=run_color,
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=4,
                )

            all_mid.extend(mid[valid].tolist())
            all_deriv.extend(dp_dv[valid].tolist())

            veq_vals.append(float(metrics["veq"]))
            vi_vals.append(float(metrics["vi"]))
            vip1_vals.append(float(metrics["vip1"]))
            sigma_vals.append(float(metrics["sigma_veq"]))

        if mid_curves:
            x_min = min(float(np.min(mv)) for mv in mid_curves if len(mv))
            x_max = max(float(np.max(mv)) for mv in mid_curves if len(mv))
            grid = np.linspace(x_min, x_max, 220)
            interp_rows = [
                _interp_curve_to_grid(mv, dv, grid)
                for mv, dv in zip(mid_curves, deriv_curves)
                if len(mv) >= 2
            ]
            if interp_rows:
                curve_matrix = np.vstack(interp_rows)
                mean_curve = np.nanmean(curve_matrix, axis=0)
                ax.plot(
                    grid,
                    mean_curve,
                    color=color,
                    linewidth=LINE_WIDTHS["mean"],
                    alpha=0.95,
                    zorder=3,
                )

        veq_mean, veq_sd = _mean_sd(veq_vals)
        if np.isfinite(veq_mean):
            if np.isfinite(veq_sd) and veq_sd > 0:
                ax.axvspan(
                    veq_mean - veq_sd,
                    veq_mean + veq_sd,
                    color=color,
                    alpha=ALPHAS["sd_band"],
                )
            ax.axvline(
                veq_mean, color=color, linewidth=LINE_WIDTHS["guide"], linestyle="-"
            )

        vi_mean, _ = _mean_sd(vi_vals)
        vip1_mean, _ = _mean_sd(vip1_vals)
        sigma_mean, _ = _mean_sd(sigma_vals)

        y_min, y_max = ax.get_ylim()
        y_span = max(y_max - y_min, 1e-6)
        y0 = y_max - 0.13 * y_span
        y1 = y_max - 0.06 * y_span
        if np.isfinite(vi_mean) and np.isfinite(vip1_mean):
            ax.plot(
                [vi_mean, vi_mean, vip1_mean, vip1_mean],
                [y0, y1, y1, y0],
                color="0.25",
                linewidth=1.1,
            )

        formula_txt = (
            r"$V_{\mathrm{eq}}=\frac{V_i+V_{i+1}}{2}$"
            "\n"
            r"$\sigma(V_{\mathrm{eq}})=\frac{V_{i+1}-V_i}{2}$"
        )
        if np.isfinite(veq_mean):
            formula_txt += (
                "\n"
                + rf"$\overline{{V_{{\mathrm{{eq}}}}}}={veq_mean:.2f}\ \mathrm{{cm^3}}$"
            )
        if np.isfinite(sigma_mean):
            formula_txt += (
                "\n" + rf"$\overline{{\sigma}}={sigma_mean:.2f}\ \mathrm{{cm^3}}$"
            )
        ax.text(
            0.025,
            0.875,
            formula_txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
        )

        ax.set_title(fmt_nacl(nacl))
        ax.grid(True)
        if idx in (0, 3):
            ax.set_ylabel(MATH_LABELS["y_derivative"])
        if idx >= 3:
            ax.set_xlabel(MATH_LABELS["x_volume"])

    legend_ax = axes_flat[5]
    add_panel_label(legend_ax, panel_tag(5))
    legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D(
                [],
                [],
                color="0.45",
                linewidth=LINE_WIDTHS["replicate"],
                alpha=ALPHAS["replicate"],
                label="Replicate line",
            ),
            Line2D(
                [],
                [],
                color="0.1",
                linewidth=LINE_WIDTHS["mean"],
                label="Condition mean line",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color="0.2",
                label="Max derivative point",
            ),
            Line2D(
                [],
                [],
                color="0.2",
                linestyle="-",
                linewidth=LINE_WIDTHS["guide"],
                label=rf"Mean {MATH_LABELS['veq']}",
            ),
            Patch(
                facecolor="0.6",
                alpha=ALPHAS["sd_band"],
                edgecolor="none",
                label="±1 SD band",
            ),
            Line2D(
                [],
                [],
                color="0.25",
                linewidth=1.1,
                label=r"Steepest interval $[V_i, V_{i+1}]$",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.85),
        frameon=False,
    )

    for ax in axes_flat[:5]:
        ax.set_xlim(10.0, 30.0)
    if all_deriv:
        ymin, ymax = float(np.min(all_deriv)), float(np.max(all_deriv))
        span = max(ymax - ymin, 1e-6)
        for ax in axes_flat[:5]:
            ax.set_ylim(ymin - 0.08 * span, ymax + 0.10 * span)

    fig.tight_layout()
    out_path = save_figure_bundle(fig, os.path.join(output_dir, f"{file_stem}.png"))
    if return_figure:
        return out_path, fig
    plt.close(fig)
    return out_path
