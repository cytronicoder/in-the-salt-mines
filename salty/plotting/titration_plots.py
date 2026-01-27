"""
Creates titration curve plots from precomputed analysis results.

Plotting functions accept fully computed results and do not recompute chemistry.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator


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


def plot_titration_curves(
    results: List[Dict], output_dir: str = "output", show_raw_pH: bool = False
) -> List[str]:
    """
    For each run, saves ONE black-and-white figure with 3 panels:
    (1) pH vs Volume (with xerr and yerr) + precomputed interpolation curve
    (2) First derivative vs Volume
    (3) Henderson-Hasselbalch diagnostic (scatter + precomputed best-fit)

    Returns list of PNG paths.
    """
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    out_paths: List[str] = []

    bar_alpha = 0.50
    ecolor_bar = (0, 0, 0, bar_alpha)

    for i, res in enumerate(results):
        raw_df: pd.DataFrame = res["data"]
        step_df: pd.DataFrame = res["step_data"]
        buffer_df: pd.DataFrame = res.get("buffer_region", pd.DataFrame())
        dense_df: pd.DataFrame = res.get("dense_curve", pd.DataFrame())

        run_name = str(res.get("run_name", f"Run {i+1}"))
        x_col = res.get("x_col", "Volume (cm³)")
        is_volume = (x_col == "Volume (cm³)") or ("Volume" in x_col)

        x_label = r"Volume of NaOH added / cm$^3$" if is_volume else r"Time / min"
        deriv_label = (
            r"$\mathrm{d\,pH}/\mathrm{d}V$ / (pH cm$^{-3}$)"
            if is_volume
            else r"$\mathrm{d\,pH}/\mathrm{d}t$ / (pH min$^{-1}$)"
        )

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6.8))
        fig.suptitle(run_name, fontweight="bold", fontsize=32, y=0.90)

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

        if show_raw_pH:
            ax1.errorbar(
                x_raw,
                y_raw,
                fmt="o",
                markersize=6,
                markerfacecolor="white",
                markeredgecolor="black",
                ecolor=ecolor_bar,
                elinewidth=1.6,
                capsize=4,
                alpha=0.05,
                label="Measurements",
            )

        if not dense_df.empty and {"Volume (cm³)", "pH_interp"}.issubset(
            dense_df.columns
        ):
            x_smooth = pd.to_numeric(
                dense_df["Volume (cm³)"], errors="coerce"
            ).to_numpy(dtype=float)
            y_smooth = pd.to_numeric(
                dense_df["pH_interp"], errors="coerce"
            ).to_numpy(dtype=float)
            mask = np.isfinite(x_smooth) & np.isfinite(y_smooth)
            x_smooth = x_smooth[mask]
            y_smooth = y_smooth[mask]
            if len(x_smooth) > 0:
                ax1.plot(
                    x_smooth,
                    y_smooth,
                    linewidth=2.6,
                    color="black",
                    label="Interpolation (precomputed)",
                )

        veq = res.get("veq_used", np.nan)
        ph_at_veq = np.nan
        ph_at_half = np.nan
        if (
            np.isfinite(veq)
            and not dense_df.empty
            and {"Volume (cm³)", "pH_interp"}.issubset(dense_df.columns)
        ):
            x_smooth = pd.to_numeric(
                dense_df["Volume (cm³)"], errors="coerce"
            ).to_numpy(dtype=float)
            y_smooth = pd.to_numeric(
                dense_df["pH_interp"], errors="coerce"
            ).to_numpy(dtype=float)
            if len(x_smooth) > 0:
                idx_veq = int(np.nanargmin(np.abs(x_smooth - veq)))
                ph_at_veq = y_smooth[idx_veq]
                idx_half = int(np.nanargmin(np.abs(x_smooth - veq / 2)))
                ph_at_half = y_smooth[idx_half]
                ax1.axvline(
                    veq,
                    color="black",
                    linestyle="--",
                    linewidth=1.8,
                    label=f"Equivalence point: pH {ph_at_veq:.2f}",
                )
                ax1.axvline(
                    veq / 2,
                    color="black",
                    linestyle=":",
                    linewidth=1.8,
                    label=f"Half-equivalence point: pH {ph_at_half:.2f}",
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
                        label=f"Equivalence point: pH {ph_at_veq:.2f}",
                    )
                    ax2.axvline(
                        veq / 2,
                        color="black",
                        linestyle=":",
                        linewidth=1.6,
                        label=f"Half-equivalence point: pH {ph_at_half:.2f}",
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
                s=60,
                edgecolor="black",
                facecolor="white",
                linewidth=1.2,
                label="Buffer region points",
            )

            if len(xhh) > 1 and np.any(np.isfinite(yfit)):
                order = np.argsort(xhh)
                ax3.plot(
                    xhh[order],
                    yfit[order],
                    color="black",
                    linewidth=2.2,
                    label="Best-fit line",
                )

                slope = res.get("slope_reg", np.nan)
                intercept = res.get("pka_app", np.nan)
                r2 = res.get("r2_reg", np.nan)
                if np.isfinite(slope) and np.isfinite(intercept):
                    label_r2 = f"\n$R^2 = {r2:.3f}$" if np.isfinite(r2) else ""
                    ax3.text(
                        0.98,
                        0.02,
                        rf"$\mathrm{{pH}} = {slope:.3f}\,x + {intercept:.3f}$"
                        + label_r2,
                        transform=ax3.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=14,
                    )

        ax3.set_title("Henderson-Hasselbalch", fontweight="bold")
        ax3.set_xlabel(r"$\log_{10}\!\left(\frac{V}{V_{eq}-V}\right)$")
        ax3.set_ylabel("pH")
        ax3.yaxis.set_major_locator(MaxNLocator(nbins=7))
        ax3.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax3.legend(loc="best")

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
