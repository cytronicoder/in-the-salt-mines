"""
Plotting module for titration analysis.
"""

import os
import re

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def setup_plot_style():
    """Configure matplotlib style for IA-friendly, print-safe plots.

    - Uses serif fonts suitable for reports
    - Larger default font sizes for readability when printed
    - Palette chosen to be distinguishable in grayscale
    """
    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 16
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["axes.labelsize"] = 16
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14
    # Use a grayscale-friendly palette
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=["black", "gray", "dimgray", "dimgrey", "silver", "darkgray"])

def plot_titration_curves(results, output_dir="output"):
    """
    Plot titration curves, derivatives, and Henderson–Hasselbalch diagnostics.

    Args:
        results (list): List of analysis result dictionaries.
        output_dir (str, optional): Directory to save figures (default: 'output').

    Returns:
        list: List of paths to saved figures.
    """
    setup_plot_style()
    colors = ["black", "gray", "steelblue", "seagreen", "indianred", "goldenrod"]

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    for i, res in enumerate(results):
        raw_df = res["data"]
        step_df = res["step_data"]
        dense_df = res["dense_curve"]
        run_name = res["run_name"]
        x_col = res.get("x_col", "Volume (cm³)")
        x_label = r"Volume of NaOH added (cm$^3$)" if x_col == "Volume (cm³)" else r"Time (min)"
        color = colors[i % len(colors)]

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7))

        # -- Panel 1: Raw and step pH with interpolated curve --
        ax1.plot(
            raw_df[x_col],
            raw_df["pH"],
            ".",
            color=color,
            alpha=0.3,
            label="Raw pH",
            markersize=6,
            clip_on=False,
        )

        # If step-level uncertainties are present, show them as error bars
        if "pH_step_sd" in step_df.columns and not step_df["pH_step_sd"].isna().all():
            ax1.errorbar(
                step_df["Volume (cm³)"],
                step_df["pH_step"],
                yerr=step_df["pH_step_sd"],
                fmt="o",
                color=color,
                label="Step pH (median ± SD)",
                markersize=8,
                capsize=4,
                clip_on=False,
            )
        else:
            ax1.plot(
                step_df["Volume (cm³)"],
                step_df["pH_step"],
                "o",
                color=color,
                label="Step pH (median)",
                markersize=8,
                clip_on=False,
            )

        if not dense_df.empty:
            ax1.plot(
                dense_df["Volume (cm³)"],
                dense_df["pH_interp"],
                "-",
                color=color,
                label="Interpolated pH",
                linewidth=2.5,
                clip_on=False,
            )

        # Annotate equivalence and half-equivalence with uncertainties if available
        veq = res.get("veq_used", float('nan'))
        veq_unc = res.get("veq_uncertainty", None)
        half_x = res.get("half_eq_x", float('nan'))
        half_pH = res.get("half_eq_pH", float('nan'))

        if pd.notna(veq):
            ax1.axvline(veq, color=color, linestyle="--", linewidth=2.5, label=f'Equivalence ({res.get("veq_method","")})')
            if pd.notna(veq_unc):
                # shade uncertainty band
                ax1.fill_betweenx(ax1.get_ylim(), veq - veq_unc, veq + veq_unc, color=color, alpha=0.08)
                ax1.annotate(f"V_eq = {veq:.2f} ± {veq_unc:.2f} cm³", (veq, ax1.get_ylim()[1]), xytext=(5, -30), textcoords='offset points', fontsize=12)

        if pd.notna(half_x) and pd.notna(half_pH):
            ax1.axvline(half_x, color=color, linestyle=":", linewidth=2.5, label=f'Half-Equivalence')
            ax1.plot(half_x, half_pH, color=color, marker="o", markersize=10)
            ax1.annotate(f"pH_1/2 = {half_pH:.2f}", (half_x, half_pH), xytext=(5, -15), textcoords='offset points', fontsize=12)

        ax1.set_title(f"Titration Curve — {run_name}", fontsize=20, fontweight="bold")
        ax1.set_xlabel(r"Volume of NaOH added / cm$^3$", fontsize=16)
        ax1.set_ylabel(r"pH", fontsize=16)
        ax1.tick_params(labelsize=14)
        ax1.legend(loc="best")
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.6)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax2.plot(
            step_df["Volume (cm³)"],
            step_df["dpH/dx"],
            "-",
            color=color,
            label=r"$\frac{dpH}{dV}$" if x_col == "Volume (cm³)" else r"$\frac{dpH}{dt}$",
            linewidth=2.5,
            clip_on=False,
        )
        ax2.axvline(
            res["veq_used"],
            color=color,
            linestyle="--",
            linewidth=2.5,
            label="Equivalence",
        )
        ax2.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.6)
        ax2.set_title(f"First Derivative — {run_name}", fontsize=18, fontweight="bold")
        ax2.set_xlabel(x_label, fontsize=14)
        ax2.set_ylabel(
            r"$\frac{dpH}{dV}$" if x_col == "Volume (cm³)" else r"$\frac{dpH}{dt}$",
            fontsize=14,
        )
        ax2.tick_params(labelsize=12)
        ax2.legend(loc="best")
        ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        buffer_df = res.get("buffer_region", pd.DataFrame())
        if not buffer_df.empty and {"log10_ratio", "pH_step"}.issubset(buffer_df.columns):
            ax3.plot(
                buffer_df["log10_ratio"],
                buffer_df["pH_step"],
                "o",
                color=color,
                label="Buffer region",
                markersize=8,
                clip_on=False,
            )
        if not buffer_df.empty and {"log10_ratio", "pH_fit"}.issubset(buffer_df.columns):
            ax3.plot(
                buffer_df["log10_ratio"],
                buffer_df["pH_fit"],
                "-",
                color="black",
                linewidth=2.5,
                label=(
                    f"Fit: slope={res.get('slope_reg', float('nan')):.2f}, "
                    f"pKa={res.get('pka_reg', float('nan')):.2f}"
                ),
            )

            # Compute R^2 for the fit if enough points are present
            if len(buffer_df) >= 2:
                x = buffer_df["log10_ratio"].values
                y = buffer_df["pH_step"].values
                coeffs = np.polyfit(x, y, 1)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
                eq_text = f"y = {coeffs[0]:.3f} x + {coeffs[1]:.3f}, $R^2$ = {r2:.3f}"
                ax3.annotate(eq_text, xy=(0.05, 0.95), xycoords="axes fraction", fontsize=12, va="top")

        ax3.set_title(
            f"Henderson–Hasselbalch — {run_name}", fontsize=18, fontweight="bold"
        )
        ax3.set_xlabel(r"$\log_{10}\left(\frac{V}{V_{eq}-V}\right)$", fontsize=14)
        ax3.set_ylabel(r"pH", fontsize=14)
        ax3.tick_params(labelsize=12)
        ax3.legend(loc="best")
        ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        # Determine x-limits based on available x column to ensure clear margins
        x_column = x_col if x_col in step_df.columns else "Volume (cm³)"
        x_min = step_df[x_column].min()
        x_max = step_df[x_column].max()
        if pd.isna(x_min) or pd.isna(x_max):
            margin = 0.1
        else:
            span = x_max - x_min if x_max != x_min else abs(x_max) if x_max != 0 else 1.0
            margin = span * 0.02

        for ax in (ax1, ax2):
            ax.set_xlim(x_min - margin, x_max + margin)

        plt.tight_layout(w_pad=3.0)
        ax3.set_title(
            f"Henderson–Hasselbalch: {run_name}", fontsize=24, fontweight="bold"
        )
        ax3.set_xlabel(r"$\log_{10}\left(\frac{V}{V_{eq}-V}\right)$", fontsize=18)
        ax3.set_ylabel(r"pH", fontsize=18)
        ax3.tick_params(labelsize=16)
        ax3.legend(fontsize=18, loc="best")
        ax3.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)

        plt.tight_layout(w_pad=3.0)
        source_file = res.get("source_file", "")
        source_base = os.path.splitext(source_file)[0] if source_file else ""
        combined_name = f"{run_name}_{source_base}" if source_base else run_name
        # Sanitize to filesystem-friendly name
        sanitized_name = re.sub(r'[^A-Za-z0-9._-]+', '_', combined_name)
        output_path = os.path.join(output_dir, f"titration_curve_{sanitized_name}.png")
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved titration curve to {output_path}")
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def plot_statistical_summary(stats_df, results_df, output_dir="output"):
    """
    Plot statistical summary with error bars.

    Args:
        stats_df (pd.DataFrame): Statistical summary DataFrame.
        results_df (pd.DataFrame): Individual results DataFrame.
        output_dir (str, optional): Directory to save figures (default: 'output').

    Returns:
        str: Path to saved figure.
    """
    setup_plot_style()
    colors = ["red", "orange", "green", "blue", "purple", "brown", "pink"]

    os.makedirs(output_dir, exist_ok=True)

    fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))

    for i, row in stats_df.iterrows():
        conc = row["NaCl Concentration (M)"]
        color = colors[i]

        subset = results_df[results_df["NaCl Concentration (M)"] == conc]

        # If individual uncertainties are available, compute asymmetric bounds from
        # max(pKa_i + Δi) down to min(pKa_i - Δi) and use that as the plotted errorbar.
        if not subset.empty and "pKa uncertainty (ΔpKa)" in subset.columns and not subset["pKa uncertainty (ΔpKa)"].isna().all():
            pka_vals = subset["pKa (buffer regression)"].to_numpy()
            pka_uncs = subset["pKa uncertainty (ΔpKa)"].to_numpy()
            tops = pka_vals + pka_uncs
            bots = pka_vals - pka_uncs
            top = float(np.nanmax(tops))
            bot = float(np.nanmin(bots))
            mean_val = float(row["Mean pKa"])
            yerr_lower = mean_val - bot if np.isfinite(bot) else row.get("Uncertainty")
            yerr_upper = top - mean_val if np.isfinite(top) else row.get("Uncertainty")
            # ensure non-negative
            yerr_lower = max(0.0, yerr_lower)
            yerr_upper = max(0.0, yerr_upper)
            yerr = np.array([[yerr_lower], [yerr_upper]])
        else:
            # fallback to symmetric uncertainty or SD
            yerr_scalar = row.get("Uncertainty") if "Uncertainty" in row else row.get("SD")
            yerr = yerr_scalar

        ax1.errorbar(
            conc,
            row["Mean pKa"],
            yerr=yerr,
            fmt="o-",
            color=color,
            linewidth=3,
            markersize=12,
            capsize=8,
            capthick=3,
            label=f"{conc} M (n={int(row['n'])})",
            elinewidth=2.5,
        )

        # plot individual points with their own pKa uncertainty (if available)
        if not subset.empty:
            x_vals = [conc] * len(subset)
            y_vals = subset["pKa (buffer regression)"].values
            y_errs = subset.get("pKa uncertainty (ΔpKa)")
            if y_errs is None or y_errs.isna().all():
                ax1.plot(x_vals, y_vals, marker="s", color=color, markersize=10, alpha=0.6, linestyle="")
            else:
                ax1.errorbar(x_vals, y_vals, yerr=y_errs.values, marker="s", linestyle="", color=color, alpha=0.6)


    ax1.scatter([], [], marker="s", color="grey", alpha=0.6, label="Individual Runs")

    ax1.set_xlabel(
        r"NaCl Concentration (mol dm$^{-3}$)", fontsize=18, fontweight="bold"
    )
    ax1.set_ylabel(r"Apparent $pK_a$", fontsize=18, fontweight="bold")
    ax1.set_title(
        r"Effect of NaCl on $pK_a$ of Ethanoic Acid",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.tick_params(labelsize=14)
    ax1.legend(
        title="NaCl Concentration",
        fontsize=16,
        title_fontsize=18,
        loc="center left",
        bbox_to_anchor=(1.0, 0.925),
        frameon=True,
        shadow=True,
    )
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    for _, row in stats_df.iterrows():
        # Use worst-case uncertainty column if available
        unc = row.get("Uncertainty")
        if pd.notna(unc):
            # Use IB-style formatting for value ± uncertainty
            from .uncertainty import _format_value_with_uncertainty

            label = _format_value_with_uncertainty(row["Mean pKa"], unc, unit="")
        else:
            label = f"{row['Mean pKa']:.3f}"
        ax1.annotate(
            label,
            (row["NaCl Concentration (M)"], row["Mean pKa"]),
            textcoords="offset points",
            xytext=(0, -45),
            ha="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white"),
        )

    plt.tight_layout()

    output_path = os.path.join(output_dir, "statistical_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved statistical summary to {output_path}")
    plt.close()

    return output_path


def save_data_to_csv(results_df, stats_df, output_dir="output"):
    """
    Save processed data to CSV files.

    Args:
        results_df (pd.DataFrame): Individual results DataFrame.
        stats_df (pd.DataFrame): Statistical summary DataFrame.
        output_dir (str, optional): Directory to save CSV files (default: 'output').

    Returns:
        tuple: Paths to saved CSV files (results_path, stats_path).
    """
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "individual_results.csv")
    stats_path = os.path.join(output_dir, "statistical_summary.csv")

    results_df.to_csv(results_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    print(f"Saved individual results to {results_path}")
    print(f"Saved statistical summary to {stats_path}")

    return results_path, stats_path
