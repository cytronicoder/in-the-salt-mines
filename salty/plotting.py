"""
Plotting module for titration analysis.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def setup_plot_style():
    """
    Configure matplotlib style for professional plots.

    Args:
        None

    Returns:
        None
    """
    plt.style.use("default")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 14


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

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        ax1.plot(
            raw_df[x_col],
            raw_df["pH"],
            ".",
            color=color,
            alpha=0.3,
            label="Raw pH",
            markersize=4,
            clip_on=False,
        )

        ax1.plot(
            step_df["Volume (cm³)"],
            step_df["pH_step"],
            "o",
            color=color,
            label="Step pH (median)",
            markersize=6,
            clip_on=False,
        )

        if not dense_df.empty:
            ax1.plot(
                dense_df["Volume (cm³)"],
                dense_df["pH_interp"],
                "-",
                color=color,
                label="Interpolated pH",
                linewidth=2,
                clip_on=False,
            )

        ax1.axvline(
            res["veq_used"],
            color=color,
            linestyle="--",
            linewidth=2.5,
            label=f'Equivalence ({res["veq_method"]})',
        )
        ax1.axvline(
            res["half_eq_x"],
            color=color,
            linestyle=":",
            linewidth=2.5,
            label=f'Half-Equivalence: pH = {res["half_eq_pH"]:.2f}',
        )
        ax1.plot(
            res["half_eq_x"],
            res["half_eq_pH"],
            color=color,
            marker="o",
            markersize=10,
        )
        ax1.set_title(f"Titration Curve: {run_name}", fontsize=24, fontweight="bold")
        ax1.set_xlabel(x_label, fontsize=18)
        ax1.set_ylabel(r"pH", fontsize=18)
        ax1.tick_params(labelsize=16)
        ax1.legend(fontsize=18, loc="best")
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        ax2.plot(
            step_df["Volume (cm³)"],
            step_df["dpH/dx"],
            "-",
            color=color,
            label=r"$\frac{dpH}{dV}$" if x_col == "Volume (cm³)" else r"$\frac{dpH}{dt}$",
            linewidth=2,
            clip_on=False,
        )
        ax2.axvline(
            res["veq_used"],
            color=color,
            linestyle="--",
            linewidth=2.5,
            label="Equivalence",
        )
        ax2.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
        ax2.set_title(f"First Derivative: {run_name}", fontsize=24, fontweight="bold")
        ax2.set_xlabel(x_label, fontsize=18)
        ax2.set_ylabel(
            r"$\frac{dpH}{dV}$" if x_col == "Volume (cm³)" else r"$\frac{dpH}{dt}$",
            fontsize=18,
        )
        ax2.tick_params(labelsize=16)
        ax2.legend(fontsize=18, loc="best")
        ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        ax3.plot(
            res["buffer_region"]["log10_ratio"],
            res["buffer_region"]["pH_step"],
            "o",
            color=color,
            label="Buffer region",
            markersize=6,
            clip_on=False,
        )
        if not res["buffer_region"].empty:
            ax3.plot(
                res["buffer_region"]["log10_ratio"],
                res["buffer_region"]["pH_fit"],
                "-",
                color="black",
                linewidth=2,
                label=(
                    f"Fit: slope={res['slope_reg']:.2f}, "
                    f"pKa={res['pka_reg']:.2f}"
                ),
            )

        x_min = step_df["Volume (cm³)"].min()
        x_max = step_df["Volume (cm³)"].max()
        if x_max is None or x_min is None:
            margin = 0.1
        else:
            span = (
                x_max - x_min if x_max != x_min else abs(x_max) if x_max != 0 else 1.0
            )
            margin = span * 0.02

        for ax in (ax1, ax2):
            ax.set_xlim(x_min - margin, x_max + margin)
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
        sanitized_name = run_name.replace(" ", "_").replace("/", "_")
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

        ax1.errorbar(
            conc,
            row["Mean pKa"],
            yerr=row["SD"],
            fmt="o-",
            color=color,
            linewidth=3,
            markersize=12,
            capsize=8,
            capthick=3,
            label=f"{conc} M (n={int(row['n'])})",
            elinewidth=2.5,
        )

        subset = results_df[results_df["NaCl Concentration (M)"] == conc]
        ax1.plot(
            [conc] * len(subset),
            subset["pKa (buffer regression)"],
            marker="s",
            color=color,
            markersize=10,
            alpha=0.6,
            linestyle="",
        )

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
        if pd.notna(row["SD"]):
            label = f"{row['Mean pKa']:.3f} ± {row['SD']:.3f}"
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
