"""
Plotting module for titration analysis.
Creates professional, publication-ready figures.

Functions:
    setup_plot_style(): Configures matplotlib style for professional plots.
    plot_titration_curves(results, output_dir="output"): Plots titration curves and first derivatives for all runs.
    plot_statistical_summary(stats_df, results_df, output_dir="output"): Plots statistical summary with error bars.
    save_data_to_csv(results_df, stats_df, output_dir="output"): Saves processed data to CSV files.
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
    Plot titration curves and first derivatives for all runs.

    Args:
        results (list): List of analysis result dictionaries.
        output_dir (str, optional): Directory to save figures (default: 'output').

    Returns:
        list: List of paths to saved figures.
    """
    setup_plot_style()
    colors = ["black", "gray"]

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []

    for i, res in enumerate(results):
        df = res["data"]
        run_name = res["run_name"]
        color = colors[int(res["nacl_conc"])]

        # Create individual figure with two subplots (titration curve and first derivative)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left panel: Titration curve
        ax1.plot(
            df["Time (min)"],
            df["pH"],
            ".-",
            color=color,
            label="pH",
            linewidth=2,
            markersize=6,
        )
        ax1.axvline(
            res["eq_time"],
            color=color,
            linestyle="--",
            linewidth=2.5,
            label=f'Equivalence: pH = {res["eq_pH"]:.2f}',
        )
        ax1.axvline(
            res["half_eq_time"],
            color=color,
            linestyle=":",
            linewidth=2.5,
            label=f'Half-Equivalence: pH = {res["half_eq_pH"]:.2f}',
        )
        ax1.plot(
            res["half_eq_time"],
            res["half_eq_pH"],
            color=color,
            marker="o",
            markersize=10,
        )
        ax1.set_title(f"Titration Curve: {run_name}", fontsize=18, fontweight="bold")
        ax1.set_xlabel(r"Time (min)", fontsize=14)
        ax1.set_ylabel(r"pH", fontsize=14)
        ax1.tick_params(labelsize=12)
        ax1.legend(fontsize=12, loc="best")
        ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Right panel: First derivative
        ax2.plot(
            df["Time (min)"],
            df["dpH/dt"],
            "-",
            color=color,
            label=r"$\frac{dpH}{dt}$",
            linewidth=2,
        )
        ax2.axvline(
            res["eq_time"],
            color=color,
            linestyle="--",
            linewidth=2.5,
            label=f'Equivalence: pH = {res["eq_pH"]:.2f}',
        )
        ax2.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)
        ax2.set_title(f"First Derivative: {run_name}", fontsize=18, fontweight="bold")
        ax2.set_xlabel(r"Time (min)", fontsize=14)
        ax2.set_ylabel(r"$\frac{dpH}{dt}$", fontsize=14)
        ax2.tick_params(labelsize=12)
        ax2.legend(fontsize=12, loc="best")
        ax2.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()

        # Save individual figure with sanitized filename
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
    colors = ["red", "orange", "green", "blue", "purple", "brown", "pink", "gray"]

    os.makedirs(output_dir, exist_ok=True)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 8))

    for i, row in stats_df.iterrows():
        conc = row["NaCl Concentration (M)"]
        color = colors[i]

        # Plot mean with error bars
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
            label=f"{conc} M NaCl (Mean ± SD)",
            elinewidth=2.5,
        )

        # Plot individual runs
        subset = results_df[results_df["NaCl Concentration (M)"] == conc]
        ax1.plot(
            [conc] * len(subset),
            subset["Half-Equivalence pH (pKa)"],
            marker="s",
            color=color,
            markersize=10,
            alpha=0.6,
            linestyle="",
            label=f"{conc} M NaCl (Individual Runs)" if i == 0 else "",
        )

    ax1.set_xlabel(
        r"NaCl Concentration (mol dm$^{-3}$)", fontsize=16, fontweight="bold"
    )
    ax1.set_ylabel(r"Apparent $pK_a$", fontsize=16, fontweight="bold")
    ax1.set_title(
        r"Effect of NaCl on $pK_a$ of Ethanoic Acid",
        fontsize=24,
        fontweight="bold",
        pad=20,
    )
    ax1.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax1.tick_params(labelsize=14)
    ax1.legend(fontsize=14, loc="upper left", frameon=True, shadow=True)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Add value annotations
    for _, row in stats_df.iterrows():
        if pd.notna(row["SD"]):
            label = f"{row['Mean pKa']:.3f} ± {row['SD']:.3f}"
        else:
            label = f"{row['Mean pKa']:.3f}"
        ax1.annotate(
            label,
            (row["NaCl Concentration (M)"], row["Mean pKa"]),
            textcoords="offset points",
            xytext=(0, 20),
            ha="center",
            fontsize=12,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
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
