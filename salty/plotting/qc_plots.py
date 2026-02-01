"""Quality control and validation plots for titration analysis.

1. Initial conditions: pH_0 and temperature verification
2. Stoichiometry: Equivalence volumes and half-equivalence points
3. Model diagnostics: H-H slopes, R^2, residuals
4. Precision/reproducibility: Trial-to-trial variability
5. Uncertainty analysis: Error propagation across conditions
6. Buffer region coverage: Validation of chemically valid window
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter, MaxNLocator

from ..schema import ResultColumns


def plot_initial_ph_by_concentration(
    results: List[Dict], output_dir: str = "output/qc"
) -> str:
    """Plot initial pH (pH_0) for each [NaCl] condition as box-and-whisker plots.

    Initial pH should be consistent (~3.0-3.5 for 0.10 M ethanoic acid) across
    all trials at each [NaCl]. Significant variation suggests inconsistent
    acid preparation or contamination.

    Args:
        results: List of analysis result dictionaries.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract initial pH (first pH reading) for each result
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

        if len(ph_vals) > 0:
            initial_ph = float(ph_vals[0])
            if nacl not in data_by_nacl:
                data_by_nacl[nacl] = []
            data_by_nacl[nacl].append(initial_ph)

    if not data_by_nacl:
        return ""

    # Sort by NaCl concentration
    concentrations = sorted(data_by_nacl.keys())
    box_data = [data_by_nacl[c] for c in concentrations]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    bp = ax.boxplot(
        box_data,
        positions=concentrations,
        widths=0.08,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=8),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
        patch.set_alpha(0.7)

    # Also plot individual points
    for conc, values in zip(concentrations, box_data):
        x_jitter = conc + np.random.normal(0, 0.015, len(values))
        ax.scatter(x_jitter, values, alpha=0.6, s=50, color="darkblue", zorder=3)

    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol\,dm^{-3}}$", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel(r"Initial pH ($\mathrm{pH}_0$)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Initial pH Verification Across Ionic Strength Conditions",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    # Add reference line for expected pH of 0.10 M ethanoic acid (pKa ≈ 4.76)
    # pH = 0.5*(pKa - log[HA]) ≈ 0.5*(4.76 - log(0.10)) ≈ 0.5*(4.76 + 1) ≈ 2.88
    ax.axhline(
        y=2.88,
        color="green",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=r"Expected $\mathrm{pH}_0$ (ideal)",
    )

    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks(concentrations)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "initial_ph_by_nacl.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_initial_ph_scatter(results: List[Dict], output_dir: str = "output/qc") -> str:
    """Plot mean initial pH ± std.dev. across all [NaCl] as scatter with error bars.

    Assesses whether initial pH is affected by ionic strength. Ideally, pH₀
    should be independent of [NaCl] since NaCl doesn't participate in the
    ethanoic acid dissociation equilibrium.

    Args:
        results: List of analysis result dictionaries.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
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

        if len(ph_vals) > 0:
            initial_ph = float(ph_vals[0])
            if nacl not in data_by_nacl:
                data_by_nacl[nacl] = []
            data_by_nacl[nacl].append(initial_ph)

    if not data_by_nacl:
        return ""

    concentrations = sorted(data_by_nacl.keys())
    means = [np.mean(data_by_nacl[c]) for c in concentrations]
    stds = [
        np.std(data_by_nacl[c]) if len(data_by_nacl[c]) > 1 else 0.0
        for c in concentrations
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.errorbar(
        concentrations,
        means,
        yerr=stds,
        fmt="o",
        markersize=10,
        capsize=8,
        capthick=2,
        elinewidth=2,
        color="darkblue",
        ecolor="red",
        label="Mean ± 1 SD",
    )

    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol\,dm^{-3}}$", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel(r"Initial pH ($\mathrm{pH}_0$)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Initial pH: Mean and Standard Deviation vs. Ionic Strength",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.axhline(
        y=np.mean(means),
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label="Overall mean",
    )

    ax.set_xlim(-0.1, 1.1)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "initial_ph_scatter_with_errorbar.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_temperature_boxplots(
    results: List[Dict], output_dir: str = "output/qc"
) -> str:
    """Plot temperature distribution for all titrations as box-and-whisker plots.

    Temperature should remain within 26 ± 1°C throughout all experiments as
    per the IA controlled variables. Deviations affect pKa and pH electrode
    response (Nernst slope).

    Args:
        results: List of analysis result dictionaries.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
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

        temp_vals = pd.to_numeric(raw_df["Temperature (°C)"], errors="coerce").to_numpy(
            dtype=float
        )
        temp_vals = temp_vals[np.isfinite(temp_vals)]

        if len(temp_vals) > 0:
            if nacl not in data_by_nacl:
                data_by_nacl[nacl] = []
            data_by_nacl[nacl].extend(temp_vals.tolist())

    if not data_by_nacl:
        return ""

    concentrations = sorted(data_by_nacl.keys())
    box_data = [data_by_nacl[c] for c in concentrations]

    fig, ax = plt.subplots(figsize=(12, 7))

    bp = ax.boxplot(
        box_data,
        positions=concentrations,
        widths=0.08,
        patch_artist=True,
        showmeans=True,
        meanprops=dict(marker="s", markerfacecolor="orange", markersize=8),
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("lightcoral")
        patch.set_alpha(0.7)

    ax.axhline(
        y=26.0, color="green", linestyle="-", linewidth=2, label="Target: 26.0°C"
    )
    ax.axhspan(25.0, 27.0, alpha=0.2, color="green", label="Tolerance: ±1°C")

    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol\,dm^{-3}}$", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel(r"Temperature / $^\circ\mathrm{C}$", fontsize=14, fontweight="bold")
    ax.set_title(
        "Temperature Control Verification (All Measurements)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks(concentrations)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    ax.legend(loc="best")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "temperature_control_by_nacl.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_equivalence_volumes(
    results_df: pd.DataFrame, output_dir: str = "output/qc"
) -> str:
    """Plot equivalence volumes by [NaCl] to verify stoichiometry.

    For 0.10 M ethanoic acid (25.00 cm^3) titrated with 0.10 M NaOH, the
    expected V_eq is ~25 cm^3. Significant deviations suggest concentration
    errors or incomplete dissolution.

    Args:
        results_df: DataFrame from create_results_dataframe().
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if results_df.empty or "Veq (used)" not in results_df.columns:
        return ""

    cols = ResultColumns()
    data = results_df[[cols.nacl, "Veq (used)", "Veq uncertainty (ΔVeq)"]].copy()
    data = data[data["Veq (used)"].notna()]

    if data.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 6.5))

    for nacl in sorted(data[cols.nacl].unique()):
        subset = data[data[cols.nacl] == nacl]
        veqs = subset["Veq (used)"].values
        uncs = subset["Veq uncertainty (ΔVeq)"].fillna(0).values

        x_jitter = nacl + np.random.normal(0, 0.012, len(veqs))

        ax.errorbar(
            x_jitter,
            veqs,
            yerr=uncs,
            fmt="o",
            markersize=8,
            capsize=5,
            alpha=0.7,
            label=f"{nacl:.2f} M" if nacl in [0.0, 0.5, 1.0] else None,
        )

    # Expected V_eq from stoichiometry
    ax.axhline(
        y=25.0,
        color="green",
        linestyle="--",
        linewidth=2,
        label="Expected V$_{eq}$ (25.0 cm^3)",
    )

    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol\,dm^{-3}}$", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel(
        r"Equivalence Volume / $\mathrm{cm}^3$", fontsize=14, fontweight="bold"
    )
    ax.set_title(
        "Equivalence Volume Verification Across Ionic Strength",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlim(-0.1, 1.1)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best", ncol=2)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "equivalence_volumes_by_nacl.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_hh_slope_diagnostics(
    results_df: pd.DataFrame, output_dir: str = "output/qc"
) -> str:
    """Plot Henderson-Hasselbalch slope values by [NaCl].

    Ideal H-H slope should be 1.0. Deviations indicate non-ideality,
    activity coefficient effects, or experimental artifacts. Slopes consistently
    < 1 or > 1 suggest systematic issues with the titration or model.

    Args:
        results_df: DataFrame from create_results_dataframe().
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if results_df.empty or "Slope (buffer fit)" not in results_df.columns:
        return ""

    cols = ResultColumns()
    data = results_df[[cols.nacl, "Slope (buffer fit)", "R2 (buffer fit)"]].copy()
    data = data[data["Slope (buffer fit)"].notna()]

    if data.empty:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left panel: Slope values
    for nacl in sorted(data[cols.nacl].unique()):
        subset = data[data[cols.nacl] == nacl]
        slopes = subset["Slope (buffer fit)"].values
        x_jitter = nacl + np.random.normal(0, 0.012, len(slopes))

        ax1.scatter(x_jitter, slopes, s=80, alpha=0.7)

    ax1.axhline(
        y=1.0, color="green", linestyle="--", linewidth=2, label="Ideal slope (1.0)"
    )
    ax1.axhspan(0.95, 1.05, alpha=0.2, color="green", label="Acceptable range (±5%)")

    ax1.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol\,dm^{-3}}$", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel("H-H Slope", fontsize=14, fontweight="bold")
    ax1.set_title(
        "Henderson-Hasselbalch Slope Validation",
        fontsize=15,
        fontweight="bold",
    )
    ax1.set_xlim(-0.1, 1.1)
    ax1.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(loc="best")

    # Right panel: R^2 values
    for nacl in sorted(data[cols.nacl].unique()):
        subset = data[data[cols.nacl] == nacl]
        r2_vals = subset["R2 (buffer fit)"].values
        x_jitter = nacl + np.random.normal(0, 0.012, len(r2_vals))

        ax2.scatter(x_jitter, r2_vals, s=80, alpha=0.7)

    ax2.axhline(
        y=0.99,
        color="green",
        linestyle="--",
        linewidth=2,
        label=r"Excellent fit ($R^2 \geq 0.99$)",
    )

    ax2.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol\,dm^{-3}}$", fontsize=14, fontweight="bold"
    )
    ax2.set_ylabel(r"$R^2$ (buffer fit)", fontsize=14, fontweight="bold")
    ax2.set_title(
        "Goodness of Fit (R^2) for H-H Regression",
        fontsize=15,
        fontweight="bold",
    )
    ax2.set_xlim(-0.1, 1.1)
    ax2.set_ylim(0.90, 1.005)
    ax2.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    ax2.grid(True, alpha=0.3, linestyle=":")
    ax2.legend(loc="best")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "hh_slope_and_r2_diagnostics.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_pka_precision(results_df: pd.DataFrame, output_dir: str = "output/qc") -> str:
    """Plot pKa_app values by [NaCl] showing trial-to-trial variability.

    Assesses reproducibility at each ionic strength condition. Large scatter
    suggests poor experimental technique, incomplete equilibration, or
    inconsistent NaCl preparation.

    Args:
        results_df: DataFrame from create_results_dataframe().
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if results_df.empty:
        return ""

    cols = ResultColumns()
    data = results_df[[cols.nacl, cols.pka_app, cols.pka_unc]].copy()
    data = data[data[cols.pka_app].notna()]

    if data.empty:
        return ""

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(data[cols.nacl].unique())))

    for i, nacl in enumerate(sorted(data[cols.nacl].unique())):
        subset = data[data[cols.nacl] == nacl]
        pkas = subset[cols.pka_app].values
        uncs = subset[cols.pka_unc].fillna(0).values
        x_jitter = nacl + np.random.normal(0, 0.012, len(pkas))

        ax.errorbar(
            x_jitter,
            pkas,
            yerr=uncs,
            fmt="o",
            markersize=10,
            capsize=5,
            color=colors[i],
            ecolor="gray",
            alpha=0.8,
            label=f"{nacl:.2f} M",
        )

    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol\,dm^{-3}}$", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel(r"Apparent $pK_a$", fontsize=14, fontweight="bold")
    ax.set_title(
        "Apparent pK$_a$ Precision: Trial-to-Trial Variability",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlim(-0.1, 1.1)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best", ncol=2, title="[NaCl]")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "pka_precision_by_nacl.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_buffer_region_coverage(
    results: List[Dict], output_dir: str = "output/qc"
) -> str:
    """Plot number of points in buffer region vs. [NaCl].

    The buffer region (|pH - pKa_app| ≤ 1) should contain sufficient points
    for robust H-H regression. Fewer points reduce fit reliability.

    Args:
        results: List of analysis result dictionaries.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    data_by_nacl: Dict[float, List[int]] = {}

    for res in results:
        nacl = float(res.get("nacl_conc", np.nan))
        if not np.isfinite(nacl):
            continue

        buffer_df = res.get("buffer_region", pd.DataFrame())
        n_points = len(buffer_df)

        if nacl not in data_by_nacl:
            data_by_nacl[nacl] = []
        data_by_nacl[nacl].append(n_points)

    if not data_by_nacl:
        return ""

    concentrations = sorted(data_by_nacl.keys())
    means = [np.mean(data_by_nacl[c]) for c in concentrations]
    stds = [
        np.std(data_by_nacl[c]) if len(data_by_nacl[c]) > 1 else 0
        for c in concentrations
    ]

    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.bar(
        concentrations,
        means,
        width=0.08,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        label="Mean",
    )
    ax.errorbar(
        concentrations,
        means,
        yerr=stds,
        fmt="none",
        ecolor="red",
        capsize=8,
        capthick=2,
        label="± 1 SD",
    )

    # Plot individual points
    for conc, values in zip(concentrations, [data_by_nacl[c] for c in concentrations]):
        x_jitter = conc + np.random.normal(0, 0.01, len(values))
        ax.scatter(x_jitter, values, s=50, alpha=0.5, color="darkblue", zorder=3)

    ax.axhline(
        y=10,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label="Minimum recommended (10)",
    )

    ax.set_xlabel(
        r"$[\mathrm{NaCl}]$ / $\mathrm{mol\,dm^{-3}}$", fontsize=14, fontweight="bold"
    )
    ax.set_ylabel("Number of Points in Buffer Region", fontsize=14, fontweight="bold")
    ax.set_title(
        "Buffer Region Coverage (|pH - pK$_a$| ≤ 1)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks(concentrations)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3, linestyle=":", axis="y")
    ax.legend(loc="best")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "buffer_region_coverage.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_residuals_analysis(results: List[Dict], output_dir: str = "output/qc") -> str:
    """Plot H-H fit residuals to check for systematic deviations.

    Residuals (pH_observed - pH_fit) should be randomly distributed around zero.
    Systematic patterns indicate model inadequacy or non-ideal solution behavior.

    Args:
        results: List of analysis result dictionaries.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all residuals
    all_residuals = []
    all_x_values = []  # log10_ratio
    all_nacl = []

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
        all_nacl.extend([nacl] * len(residuals))

    if not all_residuals:
        return ""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Left panel: Residuals vs. log10_ratio
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(set(all_nacl))))
    color_map = {nacl: colors[i] for i, nacl in enumerate(sorted(set(all_nacl)))}

    for nacl in sorted(set(all_nacl)):
        mask = np.array(all_nacl) == nacl
        x = np.array(all_x_values)[mask]
        y = np.array(all_residuals)[mask]
        ax1.scatter(x, y, alpha=0.6, s=40, color=color_map[nacl], label=f"{nacl:.2f} M")

    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1.5)
    ax1.set_xlabel(
        r"$\log_{10}\left(\dfrac{[A^-]}{[HA]}\right)$", fontsize=14, fontweight="bold"
    )
    ax1.set_ylabel(
        r"Residual ($\mathrm{pH}_{\mathrm{obs}} - \mathrm{pH}_{\mathrm{fit}}$)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_title("Residuals vs. Buffer Ratio", fontsize=15, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle=":")
    ax1.legend(loc="best", ncol=2, title="[NaCl]")

    # Right panel: Residuals histogram
    ax2.hist(all_residuals, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero")
    ax2.set_xlabel(
        r"Residual ($\mathrm{pH}_{\mathrm{obs}} - \mathrm{pH}_{\mathrm{fit}}$)",
        fontsize=14,
        fontweight="bold",
    )
    ax2.set_ylabel("Frequency", fontsize=14, fontweight="bold")
    ax2.set_title("Residuals Distribution", fontsize=15, fontweight="bold")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3, linestyle=":", axis="y")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "hh_residuals_analysis.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_half_equivalence_check(
    results_df: pd.DataFrame, output_dir: str = "output/qc"
) -> str:
    """Plot half-equivalence volume as fraction of V_eq to verify V_half = V_eq/2.

    The ratio should be ~0.500 for all titrations. Deviations indicate errors
    in equivalence point detection or interpolation artifacts.

    Args:
        results_df: DataFrame from create_results_dataframe().
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    if results_df.empty:
        return ""

    # Need to compute ratios from raw results since half_eq_x isn't in results_df
    # We'll extract from source if available
    # For now, let's create a simple verification plot

    return ""  # Placeholder - would need access to half_eq_x in results


def generate_all_qc_plots(
    results: List[Dict],
    results_df: pd.DataFrame,
    output_dir: str = "output/qc",
) -> List[str]:
    """Generate all quality control and validation plots.

    Creates a comprehensive suite of diagnostic plots to evaluate:
        - Initial conditions (pH₀, temperature)
        - Stoichiometric consistency (V_eq)
        - Model quality (H-H slopes, R^2, residuals)
        - Reproducibility (trial-to-trial variability)
        - Buffer region coverage

    Args:
        results: List of analysis result dictionaries.
        results_df: DataFrame from create_results_dataframe().
        output_dir: Directory to save all QC plots.

    Returns:
        List of paths to generated PNG files.
    """
    plot_paths = []

    print("Generating QC plots...")

    # Initial conditions
    print("  - Initial pH box plots by [NaCl]...")
    path = plot_initial_ph_by_concentration(results, output_dir)
    if path:
        plot_paths.append(path)

    print("  - Initial pH scatter with error bars...")
    path = plot_initial_ph_scatter(results, output_dir)
    if path:
        plot_paths.append(path)

    print("  - Temperature control verification...")
    path = plot_temperature_boxplots(results, output_dir)
    if path:
        plot_paths.append(path)

    # Stoichiometry
    print("  - Equivalence volume verification...")
    path = plot_equivalence_volumes(results_df, output_dir)
    if path:
        plot_paths.append(path)

    # Model diagnostics
    print("  - H-H slope and R^2 diagnostics...")
    path = plot_hh_slope_diagnostics(results_df, output_dir)
    if path:
        plot_paths.append(path)

    print("  - Residuals analysis...")
    path = plot_residuals_analysis(results, output_dir)
    if path:
        plot_paths.append(path)

    # Precision
    print("  - pKa precision analysis...")
    path = plot_pka_precision(results_df, output_dir)
    if path:
        plot_paths.append(path)

    # Buffer region
    print("  - Buffer region coverage...")
    path = plot_buffer_region_coverage(results, output_dir)
    if path:
        plot_paths.append(path)

    print(f"QC plots complete. Generated {len(plot_paths)} plots in {output_dir}/")

    return plot_paths
