#!/usr/bin/env python3
"""Command-line entry point."""

import argparse
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from salty.analysis import (
    analyze_titration,
    build_summary_plot_data,
    calculate_statistics,
    create_results_dataframe,
    print_statistics,
)
from salty.data_processing import extract_runs, load_titration_data
from salty.output import save_data_to_csv
from salty.plotting import (
    generate_all_qc_plots,
    generate_ia_figure_set,
    plot_statistical_summary,
    plot_titration_curves,
)
from salty.plotting.style import (
    LABEL_NACL,
    apply_style,
    clean_axis,
    figure_base_path,
    safe_set_lims,
    save_figure_all_formats,
    should_plot_qq,
    warn_skipped_qq,
)
from salty.reporting import add_formatted_reporting_columns

EXACT_IV_LEVELS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
IV_TOL = 1e-9


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Titration analysis pipeline")
    parser.add_argument(
        "--figures",
        choices=("standard", "ia"),
        default="standard",
        help="Figure output profile. Use 'ia' for publication-ready Figure 1-5 set.",
    )
    parser.add_argument(
        "--profile",
        choices=("all_valid", "qc_pass", "strict_fit"),
        default="all_valid",
        help="Iteration profile used for output/iterations/<profile> IA figure export.",
    )
    return parser.parse_args(argv)


def _configure_logging():
    """Configure application logging for reproducible pipeline runs.

    Returns:
        None: Update root logger handlers and levels in place.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("titration_analysis.log", mode="w"),
        ],
    )


@dataclass(frozen=True)
class IterationSpec:
    """Store metadata for one analysis-iteration subset.

    Attributes:
        name (str): Iteration identifier used in output paths.
        description (str): Human-readable description of the filter definition.
    """

    name: str
    description: str


def _extract_nacl_concentration(file_path: str) -> float | None:
    """Extract NaCl concentration from a filename token.

    Args:
        file_path (str): Input file path that may contain tokens like ``0.8M``.

    Returns:
        float | None: Parsed concentration in mol dm^-3, or ``None`` if no
        concentration token is present.
    """
    base = os.path.basename(file_path)
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)M", base, flags=re.IGNORECASE)
    if not match:
        return None
    return float(match.group(1))


def _normalize_headers(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw Logger Pro headers for parser compatibility.

    Args:
        raw_df (pandas.DataFrame): Raw Logger Pro table with original headers.

    Returns:
        pandas.DataFrame: Header-normalized table with ``cm^3`` units and
        remapped ``Latest:`` run prefixes.

    Note:
        Replace unicode superscript ``cm³`` with ``cm^3`` and convert
        ``Latest:`` columns into an explicit ``Run N:`` prefix.
    """
    df = raw_df.copy()
    df.columns = [str(col).replace("cm³", "cm^3").strip() for col in df.columns]

    run_prefixes = []
    for col in df.columns:
        if col.startswith("Run ") and ":" in col:
            prefix = col.split(":", 1)[0]
            if prefix not in run_prefixes:
                run_prefixes.append(prefix)

    max_run = 0
    for prefix in run_prefixes:
        match = re.match(r"Run\s+(\d+)$", prefix)
        if match:
            max_run = max(max_run, int(match.group(1)))

    if any(col.startswith("Latest:") for col in df.columns):
        next_run = f"Run {max_run + 1 if max_run >= 1 else 1}"
        rename_map = {}
        for col in df.columns:
            if col.startswith("Latest:"):
                rename_map[col] = col.replace("Latest:", f"{next_run}:", 1)
        df = df.rename(columns=rename_map)

    return df


def _prepare_input_files() -> List[Tuple[str, float]]:
    """Discover and standardize input files for analysis.

    Returns:
        list[tuple[str, float]]: Sorted ``(standardized_csv_path,
        nacl_concentration_M)`` pairs filtered to exact designed IV levels.

    Note:
        This step normalizes Logger Pro headers and writes standardized copies
        into ``data/_standardized_raw`` for provenance and deterministic reuse.
    """
    raw_glob = sorted(
        [
            os.path.join("data", "raw", name)
            for name in os.listdir(os.path.join("data", "raw"))
            if name.lower().endswith(".csv")
        ]
    )

    standardized_dir = os.path.join("data", "_standardized_raw")
    os.makedirs(standardized_dir, exist_ok=True)

    import glob

    for stale in glob.glob(os.path.join(standardized_dir, "*.csv")):
        os.remove(stale)

    files: List[Tuple[str, float]] = []
    for path in raw_glob:
        conc = _extract_nacl_concentration(path)
        if conc is None:
            logging.warning("Skipping file without concentration token: %s", path)
            continue

        if not any(abs(conc - allowed) <= IV_TOL for allowed in EXACT_IV_LEVELS):
            logging.warning(
                "Skipping file outside exact IV levels %s: %s (%.3f M)",
                EXACT_IV_LEVELS,
                path,
                conc,
            )
            continue

        raw_df = pd.read_csv(path)
        std_df = _normalize_headers(raw_df)
        std_path = os.path.join(standardized_dir, os.path.basename(path))
        std_df.to_csv(std_path, index=False)
        files.append((std_path, conc))

    files = sorted(files, key=lambda item: (item[1], item[0]))
    return files


def _filter_exact_iv_results(results: list[dict]) -> list[dict]:
    """Filter run results to exact designed ionic-strength levels.

    Args:
        results (list[dict]): Run-level analysis payloads.

    Returns:
        list[dict]: Subset where ``nacl_conc`` matches one of
        ``EXACT_IV_LEVELS`` within ``IV_TOL``.
    """
    filtered = []
    for res in results:
        conc = float(res.get("nacl_conc", np.nan))
        if any(abs(conc - allowed) <= IV_TOL for allowed in EXACT_IV_LEVELS):
            filtered.append(res)
    return filtered


def _generate_additional_diagnostic_outputs(
    results_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    ia_dir: str,
) -> None:
    """Generate supplemental IA diagnostics from aggregated run tables.

    Args:
        results_df (pandas.DataFrame): Run-level results table.
        processed_df (pandas.DataFrame): Condition-level processed summary table.
        ia_dir (str): Output directory path for IA diagnostics.

    Returns:
        None: Write diagnostic CSV and PNG artifacts to ``ia_dir``.
    """
    os.makedirs(ia_dir, exist_ok=True)
    apply_style(font_scale=1.0, context="paper")

    def _save_diag_figure(fig_obj: plt.Figure, key: str) -> None:
        save_figure_all_formats(
            fig_obj, figure_base_path(fig_key=key, kind="diagnostics")
        )

    repeats_cols = [
        "Run",
        "NaCl Concentration (M)",
        "Apparent pKa",
        "Uncertainty in Apparent pKa",
        "Equivalence QC Pass",
        "Veq (used)",
        "Slope (buffer fit)",
        "R2 (buffer fit)",
        "Source File",
    ]
    repeats_df = (
        results_df[repeats_cols].copy().sort_values(["NaCl Concentration (M)", "Run"])
    )
    repeats_df = add_formatted_reporting_columns(
        repeats_df,
        [("Apparent pKa", "Uncertainty in Apparent pKa")],
    )
    repeats_df.to_csv(
        os.path.join(ia_dir, "diagnostic_dv_repeats_table.csv"), index=False
    )

    mean_df = (
        processed_df[
            [
                "NaCl Concentration (M)",
                "Mean pKa_app",
                "SD pKa_app",
                "SEM pKa_app",
                "Combined uncertainty",
            ]
        ]
        .dropna(subset=["NaCl Concentration (M)", "Mean pKa_app"])
        .sort_values("NaCl Concentration (M)")
        .reset_index(drop=True)
    )

    x = mean_df["NaCl Concentration (M)"].to_numpy(dtype=float)
    y = mean_df["Mean pKa_app"].to_numpy(dtype=float)

    if len(x) < 3:
        return

    m_u, b_u = np.polyfit(x, y, 1)
    yhat_u = m_u * x + b_u
    resid_u = y - yhat_u
    ss_res_u = float(np.sum((y - yhat_u) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2_u = float(1.0 - ss_res_u / ss_tot) if ss_tot > 0 else np.nan

    sem = mean_df["SEM pKa_app"].to_numpy(dtype=float)
    valid_w = np.isfinite(sem) & (sem > 0)
    if np.all(valid_w):
        w = 1.0 / sem
        m_w, b_w = np.polyfit(x, y, 1, w=w)
        yhat_w = m_w * x + b_w
        ss_res_w = float(np.sum((y - yhat_w) ** 2))
        r2_w = float(1.0 - ss_res_w / ss_tot) if ss_tot > 0 else np.nan
    else:
        m_w, b_w, r2_w = np.nan, np.nan, np.nan

    regression_compare = pd.DataFrame(
        [
            {
                "model": "unweighted_linear",
                "slope": m_u,
                "intercept": b_u,
                "R2": r2_u,
            },
            {
                "model": "weighted_linear(1/SEM)",
                "slope": m_w,
                "intercept": b_w,
                "R2": r2_w,
            },
        ]
    )
    regression_compare.to_csv(
        os.path.join(ia_dir, "diagnostic_regression_weighted_vs_unweighted.csv"),
        index=False,
    )

    X = np.column_stack([np.ones_like(x), x])
    xtx_inv = np.linalg.inv(X.T @ X)
    h = np.array([X[i] @ xtx_inv @ X[i].T for i in range(len(x))], dtype=float)
    p = 2
    dof = max(len(x) - p, 1)
    mse = float(np.sum(resid_u**2) / dof)
    if mse > 0:
        cooks_d = (resid_u**2 / (p * mse)) * (h / ((1 - h) ** 2))
    else:
        cooks_d = np.full_like(x, np.nan)

    diagnostics_df = pd.DataFrame(
        {
            "NaCl Concentration (M)": x,
            "Mean pKa_app": y,
            "Predicted pKa_app (unweighted)": yhat_u,
            "Residual (unweighted)": resid_u,
            "Leverage": h,
            "Cooks distance": cooks_d,
        }
    )
    diagnostics_df.to_csv(
        os.path.join(ia_dir, "diagnostic_diagnostic_metrics.csv"), index=False
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.scatter(x, resid_u, color="black", zorder=3)
    ax.set_xlabel(rf"${LABEL_NACL}$")
    ax.set_ylabel(r"$\mathrm{Residual}\;/\;\mathrm{pH}$")
    ax.set_title("Residuals vs ionic strength")
    clean_axis(ax, grid_axis="both", nbins_x=5, nbins_y=5)
    safe_set_lims(
        ax,
        x=(float(np.min(x)), float(np.max(x))),
        y=(float(np.min(resid_u)), float(np.max(resid_u))),
        pad_frac=0.08,
    )
    _save_diag_figure(fig, "diagnostic_residuals_vs_iv")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.hist(resid_u, bins=min(6, max(3, len(resid_u))), edgecolor="black")
    ax.set_xlabel(r"$\mathrm{Residual}\;/\;\mathrm{pH}$")
    ax.set_ylabel("Frequency")
    ax.set_title("Residual Histogram")
    clean_axis(ax, grid_axis="y", nbins_x=6, nbins_y=5)
    _save_diag_figure(fig, "diagnostic_residual_histogram")
    plt.close(fig)

    if should_plot_qq(len(resid_u), min_n=20):
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        stats.probplot(resid_u, dist="norm", plot=ax)
        ax.set_title("Residual normal Q-Q")
        clean_axis(ax, grid_axis="both", nbins_x=6, nbins_y=6)
        _save_diag_figure(fig, "diagnostic_residual_qq")
        plt.close(fig)
    else:
        warn_skipped_qq(len(resid_u), min_n=20)
        qq_base = figure_base_path(fig_key="diagnostic_residual_qq", kind="diagnostics")
        for ext in ("png", "pdf", "svg"):
            stale = qq_base.with_suffix(f".{ext}")
            if stale.exists():
                stale.unlink()

    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    rng = np.random.default_rng(42)
    for conc, group in repeats_df.groupby("NaCl Concentration (M)"):
        vals = pd.to_numeric(group["Apparent pKa"], errors="coerce").to_numpy(
            dtype=float
        )
        vals = vals[np.isfinite(vals)]
        jitter = rng.uniform(-0.012, 0.012, size=len(vals))
        ax.scatter(np.full(len(vals), conc) + jitter, vals, color="black", alpha=0.85)
    ax.set_xlabel(rf"${LABEL_NACL}$")
    ax.set_ylabel("Apparent pKa (replicates)")
    ax.set_title("Replicate Scatter by IV (precision)")
    clean_axis(ax, grid_axis="both", nbins_x=5, nbins_y=6)
    _save_diag_figure(fig, "diagnostic_replicate_jitter")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.plot(
        mean_df["NaCl Concentration (M)"],
        mean_df["SD pKa_app"],
        marker="o",
        color="black",
    )
    ax.set_xlabel(rf"${LABEL_NACL}$")
    ax.set_ylabel("SD of pKa_app")
    ax.set_title("SD vs IV (heteroscedasticity check)")
    clean_axis(ax, grid_axis="both", nbins_x=5, nbins_y=5)
    _save_diag_figure(fig, "diagnostic_sd_vs_iv")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.scatter(yhat_u, y, color="black")
    lo = float(min(np.min(yhat_u), np.min(y)))
    hi = float(max(np.max(yhat_u), np.max(y)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="gray")
    ax.set_xlabel("Predicted mean pKa_app")
    ax.set_ylabel("Measured mean pKa_app")
    ax.set_title("Parity Plot (means)")
    clean_axis(ax, grid_axis="both", nbins_x=5, nbins_y=5)
    _save_diag_figure(fig, "diagnostic_parity_measured_vs_predicted")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.stem(x, cooks_d, basefmt=" ")
    ax.axhline(4 / len(x), linestyle="--", color="gray", linewidth=1.2)
    ax.set_xlabel(rf"${LABEL_NACL}$")
    ax.set_ylabel("Cook's distance")
    ax.set_title("Influence Diagnostics (Cook's distance)")
    clean_axis(ax, grid_axis="both", nbins_x=5, nbins_y=5)
    _save_diag_figure(fig, "diagnostic_cooks_distance")
    plt.close(fig)


def _process_all_runs_robust(
    files: List[Tuple[str, float]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Process all files while recording skipped runs explicitly.

    Args:
        files (list[tuple[str, float]]): Pairs of standardized CSV path and NaCl
            concentration (mol dm^-3).

    Returns:
        tuple[list[dict], list[dict], list[dict]]: Successful run analyses,
        skipped-run records, and raw-checked index rows.
    """
    results: list[dict] = []
    skipped_runs: list[dict] = []
    raw_checked_rows: list[dict] = []

    raw_checked_dir = os.path.join("output", "ia", "raw_checked")
    os.makedirs(raw_checked_dir, exist_ok=True)

    import glob

    for stale in glob.glob(os.path.join(raw_checked_dir, "*.csv")):
        os.remove(stale)

    for file_path, nacl_conc in files:
        logging.info("Processing %s (NaCl %.2f M)", file_path, nacl_conc)
        df_raw = load_titration_data(file_path)
        try:
            runs = extract_runs(df_raw)
        except Exception as exc:  # noqa: BLE001
            skipped_runs.append(
                {
                    "source_file": os.path.basename(file_path),
                    "run": "<file>",
                    "reason": str(exc),
                }
            )
            continue

        for run_name, run_info in runs.items():
            run_df = run_info["df"].copy()
            source_file = os.path.basename(file_path)

            checked_name = (
                f"{nacl_conc:.1f}M_{source_file.replace('.csv', '')}"
                f"_{run_name.replace(' ', '_')}.csv"
            )
            checked_path = os.path.join(raw_checked_dir, checked_name)
            run_df.to_csv(checked_path, index=False)

            raw_checked_rows.append(
                {
                    "source_file": source_file,
                    "run": run_name,
                    "nacl_conc_M": nacl_conc,
                    "rows": int(len(run_df)),
                    "raw_checked_path": checked_path,
                }
            )

            if "Volume (cm^3)" not in run_df.columns or len(run_df) < 10:
                skipped_runs.append(
                    {
                        "source_file": source_file,
                        "run": run_name,
                        "reason": "Insufficient paired pH/volume rows",
                    }
                )
                continue

            try:
                analysis = analyze_titration(run_df, f"{nacl_conc:.1f}M - {run_name}")
                analysis["nacl_conc"] = nacl_conc
                analysis["source_file"] = source_file
                results.append(analysis)
            except Exception as exc:  # noqa: BLE001
                skipped_runs.append(
                    {
                        "source_file": source_file,
                        "run": run_name,
                        "reason": str(exc),
                    }
                )

    return results, skipped_runs, raw_checked_rows


def _build_iteration_results(
    all_results: list[dict],
    all_results_df: pd.DataFrame,
    iteration: IterationSpec,
) -> tuple[list[dict], pd.DataFrame]:
    """Select run subsets for one iteration definition.

    Args:
        all_results (list[dict]): All successful run-level analysis payloads.
        all_results_df (pandas.DataFrame): Consolidated run-level results table.
        iteration (IterationSpec): Iteration filter specification.

    Returns:
        tuple[list[dict], pandas.DataFrame]: Selected result payloads and the
        aligned filtered dataframe for this iteration.

    Raises:
        ValueError: If ``iteration.name`` is unsupported.
    """
    if iteration.name == "all_valid":
        selected_df = all_results_df.copy()
    elif iteration.name == "qc_pass":
        selected_df = all_results_df[
            all_results_df["Equivalence QC Pass"]
        ].copy()  # noqa: E712
    elif iteration.name == "strict_fit":
        selected_df = all_results_df[
            (all_results_df["Equivalence QC Pass"] == True)  # noqa: E712
            & (all_results_df["R2 (buffer fit)"] >= 0.98)
            & ((all_results_df["Slope (buffer fit)"] - 1.0).abs() <= 0.20)
        ].copy()
    else:
        raise ValueError(f"Unknown iteration: {iteration.name}")

    run_keys = {
        (
            str(row["Run"]),
            float(row["NaCl Concentration (M)"]),
            str(row["Source File"]),
        )
        for _, row in selected_df.iterrows()
    }

    selected_results = []
    for res in all_results:
        key = (
            str(res.get("run_name", "")),
            float(res.get("nacl_conc", np.nan)),
            str(res.get("source_file", "")),
        )
        if key in run_keys:
            selected_results.append(res)

    return selected_results, selected_df.reset_index(drop=True)


def _save_iteration_outputs(
    iteration: IterationSpec,
    iteration_results: list[dict],
    iteration_results_df: pd.DataFrame,
) -> Dict[str, object]:
    """Generate figures/CSVs for one iteration slice and collect metrics.

    Args:
        iteration (IterationSpec): Iteration metadata and label.
        iteration_results (list[dict]): Selected run-level payloads.
        iteration_results_df (pandas.DataFrame): Filtered run-level dataframe.

    Returns:
        dict[str, object]: Iteration diagnostics row with run counts and fit
        metrics.
    """
    from salty.schema import ResultColumns

    out_dir = os.path.join("output", "iterations", iteration.name)
    os.makedirs(out_dir, exist_ok=True)

    if iteration_results_df.empty:
        return {
            "iteration": iteration.name,
            "description": iteration.description,
            "n_runs": 0,
            "n_concentrations": 0,
            "fit_slope": np.nan,
            "fit_intercept": np.nan,
            "fit_r2": np.nan,
            "summary_plot": "",
        }

    stats_df = calculate_statistics(iteration_results_df)
    summary = build_summary_plot_data(stats_df, iteration_results_df)

    plot_titration_curves(iteration_results, output_dir=None, show_raw_pH=False)
    summary_plot = plot_statistical_summary(summary, output_dir=None)
    generate_all_qc_plots(iteration_results, iteration_results_df, output_dir=None)

    save_data_to_csv(iteration_results_df, stats_df, out_dir)

    cols = ResultColumns()
    pka_vals = pd.to_numeric(iteration_results_df[cols.pka_app], errors="coerce")
    pka_vals = pka_vals[np.isfinite(pka_vals)]

    fit = summary.get("fit", {})
    fit_slope = fit.get("m", np.nan)
    fit_intercept = fit.get("b", np.nan)
    fit_r2 = fit.get("r2", np.nan)

    diagnostics_df = pd.DataFrame(
        [
            {
                "iteration": iteration.name,
                "description": iteration.description,
                "n_runs": int(len(iteration_results_df)),
                "n_concentrations": int(iteration_results_df[cols.nacl].nunique()),
                "mean_pka_all_runs": (
                    float(np.mean(pka_vals)) if len(pka_vals) else np.nan
                ),
                "sd_pka_all_runs": (
                    float(np.std(pka_vals, ddof=1)) if len(pka_vals) >= 2 else np.nan
                ),
                "fit_slope": fit_slope,
                "fit_intercept": fit_intercept,
                "fit_r2": fit_r2,
                "summary_plot": summary_plot,
            }
        ]
    )
    diagnostics_df.to_csv(
        os.path.join(out_dir, "iteration_diagnostics.csv"), index=False
    )

    return diagnostics_df.iloc[0].to_dict()


def main(argv: list[str] | None = None):
    """Execute the end-to-end titration analysis pipeline.

    Returns:
        int: Exit code ``0`` on success or ``1`` if no valid results are
        obtained.
    """
    try:
        args = _parse_args(argv)
        _configure_logging()
        start_time = time.time()
        logging.info("Initializing titration analysis pipeline")

        files = _prepare_input_files()
        logging.info("Configured %d input files for analysis", len(files))
        logging.info("Exact IV levels enforced: %s M", EXACT_IV_LEVELS)

        if not files:
            logging.error("No input files discovered in data/raw")
            return 1

        step_start = time.time()
        results, skipped_runs, raw_checked_rows = _process_all_runs_robust(files)
        results = _filter_exact_iv_results(results)
        step_duration = time.time() - step_start
        logging.info(
            "Titration data files processing completed in %.2f seconds", step_duration
        )

        if not results:
            logging.error(
                "No valid results obtained from data processing. Terminating execution."
            )
            return 1

        logging.info("Successfully analyzed %d titration runs", len(results))
        logging.info("Skipped %d runs due to validation/fit issues", len(skipped_runs))
        total_data_points = sum(len(res["data"]) for res in results)
        logging.info("Total data points processed: %d", total_data_points)

        step_start = time.time()
        results_df = create_results_dataframe(results)
        step_duration = time.time() - step_start
        logging.info(
            "Results DataFrame creation completed in %.2f seconds", step_duration
        )
        logging.info("Results DataFrame shape: %s", results_df.shape)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Output directory ensured: %s", output_dir)

        ia_dir = os.path.join(output_dir, "ia")
        os.makedirs(ia_dir, exist_ok=True)

        pd.DataFrame(raw_checked_rows).to_csv(
            os.path.join(ia_dir, "raw_checked_index.csv"), index=False
        )
        pd.DataFrame(skipped_runs).to_csv(
            os.path.join(ia_dir, "skipped_runs.csv"), index=False
        )
        results_df.to_csv(
            os.path.join(ia_dir, "individual_results_rawfiles.csv"), index=False
        )

        grouped_rows = []
        for conc, group in results_df.groupby("NaCl Concentration (M)"):
            pka_vals = pd.to_numeric(group["Apparent pKa"], errors="coerce")
            pka_vals = pka_vals[np.isfinite(pka_vals)]
            n = int(len(pka_vals))
            mean_pka = float(np.mean(pka_vals)) if n else np.nan
            sd_pka = float(np.std(pka_vals, ddof=1)) if n >= 2 else np.nan
            sem_pka = float(sd_pka / np.sqrt(n)) if n >= 2 else np.nan
            half_range = (
                float((np.max(pka_vals) - np.min(pka_vals)) / 2.0) if n >= 2 else np.nan
            )

            propagated = pd.to_numeric(
                group["Uncertainty in Apparent pKa"], errors="coerce"
            )
            propagated = propagated[np.isfinite(propagated)]
            mean_prop = float(np.mean(propagated)) if len(propagated) else np.nan

            if np.isfinite(sem_pka) and np.isfinite(mean_prop):
                combined_unc = float(np.sqrt(sem_pka**2 + mean_prop**2))
            elif np.isfinite(mean_prop):
                combined_unc = mean_prop
            elif np.isfinite(sem_pka):
                combined_unc = sem_pka
            else:
                combined_unc = np.nan

            grouped_rows.append(
                {
                    "NaCl Concentration (M)": conc,
                    "n": n,
                    "Mean pKa_app": mean_pka,
                    "SD pKa_app": sd_pka,
                    "SEM pKa_app": sem_pka,
                    "Half-range": half_range,
                    "Mean propagated uncertainty": mean_prop,
                    "Combined uncertainty": combined_unc,
                }
            )

        processed_df = (
            pd.DataFrame(grouped_rows)
            .sort_values("NaCl Concentration (M)")
            .reset_index(drop=True)
        )
        processed_df = add_formatted_reporting_columns(
            processed_df,
            [("Mean pKa_app", "Combined uncertainty")],
        )
        processed_df.to_csv(
            os.path.join(ia_dir, "processed_summary_with_sd.csv"), index=False
        )
        _generate_additional_diagnostic_outputs(results_df, processed_df, ia_dir)

        iterations = [
            IterationSpec("all_valid", "All valid fitted runs"),
            IterationSpec("qc_pass", "Only runs with Equivalence QC Pass = True"),
            IterationSpec(
                "strict_fit",
                "QC pass + R2>=0.98 + |slope-1|<=0.20",
            ),
        ]

        iteration_rows: List[Dict[str, object]] = []
        for spec in iterations:
            iter_results, iter_results_df = _build_iteration_results(
                results, results_df, spec
            )
            row = _save_iteration_outputs(spec, iter_results, iter_results_df)
            iteration_rows.append(row)

        iteration_df = pd.DataFrame(iteration_rows)
        iteration_df.to_csv(os.path.join(ia_dir, "iteration_overview.csv"), index=False)

        if args.figures == "ia":
            profile_lookup = {
                "all_valid": "All valid fitted runs",
                "qc_pass": "Only runs with Equivalence QC Pass = True",
                "strict_fit": "QC pass + R2>=0.98 + |slope-1|<=0.20",
            }
            selected_spec = IterationSpec(args.profile, profile_lookup[args.profile])
            ia_results, ia_results_df = _build_iteration_results(
                results, results_df, selected_spec
            )
            if ia_results_df.empty:
                logging.warning(
                    "IA figure generation skipped: profile '%s' has no runs.",
                    args.profile,
                )
            else:
                generate_ia_figure_set(
                    results=ia_results,
                    results_df=ia_results_df,
                    output_dir=None,
                    iteration_output_dir=None,
                    summary_csv_path=os.path.join(
                        ia_dir, "processed_summary_with_sd.csv"
                    ),
                )
                logging.info("Generated IA Figure 1-5 set in output/figures")

        step_start = time.time()
        stats_df = calculate_statistics(results_df)
        step_duration = time.time() - step_start
        logging.info(
            "Statistical summary computation completed in %.2f seconds", step_duration
        )
        logging.info(
            "Statistical summary computed for %d concentration groups", len(stats_df)
        )

        print_statistics(stats_df, results_df)

        import glob

        png_files = glob.glob(os.path.join(output_dir, "*.png"))
        for png in png_files:
            os.remove(png)
        logging.info("Removed %d existing PNG files", len(png_files))

        step_start = time.time()
        titration_plot_paths_with = plot_titration_curves(
            results, output_dir=None, show_raw_pH=True
        )
        plot_titration_curves(results, output_dir=None, show_raw_pH=False)
        summary = build_summary_plot_data(stats_df, results_df)
        plot_statistical_summary(summary, output_dir=None)
        logging.info(
            "Generated %d individual titration curve figures in output/figures",
            len(titration_plot_paths_with),
        )

        logging.info("Generating QC and validation plots for method assessment...")
        qc_plot_paths = generate_all_qc_plots(results, results_df, output_dir=None)
        logging.info(
            "Generated %d QC/validation plots in output/figures", len(qc_plot_paths)
        )

        step_start = time.time()
        save_data_to_csv(results_df, stats_df, output_dir)
        stats_df.to_csv(
            os.path.join(ia_dir, "statistical_summary_rawfiles.csv"), index=False
        )

        summary = build_summary_plot_data(stats_df, results_df)
        fit = summary.get("fit", {})
        fit_df = pd.DataFrame(
            [
                {
                    "model": "linear",
                    "equation": (
                        f"pKa_app = {fit.get('m', np.nan):.6f}*[NaCl] + "
                        f"{fit.get('b', np.nan):.6f}"
                    ),
                    "slope": fit.get("m", np.nan),
                    "intercept": fit.get("b", np.nan),
                    "R2": fit.get("r2", np.nan),
                }
            ]
        )
        fit_df.to_csv(os.path.join(ia_dir, "fit_summary.csv"), index=False)

        step_duration = time.time() - step_start
        logging.info("CSV output completed in %.2f seconds", step_duration)

        total_duration = time.time() - start_time
        logging.info("Total execution time: %.2f seconds", total_duration)
        logging.info("Analysis pipeline completed successfully")
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.exception("Analysis pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
