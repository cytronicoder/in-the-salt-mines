#!/usr/bin/env python3
"""
Main script for running titration analysis.
"""

# Pipeline overview (README-style):
# 1) Load Logger Pro CSVs and split into individual runs.
# 2) Forward-fill cumulative volume and aggregate readings into volume steps
#    using median pH from the tail of each step (robust equilibrium estimate).
# 3) Interpolate pH(V) (linear by default), compute derivatives with uneven
#    spacing, and detect equivalence with QC guardrails.
# 4) Estimate pKa using buffer-region regression and report a half-equivalence
#    check, along with uncertainties and HH-fit rescue values if needed.
# 5) Export per-run results, statistics across replicates, and diagnostics plots.

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("titration_analysis.log", mode="w"),
    ],
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from salty.analysis import (
    calculate_statistics,
    create_results_dataframe,
    print_statistics,
    process_all_files,
)
from salty.plotting import (
    plot_statistical_summary,
    plot_titration_curves,
    save_data_to_csv,
)


def main():
    """Main execution function with comprehensive technical logging."""

    start_time = time.time()
    logging.info("Initializing titration analysis pipeline")

    files = [
        ("data/ms besant go brr brr- 0.0m nacl.csv", 0.0),
        ("data/ms besant go brr brr v2- 0.0m nacl.csv", 0.0),
        ("data/ms besant go brr brr v2- 0.2m nacl.csv", 0.2),
        ("data/ms besant go brr brr v2- 0.4m nacl.csv", 0.4),
        ("data/ms besant go brr brr- 0.5m nacl.csv", 0.5),
        ("data/ms besant go brr brr v2- 0.6m nacl.csv", 0.6),
        ("data/ms besant go brr brr- 0.8m nacl.csv", 0.8),
        ("data/ms besant go brr - 1m nacl.csv", 1.0),
        ("data/ms besant go brr brr v2- 1.0m nacl.csv", 1.0),
    ]
    logging.info("Configured %d input files for analysis", len(files))

    step_start = time.time()
    results = process_all_files(files)
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
    total_data_points = sum(len(res["data"]) for res in results)
    logging.info("Total data points processed: %d", total_data_points)

    step_start = time.time()
    results_df = create_results_dataframe(results)
    step_duration = time.time() - step_start
    logging.info("Results DataFrame creation completed in %.2f seconds", step_duration)
    logging.info("Results DataFrame shape: %s", results_df.shape)

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

    # Check data coverage: ensure at least 3 runs per concentration
    underpowered = stats_df[stats_df['n'] < 3]
    if not underpowered.empty:
        # ensure output directory exists before writing report
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        msg_lines = ["Data coverage issues detected:\n"]
        for _, row in underpowered.iterrows():
            conc = row['NaCl Concentration (M)']
            n = int(row['n'])
            msg_lines.append(f" - {conc} M has only {n} valid runs (n<3)\n")
            subset = results_df[results_df['NaCl Concentration (M)'] == conc]
            for _, r in subset.iterrows():
                msg_lines.append(
                    f"     {r['Run']}: pKa={r['pKa (buffer regression)']} (QC: {r['Equivalence QC Pass']})\n"
                )
        report_path = os.path.join(output_dir, 'data_coverage_report.txt')
        with open(report_path, 'w') as fh:
            fh.writelines(msg_lines)
        logging.warning('Insufficient replication for some concentrations; report written to %s', report_path)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Output directory ensured: %s", output_dir)

    step_start = time.time()
    titration_plot_paths = plot_titration_curves(results, output_dir)
    summary_plot_path = plot_statistical_summary(stats_df, results_df, output_dir)
    step_duration = time.time() - step_start
    logging.info(
        "Generated %d individual titration curve figures", len(titration_plot_paths)
    )

    step_start = time.time()
    results_csv, stats_csv = save_data_to_csv(results_df, stats_df, output_dir)
    step_duration = time.time() - step_start

    total_duration = time.time() - start_time
    logging.info(f"Total execution time: {total_duration:.2f} seconds")

    logging.info("Analysis pipeline completed successfully")
    logging.info("Generated output files:")
    logging.info("Analysis pipeline completed successfully")
    logging.info("Generated output files:")
    for path in titration_plot_paths:
        logging.info("  - Titration curve: %s", path)
    logging.info("  - Statistical summary: %s", summary_plot_path)
    logging.info("  - Individual results CSV: %s", results_csv)
    logging.info("  - Statistical summary CSV: %s", stats_csv)


if __name__ == "__main__":
    sys.exit(main())
