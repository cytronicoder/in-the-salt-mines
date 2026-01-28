#!/usr/bin/env python3
"""Command-line entry point for two-stage pKa_app titration analysis.

This script orchestrates the complete pipeline: data loading, run extraction,
two-stage apparent pKa analysis, statistical summarization, figure creation,
and CSV output. It is intended to be executed as a standalone program.
"""

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
    build_summary_plot_data,
    calculate_statistics,
    create_results_dataframe,
    print_statistics,
    process_all_files,
)
from salty.output import save_data_to_csv
from salty.plotting import plot_statistical_summary, plot_titration_curves


def main():
    """Execute the end-to-end titration analysis pipeline.

    Returns:
        Exit code ``0`` on success or ``1`` if no valid results are obtained.
    """
    try:
        start_time = time.time()
        logging.info("Initializing titration analysis pipeline")

        files = [
            ("data/ms besant go brr brr- 0.0m nacl.csv", 0.0),
            ("data/ms besant go brr brr v2- 0.0m nacl.csv", 0.0),
            ("data/ms besant go brr brr v2- 0.2m nacl.csv", 0.2),
            ("data/ms besant go brr brr v2- 0.4m nacl.csv", 0.4),
            # ("data/ms besant go brr brr- 0.5m nacl.csv", 0.5),
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
        logging.info(
            "Results DataFrame creation completed in %.2f seconds", step_duration
        )
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

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Output directory ensured: %s", output_dir)

        import glob

        png_files = glob.glob(os.path.join(output_dir, "*.png"))
        for png in png_files:
            os.remove(png)
        logging.info("Removed %d existing PNG files", len(png_files))

        with_raw_dir = os.path.join(output_dir, "with_raw")
        without_raw_dir = os.path.join(output_dir, "without_raw")
        os.makedirs(with_raw_dir, exist_ok=True)
        os.makedirs(without_raw_dir, exist_ok=True)

        step_start = time.time()
        titration_plot_paths_with = plot_titration_curves(
            results, with_raw_dir, show_raw_pH=True
        )
        plot_titration_curves(results, without_raw_dir, show_raw_pH=False)
        summary = build_summary_plot_data(stats_df, results_df)
        plot_statistical_summary(summary, with_raw_dir)
        plot_statistical_summary(summary, without_raw_dir)
        logging.info(
            "Generated %d individual titration curve figures in each folder",
            len(titration_plot_paths_with),
        )

        step_start = time.time()
        save_data_to_csv(results_df, stats_df, output_dir)
        step_duration = time.time() - step_start
        logging.info("CSV output completed in %.2f seconds", step_duration)

        total_duration = time.time() - start_time
        logging.info("Total execution time: %.2f seconds", total_duration)
        logging.info("Analysis pipeline completed successfully")
        return 0
    except Exception as exc:
        logging.exception("Analysis pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
