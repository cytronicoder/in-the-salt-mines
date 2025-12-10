#!/usr/bin/env python3
"""
Main script for running titration analysis.
"""

import os
import sys
import time
import logging

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
    process_all_files,
    create_results_dataframe,
    calculate_statistics,
    print_statistics,
)
from salty.plotting import (
    plot_titration_curves,
    plot_statistical_summary,
    save_data_to_csv,
)


def main():
    """Main execution function with comprehensive technical logging."""

    start_time = time.time()
    logging.info("Initializing titration analysis pipeline")

    files = [
        ("data/ms besant go brr - 0m nacl.csv", 0.0),
        ("data/ms besant go brr - 1m nacl.csv", 1.0),
    ]
    logging.info(f"Configured {len(files)} input files for analysis")

    step_start = time.time()
    results = process_all_files(files)
    step_duration = time.time() - step_start
    logging.info(
        f"Titration data files processing completed in {step_duration:.2f} seconds"
    )

    if not results:
        logging.error(
            "No valid results obtained from data processing. Terminating execution."
        )
        return 1

    logging.info(f"Successfully analyzed {len(results)} titration runs")
    total_data_points = sum(len(res["data"]) for res in results)
    logging.info(f"Total data points processed: {total_data_points}")

    step_start = time.time()
    results_df = create_results_dataframe(results)
    step_duration = time.time() - step_start
    logging.info(".2f")
    logging.info(f"Results DataFrame shape: {results_df.shape}")

    step_start = time.time()
    stats_df = calculate_statistics(results_df)
    step_duration = time.time() - step_start
    logging.info(".2f")
    logging.info(
        f"Statistical summary computed for {len(stats_df)} concentration groups"
    )

    print_statistics(stats_df, results_df)

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory ensured: {output_dir}")

    step_start = time.time()
    titration_plot_paths = plot_titration_curves(results, output_dir)
    summary_plot_path = plot_statistical_summary(stats_df, results_df, output_dir)
    step_duration = time.time() - step_start
    logging.info(".2f")
    logging.info(
        f"Generated {len(titration_plot_paths)} individual titration curve figures"
    )

    step_start = time.time()
    results_csv, stats_csv = save_data_to_csv(results_df, stats_df, output_dir)
    step_duration = time.time() - step_start
    logging.info(".2f")

    total_duration = time.time() - start_time
    logging.info(f"Total execution time: {total_duration:.2f} seconds")

    logging.info("Analysis pipeline completed successfully")
    logging.info("Generated output files:")
    for path in titration_plot_paths:
        logging.info(f"  - Titration curve: {path}")
    logging.info(f"  - Statistical summary: {summary_plot_path}")
    logging.info(f"  - Individual results CSV: {results_csv}")
    logging.info(f"  - Statistical summary CSV: {stats_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
