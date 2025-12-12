"""
Analyzing titration data and finding equivalence points.
"""

import numpy as np
import pandas as pd

from .data_processing import calculate_derivatives, extract_runs, load_titration_data


def find_equivalence_point(df, time_col="Time (min)"):
    """
    Finds the equivalence point time where 1st derivative is max.

    Args:
        df (pd.DataFrame): DataFrame with calculated derivatives.
        time_col (str, optional): Name of the time column (default: 'Time (min)').

    Returns:
        tuple: (equivalence time, equivalence pH).
    """
    max_idx = df["dpH/dt"].idxmax()
    eq_time = df.loc[max_idx, time_col]
    eq_pH = df.loc[max_idx, "pH_smooth"]

    return eq_time, eq_pH


def analyze_titration(df, run_name):
    """
    Analyzes a single titration run.

    Args:
        df (pd.DataFrame): DataFrame with time and pH data.
        run_name (str): Name identifier for this run.

    Returns:
        dict: Analysis results including equivalence point, half-equivalence point, and pKa.
    """
    df = calculate_derivatives(df)
    eq_time, eq_pH = find_equivalence_point(df)

    half_eq_time = eq_time / 2
    half_eq_pH = np.interp(half_eq_time, df["Time (min)"], df["pH_smooth"])

    return {
        "run_name": run_name,
        "eq_time": eq_time,
        "eq_pH": eq_pH,
        "half_eq_time": half_eq_time,
        "half_eq_pH": half_eq_pH,
        "data": df,
    }


def process_all_files(file_list):
    """
    Process multiple titration data files.

    Args:
        file_list (list of tuples): List of (filepath, nacl_concentration) tuples.

    Returns:
        list: List of analysis result dictionaries.
    """
    results = []

    for filepath, nacl_conc in file_list:
        print(f"Processing {filepath} (NaCl: {nacl_conc} M)...")
        try:
            df_raw = load_titration_data(filepath)
            runs = extract_runs(df_raw)

            for run_name, run_df in runs.items():
                print(f"  Analyzing {run_name}...")
                if len(run_df) < 10:
                    print(f"    Skipping {run_name} (not enough data points)")
                    continue

                analysis = analyze_titration(run_df, f"{nacl_conc}M - {run_name}")
                analysis["nacl_conc"] = nacl_conc
                results.append(analysis)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print("\nAnalysis Complete.")
    return results


def create_results_dataframe(results):
    """
    Convert analysis results to a pandas DataFrame.

    Args:
        results (list): List of analysis result dictionaries.

    Returns:
        pd.DataFrame: DataFrame with organized results.
    """
    results_df = pd.DataFrame(
        [
            {
                "NaCl Concentration (M)": r["nacl_conc"],
                "Run": r["run_name"],
                "Equivalence Time (min)": r["eq_time"],
                "Equivalence pH": r["eq_pH"],
                "Half-Equivalence Time (min)": r["half_eq_time"],
                "Half-Equivalence pH (pKa)": r["half_eq_pH"],
            }
            for r in results
        ]
    )
    return results_df


def calculate_statistics(results_df):
    """
    Calculate statistical summaries of the results.

    Args:
        results_df (pd.DataFrame): Results DataFrame from create_results_dataframe.

    Returns:
        pd.DataFrame: Statistical summary with mean, SD, SEM, and count.
    """
    stats_df = (
        results_df.groupby("NaCl Concentration (M)")["Half-Equivalence pH (pKa)"]
        .agg([("Mean pKa", "mean"), ("SD", "std"), ("n", "count")])
        .reset_index()
    )

    stats_df["SEM"] = stats_df["SD"] / np.sqrt(stats_df["n"])

    return stats_df


def print_statistics(stats_df, results_df):
    """
    Print statistical summary to console.

    Args:
        stats_df (pd.DataFrame): Statistical summary DataFrame.
        results_df (pd.DataFrame): Individual results DataFrame.

    Returns:
        None
    """
    print("\n=== Statistical Summary ===")
    print(stats_df.to_string(index=False))
    print("\nIndividual Measurements:")
    for conc in stats_df["NaCl Concentration (M)"]:
        subset = results_df[results_df["NaCl Concentration (M)"] == conc]
        print(f"\n{conc} M NaCl:")
        for _, row in subset.iterrows():
            print(f"  {row['Run']}: pKa = {row['Half-Equivalence pH (pKa)']:.4f}")
