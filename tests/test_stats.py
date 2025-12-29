import pandas as pd
import numpy as np

from salty.analysis import create_results_dataframe, calculate_statistics


def test_create_results_dataframe_and_columns():
    results = [
        {"run_name": "run1", "nacl_conc": 0.0, "pka_reg": 5.0, "pka_uncertainty": 0.1, "eq_qc_pass": True, "veq_used": 25.0, "source_file": "a.csv"},
        {"run_name": "run2", "nacl_conc": 0.0, "pka_reg": 5.2, "pka_uncertainty": 0.2, "eq_qc_pass": True, "veq_used": 25.1, "source_file": "b.csv"},
        {"run_name": "run3", "nacl_conc": 0.1, "pka_reg": 4.9, "pka_uncertainty": None, "eq_qc_pass": False, "veq_used": 24.0, "source_file": "c.csv"},
    ]

    df = create_results_dataframe(results)
    assert set(df.columns) >= {
        "Run",
        "NaCl Concentration (M)",
        "pKa (buffer regression)",
        "pKa uncertainty (ΔpKa)",
        "Equivalence QC Pass",
        "Veq (used)",
        "Source File",
    }
    assert len(df) == 3


def test_calculate_statistics_values():
    results = [
        {"run_name": "r1", "nacl_conc": 0.0, "pka_reg": 5.0},
        {"run_name": "r2", "nacl_conc": 0.0, "pka_reg": 5.2},
        {"run_name": "r3", "nacl_conc": 0.1, "pka_reg": 4.9},
    ]
    df = create_results_dataframe(results)
    stats = calculate_statistics(df)

    # Expect two groups
    assert len(stats) == 2

    row0 = stats[stats["NaCl Concentration (M)"] == 0.0].iloc[0]
    assert np.isclose(row0["Mean pKa"], 5.1)
    # With uncertainties 0.1 and 0.2, Δmean = (0.1 + 0.2) / 2 = 0.15
    # If no uncertainties provided in the input, the statistic will have NaN uncertainty
    assert int(row0["n"]) == 2

    row1 = stats[stats["NaCl Concentration (M)"] == 0.1].iloc[0]
    assert np.isclose(row1["Mean pKa"], 4.9)
    assert int(row1["n"]) == 1
