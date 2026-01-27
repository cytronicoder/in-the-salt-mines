import numpy as np
import pandas as pd

from salty.analysis import calculate_statistics, create_results_dataframe
from salty.schema import ResultColumns


def test_create_results_dataframe_and_columns():
    results = [
        {
            "run_name": "run1",
            "nacl_conc": 0.0,
            "pka_app": 5.0,
            "pka_app_uncertainty": 0.1,
            "eq_qc_pass": True,
            "veq_used": 25.0,
            "source_file": "a.csv",
        },
        {
            "run_name": "run2",
            "nacl_conc": 0.0,
            "pka_app": 5.2,
            "pka_app_uncertainty": 0.2,
            "eq_qc_pass": True,
            "veq_used": 25.1,
            "source_file": "b.csv",
        },
        {
            "run_name": "run3",
            "nacl_conc": 0.1,
            "pka_app": 4.9,
            "pka_app_uncertainty": None,
            "eq_qc_pass": False,
            "veq_used": 24.0,
            "source_file": "c.csv",
        },
    ]

    df = create_results_dataframe(results)
    cols = ResultColumns()
    assert set(df.columns) >= {
        "Run",
        cols.nacl,
        "Apparent pKa (buffer regression)",
        cols.pka_unc,
        "Equivalence QC Pass",
        "Veq (used)",
        "Source File",
    }
    assert len(df) == 3


def test_calculate_statistics_values():
    results = [
        {"run_name": "r1", "nacl_conc": 0.0, "pka_app": 5.0},
        {"run_name": "r2", "nacl_conc": 0.0, "pka_app": 5.2},
        {"run_name": "r3", "nacl_conc": 0.1, "pka_app": 4.9},
    ]
    df = create_results_dataframe(results)
    stats = calculate_statistics(df)

    assert len(stats) == 2

    row0 = stats[stats[ResultColumns().nacl] == 0.0].iloc[0]
    assert np.isclose(row0["Mean Apparent pKa"], 5.1)
    assert int(row0["n"]) == 2

    row1 = stats[stats[ResultColumns().nacl] == 0.1].iloc[0]
    assert np.isclose(row1["Mean Apparent pKa"], 4.9)
    assert int(row1["n"]) == 1
