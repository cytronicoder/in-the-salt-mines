import logging

import pandas as pd

from salty.data_processing import extract_runs, load_titration_data


def test_extract_runs_skips_volume_only_run(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    csv_path = tmp_path / "synthetic_runs.csv"
    pd.DataFrame(
        {
            "Run 1: Volume (cm^3)": [0.0, 1.0, 2.0, 3.0],
            "Run 1: pH": [3.0, 3.2, 3.8, 4.5],
            "Run 2: Volume (cm^3)": [0.0, 1.0, 2.0, 3.0],
            "Run 2: pH": [None, None, None, None],
            "Run 3: Volume (cm^3)": [0.0, 1.0, 2.0, 3.0],
            "Run 3: pH": [3.1, 3.3, 3.9, 4.6],
        }
    ).to_csv(csv_path, index=False)

    df = load_titration_data(str(csv_path))
    runs = extract_runs(df)

    assert "Run 2" not in runs
    assert any(
        "contains a Volume (cm^3) axis but no paired pH readings" in rec.message
        for rec in caplog.records
    )

    assert "Run 3" in runs
