import logging

import pandas as pd

from salty.analysis import process_all_files


def test_process_all_files_skips_insufficient_runs(caplog, tmp_path):
    caplog.set_level(logging.WARNING)
    csv_path = tmp_path / "synthetic_runs.csv"
    pd.DataFrame(
        {
            "Run 1: Volume (cm^3)": [
                0.0,
                0.3,
                0.6,
                0.9,
                1.2,
                1.5,
                1.8,
                2.1,
                2.4,
                2.7,
                3.0,
            ],
            "Run 1: pH": [3.0, 3.1, 3.2, 3.3, 3.5, 3.8, 4.2, 4.8, 5.6, 6.5, 7.0],
            "Run 2: Volume (cm^3)": [
                0.0,
                0.3,
                0.6,
                0.9,
                1.2,
                1.5,
                1.8,
                2.1,
                2.4,
                2.7,
                3.0,
            ],
            "Run 2: pH": [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        }
    ).to_csv(csv_path, index=False)

    files = [(str(csv_path), 1.0)]

    results = process_all_files(files)

    assert all("Run 2" not in r.get("run_name", "") for r in results)
    assert any(
        "Skipping run 'Run 2'" in rec.message
        or "contains a Volume (cm^3) axis but no paired pH readings" in rec.message
        for rec in caplog.records
    )
