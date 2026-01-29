import logging

from salty.analysis import process_all_files


def test_process_all_files_skips_insufficient_runs(caplog):
    caplog.set_level(logging.WARNING)
    files = [("data/ms besant go brr - 1m nacl.csv", 1.0)]

    results = process_all_files(files)

    assert all("Run 2" not in r.get("run_name", "") for r in results)
    assert any(
        "Skipping run 'Run 2'" in rec.message
        or "contains a Volume (cm³) axis but no paired pH readings" in rec.message
        for rec in caplog.records
    )
