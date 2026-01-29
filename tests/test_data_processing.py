import logging

from salty.data_processing import extract_runs, load_titration_data


def test_extract_runs_skips_volume_only_run(caplog):
    caplog.set_level(logging.WARNING)
    df = load_titration_data("data/ms besant go brr - 1m nacl.csv")
    runs = extract_runs(df)

    assert "Run 2" not in runs
    assert any(
        "contains a Volume (cm³) axis but no paired pH readings" in rec.message
        for rec in caplog.records
    )

    assert "Run 3" in runs
