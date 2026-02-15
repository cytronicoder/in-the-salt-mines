"""Tests for export-layer uncertainty formatting behavior."""

import pandas as pd
import pytest

from salty.output import save_data_to_csv


def test_save_data_to_csv_fails_when_uncertainty_missing(tmp_path):
    results_df = pd.DataFrame(
        {
            "Run": ["r1", "r2"],
            "NaCl Concentration (M)": [0.0, 0.2],
            "Apparent pKa": [4.57, 4.44],
            "Uncertainty in Apparent pKa": [0.02, None],
            "Veq (used)": [25.0, 24.9],
            "Veq uncertainty (ΔVeq)": [0.1, 0.1],
        }
    )
    stats_df = pd.DataFrame(
        {
            "NaCl Concentration (M)": [0.0, 0.2],
            "Mean Apparent pKa": [4.57, 4.44],
            "Uncertainty": [0.02, 0.03],
            "n": [1, 1],
        }
    )

    with pytest.raises(ValueError, match="Uncertainty metadata missing/invalid"):
        save_data_to_csv(results_df, stats_df, output_dir=str(tmp_path))
