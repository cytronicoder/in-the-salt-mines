"""Tests for reporting-layer formatting and validation."""

import pandas as pd
import pytest

from salty.reporting import (
    add_formatted_reporting_columns,
    add_uncertainty_form_columns,
    format_value_to_uncertainty_decimals,
    uncertainty_decimal_places,
)
from salty.units import cm3_to_dm3


def test_uncertainty_decimal_places():
    assert uncertainty_decimal_places(0.02) == 2
    assert uncertainty_decimal_places(0.3) == 1
    assert uncertainty_decimal_places(1.0) == 0


def test_format_value_matches_uncertainty_decimals():
    assert format_value_to_uncertainty_decimals(4.5678, 0.02) == "4.57"
    assert format_value_to_uncertainty_decimals(12.345, 0.1) == "12.3"


def test_add_formatted_columns_per_datapoint_uncertainty():
    df = pd.DataFrame(
        {
            "Apparent pKa": [4.567, 4.567],
            "Uncertainty in Apparent pKa": [0.02, 0.1],
        }
    )

    out = add_formatted_reporting_columns(
        df,
        [("Apparent pKa", "Uncertainty in Apparent pKa")],
    )

    assert out.loc[0, "Apparent pKa (reported)"] == "4.57"
    assert out.loc[1, "Apparent pKa (reported)"] == "4.6"
    assert out.loc[0, "Uncertainty in Apparent pKa (reported)"] == "0.02"
    assert out.loc[1, "Uncertainty in Apparent pKa (reported)"] == "0.1"


def test_add_formatted_columns_fails_on_missing_uncertainty():
    df = pd.DataFrame(
        {
            "Apparent pKa": [4.567, 4.321],
            "Uncertainty in Apparent pKa": [0.02, None],
        }
    )

    with pytest.raises(ValueError, match="Uncertainty metadata missing/invalid"):
        add_formatted_reporting_columns(
            df,
            [("Apparent pKa", "Uncertainty in Apparent pKa")],
        )


def test_cm3_to_dm3_conversion():
    assert cm3_to_dm3(100.0) == 0.1


def test_add_uncertainty_form_columns():
    df = pd.DataFrame(
        {
            "Apparent pKa": [5.0],
            "Uncertainty in Apparent pKa": [0.2],
        }
    )
    out = add_uncertainty_form_columns(
        df,
        [("Apparent pKa", "Uncertainty in Apparent pKa")],
    )
    assert out.loc[0, "Apparent pKa fractional uncertainty"] == 0.04
    assert out.loc[0, "Apparent pKa percentage uncertainty (%)"] == 4.0
