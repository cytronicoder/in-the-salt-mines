"""Test systematic uncertainty propagation utilities."""

import math

import pytest

from salty.uncertainty import add_subtract, mul_div, power, uncertainty_for_equipment


def test_equipment_uncertainty_lookup():
    """Verify equipment uncertainty lookup values."""
    u = uncertainty_for_equipment("25.0 cm3 pipette", 25.0)
    assert math.isclose(u, 0.06)
    u2 = uncertainty_for_equipment("250 cm3 beaker", 200.0)
    assert math.isclose(u2, 10.0)


def test_add_subtract_simple():
    """Verify worst-case propagation for addition/subtraction."""
    vals = {
        "a": (10.0, 0.2, "cm"),
        "b": (2.0, 0.05, "cm"),
    }
    out = add_subtract(vals)
    assert math.isclose(out["value"], 12.0)
    assert math.isclose(out["uncertainty"], 0.25)
    assert "Δy =" in out["text"]


def test_mul_div_simple():
    """Verify worst-case propagation for multiplication/division."""
    num = {"a": (2.0, 0.01, "cm"), "b": (3.0, 0.02, "cm")}
    den = {"c": (4.0, 0.01, "cm")}
    out = mul_div(num, den)
    assert math.isclose(out["value"], 1.5)
    rel = 0.01 / 2.0 + 0.02 / 3.0 + 0.01 / 4.0
    assert math.isclose(out["uncertainty"], abs(out["value"]) * rel)


def test_power_simple():
    """Verify worst-case propagation for power-law relationships."""
    out = power(2.0, 0.01, 2.0, "cm")
    assert math.isclose(out["value"], 4.0)
    rel = abs(2.0) * (0.01 / 2.0)
    assert math.isclose(out["uncertainty"], abs(4.0) * rel)


def test_power_negative_with_integer_exponent():
    """Verify power propagation for negative values with integer exponents."""
    out = power(-2.0, 0.01, 2.0, "cm")
    assert math.isclose(out["value"], 4.0)
    rel = abs(2.0) * (0.01 / abs(-2.0))
    assert math.isclose(out["uncertainty"], abs(4.0) * rel)


def test_power_negative_with_non_integer_exponent():
    """Raise errors for negative values with non-integer exponents."""
    with pytest.raises(
        ValueError,
        match="Cannot compute power for negative value.*non-integer exponent.*complex number",
    ):
        power(-2.0, 0.01, 0.5)

    with pytest.raises(
        ValueError,
        match="Cannot compute power for negative value.*non-integer exponent.*complex number",
    ):
        power(-2.0, 0.01, 1.5)
