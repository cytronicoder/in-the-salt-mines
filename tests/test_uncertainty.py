import math
from salty.uncertainty import add_subtract, mul_div, power, uncertainty_for_equipment


def test_equipment_uncertainty_lookup():
    u = uncertainty_for_equipment("25.0 cm3 pipette", 25.0)
    assert math.isclose(u, 0.06)
    u2 = uncertainty_for_equipment("250 cm3 beaker", 200.0)
    # 250 cm3 beaker has 5% -> for value 200 the absolute is 10
    assert math.isclose(u2, 10.0)


def test_add_subtract_simple():
    vals = {
        "a": (10.0, 0.2, "cm"),
        "b": (2.0, 0.05, "cm"),
    }
    out = add_subtract(vals)
    assert math.isclose(out["value"], 12.0)
    assert math.isclose(out["uncertainty"], 0.25)
    assert "Δy =" in out["text"]


def test_mul_div_simple():
    num = {"a": (2.0, 0.01, "cm"), "b": (3.0, 0.02, "cm")}
    den = {"c": (4.0, 0.01, "cm")}
    out = mul_div(num, den)
    # y = (2*3)/4 = 1.5
    assert math.isclose(out["value"], 1.5)
    # relative uncertainties = 0.01/2 + 0.02/3 + 0.01/4
    rel = 0.01/2.0 + 0.02/3.0 + 0.01/4.0
    assert math.isclose(out["uncertainty"], abs(out["value"]) * rel)


def test_power_simple():
    out = power(2.0, 0.01, 2.0, "cm")
    assert math.isclose(out["value"], 4.0)
    rel = abs(2.0) * (0.01 / 2.0)
    assert math.isclose(out["uncertainty"], abs(4.0) * rel)
