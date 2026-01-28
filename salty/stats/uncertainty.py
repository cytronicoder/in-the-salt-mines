"""
Uncertainty propagation utilities following IB DP worst-case rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

_EQUIPMENT_UNCERTAINTIES: Dict[str, Tuple[float, str]] = {
    "25.0 cm3 pipette": (0.06, "abs"),
    "50.0 cm3 burette": (0.05, "abs"),
    "250 cm3 beaker": (5.0, "pct"),
    "100 cm3 volumetric flask": (0.10, "abs"),
    "Vernier pH Sensor PH-BTA": (0.2, "abs"),
    "Digital thermometer": (0.1, "abs"),
    "Analytical balance": (0.01, "abs"),
}


@dataclass(frozen=True)
class Quantity:
    value: float
    uncertainty: Optional[float] = None
    unit: str = ""


def uncertainty_for_equipment(equipment: str, value: float | None = None) -> float:
    """
    Return absolute uncertainty for named lab equipment.

    Examples: 'Vernier pH Sensor PH-BTA', '50.0 cm3 burette', '250 cm3 beaker'.
    """
    entry = _EQUIPMENT_UNCERTAINTIES.get(equipment)
    if entry is None:
        raise KeyError(f"No uncertainty available for equipment '{equipment}'.")
    u, typ = entry
    if typ == "abs":
        return float(u)
    if value is None:
        raise ValueError("Value is required for percent-based equipment uncertainty.")
    return float((u / 100.0) * float(value))


def burette_delivered_uncertainty(
    reading_uncertainty: float, readings: int = 2
) -> float:
    if readings < 1:
        raise ValueError("readings must be >= 1")
    return float(readings) * abs(float(reading_uncertainty))


def _round_uncertainty(u: float) -> Tuple[float, int]:
    if u <= 0 or not math.isfinite(u):
        return u, 0

    u = abs(float(u))
    exponent = math.floor(math.log10(u))
    leading = u / (10**exponent)

    sig_figs = 2 if 1.0 <= leading < 2.0 else 1
    ndigits = sig_figs - 1 - exponent

    ru = round(u, ndigits)

    if ru == 0:
        ndigits = sig_figs - exponent
        ru = round(u, ndigits)

    return float(ru), int(ndigits)


def round_value_to_uncertainty(value: float, uncertainty: float) -> Tuple[float, float]:
    """
    Round a value and uncertainty using IB s.f. rules:
    - uncertainty to 1 s.f. (2 if leading digit is 1)
    - value rounded to the same decimal place
    """
    ru, ndigits = _round_uncertainty(abs(float(uncertainty)))
    if not math.isfinite(ru) or ru == 0:
        return float(value), float(uncertainty)
    return float(round(float(value), ndigits)), float(ru)


def _format_number_with_rounding(x: float, ndigits: int) -> str:
    xr = round(float(x), ndigits)
    if ndigits > 0:
        return f"{xr:.{ndigits}f}"
    return f"{xr:.0f}"


def format_value_with_uncertainty(
    value: float, uncertainty: float, unit: str = ""
) -> str:
    ru, ndigits = _round_uncertainty(abs(float(uncertainty)))
    if ru == 0 or not math.isfinite(ru):
        v = f"{value:.6g}"
        u = f"{uncertainty:.6g}"
        return f"{v} ± {u} {unit}".strip()

    v_str = _format_number_with_rounding(value, ndigits)
    u_str = _format_number_with_rounding(ru, ndigits)
    return f"{v_str} ± {u_str} {unit}".strip()


def combine_uncertainties(terms: list[float], method: str = "worst_case") -> float:
    """
    Combine absolute uncertainties using a named method.

    method:
      - "worst_case": sum of absolute uncertainties (IB default)
      - "quadrature": sqrt(sum of squares), only if explicitly requested
    """
    vals = [abs(float(t)) for t in terms if math.isfinite(t) and abs(float(t)) > 0]
    if not vals:
        return math.nan
    if method == "quadrature":
        return float(math.sqrt(sum(v**2 for v in vals)))
    if method != "worst_case":
        raise ValueError("method must be 'worst_case' or 'quadrature'")
    return float(sum(vals))


def add_subtract(
    uncertainties: Mapping[str, Tuple[float, float, str]] | list[float],
) -> dict | float:
    """Worst-case uncertainty for addition/subtraction."""
    if isinstance(uncertainties, list):
        return float(sum(abs(float(u)) for u in uncertainties))

    units = {str(v[2]).strip() for v in uncertainties.values() if str(v[2]).strip()}
    if len(units) > 1:
        raise ValueError(f"Addition/subtraction requires consistent units; got {units}")
    unit = units.pop() if units else ""

    values = [float(v[0]) for v in uncertainties.values()]
    uncerts = [abs(float(v[1])) for v in uncertainties.values()]
    value = float(sum(values))
    unc = float(sum(uncerts))
    text = "Δy = " + " + ".join(f"{u:.3g}" for u in uncerts) + f" = {unc:.3g}"
    return {"value": value, "uncertainty": unc, "unit": unit, "text": text}


def mul_div(
    values: Mapping[str, Tuple[float, float, str]] | list[float],
    uncertainties: Mapping[str, Tuple[float, float, str]] | list[float] | None = None,
) -> dict | float:
    """Worst-case uncertainty for multiplication/division."""
    if isinstance(values, list):
        if uncertainties is None or isinstance(uncertainties, dict):
            raise ValueError("uncertainties list required with list-based values.")
        if len(values) != len(uncertainties):
            raise ValueError("values and uncertainties must be the same length.")
        # Guard against zero or non-finite values
        # Zero values are not allowed because relative uncertainty (u/v) requires division by value
        for v, u in zip(values, uncertainties):
            if not np.isfinite(v) or not np.isfinite(u):
                raise ValueError(f"Non-finite value or uncertainty: {v}, {u}")
            if v == 0:
                raise ValueError(f"Zero value not allowed in mul_div (relative uncertainty requires u/v): {v}")
        rel = sum(abs(u / v) for v, u in zip(values, uncertainties))
        value = float(np.prod(values)) if values else math.nan
        return abs(value) * rel

    if uncertainties is None or isinstance(uncertainties, list):
        raise ValueError("Both numerator and denominator mappings are required.")

    num_vals = [float(v[0]) for v in values.values()]
    den_vals = [float(v[0]) for v in uncertainties.values()]
    num_uncs = [float(v[1]) for v in values.values()]
    den_uncs = [float(v[1]) for v in uncertainties.values()]

    # Guard against zero or non-finite values in denominators and numerators
    # Zero values are not allowed because relative uncertainty (u/v) requires division by value
    for v in num_vals + den_vals:
        if not np.isfinite(v):
            raise ValueError(f"Non-finite value in multiplication/division: {v}")
        if v == 0:
            raise ValueError(f"Zero value not allowed in mul_div (relative uncertainty requires u/v): {v}")

    # Guard against non-finite uncertainties
    for u in num_uncs + den_uncs:
        if not np.isfinite(u):
            raise ValueError(f"Non-finite uncertainty in multiplication/division: {u}")

    value = float(np.prod(num_vals) / np.prod(den_vals))

    rel_terms = [abs(float(v[1]) / float(v[0])) for v in values.values()]
    rel_terms += [abs(float(v[1]) / float(v[0])) for v in uncertainties.values()]
    rel = float(sum(rel_terms))
    return {"value": value, "uncertainty": abs(value) * rel}


def power(value: float, uncertainty: float, exponent: float, unit: str = "") -> dict | float:
    """Worst-case uncertainty propagation for powers."""
    value = float(value)
    uncertainty = abs(float(uncertainty))
    exponent = float(exponent)
    
    # Reject value==0 with non-zero uncertainty (similar to mul_div's zero guard)
    # Computing worst-case uncertainty requires evaluating endpoints (value ± uncertainty)^exponent
    if value == 0 and uncertainty > 0:
        raise ValueError(
            f"Cannot compute power uncertainty for value=0 with non-zero uncertainty={uncertainty}. "
            "Use endpoint propagation for worst-case bound."
        )
    
    if value == 0:
        # value==0 and uncertainty==0
        out_val = 0.0
        out_unc = 0.0
        text = f"Δy/y = {abs(exponent):.3g}·(Δx/x) = 0 (value=0, uncertainty=0)"
    else:
        out_val = value**exponent
        rel = abs(exponent) * (uncertainty / abs(value))
        out_unc = abs(out_val) * rel
        text = f"Δy/y = {abs(exponent):.3g}·(Δx/x) = {abs(exponent):.3g}·({uncertainty:.3g}/{abs(value):.3g})"
    return {"value": out_val, "uncertainty": out_unc, "unit": unit, "text": text}


def concentration_uncertainty(concentration: float) -> float:
    """
    Absolute uncertainty in NaCl concentration based on balance + flask uncertainties.

    concentration in mol dm^-3 (M).
    """
    if concentration == 0.0 or not math.isfinite(concentration):
        return 0.0

    mw = 58.44  # g/mol
    volume = 0.1  # L
    mass = concentration * volume * mw  # g

    delta_mass = 0.01  # g
    delta_volume = 0.0001  # L (0.10 cm^3)

    rel_unc_m = delta_mass / mass
    rel_unc_v = delta_volume / volume
    rel_unc_c = combine_uncertainties([rel_unc_m, rel_unc_v], method="worst_case")
    _, unc = round_value_to_uncertainty(concentration, float(concentration * rel_unc_c))
    return unc
