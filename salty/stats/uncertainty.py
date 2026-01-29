"""Systematic uncertainty propagation utilities for titration analysis."""

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
    """Store a value with its associated systematic uncertainty.

    Attributes:
        value: Numerical value of the measured or computed quantity.
        uncertainty: Systematic uncertainty associated with the value.
        unit: Unit string describing the quantity.
    """

    value: float
    uncertainty: Optional[float] = None
    unit: str = ""


def uncertainty_for_equipment(equipment: str, value: float | None = None) -> float:
    """Return the absolute systematic uncertainty for named equipment.

    Args:
        equipment: Name of the laboratory instrument.
        value: Measured value required for percent-based uncertainties.

    Returns:
        Absolute uncertainty for the specified equipment.

    Raises:
        KeyError: If the equipment name is unknown.
        ValueError: If a percent-based uncertainty is requested without a value.
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
    """Compute delivered-volume uncertainty from burette readings.

    Args:
        reading_uncertainty: Absolute uncertainty per burette reading.
        readings: Number of readings contributing to the delivered volume.

    Returns:
        Absolute systematic uncertainty in delivered volume.

    Raises:
        ValueError: If ``readings`` is less than one.
    """
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
    """Round a value and uncertainty using IB significant-figure rules.

    Args:
        value: Raw numerical value.
        uncertainty: Raw absolute uncertainty.

    Returns:
        A tuple of ``(rounded_value, rounded_uncertainty)`` where the
        uncertainty is rounded to one significant figure (two when the
        leading digit is 1) and the value is rounded to the same precision.
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
    """Format a value and uncertainty as a human-readable string.

    Args:
        value: Numerical value to report.
        uncertainty: Absolute uncertainty associated with the value.
        unit: Optional unit string appended to the formatted output.

    Returns:
        A formatted string in the form ``value ± uncertainty unit`` with
        appropriate rounding.
    """
    ru, ndigits = _round_uncertainty(abs(float(uncertainty)))
    if ru == 0 or not math.isfinite(ru):
        v = f"{value:.6g}"
        u = f"{uncertainty:.6g}"
        return f"{v} ± {u} {unit}".strip()

    v_str = _format_number_with_rounding(value, ndigits)
    u_str = _format_number_with_rounding(ru, ndigits)
    return f"{v_str} ± {u_str} {unit}".strip()


def combine_uncertainties(terms: list[float], method: str = "worst_case") -> float:
    """Combine absolute uncertainties with explicit propagation rules.

    Args:
        terms: List of absolute uncertainty contributions.
        method: ``"worst_case"`` (sum of absolute values) or ``"quadrature"``.

    Returns:
        The combined absolute uncertainty.

    Raises:
        ValueError: If an unsupported method is requested.
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
    """Propagate systematic uncertainty for addition or subtraction.

    Args:
        uncertainties: Either a list of absolute uncertainties or a mapping of
            named terms ``{name: (value, uncertainty, unit)}``.

    Returns:
        The combined uncertainty (list input) or a dictionary containing the
        combined value, uncertainty, unit, and explanatory text (mapping input).

    Raises:
        ValueError: If units are inconsistent in the mapping form.
    """
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
    """Propagate worst-case uncertainty for multiplication or division.

    This function supports two explicit calling conventions:
        1) ``mul_div(values_list, uncertainties_list)``
        2) ``mul_div(numerator_mapping, denominator_mapping)``

    Args:
        values: List of values or a mapping of numerator terms.
        uncertainties: List of uncertainties (list mode) or a mapping of
            denominator terms (mapping mode).

    Returns:
        The combined uncertainty (list mode) or a dictionary containing the
        combined value, uncertainty, and metadata (mapping mode).

    Raises:
        ValueError: If inputs are malformed or contain invalid values.
    """
    if isinstance(values, list):
        if uncertainties is None or isinstance(uncertainties, dict):
            raise ValueError("uncertainties list required with list-based values.")
        if len(values) != len(uncertainties):
            raise ValueError("values and uncertainties must be the same length.")

        for v, u in zip(values, uncertainties):
            if not np.isfinite(v) or not np.isfinite(u):
                raise ValueError(f"Non-finite value or uncertainty: {v}, {u}")
            if v == 0:
                raise ValueError(
                    f"Zero value not allowed in mul_div (relative uncertainty requires u/v): {v}"
                )
        rel = sum(abs(u / v) for v, u in zip(values, uncertainties))
        value = float(np.prod(values)) if values else math.nan
        return abs(value) * rel

    if uncertainties is None or isinstance(uncertainties, list):
        raise ValueError("Both numerator and denominator mappings are required.")

    num_vals = [float(v[0]) for v in values.values()]
    den_vals = [float(v[0]) for v in uncertainties.values()]
    num_uncs = [float(v[1]) for v in values.values()]
    den_uncs = [float(v[1]) for v in uncertainties.values()]

    for v in num_vals + den_vals:
        if not np.isfinite(v):
            raise ValueError(f"Non-finite value in multiplication/division: {v}")
        if v == 0:
            raise ValueError(
                f"Zero value not allowed in mul_div (relative uncertainty requires u/v): {v}"
            )

    for u in num_uncs + den_uncs:
        if not np.isfinite(u):
            raise ValueError(f"Non-finite uncertainty in multiplication/division: {u}")

    value = float(np.prod(num_vals) / np.prod(den_vals))

    rel_terms = [abs(float(v[1]) / float(v[0])) for v in values.values()]
    rel_terms += [abs(float(v[1]) / float(v[0])) for v in uncertainties.values()]
    rel = float(sum(rel_terms))
    return {"value": value, "uncertainty": abs(value) * rel}


def power(
    value: float, uncertainty: float, exponent: float, unit: str = ""
) -> dict | float:
    """Propagate worst-case uncertainty for a power-law relationship.

    Args:
        value: Base value.
        uncertainty: Absolute uncertainty in the base value.
        exponent: Power-law exponent.
        unit: Optional unit string for the resulting value.

    Returns:
        A dictionary containing the propagated value and uncertainty.

    Raises:
        ValueError: If the propagation would require complex arithmetic or if
            the base value is zero with non-zero uncertainty.
    """
    value = float(value)
    uncertainty = abs(float(uncertainty))
    exponent = float(exponent)

    if value == 0 and uncertainty > 0:
        raise ValueError(
            f"Cannot compute power uncertainty for value=0 with non-zero uncertainty={uncertainty}. "
            "Use endpoint propagation for worst-case bound."
        )

    if value == 0:
        out_val = 0.0
        out_unc = 0.0
        text = f"Δy/y = {abs(exponent):.3g}·(Δx/x) = 0 (value=0, uncertainty=0)"
    else:
        if value < 0:
            rounded_exp = round(exponent)
            if not np.isclose(exponent, rounded_exp, rtol=1e-9, atol=1e-6):
                raise ValueError(
                    f"Cannot compute power for negative value={value} with non-integer exponent={exponent}. "
                    "This would produce a complex number."
                )

        out_val = value**exponent
        rel = abs(exponent) * (uncertainty / abs(value))
        out_unc = abs(out_val) * rel
        text = f"Δy/y = {abs(exponent):.3g}·(Δx/x) = {abs(exponent):.3g}·({uncertainty:.3g}/{abs(value):.3g})"
    return {"value": out_val, "uncertainty": out_unc, "unit": unit, "text": text}


def concentration_uncertainty(concentration: float) -> float:
    """Compute systematic uncertainty for NaCl concentration preparation.

    Args:
        concentration: NaCl concentration in mol dm⁻³ (M).

    Returns:
        The absolute systematic uncertainty in the concentration, accounting
        for balance and volumetric flask limits.
    """
    if concentration == 0.0 or not math.isfinite(concentration):
        return 0.0

    mw = 58.44
    volume = 0.1
    mass = concentration * volume * mw

    delta_mass = 0.01
    delta_volume = 0.0001

    rel_unc_m = delta_mass / mass
    rel_unc_v = delta_volume / volume
    rel_unc_c = combine_uncertainties([rel_unc_m, rel_unc_v], method="worst_case")
    _, unc = round_value_to_uncertainty(concentration, float(concentration * rel_unc_c))
    return unc
