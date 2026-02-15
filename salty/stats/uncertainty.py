"""Propagate and format uncertainty terms for IB-style titration reporting.

This module centralizes uncertainty constants, propagation rules, and value
formatting helpers used when translating analytical outputs into IA reporting
tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from salty.units import cm3_to_dm3

STANDARD_TEMP_C: float = 26.0
TEMP_TOLERANCE_C: float = 1.0
ACID_CONCENTRATION_M: float = 0.10
BASE_CONCENTRATION_M: float = 0.10
SAMPLE_VOLUME_CM3: float = 25.00
NACL_CONCENTRATIONS_M: list[float] = [0.00, 0.20, 0.40, 0.60, 0.80]
_EQUIPMENT_UNCERTAINTIES: Dict[str, Tuple[float, str]] = {
    "25.0 cm3 pipette": (0.06, "abs"),
    "50.0 cm3 burette": (0.10, "abs"),
    "burette reading": (0.02, "abs"),
    "250 cm3 beaker": (5.0, "pct"),
    "100 cm3 volumetric flask": (0.10, "abs"),
    "Vernier pH Sensor": (0.3, "abs"),
    "Digital thermometer": (0.1, "abs"),
    "Analytical balance": (0.01, "abs"),
}


@dataclass(frozen=True)
class Quantity:
    """Value container with explicit absolute systematic uncertainty.

    Args:
        value (float): Numerical value of a measured or derived quantity.
        uncertainty (float | None, optional): Absolute systematic uncertainty
            in the same unit as ``value``. Defaults to ``None``.
        unit (str, optional): Unit label for reporting. Defaults to ``""``.

    Attributes:
        value (float): Numerical value of the quantity.
        uncertainty (float | None): Absolute uncertainty in the same unit.
        unit (str): Unit label string.

    Note:
        This dataclass is immutable (`frozen=True`) to avoid accidental mutation of
        validated values in reporting pipelines.

    References:
        Structured value-plus-uncertainty representation for analytical reporting.
    """

    value: float
    uncertainty: Optional[float] = None
    unit: str = ""


def uncertainty_for_equipment(equipment: str, value: float | None = None) -> float:
    """Return absolute uncertainty for a named instrument specification.

    Args:
        equipment (str): Instrument name key in the equipment uncertainty
            table.
        value (float | None, optional): Value required when the table entry is
            percentage-based. Defaults to ``None``.

    Returns:
        float: Absolute uncertainty in the instrument's measurement unit.

    Raises:
        KeyError: If ``equipment`` is not present in the uncertainty table.
        ValueError: If a percent-based uncertainty is requested without
            ``value``.

    Note:
        Equipment keys are intentionally explicit to avoid silent fallback to
    ambiguous instrument assumptions.

    References:
        Instrument-specification uncertainty lookup tables in analytical methods.
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
    """Compute delivered-volume uncertainty from repeated burette readings.

    Args:
        reading_uncertainty (float): Absolute uncertainty per burette reading
            in cm^3.
        readings (int, optional): Number of readings used to form delivered
            volume (typically initial and final burette readings). Defaults to
            ``2``.

    Returns:
        float: Absolute delivered-volume uncertainty in cm^3.

    Raises:
        ValueError: If ``readings < 1``.

    Note:
        In this project, delivered volume usually derives from two readings:
        burette initial and burette final.

    References:
        Delivered-volume uncertainty from independent burette readings.
    """
    if readings < 1:
        raise ValueError("readings must be >= 1")
    return float(readings) * abs(float(reading_uncertainty))


def _round_uncertainty(u: float) -> Tuple[float, int]:
    """Round an uncertainty to IB-style significant figures.

    Args:
        u (float): Absolute uncertainty value.

    Returns:
        tuple[float, int]: Rounded uncertainty and decimal places used.

    Note:
        Use one significant figure by default, or two when the leading digit
        is 1.
    """
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
    """Round a value/uncertainty pair using IB-style significant figures.

    Args:
        value (float): Raw numerical value.
        uncertainty (float): Raw absolute uncertainty (same unit as
            ``value``).

    Returns:
        tuple[float, float]: ``(rounded_value, rounded_uncertainty)`` with
        aligned precision.

    Note:
        The uncertainty is rounded to one significant figure, except two when the
        leading digit is 1.

    References:
        IB/ISO-style uncertainty and value precision alignment conventions.
    """
    ru, ndigits = _round_uncertainty(abs(float(uncertainty)))
    if not math.isfinite(ru) or ru == 0:
        return float(value), float(uncertainty)
    return float(round(float(value), ndigits)), float(ru)


def _format_number_with_rounding(x: float, ndigits: int) -> str:
    """Format a number with fixed decimal places from uncertainty rules.

    Args:
        x (float): Number to format.
        ndigits (int): Decimal places to keep.

    Returns:
        str: Formatted numeric string.
    """
    xr = round(float(x), ndigits)
    if ndigits > 0:
        return f"{xr:.{ndigits}f}"
    return f"{xr:.0f}"


def format_value_with_uncertainty(
    value: float, uncertainty: float, unit: str = ""
) -> str:
    """Format a value with uncertainty as ``value +/- uncertainty [unit]``.

    Args:
        value (float): Numerical value to report.
        uncertainty (float): Absolute uncertainty paired with ``value`` in the
            same unit.
        unit (str, optional): Optional unit string appended to the output.
            Defaults to ``""``.

    Returns:
        str: Human-readable formatted result with aligned precision.

    Note:
        Falls back to compact general formatting when uncertainty is non-finite or
    rounds to zero.

    References:
        Scientific value-plus-uncertainty reporting notation.
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
    """Combine absolute uncertainty terms under a selected propagation rule.

    Args:
        terms (list[float]): Absolute uncertainty contributions in consistent
            units.
        method (str, optional): Combination rule, either ``"worst_case"``
            (arithmetic sum) or ``"quadrature"`` (root-sum-of-squares).
            Defaults to ``"worst_case"``.

    Returns:
        float: Combined absolute uncertainty in the same unit as ``terms``.

    Raises:
        ValueError: If ``method`` is unsupported.

    Note:
        ``worst_case`` is conservative and does not assume independence.
        ``quadrature`` assumes approximately independent terms.

    References:
        Worst-case sum and root-sum-of-squares uncertainty combination rules.
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
    """Propagate worst-case uncertainty for additive relationships.

    Args:
        uncertainties (Mapping[str, tuple[float, float, str]] | list[float]): Either
            a list of absolute uncertainties or a mapping
            ``{name: (value, uncertainty, unit)}``.

    Returns:
        dict | float: List mode returns combined absolute uncertainty (float).
        Mapping mode returns ``value``, ``uncertainty``, ``unit``, and an audit
        ``text`` expression.

    Raises:
        ValueError: If mapping-mode units are inconsistent.

    References:
        Linear worst-case propagation for sums and differences.

    Note:
        Mapping mode preserves units and returns explanatory text for audit trails.
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
    parts = " + ".join(f"{u:.3g}" for u in uncerts)
    text = f"Δy = {parts} = {unc:.3g}"
    return {"value": value, "uncertainty": unc, "unit": unit, "text": text}


def mul_div(
    values: Mapping[str, Tuple[float, float, str]] | list[float],
    uncertainties: Mapping[str, Tuple[float, float, str]] | list[float] | None = None,
) -> dict | float:
    """Propagate worst-case uncertainty for products and quotients.

    Args:
        values (Mapping[str, tuple[float, float, str]] | list[float]): Either
            list of values (list mode) or numerator mapping (mapping mode).
        uncertainties (Mapping[str, tuple] | list[float] | None, optional):
            In list mode, provide a list of absolute uncertainties aligned to
            ``values``. In mapping mode, provide denominator mapping with
            ``(value, uncertainty, unit)`` tuples. Defaults to ``None``.

    Returns:
        dict | float: List mode returns combined absolute uncertainty (float).
        Mapping mode returns output ``value`` and propagated ``uncertainty``.

    Raises:
        ValueError: If modes are mixed incorrectly, lengths mismatch, values
            are zero, or non-finite inputs are provided.

    Note:
        Uses worst-case relative-uncertainty addition:
        ``Δy/|y| = Σ |Δx_i/x_i|``.

    References:
        Relative uncertainty propagation for products and quotients.
    """
    if isinstance(values, list):
        if uncertainties is None or isinstance(uncertainties, dict):
            raise ValueError("uncertainties list required with list-based values.")
        if len(values) != len(uncertainties):
            raise ValueError("values and uncertainties must be the same length.")

        for v, u in zip(values, uncertainties):
            if not np.isfinite(v) or not np.isfinite(u):
                raise ValueError("Non-finite value or uncertainty")
            if v == 0:
                raise ValueError("Zero value not allowed in mul_div")
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
            raise ValueError("Non-finite value in multiplication/division")
        if v == 0:
            raise ValueError("Zero value not allowed in mul_div")

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
    """Propagate uncertainty through a power-law transform ``y = x^n``.

    Args:
        value (float): Base value ``x``.
        uncertainty (float): Absolute uncertainty ``Δx`` in the same unit as
            ``value``.
        exponent (float): Exponent ``n`` in ``y = x^n``.
        unit (str, optional): Optional output unit label. Defaults to ``""``.

    Returns:
        dict: Dictionary with propagated ``value``, ``uncertainty``, ``unit``,
        and audit ``text``.

    Raises:
        ValueError: If ``value == 0`` with non-zero uncertainty, or if
            ``value < 0`` with non-integer exponent.

    Note:
        Uses ``Δy/|y| = |n| * (Δx/|x|)`` where applicable.

    References:
        Power-law uncertainty propagation from first-order relative sensitivity.
    """
    value = float(value)
    uncertainty = abs(float(uncertainty))
    exponent = float(exponent)

    if value == 0 and uncertainty > 0:
        raise ValueError(
            "Cannot compute power uncertainty for value=0 with non-zero uncertainty."
        )

    if value == 0:
        out_val = 0.0
        out_unc = 0.0
        text = f"Δy/y = {abs(exponent):.3g}·(Δx/x) = 0 (value=0, uncertainty=0)"
    else:
        if value < 0:
            rounded_exp = round(exponent)
            is_int = np.isclose(exponent, rounded_exp, rtol=1e-9, atol=1e-6)
            if not is_int:
                raise ValueError(
                    "Cannot compute power for negative values with "
                    "non-integer exponents"
                )

        out_val = value**exponent
        rel = abs(exponent) * (uncertainty / abs(value))
        out_unc = abs(out_val) * rel
        a = f"{abs(exponent):.3g}"
        b = f"{uncertainty:.3g}"
        c = f"{abs(value):.3g}"
        text = f"Δy/y = {a}·(Δx/x) = {a}·({b}/{c})"
    return {"value": out_val, "uncertainty": out_unc, "unit": unit, "text": text}


def concentration_uncertainty(concentration: float) -> float:
    """Estimate systematic uncertainty in prepared NaCl concentration.

    Args:
        concentration (float): Target NaCl concentration in mol dm^-3.

    Returns:
        float: Absolute concentration uncertainty in mol dm^-3.

    Note:
        Assumes preparation from weighed NaCl in a 100.0 cm^3 volumetric flask.
        Worst-case relative propagation is applied to
        ``c = m / (M_r * V)``. For ``concentration == 0.0``, returns ``0.0``
        by definition. IA correspondence: this function mirrors the concentration
        preparation uncertainty narrative used for the independent variable.

    References:
        Ionic strength definition for 1:1 electrolytes and standard worst-case
        uncertainty propagation.
    """
    if concentration == 0.0 or not math.isfinite(concentration):
        return 0.0

    # NaCl molar mass: 58.44 g mol^-1
    mw_nacl = 58.44
    # Preparation volume: 100.0 cm^3 converted explicitly to dm^3
    volume_dm3 = cm3_to_dm3(100.0)
    # Required mass of NaCl for target concentration
    mass = concentration * volume_dm3 * mw_nacl

    # Uncertainty contributions from IA equipment table
    delta_mass = 0.01  # Analytical balance: ±0.01 g
    delta_volume = 0.0001  # 100 cm^3 volumetric flask: ±0.10 cm^3 = ±0.0001 dm^3

    # Check if mass is too small for reliable preparation
    # When mass < delta_mass, relative uncertainty > 100% (physically unrealistic)
    if mass < delta_mass:
        # For very low concentrations, the preparation method uncertainty
        # dominates. Use worst-case: uncertainty equals the target concentration.
        return float(concentration)

    # Relative uncertainties
    rel_unc_m = delta_mass / mass
    rel_unc_v = delta_volume / volume_dm3
    # Worst-case combination: c = m/(M·V) → Δc/c = Δm/m + ΔV/V
    rel_unc_c = combine_uncertainties([rel_unc_m, rel_unc_v], method="worst_case")
    return float(concentration * rel_unc_c)
