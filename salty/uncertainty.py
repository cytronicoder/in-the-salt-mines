"""
IB DP-style worst-case uncertainty propagation utilities.

Implements worst-case (absolute) propagation using standard IB rules:
- Addition/subtraction: Δy = Σ Δx
- Multiplication/division: Δy / |y| = Σ (Δx / |x|)
- Powers: Δy / |y| = |n| (Δa / |a|) for y = a^n (with defined handling when a = 0)

This module also provides IB-style formatting:
- Uncertainty rounded to 1 significant figure (2 if leading digit is 1)
- Value rounded to the same place value as the uncertainty

Equipment uncertainties can be retrieved from a small editable lookup table.
For burettes, note the distinction between reading uncertainty and delivered-volume uncertainty
(two readings): use `burette_delivered_uncertainty(...)`.
"""

from __future__ import annotations

# CHANGELOG:
# - Added shared rounding helper to align values with IB uncertainty s.f. rules.
# - Added configurable uncertainty combiner to keep worst-case behavior consistent.
# - Clarified delivered-volume uncertainty usage and exposed helper for rounding outputs.

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Mapping
import math


# (uncertainty, 'abs'|'pct')
_EQUIPMENT_UNCERTAINTIES: Dict[str, Tuple[float, str]] = {
    "25.0 cm3 pipette": (0.06, "abs"),
    "50.0 cm3 burette": (0.05, "abs"),  # reading uncertainty (per reading)
    "250 cm3 beaker": (5.0, "pct"),
    "100 cm3 volumetric flask": (0.10, "abs"),
    "Vernier pH Sensor PH-BTA": (0.2, "abs"),
    "Digital thermometer": (0.1, "abs"),
    "Analytical balance": (0.01, "abs"),
}


@dataclass(frozen=True)
class Quantity:
    value: float
    uncertainty: Optional[float] = None  # absolute uncertainty
    unit: str = ""


def get_equipment_uncertainty(equipment: str, value: float) -> Optional[float]:
    entry = _EQUIPMENT_UNCERTAINTIES.get(equipment)
    if entry is None:
        return None
    u, typ = entry
    if typ == "abs":
        return float(u)
    return float((u / 100.0) * value)


def set_equipment_uncertainty(
    equipment: str, uncertainty: float, typ: str = "abs"
) -> None:
    if typ not in {"abs", "pct"}:
        raise ValueError("typ must be 'abs' or 'pct'")
    _EQUIPMENT_UNCERTAINTIES[equipment] = (float(uncertainty), typ)


def burette_delivered_uncertainty(
    reading_uncertainty: float, readings: int = 2
) -> float:
    if readings < 1:
        raise ValueError("readings must be >= 1")
    return float(readings) * abs(float(reading_uncertainty))


def uncertainty_for_equipment(equipment: str, value: float) -> float:
    u = get_equipment_uncertainty(equipment, value)
    if u is None:
        raise ValueError(f"Unknown equipment '{equipment}'")
    return float(u)


def _round_uncertainty(u: float) -> Tuple[float, int]:
    """
    Returns:
      (rounded_uncertainty, ndigits_for_rounding_value)
    where ndigits may be negative (round to tens/hundreds/etc.).
    """
    if u <= 0 or not math.isfinite(u):
        return u, 0

    u = abs(float(u))
    exponent = math.floor(math.log10(u))
    leading = u / (10**exponent)

    sig_figs = 2 if 1.0 <= leading < 2.0 else 1
    ndigits = sig_figs - 1 - exponent  # may be negative

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


def combine_uncertainties(
    terms: list[float], method: str = "worst_case"
) -> float:
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


def _resolve_uncertainty(label: str, q: Quantity) -> float:
    if q.uncertainty is not None:
        return float(q.uncertainty)
    u = get_equipment_uncertainty(label, q.value)
    if u is None:
        raise ValueError(
            f"No uncertainty provided for '{label}' and no equipment match"
        )
    return float(u)


def _require_same_unit(items: Mapping[str, Quantity]) -> str:
    units = [q.unit.strip() for q in items.values() if q.unit.strip()]
    uniq = sorted(set(units))
    if len(uniq) > 1:
        raise ValueError(f"Addition/subtraction requires consistent units; got {uniq}")
    return uniq[0] if uniq else ""


def add_subtract(values: Mapping[str, Quantity]) -> Dict[str, object]:
    """
    Worst-case propagation for y = Σ x_i (use negative values for subtraction).

    Returns:
      {'value': float, 'uncertainty': float, 'unit': str, 'text': str, 'formatted': str}
    """
    if not values:
        raise ValueError("values cannot be empty")

    unit = _require_same_unit(values)

    total = 0.0
    total_unc = 0.0
    details = []

    for label, q in values.items():
        u = _resolve_uncertainty(label, q)
        total += float(q.value)
        total_unc += abs(u)
        details.append((label, float(q.value), abs(u), unit or q.unit))

    lines = []
    lines.append("Worst-case uncertainty (addition/subtraction):")
    lines.append("Δy = ΣΔx")
    for label, v, u, u_unit in details:
        u_unit = u_unit.strip() or unit
        lines.append(f"- {label}: {v} {u_unit} with Δ = {u} {u_unit}")
    lines.append(f"Δy = {total_unc:g} {unit}".strip())

    formatted = format_value_with_uncertainty(total, total_unc, unit)
    lines.append(f"Result: {formatted}")

    return {
        "value": float(total),
        "uncertainty": float(total_unc),
        "unit": unit,
        "formatted": formatted,
        "text": "\n".join(lines),
    }


def _combine_unit_product(units: list[str]) -> str:
    u = [s.strip() for s in units if s and s.strip()]
    if not u:
        return ""
    return "·".join(u)


def mul_div(
    numerators: Mapping[str, Quantity],
    denominators: Mapping[str, Quantity] | None = None,
    result_unit: str = "",
) -> Dict[str, object]:
    """
    Worst-case propagation for y = (Π a_i) / (Π b_j).

    Uses:
      Δy/|y| = Σ (Δa_i/|a_i|) + Σ (Δb_j/|b_j|)

    Units:
      - If result_unit is provided, it is used.
      - Otherwise, a simple composite unit string is constructed as (num_units)/(den_units).
        This is not algebraic simplification; for IA-grade reporting, supply result_unit explicitly.
    """
    denominators = denominators or {}

    if not numerators:
        raise ValueError("numerators cannot be empty")

    num_val = 1.0
    den_val = 1.0

    num_units = []
    den_units = []

    for _, q in numerators.items():
        num_val *= float(q.value)
        if q.unit.strip():
            num_units.append(q.unit.strip())

    for _, q in denominators.items():
        den_val *= float(q.value)
        if q.unit.strip():
            den_units.append(q.unit.strip())

    if den_val == 0:
        raise ZeroDivisionError("Denominator product is zero")

    y = num_val / den_val

    rel_unc = 0.0
    details = []

    for label, q in list(numerators.items()) + list(denominators.items()):
        u = _resolve_uncertainty(label, q)
        v = float(q.value)
        if v == 0:
            raise ZeroDivisionError(
                f"Value for '{label}' is zero; cannot compute relative uncertainty"
            )
        rel_unc += abs(u) / abs(v)
        details.append((label, v, abs(u), q.unit))

    abs_unc = abs(y) * rel_unc

    if result_unit.strip():
        unit = result_unit.strip()
    else:
        num_u = _combine_unit_product(num_units)
        den_u = _combine_unit_product(den_units)
        if den_u and num_u:
            unit = f"({num_u})/({den_u})"
        elif den_u and not num_u:
            unit = f"1/({den_u})"
        else:
            unit = num_u

    lines = []
    lines.append("Worst-case uncertainty (multiplication/division):")
    lines.append("Δy/|y| = Σ(Δx/|x|)")
    for label, v, u, u_unit in details:
        u_unit = u_unit.strip()
        part = abs(u) / abs(v)
        lines.append(
            f"- {label}: {v} {u_unit} with Δ = {u} {u_unit}  =>  Δ/|x| = {part:.6g}"
        )
    lines.append(f"Δy/|y| = {rel_unc:.6g}")
    lines.append(f"Δy = |y| × Δy/|y| = {abs(y):.6g} × {rel_unc:.6g} = {abs_unc:.6g}")

    formatted = format_value_with_uncertainty(y, abs_unc, unit)
    lines.append(f"Result: {formatted}")

    return {
        "value": float(y),
        "uncertainty": float(abs_unc),
        "unit": unit,
        "formatted": formatted,
        "text": "\n".join(lines),
    }


def power(
    value: float, uncertainty: float, n: float, unit: str = ""
) -> Dict[str, object]:
    """
    Worst-case propagation for y = a^n.

    If a != 0:
      Δy/|y| = |n| (Δa/|a|)

    If a == 0:
      - if n == 0: y = 1 (defined here), Δy = 0
      - if n > 0: y = 0, worst-case Δy = (Δa)^n
      - if n < 0: undefined; raises
    """
    a = float(value)
    da = abs(float(uncertainty))
    n = float(n)

    if not math.isfinite(a) or not math.isfinite(da) or not math.isfinite(n):
        raise ValueError("value, uncertainty, and n must be finite")

    if a == 0.0:
        if n == 0.0:
            y = 1.0
            abs_unc = 0.0
            rel_unc = 0.0
        elif n > 0.0:
            y = 0.0
            abs_unc = da**n if da > 0 else 0.0
            rel_unc = math.nan
        else:
            raise ValueError("Power propagation undefined for a = 0 with n < 0")
    else:
        y = a**n
        rel_unc = abs(n) * (da / abs(a))
        abs_unc = abs(y) * rel_unc

    lines = []
    lines.append("Worst-case uncertainty (power):")
    lines.append("y = a^n")
    if a != 0.0:
        lines.append("Δy/|y| = |n| (Δa/|a|)")
        lines.append(f"a = {a} {unit}, Δa = {da} {unit}, n = {n:g}")
        lines.append(f"Δy/|y| = {abs(n):g} × ({da:.6g}/{abs(a):.6g}) = {rel_unc:.6g}")
        lines.append(
            f"Δy = |y| × Δy/|y| = {abs(y):.6g} × {rel_unc:.6g} = {abs_unc:.6g}"
        )
    else:
        lines.append(f"a = 0, Δa = {da} {unit}, n = {n:g}")
        if n == 0.0:
            lines.append("Defined as 0^0 = 1 for reporting; Δy = 0")
        elif n > 0.0:
            lines.append("Worst-case bound: a ∈ [0, Δa] so y ∈ [0, (Δa)^n]")
            lines.append(f"Δy = (Δa)^n = {abs_unc:.6g}")
        else:
            lines.append("Undefined for n < 0")

    formatted = format_value_with_uncertainty(y, abs_unc, unit)
    lines.append(f"Result: {formatted}")

    return {
        "value": float(y),
        "uncertainty": float(abs_unc),
        "unit": unit.strip(),
        "formatted": formatted,
        "text": "\n".join(lines),
    }
