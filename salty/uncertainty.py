"""
Provides uncertainty propagation utilities following IB DP rules.

Implements worst-case propagation for addition, multiplication, and powers.

Includes equipment uncertainty lookup, value rounding, and formatting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

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


def get_equipment_uncertainty(equipment: str, value: float) -> Optional[float]:
    entry = _EQUIPMENT_UNCERTAINTIES.get(equipment)
    if entry is None:
        return None
    u, typ = entry
    if typ == "abs":
        return float(u)
    return float((u / 100.0) * value)


def burette_delivered_uncertainty(
    reading_uncertainty: float, readings: int = 2
) -> float:
    if readings < 1:
        raise ValueError("readings must be >= 1")
    return float(readings) * abs(float(reading_uncertainty))


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



