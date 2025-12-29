"""
IB SL-style worst-case uncertainty propagation utilities.

Implements only absolute uncertainty (worst-case) propagation following
explicit IB rules (addition/subtraction, multiplication/division, powers).

Functions produce both numeric results and formatted IB-style explanatory
text suitable for inclusion in an IA.

Constraints enforced:
 - Only absolute/worst-case formulas used (no calculus or statistical)
 - For multiplication/division and powers, relative uncertainties are used
   only to combine contributions as prescribed by IB worst-case rules.
 - Equipment uncertainties are encoded and used when an equipment name is
   supplied instead of an explicit uncertainty.

"""
from __future__ import annotations

from typing import Dict, Tuple, Optional
import math

# Equipment uncertainties (absolute or percentage where noted)
_EQUIPMENT_UNCERTAINTIES: Dict[str, Tuple[float, str]] = {
    # (uncertainty, 'abs'|'pct')
    "25.0 cm3 pipette": (0.06, "abs"),
    "50.0 cm3 burette": (0.05, "abs"),
    "250 cm3 beaker": (5.0, "pct"),
    "100 cm3 volumetric flask": (0.10, "abs"),
    "Vernier pH Sensor PH-BTA": (0.2, "abs"),
    "Digital thermometer": (0.1, "abs"),
    "Analytical balance": (0.01, "abs"),
}


def get_equipment_uncertainty(equipment: str, value: float) -> Optional[float]:
    """Return absolute uncertainty for a known equipment identifier.

    If the device uncertainty is given as a percentage, convert to absolute
    based on the provided numeric value.
    Returns None if the equipment is unknown.
    """
    entry = _EQUIPMENT_UNCERTAINTIES.get(equipment)
    if entry is None:
        return None
    u, typ = entry
    if typ == "abs":
        return float(u)
    else:
        # percentage -> absolute
        return float((u / 100.0) * value)


def _round_uncertainty(u: float) -> Tuple[float, int]:
    """Round uncertainty to 1 significant figure (2 if leading digit is 1).

    Returns (rounded_uncertainty, decimal_places_to_round_value_to).
    This follows common IB practice: report uncertainty to 1 sig fig unless
    the first digit is 1, in which case 2 sig figs improves readability.
    """
    if u <= 0 or not math.isfinite(u):
        return u, 0
    exponent = math.floor(math.log10(abs(u)))
    first = abs(u) / (10 ** exponent)
    # If first digit is 1, keep two sig figs
    sig_figs = 2 if 1.0 <= first < 2.0 else 1
    # compute rounding place
    digits = sig_figs - 1 - exponent
    rounded = round(u, digits)
    # If rounding produced 0 (very small), fallback to absolute rounding at exponent
    if rounded == 0:
        rounded = round(u, digits + 1)
        digits = digits + 1
    return rounded, max(digits, 0)


def _format_value_with_uncertainty(value: float, uncertainty: float, unit: str = "") -> str:
    """Format value ± uncertainty with IB-style significant figures and unit."""
    ru, digits = _round_uncertainty(abs(uncertainty))
    if ru == 0 or not math.isfinite(ru):
        # Fallback: show raw numbers
        val_str = f"{value:.6g}"
        unc_str = f"{uncertainty:.6g}"
        return f"{val_str} ± {unc_str} {unit}".strip()
    # Round value to same decimal place as uncertainty
    rounded_value = round(value, digits)
    # Choose formatting: if digits > 0 show that many decimals, else integer
    fmt = f"{{:.{digits}f}}" if digits > 0 else "{:.0f}"
    return f"{fmt.format(rounded_value)} ± {fmt.format(ru)} {unit}".strip()


def add_subtract(values: Dict[str, Tuple[float, Optional[float], str]]) -> Dict[str, str]:
    """Perform worst-case absolute uncertainty propagation for addition/subtraction.

    Args:
        values: mapping from label -> (value, absolute_uncertainty_or_None, unit)
                If absolute_uncertainty_or_None is None, the function will look
                for equipment names matching the label in the equipment table.

    Behaviour:
        - The combined value is the arithmetic sum (or difference) of values.
        - The combined uncertainty is the sum of the absolute uncertainties.

    Returns:
        dict with keys:
          - 'value': numeric result
          - 'uncertainty': numeric absolute uncertainty
          - 'text': formatted IB-style explanation suitable for IA
    """
    total = 0.0
    total_unc = 0.0
    lines = []
    for label, (val, unc, unit) in values.items():
        total += val
        if unc is None:
            # try equipment lookup
            eq_unc = get_equipment_uncertainty(label, val)
            if eq_unc is None:
                raise ValueError(f"No uncertainty provided for '{label}' and no equipment match")
            u = eq_unc
            lines.append(f"{label}: {val} {unit} with equipment uncertainty {u} {unit}")
        else:
            u = unc
            lines.append(f"{label}: {val} {unit} with stated uncertainty ±{u} {unit}")
        total_unc += abs(u)

    result_text = []
    result_text.append("Calculation (worst-case absolute uncertainties):")
    formula_terms = " + ".join([f"{v[0]}" for v in values.values()])
    result_text.append(f"Formula: y = {formula_terms}")
    result_text.append("Uncertainties combined by summation (Δy = Σ Δxi):")
    for l in lines:
        result_text.append(f" - {l}")
    result_text.append(f"Δy = " + " + ".join([f"{abs(v[1]) if v[1] is not None else get_equipment_uncertainty(k,v[0])}" for k,v in values.items()]))

    formatted = _format_value_with_uncertainty(total, total_unc, unit=list(values.values())[-1][2])
    result_text.append(f"Result: {formatted}")

    return {"value": total, "uncertainty": total_unc, "text": "\n".join(result_text)}


def mul_div(numerators: Dict[str, Tuple[float, Optional[float], str]], denominators: Dict[str, Tuple[float, Optional[float], str]]) -> Dict[str, str]:
    """Perform worst-case absolute uncertainty propagation for multiplication/division.

    Args:
        numerators: mapping label -> (value, absolute_uncertainty_or_None, unit)
        denominators: same for denominators

    Behaviour (IB worst-case):
        Δy / y = Σ (Δxi / xi) for all factors (numerators and denominators).
        Δy = y × (Δy / y)

    Returns formatted dict like add_subtract.
    """
    # Compute numeric result
    num = 1.0
    for val, *_ in numerators.values():
        num *= val
    den = 1.0
    for val, *_ in denominators.values():
        den *= val
    if den == 0:
        raise ZeroDivisionError("Denominator product is zero")
    y = num / den

    # Compute relative uncertainty
    rel_unc = 0.0
    lines = []
    for label, (val, unc, unit) in list(numerators.items()) + list(denominators.items()):
        if unc is None:
            eq_unc = get_equipment_uncertainty(label, val)
            if eq_unc is None:
                raise ValueError(f"No uncertainty provided for '{label}' and no equipment match")
            u = eq_unc
            lines.append(f"{label}: {val} {unit} with equipment uncertainty {u} {unit}")
        else:
            u = unc
            lines.append(f"{label}: {val} {unit} with stated uncertainty ±{u} {unit}")
        if val == 0:
            raise ZeroDivisionError(f"Value for '{label}' is zero; cannot compute relative uncertainty")
        rel_unc += abs(u) / abs(val)

    abs_unc = abs(y) * rel_unc

    # Build IB-style explanatory text
    result_text = ["Calculation (worst-case absolute uncertainties):"]
    # Build formula string
    num_terms = " * ".join([str(v[0]) for v in numerators.values()])
    den_terms = " * ".join([str(v[0]) for v in denominators.values()]) if denominators else "1"
    result_text.append(f"Formula: y = ({num_terms}) / ({den_terms})")
    result_text.append("Relative uncertainties combined by summation:")
    result_text.append("Δy / y = Σ (Δxi / xi)")
    for l in lines:
        result_text.append(f" - {l}")
    # numeric rel unc display
    result_text.append(f"Δy / y = {rel_unc:.6g}")
    result_text.append(f"Δy = |y| × Δy/y = {abs(y):.6g} × {rel_unc:.6g} = {abs_unc:.6g}")

    # Choose unit as unit of first numerator if present
    unit = list(numerators.values())[0][2] if numerators else list(denominators.values())[0][2]
    formatted = _format_value_with_uncertainty(y, abs_unc, unit)
    result_text.append(f"Result: {formatted}")

    return {"value": y, "uncertainty": abs_unc, "text": "\n".join(result_text)}


def power(value: float, uncertainty: Optional[float], n: float, unit: str = "") -> Dict[str, str]:
    """Perform worst-case propagation for powers: y = a^n

    Uses formula: Δy / y = |n| × (Δa / a)

    Args:
        value: numeric value a
        uncertainty: absolute uncertainty Δa (or None to lookup equipment)
        n: exponent
    """
    if uncertainty is None:
        # No equipment name to lookup here; caller should supply absolute uncertainty
        raise ValueError("Absolute uncertainty must be supplied for power calculations")
    if value == 0:
        # then y==0; uncertainty is zero if Δa is finite? worst-case: y could be 0±Δy where Δy computed via formula if defined
        y = 0.0
        abs_unc = 0.0
    else:
        y = value ** n
        rel_unc = abs(n) * (abs(uncertainty) / abs(value))
        abs_unc = abs(y) * rel_unc

    result_text = ["Calculation (worst-case absolute uncertainties):"]
    result_text.append(f"Formula: y = a^{n}")
    result_text.append("Δy / y = |n| × (Δa / a)")
    result_text.append(f"a = {value} {unit}, Δa = {uncertainty} {unit}")
    if value != 0:
        result_text.append(f"Δy / y = {abs(n):g} × ({abs(uncertainty):.6g} / {abs(value):.6g}) = {rel_unc:.6g}")
        result_text.append(f"Δy = |y| × Δy/y = {abs(y):.6g} × {rel_unc:.6g} = {abs_unc:.6g}")
    else:
        result_text.append("a = 0; treated as exact zero for power propagation, Δy = 0")

    formatted = _format_value_with_uncertainty(y, abs_unc, unit)
    result_text.append(f"Result: {formatted}")

    return {"value": y, "uncertainty": abs_unc, "text": "\n".join(result_text)}


# Public API helpers for common equipment lookups
def uncertainty_for_equipment(equipment: str, value: float) -> float:
    """Helper to get absolute uncertainty and raise if not known."""
    u = get_equipment_uncertainty(equipment, value)
    if u is None:
        raise ValueError(f"Unknown equipment '{equipment}'")
    return u
