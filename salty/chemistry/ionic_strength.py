"""Calculate ionic strength for electrolyte solutions.

Ionic strength (μ) quantifies the concentration of ions in solution and affects
activity coefficients through electrostatic shielding. This module provides
utilities to compute ionic strength for the IA investigation.

Background Theory:
    Ionic strength is formally defined as:
        μ = 0.5 Σ c_i z_i^2

    where c_i is the concentration of ion i and z_i is its charge.

    For a 1:1 electrolyte like NaCl:
        NaCl → Na⁺ + Cl⁻
        μ = 0.5 * ([Na⁺] * 1^2 + [Cl⁻] * 1^2)
        μ = 0.5 * ([NaCl] + [NaCl])
        μ = [NaCl]

    Therefore, for NaCl solutions, the ionic strength equals the NaCl concentration.

Experimental Context:
    - NaCl is added to 0.10 M ethanoic acid solutions
    - NaCl concentrations: 0.00, 0.20, 0.40, 0.60, 0.80, 1.00 M
    - NaCl does not participate in the acid-base reaction
    - Higher μ increases electrostatic shielding, affecting activity coefficients
    - This shifts the apparent pKa (pKa_app) measured at the half-equivalence point
"""

from __future__ import annotations


def ionic_strength_nacl(nacl_concentration: float) -> float:
    """Calculate ionic strength for a NaCl solution.

    For NaCl (a 1:1 electrolyte), ionic strength equals the NaCl concentration:
        μ = [NaCl]

    Args:
        nacl_concentration: NaCl concentration in mol dm⁻^3 (M).
            Valid experimental range: 0.00-1.00 M.

    Returns:
        Ionic strength in mol dm⁻^3 (M), equal to the input concentration.

    Raises:
        ValueError: If concentration is negative or non-finite.

    Examples:
        >>> ionic_strength_nacl(0.20)
        0.20
        >>> ionic_strength_nacl(0.00)
        0.0
        >>> ionic_strength_nacl(1.00)
        1.0
    """
    if not isinstance(nacl_concentration, (int, float)):
        raise TypeError(
            f"nacl_concentration must be numeric, got {type(nacl_concentration)}"
        )

    c = float(nacl_concentration)

    if c < 0:
        raise ValueError(f"NaCl concentration cannot be negative, got {c}")

    import math

    if not math.isfinite(c):
        raise ValueError(f"NaCl concentration must be finite, got {c}")

    # For 1:1 electrolyte: μ = [NaCl]
    return c


def ionic_strength_general(ion_concentrations: dict[str, tuple[float, int]]) -> float:
    """Calculate ionic strength for a general electrolyte solution.

    Uses the formal definition:
        μ = 0.5 Σ c_i z_i^2

    Args:
        ion_concentrations: Dictionary mapping ion names to (concentration, charge).
            Example: {"Na+": (0.20, 1), "Cl-": (0.20, -1)}

    Returns:
        Ionic strength in mol dm⁻^3 (M).

    Raises:
        ValueError: If concentrations are negative or charges are zero.

    Examples:
        >>> ionic_strength_general({"Na+": (0.20, 1), "Cl-": (0.20, -1)})
        0.20
        >>> ionic_strength_general({"Ca2+": (0.10, 2), "Cl-": (0.20, -1)})
        0.30
    """
    if not ion_concentrations:
        return 0.0

    mu = 0.0
    for ion_name, (conc, charge) in ion_concentrations.items():
        if conc < 0:
            raise ValueError(
                f"Ion concentration for {ion_name} cannot be negative, got {conc}"
            )
        if charge == 0:
            raise ValueError(f"Ion charge for {ion_name} cannot be zero")

        mu += float(conc) * (int(charge) ** 2)

    return 0.5 * mu
