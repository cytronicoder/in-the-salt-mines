"""Define standardized column names for result DataFrames."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultColumns:
    """Container for standardized column labels.

    Attributes:
        nacl: Column name for NaCl concentration (mol/L).
        pka_app: Column name for apparent pKa values.
        pka_unc: Column name for systematic pKa_app uncertainty.
    """

    nacl: str = "NaCl Concentration (M)"
    pka_app: str = "Apparent pKa"
    pka_unc: str = "Uncertainty in Apparent pKa"
