"""Define standardized column names for result DataFrames.

This module provides a single, explicit schema for column naming used across
the analysis, statistics, and plotting layers. The terminology intentionally
emphasizes apparent pKa values influenced by ionic strength rather than
thermodynamic dissociation constants.
"""

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
