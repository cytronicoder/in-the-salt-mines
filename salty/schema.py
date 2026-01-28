"""
Schema definitions for standardized DataFrame column names.

Provides a single source of truth for column naming conventions used across
the package. This ensures consistency between data processing, analysis,
and plotting modules.

Usage:
    cols = ResultColumns()
    df[cols.pka_app]  # Access 'Apparent pKa' column

Note:
    The explicit 'Apparent pKa' terminology (rather than simply 'pKa')
    emphasizes that these are concentration-based measurements affected
    by ionic strength, not thermodynamic dissociation constants.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultColumns:
    """
    Standard column names for result DataFrames.

    Attributes:
        nacl: NaCl concentration column name ('NaCl Concentration (M)').
        pka_app: Apparent pKa column name ('Apparent pKa').
        pka_unc: pKa uncertainty column name ('Uncertainty in Apparent pKa').
    """

    nacl: str = "NaCl Concentration (M)"
    pka_app: str = "Apparent pKa"
    pka_unc: str = "Uncertainty in Apparent pKa"
