"""Define standardized column names for result DataFrames."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultColumns:
    """Container for standardized column labels.

    These column names are used in all result DataFrames throughout the analysis
    pipeline, ensuring consistency in data processing, statistics, and plotting.

    Attributes:
        nacl: Column name for NaCl concentration in mol dm^-3 (M).
            Valid experimental range: 0.00-0.80 M in 0.20 M increments.
            NaCl acts as an ionic strength modifier; it does not participate
            in the acid-base stoichiometry.

        pka_app: Column name for apparent pKa values determined from titrations.
            The apparent pKa (pKa_app) is the pH measured at the half-equivalence
            point. It reflects the combined effects of the thermodynamic pKa and
            non-ideal solution behavior (activity coefficients) at each [NaCl].
            As ionic strength increases with [NaCl], pKa_app is expected to change.

        pka_unc: Column name for systematic uncertainty in pKa_app.
            Propagated from pH sensor (±0.3 pH units), burette readings (±0.02 cm^3),
            volume delivery (±0.10 cm^3), and NaCl concentration preparation
            uncertainties. Reported following IB significant figure conventions.
    """

    nacl: str = "NaCl Concentration (M)"
    pka_app: str = "Apparent pKa"
    pka_unc: str = "Uncertainty in Apparent pKa"
