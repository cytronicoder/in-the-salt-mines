"""Shared schema definitions for result dataframes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResultColumns:
    nacl: str = "NaCl Concentration (M)"
    pka_app: str = "Apparent pKa"
    pka_unc: str = "Uncertainty in Apparent pKa"
