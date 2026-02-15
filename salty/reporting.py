"""Format and validate result tables for IB-style uncertainty reporting.

This module is used after numerical analysis to enforce consistent value and
uncertainty presentation in exported CSV artifacts.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _ib_round_uncertainty(uncertainty: float) -> tuple[float, int]:
    """Round uncertainty using IB-style significant-figure conventions.

    Args:
        uncertainty (float): Absolute uncertainty value.

    Returns:
        tuple[float, int]: Rounded uncertainty and decimal places used.

    Raises:
        ValueError: If uncertainty is non-finite or non-positive.

    Note:
        Use one significant figure by default and two when the leading digit
        is 1.
    """
    u = float(uncertainty)
    if not np.isfinite(u) or u <= 0:
        raise ValueError(f"Uncertainty must be finite and > 0, got {uncertainty!r}")

    exponent = int(np.floor(np.log10(abs(u))))
    leading = abs(u) / (10**exponent)
    sig_figs = 2 if 1.0 <= leading < 2.0 else 1
    ndigits = sig_figs - 1 - exponent
    rounded_u = round(abs(u), ndigits)
    return float(rounded_u), int(max(0, ndigits))


def uncertainty_decimal_places(uncertainty: float) -> int:
    """Return decimal places implied by IB-style uncertainty rounding.

    Args:
        uncertainty (float): Absolute uncertainty for a reported value
            (same unit as the value being reported).

    Returns:
        int: Number of decimal places that the paired value should use.

    Note:
        This helper enforces consistency between a value and its uncertainty in
        human-readable tables.

    References:
        Significant-figure alignment for value/uncertainty reporting.
    """
    rounded_u, ndigits = _ib_round_uncertainty(uncertainty)
    if ndigits <= 0:
        return 0
    txt = f"{rounded_u:.12f}".rstrip("0")
    if "." not in txt:
        return 0
    return len(txt.split(".", 1)[1])


def format_value_to_uncertainty_decimals(value: float, uncertainty: float) -> str:
    """Format a value using decimal places implied by its uncertainty.

    Args:
        value (float): Numerical value to format.
        uncertainty (float): Absolute uncertainty used to infer decimal places
            (same unit as ``value``).

    Returns:
        str: Value string rounded to the uncertainty-implied precision.

    Note:
        Intended for reporting/export only; original numeric values should be kept
        for downstream calculations.

    References:
        Precision-matched numerical formatting for analytical reports.
    """
    dp = uncertainty_decimal_places(uncertainty)
    return f"{float(value):.{dp}f}"


def uncertainty_forms(value: float, uncertainty: float) -> tuple[float, float]:
    """Return fractional and percentage uncertainty forms.

    Args:
        value (float): Measured or calculated quantity (any unit).
        uncertainty (float): Absolute uncertainty associated with ``value`` in
            the same unit.

    Returns:
        tuple[float, float]: Pair ``(fractional_uncertainty,
        percentage_uncertainty)`` where the first term is dimensionless and the
        second is in percent.

    Note:
        Returns ``(nan, nan)`` when ``value`` is zero or either input is non-finite.

    References:
        Fractional and percentage uncertainty definitions.
    """
    v = float(value)
    u = float(uncertainty)
    if not np.isfinite(v) or not np.isfinite(u) or v == 0:
        return np.nan, np.nan
    frac = abs(u / v)
    return float(frac), float(frac * 100.0)


def validate_uncertainty_columns(
    df: pd.DataFrame,
    value_uncertainty_pairs: Iterable[tuple[str, str]],
) -> None:
    """Validate value/uncertainty column pairs for reporting safety.

    Args:
        df (pandas.DataFrame): Table containing measured values and uncertainty
            columns.
        value_uncertainty_pairs (Iterable[tuple[str, str]]): Sequence of
            ``(value_column, uncertainty_column)`` pairs to validate.

    Returns:
        None: Raise on invalid metadata and otherwise return nothing.

    Raises:
        KeyError: If any required value or uncertainty column is missing.
        ValueError: If a finite value exists in a row where uncertainty is
            missing, non-finite, or non-positive.

    Note:
        Validation is strict by design to prevent silently exporting misleading
    reported values without defensible uncertainty metadata.

    References:
        Data-quality gates for uncertainty-aware reporting.
    """
    for value_col, unc_col in value_uncertainty_pairs:
        if value_col not in df.columns:
            raise KeyError(f"Missing value column '{value_col}' for reporting format.")
        if unc_col not in df.columns:
            raise KeyError(
                f"Missing uncertainty column '{unc_col}' required for '{value_col}'."
            )

        values = pd.to_numeric(df[value_col], errors="coerce")
        uncs = pd.to_numeric(df[unc_col], errors="coerce")
        missing_mask = values.notna() & (~np.isfinite(uncs) | (uncs <= 0))

        if bool(missing_mask.any()):
            bad_rows = list(df.index[missing_mask][:5])
            raise ValueError(
                "Uncertainty metadata missing/invalid for measured values in "
                f"'{value_col}' (uncertainty '{unc_col}'). "
                f"Example row indices: {bad_rows}."
            )


def add_formatted_reporting_columns(
    df: pd.DataFrame,
    value_uncertainty_pairs: Iterable[tuple[str, str]],
    suffix: str = " (reported)",
) -> pd.DataFrame:
    """Add reporting-ready string columns for values and uncertainties.

    Args:
        df (pandas.DataFrame): Input numeric table.
        value_uncertainty_pairs (Iterable[tuple[str, str]]): Sequence of
            ``(value_column, uncertainty_column)`` pairs to format.
        suffix (str, optional): Suffix appended to generated reporting columns.
            Defaults to ``" (reported)"``.

    Returns:
        pandas.DataFrame: Copy of ``df`` with formatted string columns added.

    Raises:
        KeyError: If required value/uncertainty columns are absent.
        ValueError: If uncertainty metadata is invalid for finite values.

    Note:
        Original numeric columns are preserved for downstream computation.

    References:
        Publication-table formatting conventions with explicit uncertainties.
    """
    out = df.copy()
    validate_uncertainty_columns(out, value_uncertainty_pairs)

    for value_col, unc_col in value_uncertainty_pairs:
        values = pd.to_numeric(out[value_col], errors="coerce")
        uncs = pd.to_numeric(out[unc_col], errors="coerce")
        value_report_col = f"{value_col}{suffix}"
        unc_report_col = f"{unc_col}{suffix}"

        out[value_report_col] = [
            (
                format_value_to_uncertainty_decimals(v, u)
                if (np.isfinite(v) and np.isfinite(u) and u > 0)
                else ""
            )
            for v, u in zip(values, uncs)
        ]
        out[unc_report_col] = [
            (
                f"{_ib_round_uncertainty(u)[0]:.{uncertainty_decimal_places(u)}f}"
                if (np.isfinite(u) and u > 0)
                else ""
            )
            for u in uncs
        ]

    return out


def add_uncertainty_form_columns(
    df: pd.DataFrame,
    value_uncertainty_pairs: Iterable[tuple[str, str]],
) -> pd.DataFrame:
    """Add fractional and percentage uncertainty columns.

    Args:
        df (pandas.DataFrame): Input table containing numeric value and
            uncertainty columns.
        value_uncertainty_pairs (Iterable[tuple[str, str]]): Sequence of
            ``(value_column, uncertainty_column)`` pairs.

    Returns:
        pandas.DataFrame: Copy of ``df`` with added uncertainty-form columns.

    Raises:
        KeyError: If required value/uncertainty columns are absent.
        ValueError: If uncertainty metadata is invalid for finite values.

    Note:
        Added columns are dimensionless fractional uncertainty and percentage
        uncertainty for quick precision comparison across quantities.

    References:
        Relative uncertainty representation in laboratory reporting.
    """
    out = df.copy()
    validate_uncertainty_columns(out, value_uncertainty_pairs)

    for value_col, unc_col in value_uncertainty_pairs:
        values = pd.to_numeric(out[value_col], errors="coerce")
        uncs = pd.to_numeric(out[unc_col], errors="coerce")

        frac_col = f"{value_col} fractional uncertainty"
        pct_col = f"{value_col} percentage uncertainty (%)"

        frac_vals = []
        pct_vals = []
        for v, u in zip(values, uncs):
            frac, pct = uncertainty_forms(v, u)
            frac_vals.append(frac)
            pct_vals.append(pct)

        out[frac_col] = frac_vals
        out[pct_col] = pct_vals

    return out
