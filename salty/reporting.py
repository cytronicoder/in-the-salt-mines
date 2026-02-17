"""Format and validate result tables for IB-style uncertainty reporting.

This module is used after numerical analysis to enforce consistent value and
uncertainty presentation in exported CSV artifacts.
"""

from __future__ import annotations

import os
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


def _safe_float(value: float, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _replicate_count_text(results_df: pd.DataFrame) -> str:
    if results_df.empty or "NaCl Concentration (M)" not in results_df.columns:
        return "0"
    counts = (
        results_df.groupby("NaCl Concentration (M)")["Run"]
        .count()
        .astype(int)
        .to_numpy(dtype=int)
    )
    if len(counts) == 0:
        return "0"
    if int(np.min(counts)) == int(np.max(counts)):
        return str(int(counts[0]))
    return f"{int(np.min(counts))}-{int(np.max(counts))}"


def _temperature_stats_from_results(results: list[dict]) -> tuple[float, float]:
    temps = []
    for res in results:
        raw_df = res.get("data", pd.DataFrame())
        if (
            not isinstance(raw_df, pd.DataFrame)
            or "Temperature (°C)" not in raw_df.columns
        ):
            continue
        tvals = pd.to_numeric(raw_df["Temperature (°C)"], errors="coerce").to_numpy(
            dtype=float
        )
        tvals = tvals[np.isfinite(tvals)]
        if len(tvals) == 0:
            continue
        temps.append(float(np.mean(tvals)))
    if len(temps) == 0:
        return np.nan, np.nan
    if len(temps) == 1:
        return float(temps[0]), 0.0
    return float(np.mean(temps)), float(np.std(temps, ddof=1))


def generate_caption_texts(
    results: list[dict],
    results_df: pd.DataFrame,
    figure3_fit: dict | None = None,
    figure5_meta: dict | None = None,
) -> dict[str, str]:
    """Generate Figure 1-5 caption text using the publication templates."""
    figure3_fit = figure3_fit or {}
    figure5_meta = figure5_meta or {}

    if not results_df.empty and "NaCl Concentration (M)" in results_df.columns:
        levels = sorted(
            {
                float(v)
                for v in pd.to_numeric(
                    results_df["NaCl Concentration (M)"], errors="coerce"
                ).to_numpy(dtype=float)
                if np.isfinite(v)
            }
        )
    else:
        levels = sorted(
            {
                float(r.get("nacl_conc", np.nan))
                for r in results
                if np.isfinite(float(r.get("nacl_conc", np.nan)))
            }
        )
    levels_txt = ", ".join(f"{v:.1f}" for v in levels)

    t_mean, t_sd = _temperature_stats_from_results(results)
    n_reps = _replicate_count_text(results_df)

    fit_basis = str(figure3_fit.get("fit_basis", "condition means"))
    slope = _safe_float(figure3_fit.get("slope", np.nan))
    slope_ci_low = _safe_float(figure3_fit.get("slope_ci_low", np.nan))
    slope_ci_high = _safe_float(figure3_fit.get("slope_ci_high", np.nan))
    n_reps_fig3 = str(figure3_fit.get("n_reps", n_reps))

    captions: dict[str, str] = {}
    captions["titration_overlays_by_nacl"] = (
        "Figure 1. Titration curves of ethanoic acid with 0.10 mol·dm^-3 NaOH at "
        "varying NaCl concentrations ([NaCl] = "
        f"{levels_txt} mol·dm^-3) at {t_mean:.2f} ± {t_sd:.2f} °C "
        f"(n={n_reps} per condition). "
        "Thin lines show individual trials; thick lines show the within-condition "
        "mean curve after linear interpolation onto a common volume grid. "
        "The equivalence volume V_eq was defined as the midpoint of the steepest "
        "measured interval (identified by the maximum discrete derivative "
        "ΔpH/ΔV). Uncertainty in V_eq for each trial was taken as "
        "σ(V_eq)=(V_{i+1}-V_i)/2 where [V_i, V_{i+1}] is the steepest interval. "
        "The half-equivalence volume V_{1/2}=V_eq/2 is indicated by a dashed line. "
        "The half-equivalence pH, pH(V_{1/2}), was obtained by linear "
        "interpolation between the two measured points bracketing V_{1/2}. "
        "Shaded bands and error bars indicate replicate variability (±1 SD) "
        "for V_eq, V_{1/2}, and pH(V_{1/2}). "
        "Because the glass pH electrode reports hydrogen ion activity, "
        "pH(V_{1/2}) is interpreted as an apparent pK_a under the ionic-strength "
        "conditions of each titration."
    )

    captions["derivative_equivalence_by_nacl"] = (
        "Figure 2. Discrete derivative method used to identify V_eq. "
        "For each trial, ΔpH/ΔV was computed between successive measurements "
        "and plotted at the volume midpoint. "
        "The equivalence interval [V_i, V_{i+1}] was defined as the interval "
        "with maximum ΔpH/ΔV; the equivalence volume was taken as "
        "V_eq = (V_i + V_{i+1})/2 with σ(V_eq) = (V_{i+1} - V_i)/2. "
        "Points/lines show individual trials; vertical lines and shaded bands "
        "show within-condition mean ±1 SD. "
        "Increased resolution near equivalence (0.20 cm^3 additions) was used "
        "to minimize discretization error in V_eq."
    )

    captions["pka_app_vs_nacl_and_I"] = (
        "Figure 3. Apparent pK_a dependence on NaCl concentration and ionic strength. "
        "For each trial, pK_a,app was estimated as pH(V_{1/2}), where "
        "V_{1/2}=V_eq/2 and pH(V_{1/2}) was obtained by linear interpolation. "
        f"Points show individual trials (jittered in x); large markers show "
        f"within-condition means with 95% confidence intervals "
        f"(t-distribution, n={n_reps_fig3}). "
        f"The fitted line shows a linear regression of {fit_basis} with a 95% "
        f"confidence band; the slope was {slope:.4f} (95% CI "
        f"{slope_ci_low:.4f} to {slope_ci_high:.4f}). "
        "The observed trend is interpreted as a change in apparent pK_a driven by "
        "ionic-strength effects on activity coefficients, consistent with pH "
        "electrodes measuring activity rather than concentration."
    )

    if int(_safe_float(figure3_fit.get("excluded_outliers", 0), default=0.0)) > 0:
        captions["pka_app_vs_nacl_and_I"] += (
            " Temperature-outlier runs (|T−26|>1 °C) were shown in gray "
            "and excluded from the regression fit."
        )

    captions["hh_linearization_and_diagnostics"] = (
        "Figure 4. Henderson–Hasselbalch (HH) linearization and fit diagnostics. "
        "In the buffer region, x=log10(V/(V_eq−V)) and y=pH were computed for "
        "points satisfying 0.1<[A−]/[HA]<10, and a linear model "
        "pH = pK_a,app + m·x was fit per trial. "
        "Panel (a) shows exemplar HH plots and fitted lines; panel (b) shows "
        "residuals. Panel (c) summarizes fitted slopes m with 95% CI and the "
        "reference m=1 expected under ideal HH behavior; panel (d) summarizes "
        "R^2 across conditions. "
        "Deviations of m from 1 and systematic residual structure are interpreted "
        "as non-ideality (activity effects) and/or experimental artifacts "
        "(discretization near V_eq, electrode response lag)."
    )

    captions["temperature_and_calibration_qc"] = (
        "Figure 5. Temperature stability and electrode calibration checkpoints. "
        "Panel (a) shows temperature time series recorded continuously during each "
        "titration; traces are grouped by [NaCl]. "
        "The target condition was 26 ± 1 °C; runs outside this range are flagged on "
        "subsequent analyses. "
        "Panel (b) shows pH 7.00 buffer checks versus run order; recalibration was "
        "performed when the buffer reading deviated by more than ±0.05 pH units "
        "from 7.00. These QC checks bound systematic error from temperature "
        "variation and electrode drift."
    )

    if int(_safe_float(figure5_meta.get("temperature_outliers", 0), default=0.0)) > 0:
        captions["temperature_and_calibration_qc"] += (
            f" In this dataset, {int(figure5_meta['temperature_outliers'])} "
            "run(s) were outside 26 ± 1 °C."
        )
    return captions


def write_caption_files(captions: dict[str, str], output_dir: str) -> list[str]:
    """Write figure caption text files to the target directory."""
    os.makedirs(output_dir, exist_ok=True)
    written: list[str] = []
    for stem, text in captions.items():
        path = os.path.join(output_dir, f"{stem}_caption.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text.strip() + "\n")
        written.append(path)
    return written
