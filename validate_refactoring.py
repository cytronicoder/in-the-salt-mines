#!/usr/bin/env python3
"""Validate scientific and architectural refactoring requirements.

This script performs lightweight checks to confirm that the two-stage pKa_app
framework, documentation guardrails, and plotting validation rules are present.
"""

import inspect
import warnings

import numpy as np
import pandas as pd

from salty.analysis import calculate_statistics, detect_equivalence_point
from salty.chemistry.buffer_region import select_buffer_region
from salty.chemistry.hh_model import fit_henderson_hasselbalch
from salty.plotting.summary_plots import plot_statistical_summary
from salty.plotting.titration_plots import plot_titration_curves
from salty.stats.regression import slope_uncertainty_from_endpoints


def check_interpretation_guardrails():
    """Check interpretation guardrails in Henderson-Hasselbalch docstrings.

    Returns:
        True when required interpretation terms are present; otherwise False.
    """
    print("CHECK 1: Interpretation Guardrails")

    doc = fit_henderson_hasselbalch.__doc__
    required_terms = [
        "apparent",
        "pKa_app",
        "operational",
        "ionic strength",
        "comparative",
    ]

    found = []
    missing = []
    for term in required_terms:
        if term.lower() in doc.lower():
            found.append(term)
        else:
            missing.append(term)

    print(f"  ✓ Found: {', '.join(found)}")
    if missing:
        print(f"  ✗ Missing: {', '.join(missing)}")
        return False

    print("  ✓ All interpretation guardrails present\n")
    return True


def check_two_stage_protocol():
    """Check documentation of the two-stage pKa_app protocol.

    Returns:
        True when the protocol is documented; otherwise False.
    """
    print("CHECK 2: Two-Stage Protocol Documentation")

    hh_doc = fit_henderson_hasselbalch.__doc__
    buffer_doc = select_buffer_region.__doc__

    stage_terms = ["stage", "half-equivalence", "coarse", "refined"]
    found = [t for t in stage_terms if t.lower() in hh_doc.lower()]

    if len(found) >= 3:
        print(f"  ✓ Two-stage protocol documented: {', '.join(found)}")
    else:
        print(f"  ✗ Two-stage protocol incomplete: only found {', '.join(found)}")
        return False

    if "pKa_app" in buffer_doc or "pka_app" in buffer_doc.lower():
        print("  ✓ Buffer region docs reference pKa_app\n")
    else:
        print("  ✗ Buffer region docs missing pKa_app reference\n")
        return False

    return True


def check_slope_warning():
    """Check that non-ideal slopes trigger diagnostic warnings.

    Returns:
        True when the warning is issued; otherwise False.
    """
    print("CHECK 3: HH Slope Deviation Warning")

    veq = 25.0
    pka = 5.0
    volumes = np.linspace(5.0, 23.0, 30)
    log_ratios = np.log10(volumes / (veq - volumes))
    pH_bad = pka + 0.7 * log_ratios

    step_df = pd.DataFrame(
        {
            "Volume (cm³)": volumes,
            "pH_step": pH_bad,
        }
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka)

        if len(w) > 0 and "slope" in str(w[0].message).lower():
            print(f"  ✓ Warning triggered: {w[0].message}\n")
            return True
        else:
            print("  ✗ No slope warning issued\n")
            return False


def check_veq_bounds_warning():
    """Check for equivalence-point bounds warnings in source code.

    Returns:
        True when bounds checks are detected; otherwise False.
    """
    print("CHECK 4: V_eq Bounds Checking")

    source = inspect.getsource(detect_equivalence_point)

    if "warning" in source.lower() and "edge" in source.lower():
        print("  ✓ V_eq bounds checking implemented\n")
        return True
    else:
        print("  ✗ V_eq bounds checking not found\n")
        return False


def check_buffer_min_points():
    """Check enforcement of minimum buffer points in regression.

    Returns:
        True when insufficient points raise a ValueError; otherwise False.
    """
    print("CHECK 5: Buffer Region Minimum Points")

    veq = 25.0
    pka = 5.0

    step_df = pd.DataFrame(
        {
            "Volume (cm³)": [10.0, 12.0],
            "pH_step": [4.9, 5.1],
        }
    )

    try:
        fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka)
        print("  ✗ Did not raise error for insufficient points\n")
        return False
    except ValueError as e:
        if "insufficient" in str(e).lower() and "3" in str(e):
            print(f"  ✓ Correctly raised ValueError: {e}\n")
            return True
        else:
            print(f"  ✗ Wrong error message: {e}\n")
            return False


def check_uncertainty_documentation():
    """Check documentation of systematic uncertainty handling.

    Returns:
        True when documentation references systematic uncertainty; otherwise False.
    """
    print("CHECK 6: Uncertainty Type Documentation")

    stats_doc = calculate_statistics.__doc__
    slope_doc = slope_uncertainty_from_endpoints.__doc__

    checks = []

    if "systematic" in stats_doc.lower():
        print("  ✓ Statistics function documents systematic uncertainty")
        checks.append(True)
    else:
        print("  ✗ Statistics function missing systematic uncertainty docs")
        checks.append(False)

    if "systematic" in slope_doc.lower():
        print("  ✓ Slope uncertainty documents systematic type")
        checks.append(True)
    else:
        print("  ✗ Slope uncertainty missing systematic type docs")
        checks.append(False)

    print()
    return all(checks)


def check_plotting_validation():
    """Check plotting input validation behavior.

    Returns:
        True when invalid inputs raise expected errors; otherwise False.
    """
    print("CHECK 7: Plotting Input Validation")

    try:
        plot_titration_curves([])
        print("  ✗ Did not raise error for empty results\n")
        return False
    except ValueError as e:
        if "empty" in str(e).lower():
            print(f"  ✓ Correctly validates empty input: {e}")
        else:
            print(f"  ? Raised error but unclear: {e}")

    try:
        plot_titration_curves([{"data": pd.DataFrame()}])
        print("  ✗ Did not raise error for missing keys\n")
        return False
    except KeyError as e:
        if "required" in str(e).lower():
            print(f"  ✓ Correctly validates required keys: {e}\n")
            return True
        else:
            print(f"  ? Raised KeyError but unclear: {e}\n")
            return False


def check_apparent_pka_labels():
    """Check 8: Apparent pKa notation in outputs."""
    print("CHECK 8: Apparent pKa Notation")

    from salty.schema import ResultColumns

    cols = ResultColumns()

    checks = []

    if "apparent" in cols.pka_app.lower():
        print(f"  ✓ Schema uses apparent pKa: '{cols.pka_app}'")
        checks.append(True)
    else:
        print(f"  ✗ Schema missing 'apparent': '{cols.pka_app}'")
        checks.append(False)

    plot_doc = plot_statistical_summary.__doc__
    if "pKa" in plot_doc or "apparent" in plot_doc.lower():
        print("  ✓ Plotting docs reference apparent pKa")
        checks.append(True)
    else:
        print("  ✗ Plotting docs missing apparent pKa reference")
        checks.append(False)

    print()
    return all(checks)


def check_no_silent_nans():
    """Check 9: No silent NaN success paths."""
    print("CHECK 9: No Silent NaN Success Paths")

    veq = 25.0

    step_df_empty = pd.DataFrame(
        {
            "Volume (cm³)": [],
            "pH_step": [],
        }
    )

    try:
        fit_henderson_hasselbalch(step_df_empty, veq, pka_app_guess=5.0)
        print("  ✗ Empty data did not raise error\n")
        return False
    except (ValueError, KeyError) as e:
        print(f"  ✓ Empty data raises error: {type(e).__name__}\n")
        return True


def main():
    """Run all validation checks."""
    print("=" * 70)
    print("FINAL REVIEW CHECKLIST - Scientific Refactoring Validation")
    print("=" * 70)
    print()

    checks = [
        ("Interpretation Guardrails", check_interpretation_guardrails),
        ("Two-Stage Protocol", check_two_stage_protocol),
        ("Slope Warning", check_slope_warning),
        ("V_eq Bounds Checking", check_veq_bounds_warning),
        ("Buffer Min Points", check_buffer_min_points),
        ("Uncertainty Documentation", check_uncertainty_documentation),
        ("Plotting Validation", check_plotting_validation),
        ("Apparent pKa Labels", check_apparent_pka_labels),
        ("No Silent NaNs", check_no_silent_nans),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ Check failed with exception: {e}\n")
            results.append((name, False))

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    total = len(results)
    passed = sum(1 for _, r in results if r)

    print()
    print(f"Total: {passed}/{total} checks passed")

    if passed == total:
        print("\n✓ ALL CHECKS PASSED - Code ready for review")
        return 0
    else:
        print(f"\n✗ {total - passed} CHECKS FAILED - Review required")
        return 1


if __name__ == "__main__":
    exit(main())
