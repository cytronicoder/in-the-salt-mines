"""
Tests for chemical invariant validation and failure modes.

These tests ensure that:
1. Chemical interpretations are correct and explicitly bounded
2. Numerical procedures are scientifically justified
3. Code fails loudly on invalid science
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from salty.chemistry.buffer_region import select_buffer_region
from salty.chemistry.hh_model import fit_henderson_hasselbalch


class TestChemicalInvariants:
    """Tests for chemical invariants in Henderson-Hasselbalch analysis."""

    def test_hh_slope_near_unity_ideal_buffer(self):
        """
        CHEMICAL INVARIANT TEST:
        Henderson-Hasselbalch slope should be ≈ 1.0 for ideal buffer system.
        """
        # Generate synthetic ideal buffer data
        veq = 25.0
        pka_true = 5.0
        volumes = np.linspace(5.0, 23.0, 30)

        # Ideal HH: pH = pKa + log10(V / (Veq - V))
        log_ratios = np.log10(volumes / (veq - volumes))
        pH_ideal = pka_true + log_ratios

        step_df = pd.DataFrame(
            {
                "Volume (cm³)": volumes,
                "pH_step": pH_ideal,
            }
        )

        result = fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_true)

        slope = result["slope_reg"]
        pka_fit = result["pka_app"]

        # Slope should be very close to 1.0 for ideal buffer
        assert abs(slope - 1.0) < 0.05, (
            f"HH slope ({slope:.3f}) deviates from unity for ideal buffer. "
            "Chemical invariant violated."
        )

        # pKa should match true value
        assert abs(pka_fit - pka_true) < 0.05, (
            f"Fitted pKa_app ({pka_fit:.3f}) differs from true pKa ({pka_true:.3f}). "
            "Regression accuracy issue."
        )

    def test_hh_slope_warning_non_ideal(self):
        """
        Test that significant slope deviations trigger warning.
        """
        # Generate data with non-ideal slope
        veq = 25.0
        pka_true = 5.0
        volumes = np.linspace(5.0, 23.0, 30)

        log_ratios = np.log10(volumes / (veq - volumes))
        # Introduce non-ideal slope (m = 0.7)
        pH_non_ideal = pka_true + 0.7 * log_ratios

        step_df = pd.DataFrame(
            {
                "Volume (cm³)": volumes,
                "pH_step": pH_non_ideal,
            }
        )

        # Should trigger warning about slope deviation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_true)

            # Check that warning was issued
            assert len(w) == 1
            assert "slope" in str(w[0].message).lower()
            assert "unity" in str(w[0].message).lower()

    def test_buffer_region_bounds(self):
        """
        Test that buffer region selection enforces |pH - pKa_app| ≤ 1.
        """
        pka_app = 5.0
        pH_values = np.array([3.0, 3.9, 4.0, 4.5, 5.0, 5.5, 6.0, 6.1, 7.0])

        mask = select_buffer_region(pH_values, pka_app)

        # Only pH values within ±1 of pKa should be selected
        expected = np.abs(pH_values - pka_app) <= 1.0
        np.testing.assert_array_equal(mask, expected)

        # Should include 4.0 to 6.0 (inclusive)
        assert mask[2] == True  # pH = 4.0
        assert mask[6] == True  # pH = 6.0
        assert mask[1] == False  # pH = 3.9
        assert mask[7] == False  # pH = 6.1


class TestFailureModes:
    """Tests for proper failure behavior on invalid inputs."""

    def test_insufficient_buffer_points_raises_error(self):
        """
        FAILURE TEST:
        Insufficient buffer points should raise ValueError, not silently return NaN.
        """
        # Create data with only 2 points in buffer region
        veq = 25.0
        pka_guess = 5.0

        # Only 2 points near pKa
        step_df = pd.DataFrame(
            {
                "Volume (cm³)": [10.0, 12.0],
                "pH_step": [4.9, 5.1],
            }
        )

        with pytest.raises(ValueError) as exc_info:
            fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_guess)

        assert "insufficient" in str(exc_info.value).lower()
        assert "3" in str(exc_info.value)  # Minimum 3 points required

    def test_invalid_veq_raises_error(self):
        """
        Test that invalid V_eq values raise appropriate errors.
        """
        step_df = pd.DataFrame(
            {
                "Volume (cm³)": [5.0, 10.0, 15.0, 20.0],
                "pH_step": [4.5, 4.8, 5.2, 5.5],
            }
        )

        # Negative V_eq
        with pytest.raises(ValueError) as exc_info:
            fit_henderson_hasselbalch(step_df, -5.0, pka_app_guess=5.0)
        assert "positive" in str(exc_info.value).lower()

        # Zero V_eq
        with pytest.raises(ValueError) as exc_info:
            fit_henderson_hasselbalch(step_df, 0.0, pka_app_guess=5.0)
        assert "positive" in str(exc_info.value).lower()

        # NaN V_eq
        with pytest.raises(ValueError) as exc_info:
            fit_henderson_hasselbalch(step_df, np.nan, pka_app_guess=5.0)
        assert "finite" in str(exc_info.value).lower()

    def test_invalid_pka_guess_raises_error(self):
        """
        Test that invalid pKa_app guess raises error.
        """
        step_df = pd.DataFrame(
            {
                "Volume (cm³)": [5.0, 10.0, 15.0, 20.0],
                "pH_step": [4.5, 4.8, 5.2, 5.5],
            }
        )

        with pytest.raises(ValueError) as exc_info:
            fit_henderson_hasselbalch(step_df, 25.0, pka_app_guess=np.nan)
        assert "finite" in str(exc_info.value).lower()

    def test_buffer_region_invalid_pka_raises_error(self):
        """
        Test that invalid pKa_app in buffer region selection raises error.
        """
        pH_values = np.array([4.0, 5.0, 6.0])

        with pytest.raises(ValueError) as exc_info:
            select_buffer_region(pH_values, np.nan)
        assert "finite" in str(exc_info.value).lower()


class TestInterpretationMetadata:
    """Tests for interpretation metadata in results."""

    def test_result_indicates_apparent_pka(self):
        """
        INTERPRETATION TEST:
        Returned result metadata should clearly indicate pKa is apparent.
        """
        veq = 25.0
        pka_true = 5.0
        volumes = np.linspace(5.0, 23.0, 30)
        log_ratios = np.log10(volumes / (veq - volumes))
        pH_ideal = pka_true + log_ratios

        step_df = pd.DataFrame(
            {
                "Volume (cm³)": volumes,
                "pH_step": pH_ideal,
            }
        )

        result = fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_true)

        # Result should have 'pka_app' key (not 'pka')
        assert "pka_app" in result
        assert "pka" not in result  # Should not use ambiguous 'pka' key

        # Function docstring should mention apparent pKa
        doc = fit_henderson_hasselbalch.__doc__
        assert "apparent" in doc.lower() or "pKa_app" in doc
        assert "operational" in doc.lower() or "concentration-based" in doc.lower()


class TestTwoStageProtocol:
    """Tests for two-stage pKa_app extraction protocol."""

    def test_two_stage_protocol_documented(self):
        """
        Test that two-stage protocol is documented in function docstrings.
        """
        # Check HH model docstring
        hh_doc = fit_henderson_hasselbalch.__doc__
        assert "stage" in hh_doc.lower()
        assert "half-equivalence" in hh_doc.lower()
        assert "coarse" in hh_doc.lower() or "initial" in hh_doc.lower()

        # Check buffer region docstring
        buffer_doc = select_buffer_region.__doc__
        assert "pKa_app" in buffer_doc or "pka_app" in buffer_doc.lower()

    def test_pka_guess_defines_buffer_region(self):
        """
        Test that pKa_app guess is used to define buffer region (Stage 2).
        """
        veq = 25.0
        pka_initial = 5.0  # Stage 1 estimate

        # Create data spanning pH 3 to 7
        volumes = np.linspace(5.0, 23.0, 40)
        log_ratios = np.log10(volumes / (veq - volumes))
        pH_values = pka_initial + log_ratios

        step_df = pd.DataFrame(
            {
                "Volume (cm³)": volumes,
                "pH_step": pH_values,
            }
        )

        # Fit with initial guess
        result = fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_initial)

        # Buffer region should be centered around pka_initial ± 1
        buffer_df = result["buffer_df"]
        buffer_pH = buffer_df["pH_step"].values

        # All buffer pH values should be within ±1 of initial guess
        assert np.all(
            np.abs(buffer_pH - pka_initial) <= 1.0
        ), "Buffer region not properly defined by pKa_app initial guess"

        # Should have reasonable number of points (not too few, not all)
        n_buffer = len(buffer_df)
        n_total = len(step_df)
        assert (
            5 <= n_buffer < n_total
        ), f"Buffer region size ({n_buffer}/{n_total}) unreasonable"
