"""Validate chemical invariants and failure modes in the HH workflow."""

import warnings

import numpy as np
import pandas as pd
import pytest

from salty.chemistry.buffer_region import select_buffer_region
from salty.chemistry.hh_model import fit_henderson_hasselbalch


class TestChemicalInvariants:
    """Check chemically required invariants in Henderson-Hasselbalch fits."""

    def test_hh_slope_near_unity_ideal_buffer(self):
        """Verify that an ideal buffer yields a slope near unity.

        Returns:
            None.
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

        slope = result["slope_reg"]
        pka_fit = result["pka_app"]

        assert abs(slope - 1.0) < 0.05, (
            f"HH slope ({slope:.3f}) deviates from unity for ideal buffer. "
            "Chemical invariant violated."
        )

        assert abs(pka_fit - pka_true) < 0.05, (
            f"Fitted pKa_app ({pka_fit:.3f}) differs from true pKa ({pka_true:.3f}). "
            "Regression accuracy issue."
        )

    def test_hh_slope_warning_non_ideal(self):
        """Confirm that non-ideal slopes trigger diagnostic warnings.

        Returns:
            None.
        """
        veq = 25.0
        pka_true = 5.0
        volumes = np.linspace(5.0, 23.0, 30)

        log_ratios = np.log10(volumes / (veq - volumes))
        pH_non_ideal = pka_true + 0.7 * log_ratios

        step_df = pd.DataFrame(
            {
                "Volume (cm³)": volumes,
                "pH_step": pH_non_ideal,
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_true)

            assert len(w) == 1
            assert "slope" in str(w[0].message).lower()
            assert "unity" in str(w[0].message).lower()

    def test_buffer_region_bounds(self):
        """Ensure buffer selection enforces |pH - pKa_app| ≤ 1.

        Returns:
            None.
        """
        pka_app = 5.0
        pH_values = np.array([3.0, 3.9, 4.0, 4.5, 5.0, 5.5, 6.0, 6.1, 7.0])

        mask = select_buffer_region(pH_values, pka_app)

        expected = np.abs(pH_values - pka_app) <= 1.0
        np.testing.assert_array_equal(mask, expected)

        assert mask[2]
        assert mask[6]
        assert not mask[1]
        assert not mask[7]


class TestFailureModes:
    """Verify explicit failure behavior on invalid scientific inputs."""

    def test_insufficient_buffer_points_raises_error(self):
        """Raise an error when the buffer region is underpopulated.

        Returns:
            None.
        """
        veq = 25.0
        pka_guess = 5.0

        step_df = pd.DataFrame(
            {
                "Volume (cm³)": [10.0, 12.0],
                "pH_step": [4.9, 5.1],
            }
        )

        with pytest.raises(ValueError) as exc_info:
            fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_guess)

        assert "insufficient" in str(exc_info.value).lower()
        assert "3" in str(exc_info.value)

    def test_invalid_veq_raises_error(self):
        """Raise errors for non-physical or non-finite V_eq values.

        Returns:
            None.
        """
        step_df = pd.DataFrame(
            {
                "Volume (cm³)": [5.0, 10.0, 15.0, 20.0],
                "pH_step": [4.5, 4.8, 5.2, 5.5],
            }
        )

        with pytest.raises(ValueError) as exc_info:
            fit_henderson_hasselbalch(step_df, -5.0, pka_app_guess=5.0)
        assert "positive" in str(exc_info.value).lower()

        with pytest.raises(ValueError) as exc_info:
            fit_henderson_hasselbalch(step_df, 0.0, pka_app_guess=5.0)
        assert "positive" in str(exc_info.value).lower()

        with pytest.raises(ValueError) as exc_info:
            fit_henderson_hasselbalch(step_df, np.nan, pka_app_guess=5.0)
        assert "finite" in str(exc_info.value).lower()

    def test_invalid_pka_guess_raises_error(self):
        """Raise errors when the pKa_app initial guess is invalid.

        Returns:
            None.
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
        """Raise errors when buffer-region selection receives invalid pKa_app.

        Returns:
            None.
        """
        pH_values = np.array([4.0, 5.0, 6.0])

        with pytest.raises(ValueError) as exc_info:
            select_buffer_region(pH_values, np.nan)
        assert "finite" in str(exc_info.value).lower()


class TestInterpretationMetadata:
    """Ensure interpretation guardrails are visible in model metadata."""

    def test_result_indicates_apparent_pka(self):
        """Confirm that results explicitly indicate apparent pKa_app values.

        Returns:
            None.
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

        assert "pka_app" in result
        assert "pka" not in result

        doc = fit_henderson_hasselbalch.__doc__
        assert "apparent" in doc.lower() or "pKa_app" in doc
        assert "operational" in doc.lower() or "concentration-based" in doc.lower()


class TestTwoStageProtocol:
    """Verify documentation of the two-stage pKa_app protocol."""

    def test_two_stage_protocol_documented(self):
        """Check that two-stage workflow is documented in docstrings.

        Returns:
            None.
        """
        hh_doc = fit_henderson_hasselbalch.__doc__
        assert "stage" in hh_doc.lower()
        assert "half-equivalence" in hh_doc.lower()
        assert "coarse" in hh_doc.lower() or "initial" in hh_doc.lower()

        buffer_doc = select_buffer_region.__doc__
        assert "pKa_app" in buffer_doc or "pka_app" in buffer_doc.lower()

    def test_pka_guess_defines_buffer_region(self):
        """Confirm that Stage 1 pKa_app defines the Stage 2 buffer window.

        Returns:
            None.
        """
        veq = 25.0
        pka_initial = 5.0

        volumes = np.linspace(5.0, 23.0, 40)
        log_ratios = np.log10(volumes / (veq - volumes))
        pH_values = pka_initial + log_ratios

        step_df = pd.DataFrame(
            {
                "Volume (cm³)": volumes,
                "pH_step": pH_values,
            }
        )

        result = fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_initial)

        buffer_df = result["buffer_df"]
        buffer_pH = buffer_df["pH_step"].values

        assert np.all(
            np.abs(buffer_pH - pka_initial) <= 1.0
        ), "Buffer region not properly defined by pKa_app initial guess"

        n_buffer = len(buffer_df)
        n_total = len(step_df)
        assert (
            5 <= n_buffer < n_total
        ), f"Buffer region size ({n_buffer}/{n_total}) unreasonable"
