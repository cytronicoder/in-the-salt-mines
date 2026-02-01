"""Test ionic strength calculations for NaCl solutions."""

import math

import pytest

from salty.chemistry.ionic_strength import ionic_strength_general, ionic_strength_nacl


class TestIonicStrengthNaCl:
    """Test ionic strength calculations for NaCl (1:1 electrolyte)."""

    def test_zero_concentration(self):
        """For 0.00 M NaCl, ionic strength should be 0.0."""
        assert ionic_strength_nacl(0.0) == 0.0

    def test_nacl_equals_concentration(self):
        """For NaCl (1:1 electrolyte), μ = [NaCl]."""
        # Test all experimental concentrations from IA
        test_concentrations = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
        for conc in test_concentrations:
            mu = ionic_strength_nacl(conc)
            assert math.isclose(mu, conc, abs_tol=1e-9)

    def test_negative_concentration_raises(self):
        """Negative concentrations should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            ionic_strength_nacl(-0.1)

    def test_infinite_concentration_raises(self):
        """Non-finite concentrations should raise ValueError."""
        with pytest.raises(ValueError, match="must be finite"):
            ionic_strength_nacl(math.inf)

        with pytest.raises(ValueError, match="must be finite"):
            ionic_strength_nacl(math.nan)

    def test_type_error_for_non_numeric(self):
        """Non-numeric input should raise TypeError."""
        with pytest.raises(TypeError, match="must be numeric"):
            ionic_strength_nacl("0.20")  # type: ignore


class TestIonicStrengthGeneral:
    """Test general ionic strength calculations."""

    def test_empty_solution(self):
        """Empty dictionary should return 0.0."""
        assert ionic_strength_general({}) == 0.0

    def test_nacl_via_general_formula(self):
        """Verify NaCl calculation using general formula.

        For NaCl: μ = 0.5 * ([Na+] * 1^2 + [Cl-] * 1^2)
        If [NaCl] = 0.20 M, then [Na+] = [Cl-] = 0.20 M
        μ = 0.5 * (0.20 * 1 + 0.20 * 1) = 0.20 M
        """
        ions = {"Na+": (0.20, 1), "Cl-": (0.20, -1)}
        mu = ionic_strength_general(ions)
        assert math.isclose(mu, 0.20, abs_tol=1e-9)

    def test_calcium_chloride(self):
        """Test CaCl2 (2:1 electrolyte).

        For CaCl2 → Ca2+ + 2Cl-
        If [CaCl2] = 0.10 M, then [Ca2+] = 0.10 M, [Cl-] = 0.20 M
        μ = 0.5 * ([Ca2+] * 2^2 + [Cl-] * 1^2)
        μ = 0.5 * (0.10 * 4 + 0.20 * 1) = 0.30 M
        """
        ions = {"Ca2+": (0.10, 2), "Cl-": (0.20, -1)}
        mu = ionic_strength_general(ions)
        assert math.isclose(mu, 0.30, abs_tol=1e-9)

    def test_sodium_sulfate(self):
        """Test Na2SO4 (1:2 electrolyte).

        For Na2SO4 → 2Na+ + SO4^2⁻
        If [Na2SO4] = 0.10 M, then [Na+] = 0.20 M, [SO4^2⁻] = 0.10 M
        μ = 0.5 * ([Na+] * 1^2 + [SO4^2⁻] * 2^2)
        μ = 0.5 * (0.20 * 1 + 0.10 * 4) = 0.30 M
        """
        ions = {"Na+": (0.20, 1), "SO4^2⁻": (0.10, -2)}
        mu = ionic_strength_general(ions)
        assert math.isclose(mu, 0.30, abs_tol=1e-9)

    def test_negative_concentration_raises(self):
        """Negative ion concentrations should raise ValueError."""
        ions = {"Na+": (-0.10, 1), "Cl-": (0.10, -1)}
        with pytest.raises(ValueError, match="cannot be negative"):
            ionic_strength_general(ions)

    def test_zero_charge_raises(self):
        """Zero charge should raise ValueError."""
        ions = {"neutral": (0.10, 0)}
        with pytest.raises(ValueError, match="cannot be zero"):
            ionic_strength_general(ions)

    def test_mixed_electrolytes(self):
        """Test solution with multiple electrolytes.

        Example: 0.10 M NaCl + 0.05 M CaCl2
        Na+: 0.10 M, Cl-: 0.20 M, Ca2+: 0.05 M
        μ = 0.5 * (0.10*1 + 0.20*1 + 0.05*4) = 0.25 M
        """
        ions = {
            "Na+": (0.10, 1),
            "Ca2+": (0.05, 2),
            "Cl-": (0.20, -1),
        }
        mu = ionic_strength_general(ions)
        assert math.isclose(mu, 0.25, abs_tol=1e-9)


class TestExperimentalConditions:
    """Test ionic strength calculations for IA experimental conditions."""

    def test_all_experimental_nacl_concentrations(self):
        """Verify ionic strength for all [NaCl] used in IA."""
        expected_values = [
            (0.00, 0.00),
            (0.20, 0.20),
            (0.40, 0.40),
            (0.60, 0.60),
            (0.80, 0.80),
            (1.00, 1.00),
        ]

        for nacl_conc, expected_mu in expected_values:
            mu = ionic_strength_nacl(nacl_conc)
            assert math.isclose(mu, expected_mu, abs_tol=1e-9), (
                f"[NaCl] = {nacl_conc} M should give μ = {expected_mu} M, "
                f"got {mu} M"
            )

    def test_nacl_range_validity(self):
        """All experimental NaCl concentrations should be valid."""
        from salty.stats.uncertainty import NACL_CONCENTRATIONS_M

        for conc in NACL_CONCENTRATIONS_M:
            mu = ionic_strength_nacl(conc)
            assert mu >= 0
            assert math.isfinite(mu)
            assert math.isclose(mu, conc)
