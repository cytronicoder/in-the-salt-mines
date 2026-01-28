import numpy as np
import pandas as pd
import pytest

from salty.chemistry.hh_model import fit_henderson_hasselbalch


def test_hh_regression_slope_close_to_one():
    veq = 10.0
    pka_app = 4.75
    log_ratios = np.array([-0.5, -0.25, 0.0, 0.25, 0.5])
    ratios = 10**log_ratios
    volumes = ratios * veq / (1.0 + ratios)
    pH = pka_app + log_ratios

    step_df = pd.DataFrame({"Volume (cm³)": volumes, "pH_step": pH})
    fit = fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_app)
    assert abs(fit["slope_reg"] - 1.0) < 0.05


def test_invalid_buffer_region_raises():
    veq = 10.0
    pka_app = 4.75
    log_ratios = np.array([-0.5, 0.0, 0.5])
    ratios = 10**log_ratios
    volumes = ratios * veq / (1.0 + ratios)
    pH = np.full_like(volumes, 8.0)

    step_df = pd.DataFrame({"Volume (cm³)": volumes, "pH_step": pH})
    with pytest.raises(ValueError):
        fit_henderson_hasselbalch(step_df, veq, pka_app_guess=pka_app)
