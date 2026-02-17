"""Error diagnosis pipeline for pH-at-half-equivalence replicate data."""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - exercised by fallback logic
    scipy_stats = None
    HAVE_SCIPY = False

try:
    from salty.plotting.style import apply_style, clean_axis

    HAVE_STYLE_HELPERS = True
except Exception:  # pragma: no cover - non-critical optional integration
    HAVE_STYLE_HELPERS = False


EXPECTED_CONCENTRATIONS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_OUTPUT_DIR = Path("output") / "error_diagnosis"
EPSILON = 1e-12
CONCENTRATION_TOL = 1e-9

LABEL_NACL_M = r"$[\mathrm{NaCl}]\;(\mathrm{M})$"
LABEL_PKA_APP_LONG = r"$pH$ at half-equivalence $(pK_{a,\mathrm{app}})$"
LABEL_PKA_APP_SHORT = r"$pK_{a,\mathrm{app}}$"
LABEL_RESID_LINEAR = r"$y-\hat{y}$ (linear)"
LABEL_FITTED_LINEAR = r"Fitted values (linear) $\hat{y}$"
LABEL_SD_PKA_APP = r"$\mathrm{SD}(pK_{a,\mathrm{app}})$"

PLOT_LABEL_FONTSIZE = 12
PLOT_TICK_FONTSIZE = 11
PLOT_LINEWIDTH = 1.8
PLOT_ZERO_LINEWIDTH = 0.9
PLOT_MARKER_SIZE = 30
PLOT_ALPHA = 0.82
BOXPLOT_WIDTH = 0.12
X_LEVEL_PAD = 0.05


@dataclass(frozen=True)
class ModelFit:
    """Container for regression fit outputs."""

    model: str
    degree: int
    beta: np.ndarray
    yhat: np.ndarray
    resid: np.ndarray
    ss_res: float
    df_res: int
    r2: float
    notes: str = ""

    @property
    def beta0(self) -> float:
        return float(self.beta[0]) if len(self.beta) >= 1 else math.nan

    @property
    def beta1(self) -> float:
        return float(self.beta[1]) if len(self.beta) >= 2 else math.nan

    @property
    def beta2(self) -> float:
        return float(self.beta[2]) if len(self.beta) >= 3 else math.nan


def _resolve_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: tuple[str, ...],
    label: str,
) -> str | None:
    """Resolve one input column using explicit name or canonical candidates."""
    if explicit is not None:
        if explicit not in frame.columns:
            raise ValueError(
                f"Column '{explicit}' not found for {label}. "
                f"Available columns: {list(frame.columns)}"
            )
        return explicit

    lookup = {str(col).strip().lower(): str(col) for col in frame.columns}
    for name in candidates:
        found = lookup.get(name.lower())
        if found is not None:
            return found
    return None


def _coerce_concentration(series: pd.Series) -> pd.Series:
    """Parse concentration values from numeric or tokenized string columns."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float)

    extracted = (
        series.astype(str).str.extract(r"(-?\d+(?:\.\d+)?)", expand=False).astype(str)
    )
    parsed = pd.to_numeric(extracted, errors="coerce")
    return numeric.fillna(parsed).astype(float)


def _ensure_plot_style() -> None:
    """Apply repository plotting style when available."""
    if HAVE_STYLE_HELPERS:
        apply_style(font_scale=1.0, context="paper")
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "mathtext.default": "regular",
            "axes.labelsize": PLOT_LABEL_FONTSIZE,
            "xtick.labelsize": PLOT_TICK_FONTSIZE,
            "ytick.labelsize": PLOT_TICK_FONTSIZE,
            "axes.linewidth": 1.0,
            "savefig.dpi": 300,
            "grid.linestyle": ":",
            "grid.alpha": 0.20,
            "lines.linewidth": PLOT_LINEWIDTH,
        }
    )


def _finish_axis(ax: Any, grid_axis: str = "both") -> None:
    """Apply axis cleanup with optional repository helper fallback."""
    if HAVE_STYLE_HELPERS:
        clean_axis(ax, grid_axis=grid_axis, nbins_x=6, nbins_y=6)
    else:  # pragma: no cover - style helper exists in this repository
        ax.grid(True, alpha=0.25, linestyle=":")


def format_axes(
    ax: Any,
    xlabel: str,
    ylabel: str,
    *,
    concentration_axis: bool = False,
    grid_axis: str = "both",
) -> None:
    """Apply consistent axis labels and ticks for publication-ready plots."""
    ax.set_xlabel(xlabel, fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=PLOT_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PLOT_TICK_FONTSIZE)
    _finish_axis(ax, grid_axis=grid_axis)
    if concentration_axis:
        x_levels = np.array(EXPECTED_CONCENTRATIONS, dtype=float)
        ax.set_xticks(x_levels)
        ax.set_xticklabels([f"{value:.1f}" for value in x_levels])
        ax.set_xlim(float(x_levels[0] - X_LEVEL_PAD), float(x_levels[-1] + X_LEVEL_PAD))


def _legend_conflict_keys(legend: Any) -> set[str]:
    """Return candidate keys that may collide with legend placement."""
    if legend is None:
        return set()
    loc = getattr(legend, "_loc", None)
    mapping = {
        1: {"ur"},
        2: {"ul"},
        3: {"ll"},
        4: {"lr"},
        "upper right": {"ur"},
        "upper left": {"ul"},
        "lower left": {"ll"},
        "lower right": {"lr"},
    }
    return mapping.get(loc, set())


def add_stats_box(
    ax: Any,
    lines: list[str],
    *,
    loc: str = "upper left",
    avoid_legend: bool = True,
    legend: Any = None,
    x_data: np.ndarray | None = None,
    y_data: np.ndarray | None = None,
) -> None:
    """Add p-value stats text box with simple collision-avoidance logic."""
    candidate_positions = {
        "upper left": [
            ("ul", 0.02, 0.98, "left", "top"),
            ("ur", 0.62, 0.98, "left", "top"),
            ("ll", 0.02, 0.25, "left", "bottom"),
            ("lr", 0.62, 0.25, "left", "bottom"),
        ],
        "upper right": [
            ("ur", 0.62, 0.98, "left", "top"),
            ("ul", 0.02, 0.98, "left", "top"),
            ("lr", 0.62, 0.25, "left", "bottom"),
            ("ll", 0.02, 0.25, "left", "bottom"),
        ],
    }
    candidates = candidate_positions.get(loc, candidate_positions["upper left"])
    blocked = _legend_conflict_keys(legend) if avoid_legend else set()
    usable = [candidate for candidate in candidates if candidate[0] not in blocked]
    if not usable:
        usable = candidates

    score_by_key: dict[str, float] = {}
    if x_data is not None and y_data is not None:
        ax.figure.canvas.draw()
        x_arr = np.asarray(x_data, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)
        finite = np.isfinite(x_arr) & np.isfinite(y_arr)
        if np.any(finite):
            data_xy = np.column_stack([x_arr[finite], y_arr[finite]])
            data_axes = ax.transAxes.inverted().transform(
                ax.transData.transform(data_xy)
            )
            box_w = 0.34
            box_h = 0.11 + 0.04 * len(lines)
            for key, x0, y0, _ha, va in usable:
                y_min = y0 - box_h if va == "top" else y0
                y_max = y0 if va == "top" else y0 + box_h
                x_min = x0
                x_max = x0 + box_w
                inside = (
                    (data_axes[:, 0] >= x_min)
                    & (data_axes[:, 0] <= x_max)
                    & (data_axes[:, 1] >= y_min)
                    & (data_axes[:, 1] <= y_max)
                )
                score_by_key[key] = float(np.sum(inside))

    if score_by_key:
        ordered_keys = [item[0] for item in usable]
        best_key = min(
            ordered_keys,
            key=lambda key: (score_by_key.get(key, 0.0), ordered_keys.index(key)),
        )
        chosen = next(item for item in usable if item[0] == best_key)
    else:
        chosen = usable[0]

    _key, xpos, ypos, ha, va = chosen
    ax.text(
        xpos,
        ypos,
        "\n".join(lines),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=9,
        clip_on=False,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.8,
            "edgecolor": "0.7",
        },
    )


def standardize_long_form(
    raw_df: pd.DataFrame,
    concentration_col: str | None = None,
    replicate_col: str | None = None,
    y_col: str | None = None,
) -> pd.DataFrame:
    """Standardize supported input table formats into long-form columns."""
    conc_candidates = (
        "concentration",
        "nacl concentration (m)",
        "nacl_conc",
        "nacl",
        "[nacl]",
    )
    rep_candidates = ("replicate", "run", "run name", "run_name", "trial")
    y_candidates = (
        "y",
        "apparent pka",
        "pka_app",
        "half_eq_ph",
        "ph at half-equivalence",
        "pka at half-equivalence",
    )

    conc_name = _resolve_column(
        raw_df, concentration_col, conc_candidates, label="concentration"
    )
    y_name = _resolve_column(raw_df, y_col, y_candidates, label="response y")
    if conc_name is None:
        raise ValueError(
            "Could not detect concentration column. Provide --concentration-col."
        )
    if y_name is None:
        raise ValueError("Could not detect y column. Provide --y-col.")

    rep_name = _resolve_column(raw_df, replicate_col, rep_candidates, label="replicate")

    concentration = _coerce_concentration(raw_df[conc_name])
    y_values = pd.to_numeric(raw_df[y_name], errors="coerce").astype(float)

    if rep_name is None:
        replicate_values = (
            raw_df.assign(_conc=concentration)
            .groupby("_conc", sort=False)
            .cumcount()
            .add(1)
            .astype(int)
            .astype(str)
        )
    else:
        replicate_values = raw_df[rep_name].astype(str).str.strip()
        missing_replicate = replicate_values.eq("") | raw_df[rep_name].isna()
        if bool(missing_replicate.any()):
            fallback = (
                raw_df.assign(_conc=concentration)
                .groupby("_conc", sort=False)
                .cumcount()
                .add(1)
                .astype(int)
                .astype(str)
            )
            replicate_values = replicate_values.mask(missing_replicate, fallback)

    long_df = pd.DataFrame(
        {
            "concentration": concentration,
            "replicate": replicate_values,
            "y": y_values,
        }
    )
    long_df = long_df.reset_index(drop=True)
    return long_df


def validate_long_form(
    long_df: pd.DataFrame,
    expected_concentrations: tuple[float, ...] = EXPECTED_CONCENTRATIONS,
    min_replicates: int = 3,
    tol: float = CONCENTRATION_TOL,
) -> pd.DataFrame:
    """Validate long-form replicate data and return deterministic sorting."""
    required = {"concentration", "replicate", "y"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    df = long_df.copy()
    df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    if df["concentration"].isna().any():
        n_missing = int(df["concentration"].isna().sum())
        raise ValueError(f"Found {n_missing} rows with non-numeric concentration.")
    if df["y"].isna().any():
        n_missing = int(df["y"].isna().sum())
        raise ValueError(f"Found {n_missing} rows with missing/non-numeric y.")

    unique_levels = np.sort(df["concentration"].unique().astype(float))
    if len(unique_levels) != len(expected_concentrations):
        raise ValueError(
            "Expected exactly "
            f"{len(expected_concentrations)} unique concentrations but found "
            f"{len(unique_levels)}: {unique_levels.tolist()}"
        )

    expected = np.array(expected_concentrations, dtype=float)
    missing_expected = [
        float(val)
        for val in expected
        if not np.any(np.isclose(unique_levels, val, atol=tol, rtol=0.0))
    ]
    unexpected = [
        float(val)
        for val in unique_levels
        if not np.any(np.isclose(expected, val, atol=tol, rtol=0.0))
    ]
    if missing_expected or unexpected:
        raise ValueError(
            "Concentration levels do not match expected design. "
            f"Missing expected: {missing_expected}; unexpected: {unexpected}."
        )

    counts = df.groupby("concentration", sort=True)["y"].size()
    underpowered = counts[counts < int(min_replicates)]
    if not underpowered.empty:
        issue_text = ", ".join(
            f"{float(conc):.3f}M (n={int(n)})" for conc, n in underpowered.items()
        )
        raise ValueError(
            "Each concentration must have at least "
            f"{min_replicates} replicates. Offending levels: {issue_text}."
        )

    sorted_df = df.sort_values(["concentration", "replicate"], kind="mergesort")
    return sorted_df.reset_index(drop=True)


def compute_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-concentration summary statistics."""
    grouped = df.groupby("concentration", sort=True)["y"]
    summary = grouped.agg(
        n="count",
        mean_y="mean",
        sd_y=lambda s: float(np.std(np.asarray(s, dtype=float), ddof=1))
        if len(s) > 1
        else math.nan,
    ).reset_index()
    summary["n"] = summary["n"].astype(int)
    summary["var_y"] = summary["sd_y"] ** 2
    summary["cv_pct"] = np.where(
        np.isclose(summary["mean_y"], 0.0, atol=1e-12, rtol=0.0),
        math.nan,
        100.0 * summary["sd_y"] / summary["mean_y"],
    )
    return summary.sort_values("concentration").reset_index(drop=True)


def compute_pure_error(df: pd.DataFrame) -> dict[str, Any]:
    """Compute pooled within-group pure-error statistics."""
    means = df.groupby("concentration", sort=True)["y"].transform("mean")
    residuals = df["y"] - means
    ss_pure = float(np.sum(np.square(residuals.to_numpy(dtype=float))))

    counts = df.groupby("concentration", sort=True)["y"].size().to_numpy(dtype=int)
    df_pure = int(np.sum(counts - 1))
    ms_pure = ss_pure / df_pure if df_pure > 0 else math.nan

    return {
        "SS_pure": ss_pure,
        "df_pure": int(df_pure),
        "MS_pure": ms_pure,
    }


def _brown_forsythe_manual(groups: list[np.ndarray]) -> tuple[float, float]:
    """Manual Brown-Forsythe implementation for environments without SciPy."""
    k = len(groups)
    n_total = int(sum(len(g) for g in groups))
    if k < 2 or n_total <= k:
        return math.nan, math.nan

    centered = [np.abs(g - np.median(g)) for g in groups]
    zbar_groups = np.array([float(np.mean(z)) for z in centered], dtype=float)
    n_groups = np.array([len(z) for z in centered], dtype=float)
    zbar = float(np.sum([np.sum(z) for z in centered]) / n_total)

    ss_between = float(np.sum(n_groups * (zbar_groups - zbar) ** 2))
    ss_within = float(
        np.sum(
            [
                np.sum((z - np.mean(z)) ** 2)
                for z in centered
                if len(z) >= 1 and np.isfinite(np.mean(z))
            ]
        )
    )
    df1 = k - 1
    df2 = n_total - k
    if df1 <= 0 or df2 <= 0:
        return math.nan, math.nan
    ms_between = ss_between / df1
    ms_within = ss_within / df2
    stat = ms_between / ms_within if ms_within > 0 else math.inf

    if HAVE_SCIPY:
        pvalue = float(1.0 - scipy_stats.f.cdf(stat, df1, df2))
    else:  # pragma: no cover - hard to force in test env with SciPy installed
        pvalue = math.nan
    return float(stat), pvalue


def run_brown_forsythe(df: pd.DataFrame) -> dict[str, Any]:
    """Run robust equal-variance testing across concentration groups."""
    groups = [
        grp["y"].to_numpy(dtype=float)
        for _, grp in df.groupby("concentration", sort=True)
    ]
    if HAVE_SCIPY:
        statistic, pvalue = scipy_stats.levene(*groups, center="median")
        return {
            "stat": float(statistic),
            "pvalue": float(pvalue),
            "method": "Levene (Brown-Forsythe, center='median')",
            "notes": "",
        }

    statistic, pvalue = _brown_forsythe_manual(groups)
    warnings.warn(
        "SciPy not available: Brown-Forsythe p-value may be unavailable.",
        RuntimeWarning,
        stacklevel=2,
    )
    return {
        "stat": statistic,
        "pvalue": pvalue,
        "method": "Brown-Forsythe (manual median-centered Levene)",
        "notes": "SciPy unavailable; used manual Brown-Forsythe.",
    }


def baseline_one_sample_ttest(
    df: pd.DataFrame,
    pka_lit: float = 4.76,
    baseline_concentration: float = 0.0,
    tol: float = CONCENTRATION_TOL,
) -> dict[str, Any]:
    """Compare baseline 0.0 M replicates against literature pKa."""
    is_baseline = np.isclose(
        df["concentration"].to_numpy(dtype=float),
        float(baseline_concentration),
        atol=tol,
        rtol=0.0,
    )
    baseline = df.loc[is_baseline, "y"].to_numpy(dtype=float)
    n0 = int(len(baseline))
    if n0 == 0:
        raise ValueError(
            "No rows found for baseline concentration "
            f"{baseline_concentration:.3f} M."
        )

    mean0 = float(np.mean(baseline))
    sd0 = float(np.std(baseline, ddof=1)) if n0 > 1 else math.nan
    df_t = n0 - 1
    se = sd0 / math.sqrt(n0) if n0 > 1 and np.isfinite(sd0) else math.nan

    if n0 <= 1 or not np.isfinite(se) or np.isclose(se, 0.0):
        if np.isclose(mean0, pka_lit):
            t_stat = 0.0
        elif mean0 > pka_lit:
            t_stat = math.inf
        else:
            t_stat = -math.inf
        pvalue = math.nan
        ci95_low = math.nan
        ci95_high = math.nan
        notes = "Insufficient spread to compute p-value/CI."
    elif HAVE_SCIPY:
        ttest = scipy_stats.ttest_1samp(baseline, popmean=float(pka_lit))
        t_stat = float(ttest.statistic)
        pvalue = float(ttest.pvalue)
        t_crit = float(scipy_stats.t.ppf(0.975, df_t))
        ci95_low = mean0 - t_crit * se
        ci95_high = mean0 + t_crit * se
        notes = ""
    else:  # pragma: no cover - hard to force in test env with SciPy installed
        t_stat = float((mean0 - pka_lit) / se)
        pvalue = math.nan
        ci95_low = math.nan
        ci95_high = math.nan
        notes = "SciPy unavailable: reported t-stat only, p-value/CI are NaN."
        warnings.warn(
            "SciPy not available: baseline t-test p-value/CI are NaN.",
            RuntimeWarning,
            stacklevel=2,
        )

    return {
        "pKa_lit": float(pka_lit),
        "mean0": mean0,
        "sd0": sd0,
        "n0": int(n0),
        "t_stat": t_stat,
        "df": float(df_t),
        "pvalue": pvalue,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "notes": notes,
    }


def fit_polynomial_model(
    x: np.ndarray,
    y: np.ndarray,
    degree: int,
    model_name: str,
    weights: np.ndarray | None = None,
    notes: str = "",
) -> ModelFit:
    """Fit an unweighted or weighted polynomial model by least squares."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    design = np.vander(x_arr, N=degree + 1, increasing=True)

    if weights is None:
        beta, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(x_arr):
            raise ValueError("weights length must match x/y length.")
        if np.any(w <= 0) or np.any(~np.isfinite(w)):
            raise ValueError("weights must be finite and positive.")
        sqrt_w = np.sqrt(w)
        design_w = design * sqrt_w[:, None]
        y_w = y_arr * sqrt_w
        beta, *_ = np.linalg.lstsq(design_w, y_w, rcond=None)

    yhat = design @ beta
    resid = y_arr - yhat
    ss_res = float(np.sum(np.square(resid)))
    df_res = int(len(y_arr) - (degree + 1))

    ss_tot = float(np.sum(np.square(y_arr - np.mean(y_arr))))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else math.nan

    return ModelFit(
        model=model_name,
        degree=degree,
        beta=np.asarray(beta, dtype=float),
        yhat=np.asarray(yhat, dtype=float),
        resid=np.asarray(resid, dtype=float),
        ss_res=ss_res,
        df_res=df_res,
        r2=r2,
        notes=notes,
    )


def compute_lack_of_fit(
    linear_fit: ModelFit,
    pure_error: dict[str, float],
) -> dict[str, Any]:
    """Compute lack-of-fit ANOVA partition for the linear model."""
    ss_lof_raw = float(linear_fit.ss_res - pure_error["SS_pure"])
    notes = ""
    if ss_lof_raw < 0:
        ss_lof = 0.0
        notes = "SS_LOF was negative and clamped to 0."
    else:
        ss_lof = ss_lof_raw

    df_lof = int(linear_fit.df_res - int(pure_error["df_pure"]))
    if df_lof <= 0:
        return {
            "SS_res": float(linear_fit.ss_res),
            "df_res": float(linear_fit.df_res),
            "SS_LOF": float(ss_lof),
            "df_LOF": float(df_lof),
            "MS_pure": float(pure_error["MS_pure"]),
            "MS_LOF": math.nan,
            "F": math.nan,
            "pvalue": math.nan,
            "notes": (notes + " df_LOF <= 0; test undefined.").strip(),
        }

    ms_pure = float(pure_error["MS_pure"])
    ms_lof = float(ss_lof / df_lof)
    if ms_pure <= 0 or not np.isfinite(ms_pure):
        f_stat = math.nan
        pvalue = math.nan
        notes = (notes + " MS_pure <= 0; F-test undefined.").strip()
    else:
        f_stat = float(ms_lof / ms_pure)
        if HAVE_SCIPY:
            pvalue = float(
                1.0 - scipy_stats.f.cdf(f_stat, df_lof, int(pure_error["df_pure"]))
            )
        else:  # pragma: no cover - hard to force in test env with SciPy installed
            pvalue = math.nan
            warnings.warn(
                "SciPy not available: lack-of-fit p-value is NaN.",
                RuntimeWarning,
                stacklevel=2,
            )

    return {
        "SS_res": float(linear_fit.ss_res),
        "df_res": float(linear_fit.df_res),
        "SS_LOF": float(ss_lof),
        "df_LOF": float(df_lof),
        "MS_pure": ms_pure,
        "MS_LOF": ms_lof,
        "F": f_stat,
        "pvalue": pvalue,
        "notes": notes,
    }


def linear_vs_quadratic_f_test(
    linear_fit: ModelFit,
    quadratic_fit: ModelFit,
    n_obs: int,
) -> dict[str, Any]:
    """Run nested extra-sum-of-squares F-test: linear vs quadratic."""
    ss_diff_raw = float(linear_fit.ss_res - quadratic_fit.ss_res)
    notes = ""
    if ss_diff_raw < 0:
        ss_diff = 0.0
        notes = "SS_res_lin < SS_res_quad due rounding/noise; clamped numerator to 0."
    else:
        ss_diff = ss_diff_raw

    df1 = 1
    df2 = int(n_obs - 3)
    if df2 <= 0:
        return {
            "SS_res_lin": float(linear_fit.ss_res),
            "SS_res_quad": float(quadratic_fit.ss_res),
            "F": math.nan,
            "df1": float(df1),
            "df2": float(df2),
            "pvalue": math.nan,
            "notes": (notes + " df2 <= 0; test undefined.").strip(),
        }

    denom = float(quadratic_fit.ss_res / df2)
    if denom <= 0:
        f_stat = math.nan
        pvalue = math.nan
        notes = (notes + " Quadratic MS_res <= 0; F-test undefined.").strip()
    else:
        f_stat = float((ss_diff / df1) / denom)
        if HAVE_SCIPY:
            pvalue = float(1.0 - scipy_stats.f.cdf(f_stat, df1, df2))
        else:  # pragma: no cover - hard to force in test env with SciPy installed
            pvalue = math.nan
            warnings.warn(
                "SciPy not available: nested F-test p-value is NaN.",
                RuntimeWarning,
                stacklevel=2,
            )

    return {
        "SS_res_lin": float(linear_fit.ss_res),
        "SS_res_quad": float(quadratic_fit.ss_res),
        "F": f_stat,
        "df1": float(df1),
        "df2": float(df2),
        "pvalue": pvalue,
        "notes": notes,
    }


def _build_wls_weights(
    level_summary: pd.DataFrame,
    data: pd.DataFrame,
    epsilon: float = EPSILON,
) -> tuple[np.ndarray, str]:
    """Compute group-wise WLS weights as 1/var with epsilon safeguards."""
    safe_epsilon = float(max(epsilon, 1e-30))
    weights_by_conc: dict[float, float] = {}
    zero_var_levels: list[float] = []

    for _, row in level_summary.iterrows():
        conc = float(row["concentration"])
        var_y = float(row["var_y"])
        if (not np.isfinite(var_y)) or (var_y <= 0):
            weights_by_conc[conc] = 1.0 / safe_epsilon
            zero_var_levels.append(conc)
        else:
            weights_by_conc[conc] = 1.0 / var_y

    conc_to_weight = pd.Series(weights_by_conc)
    point_weights = (
        data["concentration"].map(conc_to_weight).to_numpy(dtype=float).reshape(-1)
    )
    if np.any(~np.isfinite(point_weights)) or np.any(point_weights <= 0):
        raise ValueError("Failed to assign finite positive weights to all rows.")

    notes = ""
    if zero_var_levels:
        levels = ", ".join(f"{lvl:.3f}M" for lvl in zero_var_levels)
        notes = f"Zero variance at {levels}; used epsilon={safe_epsilon:.1e}."
    return point_weights, notes


def _format_pvalue(value: float) -> str:
    """Format p-values consistently for plot annotations."""
    if not np.isfinite(value):
        return "NaN"
    if value < 1e-3:
        return "<0.001"
    return f"{value:.3f}"


def _predict(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate polynomial coefficients over x."""
    return np.vander(np.asarray(x, dtype=float), N=len(beta), increasing=True) @ beta


def create_plots(
    outdir: Path,
    long_df: pd.DataFrame,
    level_summary: pd.DataFrame,
    point_residuals: pd.DataFrame,
    linear_fit: ModelFit,
    quadratic_fit: ModelFit,
    weighted_fit: ModelFit | None,
    brown_forsythe: dict[str, Any],
    lack_of_fit: dict[str, Any],
    lin_vs_quad: dict[str, Any],
) -> None:
    """Generate required publication-ready PNG diagnostics at 300 dpi."""
    _ensure_plot_style()

    x_grid = np.linspace(
        float(np.min(long_df["concentration"])),
        float(np.max(long_df["concentration"])),
        400,
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.scatter(
        long_df["concentration"],
        long_df["y"],
        s=PLOT_MARKER_SIZE,
        alpha=PLOT_ALPHA,
        color="#2C4B7D",
        label="Replicates",
        zorder=2,
    )
    ax.errorbar(
        level_summary["concentration"],
        level_summary["mean_y"],
        yerr=level_summary["sd_y"],
        fmt="o",
        color="black",
        markerfacecolor="white",
        markeredgecolor="black",
        capsize=3,
        label=r"Mean $\pm$ SD",
        zorder=4,
    )

    ax.plot(
        x_grid,
        _predict(linear_fit.beta, x_grid),
        color="#C13B2A",
        linewidth=PLOT_LINEWIDTH,
        label="Linear (unweighted)",
        zorder=3,
    )
    ax.plot(
        x_grid,
        _predict(quadratic_fit.beta, x_grid),
        color="#197A40",
        linewidth=PLOT_LINEWIDTH,
        linestyle="--",
        label="Quadratic (unweighted)",
        zorder=3,
    )
    if weighted_fit is not None:
        ax.plot(
            x_grid,
            _predict(weighted_fit.beta, x_grid),
            color="#7A3B9E",
            linewidth=PLOT_LINEWIDTH,
            linestyle=":",
            label="Linear (weighted)",
            zorder=3,
        )

    format_axes(
        ax=ax,
        xlabel=LABEL_NACL_M,
        ylabel=LABEL_PKA_APP_LONG,
        concentration_axis=True,
        grid_axis="both",
    )
    ax.margins(y=0.10)
    legend = ax.legend(loc="upper right", frameon=False)
    add_stats_box(
        ax=ax,
        lines=[
            f"BF p = {_format_pvalue(float(brown_forsythe['pvalue']))}",
            f"LOF p = {_format_pvalue(float(lack_of_fit['pvalue']))}",
            f"Lin vs Quad p = {_format_pvalue(float(lin_vs_quad['pvalue']))}",
        ],
        loc="upper left",
        avoid_legend=True,
        legend=legend,
        x_data=long_df["concentration"].to_numpy(dtype=float),
        y_data=long_df["y"].to_numpy(dtype=float),
    )
    fig.savefig(outdir / "regression_with_errorbars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.scatter(
        point_residuals["yhat_linear"],
        point_residuals["resid_linear"],
        s=PLOT_MARKER_SIZE,
        color="#2C4B7D",
        alpha=PLOT_ALPHA,
    )
    ax.axhline(0.0, color="black", linewidth=PLOT_ZERO_LINEWIDTH, linestyle="--")
    format_axes(
        ax=ax,
        xlabel=LABEL_FITTED_LINEAR,
        ylabel=LABEL_RESID_LINEAR,
        concentration_axis=False,
        grid_axis="both",
    )
    ax.margins(y=0.08)
    fig.savefig(outdir / "residuals_vs_fitted_linear.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.scatter(
        point_residuals["concentration"],
        point_residuals["resid_linear"],
        s=PLOT_MARKER_SIZE,
        color="#2C4B7D",
        alpha=PLOT_ALPHA,
    )
    ax.axhline(0.0, color="black", linewidth=PLOT_ZERO_LINEWIDTH, linestyle="--")
    format_axes(
        ax=ax,
        xlabel=LABEL_NACL_M,
        ylabel=LABEL_RESID_LINEAR,
        concentration_axis=True,
        grid_axis="both",
    )
    ax.margins(y=0.08)
    fig.savefig(
        outdir / "residuals_vs_concentration_linear.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    ax.plot(
        level_summary["concentration"],
        level_summary["sd_y"],
        marker="o",
        color="#C13B2A",
        markersize=5.5,
        linewidth=PLOT_LINEWIDTH,
        alpha=0.95,
    )
    format_axes(
        ax=ax,
        xlabel=LABEL_NACL_M,
        ylabel=LABEL_SD_PKA_APP,
        concentration_axis=True,
        grid_axis="both",
    )
    ax.margins(y=0.10)
    fig.savefig(
        outdir / "sd_or_var_vs_concentration.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    levels = np.array(EXPECTED_CONCENTRATIONS, dtype=float)
    grouped = [
        long_df.loc[
            np.isclose(long_df["concentration"], lvl, atol=CONCENTRATION_TOL, rtol=0.0),
            "y",
        ].to_numpy(dtype=float)
        for lvl in levels
    ]
    ax.boxplot(
        grouped,
        positions=levels,
        widths=BOXPLOT_WIDTH,
        patch_artist=True,
        boxprops={
            "facecolor": "#B7D1F2",
            "edgecolor": "black",
            "linewidth": 1.0,
            "alpha": 0.9,
        },
        medianprops={"color": "black", "linewidth": PLOT_LINEWIDTH},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )
    format_axes(
        ax=ax,
        xlabel=LABEL_NACL_M,
        ylabel=LABEL_PKA_APP_LONG,
        concentration_axis=True,
        grid_axis="both",
    )
    ax.set_xticks(levels)
    ax.set_xticklabels([f"{value:.1f}" for value in levels])
    ax.set_xlim(float(levels[0] - X_LEVEL_PAD), float(levels[-1] + X_LEVEL_PAD))
    ax.margins(y=0.08)
    fig.savefig(outdir / "group_spread.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_error_diagnosis_pipeline(
    input_path: str | Path,
    outdir: str | Path = DEFAULT_OUTPUT_DIR,
    pka_lit: float = 4.76,
    baseline_concentration: float = 0.0,
    concentration_col: str | None = None,
    replicate_col: str | None = None,
    y_col: str | None = None,
    epsilon: float = EPSILON,
) -> dict[str, Any]:
    """Run the full IA error-diagnosis analysis pipeline."""
    input_file = Path(input_path)
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(input_file)
    long_df = standardize_long_form(
        raw_df=raw_df,
        concentration_col=concentration_col,
        replicate_col=replicate_col,
        y_col=y_col,
    )
    long_df = validate_long_form(long_df)

    level_summary = compute_level_summary(long_df)
    pure_error = compute_pure_error(long_df)
    brown_forsythe = run_brown_forsythe(long_df)
    baseline_test = baseline_one_sample_ttest(
        long_df,
        pka_lit=float(pka_lit),
        baseline_concentration=float(baseline_concentration),
    )

    x = long_df["concentration"].to_numpy(dtype=float)
    y = long_df["y"].to_numpy(dtype=float)

    linear_fit = fit_polynomial_model(
        x=x,
        y=y,
        degree=1,
        model_name="linear_unweighted",
    )
    quadratic_fit = fit_polynomial_model(
        x=x,
        y=y,
        degree=2,
        model_name="quadratic_unweighted",
    )
    lack_of_fit = compute_lack_of_fit(linear_fit=linear_fit, pure_error=pure_error)
    lin_vs_quad = linear_vs_quadratic_f_test(
        linear_fit=linear_fit,
        quadratic_fit=quadratic_fit,
        n_obs=len(long_df),
    )

    heteroscedastic = bool(
        np.isfinite(brown_forsythe["pvalue"]) and (brown_forsythe["pvalue"] < 0.05)
    )
    weighted_fit: ModelFit | None = None
    weights = np.full(shape=len(long_df), fill_value=np.nan, dtype=float)
    wls_notes = ""
    if heteroscedastic:
        weights, wls_notes = _build_wls_weights(
            level_summary=level_summary,
            data=long_df,
            epsilon=epsilon,
        )
        weighted_fit = fit_polynomial_model(
            x=x,
            y=y,
            degree=1,
            model_name="linear_weighted",
            weights=weights,
            notes=wls_notes,
        )

    point_residuals = long_df.copy()
    point_residuals["yhat_linear"] = linear_fit.yhat
    point_residuals["resid_linear"] = linear_fit.resid
    point_residuals["yhat_quad"] = quadratic_fit.yhat
    point_residuals["resid_quad"] = quadratic_fit.resid
    if weighted_fit is not None:
        point_residuals["yhat_wls"] = weighted_fit.yhat
        point_residuals["resid_wls"] = weighted_fit.resid
        point_residuals["weight"] = weights
    else:
        point_residuals["yhat_wls"] = math.nan
        point_residuals["resid_wls"] = math.nan
        point_residuals["weight"] = math.nan

    level_summary.to_csv(output_dir / "level_summary.csv", index=False)
    pd.DataFrame([pure_error]).to_csv(output_dir / "pure_error.csv", index=False)

    tests_rows = [
        {
            "test": "levene_brown_forsythe",
            "stat": brown_forsythe["stat"],
            "pvalue": brown_forsythe["pvalue"],
            "method": brown_forsythe["method"],
            "notes": brown_forsythe.get("notes", ""),
        },
        {
            "test": "baseline_one_sample_ttest",
            "pKa_lit": baseline_test["pKa_lit"],
            "t_stat": baseline_test["t_stat"],
            "df": baseline_test["df"],
            "pvalue": baseline_test["pvalue"],
            "mean0": baseline_test["mean0"],
            "sd0": baseline_test["sd0"],
            "n0": baseline_test["n0"],
            "ci95_low": baseline_test["ci95_low"],
            "ci95_high": baseline_test["ci95_high"],
            "notes": baseline_test.get("notes", ""),
        },
        {
            "test": "lack_of_fit_linear",
            "SS_res": lack_of_fit["SS_res"],
            "df_res": lack_of_fit["df_res"],
            "SS_LOF": lack_of_fit["SS_LOF"],
            "df_LOF": lack_of_fit["df_LOF"],
            "MS_pure": lack_of_fit["MS_pure"],
            "MS_LOF": lack_of_fit["MS_LOF"],
            "F": lack_of_fit["F"],
            "pvalue": lack_of_fit["pvalue"],
            "notes": lack_of_fit.get("notes", ""),
        },
        {
            "test": "lin_vs_quad",
            "SS_res_lin": lin_vs_quad["SS_res_lin"],
            "SS_res_quad": lin_vs_quad["SS_res_quad"],
            "F": lin_vs_quad["F"],
            "df1": lin_vs_quad["df1"],
            "df2": lin_vs_quad["df2"],
            "pvalue": lin_vs_quad["pvalue"],
            "notes": lin_vs_quad.get("notes", ""),
        },
    ]
    pd.DataFrame(tests_rows).to_csv(output_dir / "tests.csv", index=False)

    model_rows = [
        {
            "model": linear_fit.model,
            "beta0": linear_fit.beta0,
            "beta1": linear_fit.beta1,
            "beta2": math.nan,
            "SS_res": linear_fit.ss_res,
            "df_res": linear_fit.df_res,
            "R2": linear_fit.r2,
            "notes": linear_fit.notes,
        },
        {
            "model": quadratic_fit.model,
            "beta0": quadratic_fit.beta0,
            "beta1": quadratic_fit.beta1,
            "beta2": quadratic_fit.beta2,
            "SS_res": quadratic_fit.ss_res,
            "df_res": quadratic_fit.df_res,
            "R2": quadratic_fit.r2,
            "notes": quadratic_fit.notes,
        },
    ]
    if weighted_fit is not None:
        model_rows.append(
            {
                "model": weighted_fit.model,
                "beta0": weighted_fit.beta0,
                "beta1": weighted_fit.beta1,
                "beta2": math.nan,
                "SS_res": weighted_fit.ss_res,
                "df_res": weighted_fit.df_res,
                "R2": weighted_fit.r2,
                "notes": weighted_fit.notes,
            }
        )
    pd.DataFrame(model_rows).to_csv(output_dir / "model_fits.csv", index=False)

    point_residuals.to_csv(output_dir / "point_residuals.csv", index=False)

    create_plots(
        outdir=output_dir,
        long_df=long_df,
        level_summary=level_summary,
        point_residuals=point_residuals,
        linear_fit=linear_fit,
        quadratic_fit=quadratic_fit,
        weighted_fit=weighted_fit,
        brown_forsythe=brown_forsythe,
        lack_of_fit=lack_of_fit,
        lin_vs_quad=lin_vs_quad,
    )

    return {
        "level_summary": level_summary,
        "pure_error": pure_error,
        "brown_forsythe": brown_forsythe,
        "baseline_ttest": baseline_test,
        "lack_of_fit": lack_of_fit,
        "lin_vs_quad": lin_vs_quad,
        "linear_fit": linear_fit,
        "quadratic_fit": quadratic_fit,
        "weighted_fit": weighted_fit,
        "point_residuals": point_residuals,
        "outdir": output_dir,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for module execution."""
    parser = argparse.ArgumentParser(
        description="IA error diagnosis pipeline for pH-at-half-equivalence repeats."
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--pka-lit",
        type=float,
        default=4.76,
        help="Literature pKa value for baseline one-sample t-test.",
    )
    parser.add_argument(
        "--baseline-concentration",
        type=float,
        default=0.0,
        help="Baseline concentration for one-sample t-test (default: 0.0 M).",
    )
    parser.add_argument(
        "--concentration-col",
        default=None,
        help="Optional explicit concentration column name in input CSV.",
    )
    parser.add_argument(
        "--replicate-col",
        default=None,
        help="Optional explicit replicate column name in input CSV.",
    )
    parser.add_argument(
        "--y-col",
        default=None,
        help="Optional explicit response y column name in input CSV.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=EPSILON,
        help="Epsilon used when group variance is zero for WLS weights.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for running the error diagnosis pipeline."""
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    run_error_diagnosis_pipeline(
        input_path=args.input,
        outdir=args.outdir,
        pka_lit=args.pka_lit,
        baseline_concentration=args.baseline_concentration,
        concentration_col=args.concentration_col,
        replicate_col=args.replicate_col,
        y_col=args.y_col,
        epsilon=args.epsilon,
    )
    print(f"Wrote IA error diagnosis outputs to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
