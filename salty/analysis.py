"""
Analyzing titration data and finding equivalence points.
"""

import importlib.util
import numpy as np
import pandas as pd
import os

from .data_processing import (
    aggregate_volume_steps,
    calculate_derivatives,
    extract_runs,
    load_titration_data,
)

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None
if HAVE_SCIPY:
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import curve_fit
    from scipy.stats import t as student_t


def build_ph_interpolator(step_df, method="linear"):
    """Build a pH(V) interpolator using step-level medians."""

    if step_df.empty:
        return {"method": "linear", "func": lambda x: np.full_like(x, np.nan)}

    x = step_df["Volume (cm³)"].to_numpy()
    y = step_df["pH_step"].to_numpy()
    x_min = float(np.min(x))
    x_max = float(np.max(x))

    if method == "pchip" and HAVE_SCIPY:
        interpolator = PchipInterpolator(x, y, extrapolate=False)

        def interp_func(xq):
            xq_arr = np.atleast_1d(xq)
            yq = interpolator(xq_arr)
            return yq[0] if np.ndim(xq) == 0 else yq

        return {"method": "pchip", "func": interp_func, "x_min": x_min, "x_max": x_max}

    def interp_func(xq):
        xq_arr = np.atleast_1d(xq)
        yq = np.interp(xq_arr, x, y)
        mask = (xq_arr < x_min) | (xq_arr > x_max)
        yq[mask] = np.nan
        return yq[0] if np.ndim(xq) == 0 else yq

    return {"method": "linear", "func": interp_func, "x_min": x_min, "x_max": x_max}


def generate_dense_curve(interpolator, n_points=400):
    """Generate a dense pH(V) curve for plotting and derivatives."""

    if "x_min" not in interpolator or "x_max" not in interpolator:
        return pd.DataFrame(columns=["Volume (cm³)", "pH_interp"])
    x_dense = np.linspace(interpolator["x_min"], interpolator["x_max"], n_points)
    y_dense = interpolator["func"](x_dense)
    return pd.DataFrame({"Volume (cm³)": x_dense, "pH_interp": y_dense})


def detect_equivalence_point(step_df, edge_buffer=2, min_post_points=3, delta_pH=1.0):
    """Locate equivalence point with QC guardrails."""

    if step_df.empty or "dpH/dx" not in step_df.columns:
        return {"eq_x": np.nan, "eq_pH": np.nan, "qc_pass": False, "qc_reason": "Missing derivative"}

    idx = step_df["dpH/dx"].idxmax()
    eq_x = float(step_df.loc[idx, "Volume (cm³)"])
    eq_pH = float(step_df.loc[idx, "pH_smooth"])
    reasons = []

    if idx <= edge_buffer or idx >= len(step_df) - 1 - edge_buffer:
        reasons.append("Peak too close to data edge")

    if len(step_df) - idx - 1 < min_post_points:
        reasons.append("Insufficient post-equivalence coverage")

    before_idx = max(idx - 2, 0)
    after_idx = min(idx + 2, len(step_df) - 1)
    delta_region = step_df.loc[after_idx, "pH_step"] - step_df.loc[before_idx, "pH_step"]
    if delta_region < delta_pH:
        reasons.append("pH rise across steep region below threshold")

    qc_pass = len(reasons) == 0
    qc_reason = "OK" if qc_pass else "; ".join(reasons)

    return {"eq_x": eq_x, "eq_pH": eq_pH, "qc_pass": qc_pass, "qc_reason": qc_reason}


def _hh_model(volume, pka, veq):
    return pka + np.log10(volume / (veq - volume))


def _fit_hh_grid_search(volumes, ph_values, veq_candidates):
    best = {"sse": np.inf, "veq": np.nan, "pka": np.nan, "slope": np.nan}
    for veq in veq_candidates:
        mask = volumes < veq
        if np.sum(mask) < 3:
            continue
        v = volumes[mask]
        y = ph_values[mask]
        x = np.log10(v / (veq - v))
        if not np.all(np.isfinite(x)):
            continue
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = intercept + slope * x
        sse = np.sum((y - y_pred) ** 2)
        if sse < best["sse"]:
            best = {"sse": sse, "veq": veq, "pka": intercept, "slope": slope}
    return best


def fit_henderson_hasselbalch(step_df):
    """Rescue Veq/pKa via HH fit using pre-equivalence points only."""

    volumes = step_df["Volume (cm³)"].to_numpy()
    ph_values = step_df["pH_step"].to_numpy()
    max_v = np.max(volumes) if len(volumes) else np.nan
    if not np.isfinite(max_v) or len(volumes) < 5:
        return {
            "veq_fit": np.nan,
            "pka_fit": np.nan,
            "slope_fit": np.nan,
            "fit_quality": np.nan,
            "method": "grid",
        }

    lower = max_v * 0.6
    upper = max_v * 1.5
    candidates = np.linspace(lower, upper, 80)

    if HAVE_SCIPY:
        mask = volumes < max_v * 0.95
        v = volumes[mask]
        y = ph_values[mask]
        if len(v) >= 3 and np.all(v > 0):
            try:
                popt, _ = curve_fit(
                    _hh_model,
                    v,
                    y,
                    p0=[np.median(y), max_v],
                    bounds=([0.0, lower], [14.0, upper]),
                )
                pka_fit, veq_fit = popt
                x = np.log10(v / (veq_fit - v))
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = intercept + slope * x
                sse = np.sum((y - y_pred) ** 2)
                return {
                    "veq_fit": float(veq_fit),
                    "pka_fit": float(pka_fit),
                    "slope_fit": float(slope),
                    "fit_quality": float(sse),
                    "method": "curve_fit",
                }
            except Exception:
                pass

    best = _fit_hh_grid_search(volumes, ph_values, candidates)
    return {
        "veq_fit": float(best["veq"]),
        "pka_fit": float(best["pka"]),
        "slope_fit": float(best["slope"]),
        "fit_quality": float(best["sse"]),
        "method": "grid",
    }


def fit_buffer_region(step_df, veq):
    """Fit pH vs log10(V/(Veq-V)) in the buffer region to estimate pKa."""

    volumes = step_df["Volume (cm³)"].to_numpy()
    ph_values = step_df["pH_step"].to_numpy()
    mask = (volumes > 0.2 * veq) & (volumes < 0.8 * veq) & (volumes < veq)
    v = volumes[mask]
    y = ph_values[mask]
    if len(v) < 3:
        return {
            "pka_reg": np.nan,
            "slope_reg": np.nan,
            "r2": np.nan,
            "n_points": 0,
            "buffer_df": pd.DataFrame(columns=["Volume (cm³)", "log10_ratio", "pH_step"]),
        }

    x = np.log10(v / (veq - v))
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = intercept + slope * x
    sse = np.sum((y - y_pred) ** 2)
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - sse / sst if sst > 0 else np.nan

    buffer_df = pd.DataFrame(
        {"Volume (cm³)": v, "log10_ratio": x, "pH_step": y, "pH_fit": y_pred}
    )

    return {
        "pka_reg": float(intercept),
        "slope_reg": float(slope),
        "r2": float(r2),
        "n_points": int(len(v)),
        "buffer_df": buffer_df,
    }


def estimate_uncertainties(step_df, veq_primary, veq_secondary, pka_reg):
    """Estimate uncertainties using method spread and sensitivity checks."""

    volumes = step_df["Volume (cm³)"].to_numpy()
    delta_v = np.diff(volumes)
    median_step = np.median(delta_v) if len(delta_v) else np.nan

    if np.isfinite(veq_primary) and np.isfinite(veq_secondary):
        veq_unc = 0.5 * abs(veq_primary - veq_secondary)
    elif np.isfinite(median_step):
        veq_unc = 0.5 * median_step
    else:
        veq_unc = np.nan

    if (
        np.isfinite(pka_reg)
        and np.isfinite(veq_unc)
        and np.isfinite(veq_primary)
        and veq_primary >= 0.1
    ):
        pka_plus = fit_buffer_region(step_df, veq_primary + veq_unc)["pka_reg"]
        pka_minus = fit_buffer_region(step_df, max(veq_primary - veq_unc, 0.1))[
            "pka_reg"
        ]
        pka_unc = np.nanmax([abs(pka_plus - pka_reg), abs(pka_minus - pka_reg)])
    else:
        pka_unc = np.nan

    return veq_unc, pka_unc


def analyze_titration(df, run_name, x_col="Volume (cm³)"):
    """Analyze a single titration run."""

    step_df = aggregate_volume_steps(df)
    if step_df.empty:
        return {
            "run_name": run_name,
            "eq_x": np.nan,
            "eq_pH": np.nan,
            "eq_qc_pass": False,
            "eq_qc_reason": "No valid volume/pH data after aggregation",
            "veq_used": np.nan,
            "veq_method": None,
            "veq_uncertainty": np.nan,
            "half_eq_x": np.nan,
            "half_eq_pH": np.nan,
            "pka_reg": np.nan,
            "slope_reg": np.nan,
            "r2_reg": np.nan,
            "pka_half": np.nan,
            "pka_uncertainty": np.nan,
            "hh_fit": {},
            "x_col": x_col,
            "data": df,
            "step_data": step_df,
            "dense_curve": {},
            "buffer_region": pd.DataFrame(),
            "diagnostics": {"interpolator_method": None, "step_points": 0},
            "skip_reason": "No valid volume/pH data after aggregation",
        }

    step_df = calculate_derivatives(step_df, x_col="Volume (cm³)", ph_col="pH_step")
    interpolator = build_ph_interpolator(step_df, method="linear")
    dense_curve = generate_dense_curve(interpolator)

    eq_info = detect_equivalence_point(step_df)
    hh_fit = fit_henderson_hasselbalch(step_df)

    veq_deriv = eq_info["eq_x"] if eq_info["qc_pass"] else np.nan
    if np.isfinite(veq_deriv):
        veq_used, veq_method = veq_deriv, "derivative"
    elif np.isfinite(hh_fit.get("veq_fit", np.nan)):
        veq_used, veq_method = hh_fit["veq_fit"], "HH_fit"
    elif np.isfinite(eq_info.get("eq_x", np.nan)):
        veq_used, veq_method = eq_info["eq_x"], "derivative_weak"
    else:
        veq_used, veq_method = np.nan, "unknown"

    # ---- buffer-region pKa logic with rescue ----
    default_buffer_fit = {
        "pka_reg": np.nan,
        "slope_reg": np.nan,
        "r2": np.nan,
        "n_points": 0,
        "buffer_df": pd.DataFrame(),
    }

    if np.isfinite(veq_used):
        buffer_fit = default_buffer_fit.copy()
        buffer_fit.update(fit_buffer_region(step_df, veq_used))
    else:
        buffer_fit = default_buffer_fit.copy()

    if np.isfinite(veq_used):
        try:
            if buffer_fit["n_points"] < 3 or not np.isfinite(buffer_fit["pka_reg"]):
                for factor in (0.85, 0.9, 0.95, 1.05, 1.1):
                    bf = fit_buffer_region(step_df, veq_used * factor)
                    if bf["n_points"] >= 3 and np.isfinite(bf["pka_reg"]):
                        buffer_fit = bf
                        break

            half_eq_x = veq_used / 2
            half_eq_pH = float(interpolator["func"](half_eq_x))
            if buffer_fit["n_points"] < 3 and np.isfinite(half_eq_pH):
                buffer_fit = {
                    "pka_reg": half_eq_pH,
                    "slope_reg": np.nan,
                    "r2": np.nan,
                    "n_points": 1,
                    "buffer_df": pd.DataFrame(
                        {"Volume (cm³)": [half_eq_x], "pH_step": [half_eq_pH]}
                    ),
                }
        except Exception:
            pass

    pka_reg = buffer_fit["pka_reg"]
    if not np.isfinite(pka_reg) and np.isfinite(hh_fit.get("pka_fit", np.nan)):
        pka_reg = hh_fit["pka_fit"]

    half_eq_x = veq_used / 2 if np.isfinite(veq_used) else np.nan
    half_eq_pH = float(interpolator["func"](half_eq_x)) if np.isfinite(half_eq_x) else np.nan

    veq_unc, pka_unc = estimate_uncertainties(
        step_df, veq_used, hh_fit.get("veq_fit", np.nan), pka_reg
    )

    return {
        "run_name": run_name,
        "eq_x": eq_info["eq_x"],
        "eq_pH": eq_info["eq_pH"],
        "eq_qc_pass": eq_info["qc_pass"],
        "eq_qc_reason": eq_info["qc_reason"],
        "veq_used": veq_used,
        "veq_method": veq_method,
        "veq_uncertainty": veq_unc,
        "half_eq_x": half_eq_x,
        "half_eq_pH": half_eq_pH,
        "pka_reg": pka_reg,
        "slope_reg": buffer_fit.get("slope_reg", np.nan),
        "r2_reg": buffer_fit.get("r2", np.nan),
        "pka_half": half_eq_pH,
        "pka_uncertainty": pka_unc,
        "hh_fit": hh_fit,
        "x_col": x_col,
        "data": df,
        "step_data": step_df,
        "dense_curve": dense_curve,
        "buffer_region": buffer_fit.get("buffer_df", pd.DataFrame()),
        "diagnostics": {
            "interpolator_method": interpolator["method"],
            "step_points": len(step_df),
        },
    }


def process_all_files(file_list):
    """Process multiple titration data files."""
    results = []

    for filepath, nacl_conc in file_list:
        print(f"Processing {filepath} (NaCl: {nacl_conc} M)...")
        try:
            df_raw = load_titration_data(filepath)
            runs = extract_runs(df_raw)

            for run_name, run_info in runs.items():
                run_df = run_info["df"]
                x_col = run_info["x_col"]

                if "Volume (cm³)" not in run_df.columns or len(run_df) < 10:
                    continue

                analysis = analyze_titration(
                    run_df, f"{nacl_conc}M - {run_name}", x_col=x_col
                )
                if analysis.get("skip_reason"):
                    continue

                analysis["nacl_conc"] = nacl_conc
                analysis["source_file"] = os.path.basename(filepath)
                results.append(analysis)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    return results


def create_results_dataframe(results):
    """Create a tidy DataFrame of per-run results for downstream analysis/plotting."""
    rows = []
    for res in results:
        rows.append(
            {
                "Run": res.get("run_name"),
                "NaCl Concentration (M)": res.get("nacl_conc", np.nan),
                "pKa (buffer regression)": res.get("pka_reg", np.nan),
                "Equivalence QC Pass": bool(res.get("eq_qc_pass", False)),
                "Veq (used)": res.get("veq_used", np.nan),
                "Source File": res.get("source_file", ""),
            }
        )
    return pd.DataFrame(rows)


def calculate_statistics(results_df):
    """Compute group-level statistics (mean, SD, n) for pKa by NaCl concentration."""
    if results_df.empty:
        return pd.DataFrame(
            columns=["NaCl Concentration (M)", "Mean pKa", "SD", "n"]
        )
    grp = results_df.groupby("NaCl Concentration (M)")["pKa (buffer regression)"]
    stats = grp.agg(["mean", "std", "count"]).reset_index()
    stats = stats.rename(columns={"mean": "Mean pKa", "std": "SD", "count": "n"})
    return stats.sort_values("NaCl Concentration (M)").reset_index(drop=True)


def print_statistics(stats_df, results_df):
    """Print a concise statistical summary to stdout."""
    print("\nStatistical summary by NaCl concentration:")
    if stats_df.empty:
        print("  (no data)")
        return
    for _, row in stats_df.iterrows():
        conc = row["NaCl Concentration (M)"]
        mean = row["Mean pKa"]
        sd = row["SD"]
        n = int(row["n"])
        sd_str = f"{sd:.3f}" if pd.notna(sd) else "N/A"
        print(f" - {conc} M: mean={mean:.3f} SD={sd_str} n={n}")
        subset = results_df[results_df["NaCl Concentration (M)"] == conc]
        for _, r in subset.iterrows():
            pka = r.get("pKa (buffer regression)")
            qc = r.get("Equivalence QC Pass")
            print(f"     {r['Run']}: pKa={pka:.3f} (QC: {qc})")