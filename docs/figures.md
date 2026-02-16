# Figure Output Taxonomy

All generated figure artifacts are written under `output/figures/` using shared helpers in `salty/plotting/style.py`.

## Folder Structure

- `output/figures/summary/`
- `output/figures/methods_or_derivations/`
- `output/figures/qc/`
- `output/figures/diagnostics/`
- `output/figures/titration/`

## Shared Routing API

Use:

- `figure_path(fig_key, kind, ext, iteration=None)`
- `figure_base_path(fig_key, kind, iteration=None)`
- `save_figure_all_formats(fig, path_base)`

Supported `kind` values:

- `main_results`
- `methods`
- `qc`
- `diagnostics`
- `individual`
- `supplemental`

## Mapping: fig_key → folder

### `summary`
- `pka_app_vs_nacl_and_I`
- `pka_precision_by_nacl`
- `statistical_summary`
- `temperature_control_by_nacl`
- `initial_ph_by_nacl`
- `initial_ph_scatter_with_errorbar`
- `*_caption.txt` IA caption exports

### `methods_or_derivations`
- `titration_overlays_by_nacl`
- `derivative_equivalence_by_nacl`
- `equivalence_volumes_by_nacl`
- `half_equivalence_verification`
- `buffer_region_coverage`

### `qc`
- `temperature_and_calibration_qc`

### `diagnostics`
- `hh_linearization_and_diagnostics`
- `hh_slope_and_r2_diagnostics`
- `hh_residuals_analysis`
- `diagnostic_residuals_vs_iv`
- `diagnostic_residual_histogram`
- `diagnostic_residual_qq`
- `diagnostic_replicate_jitter`
- `diagnostic_sd_vs_iv`
- `diagnostic_parity_measured_vs_predicted`
- `diagnostic_cooks_distance`

### `titration`
- `titration_curve__{condition}__run-{k}__{source}`

## Docs Image Mirror

Markdown pages in `docs/` embed PNG files from `docs/images/`. Refresh that mirror after regenerating figures:

```bash
python3 main.py --figures ia
cp output/figures/{summary,methods_or_derivations,diagnostics,qc}/*.png docs/images/
```

## Legacy compatibility

- CSV/tabular outputs remain under `output/` and `output/ia` for compatibility.
- Figure generation defaults now target `output/figures/`.
- Plot functions still accept explicit `output_dir` values to support legacy callers.
