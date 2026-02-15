# Data Pipeline and Provenance

## Pipeline summary

```text
data/raw/*.csv
  -> file screening and concentration parsing
  -> header normalization
  -> standardized copies in data/_standardized_raw/
  -> run extraction
  -> step aggregation
  -> run-level analysis
  -> output/individual_results.csv
  -> output/statistical_summary.csv
  -> output/provenance_map.csv
  -> output/qc and output/iterations figure bundles
```

## Raw versus standardized data

- `data/raw/` stores source exports.
- `data/_standardized_raw/` stores normalized copies used for processing reproducibility.
- Standardization includes header normalization and run-label harmonization.

## Run extraction conventions

Per-run extracted tables are expected to include:
- `Volume (cm^3)`
- `pH`

Optional channels:
- `Time (min)`
- `Temperature (°C)`

## Step-level outputs

Step aggregation produces:
- `pH_step`
- `pH_step_sd`
- `pH_drift_step`
- `pH_slope_step`
- `n_step`
- `n_tail`

## Run-level and summary outputs

Run-level table:
- `output/individual_results.csv`

Summary table:
- `output/statistical_summary.csv`

Provenance mapping:
- `output/provenance_map.csv`

## Provenance interpretation

`output/provenance_map.csv` links each processed run label to its raw source filename and NaCl concentration label.

Use this file to trace every reported value and plotted point back to an original source file.

## Reproduction checklist

- Keep raw input files unchanged.
- Use repository root as working directory.
- Install dependencies from `requirements.txt`.
- Run `python3 main.py`.
- Use output iteration folders to compare all-valid, qc-pass, and strict-fit subsets.
