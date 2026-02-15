### Ionic Strength Effects on Ethanoic Acid Titration Behavior

**Research question:** How does sodium chloride $\ce{NaCl}$ concentration change the apparent acid dissociation behavior of ethanoic acid $\ce{CH3COOH}$ during titration with sodium hydroxide $\ce{NaOH}$?

Experimental design summary:

- Weak acid: ethanoic acid solution
- Titrant: sodium hydroxide
- Independent variable: sodium chloride concentration in prepared acid solutions
- Main dependent variables: apparent pKa, equivalence volume, and model-fit diagnostics
- Controlled conditions in analysis and QC: temperature band, initial pH consistency, and buffer-region model validity checks

#### Reproduce the analysis

Inputs:

- Raw Logger Pro style CSV files in `data/raw/`.
- Standardized copies written by the pipeline to `data/_standardized_raw/`.

Environment and run commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 main.py
```

Primary outputs:

- `output/individual_results.csv`
- `output/statistical_summary.csv`
- `output/provenance_map.csv`
- `output/iterations/all_valid/`
- `output/iterations/qc_pass/`
- `output/iterations/strict_fit/`

Reproducibility notes:

- Plot jitter and subsampling use fixed seeds in QC plotting utilities.
- Iteration subset definitions are fixed in `main.py`.
- Provenance linking is written to `output/provenance_map.csv`.

#### Figures

![H-H slope and R2 diagnostics](docs/images/hh_slope_and_r2_diagnostics.png)
Caption: Validates that Henderson-Hasselbalch slope values remain chemically plausible and that fit quality is high enough for interpretation.

![H-H residuals analysis](docs/images/hh_residuals_analysis.png)
Caption: Validates that residuals are pattern-light and centered, supporting linear-model use in the selected buffer region.

![Apparent pKa precision by NaCl](docs/images/pka_precision_by_nacl.png)
Caption: Validates repeatability of apparent pKa within each sodium chloride condition.

![Equivalence volumes by NaCl](docs/images/equivalence_volumes_by_nacl.png)
Caption: Validates consistency and stoichiometric plausibility of detected equivalence volumes across conditions.

![Temperature control by NaCl](docs/images/temperature_control_by_nacl.png)
Caption: Validates that run temperatures remain near the control band so temperature drift is unlikely to dominate pKa trends.

![Buffer-region coverage](docs/images/buffer_region_coverage.png)
Caption: Validates that enough buffer-region points are available to support stable regression estimates.

#### License

This repository is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
