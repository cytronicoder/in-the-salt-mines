# Ionic Strength Effects on Weak Acid Dissociation

This repository addresses a single scientific question: **how does ionic strength (via NaCl addition) shift the apparent dissociation constant of a weak acid measured by titration?** The package provides a rigorous, publication-ready workflow for extracting apparent pKa values (pKa\_app) from weak acid–strong base titration data, emphasizing chemical validity, explicit uncertainty handling, and reproducible results.

## Methodological Overview

### Apparent pKa Framework

Weak acid dissociation in aqueous solution is described by:

\[\ce{HA <=> H+ + A-}\]

Thermodynamic dissociation depends on activities, not concentrations:

\[K_a = \frac{a_{\ce{H+}}\,a_{\ce{A-}}}{a_{\ce{HA}}} = \frac{\gamma_{\ce{H+}}[\ce{H+}]\,\gamma_{\ce{A-}}[\ce{A-}]}{\gamma_{\ce{HA}}[\ce{HA}]}\]

Measured concentrations therefore yield an **apparent** dissociation constant:

\[K_{a,\mathrm{app}} = \frac{[\ce{H+}][\ce{A-}]}{[\ce{HA}]} = K_a\,\frac{\gamma_{\ce{HA}}}{\gamma_{\ce{H+}}\gamma_{\ce{A-}}}\]

Because activity coefficients \(\gamma\) depend on ionic strength, **pKa\_app values are operational and comparative**, not thermodynamic constants. All conclusions are therefore interpreted as trends across ionic strength conditions.

### Two-Stage pKa\_app Extraction

1. **Stage 1 (coarse estimate):**
   - Identify the equivalence point \(V_{\mathrm{eq}}\) from the maximum \(\mathrm{d}pH/\mathrm{d}V\).
   - Estimate \(pK_{a,\mathrm{app}}\) as the interpolated pH at \(V_{\mathrm{eq}}/2\).

2. **Stage 2 (refined regression):**
   - Define the chemically valid buffer region as \(|pH - pK_{a,\mathrm{app}}| \le 1\), corresponding to \(0.1 \le [\ce{A-}]/[\ce{HA}] \le 10\).
   - Fit the Henderson–Hasselbalch model within this region:

   \[pH = m\,\log_{10}\!\left(\frac{V}{V_{\mathrm{eq}} - V}\right) + pK_{a,\mathrm{app}}\]

   The intercept yields the refined apparent pKa\_app. Slope deviations from unity are reported as diagnostics.

### Strict Failure-on-Invalid-Science

The pipeline is deliberately strict. Missing volume axes, insufficient buffer-region points, or non-physical equivalence estimates raise explicit exceptions rather than producing ambiguous outputs. No legacy or backward-compatible pathways are retained.

## Assumptions and Limitations

- **Apparent pKa only:** pKa\_app values include ionic strength effects via activity coefficients and cannot be interpreted as thermodynamic pKa values without external corrections.
- **Henderson–Hasselbalch as operational model:** The model is applied only within the buffer region and is not assumed valid outside this window.
- **Comparative conclusions:** Trends across NaCl concentrations are meaningful; absolute acid strength is not claimed.

## Reproducibility and Uncertainty Philosophy

Uncertainty treatment follows systematic, worst-case propagation rules (IB Chemistry methodology):

- Equipment limitations are propagated explicitly.
- Trial-to-trial variability is reported as half-range (max − min)/2.
- Reported uncertainties represent bounds, **not statistical standard deviations**.

This conservative approach prioritizes reproducibility and interpretability in experimental contexts.

## Installation

```bash
git clone https://github.com/cytronicoder/in-the-salt-mines.git
cd in-the-salt-mines
pip install -r requirements.txt
```

Development tools:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

## Usage

```python
from salty.analysis import (
    analyze_titration,
    build_summary_plot_data,
    calculate_statistics,
    create_results_dataframe,
    process_all_files,
)
from salty.output import save_data_to_csv
from salty.plotting import plot_statistical_summary, plot_titration_curves

files = [
    ("data/ms besant go brr brr v2- 0.0m nacl.csv", 0.0),
    ("data/ms besant go brr brr v2- 0.2m nacl.csv", 0.2),
    ("data/ms besant go brr brr v2- 0.4m nacl.csv", 0.4),
]

results = process_all_files(files)
results_df = create_results_dataframe(results)
stats_df = calculate_statistics(results_df)

summary = build_summary_plot_data(stats_df, results_df)
plot_titration_curves(results, output_dir="output/with_raw", show_raw_pH=True)
plot_statistical_summary(summary, output_dir="output")

save_data_to_csv(results_df, stats_df, output_dir="output")
```

## Package Structure

```
salty/
├── analysis.py          # Two-stage orchestration and summary preparation
├── data_processing.py   # CSV parsing, run extraction, step aggregation
├── chemistry/           # Henderson–Hasselbalch regression and buffer selection
├── stats/               # Regression and systematic uncertainty propagation
├── plotting/            # Figures rendered from validated outputs
└── output.py            # CSV export utilities
```

- **chemistry**: chemically meaningful computations only; no I/O or plotting.
- **stats**: numerical methods only; no implicit DataFrame assumptions.
- **plotting**: visualization only; no chemistry or regression logic.

## Testing

```bash
pytest tests/ -v
```

## Outputs

- `individual_results.csv`: per-run pKa\_app values, uncertainties, V\_eq diagnostics
- `statistical_summary.csv`: mean pKa\_app per ionic strength with systematic uncertainty
- `titration_*.png`: three-panel per-run figures
- `statistical_summary.png`: pKa\_app vs. NaCl concentration trend plot

## License

MIT License — see [LICENSE](LICENSE) for details.
