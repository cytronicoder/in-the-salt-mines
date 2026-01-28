### Ionic Strength Effects on Weak Acid Dissociation

This is a Python package for analyzing weak acid-strong base titration data under varying ionic strength conditions. We extract apparent dissociation constants using a rigorous two-stage protocol with explicit treatment of activity coefficient effects.

#### Scientific Background

In aqueous solution, weak acid dissociation follows:

$\ce{HA <=> H+ + A-}$

The thermodynamic dissociation constant relates activities, not concentrations:

$K_a = \dfrac{\ce{a_{H+} a_{A-}}}{\ce{a_{HA}}} = \dfrac{\gamma_{H+}\,[\ce{H+}]\;\gamma_{A-}\,[\ce{A-}]}{\gamma_{HA}\,[\ce{HA}]}$

When we measure pH and concentrations, we obtain an apparent dissociation constant:

$K_{a,\mathrm{app}} = \dfrac{[\ce{H+}][\ce{A-}]}{[\ce{HA}]} = K_a \dfrac{\gamma_{HA}}{\gamma_{H+}\gamma_{A-}}$

Since activity coefficients (γ) depend on ionic strength, **pK\_{a,\mathrm{app}} varies with NaCl concentration**. Therefore, we ask how increasing ionic strength (via NaCl addition) can affect the measured apparent pKa of a weak acid (ethanoic acid).

For the purpose of this analysis, we apply the following two-stage protocol to extract $pK_{a,\mathrm{app}}$ from titration data:

1. We first obtain a coarse estimate of pK\_{a,\mathrm{app}}. We locate the half-equivalence point (V = V*eq/2) where, by Henderson–Hasselbalch approximation, $\mathrm{pH} \approx \mathrm{p}K*{a,\mathrm{app}}$. This provides $\mathrm{p}K_{a,\mathrm{app},\mathrm{initial}}$.
2. We then perform a refined Henderson–Hasselbalch regression within the **buffer region** ($\lvert \mathrm{pH} - \mathrm{p}K_{a,\mathrm{app}} \rvert \le 1$):

   $\displaystyle \mathrm{pH} = m\,\log_{10}\!\left(\frac{V}{V_{\mathrm{eq}} - V}\right) + \mathrm{p}K_{a,\mathrm{app}}$

   The intercept gives the refined pK\_{a,\mathrm{app}}. The slope $m$ should be $\approx 1.0$ for an ideal buffer; significant deviations trigger warnings.

Note that the Henderson–Hasselbalch equation is valid only where $0.1 \le \dfrac{[\ce{A-}]}{[\ce{HA}]} \le 10$, corresponding to $\lvert \mathrm{pH} - \mathrm{p}K_{a,\mathrm{app}} \rvert \le 1$. Restricting regression to this region ensures chemically meaningful results.

#### Uncertainty Treatment

All uncertainties are systematic (worst-case bounds), consistent with IB Chemistry Data Processing methodology:

- Equipment uncertainties: Burette (±0.05 cm³), pH meter, balance
- Propagation: Worst-case addition for sums/differences
- Trial variation: Half-range (max − min)/2 for repeated measurements

#### Installation

To install and run this package, ensure you have:

- Python 3.8 or higher
- NumPy, pandas, matplotlib
- SciPy (optional, enables PCHIP interpolation)

To install from the source repository, run:

```bash
git clone https://github.com/cytronicoder/in-the-salt-mines.git
cd in-the-salt-mines
pip install -r requirements.txt
```

Then, install development dependencies and set up pre-commit hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

#### Usage

To run the complete analysis pipeline, execute in the terminal:

```bash
python main.py
```

This will:

- Process all CSV files in `data/`
- Generate three-panel titration figures (curve, derivative, HH plot)
- Compute mean pKa_app per NaCl concentration with uncertainties
- Save results to `output/` directory

#### API

```python
from salty.analysis import (
    analyze_titration,
    calculate_statistics,
    create_results_dataframe,
    process_all_files,
    build_summary_plot_data,
)
from salty.plotting import plot_titration_curves, plot_statistical_summary
from salty.output import save_data_to_csv

### Define input files: (filepath, NaCl_concentration_in_M)
files = [
    ("data/ms besant go brr brr v2- 0.0m nacl.csv", 0.0),
    ("data/ms besant go brr brr v2- 0.2m nacl.csv", 0.2),
    ("data/ms besant go brr brr v2- 0.4m nacl.csv", 0.4),
]

### Process all titration runs
results = process_all_files(files)

### Generate individual titration plots
plot_titration_curves(results, output_dir="output/with_raw", show_raw_pH=True)

### Compute statistics
results_df = create_results_dataframe(results)
stats_df = calculate_statistics(results_df)

### Generate summary plot
summary = build_summary_plot_data(stats_df, results_df)
plot_statistical_summary(summary, output_dir="output")

### Save to CSV
save_data_to_csv(results_df, stats_df, output_dir="output")
```

#### Package Architecture

```
salty/
├── __init__.py          # Package exports
├── analysis.py          # Main orchestration (two-stage protocol)
├── data_processing.py   # CSV parsing, run extraction, step aggregation
├── output.py            # CSV export utilities
├── schema.py            # DataFrame column definitions
├── uncertainty.py       # Re-exports from stats/
├── chemistry/           # No matplotlib dependencies
│   ├── hh_model.py      # Henderson-Hasselbalch regression
│   └── buffer_region.py # Buffer region selection
├── stats/               # No chemistry dependencies
│   ├── regression.py    # Linear regression utilities
│   └── uncertainty.py   # IB DP uncertainty propagation
└── plotting/            # No chemistry calculations
    ├── titration_plots.py  # Three-panel titration figures
    └── summary_plots.py    # pKa_app vs. concentration plots
```

- `chemistry/` modules contain no matplotlib imports
- `plotting/` modules accept precomputed values only; no chemistry calculations
- `stats/` modules are pure numerical utilities

#### Testing

```bash
pytest tests/ -v
```

Tests verify:

- Chemical invariants (HH slope ≈ 1.0 for ideal buffer)
- Failure modes (proper exceptions on invalid inputs)
- Buffer region bounds enforcement
- Plotting input validation

#### Output Files

- `individual_results.csv`: Per-run pKa_app, V_eq, uncertainties, QC status
- `statistical_summary.csv`: Mean pKa_app per NaCl concentration
- `titration_*.png`: Three-panel figures for each run
- `statistical_summary.png`: $\mathrm{p}K_{a,\mathrm{app}}$ vs. $[\ce{NaCl}]$ with trend line

#### Interpretation Notes

The extracted pKa_app values are operational parameters, not thermodynamic constants. Observed trends reflect:

1. Changes in activity coefficients with ionic strength
2. Possibly ion-pairing effects at high [NaCl]
3. Combined effects that cannot be separated without additional measurements

All conclusions should compare pKa_app values across ionic strengths. Absolute pKa interpretation requires activity coefficient corrections (e.g., Debye-Hückel theory), which are beyond this package's scope.

#### License

MIT License — see [LICENSE](LICENSE) for details.
