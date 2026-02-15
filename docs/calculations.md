# Calculations and Uncertainty Protocol

## Scope

This file defines the full analysis pipeline from imported raw CSV data to final apparent pKa reporting, including uncertainty treatment and interpretation limits.

## Assumptions

- The research-design narrative is treated as the method authority.
- Instrument defaults currently encoded in software are used unless replaced with confirmed instrument specifications.
- All reported pKa values are apparent values under the measured ionic-strength conditions.

## End-to-end pipeline

### 1) Data ingestion and standardization

- Raw input files are read from `data/raw/`.
- Headers are normalized and standardized copies are written to `data/_standardized_raw/`.
- Run extraction returns per-run tables with volume and pH pairs.

### 2) Step aggregation

- Repeated pH readings at each delivered volume are reduced to step summaries.
- Step summaries include equilibrium-focused pH estimates and per-step spread diagnostics.

### 3) Equivalence volume definition and detection

Method definition from IA protocol:

- Compute first derivative of pH with respect to delivered volume.
- Identify the largest derivative region.
- Define equivalence volume from that endpoint region.

Derivative criterion:

$$
V_{\mathrm{eq}}=\arg\max_V\left(\frac{d\mathrm{pH}}{dV}\right)
$$

IA narrative formulation for interval-based reading:

- If equivalence is read between two burette marks, use the midpoint.
- Set uncertainty as half of interval width.

Interval formulation:

$$
V_{\mathrm{eq}}=\frac{V_{\mathrm{left}}+V_{\mathrm{right}}}{2}
$$

$$
\Delta V_{\mathrm{eq,interval}}=\frac{V_{\mathrm{right}}-V_{\mathrm{left}}}{2}
$$

Software implementation note:

- Current code reports the derivative-peak location from step data or dense fallback interpolation.
- Current code uncertainty combines half median step spacing and burette delivered-volume term.

### 4) Half-equivalence definition and interpolation

Half-equivalence volume:

$$
V_{\tfrac{1}{2}\mathrm{eq}}=\frac{V_{\mathrm{eq}}}{2}
$$

Half-equivalence pH:

- Interpolate pH at half-equivalence from the run interpolation model.

Stage-1 apparent pKa estimate:

$$
\mathrm{pH}_{\tfrac{1}{2}\mathrm{eq}}\approx \mathrm{p}K_{a,\mathrm{app}}
$$

### 5) Buffer-region ratio and transform

For pre-equivalence weak-acid titration points:

$$
\frac{[\ce{A^-}]}{[\ce{HA}]}=\frac{V}{V_{\mathrm{eq}}-V}
$$

Transformed predictor used for regression:

$$
x=\log_{10}\left(\frac{V}{V_{\mathrm{eq}}-V}\right)
$$

### 6) Apparent pKa extraction methods

#### Method A: single-point half-equivalence estimate

$$
\mathrm{p}K_{a,\mathrm{app}}^{(\mathrm{half})}=\mathrm{pH}_{\tfrac{1}{2}\mathrm{eq}}
$$

#### Method B: buffer-region linear regression estimate

Regression model:

$$
\mathrm{pH}=m\,x+b
$$

Intercept meaning:

$$
b=\mathrm{p}K_{a,\mathrm{app}}^{(\mathrm{reg})}
$$

Goodness-of-fit statistic:

$$
R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}{\sum_i(y_i-\bar{y})^2}
$$

Inclusion window for regression:

$$
\left|\mathrm{pH}-\mathrm{p}K_{a,\mathrm{app}}\right|\le 1
$$

Weighting choice:

- Core run-level regression is unweighted ordinary least squares.
- Weighted comparisons are produced as supplemental diagnostics only.

### 7) QC and residual diagnostics

QC checks include:

- Initial pH consistency across repeats.
- Temperature control around target band.
- Equivalence plausibility and edge/coverage checks.
- Slope sanity relative to unity.
- Fit quality thresholds using R2.
- Residual structure review.
- Buffer-region point-count sufficiency.
- Optional ratio check for half-equivalence consistency.

Half-equivalence consistency check:

$$
\frac{V_{\mathrm{half}}}{V_{\mathrm{eq}}}\approx 0.50
$$

## Uncertainty protocol

## A) NaCl concentration preparation narrative

Concentration from mass, molar mass, and volume:

$$
C=\frac{m}{MV}
$$

Percentage uncertainty propagation form:

$$
\%\Delta C=\%\Delta m+\%\Delta V
$$

Absolute concentration uncertainty:

$$
\Delta C=C\cdot\frac{\%\Delta C}{100}
$$

## B) Burette delivered volume narrative

Delivered volume from initial and final readings:

$$
\Delta V=V_f-V_i
$$

Combined absolute uncertainty for a delivered reading pair:

$$
\Delta(\Delta V)=\Delta V_f+\Delta V_i
$$

## C) Logarithm propagation narrative

For logarithmic transform uncertainty:

$$
\Delta\left(\log_{10}X\right)=0.434\cdot\frac{\%\Delta X}{100}
$$

## D) What the current code uses

Equivalence uncertainty in code:

$$
\Delta V_{\mathrm{eq,code}}=\Delta V_{\mathrm{res}}+\Delta V_{\mathrm{burette}}
$$

with:

$$
\Delta V_{\mathrm{res}}=0.5\cdot\widetilde{\Delta V_{\mathrm{step}}}
$$

Alternative quadrature mode in code:

$$
\Delta_{\mathrm{quad}}=\sqrt{\sum_i\Delta_i^2}
$$

Apparent pKa uncertainty in code combines:

- Regression intercept uncertainty term.
- Sensitivity to equivalence perturbation.
- pH sensor systematic term.

Worst-case default combination:

$$
\Delta \mathrm{p}K_{a,\mathrm{app}}=\Delta b_{\mathrm{reg}}+\Delta b_{V_{\mathrm{eq}}}+\Delta \mathrm{pH}_{\mathrm{sys}}
$$

Rounding and reporting:

- Reporting helpers apply IB-style uncertainty significant-figure formatting.
- Value decimal places are aligned to rounded uncertainty precision.

Where sensor uncertainty enters:

- pH systematic term enters the pKa uncertainty combination in run-level analysis.
- Temperature probe uncertainty is applied through QC acceptance criteria rather than direct pKa algebra.

## Worked numerical example

Example inputs:

- Median step width = 0.20 cm^3.
- Burette delivered term = 0.20 cm^3.
- Equivalence estimate = 24.92 cm^3.

Code-style Veq uncertainty:

$$
\Delta V_{\mathrm{eq}}=0.5\times0.20+0.20=0.30\ \mathrm{cm^3}
$$

Half-equivalence volume:

$$
V_{\tfrac{1}{2}\mathrm{eq}}=\frac{24.92}{2}=12.46\ \mathrm{cm^3}
$$

If uncertainty terms are:

- 0.04 from regression,
- 0.03 from equivalence perturbation,
- 0.30 from pH systematic,

then worst-case combination is:

$$
\Delta \mathrm{p}K_{a,\mathrm{app}}=0.04+0.03+0.30=0.37
$$

Reported value form:

$$
\mathrm{p}K_{a,\mathrm{app}}=4.71\pm0.37
$$

## Implementation references

- `salty/data_processing.py`
- `salty/analysis.py`
- `salty/chemistry/hh_model.py`
- `salty/chemistry/buffer_region.py`
- `salty/stats/regression.py`
- `salty/stats/uncertainty.py`
- `salty/reporting.py`
