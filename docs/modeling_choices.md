### Henderson-Hasselbalch model

Run-level Stage-2 estimation uses the linearized buffer model:

$$
\mathrm{pH}=m\log_{10}\left(\frac{V}{V_{\mathrm{eq}}-V}\right)+b
$$

Interpretation:

$$
b=\mathrm{p}K_{a,\mathrm{app}}
$$

Rationale:

- Intercept directly corresponds to apparent pKa under each ionic-strength condition
- Slope and residuals provide transparent validity checks
- Model is chemically meaningful in the buffer region

### Buffer-region definition

Points are included when:

$$
\left|\mathrm{pH}-\mathrm{p}K_{a,\mathrm{app}}\right|\le 1
$$

This window limits leverage from extreme transformed ratios and keeps the fit in a chemically interpretable region.

### Weighting policy

- Default run-level model is unweighted ordinary least squares
- Supplemental weighted fits are used only for robustness comparison at grouped level
  - The core pipeline does not currently carry a per-point variance model suitable for mandatory weighted fitting across all runs

### Residual diagnostics policy

Residual checks are used to evaluate whether linear-fit assumptions are reasonable for interpretation:

- Residual versus transformed predictor pattern check
- Residual distribution shape check

Structured residuals indicate potential mismatch in selected endpoint, buffer window, or run quality.

### Interpretation boundary

Reported pKa values are apparent values, not thermodynamic constants corrected for activity-coefficient models.

### IA correspondence

- Stage-1 corresponds to half-equivalence estimate
- Stage-2 corresponds to regression-based refinement in defined buffer region
- QC thresholds determine whether a run contributes to strict inferential subsets
