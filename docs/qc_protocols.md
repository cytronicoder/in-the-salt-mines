### QC checks

| Check                     | Criterion                                              | Interpretation if failed                        | Typical action                               |
| ------------------------- | ------------------------------------------------------ | ----------------------------------------------- | -------------------------------------------- |
| Initial pH consistency    | Replicates should remain near condition baseline       | Starting chemistry may differ between runs      | Recheck calibration and solution preparation |
| Temperature control       | Keep measurements in 26 C target with +/-1 C tolerance | Thermal drift can bias apparent pKa             | Repeat under improved thermal control        |
| Equivalence edge check    | Derivative peak should not lie at boundary             | Endpoint likely unstable                        | Extend volume range                          |
| Post-equivalence coverage | Require enough points after endpoint                   | Endpoint cannot be confidently localized        | Add post-equivalence measurements            |
| Steepness check           | Endpoint region must show clear pH jump                | Candidate endpoint may be pseudo-peak           | Use finer volume increments near endpoint    |
| Buffer-region count       | Ensure enough points in regression window              | Intercept and slope become unstable             | Increase sampling around buffer region       |
| Slope sanity              | Slope should remain near unity                         | Model assumptions may be violated               | Review endpoint and selected points          |
| Fit quality               | Use R2 threshold for strict subset                     | Linear model support is weak                    | Exclude from strict-fit claims               |
| Residual structure        | Residual plots should show no strong pattern           | Indicates missing structure or outliers         | Investigate run-specific issues              |
| Half-equivalence geometry | Check ratio near expected half-value                   | Internal consistency issue in endpoint workflow | Re-evaluate Veq and interpolation            |

### Strict subset filters

Strict-fit subset uses:

$$
R^2\ge 0.98
$$

$$
\left|\mathrm{slope}-1.00\right|\le 0.20
$$

Half-equivalence consistency reference:

$$
\frac{V_{\mathrm{half}}}{V_{\mathrm{eq}}}\approx 0.50
$$

### Figures and data columns

- `hh_slope_and_r2_diagnostics.png` validates slope and fit-quality behavior
- `hh_residuals_analysis.png` validates residual randomness and symmetry
- `pka_precision_by_nacl.png` validates repeatability and uncertainty consistency
- `equivalence_volumes_by_nacl.png` validates endpoint stability
- `temperature_control_by_nacl.png` validates thermal control
- `buffer_region_coverage.png` validates regression-point sufficiency
- `half_equivalence_verification.png` validates geometric consistency of half-equivalence calculations

Primary table used for QC decisions:

- `output/individual_results.csv`

### Reporting rule

- Use all valid runs for descriptive reporting
- Use QC-pass and strict-fit subsets for stronger inferential claims
- State the subset used whenever slope or trend conclusions are reported
