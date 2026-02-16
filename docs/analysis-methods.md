### Computational Framework

This document details the mathematical procedures used to process titration data, determine equivalence points ($V_{eq}$), and calculate apparent dissociation constants ($pK_{a,app}$).

#### Endpoint Determination

The equivalence point is identified as the maximum of the first derivative of pH with respect to volume ($\frac{dpH}{dV}$).

$$
V_{eq} = \arg \max_{V} \left( \frac{\Delta \text{pH}}{\Delta V} \right)
$$

To minimize noise artifacts, we verify the presence of a distinct peak and confirm its location by analyzing the second derivative zero-crossing.

#### Steepness Criterion

The minimum steepness $\Delta pH_{min}$ for valid inflection point identification scales with the total pH span:

$$
\Delta pH_{min} = \max(0.5, 0.10 \times \text{pH span})
$$

### Statistical Modeling

#### Linearized Henderson-Hasselbalch Equation

The dissociation of a weak acid $\text{HA}$ is modeled by the Henderson-Hasselbalch equation. For regression purposes, it is linearized as:

$$
\text{pH} = m \cdot \log_{10}\left(\frac{V}{V_{eq} - V}\right) + b
$$

where:

- $m$: Slope (theoretically 1.0)
- $b$: Y-intercept, representing $pK_{a,app}$

#### Validity Window

Regression is restricted to the buffer capacity region defined by:

$$
|\text{pH} - pK_{a,app}| \le 1.0
$$

This ensures the ratio $[\text{A}^-]/[\text{HA}]$ remains within effective buffering limits (approx. $1:10$ to $10:1$).

#### Model Diagnostics

Fit quality is assessed via:

- **Coefficient of Determination ($R^2$):** Measures variance explained.
- **Residual Analysis:** Checks for random error distribution vs. systematic bias.

### Quality Control Protocols

Strict filtering ensures high internal consistency. Runs are evaluated against the following criteria:

| Check                       | Criterion                     | Interpretation if Failed                 |
| :-------------------------- | :---------------------------- | :--------------------------------------- |
| **Initial pH Consistency**  | $\sigma_{init} \le 0.1$ pH    | Possible contamination or drift.         |
| **Temperature Control**     | $26.0 \pm 1.0 ^\circ\text{C}$ | Thermal variation affecting equilibrium. |
| **Regression Fit ($R^2$)**  | $R^2 \ge 0.98$                | Non-ideal solution behavior.             |
| **Slope Consistency ($m$)** | $0.8 \le m \le 1.2$           | Deviation from weak-acid model.          |

#### Strict Filter Subset

For robust statistical inference, a "Strict Fit" subset is defined by:

$$
R^2 \ge 0.98 \quad \text{AND} \quad |m - 1.0| \le 0.20
$$

Runs failing these checks are excluded from final aggregated statistics.

### Uncertainty Propagation

Uncertainty is propagated using standard error propagation rules for independent random errors.

#### Concentration Uncertainty

For concentration $C$ derived from mass $m$ and volume $V$:

$$
\frac{\Delta C}{C} = \sqrt{ \left( \frac{\Delta m}{m} \right)^2 + \left( \frac{\Delta V}{V} \right)^2 }
$$

#### pH and Volume Measurements

- **pH Uncertainty:** $\pm 0.05$ pH units (electrode precision).
- **Volume Uncertainty:** $\pm 0.05$ cm$^3$ (burette reading error).

Reported uncertainties represent the combined standard uncertainty of the mean for replicate trials.

### Figure Conventions

The IA publication figure set (`--figures ia`) follows a single nomenclature and styling standard across all panels:

- Symbols: `V_eq`, `V_{1/2}`, `pH(V_{1/2})`, and `pK_a,app` (apparent pKa).
- Independent-variable notation: `[NaCl]` (mol·dm^-3) and ionic strength `I` (mol·dm^-3, with `I=[NaCl]` for NaCl).
- Unit conventions: pH is unitless; titrant volume is reported as cm^3.
- Equivalence definition: `V_eq=(V_i+V_{i+1})/2` where `[V_i,V_{i+1}]` is the interval with maximum discrete `ΔpH/ΔV`; interval uncertainty is `σ(V_eq)=(V_{i+1}-V_i)/2`.
- Half-equivalence extraction: `V_{1/2}=V_eq/2`, and `pH(V_{1/2})` is obtained by explicit linear interpolation between measured points bracketing `V_{1/2}`.
- Multi-panel layouts use panel tags `(a)`, `(b)`, etc., with legends moved outside or into dedicated legend panels to avoid obscuring data.
- Temperature and calibration QC runs are flagged visually rather than silently removed; excluded points (when used) are shown in gray and documented in captions.
