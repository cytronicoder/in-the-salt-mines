### Threshold and constant log

| Choice                          | Value                                 | Why this value                            | Sensitivity    | Code location                                     |
| ------------------------------- | ------------------------------------- | ----------------------------------------- | -------------- | ------------------------------------------------- |
| Designed NaCl levels            | 0.0, 0.2, 0.4, 0.6, 0.8 M             | Matches IA independent-variable levels    | High           | `main.py`                                         |
| Equivalence edge buffer         | 2 points                              | Reduces boundary false-positive peaks     | Medium         | `salty/analysis.py`                               |
| Minimum post-equivalence points | 3 points                              | Improves endpoint robustness              | Medium to high | `salty/analysis.py`                               |
| Steepness rule                  | max of fixed and span-based threshold | Rejects weak pseudo-peaks                 | Medium         | `salty/analysis.py`                               |
| Dense derivative fallback grid  | 2500 points                           | Stabilizes dense fallback endpoint search | Medium         | `salty/analysis.py`                               |
| Buffer inclusion window         | absolute pH difference <= 1           | Standard H-H buffer window                | High           | `salty/chemistry/buffer_region.py`                |
| Slope warning trigger           | abs slope minus 1.0 > 0.10            | Flags non-ideal run behavior              | Medium         | `salty/chemistry/hh_model.py`                     |
| Strict-fit slope criterion      | abs slope minus 1.0 <= 0.20           | Defines high-confidence subset            | High           | `main.py`                                         |
| Strict-fit fit criterion        | R2 >= 0.98                            | Defines high-confidence subset            | High           | `main.py`                                         |
| QC temperature target           | 26.0 C                                | Matches IA control condition              | Medium         | `salty/plotting/qc_plots.py`                      |
| QC temperature tolerance        | +/-1.0 C                              | Practical control band                    | Medium         | `salty/plotting/qc_plots.py`                      |
| Recommended buffer points line  | 10 points                             | Visual cue for fit stability              | Medium         | `salty/plotting/qc_plots.py`                      |
| pH systematic term              | 0.30 pH                               | Instrument-bound uncertainty term         | High           | `salty/analysis.py`                               |
| Veq burette term default        | 0.10 cm^3                             | Instrument-bound uncertainty term         | Medium to high | `salty/analysis.py`                               |
| Uncertainty combination mode    | worst_case by default                 | Conservative IA-style reporting           | High           | `salty/analysis.py`, `salty/stats/uncertainty.py` |

### Formula-backed checks

Steepness rule form:

$$
\Delta \mathrm{pH}_{\mathrm{min}}=\max\left(0.5,\ 0.10\times\mathrm{pH\ span}\right)
$$

Strict-fit slope rule:

$$
\left|\mathrm{slope}-1.0\right|\le 0.20
$$

Strict-fit fit-quality rule:

$$
R^2\ge 0.98
$$

### Change-control rule

When a threshold is changed:

1. Record old and new values
2. Re-run all iteration subsets
3. Compare run counts, fitted slope/intercept, and uncertainty behavior before drawing conclusions
