### API Reference

This document outlines the core modules of the `salty` package, which drives the analysis pipeline.

#### Core Modules

**`salty.analysis`**

Encapsulates numerical methods for processing titration data.

- Numerical derivative calculation ($V_{eq}$)
- Linearized Henderson-Hasselbalch regression
- Standard error calculations for $pK_a$ and slope

**`salty.data_processing`**

Handles data ingestion and standardization.

- Parses raw CSVs into structured DataFrames
- Step aggregation and outlier rejection

**`salty.reporting`**

Provides strict IB-style formatting utilities.

- Significant figures enforcement based on uncertainty
- Consistency checks for output tables

**`salty.chemistry`**

Domain-specific logic for chemical property calculation.

- Identification of valid regression ranges
- Debye-Hückel and activity coefficient models

**`salty.stats`**

Statistical helpers for regression and uncertainty.

- OLS fitting and diagnostic calculations ($R^2$, residuals)
- Error propagation formulas

**`salty.plotting`**

Visualization routines for generating diagnostic figures.

- Slope, $R^2$, residuals, and temperature diagnostics
- pH vs. Volume plots with equivalence markers

#### Usage Example

```python
from salty import main_pipeline

# Run the full end-to-end analysis
main_pipeline.run()
```
