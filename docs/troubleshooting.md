### Troubleshooting

Common issues and solutions encountered when running the `in-the-salt-mines` pipeline.

#### "Missing Column: 'pH'" or "'Volume (cm^3)'"

**Problem:** The input CSV file does not contain the expected headers.
**Solution:** Ensure raw files are formatted correctly (e.g., Logger Pro exports). The pipeline expects case-sensitive headers like `pH` and `Volume`. Standardize column names in `data/raw/` if necessary.

#### "Empty DataFrame: Run X"

**Problem:** A run was extracted but contains no data points.
**Solution:** Check the `run_extraction` logic (`salty/data_processing.py`) to ensure run boundaries are correctly identified in the standardized files.

#### "Slope Warning: |m - 1.0| > 0.10"

**Cause:** The fitted slope deviates significantly from the theoretical value of 1.0.
**Impact:** This run may be excluded from the Strict Fit subset.
**Resolution:** Verify data quality (e.g., drift, temperature stability). If the deviation is systematic, check for interferences or calibration errors.

#### "Low R2: < 0.98"

**Cause:** The linear regression model fits the data poorly.
**Impact:** Exclusion from high-confidence analysis metrics.
**Resolution:** Inspect the corresponding titration curve and residuals plot. Look for non-linear behavior or outliers in the buffer region.

#### "Missing Uncertainty Metadata"

**Problem:** Reporting functions fail due to absent uncertainty columns.
**Solution:** Ensure all calculation steps propagate errors correctly. Check intermediate DataFrames for `NaN` or missing `_uncertainty` fields.

#### "Plot Generation Failed"

**Problem:** `matplotlib` errors during figure creation.
**Solution:** Check dependencies (`requirements.txt`). Ensure aggregating statistics (e.g., `mean`, `std`) are not operating on empty DataFrames (due to valid run filtering).
