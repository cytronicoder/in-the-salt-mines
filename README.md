This is a Python package for analyzing weak acid-strong base titration data, specifically exploring how ionic strength from NaCl affects the half-equivalence pH (apparent pKa, pKa_app) in ethanoic acid-NaOH titrations.

To install and use this package, you need:

- Python 3.8 or higher
- pip package manager

To install from source, clone the repository and install dependencies from `requirements.txt`:

```bash
git clone https://github.com/cytronicoder/in-the-salt-mines.git
cd in-the-salt-mines
pip install -r requirements.txt
```

For development, install additional dependencies from `requirements-dev.txt` and set up pre-commit hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

To get started, run the complete analysis pipeline:

```bash
python main.py
```

This will:

- Process all CSV files in the `data/` directory
- Generate titration curves and statistical summaries
- Create output in both `output/with_raw/` and `output/without_raw/` folders
- Save results to CSV files

Here's a simple example of how to use the core functions programmatically:

```python
from salty.analysis import (
    build_summary_plot_data,
    calculate_statistics,
    create_results_dataframe,
    process_all_files,
)
from salty.plotting import plot_titration_curves, plot_statistical_summary

# Process titration data
files = [("data/titration_0.0M.csv", 0.0), ("data/titration_0.5M.csv", 0.5)]
results = process_all_files(files)

# Generate plots
plot_titration_curves(results, "output", show_raw_pH=True)
results_df = create_results_dataframe(results)
stats_df = calculate_statistics(results_df)
summary = build_summary_plot_data(stats_df, results_df)
plot_statistical_summary(summary, "output")
```

This package is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
