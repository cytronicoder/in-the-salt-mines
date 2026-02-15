### Getting Started

#### Prerequisites

- **Python 3.9+** (Tested on Python 3.11)
- **Git** (for version control)
- **Virtual Environment** (recommended)

#### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/cytronicoder/in-the-salt-mines.git
   cd in-the-salt-mines
   ```

2. **Set Up a Virtual Environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   For development dependencies (e.g., testing):

   ```bash
   pip install -r requirements-dev.txt
   ```

#### Running the Analysis

To process all raw data, calculate apparent $pK_a$ values, and generate figures:

```bash
python main.py
```

This script will:

1. Read raw CSVs from `data/raw/`.
2. Standardize and aggregate titration steps.
3. Perform run-level regression analysis.
4. Generate output tables (`output/`) and diagnostic plots (`docs/images/`).

#### Verifying the Installation

Run the test suite to ensure all components are functioning correctly:

```bash
pytest
```

#### Reproducibility Notes

- Jitter and subsampling operations use fixed seeds for consistent plotting.
- `data/_standardized_raw/` stores normalized copies of input data for traceability.
- `output/provenance_map.csv` links source files to processed results.
