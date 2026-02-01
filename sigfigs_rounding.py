#!/usr/bin/env python3
"""Round titration data CSVs to appropriate significant figures based on equipment uncertainty.

This script processes raw titration data from the data/raw folder and saves rounded
versions to the data folder. Rounding follows IB conventions where decimal places are
determined by equipment uncertainty, and column headers are annotated with uncertainties.

Equipment uncertainties used for rounding:
    - Volume measurements (pipette, burette): 2 decimal places (±0.02-0.06 cm^3)
    - pH measurements: 1 decimal place (±0.3 pH units)
    - Temperature: 1 decimal place (±0.1°C)
    - Time: 2 decimal places (default for Logger Pro exports)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EQUIPMENT_UNCERTAINTIES: Dict[str, Tuple[float, str]] = {
    "25.0 cm3 pipette": (0.06, "abs"),
    "50.0 cm3 burette": (0.05, "abs"),
    "burette reading": (0.02, "abs"),
    "100 cm3 volumetric flask": (0.10, "abs"),
    "Vernier pH Sensor": (0.3, "abs"),
    "Digital thermometer": (0.1, "abs"),
    "Analytical balance": (0.01, "abs"),
    "250 cm3 beaker": (5.0, "pct"),
}

COLUMN_ROUNDING: Dict[str, int] = {
    "volume": 2,  # cm^3
    "ph": 1,  # pH units
    "temperature": 1,  # °C
    "time": 2,  # minutes
}


def _get_decimal_places_from_uncertainty(uncertainty: float) -> int:
    """Determine decimal places required to represent an uncertainty.

    Uses IB convention: uncertainty rounded to 1 significant figure determines
    the minimum precision of reported values.

    Args:
        uncertainty: Absolute measurement uncertainty.

    Returns:
        Number of decimal places (ndigits) for rounding.

    Examples:
        0.06 cm^3 → 2 decimal places
        0.3 pH → 1 decimal place
        0.1°C → 1 decimal place
        0.02 cm^3 → 2 decimal places
    """
    if uncertainty <= 0 or not math.isfinite(uncertainty):
        return 0

    uncertainty = abs(float(uncertainty))
    exponent = math.floor(math.log10(uncertainty))
    ndigits = -exponent
    return int(ndigits)


def _get_column_decimal_places(column_name: str) -> int | None:
    """Determine decimal places for a CSV column based on its name.

    Args:
        column_name: Name of the column from the CSV.

    Returns:
        Number of decimal places for rounding, or None if column should not be rounded.
    """
    col_lower = column_name.lower().strip()

    for key, decimals in COLUMN_ROUNDING.items():
        if key in col_lower:
            return decimals

    return None


def _get_uncertainty_for_column(column_name: str) -> str | None:
    """Get the uncertainty string for a column based on its name.

    Args:
        column_name: Name of the column from the CSV.

    Returns:
        Uncertainty string (e.g., "±0.02 cm^3"), or None if not applicable.
    """
    col_lower = column_name.lower().strip()

    if "volume" in col_lower or "naoh" in col_lower:
        return "±0.02 cm^3"
    elif "ph" in col_lower:
        return "±0.3 pH"
    elif "temperature" in col_lower:
        return "±0.1°C"
    elif "time" in col_lower:
        return "±0.01 min"

    return None


def _add_uncertainty_to_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Add uncertainty information to column headers.

    Args:
        df: DataFrame with original column names.

    Returns:
        DataFrame with uncertainty appended to column names.
    """
    rename_map = {}
    for col in df.columns:
        unc = _get_uncertainty_for_column(col)
        if unc is not None:
            rename_map[col] = f"{col} ({unc})"

    return df.rename(columns=rename_map)


def _round_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Round numeric columns to appropriate decimal places.

    Args:
        df: Raw DataFrame from CSV.

    Returns:
        DataFrame with numeric columns rounded to equipment precision.
    """
    df_rounded = df.copy()

    for col in df.columns:
        ndigits = _get_column_decimal_places(col)
        if ndigits is not None:
            if pd.api.types.is_numeric_dtype(df[col]):
                df_rounded[col] = df[col].round(ndigits)
                logger.debug("Rounded column '%s' to %d decimal places", col, ndigits)

    return df_rounded


def create_sigfigs_folder() -> Path:
    """Create the sigfigs folder in the data directory.

    Returns:
        Path to the sigfigs folder.
    """
    sigfigs_dir = Path("data/sigfigs")
    sigfigs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Created sigfigs folder at %s", sigfigs_dir)
    return sigfigs_dir


def process_csv_files(source_dir: str = "data/raw", output_dir: str | None = None):
    """Process all CSV files in source directory and save rounded versions.

    Args:
        source_dir: Directory containing the original CSV files. Defaults to data/raw.
        output_dir: Directory to save rounded CSVs. Defaults to data (parent of raw).
    """
    if output_dir is None:
        output_dir = Path("data")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(source_dir)
    if not source_path.exists():
        logger.error("Source directory does not exist: %s", source_path)
        return

    csv_files = list(source_path.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", source_path)
        return

    logger.info("Found %d CSV files to process", len(csv_files))

    for csv_file in csv_files:
        try:
            logger.info("Processing %s", csv_file.name)
            df = pd.read_csv(csv_file)
            logger.debug("Loaded %s with shape %s", csv_file.name, df.shape)

            df_rounded = _round_dataframe(df)
            df_rounded = _add_uncertainty_to_headers(df_rounded)

            output_file = output_dir / csv_file.name
            df_rounded.to_csv(output_file, index=False)
            logger.info("Saved rounded CSV to %s", output_file)

            changed_count = 0
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    changed_count += (
                        df[col] != df[col].round(_get_column_decimal_places(col) or 0)
                    ).sum()

            logger.info(
                "%s: %d total values rounded",
                csv_file.name,
                changed_count,
            )

        except Exception as e:
            logger.error("Failed to process %s: %s", csv_file.name, e)
            continue

    logger.info("Rounding complete. Rounded CSVs saved to %s", output_dir)


def print_rounding_reference():
    """Print a reference table of equipment uncertainties and rounding rules."""
    print("\n" + "=" * 80)
    print("EQUIPMENT UNCERTAINTIES AND ROUNDING REFERENCE")
    print("=" * 80)
    print()
    print(
        "Column Type          | Equipment                    | Uncertainty | Decimals"
    )
    print("-" * 80)

    for col_key, decimals in COLUMN_ROUNDING.items():
        if col_key == "volume":
            equip = "Burette / Pipette"
            unc = "±0.02-0.06 cm^3"
        elif col_key == "ph":
            equip = "Vernier pH Sensor"
            unc = "±0.3 pH"
        elif col_key == "temperature":
            equip = "Digital thermometer"
            unc = "±0.1°C"
        elif col_key == "time":
            equip = "Logger Pro recorder"
            unc = "±0.01 min"
        else:
            equip = "Unknown"
            unc = "Unknown"

        print(f"{col_key.capitalize():20} | {equip:28} | {unc:11} | {decimals}")

    print()
    print("=" * 80)
    print()


def main():
    """Execute the CSV rounding pipeline."""
    print("\nSigfigs CSV Rounding Tool")
    print("=" * 80)

    print_rounding_reference()

    process_csv_files(source_dir="data/raw", output_dir="data")

    logger.info(
        "All CSV files have been processed and saved to data/ with uncertainty headers"
    )


if __name__ == "__main__":
    main()
