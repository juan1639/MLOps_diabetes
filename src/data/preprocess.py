"""
This module cleans the raw dataset
It outputs a processed dataset ready for machine learning.

    *************************************************************************
    Voy a usar --- train_v1.csv --- para entrenar los datos,
    ya que el feature-engineering hecho en el notebook no mejoró las métricas.
    *************************************************************************
"""
"""
============================================================================
- Load raw data --> 'data/raw/diabetes_dataset.csv'
    (print 'rows_before cleaning')

- Clean data:
    - handle missing values
    - drop Nan and duplicates
    - ensure types are correct
    (print 'rows after cleaning')

- Save processed data 'data/processed/train_v1.csv'

============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path

# =========================================================================
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes_dataset.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"

# =========================================================================
def load_data(path: Path) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame.
    """
    return pd.read_csv(path)

# =========================================================================
def clean_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """Apply simple cleaning logic"""

    # Create a copy to avoid side effects
    cleaned_df = input_df.copy()

    # Handle missing values (demonstration)
    cleaned_df = cleaned_df.dropna()

    # Remove duplicates
    cleaned_df = cleaned_df.drop_duplicates()

    # Ensure types are correct
    for col in cleaned_df.columns:
        cleaned_df[col] = pd.to_numeric(cleaned_df[col])

    return cleaned_df

# =========================================================================
def save_data(df: pd.DataFrame, path: Path) -> None:
    """Save DataFrame to CSV without the index."""

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

# =========================================================================
def main() -> None:
    """Run the full preprocessing pipeline."""

    # ---- load raw data ---------------------------------
    print("Loading raw data...")
    df = load_data(RAW_PATH)
    print(f"Shape before cleaning: {df.shape}")

    # ---- clean data ------------------------------------
    print("Cleaning data...")
    df_cleaned = clean_data(df)
    print(f"Shape after cleaning: {df_cleaned.shape}")

    # ---- save data --> data/processed/train_v1.csv
    print("Saving processed data...")
    save_data(df_cleaned, PROCESSED_PATH)

    print(f"Final shape: {df_cleaned.shape}")
    print(f"Output path: {PROCESSED_PATH}")

# =========================================================================
if __name__ == "__main__":
    main()



