"""
This module merges the raw dataset with any new rows in data/incoming/*.csv,
cleans the result and outputs a processed dataset ready for machine learning.
(validate_data.py checks the incoming files before this step.)

    *************************************************************************
    Voy a usar --- train_v1.csv --- para entrenar los datos,
    ya que el feature-engineering hecho en el notebook no mejoró las métricas.
    *************************************************************************
"""
"""
============================================================================
- Load raw data --> 'data/raw/diabetes_dataset.csv'
- Add new rows  --> 'data/incoming/*.csv' (if any)
    (print 'rows_before cleaning')

- Clean data:
    - handle missing values
    - drop Nan and duplicates
    - ensure types are correct
    (print 'rows after cleaning')

- Save processed data 'data/processed/train_v1.csv'

============================================================================
"""

import pandas as pd
from pathlib import Path

from src.config import FEATURE_COLUMNS, INCOMING_DIR, PROCESSED_PATH, RAW_PATH, TARGET

# =========================================================================
def load_data(path: Path) -> pd.DataFrame:
    """
    Load a CSV file and return a DataFrame.
    """
    return pd.read_csv(path)

# =========================================================================
def add_incoming_data(df: pd.DataFrame, incoming_dir: Path) -> pd.DataFrame:
    """Append every CSV in ``incoming_dir`` to ``df`` (same column order as raw).

    Files are read in name order, so the result (and the dataset hash) is reproducible.
    """

    columns = FEATURE_COLUMNS + [TARGET]
    frames = [df[columns]]

    for path in sorted(incoming_dir.glob("*.csv")):
        new = pd.read_csv(path)
        print(f"  + {len(new)} rows from {path.name}")
        frames.append(new[columns])

    return pd.concat(frames, ignore_index=True)

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

    # ---- add new rows from data/incoming/ ---------------
    print("Adding incoming data...")
    df = add_incoming_data(df, INCOMING_DIR)
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



