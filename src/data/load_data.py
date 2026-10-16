"""
    This module loads the raw dataset
    and saves it as a CSV file at 'data/raw/diabetes_dataset.csv'.
"""
"""
============================================================================
- Load data from sklearn.datasets

- Save 'data/raw/diabetes_dataset.csv' (raw data)

============================================================================
"""

import os
#import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

# ==========================================================================
OUTPUT_PATH = "data/raw/diabetes_dataset.csv"

# ==========================================================================
def load_raw_data() -> pd.DataFrame:
    """Load the Diabetes dataset and return it as a DataFrame."""

    diabetes = load_diabetes()

    # Create a pandas DataFrame for features
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

    # View columns without target
    print("\nSin target:\n", df.columns.to_list())

    # Add the target column
    df['target'] = diabetes.target

    # View columns and shape after 'target' 
    print("\nTarget added:\n")
    print(f"Column Names: {df.columns.tolist()}")
    print(f"Dataset Shape: {df.shape}")

    # Check some rows of the df
    print("\ndf_head:\n", df.head())

    return df

# ==========================================================================
def save_dataset(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to a CSV file, creating directories as needed.

    Args:
        df: DataFrame to save.
        path: Destination file path.
    """

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)
    
    df.to_csv(path, index=False)

# ==========================================================================
def main() -> None:
    """Run the data ingestion pipeline."""

    print("Loading dataset...")
    df = load_raw_data()

    print("Saving dataset...")
    save_dataset(df, OUTPUT_PATH)

    print(f"Dataset shape: {df.shape}")
    print(f"Output path: {OUTPUT_PATH}")
    print("Done.")

# ==========================================================================
if __name__ == "__main__":
    main()




