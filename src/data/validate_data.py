"""
This module validates the dataset using the notebook validation.
"""
"""
============================================================================
- Validate dataframe

============================================================================
"""

import pandas as pd
import numpy as np

from src.config import FEATURE_COLUMNS, RAW_PATH, TARGET

# =========================================================================
def validate_data(dataframe: pd.DataFrame) -> None:
    """Apply simple validate logic"""

    # 1. Check if target exists
    assert TARGET in dataframe.columns, "Target column missing!"

    # 2. Check for required features
    for col in FEATURE_COLUMNS:
        assert col in dataframe.columns, f"Feature {col} is missing!"

    # 3. Ensure all columns are numeric
    assert dataframe.select_dtypes(include=[np.number]).shape[1] == dataframe.shape[1], "Non-numeric data detected!"

    # 4. Critical Missing Values
    assert dataframe.isnull().sum().sum() == 0, "Critical missing values found!"

    # 5. Dataset size
    assert len(dataframe) > 0, "Dataset is empty!"

    print("✅ Data Validation Passed!")

# =========================================================================
def main() -> None:
    """Load the raw dataset and run validation."""

    path = RAW_PATH
    print(f"Loading dataset from: {path}")
    
    df = pd.read_csv(path)
    validate_data(df)

# =========================================================================
if __name__ == "__main__":
    main()



