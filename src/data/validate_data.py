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

# =========================================================================
def validate_data(dataframe: pd.DataFrame) -> None:
    """Apply simple validate logic"""

    # 1. Check if target exists
    assert 'target' in dataframe.columns, "Target column missing!"

    # 2. Check for required features
    required_features = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

    for col in required_features:
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

    path = "data/raw/diabetes_dataset.csv"
    print(f"Loading dataset from: {path}")
    
    df = pd.read_csv(path)
    validate_data(df)

# =========================================================================
if __name__ == "__main__":
    main()



