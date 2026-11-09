"""
This module validates the dataset using the notebook validation.

It validates the raw dataset AND every new file in data/incoming/ (which
preprocess.py will merge into the training data). New files must have the same
columns and the SAME SCALE as the raw dataset.
"""
"""
============================================================================
- Validate raw dataframe
- Validate each data/incoming/*.csv (+ check it is on the same scale as raw)

============================================================================
"""

import pandas as pd
import numpy as np

from src.config import FEATURE_COLUMNS, INCOMING_DIR, RAW_PATH, TARGET

# How far (as a fraction of the raw range) incoming values may go beyond the
# raw min/max before we consider them to be on a different scale.
SCALE_TOLERANCE = 0.5

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

    # 6. No unexpected columns (they would become NaN when files are merged)
    extra = set(dataframe.columns) - set(FEATURE_COLUMNS) - {TARGET}
    assert not extra, f"Unexpected columns: {sorted(extra)}"

    print("✅ Data Validation Passed!")

# =========================================================================
def check_same_scale(new: pd.DataFrame, reference: pd.DataFrame) -> None:
    """Fail if a feature in ``new`` is clearly outside the range of ``reference``.

    The raw diabetes features are already scaled (values around -0.2 … 0.2), so a
    file with unscaled values (e.g. age = 50) would silently ruin the model.
    """

    for col in FEATURE_COLUMNS:
        low, high = reference[col].min(), reference[col].max()
        margin = SCALE_TOLERANCE * (high - low)
        outside = ~new[col].between(low - margin, high + margin)
        assert not outside.any(), (
            f"Feature {col}: {outside.sum()} value(s) far outside the raw range "
            f"[{low:.3f}, {high:.3f}]. Is the file on the same scale as the raw data?"
        )

# =========================================================================
def main() -> None:
    """Validate the raw dataset and every new file in data/incoming/."""

    print(f"Loading dataset from: {RAW_PATH}")
    raw = pd.read_csv(RAW_PATH)
    validate_data(raw)

    for path in sorted(INCOMING_DIR.glob("*.csv")):
        print(f"Validating incoming file: {path.name}")
        new = pd.read_csv(path)
        validate_data(new)
        check_same_scale(new, raw)

# =========================================================================
if __name__ == "__main__":
    main()



