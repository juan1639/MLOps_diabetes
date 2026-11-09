"""
Stable train/test split, shared by train.py and evaluate_model.py.

Each row goes to the test set or not depending on a hash of its feature values,
instead of a random shuffle. So when new rows are added (data/incoming/), every
old row stays on the same side of the split:
    - the champion is never evaluated on rows it was trained with
    - the challenger and the champion are compared on a fair test set
(With train_test_split(random_state=42), adding rows reshuffles everything.)
"""

import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET, TEST_SIZE

# Saved as a tag in every training run. If the split ever changes, bump this:
# evaluate_model.py only compares the challenger with a champion trained with
# the SAME split (otherwise the champion may have seen today's test rows).
SPLIT_METHOD = "hash-v1"


def is_test_row(df: pd.DataFrame) -> pd.Series:
    """Boolean Series: True for the rows that belong to the test set (≈ TEST_SIZE)."""
    row_hash = pd.util.hash_pandas_object(df[FEATURE_COLUMNS], index=False)
    return pd.Series((row_hash % 100).to_numpy() < TEST_SIZE * 100, index=df.index)


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return (X_train, X_test, y_train, y_test) using the stable hash-based split."""
    test = is_test_row(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X[~test], X[test], y[~test], y[test]
