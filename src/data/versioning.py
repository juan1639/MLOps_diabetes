"""
Dataset version = a short hash of the processed data's content.

Every training run is tagged with it (train.py), so register_model.py only
compares runs trained AND tested on the same data. An RMSE computed on another
test set (e.g. before new rows were added from data/incoming/) is not comparable.
"""

import hashlib

import pandas as pd

from src.config import PROCESSED_PATH


def dataset_hash(df: pd.DataFrame) -> str:
    """Return a 12-character hash of the DataFrame's columns and values.

    Based on the values (not the CSV bytes), so it is the same on Windows and Linux.
    """
    digest = hashlib.md5(",".join(df.columns).encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return digest.hexdigest()[:12]


def current_dataset_hash() -> str:
    """Hash of the current processed dataset (data/processed/train_v1.csv)."""
    return dataset_hash(pd.read_csv(PROCESSED_PATH))
