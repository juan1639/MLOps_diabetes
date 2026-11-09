"""Tests for the data pipeline: validate_data.py and preprocess.py."""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET
from src.data.preprocess import clean_data
from src.data.validate_data import validate_data


def make_df(n_rows: int = 5) -> pd.DataFrame:
    """Small valid dataset with the same columns as diabetes_dataset.csv."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n_rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df[TARGET] = rng.uniform(25, 346, size=n_rows)
    return df


# ===========================================================================
#  validate_data
# ---------------------------------------------------------------------------
def test_validate_accepts_valid_data():
    validate_data(make_df())


def test_validate_fails_without_target():
    with pytest.raises(AssertionError, match="Target"):
        validate_data(make_df().drop(columns=[TARGET]))


def test_validate_fails_with_missing_feature():
    with pytest.raises(AssertionError, match="bmi"):
        validate_data(make_df().drop(columns=["bmi"]))


def test_validate_fails_with_non_numeric_column():
    df = make_df()
    df["age"] = "x"
    with pytest.raises(AssertionError, match="Non-numeric"):
        validate_data(df)


def test_validate_fails_with_nulls():
    df = make_df()
    df.loc[0, "bp"] = np.nan
    with pytest.raises(AssertionError, match="missing values"):
        validate_data(df)


def test_validate_fails_with_empty_dataset():
    with pytest.raises(AssertionError, match="empty"):
        validate_data(make_df().iloc[0:0])


# ===========================================================================
#  clean_data
# ---------------------------------------------------------------------------
def test_clean_drops_nulls_and_duplicates():
    df = make_df(4)
    df.loc[1, "s1"] = np.nan                            # row with a null
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicated row

    cleaned = clean_data(df)

    assert len(cleaned) == 3
    assert cleaned.isnull().sum().sum() == 0
    assert not cleaned.duplicated().any()


def test_clean_converts_numeric_strings():
    df = make_df(2)
    df["age"] = df["age"].astype(str)

    cleaned = clean_data(df)

    assert pd.api.types.is_numeric_dtype(cleaned["age"])


def test_clean_does_not_modify_input():
    df = make_df(3)
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    original = df.copy()

    clean_data(df)

    pd.testing.assert_frame_equal(df, original)
