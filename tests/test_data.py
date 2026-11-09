"""Tests for the data pipeline: validate_data.py and preprocess.py (incl. data/incoming/)."""

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET
from src.data.preprocess import add_incoming_data, clean_data
from src.data.split import split_train_test
from src.data.validate_data import check_same_scale, validate_data
from src.data.versioning import dataset_hash


def make_df(n_rows: int = 5) -> pd.DataFrame:
    """Small valid dataset with the same columns as diabetes_dataset.csv."""
    rng = np.random.default_rng(n_rows)
    # Same scale as the real (already scaled) diabetes features: about -0.2 … 0.2
    df = pd.DataFrame(rng.uniform(-0.1, 0.1, size=(n_rows, len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
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


def test_validate_fails_with_unexpected_column():
    df = make_df()
    df["extra"] = 1.0
    with pytest.raises(AssertionError, match="extra"):
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


# ===========================================================================
#  data/incoming/
# ---------------------------------------------------------------------------
def test_same_scale_accepts_similar_values():
    check_same_scale(make_df(3), make_df(50))


def test_same_scale_rejects_unscaled_values():
    new = make_df(3)
    new["age"] = [45.0, 60.0, 33.0]   # real ages, not scaled like the raw data
    with pytest.raises(AssertionError, match="age"):
        check_same_scale(new, make_df(50))


def test_add_incoming_appends_files_in_name_order(tmp_path):
    raw = make_df(5)
    first, second = make_df(2), make_df(3)
    # Written with a different column order: the result must use the raw order
    second[list(reversed(second.columns))].to_csv(tmp_path / "b.csv", index=False)
    first.to_csv(tmp_path / "a.csv", index=False)

    merged = add_incoming_data(raw, tmp_path)

    assert len(merged) == 10
    assert list(merged.columns) == FEATURE_COLUMNS + [TARGET]
    pd.testing.assert_frame_equal(merged.iloc[5:7].reset_index(drop=True), first, check_exact=False)


def test_add_incoming_without_files_returns_raw(tmp_path):
    raw = make_df(5)
    pd.testing.assert_frame_equal(add_incoming_data(raw, tmp_path), raw)


# ===========================================================================
#  dataset_hash
# ---------------------------------------------------------------------------
def test_dataset_hash_changes_with_data():
    df = make_df(5)
    assert dataset_hash(df) == dataset_hash(df.copy())
    assert dataset_hash(df) != dataset_hash(pd.concat([df, make_df(1)]))


# ===========================================================================
#  Stable train/test split
# ---------------------------------------------------------------------------
def test_split_is_about_test_size():
    df = make_df(1000)
    _, X_test, _, _ = split_train_test(df)
    assert 0.15 < len(X_test) / len(df) < 0.25


def test_split_keeps_old_rows_on_the_same_side_when_rows_are_added():
    old = make_df(300)
    new = pd.concat([old, make_df(100)], ignore_index=True)

    old_test = set(split_train_test(old)[1].index)
    new_test = set(split_train_test(new)[1].index)

    # Rows 0..299 are the same rows in both: same test membership
    assert {i for i in new_test if i < 300} == old_test
