"""Tests for the quality gate in evaluate_model.py."""

import pytest

from src.config import R2_THRESHOLD, RMSE_THRESHOLD
from src.models.evaluate_model import passes_quality_gate


def metrics(rmse: float, r2: float) -> dict:
    """Build the dict that mlflow.evaluate() returns (it logs MSE, not RMSE)."""
    return {"mean_squared_error": rmse ** 2, "r2_score": r2}


def test_good_model_passes():
    assert passes_quality_gate(metrics(rmse=RMSE_THRESHOLD - 5, r2=R2_THRESHOLD + 0.05))


@pytest.mark.parametrize(
    "rmse, r2",
    [
        (RMSE_THRESHOLD + 5, R2_THRESHOLD + 0.05),   # RMSE too high
        (RMSE_THRESHOLD - 5, R2_THRESHOLD - 0.05),   # R2 too low
        (RMSE_THRESHOLD, R2_THRESHOLD + 0.05),       # thresholds are strict (< and >)
        (RMSE_THRESHOLD - 5, R2_THRESHOLD),
    ],
)
def test_bad_model_fails(rmse, r2):
    assert not passes_quality_gate(metrics(rmse, r2))


def test_missing_metrics_fail():
    assert not passes_quality_gate({})
