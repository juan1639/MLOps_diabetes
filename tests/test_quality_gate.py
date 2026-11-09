"""Tests for the quality gate and the champion/challenger decision in evaluate_model.py."""

import pytest

from src.config import R2_THRESHOLD, RMSE_THRESHOLD
from src.models.evaluate_model import choose_champion, passes_quality_gate


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


# ===========================================================================
#  choose_champion
# ---------------------------------------------------------------------------
GOOD = metrics(rmse=50.0, r2=0.50)
BETTER = metrics(rmse=48.0, r2=0.54)
BAD = metrics(rmse=70.0, r2=0.30)


def test_first_model_becomes_champion_if_good():
    assert choose_champion(GOOD, None) == "challenger"


def test_first_model_rejected_if_bad():
    assert choose_champion(BAD, None) is None


def test_better_challenger_wins():
    assert choose_champion(BETTER, GOOD) == "challenger"


def test_equal_challenger_wins():
    # Same RMSE (e.g. re-trained on the same data): the newer model is promoted
    assert choose_champion(GOOD, GOOD) == "challenger"


def test_worse_challenger_keeps_champion():
    assert choose_champion(GOOD, BETTER) == "champion"


def test_bad_challenger_keeps_good_champion():
    assert choose_champion(BAD, GOOD) == "champion"


def test_nothing_deployed_if_both_bad():
    assert choose_champion(BAD, BAD) is None
