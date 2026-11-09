"""
Tests for register_model.py against a temporary MLflow store (never touches mlruns/).

They check the two fixes in the registry step:
    - evaluation runs are ignored when choosing the best run
    - registering the same run twice does not create a duplicate version
"""

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression

from src.models.register_model import get_best_run, register_model

EXPERIMENT = "test-experiment"
MODEL_NAME = "test-model"


@pytest.fixture
def client(tmp_path):
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tmp_path.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    yield MlflowClient()
    mlflow.set_tracking_uri(previous_uri)


def log_training_run(rmse: float) -> str:
    """Log a run like train.py does (tag stage=training + rmse + model artifact)."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = pd.Series([0.0, 1.0, 2.0])
    with mlflow.start_run() as run:
        mlflow.set_tag("stage", "training")
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(LinearRegression().fit(X, y), artifact_path="model")
    return run.info.run_id


def log_evaluation_run() -> str:
    """Log a run like evaluate_model.py does (tag stage=evaluation, no 'rmse')."""
    with mlflow.start_run() as run:
        mlflow.set_tag("stage", "evaluation")
        mlflow.log_metric("mean_squared_error", 1.0)
    return run.info.run_id


def test_best_run_ignores_evaluation_runs(client):
    log_training_run(rmse=60.0)
    best_id = log_training_run(rmse=50.0)
    log_evaluation_run()

    assert get_best_run(client, EXPERIMENT).info.run_id == best_id


def test_best_run_fails_without_training_runs(client):
    log_evaluation_run()

    with pytest.raises(ValueError, match="No finished training runs"):
        get_best_run(client, EXPERIMENT)


def test_register_same_run_twice_does_not_duplicate(client):
    log_training_run(rmse=50.0)
    best = get_best_run(client, EXPERIMENT)

    first = register_model(client, best, MODEL_NAME)
    second = register_model(client, best, MODEL_NAME)

    assert first.version == second.version
    assert len(client.search_model_versions(f"name='{MODEL_NAME}'")) == 1
