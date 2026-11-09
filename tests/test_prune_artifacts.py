"""
Tests for prune_artifacts.py against a temporary MLflow store (never touches mlruns/).

They check that only the model files of unregistered training runs are deleted,
and that the pruned runs are not chosen again by register_model.py.
"""

from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression

from src.models.prune_artifacts import prune_unregistered_models
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
    """Log a run like train.py does (tags + rmse + model artifact)."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = pd.Series([0.0, 1.0, 2.0])
    with mlflow.start_run() as run:
        mlflow.set_tags({"stage": "training", "dataset_hash": "hash-A"})
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(LinearRegression().fit(X, y), artifact_path="model")
    return run.info.run_id


def model_dir(store: Path, client, run_id: str) -> Path:
    exp_id = client.get_experiment_by_name(EXPERIMENT).experiment_id
    return store / exp_id / run_id / "artifacts" / "model"


def test_prune_keeps_registered_and_deletes_the_rest(client, tmp_path):
    best = log_training_run(rmse=50.0)
    other = log_training_run(rmse=60.0)
    register_model(client, client.get_run(best), MODEL_NAME)

    pruned = prune_unregistered_models(client, EXPERIMENT, MODEL_NAME, tmp_path)

    assert pruned == [other]
    assert model_dir(tmp_path, client, best).exists()
    assert not model_dir(tmp_path, client, other).exists()
    # The run itself (metrics, tags) stays in the history
    assert client.get_run(other).data.metrics["rmse"] == 60.0
    assert client.get_run(other).data.tags["model_artifact"] == "pruned"


def test_prune_ignores_evaluation_runs_and_is_idempotent(client, tmp_path):
    best = log_training_run(rmse=50.0)
    register_model(client, client.get_run(best), MODEL_NAME)
    with mlflow.start_run():
        mlflow.set_tag("stage", "evaluation")

    assert prune_unregistered_models(client, EXPERIMENT, MODEL_NAME, tmp_path) == []
    assert prune_unregistered_models(client, EXPERIMENT, MODEL_NAME, tmp_path) == []


def test_retraining_after_prune_reuses_registered_run(client, tmp_path):
    # CI run 1: best is registered, the other one is pruned
    first = log_training_run(rmse=50.0)
    log_training_run(rmse=60.0)
    register_model(client, client.get_run(first), MODEL_NAME)
    prune_unregistered_models(client, EXPERIMENT, MODEL_NAME, tmp_path)

    # CI run 2 with the same data: same metrics → the registered (oldest) run wins
    log_training_run(rmse=50.0)
    log_training_run(rmse=60.0)

    assert get_best_run(client, EXPERIMENT, "hash-A").info.run_id == first


def test_prune_without_experiment_does_nothing(client, tmp_path):
    assert prune_unregistered_models(client, "missing-experiment", MODEL_NAME, tmp_path) == []
