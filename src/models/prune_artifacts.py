"""
Keep mlruns/ small: delete the model files of training runs that were never registered.

mlruns/ is restored from the deployed image on every CI run and copied back into
the new image, so it keeps growing (≈ 3 MB per CI run, mostly RandomForest models
that are never chosen). This script runs after evaluate_model.py and deletes
artifacts/model/ of every training run that has no version in the Model Registry.

What is kept:
    - the run itself (params, metrics, tags, feature_stats.json) → full history in the MLflow UI
    - the model files of every registered version (champion, challenger, older versions for rollback)

Pruned runs get the tag model_artifact=pruned. register_model.py never needs them:
for the same data it always picks the run that is already registered (lowest RMSE,
ties → oldest run).
"""

import logging
import shutil
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.config import EXPERIMENT_NAME, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ARTIFACT = "model"


def prune_unregistered_models(client: MlflowClient, experiment_name: str, model_name: str,
                              store_dir: Path) -> list[str]:
    """Delete <store_dir>/<exp_id>/<run_id>/artifacts/model of unregistered training runs.

    Returns the ids of the runs that were pruned.
    """

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return []

    registered_runs = {mv.run_id for mv in client.search_model_versions(f"name='{model_name}'")}

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.stage = 'training'",
        max_results=50000,
    )

    pruned = []
    for run in runs:
        run_id = run.info.run_id
        if run_id in registered_runs:
            continue
        model_dir = store_dir / experiment.experiment_id / run_id / "artifacts" / MODEL_ARTIFACT
        if not model_dir.exists():
            continue
        shutil.rmtree(model_dir)
        client.set_tag(run_id, "model_artifact", "pruned")
        pruned.append(run_id)

    return pruned


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    client = MlflowClient()

    pruned = prune_unregistered_models(client, EXPERIMENT_NAME, REGISTERED_MODEL_NAME, MLFLOW_TRACKING_URI)
    logger.info("Model files deleted from %d unregistered training run(s).", len(pruned))


if __name__ == "__main__":
    main()
