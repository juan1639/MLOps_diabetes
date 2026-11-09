"""
This module registers the best run from the MLflow experiment "diabetes-dataset"
into the MLflow Model Registry.

It searches the training runs made with the CURRENT dataset (same dataset_hash
tag), selects the one with the lowest RMSE, registers that model under the name
"diabetes-dataset-model" and gives it the alias "challenger".
evaluate_model.py then decides whether the challenger replaces the champion.
"""
"""
The Model Registry is a central store for versioned models. After tracking
experiments, we promote the best run's model artifact so it can be referenced
by name and version (and later by alias, see evaluate_model.py) without
coupling consumers to a specific run ID.
"""
"""
=============================================================================
- mlflow.set_tracking_uri()
- client = MlflowClient()

- best_run(client, EXPERIMENT_NAME, data_hash) -> buscar el mejor (mismos datos)

- model_version = register_model(
        client,
        best_run,
        REGISTERED_MODEL_NAME
    )                               -> registrar el modelo

- add_model_description(
        client,
        best_run,
        model_version
    )                               -> agregar una descripcion

- set alias "challenger"            -> lo evaluara evaluate_model.py

=============================================================================
"""
import logging
import mlflow
from mlflow.tracking import MlflowClient

from src.config import CHALLENGER_ALIAS, EXPERIMENT_NAME, MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME
from src.data.versioning import current_dataset_hash

# ===========================================================================
#  Logging
#  
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

logger = logging.getLogger(__name__)

# ===========================================================================
#  Get best run
#  
# ---------------------------------------------------------------------------
def get_best_run(
        client: MlflowClient,
        experiment_name: str,
        data_hash: str | None = None,
) -> mlflow.entities.Run:
    """Return the finished TRAINING run with the lowest RMSE in the given experiment.

    Evaluation runs (created by evaluate_model.py in the same experiment) have
    no 'rmse' metric and no model artifact, so they are excluded by tag.

    If ``data_hash`` is given, only runs trained on that dataset version are
    considered: the store may contain older runs (restored from the previous
    Docker image in CI) whose RMSE was measured on a different test set.
    On a tie, the OLDEST run wins, so re-training on the same data reuses the
    already registered version instead of creating a new one every time.

    Args:
        client: An active MlflowClient instance.
        experiment_name: Name of the MLflow experiment to search.
        data_hash: Optional dataset version (tag ``dataset_hash``) to filter by.

    Returns:
        The MLflow Run object with the best (lowest) RMSE.

    Raises:
        ValueError: If the experiment does not exist or has no finished runs.
    """

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found. "
            "Run train.py first to create runs."
        )

    filter_string = "status = 'FINISHED' and tags.stage = 'training'"
    if data_hash is not None:
        filter_string += f" and tags.dataset_hash = '{data_hash}'"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["metrics.rmse ASC", "attributes.start_time ASC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(
            f"No finished training runs found in experiment '{experiment_name}'"
            + (f" for dataset_hash '{data_hash}'." if data_hash else ".")
        )

    best = runs[0]

    logger.info(
        "Best run selected — id: %s  rmse: %.4f  model_type: %s",
        best.info.run_id,
        best.data.metrics.get("rmse", float("nan")),
        best.data.tags.get("model_type", "unknown"),
    )
    return best

# ===========================================================================
#  Register model
#  
# ---------------------------------------------------------------------------
def register_model(
        client: MlflowClient,
        run: mlflow.entities.Run,
        registered_model_name: str,
) -> mlflow.entities.model_registry.ModelVersion:
    """Register the model artifact from a run into the Model Registry.

    If the registered model does not exist yet, MLflow creates it automatically.
    If this run is already registered (e.g. the script is run twice without
    retraining), the existing version is returned instead of creating a duplicate.

    Args:
        client: An active MlflowClient instance.
        run: The MLflow Run whose artifact will be registered.
        registered_model_name: Name to use in the Model Registry.

    Returns:
        The created ModelVersion object.
    """

    existing = client.search_model_versions(
        f"name='{registered_model_name}' and run_id='{run.info.run_id}'"
    )
    if existing:
        version = max(existing, key=lambda v: int(v.version))
        logger.info(
            "Run %s is already registered as version %s — not creating a new one.",
            run.info.run_id,
            version.version,
        )
        return version

    model_uri = f"runs:/{run.info.run_id}/model"
    logger.info("Registering model from URI: %s", model_uri)

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    logger.info(
        "Model registered — name: '%s'  version: %s",
        model_version.name,
        model_version.version,
    )
    return model_version

# ===========================================================================
#  Add model description
#  
# ---------------------------------------------------------------------------
def add_model_description(
        client: MlflowClient,
        run: mlflow.entities.Run,
        model_version: mlflow.entities.model_registry.ModelVersion,
) -> None:
    """Add a description to the registered model version with its key metrics.

    Args:
        client: An active MlflowClient instance.
        run: The source run, used to pull metric values for the description.
        model_version: The ModelVersion to annotate.
    """

    metrics = run.data.metrics
    model_type = run.data.tags.get("model_type", "unknown")

    description = (
        f"Best run from experiment '{EXPERIMENT_NAME}'. "
        f"Model type: {model_type}. "
        f"RMSE: {metrics.get('rmse', float('nan')):.4f}  "
        f"MAE: {metrics.get('mae', float('nan')):.4f}  "
        f"R2: {metrics.get('r2', float('nan')):.4f}"
    )

    client.update_model_version(
        name=model_version.name,
        version=model_version.version,
        description=description,
    )
    logger.info("Description added to model version.")

# ===========================================================================
#  Main
#  
# ---------------------------------------------------------------------------
def main() -> None:
    """Find the best training run in the diabetes-dataset experiment and register it."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    client = MlflowClient()

    data_hash = current_dataset_hash()
    logger.info("Searching for best run in experiment '%s' (dataset_hash=%s)...", EXPERIMENT_NAME, data_hash)
    best_run = get_best_run(client, EXPERIMENT_NAME, data_hash)

    model_version = register_model(client, best_run, REGISTERED_MODEL_NAME)
    add_model_description(client, best_run, model_version)

    # Mark it as the candidate; evaluate_model.py compares it with the champion
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHALLENGER_ALIAS, model_version.version)

    logger.info(
        "Done. Model '%s' version %s is now in the registry with alias '%s'.",
        model_version.name,
        model_version.version,
        CHALLENGER_ALIAS,
    )

# =============================================================================
if __name__ == "__main__":
    main()





