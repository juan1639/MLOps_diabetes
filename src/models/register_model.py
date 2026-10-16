"""
This module registers the best run from the MLflow experiment "diabetes-dataset"
into the MLflow Model Registry.

It searches all runs in the experiment, selects the one with the lowest RMSE,
and registers that model under the name "diabetes-dataset-model".
"""
"""
The Model Registry is a central store for versioned models. After tracking
experiments, we promote the best run's model artifact so it can be referenced
by name and stage (None → Staging → Production) without coupling consumers
to a specific run ID.
"""
"""
=============================================================================
- mlflow.set_tracking_uri()
- client = MlflowClient()

- best_run(client, EXPERIMENT_NAME) -> buscar el mejor

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

=============================================================================
"""
import logging
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

# ===========================================================================
#  Paths & constants
#  
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

EXPERIMENT_NAME = "diabetes-dataset"
REGISTERED_MODEL_NAME = "diabetes-dataset-model"

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
def get_best_run(client: MlflowClient, experiment_name: str) -> mlflow.entities.Run:
    """Return the finished run with the lowest RMSE in the given experiment.

    Args:
        client: An active MlflowClient instance.
        experiment_name: Name of the MLflow experiment to search.

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

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No finished runs found in experiment '{experiment_name}'.")

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

    Args:
        client: An active MlflowClient instance.
        run: The MLflow Run whose artifact will be registered.
        registered_model_name: Name to use in the Model Registry.

    Returns:
        The created ModelVersion object.
    """

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
    """Find the best run in the california-housing experiment and register it."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    client = MlflowClient()

    logger.info("Searching for best run in experiment '%s'...", EXPERIMENT_NAME)
    best_run = get_best_run(client, EXPERIMENT_NAME)

    model_version = register_model(client, best_run, REGISTERED_MODEL_NAME)
    add_model_description(client, best_run, model_version)

    logger.info(
        "Done. Model '%s' version %s is now in the registry.",
        model_version.name,
        model_version.version,
    )

# =============================================================================
if __name__ == "__main__":
    main()





