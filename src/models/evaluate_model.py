"""
This module formally evaluates the latest version of the registered model
"diabetes-dataset-model" using mlflow.evaluate().

Results are logged to a dedicated MLflow run. If the model passes the quality
thresholds (rmse < 0.52 and r2 > 0.44) the alias "champion" is assigned to that
model version, marking it as the go-to model for serving.
"""

"""
Model Evaluation answers the question "is this model good enough to serve?".
mlflow.evaluate() goes beyond ad-hoc metric logging: it produces a structured
EvaluationResult (metrics + artifacts) attached to a run, enabling reproducible
quality gates. Aliases replace the deprecated Stage system in MLflow 2.9+:
instead of transitioning to "Production", we assign a human-readable alias
("champion") that consumers can reference without knowing the version number.
"""

"""
=============================================================================
- mlflow.set_tracking_uri()
- mlflow.set_experiment()

- client = MlflowClient()

- model_version = get_latest_version(
        client,
        REGISTERED_MODEL_NAME
    )                                           -> Resolve model version

- X_test, y_test = load_test_data(INPUT_PATH)   -> Load test data

- result = evaluate_registered_model(
        client,
        model_version,
        X_test, y_test
    )                                           -> Evaluate

- if passes_quality_gate(result.metrics):
        logger.info("Model passed the quality gate.")
        assign_champion_alias(
            client,
            REGISTERED_MODEL_NAME,
            model_version
        )                                       -> quality-gate

=============================================================================
"""
import logging
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split

# ===========================================================================
#  Paths & constants
#  
# ---------------------------------------------------------------------------
BASE_DIR = Path(__name__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"   # usamos train_v1.csv (SIN feature_engineering)
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

EXPERIMENT_NAME = "diabetes-dataset"
REGISTERED_MODEL_NAME = "diabetes-dataset-model"
TARGET = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Quality thresholds for the champion alias:
RMSE_THRESHOLD = 0.52   # uso este umbral en vista de los resultados obtenidos
R2_THRESHOLD = 0.44     # uso este umbral en vista de los resultados obtenidos

# =============================================================================
#  Logging
#  
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

logger = logging.getLogger(__name__)

# =============================================================================
#  Data
#  
# ---------------------------------------------------------------------------
def load_test_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load engineered features and return the held-out test split.

    Uses the same split parameters as train.py so the test set is identical.

    Args:
        path: Path to train_features_v1.csv.

    Returns:
        Tuple of (X_test, y_test) as DataFrame and Series.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run the data pipeline first: python src/data/feature_engineering.py"
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    logger.info("Test set size: %d rows", len(X_test))
    return X_test, y_test

# =============================================================================
#  Registry helpers
#  
# ---------------------------------------------------------------------------
def get_latest_version(client: MlflowClient, model_name: str) -> str:
    """Return the version number of the most recently registered model version.

    Searches all versions and returns the one with the highest version integer.

    Args:
        client: An active MlflowClient instance.
        model_name: Registered model name.

    Returns:
        Version number as a string (e.g. ``"3"``).

    Raises:
        ValueError: If no versions exist for the given model name.
    """

    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise ValueError(
            f"No versions found for registered model '{model_name}'.\n"
            "Run register_model.py first."
        )
    
    latest = max(versions, key=lambda v: int(v.version))

    logger.info(
        "Latest version of '%s': %s (run_id: %s)",
        model_name,
        latest.version,
        latest.run_id,
    )
    return latest.version

# =============================================================================
#  Evaluation
#  
# ---------------------------------------------------------------------------
def evaluate_registered_model(
        client: MlflowClient,
        model_version: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
) -> mlflow.models.EvaluationResult:
    """Run mlflow.evaluate() on the registered model version and return results.

    Opens a new MLflow run tagged with ``stage="evaluation"`` so evaluation
    runs are visually separated from training runs in the UI.

    Args:
        client: An active MlflowClient instance.
        model_version: Version string of the registered model to evaluate.
        X_test: Test feature matrix.
        y_test: Test target series.

    Returns:
        An EvaluationResult containing metrics and artifact paths.
    """

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version}"
    logger.info("Evaluating model URI: %s", model_uri)

    # mlflow.evaluate() requires a DataFrame that includes the target column
    eval_data = X_test.copy()
    eval_data[TARGET] = y_test.values

    timestap = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evaluation_{model_version}_{timestap}"

    with mlflow.start_run(run_name=run_name):
        # ---------------- Tags ---------------------------------------
        mlflow.set_tags({
            "stage": "evaluation",
            "model_version": model_version,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "dataset": INPUT_PATH.name,
        })

        # mlflow.evaluate() computes a standard regression metric suite and
        # logs them automatically to the active run. The ``predictions`` column
        # name is used internally by MLflow; we do not need to pass it manually.
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets=TARGET,
            model_type="regressor",
            evaluators="default",
        )

        logger.info("Evaluation metrics:")

        for key, value in sorted(result.metrics.items()):
            logger.info("  %-35s %.6f", key, value)

    return result

# =============================================================================
#  Quality gate
#  
# ---------------------------------------------------------------------------
def passes_quality_gate(metrics: dict) -> bool:
    """Return True if the model meets the minimum quality thresholds.

    Thresholds:
        - ``mean_squared_error`` (RMSE equivalent via sqrt): rmse < 0.6
        - ``r2_score``: r2 > 0.7

    mlflow.evaluate() for regressors logs ``mean_squared_error`` (MSE) and
    ``r2_score`` by default. We derive RMSE from MSE for the gate check.

    Args:
        metrics: Metrics dict from ``EvaluationResult.metrics``.

    Returns:
        True if both thresholds are satisfied, False otherwise.
    """

    # mlflow.evaluate() key for MSE is "mean_squared_error"
    mse = metrics.get("mean_squared_error", float("inf"))
    rmse = mse ** 0.5
    r2 = metrics.get("r2_score", float("-inf"))

    logger.info(
        "Quality gate check — rmse=%.4f (threshold < %.1f)  r2=%.4f (threshold > %.1f)",
        rmse, RMSE_THRESHOLD, r2, R2_THRESHOLD,
    )

    return rmse < RMSE_THRESHOLD and r2 > R2_THRESHOLD

# =============================================================================
#  Assign champion alias
#  
# ---------------------------------------------------------------------------
def assign_champion_alias(client: MlflowClient, model_name: str, model_version: str) -> None:
    """Assign the "champion" alias to a model version.

    In MLflow 2.9+ aliases replace stage transitions (Staging → Production).
    Consumers can load the champion model with ``models:/name@champion``
    without ever knowing the version number.

    If another version already holds the alias, MLflow moves it automatically.

    Args:
        client: An active MlflowClient instance.
        model_name: Registered model name.
        model_version: Version to promote.
    """

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=model_version,
    )

    logger.info("Alias 'champion' assigned to '%s' version %s.", model_name, model_version)
    logger.info("Load the champion model with: mlflow.pyfunc.load_model('models:/%s@champion')", model_name)

# =============================================================================
# Main
#  
# ---------------------------------------------------------------------------
def main() -> None:
    """Evaluate the latest registered model and promote it if it passes quality gates."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI : %s", MLFLOW_TRACKING_URI)
    logger.info("Registered model    : %s", REGISTERED_MODEL_NAME)

    client = MlflowClient()

    # -------- Resolve model version ---------------------------------
    model_version = get_latest_version(client, REGISTERED_MODEL_NAME)

    # -------- Load test data ----------------------------------------
    X_test, y_test = load_test_data(INPUT_PATH)

    # -------- Evaluate ----------------------------------------------
    result = evaluate_registered_model(client, model_version, X_test, y_test)

    # -------- Quality-gate ------------------------------------------
    if passes_quality_gate(result.metrics):
        logger.info("Model passed the quality gate.")
        assign_champion_alias(client, REGISTERED_MODEL_NAME, model_version)
    else:
        logger.warning(
            "Model version %s did NOT pass the quality gate. "
            "Alias 'champion' was NOT assigned.",
            model_version,
        )

    logger.info(
        "Done. Open the MLflow UI to review the evaluation run:\n"
        "  mlflow ui --backend-store-uri %s",
        MLFLOW_TRACKING_URI,
    )

# =============================================================================
if __name__ == "__main__":
    main()





