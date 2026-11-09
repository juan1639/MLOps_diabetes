"""
This module decides which model version is the "champion" (the one the API serves).

It evaluates, with mlflow.evaluate() and on the CURRENT test set:
    - the "challenger" (best new version, set by register_model.py)
    - the current "champion", if there is one

The challenger replaces the champion only if it passes the quality thresholds
(rmse < 60 and r2 > 0.44) AND is not worse than the champion (rmse <=).
If no model is good enough, the script exits with code 1 so CI deploys nothing.
"""

"""
Model Evaluation answers the question "is this model good enough to serve?".
mlflow.evaluate() goes beyond ad-hoc metric logging: it produces a structured
EvaluationResult (metrics + artifacts) attached to a run, enabling reproducible
quality gates. Aliases replace the deprecated Stage system in MLflow 2.9+:
instead of transitioning to "Production", we assign a human-readable alias
("champion") that consumers can reference without knowing the version number.

Champion / challenger: passing fixed thresholds is not enough. A new model could
pass them and still be worse than the one already in production. Both are
evaluated on the same (current) test set, so the comparison is fair even when
the data changed since the champion was trained (e.g. new rows in data/incoming/).
"""

"""
=============================================================================
- mlflow.set_tracking_uri()
- mlflow.set_experiment()

- client = MlflowClient()

- challenger = get_version_by_alias(client, "challenger")   -> candidato
- champion   = get_version_by_alias(client, "champion")     -> actual (o None)

- X_test, y_test = load_test_data(INPUT_PATH)               -> Load test data

- evaluate_registered_model(...) para challenger y champion -> Evaluate

  (si el champion se entreno con otro split -> no se compara, solo umbrales)

- winner = choose_champion(challenger_metrics, champion_metrics)
    - "challenger" -> assign_champion_alias(challenger)
    - "champion"   -> se mantiene el champion actual
    - None         -> sys.exit(1)  (CI no despliega)

=============================================================================
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# Quality thresholds (RMSE_THRESHOLD, R2_THRESHOLD) are explained in src/config.py.
from src.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROCESSED_PATH as INPUT_PATH,   # usamos train_v1.csv (SIN feature_engineering)
    R2_THRESHOLD,
    REGISTERED_MODEL_NAME,
    RMSE_THRESHOLD,
    TARGET,
)
from src.data.split import SPLIT_METHOD, split_train_test

# =============================================================================
#  Logging
#
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

logger = logging.getLogger(__name__)


def ci_warning(message: str) -> None:
    """Log a warning and, inside GitHub Actions, also show it in the run summary."""
    logger.warning(message)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::warning::{message}")

# =============================================================================
#  Data
#
# ---------------------------------------------------------------------------
def load_test_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed dataset and return the held-out test split.

    Uses the same stable split as train.py (src/data/split.py), so the test set
    never contains rows the models were trained with — not even the champion's,
    when new rows were added after it was trained.

    Args:
        path: Path to train_v1.csv.

    Returns:
        Tuple of (X_test, y_test) as DataFrame and Series.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run the data pipeline first: python -m src.data.preprocess"
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    _, X_test, _, y_test = split_train_test(df)

    logger.info("Test set size: %d rows", len(X_test))
    return X_test, y_test

# =============================================================================
#  Registry helpers
#
# ---------------------------------------------------------------------------
def get_version_by_alias(client: MlflowClient, model_name: str, alias: str) -> str | None:
    """Return the version that holds ``alias``, or None if the alias does not exist.

    Args:
        client: An active MlflowClient instance.
        model_name: Registered model name.
        alias: Alias to resolve (e.g. "champion" or "challenger").

    Returns:
        Version number as a string (e.g. ``"3"``), or None.
    """

    try:
        version = client.get_model_version_by_alias(model_name, alias).version
    except MlflowException:
        logger.info("No version of '%s' has the alias '%s'.", model_name, alias)
        return None

    logger.info("'%s@%s' -> version %s", model_name, alias, version)
    return str(version)

def same_split(client: MlflowClient, model_name: str, version: str) -> bool:
    """True if that version was trained with the current train/test split.

    A champion trained with another split (e.g. the old random split, before
    src/data/split.py existed) may have been trained on rows that are in today's
    test set, so its metrics would look better than they are.
    """

    run_id = client.get_model_version(model_name, version).run_id
    method = client.get_run(run_id).data.tags.get("split_method")
    return method == SPLIT_METHOD

# =============================================================================
#  Evaluation
#
# ---------------------------------------------------------------------------
def evaluate_registered_model(
        model_version: str,
        role: str,
        X_test: pd.DataFrame,
        y_test: pd.Series,
) -> dict:
    """Run mlflow.evaluate() on a registered model version and return its metrics.

    Opens a new MLflow run tagged with ``stage="evaluation"`` (so register_model.py
    ignores it) and ``role`` ("challenger" / "champion").

    Args:
        model_version: Version string of the registered model to evaluate.
        role: "challenger" or "champion", only used for the run name and tags.
        X_test: Test feature matrix.
        y_test: Test target series.

    Returns:
        The metrics dict from the EvaluationResult.
    """

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{model_version}"
    logger.info("Evaluating %s: %s", role, model_uri)

    # mlflow.evaluate() requires a DataFrame that includes the target column
    eval_data = X_test.copy()
    eval_data[TARGET] = y_test.values

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evaluation_{role}_v{model_version}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        # ---------------- Tags ---------------------------------------
        mlflow.set_tags({
            "stage": "evaluation",
            "role": role,
            "model_version": model_version,
            "registered_model_name": REGISTERED_MODEL_NAME,
            "dataset": INPUT_PATH.name,
        })

        # mlflow.evaluate() computes a standard regression metric suite and
        # logs them automatically to the active run.
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets=TARGET,
            model_type="regressor",
            evaluators="default",
        )

    logger.info(
        "[%s v%s] rmse=%.4f  r2=%.4f",
        role, model_version, rmse_of(result.metrics), result.metrics.get("r2_score", float("nan")),
    )
    return result.metrics

# =============================================================================
#  Quality gate & champion decision
#
# ---------------------------------------------------------------------------
def rmse_of(metrics: dict) -> float:
    """RMSE from mlflow.evaluate() metrics (it logs MSE as "mean_squared_error")."""
    return metrics.get("mean_squared_error", float("inf")) ** 0.5


def passes_quality_gate(metrics: dict) -> bool:
    """Return True if the model meets the minimum quality thresholds.

    Thresholds:
        - ``mean_squared_error`` (RMSE equivalent via sqrt): rmse < RMSE_THRESHOLD
        - ``r2_score``: r2 > R2_THRESHOLD

    Args:
        metrics: Metrics dict from ``EvaluationResult.metrics``.

    Returns:
        True if both thresholds are satisfied, False otherwise.
    """

    rmse = rmse_of(metrics)
    r2 = metrics.get("r2_score", float("-inf"))

    logger.info(
        "Quality gate check — rmse=%.4f (threshold < %.2f)  r2=%.4f (threshold > %.2f)",
        rmse, RMSE_THRESHOLD, r2, R2_THRESHOLD,
    )

    return rmse < RMSE_THRESHOLD and r2 > R2_THRESHOLD


def choose_champion(challenger_metrics: dict, champion_metrics: dict | None) -> str | None:
    """Decide which model should be served.

    Rules:
        - The challenger wins if it passes the quality gate and its RMSE is not
          worse than the champion's (or there is no champion yet).
        - Otherwise the current champion stays, if it still passes the gate
          on the current data.
        - If neither passes, return None (nothing should be deployed).

    Args:
        challenger_metrics: Metrics of the challenger on the current test set.
        champion_metrics: Metrics of the current champion, or None if there is none.

    Returns:
        "challenger", "champion" or None.
    """

    challenger_ok = passes_quality_gate(challenger_metrics)

    if champion_metrics is None:
        return "challenger" if challenger_ok else None

    if challenger_ok and rmse_of(challenger_metrics) <= rmse_of(champion_metrics):
        return "challenger"

    return "champion" if passes_quality_gate(champion_metrics) else None

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
        alias=CHAMPION_ALIAS,
        version=model_version,
    )

    logger.info("Alias 'champion' assigned to '%s' version %s.", model_name, model_version)
    logger.info("Load the champion model with: mlflow.pyfunc.load_model('models:/%s@champion')", model_name)

# =============================================================================
# Main
#
# ---------------------------------------------------------------------------
def main() -> None:
    """Compare challenger and champion on the current test set and pick the champion."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI : %s", MLFLOW_TRACKING_URI)
    logger.info("Registered model    : %s", REGISTERED_MODEL_NAME)

    client = MlflowClient()

    # -------- Resolve model versions --------------------------------
    challenger = get_version_by_alias(client, REGISTERED_MODEL_NAME, CHALLENGER_ALIAS)
    if challenger is None:
        raise ValueError("No challenger found. Run 'python -m src.models.register_model' first.")
    champion = get_version_by_alias(client, REGISTERED_MODEL_NAME, CHAMPION_ALIAS)

    # -------- Load test data ----------------------------------------
    X_test, y_test = load_test_data(INPUT_PATH)

    # -------- Evaluate ----------------------------------------------
    challenger_metrics = evaluate_registered_model(challenger, CHALLENGER_ALIAS, X_test, y_test)

    if champion == challenger:
        # Same version (e.g. re-training on unchanged data): just re-check the gate
        logger.info("The challenger (v%s) is already the champion.", challenger)
        winner = "challenger" if passes_quality_gate(challenger_metrics) else None
    else:
        if champion and not same_split(client, REGISTERED_MODEL_NAME, champion):
            logger.info(
                "Champion v%s was trained with a different train/test split: comparing "
                "on this test set would not be fair. The challenger only needs to pass the gate.",
                champion,
            )
            champion_metrics = None
        elif champion:
            champion_metrics = evaluate_registered_model(champion, CHAMPION_ALIAS, X_test, y_test)
        else:
            champion_metrics = None
        winner = choose_champion(challenger_metrics, champion_metrics)

    # -------- Decision ----------------------------------------------
    if winner == "challenger":
        if champion != challenger:
            logger.info("Challenger v%s wins (previous champion: %s).", challenger, champion or "none")
        assign_champion_alias(client, REGISTERED_MODEL_NAME, challenger)

    elif winner == "champion":
        if passes_quality_gate(challenger_metrics):
            logger.info("Challenger v%s is worse than champion v%s — champion kept.", challenger, champion)
        else:
            ci_warning(
                f"Challenger v{challenger} did NOT pass the quality gate — "
                f"champion v{champion} kept. Check the new data/training."
            )

    else:
        ci_warning("No model passes the quality gate on the current data. Nothing will be deployed.")
        # Exit with an error so the CI pipeline stops here and never
        # builds or deploys an image with a model that failed the gate.
        sys.exit(1)

    logger.info(
        "Done. Open the MLflow UI to review the evaluation runs:\n"
        "  mlflow ui --backend-store-uri %s",
        MLFLOW_TRACKING_URI,
    )

# =============================================================================
if __name__ == "__main__":
    main()
