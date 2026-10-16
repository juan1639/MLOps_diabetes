"""
****************************************************************************************
            ----- POR QUE HE CREADO ESTE ARCHIVO train_db.py -----

Las versiones actuales de MLflow han puesto ese backend de archivos en maintenance mode
y ya no permiten utilizarlo por defecto para nuevas operaciones.

MLflow recomienda utilizar una base de datos,
y para desarrollo local la opción más sencilla es SQLite.

En el archivo .github/workflows/ci.yml ... tendremos que especificar si usar
los archivos tipo train_db.py etc... o los tipo train.py

****************************************************************************************
"""
"""
This module trains three regression models on the California Housing dataset
and tracks every run with MLflow Experiment Tracking.

For each model the script logs hyperparameters, evaluation metrics and the
serialised model artifact so runs can be compared in the MLflow UI.

- MLflow records every run automatically:
    - The code version, hyperparameters, metrics and the model artifact itself.

- (After running this script, open `mlflow ui` and compare both runs side-by-side).
"""
"""
=============================================================================
- mlflow.set_tracking_uri()
- mlflow.set_experiment()

- load_features()       --> cargar 'data/processed/train_v1.csv'
- train_test_split()

- run_experiment(
        model,
        model_type,
        params,
        X_train, X_test,
        y_train, y_test
    )

=============================================================================
"""

import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ===========================================================================
#  Paths & constants
#  
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"
#MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

# ---- sqlite tracking-uri ------------------------------------
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"
# -------------------------------------------------------------

EXPERIMENT_NAME = "diabetes-dataset"
TARGET = "target"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===========================================================================
#  Logging (creamos un logger asociado a este módulo)
#  
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

logger = logging.getLogger(__name__)

# ===========================================================================
#  Data
#  
# ---------------------------------------------------------------------------
def load_features(path: Path) -> pd.DataFrame:
    """Load the engineered feature dataset from CSV.

    Args:
        path: Path to train_v1.csv.

    Returns:
        DataFrame with all features and target column.

    Raises:
        FileNotFoundError: If the input file does not exist.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}\n"
            "Run the data pipeline first: python src/data/feature_engineering.py"
        )

    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns from %s", len(df), df.shape[1], path)
    return df

# ===========================================================================
def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split DataFrame into train/test sets.

    Args:
        df: Full feature DataFrame including the target column.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    logger.info("Train size: %d | Test size: %d", len(X_train), len(X_test))

    return X_train, X_test, y_train, y_test

# ===========================================================================
#  Metrics
#  
# ---------------------------------------------------------------------------
def compute_metrics(y_true: pd.Series, y_pred) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions.

    Returns:
        Dictionary with keys ``rmse``, ``mae``, ``r2``.
    """

    return {
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

# ===========================================================================
#  MLflow run
#  
# ---------------------------------------------------------------------------
def run_experiment(model, model_type: str, params: dict,
                   X_train: pd.DataFrame, X_test: pd.DataFrame,
                   y_train: pd.Series, y_test: pd.Series) -> None:
    """Train a model and log everything to a single MLflow run.

    The run name follows the convention ``{model_type}_{YYYYMMDD_HHMMSS}``
    so runs are identifiable at a glance in the UI.

    Tags logged:
        - ``model_type``: class name of the estimator
        - ``dataset``: source file name
        - ``stage``: always ``"training"`` for this script

    Args:
        model: An unfitted scikit-learn estimator.
        model_type: Short label used in the run name and ``model_type`` tag.
        params: Hyperparameter dict to log with ``mlflow.log_params``.
        X_train: Training feature matrix.
        X_test: Test feature matrix.
        y_train: Training target series.
        y_test: Test target series.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        # ---------- Tags -------------------------------------------
        mlflow.set_tags({
            "model_type": model_type,
            "dataset": INPUT_PATH.name,
            "stage": "training",
        })

        # ---------- Params -----------------------------------------
        mlflow.log_params(params)

        # ---------- Training ---------------------------------------
        logger.info("[%s] Training...", model_type)
        model.fit(X_train, y_train)

        # ---------- Metrics ----------------------------------------
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        mlflow.log_metrics(metrics)

        logger.info(
            "[%s] rmse=%.4f  mae=%.4f  r2=%.4f",
            model_type,
            metrics["rmse"],
            metrics["mae"],
            metrics["r2"],
        )

        # ---------- Artifact ---------------------------------------
        # registered_model_name=None: registry is handled by register_model.py
        mlflow.sklearn.log_model(
            sk_model=model,
            #artifact_path="model", # deprecated
            name="model",
            registered_model_name=None,
            input_example=X_train.head(5),   
        )

        logger.info("[%s] Model artifact logged.", model_type)

# ===========================================================================
#  Main
#  
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full experiment tracking pipeline for both models."""

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow tracking URI : %s", MLFLOW_TRACKING_URI)
    logger.info("Experiment          : %s", EXPERIMENT_NAME)

    df = load_features(INPUT_PATH)  # data/processed/train_v1.csv
    X_train, X_test, y_train, y_test = split_data(df)

    # --- Model 1: LinearRegression -----------------------------------------
    # Baseline model with no hyperparameters — quick to fit and easy to interpret.
    run_experiment(
        model=LinearRegression(),
        model_type="LinearRegression",
        params={"fit_intercept": True},
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # --- Model 2: RandomForestRegressor ------------------------------------
    # Ensemble model that typically outperforms the linear baseline on this
    # dataset due to non-linear interactions between features.
    run_experiment(
        model=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        model_type="RandomForestRegressor",
        params={"n_estimators": 100, "random_state": RANDOM_STATE},
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # --- Model 3: GradientBoostingRegressor ------------------------------------
    run_experiment(
        model=GradientBoostingRegressor(random_state=RANDOM_STATE),
        model_type="GradientBoostingRegressor",
        params={"random_state": RANDOM_STATE},
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    logger.info(
        "Done. Open the MLflow UI to compare runs:\n"
        "  mlflow ui --backend-store-uri %s",
        MLFLOW_TRACKING_URI,
    )

# =========================================================================
if __name__ == "__main__":
    main()





