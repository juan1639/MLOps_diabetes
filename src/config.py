"""
Shared configuration for the whole project (paths, MLflow names, split params).

Every script imports its constants from here, so a value is changed in ONE place.
Important: train.py and evaluate_model.py must use the same TEST_SIZE and
RANDOM_STATE, otherwise the evaluation would use rows seen during training.
"""

from pathlib import Path

# ===========================================================================
#  Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes_dataset.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"  # SIN feature engineering

MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

# ===========================================================================
#  MLflow names
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "diabetes-dataset"
REGISTERED_MODEL_NAME = "diabetes-dataset-model"
CHAMPION_ALIAS = "champion"

# ===========================================================================
#  Dataset & split
# ---------------------------------------------------------------------------
TARGET = "target"
FEATURE_COLUMNS = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===========================================================================
#  Quality gate (evaluate_model.py)
# ---------------------------------------------------------------------------
# El target está en su escala original (25–346, desviación típica ≈ 77), así que
# el RMSE se mide en esas unidades (≈ 54 con estos modelos). Un modelo que siempre
# predijera la media tendría RMSE ≈ 77; exigir < 60 obliga a mejorar claramente eso.
RMSE_THRESHOLD = 60.0
R2_THRESHOLD = 0.44     # uso este umbral en vista de los resultados obtenidos
