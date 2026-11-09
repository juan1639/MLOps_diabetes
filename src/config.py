"""
Shared configuration for the whole project (paths, MLflow names, split params).

Every script imports its constants from here, so a value is changed in ONE place.
train.py and evaluate_model.py both split the data with src/data/split.py
(stable, hash-based), so the evaluation never uses rows seen during training.
"""

from pathlib import Path

# ===========================================================================
#  Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]

RAW_PATH = BASE_DIR / "data" / "raw" / "diabetes_dataset.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "train_v1.csv"  # SIN feature engineering

# New labelled rows to add to the training data (same columns and SAME SCALE as
# the raw dataset). preprocess.py merges every *.csv found here.
INCOMING_DIR = BASE_DIR / "data" / "incoming"

MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

# ===========================================================================
#  MLflow names
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "diabetes-dataset"
REGISTERED_MODEL_NAME = "diabetes-dataset-model"
CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"     # set by register_model.py, read by evaluate_model.py

# Per-run artifact with the min/max of every feature in the training data.
# The API uses it to warn about inputs outside the range seen in training.
FEATURE_STATS_FILE = "feature_stats.json"

# ===========================================================================
#  Dataset & split
# ---------------------------------------------------------------------------
TARGET = "target"
FEATURE_COLUMNS = ["age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]

TEST_SIZE = 0.2        # approximate share of rows in the test set (see src/data/split.py)
RANDOM_STATE = 42      # seed for the models (RandomForest, GradientBoosting)

# ===========================================================================
#  Quality gate (evaluate_model.py)
# ---------------------------------------------------------------------------
# El target está en su escala original (25–346, desviación típica ≈ 77), así que
# el RMSE se mide en esas unidades (≈ 54 con estos modelos). Un modelo que siempre
# predijera la media tendría RMSE ≈ 77; exigir < 60 obliga a mejorar claramente eso.
RMSE_THRESHOLD = 60.0
R2_THRESHOLD = 0.44     # uso este umbral en vista de los resultados obtenidos
