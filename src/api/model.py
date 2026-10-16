"""
MLflow model loader for the Diabetes dataset prediction API.

Owns the module-level _state dict, the load_champion_model() startup function,
and two helpers (get_model, get_model_info) that endpoints call to access the
loaded model safely.
"""

import logging
import os
from pathlib import Path
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from fastapi import HTTPException

# ===========================================================================
#  Paths & constants — tracking URI and model registry identifiers
#  
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
MLFLOW_TRACKING_URI = BASE_DIR / "mlruns"

MODEL_NAME = "diabetes-dataset-model"
MODEL_ALIAS = "champion"
MODEL_URI = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

# ===========================================================================
#  Logging — module-level logger (basicConfig is configured in main.py)
#  
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ===========================================================================
#  State — holds the loaded model and registry metadata between requests
#  
# ---------------------------------------------------------------------------
_state: dict = {
    "model": None,
    "model_version": None,
    "run_id": None,
}

# ===========================================================================
#  Model loader — populates _state from the MLflow registry at startup
#  
# ---------------------------------------------------------------------------
def load_champion_model() -> None:
    """
    Load the champion model from the MLflow registry into _state.

        - Sets the MLflow tracking URI
        - Loads the pyfunc model
        - And fetches version metadata via MlflowClient
        
    Raises RuntimeError if the alias does not exist.
    """

    # Set env variable 'MLFLOW_TRACKING_URI':
    os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI.as_uri()

    # Use this tracking store:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.as_uri())

    logger.info("Loading model from registry: %s", MODEL_URI)

    try:
        client = MlflowClient()
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)

        # Build the artifact path from known components INSTEAD of following the
        # stored artifact_uri, which is an absolute host path and BREAKS in Docker.
        run = client.get_run(mv.run_id)

        # set a relative path about tracking store:
        model_path = (
            MLFLOW_TRACKING_URI / run.info.experiment_id / mv.run_id / "artifacts" / "model"
        )

        # Set the '_state dict' values:
        _state["model"] = mlflow.pyfunc.load_model(str(model_path))
        _state["model_version"] = str(mv.version)
        _state["run_id"] = mv.run_id

        logger.info("Model loaded — version=%s  run_id=%s", mv.version, mv.run_id)
    
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)

        raise RuntimeError(
            f"Could not load '{MODEL_URI}'. "
            "Run evaluate_model.py first to assign the 'champion' alias."
        ) from exc

# ===========================================================================
#  Helpers — safe accessors for _state used by the API endpoints
#  
# ---------------------------------------------------------------------------
def get_model():
    """Simply return the loaded model, or raise HTTP 503 if startup has not completed."""

    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _state["model"]

# ---------------------------------------------------------------------------
def get_model_info() -> dict:
    """Return registry metadata for the loaded model, or raise HTTP 503 if not ready."""
    
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {
        "model_name": MODEL_NAME,
        "version": _state["model_version"],
        "alias": MODEL_ALIAS,
        "run_id": _state["run_id"],
    }




