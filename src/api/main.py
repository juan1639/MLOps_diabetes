"""
FastAPI application that serves predictions from the MLflow champion model.

Loads the model registered under 'diabetes-dataset-model@champion' at startup
and exposes three endpoints:

    - a health check
    - model metadata
    - and a predict route.
"""
"""
Model Serving is the step that makes a trained model available to other systems.
Instead of running a script manually, we expose the model behind an HTTP API so
that any client (a web app, a pipeline, a notebook) can request predictions at any
time without knowing anything about scikit-learn or MLflow internals.

1. Load once at startup — loading a model is expensive; we do it once via FastAPI's
    lifespan context so every request hits an already-warm model in memory.

2. Pydantic validation — the input schema is declared explicitly; FastAPI rejects
    malformed requests automatically before they reach the model.

3. Alias-based loading — we load 'models:/diabetes-dataset-model@champion' instead
    of a hard-coded version number, so promoting a new champion in the registry
    is enough to update what the API serves (requires a restart).
"""

import logging
from contextlib import asynccontextmanager
import pandas as pd
from fastapi import FastAPI, HTTPException

from .model import (
    MODEL_NAME,
    _state,
    get_model,
    get_model_info,
    load_champion_model
)

from .schemas import FEATURE_COLUMNS, DiabetesFeatures, PredictionResponse

# ===========================================================================
#  Logging —> configure once for the whole application at entry point
#  
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

logger = logging.getLogger(__name__)

# ===========================================================================
#                            (Funcion asincrona)
#  
#  Lifespan —> startup/shutdown hook; delegates model loading to model.py
#  
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Call load_champion_model() before the app starts accepting requests.

    The model is loaded only once to handle requests—specifically,
    during the initial model load—rather than every time a request is made.
    
    The shutdown occurs once the session ends (after the `yield`).
    """
    load_champion_model()

    yield   # application runs here
    logger.info("Shutting down - releasing model from memory.")
    _state["model"] = None

# ===========================================================================
#  App & endpoints — FastAPI instance and the three API routes
#  
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Diabetes dataset API",
    description="Serves predictions from the MLflow champion model.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Return service liveness and the registered model name."""
    return {"status": "ok", "model": MODEL_NAME}

# ---------------------------------------------------------------------------
@app.get("/model-info")
def model_info() -> dict:
    """Return metadata about the loaded model version."""
    return get_model_info()

# ---------------------------------------------------------------------------
@app.post("/predict")
def predict(features: DiabetesFeatures) -> PredictionResponse:
    """Return a pydantic object:
        - prediction
        - model version

    Args:
        features: The 10 feature values

    Returns:
        PredictionResponse object:
            - prediction
            - model version
    """

    model = get_model()

    # Converts pydantic object into a DataFrame (predict needs a DataFrame param)
    # ( pydantic-object --> dict --> DataFrame )
    input_df = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)

    try:
        prediction = model.predict(input_df)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction error.") from exc

    result = float(prediction[0])
    logger.info("Prediction: %.4f", result)

    # And return a pydantic object:
    return PredictionResponse(prediction=result, model_version=_state["model_version"])






