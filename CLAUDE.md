# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

End-to-end MLOps pipeline on the sklearn diabetes regression dataset: data ingestion → validation → preprocessing → MLflow training/registry/evaluation → FastAPI serving → Docker image → auto-deploy to Render via GitHub Actions. Python 3.11, dependencies pinned exactly in `requirements.txt` (Dependabot bumps them; major updates of `mlflow` and `pandas` are ignored). Comments and docs mix Spanish and English.

## Commands

All scripts are run as modules from the repository root (`python -m src.<folder>.<script>`), because they import the shared `src/config.py`. Each step depends on the previous one's output:

```bash
pip install -r requirements-dev.txt   # runtime deps + pytest + ruff

python -m src.data.load_data         # → data/raw/diabetes_dataset.csv
python -m src.data.validate_data     # assert-based checks on raw data
python -m src.data.preprocess        # → data/processed/train_v1.csv
python -m src.models.train           # 3 runs (LinearRegression, RandomForest, GradientBoosting) → mlruns/
python -m src.models.register_model  # registers lowest-RMSE run as "diabetes-dataset-model"
python -m src.models.evaluate_model  # quality gate; sets @champion alias or exits 1
python -m src.models.prune_artifacts # deletes artifacts/model of unregistered training runs

uvicorn src.api.main:app --reload    # http://localhost:8000/docs
mlflow ui --backend-store-uri mlruns/

docker build -t mlops_diabetes .     # requires a local mlruns/ containing a champion model
docker run -p 8000:8000 mlops_diabetes
```

Tests and lint (config in `pyproject.toml`; both run in CI before the pipeline):

```bash
pytest                                   # all tests
pytest tests/test_api.py::test_predict   # a single test
ruff check .
```

`tests/test_api.py` replaces `load_champion_model` with a fake model, and `tests/test_register_model.py` uses a temporary MLflow store, so tests never need `mlruns/`.

## Architecture

- **Tracking store is the file-based `mlruns/`** at the repo root (gitignored), set explicitly from `config.MLFLOW_TRACKING_URI` in each model script and in the API. `src/models/train_db.py` is an experimental SQLite-backed (`mlflow.db`) copy of `train.py`; it is not used by CI, register, evaluate, or the API, so don't mix the two stores.
- **Shared constants live in `src/config.py`** (paths, MLflow names/aliases, `FEATURE_COLUMNS`, `TEST_SIZE`/`RANDOM_STATE`, quality-gate thresholds). `train_db.py` still defines its own.
- **Data**: `preprocess.py` merges `data/raw/diabetes_dataset.csv` with every `data/incoming/*.csv` (name order); `validate_data.py` checks incoming files have the same columns and scale as raw. The train/test split is the stable hash-based `src/data/split.py` (shared by `train.py` and `evaluate_model.py`), so adding rows never moves old rows between train and test — don't replace it with `train_test_split`, and bump `SPLIT_METHOD` if the split logic changes.
- **Training-run tags** (`train.py`): `stage=training`, `dataset_hash` (`src/data/versioning.py`), `split_method`; artifacts `model/` and `feature_stats.json`.
- **Model selection flow (champion/challenger)**: `register_model.py` picks the FINISHED `stage=training` run with the current `dataset_hash` and lowest `metrics.rmse` (ties → oldest run, so re-training on unchanged data reuses the same version) and sets alias `challenger`. `evaluate_model.py` runs `mlflow.evaluate()` on the current test set for `@challenger` and `@champion`; `choose_champion()` promotes the challenger if it passes RMSE < 60 / R² > 0.44 and its RMSE is <= the champion's. A champion trained with another `split_method` is not compared (only thresholds apply). Exits 1 only if no model passes the gate, which stops CI before any image is built.
- **API (`src/api/`)**: `main.py` loads the model once in the FastAPI `lifespan` hook; `model.py` owns a module-level `_state` dict and resolves `@champion` via `MlflowClient`, but builds the artifact path manually (`mlruns/<exp_id>/<run_id>/artifacts/model`) instead of using the stored absolute `artifact_uri`, because that host path breaks inside Docker. `schemas.py` holds the Pydantic request/response models; `main.py` builds the input DataFrame with `config.FEATURE_COLUMNS` (order must match `train_v1.csv` minus `target`). Endpoints: `/health`, `/model-info`, `/predict`. `monitoring.py` logs each prediction as a JSON line and warns when an input is outside the champion's `feature_stats.json` range (skipped if the artifact is missing). Inputs arrive already scaled by a separate intermediate backend, so don't add range validation that rejects requests.
- **CI/CD (`.github/workflows/ci.yml`)**: on push/PR to `master` runs ruff + pytest, restores `mlruns/` from the currently deployed `:latest` image (`docker create` + `docker cp`; empty store if none), runs the whole pipeline, then builds the Docker image. `mlruns/` is not in git, which is why the image is built in the same job (`COPY mlruns/` in the Dockerfile) and the store's absolute runner paths stay valid across runs. After evaluation, `prune_artifacts.py` deletes `artifacts/model/` of training runs with no registered version (run metadata kept, tag `model_artifact=pruned`), so the store grows only ~0.15 MB per run; it locates files under `mlruns/<exp_id>/<run_id>/` like the API does. On push (not PRs) it pushes `ghcr.io/juan1639/mlops_diabetes:{latest,<sha>}` and POSTs to the `RENDER_DEPLOY_HOOK` secret. The container runs as a non-root user and listens on `${PORT:-8000}`.
- `data/processed/train_v1.csv` deliberately has no feature engineering (the notebook's feature engineering didn't improve metrics).
