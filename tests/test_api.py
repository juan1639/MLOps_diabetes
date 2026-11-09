"""Tests for the FastAPI app, using a fake model instead of the real MLflow one."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.api.main as main
from src.api.model import MODEL_NAME, _state
from src.config import FEATURE_COLUMNS

VALID_BODY = {col: 0.01 for col in FEATURE_COLUMNS}


class FakeModel:
    """Returns the sum of the features, and remembers the input it got."""

    def __init__(self):
        self.last_input = None

    def predict(self, df):
        self.last_input = df
        return np.array([df.to_numpy().sum()])


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()

    def fake_load():
        _state.update(model=model, model_version="7", run_id="abc123")

    # main.py calls load_champion_model() in the lifespan hook at startup
    monkeypatch.setattr(main, "load_champion_model", fake_load)
    return model


@pytest.fixture
def client(fake_model):
    with TestClient(main.app) as c:   # "with" runs the lifespan (startup/shutdown)
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": MODEL_NAME}


def test_model_info(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": MODEL_NAME,
        "version": "7",
        "alias": "champion",
        "run_id": "abc123",
    }


def test_predict(client, fake_model):
    response = client.post("/predict", json=VALID_BODY)

    assert response.status_code == 200
    assert response.json() == {"prediction": pytest.approx(0.1), "model_version": "7"}
    # The model must receive the columns in the same order used in training
    assert list(fake_model.last_input.columns) == FEATURE_COLUMNS


def test_predict_rejects_missing_feature(client):
    body = {k: v for k, v in VALID_BODY.items() if k != "bmi"}
    assert client.post("/predict", json=body).status_code == 422


def test_predict_rejects_non_numeric_value(client):
    assert client.post("/predict", json={**VALID_BODY, "age": "abc"}).status_code == 422


def test_predict_returns_500_if_model_fails(client, fake_model, monkeypatch):
    def broken_predict(df):
        raise ValueError("boom")

    monkeypatch.setattr(fake_model, "predict", broken_predict)
    assert client.post("/predict", json=VALID_BODY).status_code == 500


def test_endpoints_return_503_if_model_not_loaded(monkeypatch):
    # Without the "with" block the lifespan does not run, so no model is loaded
    monkeypatch.setitem(_state, "model", None)
    c = TestClient(main.app)
    assert c.get("/model-info").status_code == 503
    assert c.post("/predict", json=VALID_BODY).status_code == 503
