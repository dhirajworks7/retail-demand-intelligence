import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app


class MockPoissonModel:
    """
    Deterministic replacement for the persisted Poisson model.
    """

    def predict(self, X):
        return np.arange(
            1,
            len(X) + 1,
            dtype=float,
        )


class MockClassifierModel:
    """
    Deterministic replacement for the persisted demand classifier.
    """

    def predict_proba(self, X):
        # Use one probability per scored row so the tests can verify
        # both sides of the two-stage threshold.
        probabilities = np.array(
            [0.40, 0.80],
            dtype=float,
        )[:len(X)]

        return np.column_stack(
            [
                1.0 - probabilities,
                probabilities,
            ]
        )


@pytest.fixture
def client():
    """
    Run FastAPI's lifespan for every test so application state is
    initialized in the same way as it is in production.
    """

    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """
    The root endpoint should identify the running API service.
    """

    response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "service": "Retail Demand Intelligence API",
        "status": "running",
    }


def test_health_endpoint_without_model(client):
    """
    The API can be operational even when no persisted model bundle
    exists yet, but health should make model readiness explicit.
    """

    app.state.model_bundle = None

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
        "model_loaded": False,
    }


def test_health_endpoint_with_model(client):
    """
    Health should report model readiness when a bundle is available.
    """

    app.state.model_bundle = {
        "poisson_model": MockPoissonModel(),
        "classifier_model": MockClassifierModel(),
        "model_features": [
            "feature_a",
        ],
        "threshold": 0.50,
    }

    response = client.get(
        "/health"
    )

    assert response.status_code == 200
    assert response.json()[
        "model_loaded"
    ] is True


def test_forecast_endpoint_returns_predictions(client):
    """
    Forecast requests should delegate inference to the production
    prediction pipeline and return one output per submitted record.
    """

    app.state.model_bundle = {
        "poisson_model": MockPoissonModel(),
        "classifier_model": MockClassifierModel(),
        "model_features": [
            "feature_a",
        ],
        "threshold": 0.50,
    }

    response = client.post(
        "/forecast",
        json={
            "records": [
                {
                    "feature_a": 10.0,
                    "is_available": 1,
                },
                {
                    "feature_a": 20.0,
                    "is_available": 1,
                },
            ]
        },
    )

    assert response.status_code == 200

    records = response.json()[
        "records"
    ]

    assert len(records) == 2

    # First row falls below the 0.50 demand-probability threshold.
    assert records[0][
        "poisson_prediction"
    ] == pytest.approx(1.0)

    assert records[0][
        "positive_demand_probability"
    ] == pytest.approx(0.40)

    assert records[0][
        "two_stage_prediction"
    ] == pytest.approx(0.0)

    # Second row exceeds the threshold and retains its Poisson forecast.
    assert records[1][
        "poisson_prediction"
    ] == pytest.approx(2.0)

    assert records[1][
        "positive_demand_probability"
    ] == pytest.approx(0.80)

    assert records[1][
        "two_stage_prediction"
    ] == pytest.approx(2.0)


def test_forecast_endpoint_returns_503_without_model(client):
    """
    Forecasting should fail explicitly when model artifacts have not
    been loaded instead of producing an ungrounded prediction.
    """

    app.state.model_bundle = None

    response = client.post(
        "/forecast",
        json={
            "records": [
                {
                    "feature_a": 10.0,
                    "is_available": 1,
                }
            ]
        },
    )

    assert response.status_code == 503

    assert (
        "model bundle is not available"
        in response.json()["detail"]
    )


def test_forecast_endpoint_rejects_empty_records(client):
    """
    A request containing no item-day records is not forecastable.
    """

    app.state.model_bundle = {
        "poisson_model": MockPoissonModel(),
        "classifier_model": MockClassifierModel(),
        "model_features": [
            "feature_a",
        ],
        "threshold": 0.50,
    }

    response = client.post(
        "/forecast",
        json={
            "records": []
        },
    )

    assert response.status_code == 400


def test_forecast_endpoint_rejects_missing_model_feature(client):
    """
    Feature-contract violations should be exposed as client errors
    rather than generic internal server failures.
    """

    app.state.model_bundle = {
        "poisson_model": MockPoissonModel(),
        "classifier_model": MockClassifierModel(),
        "model_features": [
            "feature_a",
            "feature_b",
        ],
        "threshold": 0.50,
    }

    response = client.post(
        "/forecast",
        json={
            "records": [
                {
                    "feature_a": 10.0,
                    "is_available": 1,
                }
            ]
        },
    )

    assert response.status_code == 422

    assert (
        "missing required model features"
        in response.json()["detail"]
    )