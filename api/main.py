from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src.models.persistence import (
    DEFAULT_MODEL_DIR,
    load_model_bundle,
)
from src.models.predict import generate_forecasts


# Keep the loaded model bundle in application state so the models are
# loaded once at startup rather than from disk for every API request.
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model_bundle = load_model_bundle(
            DEFAULT_MODEL_DIR
        )
    except FileNotFoundError:
        # Development and automated tests may run before a real trained
        # model bundle has been generated. The health endpoint can still
        # report that the API itself is running.
        app.state.model_bundle = None

    yield


app = FastAPI(
    title="Retail Demand Intelligence API",
    description=(
        "Inference API for availability-aware retail demand forecasting."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ForecastRequest(BaseModel):
    """
    One or more fully engineered item-day records for forecasting.

    The feature dictionary remains flexible because the authoritative
    feature contract is stored with the persisted model bundle.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    records: list[dict[str, object]]


class ForecastResponse(BaseModel):
    """
    Forecast outputs returned for each submitted item-day record.
    """

    records: list[dict[str, object]]


@app.get("/")
def root() -> dict[str, str]:
    """
    Provide a lightweight service discovery endpoint.
    """

    return {
        "service": "Retail Demand Intelligence API",
        "status": "running",
    }


@app.get("/health")
def health() -> dict[str, object]:
    """
    Report API health and whether a persisted model bundle is loaded.
    """

    model_loaded = (
        app.state.model_bundle
        is not None
    )

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
    }


@app.post(
    "/forecast",
    response_model=ForecastResponse,
)
def forecast(
    request: ForecastRequest,
) -> ForecastResponse:
    """
    Generate demand forecasts for fully engineered item-day records.

    Feature validation and availability-aware inference are delegated to
    the reusable production inference layer rather than duplicated here.
    """

    bundle = app.state.model_bundle

    if bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast model bundle is not available. "
                "Train and persist the models before requesting forecasts."
            ),
        )

    if not request.records:
        raise HTTPException(
            status_code=400,
            detail="At least one forecast record is required.",
        )

    inference_df = pd.DataFrame(
        request.records
    )

    try:
        forecasts = generate_forecasts(
            df=inference_df,
            poisson_model=bundle["poisson_model"],
            classifier_model=bundle["classifier_model"],
            model_features=bundle["model_features"],
            threshold=bundle["threshold"],
        )
    except ValueError as exc:
        # Invalid feature contracts or business-rule inputs are client
        # errors rather than internal API failures.
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    # Convert only the prediction outputs to JSON-friendly records.
    # Input feature columns do not need to be echoed back to the caller.
    output_columns = [
        "poisson_prediction",
        "positive_demand_probability",
        "two_stage_prediction",
    ]

    response_records = forecasts[
        output_columns
    ].to_dict(
        orient="records"
    )

    return ForecastResponse(
        records=response_records
    )