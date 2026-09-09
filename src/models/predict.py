from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.forecast_models import (
    apply_availability_rule,
    predict_poisson_demand,
    predict_positive_demand_probability,
    two_stage_forecast,
)
from src.models.persistence import (
    DEFAULT_MODEL_DIR,
    load_model_bundle,
)


def validate_inference_features(
    df: pd.DataFrame,
    model_features: list[str],
) -> None:
    """
    Validate that inference data contains the complete feature contract
    expected by the persisted models.

    Extra columns are allowed because production datasets may also
    contain identifiers, dates, targets, or other business metadata.
    """

    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Inference data is missing required model features: "
            + ", ".join(missing_features)
        )


def prepare_inference_matrix(
    df: pd.DataFrame,
    model_features: list[str],
) -> pd.DataFrame:
    """
    Select model features in the exact order used during training.

    Preserving feature order is important because the persisted models
    expect the same feature contract used when they were fitted.
    """

    validate_inference_features(
        df,
        model_features,
    )

    return df.loc[
        :,
        model_features,
    ].copy()


def generate_forecasts(
    df: pd.DataFrame,
    poisson_model,
    classifier_model,
    model_features: list[str],
    threshold: float,
    availability_column: str = "is_available",
) -> pd.DataFrame:
    """
    Generate Poisson and two-stage forecasts for inference data.

    Available rows are scored by the ML models. Unavailable rows bypass
    ML and receive deterministic zero forecasts.

    The returned dataframe preserves the original input rows and adds:
        - poisson_prediction
        - positive_demand_probability
        - two_stage_prediction
    """

    if availability_column not in df.columns:
        raise ValueError(
            "Inference data is missing the availability column: "
            f"{availability_column}"
        )

    if df.empty:
        raise ValueError(
            "Inference dataframe is empty."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    forecasts = df.copy()

    # Initialize all outputs to zero. This naturally implements the
    # deterministic forecast for unavailable product-days.
    forecasts["poisson_prediction"] = 0.0
    forecasts["positive_demand_probability"] = 0.0
    forecasts["two_stage_prediction"] = 0.0

    available_mask = (
        forecasts[availability_column] == 1
    )

    available_df = forecasts.loc[
        available_mask
    ]

    # If every requested product-day is unavailable, no model call is
    # necessary because all forecasts are deterministically zero.
    if available_df.empty:
        return forecasts

    X = prepare_inference_matrix(
        available_df,
        model_features,
    )

    poisson_prediction = predict_poisson_demand(
        poisson_model,
        X,
    )

    positive_probability = (
        predict_positive_demand_probability(
            classifier_model,
            X,
        )
    )

    staged_prediction = two_stage_forecast(
        positive_probability,
        poisson_prediction,
        threshold=threshold,
    )

    available_index = available_df.index

    # Assign predictions back to their original rows so identifiers,
    # dates, and any other metadata remain aligned with each forecast.
    forecasts.loc[
        available_index,
        "poisson_prediction",
    ] = poisson_prediction

    forecasts.loc[
        available_index,
        "positive_demand_probability",
    ] = positive_probability

    forecasts.loc[
        available_index,
        "two_stage_prediction",
    ] = staged_prediction

    # Apply the business rule explicitly as a final safeguard. Even if
    # this function changes later, unavailable rows must forecast zero.
    forecasts["poisson_prediction"] = apply_availability_rule(
        forecasts["poisson_prediction"],
        forecasts[availability_column],
    )

    forecasts["two_stage_prediction"] = apply_availability_rule(
        forecasts["two_stage_prediction"],
        forecasts[availability_column],
    )

    return forecasts


def load_and_generate_forecasts(
    df: pd.DataFrame,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    availability_column: str = "is_available",
) -> pd.DataFrame:
    """
    Load a persisted model bundle and generate forecasts.

    This is the main inference entry point intended for later use by
    applications such as the forecasting API.
    """

    bundle = load_model_bundle(
        model_dir
    )

    return generate_forecasts(
        df=df,
        poisson_model=bundle["poisson_model"],
        classifier_model=bundle["classifier_model"],
        model_features=bundle["model_features"],
        threshold=bundle["threshold"],
        availability_column=availability_column,
    )