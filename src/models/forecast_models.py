# ---------------------------------------------------------
# Forecasting Models
# ---------------------------------------------------------
# This module contains reusable forecasting logic for the
# Retail Demand Intelligence project.
#
# The modeling strategy was selected using chronological
# validation data. The untouched test set must not be used
# for further model or threshold tuning.

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

FORECAST_HORIZON = 28

# This threshold was selected using validation data only.
# It must remain fixed when evaluating the final holdout set.
DEFAULT_DEMAND_THRESHOLD = 0.50


# ---------------------------------------------------------
# Seasonal-naive baseline
# ---------------------------------------------------------

def seasonal_naive_forecast(
    df: pd.DataFrame,
    lag_column: str = "sales_lag_28",
    availability_column: str = "is_available",
) -> np.ndarray:
    """
    Generate the project's 28-day seasonal-naive forecast.

    The baseline predicts demand using sales observed exactly
    28 days earlier.

    When an item is unavailable, the business rule overrides
    the historical forecast and predicts zero demand.

    Parameters
    ----------
    df :
        Feature dataset containing the lag and availability
        columns.

    lag_column :
        Column containing historical demand from the seasonal
        lag. Defaults to ``sales_lag_28``.

    availability_column :
        Binary column where 1 means the product is available
        and 0 means it is unavailable.

    Returns
    -------
    np.ndarray
        Non-negative baseline forecasts.
    """

    _validate_required_columns(
        df,
        {
            lag_column,
            availability_column,
        },
    )

    prediction = (
        df[lag_column]
        .to_numpy(dtype="float64")
        .copy()
    )

    availability = (
        df[availability_column]
        .to_numpy()
    )

    # Missing lag values can occur in early history. A neutral
    # zero forecast is safer than propagating NaN predictions.
    prediction = np.nan_to_num(
        prediction,
        nan=0.0,
    )

    # Demand cannot occur when the product is unavailable.
    prediction[availability == 0] = 0.0

    # Demand forecasts should never be negative.
    return np.clip(
        prediction,
        a_min=0.0,
        a_max=None,
    )


# ---------------------------------------------------------
# LightGBM model constructors
# ---------------------------------------------------------

def create_poisson_regressor(
    random_state: int = 42,
) -> lgb.LGBMRegressor:
    """
    Create the LightGBM Poisson demand regressor.

    The configuration mirrors the validated notebook model.

    Parameters
    ----------
    random_state :
        Random seed used for reproducibility.

    Returns
    -------
    lightgbm.LGBMRegressor
        Unfitted Poisson regression model.
    """

    return lgb.LGBMRegressor(
        objective="poisson",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,

        # Row subsampling is enabled explicitly using
        # subsample_freq=1. This avoids relying on LightGBM's
        # default bagging frequency.
        subsample=0.8,
        subsample_freq=1,

        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


def create_demand_classifier(
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """
    Create the binary LightGBM demand-occurrence classifier.

    The classifier predicts whether daily demand is positive:

        1 -> sales > 0
        0 -> sales = 0

    Parameters
    ----------
    random_state :
        Random seed used for reproducibility.

    Returns
    -------
    lightgbm.LGBMClassifier
        Unfitted binary classifier.
    """

    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )


# ---------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------

def predict_poisson_demand(
    model: lgb.LGBMRegressor,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Generate non-negative demand forecasts from a fitted
    Poisson LightGBM model.

    Parameters
    ----------
    model :
        Fitted LightGBM Poisson regressor.

    features :
        Model feature matrix.

    Returns
    -------
    np.ndarray
        Non-negative demand predictions.
    """

    prediction = np.asarray(
        model.predict(features),
        dtype="float64",
    )

    return np.clip(
        prediction,
        a_min=0.0,
        a_max=None,
    )


def predict_positive_demand_probability(
    model: lgb.LGBMClassifier,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Predict the probability that demand is greater than zero.

    Parameters
    ----------
    model :
        Fitted binary demand classifier.

    features :
        Model feature matrix.

    Returns
    -------
    np.ndarray
        Probability of positive demand for each observation.
    """

    probabilities = np.asarray(
        model.predict_proba(features),
        dtype="float64",
    )

    # LightGBM binary classifiers return two columns:
    # column 0 = probability of class 0
    # column 1 = probability of class 1.
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 2
    ):
        raise ValueError(
            "Binary classifier predict_proba() must return "
            "an array with exactly two probability columns."
        )

    return probabilities[:, 1]


def two_stage_forecast(
    positive_demand_probability,
    demand_prediction,
    threshold: float = DEFAULT_DEMAND_THRESHOLD,
) -> np.ndarray:
    """
    Combine binary demand occurrence and demand magnitude.

    The two-stage model works as follows:

        P(sales > 0) < threshold
            -> forecast 0

        P(sales > 0) >= threshold
            -> retain Poisson demand forecast

    Parameters
    ----------
    positive_demand_probability :
        Probability that demand is positive.

    demand_prediction :
        Demand magnitude forecast from the Poisson regressor.

    threshold :
        Classification threshold. The project's selected
        validation threshold is 0.50.

    Returns
    -------
    np.ndarray
        Final two-stage demand forecasts.
    """

    probabilities = np.asarray(
        positive_demand_probability,
        dtype="float64",
    )

    prediction = np.asarray(
        demand_prediction,
        dtype="float64",
    )

    if probabilities.shape != prediction.shape:
        raise ValueError(
            "Demand probabilities and demand predictions "
            "must have the same shape."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    # Ensure the magnitude component cannot introduce negative
    # demand forecasts.
    prediction = np.clip(
        prediction,
        a_min=0.0,
        a_max=None,
    )

    final_prediction = np.where(
        probabilities >= threshold,
        prediction,
        0.0,
    )

    return final_prediction.astype(
        "float64"
    )


def apply_availability_rule(
    prediction,
    availability,
) -> np.ndarray:
    """
    Force forecasts to zero when a product is unavailable.

    Availability is treated as a deterministic business rule
    rather than as something the ML model must learn.

    Parameters
    ----------
    prediction :
        Forecast demand values.

    availability :
        Binary availability indicator.

    Returns
    -------
    np.ndarray
        Forecasts after applying the availability rule.
    """

    prediction = np.asarray(
        prediction,
        dtype="float64",
    ).copy()

    availability = np.asarray(
        availability,
    )

    if prediction.shape != availability.shape:
        raise ValueError(
            "Prediction and availability arrays must have "
            "the same shape."
        )

    prediction[availability == 0] = 0.0

    return np.clip(
        prediction,
        a_min=0.0,
        a_max=None,
    )


# ---------------------------------------------------------
# Training-target helper
# ---------------------------------------------------------

def create_positive_demand_target(
    sales,
) -> np.ndarray:
    """
    Convert numeric demand into the binary classifier target.

    Parameters
    ----------
    sales :
        Observed demand values.

    Returns
    -------
    np.ndarray
        Binary target where:

        1 = positive demand
        0 = zero demand
    """

    sales = np.asarray(
        sales,
        dtype="float64",
    )

    if np.any(sales < 0):
        raise ValueError(
            "Sales values cannot be negative."
        )

    return (
        sales > 0
    ).astype(
        "uint8"
    )


# ---------------------------------------------------------
# Internal validation
# ---------------------------------------------------------

def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """
    Verify that required DataFrame columns are present.
    """

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        missing_list = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Input DataFrame is missing required columns: "
            f"{missing_list}"
        )