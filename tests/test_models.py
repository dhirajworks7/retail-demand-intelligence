# ---------------------------------------------------------
# Tests for forecasting model utilities
# ---------------------------------------------------------
# These tests verify deterministic model behavior without
# fitting expensive LightGBM models.

import numpy as np
import pandas as pd
import pytest

from src.models.forecast_models import (
    seasonal_naive_forecast,
    two_stage_forecast,
    apply_availability_rule,
    create_positive_demand_target,
    create_poisson_regressor,
    create_demand_classifier,
)


def test_seasonal_naive_forecast():
    """
    Seasonal naive should use the 28-day lag and force
    unavailable products to zero demand.
    """

    df = pd.DataFrame({
        "sales_lag_28": [5.0, 3.0, 0.0, 8.0],
        "is_available": [1, 0, 1, 1],
    })

    result = seasonal_naive_forecast(
        df
    )

    expected = np.array([
        5.0,
        0.0,
        0.0,
        8.0,
    ])

    assert np.allclose(
        result,
        expected,
    )


def test_seasonal_naive_missing_lag_becomes_zero():
    """
    Missing historical demand should not propagate NaN
    predictions into downstream evaluation.
    """

    df = pd.DataFrame({
        "sales_lag_28": [np.nan, 4.0],
        "is_available": [1, 1],
    })

    result = seasonal_naive_forecast(
        df
    )

    assert np.allclose(
        result,
        [0.0, 4.0],
    )


def test_create_positive_demand_target():
    """
    Positive sales should map to class 1 and zero sales
    should map to class 0.
    """

    sales = [0, 3, 0, 7]

    result = create_positive_demand_target(
        sales
    )

    assert np.array_equal(
        result,
        np.array(
            [0, 1, 0, 1],
            dtype="uint8",
        ),
    )


def test_negative_sales_raise_error():
    """
    Negative unit sales are invalid for this demand task.
    """

    sales = [0, 2, -1]

    with pytest.raises(
        ValueError,
        match="negative",
    ):
        create_positive_demand_target(
            sales
        )


def test_two_stage_forecast():
    """
    Probabilities below the threshold should be gated to zero.
    Probabilities at or above the threshold should retain the
    demand magnitude prediction.
    """

    probabilities = np.array([
        0.20,
        0.50,
        0.80,
        0.49,
    ])

    demand_prediction = np.array([
        2.0,
        3.0,
        4.0,
        5.0,
    ])

    result = two_stage_forecast(
        probabilities,
        demand_prediction,
        threshold=0.50,
    )

    expected = np.array([
        0.0,
        3.0,
        4.0,
        0.0,
    ])

    assert np.allclose(
        result,
        expected,
    )


def test_two_stage_threshold_boundary():
    """
    A probability exactly equal to the selected threshold
    should be treated as positive demand.
    """

    result = two_stage_forecast(
        [0.50],
        [6.0],
        threshold=0.50,
    )

    assert result[0] == pytest.approx(
        6.0
    )


def test_invalid_two_stage_threshold_raises_error():
    """
    Classification thresholds must remain between 0 and 1.
    """

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        two_stage_forecast(
            [0.5],
            [2.0],
            threshold=1.5,
        )


def test_two_stage_shape_mismatch_raises_error():
    """
    Probability and magnitude arrays must align observation
    by observation.
    """

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        two_stage_forecast(
            [0.4, 0.7],
            [3.0],
        )


def test_apply_availability_rule():
    """
    Unavailable products must always receive zero forecast.
    """

    prediction = np.array([
        2.0,
        3.0,
        4.0,
        5.0,
    ])

    availability = np.array([
        1,
        0,
        1,
        0,
    ])

    result = apply_availability_rule(
        prediction,
        availability,
    )

    expected = np.array([
        2.0,
        0.0,
        4.0,
        0.0,
    ])

    assert np.allclose(
        result,
        expected,
    )


def test_apply_availability_shape_mismatch():
    """
    Availability and prediction arrays must align exactly.
    """

    with pytest.raises(
        ValueError,
        match="same shape",
    ):
        apply_availability_rule(
            [1.0, 2.0],
            [1],
        )


def test_poisson_regressor_configuration():
    """
    The production Poisson constructor should expose the
    intended LightGBM configuration.
    """

    model = create_poisson_regressor(
        random_state=42
    )

    params = model.get_params()

    assert params["objective"] == "poisson"
    assert params["n_estimators"] == 1000
    assert params["subsample"] == pytest.approx(
        0.8
    )
    assert params["subsample_freq"] == 1
    assert params["random_state"] == 42


def test_classifier_configuration():
    """
    The production classifier constructor should preserve the
    validated binary demand-occurrence configuration.
    """

    model = create_demand_classifier(
        random_state=42
    )

    params = model.get_params()

    assert params["objective"] == "binary"
    assert params["n_estimators"] == 1000
    assert params["subsample"] == pytest.approx(
        0.8
    )
    assert params["subsample_freq"] == 1
    assert params["random_state"] == 42