# ---------------------------------------------------------
# Tests for forecast evaluation metrics
# ---------------------------------------------------------
# These tests verify that the reusable metric functions in
# src/evaluation/metrics.py return mathematically correct
# results for known examples.

import numpy as np
import pytest

from src.evaluation.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    weighted_mean_absolute_percentage_error,
    forecast_bias_percentage,
    calculate_forecast_metrics,
)


def test_mean_absolute_error():
    """
    MAE should equal the average absolute forecast error.
    """

    actual = [10, 0, 5, 8]
    prediction = [8, 1, 6, 8]

    result = mean_absolute_error(
        actual,
        prediction,
    )

    assert result == pytest.approx(
        1.0
    )


def test_root_mean_squared_error():
    """
    RMSE should correctly penalize larger errors.
    """

    actual = [10, 0, 5, 8]
    prediction = [8, 1, 6, 8]

    result = root_mean_squared_error(
        actual,
        prediction,
    )

    expected = np.sqrt(
        1.5
    )

    assert result == pytest.approx(
        expected
    )


def test_wmape():
    """
    WMAPE should equal total absolute error divided by
    total actual demand.
    """

    actual = [10, 0, 5, 8]
    prediction = [8, 1, 6, 8]

    result = weighted_mean_absolute_percentage_error(
        actual,
        prediction,
    )

    expected = (
        4 / 23
    )

    assert result == pytest.approx(
        expected
    )


def test_forecast_bias_percentage():
    """
    Bias should be zero when aggregate predicted demand
    equals aggregate actual demand.
    """

    actual = [10, 0, 5, 8]
    prediction = [8, 1, 6, 8]

    result = forecast_bias_percentage(
        actual,
        prediction,
    )

    assert result == pytest.approx(
        0.0
    )


def test_calculate_forecast_metrics():
    """
    The convenience function should return all expected
    metrics using the same underlying definitions.
    """

    actual = [10, 0, 5, 8]
    prediction = [8, 1, 6, 8]

    result = calculate_forecast_metrics(
        actual,
        prediction,
    )

    assert result["MAE"] == pytest.approx(
        1.0
    )

    assert result["RMSE"] == pytest.approx(
        np.sqrt(1.5)
    )

    assert result["WMAPE"] == pytest.approx(
        4 / 23
    )

    assert result["Bias (%)"] == pytest.approx(
        0.0
    )


def test_wmape_raises_when_actual_total_is_zero():
    """
    WMAPE is undefined when the total actual demand is zero.
    """

    actual = [0, 0, 0]
    prediction = [0, 1, 2]

    with pytest.raises(
        ValueError
    ):
        weighted_mean_absolute_percentage_error(
            actual,
            prediction,
        )


def test_bias_raises_when_actual_total_is_zero():
    """
    Forecast bias percentage is undefined when total actual
    demand is zero.
    """

    actual = [0, 0, 0]
    prediction = [0, 1, 2]

    with pytest.raises(
        ValueError
    ):
        forecast_bias_percentage(
            actual,
            prediction,
        )