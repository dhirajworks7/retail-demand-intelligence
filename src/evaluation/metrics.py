# ---------------------------------------------------------
# Forecast Evaluation Metrics
# ---------------------------------------------------------
# This module contains reusable evaluation functions for the
# Retail Demand Intelligence forecasting pipeline.
#
# Keeping metrics in one module ensures that notebooks,
# training pipelines, tests, and future API components all
# use exactly the same metric definitions.

from __future__ import annotations

import numpy as np


def mean_absolute_error(
    actual,
    prediction,
) -> float:
    """
    Calculate Mean Absolute Error (MAE).

    MAE measures the average absolute difference between
    observed demand and predicted demand.

    Lower values indicate better forecasting performance.

    Parameters
    ----------
    actual :
        Observed demand values.

    prediction :
        Forecasted demand values.

    Returns
    -------
    float
        Mean absolute error.
    """

    # Convert inputs to NumPy arrays so the function works
    # consistently with lists, pandas Series, and arrays.
    actual = np.asarray(
        actual,
        dtype="float64",
    )

    prediction = np.asarray(
        prediction,
        dtype="float64",
    )

    absolute_errors = np.abs(
        actual - prediction
    )

    return float(
        absolute_errors.mean()
    )


def root_mean_squared_error(
    actual,
    prediction,
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE).

    RMSE penalizes large forecasting errors more strongly
    than MAE because errors are squared before averaging.

    Lower values indicate better forecasting performance.

    Parameters
    ----------
    actual :
        Observed demand values.

    prediction :
        Forecasted demand values.

    Returns
    -------
    float
        Root mean squared error.
    """

    actual = np.asarray(
        actual,
        dtype="float64",
    )

    prediction = np.asarray(
        prediction,
        dtype="float64",
    )

    errors = (
        actual - prediction
    )

    return float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )


def weighted_mean_absolute_percentage_error(
    actual,
    prediction,
) -> float:
    """
    Calculate Weighted Mean Absolute Percentage Error (WMAPE).

    WMAPE is calculated as:

        sum(abs(actual - prediction)) / sum(actual)

    Unlike traditional MAPE, this formulation works well with
    individual zero-demand observations as long as total actual
    demand across the evaluated dataset is greater than zero.

    The result is returned as a decimal. For example:

        0.70 = 70%

    Parameters
    ----------
    actual :
        Observed demand values.

    prediction :
        Forecasted demand values.

    Returns
    -------
    float
        Weighted mean absolute percentage error.

    Raises
    ------
    ValueError
        If total observed demand equals zero.
    """

    actual = np.asarray(
        actual,
        dtype="float64",
    )

    prediction = np.asarray(
        prediction,
        dtype="float64",
    )

    actual_total = actual.sum()

    # WMAPE cannot be calculated when the denominator is zero.
    if actual_total == 0:
        raise ValueError(
            "WMAPE is undefined when total actual demand is zero."
        )

    absolute_error_total = np.abs(
        actual - prediction
    ).sum()

    return float(
        absolute_error_total
        / actual_total
    )


def forecast_bias_percentage(
    actual,
    prediction,
) -> float:
    """
    Calculate aggregate forecast bias as a percentage.

    Interpretation:

        positive value -> overall overforecasting
        negative value -> overall underforecasting
        zero           -> aggregate forecast is unbiased

    Parameters
    ----------
    actual :
        Observed demand values.

    prediction :
        Forecasted demand values.

    Returns
    -------
    float
        Aggregate forecast bias percentage.

    Raises
    ------
    ValueError
        If total observed demand equals zero.
    """

    actual = np.asarray(
        actual,
        dtype="float64",
    )

    prediction = np.asarray(
        prediction,
        dtype="float64",
    )

    actual_total = actual.sum()

    if actual_total == 0:
        raise ValueError(
            "Forecast bias percentage is undefined when "
            "total actual demand is zero."
        )

    prediction_total = prediction.sum()

    bias_percentage = (
        (prediction_total - actual_total)
        / actual_total
        * 100
    )

    return float(
        bias_percentage
    )


def calculate_forecast_metrics(
    actual,
    prediction,
) -> dict[str, float]:
    """
    Calculate all core forecasting metrics used in the project.

    This convenience function provides one consistent interface
    for evaluating forecasting models throughout the pipeline.

    Parameters
    ----------
    actual :
        Observed demand values.

    prediction :
        Forecasted demand values.

    Returns
    -------
    dict[str, float]
        Dictionary containing:

        - MAE
        - RMSE
        - WMAPE
        - Bias (%)
    """

    return {
        "MAE": mean_absolute_error(
            actual,
            prediction,
        ),

        "RMSE": root_mean_squared_error(
            actual,
            prediction,
        ),

        "WMAPE": weighted_mean_absolute_percentage_error(
            actual,
            prediction,
        ),

        "Bias (%)": forecast_bias_percentage(
            actual,
            prediction,
        ),
    }