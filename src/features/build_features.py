# ---------------------------------------------------------
# Feature Engineering Utilities
# ---------------------------------------------------------
# This module contains reusable feature-engineering logic for
# the Retail Demand Intelligence forecasting pipeline.
#
# The project uses a non-recursive 28-day forecasting design.
# Therefore, demand-history features must only use information
# that would have been available at least 28 days before the
# prediction date. This prevents target leakage across the
# full 28-day forecast horizon.

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Project-wide forecasting configuration
# ---------------------------------------------------------

# The M5 forecasting task predicts the next 28 days.
FORECAST_HORIZON = 28

# Demand lags are deliberately >= 28 days so they are known
# at the forecast origin for every day in the prediction window.
DEMAND_LAGS = (
    28,
    35,
    42,
    56,
)

# Rolling windows summarize recent historical demand, but each
# window is shifted by the full 28-day forecasting horizon first.
ROLLING_WINDOWS = (
    7,
    28,
)


# ---------------------------------------------------------
# Calendar features
# ---------------------------------------------------------

def add_calendar_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add calendar-based forecasting features.

    Parameters
    ----------
    df :
        Input dataset containing at least:
        - date
        - event_name_1
        - event_name_2

    Returns
    -------
    pd.DataFrame
        Copy of the input dataset with additional calendar
        features.

    Notes
    -----
    These features are safe for forecasting because calendar
    information is known in advance.
    """

    required_columns = {
        "date",
        "event_name_1",
        "event_name_2",
    }

    _validate_required_columns(
        df,
        required_columns,
    )

    result = df.copy()

    # Ensure date-based operations behave consistently even if
    # the dataset was loaded from CSV rather than Parquet.
    result["date"] = pd.to_datetime(
        result["date"]
    )

    result["day_of_month"] = (
        result["date"].dt.day.astype("uint8")
    )

    result["week_of_year"] = (
        result["date"]
        .dt
        .isocalendar()
        .week
        .astype("uint8")
    )

    result["quarter"] = (
        result["date"].dt.quarter.astype("uint8")
    )

    # Pandas weekday numbering:
    # Monday = 0
    # ...
    # Saturday = 5
    # Sunday = 6
    result["is_weekend"] = (
        result["date"]
        .dt
        .dayofweek
        .isin([5, 6])
        .astype("uint8")
    )

    # An observation is treated as an event day when either
    # primary or secondary event information is present.
    result["is_event"] = (
        result["event_name_1"].notna()
        | result["event_name_2"].notna()
    ).astype("uint8")

    return result


# ---------------------------------------------------------
# Demand-history features
# ---------------------------------------------------------

def add_demand_history_features(
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Create leakage-safe lag and rolling demand features.

    Parameters
    ----------
    df :
        Dataset containing:
        - item_id
        - date
        - sales

    horizon :
        Forecast horizon in days. The default is 28.

    Returns
    -------
    pd.DataFrame
        Dataset with demand lag and rolling features.

    Notes
    -----
    The function sorts observations by product and date before
    generating time-series features.

    Rolling statistics are computed after shifting sales by the
    full forecast horizon. For example:

        rolling_mean_7_lag28

    uses only demand observations that were known at least
    28 days before the prediction date.
    """

    required_columns = {
        "item_id",
        "date",
        "sales",
    }

    _validate_required_columns(
        df,
        required_columns,
    )

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be greater than zero."
        )

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"]
    )

    # Correct ordering is essential because shift() and rolling()
    # operate according to row order within each product.
    result = (
        result
        .sort_values(
            ["item_id", "date"]
        )
        .reset_index(drop=True)
    )

    grouped_sales = (
        result
        .groupby(
            "item_id",
            sort=False,
        )["sales"]
    )

    # -----------------------------------------------------
    # Fixed demand lags
    # -----------------------------------------------------

    for lag in DEMAND_LAGS:

        # Guard against accidentally creating a feature whose
        # lag is shorter than the forecasting horizon.
        if lag < horizon:
            raise ValueError(
                f"Demand lag {lag} is shorter than the "
                f"{horizon}-day forecast horizon."
            )

        result[f"sales_lag_{lag}"] = (
            grouped_sales.shift(lag)
        )

    # -----------------------------------------------------
    # Horizon-safe rolling statistics
    # -----------------------------------------------------

    # First shift demand by the forecast horizon. Any rolling
    # statistics calculated from this series therefore exclude
    # information inside the future 28-day prediction window.
    shifted_sales = (
        grouped_sales.shift(horizon)
    )

    for window in ROLLING_WINDOWS:

        # Rolling mean captures recent demand level.
        result[
            f"rolling_mean_{window}_lag{horizon}"
        ] = (
            shifted_sales
            .groupby(
                result["item_id"],
                sort=False,
            )
            .transform(
                lambda series: (
                    series
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
        )

        # Rolling standard deviation captures recent demand
        # volatility.
        result[
            f"rolling_std_{window}_lag{horizon}"
        ] = (
            shifted_sales
            .groupby(
                result["item_id"],
                sort=False,
            )
            .transform(
                lambda series: (
                    series
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .std()
                )
            )
        )

    return result


# ---------------------------------------------------------
# Price features
# ---------------------------------------------------------

def add_price_features(
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Add historical price-change features.

    Parameters
    ----------
    df :
        Dataset containing:
        - item_id
        - date
        - sell_price

    horizon :
        Number of days used for historical price comparison.

    Returns
    -------
    pd.DataFrame
        Dataset with lagged and relative price features.

    Notes
    -----
    Current sell_price is retained because the M5 setup provides
    target-period pricing information.

    Historical price features are shifted by 28 days so they are
    consistent with the project's forecast-origin design.
    """

    required_columns = {
        "item_id",
        "date",
        "sell_price",
    }

    _validate_required_columns(
        df,
        required_columns,
    )

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be greater than zero."
        )

    result = df.copy()

    result["date"] = pd.to_datetime(
        result["date"]
    )

    result = (
        result
        .sort_values(
            ["item_id", "date"]
        )
        .reset_index(drop=True)
    )

    grouped_price = (
        result
        .groupby(
            "item_id",
            sort=False,
        )["sell_price"]
    )

    price_lag_column = (
        f"sell_price_lag_{horizon}"
    )

    result[price_lag_column] = (
        grouped_price.shift(horizon)
    )

    # Record whether historical price information exists before
    # filling derived features. This preserves the distinction
    # between:
    #   - a real zero price change
    #   - unavailable historical price information
    result[
        f"has_price_history_{horizon}"
    ] = (
        result[price_lag_column]
        .notna()
        .astype("uint8")
    )

    # Absolute price change relative to the price observed
    # 28 days earlier.
    result[
        f"price_diff_{horizon}"
    ] = (
        result["sell_price"]
        - result[price_lag_column]
    )

    # -----------------------------------------------------
    # Log price ratio
    # -----------------------------------------------------
    # A log ratio provides a symmetric representation of
    # proportional increases and decreases.
    #
    # Historical prices <= 0 are excluded from the ratio to
    # avoid division-by-zero and invalid logarithms.
    valid_price_ratio = (
        result["sell_price"].notna()
        & result[price_lag_column].notna()
        & (result["sell_price"] > 0)
        & (result[price_lag_column] > 0)
    )

    ratio = pd.Series(
        np.nan,
        index=result.index,
        dtype="float64",
    )

    ratio.loc[valid_price_ratio] = (
        result.loc[
            valid_price_ratio,
            "sell_price",
        ]
        / result.loc[
            valid_price_ratio,
            price_lag_column,
        ]
    )

    result[
        f"log_price_ratio_{horizon}"
    ] = np.log(
        ratio
    )

    # Missing historical price information is represented
    # explicitly by has_price_history_28. Therefore, the derived
    # price-change features can safely use neutral values.
    result[
        f"price_diff_{horizon}"
    ] = (
        result[
            f"price_diff_{horizon}"
        ]
        .fillna(0.0)
    )

    result[
        f"log_price_ratio_{horizon}"
    ] = (
        result[
            f"log_price_ratio_{horizon}"
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .fillna(0.0)
    )

    return result


# ---------------------------------------------------------
# Complete feature pipeline
# ---------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    horizon: int = FORECAST_HORIZON,
) -> pd.DataFrame:
    """
    Run the complete feature-engineering pipeline.

    Parameters
    ----------
    df :
        Prepared item-store-day dataset.

    horizon :
        Forecast horizon used for leakage-safe historical
        feature construction.

    Returns
    -------
    pd.DataFrame
        Feature-engineered dataset ready for chronological
        model splitting.

    Notes
    -----
    This function does not remove early rows with incomplete
    demand history. That filtering belongs to the model-training
    stage because validation and test rows must remain intact.
    """

    result = add_calendar_features(
        df
    )

    result = add_demand_history_features(
        result,
        horizon=horizon,
    )

    result = add_price_features(
        result,
        horizon=horizon,
    )

    return result


# ---------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------

def validate_daily_product_history(
    df: pd.DataFrame,
) -> None:
    """
    Validate the item-day time-series structure.

    The function checks for duplicate item-date observations
    and gaps in each product's daily history.

    Raises
    ------
    ValueError
        If duplicate observations or date gaps are detected.
    """

    required_columns = {
        "item_id",
        "date",
    }

    _validate_required_columns(
        df,
        required_columns,
    )

    check_df = df[
        ["item_id", "date"]
    ].copy()

    check_df["date"] = pd.to_datetime(
        check_df["date"]
    )

    # Each product should contain at most one row per date.
    duplicate_count = (
        check_df
        .duplicated(
            subset=[
                "item_id",
                "date",
            ]
        )
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"Found {duplicate_count:,} duplicate "
            "item-date observations."
        )

    check_df = check_df.sort_values(
        [
            "item_id",
            "date",
        ]
    )

    # Consecutive observations within an item should differ by
    # exactly one calendar day.
    date_difference = (
        check_df
        .groupby(
            "item_id",
            sort=False,
        )["date"]
        .diff()
    )

    invalid_gap_count = (
        date_difference
        .dropna()
        .ne(
            pd.Timedelta(days=1)
        )
        .sum()
    )

    if invalid_gap_count > 0:
        raise ValueError(
            f"Found {invalid_gap_count:,} non-daily gaps "
            "in product histories."
        )


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: set[str],
) -> None:
    """
    Verify that all columns required by a feature function exist.

    This private helper provides clearer error messages than
    allowing pandas to fail later with an indirect KeyError.
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