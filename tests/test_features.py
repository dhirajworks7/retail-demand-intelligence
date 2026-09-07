# ---------------------------------------------------------
# Tests for feature engineering
# ---------------------------------------------------------
# These tests verify that the production feature pipeline
# creates mathematically correct and leakage-safe features.

import pandas as pd
import pytest

from src.features.build_features import (
    add_calendar_features,
    add_demand_history_features,
    add_price_features,
    build_features,
    validate_daily_product_history,
)


def create_test_dataframe() -> pd.DataFrame:
    """
    Create a simple 70-day product history for testing.

    Demand increases from 1 to 70, making historical lag and
    rolling calculations easy to verify manually.
    """

    dates = pd.date_range(
        start="2026-01-01",
        periods=70,
        freq="D",
    )

    return pd.DataFrame({
        "item_id": ["TEST_ITEM"] * 70,
        "date": dates,
        "sales": list(range(1, 71)),
        "sell_price": [10.0] * 70,
        "event_name_1": [None] * 70,
        "event_name_2": [None] * 70,
    })


def test_sales_lag_28():
    """
    The 28-day lag must contain demand from exactly
    28 calendar days earlier.

    On the final row:
        current sales = 70
        sales 28 days earlier = 42
    """

    df = create_test_dataframe()

    result = add_demand_history_features(
        df
    )

    assert result.iloc[-1]["sales_lag_28"] == pytest.approx(
        42.0
    )


def test_sales_lag_56():
    """
    Verify the longest demand lag used by the model.

    On the final row:
        current sales = 70
        sales 56 days earlier = 14
    """

    df = create_test_dataframe()

    result = add_demand_history_features(
        df
    )

    assert result.iloc[-1]["sales_lag_56"] == pytest.approx(
        14.0
    )


def test_rolling_mean_7_lag28():
    """
    Verify the leakage-safe 7-day rolling mean.

    For the final observation, sales is first shifted by
    28 days. The rolling window therefore contains:

        36, 37, 38, 39, 40, 41, 42

    whose mean is 39.
    """

    df = create_test_dataframe()

    result = add_demand_history_features(
        df
    )

    assert result.iloc[-1][
        "rolling_mean_7_lag28"
    ] == pytest.approx(
        39.0
    )


def test_rolling_mean_28_lag28():
    """
    Verify the leakage-safe 28-day rolling mean.

    For the final observation, the available historical
    window contains sales values 15 through 42.

    Their mean is:

        (15 + 42) / 2 = 28.5
    """

    df = create_test_dataframe()

    result = add_demand_history_features(
        df
    )

    assert result.iloc[-1][
        "rolling_mean_28_lag28"
    ] == pytest.approx(
        28.5
    )


def test_price_features_with_constant_price():
    """
    Constant prices should produce zero price change and
    zero log price ratio once historical price exists.
    """

    df = create_test_dataframe()

    result = add_price_features(
        df
    )

    final_row = result.iloc[-1]

    assert final_row[
        "sell_price_lag_28"
    ] == pytest.approx(
        10.0
    )

    assert final_row[
        "price_diff_28"
    ] == pytest.approx(
        0.0
    )

    assert final_row[
        "log_price_ratio_28"
    ] == pytest.approx(
        0.0
    )

    assert final_row[
        "has_price_history_28"
    ] == 1


def test_missing_price_history_flag():
    """
    Observations without 28 days of historical price data
    should explicitly record that history as unavailable.
    """

    df = create_test_dataframe()

    result = add_price_features(
        df
    )

    # The first observation cannot have a 28-day price lag.
    first_row = result.iloc[0]

    assert first_row[
        "has_price_history_28"
    ] == 0

    # Derived features use neutral values when historical
    # price information is unavailable.
    assert first_row[
        "price_diff_28"
    ] == pytest.approx(
        0.0
    )

    assert first_row[
        "log_price_ratio_28"
    ] == pytest.approx(
        0.0
    )


def test_calendar_features():
    """
    Verify that calendar features are generated correctly.
    """

    df = create_test_dataframe()

    result = add_calendar_features(
        df
    )

    # January 1, 2026 is not a weekend.
    assert result.iloc[0][
        "is_weekend"
    ] == 0

    # No event names were supplied in the synthetic dataset.
    assert result.iloc[0][
        "is_event"
    ] == 0

    assert result.iloc[0][
        "day_of_month"
    ] == 1

    assert result.iloc[0][
        "quarter"
    ] == 1


def test_complete_feature_pipeline():
    """
    The complete pipeline should generate all major feature
    groups in one call.
    """

    df = create_test_dataframe()

    result = build_features(
        df
    )

    expected_columns = {
        "sales_lag_28",
        "sales_lag_35",
        "sales_lag_42",
        "sales_lag_56",
        "rolling_mean_7_lag28",
        "rolling_mean_28_lag28",
        "rolling_std_7_lag28",
        "rolling_std_28_lag28",
        "sell_price_lag_28",
        "price_diff_28",
        "log_price_ratio_28",
        "has_price_history_28",
        "day_of_month",
        "week_of_year",
        "quarter",
        "is_weekend",
        "is_event",
    }

    assert expected_columns.issubset(
        result.columns
    )


def test_duplicate_item_dates_raise_error():
    """
    Duplicate item-date observations would make lag features
    ambiguous and must therefore be rejected.
    """

    df = create_test_dataframe()

    # Add a duplicate of the first observation.
    df = pd.concat(
        [
            df,
            df.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicate",
    ):
        validate_daily_product_history(
            df
        )


def test_daily_gap_raises_error():
    """
    Missing calendar days can make row-based lags represent
    the wrong number of actual calendar days.
    """

    df = create_test_dataframe()

    # Remove one date from the middle of the product history.
    df = df.drop(
        index=30
    ).reset_index(
        drop=True
    )

    with pytest.raises(
        ValueError,
        match="non-daily gaps",
    ):
        validate_daily_product_history(
            df
        )


def test_missing_required_columns_raise_error():
    """
    Feature functions should fail with a clear message when
    required input columns are missing.
    """

    df = create_test_dataframe().drop(
        columns=["sales"]
    )

    with pytest.raises(
        ValueError,
        match="sales",
    ):
        add_demand_history_features(
            df
        )