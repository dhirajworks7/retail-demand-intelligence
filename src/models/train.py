from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------
# Forecasting configuration
# ---------------------------------------------------------

FORECAST_HORIZON = 28

CATEGORICAL_FEATURES = [
    "item_id",
    "dept_id",
    "cat_id",
    "weekday",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]

NUMERIC_FEATURES = [
    "wday",
    "month",
    "year",
    "wm_yr_wk",
    "snap_CA",
    "day_of_month",
    "week_of_year",
    "quarter",
    "is_weekend",
    "is_event",
    "sales_lag_28",
    "sales_lag_35",
    "sales_lag_42",
    "sales_lag_56",
    "rolling_mean_7_lag28",
    "rolling_mean_28_lag28",
    "rolling_std_7_lag28",
    "rolling_std_28_lag28",
    "sell_price",
    "price_diff_28",
    "log_price_ratio_28",
    "has_price_history_28",
]

MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

TARGET_COLUMN = "sales"
DATE_COLUMN = "date"


# ---------------------------------------------------------
# Feature-contract validation
# ---------------------------------------------------------

def validate_training_columns(
    df: pd.DataFrame,
) -> None:
    """
    Verify that the engineered dataset contains every column
    required for model training and chronological splitting.
    """

    required_columns = {
        *MODEL_FEATURES,
        TARGET_COLUMN,
        DATE_COLUMN,
        "is_available",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        missing_list = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Engineered dataset is missing required training "
            f"columns: {missing_list}"
        )


# ---------------------------------------------------------
# Chronological splitting
# ---------------------------------------------------------

def chronological_train_validation_test_split(
    df: pd.DataFrame,
    validation_days: int = FORECAST_HORIZON,
    test_days: int = FORECAST_HORIZON,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Split an engineered forecasting dataset chronologically.

    The most recent ``test_days`` form the test set.
    The immediately preceding ``validation_days`` form the
    validation set.
    All earlier observations form the training set.

    This avoids random splitting, which would leak future
    temporal information into training.
    """

    validate_training_columns(
        df
    )

    if validation_days <= 0:
        raise ValueError(
            "validation_days must be greater than zero."
        )

    if test_days <= 0:
        raise ValueError(
            "test_days must be greater than zero."
        )

    result = df.copy()

    result[DATE_COLUMN] = pd.to_datetime(
        result[DATE_COLUMN]
    )

    unique_dates = (
        result[DATE_COLUMN]
        .drop_duplicates()
        .sort_values()
    )

    required_days = (
        validation_days
        + test_days
        + 1
    )

    if len(unique_dates) < required_days:
        raise ValueError(
            "Not enough unique dates to create training, "
            "validation, and test sets."
        )

    # Test starts at the first date of the final test window.
    test_start_date = unique_dates.iloc[
        -test_days
    ]

    # Validation starts immediately before the test period.
    validation_start_date = unique_dates.iloc[
        -(validation_days + test_days)
    ]

    train_df = result.loc[
        result[DATE_COLUMN]
        < validation_start_date
    ].copy()

    validation_df = result.loc[
        (
            result[DATE_COLUMN]
            >= validation_start_date
        )
        & (
            result[DATE_COLUMN]
            < test_start_date
        )
    ].copy()

    test_df = result.loc[
        result[DATE_COLUMN]
        >= test_start_date
    ].copy()

    if train_df.empty:
        raise ValueError(
            "Chronological split produced an empty training set."
        )

    if validation_df.empty:
        raise ValueError(
            "Chronological split produced an empty validation set."
        )

    if test_df.empty:
        raise ValueError(
            "Chronological split produced an empty test set."
        )

    return (
        train_df,
        validation_df,
        test_df,
    )