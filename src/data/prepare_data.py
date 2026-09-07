# ---------------------------------------------------------
# M5 Data Preparation Utilities
# ---------------------------------------------------------
# This module transforms the raw M5 source files into the
# modeling table used by the Retail Demand Intelligence
# forecasting pipeline.
#
# Final analytical grain:
#
#     one item x one store x one day
#
# The current project scope focuses on store CA_1.

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

DEFAULT_STORE_ID = "CA_1"


# ---------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------

def load_m5_raw_data(
    raw_data_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three raw M5 source files used by the project.

    Parameters
    ----------
    raw_data_dir :
        Directory containing:
        - calendar.csv
        - sales_train_evaluation.csv
        - sell_prices.csv

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Calendar, sales, and price DataFrames.
    """

    raw_data_dir = Path(
        raw_data_dir
    )

    calendar_path = (
        raw_data_dir
        / "calendar.csv"
    )

    sales_path = (
        raw_data_dir
        / "sales_train_evaluation.csv"
    )

    prices_path = (
        raw_data_dir
        / "sell_prices.csv"
    )

    # Fail early with a clear message if any required source
    # file is missing.
    required_paths = {
        "calendar.csv": calendar_path,
        "sales_train_evaluation.csv": sales_path,
        "sell_prices.csv": prices_path,
    }

    missing_files = [
        filename
        for filename, path in required_paths.items()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required M5 raw files: "
            + ", ".join(
                missing_files
            )
        )

    calendar = pd.read_csv(
        calendar_path
    )

    sales = pd.read_csv(
        sales_path
    )

    prices = pd.read_csv(
        prices_path
    )

    return (
        calendar,
        sales,
        prices,
    )


# ---------------------------------------------------------
# Memory optimization
# ---------------------------------------------------------

def optimize_calendar_dtypes(
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply memory-efficient dtypes to the M5 calendar table.
    """

    result = calendar.copy()

    if "date" in result.columns:
        result["date"] = pd.to_datetime(
            result["date"]
        )

    categorical_columns = [
        "weekday",
        "event_name_1",
        "event_type_1",
        "event_name_2",
        "event_type_2",
    ]

    for column in categorical_columns:
        if column in result.columns:
            result[column] = result[column].astype(
                "category"
            )

    integer_columns = [
        "wday",
        "month",
        "year",
        "wm_yr_wk",
        "snap_CA",
        "snap_TX",
        "snap_WI",
    ]

    for column in integer_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                downcast="integer",
            )

    return result


def optimize_sales_dtypes(
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply memory-efficient dtypes to the wide M5 sales table.
    """

    result = sales.copy()

    categorical_columns = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]

    for column in categorical_columns:
        if column in result.columns:
            result[column] = result[column].astype(
                "category"
            )

    day_columns = [
        column
        for column in result.columns
        if column.startswith("d_")
    ]

    # Daily unit sales are non-negative integers. Downcasting
    # substantially reduces memory use before reshaping.
    for column in day_columns:
        result[column] = pd.to_numeric(
            result[column],
            downcast="unsigned",
        )

    return result


def optimize_price_dtypes(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply memory-efficient dtypes to the M5 price table.
    """

    result = prices.copy()

    categorical_columns = [
        "store_id",
        "item_id",
    ]

    for column in categorical_columns:
        if column in result.columns:
            result[column] = result[column].astype(
                "category"
            )

    if "wm_yr_wk" in result.columns:
        result["wm_yr_wk"] = pd.to_numeric(
            result["wm_yr_wk"],
            downcast="integer",
        )

    if "sell_price" in result.columns:
        result["sell_price"] = pd.to_numeric(
            result["sell_price"],
            downcast="float",
        )

    return result


# ---------------------------------------------------------
# Store filtering
# ---------------------------------------------------------

def filter_store_data(
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    store_id: str = DEFAULT_STORE_ID,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter sales and price tables to a single store.
    """

    _validate_required_columns(
        sales,
        {"store_id"},
    )

    _validate_required_columns(
        prices,
        {"store_id"},
    )

    sales_store = (
        sales.loc[
            sales["store_id"].astype("string")
            == store_id
        ]
        .copy()
    )

    prices_store = (
        prices.loc[
            prices["store_id"].astype("string")
            == store_id
        ]
        .copy()
    )

    if sales_store.empty:
        raise ValueError(
            f"No sales rows found for store {store_id}."
        )

    if prices_store.empty:
        raise ValueError(
            f"No price rows found for store {store_id}."
        )

    return (
        sales_store,
        prices_store,
    )


# ---------------------------------------------------------
# Wide-to-long sales transformation
# ---------------------------------------------------------

def reshape_sales_long(
    sales_store: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the wide M5 sales table to item-day format.

    Returns
    -------
    pd.DataFrame
        Long-form sales table with one row per item and day.
    """

    required_id_columns = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]

    _validate_required_columns(
        sales_store,
        set(required_id_columns),
    )

    day_columns = [
        column
        for column in sales_store.columns
        if column.startswith("d_")
    ]

    if not day_columns:
        raise ValueError(
            "No M5 daily sales columns were found."
        )

    sales_long = sales_store.melt(
        id_vars=required_id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    # Preserve integer unit demand while keeping memory usage
    # lower than the default int64 dtype.
    sales_long["sales"] = pd.to_numeric(
        sales_long["sales"],
        downcast="unsigned",
    )

    return sales_long


# ---------------------------------------------------------
# Calendar join
# ---------------------------------------------------------

def join_calendar(
    sales_long: pd.DataFrame,
    calendar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join calendar metadata onto the item-day sales table.

    The M5 day identifier ``d`` must map to exactly one calendar
    observation, so the join is validated as many-to-one.
    """

    _validate_required_columns(
        sales_long,
        {"d"},
    )

    _validate_required_columns(
        calendar,
        {"d"},
    )

    if calendar["d"].duplicated().any():
        raise ValueError(
            "Calendar contains duplicate values in column 'd'."
        )

    result = sales_long.merge(
        calendar,
        on="d",
        how="left",
        validate="many_to_one",
    )

    if result["date"].isna().any():
        missing_count = int(
            result["date"].isna().sum()
        )

        raise ValueError(
            f"Calendar join produced {missing_count:,} "
            "rows without dates."
        )

    return result


# ---------------------------------------------------------
# Price join
# ---------------------------------------------------------

def join_prices(
    sales_calendar: pd.DataFrame,
    prices_store: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join weekly sell prices onto the item-day table.

    Price observations are uniquely identified by:

        item_id + store_id + wm_yr_wk

    Missing prices are retained because, in the M5 data, they
    represent periods before an item became available for sale.
    """

    join_columns = [
        "item_id",
        "store_id",
        "wm_yr_wk",
    ]

    _validate_required_columns(
        sales_calendar,
        set(join_columns),
    )

    _validate_required_columns(
        prices_store,
        set(
            join_columns
            + ["sell_price"]
        ),
    )

    duplicate_price_keys = (
        prices_store
        .duplicated(
            subset=join_columns
        )
        .sum()
    )

    if duplicate_price_keys > 0:
        raise ValueError(
            f"Price table contains {duplicate_price_keys:,} "
            "duplicate item-store-week keys."
        )

    result = sales_calendar.merge(
        prices_store[
            join_columns
            + ["sell_price"]
        ],
        on=join_columns,
        how="left",
        validate="many_to_one",
    )

    return result


# ---------------------------------------------------------
# Availability feature
# ---------------------------------------------------------

def add_availability_flag(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a binary product availability indicator.

    A non-missing sell price indicates that the product was
    commercially available during that item-week.

    Missing price is therefore not imputed to zero.
    """

    _validate_required_columns(
        df,
        {"sell_price"},
    )

    result = df.copy()

    result["is_available"] = (
        result["sell_price"]
        .notna()
        .astype("uint8")
    )

    return result


# ---------------------------------------------------------
# Business-rule validation
# ---------------------------------------------------------

def validate_unavailable_sales(
    df: pd.DataFrame,
) -> None:
    """
    Verify that unavailable observations do not contain
    positive unit sales.

    This validates the project's interpretation that missing
    prices correspond to pre-release or unavailable periods.
    """

    _validate_required_columns(
        df,
        {
            "sales",
            "is_available",
        },
    )

    invalid_rows = (
        (df["is_available"] == 0)
        & (df["sales"] > 0)
    )

    invalid_count = int(
        invalid_rows.sum()
    )

    if invalid_count > 0:
        raise ValueError(
            f"Found {invalid_count:,} unavailable observations "
            "with positive sales."
        )


# ---------------------------------------------------------
# Complete preparation pipeline
# ---------------------------------------------------------

def prepare_model_data(
    calendar: pd.DataFrame,
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    store_id: str = DEFAULT_STORE_ID,
) -> pd.DataFrame:
    """
    Run the complete in-memory data preparation pipeline.

    Parameters
    ----------
    calendar :
        Raw M5 calendar table.

    sales :
        Raw M5 wide sales table.

    prices :
        Raw M5 sell-price table.

    store_id :
        Store to include in the modeling dataset.

    Returns
    -------
    pd.DataFrame
        Prepared item-store-day modeling table.
    """

    calendar = optimize_calendar_dtypes(
        calendar
    )

    sales = optimize_sales_dtypes(
        sales
    )

    prices = optimize_price_dtypes(
        prices
    )

    sales_store, prices_store = filter_store_data(
        sales,
        prices,
        store_id=store_id,
    )

    sales_long = reshape_sales_long(
        sales_store
    )

    sales_calendar = join_calendar(
        sales_long,
        calendar,
    )

    model_data = join_prices(
        sales_calendar,
        prices_store,
    )

    model_data = add_availability_flag(
        model_data
    )

    validate_unavailable_sales(
        model_data
    )

    return model_data


def prepare_model_data_from_files(
    raw_data_dir: str | Path,
    store_id: str = DEFAULT_STORE_ID,
) -> pd.DataFrame:
    """
    Load raw M5 files and run the complete preparation pipeline.
    """

    calendar, sales, prices = load_m5_raw_data(
        raw_data_dir
    )

    return prepare_model_data(
        calendar,
        sales,
        prices,
        store_id=store_id,
    )


# ---------------------------------------------------------
# Internal validation helper
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