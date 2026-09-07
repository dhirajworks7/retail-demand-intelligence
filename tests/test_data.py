import pandas as pd
import pytest

from src.data.prepare_data import (
    add_availability_flag,
    filter_store_data,
    join_calendar,
    join_prices,
    prepare_model_data,
    reshape_sales_long,
    validate_unavailable_sales,
)


def make_calendar() -> pd.DataFrame:
    """
    Create a minimal calendar table for deterministic unit tests.
    """
    return pd.DataFrame({
        "date": [
            "2026-01-01",
            "2026-01-02",
            "2026-01-03",
        ],
        "wm_yr_wk": [100, 100, 101],
        "weekday": ["Thursday", "Friday", "Saturday"],
        "wday": [5, 6, 7],
        "month": [1, 1, 1],
        "year": [2026, 2026, 2026],
        "d": ["d_1", "d_2", "d_3"],
        "event_name_1": [None, None, None],
        "event_type_1": [None, None, None],
        "event_name_2": [None, None, None],
        "event_type_2": [None, None, None],
        "snap_CA": [0, 1, 0],
        "snap_TX": [0, 0, 0],
        "snap_WI": [0, 0, 0],
    })


def make_sales() -> pd.DataFrame:
    """
    Create a two-product wide M5-style sales table.
    """
    return pd.DataFrame({
        "id": [
            "ITEM_1_CA_1_evaluation",
            "ITEM_2_CA_1_evaluation",
        ],
        "item_id": ["ITEM_1", "ITEM_2"],
        "dept_id": ["FOODS_1", "FOODS_1"],
        "cat_id": ["FOODS", "FOODS"],
        "store_id": ["CA_1", "CA_1"],
        "state_id": ["CA", "CA"],
        "d_1": [2, 0],
        "d_2": [3, 1],
        "d_3": [4, 2],
    })


def make_prices() -> pd.DataFrame:
    """
    Create weekly price observations for both test products.
    """
    return pd.DataFrame({
        "store_id": ["CA_1", "CA_1", "CA_1", "CA_1"],
        "item_id": ["ITEM_1", "ITEM_1", "ITEM_2", "ITEM_2"],
        "wm_yr_wk": [100, 101, 100, 101],
        "sell_price": [2.50, 2.75, 3.00, 3.25],
    })


def test_filter_store_data_returns_requested_store():
    sales = make_sales()
    prices = make_prices()

    filtered_sales, filtered_prices = filter_store_data(
        sales,
        prices,
        store_id="CA_1",
    )

    assert set(filtered_sales["store_id"]) == {"CA_1"}
    assert set(filtered_prices["store_id"]) == {"CA_1"}


def test_filter_store_data_raises_for_unknown_store():
    with pytest.raises(ValueError):
        filter_store_data(
            make_sales(),
            make_prices(),
            store_id="TX_9",
        )


def test_reshape_sales_long_creates_item_day_rows():
    sales_long = reshape_sales_long(
        make_sales()
    )

    # Two items over three days should create six observations.
    assert len(sales_long) == 6

    assert set(sales_long["d"]) == {
        "d_1",
        "d_2",
        "d_3",
    }


def test_join_calendar_maps_day_to_date():
    sales_long = reshape_sales_long(
        make_sales()
    )

    result = join_calendar(
        sales_long,
        make_calendar(),
    )

    d1_dates = result.loc[
        result["d"] == "d_1",
        "date",
    ]

    # join_calendar() is responsible for key-based joining, not dtype
# conversion. The raw synthetic calendar stores dates as strings.
    assert (
    d1_dates
    == "2026-01-01"
    ).all()


def test_join_calendar_raises_for_duplicate_day_keys():
    calendar = make_calendar()

    # Duplicate one calendar day to violate the many-to-one
    # relationship required by the sales-calendar join.
    calendar = pd.concat(
        [
            calendar,
            calendar.iloc[[0]],
        ],
        ignore_index=True,
    )

    sales_long = reshape_sales_long(
        make_sales()
    )

    with pytest.raises(ValueError):
        join_calendar(
            sales_long,
            calendar,
        )


def test_join_calendar_raises_when_day_has_no_calendar_match():
    calendar = make_calendar()

    # Removing d_3 means two sales rows will fail to receive
    # calendar metadata.
    calendar = calendar[
        calendar["d"] != "d_3"
    ]

    sales_long = reshape_sales_long(
        make_sales()
    )

    with pytest.raises(ValueError):
        join_calendar(
            sales_long,
            calendar,
        )


def test_join_prices_maps_weekly_prices_correctly():
    sales_long = reshape_sales_long(
        make_sales()
    )

    sales_calendar = join_calendar(
        sales_long,
        make_calendar(),
    )

    result = join_prices(
        sales_calendar,
        make_prices(),
    )

    item_1_week_100 = result.loc[
        (result["item_id"] == "ITEM_1")
        & (result["wm_yr_wk"] == 100),
        "sell_price",
    ]

    assert (
        item_1_week_100 == 2.50
    ).all()


def test_join_prices_raises_for_duplicate_price_keys():
    prices = make_prices()

    # Duplicate one item-store-week record to verify that
    # ambiguous weekly prices are rejected.
    prices = pd.concat(
        [
            prices,
            prices.iloc[[0]],
        ],
        ignore_index=True,
    )

    sales_long = reshape_sales_long(
        make_sales()
    )

    sales_calendar = join_calendar(
        sales_long,
        make_calendar(),
    )

    with pytest.raises(ValueError):
        join_prices(
            sales_calendar,
            prices,
        )


def test_add_availability_flag_marks_missing_price_unavailable():
    df = pd.DataFrame({
        "sell_price": [
            2.50,
            None,
        ],
    })

    result = add_availability_flag(
        df
    )

    assert result["is_available"].tolist() == [
        1,
        0,
    ]


def test_validate_unavailable_sales_accepts_zero_sales():
    df = pd.DataFrame({
        "sales": [0, 3],
        "is_available": [0, 1],
    })

    # The function should complete without raising because
    # unavailable observations have zero sales.
    validate_unavailable_sales(
        df
    )


def test_validate_unavailable_sales_rejects_positive_sales():
    df = pd.DataFrame({
        "sales": [2],
        "is_available": [0],
    })

    with pytest.raises(ValueError):
        validate_unavailable_sales(
            df
        )


def test_prepare_model_data_end_to_end():
    result = prepare_model_data(
        make_calendar(),
        make_sales(),
        make_prices(),
        store_id="CA_1",
    )

    assert len(result) == 6

    assert {
        "date",
        "sales",
        "sell_price",
        "is_available",
    }.issubset(
        result.columns
    )

    # All synthetic products have price coverage in both weeks.
    assert result["is_available"].eq(1).all()

    # The prepared analytical grain should remain unique.
    assert (
        result[
            [
                "item_id",
                "store_id",
                "date",
            ]
        ]
        .duplicated()
        .sum()
        == 0
    )