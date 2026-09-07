import pandas as pd
import pytest

from src.models.train import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    chronological_train_validation_test_split,
    validate_training_columns,
)


def make_engineered_dataset(
    periods: int = 70,
) -> pd.DataFrame:
    """
    Create a minimal engineered-style dataset containing
    every column required by the training contract.
    """

    dates = pd.date_range(
        "2026-01-01",
        periods=periods,
        freq="D",
    )

    df = pd.DataFrame({
        "date": dates,
        "sales": range(periods),
        "is_available": [1] * periods,
    })

    # Populate categorical features with stable labels and
    # numeric features with simple constant values. The exact
    # values are irrelevant for testing split behavior.
    for feature in MODEL_FEATURES:
        if feature in CATEGORICAL_FEATURES:
            df[feature] = "test"
        else:
            df[feature] = 1

    return df


def test_validate_training_columns_accepts_complete_schema():
    df = make_engineered_dataset()

    # A complete engineered dataset should satisfy the
    # training feature contract without raising an exception.
    validate_training_columns(
        df
    )


def test_validate_training_columns_rejects_missing_feature():
    df = make_engineered_dataset()

    df = df.drop(
        columns=["sales_lag_28"]
    )

    with pytest.raises(ValueError):
        validate_training_columns(
            df
        )


def test_chronological_split_has_expected_sizes():
    df = make_engineered_dataset(
        periods=70
    )

    train_df, validation_df, test_df = (
        chronological_train_validation_test_split(
            df,
            validation_days=28,
            test_days=28,
        )
    )

    assert len(train_df) == 14
    assert len(validation_df) == 28
    assert len(test_df) == 28


def test_chronological_split_has_no_date_overlap():
    df = make_engineered_dataset(
        periods=70
    )

    train_df, validation_df, test_df = (
        chronological_train_validation_test_split(
            df,
            validation_days=28,
            test_days=28,
        )
    )

    train_dates = set(
        train_df["date"]
    )

    validation_dates = set(
        validation_df["date"]
    )

    test_dates = set(
        test_df["date"]
    )

    assert train_dates.isdisjoint(
        validation_dates
    )

    assert train_dates.isdisjoint(
        test_dates
    )

    assert validation_dates.isdisjoint(
        test_dates
    )


def test_chronological_split_preserves_temporal_order():
    df = make_engineered_dataset(
        periods=70
    )

    train_df, validation_df, test_df = (
        chronological_train_validation_test_split(
            df,
            validation_days=28,
            test_days=28,
        )
    )

    # Every training date must occur before validation, and
    # every validation date must occur before the test period.
    assert (
        train_df["date"].max()
        < validation_df["date"].min()
    )

    assert (
        validation_df["date"].max()
        < test_df["date"].min()
    )


def test_chronological_split_preserves_all_rows():
    df = make_engineered_dataset(
        periods=70
    )

    train_df, validation_df, test_df = (
        chronological_train_validation_test_split(
            df,
            validation_days=28,
            test_days=28,
        )
    )

    total_split_rows = (
        len(train_df)
        + len(validation_df)
        + len(test_df)
    )

    assert total_split_rows == len(df)


def test_chronological_split_uses_unique_dates_not_row_count():
    df = make_engineered_dataset(
        periods=70
    )

    # Duplicate the complete dataset to simulate two products
    # observed on every date. Split boundaries should still be
    # based on unique dates rather than total row count.
    second_product = df.copy()

    second_product["item_id"] = (
        "second_item"
    )

    multi_product_df = pd.concat(
        [
            df,
            second_product,
        ],
        ignore_index=True,
    )

    train_df, validation_df, test_df = (
        chronological_train_validation_test_split(
            multi_product_df,
            validation_days=28,
            test_days=28,
        )
    )

    assert train_df["date"].nunique() == 14
    assert validation_df["date"].nunique() == 28
    assert test_df["date"].nunique() == 28

    # Two products should produce two observations per date.
    assert len(train_df) == 28
    assert len(validation_df) == 56
    assert len(test_df) == 56


def test_chronological_split_rejects_zero_validation_days():
    df = make_engineered_dataset()

    with pytest.raises(ValueError):
        chronological_train_validation_test_split(
            df,
            validation_days=0,
            test_days=28,
        )


def test_chronological_split_rejects_zero_test_days():
    df = make_engineered_dataset()

    with pytest.raises(ValueError):
        chronological_train_validation_test_split(
            df,
            validation_days=28,
            test_days=0,
        )


def test_chronological_split_rejects_insufficient_history():
    # 56 dates are exactly the combined validation and test
    # windows, leaving no observations available for training.
    df = make_engineered_dataset(
        periods=56
    )

    with pytest.raises(ValueError):
        chronological_train_validation_test_split(
            df,
            validation_days=28,
            test_days=28,
        )