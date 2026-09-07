import pandas as pd
import pytest


from src.models.train import (
    CATEGORICAL_FEATURES,
    DEMAND_HISTORY_FEATURES,
    MODEL_FEATURES,
    align_categorical_features,
    chronological_train_validation_test_split,
    create_model_matrices,
    create_validation_forecasts,
    encode_event_missingness,
    fit_demand_classifier,
    fit_poisson_regressor,
    prepare_training_splits,
    remove_incomplete_demand_history,
    select_available_observations,
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

def test_remove_incomplete_demand_history_drops_missing_rows():
    df = make_engineered_dataset(
        periods=70
    )

    # Simulate structural lag unavailability at the beginning
    # of a product history.
    df.loc[
        df.index[:5],
        "sales_lag_56",
    ] = None

    result = remove_incomplete_demand_history(
        df
    )

    assert len(result) == 65

    assert result[
        DEMAND_HISTORY_FEATURES
    ].isna().sum().sum() == 0


def test_remove_incomplete_demand_history_rejects_missing_feature():
    df = make_engineered_dataset()

    df = df.drop(
        columns=["sales_lag_56"]
    )

    with pytest.raises(ValueError):
        remove_incomplete_demand_history(
            df
        )


def test_select_available_observations_filters_unavailable_rows():
    df = make_engineered_dataset(
        periods=10
    )

    df.loc[
        df.index[:3],
        "is_available",
    ] = 0

    result = select_available_observations(
        df
    )

    assert len(result) == 7
    assert result["is_available"].eq(1).all()


def test_select_available_observations_rejects_missing_column():
    df = make_engineered_dataset()

    df = df.drop(
        columns=["is_available"]
    )

    with pytest.raises(ValueError):
        select_available_observations(
            df
        )


def test_encode_event_missingness_uses_noevent_category():
    df = make_engineered_dataset(
        periods=5
    )

    df["event_name_1"] = None
    df["event_type_1"] = None

    result = encode_event_missingness(
        df
    )

    assert result[
        "event_name_1"
    ].astype("string").eq(
        "NoEvent"
    ).all()

    assert result[
        "event_type_1"
    ].astype("string").eq(
        "NoEvent"
    ).all()

    assert (
        "NoEvent"
        in result["event_name_1"].cat.categories
    )


def test_align_categorical_features_uses_training_categories():
    train_df = make_engineered_dataset(
        periods=5
    )

    evaluation_df = make_engineered_dataset(
        periods=3
    )

    train_df["item_id"] = [
        "ITEM_A",
        "ITEM_A",
        "ITEM_B",
        "ITEM_B",
        "ITEM_A",
    ]

    # ITEM_C is intentionally unseen in training.
    evaluation_df["item_id"] = [
        "ITEM_A",
        "ITEM_C",
        "ITEM_B",
    ]

    train_result, evaluation_result = (
        align_categorical_features(
            train_df,
            evaluation_df,
        )
    )

    assert set(
        train_result["item_id"].cat.categories
    ) == {
        "ITEM_A",
        "ITEM_B",
    }

    # Unseen evaluation categories should become missing rather
    # than creating incompatible category codes.
    assert pd.isna(
        evaluation_result.loc[
            evaluation_result.index[1],
            "item_id",
        ]
    )


def test_create_model_matrices_excludes_unavailable_rows():
    train_df = make_engineered_dataset(
        periods=20
    )

    evaluation_df = make_engineered_dataset(
        periods=10
    )

    train_df.loc[
        train_df.index[:4],
        "is_available",
    ] = 0

    evaluation_df.loc[
        evaluation_df.index[:2],
        "is_available",
    ] = 0

    X_train, y_train, X_eval, y_eval = (
        create_model_matrices(
            train_df,
            evaluation_df,
        )
    )

    assert len(X_train) == 16
    assert len(y_train) == 16
    assert len(X_eval) == 8
    assert len(y_eval) == 8


def test_create_model_matrices_uses_model_feature_contract():
    train_df = make_engineered_dataset(
        periods=20
    )

    evaluation_df = make_engineered_dataset(
        periods=10
    )

    X_train, _, X_eval, _ = (
        create_model_matrices(
            train_df,
            evaluation_df,
        )
    )

    assert X_train.columns.tolist() == MODEL_FEATURES
    assert X_eval.columns.tolist() == MODEL_FEATURES

    # Availability is handled outside the ML model, so it
    # must not appear among the model predictors.
    assert "is_available" not in X_train.columns

def make_lightgbm_test_data():
    """
    Create a small deterministic dataset for testing the
    production LightGBM fitting functions.

    The dataset is intentionally small so the unit tests remain
    fast while still exercising real model fitting.
    """

    import numpy as np

    rng = np.random.default_rng(
        seed=42
    )

    n_train = 120
    n_validation = 40

    def make_features(
        n_rows: int,
    ) -> pd.DataFrame:
        features = pd.DataFrame(
            index=range(n_rows)
        )

        for feature in MODEL_FEATURES:
            if feature in CATEGORICAL_FEATURES:
                # Both training and validation use the same
                # categorical vocabulary.
                features[feature] = pd.Categorical(
                    np.where(
                        np.arange(n_rows) % 2 == 0,
                        "A",
                        "B",
                    ),
                    categories=[
                        "A",
                        "B",
                    ],
                )
            else:
                features[feature] = rng.normal(
                    loc=1.0,
                    scale=0.5,
                    size=n_rows,
                )

        return features

    X_train = make_features(
        n_train
    )

    X_validation = make_features(
        n_validation
    )

    # Use count-valued targets containing both zeros and
    # positive demand observations.
    y_train = pd.Series(
        rng.poisson(
            lam=1.5,
            size=n_train,
        ),
        name="sales",
    )

    y_validation = pd.Series(
        rng.poisson(
            lam=1.5,
            size=n_validation,
        ),
        name="sales",
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
    )


def test_fit_poisson_regressor_can_predict():
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = make_lightgbm_test_data()

    model = fit_poisson_regressor(
        X_train,
        y_train,
        X_validation,
        y_validation,
        early_stopping_rounds=5,
    )

    predictions = model.predict(
        X_validation
    )

    assert len(predictions) == len(
        X_validation
    )

    # Poisson demand predictions should never be negative.
    assert (predictions >= 0).all()

    assert model.best_iteration_ > 0


def test_fit_demand_classifier_can_predict_probabilities():
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = make_lightgbm_test_data()

    model = fit_demand_classifier(
        X_train,
        y_train,
        X_validation,
        y_validation,
        early_stopping_rounds=5,
    )

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    assert len(probabilities) == len(
        X_validation
    )

    # Binary-classification probabilities must remain within
    # their mathematical [0, 1] range.
    assert (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all()

    assert model.best_iteration_ > 0


def test_fit_poisson_regressor_rejects_mismatched_training_rows():
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = make_lightgbm_test_data()

    # Remove one target observation so X and y no longer have
    # matching row counts.
    y_train = y_train.iloc[:-1]

    with pytest.raises(ValueError):
        fit_poisson_regressor(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )


def test_fit_demand_classifier_rejects_invalid_early_stopping():
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = make_lightgbm_test_data()

    with pytest.raises(ValueError):
        fit_demand_classifier(
            X_train,
            y_train,
            X_validation,
            y_validation,
            early_stopping_rounds=0,
        )

def test_prepare_training_splits_preserves_holdout_windows():
    """
    Confirm that orchestration removes incomplete history only
    from training while preserving the full validation and test
    forecast windows.
    """

# Build 100 consecutive daily observations so the chronological
# split has enough history for train, validation, and test windows.
    df = make_engineered_dataset(
    	periods=100
   )
    # Simulate structurally missing demand-history features at the
    # beginning of the series, as occurs with long lag features.
    df.loc[
        df.index[:10],
        DEMAND_HISTORY_FEATURES,
    ] = pd.NA

    result = prepare_training_splits(
        df,
        validation_days=28,
        test_days=28,
    )

    # Chronological splitting on 100 unique dates gives:
    # 44 training days, 28 validation days, and 28 test days.
    # Ten incomplete-history training rows are then removed.
    assert len(
        result["train_df"]
    ) == 34

    assert len(
        result["validation_df"]
    ) == 28

    assert len(
        result["test_df"]
    ) == 28

    # The ML matrices use only available observations.
    # make_engineered_dataset currently marks all rows available,
    # so their sizes should match their prepared source splits.
    assert len(
        result["X_train"]
    ) == 34

    assert len(
        result["y_train"]
    ) == 34

    assert len(
        result["X_validation"]
    ) == 28

    assert len(
        result["y_validation"]
    ) == 28

    # The orchestration result should preserve the exact production
    # feature contract used by the LightGBM models.
    assert list(
        result["X_train"].columns
    ) == MODEL_FEATURES

    assert list(
        result["X_validation"].columns
    ) == MODEL_FEATURES

class MockPoissonModel:
    """
    Deterministic stand-in for the production Poisson model.

    It returns a simple sequence so we can verify that predictions
    are mapped back to the correct validation dataframe indices.
    """

    def predict(
        self,
        X: pd.DataFrame,
    ):
        return [
            float(index + 1)
            for index in range(len(X))
        ]


class MockClassifierModel:
    """
    Deterministic stand-in for the positive-demand classifier.

    The first available row is below the 0.50 gate and the second
    is above it, allowing the test to exercise both branches of
    the two-stage rule.
    """

    def predict_proba(
        self,
        X: pd.DataFrame,
    ):
        import numpy as np

        positive_probability = np.array(
            [
                0.40,
                0.80,
            ][:len(X)],
            dtype=float,
        )

        return np.column_stack(
            [
                1.0 - positive_probability,
                positive_probability,
            ]
        )


def test_create_validation_forecasts_restores_unavailable_rows():
    """
    Confirm that ML predictions are assigned only to available
    observations and unavailable rows remain deterministic zeros.
    """

    validation_df = make_engineered_dataset(
        periods=3
    )

    # Row 1 is unavailable and therefore must never receive an
    # ML forecast.
    validation_df["is_available"] = [
        1,
        0,
        1,
    ]

    # create_validation_forecasts relies on original dataframe
    # indices to map available predictions back to the full split.
    X_validation = validation_df.loc[
        validation_df["is_available"] == 1,
        MODEL_FEATURES,
    ].copy()

    forecasts = create_validation_forecasts(
        validation_df=validation_df,
        X_validation=X_validation,
        poisson_model=MockPoissonModel(),
        classifier_model=MockClassifierModel(),
        threshold=0.50,
    )

    assert len(
        forecasts
    ) == 3

    # Available row 0 receives Poisson prediction 1.0, but its
    # positive-demand probability is below 0.50, so two-stage
    # prediction becomes zero.
    assert forecasts.loc[
        0,
        "poisson_prediction",
    ] == pytest.approx(
        1.0
    )

    assert forecasts.loc[
        0,
        "two_stage_prediction",
    ] == pytest.approx(
        0.0
    )

    # Unavailable row 1 must remain zero for both forecasting
    # approaches and should not receive an ML probability.
    assert forecasts.loc[
        1,
        "poisson_prediction",
    ] == pytest.approx(
        0.0
    )

    assert forecasts.loc[
        1,
        "positive_demand_probability",
    ] == pytest.approx(
        0.0
    )

    assert forecasts.loc[
        1,
        "two_stage_prediction",
    ] == pytest.approx(
        0.0
    )

    # Available row 2 receives the second ML prediction and passes
    # the classifier gate because its probability is 0.80.
    assert forecasts.loc[
        2,
        "poisson_prediction",
    ] == pytest.approx(
        2.0
    )

    assert forecasts.loc[
        2,
        "positive_demand_probability",
    ] == pytest.approx(
        0.80
    )

    assert forecasts.loc[
        2,
        "two_stage_prediction",
    ] == pytest.approx(
        2.0
    )
