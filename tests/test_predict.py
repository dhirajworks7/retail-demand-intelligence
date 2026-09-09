import numpy as np
import pandas as pd
import pytest

from src.models.predict import (
    generate_forecasts,
    prepare_inference_matrix,
    validate_inference_features,
)


class MockPoissonModel:
    """
    Deterministic stand-in for the Poisson LightGBM regressor.
    """

    def predict(self, X):
        # Return simple increasing predictions so row alignment is easy
        # to verify in the tests below.
        return np.arange(
            1,
            len(X) + 1,
            dtype=float,
        )


class MockClassifierModel:
    """
    Deterministic stand-in for the positive-demand classifier.
    """

    def predict_proba(self, X):
        # Positive-demand probabilities are deliberately chosen so one
        # row falls below the two-stage threshold and another exceeds it.
        positive_probability = np.array(
            [0.40, 0.80],
            dtype=float,
        )

        return np.column_stack(
            [
                1.0 - positive_probability,
                positive_probability,
            ]
        )


def test_validate_inference_features_accepts_required_columns():
    """
    Validation should allow datasets containing all required model
    features, even when additional metadata columns are present.
    """

    df = pd.DataFrame({
        "feature_a": [1.0],
        "feature_b": [2.0],
        "date": ["2026-01-01"],
    })

    validate_inference_features(
        df,
        [
            "feature_a",
            "feature_b",
        ],
    )


def test_validate_inference_features_rejects_missing_columns():
    """
    Missing features must fail clearly before model prediction begins.
    """

    df = pd.DataFrame({
        "feature_a": [1.0],
    })

    with pytest.raises(
        ValueError,
        match="missing required model features",
    ):
        validate_inference_features(
            df,
            [
                "feature_a",
                "feature_b",
            ],
        )


def test_prepare_inference_matrix_preserves_feature_order():
    """
    Inference columns must be returned in the exact persisted training
    order rather than the order found in the incoming dataframe.
    """

    df = pd.DataFrame({
        "feature_b": [2.0],
        "feature_a": [1.0],
        "extra_column": [99],
    })

    X = prepare_inference_matrix(
        df,
        [
            "feature_a",
            "feature_b",
        ],
    )

    assert list(X.columns) == [
        "feature_a",
        "feature_b",
    ]


def test_generate_forecasts_scores_only_available_rows():
    """
    Available rows should be scored by the models while unavailable
    rows bypass ML and receive deterministic zero forecasts.
    """

    df = pd.DataFrame({
        "item_id": [
            "ITEM_1",
            "ITEM_2",
            "ITEM_3",
        ],
        "feature_a": [
            10.0,
            20.0,
            30.0,
        ],
        "feature_b": [
            1.0,
            2.0,
            3.0,
        ],
        "is_available": [
            1,
            0,
            1,
        ],
    })

    forecasts = generate_forecasts(
        df=df,
        poisson_model=MockPoissonModel(),
        classifier_model=MockClassifierModel(),
        model_features=[
            "feature_a",
            "feature_b",
        ],
        threshold=0.50,
    )

    # The first and third rows are the only rows passed to the mock
    # models, so they receive Poisson predictions 1 and 2.
    assert forecasts.loc[
        0,
        "poisson_prediction",
    ] == pytest.approx(1.0)

    assert forecasts.loc[
        1,
        "poisson_prediction",
    ] == pytest.approx(0.0)

    assert forecasts.loc[
        2,
        "poisson_prediction",
    ] == pytest.approx(2.0)

    # The first available row has probability 0.40, so the two-stage
    # gate suppresses its Poisson prediction.
    assert forecasts.loc[
        0,
        "positive_demand_probability",
    ] == pytest.approx(0.40)

    assert forecasts.loc[
        0,
        "two_stage_prediction",
    ] == pytest.approx(0.0)

    # Unavailable rows bypass the classifier and retain zero probability
    # as the business-layer placeholder.
    assert forecasts.loc[
        1,
        "positive_demand_probability",
    ] == pytest.approx(0.0)

    # The final available row exceeds the threshold, so its Poisson
    # magnitude is retained by the two-stage forecast.
    assert forecasts.loc[
        2,
        "positive_demand_probability",
    ] == pytest.approx(0.80)

    assert forecasts.loc[
        2,
        "two_stage_prediction",
    ] == pytest.approx(2.0)


def test_generate_forecasts_returns_zero_when_all_rows_unavailable():
    """
    If every row is unavailable, inference should return immediately
    without requiring either model to make a prediction.
    """

    class FailingModel:
        def predict(self, X):
            raise AssertionError(
                "Model should not be called."
            )

        def predict_proba(self, X):
            raise AssertionError(
                "Model should not be called."
            )

    df = pd.DataFrame({
        "feature_a": [
            1.0,
            2.0,
        ],
        "is_available": [
            0,
            0,
        ],
    })

    forecasts = generate_forecasts(
        df=df,
        poisson_model=FailingModel(),
        classifier_model=FailingModel(),
        model_features=[
            "feature_a",
        ],
        threshold=0.50,
    )

    assert (
        forecasts[
            "poisson_prediction"
        ] == 0.0
    ).all()

    assert (
        forecasts[
            "positive_demand_probability"
        ] == 0.0
    ).all()

    assert (
        forecasts[
            "two_stage_prediction"
        ] == 0.0
    ).all()


def test_generate_forecasts_rejects_missing_availability():
    """
    Availability is part of the forecasting business rule and must be
    present before inference can proceed.
    """

    df = pd.DataFrame({
        "feature_a": [1.0],
    })

    with pytest.raises(
        ValueError,
        match="missing the availability column",
    ):
        generate_forecasts(
            df=df,
            poisson_model=MockPoissonModel(),
            classifier_model=MockClassifierModel(),
            model_features=[
                "feature_a",
            ],
            threshold=0.50,
        )


def test_generate_forecasts_rejects_invalid_threshold():
    """
    The two-stage probability threshold must remain within [0, 1].
    """

    df = pd.DataFrame({
        "feature_a": [1.0],
        "is_available": [1],
    })

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        generate_forecasts(
            df=df,
            poisson_model=MockPoissonModel(),
            classifier_model=MockClassifierModel(),
            model_features=[
                "feature_a",
            ],
            threshold=1.20,
        )