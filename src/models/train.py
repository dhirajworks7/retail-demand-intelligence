from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src.evaluation.metrics import (
    calculate_forecast_metrics,
)

from src.models.forecast_models import (
    DEFAULT_DEMAND_THRESHOLD,
    create_demand_classifier,
    create_poisson_regressor,
    create_positive_demand_target,
    predict_poisson_demand,
    predict_positive_demand_probability,
    two_stage_forecast,
)


# ---------------------------------------------------------
# Forecasting configuration
# ---------------------------------------------------------

# M5 forecasting horizon used throughout the project.
FORECAST_HORIZON = 28


# Categorical predictors supplied directly to LightGBM.
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


# Numeric predictors produced by the leakage-safe feature
# engineering pipeline.
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


# Final model feature contract.
#
# is_available is deliberately excluded because unavailable
# observations are handled by a deterministic zero-demand rule
# rather than by the ML models.
MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


TARGET_COLUMN = "sales"
DATE_COLUMN = "date"


# Demand-history features that must be available before an
# observation can be used for model training.
DEMAND_HISTORY_FEATURES = [
    "sales_lag_28",
    "sales_lag_35",
    "sales_lag_42",
    "sales_lag_56",
    "rolling_mean_7_lag28",
    "rolling_mean_28_lag28",
    "rolling_std_7_lag28",
    "rolling_std_28_lag28",
]


# Event columns where missing values mean "no event" rather
# than unknown information.
EVENT_COLUMNS = [
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
]


# ---------------------------------------------------------
# Feature-contract validation
# ---------------------------------------------------------

def validate_training_columns(
    df: pd.DataFrame,
) -> None:
    """
    Verify that the engineered dataset contains every column
    required for training and chronological splitting.
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

    Random splitting is intentionally avoided because it would
    leak future temporal information into model training.
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

    # Normalize dates so chronological comparisons behave
    # consistently even if input dates are strings.
    result[DATE_COLUMN] = pd.to_datetime(
        result[DATE_COLUMN]
    )

    unique_dates = (
        result[DATE_COLUMN]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    # At least one date must remain for model training after
    # reserving validation and test windows.
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

    # Test begins at the first date in the final test window.
    test_start_date = unique_dates.iloc[
        -test_days
    ]

    # Validation begins immediately before the test window.
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


# ---------------------------------------------------------
# Demand-history filtering
# ---------------------------------------------------------

def remove_incomplete_demand_history(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove observations without complete horizon-safe demand
    history.

    Early observations for each product cannot have long-lag
    features such as sales_lag_56. Those rows are structurally
    incomplete and therefore unsuitable for model training.
    """

    missing_columns = (
        set(DEMAND_HISTORY_FEATURES)
        - set(df.columns)
    )

    if missing_columns:
        missing_list = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Dataset is missing required demand-history "
            f"features: {missing_list}"
        )

    result = df.dropna(
        subset=DEMAND_HISTORY_FEATURES
    ).copy()

    if result.empty:
        raise ValueError(
            "No observations remain after removing rows with "
            "incomplete demand history."
        )

    return result


# ---------------------------------------------------------
# Availability filtering
# ---------------------------------------------------------

def select_available_observations(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select product-day observations where the item is available.

    Unavailable observations are not passed to the ML models.
    They are handled separately using a deterministic forecast
    of zero demand.
    """

    if "is_available" not in df.columns:
        raise ValueError(
            "Dataset is missing required column: is_available"
        )

    result = df.loc[
        df["is_available"] == 1
    ].copy()

    if result.empty:
        raise ValueError(
            "No available observations remain for model training."
        )

    return result


# ---------------------------------------------------------
# Event missingness encoding
# ---------------------------------------------------------

def encode_event_missingness(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Replace missing event values with the explicit category
    ``NoEvent``.

    Missing M5 event values indicate that no corresponding event
    occurred on that date, so they contain meaningful information.
    """

    result = df.copy()

    missing_columns = (
        set(EVENT_COLUMNS)
        - set(result.columns)
    )

    if missing_columns:
        missing_list = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Dataset is missing required event columns: "
            f"{missing_list}"
        )

    for column in EVENT_COLUMNS:
        # Convert through object first so NoEvent can be added
        # safely even if the original column is categorical.
        values = result[column].astype(
            "object"
        )

        values = values.where(
            values.notna(),
            "NoEvent",
        )

        result[column] = values.astype(
            "category"
        )

    return result


# ---------------------------------------------------------
# Categorical feature alignment
# ---------------------------------------------------------

def align_categorical_features(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Align categorical levels between training data and another
    split such as validation or test.

    Training categories define the vocabulary understood by the
    fitted LightGBM model. Categories appearing only in another
    split are converted to missing categorical values instead of
    introducing incompatible category codes.
    """

    train_result = train_df.copy()
    other_result = other_df.copy()

    for column in CATEGORICAL_FEATURES:
        if column not in train_result.columns:
            raise ValueError(
                "Training dataset is missing categorical "
                f"feature: {column}"
            )

        if column not in other_result.columns:
            raise ValueError(
                "Comparison dataset is missing categorical "
                f"feature: {column}"
            )

        # Establish the categorical vocabulary from training.
        train_result[column] = (
            train_result[column]
            .astype("category")
        )

        training_categories = (
            train_result[column]
            .cat
            .categories
        )

        # Convert evaluation/test values through object so unseen
        # values can be explicitly detected and replaced.
        other_values = (
            other_result[column]
            .astype("object")
            .copy()
        )

        unseen_mask = (
            other_values.notna()
            & ~other_values.isin(
                training_categories
            )
        )

        # Unseen categories become missing values rather than
        # receiving incompatible category codes.
        other_values.loc[
            unseen_mask
        ] = None

        other_result[column] = pd.Categorical(
            other_values,
            categories=training_categories,
        )

    return (
        train_result,
        other_result,
    )


# ---------------------------------------------------------
# LightGBM model matrices
# ---------------------------------------------------------

def create_model_matrices(
    train_df: pd.DataFrame,
    evaluation_df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """
    Construct LightGBM feature matrices and target vectors.

    Unavailable product-days are excluded because product
    availability is handled separately by the deterministic
    zero-demand business rule.
    """

    validate_training_columns(
        train_df
    )

    validate_training_columns(
        evaluation_df
    )

    # ML fitting and prediction operate only on observations
    # where products are commercially available.
    train_available = select_available_observations(
        train_df
    )

    evaluation_available = select_available_observations(
        evaluation_df
    )

    # Missing event values have a specific business meaning,
    # so represent them explicitly rather than leaving them NaN.
    train_available = encode_event_missingness(
        train_available
    )

    evaluation_available = encode_event_missingness(
        evaluation_available
    )

    # LightGBM requires categorical definitions to remain
    # consistent between training and evaluation data.
    (
        train_available,
        evaluation_available,
    ) = align_categorical_features(
        train_available,
        evaluation_available,
    )

    X_train = train_available[
        MODEL_FEATURES
    ].copy()

    y_train = train_available[
        TARGET_COLUMN
    ].copy()

    X_evaluation = evaluation_available[
        MODEL_FEATURES
    ].copy()

    y_evaluation = evaluation_available[
        TARGET_COLUMN
    ].copy()

    return (
        X_train,
        y_train,
        X_evaluation,
        y_evaluation,
    )


# ---------------------------------------------------------
# Training-data orchestration
# ---------------------------------------------------------

def prepare_training_splits(
    df: pd.DataFrame,
    validation_days: int = FORECAST_HORIZON,
    test_days: int = FORECAST_HORIZON,
):
    """
    Prepare chronological train, validation, and test splits
    for the production forecasting pipeline.

    Training rows with incomplete lag/rolling history are removed.
    Validation and test rows remain complete so evaluation still
    represents the full forecast horizon.
    """

    # Create leakage-safe chronological partitions.
    train_df, validation_df, test_df = (
        chronological_train_validation_test_split(
            df,
            validation_days=validation_days,
            test_days=test_days,
        )
    )

    # Only training rows with complete demand history are eligible
    # for model fitting.
    train_df = remove_incomplete_demand_history(
        train_df
    )

    # Build ML-ready matrices for training and validation.
    (
        X_train,
        y_train,
        X_validation,
        y_validation,
    ) = create_model_matrices(
        train_df,
        validation_df,
    )

    return {
        "train_df": train_df,
        "validation_df": validation_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
    }


# ---------------------------------------------------------
# Model fitting
# ---------------------------------------------------------

def fit_poisson_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    early_stopping_rounds: int = 50,
):
    """
    Fit the LightGBM Poisson demand model.

    Validation data is used only for early stopping. The model
    configuration itself comes from forecast_models.py so model
    parameters remain defined in one place.
    """

    if early_stopping_rounds <= 0:
        raise ValueError(
            "early_stopping_rounds must be greater than zero."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number "
            "of observations."
        )

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "X_validation and y_validation must contain the "
            "same number of observations."
        )

    if X_train.empty:
        raise ValueError(
            "Training feature matrix is empty."
        )

    if X_validation.empty:
        raise ValueError(
            "Validation feature matrix is empty."
        )

    model = create_poisson_regressor()

    # Use the current LightGBM validation API with early stopping.
    model.fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_metric="l1",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=False,
            ),
        ],
    )

    return model


def fit_demand_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    early_stopping_rounds: int = 50,
):
    """
    Fit the binary LightGBM demand-occurrence classifier.

    The classifier predicts whether sales are greater than zero.
    Its probability output is later combined with the Poisson
    magnitude forecast by the two-stage forecasting rule.
    """

    if early_stopping_rounds <= 0:
        raise ValueError(
            "early_stopping_rounds must be greater than zero."
        )

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must contain the same number "
            "of observations."
        )

    if len(X_validation) != len(y_validation):
        raise ValueError(
            "X_validation and y_validation must contain the "
            "same number of observations."
        )

    if X_train.empty:
        raise ValueError(
            "Training feature matrix is empty."
        )

    if X_validation.empty:
        raise ValueError(
            "Validation feature matrix is empty."
        )

    # Convert count-valued sales into the binary occurrence target:
    # 0 = zero demand
    # 1 = positive demand
    y_train_binary = create_positive_demand_target(
        y_train
    )

    y_validation_binary = create_positive_demand_target(
        y_validation
    )

    model = create_demand_classifier()

    model.fit(
        X_train,
        y_train_binary,
        eval_X=X_validation,
        eval_y=y_validation_binary,
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=early_stopping_rounds,
                verbose=False,
            ),
        ],
    )

    # Critical: return the fitted classifier so downstream code
    # receives a real model rather than None.
    return model


# ---------------------------------------------------------
# Full-split forecasting
# ---------------------------------------------------------

def create_validation_forecasts(
    validation_df: pd.DataFrame,
    X_validation: pd.DataFrame,
    poisson_model,
    classifier_model,
    threshold: float = DEFAULT_DEMAND_THRESHOLD,
) -> pd.DataFrame:
    """
    Generate Poisson and two-stage forecasts for the complete
    validation horizon.

    ML models predict only available product-days. Unavailable
    rows are restored afterward with a deterministic forecast
    of zero demand.
    """

    validate_training_columns(
        validation_df
    )

    if X_validation.empty:
        raise ValueError(
            "Validation feature matrix is empty."
        )

    # X_validation preserves original dataframe indices after
    # availability filtering. These indices map model predictions
    # back to the complete validation horizon.
    available_indices = X_validation.index

    if not available_indices.isin(
        validation_df.index
    ).all():
        raise ValueError(
            "X_validation contains indices that are not present "
            "in validation_df."
        )

    # The ML matrix must contain available product-days only.
    if not (
        validation_df.loc[
            available_indices,
            "is_available",
        ] == 1
    ).all():
        raise ValueError(
            "X_validation contains unavailable observations."
        )

    # Predict demand magnitude from the Poisson model.
    poisson_available = predict_poisson_demand(
        poisson_model,
        X_validation,
    )

    # Predict the probability that demand is greater than zero.
    positive_probability = (
        predict_positive_demand_probability(
            classifier_model,
            X_validation,
        )
    )

    # Apply the frozen two-stage gating threshold.
    two_stage_available = two_stage_forecast(
        positive_probability,
        poisson_available,
        threshold=threshold,
    )

    # Begin with the entire validation horizon. Zero initialization
    # automatically implements the deterministic unavailable-item
    # business rule.
    forecasts = validation_df[
        [
            DATE_COLUMN,
            "item_id",
            TARGET_COLUMN,
            "is_available",
        ]
    ].copy()

    forecasts["poisson_prediction"] = 0.0
    forecasts["positive_demand_probability"] = 0.0
    forecasts["two_stage_prediction"] = 0.0

    # Map available-row predictions back using original indices.
    forecasts.loc[
        available_indices,
        "poisson_prediction",
    ] = poisson_available

    forecasts.loc[
        available_indices,
        "positive_demand_probability",
    ] = positive_probability

    forecasts.loc[
        available_indices,
        "two_stage_prediction",
    ] = two_stage_available

    return forecasts

# ---------------------------------------------------------
# Forecast evaluation
# ---------------------------------------------------------

def evaluate_validation_forecasts(
    forecasts: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Evaluate the Poisson and two-stage forecasts on the complete
    validation horizon.

    Metrics are calculated on all validation rows, including
    unavailable product-days that receive deterministic zero
    forecasts.
    """

    required_columns = {
        TARGET_COLUMN,
        "poisson_prediction",
        "two_stage_prediction",
    }

    missing_columns = (
        required_columns
        - set(forecasts.columns)
    )

    if missing_columns:
        missing_list = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Forecast dataframe is missing required evaluation "
            f"columns: {missing_list}"
        )

    if forecasts.empty:
        raise ValueError(
            "Forecast dataframe is empty."
        )

    actual = forecasts[
        TARGET_COLUMN
    ]

    # Evaluate the standalone Poisson demand model.
    poisson_metrics = calculate_forecast_metrics(
        actual,
        forecasts["poisson_prediction"],
    )

    # Evaluate the gated two-stage forecast.
    two_stage_metrics = calculate_forecast_metrics(
        actual,
        forecasts["two_stage_prediction"],
    )

    return {
        "poisson": poisson_metrics,
        "two_stage": two_stage_metrics,
    }

# ---------------------------------------------------------
# End-to-end training and validation orchestration
# ---------------------------------------------------------

def train_and_validate(
    df: pd.DataFrame,
    validation_days: int = FORECAST_HORIZON,
    test_days: int = FORECAST_HORIZON,
    threshold: float = DEFAULT_DEMAND_THRESHOLD,
    early_stopping_rounds: int = 50,
) -> dict[str, object]:
    """
    Run the complete production training and validation workflow.

    The untouched test split is preserved but is not evaluated here.
    This prevents the test horizon from becoming part of routine
    model selection or threshold tuning.

    Workflow:
        1. Create chronological train/validation/test splits.
        2. Remove incomplete demand history from training only.
        3. Build leakage-safe model matrices.
        4. Fit the Poisson demand regressor.
        5. Fit the positive-demand classifier.
        6. Assemble forecasts across the full validation horizon.
        7. Calculate validation metrics for both model strategies.

    Returns
    -------
    dict
        Trained models, validation forecasts, evaluation metrics,
        and the chronological data splits required for later use.
    """

    # Prepare chronological splits and ML-ready matrices using the
    # existing production feature contract.
    prepared = prepare_training_splits(
        df,
        validation_days=validation_days,
        test_days=test_days,
    )

    # Train the standalone demand magnitude model.
    poisson_model = fit_poisson_regressor(
        prepared["X_train"],
        prepared["y_train"],
        prepared["X_validation"],
        prepared["y_validation"],
        early_stopping_rounds=early_stopping_rounds,
    )

    # Train the classifier that estimates whether demand will be
    # positive on an available item-day.
    classifier_model = fit_demand_classifier(
        prepared["X_train"],
        prepared["y_train"],
        prepared["X_validation"],
        prepared["y_validation"],
        early_stopping_rounds=early_stopping_rounds,
    )

    # Reconstruct predictions for the complete validation horizon.
    # Unavailable product-days bypass ML and remain deterministic zero.
    validation_forecasts = create_validation_forecasts(
        prepared["validation_df"],
        prepared["X_validation"],
        poisson_model,
        classifier_model,
        threshold=threshold,
    )

    # Use the shared evaluation module so training orchestration does
    # not duplicate metric definitions.
    validation_metrics = evaluate_validation_forecasts(
        validation_forecasts
    )

    return {
        "poisson_model": poisson_model,
        "classifier_model": classifier_model,
        "validation_forecasts": validation_forecasts,
        "validation_metrics": validation_metrics,
        "train_df": prepared["train_df"],
        "validation_df": prepared["validation_df"],
        "test_df": prepared["test_df"],
    }