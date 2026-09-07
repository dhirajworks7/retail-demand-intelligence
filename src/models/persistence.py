from __future__ import annotations

import json
from pathlib import Path

import joblib

from src.models.train import MODEL_FEATURES


# Default location for locally generated model artifacts.
DEFAULT_MODEL_DIR = Path("artifacts/models")


def save_model_bundle(
    poisson_model,
    classifier_model,
    threshold: float,
    output_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, Path]:
    """
    Persist the trained forecasting models and their metadata.

    The bundle contains:
        - Poisson LightGBM regressor
        - Positive-demand classifier
        - Metadata describing the feature contract and threshold

    Parameters
    ----------
    poisson_model
        Trained demand magnitude model.

    classifier_model
        Trained positive-demand classifier.

    threshold
        Probability threshold used by the two-stage forecast.

    output_dir
        Directory where model artifacts should be written.

    Returns
    -------
    dict[str, Path]
        Paths to the saved artifacts.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    output_path = Path(output_dir)

    # Create the local artifact directory automatically so callers do
    # not need to prepare it manually before saving models.
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    poisson_path = (
        output_path / "poisson_model.joblib"
    )

    classifier_path = (
        output_path / "demand_classifier.joblib"
    )

    metadata_path = (
        output_path / "model_metadata.json"
    )

    # joblib preserves the trained Python model objects so they can be
    # loaded later for inference without retraining.
    joblib.dump(
        poisson_model,
        poisson_path,
    )

    joblib.dump(
        classifier_model,
        classifier_path,
    )

    metadata = {
        "threshold": float(threshold),

        # Persist the exact training feature order because inference
        # must provide columns in the same contract used during fitting.
        "model_features": list(
            MODEL_FEATURES
        ),
    }

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    return {
        "poisson_model": poisson_path,
        "classifier_model": classifier_path,
        "metadata": metadata_path,
    }


def load_model_bundle(
    model_dir: str | Path = DEFAULT_MODEL_DIR,
) -> dict[str, object]:
    """
    Load a previously persisted forecasting model bundle.

    Returns
    -------
    dict[str, object]
        Loaded models together with threshold and feature metadata.
    """

    model_path = Path(model_dir)

    poisson_path = (
        model_path / "poisson_model.joblib"
    )

    classifier_path = (
        model_path / "demand_classifier.joblib"
    )

    metadata_path = (
        model_path / "model_metadata.json"
    )

    required_paths = {
        "poisson_model": poisson_path,
        "classifier_model": classifier_path,
        "metadata": metadata_path,
    }

    missing_files = [
        str(path)
        for path in required_paths.values()
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Model bundle is incomplete. Missing files: "
            + ", ".join(missing_files)
        )

    poisson_model = joblib.load(
        poisson_path
    )

    classifier_model = joblib.load(
        classifier_path
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    if "threshold" not in metadata:
        raise ValueError(
            "Model metadata is missing the threshold."
        )

    if "model_features" not in metadata:
        raise ValueError(
            "Model metadata is missing the feature contract."
        )

    return {
        "poisson_model": poisson_model,
        "classifier_model": classifier_model,
        "threshold": float(
            metadata["threshold"]
        ),
        "model_features": list(
            metadata["model_features"]
        ),
    }