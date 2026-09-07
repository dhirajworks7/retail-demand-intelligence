import json

import pytest

from src.models.persistence import (
    load_model_bundle,
    save_model_bundle,
)
from src.models.train import MODEL_FEATURES


class DummyModel:
    """
    Small serializable stand-in used to test persistence without
    fitting real LightGBM models.
    """

    def __init__(self, name: str):
        self.name = name


def test_save_and_load_model_bundle(tmp_path):
    """
    Saving and loading should preserve both model objects and the
    metadata required for later inference.
    """

    poisson_model = DummyModel(
        "poisson"
    )

    classifier_model = DummyModel(
        "classifier"
    )

    output_dir = (
        tmp_path / "models"
    )

    saved_paths = save_model_bundle(
        poisson_model=poisson_model,
        classifier_model=classifier_model,
        threshold=0.50,
        output_dir=output_dir,
    )

    # All expected bundle files should be created.
    assert saved_paths[
        "poisson_model"
    ].exists()

    assert saved_paths[
        "classifier_model"
    ].exists()

    assert saved_paths[
        "metadata"
    ].exists()

    loaded = load_model_bundle(
        output_dir
    )

    # joblib should reconstruct the original Python objects.
    assert loaded[
        "poisson_model"
    ].name == "poisson"

    assert loaded[
        "classifier_model"
    ].name == "classifier"

    # The two-stage threshold must survive persistence exactly.
    assert loaded[
        "threshold"
    ] == pytest.approx(
        0.50
    )

    # Inference must use the exact same ordered feature contract
    # that was used by the training pipeline.
    assert loaded[
        "model_features"
    ] == list(
        MODEL_FEATURES
    )


def test_save_model_bundle_writes_expected_metadata(
    tmp_path,
):
    """
    The JSON metadata should remain human-readable and contain the
    frozen threshold and ordered model feature contract.
    """

    output_dir = (
        tmp_path / "models"
    )

    save_model_bundle(
        poisson_model=DummyModel(
            "poisson"
        ),
        classifier_model=DummyModel(
            "classifier"
        ),
        threshold=0.60,
        output_dir=output_dir,
    )

    metadata_path = (
        output_dir
        / "model_metadata.json"
    )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    assert metadata[
        "threshold"
    ] == pytest.approx(
        0.60
    )

    assert metadata[
        "model_features"
    ] == list(
        MODEL_FEATURES
    )


def test_save_model_bundle_rejects_invalid_threshold(
    tmp_path,
):
    """
    Invalid probability thresholds should be rejected before any
    model bundle is written.
    """

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        save_model_bundle(
            poisson_model=DummyModel(
                "poisson"
            ),
            classifier_model=DummyModel(
                "classifier"
            ),
            threshold=1.50,
            output_dir=tmp_path,
        )


def test_load_model_bundle_rejects_incomplete_bundle(
    tmp_path,
):
    """
    Loading should fail clearly when one or more required artifacts
    are missing instead of returning a partially usable bundle.
    """

    model_dir = (
        tmp_path / "models"
    )

    model_dir.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="Model bundle is incomplete",
    ):
        load_model_bundle(
            model_dir
        )


def test_load_model_bundle_rejects_missing_metadata_fields(
    tmp_path,
):
    """
    Metadata validation should catch a structurally incomplete
    model bundle even when all three artifact files exist.
    """

    model_dir = (
        tmp_path / "models"
    )

    save_model_bundle(
        poisson_model=DummyModel(
            "poisson"
        ),
        classifier_model=DummyModel(
            "classifier"
        ),
        threshold=0.50,
        output_dir=model_dir,
    )

    metadata_path = (
        model_dir
        / "model_metadata.json"
    )

    # Deliberately remove the feature contract to verify that the
    # loader rejects incomplete inference metadata.
    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "threshold": 0.50,
            },
            file,
        )

    with pytest.raises(
        ValueError,
        match="missing the feature contract",
    ):
        load_model_bundle(
            model_dir
        )