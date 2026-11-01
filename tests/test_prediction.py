"""Unit tests for HypoGuardian AI: Prediction Engine & Artifact Loading."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.prediction import (
    ArtifactsNotFoundError,
    HypoPredictionEngine,
    InvalidSequenceError,
)


def test_engine_missing_artifacts_handling(tmp_path):
    """Verifies engine handles missing files gracefully without crashing."""
    bogus_model = tmp_path / "nonexistent_model.keras"
    bogus_scaler = tmp_path / "nonexistent_scaler.pkl"
    bogus_meta = tmp_path / "nonexistent_meta.json"

    engine = HypoPredictionEngine(
        model_path=bogus_model,
        scaler_path=bogus_scaler,
        metadata_path=bogus_meta,
    )
    assert not engine.is_ready()
    status = engine.get_status()
    assert status["ready"] is False
    assert len(status["missing_artifacts"]) == 2

    # Predict should raise ArtifactsNotFoundError
    with pytest.raises(ArtifactsNotFoundError):
        engine.predict([100.0] * 12)


def test_engine_validate_input_short_sequence():
    engine = HypoPredictionEngine()
    # Less than 12 readings
    with pytest.raises(InvalidSequenceError):
        engine.validate_input([100.0] * 10)


def test_engine_validate_input_nan_glucose():
    engine = HypoPredictionEngine()
    with pytest.raises(InvalidSequenceError):
        engine.validate_input([100.0] * 11 + [np.nan])


def test_engine_validate_input_out_of_bounds():
    engine = HypoPredictionEngine()
    with pytest.raises(InvalidSequenceError):
        engine.validate_input([100.0] * 11 + [999.0])


def test_engine_validate_input_valid_array():
    engine = HypoPredictionEngine()
    vals, ts = engine.validate_input([120.0] * 12)
    assert len(vals) == 12
    assert ts is not None


def test_engine_validate_input_dataframe_gap():
    engine = HypoPredictionEngine()
    df = pd.DataFrame({
        "timestamp": [
            "2026-03-01 08:00:00",
            "2026-03-01 08:05:00",
            "2026-03-01 08:10:00",
            "2026-03-01 08:15:00",
            "2026-03-01 08:20:00",
            "2026-03-01 08:25:00",
            "2026-03-01 08:30:00",
            "2026-03-01 08:35:00",
            "2026-03-01 08:40:00",
            "2026-03-01 08:45:00",
            "2026-03-01 08:50:00",
            "2026-03-01 09:30:00",  # 40-minute gap!
        ],
        "glucose": [120.0] * 12,
    })
    with pytest.raises(InvalidSequenceError):
        engine.validate_input(df)
