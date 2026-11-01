"""Unit tests for HypoGuardian AI: App Data, Artifacts & Config Consistency."""

import json
from pathlib import Path
import pandas as pd
import pytest

from config import (
    DATA_SAMPLE_DIR,
    EVALUATION_RESULTS_PATH,
    METADATA_PATH,
    MODEL_PATH,
    SCALER_PATH,
)
from src.data_loader import load_glucose_csv
from src.preprocessing import preprocess_glucose_data


def test_sample_cgm_file_exists_and_valid():
    sample_path = DATA_SAMPLE_DIR / "synthetic_cgm_sample.csv"
    assert sample_path.exists(), "Sample demo CSV file must exist for dashboard testing."

    df = load_glucose_csv(sample_path)
    assert not df.empty
    assert "timestamp" in df.columns
    assert "glucose" in df.columns
    assert "patient_id" in df.columns

    clean_df, audit = preprocess_glucose_data(df)
    assert audit.clean_rows > 0
    assert audit.clean_rows == audit.initial_rows


def test_saved_artifacts_exist_and_consistent():
    assert MODEL_PATH.exists(), f"Model file must exist at: {MODEL_PATH}"
    assert SCALER_PATH.exists(), f"Scaler file must exist at: {SCALER_PATH}"
    assert METADATA_PATH.exists(), f"Metadata file must exist at: {METADATA_PATH}"
    assert EVALUATION_RESULTS_PATH.exists(), f"Evaluation results must exist at: {EVALUATION_RESULTS_PATH}"

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert "model_name" in meta
    assert "forecast_horizon_min" in meta
    assert meta["forecast_horizon_min"] == 30
    assert meta["hypo_threshold_mg_dl"] == 70.0

    with open(EVALUATION_RESULTS_PATH, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    assert "metrics" in eval_data
    assert "confusion_matrix" in eval_data
    assert "accuracy" in eval_data["metrics"]
