"""Unit tests for HypoGuardian AI: Preprocessing Pipeline."""

from io import StringIO
import pandas as pd
import pytest

from src.data_loader import EmptyDatasetError, MissingColumnsError, load_glucose_csv
from src.preprocessing import preprocess_glucose_data


def test_load_glucose_csv_valid():
    csv_data = StringIO(
        "timestamp,glucose,patient_id\n"
        "2026-03-01 08:00:00,120.0,P01\n"
        "2026-03-01 08:05:00,125.0,P01\n"
    )
    df = load_glucose_csv(csv_data)
    assert len(df) == 2
    assert "patient_id" in df.columns
    assert "timestamp" in df.columns
    assert "glucose" in df.columns


def test_load_glucose_csv_missing_columns():
    csv_data = StringIO(
        "time_recorded,value\n"
        "2026-03-01 08:00:00,120.0\n"
    )
    # Note: alias mapper converts 'time_recorded' is not an alias, but 'time' is
    csv_data_bad = StringIO("height,weight\n170,65\n")
    with pytest.raises(MissingColumnsError):
        load_glucose_csv(csv_data_bad)


def test_load_glucose_csv_empty():
    csv_empty = StringIO("")
    with pytest.raises((EmptyDatasetError, Exception)):
        load_glucose_csv(csv_empty)


def test_preprocessing_removes_invalid_and_outliers():
    raw_data = pd.DataFrame({
        "patient_id": ["P01", "P01", "P01", "P01", "P01", "P01"],
        "timestamp": [
            "2026-03-01 08:00:00",
            "2026-03-01 08:05:00",
            "INVALID_TIME",        # Should be dropped (unparseable timestamp)
            "2026-03-01 08:15:00",
            "2026-03-01 08:20:00",
            "2026-03-01 08:25:00",
        ],
        "glucose": [
            110.0,
            999.0,                # Out of bounds (> 500 mg/dL)
            115.0,
            "corrupt_text",       # Should be dropped (non-numeric)
            10.0,                 # Out of bounds (< 20 mg/dL)
            105.0,
        ],
    })

    clean_df, audit = preprocess_glucose_data(raw_data)
    # Valid readings should only be 8:00 (110) and 8:25 (105)
    assert len(clean_df) == 2
    assert audit.missing_timestamps == 1
    assert audit.missing_glucose == 1
    assert audit.out_of_bounds_glucose == 2
    assert list(clean_df["glucose"].values) == [110.0, 105.0]


def test_preprocessing_deduplication_and_sorting():
    raw_data = pd.DataFrame({
        "patient_id": ["P01", "P01", "P01"],
        "timestamp": [
            "2026-03-01 08:10:00",  # Unsorted
            "2026-03-01 08:00:00",
            "2026-03-01 08:00:00",  # Duplicate timestamp
        ],
        "glucose": [130.0, 120.0, 122.0],
    })

    clean_df, audit = preprocess_glucose_data(raw_data)
    assert len(clean_df) == 2
    assert audit.duplicates_removed == 1
    # Check sorting
    assert clean_df.iloc[0]["timestamp"] < clean_df.iloc[1]["timestamp"]
    assert clean_df.iloc[0]["glucose"] == 120.0
