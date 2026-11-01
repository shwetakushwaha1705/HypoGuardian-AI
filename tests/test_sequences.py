"""Unit tests for HypoGuardian AI: Time-Series Sequence Builder."""

from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import pytest

from src.sequence_builder import create_hypo_sequences


def test_create_hypo_sequences_shape_and_labels():
    # 25 continuous 5-minute readings:
    # First 12 readings (input): normal glucose ~120
    # Next 6 readings (horizon): drops to 65 mg/dL (hypo event!)
    # Remaining readings: recovery
    start_time = datetime(2026, 3, 1, 8, 0, 0)
    timestamps = [start_time + timedelta(minutes=5 * i) for i in range(25)]

    glucose = [120.0] * 12 + [65.0, 68.0, 80.0, 90.0, 100.0, 110.0] + [120.0] * 7
    df = pd.DataFrame({
        "patient_id": "P01",
        "timestamp": timestamps,
        "glucose": glucose,
    })

    ds = create_hypo_sequences(df, seq_length=12, horizon_steps=6, hypo_threshold=70.0)

    # Total possible windows = 25 - (12 + 6) + 1 = 8 windows
    assert len(ds.X) == 8
    assert ds.X.shape == (8, 12, 1)
    assert len(ds.y) == 8
    # For the first window, the horizon includes 65.0, so y[0] must be 1 (hypo)
    assert ds.y[0] == 1


def test_create_hypo_sequences_rejects_sensor_gaps():
    # Readings with an irregular 45-minute sensor gap between step 5 and 6
    start_time = datetime(2026, 3, 1, 8, 0, 0)
    timestamps = []
    current = start_time
    for i in range(20):
        if i == 5:
            current += timedelta(minutes=45)  # Disconnect gap
        else:
            current += timedelta(minutes=5)
        timestamps.append(current)

    glucose = [120.0] * 20
    df = pd.DataFrame({
        "patient_id": "P01",
        "timestamp": timestamps,
        "glucose": glucose,
    })

    # Since the gap is within the 18-step total window, any window spanning across it is rejected
    ds = create_hypo_sequences(df, seq_length=12, horizon_steps=6, max_gap_minutes=15.0)
    # The gap breaks continuity; no valid continuous 18-reading window can span across it
    assert len(ds.X) < (20 - 18 + 1)


def test_create_hypo_sequences_patient_boundary_isolation():
    # 2 patients, each with only 10 readings (< 18 required for window)
    start_time = datetime(2026, 3, 1, 8, 0, 0)
    p1_ts = [start_time + timedelta(minutes=5 * i) for i in range(10)]
    p2_ts = [start_time + timedelta(minutes=5 * i) for i in range(10)]

    df = pd.DataFrame({
        "patient_id": ["P01"] * 10 + ["P02"] * 10,
        "timestamp": p1_ts + p2_ts,
        "glucose": [120.0] * 20,
    })

    # Should NOT combine P01 and P02 across boundaries
    ds = create_hypo_sequences(df, seq_length=12, horizon_steps=6)
    assert len(ds.X) == 0
