"""HypoGuardian AI: Time-Series Sequence Builder Module.

Constructs 12-reading sliding input windows (~55 min span) and aligns them with a
30-minute forward-looking binary classification target (hypoglycemia < 70 mg/dL).
Enforces zero data leakage, patient boundary isolation, and sensor gap rejection.
"""

from typing import List, NamedTuple, Tuple
import numpy as np
import pandas as pd

from config import (
    FORECAST_HORIZON_STEPS,
    HYPO_THRESHOLD_MG_DL,
    MAX_GAP_MINUTES,
    SEQUENCE_LENGTH,
)


class SequenceDataset(NamedTuple):
    """Container for constructed time-series dataset."""
    X: np.ndarray             # Shape: (N, SEQUENCE_LENGTH, 1)
    y: np.ndarray             # Shape: (N,) binary targets (0 or 1)
    metadata: pd.DataFrame    # Audit info: patient_id, prediction_timestamp, min_future_glucose


def create_hypo_sequences(
    df: pd.DataFrame,
    seq_length: int = SEQUENCE_LENGTH,
    horizon_steps: int = FORECAST_HORIZON_STEPS,
    hypo_threshold: float = HYPO_THRESHOLD_MG_DL,
    max_gap_minutes: float = MAX_GAP_MINUTES,
) -> SequenceDataset:
    """Extracts chronological sliding sequences and future hypoglycemia labels.

    Protocol:
    - Input sequence: Window of `seq_length` (12) consecutive glucose readings up to t_0.
    - Forecast window: Next `horizon_steps` (6) readings from t_+1 to t_+6 (30 minutes ahead).
    - Label: y = 1 if ANY glucose reading in (t_+1 ... t_+6) < 70.0 mg/dL, else 0.
    - Continuity: If any interval between consecutive readings in the window or horizon
      exceeds `max_gap_minutes` (15 min), the sequence is discarded to prevent false temporal continuity.
    - Isolation: Windows NEVER cross patient boundaries.

    Args:
        df: Preprocessed DataFrame with columns ['patient_id', 'timestamp', 'glucose'].
        seq_length: Number of historical steps in the input sequence (default: 12).
        horizon_steps: Number of future steps in the forecast horizon (default: 6).
        hypo_threshold: Clinical low-glucose threshold (default: 70.0 mg/dL).
        max_gap_minutes: Maximum allowable gap between readings (default: 15.0 min).

    Returns:
        SequenceDataset containing X, y, and metadata DataFrame.
    """
    sequences: List[np.ndarray] = []
    labels: List[int] = []
    meta_records: List[dict] = []

    # Group strictly by patient_id to prevent cross-patient contamination
    for patient_id, group in df.groupby("patient_id"):
        # Ensure chronological ordering
        group = group.sort_values(by="timestamp").reset_index(drop=True)
        timestamps = pd.to_datetime(group["timestamp"]).values
        glucose_values = group["glucose"].values.astype(np.float32)

        total_points = len(glucose_values)
        total_window_len = seq_length + horizon_steps

        if total_points < total_window_len:
            continue

        # Slide window across patient timeline
        for i in range(total_points - total_window_len + 1):
            window_ts = timestamps[i : i + total_window_len]
            window_glucose = glucose_values[i : i + total_window_len]

            # Check time-interval continuity across the entire window (input + horizon)
            time_diffs_min = np.diff(window_ts).astype("timedelta64[s]").astype(float) / 60.0

            # If any gap exceeds tolerance, sensor was disconnected; discard sequence
            if np.any(time_diffs_min > max_gap_minutes) or np.any(time_diffs_min <= 0):
                continue

            # Split into historical input and future horizon
            input_seq = window_glucose[:seq_length]
            horizon_seq = window_glucose[seq_length : seq_length + horizon_steps]

            # Define prediction target: Did hypoglycemia (< 70 mg/dL) occur in the horizon?
            min_future = float(np.min(horizon_seq))
            target_label = 1 if min_future < hypo_threshold else 0

            # Input shape: (seq_length, 1)
            sequences.append(input_seq.reshape(seq_length, 1))
            labels.append(target_label)

            meta_records.append({
                "patient_id": str(patient_id),
                "prediction_timestamp": str(window_ts[seq_length - 1]),
                "horizon_end_timestamp": str(window_ts[-1]),
                "current_glucose": float(input_seq[-1]),
                "min_future_glucose": min_future,
                "label": target_label,
            })

    if not sequences:
        # Return empty structured dataset
        empty_X = np.empty((0, seq_length, 1), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int32)
        empty_meta = pd.DataFrame(columns=[
            "patient_id", "prediction_timestamp", "horizon_end_timestamp",
            "current_glucose", "min_future_glucose", "label"
        ])
        return SequenceDataset(X=empty_X, y=empty_y, metadata=empty_meta)

    X_arr = np.array(sequences, dtype=np.float32)
    y_arr = np.array(labels, dtype=np.int32)
    meta_df = pd.DataFrame(meta_records)

    return SequenceDataset(X=X_arr, y=y_arr, metadata=meta_df)
