"""HypoGuardian AI: Preprocessing Pipeline Module.

Cleans, validates, deduplicates, and structures raw continuous glucose monitoring data.
Ensures strict patient isolation, chronological sorting, and sensor gap detection.
"""

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from config import (
    MAX_GAP_MINUTES,
    MAX_PLAUSIBLE_GLUCOSE,
    MIN_PLAUSIBLE_GLUCOSE,
)


@dataclass
class CleaningAuditReport:
    """Detailed audit metrics from the data preprocessing pipeline."""
    initial_rows: int
    clean_rows: int
    missing_timestamps: int
    missing_glucose: int
    out_of_bounds_glucose: int
    duplicates_removed: int
    sampling_gaps_count: int
    patients_detected: int
    earliest_timestamp: Optional[str] = None
    latest_timestamp: Optional[str] = None
    median_interval_minutes: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


def preprocess_glucose_data(
    df: pd.DataFrame,
    min_glucose: float = MIN_PLAUSIBLE_GLUCOSE,
    max_glucose: float = MAX_PLAUSIBLE_GLUCOSE,
    max_gap_minutes: int = MAX_GAP_MINUTES,
) -> Tuple[pd.DataFrame, CleaningAuditReport]:
    """Applies rigorous data validation, cleaning, and sorting to CGM data.

    Steps:
    1. Records initial row count.
    2. Converts 'timestamp' to datetime and drops unparseable values.
    3. Converts 'glucose' to numeric and drops unparseable values.
    4. Drops rows outside physiological boundaries [20.0, 500.0] mg/dL.
    5. Deduplicates identical (patient_id, timestamp) pairs, retaining the first valid reading.
    6. Sorts chronologically per patient.
    7. Computes time delta between consecutive readings to detect sensor disconnect gaps (> 15 min).
    8. Generates an audit report.

    Args:
        df: Input DataFrame with columns: ['patient_id', 'timestamp', 'glucose'].
        min_glucose: Minimum acceptable clinical glucose level (mg/dL).
        max_glucose: Maximum acceptable clinical glucose level (mg/dL).
        max_gap_minutes: Interval threshold defining an irregular sensor gap.

    Returns:
        Tuple of (cleaned_df, audit_report).
    """
    initial_rows = len(df)
    clean_df = df.copy()

    # 1. Parse timestamps
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
    missing_ts = int(clean_df["timestamp"].isna().sum())
    clean_df = clean_df.dropna(subset=["timestamp"])

    # 2. Parse numeric glucose
    clean_df["glucose"] = pd.to_numeric(clean_df["glucose"], errors="coerce")
    missing_glucose = int(clean_df["glucose"].isna().sum())
    clean_df = clean_df.dropna(subset=["glucose"])

    # 3. Filter physiological outliers
    out_of_bounds_mask = (clean_df["glucose"] < min_glucose) | (clean_df["glucose"] > max_glucose)
    out_of_bounds_count = int(out_of_bounds_mask.sum())
    clean_df = clean_df[~out_of_bounds_mask]

    # 4. Chronological sorting per patient
    clean_df = clean_df.sort_values(by=["patient_id", "timestamp"]).reset_index(drop=True)

    # 5. Deduplication
    dups_mask = clean_df.duplicated(subset=["patient_id", "timestamp"], keep="first")
    duplicates_removed = int(dups_mask.sum())
    clean_df = clean_df[~dups_mask].reset_index(drop=True)

    # 6. Gap analysis per patient
    sampling_gaps = 0
    all_intervals = []

    for _, patient_group in clean_df.groupby("patient_id"):
        if len(patient_group) > 1:
            time_diffs = patient_group["timestamp"].diff().dropna()
            diff_minutes = time_diffs.dt.total_seconds() / 60.0
            all_intervals.extend(diff_minutes.tolist())
            sampling_gaps += int((diff_minutes > max_gap_minutes).sum())

    median_interval = float(np.median(all_intervals)) if all_intervals else None
    earliest_ts = clean_df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S") if not clean_df.empty else None
    latest_ts = clean_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S") if not clean_df.empty else None
    patients_count = clean_df["patient_id"].nunique() if not clean_df.empty else 0

    audit_report = CleaningAuditReport(
        initial_rows=initial_rows,
        clean_rows=len(clean_df),
        missing_timestamps=missing_ts,
        missing_glucose=missing_glucose,
        out_of_bounds_glucose=out_of_bounds_count,
        duplicates_removed=duplicates_removed,
        sampling_gaps_count=sampling_gaps,
        patients_detected=patients_count,
        earliest_timestamp=earliest_ts,
        latest_timestamp=latest_ts,
        median_interval_minutes=round(median_interval, 1) if median_interval is not None else None,
    )

    return clean_df, audit_report
