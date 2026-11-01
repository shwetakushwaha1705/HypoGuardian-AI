"""HypoGuardian AI: Synthetic CGM Data Generator.

Generates physiologically plausible continuous glucose monitoring (CGM) time-series data
for interface demonstration, pipeline testing, and research model training.

DISCLAIMER:
This module generates SYNTHETIC data for academic research prototyping and interface testing.
It does NOT contain real patient health records and must never be used for clinical decision-making.
"""

from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    DATA_RAW_DIR,
    DATA_SAMPLE_DIR,
    EXPECTED_SAMPLING_INTERVAL_MIN,
    RANDOM_SEED,
)


def generate_patient_cgm_trace(
    patient_id: str,
    start_time: datetime,
    days: int = 5,
    sampling_min: int = EXPECTED_SAMPLING_INTERVAL_MIN,
    hypo_frequency: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulates a continuous glucose monitor trace with circadian variations,

    meals, and occasional hypoglycemic excursions.

    Args:
        patient_id: Unique string identifier for the virtual subject.
        start_time: Starting datetime of the simulation.
        days: Number of days to simulate.
        sampling_min: Minutes between consecutive CGM readings (default: 5 min).
        hypo_frequency: Probability of a meal/exercise event triggering hypoglycemia.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: ['patient_id', 'timestamp', 'glucose', 'source']
    """
    rng = np.random.RandomState(seed)
    total_steps = int((days * 24 * 60) / sampling_min)

    timestamps = [start_time + timedelta(minutes=i * sampling_min) for i in range(total_steps)]

    # Baseline physiological glucose (~115 mg/dL)
    glucose_values = np.zeros(total_steps)
    current_glucose = 115.0 + rng.normal(0, 5)

    # Circadian parameters: lowest around 4:00 AM, highest in afternoon
    for i, t in enumerate(timestamps):
        hour = t.hour + t.minute / 60.0

        # Circadian wave (amplitude ~10 mg/dL)
        circadian_effect = -10.0 * np.cos((hour - 4.0) * (2 * np.pi / 24.0))

        # Meal surges: Breakfast (08:00), Lunch (13:00), Dinner (19:30)
        meal_effect = 0.0
        # Check proximity to meals
        if 8.0 <= hour <= 9.5:
            meal_effect += 35.0 * np.sin((hour - 8.0) / 1.5 * np.pi)
        elif 13.0 <= hour <= 14.5:
            meal_effect += 40.0 * np.sin((hour - 13.0) / 1.5 * np.pi)
        elif 19.5 <= hour <= 21.0:
            meal_effect += 45.0 * np.sin((hour - 19.5) / 1.5 * np.pi)

        # Nocturnal / post-exercise hypoglycemic dip (e.g. 02:00-04:00 or 16:00-17:30)
        hypo_dip = 0.0
        # Trigger hypo dip if scheduled for this day
        day_idx = i // (24 * 60 // sampling_min)
        if (day_idx % 2 == 1 and 2.0 <= hour <= 4.0) or (day_idx % 3 == 0 and 16.0 <= hour <= 17.5):
            hypo_dip = -45.0 * np.sin(np.pi * (hour - (2.0 if hour < 5 else 16.0)) / 2.0)

        # Drift toward target with mean-reversion (Ornstein-Uhlenbeck process)
        target = 115.0 + circadian_effect + meal_effect + hypo_dip
        drift = 0.08 * (target - current_glucose)
        noise = rng.normal(0, 2.5)  # Sensor interstitial noise

        current_glucose += drift + noise

        # Clamp to realistic clinical range [35, 400]
        current_glucose = float(np.clip(current_glucose, 35.0, 400.0))
        glucose_values[i] = round(current_glucose, 1)

    df = pd.DataFrame({
        "patient_id": patient_id,
        "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamps],
        "glucose": glucose_values,
        "source": "SYNTHETIC_RESEARCH_BENCHMARK",
    })
    return df


def generate_all_datasets() -> tuple[Path, Path]:
    """Generates both the sample demonstration file and the multi-patient training cohort."""
    base_start = datetime(2026, 3, 1, 0, 0, 0)

    # 1. Generate multi-patient training cohort (3 patients, 7 days each ~ 6,000 readings)
    cohort_dfs = []
    patients = [("SYNTH_P01", 101), ("SYNTH_P02", 202), ("SYNTH_P03", 303)]
    for pid, p_seed in patients:
        pdf = generate_patient_cgm_trace(
            patient_id=pid,
            start_time=base_start,
            days=7,
            seed=p_seed,
        )
        cohort_dfs.append(pdf)

    cohort_df = pd.concat(cohort_dfs, ignore_index=True)
    cohort_path = DATA_RAW_DIR / "synthetic_cgm_cohort.csv"
    cohort_df.to_csv(cohort_path, index=False)
    print(f"-> Generated cohort dataset: {cohort_path} ({len(cohort_df)} records)")

    # 2. Generate small sample file for Streamlit interface demonstration (~288 readings = 24 hours)
    sample_df = generate_patient_cgm_trace(
        patient_id="DEMO_PATIENT",
        start_time=base_start + timedelta(days=1),
        days=1,
        seed=777,
    )
    sample_path = DATA_SAMPLE_DIR / "synthetic_cgm_sample.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"-> Generated sample demo dataset: {sample_path} ({len(sample_df)} records)")

    return cohort_path, sample_path


if __name__ == "__main__":
    generate_all_datasets()
