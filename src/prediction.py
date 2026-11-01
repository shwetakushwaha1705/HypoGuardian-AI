"""HypoGuardian AI: Prediction Engine Module.

Loads trained LSTM model, fitted StandardScaler, and model metadata to execute
real-time inference on 1-hour historical glucose sequences. Enforces strict input
validation, identical preprocessing, and transparent error reporting without fallback guessing.
"""

import sys
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd

from config import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    MAX_GAP_MINUTES,
    MAX_PLAUSIBLE_GLUCOSE,
    METADATA_PATH,
    MIN_PLAUSIBLE_GLUCOSE,
    MODEL_PATH,
    SCALER_PATH,
    SEQUENCE_LENGTH,
)


class PredictionEngineError(Exception):
    """Raised when prediction input is invalid or inference fails."""
    pass


class ArtifactsNotFoundError(PredictionEngineError):
    """Raised when model weights or scaler pickle files are missing."""
    pass


class InvalidSequenceError(PredictionEngineError):
    """Raised when an input sequence violates length, spacing, or physiological constraints."""
    pass


class HypoPredictionEngine:
    """Production-ready inference engine for 30-minute hypoglycemia early warning."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        metadata_path: Path = METADATA_PATH,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.metadata_path = Path(metadata_path)

        self.model = None
        self.scaler = None
        self.metadata: Dict[str, Any] = {}
        self.decision_threshold = DEFAULT_CLASSIFICATION_THRESHOLD

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads model, scaler, and metadata if they exist."""
        # Check files exist
        if not self.model_path.exists() or not self.scaler_path.exists():
            return

        try:
            # Lazy import TensorFlow to avoid import penalty when not needed
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)

            if self.metadata_path.exists():
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.decision_threshold = float(
                        self.metadata.get("selected_classification_threshold", DEFAULT_CLASSIFICATION_THRESHOLD)
                    )
        except Exception as e:
            # Keep model as None and log error
            self.model = None
            self.scaler = None
            self.load_error = str(e)

    def is_ready(self) -> bool:
        """Returns True if the engine has successfully loaded the trained model and scaler."""
        return self.model is not None and self.scaler is not None

    def get_status(self) -> Dict[str, Any]:
        """Provides status summary of model availability for the dashboard."""
        if self.is_ready():
            return {
                "ready": True,
                "model_name": self.metadata.get("model_name", "HypoGuardian-LSTM"),
                "version": self.metadata.get("model_version", "1.0.0"),
                "trained_date": self.metadata.get("training_date", "Unknown"),
                "threshold": self.decision_threshold,
                "status_message": "Trained LSTM model and preprocessing scaler are active and ready for inference.",
            }
        else:
            missing = []
            if not self.model_path.exists():
                missing.append(f"Model ({self.model_path.name})")
            if not self.scaler_path.exists():
                missing.append(f"Scaler ({self.scaler_path.name})")

            return {
                "ready": False,
                "missing_artifacts": missing,
                "status_message": (
                    f"Model artifacts not found: {', '.join(missing)}. "
                    "Please run the model training pipeline (`python -m src.train_model`) "
                    "or transfer artifacts from Google Colab."
                ),
            }

    def validate_input(
        self,
        data: Union[pd.DataFrame, List[float], np.ndarray],
        max_gap_minutes: float = MAX_GAP_MINUTES,
    ) -> Tuple[np.ndarray, Optional[str]]:
        """Validates that the input constitutes an eligible, continuous 12-reading sequence.

        Args:
            data: DataFrame with ['timestamp', 'glucose'] or numeric 1D sequence of length 12.
            max_gap_minutes: Tolerance threshold for sensor gaps.

        Returns:
            Tuple of (valid_glucose_1d_array, prediction_timestamp_str).
        """
        if isinstance(data, pd.DataFrame):
            if "glucose" not in data.columns:
                raise InvalidSequenceError("Input DataFrame must contain a 'glucose' column.")

            if len(data) < SEQUENCE_LENGTH:
                raise InvalidSequenceError(
                    f"Insufficient historical data: provided {len(data)} readings, "
                    f"but exactly {SEQUENCE_LENGTH} consecutive readings (~55 mins) are required."
                )

            # Take the most recent SEQUENCE_LENGTH readings
            window_df = data.iloc[-SEQUENCE_LENGTH:].copy()

            # Verify timestamp continuity if timestamp column exists
            pred_timestamp = None
            if "timestamp" in window_df.columns:
                ts = pd.to_datetime(window_df["timestamp"], errors="coerce")
                if ts.isna().any():
                    raise InvalidSequenceError("Sequence contains unparseable or missing timestamps.")

                # Check chronological order
                if not ts.is_monotonic_increasing:
                    raise InvalidSequenceError("Timestamps in input sequence are not strictly chronological.")

                # Check gap intervals
                diffs_min = ts.diff().dropna().dt.total_seconds() / 60.0
                if (diffs_min > max_gap_minutes).any():
                    raise InvalidSequenceError(
                        f"Sequence contains a sensor gap exceeding {max_gap_minutes} minutes. "
                        "Cannot guarantee continuous temporal sequence."
                    )
                pred_timestamp = ts.iloc[-1].strftime("%Y-%m-%d %H:%M:%S")

            # Validate glucose values
            glucose_vals = pd.to_numeric(window_df["glucose"], errors="coerce").values
        else:
            arr = np.asarray(data, dtype=float).ravel()
            if len(arr) < SEQUENCE_LENGTH:
                raise InvalidSequenceError(
                    f"Input array has {len(arr)} readings; exactly {SEQUENCE_LENGTH} are required."
                )
            glucose_vals = arr[-SEQUENCE_LENGTH:]
            pred_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if np.isnan(glucose_vals).any():
            raise InvalidSequenceError("Input sequence contains NaN or missing values.")

        # Check physiological plausibility
        if (glucose_vals < MIN_PLAUSIBLE_GLUCOSE).any() or (glucose_vals > MAX_PLAUSIBLE_GLUCOSE).any():
            raise InvalidSequenceError(
                f"Glucose values must be between {MIN_PLAUSIBLE_GLUCOSE} and {MAX_PLAUSIBLE_GLUCOSE} mg/dL."
            )

        return glucose_vals.astype(np.float32), pred_timestamp

    def predict(
        self,
        data: Union[pd.DataFrame, List[float], np.ndarray],
        override_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Performs forward inference on a validated 12-reading glucose window.

        Args:
            data: Input sequence data (DataFrame or list/array).
            override_threshold: Optional custom classification threshold for sensitivity tuning.

        Returns:
            Dictionary with prediction results, risk level, raw probability, and metadata.
        """
        if not self.is_ready():
            raise ArtifactsNotFoundError(
                "Model or scaler is not loaded. Train the model first via `python -m src.train_model`."
            )

        # 1. Validate input sequence
        glucose_window, pred_ts = self.validate_input(data)

        # 2. Preprocess with the EXACT scaler fitted during training
        # Scaler expects shape (N, 1)
        scaled_glucose = self.scaler.transform(glucose_window.reshape(-1, 1))

        # 3. Reshape for LSTM input: (batch_size=1, time_steps=12, features=1)
        lstm_input = scaled_glucose.reshape(1, SEQUENCE_LENGTH, 1)

        # 4. Model inference
        raw_output = float(self.model.predict(lstm_input, verbose=0)[0][0])
        # Sigmoid output in range [0.0, 1.0]

        threshold = override_threshold if override_threshold is not None else self.decision_threshold
        is_hypo = bool(raw_output >= threshold)

        risk_level = "HIGH RISK: Hypoglycemia Likely" if is_hypo else "LOW RISK: Stable Glycemia"
        clinical_advice = (
            "Model forecasts high probability of glucose dropping below 70 mg/dL within 30 minutes. "
            "Consider verifying with a fingerstick capillary check and preparing 15g fast-acting carbs."
            if is_hypo else
            "Model forecasts stable glucose trajectory above 70 mg/dL for the next 30 minutes."
        )

        return {
            "status": "SUCCESS",
            "prediction_timestamp": pred_ts,
            "raw_probability": round(raw_output, 4),
            "threshold": round(threshold, 4),
            "is_hypo_predicted": is_hypo,
            "risk_level": risk_level,
            "confidence_score": round(raw_output * 100 if is_hypo else (1.0 - raw_output) * 100, 1),
            "latest_glucose": float(glucose_window[-1]),
            "historical_window": glucose_window.tolist(),
            "clinical_advice": clinical_advice,
            "model_version": self.metadata.get("model_version", "1.0.0"),
        }
