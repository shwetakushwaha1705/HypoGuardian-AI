"""HypoGuardian AI: LSTM Model Training Pipeline.

Builds, compiles, and trains an LSTM neural network for 30-minute hypoglycemia early warning.
Features chronological/patient-aware data splitting, strict train-only scaler fitting,
class-imbalance weighting, early stopping, and artifact serialization.
"""

import sys
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BATCH_SIZE,
    DATA_RAW_DIR,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    DENSE_UNITS,
    DROPOUT_RATE_1,
    DROPOUT_RATE_2,
    EARLY_STOPPING_PATIENCE,
    EXPECTED_SAMPLING_INTERVAL_MIN,
    FORECAST_HORIZON_STEPS,
    HYPO_THRESHOLD_MG_DL,
    LEARNING_RATE,
    LSTM_UNITS,
    MAX_EPOCHS,
    METADATA_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NUM_FEATURES,
    RANDOM_SEED,
    REDUCE_LR_PATIENCE,
    SCALER_PATH,
    SEQUENCE_LENGTH,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.data_loader import load_glucose_csv
from src.preprocessing import preprocess_glucose_data
from src.sequence_builder import SequenceDataset, create_hypo_sequences


def build_lstm_model(
    seq_length: int = SEQUENCE_LENGTH,
    num_features: int = NUM_FEATURES,
    lstm_units: int = LSTM_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout_1: float = DROPOUT_RATE_1,
    dropout_2: float = DROPOUT_RATE_2,
    learning_rate: float = LEARNING_RATE,
):
    """Constructs and compiles the Keras LSTM binary classification model."""
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers

    # Set seeds for reproducibility
    tf.random.set_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    model = models.Sequential([
        layers.Input(shape=(seq_length, num_features)),
        layers.LSTM(lstm_units, return_sequences=False, name="lstm_feature_extractor"),
        layers.Dropout(dropout_1, name="lstm_dropout"),
        layers.Dense(dense_units, activation="relu", name="dense_representation"),
        layers.Dropout(dropout_2, name="dense_dropout"),
        layers.Dense(1, activation="sigmoid", name="hypo_probability_output"),
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def split_sequence_data(
    ds: SequenceDataset,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Splits sequence dataset chronologically to strictly avoid time-series leakage.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    total = len(ds.X)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    X_train, y_train = ds.X[:train_end], ds.y[:train_end]
    X_val, y_val = ds.X[train_end:val_end], ds.y[train_end:val_end]
    X_test, y_test = ds.X[val_end:], ds.y[val_end:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_hypoguardian_model(
    data_csv_path: Path = DATA_RAW_DIR / "synthetic_cgm_cohort.csv",
    epochs: int = MAX_EPOCHS,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """End-to-end training pipeline:

    1. Loads and preprocesses CGM data.
    2. Builds sequences and 30-min hypoglycemia labels.
    3. Splits data into chronological Train, Val, Test sets.
    4. Fits StandardScaler on Train set ONLY.
    5. Calculates balanced class weights to handle class imbalance.
    6. Trains LSTM model with EarlyStopping and ReduceLROnPlateau.
    7. Serializes model, scaler, training history, and audit metadata.
    """
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    print(f"-> Loading training data from: {data_csv_path}")
    raw_df = load_glucose_csv(data_csv_path)
    clean_df, audit = preprocess_glucose_data(raw_df)
    print(f"-> Preprocessed: {audit.clean_rows} clean readings across {audit.patients_detected} patient(s).")

    # Build sliding sequences
    ds = create_hypo_sequences(clean_df)
    print(f"-> Extracted {len(ds.X)} sequences with input shape {ds.X.shape[1:]}.")

    if len(ds.X) < 100:
        raise ValueError(f"Insufficient sequences ({len(ds.X)}) to train an LSTM model reliably.")

    # Chronological Split
    X_train, y_train, X_val, y_val, X_test, y_test = split_sequence_data(ds)
    print(f"-> Dataset Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # Check class distributions
    pos_train = int(np.sum(y_train == 1))
    neg_train = int(np.sum(y_train == 0))
    print(f"-> Train class distribution: {neg_train} Normal (0), {pos_train} Hypo Events (1)")

    # Compute balanced class weights
    classes = np.unique(y_train)
    if len(classes) > 1:
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    else:
        class_weight_dict = {0: 1.0, 1: 1.0}
    print(f"-> Computed class weights: {class_weight_dict}")

    # Standardize Features: Fit scaler on TRAIN ONLY
    scaler = StandardScaler()
    # Flatten (N, 12, 1) -> (N*12, 1) for fitting
    X_train_flat = X_train.reshape(-1, 1)
    scaler.fit(X_train_flat)

    # Transform Train, Val, Test
    X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, 1)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    # Save test set for independent evaluation script
    test_split_path = MODELS_DIR / "test_split_cache.npz"
    np.savez_compressed(test_split_path, X_test=X_test_scaled, y_test=y_test)

    # Build LSTM Model
    model = build_lstm_model()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=REDUCE_LR_PATIENCE,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print(f"-> Training LSTM for up to {epochs} epochs...")
    history = model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1,
    )

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"-> Model saved to: {MODEL_PATH}")
    print(f"-> Scaler saved to: {SCALER_PATH}")

    # Save training history
    history_clean = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = MODELS_DIR / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_clean, f, indent=2)

    # Create model metadata
    metadata = {
        "model_name": "HypoGuardian-LSTM",
        "model_version": "1.0.0",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_description": "Synthetic multi-patient continuous glucose monitoring research benchmark",
        "input_features": ["glucose_mg_dl"],
        "input_sequence_length": SEQUENCE_LENGTH,
        "expected_sampling_interval_min": EXPECTED_SAMPLING_INTERVAL_MIN,
        "forecast_horizon_min": 30,
        "forecast_horizon_steps": FORECAST_HORIZON_STEPS,
        "target_definition": "At least one reading in next 30 minutes < 70 mg/dL",
        "hypo_threshold_mg_dl": HYPO_THRESHOLD_MG_DL,
        "preprocessing_method": "StandardScaler (mean=0, variance=1) fitted strictly on training partition",
        "split_strategy": f"Chronological split (Train={TRAIN_RATIO:.0%}, Val={VAL_RATIO:.0%}, Test={TEST_RATIO:.0%})",
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "class_weights": class_weight_dict,
        "selected_classification_threshold": DEFAULT_CLASSIFICATION_THRESHOLD,
        "limitations": [
            "Trained on research benchmark data; clinical safety not certified.",
            "Requires continuous 5-minute sampling intervals.",
            "Predictions are uncalibrated probabilities for academic study only.",
        ],
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"-> Model metadata saved to: {METADATA_PATH}")

    return {
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
        "metadata_path": str(METADATA_PATH),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
    }


if __name__ == "__main__":
    train_hypoguardian_model()
