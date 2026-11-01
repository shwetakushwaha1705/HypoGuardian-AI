"""HypoGuardian AI: Configuration Module.

Centralized configuration for paths, clinical thresholds, time-series window parameters,
model hyperparameters, and data-split strategies.
"""

from pathlib import Path

# ==========================================
# 1. FILE SYSTEM PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_SAMPLE_DIR = DATA_DIR / "sample"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "hypo_lstm_model.keras"
SCALER_PATH = MODELS_DIR / "glucose_scaler.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
EVALUATION_RESULTS_PATH = MODELS_DIR / "evaluation_results.json"

NOTEBOOKS_DIR = BASE_DIR / "notebooks"
DOCS_DIR = BASE_DIR / "docs"

# Ensure all primary runtime directories exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, DATA_SAMPLE_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CLINICAL & CGM SAMPLING PARAMETERS
# ==========================================
# Standard Continuous Glucose Monitor (CGM) sampling rate (Dexcom G6/Medtronic Guardian)
EXPECTED_SAMPLING_INTERVAL_MIN = 5

# Maximum allowable gap between consecutive readings before sequence is flagged/split
MAX_GAP_MINUTES = 15

# Physiological plausibility boundaries (readings outside this are sensor errors/artifacts)
MIN_PLAUSIBLE_GLUCOSE = 20.0   # mg/dL
MAX_PLAUSIBLE_GLUCOSE = 500.0  # mg/dL

# Standard Clinical Hypoglycemia Threshold (ADA / Endocrine Society Level 1 Hypoglycemia)
HYPO_THRESHOLD_MG_DL = 70.0    # mg/dL


# ==========================================
# 3. TIME-SERIES WINDOW & HORIZON SETTINGS
# ==========================================
# Sequence length: 12 readings at ~5-min intervals (covers 11 intervals = ~55 minutes of history)
SEQUENCE_LENGTH = 12

# Forecast horizon: 6 future readings at ~5-min intervals (30 minutes ahead)
# Prediction Target: Does glucose drop below 70 mg/dL at ANY point within the next 30 minutes?
FORECAST_HORIZON_STEPS = 6

# Number of features fed into LSTM (univariate blood glucose = 1)
NUM_FEATURES = 1


# ==========================================
# 4. DATA SPLITTING RATIOS
# ==========================================
# Chronological / Patient-aware split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


# ==========================================
# 5. MODEL ARCHITECTURE & TRAINING HYPERPARAMETERS
# ==========================================
LSTM_UNITS = 64
DROPOUT_RATE_1 = 0.30
DENSE_UNITS = 32
DROPOUT_RATE_2 = 0.20
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 3

# Default decision threshold (adjusted via validation tuning if class imbalance shifts ROC curve)
DEFAULT_CLASSIFICATION_THRESHOLD = 0.50
