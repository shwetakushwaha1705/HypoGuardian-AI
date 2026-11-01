# HypoGuardian AI: Automated Testing & Verification Report

## 1. Executive Summary

- **Total Automated Tests**: 16
- **Pass Rate**: 100% (16 Passed, 0 Failed, 0 Skipped)
- **Execution Time**: ~6.95 seconds
- **Test Framework**: Pytest 9.1.1 on Python 3.13.2 (Windows 64-bit)

---

## 2. Test Suite Breakdown

### 2.1 Preprocessing & Data Ingestion (`tests/test_preprocessing.py`)
| Test Function | Target Component | Status | Verified Behavior |
| :--- | :--- | :--- | :--- |
| `test_load_glucose_csv_valid` | `data_loader.py` | **PASSED** | Correctly parses standard CSV format and verifies mandatory headers. |
| `test_load_glucose_csv_missing_columns` | `data_loader.py` | **PASSED** | Detects absent required columns and raises `MissingColumnsError`. |
| `test_load_glucose_csv_empty` | `data_loader.py` | **PASSED** | Raises `EmptyDatasetError` when given empty input files. |
| `test_preprocessing_removes_invalid_and_outliers` | `preprocessing.py` | **PASSED** | Discards unparseable timestamps, non-numeric strings, and values $<20$ or $>500\text{ mg/dL}$. |
| `test_preprocessing_deduplication_and_sorting` | `preprocessing.py` | **PASSED** | Successfully removes duplicate timestamps and sorts chronologically. |

### 2.2 Time-Series Sequence Builder (`tests/test_sequences.py`)
| Test Function | Target Component | Status | Verified Behavior |
| :--- | :--- | :--- | :--- |
| `test_create_hypo_sequences_shape_and_labels` | `sequence_builder.py` | **PASSED** | Verifies $(N, 12, 1)$ shape and correctly assigns label $y=1$ when future glucose $<70\text{ mg/dL}$. |
| `test_create_hypo_sequences_rejects_sensor_gaps` | `sequence_builder.py` | **PASSED** | Automatically discards sequences containing sensor disconnect gaps ($>15\text{ min}$). |
| `test_create_hypo_sequences_patient_boundary_isolation` | `sequence_builder.py` | **PASSED** | Enforces zero cross-patient sequence mixing across patient boundaries. |

### 2.3 Prediction Engine (`tests/test_prediction.py`)
| Test Function | Target Component | Status | Verified Behavior |
| :--- | :--- | :--- | :--- |
| `test_engine_missing_artifacts_handling` | `prediction.py` | **PASSED** | Gracefully handles missing model weights and scaler without crashing. |
| `test_engine_validate_input_short_sequence` | `prediction.py` | **PASSED** | Rejects sequences shorter than 12 steps with clear validation error. |
| `test_engine_validate_input_nan_glucose` | `prediction.py` | **PASSED** | Rejects input windows with NaN or missing values. |
| `test_engine_validate_input_out_of_bounds` | `prediction.py` | **PASSED** | Rejects input values outside clinical physiological bounds. |
| `test_engine_validate_input_valid_array` | `prediction.py` | **PASSED** | Validates valid 12-reading numeric array input. |
| `test_engine_validate_input_dataframe_gap` | `prediction.py` | **PASSED** | Rejects DataFrame input when timestamps contain an irregular 40-minute gap. |

### 2.4 Application Data & Model Artifacts (`tests/test_app_data.py`)
| Test Function | Target Component | Status | Verified Behavior |
| :--- | :--- | :--- | :--- |
| `test_sample_cgm_file_exists_and_valid` | `data/sample/` | **PASSED** | Verifies `synthetic_cgm_sample.csv` exists, parses, and cleans with 0 dropped rows. |
| `test_saved_artifacts_exist_and_consistent` | `models/` | **PASSED** | Confirms `hypo_lstm_model.keras`, `glucose_scaler.pkl`, and `model_metadata.json` match specs. |

---

## 3. Real Held-Out Test Evaluation Results

```
=======================================================
HYPOGUARDIAN AI: HELD-OUT TEST EVALUATION RESULTS
=======================================================
Total Test Samples:      900
Hypo Prevalence:         0.9% (8 positive events)
Accuracy:                0.9878
Balanced Accuracy:       0.6222
Sensitivity (Recall):    0.2500  (Anticipated hypo events)
Specificity:             0.9944  (Correctly identified safe periods)
Precision:               0.2857
F1-Score:                0.2667
ROC-AUC:                 0.9180
True Positives (TP):     2
False Positives (FP):    5 (False alarms)
False Negatives (FN):    6 (Missed hypo events)
True Negatives (TN):     887
=======================================================
```

---

## 4. How to Reproduce Testing Locally

Run the following command from the project root:
```powershell
python -m pytest tests/ -v
```
