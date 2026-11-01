"""HypoGuardian AI: Model Evaluation Module.

Evaluates the trained LSTM model on the held-out test set.
Computes confusion matrix, sensitivity, specificity, ROC-AUC, and PR-AUC.
Generates evaluation charts and saves structured results to models/evaluation_results.json.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from config import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    EVALUATION_RESULTS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    SCALER_PATH,
)
from src.metrics import calculate_clinical_metrics


def evaluate_trained_model(
    model_path: Path = MODEL_PATH,
    test_cache_path: Path = MODELS_DIR / "test_split_cache.npz",
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> dict:
    """Evaluates the saved LSTM model on held-out test data and exports results and plots."""
    import tensorflow as tf

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}. Train the model first.")

    if not test_cache_path.exists():
        raise FileNotFoundError(
            f"Held-out test split cache not found at: {test_cache_path}. "
            "Please run `python -m src.train_model` to generate test data and model."
        )

    print(f"-> Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"-> Loading held-out test split from: {test_cache_path}")
    test_data = np.load(test_cache_path)
    X_test = test_data["X_test"]
    y_test = test_data["y_test"]

    print(f"-> Running test inference on {len(X_test)} samples...")
    raw_preds = model.predict(X_test, verbose=0).ravel()

    # Calculate comprehensive metrics
    eval_metrics = calculate_clinical_metrics(y_true=y_test, y_prob=raw_preds, threshold=threshold)
    results_dict = eval_metrics.to_dict()

    print("\n" + "=" * 55)
    print("HYPOGUARDIAN AI: HELD-OUT TEST EVALUATION RESULTS")
    print("=" * 55)
    print(f"Total Test Samples:      {results_dict['total_samples']}")
    print(f"Hypo Prevalence:         {results_dict['prevalence']:.1%}")
    print(f"Accuracy:                {results_dict['accuracy']:.4f}")
    print(f"Balanced Accuracy:       {results_dict['balanced_accuracy']:.4f}")
    print(f"Sensitivity (Recall):    {results_dict['sensitivity_recall']:.4f}  (Hypo events anticipated)")
    print(f"Specificity:             {results_dict['specificity']:.4f}  (Safe periods correctly identified)")
    print(f"Precision:               {results_dict['precision']:.4f}")
    print(f"F1-Score:                {results_dict['f1_score']:.4f}")
    print(f"ROC-AUC:                 {results_dict['roc_auc']}")
    print(f"PR-AUC:                  {results_dict['pr_auc']}")
    print(f"True Positives (TP):     {results_dict['true_positives']}")
    print(f"False Positives (FP):    {results_dict['false_positives']} (False alarms)")
    print(f"False Negatives (FN):    {results_dict['false_negatives']} (Missed hypo events)")
    print(f"True Negatives (TN):     {results_dict['true_negatives']}")
    print("=" * 55 + "\n")

    # Save evaluation results as JSON
    results_payload = {
        "metrics": results_dict,
        "confusion_matrix": [
            [results_dict["true_negatives"], results_dict["false_positives"]],
            [results_dict["false_negatives"], results_dict["true_positives"]],
        ],
        "test_samples_count": len(X_test),
        "threshold": threshold,
    }

    # Save ROC curve points if two classes present
    if results_dict["roc_auc"] is not None:
        fpr, tpr, _ = roc_curve(y_test, raw_preds)
        results_payload["roc_curve"] = {
            "fpr": [float(x) for x in fpr[::max(1, len(fpr) // 50)]],
            "tpr": [float(x) for x in tpr[::max(1, len(tpr) // 50)]],
        }

    # Save PR curve points if two classes present
    if results_dict["pr_auc"] is not None:
        precisions, recalls, _ = precision_recall_curve(y_test, raw_preds)
        results_payload["pr_curve"] = {
            "recalls": [float(x) for x in recalls[::max(1, len(recalls) // 50)]],
            "precisions": [float(x) for x in precisions[::max(1, len(precisions) // 50)]],
        }

    with open(EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"-> Evaluation results saved to: {EVALUATION_RESULTS_PATH}")

    # Generate Matplotlib plots for report documentation
    _save_evaluation_plots(y_test, raw_preds, results_dict)

    return results_payload


def _save_evaluation_plots(y_true: np.ndarray, y_prob: np.ndarray, metrics: dict) -> None:
    """Renders and saves PNG figures for the project documentation."""
    # 1. Confusion Matrix
    cm = np.array([
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]],
    ])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    classes = ["Normal (0)", "Hypo (1)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Held-out Test Confusion Matrix",
        ylabel="True Observation",
        xlabel="Predicted Label",
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    fig.tight_layout()
    fig.savefig(MODELS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 2. ROC & PR Curves
    if metrics.get("roc_auc") is not None and metrics.get("pr_auc") is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        precision_c, recall_c, _ = precision_recall_curve(y_true, y_prob)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

        # ROC
        ax1.plot(fpr, tpr, color="#2563eb", lw=2, label=f"ROC (AUC = {metrics['roc_auc']:.2f})")
        ax1.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate (Sensitivity)")
        ax1.set_title("Receiver Operating Characteristic")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # PR
        ax2.plot(recall_c, precision_c, color="#059669", lw=2, label=f"PR (AUC = {metrics['pr_auc']:.2f})")
        ax2.axhline(y=metrics["prevalence"], color="gray", lw=1, linestyle="--", label=f"Baseline ({metrics['prevalence']:.1%})")
        ax2.set_xlabel("Recall (Sensitivity)")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(MODELS_DIR / "roc_pr_curves.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    evaluate_trained_model()
