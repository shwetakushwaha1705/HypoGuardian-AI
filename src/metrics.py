"""HypoGuardian AI: Clinical & Machine Learning Evaluation Metrics.

Computes comprehensive classification metrics tailored for medical early-warning systems,
including Sensitivity (Recall), Specificity (True Negative Rate), Precision (PPV),
ROC-AUC, Precision-Recall AUC, and confusion matrix statistics.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
)


@dataclass
class EvaluationMetrics:
    """Encapsulates all evaluation metrics for model assessment and reporting."""
    total_samples: int
    positive_samples: int
    negative_samples: int
    prevalence: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    balanced_accuracy: float
    sensitivity_recall: float
    specificity: float
    precision: float
    f1_score: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    classification_threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_clinical_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> EvaluationMetrics:
    """Computes clinical and statistical performance metrics for binary hypoglycemia prediction.

    Args:
        y_true: Ground truth binary labels (0 = normal, 1 = hypoglycemia).
        y_prob: Predicted uncalibrated probability or risk score in range [0, 1].
        threshold: Decision threshold for positive classification (default: 0.50).

    Returns:
        EvaluationMetrics dataclass instance containing all performance scores.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    total_samples = len(y_true_arr)
    positives = int(np.sum(y_true_arr == 1))
    negatives = int(np.sum(y_true_arr == 0))
    prevalence = float(positives / total_samples) if total_samples > 0 else 0.0

    # Handle single-class edge cases in test batches
    unique_classes = np.unique(y_true_arr)
    if len(unique_classes) < 2:
        # Cannot compute meaningful ROC/PR curves if only one class exists
        cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        return EvaluationMetrics(
            total_samples=total_samples,
            positive_samples=positives,
            negative_samples=negatives,
            prevalence=round(prevalence, 4),
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            accuracy=round(float(accuracy_score(y_true_arr, y_pred)), 4),
            balanced_accuracy=round(float(balanced_accuracy_score(y_true_arr, y_pred)), 4),
            sensitivity_recall=round(float(recall_score(y_true_arr, y_pred, zero_division=0)), 4),
            specificity=round(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0, 4),
            precision=round(float(precision_score(y_true_arr, y_pred, zero_division=0)), 4),
            f1_score=round(float(f1_score(y_true_arr, y_pred, zero_division=0)), 4),
            roc_auc=None,
            pr_auc=None,
            classification_threshold=threshold,
        )

    # Standard two-class evaluation
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Clinical metrics
    sens = float(recall_score(y_true_arr, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred))

    # ROC AUC
    try:
        roc_val = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        roc_val = None

    # Precision-Recall AUC (critical for imbalanced medical data)
    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        pr_val = float(auc(recall_curve, precision_curve))
    except Exception:
        pr_val = None

    return EvaluationMetrics(
        total_samples=total_samples,
        positive_samples=positives,
        negative_samples=negatives,
        prevalence=round(prevalence, 4),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        accuracy=round(acc, 4),
        balanced_accuracy=round(bal_acc, 4),
        sensitivity_recall=round(sens, 4),
        specificity=round(spec, 4),
        precision=round(prec, 4),
        f1_score=round(f1, 4),
        roc_auc=round(roc_val, 4) if roc_val is not None else None,
        pr_auc=round(pr_val, 4) if pr_val is not None else None,
        classification_threshold=threshold,
    )
