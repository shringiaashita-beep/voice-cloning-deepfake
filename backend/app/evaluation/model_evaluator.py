"""Model Evaluator Service for VoxGuard.

Computes comprehensive binary classification metrics (Accuracy, Balanced Accuracy, Precision,
Recall/Sensitivity, Specificity, F1-Score, ROC-AUC, PR-AUC, Confusion Matrix) strictly enforcing
the convention: human = negative class (0), synthetic = positive class (1).
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EvaluationMetrics:
    """Strongly typed metric evaluation results."""
    split: str
    sample_count: int
    human_count: int
    synthetic_count: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    specificity: float
    f1_score: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    confusion_matrix: List[List[int]]  # [[TN, FP], [FN, TP]]
    class_labels: List[str] = field(default_factory=lambda: ["human", "synthetic"])
    threshold: float = 0.5
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts metrics to clean dictionary representation."""
        return asdict(self)


class ModelEvaluator:
    """Evaluates prediction probabilities against ground truth labels."""

    def evaluate(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        split: str = "validation",
        threshold: float = 0.5
    ) -> EvaluationMetrics:
        """Computes complete suite of evaluation metrics for binary classification.

        Args:
            y_true: 1D array of ground truth labels (0 for human, 1 for synthetic).
            y_probs: 1D array of synthetic class probabilities (P(synthetic)).
            split: Dataset split identifier ("train", "validation", "test", "ood").
            threshold: Decision threshold for positive class prediction (default 0.5).

        Returns:
            EvaluationMetrics object.
        """
        y_true = np.asarray(y_true, dtype=np.int64)
        y_probs = np.asarray(y_probs, dtype=np.float64)

        sample_count = len(y_true)
        if sample_count == 0:
            return EvaluationMetrics(
                split=split,
                sample_count=0,
                human_count=0,
                synthetic_count=0,
                accuracy=0.0,
                balanced_accuracy=0.0,
                precision=0.0,
                recall=0.0,
                specificity=0.0,
                f1_score=0.0,
                roc_auc=None,
                pr_auc=None,
                confusion_matrix=[[0, 0], [0, 0]],
                threshold=threshold,
                notes="Split contains 0 samples."
            )

        human_count = int(np.sum(y_true == 0))
        synthetic_count = int(np.sum(y_true == 1))

        # Binary predictions based on threshold
        y_pred = (y_probs >= threshold).astype(np.int64)

        # Confusion matrix components (0=human/negative, 1=synthetic/positive)
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        confusion_matrix = [[tn, fp], [fn, tp]]

        accuracy = float((tp + tn) / sample_count) if sample_count > 0 else 0.0

        # Sensitivity / Recall
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # Specificity
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        balanced_accuracy = float((recall + specificity) / 2.0)

        # Precision
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

        # F1 Score
        f1_score = float((2 * precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0.0

        # Compute ROC-AUC and PR-AUC using numerical trapezoidal integration if both classes exist
        roc_auc = None
        pr_auc = None
        notes = None

        if human_count > 0 and synthetic_count > 0:
            roc_auc = self._compute_roc_auc(y_true, y_probs)
            pr_auc = self._compute_pr_auc(y_true, y_probs)
        else:
            notes = f"ROC-AUC and PR-AUC undefined: split contains only single class (human={human_count}, synthetic={synthetic_count})."

        return EvaluationMetrics(
            split=split,
            sample_count=sample_count,
            human_count=human_count,
            synthetic_count=synthetic_count,
            accuracy=round(accuracy, 4),
            balanced_accuracy=round(balanced_accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            specificity=round(specificity, 4),
            f1_score=round(f1_score, 4),
            roc_auc=round(roc_auc, 4) if roc_auc is not None else None,
            pr_auc=round(pr_auc, 4) if pr_auc is not None else None,
            confusion_matrix=confusion_matrix,
            threshold=threshold,
            notes=notes
        )

    def _trapz(self, y: np.ndarray, x: np.ndarray) -> float:
        """Computes trapezoidal numerical integration robustly across NumPy versions."""
        if hasattr(np, "trapezoid"):
            return float(np.trapezoid(y, x))
        elif hasattr(np, "trapz"):
            return float(getattr(np, "trapz")(y, x))
        else:
            return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))

    def _compute_roc_auc(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """Computes ROC-AUC via threshold sorting."""
        desc_indices = np.argsort(-y_probs)
        y_true_sorted = y_true[desc_indices]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        tps = np.cumsum(y_true_sorted == 1)
        fps = np.cumsum(y_true_sorted == 0)

        tpr = tps / n_pos
        fpr = fps / n_neg

        tpr = np.concatenate(([0.0], tpr))
        fpr = np.concatenate(([0.0], fpr))

        return self._trapz(tpr, fpr)

    def _compute_pr_auc(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """Computes PR-AUC via precision-recall curve trapezoidal integration."""
        desc_indices = np.argsort(-y_probs)
        y_true_sorted = y_true[desc_indices]

        n_pos = np.sum(y_true == 1)
        tps = np.cumsum(y_true_sorted == 1)
        fps = np.cumsum(y_true_sorted == 0)

        precisions = tps / (tps + fps)
        recalls = tps / n_pos

        precisions = np.concatenate(([1.0], precisions))
        recalls = np.concatenate(([0.0], recalls))

        return self._trapz(precisions, recalls)
