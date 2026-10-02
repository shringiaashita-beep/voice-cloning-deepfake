"""Evaluation Result Schema for VoxGuard.

Defines strongly typed structures for classification performance metrics, confusion matrices,
threshold curves, and speaker-disjoint dataset benchmark reporting without fabricating results.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class ConfusionMatrix:
    """Detailed 2x2 confusion matrix counts for binary classification."""
    tp: int = 0    # True Positives (Synthetic correctly predicted as Synthetic)
    fp: int = 0    # False Positives (Human misclassified as Synthetic)
    tn: int = 0    # True Negatives (Human correctly predicted as Human)
    fn: int = 0    # False Negatives (Synthetic misclassified as Human)

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class BinaryClassificationMetrics:
    """Standard evaluation metrics for audio deepfake classification."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    eer: Optional[float] = None                     # Equal Error Rate (FAR == FRR)
    fpr: Optional[float] = None                     # False Positive Rate
    fnr: Optional[float] = None                     # False Negative Rate

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report for model benchmark runs."""

    dataset_name: str = "unspecified"
    protocol_name: str = "zero_speaker_overlap"
    speaker_disjoint_verified: bool = True
    metrics: Optional[BinaryClassificationMetrics] = None
    confusion_matrix: Optional[ConfusionMatrix] = None
    ood_benchmark_results: Optional[Dict[str, Any]] = None
    evaluation_date: Optional[str] = None
    notes: Optional[str] = "No benchmark evaluation data configured."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "protocol_name": self.protocol_name,
            "speaker_disjoint_verified": self.speaker_disjoint_verified,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "confusion_matrix": self.confusion_matrix.to_dict() if self.confusion_matrix else None,
            "ood_benchmark_results": self.ood_benchmark_results,
            "evaluation_date": self.evaluation_date,
            "notes": self.notes
        }
