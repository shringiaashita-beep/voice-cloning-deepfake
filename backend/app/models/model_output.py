"""Model Output Contract Specification for VoxGuard.

Defines a strongly typed output contract for evaluated classifier predictions,
enforcing probability summation checks, status checks, provenance tags, and
guaranteeing that uninitialized models return null probabilities without fabrication.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from app.models.base_classifier import ModelStatus, PredictionLabel


@dataclass
class ModelOutputContract:
    """Strongly typed output contract representation for ML classifier decisions."""

    label: PredictionLabel
    human_probability: Optional[float]
    synthetic_probability: Optional[float]
    confidence_score: Optional[float]
    model_name: str
    model_version: str
    model_type: str
    inference_time_ms: float
    device: str = "cpu"
    provenance: str = "ml_model"
    evaluation_status: ModelStatus = ModelStatus.NOT_READY

    def __post_init__(self):
        """Enforces probability safety rules upon instantiation."""
        import math
        # 1. Probabilities MUST be absent/null unless status is READY
        if self.evaluation_status != ModelStatus.READY:
            self.human_probability = None
            self.synthetic_probability = None
            self.confidence_score = None
            if self.label != PredictionLabel.UNCERTAIN:
                self.label = PredictionLabel.UNCERTAIN

        # 2. When status is READY, probabilities must sum to 1.0 (tolerance < 1e-5)
        elif self.human_probability is not None or self.synthetic_probability is not None:
            if self.human_probability is not None and (math.isnan(float(self.human_probability)) or math.isinf(float(self.human_probability))):
                raise ValueError("Human probability cannot be NaN or Infinity")
            if self.synthetic_probability is not None and (math.isnan(float(self.synthetic_probability)) or math.isinf(float(self.synthetic_probability))):
                raise ValueError("Synthetic probability cannot be NaN or Infinity")

            if self.synthetic_probability is not None and self.human_probability is None:
                self.human_probability = round(1.0 - float(self.synthetic_probability), 4)
            elif self.human_probability is not None and self.synthetic_probability is None:
                self.synthetic_probability = round(1.0 - float(self.human_probability), 4)

            if self.human_probability is not None and self.synthetic_probability is not None:
                self.human_probability = max(0.0, min(1.0, float(self.human_probability)))
                self.synthetic_probability = max(0.0, min(1.0, float(self.synthetic_probability)))

                total = self.human_probability + self.synthetic_probability
                if abs(total - 1.0) > 1e-5:
                    self.synthetic_probability = round(self.synthetic_probability / total, 4)
                    self.human_probability = round(1.0 - self.synthetic_probability, 4)

                # Ensure label agreement
                if self.synthetic_probability > 0.5:
                    self.label = PredictionLabel.SYNTHETIC
                elif self.human_probability > 0.5:
                    self.label = PredictionLabel.HUMAN
                else:
                    self.label = PredictionLabel.UNCERTAIN

    def to_dict(self) -> Dict[str, Any]:
        """Converts output object to dictionary for API serialization."""
        return {
            "label": self.label.value,
            "probabilities": {
                "human": self.human_probability,
                "synthetic": self.synthetic_probability,
            },
            "confidence_score": self.confidence_score,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "inference_time_ms": self.inference_time_ms,
            "device": self.device,
            "provenance": self.provenance,
            "evaluation_status": self.evaluation_status.value,
        }
