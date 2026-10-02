"""Abstract Classifier Interface for VoxGuard.

Defines the contract for all ML deepfake classification models, baseline runners,
and model registries. Enforces typed output structures, model metadata, provenance,
and status reporting without fabricating unverified probabilities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Optional

from app.features.extractor import AudioFeatures


class PredictionLabel(str, Enum):
    """Standard classification decision labels."""
    HUMAN = "human"
    SYNTHETIC = "synthetic"
    UNCERTAIN = "uncertain"


class ModelStatus(str, Enum):
    """Operational state of the classifier engine."""
    READY = "ready"                        # Valid trained model with evaluated weights
    ANALYSIS_ONLY = "analysis_only"        # Baseline / statistical acoustic analysis
    NOT_READY = "not_ready"                # Model weights uninitialized or unavailable


@dataclass
class ModelMetadata:
    """Strongly typed model metadata structure for provenance tracking."""
    model_name: str
    model_version: str
    framework: str                             # e.g., "pytorch", "onnx", "statistical_baseline"
    input_sample_rate: int = 16000
    input_channels: int = 1
    model_source: Optional[str] = "not_configured"
    license: Optional[str] = "not_configured"
    weights_location: Optional[str] = "not_configured"
    evaluation_dataset: Optional[str] = None
    evaluation_protocol: Optional[str] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Converts model metadata object to dictionary."""
        return asdict(self)


@dataclass
class ClassificationResult:
    """Strongly typed classification result returned by classifiers and baseline runners."""
    label: PredictionLabel
    probabilities: Optional[Dict[str, Optional[float]]]  # {"human": prob, "synthetic": prob} or None
    confidence_score: Optional[float]                    # Calibrated confidence (0.0 to 1.0) or None
    model_name: str
    model_version: str
    status: ModelStatus
    metadata: Dict[str, Any]                             # Timing, device, disclaimers, acoustic stats

    def __post_init__(self):
        """Enforces canonical probability invariants and label consistency upon instantiation."""
        import math
        if self.status != ModelStatus.READY:
            if self.probabilities is not None:
                self.probabilities = {"human": None, "synthetic": None}
            self.confidence_score = None
            if self.label != PredictionLabel.UNCERTAIN:
                self.label = PredictionLabel.UNCERTAIN
        elif self.probabilities is not None:
            p_h = self.probabilities.get("human")
            p_s = self.probabilities.get("synthetic")

            # Check for NaN / Infinity
            if p_h is not None and (math.isnan(float(p_h)) or math.isinf(float(p_h))):
                raise ValueError("Human probability cannot be NaN or Infinity")
            if p_s is not None and (math.isnan(float(p_s)) or math.isinf(float(p_s))):
                raise ValueError("Synthetic probability cannot be NaN or Infinity")

            # Complete missing complement probability if needed
            if p_s is not None and p_h is None:
                p_h = round(1.0 - float(p_s), 4)
            elif p_h is not None and p_s is None:
                p_s = round(1.0 - float(p_h), 4)

            if p_h is not None and p_s is not None:
                p_h = max(0.0, min(1.0, float(p_h)))
                p_s = max(0.0, min(1.0, float(p_s)))

                # Enforce invariant: synthetic + human = 1.0
                total = p_h + p_s
                if abs(total - 1.0) > 1e-5:
                    p_s = round(p_s / total, 4)
                    p_h = round(1.0 - p_s, 4)

                self.probabilities = {"human": p_h, "synthetic": p_s}

                # Enforce label consistency with probabilities
                if p_s > 0.5:
                    expected_label = PredictionLabel.SYNTHETIC
                elif p_h > 0.5:
                    expected_label = PredictionLabel.HUMAN
                else:
                    expected_label = PredictionLabel.UNCERTAIN

                if self.label != expected_label and self.label != PredictionLabel.UNCERTAIN:
                    self.label = expected_label

    def to_dict(self) -> Dict[str, Any]:
        """Converts result object to clean dictionary for REST API serialization."""
        return {
            "label": self.label.value,
            "probabilities": self.probabilities,
            "confidence_score": self.confidence_score,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "status": self.status.value,
            "metadata": self.metadata
        }


class AbstractDeepfakeClassifier(ABC):
    """Abstract base class for all deepfake audio classification engines."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the human-readable model identifier."""
        pass

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Returns the semantic version string of the model."""
        pass

    @property
    @abstractmethod
    def status(self) -> ModelStatus:
        """Returns current operational status of the model engine."""
        pass

    @abstractmethod
    def get_model_metadata(self) -> ModelMetadata:
        """Returns structured metadata regarding framework, provenance, license, and metrics."""
        pass

    @abstractmethod
    def predict(self, features: AudioFeatures) -> ClassificationResult:
        """Executes model inference on extracted audio features.

        Args:
            features: AudioFeatures object containing extracted NumPy feature arrays.

        Returns:
            ClassificationResult object containing verdict, probabilities, and metadata.
        """
        pass
