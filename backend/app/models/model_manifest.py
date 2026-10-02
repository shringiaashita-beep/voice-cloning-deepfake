"""Model Manifest Specification for VoxGuard.

Defines a structured manifest descriptor for external trained model artifacts,
recording model metadata, expected input shape, class labels, checksums,
training/evaluation dataset provenance, and calibration status without fabricating metrics.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ModelManifest:
    """Descriptor manifest for an externally supplied deepfake detection model artifact."""

    model_name: str
    model_version: str
    model_type: str = "pytorch"                      # "pytorch", "torchscript", "onnx"
    architecture: str = "resnet18_logmel"           # e.g., "resnet18", "rawnet2", "wav2vec2"
    framework: str = "torch"                        # "torch", "onnxruntime"
    expected_input: Dict[str, Any] = field(default_factory=lambda: {
        "sample_rate": 16000,
        "channels": 1,
        "feature_type": "log_mel_spectrogram",
        "freq_bins": 80
    })
    expected_sample_rate: int = 16000
    expected_channels: int = 1
    class_names: List[str] = field(default_factory=lambda: ["human", "synthetic"])
    checkpoint_path: Optional[str] = None
    checksum: Optional[str] = None                  # SHA-256 hash string
    training_dataset: Optional[str] = None
    training_dataset_version: Optional[str] = None
    evaluation_dataset: Optional[str] = None
    evaluation_protocol: Optional[str] = None
    speaker_overlap_policy: str = "speaker_disjoint_mandatory"
    metrics: Optional[Dict[str, Any]] = None         # Set to None / null if un-evaluated
    calibration_method: str = "not_calibrated"      # "temperature_scaling", "platt", "not_calibrated"
    threshold: float = 0.5
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelManifest":
        """Constructs a ModelManifest instance from a dictionary representation."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """Converts manifest object to clean dictionary for serialization."""
        return asdict(self)
