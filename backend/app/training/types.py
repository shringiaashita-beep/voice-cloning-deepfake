"""Typed Representations for VoxGuard Training DataLoader & Batching Engine."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List
import numpy as np


@dataclass
class TrainingSample:
    """Strongly typed representation of a windowed, training-ready feature sample."""
    log_mel: np.ndarray             # Shape: (80, 256), float32
    mfcc: np.ndarray                # Shape: (20, 256), float32
    spectral_centroid: np.ndarray   # Shape: (1, 256), float32
    spectral_rolloff: np.ndarray    # Shape: (1, 256), float32
    label: int                      # Integer code: 0 (human), 1 (synthetic)
    label_str: str                  # Original label string ("human", "synthetic")
    speaker_id: str
    source: str
    source_id: str
    sample_id: str
    window_index: int
    split: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts sample metadata to a JSON-serializable dictionary (excluding raw arrays)."""
        return {
            "sample_id": self.sample_id,
            "window_index": self.window_index,
            "label": self.label,
            "label_str": self.label_str,
            "speaker_id": self.speaker_id,
            "source": self.source,
            "source_id": self.source_id,
            "split": self.split,
            "log_mel_shape": list(self.log_mel.shape),
            "mfcc_shape": list(self.mfcc.shape),
            "spectral_centroid_shape": list(self.spectral_centroid.shape),
            "spectral_rolloff_shape": list(self.spectral_rolloff.shape)
        }


@dataclass
class TrainingBatch:
    """Strongly typed representation of a batched set of TrainingSample tensors."""
    log_mel: np.ndarray             # Shape: (B, 80, 256), float32
    mfcc: np.ndarray                # Shape: (B, 20, 256), float32
    spectral_centroid: np.ndarray   # Shape: (B, 1, 256), float32
    spectral_rolloff: np.ndarray    # Shape: (B, 1, 256), float32
    labels: np.ndarray              # Shape: (B,), int64
    metadata: List[Dict[str, Any]]  # Preserved per-sample metadata dictionary

    @property
    def batch_size(self) -> int:
        """Returns batch size B."""
        return len(self.labels)

    def __getitem__(self, key: str) -> Any:
        """Enables dictionary subscription interface for batch attributes and metadata fields."""
        if key == "speaker_ids":
            return [m["speaker_id"] for m in self.metadata]
        elif key == "sample_ids":
            return [m["sample_id"] for m in self.metadata]
        elif key == "sources":
            return [m["source"] for m in self.metadata]
        elif key == "source_ids":
            return [m["source_id"] for m in self.metadata]
        elif hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"TrainingBatch object has no key or metadata field '{key}'")

    def __contains__(self, key: str) -> bool:
        """Checks if key is a valid attribute or derived metadata field."""
        return key in {"speaker_ids", "sample_ids", "sources", "source_ids"} or hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Safely retrieves key with default fallback."""
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        """Returns batch summary excluding raw array buffers."""
        return {
            "batch_size": self.batch_size,
            "log_mel_shape": list(self.log_mel.shape),
            "mfcc_shape": list(self.mfcc.shape),
            "spectral_centroid_shape": list(self.spectral_centroid.shape),
            "spectral_rolloff_shape": list(self.spectral_rolloff.shape),
            "labels_shape": list(self.labels.shape),
            "metadata": self.metadata
        }
