"""Training Batch & Tensor Validation Engine for VoxGuard.

Validates single samples and collated training batches against target tensor shapes,
dtypes, numerical finiteness (no NaN or Inf), valid integer labels, metadata consistency,
and non-missing speaker/sample IDs.
"""

from typing import Any, Dict, List, Optional
import numpy as np

from app.training.types import TrainingBatch, TrainingSample


class TrainingValidationError(Exception):
    """Raised when training sample or batch validation fails."""
    pass


class TrainingBatchValidator:
    """Validator auditing training samples and collated batches."""

    def __init__(self, target_frames: int = 256):
        self.target_frames = target_frames

    def validate_sample(self, sample: TrainingSample) -> bool:
        """Audits a single TrainingSample for shape, dtype, finiteness, and metadata integrity.

        Args:
            sample: TrainingSample instance.

        Returns:
            True if sample is valid.

        Raises:
            TrainingValidationError: If sample tensor or metadata is invalid.
        """
        if sample is None:
            raise TrainingValidationError("TrainingSample is None.")

        # Check metadata fields
        if not sample.sample_id or not isinstance(sample.sample_id, str):
            raise TrainingValidationError("TrainingSample missing valid sample_id.")

        if not sample.speaker_id or not isinstance(sample.speaker_id, str):
            raise TrainingValidationError(f"TrainingSample '{sample.sample_id}' missing valid speaker_id.")

        if sample.label not in (0, 1):
            raise TrainingValidationError(f"Invalid label integer '{sample.label}'. Must be 0 (human) or 1 (synthetic).")

        tensors = {
            "log_mel": (sample.log_mel, (80, self.target_frames)),
            "mfcc": (sample.mfcc, (20, self.target_frames)),
            "spectral_centroid": (sample.spectral_centroid, (1, self.target_frames)),
            "spectral_rolloff": (sample.spectral_rolloff, (1, self.target_frames)),
        }

        for name, (arr, expected_shape) in tensors.items():
            if arr is None or not isinstance(arr, np.ndarray):
                raise TrainingValidationError(f"Feature '{name}' in sample '{sample.sample_id}' is not a NumPy array.")

            if str(arr.dtype) != "float32":
                raise TrainingValidationError(
                    f"Feature '{name}' has invalid dtype '{arr.dtype}'. Expected 'float32'."
                )

            if arr.shape != expected_shape:
                raise TrainingValidationError(
                    f"Feature '{name}' has invalid shape {arr.shape}. Expected {expected_shape}."
                )

            if np.isnan(arr).any():
                raise TrainingValidationError(f"Feature '{name}' in sample '{sample.sample_id}' contains NaN values.")

            if np.isinf(arr).any():
                raise TrainingValidationError(f"Feature '{name}' in sample '{sample.sample_id}' contains Infinity values.")

        return True

    def validate_batch(self, batch: TrainingBatch) -> bool:
        """Audits a collated TrainingBatch.

        Args:
            batch: TrainingBatch instance.

        Returns:
            True if batch is valid.

        Raises:
            TrainingValidationError: If batch tensors or metadata are invalid.
        """
        if batch is None:
            raise TrainingValidationError("TrainingBatch is None.")

        b_size = batch.batch_size
        if b_size == 0:
            raise TrainingValidationError("TrainingBatch is empty (batch_size = 0).")

        if len(batch.metadata) != b_size:
            raise TrainingValidationError(
                f"Batch metadata length ({len(batch.metadata)}) does not match batch size ({b_size})."
            )

        # Check metadata fields per sample
        for idx, meta in enumerate(batch.metadata):
            if not meta.get("sample_id"):
                raise TrainingValidationError(f"Batch metadata at index {idx} missing 'sample_id'.")
            if not meta.get("speaker_id"):
                raise TrainingValidationError(f"Batch metadata at index {idx} missing 'speaker_id'.")

        # Check label tensor
        if batch.labels is None or not isinstance(batch.labels, np.ndarray):
            raise TrainingValidationError("Batch labels tensor is missing or not a NumPy array.")

        if batch.labels.shape != (b_size,):
            raise TrainingValidationError(
                f"Batch labels shape mismatch: expected ({b_size},), got {batch.labels.shape}."
            )

        for lbl in batch.labels:
            if int(lbl) not in (0, 1):
                raise TrainingValidationError(f"Batch contains invalid label integer '{lbl}'. Must be 0 or 1.")

        # Check feature tensors
        expected_batch_shapes = {
            "log_mel": (batch.log_mel, (b_size, 80, self.target_frames)),
            "mfcc": (batch.mfcc, (b_size, 20, self.target_frames)),
            "spectral_centroid": (batch.spectral_centroid, (b_size, 1, self.target_frames)),
            "spectral_rolloff": (batch.spectral_rolloff, (b_size, 1, self.target_frames)),
        }

        for name, (arr, expected_shape) in expected_batch_shapes.items():
            if arr is None or not isinstance(arr, np.ndarray):
                raise TrainingValidationError(f"Batch feature '{name}' is missing or not a NumPy array.")

            if str(arr.dtype) != "float32":
                raise TrainingValidationError(
                    f"Batch feature '{name}' has invalid dtype '{arr.dtype}'. Expected 'float32'."
                )

            if arr.shape != expected_shape:
                raise TrainingValidationError(
                    f"Batch feature '{name}' has shape {arr.shape}. Expected {expected_shape}."
                )

            if np.isnan(arr).any():
                raise TrainingValidationError(f"Batch feature '{name}' contains NaN values.")

            if np.isinf(arr).any():
                raise TrainingValidationError(f"Batch feature '{name}' contains Infinity values.")

        return True
