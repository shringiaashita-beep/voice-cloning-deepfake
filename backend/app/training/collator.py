"""Batch Collator Engine for VoxGuard Training Pipeline."""

from typing import List
import numpy as np

from app.training.types import TrainingBatch, TrainingSample
from app.training.validation import TrainingBatchValidator, TrainingValidationError


class TrainingBatchCollator:
    """Collates individual TrainingSample records into fixed-shape TrainingBatch objects."""

    def __init__(self, validate_batch: bool = True):
        self.validate_batch = validate_batch
        self.validator = TrainingBatchValidator()

    def collate(self, samples: List[TrainingSample]) -> TrainingBatch:
        """Collates a list of B TrainingSample records into a TrainingBatch.

        Args:
            samples: List of TrainingSample items.

        Returns:
            TrainingBatch object.

        Raises:
            ValueError: If sample list is empty.
            TrainingValidationError: If batch tensors or metadata fail validation.
        """
        if not samples:
            raise ValueError("Cannot collate empty list of TrainingSample records.")

        b_size = len(samples)

        log_mels = np.stack([s.log_mel for s in samples], axis=0).astype(np.float32)
        mfccs = np.stack([s.mfcc for s in samples], axis=0).astype(np.float32)
        centroids = np.stack([s.spectral_centroid for s in samples], axis=0).astype(np.float32)
        rolloffs = np.stack([s.spectral_rolloff for s in samples], axis=0).astype(np.float32)
        labels = np.array([s.label for s in samples], dtype=np.int64)

        metadata = [
            {
                "sample_id": s.sample_id,
                "window_index": s.window_index,
                "speaker_id": s.speaker_id,
                "source": s.source,
                "source_id": s.source_id,
                "split": s.split,
                "label_str": s.label_str
            }
            for s in samples
        ]

        batch = TrainingBatch(
            log_mel=log_mels,
            mfcc=mfccs,
            spectral_centroid=centroids,
            spectral_rolloff=rolloffs,
            labels=labels,
            metadata=metadata
        )

        if self.validate_batch:
            self.validator.validate_batch(batch)

        return batch
