"""Deterministic Training Sampler & DataLoader Engine for VoxGuard.

Provides speaker-safe deterministic batch sampling and iteration for train, validation,
test, and OOD splits without cross-split leakage or unseeded randomness.
"""

import math
import random
from typing import Generator, List, Optional
import numpy as np

from app.training.config import TrainingConfig
from app.training.collator import TrainingBatchCollator
from app.training.dataset import FeatureDataset
from app.training.types import TrainingBatch, TrainingSample

try:
    import torch
    from torch.utils.data import Sampler as BaseSampler, DataLoader as PyTorchDataLoader
    HAS_TORCH = True
except ImportError:
    BaseSampler = object
    PyTorchDataLoader = object
    HAS_TORCH = False


class DeterministicSampler(BaseSampler):
    """Deterministic index sampler supporting seeded shuffling for training and sequential sampling for evaluation."""

    def __init__(
        self,
        dataset_size: int,
        shuffle: bool = False,
        seed: int = 42
    ):
        self.dataset_size = dataset_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = list(range(self.dataset_size))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.dataset_size


class TrainingDataLoader:
    """Deterministic batch data loader yielding TrainingBatch objects."""

    def __init__(
        self,
        dataset: FeatureDataset,
        config: Optional[TrainingConfig] = None,
        batch_size: Optional[int] = None,
        shuffle: Optional[bool] = None,
        seed: Optional[int] = None
    ):
        self.dataset = dataset
        self.config = config or TrainingConfig()

        self.batch_size = batch_size if batch_size is not None else self.config.BATCH_SIZE
        self.seed = seed if seed is not None else self.config.RANDOM_SEED

        # Split-dependent shuffle default
        if shuffle is not None:
            self.shuffle = shuffle
        else:
            if self.dataset.split == "train":
                self.shuffle = self.config.SHUFFLE_TRAIN
            elif self.dataset.split == "validation":
                self.shuffle = self.config.SHUFFLE_VALIDATION
            elif self.dataset.split == "test":
                self.shuffle = self.config.SHUFFLE_TEST
            else:
                self.shuffle = self.config.SHUFFLE_OOD

        self.collator = TrainingBatchCollator()
        self.sampler = DeterministicSampler(
            dataset_size=len(self.dataset),
            shuffle=self.shuffle,
            seed=self.seed
        )

    def __len__(self) -> int:
        """Returns total number of batches."""
        if len(self.dataset) == 0:
            return 0
        if self.config.DROP_LAST:
            return len(self.dataset) // self.batch_size
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Generator[TrainingBatch, None, None]:
        """Iterates over batches deterministically."""
        indices = list(self.sampler)
        num_samples = len(indices)

        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]

            if self.config.DROP_LAST and len(batch_indices) < self.batch_size:
                break

            samples = [self.dataset[idx] for idx in batch_indices]
            batch = self.collator.collate(samples)
            yield batch
