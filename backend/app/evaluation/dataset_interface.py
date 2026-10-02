"""Dataset Abstraction Interface & Speaker-Disjoint Validation for VoxGuard.

Defines dataset sample representations, split enumerations (TRAIN, VALIDATION, TEST, OOD),
and explicit validation rules enforcing zero speaker overlap across partitions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Set, Tuple


class DatasetSplit(str, Enum):
    """Dataset partition splits."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    OOD = "ood"                              # Out-Of-Distribution (unseen generators/codecs)


@dataclass
class DatasetSample:
    """Individual dataset sample item metadata."""
    sample_id: str
    audio_path: str
    label: str                              # "human" or "synthetic"
    speaker_id: str                         # Unique speaker identifier for split partitioning
    split: DatasetSplit = DatasetSplit.TEST
    domain_metadata: Dict[str, Any] = field(default_factory=dict)


class AbstractAudioDataset:
    """Dataset container enforcing zero-speaker-overlap validation between train/val/test splits."""

    def __init__(self, name: str, samples: Optional[List[DatasetSample]] = None):
        self.name = name
        self.samples: List[DatasetSample] = samples or []

    def add_sample(self, sample: DatasetSample) -> None:
        """Adds a sample to the dataset container."""
        self.samples.append(sample)

    def validate_speaker_disjointness(self) -> Tuple[bool, Set[str]]:
        """Verifies that no speaker ID present in TRAIN appears in VALIDATION, TEST, or OOD splits.

        Returns:
            Tuple of (is_speaker_disjoint: bool, overlapping_speaker_ids: Set[str])
        """
        train_speakers: Set[str] = set()
        eval_speakers: Set[str] = set()

        for s in self.samples:
            if not s.speaker_id or s.speaker_id == "unknown":
                continue

            if s.split == DatasetSplit.TRAIN:
                train_speakers.add(s.speaker_id)
            elif s.split in (DatasetSplit.VALIDATION, DatasetSplit.TEST, DatasetSplit.OOD):
                eval_speakers.add(s.speaker_id)

        overlapping = train_speakers.intersection(eval_speakers)
        is_disjoint = len(overlapping) == 0

        return is_disjoint, overlapping
