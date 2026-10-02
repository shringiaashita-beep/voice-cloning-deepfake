"""Speaker Split Leakage Validation Service for VoxGuard.

Enforces zero-speaker-overlap partitioning across supervised splits (train, validation, test, ood).
Identifies speaker leakage, reports unique speaker distributions, and produces structured audit reports.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from app.evaluation.manifest_validator import ManifestRow

logger = logging.getLogger("voxguard.speaker_split_validator")


@dataclass
class SpeakerSplitReport:
    """Strongly typed report evaluating speaker-disjoint dataset integrity."""
    is_speaker_disjoint: bool
    total_speakers: int
    speakers_per_split: Dict[str, int]
    overlapping_speakers: Dict[str, List[str]]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_speaker_disjoint": self.is_speaker_disjoint,
            "total_speakers": self.total_speakers,
            "speakers_per_split": self.speakers_per_split,
            "overlapping_speakers": self.overlapping_speakers,
            "errors": self.errors,
            "warnings": self.warnings
        }


class SpeakerSplitValidator:
    """Validator auditing speaker-disjoint dataset split policies."""

    def validate_speaker_splits(self, rows: List[ManifestRow]) -> SpeakerSplitReport:
        """Audits a list of ManifestRow entries for speaker data leakage across splits.

        Args:
            rows: List of validated ManifestRow entries.

        Returns:
            SpeakerSplitReport detailing total speakers, split counts, and leakage overlaps.
        """
        split_speakers: Dict[str, Set[str]] = {
            "train": set(),
            "validation": set(),
            "test": set(),
            "ood": set()
        }

        all_speakers: Set[str] = set()

        for r in rows:
            spk = r.speaker_id.strip()
            split = r.split.strip().lower()
            
            if spk and spk != "unknown":
                all_speakers.add(spk)
                if split in split_speakers:
                    split_speakers[split].add(spk)

        # Check pairwise overlaps
        overlapping: Dict[str, List[str]] = {}
        errors: List[str] = []

        # 1. Train vs Validation
        train_val = list(split_speakers["train"].intersection(split_speakers["validation"]))
        if train_val:
            overlapping["train_and_validation"] = sorted(train_val)
            errors.append(f"Speaker leakage detected between TRAIN and VALIDATION splits: {sorted(train_val)}")

        # 2. Train vs Test
        train_test = list(split_speakers["train"].intersection(split_speakers["test"]))
        if train_test:
            overlapping["train_and_test"] = sorted(train_test)
            errors.append(f"Speaker leakage detected between TRAIN and TEST splits: {sorted(train_test)}")

        # 3. Train vs OOD
        train_ood = list(split_speakers["train"].intersection(split_speakers["ood"]))
        if train_ood:
            overlapping["train_and_ood"] = sorted(train_ood)
            errors.append(f"Speaker leakage detected between TRAIN and OOD splits: {sorted(train_ood)}")

        # 4. Validation vs Test
        val_test = list(split_speakers["validation"].intersection(split_speakers["test"]))
        if val_test:
            overlapping["validation_and_test"] = sorted(val_test)
            errors.append(f"Speaker leakage detected between VALIDATION and TEST splits: {sorted(val_test)}")

        speakers_per_split = {s: len(spks) for s, spks in split_speakers.items()}
        is_disjoint = len(errors) == 0

        return SpeakerSplitReport(
            is_speaker_disjoint=is_disjoint,
            total_speakers=len(all_speakers),
            speakers_per_split=speakers_per_split,
            overlapping_speakers=overlapping,
            errors=errors
        )
