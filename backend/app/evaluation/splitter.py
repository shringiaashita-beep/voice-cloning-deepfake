"""Deterministic Speaker-Level Dataset Splitter for VoxGuard.

Partitions dataset CSV manifests strictly by Speaker ID (not individual frames or files)
using deterministic pseudo-random seed shuffling (default seed=42).
Guarantees 0% speaker leakage between train, validation, and test splits.
"""

import csv
import os
import random
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from app.evaluation.manifest_validator import ManifestRow, ManifestValidator
from app.evaluation.speaker_split_validator import SpeakerSplitValidator

logger = logging.getLogger("voxguard.splitter")


class DeterministicSpeakerSplitter:
    """Splitter partitioning dataset entries deterministically by Speaker ID."""

    def __init__(self, validator: Optional[ManifestValidator] = None):
        self.validator = validator or ManifestValidator()
        self.speaker_validator = SpeakerSplitValidator()

    def split_manifest(
        self,
        manifest_path: str,
        output_path: Optional[str] = None,
        train_ratio: float = 0.70,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> List[ManifestRow]:
        """Partitions dataset rows deterministically by Speaker ID.

        Args:
            manifest_path: Path to source manifest CSV file.
            output_path: Optional path to write newly partitioned manifest CSV.
            train_ratio: Target proportion of speakers assigned to TRAIN (default 0.70).
            validation_ratio: Target proportion of speakers assigned to VALIDATION (default 0.15).
            test_ratio: Target proportion of speakers assigned to TEST (default 0.15).
            seed: Fixed pseudo-random seed integer (default 42).

        Returns:
            List of updated ManifestRow entries.
        """
        # Ratio validation
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Ratios must sum to 1.0. Got train={train_ratio}, val={validation_ratio}, test={test_ratio} (sum={total_ratio}).")

        val_res = self.validator.validate_csv_manifest(manifest_path)
        if not val_res.is_valid or not val_res.valid_rows:
            raise ValueError(f"Cannot split invalid CSV manifest: {val_res.errors}")

        rows = val_res.valid_rows

        # 1. Group rows by speaker_id (separate OOD rows which preserve their 'ood' split)
        speaker_map: Dict[str, List[ManifestRow]] = {}
        ood_rows: List[ManifestRow] = []

        for r in rows:
            if r.split == "ood":
                ood_rows.append(r)
            else:
                spk = r.speaker_id.strip()
                if spk not in speaker_map:
                    speaker_map[spk] = []
                speaker_map[spk].append(r)

        unique_speakers = sorted(list(speaker_map.keys()))
        if len(unique_speakers) < 3:
            raise ValueError(f"Insufficient unique speakers ({len(unique_speakers)}) to create train/val/test splits. Need at least 3 unique speakers.")

        # 2. Deterministically shuffle unique speaker IDs using fixed seed
        rng = random.Random(seed)
        shuffled_speakers = list(unique_speakers)
        rng.shuffle(shuffled_speakers)

        # 3. Partition speaker IDs into train, val, test buckets
        n_speakers = len(shuffled_speakers)
        n_train = max(1, int(round(n_speakers * train_ratio)))
        n_val = max(1, int(round(n_speakers * validation_ratio)))
        
        # Ensure at least 1 speaker per split if n_speakers >= 3
        if n_train + n_val >= n_speakers:
            n_train = max(1, n_speakers - 2)
            n_val = 1

        train_spks = set(shuffled_speakers[:n_train])
        val_spks = set(shuffled_speakers[n_train:n_train + n_val])
        test_spks = set(shuffled_speakers[n_train + n_val:])

        if not test_spks and len(val_spks) > 1:
            # Shift 1 speaker from val to test if test is empty
            spk_to_shift = val_spks.pop()
            test_spks.add(spk_to_shift)

        # 4. Re-assign rows based on speaker split assignment
        updated_rows: List[ManifestRow] = []

        for spk, spk_rows in speaker_map.items():
            if spk in train_spks:
                target_split = "train"
            elif spk in val_spks:
                target_split = "validation"
            else:
                target_split = "test"

            for r in spk_rows:
                updated_rows.append(ManifestRow(
                    file_path=r.file_path,
                    label=r.label,
                    speaker_id=r.speaker_id,
                    source=r.source,
                    source_id=r.source_id,
                    split=target_split,
                    line_number=r.line_number
                ))

        # Re-attach OOD rows unchanged
        updated_rows.extend(ood_rows)

        # 5. Verify zero speaker leakage in resulting dataset
        spk_report = self.speaker_validator.validate_speaker_splits(updated_rows)
        if not spk_report.is_speaker_disjoint:
            raise RuntimeError(f"Deterministic splitting produced speaker leakage errors: {spk_report.errors}")

        # 6. Optional CSV export
        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["file_path", "label", "speaker_id", "source", "source_id", "split"])
                for r in updated_rows:
                    writer.writerow([r.file_path, r.label, r.speaker_id, r.source, r.source_id, r.split])
            logger.info("Successfully saved prepared speaker-disjoint manifest to '%s'", output_path)

        return updated_rows
