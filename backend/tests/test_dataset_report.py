"""Unit tests for VoxGuard DeterministicSpeakerSplitter and Dataset Quality Reporting.

Tests deterministic speaker-level dataset partitioning, seed reproducibility,
ratio allocation, speaker leakage prevention, and insufficient speaker handling.
"""

import os
import shutil
import tempfile
import unittest

from app.evaluation.splitter import DeterministicSpeakerSplitter
from app.evaluation.speaker_split_validator import SpeakerSplitValidator


class TestDatasetReportAndSplitter(unittest.TestCase):
    """Test suite for speaker-level splitting and seed reproducibility."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="voxguard_test_splitter_")
        self.splitter = DeterministicSpeakerSplitter()
        self.spk_validator = SpeakerSplitValidator()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_multi_speaker_manifest(self, n_speakers: int = 10) -> str:
        path = os.path.join(self.temp_dir, "multi_spk_manifest.csv")
        lines = ["file_path,label,speaker_id,source,source_id,split"]
        for i in range(1, n_speakers + 1):
            spk_id = f"spk_{i:03d}"
            label = "human" if i % 2 == 0 else "synthetic"
            # 2 files per speaker
            lines.append(f"raw/{label}/{spk_id}_a.wav,{label},{spk_id},src_1,src_id_1,train")
            lines.append(f"raw/{label}/{spk_id}_b.wav,{label},{spk_id},src_1,src_id_1,train")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return path

    def test_10_deterministic_speaker_splitting(self):
        """10. Verify speaker-level splitting produces 0% speaker leakage and is seed reproducible."""
        manifest_path = self._create_multi_speaker_manifest(n_speakers=10)
        output_path = os.path.join(self.temp_dir, "prepared_manifest.csv")

        # Run split with seed 42
        rows_run1 = self.splitter.split_manifest(
            manifest_path=manifest_path,
            output_path=output_path,
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.15,
            seed=42
        )

        # Verify zero speaker leakage in resulting split
        report = self.spk_validator.validate_speaker_splits(rows_run1)
        self.assertTrue(report.is_speaker_disjoint)
        self.assertEqual(len(report.errors), 0)

        # Run split again with identical seed 42
        rows_run2 = self.splitter.split_manifest(
            manifest_path=manifest_path,
            seed=42
        )

        # Verify exact reproducibility
        splits1 = [r.split for r in rows_run1]
        splits2 = [r.split for r in rows_run2]
        self.assertEqual(splits1, splits2)

    def test_insufficient_speakers_handling(self):
        """16. Verify exception raised when manifest has fewer than 3 unique speakers."""
        path = os.path.join(self.temp_dir, "few_spk_manifest.csv")
        csv_text = (
            "file_path,label,speaker_id,source,source_id,split\n"
            "raw/human/1.wav,human,spk_1,src_1,src_id_1,train\n"
            "raw/human/2.wav,human,spk_2,src_1,src_id_2,train\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(csv_text)

        with self.assertRaises(ValueError) as ctx:
            self.splitter.split_manifest(path, seed=42)
        
        self.assertIn("Insufficient unique speakers", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
