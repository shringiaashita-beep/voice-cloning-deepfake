"""Unit tests for VoxGuard SpeakerSplitValidator and Speaker Distribution.

Tests speaker leakage detection across train/val/test/ood splits,
zero-speaker-overlap verification, and source distribution tracking.
"""

import unittest
from app.evaluation.manifest_validator import ManifestRow
from app.evaluation.speaker_split_validator import SpeakerSplitValidator
from app.evaluation.class_balance import ClassBalanceAnalyzer


class TestSpeakerSplit(unittest.TestCase):
    """Test suite for speaker-disjoint validation and source distributions."""

    def setUp(self):
        self.validator = SpeakerSplitValidator()
        self.analyzer = ClassBalanceAnalyzer()

    def test_speaker_disjoint_pass(self):
        """Verify valid speaker-disjoint dataset passes without leakage."""
        rows = [
            ManifestRow("1.wav", "human", "spk_1", "src_A", "src_id_1", "train"),
            ManifestRow("2.wav", "human", "spk_2", "src_A", "src_id_2", "validation"),
            ManifestRow("3.wav", "synthetic", "spk_3", "src_B", "src_id_3", "test"),
            ManifestRow("4.wav", "synthetic", "spk_4", "src_C", "src_id_4", "ood")
        ]
        report = self.validator.validate_speaker_splits(rows)
        self.assertTrue(report.is_speaker_disjoint)
        self.assertEqual(report.total_speakers, 4)
        self.assertEqual(len(report.errors), 0)

    def test_speaker_leakage_detection(self):
        """9. Verify speaker leakage between train and test splits is caught."""
        rows = [
            ManifestRow("1.wav", "human", "spk_LEAK", "src_A", "src_id_1", "train"),
            ManifestRow("2.wav", "human", "spk_LEAK", "src_A", "src_id_1", "test")
        ]
        report = self.validator.validate_speaker_splits(rows)
        self.assertFalse(report.is_speaker_disjoint)
        self.assertIn("train_and_test", report.overlapping_speakers)
        self.assertIn("spk_LEAK", report.overlapping_speakers["train_and_test"])

    def test_source_distribution_and_ood_metadata(self):
        """13 & 14. Verify class balance and source distribution tracking."""
        rows = [
            ManifestRow("1.wav", "human", "spk_1", "librispeech", "reader_1", "train"),
            ManifestRow("2.wav", "synthetic", "spk_2", "elevenlabs", "neural_v2", "test"),
            ManifestRow("3.wav", "synthetic", "spk_3", "bark", "bark_v1", "ood")
        ]
        report = self.analyzer.analyze(rows)
        self.assertEqual(report.overall.total, 3)
        self.assertEqual(report.overall.human, 1)
        self.assertEqual(report.overall.synthetic, 2)
        self.assertEqual(report.source_distribution["librispeech"], 1)
        self.assertEqual(report.source_distribution["elevenlabs"], 1)
        self.assertEqual(report.source_distribution["bark"], 1)


if __name__ == "__main__":
    unittest.main()
