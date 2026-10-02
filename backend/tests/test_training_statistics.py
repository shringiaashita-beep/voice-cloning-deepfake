"""Unit Tests for VoxGuard Dataset Statistics & Risk Assessment Module (app.training.statistics).

Validates window counts, speaker counts, class balance percentages, sequence length calculations,
class imbalance risk detection, dominant speaker risk warnings, dominant source warnings,
suspicious split size warnings, and JSON dictionary serialization.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from app.training.dataset import FeatureDataset
from app.training.statistics import DatasetStatisticsCalculator, DatasetStatisticsReport, DatasetRiskAssessment
from app.training.types import TrainingSample


class TestTrainingStatistics(unittest.TestCase):
    """Test suite for DatasetStatisticsCalculator component."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.manifest_path = os.path.join(self.test_dir, "manifest.csv")
        self.features_dir = os.path.join(self.test_dir, "features")
        os.makedirs(self.features_dir, exist_ok=True)
        self.calculator = DatasetStatisticsCalculator()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_dataset(self, samples_data: list) -> FeatureDataset:
        """Helper to build a FeatureDataset from sample metadata dicts."""
        manifest_lines = ["sample_id,file_path,speaker_id,source,source_id,label,split"]
        for i, s in enumerate(samples_data):
            sid = s.get("sample_id", f"sample_{i:03d}")
            spk = s.get("speaker_id", f"spk_{i % 3}")
            src = s.get("source", "librispeech")
            src_id = s.get("source_id", "src_001")
            lbl = s.get("label", "human")
            split = s.get("split", "train")
            npz_rel = f"{sid}.npz"
            manifest_lines.append(f"{sid},{npz_rel},{spk},{src},{src_id},{lbl},{split}")

            # Save npz feature file
            npz_path = os.path.join(self.features_dir, npz_rel)
            num_frames = s.get("num_frames", 512)
            np.savez_compressed(
                npz_path,
                log_mel=np.ones((80, num_frames), dtype=np.float32),
                mfcc=np.ones((20, num_frames), dtype=np.float32),
                spectral_centroid=np.ones((1, num_frames), dtype=np.float32),
                spectral_rolloff=np.ones((1, num_frames), dtype=np.float32)
            )

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write("\n".join(manifest_lines))

        return FeatureDataset(self.manifest_path, feature_dir=self.features_dir, split="train")

    def test_empty_dataset_statistics(self):
        """Verify calculator handles empty dataset gracefully."""
        # Create dataset manifest with no matching samples for split="val"
        dataset = self._create_mock_dataset([{"split": "train"}])
        val_dataset = FeatureDataset(self.manifest_path, feature_dir=self.features_dir, split="val")

        report = self.calculator.calculate(val_dataset)
        self.assertEqual(report.split, "val")
        self.assertEqual(report.total_windows, 0)
        self.assertEqual(report.total_samples, 0)
        self.assertEqual(report.total_speakers, 0)
        self.assertTrue(report.risk_assessment.has_suspicious_split)
        self.assertEqual(len(report.risk_assessment.warnings), 1)

    def test_balanced_dataset_statistics(self):
        """Verify accurate window count, speaker count, class percentage, and sequence stats."""
        # 10 samples: 5 human, 5 synthetic across 3 speakers, 512 frames each
        samples_data = []
        for i in range(10):
            samples_data.append({
                "sample_id": f"sample_{i:02d}",
                "speaker_id": f"spk_{i % 3}",
                "source": "librispeech" if i % 2 == 0 else "elevenlabs",
                "label": "human" if i < 5 else "synthetic",
                "num_frames": 512
            })
        dataset = self._create_mock_dataset(samples_data)
        report = self.calculator.calculate(dataset)

        self.assertEqual(report.split, "train")
        self.assertEqual(report.total_samples, 10)
        # 512 frames windowed (win=256, stride=128) yields 3 windows per sample -> total 30 windows
        self.assertEqual(report.total_windows, 30)
        self.assertEqual(report.total_speakers, 3)
        self.assertEqual(report.human_windows, 15)
        self.assertEqual(report.synthetic_windows, 15)
        self.assertEqual(report.human_percentage, 50.0)
        self.assertEqual(report.synthetic_percentage, 50.0)
        self.assertFalse(report.risk_assessment.has_class_imbalance)
        self.assertFalse(report.risk_assessment.has_suspicious_split)

    def test_class_imbalance_risk(self):
        """Verify warning is flagged when one class dominates or has 0 windows."""
        # All human samples
        samples_data = [{"label": "human", "speaker_id": f"spk_{i}"} for i in range(6)]
        dataset = self._create_mock_dataset(samples_data)
        report = self.calculator.calculate(dataset)

        self.assertTrue(report.risk_assessment.has_class_imbalance)
        self.assertIsNotNone(report.risk_assessment.imbalance_warning)
        self.assertIn("zero samples", report.risk_assessment.imbalance_warning.lower())

    def test_dominant_speaker_risk(self):
        """Verify warning is flagged when a single speaker owns >30% of total windows."""
        # 10 samples total: speaker_0 has 6 samples (60%), speaker_1 has 2, speaker_2 has 2
        samples_data = []
        for i in range(10):
            spk = "spk_dominant" if i < 6 else f"spk_{i}"
            samples_data.append({
                "sample_id": f"sample_{i:02d}",
                "speaker_id": spk,
                "label": "human" if i % 2 == 0 else "synthetic",
                "num_frames": 256
            })
        dataset = self._create_mock_dataset(samples_data)
        report = self.calculator.calculate(dataset)

        self.assertTrue(report.risk_assessment.has_dominant_speaker)
        self.assertIsNotNone(report.risk_assessment.dominant_speaker_warning)
        self.assertIn("spk_dominant", report.risk_assessment.dominant_speaker_warning)

    def test_dominant_source_risk(self):
        """Verify warning is flagged when a single source owns >50% of total windows."""
        samples_data = []
        for i in range(10):
            src = "dominant_vendor" if i < 7 else "secondary_vendor"
            samples_data.append({
                "sample_id": f"sample_{i:02d}",
                "speaker_id": f"spk_{i % 4}",
                "source": src,
                "label": "human" if i % 2 == 0 else "synthetic",
                "num_frames": 256
            })
        dataset = self._create_mock_dataset(samples_data)
        report = self.calculator.calculate(dataset)

        self.assertTrue(report.risk_assessment.has_dominant_source)
        self.assertIsNotNone(report.risk_assessment.dominant_source_warning)
        self.assertIn("dominant_vendor", report.risk_assessment.dominant_source_warning)

    def test_to_dict_serialization(self):
        """Verify report converts cleanly to JSON-serializable dictionary."""
        samples_data = [
            {"sample_id": "s1", "speaker_id": "spk1", "label": "human", "num_frames": 256},
            {"sample_id": "s2", "speaker_id": "spk2", "label": "synthetic", "num_frames": 256},
            {"sample_id": "s3", "speaker_id": "spk3", "label": "human", "num_frames": 256},
            {"sample_id": "s4", "speaker_id": "spk4", "label": "synthetic", "num_frames": 256},
            {"sample_id": "s5", "speaker_id": "spk5", "label": "human", "num_frames": 256},
            {"sample_id": "s6", "speaker_id": "spk6", "label": "synthetic", "num_frames": 256},
        ]
        dataset = self._create_mock_dataset(samples_data)
        report = self.calculator.calculate(dataset)
        report_dict = report.to_dict()

        self.assertIsInstance(report_dict, dict)
        self.assertEqual(report_dict["split"], "train")
        self.assertEqual(report_dict["total_samples"], 6)
        self.assertIn("risk_assessment", report_dict)
        self.assertIsInstance(report_dict["risk_assessment"], dict)


if __name__ == "__main__":
    unittest.main()
