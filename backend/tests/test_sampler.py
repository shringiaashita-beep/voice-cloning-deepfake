"""Unit tests for VoxGuard Training Sampler & Data Loader (app.training.sampler)."""

import csv
import os
import tempfile
import unittest
import numpy as np

from app.training.config import TrainingConfig
from app.training.dataset import FeatureDataset
from app.training.sampler import DeterministicSampler, TrainingDataLoader
from app.training.types import TrainingBatch


def create_dummy_npz(filepath: str, t_frames: int = 256, label_code: int = 0, speaker_id: str = "spk_1"):
    """Creates a dummy .npz feature artifact file on disk."""
    log_mel = np.ones((80, t_frames), dtype=np.float32)
    mfcc = np.ones((20, t_frames), dtype=np.float32)
    centroid = np.ones((1, t_frames), dtype=np.float32)
    rolloff = np.ones((1, t_frames), dtype=np.float32)
    np.savez(
        filepath,
        log_mel=log_mel,
        mfcc=mfcc,
        spectral_centroid=centroid,
        spectral_rolloff=rolloff,
        label_code=np.int64(label_code),
        speaker_id=np.array(speaker_id)
    )


class TestSampler(unittest.TestCase):
    """Test suite for DeterministicSampler and TrainingDataLoader."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.feature_dir = os.path.join(self.temp_dir.name, "features")
        os.makedirs(os.path.join(self.feature_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.feature_dir, "val"), exist_ok=True)

        # Create 5 train samples
        self.manifest_path = os.path.join(self.temp_dir.name, "manifest.csv")
        rows = []
        for i in range(5):
            fname = f"train/sample_{i}.npz"
            path = os.path.join(self.feature_dir, fname)
            create_dummy_npz(path, speaker_id=f"spk_{i}")
            rows.append({"feature_path": fname, "label": "human", "speaker_id": f"spk_{i}", "source": "src", "source_id": "v", "split": "train"})

        # Create 3 val samples
        for i in range(3):
            fname = f"val/sample_{i}.npz"
            path = os.path.join(self.feature_dir, fname)
            create_dummy_npz(path, speaker_id=f"val_spk_{i}")
            rows.append({"feature_path": fname, "label": "synthetic", "speaker_id": f"val_spk_{i}", "source": "src", "source_id": "v", "split": "validation"})

        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerows(rows)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_train_shuffle_behavior(self):
        """Verify training split with shuffle=True produces shuffled index sequence."""
        sampler = DeterministicSampler(dataset_size=10, shuffle=True, seed=42)
        indices1 = list(sampler)
        self.assertNotEqual(indices1, list(range(10)))

    def test_validation_deterministic_ordering(self):
        """Verify validation split with shuffle=False produces sequential index ordering."""
        dataset_val = FeatureDataset(manifest_path=self.manifest_path, feature_dir=self.feature_dir, split="validation")
        loader = TrainingDataLoader(dataset=dataset_val, batch_size=2)

        self.assertFalse(loader.shuffle)
        batches = list(loader)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].metadata[0]["speaker_id"], "val_spk_0")
        self.assertEqual(batches[0].metadata[1]["speaker_id"], "val_spk_1")

    def test_deterministic_batch_ordering_across_runs(self):
        """Verify two loader instances with same seed produce identical batch order."""
        dataset_train = FeatureDataset(manifest_path=self.manifest_path, feature_dir=self.feature_dir, split="train")

        config = TrainingConfig(BATCH_SIZE=2, RANDOM_SEED=42)
        loader1 = TrainingDataLoader(dataset=dataset_train, config=config)
        loader2 = TrainingDataLoader(dataset=dataset_train, config=config)

        batches1 = list(loader1)
        batches2 = list(loader2)

        self.assertEqual(len(batches1), len(batches2))
        for b1, b2 in zip(batches1, batches2):
            np.testing.assert_array_equal(b1.log_mel, b2.log_mel)
            np.testing.assert_array_equal(b1.labels, b2.labels)
            self.assertEqual(b1.metadata, b2.metadata)


if __name__ == "__main__":
    unittest.main()
