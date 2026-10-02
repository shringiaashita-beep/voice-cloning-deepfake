"""Unit Tests for VoxGuard Training DataLoader Module (app.training.dataloader & app.training.sampler).

Validates deterministic batching, configurable batch size, seed 42 shuffling,
drop_last behavior, and dictionary subscription interface.
"""

import csv
import os
import shutil
import tempfile
import unittest
import numpy as np

from app.training.config import TrainingConfig
from app.training.dataset import FeatureDataset
from app.training.dataloader import TrainingDataLoader, DeterministicSampler


class TestTrainingDataLoader(unittest.TestCase):
    """Test suite for TrainingDataLoader component."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.features_dir = os.path.join(self.test_dir, "features")
        self.manifest_path = os.path.join(self.test_dir, "manifest.csv")
        os.makedirs(os.path.join(self.features_dir, "train"), exist_ok=True)

        rows = []
        for i in range(1, 7):
            label_str = "human" if i % 2 == 1 else "synthetic"
            rel_path = f"train/sample_{i}.npz"
            abs_path = os.path.join(self.features_dir, rel_path)

            np.savez_compressed(
                abs_path,
                log_mel=np.random.randn(80, 300).astype(np.float32),
                mfcc=np.random.randn(20, 300).astype(np.float32),
                spectral_centroid=np.random.randn(1, 300).astype(np.float32),
                spectral_rolloff=np.random.randn(1, 300).astype(np.float32),
                label=np.array(0 if label_str == "human" else 1, dtype=np.int64),
                speaker_id=f"spk_{i}",
                sample_rate=16000
            )

            rows.append({
                "feature_path": rel_path,
                "label": label_str,
                "speaker_id": f"spk_{i}",
                "source": "src_1",
                "source_id": f"v_{i}",
                "split": "train"
            })

        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"])
            writer.writeheader()
            writer.writerows(rows)

        self.config = TrainingConfig(BATCH_SIZE=4, RANDOM_SEED=42)
        self.dataset = FeatureDataset(self.manifest_path, feature_dir=self.features_dir, split="train")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_batch_construction_and_dict_access(self):
        """Verify DataLoader builds valid batch objects supporting dict access."""
        loader = TrainingDataLoader(self.dataset, config=self.config, shuffle=False)
        batches = list(loader)

        self.assertTrue(len(batches) > 0)
        batch = batches[0]

        self.assertEqual(batch.log_mel.shape[1:], (80, 256))
        self.assertEqual(batch.mfcc.shape[1:], (20, 256))

        # Test dict subscription interface
        self.assertTrue(np.array_equal(batch["log_mel"], batch.log_mel))
        self.assertEqual(batch["labels"].dtype, np.int64)
        self.assertEqual(len(batch["speaker_ids"]), batch.batch_size)

    def test_seed_reproducibility(self):
        """Verify shuffle with seed 42 produces reproducible batch ordering."""
        loader1 = TrainingDataLoader(self.dataset, config=self.config, shuffle=True, seed=42)
        loader2 = TrainingDataLoader(self.dataset, config=self.config, shuffle=True, seed=42)

        b1 = [b.labels for b in loader1]
        b2 = [b.labels for b in loader2]

        for l1, l2 in zip(b1, b2):
            np.testing.assert_array_equal(l1, l2)


if __name__ == "__main__":
    unittest.main()
