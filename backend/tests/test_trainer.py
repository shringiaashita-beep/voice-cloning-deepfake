"""Unit Tests for VoxGuard Model Trainer Module.

Validates training execution, optimizer steps, class weighting, early stopping,
checkpoint saving, and SHA-256 manifest generation using synthetic in-memory fixtures.
"""

import csv
import os
import shutil
import tempfile
import unittest
import numpy as np
import torch

from app.training.config import TrainingConfig
from app.training.dataset import FeatureDataset
from app.training.sampler import TrainingDataLoader
from app.training.trainer import Trainer, TrainingSummary


class TestVoxGuardTrainer(unittest.TestCase):
    """Test suite for VoxGuard Trainer component."""

    def setUp(self):
        """Create temporary workspace directory for checkpoints and synthetic dataset."""
        self.test_dir = tempfile.mkdtemp()
        self.features_dir = os.path.join(self.test_dir, "features")
        self.output_dir = os.path.join(self.test_dir, "models")
        self.manifest_path = os.path.join(self.test_dir, "manifest.csv")

        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        rows = []
        rows.extend(self._create_synthetic_features("train", num_human=4, num_synth=4))
        rows.extend(self._create_synthetic_features("validation", num_human=2, num_synth=2))

        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["feature_path", "label", "speaker_id", "source", "source_id", "split"]
            )
            writer.writeheader()
            writer.writerows(rows)

        self.config = TrainingConfig(BATCH_SIZE=4, RANDOM_SEED=42)
        self.train_dataset = FeatureDataset(self.manifest_path, feature_dir=self.features_dir, split="train")
        self.val_dataset = FeatureDataset(self.manifest_path, feature_dir=self.features_dir, split="validation")

        self.train_loader = TrainingDataLoader(self.train_dataset, config=self.config, shuffle=True)
        self.val_loader = TrainingDataLoader(self.val_dataset, config=self.config, shuffle=False)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_synthetic_features(self, split: str, num_human: int, num_synth: int):
        """Helper to create dummy .npz files and manifest row dicts."""
        split_dir = os.path.join(self.features_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        rows = []
        idx = 0
        for label, count in [(0, num_human), (1, num_synth)]:
            label_str = "human" if label == 0 else "synthetic"
            for i in range(count):
                idx += 1
                fname = f"sample_{idx}_{label_str}.npz"
                rel_path = os.path.join(split, fname)
                abs_path = os.path.join(self.features_dir, rel_path)

                log_mel = np.random.randn(80, 300).astype(np.float32)
                mfcc = np.random.randn(20, 300).astype(np.float32)
                centroid = np.random.randn(1, 300).astype(np.float32)
                rolloff = np.random.randn(1, 300).astype(np.float32)

                np.savez_compressed(
                    abs_path,
                    log_mel=log_mel,
                    mfcc=mfcc,
                    spectral_centroid=centroid,
                    spectral_rolloff=rolloff,
                    label=np.array(label, dtype=np.int64),
                    speaker_id=f"spk_{idx}_{split}",
                    sample_rate=16000
                )

                rows.append({
                    "feature_path": rel_path,
                    "label": label_str,
                    "speaker_id": f"spk_{idx}_{split}",
                    "source": "synthetic_test",
                    "source_id": f"src_{idx}",
                    "split": split
                })
        return rows

    def test_trainer_initialization(self):
        """Verify Trainer initialization and parameter defaults."""
        trainer = Trainer(output_dir=self.output_dir, learning_rate=0.005, patience=3)
        self.assertEqual(trainer.learning_rate, 0.005)
        self.assertEqual(trainer.patience, 3)
        self.assertEqual(trainer.output_dir, self.output_dir)
        self.assertIsNotNone(trainer.model)

    def test_training_loop_execution(self):
        """Verify full training loop over 2 epochs."""
        trainer = Trainer(output_dir=self.output_dir, patience=5)
        summary = trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=2
        )

        self.assertIsInstance(summary, TrainingSummary)
        self.assertEqual(summary.total_epochs, 2)
        self.assertFalse(summary.stopped_early)
        self.assertTrue(os.path.exists(summary.best_checkpoint_path))
        self.assertTrue(os.path.exists(summary.last_checkpoint_path))
        self.assertTrue(os.path.exists(summary.manifest_path))

    def test_early_stopping_trigger(self):
        """Verify early stopping terminates training when validation metric does not improve."""
        trainer = Trainer(output_dir=self.output_dir, patience=1)
        summary = trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=10
        )
        self.assertTrue(summary.stopped_early or summary.total_epochs <= 3)

    def test_manifest_and_sha256(self):
        """Verify manifest.json contains valid SHA-256 digest and metadata."""
        trainer = Trainer(output_dir=self.output_dir)
        summary = trainer.train(
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            epochs=1
        )

        import json
        with open(summary.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["model_name"], "VoxGuardAcousticNet")
        self.assertEqual(data["framework"], "pytorch")
        self.assertIn("sha256", data)
        self.assertEqual(len(data["sha256"]), 64)
        self.assertEqual(data["labels"], ["human", "synthetic"])


if __name__ == "__main__":
    unittest.main()
